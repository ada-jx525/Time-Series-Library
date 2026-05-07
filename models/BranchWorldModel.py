import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.BranchWorld import LatentStateEncoder, MemoryBranchDecoder, kmeans_torch


class DLinearBase(nn.Module):
    def __init__(self, seq_len, pred_len, channels, moving_avg, individual=False):
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.decomposition = series_decomp(moving_avg)
        if individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
            for seasonal, trend in zip(self.linear_seasonal, self.linear_trend):
                nn.init.constant_(seasonal.weight, 1.0 / seq_len)
                nn.init.constant_(trend.weight, 1.0 / seq_len)
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)
            nn.init.constant_(self.linear_seasonal.weight, 1.0 / seq_len)
            nn.init.constant_(self.linear_trend.weight, 1.0 / seq_len)

    def forward(self, x):
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        if self.individual:
            seasonal_out = torch.zeros(
                seasonal.size(0), seasonal.size(1), self.pred_len,
                dtype=seasonal.dtype,
                device=seasonal.device,
            )
            trend_out = torch.zeros_like(seasonal_out)
            for channel in range(self.channels):
                seasonal_out[:, channel] = self.linear_seasonal[channel](seasonal[:, channel])
                trend_out[:, channel] = self.linear_trend[channel](trend[:, channel])
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)
        return (seasonal_out + trend_out).permute(0, 2, 1)


class LinearBase(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.proj = nn.Linear(seq_len, pred_len)
        nn.init.constant_(self.proj.weight, 1.0 / seq_len)

    def forward(self, x):
        return self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class Model(nn.Module):
    """
    World-Trajectory Memory Enhancement Module.

    This is a memory-enhanced forecaster, not an end-to-end latent world model.
    It keeps a base forecast, learns an offline bank of multi-horizon latent
    displacement prototypes from training windows, selects state-conditioned
    dynamics prototypes online, decodes memory branches, and learns a
    base-vs-memory fusion gate.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.latent_dim = getattr(configs, "wm_latent_dim", configs.d_model)
        self.branch_num = getattr(configs, "wm_branch_num", 3)
        self.retrieve_k = getattr(configs, "wm_retrieve_k", 64)
        self.memory_size = getattr(configs, "wm_memory_size", 4096)
        self.global_proto_num = getattr(configs, "wm_global_proto_num", 32)
        self.prototype_mode = getattr(configs, "wm_proto_mode", "offline")
        self.kmeans_iters = getattr(configs, "wm_kmeans_iters", 8)
        self.prototype_refine_iters = getattr(configs, "wm_proto_refine_iters", 2)
        self.proto_state_alpha = getattr(configs, "wm_proto_state_alpha", 1.0)
        self.proto_traj_beta = getattr(configs, "wm_proto_traj_beta", 2.0)
        self.use_memory = bool(getattr(configs, "wm_use_memory", 1))
        self.use_branch_discovery = bool(getattr(configs, "wm_use_branch_discovery", 1))
        self.mem_weight = getattr(configs, "wm_mem_weight", 0.1)
        self.traj_weight = getattr(configs, "wm_traj_weight", 0.0)
        self.base_weight = getattr(configs, "wm_base_weight", 0.0)
        self.base_type = getattr(configs, "wm_base_type", "dlinear")
        self.mem_loss_type = getattr(configs, "wm_mem_loss_type", "min")
        self.freeze_base = bool(getattr(configs, "wm_freeze_base", 1))

        horizons = getattr(configs, "wm_horizons", None)
        if horizons is None or len(horizons) == 0:
            horizons = [max(1, self.pred_len // 4), max(1, self.pred_len // 2), max(1, 3 * self.pred_len // 4), self.pred_len]
        self.horizons = sorted(set(int(h) for h in horizons if int(h) > 0 and int(h) <= self.pred_len))
        if not self.horizons:
            self.horizons = [self.pred_len]
        self.num_horizons = len(self.horizons)

        if self.base_type == "linear":
            self.base_forecaster = LinearBase(self.seq_len, self.pred_len, self.c_out)
        else:
            self.base_forecaster = DLinearBase(
                self.seq_len,
                self.pred_len,
                self.c_out,
                getattr(configs, "moving_avg", 25),
                bool(getattr(configs, "individual", False)),
            )
        if self.freeze_base:
            for param in self.base_forecaster.parameters():
                param.requires_grad = False

        self.state_encoder = LatentStateEncoder(
            c_in=configs.enc_in,
            seq_len=configs.seq_len,
            dropout=configs.dropout,
            latent_dim=self.latent_dim,
        )
        self.memory_decoder = MemoryBranchDecoder(
            latent_dim=self.latent_dim,
            num_horizons=self.num_horizons,
            pred_len=self.pred_len,
            c_out=self.c_out,
            hidden_dim=configs.d_ff,
            dropout=configs.dropout,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.num_horizons * self.latent_dim + 4),
            nn.Linear(self.latent_dim + self.num_horizons * self.latent_dim + 4, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1),
        )
        self.base_gate = nn.Sequential(
            nn.LayerNorm(self.latent_dim + 4),
            nn.Linear(self.latent_dim + 4, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1),
        )
        self.traj_head = nn.Sequential(
            nn.LayerNorm(self.pred_len * self.c_out),
            nn.Linear(self.pred_len * self.c_out, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, self.num_horizons * self.latent_dim),
        )
        self._last_aux_loss = None

        self.register_buffer("memory_keys", torch.empty(0, self.latent_dim), persistent=False)
        self.register_buffer("memory_states", torch.empty(0, self.latent_dim), persistent=False)
        self.register_buffer("memory_trajectories", torch.empty(0, self.num_horizons, self.latent_dim), persistent=False)
        # Future evolution patterns in each memory item's local normalized
        # coordinates. These are branch targets for D_theta(z_q, P_m); they
        # are denormalized with the query window statistics only at output time.
        self.register_buffer("memory_values", torch.empty(0, self.pred_len, self.c_out), persistent=False)
        self.register_buffer("memory_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("memory_proto_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("prototype_bank", torch.empty(0, self.num_horizons, self.latent_dim), persistent=False)
        self.register_buffer("prototype_confidence", torch.empty(0), persistent=False)
        self.register_buffer("prototype_support", torch.empty(0), persistent=False)
        self.register_buffer("memory_ready", torch.tensor(False), persistent=False)

    def _normalize(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return x / stdev, means, stdev

    def _denormalize(self, y, means, stdev):
        if means.size(-1) != y.size(-1):
            means = means[..., -y.size(-1):]
            stdev = stdev[..., -y.size(-1):]
        return y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    def _base_forecast(self, norm_x):
        return self.base_forecaster(norm_x[:, :, -self.c_out:])

    def _shifted_windows(self, norm_x, norm_future):
        series = torch.cat([norm_x, norm_future], dim=1)
        windows = []
        for horizon in self.horizons:
            windows.append(series[:, horizon : horizon + self.seq_len])
        return windows

    def _encode_trajectory(self, norm_x, norm_future):
        state, key = self.state_encoder(norm_x)
        future_states = []
        for future_window in self._shifted_windows(norm_x, norm_future):
            future_state, _ = self.state_encoder(future_window)
            future_states.append(future_state)
        trajectory = torch.stack(future_states, dim=1) - state.unsqueeze(1)
        return state, key, trajectory

    def _reset_memory(self):
        device = self.memory_ready.device
        self.memory_keys = torch.empty(0, self.latent_dim, device=device)
        self.memory_states = torch.empty(0, self.latent_dim, device=device)
        self.memory_trajectories = torch.empty(0, self.num_horizons, self.latent_dim, device=device)
        self.memory_values = torch.empty(0, self.pred_len, self.c_out, device=device)
        self.memory_indices = torch.empty(0, dtype=torch.long, device=device)
        self.memory_proto_ids = torch.empty(0, dtype=torch.long, device=device)
        self.prototype_bank = torch.empty(0, self.num_horizons, self.latent_dim, device=device)
        self.prototype_confidence = torch.empty(0, device=device)
        self.prototype_support = torch.empty(0, device=device)
        self.memory_ready.fill_(False)

    @torch.no_grad()
    def _build_prototype_bank(self, trajectories):
        flat_dim = self.num_horizons * self.latent_dim
        proto_num = min(max(1, self.global_proto_num), trajectories.size(0))
        flat = trajectories.reshape(trajectories.size(0), flat_dim)
        centers, assign = kmeans_torch(flat, proto_num, self.kmeans_iters)
        centers = centers.view(proto_num, self.num_horizons, self.latent_dim)

        confidence = trajectories.new_zeros(proto_num)
        support = trajectories.new_zeros(proto_num)
        for proto_id in range(proto_num):
            mask = assign == proto_id
            support[proto_id] = mask.float().mean()
            if mask.any():
                dist = (trajectories[mask] - centers[proto_id].unsqueeze(0)).flatten(start_dim=1)
                dist = dist.pow(2).sum(dim=-1).sqrt()
                confidence[proto_id] = torch.exp(-dist.mean() / (flat_dim ** 0.5))

        return centers, assign.long(), confidence.clamp_min(1e-6), support.clamp_min(1e-6)

    @torch.no_grad()
    def build_memory(self, data_loader, device):
        if not self.use_memory or self.memory_size <= 0:
            self._reset_memory()
            return

        was_training = self.training
        self.eval()
        states = []
        keys = []
        trajectories = []
        values = []
        source_indices = []
        offset = 0

        for batch in data_loader:
            if len(batch) == 5:
                batch_x, batch_y, _, _, batch_index = batch
            else:
                batch_x, batch_y, _, _ = batch
                batch_index = torch.arange(offset, offset + batch_x.size(0))
            offset += batch_x.size(0)
            batch_x = batch_x.float().to(device)
            future_y = batch_y[:, -self.pred_len:, :].float().to(device)
            if future_y.size(-1) != self.c_out:
                future_y = future_y[:, :, -self.c_out:]

            norm_x, means, stdev = self._normalize(batch_x)
            norm_future = (future_y - means[:, :, -future_y.size(-1):]) / stdev[:, :, -future_y.size(-1):]
            state, key, trajectory = self._encode_trajectory(norm_x, norm_future)
            states.append(state.detach().cpu())
            keys.append(key.detach().cpu())
            trajectories.append(trajectory.detach().cpu())
            values.append(norm_future.detach().cpu())
            source_indices.append(batch_index.detach().cpu().long())

        if not keys:
            self._reset_memory()
            if was_training:
                self.train()
            return

        states = torch.cat(states, dim=0).to(device)
        keys = torch.cat(keys, dim=0).to(device)
        trajectories = torch.cat(trajectories, dim=0).to(device)
        values = torch.cat(values, dim=0).to(device)
        source_indices = torch.cat(source_indices, dim=0).to(device)

        if keys.size(0) > self.memory_size:
            ids = torch.linspace(0, keys.size(0) - 1, self.memory_size, device=device).long()
            states = states[ids]
            keys = keys[ids]
            trajectories = trajectories[ids]
            values = values[ids]
            source_indices = source_indices[ids]

        prototype_bank, proto_ids, proto_confidence, proto_support = self._build_prototype_bank(trajectories)

        self.memory_states = states.detach()
        self.memory_keys = F.normalize(keys.detach(), dim=-1)
        self.memory_trajectories = trajectories.detach()
        self.memory_values = values.detach()
        self.memory_indices = source_indices.detach()
        self.memory_proto_ids = proto_ids.detach()
        self.prototype_bank = prototype_bank.detach()
        self.prototype_confidence = proto_confidence.detach()
        self.prototype_support = proto_support.detach()
        self.memory_ready.fill_(True)
        if was_training:
            self.train()

    def _fallback_prototypes(self, state):
        B = state.size(0)
        prototypes = state.new_zeros(B, self.branch_num, self.num_horizons, self.latent_dim)
        reliability = state.new_zeros(B, self.branch_num, 3)
        return prototypes, reliability

    def _discover_prototypes(self, query_key, query_index=None):
        B = query_key.size(0)
        if not (
            self.use_memory
            and bool(self.memory_ready)
            and self.memory_keys.numel() > 0
            and self.prototype_bank.numel() > 0
            and self.memory_proto_ids.numel() == self.memory_keys.size(0)
        ):
            prototypes, reliability = self._fallback_prototypes(query_key)
            return prototypes, reliability, None

        retrieve_k = min(self.retrieve_k, self.memory_keys.size(0))
        # Retrieval and prototype selection are memory operations, not
        # differentiable sequence modeling layers. Stop gradients here to avoid
        # unstable TopK/discrete-selection gradients leaking into the encoder.
        query_key = query_key.detach()
        sim_all = query_key @ self.memory_keys.t()
        if self.training and query_index is not None and self.memory_indices.numel() == self.memory_keys.size(0):
            exclusion_radius = self.seq_len + self.pred_len
            query_index = query_index.to(sim_all.device).long()
            overlap = (query_index.unsqueeze(1) - self.memory_indices.unsqueeze(0)).abs() < exclusion_radius
            sim_all = sim_all.masked_fill(overlap, -torch.inf)
            empty = torch.isneginf(sim_all).all(dim=-1)
            if empty.any():
                fallback_sim = query_key[empty] @ self.memory_keys.t()
                sim_all[empty] = fallback_sim
        score, idx = torch.topk(sim_all, k=retrieve_k, dim=-1)
        neighbor_traj = self.memory_trajectories[idx]
        neighbor_values = self.memory_values[idx]
        neighbor_proto_ids = self.memory_proto_ids[idx]

        if not self.use_branch_discovery:
            order = torch.linspace(0, retrieve_k - 1, steps=self.branch_num, device=query_key.device).long()
            prototypes = neighbor_traj[:, order]
            branch_targets = neighbor_values[:, order]
            state_rel = ((score[:, order] + 1.0) * 0.5).clamp_min(1e-6)
            traj_rel = prototypes.new_ones(B, self.branch_num)
            size_rel = prototypes.new_full((B, self.branch_num), 1.0 / max(1, retrieve_k))
            return prototypes, torch.stack([state_rel, traj_rel, size_rel], dim=-1), branch_targets

        if self.prototype_mode == "local":
            prototypes = []
            reliabilities = []
            branch_targets = []
            flat_dim = self.num_horizons * self.latent_dim
            for batch_id in range(B):
                local = neighbor_traj[batch_id]
                local_values = neighbor_values[batch_id]
                local_flat = local.reshape(retrieve_k, flat_dim)
                _, assign = kmeans_torch(local_flat, self.branch_num, self.kmeans_iters)
                proto_m = []
                rel_m = []
                target_m = []
                for branch_id in range(self.branch_num):
                    mask = assign == branch_id
                    if not mask.any():
                        best_id = min(branch_id, retrieve_k - 1)
                        proto = local[best_id]
                        target = local_values[best_id]
                        state_rel = ((score[batch_id, best_id] + 1.0) * 0.5).clamp_min(1e-6)
                        traj_rel = proto.new_tensor(0.0)
                        size_rel = proto.new_tensor(0.0)
                    else:
                        member = local[mask]
                        member_values = local_values[mask]
                        member_score = score[batch_id, mask]
                        proto = member.mean(dim=0)
                        state_rel_vec = ((member_score + 1.0) * 0.5).clamp_min(1e-6)
                        weight = torch.full_like(state_rel_vec, 1.0 / max(1, state_rel_vec.numel()))
                        for _ in range(max(1, self.prototype_refine_iters)):
                            dist = (member - proto.unsqueeze(0)).flatten(start_dim=1).pow(2).sum(dim=-1).sqrt()
                            traj_rel_vec = torch.exp(-dist)
                            weight = state_rel_vec.pow(self.proto_state_alpha) * traj_rel_vec.pow(self.proto_traj_beta)
                            weight = weight / weight.sum().clamp_min(1e-6)
                            proto = (member * weight.view(-1, 1, 1)).sum(dim=0)
                        target = (member_values * weight.view(-1, 1, 1)).sum(dim=0)
                        dist = (member - proto.unsqueeze(0)).flatten(start_dim=1).pow(2).sum(dim=-1).sqrt()
                        traj_rel_vec = torch.exp(-dist)
                        state_rel = state_rel_vec.mean()
                        traj_rel = traj_rel_vec.mean()
                        size_rel = proto.new_tensor(float(mask.sum().item()) / float(retrieve_k))
                    proto_m.append(proto)
                    rel_m.append(torch.stack([state_rel, traj_rel, size_rel]))
                    target_m.append(target)
                prototypes.append(torch.stack(proto_m, dim=0))
                reliabilities.append(torch.stack(rel_m, dim=0))
                branch_targets.append(torch.stack(target_m, dim=0))

            branch_targets = torch.stack(branch_targets, dim=0)
            return torch.stack(prototypes, dim=0), torch.stack(reliabilities, dim=0), branch_targets

        proto_count = self.prototype_bank.size(0)
        branch_count = min(self.branch_num, proto_count)
        state_score = ((score + 1.0) * 0.5).clamp_min(1e-6)
        proto_scores = query_key.new_zeros(B, proto_count)
        proto_scores.scatter_add_(1, neighbor_proto_ids, state_score)
        proto_scores = proto_scores * self.prototype_confidence.unsqueeze(0)
        selected_score, selected_ids = torch.topk(proto_scores, k=branch_count, dim=-1)

        prototypes = []
        reliabilities = []
        branch_targets = []
        flat_dim = self.num_horizons * self.latent_dim
        for batch_id in range(B):
            local = neighbor_traj[batch_id]
            local_values = neighbor_values[batch_id]
            proto_m = []
            rel_m = []
            target_m = []
            for branch_id in range(branch_count):
                proto_id = selected_ids[batch_id, branch_id]
                proto = self.prototype_bank[proto_id]
                mask = neighbor_proto_ids[batch_id] == proto_id
                if not mask.any():
                    local_flat = local.reshape(retrieve_k, flat_dim)
                    proto_flat = proto.reshape(1, flat_dim)
                    dist = (local_flat - proto_flat).pow(2).sum(dim=-1).sqrt()
                    traj_weight = torch.exp(-dist / (flat_dim ** 0.5))
                    weight = state_score[batch_id] * traj_weight
                    weight = weight / weight.sum().clamp_min(1e-6)
                    target = (local_values * weight.view(-1, 1, 1)).sum(dim=0)
                    state_rel = (selected_score[batch_id, branch_id] / float(retrieve_k)).clamp_min(1e-6)
                    traj_rel = self.prototype_confidence[proto_id]
                    size_rel = proto.new_tensor(0.0)
                else:
                    member = local[mask]
                    member_values = local_values[mask]
                    member_score = state_score[batch_id, mask]
                    dist = (member - proto.unsqueeze(0)).flatten(start_dim=1).pow(2).sum(dim=-1).sqrt()
                    traj_rel_vec = torch.exp(-dist / (flat_dim ** 0.5))
                    weight = member_score.pow(self.proto_state_alpha) * traj_rel_vec.pow(self.proto_traj_beta)
                    weight = weight / weight.sum().clamp_min(1e-6)
                    target = (member_values * weight.view(-1, 1, 1)).sum(dim=0)
                    state_rel = (selected_score[batch_id, branch_id] / float(retrieve_k)).clamp_min(1e-6)
                    traj_rel = traj_rel_vec.mean()
                    size_rel = proto.new_tensor(float(mask.sum().item()) / float(retrieve_k))
                proto_m.append(proto)
                rel_m.append(torch.stack([state_rel, traj_rel, size_rel]))
                target_m.append(target)

            if branch_count < self.branch_num:
                pad = self.branch_num - branch_count
                proto_m.extend([proto_m[-1].clone() for _ in range(pad)])
                rel_m.extend([rel_m[-1].clone() * 0.0 for _ in range(pad)])
                target_m.extend([target_m[-1].clone() for _ in range(pad)])

            prototypes.append(torch.stack(proto_m, dim=0))
            reliabilities.append(torch.stack(rel_m, dim=0))
            branch_targets.append(torch.stack(target_m, dim=0))

        branch_targets = torch.stack(branch_targets, dim=0)
        return torch.stack(prototypes, dim=0), torch.stack(reliabilities, dim=0), branch_targets

    def _fusion(self, state, prototypes, reliability, y_base, y_mem):
        B, M = y_mem.shape[:2]
        disagreement = (y_mem - y_base.unsqueeze(1)).abs().mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        branch_features = torch.cat([reliability, disagreement], dim=-1)
        gate_in = torch.cat([
            state.unsqueeze(1).expand(B, M, self.latent_dim),
            prototypes.flatten(start_dim=2),
            branch_features,
        ], dim=-1)
        mem_logits = self.gate(gate_in).squeeze(-1)

        global_reliability = reliability.mean(dim=1)
        global_disagreement = disagreement.mean(dim=1)
        base_features = torch.cat([global_reliability, global_disagreement], dim=-1)
        base_logit = self.base_gate(torch.cat([state, base_features], dim=-1)).squeeze(-1)
        weights = torch.softmax(torch.cat([base_logit.unsqueeze(-1), mem_logits], dim=-1), dim=-1)
        y_hat = weights[:, :1].unsqueeze(-1) * y_base
        y_hat = y_hat + (weights[:, 1:].unsqueeze(-1).unsqueeze(-1) * y_mem).sum(dim=1)
        return y_hat, weights

    def _forecast_normalized(self, x_enc, future_y=None, query_index=None):
        norm_x, means, stdev = self._normalize(x_enc)
        y_base = self._base_forecast(norm_x)
        state, key = self.state_encoder(norm_x)
        prototypes, reliability, branch_targets = self._discover_prototypes(key, query_index=query_index)
        y_mem = self.memory_decoder(state, prototypes)
        pred, weights = self._fusion(state, prototypes, reliability, y_base, y_mem)

        self._last_aux_loss = None
        if self.training and future_y is not None:
            if future_y.size(-1) != self.c_out:
                future_y = future_y[:, :, -self.c_out:]
            norm_future = (future_y - means[:, :, -future_y.size(-1):]) / stdev[:, :, -future_y.size(-1):]
            mem_target = branch_targets.detach() if branch_targets is not None else norm_future.unsqueeze(1)
            branch_error = (y_mem - mem_target).pow(2).mean(dim=(2, 3))
            if self.mem_loss_type == "all":
                mem_loss = branch_error.mean()
            elif self.mem_loss_type == "weighted":
                mem_loss = (weights[:, 1:].detach() * branch_error).sum(dim=1).mean()
            else:
                mem_loss = branch_error.min(dim=1).values.mean()
            aux_loss = self.mem_weight * mem_loss
            if self.base_weight > 0:
                aux_loss = aux_loss + self.base_weight * F.mse_loss(y_base, norm_future)
            if self.traj_weight > 0:
                pred_traj = self.traj_head(y_mem.flatten(start_dim=2)).view_as(prototypes)
                aux_loss = aux_loss + self.traj_weight * F.mse_loss(pred_traj, prototypes.detach())
            self._last_aux_loss = aux_loss

        return pred, means, stdev

    def get_auxiliary_loss(self):
        return self._last_aux_loss

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=None, query_index=None):
        pred, means, stdev = self._forecast_normalized(x_enc, future_y=future_y, query_index=query_index)
        return self._denormalize(pred, means, stdev)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_y=None, query_index=None):
        if self.task_name != "long_term_forecast":
            raise NotImplementedError("BranchWorldModel currently supports long_term_forecast.")
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=future_y, query_index=query_index)
