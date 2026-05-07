import atexit
import hashlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.BranchWorld import LatentStateEncoder, MemoryBranchDecoder, SequenceMemoryBranchDecoder, kmeans_torch


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


class ChronosBoltBase(nn.Module):
    def __init__(
        self,
        model_id,
        pred_len,
        quantile_index=None,
        local_files_only=False,
        cache_path="",
        cache_save_interval=512,
    ):
        super().__init__()
        from chronos import ChronosBoltPipeline

        self.pred_len = pred_len
        self.quantile_index = quantile_index
        self.cache_path = cache_path
        self.cache_save_interval = max(0, int(cache_save_interval))
        self.cache = {}
        self.cache_dirty = 0
        if self.cache_path and os.path.exists(self.cache_path):
            self.cache = torch.load(self.cache_path, map_location="cpu")
            if not isinstance(self.cache, dict):
                self.cache = {}
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            model_id,
            device_map="cpu",
            local_files_only=local_files_only,
        )
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            atexit.register(self.save_cache)

    def save_cache(self):
        if not self.cache_path or self.cache_dirty == 0:
            return
        torch.save(self.cache, self.cache_path)
        self.cache_dirty = 0

    def _cache_key(self, row):
        row = row.detach().contiguous().cpu()
        return hashlib.sha1(row.numpy().tobytes()).hexdigest()

    def forward(self, x):
        B, L, C = x.shape
        context = x.permute(0, 2, 1).reshape(B * C, L).detach().float().cpu()
        keys = [self._cache_key(row) for row in context]
        outputs = [self.cache.get(key) for key in keys]
        missing_ids = [idx for idx, value in enumerate(outputs) if value is None]
        if missing_ids:
            missing_context = context[missing_ids]
            with torch.no_grad():
                forecast = self.pipeline.predict(
                    missing_context,
                    prediction_length=self.pred_len,
                    limit_prediction_length=False,
                )
            q_idx = self.quantile_index
            if q_idx is None:
                q_idx = forecast.size(1) // 2
            forecast = forecast[:, q_idx, :].cpu()
            for local_id, row_id in enumerate(missing_ids):
                outputs[row_id] = forecast[local_id]
                self.cache[keys[row_id]] = forecast[local_id]
            self.cache_dirty += len(missing_ids)
            if self.cache_save_interval > 0 and self.cache_dirty >= self.cache_save_interval:
                self.save_cache()
        forecast = torch.stack(outputs, dim=0).to(device=x.device, dtype=x.dtype)
        return forecast.view(B, C, self.pred_len).permute(0, 2, 1)


class Model(nn.Module):
    """
    World-Trajectory Memory Enhancement Module.

    This is a memory-enhanced forecaster, not an end-to-end latent world model.
    It keeps a base forecast, retrieves historical states by z_i^0, discovers
    multi-horizon latent displacement prototypes, decodes memory branches, and
    learns a base-vs-memory fusion gate.
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
        self.kmeans_iters = getattr(configs, "wm_kmeans_iters", 8)
        self.prototype_refine_iters = getattr(configs, "wm_proto_refine_iters", 2)
        self.proto_state_alpha = getattr(configs, "wm_proto_state_alpha", 1.0)
        self.proto_traj_beta = getattr(configs, "wm_proto_traj_beta", 2.0)
        self.use_memory = bool(getattr(configs, "wm_use_memory", 1))
        self.use_branch_discovery = bool(getattr(configs, "wm_use_branch_discovery", 1))
        self.mem_weight = getattr(configs, "wm_mem_weight", 0.1)
        self.traj_weight = getattr(configs, "wm_traj_weight", 0.0)
        self.base_weight = getattr(configs, "wm_base_weight", 0.0)
        self.residual_weight = getattr(configs, "wm_residual_weight", 0.0)
        self.gate_reg_weight = getattr(configs, "wm_gate_reg_weight", 0.0)
        self.gate_reg_threshold = getattr(configs, "wm_gate_reg_threshold", 0.05)
        self.gate_weight = getattr(configs, "wm_gate_weight", 0.0)
        self.gate_margin = getattr(configs, "wm_gate_margin", 0.0)
        self.gate_loss_type = getattr(configs, "wm_gate_loss_type", "pairwise")
        self.gate_soft_weight = getattr(configs, "wm_gate_soft_weight", 0.0)
        self.gate_soft_tau = getattr(configs, "wm_gate_soft_tau", 0.1)
        self.adv_gate_weight = getattr(configs, "wm_adv_gate_weight", 0.0)
        self.adv_gate_margin = getattr(configs, "wm_adv_gate_margin", 0.0)
        self.adv_gate_target = getattr(configs, "wm_adv_gate_target", 0.2)
        self.branch_div_weight = getattr(configs, "wm_branch_div_weight", 0.0)
        self.branch_div_tau = getattr(configs, "wm_branch_div_tau", 1.0)
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
        elif self.base_type == "chronos_bolt":
            self.base_forecaster = ChronosBoltBase(
                getattr(configs, "wm_chronos_model", "amazon/chronos-bolt-tiny"),
                self.pred_len,
                getattr(configs, "wm_chronos_quantile_index", None),
                bool(getattr(configs, "wm_chronos_local_files_only", 0)),
                getattr(configs, "wm_chronos_cache_path", ""),
                getattr(configs, "wm_chronos_cache_save_interval", 512),
            )
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
            backbone=getattr(configs, "wm_backbone", "conv"),
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            patch_len=getattr(configs, "wm_patch_len", 16),
            patch_stride=getattr(configs, "wm_patch_stride", 8),
        )
        decoder_type = getattr(configs, "wm_decoder_type", "mlp")
        if decoder_type == "mlp":
            self.memory_decoder = MemoryBranchDecoder(
                latent_dim=self.latent_dim,
                num_horizons=self.num_horizons,
                pred_len=self.pred_len,
                c_out=self.c_out,
                hidden_dim=configs.d_ff,
                dropout=configs.dropout,
            )
        else:
            self.memory_decoder = SequenceMemoryBranchDecoder(
                latent_dim=self.latent_dim,
                num_horizons=self.num_horizons,
                pred_len=self.pred_len,
                c_out=self.c_out,
                hidden_dim=configs.d_ff,
                dropout=configs.dropout,
                decoder_type=decoder_type,
                n_heads=configs.n_heads,
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

    @torch.no_grad()
    def build_memory(self, data_loader, device):
        if not self.use_memory or self.memory_size <= 0:
            self.memory_ready.fill_(False)
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
            self.memory_ready.fill_(False)
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

        self.memory_states = states.detach()
        self.memory_keys = F.normalize(keys.detach(), dim=-1)
        self.memory_trajectories = trajectories.detach()
        self.memory_values = values.detach()
        self.memory_indices = source_indices.detach()
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
        if not (self.use_memory and bool(self.memory_ready) and self.memory_keys.numel() > 0):
            prototypes, reliability = self._fallback_prototypes(query_key)
            return prototypes, reliability, None

        retrieve_k = min(self.retrieve_k, self.memory_keys.size(0))
        # Retrieval and clustering are memory operations, not differentiable
        # sequence modeling layers. Stop gradients here to avoid unstable
        # TopK/k-means gradients leaking into the state encoder.
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

        if not self.use_branch_discovery:
            order = torch.linspace(0, retrieve_k - 1, steps=self.branch_num, device=query_key.device).long()
            prototypes = neighbor_traj[:, order]
            branch_targets = neighbor_values[:, order]
            state_rel = ((score[:, order] + 1.0) * 0.5).clamp_min(1e-6)
            traj_rel = prototypes.new_ones(B, self.branch_num)
            size_rel = prototypes.new_full((B, self.branch_num), 1.0 / max(1, retrieve_k))
            return prototypes, torch.stack([state_rel, traj_rel, size_rel], dim=-1), branch_targets

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
                    state_rel = ((member_score + 1.0) * 0.5).clamp_min(1e-6)
                    weight = torch.full_like(state_rel, 1.0 / max(1, state_rel.numel()))
                    for _ in range(max(1, self.prototype_refine_iters)):
                        dist = (member - proto.unsqueeze(0)).flatten(start_dim=1).pow(2).sum(dim=-1).sqrt()
                        traj_rel_vec = torch.exp(-dist)
                        weight = state_rel.pow(self.proto_state_alpha) * traj_rel_vec.pow(self.proto_traj_beta)
                        weight = weight / weight.sum().clamp_min(1e-6)
                        proto = (member * weight.view(-1, 1, 1)).sum(dim=0)
                    target = (member_values * weight.view(-1, 1, 1)).sum(dim=0)
                    dist = (member - proto.unsqueeze(0)).flatten(start_dim=1).pow(2).sum(dim=-1).sqrt()
                    traj_rel_vec = torch.exp(-dist)
                    state_rel = state_rel.mean()
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

    def _fusion(self, state, prototypes, reliability, y_base, y_mem, return_logits=False):
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
        logits = torch.cat([base_logit.unsqueeze(-1), mem_logits], dim=-1)
        weights = torch.softmax(logits, dim=-1)
        y_hat = weights[:, :1].unsqueeze(-1) * y_base
        y_hat = y_hat + (weights[:, 1:].unsqueeze(-1).unsqueeze(-1) * y_mem).sum(dim=1)
        if return_logits:
            return y_hat, weights, logits
        return y_hat, weights

    def _forecast_normalized(self, x_enc, future_y=None, query_index=None):
        norm_x, means, stdev = self._normalize(x_enc)
        y_base = self._base_forecast(norm_x)
        state, key = self.state_encoder(norm_x)
        prototypes, reliability, branch_targets = self._discover_prototypes(key, query_index=query_index)
        y_mem = self.memory_decoder(state, prototypes)
        need_gate_logits = self.gate_weight > 0 or self.gate_soft_weight > 0
        if self.training and future_y is not None and need_gate_logits:
            pred, weights, gate_logits = self._fusion(
                state, prototypes, reliability, y_base, y_mem, return_logits=True
            )
        else:
            pred, weights = self._fusion(state, prototypes, reliability, y_base, y_mem)
            gate_logits = None

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
            if self.gate_weight > 0 and gate_logits is not None:
                base_error = (y_base - norm_future).pow(2).mean(dim=(1, 2))
                pred_branch_error = (y_mem - norm_future.unsqueeze(1)).pow(2).mean(dim=(2, 3))
                best_mem_error, best_mem_id = pred_branch_error.min(dim=1)
                if self.gate_loss_type == "ce":
                    gate_target = torch.zeros(norm_future.size(0), dtype=torch.long, device=norm_future.device)
                    use_mem = best_mem_error < (base_error - self.gate_margin)
                    gate_target[use_mem] = best_mem_id[use_mem] + 1
                    gate_loss = F.cross_entropy(gate_logits, gate_target)
                else:
                    best_mem_logit = gate_logits[:, 1:].gather(1, best_mem_id.unsqueeze(1)).squeeze(1)
                    base_logit = gate_logits[:, 0]
                    improvement = (base_error - best_mem_error).detach()
                    use_mem = improvement > self.gate_margin
                    use_base = improvement < -self.gate_margin
                    gate_terms = []
                    if use_mem.any():
                        gate_terms.append(F.softplus(base_logit[use_mem] - best_mem_logit[use_mem]).mean())
                    if use_base.any():
                        gate_terms.append(F.softplus(best_mem_logit[use_base] - base_logit[use_base]).mean())
                    if gate_terms:
                        gate_loss = torch.stack(gate_terms).mean()
                    else:
                        gate_loss = gate_logits.new_tensor(0.0)
                aux_loss = aux_loss + self.gate_weight * gate_loss
            if self.gate_soft_weight > 0 and gate_logits is not None:
                base_error = (y_base - norm_future).pow(2).mean(dim=(1, 2))
                pred_branch_error = (y_mem - norm_future.unsqueeze(1)).pow(2).mean(dim=(2, 3))
                expert_error = torch.cat([base_error.unsqueeze(1), pred_branch_error], dim=1).detach()
                tau = max(float(self.gate_soft_tau), 1e-4)
                soft_target = torch.softmax(-expert_error / tau, dim=-1)
                gate_soft_loss = F.kl_div(
                    F.log_softmax(gate_logits, dim=-1),
                    soft_target,
                    reduction="batchmean",
                )
                aux_loss = aux_loss + self.gate_soft_weight * gate_soft_loss
            if self.adv_gate_weight > 0:
                base_error = (y_base - norm_future).pow(2).mean(dim=(1, 2))
                pred_branch_error = (y_mem - norm_future.unsqueeze(1)).pow(2).mean(dim=(2, 3))
                best_mem_error = pred_branch_error.min(dim=1).values
                improvement = (base_error - best_mem_error).detach()
                mem_weight = weights[:, 1:].sum(dim=1)
                weak_memory = improvement <= self.adv_gate_margin
                adv_terms = []
                if weak_memory.any():
                    adv_terms.append(mem_weight[weak_memory].pow(2).mean())
                strong_memory = improvement > self.adv_gate_margin
                if strong_memory.any() and self.adv_gate_target > 0:
                    target = float(self.adv_gate_target)
                    adv_terms.append(F.relu(target - mem_weight[strong_memory]).pow(2).mean())
                if adv_terms:
                    aux_loss = aux_loss + self.adv_gate_weight * torch.stack(adv_terms).mean()
            if self.base_weight > 0:
                aux_loss = aux_loss + self.base_weight * F.mse_loss(y_base, norm_future)
            if self.residual_weight > 0:
                residual_loss = F.smooth_l1_loss(y_mem, y_base.unsqueeze(1).expand_as(y_mem))
                aux_loss = aux_loss + self.residual_weight * residual_loss
            if self.gate_reg_weight > 0:
                mem_weight = weights[:, 1:].sum(dim=1)
                gate_reg = F.relu(mem_weight - self.gate_reg_threshold).pow(2).mean()
                aux_loss = aux_loss + self.gate_reg_weight * gate_reg
            if self.traj_weight > 0:
                pred_traj = self.traj_head(y_mem.flatten(start_dim=2)).view_as(prototypes)
                aux_loss = aux_loss + self.traj_weight * F.mse_loss(pred_traj, prototypes.detach())
            if self.branch_div_weight > 0 and y_mem.size(1) > 1:
                branch_flat = y_mem.flatten(start_dim=2)
                pair_dist = torch.cdist(branch_flat, branch_flat, p=2)
                off_diag = ~torch.eye(y_mem.size(1), dtype=torch.bool, device=y_mem.device)
                tau = max(float(self.branch_div_tau), 1e-4)
                branch_div_loss = torch.exp(-pair_dist[:, off_diag] / tau).mean()
                aux_loss = aux_loss + self.branch_div_weight * branch_div_loss
            self._last_aux_loss = aux_loss

        return pred, means, stdev

    def get_auxiliary_loss(self):
        return self._last_aux_loss

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=None, query_index=None):
        pred, means, stdev = self._forecast_normalized(x_enc, future_y=future_y, query_index=query_index)
        return self._denormalize(pred, means, stdev)

    def memory_diagnostics(self, x_enc, query_index=None):
        norm_x, means, stdev = self._normalize(x_enc)
        y_base = self._base_forecast(norm_x)
        state, key = self.state_encoder(norm_x)
        prototypes, reliability, branch_targets = self._discover_prototypes(key, query_index=query_index)
        y_mem = self.memory_decoder(state, prototypes)
        pred, weights = self._fusion(state, prototypes, reliability, y_base, y_mem)

        diagnostics = {
            "pred": self._denormalize(pred, means, stdev),
            "base": self._denormalize(y_base, means, stdev),
            "memory": self._denormalize(
                y_mem.flatten(0, 1), means.repeat_interleave(y_mem.size(1), dim=0),
                stdev.repeat_interleave(y_mem.size(1), dim=0),
            ).view(x_enc.size(0), y_mem.size(1), self.pred_len, self.c_out),
            "weights": weights,
        }
        if branch_targets is not None:
            diagnostics["target"] = self._denormalize(
                branch_targets.flatten(0, 1),
                means.repeat_interleave(branch_targets.size(1), dim=0),
                stdev.repeat_interleave(branch_targets.size(1), dim=0),
            ).view(x_enc.size(0), branch_targets.size(1), self.pred_len, self.c_out)
        return diagnostics

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_y=None, query_index=None):
        if self.task_name != "long_term_forecast":
            raise NotImplementedError("BranchWorldModel currently supports long_term_forecast.")
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=future_y, query_index=query_index)
