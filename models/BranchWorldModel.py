import importlib
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
    dynamics prototypes online, and uses them as a residual correction adapter
    around an existing forecaster.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.latent_dim = getattr(configs, "wm_latent_dim", configs.d_model)
        self.branch_num = getattr(configs, "wm_branch_num", 3)
        self.retrieve_k = getattr(configs, "wm_retrieve_k", 64)
        self.exclusion_radius = getattr(configs, "wm_exclusion_radius", -1)
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
        self.residual_context_weight = getattr(configs, "wm_residual_context_weight", 0.01)
        self.alpha_weight = getattr(configs, "wm_alpha_weight", 0.01)
        self.force_alpha = getattr(configs, "wm_force_alpha", -1.0)
        self.confidence_weight = getattr(configs, "wm_confidence_weight", 0.05)
        self.confidence_temperature = getattr(configs, "wm_confidence_temperature", 0.02)
        self.branch_oracle_weight = getattr(configs, "wm_branch_oracle_weight", 0.0)
        self.alpha_max = getattr(configs, "wm_alpha_max", 0.3)
        self.alpha_reliability_power = getattr(configs, "wm_alpha_reliability_power", 1.0)
        self.branch_temperature = getattr(configs, "wm_branch_temperature", 1.0)
        self.branch_dropout = getattr(configs, "wm_branch_dropout", 0.1)
        self.correction_scale = getattr(configs, "wm_correction_scale", 1.0)
        self.conflict_weight = getattr(configs, "wm_conflict_weight", 0.0)
        self.delta_clamp = getattr(configs, "wm_delta_clamp", 3.0)
        self.base_type = getattr(configs, "wm_base_type", "dlinear")
        self.base_model = getattr(configs, "wm_base_model", self.base_type)
        self.base_checkpoint = getattr(configs, "wm_base_checkpoint", "")
        self.mem_loss_type = getattr(configs, "wm_mem_loss_type", "min")
        self.freeze_base = bool(getattr(configs, "wm_freeze_base", 1))
        self.reliability_dim = 5

        horizons = getattr(configs, "wm_horizons", None)
        if horizons is None or len(horizons) == 0:
            horizons = [max(1, self.pred_len // 4), max(1, self.pred_len // 2), max(1, 3 * self.pred_len // 4), self.pred_len]
        self.horizons = sorted(set(int(h) for h in horizons if int(h) > 0 and int(h) <= self.pred_len))
        if not self.horizons:
            self.horizons = [self.pred_len]
        self.num_horizons = len(self.horizons)

        if self.base_model in ["linear", "dlinear"]:
            self.base_type = self.base_model
        if self.base_model == "linear":
            self.base_forecaster = LinearBase(self.seq_len, self.pred_len, self.c_out)
            self.base_is_external = False
        elif self.base_model == "dlinear":
            self.base_forecaster = DLinearBase(
                self.seq_len,
                self.pred_len,
                self.c_out,
                getattr(configs, "moving_avg", 25),
                bool(getattr(configs, "individual", False)),
            )
            self.base_is_external = False
        else:
            if self.base_model == "BranchWorldModel":
                raise ValueError("wm_base_model cannot be BranchWorldModel.")
            module = importlib.import_module("models.{}".format(self.base_model))
            self.base_forecaster = module.Model(configs)
            self.base_is_external = True
        if self.base_checkpoint:
            self._load_base_checkpoint(self.base_checkpoint)
        if self.freeze_base:
            for param in self.base_forecaster.parameters():
                param.requires_grad = False

        self.state_encoder = LatentStateEncoder(
            c_in=configs.enc_in,
            seq_len=configs.seq_len,
            dropout=configs.dropout,
            latent_dim=self.latent_dim,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            backbone=getattr(configs, "wm_backbone", "patch_transformer"),
            patch_len=getattr(configs, "wm_patch_len", 16),
            patch_stride=getattr(configs, "wm_patch_stride", None),
        )
        self.memory_adapter = MemoryBranchDecoder(
            latent_dim=self.latent_dim,
            num_horizons=self.num_horizons,
            pred_len=self.pred_len,
            c_out=self.c_out,
            hidden_dim=configs.d_ff,
            dropout=configs.dropout,
        )
        gate_in_dim = (
            self.latent_dim
            + self.num_horizons * self.latent_dim
            + self.reliability_dim
            + 1
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1),
        )

        self.conf_head = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1),
        )
        nn.init.constant_(self.conf_head[-1].bias, -5.0)
        self._last_aux_loss = None
        self._last_memory_stats = None
        self._last_retrieval_stats = None

        self.register_buffer("memory_keys", torch.empty(0, self.latent_dim), persistent=False)
        self.register_buffer("memory_states", torch.empty(0, self.latent_dim), persistent=False)
        self.register_buffer("memory_trajectories", torch.empty(0, self.num_horizons, self.latent_dim), persistent=False)
        # Base-aware residuals in each memory item's local normalized
        # coordinates: y_i - y_base_i. These are the stored correction patterns
        # that make the memory complementary to the base forecaster.
        self.register_buffer("memory_residuals", torch.empty(0, self.pred_len, self.c_out), persistent=False)
        self.register_buffer("memory_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("memory_proto_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("prototype_bank", torch.empty(0, self.num_horizons, self.latent_dim), persistent=False)
        self.register_buffer("prototype_confidence", torch.empty(0), persistent=False)
        self.register_buffer("prototype_residual_confidence", torch.empty(0), persistent=False)
        self.register_buffer("prototype_support", torch.empty(0), persistent=False)
        self.register_buffer("memory_ready", torch.tensor(False), persistent=False)

    def _load_base_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError("Base checkpoint must contain a state_dict.")

        base_state = self.base_forecaster.state_dict()
        stripped = {k[len("module.") :]: v for k, v in state.items() if k.startswith("module.")}
        base_prefixed = {k[len("base_forecaster.") :]: v for k, v in state.items() if k.startswith("base_forecaster.")}
        module_base_prefixed = {
            k[len("module.base_forecaster.") :]: v
            for k, v in state.items()
            if k.startswith("module.base_forecaster.")
        }

        def official_dlinear_to_internal(candidate):
            mapped = {}
            for key, value in candidate.items():
                mapped_key = key
                if key.startswith("Linear_Seasonal"):
                    mapped_key = "linear_seasonal" + key[len("Linear_Seasonal") :]
                elif key.startswith("Linear_Trend"):
                    mapped_key = "linear_trend" + key[len("Linear_Trend") :]
                mapped[mapped_key] = value
            return mapped

        candidates = [
            state,
            stripped,
            base_prefixed,
            module_base_prefixed,
        ]
        candidates.extend([official_dlinear_to_internal(candidate) for candidate in candidates])

        best = {}
        for candidate in candidates:
            matched = {
                k: v for k, v in candidate.items()
                if k in base_state and tuple(v.shape) == tuple(base_state[k].shape)
            }
            if len(matched) > len(best):
                best = matched

        if not best:
            raise RuntimeError(
                "No compatible parameters found in base checkpoint: {}".format(checkpoint_path)
            )

        missing, unexpected = self.base_forecaster.load_state_dict(best, strict=False)
        print(
            "Loaded base checkpoint: {} (matched {}, missing {}, unexpected {})".format(
                checkpoint_path, len(best), len(missing), len(unexpected)
            )
        )

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

    def _call_external_base(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        y_base = self.base_forecaster(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if isinstance(y_base, tuple):
            y_base = y_base[0]

        if y_base.size(-1) != self.c_out:
            y_base = y_base[:, :, -self.c_out:]

        return y_base[:, -self.pred_len:, :]

    def _base_forecast(self, norm_x, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None, means=None, stdev=None):
        if not self.base_is_external:
            return self.base_forecaster(norm_x[:, :, -self.c_out:])

        # External base models use raw inputs, matching their standalone baseline path.
        y_base_raw = self._call_external_base(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
        )

        if means.size(-1) != y_base_raw.size(-1):
            means_y = means[..., -y_base_raw.size(-1):]
            stdev_y = stdev[..., -y_base_raw.size(-1):]
        else:
            means_y = means
            stdev_y = stdev

        y_base_norm = (y_base_raw - means_y[:, 0, :].unsqueeze(1)) / stdev_y[:, 0, :].unsqueeze(1)
        return y_base_norm

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
        self.memory_residuals = torch.empty(0, self.pred_len, self.c_out, device=device)
        self.memory_indices = torch.empty(0, dtype=torch.long, device=device)
        self.memory_proto_ids = torch.empty(0, dtype=torch.long, device=device)
        self.prototype_bank = torch.empty(0, self.num_horizons, self.latent_dim, device=device)
        self.prototype_confidence = torch.empty(0, device=device)
        self.prototype_residual_confidence = torch.empty(0, device=device)
        self.prototype_support = torch.empty(0, device=device)
        self.memory_ready.fill_(False)

    @torch.no_grad()
    def _build_prototype_bank(self, trajectories, residuals):
        flat_dim = self.num_horizons * self.latent_dim
        proto_num = min(max(1, self.global_proto_num), trajectories.size(0))
        flat = trajectories.reshape(trajectories.size(0), flat_dim)
        centers, assign = kmeans_torch(flat, proto_num, self.kmeans_iters)
        centers = centers.view(proto_num, self.num_horizons, self.latent_dim)

        confidence = trajectories.new_zeros(proto_num)
        residual_confidence = trajectories.new_zeros(proto_num)
        support = trajectories.new_zeros(proto_num)
        for proto_id in range(proto_num):
            mask = assign == proto_id
            support[proto_id] = mask.float().mean()
            if mask.any():
                dist = (trajectories[mask] - centers[proto_id].unsqueeze(0)).flatten(start_dim=1)
                dist = dist.pow(2).sum(dim=-1).sqrt()
                confidence[proto_id] = torch.exp(-dist.mean() / (flat_dim ** 0.5))
                cluster_residual = residuals[mask]
                residual_center = cluster_residual.mean(dim=0, keepdim=True)
                residual_dispersion = (cluster_residual - residual_center).pow(2).mean(dim=(1, 2)).sqrt()
                residual_confidence[proto_id] = torch.exp(-residual_dispersion.mean())

        return (
            centers,
            assign.long(),
            confidence.clamp_min(1e-6),
            residual_confidence.clamp_min(1e-6),
            support.clamp_min(1e-6),
        )

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
        residuals = []
        source_indices = []
        offset = 0

        for batch in data_loader:
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_index = torch.arange(offset, offset + batch_x.size(0))
            offset += batch_x.size(0)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            future_y = batch_y[:, -self.pred_len:, :].float().to(device)
            if future_y.size(-1) != self.c_out:
                future_y = future_y[:, :, -self.c_out:]

            norm_x, means, stdev = self._normalize(batch_x)
            norm_future = (future_y - means[:, :, -future_y.size(-1):]) / stdev[:, :, -future_y.size(-1):]
            dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, : self.label_len, :], dec_inp], dim=1)
            y_base = self._base_forecast(
                norm_x,
                x_enc=batch_x,
                x_mark_enc=batch_x_mark,
                x_dec=dec_inp,
                x_mark_dec=batch_y_mark,
                means=means,
                stdev=stdev,
            )
            base_residual = norm_future - y_base.detach()
            if self.delta_clamp > 0:
                base_residual = base_residual.clamp(-self.delta_clamp, self.delta_clamp)
            state, key, trajectory = self._encode_trajectory(norm_x, norm_future)
            states.append(state.detach().cpu())
            keys.append(key.detach().cpu())
            trajectories.append(trajectory.detach().cpu())
            residuals.append(base_residual.detach().cpu())
            source_indices.append(batch_index.detach().cpu().long())

        if not keys:
            self._reset_memory()
            if was_training:
                self.train()
            return

        states = torch.cat(states, dim=0).to(device)
        keys = torch.cat(keys, dim=0).to(device)
        trajectories = torch.cat(trajectories, dim=0).to(device)
        residuals = torch.cat(residuals, dim=0).to(device)
        source_indices = torch.cat(source_indices, dim=0).to(device)

        if keys.size(0) > self.memory_size:
            ids = torch.linspace(0, keys.size(0) - 1, self.memory_size, device=device).long()
            states = states[ids]
            keys = keys[ids]
            trajectories = trajectories[ids]
            residuals = residuals[ids]
            source_indices = source_indices[ids]

        prototype_bank, proto_ids, proto_confidence, proto_residual_confidence, proto_support = (
            self._build_prototype_bank(trajectories, residuals)
        )

        self.memory_states = states.detach()
        self.memory_keys = F.normalize(keys.detach(), dim=-1)
        self.memory_trajectories = trajectories.detach()
        self.memory_residuals = residuals.detach()
        self.memory_indices = source_indices.detach()
        self.memory_proto_ids = proto_ids.detach()
        self.prototype_bank = prototype_bank.detach()
        self.prototype_confidence = proto_confidence.detach()
        self.prototype_residual_confidence = proto_residual_confidence.detach()
        self.prototype_support = proto_support.detach()
        self.memory_ready.fill_(True)
        if was_training:
            self.train()

    def _fallback_prototypes(self, state):
        B = state.size(0)
        prototypes = state.new_zeros(B, self.branch_num, self.num_horizons, self.latent_dim)
        reliability = state.new_zeros(B, self.branch_num, self.reliability_dim)
        residual_context = state.new_zeros(B, self.branch_num, self.pred_len, self.c_out)
        self._last_retrieval_stats = {
            "retrieval_top1_mean": state.new_tensor(0.0),
            "retrieval_topk_mean": state.new_tensor(0.0),
            "retrieval_topk_std": state.new_tensor(0.0),
            "retrieval_excluded_frac": state.new_tensor(0.0),
            "retrieval_empty_after_exclusion": state.new_tensor(0.0),
            "retrieval_scarce_after_exclusion": state.new_tensor(0.0),
            "retrieval_raw_top1_overlap": state.new_tensor(0.0),
            "retrieval_raw_top1_gap_mean": state.new_tensor(0.0),
        }
        return prototypes, reliability, residual_context

    def _residual_reliability(self, residuals, target):
        if residuals.size(0) <= 1:
            agreement = target.new_tensor(1.0)
        else:
            dispersion = (residuals - target.unsqueeze(0)).pow(2).mean(dim=(1, 2)).sqrt().mean()
            agreement = torch.exp(-dispersion)
        scale = target.abs().mean()
        return agreement.clamp_min(1e-6), scale.clamp_min(1e-6)

    def _discover_prototypes(self, query_key, query_index=None):
        B = query_key.size(0)
        if not (
            self.use_memory
            and bool(self.memory_ready)
            and self.memory_keys.numel() > 0
            and self.prototype_bank.numel() > 0
            and self.memory_proto_ids.numel() == self.memory_keys.size(0)
            and self.memory_residuals.numel() > 0
            and self.memory_residuals.size(0) == self.memory_keys.size(0)
            and self.prototype_residual_confidence.numel() == self.prototype_bank.size(0)
        ):
            return self._fallback_prototypes(query_key)

        retrieve_k = min(self.retrieve_k, self.memory_keys.size(0))
        # Retrieval and prototype selection are memory operations, not
        # differentiable sequence modeling layers. Stop gradients here to avoid
        # unstable TopK/discrete-selection gradients leaking into the encoder.
        query_key = query_key.detach()
        sim_all = query_key @ self.memory_keys.t()
        raw_top_score, raw_top_idx = sim_all.max(dim=-1)
        excluded_frac = sim_all.new_tensor(0.0)
        empty_frac = sim_all.new_tensor(0.0)
        raw_top1_overlap = sim_all.new_tensor(0.0)
        raw_top1_gap_mean = sim_all.new_tensor(0.0)
        scarce_frac = sim_all.new_tensor(0.0)
        if self.training and query_index is not None and self.memory_indices.numel() == self.memory_keys.size(0):
            exclusion_radius = self.exclusion_radius
            if exclusion_radius < 0:
                exclusion_radius = self.seq_len + self.pred_len
            query_index = query_index.to(sim_all.device).long()
            overlap = (query_index.unsqueeze(1) - self.memory_indices.unsqueeze(0)).abs() < exclusion_radius
            excluded_frac = overlap.float().mean()
            raw_top1_gap = (query_index - self.memory_indices[raw_top_idx]).abs().float()
            raw_top1_gap_mean = raw_top1_gap.mean()
            raw_top1_overlap = (raw_top1_gap < float(exclusion_radius)).float().mean()
            sim_all = sim_all.masked_fill(overlap, -torch.inf)
            empty = torch.isneginf(sim_all).all(dim=-1)
            empty_frac = empty.float().mean()
            valid_count = torch.isfinite(sim_all).sum(dim=-1)
            scarce = valid_count < retrieve_k
            scarce_frac = scarce.float().mean()
            if scarce.any():
                fallback_sim = query_key[scarce] @ self.memory_keys.t()
                sim_all[scarce] = fallback_sim
        score, idx = torch.topk(sim_all, k=retrieve_k, dim=-1)
        finite_score = score[torch.isfinite(score)]
        if finite_score.numel() == 0:
            finite_score = raw_top_score.new_zeros(1)
        self._last_retrieval_stats = {
            "retrieval_top1_mean": score[:, 0][torch.isfinite(score[:, 0])].mean()
            if torch.isfinite(score[:, 0]).any() else raw_top_score.new_tensor(0.0),
            "retrieval_topk_mean": finite_score.mean(),
            "retrieval_topk_std": finite_score.std(unbiased=False),
            "retrieval_excluded_frac": excluded_frac.detach(),
            "retrieval_empty_after_exclusion": empty_frac.detach(),
            "retrieval_scarce_after_exclusion": scarce_frac.detach(),
            "retrieval_raw_top1_overlap": raw_top1_overlap.detach(),
            "retrieval_raw_top1_gap_mean": raw_top1_gap_mean.detach(),
        }
        neighbor_traj = self.memory_trajectories[idx]
        neighbor_residuals = self.memory_residuals[idx]
        neighbor_proto_ids = self.memory_proto_ids[idx]

        if not self.use_branch_discovery:
            order = torch.linspace(0, retrieve_k - 1, steps=self.branch_num, device=query_key.device).long()
            prototypes = neighbor_traj[:, order]
            residual_context = neighbor_residuals[:, order]
            state_rel = ((score[:, order] + 1.0) * 0.5).clamp_min(1e-6)
            traj_rel = prototypes.new_ones(B, self.branch_num)
            size_rel = prototypes.new_full((B, self.branch_num), 1.0 / max(1, retrieve_k))
            residual_agree = prototypes.new_ones(B, self.branch_num)
            residual_scale = residual_context.abs().mean(dim=(2, 3)).clamp_min(1e-6)
            reliability = torch.stack([state_rel, traj_rel, size_rel, residual_agree, residual_scale], dim=-1)
            return prototypes, reliability, residual_context

        if self.prototype_mode == "local":
            prototypes = []
            reliabilities = []
            residual_contexts = []
            flat_dim = self.num_horizons * self.latent_dim
            for batch_id in range(B):
                local = neighbor_traj[batch_id]
                local_residuals = neighbor_residuals[batch_id]
                local_flat = local.reshape(retrieve_k, flat_dim)
                _, assign = kmeans_torch(local_flat, self.branch_num, self.kmeans_iters)
                proto_m = []
                rel_m = []
                residual_m = []
                for branch_id in range(self.branch_num):
                    mask = assign == branch_id
                    if not mask.any():
                        best_id = min(branch_id, retrieve_k - 1)
                        proto = local[best_id]
                        residual_target = local_residuals[best_id]
                        state_rel = ((score[batch_id, best_id] + 1.0) * 0.5).clamp_min(1e-6)
                        traj_rel = proto.new_tensor(0.0)
                        size_rel = proto.new_tensor(0.0)
                        residual_agree = proto.new_tensor(1.0)
                        residual_scale = residual_target.abs().mean().clamp_min(1e-6)
                    else:
                        member = local[mask]
                        member_residuals = local_residuals[mask]
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
                        residual_target = (member_residuals * weight.view(-1, 1, 1)).sum(dim=0)
                        dist = (member - proto.unsqueeze(0)).flatten(start_dim=1).pow(2).sum(dim=-1).sqrt()
                        traj_rel_vec = torch.exp(-dist)
                        state_rel = state_rel_vec.mean()
                        traj_rel = traj_rel_vec.mean()
                        size_rel = proto.new_tensor(float(mask.sum().item()) / float(retrieve_k))
                        residual_agree, residual_scale = self._residual_reliability(member_residuals, residual_target)
                    proto_m.append(proto)
                    rel_m.append(torch.stack([state_rel, traj_rel, size_rel, residual_agree, residual_scale]))
                    residual_m.append(residual_target)
                prototypes.append(torch.stack(proto_m, dim=0))
                reliabilities.append(torch.stack(rel_m, dim=0))
                residual_contexts.append(torch.stack(residual_m, dim=0))

            residual_contexts = torch.stack(residual_contexts, dim=0)
            return torch.stack(prototypes, dim=0), torch.stack(reliabilities, dim=0), residual_contexts

        proto_count = self.prototype_bank.size(0)
        branch_count = min(self.branch_num, proto_count)
        state_score = ((score + 1.0) * 0.5).clamp_min(1e-6)
        proto_scores = query_key.new_zeros(B, proto_count)
        proto_scores.scatter_add_(1, neighbor_proto_ids, state_score)
        proto_scores = proto_scores * self.prototype_confidence.unsqueeze(0)
        proto_scores = proto_scores * self.prototype_residual_confidence.unsqueeze(0)
        selected_score, selected_ids = torch.topk(proto_scores, k=branch_count, dim=-1)

        prototypes = []
        reliabilities = []
        residual_contexts = []
        flat_dim = self.num_horizons * self.latent_dim
        for batch_id in range(B):
            local = neighbor_traj[batch_id]
            local_residuals = neighbor_residuals[batch_id]
            proto_m = []
            rel_m = []
            residual_m = []
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
                    residual_target = (local_residuals * weight.view(-1, 1, 1)).sum(dim=0)
                    state_rel = (selected_score[batch_id, branch_id] / float(retrieve_k)).clamp_min(1e-6)
                    traj_rel = self.prototype_confidence[proto_id]
                    size_rel = proto.new_tensor(0.0)
                    residual_agree, residual_scale = self._residual_reliability(local_residuals, residual_target)
                else:
                    member = local[mask]
                    member_residuals = local_residuals[mask]
                    member_score = state_score[batch_id, mask]
                    dist = (member - proto.unsqueeze(0)).flatten(start_dim=1).pow(2).sum(dim=-1).sqrt()
                    traj_rel_vec = torch.exp(-dist / (flat_dim ** 0.5))
                    weight = member_score.pow(self.proto_state_alpha) * traj_rel_vec.pow(self.proto_traj_beta)
                    weight = weight / weight.sum().clamp_min(1e-6)
                    residual_target = (member_residuals * weight.view(-1, 1, 1)).sum(dim=0)
                    state_rel = (selected_score[batch_id, branch_id] / float(retrieve_k)).clamp_min(1e-6)
                    traj_rel = traj_rel_vec.mean()
                    size_rel = proto.new_tensor(float(mask.sum().item()) / float(retrieve_k))
                    residual_agree, residual_scale = self._residual_reliability(member_residuals, residual_target)
                proto_m.append(proto)
                rel_m.append(torch.stack([state_rel, traj_rel, size_rel, residual_agree, residual_scale]))
                residual_m.append(residual_target)

            if branch_count < self.branch_num:
                pad = self.branch_num - branch_count
                proto_m.extend([proto_m[-1].clone() for _ in range(pad)])
                rel_m.extend([rel_m[-1].clone() * 0.0 for _ in range(pad)])
                residual_m.extend([residual_m[-1].clone() * 0.0 for _ in range(pad)])

            prototypes.append(torch.stack(proto_m, dim=0))
            reliabilities.append(torch.stack(rel_m, dim=0))
            residual_contexts.append(torch.stack(residual_m, dim=0))

        residual_contexts = torch.stack(residual_contexts, dim=0)
        return torch.stack(prototypes, dim=0), torch.stack(reliabilities, dim=0), residual_contexts

    def _fusion(self, state, prototypes, reliability, y_base, delta_mem):
        """
        Prototype-guided residual adapter with sample-wise memory confidence.

        y_hat = y_base + alpha_q * sum_m w_m * Delta_m

        where:
            w_m     decides which memory branch direction to use
            alpha_q decides how much memory should be used for each sample
        """
        B, M = delta_mem.shape[:2]

        correction_scale = delta_mem.abs().mean(dim=(2, 3)).unsqueeze(-1)

        branch_features = torch.cat([reliability, correction_scale], dim=-1)

        gate_in = torch.cat([
            state.unsqueeze(1).expand(B, M, self.latent_dim),
            prototypes.flatten(start_dim=2),
            branch_features,
        ], dim=-1)

        tau = max(float(self.branch_temperature), 1e-6)
        mem_logits = self.gate(gate_in).squeeze(-1) / tau
        mem_weights = torch.softmax(mem_logits, dim=-1)
        if self.training and self.branch_dropout > 0:
            mem_weights = F.dropout(mem_weights, p=self.branch_dropout, training=True)
            empty = mem_weights.sum(dim=-1, keepdim=True) <= 1e-6
            mem_weights = torch.where(empty, torch.softmax(mem_logits, dim=-1), mem_weights)
            mem_weights = mem_weights / mem_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        conf_in = gate_in.mean(dim=1)
        raw_alpha_sample = torch.sigmoid(self.conf_head(conf_in))
        raw_alpha = raw_alpha_sample.view(B, 1, 1).expand(B, self.pred_len, self.c_out)
        reliability_score = (
            reliability[..., 0].clamp_min(1e-6)
            * reliability[..., 1].clamp_min(1e-6)
            * reliability[..., 3].clamp_min(1e-6)
        ).pow(1.0 / 3.0)
        reliability_score = (
            mem_weights.detach() * reliability_score
        ).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        if self.alpha_reliability_power != 1.0:
            reliability_score = reliability_score.pow(self.alpha_reliability_power)
        alpha_upper = (self.alpha_max * reliability_score).view(B, 1, 1)
        if self.force_alpha >= 0:
            forced = max(0.0, float(self.force_alpha))
            alpha = raw_alpha.new_full((B, self.pred_len, self.c_out), forced)
        else:
            alpha = alpha_upper * raw_alpha

        branch_scale = reliability[..., 4].clamp_min(1e-6).view(B, M, 1, 1)
        branch_scale = (branch_scale * self.correction_scale).clamp_min(0.05)
        branch_delta = torch.tanh(delta_mem / branch_scale.clamp_min(1e-6)) * branch_scale

        delta_raw = (
            mem_weights.unsqueeze(-1).unsqueeze(-1)
            * delta_mem
        ).sum(dim=1)
        delta_scale = (
            mem_weights * reliability[..., 4].clamp_min(1e-6)
        ).sum(dim=1).view(B, 1, 1).clamp_min(0.05)
        delta_scale = delta_scale * self.correction_scale
        delta = torch.tanh(delta_raw / delta_scale.clamp_min(1e-6)) * delta_scale

        y_hat = y_base + alpha * delta

        sample_alpha = alpha.mean(dim=(1, 2), keepdim=True)
        effective_mem = sample_alpha.squeeze(-1) * mem_weights
        effective_base = 1.0 - sample_alpha.squeeze(-1)
        weights = torch.cat([effective_base, effective_mem], dim=-1)
        stats = {
            "alpha_mean": alpha.detach().mean(),
            "alpha_max": alpha.detach().max(),
            "raw_alpha_mean": raw_alpha.detach().mean(),
            "raw_alpha_std": raw_alpha.detach().std(unbiased=False),
            "reliability_gate_mean": reliability_score.detach().mean(),
            "reliability_gate_std": reliability_score.detach().std(unbiased=False),
            "residual_agreement_mean": reliability[..., 3].detach().mean(),
            "memory_residual_abs_mean": reliability[..., 4].detach().mean(),
            "delta_abs_mean": delta.detach().abs().mean(),
            "effective_delta_abs_mean": (alpha * delta).detach().abs().mean(),
            "correction_scale_mean": correction_scale.detach().mean(),
            "branch_weight_max_mean": mem_weights.detach().max(dim=1).values.mean(),
            "branch_weight_entropy": (
                -(mem_weights.detach() * mem_weights.detach().clamp_min(1e-8).log()).sum(dim=1).mean()
            ),
        }
        if self._last_retrieval_stats:
            stats.update(self._last_retrieval_stats)

        return y_hat, weights, mem_weights, delta, branch_delta, alpha, raw_alpha, alpha_upper, stats

    def _binary_auc(self, scores, labels):
        scores = scores.detach().flatten()
        labels = labels.detach().flatten() > 0.5
        pos = scores[labels]
        neg = scores[~labels]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.new_tensor(0.5)
        cmp = (pos.unsqueeze(1) > neg.unsqueeze(0)).float()
        ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float() * 0.5
        return (cmp + ties).mean()

    def _raw_sample_oracle(self, y_base, delta, future_y, means, stdev):
        y_base_raw = self._denormalize(y_base, means, stdev)
        candidate = y_base + delta.detach()
        candidate_raw = self._denormalize(candidate, means, stdev)
        base_err = (y_base_raw - future_y).pow(2).mean(dim=(1, 2))
        candidate_err = (candidate_raw - future_y).pow(2).mean(dim=(1, 2))
        gain = base_err - candidate_err
        temperature = max(float(self.confidence_temperature), 1e-6)
        soft_oracle = torch.sigmoid(gain / temperature)
        hard_oracle = (gain > 0).float()
        return base_err, candidate_err, gain, soft_oracle, hard_oracle

    def _raw_branch_oracle(self, y_base, branch_delta, future_y, means, stdev):
        y_base_raw = self._denormalize(y_base, means, stdev)
        base_err = (y_base_raw - future_y).pow(2).mean(dim=(1, 2))
        B, M = branch_delta.shape[:2]
        y_base_branch = y_base.unsqueeze(1).expand(B, M, self.pred_len, self.c_out)
        candidate_raw = self._denormalize(
            (y_base_branch + branch_delta.detach()).reshape(B * M, self.pred_len, self.c_out),
            means.unsqueeze(1).expand(B, M, *means.shape[1:]).reshape(B * M, *means.shape[1:]),
            stdev.unsqueeze(1).expand(B, M, *stdev.shape[1:]).reshape(B * M, *stdev.shape[1:]),
        ).view(B, M, self.pred_len, self.c_out)
        branch_err = (candidate_raw - future_y.unsqueeze(1)).pow(2).mean(dim=(2, 3))
        gain = base_err.unsqueeze(1) - branch_err
        temperature = max(float(self.confidence_temperature), 1e-6)
        soft_oracle = torch.softmax(gain / temperature, dim=1)
        hard_oracle = F.one_hot(gain.argmax(dim=1), num_classes=M).to(gain.dtype)
        return base_err, branch_err, gain, soft_oracle, hard_oracle

    def _corrcoef(self, x, y):
        x = x.detach().flatten().float()
        y = y.detach().flatten().float()
        if x.numel() < 2 or y.numel() < 2:
            return x.new_tensor(0.0)
        x = x - x.mean()
        y = y - y.mean()
        denom = x.pow(2).mean().sqrt() * y.pow(2).mean().sqrt()
        if denom <= 1e-12:
            return x.new_tensor(0.0)
        return (x * y).mean() / denom

    def _forecast_normalized(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, future_y=None, query_index=None):
        norm_x, means, stdev = self._normalize(x_enc)

        y_base = self._base_forecast(
            norm_x,
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            means=means,
            stdev=stdev,
        )

        # Pure base mode: do not pass through memory encoder / decoder / fusion.
        if not self.use_memory:
            self._last_aux_loss = None
            self._last_memory_stats = None
            return y_base, means, stdev

        state, key = self.state_encoder(norm_x)
        prototypes, reliability, residual_context = self._discover_prototypes(key, query_index=query_index)
        delta_mem = self.memory_adapter(state, prototypes, y_base.detach(), residual_context.detach())
        pred, weights, mem_weights, delta, branch_delta, alpha, raw_alpha, alpha_upper, stats = self._fusion(
            state, prototypes, reliability, y_base, delta_mem
        )

        self._last_aux_loss = None
        norm_future = None
        if future_y is not None:
            if future_y.size(-1) != self.c_out:
                future_y = future_y[:, :, -self.c_out:]
            norm_future = (future_y - means[:, :, -future_y.size(-1):]) / stdev[:, :, -future_y.size(-1):]
            base_mse_norm = (y_base - norm_future).pow(2).mean()
            adapted_mse_norm = (pred - norm_future).pow(2).mean()
            y_base_raw = self._denormalize(y_base, means, stdev)
            pred_raw = self._denormalize(pred, means, stdev)
            base_mse_raw = (y_base_raw - future_y).pow(2).mean()
            adapted_mse_raw = (pred_raw - future_y).pow(2).mean()
            _, _, sample_gain, soft_oracle, hard_oracle = self._raw_sample_oracle(
                y_base, delta, future_y, means, stdev
            )
            _, _, branch_gain, branch_soft, branch_hard = self._raw_branch_oracle(
                y_base, branch_delta, future_y, means, stdev
            )
            sample_raw_alpha = raw_alpha[:, 0, 0].detach()
            reliability_sample = alpha_upper.view(alpha_upper.size(0)).detach()
            branch_pred = mem_weights.detach().argmax(dim=1)
            branch_label = branch_hard.argmax(dim=1)
            stdev_y = stdev[..., -future_y.size(-1):]
            stats = dict(stats)
            stats.update({
                "base_mse": base_mse_raw.detach(),
                "adapted_mse": adapted_mse_raw.detach(),
                "mse_gain": (base_mse_raw - adapted_mse_raw).detach(),
                "base_mse_norm": base_mse_norm.detach(),
                "adapted_mse_norm": adapted_mse_norm.detach(),
                "mse_gain_norm": (base_mse_norm - adapted_mse_norm).detach(),
                "oracle_soft_mean": soft_oracle.detach().mean(),
                "oracle_hard_mean": hard_oracle.detach().mean(),
                "oracle_gain_mean": sample_gain.detach().mean(),
                "oracle_gain_std": sample_gain.detach().std(unbiased=False),
                "gate_acc": ((sample_raw_alpha > 0.5).float() == hard_oracle.detach()).float().mean(),
                "gate_auc": self._binary_auc(sample_raw_alpha, hard_oracle),
                "gate_gain_corr": self._corrcoef(sample_raw_alpha, sample_gain),
                "reliability_gain_corr": self._corrcoef(reliability_sample, sample_gain),
                "branch_oracle_gain_mean": branch_gain.detach().max(dim=1).values.mean(),
                "branch_oracle_entropy": (
                    -(branch_soft.detach() * branch_soft.detach().clamp_min(1e-8).log()).sum(dim=1).mean()
                ),
                "branch_oracle_acc": (branch_pred == branch_label).float().mean(),
                "stdev_mean": stdev_y.detach().mean(),
                "stdev_min": stdev_y.detach().min(),
                "stdev_p01": torch.quantile(stdev_y.detach().flatten(), 0.01),
            })
        self._last_memory_stats = stats

        if self.training and norm_future is not None:
            delta_target = (norm_future - y_base).detach()
            if self.delta_clamp > 0:
                delta_target = delta_target.clamp(-self.delta_clamp, self.delta_clamp)
            branch_error = (delta_mem - delta_target.unsqueeze(1)).pow(2).mean(dim=(2, 3))

            if self.mem_loss_type == "all":
                mem_loss = branch_error.mean()
            elif self.mem_loss_type == "weighted":
                mem_loss = (weights[:, 1:].detach() * branch_error).sum(dim=1).mean()
            else:
                mem_loss = branch_error.min(dim=1).values.mean()

            aux_loss = self.mem_weight * mem_loss
            if self.residual_context_weight > 0:
                context_loss = (delta_mem - residual_context.detach()).pow(2).mean()
                aux_loss = aux_loss + self.residual_context_weight * context_loss
            if self.alpha_weight > 0:
                aux_loss = aux_loss + self.alpha_weight * alpha.mean()
            if self.confidence_weight > 0:
                with torch.no_grad():
                    _, _, _, confidence_target, _ = self._raw_sample_oracle(
                        y_base, delta, future_y, means, stdev
                    )
                    confidence_target = confidence_target.view(-1, 1, 1).expand_as(raw_alpha)
                confidence_loss = F.binary_cross_entropy(
                    raw_alpha.clamp(1e-5, 1.0 - 1e-5),
                    confidence_target,
                )
                aux_loss = aux_loss + self.confidence_weight * confidence_loss
            if self.branch_oracle_weight > 0:
                with torch.no_grad():
                    _, _, _, branch_target, _ = self._raw_branch_oracle(
                        y_base, branch_delta, future_y, means, stdev
                    )
                branch_loss = -(
                    branch_target * mem_weights.clamp_min(1e-8).log()
                ).sum(dim=1).mean()
                aux_loss = aux_loss + self.branch_oracle_weight * branch_loss
            if self.conflict_weight > 0:
                aux_loss = aux_loss + self.conflict_weight * (alpha * delta.abs()).mean()

            self._last_aux_loss = aux_loss

        return pred, means, stdev

    def get_auxiliary_loss(self):
        return self._last_aux_loss

    def get_memory_stats(self):
        return self._last_memory_stats

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=None, query_index=None):
        pred, means, stdev = self._forecast_normalized(
            x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            future_y=future_y,
            query_index=query_index,
        )
        return self._denormalize(pred, means, stdev)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_y=None, query_index=None):
        if self.task_name != "long_term_forecast":
            raise NotImplementedError("BranchWorldModel currently supports long_term_forecast.")
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=future_y, query_index=query_index)
