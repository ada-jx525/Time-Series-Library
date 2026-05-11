import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.BranchWorld import LatentStateEncoder, kmeans_torch


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
    Minimal latent-trajectory memory correction.

    The memory is built only from training windows. Each item stores:
      1. current latent key/state,
      2. latent transition trajectory from current window to future-shifted windows,
      3. normalized base residual y_true - y_base.

    Offline k-means clusters the latent transition trajectories into a small
    global prototype bank. Each cluster keeps the mean key and mean residual.
    Online inference uses only the current input window: query key -> soft
    attention over global prototype keys -> weighted residual correction.
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
        self.memory_size = getattr(configs, "wm_memory_size", 4096)
        self.global_proto_num = getattr(configs, "wm_global_proto_num", 32)
        self.kmeans_iters = getattr(configs, "wm_kmeans_iters", 8)
        self.use_memory = bool(getattr(configs, "wm_use_memory", 1))
        self.attention_mode = getattr(configs, "wm_attention_mode", "dot")
        if self.attention_mode not in ["dot", "linear", "oracle"]:
            raise ValueError("wm_attention_mode must be 'dot', 'linear', or 'oracle'.")
        self.temperature = max(float(getattr(configs, "wm_branch_temperature", 1.0)), 1e-6)
        self.correction_lambda = float(getattr(configs, "wm_correction_lambda", 0.1))
        self.predictor_loss = getattr(configs, "wm_predictor_loss", "ce")
        if self.predictor_loss not in ["ce", "mse"]:
            raise ValueError("wm_predictor_loss must be 'ce' or 'mse'.")
        self.predictor_checkpoint = getattr(configs, "wm_predictor_checkpoint", "")
        self.predictor_ce_weight = float(getattr(configs, "wm_predictor_ce_weight", 1.0))
        self.predictor_ce_only = bool(getattr(configs, "wm_predictor_ce_only", 1))
        self.predictor_entropy_weight = float(getattr(configs, "wm_predictor_entropy_weight", 0.0))
        self.predictor_context = getattr(configs, "wm_predictor_context", "z0")
        if self.predictor_context not in ["z0", "summary"]:
            raise ValueError("wm_predictor_context must be 'z0' or 'summary'.")
        self.predictor_type = getattr(configs, "wm_predictor_type", "linear")
        if self.predictor_type not in ["linear", "mlp"]:
            raise ValueError("wm_predictor_type must be 'linear' or 'mlp'.")
        self.predictor_hidden_dim = int(getattr(configs, "wm_predictor_hidden_dim", max(128, self.latent_dim)))
        self.residual_encoder_type = getattr(configs, "wm_residual_encoder", "identity")
        if self.residual_encoder_type not in ["identity", "linear", "mlp"]:
            raise ValueError("wm_residual_encoder must be 'identity', 'linear', or 'mlp'.")
        self.residual_latent_dim = int(getattr(configs, "wm_residual_latent_dim", self.latent_dim))
        self.residual_hidden_dim = int(getattr(configs, "wm_residual_hidden_dim", max(128, self.latent_dim)))
        self.hard_selection = bool(getattr(configs, "wm_hard_selection", 0))
        self.delta_clamp = float(getattr(configs, "wm_delta_clamp", 3.0))
        self.base_type = getattr(configs, "wm_base_type", "dlinear")
        self.base_model = getattr(configs, "wm_base_model", self.base_type)
        self.base_checkpoint = getattr(configs, "wm_base_checkpoint", "")
        self.freeze_base = bool(getattr(configs, "wm_freeze_base", 1))
        self.freeze_encoder = bool(getattr(configs, "wm_freeze_encoder", 1))

        horizons = getattr(configs, "wm_horizons", None)
        if horizons is None or len(horizons) == 0:
            horizons = [
                max(1, self.pred_len // 4),
                max(1, self.pred_len // 2),
                max(1, 3 * self.pred_len // 4),
                self.pred_len,
            ]
        self.horizons = sorted(set(int(h) for h in horizons if 0 < int(h) <= self.pred_len))
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
        if self.freeze_encoder:
            for param in self.state_encoder.parameters():
                param.requires_grad = False
        if self.residual_encoder_type == "mlp":
            self.residual_encoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.residual_hidden_dim, self.residual_latent_dim),
            )
        elif self.residual_encoder_type == "linear":
            self.residual_encoder = nn.Linear(self.latent_dim, self.residual_latent_dim)
        else:
            self.residual_encoder = nn.Identity()
            self.residual_latent_dim = self.latent_dim

        self.predictor_input_dim = self.residual_latent_dim
        if self.predictor_context == "summary":
            self.predictor_input_dim += 8 * self.c_out
        self.predictor_norm = nn.LayerNorm(self.predictor_input_dim) if self.predictor_context == "summary" else nn.Identity()
        if self.predictor_type == "mlp":
            self.cluster_predictor = nn.Sequential(
                nn.Linear(self.predictor_input_dim, self.predictor_hidden_dim),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.predictor_hidden_dim, self.global_proto_num),
            )
        else:
            self.cluster_predictor = nn.Linear(self.predictor_input_dim, self.global_proto_num)
        if self.predictor_checkpoint:
            self._load_predictor_checkpoint(self.predictor_checkpoint)

        self._last_aux_loss = None
        self._last_memory_stats = None

        self.register_buffer("prototype_keys", torch.empty(0, self.latent_dim), persistent=False)
        self.register_buffer("prototype_trajectories", torch.empty(0, self.num_horizons, self.latent_dim), persistent=False)
        self.register_buffer("prototype_residuals", torch.empty(0, self.pred_len, self.c_out), persistent=False)
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

        candidates = [state, stripped, base_prefixed, module_base_prefixed]
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
            raise RuntimeError("No compatible parameters found in base checkpoint: {}".format(checkpoint_path))

        missing, unexpected = self.base_forecaster.load_state_dict(best, strict=False)
        print(
            "Loaded base checkpoint: {} (matched {}, missing {}, unexpected {})".format(
                checkpoint_path, len(best), len(missing), len(unexpected)
            )
        )

    def _load_predictor_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError("Predictor checkpoint must contain a state_dict.")

        predictor_state = self.cluster_predictor.state_dict()
        candidates = [
            state,
            {k[len("module.") :]: v for k, v in state.items() if k.startswith("module.")},
            {k[len("cluster_predictor.") :]: v for k, v in state.items() if k.startswith("cluster_predictor.")},
            {
                k[len("module.cluster_predictor.") :]: v
                for k, v in state.items()
                if k.startswith("module.cluster_predictor.")
            },
        ]
        best = {}
        for candidate in candidates:
            matched = {
                k: v for k, v in candidate.items()
                if k in predictor_state and tuple(v.shape) == tuple(predictor_state[k].shape)
            }
            if len(matched) > len(best):
                best = matched

        if not best:
            raise RuntimeError("No compatible predictor parameters found in checkpoint: {}".format(checkpoint_path))
        missing, unexpected = self.cluster_predictor.load_state_dict(best, strict=False)
        print(
            "Loaded predictor checkpoint: {} (matched {}, missing {}, unexpected {})".format(
                checkpoint_path, len(best), len(missing), len(unexpected)
            )
        )
        self._load_optional_submodule_from_checkpoint(state, "residual_encoder", checkpoint_path)
        self._load_optional_submodule_from_checkpoint(state, "predictor_norm", checkpoint_path)

    def _load_optional_submodule_from_checkpoint(self, state, module_name, checkpoint_path):
        module = getattr(self, module_name, None)
        if module is None:
            return
        module_state = module.state_dict()
        if not module_state:
            return

        candidates = [
            {k[len(module_name) + 1 :]: v for k, v in state.items() if k.startswith(module_name + ".")},
            {
                k[len("module." + module_name) + 1 :]: v
                for k, v in state.items()
                if k.startswith("module." + module_name + ".")
            },
        ]
        best = {}
        for candidate in candidates:
            matched = {
                k: v for k, v in candidate.items()
                if k in module_state and tuple(v.shape) == tuple(module_state[k].shape)
            }
            if len(matched) > len(best):
                best = matched
        if best:
            missing, unexpected = module.load_state_dict(best, strict=False)
            print(
                "Loaded {} from predictor checkpoint: {} (matched {}, missing {}, unexpected {})".format(
                    module_name, checkpoint_path, len(best), len(missing), len(unexpected)
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

        y_base_raw = self._call_external_base(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
        )
        means_y = means[..., -y_base_raw.size(-1):]
        stdev_y = stdev[..., -y_base_raw.size(-1):]
        return (y_base_raw - means_y[:, 0, :].unsqueeze(1)) / stdev_y[:, 0, :].unsqueeze(1)

    def _shifted_windows(self, norm_x, norm_future):
        series = torch.cat([norm_x, norm_future], dim=1)
        return [series[:, horizon : horizon + self.seq_len] for horizon in self.horizons]

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
        self.prototype_keys = torch.empty(0, self.latent_dim, device=device)
        self.prototype_trajectories = torch.empty(0, self.num_horizons, self.latent_dim, device=device)
        self.prototype_residuals = torch.empty(0, self.pred_len, self.c_out, device=device)
        self.prototype_support = torch.empty(0, device=device)
        self.memory_ready.fill_(False)

    @torch.no_grad()
    def _build_prototype_bank(self, keys, trajectories, residuals):
        proto_num = min(max(1, self.global_proto_num), trajectories.size(0))
        flat = trajectories.flatten(start_dim=1)
        centers, assign = kmeans_torch(flat, proto_num, self.kmeans_iters)
        trajectory_centers = centers.view(proto_num, self.num_horizons, self.latent_dim)
        key_centers = keys.new_zeros(proto_num, self.latent_dim)
        residual_centers = residuals.new_zeros(proto_num, self.pred_len, self.c_out)
        support = residuals.new_zeros(proto_num)

        for proto_id in range(proto_num):
            mask = assign == proto_id
            if mask.any():
                key_centers[proto_id] = keys[mask].mean(dim=0)
                residual_centers[proto_id] = residuals[mask].mean(dim=0)
                support[proto_id] = mask.float().mean()
            else:
                key_centers[proto_id] = keys[0]
                residual_centers[proto_id] = residuals[0]
        return F.normalize(key_centers, dim=-1), trajectory_centers, residual_centers, support.clamp_min(1e-6)

    @torch.no_grad()
    def build_memory(self, data_loader, device):
        if not self.use_memory or self.memory_size <= 0:
            self._reset_memory()
            return

        was_training = self.training
        self.eval()
        keys = []
        trajectories = []
        residuals = []

        for batch in data_loader:
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
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
            _, key, trajectory = self._encode_trajectory(norm_x, norm_future)
            keys.append(key.detach().cpu())
            trajectories.append(trajectory.detach().cpu())
            residuals.append(base_residual.detach().cpu())

        if not keys:
            self._reset_memory()
            if was_training:
                self.train()
            return

        keys = torch.cat(keys, dim=0).to(device)
        trajectories = torch.cat(trajectories, dim=0).to(device)
        residuals = torch.cat(residuals, dim=0).to(device)

        if keys.size(0) > self.memory_size:
            ids = torch.linspace(0, keys.size(0) - 1, self.memory_size, device=device).long()
            keys = keys[ids]
            trajectories = trajectories[ids]
            residuals = residuals[ids]

        proto_keys, trajectory_centers, proto_residuals, support = self._build_prototype_bank(keys, trajectories, residuals)
        self.prototype_keys = proto_keys.detach()
        self.prototype_trajectories = trajectory_centers.detach()
        self.prototype_residuals = proto_residuals.detach()
        self.prototype_support = support.detach()
        self.memory_ready.fill_(True)
        if was_training:
            self.train()

    def _assign_cluster_labels(self, trajectory):
        if self.prototype_trajectories.numel() == 0:
            return None
        centers = self.prototype_trajectories.flatten(start_dim=1)
        flat = trajectory.flatten(start_dim=1)
        return torch.cdist(flat, centers).argmin(dim=1)

    def _series_summary(self, series):
        first = series[:, 0, :]
        last = series[:, -1, :]
        mean = series.mean(dim=1)
        std = series.std(dim=1, unbiased=False)
        slope = last - first
        return torch.cat([mean, std, last, slope], dim=-1)

    def _predictor_features(self, query_key, norm_x, y_base):
        residual_key = self.residual_encoder(query_key)
        if self.predictor_context == "z0":
            return residual_key
        x_summary = self._series_summary(norm_x[:, :, -self.c_out:])
        base_summary = self._series_summary(y_base)
        return torch.cat([residual_key, x_summary, base_summary], dim=-1)

    def _memory_correction(self, query_key, y_base, norm_x=None, oracle_labels=None):
        B = query_key.size(0)
        if not (
            self.use_memory
            and bool(self.memory_ready)
            and self.prototype_keys.numel() > 0
            and self.prototype_residuals.numel() > 0
        ):
            correction = y_base.new_zeros(B, self.pred_len, self.c_out)
            weights = y_base.new_zeros(B, 0)
            logits = y_base.new_zeros(B, 0)
            return correction, weights, logits

        proto_count = self.prototype_residuals.size(0)
        if self.attention_mode == "oracle" and oracle_labels is not None:
            logits = y_base.new_full((B, proto_count), -20.0)
            logits.scatter_(1, oracle_labels.view(B, 1), 20.0)
        elif self.attention_mode == "linear":
            predictor_features = self._predictor_features(query_key, norm_x, y_base)
            predictor_features = self.predictor_norm(predictor_features)
            logits = self.cluster_predictor(predictor_features)[:, :proto_count]
        else:
            logits = query_key @ self.prototype_keys.t()
        weights = torch.softmax(logits / self.temperature, dim=-1)
        if self.hard_selection and self.attention_mode in ["linear", "dot"]:
            hard = F.one_hot(weights.argmax(dim=1), num_classes=proto_count).to(weights.dtype)
            weights = hard.detach() - weights.detach() + weights
        correction = torch.einsum("bp,plc->blc", weights, self.prototype_residuals)
        if self.delta_clamp > 0:
            correction = correction.clamp(-self.delta_clamp, self.delta_clamp)
        return correction, weights, logits

    def _forecast_normalized(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, future_y=None, query_index=None):
        del query_index
        self._last_aux_loss = None
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

        if not self.use_memory:
            self._last_memory_stats = None
            return y_base, means, stdev

        _, key = self.state_encoder(norm_x)
        norm_future = None
        oracle_labels = None

        need_proto_label = (
        self.attention_mode == "oracle"
        or (self.attention_mode == "linear" and self.predictor_loss == "ce")
        or (self.training and self.proto_align_weight > 0)
        )
        if future_y is not None:
            if future_y.size(-1) != self.c_out:
                future_y = future_y[:, :, -self.c_out:]
            norm_future = (future_y - means[:, :, -future_y.size(-1):]) / stdev[:, :, -future_y.size(-1):]
            if (
                self.attention_mode == "oracle"
                or (self.attention_mode == "linear" and self.predictor_loss == "ce")
            ) and self.prototype_trajectories.numel() > 0:
                with torch.set_grad_enabled(False):
                    _, _, trajectory = self._encode_trajectory(norm_x, norm_future)
                oracle_labels = self._assign_cluster_labels(trajectory.detach())

        correction, weights, logits = self._memory_correction(key, y_base, norm_x=norm_x, oracle_labels=oracle_labels)
        effective_correction = correction
        if (
            self.training
            and self.attention_mode == "linear"
            and self.predictor_loss == "ce"
            and self.predictor_ce_only
        ):
            effective_correction = correction.detach()
        pred = y_base + self.correction_lambda * effective_correction

        stats = {
            "memory_ready": y_base.new_tensor(float(bool(self.memory_ready))),
            "prototype_count": y_base.new_tensor(float(self.prototype_residuals.size(0))),
            "linear_attention": y_base.new_tensor(float(self.attention_mode == "linear")),
            "oracle_attention": y_base.new_tensor(float(self.attention_mode == "oracle")),
            "correction_lambda": y_base.new_tensor(self.correction_lambda),
            "correction_abs_mean": correction.detach().abs().mean(),
            "effective_correction_abs_mean": (self.correction_lambda * correction).detach().abs().mean(),
        }
        if weights.numel() > 0:
            weight_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
            stats.update({
                "prototype_weight_max_mean": weights.detach().max(dim=1).values.mean(),
                "prototype_weight_entropy": weight_entropy.detach(),
            })
        else:
            stats.update({
                "prototype_weight_max_mean": y_base.new_tensor(0.0),
                "prototype_weight_entropy": y_base.new_tensor(0.0),
            })

        if future_y is not None:
            y_base_raw = self._denormalize(y_base, means, stdev)
            pred_raw = self._denormalize(pred, means, stdev)
            base_mse_raw = (y_base_raw - future_y).pow(2).mean()
            adapted_mse_raw = (pred_raw - future_y).pow(2).mean()
            stats.update({
                "base_mse": base_mse_raw.detach(),
                "adapted_mse": adapted_mse_raw.detach(),
                "mse_gain": (base_mse_raw - adapted_mse_raw).detach(),
                "base_mse_norm": (y_base - norm_future).pow(2).mean().detach(),
                "adapted_mse_norm": (pred - norm_future).pow(2).mean().detach(),
                "mse_gain_norm": ((y_base - norm_future).pow(2).mean() - (pred - norm_future).pow(2).mean()).detach(),
            })
            if (
                self.attention_mode == "linear"
                and self.predictor_loss == "ce"
                and logits.numel() > 0
                and oracle_labels is not None
            ):
                predictor_ce = F.cross_entropy(logits, oracle_labels)
                predictor_acc = (logits.detach().argmax(dim=1) == oracle_labels).float().mean()
                stats.update({
                    "predictor_ce": predictor_ce.detach(),
                    "predictor_acc": predictor_acc.detach(),
                })
                if self.training:
                    self._last_aux_loss = self.predictor_ce_weight * predictor_ce
            if (
                self.training
                and self.attention_mode == "linear"
                and self.predictor_loss == "mse"
                and self.predictor_entropy_weight > 0
                and weights.numel() > 0
            ):
                self._last_aux_loss = -self.predictor_entropy_weight * weight_entropy

        self._last_memory_stats = stats
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
        del mask
        if self.task_name != "long_term_forecast":
            raise NotImplementedError("BranchWorldModel currently supports long_term_forecast.")
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=future_y, query_index=query_index)
