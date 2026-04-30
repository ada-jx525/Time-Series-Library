import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.BranchWorld import (
    BranchAwareLatentStateEncoder,
    BranchConditionedRollout,
    FutureTrajectoryEncoder,
    kmeans_torch,
)


class Model(nn.Module):
    """
    BranchWorldModel: branch-aware latent world model for time-series forecasting.

    The model follows four explicit modules:
    1. Branch-aware latent state encoder.
    2. Offline future branch discovery and prototype memory.
    3. Retrieval-conditioned multi-branch latent rollout.
    4. Branch selection / mixture head.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.enc_in = configs.enc_in

        self.latent_dim = getattr(configs, "wm_latent_dim", configs.d_model)
        self.branch_num = getattr(configs, "wm_branch_num", 4)
        self.retrieve_k = getattr(configs, "wm_retrieve_k", self.branch_num)
        self.neighbor_k = getattr(configs, "wm_neighbor_k", 64)
        self.memory_size = getattr(configs, "wm_memory_size", 2048)
        self.kmeans_iters = getattr(configs, "wm_kmeans_iters", 8)
        self.aux_weight = getattr(configs, "wm_aux_weight", 0.1)
        self.diversity_weight = getattr(configs, "wm_diversity_weight", 0.02)
        self.oracle_weight = getattr(configs, "wm_oracle_weight", 0.1)
        self.use_memory = bool(getattr(configs, "wm_use_memory", 1))
        self.use_branch_discovery = bool(getattr(configs, "wm_use_branch_discovery", 1))
        self.use_gating = bool(getattr(configs, "wm_use_gating", 1))
        self.use_aux_losses = bool(getattr(configs, "wm_use_aux_losses", 1))
        self.head_type = getattr(configs, "wm_head_type", "moe")
        self.balance_weight = getattr(configs, "wm_balance_weight", 0.01)
        self.backbone_name = getattr(configs, "wm_backbone", "patch_transformer")
        self.freeze_backbone = bool(getattr(configs, "wm_freeze_backbone", 0))

        self.state_encoder = BranchAwareLatentStateEncoder(
            c_in=configs.enc_in,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            latent_dim=self.latent_dim,
            backbone=self.backbone_name,
            patch_len=getattr(configs, "wm_patch_len", getattr(configs, "patch_len", 16)),
            freeze_backbone=self.freeze_backbone,
        )
        self.future_encoder = FutureTrajectoryEncoder(
            c_in=configs.c_out,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            latent_dim=self.latent_dim,
        )

        self.branch_fuser = nn.Sequential(
            nn.Linear(self.latent_dim * 2 + 1, self.latent_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.rollout = BranchConditionedRollout(self.latent_dim, configs.dropout)
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.c_out),
        )
        self.expert_decoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, configs.d_ff),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.c_out),
            )
            for _ in range(self.branch_num)
        ])
        self.gate = nn.Sequential(
            nn.Linear(self.latent_dim * 3 + 1, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, 1),
        )

        self.fallback_branches = nn.Parameter(torch.randn(self.branch_num, self.latent_dim) * 0.02)
        self._last_aux_loss = None

        self.register_buffer("memory_centers", torch.empty(0, self.latent_dim), persistent=False)
        self.register_buffer("memory_branches", torch.empty(0, self.latent_dim), persistent=False)
        self.register_buffer("memory_ready", torch.tensor(False), persistent=False)

    def _normalize(self, x):
        # x: [B, L, C]. Statistics are detached as in common long-term forecasting baselines.
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev
        return x, means, stdev

    def _denormalize(self, y, means, stdev):
        # y: [B, H, C_out]. The ETT-style benchmarks usually have C_out == enc_in.
        if means.size(-1) != y.size(-1):
            means = means[..., -y.size(-1):]
            stdev = stdev[..., -y.size(-1):]
        return y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    @torch.no_grad()
    def build_memory(self, data_loader, device):
        """
        Offline future branch discovery.

        For each training sample, encode current state z_i and future dynamics u_i.
        Then each anchor state builds a local neighborhood and compresses the
        neighborhood's future dynamics into branch prototypes by k-means.
        """
        if not self.use_memory or self.memory_size <= 0:
            self.memory_ready.fill_(False)
            return

        was_training = self.training
        self.eval()

        states = []
        retrieval_states = []
        traj_codes = []
        for batch_x, batch_y, _, _ in data_loader:
            batch_x = batch_x.float().to(device)
            future_y = batch_y[:, -self.pred_len:, :].float().to(device)
            if future_y.size(-1) != self.c_out:
                future_y = future_y[:, :, -self.c_out:]

            norm_x, means, stdev = self._normalize(batch_x)
            norm_future = (future_y - means[:, :, -future_y.size(-1):]) / stdev[:, :, -future_y.size(-1):]

            state, retrieval_state = self.state_encoder(norm_x)
            _, traj_code = self.future_encoder(norm_future, state)
            states.append(state.detach().cpu())
            retrieval_states.append(retrieval_state.detach().cpu())
            traj_codes.append(F.normalize(traj_code, dim=-1).detach().cpu())

            if sum(x.size(0) for x in states) >= self.memory_size:
                break

        if not states:
            self.memory_ready.fill_(False)
            if was_training:
                self.train()
            return

        states = torch.cat(states, dim=0)[: self.memory_size].to(device)
        retrieval_states = torch.cat(retrieval_states, dim=0)[: self.memory_size].to(device)
        traj_codes = torch.cat(traj_codes, dim=0)[: self.memory_size].to(device)

        sim = retrieval_states @ retrieval_states.t()  # [N, N], state-neighborhood similarity.
        neighbor_k = min(self.neighbor_k, retrieval_states.size(0))
        _, nn_idx = torch.topk(sim, k=neighbor_k, dim=-1)

        if self.use_branch_discovery:
            centers = []
            branches = []
            for anchor_id in range(retrieval_states.size(0)):
                ids = nn_idx[anchor_id]
                local_u = traj_codes[ids]  # [neighbor_k, Z], future dynamics around one state.
                _, assign = kmeans_torch(local_u, self.branch_num, self.kmeans_iters)

                for branch_id in range(self.branch_num):
                    mask = assign == branch_id
                    if not mask.any():
                        continue
                    local_ids = ids[mask]
                    centers.append(retrieval_states[local_ids].mean(dim=0))
                    branches.append(traj_codes[local_ids].mean(dim=0))

            centers = F.normalize(torch.stack(centers, dim=0), dim=-1)
            branches = F.normalize(torch.stack(branches, dim=0), dim=-1)
        else:
            # Raw retrieval ablation: store one future dynamics code per training state.
            centers = retrieval_states
            branches = traj_codes
        if centers.size(0) > self.memory_size:
            ids = torch.linspace(0, centers.size(0) - 1, self.memory_size, device=device).long()
            centers = centers[ids]
            branches = branches[ids]

        self.memory_centers = centers.detach()
        self.memory_branches = branches.detach()
        self.memory_ready.fill_(True)
        if was_training:
            self.train()

    def _retrieve(self, retrieval_state):
        B = retrieval_state.size(0)
        K = min(self.retrieve_k, max(1, self.memory_centers.size(0)))

        if bool(self.memory_ready) and self.memory_centers.numel() > 0:
            sim = retrieval_state @ self.memory_centers.t()  # [B, P]
            score, idx = torch.topk(sim, k=K, dim=-1)
            centers = self.memory_centers[idx]  # [B, K, Z]
            branches = self.memory_branches[idx]  # [B, K, Z]
            score = score.unsqueeze(-1)  # [B, K, 1]
        else:
            K = self.branch_num
            centers = retrieval_state.unsqueeze(1).expand(B, K, self.latent_dim)
            branches = F.normalize(self.fallback_branches, dim=-1).unsqueeze(0).expand(B, K, self.latent_dim)
            score = torch.zeros(B, K, 1, device=retrieval_state.device)

        return centers, branches, score

    def _forecast_normalized(self, x_enc, future_y=None):
        norm_x, means, stdev = self._normalize(x_enc)
        state, retrieval_state = self.state_encoder(norm_x)
        centers, branches, sim_score = self._retrieve(retrieval_state)

        # Branch condition r^(k): [state prototype, dynamics prototype, retrieval score] -> [B, K, Z].
        branch_cond = self.branch_fuser(torch.cat([centers, branches, sim_score], dim=-1))
        latent_paths = self.rollout(state, branch_cond, self.pred_len)  # [B, K, H, Z]

        # Decode each branch independently before mixing, preserving branch-specific futures.
        if self.head_type == "moe":
            expert_outputs = []
            for branch_id in range(latent_paths.size(1)):
                # latent_paths[:, branch_id]: [B, H, Z] -> expert forecast [B, H, C].
                expert = self.expert_decoders[branch_id % len(self.expert_decoders)]
                expert_outputs.append(expert(latent_paths[:, branch_id]))
            branch_pred = torch.stack(expert_outputs, dim=1)  # [B, K, H, C]
        else:
            branch_pred = self.decoder(latent_paths)  # [B, K, H, C]

        gate_state = state.unsqueeze(1).expand_as(branch_cond)
        gate_in = torch.cat([gate_state, centers, branches, sim_score], dim=-1)  # [B, K, 3Z + 1]
        if self.use_gating:
            logits = self.gate(gate_in).squeeze(-1)  # [B, K]
            weights = torch.softmax(logits, dim=-1)
        else:
            weights = torch.softmax(sim_score.squeeze(-1), dim=-1)
        pred = (branch_pred * weights[:, :, None, None]).sum(dim=1)  # [B, H, C]

        self._last_aux_loss = None
        if self.training and self.use_aux_losses and future_y is not None:
            if future_y.size(-1) != self.c_out:
                future_y = future_y[:, :, -self.c_out:]
            norm_future = (future_y - means[:, :, -future_y.size(-1):]) / stdev[:, :, -future_y.size(-1):]
            future_states, _ = self.future_encoder(norm_future, state)

            mixed_latent = (latent_paths * weights[:, :, None, None]).sum(dim=1)  # [B, H, Z]
            latent_loss = F.mse_loss(mixed_latent, future_states.detach())

            branch_error = (branch_pred - norm_future.unsqueeze(1)).pow(2).mean(dim=(2, 3))  # [B, K]
            oracle_loss = branch_error.min(dim=1).values.mean()

            if branch_pred.size(1) > 1:
                flat = branch_pred.flatten(start_dim=2)  # [B, K, H*C]
                pair_dist = torch.cdist(flat, flat, p=2)
                eye = torch.eye(pair_dist.size(1), device=pair_dist.device).bool().unsqueeze(0)
                diversity_loss = torch.exp(-pair_dist.masked_fill(eye, 1e6)).mean()
            else:
                diversity_loss = pred.new_tensor(0.0)

            balance_loss = pred.new_tensor(0.0)
            if self.head_type == "moe" and weights.size(1) > 1:
                # Encourage the router to use multiple branch experts over a batch.
                # mean_usage: [K], ideal usage is approximately uniform.
                mean_usage = weights.mean(dim=0)
                balance_loss = weights.size(1) * torch.sum(mean_usage * mean_usage)

            self._last_aux_loss = (
                self.aux_weight * latent_loss
                + self.oracle_weight * oracle_loss
                + self.diversity_weight * diversity_loss
                + self.balance_weight * balance_loss
            )

        return pred, means, stdev

    def get_auxiliary_loss(self):
        if self._last_aux_loss is None:
            return None
        return self._last_aux_loss

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=None):
        pred, means, stdev = self._forecast_normalized(x_enc, future_y=future_y)
        return self._denormalize(pred, means, stdev)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_y=None):
        if self.task_name not in ["long_term_forecast"]:
            raise NotImplementedError("BranchWorldModel currently supports long_term_forecast.")
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, future_y=future_y)
