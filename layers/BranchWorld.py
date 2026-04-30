import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosition(nn.Module):
    """Fixed positional encoding for latent state and future trajectory encoders."""

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length):
        return self.pe[:, :length]


class BranchAwareLatentStateEncoder(nn.Module):
    """
    Encodes the observed window into two latents:
    - state: transition-aware latent used by the rollout dynamics.
    - retrieval_state: L2-normalized projection used by branch memory retrieval.
    """

    def __init__(
        self,
        c_in,
        seq_len,
        d_model,
        n_heads,
        d_ff,
        e_layers,
        dropout,
        latent_dim,
        backbone="patch_transformer",
        patch_len=16,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = StateBackbone(
            c_in=c_in,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            backbone=backbone,
            patch_len=patch_len,
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.retrieval_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: [B, L, C], where L is look-back length and C is number of variables.
        pooled = self.backbone(x)  # [B, D], state summary from the selected backbone.
        state = self.state_head(pooled)  # [B, Z]
        retrieval_state = F.normalize(self.retrieval_head(state), dim=-1)  # [B, Z]
        return state, retrieval_state


class StateBackbone(nn.Module):
    """Backbone zoo for BranchWorld state encoding."""

    def __init__(self, c_in, seq_len, d_model, n_heads, d_ff, e_layers, dropout, backbone, patch_len):
        super().__init__()
        self.backbone = backbone
        self.seq_len = seq_len
        self.patch_len = max(1, min(patch_len, seq_len))

        if backbone == "temporal_transformer":
            self.value_proj = nn.Linear(c_in, d_model)
            self.position = SinusoidalPosition(d_model)
            self.encoder = _make_transformer_encoder(d_model, n_heads, d_ff, e_layers, dropout)
        elif backbone == "patch_transformer":
            self.patch_proj = nn.Linear(c_in * self.patch_len, d_model)
            self.position = SinusoidalPosition(d_model)
            self.encoder = _make_transformer_encoder(d_model, n_heads, d_ff, e_layers, dropout)
        elif backbone == "inverted_transformer":
            # iTransformer-style: each variable becomes a token whose features are the full look-back sequence.
            self.value_proj = nn.Linear(seq_len, d_model)
            self.var_embedding = nn.Parameter(torch.randn(1, c_in, d_model) * 0.02)
            self.encoder = _make_transformer_encoder(d_model, n_heads, d_ff, e_layers, dropout)
        elif backbone == "tcn":
            layers = []
            in_ch = c_in
            for layer_id in range(max(1, e_layers)):
                dilation = 2 ** layer_id
                layers.extend([
                    nn.Conv1d(in_ch, d_model, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_ch = d_model
            self.net = nn.Sequential(*layers)
            self.norm = nn.LayerNorm(d_model)
        elif backbone == "mlp":
            self.net = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(seq_len * c_in, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            raise ValueError(
                f"Unknown wm_backbone={backbone}. Choose from "
                "temporal_transformer, patch_transformer, inverted_transformer, tcn, mlp."
            )

    def forward(self, x):
        # x: [B, L, C].
        if self.backbone == "temporal_transformer":
            h = self.value_proj(x) + self.position(x.size(1)).to(x.device)
            h = self.encoder(h)  # [B, L, D]
            return h.mean(dim=1)  # [B, D]

        if self.backbone == "patch_transformer":
            B, L, C = x.shape
            pad_len = (self.patch_len - L % self.patch_len) % self.patch_len
            if pad_len > 0:
                # Pad on the right so reshape below forms complete non-overlapping patches.
                x = F.pad(x, (0, 0, 0, pad_len))
            # [B, L_pad, C] -> [B, P, patch_len, C] -> [B, P, patch_len*C].
            patches = x.reshape(B, -1, self.patch_len, C).flatten(start_dim=2)
            h = self.patch_proj(patches) + self.position(patches.size(1)).to(x.device)
            h = self.encoder(h)  # [B, P, D]
            return h.mean(dim=1)  # [B, D]

        if self.backbone == "inverted_transformer":
            # [B, L, C] -> [B, C, L], then variables are treated as tokens.
            h = self.value_proj(x.permute(0, 2, 1)) + self.var_embedding
            h = self.encoder(h)  # [B, C, D]
            return h.mean(dim=1)  # [B, D]

        if self.backbone == "tcn":
            # Conv1d expects [B, C, L]; crop to original length after dilated padding.
            h = self.net(x.permute(0, 2, 1))
            h = h[..., : x.size(1)].permute(0, 2, 1)  # [B, L, D]
            return self.norm(h.mean(dim=1))  # [B, D]

        return self.net(x)  # [B, D]


def _make_transformer_encoder(d_model, n_heads, d_ff, e_layers, dropout):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=e_layers)


class FutureTrajectoryEncoder(nn.Module):
    """
    Encodes the future horizon as latent states and a compact dynamics code.
    The dynamics code is used for offline future branch discovery.
    """

    def __init__(self, c_in, d_model, n_heads, d_ff, e_layers, dropout, latent_dim):
        super().__init__()
        self.value_proj = nn.Linear(c_in, d_model)
        self.position = SinusoidalPosition(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, e_layers // 2))
        self.step_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )
        self.traj_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, future_y, state):
        # future_y: [B, H, C], state: [B, Z].
        h = self.value_proj(future_y) + self.position(future_y.size(1)).to(future_y.device)
        h = self.encoder(h)  # [B, H, D]
        future_states = self.step_head(h)  # [B, H, Z]

        # Delta trajectory anchors the future evolution at the current world state.
        delta = future_states - state.unsqueeze(1)  # [B, H, Z]
        traj_code = self.traj_head(delta.mean(dim=1))  # [B, Z]
        return future_states, traj_code


class BranchConditionedRollout(nn.Module):
    """Recurrent latent transition f(z, r) conditioned on one branch prior."""

    def __init__(self, latent_dim, dropout):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, state, branch_cond, horizon):
        # state: [B, Z], branch_cond: [B, K, Z].
        B, K, Z = branch_cond.shape
        cur = state.unsqueeze(1).expand(B, K, Z)  # [B, K, Z], one latent per branch.
        steps = []
        for _ in range(horizon):
            trans_in = torch.cat([cur, branch_cond], dim=-1)  # [B, K, 2Z]
            delta = self.transition(trans_in)  # [B, K, Z]
            cur = self.norm(cur + delta)
            steps.append(cur)
        return torch.stack(steps, dim=2)  # [B, K, H, Z]


def kmeans_torch(x, num_clusters, num_iters=8):
    """
    Small deterministic k-means used during offline branch discovery.
    x: [N, Z], returns centers [M, Z] and assignments [N].
    """
    if x.size(0) <= num_clusters:
        pad = num_clusters - x.size(0)
        centers = x
        if pad > 0:
            centers = torch.cat([centers, x[-1:].expand(pad, -1)], dim=0)
        assign = torch.arange(x.size(0), device=x.device).clamp(max=num_clusters - 1)
        return centers, assign

    init_ids = torch.linspace(0, x.size(0) - 1, steps=num_clusters, device=x.device).long()
    centers = x[init_ids].clone()
    assign = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    for _ in range(num_iters):
        dist = torch.cdist(x, centers, p=2)  # [N, M]
        assign = dist.argmin(dim=1)
        new_centers = []
        for cluster_id in range(num_clusters):
            mask = assign == cluster_id
            if mask.any():
                new_centers.append(x[mask].mean(dim=0))
            else:
                new_centers.append(centers[cluster_id])
        centers = torch.stack(new_centers, dim=0)
    return centers, assign
