import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosition(nn.Module):
    """Fixed positional encoding used by the latent state encoder."""

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


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        weight = torch.softmax(self.score(x), dim=1)
        return (x * weight).sum(dim=1)


class PreNormMLP(nn.Module):
    """Small pre-norm residual MLP block."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class StateBackbone(nn.Module):
    def __init__(
        self,
        c_in,
        seq_len,
        d_model,
        n_heads,
        d_ff,
        e_layers,
        dropout,
        backbone="patch_transformer",
        patch_len=16,
        patch_stride=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.seq_len = seq_len
        self.patch_len = max(1, min(patch_len, seq_len))
        self.patch_stride = max(1, patch_stride or max(1, self.patch_len // 2))

        if backbone == "temporal_transformer":
            self.value_proj = nn.Linear(c_in, d_model)
            self.position = SinusoidalPosition(d_model)
            self.encoder = make_transformer_encoder(d_model, n_heads, d_ff, e_layers, dropout)
            self.pool = AttentionPool(d_model)
        elif backbone == "patch_transformer":
            self.patch_proj = nn.Linear(c_in * self.patch_len, d_model)
            self.position = SinusoidalPosition(d_model)
            self.encoder = make_transformer_encoder(d_model, n_heads, d_ff, e_layers, dropout)
            self.pool = AttentionPool(d_model)
        elif backbone == "inverted_transformer":
            self.value_proj = nn.Linear(seq_len, d_model)
            self.var_embedding = nn.Parameter(torch.randn(1, c_in, d_model) * 0.02)
            self.encoder = make_transformer_encoder(d_model, n_heads, d_ff, e_layers, dropout)
            self.pool = AttentionPool(d_model)
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
                "Unknown wm_backbone={}. Choose from temporal_transformer, "
                "patch_transformer, inverted_transformer, tcn, mlp.".format(backbone)
            )

    def forward(self, x):
        if self.backbone == "temporal_transformer":
            h = self.value_proj(x) + self.position(x.size(1)).to(x.device)
            return self.pool(self.encoder(h))

        if self.backbone == "patch_transformer":
            B, L, C = x.shape
            if L < self.patch_len:
                x = F.pad(x, (0, 0, 0, self.patch_len - L))
                L = x.size(1)
            remainder = (L - self.patch_len) % self.patch_stride
            if remainder != 0:
                x = F.pad(x, (0, 0, 0, self.patch_stride - remainder))
            patches = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
            patches = patches.permute(0, 2, 1, 3).flatten(start_dim=2)
            h = self.patch_proj(patches) + self.position(patches.size(1)).to(x.device)
            return self.pool(self.encoder(h))

        if self.backbone == "inverted_transformer":
            h = self.value_proj(x.permute(0, 2, 1)) + self.var_embedding
            return self.pool(self.encoder(h))

        if self.backbone == "tcn":
            h = self.net(x.permute(0, 2, 1))
            h = h[..., : x.size(1)].permute(0, 2, 1)
            return self.norm(h.mean(dim=1))

        return self.net(x)


class LatentStateEncoder(nn.Module):
    """
    Encodes a window into z_i^0. The normalized projection is used as the
    memory key, while the unnormalized state is kept for decoding.
    state = s_i: 用于 decoder / rollout / trajectory construction
    key   = k_i: 用于 memory retrieval
    """
    def __init__(self, c_in, seq_len, latent_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(c_in, latent_dim, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(latent_dim),   # state 尺度稳定
        )
        self.key_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        # x: (B, seq_len, c_in)
        x = x.permute(0, 2, 1).contiguous()
        state = self.head(self.encoder(x))
        key = F.normalize(self.key_head(state), dim=-1)
        return state, key


class MemoryBranchDecoder(nn.Module):
    def __init__(self, latent_dim, num_horizons, pred_len, c_out, hidden_dim, dropout):
        super().__init__()
        in_dim = latent_dim * (num_horizons + 1)
        self.pred_len = pred_len
        self.c_out = c_out
        self.input = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
        )
        self.blocks = nn.Sequential(
            PreNormMLP(hidden_dim, hidden_dim * 2, dropout),
            PreNormMLP(hidden_dim, hidden_dim * 2, dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, pred_len * c_out)

    def forward(self, state, prototypes):
        B, M, S, Z = prototypes.shape
        state_ctx = state.unsqueeze(1).expand(B, M, Z)
        future_states = state_ctx.unsqueeze(2) + prototypes
        x = torch.cat([state_ctx, future_states.flatten(start_dim=2)], dim=-1)
        h = self.blocks(self.input(x))
        y = self.out(self.out_norm(h))
        return y.view(B, M, self.pred_len, self.c_out)


def make_transformer_encoder(d_model, n_heads, d_ff, e_layers, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=max(1, e_layers))


def kmeans_torch(x, num_clusters, num_iters=8):
    if x.size(0) == 0:
        raise ValueError("kmeans_torch received an empty tensor.")
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
    for _ in range(max(1, num_iters)):
        dist = torch.cdist(x, centers, p=2)
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
