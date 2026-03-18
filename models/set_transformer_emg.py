"""
Channel-Permutation Equivariant Set Transformer for EMG Classification.

Treats EMG channels as an unordered set. Each channel is independently
encoded via a shared temporal CNN, then aggregated using Set Transformer
(ISAB + PMA) which is inherently permutation equivariant/invariant.

Architecture:
    Input (B, C, T)
    -> Per-channel shared temporal CNN: (B*C, 1, T) -> (B*C, D) -> (B, C, D)
    -> ISAB layers (permutation equivariant): (B, C, D) -> (B, C, D)
    -> PMA (permutation invariant pooling): (B, C, D) -> (B, 1, D)
    -> Classifier MLP: (B, D) -> (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MAB(nn.Module):
    """Multi-head Attention Block: Attention(Q, K, V) + residual + FF."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Args: Q (B, N, D), K (B, M, D)
        Returns: (B, N, D)
        """
        h, _ = self.attn(Q, K, K)
        h = self.norm1(Q + h)
        h = self.norm2(h + self.ff(h))
        return h


class ISAB(nn.Module):
    """
    Induced Set Attention Block.

    Uses M learnable inducing points for O(N*M) complexity
    instead of O(N^2) full self-attention.
    """

    def __init__(self, dim: int, n_heads: int, n_inducing: int, dropout: float = 0.1):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing, dim) * 0.02)
        self.mab1 = MAB(dim, n_heads, dropout)  # inducing <- set
        self.mab2 = MAB(dim, n_heads, dropout)  # set <- inducing

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args: X (B, N, D) -- set of N elements
        Returns: (B, N, D) -- permutation equivariant
        """
        B = X.size(0)
        I = self.inducing_points.expand(B, -1, -1)
        H = self.mab1(I, X)      # (B, M, D) -- inducing attends to set
        return self.mab2(X, H)    # (B, N, D) -- set attends to inducing


class PMA(nn.Module):
    """
    Pooling by Multi-head Attention.

    Uses K learnable seed vectors to pool a set of arbitrary size
    into K fixed-size output vectors.
    """

    def __init__(self, dim: int, n_heads: int, n_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, dim) * 0.02)
        self.mab = MAB(dim, n_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args: X (B, N, D) -- set of N elements
        Returns: (B, K, D) -- K pooled vectors (permutation invariant)
        """
        B = X.size(0)
        S = self.seeds.expand(B, -1, -1)
        return self.mab(S, X)


class SetTransformerEMG(nn.Module):
    """
    Set Transformer for EMG: treats channels as an unordered set.

    Args:
        in_channels: Number of EMG channels (the set size).
        num_classes: Number of gesture classes.
        dropout: Dropout rate.
        channel_embed_dim: Dimension of per-channel temporal embedding.
        n_heads: Number of attention heads in ISAB/PMA.
        n_isab: Number of ISAB layers.
        n_inducing: Number of inducing points in ISAB.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        channel_embed_dim: int = 128,
        n_heads: int = 4,
        n_isab: int = 2,
        n_inducing: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channel_embed_dim = channel_embed_dim

        # ===== Shared per-channel temporal encoder =====
        # Processes each channel independently: (B*C, 1, T) -> (B*C, D)
        self.temporal_encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, channel_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(channel_embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B*C, D, 1)
        )

        # ===== ISAB layers (permutation equivariant) =====
        self.isab_layers = nn.ModuleList([
            ISAB(channel_embed_dim, n_heads, n_inducing, dropout)
            for _ in range(n_isab)
        ])

        # ===== PMA (permutation invariant pooling) =====
        self.pma = PMA(channel_embed_dim, n_heads, n_seeds=1, dropout=dropout)

        # ===== Classifier MLP =====
        self.classifier = nn.Sequential(
            nn.Linear(channel_embed_dim, channel_embed_dim),
            nn.LayerNorm(channel_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_embed_dim, channel_embed_dim // 2),
            nn.LayerNorm(channel_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(channel_embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args: x (B, C, T) -- batch of EMG windows (Conv1d format)
        Returns: logits (B, num_classes)
        """
        B, C, T = x.shape

        # Per-channel temporal encoding: (B*C, 1, T) -> (B*C, D, 1)
        x_flat = x.reshape(B * C, 1, T)
        h = self.temporal_encoder(x_flat)        # (B*C, D, 1)
        h = h.squeeze(-1)                         # (B*C, D)
        h = h.reshape(B, C, self.channel_embed_dim)  # (B, C, D)

        # Set attention (permutation equivariant)
        for isab in self.isab_layers:
            h = isab(h)                            # (B, C, D)

        # Pool set to fixed vector (permutation invariant)
        pooled = self.pma(h).squeeze(1)            # (B, D)

        # Classify
        logits = self.classifier(pooled)           # (B, num_classes)
        return logits
