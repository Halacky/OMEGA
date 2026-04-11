"""
Multi-Resolution Temporal Consensus Model for EMG Classification.

Processes EMG at 4 temporal resolutions (original, 2x, 4x, 8x downsampled)
with independent CNN encoders, fuses via cross-resolution attention, and
enforces consensus via KL divergence between per-resolution predictions.

Architecture:
    Input (B, C, T) at 2000Hz
    -> Downsample to 4 resolutions: T, T/2, T/4, T/8
    -> Per-resolution CNN encoder: (B, C, T_r) -> (B, D)
    -> Cross-resolution attention fusion: 4 x (B, D) -> (B, D)
    -> Classifier: (B, D) -> (B, num_classes)
    + Per-resolution classifiers for consensus loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


RESOLUTIONS = [1, 2, 4, 8]  # downsample factors


class PerResolutionEncoder(nn.Module):
    """1D-CNN encoder adapted for a specific temporal resolution."""

    def __init__(self, in_channels: int, hidden_dim: int, downsample_factor: int, dropout: float = 0.3):
        super().__init__()
        self.downsample_factor = downsample_factor

        # Adapt kernel sizes based on resolution
        if downsample_factor <= 2:
            k1, k2, k3 = 7, 5, 3
        elif downsample_factor <= 4:
            k1, k2, k3 = 5, 3, 3
        else:
            k1, k2, k3 = 3, 3, 3

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(64, 128, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(128, hidden_dim, kernel_size=k3, padding=k3 // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x (B, C, T_r) -- input at this resolution
        Returns: (B, hidden_dim)
        """
        return self.conv(x).squeeze(-1)


class MultiResolutionEMG(nn.Module):
    """
    Multi-resolution temporal consensus model.

    Args:
        in_channels: Number of EMG channels.
        num_classes: Number of gesture classes.
        dropout: Dropout rate.
        hidden_dim: Feature dimension per resolution.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_resolutions = len(RESOLUTIONS)

        # Per-resolution encoders
        self.encoders = nn.ModuleList([
            PerResolutionEncoder(in_channels, hidden_dim, ds, dropout)
            for ds in RESOLUTIONS
        ])

        # Per-resolution linear classifiers (for consensus loss)
        self.res_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in RESOLUTIONS
        ])

        # Cross-resolution attention fusion
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Learnable resolution weights for final prediction
        self.resolution_logits = nn.Parameter(torch.zeros(self.num_resolutions))

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Store per-resolution logits for consensus loss
        self._per_res_logits: Optional[torch.Tensor] = None

    @staticmethod
    def _downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsample temporal dimension by averaging non-overlapping windows.
        Args: x (B, C, T), factor: downsample factor
        Returns: (B, C, T // factor)
        """
        if factor == 1:
            return x
        B, C, T = x.shape
        # Truncate to multiple of factor
        T_new = (T // factor) * factor
        x = x[:, :, :T_new]
        # Reshape and average
        x = x.reshape(B, C, T_new // factor, factor).mean(dim=3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-resolution processing.
        Args: x (B, C, T) at original resolution
        Returns: logits (B, num_classes)
        """
        B = x.size(0)

        # Encode each resolution
        per_res_features = []
        per_res_logits_list = []

        for i, (encoder, res_clf, ds_factor) in enumerate(
            zip(self.encoders, self.res_classifiers, RESOLUTIONS)
        ):
            x_ds = self._downsample(x, ds_factor)        # (B, C, T/ds)
            feat = encoder(x_ds)                           # (B, D)
            per_res_features.append(feat)
            per_res_logits_list.append(res_clf(feat))     # (B, num_classes)

        # Stack for attention: (B, num_resolutions, D)
        stacked = torch.stack(per_res_features, dim=1)

        # Cross-resolution attention
        attn_out, _ = self.attn(stacked, stacked, stacked)
        attn_out = self.attn_norm(attn_out + stacked)

        # Learnable resolution weights
        res_weights = F.softmax(self.resolution_logits, dim=0)

        # Weighted combination: (B, D)
        fused = (attn_out * res_weights.view(1, -1, 1)).sum(dim=1)

        # Final classification
        logits = self.classifier(fused)

        # Store per-resolution logits for consensus loss
        self._per_res_logits = torch.stack(per_res_logits_list, dim=1)

        return logits

    def get_per_resolution_logits(self) -> Optional[torch.Tensor]:
        """
        Per-resolution logits from last forward pass.
        Returns: (B, num_resolutions, num_classes) or None
        """
        return self._per_res_logits

    def get_consensus_loss(self, target_logits: torch.Tensor) -> torch.Tensor:
        """
        Consensus loss: KL divergence between each resolution's
        prediction and the ensemble prediction.

        Args: target_logits (B, num_classes) -- ensemble logits from forward()
        Returns: scalar consensus loss
        """
        if self._per_res_logits is None:
            return torch.tensor(0.0, device=target_logits.device)

        target_probs = F.softmax(target_logits.detach(), dim=-1)
        total_kl = torch.tensor(0.0, device=target_logits.device)
        num_res = self._per_res_logits.size(1)

        for i in range(num_res):
            res_log_probs = F.log_softmax(self._per_res_logits[:, i, :], dim=-1)
            kl = F.kl_div(res_log_probs, target_probs, reduction="batchmean")
            total_kl = total_kl + kl

        return total_kl / num_res
