"""
Test-Time Training (TTT) Content-Style EMG Model with Masked Channel SSL.

Combines content-style disentanglement with a self-supervised masked
channel reconstruction task. At test time, the SSL task is used to
adapt the encoder to the test subject's signal distribution WITHOUT
using any gesture labels.

Architecture:
    Input (B, C, T) -> CNN-GRU Encoder (shared) -> features (B, hidden_dim)
      |-- Content projection -> z_content -> Gesture Classifier -> logits
      |-- Style projection -> z_style (adversarial, not used at test time)
      |-- SSL head: Masked Channel Reconstruction -> reconstructed channels

Training:
    L = L_gesture + lambda_ssl * L_ssl
    L_ssl = MSE on masked channels only

Test-Time Training:
    1. Save encoder state
    2. For each test subject: N SGD steps on L_ssl only (no labels)
    3. Classify with adapted encoder + frozen classifier
    4. Restore encoder state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class MaskedChannelReconHead(nn.Module):
    """
    SSL head for masked channel reconstruction.

    Given encoder features (B, D), predict a coarse temporal representation
    of ALL channels: (B, C, t_bins) where t_bins << T.
    """

    def __init__(self, hidden_dim: int, n_channels: int, t_bins: int = 16):
        super().__init__()
        self.n_channels = n_channels
        self.t_bins = t_bins

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_channels * t_bins),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args: features (B, hidden_dim)
        Returns: (B, n_channels, t_bins)
        """
        out = self.net(features)
        return out.reshape(-1, self.n_channels, self.t_bins)


class TTTContentStyleEMG(nn.Module):
    """
    Test-Time Training model with content-style disentanglement and
    masked channel reconstruction SSL.

    Args:
        in_channels: Number of EMG channels (C).
        num_classes: Number of gesture classes.
        dropout: Dropout rate.
        hidden_dim: Encoder output dimension.
        mask_ratio: Fraction of channels to mask for SSL.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        hidden_dim: int = 128,
        mask_ratio: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio

        # ===== Shared CNN-GRU Encoder =====
        self.encoder = nn.Sequential(
            # CNN blocks
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ===== Content projection -> gesture classification =====
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gesture_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # ===== Style projection (for disentanglement) =====
        self.style_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # ===== SSL: Masked Channel Reconstruction head =====
        self.ssl_recon_head = MaskedChannelReconHead(
            hidden_dim=hidden_dim,
            n_channels=in_channels,
            t_bins=16,
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input through CNN.
        Args: x (B, C, T)
        Returns: features (B, hidden_dim)
        """
        return self.encoder(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward: encode -> content -> classify.
        Args: x (B, C, T)
        Returns: logits (B, num_classes)
        """
        features = self._encode(x)
        content = self.content_proj(features)
        logits = self.gesture_classifier(content)
        return logits

    def forward_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward returning all components for training.
        Returns dict with: logits, features, content, style
        """
        features = self._encode(x)
        content = self.content_proj(features)
        style = self.style_proj(features)
        logits = self.gesture_classifier(content)
        return {
            "logits": logits,
            "features": features,
            "content": content,
            "style": style,
        }

    # -------- SSL: Masked Channel Reconstruction --------

    def _create_channel_mask(self, B: int, C: int, device: torch.device) -> torch.Tensor:
        """
        Create binary mask: 1=keep, 0=mask.
        At least 1 channel kept and at least 1 masked.
        Returns: (B, C, 1)
        """
        n_mask = max(1, int(C * self.mask_ratio))
        n_mask = min(n_mask, C - 1)

        mask = torch.ones(B, C, device=device)
        for i in range(B):
            indices = torch.randperm(C, device=device)[:n_mask]
            mask[i, indices] = 0.0

        return mask.unsqueeze(-1)  # (B, C, 1)

    def ssl_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Self-supervised loss: masked channel reconstruction.

        1. Create random channel mask
        2. Zero out masked channels
        3. Encode corrupted input
        4. Predict coarse temporal representation of ALL channels
        5. MSE on MASKED channels only

        Args: x (B, C, T)
        Returns: scalar MSE loss
        """
        B, C, T = x.shape

        # Target: coarse temporal binning of original signal
        t_bins = self.ssl_recon_head.t_bins
        with torch.no_grad():
            target = F.adaptive_avg_pool1d(x, t_bins)  # (B, C, t_bins)

        # Mask channels
        mask = self._create_channel_mask(B, C, x.device)  # (B, C, 1)
        x_masked = x * mask

        # Encode corrupted input
        features = self._encode(x_masked)  # (B, hidden_dim)

        # Reconstruct
        recon = self.ssl_recon_head(features)  # (B, C, t_bins)

        # MSE on masked channels only
        inv_mask = (1.0 - mask)  # (B, C, 1)
        inv_mask_expanded = inv_mask.expand_as(target)

        n_masked = inv_mask_expanded.sum().clamp(min=1.0)
        mse = ((recon - target) ** 2 * inv_mask_expanded).sum() / n_masked

        return mse

    # -------- Parameter groups for TTT --------

    def get_encoder_params(self) -> List[nn.Parameter]:
        """Parameters updated during test-time training (encoder + SSL head)."""
        params = list(self.encoder.parameters())
        params += list(self.ssl_recon_head.parameters())
        return params

    def get_classifier_params(self) -> List[nn.Parameter]:
        """Parameters frozen during TTT (content + classifier + style)."""
        params = list(self.content_proj.parameters())
        params += list(self.gesture_classifier.parameters())
        params += list(self.style_proj.parameters())
        return params
