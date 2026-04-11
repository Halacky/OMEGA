"""
Causal Disentangled CNN-GRU with Barlow Twins + Reconstruction (Hypothesis H_causal).

Replaces distance correlation (exp_31) with:
- Barlow Twins redundancy reduction: removes linear cross-correlations between
  z_content and z_style, but allows nonlinear causal dependencies to persist.
  Complexity O(D_content * D_style) vs O(B^2) for distance correlation.
- Causal consistency loss: for same-gesture, different-subject pairs,
  minimizes ||z_content(x_i) - z_content(x_j)||, forcing z_content toward
  subject-invariant causal factors.
- Reconstruction: decoder(z_content, z_style) -> x_hat. Prevents information
  loss from over-aggressive disentanglement.
- GroupDRO on gesture loss (from exp_57) for worst-case subject robustness.

Architecture:
    Input (B, C, T) -> SharedEncoder (CNN+BiGRU+Attention) -> (B, 256)
      ├─ ContentHead  -> z_content (B, content_dim) -> GestureClassifier
      ├─ StyleHead    -> z_style   (B, style_dim)   -> SubjectClassifier
      └─ Decoder(z_content || z_style)               -> x_hat (B, C, T)

Loss:
    L = L_gesture_dro + α·L_subject + β_bt·L_barlow + β_cc·L_causal + δ·L_recon

References:
    - CDDG (Neural Networks 2024): structural causal model for time-series
    - Barlow Twins (ICML 2021): redundancy reduction without mode collapse
    - Sagawa et al. 2020: GroupDRO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.disentangled_cnn_gru import SharedEncoder, ProjectionHead


# ─────────────────────────── Loss functions ────────────────────────────


def barlow_twins_cross_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    """
    Barlow Twins redundancy reduction between z_content and z_style.

    Normalizes both representations along the batch dimension, computes
    the cross-correlation matrix C (D1 x D2), and penalizes all entries.
    Since z_content and z_style should encode independent factors,
    ALL cross-correlations should be zero.

    Complexity: O(B * D1 * D2) — linear in batch size.

    Args:
        z1: (B, D1) content representations
        z2: (B, D2) style representations

    Returns:
        Scalar loss = mean(C^2).
    """
    B = z1.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    # Batch-normalize: mean=0, std=1 per feature
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)

    # Cross-correlation matrix: (D1, D2)
    C = z1_norm.T @ z2_norm / B

    # All entries should be 0 (independence)
    loss = (C ** 2).mean()
    return loss


def causal_consistency_loss(
    z_content: torch.Tensor,
    gesture_labels: torch.Tensor,
    subject_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Causal consistency: same gesture, different subjects → similar z_content.

    For each gesture class, computes per-subject centroids of z_content,
    then minimizes pairwise L2 distances between centroids.  Using centroids
    rather than all O(N^2) sample pairs is more stable and gradient-friendly.

    All pairs come from training subjects only (test subject is never in batch).

    Args:
        z_content: (B, D) content embeddings
        gesture_labels: (B,) integer gesture class indices
        subject_labels: (B,) integer subject indices (train subjects only)

    Returns:
        Scalar loss.
    """
    total_loss = torch.tensor(0.0, device=z_content.device)
    num_valid = 0

    for g in gesture_labels.unique():
        mask_g = gesture_labels == g
        z_g = z_content[mask_g]
        s_g = subject_labels[mask_g]

        unique_subjects = s_g.unique()
        if len(unique_subjects) < 2:
            continue

        # Per-subject centroids for this gesture
        centroids = []
        for s in unique_subjects:
            centroids.append(z_g[s_g == s].mean(dim=0))
        centroids = torch.stack(centroids)  # (S_g, D)

        # Pairwise L2 distances between centroids
        dists = torch.cdist(centroids, centroids, p=2)  # (S_g, S_g)

        # Mean of upper-triangle (unique pairs)
        n_s = len(centroids)
        triu_mask = torch.triu(
            torch.ones(n_s, n_s, device=z_content.device, dtype=torch.bool),
            diagonal=1,
        )
        if triu_mask.any():
            total_loss = total_loss + dists[triu_mask].mean()
            num_valid += 1

    if num_valid == 0:
        return torch.tensor(0.0, device=z_content.device, requires_grad=True)
    return total_loss / num_valid


# ────────────────────────── Model components ───────────────────────────


class SignalDecoder(nn.Module):
    """
    Convolutional decoder: z -> reconstructed signal (B, C, T).

    Mirrors the encoder's downsampling (3× MaxPool(2) = 8× reduction)
    with 3× ConvTranspose1d (stride=2) for upsampling.
    """

    def __init__(
        self,
        z_dim: int,
        out_channels: int,
        out_length: int,
        init_channels: int = 64,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.out_length = out_length
        self.init_channels = init_channels

        # Compute initial spatial size (matches encoder's 3× MaxPool(2))
        self.init_length = out_length // 8
        if self.init_length < 1:
            self.init_length = 1

        self.fc = nn.Sequential(
            nn.Linear(z_dim, init_channels * self.init_length),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                init_channels, init_channels // 2,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm1d(init_channels // 2),
            nn.ReLU(),
            nn.ConvTranspose1d(
                init_channels // 2, init_channels // 4,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm1d(init_channels // 4),
            nn.ReLU(),
            nn.ConvTranspose1d(
                init_channels // 4, out_channels,
                kernel_size=4, stride=2, padding=1,
            ),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, z_dim)
        Returns:
            (B, out_channels, out_length)
        """
        h = self.fc(z)                                          # (B, C0 * L0)
        h = h.view(-1, self.init_channels, self.init_length)    # (B, C0, L0)
        h = self.deconv(h)                                      # (B, out_ch, ~out_len)

        # Trim or pad to exact output length
        if h.size(2) > self.out_length:
            h = h[:, :, :self.out_length]
        elif h.size(2) < self.out_length:
            h = F.pad(h, (0, self.out_length - h.size(2)))
        return h


# ──────────────────────────── Main model ───────────────────────────────


class CausalDisentangledCNNGRU(nn.Module):
    """
    Causal Content-Style Disentangled CNN-GRU.

    Extends DisentangledCNNGRU with:
    - SignalDecoder for reconstruction regularization
    - Designed for Barlow Twins + causal consistency losses (in trainer)

    Training: returns dict with gesture_logits, subject_logits,
              z_content, z_style, x_recon.
    Eval:     returns gesture_logits tensor only (no subject info needed).
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_gestures: int = 10,
        num_subjects: int = 4,
        content_dim: int = 128,
        style_dim: int = 64,
        window_size: int = 600,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim

        self.encoder = SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # 256

        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head = ProjectionHead(shared_dim, style_dim, dropout)

        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

        self.decoder = SignalDecoder(
            z_dim=content_dim + style_dim,
            out_channels=in_channels,
            out_length=window_size,
        )

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x:          (B, C, T) input signal
            return_all: force returning full dict even in eval mode
        Returns:
            training / return_all:
                dict with gesture_logits, subject_logits, z_content, z_style, x_recon
            eval (default):
                gesture_logits (B, num_gestures)
        """
        shared = self.encoder(x)                          # (B, 256)
        z_content = self.content_head(shared)              # (B, content_dim)
        z_style = self.style_head(shared)                  # (B, style_dim)

        gesture_logits = self.gesture_classifier(z_content)

        if self.training or return_all:
            subject_logits = self.subject_classifier(z_style)
            z_combined = torch.cat([z_content, z_style], dim=1)
            x_recon = self.decoder(z_combined)
            return {
                "gesture_logits": gesture_logits,
                "subject_logits": subject_logits,
                "z_content": z_content,
                "z_style": z_style,
                "x_recon": x_recon,
            }
        return gesture_logits
