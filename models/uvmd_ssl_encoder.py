"""
Shared UVMD + Per-Band Encoder backbone for SSL pretraining and fine-tuning.

This module provides the common encoder architecture used across all
frequency-aware SSL experiments (H1s–H9s).  It wraps:
  1. UVMDBlock — learnable K-mode frequency decomposition
  2. K independent per-band CNN encoders (GroupNorm variant)
  3. Optional per-band MixStyle augmentation (training only)

The encoder outputs can be consumed by:
  - SSL heads (VICReg, MAE, CPC, contrastive)  during pretraining
  - Linear / MLP classifier head                during fine-tuning

Architecture
────────────
  Raw EMG  (B, T, C)
      │
  UVMDBlock  [K=4 modes, L=8 unrolled ADMM]
      │
  [Optional PerBandMixStyle — training only]
      │
  K × PerBandCNNEncoder  →  per-band features  (B, K, feat_dim)
      │
  Concatenate  →  (B, K·feat_dim)

Design choices for SSL compatibility
─────────────────────────────────────
  ✓  GroupNorm instead of BatchNorm — per-sample normalisation avoids
     batch-composition dependence (critical for contrastive pairs).
  ✓  All UVMD parameters are global (no subject-specific state).
  ✓  PerBandMixStyle disabled at eval() — clean inference path.
  ✓  Encoder weights are identical for supervised and SSL pipelines,
     enabling direct weight transfer from pretrain to finetune.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.uvmd_classifier import UVMDBlock


# ═════════════════════════════════════════════════════════════════════════════
# Per-Band MixStyle (reused from H7, training only)
# ═════════════════════════════════════════════════════════════════════════════

class PerBandMixStyle(nn.Module):
    """Per-band instance statistics mixing (training only, zero parameters)."""

    def __init__(self, K: int, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.K = K
        self.p = p
        self.alpha = alpha

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """(B, K, T, C) → (B, K, T, C)"""
        if not self.training or torch.rand(1).item() > self.p:
            return modes
        B, K, T, C = modes.shape
        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)

        out_bands = []
        for k in range(K):
            x_k = modes[:, k]                                    # (B, T, C)
            mu = x_k.mean(dim=1, keepdim=True)                   # (B, 1, C)
            sigma = x_k.std(dim=1, keepdim=True) + 1e-6          # (B, 1, C)
            x_normed = (x_k - mu) / sigma

            perm = torch.randperm(B, device=x_k.device)
            lam = beta_dist.sample((B, 1, 1)).to(x_k.device)
            mu_mix = lam * mu + (1 - lam) * mu[perm]
            sigma_mix = lam * sigma + (1 - lam) * sigma[perm]

            out_bands.append(x_normed * sigma_mix + mu_mix)

        return torch.stack(out_bands, dim=1)


# ═════════════════════════════════════════════════════════════════════════════
# Per-Band CNN Encoder (GroupNorm variant for SSL compatibility)
# ═════════════════════════════════════════════════════════════════════════════

class PerBandCNNEncoder(nn.Module):
    """
    K parallel 3-layer Conv1d encoders with GroupNorm.

    Each encoder processes one frequency mode independently:
      Conv1d(C → 32, k=7) → GN → ReLU →
      Conv1d(32 → 64, k=5) → GN → ReLU →
      Conv1d(64 → feat_dim, k=3) → GN → ReLU →
      AdaptiveAvgPool1d(1) → Flatten → (B, feat_dim)

    GroupNorm (groups=1 = LayerNorm-like, or groups=8/16) is used instead
    of BatchNorm for two reasons:
      1. SSL contrastive methods form non-i.i.d. batches (augmented views)
         where BN statistics are biased.
      2. Per-sample normalisation ensures LOSO safety — test-subject
         statistics never leak through running mean/var.
    """

    def __init__(
        self,
        K: int = 4,
        in_channels: int = 12,
        feat_dim: int = 64,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.K = K
        self.feat_dim = feat_dim
        self.encoders = nn.ModuleList([
            self._make_encoder(in_channels, feat_dim, gn_groups)
            for _ in range(K)
        ])

    @staticmethod
    def _make_encoder(
        in_ch: int, feat_dim: int, gn_groups: int,
    ) -> nn.Sequential:
        def _gn(ch: int) -> nn.GroupNorm:
            g = min(gn_groups, ch)
            # Ensure ch is divisible by g
            while ch % g != 0 and g > 1:
                g -= 1
            return nn.GroupNorm(g, ch)

        return nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            _gn(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, feat_dim, kernel_size=3, padding=1),
            _gn(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, modes: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        modes : (B, K, T, C)

        Returns
        -------
        per_band : list of K tensors, each (B, feat_dim)
        """
        return [
            self.encoders[k](modes[:, k].permute(0, 2, 1))
            for k in range(self.K)
        ]


# ═════════════════════════════════════════════════════════════════════════════
# UVMD SSL Encoder — Shared Backbone
# ═════════════════════════════════════════════════════════════════════════════

class UVMDSSLEncoder(nn.Module):
    """
    Shared UVMD + PerBandCNN encoder for SSL and supervised pipelines.

    Used as the backbone in:
      - FreqAwareVICReg (pretrain)
      - FreqAwareMAE (pretrain)
      - FreqAwareCPC (pretrain)
      - FreqSelectiveContrastive (pretrain)
      - UVMDSSLClassifier (finetune)

    Parameters
    ----------
    K : int
        Number of UVMD modes (frequency bands).
    L : int
        Number of unrolled ADMM iterations.
    in_channels : int
        Number of EMG channels (12 for NinaPro DB2).
    feat_dim : int
        Output dimension per band encoder.
    use_mixstyle : bool
        Enable per-band MixStyle augmentation (training only).
    mixstyle_p : float
        Probability of applying MixStyle per forward pass.
    mixstyle_alpha : float
        Beta distribution concentration for style mixing.
    alpha_init, tau_init : float
        UVMD initialisation parameters (see UVMDBlock).
    gn_groups : int
        Number of groups for GroupNorm in band encoders.
    """

    def __init__(
        self,
        K: int = 4,
        L: int = 8,
        in_channels: int = 12,
        feat_dim: int = 64,
        use_mixstyle: bool = False,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        alpha_init: float = 2000.0,
        tau_init: float = 0.01,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.K = K
        self.feat_dim = feat_dim
        self.total_feat_dim = K * feat_dim

        # ── Learnable decomposition ──────────────────────────────────
        self.uvmd = UVMDBlock(K=K, L=L, alpha_init=alpha_init, tau_init=tau_init)

        # ── Optional per-band MixStyle ───────────────────────────────
        if use_mixstyle:
            self.mixstyle = PerBandMixStyle(K=K, p=mixstyle_p, alpha=mixstyle_alpha)
        else:
            self.mixstyle = nn.Identity()

        # ── Per-band CNN encoders ────────────────────────────────────
        self.band_encoders = PerBandCNNEncoder(
            K=K, in_channels=in_channels, feat_dim=feat_dim, gn_groups=gn_groups,
        )

    # ── Encoding stages ──────────────────────────────────────────────────

    def decompose(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw EMG → UVMD modes.

        Parameters
        ----------
        x : (B, T, C)

        Returns
        -------
        modes : (B, K, T, C)
        """
        modes = self.uvmd(x)
        modes = self.mixstyle(modes)
        return modes

    def encode_per_band(
        self, modes: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        UVMD modes → per-band features.

        Parameters
        ----------
        modes : (B, K, T, C)

        Returns
        -------
        per_band : list of K tensors, each (B, feat_dim)
        """
        return self.band_encoders(modes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw EMG → concatenated feature vector.

        Parameters
        ----------
        x : (B, T, C)

        Returns
        -------
        features : (B, K * feat_dim)
        """
        modes = self.decompose(x)
        per_band = self.encode_per_band(modes)
        return torch.cat(per_band, dim=1)

    def encode_selected_bands(
        self, x: torch.Tensor, band_indices: List[int],
    ) -> torch.Tensor:
        """
        Raw EMG → features from selected bands only.

        Useful for frequency-selective contrastive learning (H4_new):
        contrast only on low-frequency (gesture-informative) bands.

        Parameters
        ----------
        x : (B, T, C)
        band_indices : list of int
            Which bands to include (e.g. [0, 1] for low-frequency bands).

        Returns
        -------
        features : (B, len(band_indices) * feat_dim)
        """
        modes = self.decompose(x)
        per_band = self.encode_per_band(modes)
        selected = [per_band[k] for k in band_indices]
        return torch.cat(selected, dim=1)

    # ── UVMD analysis helpers ────────────────────────────────────────────

    def spectral_overlap_penalty(self, sigma: float = 0.05) -> torch.Tensor:
        return self.uvmd.spectral_overlap_penalty(sigma=sigma)

    def get_learned_uvmd_params(self) -> Dict:
        with torch.no_grad():
            return {
                "omega_k": self.uvmd.omega.cpu().numpy().tolist(),
                "alpha_lk": self.uvmd.alpha.cpu().numpy().tolist(),
                "tau_l": self.uvmd.tau.cpu().numpy().tolist(),
            }


# ═════════════════════════════════════════════════════════════════════════════
# Classifier Head (for fine-tuning)
# ═════════════════════════════════════════════════════════════════════════════

class UVMDSSLClassifier(nn.Module):
    """
    Fine-tuning classifier: pretrained UVMDSSLEncoder + MLP head.

    Usage:
      1. Pretrain UVMDSSLEncoder with an SSL objective.
      2. Create UVMDSSLClassifier(encoder, num_classes).
      3. Fine-tune: either freeze encoder (linear probe) or unfreeze all.

    Parameters
    ----------
    encoder : UVMDSSLEncoder
        Pretrained backbone (weights will be shared, not copied).
    num_classes : int
        Number of gesture classes.
    hidden_dim : int
        Hidden dimension in the 2-layer MLP classifier.
    dropout : float
        Dropout rate in the classifier.
    """

    def __init__(
        self,
        encoder: UVMDSSLEncoder,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.total_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, num_classes)"""
        features = self.encoder.encode(x)
        return self.classifier(features)

    def freeze_encoder(self) -> None:
        """Freeze encoder weights for linear probing."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder for full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def spectral_overlap_penalty(self, sigma: float = 0.05) -> torch.Tensor:
        return self.encoder.spectral_overlap_penalty(sigma=sigma)

    def get_learned_uvmd_params(self) -> Dict:
        return self.encoder.get_learned_uvmd_params()


# ═════════════════════════════════════════════════════════════════════════════
# Frequency-Aware Augmentations (applied to UVMD modes)
# ═════════════════════════════════════════════════════════════════════════════

class FreqAwareAugmentation(nn.Module):
    """
    Apply frequency-band-dependent augmentations to UVMD modes.

    Insight from H1/H8: inter-subject variability increases 10× from low to
    high frequency bands.  Therefore:
      - Low bands (gesture info): light augmentation to preserve content
      - High bands (subject noise): aggressive augmentation for invariance

    This module generates two augmented views of the same input for
    contrastive SSL objectives (VICReg, SimCLR, BYOL).

    Parameters
    ----------
    K : int
        Number of frequency bands.
    noise_low : float
        Noise std for low-frequency bands (bands 0, 1).
    noise_high : float
        Noise std for high-frequency bands (bands K-2, K-1).
    scale_range_low : tuple
        Random scaling range for low bands.
    scale_range_high : tuple
        Random scaling range for high bands.
    time_warp_low : float
        Max time warp factor for low bands.
    time_warp_high : float
        Max time warp factor for high bands.
    """

    def __init__(
        self,
        K: int = 4,
        noise_low: float = 0.01,
        noise_high: float = 0.1,
        scale_range_low: Tuple[float, float] = (0.9, 1.1),
        scale_range_high: Tuple[float, float] = (0.5, 2.0),
        time_warp_low: float = 0.05,
        time_warp_high: float = 0.15,
    ):
        super().__init__()
        self.K = K
        # Per-band noise levels: linearly interpolated from low to high
        self.register_buffer(
            "noise_stds",
            torch.linspace(noise_low, noise_high, K),
        )
        # Per-band scale ranges
        scale_lo = torch.linspace(scale_range_low[0], scale_range_high[0], K)
        scale_hi = torch.linspace(scale_range_low[1], scale_range_high[1], K)
        self.register_buffer("scale_lo", scale_lo)
        self.register_buffer("scale_hi", scale_hi)
        self.time_warp_low = time_warp_low
        self.time_warp_high = time_warp_high

    @torch.no_grad()
    def forward(self, modes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views of UVMD modes.

        Parameters
        ----------
        modes : (B, K, T, C) — output of UVMDBlock

        Returns
        -------
        view1, view2 : each (B, K, T, C)
        """
        return self._augment(modes), self._augment(modes)

    def _augment(self, modes: torch.Tensor) -> torch.Tensor:
        B, K, T, C = modes.shape
        out = []
        for k in range(K):
            x_k = modes[:, k].clone()  # (B, T, C)

            # 1. Additive Gaussian noise (band-dependent intensity)
            noise = torch.randn_like(x_k) * self.noise_stds[k]
            x_k = x_k + noise

            # 2. Random per-sample channel scaling
            scale = (
                torch.rand(B, 1, C, device=x_k.device)
                * (self.scale_hi[k] - self.scale_lo[k])
                + self.scale_lo[k]
            )
            x_k = x_k * scale

            # 3. Time warp (interpolation-based)
            warp_max = self.time_warp_low + (
                self.time_warp_high - self.time_warp_low
            ) * k / max(K - 1, 1)
            warp_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * warp_max
            new_T = max(1, int(T * warp_factor))
            # (B, T, C) → (B, C, T) → interpolate → (B, C, T) → (B, T, C)
            x_k_ct = x_k.permute(0, 2, 1)  # (B, C, T)
            x_k_warped = F.interpolate(
                x_k_ct, size=T, mode="linear", align_corners=False,
            )
            x_k = x_k_warped.permute(0, 2, 1)

            out.append(x_k)

        return torch.stack(out, dim=1)  # (B, K, T, C)
