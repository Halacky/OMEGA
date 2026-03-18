#!/usr/bin/env python3
"""
H8: Adaptive Per-Band MixStyle — Learnable per-band augmentation strength.

Motivation:
  H1 shows inter-subject variability is frequency-dependent (10× CV ratio).
  H3/H6/H7 apply MixStyle with the SAME alpha & p across all bands.
  If H1 is correct, optimal augmentation strength should differ by band:
    - Low-freq bands (motor-unit): weak augmentation (low alpha, low p)
    - High-freq bands (subject noise): strong augmentation (high alpha, high p)

  This experiment makes alpha_k and p_k learnable per band, providing:
    1) Potentially stronger MixStyle effect (may reach statistical significance)
    2) Independent confirmation of H1 — if learned params show frequency gradient
    3) Methodological novelty — "frequency-adaptive style augmentation"

Variants:
  C:  Sinc FB + fixed MixStyle     (reproduction of H6-C for fair comparison)
  G:  Sinc FB + AdaptiveMixStyle   (learnable alpha_k, p_k per band)
  F:  UVMD + fixed MixStyle        (reproduction of H7-F for fair comparison)
  H:  UVMD + AdaptiveMixStyle      (learnable alpha_k, p_k per band)

All variants share the same backbone, hyperparameters, and LOSO protocol.
H6/H7 baselines (A, B, E) are NOT re-run — we compare against saved results.

Usage:
  python experiments/h8_adaptive_mixstyle_loso.py                   # 5 CI subjects
  python experiments/h8_adaptive_mixstyle_loso.py --full            # 20 subjects
  python experiments/h8_adaptive_mixstyle_loso.py --variants G H    # adaptive only
  python experiments/h8_adaptive_mixstyle_loso.py --batch_size 64   # match H2-H5
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from sklearn.metrics import accuracy_score, f1_score

# ── project imports ──────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from utils.logging import setup_logging

# ── constants ────────────────────────────────────────────────────────
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [
    f"DB2_s{i}" for i in
    [1, 2, 3, 4, 5, 11, 12, 13, 14, 15,
     26, 27, 28, 29, 30, 36, 37, 38, 39, 40]
]

SEED = 42
WINDOW_SIZE = 200
WINDOW_OVERLAP = 100
SAMPLING_RATE = 2000
N_CHANNELS = 12
K_BANDS = 4
FEAT_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.3

EPOCHS = 80
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 15
LR_PATIENCE = 5
LR_FACTOR = 0.5
VAL_RATIO = 0.15

# Fixed MixStyle params (for variant C/F baselines)
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

# Adaptive MixStyle init (for variant G/H)
ADAPTIVE_ALPHA_INIT = 0.1   # initial alpha for all bands
ADAPTIVE_P_INIT = 0.5       # initial p for all bands
ADAPTIVE_LR_MULT = 10.0     # learning rate multiplier for alpha/p params

# UVMD params (for variants F/H)
K_MODES = 4
L_LAYERS = 8
ALPHA_INIT = 100.0
TAU_INIT = 0.05
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 10.0

VARIANT_NAMES = ["C", "G", "F", "H", "I", "J"]
VARIANT_LABELS = {
    "C": "Sinc FB + fixed MixStyle (H6-C repro)",
    "G": "Sinc FB + AdaptiveMixStyle (learnable alpha_k, p_k)",
    "F": "UVMD + fixed MixStyle (H7-F repro)",
    "H": "UVMD + AdaptiveMixStyle (learnable alpha_k, p_k)",
    "I": "Sinc FB + REVERSED gradient MixStyle (ablation)",
    "J": "UVMD + REVERSED gradient MixStyle (ablation)",
}

# Fixed reversed-gradient params from 5-subject H8 results:
# G learned: alpha_k=[1.50, 1.22, 1.81, 2.64], p_k=[0.016, 0.013, 0.026, 0.054]
# Reversed: swap band1↔band4, band2↔band3
REVERSED_ALPHA_SINC = [2.64, 1.81, 1.22, 1.50]
REVERSED_P_SINC     = [0.054, 0.026, 0.013, 0.016]
# H learned: alpha_k=[1.20, 1.24, 1.71, 1.70], p_k=[0.029, 0.021, 0.033, 0.053]
REVERSED_ALPHA_UVMD = [1.70, 1.71, 1.24, 1.20]
REVERSED_P_UVMD     = [0.053, 0.033, 0.021, 0.029]


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════
def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grouped_windows dict to flat (windows, labels) arrays."""
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            windows_list.append(rep_arr)
            labels_list.extend([gid] * len(rep_arr))
    return np.concatenate(windows_list, axis=0), np.array(labels_list)


# ═════════════════════════════════════════════════════════════════════
#  Model components
# ═════════════════════════════════════════════════════════════════════

# ── Sinc Filterbank ──────────────────────────────────────────────────
class FixedSincFilterbank(nn.Module):
    """Non-learnable K-band Sinc FIR filterbank (FFT convolution)."""

    def __init__(self, K: int = K_BANDS, T: int = WINDOW_SIZE,
                 fs: int = SAMPLING_RATE):
        super().__init__()
        self.K = K
        nyq = fs / 2.0
        edges = np.linspace(20.0, nyq, K + 1)
        order = min(63, T - 1)
        if order % 2 == 0:
            order -= 1

        filters = []
        for k in range(K):
            lo = edges[k] / nyq
            hi = edges[k + 1] / nyq
            lo = max(lo, 1e-4)
            hi = min(hi, 1.0 - 1e-4)
            if lo >= hi:
                hi = lo + 1e-3
            b = sp_signal.firwin(order, [lo, hi], pass_zero=False)
            pad_total = T - len(b)
            pad_l = pad_total // 2
            pad_r = pad_total - pad_l
            b_padded = np.pad(b, (pad_l, pad_r))
            filters.append(b_padded)

        filt_t = torch.tensor(np.stack(filters), dtype=torch.float32)
        self.register_buffer("filters", filt_t)
        self.register_buffer("filters_fft", torch.fft.rfft(filt_t, n=T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, K, T, C)"""
        B, T, C = x.shape
        x_ct = x.permute(0, 2, 1).reshape(B * C, T)
        x_fft = torch.fft.rfft(x_ct, n=T)

        modes = []
        for k in range(self.K):
            y_fft = x_fft * self.filters_fft[k]
            y = torch.fft.irfft(y_fft, n=T)
            modes.append(y)

        out = torch.stack(modes, dim=0)
        out = out.view(self.K, B, C, T).permute(1, 0, 3, 2)
        return out


# ── UVMD Block ───────────────────────────────────────────────────────
class UVMDBlock(nn.Module):
    """Unfolded VMD: L ADMM iterations with learnable alpha, tau, omega."""

    def __init__(self, K: int = K_MODES, L: int = L_LAYERS,
                 alpha_init: float = ALPHA_INIT,
                 tau_init: float = TAU_INIT):
        super().__init__()
        self.K = K
        self.L = L

        self.alpha = nn.Parameter(torch.full((L, K), alpha_init))
        self.tau = nn.Parameter(torch.full((L,), tau_init))
        omega_init = torch.linspace(0.05, 0.45, K)
        self.omega = nn.Parameter(omega_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, K, T, C)"""
        B, T, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * C, T)
        X_fft = torch.fft.rfft(x_flat, n=T)
        freqs = torch.linspace(0, 0.5, T // 2 + 1, device=x.device)

        lambda_hat = torch.zeros_like(X_fft)
        u_hats = [torch.zeros_like(X_fft) for _ in range(self.K)]

        for l_idx in range(self.L):
            for k in range(self.K):
                residual = X_fft - sum(
                    u_hats[j] for j in range(self.K) if j != k
                ) + lambda_hat / 2.0
                denom = 1.0 + self.alpha[l_idx, k] * (freqs - self.omega[k]) ** 2
                u_hats[k] = residual / denom

            sum_u = sum(u_hats)
            lambda_hat = lambda_hat + self.tau[l_idx] * (X_fft - sum_u)

        modes = []
        for k in range(self.K):
            mode_t = torch.fft.irfft(u_hats[k], n=T)
            modes.append(mode_t)

        out = torch.stack(modes, dim=0)
        out = out.view(self.K, B, C, T).permute(1, 0, 3, 2)
        return out

    def spectral_overlap_penalty(self, sigma: float = OVERLAP_SIGMA) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=self.omega.device)
        for j in range(self.K):
            for k in range(j + 1, self.K):
                penalty = penalty + torch.exp(
                    -sigma * (self.omega[j] - self.omega[k]) ** 2
                )
        return penalty


# ── Fixed Per-band MixStyle ─────────────────────────────────────────
class PerBandMixStyle(nn.Module):
    """Per-band style augmentation with FIXED alpha and p (training only)."""

    def __init__(self, K: int, p: float = MIXSTYLE_P,
                 alpha: float = MIXSTYLE_ALPHA):
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
            x_k = modes[:, k]
            mu = x_k.mean(dim=1, keepdim=True)
            sigma = x_k.std(dim=1, keepdim=True) + 1e-6
            x_normed = (x_k - mu) / sigma

            perm = torch.randperm(B, device=x_k.device)
            lam = beta_dist.sample((B, 1, 1)).to(x_k.device)
            mu_mix = lam * mu + (1 - lam) * mu[perm]
            sigma_mix = lam * sigma + (1 - lam) * sigma[perm]

            out_bands.append(x_normed * sigma_mix + mu_mix)

        return torch.stack(out_bands, dim=1)


# ── Adaptive Per-band MixStyle (NEW) ────────────────────────────────
class AdaptivePerBandMixStyle(nn.Module):
    """
    Per-band style augmentation with LEARNABLE per-band alpha_k and p_k.

    Each band k has:
      - alpha_k: Beta distribution concentration (via softplus of raw param)
        Higher alpha → mixing lambda concentrated near 0.5 → stronger mixing
        Lower alpha → lambda concentrated near 0 or 1 → weaker mixing
      - p_k: application probability (via sigmoid of raw param)

    The learned values after training provide an independent diagnostic:
    if H1 is correct, high-freq bands should learn larger alpha_k and p_k.

    Adds only 2K = 8 learnable parameters (for K=4 bands).
    """

    def __init__(self, K: int,
                 alpha_init: float = ADAPTIVE_ALPHA_INIT,
                 p_init: float = ADAPTIVE_P_INIT):
        super().__init__()
        self.K = K

        # Raw parameters (transformed via softplus / sigmoid)
        # Initialize so that softplus(raw) ≈ alpha_init
        raw_alpha = torch.full((K,), self._inv_softplus(alpha_init))
        self.raw_alpha = nn.Parameter(raw_alpha)

        # Initialize so that sigmoid(raw) ≈ p_init
        raw_p = torch.full((K,), self._inv_sigmoid(p_init))
        self.raw_p = nn.Parameter(raw_p)

    @staticmethod
    def _inv_softplus(y: float) -> float:
        """Inverse of softplus: x = log(exp(y) - 1)."""
        return float(np.log(np.exp(y) - 1.0 + 1e-8))

    @staticmethod
    def _inv_sigmoid(y: float) -> float:
        """Inverse of sigmoid: x = log(y / (1-y))."""
        y = max(min(y, 1 - 1e-6), 1e-6)
        return float(np.log(y / (1 - y)))

    @property
    def alpha_k(self) -> torch.Tensor:
        """Per-band alpha values (K,), always positive."""
        return F.softplus(self.raw_alpha) + 1e-3  # min 0.001

    @property
    def p_k(self) -> torch.Tensor:
        """Per-band application probabilities (K,), in [0, 1]."""
        return torch.sigmoid(self.raw_p)

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """(B, K, T, C) → (B, K, T, C)"""
        if not self.training:
            return modes
        B, K, T, C = modes.shape
        alpha_vals = self.alpha_k  # (K,)
        p_vals = self.p_k          # (K,)

        out_bands = []
        for k in range(K):
            x_k = modes[:, k]  # (B, T, C)

            mu = x_k.mean(dim=1, keepdim=True)      # (B, 1, C)
            sigma = x_k.std(dim=1, keepdim=True) + 1e-6
            x_normed = (x_k - mu) / sigma

            # Per-band Beta distribution — rsample for gradient flow through alpha_k
            a_k = alpha_vals[k].clamp(min=0.01)
            beta_dist = torch.distributions.Beta(a_k, a_k)

            perm = torch.randperm(B, device=x_k.device)
            lam = beta_dist.rsample((B, 1, 1)).to(x_k.device)
            mu_mix = lam * mu + (1 - lam) * mu[perm]
            sigma_mix = lam * sigma + (1 - lam) * sigma[perm]

            x_mixed = x_normed * sigma_mix + mu_mix

            # Soft blending with p_k — differentiable (no .item() detach)
            p_k = p_vals[k]  # scalar in [0, 1]
            out_bands.append(p_k * x_mixed + (1 - p_k) * x_k)

        return torch.stack(out_bands, dim=1)

    def get_learned_params(self) -> Dict[str, list]:
        """Return human-readable learned parameters."""
        with torch.no_grad():
            return {
                "alpha_k": self.alpha_k.cpu().numpy().tolist(),
                "p_k": self.p_k.cpu().numpy().tolist(),
                "raw_alpha": self.raw_alpha.cpu().numpy().tolist(),
                "raw_p": self.raw_p.cpu().numpy().tolist(),
            }


# ── Fixed per-band MixStyle with arbitrary alpha_k/p_k (non-learnable) ─
class FixedPerBandMixStyleCustom(nn.Module):
    """Per-band MixStyle with fixed (non-learnable) per-band alpha_k, p_k.
    Used for ablation: reversed gradient test."""

    def __init__(self, alpha_k: List[float], p_k: List[float]):
        super().__init__()
        self.K = len(alpha_k)
        self.register_buffer("alpha_vals", torch.tensor(alpha_k, dtype=torch.float32))
        self.register_buffer("p_vals", torch.tensor(p_k, dtype=torch.float32))

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return modes
        B, K, T, C = modes.shape
        out_bands = []
        for k in range(K):
            x_k = modes[:, k]
            mu = x_k.mean(dim=1, keepdim=True)
            sigma = x_k.std(dim=1, keepdim=True) + 1e-6
            x_normed = (x_k - mu) / sigma

            a_k = self.alpha_vals[k].clamp(min=0.01)
            beta_dist = torch.distributions.Beta(a_k, a_k)
            perm = torch.randperm(B, device=x_k.device)
            lam = beta_dist.sample((B, 1, 1)).to(x_k.device)
            mu_mix = lam * mu + (1 - lam) * mu[perm]
            sigma_mix = lam * sigma + (1 - lam) * sigma[perm]
            x_mixed = x_normed * sigma_mix + mu_mix

            p_k_val = self.p_vals[k]
            out_bands.append(p_k_val * x_mixed + (1 - p_k_val) * x_k)
        return torch.stack(out_bands, dim=1)

    def get_learned_params(self) -> Dict[str, list]:
        return {
            "alpha_k": self.alpha_vals.cpu().numpy().tolist(),
            "p_k": self.p_vals.cpu().numpy().tolist(),
        }


# ── Identity style norm ──────────────────────────────────────────────
class IdentityStyleNorm(nn.Module):
    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        return modes


# ── Per-band CNN encoder ─────────────────────────────────────────────
class PerBandEncoder(nn.Module):
    """K parallel 3-layer Conv1d encoders (shared architecture, independent weights)."""

    def __init__(self, K: int, in_channels: int = N_CHANNELS,
                 feat_dim: int = FEAT_DIM):
        super().__init__()
        self.K = K
        self.encoders = nn.ModuleList([
            self._make_encoder(in_channels, feat_dim) for _ in range(K)
        ])

    @staticmethod
    def _make_encoder(in_ch: int, feat_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """(B, K, T, C) → (B, K*feat_dim)"""
        feats = []
        for k in range(self.K):
            x_k = modes[:, k].permute(0, 2, 1)  # (B, C, T)
            feats.append(self.encoders[k](x_k))  # (B, feat_dim)
        return torch.cat(feats, dim=1)  # (B, K*feat_dim)


# ═════════════════════════════════════════════════════════════════════
#  Model: Sinc-based variants (C, G)
# ═════════════════════════════════════════════════════════════════════
class SincModel(nn.Module):
    """Sinc FB + (fixed or adaptive) MixStyle + encoder + MLP."""

    def __init__(self, variant: str, num_classes: int):
        super().__init__()
        self.variant = variant
        self.K = K_BANDS
        self.frontend = FixedSincFilterbank(K=K_BANDS)

        if variant == "C":
            self.style_norm = PerBandMixStyle(K=self.K)
        elif variant == "G":
            self.style_norm = AdaptivePerBandMixStyle(K=self.K)
        elif variant == "I":
            self.style_norm = FixedPerBandMixStyleCustom(
                alpha_k=REVERSED_ALPHA_SINC, p_k=REVERSED_P_SINC)
        else:
            self.style_norm = IdentityStyleNorm()

        self.encoder = PerBandEncoder(K=self.K)
        self.classifier = nn.Sequential(
            nn.Linear(self.K * FEAT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        modes = self.frontend(x)
        modes = self.style_norm(modes)
        flat = self.encoder(modes)
        return self.classifier(flat)


# ═════════════════════════════════════════════════════════════════════
#  Model: UVMD-based variants (F, H, J)
# ═════════════════════════════════════════════════════════════════════
class UVMDModel(nn.Module):
    """UVMD + (fixed or adaptive) MixStyle + encoder + MLP."""

    def __init__(self, variant: str, num_classes: int):
        super().__init__()
        self.variant = variant
        self.K = K_MODES
        self.uvmd = UVMDBlock(K=K_MODES, L=L_LAYERS,
                              alpha_init=ALPHA_INIT, tau_init=TAU_INIT)

        if variant == "F":
            self.style_norm = PerBandMixStyle(K=self.K)
        elif variant == "H":
            self.style_norm = AdaptivePerBandMixStyle(K=self.K)
        elif variant == "J":
            self.style_norm = FixedPerBandMixStyleCustom(
                alpha_k=REVERSED_ALPHA_UVMD, p_k=REVERSED_P_UVMD)
        else:
            self.style_norm = IdentityStyleNorm()

        self.encoder = PerBandEncoder(K=self.K)
        self.classifier = nn.Sequential(
            nn.Linear(self.K * FEAT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        modes = self.uvmd(x)
        modes = self.style_norm(modes)
        flat = self.encoder(modes)
        return self.classifier(flat)

    def spectral_overlap_penalty(self, sigma: float = OVERLAP_SIGMA) -> torch.Tensor:
        return self.uvmd.spectral_overlap_penalty(sigma=sigma)

    def get_learned_uvmd_params(self) -> Dict:
        with torch.no_grad():
            return {
                "omega_k": self.uvmd.omega.cpu().numpy().tolist(),
                "alpha_lk": self.uvmd.alpha.cpu().numpy().tolist(),
                "tau_l": self.uvmd.tau.cpu().numpy().tolist(),
            }


# ═════════════════════════════════════════════════════════════════════
#  Training / Evaluation
# ═════════════════════════════════════════════════════════════════════
def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / (counts.astype(np.float64) + 1e-8)
    weights /= weights.sum()
    weights *= len(classes)
    w = torch.zeros(int(classes.max()) + 1, dtype=torch.float32, device=device)
    for c, wt in zip(classes, weights):
        w[int(c)] = wt
    return w


def build_model(variant: str, num_classes: int) -> nn.Module:
    if variant in ("C", "G", "I"):
        return SincModel(variant=variant, num_classes=num_classes)
    elif variant in ("F", "H", "J"):
        return UVMDModel(variant=variant, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def train_one_fold(
    variant: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """Train a single LOSO fold and return metrics."""
    seed_everything(SEED)

    model = build_model(variant, num_classes).to(device)

    # Separate param groups: higher LR for adaptive MixStyle params
    mixstyle_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "raw_alpha" in name or "raw_p" in name:
            mixstyle_params.append(param)
        else:
            other_params.append(param)

    param_groups = [{"params": other_params, "lr": LR, "weight_decay": WEIGHT_DECAY}]
    if mixstyle_params:
        param_groups.append({"params": mixstyle_params, "lr": LR * 10,
                             "weight_decay": 0.0})  # 10x LR, no decay
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    use_uvmd = variant in ("F", "H", "J")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            ce_loss = criterion(logits, yb)

            if use_uvmd:
                overlap_loss = model.spectral_overlap_penalty()
                loss = ce_loss + OVERLAP_LAMBDA * overlap_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── val ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), batch_size):
                val_logits.append(model(Xv[vs: vs + batch_size]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"    Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            # Log adaptive MixStyle params if applicable
            ms_info = ""
            if variant in ("G", "H"):
                params = model.style_norm.get_learned_params()
                alpha_str = [f"{a:.3f}" for a in params["alpha_k"]]
                p_str = [f"{p:.3f}" for p in params["p_k"]]
                ms_info = f"  alpha_k={alpha_str}  p_k={p_str}"

            logger.info(
                f"    Epoch {epoch+1}/{EPOCHS}  "
                f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"patience={patience_counter}/{PATIENCE}{ms_info}"
            )

    # ── test ─────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_logits = []
        for ts in range(0, len(Xte), batch_size):
            test_logits.append(model(Xte[ts: ts + batch_size]))
        preds = torch.cat(test_logits).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    result = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
    }

    # Collect adaptive/fixed per-band MixStyle params
    if variant in ("G", "H", "I", "J"):
        ms_params = model.style_norm.get_learned_params()
        result["adaptive_mixstyle"] = ms_params

    # Collect UVMD params
    if use_uvmd:
        uvmd_params = model.get_learned_uvmd_params()
        result["uvmd_omega_k"] = uvmd_params["omega_k"]
        result["overlap_penalty"] = model.spectral_overlap_penalty().item()

    return result


# ═════════════════════════════════════════════════════════════════════
#  LOSO loop
# ═════════════════════════════════════════════════════════════════════
def run_variant(
    variant: str,
    subjects: List[str],
    base_dir: str,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """Run full LOSO for a single variant."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Variant {variant}: {VARIANT_LABELS[variant]}")
    logger.info(f"{'='*60}")

    proc_cfg = ProcessingConfig(
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP,
        sampling_rate=SAMPLING_RATE,
    )
    multi_loader = MultiSubjectLoader(proc_cfg, logger, use_gpu=False,
                                       use_improved_processing=True)
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=subjects,
        exercises=["E1"],
        include_rest=False,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data,
                                                       max_gestures=10)
    gesture_to_class = {g: i for i, g in enumerate(sorted(common_gestures))}
    num_classes = len(gesture_to_class)

    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, (_, _, gw) in subjects_data.items():
        wins, labs = grouped_to_arrays(gw)
        mask = np.isin(labs, list(gesture_to_class.keys()))
        wins, labs = wins[mask], labs[mask]
        labs = np.array([gesture_to_class[g] for g in labs])
        subj_arrays[sid] = (wins, labs)

    fold_results = []
    for test_sid in subjects:
        t0 = time.time()
        logger.info(f"  Fold: test={test_sid}  (variant {variant})")

        X_test, y_test = subj_arrays[test_sid]
        if len(X_test) == 0:
            logger.warning(f"    Skipping {test_sid}: no test data")
            continue

        Xs, ys = [], []
        for sid in subjects:
            if sid == test_sid:
                continue
            w, l = subj_arrays[sid]
            if len(w) > 0:
                Xs.append(w)
                ys.append(l)
        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)

        n = len(X_all)
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(n)
        n_val = max(1, int(n * VAL_RATIO))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        mean_c = X_train.mean(axis=(0, 1), keepdims=True)
        std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train = (X_train - mean_c) / std_c
        X_val = (X_val - mean_c) / std_c
        X_test_norm = (X_test - mean_c) / std_c

        metrics = train_one_fold(
            variant=variant,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test_norm, y_test=y_test,
            num_classes=num_classes,
            device=device, logger=logger,
            batch_size=batch_size,
        )

        elapsed = time.time() - t0
        metrics["test_subject"] = test_sid
        metrics["time_s"] = round(elapsed, 1)
        fold_results.append(metrics)

        # Log results
        ms_str = ""
        if "adaptive_mixstyle" in metrics:
            ms = metrics["adaptive_mixstyle"]
            ms_str = (
                f"  alpha_k={[f'{a:.3f}' for a in ms['alpha_k']]}  "
                f"p_k={[f'{p:.3f}' for p in ms['p_k']]}"
            )
        uvmd_str = ""
        if "uvmd_omega_k" in metrics:
            uvmd_str = f"  omega={[f'{w:.3f}' for w in metrics['uvmd_omega_k']]}"

        logger.info(
            f"    Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}"
            f"{ms_str}{uvmd_str}  ({elapsed:.0f}s)"
        )

    # ── aggregate ────────────────────────────────────────────────
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]

    summary = {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "n_subjects": len(fold_results),
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "std_f1_macro": float(np.std(f1s)) if f1s else 0.0,
        "per_subject": fold_results,
    }

    # Aggregate adaptive MixStyle params across folds
    if variant in ("G", "H", "I", "J"):
        all_alpha = []
        all_p = []
        for r in fold_results:
            if "adaptive_mixstyle" in r:
                all_alpha.append(r["adaptive_mixstyle"]["alpha_k"])
                all_p.append(r["adaptive_mixstyle"]["p_k"])
        if all_alpha:
            alpha_arr = np.array(all_alpha)
            p_arr = np.array(all_p)
            summary["adaptive_mixstyle_aggregate"] = {
                "mean_alpha_k": alpha_arr.mean(axis=0).tolist(),
                "std_alpha_k": alpha_arr.std(axis=0).tolist(),
                "mean_p_k": p_arr.mean(axis=0).tolist(),
                "std_p_k": p_arr.std(axis=0).tolist(),
            }
            logger.info(
                f"  AdaptiveMixStyle aggregate:"
                f"\n    alpha_k = {[f'{a:.4f}±{s:.4f}' for a, s in zip(alpha_arr.mean(0), alpha_arr.std(0))]}"
                f"\n    p_k     = {[f'{p:.4f}±{s:.4f}' for p, s in zip(p_arr.mean(0), p_arr.std(0))]}"
            )

    # Aggregate UVMD omega
    if variant in ("F", "H", "J"):
        all_omega = [r["uvmd_omega_k"] for r in fold_results if "uvmd_omega_k" in r]
        if all_omega:
            omega_arr = np.array(all_omega)
            summary["mean_omega_k"] = omega_arr.mean(axis=0).tolist()
            summary["std_omega_k"] = omega_arr.std(axis=0).tolist()

    logger.info(
        f"\n  >>> Variant {variant}: "
        f"Acc={summary['mean_accuracy']*100:.2f}±{summary['std_accuracy']*100:.2f}  "
        f"F1={summary['mean_f1_macro']*100:.2f}±{summary['std_f1_macro']*100:.2f}"
    )
    return summary


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="H8: Adaptive Per-Band MixStyle ablation"
    )
    parser.add_argument("--full", action="store_true",
                        help="Use 20 subjects (default: 5 CI)")
    parser.add_argument("--ci", type=int, default=0,
                        help="Force CI mode (5 subjects)")
    parser.add_argument("--subjects", type=str, default="",
                        help="Comma-separated subject list override")
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        choices=VARIANT_NAMES,
                        help="Which variants to run (default: C G F H)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory (auto-generated if empty)")
    args, _ = parser.parse_known_args()

    if args.subjects:
        ALL_SUBJECTS = [s.strip() for s in args.subjects.split(",")]
    elif args.full:
        ALL_SUBJECTS = _FULL_SUBJECTS
    else:
        ALL_SUBJECTS = _CI_SUBJECTS

    base_dir = Path(PROJECT_ROOT) / args.data_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "experiments_output",
        f"h8_adaptive_mixstyle_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logging(Path(out_dir))
    logger.info(f"H8: Adaptive Per-Band MixStyle")
    logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Fixed MixStyle: alpha={MIXSTYLE_ALPHA}, p={MIXSTYLE_P}")
    logger.info(f"Adaptive init: alpha={ADAPTIVE_ALPHA_INIT}, p={ADAPTIVE_P_INIT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    all_results = {}
    for v in args.variants:
        result = run_variant(v, ALL_SUBJECTS, base_dir, device, logger,
                             batch_size=args.batch_size)
        all_results[v] = result

        with open(os.path.join(out_dir, f"variant_{v}.json"), "w") as f:
            json.dump(result, f, indent=2)

    # ── comparison table ─────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  H8: ADAPTIVE MIXSTYLE COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Variant':<8} {'Label':<45} {'F1':>10} {'Acc':>10}")
    logger.info("-" * 73)
    for v in args.variants:
        if v in all_results:
            r = all_results[v]
            f1_str = f"{r['mean_f1_macro']*100:.2f}±{r['std_f1_macro']*100:.2f}"
            acc_str = f"{r['mean_accuracy']*100:.2f}±{r['std_accuracy']*100:.2f}"
            logger.info(f"{v:<8} {VARIANT_LABELS[v]:<45} {f1_str:>10} {acc_str:>10}")

    # ── key question: do learned alpha_k show frequency gradient? ─
    from scipy.stats import spearmanr, wilcoxon

    for v in ("G", "H"):
        if v in all_results and "adaptive_mixstyle_aggregate" in all_results[v]:
            agg = all_results[v]["adaptive_mixstyle_aggregate"]
            logger.info(f"\n  Variant {v} — Learned MixStyle parameters:")
            logger.info(f"    Band   Alpha (mean±std)      P (mean±std)")
            for k in range(K_BANDS):
                a_m = agg["mean_alpha_k"][k]
                a_s = agg["std_alpha_k"][k]
                p_m = agg["mean_p_k"][k]
                p_s = agg["std_p_k"][k]
                logger.info(f"    {k+1}      {a_m:.4f}±{a_s:.4f}         {p_m:.4f}±{p_s:.4f}")

            # Gradient test: is alpha_K > alpha_1?
            a = agg["mean_alpha_k"]
            if a[-1] > a[0]:
                ratio = a[-1] / max(a[0], 1e-6)
                logger.info(
                    f"    → alpha gradient confirmed: band {K_BANDS} / band 1 = "
                    f"{ratio:.1f}× (consistent with H1)"
                )
            else:
                logger.info(
                    f"    → No clear alpha gradient (band {K_BANDS}: {a[-1]:.4f}, "
                    f"band 1: {a[0]:.4f})"
                )

            # Spearman rank correlation: band index vs alpha_k (across all folds)
            all_alpha = [r["adaptive_mixstyle"]["alpha_k"]
                         for r in all_results[v]["per_subject"]
                         if "adaptive_mixstyle" in r]
            if len(all_alpha) >= 3:
                band_indices = list(range(K_BANDS))
                # Pool all folds: (n_folds * K_BANDS) observations
                all_band_idx = band_indices * len(all_alpha)
                all_alpha_flat = [a_k for fold_alpha in all_alpha for a_k in fold_alpha]
                rho_alpha, p_alpha = spearmanr(all_band_idx, all_alpha_flat)

                all_p_params = [r["adaptive_mixstyle"]["p_k"]
                                for r in all_results[v]["per_subject"]
                                if "adaptive_mixstyle" in r]
                all_p_flat = [p_k for fold_p in all_p_params for p_k in fold_p]
                rho_p, p_p = spearmanr(all_band_idx, all_p_flat)

                logger.info(
                    f"    Spearman (band_index vs alpha_k): rho={rho_alpha:.3f}, "
                    f"p={p_alpha:.4f} {'***' if p_alpha < 0.001 else '**' if p_alpha < 0.01 else '*' if p_alpha < 0.05 else 'n.s.'}"
                )
                logger.info(
                    f"    Spearman (band_index vs p_k):     rho={rho_p:.3f}, "
                    f"p={p_p:.4f} {'***' if p_p < 0.001 else '**' if p_p < 0.01 else '*' if p_p < 0.05 else 'n.s.'}"
                )

    # ── statistical tests: paired comparisons ────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  STATISTICAL TESTS (Wilcoxon signed-rank, paired per subject)")
    logger.info("=" * 70)

    def _paired_wilcoxon(v1: str, v2: str, metric: str = "f1_macro"):
        if v1 not in all_results or v2 not in all_results:
            return
        r1 = {r["test_subject"]: r[metric] for r in all_results[v1]["per_subject"]}
        r2 = {r["test_subject"]: r[metric] for r in all_results[v2]["per_subject"]}
        common = sorted(set(r1.keys()) & set(r2.keys()))
        if len(common) < 5:
            logger.info(f"  {v1} vs {v2}: too few paired subjects ({len(common)})")
            return
        x = np.array([r1[s] for s in common])
        y = np.array([r2[s] for s in common])
        diff = y - x
        stat, p_val = wilcoxon(diff, alternative="greater")
        mean_diff = diff.mean() * 100
        logger.info(
            f"  {v2} vs {v1} ({VARIANT_LABELS[v2][:30]}... vs {VARIANT_LABELS[v1][:20]}...): "
            f"ΔF1={mean_diff:+.2f}pp, p={p_val:.4f} "
            f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'}"
        )

    # Adaptive vs Fixed
    _paired_wilcoxon("C", "G")  # Sinc: adaptive vs fixed
    _paired_wilcoxon("F", "H")  # UVMD: adaptive vs fixed
    # Reversed gradient ablation
    _paired_wilcoxon("I", "G")  # Sinc: correct vs reversed
    _paired_wilcoxon("J", "H")  # UVMD: correct vs reversed
    # Reversed vs fixed (should be worse or same as fixed)
    _paired_wilcoxon("C", "I")  # reversed vs fixed (Sinc)
    _paired_wilcoxon("F", "J")  # reversed vs fixed (UVMD)

    # save combined results
    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
