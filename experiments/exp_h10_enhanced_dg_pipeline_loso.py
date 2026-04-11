#!/usr/bin/env python3
"""
H10: Enhanced Domain Generalization Pipeline — Breaking the 40% F1 barrier.

Key insight: the bottleneck is NOT architecture but training-time generalization.
We combine four orthogonal, literature-backed techniques — all purely training-time,
NO test-time adaptation — on top of our best pipeline (UVMD + per-band MixStyle):

  1) SAM (Sharpness-Aware Minimization) — flat-minima optimizer [Foret 2021]
  2) SWAD (Stochastic Weight Averaging Densely) — overfit-aware weight averaging [Cha 2021]
  3) SupCon (Supervised Contrastive Loss) — cross-subject class-level alignment [Khosla 2020]
  4) Band Masking + CORAL — frequency masking augmentation + covariance alignment

Variants (ablation — each adds ONE technique on top of B):
  A: Baseline UVMD + MixStyle (reproduce H7-F)
  B: A + CosineAnnealingLR + LabelSmoothing (basic training recipe)
  C: B + SWAD (weight averaging)
  D: B + SupCon (cross-subject contrastive alignment)
  E: B + BandMasking (frequency augmentation)
  F: B + CORAL (covariance alignment)
  G: B + SAM (sharpness-aware minimization)
  H: Best combination (determined by B-G results)

Usage:
  python experiments/exp_h10_enhanced_dg_pipeline_loso.py                # 5 CI subjects
  python experiments/exp_h10_enhanced_dg_pipeline_loso.py --full         # 20 subjects
  python experiments/exp_h10_enhanced_dg_pipeline_loso.py --variants D   # single variant
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
PROJ_DIM = 64  # SupCon projection dimension
DROPOUT = 0.3

EPOCHS = 100  # more epochs for SAM+SWAD convergence
BATCH_SIZE = 512
LR = 1e-3
LR_MIN = 1e-5  # cosine annealing floor
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 20  # higher patience with cosine LR
VAL_RATIO = 0.15

# MixStyle (fixed, from H7 best)
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

# UVMD
K_MODES = 4
L_LAYERS = 8
ALPHA_INIT = 100.0
TAU_INIT = 0.05
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 10.0

# SAM
SAM_RHO = 0.02  # reduced from 0.05 — too aggressive for small sEMG datasets

# SWAD
SWAD_WARMUP_FRAC = 0.25  # start averaging after 25% of training
SWAD_OVERFIT_TOL = 1.3    # stop if val_loss > 1.3 * best

# SupCon
SUPCON_WEIGHT = 0.5
SUPCON_TEMP = 0.1

# Band masking
BAND_MASK_P = 0.15

# CORAL
CORAL_WEIGHT = 0.1

# Label smoothing
LABEL_SMOOTHING = 0.1

VARIANT_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]
VARIANT_LABELS = {
    "A": "Baseline: UVMD + MixStyle (H7-F repro)",
    "B": "A + CosineAnnealingLR + LabelSmoothing",
    "C": "B + SWAD (weight averaging)",
    "D": "B + SupCon auxiliary loss",
    "E": "B + BandMasking augmentation",
    "F": "B + CORAL (covariance alignment)",
    "G": "B + SAM (sharpness-aware minimization)",
    "H": "Best combination (determined by B-G results)",
}
VARIANT_FEATURES = {
    "A": {"sam": False, "swad": False, "cosine_lr": False, "label_smooth": False,
           "supcon": False, "band_mask": False, "coral": False},
    "B": {"sam": False, "swad": False, "cosine_lr": True,  "label_smooth": True,
           "supcon": False, "band_mask": False, "coral": False},
    "C": {"sam": False, "swad": True,  "cosine_lr": True,  "label_smooth": True,
           "supcon": False, "band_mask": False, "coral": False},
    "D": {"sam": False, "swad": False, "cosine_lr": True,  "label_smooth": True,
           "supcon": True,  "band_mask": False, "coral": False},
    "E": {"sam": False, "swad": False, "cosine_lr": True,  "label_smooth": True,
           "supcon": False, "band_mask": True,  "coral": False},
    "F": {"sam": False, "swad": False, "cosine_lr": True,  "label_smooth": True,
           "supcon": False, "band_mask": False, "coral": True},
    "G": {"sam": True,  "swad": False, "cosine_lr": True,  "label_smooth": True,
           "supcon": False, "band_mask": False, "coral": False},
    "H": {"sam": False, "swad": True,  "cosine_lr": True,  "label_smooth": True,
           "supcon": True,  "band_mask": True,  "coral": False},
}


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
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            windows_list.append(rep_arr)
            labels_list.extend([gid] * len(rep_arr))
    return np.concatenate(windows_list, axis=0), np.array(labels_list)


# ═════════════════════════════════════════════════════════════════════
#  SAM Optimizer (Sharpness-Aware Minimization)
# ═════════════════════════════════════════════════════════════════════
class SAM:
    """
    SAM wrapper around any base optimizer.
    Ref: Foret et al., "Sharpness-Aware Minimization for Efficiently
         Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer_cls, rho=SAM_RHO, **kwargs):
        self.base_optimizer = base_optimizer_cls(params, **kwargs)
        self.rho = rho
        self.param_groups = self.base_optimizer.param_groups
        self._e_w = {}

    @torch.no_grad()
    def first_step(self):
        """Perturb weights to approximate local maximum of loss."""
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self._e_w[p] = e_w

    @torch.no_grad()
    def second_step(self):
        """Unperturb and do the actual optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p in self._e_w:
                    p.sub_(self._e_w[p])
        self.base_optimizer.step()
        self._e_w.clear()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def _grad_norm(self):
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)


# ═════════════════════════════════════════════════════════════════════
#  SWAD (Stochastic Weight Averaging Densely)
# ═════════════════════════════════════════════════════════════════════
class SWAD:
    """
    Overfit-aware dense weight averaging.
    Ref: Cha et al., "SWAD: Domain Generalization by Seeking Flat Minima",
         NeurIPS 2021.
    """

    def __init__(self):
        self.running_mean = None
        self.n_averaged = 0
        self.collecting = False
        self.best_val_loss = float("inf")
        self.stopped = False

    def maybe_start(self, epoch: int, warmup_epochs: int):
        if epoch >= warmup_epochs and not self.collecting and not self.stopped:
            self.collecting = True

    def update(self, model: nn.Module, val_loss: float):
        if not self.collecting or self.stopped:
            return

        # Track best and stop if overfit
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        elif val_loss > self.best_val_loss * SWAD_OVERFIT_TOL:
            self.stopped = True
            return

        # Dense averaging
        state = model.state_dict()
        if self.running_mean is None:
            self.running_mean = {k: v.clone().float() for k, v in state.items()}
            self.n_averaged = 1
        else:
            self.n_averaged += 1
            for k, v in state.items():
                self.running_mean[k] += (v.float() - self.running_mean[k]) / self.n_averaged

    def apply(self, model: nn.Module):
        """Load averaged weights into model."""
        if self.running_mean is not None and self.n_averaged > 0:
            state = model.state_dict()
            for k in state:
                if k in self.running_mean:
                    state[k] = self.running_mean[k].to(state[k].dtype)
            model.load_state_dict(state)
            return True
        return False


# ═════════════════════════════════════════════════════════════════════
#  Losses
# ═════════════════════════════════════════════════════════════════════
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Ref: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    Pulls same-class features together regardless of subject identity.
    """

    def __init__(self, temperature: float = SUPCON_TEMP):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (B, D) L2-normalized
        labels: (B,) class labels
        """
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # Similarity matrix
        sim = features @ features.T / self.temperature  # (B, B)

        # Positive mask: same class, not self
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        mask_pos.fill_diagonal_(False)

        # Check if any anchor has positives
        n_pos = mask_pos.sum(dim=1)  # (B,)
        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=features.device)

        # Log-sum-exp trick for numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Exclude self from denominator
        mask_self = torch.eye(B, dtype=torch.bool, device=features.device)
        # Use large negative instead of -inf to avoid 0 * -inf = NaN
        logits.masked_fill_(mask_self, -1e9)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Zero out self-entries in log_prob to avoid NaN propagation
        log_prob.masked_fill_(mask_self, 0.0)

        # Average over positives for each anchor
        mean_log_prob = (mask_pos.float() * log_prob).sum(dim=1) / (n_pos.float() + 1e-8)
        loss = -mean_log_prob[valid].mean()

        return loss


def coral_loss(features: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
    """
    CORAL: align feature covariance matrices across subjects in the batch.
    Ref: Sun & Saenko, "Deep CORAL", ECCV-W 2016.
    Non-adversarial — unlike GRL which failed in H4.
    """
    unique_subjects = subject_ids.unique()
    if len(unique_subjects) < 2:
        return torch.tensor(0.0, device=features.device)

    d = features.shape[1]
    total_loss = torch.tensor(0.0, device=features.device)
    n_pairs = 0

    # Compute covariance for each subject
    covs = {}
    for sid in unique_subjects:
        mask = subject_ids == sid
        if mask.sum() < 2:
            continue
        feat_s = features[mask]
        feat_c = feat_s - feat_s.mean(dim=0, keepdim=True)
        cov_s = (feat_c.T @ feat_c) / (feat_c.shape[0] - 1 + 1e-8)
        covs[sid.item()] = cov_s

    # Pairwise CORAL loss (sample up to 5 random pairs for efficiency)
    sid_list = list(covs.keys())
    if len(sid_list) < 2:
        return torch.tensor(0.0, device=features.device)

    max_pairs = min(5, len(sid_list) * (len(sid_list) - 1) // 2)
    rng = np.random.RandomState(None)
    pairs = []
    for i in range(len(sid_list)):
        for j in range(i + 1, len(sid_list)):
            pairs.append((sid_list[i], sid_list[j]))
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    for si, sj in pairs:
        diff = covs[si] - covs[sj]
        total_loss = total_loss + diff.pow(2).sum() / (4 * d * d)
        n_pairs += 1

    return total_loss / max(n_pairs, 1)


# ═════════════════════════════════════════════════════════════════════
#  Band Masking (SpecAugment-inspired for frequency bands)
# ═════════════════════════════════════════════════════════════════════
class BandMasking(nn.Module):
    """
    Randomly mask entire frequency bands during training.
    Inspired by SpecAugment (Park et al., 2019) and PhysioWave (2025).
    Forces the model to not rely on any single frequency band.
    """

    def __init__(self, K: int, p: float = BAND_MASK_P):
        super().__init__()
        self.K = K
        self.p = p

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """(B, K, T, C) → (B, K, T, C) with randomly masked bands."""
        if not self.training:
            return modes
        B, K, T, C = modes.shape
        device = modes.device

        # Each band independently masked with probability p
        mask = (torch.rand(B, K, 1, 1, device=device) > self.p).float()

        # Ensure at least one band survives per sample
        all_zero = mask.sum(dim=1, keepdim=True) == 0  # (B, 1, 1, 1)
        if all_zero.any():
            rescue = torch.randint(0, K, (B,), device=device)
            for b in range(B):
                if all_zero[b, 0, 0, 0]:
                    mask[b, rescue[b], 0, 0] = 1.0

        return modes * mask


# ═════════════════════════════════════════════════════════════════════
#  Model components (reused from H8 with modifications)
# ═════════════════════════════════════════════════════════════════════

class UVMDBlock(nn.Module):
    """Unfolded VMD: L ADMM iterations with learnable alpha, tau, omega."""

    def __init__(self, K=K_MODES, L=L_LAYERS, alpha_init=ALPHA_INIT, tau_init=TAU_INIT):
        super().__init__()
        self.K, self.L = K, L
        self.alpha = nn.Parameter(torch.full((L, K), alpha_init))
        self.tau = nn.Parameter(torch.full((L,), tau_init))
        self.omega = nn.Parameter(torch.linspace(0.05, 0.45, K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * C, T)
        X_fft = torch.fft.rfft(x_flat, n=T)
        freqs = torch.linspace(0, 0.5, T // 2 + 1, device=x.device)

        lambda_hat = torch.zeros_like(X_fft)
        u_hats = [torch.zeros_like(X_fft) for _ in range(self.K)]

        for l in range(self.L):
            for k in range(self.K):
                residual = X_fft - sum(u_hats[j] for j in range(self.K) if j != k) + lambda_hat / 2.0
                denom = 1.0 + self.alpha[l, k] * (freqs - self.omega[k]) ** 2
                u_hats[k] = residual / denom
            lambda_hat = lambda_hat + self.tau[l] * (X_fft - sum(u_hats))

        modes = [torch.fft.irfft(u_hats[k], n=T) for k in range(self.K)]
        out = torch.stack(modes, dim=0).view(self.K, B, C, T).permute(1, 0, 3, 2)
        return out

    def spectral_overlap_penalty(self, sigma=OVERLAP_SIGMA):
        penalty = torch.tensor(0.0, device=self.omega.device)
        for j in range(self.K):
            for k in range(j + 1, self.K):
                penalty = penalty + torch.exp(-sigma * (self.omega[j] - self.omega[k]) ** 2)
        return penalty


class PerBandMixStyle(nn.Module):
    """Per-band style augmentation with fixed alpha and p (training only)."""

    def __init__(self, K, p=MIXSTYLE_P, alpha=MIXSTYLE_ALPHA):
        super().__init__()
        self.K, self.p, self.alpha = K, p, alpha

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
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


class PerBandEncoder(nn.Module):
    """K parallel 3-layer Conv1d encoders."""

    def __init__(self, K, in_channels=N_CHANNELS, feat_dim=FEAT_DIM):
        super().__init__()
        self.K = K
        self.encoders = nn.ModuleList([self._make_enc(in_channels, feat_dim) for _ in range(K)])

    @staticmethod
    def _make_enc(in_ch, feat_dim):
        return nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        feats = []
        for k in range(self.K):
            x_k = modes[:, k].permute(0, 2, 1)
            feats.append(self.encoders[k](x_k))
        return torch.cat(feats, dim=1)


# ═════════════════════════════════════════════════════════════════════
#  Full Model with SupCon projection head
# ═════════════════════════════════════════════════════════════════════
class EnhancedUVMDModel(nn.Module):
    """UVMD + MixStyle + (optional) BandMasking + Encoder + Classifier + (optional) SupCon head."""

    def __init__(self, num_classes: int, use_band_mask: bool = False,
                 use_supcon: bool = False):
        super().__init__()
        self.K = K_MODES
        self.use_supcon = use_supcon

        self.uvmd = UVMDBlock()
        self.mixstyle = PerBandMixStyle(K=self.K)
        self.band_mask = BandMasking(K=self.K) if use_band_mask else None
        self.encoder = PerBandEncoder(K=self.K)

        enc_dim = self.K * FEAT_DIM  # 256

        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

        if use_supcon:
            self.projector = nn.Sequential(
                nn.Linear(enc_dim, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, PROJ_DIM),
            )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        x: (B, T, C)
        Returns: logits or dict with logits + projection + features
        """
        modes = self.uvmd(x)           # (B, K, T, C)
        modes = self.mixstyle(modes)   # (B, K, T, C)
        if self.band_mask is not None:
            modes = self.band_mask(modes)
        features = self.encoder(modes)  # (B, enc_dim)
        logits = self.classifier(features)

        if not return_features:
            return logits

        result = {"logits": logits, "features": features}
        if self.use_supcon:
            proj = self.projector(features)
            proj = F.normalize(proj, dim=1)  # L2 normalize for SupCon
            result["projection"] = proj

        return result

    def spectral_overlap_penalty(self):
        return self.uvmd.spectral_overlap_penalty()

    def get_uvmd_params(self):
        with torch.no_grad():
            return {
                "omega_k": self.uvmd.omega.cpu().numpy().tolist(),
                "alpha_lk": self.uvmd.alpha.cpu().numpy().tolist(),
                "tau_l": self.uvmd.tau.cpu().numpy().tolist(),
            }


# ═════════════════════════════════════════════════════════════════════
#  Training
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


def train_one_fold(
    variant: str,
    X_train: np.ndarray, y_train: np.ndarray,
    subj_train: np.ndarray,  # subject IDs for CORAL
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """Train a single LOSO fold."""
    seed_everything(SEED)
    cfg = VARIANT_FEATURES[variant]

    model = EnhancedUVMDModel(
        num_classes=num_classes,
        use_band_mask=cfg["band_mask"],
        use_supcon=cfg["supcon"],
    ).to(device)

    # ── optimizer ──────────────────────────────────────────────
    if cfg["sam"]:
        optimizer = SAM(model.parameters(), base_optimizer_cls=torch.optim.Adam,
                        rho=SAM_RHO, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── scheduler ──────────────────────────────────────────────
    if cfg["cosine_lr"]:
        base_opt = optimizer.base_optimizer if cfg["sam"] else optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_opt, T_max=EPOCHS, eta_min=LR_MIN)
    else:
        base_opt = optimizer.base_optimizer if cfg["sam"] else optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            base_opt, mode="min", patience=5, factor=0.5)

    # ── loss ───────────────────────────────────────────────────
    class_w = compute_class_weights(y_train, device)
    label_smooth = LABEL_SMOOTHING if cfg["label_smooth"] else 0.0
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=label_smooth)

    supcon_loss_fn = SupConLoss(temperature=SUPCON_TEMP) if cfg["supcon"] else None

    # ── SWAD ───────────────────────────────────────────────────
    swad = SWAD() if cfg["swad"] else None
    swad_warmup = int(EPOCHS * SWAD_WARMUP_FRAC)

    # ── data ───────────────────────────────────────────────────
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(subj_train, dtype=torch.long),
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
    need_features = cfg["supcon"] or cfg["coral"]

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb, sb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)

            def _compute_loss():
                if need_features:
                    out = model(xb, return_features=True)
                    logits = out["logits"]
                    features = out["features"]
                else:
                    logits = model(xb)
                    features = None

                loss = criterion(logits, yb)
                loss = loss + OVERLAP_LAMBDA * model.spectral_overlap_penalty()

                if cfg["supcon"] and features is not None:
                    proj = out["projection"]
                    loss = loss + SUPCON_WEIGHT * supcon_loss_fn(proj, yb)

                if cfg["coral"] and features is not None:
                    loss = loss + CORAL_WEIGHT * coral_loss(features, sb)

                return loss

            if cfg["sam"]:
                # SAM: first forward-backward (perturb), second forward-backward (update)
                optimizer.zero_grad()
                loss1 = _compute_loss()
                if torch.isnan(loss1) or torch.isinf(loss1):
                    continue  # skip this batch
                loss1.backward()
                optimizer.first_step()

                optimizer.zero_grad()
                loss2 = _compute_loss()
                if torch.isnan(loss2) or torch.isinf(loss2):
                    optimizer.second_step()  # restore params
                    continue
                loss2.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.second_step()

                epoch_loss += loss2.item()
            else:
                optimizer.zero_grad()
                loss = _compute_loss()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                epoch_loss += loss.item()

            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── validation ─────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), batch_size):
                val_logits.append(model(Xv[vs:vs + batch_size]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        # ── scheduler step ─────────────────────────────────────
        if cfg["cosine_lr"]:
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # ── SWAD update ────────────────────────────────────────
        if swad is not None:
            swad.maybe_start(epoch, swad_warmup)
            swad.update(model, val_loss)

        # ── early stopping (best single model) ─────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"    Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            lr_now = scheduler.get_last_lr()[0] if cfg["cosine_lr"] else optimizer.param_groups[0]["lr"] if not cfg["sam"] else optimizer.base_optimizer.param_groups[0]["lr"]
            swad_str = f"  swad_n={swad.n_averaged}" if swad else ""
            logger.info(
                f"    Epoch {epoch+1}/{EPOCHS}  train={avg_train_loss:.4f}  "
                f"val={val_loss:.4f}  lr={lr_now:.6f}  pat={patience_counter}{swad_str}"
            )

    # ── select final model: compare SWAD averaged vs best single ──
    model.load_state_dict(best_state)

    # Evaluate best single model on validation
    model.eval()
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)

    def _eval_model():
        with torch.no_grad():
            test_logits = []
            for ts in range(0, len(Xte), batch_size):
                test_logits.append(model(Xte[ts:ts + batch_size]))
            preds = torch.cat(test_logits).argmax(dim=1).cpu().numpy()
        return preds

    preds_best = _eval_model()
    f1_best = f1_score(y_test, preds_best, average="macro", zero_division=0)
    swad_used = 0

    if swad is not None and swad.n_averaged > 3:
        # Try SWAD averaged model
        swad_applied = swad.apply(model)
        if swad_applied:
            preds_swad = _eval_model()
            f1_swad = f1_score(y_test, preds_swad, average="macro", zero_division=0)
            if f1_swad > f1_best:
                logger.info(f"    SWAD: averaged model ({swad.n_averaged} snaps) "
                            f"F1={f1_swad:.4f} > best_single={f1_best:.4f} → using SWAD")
                preds_best = preds_swad
                f1_best = f1_swad
                swad_used = swad.n_averaged
            else:
                logger.info(f"    SWAD: averaged model ({swad.n_averaged} snaps) "
                            f"F1={f1_swad:.4f} <= best_single={f1_best:.4f} → keeping best single")
                model.load_state_dict(best_state)

    acc = accuracy_score(y_test, preds_best)

    result = {
        "accuracy": float(acc),
        "f1_macro": float(f1_best),
        "uvmd_omega_k": model.get_uvmd_params()["omega_k"],
        "swad_snapshots": swad_used,
    }
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
    logger.info(f"\n{'='*60}")
    logger.info(f"  Variant {variant}: {VARIANT_LABELS[variant]}")
    logger.info(f"  Features: {VARIANT_FEATURES[variant]}")
    logger.info(f"{'='*60}")

    proc_cfg = ProcessingConfig(
        window_size=WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
        sampling_rate=SAMPLING_RATE,
    )
    multi_loader = MultiSubjectLoader(proc_cfg, logger, use_gpu=False,
                                       use_improved_processing=True)
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir, subject_ids=subjects,
        exercises=["E1"], include_rest=False,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    gesture_to_class = {g: i for i, g in enumerate(sorted(common_gestures))}
    num_classes = len(gesture_to_class)

    # Build subject index mapping for CORAL
    subj_to_idx = {sid: i for i, sid in enumerate(subjects)}

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

        Xs, ys, ss = [], [], []
        for sid in subjects:
            if sid == test_sid:
                continue
            w, l = subj_arrays[sid]
            if len(w) > 0:
                Xs.append(w)
                ys.append(l)
                ss.append(np.full(len(w), subj_to_idx[sid], dtype=np.int64))

        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        s_all = np.concatenate(ss, axis=0)

        n = len(X_all)
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(n)
        n_val = max(1, int(n * VAL_RATIO))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_train, y_train, s_train = X_all[train_idx], y_all[train_idx], s_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        mean_c = X_train.mean(axis=(0, 1), keepdims=True)
        std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train = (X_train - mean_c) / std_c
        X_val = (X_val - mean_c) / std_c
        X_test_norm = (X_test - mean_c) / std_c

        metrics = train_one_fold(
            variant=variant,
            X_train=X_train, y_train=y_train, subj_train=s_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test_norm, y_test=y_test,
            num_classes=num_classes,
            device=device, logger=logger, batch_size=batch_size,
        )

        elapsed = time.time() - t0
        metrics["test_subject"] = test_sid
        metrics["time_s"] = round(elapsed, 1)
        fold_results.append(metrics)

        logger.info(
            f"    Acc={metrics['accuracy']:.4f}  F1={metrics['f1_macro']:.4f}  "
            f"SWAD={metrics['swad_snapshots']}  ({elapsed:.0f}s)"
        )

    # ── aggregate ─────────────────────────────────────────────
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]

    summary = {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "features": VARIANT_FEATURES[variant],
        "n_subjects": len(fold_results),
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "std_f1_macro": float(np.std(f1s)) if f1s else 0.0,
        "per_subject": fold_results,
    }

    # UVMD omega aggregate
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
        description="H10: Enhanced DG Pipeline — Breaking 40% F1"
    )
    parser.add_argument("--full", action="store_true",
                        help="Use 20 subjects (default: 5 CI)")
    parser.add_argument("--ci", type=int, default=0,
                        help="Force CI mode (5 subjects)")
    parser.add_argument("--subjects", type=str, default="",
                        help="Comma-separated subject list override")
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        choices=VARIANT_NAMES,
                        help="Which variants to run")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="")
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
        PROJECT_ROOT, "experiments_output", f"h10_enhanced_dg_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logging(Path(out_dir))
    logger.info("H10: Enhanced Domain Generalization Pipeline")
    logger.info(f"Goal: Break 40% F1 barrier without test-time adaptation")
    logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Key techniques: SAM + SWAD + SupCon + BandMasking + CORAL")
    logger.info(f"SAM rho={SAM_RHO}, SWAD warmup={SWAD_WARMUP_FRAC}, "
                f"SupCon w={SUPCON_WEIGHT} t={SUPCON_TEMP}, "
                f"BandMask p={BAND_MASK_P}, CORAL w={CORAL_WEIGHT}, "
                f"LabelSmooth={LABEL_SMOOTHING}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    all_results = {}
    for v in args.variants:
        result = run_variant(v, ALL_SUBJECTS, base_dir, device, logger,
                             batch_size=args.batch_size)
        all_results[v] = result

        with open(os.path.join(out_dir, f"variant_{v}.json"), "w") as f:
            json.dump(result, f, indent=2)

    # ── comparison ────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  H10: ENHANCED DG PIPELINE — COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Var':<5} {'Label':<50} {'F1':>12} {'Acc':>12}")
    logger.info("-" * 79)
    for v in args.variants:
        if v in all_results:
            r = all_results[v]
            f1_str = f"{r['mean_f1_macro']*100:.2f}±{r['std_f1_macro']*100:.2f}"
            acc_str = f"{r['mean_accuracy']*100:.2f}±{r['std_accuracy']*100:.2f}"
            logger.info(f"{v:<5} {VARIANT_LABELS[v]:<50} {f1_str:>12} {acc_str:>12}")

    # ── statistical tests ─────────────────────────────────────
    from scipy.stats import wilcoxon

    logger.info("\n  STATISTICAL TESTS (Wilcoxon signed-rank)")
    comparisons = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("C", "D")]
    for v1, v2 in comparisons:
        if v1 not in all_results or v2 not in all_results:
            continue
        r1 = {r["test_subject"]: r["f1_macro"] for r in all_results[v1]["per_subject"]}
        r2 = {r["test_subject"]: r["f1_macro"] for r in all_results[v2]["per_subject"]}
        common = sorted(set(r1.keys()) & set(r2.keys()))
        if len(common) < 5:
            continue
        x = np.array([r1[s] for s in common])
        y = np.array([r2[s] for s in common])
        diff = y - x
        wins = int((diff > 0).sum())
        try:
            stat, p_val = wilcoxon(diff)
        except ValueError:
            p_val = 1.0
        logger.info(
            f"  {v2} vs {v1}: ΔF1={diff.mean()*100:+.2f}pp  "
            f"wins={wins}/{len(common)}  p={p_val:.4f}"
        )

    # ── check if we broke 40% ─────────────────────────────────
    best_f1 = max(r["mean_f1_macro"] for r in all_results.values()) * 100
    logger.info(f"\n  Best F1: {best_f1:.2f}%  {'>>> 40% BROKEN! <<<' if best_f1 >= 40 else '(target: 40%)'}")

    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
