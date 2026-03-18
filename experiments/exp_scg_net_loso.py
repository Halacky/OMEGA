#!/usr/bin/env python3
"""
SCG-Net: Spectral Content Gate Network — novel architecture for cross-subject
sEMG gesture recognition.

Novel contribution: Spectral Content Gate (SCG) — a learnable per-band gate
γ_k ∈ [0,1] that controls how much instance-level style (amplitude/scale
statistics) to remove from each frequency band.  Combined with a per-band
subject adversary (gradient reversal), the gate learns the optimal
content/style trade-off per frequency band end-to-end.

Key difference from prior work:
  - Instance Norm removes ALL style → loses useful low-freq info
  - MixStyle randomly mixes styles → is a regularizer, not a learned gate
  - SCG learns HOW MUCH style to remove per band → frequency-aware invariance

Expected behavior (based on H1/H8 findings):
  γ_1 ≈ 0.1 (low freq: keep style — low inter-subject variability)
  γ_4 ≈ 0.9 (high freq: remove style — high inter-subject variability)

Variants:
  A:  Sinc FB + SCG (adversary ON)  — main architecture
  B:  UVMD + SCG (adversary ON)     — main architecture
  C:  Sinc FB + SCG (no adversary)  — ablation: adversary contribution
  D:  Sinc FB + full IN (γ=1 fixed) — ablation: gate contribution

Usage:
  python experiments/exp_scg_net_loso.py                     # 5 CI subjects
  python experiments/exp_scg_net_loso.py --full              # 20 subjects
  python experiments/exp_scg_net_loso.py --variants A B      # specific variants
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

# SCG-specific
ADV_WEIGHT = 0.3          # λ_adv — adversary loss weight
ADV_WARMUP_EPOCHS = 10    # GRL alpha anneals from 0→1 over this many epochs
GATE_LR_MULT = 10.0       # learning rate multiplier for gate params
ADV_HIDDEN = 128           # adversary MLP hidden size

# UVMD params
K_MODES = 4
L_LAYERS = 8
ALPHA_INIT = 100.0
TAU_INIT = 0.05
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 10.0

VARIANT_NAMES = ["A", "B", "C", "D"]
VARIANT_LABELS = {
    "A": "Sinc FB + SCG (adversary ON)",
    "B": "UVMD + SCG (adversary ON)",
    "C": "Sinc FB + SCG (no adversary, ablation)",
    "D": "Sinc FB + full IN (γ=1 fixed, ablation)",
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
    """Convert grouped_windows dict to flat (windows, labels) arrays."""
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            windows_list.append(rep_arr)
            labels_list.extend([gid] * len(rep_arr))
    return np.concatenate(windows_list, axis=0), np.array(labels_list)


# ═════════════════════════════════════════════════════════════════════
#  Gradient Reversal Layer
# ═════════════════════════════════════════════════════════════════════
class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradient during backward pass, scales by alpha."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ═════════════════════════════════════════════════════════════════════
#  Spectral Content Gate (SCG) — the novel primitive
# ═════════════════════════════════════════════════════════════════════
class SpectralContentGate(nn.Module):
    """
    Learnable per-band content gate for frequency-aware style removal.

    For each frequency band k, blends between instance-normalized
    (style-free) and raw (style-preserved) input:

        x_gated_k = γ_k · IN(x_k) + (1 - γ_k) · x_k

    where γ_k = sigmoid(raw_gate_k) is a learnable scalar in [0, 1].

    - γ_k → 1: full instance norm (removes all style)
    - γ_k → 0: no normalization (preserves all style)

    The gate values provide interpretable per-band style-removal strength,
    expected to correlate with inter-subject variability per H1.

    Adds only K learnable parameters (K=4 → 4 params).
    """

    def __init__(self, K: int, init_gate: float = 0.5, fixed_gate: float = -1.0):
        """
        Args:
            K: number of frequency bands
            init_gate: initial γ value (0.5 = neutral)
            fixed_gate: if >= 0, use this fixed value for all gates (no learning)
        """
        super().__init__()
        self.K = K
        self.fixed_gate = fixed_gate

        if fixed_gate < 0:
            # Learnable gates: sigmoid(raw) ≈ init_gate
            raw_init = float(np.log(init_gate / (1.0 - init_gate + 1e-8) + 1e-8))
            self.raw_gate = nn.Parameter(torch.full((K,), raw_init))
        else:
            # Fixed gates (for ablation variant D)
            self.register_buffer(
                "raw_gate",
                torch.full((K,), float(np.log(fixed_gate / (1.0 - fixed_gate + 1e-8) + 1e-8)))
            )

    @property
    def gate_values(self) -> torch.Tensor:
        """Per-band gate values γ_k ∈ [0, 1], shape (K,)."""
        return torch.sigmoid(self.raw_gate)

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """
        Apply per-band content gating.

        Args:
            modes: (B, K, T, C) — frequency-decomposed EMG
        Returns:
            gated: (B, K, T, C) — with per-band style partially removed
        """
        B, K, T, C = modes.shape
        gamma = self.gate_values  # (K,)

        out_bands = []
        for k in range(K):
            x_k = modes[:, k]  # (B, T, C)

            # Instance statistics (per-sample, per-channel)
            mu = x_k.mean(dim=1, keepdim=True)       # (B, 1, C)
            sigma = x_k.std(dim=1, keepdim=True) + 1e-6  # (B, 1, C)

            # Instance-normalized (content only)
            x_normed = (x_k - mu) / sigma

            # Blend: γ=1 → full IN, γ=0 → raw
            g_k = gamma[k]
            x_gated = g_k * x_normed + (1.0 - g_k) * x_k
            out_bands.append(x_gated)

        return torch.stack(out_bands, dim=1)

    def get_learned_params(self) -> Dict[str, list]:
        with torch.no_grad():
            return {
                "gate_values": self.gate_values.cpu().numpy().tolist(),
                "raw_gate": self.raw_gate.cpu().numpy().tolist(),
            }


# ═════════════════════════════════════════════════════════════════════
#  Frontend: Sinc Filterbank
# ═════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════
#  Frontend: UVMD Block
# ═════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════
#  Per-band CNN encoder
# ═════════════════════════════════════════════════════════════════════
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
#  SCG-Net: Spectral Content Gate Network
# ═════════════════════════════════════════════════════════════════════
class SCGNet(nn.Module):
    """
    Spectral Content Gate Network.

    Architecture:
        Input (B, T, C)
          → Frontend (Sinc/UVMD) → (B, K, T, C) modes
          → SpectralContentGate   → (B, K, T, C) gated modes
          → PerBandEncoder        → (B, K*feat_dim)
          → Gesture classifier    → (B, num_classes)
          → Subject adversary     → (B, num_subjects) [training only, with GRL]

    The SCG learns per-band γ_k that controls style removal strength.
    The adversary provides gradient signal to push representations
    toward subject invariance, with the GRL ensuring the encoder
    learns to remove subject information.
    """

    def __init__(
        self,
        num_classes: int,
        num_subjects: int,
        frontend: str = "sinc",
        K: int = K_BANDS,
        use_adversary: bool = True,
        fixed_gate: float = -1.0,
        gate_init: float = 0.5,
    ):
        super().__init__()
        self.frontend_type = frontend
        self.K = K
        self.use_adversary = use_adversary
        self.num_subjects = num_subjects

        # Frontend
        if frontend == "sinc":
            self.frontend = FixedSincFilterbank(K=K)
        elif frontend == "uvmd":
            self.uvmd = UVMDBlock(K=K, L=L_LAYERS,
                                  alpha_init=ALPHA_INIT, tau_init=TAU_INIT)
        else:
            raise ValueError(f"Unknown frontend: {frontend}")

        # Spectral Content Gate
        self.scg = SpectralContentGate(K=K, init_gate=gate_init,
                                       fixed_gate=fixed_gate)

        # Per-band encoder
        self.encoder = PerBandEncoder(K=K)

        # Gesture classifier
        self.classifier = nn.Sequential(
            nn.Linear(K * FEAT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

        # Subject adversary (with gradient reversal)
        if use_adversary:
            self.grl = GradientReversalLayer()
            self.adversary = nn.Sequential(
                nn.Linear(K * FEAT_DIM, ADV_HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(ADV_HIDDEN, num_subjects),
            )

    def set_grl_alpha(self, alpha: float):
        """Set gradient reversal scaling factor."""
        if self.use_adversary:
            self.grl.set_alpha(alpha)

    def forward(self, x: torch.Tensor, return_adversary: bool = False):
        """
        Forward pass.

        Args:
            x: (B, T, C) raw EMG windows
            return_adversary: if True, return dict with gesture + subject logits

        Returns:
            - If return_adversary=False: gesture logits (B, num_classes)
            - If return_adversary=True: dict with 'gesture_logits', 'subject_logits'
        """
        # Frontend decomposition
        if self.frontend_type == "sinc":
            modes = self.frontend(x)     # (B, K, T, C)
        else:
            modes = self.uvmd(x)         # (B, K, T, C)

        # Spectral Content Gate
        modes = self.scg(modes)          # (B, K, T, C)

        # Encode
        features = self.encoder(modes)   # (B, K*feat_dim)

        # Gesture classification
        gesture_logits = self.classifier(features)

        if return_adversary and self.use_adversary and self.training:
            # Subject adversary with gradient reversal
            features_rev = self.grl(features)
            subject_logits = self.adversary(features_rev)
            return {
                "gesture_logits": gesture_logits,
                "subject_logits": subject_logits,
            }

        return gesture_logits

    def spectral_overlap_penalty(self, sigma: float = OVERLAP_SIGMA):
        if self.frontend_type == "uvmd":
            return self.uvmd.spectral_overlap_penalty(sigma=sigma)
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_gate_values(self) -> Dict[str, list]:
        return self.scg.get_learned_params()

    def get_uvmd_params(self) -> Dict:
        if self.frontend_type == "uvmd":
            with torch.no_grad():
                return {
                    "omega_k": self.uvmd.omega.cpu().numpy().tolist(),
                }
        return {}


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


def build_model(variant: str, num_classes: int, num_subjects: int) -> SCGNet:
    if variant == "A":
        return SCGNet(num_classes, num_subjects, frontend="sinc",
                      use_adversary=True)
    elif variant == "B":
        return SCGNet(num_classes, num_subjects, frontend="uvmd",
                      use_adversary=True)
    elif variant == "C":
        return SCGNet(num_classes, num_subjects, frontend="sinc",
                      use_adversary=False)
    elif variant == "D":
        return SCGNet(num_classes, num_subjects, frontend="sinc",
                      use_adversary=False, fixed_gate=0.999)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def train_one_fold(
    variant: str,
    X_train: np.ndarray, y_train: np.ndarray, s_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    num_classes: int,
    num_subjects: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """Train a single LOSO fold and return metrics.

    s_train: subject index for each training sample (for adversary).
    """
    seed_everything(SEED)

    model = build_model(variant, num_classes, num_subjects).to(device)
    use_adversary = variant in ("A", "B")
    use_uvmd = variant == "B"

    # Separate param groups: higher LR for gate params
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "raw_gate" in name and param.requires_grad:
            gate_params.append(param)
        else:
            other_params.append(param)

    param_groups = [{"params": other_params, "lr": LR, "weight_decay": WEIGHT_DECAY}]
    if gate_params:
        param_groups.append({
            "params": gate_params, "lr": LR * GATE_LR_MULT,
            "weight_decay": 0.0
        })

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    class_w = compute_class_weights(y_train, device)
    gesture_criterion = nn.CrossEntropyLoss(weight=class_w)
    subject_criterion = nn.CrossEntropyLoss()

    # DataLoader with subject labels
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(s_train, dtype=torch.long),
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

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_adv_loss = 0.0
        n_batches = 0

        # Anneal GRL alpha: 0 → 1 over ADV_WARMUP_EPOCHS
        grl_alpha = min(1.0, epoch / max(1, ADV_WARMUP_EPOCHS))
        model.set_grl_alpha(grl_alpha)

        for xb, yb, sb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)

            if use_adversary:
                outputs = model(xb, return_adversary=True)
                gesture_loss = gesture_criterion(outputs["gesture_logits"], yb)
                adv_loss = subject_criterion(outputs["subject_logits"], sb)
                loss = gesture_loss + ADV_WEIGHT * adv_loss
                epoch_adv_loss += adv_loss.item()
            else:
                logits = model(xb)
                gesture_loss = gesture_criterion(logits, yb)
                loss = gesture_loss

            if use_uvmd:
                overlap_loss = model.spectral_overlap_penalty()
                loss = loss + OVERLAP_LAMBDA * overlap_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_adv_loss = epoch_adv_loss / max(n_batches, 1) if use_adversary else 0.0

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
            gate_info = ""
            gate_vals = model.scg.gate_values.detach().cpu().numpy()
            gate_str = [f"{g:.3f}" for g in gate_vals]
            gate_info = f"  γ_k={gate_str}"
            adv_str = f"  adv_loss={avg_adv_loss:.4f}  grl_α={grl_alpha:.2f}" if use_adversary else ""

            logger.info(
                f"    Epoch {epoch+1}/{EPOCHS}  "
                f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"patience={patience_counter}/{PATIENCE}{gate_info}{adv_str}"
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
        "scg_gate": model.get_gate_values(),
    }

    if use_uvmd:
        result["uvmd_omega_k"] = model.get_uvmd_params().get("omega_k", [])
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

    # Build subject_to_idx mapping (for adversary)
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    num_subjects = len(subjects) - 1  # max N-1 training subjects per fold

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
        train_subj_idx = 0
        for sid in subjects:
            if sid == test_sid:
                continue
            w, l = subj_arrays[sid]
            if len(w) > 0:
                Xs.append(w)
                ys.append(l)
                ss.append(np.full(len(l), train_subj_idx, dtype=np.int64))
                train_subj_idx += 1

        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        s_all = np.concatenate(ss, axis=0)
        num_train_subjects = train_subj_idx

        n = len(X_all)
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(n)
        n_val = max(1, int(n * VAL_RATIO))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train, y_train, s_train = X_all[train_idx], y_all[train_idx], s_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        mean_c = X_train.mean(axis=(0, 1), keepdims=True)
        std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train = (X_train - mean_c) / std_c
        X_val = (X_val - mean_c) / std_c
        X_test_norm = (X_test - mean_c) / std_c

        metrics = train_one_fold(
            variant=variant,
            X_train=X_train, y_train=y_train, s_train=s_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test_norm, y_test=y_test,
            num_classes=num_classes,
            num_subjects=num_train_subjects,
            device=device, logger=logger,
            batch_size=batch_size,
        )

        elapsed = time.time() - t0
        metrics["test_subject"] = test_sid
        metrics["time_s"] = round(elapsed, 1)
        fold_results.append(metrics)

        # Log results
        gate_str = ""
        if "scg_gate" in metrics:
            gv = metrics["scg_gate"]["gate_values"]
            gate_str = f"  γ_k={[f'{g:.3f}' for g in gv]}"
        uvmd_str = ""
        if "uvmd_omega_k" in metrics:
            uvmd_str = f"  ω={[f'{w:.3f}' for w in metrics['uvmd_omega_k']]}"

        logger.info(
            f"    Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}"
            f"{gate_str}{uvmd_str}  ({elapsed:.0f}s)"
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

    # Aggregate gate values across folds
    all_gates = [r["scg_gate"]["gate_values"] for r in fold_results if "scg_gate" in r]
    if all_gates:
        gate_arr = np.array(all_gates)
        summary["scg_gate_aggregate"] = {
            "mean_gate": gate_arr.mean(axis=0).tolist(),
            "std_gate": gate_arr.std(axis=0).tolist(),
        }
        logger.info(
            f"  SCG gate aggregate:"
            f"\n    γ_k = {[f'{g:.4f}±{s:.4f}' for g, s in zip(gate_arr.mean(0), gate_arr.std(0))]}"
        )

        # Gradient test
        g = gate_arr.mean(0)
        if g[-1] > g[0]:
            ratio = g[-1] / max(g[0], 1e-6)
            logger.info(f"    → gate gradient confirmed: γ_4/γ_1 = {ratio:.1f}×")
        else:
            logger.info(f"    → No clear gate gradient (γ_4={g[-1]:.4f}, γ_1={g[0]:.4f})")

    # Aggregate UVMD omega
    if variant == "B":
        all_omega = [r["uvmd_omega_k"] for r in fold_results if "uvmd_omega_k" in r]
        if all_omega:
            omega_arr = np.array(all_omega)
            summary["mean_omega_k"] = omega_arr.mean(axis=0).tolist()

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
        description="SCG-Net: Spectral Content Gate Network"
    )
    parser.add_argument("--full", action="store_true",
                        help="Use 20 subjects (default: 5 CI)")
    parser.add_argument("--ci", type=int, default=0,
                        help="Force CI mode (5 subjects)")
    parser.add_argument("--subjects", type=str, default="",
                        help="Comma-separated subject list override")
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        choices=VARIANT_NAMES,
                        help="Which variants to run (default: all)")
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
        f"scg_net_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logging(Path(out_dir))
    logger.info("SCG-Net: Spectral Content Gate Network")
    logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Adversary: weight={ADV_WEIGHT}, warmup={ADV_WARMUP_EPOCHS} epochs")
    logger.info(f"Gate LR multiplier: {GATE_LR_MULT}x")

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
    logger.info("  SCG-NET: RESULTS COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Variant':<8} {'Label':<40} {'F1':>10} {'Acc':>10}")
    logger.info("-" * 70)
    for v in args.variants:
        if v in all_results:
            r = all_results[v]
            f1_str = f"{r['mean_f1_macro']*100:.2f}±{r['std_f1_macro']*100:.2f}"
            acc_str = f"{r['mean_accuracy']*100:.2f}±{r['std_accuracy']*100:.2f}"
            logger.info(f"{v:<8} {VARIANT_LABELS[v]:<40} {f1_str:>10} {acc_str:>10}")

    # ── Spearman: gate vs band index ─────────────────────────────
    from scipy.stats import spearmanr, wilcoxon

    for v in ("A", "B", "C"):
        if v not in all_results:
            continue
        all_gates = [r["scg_gate"]["gate_values"]
                     for r in all_results[v]["per_subject"]
                     if "scg_gate" in r]
        if len(all_gates) >= 3:
            band_indices = list(range(K_BANDS))
            all_band_idx = band_indices * len(all_gates)
            all_gate_flat = [g for fold_g in all_gates for g in fold_g]
            rho, p_val = spearmanr(all_band_idx, all_gate_flat)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            logger.info(
                f"\n  Variant {v} — Spearman (band_index vs γ_k): "
                f"rho={rho:.3f}, p={p_val:.2e} {sig}"
            )

    # ── Wilcoxon paired tests ────────────────────────────────────
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
            return
        x = np.array([r1[s] for s in common])
        y = np.array([r2[s] for s in common])
        diff = y - x
        stat, p_val = wilcoxon(diff)
        mean_diff = diff.mean() * 100
        wins = sum(d > 0 for d in diff)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        logger.info(
            f"  {v2} vs {v1}: ΔF1={mean_diff:+.2f}pp  "
            f"Wilcoxon p={p_val:.4f} {sig}  "
            f"({wins}/{len(common)} wins)"
        )

    # SCG (A) vs no-adversary (C)
    _paired_wilcoxon("C", "A")
    # SCG (A) vs full-IN (D)
    _paired_wilcoxon("D", "A")
    # SCG-no-adv (C) vs full-IN (D)
    _paired_wilcoxon("D", "C")
    # UVMD SCG (B) vs Sinc SCG (A)
    _paired_wilcoxon("A", "B")

    # save combined results
    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
