"""
Hypothesis 2: Learnable vs Fixed Frequency Decomposition — Ablation Study (LOSO)

Goal: Show that end-to-end optimised decomposition (UVMD) outperforms fixed
approaches, and that ANY frequency decomposition beats no decomposition.

Ablation variants (same CNN classifier backbone for all):
  1. none        — raw sEMG -> CNN classifier (no decomposition)
  2. fixed_fb    — K fixed uniform Sinc bandpass filters -> K-branch CNN
  3. fixed_vmd   — classical VMD (hand-tuned alpha, iterative) -> K-branch CNN
  4. uvmd        — learnable Unfolded VMD (ADMM unrolled) -> K-branch CNN
  5. uvmd_overlap — UVMD + spectral overlap penalty -> K-branch CNN

The CNN classifier (per-mode encoder + concat + MLP) is identical across
variants 2-5.  Variant 1 uses a single-branch version with K*feat_dim capacity.

Protocol: E1 only (gestures 1-10, no REST), strict LOSO.
Standardisation: per-channel z-score from train stats only.

Usage:
  python experiments/h2_ablation_decomposition_loso.py --variant uvmd --ci
  python experiments/h2_ablation_decomposition_loso.py --variant none --ci
  python experiments/h2_ablation_decomposition_loso.py --variant all --ci
  python experiments/h2_ablation_decomposition_loso.py --variant all --subjects DB2_s1,DB2_s2,...
"""

import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from utils.logging import setup_logging, seed_everything
from models.uvmd_classifier import UVMDBlock

# ═════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "h2_ablation_decomposition"
EXERCISES = ["E1"]
MAX_GESTURES = 10

_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [f"DB2_s{i}" for i in [1,2,3,4,5,11,12,13,14,15,26,27,28,29,30,36,37,38,39,40]]

# Shared architecture
K = 4             # number of modes / filter banks
FEAT_DIM = 64     # feature dim per branch
HIDDEN_DIM = 128  # classifier MLP hidden
DROPOUT = 0.3
IN_CHANNELS = 12  # Ninapro DB2

# Training (identical for all variants)
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
GRAD_CLIP = 1.0
VAL_RATIO = 0.15
SEED = 42

# UVMD-specific
UVMD_LAYERS = 8
ALPHA_INIT = 2000.0
TAU_INIT = 0.01
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 0.05

# VMD-specific (classical)
VMD_ALPHA = 2000
VMD_TAU = 0.0
VMD_DC = 0
VMD_TOL = 1e-7
VMD_MAX_ITER = 500

SAMPLING_RATE = 2000


# ═════════════════════════════════════════════════════════════════════════
# Helper: grouped_to_arrays
# ═════════════════════════════════════════════════════════════════════════

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
    gesture_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if gesture_ids is None:
        gesture_ids = sorted(grouped_windows.keys())
    all_windows, all_labels = [], []
    for gid in gesture_ids:
        if gid not in grouped_windows:
            continue
        for rep_arr in grouped_windows[gid]:
            if isinstance(rep_arr, np.ndarray) and rep_arr.ndim == 3 and len(rep_arr) > 0:
                all_windows.append(rep_arr)
                all_labels.append(np.full(len(rep_arr), gid, dtype=np.int64))
    if not all_windows:
        return np.empty((0, 1, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_windows, axis=0).astype(np.float32), np.concatenate(all_labels, axis=0)


# ═════════════════════════════════════════════════════════════════════════
# Frontend modules (the ONLY thing that changes between variants)
# ═════════════════════════════════════════════════════════════════════════

class FixedSincFilterbank(nn.Module):
    """K fixed bandpass Sinc filters (non-learnable), uniformly spaced."""

    def __init__(self, K: int = 4, T: int = 600, fs: int = 2000):
        super().__init__()
        self.K = K
        self.T = T
        self.fs = fs

        # Build K bandpass filters uniformly spanning 20-1000 Hz
        nyq = fs / 2.0
        band_width = (nyq - 20) / K
        filters = []
        for k in range(K):
            lo = (20 + k * band_width) / nyq
            hi = (20 + (k + 1) * band_width) / nyq
            lo = max(lo, 0.01)
            hi = min(hi, 0.99)
            # Design bandpass FIR filter
            order = min(63, T - 1)
            if order % 2 == 0:
                order -= 1
            h = sp_signal.firwin(order, [lo, hi], pass_zero=False)
            # Pad to fixed length
            padded = np.zeros(T, dtype=np.float32)
            padded[:len(h)] = h.astype(np.float32)
            filters.append(padded)

        # Register as buffer (not learnable)
        self.register_buffer('filters', torch.tensor(np.array(filters)))  # (K, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> modes: (B, K, T, C)"""
        B, T, C = x.shape
        # Apply each filter via FFT convolution per channel
        x_bc = x.permute(0, 2, 1).reshape(B * C, T)  # (B*C, T)
        x_fft = torch.fft.rfft(x_bc, n=T, dim=-1)    # (B*C, T_rfft)

        modes_list = []
        for k in range(self.K):
            h_fft = torch.fft.rfft(self.filters[k], n=T)  # (T_rfft,)
            filtered = torch.fft.irfft(x_fft * h_fft.unsqueeze(0), n=T, dim=-1)
            modes_list.append(filtered)

        modes = torch.stack(modes_list, dim=1)  # (B*C, K, T)
        modes = modes.reshape(B, C, self.K, T).permute(0, 2, 3, 1)  # (B, K, T, C)
        return modes


class ClassicalVMD(nn.Module):
    """Classical VMD (non-learnable, iterative, on CPU). Cached per-batch."""

    def __init__(self, K: int = 4, alpha: float = 2000, tau: float = 0.0,
                 tol: float = 1e-7, max_iter: int = 500):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter

    def _vmd_1d(self, signal_1d: np.ndarray) -> np.ndarray:
        """Run VMD on a single 1D signal. Returns (K, T)."""
        T = len(signal_1d)
        T_rfft = T // 2 + 1
        freqs = np.linspace(0, 0.5, T_rfft)

        f_hat = np.fft.rfft(signal_1d)

        u_hat = np.zeros((self.K, T_rfft), dtype=complex)
        omega = np.linspace(0.05, 0.45, self.K)
        lambda_hat = np.zeros(T_rfft, dtype=complex)

        for _ in range(self.max_iter):
            u_hat_old = u_hat.copy()
            u_sum = u_hat.sum(axis=0)

            for k in range(self.K):
                sum_other = u_sum - u_hat[k]
                numer = f_hat - sum_other + lambda_hat / 2
                denom = 1 + 2 * self.alpha * (freqs - omega[k]) ** 2
                u_hat[k] = numer / denom

                # Update centre frequency
                if np.sum(np.abs(u_hat[k]) ** 2) > 1e-12:
                    omega[k] = np.sum(freqs * np.abs(u_hat[k]) ** 2) / np.sum(np.abs(u_hat[k]) ** 2)

            u_sum_new = u_hat.sum(axis=0)
            lambda_hat = lambda_hat + self.tau * (f_hat - u_sum_new)

            # Convergence check
            diff = np.sum(np.abs(u_hat - u_hat_old) ** 2)
            if diff < self.tol:
                break

        modes = np.zeros((self.K, T))
        for k in range(self.K):
            modes[k] = np.fft.irfft(u_hat[k], n=T)
        return modes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> modes: (B, K, T, C)"""
        B, T, C = x.shape
        x_np = x.detach().cpu().numpy()

        modes_all = np.zeros((B, self.K, T, C), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                modes_all[b, :, :, c] = self._vmd_1d(x_np[b, :, c])

        return torch.tensor(modes_all, dtype=x.dtype, device=x.device)


class IdentityFrontend(nn.Module):
    """No decomposition — passes raw signal as a single 'mode'."""

    def __init__(self, K: int = 1):
        super().__init__()
        self.K = K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, 1, T, C)"""
        return x.unsqueeze(1)


# ═════════════════════════════════════════════════════════════════════════
# Unified classifier (shared across all variants)
# ═════════════════════════════════════════════════════════════════════════

class AblationClassifier(nn.Module):
    """
    Shared classifier backbone for all decomposition variants.

    Frontend (interchangeable):
      raw EMG (B, T, C) -> modes (B, K, T, C)

    Backend (fixed):
      K × per-mode 1-D CNN encoder -> feat (B, K*feat_dim)
      -> Linear classifier -> (B, num_classes)
    """

    def __init__(self, frontend: nn.Module, K: int, in_channels: int,
                 num_classes: int, feat_dim: int = 64, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.frontend = frontend
        self.K = K

        self.mode_encoders = nn.ModuleList([
            self._make_encoder(in_channels, feat_dim) for _ in range(K)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(K * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _make_encoder(in_channels: int, feat_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        modes = self.frontend(x)   # (B, K, T, C)
        feats = []
        for k in range(self.K):
            mode_k = modes[:, k].permute(0, 2, 1)   # (B, C, T)
            feats.append(self.mode_encoders[k](mode_k))
        fused = torch.cat(feats, dim=1)
        return self.classifier(fused)

    def spectral_overlap_penalty(self, sigma: float = 0.05) -> torch.Tensor:
        if hasattr(self.frontend, 'spectral_overlap_penalty'):
            return self.frontend.spectral_overlap_penalty(sigma)
        return torch.tensor(0.0)

    def get_learned_params(self) -> Optional[Dict]:
        if hasattr(self.frontend, 'omega'):
            with torch.no_grad():
                return {"omega_k": self.frontend.omega.cpu().numpy().tolist()}
        return None


# ═════════════════════════════════════════════════════════════════════════
# Build model for a given variant
# ═════════════════════════════════════════════════════════════════════════

def build_model(variant: str, num_classes: int, window_size: int,
                in_channels: int = 12) -> Tuple[AblationClassifier, bool]:
    """
    Returns (model, uses_overlap_penalty).
    in_channels determined dynamically from data shape.
    """
    if variant == "none":
        frontend = IdentityFrontend(K=1)
        model = AblationClassifier(
            frontend, K=1, in_channels=in_channels,
            num_classes=num_classes, feat_dim=K * FEAT_DIM,  # same total capacity
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        )
        return model, False

    elif variant == "fixed_fb":
        frontend = FixedSincFilterbank(K=K, T=window_size, fs=SAMPLING_RATE)
        model = AblationClassifier(
            frontend, K=K, in_channels=in_channels,
            num_classes=num_classes, feat_dim=FEAT_DIM,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        )
        return model, False

    elif variant == "fixed_vmd":
        frontend = ClassicalVMD(K=K, alpha=VMD_ALPHA, tau=VMD_TAU,
                                tol=VMD_TOL, max_iter=VMD_MAX_ITER)
        model = AblationClassifier(
            frontend, K=K, in_channels=in_channels,
            num_classes=num_classes, feat_dim=FEAT_DIM,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        )
        return model, False

    elif variant == "uvmd":
        frontend = UVMDBlock(K=K, L=UVMD_LAYERS, alpha_init=ALPHA_INIT, tau_init=TAU_INIT)
        model = AblationClassifier(
            frontend, K=K, in_channels=in_channels,
            num_classes=num_classes, feat_dim=FEAT_DIM,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        )
        return model, False

    elif variant == "uvmd_overlap":
        frontend = UVMDBlock(K=K, L=UVMD_LAYERS, alpha_init=ALPHA_INIT, tau_init=TAU_INIT)
        model = AblationClassifier(
            frontend, K=K, in_channels=in_channels,
            num_classes=num_classes, feat_dim=FEAT_DIM,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        )
        return model, True

    else:
        raise ValueError(f"Unknown variant: {variant}")


# ═════════════════════════════════════════════════════════════════════════
# LOSO infrastructure (reused from exp_93 pattern)
# ═════════════════════════════════════════════════════════════════════════

def build_loso_splits(subjects_data, train_subjects, test_subject,
                      common_gestures, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    gesture_to_class = {g: i for i, g in enumerate(common_gestures)}

    train_wins, train_labs = [], []
    for subj_id in sorted(train_subjects):
        if subj_id not in subjects_data:
            continue
        _, _, grouped = subjects_data[subj_id]
        wins, gid_labels = grouped_to_arrays(grouped, common_gestures)
        if len(wins) == 0:
            continue
        cls_labels = np.array([gesture_to_class[g] for g in gid_labels], dtype=np.int64)
        train_wins.append(wins)
        train_labs.append(cls_labels)

    X_all = np.concatenate(train_wins, axis=0)
    y_all = np.concatenate(train_labs, axis=0)
    n = len(X_all)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    X_train, y_train = X_all[idx[n_val:]], y_all[idx[n_val:]]
    X_val, y_val = X_all[idx[:n_val]], y_all[idx[:n_val]]

    _, _, test_grouped = subjects_data[test_subject]
    X_test, test_gids = grouped_to_arrays(test_grouped, common_gestures)
    y_test = np.array([gesture_to_class[g] for g in test_gids], dtype=np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test, gesture_to_class


def standardize_channels(X_train, X_val, X_test):
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c = np.maximum(X_train.std(axis=(0, 1), keepdims=True), 1e-8)
    return (X_train - mean_c) / std_c, (X_val - mean_c) / std_c, (X_test - mean_c) / std_c


def train_model(model, train_loader, val_loader, num_classes, y_train,
                device, use_overlap, logger):
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    ce_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = ce_criterion(logits, y_b)
            if use_overlap:
                loss = loss + OVERLAP_LAMBDA * model.spectral_overlap_penalty(OVERLAP_SIGMA)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_train = running_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss, val_correct, val_total, n_vb = 0.0, 0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                val_loss += ce_criterion(logits, y_b).item()
                n_vb += 1
                val_correct += (logits.argmax(1) == y_b).sum().item()
                val_total += len(y_b)

        avg_val = val_loss / max(n_vb, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(avg_val)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"    Ep {epoch+1:3d}/{EPOCHS} | train={avg_train:.4f} "
                        f"val={avg_val:.4f} val_acc={val_acc:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(device))
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y_b.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ═════════════════════════════════════════════════════════════════════════
# Main LOSO loop
# ═════════════════════════════════════════════════════════════════════════

def run_variant(variant: str, subjects: List[str], base_dir: Path,
                output_dir: Path, logger):
    """Run full LOSO evaluation for one decomposition variant."""
    logger.info(f"\n{'#'*70}")
    logger.info(f"# Variant: {variant}")
    logger.info(f"# Subjects: {len(subjects)}")
    logger.info(f"{'#'*70}\n")

    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    proc_cfg = ProcessingConfig(
        window_size=200, window_overlap=100, sampling_rate=SAMPLING_RATE,
        segment_edge_margin=0.1)
    multi_loader = MultiSubjectLoader(proc_cfg, logger, use_gpu=False,
                                       use_improved_processing=True)

    subjects_data = {}
    for subj_id in subjects:
        try:
            result = multi_loader.load_subject(
                base_dir=base_dir, subject_id=subj_id,
                exercise="E1", include_rest=False)
            subjects_data[subj_id] = result
        except Exception as e:
            logger.error(f"Failed to load {subj_id}: {e}")

    loaded = sorted(subjects_data.keys())
    logger.info(f"Loaded {len(loaded)} subjects")

    # Common gestures
    common = None
    for subj_id in loaded:
        _, _, gw = subjects_data[subj_id]
        gids = set(gw.keys())
        common = gids if common is None else common & gids
    common_gestures = sorted(common)[:MAX_GESTURES]
    num_classes = len(common_gestures)
    logger.info(f"Common gestures: {common_gestures} ({num_classes} classes)")

    # LOSO
    per_subject = []
    total_start = time.time()

    for test_subj in loaded:
        train_subjs = [s for s in loaded if s != test_subj]
        fold_start = time.time()

        logger.info(f"\n  Fold: test={test_subj}, train={len(train_subjs)} subjects")

        try:
            X_tr, y_tr, X_v, y_v, X_te, y_te, g2c = build_loso_splits(
                subjects_data, train_subjs, test_subj, common_gestures,
                val_ratio=VAL_RATIO, seed=SEED)
            X_tr, X_v, X_te = standardize_channels(X_tr, X_v, X_te)

            logger.info(f"    Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

            window_size = X_tr.shape[1]

            seed_everything(SEED, verbose=False)
            in_channels = X_tr.shape[2]
            model, use_overlap = build_model(variant, num_classes, window_size, in_channels)
            model = model.to(device)

            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"    Model params: {n_params:,}")

            train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
            val_ds = TensorDataset(torch.tensor(X_v), torch.tensor(y_v))
            test_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))

            # For fixed_vmd, use smaller batch to manage CPU computation
            bs = 16 if variant == "fixed_vmd" else BATCH_SIZE

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

            model = train_model(model, train_loader, val_loader, num_classes,
                                y_tr, device, use_overlap, logger)
            metrics = evaluate_model(model, test_loader, device)

            fold_time = time.time() - fold_start
            learned = model.get_learned_params()

            result = {
                "test_subject": test_subj,
                "test_accuracy": metrics["accuracy"],
                "test_f1_macro": metrics["f1_macro"],
                "train_time_s": round(fold_time, 1),
                "n_params": n_params,
            }
            if learned is not None:
                result["learned_params"] = learned

            per_subject.append(result)

            logger.info(f"    -> Acc={metrics['accuracy']:.4f}, "
                        f"F1={metrics['f1_macro']:.4f}, "
                        f"Time={fold_time:.0f}s")

        except Exception as e:
            logger.error(f"    Fold {test_subj} FAILED: {e}")
            logger.error(traceback.format_exc())
            per_subject.append({
                "test_subject": test_subj,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    valid = [r for r in per_subject if r["test_f1_macro"] is not None]
    accs = [r["test_accuracy"] for r in valid]
    f1s = [r["test_f1_macro"] for r in valid]

    summary = {
        "experiment": EXPERIMENT_NAME,
        "variant": variant,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_subjects": len(loaded),
        "subjects": loaded,
        "K": K if variant != "none" else 1,
        "architecture": {
            "feat_dim": FEAT_DIM if variant != "none" else K * FEAT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "uvmd_layers": UVMD_LAYERS if "uvmd" in variant else None,
        },
        "results": {
            "mean_accuracy": float(np.mean(accs)) if accs else None,
            "std_accuracy": float(np.std(accs)) if accs else None,
            "mean_f1_macro": float(np.mean(f1s)) if f1s else None,
            "std_f1_macro": float(np.std(f1s)) if f1s else None,
        },
        "per_subject": per_subject,
        "total_time_s": round(time.time() - total_start, 1),
    }

    with open(variant_dir / "loso_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  Variant '{variant}': "
                f"Acc={summary['results']['mean_accuracy']:.4f} "
                f"(+/-{summary['results']['std_accuracy']:.4f}), "
                f"F1={summary['results']['mean_f1_macro']:.4f} "
                f"(+/-{summary['results']['std_f1_macro']:.4f})")

    return summary


# ═════════════════════════════════════════════════════════════════════════
# Comparison table generation
# ═════════════════════════════════════════════════════════════════════════

def generate_comparison(results: Dict[str, Dict], output_dir: Path, logger):
    """Generate comparison table across variants."""
    logger.info(f"\n{'='*70}")
    logger.info("H2 ABLATION RESULTS — COMPARISON")
    logger.info(f"{'='*70}")

    rows = []
    for variant, summary in sorted(results.items()):
        r = summary["results"]
        if r["mean_f1_macro"] is None:
            continue
        rows.append({
            "variant": variant,
            "acc": r["mean_accuracy"],
            "acc_std": r["std_accuracy"],
            "f1": r["mean_f1_macro"],
            "f1_std": r["std_f1_macro"],
            "time": summary["total_time_s"],
        })

    rows.sort(key=lambda x: x["f1"], reverse=True)

    logger.info(f"\n{'Variant':<16} {'Acc':>8} {'±Std':>7} {'F1':>8} {'±Std':>7} {'Time(s)':>8}")
    logger.info("-" * 60)
    for r in rows:
        logger.info(f"{r['variant']:<16} {r['acc']:>8.4f} {r['acc_std']:>7.4f} "
                     f"{r['f1']:>8.4f} {r['f1_std']:>7.4f} {r['time']:>8.0f}")

    # Save comparison
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"\nComparison saved to {output_dir / 'comparison.json'}")


# ═════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="H2: Ablation - decomposition variants")
    parser.add_argument("--variant", type=str, default="all",
                        choices=["none", "fixed_fb", "fixed_vmd", "uvmd", "uvmd_overlap", "all"],
                        help="Which decomposition variant to run")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--ci", action="store_true", help="Use CI subjects (5)")
    parser.add_argument("--full", action="store_true", help="Use full 20 subjects")
    _args, _ = parser.parse_known_args()

    # Subject selection
    if _args.subjects:
        subjects = [s.strip() for s in _args.subjects.split(",")]
    elif _args.ci:
        subjects = _CI_SUBJECTS
    elif _args.full:
        subjects = _FULL_SUBJECTS
    else:
        subjects = _CI_SUBJECTS  # safe default

    # Paths
    base_dir = Path(_args.data_dir)
    if not base_dir.is_absolute():
        base_dir = ROOT / _args.data_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if _args.output_dir:
        output_dir = Path(_args.output_dir)
    else:
        output_dir = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"H2 Ablation: decomposition variants")
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Output: {output_dir}")

    # Determine variants to run
    if _args.variant == "all":
        variants = ["none", "fixed_fb", "uvmd", "uvmd_overlap"]
        # fixed_vmd is very slow — only include if explicitly requested
        logger.info("NOTE: 'fixed_vmd' excluded from 'all' due to extreme runtime. "
                     "Use --variant fixed_vmd to run it separately.")
    else:
        variants = [_args.variant]

    results = {}
    for variant in variants:
        try:
            summary = run_variant(variant, subjects, base_dir, output_dir, logger)
            results[variant] = summary
        except Exception as e:
            logger.error(f"Variant '{variant}' FAILED: {e}")
            logger.error(traceback.format_exc())

    if len(results) > 1:
        generate_comparison(results, output_dir, logger)

    logger.info("\nH2 Ablation complete!")


if __name__ == "__main__":
    main()
