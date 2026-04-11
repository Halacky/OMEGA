"""
Hypothesis 3: Per-Band Style Normalization vs Global — Ablation Study (LOSO)

Goal: Show that per-band style normalization outperforms global normalization,
motivated by H1 finding that normalized CV differs 10x between frequency bands.

All variants use fixed_fb decomposition (K=4 Sinc filterbank), since H2 showed
fixed ≈ learnable. The ONLY difference is the normalization layer inserted
between decomposition and per-mode CNN encoder.

Ablation variants:
  1. baseline     — no style norm (just channel z-score from train stats)
  2. global_in    — single InstanceNorm1d after concat of all modes
  3. per_band_in  — separate InstanceNorm1d per frequency band
  4. per_band_mix — per-band MixStyle (mix style statistics between samples)
  5. adaptive_in  — learnable per-band IN strength (gamma_k per band)

Protocol: E1 only (gestures 1-10, no REST), strict LOSO, 20 subjects.
Backbone: same K-branch CNN encoder as H2.

Usage:
  python experiments/h3_style_normalization_ablation_loso.py --variant all --ci
  python experiments/h3_style_normalization_ablation_loso.py --variant all --full
  python experiments/h3_style_normalization_ablation_loso.py --variant per_band_in --full
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from utils.logging import setup_logging, seed_everything

# ═════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "h3_style_normalization"
EXERCISES = ["E1"]
MAX_GESTURES = 10

_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [f"DB2_s{i}" for i in [1,2,3,4,5,11,12,13,14,15,26,27,28,29,30,36,37,38,39,40]]

# Shared architecture
K = 4             # number of frequency bands
FEAT_DIM = 64     # feature dim per branch
HIDDEN_DIM = 128  # classifier MLP hidden
DROPOUT = 0.3
IN_CHANNELS = 12  # Ninapro DB2

# Training
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
GRAD_CLIP = 1.0
VAL_RATIO = 0.15
SEED = 42

SAMPLING_RATE = 2000

# MixStyle
MIXSTYLE_P = 0.5   # probability of applying MixStyle
MIXSTYLE_ALPHA = 0.1  # Beta distribution parameter


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
# Decomposition frontend (fixed Sinc filterbank only — H2 showed fixed ≈ learnable)
# ═════════════════════════════════════════════════════════════════════════

class FixedSincFilterbank(nn.Module):
    """K fixed bandpass Sinc filters (non-learnable), uniformly spaced."""

    def __init__(self, K: int = 4, T: int = 200, fs: int = 2000):
        super().__init__()
        self.K = K
        self.T = T
        self.fs = fs

        nyq = fs / 2.0
        band_width = (nyq - 20) / K
        filters = []
        for k in range(K):
            lo = (20 + k * band_width) / nyq
            hi = (20 + (k + 1) * band_width) / nyq
            lo = max(lo, 0.01)
            hi = min(hi, 0.99)
            order = min(63, T - 1)
            if order % 2 == 0:
                order -= 1
            h = sp_signal.firwin(order, [lo, hi], pass_zero=False)
            padded = np.zeros(T, dtype=np.float32)
            padded[:len(h)] = h.astype(np.float32)
            filters.append(padded)

        self.register_buffer('filters', torch.tensor(np.array(filters)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> modes: (B, K, T, C)"""
        B, T, C = x.shape
        x_bc = x.permute(0, 2, 1).reshape(B * C, T)
        x_fft = torch.fft.rfft(x_bc, n=T, dim=-1)

        modes_list = []
        for k in range(self.K):
            h_fft = torch.fft.rfft(self.filters[k], n=T)
            filtered = torch.fft.irfft(x_fft * h_fft.unsqueeze(0), n=T, dim=-1)
            modes_list.append(filtered)

        modes = torch.stack(modes_list, dim=1)
        modes = modes.reshape(B, C, self.K, T).permute(0, 2, 3, 1)
        return modes


# ═════════════════════════════════════════════════════════════════════════
# Style normalization modules (the ONLY thing that changes between variants)
# ═════════════════════════════════════════════════════════════════════════

class NoStyleNorm(nn.Module):
    """Baseline — no style normalization."""
    def __init__(self, K, C):
        super().__init__()
    def forward(self, modes):
        return modes


class GlobalInstanceNorm(nn.Module):
    """Global InstanceNorm: compute mean/var over ALL bands jointly, then normalize.
    modes: (B, K, T, C) -> concat bands along T -> compute stats -> normalize each band.

    Key difference from PerBandIN: statistics are shared across all K bands,
    so a high-energy band cannot be normalized independently from a low-energy one.
    """
    def __init__(self, K, C):
        super().__init__()

    def forward(self, modes):
        B, K, T, C = modes.shape
        # Concat all bands: (B, K*T, C)
        x_cat = modes.reshape(B, K * T, C)
        # Compute global mean/var per sample, per channel over all time steps
        mu = x_cat.mean(dim=1, keepdim=True)    # (B, 1, C)
        var = x_cat.var(dim=1, keepdim=True)     # (B, 1, C)
        # Normalize each band with the SAME global statistics
        x_normed = (modes - mu.unsqueeze(1)) / (var.unsqueeze(1).sqrt() + 1e-5)
        return x_normed


class PerBandInstanceNorm(nn.Module):
    """Separate InstanceNorm per frequency band.
    Each band gets its own normalization statistics.
    """
    def __init__(self, K, C):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.InstanceNorm1d(C, affine=False) for _ in range(K)
        ])

    def forward(self, modes):
        B, K, T, C = modes.shape
        out = []
        for k in range(K):
            x_k = modes[:, k].permute(0, 2, 1)  # (B, C, T)
            x_k = self.norms[k](x_k)
            out.append(x_k.permute(0, 2, 1))  # (B, T, C)
        return torch.stack(out, dim=1)  # (B, K, T, C)


class PerBandMixStyle(nn.Module):
    """Per-band MixStyle: mix instance-level style statistics between samples.
    Applied only during training with probability p.

    Key idea: each band has different subject variability (H1),
    so mixing styles per-band creates more realistic augmentation.
    """
    def __init__(self, K, C, p=0.5, alpha=0.1):
        super().__init__()
        self.K = K
        self.p = p
        self.alpha = alpha

    def forward(self, modes):
        if not self.training or np.random.random() > self.p:
            return modes

        B, K, T, C = modes.shape
        out = []
        for k in range(K):
            x_k = modes[:, k]  # (B, T, C)
            # Compute per-sample style (mean, std over T)
            mu = x_k.mean(dim=1, keepdim=True)       # (B, 1, C)
            sigma = x_k.std(dim=1, keepdim=True) + 1e-6

            # Normalize
            x_normed = (x_k - mu) / sigma

            # Shuffle and mix
            perm = torch.randperm(B, device=x_k.device)
            mu_mix = mu[perm]
            sigma_mix = sigma[perm]

            # Beta distribution mixing coefficient
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x_k.device)
            mu_new = lam * mu + (1 - lam) * mu_mix
            sigma_new = lam * sigma + (1 - lam) * sigma_mix

            # Re-style
            x_styled = x_normed * sigma_new + mu_new
            out.append(x_styled)

        return torch.stack(out, dim=1)


class AdaptivePerBandIN(nn.Module):
    """Per-band InstanceNorm with learnable blending strength.
    gamma_k in [0,1] controls how much style to remove per band.
    gamma_k=0 -> no normalization, gamma_k=1 -> full InstanceNorm.

    Hypothesis: low-frequency bands (more stable, H1) need less normalization,
    high-frequency bands (more variable) need more.
    """
    def __init__(self, K, C):
        super().__init__()
        self.K = K
        # Initialize with uniform 0.5 (let gradient decide)
        self.gamma_logit = nn.Parameter(torch.zeros(K))
        self.norms = nn.ModuleList([
            nn.InstanceNorm1d(C, affine=False) for _ in range(K)
        ])

    def forward(self, modes):
        B, K, T, C = modes.shape
        gamma = torch.sigmoid(self.gamma_logit)  # (K,) in [0, 1]
        out = []
        for k in range(K):
            x_k = modes[:, k]  # (B, T, C)
            x_normed = self.norms[k](x_k.permute(0, 2, 1)).permute(0, 2, 1)
            # Blend: gamma_k * normalized + (1 - gamma_k) * original
            x_out = gamma[k] * x_normed + (1 - gamma[k]) * x_k
            out.append(x_out)
        return torch.stack(out, dim=1)

    def get_learned_params(self):
        with torch.no_grad():
            gamma = torch.sigmoid(self.gamma_logit).cpu().numpy().tolist()
            return {"gamma_k": gamma}


# ═════════════════════════════════════════════════════════════════════════
# Classifier (same backbone as H2)
# ═════════════════════════════════════════════════════════════════════════

class H3Classifier(nn.Module):
    """
    Fixed filterbank -> Style norm (varies) -> K-branch CNN encoder -> MLP.
    """

    def __init__(self, frontend: nn.Module, style_norm: nn.Module,
                 K: int, in_channels: int, num_classes: int,
                 feat_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.frontend = frontend
        self.style_norm = style_norm
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
        modes = self.frontend(x)       # (B, K, T, C)
        modes = self.style_norm(modes)  # (B, K, T, C)
        feats = []
        for k in range(self.K):
            mode_k = modes[:, k].permute(0, 2, 1)  # (B, C, T)
            feats.append(self.mode_encoders[k](mode_k))
        fused = torch.cat(feats, dim=1)
        return self.classifier(fused)

    def get_learned_params(self) -> Optional[Dict]:
        if hasattr(self.style_norm, 'get_learned_params'):
            return self.style_norm.get_learned_params()
        return None


# ═════════════════════════════════════════════════════════════════════════
# Build model
# ═════════════════════════════════════════════════════════════════════════

VARIANTS = ["baseline", "global_in", "per_band_in", "per_band_mix", "adaptive_in"]

def build_model(variant: str, num_classes: int, window_size: int,
                in_channels: int = 12) -> H3Classifier:
    frontend = FixedSincFilterbank(K=K, T=window_size, fs=SAMPLING_RATE)

    if variant == "baseline":
        style_norm = NoStyleNorm(K, in_channels)
    elif variant == "global_in":
        style_norm = GlobalInstanceNorm(K, in_channels)
    elif variant == "per_band_in":
        style_norm = PerBandInstanceNorm(K, in_channels)
    elif variant == "per_band_mix":
        style_norm = PerBandMixStyle(K, in_channels, p=MIXSTYLE_P, alpha=MIXSTYLE_ALPHA)
    elif variant == "adaptive_in":
        style_norm = AdaptivePerBandIN(K, in_channels)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return H3Classifier(
        frontend=frontend, style_norm=style_norm,
        K=K, in_channels=in_channels, num_classes=num_classes,
        feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    )


# ═════════════════════════════════════════════════════════════════════════
# LOSO infrastructure
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
                device, logger):
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
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_train = running_loss / max(n_batches, 1)

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
                output_dir: Path, logger, batch_size: int = BATCH_SIZE,
                num_workers: int = 0):
    logger.info(f"\n{'#'*70}")
    logger.info(f"# Variant: {variant}")
    logger.info(f"# Subjects: {len(subjects)}")
    logger.info(f"{'#'*70}\n")

    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info(f"Device: {device}")

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
            raise

    loaded = sorted(subjects_data.keys())
    logger.info(f"Loaded {len(loaded)} subjects")

    common = None
    for subj_id in loaded:
        _, _, gw = subjects_data[subj_id]
        gids = set(gw.keys())
        common = gids if common is None else common & gids
    common_gestures = sorted(common)[:MAX_GESTURES]
    num_classes = len(common_gestures)
    logger.info(f"Common gestures: {common_gestures} ({num_classes} classes)")

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
            in_channels = X_tr.shape[2]

            seed_everything(SEED, verbose=False)
            model = build_model(variant, num_classes, window_size, in_channels)
            model = model.to(device)

            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"    Model params: {n_params:,}")

            train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
            val_ds = TensorDataset(torch.tensor(X_v), torch.tensor(y_v))
            test_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=(device.type == "cuda"))
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=(device.type == "cuda"))
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=(device.type == "cuda"))

            model = train_model(model, train_loader, val_loader, num_classes,
                                y_tr, device, logger)
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Compute summary
    valid = [s for s in per_subject if s["test_f1_macro"] is not None]
    accs = [s["test_accuracy"] for s in valid]
    f1s = [s["test_f1_macro"] for s in valid]

    summary = {
        "experiment": EXPERIMENT_NAME,
        "variant": variant,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_subjects": len(loaded),
        "subjects": loaded,
        "K": K,
        "architecture": {
            "feat_dim": FEAT_DIM,
            "hidden_dim": HIDDEN_DIM,
        },
        "results": {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
        },
        "per_subject": per_subject,
        "total_time_s": round(total_time, 1),
    }

    summary_path = variant_dir / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)

    logger.info(f"\n  Variant '{variant}': "
                f"Acc={np.mean(accs):.4f} (+/-{np.std(accs):.4f}), "
                f"F1={np.mean(f1s):.4f} (+/-{np.std(f1s):.4f})")

    return summary


# ═════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    _parser = argparse.ArgumentParser(description="H3: Style normalization ablation")
    _parser.add_argument("--variant", type=str, default="all",
                         help=f"Variant to run: {VARIANTS} or 'all'")
    _parser.add_argument("--data_dir", type=str, default="data")
    _parser.add_argument("--output_dir", type=str, default=None)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true", help="Use CI subjects (5)")
    _parser.add_argument("--full", action="store_true", help="Use full 20 subjects")
    _parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    _parser.add_argument("--num_workers", type=int, default=0)
    _args, _ = _parser.parse_known_args()

    # Determine subjects
    if _args.subjects:
        subjects = [s.strip() for s in _args.subjects.split(",")]
    elif _args.ci:
        subjects = _CI_SUBJECTS
    elif _args.full:
        subjects = _FULL_SUBJECTS
    else:
        subjects = _CI_SUBJECTS  # safe default

    base_dir = ROOT / _args.data_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(_args.output_dir) if _args.output_dir else \
        ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"H3 Style Normalization Ablation")
    logger.info(f"  Subjects: {len(subjects)} — {subjects}")
    logger.info(f"  Output:   {output_dir}")
    logger.info(f"  Batch:    {_args.batch_size}, Workers: {_args.num_workers}")

    # Determine which variants to run
    if _args.variant == "all":
        variants_to_run = VARIANTS
    else:
        variants_to_run = [v.strip() for v in _args.variant.split(",")]
        for v in variants_to_run:
            if v not in VARIANTS:
                logger.error(f"Unknown variant: {v}. Choose from {VARIANTS}")
                sys.exit(1)

    all_summaries = []
    for variant in variants_to_run:
        summary = run_variant(variant, subjects, base_dir, output_dir, logger,
                              batch_size=_args.batch_size,
                              num_workers=_args.num_workers)
        all_summaries.append(summary)

    # Comparison
    logger.info(f"\n{'='*70}")
    logger.info("H3 STYLE NORMALIZATION — COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(f"\n{'Variant':22s} {'Acc':>6s} {'±Std':>6s}   {'F1':>6s} {'±Std':>6s} {'Time(s)':>8s}")
    logger.info("-" * 60)

    comparison = []
    for s in sorted(all_summaries, key=lambda x: x["results"]["mean_f1_macro"], reverse=True):
        r = s["results"]
        logger.info(f"{s['variant']:22s} {r['mean_accuracy']:.4f} {r['std_accuracy']:.4f}  "
                     f" {r['mean_f1_macro']:.4f} {r['std_f1_macro']:.4f} {s['total_time_s']:>8.0f}")
        comparison.append({
            "variant": s["variant"],
            "acc": r["mean_accuracy"],
            "acc_std": r["std_accuracy"],
            "f1": r["mean_f1_macro"],
            "f1_std": r["std_f1_macro"],
            "time": s["total_time_s"],
        })

    comp_path = output_dir / "comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\nComparison saved to {comp_path}")
    logger.info(f"\nH3 Ablation complete!")


if __name__ == "__main__":
    main()
