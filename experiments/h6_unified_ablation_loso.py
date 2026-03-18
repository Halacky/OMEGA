#!/usr/bin/env python3
"""
H6: Unified Ablation — clean cumulative ablation on one backbone.

Variants (each adds ONE component on top of the previous):
  A: raw        — Raw EMG → K-branch CNN → classifier  (no decomposition)
  B: sinc       — + Fixed Sinc Filterbank (K=4)
  C: mixstyle   — + Per-band MixStyle augmentation
  D: cs_heads   — + Content / Style heads (gesture classifier on content only)

All variants share the SAME backbone (3-layer Conv1d per branch),
the same hyper-parameters, and the same 20-subject LOSO protocol.

Usage:
  python experiments/h6_unified_ablation_loso.py --full          # 20 subjects
  python experiments/h6_unified_ablation_loso.py                 # 5 CI subjects
  python experiments/h6_unified_ablation_loso.py --variants A B  # specific variants
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
CONTENT_DIM = 48
STYLE_DIM = 16
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

MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

VARIANT_NAMES = ["A", "B", "C", "D"]
VARIANT_LABELS = {
    "A": "Raw CNN (no decomp.)",
    "B": "+ Sinc Filterbank (K=4)",
    "C": "+ Per-band MixStyle",
    "D": "+ Content/Style heads",
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
            # zero-pad to length T
            pad_total = T - len(b)
            pad_l = pad_total // 2
            pad_r = pad_total - pad_l
            b_padded = np.pad(b, (pad_l, pad_r))
            filters.append(b_padded)

        # (K, T) — registered as buffer (no grad)
        filt_t = torch.tensor(np.stack(filters), dtype=torch.float32)
        self.register_buffer("filters", filt_t)
        # pre-compute FFT of filters
        self.register_buffer("filters_fft", torch.fft.rfft(filt_t, n=T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, K, T, C)"""
        B, T, C = x.shape
        # (B, C, T) → (B*C, T)
        x_ct = x.permute(0, 2, 1).reshape(B * C, T)
        x_fft = torch.fft.rfft(x_ct, n=T)  # (B*C, T//2+1)

        modes = []
        for k in range(self.K):
            y_fft = x_fft * self.filters_fft[k]  # element-wise
            y = torch.fft.irfft(y_fft, n=T)  # (B*C, T)
            modes.append(y)

        # (K, B*C, T) → (B, K, T, C)
        out = torch.stack(modes, dim=0)  # (K, B*C, T)
        out = out.view(self.K, B, C, T).permute(1, 0, 3, 2)  # (B, K, T, C)
        return out


# ── Identity frontend (variant A) ───────────────────────────────────
class IdentityFrontend(nn.Module):
    """Pass-through: (B, T, C) → (B, 1, T, C) — single 'mode'."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


# ── Per-band MixStyle ────────────────────────────────────────────────
class PerBandMixStyle(nn.Module):
    """Per-band style augmentation (training only)."""

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
            x_k = modes[:, k]  # (B, T, C)
            mu = x_k.mean(dim=1, keepdim=True)      # (B, 1, C)
            sigma = x_k.std(dim=1, keepdim=True) + 1e-6  # (B, 1, C)
            x_normed = (x_k - mu) / sigma

            perm = torch.randperm(B, device=x_k.device)
            lam = beta_dist.sample((B, 1, 1)).to(x_k.device)  # (B, 1, 1)
            mu_mix = lam * mu + (1 - lam) * mu[perm]
            sigma_mix = lam * sigma + (1 - lam) * sigma[perm]

            out_bands.append(x_normed * sigma_mix + mu_mix)

        return torch.stack(out_bands, dim=1)


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
        """(B, K, T, C) → (B, K, feat_dim)"""
        feats = []
        for k in range(self.K):
            x_k = modes[:, k]  # (B, T, C)
            x_k = x_k.permute(0, 2, 1)  # (B, C, T)
            feats.append(self.encoders[k](x_k))  # (B, feat_dim)
        return torch.stack(feats, dim=1)  # (B, K, feat_dim)


# ── Content / Style heads ────────────────────────────────────────────
class ContentStyleHeads(nn.Module):
    """Split per-band features into content + style branches."""

    def __init__(self, K: int, feat_dim: int = FEAT_DIM,
                 content_dim: int = CONTENT_DIM,
                 style_dim: int = STYLE_DIM):
        super().__init__()
        self.K = K
        self.content_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, content_dim), nn.ReLU())
            for _ in range(K)
        ])
        self.style_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, style_dim), nn.ReLU())
            for _ in range(K)
        ])

    def forward(
        self, band_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, K, feat_dim) → content (B, K*content_dim), style (B, K*style_dim)"""
        contents, styles = [], []
        for k in range(self.K):
            f_k = band_feats[:, k]  # (B, feat_dim)
            contents.append(self.content_heads[k](f_k))
            styles.append(self.style_heads[k](f_k))
        content = torch.cat(contents, dim=1)  # (B, K*content_dim)
        style = torch.cat(styles, dim=1)      # (B, K*style_dim)
        return content, style


# ═════════════════════════════════════════════════════════════════════
#  Unified Ablation Model
# ═════════════════════════════════════════════════════════════════════
class AblationModel(nn.Module):
    """
    Unified model for variants A–D.

    A: IdentityFrontend  + IdentityNorm  + PerBandEncoder(K=1) + flat classifier
    B: SincFilterbank     + IdentityNorm  + PerBandEncoder(K=4) + flat classifier
    C: SincFilterbank     + PerBandMixStyle + PerBandEncoder(K=4) + flat classifier
    D: SincFilterbank     + PerBandMixStyle + PerBandEncoder(K=4) + CS heads + classifier
    """

    def __init__(self, variant: str, num_classes: int):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes

        # ── frontend ─────────────────────────────────────────────
        if variant == "A":
            self.frontend = IdentityFrontend()
            self.K = 1
        else:
            self.frontend = FixedSincFilterbank(K=K_BANDS)
            self.K = K_BANDS

        # ── style norm ───────────────────────────────────────────
        if variant in ("C", "D"):
            self.style_norm = PerBandMixStyle(K=self.K)
        else:
            self.style_norm = IdentityStyleNorm()

        # ── encoder ──────────────────────────────────────────────
        self.encoder = PerBandEncoder(K=self.K)

        # ── head ─────────────────────────────────────────────────
        self.use_cs_heads = variant == "D"
        if self.use_cs_heads:
            self.cs_heads = ContentStyleHeads(K=self.K)
            clf_in = self.K * CONTENT_DIM
        else:
            clf_in = self.K * FEAT_DIM

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, num_classes)"""
        modes = self.frontend(x)         # (B, K, T, C)
        modes = self.style_norm(modes)   # (B, K, T, C)
        band_feats = self.encoder(modes) # (B, K, feat_dim)

        if self.use_cs_heads:
            content, _style = self.cs_heads(band_feats)
            return self.classifier(content)
        else:
            flat = band_feats.reshape(band_feats.size(0), -1)  # (B, K*feat_dim)
            return self.classifier(flat)


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


def train_one_fold(
    variant: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Train a single LOSO fold and return metrics."""
    seed_everything(SEED)

    model = AblationModel(variant=variant, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # numpy → tensors with DataLoader for fast GPU transfer
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ── train ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

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
            for vs in range(0, len(Xv), BATCH_SIZE):
                val_logits.append(model(Xv[vs: vs + BATCH_SIZE]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)

        # ── early stopping ───────────────────────────────────────
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
            logger.info(
                f"    Epoch {epoch+1}/{EPOCHS}  "
                f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"patience={patience_counter}/{PATIENCE}"
            )

    # ── test ─────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    Xte = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_logits = []
        for ts in range(0, len(Xte), BATCH_SIZE):
            test_logits.append(model(Xte[ts: ts + BATCH_SIZE]))
        preds = torch.cat(test_logits).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}


# ═════════════════════════════════════════════════════════════════════
#  LOSO loop
# ═════════════════════════════════════════════════════════════════════
def run_variant(
    variant: str,
    subjects: List[str],
    base_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Run full LOSO for a single variant, return aggregated results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Variant {variant}: {VARIANT_LABELS[variant]}")
    logger.info(f"{'='*60}")

    # ── load data ────────────────────────────────────────────────
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

    # common gestures & class mapping
    common_gestures = multi_loader.get_common_gestures(subjects_data,
                                                       max_gestures=10)
    gesture_to_class = {g: i for i, g in enumerate(sorted(common_gestures))}
    num_classes = len(gesture_to_class)

    # pre-extract per-subject arrays
    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, (_, _, gw) in subjects_data.items():
        wins, labs = grouped_to_arrays(gw)
        # filter to common gestures & remap
        mask = np.isin(labs, list(gesture_to_class.keys()))
        wins, labs = wins[mask], labs[mask]
        labs = np.array([gesture_to_class[g] for g in labs])
        subj_arrays[sid] = (wins, labs)

    # ── LOSO folds ───────────────────────────────────────────────
    fold_results = []
    for test_sid in subjects:
        t0 = time.time()
        logger.info(f"  Fold: test={test_sid}  (variant {variant})")

        X_test, y_test = subj_arrays[test_sid]
        if len(X_test) == 0:
            logger.warning(f"    Skipping {test_sid}: no test data")
            continue

        # aggregate training data
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

        # train / val split (stratified-ish: random)
        n = len(X_all)
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(n)
        n_val = max(1, int(n * VAL_RATIO))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # channel standardisation (train stats only)
        mean_c = X_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
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
        )

        elapsed = time.time() - t0
        metrics["test_subject"] = test_sid
        metrics["time_s"] = round(elapsed, 1)
        fold_results.append(metrics)

        logger.info(
            f"    Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}  "
            f"({elapsed:.0f}s)"
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
    parser = argparse.ArgumentParser(description="H6 Unified Ablation")
    parser.add_argument("--full", action="store_true",
                        help="Use 20 subjects (default: 5 CI)")
    parser.add_argument("--ci", type=int, default=0,
                        help="Force CI mode (5 subjects)")
    parser.add_argument("--subjects", type=str, default="",
                        help="Comma-separated subject list override")
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        choices=VARIANT_NAMES,
                        help="Which variants to run (default: all)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory (auto-generated if empty)")
    args, _ = parser.parse_known_args()

    # subjects
    if args.subjects:
        ALL_SUBJECTS = [s.strip() for s in args.subjects.split(",")]
    elif args.full:
        ALL_SUBJECTS = _FULL_SUBJECTS
    else:
        ALL_SUBJECTS = _CI_SUBJECTS

    # paths
    base_dir = Path(PROJECT_ROOT) / args.data_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "experiments_output", f"h6_unified_ablation_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logging(Path(out_dir))
    logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── run each variant ─────────────────────────────────────────
    all_results = {}
    for v in args.variants:
        result = run_variant(v, ALL_SUBJECTS, base_dir, device, logger)
        all_results[v] = result

        # save per-variant result
        with open(os.path.join(out_dir, f"variant_{v}.json"), "w") as f:
            json.dump(result, f, indent=2)

    # ── comparison table ─────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  UNIFIED ABLATION SUMMARY")
    logger.info("=" * 70)

    baseline_f1 = None
    rows = []
    for v in args.variants:
        if v not in all_results:
            continue
        r = all_results[v]
        f1_mean = r["mean_f1_macro"] * 100
        f1_std = r["std_f1_macro"] * 100
        acc_mean = r["mean_accuracy"] * 100

        if baseline_f1 is None:
            delta = "—"
            baseline_f1 = f1_mean
        else:
            d = f1_mean - baseline_f1
            delta = f"{d:+.2f} pp"

        rows.append({
            "variant": v,
            "label": VARIANT_LABELS[v],
            "acc": acc_mean,
            "f1": f1_mean,
            "f1_std": f1_std,
            "delta": delta,
        })
        logger.info(
            f"  {v}: {r['label']:<30s}  "
            f"Acc={acc_mean:.2f}%  F1={f1_mean:.2f}±{f1_std:.2f}%  "
            f"Δ={delta}"
        )

    # cumulative deltas (each vs previous)
    logger.info("\n  Cumulative component contributions:")
    prev_f1 = None
    for row in rows:
        if prev_f1 is not None:
            cum = row["f1"] - prev_f1
            logger.info(
                f"    {row['variant']}: {row['label']:<30s} → +{cum:.2f} pp"
            )
        prev_f1 = row["f1"]

    # ── save comparison ──────────────────────────────────────────
    comparison = {
        "experiment": "h6_unified_ablation",
        "timestamp": ts,
        "n_subjects": len(ALL_SUBJECTS),
        "subjects": ALL_SUBJECTS,
        "hyperparams": {
            "K": K_BANDS, "feat_dim": FEAT_DIM,
            "content_dim": CONTENT_DIM, "style_dim": STYLE_DIM,
            "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE, "mixstyle_p": MIXSTYLE_P,
            "mixstyle_alpha": MIXSTYLE_ALPHA,
        },
        "results": all_results,
        "comparison_rows": rows,
    }
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
