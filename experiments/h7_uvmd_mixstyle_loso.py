#!/usr/bin/env python3
"""
H7: UVMD + MixStyle — Can the best decomposition benefit from style mixing?

Variants:
  E: UVMD only       — Learnable VMD (K=4 modes, L=8 ADMM iterations) → per-mode CNN
  F: UVMD + MixStyle — Same + Per-band MixStyle augmentation (training only)

Motivation:
  - H6 showed Sinc FB + MixStyle = best fixed-decomposition pipeline (F1=37.19%)
  - exp_93 showed UVMD alone = 40.22% F1 (best overall)
  - Question: does MixStyle help learnable decomposition too?

Architecture:
  Raw EMG (B, T, C)
      │
  UVMDBlock (K=4 modes, L=8 unrolled ADMM)
      │     learnable: alpha (L,K), tau (L,), omega (K,)
      ↓
  [PerBandMixStyle — variant F only, training only]
      │
  K × per-mode 1-D CNN encoder → feat (B, K, feat_dim)
      │
  Concatenate → (B, K·feat_dim) → MLP classifier → (B, num_classes)

Loss: CrossEntropy + lambda_overlap × spectral_overlap_penalty(omega)

LOSO: strict leave-one-subject-out, train stats only for standardisation.
      Exercise E1 only (10 gestures, IDs 8-17).

Usage:
  python experiments/h7_uvmd_mixstyle_loso.py                  # 5 CI subjects
  python experiments/h7_uvmd_mixstyle_loso.py --full           # 20 subjects
  python experiments/h7_uvmd_mixstyle_loso.py --variants E     # UVMD only
  python experiments/h7_uvmd_mixstyle_loso.py --variants F     # UVMD+MixStyle only
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
from sklearn.metrics import accuracy_score, f1_score

# ── project imports ──────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from models.uvmd_classifier import UVMDBlock
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

# UVMD architecture
K_MODES = 4
L_LAYERS = 8
ALPHA_INIT = 2000.0
TAU_INIT = 0.01
FEAT_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.3

# Training
EPOCHS = 80
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 15
LR_PATIENCE = 5
LR_FACTOR = 0.5
VAL_RATIO = 0.15

# Regularisation
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 0.05

# MixStyle
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

VARIANT_NAMES = ["E", "F"]
VARIANT_LABELS = {
    "E": "UVMD (learnable decomp.)",
    "F": "UVMD + Per-band MixStyle",
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

# ── Per-band MixStyle (same as H6) ──────────────────────────────────
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


class IdentityStyleNorm(nn.Module):
    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        return modes


# ── Per-band CNN encoder (same architecture as H6 and exp_93) ──────
class PerBandEncoder(nn.Module):
    """K parallel 3-layer Conv1d encoders."""

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
#  UVMD + MixStyle Model
# ═════════════════════════════════════════════════════════════════════
class UVMDMixStyleModel(nn.Module):
    """
    UVMD-based EMG classifier with optional per-band MixStyle.

    E: UVMDBlock → IdentityNorm → PerBandEncoder → MLP
    F: UVMDBlock → PerBandMixStyle → PerBandEncoder → MLP
    """

    def __init__(self, variant: str, num_classes: int):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.K = K_MODES

        # ── UVMD decomposition (learnable) ────────────────────────
        self.uvmd = UVMDBlock(
            K=K_MODES, L=L_LAYERS,
            alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
        )

        # ── style norm ────────────────────────────────────────────
        if variant == "F":
            self.style_norm = PerBandMixStyle(K=self.K)
        else:
            self.style_norm = IdentityStyleNorm()

        # ── encoder ───────────────────────────────────────────────
        self.encoder = PerBandEncoder(K=self.K)

        # ── classifier ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.K * FEAT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, num_classes)"""
        modes = self.uvmd(x)                # (B, K, T, C)
        modes = self.style_norm(modes)      # (B, K, T, C)
        flat = self.encoder(modes)          # (B, K*feat_dim)
        return self.classifier(flat)        # (B, num_classes)

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
    """Train a single LOSO fold and return metrics + UVMD analysis."""
    seed_everything(SEED)

    model = UVMDMixStyleModel(variant=variant, num_classes=num_classes).to(device)

    # Log initial omega
    init_omega = model.uvmd.omega.detach().cpu().numpy().tolist()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR,
    )

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # DataLoader for fast GPU transfer
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

    for epoch in range(EPOCHS):
        # ── train ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            ce_loss = criterion(logits, yb)
            overlap_loss = model.spectral_overlap_penalty()
            loss = ce_loss + OVERLAP_LAMBDA * overlap_loss

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
        for ts in range(0, len(Xte), batch_size):
            test_logits.append(model(Xte[ts: ts + batch_size]))
        preds = torch.cat(test_logits).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    # UVMD analysis
    learned = model.get_learned_uvmd_params()
    overlap_val = model.spectral_overlap_penalty().item()

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "init_omega_k": init_omega,
        "final_omega_k": learned["omega_k"],
        "overlap_penalty": overlap_val,
    }


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

        # train / val split
        n = len(X_all)
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(n)
        n_val = max(1, int(n * VAL_RATIO))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # channel standardisation (train stats only)
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

        logger.info(
            f"    Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}  "
            f"omega={[f'{w:.3f}' for w in metrics['final_omega_k']]}  "
            f"({elapsed:.0f}s)"
        )

    # ── aggregate ────────────────────────────────────────────────
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]
    omega_all = np.array([r["final_omega_k"] for r in fold_results])

    summary = {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "n_subjects": len(fold_results),
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "std_f1_macro": float(np.std(f1s)) if f1s else 0.0,
        "mean_omega_k": omega_all.mean(axis=0).tolist() if len(omega_all) > 0 else [],
        "std_omega_k": omega_all.std(axis=0).tolist() if len(omega_all) > 0 else [],
        "per_subject": fold_results,
    }

    logger.info(
        f"\n  >>> Variant {variant}: "
        f"Acc={summary['mean_accuracy']*100:.2f}±{summary['std_accuracy']*100:.2f}  "
        f"F1={summary['mean_f1_macro']*100:.2f}±{summary['std_f1_macro']*100:.2f}"
    )
    if summary["mean_omega_k"]:
        logger.info(
            f"      Mean omega_k: {[f'{w:.3f}' for w in summary['mean_omega_k']]}"
        )
    return summary


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="H7: UVMD + MixStyle ablation")
    parser.add_argument("--full", action="store_true",
                        help="Use 20 subjects (default: 5 CI)")
    parser.add_argument("--ci", type=int, default=0,
                        help="Force CI mode (5 subjects)")
    parser.add_argument("--subjects", type=str, default="",
                        help="Comma-separated subject list override")
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        choices=VARIANT_NAMES,
                        help="Which variants to run (default: E F)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size (default: 512)")
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
        PROJECT_ROOT, "experiments_output", f"h7_uvmd_mixstyle_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logging(Path(out_dir))
    logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"UVMD: K={K_MODES}, L={L_LAYERS}, alpha_init={ALPHA_INIT}")
    logger.info(f"MixStyle: p={MIXSTYLE_P}, alpha={MIXSTYLE_ALPHA}")
    logger.info(f"Overlap reg: lambda={OVERLAP_LAMBDA}, sigma={OVERLAP_SIGMA}")
    logger.info(f"Batch size: {args.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── run each variant ─────────────────────────────────────────
    all_results = {}
    for v in args.variants:
        result = run_variant(v, ALL_SUBJECTS, base_dir, device, logger,
                             batch_size=args.batch_size)
        all_results[v] = result

        # save per-variant result
        with open(os.path.join(out_dir, f"variant_{v}.json"), "w") as f:
            json.dump(result, f, indent=2)

    # ── comparison table ─────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  H7: UVMD + MIXSTYLE COMPARISON")
    logger.info("=" * 70)

    # Include H6 baselines for context
    h6_baselines = {
        "A": {"f1": 32.40, "label": "Raw CNN (H6-A)"},
        "B": {"f1": 36.08, "label": "Sinc FB (H6-B)"},
        "C": {"f1": 37.19, "label": "Sinc FB + MixStyle (H6-C)"},
    }

    logger.info("  Reference (H6 baselines, 20 subjects):")
    for v, ref in h6_baselines.items():
        logger.info(f"    {v}: {ref['label']:<35s} F1={ref['f1']:.2f}%")

    logger.info("")

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
            f"  {v}: {r['label']:<35s}  "
            f"Acc={acc_mean:.2f}%  F1={f1_mean:.2f}±{f1_std:.2f}%  "
            f"Δ={delta}"
        )

    # ── Key finding ──────────────────────────────────────────────
    if len(rows) == 2:
        delta_val = rows[1]["f1"] - rows[0]["f1"]
        if delta_val > 0:
            logger.info(
                f"\n  KEY FINDING: MixStyle HELPS UVMD by +{delta_val:.2f} pp F1"
            )
        elif delta_val < -0.5:
            logger.info(
                f"\n  KEY FINDING: MixStyle HURTS UVMD by {delta_val:.2f} pp F1"
            )
        else:
            logger.info(
                f"\n  KEY FINDING: MixStyle has NEGLIGIBLE effect on UVMD ({delta_val:+.2f} pp)"
            )

    # ── save comparison ──────────────────────────────────────────
    comparison = {
        "experiment": "h7_uvmd_mixstyle",
        "timestamp": ts,
        "n_subjects": len(ALL_SUBJECTS),
        "subjects": ALL_SUBJECTS,
        "hyperparams": {
            "K_modes": K_MODES, "L_layers": L_LAYERS,
            "alpha_init": ALPHA_INIT, "tau_init": TAU_INIT,
            "feat_dim": FEAT_DIM, "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "epochs": EPOCHS, "batch_size": args.batch_size,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "overlap_lambda": OVERLAP_LAMBDA,
            "overlap_sigma": OVERLAP_SIGMA,
            "mixstyle_p": MIXSTYLE_P,
            "mixstyle_alpha": MIXSTYLE_ALPHA,
        },
        "results": all_results,
        "comparison_rows": rows,
        "h6_reference": h6_baselines,
    }
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
