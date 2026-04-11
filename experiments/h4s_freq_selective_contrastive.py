#!/usr/bin/env python3
"""
H4s: Frequency-Selective Contrastive Learning (Confirmatory, Family B).

Tests whether restricting VICReg contrastive learning to specific frequency
bands improves cross-subject sEMG gesture recognition.  H1 showed that low
bands carry gesture information while high bands carry subject noise.

Variants
────────
  all_bands: FreqAwareVICReg with contrastive_bands=None (all 4 bands)
  low_bands: FreqAwareVICReg with contrastive_bands=[0, 1]
  high_bands: FreqAwareVICReg with contrastive_bands=[2, 3]
  weighted: PerBandVICReg with band_weights=[1.0, 0.8, 0.4, 0.2]

Protocol (per LOSO fold)
────────────────────────
  Phase 1 — SSL Pretrain (all variants):
    Data: train + val windows (no labels, no test subject)
    Epochs: 50, batch=512, lr=1e-3, cosine decay

  Phase 2 — Supervised Fine-tune:
    a) Linear probe: 10 epochs, frozen encoder
    b) Full fine-tune: 50 epochs, lr=1e-4

  Phase 3 — Evaluate:
    Data: held-out test subject
    Metrics: accuracy, macro-F1

Statistical design
──────────────────
  Family B (k=2): low_bands vs all_bands, weighted vs all_bands
  Wilcoxon signed-rank + Holm-Bonferroni correction (alpha=0.05)

Usage
─────
  python experiments/h4s_freq_selective_contrastive.py                     # CI
  python experiments/h4s_freq_selective_contrastive.py --full               # 20 subj
  python experiments/h4s_freq_selective_contrastive.py --variants all_bands low_bands
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
from models.uvmd_ssl_encoder import UVMDSSLEncoder, UVMDSSLClassifier
from models.freq_aware_vicreg import FreqAwareVICReg, PerBandVICReg
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

# SSL pretraining (optimized for speed)
SSL_EPOCHS = 30
SSL_BATCH = 1024
SSL_LR = 1e-3
SSL_WARMUP_EPOCHS = 5

# Fine-tuning (optimized for speed)
FT_EPOCHS = 40
FT_BATCH = 1024
FT_LR = 1e-4
FT_PATIENCE = 10
PROBE_EPOCHS = 5
PROBE_LR = 1e-3

# Common
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 0.05
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

VARIANT_NAMES = ["all_bands", "low_bands", "high_bands", "weighted"]
VARIANT_LABELS = {
    "all_bands": "VICReg all bands",
    "low_bands": "VICReg low bands [0,1]",
    "high_bands": "VICReg high bands [2,3]",
    "weighted": "PerBandVICReg weighted [1.0,0.8,0.4,0.2]",
}

EXPERIMENT_NAME = "h4s_freq_selective_contrastive"


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / (counts.astype(np.float64) + 1e-8)
    weights /= weights.sum()
    weights *= len(classes)
    w = torch.zeros(int(classes.max()) + 1, dtype=torch.float32, device=device)
    for c, wt in zip(classes, weights):
        w[int(c)] = wt
    return w


def cosine_schedule(epoch: int, total_epochs: int, base_lr: float, warmup: int = 5) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate_model(
    model: nn.Module,
    X: np.ndarray, y: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_list = []
        for s in range(0, len(Xt), batch_size):
            with torch.amp.autocast("cuda"):
                logits_list.append(model(Xt[s: s + batch_size]))
        preds = torch.cat(logits_list).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}


# ═════════════════════════════════════════════════════════════════════
#  SSL Pretraining
# ═════════════════════════════════════════════════════════════════════

def ssl_pretrain_vicreg(
    variant: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLEncoder:
    """VICReg pretraining with variant-specific band selection."""
    logger.info(f"  [VICReg-{variant}] Starting SSL pretraining...")

    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=False,
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    if variant == "weighted":
        ssl_model = PerBandVICReg(
            encoder=encoder,
            band_weights=[1.0, 0.8, 0.4, 0.2],
            proj_dim_per_band=512,
            overlap_lambda=OVERLAP_LAMBDA, overlap_sigma=OVERLAP_SIGMA,
        ).to(device)
    else:
        contrastive_bands = {
            "all_bands": None,
            "low_bands": [0, 1],
            "high_bands": [2, 3],
        }[variant]

        ssl_model = FreqAwareVICReg(
            encoder=encoder,
            proj_hidden=2048, proj_dim=2048,
            contrastive_bands=contrastive_bands,
            overlap_lambda=OVERLAP_LAMBDA, overlap_sigma=OVERLAP_SIGMA,
        ).to(device)

    X_all = np.concatenate([X_train, X_val], axis=0)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=SSL_BATCH, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(
        ssl_model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(SSL_EPOCHS):
        ssl_model.train()
        lr = cosine_schedule(epoch, SSL_EPOCHS, SSL_LR, SSL_WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss, details = ssl_model(xb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(ssl_model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            avg = epoch_loss / max(n_batches, 1)
            logger.info(
                f"    VICReg-{variant} ep {epoch+1}/{SSL_EPOCHS}  "
                f"loss={avg:.4f}  lr={lr:.6f}"
            )

    logger.info(f"  [VICReg-{variant}] Pretraining complete.")
    return encoder


# ═════════════════════════════════════════════════════════════════════
#  Fine-Tuning
# ═════════════════════════════════════════════════════════════════════

def finetune(
    encoder: UVMDSSLEncoder,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLClassifier:
    """Two-phase fine-tuning: linear probe then full fine-tune."""
    model = UVMDSSLClassifier(
        encoder=encoder, num_classes=num_classes,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=FT_BATCH, shuffle=True,
        num_workers=4, pin_memory=True,
        persistent_workers=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)
    scaler = torch.amp.GradScaler("cuda")

    # ── Phase 1: Linear probe (frozen encoder) ────────────────────
    model.freeze_encoder()
    probe_optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=PROBE_LR,
    )
    for epoch in range(PROBE_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            probe_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(probe_optimizer)
            scaler.update()
    logger.info(f"    Probe done ({PROBE_EPOCHS} ep)")

    # ── Phase 2: Full fine-tune (unfrozen) ────────────────────────
    model.unfreeze_encoder()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=FT_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(FT_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                ce_loss = criterion(logits, yb)
                overlap_loss = model.spectral_overlap_penalty()
                loss = ce_loss + OVERLAP_LAMBDA * overlap_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), FT_BATCH):
                with torch.amp.autocast("cuda"):
                    val_logits.append(model(Xv[vs: vs + FT_BATCH]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)
        train_avg = epoch_loss / max(n_batches, 1)
        mark = " *" if val_loss < best_val_loss else ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FT_PATIENCE:
                logger.info(f"    FT early stop at ep {epoch+1}/{FT_EPOCHS}")
                break

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"    FT ep {epoch+1}/{FT_EPOCHS}  "
                f"train={train_avg:.4f}  val={val_loss:.4f}  "
                f"pat={patience_counter}/{FT_PATIENCE}{mark}"
            )

    model.load_state_dict(best_state)
    return model


# ═════════════════════════════════════════════════════════════════════
#  Single LOSO Fold
# ═════════════════════════════════════════════════════════════════════

def train_one_fold(
    variant: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    seed_everything()

    encoder = ssl_pretrain_vicreg(variant, X_train, X_val, device, logger)
    model = finetune(
        encoder, X_train, y_train, X_val, y_val,
        num_classes, device, logger,
    )

    test_metrics = evaluate_model(model, X_test, y_test, device)
    learned = model.get_learned_uvmd_params()

    return {
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "final_omega_k": learned["omega_k"],
    }


# ═════════════════════════════════════════════════════════════════════
#  LOSO Loop
# ═════════════════════════════════════════════════════════════════════

def load_data_once(
    subjects: List[str],
    base_dir: str,
    logger: logging.Logger,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], int]:
    """Load and preprocess data once, shared across variants."""
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

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    gesture_to_class = {g: i for i, g in enumerate(sorted(common_gestures))}
    num_classes = len(gesture_to_class)

    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, (_, _, gw) in subjects_data.items():
        wins, labs = grouped_to_arrays(gw)
        mask = np.isin(labs, list(gesture_to_class.keys()))
        wins, labs = wins[mask], labs[mask]
        labs = np.array([gesture_to_class[g] for g in labs])
        subj_arrays[sid] = (wins, labs)

    del subjects_data
    return subj_arrays, num_classes


def run_variant(
    variant: str,
    subjects: List[str],
    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
    output_dir: Optional[Path] = None,
    fold_indices: Optional[List[int]] = None,
) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"  Variant: {VARIANT_LABELS[variant]}")
    logger.info(f"{'='*60}")

    # Setup folds dir for incremental saving
    folds_dir = None
    if output_dir is not None:
        folds_dir = output_dir / variant / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)

    # Determine which folds to run
    if fold_indices is not None:
        run_subjects = [(i, subjects[i]) for i in fold_indices if i < len(subjects)]
    else:
        run_subjects = list(enumerate(subjects))

    # Check for already-completed folds (resume support)
    completed_folds = set()
    if folds_dir is not None:
        for f in folds_dir.glob("fold_*.json"):
            completed_folds.add(f.stem)

    fold_results = []
    for fold_idx, test_sid in run_subjects:
        fold_key = f"fold_{fold_idx:02d}_{test_sid}"
        if fold_key in completed_folds:
            logger.info(f"  Fold {fold_idx} ({test_sid}): already done, skipping")
            # Load existing result
            with open(folds_dir / f"{fold_key}.json") as fh:
                fold_results.append(json.load(fh))
            continue

        t0 = time.time()
        logger.info(f"  Fold {fold_idx+1}/{len(subjects)} (idx={fold_idx}): test={test_sid}")

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

        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(X_all))
        n_val = max(1, int(len(X_all) * VAL_RATIO))
        X_train, y_train = X_all[perm[n_val:]], y_all[perm[n_val:]]
        X_val, y_val = X_all[perm[:n_val]], y_all[perm[:n_val]]

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
        )

        elapsed = time.time() - t0
        metrics["test_subject"] = test_sid
        metrics["fold_idx"] = fold_idx
        metrics["time_s"] = round(elapsed, 1)
        fold_results.append(metrics)

        # Incremental save
        if folds_dir is not None:
            with open(folds_dir / f"{fold_key}.json", "w") as fh:
                json.dump(metrics, fh, indent=2)

        logger.info(
            f"    Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}  ({elapsed:.0f}s)"
        )

        # Cleanup GPU memory
        torch.cuda.empty_cache()

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

    # Save variant summary
    if output_dir is not None:
        variant_dir = output_dir / variant
        variant_dir.mkdir(exist_ok=True)
        with open(variant_dir / "loso_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    logger.info(
        f"\n  >>> {VARIANT_LABELS[variant]}: "
        f"Acc={summary['mean_accuracy']*100:.2f}+-{summary['std_accuracy']*100:.2f}  "
        f"F1={summary['mean_f1_macro']*100:.2f}+-{summary['std_f1_macro']*100:.2f}"
    )
    return summary


# ═════════════════════════════════════════════════════════════════════
#  Statistical Tests
# ═════════════════════════════════════════════════════════════════════

def statistical_comparison(
    results: Dict[str, Dict],
    logger: logging.Logger,
) -> Dict:
    """Wilcoxon signed-rank tests with Holm-Bonferroni correction (Family B, k=2)."""
    from scipy import stats

    comparisons = [
        ("low_bands", "all_bands"),
        ("weighted", "all_bands"),
    ]

    stats_results = {}
    p_values = []

    for v1, v2 in comparisons:
        if v1 not in results or v2 not in results:
            continue

        f1_v1 = {r["test_subject"]: r["f1_macro"] for r in results[v1]["per_subject"]}
        f1_v2 = {r["test_subject"]: r["f1_macro"] for r in results[v2]["per_subject"]}
        common = sorted(set(f1_v1.keys()) & set(f1_v2.keys()))

        if len(common) < 5:
            continue

        x = np.array([f1_v1[s] for s in common])
        y = np.array([f1_v2[s] for s in common])
        diff = x - y

        stat, p = stats.wilcoxon(diff, alternative="two-sided")
        d = diff.mean() / (diff.std() + 1e-8)

        key = f"{v1}_vs_{v2}"
        stats_results[key] = {
            "statistic": float(stat),
            "p_uncorrected": float(p),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std()),
            "cohens_d": float(d),
            "n_subjects": len(common),
            "n_v1_wins": int((diff > 0).sum()),
        }
        p_values.append((key, p))

    # Holm-Bonferroni correction
    if p_values:
        p_values.sort(key=lambda x: x[1])
        k = len(p_values)
        for rank, (key, p_raw) in enumerate(p_values):
            p_corrected = min(1.0, p_raw * (k - rank))
            stats_results[key]["p_holm"] = float(p_corrected)
            sig = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*" if p_corrected < 0.05 else "ns"
            stats_results[key]["significance"] = sig

            logger.info(
                f"  {key}: Delta={stats_results[key]['mean_diff']*100:+.2f} pp  "
                f"d={stats_results[key]['cohens_d']:.2f}  "
                f"p_Holm={p_corrected:.4f} ({sig})"
            )

    return stats_results


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="H4s: Frequency-selective contrastive learning")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--ci", type=int, default=0)
    parser.add_argument("--subjects", type=str, default="")
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        choices=VARIANT_NAMES)
    parser.add_argument("--batch_size", type=int, default=FT_BATCH)
    parser.add_argument("--base_dir", type=str, default="data")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (0-3). None = auto-detect.")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold indices (0-based)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Shared output directory (for parallel runs)")
    args, _ = parser.parse_known_args()

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    elif args.full:
        subjects = _FULL_SUBJECTS
    elif args.ci:
        subjects = _CI_SUBJECTS
    else:
        subjects = _CI_SUBJECTS

    # GPU selection
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fold selection
    fold_indices = None
    if args.folds:
        fold_indices = [int(x) for x in args.folds.split(",")]

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("experiments_output") / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"H4s: Frequency-Selective Contrastive Learning")
    logger.info(f"Subjects ({len(subjects)}): {subjects}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Device: {device} (gpu={args.gpu})")
    if fold_indices:
        logger.info(f"Fold indices: {fold_indices}")

    # Load data once
    logger.info("Loading data...")
    subj_arrays, num_classes = load_data_once(subjects, args.base_dir, logger)
    logger.info(f"Data loaded: {len(subj_arrays)} subjects, {num_classes} classes")

    all_results = {}
    for variant in args.variants:
        summary = run_variant(
            variant=variant,
            subjects=subjects,
            subj_arrays=subj_arrays,
            num_classes=num_classes,
            device=device,
            logger=logger,
            output_dir=output_dir,
            fold_indices=fold_indices,
        )
        all_results[variant] = summary

    # Statistical comparison (only if all variants ran)
    if len(all_results) == len(VARIANT_NAMES):
        logger.info(f"\n{'='*60}")
        logger.info("  Statistical Comparison (Family B, Holm-Bonferroni, k=2)")
        logger.info(f"{'='*60}")
        stats_results = statistical_comparison(all_results, logger)

        combined = {
            "experiment": EXPERIMENT_NAME,
            "subjects": subjects,
            "n_subjects": len(subjects),
            "variants": {v: all_results[v] for v in args.variants if v in all_results},
            "statistical_tests": stats_results,
        }
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(combined, f, indent=2)

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"{'Variant':<40} {'Acc':>10} {'F1':>10}")
    logger.info("-" * 65)
    for v in args.variants:
        if v in all_results:
            r = all_results[v]
            acc_str = f"{r['mean_accuracy']*100:.2f}+-{r['std_accuracy']*100:.2f}"
            f1_str = f"{r['mean_f1_macro']*100:.2f}+-{r['std_f1_macro']*100:.2f}"
            logger.info(f"{VARIANT_LABELS[v]:<40} {acc_str:>10} {f1_str:>10}")

    logger.info(f"\nResults saved to {output_dir}")

    # Hypothesis tracking
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        best_variant = max(
            [v for v in args.variants if v in all_results],
            key=lambda v: all_results[v]["mean_f1_macro"],
        )
        baseline_f1 = all_results.get("all_bands", {}).get("mean_f1_macro", 0)
        if best_variant != "all_bands" and all_results[best_variant]["mean_f1_macro"] > baseline_f1:
            mark_hypothesis_verified("H4s", {
                "best_variant": best_variant,
                "mean_f1_macro": all_results[best_variant]["mean_f1_macro"],
                "delta_vs_all_bands": all_results[best_variant]["mean_f1_macro"] - baseline_f1,
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H4s", "Frequency-selective contrastive did not improve over all-bands baseline")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
