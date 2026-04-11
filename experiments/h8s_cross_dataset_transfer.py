#!/usr/bin/env python3
"""
H8s: Cross-Dataset Transfer Learning (Confirmatory, Family C).

Tests whether SSL representations learned on one exercise transfer to
another exercise within NinaPro DB2, using exercises as a proxy for
different datasets.

Variants
--------
  scratch_e1:      Train from scratch on E1 -> LOSO on E1 (baseline)
  scratch_e2:      Train from scratch on E2 -> LOSO on E2 (baseline)
  transfer_e1_e2:  SSL pretrain on E1 (all subjects, no labels) ->
                   fine-tune on E2 -> LOSO on E2
  transfer_e2_e1:  SSL pretrain on E2 -> fine-tune on E1 -> LOSO on E1

Protocol (per LOSO fold)
------------------------
  Phase 1 -- SSL Pretrain (transfer variants only):
    Data: source exercise, all train subjects (no labels, no test subject)
    Method: VICReg, 50 epochs, batch=512, lr=1e-3, cosine decay

  Phase 2 -- Supervised Fine-tune:
    Data: target exercise train subjects with labels
    Method: linear probe (10 ep) + full fine-tune (50 ep)

  Phase 3 -- Evaluate:
    Data: target exercise held-out test subject
    Metrics: accuracy, macro-F1

Statistical design
------------------
  Family C (k=2):
    transfer_e1_e2 vs scratch_e2
    transfer_e2_e1 vs scratch_e1
  Wilcoxon signed-rank + Holm-Bonferroni (alpha=0.05)

Usage
-----
  python experiments/h8s_cross_dataset_transfer.py
  python experiments/h8s_cross_dataset_transfer.py --full
  python experiments/h8s_cross_dataset_transfer.py --variants scratch_e1 transfer_e2_e1
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

# -- project imports ----------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from models.uvmd_ssl_encoder import UVMDSSLEncoder, UVMDSSLClassifier
from models.freq_aware_vicreg import FreqAwareVICReg
from utils.logging import setup_logging

# -- constants ----------------------------------------------------------------
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
K_MODES = 4

# UVMD architecture
L_LAYERS = 8
ALPHA_INIT = 2000.0
TAU_INIT = 0.01
FEAT_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.3

# SSL pretraining
SSL_EPOCHS = 50
SSL_BATCH = 512
SSL_LR = 1e-3
SSL_WARMUP_EPOCHS = 5

# Fine-tuning
FT_EPOCHS = 50
FT_BATCH = 512
FT_LR = 1e-4
FT_PATIENCE = 15
PROBE_EPOCHS = 10
PROBE_LR = 1e-3

# Supervised baseline
SUP_EPOCHS = 80
SUP_LR = 1e-3
SUP_PATIENCE = 15

# Common
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 0.05
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

EXPERIMENT_NAME = "h8s_cross_dataset_transfer"

VARIANT_NAMES = ["scratch_e1", "scratch_e2", "transfer_e1_e2", "transfer_e2_e1"]
VARIANT_LABELS = {
    "scratch_e1":      "Scratch E1 (supervised)",
    "scratch_e2":      "Scratch E2 (supervised)",
    "transfer_e1_e2":  "VICReg E1 -> fine-tune E2",
    "transfer_e2_e1":  "VICReg E2 -> fine-tune E1",
}

# Which exercise is source/target for each variant
VARIANT_CONFIG = {
    "scratch_e1":      {"source": None, "target": "E1"},
    "scratch_e2":      {"source": None, "target": "E2"},
    "transfer_e1_e2":  {"source": "E1", "target": "E2"},
    "transfer_e2_e1":  {"source": "E2", "target": "E1"},
}


# =============================================================================
#  Helpers
# =============================================================================

def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grouped windows to flat (windows, labels) arrays."""
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            windows_list.append(rep_arr)
            labels_list.extend([gid] * len(rep_arr))
    return np.concatenate(windows_list, axis=0), np.array(labels_list)


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights."""
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / (counts.astype(np.float64) + 1e-8)
    weights /= weights.sum()
    weights *= len(classes)
    w = torch.zeros(int(classes.max()) + 1, dtype=torch.float32, device=device)
    for c, wt in zip(classes, weights):
        w[int(c)] = wt
    return w


def cosine_schedule(
    epoch: int, total_epochs: int, base_lr: float, warmup: int = 5,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:
    """Evaluate classification model on numpy arrays."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_list = []
        for s in range(0, len(Xt), batch_size):
            logits_list.append(model(Xt[s: s + batch_size]))
        preds = torch.cat(logits_list).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}


def load_exercise_data(
    exercise: str,
    subjects: List[str],
    base_dir: str,
    logger: logging.Logger,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load windowed data for one exercise across subjects.

    Returns
    -------
    {subject_id: (windows, labels)} with remapped class labels.
    """
    proc_cfg = ProcessingConfig(
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP,
        sampling_rate=SAMPLING_RATE,
    )
    multi_loader = MultiSubjectLoader(
        proc_cfg, logger, use_gpu=False, use_improved_processing=True,
    )
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=subjects,
        exercises=[exercise],
        include_rest=False,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    gesture_to_class = {g: i for i, g in enumerate(sorted(common_gestures))}

    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, (_, _, gw) in subjects_data.items():
        wins, labs = grouped_to_arrays(gw)
        mask = np.isin(labs, list(gesture_to_class.keys()))
        wins, labs = wins[mask], labs[mask]
        labs = np.array([gesture_to_class[g] for g in labs])
        if len(wins) > 0:
            subj_arrays[sid] = (wins, labs)

    return subj_arrays


# =============================================================================
#  SSL Pretraining (VICReg)
# =============================================================================

def ssl_pretrain_vicreg(
    X_train: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLEncoder:
    """VICReg pretraining on unlabelled data."""
    logger.info("  [VICReg] Starting SSL pretraining...")

    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=False,
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    ssl_model = FreqAwareVICReg(
        encoder=encoder,
        proj_hidden=2048, proj_dim=2048,
        overlap_lambda=OVERLAP_LAMBDA, overlap_sigma=OVERLAP_SIGMA,
    ).to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=SSL_BATCH, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        ssl_model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(SSL_EPOCHS):
        ssl_model.train()
        lr = cosine_schedule(epoch, SSL_EPOCHS, SSL_LR, SSL_WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            loss, details = ssl_model(xb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ssl_model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / max(n_batches, 1)
            logger.info(
                f"    VICReg epoch {epoch+1}/{SSL_EPOCHS}  "
                f"loss={avg:.4f}  lr={lr:.6f}"
            )

    logger.info("  [VICReg] Pretraining complete.")
    return encoder


# =============================================================================
#  Fine-Tuning
# =============================================================================

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
        num_workers=2, pin_memory=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    # -- Phase 1: Linear probe (frozen encoder) -------------------------------
    model.freeze_encoder()
    probe_optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=PROBE_LR,
    )
    for epoch in range(PROBE_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            probe_optimizer.zero_grad()
            loss.backward()
            probe_optimizer.step()

    # -- Phase 2: Full fine-tune (unfrozen) -----------------------------------
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

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), FT_BATCH):
                val_logits.append(model(Xv[vs: vs + FT_BATCH]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FT_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
#  Supervised Baseline
# =============================================================================

def train_supervised(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLClassifier:
    """Train UVMD + MixStyle supervised from scratch."""
    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=True, mixstyle_p=MIXSTYLE_P, mixstyle_alpha=MIXSTYLE_ALPHA,
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    model = UVMDSSLClassifier(
        encoder=encoder, num_classes=num_classes,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=SUP_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=FT_BATCH, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(SUP_EPOCHS):
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

        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), FT_BATCH):
                val_logits.append(model(Xv[vs: vs + FT_BATCH]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= SUP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
#  Single LOSO Fold
# =============================================================================

def run_one_fold(
    variant: str,
    target_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    source_arrays: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    test_sid: str,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Run a single LOSO fold for one variant."""
    seed_everything()

    X_test, y_test = target_arrays[test_sid]
    if len(X_test) == 0:
        return {}

    # Collect train data from target exercise (all subjects except test)
    Xs, ys = [], []
    for sid in sorted(target_arrays.keys()):
        if sid == test_sid:
            continue
        w, l = target_arrays[sid]
        if len(w) > 0:
            Xs.append(w)
            ys.append(l)

    if not Xs:
        return {}

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    # Train/val split
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X_all))
    n_val = max(1, int(len(X_all) * VAL_RATIO))
    X_train, y_train = X_all[perm[n_val:]], y_all[perm[n_val:]]
    X_val, y_val = X_all[perm[:n_val]], y_all[perm[:n_val]]

    # Channel standardisation
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean_c) / std_c
    X_val = (X_val - mean_c) / std_c
    X_test_norm = (X_test - mean_c) / std_c

    cfg = VARIANT_CONFIG[variant]

    if cfg["source"] is None:
        # Supervised from scratch
        model = train_supervised(
            X_train, y_train, X_val, y_val, num_classes, device, logger,
        )
    else:
        # SSL pretrain on source exercise, fine-tune on target
        # Collect source exercise data (all subjects except test)
        source_Xs = []
        for sid in sorted(source_arrays.keys()):
            if sid == test_sid:
                continue
            w, _ = source_arrays[sid]
            if len(w) > 0:
                source_Xs.append(w)

        if not source_Xs:
            logger.warning(f"    No source data for {variant}, falling back to scratch")
            model = train_supervised(
                X_train, y_train, X_val, y_val, num_classes, device, logger,
            )
        else:
            X_source = np.concatenate(source_Xs, axis=0)
            # Standardise source with its own stats
            src_mean = X_source.mean(axis=(0, 1), keepdims=True)
            src_std = X_source.std(axis=(0, 1), keepdims=True) + 1e-8
            X_source = (X_source - src_mean) / src_std

            encoder = ssl_pretrain_vicreg(X_source, device, logger)
            model = finetune(
                encoder, X_train, y_train, X_val, y_val,
                num_classes, device, logger,
            )

    metrics = evaluate_model(model, X_test_norm, y_test, device)
    learned = model.get_learned_uvmd_params()

    return {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "final_omega_k": learned["omega_k"],
    }


# =============================================================================
#  LOSO Loop per Variant
# =============================================================================

def run_variant(
    variant: str,
    subjects: List[str],
    base_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Run full LOSO for one variant."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Variant: {VARIANT_LABELS[variant]}")
    logger.info(f"{'='*60}")

    cfg = VARIANT_CONFIG[variant]

    # Load target exercise data
    logger.info(f"  Loading target exercise: {cfg['target']}")
    target_arrays = load_exercise_data(
        cfg["target"], subjects, base_dir, logger,
    )

    # Load source exercise data if needed
    source_arrays = None
    if cfg["source"] is not None:
        logger.info(f"  Loading source exercise: {cfg['source']}")
        source_arrays = load_exercise_data(
            cfg["source"], subjects, base_dir, logger,
        )

    num_classes = 0
    for _, (_, labs) in target_arrays.items():
        num_classes = max(num_classes, int(labs.max()) + 1)

    fold_results = []
    for test_sid in subjects:
        if test_sid not in target_arrays:
            logger.warning(f"  Skipping {test_sid}: no target data")
            continue

        t0 = time.time()
        logger.info(f"  Fold: test={test_sid}")

        metrics = run_one_fold(
            variant=variant,
            target_arrays=target_arrays,
            source_arrays=source_arrays,
            test_sid=test_sid,
            num_classes=num_classes,
            device=device,
            logger=logger,
        )

        if not metrics:
            logger.warning(f"    Skipping {test_sid}: empty fold result")
            continue

        elapsed = time.time() - t0
        metrics["test_subject"] = test_sid
        metrics["time_s"] = round(elapsed, 1)
        fold_results.append(metrics)

        acc_str = f"{metrics['accuracy']:.4f}" if metrics.get("accuracy") is not None else "N/A"
        f1_str = f"{metrics['f1_macro']:.4f}" if metrics.get("f1_macro") is not None else "N/A"
        logger.info(f"    Acc={acc_str}  F1={f1_str}  ({elapsed:.0f}s)")

    accs = [r["accuracy"] for r in fold_results if r.get("accuracy") is not None]
    f1s = [r["f1_macro"] for r in fold_results if r.get("f1_macro") is not None]

    summary = {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "source_exercise": cfg["source"],
        "target_exercise": cfg["target"],
        "n_subjects": len(fold_results),
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "std_f1_macro": float(np.std(f1s)) if f1s else 0.0,
        "per_subject": fold_results,
    }

    if accs:
        logger.info(
            f"\n  >>> {VARIANT_LABELS[variant]}: "
            f"Acc={summary['mean_accuracy']*100:.2f}"
            f"+/-{summary['std_accuracy']*100:.2f}  "
            f"F1={summary['mean_f1_macro']*100:.2f}"
            f"+/-{summary['std_f1_macro']*100:.2f}"
        )
    return summary


# =============================================================================
#  Statistical Comparison
# =============================================================================

def statistical_comparison(
    results: Dict[str, Dict],
    logger: logging.Logger,
) -> Dict:
    """Wilcoxon signed-rank tests with Holm-Bonferroni correction (Family C)."""
    from scipy import stats

    # Family C comparisons
    comparisons = [
        ("transfer_e1_e2", "scratch_e2"),
        ("transfer_e2_e1", "scratch_e1"),
    ]

    stats_results = {}
    p_values = []

    for v1, v2 in comparisons:
        if v1 not in results or v2 not in results:
            continue

        f1_v1 = {r["test_subject"]: r["f1_macro"]
                 for r in results[v1]["per_subject"]
                 if r.get("f1_macro") is not None}
        f1_v2 = {r["test_subject"]: r["f1_macro"]
                 for r in results[v2]["per_subject"]
                 if r.get("f1_macro") is not None}
        common = sorted(set(f1_v1.keys()) & set(f1_v2.keys()))

        if len(common) < 5:
            logger.warning(
                f"  {v1} vs {v2}: only {len(common)} common subjects, "
                f"need >= 5 for Wilcoxon"
            )
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
            "n_transfer_wins": int((diff > 0).sum()),
        }
        p_values.append((key, p))

    # Holm-Bonferroni correction
    if p_values:
        p_values.sort(key=lambda x: x[1])
        k = len(p_values)
        for rank, (key, p_raw) in enumerate(p_values):
            p_corrected = min(1.0, p_raw * (k - rank))
            stats_results[key]["p_holm"] = float(p_corrected)
            sig = (
                "***" if p_corrected < 0.001
                else "**" if p_corrected < 0.01
                else "*" if p_corrected < 0.05
                else "ns"
            )
            stats_results[key]["significance"] = sig

            logger.info(
                f"  {key}: delta={stats_results[key]['mean_diff']*100:+.2f} pp  "
                f"d={stats_results[key]['cohens_d']:.2f}  "
                f"p_Holm={p_corrected:.4f} ({sig})"
            )

    return stats_results


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    _parser = argparse.ArgumentParser(
        description="H8s: Cross-Dataset Transfer Learning",
    )
    _parser.add_argument("--full", action="store_true",
                         help="Use full 20-subject set")
    _parser.add_argument("--ci", type=int, default=0,
                         help="Use CI subject set (default)")
    _parser.add_argument("--subjects", type=str, default="",
                         help="Comma-separated subject IDs")
    _parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                         choices=VARIANT_NAMES,
                         help="Which variants to run")
    _parser.add_argument("--batch_size", type=int, default=FT_BATCH)
    _parser.add_argument("--base_dir", type=str, default="data",
                         help="Base data directory")
    _args, _ = _parser.parse_known_args()

    # Subject selection (default to CI)
    if _args.subjects:
        subjects = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        subjects = _FULL_SUBJECTS
    elif _args.ci:
        subjects = _CI_SUBJECTS
    else:
        subjects = _CI_SUBJECTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments_output") / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("H8s: Cross-Dataset Transfer Learning")
    logger.info(f"Subjects ({len(subjects)}): {subjects}")
    logger.info(f"Variants: {_args.variants}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    all_results = {}
    for variant in _args.variants:
        summary = run_variant(
            variant=variant,
            subjects=subjects,
            base_dir=_args.base_dir,
            device=device,
            logger=logger,
        )
        all_results[variant] = summary

        # Save per-variant results
        variant_dir = output_dir / variant
        variant_dir.mkdir(exist_ok=True)
        with open(variant_dir / "loso_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Statistical comparison
    logger.info(f"\n{'='*60}")
    logger.info("  Statistical Comparison (Family C, Holm-Bonferroni, k=2)")
    logger.info(f"{'='*60}")
    stats_results = statistical_comparison(all_results, logger)

    # Save combined results
    combined = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "variants": {
            v: all_results[v] for v in _args.variants if v in all_results
        },
        "statistical_tests": stats_results,
    }
    with open(output_dir / "transfer_comparison.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"{'Variant':<35} {'Acc':>12} {'F1':>12}")
    logger.info("-" * 60)
    for v in _args.variants:
        if v in all_results:
            r = all_results[v]
            acc_str = (
                f"{r['mean_accuracy']*100:.2f}+/-{r['std_accuracy']*100:.2f}"
                if r["mean_accuracy"] > 0 else "N/A"
            )
            f1_str = (
                f"{r['mean_f1_macro']*100:.2f}+/-{r['std_f1_macro']*100:.2f}"
                if r["mean_f1_macro"] > 0 else "N/A"
            )
            logger.info(f"{VARIANT_LABELS[v]:<35} {acc_str:>12} {f1_str:>12}")

    logger.info(f"\nResults saved to {output_dir}")

    # Hypothesis tracking
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        # Check if any transfer variant beats its scratch counterpart
        transfer_wins = False
        for t_var, s_var in [("transfer_e1_e2", "scratch_e2"),
                             ("transfer_e2_e1", "scratch_e1")]:
            if t_var in all_results and s_var in all_results:
                if (all_results[t_var]["mean_f1_macro"] >
                        all_results[s_var]["mean_f1_macro"]):
                    transfer_wins = True
                    break

        if transfer_wins:
            best_delta = 0.0
            for t_var, s_var in [("transfer_e1_e2", "scratch_e2"),
                                 ("transfer_e2_e1", "scratch_e1")]:
                if t_var in all_results and s_var in all_results:
                    delta = (all_results[t_var]["mean_f1_macro"] -
                             all_results[s_var]["mean_f1_macro"])
                    best_delta = max(best_delta, delta)

            mark_hypothesis_verified("H8s", {
                "best_transfer_delta_f1": best_delta,
                "statistical_tests": stats_results,
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed(
                "H8s",
                "Transfer learning did not improve over training from scratch",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
