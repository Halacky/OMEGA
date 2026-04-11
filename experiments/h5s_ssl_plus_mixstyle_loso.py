#!/usr/bin/env python3
"""
H5s: SSL + MixStyle Combination (Confirmatory, Family A).

2x2 ablation testing the interaction between SSL pretraining and MixStyle
augmentation for cross-subject sEMG gesture recognition.

Variants (2x2 factorial)
────────────────────────
  sup_only:       Supervised UVMD (no MixStyle, no SSL)
  sup_mixstyle:   Supervised UVMD + MixStyle (= H7-F baseline)
  ssl_only:       VICReg SSL pretrain -> fine-tune (no MixStyle in finetune)
  ssl_mixstyle:   VICReg SSL pretrain -> fine-tune WITH MixStyle in finetune

Protocol (per LOSO fold)
────────────────────────
  sup_only / sup_mixstyle:
    Supervised training: 80 epochs, lr=1e-3, early stopping

  ssl_only / ssl_mixstyle:
    Phase 1 — SSL Pretrain: 50 ep, batch=512, lr=1e-3, cosine decay
    Phase 2 — Fine-tune: linear probe (10 ep) + full (50 ep, lr=1e-4)

  All variants:
    Evaluate on held-out test subject: accuracy, macro-F1

Statistical design
──────────────────
  Family A (k=3):
    ssl_mixstyle vs sup_mixstyle
    ssl_only vs sup_only
    ssl_mixstyle vs ssl_only
  Wilcoxon signed-rank + Holm-Bonferroni correction (alpha=0.05)

Usage
─────
  python experiments/h5s_ssl_plus_mixstyle_loso.py                   # CI
  python experiments/h5s_ssl_plus_mixstyle_loso.py --full             # 20 subj
  python experiments/h5s_ssl_plus_mixstyle_loso.py --variants sup_only ssl_only
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
from models.uvmd_ssl_encoder import UVMDSSLEncoder, UVMDSSLClassifier, PerBandMixStyle
from models.freq_aware_vicreg import FreqAwareVICReg
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

VARIANT_NAMES = ["sup_only", "sup_mixstyle", "ssl_only", "ssl_mixstyle"]
VARIANT_LABELS = {
    "sup_only": "Supervised (no MixStyle, no SSL)",
    "sup_mixstyle": "Supervised + MixStyle (H7-F)",
    "ssl_only": "VICReg SSL -> fine-tune (no MixStyle)",
    "ssl_mixstyle": "VICReg SSL -> fine-tune + MixStyle",
}

EXPERIMENT_NAME = "h5s_ssl_plus_mixstyle_loso"


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
            logits_list.append(model(Xt[s: s + batch_size]))
        preds = torch.cat(logits_list).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}


# ═════════════════════════════════════════════════════════════════════
#  SSL Pretraining (VICReg)
# ═════════════════════════════════════════════════════════════════════

def ssl_pretrain_vicreg(
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLEncoder:
    """VICReg pretraining on unlabelled train+val data."""
    logger.info("  [VICReg] Starting SSL pretraining...")

    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=False,  # MixStyle only added during finetune if needed
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    ssl_model = FreqAwareVICReg(
        encoder=encoder,
        proj_hidden=2048, proj_dim=2048,
        overlap_lambda=OVERLAP_LAMBDA, overlap_sigma=OVERLAP_SIGMA,
    ).to(device)

    X_all = np.concatenate([X_train, X_val], axis=0)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
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


# ═════════════════════════════════════════════════════════════════════
#  Fine-Tuning (after SSL pretrain)
# ═════════════════════════════════════════════════════════════════════

def finetune(
    encoder: UVMDSSLEncoder,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
    use_mixstyle: bool = False,
) -> UVMDSSLClassifier:
    """Two-phase fine-tuning: linear probe then full fine-tune."""
    # Optionally enable MixStyle for fine-tuning
    if use_mixstyle:
        encoder.mixstyle = PerBandMixStyle(
            K=encoder.K, p=MIXSTYLE_P, alpha=MIXSTYLE_ALPHA,
        ).to(device)

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
            logits = model(xb)
            loss = criterion(logits, yb)
            probe_optimizer.zero_grad()
            loss.backward()
            probe_optimizer.step()

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
            if patience_counter >= FT_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


# ═════════════════════════════════════════════════════════════════════
#  Supervised Training (H7-style)
# ═════════════════════════════════════════════════════════════════════

def train_supervised(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
    use_mixstyle: bool = False,
) -> UVMDSSLClassifier:
    """Train UVMD supervised with optional MixStyle."""
    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=use_mixstyle,
        mixstyle_p=MIXSTYLE_P, mixstyle_alpha=MIXSTYLE_ALPHA,
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

    if variant == "sup_only":
        model = train_supervised(
            X_train, y_train, X_val, y_val, num_classes, device, logger,
            use_mixstyle=False,
        )
    elif variant == "sup_mixstyle":
        model = train_supervised(
            X_train, y_train, X_val, y_val, num_classes, device, logger,
            use_mixstyle=True,
        )
    elif variant == "ssl_only":
        encoder = ssl_pretrain_vicreg(X_train, X_val, device, logger)
        model = finetune(
            encoder, X_train, y_train, X_val, y_val,
            num_classes, device, logger,
            use_mixstyle=False,
        )
    elif variant == "ssl_mixstyle":
        encoder = ssl_pretrain_vicreg(X_train, X_val, device, logger)
        model = finetune(
            encoder, X_train, y_train, X_val, y_val,
            num_classes, device, logger,
            use_mixstyle=True,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

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

def run_variant(
    variant: str,
    subjects: List[str],
    base_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"  Variant: {VARIANT_LABELS[variant]}")
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

    fold_results = []
    for test_sid in subjects:
        t0 = time.time()
        logger.info(f"  Fold: test={test_sid}")

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
        metrics["time_s"] = round(elapsed, 1)
        fold_results.append(metrics)

        logger.info(
            f"    Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}  ({elapsed:.0f}s)"
        )

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
    """Wilcoxon signed-rank tests with Holm-Bonferroni correction (Family A, k=3)."""
    from scipy import stats

    comparisons = [
        ("ssl_mixstyle", "sup_mixstyle"),
        ("ssl_only", "sup_only"),
        ("ssl_mixstyle", "ssl_only"),
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
    parser = argparse.ArgumentParser(description="H5s: SSL + MixStyle combination")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--ci", type=int, default=0)
    parser.add_argument("--subjects", type=str, default="")
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        choices=VARIANT_NAMES)
    parser.add_argument("--batch_size", type=int, default=FT_BATCH)
    parser.add_argument("--base_dir", type=str, default="data")
    args, _ = parser.parse_known_args()

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    elif args.full:
        subjects = _FULL_SUBJECTS
    elif args.ci:
        subjects = _CI_SUBJECTS
    else:
        subjects = _CI_SUBJECTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments_output") / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"H5s: SSL + MixStyle Combination (2x2 Ablation)")
    logger.info(f"Subjects ({len(subjects)}): {subjects}")
    logger.info(f"Variants: {args.variants}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    all_results = {}
    for variant in args.variants:
        summary = run_variant(
            variant=variant,
            subjects=subjects,
            base_dir=args.base_dir,
            device=device,
            logger=logger,
        )
        all_results[variant] = summary

        variant_dir = output_dir / variant
        variant_dir.mkdir(exist_ok=True)
        with open(variant_dir / "loso_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Statistical comparison
    logger.info(f"\n{'='*60}")
    logger.info("  Statistical Comparison (Family A, Holm-Bonferroni, k=3)")
    logger.info(f"{'='*60}")
    stats_results = statistical_comparison(all_results, logger)

    # Save combined results
    combined = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "variants": {v: all_results[v] for v in args.variants if v in all_results},
        "statistical_tests": stats_results,
    }
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"{'Variant':<42} {'Acc':>10} {'F1':>10}")
    logger.info("-" * 65)
    for v in args.variants:
        if v in all_results:
            r = all_results[v]
            acc_str = f"{r['mean_accuracy']*100:.2f}+-{r['std_accuracy']*100:.2f}"
            f1_str = f"{r['mean_f1_macro']*100:.2f}+-{r['std_f1_macro']*100:.2f}"
            logger.info(f"{VARIANT_LABELS[v]:<42} {acc_str:>10} {f1_str:>10}")

    logger.info(f"\nResults saved to {output_dir}")

    # Hypothesis tracking
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        # Check if SSL + MixStyle is the best
        best_variant = max(
            [v for v in args.variants if v in all_results],
            key=lambda v: all_results[v]["mean_f1_macro"],
        )
        if best_variant == "ssl_mixstyle":
            sup_ms_f1 = all_results.get("sup_mixstyle", {}).get("mean_f1_macro", 0)
            mark_hypothesis_verified("H5s", {
                "best_variant": best_variant,
                "mean_f1_macro": all_results[best_variant]["mean_f1_macro"],
                "delta_vs_sup_mixstyle": all_results[best_variant]["mean_f1_macro"] - sup_ms_f1,
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H5s", f"Best variant was {best_variant}, not ssl_mixstyle")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
