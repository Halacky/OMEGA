"""
Experiment 82: VMD Signal Decomposition with Multi-Stream CNN (LOSO)

Hypothesis
──────────
Inter-subject variability in EMG is concentrated in specific frequency modes.
Variational Mode Decomposition (VMD) decomposes each channel into K intrinsic
mode functions, each centred around a different frequency band. A multi-stream
CNN with learned mode attention can automatically focus on gesture-discriminative
modes while suppressing subject-specific ones (e.g., low-frequency baseline drift,
skin-electrode impedance variation).

Key difference from prior experiments:
  - exp_54 used multi-resolution convolutions (architectural), not explicit
    signal decomposition.
  - exp_67/68 used spectral features but did not decompose the signal.
  - exp_33 used wavelets (fixed basis); VMD is adaptive and data-driven.

VMD is a per-signal operation — each window of each channel is decomposed
independently with no cross-sample information. This makes it inherently
safe for LOSO with zero risk of data leakage from the decomposition step.

LOSO Protocol (strictly enforced — zero adaptation)
────────────────────────────────────────────────────
  For each fold (test_subject ∈ ALL_SUBJECTS):
    train_subjects = ALL_SUBJECTS \\ {test_subject}

    1. Load ALL subjects' windows via MultiSubjectLoader.
    2. Precompute VMD decomposition per window per channel (done once,
       reused across folds — safe because VMD is per-signal).
    3. Build train split: pool train-subject windows, carve val_ratio
       as validation using RNG — NO test data used.
    4. Build test split: test_subject windows ONLY.
    5. Per-mode per-channel standardisation (mean/std) from TRAIN only.
       Apply same statistics to val and test.
    6. Train VMDMultiStreamCNN end-to-end.
    7. Evaluate on test split (model.eval(), no adaptation).

Data-leakage guard summary
──────────────────────────
  ✓ VMD decomposition: per-window, per-channel — no cross-sample info.
  ✓ _build_splits(): train/val from train_subjects only.
  ✓ Test split: test_subject data only.
  ✓ Standardisation: computed from TRAIN windows only, applied to val/test.
  ✓ model.eval() at inference: BatchNorm frozen, no adaptation.
  ✓ common_gestures: derived only from gesture ID sets (not signal values).

Run examples
────────────
  # 5-subject CI run (safe default):
  python experiments/exp_82_vmd_imf_decomposition_multi_stream_loso.py --ci

  # Explicit subject list:
  python experiments/exp_82_vmd_imf_decomposition_multi_stream_loso.py \\
      --subjects DB2_s1,DB2_s12,DB2_s15,DB2_s28,DB2_s39

  # Custom VMD modes:
  python experiments/exp_82_vmd_imf_decomposition_multi_stream_loso.py \\
      --ci --num_modes 6 --alpha 3000

  # Full 20-subject run (local only):
  python experiments/exp_82_vmd_imf_decomposition_multi_stream_loso.py --full

Reference: Li et al. (Processes 2022) — EMD-PSO-DBN; Dragomiretskiy & Zosso
           (2014) — Variational Mode Decomposition.
"""

import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from models.vmd_multi_stream import VMDMultiStreamCNN, decompose_windows_vmd
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ═══════════════════════════════ SETTINGS ═════════════════════════════════

EXPERIMENT_NAME = "exp_82_vmd_imf_decomposition"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# VMD parameters
VMD_NUM_MODES   = 4       # K — number of IMF components
VMD_ALPHA       = 2000.0  # bandwidth constraint (larger → narrower bands)
VMD_TAU         = 0.0     # noise tolerance (0 = strict reconstruction)
VMD_TOL         = 1e-7    # convergence tolerance
VMD_MAX_ITER    = 300     # max ADMM iterations per signal

# Model architecture
BACKBONE_TYPE   = "shared"  # "shared" or "separate"
FEAT_DIM        = 64
HIDDEN_DIM      = 128
DROPOUT         = 0.3

# Training
BATCH_SIZE      = 64
EPOCHS          = 80
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
PATIENCE        = 15
GRAD_CLIP       = 1.0


# ═══════════════════════════ HELPER FUNCTIONS ═════════════════════════════

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
    gesture_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert grouped_windows to flat arrays.

    NOTE: grouped_to_arrays does NOT exist in any processing/ module.
    This local helper must be defined in every experiment that needs it.

    Returns
    -------
    windows : (N, T, C) float32
    labels  : (N,) int64   (values = gesture IDs, NOT class indices)
    """
    if gesture_ids is None:
        gesture_ids = sorted(grouped_windows.keys())

    all_windows: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for gid in gesture_ids:
        if gid not in grouped_windows:
            continue
        for rep_arr in grouped_windows[gid]:
            if isinstance(rep_arr, np.ndarray) and rep_arr.ndim == 3 and len(rep_arr) > 0:
                all_windows.append(rep_arr)
                all_labels.append(np.full(len(rep_arr), gid, dtype=np.int64))

    if not all_windows:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return (
        np.concatenate(all_windows, axis=0).astype(np.float32),
        np.concatenate(all_labels, axis=0),
    )


def precompute_vmd_all_subjects(
    subjects_data: Dict[str, tuple],
    common_gestures: List[int],
    K: int,
    alpha: float,
    tau: float,
    tol: float,
    max_iter: int,
    logger,
) -> Dict[str, Dict[int, List[np.ndarray]]]:
    """
    Precompute VMD decomposition for all subjects' windows.

    Returns dict: subject_id → {gesture_id → [decomposed_rep_array, ...]}
    where each decomposed_rep_array has shape (N_rep, K, T, C).

    This is safe for LOSO: VMD operates on each window independently.
    """
    subjects_vmd: Dict[str, Dict[int, List[np.ndarray]]] = {}

    for subj_id, (emg, segments, grouped_windows) in subjects_data.items():
        logger.info(f"  VMD decomposition for {subj_id}...")
        t0 = time.time()
        subjects_vmd[subj_id] = {}

        for gid in common_gestures:
            if gid not in grouped_windows:
                continue
            subjects_vmd[subj_id][gid] = []
            for rep_windows in grouped_windows[gid]:
                if not isinstance(rep_windows, np.ndarray):
                    continue
                if rep_windows.ndim != 3 or len(rep_windows) == 0:
                    continue
                # rep_windows: (N_rep, T, C) → decomposed: (N_rep, K, T, C)
                decomposed = decompose_windows_vmd(
                    rep_windows, K=K, alpha=alpha, tau=tau,
                    tol=tol, max_iter=max_iter,
                )
                subjects_vmd[subj_id][gid].append(decomposed)

        elapsed = time.time() - t0
        total_w = sum(
            arr.shape[0]
            for reps in subjects_vmd[subj_id].values()
            for arr in reps
        )
        logger.info(f"    {subj_id}: {total_w} windows decomposed in {elapsed:.1f}s")

    return subjects_vmd


# ═══════════════════════════ SPLITS BUILDER ═══════════════════════════════

def build_loso_splits_vmd(
    subjects_vmd: Dict[str, Dict[int, List[np.ndarray]]],
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           List[int], Dict[int, int]]:
    """
    Build strict LOSO train/val/test splits from precomputed VMD data.

    LOSO invariants:
      - train + val: ONLY from train_subjects.
      - val carved from train by random permutation (no test data).
      - test: ONLY from test_subject.
      - No signal statistics computed here (done later from train only).

    Returns
    -------
    X_train, y_train : (N, K, T, C), (N,)    — class indices
    X_val,   y_val   : (N, K, T, C), (N,)    — class indices
    X_test,  y_test  : (N, K, T, C), (N,)    — class indices
    class_ids        : List[int]              — sorted gesture IDs
    gesture_to_class : Dict[int, int]         — gid → class index
    """
    rng = np.random.RandomState(seed)
    gesture_to_class = {g: i for i, g in enumerate(common_gestures)}

    # ── Accumulate train windows (VMD-decomposed) ─────────────────────────
    train_windows: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []

    for subj_id in sorted(train_subjects):
        if subj_id not in subjects_vmd:
            continue
        for gid in common_gestures:
            if gid not in subjects_vmd[subj_id]:
                continue
            for rep_decomposed in subjects_vmd[subj_id][gid]:
                # rep_decomposed: (N_rep, K, T, C)
                train_windows.append(rep_decomposed)
                train_labels.append(
                    np.full(len(rep_decomposed), gesture_to_class[gid], dtype=np.int64)
                )

    if not train_windows:
        raise ValueError("No training windows collected — check subjects/gestures.")

    X_all_train = np.concatenate(train_windows, axis=0)  # (N, K, T, C)
    y_all_train = np.concatenate(train_labels, axis=0)   # (N,)

    # ── Shuffle and split train/val ───────────────────────────────────────
    n = len(X_all_train)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))

    val_idx = indices[:n_val]
    trn_idx = indices[n_val:]

    X_train = X_all_train[trn_idx]
    y_train = y_all_train[trn_idx]
    X_val = X_all_train[val_idx]
    y_val = y_all_train[val_idx]

    # ── Test: test subject only ───────────────────────────────────────────
    test_windows: List[np.ndarray] = []
    test_labels: List[np.ndarray] = []

    if test_subject in subjects_vmd:
        for gid in common_gestures:
            if gid not in subjects_vmd[test_subject]:
                continue
            for rep_decomposed in subjects_vmd[test_subject][gid]:
                test_windows.append(rep_decomposed)
                test_labels.append(
                    np.full(len(rep_decomposed), gesture_to_class[gid], dtype=np.int64)
                )

    if not test_windows:
        raise ValueError(f"No test windows for subject {test_subject}.")

    X_test = np.concatenate(test_windows, axis=0)
    y_test = np.concatenate(test_labels, axis=0)

    return X_train, y_train, X_val, y_val, X_test, y_test, common_gestures, gesture_to_class


# ═══════════════════════════ STANDARDISATION ══════════════════════════════

def standardize_per_mode(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    K: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Per-mode, per-channel standardisation using TRAIN statistics only.

    X arrays have shape (N, K, T, C). For each mode k and channel c,
    compute mean and std over all train samples and time steps, then
    apply to val and test.

    Returns
    -------
    X_train, X_val, X_test : standardised arrays (modified in-place copies)
    mode_stats : List of (mean, std) per mode, each shape (1, 1, C)
    """
    # Work on copies to avoid modifying the precomputed VMD cache
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    mode_stats = []
    for k in range(K):
        # mode_train: (N_train, T, C)
        mode_train = X_train[:, k, :, :]
        mean_k = mode_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
        std_k = mode_train.std(axis=(0, 1), keepdims=True)    # (1, 1, C)
        std_k = np.maximum(std_k, 1e-8)  # avoid division by zero

        X_train[:, k] = (X_train[:, k] - mean_k) / std_k
        X_val[:, k] = (X_val[:, k] - mean_k) / std_k
        X_test[:, k] = (X_test[:, k] - mean_k) / std_k

        mode_stats.append((mean_k.squeeze(), std_k.squeeze()))

    return X_train, X_val, X_test, mode_stats


# ═══════════════════════════ TRAINING & EVALUATION ════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grad_clip: float,
    logger,
) -> Tuple[nn.Module, Dict]:
    """
    Train VMDMultiStreamCNN with early stopping on validation loss.

    Uses class-weighted cross-entropy to handle gesture imbalance.
    Returns trained model and training history.
    """
    # Class-weighted loss (computed from TRAIN labels only)
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # NOTE: no verbose=True — removed in PyTorch 2.4+
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                n_val_batches += 1

                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e}"
            )

        # ── Early stopping ────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"  Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    history["best_epoch"] = len(history["train_loss"]) - patience_counter
    history["total_epochs"] = len(history["train_loss"])

    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate model on test data.  Returns metrics and mode attention analysis.

    model.eval() ensures: BatchNorm frozen, no stochastic layers active.
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_attn: List[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits, attn_weights = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
            all_attn.append(attn_weights.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    attn = np.concatenate(all_attn)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Mode attention analysis
    mean_attn = attn.mean(axis=0).tolist()  # average over test samples
    std_attn = attn.std(axis=0).tolist()

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "report": report,
        "confusion_matrix": cm,
        "mode_attention_mean": mean_attn,
        "mode_attention_std": std_attn,
    }


# ═══════════════════════════════ SINGLE FOLD ══════════════════════════════

def run_single_loso_fold(
    subjects_vmd: Dict[str, Dict[int, List[np.ndarray]]],
    all_subjects: List[str],
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    output_dir: Path,
    cfg: Dict,
    logger,
) -> Dict:
    """
    Execute one LOSO fold using precomputed VMD decomposition.

    Parameters
    ----------
    subjects_vmd : precomputed VMD data for all subjects
    train_subjects : subjects for training
    test_subject : held-out subject for testing
    common_gestures : gesture IDs present in all subjects
    output_dir : directory for fold outputs
    cfg : hyperparameters dict
    logger : logging instance

    Returns
    -------
    dict with test_accuracy, test_f1_macro, mode_attention_mean, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    K = cfg["num_modes"]
    device = torch.device(cfg["device"])

    seed_everything(cfg["seed"], verbose=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO Fold: test={test_subject}, train={train_subjects}")
    logger.info(f"{'='*60}")

    # ── Build splits ──────────────────────────────────────────────────────
    try:
        (X_train, y_train, X_val, y_val, X_test, y_test,
         class_ids, gesture_to_class) = build_loso_splits_vmd(
            subjects_vmd=subjects_vmd,
            train_subjects=train_subjects,
            test_subject=test_subject,
            common_gestures=common_gestures,
            val_ratio=cfg["val_ratio"],
            seed=cfg["seed"],
        )
    except ValueError as e:
        logger.error(f"Split building failed: {e}")
        return {
            "test_subject": test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    num_classes = len(common_gestures)
    in_channels = X_train.shape[3]  # C
    logger.info(
        f"  Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} | "
        f"classes={num_classes}, channels={in_channels}, modes={K}"
    )

    # ── Standardise per mode (TRAIN stats only) ──────────────────────────
    X_train, X_val, X_test, mode_stats = standardize_per_mode(
        X_train, X_val, X_test, K=K,
    )

    # ── Convert to PyTorch tensors ────────────────────────────────────────
    # Shape: (N, K, T, C) → (N, K, C, T) for Conv1d
    X_train_t = torch.FloatTensor(X_train.transpose(0, 1, 3, 2))
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val.transpose(0, 1, 3, 2))
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test.transpose(0, 1, 3, 2))
    y_test_t = torch.LongTensor(y_test)

    # Free numpy arrays
    del X_train, X_val, X_test
    gc.collect()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    # ── Create model ──────────────────────────────────────────────────────
    model = VMDMultiStreamCNN(
        num_modes=K,
        in_channels=in_channels,
        num_classes=num_classes,
        backbone_type=cfg["backbone_type"],
        feat_dim=cfg["feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model: VMDMultiStreamCNN ({n_params:,} parameters)")

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        y_train=y_train,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        grad_clip=cfg["grad_clip"],
        logger=logger,
    )
    train_time = time.time() - t0
    logger.info(f"  Training: {train_time:.1f}s, {history['total_epochs']} epochs")

    # ── Evaluate (model.eval() — BatchNorm frozen, no adaptation) ─────────
    class_names = [f"gesture_{g}" for g in common_gestures]
    test_results = evaluate_model(model, test_loader, device, class_names)

    test_acc = test_results["accuracy"]
    test_f1 = test_results["f1_macro"]

    logger.info(
        f"  Test: acc={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )
    logger.info(
        f"  Mode attention (mean): {test_results['mode_attention_mean']}"
    )

    # ── Save fold results ─────────────────────────────────────────────────
    fold_result = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "common_gestures": [int(g) for g in common_gestures],
        "num_classes": num_classes,
        "model_params": n_params,
        "training": {
            "epochs": history["total_epochs"],
            "best_epoch": history["best_epoch"],
            "train_time_s": round(train_time, 1),
        },
        "test_metrics": {
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "report": test_results["report"],
            "confusion_matrix": test_results["confusion_matrix"],
        },
        "mode_analysis": {
            "mode_attention_mean": test_results["mode_attention_mean"],
            "mode_attention_std": test_results["mode_attention_std"],
        },
    }

    with open(output_dir / "fold_results.json", "w") as fh:
        json.dump(make_json_serializable(fold_result), fh, indent=4, ensure_ascii=False)

    # ── Cleanup ───────────────────────────────────────────────────────────
    del model, train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds
    del X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "test_subject": test_subject,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "mode_attention_mean": test_results["mode_attention_mean"],
    }


# ════════════════════════════════════ MAIN ════════════════════════════════

def main():
    """
    LOSO evaluation loop with VMD signal decomposition.

    Subject list priority (safe server default = CI_TEST_SUBJECTS):
      1. --subjects DB2_s1,DB2_s12,...  — explicit list
      2. --full                         — all 20 DEFAULT_SUBJECTS
      3. --ci (or no flag)              — 5 CI_TEST_SUBJECTS  ← server safe

    VMD overrides:
      --num_modes 6    — number of VMD modes (default 4)
      --alpha 3000     — VMD bandwidth constraint (default 2000)
      --backbone sep   — use separate CNN branches per mode
    """
    import argparse

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject IDs")
    _parser.add_argument("--ci", action="store_true",
                         help="Use 5 CI test subjects (server-safe default)")
    _parser.add_argument("--full", action="store_true",
                         help="Use all 20 DEFAULT_SUBJECTS")
    _parser.add_argument("--num_modes", type=int, default=VMD_NUM_MODES,
                         help=f"Number of VMD modes K (default {VMD_NUM_MODES})")
    _parser.add_argument("--alpha", type=float, default=VMD_ALPHA,
                         help=f"VMD bandwidth constraint (default {VMD_ALPHA})")
    _parser.add_argument("--backbone", type=str, default=BACKBONE_TYPE,
                         choices=["shared", "separate"],
                         help="CNN backbone type: shared or separate per mode")
    _parser.add_argument("--exercises", type=str, default=None,
                         help="Comma-separated exercises, e.g. E1,E2")
    _args, _ = _parser.parse_known_args()

    # ── Subject list ──────────────────────────────────────────────────────
    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS  # safe default for vast.ai

    exercises = (
        [e.strip() for e in _args.exercises.split(",")]
        if _args.exercises else EXERCISES
    )

    # ── Hyperparameters ───────────────────────────────────────────────────
    cfg = {
        "num_modes": _args.num_modes,
        "alpha": _args.alpha,
        "tau": VMD_TAU,
        "tol": VMD_TOL,
        "max_iter": VMD_MAX_ITER,
        "backbone_type": _args.backbone,
        "feat_dim": FEAT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "grad_clip": GRAD_CLIP,
        "val_ratio": 0.15,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_ROOT)

    print("=" * 80)
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis  : VMD signal decomposition (K={cfg['num_modes']} modes,\n"
        f"              alpha={cfg['alpha']}) with multi-stream CNN\n"
        f"              + learned mode attention for subject-invariant features."
    )
    print(f"Subjects    : {ALL_SUBJECTS}")
    print(f"Exercises   : {exercises}")
    print(f"Backbone    : {cfg['backbone_type']}")
    print(f"Device      : {cfg['device']}")
    print(f"Output      : {OUTPUT_ROOT}")
    print("=" * 80)

    # ── Processing config ─────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=200,
        window_overlap=100,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )
    split_cfg = SplitConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    # Save configs
    proc_cfg.save(OUTPUT_ROOT / "processing_config.json")
    with open(OUTPUT_ROOT / "split_config.json", "w") as fh:
        json.dump(asdict(split_cfg), fh, indent=4)
    with open(OUTPUT_ROOT / "vmd_config.json", "w") as fh:
        json.dump(make_json_serializable(cfg), fh, indent=4)

    # ── Load all subjects ─────────────────────────────────────────────────
    logger.info("Loading all subjects...")
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=BASE_DIR,
        subject_ids=ALL_SUBJECTS,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES,
    )
    logger.info(
        f"Common gestures: {common_gestures} ({len(common_gestures)} total)"
    )

    # ── Precompute VMD for ALL subjects (safe: per-window operation) ──────
    logger.info(
        f"Precomputing VMD (K={cfg['num_modes']}, alpha={cfg['alpha']}) "
        f"for all subjects..."
    )
    t_vmd_start = time.time()
    subjects_vmd = precompute_vmd_all_subjects(
        subjects_data=subjects_data,
        common_gestures=common_gestures,
        K=cfg["num_modes"],
        alpha=cfg["alpha"],
        tau=cfg["tau"],
        tol=cfg["tol"],
        max_iter=cfg["max_iter"],
        logger=logger,
    )
    t_vmd_total = time.time() - t_vmd_start
    logger.info(f"VMD precomputation total: {t_vmd_total:.1f}s")

    # Free raw data (VMD cache holds what we need)
    del subjects_data
    gc.collect()

    # ── LOSO loop ─────────────────────────────────────────────────────────
    all_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / f"fold_test_{test_subject}"

        try:
            result = run_single_loso_fold(
                subjects_vmd=subjects_vmd,
                all_subjects=ALL_SUBJECTS,
                train_subjects=train_subjects,
                test_subject=test_subject,
                common_gestures=common_gestures,
                output_dir=fold_dir,
                cfg=cfg,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"Fold {test_subject} failed: {e}")
            traceback.print_exc()
            result = {
                "test_subject": test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            }

        all_results.append(result)

    # ── Aggregate LOSO summary ────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s = [r["test_f1_macro"] for r in valid]
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))

        # Aggregate mode attention across folds
        attn_per_fold = [
            r["mode_attention_mean"]
            for r in valid
            if "mode_attention_mean" in r
        ]
        if attn_per_fold:
            attn_array = np.array(attn_per_fold)
            mean_attn_across_folds = attn_array.mean(axis=0).tolist()
            std_attn_across_folds = attn_array.std(axis=0).tolist()
        else:
            mean_attn_across_folds = []
            std_attn_across_folds = []

        print("\n" + "=" * 80)
        print(f"LOSO SUMMARY — VMD Multi-Stream CNN (K={cfg['num_modes']})")
        print(f"  Subjects evaluated : {len(valid)}")
        print(
            f"  Accuracy  : {mean_acc:.4f} +/- {std_acc:.4f}"
            f"  (min={min(accs):.4f}, max={max(accs):.4f})"
        )
        print(
            f"  F1-macro  : {mean_f1:.4f} +/- {std_f1:.4f}"
            f"  (min={min(f1s):.4f}, max={max(f1s):.4f})"
        )
        if mean_attn_across_folds:
            print(
                f"  Mode attn : {['%.3f' % a for a in mean_attn_across_folds]}"
            )
        print(f"  VMD time  : {t_vmd_total:.1f}s")
        print("=" * 80)

        summary = {
            "experiment": EXPERIMENT_NAME,
            "model": "VMDMultiStreamCNN",
            "subjects": ALL_SUBJECTS,
            "exercises": exercises,
            "vmd_config": {
                "num_modes": cfg["num_modes"],
                "alpha": cfg["alpha"],
                "tau": cfg["tau"],
                "backbone_type": cfg["backbone_type"],
            },
            "loso_metrics": {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro": std_f1,
                "per_subject": all_results,
            },
            "mode_attention_analysis": {
                "mean_across_folds": mean_attn_across_folds,
                "std_across_folds": std_attn_across_folds,
                "interpretation": (
                    "Higher attention weight = mode is more important for classification. "
                    "If low-frequency modes (early indices) get lower weights, it suggests "
                    "they carry subject-specific info that the model learned to suppress."
                ),
            },
            "vmd_precomputation_time_s": round(t_vmd_total, 1),
        }

        with open(OUTPUT_ROOT / "loso_summary.json", "w") as fh:
            json.dump(
                make_json_serializable(summary), fh, indent=4, ensure_ascii=False,
            )
        print(f"Summary saved -> {OUTPUT_ROOT / 'loso_summary.json'}")
    else:
        print("No successful folds to summarise.")

    # ── Hypothesis executor (optional) ────────────────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        hypothesis_id = os.environ.get("HYPOTHESIS_ID", "")
        if hypothesis_id and valid:
            mark_hypothesis_verified(
                hypothesis_id,
                metrics={
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "mean_f1_macro": mean_f1,
                    "std_f1_macro": std_f1,
                    "n_folds": len(valid),
                },
                experiment_name=EXPERIMENT_NAME,
            )
        elif hypothesis_id and not valid:
            mark_hypothesis_failed(
                hypothesis_id,
                error_message="All LOSO folds failed — no valid results.",
            )
    except ImportError:
        pass  # hypothesis_executor not installed


if __name__ == "__main__":
    main()
