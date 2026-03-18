"""
Experiment 94: Learnable Filterbank + Mode-Level Attention + Gradient Reversal (LOSO)

Hypothesis
──────────
Instead of VMD (fixed algorithm), use a bank of K learnable Sinc bandpass filters
that separate the EMG signal into K frequency-band "modes".  A multi-head mode
attention layer automatically learns which frequency bands are task-relevant.  A
Gradient Reversal Layer (GRL) forces the learned mode representations to be
subject-invariant, reducing inter-subject variance.

Architecture:
    SincFilterbank (K=8 learnable bandpass filters)  →  (B, K, C, T) mode streams
    SharedModeGRUEncoder (shared weights across modes)  →  (B, K, D)
    MultiHeadModeAttention (task query → keys/values = modes)  →  (B, D)
    task_classifier   →  gesture logits   (B, num_classes)
    GradientReversalLayer → subject_classifier  →  (B, num_subjects)
    Loss = CE_task + adv_weight · CE_subject
    (GRL reverses gradient of CE_subject through the encoder)

LOSO Protocol (strictly enforced — zero adaptation to test subject)
────────────────────────────────────────────────────────────────────
  For each fold (test_subject ∈ ALL_SUBJECTS):
    train_subjects = ALL_SUBJECTS \\ {test_subject}

    1. Load all subjects' windows via MultiSubjectLoader.
    2. Build splits:
         train + val: ONLY from train_subjects (val carved by RNG).
         test: ONLY from test_subject.
    3. Per-channel standardization (mean/std) computed from TRAIN windows only.
       Applied identically to val and test.
    4. Create LearnableFilterbankGRL with num_subjects = len(train_subjects).
    5. Train with combined task + adversarial loss.
       GRL lambda follows DANN warm-up schedule (0 → grl_lambda_max).
    6. Evaluate on test split: model.eval(), subject_logits discarded.

Data-leakage guard summary
──────────────────────────
  ✓ SincFilterbank: per-window per-channel linear filter — no cross-sample state.
  ✓ Standardization: mean/std from TRAIN only; same stats applied to val/test.
  ✓ GRL trains subject-invariance from train subjects; no test data involved.
  ✓ model.eval() at inference: BatchNorm frozen, GRU state reset per window.
  ✓ subject_logits: IGNORED at inference.
  ✓ common_gestures: derived from gesture-ID sets only (not signal values).
  ✓ val split carved from train subjects only; no test subject in val.

Run examples
────────────
  # 5-subject CI run (server-safe default):
  python experiments/exp_94_learnable_filterbank_grl_loso.py --ci

  # Full 20-subject run (local only):
  python experiments/exp_94_learnable_filterbank_grl_loso.py --full

  # Custom settings:
  python experiments/exp_94_learnable_filterbank_grl_loso.py \\
      --ci --num_filters 8 --mode_dim 64 --grl_lambda_max 1.0

Reference:
    Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR 2016.
"""

import gc
import json
import math
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
from config.base import ProcessingConfig, SplitConfig
from data.multi_subject_loader import MultiSubjectLoader
from models.learnable_filterbank_grl import LearnableFilterbankGRL
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ═══════════════════════════════ SETTINGS ═════════════════════════════════════

EXPERIMENT_NAME = "exp_94_learnable_filterbank_grl"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# Filterbank / model
NUM_FILTERS     = 8       # K — number of learnable Sinc bandpass filters
SINC_KERNEL_SZ  = 51      # Sinc FIR kernel length (must be odd)
MIN_FREQ        = 5.0     # Hz
MAX_FREQ        = 500.0   # Hz
MODE_DIM        = 64      # D — mode representation dimension
NUM_HEADS       = 4       # attention heads (must divide MODE_DIM)
GRU_LAYERS      = 1       # GRU layers in per-mode encoder
DROPOUT         = 0.3

# Training
GRL_LAMBDA_MAX  = 1.0     # peak GRL reversal strength (reached at epoch=EPOCHS)
ADV_LOSS_WEIGHT = 1.0     # coefficient of CE_subject in total loss
BATCH_SIZE      = 64
EPOCHS          = 80
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
PATIENCE        = 15
GRAD_CLIP       = 1.0
VAL_RATIO       = 0.15
SEED            = 42


# ═══════════════════════════ HELPER FUNCTIONS ═════════════════════════════════

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
    gesture_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert grouped_windows to flat (windows, labels) arrays.

    NOTE: grouped_to_arrays does NOT exist in any processing/ module.
    This local helper must be defined in every experiment that needs it.

    Returns
    -------
    windows : (N, T, C) float32
    labels  : (N,) int64   — gesture IDs (NOT class indices)
    """
    if gesture_ids is None:
        gesture_ids = sorted(grouped_windows.keys())

    all_windows: List[np.ndarray] = []
    all_labels:  List[np.ndarray] = []

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


# ═══════════════════════════ SPLITS BUILDER ═══════════════════════════════════

def build_loso_splits(
    subjects_data: Dict[str, tuple],
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,  # X_train, y_train, s_train
    np.ndarray, np.ndarray, np.ndarray,  # X_val,   y_val,   s_val
    np.ndarray, np.ndarray,              # X_test,  y_test
    List[int], Dict[int, int], Dict[str, int],
]:
    """
    Build strict LOSO splits with subject labels for the GRL adversarial branch.

    LOSO invariants:
      - train + val: ONLY from train_subjects (val carved from train by RNG).
      - test: ONLY from test_subject — NO subject label assigned (unseen subject).
      - No signal statistics computed here; standardization done separately.

    Returns
    -------
    X_train, y_train, s_train : (N, T, C), (N,) gesture class idx, (N,) subject idx
    X_val,   y_val,   s_val   : same structure
    X_test,  y_test            : (N, T, C), (N,) gesture class idx
    class_ids                  : sorted gesture IDs
    gesture_to_class           : gid → class index
    subject_to_idx             : subject_id → index (0..len(train_subjects)-1)
    """
    rng = np.random.RandomState(seed)
    gesture_to_class = {g: i for i, g in enumerate(common_gestures)}

    # Subject index mapping for GRL — train subjects only
    sorted_train = sorted(train_subjects)
    subject_to_idx = {s: i for i, s in enumerate(sorted_train)}

    # ── Accumulate train windows with gesture + subject labels ─────────────
    train_windows:         List[np.ndarray] = []
    train_gesture_labels:  List[np.ndarray] = []
    train_subject_labels:  List[np.ndarray] = []

    for subj_id in sorted_train:
        if subj_id not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[subj_id]
        subj_idx = subject_to_idx[subj_id]

        for gid in common_gestures:
            if gid not in grouped_windows:
                continue
            for rep_arr in grouped_windows[gid]:
                if (not isinstance(rep_arr, np.ndarray)
                        or rep_arr.ndim != 3 or len(rep_arr) == 0):
                    continue
                n_rep = len(rep_arr)
                train_windows.append(rep_arr)
                train_gesture_labels.append(
                    np.full(n_rep, gesture_to_class[gid], dtype=np.int64)
                )
                train_subject_labels.append(
                    np.full(n_rep, subj_idx, dtype=np.int64)
                )

    if not train_windows:
        raise ValueError("No training windows collected — check subjects/gestures.")

    X_all = np.concatenate(train_windows, axis=0).astype(np.float32)  # (N, T, C)
    y_all = np.concatenate(train_gesture_labels, axis=0)               # (N,)
    s_all = np.concatenate(train_subject_labels, axis=0)               # (N,)

    # ── Shuffle and carve val from train ──────────────────────────────────
    n = len(X_all)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx, trn_idx = idx[:n_val], idx[n_val:]

    X_train, y_train, s_train = X_all[trn_idx], y_all[trn_idx], s_all[trn_idx]
    X_val,   y_val,   s_val   = X_all[val_idx],  y_all[val_idx],  s_all[val_idx]

    # ── Test split: test_subject only — no subject label needed ───────────
    if test_subject not in subjects_data:
        raise ValueError(f"Test subject {test_subject} not in subjects_data.")

    _, _, test_grouped = subjects_data[test_subject]
    test_windows: List[np.ndarray] = []
    test_labels:  List[np.ndarray] = []

    for gid in common_gestures:
        if gid not in test_grouped:
            continue
        for rep_arr in test_grouped[gid]:
            if (not isinstance(rep_arr, np.ndarray)
                    or rep_arr.ndim != 3 or len(rep_arr) == 0):
                continue
            test_windows.append(rep_arr)
            test_labels.append(
                np.full(len(rep_arr), gesture_to_class[gid], dtype=np.int64)
            )

    if not test_windows:
        raise ValueError(f"No test windows for subject {test_subject}.")

    X_test = np.concatenate(test_windows, axis=0).astype(np.float32)
    y_test = np.concatenate(test_labels, axis=0)

    return (
        X_train, y_train, s_train,
        X_val,   y_val,   s_val,
        X_test,  y_test,
        common_gestures, gesture_to_class, subject_to_idx,
    )


# ═══════════════════════════ STANDARDISATION ══════════════════════════════════

def standardize_channels(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel standardization using TRAIN statistics only.

    X arrays shape: (N, T, C).  Statistics computed over (N*T) for each channel c.
    Applied identically to val and test — no information from val/test used.

    Returns
    -------
    X_train, X_val, X_test : standardized copies
    mean_c, std_c          : (C,) train statistics (for logging)
    """
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
    std_c  = X_train.std(axis=(0, 1), keepdims=True)   # (1, 1, C)
    std_c  = np.maximum(std_c, 1e-8)

    X_train = (X_train - mean_c) / std_c
    X_val   = (X_val   - mean_c) / std_c
    X_test  = (X_test  - mean_c) / std_c

    return X_train, X_val, X_test, mean_c.squeeze(), std_c.squeeze()


# ════════════════════════════ GRL LAMBDA SCHEDULE ═════════════════════════════

def get_grl_lambda(epoch: int, total_epochs: int, lambda_max: float = 1.0) -> float:
    """
    DANN warm-up schedule: GRL strength from ~0 to lambda_max over training.

        λ(p) = λ_max · (2 / (1 + exp(-10·p)) - 1),   p = epoch / total_epochs

    At p=0: λ ≈ 0   (feature extractor learns gestures first; GRL barely active)
    At p=1: λ ≈ λ_max  (full adversarial pressure)

    Reference: Ganin et al., JMLR 2016, Eq. 4.
    """
    p = epoch / max(total_epochs, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ══════════════════════════ TRAINING & EVALUATION ═════════════════════════════

def train_model(
    model: LearnableFilterbankGRL,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_classes:  int,
    y_train:      np.ndarray,
    device:       torch.device,
    epochs:       int,
    lr:           float,
    weight_decay: float,
    patience:     int,
    grad_clip:    float,
    grl_lambda_max:  float,
    adv_loss_weight: float,
    logger,
) -> Tuple[LearnableFilterbankGRL, Dict]:
    """
    Train with combined task CE + adversarial subject CE (GRL handles reversal).

    Loss = CE_task(task_logits, y_gesture)
         + adv_loss_weight * CE_subject(subject_logits, y_subject)

    Gradients from CE_subject flow backward through GRL, which reverses them,
    pushing the encoder toward subject-invariant representations.

    Early stopping monitors val task loss (task performance only).
    """
    # Class-weighted task loss (computed from TRAIN labels only — LOSO safe)
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    class_counts  = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    task_criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device)
    )
    # Subject CE — uniform across subjects (they contribute equal windows)
    subject_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # NOTE: no verbose=True — parameter removed in PyTorch 2.4+
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history: Dict = {
        "train_task_loss": [], "train_adv_loss": [],
        "val_task_loss":   [], "val_acc":        [],
        "grl_lambda":      [],
    }

    for epoch in range(epochs):
        # ── Update GRL lambda (DANN schedule) ─────────────────────────────
        grl_lambda = get_grl_lambda(epoch, epochs, lambda_max=grl_lambda_max)
        model.set_grl_lambda(grl_lambda)

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        run_task_loss = 0.0
        run_adv_loss  = 0.0
        n_batches = 0

        for X_batch, y_gesture, y_subject in train_loader:
            X_batch   = X_batch.to(device)
            y_gesture = y_gesture.to(device)
            y_subject = y_subject.to(device)

            optimizer.zero_grad()
            task_logits, subject_logits, _ = model(X_batch)

            task_loss = task_criterion(task_logits, y_gesture)
            adv_loss  = subject_criterion(subject_logits, y_subject)
            # GRL reversed the gradient of adv_loss through the encoder.
            # adv_loss_weight scales the importance relative to task_loss.
            loss = task_loss + adv_loss_weight * adv_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            run_task_loss += task_loss.item()
            run_adv_loss  += adv_loss.item()
            n_batches += 1

        avg_task_loss = run_task_loss / max(n_batches, 1)
        avg_adv_loss  = run_adv_loss  / max(n_batches, 1)

        # ── Validate (task only — for monitoring and early stopping) ───────
        model.eval()
        val_task_loss = 0.0
        val_correct   = 0
        val_total     = 0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_gesture, _y_subject in val_loader:
                X_batch   = X_batch.to(device)
                y_gesture = y_gesture.to(device)

                task_logits, _, _ = model(X_batch)
                loss = task_criterion(task_logits, y_gesture)
                val_task_loss += loss.item()
                n_val_batches += 1

                preds = task_logits.argmax(dim=1)
                val_correct += (preds == y_gesture).sum().item()
                val_total   += len(y_gesture)

        avg_val_loss = val_task_loss / max(n_val_batches, 1)
        val_acc      = val_correct / max(val_total, 1)

        scheduler.step(avg_val_loss)

        history["train_task_loss"].append(avg_task_loss)
        history["train_adv_loss"].append(avg_adv_loss)
        history["val_task_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["grl_lambda"].append(grl_lambda)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"task={avg_task_loss:.4f} adv={avg_adv_loss:.4f} | "
                f"val_task={avg_val_loss:.4f} val_acc={val_acc:.4f} | "
                f"grl_λ={grl_lambda:.3f} lr={current_lr:.2e}"
            )

        # ── Early stopping ─────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"  Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    history["best_epoch"]    = len(history["train_task_loss"]) - patience_counter
    history["total_epochs"]  = len(history["train_task_loss"])
    return model, history


def evaluate_model(
    model:       LearnableFilterbankGRL,
    test_loader: DataLoader,
    device:      torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate model on test data using task_logits only.

    model.eval() guarantees:
      - BatchNorm uses frozen running statistics (from training data only).
      - GRU state reset per window (no cross-sample memory).
      - GRL has no effect (no gradient flow during eval).
      - subject_logits are computed but IGNORED.

    test_loader yields 2-item batches: (X_batch, y_batch).
    No subject labels for the unseen test subject.
    """
    model.eval()
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_attn:   List[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            task_logits, _, attn_weights = model(X_batch)

            preds = task_logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
            all_attn.append(attn_weights.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    attn   = np.concatenate(all_attn)

    acc    = float(accuracy_score(y_true, y_pred))
    f1     = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(
        y_true, y_pred,
        target_names=class_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy":           acc,
        "f1_macro":           f1,
        "report":             report,
        "confusion_matrix":   cm,
        "mode_attention_mean": attn.mean(axis=0).tolist(),
        "mode_attention_std":  attn.std(axis=0).tolist(),
    }


# ══════════════════════════════ SINGLE FOLD ═══════════════════════════════════

def run_single_loso_fold(
    subjects_data:    Dict[str, tuple],
    train_subjects:   List[str],
    test_subject:     str,
    common_gestures:  List[int],
    output_dir:       Path,
    cfg:              Dict,
    logger,
) -> Dict:
    """
    Execute one LOSO fold.

    train_subjects: subjects used for training and GRL adversarial branch.
    test_subject:   held-out subject — ZERO information used during training.

    Returns dict with test_accuracy, test_f1_macro, mode_attention_mean.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"])
    seed_everything(cfg["seed"], verbose=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO Fold: test={test_subject}, train={train_subjects}")
    logger.info(f"{'='*60}")

    # ── Build splits ───────────────────────────────────────────────────────
    try:
        (X_train, y_train, s_train,
         X_val,   y_val,   s_val,
         X_test,  y_test,
         class_ids, gesture_to_class, subject_to_idx) = build_loso_splits(
            subjects_data   = subjects_data,
            train_subjects  = train_subjects,
            test_subject    = test_subject,
            common_gestures = common_gestures,
            val_ratio       = cfg["val_ratio"],
            seed            = cfg["seed"],
        )
    except ValueError as e:
        logger.error(f"Split building failed: {e}")
        return {
            "test_subject": test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    num_classes  = len(common_gestures)
    num_subjects = len(train_subjects)   # GRL head size for this fold
    in_channels  = X_train.shape[2]      # C

    logger.info(
        f"  Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} | "
        f"classes={num_classes}, channels={in_channels}, "
        f"train_subjects={num_subjects}"
    )

    # ── Per-channel standardization (TRAIN stats only) ─────────────────────
    X_train, X_val, X_test, mean_c, std_c = standardize_channels(
        X_train, X_val, X_test
    )

    # ── Convert to tensors, transpose to (N, C, T) for Conv1d ─────────────
    def to_tensor_ct(X: np.ndarray) -> torch.Tensor:
        # (N, T, C) → (N, C, T)
        return torch.FloatTensor(X.transpose(0, 2, 1))

    X_train_t = to_tensor_ct(X_train)
    X_val_t   = to_tensor_ct(X_val)
    X_test_t  = to_tensor_ct(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t   = torch.LongTensor(y_val)
    y_test_t  = torch.LongTensor(y_test)
    s_train_t = torch.LongTensor(s_train)
    s_val_t   = torch.LongTensor(s_val)

    del X_train, X_val, X_test
    gc.collect()

    # train/val include subject labels for GRL; test does NOT
    train_ds = TensorDataset(X_train_t, y_train_t, s_train_t)
    val_ds   = TensorDataset(X_val_t,   y_val_t,   s_val_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)          # no subject label

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False)

    # ── Create model (fresh per fold — num_subjects differs each fold) ─────
    model = LearnableFilterbankGRL(
        in_channels      = in_channels,
        num_classes      = num_classes,
        num_subjects     = num_subjects,
        num_filters      = cfg["num_filters"],
        sinc_kernel_size = cfg["sinc_kernel_size"],
        sample_rate      = cfg["sample_rate"],
        min_freq         = cfg["min_freq"],
        max_freq         = cfg["max_freq"],
        mode_dim         = cfg["mode_dim"],
        num_heads        = cfg["num_heads"],
        gru_layers       = cfg["gru_layers"],
        dropout          = cfg["dropout"],
        grl_lambda       = 0.0,   # starts at 0; warm-up schedule sets it per epoch
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model: LearnableFilterbankGRL ({n_params:,} parameters)")

    # ── Train ──────────────────────────────────────────────────────────────
    t0 = time.time()
    model, history = train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        num_classes     = num_classes,
        y_train         = y_train,
        device          = device,
        epochs          = cfg["epochs"],
        lr              = cfg["lr"],
        weight_decay    = cfg["weight_decay"],
        patience        = cfg["patience"],
        grad_clip       = cfg["grad_clip"],
        grl_lambda_max  = cfg["grl_lambda_max"],
        adv_loss_weight = cfg["adv_loss_weight"],
        logger          = logger,
    )
    train_time = time.time() - t0
    logger.info(
        f"  Training: {train_time:.1f}s, "
        f"{history['total_epochs']} epochs (best={history['best_epoch']})"
    )

    # ── Evaluate on test split (model.eval() — frozen, no adaptation) ──────
    class_names  = [f"gesture_{g}" for g in common_gestures]
    test_results = evaluate_model(model, test_loader, device, class_names)

    test_acc = test_results["accuracy"]
    test_f1  = test_results["f1_macro"]
    logger.info(f"  Test: acc={test_acc:.4f}, F1-macro={test_f1:.4f}")
    logger.info(
        f"  Mode attention (mean): "
        f"{['%.3f' % a for a in test_results['mode_attention_mean']]}"
    )

    # ── Save fold results ──────────────────────────────────────────────────
    fold_result = {
        "test_subject":    test_subject,
        "train_subjects":  train_subjects,
        "common_gestures": [int(g) for g in common_gestures],
        "num_classes":     num_classes,
        "num_subjects":    num_subjects,
        "model_params":    n_params,
        "training": {
            "epochs":       history["total_epochs"],
            "best_epoch":   history["best_epoch"],
            "train_time_s": round(train_time, 1),
        },
        "test_metrics": {
            "accuracy":         test_acc,
            "f1_macro":         test_f1,
            "report":           test_results["report"],
            "confusion_matrix": test_results["confusion_matrix"],
        },
        "mode_analysis": {
            "mode_attention_mean": test_results["mode_attention_mean"],
            "mode_attention_std":  test_results["mode_attention_std"],
            "interpretation": (
                "Attention weight ≈ importance of that frequency band for gesture "
                "classification. GRL pushes ALL modes to be subject-invariant; "
                "high-attention modes are the most gesture-discriminative ones."
            ),
        },
    }

    with open(output_dir / "fold_results.json", "w") as fh:
        json.dump(make_json_serializable(fold_result), fh, indent=4, ensure_ascii=False)

    # ── Cleanup ────────────────────────────────────────────────────────────
    del model, train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds
    del X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t
    del s_train_t, s_val_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "test_subject":       test_subject,
        "test_accuracy":      test_acc,
        "test_f1_macro":      test_f1,
        "mode_attention_mean": test_results["mode_attention_mean"],
    }


# ══════════════════════════════════ MAIN ══════════════════════════════════════

def main() -> None:
    """
    LOSO evaluation loop for Learnable Filterbank + Mode Attention + GRL.

    Subject list priority (default = CI_TEST_SUBJECTS, server-safe):
      1. --subjects DB2_s1,DB2_s12,...  — explicit list
      2. --full                         — all 20 DEFAULT_SUBJECTS (local only)
      3. --ci (or no flag)              — 5 CI_TEST_SUBJECTS  ← vast.ai safe
    """
    import argparse

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects",       type=str,   default=None)
    _parser.add_argument("--ci",             action="store_true")
    _parser.add_argument("--full",           action="store_true")
    _parser.add_argument("--exercises",      type=str,   default=None)
    _parser.add_argument("--num_filters",    type=int,   default=NUM_FILTERS)
    _parser.add_argument("--mode_dim",       type=int,   default=MODE_DIM)
    _parser.add_argument("--num_heads",      type=int,   default=NUM_HEADS)
    _parser.add_argument("--grl_lambda_max", type=float, default=GRL_LAMBDA_MAX)
    _parser.add_argument("--adv_weight",     type=float, default=ADV_LOSS_WEIGHT)
    _args, _ = _parser.parse_known_args()

    # ── Subject list ───────────────────────────────────────────────────────
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

    # ── Hyperparameters ────────────────────────────────────────────────────
    cfg = {
        "num_filters":    _args.num_filters,
        "sinc_kernel_size": SINC_KERNEL_SZ,
        "sample_rate":    2000,
        "min_freq":       MIN_FREQ,
        "max_freq":       MAX_FREQ,
        "mode_dim":       _args.mode_dim,
        "num_heads":      _args.num_heads,
        "gru_layers":     GRU_LAYERS,
        "dropout":        DROPOUT,
        "grl_lambda_max": _args.grl_lambda_max,
        "adv_loss_weight": _args.adv_weight,
        "batch_size":     BATCH_SIZE,
        "epochs":         EPOCHS,
        "lr":             LR,
        "weight_decay":   WEIGHT_DECAY,
        "patience":       PATIENCE,
        "grad_clip":      GRAD_CLIP,
        "val_ratio":      VAL_RATIO,
        "seed":           SEED,
        "device":         "cuda" if torch.cuda.is_available() else "cpu",
    }

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_ROOT)

    print("=" * 80)
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis  : Learnable Sinc filterbank (K={cfg['num_filters']} modes)\n"
        f"              + mode-level attention + GRL adversarial branch.\n"
        f"              GRL lambda warm-up: 0 → {cfg['grl_lambda_max']}"
    )
    print(f"Subjects    : {ALL_SUBJECTS}")
    print(f"Exercises   : {exercises}")
    print(f"mode_dim    : {cfg['mode_dim']}, num_heads: {cfg['num_heads']}")
    print(f"adv_weight  : {cfg['adv_loss_weight']}")
    print(f"Device      : {cfg['device']}")
    print(f"Output      : {OUTPUT_ROOT}")
    print("=" * 80)

    # ── Processing config ──────────────────────────────────────────────────
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
        seed=SEED,
        include_rest_in_splits=False,
    )

    proc_cfg.save(OUTPUT_ROOT / "processing_config.json")
    with open(OUTPUT_ROOT / "split_config.json", "w") as fh:
        json.dump(asdict(split_cfg), fh, indent=4)
    with open(OUTPUT_ROOT / "experiment_config.json", "w") as fh:
        json.dump(make_json_serializable(cfg), fh, indent=4)

    # ── Load all subjects once (reused across folds) ───────────────────────
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

    # ── LOSO loop ──────────────────────────────────────────────────────────
    all_results: List[Dict] = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / f"fold_test_{test_subject}"

        try:
            result = run_single_loso_fold(
                subjects_data   = subjects_data,
                train_subjects  = train_subjects,
                test_subject    = test_subject,
                common_gestures = common_gestures,
                output_dir      = fold_dir,
                cfg             = cfg,
                logger          = logger,
            )
        except Exception as e:
            logger.error(f"Fold {test_subject} failed: {e}")
            traceback.print_exc()
            result = {
                "test_subject":  test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error":         str(e),
            }

        all_results.append(result)

    # ── Aggregate LOSO summary ─────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]

    if valid:
        accs    = [r["test_accuracy"] for r in valid]
        f1s     = [r["test_f1_macro"] for r in valid]
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        mean_f1  = float(np.mean(f1s))
        std_f1   = float(np.std(f1s))

        # Aggregate mode attention across folds
        attn_per_fold = [
            r["mode_attention_mean"]
            for r in valid if "mode_attention_mean" in r
        ]
        if attn_per_fold:
            attn_arr = np.array(attn_per_fold)
            mean_attn = attn_arr.mean(axis=0).tolist()
            std_attn  = attn_arr.std(axis=0).tolist()
        else:
            mean_attn = []
            std_attn  = []

        print("\n" + "=" * 80)
        print(f"LOSO SUMMARY — {EXPERIMENT_NAME} (K={cfg['num_filters']})")
        print(f"  Folds evaluated : {len(valid)}")
        print(
            f"  Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}"
            f"  (min={min(accs):.4f}, max={max(accs):.4f})"
        )
        print(
            f"  F1-macro  : {mean_f1:.4f} ± {std_f1:.4f}"
            f"  (min={min(f1s):.4f}, max={max(f1s):.4f})"
        )
        if mean_attn:
            print(f"  Mode attn : {['%.3f' % a for a in mean_attn]}")
        print("=" * 80)

        summary = {
            "experiment":  EXPERIMENT_NAME,
            "model":       "LearnableFilterbankGRL",
            "subjects":    ALL_SUBJECTS,
            "exercises":   exercises,
            "model_config": {
                "num_filters":     cfg["num_filters"],
                "sinc_kernel_size": cfg["sinc_kernel_size"],
                "mode_dim":        cfg["mode_dim"],
                "num_heads":       cfg["num_heads"],
                "grl_lambda_max":  cfg["grl_lambda_max"],
                "adv_loss_weight": cfg["adv_loss_weight"],
            },
            "loso_metrics": {
                "mean_accuracy":  mean_acc,
                "std_accuracy":   std_acc,
                "mean_f1_macro":  mean_f1,
                "std_f1_macro":   std_f1,
                "per_subject":    all_results,
            },
            "mode_attention_analysis": {
                "mean_across_folds": mean_attn,
                "std_across_folds":  std_attn,
                "interpretation": (
                    "Higher weight = frequency band more important for gesture "
                    "classification.  GRL pushes all mode representations toward "
                    "subject-invariance; dominant attention modes are those where "
                    "gesture signal outweighs subject-specific variation."
                ),
            },
        }

        with open(OUTPUT_ROOT / "loso_summary.json", "w") as fh:
            json.dump(
                make_json_serializable(summary), fh, indent=4, ensure_ascii=False,
            )
        print(f"Summary saved → {OUTPUT_ROOT / 'loso_summary.json'}")
    else:
        print("No successful folds to summarise.")

    # ── Hypothesis executor (optional — may not be installed) ─────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        hypothesis_id = os.environ.get("HYPOTHESIS_ID", "")
        if hypothesis_id and valid:
            mark_hypothesis_verified(
                hypothesis_id,
                metrics={
                    "mean_accuracy": mean_acc,
                    "std_accuracy":  std_acc,
                    "mean_f1_macro": mean_f1,
                    "std_f1_macro":  std_f1,
                    "n_folds":       len(valid),
                },
                experiment_name=EXPERIMENT_NAME,
            )
        elif hypothesis_id and not valid:
            mark_hypothesis_failed(
                hypothesis_id,
                error_message="All LOSO folds failed — no valid results.",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
