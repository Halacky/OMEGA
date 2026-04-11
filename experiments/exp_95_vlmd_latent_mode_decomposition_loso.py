"""
Experiment 95: Variational Latent Mode Decomposition (VLMD) for EMG (LOSO)

Hypothesis
──────────
Multi-channel (C=12) EMG windows can be represented as a sparse linear
combination of M << C shared latent oscillatory modes via a globally-learned
mixing matrix A ∈ ℝ^{C×M}.  Unlike channel-independent VMD (exp_82, exp_93),
VLMD finds shared latent modes across all channels, naturally modelling
muscle synergies that are more stable across subjects than raw channel signals.

Architecture:
  1. VLMD Encoder  (global, subject-invariant)
       Z = X @ A,   X: (B, T, C) → Z: (B, T, M),  M << C
       A is a single learnable matrix shared across all subjects.
  2. Per-mode feature extraction (differentiable, shared weights)
       For each of the M latent modes: temporal stats + spectral band power.
       Feature vector: (B, M × (8 + num_bands)).
  3. MLP classifier on the concatenated mode features.

Regularisers (no data leakage — all operate on train-batch signals only):
  • Orthogonality  λ_orth  × ‖AᵀA − I_M‖_F² / M²
    Encourages A's columns to be orthonormal (independent modes).
  • Reconstruction λ_recon × ‖X − ZAᵀ‖_F² / ‖X‖_F²
    Ensures the M modes preserve the full C-channel signal.

LOSO Protocol (strictly enforced — zero subject adaptation)
────────────────────────────────────────────────────────────
  For each fold (test_subject ∈ ALL_SUBJECTS):
    train_subjects = ALL_SUBJECTS \ {test_subject}

    1. Load raw (N, T, C) windows for all subjects.
    2. Build train pool from train_subjects only.
       Val: 15 % carve from pooled training set (random split, no test data).
       Test: test_subject windows ONLY.
    3. Per-channel standardisation: mean/std from X_train.
       Apply the SAME statistics to X_val and X_test.
    4. Train VLMDClassifier end-to-end on (X_train, y_train):
         loss = CrossEntropy(logits, y)
              + λ_orth  × orthogonality_penalty(A)
              + λ_recon × reconstruction_penalty(X, Z)
    5. Evaluate on X_test with model.eval():
         BatchNorm frozen, Dropout disabled.
         No test-time adaptation of any kind.

Data-leakage guard summary
──────────────────────────
  ✓ VLMD encoder:  Z = X @ A  — a global linear map.  Per-sample, no cross-
                   sample information (no BatchNorm, no statistics computed
                   over the batch in the encode step itself).
  ✓ train/val sets built exclusively from train_subjects.
  ✓ X_test: test_subject windows ONLY — never seen during training.
  ✓ Standardisation: mean/std computed from X_train, applied to X_val, X_test.
  ✓ model.eval(): BatchNorm running statistics frozen at inference.
  ✓ Regularisation losses operate on training-batch signals and model params.
  ✓ common_gestures: derived from gesture ID sets (not signal values).
  ✓ Class weights: computed from y_train (train labels only).
  ✓ No test-data statistics computed anywhere in the code.

Run examples
────────────
  # 5-subject CI run (server-safe default):
  python experiments/exp_95_vlmd_latent_mode_decomposition_loso.py

  # Explicit flags:
  python experiments/exp_95_vlmd_latent_mode_decomposition_loso.py --ci
  python experiments/exp_95_vlmd_latent_mode_decomposition_loso.py \\
      --subjects DB2_s1,DB2_s12,DB2_s15,DB2_s28,DB2_s39

  # Architecture tuning:
  python experiments/exp_95_vlmd_latent_mode_decomposition_loso.py \\
      --num_modes 4 --num_bands 12 --lambda_orth 0.05 --lambda_recon 0.02

  # Full 20-subject run (local only — server has only CI subjects):
  python experiments/exp_95_vlmd_latent_mode_decomposition_loso.py --full

Success criterion: mean F1-macro ≥ 0.35.
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
from config.base import ProcessingConfig, SplitConfig
from data.multi_subject_loader import MultiSubjectLoader
from models.vlmd_classifier import VLMDClassifier
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ═══════════════════════════════════════════════════════════════════════════
# Experiment settings
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_95_vlmd_latent_mode_decomposition"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# VLMD architecture
NUM_MODES       = 6       # M — latent modes (M < C=12)
NUM_BANDS       = 8       # spectral frequency bands per mode
HIDDEN_DIM      = 128     # MLP hidden layer width
DROPOUT         = 0.3

# Training
BATCH_SIZE      = 128
EPOCHS          = 100
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
PATIENCE        = 20
GRAD_CLIP       = 1.0

# Regularisation weights
LAMBDA_ORTH     = 0.1     # orthogonality penalty weight
LAMBDA_RECON    = 0.05    # reconstruction penalty weight


# ═══════════════════════════════════════════════════════════════════════════
# Local helper: grouped_to_arrays
# NOTE: this function does NOT exist in any processing/ module.
#       It MUST be defined locally in every experiment that needs it.
# ═══════════════════════════════════════════════════════════════════════════

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
    gesture_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten grouped_windows to (windows, labels) arrays.

    Parameters
    ----------
    grouped_windows : {gesture_id: [rep_array, ...]},
                      each rep_array has shape (N_rep, T, C).
    gesture_ids     : optional ordered list of gesture IDs to include.
                      If None, uses sorted(grouped_windows.keys()).

    Returns
    -------
    windows : (N, T, C) float32
    labels  : (N,) int64  — gesture IDs (NOT class indices).
    """
    if gesture_ids is None:
        gesture_ids = sorted(grouped_windows.keys())

    all_windows: List[np.ndarray] = []
    all_labels:  List[np.ndarray] = []

    for gid in gesture_ids:
        if gid not in grouped_windows:
            continue
        for rep_arr in grouped_windows[gid]:
            if (
                isinstance(rep_arr, np.ndarray)
                and rep_arr.ndim == 3
                and len(rep_arr) > 0
            ):
                all_windows.append(rep_arr)
                all_labels.append(np.full(len(rep_arr), gid, dtype=np.int64))

    if not all_windows:
        return np.empty((0, 1, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return (
        np.concatenate(all_windows, axis=0).astype(np.float32),
        np.concatenate(all_labels,  axis=0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# LOSO split builder
# ═══════════════════════════════════════════════════════════════════════════

def build_loso_splits(
    subjects_data:   Dict[str, tuple],
    train_subjects:  List[str],
    test_subject:    str,
    common_gestures: List[int],
    val_ratio:       float = 0.15,
    seed:            int   = 42,
) -> Tuple[
    np.ndarray, np.ndarray,   # X_train, y_train
    np.ndarray, np.ndarray,   # X_val,   y_val
    np.ndarray, np.ndarray,   # X_test,  y_test
    Dict[int, int],           # gesture_to_class
]:
    """
    Build strict LOSO train/val/test splits.

    LOSO invariants enforced here:
      - train + val: pooled exclusively from train_subjects.
      - val carved from the pooled training set (no test data ever involved).
      - test: test_subject windows ONLY.
      - No signal statistics computed here (done later from train only).

    Parameters
    ----------
    subjects_data   : {subj_id: (emg, segments, grouped_windows)}
    train_subjects  : subjects contributing to training/validation.
    test_subject    : held-out subject.
    common_gestures : sorted list of shared gesture IDs.
    val_ratio       : fraction of train pool reserved for validation.
    seed            : RNG seed for reproducibility.

    Returns
    -------
    X_train, y_train : (N_tr, T, C), (N_tr,) — class-index labels
    X_val,   y_val   : (N_v,  T, C), (N_v,)
    X_test,  y_test  : (N_te, T, C), (N_te,)
    gesture_to_class : {gesture_id: class_index}
    """
    rng = np.random.RandomState(seed)
    gesture_to_class = {g: i for i, g in enumerate(common_gestures)}

    # ── Collect all training windows from train_subjects ONLY ─────────────
    train_wins: List[np.ndarray] = []
    train_labs: List[np.ndarray] = []

    for subj_id in sorted(train_subjects):   # sorted for reproducibility
        if subj_id not in subjects_data:
            continue
        # subjects_data values are tuples — unpack, not dict-access
        _, _, grouped = subjects_data[subj_id]
        wins, gid_labels = grouped_to_arrays(grouped, common_gestures)
        if len(wins) == 0:
            continue
        cls_labels = np.array(
            [gesture_to_class[g] for g in gid_labels], dtype=np.int64
        )
        train_wins.append(wins)
        train_labs.append(cls_labels)

    if not train_wins:
        raise ValueError(
            "No training windows collected. Check subject list and gestures."
        )

    X_all = np.concatenate(train_wins, axis=0)   # (N, T, C)
    y_all = np.concatenate(train_labs, axis=0)   # (N,)

    # ── Shuffle and carve val from the pooled training set ────────────────
    n     = len(X_all)
    idx   = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))

    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    X_train, y_train = X_all[trn_idx], y_all[trn_idx]
    X_val,   y_val   = X_all[val_idx], y_all[val_idx]

    # ── Collect test windows (test_subject data ONLY) ─────────────────────
    if test_subject not in subjects_data:
        raise ValueError(f"Test subject '{test_subject}' not found in subjects_data.")

    _, _, test_grouped = subjects_data[test_subject]
    X_test_raw, test_gid_labels = grouped_to_arrays(test_grouped, common_gestures)

    if len(X_test_raw) == 0:
        raise ValueError(f"No test windows for subject {test_subject}.")

    y_test = np.array(
        [gesture_to_class[g] for g in test_gid_labels], dtype=np.int64
    )

    return X_train, y_train, X_val, y_val, X_test_raw, y_test, gesture_to_class


# ═══════════════════════════════════════════════════════════════════════════
# Standardisation — train statistics ONLY
# ═══════════════════════════════════════════════════════════════════════════

def standardize_channels(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel z-score normalisation using TRAIN statistics only.

    Mean and std are computed over all N_train samples and all T time steps
    of the training set, giving per-channel scalars (broadcasted over N and T).
    The SAME mean/std are then applied to validation and test data.

    LEAKAGE GUARD: test data statistics are never computed or used here.

    X arrays: (N, T, C).  Returns standardised copies (float32).
    """
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)     # (1, 1, C)
    std_c  = X_train.std(axis=(0, 1),  keepdims=True)
    std_c  = np.maximum(std_c, 1e-8)

    X_train = (X_train - mean_c) / std_c
    X_val   = (X_val   - mean_c) / std_c
    X_test  = (X_test  - mean_c) / std_c

    return X_train, X_val, X_test


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    model:         VLMDClassifier,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    num_classes:   int,
    y_train:       np.ndarray,
    device:        torch.device,
    epochs:        int,
    lr:            float,
    weight_decay:  float,
    patience:      int,
    grad_clip:     float,
    lambda_orth:   float,
    lambda_recon:  float,
    logger,
) -> Tuple[VLMDClassifier, Dict]:
    """
    Train VLMDClassifier with CE loss + orthogonality + reconstruction regularisers.

    Total loss (per batch):
        L = CrossEntropy(logits, y)
          + λ_orth  × ‖AᵀA − I_M‖_F² / M²          (orthogonality penalty)
          + λ_recon × ‖X − ZAᵀ‖_F² / ‖X‖_F²          (reconstruction penalty)

    Both regularisers operate on model parameters and training-batch signals.
    No test-data statistics involved — no data leakage.

    Uses:
      - Class-weighted CrossEntropy (weights from TRAIN labels only).
      - Adam optimiser with ReduceLROnPlateau on validation loss.
      - Early stopping on validation loss.
      - Gradient clipping.

    Validation loss uses CrossEntropy only (no regularisation) to give a clean
    measure of classification performance on the held-out validation set.

    Returns the best checkpoint (by val loss) and a training history dict.
    """
    # Class weights from TRAIN labels only
    counts  = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts  = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    ce_criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(weights).to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # NOTE: verbose=True removed in PyTorch 2.4+ — do NOT add it.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None
    history: Dict = {
        "train_loss": [],
        "train_ce_loss": [],
        "train_reg_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        running_total  = 0.0
        running_ce     = 0.0
        running_reg    = 0.0
        n_batches      = 0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device)          # (B, T, C)
            y_b = y_b.to(device)

            optimizer.zero_grad()

            # forward_with_modes returns (logits, Z) so we can compute
            # the reconstruction regulariser without a second encoder call.
            logits, Z = model.forward_with_modes(X_b)

            ce_loss  = ce_criterion(logits, y_b)
            reg_loss = model.regularisation_loss(
                X_b, Z,
                lambda_orth=lambda_orth,
                lambda_recon=lambda_recon,
            )
            loss = ce_loss + reg_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_total += loss.item()
            running_ce    += ce_loss.item()
            running_reg   += reg_loss.item()
            n_batches     += 1

        avg_train     = running_total / max(n_batches, 1)
        avg_train_ce  = running_ce    / max(n_batches, 1)
        avg_train_reg = running_reg   / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────────
        # Validation uses CE only (no regularisation) — clean performance signal.
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0
        n_val       = 0

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b   = X_b.to(device)
                y_b   = y_b.to(device)
                logits = model(X_b)                     # standard forward (no Z)
                loss_v = ce_criterion(logits, y_b)
                val_loss    += loss_v.item()
                n_val       += 1
                preds        = logits.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total   += len(y_b)

        avg_val = val_loss / max(n_val, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["train_ce_loss"].append(avg_train_ce)
        history["train_reg_loss"].append(avg_train_reg)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"train={avg_train:.4f} (ce={avg_train_ce:.4f}, reg={avg_train_reg:.4f}) | "
                f"val={avg_val:.4f} | val_acc={val_acc:.4f} | lr={cur_lr:.2e}"
            )

        # ── Early stopping ────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(
                    f"  Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience} epochs)."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    history["best_epoch"]   = len(history["train_loss"]) - patience_ctr
    history["total_epochs"] = len(history["train_loss"])
    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model:       VLMDClassifier,
    loader:      DataLoader,
    device:      torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate model on a held-out dataset.

    model.eval() is called here to ensure:
      - BatchNorm uses frozen running statistics (no test-data adaptation).
      - Dropout is disabled.

    Returns accuracy, F1-macro, per-class classification report, and the
    confusion matrix.
    """
    model.eval()
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b    = X_b.to(device)
            logits = model(X_b)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_b.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc    = float(accuracy_score(y_true, y_pred))
    f1     = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy":         acc,
        "f1_macro":         f1,
        "report":           report,
        "confusion_matrix": cm,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single LOSO fold
# ═══════════════════════════════════════════════════════════════════════════

def run_single_loso_fold(
    subjects_data:   Dict[str, tuple],
    train_subjects:  List[str],
    test_subject:    str,
    common_gestures: List[int],
    output_dir:      Path,
    cfg:             Dict,
    logger,
) -> Dict:
    """
    Execute one complete LOSO fold:
      build splits → standardise → create model → train → evaluate.

    Parameters
    ----------
    subjects_data   : {subj_id: (emg, segments, grouped_windows)}
    train_subjects  : subjects contributing to train/val (all except test).
    test_subject    : held-out subject.
    common_gestures : sorted list of gesture IDs shared across all subjects.
    output_dir      : directory for this fold's output files.
    cfg             : hyperparameter dict.
    logger          : logging instance.

    Returns
    -------
    dict with test_accuracy, test_f1_macro, and analysis info.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"])
    seed_everything(cfg["seed"], verbose=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO Fold: test={test_subject}")
    logger.info(f"  train: {train_subjects}")
    logger.info(f"{'='*60}")

    # ── Build LOSO splits ─────────────────────────────────────────────────
    try:
        (X_train, y_train, X_val, y_val,
         X_test,  y_test,  gesture_to_class) = build_loso_splits(
            subjects_data   = subjects_data,
            train_subjects  = train_subjects,
            test_subject    = test_subject,
            common_gestures = common_gestures,
            val_ratio       = cfg["val_ratio"],
            seed            = cfg["seed"],
        )
    except ValueError as exc:
        logger.error(f"Split building failed: {exc}")
        return {
            "test_subject":  test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(exc),
        }

    num_classes = len(common_gestures)
    in_channels = X_train.shape[2]   # C

    logger.info(
        f"  Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} | "
        f"classes={num_classes}, channels={in_channels}"
    )

    # ── Per-channel standardisation (TRAIN stats only) ────────────────────
    # LEAKAGE GUARD: mean/std computed from X_train only, NEVER from X_test.
    X_train, X_val, X_test = standardize_channels(X_train, X_val, X_test)

    # ── Convert to tensors ────────────────────────────────────────────────
    # VLMDClassifier.forward() expects (B, T, C) — same as array layout.
    # No transpose needed: windows already have shape (N, T, C).
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.LongTensor(y_val)
    X_test_t  = torch.FloatTensor(X_test)
    y_test_t  = torch.LongTensor(y_test)

    del X_train, X_val, X_test
    gc.collect()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t,   y_val_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False
    )
    val_loader   = DataLoader(val_ds,  batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    # ── Create model ──────────────────────────────────────────────────────
    # Verify num_modes < in_channels (VLMD constraint).
    num_modes = cfg["num_modes"]
    if num_modes >= in_channels:
        num_modes = in_channels - 1
        logger.warning(
            f"  num_modes reduced to {num_modes} "
            f"(must be < in_channels={in_channels})."
        )

    model = VLMDClassifier(
        in_channels = in_channels,
        num_modes   = num_modes,
        num_classes = num_classes,
        num_bands   = cfg["num_bands"],
        hidden_dim  = cfg["hidden_dim"],
        dropout     = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc    = sum(p.numel() for p in model.encoder.parameters())
    n_cls    = sum(p.numel() for p in model.classifier.parameters())
    feat_dim = num_modes * (8 + cfg["num_bands"])
    logger.info(
        f"  Model: VLMDClassifier | total={n_params:,} params "
        f"(encoder={n_enc} [{in_channels}×{num_modes}], "
        f"feat_dim={feat_dim}, classifier={n_cls})"
    )

    # Log initial mixing matrix quality (should be near-orthonormal from QR init)
    init_analysis = model.analyse_modes()
    logger.info(
        f"  Initial A: orthogonality_err={init_analysis['orthogonality']:.4f} "
        f"(0=perfect), mean_col_norm={init_analysis['mean_col_norm']:.4f}"
    )

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    model, history = train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_classes  = num_classes,
        y_train      = y_train_t.numpy(),
        device       = device,
        epochs       = cfg["epochs"],
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
        patience     = cfg["patience"],
        grad_clip    = cfg["grad_clip"],
        lambda_orth  = cfg["lambda_orth"],
        lambda_recon = cfg["lambda_recon"],
        logger       = logger,
    )
    train_time = time.time() - t0
    logger.info(
        f"  Training: {train_time:.1f}s | "
        f"{history['total_epochs']} epochs (best={history['best_epoch']})"
    )

    # Log final mixing matrix analysis
    final_analysis = model.analyse_modes()
    logger.info(
        f"  Final A: orthogonality_err={final_analysis['orthogonality']:.4f}, "
        f"mean_col_norm={final_analysis['mean_col_norm']:.4f}, "
        f"col_norms={[f'{v:.3f}' for v in final_analysis['col_norms']]}"
    )

    # ── Evaluate (model.eval() — zero test-time adaptation) ───────────────
    class_names  = [f"gesture_{g}" for g in common_gestures]
    test_results = evaluate_model(model, test_loader, device, class_names)

    test_acc = test_results["accuracy"]
    test_f1  = test_results["f1_macro"]
    logger.info(f"  Test: accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}")

    # ── Save fold outputs ─────────────────────────────────────────────────
    fold_result = {
        "test_subject":    test_subject,
        "train_subjects":  train_subjects,
        "common_gestures": [int(g) for g in common_gestures],
        "num_classes":     num_classes,
        "model_params":    n_params,
        "arch": {
            "in_channels": in_channels,
            "num_modes":   num_modes,
            "num_bands":   cfg["num_bands"],
            "hidden_dim":  cfg["hidden_dim"],
            "feat_dim":    feat_dim,
        },
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
        "vlmd_analysis": {
            "init_orthogonality": init_analysis["orthogonality"],
            "final_orthogonality": final_analysis["orthogonality"],
            "init_mean_col_norm": init_analysis["mean_col_norm"],
            "final_mean_col_norm": final_analysis["mean_col_norm"],
            "final_col_norms":    final_analysis["col_norms"],
            # Gram matrix (M×M) shows residual mode correlation
            "final_gram_matrix":  final_analysis["gram_matrix"],
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
        "test_subject":         test_subject,
        "test_accuracy":        test_acc,
        "test_f1_macro":        test_f1,
        "train_time_s":         round(train_time, 1),
        "final_orthogonality":  final_analysis["orthogonality"],
        "final_col_norms":      final_analysis["col_norms"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    LOSO evaluation loop for VLMD experiment.

    Subject list priority (safe server default = CI_TEST_SUBJECTS):
      1. --subjects DB2_s1,DB2_s12,...  — explicit comma-separated list
      2. --full                         — all 20 DEFAULT_SUBJECTS (local only)
      3. --ci or no flag                — 5 CI_TEST_SUBJECTS  ← vast.ai safe

    Architecture overrides:
      --num_modes   M    — latent modes (default 6, must be < C)
      --num_bands   N    — spectral bands per mode (default 8)
      --lambda_orth λ    — orthogonality penalty weight (default 0.1)
      --lambda_recon λ   — reconstruction penalty weight (default 0.05)
    """
    import argparse

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects",      type=str,   default=None)
    _parser.add_argument("--ci",            action="store_true")
    _parser.add_argument("--full",          action="store_true")
    _parser.add_argument("--exercises",     type=str,   default=None)
    _parser.add_argument("--num_modes",     type=int,   default=NUM_MODES)
    _parser.add_argument("--num_bands",     type=int,   default=NUM_BANDS)
    _parser.add_argument("--hidden_dim",    type=int,   default=HIDDEN_DIM)
    _parser.add_argument("--lambda_orth",   type=float, default=LAMBDA_ORTH)
    _parser.add_argument("--lambda_recon",  type=float, default=LAMBDA_RECON)
    _args, _ = _parser.parse_known_args()

    # ── Subject list ──────────────────────────────────────────────────────
    # Default is CI_TEST_SUBJECTS — safe on vast.ai server (only CI symlinks).
    # _FULL_SUBJECTS activates ONLY when --full is explicitly passed.
    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS      # 20 subjects — local only
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS      # 5 subjects — server-safe default

    exercises = (
        [e.strip() for e in _args.exercises.split(",")]
        if _args.exercises else EXERCISES
    )

    # ── Hyperparameters ───────────────────────────────────────────────────
    cfg = {
        "num_modes":    _args.num_modes,
        "num_bands":    _args.num_bands,
        "hidden_dim":   _args.hidden_dim,
        "dropout":      DROPOUT,
        "batch_size":   BATCH_SIZE,
        "epochs":       EPOCHS,
        "lr":           LR,
        "weight_decay": WEIGHT_DECAY,
        "patience":     PATIENCE,
        "grad_clip":    GRAD_CLIP,
        "lambda_orth":  _args.lambda_orth,
        "lambda_recon": _args.lambda_recon,
        "val_ratio":    0.15,
        "seed":         42,
        "device":       "cuda" if torch.cuda.is_available() else "cpu",
    }

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_ROOT)

    feat_dim_info = cfg["num_modes"] * (8 + cfg["num_bands"])
    print("=" * 80)
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis  : VLMD — M={cfg['num_modes']} latent modes from C channels\n"
        f"              via global mixing matrix A ∈ ℝ^{{C×{cfg['num_modes']}}}.\n"
        f"              Temporal + spectral features per mode → {feat_dim_info}-dim vector.\n"
        f"              Loss = CE "
        f"+ {cfg['lambda_orth']}×orth_penalty "
        f"+ {cfg['lambda_recon']}×recon_penalty"
    )
    print(f"Subjects    : {ALL_SUBJECTS}")
    print(f"Exercises   : {exercises}")
    print(f"Device      : {cfg['device']}")
    print(f"Output      : {OUTPUT_ROOT}")
    print("=" * 80)

    # ── Processing config ─────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size         = 200,
        window_overlap      = 100,
        sampling_rate       = 2000,
        segment_edge_margin = 0.1,
    )
    split_cfg = SplitConfig(
        train_ratio             = 0.70,
        val_ratio               = 0.15,
        test_ratio              = 0.15,
        mode                    = "by_segments",
        shuffle_segments        = True,
        seed                    = 42,
        include_rest_in_splits  = False,
    )

    proc_cfg.save(OUTPUT_ROOT / "processing_config.json")
    with open(OUTPUT_ROOT / "split_config.json", "w") as fh:
        json.dump(asdict(split_cfg), fh, indent=4)
    with open(OUTPUT_ROOT / "vlmd_config.json", "w") as fh:
        json.dump(make_json_serializable(cfg), fh, indent=4)

    # ── Load all subjects ─────────────────────────────────────────────────
    logger.info("Loading all subjects...")
    multi_loader = MultiSubjectLoader(
        processing_config       = proc_cfg,
        logger                  = logger,
        use_gpu                 = True,
        use_improved_processing = USE_IMPROVED,
    )

    # subjects_data: {subj_id: (emg, segments, grouped_windows)}
    # Values are TUPLES — unpack as (emg, segs, grouped), do NOT dict-access.
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir    = BASE_DIR,
        subject_ids = ALL_SUBJECTS,
        exercises   = exercises,
        include_rest= split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES
    )
    logger.info(
        f"Common gestures: {common_gestures} ({len(common_gestures)} total)"
    )

    # ── LOSO loop ─────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    t_loso_start = time.time()

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir       = OUTPUT_ROOT / f"fold_test_{test_subject}"

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
        except Exception as exc:
            logger.error(f"Fold {test_subject} failed: {exc}")
            traceback.print_exc()
            result = {
                "test_subject":  test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error":         str(exc),
            }

        all_results.append(result)

    t_loso_total = time.time() - t_loso_start

    # ── Aggregate LOSO summary ────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]

    if valid:
        accs  = [r["test_accuracy"] for r in valid]
        f1s   = [r["test_f1_macro"] for r in valid]
        times = [r.get("train_time_s", 0.0) for r in valid]
        ortho = [r.get("final_orthogonality", None) for r in valid]
        ortho = [v for v in ortho if v is not None]

        mean_acc  = float(np.mean(accs))
        std_acc   = float(np.std(accs))
        mean_f1   = float(np.mean(f1s))
        std_f1    = float(np.std(f1s))
        mean_t    = float(np.mean(times))

        # Aggregate learned column norms across folds (M values per fold)
        norms_per_fold = [
            r.get("final_col_norms", []) for r in valid
            if r.get("final_col_norms")
        ]
        if norms_per_fold:
            norms_arr      = np.array(norms_per_fold)          # (n_folds, M)
            mean_col_norms = norms_arr.mean(axis=0).tolist()
            std_col_norms  = norms_arr.std(axis=0).tolist()
        else:
            mean_col_norms, std_col_norms = [], []

        criterion_met = mean_f1 >= 0.35

        # Guard .4f formatting — values are not None here since we filtered valid
        print("\n" + "=" * 80)
        print(f"LOSO SUMMARY — VLMD (M={cfg['num_modes']} modes, {cfg['num_bands']} bands/mode)")
        print(f"  Subjects evaluated : {len(valid)}")
        print(
            f"  Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}"
            f"  (min={min(accs):.4f}, max={max(accs):.4f})"
        )
        print(
            f"  F1-macro  : {mean_f1:.4f} ± {std_f1:.4f}"
            f"  (min={min(f1s):.4f}, max={max(f1s):.4f})"
        )
        print(f"  Train time: {mean_t:.1f}s/fold (total LOSO: {t_loso_total:.1f}s)")
        if ortho:
            print(f"  Orth err  : mean={float(np.mean(ortho)):.4f} (0=perfect)")
        if mean_col_norms:
            print(
                f"  Col norms : mean={[f'{v:.3f}' for v in mean_col_norms]}"
                f"  (± {[f'{v:.3f}' for v in std_col_norms]})"
            )
        print(
            f"  Success   : {'YES' if criterion_met else 'NO'} "
            f"(F1≥0.35: {mean_f1>=0.35})"
        )
        print("=" * 80)

        summary = {
            "experiment":   EXPERIMENT_NAME,
            "model":        "VLMDClassifier",
            "subjects":     ALL_SUBJECTS,
            "exercises":    exercises,
            "vlmd_config": {
                "num_modes":    cfg["num_modes"],
                "num_bands":    cfg["num_bands"],
                "hidden_dim":   cfg["hidden_dim"],
                "lambda_orth":  cfg["lambda_orth"],
                "lambda_recon": cfg["lambda_recon"],
                "feat_dim":     feat_dim_info,
            },
            "loso_metrics": {
                "mean_accuracy": mean_acc,
                "std_accuracy":  std_acc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro":  std_f1,
                "per_subject":   all_results,
            },
            "timing": {
                "mean_fold_time_s":  mean_t,
                "total_loso_time_s": round(t_loso_total, 1),
            },
            "vlmd_convergence": {
                "mean_orthogonality_error": float(np.mean(ortho)) if ortho else None,
                "mean_col_norms_across_folds": mean_col_norms,
                "std_col_norms_across_folds":  std_col_norms,
                "interpretation": (
                    "orthogonality_error near 0 indicates the mixing matrix A "
                    "maintained approximately orthonormal columns, ensuring the M "
                    "latent modes remain linearly independent.  "
                    "Stable col_norms across folds indicate a consistent, "
                    "subject-invariant decomposition was discovered."
                ),
            },
            "success_criterion": {
                "f1_ge_035": bool(mean_f1 >= 0.35),
                "met":       bool(criterion_met),
            },
        }

        with open(OUTPUT_ROOT / "loso_summary.json", "w") as fh:
            json.dump(
                make_json_serializable(summary), fh, indent=4, ensure_ascii=False
            )
        print(f"Summary saved → {OUTPUT_ROOT / 'loso_summary.json'}")

    else:
        print("No successful folds to summarise.")

    # ── Hypothesis executor (optional, guarded import) ────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        hypothesis_id = os.environ.get("HYPOTHESIS_ID", "")
        if hypothesis_id:
            if valid:
                mark_hypothesis_verified(
                    hypothesis_id,
                    metrics={
                        "mean_accuracy":    mean_acc,
                        "std_accuracy":     std_acc,
                        "mean_f1_macro":    mean_f1,
                        "std_f1_macro":     std_f1,
                        "mean_fold_time_s": mean_t,
                        "n_folds":          len(valid),
                    },
                    experiment_name=EXPERIMENT_NAME,
                )
            else:
                mark_hypothesis_failed(
                    hypothesis_id,
                    "All LOSO folds failed — no valid results.",
                )
    except ImportError:
        pass   # hypothesis_executor not installed — silently skip


if __name__ == "__main__":
    main()
