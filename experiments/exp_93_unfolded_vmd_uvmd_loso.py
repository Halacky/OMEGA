"""
Experiment 93: Unfolded VMD (UVMD) for EMG Gesture Classification (LOSO)

Hypothesis
──────────
Replacing classical iterative VMD with L unrolled ADMM iterations (deep
unfolding) and making all VMD parameters — bandwidth constraints α_{l,k},
dual step sizes τ_l, and mode centre frequencies ω_k — learnable through
backpropagation enables the network to discover a frequency decomposition
that is jointly optimal for cross-subject gesture classification.

Key advantages over exp_82 (classical VMD):
  1. Speed: UVMD runs entirely on GPU as a single differentiable pass.
     No iterative preprocessing — VMD loops are unrolled into L fixed
     layers.  Expected < 5 min/fold vs 49 min/fold for classical VMD.
  2. Task-aligned decomposition: alpha, tau, omega are optimised to
     minimise classification loss (+ spectral overlap regulariser),
     not just to reconstruct the signal.
  3. Fully differentiable: end-to-end gradient flow from logits to
     the VMD parameters.

LOSO Protocol (strictly enforced — zero adaptation)
────────────────────────────────────────────────────
  For each fold (test_subject ∈ ALL_SUBJECTS):
    train_subjects = ALL_SUBJECTS \ {test_subject}

    1. Collect raw (N, T, C) windows from train_subjects only.
    2. Carve val_ratio from the pooled training windows (RNG split).
       NO test-subject data is used here.
    3. Collect raw (N, T, C) windows from test_subject only.
    4. Per-channel standardisation: mean/std computed from X_train.
       Apply the SAME statistics to X_val and X_test.
    5. Train UVMDClassifier end-to-end on (X_train, y_train):
         loss = CrossEntropy(logits, y) + λ·SpectralOverlapPenalty(ω)
    6. Evaluate on X_test with model.eval() — BatchNorm frozen,
       Dropout disabled.  No test-time adaptation of any kind.

Data-leakage guard summary
──────────────────────────
  ✓ UVMDBlock forward: per-window, per-channel — no cross-sample info.
  ✓ train/val built exclusively from train_subjects.
  ✓ X_test: test_subject data ONLY — never seen during training.
  ✓ Standardisation: mean/std from X_train, applied to X_val, X_test.
  ✓ model.eval(): BatchNorm running stats frozen at inference.
  ✓ No test-data statistics computed anywhere.
  ✓ common_gestures: derived from gesture ID sets (not signal values).
  ✓ SpectralOverlapPenalty operates on ω (model param), not data.
  ✓ VMD parameters (alpha, tau, omega) are global — same for every subject.

Run examples
────────────
  # 5-subject CI run (server-safe default):
  python experiments/exp_93_unfolded_vmd_uvmd_loso.py --ci

  # Explicit subject list:
  python experiments/exp_93_unfolded_vmd_uvmd_loso.py \\
      --subjects DB2_s1,DB2_s12,DB2_s15,DB2_s28,DB2_s39

  # Tune UVMD depth:
  python experiments/exp_93_unfolded_vmd_uvmd_loso.py --ci --num_modes 6 --num_layers 12

  # Full 20-subject run (local only — server has only CI subjects):
  python experiments/exp_93_unfolded_vmd_uvmd_loso.py --full

Success criterion: mean F1-macro ≥ 0.35, training time < 5 min/fold.
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
from models.uvmd_classifier import UVMDClassifier
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ═══════════════════════════════════════════════════════════════════════════
# Experiment settings
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_93_unfolded_vmd_uvmd"
EXERCISES       = ["E1",]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# UVMD architecture
NUM_MODES       = 4      # K — number of decomposed frequency modes
NUM_LAYERS      = 8      # L — number of unrolled ADMM iterations
ALPHA_INIT      = 2000.0 # initial bandwidth constraint (classic VMD default)
TAU_INIT        = 0.01   # initial dual step size (near 0 = noise-free)
FEAT_DIM        = 64     # feature dimension per mode CNN branch
HIDDEN_DIM      = 128    # hidden units in classifier MLP
DROPOUT         = 0.3

# Training
BATCH_SIZE      = 64
EPOCHS          = 80
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
PATIENCE        = 15
GRAD_CLIP       = 1.0
OVERLAP_LAMBDA  = 0.01   # weight of spectral overlap regulariser
OVERLAP_SIGMA   = 0.05   # Gaussian kernel width for overlap penalty


# ═══════════════════════════════════════════════════════════════════════════
# Local helper — must be defined here (not importable from processing/)
# ═══════════════════════════════════════════════════════════════════════════

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
    gesture_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert grouped_windows to flat (windows, labels) arrays.

    NOTE: grouped_to_arrays does NOT exist in any processing/ module.
    This helper MUST be defined locally in every experiment that needs it.

    Parameters
    ----------
    grouped_windows : Dict[gesture_id -> list of rep_arrays]
        Each rep_array has shape (N_rep, T, C).
    gesture_ids : optional list of gesture IDs to include (in order).
        If None, uses sorted(grouped_windows.keys()).

    Returns
    -------
    windows : np.ndarray, shape (N, T, C), float32
    labels  : np.ndarray, shape (N,), int64
        Values are gesture IDs (NOT class indices).
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
                all_labels.append(
                    np.full(len(rep_arr), gid, dtype=np.int64)
                )

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
    subjects_data: Dict[str, tuple],
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray,   # X_train, y_train
    np.ndarray, np.ndarray,   # X_val,   y_val
    np.ndarray, np.ndarray,   # X_test,  y_test
    Dict[int, int],           # gesture_to_class
]:
    """
    Build strict LOSO train/val/test splits from raw grouped windows.

    LOSO invariants:
      - train + val: pooled from train_subjects ONLY.
      - val carved from the pooled training set (no test data).
      - test: test_subject windows ONLY.
      - No signal statistics computed here (done later from train only).

    Parameters
    ----------
    subjects_data : {subj_id: (emg, segments, grouped_windows)}
    train_subjects : subjects contributing to training/validation.
    test_subject : held-out subject.
    common_gestures : sorted list of gesture IDs present in all subjects.
    val_ratio : fraction of train pool reserved for validation.
    seed : RNG seed for reproducibility.

    Returns
    -------
    X_train, y_train : (N_tr, T, C), (N_tr,)  — class-index labels
    X_val,   y_val   : (N_v,  T, C), (N_v,)
    X_test,  y_test  : (N_te, T, C), (N_te,)
    gesture_to_class : {gesture_id: class_index}
    """
    rng = np.random.RandomState(seed)
    gesture_to_class = {g: i for i, g in enumerate(common_gestures)}

    # ── Collect all training windows ──────────────────────────────────────
    train_wins:  List[np.ndarray] = []
    train_labs:  List[np.ndarray] = []

    for subj_id in sorted(train_subjects):   # sorted for reproducibility
        if subj_id not in subjects_data:
            continue
        _, _, grouped = subjects_data[subj_id]   # unpack tuple
        wins, gid_labels = grouped_to_arrays(grouped, common_gestures)
        if len(wins) == 0:
            continue
        # Convert gesture IDs to class indices
        cls_labels = np.array(
            [gesture_to_class[g] for g in gid_labels], dtype=np.int64
        )
        train_wins.append(wins)
        train_labs.append(cls_labels)

    if not train_wins:
        raise ValueError("No training windows collected — check subjects/gestures.")

    X_all = np.concatenate(train_wins, axis=0)   # (N, T, C)
    y_all = np.concatenate(train_labs, axis=0)   # (N,)

    # ── Shuffle and split into train / val ────────────────────────────────
    n = len(X_all)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))

    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    X_train, y_train = X_all[trn_idx], y_all[trn_idx]
    X_val,   y_val   = X_all[val_idx], y_all[val_idx]

    # ── Collect test windows (test_subject ONLY) ──────────────────────────
    if test_subject not in subjects_data:
        raise ValueError(f"Test subject '{test_subject}' not in subjects_data.")

    _, _, test_grouped = subjects_data[test_subject]   # unpack tuple
    X_test_raw, test_gid_labels = grouped_to_arrays(test_grouped, common_gestures)

    if len(X_test_raw) == 0:
        raise ValueError(f"No test windows for subject {test_subject}.")

    y_test = np.array(
        [gesture_to_class[g] for g in test_gid_labels], dtype=np.int64
    )
    X_test = X_test_raw

    return X_train, y_train, X_val, y_val, X_test, y_test, gesture_to_class


# ═══════════════════════════════════════════════════════════════════════════
# Standardisation (per-channel, train statistics only)
# ═══════════════════════════════════════════════════════════════════════════

def standardize_channels(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel z-score normalisation using TRAIN statistics only.

    Statistics are computed over all samples AND all time steps of the
    training set, then applied identically to validation and test sets.

    X arrays have shape (N, T, C).  Returns standardised copies.
    """
    # Aggregate over N and T to get per-channel mean/std: shape (1, 1, C)
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c  = X_train.std(axis=(0, 1),  keepdims=True)
    std_c  = np.maximum(std_c, 1e-8)   # avoid division by zero

    X_train = (X_train - mean_c) / std_c
    X_val   = (X_val   - mean_c) / std_c
    X_test  = (X_test  - mean_c) / std_c

    return X_train, X_val, X_test


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    model: UVMDClassifier,
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
    overlap_lambda: float,
    overlap_sigma:  float,
    logger,
) -> Tuple[UVMDClassifier, Dict]:
    """
    Train UVMDClassifier with CE loss + spectral overlap regulariser.

    Loss = CrossEntropy(logits, y) + overlap_lambda × SpectralOverlapPenalty(ω)

    The overlap penalty prevents mode centre frequencies from collapsing
    to the same value, ensuring diverse frequency coverage.  It operates
    on the model parameter ω — not on any data — so it cannot leak test
    information.

    Uses:
      - Class-weighted CrossEntropy (computed from TRAIN labels only).
      - Adam optimiser with ReduceLROnPlateau on validation loss.
      - Early stopping on validation loss.
      - Gradient clipping.

    Returns trained model (best checkpoint) and training history.
    """
    # Class weights computed from TRAIN labels only
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    ce_criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(weights).to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # NOTE: verbose=True removed in PyTorch 2.4+ — do NOT use it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None
    history       = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device)   # (B, T, C)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(X_b)                            # (B, num_classes)
            ce_loss = ce_criterion(logits, y_b)

            # Spectral overlap regulariser — operates on model params only
            overlap_loss = model.spectral_overlap_penalty(sigma=overlap_sigma)
            loss = ce_loss + overlap_lambda * overlap_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        avg_train = running_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0
        n_val_batches = 0

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                y_b = y_b.to(device)
                logits   = model(X_b)
                loss_v   = ce_criterion(logits, y_b)
                val_loss += loss_v.item()
                n_val_batches += 1
                preds       = logits.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total   += len(y_b)

        avg_val = val_loss / max(n_val_batches, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss={avg_train:.4f} | "
                f"val_loss={avg_val:.4f} | val_acc={val_acc:.4f} | "
                f"lr={cur_lr:.2e}"
            )

        # ── Early stopping ────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr  = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(
                    f"  Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    history["best_epoch"]    = len(history["train_loss"]) - patience_ctr
    history["total_epochs"]  = len(history["train_loss"])
    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model:       UVMDClassifier,
    loader:      DataLoader,
    device:      torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate model on held-out data.

    model.eval() is called here to ensure:
      - BatchNorm uses frozen running statistics (no test-data adaptation).
      - Dropout is disabled.

    Returns accuracy, F1-macro, per-class report, and confusion matrix.
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
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
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
    Execute one LOSO fold.

    Parameters
    ----------
    subjects_data   : {subj_id: (emg, segments, grouped_windows)}
    train_subjects  : subjects used for training (all except test_subject).
    test_subject    : held-out subject.
    common_gestures : sorted list of gesture IDs shared across all subjects.
    output_dir      : directory for this fold's outputs.
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
            subjects_data    = subjects_data,
            train_subjects   = train_subjects,
            test_subject     = test_subject,
            common_gestures  = common_gestures,
            val_ratio        = cfg["val_ratio"],
            seed             = cfg["seed"],
        )
    except ValueError as exc:
        logger.error(f"Split building failed: {exc}")
        return {
            "test_subject": test_subject,
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
    # LEAKAGE GUARD: mean/std computed from X_train only, never from X_test.
    X_train, X_val, X_test = standardize_channels(X_train, X_val, X_test)

    # ── Convert to tensors — shape (B, T, C) for UVMDClassifier ──────────
    # UVMDClassifier.forward() expects (B, T, C) — no transpose needed.
    X_train_t = torch.FloatTensor(X_train)   # (N_tr, T, C)
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

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False)

    # ── Create model ──────────────────────────────────────────────────────
    model = UVMDClassifier(
        K           = cfg["num_modes"],
        L           = cfg["num_layers"],
        in_channels = in_channels,
        num_classes = num_classes,
        feat_dim    = cfg["feat_dim"],
        hidden_dim  = cfg["hidden_dim"],
        dropout     = cfg["dropout"],
        alpha_init  = cfg["alpha_init"],
        tau_init    = cfg["tau_init"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_uvmd   = sum(p.numel() for p in model.uvmd.parameters())
    logger.info(
        f"  Model: UVMDClassifier | total={n_params:,} params "
        f"(UVMD={n_uvmd}, CNN+head={n_params-n_uvmd})"
    )

    # Log initial UVMD centre frequencies
    init_omega = model.uvmd.omega.detach().cpu().numpy().tolist()
    logger.info(f"  Initial omega_k: {[f'{w:.3f}' for w in init_omega]}")

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    model, history = train_model(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        num_classes    = num_classes,
        y_train        = y_train,
        device         = device,
        epochs         = cfg["epochs"],
        lr             = cfg["lr"],
        weight_decay   = cfg["weight_decay"],
        patience       = cfg["patience"],
        grad_clip      = cfg["grad_clip"],
        overlap_lambda = cfg["overlap_lambda"],
        overlap_sigma  = cfg["overlap_sigma"],
        logger         = logger,
    )
    train_time = time.time() - t0
    logger.info(
        f"  Training: {train_time:.1f}s | "
        f"{history['total_epochs']} epochs (best={history['best_epoch']})"
    )

    # Log learned UVMD parameters after training
    learned = model.get_learned_uvmd_params()
    logger.info(f"  Learned omega_k: {[f'{w:.3f}' for w in learned['omega_k']]}")
    overlap_val = model.spectral_overlap_penalty(sigma=cfg["overlap_sigma"])
    logger.info(
        f"  Spectral overlap penalty (final): {overlap_val.item():.4f}"
    )

    # ── Evaluate (model.eval() — no test-time adaptation) ─────────────────
    class_names = [f"gesture_{g}" for g in common_gestures]
    test_results = evaluate_model(model, test_loader, device, class_names)

    test_acc = test_results["accuracy"]
    test_f1  = test_results["f1_macro"]

    logger.info(f"  Test: accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}")

    # ── Save fold outputs ─────────────────────────────────────────────────
    fold_result = {
        "test_subject":   test_subject,
        "train_subjects": train_subjects,
        "common_gestures": [int(g) for g in common_gestures],
        "num_classes":    num_classes,
        "model_params":   n_params,
        "training": {
            "epochs":        history["total_epochs"],
            "best_epoch":    history["best_epoch"],
            "train_time_s":  round(train_time, 1),
        },
        "test_metrics": {
            "accuracy":         test_acc,
            "f1_macro":         test_f1,
            "report":           test_results["report"],
            "confusion_matrix": test_results["confusion_matrix"],
        },
        "uvmd_analysis": {
            "init_omega_k":   init_omega,
            "final_omega_k":  learned["omega_k"],
            "final_alpha_lk": learned["alpha_lk"],
            "final_tau_l":    learned["tau_l"],
            "overlap_penalty_final": float(overlap_val.item()),
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
        "test_subject":  test_subject,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "train_time_s":  round(train_time, 1),
        "final_omega_k": learned["omega_k"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    LOSO evaluation loop for Unfolded VMD experiment.

    Subject list priority (safe server default = CI_TEST_SUBJECTS):
      1. --subjects DB2_s1,DB2_s12,...  — explicit list
      2. --full                         — all 20 DEFAULT_SUBJECTS
      3. --ci  (or no flag)             — 5 CI_TEST_SUBJECTS  ← vast.ai safe

    Architecture overrides:
      --num_modes   K    — number of VMD modes (default 4)
      --num_layers  L    — number of unrolled ADMM iterations (default 8)
      --alpha_init  A    — initial bandwidth constraint (default 2000)
      --overlap_lambda λ — weight of spectral overlap penalty (default 0.01)
    """
    import argparse

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects",        type=str,   default=None)
    _parser.add_argument("--ci",              action="store_true")
    _parser.add_argument("--full",            action="store_true")
    _parser.add_argument("--exercises",       type=str,   default=None)
    _parser.add_argument("--num_modes",       type=int,   default=NUM_MODES)
    _parser.add_argument("--num_layers",      type=int,   default=NUM_LAYERS)
    _parser.add_argument("--alpha_init",      type=float, default=ALPHA_INIT)
    _parser.add_argument("--overlap_lambda",  type=float, default=OVERLAP_LAMBDA)
    _args, _ = _parser.parse_known_args()

    # ── Subject list ──────────────────────────────────────────────────────
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
        "num_modes":      _args.num_modes,
        "num_layers":     _args.num_layers,
        "alpha_init":     _args.alpha_init,
        "tau_init":       TAU_INIT,
        "feat_dim":       FEAT_DIM,
        "hidden_dim":     HIDDEN_DIM,
        "dropout":        DROPOUT,
        "batch_size":     BATCH_SIZE,
        "epochs":         EPOCHS,
        "lr":             LR,
        "weight_decay":   WEIGHT_DECAY,
        "patience":       PATIENCE,
        "grad_clip":      GRAD_CLIP,
        "overlap_lambda": _args.overlap_lambda,
        "overlap_sigma":  OVERLAP_SIGMA,
        "val_ratio":      0.15,
        "seed":           42,
        "device":         "cuda" if torch.cuda.is_available() else "cpu",
    }

    BASE_DIR   = ROOT / "data"
    TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_ROOT)

    print("=" * 80)
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis  : Unfolded VMD (K={cfg['num_modes']} modes,\n"
        f"              L={cfg['num_layers']} unrolled iterations)\n"
        f"              All VMD params (alpha, tau, omega) learned via backprop.\n"
        f"              Loss = CE + {cfg['overlap_lambda']}×SpectralOverlapPenalty(ω)"
    )
    print(f"Subjects    : {ALL_SUBJECTS}")
    print(f"Exercises   : {exercises}")
    print(f"Device      : {cfg['device']}")
    print(f"Output      : {OUTPUT_ROOT}")
    print("=" * 80)

    # ── Processing config ─────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size     = 200,
        window_overlap  = 100,
        sampling_rate   = 2000,
        segment_edge_margin = 0.1,
    )
    split_cfg = SplitConfig(
        train_ratio   = 0.70,
        val_ratio     = 0.15,
        test_ratio    = 0.15,
        mode          = "by_segments",
        shuffle_segments = True,
        seed          = 42,
        include_rest_in_splits = False,
    )

    proc_cfg.save(OUTPUT_ROOT / "processing_config.json")
    with open(OUTPUT_ROOT / "split_config.json", "w") as fh:
        json.dump(asdict(split_cfg), fh, indent=4)
    with open(OUTPUT_ROOT / "uvmd_config.json", "w") as fh:
        json.dump(make_json_serializable(cfg), fh, indent=4)

    # ── Load all subjects ─────────────────────────────────────────────────
    logger.info("Loading all subjects...")
    multi_loader = MultiSubjectLoader(
        processing_config      = proc_cfg,
        logger                 = logger,
        use_gpu                = True,
        use_improved_processing= USE_IMPROVED,
    )

    # subjects_data: {subj_id: (emg, segments, grouped_windows)}
    # NOTE: values are TUPLES not dicts — unpack as (emg, segs, grouped)
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir    = BASE_DIR,
        subject_ids = ALL_SUBJECTS,
        exercises   = exercises,
        include_rest= split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES,
    )
    logger.info(
        f"Common gestures: {common_gestures} ({len(common_gestures)} total)"
    )

    # ── LOSO loop ─────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    t_loso_start = time.time()

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
        except Exception as exc:
            logger.error(f"Fold {test_subject} failed: {exc}")
            traceback.print_exc()
            result = {
                "test_subject":  test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(exc),
            }

        all_results.append(result)

    t_loso_total = time.time() - t_loso_start

    # ── Aggregate LOSO summary ────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]

    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"] for r in valid]
        times = [r.get("train_time_s", 0.0) for r in valid]

        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        mean_f1  = float(np.mean(f1s))
        std_f1   = float(np.std(f1s))
        mean_t   = float(np.mean(times))

        # Aggregate learned omega across folds
        omega_per_fold = [r["final_omega_k"] for r in valid if "final_omega_k" in r]
        if omega_per_fold:
            omega_arr = np.array(omega_per_fold)          # (n_folds, K)
            mean_omega = omega_arr.mean(axis=0).tolist()
            std_omega  = omega_arr.std(axis=0).tolist()
        else:
            mean_omega, std_omega = [], []

        # Check against success criterion
        criterion_met = mean_f1 >= 0.35 and mean_t < 300.0   # < 5 min/fold

        print("\n" + "=" * 80)
        print(
            f"LOSO SUMMARY — Unfolded VMD "
            f"(K={cfg['num_modes']}, L={cfg['num_layers']})"
        )
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
        if mean_omega:
            print(
                f"  Mean ω_k  : {[f'{w:.3f}' for w in mean_omega]}"
                f"  (± {[f'{s:.3f}' for s in std_omega]})"
            )
        print(
            f"  Success   : {'YES' if criterion_met else 'NO'} "
            f"(F1≥0.35: {mean_f1>=0.35}, <5min/fold: {mean_t<300})"
        )
        print("=" * 80)

        summary = {
            "experiment":   EXPERIMENT_NAME,
            "model":        "UVMDClassifier",
            "subjects":     ALL_SUBJECTS,
            "exercises":    exercises,
            "uvmd_config": {
                "num_modes":      cfg["num_modes"],
                "num_layers":     cfg["num_layers"],
                "alpha_init":     cfg["alpha_init"],
                "tau_init":       cfg["tau_init"],
                "overlap_lambda": cfg["overlap_lambda"],
                "overlap_sigma":  cfg["overlap_sigma"],
            },
            "loso_metrics": {
                "mean_accuracy":  mean_acc,
                "std_accuracy":   std_acc,
                "mean_f1_macro":  mean_f1,
                "std_f1_macro":   std_f1,
                "per_subject":    all_results,
            },
            "timing": {
                "mean_fold_time_s": mean_t,
                "total_loso_time_s": round(t_loso_total, 1),
            },
            "uvmd_convergence": {
                "mean_omega_k_across_folds": mean_omega,
                "std_omega_k_across_folds":  std_omega,
                "interpretation": (
                    "omega_k shows which frequency bands the model learned to focus on. "
                    "Well-separated values indicate distinct frequency coverage. "
                    "Stable values across folds indicate a universal (subject-independent) "
                    "decomposition was found."
                ),
            },
            "success_criterion": {
                "f1_ge_035":   bool(mean_f1 >= 0.35),
                "lt_5min_fold": bool(mean_t < 300.0),
                "met":          bool(criterion_met),
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
                        "mean_accuracy":  mean_acc,
                        "std_accuracy":   std_acc,
                        "mean_f1_macro":  mean_f1,
                        "std_f1_macro":   std_f1,
                        "mean_fold_time_s": mean_t,
                        "n_folds":        len(valid),
                    },
                    experiment_name=EXPERIMENT_NAME,
                )
            else:
                mark_hypothesis_failed(
                    hypothesis_id,
                    error_message="All LOSO folds failed — no valid results.",
                )
    except ImportError:
        pass   # hypothesis_executor not installed — silently skip


if __name__ == "__main__":
    main()
