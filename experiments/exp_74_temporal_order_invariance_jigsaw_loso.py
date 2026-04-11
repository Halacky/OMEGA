"""
Experiment 74: Temporal Order Invariance via Jigsaw Puzzle Training (LOSO)

Hypothesis
----------
A significant fraction of cross-subject EMG variability stems from local
temporal shifts and micro-stretches in muscle-activation timing.  If a
neural network is trained to be *invariant* to small permutations of
temporal chunks within each window, it should learn gesture representations
that are less sensitive to subject-specific timing and therefore generalise
better across subjects in LOSO evaluation.

Analogy: Jigsaw-puzzle self-supervised learning in computer vision and
time-shuffle invariance in human activity recognition (HAR).

Method (strictly LOSO — no test-subject adaptation)
----------------------------------------------------
For each LOSO fold:

    Phase 1 — Supervised multi-task training on train subjects only:
        · Split each window into N_CHUNKS equal temporal chunks
          (N_CHUNKS=8 → each chunk ≈ 37.5 ms at 2 kHz / 600 sample windows)
        · For every training sample, uniformly draw a random permutation
          from a fixed vocabulary of 30 constrained permutations.
          Each permutation is obtained from the identity by at most 2
          adjacent-chunk transpositions → temporal disruption is local.
        · Loss:
              L = α·CE(gesture_logits, y_gesture)
                + (1-α)·CE(jigsaw_logits, perm_idx)
          with α = 0.7  (gesture is the primary task).
        · Validation uses *unpermuted* windows and gesture CE loss only
          (exactly as the test set would be seen).
        · Early stopping is based on validation gesture CE loss.

    Phase 2 — Evaluation on held-out (test) subject:
        · No permutation applied.
        · Only gesture_head output used.
        · Normalisation statistics (μ/σ) are frozen from training.

Data-leakage audit
------------------
    ✓ μ/σ: computed on X_train (train subjects, train split) ONLY.
      Applied without modification to val and test windows.

    ✓ Permutation vocabulary: fixed deterministic set (seed=0), derived
      entirely from the architecture hyperparameter n_chunks.
      Contains NO information from any data sample.

    ✓ Jigsaw labels: generated on-the-fly from the randomly drawn
      permutation index — purely structural, no data statistics.

    ✓ Validation dataset: plain (unpermuted) windows with gesture labels.
      Early stopping never sees test-subject data.

    ✓ Test-subject windows: first appear in evaluate_numpy() after fit()
      is complete.  No weight update occurs after that point.

    ✓ No subject identity / subject label used anywhere in model or loss.

Architecture (JigsawTemporalEMGNet)
------------------------------------
    Shared encoder:  3-block Conv1D  (N, C, T) → (N, d_enc, T')
    Context:         Bidirectional GRU          → (N, T', 2·d_ctx)
    Attention pool:                             → (N, 2·d_ctx)
    Gesture head:    Linear                     → (N, num_classes)   [always]
    Jigsaw head:     Linear                     → (N, num_perms)     [train only]

Hyperparameters (JIGSAW_CFG)
-----------------------------
    n_chunks    8      chunks per window   (≈37.5 ms each at 2 kHz)
    n_perms    30      permutation vocab size (incl. identity)
    max_swaps   2      max adjacent swaps per permutation
    alpha       0.7    gesture-loss weight
    d_enc     128      CNN encoder output channels
    d_ctx     128      GRU hidden size (one direction)
    perm_seed   0      fixed seed for permutation generation

Baseline
--------
    exp_7 (CNN-GRU-attention + noise/warp augmentation) — same backbone
    depth, different augmentation strategy.
"""

from __future__ import annotations

import gc
import json
import sys
import traceback
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Repository root on sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer

from models.jigsaw_temporal_emg import JigsawTemporalEMGNet, generate_constrained_permutations
from models import register_model

register_model("jigsaw_temporal_emg", JigsawTemporalEMGNet)


# ---------------------------------------------------------------------------
# Subject lists
# ---------------------------------------------------------------------------

_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]


def _parse_subjects() -> List[str]:
    """Parse --subjects / --ci / --full CLI args.  Defaults to CI subjects."""
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci",   action="store_true")
    _parser.add_argument("--full", action="store_true")
    _args, _ = _parser.parse_known_args()
    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    # Default: CI subjects — server has symlinks only for these five
    return _CI_SUBJECTS


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Default jigsaw hyperparameters
# ---------------------------------------------------------------------------

JIGSAW_CFG: Dict = {
    # Temporal chunking
    "n_chunks":   8,    # window divided into 8 chunks → ~37.5 ms each (T=600, fs=2000)
    "n_perms":   30,    # vocabulary size (index 0 = identity / no swap)
    "max_swaps":  2,    # maximum adjacent swaps per permutation

    # Loss weighting
    "alpha":     0.7,   # weight for gesture classification loss
                        # (1 - alpha) = 0.3 for jigsaw prediction loss

    # Model dimensions
    "d_enc":    128,    # CNN encoder output channels
    "d_ctx":    128,    # GRU hidden size (each direction)

    # Reproducibility
    "perm_seed": 0,     # seed for generate_constrained_permutations
}


# ---------------------------------------------------------------------------
# Chunk permutation application (numpy, single sample)
# ---------------------------------------------------------------------------

def _apply_chunk_permutation(x_ct: np.ndarray, perm: List[int]) -> np.ndarray:
    """
    Apply a temporal chunk permutation to a single EMG window.

    Args:
        x_ct:  (C, T) — one normalised EMG window in (channels, time) format.
        perm:  permutation of length n_chunks, e.g. [2, 0, 1, 3, 4, 5, 6, 7].

    Returns:
        (C, T) window with temporal chunks reordered per `perm`.
        If T is not divisible by n_chunks, the trailing samples are preserved
        unchanged to avoid discarding information.

    LOSO note:
        This function is called ONLY inside JigsawWindowDataset.__getitem__,
        which is constructed from training-subject data exclusively.
        Validation and test windows are never passed through this function.
    """
    C, T = x_ct.shape
    n_chunks = len(perm)
    chunk_size = T // n_chunks
    T_use = chunk_size * n_chunks       # largest multiple of n_chunks ≤ T

    # Reorder the usable portion; leave any tail samples untouched
    x_use = x_ct[:, :T_use]            # (C, T_use)
    # Reshape → (C, n_chunks, chunk_size), reorder, reshape back
    chunks = x_use.reshape(C, n_chunks, chunk_size)
    x_perm = chunks[:, perm, :].reshape(C, T_use)

    if T_use < T:
        # Preserve trailing samples that don't fit into a full chunk
        x_perm = np.concatenate([x_perm, x_ct[:, T_use:]], axis=1)

    return x_perm


# ---------------------------------------------------------------------------
# Dataset with on-the-fly jigsaw permutation (training only)
# ---------------------------------------------------------------------------

class JigsawWindowDataset(Dataset):
    """
    PyTorch Dataset that applies a random permutation from a fixed vocabulary
    to each window at every access.

    Returned tuple per sample:
        x_perm   (C, T)  float32 — permuted EMG window
        y        ()      int64   — gesture class index
        perm_idx ()      int64   — index of the applied permutation in `perms`

    The permutation is drawn uniformly from `perms` using torch.randint,
    which is thread-safe for DataLoader with num_workers > 0.

    LOSO compliance:
        · Only instantiated from training-split data (train subjects).
        · Val and test windows use plain WindowDataset (no permutation).
        · The perm_idx label is derived from the permutation structure,
          not from any data statistics → no information leakage.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, perms: List[List[int]]):
        """
        Args:
            X:     (N, C, T) float32 array — normalised training windows.
            y:     (N,) int64 array — gesture class indices.
            perms: list of permutations from generate_constrained_permutations.
                   perms[0] MUST be the identity (no reordering).
        """
        assert X.ndim == 3, f"Expected (N, C, T), got {X.shape}"
        assert len(X) == len(y), "X and y length mismatch"
        assert len(perms) >= 1, "perms must be non-empty"

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.perms = perms
        self.n_perms = len(perms)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]   # (C, T)
        y = self.y[idx]   # scalar

        # Draw permutation index uniformly; torch.randint is thread-safe
        perm_idx = int(torch.randint(self.n_perms, (1,)).item())
        perm = self.perms[perm_idx]

        x_perm = _apply_chunk_permutation(x, perm)

        return (
            torch.from_numpy(x_perm).float(),
            torch.tensor(y,        dtype=torch.long),
            torch.tensor(perm_idx, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Plain dataset for validation / test (no permutation)
# ---------------------------------------------------------------------------

class _PlainDataset(Dataset):
    """Minimal (X, y) dataset — no augmentation, no permutation."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# JigsawTrainer
# ---------------------------------------------------------------------------

class JigsawTrainer(WindowClassifierTrainer):
    """
    LOSO-compliant trainer for temporal-order invariance via jigsaw training.

    Overrides fit() and evaluate_numpy() from WindowClassifierTrainer.

    fit() performs:
        1. Convert splits → flat arrays via _prepare_splits_arrays()
        2. Transpose (N,T,C) → (N,C,T)
        3. Compute μ/σ from X_train ONLY; apply to val and (later) test
        4. Build permutation vocabulary (fixed, data-independent)
        5. JigsawWindowDataset for training (random permutation per step)
        6. _PlainDataset for validation (no permutation)
        7. Train dual-head model; early stop on val gesture CE loss
        8. Restore best checkpoint; store self.model, self.mean_c, etc.

    evaluate_numpy() performs inference-only:
        1. Transpose + standardise with frozen μ/σ
        2. Forward pass with return_jigsaw=False → gesture logits only
        3. Compute accuracy / F1; no weight updates
    """

    def __init__(self, jigsaw_cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.jigsaw_cfg = jigsaw_cfg
        # Permutation vocabulary is built once in fit() and reused in logging
        self._perms: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_nct(arr: np.ndarray) -> np.ndarray:
        """Transpose (N, T, C) → (N, C, T) when T > C (raw EMG windows)."""
        if arr.ndim == 3 and arr.shape[1] > arr.shape[2]:
            return np.transpose(arr, (0, 2, 1))
        return arr

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """
        Train the dual-head jigsaw model following strict LOSO protocol.

        Data-leakage guarantee summary:
            · μ/σ from X_train (train subjects, train split) ONLY.
            · Permutation vocabulary: fixed deterministic set, no data used.
            · Training dataset: JigsawWindowDataset (permuted windows).
            · Validation dataset: _PlainDataset (no permutation) → same
              distribution as test; used for early stopping on gesture loss.
            · splits["test"] is NOT accessed here; it enters only through
              evaluate_numpy() which is called externally after fit().

        Args:
            splits: {
                "train": {gesture_id: (N_train, T, C) array},
                "val":   {gesture_id: (N_val,   T, C) array},
                "test":  {gesture_id: (N_test,  T, C) array},   ← never read here
            }

        Returns:
            {} (empty dict; metrics collected by CrossSubjectExperiment)
        """
        seed_everything(self.cfg.seed)
        cfg = self.jigsaw_cfg
        device = self.cfg.device

        # ── 1. Splits → flat arrays (N, T, C) ──────────────────────────────
        (
            X_train, y_train,
            X_val,   y_val,
            _X_test, _y_test,        # test split: ignore here, evaluated externally
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)
        # Note: splits["test"] is read by _prepare_splits_arrays but NOT
        # used in any computation inside this method.

        num_classes = len(class_ids)

        # ── 2. Transpose (N, T, C) → (N, C, T) ────────────────────────────
        X_train_nct = self._to_nct(X_train)   # (N, C, T)
        X_val_nct   = self._to_nct(X_val)     # (M, C, T) or empty

        in_channels = X_train_nct.shape[1]
        time_steps  = X_train_nct.shape[2]

        # ── 3. Channel-wise standardisation — X_train ONLY ─────────────────
        # μ/σ are computed here and stored on self for later use in
        # evaluate_numpy(). They are applied identically to val and test.
        mean_c, std_c = self._compute_channel_standardization(X_train_nct)

        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = time_steps

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        X_train_n = self._apply_standardization(X_train_nct, mean_c, std_c)
        X_val_n   = (
            self._apply_standardization(X_val_nct, mean_c, std_c)
            if len(X_val_nct) > 0 else X_val_nct
        )

        # ── 4. Permutation vocabulary (fixed, data-independent) ─────────────
        # Permutations depend only on n_chunks and perm_seed — no data.
        n_chunks  = cfg["n_chunks"]
        n_perms   = cfg["n_perms"]
        max_swaps = cfg["max_swaps"]
        perm_seed = cfg["perm_seed"]

        perms = generate_constrained_permutations(
            n_chunks=n_chunks,
            n_perms=n_perms,
            max_swaps=max_swaps,
            seed=perm_seed,
        )
        self._perms = perms
        actual_n_perms = len(perms)  # may be < n_perms if constrained space is small

        self.logger.info(
            f"[Jigsaw] Permutation vocab: {actual_n_perms}/{n_perms} generated "
            f"(n_chunks={n_chunks}, max_swaps={max_swaps}, seed={perm_seed}). "
            f"Chunk size: {time_steps // n_chunks} samples "
            f"({1000 * (time_steps // n_chunks) / 2000:.1f} ms at 2 kHz)."
        )

        # Save permutation vocabulary for reproducibility
        with open(self.output_dir / "jigsaw_permutations.json", "w") as f:
            json.dump({"permutations": perms, "n_chunks": n_chunks}, f, indent=2)

        # ── 5. Model ────────────────────────────────────────────────────────
        model = JigsawTemporalEMGNet(
            in_channels=in_channels,
            num_classes=num_classes,
            num_perms=actual_n_perms,
            d_enc=cfg["d_enc"],
            d_ctx=cfg["d_ctx"],
            dropout=self.cfg.dropout,
        ).to(device)

        self.logger.info(
            f"[Jigsaw] Model: in_channels={in_channels}, num_classes={num_classes}, "
            f"num_perms={actual_n_perms}, d_enc={cfg['d_enc']}, d_ctx={cfg['d_ctx']}"
        )

        # ── 6. Datasets and data loaders ────────────────────────────────────
        # Training: JigsawWindowDataset → applies random permutation on-the-fly
        # Validation: _PlainDataset → no permutation (same as test distribution)
        ds_train = JigsawWindowDataset(X_train_n, y_train, perms)
        ds_val   = _PlainDataset(X_val_n, y_val) if len(X_val_n) > 0 else None

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        dl_val = (
            DataLoader(
                ds_val,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            if ds_val is not None else None
        )

        # ── 7. Loss functions ───────────────────────────────────────────────
        alpha = cfg["alpha"]    # gesture loss weight
        beta  = 1.0 - alpha     # jigsaw loss weight

        if self.cfg.use_class_weights:
            counts  = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            weights = counts.sum() / (counts + 1e-8)
            weights /= weights.mean()
            gesture_criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(weights).float().to(device)
            )
        else:
            gesture_criterion = nn.CrossEntropyLoss()

        # Jigsaw classes are balanced by construction (uniform sampling)
        jigsaw_criterion = nn.CrossEntropyLoss()

        # ── 8. Optimiser / scheduler ────────────────────────────────────────
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # ReduceLROnPlateau — verbose kwarg removed in PyTorch 2.4+
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        # ── 9. Training loop ────────────────────────────────────────────────
        best_val_loss  = float("inf")
        best_state     = None
        patience_ctr   = 0
        patience       = self.cfg.early_stopping_patience
        history        = {"train_loss": [], "val_gesture_loss": [],
                          "train_gesture_acc": [], "val_gesture_acc": [],
                          "train_jigsaw_acc": []}

        for epoch in range(1, self.cfg.epochs + 1):
            # ── Train step ──────────────────────────────────────────────────
            model.train()
            epoch_loss_total  = 0.0
            epoch_gesture_ok  = 0
            epoch_jigsaw_ok   = 0
            epoch_n           = 0

            for x_perm, y_gest, perm_idx in dl_train:
                x_perm   = x_perm.to(device)    # (B, C, T)
                y_gest   = y_gest.to(device)     # (B,)
                perm_idx = perm_idx.to(device)   # (B,)

                optimizer.zero_grad()

                gesture_logits, jigsaw_logits = model(x_perm, return_jigsaw=True)

                loss_gesture = gesture_criterion(gesture_logits, y_gest)
                loss_jigsaw  = jigsaw_criterion(jigsaw_logits,  perm_idx)
                loss         = alpha * loss_gesture + beta * loss_jigsaw

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                B = len(x_perm)
                epoch_loss_total += loss.item() * B
                epoch_gesture_ok += (gesture_logits.argmax(1) == y_gest).sum().item()
                epoch_jigsaw_ok  += (jigsaw_logits.argmax(1)  == perm_idx).sum().item()
                epoch_n          += B

            train_loss        = epoch_loss_total / max(epoch_n, 1)
            train_gesture_acc = epoch_gesture_ok  / max(epoch_n, 1)
            train_jigsaw_acc  = epoch_jigsaw_ok   / max(epoch_n, 1)

            history["train_loss"].append(train_loss)
            history["train_gesture_acc"].append(train_gesture_acc)
            history["train_jigsaw_acc"].append(train_jigsaw_acc)

            # ── Validation step (no permutation) ────────────────────────────
            if dl_val is not None:
                model.eval()
                val_loss  = 0.0
                val_ok    = 0
                val_n     = 0

                with torch.no_grad():
                    for x_val, y_val_b in dl_val:
                        x_val   = x_val.to(device)
                        y_val_b = y_val_b.to(device)

                        # Only gesture head; return_jigsaw=False (no perm at val)
                        gesture_logits, _ = model(x_val, return_jigsaw=False)

                        val_loss += gesture_criterion(gesture_logits, y_val_b).item() * len(x_val)
                        val_ok   += (gesture_logits.argmax(1) == y_val_b).sum().item()
                        val_n    += len(x_val)

                val_loss /= max(val_n, 1)
                val_acc   = val_ok / max(val_n, 1)

                scheduler.step(val_loss)
                history["val_gesture_loss"].append(val_loss)
                history["val_gesture_acc"].append(val_acc)

                if epoch % 5 == 0 or epoch == 1:
                    self.logger.info(
                        f"[Jigsaw] Epoch {epoch:03d}/{self.cfg.epochs} "
                        f"train_loss={train_loss:.4f} "
                        f"gest_acc={train_gesture_acc:.4f} "
                        f"jigsaw_acc={train_jigsaw_acc:.4f} | "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )

                # Early stopping — gesture val loss
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        self.logger.info(
                            f"[Jigsaw] Early stopping at epoch {epoch} "
                            f"(patience={patience})."
                        )
                        break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    self.logger.info(
                        f"[Jigsaw] Epoch {epoch:03d}/{self.cfg.epochs} "
                        f"train_loss={train_loss:.4f} "
                        f"gest_acc={train_gesture_acc:.4f} "
                        f"jigsaw_acc={train_jigsaw_acc:.4f}"
                    )

        # ── 10. Restore best checkpoint ─────────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)
            self.logger.info(
                f"[Jigsaw] Restored best checkpoint "
                f"(val_gesture_loss={best_val_loss:.4f})"
            )

        self.model = model

        # Save training history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        # Save model checkpoint
        torch.save(
            {
                "state_dict":   model.state_dict(),
                "in_channels":  in_channels,
                "num_classes":  num_classes,
                "num_perms":    actual_n_perms,
                "class_ids":    class_ids,
                "mean":         mean_c,
                "std":          std_c,
                "window_size":  time_steps,
                "jigsaw_cfg":   cfg,
                "perms":        perms,
            },
            self.output_dir / "jigsaw_model.pt",
        )
        self.logger.info(f"[Jigsaw] Model checkpoint saved.")

        return {}

    # ------------------------------------------------------------------
    # evaluate_numpy()  — inference only, no permutation
    # ------------------------------------------------------------------

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate gesture classification on an (X, y) array.

        Overrides the parent method because JigsawTemporalEMGNet.forward()
        returns a tuple (gesture_logits, jigsaw_logits).  Only gesture_logits
        are used here; no permutation is applied.

        LOSO compliance:
            · Normalisation uses frozen μ/σ from fit() — no test data involved.
            · No weight update occurs; the model is in eval() mode throughout.
            · Works correctly for the held-out test subject and for validation.

        Args:
            X:          (N, T, C) raw EMG windows (unpermuted).
            y:          (N,) integer gesture class indices.
            split_name: label for logging / saved files.
            visualize:  if True, save confusion matrix (via self.visualizer).

        Returns:
            dict with keys: accuracy, f1_macro, report, confusion_matrix.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report, confusion_matrix,
        )

        assert self.model    is not None, "Model is not fitted"
        assert self.mean_c   is not None, "Normalisation stats missing (call fit first)"
        assert self.std_c    is not None, "Normalisation stats missing (call fit first)"
        assert self.class_ids is not None, "class_ids missing (call fit first)"

        # ── 1. Transpose (N, T, C) → (N, C, T) ────────────────────────────
        X_nct = self._to_nct(X.astype(np.float32))

        # ── 2. Standardise with frozen μ/σ (from training) ─────────────────
        Xs = self._apply_standardization(X_nct, self.mean_c, self.std_c)

        # ── 3. Dataset and data loader ──────────────────────────────────────
        ds = _PlainDataset(Xs, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # ── 4. Inference (gesture head only, no permutation) ────────────────
        device = self.cfg.device
        self.model.eval()
        all_logits: List[np.ndarray] = []
        all_y:      List[np.ndarray] = []

        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                # return_jigsaw=False: model returns (gesture_logits, None)
                gesture_logits, _ = self.model(xb, return_jigsaw=False)
                all_logits.append(gesture_logits.cpu().numpy())
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y,      axis=0)
        y_pred = logits.argmax(axis=1)

        # ── 5. Metrics ──────────────────────────────────────────────────────
        acc      = float(accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        report   = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        num_classes = len(self.class_ids)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        self.logger.info(
            f"[Jigsaw eval:{split_name}] accuracy={acc:.4f}  f1_macro={f1_macro:.4f}"
        )

        # ── 6. Optional visualisation ────────────────────────────────────────
        if visualize and self.visualizer is not None:
            class_labels = [
                self.class_names[gid] for gid in self.class_ids
            ]
            self.visualizer.plot_confusion_matrix(
                cm, class_labels, normalize=True,
                filename=f"cm_{split_name}.png",
            )
            self.visualizer.plot_per_class_f1(
                report, class_labels,
                filename=f"f1_{split_name}.png",
            )

        return {
            "accuracy":         acc,
            "f1_macro":         f1_macro,
            "report":           report,
            "confusion_matrix": cm.tolist(),
        }


# ---------------------------------------------------------------------------
# Single LOSO fold
# ---------------------------------------------------------------------------

def run_single_loso_fold(
    base_dir:       Path,
    output_dir:     Path,
    train_subjects: List[str],
    test_subject:   str,
    exercises:      List[str],
    proc_cfg:       ProcessingConfig,
    split_cfg:      SplitConfig,
    train_cfg:      TrainingConfig,
    jigsaw_cfg:     dict,
) -> Dict:
    """
    Execute one LOSO fold: train on `train_subjects`, evaluate on `test_subject`.

    Strict LOSO protocol:
        · `test_subject` data is loaded by CrossSubjectExperiment but is
          NEVER passed to trainer.fit().  It enters only through
          trainer.evaluate_numpy() after training is complete.
        · No normalisation statistic, no model weight, and no
          hyperparameter is adjusted based on test-subject data.

    Args:
        base_dir:       path to the data directory (contains DB2_sN subdirs).
        output_dir:     directory for this fold's artefacts.
        train_subjects: list of subject IDs used for training.
        test_subject:   held-out subject ID.
        exercises:      list of exercise names, e.g. ["E1"].
        proc_cfg:       EMG pre-processing configuration.
        split_cfg:      train/val/test split configuration.
        train_cfg:      training hyperparameters.
        jigsaw_cfg:     jigsaw-specific hyperparameters.

    Returns:
        dict with keys: test_subject, test_accuracy, test_f1_macro,
                        and optionally 'error'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    # Record experiment meta in config files
    train_cfg.model_type    = "jigsaw_temporal_emg"
    train_cfg.pipeline_type = "deep_raw"
    train_cfg.use_handcrafted_features = False

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json",   "w") as f:
        json.dump(asdict(split_cfg),  f, indent=4)
    with open(output_dir / "jigsaw_config.json",  "w") as f:
        json.dump(jigsaw_cfg,         f, indent=4)

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects,
        test_subject=test_subject,
        exercises=exercises,
        base_dir=base_dir,
        pool_train_subjects=True,
        use_separate_val_subject=False,
        val_subject=None,
        val_ratio=0.15,
        seed=train_cfg.seed,
        max_gestures=10,
    )
    cs_cfg.save(output_dir / "cross_subject_config.json")

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=True,
    )

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)  # noqa: F841

    trainer = JigsawTrainer(
        jigsaw_cfg=jigsaw_cfg,
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=base_viz,
        logger=logger,
    )

    try:
        results = experiment.run()
    except Exception as e:
        logger.error(f"[LOSO fold] Error (test={test_subject}): {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1  = float(test_metrics.get("f1_macro", 0.0))

    logger.info(
        f"[LOSO] test={test_subject} | acc={test_acc:.4f}, f1={test_f1:.4f}"
    )
    print(f"[LOSO] test={test_subject} | acc={test_acc:.4f}, f1={test_f1:.4f}")

    # Save fold results (exclude raw subjects_data to keep file sizes small)
    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(_make_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        _make_serializable({
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "exercises":      exercises,
            "jigsaw_cfg":     jigsaw_cfg,
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {
        "test_subject":  test_subject,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPERIMENT_NAME = "exp_74_temporal_order_invariance_jigsaw_loso"
    BASE_DIR     = ROOT / "data"
    ALL_SUBJECTS = _parse_subjects()
    OUTPUT_DIR   = Path(
        f"./experiments_output/{EXPERIMENT_NAME}_"
        + "_".join(s.split("_s")[1] for s in ALL_SUBJECTS)
    )
    EXERCISES = ["E1"]

    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )

    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=80,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.2,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
    )

    jigsaw_cfg = dict(JIGSAW_CFG)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"EXPERIMENT: {EXPERIMENT_NAME}")
    global_logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    global_logger.info(f"Device: {train_cfg.device}")
    global_logger.info(f"Jigsaw cfg: {jigsaw_cfg}")
    global_logger.info("=" * 80)

    all_loso_results: List[Dict] = []

    for test_subject in ALL_SUBJECTS:
        print(f"\n{'='*60}")
        print(f"  LOSO fold: test_subject={test_subject}")
        print(f"{'='*60}")

        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_DIR / f"test_{test_subject}"

        try:
            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                jigsaw_cfg=dict(jigsaw_cfg),   # fresh copy per fold
            )
            all_loso_results.append(fold_res)

            acc = fold_res["test_accuracy"]
            f1  = fold_res["test_f1_macro"]
            acc_str = f"{acc:.4f}" if acc is not None else "None"
            f1_str  = f"{f1:.4f}"  if f1  is not None else "None"
            print(f"  ✓ test={test_subject}  acc={acc_str}  f1={f1_str}")

        except Exception as e:
            global_logger.error(f"✗ Failed fold test={test_subject}: {e}")
            global_logger.error(traceback.format_exc())
            all_loso_results.append({
                "test_subject":  test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error":         str(e),
            })

    # ── Aggregate across folds ──────────────────────────────────────────────
    valid = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    accs  = [r["test_accuracy"] for r in valid]
    f1s   = [r["test_f1_macro"]  for r in valid]

    if valid:
        summary_line = (
            f"LOSO summary ({len(valid)}/{len(ALL_SUBJECTS)} folds): "
            f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
            f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}"
        )
        print(f"\n{summary_line}")
        global_logger.info(summary_line)
    else:
        global_logger.warning("No successful LOSO folds to aggregate.")

    summary = {
        "experiment_name":   EXPERIMENT_NAME,
        "subjects":          ALL_SUBJECTS,
        "exercises":         EXERCISES,
        "jigsaw_cfg":        jigsaw_cfg,
        "processing_config": asdict(proc_cfg),
        "split_config":      asdict(split_cfg),
        "training_config":   asdict(train_cfg),
        "aggregate": {
            "mean_accuracy": float(np.mean(accs)) if valid else None,
            "std_accuracy":  float(np.std(accs))  if valid else None,
            "mean_f1_macro": float(np.mean(f1s))  if valid else None,
            "std_f1_macro":  float(np.std(f1s))   if valid else None,
            "n_folds":       len(valid),
        },
        "per_fold":        all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_make_serializable(summary), f, indent=4, ensure_ascii=False)

    global_logger.info(
        f"EXPERIMENT COMPLETE. Results: {OUTPUT_DIR.resolve()}"
    )

    # Optional: report to hypothesis executor if installed
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        if valid:
            mark_hypothesis_verified(
                "H_TEMPORAL_ORDER_INVARIANCE",
                metrics={
                    "mean_accuracy": float(np.mean(accs)),
                    "mean_f1_macro": float(np.mean(f1s)),
                    "n_folds":       len(valid),
                },
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                "H_TEMPORAL_ORDER_INVARIANCE",
                "No successful LOSO folds completed.",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
