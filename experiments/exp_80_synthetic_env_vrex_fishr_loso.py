# FILE: experiments/exp_80_synthetic_env_vrex_fishr_loso.py
"""
Experiment 80: V-REx / Fishr on Synthetic Pseudo-Environments for Invariant EMG

Hypothesis:
  V-REx and Fishr IRM variants under-perform (exp_69) partly because the number
  of training environments equals the number of training subjects — as few as 4
  in CI mode.  Generating E pseudo-environments through fixed, deterministic
  signal-space transformations (amplitude scaling, near-orthogonal channel
  mixing, FFT-based band-stop, mild time-compression) provides richer gradient
  diversity for the invariance penalty without any additional data or access to
  the test subject.

Two environment-building modes:
  "transforms_only"   — E environments, one per transform, each applied to a
                         random sub-sample of pooled normalised train data.
                         |envs| = E (default 7).
  "subject_transform" — each (train-subject, transform) pair is one environment.
                         |envs| = N_train_subjects × E.

LOSO Compliance (critical):
  1. All pseudo-environments are constructed from TRAIN-subject windows only.
  2. Transforms are fixed/deterministic (seeded at construction time).
  3. Test-subject data is loaded separately and is NEVER used to:
       - build environments
       - compute invariance penalty
       - set normalisation statistics (mean_c / std_c)
       - influence model selection (early stopping uses val from train subjects)
  4. Final evaluation is a single, gradient-free forward pass on the held-out
     test subject — no model update of any kind takes place after training.

Comparison targets:
  exp_69 — V-REx / Fishr with subject-based environments (same backbone)
"""

import gc
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
    parse_subjects_args,
)
from models import register_model
from models.irm_content_style_emg import IRMContentStyleEMG
from training.trainer import WindowClassifierTrainer
from utils.artifacts import ArtifactSaver
from utils.logging import seed_everything, setup_logging
from visualization.base import Visualizer

# Register backbone so that _create_model("irm_content_style_emg") resolves it.
register_model("irm_content_style_emg", IRMContentStyleEMG)


# ---------------------------------------------------------------------------
# grouped_to_arrays — local copy (processing.window_extraction does NOT exist)
# ---------------------------------------------------------------------------
def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grouped_windows {gesture_id: [rep_array, ...]} to flat arrays."""
    windows_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for gesture_id in sorted(grouped_windows.keys()):
        for rep_array in grouped_windows[gesture_id]:
            windows_list.append(rep_array)
            labels_list.append(
                np.full(len(rep_array), gesture_id, dtype=np.int64)
            )
    if not windows_list:
        raise RuntimeError("grouped_windows is empty — no data to concatenate")
    return np.concatenate(windows_list, axis=0), np.concatenate(labels_list, axis=0)


# ---------------------------------------------------------------------------
# SyntheticEMGTransforms
# ---------------------------------------------------------------------------
class SyntheticEMGTransforms:
    """
    Collection of fixed, deterministic signal transforms for building
    pseudo-environments.  All transforms operate on (N, C, T) float32 arrays
    that are already channel-normalised.

    LOSO safety:
      - Rotation matrices are generated once at construction time from fixed
        seeds that do NOT depend on any EMG data (train or test).
      - Every transform is a pure function of its input — no data-adaptive
        parameters are fitted.
      - These transforms are ONLY applied to train-subject windows; the test
        subject never enters this code path.
    """

    def __init__(self, num_channels: int, seed: int = 42):
        self.num_channels = num_channels
        # Pre-compute two orthogonal rotation matrices (different seeds).
        self._rot_1 = self._make_rotation_matrix(num_channels, seed=seed + 1)
        self._rot_2 = self._make_rotation_matrix(num_channels, seed=seed + 2)

    @staticmethod
    def _make_rotation_matrix(n: int, seed: int) -> np.ndarray:
        """
        Return an n×n proper orthogonal matrix via QR of a random normal.
        Fixed seed guarantees cross-fold and cross-run reproducibility.
        """
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n)).astype(np.float64)
        Q, _ = np.linalg.qr(A)
        # Ensure det = +1 (proper rotation, no reflection).
        if np.linalg.det(Q) < 0.0:
            Q[:, 0] = -Q[:, 0]
        return Q.astype(np.float32)

    def identity(self, X: np.ndarray) -> np.ndarray:
        """No transform — baseline environment (same as pooled train data)."""
        return X.copy()

    def amplitude_scale_down(self, X: np.ndarray) -> np.ndarray:
        """Simulate increased electrode impedance: multiply amplitude by 0.75."""
        return (X * np.float32(0.75)).astype(np.float32)

    def amplitude_scale_up(self, X: np.ndarray) -> np.ndarray:
        """Simulate decreased electrode impedance: multiply amplitude by 1.33."""
        return (X * np.float32(1.33)).astype(np.float32)

    def channel_mix_1(self, X: np.ndarray) -> np.ndarray:
        """
        Apply orthogonal linear mixing to channels (rotation matrix M_1).
        Simulates small electrode placement variation / inter-channel cross-talk.
        X: (N, C, T) → (N, C, T).
        """
        # einsum "ij,njt->nit": for each window n, output_i = Σ_j M[i,j] * X[n,j,:]
        return np.einsum("ij,njt->nit", self._rot_1, X).astype(np.float32)

    def channel_mix_2(self, X: np.ndarray) -> np.ndarray:
        """Second orthogonal channel mixing (rotation matrix M_2, different seed)."""
        return np.einsum("ij,njt->nit", self._rot_2, X).astype(np.float32)

    def bandstop_powerline(self, X: np.ndarray, fs_nominal: int = 2000) -> np.ndarray:
        """
        Zero-out FFT bins at 45–55 Hz (power-line interference band).
        Simulates an environment where this noise source is absent or suppressed.
        X: (N, C, T) — transform applied along the time axis.
        """
        N, C, T = X.shape
        X_fft = np.fft.rfft(X.astype(np.float64), axis=2)   # (N, C, T//2+1)
        # bin_k corresponds to frequency f_k = k * fs / T
        bin_lo = max(1, int(45.0 * T / fs_nominal))
        bin_hi = min(X_fft.shape[2], int(55.0 * T / fs_nominal) + 2)
        X_fft[:, :, bin_lo:bin_hi] = 0.0
        return np.fft.irfft(X_fft, n=T, axis=2).astype(np.float32)

    def mild_time_compress(self, X: np.ndarray) -> np.ndarray:
        """
        Simulate gesture-speed variation: compress the time axis to 95 % of T,
        then linearly resample back to T.  Fully vectorised (no Python loops).

        Algorithm:
          Step 1 — compress: sample T_new = int(T*0.95) equidistant points
                   from the original T samples via linear interpolation.
          Step 2 — expand:  resample the T_new compressed signal back to T
                   via linear interpolation.
        X: (N, C, T) → (N, C, T).
        """
        N, C, T = X.shape
        T_new = max(2, int(T * 0.95))

        # ── Step 1: T → T_new ──────────────────────────────────────────────
        src_pos   = np.linspace(0, T - 1, T_new, dtype=np.float32)
        fl1       = src_pos.astype(np.int32)
        ce1       = np.minimum(fl1 + 1, T - 1)
        alpha1    = (src_pos - fl1).astype(np.float32)            # (T_new,)

        X_2d      = X.reshape(N * C, T)                            # (NC, T)
        compressed = (
            X_2d[:, fl1] * (1.0 - alpha1)
            + X_2d[:, ce1] * alpha1
        )                                                           # (NC, T_new)

        # ── Step 2: T_new → T ──────────────────────────────────────────────
        dst_pos   = np.linspace(0, T_new - 1, T, dtype=np.float32)
        fl2       = dst_pos.astype(np.int32)
        ce2       = np.minimum(fl2 + 1, T_new - 1)
        alpha2    = (dst_pos - fl2).astype(np.float32)             # (T,)

        resampled = (
            compressed[:, fl2] * (1.0 - alpha2)
            + compressed[:, ce2] * alpha2
        )                                                           # (NC, T)
        return resampled.reshape(N, C, T)

    def get_all_transforms(self) -> List[Tuple[str, Callable]]:
        """Ordered list of (name, fn) for all seven environments."""
        return [
            ("identity",           self.identity),
            ("amp_scale_down",     self.amplitude_scale_down),
            ("amp_scale_up",       self.amplitude_scale_up),
            ("channel_mix_1",      self.channel_mix_1),
            ("channel_mix_2",      self.channel_mix_2),
            ("bandstop_powerline", self.bandstop_powerline),
            ("time_compress",      self.mild_time_compress),
        ]


# ---------------------------------------------------------------------------
# SyntheticEnvTrainer
# ---------------------------------------------------------------------------
class SyntheticEnvTrainer(WindowClassifierTrainer):
    """
    Trains with V-REx or Fishr penalty over synthetic pseudo-environments
    constructed from fixed signal transforms applied to normalised train data.

    Environment modes:
      "transforms_only"   — E environments from pooled train data: one env per
                             transform, each applied to the same random subsample.
      "subject_transform" — N_subj × E environments: each (subject, transform)
                             pair yields one environment.

    LOSO invariant:
      - _build_environments() only receives train_X_norm and train_y.
        Test-subject windows are NEVER passed here.
      - mean_c / std_c are computed from train data only.
      - evaluate_numpy() normalises the test input with the stored (train-derived)
        statistics — no re-fitting on test data takes place.
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        visualizer,
        method: str = "vrex",            # "vrex" | "fishr"
        lambda_penalty: float = 1.0,
        warmup_ratio: float = 0.25,
        num_envs: int = 7,               # max number of transforms used
        env_mode: str = "transforms_only",
        penalty_max_samples: int = 512,  # cap per environment
        env_seed: int = 42,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        assert method in ("vrex", "fishr"), f"Unknown method: {method!r}"
        assert env_mode in ("transforms_only", "subject_transform"), (
            f"Unknown env_mode: {env_mode!r}"
        )
        self.method = method
        self.lambda_penalty = lambda_penalty
        self.warmup_ratio = warmup_ratio
        self.num_envs = num_envs
        self.env_mode = env_mode
        self.penalty_max_samples = penalty_max_samples
        self._env_seed = env_seed

        # Set by fit() — required by evaluate_numpy().
        self.class_ids: Optional[List[int]] = None
        self.class_names: Optional[Dict[int, str]] = None
        self.mean_c: Optional[np.ndarray] = None   # (1, C, 1)
        self.std_c: Optional[np.ndarray] = None    # (1, C, 1)

    # ------------------------------------------------------------------
    # Penalty weight schedule (same annealing as exp_69)
    # ------------------------------------------------------------------
    def _get_penalty_weight(self, epoch: int, total_epochs: int) -> float:
        warmup_epochs = int(self.warmup_ratio * total_epochs)
        if epoch < warmup_epochs:
            return 0.0
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        return self.lambda_penalty * min(progress, 1.0)

    # ------------------------------------------------------------------
    # Build environments (LOSO-safe: train data only)
    # ------------------------------------------------------------------
    def _build_environments(
        self,
        train_X_norm: np.ndarray,           # (N, C, T)  already normalised
        train_y: np.ndarray,                # (N,)
        subject_indices: Dict[str, List[int]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Construct synthetic pseudo-environments for invariance penalty.

        LOSO guarantees:
          - train_X_norm and train_y contain ONLY train-subject windows.
          - subject_indices maps train-subject IDs to index slices in the
            pooled train array — the test subject is never included.
          - Transforms are deterministic functions seeded at __init__ time,
            independent of any data.
        """
        num_ch = train_X_norm.shape[1]
        tfm_bank = SyntheticEMGTransforms(num_channels=num_ch, seed=self._env_seed)
        all_transforms = tfm_bank.get_all_transforms()[: self.num_envs]

        rng = np.random.default_rng(self._env_seed)
        environments: List[Tuple[torch.Tensor, torch.Tensor]] = []

        if self.env_mode == "transforms_only":
            # Sub-sample train data ONCE; apply each transform → one env per transform.
            # All environments share the same base indices, so the penalty measures
            # invariance to transforms applied to the same underlying samples.
            N = train_X_norm.shape[0]
            if N > self.penalty_max_samples:
                sub_idx = rng.choice(N, size=self.penalty_max_samples, replace=False)
            else:
                sub_idx = np.arange(N)

            X_sub = train_X_norm[sub_idx]   # (M, C, T)
            y_sub = train_y[sub_idx]        # (M,)

            for _name, tfm in all_transforms:
                X_env = tfm(X_sub)          # (M, C, T)
                environments.append((
                    torch.tensor(X_env, dtype=torch.float32),
                    torch.tensor(y_sub, dtype=torch.long),
                ))

        else:  # "subject_transform"
            # Each (subject, transform) pair forms one environment.
            # Provides N_subj × E_transforms environments, combining subject
            # diversity (few real domains) with transform diversity.
            for subj_id, idxs in subject_indices.items():
                if not idxs:
                    continue
                idxs_arr = np.array(idxs, dtype=np.int64)
                if len(idxs_arr) > self.penalty_max_samples:
                    # Per-subject sub-sampling uses a subject-specific RNG to
                    # ensure independence from the order subjects are iterated.
                    rng_s = np.random.default_rng(
                        self._env_seed + abs(hash(subj_id)) % (2 ** 20)
                    )
                    idxs_arr = rng_s.choice(
                        idxs_arr, size=self.penalty_max_samples, replace=False
                    )
                X_s = train_X_norm[idxs_arr]   # (M_s, C, T)
                y_s = train_y[idxs_arr]

                for _name, tfm in all_transforms:
                    X_env = tfm(X_s)
                    environments.append((
                        torch.tensor(X_env, dtype=torch.float32),
                        torch.tensor(y_s, dtype=torch.long),
                    ))

        self.logger.info(
            f"  Environments: {len(environments)} "
            f"(mode={self.env_mode}, transforms={len(all_transforms)})"
        )
        return environments

    # ------------------------------------------------------------------
    # V-REx penalty: Var_e(R_e)
    # ------------------------------------------------------------------
    def _compute_vrex_penalty(
        self,
        environments: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        V-REx variance penalty.

        For each environment e (a synthetic transform of train data), compute
        the mean CE loss R_e.  Penalty = Var_e(R_e).

        LOSO note: `environments` contains ONLY synthetic views of TRAIN data.
        The test subject is never part of this list.
        """
        per_env_losses: List[torch.Tensor] = []
        for env_X, env_y in environments:
            env_X = env_X.to(self.cfg.device)
            env_y = env_y.to(self.cfg.device)
            logits   = self.model(env_X)
            env_loss = criterion(logits, env_y)
            per_env_losses.append(env_loss)

        if len(per_env_losses) < 2:
            return torch.tensor(0.0, device=self.cfg.device)

        stacked  = torch.stack(per_env_losses)          # (E,)
        variance = ((stacked - stacked.mean()) ** 2).mean()
        return variance

    # ------------------------------------------------------------------
    # Fishr penalty: Var_e(F_e) where F_e ≈ (∇_θ_cls L_e)²
    # ------------------------------------------------------------------
    def _compute_fishr_penalty(
        self,
        environments: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Fishr gradient-variance penalty.

        Approximates diagonal Fisher of the classifier head as (∇L_e)² and
        penalises the variance of these approximations across environments.
        `create_graph=True` keeps the penalty differentiable so that gradients
        flow through it during the outer loss.backward() call.

        LOSO note: `environments` contains ONLY synthetic transforms of TRAIN
        data — the test subject never contributes to this computation.
        """
        classifier_params = [
            p for p in self.model.gesture_classifier.parameters()
            if p.requires_grad
        ]

        env_fishers: List[torch.Tensor] = []
        for env_X, env_y in environments:
            env_X = env_X.to(self.cfg.device)
            env_y = env_y.to(self.cfg.device)

            logits   = self.model(env_X)
            env_loss = criterion(logits, env_y)

            # Each env has its own independent forward graph — retain_graph
            # is not needed.  create_graph=True makes the penalty part of the
            # main computational graph.
            grads = torch.autograd.grad(
                env_loss,
                classifier_params,
                create_graph=True,
                allow_unused=True,
            )

            grad_parts: List[torch.Tensor] = []
            for g, p in zip(grads, classifier_params):
                if g is None:
                    grad_parts.append(torch.zeros(p.numel(), device=self.cfg.device))
                else:
                    grad_parts.append(g.flatten())
            env_fishers.append(torch.cat(grad_parts) ** 2)   # (param_dim,)

        if len(env_fishers) < 2:
            return torch.tensor(0.0, device=self.cfg.device)

        stacked  = torch.stack(env_fishers)               # (E, param_dim)
        variance = ((stacked - stacked.mean(0)) ** 2).mean()
        return variance

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, splits: Dict) -> Dict:
        """
        Train with V-REx or Fishr penalty over synthetic environments.

        Expected splits keys:
          train_windows          (N_train, T, C)
          train_labels           (N_train,)
          val_windows            (N_val, T, C)
          val_labels             (N_val,)
          test_windows           (N_test, T, C) | None  — held-out, never used
          test_labels            (N_test,)      | None
          train_subject_indices  Dict[subj_id → List[int]] into train arrays

        LOSO contract:
          - test_windows / test_labels are present in `splits` for convenience
            but MUST NOT be accessed during training, environment construction,
            normalisation, or penalty computation.
          - train_subject_indices maps only train-subject IDs to their index
            slices in the pooled train arrays.
        """
        # ---- Unpack ----
        train_windows = splits["train_windows"]    # (N, T, C)
        train_labels  = splits["train_labels"]
        val_windows   = splits["val_windows"]
        val_labels    = splits["val_labels"]
        # test_windows / test_labels: loaded for final evaluation AFTER training.
        # They are NOT used below this line until evaluate_numpy() is called
        # from outside fit().
        subject_indices: Dict[str, List[int]] = splits.get(
            "train_subject_indices", {}
        )

        # ---- Infer dimensions ----
        num_classes    = int(np.unique(train_labels).shape[0])
        self.class_ids = list(range(num_classes))
        self.class_names = {i: f"Gesture {i}" for i in range(num_classes)}

        self.logger.info(
            f"SyntheticEnvTrainer.fit: method={self.method}, mode={self.env_mode}, "
            f"num_classes={num_classes}, lambda={self.lambda_penalty}"
        )

        # ---- Transpose (N, T, C) → (N, C, T) for 1-D CNN ----
        train_X_raw = train_windows.transpose(0, 2, 1).astype(np.float32)
        val_X_raw   = val_windows.transpose(0, 2, 1).astype(np.float32)
        in_channels = train_X_raw.shape[1]

        # ---- Channel normalisation from TRAIN data only (LOSO safe) ----
        # mean_c / std_c computed here and stored for use in evaluate_numpy().
        # Test-subject data never influences these statistics.
        self.mean_c = train_X_raw.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
        self.std_c  = train_X_raw.std(axis=(0, 2),  keepdims=True) + 1e-8
        train_X_norm = (train_X_raw - self.mean_c) / self.std_c
        val_X_norm   = (val_X_raw   - self.mean_c) / self.std_c

        # ---- Build pseudo-environments from normalised TRAIN data ----
        # This call never receives test-subject windows.
        environments = self._build_environments(
            train_X_norm, train_labels, subject_indices
        )

        # ---- Create model ----
        self.model = self._create_model(
            in_channels, num_classes, self.cfg.model_type
        ).to(self.cfg.device)

        # ---- Class-balanced loss ----
        if self.cfg.use_class_weights:
            counts = np.bincount(train_labels, minlength=num_classes).astype(float)
            counts = np.maximum(counts, 1.0)
            w = torch.tensor(
                (1.0 / counts) / (1.0 / counts).sum() * num_classes,
                dtype=torch.float32,
                device=self.cfg.device,
            )
        else:
            w = None
        criterion = nn.CrossEntropyLoss(weight=w)

        # ---- Optimiser & scheduler ----
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # Note: verbose=True removed in PyTorch ≥2.4
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        # ---- Torch tensors ----
        train_X_t = torch.tensor(train_X_norm, dtype=torch.float32)
        train_y_t = torch.tensor(train_labels, dtype=torch.long)
        val_X_t   = torch.tensor(val_X_norm,   dtype=torch.float32)
        val_y_t   = torch.tensor(val_labels,   dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(train_X_t, train_y_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(val_X_t, val_y_t),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # ---- Training loop ----
        best_val_acc   = 0.0
        best_epoch     = 0
        patience_count = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_correct, train_total = 0, 0
            penalty_weight = self._get_penalty_weight(epoch, self.cfg.epochs)

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.cfg.device)
                batch_y = batch_y.to(self.cfg.device)

                optimizer.zero_grad()

                logits   = self.model(batch_X)
                loss_erm = criterion(logits, batch_y)
                loss     = loss_erm

                # Invariance penalty across synthetic environments (TRAIN only).
                # Penalty weight is 0 during warmup, then linearly annealed.
                if penalty_weight > 0.0 and len(environments) > 1:
                    if self.method == "vrex":
                        penalty = self._compute_vrex_penalty(environments, criterion)
                    else:
                        penalty = self._compute_fishr_penalty(environments, criterion)
                    loss = loss + penalty_weight * penalty

                loss.backward()
                optimizer.step()

                _, pred = logits.max(1)
                train_total   += batch_y.size(0)
                train_correct += pred.eq(batch_y).sum().item()

            train_acc = train_correct / max(train_total, 1)
            val_acc, val_f1, _ = self._eval_loader(val_loader, criterion)
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                best_epoch     = epoch
                patience_count = 0
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best_model.pt",
                )
            else:
                patience_count += 1

            if epoch % 5 == 0 or epoch == self.cfg.epochs - 1:
                self.logger.info(
                    f"Epoch {epoch}/{self.cfg.epochs}: "
                    f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
                    f"val_f1={val_f1:.4f}, penalty_w={penalty_weight:.4f}"
                )

            if patience_count >= self.cfg.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best checkpoint.
        best_ckpt = self.output_dir / "best_model.pt"
        if best_ckpt.exists():
            self.model.load_state_dict(
                torch.load(best_ckpt, map_location=self.cfg.device)
            )
        self.logger.info(
            f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}"
        )

        return {
            "best_epoch":        best_epoch,
            "best_val_accuracy": best_val_acc,
            "num_environments":  len(environments),
        }

    # ------------------------------------------------------------------
    # Internal evaluation helper
    # ------------------------------------------------------------------
    def _eval_loader(
        self, loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float, float]:
        self.model.eval()
        all_preds, all_labels, total_loss = [], [], 0.0
        with torch.no_grad():
            for bX, by in loader:
                bX = bX.to(self.cfg.device)
                by = by.to(self.cfg.device)
                logits      = self.model(bX)
                total_loss += criterion(logits, by).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(by.cpu().numpy())
        arr_pred = np.array(all_preds)
        arr_true = np.array(all_labels)
        acc  = float((arr_pred == arr_true).mean())
        f1   = self._compute_f1_macro(arr_true, arr_pred)
        loss = total_loss / max(len(loader), 1)
        return acc, f1, loss

    @staticmethod
    def _compute_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ------------------------------------------------------------------
    # evaluate_numpy — single forward pass on held-out test subject
    # ------------------------------------------------------------------
    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "test",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate on raw (N, T, C) windows.

        Uses the train-derived mean_c / std_c for normalisation.
        This is the ONLY contact between the model and test-subject data:
        one forward pass with no gradient updates.  No information from X
        (test subject) flows back into the model.
        """
        assert self.mean_c is not None, "fit() must be called before evaluate_numpy()"
        self.model.eval()

        X_t    = X.transpose(0, 2, 1).astype(np.float32)   # (N, C, T)
        X_norm = (X_t - self.mean_c) / self.std_c
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.cfg.device)

        with torch.no_grad():
            preds = self.model(X_tensor).argmax(1).cpu().numpy()

        acc = float((preds == y).mean())
        f1  = self._compute_f1_macro(y, preds)
        return {"accuracy": acc, "f1_macro": f1, "predictions": preds}


# ---------------------------------------------------------------------------
# Single LOSO fold
# ---------------------------------------------------------------------------
def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    method: str,
    env_mode: str,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    lambda_penalty: float = 1.0,
    num_envs: int = 7,
    penalty_max_samples: int = 512,
    val_ratio: float = 0.15,
    max_gestures: int = 10,
    seed: int = 42,
) -> Dict:
    """
    One LOSO fold: train on ``train_subjects``, evaluate on ``test_subject``.

    LOSO compliance contract:
      1. ``test_subject`` data is loaded via load_multiple_subjects but stored
         in a separate entry — it is extracted only after all train splits are
         built and never used to compute normalization statistics, environments,
         or training loss.
      2. ``subject_train_indices`` is populated only for train-subject IDs.
         The test subject never has an entry here.
      3. ``_build_environments()`` receives only ``train_X_norm`` and
         ``train_labels`` — no test-subject data passes through it.
      4. Final evaluation calls ``trainer.evaluate_numpy(test_windows, test_labels)``
         — a single gradient-free forward pass with no model update.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(seed, verbose=False)

    train_cfg.pipeline_type = "deep_raw"
    train_cfg.model_type    = model_type

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "fold_params.json", "w") as f:
        json.dump(
            {
                "train_subjects": train_subjects,
                "test_subject":   test_subject,
                "method":         method,
                "env_mode":       env_mode,
                "lambda_penalty": lambda_penalty,
                "num_envs":       num_envs,
            },
            f,
            indent=4,
        )

    # ---- Load subjects (train + test) ----
    # The test subject is loaded here but its data is isolated below and
    # never contributes to training, normalisation, or environment construction.
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=True,
    )
    all_subjects = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subjects,
        exercises=exercises,
        include_rest=False,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=max_gestures
    )
    gesture_to_class: Dict[int, int] = {
        gid: idx for idx, gid in enumerate(sorted(common_gestures))
    }
    logger.info(
        f"Common gestures: {sorted(common_gestures)}, "
        f"num_classes={len(gesture_to_class)}"
    )

    # ---- Build per-subject train/val splits (train subjects ONLY) ----
    train_windows_list: List[np.ndarray] = []
    train_labels_list:  List[np.ndarray] = []
    val_windows_list:   List[np.ndarray] = []
    val_labels_list:    List[np.ndarray] = []
    subject_train_indices: Dict[str, List[int]] = {}
    current_idx = 0

    for subj_id in train_subjects:
        if subj_id not in subjects_data:
            logger.warning(f"Subject {subj_id} not found — skipping")
            continue

        _, _, grouped_windows = subjects_data[subj_id]
        windows, labels = grouped_to_arrays(grouped_windows)

        mask    = np.isin(labels, list(gesture_to_class.keys()))
        windows = windows[mask]
        labels  = np.array(
            [gesture_to_class[lbl] for lbl in labels[mask]], dtype=np.int64
        )

        if len(windows) == 0:
            logger.warning(f"Subject {subj_id}: 0 windows after gesture filter")
            continue

        # Per-subject deterministic train/val split.
        # Using a subject-specific seed (derived from global seed + hash of subject
        # ID) ensures that the split for each subject is independent of the order
        # subjects are processed — a necessary condition for reproducibility.
        subj_seed = seed + abs(hash(subj_id)) % (2 ** 31)
        rng   = np.random.default_rng(subj_seed)
        perm  = rng.permutation(len(windows))
        n_val = max(1, int(len(windows) * val_ratio))
        t_idx = perm[n_val:]
        v_idx = perm[:n_val]

        # Record contiguous slice in the pooled train array for this subject.
        subject_train_indices[subj_id] = list(
            range(current_idx, current_idx + len(t_idx))
        )
        current_idx += len(t_idx)

        train_windows_list.append(windows[t_idx])
        train_labels_list.append(labels[t_idx])
        val_windows_list.append(windows[v_idx])
        val_labels_list.append(labels[v_idx])

    if not train_windows_list:
        raise RuntimeError(
            f"No training data for fold test={test_subject}"
        )

    # ---- Test split (test subject — NEVER used during training) ----
    test_windows_arr: Optional[np.ndarray] = None
    test_labels_arr:  Optional[np.ndarray] = None

    if test_subject in subjects_data:
        _, _, gw_test = subjects_data[test_subject]
        w_test, l_test = grouped_to_arrays(gw_test)
        mask_t = np.isin(l_test, list(gesture_to_class.keys()))
        w_test = w_test[mask_t]
        l_test = np.array(
            [gesture_to_class[lbl] for lbl in l_test[mask_t]], dtype=np.int64
        )
        if len(w_test) > 0:
            test_windows_arr = w_test
            test_labels_arr  = l_test
        else:
            logger.warning(
                f"Test subject {test_subject}: 0 windows after gesture filter"
            )
    else:
        logger.warning(f"Test subject {test_subject} not found in loaded data!")

    # ---- Concatenate pooled train / val arrays ----
    train_windows = np.concatenate(train_windows_list, axis=0)
    train_labels  = np.concatenate(train_labels_list,  axis=0)
    val_windows   = np.concatenate(val_windows_list,   axis=0)
    val_labels    = np.concatenate(val_labels_list,    axis=0)

    logger.info(
        f"Shapes: train={train_windows.shape}, val={val_windows.shape}, "
        f"test={test_windows_arr.shape if test_windows_arr is not None else None}"
    )
    logger.info(f"Subject train indices: {list(subject_train_indices.keys())}")

    # ---- Create trainer and run ----
    visualizer = Visualizer(output_dir, logger)
    trainer = SyntheticEnvTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=visualizer,
        method=method,
        lambda_penalty=lambda_penalty,
        warmup_ratio=0.25,
        num_envs=num_envs,
        env_mode=env_mode,
        penalty_max_samples=penalty_max_samples,
        env_seed=seed,
    )

    # The splits dict contains test_windows for convenience (evaluate_numpy is
    # called from outside fit()), but fit() must not access them for training.
    splits = {
        "train_windows":         train_windows,
        "train_labels":          train_labels,
        "val_windows":           val_windows,
        "val_labels":            val_labels,
        "test_windows":          test_windows_arr,
        "test_labels":           test_labels_arr,
        "train_subject_indices": subject_train_indices,
    }

    try:
        fit_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed for fold test={test_subject}: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "method":        method,
            "env_mode":      env_mode,
            "model_type":    model_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ---- Final evaluation on held-out test subject ----
    # This is the ONLY point at which test-subject data interacts with the model:
    # a single forward pass with no gradient computation.
    if test_windows_arr is not None:
        test_metrics = trainer.evaluate_numpy(
            test_windows_arr, test_labels_arr, "test"
        )
        test_acc = float(test_metrics["accuracy"])
        test_f1  = float(test_metrics["f1_macro"])
    else:
        test_acc = test_f1 = None

    acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
    f1_str  = f"{test_f1:.4f}"  if test_f1  is not None else "N/A"
    print(
        f"[LOSO] {test_subject} | method={method}, env_mode={env_mode} | "
        f"acc={acc_str}, f1={f1_str}"
    )

    # Save fold metadata.
    fold_meta = {
        "test_subject":      test_subject,
        "train_subjects":    train_subjects,
        "method":            method,
        "env_mode":          env_mode,
        "model_type":        model_type,
        "lambda_penalty":    lambda_penalty,
        "num_envs":          num_envs,
        "exercises":         exercises,
        "num_environments":  fit_results.get("num_environments"),
        "best_val_accuracy": fit_results.get("best_val_accuracy"),
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable(fold_meta), filename="fold_metadata.json"
    )

    # Memory cleanup.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, visualizer, subjects_data
    gc.collect()

    return {
        "test_subject":  test_subject,
        "method":        method,
        "env_mode":      env_mode,
        "model_type":    model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    EXPERIMENT_NAME = "exp_80_synthetic_env_vrex_fishr_loso"
    HYPOTHESIS_ID   = "h-080-synthetic-env-vrex-fishr"

    BASE_DIR   = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")

    # ---- CLI parsing ----
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Comma-separated methods to run: vrex,fishr (default: both)",
    )
    _parser.add_argument(
        "--env_mode",
        type=str,
        default=None,
        help=(
            "Comma-separated env modes to run: "
            "transforms_only,subject_transform (default: both)"
        ),
    )
    _args, _ = _parser.parse_known_args()

    ALL_SUBJECTS = parse_subjects_args()   # defaults to CI_TEST_SUBJECTS
    EXERCISES    = ["E1"]
    MODEL_TYPE   = "irm_content_style_emg"   # same backbone as exp_69

    METHODS = (
        [m.strip() for m in _args.method.split(",")]
        if _args.method
        else ["vrex", "fishr"]
    )
    ENV_MODES = (
        [m.strip() for m in _args.env_mode.split(",")]
        if _args.env_mode
        else ["transforms_only", "subject_transform"]
    )

    # Lambda values: V-REx penalises loss variance (scale comparable to CE),
    # Fishr penalises gradient variance (tends to be larger → smaller lambda).
    LAMBDA_VREX         = 1.0
    LAMBDA_FISHR        = 0.1
    NUM_ENVS            = 7     # all seven transforms
    PENALTY_MAX_SAMPLES = 512   # cap per environment for penalty computation
    MAX_GESTURES        = 10
    VAL_RATIO           = 0.15
    SEED                = 42

    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=VAL_RATIO,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=SEED,
        include_rest_in_splits=False,
    )
    # No data augmentation in the base training loop — stimulus diversity is
    # handled entirely through the synthetic environments.
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=60,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=SEED,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type=MODEL_TYPE,
        pipeline_type="deep_raw",
        aug_apply=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"HYPOTHESIS: {HYPOTHESIS_ID}")
    print(f"Methods: {METHODS}  |  Env modes: {ENV_MODES}")
    print(f"Backbone: {MODEL_TYPE}  (same as exp_69 for fair comparison)")
    print(f"Subjects: {ALL_SUBJECTS}  ({len(ALL_SUBJECTS)} total)")
    print(
        f"Transforms (environments): identity, amp_scale_down, amp_scale_up, "
        f"channel_mix_1, channel_mix_2, bandstop_powerline, time_compress  "
        f"(NUM_ENVS={NUM_ENVS})"
    )
    print(
        f"Lambda V-REx={LAMBDA_VREX},  Lambda Fishr={LAMBDA_FISHR},  "
        f"penalty_max_samples={PENALTY_MAX_SAMPLES}"
    )

    # ---- LOSO loop ----
    all_results: List[Dict] = []

    for env_mode in ENV_MODES:
        for method in METHODS:
            lambda_penalty = LAMBDA_VREX if method == "vrex" else LAMBDA_FISHR

            for test_subject in ALL_SUBJECTS:
                train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
                fold_out = OUTPUT_DIR / env_mode / method / f"test_{test_subject}"

                try:
                    fold_res = run_single_loso_fold(
                        base_dir=BASE_DIR,
                        output_dir=fold_out,
                        train_subjects=train_subjects,
                        test_subject=test_subject,
                        exercises=EXERCISES,
                        model_type=MODEL_TYPE,
                        method=method,
                        env_mode=env_mode,
                        proc_cfg=proc_cfg,
                        split_cfg=split_cfg,
                        train_cfg=train_cfg,
                        lambda_penalty=lambda_penalty,
                        num_envs=NUM_ENVS,
                        penalty_max_samples=PENALTY_MAX_SAMPLES,
                        val_ratio=VAL_RATIO,
                        max_gestures=MAX_GESTURES,
                        seed=SEED,
                    )
                    all_results.append(fold_res)

                    acc_str = (
                        f"{fold_res['test_accuracy']:.4f}"
                        if fold_res.get("test_accuracy") is not None
                        else "N/A"
                    )
                    f1_str = (
                        f"{fold_res['test_f1_macro']:.4f}"
                        if fold_res.get("test_f1_macro") is not None
                        else "N/A"
                    )
                    print(
                        f"  [{env_mode}/{method}] {test_subject}: "
                        f"acc={acc_str}, f1={f1_str}"
                    )

                except Exception as e:
                    global_logger.error(
                        f"Fold failed: env_mode={env_mode}, method={method}, "
                        f"test={test_subject}: {e}"
                    )
                    traceback.print_exc()
                    all_results.append({
                        "test_subject":  test_subject,
                        "method":        method,
                        "env_mode":      env_mode,
                        "model_type":    MODEL_TYPE,
                        "test_accuracy": None,
                        "test_f1_macro": None,
                        "error":         str(e),
                    })

    # ---- Aggregate by (env_mode, method) ----
    aggregate: Dict[str, Dict] = {}
    for env_mode in ENV_MODES:
        for method in METHODS:
            key = f"{env_mode}/{method}"
            subset = [
                r for r in all_results
                if r.get("env_mode") == env_mode
                and r.get("method") == method
                and r.get("test_accuracy") is not None
            ]
            if not subset:
                continue
            accs = [r["test_accuracy"] for r in subset]
            f1s  = [r["test_f1_macro"] for r in subset]
            aggregate[key] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
                "num_subjects":  len(accs),
            }
            m = aggregate[key]
            print(
                f"\n[{key.upper()}]  "
                f"Acc = {m['mean_accuracy']:.4f} ± {m['std_accuracy']:.4f}  |  "
                f"F1 = {m['mean_f1_macro']:.4f} ± {m['std_f1_macro']:.4f}"
            )

    # ---- Save summary ----
    summary = {
        "experiment_name":      EXPERIMENT_NAME,
        "hypothesis_id":        HYPOTHESIS_ID,
        "backbone":             MODEL_TYPE,
        "note": (
            "exp_80 (synthetic transform environments) vs "
            "exp_69 (subject-based environments) — same backbone and "
            "penalty formulas, different environment source."
        ),
        "methods":              METHODS,
        "env_modes":            ENV_MODES,
        "subjects":             ALL_SUBJECTS,
        "exercises":            EXERCISES,
        "num_envs":             NUM_ENVS,
        "lambda_vrex":          LAMBDA_VREX,
        "lambda_fishr":         LAMBDA_FISHR,
        "penalty_max_samples":  PENALTY_MAX_SAMPLES,
        "transforms": [
            "identity",
            "amp_scale_down (×0.75)",
            "amp_scale_up (×1.33)",
            "channel_mix_1 (orthogonal rotation, seed+1)",
            "channel_mix_2 (orthogonal rotation, seed+2)",
            "bandstop_powerline (FFT null 45–55 Hz)",
            "time_compress (95 % speed, resampled back to T)",
        ],
        "processing_config":    asdict(proc_cfg),
        "training_config":      asdict(train_cfg),
        "aggregate_results":    aggregate,
        "individual_results":   all_results,
        "experiment_date":      datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")

    # ---- Hypothesis executor callback (optional dependency) ----
    try:
        from hypothesis_executor.qdrant_callback import (
            mark_hypothesis_failed,
            mark_hypothesis_verified,
        )
        if aggregate:
            best_key = max(
                aggregate, key=lambda k: aggregate[k]["mean_accuracy"]
            )
            best_metrics = dict(aggregate[best_key])
            best_metrics["best_config"] = best_key
            mark_hypothesis_verified(
                hypothesis_id=HYPOTHESIS_ID,
                metrics=best_metrics,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                hypothesis_id=HYPOTHESIS_ID,
                error_message="No successful LOSO folds completed",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
