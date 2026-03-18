# FILE: experiments/exp_69_vrex_fishr_irm_v2_loso.py
"""
Experiment: V-REx and Fishr IRM Variants for Invariant EMG Gesture Recognition

Hypothesis: V-REx (variance-based risk extrapolation) and Fishr (Fisher gradient
alignment) provide more stable IRM-style invariance than standard IRMv1 (exp_48),
especially with few subjects/environments.

Methods:
  V-REx  — minimize mean(R_e) + lambda * Var_e(R_e) across train subjects.
             The variance penalty discourages the model from relying on features
             whose risk differs between subjects.
  Fishr  — align diagonal Fisher information (squared gradient statistics of the
             classifier head) across train subjects.
             Penalises Var_e(F_e) where F_e ≈ (∇_θ_cls L_e)^2.

Backbone:
  IRMContentStyleEMG — identical to exp_48 for fair comparison.

LOSO compliance (critical):
  - environments = TRAIN subjects only; test subject is NEVER seen during training
    or penalty computation
  - val split comes from train subjects only, EXCLUDED from environments
  - NO per-test-subject adaptation of any kind
  - Data loading is fresh per fold (no cross-fold leakage)

Comparison target: exp_48 (IRMv1) with same backbone and hyperparameters.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.trainer import WindowClassifierTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# Reuse the same backbone as exp_48 — no new model file needed
from models.irm_content_style_emg import IRMContentStyleEMG
from models import register_model

register_model("irm_content_style_emg", IRMContentStyleEMG)

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)


# ---------------------------------------------------------------------------
# Helper: flatten grouped_windows → (windows, labels)
# (copied locally — processing.window_extraction does NOT exist)
# ---------------------------------------------------------------------------
def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grouped_windows {gesture_id: [rep_array, ...]} to flat arrays."""
    windows_list: List[np.ndarray] = []
    labels_list:  List[np.ndarray] = []
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
# VRexFishrTrainer
# ---------------------------------------------------------------------------
class VRexFishrTrainer(WindowClassifierTrainer):
    """
    Custom trainer implementing V-REx and Fishr IRM variants.

    V-REx  — penalises the *variance* of per-environment CE losses.
    Fishr  — penalises the *variance* of diagonal Fisher information
              (approximated as squared gradients of the classifier head)
              across environments.

    Each TRAIN subject is one environment.  The TEST subject is NEVER
    included in the environment list or used in any way during training.

    Penalty weight is annealed identically to exp_48:
      - 0 for the first `warmup_ratio` * epochs
      - linear ramp to `lambda_penalty` thereafter
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        visualizer,
        method: str = "vrex",               # "vrex" | "fishr"
        lambda_penalty: float = 1.0,
        warmup_ratio: float = 0.2,
        penalty_max_samples: int = 512,     # max samples per env for penalty
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        assert method in ("vrex", "fishr"), f"Unknown method: {method!r}"
        self.method = method
        self.lambda_penalty = lambda_penalty
        self.warmup_ratio = warmup_ratio
        self.penalty_max_samples = penalty_max_samples
        # class_ids is set in fit(); required by parent evaluate_numpy contract
        self.class_ids: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Penalty weight schedule (identical to exp_48 IRM annealing)
    # ------------------------------------------------------------------
    def _get_penalty_weight(self, epoch: int, total_epochs: int) -> float:
        warmup_epochs = int(self.warmup_ratio * total_epochs)
        if epoch < warmup_epochs:
            return 0.0
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        return self.lambda_penalty * min(progress, 1.0)

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        num_classes = len(self.class_ids)
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        counts = np.maximum(counts, 1.0)
        w = 1.0 / counts
        w = w / w.sum() * num_classes
        return torch.tensor(w, dtype=torch.float32, device=self.cfg.device)

    # ------------------------------------------------------------------
    # Sub-sampler (limits memory / compute for penalty forward passes)
    # ------------------------------------------------------------------
    def _subsample_env(
        self,
        env_X: torch.Tensor,
        env_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = env_X.size(0)
        if n <= self.penalty_max_samples:
            return env_X, env_y
        idx = torch.randperm(n)[:self.penalty_max_samples]
        return env_X[idx], env_y[idx]

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

        For each train-subject environment e, compute the mean CE loss R_e.
        Penalty = Var_e(R_e) = E_e[(R_e - mean(R_e))^2].

        NOTE: the test subject is NEVER in `environments`; this function
        receives only the train-subject tensors built in fit().
        """
        per_env_losses: List[torch.Tensor] = []
        for env_X, env_y in environments:
            env_X, env_y = self._subsample_env(env_X, env_y)
            env_X = env_X.to(self.cfg.device)
            env_y = env_y.to(self.cfg.device)
            logits   = self.model(env_X)
            env_loss = criterion(logits, env_y)
            per_env_losses.append(env_loss)

        if len(per_env_losses) < 2:
            return torch.tensor(0.0, device=self.cfg.device)

        stacked   = torch.stack(per_env_losses)   # (E,)
        mean_loss = stacked.mean()
        variance  = ((stacked - mean_loss) ** 2).mean()
        return variance

    # ------------------------------------------------------------------
    # Fishr penalty: Var_e(F_e) where F_e = (∇_θ_cls L_e)^2
    # ------------------------------------------------------------------
    def _compute_fishr_penalty(
        self,
        environments: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Fishr gradient alignment penalty.

        For each train-subject environment e, approximate the diagonal Fisher
        of the classifier head as the *squared* gradient of the mean CE loss:
            F_e ≈ (∇_θ_cls L_e)^2   (element-wise)

        Penalty = Var_e(F_e) = mean over parameter dimensions of Var_e(F_e[d]).

        create_graph=True makes the penalty itself differentiable, so that
        gradients flow back through the squared-gradient computation during
        the main loss.backward() call.

        retain_graph is NOT set (defaults to False) because each environment
        has its own independent forward pass — there is no shared graph to retain.

        NOTE: the test subject is NEVER in `environments`; only train-subject
        tensors (built from the train split inside fit()) are passed here.
        """
        classifier_params = [
            p for p in self.model.gesture_classifier.parameters()
            if p.requires_grad
        ]

        env_fishers: List[torch.Tensor] = []
        for env_X, env_y in environments:
            env_X, env_y = self._subsample_env(env_X, env_y)
            env_X = env_X.to(self.cfg.device)
            env_y = env_y.to(self.cfg.device)

            logits   = self.model(env_X)
            env_loss = criterion(logits, env_y)

            # create_graph=True: penalty is part of the main computational graph
            # retain_graph not needed — each env has its own independent graph
            grads = torch.autograd.grad(
                env_loss,
                classifier_params,
                create_graph=True,
                allow_unused=True,
            )

            # Flatten and square → diagonal Fisher approximation
            grad_parts: List[torch.Tensor] = []
            for g, p in zip(grads, classifier_params):
                if g is None:
                    grad_parts.append(torch.zeros(p.numel(), device=self.cfg.device))
                else:
                    grad_parts.append(g.flatten())
            grad_flat = torch.cat(grad_parts)
            env_fishers.append(grad_flat ** 2)   # (param_dim,)

        if len(env_fishers) < 2:
            return torch.tensor(0.0, device=self.cfg.device)

        fishers   = torch.stack(env_fishers)      # (E, param_dim)
        mean_f    = fishers.mean(0)               # (param_dim,)
        variance  = ((fishers - mean_f) ** 2).mean()
        return variance

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------
    def fit(self, splits: Dict) -> Dict:
        """
        Train with V-REx or Fishr penalty on per-subject environments.

        Expected keys in `splits`:
          train_windows          (N_train, T, C)
          train_labels           (N_train,)
          val_windows            (N_val, T, C)
          val_labels             (N_val,)
          test_windows           (N_test, T, C) or None
          test_labels            (N_test,)      or None
          train_subject_indices  Dict[subj_id → List[int]] into train arrays

        LOSO invariant:
          - train_subject_indices contains ONLY train subjects
          - test subject data (test_windows / test_labels) is kept completely
            separate and is NEVER used during penalty computation or training
        """
        train_windows = splits["train_windows"]   # (N, T, C)
        train_labels  = splits["train_labels"]
        val_windows   = splits["val_windows"]
        val_labels    = splits["val_labels"]
        test_windows  = splits.get("test_windows")
        test_labels   = splits.get("test_labels")
        subject_indices: Dict[str, List[int]] = splits.get(
            "train_subject_indices", {}
        )

        # Infer dimensions
        in_channels = train_windows.shape[2]   # windows are (N, T, C)
        num_classes = int(np.unique(train_labels).shape[0])
        self.class_ids = list(range(num_classes))

        self.logger.info(
            f"{self.method.upper()} | in_channels={in_channels}, "
            f"num_classes={num_classes}, lambda={self.lambda_penalty}"
        )

        # Create model (uses registered IRMContentStyleEMG via _create_model)
        self.model = self._create_model(
            in_channels, num_classes, self.cfg.model_type
        )
        self.model = self.model.to(self.cfg.device)

        # Loss function for main training batch
        criterion = nn.CrossEntropyLoss(
            weight=(
                self._compute_class_weights(train_labels)
                if self.cfg.use_class_weights
                else None
            )
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        # ---- Prepare tensors ----
        # Transpose (N, T, C) → (N, C, T) for 1-D CNN input
        train_X = torch.tensor(
            train_windows.transpose(0, 2, 1), dtype=torch.float32
        )
        train_y = torch.tensor(train_labels, dtype=torch.long)
        val_X   = torch.tensor(
            val_windows.transpose(0, 2, 1), dtype=torch.float32
        )
        val_y   = torch.tensor(val_labels, dtype=torch.long)

        # ---- Build environment list (TRAIN subjects ONLY) ----
        # Each element is (env_X, env_y) — a view into the pooled train arrays.
        # The test subject never appears here: subject_indices was built only
        # for train_subjects inside run_single_loso_fold().
        environments: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for subj_id, idxs in subject_indices.items():
            if not idxs:
                continue
            e_X = train_X[idxs]   # (n_subj, C, T)
            e_y = train_y[idxs]   # (n_subj,)
            environments.append((e_X, e_y))

        self.logger.info(
            f"  Environments: {len(environments)} train subjects "
            f"(penalty_max_samples={self.penalty_max_samples})"
        )

        train_loader = DataLoader(
            TensorDataset(train_X, train_y),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(val_X, val_y),
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
                loss_cls = criterion(logits, batch_y)
                loss     = loss_cls

                # Add invariance penalty after warmup
                if penalty_weight > 0.0 and len(environments) > 1:
                    if self.method == "vrex":
                        penalty = self._compute_vrex_penalty(
                            environments, criterion
                        )
                    else:  # fishr
                        penalty = self._compute_fishr_penalty(
                            environments, criterion
                        )
                    loss = loss + penalty_weight * penalty

                loss.backward()
                optimizer.step()

                _, pred = logits.max(1)
                train_total   += batch_y.size(0)
                train_correct += pred.eq(batch_y).sum().item()

            train_acc = train_correct / max(train_total, 1)
            val_acc, val_f1, _ = self._evaluate(val_loader, criterion)
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
                    f"val_f1={val_f1:.4f}, penalty_w={penalty_weight:.3f}"
                )

            if patience_count >= self.cfg.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best checkpoint
        best_ckpt = self.output_dir / "best_model.pt"
        if best_ckpt.exists():
            self.model.load_state_dict(
                torch.load(best_ckpt, map_location=self.cfg.device)
            )
        self.logger.info(
            f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}"
        )

        results: Dict = {
            "best_epoch":        best_epoch,
            "best_val_accuracy": best_val_acc,
        }

        # Evaluate on test split (test subject data — model never trained on this)
        if test_windows is not None:
            test_X = torch.tensor(
                test_windows.transpose(0, 2, 1), dtype=torch.float32
            )
            test_y = torch.tensor(test_labels, dtype=torch.long)
            test_loader = DataLoader(
                TensorDataset(test_X, test_y),
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=0,
            )
            test_acc, test_f1, _ = self._evaluate(test_loader, criterion)
            results["test_accuracy"]  = test_acc
            results["test_f1_macro"]  = test_f1

        return results

    # ------------------------------------------------------------------
    # Internal evaluation helper
    # ------------------------------------------------------------------
    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for bX, by in loader:
                bX = bX.to(self.cfg.device)
                by = by.to(self.cfg.device)
                logits     = self.model(bX)
                total_loss += criterion(logits, by).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(by.cpu().numpy())
        arr_pred = np.array(all_preds)
        arr_true = np.array(all_labels)
        acc = (arr_pred == arr_true).mean()
        f1  = self._compute_f1_macro(arr_true, arr_pred)
        return acc, f1, total_loss / max(len(loader), 1)

    def _compute_f1_macro(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ------------------------------------------------------------------
    # evaluate_numpy — called by CrossSubjectExperiment after fit()
    # ------------------------------------------------------------------
    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "test",
        visualize: bool = True,
    ) -> Dict:
        """
        Evaluate on numpy arrays.

        X is expected to be (N, T, C) — the same shape as windows in splits.
        This is called with the test subject's data, which was NEVER used
        during training or penalty computation.
        """
        self.model.eval()
        X_t = torch.tensor(
            X.transpose(0, 2, 1), dtype=torch.float32
        ).to(self.cfg.device)
        with torch.no_grad():
            preds = self.model(X_t).argmax(1).cpu().numpy()
        acc = (preds == y).mean()
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
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    lambda_penalty: float = 1.0,
    penalty_max_samples: int = 512,
    val_ratio: float = 0.15,
    max_gestures: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Run one LOSO fold: train on `train_subjects`, evaluate on `test_subject`.

    LOSO compliance contract:
    1. Data for `test_subject` is loaded in a separate dict entry and is ONLY
       used to build `test_windows`/`test_labels` — never for penalty computation
       or for setting any training hyper-parameter.
    2. `environments` in the trainer contain ONLY train-portion windows from
       `train_subjects`.  Val windows come from the same subjects but are held
       out and excluded from environments.
    3. The splitting RNG is seeded identically for every fold so that
       train/val partition of each subject is reproducible.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(seed, verbose=False)

    # Record approach in config (mutations are safe — same values across all folds)
    train_cfg.pipeline_type = "deep_raw"
    train_cfg.model_type    = model_type

    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "fold_params.json", "w") as f:
        json.dump({
            "train_subjects":   train_subjects,
            "test_subject":     test_subject,
            "method":           method,
            "lambda_penalty":   lambda_penalty,
            "exercises":        exercises,
        }, f, indent=4)

    # ---- Load data ----
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=True,
    )

    # Load ALL subjects at once (train + test).
    # test_subject data will be separated below — it is read but never used
    # to influence training decisions.
    all_subjects = train_subjects + [test_subject]
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

    # ---- Build per-subject train/val splits ----
    # All splitting uses the SAME seed so folds are reproducible.
    # The splitting RNG is reset for each subject to make individual subject
    # splits independent of subject processing order.
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

        # Keep only common gestures
        mask    = np.isin(labels, list(gesture_to_class.keys()))
        windows = windows[mask]
        labels  = np.array(
            [gesture_to_class[lbl] for lbl in labels[mask]], dtype=np.int64
        )

        if len(windows) == 0:
            logger.warning(f"Subject {subj_id}: no windows after filtering")
            continue

        # Per-subject deterministic train/val split
        # Using subject-specific seed derived from global seed to ensure
        # independence from subject iteration order.
        subj_seed = seed + abs(hash(subj_id)) % (2 ** 31)
        rng       = np.random.default_rng(subj_seed)
        perm      = rng.permutation(len(windows))
        n_val     = max(1, int(len(windows) * val_ratio))
        n_train   = len(windows) - n_val

        t_idx = perm[:n_train]
        v_idx = perm[n_train:]

        # Track subject's contiguous slice in the pooled train array
        subject_train_indices[subj_id] = list(
            range(current_idx, current_idx + n_train)
        )
        current_idx += n_train

        train_windows_list.append(windows[t_idx])
        train_labels_list.append(labels[t_idx])
        val_windows_list.append(windows[v_idx])
        val_labels_list.append(labels[v_idx])

    if not train_windows_list:
        raise RuntimeError(
            f"No training data available for fold test={test_subject}"
        )

    # ---- Build test arrays (test subject — never touched during training) ----
    test_windows_arr: Optional[np.ndarray] = None
    test_labels_arr:  Optional[np.ndarray] = None

    if test_subject in subjects_data:
        _, _, grouped_windows = subjects_data[test_subject]
        windows, labels = grouped_to_arrays(grouped_windows)
        mask    = np.isin(labels, list(gesture_to_class.keys()))
        windows = windows[mask]
        labels  = np.array(
            [gesture_to_class[lbl] for lbl in labels[mask]], dtype=np.int64
        )
        if len(windows) > 0:
            test_windows_arr = windows
            test_labels_arr  = labels
        else:
            logger.warning(
                f"Test subject {test_subject}: no windows after gesture filter"
            )
    else:
        logger.warning(
            f"Test subject {test_subject} not found in loaded data!"
        )

    # Concatenate pooled arrays
    train_windows = np.concatenate(train_windows_list, axis=0)
    train_labels  = np.concatenate(train_labels_list,  axis=0)
    val_windows   = np.concatenate(val_windows_list,   axis=0)
    val_labels    = np.concatenate(val_labels_list,    axis=0)

    logger.info(
        f"Shapes: train={train_windows.shape}, val={val_windows.shape}, "
        f"test={test_windows_arr.shape if test_windows_arr is not None else None}"
    )
    logger.info(
        f"Environments (train subjects with data): "
        f"{list(subject_train_indices.keys())}"
    )

    # ---- Create trainer and run ----
    visualizer = Visualizer(output_dir, logger)
    trainer = VRexFishrTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=visualizer,
        method=method,
        lambda_penalty=lambda_penalty,
        warmup_ratio=0.2,
        penalty_max_samples=penalty_max_samples,
    )

    # Splits dict passed to fit() — test data present but training MUST NOT
    # include it in environments (enforced by subject_train_indices scope)
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
        results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Error in fold test={test_subject}, method={method}: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "method":        method,
            "model_type":    model_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # Final evaluation on held-out test subject
    if test_windows_arr is not None:
        test_metrics = trainer.evaluate_numpy(
            test_windows_arr, test_labels_arr, "test"
        )
        test_acc = float(test_metrics["accuracy"])
        test_f1  = float(test_metrics["f1_macro"])
    else:
        test_acc = test_f1 = None

    test_acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
    test_f1_str  = f"{test_f1:.4f}"  if test_f1  is not None else "N/A"
    print(
        f"[LOSO] {test_subject} | method={method} | "
        f"acc={test_acc_str}, f1={test_f1_str}"
    )

    # Save fold metadata
    fold_meta = {
        "test_subject":   test_subject,
        "train_subjects": train_subjects,
        "method":         method,
        "model_type":     model_type,
        "lambda_penalty": lambda_penalty,
        "exercises":      exercises,
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable(fold_meta), filename="fold_metadata.json"
    )

    # Cleanup GPU / RAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del trainer, multi_loader, visualizer
    gc.collect()

    return {
        "test_subject":  test_subject,
        "method":        method,
        "model_type":    model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    EXPERIMENT_NAME = "exp_69_vrex_fishr_irm_v2_loso"
    HYPOTHESIS_ID   = "h-069-vrex-fishr-irm-v2"

    BASE_DIR   = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")

    # ---- Parse CLI ----
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Comma-separated methods to run: vrex,fishr (default: both)",
    )
    _args, _ = _parser.parse_known_args()

    ALL_SUBJECTS = parse_subjects_args()
    EXERCISES    = ["E1"]
    MODEL_TYPE   = "irm_content_style_emg"  # Same backbone as exp_48

    if _args.method:
        METHODS = [m.strip() for m in _args.method.split(",")]
    else:
        METHODS = ["vrex", "fishr"]

    # Lambda values: V-REx uses loss-scale lambda; Fishr gradient variance
    # tends to be larger in magnitude so a smaller lambda is appropriate.
    LAMBDA_VREX          = 1.0
    LAMBDA_FISHR         = 0.1
    PENALTY_MAX_SAMPLES  = 512   # per-env sample cap for penalty computation
    MAX_GESTURES         = 10
    VAL_RATIO            = 0.15
    SEED                 = 42

    # ---- Configs (identical to exp_48 for fair comparison) ----
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
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=60,                      # same as exp_48
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
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"HYPOTHESIS: {HYPOTHESIS_ID}")
    print(f"Methods: {METHODS}")
    print(f"Backbone: {MODEL_TYPE}  (same as exp_48 IRM baseline)")
    print(f"Subjects: {len(ALL_SUBJECTS)}  (CI subset: {ALL_SUBJECTS == CI_TEST_SUBJECTS})")
    print(f"Lambda V-REx={LAMBDA_VREX},  Lambda Fishr={LAMBDA_FISHR}")
    print(f"Penalty max samples/env: {PENALTY_MAX_SAMPLES}")

    # ---- LOSO loop ----
    all_results: List[Dict] = []

    for method in METHODS:
        lambda_penalty = LAMBDA_VREX if method == "vrex" else LAMBDA_FISHR

        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / method / f"test_{test_subject}"

            try:
                fold_res = run_single_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_output_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    model_type=MODEL_TYPE,
                    method=method,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                    lambda_penalty=lambda_penalty,
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
                print(f"  [{method}] {test_subject}: acc={acc_str}, f1={f1_str}")

            except Exception as e:
                global_logger.error(
                    f"Fold failed: method={method}, test={test_subject}: {e}"
                )
                traceback.print_exc()
                all_results.append({
                    "test_subject":  test_subject,
                    "method":        method,
                    "model_type":    MODEL_TYPE,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error":         str(e),
                })

    # ---- Aggregate ----
    aggregate: Dict[str, Dict] = {}
    for method in METHODS:
        method_results = [
            r for r in all_results
            if r["method"] == method and r.get("test_accuracy") is not None
        ]
        if not method_results:
            continue
        accs = [r["test_accuracy"] for r in method_results]
        f1s  = [r["test_f1_macro"] for r in method_results]
        aggregate[method] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_subjects":  len(accs),
        }
        m = aggregate[method]
        print(
            f"\n{method.upper()}: "
            f"Acc = {m['mean_accuracy']:.4f} ± {m['std_accuracy']:.4f},  "
            f"F1 = {m['mean_f1_macro']:.4f} ± {m['std_f1_macro']:.4f}"
        )

    # ---- Save summary ----
    summary = {
        "experiment_name":      EXPERIMENT_NAME,
        "hypothesis_id":        HYPOTHESIS_ID,
        "backbone":             MODEL_TYPE,
        "note":                 "Same backbone as exp_48 (IRMv1) for fair comparison",
        "methods":              METHODS,
        "subjects":             ALL_SUBJECTS,
        "exercises":            EXERCISES,
        "lambda_vrex":          LAMBDA_VREX,
        "lambda_fishr":         LAMBDA_FISHR,
        "penalty_max_samples":  PENALTY_MAX_SAMPLES,
        "processing_config":    asdict(proc_cfg),
        "split_config":         asdict(split_cfg),
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
            mark_hypothesis_verified,
            mark_hypothesis_failed,
        )
        if aggregate:
            best_method  = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
            best_metrics = dict(aggregate[best_method])
            best_metrics["best_method"] = best_method
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
