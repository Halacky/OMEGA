"""
Trainer for SubcenterArcFaceEMG (Experiment 79).

Extends WindowClassifierTrainer with a custom fit() that:
  1. Converts splits (Dict[str, Dict[int, np.ndarray]]) → flat (N, T, C) arrays.
  2. Transposes to (N, C, T) for PyTorch Conv1d format.
  3. Computes per-channel mean/std from TRAINING DATA ONLY (LOSO-clean).
  4. Trains SubcenterArcFaceEMG end-to-end:
       - Training forward: model(x, labels=y) — ArcFace margin applied to
         the max-similarity sub-center of the target class.
       - Validation forward: model(x) — no margin, pure max cosine similarity.
  5. Early stopping on validation cross-entropy loss.
  6. Saves checkpoint and training history.

evaluate_numpy() applies the same preprocessing (transpose + training-stats
standardisation) and runs inference with the frozen model in eval() mode —
no BatchNorm updates, no sub-center adaptation.

LOSO data-leakage checklist
────────────────────────────
  ✓ Channel mean/std computed from X_train ONLY — never from val or test data.
  ✓ Val and test receive the same training statistics via _apply_standardization.
  ✓ Test subject data flows in only through evaluate_numpy() — never touches fit().
  ✓ model.eval() at inference: BatchNorm uses training running statistics,
    no gradient updates from test-subject batches.
  ✓ No per-subject centering, no sub-center adaptation at test time.
  ✓ ArcFace labels passed during training are class indices derived from
    the TRAINING split — no test-subject label information used.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from config.base import TrainingConfig
from models.subcenter_arcface_emg import SubcenterArcFaceEMG
from training.trainer import WindowClassifierTrainer
from training.datasets import WindowDataset
from utils.logging import get_worker_init_fn, seed_everything


class SubcenterArcFaceTrainer(WindowClassifierTrainer):
    """
    Trainer for SubcenterArcFaceEMG.

    Two-phase forward during training:
      Phase A (gradient) : model(x, labels=y) — ArcFace margin active.
      Phase B (val/test) : model(x)           — no margin (eval mode).

    All normalisation statistics are derived from training windows only.
    The model is evaluated on the held-out test subject using frozen parameters
    (model.eval()) — no adaptation of any kind.

    Args:
        train_cfg     : TrainingConfig dataclass.
        logger        : Python logger instance.
        output_dir    : Directory for checkpoints, logs, and CM plots.
        visualizer    : Optional Visualizer for training curves / CM plots.
        channels      : C — ECAPA internal feature width (default 128).
        scale         : Res2Net scale / sub-group count (default 4).
        embedding_dim : Pre-head embedding dimension E (default 128).
        dilations     : SE-Res2Net block dilations (default [2, 3, 4]).
        se_reduction  : SE bottleneck reduction factor (default 8).
        K             : Sub-centers per class (default 3).
        margin        : ArcFace angular margin in radians (default 0.35).
        arc_scale     : ArcFace logit temperature (default 32.0).
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        dilations: Optional[List[int]] = None,
        se_reduction: int = 8,
        K: int = 3,
        margin: float = 0.35,
        arc_scale: float = 32.0,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.channels      = channels
        self.scale         = scale
        self.embedding_dim = embedding_dim
        self.dilations     = dilations if dilations is not None else [2, 3, 4]
        self.se_reduction  = se_reduction
        self.K             = K
        self.margin        = margin
        self.arc_scale     = arc_scale

    # ─────────────────────────────── fit ─────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train SubcenterArcFaceEMG on the LOSO training split.

        Args:
            splits: {"train": Dict[int, np.ndarray],
                     "val":   Dict[int, np.ndarray],
                     "test":  Dict[int, np.ndarray]}
                    Arrays have shape (N, T, C).

        Returns:
            Results dict with class_ids and in-fold metrics.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Splits → flat (N, T, C) arrays ────────────────────────────
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T) ──────────────────────────
        # _prepare_splits_arrays returns (N, T, C).
        # SubcenterArcFaceEMG (via ECAPA backbone) expects (B, C, T) — Conv1d.
        # Heuristic: for EMG data, T (time steps) >> C (channels) always.
        if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
            X_train = X_train.transpose(0, 2, 1)
            if X_val.ndim == 3 and len(X_val) > 0:
                X_val = X_val.transpose(0, 2, 1)
            if X_test.ndim == 3 and len(X_test) > 0:
                X_test = X_test.transpose(0, 2, 1)
            self.logger.info(
                f"Transposed windows to (N, C, T): X_train={X_train.shape}"
            )

        in_channels = X_train.shape[1]   # C (EMG channels)
        window_size = X_train.shape[2]   # T (time steps)
        num_classes = len(class_ids)

        # ── 3. Per-channel standardisation — TRAINING DATA ONLY ──────────
        # LOSO integrity: mean_c and std_c are derived exclusively from
        # training windows.  The same frozen statistics are then applied to
        # val and test.  Test-subject signal statistics never influence the
        # normalisation pipeline.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val  = self._apply_standardization(X_val,  mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)

        # Persist for offline reproducibility
        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c,
            std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )
        self.logger.info(
            "Per-channel standardisation applied (training statistics only)."
        )

        # ── 4. Build model ────────────────────────────────────────────────
        model = SubcenterArcFaceEMG(
            in_channels=in_channels,
            num_classes=num_classes,
            K=self.K,
            channels=self.channels,
            scale=self.scale,
            embedding_dim=self.embedding_dim,
            dilations=self.dilations,
            dropout=self.cfg.dropout,
            se_reduction=self.se_reduction,
            margin=self.margin,
            arc_scale=self.arc_scale,
        ).to(self.cfg.device)

        total_params = model.count_parameters()
        self.logger.info(
            f"SubcenterArcFaceEMG: in_ch={in_channels}, classes={num_classes}, "
            f"K={self.K}, C={self.channels}, embed={self.embedding_dim}, "
            f"dilations={self.dilations}, margin={self.margin:.2f}, "
            f"arc_scale={self.arc_scale:.1f} | total_params={total_params:,}"
        )

        # ── 5. Datasets & DataLoaders ─────────────────────────────────────
        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val,   y_val)   if len(X_val)  > 0 else None
        ds_test  = WindowDataset(X_test,  y_test)  if len(X_test) > 0 else None

        worker_init = get_worker_init_fn(self.cfg.seed)
        g = torch.Generator().manual_seed(self.cfg.seed)

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
            generator=g,
        )
        dl_val = (
            DataLoader(
                ds_val,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
            if ds_val else None
        )
        dl_test = (
            DataLoader(
                ds_test,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
            if ds_test else None
        )

        # ── 6. Loss function ──────────────────────────────────────────────
        # Standard cross-entropy over ArcFace logits.
        # Class weights address any per-gesture imbalance in the training pool.
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = counts.sum() / (counts + 1e-8)
            cw /= cw.mean()
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(self.cfg.device)
            )
            self.logger.info(f"Class weights: {cw.round(3).tolist()}")
        else:
            criterion = nn.CrossEntropyLoss()

        # ── 7. Optimizer + LR scheduler ───────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # ReduceLROnPlateau: `verbose` parameter removed in PyTorch ≥ 2.4
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 8. Training loop ──────────────────────────────────────────────
        history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }
        best_val_loss: float = float("inf")
        best_state: Optional[Dict] = None
        no_improve: int = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):

            # ── training pass ─────────────────────────────────────────────
            # model.train() activates ArcFace margin in the head's forward().
            # Labels are passed to model() so the margin can be applied to the
            # correct target class.  This does NOT leak test-subject information:
            # all labels here are y_train, derived from training subjects only.
            model.train()
            ep_loss, ep_correct, ep_total = 0.0, 0, 0

            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                # ArcFace: pass labels so margin is applied to target class
                logits = model(xb, labels=yb)
                loss   = criterion(logits, yb)
                loss.backward()
                # Gradient clipping: stabilises training with SE attention + ArcFace
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs          = xb.size(0)
                ep_loss    += loss.item() * bs
                ep_correct += (logits.argmax(1) == yb).sum().item()
                ep_total   += bs

            train_loss = ep_loss    / max(1, ep_total)
            train_acc  = ep_correct / max(1, ep_total)

            # ── validation pass ───────────────────────────────────────────
            # model.eval() deactivates ArcFace margin → pure max cosine sim.
            # No BatchNorm updates from val batches.
            if dl_val is not None:
                model.eval()
                vl_sum, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits  = model(xb)       # no margin
                        vl_sum += criterion(logits, yb).item() * yb.size(0)
                        vc     += (logits.argmax(1) == yb).sum().item()
                        vt     += yb.size(0)
                val_loss = vl_sum / max(1, vt)
                val_acc  = vc    / max(1, vt)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

            # ── early stopping ────────────────────────────────────────────
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}.")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
            self.logger.info(
                f"Restored best checkpoint (val_loss={best_val_loss:.4f})."
            )

        # ── 9. Store trainer state (needed by evaluate_numpy) ─────────────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 10. Training history ──────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png"
            )

        # ── 11. In-fold evaluation (val / internal test splits) ───────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dl, split_name: str):
            if dl is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(device)
                    all_logits.append(model(xb).cpu().numpy())
                    all_y.append(yb.numpy())
            y_true = np.concatenate(all_y,      axis=0)
            y_pred = np.concatenate(all_logits, axis=0).argmax(axis=1)
            acc = float(accuracy_score(y_true, y_pred))
            f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            rep = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                labels_str = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels_str, normalize=True,
                    filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy":         acc,
                "f1_macro":         f1,
                "report":           rep,
                "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "internal_test")

        # ── 12. Checkpoint ────────────────────────────────────────────────
        torch.save(
            {
                "state_dict":  model.state_dict(),
                "in_channels": in_channels,
                "num_classes": num_classes,
                "class_ids":   class_ids,
                "mean":        mean_c,
                "std":         std_c,
                "window_size": window_size,
                "model_config": {
                    "K":             self.K,
                    "channels":      self.channels,
                    "scale":         self.scale,
                    "embedding_dim": self.embedding_dim,
                    "dilations":     self.dilations,
                    "se_reduction":  self.se_reduction,
                    "margin":        self.margin,
                    "arc_scale":     self.arc_scale,
                    "dropout":       self.cfg.dropout,
                },
                "training_config": asdict(self.cfg),
            },
            self.output_dir / "subcenter_arcface_emg.pt",
        )
        self.logger.info(
            f"Checkpoint saved → {self.output_dir / 'subcenter_arcface_emg.pt'}"
        )

        with open(self.output_dir / "classification_results.json", "w") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False)

        return results

    # ──────────────────────────── evaluate_numpy ──────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate the trained model on arbitrary (X, y) numpy arrays.

        Applies EXACTLY the same preprocessing as fit():
          1. Transpose (N, T, C) → (N, C, T) if needed (same heuristic).
          2. Channel standardisation using TRAINING statistics (mean_c, std_c).

        No test-subject statistics are used — the model and normalisation
        parameters are fully determined by the training data.
        model.eval() ensures BatchNorm uses training running stats; no updates.

        Args:
            X          : Raw EMG windows, (N, T, C) or (N, C, T).
            y          : Integer class indices matching class_ids from fit().
            split_name : Prefix for saved confusion matrix image.
            visualize  : Whether to save a confusion matrix plot.

        Returns:
            dict with keys "accuracy", "f1_macro", "report",
            "confusion_matrix", "logits".
        """
        assert self.model       is not None, "Call fit() before evaluate_numpy()."
        assert self.mean_c      is not None and self.std_c is not None
        assert self.class_ids   is not None
        assert self.class_names is not None

        X_in = X.copy().astype(np.float32)

        # Transpose (N, T, C) → (N, C, T) using the same heuristic as fit()
        if X_in.ndim == 3 and X_in.shape[1] > X_in.shape[2]:
            X_in = X_in.transpose(0, 2, 1)

        # Apply training-data standardisation (no test-subject statistics used)
        X_in = self._apply_standardization(X_in, self.mean_c, self.std_c)

        ds = WindowDataset(X_in, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        # model.eval(): BatchNorm frozen to training running stats; no updates.
        # ArcFace head: no margin (labels=None path / eval mode).
        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                all_logits.append(self.model(xb).cpu().numpy())   # no labels
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y,      axis=0)
        y_pred = logits.argmax(axis=1)

        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        rep = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        num_classes = len(self.class_ids)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        if visualize and self.visualizer is not None:
            labels_str = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, labels_str, normalize=True,
                filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy":         acc,
            "f1_macro":         f1,
            "report":           rep,
            "confusion_matrix": cm.tolist(),
            "logits":           logits,
        }
