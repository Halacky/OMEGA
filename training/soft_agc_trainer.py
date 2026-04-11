"""
Trainer for SoftAGCCNNGRU (Experiment 76).

Extends WindowClassifierTrainer with a custom fit() that:
  1. Converts splits (Dict[str, Dict[int, np.ndarray]]) → flat (N, C, T) arrays.
  2. Computes per-channel mean/std from TRAINING DATA ONLY (strict LOSO).
  3. Trains SoftAGCCNNGRU end-to-end (frontend + encoder + classifier).
  4. Saves model checkpoint, normalization statistics, and training history.

evaluate_numpy() applies the SAME preprocessing (transpose + training-statistics
standardization) as fit() so test-subject data is handled identically to training
data — without using any test-subject statistics.

LOSO integrity checklist:
  ✓ Channel mean/std computed from X_train only, NEVER from X_val or X_test.
  ✓ Model trained ONLY on training-subject windows.
  ✓ No test-time adaptation: model.eval() → BatchNorm uses frozen running stats.
  ✓ Frontend parameters (LogAffine: scale/bias; SoftAGC: alpha_raw/log_s) are
    updated only via gradients from training data.
  ✓ RMSWindowLayer has NO learnable parameters → zero leakage possible.
  ✓ SoftAGCLayer EMA is stateless (recomputed per forward call, no persistent state).
  ✓ Early stopping uses val loss from validation windows carved from train subjects.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from training.trainer import (
    WindowClassifierTrainer, WindowDataset, get_worker_init_fn, seed_everything,
)
from models.soft_agc_cnn_gru import SoftAGCCNNGRU


class SoftAGCTrainer(WindowClassifierTrainer):
    """
    Trainer for the Soft AGC Frontend + CNN-GRU-Attention model (exp_76).

    Supports three frontend variants passed via `frontend_type`:
      - "log_affine": log-compression + learnable per-channel affine
      - "rms_window": causal local-RMS normalization (fixed, no parameters)
      - "soft_agc":   EMA-based AGC with bounded exponent alpha ∈ (0, 0.5]

    All normalization statistics (channel mean/std) come exclusively from
    training-subject data.  The test subject sees the frozen model trained on
    other subjects — no adaptation of any kind.

    Args:
        train_cfg:       TrainingConfig dataclass
        logger:          Python logger
        output_dir:      directory for checkpoints and logs
        visualizer:      optional Visualizer for curves / confusion matrices
        frontend_type:   one of {"log_affine", "rms_window", "soft_agc"}
        rms_window_size: causal window length (samples) for RMSWindowLayer
        agc_ema_length:  EMA kernel length for SoftAGCLayer
        agc_delta:       FIXED stabilizer δ for SoftAGCLayer (not learned)
        cnn_channels:    CNN channel widths for the encoder
        gru_hidden:      GRU hidden size
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        frontend_type: str = "soft_agc",
        rms_window_size: int = 50,
        agc_ema_length: int = 100,
        agc_delta: float = 0.1,
        cnn_channels: Optional[List[int]] = None,
        gru_hidden: int = 128,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.frontend_type   = frontend_type
        self.rms_window_size = rms_window_size
        self.agc_ema_length  = agc_ema_length
        self.agc_delta       = agc_delta
        self.cnn_channels    = cnn_channels or [64, 128, 256]
        self.gru_hidden      = gru_hidden

    # ──────────────────────────────── fit ───────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train SoftAGCCNNGRU on the given LOSO training split.

        Args:
            splits: {
                "train": Dict[int, np.ndarray],   gesture_id → (N, T, C) windows
                "val":   Dict[int, np.ndarray],
                "test":  Dict[int, np.ndarray],   (only used for in-fold diagnostics)
            }

        Returns:
            dict with val/test metrics from fit-time evaluation.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Split dicts → flat numpy arrays (N, T, C) ────────────────
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T) ─────────────────────────
        # _prepare_splits_arrays returns (N, T, C); the model expects (N, C, T).
        # Detect format: T (time, hundreds) is always >> C (channels, usually 8).
        if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
            X_train = X_train.transpose(0, 2, 1)
            if X_val.ndim == 3 and len(X_val) > 0:
                X_val = X_val.transpose(0, 2, 1)
            if X_test.ndim == 3 and len(X_test) > 0:
                X_test = X_test.transpose(0, 2, 1)
            self.logger.info(
                f"Transposed to (N, C, T): X_train={X_train.shape}"
            )

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 3. Per-channel standardization — TRAINING DATA ONLY ─────────
        # This is the only normalization using global statistics; they must
        # NEVER be computed from val or test windows.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c,
            std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )
        self.logger.info(
            f"Per-channel standardization applied (train statistics only). "
            f"mean shape={mean_c.shape}"
        )

        # ── 4. Build model ───────────────────────────────────────────────
        model = SoftAGCCNNGRU(
            in_channels=in_channels,
            num_classes=num_classes,
            frontend_type=self.frontend_type,
            rms_window_size=self.rms_window_size,
            agc_ema_length=self.agc_ema_length,
            agc_delta=self.agc_delta,
            cnn_channels=self.cnn_channels,
            gru_hidden=self.gru_hidden,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params    = sum(p.numel() for p in model.parameters())
        frontend_params = model.count_frontend_params()
        self.logger.info(
            f"SoftAGCCNNGRU [{self.frontend_type}]: "
            f"in_ch={in_channels}, classes={num_classes}, "
            f"total_params={total_params:,}, "
            f"frontend_params={frontend_params:,} "
            f"({100 * frontend_params / max(1, total_params):.1f}%)"
        )

        # ── 5. Datasets & DataLoaders ────────────────────────────────────
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
        dl_val = DataLoader(
            ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        ) if ds_test else None

        # ── 6. Loss function ─────────────────────────────────────────────
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

        # ── 7. Optimizer + scheduler ─────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # NOTE: verbose= removed from ReduceLROnPlateau in PyTorch ≥ 2.4
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 8. Training loop ─────────────────────────────────────────────
        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        no_improve = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            # ── train epoch ──────────────────────────────────────────────
            model.train()
            ep_loss, ep_correct, ep_total = 0.0, 0, 0

            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs          = xb.size(0)
                ep_loss    += loss.item() * bs
                ep_correct += (logits.argmax(1) == yb).sum().item()
                ep_total   += bs

            train_loss = ep_loss    / max(1, ep_total)
            train_acc  = ep_correct / max(1, ep_total)

            # ── validation epoch ─────────────────────────────────────────
            if dl_val is not None:
                model.eval()
                vl_sum, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
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
                f"[{self.frontend_type}] Epoch {epoch:02d}/{self.cfg.epochs} "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

            # ── early stopping ───────────────────────────────────────────
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}.")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── 9. Store trainer state (needed by evaluate_numpy) ────────────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 10. Save training history and curves ─────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename=f"training_curves_{self.frontend_type}.png"
            )

        # ── 11. In-fold evaluation (val + internal test) ─────────────────
        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dl, split_name: str) -> Optional[Dict]:
            if dl is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(device)
                    all_logits.append(model(xb).cpu().numpy())
                    all_y.append(yb.numpy())
            preds = np.concatenate(all_logits, axis=0)
            y_true = np.concatenate(all_y, axis=0)
            y_pred = preds.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro")
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm  = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True,
                    filename=f"cm_{split_name}_{self.frontend_type}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report":   rep,
                "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "internal_test")

        # ── 12. Save model checkpoint ─────────────────────────────────────
        ckpt_path = self.output_dir / f"soft_agc_cnn_gru_{self.frontend_type}.pt"
        torch.save(
            {
                "state_dict":      model.state_dict(),
                "frontend_type":   self.frontend_type,
                "in_channels":     in_channels,
                "num_classes":     num_classes,
                "class_ids":       class_ids,
                "mean":            mean_c,
                "std":             std_c,
                "window_size":     window_size,
                "rms_window_size": self.rms_window_size,
                "agc_ema_length":  self.agc_ema_length,
                "agc_delta":       self.agc_delta,
                "cnn_channels":    self.cnn_channels,
                "gru_hidden":      self.gru_hidden,
                "training_config": asdict(self.cfg),
            },
            ckpt_path,
        )
        self.logger.info(f"Model saved: {ckpt_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ──────────────────────────── evaluate_numpy ─────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate the trained model on arbitrary (X, y) numpy arrays.

        Applies the SAME preprocessing as fit():
          1. Transpose (N, T, C) → (N, C, T) if needed.
          2. Standardize with TRAINING-DATA mean/std (no test statistics).

        This is the function called by the experiment for cross-subject testing:
        the test-subject windows are normalized using training-subject statistics,
        maintaining strict LOSO integrity.

        Args:
            X:          (N, T, C) or (N, C, T) raw EMG windows
            y:          (N,) class indices (must match class_ids ordering from fit())
            split_name: prefix for saved confusion matrix image filename
            visualize:  whether to save a confusion matrix plot

        Returns:
            dict with keys "accuracy", "f1_macro", "report", "confusion_matrix", "logits"
        """
        assert self.model is not None, "Model has not been trained yet (call fit() first)."
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None and self.class_names is not None

        X_in = X.copy().astype(np.float32)

        # Transpose (N, T, C) → (N, C, T) if needed (same heuristic as fit)
        if X_in.ndim == 3 and X_in.shape[1] > X_in.shape[2]:
            X_in = X_in.transpose(0, 2, 1)

        # Apply TRAINING standardization — no test statistics used
        X_in = self._apply_standardization(X_in, self.mean_c, self.std_c)

        ds = WindowDataset(X_in, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        # Frozen inference — no parameter updates, no BN adaptation
        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                all_logits.append(self.model(xb).cpu().numpy())
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y, axis=0)
        y_pred = logits.argmax(axis=1)

        acc    = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average="macro")
        rep    = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm     = confusion_matrix(
            y_true, y_pred, labels=np.arange(len(self.class_ids))
        )

        if visualize and self.visualizer is not None:
            labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, labels, normalize=True,
                filename=f"cm_{split_name}_{self.frontend_type}.png",
            )

        return {
            "accuracy":         float(acc),
            "f1_macro":         float(f1_mac),
            "report":           rep,
            "confusion_matrix": cm.tolist(),
            "logits":           logits,
        }
