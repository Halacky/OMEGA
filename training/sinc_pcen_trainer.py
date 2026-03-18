"""
Trainer for SincPCENCNNGRU (Experiment 61).

Extends WindowClassifierTrainer with a custom fit() that:
  1. Converts splits (Dict[str, Dict[int, np.ndarray]]) → (N, C, T) arrays.
  2. Computes per-channel mean/std from TRAINING DATA ONLY (LOSO-clean).
  3. Trains SincPCENCNNGRU end-to-end (frontend + encoder + classifier).
  4. Saves model checkpoint and training history.

evaluate_numpy() applies the same preprocessing (transpose + standardization
from training stats) and runs inference with the frozen model.

LOSO integrity checklist:
  ✓ Channel mean/std computed from X_train only, never from X_val or X_test
  ✓ Model trained only on training-subject windows
  ✓ No test-time adaptation (model.eval() → no BatchNorm updates)
  ✓ PCEN EMA smoother re-initialized per window (no cross-subject state)
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
from models.sinc_pcen_cnn_gru import SincPCENCNNGRU


class SincPCENTrainer(WindowClassifierTrainer):
    """
    Trainer for the SincNet-PCEN frontend + CNN-GRU-Attention model.

    All normalization statistics come from training data exclusively.
    The frontend (SincFilterbank + PCENLayer) and encoder are trained jointly
    on the pooled training subjects in each LOSO fold.  Test subject sees
    frozen model parameters — no adaptation.

    Args:
        train_cfg:          TrainingConfig dataclass
        logger:             Python logger
        output_dir:         directory to save checkpoints and logs
        visualizer:         optional Visualizer for curves / confusion matrices
        sample_rate:        EMG sampling rate in Hz (must match ProcessingConfig)
        num_sinc_filters:   K — number of learnable bandpass filters
        sinc_kernel_size:   odd length of sinc impulse response kernel
        min_freq:           minimum cutoff frequency in Hz
        max_freq:           maximum cutoff frequency in Hz
        pcen_ema_length:    length of the truncated PCEN EMA convolution kernel
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        sample_rate: int = 2000,
        num_sinc_filters: int = 32,
        sinc_kernel_size: int = 51,
        min_freq: float = 5.0,
        max_freq: float = 500.0,
        pcen_ema_length: int = 128,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.sample_rate = sample_rate
        self.num_sinc_filters = num_sinc_filters
        self.sinc_kernel_size = sinc_kernel_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.pcen_ema_length = pcen_ema_length

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, splits: Dict) -> Dict:
        """
        Train SincPCENCNNGRU on the given LOSO training split.

        Args:
            splits: {"train": Dict[int, np.ndarray],
                     "val":   Dict[int, np.ndarray],
                     "test":  Dict[int, np.ndarray]}
                    Arrays are (N_gesture, T, C) — (N, time, channels).

        Returns:
            results dict with val/test metrics from fit-time evaluation.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Splits → flat (N, T, C) arrays ──────────────────────────
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T) ─────────────────────────
        # _prepare_splits_arrays returns (N, T, C); the model expects (N, C, T).
        # We detect the format by checking which dim is larger (T >> C for EMG).
        if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
            X_train = X_train.transpose(0, 2, 1)
            if X_val.ndim == 3 and len(X_val) > 0:
                X_val = X_val.transpose(0, 2, 1)
            if X_test.ndim == 3 and len(X_test) > 0:
                X_test = X_test.transpose(0, 2, 1)
            self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 3. Channel standardization (training data ONLY) ─────────────
        # _compute_channel_standardization expects (N, C, T) → we already transposed.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )
        self.logger.info("Per-channel standardization applied (training stats only).")

        # ── 4. Build model ───────────────────────────────────────────────
        model = SincPCENCNNGRU(
            in_channels=in_channels,
            num_classes=num_classes,
            num_sinc_filters=self.num_sinc_filters,
            sinc_kernel_size=self.sinc_kernel_size,
            sample_rate=self.sample_rate,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            pcen_ema_length=self.pcen_ema_length,
        ).to(self.cfg.device)

        total_params   = sum(p.numel() for p in model.parameters())
        frontend_params = (
            sum(p.numel() for p in model.sinc.parameters()) +
            sum(p.numel() for p in model.pcen.parameters())
        )
        self.logger.info(
            f"SincPCENCNNGRU: in_ch={in_channels}, classes={num_classes}, "
            f"K={self.num_sinc_filters}, total_params={total_params:,}, "
            f"frontend_params={frontend_params:,} "
            f"({100 * frontend_params / total_params:.1f}%)"
        )

        # ── 5. Datasets & DataLoaders ────────────────────────────────────
        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test  = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

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
        # ReduceLROnPlateau: verbose removed in PyTorch ≥ 2.4
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 8. Training loop ─────────────────────────────────────────────
        history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        no_improve = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
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

                bs = xb.size(0)
                ep_loss    += loss.item() * bs
                ep_correct += (logits.argmax(1) == yb).sum().item()
                ep_total   += bs

            train_loss = ep_loss    / max(1, ep_total)
            train_acc  = ep_correct / max(1, ep_total)

            # ── validation ──────────────────────────────────────────────
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
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
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

        # ── 10. Save training history ─────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── 11. In-fold evaluation on val / internal test split ───────────
        results = {"class_ids": class_ids, "class_names": class_names}

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
            preds_arr = np.concatenate(all_logits, axis=0)
            y_true    = np.concatenate(all_y, axis=0)
            y_pred    = preds_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro")
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm  = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True, filename=f"cm_{split_name}.png"
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report":   rep,
                "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── 12. Save model checkpoint ─────────────────────────────────────
        torch.save(
            {
                "state_dict":      model.state_dict(),
                "in_channels":     in_channels,
                "num_classes":     num_classes,
                "class_ids":       class_ids,
                "mean":            mean_c,
                "std":             std_c,
                "window_size":     window_size,
                "sample_rate":     self.sample_rate,
                "num_sinc_filters": self.num_sinc_filters,
                "sinc_kernel_size": self.sinc_kernel_size,
                "min_freq":        self.min_freq,
                "max_freq":        self.max_freq,
                "pcen_ema_length": self.pcen_ema_length,
                "training_config": asdict(self.cfg),
            },
            self.output_dir / "sinc_pcen_cnn_gru.pt",
        )
        self.logger.info(f"Model saved: {self.output_dir / 'sinc_pcen_cnn_gru.pt'}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ------------------------------------------------------------------
    # evaluate_numpy — called by the experiment for cross-subject test
    # ------------------------------------------------------------------

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate the trained model on arbitrary (X, y) numpy arrays.

        Applies the SAME preprocessing as fit() (transpose + training-data
        standardization) so that the test-subject data is treated identically
        to what the model saw during training — without using test statistics.

        Args:
            X:          (N, T, C) or (N, C, T) raw EMG windows
            y:          (N,) class indices matching class_ids ordering from fit()
            split_name: prefix for saved confusion matrix image
            visualize:  whether to save confusion matrix plot

        Returns:
            dict with "accuracy", "f1_macro", "report", "confusion_matrix", "logits"
        """
        assert self.model is not None, "Model has not been trained yet."
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None and self.class_names is not None

        X_in = X.copy().astype(np.float32)

        # Transpose (N, T, C) → (N, C, T) if needed
        if X_in.ndim == 3 and X_in.shape[1] > X_in.shape[2]:
            X_in = X_in.transpose(0, 2, 1)

        # Apply training-data standardization (no test statistics used)
        X_in = self._apply_standardization(X_in, self.mean_c, self.std_c)

        ds = WindowDataset(X_in, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

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
        cm     = confusion_matrix(y_true, y_pred, labels=np.arange(len(self.class_ids)))

        if visualize and self.visualizer is not None:
            labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, labels, normalize=True, filename=f"cm_{split_name}.png"
            )

        return {
            "accuracy":         float(acc),
            "f1_macro":         float(f1_mac),
            "report":           rep,
            "confusion_matrix": cm.tolist(),
            "logits":           logits,
        }
