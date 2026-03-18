"""
Trainer for FIRDeconvCNNGRU (Experiment 65).

Extends WindowClassifierTrainer with a custom fit() that:

    1. Converts splits (Dict[str, Dict[int, np.ndarray]]) → (N, C, T) arrays
       using the parent's _prepare_splits_arrays() helper.
    2. Computes per-channel mean/std EXCLUSIVELY from training data.
    3. Trains FIRDeconvCNNGRU end-to-end (FIR frontend + CNN-BiGRU-Attention).
       The training loss is:
           L = CrossEntropy(logits, y) + model.regularization_loss(λ_l2, λ_smooth)
    4. Saves checkpoint and training history.

evaluate_numpy() applies the identical preprocessing (transpose + training-data
standardization) and runs inference with the frozen model.  No test-time
adaptation of any kind.

LOSO integrity checklist:
    ✓  Channel mean/std computed from X_train ONLY — never X_val or X_test.
    ✓  FIR filter weights trained ONLY on train-subject pool.
    ✓  λ_l2 / λ_smooth are fixed hyper-parameters, NOT data-estimated.
    ✓  model.eval() at test time disables BN running-stat updates.
    ✓  No per-subject statistics anywhere in the preprocessing pipeline.
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
from models.fir_deconv_cnn_gru import FIRDeconvCNNGRU


class FIRDeconvTrainer(WindowClassifierTrainer):
    """
    Trainer for the FIR Deconvolution Frontend + CNN-BiGRU-Attention model.

    The FIR frontend is a per-channel depthwise 1D FIR filter initialised as
    the identity.  Its weights are learned jointly with the encoder on the
    pooled training-subject data for each LOSO fold.  The test subject sees
    a completely frozen model — no adaptation whatsoever.

    Args:
        train_cfg:     TrainingConfig dataclass.
        logger:        Python logger.
        output_dir:    directory to save checkpoints / logs.
        visualizer:    optional Visualizer for curves / confusion matrices.
        filter_len:    FIR filter length in taps (odd, default 63).
                       63 taps @ 2000 Hz ≈ 31.5 ms — covers one MU action potential.
        cnn_channels:  CNN block output channels (3 blocks by default).
        gru_hidden:    BiGRU hidden units per direction.
        num_heads:     Multi-head attention heads.
        lambda_l2:     L2 regularization weight for FIR taps.
        lambda_smooth: Smoothness regularization weight for FIR taps.
    """

    def __init__(
        self,
        train_cfg,
        logger:        logging.Logger,
        output_dir:    Path,
        visualizer     = None,
        filter_len:    int             = 63,
        cnn_channels:  Tuple[int, ...] = (64, 128, 256),
        gru_hidden:    int             = 128,
        num_heads:     int             = 4,
        lambda_l2:     float           = 1e-3,
        lambda_smooth: float           = 5e-3,
    ) -> None:
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.filter_len    = filter_len
        self.cnn_channels  = cnn_channels
        self.gru_hidden    = gru_hidden
        self.num_heads     = num_heads
        self.lambda_l2     = lambda_l2
        self.lambda_smooth = lambda_smooth

    # ──────────────────────────────────────────────────────────────────────────
    # fit
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train FIRDeconvCNNGRU on the provided LOSO training split.

        Args:
            splits: {
                "train": Dict[int, np.ndarray],   # gesture_id → (N, T, C)
                "val":   Dict[int, np.ndarray],
                "test":  Dict[int, np.ndarray],
            }

        Returns:
            dict with in-fold val / test metrics.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Splits → flat (N, T, C) numpy arrays ──────────────────────
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T) ──────────────────────────
        # _prepare_splits_arrays returns (N, T, C); model expects (N, C, T).
        # Detect by checking which axis is larger (T >> C for EMG).
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

        # ── 3. Per-channel standardization — training data ONLY ───────────
        # LOSO contract: statistics must NEVER be computed from val or test.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean     = mean_c,
            std      = std_c,
            class_ids = np.array(class_ids, dtype=np.int32),
        )
        self.logger.info(
            "Per-channel standardization applied (training statistics only). "
            f"in_ch={in_channels}, T={window_size}, classes={num_classes}"
        )

        # ── 4. Build model ────────────────────────────────────────────────
        model = FIRDeconvCNNGRU(
            in_channels  = in_channels,
            num_classes  = num_classes,
            filter_len   = self.filter_len,
            cnn_channels = self.cnn_channels,
            gru_hidden   = self.gru_hidden,
            num_heads    = self.num_heads,
            dropout      = self.cfg.dropout,
        ).to(self.cfg.device)

        total_params    = sum(p.numel() for p in model.parameters())
        frontend_params = sum(p.numel() for p in model.frontend.parameters())
        self.logger.info(
            f"FIRDeconvCNNGRU built: "
            f"filter_len={self.filter_len}, in_ch={in_channels}, "
            f"classes={num_classes}, "
            f"total_params={total_params:,}, "
            f"frontend_params={frontend_params:,} "
            f"({100.0 * frontend_params / max(1, total_params):.1f}%)"
        )

        # ── 5. DataLoaders ────────────────────────────────────────────────
        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val,   y_val)   if len(X_val)  > 0 else None
        ds_test  = WindowDataset(X_test,  y_test)  if len(X_test) > 0 else None

        worker_init = get_worker_init_fn(self.cfg.seed)
        g = torch.Generator().manual_seed(self.cfg.seed)

        dl_train = DataLoader(
            ds_train,
            batch_size   = self.cfg.batch_size,
            shuffle      = True,
            num_workers  = self.cfg.num_workers,
            pin_memory   = True,
            worker_init_fn = worker_init if self.cfg.num_workers > 0 else None,
            generator    = g,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size  = self.cfg.batch_size,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size  = self.cfg.batch_size,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
        ) if ds_test else None

        # ── 6. Loss function (class-weighted CrossEntropy) ─────────────────
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw     = counts.sum() / (counts + 1e-8)
            cw    /= cw.mean()
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(self.cfg.device)
            )
            self.logger.info(f"Class weights applied: {cw.round(3).tolist()}")
        else:
            criterion = nn.CrossEntropyLoss()

        # ── 7. Optimizer + LR scheduler ───────────────────────────────────
        # weight_decay is applied to ALL parameters (including FIR weights).
        # The regularization_loss() adds ADDITIONAL, explicit L2+smooth terms
        # for the FIR taps so their penalty can be tuned independently.
        optimizer = optim.Adam(
            model.parameters(),
            lr           = self.cfg.learning_rate,
            weight_decay = self.cfg.weight_decay,
        )
        # verbose removed in PyTorch ≥ 2.4
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 8. Training loop ──────────────────────────────────────────────
        history: Dict[str, list] = {
            "train_loss": [], "val_loss":  [],
            "train_acc":  [], "val_acc":   [],
            "fir_reg":    [],  # track FIR regularization separately
        }
        best_val_loss = float("inf")
        best_state:   Optional[Dict] = None
        no_improve    = 0
        device        = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            ep_loss, ep_reg, ep_correct, ep_total = 0.0, 0.0, 0, 0

            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()

                logits = model(xb)
                ce_loss  = criterion(logits, yb)
                reg_loss = model.regularization_loss(
                    self.lambda_l2, self.lambda_smooth
                )
                loss = ce_loss + reg_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs          = xb.size(0)
                ep_loss    += ce_loss.item() * bs
                ep_reg     += reg_loss.item()
                ep_correct += (logits.argmax(1) == yb).sum().item()
                ep_total   += bs

            train_loss = ep_loss    / max(1, ep_total)
            train_acc  = ep_correct / max(1, ep_total)
            # Average regularization over batches (not per-sample)
            num_batches = max(1, len(dl_train))
            train_reg   = ep_reg / num_batches

            # ── validation ────────────────────────────────────────────────
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
            history["fir_reg"].append(train_reg)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"ce_loss={train_loss:.4f}, fir_reg={train_reg:.5f}, "
                f"train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

            # ── early stopping ─────────────────────────────────────────────
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

        # ── 9. Log learned FIR filter properties ─────────────────────────
        with torch.no_grad():
            w = model.frontend.conv.weight.cpu().numpy()   # (C, 1, filter_len)
            w_sq  = w.reshape(in_channels, -1)             # (C, filter_len)
            energy = (w_sq ** 2).sum(axis=1)               # per-channel L2
            center = self.filter_len // 2
            identity_dev = np.abs(w_sq[:, center] - 1.0).mean()
        self.logger.info(
            f"FIR filter stats after training: "
            f"per-channel L2-energy (mean/std)="
            f"{energy.mean():.4f}/{energy.std():.4f}, "
            f"center-tap deviation from 1 (mean)={identity_dev:.4f}"
        )

        # ── 10. Store trainer state (needed by evaluate_numpy) ────────────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 11. Persist training history ──────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png"
            )

        # ── 12. In-fold evaluation on val / internal test split ───────────
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
            logits_np = np.concatenate(all_logits, axis=0)
            y_true    = np.concatenate(all_y, axis=0)
            y_pred    = logits_np.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
            rep = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            cm  = confusion_matrix(
                y_true, y_pred, labels=np.arange(num_classes)
            )
            if self.visualizer is not None:
                labels = [class_names.get(gid, str(gid)) for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True,
                    filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report":   rep,
                "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── 13. Save model checkpoint ─────────────────────────────────────
        torch.save(
            {
                "state_dict":    model.state_dict(),
                "in_channels":   in_channels,
                "num_classes":   num_classes,
                "class_ids":     class_ids,
                "mean":          mean_c,
                "std":           std_c,
                "window_size":   window_size,
                "filter_len":    self.filter_len,
                "cnn_channels":  list(self.cnn_channels),
                "gru_hidden":    self.gru_hidden,
                "num_heads":     self.num_heads,
                "lambda_l2":     self.lambda_l2,
                "lambda_smooth": self.lambda_smooth,
                "training_config": asdict(self.cfg),
            },
            self.output_dir / "fir_deconv_cnn_gru.pt",
        )
        self.logger.info(
            f"Checkpoint saved: {self.output_dir / 'fir_deconv_cnn_gru.pt'}"
        )

        with open(self.output_dir / "classification_results.json", "w") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # evaluate_numpy — called by the experiment for cross-subject test
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        split_name: str  = "custom",
        visualize:  bool = False,
    ) -> Dict:
        """
        Evaluate the trained model on arbitrary (X, y) numpy arrays.

        Applies EXACTLY the same preprocessing as fit():
            1. Transpose (N, T, C) → (N, C, T) if needed.
            2. Standardize using training-data mean/std (no test statistics).

        Args:
            X:          (N, T, C) or (N, C, T) raw EMG windows.
            y:          (N,) class indices matching class_ids from fit().
            split_name: prefix for saved confusion-matrix image.
            visualize:  whether to save the confusion-matrix plot.

        Returns:
            dict with "accuracy", "f1_macro", "report", "confusion_matrix",
            "logits".
        """
        assert self.model     is not None, "Call fit() before evaluate_numpy()."
        assert self.mean_c    is not None and self.std_c is not None
        assert self.class_ids is not None and self.class_names is not None

        X_in = X.copy().astype(np.float32)

        # Transpose to (N, C, T) if input is (N, T, C)
        if X_in.ndim == 3 and X_in.shape[1] > X_in.shape[2]:
            X_in = X_in.transpose(0, 2, 1)

        # Apply training-data standardization (never test statistics)
        X_in = self._apply_standardization(X_in, self.mean_c, self.std_c)

        ds = WindowDataset(X_in, y)
        dl = DataLoader(
            ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
        )

        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                all_logits.append(self.model(xb).cpu().numpy())
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y,    axis=0)
        y_pred = logits.argmax(axis=1)

        acc    = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rep    = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        cm     = confusion_matrix(
            y_true, y_pred, labels=np.arange(len(self.class_ids))
        )

        if visualize and self.visualizer is not None:
            labels = [
                self.class_names.get(gid, str(gid)) for gid in self.class_ids
            ]
            self.visualizer.plot_confusion_matrix(
                cm, labels, normalize=True,
                filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy":         float(acc),
            "f1_macro":         float(f1_mac),
            "report":           rep,
            "confusion_matrix": cm.tolist(),
            "logits":           logits,
        }
