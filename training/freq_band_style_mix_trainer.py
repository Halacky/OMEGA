"""
Trainer for Frequency-Band Style Mixing EMG model (Experiment 102).

The model (FreqBandStyleMixEMG) applies AdaIN-based style mixing to EMG
frequency bands during training, making the encoder robust to subject-specific
low-frequency statistics without corrupting mid-band gesture information.

Training loss:  cross-entropy on gesture logits only.
No subject-adversarial or MI auxiliary losses are used — the augmentation
itself provides the domain-generalization signal.

LOSO data-leakage audit
-----------------------
✓ subject_labels fed to model.forward() during training are indices into
  train_subjects only.  Test subject has no index and is never in any batch.
✓ Channel standardization (mean_c, std_c) computed from X_train only.
✓ Validation uses data from training subjects; early stopping is driven by
  val_loss (no test subject information).
✓ evaluate_numpy() (inherited from DisentangledTrainer) calls model in eval
  mode where FreqBandStyleMixer is a no-op.  No test-subject statistics are
  ever used for normalization decisions.
✓ No test-time adaptation.
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
from torch.utils.data import DataLoader

from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from models.freq_band_style_mix_emg import FreqBandStyleMixEMG


class FreqBandStyleMixTrainer(DisentangledTrainer):
    """
    Trainer for FreqBandStyleMixEMG.

    Inherits from DisentangledTrainer to reuse:
      * _prepare_splits_arrays()          — flat array extraction from splits dict
      * _build_subject_labels_array()     — subject label alignment with class_ids
      * _compute_channel_standardization()
      * _apply_standardization()
      * evaluate_numpy()                  — test-time inference, no subject labels

    Only fit() is overridden to:
      1. Build FreqBandStyleMixEMG instead of DisentangledCNNGRU.
      2. Pass subject_labels into model.forward() during training batches.
      3. Use a single gesture cross-entropy loss (no auxiliary losses).

    Expects splits to contain:
        "train":               Dict[gesture_id → np.ndarray (N,T,C)]
        "val":                 Dict[gesture_id → np.ndarray (N,T,C)]
        "test":                Dict[gesture_id → np.ndarray (N,T,C)]
        "train_subject_labels": Dict[gesture_id → np.ndarray (N,) int]
        "num_train_subjects":  int
    These are injected by the experiment file that tracks subject provenance
    during split construction.
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # Frequency-band mixing hyperparameters
        low_band: tuple = (20.0, 150.0),
        mid_band: tuple = (150.0, 450.0),
        low_mix_alpha: float = 0.2,
        mid_mix_alpha: float = 0.8,
        # Encoder / classifier dimensions
        classifier_dim: int = 128,
        sampling_rate: int = 2000,
    ):
        """
        Args:
            train_cfg:        TrainingConfig
            logger:           Python logger
            output_dir:       Path for checkpoints and artifacts
            visualizer:       optional Visualizer for curve/CM plots
            low_band:         (f_lo, f_hi) Hz for low-frequency band
            mid_band:         (f_lo, f_hi) Hz for mid-frequency band
            low_mix_alpha:    Beta α for low-band mixing (0.2 → aggressive)
            mid_mix_alpha:    Beta α for mid-band mixing (0.8 → conservative)
            classifier_dim:   hidden dim of the 2-layer gesture head
            sampling_rate:    EMG sampling rate (Hz)
        """
        # Pass dummy disentanglement args to parent — they are not used here
        # because fit() is fully overridden and evaluate_numpy() does not need them.
        super().__init__(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=visualizer,
            content_dim=classifier_dim,   # repurposed as classifier hidden dim
            style_dim=64,                 # unused by this model
            alpha=0.0,                    # no subject loss
            beta=0.0,                     # no MI loss
        )
        self.low_band = low_band
        self.mid_band = mid_band
        self.low_mix_alpha = low_mix_alpha
        self.mid_mix_alpha = mid_mix_alpha
        self.classifier_dim = classifier_dim
        self.sampling_rate = sampling_rate

    # ─────────────────────────── fit ────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train FreqBandStyleMixEMG.

        Steps
        -----
        1. Extract flat train/val/test arrays and class_ids.
        2. Build subject label array aligned with class_ids ordering.
        3. Transpose (N,T,C) → (N,C,T) if needed.
        4. Compute per-channel standardization from X_train only.
        5. Instantiate FreqBandStyleMixEMG.
        6. Train loop: pass (windows, subject_labels) during training,
           evaluate with base (no-mixing) path for val/early-stopping.
        7. Restore best checkpoint; save model and history.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Flat arrays ────────────────────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # ── 2. Subject labels (aligned with class_ids iteration order) ────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "FreqBandStyleMixTrainer requires 'train_subject_labels' in splits. "
                "Use the experiment's _build_splits_with_subject_labels() helper."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        if len(y_subject_train) != len(y_train):
            raise AssertionError(
                f"Subject labels ({len(y_subject_train)}) must match "
                f"gesture labels ({len(y_train)})"
            )
        self.logger.info(
            f"Training: {num_train_subjects} training subjects, "
            f"subject distribution: {np.bincount(y_subject_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) ──────────────────────────
        # Windows from load_multiple_subjects come as (N, T, C).
        # PyTorch Conv1d expects (N, C, T).
        if X_train.ndim == 3:
            N, dim1, dim2 = X_train.shape
            if dim1 > dim2:
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(
                    f"Transposed (N,T,C)→(N,C,T): X_train={X_train.shape}"
                )

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 4. Per-channel standardization (training data only) ───────────
        # LOSO clean: test-subject data never influences mean_c / std_c.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation (train statistics only).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Instantiate model ──────────────────────────────────────────
        model = FreqBandStyleMixEMG(
            in_channels=in_channels,
            num_gestures=num_classes,
            sampling_rate=self.sampling_rate,
            classifier_dim=self.classifier_dim,
            low_band=self.low_band,
            mid_band=self.mid_band,
            low_mix_alpha=self.low_mix_alpha,
            mid_mix_alpha=self.mid_mix_alpha,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"FreqBandStyleMixEMG: in_ch={in_channels}, gestures={num_classes}, "
            f"low_band={self.low_band} Hz (alpha={self.low_mix_alpha}), "
            f"mid_band={self.mid_band} Hz (alpha={self.mid_mix_alpha}), "
            f"params={total_params:,}"
        )

        # ── 6. Datasets and DataLoaders ───────────────────────────────────
        # Training set wraps both windows (X) and subject labels (y_subject)
        # so we can pass subject_labels into model.forward() for style mixing.
        # Validation and test sets do NOT need subject labels.
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
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
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
        ) if ds_test else None

        # ── 7. Loss function ──────────────────────────────────────────────
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        # ── 8. Optimizer + LR scheduler ──────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # Note: verbose=True removed in PyTorch 2.4+; omit here.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 9. Training loop ──────────────────────────────────────────────
        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
        }
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        no_improve = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):

            # ── Train ─────────────────────────────────────────────────────
            model.train()
            ep_loss, ep_correct, ep_total = 0.0, 0, 0

            for windows, gesture_labels, subject_labels in dl_train:
                windows         = windows.to(device)
                gesture_labels  = gesture_labels.to(device)
                subject_labels  = subject_labels.to(device)

                optimizer.zero_grad()

                # model.forward() in training mode calls FreqBandStyleMixer
                # which mixes band statistics across training subjects.
                # subject_labels are training-subject indices ONLY — test
                # subject is never present in this DataLoader.
                logits = model(windows, subject_labels=subject_labels)

                loss = criterion(logits, gesture_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_loss    += loss.item() * bs
                preds       = logits.argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total   += bs

            train_loss = ep_loss / max(1, ep_total)
            train_acc  = ep_correct / max(1, ep_total)

            # ── Validate (no band mixing: model.eval()) ───────────────────
            # The validation loop calls model(xb) without subject_labels.
            # model.eval() disables FreqBandStyleMixer → raw signal → encoder.
            # This mirrors inference behaviour and gives an unbiased val metric.
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)  # no subject_labels → no mixing
                        val_loss_sum += criterion(logits, yb).item() * yb.size(0)
                        val_correct  += (logits.argmax(1) == yb).sum().item()
                        val_total    += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc  = val_correct  / max(1, val_total)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f}"
            )

            # ── Early stopping ────────────────────────────────────────────
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}.")
                        break

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── Store trainer state (required by evaluate_numpy) ─────────────
        # evaluate_numpy() is inherited from DisentangledTrainer and relies on
        # these attributes being set after fit().
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── Save training history ─────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── In-fold evaluation (val + internal test split) ────────────────
        # Uses inherited evaluate_numpy() which runs in eval mode (no mixing).
        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str):
            """Evaluate a DataLoader using the inference path (no mixing)."""
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(device)
                    all_logits.append(model(xb).cpu().numpy())
                    all_y.append(yb.numpy())

            from sklearn.metrics import (
                accuracy_score, f1_score, classification_report, confusion_matrix
            )
            import numpy as _np

            logits_arr = _np.concatenate(all_logits, axis=0)
            y_true     = _np.concatenate(all_y,      axis=0)
            y_pred     = logits_arr.argmax(axis=1)

            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average="macro")
            rep  = classification_report(y_true, y_pred, output_dict=True,
                                          zero_division=0)
            cm   = confusion_matrix(y_true, y_pred,
                                     labels=_np.arange(num_classes))

            if self.visualizer is not None:
                class_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, class_labels, normalize=True,
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

        # ── Save model checkpoint ─────────────────────────────────────────
        model_path = self.output_dir / "freq_band_style_mix_emg.pt"
        torch.save({
            "state_dict":      model.state_dict(),
            "in_channels":     in_channels,
            "num_classes":     num_classes,
            "class_ids":       class_ids,
            "mean":            mean_c,
            "std":             std_c,
            "window_size":     window_size,
            "sampling_rate":   self.sampling_rate,
            "low_band":        self.low_band,
            "mid_band":        self.mid_band,
            "low_mix_alpha":   self.low_mix_alpha,
            "mid_mix_alpha":   self.mid_mix_alpha,
            "classifier_dim":  self.classifier_dim,
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model checkpoint saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy() is inherited from DisentangledTrainer unchanged.
    # It calls model(xb) in eval mode → FreqBandStyleMixer is a no-op →
    # raw (standardised) signal passes directly through SharedEncoder.
    # No subject information is needed or used at inference time.
