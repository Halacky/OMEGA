"""
Trainer for OrthoMixECAPAEmg (Experiment 81).

Extends ECAPATDNNTrainer with orthogonal channel-mixing augmentation.
The only structural difference from ECAPATDNNTrainer is:

  1. Model class: OrthoMixECAPAEmg (wraps ECAPATDNNEmg with mixing module).
  2. Two extra hyperparameters: mix_epsilon, mix_prob.
  3. Checkpoint filename: ortho_mix_ecapa_emg.pt
  4. model_config in checkpoint includes mix_epsilon and mix_prob.

The training loop explicitly comments where model.train() / model.eval() is
called, making the augmentation guard visible in the code.

LOSO data-leakage checklist (inherits from ECAPATDNNTrainer)
────────────────────────────
  ✓ Channel mean/std computed from X_train ONLY — never from val or test.
  ✓ OrthoMixECAPAEmg._ortho_mix() uses no subject statistics whatsoever.
  ✓ _ortho_mix is gated by self.training → OFF at eval() / test time.
  ✓ model.eval() is called before every val/test pass — no adaptation.
  ✓ No per-subject centering, adaptive BN, or fine-tuning at test time.
  ✓ evaluate_numpy() inherited unchanged from ECAPATDNNTrainer —
    applies training-stats normalisation + model.eval() internally.
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
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader

from config.base import TrainingConfig
from models.ortho_mix_ecapa_emg import OrthoMixECAPAEmg
from training.ecapa_tdnn_trainer import ECAPATDNNTrainer
from training.datasets import WindowDataset
from utils.logging import get_worker_init_fn, seed_everything


class OrthoMixECAPATrainer(ECAPATDNNTrainer):
    """
    Trainer for OrthoMixECAPAEmg.

    Structurally identical to ECAPATDNNTrainer; the only change is using
    OrthoMixECAPAEmg instead of ECAPATDNNEmg so that stochastic orthogonal
    channel-mixing augmentation is applied during training batches.

    evaluate_numpy() is inherited unchanged from ECAPATDNNTrainer.
    It calls model.eval() internally, so the augmentation is always OFF
    during cross-subject test evaluation.

    Args:
        train_cfg:     TrainingConfig dataclass.
        logger:        Python logger instance.
        output_dir:    Directory for checkpoints and logs.
        visualizer:    Optional Visualizer for training curves / CM plots.
        channels:      C — internal TDNN feature width (default 128).
        scale:         Res2Net scale (default 4).
        embedding_dim: Pre-classifier embedding dimension (default 128).
        dilations:     Dilation per SE-Res2Net block (default [2, 3, 4]).
        se_reduction:  SE bottleneck reduction factor (default 8).
        mix_epsilon:   ε controlling orthogonal mixing distance from I (0.1).
        mix_prob:      Per-batch probability of applying mixing (0.7).
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
        mix_epsilon: float = 0.1,
        mix_prob: float = 0.7,
    ):
        super().__init__(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=visualizer,
            channels=channels,
            scale=scale,
            embedding_dim=embedding_dim,
            dilations=dilations,
            se_reduction=se_reduction,
        )
        self.mix_epsilon = mix_epsilon
        self.mix_prob = mix_prob

    # ─────────────────────────────── fit ─────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train OrthoMixECAPAEmg on the LOSO training split.

        Structurally identical to ECAPATDNNTrainer.fit().
        Changes from the parent are marked with  # ORTHO-MIX CHANGE.

        LOSO augmentation guard
        ───────────────────────
        The training loop calls:
          model.train()   before iterating dl_train  → augmentation ON
          model.eval()    before iterating dl_val     → augmentation OFF
          model.eval()    before iterating dl_test    → augmentation OFF

        OrthoMixECAPAEmg.forward() checks self.training — this flag is set
        by model.train() / model.eval().  The augmentation therefore cannot
        leak into val or test batches even if this method is called incorrectly.

        Args:
            splits: {"train": Dict[int, np.ndarray],
                     "val":   Dict[int, np.ndarray],
                     "test":  Dict[int, np.ndarray]}
                    Arrays have shape (N, T, C).

        Returns:
            results dict with val and internal-test metrics.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Splits → flat (N, T, C) arrays ───────────────────────────────
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T) ──────────────────────────────
        # ECAPATDNNEmg (and OrthoMixECAPAEmg) expect channels-first: (B, C, T).
        # _prepare_splits_arrays returns (N, T, C) from the loader.
        # Detect format: for EMG data, T (time samples) >> C (8 channels).
        if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
            X_train = X_train.transpose(0, 2, 1)
            if X_val.ndim == 3 and len(X_val) > 0:
                X_val = X_val.transpose(0, 2, 1)
            if X_test.ndim == 3 and len(X_test) > 0:
                X_test = X_test.transpose(0, 2, 1)
            self.logger.info(
                f"Transposed windows to (N, C, T): X_train={X_train.shape}"
            )

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 3. Per-channel standardisation (training data ONLY) ──────────────
        # LOSO integrity: mean/std computed exclusively from X_train.
        # The same statistics are then applied to val and test (no re-computing).
        # This is the only permissible statistics flow in LOSO protocol.
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
            "Per-channel standardisation applied (training statistics only)."
        )

        # ── 4. Build model ────────────────────────────────────────────────────
        # ORTHO-MIX CHANGE: OrthoMixECAPAEmg instead of ECAPATDNNEmg.
        # mix_epsilon and mix_prob control the augmentation strength.
        model = OrthoMixECAPAEmg(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=self.channels,
            scale=self.scale,
            embedding_dim=self.embedding_dim,
            dilations=self.dilations,
            dropout=self.cfg.dropout,
            se_reduction=self.se_reduction,
            mix_epsilon=self.mix_epsilon,   # ORTHO-MIX CHANGE
            mix_prob=self.mix_prob,         # ORTHO-MIX CHANGE
        ).to(self.cfg.device)

        total_params = model.count_parameters()
        self.logger.info(
            f"OrthoMixECAPAEmg: in_ch={in_channels}, classes={num_classes}, "  # CHANGE
            f"C={self.channels}, scale={self.scale}, "
            f"embed={self.embedding_dim}, dilations={self.dilations} | "
            f"mix_epsilon={self.mix_epsilon}, mix_prob={self.mix_prob} | "      # CHANGE
            f"total_params={total_params:,}"
        )

        # ── 5. Datasets & DataLoaders ─────────────────────────────────────────
        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val, y_val)   if len(X_val)  > 0 else None
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

        # ── 6. Loss function ──────────────────────────────────────────────────
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

        # ── 7. Optimizer + LR scheduler ───────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # verbose removed in PyTorch >= 2.4 — do not pass it
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 8. Training loop ──────────────────────────────────────────────────
        # AUGMENTATION GUARD:
        #   model.train()  →  OrthoMixECAPAEmg.training = True   → mixing ON
        #   model.eval()   →  OrthoMixECAPAEmg.training = False  → mixing OFF
        # This is enforced by PyTorch's Module.train() / Module.eval() which
        # recursively sets the flag on all sub-modules including the backbone.
        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }
        best_val_loss: float = float("inf")
        best_state: Optional[Dict] = None
        no_improve: int = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            # ── training pass: augmentation ENABLED ──────────────────────────
            model.train()
            ep_loss, ep_correct, ep_total = 0.0, 0, 0

            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                # OrthoMixECAPAEmg.forward() applies _ortho_mix() here
                # (self.training == True, random draw vs mix_prob)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                # Gradient clipping: stabilises training with SE attention layers
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = xb.size(0)
                ep_loss    += loss.item() * bs
                ep_correct += (logits.argmax(1) == yb).sum().item()
                ep_total   += bs

            train_loss = ep_loss    / max(1, ep_total)
            train_acc  = ep_correct / max(1, ep_total)

            # ── validation pass: augmentation DISABLED ────────────────────────
            if dl_val is not None:
                model.eval()  # ← self.training = False → mixing OFF
                vl_sum, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)  # no augmentation
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

            # ── early stopping ─────────────────────────────────────────────────
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

        # ── 9. Store trainer state (required by inherited evaluate_numpy) ─────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 10. Save training history ─────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── 11. In-fold evaluation (val / internal test splits) ───────────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dl, split_name: str):
            if dl is None:
                return None
            model.eval()  # ← augmentation OFF
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(device)
                    all_logits.append(model(xb).cpu().numpy())
                    all_y.append(yb.numpy())
            y_true = np.concatenate(all_y, axis=0)
            y_pred = np.concatenate(all_logits, axis=0).argmax(axis=1)
            acc = float(accuracy_score(y_true, y_pred))
            f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            rep = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            cm  = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True, filename=f"cm_{split_name}.png"
                )
            return {
                "accuracy": acc,
                "f1_macro": f1,
                "report":   rep,
                "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "internal_test")

        # ── 12. Save model checkpoint ─────────────────────────────────────────
        # ORTHO-MIX CHANGE: filename + mix params added to model_config
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
                    "channels":      self.channels,
                    "scale":         self.scale,
                    "embedding_dim": self.embedding_dim,
                    "dilations":     self.dilations,
                    "se_reduction":  self.se_reduction,
                    "dropout":       self.cfg.dropout,
                    "mix_epsilon":   self.mix_epsilon,   # ORTHO-MIX CHANGE
                    "mix_prob":      self.mix_prob,       # ORTHO-MIX CHANGE
                },
                "training_config": asdict(self.cfg),
            },
            self.output_dir / "ortho_mix_ecapa_emg.pt",  # ORTHO-MIX CHANGE
        )
        self.logger.info(
            f"Checkpoint saved → {self.output_dir / 'ortho_mix_ecapa_emg.pt'}"
        )

        with open(self.output_dir / "classification_results.json", "w") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False)

        return results

    # ─── evaluate_numpy ───────────────────────────────────────────────────────
    # Inherited from ECAPATDNNTrainer without changes.
    #
    # The inherited method:
    #   1. Calls model.eval() → OrthoMixECAPAEmg.training = False → mixing OFF.
    #   2. Applies transpose (N,T,C)→(N,C,T) using training heuristic.
    #   3. Applies training-data standardisation (mean_c, std_c from fit()).
    #   4. Runs inference with torch.no_grad().
    #
    # No test-subject data influences any parameter — LOSO compliant.
