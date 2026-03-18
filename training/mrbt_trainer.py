"""
Trainer for MRBT-CG — Multi-Resolution Barlow Twins with Channel-Group Factorization.

Training procedure (per batch):
    1.  L_cls  : CrossEntropy on content-pathway logits (gesture classification).
    2.  L_bt   : Per-scale Barlow Twins cross-correlation decorrelation loss.
                 At each scale l:
                   z_content_l = content_l.mean(dim=2)  — (B, C_g) temporal mean
                   z_style_l   = style_l.mean(dim=2)     — (B, C_g) temporal mean
                   L_bt_l      = barlow_twins_decor_loss(z_content_l, z_style_l)
                 L_bt = (1/L) * sum_l L_bt_l
    3.  L_total = L_cls + lambda_bt * L_bt

    Validation and inference use forward() (content pathway only) without L_bt.

LOSO Compliance — zero data leakage guaranteed:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Training                                                         │
    │  • mean_c, std_c  → computed from TRAINING windows only         │
    │  • model.forward_with_style() → called only on training batches  │
    │  • BT loss → computed on training-batch features only           │
    │  • class_ids → set from training-split gesture IDs              │
    │                                                                  │
    │ Validation                                                       │
    │  • model.forward() → content pathway; no style, no BT loss      │
    │  • val windows normalised with training mean_c/std_c             │
    │                                                                  │
    │ Inference (evaluate_numpy)                                       │
    │  • X: (N, T, C) test-subject windows, never seen during training │
    │  • Transposed to (N, C, T), normalised with training mean_c/std_c│
    │  • model.eval() → BN uses training running stats                 │
    │  • model.forward() → content pathway only; no AdaIN, no decoder, │
    │    no subject-specific statistics                                │
    └──────────────────────────────────────────────────────────────────┘
"""

import json
import logging
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
from models.mrbt_channel_group_loso import MRBTChannelGroupModel, barlow_twins_decor_loss
from training.datasets import WindowDataset
from training.trainer import WindowClassifierTrainer
from utils.logging import get_worker_init_fn, seed_everything


class MRBTTrainer(WindowClassifierTrainer):
    """
    Trainer for MRBTChannelGroupModel (Multi-Resolution Barlow Twins).

    Overrides fit() and evaluate_numpy() from WindowClassifierTrainer.
    All normalisation helpers (_compute_channel_standardization,
    _apply_standardization, _prepare_splits_arrays) are inherited.

    Parameters
    ----------
    n_groups : int
        K — number of EMG channel groups (must divide in_channels evenly).
    group_channels : int
        C_g — feature channels per group.  Must be divisible by `res2_scale`.
    n_scales : int
        L — number of parallel temporal scales.
    dilations : list of int
        Dilation per scale branch (length must equal n_scales).
    embedding_dim : int
        E — pre-classifier embedding dimension.
    res2_scale : int
        Res2Net branching factor inside each group block.
    se_reduction : int
        SE bottleneck reduction factor.
    lambda_bt : float
        Weight of the Barlow Twins decorrelation loss.
    bt_warmup_frac : float
        Fraction of total epochs over which lambda_bt is linearly ramped
        from 0 to its target value.  Ramp stabilises early training when
        content/style representations are still forming.
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # ── Architecture ─────────────────────────────────────────────────
        n_groups: int = 4,
        group_channels: int = 32,
        n_scales: int = 3,
        dilations: Optional[List[int]] = None,
        embedding_dim: int = 128,
        res2_scale: int = 4,
        se_reduction: int = 8,
        # ── Loss ─────────────────────────────────────────────────────────
        lambda_bt: float = 0.1,
        bt_warmup_frac: float = 0.15,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.n_groups = n_groups
        self.group_channels = group_channels
        self.n_scales = n_scales
        self.dilations = dilations if dilations is not None else [1, 2, 4]
        self.embedding_dim = embedding_dim
        self.res2_scale = res2_scale
        self.se_reduction = se_reduction
        self.lambda_bt = lambda_bt
        self.bt_warmup_frac = bt_warmup_frac

    # ─────────────────────────────────────────────────────────────────────────
    # fit()
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train MRBTChannelGroupModel.

        Args:
            splits: {
                "train": Dict[int, np.ndarray],  # gesture_id → (N, T, C)
                "val":   Dict[int, np.ndarray],
                "test":  Dict[int, np.ndarray],
            }

        LOSO guarantee:
            mean_c / std_c computed from X_train only.
            BT loss uses only training-batch tensors.
            model.forward_with_style() never receives test-subject windows.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Prepare flat arrays ────────────────────────────────────────
        (X_train, y_train, X_val, y_val, X_test, y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T) ──────────────────────────
        # _prepare_splits_arrays returns (N, T, C); PyTorch Conv1d needs (N, C, T).
        def _to_channels_first(X: np.ndarray) -> np.ndarray:
            if X.ndim == 3 and X.shape[2] < X.shape[1]:
                # Already channels-first (C < T)
                return X
            return X.transpose(0, 2, 1)

        X_train = _to_channels_first(X_train)
        X_val = _to_channels_first(X_val)
        X_test = _to_channels_first(X_test)

        # ── 3. Channel standardisation — TRAINING DATA ONLY ──────────────
        # mean_c and std_c are computed exclusively from training windows.
        # Val and test windows are normalised using training statistics.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        X_val = self._apply_standardization(X_val, mean_c, std_c)
        X_test = self._apply_standardization(X_test, mean_c, std_c)
        np.savez(self.output_dir / "normalization_stats.npz", mean=mean_c, std=std_c)

        # ── 4. Store trainer state ────────────────────────────────────────
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        in_channels = X_train.shape[1]    # (N, C, T) → C
        window_size = X_train.shape[2]
        self.in_channels = in_channels
        self.window_size = window_size

        self.logger.info(
            f"[MRBT] train={X_train.shape}, val={X_val.shape}, "
            f"test={X_test.shape}, classes={len(class_ids)}"
        )
        self.logger.info(
            f"  Groups: n_groups={self.n_groups}, "
            f"group_channels={self.group_channels}, "
            f"n_scales={self.n_scales}, dilations={self.dilations}"
        )

        # ── 5. Build model ────────────────────────────────────────────────
        device = self.cfg.device
        model = MRBTChannelGroupModel(
            in_channels=in_channels,
            num_classes=len(class_ids),
            n_groups=self.n_groups,
            group_channels=self.group_channels,
            n_scales=self.n_scales,
            dilations=self.dilations,
            embedding_dim=self.embedding_dim,
            dropout=self.cfg.dropout,
            scale=self.res2_scale,
            se_reduction=self.se_reduction,
        ).to(device)
        self.logger.info(f"  Parameters: {model.count_parameters():,}")

        # ── 6. Data loaders ───────────────────────────────────────────────
        worker_init = get_worker_init_fn(self.cfg.seed)

        train_ds = WindowDataset(X_train, y_train)
        val_ds = WindowDataset(X_val, y_val)
        test_ds = WindowDataset(X_test, y_test)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            worker_init_fn=worker_init,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        # ── 7. Loss, optimiser, scheduler ────────────────────────────────
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=len(class_ids)).astype(np.float32)
            w = 1.0 / (counts + 1e-6)
            w = w / w.sum() * len(class_ids)
            cls_weight = torch.tensor(w, dtype=torch.float32, device=device)
        else:
            cls_weight = None

        criterion = nn.CrossEntropyLoss(weight=cls_weight)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # NOTE: verbose removed in PyTorch 2.4+; not passed here.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 8. Training loop ──────────────────────────────────────────────
        total_epochs = self.cfg.epochs
        warmup_epochs = max(1, int(total_epochs * self.bt_warmup_frac))

        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        patience_ctr = 0

        history: Dict[str, List] = {
            "train_loss": [], "train_cls_loss": [], "train_bt_loss": [],
            "train_acc": [], "val_loss": [], "val_acc": [],
        }

        for epoch in range(1, total_epochs + 1):
            # Linear warmup for BT loss: 0 → lambda_bt over first warmup_epochs.
            # Prevents the decorrelation signal from destabilising early
            # training before the content pathway has learned useful features.
            bt_weight = self.lambda_bt * min(1.0, epoch / warmup_epochs)

            # ── Train ────────────────────────────────────────────────────
            model.train()
            sum_cls = sum_bt = sum_total = 0.0
            n_correct = n_total = 0

            for X_b, y_b in train_loader:
                X_b = X_b.to(device)          # (B, C, T)
                y_b = y_b.to(device)          # (B,)

                # Training path: forward_with_style returns logits + per-scale
                # (content, style) tensors.  LOSO: only training windows here.
                logits, content_list, style_list = model.forward_with_style(X_b)

                # Classification loss
                L_cls = criterion(logits, y_b)

                # Barlow Twins decorrelation loss across L scales
                # Pool each scale's content and style in time (temporal mean).
                # Mean pooling is lightweight and differentiable.
                L_bt = torch.zeros(1, device=device)
                n_scale = len(content_list)
                for content_l, style_l in zip(content_list, style_list):
                    z_c = content_l.mean(dim=2)   # (B, C_g) — temporal mean
                    z_s = style_l.mean(dim=2)     # (B, C_g) — temporal mean
                    L_bt = L_bt + barlow_twins_decor_loss(z_c, z_s)
                if n_scale > 0:
                    L_bt = L_bt / n_scale

                L_total = L_cls + bt_weight * L_bt

                optimizer.zero_grad()
                L_total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                B = X_b.size(0)
                sum_cls += L_cls.item() * B
                sum_bt += L_bt.item() * B
                sum_total += L_total.item() * B
                n_correct += (logits.argmax(1) == y_b).sum().item()
                n_total += B

            train_acc = n_correct / max(n_total, 1)
            avg_cls = sum_cls / max(n_total, 1)
            avg_bt = sum_bt / max(n_total, 1)
            avg_total = sum_total / max(n_total, 1)

            # ── Validate ─────────────────────────────────────────────────
            # Content pathway only — consistent with inference behaviour.
            # No BT loss on validation (not needed for early stopping).
            model.eval()
            val_loss = val_n = val_correct = 0
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(device), y_v.to(device)
                    out = model(X_v)                       # content pathway
                    val_loss += criterion(out, y_v).item() * X_v.size(0)
                    val_correct += (out.argmax(1) == y_v).sum().item()
                    val_n += X_v.size(0)
            val_loss /= max(val_n, 1)
            val_acc = val_correct / max(val_n, 1)

            scheduler.step(val_loss)

            history["train_loss"].append(avg_total)
            history["train_cls_loss"].append(avg_cls)
            history["train_bt_loss"].append(avg_bt)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if epoch % 5 == 0 or epoch == 1:
                self.logger.info(
                    f"  Ep {epoch:3d}/{total_epochs}"
                    f" | L={avg_total:.4f}"
                    f" cls={avg_cls:.4f}"
                    f" bt={avg_bt:.4f} (w={bt_weight:.3f})"
                    f" | tr_acc={train_acc:.4f}"
                    f" | val_loss={val_loss:.4f}"
                    f" | val_acc={val_acc:.4f}"
                )

            # ── Early stopping ────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.early_stopping_patience:
                    self.logger.info(f"  Early stopping at epoch {epoch}.")
                    break

        # ── 9. Restore best checkpoint ────────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model

        # ── 10. Save artefacts ────────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=2)

        if self.visualizer is not None:
            try:
                self.visualizer.plot_training_curves(
                    {
                        "train_loss": history["train_loss"],
                        "val_loss": history["val_loss"],
                        "train_acc": history["train_acc"],
                        "val_acc": history["val_acc"],
                    },
                    "training_curves.png",
                )
            except Exception as exc:
                self.logger.warning(f"Could not plot training curves: {exc}")

        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_channels": in_channels,
                "num_classes": len(class_ids),
                "class_ids": class_ids,
                "mean": mean_c,
                "std": std_c,
                "window_size": window_size,
                "model_config": {
                    "n_groups": self.n_groups,
                    "group_channels": self.group_channels,
                    "n_scales": self.n_scales,
                    "dilations": self.dilations,
                    "embedding_dim": self.embedding_dim,
                    "res2_scale": self.res2_scale,
                    "se_reduction": self.se_reduction,
                },
                "loss_config": {
                    "lambda_bt": self.lambda_bt,
                    "bt_warmup_frac": self.bt_warmup_frac,
                },
            },
            self.output_dir / "mrbt_model.pt",
        )

        # ── 11. Inline evaluation helper ──────────────────────────────────
        def _eval_loader(loader: DataLoader, name: str) -> Dict:
            model.eval()
            all_preds, all_labels, all_logits = [], [], []
            with torch.no_grad():
                for X_b, y_b in loader:
                    out = model(X_b.to(device))        # content pathway → logits
                    all_logits.append(out.cpu().numpy())
                    all_preds.append(out.argmax(1).cpu().numpy())
                    all_labels.append(y_b.numpy())
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_labels)
            logits_np = np.concatenate(all_logits)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred).tolist()
            rep = classification_report(
                y_true, y_pred,
                target_names=[class_names[c] for c in class_ids],
                output_dict=True, zero_division=0,
            )
            self.logger.info(f"  [{name}] acc={acc:.4f}  f1_macro={f1:.4f}")
            return {
                "accuracy": acc, "f1_macro": f1,
                "confusion_matrix": cm, "report": rep, "logits": logits_np,
            }

        val_results = _eval_loader(val_loader, "val")
        test_results = _eval_loader(test_loader, "test")

        results = {
            "class_ids": class_ids,
            "class_names": {str(k): v for k, v in class_names.items()},
            "val": val_results,
            "test": test_results,
        }
        with open(self.output_dir / "classification_results.json", "w") as fh:
            json.dump(
                results, fh, indent=2,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
            )
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # evaluate_numpy — content pathway only
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate on (N, T, C) windows using the content pathway exclusively.

        LOSO guarantee:
            test-subject windows are normalised with training mean_c/std_c
            (set in fit() from training data only).  The content pathway
            (group_init → scale_blocks → cross_attn content → MFA → ASP
             → embedding → classifier) does NOT use any subject-specific
            statistics or the style representations.

        Args:
            X:          (N, T, C) raw test windows
            y:          (N,) class indices matching class_ids order
            split_name: tag for logging / plot filenames
            visualize:  if True and visualizer set, saves confusion matrix
        Returns:
            dict with accuracy, f1_macro, confusion_matrix, report, logits
        """
        assert self.model is not None, "Call fit() before evaluate_numpy()"
        assert self.mean_c is not None, "mean_c not set — fit() must run first"
        assert self.std_c is not None, "std_c not set — fit() must run first"
        assert self.class_ids is not None
        assert self.class_names is not None

        # Transpose (N, T, C) → (N, C, T) if needed (same logic as fit)
        if X.ndim == 3 and X.shape[2] < X.shape[1]:
            pass   # already (N, C, T)
        else:
            X = X.transpose(0, 2, 1)

        # Normalise with training statistics — NO test-subject info used
        X = self._apply_standardization(X, self.mean_c, self.std_c)

        device = self.cfg.device
        dataset = WindowDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        self.model.eval()
        all_preds, all_labels, all_logits = [], [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                out = self.model(X_b.to(device))    # content pathway
                all_logits.append(out.cpu().numpy())
                all_preds.append(out.argmax(1).cpu().numpy())
                all_labels.append(y_b.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        logits_np = np.concatenate(all_logits)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        rep = classification_report(
            y_true, y_pred,
            target_names=[self.class_names[c] for c in self.class_ids],
            output_dict=True, zero_division=0,
        )

        self.logger.info(
            f"  evaluate_numpy [{split_name}] acc={acc:.4f}  f1_macro={f1:.4f}"
        )

        if visualize and self.visualizer is not None:
            try:
                self.visualizer.plot_confusion_matrix(
                    np.array(cm),
                    class_labels=[self.class_names[c] for c in self.class_ids],
                    normalize=True,
                    filename=f"confusion_matrix_{split_name}.png",
                )
                self.visualizer.plot_per_class_f1(
                    rep,
                    class_labels=[self.class_names[c] for c in self.class_ids],
                    filename=f"per_class_f1_{split_name}.png",
                )
            except Exception as exc:
                self.logger.warning(f"Visualisation failed: {exc}")

        return {
            "accuracy": acc,
            "f1_macro": f1,
            "confusion_matrix": cm,
            "report": rep,
            "logits": logits_np,
        }
