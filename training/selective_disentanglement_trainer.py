"""
Trainer for SelectiveDisentanglementECAPA (Experiment 89).

Implements multi-loss training with:
  1. Gesture classification loss (CrossEntropy on gesture_logits)
  2. Auxiliary subject classification loss (CrossEntropy on subject_logits,
     applied to domain-aware embedding z — encourages encoder to encode subjects)
  3. Domain confusion loss (CrossEntropy on domain_logits after GRL,
     applied to projection h — pushes h to be domain-invariant)

Loss: L = L_cls + lambda_subj * L_subj_aux + lambda_dom * L_domain_confusion

GRL alpha annealing (standard DANN schedule):
  alpha(p) = 2 / (1 + exp(-10 * p)) - 1
  where p = epoch / total_epochs goes from 0 → 1.
  This starts alpha ≈ 0 (no domain confusion pressure) and ramps to 1.

After training, compute_mean_subject_embedding() is called to set the FiLM
mean embedding for LOSO-clean test-time inference.

LOSO data-leakage checklist
────────────────────────────
  ✓ Channel mean/std computed from X_train only — never from val or test.
  ✓ Subject labels used during training only — evaluate_numpy() uses inference=True.
  ✓ mean_subject_emb computed from model parameters only (embedding table),
    NOT from any test-subject signal data.
  ✓ model.eval() at inference: BatchNorm frozen to training running stats.
  ✓ No per-subject centering, no adaptive normalisation at test time.
  ✓ Validation loss monitors gesture classification only (not subject loss).
  ✓ Early stopping criterion is val gesture loss — subject labels not involved.
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
from torch.utils.data import Dataset, DataLoader

from config.base import TrainingConfig
from models.selective_disentanglement_ecapa import SelectiveDisentanglementECAPA
from training.trainer import WindowClassifierTrainer
from training.datasets import WindowDataset
from utils.logging import get_worker_init_fn, seed_everything


# ══════════════════════ Dataset with subject labels ════════════════════════

class SubjectAwareDataset(Dataset):
    """
    Dataset returning (window, gesture_label, subject_label).

    Used during training only.  At test time, WindowDataset is used
    (no subject labels required or accessed).

    Args:
        X:         (N, C, T) float32 — normalised EMG windows (channels-first).
        y_gesture: (N,)      int64   — class indices [0, num_classes).
        y_subject: (N,)      int64   — subject indices [0, num_subjects).
    """

    def __init__(
        self,
        X: np.ndarray,
        y_gesture: np.ndarray,
        y_subject: np.ndarray,
    ):
        self.X = torch.from_numpy(X).float()
        self.y_gesture = torch.from_numpy(y_gesture).long()
        self.y_subject = torch.from_numpy(y_subject).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_gesture[idx], self.y_subject[idx]


# ══════════════════════════════ Trainer ════════════════════════════════════

class SelectiveDisentanglementTrainer(WindowClassifierTrainer):
    """
    Trainer for SelectiveDisentanglementECAPA (exp_89).

    Expects splits dict produced by _build_splits_with_subject_labels():
        "train":               Dict[int, np.ndarray]  — gesture_id → (N, T, C)
        "val":                 Dict[int, np.ndarray]
        "test":                Dict[int, np.ndarray]
        "train_subject_labels": Dict[int, np.ndarray] — gesture_id → subject indices
        "num_train_subjects":  int

    Architecture hyper-parameters:
        channels, scale, embedding_dim, subject_emb_dim, proj_dim,
        dilations, se_reduction  — forwarded to SelectiveDisentanglementECAPA.

    Loss hyper-parameters:
        lambda_subj: weight for auxiliary subject classification loss.
        lambda_dom:  weight for domain confusion loss (after GRL).

    LOSO safety is enforced by:
        - Using channel normalisation computed from X_train exclusively.
        - Calling model.forward(inference=True) in evaluate_numpy(), which
          uses mean_subject_emb (model parameter average, not test data).
        - Validation and early stopping use gesture loss only.
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # ── architecture ──────────────────────────────────────────────────
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        subject_emb_dim: int = 32,
        proj_dim: int = 128,
        dilations: Optional[List[int]] = None,
        se_reduction: int = 8,
        # ── loss weights ──────────────────────────────────────────────────
        lambda_subj: float = 0.5,
        lambda_dom: float = 0.3,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.channels       = channels
        self.scale          = scale
        self.embedding_dim  = embedding_dim
        self.subject_emb_dim = subject_emb_dim
        self.proj_dim       = proj_dim
        self.dilations      = dilations if dilations is not None else [2, 3, 4]
        self.se_reduction   = se_reduction
        self.lambda_subj    = lambda_subj
        self.lambda_dom     = lambda_dom

    # ──────────────────────────── helpers ────────────────────────────────────

    @staticmethod
    def _compute_dann_alpha(epoch: int, total_epochs: int) -> float:
        """
        Standard DANN alpha schedule: ramps from ≈0 to 1 over training.

        alpha(p) = 2 / (1 + exp(-10 * p)) - 1,  p = epoch / total_epochs

        Starts near 0 so the gesture classifier has time to stabilise before
        domain confusion pressure is applied.
        """
        p = epoch / max(total_epochs, 1)
        return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

    def _build_subject_label_arrays(
        self,
        class_ids: List[int],
        train_dict: Dict[int, np.ndarray],
        train_subj_labels: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Align subject labels with training windows in the same order as
        _prepare_splits_arrays() produces for y_train.

        _prepare_splits_arrays iterates class_ids in sorted order, so we
        must produce subject labels in exactly the same order.

        Returns:
            y_subj_train: (N,) int64 — subject index for each training window.
        """
        parts = []
        for gid in class_ids:
            if gid in train_subj_labels:
                parts.append(train_subj_labels[gid])
        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0).astype(np.int64)

    # ──────────────────────────────── fit ────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train SelectiveDisentanglementECAPA on LOSO training split.

        Args:
            splits: dict with keys:
                "train":               Dict[int, np.ndarray]  (N, T, C)
                "val":                 Dict[int, np.ndarray]  (N, T, C)
                "test":                Dict[int, np.ndarray]  (N, T, C)
                "train_subject_labels": Dict[int, np.ndarray] subject indices
                "num_train_subjects":  int

        Returns:
            results dict with val and internal-test metrics.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Extract extra keys before passing to parent helper ─────────
        train_subj_labels: Dict[int, np.ndarray] = splits.get(
            "train_subject_labels", {}
        )
        num_train_subjects: int = splits.get("num_train_subjects", 1)

        # ── 2. Splits → flat (N, T, C) arrays ────────────────────────────
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        # ── 3. Build aligned subject label vector for training ─────────────
        y_subj_train = self._build_subject_label_arrays(
            class_ids, splits["train"], train_subj_labels
        )
        if len(y_subj_train) != len(X_train):
            self.logger.warning(
                f"Subject label count ({len(y_subj_train)}) != "
                f"training window count ({len(X_train)}). "
                "Falling back to zero subject labels (no subject loss)."
            )
            y_subj_train = np.zeros(len(X_train), dtype=np.int64)

        # ── 4. Transpose (N, T, C) → (N, C, T) ───────────────────────────
        # ECAPA-TDNN expects channels-first (B, C, T) like Conv1d.
        # Heuristic: T >> C for EMG data (e.g. 600 >> 8).
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
        num_classes = len(class_ids)

        # ── 5. Channel standardisation (training data ONLY) ───────────────
        # LOSO integrity: mean/std exclusively from training windows.
        # Applied identically to val and test — no re-computation.
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
        self.logger.info("Per-channel standardisation applied (training stats only).")

        # ── 6. Build model ────────────────────────────────────────────────
        model = SelectiveDisentanglementECAPA(
            in_channels=in_channels,
            num_classes=num_classes,
            num_subjects_train=num_train_subjects,
            channels=self.channels,
            scale=self.scale,
            embedding_dim=self.embedding_dim,
            subject_emb_dim=self.subject_emb_dim,
            proj_dim=self.proj_dim,
            dilations=self.dilations,
            dropout=self.cfg.dropout,
            se_reduction=self.se_reduction,
        ).to(self.cfg.device)

        total_params = model.count_parameters()
        self.logger.info(
            f"SelectiveDisentanglementECAPA: "
            f"in_ch={in_channels}, classes={num_classes}, "
            f"subjects={num_train_subjects}, C={self.channels}, "
            f"embed={self.embedding_dim}, subj_emb={self.subject_emb_dim}, "
            f"proj={self.proj_dim}, dilations={self.dilations} | "
            f"total_params={total_params:,}"
        )

        # ── 7. Datasets & DataLoaders ─────────────────────────────────────
        ds_train = SubjectAwareDataset(X_train, y_train, y_subj_train)
        ds_val   = WindowDataset(X_val, y_val)   if len(X_val) > 0  else None
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

        # ── 8. Loss functions ─────────────────────────────────────────────
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = counts.sum() / (counts + 1e-8)
            cw /= cw.mean()
            criterion_cls = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(self.cfg.device)
            )
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
        else:
            criterion_cls = nn.CrossEntropyLoss()

        # Subject and domain confusion heads use uniform class weights
        criterion_subj = nn.CrossEntropyLoss()
        criterion_dom  = nn.CrossEntropyLoss()

        # ── 9. Optimiser + LR scheduler ───────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 10. Training loop ──────────────────────────────────────────────
        history: Dict = {
            "train_loss_cls":  [],
            "train_loss_subj": [],
            "train_loss_dom":  [],
            "train_loss_total":[],
            "train_acc": [],
            "val_loss":  [],
            "val_acc":   [],
        }
        best_val_loss: float = float("inf")
        best_state: Optional[Dict] = None
        no_improve: int = 0
        device = self.cfg.device
        total_epochs = self.cfg.epochs

        for epoch in range(1, total_epochs + 1):
            # DANN alpha annealing: 0 → 1 over training
            alpha = self._compute_dann_alpha(epoch, total_epochs)

            model.train()
            ep_cls, ep_subj, ep_dom = 0.0, 0.0, 0.0
            ep_correct, ep_total = 0, 0

            for xb, yb_gest, yb_subj in dl_train:
                xb       = xb.to(device)
                yb_gest  = yb_gest.to(device)
                yb_subj  = yb_subj.to(device)

                optimizer.zero_grad()

                gesture_logits, subject_logits, domain_logits = model(
                    xb, subject_ids=yb_subj, alpha=alpha, inference=False,
                )

                loss_cls  = criterion_cls(gesture_logits, yb_gest)
                loss_subj = criterion_subj(subject_logits, yb_subj)
                loss_dom  = criterion_dom(domain_logits, yb_subj)

                loss = (
                    loss_cls
                    + self.lambda_subj * loss_subj
                    + self.lambda_dom  * loss_dom
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = xb.size(0)
                ep_cls     += loss_cls.item()  * bs
                ep_subj    += loss_subj.item() * bs
                ep_dom     += loss_dom.item()  * bs
                ep_correct += (gesture_logits.argmax(1) == yb_gest).sum().item()
                ep_total   += bs

            n = max(1, ep_total)
            train_loss_cls  = ep_cls  / n
            train_loss_subj = ep_subj / n
            train_loss_dom  = ep_dom  / n
            train_loss_total = (
                train_loss_cls
                + self.lambda_subj * train_loss_subj
                + self.lambda_dom  * train_loss_dom
            )
            train_acc = ep_correct / n

            # ── Validation (gesture loss only, no subject data) ───────────
            if dl_val is not None:
                model.eval()
                vl_sum, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        # inference=True: uses mean_subject_emb (not yet computed,
                        # but mean_subject_emb buffer is initialised to zeros,
                        # giving identity FiLM — acceptable for val monitoring)
                        g_logits, _, _ = model(xb, inference=True)
                        vl_sum += criterion_cls(g_logits, yb).item() * yb.size(0)
                        vc     += (g_logits.argmax(1) == yb).sum().item()
                        vt     += yb.size(0)
                val_loss = vl_sum / max(1, vt)
                val_acc  = vc    / max(1, vt)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["train_loss_cls"].append(train_loss_cls)
            history["train_loss_subj"].append(train_loss_subj)
            history["train_loss_dom"].append(train_loss_dom)
            history["train_loss_total"].append(train_loss_total)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            self.logger.info(
                f"[Epoch {epoch:02d}/{total_epochs}] alpha={alpha:.3f} | "
                f"L_cls={train_loss_cls:.4f}, "
                f"L_subj={train_loss_subj:.4f}, "
                f"L_dom={train_loss_dom:.4f} | "
                f"train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

            # ── Early stopping (gesture val loss) ─────────────────────────
            if dl_val is not None and not np.isnan(val_loss):
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

        # ── 11. Compute mean subject embedding for inference ────────────────
        # LOSO safety: uses only model parameters (embedding table), no test data.
        model.compute_mean_subject_embedding()
        self.logger.info(
            "Mean subject embedding computed from training embedding table "
            "(no test data involved)."
        )

        # ── 12. Store trainer state ────────────────────────────────────────
        self.model          = model
        self.mean_c         = mean_c
        self.std_c          = std_c
        self.class_ids      = class_ids
        self.class_names    = class_names
        self.in_channels    = in_channels

        # ── 13. Save training history ──────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=4)
        if self.visualizer is not None:
            # Plot gesture loss curve as the primary metric
            vis_history = {
                "train_loss": history["train_loss_cls"],
                "val_loss":   history["val_loss"],
                "train_acc":  history["train_acc"],
                "val_acc":    history["val_acc"],
            }
            self.visualizer.plot_training_curves(
                vis_history, filename="training_curves.png"
            )

        # ── 14. In-fold evaluation (val / internal test) ───────────────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dl, split_name: str):
            if dl is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(device)
                    g_logits, _, _ = model(xb, inference=True)
                    all_logits.append(g_logits.cpu().numpy())
                    all_y.append(yb.numpy())
            y_true = np.concatenate(all_y)
            y_pred = np.concatenate(all_logits).argmax(axis=1)
            acc = float(accuracy_score(y_true, y_pred))
            f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm  = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True, filename=f"cm_{split_name}.png"
                )
            return {"accuracy": acc, "f1_macro": f1, "report": rep,
                    "confusion_matrix": cm.tolist()}

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "internal_test")

        # ── 15. Save checkpoint ────────────────────────────────────────────
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_channels": in_channels,
                "num_classes": num_classes,
                "num_subjects_train": num_train_subjects,
                "class_ids": class_ids,
                "mean": mean_c,
                "std":  std_c,
                "model_config": {
                    "channels":       self.channels,
                    "scale":          self.scale,
                    "embedding_dim":  self.embedding_dim,
                    "subject_emb_dim": self.subject_emb_dim,
                    "proj_dim":       self.proj_dim,
                    "dilations":      self.dilations,
                    "se_reduction":   self.se_reduction,
                    "dropout":        self.cfg.dropout,
                },
                "loss_config": {
                    "lambda_subj": self.lambda_subj,
                    "lambda_dom":  self.lambda_dom,
                },
                "training_config": asdict(self.cfg),
            },
            self.output_dir / "selective_disentanglement_ecapa.pt",
        )
        self.logger.info(
            f"Checkpoint saved → {self.output_dir / 'selective_disentanglement_ecapa.pt'}"
        )

        with open(self.output_dir / "classification_results.json", "w") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False)

        return results

    # ──────────────────────────── evaluate_numpy ─────────────────────────────

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
          1. Transpose (N, T, C) → (N, C, T) if needed.
          2. Channel standardisation using TRAINING statistics (mean_c, std_c).
          3. model.forward(inference=True) — mean_subject_emb for FiLM,
             no subject head, no GRL, no test-subject data involved.

        LOSO safety:
          mean_subject_emb was computed from model parameters after training.
          No test-subject statistics are used anywhere in this method.

        Args:
            X:          Raw EMG windows, (N, T, C) or (N, C, T).
            y:          Integer class labels matching class_ids from fit().
            split_name: Prefix for confusion matrix image file.
            visualize:  Whether to save confusion matrix plot.

        Returns:
            dict with "accuracy", "f1_macro", "report", "confusion_matrix", "logits".
        """
        assert self.model       is not None, "Call fit() before evaluate_numpy()."
        assert self.mean_c      is not None and self.std_c is not None
        assert self.class_ids   is not None

        X_in = X.copy().astype(np.float32)

        # Transpose if (N, T, C) — same heuristic as fit()
        if X_in.ndim == 3 and X_in.shape[1] > X_in.shape[2]:
            X_in = X_in.transpose(0, 2, 1)

        # Apply training-data standardisation (no test stats)
        X_in = self._apply_standardization(X_in, self.mean_c, self.std_c)

        ds = WindowDataset(X_in, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        # model.eval() — BatchNorm uses training running stats; no updates
        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                # inference=True: mean_subject_emb FiLM, no subject head
                g_logits, _, _ = self.model(xb, inference=True)
                all_logits.append(g_logits.cpu().numpy())
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y, axis=0)
        y_pred = logits.argmax(axis=1)

        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        num_classes = len(self.class_ids)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        if visualize and self.visualizer is not None:
            labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, labels, normalize=True,
                filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy":         acc,
            "f1_macro":         f1,
            "report":           rep,
            "confusion_matrix": cm.tolist(),
            "logits":           logits,
        }
