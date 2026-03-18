"""
Trainer for MATEKroneckerEMG: Shared-Specific with Kronecker Attention.

Loss components:
  1. L_gesture:  CrossEntropy on gesture logits from the fused representation.
  2. L_subject:  CrossEntropy on subject logits from z_shared via GRL
                 (adversarial — gradient reversal makes z_shared subject-invariant).

Key difference from CausalDisentanglementTrainer (exp_88) and DisentangledTrainer
(exp_31/57/59):
  - NO Barlow Twins / distance-correlation / MI penalty between z_shared and
    z_specific. Shared and specific representations are allowed to be correlated.
  - Subject adversary via GRL (not explicit MI minimisation).
  - GRL alpha annealed from 0 → 1 following the DANN schedule:
        alpha(p) = 2 / (1 + exp(-10 * p)) - 1,  p = epoch / total_epochs
    This stabilises gesture learning before the adversary ramps up.

LOSO safety:
  - Channel standardisation computed from training fold data only.
  - Val / test normalised with training-fold mean_c / std_c.
  - model.eval() freezes BatchNorm running statistics (no test-subject leakage).
  - Inference: model.forward(x, return_all=False) — subject adversary and GRL
    are completely bypassed; only gesture_logits returned.
  - Subject labels never used outside the training loop.
"""

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from training.trainer import (
    WindowClassifierTrainer,
    WindowDataset,
    get_worker_init_fn,
    seed_everything,
)
from models.mate_kronecker_shared_specific import MATEKroneckerEMG


# ─────────────────────── Dataset with subject labels ─────────────────────────


class MATEDisentangledDataset(Dataset):
    """Dataset returning (window, gesture_label, subject_label) triples."""

    def __init__(
        self,
        X: np.ndarray,
        y_gesture: np.ndarray,
        y_subject: np.ndarray,
    ):
        self.X = torch.from_numpy(X).float()
        self.y_gesture = torch.from_numpy(y_gesture).long()
        self.y_subject = torch.from_numpy(y_subject).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_gesture[idx], self.y_subject[idx]


# ─────────────────────────────── Trainer ─────────────────────────────────────


class MATEKroneckerTrainer(WindowClassifierTrainer):
    """
    Trainer for MATEKroneckerEMG (Shared-Specific + Kronecker Attention).

    Expects splits dict with extra keys injected by the experiment:
        "train_subject_labels": Dict[int, np.ndarray]  gesture_id → subject indices
        "num_train_subjects":   int

    Loss:
        total = L_gesture + lambda_adv * L_subject_adv
        where L_subject_adv is computed on subject_logits produced via GRL.

    GRL alpha schedule (DANN): alpha(p) = 2/(1+exp(-10*p)) - 1
        p = current_epoch / total_epochs  ∈ [0, 1]
        alpha rises from ~0 at epoch 1 to ~1 at epoch total_epochs.

    Constructor params:
        lambda_adv:      Weight for adversarial subject loss (default 0.5).
        ecapa_channels:  ECAPA internal dimension (default 128).
        ecapa_scale:     Res2Net scale (default 4).
        embedding_dim:   ECAPA embedding dimension (default 128).
        shared_dim:      Shared prior network output dim (default 128).
        specific_dim:    Per-channel specific dim (default 32).
        ch_enc_dim:      ChannelEncoder intermediate dim (default 64).
        kron_d_k:        Kronecker attention projection dim (default 16).
        dilations:       ECAPA SE-Res2Net dilations (default [2, 3, 4]).
        se_reduction:    SE bottleneck factor (default 8).
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # Loss weight
        lambda_adv: float = 0.5,
        # ECAPA architecture
        ecapa_channels: int = 128,
        ecapa_scale: int = 4,
        embedding_dim: int = 128,
        # Disentanglement dimensions
        shared_dim: int = 128,
        specific_dim: int = 32,
        ch_enc_dim: int = 64,
        kron_d_k: int = 16,
        # ECAPA blocks
        dilations: list = None,
        se_reduction: int = 8,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)

        self.lambda_adv = lambda_adv
        self.ecapa_channels = ecapa_channels
        self.ecapa_scale = ecapa_scale
        self.embedding_dim = embedding_dim
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.ch_enc_dim = ch_enc_dim
        self.kron_d_k = kron_d_k
        self.dilations = dilations if dilations is not None else [2, 3, 4]
        self.se_reduction = se_reduction

    # ── Helper: build subject-label array aligned with _prepare_splits_arrays ─

    def _build_subject_labels_array(
        self,
        subject_labels_dict: Dict[int, np.ndarray],
        class_ids: List[int],
    ) -> np.ndarray:
        """
        Build a flat subject-label array aligned with _prepare_splits_arrays output.

        _prepare_splits_arrays iterates class_ids in sorted order (it processes
        gestures by their ID). We replicate the same iteration order so that
        subject_label[i] corresponds to gesture_label[i].
        """
        parts = []
        for gid in class_ids:
            if gid in subject_labels_dict:
                parts.append(subject_labels_dict[gid])
        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    # ── GRL alpha schedule ────────────────────────────────────────────────────

    @staticmethod
    def _grl_alpha(epoch: int, total_epochs: int) -> float:
        """
        DANN gradient reversal schedule.

        alpha(p) = 2 / (1 + exp(-10 * p)) - 1   where p = epoch / total_epochs.

        Rises from ~0.00 at epoch 0 to ~1.00 at epoch total_epochs.
        Ensures gesture classifier stabilises before adversary ramps up.

        LOSO note: alpha only controls training-time gradient reversal.
        It has no effect at inference (return_all=False, model.eval(),
        torch.no_grad()).
        """
        p = epoch / max(1, total_epochs)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train MATEKroneckerEMG with gesture CE + adversarial subject CE.

        No orthogonality constraint between z_shared and z_specific.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Prepare flat arrays from splits ────────────────────────────
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        # ── 2. Build flat subject-label array ─────────────────────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "MATEKroneckerTrainer requires 'train_subject_labels' in splits. "
                "Use the experiment that injects subject provenance via "
                "_build_splits_with_subject_labels()."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        if len(y_subject_train) != len(y_train):
            raise ValueError(
                f"Subject labels length ({len(y_subject_train)}) must match "
                f"gesture labels ({len(y_train)}). Check split builder."
            )
        self.logger.info(
            f"Training: {len(y_train)} windows, {num_train_subjects} subjects, "
            f"subject dist: {np.bincount(y_subject_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) if needed ─────────────────
        # Conv1d requires channels-first. T >> C so shape[1] > shape[2] signals
        # that the array is (N, T, C) and needs transposing.
        if X_train.ndim == 3:
            N, dim1, dim2 = X_train.shape
            if dim1 > dim2:
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val   = np.transpose(X_val,   (0, 2, 1))
                if len(X_test) > 0:
                    X_test  = np.transpose(X_test,  (0, 2, 1))
                self.logger.info(f"Transposed windows to (N, C, T): {X_train.shape}")

        in_channels = X_train.shape[1]
        num_classes  = len(class_ids)

        # ── 4. Channel standardisation (training data only) ───────────────
        # mean_c / std_c shape: (1, C, 1) — broadcast-compatible with (N, C, T).
        # Val/test are normalised using TRAINING mean/std — no test-time adaptation.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val  = self._apply_standardization(X_val,  mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation (training stats only).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Build model ────────────────────────────────────────────────
        model = MATEKroneckerEMG(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            ecapa_channels=self.ecapa_channels,
            ecapa_scale=self.ecapa_scale,
            embedding_dim=self.embedding_dim,
            shared_dim=self.shared_dim,
            specific_dim=self.specific_dim,
            ch_enc_dim=self.ch_enc_dim,
            kron_d_k=self.kron_d_k,
            dilations=self.dilations,
            se_reduction=self.se_reduction,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = model.count_parameters()
        fused_dim = self.shared_dim + in_channels * self.specific_dim
        self.logger.info(
            f"MATEKroneckerEMG: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, "
            f"shared={self.shared_dim}, specific={self.specific_dim}, "
            f"fused={fused_dim}, params={total_params:,}"
        )

        # ── 6. Datasets & DataLoaders ─────────────────────────────────────
        ds_train = MATEDisentangledDataset(X_train, y_train, y_subject_train)
        ds_val   = WindowDataset(X_val, y_val)   if len(X_val)   > 0 else None
        ds_test  = WindowDataset(X_test, y_test) if len(X_test)  > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        _num_workers   = self.cfg.num_workers

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if _num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if _num_workers > 0 else None,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if _num_workers > 0 else None,
        ) if ds_test else None

        # ── 7. Loss functions ──────────────────────────────────────────────
        # Gesture loss: optionally class-weighted to handle imbalance.
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            gesture_criterion = nn.CrossEntropyLoss()

        # Adversarial subject loss: unweighted (subjects roughly balanced in LOSO).
        subject_criterion = nn.CrossEntropyLoss()

        # ── 8. Optimizer & scheduler ───────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # ReduceLROnPlateau without verbose (removed in PyTorch 2.4+).
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 9. Training loop ───────────────────────────────────────────────
        history = {
            "train_loss":    [],
            "val_loss":      [],
            "train_acc":     [],
            "val_acc":       [],
            "gesture_loss":  [],
            "subject_loss":  [],
            "grl_alpha":     [],
        }
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        device        = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()

            # DANN alpha schedule: starts near 0, converges to 1.
            grl_alpha = self._grl_alpha(epoch, self.cfg.epochs)

            ep_total      = 0
            ep_correct    = 0
            ep_loss_total = 0.0
            ep_loss_gest  = 0.0
            ep_loss_subj  = 0.0

            for windows, gesture_labels, subject_labels in dl_train:
                windows        = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                # Training forward: all branches returned.
                outputs = model(windows, return_all=True, grl_alpha=grl_alpha)

                # Loss 1: gesture classification on fused z_shared ⊕ Z_attended
                L_gesture = gesture_criterion(
                    outputs["gesture_logits"], gesture_labels
                )

                # Loss 2: adversarial subject loss on z_shared (via GRL).
                # The GRL already reversed the gradient; we just maximise
                # subject prediction from the adversary's perspective —
                # which drives the encoder to minimise it.
                # No orthogonality/MI penalty between z_shared and z_specific.
                L_subject = subject_criterion(
                    outputs["subject_logits"], subject_labels
                )

                total_loss = L_gesture + self.lambda_adv * L_subject

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_loss_total += total_loss.item() * bs
                ep_loss_gest  += L_gesture.item() * bs
                ep_loss_subj  += L_subject.item() * bs
                preds = outputs["gesture_logits"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total   += bs

            train_loss = ep_loss_total / max(1, ep_total)
            train_acc  = ep_correct   / max(1, ep_total)
            avg_gest   = ep_loss_gest / max(1, ep_total)
            avg_subj   = ep_loss_subj / max(1, ep_total)

            # ── Validation (gesture only, no subject labels required) ──────
            # model.eval() freezes BatchNorm running stats → no test leakage.
            # return_all=False → subject adversary not invoked.
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb, return_all=False)
                        loss = gesture_criterion(logits, yb)
                        val_loss_sum += loss.item() * yb.size(0)
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
            history["gesture_loss"].append(avg_gest)
            history["subject_loss"].append(avg_subj)
            history["grl_alpha"].append(grl_alpha)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} "
                f"(gest={avg_gest:.4f}, subj_adv={avg_subj:.4f}), "
                f"acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"grl_alpha={grl_alpha:.3f}"
            )

            # ── Early stopping on val gesture loss ────────────────────────
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
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        # Restore best checkpoint (lowest val gesture loss)
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── Store trainer state (required for evaluate_numpy) ─────────────
        self.model      = model
        self.mean_c     = mean_c
        self.std_c      = std_c
        self.class_ids  = class_ids
        self.class_names = class_names
        self.in_channels = in_channels

        # ── Save training history ──────────────────────────────────────────
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png"
            )

        # ── In-fold evaluation (val & test splits) ────────────────────────
        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(device)
                    logits = model(xb, return_all=False)  # inference path
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits, axis=0)
            y_true     = np.concatenate(all_y, axis=0)
            y_pred     = logits_arr.argmax(axis=1)
            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average="macro")
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                class_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, class_labels, normalize=True,
                    filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy":         float(acc),
                "f1_macro":         float(f1),
                "report":           report,
                "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── Save model checkpoint ──────────────────────────────────────────
        model_path = self.output_dir / "mate_kronecker.pt"
        torch.save({
            "state_dict":    model.state_dict(),
            "in_channels":   in_channels,
            "num_classes":   num_classes,
            "num_subjects":  num_train_subjects,
            "class_ids":     class_ids,
            "mean":          mean_c,
            "std":           std_c,
            "architecture": {
                "ecapa_channels": self.ecapa_channels,
                "ecapa_scale":    self.ecapa_scale,
                "embedding_dim":  self.embedding_dim,
                "shared_dim":     self.shared_dim,
                "specific_dim":   self.specific_dim,
                "ch_enc_dim":     self.ch_enc_dim,
                "kron_d_k":       self.kron_d_k,
                "dilations":      self.dilations,
                "se_reduction":   self.se_reduction,
            },
            "loss_config": {
                "lambda_adv": self.lambda_adv,
            },
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        results_path = self.output_dir / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ── evaluate_numpy ────────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate on raw (N, T, C) EMG windows using training-time statistics.

        LOSO safety:
          - Transposes and normalises using training-fold mean_c / std_c.
          - Uses model in eval mode: BatchNorm frozen to training statistics.
          - Calls model.forward(x, return_all=False): subject adversary excluded.
          - No subject labels required.

        Args:
            X:          (N, T, C) raw EMG windows (not yet normalised).
            y:          (N,) class indices (0-based, aligned with class_ids).
            split_name: Label for logging and confusion matrix filename.
            visualize:  Whether to plot confusion matrix.

        Returns:
            dict with accuracy, f1_macro, report, confusion_matrix, logits.
        """
        assert self.model    is not None, "Call fit() before evaluate_numpy()"
        assert self.mean_c   is not None
        assert self.std_c    is not None
        assert self.class_ids is not None

        # Transpose (N, T, C) → (N, C, T) if needed (same logic as fit)
        X_in = X.copy().astype(np.float32)
        if X_in.ndim == 3:
            N, dim1, dim2 = X_in.shape
            if dim1 > dim2:
                X_in = np.transpose(X_in, (0, 2, 1))

        # Apply training-fold normalisation — never recompute from test data
        Xs = self._apply_standardization(X_in, self.mean_c, self.std_c)

        ds = WindowDataset(Xs, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        # Inference: model.eval() freezes BN; return_all=False skips adversary
        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                logits = self.model(xb, return_all=False)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())

        logits_arr = np.concatenate(all_logits, axis=0)
        y_true     = np.concatenate(all_y, axis=0)
        y_pred     = logits_arr.argmax(axis=1)

        acc      = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        report   = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        num_classes = len(self.class_ids)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        if visualize and self.visualizer is not None:
            class_labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, class_labels, normalize=True,
                filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy":         float(acc),
            "f1_macro":         float(f1_macro),
            "report":           report,
            "confusion_matrix": cm.tolist(),
            "logits":           logits_arr,
        }
