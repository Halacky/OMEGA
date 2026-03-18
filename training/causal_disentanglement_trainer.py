"""
Trainer for Causal ECAPA-TDNN with content/style disentanglement.

Extends WindowClassifierTrainer with multi-loss training:
  1. Gesture classification (CrossEntropy on content branch)
  2. Subject classification (CrossEntropy on style branch, training only)
  3. Causal aggregation (per-class cross-subject variance minimization)
  4. Barlow Twins redundancy reduction (content–style decorrelation)
  5. Reconstruction (content + style → shared embedding, prevents collapse)

Evaluation uses only the content branch — no subject labels or adaptation.

LOSO safety:
  - Channel standardization computed from training data only.
  - Subject labels used during training only (not in val/test/inference).
  - model.eval() freezes BatchNorm to training-time running statistics.
  - Validation early stopping monitors gesture loss, not subject loss.
"""

import json
import logging
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
from models.causal_ecapa_tdnn import (
    CausalECAPATDNN,
    barlow_twins_redundancy_loss,
    causal_aggregation_loss,
)


# ─────────────────────── dataset with subject labels ─────────────────────────

class CausalDisentangledDataset(Dataset):
    """Dataset returning (window, gesture_label, subject_label)."""

    def __init__(self, X: np.ndarray, y_gesture: np.ndarray, y_subject: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y_gesture = torch.from_numpy(y_gesture).long()
        self.y_subject = torch.from_numpy(y_subject).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_gesture[idx], self.y_subject[idx]


# ──────────────────────────── trainer ────────────────────────────────────────

class CausalDisentanglementTrainer(WindowClassifierTrainer):
    """
    Trainer for CausalECAPATDNN with causal disentanglement.

    Expects splits dict to contain extra keys (injected by experiment):
        "train_subject_labels": Dict[int, np.ndarray]  — gesture_id → subject indices
        "num_train_subjects": int

    Hyperparameters:
        alpha:        Weight for subject classification loss.
        lambda_causal: Weight for causal aggregation loss (annealed).
        lambda_barlow: Weight for Barlow Twins redundancy loss (annealed).
        lambda_recon:  Weight for reconstruction loss.
        anneal_epochs: Epochs over which causal/barlow ramp from 0 to full.

    ECAPA-TDNN architecture params:
        channels, scale, embedding_dim, content_dim, style_dim, dilations,
        se_reduction — forwarded to CausalECAPATDNN constructor.
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # --- disentanglement loss weights ---
        alpha: float = 0.5,
        lambda_causal: float = 0.5,
        lambda_barlow: float = 0.1,
        lambda_recon: float = 0.1,
        anneal_epochs: int = 10,
        # --- ECAPA-TDNN architecture ---
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        content_dim: int = 128,
        style_dim: int = 64,
        dilations: list = None,
        se_reduction: int = 8,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)

        # Loss weights
        self.alpha = alpha
        self.lambda_causal = lambda_causal
        self.lambda_barlow = lambda_barlow
        self.lambda_recon = lambda_recon
        self.anneal_epochs = anneal_epochs

        # Architecture params
        self.channels = channels
        self.scale = scale
        self.embedding_dim = embedding_dim
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.dilations = dilations if dilations is not None else [2, 3, 4]
        self.se_reduction = se_reduction

    # ── helper: build subject labels aligned with _prepare_splits_arrays ──

    def _build_subject_labels_array(
        self,
        subject_labels_dict: Dict[int, np.ndarray],
        class_ids: List[int],
    ) -> np.ndarray:
        """
        Build flat subject-label array aligned with _prepare_splits_arrays.

        _prepare_splits_arrays iterates class_ids in sorted order and
        concatenates per-gesture arrays. We replicate that exact order.
        """
        parts = []
        for gid in class_ids:
            if gid in subject_labels_dict:
                parts.append(subject_labels_dict[gid])
        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """Train CausalECAPATDNN with multi-loss optimization."""
        seed_everything(self.cfg.seed)

        # ── 1. Prepare arrays ────────────────────────────────────────────
        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        # ── 2. Subject labels ────────────────────────────────────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "CausalDisentanglementTrainer requires 'train_subject_labels' "
                "in splits. Use the experiment file that injects subject provenance."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        assert len(y_subject_train) == len(y_train), (
            f"Subject labels ({len(y_subject_train)}) must match "
            f"gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Subject labels: {num_train_subjects} subjects, "
            f"distribution: {np.bincount(y_subject_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) ─────────────────────────
        if X_train.ndim == 3:
            N, dim1, dim2 = X_train.shape
            if dim1 > dim2:  # T > C → need transpose
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(f"Transposed to (N, C, T): {X_train.shape}")

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 4. Channel standardization (training data only) ─────────────
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardization (training stats).")

        # Save normalization stats
        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Create model ──────────────────────────────────────────────
        model = CausalECAPATDNN(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            channels=self.channels,
            scale=self.scale,
            embedding_dim=self.embedding_dim,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dilations=self.dilations,
            dropout=self.cfg.dropout,
            se_reduction=self.se_reduction,
        ).to(self.cfg.device)

        total_params = model.count_parameters()
        self.logger.info(
            f"CausalECAPATDNN: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, C={self.channels}, "
            f"content={self.content_dim}, style={self.style_dim}, "
            f"params={total_params:,}"
        )

        # ── 6. Datasets & DataLoaders ────────────────────────────────────
        ds_train = CausalDisentangledDataset(X_train, y_train, y_subject_train)
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        ) if ds_test else None

        # ── 7. Loss functions ────────────────────────────────────────────
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            gesture_criterion = nn.CrossEntropyLoss()

        subject_criterion = nn.CrossEntropyLoss()

        # ── 8. Optimizer & scheduler ─────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 9. Training loop ────────────────────────────────────────────
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_loss": [], "subject_loss": [],
            "causal_loss": [], "barlow_loss": [], "recon_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()

            # Anneal causal and barlow losses (ramp 0 → full over anneal_epochs)
            anneal_factor = min(1.0, epoch / max(1, self.anneal_epochs))

            ep_total = 0
            ep_correct = 0
            ep_loss_total = 0.0
            ep_loss_gesture = 0.0
            ep_loss_subject = 0.0
            ep_loss_causal = 0.0
            ep_loss_barlow = 0.0
            ep_loss_recon = 0.0

            for windows, gesture_labels, subject_labels in dl_train:
                windows = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                outputs = model(windows, return_all=True)

                # Loss 1: Gesture classification (content branch)
                L_gesture = gesture_criterion(
                    outputs["gesture_logits"], gesture_labels
                )

                # Loss 2: Subject classification (style branch)
                L_subject = subject_criterion(
                    outputs["subject_logits"], subject_labels
                )

                # Loss 3: Causal aggregation (per-class cross-subject invariance)
                L_causal = causal_aggregation_loss(
                    outputs["z_content"], gesture_labels, subject_labels
                )

                # Loss 4: Barlow Twins redundancy reduction
                L_barlow = barlow_twins_redundancy_loss(
                    outputs["z_content"], outputs["z_style"]
                )

                # Loss 5: Reconstruction (content + style → embedding)
                L_recon = nn.functional.mse_loss(
                    outputs["reconstruction"], outputs["embedding"]
                )

                # Total loss
                total_loss = (
                    L_gesture
                    + self.alpha * L_subject
                    + anneal_factor * self.lambda_causal * L_causal
                    + anneal_factor * self.lambda_barlow * L_barlow
                    + self.lambda_recon * L_recon
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_loss_total += total_loss.item() * bs
                ep_loss_gesture += L_gesture.item() * bs
                ep_loss_subject += L_subject.item() * bs
                ep_loss_causal += L_causal.item() * bs
                ep_loss_barlow += L_barlow.item() * bs
                ep_loss_recon += L_recon.item() * bs
                preds = outputs["gesture_logits"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total += bs

            train_loss = ep_loss_total / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            avg_gesture = ep_loss_gesture / max(1, ep_total)
            avg_subject = ep_loss_subject / max(1, ep_total)
            avg_causal = ep_loss_causal / max(1, ep_total)
            avg_barlow = ep_loss_barlow / max(1, ep_total)
            avg_recon = ep_loss_recon / max(1, ep_total)

            # ── Validation (gesture only, no subject labels) ─────────────
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)  # eval → gesture_logits only
                        loss = gesture_criterion(logits, yb)
                        val_loss_sum += loss.item() * yb.size(0)
                        preds = logits.argmax(dim=1)
                        val_correct += (preds == yb).sum().item()
                        val_total += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["gesture_loss"].append(avg_gesture)
            history["subject_loss"].append(avg_subject)
            history["causal_loss"].append(avg_causal)
            history["barlow_loss"].append(avg_barlow)
            history["recon_loss"].append(avg_recon)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} "
                f"(gest={avg_gesture:.4f}, subj={avg_subject:.4f}, "
                f"causal={avg_causal:.4f}, barlow={avg_barlow:.4f}, "
                f"recon={avg_recon:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"anneal={anneal_factor:.2f}"
            )

            # ── Early stopping on val gesture loss ───────────────────────
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

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── Store trainer state (required for evaluate_numpy) ────────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── Save training history ────────────────────────────────────────
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── In-fold evaluation (val & test) ──────────────────────────────
        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(device)
                    logits = model(xb)  # eval → gesture_logits only
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits, axis=0)
            y_true = np.concatenate(all_y, axis=0)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(
                y_true, y_pred, labels=np.arange(num_classes)
            )
            if self.visualizer is not None:
                class_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, class_labels, normalize=True,
                    filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report": report,
                "confusion_matrix": cm.tolist(),
            }

        results["val"] = _eval_loader(dl_val, "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── Save model checkpoint ────────────────────────────────────────
        model_path = self.output_dir / "causal_ecapa_tdnn.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "num_subjects": num_train_subjects,
            "class_ids": class_ids,
            "mean": mean_c,
            "std": std_c,
            "window_size": window_size,
            "architecture": {
                "channels": self.channels,
                "scale": self.scale,
                "embedding_dim": self.embedding_dim,
                "content_dim": self.content_dim,
                "style_dim": self.style_dim,
                "dilations": self.dilations,
                "se_reduction": self.se_reduction,
            },
            "loss_weights": {
                "alpha": self.alpha,
                "lambda_causal": self.lambda_causal,
                "lambda_barlow": self.lambda_barlow,
                "lambda_recon": self.lambda_recon,
                "anneal_epochs": self.anneal_epochs,
            },
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        results_path = self.output_dir / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ── evaluate_numpy ───────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate using only gesture classification from content branch.

        No subject labels needed. Uses training-time normalization stats.
        Model is in eval mode — BatchNorm frozen to training statistics.

        Args:
            X: (N, T, C) raw EMG windows.
            y: (N,) class indices (0-based).
            split_name: label for logging/saving.
            visualize: whether to plot confusion matrix.

        Returns:
            dict with accuracy, f1_macro, report, confusion_matrix, logits.
        """
        assert self.model is not None, "Model is not trained/loaded"
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None and self.class_names is not None

        # Transpose (N, T, C) → (N, C, T) if needed
        X_input = X.copy().astype(np.float32)
        if X_input.ndim == 3:
            N, dim1, dim2 = X_input.shape
            if dim1 > dim2:
                X_input = np.transpose(X_input, (0, 2, 1))

        # Standardize with training-time stats (NO test stats computed)
        Xs = self._apply_standardization(X_input, self.mean_c, self.std_c)

        # Inference: content branch only
        ds = WindowDataset(Xs, y)
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
                logits = self.model(xb)  # eval → gesture_logits only
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())

        logits_arr = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y, axis=0)
        y_pred = logits_arr.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(
            y_true, y_pred, labels=np.arange(len(self.class_ids))
        )

        if visualize and self.visualizer is not None:
            class_labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, class_labels, normalize=True,
                filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "logits": logits_arr,
        }
