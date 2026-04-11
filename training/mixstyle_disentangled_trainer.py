"""
Trainer for MixStyle Disentangled CNN-GRU (Experiment 60).

Extends DisentangledTrainer with:
- Dual gesture loss: base path (no FiLM) + mixed-style path (FiLM-conditioned)
- Style mixing only within training batches — test subject never involved
- Inference uses base path only (no FiLM, no test-subject style)

Loss:
    L_total = L_gesture_base                 (CE on z_content, no FiLM)
            + γ · L_gesture_mix              (CE on FiLM(z_content, z_style_mix))
            + α · L_subject                  (CE on z_style → subject ID)
            + β(t) · L_MI(z_content, z_style) (distance correlation)

LOSO guarantee:
    The training DataLoader is built only from training-subject windows.
    subject_labels passed to model.forward() are indices into the train-subject
    list — test subject has no index and never appears in any batch.
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
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from models.mixstyle_disentangled_cnn_gru import MixStyleDisentangledCNNGRU
from models.disentangled_cnn_gru import distance_correlation_loss, orthogonality_loss


class MixStyleDisentangledTrainer(DisentangledTrainer):
    """
    Trainer for MixStyle-augmented content-style disentangled CNN-GRU.

    Inherits data preparation, normalization, early stopping, evaluate_numpy,
    and artifact saving from DisentangledTrainer. Only fit() is overridden to:
        1. Instantiate MixStyleDisentangledCNNGRU instead of DisentangledCNNGRU.
        2. Pass subject_labels into model.forward() for in-batch style mixing.
        3. Apply dual gesture loss (base + γ·mix).

    The evaluate_numpy() method is inherited unchanged because MixStyleDisentangledCNNGRU
    returns a plain gesture_logits tensor in eval mode — identical interface.
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        content_dim: int = 128,
        style_dim: int = 64,
        alpha: float = 0.5,    # weight of subject classification loss
        beta: float = 0.1,     # weight of MI minimization loss (annealed)
        gamma: float = 0.5,    # weight of mixed-style gesture loss
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",
        mix_alpha: float = 0.4,   # Beta distribution parameter for style mixing
    ):
        """
        Args:
            train_cfg:           TrainingConfig
            logger:              Python logger
            output_dir:          Path for checkpoints / artifacts
            visualizer:          optional Visualizer
            content_dim:         dimensionality of z_content
            style_dim:           dimensionality of z_style
            alpha:               weight of subject classifier loss
            beta:                weight of MI (distance-correlation) loss
            gamma:               weight of mixed-style path gesture loss
            beta_anneal_epochs:  epochs over which beta ramps from 0 to beta
            mi_loss_type:        "distance_correlation" | "orthogonal" | "both"
            mix_alpha:           Beta(mix_alpha, mix_alpha) for style interpolation
        """
        super().__init__(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=visualizer,
            content_dim=content_dim,
            style_dim=style_dim,
            alpha=alpha,
            beta=beta,
            beta_anneal_epochs=beta_anneal_epochs,
            mi_loss_type=mi_loss_type,
        )
        self.gamma = gamma
        self.mix_alpha = mix_alpha

    # ─────────────────────────── fit ───────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train MixStyle-disentangled model.

        Expects splits to contain:
            "train":               Dict[gesture_id, np.ndarray]
            "val":                 Dict[gesture_id, np.ndarray]
            "test":                Dict[gesture_id, np.ndarray]  (optional)
            "train_subject_labels": Dict[gesture_id, np.ndarray]  int subject indices
            "num_train_subjects":  int
        """
        seed_everything(self.cfg.seed)

        # 1. Build flat arrays — same as parent
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # 2. Extract subject labels aligned to class_ids order
        if "train_subject_labels" not in splits:
            raise ValueError(
                "MixStyleDisentangledTrainer requires 'train_subject_labels' in splits. "
                "Use the experiment file that injects subject provenance."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        assert len(y_subject_train) == len(y_train), (
            f"Subject labels ({len(y_subject_train)}) must match gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Training: {num_train_subjects} subjects, "
            f"subject distribution: {np.bincount(y_subject_train).tolist()}"
        )

        # 3. Transpose (N, T, C) → (N, C, T) if needed
        if X_train.ndim == 3:
            N, dim1, dim2 = X_train.shape
            if dim1 > dim2:
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # 4. Per-channel standardisation (computed on train only — LOSO clean)
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation.")

        norm_path = self.output_dir / "normalization_stats.npz"
        np.savez_compressed(
            norm_path, mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # 5. Build MixStyleDisentangledCNNGRU
        model = MixStyleDisentangledCNNGRU(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dropout=self.cfg.dropout,
            mix_alpha=self.mix_alpha,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"MixStyleDisentangledCNNGRU: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, content_dim={self.content_dim}, "
            f"style_dim={self.style_dim}, mix_alpha={self.mix_alpha}, "
            f"gamma={self.gamma}, params={total_params:,}"
        )

        # 6. Datasets and loaders
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
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

        # 7. Loss functions
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

        # 8. Optimizer + scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # 9. Training loop
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_base_loss": [], "gesture_mix_loss": [],
            "subject_loss": [], "mi_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            # MI loss beta annealing: gradually increase from 0 to self.beta
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))

            ep_total_loss = 0.0
            ep_gest_base_loss = 0.0
            ep_gest_mix_loss = 0.0
            ep_subject_loss = 0.0
            ep_mi_loss = 0.0
            ep_correct = 0
            ep_total = 0

            for windows, gesture_labels, subject_labels in dl_train:
                windows = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                # Forward: training mode passes subject_labels for in-batch style mixing
                outputs = model(windows, subject_labels=subject_labels)
                # outputs is a dict (model.training=True)

                # Base gesture loss: z_content → classifier (no FiLM)
                L_gest_base = gesture_criterion(
                    outputs["gesture_logits_base"], gesture_labels
                )

                # Mixed-style gesture loss: FiLM(z_content, z_style_mix) → classifier
                L_gest_mix = gesture_criterion(
                    outputs["gesture_logits_mix"], gesture_labels
                )

                # Subject classification loss (only on z_style, not z_style_mix)
                L_subject = subject_criterion(outputs["subject_logits"], subject_labels)

                # MI minimization between z_content and z_style
                if self.mi_loss_type == "distance_correlation":
                    L_MI = distance_correlation_loss(
                        outputs["z_content"], outputs["z_style"]
                    )
                elif self.mi_loss_type == "orthogonal":
                    L_MI = orthogonality_loss(outputs["z_content"], outputs["z_style"])
                elif self.mi_loss_type == "both":
                    L_MI = (
                        distance_correlation_loss(
                            outputs["z_content"], outputs["z_style"]
                        )
                        + 0.1 * orthogonality_loss(
                            outputs["z_content"], outputs["z_style"]
                        )
                    )
                else:
                    L_MI = distance_correlation_loss(
                        outputs["z_content"], outputs["z_style"]
                    )

                # Combined loss
                total_loss = (
                    L_gest_base
                    + self.gamma * L_gest_mix
                    + self.alpha * L_subject
                    + current_beta * L_MI
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_total_loss += total_loss.item() * bs
                ep_gest_base_loss += L_gest_base.item() * bs
                ep_gest_mix_loss += L_gest_mix.item() * bs
                ep_subject_loss += L_subject.item() * bs
                ep_mi_loss += L_MI.item() * bs

                # Accuracy from base path (this is the inference-time prediction)
                preds = outputs["gesture_logits_base"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total += bs

            train_loss = ep_total_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            avg_gest_base = ep_gest_base_loss / max(1, ep_total)
            avg_gest_mix = ep_gest_mix_loss / max(1, ep_total)
            avg_subject = ep_subject_loss / max(1, ep_total)
            avg_mi = ep_mi_loss / max(1, ep_total)

            # Validation: base path only (mirrors inference, no FiLM)
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        # eval mode → returns gesture_logits_base tensor (no FiLM)
                        logits = model(xb)
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
            history["gesture_base_loss"].append(avg_gest_base)
            history["gesture_mix_loss"].append(avg_gest_mix)
            history["subject_loss"].append(avg_subject)
            history["mi_loss"].append(avg_mi)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} "
                f"(gest_base={avg_gest_base:.4f}, gest_mix={avg_gest_mix:.4f}, "
                f"subj={avg_subject:.4f}, MI={avg_mi:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | beta={current_beta:.4f}"
            )

            # Early stopping on val gesture loss (base path = inference metric)
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # Store trainer state (same attributes as DisentangledTrainer for compatibility)
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # Save training history
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── In-fold evaluation (val + test from training-time splits) ──

        results = {"class_ids": class_ids, "class_names": class_names}

        def eval_loader(dloader, split_name):
            """Evaluate a DataLoader using base path (no FiLM)."""
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(device)
                    logits = model(xb)  # eval mode → base path tensor
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
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
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

        results["val"] = eval_loader(dl_val, "val")
        results["test"] = eval_loader(dl_test, "test")

        # Save model checkpoint
        model_path = self.output_dir / "mixstyle_disentangled_cnn_gru.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "num_subjects": num_train_subjects,
            "class_ids": class_ids,
            "mean": mean_c,
            "std": std_c,
            "window_size": window_size,
            "content_dim": self.content_dim,
            "style_dim": self.style_dim,
            "mix_alpha": self.mix_alpha,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        results_path = self.output_dir / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy is inherited from DisentangledTrainer unchanged:
    # it calls model(xb) in eval mode, which returns gesture_logits_base tensor.
    # The interface is identical regardless of whether the model has FiLM or not.
