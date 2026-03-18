"""
Trainer for DSFE Style Bank CNN-GRU (Experiment 105).

Extends DisentangledTrainer with:
- EMA style bank: accumulates per-subject mean z_style during training
- Multi-style FiLM conditioning via style bank anchors
- GroupDRO on style-averaged logits per training subject
- Dual gesture loss (base + FiLM MixStyle) as in exp_60
- Multi-style inference: average predictions across style anchors

Loss:
    L_total = L_gesture_base                 (CE on z_content, no FiLM)
            + γ · L_gesture_mix              (CE on FiLM(z_content, z_style_mix))
            + δ · L_groupdro                 (GroupDRO on style-averaged logits)
            + α · L_subject                  (CE on z_style → subject ID)
            + β(t) · L_MI(z_content, z_style) (distance correlation)

LOSO guarantee:
    The style bank is populated ONLY from training subjects' z_style vectors.
    At inference, z_style of the test subject is never computed. The model
    uses only training-subject anchors for multi-style prediction.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from models.dsfe_style_bank_cnn_gru import DSFEStyleBankCNNGRU
from models.disentangled_cnn_gru import distance_correlation_loss, orthogonality_loss


class DSFEStyleBankTrainer(DisentangledTrainer):
    """
    Trainer for DSFE Style Bank CNN-GRU.

    Inherits data preparation, normalization, and subject label building
    from DisentangledTrainer. Overrides fit() and evaluate_numpy() to add
    EMA style bank, GroupDRO, and multi-style inference.
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        content_dim: int = 128,
        style_dim: int = 64,
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float = 0.5,
        delta: float = 0.3,
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",
        mix_alpha: float = 0.4,
        ema_momentum: float = 0.1,
        groupdro_eta: float = 0.01,
        style_bank_warmup: int = 5,
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
            gamma:               weight of FiLM MixStyle gesture loss
            delta:               weight of GroupDRO loss on style-averaged logits
            beta_anneal_epochs:  epochs over which beta ramps from 0 to beta
            mi_loss_type:        "distance_correlation" | "orthogonal" | "both"
            mix_alpha:           Beta(mix_alpha, mix_alpha) for style interpolation
            ema_momentum:        EMA momentum for style bank updates
            groupdro_eta:        GroupDRO step size for weight updates
            style_bank_warmup:   epochs before activating GroupDRO
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
        self.delta = delta
        self.mix_alpha = mix_alpha
        self.ema_momentum = ema_momentum
        self.groupdro_eta = groupdro_eta
        self.style_bank_warmup = style_bank_warmup
        self.style_bank = None  # set after training

    # ─────────────────────────── fit ───────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train DSFE Style Bank model with EMA anchors and GroupDRO.

        Expects splits to contain:
            "train":                Dict[gesture_id, np.ndarray]
            "val":                  Dict[gesture_id, np.ndarray]
            "test":                 Dict[gesture_id, np.ndarray]  (optional)
            "train_subject_labels": Dict[gesture_id, np.ndarray]
            "num_train_subjects":   int
        """
        seed_everything(self.cfg.seed)

        # 1. Build flat arrays
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # 2. Extract subject labels
        if "train_subject_labels" not in splits:
            raise ValueError(
                "DSFEStyleBankTrainer requires 'train_subject_labels' in splits."
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

        # 4. Per-channel standardization (computed on train only — LOSO clean)
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardization.")

        norm_path = self.output_dir / "normalization_stats.npz"
        np.savez_compressed(
            norm_path, mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # 5. Build model
        device = self.cfg.device
        model = DSFEStyleBankCNNGRU(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dropout=self.cfg.dropout,
            mix_alpha=self.mix_alpha,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"DSFEStyleBankCNNGRU: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, content_dim={self.content_dim}, "
            f"style_dim={self.style_dim}, mix_alpha={self.mix_alpha}, "
            f"gamma={self.gamma}, delta={self.delta}, params={total_params:,}"
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
            weight_tensor = torch.from_numpy(cw).float().to(device)
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

        # 9. Style bank and GroupDRO state
        style_bank = torch.zeros(
            num_train_subjects, self.style_dim, device=device
        )
        style_bank_initialized = torch.zeros(
            num_train_subjects, dtype=torch.bool, device=device
        )
        groupdro_weights = torch.ones(
            num_train_subjects, device=device
        ) / num_train_subjects

        # 10. Training loop
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_base_loss": [], "gesture_mix_loss": [],
            "groupdro_loss": [],
            "subject_loss": [], "mi_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        best_style_bank = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            current_beta = self.beta * min(
                1.0, epoch / max(1, self.beta_anneal_epochs)
            )

            # Activate style bank after warmup + all subjects initialized
            use_style_bank = (
                epoch > self.style_bank_warmup
                and style_bank_initialized.all()
            )

            ep_total_loss = 0.0
            ep_gest_base_loss = 0.0
            ep_gest_mix_loss = 0.0
            ep_groupdro_loss = 0.0
            ep_subject_loss = 0.0
            ep_mi_loss = 0.0
            ep_correct = 0
            ep_total = 0

            for windows, gesture_labels, subject_labels in dl_train:
                windows = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                # Forward pass — clone style bank to avoid aliasing issues
                # during EMA update and backward pass
                outputs = model(
                    windows,
                    subject_labels=subject_labels,
                    style_anchors=(
                        style_bank.clone() if use_style_bank else None
                    ),
                )

                # Save z_style for EMA update BEFORE backward frees graph
                z_style_for_ema = outputs["z_style"].detach().clone()

                # ── Losses ──

                # Base gesture loss (no FiLM, regularizer)
                L_base = gesture_criterion(
                    outputs["gesture_logits_base"], gesture_labels
                )

                # FiLM + MixStyle gesture loss
                L_mix = gesture_criterion(
                    outputs["gesture_logits_mix"], gesture_labels
                )

                # Subject classifier loss
                L_subject = subject_criterion(
                    outputs["subject_logits"], subject_labels
                )

                # MI minimization (distance correlation)
                if self.mi_loss_type == "distance_correlation":
                    L_MI = distance_correlation_loss(
                        outputs["z_content"], outputs["z_style"]
                    )
                elif self.mi_loss_type == "orthogonal":
                    L_MI = orthogonality_loss(
                        outputs["z_content"], outputs["z_style"]
                    )
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

                # GroupDRO on style-averaged logits per training subject
                L_groupdro = torch.tensor(0.0, device=device)
                if (
                    use_style_bank
                    and outputs["style_bank_logits"] is not None
                ):
                    # Average logits across all K style anchors
                    avg_logits = outputs["style_bank_logits"].mean(dim=0)

                    # Per-subject CE loss on style-averaged logits
                    unique_subjects = subject_labels.unique()
                    active_losses = []
                    active_indices = []

                    for s_idx in unique_subjects:
                        mask = (subject_labels == s_idx)
                        if mask.sum() >= 2:
                            loss_s = F.cross_entropy(
                                avg_logits[mask], gesture_labels[mask]
                            )
                            active_losses.append(loss_s)
                            active_indices.append(s_idx.item())

                    if active_losses:
                        active_losses_t = torch.stack(active_losses)
                        active_idx = torch.tensor(
                            active_indices, dtype=torch.long, device=device
                        )

                        # Update GroupDRO weights (no gradient)
                        with torch.no_grad():
                            cur_w = groupdro_weights[active_idx].clone()
                            cur_w = cur_w * torch.exp(
                                self.groupdro_eta * active_losses_t.detach()
                            )
                            cur_w = cur_w / (cur_w.sum() + 1e-10)
                            groupdro_weights[active_idx] = cur_w
                            groupdro_weights.div_(
                                groupdro_weights.sum() + 1e-10
                            )

                        # Weighted loss — weights are detached constants
                        L_groupdro = (
                            groupdro_weights[active_idx] * active_losses_t
                        ).sum()

                # Combined loss
                total_loss = (
                    L_base
                    + self.gamma * L_mix
                    + self.delta * L_groupdro
                    + self.alpha * L_subject
                    + current_beta * L_MI
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # ── Update EMA style bank (after backward) ──
                with torch.no_grad():
                    for s_idx in subject_labels.unique():
                        mask = (subject_labels == s_idx)
                        z_s_mean = z_style_for_ema[mask].mean(dim=0)
                        s = s_idx.item()
                        if style_bank_initialized[s]:
                            style_bank[s] = (
                                (1 - self.ema_momentum) * style_bank[s]
                                + self.ema_momentum * z_s_mean
                            )
                        else:
                            style_bank[s] = z_s_mean
                            style_bank_initialized[s] = True

                # Accumulate metrics
                bs = windows.size(0)
                ep_total_loss += total_loss.item() * bs
                ep_gest_base_loss += L_base.item() * bs
                ep_gest_mix_loss += L_mix.item() * bs
                ep_groupdro_loss += L_groupdro.item() * bs
                ep_subject_loss += L_subject.item() * bs
                ep_mi_loss += L_MI.item() * bs

                # Accuracy from base path (inference-time metric)
                preds = outputs["gesture_logits_base"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total += bs

            train_loss = ep_total_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            avg_gest_base = ep_gest_base_loss / max(1, ep_total)
            avg_gest_mix = ep_gest_mix_loss / max(1, ep_total)
            avg_groupdro = ep_groupdro_loss / max(1, ep_total)
            avg_subject = ep_subject_loss / max(1, ep_total)
            avg_mi = ep_mi_loss / max(1, ep_total)

            # ── Validation: base path (for early stopping) ──
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        # Base path for stable early stopping metric
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
            history["groupdro_loss"].append(avg_groupdro)
            history["subject_loss"].append(avg_subject)
            history["mi_loss"].append(avg_mi)

            sb_status = "active" if use_style_bank else "warmup"
            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} "
                f"(base={avg_gest_base:.4f}, mix={avg_gest_mix:.4f}, "
                f"gdro={avg_groupdro:.4f}, subj={avg_subject:.4f}, "
                f"MI={avg_mi:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"beta={current_beta:.4f}, style_bank={sb_status}"
            )

            # Early stopping on val gesture loss (base path)
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    # Save style bank alongside best model
                    best_style_bank = style_bank.clone().cpu()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        # Restore best model and corresponding style bank
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
        if best_style_bank is not None:
            style_bank = best_style_bank.to(device)

        # Store trainer state
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # Store style bank for inference
        if style_bank_initialized.all():
            self.style_bank = style_bank.cpu()
            self.logger.info(
                f"Style bank ready: {num_train_subjects} anchors, "
                f"dim={self.style_dim}"
            )
        else:
            uninit = (~style_bank_initialized).sum().item()
            self.logger.warning(
                f"Style bank incomplete: {uninit}/{num_train_subjects} "
                "subjects uninitialized. Falling back to base path inference."
            )
            self.style_bank = None

        # Save training history
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png"
            )

        # ── In-fold evaluation ──
        results = {"class_ids": class_ids, "class_names": class_names}

        def eval_loader(dloader, split_name):
            """Evaluate a DataLoader using multi-style inference if available."""
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                sa = (
                    self.style_bank.to(device)
                    if self.style_bank is not None else None
                )
                for xb, yb in dloader:
                    xb = xb.to(device)
                    logits = model(xb, style_anchors=sa)
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

        results["val"] = eval_loader(dl_val, "val")
        results["test"] = eval_loader(dl_test, "test")

        # Save model checkpoint
        model_path = self.output_dir / "dsfe_style_bank_cnn_gru.pt"
        save_dict = {
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
            "delta": self.delta,
            "training_config": asdict(self.cfg),
        }
        if self.style_bank is not None:
            save_dict["style_bank"] = self.style_bank
        torch.save(save_dict, model_path)
        self.logger.info(f"Model saved: {model_path}")

        results_path = self.output_dir / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ─────────────────────── evaluate_numpy ─────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Multi-style inference using style bank anchors.

        At test time, z_style of the test subject is NOT computed.
        Instead, predictions are averaged across all training-subject
        style anchors (condition averaging).

        If style bank is not available, falls back to base path
        (identical to DisentangledTrainer.evaluate_numpy).
        """
        assert self.model is not None, "Model is not trained/loaded"
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None and self.class_names is not None

        device = self.cfg.device

        # Transpose (N, T, C) → (N, C, T) if needed
        X_input = X.copy()
        if X_input.ndim == 3:
            N, dim1, dim2 = X_input.shape
            if dim1 > dim2:
                X_input = np.transpose(X_input, (0, 2, 1))

        # Standardize using training statistics (LOSO clean)
        Xs = self._apply_standardization(X_input, self.mean_c, self.std_c)

        # Create DataLoader
        ds = WindowDataset(Xs, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        # Multi-style inference with style bank
        self.model.eval()
        all_logits, all_y = [], []

        style_anchors = None
        if self.style_bank is not None:
            style_anchors = self.style_bank.to(device)

        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                # model(xb, style_anchors=...) returns:
                #   - averaged logits across anchors if style_anchors provided
                #   - base path logits otherwise (fallback)
                logits = self.model(xb, style_anchors=style_anchors)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y, axis=0)
        y_pred = logits.argmax(axis=1)

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

        inference_mode = (
            "multi-style" if self.style_bank is not None else "base-path"
        )
        self.logger.info(
            f"[{split_name}] Inference mode: {inference_mode}, "
            f"Accuracy={acc:.4f}, F1-macro={f1_macro:.4f}"
        )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "logits": logits,
        }
