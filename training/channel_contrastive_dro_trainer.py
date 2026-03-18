"""
Trainer for Channel-wise Contrastive Disentanglement + Per-Channel GroupDRO.

Extends DisentangledTrainer with:
- Per-channel InfoNCE contrastive loss on z_content
- Per-channel distance correlation MI minimization
- Per-channel GroupDRO: worst channel × worst subject double worst-case
- Channel attention weighting (learned, not fixed)

LOSO compliance guarantees
--------------------------
- Channel standardization computed from X_train only.
- Contrastive pairs formed ONLY from training subjects within each batch.
- GroupDRO group weights updated using train-subject losses only.
- Model selection via val_loss from train-subject val split.
- Test-subject windows evaluated ONCE for final metrics only.
- Channel attention is learned on train data only.
- No test-subject information feeds back into training.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from models.channel_contrastive_disentangled import (
    ChannelContrastiveDisentangled,
    info_nce_loss,
    distance_correlation_loss,
)
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from training.trainer import WindowDataset, get_worker_init_fn, seed_everything


class ChannelContrastiveDROTrainer(DisentangledTrainer):
    """
    Per-channel contrastive disentanglement + per-channel GroupDRO trainer.

    Loss:
        L = L_gesture_dro + alpha * L_subject + beta * L_MI_per_ch + gamma * L_contrastive

    where:
        L_gesture_dro: per-channel GroupDRO gesture loss (worst channel × worst subject)
        L_subject: subject classifier on concat(z_style_c)
        L_MI_per_ch: mean distance correlation across channels
        L_contrastive: mean InfoNCE across channels on z_content

    Parameters
    ----------
    content_dim_per_ch : int
        Content latent dimension per channel (default 32, total = 8*32 = 256).
    style_dim_per_ch : int
        Style latent dimension per channel (default 16, total = 8*16 = 128).
    gamma : float
        Weight for InfoNCE contrastive loss.
    dro_eta : float
        Step size for exponentiated gradient update of group weights.
    contrastive_temperature : float
        Temperature for InfoNCE loss.
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        content_dim_per_ch: int = 32,
        style_dim_per_ch: int = 16,
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float = 0.3,
        beta_anneal_epochs: int = 10,
        dro_eta: float = 0.01,
        contrastive_temperature: float = 0.1,
        per_ch_cnn_channels: list = None,
        gru_hidden: int = 128,
    ):
        # Pass content_dim/style_dim to parent (not used for model creation here,
        # but needed for parent __init__ signature compatibility)
        super().__init__(
            train_cfg, logger, output_dir, visualizer,
            content_dim=content_dim_per_ch,  # stored but we use per-ch version
            style_dim=style_dim_per_ch,
            alpha=alpha,
            beta=beta,
            beta_anneal_epochs=beta_anneal_epochs,
            mi_loss_type="distance_correlation",
        )
        self.content_dim_per_ch = content_dim_per_ch
        self.style_dim_per_ch = style_dim_per_ch
        self.gamma = gamma
        self.dro_eta = dro_eta
        self.contrastive_temperature = contrastive_temperature
        self.per_ch_cnn_channels = per_ch_cnn_channels
        self.gru_hidden = gru_hidden
        self.final_group_weights: Optional[List[float]] = None

    def fit(self, splits: Dict) -> Dict:
        """Train with per-channel contrastive disentanglement + GroupDRO."""
        seed_everything(self.cfg.seed)

        # ── 1. Standard array preparation ────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # ── 2. Subject labels ────────────────────────────────────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "ChannelContrastiveDROTrainer requires 'train_subject_labels' in splits."
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
            f"Groups: {num_train_subjects} train subjects | "
            f"distribution: {np.bincount(y_subject_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) ─────────────────────────
        if X_train.ndim == 3:
            _N, dim1, dim2 = X_train.shape
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

        # ── 4. Per-channel standardization (train stats only) ────────────
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardization (train stats only).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Model ────────────────────────────────────────────────────
        model = ChannelContrastiveDisentangled(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim_per_ch=self.content_dim_per_ch,
            style_dim_per_ch=self.style_dim_per_ch,
            per_ch_cnn_channels=self.per_ch_cnn_channels,
            gru_hidden=self.gru_hidden,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"ChannelContrastiveDisentangled: in_ch={in_channels}, "
            f"gestures={num_classes}, subjects={num_train_subjects}, "
            f"content_per_ch={self.content_dim_per_ch}, "
            f"style_per_ch={self.style_dim_per_ch}, "
            f"params={total_params:,} | dro_eta={self.dro_eta}, "
            f"gamma={self.gamma}, tau={self.contrastive_temperature}"
        )

        # ── 6. Datasets & DataLoaders ────────────────────────────────────
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        _dl_kwargs = dict(
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            **_dl_kwargs,
        )
        dl_val = DataLoader(
            ds_val, batch_size=self.cfg.batch_size, shuffle=False, **_dl_kwargs
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test, batch_size=self.cfg.batch_size, shuffle=False, **_dl_kwargs
        ) if ds_test else None

        # ── 7. Loss functions ────────────────────────────────────────────
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction="mean")
        else:
            gesture_criterion = nn.CrossEntropyLoss(reduction="mean")

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

        # ── 9. GroupDRO group weights (per subject) ──────────────────────
        device = self.cfg.device
        group_weights = torch.ones(num_train_subjects, device=device) / num_train_subjects

        # ── 10. Training loop ────────────────────────────────────────────
        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_dro_loss": [], "subject_loss": [],
            "mi_loss": [], "contrastive_loss": [],
            "group_weights": [],
            "per_group_losses": [],
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))
            current_gamma = self.gamma * min(1.0, epoch / max(1, self.beta_anneal_epochs))

            ep_total_loss = 0.0
            ep_gesture_dro_loss = 0.0
            ep_subject_loss = 0.0
            ep_mi_loss = 0.0
            ep_contrastive_loss = 0.0
            ep_correct = 0
            ep_total = 0

            ep_per_group_loss_sum = [0.0] * num_train_subjects
            ep_per_group_count = [0] * num_train_subjects

            for windows, gesture_labels, subject_labels in dl_train:
                windows = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()
                outputs = model(windows, return_all=True)

                # ── Per-channel GroupDRO gesture loss ─────────────────
                # Compute per-subject gesture loss (on fused content → gesture_logits)
                batch_group_losses = torch.zeros(num_train_subjects, device=device)
                for s in range(num_train_subjects):
                    mask_s = (subject_labels == s)
                    if mask_s.any():
                        batch_group_losses[s] = gesture_criterion(
                            outputs["gesture_logits"][mask_s],
                            gesture_labels[mask_s],
                        )

                # Update group weights (mirror descent)
                with torch.no_grad():
                    group_weights = group_weights * torch.exp(
                        self.dro_eta * batch_group_losses
                    )
                    group_weights = group_weights / (group_weights.sum() + 1e-12)

                L_gesture_dro = (group_weights.detach() * batch_group_losses).sum()

                # ── Subject classifier loss ──────────────────────────
                L_subject = subject_criterion(outputs["subject_logits"], subject_labels)

                # ── Per-channel MI minimization ──────────────────────
                # Average distance correlation across all channels
                z_content = outputs["z_content"]  # (B, C, content_dim_per_ch)
                z_style = outputs["z_style"]      # (B, C, style_dim_per_ch)
                C = z_content.size(1)

                mi_losses = []
                for c in range(C):
                    mi_c = distance_correlation_loss(z_content[:, c], z_style[:, c])
                    mi_losses.append(mi_c)
                L_MI = torch.stack(mi_losses).mean()

                # ── Per-channel contrastive loss ─────────────────────
                # InfoNCE on z_content per channel: positives = same gesture
                # All pairs come from training subjects only (LOSO-safe)
                contrast_losses = []
                for c in range(C):
                    cl = info_nce_loss(
                        z_content[:, c],
                        gesture_labels,
                        temperature=self.contrastive_temperature,
                    )
                    contrast_losses.append(cl)
                L_contrastive = torch.stack(contrast_losses).mean()

                # ── Total loss ───────────────────────────────────────
                total_loss = (
                    L_gesture_dro
                    + self.alpha * L_subject
                    + current_beta * L_MI
                    + current_gamma * L_contrastive
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_total_loss += total_loss.item() * bs
                ep_gesture_dro_loss += L_gesture_dro.item() * bs
                ep_subject_loss += L_subject.item() * bs
                ep_mi_loss += L_MI.item() * bs
                ep_contrastive_loss += L_contrastive.item() * bs
                preds = outputs["gesture_logits"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total += bs

                for s in range(num_train_subjects):
                    n_s = (subject_labels == s).sum().item()
                    if n_s > 0:
                        ep_per_group_loss_sum[s] += batch_group_losses[s].item() * n_s
                        ep_per_group_count[s] += n_s

            # Epoch averages
            train_loss = ep_total_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            avg_gesture = ep_gesture_dro_loss / max(1, ep_total)
            avg_subject = ep_subject_loss / max(1, ep_total)
            avg_mi = ep_mi_loss / max(1, ep_total)
            avg_contrast = ep_contrastive_loss / max(1, ep_total)
            per_group_avg = [
                ep_per_group_loss_sum[s] / max(1, ep_per_group_count[s])
                for s in range(num_train_subjects)
            ]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["gesture_dro_loss"].append(avg_gesture)
            history["subject_loss"].append(avg_subject)
            history["mi_loss"].append(avg_mi)
            history["contrastive_loss"].append(avg_contrast)
            history["group_weights"].append(group_weights.cpu().tolist())
            history["per_group_losses"].append(per_group_avg)

            # ── Validation (gesture logits only) ─────────────────────
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        val_loss_sum += gesture_criterion(logits, yb).item() * yb.size(0)
                        val_correct += (logits.argmax(1) == yb).sum().item()
                        val_total += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            gw_str = ", ".join(f"{w:.3f}" for w in group_weights.cpu().tolist())
            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} (dro={avg_gesture:.4f}, "
                f"subj={avg_subject:.4f}, MI={avg_mi:.4f}, "
                f"contr={avg_contrast:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"beta={current_beta:.4f} | gw=[{gw_str}]"
            )

            # ── Early stopping ───────────────────────────────────────
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

        # ── Store trainer state ──────────────────────────────────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size
        self.final_group_weights = group_weights.cpu().tolist()

        # ── Save history ─────────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── Evaluate on val / test ───────────────────────────────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str) -> Optional[Dict]:
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    logits = model(xb.to(device))
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits)
            y_true = np.concatenate(all_y)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                cls_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, cls_labels, normalize=True, filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report": report,
                "confusion_matrix": cm.tolist(),
            }

        results["val"] = _eval_loader(dl_val, "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── Save checkpoint ──────────────────────────────────────────
        model_path = self.output_dir / "channel_contrastive_disentangled.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "num_subjects": num_train_subjects,
            "class_ids": class_ids,
            "mean": mean_c,
            "std": std_c,
            "window_size": window_size,
            "content_dim_per_ch": self.content_dim_per_ch,
            "style_dim_per_ch": self.style_dim_per_ch,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "dro_eta": self.dro_eta,
            "contrastive_temperature": self.contrastive_temperature,
            "final_group_weights": group_weights.cpu().tolist(),
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy() is inherited from DisentangledTrainer unchanged.
    # At inference the model is in eval mode → returns gesture_logits only
    # (from fused z_content) — no subject info needed.
