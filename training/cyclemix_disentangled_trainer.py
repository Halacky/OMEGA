"""
Trainer for CycleMix Disentangled CNN-GRU (Experiment 98).

Extends DisentangledTrainer with:
    - CycleMixDisentangledCNNGRU instead of DisentangledCNNGRU / MixStyleDisentangledCNNGRU
    - Per-epoch α randomisation: epoch_alpha ~ Uniform(alpha_low, alpha_high)
      sampled once per epoch and passed to model.forward() for channel-wise Beta mixing
    - Dual gesture loss: base path (no FiLM) + γ · mixed-style path (FiLM-conditioned)

Loss:
    L_total = L_gesture_base
            + γ · L_gesture_mix
            + α_subj · L_subject
            + β(t) · L_MI(z_content, z_style)

    where β(t) ramps from 0 to β over beta_anneal_epochs (identical to DisentangledTrainer).

LOSO guarantee:
    The training DataLoader is built exclusively from train-subject windows.
    subject_labels passed to model.forward() are indices into the train-subject
    list [0, N_train-1].  The test subject has no index and never appears in any batch.
    Per-channel donors are drawn from [0, B-1] within the current training batch —
    all test-time safe.
    Validation and test evaluation use base path only: model(x) → gesture_logits_base.
    No test-time adaptation.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from models.cyclemix_disentangled_cnn_gru import CycleMixDisentangledCNNGRU
from models.disentangled_cnn_gru import distance_correlation_loss, orthogonality_loss


class CycleMixDisentangledTrainer(DisentangledTrainer):
    """
    Trainer for CycleMix channel-wise stochastic style mixing model.

    Inherits from DisentangledTrainer (which in turn inherits from WindowClassifierTrainer):
        - _prepare_splits_arrays()             — flatten Dict splits → arrays
        - _build_subject_labels_array()        — align subject labels with class_ids order
        - _compute_channel_standardization()   — compute per-channel mean/std from train set
        - _apply_standardization()             — apply stored stats
        - evaluate_numpy()                     — eval using base path (no FiLM, LOSO safe)

    Overrides fit() to:
        1. Instantiate CycleMixDisentangledCNNGRU.
        2. Sample epoch_alpha ~ Uniform(alpha_low, alpha_high) once per epoch.
        3. Pass epoch_alpha and subject_labels to model.forward() for CycleMix augmentation.
        4. Apply dual gesture loss (base path + γ · mix path).
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        content_dim: int = 128,
        style_dim: int = 64,
        alpha_subj: float = 0.5,       # weight of subject classifier loss
        beta_mi: float = 0.1,          # weight of MI minimisation loss (annealed)
        gamma: float = 0.5,            # weight of mixed-style gesture path
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",
        alpha_low: float = 0.1,        # lower bound of per-epoch α ~ Uniform
        alpha_high: float = 0.5,       # upper bound of per-epoch α ~ Uniform
    ):
        """
        Args:
            train_cfg:           TrainingConfig
            logger:              Python logger
            output_dir:          Path for checkpoints / artifacts
            visualizer:          optional Visualizer
            content_dim:         dimensionality of z_content
            style_dim:           dimensionality of z_style (must be divisible by 8 for EMG)
            alpha_subj:          weight of subject classification loss
            beta_mi:             weight of MI (distance-correlation) loss
            gamma:               weight of mixed-style path gesture loss
            beta_anneal_epochs:  epochs over which beta_mi ramps from 0 to beta_mi
            mi_loss_type:        "distance_correlation" | "orthogonal" | "both"
            alpha_low:           minimum α value for Beta(α, α) per-channel mixing
            alpha_high:          maximum α value for Beta(α, α) per-channel mixing
        """
        # Pass alpha_subj as alpha and beta_mi as beta to parent DisentangledTrainer
        super().__init__(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=visualizer,
            content_dim=content_dim,
            style_dim=style_dim,
            alpha=alpha_subj,
            beta=beta_mi,
            beta_anneal_epochs=beta_anneal_epochs,
            mi_loss_type=mi_loss_type,
        )
        self.gamma      = gamma
        self.alpha_low  = alpha_low
        self.alpha_high = alpha_high

    # ─────────────────────────── fit ───────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train CycleMix-disentangled model.

        Expects splits dict (from _build_splits_with_subject_labels in experiment):
            "train":                Dict[gesture_id → np.ndarray (N, T, C)]
            "val":                  Dict[gesture_id → np.ndarray (N, T, C)]
            "test":                 Dict[gesture_id → np.ndarray (N, T, C)]  (optional)
            "train_subject_labels": Dict[gesture_id → np.ndarray (N,) int]
            "num_train_subjects":   int

        Returns training + in-fold evaluation results dict.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Flatten splits to arrays ─────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # ── 2. Subject labels aligned to class_ids ordering ─────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "CycleMixDisentangledTrainer requires 'train_subject_labels' in splits. "
                "Use the experiment file that injects subject provenance labels."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        if len(y_subject_train) != len(y_train):
            raise ValueError(
                f"Subject labels length ({len(y_subject_train)}) "
                f"must equal gesture labels length ({len(y_train)})"
            )
        self.logger.info(
            f"Training: {num_train_subjects} subjects, "
            f"subject distribution: {np.bincount(y_subject_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) if needed ───────────────
        # Windows from load_multiple_subjects are (N, T, C); model expects (N, C, T)
        if X_train.ndim == 3:
            N, dim1, dim2 = X_train.shape
            if dim1 > dim2:  # dim1=T (600), dim2=C (8): T > C → transpose
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        in_channels = X_train.shape[1]   # C = 8
        window_size = X_train.shape[2]   # T = 600 / 8 = 75 after 3× MaxPool2
        num_classes = len(class_ids)

        # ── 4. Per-channel standardisation — computed on train only ─────
        # LOSO clean: test-subject windows are never included here
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val)  > 0:
            X_val  = self._apply_standardization(X_val,  mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation (train stats only).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Build CycleMixDisentangledCNNGRU ─────────────────────────
        model = CycleMixDisentangledCNNGRU(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            num_channels=in_channels,   # style groups = EMG channels
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"CycleMixDisentangledCNNGRU: in_ch={in_channels}, "
            f"gestures={num_classes}, subjects={num_train_subjects}, "
            f"content_dim={self.content_dim}, style_dim={self.style_dim}, "
            f"num_channels={in_channels}, "
            f"alpha_range=({self.alpha_low}, {self.alpha_high}), "
            f"gamma={self.gamma}, params={total_params:,}"
        )

        # ── 6. Datasets and DataLoaders ─────────────────────────────────
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
        ds_val   = WindowDataset(X_val,  y_val)  if len(X_val)  > 0 else None
        ds_test  = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

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

        # ── 8. Optimizer + LR scheduler ─────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # No verbose= kwarg: removed in PyTorch 2.4
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 9. Training loop ─────────────────────────────────────────────
        history = {
            "train_loss": [],       "val_loss": [],
            "train_acc":  [],       "val_acc":  [],
            "gesture_base_loss": [], "gesture_mix_loss": [],
            "subject_loss": [],     "mi_loss": [],
            "epoch_alpha": [],
        }
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        device        = self.cfg.device
        rng           = np.random.RandomState(self.cfg.seed)

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()

            # β annealing: ramp MI weight from 0 → beta_mi over beta_anneal_epochs
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))

            # Per-epoch α randomisation: how strongly styles are mixed this epoch.
            # This provides stochastic curriculum over mixing strength rather than
            # a fixed β (exp_60) or pure curriculum (exp_34).
            epoch_alpha = float(rng.uniform(self.alpha_low, self.alpha_high))

            ep_total_loss     = 0.0
            ep_gest_base_loss = 0.0
            ep_gest_mix_loss  = 0.0
            ep_subject_loss   = 0.0
            ep_mi_loss        = 0.0
            ep_correct        = 0
            ep_total          = 0

            for windows, gesture_labels, subject_labels in dl_train:
                windows        = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                # Forward: training mode performs CycleMix augmentation
                # subject_labels contains ONLY training-subject indices — LOSO safe
                outputs = model(
                    windows,
                    subject_labels=subject_labels,
                    epoch_alpha=epoch_alpha,
                )

                # Base gesture loss: z_content → classifier (no FiLM)
                # This is also the inference-time prediction — always co-trained
                L_gest_base = gesture_criterion(
                    outputs["gesture_logits_base"], gesture_labels
                )

                # CycleMix path: FiLM(z_content, z_style_mix) → classifier
                L_gest_mix = gesture_criterion(
                    outputs["gesture_logits_mix"], gesture_labels
                )

                # Subject classification loss: drives z_style to encode subject identity
                L_subject = subject_criterion(
                    outputs["subject_logits"], subject_labels
                )

                # MI minimisation: decorrelates z_content from z_style
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
                ep_total_loss     += total_loss.item()  * bs
                ep_gest_base_loss += L_gest_base.item() * bs
                ep_gest_mix_loss  += L_gest_mix.item()  * bs
                ep_subject_loss   += L_subject.item()   * bs
                ep_mi_loss        += L_MI.item()        * bs

                # Accuracy from base path (= inference path)
                preds      = outputs["gesture_logits_base"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total   += bs

            n = max(1, ep_total)
            train_loss    = ep_total_loss     / n
            train_acc     = ep_correct        / n
            avg_gest_base = ep_gest_base_loss / n
            avg_gest_mix  = ep_gest_mix_loss  / n
            avg_subject   = ep_subject_loss   / n
            avg_mi        = ep_mi_loss        / n

            # Validation: base path only (mirrors inference, no FiLM)
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb     = xb.to(device)
                        yb     = yb.to(device)
                        logits = model(xb)          # eval mode → gesture_logits_base
                        loss   = gesture_criterion(logits, yb)
                        val_loss_sum += loss.item() * yb.size(0)
                        preds         = logits.argmax(dim=1)
                        val_correct  += (preds == yb).sum().item()
                        val_total    += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc  = val_correct  / max(1, val_total)
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
            history["epoch_alpha"].append(epoch_alpha)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} "
                f"(base={avg_gest_base:.4f}, mix={avg_gest_mix:.4f}, "
                f"subj={avg_subject:.4f}, MI={avg_mi:.4f}), "
                f"acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"β={current_beta:.4f}, α={epoch_alpha:.3f}"
            )

            # Early stopping on val gesture loss (base path = inference metric)
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve  = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}.")
                        break

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── Store trainer state (used by evaluate_numpy from parent) ────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── Save training history ────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── In-fold evaluation (val + test from training-time data) ─────

        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str):
            """Evaluate a DataLoader using the base path (no FiLM)."""
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb     = xb.to(device)
                    logits = model(xb)   # eval → gesture_logits_base
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits, axis=0)
            y_true     = np.concatenate(all_y,      axis=0)
            y_pred     = logits_arr.argmax(axis=1)

            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
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

        # ── Save model checkpoint ────────────────────────────────────────
        model_path = self.output_dir / "cyclemix_disentangled_cnn_gru.pt"
        torch.save({
            "state_dict":       model.state_dict(),
            "in_channels":      in_channels,
            "num_classes":      num_classes,
            "num_subjects":     num_train_subjects,
            "class_ids":        class_ids,
            "mean":             mean_c,
            "std":              std_c,
            "window_size":      window_size,
            "content_dim":      self.content_dim,
            "style_dim":        self.style_dim,
            "alpha_subj":       self.alpha,
            "beta_mi":          self.beta,
            "gamma":            self.gamma,
            "alpha_low":        self.alpha_low,
            "alpha_high":       self.alpha_high,
            "training_config":  asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy is inherited from DisentangledTrainer:
    # It transposes (N,T,C)→(N,C,T), applies stored mean_c/std_c,
    # then calls model(xb) in eval mode → gesture_logits_base tensor.
    # The test-subject style is never requested or used. LOSO clean.
