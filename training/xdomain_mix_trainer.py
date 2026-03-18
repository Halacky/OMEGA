"""
Trainer for XDomainMix 4-Component Decomposition model (Experiment 101).

Extends DisentangledTrainer with:
    - XDomainMixEMG model (4 projection heads: z_cg, z_cs, z_dg, z_ds)
    - Dual gesture loss: base path + gamma * augmented path (FiLM-conditioned)
    - Domain (subject) classification loss on concat(z_ds, z_dg)
    - Cross-type orthogonality losses: 4 pairs (cs-ds, cs-dg, cg-ds, cg-dg)
    - Beta annealing for orthogonality losses

Loss:
    L_total = L_gesture_base                                    (main CE, base path)
            + gamma · L_gesture_aug                             (CE, FiLM-augmented path)
            + alpha_d · L_domain                                (CE, domain classifier)
            + beta(t) · (orth(z_cs, z_ds) + orth(z_cs, z_dg)   (cross-type orthogonality)
                        + orth(z_cg, z_ds) + orth(z_cg, z_dg)) / 4

Inference: model.eval() → GestureHead(concat(z_cs, z_cg)) tensor only.

LOSO guarantee:
    Training DataLoader is built exclusively from training-subject windows.
    subject_labels are consecutive integers 0…K-1 for K training subjects.
    The test subject has no registered index and never appears in any training batch,
    in the domain-specific swap pool, or in any loss computation.
    evaluate_numpy() applies only per-channel standardisation (stats from training)
    and routes through the base gesture path — no domain info required.
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
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from models.xdomain_mix_emg import XDomainMixEMG, orthogonality_loss


class XDomainMixTrainer(DisentangledTrainer):
    """
    Trainer for XDomainMix 4-component EMG model.

    Inherits from DisentangledTrainer:
        - _prepare_splits_arrays()          flat array extraction from splits dict
        - _build_subject_labels_array()     aligns subject labels to class_ids order
        - _compute_channel_standardization() training-only stats
        - _apply_standardization()          applies stats to any split
        - evaluate_numpy()                  eval-mode inference on z_cs + z_cg path

    Overrides:
        - fit()   creates XDomainMixEMG and trains with 4-component losses

    Attributes set by fit() (required by evaluate_numpy):
        self.model, self.mean_c, self.std_c, self.class_ids, self.class_names,
        self.in_channels, self.window_size
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # 4-component dimensions
        cg_dim: int = 32,
        cs_dim: int = 96,
        dg_dim: int = 32,
        ds_dim: int = 64,
        # Loss weights
        gamma: float = 0.5,          # augmented gesture path weight
        alpha_d: float = 0.3,        # domain (subject) classification weight
        beta_orth: float = 0.1,      # cross-type orthogonality weight (annealed)
        beta_anneal_epochs: int = 10,
    ):
        """
        Args:
            train_cfg:            TrainingConfig
            logger:               Python logger
            output_dir:           Path for checkpoints / artefacts
            visualizer:           optional Visualizer
            cg_dim:               class-generic latent dimensionality
            cs_dim:               class-specific latent dimensionality
            dg_dim:               domain-generic latent dimensionality
            ds_dim:               domain-specific latent dimensionality
            gamma:                weight of augmented gesture path loss
            alpha_d:              weight of domain (subject) classification loss
            beta_orth:            maximum weight of cross-type orthogonality losses
            beta_anneal_epochs:   epochs over which beta_orth ramps from 0 → beta_orth
        """
        # Call DisentangledTrainer init with dummy content/style/alpha/beta;
        # we override fit() completely so parent hyperparams are unused.
        super().__init__(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=visualizer,
            content_dim=cs_dim,  # not used by our fit(), kept for interface compat
            style_dim=ds_dim,    # not used by our fit(), kept for interface compat
            alpha=alpha_d,       # stored as self.alpha by parent; we also store as alpha_d
            beta=beta_orth,      # stored as self.beta by parent; we also store as beta_orth
            beta_anneal_epochs=beta_anneal_epochs,
            mi_loss_type="orthogonal",
        )
        self.cg_dim = cg_dim
        self.cs_dim = cs_dim
        self.dg_dim = dg_dim
        self.ds_dim = ds_dim
        self.gamma = gamma
        self.alpha_d = alpha_d
        self.beta_orth = beta_orth
        self.beta_anneal_epochs = beta_anneal_epochs

    # ─────────────────────────────── fit ────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train XDomainMix 4-component model.

        Expects splits to contain:
            "train":                Dict[gesture_id, np.ndarray (N, T, C)]
            "val":                  Dict[gesture_id, np.ndarray (N, T, C)]
            "test":                 Dict[gesture_id, np.ndarray (N, T, C)]  (optional)
            "train_subject_labels": Dict[gesture_id, np.ndarray (N,) int]
            "num_train_subjects":   int

        LOSO guarantee:
            - splits["train"] and splits["train_subject_labels"] contain ONLY
              training-subject data. The test subject is exclusively in splits["test"].
            - Standardisation (mean_c, std_c) computed on training windows only.
            - Training batches (dl_train) contain only training-subject samples.
            - subject_labels in each batch are indices 0…K-1 for K training subjects.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Build flat arrays ───────────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # ── 2. Subject labels aligned to class_ids order ───────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "XDomainMixTrainer requires 'train_subject_labels' in splits. "
                "Use the experiment file that injects subject provenance."
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
            f"XDomainMix training: {num_train_subjects} subjects, "
            f"subject distribution: {np.bincount(y_subject_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) ───────────────────────
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

        # ── 4. Per-channel standardisation (training data only, LOSO clean) ──
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation.")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Build XDomainMixEMG ──────────────────────────────────────
        model = XDomainMixEMG(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            cg_dim=self.cg_dim,
            cs_dim=self.cs_dim,
            dg_dim=self.dg_dim,
            ds_dim=self.ds_dim,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"XDomainMixEMG: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, "
            f"cg={self.cg_dim}, cs={self.cs_dim}, "
            f"dg={self.dg_dim}, ds={self.ds_dim}, "
            f"gamma={self.gamma}, alpha_d={self.alpha_d}, "
            f"beta_orth={self.beta_orth}, params={total_params:,}"
        )

        # ── 6. Datasets and DataLoaders ──────────────────────────────────
        # Training: DisentangledWindowDataset returns (window, gesture_label, subject_label)
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

        # Domain (subject) classifier loss — no class weighting needed here
        domain_criterion = nn.CrossEntropyLoss()

        # ── 8. Optimizer and scheduler ───────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 9. Training loop ─────────────────────────────────────────────
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_base_loss": [], "gesture_aug_loss": [],
            "domain_loss": [], "orth_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            # Orthogonality loss beta: anneal from 0 to beta_orth
            current_beta = self.beta_orth * min(
                1.0, epoch / max(1, self.beta_anneal_epochs)
            )

            ep_total_loss = 0.0
            ep_gest_base_loss = 0.0
            ep_gest_aug_loss = 0.0
            ep_domain_loss = 0.0
            ep_orth_loss = 0.0
            ep_correct = 0
            ep_total = 0

            for windows, gesture_labels, subject_labels in dl_train:
                windows = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                # Forward: training mode triggers domain-specific swap and FiLM path
                outputs = model(windows, subject_labels=subject_labels)

                # (a) Base gesture loss — main task, identical to inference path
                L_gesture_base = gesture_criterion(
                    outputs["gesture_logits_base"], gesture_labels
                )

                # (b) Augmented gesture loss — FiLM(z_cs, z_ds_swap) path
                #     Labels unchanged: swapping z_ds doesn't change gesture identity
                L_gesture_aug = gesture_criterion(
                    outputs["gesture_logits_aug"], gesture_labels
                )

                # (c) Domain (subject) classification from concat(z_ds, z_dg)
                #     Drives domain latents to encode subject identity.
                #     Uses ORIGINAL z_ds (not swapped) for clean domain semantics.
                L_domain = domain_criterion(outputs["domain_logits"], subject_labels)

                # (d) Cross-type orthogonality: 4 class-vs-domain pairs
                #     Ensures class and domain components encode independent information.
                #     orth(z_cs, z_ds): most critical — gesture-specific vs subject-specific
                #     orth(z_cs, z_dg): gesture-specific vs universal physiological
                #     orth(z_cg, z_ds): common activation vs subject-specific
                #     orth(z_cg, z_dg): common activation vs universal physiological
                L_orth = (
                    orthogonality_loss(outputs["z_cs"], outputs["z_ds"])
                    + orthogonality_loss(outputs["z_cs"], outputs["z_dg"])
                    + orthogonality_loss(outputs["z_cg"], outputs["z_ds"])
                    + orthogonality_loss(outputs["z_cg"], outputs["z_dg"])
                ) / 4.0

                # Combined loss
                total_loss = (
                    L_gesture_base
                    + self.gamma * L_gesture_aug
                    + self.alpha_d * L_domain
                    + current_beta * L_orth
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_total_loss += total_loss.item() * bs
                ep_gest_base_loss += L_gesture_base.item() * bs
                ep_gest_aug_loss += L_gesture_aug.item() * bs
                ep_domain_loss += L_domain.item() * bs
                ep_orth_loss += L_orth.item() * bs

                # Accuracy tracked on base path (inference-time metric)
                preds = outputs["gesture_logits_base"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total += bs

            train_loss = ep_total_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            avg_gest_base = ep_gest_base_loss / max(1, ep_total)
            avg_gest_aug = ep_gest_aug_loss / max(1, ep_total)
            avg_domain = ep_domain_loss / max(1, ep_total)
            avg_orth = ep_orth_loss / max(1, ep_total)

            # Validation: base path only (mirrors exact inference path)
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total_n = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        # eval mode → returns gesture_logits_base tensor only
                        logits = model(xb)
                        loss = gesture_criterion(logits, yb)
                        val_loss_sum += loss.item() * yb.size(0)
                        preds = logits.argmax(dim=1)
                        val_correct += (preds == yb).sum().item()
                        val_total_n += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total_n)
                val_acc = val_correct / max(1, val_total_n)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["gesture_base_loss"].append(avg_gest_base)
            history["gesture_aug_loss"].append(avg_gest_aug)
            history["domain_loss"].append(avg_domain)
            history["orth_loss"].append(avg_orth)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} "
                f"(gest_base={avg_gest_base:.4f}, gest_aug={avg_gest_aug:.4f}, "
                f"domain={avg_domain:.4f}, orth={avg_orth:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | beta={current_beta:.4f}"
            )

            # Early stopping on val gesture loss (base path = inference metric)
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
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── Store trainer state (required by evaluate_numpy) ─────────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── Save training history ─────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── In-fold evaluation (val + test from training-time splits) ─────
        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name):
            """Evaluate a DataLoader using base path only (eval-mode model call)."""
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

        results["val"] = _eval_loader(dl_val, "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── Save model checkpoint ─────────────────────────────────────────
        model_path = self.output_dir / "xdomain_mix_emg.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "num_subjects": num_train_subjects,
            "class_ids": class_ids,
            "mean": mean_c,
            "std": std_c,
            "window_size": window_size,
            "cg_dim": self.cg_dim,
            "cs_dim": self.cs_dim,
            "dg_dim": self.dg_dim,
            "ds_dim": self.ds_dim,
            "gamma": self.gamma,
            "alpha_d": self.alpha_d,
            "beta_orth": self.beta_orth,
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy is inherited from DisentangledTrainer unchanged.
    # It calls model(xb) in eval mode which returns gesture_logits_base tensor,
    # then applies trained-data standardisation (mean_c, std_c). LOSO safe.
