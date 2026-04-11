"""
Trainer for Disentangled CNN-GRU with EMA Prototype Regularization.

Extends DisentangledTrainer with two additional losses in z_content space:

  L_center  = mean_i  ||z_content_i - ema_proto[y_i]||^2
                Pull each sample toward its class's stable EMA anchor.
                Gradient flows only through z_content (EMA prototypes are detached).

  L_push    = mean_{a<b} max(0, margin - ||batch_proto_a - batch_proto_b||_2)
                Push mini-batch class prototypes apart.
                Gradient flows through batch_proto -> z_content.

Total loss:
  L = L_gesture + α·L_subject + β(t)·L_MI + γ(t)·L_center + δ(t)·L_push

LOSO compliance (no data leakage):
  - EMA prototypes are updated ONLY inside the training loop, one mini-batch at a time.
  - Validation data NEVER triggers a prototype update.
  - Test subject data NEVER enters the trainer (injected externally via splits["test"]).
  - At inference, only the gesture classifier head is used; prototypes are discarded.
  - compute_content_variance_by_subject() is a diagnostic-only method that must be
    called by the experiment with TRAIN-subject data only.
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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from models.disentangled_cnn_gru import (
    DisentangledCNNGRU,
    distance_correlation_loss,
    orthogonality_loss,
)


class ProtoDisentangledTrainer(DisentangledTrainer):
    """
    DisentangledTrainer augmented with EMA prototype regularization.

    Additional constructor parameters compared to DisentangledTrainer:
        lambda_center       float  Center-loss weight                  (default 0.10)
        lambda_push         float  Inter-class push-loss weight         (default 0.05)
        push_margin         float  Minimum L2 distance between protos   (default 4.0)
        ema_decay           float  EMA smoothing coefficient            (default 0.99)
        proto_anneal_epochs int    Epochs to linearly ramp proto losses (default 10)
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # ── disentanglement params (forwarded to parent) ──────────────────────
        content_dim: int = 128,
        style_dim: int = 64,
        alpha: float = 0.5,
        beta: float = 0.1,
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",
        # ── prototype params ──────────────────────────────────────────────────
        lambda_center: float = 0.10,
        lambda_push: float = 0.05,
        push_margin: float = 4.0,
        ema_decay: float = 0.99,
        proto_anneal_epochs: int = 10,
    ):
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
        self.lambda_center = lambda_center
        self.lambda_push = lambda_push
        self.push_margin = push_margin
        self.ema_decay = ema_decay
        self.proto_anneal_epochs = proto_anneal_epochs

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train with:
          gesture CE + subject CE + MI + center loss + inter-class push loss.

        Prototype updates happen ONLY on train batches (LOSO compliant).
        """
        seed_everything(self.cfg.seed)

        # ── 1. Prepare data arrays ──────────────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = (
            self._prepare_splits_arrays(splits)
        )
        num_classes = len(class_ids)

        # ── 2. Subject labels (injected by experiment file) ─────────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "ProtoDisentangledTrainer requires 'train_subject_labels' in splits. "
                "Build splits with _build_splits_with_subject_labels()."
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

        # ── 3. Transpose (N, T, C) → (N, C, T) ─────────────────────────────
        # windows from grouped_windows are (N, T, C); model expects (B, C, T)
        if X_train.ndim == 3:
            _N, d1, d2 = X_train.shape
            if d1 > d2:  # T > C
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]

        # ── 4. Channel standardization (train stats only) ───────────────────
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

        # ── 5. Model ────────────────────────────────────────────────────────
        device = self.cfg.device
        model = DisentangledCNNGRU(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dropout=self.cfg.dropout,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"ProtoDisentangled model: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, content_dim={self.content_dim}, "
            f"style_dim={self.style_dim}, params={total_params:,}"
        )
        self.logger.info(
            f"Prototype config: λ_center={self.lambda_center}, "
            f"λ_push={self.lambda_push}, margin={self.push_margin}, "
            f"ema_decay={self.ema_decay}, anneal_epochs={self.proto_anneal_epochs}"
        )

        # ── 6. EMA prototypes (LOSO-safe: updated on train batches only) ────
        # Shape: (num_classes, content_dim).
        # proto_initialized[c] = False until class c is seen in the first batch.
        # After initialization, standard EMA update is applied.
        ema_protos = torch.zeros(num_classes, self.content_dim, device=device)
        proto_initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

        # ── 7. Datasets & dataloaders ────────────────────────────────────────
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        wifn = get_worker_init_fn(self.cfg.seed)
        nw = self.cfg.num_workers

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            worker_init_fn=wifn if nw > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        dl_val = (
            DataLoader(
                ds_val,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=nw,
                pin_memory=True,
                worker_init_fn=wifn if nw > 0 else None,
            )
            if ds_val
            else None
        )
        dl_test = (
            DataLoader(
                ds_test,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=nw,
                pin_memory=True,
                worker_init_fn=wifn if nw > 0 else None,
            )
            if ds_test
            else None
        )

        # ── 8. Loss functions ────────────────────────────────────────────────
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = counts.sum() / (counts + 1e-8)
            cw = cw / cw.mean()
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_crit = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(device)
            )
        else:
            gesture_crit = nn.CrossEntropyLoss()

        subject_crit = nn.CrossEntropyLoss()

        # ── 9. Optimizer & LR scheduler ─────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # ── 10. Training loop ────────────────────────────────────────────────
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "gesture_loss": [],
            "subject_loss": [],
            "mi_loss": [],
            "center_loss": [],
            "push_loss": [],
        }
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()

            # Annealing schedules
            beta_t = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))
            proto_scale = min(1.0, epoch / max(1, self.proto_anneal_epochs))
            lam_ctr = self.lambda_center * proto_scale
            lam_push = self.lambda_push * proto_scale

            ep_loss = ep_gest = ep_subj = ep_mi = ep_ctr = ep_push = 0.0
            ep_correct = ep_total = 0

            for windows, gest_labels, subj_labels in dl_train:
                windows = windows.to(device)
                gest_labels = gest_labels.to(device)
                subj_labels = subj_labels.to(device)

                optimizer.zero_grad()

                # Forward — train mode returns full dict
                out = model(windows, return_all=True)
                z_content = out["z_content"]  # (B, content_dim)
                z_style = out["z_style"]       # (B, style_dim)

                # ── Base losses ───────────────────────────────────────────
                L_gesture = gesture_crit(out["gesture_logits"], gest_labels)
                L_subject = subject_crit(out["subject_logits"], subj_labels)

                if self.mi_loss_type == "distance_correlation":
                    L_MI = distance_correlation_loss(z_content, z_style)
                elif self.mi_loss_type == "orthogonal":
                    L_MI = orthogonality_loss(z_content, z_style)
                elif self.mi_loss_type == "both":
                    L_MI = distance_correlation_loss(
                        z_content, z_style
                    ) + 0.1 * orthogonality_loss(z_content, z_style)
                else:
                    L_MI = distance_correlation_loss(z_content, z_style)

                # ── Prototype losses ──────────────────────────────────────
                # Differentiable batch prototype: mean z_content per class
                # in this mini-batch.  Gradient flows: batch_proto → z_content
                # → encoder weights.
                batch_proto: Dict[int, torch.Tensor] = {}
                for c in range(num_classes):
                    mask = gest_labels == c
                    if mask.sum() > 0:
                        batch_proto[c] = z_content[mask].mean(dim=0)

                # Center loss: pull z_content_i toward its EMA anchor proto.
                # The EMA proto is detached → gradient only through z_content.
                if lam_ctr > 0 and batch_proto:
                    ctr_parts = []
                    for c, bp in batch_proto.items():
                        mask = gest_labels == c
                        # Anchor: EMA proto if initialized, else batch mean
                        # (detached in both cases so no gradient through target)
                        if proto_initialized[c]:
                            target = ema_protos[c].detach()
                        else:
                            target = bp.detach()
                        diff = z_content[mask] - target  # (|mask|, D)
                        ctr_parts.append((diff ** 2).sum(dim=1).mean())
                    L_center = torch.stack(ctr_parts).mean()
                else:
                    L_center = torch.tensor(0.0, device=device, requires_grad=True)

                # Push-apart: push batch prototypes of different classes apart.
                # Margin is in L2 distance units.
                # Gradient flows through batch_proto → z_content → encoder.
                if lam_push > 0 and len(batch_proto) >= 2:
                    classes_here = sorted(batch_proto.keys())
                    push_parts = []
                    for i in range(len(classes_here)):
                        for j in range(i + 1, len(classes_here)):
                            ci, cj = classes_here[i], classes_here[j]
                            diff_p = batch_proto[ci] - batch_proto[cj]
                            dist = torch.sqrt((diff_p ** 2).sum() + 1e-8)
                            loss_ij = torch.clamp(self.push_margin - dist, min=0.0)
                            push_parts.append(loss_ij)
                    L_push = torch.stack(push_parts).mean()
                else:
                    L_push = torch.tensor(0.0, device=device, requires_grad=True)

                # ── Total loss & backward ─────────────────────────────────
                total_loss = (
                    L_gesture
                    + self.alpha * L_subject
                    + beta_t * L_MI
                    + lam_ctr * L_center
                    + lam_push * L_push
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # ── EMA prototype update (TRAIN DATA ONLY) ────────────────
                # LOSO compliance: this block runs exclusively inside the
                # training loop.  Val and test data never reach this code.
                with torch.no_grad():
                    for c, bp in batch_proto.items():
                        bp_detach = bp.detach()
                        if not proto_initialized[c]:
                            ema_protos[c] = bp_detach.clone()
                            proto_initialized[c] = True
                        else:
                            ema_protos[c] = (
                                self.ema_decay * ema_protos[c]
                                + (1.0 - self.ema_decay) * bp_detach
                            )

                # ── Batch stats ───────────────────────────────────────────
                bs = windows.size(0)
                ep_loss += total_loss.item() * bs
                ep_gest += L_gesture.item() * bs
                ep_subj += L_subject.item() * bs
                ep_mi += L_MI.item() * bs
                ep_ctr += L_center.item() * bs
                ep_push += L_push.item() * bs
                preds = out["gesture_logits"].argmax(dim=1)
                ep_correct += (preds == gest_labels).sum().item()
                ep_total += bs

            n = max(1, ep_total)
            train_loss = ep_loss / n
            train_acc = ep_correct / n

            # ── Validation (no prototype updates) ────────────────────────
            val_loss, val_acc = float("nan"), float("nan")
            if dl_val is not None:
                model.eval()
                vl_sum = vc = vt = 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)  # eval mode: gesture_logits only
                        vl_sum += gesture_crit(logits, yb).item() * yb.size(0)
                        vc += (logits.argmax(1) == yb).sum().item()
                        vt += yb.size(0)
                val_loss = vl_sum / max(1, vt)
                val_acc = vc / max(1, vt)

            # ── Logging ──────────────────────────────────────────────────
            n_init = proto_initialized.sum().item()
            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"train={train_loss:.4f} "
                f"(g={ep_gest/n:.4f} s={ep_subj/n:.4f} "
                f"MI={ep_mi/n:.4f} ctr={ep_ctr/n:.4f} push={ep_push/n:.4f}) "
                f"acc={train_acc:.3f} | "
                f"val={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"protos={n_init}/{num_classes} β={beta_t:.3f}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["gesture_loss"].append(ep_gest / n)
            history["subject_loss"].append(ep_subj / n)
            history["mi_loss"].append(ep_mi / n)
            history["center_loss"].append(ep_ctr / n)
            history["push_loss"].append(ep_push / n)

            # ── Early stopping ────────────────────────────────────────────
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

        # ── 11. Store trainer state ──────────────────────────────────────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 12. Save EMA prototypes (diagnostic artifact) ───────────────────
        protos_path = self.output_dir / "ema_prototypes.npz"
        np.savez_compressed(
            protos_path,
            prototypes=ema_protos.cpu().numpy(),
            initialized=proto_initialized.cpu().numpy(),
            class_ids=np.array(class_ids, dtype=np.int32),
        )
        self.logger.info(
            f"EMA prototypes saved: {protos_path} "
            f"({proto_initialized.sum().item()}/{num_classes} initialized)"
        )

        # ── 13. Save training history & curves ───────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png"
            )

        # ── 14. Evaluate on val / test splits ────────────────────────────────
        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    logits = model(xb.to(device))
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_np = np.concatenate(all_logits, axis=0)
            y_true = np.concatenate(all_y, axis=0)
            y_pred = logits_np.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True, filename=f"cm_{split_name}.png"
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report": report,
                "confusion_matrix": cm.tolist(),
            }

        results["val"] = _eval_loader(dl_val, "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── 15. Save model checkpoint ────────────────────────────────────────
        model_path = self.output_dir / "proto_disentangled_cnn_gru.pt"
        torch.save(
            {
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
                "ema_prototypes": ema_protos.cpu(),
                "proto_initialized": proto_initialized.cpu(),
                "alpha": self.alpha,
                "beta": self.beta,
                "lambda_center": self.lambda_center,
                "lambda_push": self.lambda_push,
                "push_margin": self.push_margin,
                "training_config": asdict(self.cfg),
            },
            model_path,
        )
        self.logger.info(f"Model checkpoint saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy() is inherited from DisentangledTrainer unchanged:
    # it uses only gesture_logits (eval mode), so prototypes are irrelevant.

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostic
    # ──────────────────────────────────────────────────────────────────────────

    def compute_content_variance_by_subject(
        self,
        subject_windows: Dict[str, np.ndarray],
        subject_labels: Dict[str, np.ndarray],
    ) -> Dict:
        """
        Compute intra-class variance of z_content across subjects.

        Measures whether z_content is subject-invariant: low variance means the
        same gesture from different subjects maps to similar z_content vectors.

        LOSO compliance: this method must ONLY be called with train-subject data.
        The experiment file is responsible for never passing test-subject data here.

        Args:
            subject_windows: {subject_id: (N, C, T) numpy array, already
                              transposed and standardized to match trainer's
                              preprocessing pipeline}
            subject_labels:  {subject_id: (N,) class-index int64 array}

        Returns:
            dict with:
              "intra_class_variance_by_class": {class_idx: float}  per-class variance
              "mean_intra_class_variance": float   mean across classes
        """
        assert self.model is not None, "call fit() before computing diagnostics"
        assert self.mean_c is not None and self.std_c is not None

        device = self.cfg.device
        self.model.eval()

        # Accumulate per-class, per-subject mean z_content.
        # {class_idx: {subject_id: mean_z_content (D,)}}
        per_class_subj_means: Dict[int, Dict[str, np.ndarray]] = {}

        with torch.no_grad():
            for sid, X in subject_windows.items():
                y = subject_labels[sid]
                X_t = torch.from_numpy(X).float()

                # Collect z_content in batches
                z_parts: List[np.ndarray] = []
                for start in range(0, len(X_t), self.cfg.batch_size):
                    batch = X_t[start : start + self.cfg.batch_size].to(device)
                    # return_all=True works in eval mode for DisentangledCNNGRU
                    out = self.model(batch, return_all=True)
                    z_parts.append(out["z_content"].cpu().numpy())
                z_all = np.concatenate(z_parts, axis=0)  # (N, content_dim)

                for c in np.unique(y):
                    mask = y == c
                    mean_z = z_all[mask].mean(axis=0)  # (D,)
                    if c not in per_class_subj_means:
                        per_class_subj_means[c] = {}
                    per_class_subj_means[c][sid] = mean_z

        # Intra-class variance across subjects:
        #   var(c) = mean_s ||mu_c_s - mu_c_global||^2
        variance_by_class: Dict[int, float] = {}
        for c, subj_means in per_class_subj_means.items():
            if len(subj_means) < 2:
                continue
            means = np.stack(list(subj_means.values()), axis=0)  # (S, D)
            global_mean = means.mean(axis=0)                      # (D,)
            var = float(np.mean(((means - global_mean) ** 2).sum(axis=1)))
            variance_by_class[int(c)] = var

        overall = (
            float(np.mean(list(variance_by_class.values())))
            if variance_by_class
            else float("nan")
        )

        return {
            "intra_class_variance_by_class": variance_by_class,
            "mean_intra_class_variance": overall,
        }
