"""
Trainer combining three hypotheses for EMG cross-subject gesture recognition:

  1. Curriculum subject ordering: train on similar→distant subjects progressively
  2. Content-style disentanglement: DisentangledCNNGRU separates z_content / z_style
  3. Class-balanced oversampling + subject-aware MixUp: reduces class bias and
     forces cross-subject content invariance via cross-subject sample interpolation

Why the combination should work:
  - Curriculum ensures the model first learns from "easy" (similar) subjects,
    building a strong prior before encountering highly variable ones.
  - Disentanglement explicitly removes subject identity from the gesture classifier,
    so the classifier operates only on z_content.
  - Class-balanced oversampling removes gestural class bias that naturally arises
    when some gestures are less represented in certain subjects.
  - Subject-aware MixUp generates in-between EMG signals from pairs with the SAME
    gesture label but DIFFERENT subjects, forcing z_content to be subject-invariant
    even before the MI loss kicks in.

Expected splits dict keys:
    "train":                Dict[int, np.ndarray]   gesture_id → windows (N, T, C)
    "val":                  Dict[int, np.ndarray]
    "test":                 Dict[int, np.ndarray]
    "train_subject_labels": Dict[int, np.ndarray]   gesture_id → subject index per window
    "train_subject_order":  List[int]               subject indices, similar→distant
    "num_train_subjects":   int
Optional:
    "val_subject_labels":   Dict[int, np.ndarray]   for embedding t-SNE visualization
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from training.trainer import (
    WindowClassifierTrainer, WindowDataset, get_worker_init_fn, seed_everything
)
from models.disentangled_cnn_gru import (
    DisentangledCNNGRU,
    distance_correlation_loss,
    orthogonality_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumDisentangledDataset(Dataset):
    """
    Returns (window, gesture_label, subject_label).
    Also exposes subject_ids as numpy for curriculum filtering.
    """

    def __init__(self, X: np.ndarray, y_gesture: np.ndarray, y_subject: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y_gesture = torch.from_numpy(y_gesture).long()
        self.y_subject = torch.from_numpy(y_subject).long()
        self.subject_ids = y_subject  # numpy, kept for masking

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_gesture[idx], self.y_subject[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumDisentangledTrainer(WindowClassifierTrainer):
    """
    Combined trainer implementing curriculum + disentanglement + class-balanced MixUp.

    Inherits from WindowClassifierTrainer for shared utilities
    (_prepare_splits_arrays, _compute_channel_standardization, evaluate_numpy, etc.).
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # ── Disentanglement ─────────────────────────────────────────────────
        content_dim: int = 128,
        style_dim: int = 64,
        alpha: float = 0.5,           # subject-classifier loss weight
        beta: float = 0.1,            # MI loss weight (annealed)
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",  # "distance_correlation" | "orthogonal" | "both"
        # ── Curriculum ──────────────────────────────────────────────────────
        k_init: int = 2,              # start with k_init nearest subjects
        expand_every: int = 8,        # add one subject every N epochs
        consolidation_epochs: int = 20,
        lr_on_expand: Optional[float] = None,
        # ── Class balancing & MixUp ─────────────────────────────────────────
        use_class_balanced_oversampling: bool = True,
        use_subject_mixup: bool = True,
        mixup_alpha: float = 0.2,     # Beta(α, α) parameter for MixUp λ
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)

        # disentanglement
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal_epochs = beta_anneal_epochs
        self.mi_loss_type = mi_loss_type

        # curriculum
        self.k_init = k_init
        self.expand_every = expand_every
        self.consolidation_epochs = consolidation_epochs
        self.lr_on_expand = lr_on_expand

        # augmentation
        self.use_class_balanced_oversampling = use_class_balanced_oversampling
        self.use_subject_mixup = use_subject_mixup
        self.mixup_alpha = mixup_alpha

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_subject_labels_array(
        self,
        subject_labels_dict: Dict[int, np.ndarray],
        class_ids: List[int],
    ) -> np.ndarray:
        """Flatten per-gesture subject labels, aligned with _prepare_splits_arrays order."""
        parts = [subject_labels_dict[gid] for gid in class_ids if gid in subject_labels_dict]
        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.int64)

    def _compute_schedule(
        self,
        num_subjects: int,
        subject_order: List[int],
    ) -> List[Tuple[int, int, Set[int]]]:
        """
        Build curriculum stage list: [(start_epoch, end_epoch, set_of_allowed_subject_indices)].
        Mirrors CurriculumTrainer._compute_schedule().
        """
        stages: List[Tuple[int, int, Set[int]]] = []
        epoch_cursor = 1
        num_expand = num_subjects - self.k_init

        if num_expand <= 0:
            total = self.expand_every + self.consolidation_epochs
            stages.append((epoch_cursor, epoch_cursor + total - 1, set(subject_order)))
            return stages

        # Stage 0: k_init subjects
        stages.append((
            epoch_cursor,
            epoch_cursor + self.expand_every - 1,
            set(subject_order[:self.k_init]),
        ))
        epoch_cursor += self.expand_every

        # Expansion stages
        for i in range(self.k_init, num_subjects):
            stages.append((
                epoch_cursor,
                epoch_cursor + self.expand_every - 1,
                set(subject_order[:i + 1]),
            ))
            epoch_cursor += self.expand_every

        # Consolidation: all subjects
        if self.consolidation_epochs > 0:
            stages.append((
                epoch_cursor,
                epoch_cursor + self.consolidation_epochs - 1,
                set(subject_order),
            ))

        return stages

    def _class_balanced_oversample(
        self,
        X: np.ndarray,
        y_gesture: np.ndarray,
        y_subject: np.ndarray,
        num_classes: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Upsample minority gesture classes to match the majority class count.
        Returns shuffled (X_balanced, y_gesture_balanced, y_subject_balanced).
        """
        class_counts = np.bincount(y_gesture, minlength=num_classes)
        max_count = int(class_counts.max())

        all_X = [X]
        all_y = [y_gesture]
        all_s = [y_subject]

        for c in range(num_classes):
            n_c = int(class_counts[c])
            n_extra = max_count - n_c
            if n_extra <= 0 or n_c == 0:
                continue
            c_indices = np.where(y_gesture == c)[0]
            extra_idx = rng.choice(c_indices, size=n_extra, replace=True)
            all_X.append(X[extra_idx])
            all_y.append(y_gesture[extra_idx])
            all_s.append(y_subject[extra_idx])

        X_bal = np.concatenate(all_X, axis=0)
        y_bal = np.concatenate(all_y, axis=0)
        s_bal = np.concatenate(all_s, axis=0)
        perm = rng.permutation(len(X_bal))
        return X_bal[perm], y_bal[perm], s_bal[perm]

    def _subject_aware_mixup(
        self,
        X: np.ndarray,
        y_gesture: np.ndarray,
        y_subject: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Cross-subject MixUp: for each window (x_i, g_i, s_i), find a partner
        (x_j, g_j, s_j) where g_j == g_i AND s_j != s_i.
        Mix: x_mix = λ·x_i + (1−λ)·x_j,  label = g_i,  subject = s_i.
        λ ~ Beta(mixup_alpha, mixup_alpha).

        Appends mixed samples to the original dataset.
        Returns (X_aug, y_gesture_aug, y_subject_aug, n_mixed_pairs).
        """
        mixed_X_list: List[np.ndarray] = []
        mixed_y_list: List[int] = []
        mixed_s_list: List[int] = []

        for c in np.unique(y_gesture):
            c_mask = y_gesture == c
            c_indices = np.where(c_mask)[0]
            unique_subjs = np.unique(y_subject[c_mask])
            if len(unique_subjs) < 2:
                continue  # can't do cross-subject mix with only 1 subject

            for idx in c_indices:
                s_i = y_subject[idx]
                partner_mask = (y_gesture == c) & (y_subject != s_i)
                partner_indices = np.where(partner_mask)[0]
                if len(partner_indices) == 0:
                    continue

                j = rng.choice(partner_indices)
                lam = rng.beta(self.mixup_alpha, self.mixup_alpha)
                x_mix = lam * X[idx] + (1.0 - lam) * X[j]
                mixed_X_list.append(x_mix)
                mixed_y_list.append(int(c))
                mixed_s_list.append(int(s_i))

        n_mixed = len(mixed_X_list)
        if n_mixed == 0:
            return X, y_gesture, y_subject, 0

        mixed_X = np.stack(mixed_X_list, axis=0)
        mixed_y = np.array(mixed_y_list, dtype=np.int64)
        mixed_s = np.array(mixed_s_list, dtype=np.int64)

        X_aug = np.concatenate([X, mixed_X], axis=0)
        y_aug = np.concatenate([y_gesture, mixed_y], axis=0)
        s_aug = np.concatenate([y_subject, mixed_s], axis=0)
        perm = rng.permutation(len(X_aug))
        return X_aug[perm], y_aug[perm], s_aug[perm], n_mixed

    def _build_stage_dataloader(
        self,
        allowed_subjects: Set[int],
        X_train_std: np.ndarray,
        y_train: np.ndarray,
        y_subject: np.ndarray,
        num_classes: int,
        rng: np.random.RandomState,
        stage_idx: int,
        epoch: int,
    ) -> Tuple[DataLoader, Dict]:
        """
        Build DataLoader for a curriculum stage:
          1. Filter to allowed subjects.
          2. Class-balanced oversampling (optional).
          3. Subject-aware MixUp (optional, requires ≥2 subjects).
        Returns (DataLoader, stage_stats dict for logging/visualization).
        """
        mask = np.isin(y_subject, list(allowed_subjects))
        X_s = X_train_std[mask]
        y_g_s = y_train[mask]
        y_s_s = y_subject[mask]

        stats: Dict = {
            "stage": stage_idx,
            "epoch_start": epoch,
            "n_raw": int(len(X_s)),
            "subjects": sorted(int(x) for x in allowed_subjects),
            "class_counts_raw": np.bincount(y_g_s, minlength=num_classes).tolist(),
        }

        if self.use_class_balanced_oversampling and len(X_s) > 0:
            X_s, y_g_s, y_s_s = self._class_balanced_oversample(
                X_s, y_g_s, y_s_s, num_classes, rng
            )
            stats["n_after_oversampling"] = int(len(X_s))
            stats["class_counts_oversampled"] = np.bincount(y_g_s, minlength=num_classes).tolist()

        n_mixed = 0
        if self.use_subject_mixup and len(allowed_subjects) >= 2 and len(X_s) > 0:
            X_s, y_g_s, y_s_s, n_mixed = self._subject_aware_mixup(
                X_s, y_g_s, y_s_s, rng
            )
            stats["n_after_mixup"] = int(len(X_s))
            stats["n_mixed_pairs"] = int(n_mixed)

        self.logger.info(
            f"  [Stage {stage_idx}] {len(X_s)} samples "
            f"({stats['n_raw']} raw + {n_mixed} mixed, {len(allowed_subjects)} subjects)"
        )

        ds = CurriculumDisentangledDataset(X_s, y_g_s, y_s_s)
        worker_init = get_worker_init_fn(self.cfg.seed)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed + epoch),
        )
        return dl, stats

    # ── main training ─────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train model with curriculum + disentanglement + class-balanced MixUp.
        """
        seed_everything(self.cfg.seed)
        rng = np.random.RandomState(self.cfg.seed)

        # ── 1. Prepare flat arrays ───────────────────────────────────────────
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)
        num_classes = len(class_ids)

        # ── 2. Extract subject metadata ──────────────────────────────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "CurriculumDisentangledTrainer requires 'train_subject_labels' in splits."
            )
        y_subject = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        subject_order: List[int] = splits["train_subject_order"]
        num_train_subjects: int = splits["num_train_subjects"]

        assert len(y_subject) == len(y_train), (
            f"Subject labels ({len(y_subject)}) ≠ gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Training: {num_train_subjects} subjects, order={subject_order}, "
            f"k_init={self.k_init}, expand_every={self.expand_every}, "
            f"consolidation={self.consolidation_epochs}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) ─────────────────────────────
        if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
            X_train = np.transpose(X_train, (0, 2, 1))
            if len(X_val) > 0:
                X_val = np.transpose(X_val, (0, 2, 1))
            if len(X_test) > 0:
                X_test = np.transpose(X_test, (0, 2, 1))
            self.logger.info(f"Transposed → (N, C, T): {X_train.shape}")

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]

        # ── 4. Channel standardization ───────────────────────────────────────
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train_std = self._apply_standardization(X_train, mean_c, std_c)
        X_val_std = self._apply_standardization(X_val, mean_c, std_c) if len(X_val) > 0 else X_val
        X_test_std = self._apply_standardization(X_test, mean_c, std_c) if len(X_test) > 0 else X_test
        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )
        self.logger.info("Applied per-channel standardization.")

        # ── 5. Create DisentangledCNNGRU ─────────────────────────────────────
        model = DisentangledCNNGRU(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        n_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"DisentangledCNNGRU: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, content={self.content_dim}, "
            f"style={self.style_dim}, params={n_params:,}"
        )

        # ── 6. Val / Test DataLoaders (static) ───────────────────────────────
        worker_init = get_worker_init_fn(self.cfg.seed)
        ds_val = WindowDataset(X_val_std, y_val) if len(X_val_std) > 0 else None
        ds_test = WindowDataset(X_test_std, y_test) if len(X_test_std) > 0 else None
        dl_val = DataLoader(
            ds_val, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
        ) if ds_test else None

        # ── 7. Loss functions ────────────────────────────────────────────────
        if self.cfg.use_class_weights:
            cc = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = cc.sum() / (cc + 1e-8)
            cw = cw / cw.mean()
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(self.cfg.device)
            )
        else:
            gesture_criterion = nn.CrossEntropyLoss()
        subject_criterion = nn.CrossEntropyLoss()

        # ── 8. Optimizer ─────────────────────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # ── 9. Curriculum schedule ────────────────────────────────────────────
        schedule = self._compute_schedule(num_train_subjects, subject_order)
        total_epochs = schedule[-1][1] if schedule else self.cfg.epochs

        self.logger.info(f"Curriculum schedule ({len(schedule)} stages, {total_epochs} total epochs):")
        for i, (s_ep, e_ep, subjs) in enumerate(schedule):
            self.logger.info(
                f"  Stage {i}: epochs {s_ep}–{e_ep}, "
                f"subjects={sorted(subjs)} ({len(subjs)}/{num_train_subjects})"
            )

        # ── 10. Training loop ─────────────────────────────────────────────────
        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_loss": [], "subject_loss": [], "mi_loss": [], "beta": [],
            "stage": [], "num_active_subjects": [],
        }
        stage_stats_history: List[Dict] = []

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        device = self.cfg.device
        current_stage_idx = 0
        current_dl_train = None

        for epoch in range(1, total_epochs + 1):
            # Advance stage pointer
            while current_stage_idx < len(schedule) - 1:
                if epoch > schedule[current_stage_idx][1]:
                    current_stage_idx += 1
                else:
                    break

            stage_start, stage_end, allowed_subjects = schedule[current_stage_idx]

            # Rebuild DataLoader on stage transition
            if epoch == stage_start:
                if self.lr_on_expand is not None and epoch > 1:
                    for pg in optimizer.param_groups:
                        pg["lr"] = self.lr_on_expand
                    self.logger.info(f"  LR reset → {self.lr_on_expand}")

                current_dl_train, s_stats = self._build_stage_dataloader(
                    allowed_subjects=allowed_subjects,
                    X_train_std=X_train_std,
                    y_train=y_train,
                    y_subject=y_subject,
                    num_classes=num_classes,
                    rng=rng,
                    stage_idx=current_stage_idx,
                    epoch=epoch,
                )
                stage_stats_history.append(s_stats)

            # Beta annealing
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))

            # ── Train epoch ──────────────────────────────────────────────────
            model.train()
            ep_total = ep_correct = 0
            ep_total_loss = ep_gesture_loss = ep_subject_loss = ep_mi_loss = 0.0

            for windows, g_labels, s_labels in current_dl_train:
                windows = windows.to(device)
                g_labels = g_labels.to(device)
                s_labels = s_labels.to(device)

                optimizer.zero_grad()
                out = model(windows, return_all=True)

                L_gesture = gesture_criterion(out["gesture_logits"], g_labels)
                L_subject = subject_criterion(out["subject_logits"], s_labels)

                if self.mi_loss_type == "distance_correlation":
                    L_MI = distance_correlation_loss(out["z_content"], out["z_style"])
                elif self.mi_loss_type == "orthogonal":
                    L_MI = orthogonality_loss(out["z_content"], out["z_style"])
                elif self.mi_loss_type == "both":
                    L_MI = (
                        distance_correlation_loss(out["z_content"], out["z_style"])
                        + 0.1 * orthogonality_loss(out["z_content"], out["z_style"])
                    )
                else:
                    L_MI = distance_correlation_loss(out["z_content"], out["z_style"])

                total_loss = L_gesture + self.alpha * L_subject + current_beta * L_MI
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_total += bs
                ep_total_loss += total_loss.item() * bs
                ep_gesture_loss += L_gesture.item() * bs
                ep_subject_loss += L_subject.item() * bs
                ep_mi_loss += L_MI.item() * bs
                ep_correct += (out["gesture_logits"].argmax(1) == g_labels).sum().item()

            train_loss = ep_total_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            avg_gest = ep_gesture_loss / max(1, ep_total)
            avg_subj = ep_subject_loss / max(1, ep_total)
            avg_mi = ep_mi_loss / max(1, ep_total)

            # ── Validation ───────────────────────────────────────────────────
            if dl_val is not None:
                model.eval()
                val_ls = val_cor = val_tot = 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)
                        val_ls += gesture_criterion(logits, yb).item() * yb.size(0)
                        val_cor += (logits.argmax(1) == yb).sum().item()
                        val_tot += yb.size(0)
                val_loss = val_ls / max(1, val_tot)
                val_acc = val_cor / max(1, val_tot)
            else:
                val_loss = val_acc = float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["gesture_loss"].append(avg_gest)
            history["subject_loss"].append(avg_subj)
            history["mi_loss"].append(avg_mi)
            history["beta"].append(current_beta)
            history["stage"].append(current_stage_idx)
            history["num_active_subjects"].append(len(allowed_subjects))

            self.logger.info(
                f"[Epoch {epoch:03d}/{total_epochs}] "
                f"Stage {current_stage_idx} ({len(allowed_subjects)}s) | "
                f"total={train_loss:.4f} (gest={avg_gest:.4f} subj={avg_subj:.4f} MI={avg_mi:.4f}) "
                f"acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f} | β={current_beta:.4f}"
            )

            # ── Early stopping (consolidation stage only) ─────────────────────
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    if current_stage_idx == len(schedule) - 1:
                        no_improve += 1
                        if no_improve >= self.cfg.early_stopping_patience:
                            self.logger.info(f"Early stopping at epoch {epoch}.")
                            break
                    else:
                        no_improve = 0  # reset during expansion stages

        # ── Restore best ─────────────────────────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)

        # ── Store trainer state ───────────────────────────────────────────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── Persist artifacts ─────────────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        with open(self.output_dir / "stage_stats.json", "w") as f:
            json.dump(stage_stats_history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── Evaluate ──────────────────────────────────────────────────────────
        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval(dl, name):
            if dl is None:
                return None
            model.eval()
            lgs, ys = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(device)
                    lgs.append(model(xb).cpu().numpy())
                    ys.append(yb.numpy())
            logits_arr = np.concatenate(lgs, axis=0)
            y_true = np.concatenate(ys, axis=0)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                cl = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(cm, cl, normalize=True, filename=f"cm_{name}.png")
            return {"accuracy": float(acc), "f1_macro": float(f1), "report": report, "confusion_matrix": cm.tolist()}

        results["val"] = _eval(dl_val, "val")
        results["test"] = _eval(dl_test, "test")
        results["stage_stats_history"] = stage_stats_history

        # ── Save model ────────────────────────────────────────────────────────
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
            "alpha": self.alpha,
            "beta": self.beta,
            "training_config": asdict(self.cfg),
        }, self.output_dir / "curriculum_disentangled_model.pt")
        self.logger.info("Model checkpoint saved.")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ── Inference helpers ─────────────────────────────────────────────────────

    def get_content_style_embeddings(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract z_content and z_style embeddings from trained model.

        Args:
            X: (N, T, C) or (N, C, T) windows — will be auto-transposed if needed.

        Returns:
            (z_content, z_style): each (N, latent_dim) numpy arrays.
        """
        assert self.model is not None, "Model not trained."
        assert self.mean_c is not None

        X_inp = X.copy()
        if X_inp.ndim == 3 and X_inp.shape[1] > X_inp.shape[2]:
            X_inp = np.transpose(X_inp, (0, 2, 1))
        X_std = self._apply_standardization(X_inp, self.mean_c, self.std_c)

        dummy_y = np.zeros(len(X_std), dtype=np.int64)
        ds = WindowDataset(X_std, dummy_y)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False)

        self.model.eval()
        z_c_list, z_s_list = [], []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(self.cfg.device)
                out = self.model(xb, return_all=True)
                z_c_list.append(out["z_content"].cpu().numpy())
                z_s_list.append(out["z_style"].cpu().numpy())

        return np.concatenate(z_c_list, axis=0), np.concatenate(z_s_list, axis=0)

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate on raw windows. Uses only z_content for gesture classification.
        X: (N, T, C) or (N, C, T).
        """
        assert self.model is not None
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None

        X_inp = X.copy()
        if X_inp.ndim == 3 and X_inp.shape[1] > X_inp.shape[2]:
            X_inp = np.transpose(X_inp, (0, 2, 1))
        X_std = self._apply_standardization(X_inp, self.mean_c, self.std_c)

        ds = WindowDataset(X_std, y)
        dl = DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )

        self.model.eval()
        lgs, ys = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                lgs.append(self.model(xb).cpu().numpy())
                ys.append(yb.numpy())

        logits = np.concatenate(lgs, axis=0)
        y_true = np.concatenate(ys, axis=0)
        y_pred = logits.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(self.class_ids)))

        if visualize and self.visualizer is not None:
            cl = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, cl, normalize=True, filename=f"cm_{split_name}.png"
            )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "logits": logits,
        }
