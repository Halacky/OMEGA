"""
Trainer with curriculum learning: train from similar to distant subjects.

Extends WindowClassifierTrainer with a curriculum schedule:
- Training subjects are ordered by similarity to the test subject
- Training begins with k_init nearest subjects
- Every `expand_every` epochs the next closest subject is added
- After all subjects are included, training continues for `consolidation_epochs`

Expects splits dict to contain extra keys (injected by experiment file):
    "train_subject_labels": Dict[int, np.ndarray]  — gesture_id → subject index per window
    "train_subject_order":  List[int]               — subject indices ordered similar→distant
    "num_train_subjects":   int
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from training.trainer import WindowClassifierTrainer, WindowDataset, get_worker_init_fn, seed_everything


class IndexedWindowDataset(Dataset):
    """Dataset that also stores a subject index per sample for curriculum filtering."""

    def __init__(self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.subject_ids = subject_ids  # kept as numpy for masking

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_mask_for_subjects(self, allowed_subject_indices: set) -> np.ndarray:
        """Return boolean mask of samples belonging to allowed subjects."""
        return np.isin(self.subject_ids, list(allowed_subject_indices))


class CurriculumTrainer(WindowClassifierTrainer):
    """
    WindowClassifierTrainer with curriculum-based subject ordering.

    Hyperparameters (beyond parent):
        k_init:               number of nearest subjects to start with
        expand_every:         add next subject every N epochs
        consolidation_epochs: epochs on full training set after all subjects added
        lr_on_expand:         if set, reset LR to this value when expanding subject set
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        k_init: int = 1,
        expand_every: int = 10,
        consolidation_epochs: int = 20,
        lr_on_expand: Optional[float] = None,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.k_init = k_init
        self.expand_every = expand_every
        self.consolidation_epochs = consolidation_epochs
        self.lr_on_expand = lr_on_expand

    # ------------------------------------------------------------------
    # Helper: build subject label array aligned with _prepare_splits_arrays
    # ------------------------------------------------------------------
    def _build_subject_labels_array(
        self,
        subject_labels_dict: Dict[int, np.ndarray],
        class_ids: List[int],
    ) -> np.ndarray:
        """Build flat subject label array aligned with _prepare_splits_arrays order."""
        parts = []
        for gid in class_ids:
            if gid in subject_labels_dict:
                parts.append(subject_labels_dict[gid])
        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    # ------------------------------------------------------------------
    # Curriculum schedule computation
    # ------------------------------------------------------------------
    def _compute_schedule(self, num_subjects: int, subject_order: List[int]) -> List[Tuple[int, int, set]]:
        """
        Compute curriculum schedule.

        Returns list of (start_epoch, end_epoch, allowed_subject_indices).
        """
        stages = []
        epoch_cursor = 1

        num_expand_stages = num_subjects - self.k_init  # how many times we expand
        # Stage 0: k_init subjects
        current_subjects = set(subject_order[:self.k_init])
        if num_expand_stages <= 0:
            # All subjects from the start
            total = self.consolidation_epochs + self.expand_every
            stages.append((epoch_cursor, epoch_cursor + total - 1, set(subject_order)))
            return stages

        stages.append((
            epoch_cursor,
            epoch_cursor + self.expand_every - 1,
            set(current_subjects),
        ))
        epoch_cursor += self.expand_every

        # Expansion stages: add one subject at a time
        for i in range(self.k_init, num_subjects):
            current_subjects = set(subject_order[:i + 1])
            stage_epochs = self.expand_every
            stages.append((
                epoch_cursor,
                epoch_cursor + stage_epochs - 1,
                set(current_subjects),
            ))
            epoch_cursor += stage_epochs

        # Consolidation: all subjects
        if self.consolidation_epochs > 0:
            stages.append((
                epoch_cursor,
                epoch_cursor + self.consolidation_epochs - 1,
                set(subject_order),
            ))

        return stages

    # ------------------------------------------------------------------
    # fit() — main training with curriculum
    # ------------------------------------------------------------------
    def fit(self, splits: Dict) -> Dict:
        """Train with curriculum: start from similar subjects, expand gradually."""
        seed_everything(self.cfg.seed)

        # --- 1. Prepare arrays (standard parent method) ---
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # --- 2. Extract subject metadata ---
        if "train_subject_labels" not in splits:
            raise ValueError(
                "CurriculumTrainer requires 'train_subject_labels' in splits. "
                "Use the experiment file that injects subject provenance."
            )
        y_subject = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        subject_order = splits["train_subject_order"]  # List[int], similar→distant
        num_train_subjects = splits["num_train_subjects"]

        assert len(y_subject) == len(y_train), (
            f"Subject labels ({len(y_subject)}) must match "
            f"gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Curriculum: {num_train_subjects} subjects, order={subject_order}, "
            f"k_init={self.k_init}, expand_every={self.expand_every}, "
            f"consolidation={self.consolidation_epochs}"
        )

        # --- 3. Transpose (N, T, C) → (N, C, T) ---
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

        # --- 4. Channel standardization (on ALL training data) ---
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardization.")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # --- 5. Create model ---
        model_type = getattr(self.cfg, "model_type", "cnn_gru_attention")
        model = self._create_model(in_channels, num_classes, model_type).to(self.cfg.device)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model: {model_type}, params={total_params:,}")

        # --- 6. Datasets ---
        ds_train = IndexedWindowDataset(X_train, y_train, y_subject)
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        dl_val = DataLoader(
            ds_val, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        ) if ds_test else None

        # --- 7. Loss ---
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Class weights: {cw.round(3).tolist()}")
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        # --- 8. Optimizer ---
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # --- 9. Compute curriculum schedule ---
        schedule = self._compute_schedule(num_train_subjects, subject_order)
        total_epochs = schedule[-1][1] if schedule else self.cfg.epochs

        self.logger.info(f"Curriculum schedule ({len(schedule)} stages, {total_epochs} total epochs):")
        for i, (start, end, subjs) in enumerate(schedule):
            self.logger.info(
                f"  Stage {i}: epochs {start}-{end}, "
                f"subjects={sorted(subjs)} ({len(subjs)}/{num_train_subjects})"
            )

        # --- 10. Training loop with curriculum ---
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "stage": [], "num_active_subjects": [],
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        device = self.cfg.device

        current_stage_idx = 0
        current_dl_train = None

        for epoch in range(1, total_epochs + 1):
            # --- Determine which stage we're in ---
            while current_stage_idx < len(schedule) - 1:
                if epoch > schedule[current_stage_idx][1]:
                    current_stage_idx += 1
                else:
                    break

            stage_start, stage_end, allowed_subjects = schedule[current_stage_idx]

            # --- Rebuild DataLoader if stage changed ---
            if epoch == stage_start:
                mask = ds_train.get_mask_for_subjects(allowed_subjects)
                active_indices = np.where(mask)[0].tolist()
                subset = Subset(ds_train, active_indices)
                current_dl_train = DataLoader(
                    subset,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.num_workers,
                    pin_memory=True,
                    worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
                    generator=torch.Generator().manual_seed(self.cfg.seed + epoch),
                )
                self.logger.info(
                    f"[Epoch {epoch}] Stage transition → "
                    f"{len(allowed_subjects)} subjects, "
                    f"{len(active_indices)} training samples"
                )
                # Optionally reset LR on expansion
                if self.lr_on_expand is not None and epoch > 1:
                    for pg in optimizer.param_groups:
                        pg["lr"] = self.lr_on_expand
                    self.logger.info(f"  LR reset to {self.lr_on_expand}")

            # --- Train epoch ---
            model.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            for xb, yb in current_dl_train:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                if hasattr(model, '_aux_loss') and model._aux_loss is not None:
                    loss = loss + model._aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = xb.size(0)
                train_loss_sum += loss.item() * bs
                preds = logits.argmax(dim=1)
                train_correct += (preds == yb).sum().item()
                train_total += bs

            train_loss = train_loss_sum / max(1, train_total)
            train_acc = train_correct / max(1, train_total)

            # --- Validation ---
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)
                        loss_v = criterion(logits, yb)
                        val_loss_sum += loss_v.item() * yb.size(0)
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
            history["stage"].append(current_stage_idx)
            history["num_active_subjects"].append(len(allowed_subjects))

            self.logger.info(
                f"[Epoch {epoch:02d}/{total_epochs}] "
                f"Stage {current_stage_idx} ({len(allowed_subjects)} subj) | "
                f"Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f}"
            )

            # --- Early stopping (only in consolidation stage) ---
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    # Only trigger early stopping during the last (consolidation) stage
                    if current_stage_idx == len(schedule) - 1:
                        no_improve += 1
                        if no_improve >= self.cfg.early_stopping_patience:
                            self.logger.info(f"Early stopping at epoch {epoch}")
                            break
                    else:
                        # During expansion stages, just track but don't stop
                        no_improve = 0

        # --- Restore best ---
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # --- Store trainer state ---
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # --- Save history ---
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # --- Evaluate val/test ---
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

        results = {"class_ids": class_ids, "class_names": class_names}

        def eval_loader(dloader, split_name):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(device)
                    out = model(xb)
                    all_logits.append(out.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits, axis=0)
            y_true = np.concatenate(all_y, axis=0)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                cl = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, cl, normalize=True, filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report": report,
                "confusion_matrix": cm.tolist(),
            }

        results["val"] = eval_loader(dl_val, "val")
        results["test"] = eval_loader(dl_test, "test")

        # --- Save model ---
        model_path = self.output_dir / "curriculum_model.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "class_ids": class_ids,
            "mean": mean_c,
            "std": std_c,
            "window_size": window_size,
            "curriculum_config": {
                "k_init": self.k_init,
                "expand_every": self.expand_every,
                "consolidation_epochs": self.consolidation_epochs,
                "subject_order": subject_order,
            },
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        results_path = self.output_dir / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results
