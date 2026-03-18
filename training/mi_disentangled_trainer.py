"""
Trainer for MI-Disentangled CNN-GRU (Hypothesis H5b — exp_58).

Extends WindowClassifierTrainer with:
  - CLUB-based MI upper bound minimization (z_content ⊥ subject)
  - CLUB-based MI upper bound maximization (z_style ↔ subject, optional)
  - Two-step update per batch (CLUB parameters and model parameters are
    on separate optimizers to prevent gradient coupling)
  - Post-training subject-probe accuracy (linear classifier on z_content)
    measured entirely on training-fold subjects — LOSO safe

Two-step update per batch
─────────────────────────
  Step 1 (CLUB update):
      loss_club = club_content.learning_loss(z_content.detach(), y_subj)
                + club_style.learning_loss(z_style.detach(), y_subj)
      club_opt.zero_grad(); loss_club.backward(); club_opt.step()

  Step 2 (main model update):
      L_total = L_gesture + α·L_subject + β·Î(z_content;subj) − γ·Î(z_style;subj)
      main_opt.zero_grad(); L_total.backward(); main_opt.step()
      club_opt.zero_grad()  # discard stale CLUB grads from Step 2

After training:
  - Linear probe (LogisticRegression) fitted on training z_content vectors,
    evaluated on val z_content vectors (val comes from training subjects).
  - Probe accuracy reported in results — should decrease as β increases.

LOSO invariant:
  - Test subject is NEVER passed to CLUB training or the subject probe.
  - CLUB only sees y_subj ∈ {0, ..., N_train − 1}.
  - Inference (evaluate_numpy) uses gesture_logits only; no subject label needed.
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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression

from training.trainer import (
    WindowClassifierTrainer,
    WindowDataset,
    get_worker_init_fn,
    seed_everything,
)
from models.mi_disentangled_cnn_gru import (
    CLUBEstimator,
    MIDisentangledCNNGRU,
)


# ─────────────────────────── Dataset ────────────────────────────────────────


class MIDisentangledDataset(Dataset):
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

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_gesture[idx], self.y_subject[idx]


# ─────────────────────────── Trainer ────────────────────────────────────────


class MIDisentangledTrainer(WindowClassifierTrainer):
    """
    Trainer for MIDisentangledCNNGRU (exp_58).

    Expects splits dict to contain:
        "train":               Dict[int, np.ndarray]   gesture_id → windows (N,T,C)
        "val":                 Dict[int, np.ndarray]
        "test":                Dict[int, np.ndarray]
        "train_subject_labels": Dict[int, np.ndarray]  gesture_id → subject indices
        "val_subject_labels":   Dict[int, np.ndarray]  gesture_id → subject indices
        "num_train_subjects":  int

    These extra keys are injected by exp_58's split builder which tracks which
    training subject contributed each window.

    Args:
        content_dim:          Dimensionality of z_content.
        style_dim:            Dimensionality of z_style.
        alpha:                Weight for subject classifier loss on z_style.
        beta:                 Weight for CLUB MI minimization on z_content.
                              Annealed linearly from 0 over beta_anneal_epochs.
        gamma:                Weight for CLUB MI maximization on z_style.
                              Annealed linearly from 0 over gamma_anneal_epochs.
                              Set to 0 to disable (alpha already pushes MI(z_style;subj)).
        beta_anneal_epochs:   Epochs to ramp beta from 0 → beta.
        gamma_anneal_epochs:  Epochs to ramp gamma from 0 → gamma.
        club_hidden_dim:      Hidden dim of the CLUB Gaussian heads.
        club_lr:              Learning rate for the CLUB networks (usually
                              smaller than the main model LR to keep q stable).
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
        beta: float = 0.2,
        gamma: float = 0.05,
        beta_anneal_epochs: int = 10,
        gamma_anneal_epochs: int = 10,
        club_hidden_dim: int = 64,
        club_lr: float = 1e-3,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.beta_anneal_epochs = beta_anneal_epochs
        self.gamma_anneal_epochs = gamma_anneal_epochs
        self.club_hidden_dim = club_hidden_dim
        self.club_lr = club_lr

    # ------------------------------------------------------------------ helpers

    def _build_subject_labels_array(
        self,
        subject_labels_dict: Dict[int, np.ndarray],
        class_ids: List[int],
    ) -> np.ndarray:
        """
        Build a flat subject-label array aligned with the windows returned by
        _prepare_splits_arrays (which iterates class_ids in sorted order).
        """
        parts = [
            subject_labels_dict[gid]
            for gid in class_ids
            if gid in subject_labels_dict
        ]
        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.int64)

    def _extract_z_content(
        self,
        model: MIDisentangledCNNGRU,
        X_std: np.ndarray,
        device: str,
    ) -> np.ndarray:
        """
        Pass standardised windows through the model (eval mode) and collect z_content.

        Args:
            X_std: (N, C, T) float32 — already standardised.
        Returns:
            (N, content_dim) float32 array.
        """
        model.eval()
        all_z: list[np.ndarray] = []
        ds = TensorDataset(torch.from_numpy(X_std).float())
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        with torch.no_grad():
            for (xb,) in dl:
                out = model(xb.to(device), return_all=True)
                all_z.append(out["z_content"].cpu().numpy())
        return np.concatenate(all_z, axis=0)

    def _compute_subject_probe(
        self,
        model: MIDisentangledCNNGRU,
        X_train_std: np.ndarray,
        y_subj_train: np.ndarray,
        X_val_std: np.ndarray,
        y_subj_val: np.ndarray,
        device: str,
    ) -> Dict[str, float]:
        """
        Fit a linear probe (LogisticRegression) on z_content from training data,
        evaluate on z_content from val data (both from training-fold subjects only).

        Measures: «how much subject identity remains in z_content?»
        If MI minimisation worked → val accuracy should approach 1/num_subjects.

        LOSO safety:
            All data comes from training-fold subjects.
            The LOSO test subject is never used here.

        Returns:
            dict with "train_acc" and "val_acc".
        """
        n_classes = len(np.unique(y_subj_train))
        if n_classes < 2 or len(X_val_std) == 0:
            return {"train_acc": float("nan"), "val_acc": float("nan")}

        self.logger.info("Computing subject probe on z_content (linear probe)...")

        Z_train = self._extract_z_content(model, X_train_std, device)
        Z_val = self._extract_z_content(model, X_val_std, device)

        clf = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", multi_class="auto", n_jobs=1
        )
        clf.fit(Z_train, y_subj_train)
        probe_train = float(clf.score(Z_train, y_subj_train))
        probe_val = float(clf.score(Z_val, y_subj_val))

        chance = 1.0 / n_classes
        self.logger.info(
            f"Subject probe on z_content: train={probe_train:.4f}, "
            f"val={probe_val:.4f}  (chance={chance:.4f}, "
            f"val/chance ratio={probe_val/chance:.2f})"
        )
        return {"train_acc": probe_train, "val_acc": probe_val}

    # ------------------------------------------------------------------ main fit

    def fit(self, splits: Dict) -> Dict:
        """
        Train MIDisentangledCNNGRU with the two-step CLUB update.

        Expects splits to contain the standard keys ("train", "val", "test")
        plus "train_subject_labels", "val_subject_labels", "num_train_subjects".
        """
        seed_everything(self.cfg.seed)

        # ── 1. Prepare window/label arrays (standard base-class helper) ──────
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        # ── 2. Extract and validate subject labels ────────────────────────────
        for key in ("train_subject_labels", "val_subject_labels", "num_train_subjects"):
            if key not in splits:
                raise ValueError(
                    f"MIDisentangledTrainer requires '{key}' in splits. "
                    "Use exp_58's split builder which injects subject provenance."
                )

        y_subj_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        y_subj_val = self._build_subject_labels_array(
            splits["val_subject_labels"], class_ids
        )
        num_train_subjects = int(splits["num_train_subjects"])

        assert len(y_subj_train) == len(y_train), (
            f"Subject labels ({len(y_subj_train)}) must align "
            f"with gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Subject labels: {num_train_subjects} training subjects, "
            f"distribution: {np.bincount(y_subj_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) for Conv1d ───────────────────
        # _prepare_splits_arrays returns (N, T, C) per docstring.
        # We detect by shape: T > C (T=600, C=8 typically).
        if X_train.ndim == 3:
            _, d1, d2 = X_train.shape
            if d1 > d2:
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 4. Per-channel standardisation (computed on training data only) ───
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train_std = self._apply_standardization(X_train, mean_c, std_c)
        X_val_std = (
            self._apply_standardization(X_val, mean_c, std_c) if len(X_val) > 0
            else X_val
        )
        X_test_std = (
            self._apply_standardization(X_test, mean_c, std_c) if len(X_test) > 0
            else X_test
        )
        self.logger.info("Applied per-channel standardisation (training stats).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        device = self.cfg.device

        # ── 5. Build model and CLUB estimators ────────────────────────────────
        model = MIDisentangledCNNGRU(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dropout=self.cfg.dropout,
        ).to(device)

        club_content = CLUBEstimator(
            z_dim=self.content_dim,
            num_conditions=num_train_subjects,
            hidden_dim=self.club_hidden_dim,
        ).to(device)

        club_style = CLUBEstimator(
            z_dim=self.style_dim,
            num_conditions=num_train_subjects,
            hidden_dim=self.club_hidden_dim,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        club_params = (
            sum(p.numel() for p in club_content.parameters())
            + sum(p.numel() for p in club_style.parameters())
        )
        self.logger.info(
            f"MIDisentangledCNNGRU: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, content_dim={self.content_dim}, "
            f"style_dim={self.style_dim}, params={total_params:,} | "
            f"CLUB params={club_params:,}"
        )

        # ── 6. DataLoaders ────────────────────────────────────────────────────
        ds_train = MIDisentangledDataset(X_train_std, y_train, y_subj_train)
        ds_val = WindowDataset(X_val_std, y_val) if len(X_val_std) > 0 else None
        ds_test = WindowDataset(X_test_std, y_test) if len(X_test_std) > 0 else None

        worker_init = get_worker_init_fn(self.cfg.seed)
        gen = torch.Generator().manual_seed(self.cfg.seed)

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
            generator=gen,
        )
        dl_val = (
            DataLoader(
                ds_val,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
            if ds_val else None
        )
        dl_test = (
            DataLoader(
                ds_test,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
            if ds_test else None
        )

        # ── 7. Loss functions ─────────────────────────────────────────────────
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = counts.sum() / (counts + 1e-8)
            cw = cw / cw.mean()
            gesture_criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(device)
            )
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
        else:
            gesture_criterion = nn.CrossEntropyLoss()

        subject_criterion = nn.CrossEntropyLoss()

        # ── 8. Optimisers ─────────────────────────────────────────────────────
        # Two separate optimisers ensure CLUB parameters and model parameters
        # are never updated by each other's gradients.
        main_opt = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        club_opt = optim.Adam(
            list(club_content.parameters()) + list(club_style.parameters()),
            lr=self.club_lr,
        )
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            main_opt, mode="min", factor=0.5, patience=5
        )

        # ── 9. Training loop ──────────────────────────────────────────────────
        history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_loss": [], "subject_loss": [],
            "mi_content_loss": [], "mi_style_loss": [],
            "club_learning_loss": [],
            "beta": [], "gamma": [],
        }
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            club_content.train()
            club_style.train()

            # Linear annealing of MI penalty weights
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))
            current_gamma = self.gamma * min(1.0, epoch / max(1, self.gamma_anneal_epochs))

            ep_total = 0
            ep_correct = 0
            ep_total_loss = 0.0
            ep_gesture_loss = 0.0
            ep_subject_loss = 0.0
            ep_mi_content = 0.0
            ep_mi_style = 0.0
            ep_club_loss = 0.0

            for windows, y_gest, y_subj in dl_train:
                windows = windows.to(device)
                y_gest = y_gest.to(device)
                y_subj = y_subj.to(device)
                B = windows.size(0)

                # ── Step 1: Update CLUB networks ─────────────────────────────
                # Get latent vectors without gradients to encoder
                with torch.no_grad():
                    out_nograd = model(windows, return_all=True)
                z_c_detached = out_nograd["z_content"].detach()
                z_s_detached = out_nograd["z_style"].detach()

                club_opt.zero_grad()
                loss_club = (
                    club_content.learning_loss(z_c_detached, y_subj)
                    + club_style.learning_loss(z_s_detached, y_subj)
                )
                loss_club.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(club_content.parameters()) + list(club_style.parameters()),
                    max_norm=5.0,
                )
                club_opt.step()

                # ── Step 2: Update main model ─────────────────────────────────
                # Fresh forward pass so z_content/z_style have valid gradients
                main_opt.zero_grad()
                outputs = model(windows, return_all=True)
                z_c = outputs["z_content"]
                z_s = outputs["z_style"]

                L_gesture = gesture_criterion(outputs["gesture_logits"], y_gest)
                L_subject = subject_criterion(outputs["subject_logits"], y_subj)

                # MI bounds from CLUB (CLUB params will accumulate gradients
                # here but these are discarded by club_opt.zero_grad() below)
                mi_c = club_content.mi_upper_bound(z_c, y_subj)
                mi_s = club_style.mi_upper_bound(z_s, y_subj)

                L_total = (
                    L_gesture
                    + self.alpha * L_subject
                    + current_beta * mi_c
                    - current_gamma * mi_s
                )

                L_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                main_opt.step()

                # Discard any CLUB grad accumulation from Step 2
                club_opt.zero_grad()

                # ── Bookkeeping ───────────────────────────────────────────────
                ep_total += B
                ep_correct += (outputs["gesture_logits"].argmax(dim=1) == y_gest).sum().item()
                ep_total_loss += L_total.item() * B
                ep_gesture_loss += L_gesture.item() * B
                ep_subject_loss += L_subject.item() * B
                ep_mi_content += mi_c.item() * B
                ep_mi_style += mi_s.item() * B
                ep_club_loss += loss_club.item() * B

            n = max(1, ep_total)
            train_loss = ep_total_loss / n
            train_acc = ep_correct / n
            avg_gest = ep_gesture_loss / n
            avg_subj = ep_subject_loss / n
            avg_mi_c = ep_mi_content / n
            avg_mi_s = ep_mi_style / n
            avg_club = ep_club_loss / n

            # ── Validation (gesture accuracy only — no subject label needed) ──
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        val_loss_sum += gesture_criterion(logits, yb).item() * yb.size(0)
                        val_correct += (logits.argmax(dim=1) == yb).sum().item()
                        val_total += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss = float("nan")
                val_acc = float("nan")

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["gesture_loss"].append(avg_gest)
            history["subject_loss"].append(avg_subj)
            history["mi_content_loss"].append(avg_mi_c)
            history["mi_style_loss"].append(avg_mi_s)
            history["club_learning_loss"].append(avg_club)
            history["beta"].append(current_beta)
            history["gamma"].append(current_gamma)

            self.logger.info(
                f"[Ep {epoch:02d}/{self.cfg.epochs}] "
                f"loss={train_loss:.4f} (gest={avg_gest:.4f}, "
                f"subj={avg_subj:.4f}, "
                f"MI_c={avg_mi_c:.4f}×β={current_beta:.3f}, "
                f"MI_s={avg_mi_s:.4f}×γ={current_gamma:.3f}), "
                f"club={avg_club:.4f} | "
                f"val_loss={val_loss:.4f}, "
                f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
            )

            # ── Early stopping on val gesture loss ────────────────────────────
            if dl_val is not None:
                main_scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}.")
                        break

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
            self.logger.info("Restored best checkpoint by val_loss.")

        # ── 10. Store trainer state (needed by evaluate_numpy) ────────────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 11. Subject probe (LOSO-safe: only training-fold subjects) ────────
        probe_results = self._compute_subject_probe(
            model=model,
            X_train_std=X_train_std,
            y_subj_train=y_subj_train,
            X_val_std=X_val_std,
            y_subj_val=y_subj_val,
            device=device,
        )

        # ── 12. Save artefacts ────────────────────────────────────────────────
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── 13. Evaluate on val / test (gesture accuracy) ─────────────────────
        results: Dict = {
            "class_ids": class_ids,
            "class_names": class_names,
            "subject_probe": probe_results,
            "num_train_subjects": num_train_subjects,
        }

        def _eval_loader(dloader: Optional[DataLoader], split_name: str) -> Optional[Dict]:
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    logits = model(xb.to(device))
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits, axis=0)
            y_true = np.concatenate(all_y, axis=0)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
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

        # Save model checkpoint
        model_path = self.output_dir / "mi_disentangled_cnn_gru.pt"
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
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "training_config": asdict(self.cfg),
            },
            model_path,
        )
        self.logger.info(f"Model saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ────────────────────────── Inference ───────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate using gesture_logits from z_content only.
        No subject information is needed or used — LOSO safe.

        Args:
            X: (N, T, C) or (N, C, T) windows (shape detected automatically).
            y: (N,) integer class indices (0-indexed, aligned with class_ids).
        """
        assert self.model is not None, "Model not yet trained (call fit first)"
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None and self.class_names is not None

        X_in = X.copy()
        if X_in.ndim == 3:
            _, d1, d2 = X_in.shape
            if d1 > d2:
                X_in = np.transpose(X_in, (0, 2, 1))  # (N,T,C) → (N,C,T)

        Xs = self._apply_standardization(X_in, self.mean_c, self.std_c)

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
                logits = self.model(xb.to(self.cfg.device))
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())

        logits_arr = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y, axis=0)
        y_pred = logits_arr.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        num_classes = len(self.class_ids)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        if visualize and self.visualizer is not None:
            labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, labels, normalize=True, filename=f"cm_{split_name}.png"
            )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "logits": logits_arr,
        }
