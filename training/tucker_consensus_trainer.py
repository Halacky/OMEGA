"""
Trainer for ChannelBandTuckerConsensusEMG.

Loss:
    L_total = L_gesture  +  λ_adv  * L_subject_adv  +  λ_cons * L_temporal_cons

  L_gesture:      CrossEntropy on gesture logits (main task)
  L_subject_adv:  CrossEntropy on adversary logits for subject ID — gradient
                  flows through GRL, so the channel-factor encoder U_ch is
                  pushed to be subject-uninformative (DANN)
  L_temporal_cons: KL divergence between full-window and quarter-window
                  predictions — encourages temporal robustness of the Tucker
                  core tensor

GRL alpha schedule (DANN):
    alpha(p) = 2 / (1 + exp(-10 * p)) - 1,   p = epoch / total_epochs
    Rises smoothly from ~0 to ~1, stabilising gesture learning before the
    adversary ramps up.

LOSO Safety:
  - Channel standardisation computed from training-fold data only.
  - Val / test normalised with training-fold mean_c / std_c (no test leakage).
  - model.eval() freezes BatchNorm running statistics at val/test time.
  - Subject labels used only in the training loop; not created for val/test.
  - model.forward(x, return_all=False) at inference: adversary + consensus
    loss are completely bypassed.
  - No test-time adaptation.
"""

import json
import logging
import math
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
from models.channel_band_tucker_consensus_loso import ChannelBandTuckerConsensusEMG


# ─────────────────────── Dataset with subject labels ──────────────────────────


class TuckerDisentangledDataset(Dataset):
    """Dataset yielding (window, gesture_label, subject_label) triples."""

    def __init__(
        self,
        X: np.ndarray,
        y_gesture: np.ndarray,
        y_subject: np.ndarray,
    ):
        self.X         = torch.from_numpy(X).float()
        self.y_gesture = torch.from_numpy(y_gesture).long()
        self.y_subject = torch.from_numpy(y_subject).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_gesture[idx], self.y_subject[idx]


# ─────────────────────────────────── Trainer ──────────────────────────────────


class TuckerConsensusTrainer(WindowClassifierTrainer):
    """
    Trainer for the Channel-Band Tucker Consensus model (Hypothesis 5).

    Inherits from WindowClassifierTrainer and overrides fit().
    evaluate_numpy() from the parent class works directly because the model
    returns a plain tensor when called with return_all=False.

    Expects the splits dict (from _build_splits_with_subject_labels) to contain:
        "train":               Dict[gesture_id → np.ndarray (N, T, C)]
        "val":                 Dict[gesture_id → np.ndarray (N, T, C)]
        "test":                Dict[gesture_id → np.ndarray (N, T, C)]
        "train_subject_labels": Dict[gesture_id → np.ndarray (N,)]  subject indices
        "num_train_subjects":   int

    Constructor args (beyond WindowClassifierTrainer):
        lambda_adv:   Weight for adversarial subject loss (default 0.3)
        lambda_cons:  Weight for temporal consistency loss (default 0.1)
        r_c:          Channel Tucker rank (default 8)
        r_f:          Frequency Tucker rank (default 16)
        n_fft:        STFT window size (default 64 → F=33 bins)
        hop_length:   STFT hop length (default 16 → T'=34 for 600-sample windows)
        hidden_dim:   Classifier hidden units (default 128)
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # Loss weights
        lambda_adv: float  = 0.3,
        lambda_cons: float = 0.1,
        # Tucker architecture
        r_c: int        = 8,
        r_f: int        = 16,
        # STFT parameters
        n_fft: int      = 64,
        hop_length: int = 16,
        # Classifier
        hidden_dim: int = 128,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.lambda_adv  = lambda_adv
        self.lambda_cons = lambda_cons
        self.r_c         = r_c
        self.r_f         = r_f
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.hidden_dim  = hidden_dim

    # ── Helper: flat subject-label array aligned with _prepare_splits_arrays ──

    def _build_subject_labels_array(
        self,
        subject_labels_dict: Dict[int, np.ndarray],
        class_ids: List[int],
    ) -> np.ndarray:
        """
        Flatten subject labels in the same gesture-ID order that
        _prepare_splits_arrays uses for y_train, so indices align.
        """
        parts = []
        for gid in class_ids:
            if gid in subject_labels_dict:
                parts.append(subject_labels_dict[gid])
        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    # ── DANN GRL alpha schedule ────────────────────────────────────────────────

    @staticmethod
    def _grl_alpha(epoch: int, total_epochs: int) -> float:
        """
        DANN gradient reversal schedule.
        alpha(p) = 2 / (1 + exp(-10 * p)) - 1,  p = epoch / total_epochs.
        Smoothly rises from ~0 to ~1 over training.
        """
        p = epoch / max(1, total_epochs)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train ChannelBandTuckerConsensusEMG.

        Three-axis subject-invariance:
          (1) GRL adversary on channel Tucker factors (L_subject_adv)
          (2) Soft Freq AGC inside the model (learnable γ_f)
          (3) Quarter-window temporal consensus loss (L_temporal_cons)
        """
        seed_everything(self.cfg.seed)

        # ── 1. Flat arrays from splits ─────────────────────────────────────
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        # ── 2. Subject-label array for adversary ───────────────────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "TuckerConsensusTrainer requires 'train_subject_labels' in splits. "
                "Build splits with _build_splits_with_subject_labels()."
            )
        y_subject_train  = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        if len(y_subject_train) != len(y_train):
            raise ValueError(
                f"Subject label length ({len(y_subject_train)}) != "
                f"gesture label length ({len(y_train)}). "
                "Check _build_splits_with_subject_labels alignment."
            )
        self.logger.info(
            f"Training: {len(y_train)} windows, {num_train_subjects} subjects"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) ───────────────────────────
        # Conv1d and STFT both expect channels-first (B, C, T).
        # T >> C: if dim1 > dim2 the array is (N, T, C) and needs transposing.
        if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
            X_train = np.transpose(X_train, (0, 2, 1))
            if len(X_val)  > 0: X_val  = np.transpose(X_val,  (0, 2, 1))
            if len(X_test) > 0: X_test = np.transpose(X_test, (0, 2, 1))
            self.logger.info(f"Transposed windows to (N, C, T): {X_train.shape}")

        in_channels = X_train.shape[1]
        t_samples   = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 4. Channel standardisation (training data only) ────────────────
        # mean_c / std_c shape: (1, C, 1) — broadcast with (N, C, T).
        # Val / test use the SAME statistics → no test-time adaptation.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val)  > 0: X_val  = self._apply_standardization(X_val,  mean_c, std_c)
        if len(X_test) > 0: X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation (training stats only).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Build model ────────────────────────────────────────────────
        model = ChannelBandTuckerConsensusEMG(
            n_classes        = num_classes,
            n_subjects_train = num_train_subjects,
            n_channels       = in_channels,
            t_samples        = t_samples,
            n_fft            = self.n_fft,
            hop_length       = self.hop_length,
            r_c              = self.r_c,
            r_f              = self.r_f,
            hidden_dim       = self.hidden_dim,
            dropout          = self.cfg.dropout,
        ).to(self.cfg.device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        core_dim = self.r_c * self.r_f
        self.logger.info(
            f"ChannelBandTuckerConsensusEMG | "
            f"in_ch={in_channels}, T={t_samples}, classes={num_classes}, "
            f"subjects={num_train_subjects}, "
            f"r_c={self.r_c}, r_f={self.r_f}, core_dim={core_dim}, "
            f"params={n_params:,}"
        )
        self.logger.info(
            f"  STFT: n_fft={self.n_fft}, hop={self.hop_length} "
            f"→ F={self.n_fft//2+1}, T'={(t_samples-self.n_fft)//self.hop_length+1}"
        )

        # ── 6. Datasets & DataLoaders ─────────────────────────────────────
        ds_train = TuckerDisentangledDataset(X_train, y_train, y_subject_train)
        ds_val   = WindowDataset(X_val,  y_val)  if len(X_val)  > 0 else None
        ds_test  = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        n_workers      = self.cfg.num_workers

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,           # ensures BatchNorm always sees ≥2 samples
            num_workers=n_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if n_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
        ) if ds_test else None

        # ── 7. Loss functions ─────────────────────────────────────────────
        # Gesture: optionally class-weighted.
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            w_tensor = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_criterion = nn.CrossEntropyLoss(weight=w_tensor)
        else:
            gesture_criterion = nn.CrossEntropyLoss()

        # Subject adversary: unweighted (subjects roughly balanced in LOSO).
        subject_criterion = nn.CrossEntropyLoss()

        # ── 8. Optimizer & scheduler ──────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # NOTE: verbose kwarg removed in PyTorch 2.4+ → omitted deliberately.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8,
        )

        # ── 9. Training loop ──────────────────────────────────────────────
        history = {
            "train_loss": [], "val_loss":    [],
            "train_acc":  [], "val_acc":     [],
            "gest_loss":  [], "subj_loss":   [],
            "cons_loss":  [], "grl_alpha":   [],
        }
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        device        = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            grl_alpha = self._grl_alpha(epoch, self.cfg.epochs)

            ep_total = ep_correct = 0
            ep_loss_total = ep_loss_gest = ep_loss_subj = ep_loss_cons = 0.0

            for windows, gesture_labels, subject_labels in dl_train:
                windows        = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                # Training forward: returns dict with all auxiliary outputs.
                out = model(
                    windows,
                    subject_ids=subject_labels,
                    grl_alpha=grl_alpha,
                    return_all=True,
                )

                # Loss 1: gesture classification (primary objective)
                L_gesture = gesture_criterion(out["logits"], gesture_labels)

                # Loss 2: adversarial subject loss — gradient reversed by GRL
                # so U_ch is pushed to be subject-uninformative.
                L_subject = subject_criterion(out["adv_logits"], subject_labels)

                # Loss 3: temporal consensus — quarter-window predictions should
                # match full-window predictions (computed inside model.forward).
                L_cons = out["cons_loss"]

                total_loss = (
                    L_gesture
                    + self.lambda_adv  * L_subject
                    + self.lambda_cons * L_cons
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_loss_total += total_loss.item() * bs
                ep_loss_gest  += L_gesture.item()  * bs
                ep_loss_subj  += L_subject.item()  * bs
                ep_loss_cons  += L_cons.item()      * bs
                preds = out["logits"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total   += bs

            n = max(ep_total, 1)
            train_loss = ep_loss_total / n
            train_acc  = ep_correct    / n
            avg_gest   = ep_loss_gest  / n
            avg_subj   = ep_loss_subj  / n
            avg_cons   = ep_loss_cons  / n

            # ── Validation ─────────────────────────────────────────────────
            # model.eval() freezes BatchNorm → no test-subject stats.
            # return_all=False → adversary and consensus bypassed.
            if dl_val is not None:
                model.eval()
                val_correct = val_total = 0
                val_loss_sum = 0.0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb, return_all=False)  # tensor
                        loss   = gesture_criterion(logits, yb)
                        val_loss_sum += loss.item() * yb.size(0)
                        val_correct  += (logits.argmax(1) == yb).sum().item()
                        val_total    += yb.size(0)
                val_loss = val_loss_sum / max(val_total, 1)
                val_acc  = val_correct  / max(val_total, 1)
            else:
                val_loss = val_acc = float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["gest_loss"].append(avg_gest)
            history["subj_loss"].append(avg_subj)
            history["cons_loss"].append(avg_cons)
            history["grl_alpha"].append(grl_alpha)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} "
                f"(gest={avg_gest:.4f}, subj_adv={avg_subj:.4f}, "
                f"cons={avg_cons:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"alpha={grl_alpha:.3f}"
            )

            # ── Early stopping ─────────────────────────────────────────────
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

        # Restore best checkpoint (by validation gesture loss)
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── Store trainer state (required by evaluate_numpy) ───────────────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels

        # ── Save training history ──────────────────────────────────────────
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png"
            )

        # ── In-fold evaluation (val & test splits) ─────────────────────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb     = xb.to(device)
                    logits = model(xb, return_all=False)   # tensor
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits, axis=0)
            y_true     = np.concatenate(all_y,      axis=0)
            y_pred     = logits_arr.argmax(axis=1)
            acc    = accuracy_score(y_true, y_pred)
            f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
            report = classification_report(y_true, y_pred, output_dict=True,
                                           zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True,
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

        # ── Save model checkpoint ──────────────────────────────────────────
        model_path = self.output_dir / "tucker_consensus.pt"
        torch.save(
            {
                "state_dict":     model.state_dict(),
                "in_channels":    in_channels,
                "t_samples":      t_samples,
                "num_classes":    num_classes,
                "num_subjects":   num_train_subjects,
                "class_ids":      class_ids,
                "mean":           mean_c,
                "std":            std_c,
                "architecture": {
                    "r_c":        self.r_c,
                    "r_f":        self.r_f,
                    "n_fft":      self.n_fft,
                    "hop_length": self.hop_length,
                    "hidden_dim": self.hidden_dim,
                },
                "loss_config": {
                    "lambda_adv":  self.lambda_adv,
                    "lambda_cons": self.lambda_cons,
                    "grl_schedule": "DANN: alpha=2/(1+exp(-10*p))-1",
                },
                "training_config": asdict(self.cfg),
            },
            model_path,
        )
        self.logger.info(f"Model saved: {model_path}")

        results_path = self.output_dir / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results
