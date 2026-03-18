"""
Trainer for DSDRNetECAPA — cyclic inter-subject reconstruction with multi-scale AdaIN.

Training procedure (per batch):
    1. L_cls        : CrossEntropy on content embedding (gesture classification)
    2. L_self_recon : MSE(decode(encode(x_A)), x_A)  — autoencoder regularisation
    3. L_cycle      : MSE(decode(adain(encode(decode(adain(f_A, f_B))), f_A)), x_A)
                      i.e. A → B-style → back to A-style ≈ original A
    4. L_perceptual : mean MSE per scale between encode(decode(f_A)) and f_A features
                      (feature-level autoencoder consistency)

Loss schedule:
    Reconstruction losses (L_self_recon, L_cycle, L_perceptual) are linearly
    ramped up from 0 over the first `cycle_warmup_frac` fraction of epochs.
    This prevents unstable gradients before the encoder has warmed up.

LOSO Compliance — zero data leakage guaranteed:
    ┌──────────────────────────────────────────────────────────────┐
    │ Training                                                     │
    │  • mean_c, std_c  → computed from TRAINING windows only     │
    │  • style pairs    → sampled within each batch, all from      │
    │                     TRAINING subjects (subject index 0..N-1) │
    │  • f_A, f_B       → encode() called on training windows only │
    │  • decode()       → called on stylised training features only│
    │                                                              │
    │ Inference (evaluate_numpy)                                   │
    │  • model.forward(x) → content pathway ONLY                  │
    │  • no AdaIN, no decoder, no test-subject statistics used     │
    │  • test data normalised with training mean_c/std_c           │
    └──────────────────────────────────────────────────────────────┘
"""

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader, Dataset

from config.base import TrainingConfig
from models.dsdrnet_ecapa_adain_loso import DSDRNetECAPA
from training.datasets import WindowDataset
from training.trainer import WindowClassifierTrainer
from utils.logging import get_worker_init_fn, seed_everything


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CyclicWindowDataset(Dataset):
    """
    EMG window dataset that additionally exposes per-sample subject indices.

    Used only for the training split so that the training loop can form
    cross-subject style pairs.

    LOSO note: `subj` must contain TRAINING-subject indices (0 … N_train−1).
    Test-subject data is never stored here.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, subj: np.ndarray):
        """
        Args:
            X:    (N, C, T) standardised windows — training subjects ONLY
            y:    (N,)      gesture class indices (0-based, matching class_ids)
            subj: (N,)      subject indices, 0 … num_train_subjects − 1
        """
        assert X.ndim == 3, f"Expected (N, C, T), got {X.shape}"
        assert len(X) == len(y) == len(subj), "X, y, subj must have equal length"
        self.X    = torch.from_numpy(X.astype(np.float32))
        self.y    = torch.from_numpy(y.astype(np.int64))
        self.subj = torch.from_numpy(subj.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subj[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Style-pair sampling helper
# ─────────────────────────────────────────────────────────────────────────────

def _sample_style_indices(subj_labels: np.ndarray) -> np.ndarray:
    """
    For each sample i, select a random index j such that subj_labels[j] ≠ subj_labels[i].

    Guarantees cross-subject style pairing inside a single batch.
    Falls back to the adjacent sample when all batch members share the same subject
    (edge case that should only arise with very small batches or single-subject data).

    LOSO note: subj_labels contains only training-subject indices (≥ 0).
    Test-subject data is never present in any batch that calls this function.

    Returns:
        (N,) int64 array of style-sample indices into the batch
    """
    B = len(subj_labels)
    style_idx = np.empty(B, dtype=np.int64)
    for i in range(B):
        candidates = np.where(subj_labels != subj_labels[i])[0]
        if len(candidates) > 0:
            style_idx[i] = candidates[random.randrange(len(candidates))]
        else:
            # All samples from the same subject — skip cyclic loss for this batch
            # (caller checks `has_multi_subj` before using style_idx)
            style_idx[i] = (i + 1) % B
    return style_idx


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DSDRNetECAPATrainer(WindowClassifierTrainer):
    """
    Trainer for DSDRNetECAPA with multi-scale AdaIN cyclic reconstruction.

    Overrides fit() and evaluate_numpy() from WindowClassifierTrainer.
    All other helper methods (_prepare_splits_arrays, _compute_channel_standardization,
    _apply_standardization) are inherited from the parent class.
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # ── ECAPA architecture ────────────────────────────────────
        channels:       int = 128,
        scale:          int = 4,
        embedding_dim:  int = 128,
        dilations:      Optional[List[int]] = None,
        se_reduction:   int = 8,
        decoder_hidden: int = 256,
        # ── Loss weights ──────────────────────────────────────────
        lambda_cycle:      float = 1.0,
        lambda_self_recon: float = 0.5,
        lambda_perceptual: float = 0.1,
        # ── Warmup: fraction of epochs over which recon losses ramp ─
        cycle_warmup_frac: float = 0.2,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.channels          = channels
        self.scale             = scale
        self.embedding_dim     = embedding_dim
        self.dilations         = dilations if dilations is not None else [2, 3, 4]
        self.se_reduction      = se_reduction
        self.decoder_hidden    = decoder_hidden
        self.lambda_cycle      = lambda_cycle
        self.lambda_self_recon = lambda_self_recon
        self.lambda_perceptual = lambda_perceptual
        self.cycle_warmup_frac = cycle_warmup_frac

    # ─────────────────────────────────────────────────────────────────────────
    # fit()
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train DSDRNetECAPA.

        Args:
            splits: {
                "train":                Dict[int, np.ndarray],  # gesture_id → (N, T, C)
                "val":                  Dict[int, np.ndarray],
                "test":                 Dict[int, np.ndarray],
                "train_subject_labels": Dict[int, np.ndarray],  # gesture_id → (N,) subj idx
                "num_train_subjects":   int,
            }

        LOSO guarantee:
            mean_c / std_c are computed from X_train ONLY.
            Style pairs are sampled within each batch from training subjects.
            X_test is evaluated post-training via evaluate_numpy(), using the
            content pathway only — no AdaIN, no decoder, no test-subject statistics.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Prepare flat arrays from split dicts ───────────────────────
        (X_train, y_train, X_val, y_val, X_test, y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        # ── 2. Extract subject labels aligned with _prepare_splits_arrays ordering.
        #       _prepare_splits_arrays iterates sorted(splits["train"].keys()) = class_ids.
        #       We must do the same to keep subject labels aligned with y_train.
        subj_label_dict = splits.get("train_subject_labels", {})
        train_subj_parts: List[np.ndarray] = []
        for gid in class_ids:
            if gid in subj_label_dict:
                train_subj_parts.append(subj_label_dict[gid])
            elif gid in splits["train"]:
                # Missing subject labels for this gesture → assign sentinel 0
                n = len(splits["train"][gid])
                self.logger.warning(
                    f"No subject labels for gesture {gid}, assigning sentinel 0 "
                    f"({n} windows). Style pairing may be suboptimal for this gesture."
                )
                train_subj_parts.append(np.zeros(n, dtype=np.int64))
        y_train_subj = np.concatenate(train_subj_parts, axis=0) if train_subj_parts \
            else np.zeros(len(y_train), dtype=np.int64)
        num_train_subjects = splits.get("num_train_subjects", 1)

        # ── 3. Transpose (N, T, C) → (N, C, T) ───────────────────────────
        def _t(X: np.ndarray) -> np.ndarray:
            return X.transpose(0, 2, 1) if X.ndim == 3 and X.shape[1] > X.shape[2] else X

        X_train = _t(X_train)
        X_val   = _t(X_val)
        X_test  = _t(X_test)

        # ── 4. Channel standardisation — TRAINING DATA ONLY ───────────────
        #       mean_c and std_c are computed exclusively from training windows.
        #       Validation and test windows are normalised using training statistics,
        #       so no information from val/test leaks into the normalisation.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        X_val   = self._apply_standardization(X_val,   mean_c, std_c)
        X_test  = self._apply_standardization(X_test,  mean_c, std_c)

        np.savez(self.output_dir / "normalization_stats.npz", mean=mean_c, std=std_c)

        # ── 5. Store trainer state (required by evaluate_numpy) ───────────
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        in_channels  = X_train.shape[1]   # (N, C, T) → C
        window_size  = X_train.shape[2]
        self.in_channels = in_channels
        self.window_size = window_size

        self.logger.info(
            f"[DSDRNetECAPA] train={X_train.shape}, val={X_val.shape}, "
            f"test={X_test.shape}, classes={len(class_ids)}, "
            f"train_subjects={num_train_subjects}"
        )

        # ── 6. Build model ────────────────────────────────────────────────
        device = self.cfg.device
        model = DSDRNetECAPA(
            in_channels   = in_channels,
            num_classes   = len(class_ids),
            channels      = self.channels,
            scale         = self.scale,
            embedding_dim = self.embedding_dim,
            dilations     = self.dilations,
            dropout       = self.cfg.dropout,
            se_reduction  = self.se_reduction,
            decoder_hidden = self.decoder_hidden,
        ).to(device)
        self.logger.info(f"  Parameters: {model.count_parameters():,}")

        # ── 7. Datasets and data loaders ──────────────────────────────────
        train_ds = CyclicWindowDataset(X_train, y_train, y_train_subj)
        val_ds   = WindowDataset(X_val, y_val)
        test_ds  = WindowDataset(X_test, y_test)

        worker_init = get_worker_init_fn(self.cfg.seed)
        train_loader = DataLoader(
            train_ds,
            batch_size     = self.cfg.batch_size,
            shuffle        = True,
            num_workers    = self.cfg.num_workers,
            worker_init_fn = worker_init,
            drop_last      = False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size * 2,
            shuffle=False, num_workers=self.cfg.num_workers,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.cfg.batch_size * 2,
            shuffle=False, num_workers=self.cfg.num_workers,
        )

        # ── 8. Loss, optimiser, scheduler ─────────────────────────────────
        if self.cfg.use_class_weights:
            counts     = np.bincount(y_train, minlength=len(class_ids)).astype(np.float32)
            w          = 1.0 / (counts + 1e-6)
            w          = w / w.sum() * len(class_ids)
            cls_weight = torch.tensor(w, dtype=torch.float32, device=device)
        else:
            cls_weight = None

        criterion = nn.CrossEntropyLoss(weight=cls_weight)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 9. Training loop ──────────────────────────────────────────────
        total_epochs  = self.cfg.epochs
        warmup_epochs = max(1, int(total_epochs * self.cycle_warmup_frac))

        best_val_loss  = float("inf")
        best_state     = None
        patience_ctr   = 0

        history: Dict[str, List] = {
            "train_loss": [], "train_cls_loss": [],
            "train_cycle_loss": [], "train_self_loss": [],
            "train_perceptual_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        for epoch in range(1, total_epochs + 1):
            # Linear warmup factor for reconstruction losses.
            # Epoch 1..warmup_epochs: factor goes 1/W → 1.0
            # After warmup: factor = 1.0
            # This stabilises training while the encoder warms up.
            recon_w = min(1.0, epoch / warmup_epochs)

            # ── Train ──────────────────────────────────────────────────
            model.train()
            sum_cls = sum_cyc = sum_self = sum_perc = sum_total = 0.0
            n_correct = n_total = 0

            for windows, gesture_labels, subj_labels in train_loader:
                windows        = windows.to(device)          # (B, C, T)
                gesture_labels = gesture_labels.to(device)   # (B,)
                subj_np        = subj_labels.numpy()          # keep on CPU for np ops

                # ── Content pathway ──────────────────────────────────
                # encode() uses training windows; forward() path for classification.
                # LOSO: only training subjects in this batch.
                f_A   = model.encode(windows)               # list of 3 (B, C, T)
                emb_A = model.features_to_embedding(f_A)
                logits = model.classifier(emb_A)
                L_cls  = criterion(logits, gesture_labels)

                # ── Reconstruction losses (cross-subject pairs required) ──
                # Skip reconstruction losses when all samples share the same subject
                # (degenerate batch — style transfer would be a near-identity).
                has_multi_subj = len(np.unique(subj_np)) >= 2

                L_self_recon = torch.zeros(1, device=device)
                L_cycle      = torch.zeros(1, device=device)
                L_perceptual = torch.zeros(1, device=device)

                if has_multi_subj:
                    # ── Self-reconstruction: decode(f_A) ≈ windows ───────
                    # AdaIN(f_A, f_A) = f_A (algebraic identity), so
                    # self-recon is equivalent to a standard autoencoder step.
                    # LOSO: only training windows involved.
                    x_A_self     = model.decode(f_A)
                    L_self_recon = F.mse_loss(x_A_self, windows)

                    # ── Perceptual loss ───────────────────────────────────
                    # encode(decode(f_A)) features should resemble f_A features.
                    # f_A targets are detached to separate this gradient path
                    # from L_cls, preventing conflicting objective signals.
                    f_A_reenc    = model.encode(x_A_self)
                    L_perceptual = sum(
                        F.mse_loss(f_r, f_a.detach())
                        for f_r, f_a in zip(f_A_reenc, f_A)
                    ) / 3.0

                    # ── Cross-subject style pairing ───────────────────────
                    # For each sample i, choose j with a different subject.
                    # Both i and j are from training subjects — no test data.
                    style_idx_np = _sample_style_indices(subj_np)
                    style_idx_t  = torch.from_numpy(style_idx_np).long()
                    x_style      = windows[style_idx_t]          # (B, C, T)

                    # ── Cyclic consistency: A → B-style → A-style ≈ A ────
                    # Step 1: A with B's style → x_AB
                    f_B   = model.encode(x_style)                # encode style windows
                    f_AB  = model.apply_multi_scale_adain(f_A, f_B)
                    x_AB  = model.decode(f_AB)                   # (B, C_emg, T)

                    # Step 2: Re-encode x_AB, then apply A's original style
                    # Gradient flows through this full chain: x_ABA ← decode ←
                    # adain(encode(x_AB), f_A) ← decode(f_AB) ← adain(f_A, f_B)
                    # ← encode(windows).  Full end-to-end differentiation.
                    f_AB_reenc = model.encode(x_AB)
                    f_ABA      = model.apply_multi_scale_adain(f_AB_reenc, f_A)
                    x_ABA      = model.decode(f_ABA)

                    L_cycle = F.mse_loss(x_ABA, windows)

                # ── Total loss ────────────────────────────────────────
                L = (
                    L_cls
                    + recon_w * self.lambda_cycle      * L_cycle
                    + recon_w * self.lambda_self_recon * L_self_recon
                    + recon_w * self.lambda_perceptual * L_perceptual
                )

                optimizer.zero_grad()
                L.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                B            = windows.size(0)
                sum_cls     += L_cls.item()          * B
                sum_cyc     += L_cycle.item()        * B
                sum_self    += L_self_recon.item()   * B
                sum_perc    += L_perceptual.item()   * B
                sum_total   += L.item()              * B
                preds        = logits.argmax(dim=1)
                n_correct   += (preds == gesture_labels).sum().item()
                n_total     += B

            train_acc = n_correct / max(n_total, 1)
            avg = {k: v / max(n_total, 1) for k, v in {
                "total": sum_total, "cls": sum_cls,
                "cyc": sum_cyc, "self": sum_self, "perc": sum_perc,
            }.items()}

            # ── Validate ───────────────────────────────────────────────
            # Validation uses content pathway only (model.forward = content pathway).
            # No reconstruction losses on val — consistent with inference behaviour.
            model.eval()
            val_loss = val_n = val_correct = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    out       = model(X_b)              # content pathway
                    val_loss  += criterion(out, y_b).item() * X_b.size(0)
                    val_correct += (out.argmax(1) == y_b).sum().item()
                    val_n       += X_b.size(0)
            val_loss /= max(val_n, 1)
            val_acc   = val_correct / max(val_n, 1)

            scheduler.step(val_loss)

            history["train_loss"].append(avg["total"])
            history["train_cls_loss"].append(avg["cls"])
            history["train_cycle_loss"].append(avg["cyc"])
            history["train_self_loss"].append(avg["self"])
            history["train_perceptual_loss"].append(avg["perc"])
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if epoch % 5 == 0 or epoch == 1:
                self.logger.info(
                    f"  Ep {epoch:3d}/{total_epochs}"
                    f" | L={avg['total']:.4f}"
                    f" cls={avg['cls']:.4f}"
                    f" cyc={avg['cyc']:.4f}"
                    f" self={avg['self']:.4f}"
                    f" perc={avg['perc']:.4f}"
                    f" | tr_acc={train_acc:.4f}"
                    f" | val_loss={val_loss:.4f}"
                    f" | val_acc={val_acc:.4f}"
                    f" | w={recon_w:.2f}"
                )

            # ── Early stopping ────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.early_stopping_patience:
                    self.logger.info(f"  Early stopping at epoch {epoch}.")
                    break

        # ── 10. Restore best checkpoint ───────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model

        # ── 11. Save training artefacts ───────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=2)

        if self.visualizer is not None:
            try:
                self.visualizer.plot_training_curves(
                    {"train_loss": history["train_loss"],
                     "val_loss":   history["val_loss"],
                     "train_acc":  history["train_acc"],
                     "val_acc":    history["val_acc"]},
                    "training_curves.png",
                )
            except Exception as exc:
                self.logger.warning(f"Could not plot training curves: {exc}")

        torch.save(
            {
                "state_dict":  model.state_dict(),
                "in_channels": in_channels,
                "num_classes": len(class_ids),
                "class_ids":   class_ids,
                "mean":        mean_c,
                "std":         std_c,
                "window_size": window_size,
                "model_config": {
                    "channels":       self.channels,
                    "scale":          self.scale,
                    "embedding_dim":  self.embedding_dim,
                    "dilations":      self.dilations,
                    "se_reduction":   self.se_reduction,
                    "decoder_hidden": self.decoder_hidden,
                },
                "loss_config": {
                    "lambda_cycle":      self.lambda_cycle,
                    "lambda_self_recon": self.lambda_self_recon,
                    "lambda_perceptual": self.lambda_perceptual,
                    "cycle_warmup_frac": self.cycle_warmup_frac,
                },
            },
            self.output_dir / "dsdrnet_ecapa.pt",
        )

        # ── 12. Final evaluation (content pathway only) ───────────────────
        def _eval_loader(loader: DataLoader, name: str) -> Dict:
            model.eval()
            all_preds, all_labels, all_logits = [], [], []
            with torch.no_grad():
                for X_b, y_b in loader:
                    out = model(X_b.to(device))   # content pathway → logits
                    all_logits.append(out.cpu().numpy())
                    all_preds.append(out.argmax(1).cpu().numpy())
                    all_labels.append(y_b.numpy())
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_labels)
            logits = np.concatenate(all_logits)
            acc    = accuracy_score(y_true, y_pred)
            f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
            cm     = confusion_matrix(y_true, y_pred).tolist()
            rep    = classification_report(
                y_true, y_pred,
                target_names=[class_names[c] for c in class_ids],
                output_dict=True, zero_division=0,
            )
            self.logger.info(f"  [{name}] acc={acc:.4f}  f1_macro={f1:.4f}")
            return {"accuracy": acc, "f1_macro": f1,
                    "confusion_matrix": cm, "report": rep, "logits": logits}

        val_results  = _eval_loader(val_loader,  "val")
        test_results = _eval_loader(test_loader, "test")

        results = {
            "class_ids":   class_ids,
            "class_names": {str(k): v for k, v in class_names.items()},
            "val":         val_results,
            "test":        test_results,
        }
        with open(self.output_dir / "classification_results.json", "w") as fh:
            json.dump(
                results, fh, indent=2,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
            )
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # evaluate_numpy() — content pathway only
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate on (N, T, C) windows using the content pathway exclusively.

        LOSO guarantee:
            The content pathway (encode → ASP → embed → classify) does NOT
            invoke AdaIN or the decoder and does NOT access any subject-specific
            statistics beyond what was fixed during training.  Test-subject
            windows are normalised with training mean_c / std_c (computed in
            fit() from training data only).

        Args:
            X:          (N, T, C) raw windows — may be test-subject data
            y:          (N,)      class indices matching class_ids ordering
            split_name: string tag for logging and plot filenames
            visualize:  if True and visualizer is set, saves confusion matrix
        Returns:
            dict with keys: accuracy, f1_macro, report, confusion_matrix, logits
        """
        assert self.model       is not None, "Call fit() before evaluate_numpy()"
        assert self.mean_c      is not None, "mean_c not set — fit() must run first"
        assert self.std_c       is not None, "std_c not set — fit() must run first"
        assert self.class_ids   is not None
        assert self.class_names is not None

        # Transpose if needed — same logic as fit()
        if X.ndim == 3 and X.shape[1] > X.shape[2]:
            X = X.transpose(0, 2, 1)
        # Normalise with training statistics
        X = self._apply_standardization(X, self.mean_c, self.std_c)

        device  = self.cfg.device
        dataset = WindowDataset(X, y)
        loader  = DataLoader(
            dataset, batch_size=self.cfg.batch_size * 2,
            shuffle=False, num_workers=self.cfg.num_workers,
        )

        self.model.eval()
        all_preds, all_labels, all_logits = [], [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                out = self.model(X_b.to(device))   # content pathway only
                all_logits.append(out.cpu().numpy())
                all_preds.append(out.argmax(1).cpu().numpy())
                all_labels.append(y_b.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        logits = np.concatenate(all_logits)

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm  = confusion_matrix(y_true, y_pred).tolist()
        rep = classification_report(
            y_true, y_pred,
            target_names=[self.class_names[c] for c in self.class_ids],
            output_dict=True, zero_division=0,
        )

        self.logger.info(
            f"  evaluate_numpy [{split_name}] acc={acc:.4f}  f1_macro={f1:.4f}"
        )

        if visualize and self.visualizer is not None:
            try:
                self.visualizer.plot_confusion_matrix(
                    np.array(cm),
                    class_labels=[self.class_names[c] for c in self.class_ids],
                    normalize=True,
                    filename=f"confusion_matrix_{split_name}.png",
                )
                self.visualizer.plot_per_class_f1(
                    rep,
                    class_labels=[self.class_names[c] for c in self.class_ids],
                    filename=f"per_class_f1_{split_name}.png",
                )
            except Exception as exc:
                self.logger.warning(f"Visualisation failed: {exc}")

        return {
            "accuracy":         acc,
            "f1_macro":         f1,
            "confusion_matrix": cm,
            "report":           rep,
            "logits":           logits,
        }
