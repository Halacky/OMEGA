"""
Trainer for HyGDL: Analytical Orthogonal Content/Style Projection (Experiment 90).

Training protocol (two phases, strictly LOSO-compatible):

  Phase 1 — Warmup  (first `warmup_epochs` epochs):
    • Train encoder + classifier with classification loss only.
    • projection_valid = False → classifier receives the full embedding z.
    • Allows the encoder to converge to meaningful gesture representations
      before the orthogonal structure is imposed.

  Phase 2 — Disentangled training  (remaining epochs):
    • At the start of Phase 2 and then every `subspace_update_interval` epochs:
        1. Forward-pass ALL training-subject windows through the encoder
           in no_grad + model.eval() mode  (frozen BN running stats).
        2. Compute mean embedding μ_s per training subject.
        3. Stack {μ_s}, centre, SVD → V_style ∈ R^{E×k}
           (top-k right singular vectors of inter-subject variance).
        4. Call model.update_style_subspace(V_style).
           V_style is a buffer — no backprop path exists from this update.
    • Loss = L_cls (CE on z_content) + λ_rec * L_rec (MSE on decoder output)

LOSO data-leakage checklist
────────────────────────────
  ✓ Channel mean/std: computed from X_train only; applied unchanged to val/test.
  ✓ V_style: built from encoder(X_s) for each training subject s in no_grad mode.
    Neither val nor test windows ever enter the SVD computation.
  ✓ BN during V_style pass: model.eval() → BN uses frozen running stats from
    training batches.  No BN updates from the V_style forward pass.
  ✓ Test evaluation: evaluate_numpy() only → model.eval() + frozen V_style.
    No BN updates, no V_style updates at test time.
  ✓ Val data: used only as early-stopping criterion on classification loss.
    Never enters normalisation or V_style computation.
  ✓ per_subject_windows: populated in the experiment file exclusively from
    train_subjects (excluding test_subject by construction).
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from config.base import TrainingConfig
from models.hygdl_ecapa_loso import HyGDLModel
from training.trainer import WindowClassifierTrainer
from training.datasets import WindowDataset
from utils.logging import get_worker_init_fn, seed_everything


class HyGDLTrainer(WindowClassifierTrainer):
    """
    Trainer for HyGDL analytical orthogonal content/style projection.

    Inherits from WindowClassifierTrainer for shared utilities:
        _prepare_splits_arrays(), _compute_channel_standardization(),
        _apply_standardization().

    Additional constructor parameters (beyond WindowClassifierTrainer):
        warmup_epochs:             Number of Phase-1 epochs before the first
                                   V_style update (default 15).
        subspace_update_interval:  Epochs between V_style recomputations during
                                   Phase 2 (default 5).
        style_dim:                 k — rank of the style projection subspace
                                   (default 4).  Effective rank is clamped to
                                   min(k, n_train_subjects − 1).
        t_compressed:              Time steps in the reconstruction target
                                   (default 75 = 600 // 8 for standard windows).
        lambda_rec:                Weight for reconstruction regularisation loss
                                   (default 0.1).
        channels, scale, embedding_dim, dilations, se_reduction:
                                   ECAPA-TDNN encoder hyper-parameters.
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # HyGDL-specific
        warmup_epochs: int = 15,
        subspace_update_interval: int = 5,
        style_dim: int = 4,
        t_compressed: int = 75,
        lambda_rec: float = 0.1,
        # ECAPA encoder hyper-parameters
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        dilations: Optional[List[int]] = None,
        se_reduction: int = 8,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.warmup_epochs = warmup_epochs
        self.subspace_update_interval = subspace_update_interval
        self.style_dim = style_dim
        self.t_compressed = t_compressed
        self.lambda_rec = lambda_rec
        self.channels = channels
        self.scale = scale
        self.embedding_dim = embedding_dim
        self.dilations = dilations if dilations is not None else [2, 3, 4]
        self.se_reduction = se_reduction

    # ──────────────────────────────── fit ────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train HyGDLModel with warmup + analytical-projection phases.

        Args:
            splits: dict with the following keys:
                "train": Dict[int, np.ndarray]   gesture_id → (N, T, C) windows
                "val":   Dict[int, np.ndarray]   gesture_id → (N, T, C) windows
                "test":  Dict[int, np.ndarray]   gesture_id → (N, T, C) windows
                                                 (internal test split, NOT the
                                                  held-out LOSO test subject)
                "per_subject_windows":            (LOSO-critical extra key)
                    Dict[int, np.ndarray]  subject_index → (N_s, T, C)
                    Contains training-subject windows for V_style computation.
                    Must be populated exclusively from train_subjects.

        Returns:
            results dict with val and internal-test metrics.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Extract per-subject windows for V_style computation ────────
        # LOSO contract: populated by the experiment file from train_subjects only.
        per_subject_windows_raw: Dict[int, np.ndarray] = splits.get(
            "per_subject_windows", {}
        )
        if not per_subject_windows_raw:
            self.logger.warning(
                "No 'per_subject_windows' key in splits — V_style update will "
                "be skipped. Ensure the experiment file populates this key "
                "from training subjects only."
            )

        # ── 2. Splits → flat (N, T, C) arrays ────────────────────────────
        # _prepare_splits_arrays processes only the standard train/val/test keys.
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        # ── 3. Transpose (N, T, C) → (N, C, T) ───────────────────────────
        # HyGDLEncoderECAPA expects channels-first Conv1d format (B, C, T).
        # Heuristic: for EMG T (time steps, ~600) >> C (channels, ~8).
        def _maybe_transpose(X: np.ndarray) -> np.ndarray:
            if X.ndim == 3 and X.shape[1] > X.shape[2]:
                return X.transpose(0, 2, 1)
            return X

        X_train = _maybe_transpose(X_train)
        X_val   = _maybe_transpose(X_val)   if len(X_val)  > 0 else X_val
        X_test  = _maybe_transpose(X_test)  if len(X_test) > 0 else X_test

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        self.logger.info(
            f"After transpose — X_train={X_train.shape}, "
            f"in_channels={in_channels}, window_size={window_size}, "
            f"num_classes={num_classes}"
        )

        # ── 4. Per-channel standardisation (training data ONLY) ───────────
        # LOSO integrity: mean_c and std_c are computed exclusively from X_train.
        # The same statistics are then applied to val and test data without
        # re-computation — this is the only permissible normalisation flow.
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
        self.logger.info("Per-channel standardisation applied (training stats only).")

        # Apply the SAME normalisation to per-subject windows.
        # These will be used for V_style computation in _update_style_subspace.
        # LOSO guard: per_subject_windows_raw contains ONLY training-subject
        # windows (guaranteed by the experiment's _build_splits function).
        per_subject_windows_norm: Dict[int, np.ndarray] = {}
        for sidx, X_s_raw in per_subject_windows_raw.items():
            X_s = _maybe_transpose(X_s_raw.astype(np.float32))
            X_s = self._apply_standardization(X_s, mean_c, std_c)
            per_subject_windows_norm[sidx] = X_s

        # ── 5. Build model ────────────────────────────────────────────────
        model = HyGDLModel(
            in_channels=in_channels,
            num_classes=num_classes,
            embedding_dim=self.embedding_dim,
            style_dim=self.style_dim,
            t_compressed=self.t_compressed,
            channels=self.channels,
            scale=self.scale,
            dilations=self.dilations,
            dropout=self.cfg.dropout,
            se_reduction=self.se_reduction,
        ).to(self.cfg.device)

        total_params = model.count_parameters()
        self.logger.info(
            f"HyGDLModel: in_ch={in_channels}, classes={num_classes}, "
            f"E={self.embedding_dim}, k={self.style_dim}, "
            f"t_compressed={self.t_compressed}, "
            f"C={self.channels}, scale={self.scale}, "
            f"dilations={self.dilations} | total_params={total_params:,}"
        )

        # ── 6. Datasets & DataLoaders ─────────────────────────────────────
        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val,   y_val)   if len(X_val)  > 0 else None
        ds_test  = WindowDataset(X_test,  y_test)  if len(X_test) > 0 else None

        worker_init = get_worker_init_fn(self.cfg.seed)
        g = torch.Generator().manual_seed(self.cfg.seed)

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
            generator=g,
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

        # ── 7. Loss functions & optimiser ─────────────────────────────────
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = counts.sum() / (counts + 1e-8)
            cw /= cw.mean()
            criterion_cls = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(self.cfg.device)
            )
            self.logger.info(f"Class weights: {cw.round(3).tolist()}")
        else:
            criterion_cls = nn.CrossEntropyLoss()

        criterion_rec = nn.MSELoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # ReduceLROnPlateau: `verbose` kwarg removed in PyTorch ≥ 2.4
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 8. Training loop ──────────────────────────────────────────────
        history: Dict = {
            "train_loss":      [],
            "val_loss":        [],
            "train_acc":       [],
            "val_acc":         [],
            "train_cls_loss":  [],
            "train_rec_loss":  [],
            "v_style_updates": [],   # epoch numbers when V_style was recomputed
        }
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        no_improve = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            in_warmup      = epoch <= self.warmup_epochs
            in_disentangle = not in_warmup

            # ── V_style update (Phase 2, at start and every N epochs) ─────
            # Triggers at first Phase-2 epoch and every subspace_update_interval
            # epochs thereafter.
            #
            # LOSO guard: _update_style_subspace uses only per_subject_windows_norm,
            # which contains training-subject data exclusively (see step 4 above).
            # The function calls model.eval() internally and restores model.train()
            # before returning, so the subsequent training step is unaffected.
            if in_disentangle and per_subject_windows_norm:
                epochs_since_warmup = epoch - self.warmup_epochs
                is_first_phase2 = epochs_since_warmup == 1
                is_update_step  = (epochs_since_warmup - 1) % self.subspace_update_interval == 0
                if is_first_phase2 or is_update_step:
                    self._update_style_subspace(
                        model, per_subject_windows_norm, device
                    )
                    history["v_style_updates"].append(epoch)
                    self.logger.info(
                        f"[Epoch {epoch}] V_style updated "
                        f"(k={self.style_dim}, "
                        f"{len(per_subject_windows_norm)} training subjects)"
                    )

            # ── Train epoch ───────────────────────────────────────────────
            model.train()
            ep_cls_loss = 0.0
            ep_rec_loss = 0.0
            ep_correct  = 0
            ep_total    = 0

            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()

                if in_warmup:
                    # Phase 1: classification loss only; no reconstruction term.
                    # projection_valid=False → model.forward() returns logits
                    # from the full embedding z (no projection overhead).
                    logits = model(xb)
                    loss_cls = criterion_cls(logits, yb)
                    loss     = loss_cls
                    # ep_rec_loss stays 0.0 for warmup epochs
                else:
                    # Phase 2: classification on z_content + reconstruction.
                    # Reconstruction is on the decoder output from full z,
                    # so the encoder must retain all signal information in z
                    # (not just the content component).
                    logits, x_hat, x_target, z_content, z_style = (
                        model.forward_with_reconstruction(xb)
                    )
                    loss_cls = criterion_cls(logits, yb)
                    loss_rec = criterion_rec(x_hat, x_target)
                    loss     = loss_cls + self.lambda_rec * loss_rec
                    ep_rec_loss += loss_rec.item() * xb.size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = xb.size(0)
                ep_cls_loss += loss_cls.item() * bs
                ep_correct  += (logits.argmax(1) == yb).sum().item()
                ep_total    += bs

            train_cls  = ep_cls_loss / max(1, ep_total)
            train_rec  = ep_rec_loss / max(1, ep_total)
            train_loss = train_cls + self.lambda_rec * train_rec
            train_acc  = ep_correct / max(1, ep_total)

            # ── Validation pass ───────────────────────────────────────────
            # Validation uses model.forward() — projection is active if
            # projection_valid=True (Phase 2), otherwise full z (Phase 1).
            # Val loss is classification CE only (no reconstruction term),
            # giving a clean signal for early stopping.
            if dl_val is not None:
                model.eval()
                vl_sum, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits  = model(xb)
                        vl_sum += criterion_cls(logits, yb).item() * yb.size(0)
                        vc     += (logits.argmax(1) == yb).sum().item()
                        vt     += yb.size(0)
                val_loss = vl_sum / max(1, vt)
                val_acc  = vc    / max(1, vt)
            else:
                val_loss = val_acc = float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_cls_loss"].append(train_cls)
            history["train_rec_loss"].append(train_rec)

            phase_tag = "warmup" if in_warmup else "disentangle"
            v_valid = model.projection_valid.item()
            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}|{phase_tag}|proj={v_valid}] "
                f"cls={train_cls:.4f} rec={train_rec:.4f} "
                f"train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

            # ── Early stopping ────────────────────────────────────────────
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}.")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── 9. Store trainer state (needed by evaluate_numpy) ─────────────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 10. Save training history ──────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png"
            )

        # ── 11. In-fold evaluation (val / internal test) ───────────────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dl, split_name: str):
            if dl is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(device)
                    all_logits.append(model(xb).cpu().numpy())
                    all_y.append(yb.numpy())
            y_true = np.concatenate(all_y,    axis=0)
            y_pred = np.concatenate(all_logits, axis=0).argmax(axis=1)
            acc = float(accuracy_score(y_true, y_pred))
            f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm  = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True, filename=f"cm_{split_name}.png"
                )
            return {
                "accuracy": acc, "f1_macro": f1,
                "report": rep, "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "internal_test")

        # ── 12. Save checkpoint ────────────────────────────────────────────
        V_style_np = model.V_style.cpu().numpy()
        torch.save(
            {
                "state_dict":       model.state_dict(),
                "in_channels":      in_channels,
                "num_classes":      num_classes,
                "class_ids":        class_ids,
                "mean":             mean_c,
                "std":              std_c,
                "window_size":      window_size,
                "v_style":          V_style_np,
                "projection_valid": model.projection_valid.item(),
                "model_config": {
                    "embedding_dim":            self.embedding_dim,
                    "style_dim":                self.style_dim,
                    "t_compressed":             self.t_compressed,
                    "channels":                 self.channels,
                    "scale":                    self.scale,
                    "dilations":                self.dilations,
                    "se_reduction":             self.se_reduction,
                    "dropout":                  self.cfg.dropout,
                    "warmup_epochs":            self.warmup_epochs,
                    "subspace_update_interval": self.subspace_update_interval,
                    "lambda_rec":               self.lambda_rec,
                },
                "training_config": asdict(self.cfg),
            },
            self.output_dir / "hygdl_model.pt",
        )
        self.logger.info(
            f"Checkpoint saved → {self.output_dir / 'hygdl_model.pt'}"
        )

        with open(self.output_dir / "classification_results.json", "w") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False)

        return results

    # ─────────────────────────── V_style update ──────────────────────────────

    def _update_style_subspace(
        self,
        model: HyGDLModel,
        per_subject_windows: Dict[int, np.ndarray],
        device: str,
    ) -> None:
        """
        Compute training-subject mean embeddings, run SVD, update V_style buffer.

        LOSO contract:
          • per_subject_windows contains ONLY training-subject windows.
            The experiment file is responsible for this invariant.
          • model.eval() mode: BN uses frozen running statistics accumulated
            during training.  No BN parameters are updated by this forward pass.
          • torch.no_grad(): no gradient computation; no effect on the
            computation graph of subsequent training steps.
          • model.update_style_subspace() copies a float32 tensor into a
            registered buffer (not a Parameter) — no backpropagation path.

        SVD details:
          M ∈ R^{n_subjects × E}  — stacked mean embeddings
          M_c = M − mean(M, axis=0)   — centre inter-subject variation
          U, S, Vt = SVD(M_c)
          V_style = Vt[:k].T ∈ R^{E × k}  — top-k right singular vectors

        If the number of subjects is less than style_dim, the effective rank k
        is clamped to (n_subjects − 1) and the remaining columns are zero-padded.
        Zero columns contribute nothing to the projection, so the model degrades
        gracefully when few training subjects are available.

        Args:
            model:                HyGDLModel instance (on device).
            per_subject_windows:  {subject_idx: (N_s, C, T) float32 ndarray}
                                  Already transposed to channels-first and
                                  normalised with training-data mean/std.
            device:               Training device string (e.g. "cuda" or "cpu").
        """
        model.eval()

        subject_mean_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for sidx in sorted(per_subject_windows.keys()):
                X_s = per_subject_windows[sidx]   # (N_s, C, T) float32
                if len(X_s) == 0:
                    continue

                # Forward in mini-batches to avoid GPU OOM on large subjects.
                embeddings: List[np.ndarray] = []
                for start in range(0, len(X_s), self.cfg.batch_size):
                    xb = (
                        torch.from_numpy(X_s[start: start + self.cfg.batch_size])
                        .float()
                        .to(device)
                    )
                    z = model.encoder(xb)          # (B, E)
                    embeddings.append(z.cpu().numpy())

                if not embeddings:
                    continue

                Z_s  = np.concatenate(embeddings, axis=0)  # (N_s, E)
                mu_s = Z_s.mean(axis=0)                    # (E,)
                subject_mean_embeddings.append(mu_s)

        if len(subject_mean_embeddings) < 2:
            self.logger.warning(
                f"Only {len(subject_mean_embeddings)} training subject(s) "
                "available for V_style SVD (need ≥ 2) — skipping update."
            )
            model.train()
            return

        # M: (n_subjects, E)  — stack mean embeddings
        M   = np.stack(subject_mean_embeddings, axis=0).astype(np.float64)
        # Centre: remove the global mean so SVD captures inter-subject variance
        M_c = M - M.mean(axis=0, keepdims=True)

        # Clamp style_dim to the maximum achievable rank
        n_subj = M_c.shape[0]
        k = min(self.style_dim, n_subj - 1)   # rank ≤ n_subjects − 1
        if k < 1:
            self.logger.warning(
                "Effective style_dim < 1 after clamping — skipping V_style update."
            )
            model.train()
            return

        try:
            # full_matrices=False: economy-size SVD — Vt is (min(n,E), E)
            _, _, Vt = np.linalg.svd(M_c, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            self.logger.error(f"SVD failed: {exc} — skipping V_style update.")
            model.train()
            return

        # V_style: (E, k) — columns are top-k right singular vectors (orthonormal)
        V_style_np = Vt[:k].T.astype(np.float32)   # (E, k)

        # If effective rank < style_dim, zero-pad remaining columns.
        # Zero columns produce no projection contribution (z @ 0 = 0).
        if k < self.style_dim:
            E = V_style_np.shape[0]
            pad = np.zeros((E, self.style_dim - k), dtype=np.float32)
            V_style_np = np.concatenate([V_style_np, pad], axis=1)  # (E, style_dim)

        # Update buffer — does NOT affect the computation graph.
        model.update_style_subspace(torch.from_numpy(V_style_np))

        # Restore training mode for the subsequent training step.
        model.train()

    # ──────────────────────────── evaluate_numpy ──────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate the trained HyGDL model on arbitrary (X, y) numpy arrays.

        Applies EXACTLY the same preprocessing pipeline as fit():
          1. Transpose (N, T, C) → (N, C, T) if needed.
          2. Channel standardisation using TRAINING statistics (mean_c, std_c).
          3. Analytical projection using V_style fixed from training subjects.

        No test-subject statistics are used at any point.  The model, normalisation
        parameters, and V_style are fully determined by training data.

        LOSO integrity: model.eval() is set before inference.  BatchNorm uses
        running statistics accumulated during model.train() (training batches).
        No BN updates occur from test-subject data.

        Args:
            X:          Raw EMG windows, (N, T, C) or (N, C, T).
            y:          Integer class labels aligned with class_ids from fit().
            split_name: Prefix for saved confusion-matrix image.
            visualize:  Whether to generate and save a confusion-matrix plot.

        Returns:
            dict with keys "accuracy", "f1_macro", "report",
            "confusion_matrix", "logits".
        """
        assert self.model       is not None, "Call fit() before evaluate_numpy()."
        assert self.mean_c      is not None and self.std_c is not None
        assert self.class_ids   is not None
        assert self.class_names is not None

        X_in = X.copy().astype(np.float32)

        # Transpose to (N, C, T) — same heuristic as fit()
        if X_in.ndim == 3 and X_in.shape[1] > X_in.shape[2]:
            X_in = X_in.transpose(0, 2, 1)

        # Apply training-data standardisation — no test-subject stats used
        X_in = self._apply_standardization(X_in, self.mean_c, self.std_c)

        ds = WindowDataset(X_in, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        # model.eval(): BN uses frozen training running stats; no updates.
        # Analytical projection: V_style fixed from training SVD; no adaptation.
        self.model.eval()
        all_logits: List[np.ndarray] = []
        all_y:      List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                all_logits.append(self.model(xb).cpu().numpy())
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y,      axis=0)
        y_pred = logits.argmax(axis=1)

        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        num_classes = len(self.class_ids)
        cm  = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        if visualize and self.visualizer is not None:
            labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, labels, normalize=True, filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy":         acc,
            "f1_macro":         f1,
            "report":           rep,
            "confusion_matrix": cm.tolist(),
            "logits":           logits,
        }
