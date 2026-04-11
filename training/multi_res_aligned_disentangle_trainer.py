"""
Trainer for Multi-Resolution Aligned Disentanglement (Hypothesis 3, Exp 110).

Two-stage training scheme
--------------------------
Stage 1  (epochs 1 .. stage1_epochs):
    loss = CE(logits, y_gesture)
          + alpha_align * AlignLoss(projections)

    AlignLoss = mean NT-Xent over 3 band-pairs: (low,mid), (low,high), (mid,high).
    Positive pair: same window, different bands.  Negative: different windows.
    This creates shared gesture representations invariant to frequency content.

Stage 2  (stage1_epochs+1 .. total_epochs):
    loss = CE(logits, y_gesture)
          + alpha_align * AlignLoss(projections)
          + beta_adv   * AdvLoss(adv_logits, y_gesture)

    AdvLoss: CE on per-band adversarial classifier logits (produced via GRL).
    GRL reverses gradients → SpecificEncoder maximises adversarial CE →
    specific features become uninformative about gesture class.

    At the start of Stage 2 the best_val_loss and no_improve counter are reset
    so early stopping gives Stage 2 a fair window to improve.

LOSO data-leakage audit
------------------------
✓ _prepare_splits_arrays(): test-subject windows → splits["test"] ONLY.
✓ mean_c, std_c: computed from X_train, never from val or test.
✓ All DataLoaders: train/val use training subjects; test subject never in a batch.
✓ NT-Xent: positive/negative pairs chosen within the current training batch;
  test subject not present → zero leakage into contrastive loss.
✓ Adversarial loss: uses gesture labels from the training batch; no subject IDs,
  no test-subject information.
✓ Validation loss drives early stopping (no test information).
✓ evaluate_numpy(): model.eval() → BN frozen; no test-time adaptation.
✓ stage=1 forward at inference: specific encoders never activated on test data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from training.disentangled_trainer import DisentangledTrainer
from models.multi_res_aligned_disentangle import MultiResAlignedDisentangle


class MultiResAlignedDisentangleTrainer(DisentangledTrainer):
    """
    Two-stage LOSO trainer for MultiResAlignedDisentangle.

    Inherits from DisentangledTrainer to reuse:
      _prepare_splits_arrays()          — flat array extraction
      _compute_channel_standardization() — training-stats-only normalization
      _apply_standardization()           — apply frozen stats to any split

    Overrides:
      fit()           — two-stage training loop
      evaluate_numpy() — dict-returning forward (model returns a dict, not tensor)

    Note: subject labels are NOT required in splits for this experiment.
    The adversarial loss operates on gesture labels only, which are always
    available from the standard split structure.
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # ── Band-split geometry ──────────────────────────────────────────
        sampling_rate: int = 2000,
        f_low: float = 200.0,
        f_mid: float = 500.0,
        # ── Aligned encoder ──────────────────────────────────────────────
        channels: int = 64,
        embed_dim: int = 64,
        proj_dim: int = 32,
        # ── Specific encoder ─────────────────────────────────────────────
        spec_hidden: int = 32,
        spec_embed_dim: int = 32,
        # ── Two-stage schedule ───────────────────────────────────────────
        stage1_epochs: int = 40,
        # ── Loss weights ─────────────────────────────────────────────────
        alpha_align: float = 0.5,
        beta_adv: float = 0.3,
        temperature: float = 0.1,
        grl_alpha: float = 1.0,
    ):
        # Pass minimal dummy args to DisentangledTrainer (fit/evaluate overridden).
        super().__init__(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=visualizer,
            content_dim=embed_dim,   # unused in this subclass
            style_dim=32,            # unused in this subclass
        )
        self.sampling_rate  = sampling_rate
        self.f_low          = f_low
        self.f_mid          = f_mid
        self.channels       = channels
        self.embed_dim      = embed_dim
        self.proj_dim       = proj_dim
        self.spec_hidden    = spec_hidden
        self.spec_embed_dim = spec_embed_dim
        self.stage1_epochs  = stage1_epochs
        self.alpha_align    = alpha_align
        self.beta_adv       = beta_adv
        self.temperature    = temperature
        self.grl_alpha      = grl_alpha

    # ── Contrastive alignment loss ────────────────────────────────────────

    @staticmethod
    def _nt_xent_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        NT-Xent (InfoNCE) loss for one pair of views.

        Args:
            z1, z2: (B, D) — L2-normalized embeddings, one per band.
                    (z1[i], z2[i]) is a positive pair (same window, diff band).
        Returns:
            scalar NT-Xent loss

        LOSO: z1, z2 contain only training-subject windows (never test subject).
        """
        B  = z1.shape[0]
        z  = torch.cat([z1, z2], dim=0)          # (2B, D)
        sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

        # Mask self-similarity (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim  = sim.masked_fill(mask, float("-inf"))

        # For z1[i] at row i,     positive is z2[i] at column (B + i)
        # For z2[i] at row (B+i), positive is z1[i] at column i
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(B,        device=z.device),
        ])
        return nn.functional.cross_entropy(sim, labels)

    def _alignment_loss(self, projections: List[torch.Tensor]) -> torch.Tensor:
        """
        Multi-band alignment: average NT-Xent over all 3 band pairs.

        Pairs: (low, mid), (low, high), (mid, high).
        Each pair shares the same positive/negative interpretation:
          positive = same window seen through two different frequency bands.
        """
        p_low, p_mid, p_high = projections
        t = self.temperature
        loss = (
            self._nt_xent_loss(p_low, p_mid,  t)
            + self._nt_xent_loss(p_low, p_high, t)
            + self._nt_xent_loss(p_mid, p_high, t)
        ) / 3.0
        return loss

    @staticmethod
    def _adversarial_loss(
        adv_logits: List[torch.Tensor],
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Adversarial disentanglement loss on specific features.

        The GRL has already been applied to specific features inside the model's
        forward pass (stage=2).  We compute standard CE on each band's adversarial
        logits versus the gesture labels — the GRL ensures gradients reaching the
        SpecificEncoder are reversed, pushing it away from gesture prediction.

        For the AdversarialClassifier itself: normal (non-reversed) gradients →
        it improves at predicting gestures, increasing the gradient signal that
        the GRL must reverse.
        """
        band_losses = [criterion(al, y) for al in adv_logits]
        return sum(band_losses) / len(band_losses)

    # ── Main training method ──────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Two-stage LOSO training for MultiResAlignedDisentangle.

        Expects splits with keys "train", "val", "test":
            each:  Dict[gesture_id → np.ndarray (N, T, C)]

        Returns dict with "best_val_loss", "history", "val_metrics".
        """
        seed_everything(self.cfg.seed)

        # ── 1. Prepare arrays ─────────────────────────────────────────────
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        self.class_ids   = class_ids
        self.class_names = class_names

        # ── 2. Transpose (N, T, C) → (N, C, T) for Conv1d ───────────────
        def _maybe_transpose(X: np.ndarray) -> np.ndarray:
            if X.ndim == 3 and X.shape[1] > X.shape[2]:
                return np.transpose(X, (0, 2, 1))
            return X

        X_train = _maybe_transpose(X_train)
        X_val   = _maybe_transpose(X_val)
        X_test  = _maybe_transpose(X_test)
        self.logger.info(
            f"After transpose: X_train={X_train.shape}  "
            f"X_val={X_val.shape}  X_test={X_test.shape}"
        )

        in_channels = X_train.shape[1]
        num_classes = len(class_ids)

        # ── 3. Channel standardisation — from X_train ONLY ───────────────
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val)  > 0:
            X_val  = self._apply_standardization(X_val,  mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)

        # Store for evaluate_numpy()
        self.mean_c = mean_c
        self.std_c  = std_c

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )
        self.logger.info("Saved normalisation stats.")

        # ── 4. Build model ────────────────────────────────────────────────
        # Use the actual window size after transpose (shape[2]) so BandSplitter
        # registers the correct FFT masks.
        actual_window_size = X_train.shape[2]

        model = MultiResAlignedDisentangle(
            in_channels=in_channels,
            num_classes=num_classes,
            window_size=actual_window_size,
            sampling_rate=self.sampling_rate,
            f_low=self.f_low,
            f_mid=self.f_mid,
            channels=self.channels,
            embed_dim=self.embed_dim,
            proj_dim=self.proj_dim,
            spec_hidden=self.spec_hidden,
            spec_embed_dim=self.spec_embed_dim,
            dropout=self.cfg.dropout,
            grl_alpha=self.grl_alpha,
        ).to(self.cfg.device)

        self.logger.info(
            f"MultiResAlignedDisentangle: "
            f"in_ch={in_channels}, classes={num_classes}, "
            f"window={actual_window_size}, "
            f"total_params={model.count_parameters():,} "
            f"(aligned={model.count_aligned_parameters():,}, "
            f"specific={model.count_specific_parameters():,})"
        )

        # ── 5. DataLoaders ────────────────────────────────────────────────
        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val,   y_val)   if len(X_val)  > 0 else None
        ds_test  = WindowDataset(X_test,  y_test)  if len(X_test) > 0 else None

        worker_fn = get_worker_init_fn(self.cfg.seed)
        g = torch.Generator().manual_seed(self.cfg.seed)

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_fn if self.cfg.num_workers > 0 else None,
            generator=g,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        ) if ds_test else None

        # ── 6. Loss functions ─────────────────────────────────────────────
        if self.cfg.use_class_weights:
            cw = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = cw.sum() / (cw + 1e-8)
            cw = cw / cw.mean()
            weight_t  = torch.from_numpy(cw).float().to(self.cfg.device)
            ce_criterion = nn.CrossEntropyLoss(weight=weight_t)
            self.logger.info(f"Class weights: {cw.round(3).tolist()}")
        else:
            ce_criterion = nn.CrossEntropyLoss()

        # Adversarial criterion: unweighted so the adversarial classifier does
        # not disproportionately focus on majority classes.
        adv_criterion = nn.CrossEntropyLoss()

        # ── 7. Optimizer ──────────────────────────────────────────────────
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 8. Training loop ──────────────────────────────────────────────
        total_epochs  = self.cfg.epochs
        stage1_epochs = min(self.stage1_epochs, total_epochs)
        device        = self.cfg.device

        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "ce_loss":    [], "align_loss": [], "adv_loss": [],
            "stage":      [],
        }
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        stage_prev    = 0  # track stage transitions for counter reset

        for epoch in range(1, total_epochs + 1):
            stage = 1 if epoch <= stage1_epochs else 2

            # Reset early-stopping on stage transition so Stage 2 gets a fair
            # window.  best_val_loss resets too since Stage-2 loss includes
            # the adversarial term which changes the loss scale.
            if stage != stage_prev:
                best_val_loss = float("inf")
                best_state    = None
                no_improve    = 0
                stage_prev    = stage
                if stage == 2:
                    self.logger.info(
                        f"[Epoch {epoch}] Transitioning to Stage 2 "
                        f"(adversarial disentanglement)."
                    )

            # ── Train ─────────────────────────────────────────────────────
            model.train()
            tot_loss = tot_ce = tot_align = tot_adv = 0.0
            correct = n_samples = 0

            for X_b, y_b in dl_train:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()

                out  = model(X_b, stage=stage)
                ce   = ce_criterion(out["logits"], y_b)
                aln  = self._alignment_loss(out["projections"])
                loss = ce + self.alpha_align * aln

                if stage == 2:
                    adv  = self._adversarial_loss(
                        out["adv_logits"], y_b, adv_criterion
                    )
                    loss = loss + self.beta_adv * adv
                    tot_adv += adv.item() * X_b.size(0)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = X_b.size(0)
                tot_loss  += loss.item() * bs
                tot_ce    += ce.item()   * bs
                tot_align += aln.item()  * bs
                correct   += (out["logits"].argmax(1) == y_b).sum().item()
                n_samples += bs

            train_loss = tot_loss  / n_samples
            train_acc  = correct   / n_samples
            avg_ce     = tot_ce    / n_samples
            avg_align  = tot_align / n_samples
            avg_adv    = tot_adv   / n_samples if stage == 2 else 0.0

            # ── Validate ──────────────────────────────────────────────────
            val_loss = train_loss
            val_acc  = train_acc

            if dl_val:
                model.eval()
                v_loss = v_correct = v_n = 0
                with torch.no_grad():
                    for X_v, y_v in dl_val:
                        X_v, y_v = X_v.to(device), y_v.to(device)
                        # Validation always uses stage=1 forward: only aligned
                        # encoder + CE + alignment.  No adversarial components.
                        # This gives a stable, comparable validation signal
                        # across both stages.
                        out_v  = model(X_v, stage=1)
                        l_v    = ce_criterion(out_v["logits"], y_v)
                        v_loss += l_v.item() * X_v.size(0)
                        v_correct += (
                            out_v["logits"].argmax(1) == y_v
                        ).sum().item()
                        v_n += X_v.size(0)
                val_loss = v_loss    / v_n
                val_acc  = v_correct / v_n

            scheduler.step(val_loss)

            # Record
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["ce_loss"].append(avg_ce)
            history["align_loss"].append(avg_align)
            history["adv_loss"].append(avg_adv)
            history["stage"].append(stage)

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(
                    f"[Epoch {epoch:3d}/{total_epochs}] "
                    f"Stage={stage} | "
                    f"Loss={train_loss:.4f} "
                    f"(CE={avg_ce:.4f} Align={avg_align:.4f}"
                    + (f" Adv={avg_adv:.4f}" if stage == 2 else "")
                    + f") | ValLoss={val_loss:.4f} | "
                    f"TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f} | "
                    f"NoImprove={no_improve}"
                )

            if no_improve >= self.cfg.early_stopping_patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {no_improve} epochs, Stage={stage})"
                )
                break

        # ── Restore best model ────────────────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)
            self.logger.info(f"Restored best model (val_loss={best_val_loss:.4f})")

        torch.save(model.state_dict(), self.output_dir / "best_model.pth")
        self.model = model

        # ── Evaluate on internal (val-split) test windows ─────────────────
        val_metrics: Dict = {}
        if dl_test:
            preds_list, labels_list = [], []
            model.eval()
            with torch.no_grad():
                for X_t, y_t in dl_test:
                    out_t = model(X_t.to(device), stage=1)
                    preds_list.append(out_t["logits"].argmax(1).cpu().numpy())
                    labels_list.append(y_t.numpy())
            preds_arr  = np.concatenate(preds_list)
            labels_arr = np.concatenate(labels_list)
            val_metrics = {
                "accuracy": float(accuracy_score(labels_arr, preds_arr)),
                "f1_macro": float(
                    f1_score(labels_arr, preds_arr, average="macro",
                             zero_division=0)
                ),
            }
            self.logger.info(
                f"Internal test split: "
                f"Acc={val_metrics['accuracy']:.4f} "
                f"F1={val_metrics['f1_macro']:.4f}"
            )

        return {
            "best_val_loss": best_val_loss,
            "history":       history,
            "val_metrics":   val_metrics,
        }

    # ── Inference ─────────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "test",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate model on numpy arrays (channels-first required by caller).

        LOSO integrity:
          - model.eval(): all BN statistics frozen from training.
          - stage=1: specific encoders + adversarial classifiers are NOT run.
          - _apply_standardization: uses mean_c/std_c from training split.
          - No test-time parameter updates.

        Args:
            X:          (N, C, T) or (N, T, C) — windows from the test subject
            y:          (N,) — class indices (0..K-1), NOT gesture IDs
            split_name: label for logging
            visualize:  unused (kept for interface compatibility)

        Returns:
            dict with "accuracy", "f1_macro", "report", "confusion_matrix"
        """
        assert self.model is not None, "Model must be trained before evaluation."
        assert self.mean_c is not None, "Normalisation stats missing."
        assert self.class_ids is not None, "class_ids not set."

        model  = self.model
        device = self.cfg.device
        model.eval()

        # Ensure (N, C, T) format
        if X.ndim == 3 and X.shape[1] > X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        # Apply training normalisation (frozen)
        X = self._apply_standardization(X, self.mean_c, self.std_c)

        ds = WindowDataset(X, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        preds_all, labels_all = [], []
        with torch.no_grad():
            for X_b, y_b in dl:
                # stage=1: only aligned encoders → gesture logits
                out = model(X_b.to(device), stage=1)
                preds_all.append(out["logits"].argmax(1).cpu().numpy())
                labels_all.append(y_b.numpy())

        preds  = np.concatenate(preds_all)
        labels = np.concatenate(labels_all)

        acc    = float(accuracy_score(labels, preds))
        f1     = float(f1_score(labels, preds, average="macro", zero_division=0))
        # class_names is Dict[gesture_id → str]; class_ids is sorted List[int].
        # target_names must align with class indices 0..K-1 in sorted order.
        if isinstance(self.class_names, dict):
            target_names = [
                self.class_names.get(cid, str(cid)) for cid in self.class_ids
            ]
        else:
            target_names = [str(c) for c in self.class_names]

        report = classification_report(
            labels, preds,
            target_names=target_names,
            zero_division=0,
        )
        cm = confusion_matrix(labels, preds).tolist()

        self.logger.info(
            f"[{split_name}] Accuracy={acc:.4f}  F1-macro={f1:.4f}"
        )
        self.logger.info(f"\n{report}")

        return {
            "accuracy":         acc,
            "f1_macro":         f1,
            "report":           report,
            "confusion_matrix": cm,
        }
