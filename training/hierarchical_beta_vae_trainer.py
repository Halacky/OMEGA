"""
Trainer for Hierarchical Conditional β-VAE with UVMD (Experiment 108).

Loss
────
    L = L_class
      + β_content × KL_content               (content latent regularisation)
      + β_style   × KL_style                 (style latent regularisation)
      + λ_mi      × MI(z_content, z_style)   (content/style independence)
      + λ_overlap × spectral_overlap_penalty  (mode frequency separation)

where:
    KL_content = mean over K modes and C channels of KL(q_content || N(0,I))
    KL_style   = mean over K modes and C channels of KL(q_style   || N(0,I))
    MI         = mean distance correlation over all K×C (mode, channel) pairs
    overlap    = UVMD mode centre-frequency clustering penalty

β_content and β_style are annealed from 0 → target over beta_anneal_epochs
to allow the classifier to first learn a reasonable representation before the
KL pressure starts diverging latent codes toward N(0, I).

LOSO compliance (strictly enforced)
────────────────────────────────────
    ✓  Channel standardisation computed from X_train only.
       mean_c, std_c stored in trainer; applied to X_val, X_test without refit.
    ✓  Model parameters (UVMD, SoftAGC, CNN, VAE heads, ASP, classifier)
       receive gradients ONLY from training-subject batches.
    ✓  Validation loss computed in eval() mode; no gradients back-propagated.
    ✓  Early stopping monitors val_loss from train-subject validation split.
    ✓  Test subject is evaluated ONCE (model.eval(), no adaptation).
    ✓  reparameterisation at test time uses μ only — no stochastic sampling.
    ✓  BatchNorm running statistics accumulated from training samples; frozen
       in eval mode.
    ✓  No test-subject data or labels influence any trainable parameter.
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from models.hierarchical_beta_vae_uvmd import (
    HierarchicalBetaVAEUVMD,
    kl_divergence_gaussian,
    mean_distance_correlation_loss,
)
from training.trainer import (
    WindowClassifierTrainer,
    WindowDataset,
    get_worker_init_fn,
    seed_everything,
)


class HierarchicalBetaVAETrainer(WindowClassifierTrainer):
    """
    Trainer for Hierarchical Conditional β-VAE with UVMD.

    Inherits from WindowClassifierTrainer to reuse:
        _prepare_splits_arrays()        — converts Dict[gesture_id, ndarray] → flat arrays
        _compute_channel_standardization() — train-stats mean/std per channel
        _apply_standardization()        — applies stored stats to any split

    Parameters
    ──────────
    beta_content       : KL weight for content latent space
    beta_style         : KL weight for style latent space
    lambda_mi          : weight for distance-correlation MI term
    lambda_overlap     : weight for UVMD spectral overlap penalty
    beta_anneal_epochs : anneal β from 0 to target over this many epochs
    K                  : number of VMD modes
    L                  : number of unrolled ADMM iterations
    content_dim        : z_content dimension per (mode, channel)
    style_dim          : z_style dimension per (mode, channel)
    cnn_channels       : per-channel CNN hidden channel widths
    asp_bottleneck     : AttentiveStatsPooling attention bottleneck
    clf_hidden         : classification head hidden size
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # β-VAE loss weights
        beta_content: float = 0.05,
        beta_style: float = 0.05,
        lambda_mi: float = 0.1,
        lambda_overlap: float = 0.01,
        beta_anneal_epochs: int = 10,
        # Model architecture
        K: int = 4,
        L: int = 8,
        content_dim: int = 16,
        style_dim: int = 8,
        cnn_channels: Tuple[int, ...] = (32, 64),
        asp_bottleneck: int = 64,
        clf_hidden: int = 128,
    ) -> None:
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.beta_content = beta_content
        self.beta_style = beta_style
        self.lambda_mi = lambda_mi
        self.lambda_overlap = lambda_overlap
        self.beta_anneal_epochs = beta_anneal_epochs
        self.K = K
        self.L = L
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.cnn_channels = cnn_channels
        self.asp_bottleneck = asp_bottleneck
        self.clf_hidden = clf_hidden

        # Set by fit(); used by evaluate_numpy()
        self.model: Optional[HierarchicalBetaVAEUVMD] = None
        self.mean_c: Optional[np.ndarray] = None
        self.std_c: Optional[np.ndarray] = None
        self.class_ids: Optional[List[int]] = None
        self.class_names: Optional[Dict[int, str]] = None
        self.in_channels: Optional[int] = None
        self.window_size: Optional[int] = None

    # ────────────────────────────────────────────────────────────────────────
    # fit
    # ────────────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train the Hierarchical β-VAE UVMD model on training-subject splits.

        Args:
            splits: dict with keys "train", "val", "test" mapping
                    gesture_id (int) → ndarray (N, T, C)

        Returns:
            dict with "val" and "test" metric sub-dicts.

        LOSO integrity:
            · X_train / y_train contain ONLY train-subject windows.
            · X_val   / y_val   contain ONLY train-subject windows (held-out val split).
            · X_test  / y_test  contain ONLY test-subject windows.
            · Channel standardisation computed from X_train only.
            · The model never sees test-subject data during parameter updates.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Convert Dict splits → flat (N, T, C) arrays ───────────────
        # _prepare_splits_arrays returns arrays in (N, T, C) format.
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T) ──────────────────────────
        # All downstream operations (SoftAGC, CNN) expect channel-first format.
        # The UVMDBlock permutes back to (B, T, C) inside the model forward pass.
        # Heuristic: if dim1 > dim2 the data is likely (N, T, C).
        if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
            X_train = np.transpose(X_train, (0, 2, 1))   # (N, C, T)
            if len(X_val)  > 0: X_val  = np.transpose(X_val,  (0, 2, 1))
            if len(X_test) > 0: X_test = np.transpose(X_test, (0, 2, 1))
            self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        in_channels  = X_train.shape[1]
        window_size  = X_train.shape[2]
        num_classes  = len(class_ids)

        self.logger.info(
            f"Dataset — train: {X_train.shape}, val: {X_val.shape}, "
            f"test: {X_test.shape} | gestures: {num_classes}"
        )

        # ── 3. Per-channel standardisation (TRAIN STATS ONLY) ─────────────
        # mean_c, std_c are computed from X_train and stored in the trainer.
        # They are applied identically to X_val and X_test — no refit.
        # This prevents any leakage of test-subject amplitude information.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val)  > 0: X_val  = self._apply_standardization(X_val,  mean_c, std_c)
        if len(X_test) > 0: X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation (train stats only).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 4. Build model ────────────────────────────────────────────────
        model = HierarchicalBetaVAEUVMD(
            K=self.K,
            L=self.L,
            in_channels=in_channels,
            num_classes=num_classes,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            cnn_channels=self.cnn_channels,
            asp_bottleneck=self.asp_bottleneck,
            clf_hidden=self.clf_hidden,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"HierarchicalBetaVAEUVMD — K={self.K}, L={self.L}, "
            f"C={in_channels}, classes={num_classes}, "
            f"content_dim={self.content_dim}, style_dim={self.style_dim}, "
            f"params={total_params:,}"
        )
        self.logger.info(
            f"Loss weights — β_content={self.beta_content}, β_style={self.beta_style}, "
            f"λ_mi={self.lambda_mi}, λ_overlap={self.lambda_overlap}, "
            f"anneal_epochs={self.beta_anneal_epochs}"
        )

        # ── 5. Datasets & DataLoaders ─────────────────────────────────────
        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val,  y_val)  if len(X_val)  > 0 else None
        ds_test  = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init = get_worker_init_fn(self.cfg.seed)
        _dl_kw = dict(
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            **_dl_kw,
        )
        dl_val  = DataLoader(ds_val,  batch_size=self.cfg.batch_size, shuffle=False, **_dl_kw) if ds_val  else None
        dl_test = DataLoader(ds_test, batch_size=self.cfg.batch_size, shuffle=False, **_dl_kw) if ds_test else None

        # ── 6. Loss and optimiser ─────────────────────────────────────────
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = counts.sum() / (counts + 1e-8)
            cw = cw / cw.mean()
            weight_t = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Class weights: {cw.round(3).tolist()}")
            cls_criterion = nn.CrossEntropyLoss(weight=weight_t)
        else:
            cls_criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        device = self.cfg.device

        # ── 7. Training loop ──────────────────────────────────────────────
        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "loss_class": [], "loss_kl_content": [],
            "loss_kl_style": [], "loss_mi": [], "loss_overlap": [],
        }
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()

            # Annealing schedule: β ramps from 0 → target over beta_anneal_epochs.
            # Rationale: allow classifier to learn a useful representation first
            # before KL pressure pushes latents toward N(0, I).
            anneal_frac = min(1.0, epoch / max(1, self.beta_anneal_epochs))
            eff_beta_content = self.beta_content * anneal_frac
            eff_beta_style   = self.beta_style   * anneal_frac
            eff_lambda_mi    = self.lambda_mi     * anneal_frac

            ep_total = ep_correct = 0
            ep_loss = ep_loss_cls = ep_loss_kl_c = ep_loss_kl_s = 0.0
            ep_loss_mi = ep_loss_ov = 0.0

            for xb, yb in dl_train:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()

                # Forward (training mode → dict output with latent variables)
                out = model(xb)

                # ── Classification loss ───────────────────────────────
                L_class = cls_criterion(out["logits"], yb)

                # ── KL divergence for content latents ─────────────────
                # Mean over K modes (each gives (B, C, content_dim) mu/lv tensors).
                # kl_divergence_gaussian averages over all elements internally.
                kl_content_parts = [
                    kl_divergence_gaussian(mu, lv)
                    for mu, lv in zip(out["mu_content"], out["lv_content"])
                ]
                L_kl_content = torch.stack(kl_content_parts).mean()

                # ── KL divergence for style latents ───────────────────
                kl_style_parts = [
                    kl_divergence_gaussian(mu, lv)
                    for mu, lv in zip(out["mu_style"], out["lv_style"])
                ]
                L_kl_style = torch.stack(kl_style_parts).mean()

                # ── Distance correlation MI proxy ─────────────────────
                # Computes distance correlation for every (mode, channel) pair
                # using only the current training batch — no test-subject data.
                L_mi = mean_distance_correlation_loss(
                    out["z_content"], out["z_style"]
                )

                # ── UVMD spectral overlap penalty ─────────────────────
                # Penalises mode centre-frequency clustering.
                # Called on the UVMDBlock parameters (no input dependence).
                L_overlap = model.spectral_overlap_penalty()

                # ── Total loss ────────────────────────────────────────
                total_loss = (
                    L_class
                    + eff_beta_content * L_kl_content
                    + eff_beta_style   * L_kl_style
                    + eff_lambda_mi    * L_mi
                    + self.lambda_overlap * L_overlap
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = xb.size(0)
                ep_total    += bs
                ep_correct  += (out["logits"].argmax(1) == yb).sum().item()
                ep_loss     += total_loss.item() * bs
                ep_loss_cls += L_class.item()    * bs
                ep_loss_kl_c += L_kl_content.item() * bs
                ep_loss_kl_s += L_kl_style.item()   * bs
                ep_loss_mi  += L_mi.item()       * bs
                ep_loss_ov  += L_overlap.item()  * bs

            n = max(1, ep_total)
            train_loss = ep_loss / n
            train_acc  = ep_correct / n

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["loss_class"].append(ep_loss_cls / n)
            history["loss_kl_content"].append(ep_loss_kl_c / n)
            history["loss_kl_style"].append(ep_loss_kl_s / n)
            history["loss_mi"].append(ep_loss_mi / n)
            history["loss_overlap"].append(ep_loss_ov / n)

            # ── Validation ────────────────────────────────────────────
            if dl_val is not None:
                model.eval()
                val_loss_sum = val_correct = val_total = 0
                with torch.no_grad():
                    for xb_v, yb_v in dl_val:
                        xb_v, yb_v = xb_v.to(device), yb_v.to(device)
                        # Eval mode: model returns logits (z = μ, deterministic)
                        logits_v = model(xb_v)
                        val_loss_sum += cls_criterion(logits_v, yb_v).item() * yb_v.size(0)
                        val_correct  += (logits_v.argmax(1) == yb_v).sum().item()
                        val_total    += yb_v.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc  = val_correct  / max(1, val_total)
            else:
                val_loss = val_acc = float("nan")

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"loss={train_loss:.4f} "
                f"(cls={history['loss_class'][-1]:.4f}, "
                f"kl_c={history['loss_kl_content'][-1]:.4f}, "
                f"kl_s={history['loss_kl_style'][-1]:.4f}, "
                f"mi={history['loss_mi'][-1]:.4f}, "
                f"ov={history['loss_overlap'][-1]:.4f}), "
                f"train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
                f"β_c={eff_beta_content:.4f}, β_s={eff_beta_style:.4f}"
            )

            # ── Early stopping ────────────────────────────────────────
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── 8. Store trainer state ────────────────────────────────────────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 9. Save history and UVMD params ──────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        try:
            uvmd_params = model.get_learned_uvmd_params()
            with open(self.output_dir / "learned_uvmd_params.json", "w") as f:
                json.dump(uvmd_params, f, indent=4)
            self.logger.info(f"UVMD omega_k = {[f'{w:.4f}' for w in uvmd_params['omega_k']]}")
        except Exception:
            pass  # non-critical diagnostic

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── 10. Final evaluation on val and test splits ───────────────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str) -> Optional[Dict]:
            if dloader is None:
                return None
            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    # Eval mode: logits only (z = μ, no stochastic sampling)
                    logits = model(xb.to(device))
                    all_preds.append(logits.argmax(1).cpu().numpy())
                    all_true.append(yb.numpy())
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_true)
            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
            rpt  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm   = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                cls_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, cls_labels, normalize=True,
                    filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report": rpt,
                "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── 11. Save model checkpoint ─────────────────────────────────────
        ckpt_path = self.output_dir / "hierarchical_beta_vae_uvmd.pt"
        torch.save(
            {
                "state_dict":   model.state_dict(),
                "in_channels":  in_channels,
                "window_size":  window_size,
                "num_classes":  num_classes,
                "class_ids":    class_ids,
                "mean":         mean_c,
                "std":          std_c,
                "K":            self.K,
                "L":            self.L,
                "content_dim":  self.content_dim,
                "style_dim":    self.style_dim,
                "beta_content": self.beta_content,
                "beta_style":   self.beta_style,
                "lambda_mi":    self.lambda_mi,
                "lambda_overlap": self.lambda_overlap,
                "training_config": asdict(self.cfg),
            },
            ckpt_path,
        )
        self.logger.info(f"Checkpoint saved: {ckpt_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ────────────────────────────────────────────────────────────────────────
    # evaluate_numpy
    # ────────────────────────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate the trained model on raw (N, T, C) or (N, C, T) windows.

        Applies the same preprocessing pipeline used during training:
          1. Transpose (N, T, C) → (N, C, T) if needed.
          2. Per-channel standardisation using TRAIN-ONLY mean_c / std_c.
          3. model.eval() → forward pass → z = μ (deterministic, LOSO-safe).
          4. Argmax predictions → accuracy and macro-F1.

        LOSO safety:
            · mean_c / std_c are from training; no test-subject info used.
            · model.eval() freezes all BatchNorm and Dropout layers.
            · reparameterisation uses μ only (no sampling).
            · No gradients are computed (torch.no_grad()).
        """
        assert self.model is not None, "Must call fit() before evaluate_numpy()."
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None and self.class_names is not None

        # Transpose if (N, T, C)
        if X.ndim == 3 and X.shape[1] > X.shape[2]:
            X = np.transpose(X, (0, 2, 1))   # (N, C, T)

        X = self._apply_standardization(X, self.mean_c, self.std_c)

        ds = WindowDataset(X, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,   # safe for eval; no parallel workers needed
        )

        device = self.cfg.device
        self.model.eval()

        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in dl:
                logits = self.model(xb.to(device))  # (B, num_classes) — z = μ
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_true.append(yb.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        num_classes = len(self.class_ids)

        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rpt  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        if visualize and self.visualizer is not None:
            cls_labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, cls_labels, normalize=True,
                filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "report": rpt,
            "confusion_matrix": cm.tolist(),
        }
