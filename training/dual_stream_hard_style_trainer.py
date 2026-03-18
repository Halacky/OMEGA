"""
Trainer for Dual-Stream Hard Style Augmentation CNN-GRU (Experiment 100).

Extends DisentangledTrainer with:
- Three-path gesture loss: base (no FiLM) + easy (MixStyle) + hard (adversarial FGSM)
- FGSM-like adversarial style perturbation computed mid-batch via torch.autograd.grad
- Uncertainty masking applied on the hard path: z_content * (1 − M)
- Style mixing and adversarial perturbation only within training batches

Loss:
    L_total = L_gesture_base
            + 0.3 · L_gesture_easy     (MixStyle convex combination)
            + 0.7 · L_gesture_hard     (FGSM adversarial perturbation)
            + α   · L_subject          (subject classifier on z_style)
            + β(t)· L_MI               (distance correlation, annealed)

LOSO guarantee:
    ✓ dl_train built only from training-subject windows.
    ✓ subject_labels are indices into train_subjects list; test subject absent.
    ✓ z_style_adv = z_style.detach() — only training-batch styles used.
    ✓ ε = 0.5 · std(z_style across batch) — only training-batch statistics.
    ✓ Clipping [μ−3σ, μ+3σ] computed from training batch z_style only.
    ✓ torch.autograd.grad inner pass uses only z_content.detach() +
      z_style_adv (detached copy) — does NOT accumulate model .grad.
    ✓ Inference: model.forward(x) returns GestureClassifier(z_content) only.
    ✓ Early stopping monitored on val split from TRAINING subjects only.
"""

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

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

from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from models.dual_stream_hard_style_cnn_gru import DualStreamHardStyleCNNGRU
from models.disentangled_cnn_gru import distance_correlation_loss, orthogonality_loss


# ─────────────────────────── Style mixing (easy path) ───────────────────


def _mix_styles_across_subjects(
    z_style: torch.Tensor,
    subject_labels: torch.Tensor,
    alpha: float = 0.4,
) -> torch.Tensor:
    """
    Latent-space MixStyle for the easy stream.

    For each sample i finds a random sample j from a DIFFERENT training
    subject and linearly interpolates:
        z_mix[i] = λ[i] · z_style[i] + (1−λ[i]) · z_style[j].detach()

    λ ~ Beta(alpha, alpha).  Falls back to identity if all samples share
    the same subject (single-subject batch edge case).

    Input z_style must already be detached from the main graph so gradients
    do NOT flow through the mixed-style vector back to the style head.
    """
    B = z_style.size(0)
    device = z_style.device

    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32, device=device),
        torch.tensor(alpha, dtype=torch.float32, device=device),
    )
    lam = beta_dist.sample((B,)).unsqueeze(1)  # (B, 1)

    subj_list = subject_labels.cpu().tolist()
    perm_indices = []
    for i in range(B):
        candidates = [j for j in range(B) if subj_list[j] != subj_list[i]]
        perm_indices.append(random.choice(candidates) if candidates else i)

    perm = torch.tensor(perm_indices, dtype=torch.long, device=device)
    # Both z_style and z_style[perm] are already detached — no gradient flows
    z_style_mix = lam * z_style + (1.0 - lam) * z_style[perm]
    return z_style_mix


# ─────────────────────────── Adversarial perturbation (hard path) ────────


def _compute_adversarial_style(
    z_content: torch.Tensor,
    z_style: torch.Tensor,
    film: nn.Module,
    gesture_classifier: nn.Module,
    gesture_criterion: nn.Module,
    gesture_labels: torch.Tensor,
    epsilon_factor: float = 0.5,
) -> torch.Tensor:
    """
    FGSM-like adversarial perturbation of z_style.

    Computes the gradient of the gesture loss w.r.t. z_style (holding
    z_content fixed) and perturbs z_style in the direction that MAXIMISES
    the gesture loss — creating a hard virtual style.

    Implementation notes:
    - Uses a separate computation graph (z_style_adv is a new leaf tensor).
    - torch.autograd.grad does NOT accumulate gradients into model .grad
      attributes — the inner backward is invisible to the main optimizer.
    - z_content.detach() ensures the inner graph cannot reach the encoder.
    - ε = epsilon_factor · per-dim std(z_style across batch) — calibrated
      to training-batch scale; no test-subject statistics used.
    - Result is clipped to [μ−3σ, μ+3σ] per dimension for plausibility.

    LOSO safety:
    - z_style comes exclusively from training-batch windows (never test).
    - All statistics (mean, std) computed from the same training batch.
    - Returns a plain detached tensor; gradients do NOT flow back through
      z_style_hard to z_style or the style_head encoder.

    Args:
        z_content:          (B, content_dim) — detach() applied internally
        z_style:            (B, style_dim)   — training-batch styles
        film:               FiLMLayer module
        gesture_classifier: GestureClassifier module
        gesture_criterion:  CrossEntropyLoss (possibly weighted)
        gesture_labels:     (B,) int64 — training gesture labels
        epsilon_factor:     ε = epsilon_factor · std(z_style, dim=0)

    Returns:
        z_style_hard: (B, style_dim) — adversarially perturbed, no gradient
    """
    # ── Inner graph: only z_style_adv is a leaf requiring grad ────────
    z_style_adv = z_style.detach().requires_grad_(True)
    z_content_det = z_content.detach()

    # Forward through FiLM + classifier — inner graph only
    logits_inner = gesture_classifier(film(z_content_det, z_style_adv))
    loss_inner = gesture_criterion(logits_inner, gesture_labels)

    # Gradient of loss w.r.t. z_style_adv (does NOT touch model .grad)
    (grad_style,) = torch.autograd.grad(loss_inner, z_style_adv)

    # ── Perturbation (no gradient) ─────────────────────────────────────
    with torch.no_grad():
        z_style_det = z_style.detach()

        # Per-dimension ε from batch statistics (training data only)
        if z_style_det.size(0) >= 2:
            batch_std = z_style_det.std(dim=0, unbiased=False).clamp(min=1e-6)
        else:
            batch_std = torch.ones(z_style_det.size(1), device=z_style_det.device) * 0.1
        eps = epsilon_factor * batch_std.unsqueeze(0)  # (1, style_dim)

        # FGSM step: maximise loss → move in sign(grad) direction
        z_style_hard = z_style_det + eps * grad_style.sign()

        # Clip to plausible range [μ−3σ, μ+3σ] (training-batch statistics)
        batch_mean = z_style_det.mean(dim=0)   # (style_dim,)
        lo = (batch_mean - 3.0 * batch_std).unsqueeze(0)  # (1, style_dim)
        hi = (batch_mean + 3.0 * batch_std).unsqueeze(0)  # (1, style_dim)
        z_style_hard = z_style_hard.clamp(lo, hi)

    # z_style_hard is a plain tensor — no gradient, no computation graph
    return z_style_hard


# ─────────────────────────── Trainer ─────────────────────────────────────


class DualStreamHardStyleTrainer(DisentangledTrainer):
    """
    Trainer for Dual-Stream Hard Style Augmentation CNN-GRU.

    Inherits from DisentangledTrainer:
        _prepare_splits_arrays, _build_subject_labels_array
        _compute_channel_standardization, _apply_standardization
        evaluate_numpy (calls model.forward(x) in eval mode → gesture logits)

    Overrides fit() to:
        1. Instantiate DualStreamHardStyleCNNGRU.
        2. Implement the three-path training loop (base / easy / hard).
        3. Store self.model, self.mean_c, self.std_c, self.class_ids, etc.
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        content_dim: int = 128,
        style_dim: int = 64,
        alpha: float = 0.5,         # weight of subject classification loss
        beta: float = 0.1,          # weight of MI loss (annealed)
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",
        mix_alpha: float = 0.4,     # Beta param for easy MixStyle
        easy_weight: float = 0.3,   # weight of L_gesture_easy
        hard_weight: float = 0.7,   # weight of L_gesture_hard
        epsilon_factor: float = 0.5,  # ε = epsilon_factor · std(z_style)
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
        self.mix_alpha = mix_alpha
        self.easy_weight = easy_weight
        self.hard_weight = hard_weight
        self.epsilon_factor = epsilon_factor

    # ─────────────────────────── fit ─────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train DualStreamHardStyleCNNGRU.

        Expects splits:
            "train":               Dict[gesture_id, np.ndarray (N,T,C)]
            "val":                 Dict[gesture_id, np.ndarray (N,T,C)]
            "test":                Dict[gesture_id, np.ndarray (N,T,C)]  (optional)
            "train_subject_labels": Dict[gesture_id, np.ndarray (N,) int]
            "num_train_subjects":  int
        """
        seed_everything(self.cfg.seed)

        # ── 1. Prepare flat arrays ─────────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # ── 2. Subject labels (required for MixStyle easy path) ────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "DualStreamHardStyleTrainer requires 'train_subject_labels' in splits."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        assert len(y_subject_train) == len(y_train), (
            f"Subject labels ({len(y_subject_train)}) != gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Training: {num_train_subjects} subjects, "
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

        # ── 4. Per-channel standardisation (train stats only) ─────────
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation (train stats only).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Build model ─────────────────────────────────────────────
        model = DualStreamHardStyleCNNGRU(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"DualStreamHardStyleCNNGRU: in_ch={in_channels}, "
            f"gestures={num_classes}, subjects={num_train_subjects}, "
            f"content_dim={self.content_dim}, style_dim={self.style_dim}, "
            f"easy_w={self.easy_weight}, hard_w={self.hard_weight}, "
            f"epsilon_factor={self.epsilon_factor}, mix_alpha={self.mix_alpha}, "
            f"params={total_params:,}"
        )

        # ── 6. Datasets and loaders ────────────────────────────────────
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init = get_worker_init_fn(self.cfg.seed)
        dl_kwargs = dict(
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        dl_train = DataLoader(
            ds_train, shuffle=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            **dl_kwargs,
        )
        dl_val = DataLoader(ds_val, shuffle=False, **dl_kwargs) if ds_val else None
        dl_test = DataLoader(ds_test, shuffle=False, **dl_kwargs) if ds_test else None

        # ── 7. Loss functions ──────────────────────────────────────────
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            gesture_criterion = nn.CrossEntropyLoss()

        subject_criterion = nn.CrossEntropyLoss()

        # ── 8. Optimizer + scheduler ───────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 9. Training loop ───────────────────────────────────────────
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "L_base": [], "L_easy": [], "L_hard": [],
            "L_subject": [], "L_MI": [],
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))

            ep_total = ep_correct = 0
            ep_L_total = ep_L_base = ep_L_easy = ep_L_hard = 0.0
            ep_L_subject = ep_L_MI = 0.0

            for windows, gesture_labels, subject_labels in dl_train:
                windows = windows.to(device)            # (B, C, T)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                # ── Encode x → z_content, z_style ─────────────────
                shared = model.encoder(windows)          # (B, shared_dim)
                z_content = model.content_head(shared)   # (B, content_dim)
                z_style = model.style_head(shared)       # (B, style_dim)

                # ── BASE PATH: pure content, no conditioning ────────
                # Gradient flows: gesture_classifier ← z_content ← encoder
                logits_base = model.gesture_classifier(z_content)
                L_base = gesture_criterion(logits_base, gesture_labels)

                # ── EASY PATH: MixStyle convex combination ──────────
                # z_style is detached before mixing → gradient does NOT
                # flow back through the mixing operation to the style_head.
                # Gradient flows: gesture_classifier ← FiLM ← z_content ← encoder
                #                                    ← film params
                z_style_easy = _mix_styles_across_subjects(
                    z_style.detach(), subject_labels, self.mix_alpha
                )
                z_content_easy_cond = model.film(z_content, z_style_easy)
                logits_easy = model.gesture_classifier(z_content_easy_cond)
                L_easy = gesture_criterion(logits_easy, gesture_labels)

                # ── HARD PATH: adversarial perturbation ────────────
                #
                # Step A: compute z_style_hard via FGSM-like attack.
                #   - Inner graph is isolated: z_style_adv leaf, z_content.detach()
                #   - torch.autograd.grad returns grad tensor without accumulating
                #     into any model parameter's .grad
                #   - z_style_hard is a plain detached tensor (no grad)
                #
                # LOSO safety: z_style_adv uses only training-batch z_style
                # values; epsilon and clipping use only training-batch stats.
                z_style_hard = _compute_adversarial_style(
                    z_content=z_content,
                    z_style=z_style,
                    film=model.film,
                    gesture_classifier=model.gesture_classifier,
                    gesture_criterion=gesture_criterion,
                    gesture_labels=gesture_labels,
                    epsilon_factor=self.epsilon_factor,
                )

                # Step B: uncertainty masking on z_content.
                # Gradient flows: masker params ← mask ← z_content ← encoder
                # The masker learns which content dims correlate with
                # adversarial style shifts and should be suppressed.
                mask = model.uncertainty_masker(z_content)       # (B, content_dim)
                z_content_masked = z_content * (1.0 - mask)      # (B, content_dim)

                # Step C: FiLM with adversarial style on masked content.
                # z_style_hard has no gradient → gradient flows only through
                # z_content_masked and film params.
                z_content_hard_cond = model.film(z_content_masked, z_style_hard)
                logits_hard = model.gesture_classifier(z_content_hard_cond)
                L_hard = gesture_criterion(logits_hard, gesture_labels)

                # ── DISENTANGLEMENT ─────────────────────────────────
                # Subject classifier on original z_style (not mixed/perturbed)
                subject_logits = model.subject_classifier(z_style)
                L_subject = subject_criterion(subject_logits, subject_labels)

                # MI loss between z_content and z_style
                if self.mi_loss_type == "distance_correlation":
                    L_MI = distance_correlation_loss(z_content, z_style)
                elif self.mi_loss_type == "orthogonal":
                    L_MI = orthogonality_loss(z_content, z_style)
                elif self.mi_loss_type == "both":
                    L_MI = (
                        distance_correlation_loss(z_content, z_style)
                        + 0.1 * orthogonality_loss(z_content, z_style)
                    )
                else:
                    L_MI = distance_correlation_loss(z_content, z_style)

                # ── COMBINED LOSS ───────────────────────────────────
                # L_base + easy_weight·L_easy + hard_weight·L_hard
                #        + alpha·L_subject + beta(t)·L_MI
                total_loss = (
                    L_base
                    + self.easy_weight * L_easy
                    + self.hard_weight * L_hard
                    + self.alpha * L_subject
                    + current_beta * L_MI
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_total += bs
                ep_L_total += total_loss.item() * bs
                ep_L_base += L_base.item() * bs
                ep_L_easy += L_easy.item() * bs
                ep_L_hard += L_hard.item() * bs
                ep_L_subject += L_subject.item() * bs
                ep_L_MI += L_MI.item() * bs

                # Training accuracy from the base path (= inference metric)
                ep_correct += (logits_base.argmax(dim=1) == gesture_labels).sum().item()

            n = max(1, ep_total)
            train_loss = ep_L_total / n
            train_acc = ep_correct / n

            # ── Validation (base path = inference path, no FiLM) ────
            if dl_val is not None:
                model.eval()
                val_loss_sum = val_correct = val_total = 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)          # eval → base path only
                        val_loss_sum += gesture_criterion(logits, yb).item() * yb.size(0)
                        val_correct += (logits.argmax(1) == yb).sum().item()
                        val_total += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss = val_acc = float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["L_base"].append(ep_L_base / n)
            history["L_easy"].append(ep_L_easy / n)
            history["L_hard"].append(ep_L_hard / n)
            history["L_subject"].append(ep_L_subject / n)
            history["L_MI"].append(ep_L_MI / n)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"total={train_loss:.4f} "
                f"(base={ep_L_base/n:.4f}, easy={ep_L_easy/n:.4f}, "
                f"hard={ep_L_hard/n:.4f}, subj={ep_L_subject/n:.4f}, "
                f"MI={ep_L_MI/n:.4f}) "
                f"train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
                f"beta={current_beta:.4f}"
            )

            # ── Early stopping on val gesture loss (base path) ──────
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

        # ── Store trainer state for evaluate_numpy (inherited) ────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── Save training history ──────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── In-fold evaluation (val + test via base path) ─────────────

        results = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(device)
                    logits = model(xb)          # eval → base path
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits, axis=0)
            y_true = np.concatenate(all_y, axis=0)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
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

        # ── Save checkpoint ────────────────────────────────────────────
        model_path = self.output_dir / "dual_stream_hard_style_cnn_gru.pt"
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
                "mix_alpha": self.mix_alpha,
                "easy_weight": self.easy_weight,
                "hard_weight": self.hard_weight,
                "epsilon_factor": self.epsilon_factor,
                "alpha": self.alpha,
                "beta": self.beta,
                "training_config": asdict(self.cfg),
            },
            model_path,
        )
        self.logger.info(f"Model saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy is inherited from DisentangledTrainer:
    # calls model(xb) in eval mode → gesture_logits (base path, no FiLM).
