"""
Progressive Environment Diversification with Adaptive DRO Trainer (Experiment 107).

Three-phase training schedule:
    Phase 1 (epochs 1..phase1_end):
        Disentanglement only (ERM). No DRO, no MixStyle.
        Goal: learn good content/style representations.
        Loss = L_gest_base + alpha*L_subject + beta(t)*L_MI

    Phase 2 (epochs phase1_end+1..phase2_end):
        Add MixStyle virtual domains + soft GroupDRO.
        N real + M_mix virtual domains from pairwise style interpolation.
        Goal: expand style-space coverage.
        Loss = DRO(per-group losses) + alpha*L_subject + beta*L_MI

    Phase 3 (epochs phase2_end+1..end):
        Aggressive DRO + style extrapolation beyond convex hull.
        N real + M_mix + M_extrap virtual domains.
        Goal: prepare model for OOD subjects.
        Loss = DRO(per-group losses) + alpha*L_subject + beta*L_MI

GroupDRO mechanism:
    Groups = real subjects + virtual domains (unordered subject pairs).
    Real domain s: loss from base gesture path on samples from subject s.
    Virtual mix domain v: loss from FiLM-conditioned path on style-mixed samples
        where the subject pair matches pre-selected pair v.
    Virtual extrap domain e: loss from FiLM-conditioned path on style-extrapolated
        samples where the subject pair matches pre-selected pair e.

    Weight update (mirror descent):
        q_s <- q_s * exp(eta * L_s)
        q   <- q / sum(q)

Adaptive eta:
    eta(t) = eta_base * (1 + H(q)/H_max)
    where H(q) is Shannon entropy of group weights, H_max = log(num_groups).
    High entropy (uniform) -> increase step; low entropy -> decrease for stability.

Anti-collapse:
    If max(q) > threshold, reset q <- uniform.
    Prevents DRO weight degeneration.

LOSO compliance guarantees:
    - All virtual domains created from TRAINING subjects only.
    - Style extrapolation uses directed perturbation of training z_style,
      no access to test subject.
    - Phase transitions determined by epoch number, not test statistics.
    - Channel standardisation from train data only.
    - Validation from train-subject held-out split.
    - Test subject evaluated once after training completes.
    - Inference uses z_content only (no FiLM, no style).
"""

import itertools
import json
import logging
import random
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

from models.disentangled_cnn_gru import distance_correlation_loss, orthogonality_loss
from models.progressive_env_dro_emg import ProgressiveEnvDROModel
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from training.trainer import WindowDataset, get_worker_init_fn, seed_everything


# ─────────────────────────── Style manipulation helpers ──────────────────


def _mix_styles_for_pairs(
    z_style: torch.Tensor,
    subject_labels: torch.Tensor,
    selected_pairs: List[Tuple[int, int]],
    subject_to_indices: Dict[int, List[int]],
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create mixed styles for specific subject pairs present in the batch.

    For each sample i from subject s_i, find a partner j from subject s_j
    such that (s_i, s_j) is one of the selected pairs. Mix their styles:
        z_mix[i] = lambda * z_style[i] + (1-lambda) * z_style[j].detach()
        lambda ~ Beta(alpha, alpha)

    Samples that don't match any selected pair keep their original z_style.

    Args:
        z_style:            (B, style_dim) encoded styles
        subject_labels:     (B,) integer subject indices
        selected_pairs:     list of (sa, sb) unordered pair tuples
        subject_to_indices: {subject_id: [sample indices in batch]}
        alpha:              Beta distribution parameter

    Returns:
        z_style_mix:  (B, style_dim)
        domain_ids:   (B,) int — pair index for matched samples, -1 otherwise
    """
    B = z_style.size(0)
    device = z_style.device
    subj_list = subject_labels.cpu().tolist()

    # Build lookup: for each subject, which partners and pair IDs are available
    subject_partners: Dict[int, List[Tuple[int, int]]] = {}
    for idx, (sa, sb) in enumerate(selected_pairs):
        subject_partners.setdefault(sa, []).append((sb, idx))
        subject_partners.setdefault(sb, []).append((sa, idx))

    partner_idx = list(range(B))  # default: self (identity)
    domain_ids_np = np.full(B, -1, dtype=np.int64)
    has_partner = np.zeros(B, dtype=bool)

    for i in range(B):
        s_i = subj_list[i]
        if s_i not in subject_partners:
            continue

        # Collect candidate partner samples from batch
        candidates = []
        for partner_subj, pair_id in subject_partners[s_i]:
            if partner_subj in subject_to_indices:
                for j in subject_to_indices[partner_subj]:
                    candidates.append((j, pair_id))

        if candidates:
            chosen_j, chosen_domain = random.choice(candidates)
            partner_idx[i] = chosen_j
            domain_ids_np[i] = chosen_domain
            has_partner[i] = True

    # Vectorised mixing
    perm = torch.tensor(partner_idx, dtype=torch.long, device=device)
    domain_ids = torch.tensor(domain_ids_np, dtype=torch.long, device=device)

    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(alpha, device=device),
    )
    lam = beta_dist.sample((B,)).unsqueeze(1)  # (B, 1)

    z_partner = z_style[perm].detach()
    has_partner_t = torch.tensor(
        has_partner, device=device, dtype=torch.float32,
    ).unsqueeze(1)

    # For partnered: z_mix = lam*z_style + (1-lam)*z_partner
    # For non-partnered: z_mix = z_style (effective_lam = 1.0)
    effective_lam = 1.0 - has_partner_t * (1.0 - lam)
    z_style_mix = effective_lam * z_style + (1.0 - effective_lam) * z_partner

    return z_style_mix, domain_ids


def _extrapolate_styles_for_pairs(
    z_style: torch.Tensor,
    subject_labels: torch.Tensor,
    selected_pairs: List[Tuple[int, int]],
    subject_to_indices: Dict[int, List[int]],
    alpha_min: float = 0.1,
    alpha_max: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extrapolate styles BEYOND the convex hull for specific subject pairs.

    For each sample i from subject s_i, find partner j from subject s_j:
        z_extrap[i] = z_style[i] + a * (z_style[i] - z_style[j].detach())
        a ~ Uniform(alpha_min, alpha_max)

    This pushes the style vector away from the partner, creating OOD styles
    while staying in a controlled neighbourhood of the training distribution.

    LOSO-safe: only training-subject styles are used for extrapolation.

    Args:
        z_style:            (B, style_dim) encoded styles
        subject_labels:     (B,) integer subject indices
        selected_pairs:     list of (sa, sb) unordered pair tuples
        subject_to_indices: {subject_id: [sample indices in batch]}
        alpha_min, alpha_max: range for extrapolation magnitude

    Returns:
        z_style_extrap: (B, style_dim)
        domain_ids:     (B,) int — pair index for matched samples, -1 otherwise
    """
    B = z_style.size(0)
    device = z_style.device
    subj_list = subject_labels.cpu().tolist()

    subject_partners: Dict[int, List[Tuple[int, int]]] = {}
    for idx, (sa, sb) in enumerate(selected_pairs):
        subject_partners.setdefault(sa, []).append((sb, idx))
        subject_partners.setdefault(sb, []).append((sa, idx))

    partner_idx = list(range(B))
    domain_ids_np = np.full(B, -1, dtype=np.int64)
    has_partner = np.zeros(B, dtype=bool)

    for i in range(B):
        s_i = subj_list[i]
        if s_i not in subject_partners:
            continue

        candidates = []
        for partner_subj, pair_id in subject_partners[s_i]:
            if partner_subj in subject_to_indices:
                for j in subject_to_indices[partner_subj]:
                    candidates.append((j, pair_id))

        if candidates:
            chosen_j, chosen_domain = random.choice(candidates)
            partner_idx[i] = chosen_j
            domain_ids_np[i] = chosen_domain
            has_partner[i] = True

    perm = torch.tensor(partner_idx, dtype=torch.long, device=device)
    domain_ids = torch.tensor(domain_ids_np, dtype=torch.long, device=device)

    # Extrapolation magnitude: a ~ Uniform(alpha_min, alpha_max)
    a = torch.empty(B, 1, device=device).uniform_(alpha_min, alpha_max)

    z_partner = z_style[perm].detach()
    has_partner_t = torch.tensor(
        has_partner, device=device, dtype=torch.float32,
    ).unsqueeze(1)

    # For partnered: z_extrap = z_style + a*(z_style - z_partner)
    # For non-partnered: z_extrap = z_style (no extrapolation)
    direction = z_style - z_partner
    z_style_extrap = z_style + has_partner_t * a * direction

    return z_style_extrap, domain_ids


# ─────────────────────────── Trainer ─────────────────────────────────────


class ProgressiveEnvDROTrainer(DisentangledTrainer):
    """
    Progressive Environment Diversification with Adaptive DRO trainer.

    Inherits data preparation, normalization, evaluate_numpy, and artifact
    saving from DisentangledTrainer. Overrides fit() with phased training,
    GroupDRO, MixStyle, and style extrapolation.

    evaluate_numpy() is inherited unchanged because ProgressiveEnvDROModel
    returns a plain gesture_logits tensor in eval mode.
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # Disentanglement params (inherited)
        content_dim: int = 128,
        style_dim: int = 64,
        alpha: float = 0.5,
        beta: float = 0.1,
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",
        # MixStyle / extrapolation loss weights
        gamma: float = 0.5,         # weight of mixed-style gesture loss
        delta: float = 0.3,         # weight of extrapolated-style gesture loss
        # Phase boundaries (epoch numbers)
        phase1_end: int = 15,
        phase2_end: int = 30,
        # DRO step sizes
        eta_phase2: float = 0.003,
        eta_phase3: float = 0.01,
        # Virtual domain counts (capped at C(N,2))
        num_mix_pairs: int = 6,
        num_extrap_pairs: int = 6,
        # Style manipulation params
        mix_alpha: float = 0.4,
        extrap_alpha_min: float = 0.1,
        extrap_alpha_max: float = 0.5,
        # Anti-collapse
        collapse_threshold: float = 0.5,
    ):
        super().__init__(
            train_cfg, logger, output_dir, visualizer,
            content_dim, style_dim, alpha, beta, beta_anneal_epochs, mi_loss_type,
        )
        self.gamma = gamma
        self.delta = delta
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.eta_phase2 = eta_phase2
        self.eta_phase3 = eta_phase3
        self.num_mix_pairs = num_mix_pairs
        self.num_extrap_pairs = num_extrap_pairs
        self.mix_alpha = mix_alpha
        self.extrap_alpha_min = extrap_alpha_min
        self.extrap_alpha_max = extrap_alpha_max
        self.collapse_threshold = collapse_threshold

        self.final_group_weights: Optional[List[float]] = None

    # ── Phase logic ──────────────────────────────────────────────────────

    def _get_phase(self, epoch: int) -> str:
        """Determine training phase from epoch number."""
        if epoch <= self.phase1_end:
            return "phase1"
        elif epoch <= self.phase2_end:
            return "phase2"
        else:
            return "phase3"

    @staticmethod
    def _compute_adaptive_eta(
        group_weights: torch.Tensor,
        eta_base: float,
    ) -> float:
        """
        Adaptive DRO step size based on entropy of group weights.

        eta(t) = eta_base * (1 + H(q) / H_max)

        High entropy (uniform weights) -> larger step (more exploration).
        Low entropy (concentrated weights) -> smaller step (stability).
        """
        q = group_weights.clamp(min=1e-10)
        entropy = -(q * q.log()).sum().item()
        max_entropy = np.log(max(len(group_weights), 2))
        return eta_base * (1.0 + entropy / max(max_entropy, 1e-10))

    def _check_anti_collapse(self, group_weights: torch.Tensor) -> torch.Tensor:
        """Reset weights to uniform if any weight exceeds threshold."""
        if group_weights.max().item() > self.collapse_threshold:
            self.logger.debug(
                f"DRO collapse detected (max_q={group_weights.max().item():.3f}), "
                f"resetting to uniform"
            )
            return torch.ones_like(group_weights) / len(group_weights)
        return group_weights

    # ── Main training method ─────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:  # noqa: C901
        """
        Train with Progressive Environment Diversification + Adaptive DRO.

        Args:
            splits: dict with keys: "train", "val", "test",
                    "train_subject_labels", "num_train_subjects".

        Returns:
            Training / evaluation results dict.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Standard array preparation ────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # ── 2. Subject labels ────────────────────────────────────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "ProgressiveEnvDROTrainer requires 'train_subject_labels' in splits."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        assert len(y_subject_train) == len(y_train), (
            f"Subject labels ({len(y_subject_train)}) != "
            f"gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Progressive DRO: {num_train_subjects} train subjects | "
            f"label dist: {np.bincount(y_subject_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) -> (N, C, T) ─────────────────────────
        if X_train.ndim == 3:
            _N, dim1, dim2 = X_train.shape
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

        # ── 4. Per-channel standardisation (train only — LOSO clean) ─────
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

        # ── 5. Select virtual domain pairs ───────────────────────────────
        all_pairs = list(itertools.combinations(range(num_train_subjects), 2))
        rng = np.random.RandomState(self.cfg.seed)
        pair_order = rng.permutation(len(all_pairs))
        all_pairs = [all_pairs[i] for i in pair_order]

        mix_pairs = all_pairs[:min(self.num_mix_pairs, len(all_pairs))]
        extrap_pairs = all_pairs[:min(self.num_extrap_pairs, len(all_pairs))]

        n_mix = len(mix_pairs)
        n_extrap = len(extrap_pairs)

        num_groups_phase1 = num_train_subjects
        num_groups_phase2 = num_train_subjects + n_mix
        num_groups_phase3 = num_train_subjects + n_mix + n_extrap
        max_groups = num_groups_phase3

        self.logger.info(
            f"Virtual domains: {n_mix} mix pairs, {n_extrap} extrap pairs | "
            f"Groups per phase: P1={num_groups_phase1}, P2={num_groups_phase2}, "
            f"P3={num_groups_phase3}"
        )
        self.logger.info(f"Mix pairs: {mix_pairs}")
        self.logger.info(f"Extrap pairs: {extrap_pairs}")

        # ── 6. Model ─────────────────────────────────────────────────────
        model = ProgressiveEnvDROModel(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"ProgressiveEnvDROModel: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, content_dim={self.content_dim}, "
            f"style_dim={self.style_dim}, params={total_params:,}"
        )
        self.logger.info(
            f"Phases: 1->{self.phase1_end}, 2->{self.phase2_end}, "
            f"3->{self.cfg.epochs} | eta2={self.eta_phase2}, eta3={self.eta_phase3} | "
            f"gamma={self.gamma}, delta={self.delta}"
        )

        # ── 7. Datasets & DataLoaders ────────────────────────────────────
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        _dl_kwargs = dict(
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        )
        dl_train = DataLoader(
            ds_train, batch_size=self.cfg.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.cfg.seed), **_dl_kwargs,
        )
        dl_val = DataLoader(
            ds_val, batch_size=self.cfg.batch_size, shuffle=False, **_dl_kwargs,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test, batch_size=self.cfg.batch_size, shuffle=False, **_dl_kwargs,
        ) if ds_test else None

        # ── 8. Loss functions ────────────────────────────────────────────
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(self.cfg.device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            # reduction="none" for per-sample losses (needed by GroupDRO)
            gesture_criterion_none = nn.CrossEntropyLoss(
                weight=weight_tensor, reduction="none",
            )
            gesture_criterion_mean = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            gesture_criterion_none = nn.CrossEntropyLoss(reduction="none")
            gesture_criterion_mean = nn.CrossEntropyLoss()

        subject_criterion = nn.CrossEntropyLoss()

        # ── 9. Optimizer & scheduler ─────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 10. GroupDRO weights ─────────────────────────────────────────
        device = self.cfg.device
        group_weights = torch.ones(max_groups, device=device) / max_groups

        # ── 11. Training loop ────────────────────────────────────────────
        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_base_loss": [], "gesture_mix_loss": [],
            "gesture_extrap_loss": [],
            "subject_loss": [], "mi_loss": [],
            "group_weights": [], "phase": [],
            "effective_eta": [], "dro_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        prev_phase = None

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            phase = self._get_phase(epoch)
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))

            # ── Phase transition: reset group weights to uniform ─────────
            if phase != prev_phase and prev_phase is not None and phase != "phase1":
                if phase == "phase2":
                    num_g = num_groups_phase2
                else:
                    num_g = num_groups_phase3
                group_weights[:num_g] = 1.0 / num_g
                self.logger.info(
                    f"Phase transition {prev_phase} -> {phase}: "
                    f"reset {num_g} group weights to uniform"
                )
            prev_phase = phase

            # Active group count and eta for this phase
            if phase == "phase1":
                num_active_groups = num_groups_phase1
                current_eta = 0.0
            elif phase == "phase2":
                num_active_groups = num_groups_phase2
                current_eta = self._compute_adaptive_eta(
                    group_weights[:num_active_groups], self.eta_phase2,
                )
            else:
                num_active_groups = num_groups_phase3
                current_eta = self._compute_adaptive_eta(
                    group_weights[:num_active_groups], self.eta_phase3,
                )

            ep_total_loss = 0.0
            ep_gest_base = 0.0
            ep_gest_mix = 0.0
            ep_gest_extrap = 0.0
            ep_subject = 0.0
            ep_mi = 0.0
            ep_dro = 0.0
            ep_correct = 0
            ep_total = 0

            for windows, gesture_labels, subject_labels_batch in dl_train:
                windows = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels_batch = subject_labels_batch.to(device)

                optimizer.zero_grad()
                outputs = model(windows, return_all=True)

                z_content = outputs["z_content"]
                z_style = outputs["z_style"]

                # Build subject-to-indices mapping for this batch
                subj_list = subject_labels_batch.cpu().tolist()
                subject_to_indices: Dict[int, List[int]] = {}
                for i, s in enumerate(subj_list):
                    subject_to_indices.setdefault(s, []).append(i)

                # ── Per-sample base gesture loss ─────────────────────────
                per_sample_base_loss = gesture_criterion_none(
                    outputs["gesture_logits_base"], gesture_labels,
                )

                # ── Build group losses ───────────────────────────────────
                group_losses = []

                # Real domains (base path, per subject)
                for s in range(num_train_subjects):
                    mask_s = (subject_labels_batch == s)
                    if mask_s.any():
                        group_losses.append(per_sample_base_loss[mask_s].mean())
                    else:
                        group_losses.append(per_sample_base_loss.new_zeros(()))

                # Phase 2+: MixStyle virtual domains
                batch_mix_loss_val = 0.0
                if phase in ("phase2", "phase3") and mix_pairs:
                    z_style_mix, mix_domain_ids = _mix_styles_for_pairs(
                        z_style, subject_labels_batch, mix_pairs,
                        subject_to_indices, self.mix_alpha,
                    )
                    mix_logits = model.compute_film_logits(z_content, z_style_mix)
                    per_sample_mix_loss = gesture_criterion_none(
                        mix_logits, gesture_labels,
                    )

                    for v_idx in range(n_mix):
                        mask_v = (mix_domain_ids == v_idx)
                        if mask_v.any():
                            group_losses.append(per_sample_mix_loss[mask_v].mean())
                        else:
                            group_losses.append(per_sample_base_loss.new_zeros(()))

                    valid_mix = (mix_domain_ids >= 0)
                    if valid_mix.any():
                        batch_mix_loss_val = per_sample_mix_loss[valid_mix].mean().item()

                # Phase 3: Extrapolation virtual domains
                batch_extrap_loss_val = 0.0
                if phase == "phase3" and extrap_pairs:
                    z_style_extrap, extrap_domain_ids = _extrapolate_styles_for_pairs(
                        z_style, subject_labels_batch, extrap_pairs,
                        subject_to_indices,
                        self.extrap_alpha_min, self.extrap_alpha_max,
                    )
                    extrap_logits = model.compute_film_logits(
                        z_content, z_style_extrap,
                    )
                    per_sample_extrap_loss = gesture_criterion_none(
                        extrap_logits, gesture_labels,
                    )

                    for e_idx in range(n_extrap):
                        mask_e = (extrap_domain_ids == e_idx)
                        if mask_e.any():
                            group_losses.append(
                                per_sample_extrap_loss[mask_e].mean()
                            )
                        else:
                            group_losses.append(per_sample_base_loss.new_zeros(()))

                    valid_extrap = (extrap_domain_ids >= 0)
                    if valid_extrap.any():
                        batch_extrap_loss_val = (
                            per_sample_extrap_loss[valid_extrap].mean().item()
                        )

                group_losses_tensor = torch.stack(group_losses)

                # ── DRO weight update + loss ─────────────────────────────
                if phase != "phase1" and current_eta > 0:
                    with torch.no_grad():
                        active_gw = group_weights[:num_active_groups].clone()
                        active_gw = active_gw * torch.exp(
                            current_eta * group_losses_tensor.detach()
                        )
                        active_gw = active_gw / (active_gw.sum() + 1e-12)
                        active_gw = self._check_anti_collapse(active_gw)
                        group_weights[:num_active_groups] = active_gw

                    L_dro = (
                        group_weights[:num_active_groups].detach()
                        * group_losses_tensor
                    ).sum()
                else:
                    # Phase 1: ERM (simple mean of base gesture loss)
                    L_dro = per_sample_base_loss.mean()

                # ── Auxiliary losses ──────────────────────────────────────
                L_subject = subject_criterion(
                    outputs["subject_logits"], subject_labels_batch,
                )

                if self.mi_loss_type == "distance_correlation":
                    L_MI = distance_correlation_loss(z_content, z_style)
                elif self.mi_loss_type == "orthogonal":
                    L_MI = orthogonality_loss(z_content, z_style)
                else:  # "both"
                    L_MI = (
                        distance_correlation_loss(z_content, z_style)
                        + 0.1 * orthogonality_loss(z_content, z_style)
                    )

                # ── Total loss ────────────────────────────────────────────
                total_loss = L_dro + self.alpha * L_subject + current_beta * L_MI

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # ── Accumulate epoch stats ────────────────────────────────
                bs = windows.size(0)
                ep_total_loss += total_loss.item() * bs
                ep_gest_base += per_sample_base_loss.mean().item() * bs
                ep_gest_mix += batch_mix_loss_val * bs
                ep_gest_extrap += batch_extrap_loss_val * bs
                ep_subject += L_subject.item() * bs
                ep_mi += L_MI.item() * bs
                ep_dro += L_dro.item() * bs
                preds = outputs["gesture_logits_base"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total += bs

            # ── Epoch averages ────────────────────────────────────────────
            train_loss = ep_total_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            avg_base = ep_gest_base / max(1, ep_total)
            avg_mix = ep_gest_mix / max(1, ep_total)
            avg_extrap = ep_gest_extrap / max(1, ep_total)
            avg_subj = ep_subject / max(1, ep_total)
            avg_mi = ep_mi / max(1, ep_total)
            avg_dro = ep_dro / max(1, ep_total)

            # ── Validation (gesture base path only — mirrors inference) ───
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)  # eval -> gesture_logits_base
                        val_loss_sum += gesture_criterion_mean(
                            logits, yb
                        ).item() * yb.size(0)
                        val_correct += (logits.argmax(1) == yb).sum().item()
                        val_total += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            # ── Record history ────────────────────────────────────────────
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["gesture_base_loss"].append(avg_base)
            history["gesture_mix_loss"].append(avg_mix)
            history["gesture_extrap_loss"].append(avg_extrap)
            history["subject_loss"].append(avg_subj)
            history["mi_loss"].append(avg_mi)
            history["dro_loss"].append(avg_dro)
            history["group_weights"].append(
                group_weights[:num_active_groups].cpu().tolist()
            )
            history["phase"].append(phase)
            history["effective_eta"].append(current_eta)

            gw_top5 = sorted(
                group_weights[:num_active_groups].cpu().tolist(), reverse=True,
            )[:5]
            gw_str = ", ".join(f"{w:.3f}" for w in gw_top5)
            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] {phase.upper()} | "
                f"Train: total={train_loss:.4f} (base={avg_base:.4f}, "
                f"mix={avg_mix:.4f}, extrap={avg_extrap:.4f}, "
                f"subj={avg_subj:.4f}, MI={avg_mi:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"beta={current_beta:.4f}, eta={current_eta:.4f} | "
                f"top5_gw=[{gw_str}]"
            )

            # ── Early stopping on val gesture loss ────────────────────────
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

        # ── Store trainer state (needed by evaluate_numpy()) ──────────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size
        self.final_group_weights = group_weights.cpu().tolist()

        # ── Save training history ─────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png",
            )

        # ── Evaluate on internal val / test splits ────────────────────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str) -> Optional[Dict]:
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    logits = model(xb.to(device))
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits)
            y_true = np.concatenate(all_y)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0,
            )
            cm = confusion_matrix(
                y_true, y_pred, labels=np.arange(num_classes),
            )
            if self.visualizer is not None:
                cls_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, cls_labels, normalize=True,
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

        # ── Save checkpoint ───────────────────────────────────────────────
        model_path = self.output_dir / "progressive_env_dro.pt"
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
            "gamma": self.gamma,
            "delta": self.delta,
            "phase1_end": self.phase1_end,
            "phase2_end": self.phase2_end,
            "eta_phase2": self.eta_phase2,
            "eta_phase3": self.eta_phase3,
            "collapse_threshold": self.collapse_threshold,
            "final_group_weights": group_weights.cpu().tolist(),
            "mix_pairs": mix_pairs,
            "extrap_pairs": extrap_pairs,
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy() is inherited from DisentangledTrainer unchanged.
    # ProgressiveEnvDROModel returns gesture_logits_base tensor in eval mode,
    # so the inherited method works correctly for test-subject evaluation.
