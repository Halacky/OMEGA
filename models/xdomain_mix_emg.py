"""
XDomainMix EMG: 4-Component Domain Decomposition with Cross-Domain Recombination.

Experiment 101 — Hypothesis H101.

Compared to MixStyle (exp_60) which uses 2 components (content, style), XDomainMix
decomposes the representation into 4 semantically distinct components:

    z_cg (class-generic,   32-dim): common EMG activation shared across all gestures
    z_cs (class-specific,  96-dim): gesture-discriminative features
    z_dg (domain-generic,  32-dim): stable physiological patterns shared across subjects
    z_ds (domain-specific, 64-dim): individual subject characteristics (amplitude, etc.)

Gesture classifier input: concat(z_cs, z_cg) → 128-dim
Domain classifier input:  concat(z_ds, z_dg) → 96-dim

Cross-domain augmentation (training only):
    For each sample i in training batch, find sample j from a DIFFERENT training subject.
    z_ds_swap[i] = z_ds[j].detach()  ← only domain-specific is swapped
    z_dg[i] stays unchanged           ← domain-generic is stable across subjects
    z_cs_film[i] = FiLM(z_cs[i], z_ds_swap[i])
    gesture_logits_aug = GestureHead(concat(z_cs_film, z_cg))  ← same gesture label

Disentanglement via cross-type orthogonality losses (4 pairs: cs-ds, cs-dg, cg-ds, cg-dg).

Inference path: encoder → z_cs, z_cg → GestureHead(concat(z_cs, z_cg)).
No domain information is needed or used at test time.

LOSO data-leakage audit:
    ✓ Domain-specific swap is performed only within the current training batch.
    ✓ subject_labels contains ONLY training-subject indices; test subject has no index.
    ✓ Test subject data never appears in any training batch or in the swap pool.
    ✓ Inference path uses z_cs + z_cg only — subject-invariant by construction.
    ✓ Channel standardisation statistics are computed on training data only.
    ✓ BatchNorm running statistics accumulated exclusively during training epochs.
    ✓ Validation split drawn from training subjects only (enforced in experiment file).
    ✓ FiLM conditioning is active only in training mode; eval path is identical to
      a plain gesture classifier on content embeddings.

References:
    MixStyle (Zhou et al., ICLR 2021): feature-statistic mixing for domain generalisation.
    XDomainMix concept: extend to 4-component decomposition for finer-grained separation
    of class and domain information in sEMG signals.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.disentangled_cnn_gru import (
    SharedEncoder,
    ProjectionHead,
    orthogonality_loss,
    distance_correlation_loss,
)

__all__ = [
    "XDomainMixEMG",
    "swap_domain_specific",
    "FiLMLayer",
    "orthogonality_loss",
    "distance_correlation_loss",
]


# ─────────────────────────── FiLM layer ─────────────────────────────────


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Applies a learned affine transformation to a feature vector conditioned
    on an external style vector:

        output = (1 + γ(style)) · feature + β(style)

    The residual form (gamma offset by 1.0) ensures the layer starts as an
    identity transform when weights/biases are zero-initialised.

    Args:
        feature_dim: dimensionality of the feature to modulate (z_cs)
        style_dim:   dimensionality of the conditioning vector (z_ds)
    """

    def __init__(self, feature_dim: int, style_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(style_dim, feature_dim)
        self.beta_net = nn.Linear(style_dim, feature_dim)

        # Zero-init → identity at epoch 0, learned gradually
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, feature_dim) — class-specific latent z_cs
            style: (B, style_dim)   — swapped domain-specific latent z_ds_swap
        Returns:
            (B, feature_dim) — FiLM-modulated z_cs
        """
        gamma = 1.0 + self.gamma_net(style)  # (B, feature_dim)
        beta = self.beta_net(style)           # (B, feature_dim)
        return gamma * x + beta


# ──────────────────── Domain-specific style swap ─────────────────────────


def swap_domain_specific(
    z_ds: torch.Tensor,
    subject_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-domain augmentation: replace each sample's z_ds with z_ds from a
    randomly selected sample belonging to a DIFFERENT training subject.

    Domain-generic (z_dg) is intentionally NOT swapped — it encodes physiological
    patterns presumed to be stable across subjects and should not be perturbed.

    LOSO safety guarantee:
        subject_labels contains ONLY training-subject indices (0 … K-1).
        The test subject is never registered, never appears in any training batch,
        and therefore can never enter the swap pool.

    Args:
        z_ds:           (B, ds_dim) — domain-specific latents for the training batch
        subject_labels: (B,)        — integer training-subject indices

    Returns:
        z_ds_swap: (B, ds_dim) — swapped domain-specific latents (detached).
            Detached so gradients only flow through the anchor sample z_ds[i],
            not through the partner z_ds[j], preventing gradient coupling between
            arbitrary pairs while still delivering the augmentation signal.
    """
    B = z_ds.size(0)
    device = z_ds.device
    subjects = subject_labels.cpu().tolist()

    perm_indices = []
    for i in range(B):
        # Find all samples in the batch from a different training subject
        candidates = [j for j in range(B) if subjects[j] != subjects[i]]
        if candidates:
            perm_indices.append(random.choice(candidates))
        else:
            # Entire batch from one subject (small batches or single-subject fold):
            # identity swap — no augmentation signal, but training continues safely.
            perm_indices.append(i)

    perm = torch.tensor(perm_indices, dtype=torch.long, device=device)
    return z_ds[perm].detach()


# ──────────────────────────── Main model ────────────────────────────────


class XDomainMixEMG(nn.Module):
    """
    XDomainMix: 4-Component EMG Domain Generalisation Model.

    Training forward pass (model.training=True, subject_labels provided):
        shared = Encoder(x)                              # (B, 256)
        z_cg = ProjCG(shared)                            # class-generic
        z_cs = ProjCS(shared)                            # class-specific
        z_dg = ProjDG(shared)                            # domain-generic
        z_ds = ProjDS(shared)                            # domain-specific

        [Base gesture path — identical to inference]
        gesture_logits_base = GestureHead(concat(z_cs, z_cg))

        [Cross-domain augmentation path — training only]
        z_ds_swap   = swap_domain_specific(z_ds, subject_labels)  # swap ds only
        z_cs_film   = FiLM(z_cs, z_ds_swap)                       # condition on swapped style
        gesture_logits_aug = GestureHead(concat(z_cs_film, z_cg)) # same gesture label

        [Domain classification — training only]
        domain_logits = DomainHead(concat(z_ds, z_dg))  # original z_ds, not swapped

        Returns dict with all tensors for loss computation.

    Inference forward pass (model.training=False):
        shared = Encoder(x)
        z_cg = ProjCG(shared)
        z_cs = ProjCS(shared)
        return GestureHead(concat(z_cs, z_cg))   ← pure content path, no domain info

    Notes:
        - GestureHead is SHARED between base and augmented paths. Training on virtual
          domains forces the classifier to generalise across diverse domain-style combinations.
        - DomainHead is trained on original z_ds (not swapped) to maintain clean semantics:
          each sample's domain head sees its own true domain representation.
        - FiLM weights are zero-initialised → identity at epoch 0, learned progressively.
        - z_dg is fixed and NOT swapped: domain-generic patterns are shared across subjects
          and swapping them would introduce spurious cross-subject variation in the
          "universal" component.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_gestures: int = 10,
        num_subjects: int = 4,
        cg_dim: int = 32,
        cs_dim: int = 96,
        dg_dim: int = 32,
        ds_dim: int = 64,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_channels:  number of EMG channels (typically 8)
            num_gestures: number of gesture classes
            num_subjects: number of training subjects (domain classifier output dim)
            cg_dim:       class-generic latent dimension
            cs_dim:       class-specific latent dimension
            dg_dim:       domain-generic latent dimension
            ds_dim:       domain-specific latent dimension
            cnn_channels: CNN block output channels (default: [32, 64, 128])
            gru_hidden:   GRU hidden size (bidirectional → 2 × gru_hidden)
            gru_layers:   number of GRU layers
            dropout:      dropout probability
        """
        super().__init__()
        self.cg_dim = cg_dim
        self.cs_dim = cs_dim
        self.dg_dim = dg_dim
        self.ds_dim = ds_dim

        # Shared encoder: CNN → BiGRU → Attention → (B, 256)
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # 2 × gru_hidden = 256

        # Four independent projection heads
        self.proj_cg = ProjectionHead(shared_dim, cg_dim, dropout)
        self.proj_cs = ProjectionHead(shared_dim, cs_dim, dropout)
        self.proj_dg = ProjectionHead(shared_dim, dg_dim, dropout)
        self.proj_ds = ProjectionHead(shared_dim, ds_dim, dropout)

        # FiLM: conditions z_cs on swapped z_ds (training augmentation only)
        self.film = FiLMLayer(feature_dim=cs_dim, style_dim=ds_dim)

        # Gesture classifier: input = concat(z_cs, z_cg)
        gesture_input_dim = cs_dim + cg_dim  # 96 + 32 = 128
        self.gesture_classifier = nn.Sequential(
            nn.Linear(gesture_input_dim, gesture_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gesture_input_dim // 2, num_gestures),
        )

        # Domain classifier: input = concat(z_ds, z_dg) — training only
        domain_input_dim = ds_dim + dg_dim  # 64 + 32 = 96
        self.domain_classifier = nn.Sequential(
            nn.Linear(domain_input_dim, domain_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(domain_input_dim, num_subjects),
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_labels: torch.Tensor = None,
        return_all: bool = False,
    ):
        """
        Args:
            x:              (B, C, T) — standardised EMG windows
            subject_labels: (B,) int  — training-subject indices (required in training mode)
            return_all:     if True, force full-dict output even in eval mode

        Returns:
            Training mode (or return_all=True) → dict:
                "gesture_logits_base"  (B, num_gestures)  base path (z_cs + z_cg)
                "gesture_logits_aug"   (B, num_gestures)  FiLM-augmented path
                "domain_logits"        (B, num_subjects)  domain (subject) predictions
                "z_cg"                 (B, cg_dim)
                "z_cs"                 (B, cs_dim)
                "z_dg"                 (B, dg_dim)
                "z_ds"                 (B, ds_dim)
                "z_ds_swap"            (B, ds_dim)        swapped domain-specific
            Eval mode (default) → tensor:
                gesture_logits_base    (B, num_gestures)  — pure inference path
        """
        shared = self.encoder(x)         # (B, shared_dim)
        z_cg = self.proj_cg(shared)      # (B, cg_dim)
        z_cs = self.proj_cs(shared)      # (B, cs_dim)
        z_dg = self.proj_dg(shared)      # (B, dg_dim)
        z_ds = self.proj_ds(shared)      # (B, ds_dim)

        # Base gesture path: f(concat(z_cs, z_cg)) — the inference-time path
        gesture_logits_base = self.gesture_classifier(
            torch.cat([z_cs, z_cg], dim=1)
        )

        if not (self.training or return_all):
            # Pure inference: no domain information, no test-subject leakage
            return gesture_logits_base

        # ── Training / analysis path ───────────────────────────────────────
        if self.training and subject_labels is not None:
            # Swap z_ds with a randomly selected training sample from a different subject.
            # z_dg is NOT swapped: domain-generic patterns are presumed stable.
            z_ds_swap = swap_domain_specific(z_ds, subject_labels)
        else:
            # return_all=True in eval context: identity swap (no actual mixing)
            z_ds_swap = z_ds.detach().clone()

        # FiLM-condition z_cs on the swapped domain-specific style
        z_cs_film = self.film(z_cs, z_ds_swap)

        # Augmented gesture path: f(concat(FiLM(z_cs, z_ds_swap), z_cg))
        gesture_logits_aug = self.gesture_classifier(
            torch.cat([z_cs_film, z_cg], dim=1)
        )

        # Domain classifier on ORIGINAL z_ds (not swapped) — clean domain semantics
        domain_logits = self.domain_classifier(
            torch.cat([z_ds, z_dg], dim=1)
        )

        return {
            "gesture_logits_base": gesture_logits_base,
            "gesture_logits_aug": gesture_logits_aug,
            "domain_logits": domain_logits,
            "z_cg": z_cg,
            "z_cs": z_cs,
            "z_dg": z_dg,
            "z_ds": z_ds,
            "z_ds_swap": z_ds_swap,
        }
