"""
CycleMix Disentangled CNN-GRU for channel-wise stochastic style mixing in EMG.

Hypothesis H98:
    Replace the global pairwise Beta-mixing of MixStyle (exp_60) with
    CycleMix: per-EMG-channel independent style recombination with stochastic
    mixing magnitude.

Key differences from MixStyleDisentangledCNNGRU (exp_60):
    1. Per-channel independent donor selection:
           For each of the 8 EMG-channel style groups the donor j_k is drawn
           independently from a different training subject.
           Virtual styles = N^8 = 65,536 (vs 6 pairs in MixStyle, N=4).
    2. Per-channel λ_k ~ Beta(α, α):
           Each channel group gets an independent interpolation weight.
    3. Epoch-level α randomisation:
           The trainer draws α ~ Uniform(alpha_low, alpha_high) once per epoch
           and passes it to forward() as epoch_alpha.  This anneals the mixing
           strength stochastically across training rather than using a fixed α.

Architecture (identical to MixStyleDisentangledCNNGRU, only mixing logic changed):
    Input (B, C, T) → SharedEncoder (CNN + BiGRU + Attention) → (B, 256)
      ├─ ContentHead → z_content (B, content_dim)
      └─ StyleHead   → z_style   (B, style_dim)   [style_dim must be divisible by num_channels]

    [Training only — base path at inference]
    z_style_mix = cyclemix_styles_channel_wise(z_style, subject_labels, num_channels, epoch_alpha)
    z_content_film = FiLM(z_content, z_style_mix)

    Losses (in trainer):
        L_gesture_base = CE(GestureClassifier(z_content),      y)
        L_gesture_mix  = CE(GestureClassifier(z_content_film), y)
        L_subject      = CE(SubjectClassifier(z_style),        y_subject)
        L_MI           = DistanceCorrelation(z_content, z_style)

    Inference: GestureClassifier(z_content) only — NO FiLM, NO style.

LOSO data-leakage audit:
    ✓ cyclemix_styles_channel_wise() operates only on the current training batch.
      All samples in the batch come from the training DataLoader (train subjects only).
    ✓ Donor indices j_k are drawn from batch positions [0, B-1]; since the batch
      contains NO test-subject windows, test style can never enter the mixing pool.
    ✓ subject_labels contains integer indices [0, N_train-1]; test subject has no index.
    ✓ Inference (eval mode) uses only z_content; no style information is required.
    ✓ Channel standardisation statistics computed on training windows only.
    ✓ Early stopping monitored on a held-out val subset drawn from TRAINING subjects.
    ✓ No test-time adaptation.

Reference:
    MixStyle: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
    Extended to channel-wise cyclic style recombination for EMG domain generalisation.
"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn

# Reuse encoder and projection head from exp_31 base model
from models.disentangled_cnn_gru import (
    SharedEncoder,
    ProjectionHead,
    distance_correlation_loss,
    orthogonality_loss,
)

# Reuse FiLM layer from exp_60 model (zero-init identity at epoch 0)
from models.mixstyle_disentangled_cnn_gru import FiLMLayer


# ─────────────────────────── CycleMix mixing ───────────────────────────


def cyclemix_styles_channel_wise(
    z_style: torch.Tensor,
    subject_labels: torch.Tensor,
    num_channels: int,
    epoch_alpha: float,
) -> torch.Tensor:
    """
    CycleMix: per-EMG-channel independent stochastic style mixing.

    For each of the num_channels style groups (contiguous slices of z_style):
        1. Select a random donor j_k from a DIFFERENT training subject.
           j_k is drawn independently for each channel — this is the key novelty.
        2. Sample λ_k ~ Beta(epoch_alpha, epoch_alpha) per sample per channel.
        3. Mix: z_mix[k] = λ_k · z_style[k] + (1 - λ_k) · z_donor_j_k[k]

    Virtual style diversity: with N training subjects and C EMG channels,
    the number of unique channel-wise style combinations is N^C
    (e.g. 4^8 = 65,536 vs C(4,2) = 6 in standard MixStyle).

    LOSO audit:
        ✓ z_style comes exclusively from the training DataLoader — no test windows.
        ✓ perm_k selects from batch positions [0, B-1] ⊂ training set only.
        ✓ subject_labels contains training-subject indices only (test has no index).
        ✓ Gradient flows through z_style[:, sl] (self component) only.
          z_style[perm_k, sl].detach() is a constant offset → no inter-sample
          gradient coupling, stable training.

    Args:
        z_style:        (B, style_dim)  — encoded style vectors, training batch only
        subject_labels: (B,) int        — integer indices of training subjects
        num_channels:   int             — number of EMG channels = number of style groups
        epoch_alpha:    float           — α parameter for Beta(α, α);
                                          sampled once per epoch by the trainer

    Returns:
        z_style_mix: (B, style_dim) — per-channel mixed style vectors;
                     grad flows through the self component of each channel group
    """
    B, style_dim = z_style.shape
    device = z_style.device

    # Divide style_dim into num_channels groups.
    # If style_dim % num_channels != 0, the last group absorbs the remainder.
    base_size = style_dim // num_channels
    remainder = style_dim - base_size * num_channels
    group_sizes: List[int] = [base_size] * num_channels
    if remainder > 0:
        group_sizes[-1] += remainder

    # Pre-compute slice boundaries
    group_slices: List[slice] = []
    offset = 0
    for gs in group_sizes:
        group_slices.append(slice(offset, offset + gs))
        offset += gs

    # Per-channel λ ~ Beta(epoch_alpha, epoch_alpha): shape (num_channels, B, 1)
    # Each channel and each sample gets an independent mixing coefficient.
    alpha_t = torch.tensor(epoch_alpha, dtype=torch.float32, device=device)
    beta_dist = torch.distributions.Beta(alpha_t, alpha_t)
    lam = beta_dist.sample((num_channels, B, 1))  # (C, B, 1)

    subject_labels_list = subject_labels.cpu().tolist()

    # Build mixed groups and concatenate — avoids in-place mutation of gradient tensor
    mixed_groups: List[torch.Tensor] = []

    for k, sl in enumerate(group_slices):
        # --- Independent donor selection for channel k ---
        # For each sample i in the batch, pick a random j from a DIFFERENT subject.
        # This is done per-channel, so donors differ across channels for the same sample.
        perm_indices: List[int] = []
        for i in range(B):
            subj_i = subject_labels_list[i]
            candidates = [j for j in range(B) if subject_labels_list[j] != subj_i]
            if candidates:
                perm_indices.append(random.choice(candidates))
            else:
                # Fallback: entire batch from one subject — identity (no mixing signal)
                perm_indices.append(i)

        perm_k = torch.tensor(perm_indices, dtype=torch.long, device=device)

        # Donor slice for channel k — detached: no gradient through donor
        z_donor_k = z_style[perm_k, sl].detach()  # (B, gs)

        lam_k = lam[k]  # (B, 1) — independent per sample

        # z_style[:, sl] is differentiable; z_donor_k is a constant
        mixed_k = lam_k * z_style[:, sl] + (1.0 - lam_k) * z_donor_k  # (B, gs)
        mixed_groups.append(mixed_k)

    # Reconstruct full style vector from per-channel mixed groups
    z_style_mix = torch.cat(mixed_groups, dim=1)  # (B, style_dim)
    return z_style_mix


# ─────────────────────────── Main model ────────────────────────────────


class CycleMixDisentangledCNNGRU(nn.Module):
    """
    CycleMix-augmented Content-Style Disentangled CNN-GRU.

    Architecture identical to MixStyleDisentangledCNNGRU (exp_60); only the
    style mixing function changes (global pair → per-channel cyclic mix with
    stochastic epoch-level α).

    Training forward (model.train()):
        shared             = SharedEncoder(x)           → (B, shared_dim)
        z_content          = ContentHead(shared)        → (B, content_dim)
        z_style            = StyleHead(shared)          → (B, style_dim)

        [Base path — used at inference, no style needed]
        gesture_logits_base = GestureClassifier(z_content)

        [CycleMix path — training only]
        z_style_mix         = cyclemix_styles_channel_wise(z_style, subject_labels,
                                                            num_channels, epoch_alpha)
        z_content_film      = FiLM(z_content, z_style_mix)
        gesture_logits_mix  = GestureClassifier(z_content_film)

        [Disentanglement]
        subject_logits      = SubjectClassifier(z_style)  # original, not mixed

        Returns dict with all tensors for loss computation.

    Inference forward (model.eval()):
        shared    = SharedEncoder(x)
        z_content = ContentHead(shared)
        Returns GestureClassifier(z_content) tensor — NO FiLM, NO style, NO epoch_alpha.
        The test subject's style is never computed, requested, or needed.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_gestures: int = 10,
        num_subjects: int = 4,
        content_dim: int = 128,
        style_dim: int = 64,   # recommended: divisible by num_channels (8 × 8 = 64)
        num_channels: int = 8,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_channels:  number of EMG input channels
            num_gestures: number of gesture classes
            num_subjects: number of training subjects (for subject classifier head)
            content_dim:  dimensionality of z_content
            style_dim:    dimensionality of z_style; must be ≥ num_channels
                          (ideally divisible: style_dim = num_channels × k)
            num_channels: number of EMG channels = number of independent style groups
            cnn_channels: list of CNN filter counts (default: [32, 64, 128])
            gru_hidden:   GRU hidden units per direction (BiGRU → 2 × gru_hidden output)
            gru_layers:   number of stacked GRU layers
            dropout:      dropout probability
        """
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.num_channels = num_channels

        self.encoder = SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # 2 × gru_hidden = 256

        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head   = ProjectionHead(shared_dim, style_dim,   dropout)

        # FiLM modulates z_content conditioned on per-channel mixed z_style
        # Zero-init: identity transform at epoch 0, learned progressively
        self.film = FiLMLayer(feature_dim=content_dim, style_dim=style_dim)

        # Shared gesture classifier — used by both base and mixed-style paths
        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # Subject classifier: drives z_style to encode subject identity
        # (trained on original z_style, not z_style_mix)
        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_labels: torch.Tensor = None,
        epoch_alpha: float = 0.3,
    ):
        """
        Args:
            x:              (B, C, T) — per-channel standardised EMG windows
            subject_labels: (B,) int  — training subject indices [0, N_train-1]
                            Required during training; absent at inference.
            epoch_alpha:    float     — α for Beta(α, α) sampling of per-channel λ.
                            Randomised per epoch by the trainer; ignored at inference.

        Returns:
            Training mode → dict:
                "gesture_logits_base"  (B, num_gestures)  base path, no FiLM
                "gesture_logits_mix"   (B, num_gestures)  FiLM-conditioned path
                "subject_logits"       (B, num_subjects)
                "z_content"            (B, content_dim)
                "z_style"              (B, style_dim)
                "z_style_mix"          (B, style_dim)
            Eval mode → tensor (B, num_gestures):
                gesture_logits_base — pure content path, no style, no FiLM.
                The test subject's identity is never requested or used.
        """
        shared    = self.encoder(x)             # (B, shared_dim)
        z_content = self.content_head(shared)   # (B, content_dim)
        z_style   = self.style_head(shared)     # (B, style_dim)

        # Base path: gesture from content only — this is the inference path
        gesture_logits_base = self.gesture_classifier(z_content)

        if not self.training:
            # Inference: return gesture logits from z_content exclusively.
            # No style mixing, no FiLM, no subject_labels needed.
            # Test-subject style is never computed — LOSO clean.
            return gesture_logits_base

        # ── Training path: CycleMix augmentation ────────────────────────
        if subject_labels is not None:
            # Mix per EMG channel independently using training-batch donors only
            z_style_mix = cyclemix_styles_channel_wise(
                z_style=z_style,
                subject_labels=subject_labels,
                num_channels=self.num_channels,
                epoch_alpha=epoch_alpha,
            )
        else:
            # subject_labels absent in training (shouldn't happen, but safe fallback)
            z_style_mix = z_style.detach().clone()

        z_content_film     = self.film(z_content, z_style_mix)
        gesture_logits_mix = self.gesture_classifier(z_content_film)

        # Subject classifier on original z_style (not the mixed version)
        # Keeps disentanglement semantics: z_style should encode subject identity,
        # not the mixed virtual subject
        subject_logits = self.subject_classifier(z_style)

        return {
            "gesture_logits_base": gesture_logits_base,
            "gesture_logits_mix":  gesture_logits_mix,
            "subject_logits":      subject_logits,
            "z_content":           z_content,
            "z_style":             z_style,
            "z_style_mix":         z_style_mix,
        }
