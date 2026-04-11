"""
MixStyle Disentangled CNN-GRU for subject-invariant EMG gesture recognition.

Extends DisentangledCNNGRU (exp_31) with latent-space style mixing augmentation
inspired by MixStyle (Zhou et al., ICLR 2021) adapted for EMG domain generalization.

Key idea:
    Instead of trying to remove style (as in adversarial methods), we make
    z_content ROBUST to arbitrary convex combinations of training-subject styles.

Mechanism:
    1. Encode each window → z_content (content) + z_style (style)
    2. For each sample i in the batch, find sample j from a DIFFERENT training subject
    3. Mix their styles:  z_style_mix = λ·z_style_i + (1-λ)·z_style_j,  λ ~ Beta(α, α)
    4. Apply FiLM conditioning:  z_content_film = FiLM(z_content, z_style_mix)
    5. Dual-path gesture loss:
           L_gesture = CE(GestureClassifier(z_content),      y_gesture)      ← base path
                     + γ · CE(GestureClassifier(z_content_film), y_gesture)  ← mixed-style path
    6. At inference: GestureClassifier(z_content) only — NO FiLM, NO style needed

LOSO data-leakage guarantee:
    - style mixing happens ONLY within the current training batch
    - test subject's z_style NEVER enters the mixing pool
    - no test-time adaptation whatsoever
    - FiLM is only active during training; eval path is identical to exp_31 eval

FiLM residual initialisation:
    gamma_net and beta_net weights/biases are zero-initialised so the FiLM layer
    starts as an identity transform and gradually learns useful style modulation.

Reference:
    Zhou, K., et al. "Domain Generalization with MixStyle." ICLR 2021.
    Adapted from feature-statistic mixing to latent z_style mixing.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse encoder, projection head, and MI loss functions from exp_31 model
from models.disentangled_cnn_gru import (
    SharedEncoder,
    ProjectionHead,
    distance_correlation_loss,
    orthogonality_loss,
)


# ─────────────────────────── FiLM layer ────────────────────────────────


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Applies a learned affine transformation to a feature vector conditioned
    on an external style vector:

        output = (1 + γ(style)) · feature + β(style)

    The residual form (gamma offset by 1.0) ensures the layer starts as an
    identity transform when weights/biases are zero-initialised.

    Args:
        feature_dim: dimensionality of the feature to be modulated (z_content)
        style_dim:   dimensionality of the conditioning vector (z_style)
    """

    def __init__(self, feature_dim: int, style_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(style_dim, feature_dim)
        self.beta_net = nn.Linear(style_dim, feature_dim)

        # Zero-init → identity at the start of training
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, feature_dim)
            style: (B, style_dim)
        Returns:
            (B, feature_dim) — modulated features
        """
        gamma = 1.0 + self.gamma_net(style)   # (B, feature_dim)
        beta = self.beta_net(style)            # (B, feature_dim)
        return gamma * x + beta


# ─────────────────────────── Style mixing ──────────────────────────────


def mix_styles_across_subjects(
    z_style: torch.Tensor,
    subject_labels: torch.Tensor,
    alpha: float = 0.4,
) -> torch.Tensor:
    """
    Latent-space MixStyle augmentation.

    For each sample i in the batch, finds a random sample j from a DIFFERENT
    training subject and linearly interpolates their style vectors:

        z_style_mix[i] = λ[i] · z_style[i] + (1 - λ[i]) · z_style[j]

    where λ[i] ~ Beta(alpha, alpha).

    If all samples in the batch belong to the same subject (only possible when
    batch_size is small or there is a single training subject), the function
    falls back to an identity mix (z_style_mix[i] = z_style[i]).

    Args:
        z_style:        (B, style_dim) — encoded style vectors for current batch
        subject_labels: (B,)           — integer subject indices (train subjects only)
        alpha:          concentration parameter of the Beta distribution

    Returns:
        z_style_mix: (B, style_dim) — mixed style vectors (gradient flows through
                     z_style but not through the permuted partner z_style[j])
    """
    B = z_style.size(0)
    device = z_style.device

    # λ ~ Beta(alpha, alpha) — symmetric, centred at 0.5
    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32, device=device),
        torch.tensor(alpha, dtype=torch.float32, device=device),
    )
    lam = beta_dist.sample((B,)).unsqueeze(1)  # (B, 1)

    # For each sample find a random partner from a different training subject
    subject_labels_list = subject_labels.cpu().tolist()
    perm_indices = []
    for i in range(B):
        subj_i = subject_labels_list[i]
        candidates = [j for j in range(B) if subject_labels_list[j] != subj_i]
        if candidates:
            perm_indices.append(random.choice(candidates))
        else:
            # Entire batch is from one subject → identity (no mixing information)
            perm_indices.append(i)

    perm = torch.tensor(perm_indices, dtype=torch.long, device=device)

    # Detach the permuted partner so gradients only flow through the "anchor" sample.
    # This prevents gradient coupling between arbitrary pairs, which could lead
    # to training instability while still providing the augmentation signal.
    z_style_perm = z_style[perm].detach()

    z_style_mix = lam * z_style + (1.0 - lam) * z_style_perm
    return z_style_mix


# ─────────────────────────── Main model ────────────────────────────────


class MixStyleDisentangledCNNGRU(nn.Module):
    """
    MixStyle-augmented Content-Style Disentangled CNN-GRU.

    Training forward pass (model.train(), subject_labels provided):
        shared  = Encoder(x)
        z_content = ContentHead(shared)
        z_style   = StyleHead(shared)

        [Base path — used at inference]
        gesture_logits_base = GestureClassifier(z_content)

        [MixStyle path — training only]
        z_style_mix         = mix_styles_across_subjects(z_style, subject_labels)
        z_content_film      = FiLM(z_content, z_style_mix)
        gesture_logits_mix  = GestureClassifier(z_content_film)

        [Disentanglement auxiliaries]
        subject_logits      = SubjectClassifier(z_style)

        Returns dict with all tensors (for loss computation in trainer).

    Inference forward pass (model.eval()):
        shared  = Encoder(x)
        z_content = ContentHead(shared)
        gesture_logits = GestureClassifier(z_content)   ← NO FiLM, NO style needed

        Returns gesture_logits tensor.

    Notes:
        - The GestureClassifier is SHARED between both paths. This is intentional:
          training on z_content_film forces the classifier to generalise across
          diverse style combinations, making the z_content path more robust.
        - The SubjectClassifier is trained only from z_style (original), not z_style_mix.
          This keeps the disentanglement semantics clean.
        - FiLM weights are zero-initialised → identity at epoch 0; learned progressively.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_gestures: int = 10,
        num_subjects: int = 4,
        content_dim: int = 128,
        style_dim: int = 64,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
        mix_alpha: float = 0.4,
    ):
        """
        Args:
            in_channels:  number of EMG channels
            num_gestures: number of gesture classes
            num_subjects: number of training subjects (for subject classifier)
            content_dim:  dimensionality of z_content
            style_dim:    dimensionality of z_style
            cnn_channels: channel sizes for CNN blocks (default: [32, 64, 128])
            gru_hidden:   GRU hidden size (bidirectional → 2 × gru_hidden output)
            gru_layers:   number of GRU layers
            dropout:      dropout probability
            mix_alpha:    Beta distribution alpha for style mixing (0 → no mix, →0.5 → equal)
        """
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.mix_alpha = mix_alpha

        self.encoder = SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # 2 × gru_hidden = 256

        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head = ProjectionHead(shared_dim, style_dim, dropout)

        # FiLM: conditions z_content on mixed z_style (training only)
        self.film = FiLMLayer(feature_dim=content_dim, style_dim=style_dim)

        # Shared gesture classifier — used by BOTH base and mixed-style paths
        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # Subject classifier — training only, drives z_style to encode subject identity
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
        return_all: bool = False,
    ):
        """
        Args:
            x:              (B, C, T) — standardised EMG windows
            subject_labels: (B,) int  — training subject indices (required during training)
            return_all:     if True, force full-dict output even in eval mode

        Returns:
            Training mode or return_all=True → dict:
                "gesture_logits_base"  (B, num_gestures)  pure z_content, no FiLM
                "gesture_logits_mix"   (B, num_gestures)  FiLM-conditioned z_content
                "subject_logits"       (B, num_subjects)
                "z_content"            (B, content_dim)
                "z_style"              (B, style_dim)
                "z_style_mix"          (B, style_dim)
            Eval mode (default) → tensor:
                gesture_logits_base    (B, num_gestures)  — identical to inference path
        """
        shared = self.encoder(x)                      # (B, shared_dim)
        z_content = self.content_head(shared)          # (B, content_dim)
        z_style = self.style_head(shared)              # (B, style_dim)

        # Base path: gesture classification from raw content, no style conditioning
        gesture_logits_base = self.gesture_classifier(z_content)

        if not (self.training or return_all):
            # Pure inference: no FiLM, no subject labels needed, no leakage possible
            return gesture_logits_base

        # ── Training path ──────────────────────────────────────────────
        if self.training and subject_labels is not None:
            # Mix styles from different training subjects (LOSO-safe: test subject
            # never appears in subject_labels during training)
            z_style_mix = mix_styles_across_subjects(
                z_style, subject_labels, self.mix_alpha
            )
        else:
            # return_all=True in eval context: use identity (no actual mixing)
            z_style_mix = z_style.detach().clone()

        # FiLM-conditioned content
        z_content_film = self.film(z_content, z_style_mix)
        gesture_logits_mix = self.gesture_classifier(z_content_film)

        # Subject classifier on original z_style (not mixed)
        subject_logits = self.subject_classifier(z_style)

        return {
            "gesture_logits_base": gesture_logits_base,
            "gesture_logits_mix": gesture_logits_mix,
            "subject_logits": subject_logits,
            "z_content": z_content,
            "z_style": z_style,
            "z_style_mix": z_style_mix,
        }
