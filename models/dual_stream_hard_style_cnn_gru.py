"""
Dual-Stream Hard Style Augmentation CNN-GRU.

Hypothesis H100: Hard adversarial style perturbation forces z_content to be
invariant to EXTREME style variations beyond the convex hull of training styles.

Architecture:
    SharedEncoder → z_content (content_head) + z_style (style_head)
    FiLMLayer: conditions z_content on a (possibly adversarial) z_style
    UncertaintyMasker: soft mask M ∈ [0,1]^content_dim; suppresses content
        dims that correlate with adversarial style shifts
    GestureClassifier: shared across all three paths (base, easy, hard)
    SubjectClassifier: training only, supervises z_style

Training is orchestrated by DualStreamHardStyleTrainer:
    Stream 1 — Easy:
        z_style_easy = λ·z_style_i + (1-λ)·z_style_j  (λ ~ Beta)
        logits_easy  = GestureClassifier(FiLM(z_content, z_style_easy))

    Stream 2 — Hard:
        grad = ∇_{z_style} L_gesture(GestureClassifier(FiLM(z_content, z_style)))
        z_style_hard = z_style + ε · sign(grad)             FGSM-like
        ε = 0.5 · per-dim std(z_style in batch)
        z_style_hard clipped to [μ−3σ, μ+3σ] per dimension (plausibility)
        M = UncertaintyMasker(z_content)                    soft gate
        logits_hard  = GestureClassifier(FiLM(z_content*(1−M), z_style_hard))

    Loss:
        L_total = L_base + 0.3·L_easy + 0.7·L_hard + α·L_subject + β·L_MI

Inference (model.eval()):
    forward(x) → GestureClassifier(z_content)    ← NO FiLM, NO perturbation

LOSO safety:
    ✓ Style mixing/perturbation uses only TRAINING-batch z_style tensors
    ✓ ε and clipping bounds computed from training batch statistics only
    ✓ Test subject never appears in subject_labels, style pool, or running stats
    ✓ eval-mode forward uses only z_content — no style conditioning whatsoever
"""

import torch
import torch.nn as nn

from models.disentangled_cnn_gru import (
    SharedEncoder,
    ProjectionHead,
)


# ─────────────────────────── FiLM layer ─────────────────────────────────


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    output = (1 + γ(style)) · feature + β(style)

    Residual form ensures near-identity initialization when weights are
    initialised to small values. We use small random init (σ=0.01) so that
    the adversarial gradient w.r.t. z_style is non-zero from epoch 1 —
    unlike zero-init which would make the hard path dormant early in training.

    Args:
        feature_dim: dim of the feature to modulate (z_content)
        style_dim:   dim of the conditioning vector (z_style)
    """

    def __init__(self, feature_dim: int, style_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(style_dim, feature_dim)
        self.beta_net = nn.Linear(style_dim, feature_dim)

        # Small random init: FiLM ≈ identity at start, but non-zero
        # gradient w.r.t. z_style is immediately available for FGSM.
        nn.init.normal_(self.gamma_net.weight, std=0.01)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.normal_(self.beta_net.weight, std=0.01)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, feature_dim)
            style: (B, style_dim)
        Returns:
            (B, feature_dim) — modulated features
        """
        gamma = 1.0 + self.gamma_net(style)  # (B, feature_dim)
        beta = self.beta_net(style)           # (B, feature_dim)
        return gamma * x + beta


# ─────────────────────────── Uncertainty masker ──────────────────────────


class UncertaintyMasker(nn.Module):
    """
    Learned soft mask over content dimensions.

    M = σ(net(z_content))  in [0, 1]^content_dim

    Applied as: z_content_filtered = z_content * (1 − M)

    Suppresses content dimensions that correlate with adversarial style
    perturbations. Trained end-to-end via the hard-path gesture loss:
    dims that, when masked, reduce L_hard → the masker learns to mask them.

    Initialised so the mask starts near 0 (minimal suppression at t=0).
    This prevents destroying content representations at the start of training.
    """

    def __init__(self, content_dim: int):
        super().__init__()
        hidden_dim = max(content_dim // 4, 16)
        self.net = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim),
        )
        # Near-zero init: output layer bias = -3 → sigmoid(-3) ≈ 0.05
        # Initial mask suppresses only ~5% per dimension — nearly transparent.
        nn.init.normal_(self.net[0].weight, std=0.01)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.constant_(self.net[2].bias, -3.0)

    def forward(self, z_content: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_content: (B, content_dim)
        Returns:
            mask M:    (B, content_dim) in [0, 1]
        """
        return torch.sigmoid(self.net(z_content))


# ─────────────────────────── Main model ─────────────────────────────────


class DualStreamHardStyleCNNGRU(nn.Module):
    """
    Dual-Stream Hard Style Augmentation CNN-GRU.

    Exposes all sub-modules as public attributes so the trainer can
    orchestrate the three-path computation (base, easy, hard) in the
    training loop, including the FGSM adversarial style computation.

    eval-mode forward(x):
        shared    = encoder(x)
        z_content = content_head(shared)
        return    gesture_classifier(z_content)
        ← No FiLM, no style, no perturbation — LOSO clean.

    Training paths are managed externally by DualStreamHardStyleTrainer.
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
    ):
        """
        Args:
            in_channels:  EMG channel count
            num_gestures: number of gesture classes
            num_subjects: number of TRAINING subjects (for subject classifier)
            content_dim:  z_content dimensionality
            style_dim:    z_style dimensionality
            cnn_channels: channel sizes for CNN blocks (default: [32, 64, 128])
            gru_hidden:   GRU hidden size (bidirectional → 2 × gru_hidden output)
            gru_layers:   number of GRU layers
            dropout:      dropout probability
        """
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim

        # Shared encoder (CNN + BiGRU + Attention) — same as DisentangledCNNGRU
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # 2 × gru_hidden = 256

        # Projection heads
        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head = ProjectionHead(shared_dim, style_dim, dropout)

        # FiLM: conditions z_content on z_style (easy or adversarial)
        self.film = FiLMLayer(feature_dim=content_dim, style_dim=style_dim)

        # Uncertainty masker: soft gate over content dims for hard path
        self.uncertainty_masker = UncertaintyMasker(content_dim)

        # Shared gesture classifier — used by ALL three paths
        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # Subject classifier — training only, drives z_style → subject identity
        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass.

        Uses only z_content → gesture_classifier.
        No FiLM, no style conditioning, no masking.
        Safe for LOSO evaluation: no test-subject information required.

        Args:
            x: (B, C, T) — standardised EMG windows
        Returns:
            gesture_logits: (B, num_gestures)
        """
        shared = self.encoder(x)               # (B, shared_dim)
        z_content = self.content_head(shared)  # (B, content_dim)
        return self.gesture_classifier(z_content)
