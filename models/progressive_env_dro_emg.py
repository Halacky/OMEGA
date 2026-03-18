"""
Progressive Environment Diversification with Adaptive DRO model.

Hypothesis H107: Combine phased training, progressive virtual domain creation,
and adaptive GroupDRO to improve robustness in LOSO EMG gesture recognition.

Architecture (same as DisentangledCNNGRU + FiLM):
    Input (B, C, T) -> SharedEncoder (CNN+BiGRU+Attention) -> (B, 256)
      +-> ContentHead -> z_content (B, 128) -> GestureClassifier -> gesture_logits
      +-> StyleHead   -> z_style   (B, 64)  -> SubjectClassifier -> subject_logits
      + FiLMLayer for style-conditioned content modulation (training only)

Training mode: returns dict with gesture_logits_base, subject_logits, z_content, z_style.
Eval mode: returns gesture_logits_base tensor (no FiLM, no subject info).

The trainer handles:
    - Style mixing (MixStyle interpolation within convex hull)
    - Style extrapolation (beyond convex hull)
    - GroupDRO weight updates and phased training

via compute_film_logits() which applies FiLM conditioning to z_content
using externally computed style vectors and returns gesture predictions.

LOSO compliance:
    - Inference path uses only z_content (gesture-relevant, subject-invariant)
    - No subject labels or style info needed at test time
    - FiLM only active during training on training-subject data
"""

import torch
import torch.nn as nn

from models.disentangled_cnn_gru import SharedEncoder, ProjectionHead
from models.mixstyle_disentangled_cnn_gru import FiLMLayer


class ProgressiveEnvDROModel(nn.Module):
    """
    Content-Style Disentangled model with FiLM for progressive training.

    The model provides the architectural components; the trainer controls
    which components are active in each training phase.

    Phase 1: Only base path (z_content -> gesture_logits) + disentanglement
    Phase 2: + FiLM(z_content, z_style_mix) for MixStyle virtual domains
    Phase 3: + FiLM(z_content, z_style_extrap) for extrapolation domains

    Inference (all phases): gesture_logits from z_content only.
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
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim

        self.encoder = SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # 2 * gru_hidden = 256

        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head = ProjectionHead(shared_dim, style_dim, dropout)

        # FiLM: conditions z_content on modified z_style (training only).
        # Zero-initialised -> identity at epoch 0; learned progressively.
        self.film = FiLMLayer(feature_dim=content_dim, style_dim=style_dim)

        # Shared gesture classifier (used by base and FiLM-conditioned paths)
        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # Subject classifier (training only, drives z_style to encode subject identity)
        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x:          (B, C, T) standardised EMG windows
            return_all: force full-dict output even in eval mode

        Returns:
            Training or return_all=True -> dict:
                "gesture_logits_base"  (B, num_gestures)  z_content path, no FiLM
                "subject_logits"       (B, num_subjects)
                "z_content"            (B, content_dim)
                "z_style"              (B, style_dim)
            Eval mode (default) -> tensor:
                gesture_logits_base    (B, num_gestures)
        """
        shared = self.encoder(x)                    # (B, shared_dim)
        z_content = self.content_head(shared)        # (B, content_dim)
        z_style = self.style_head(shared)            # (B, style_dim)

        gesture_logits_base = self.gesture_classifier(z_content)

        if not (self.training or return_all):
            return gesture_logits_base

        subject_logits = self.subject_classifier(z_style)

        return {
            "gesture_logits_base": gesture_logits_base,
            "subject_logits": subject_logits,
            "z_content": z_content,
            "z_style": z_style,
        }

    def compute_film_logits(
        self,
        z_content: torch.Tensor,
        z_style_modified: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning and classify gestures.

        Used by the trainer for mixed-style and extrapolated-style paths.
        Gradients flow through both z_content and z_style_modified (unless
        the caller detaches the style vectors).

        Args:
            z_content:        (B, content_dim) — from forward pass
            z_style_modified: (B, style_dim)   — mixed or extrapolated style

        Returns:
            (B, num_gestures) gesture logits
        """
        z_film = self.film(z_content, z_style_modified)
        return self.gesture_classifier(z_film)
