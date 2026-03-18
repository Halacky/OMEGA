"""
DSFE (Domain-Specific Feature Exploitation) Style Bank CNN-GRU.

Key insight from CDDG: domain-specific features contain useful information
that should be exploited, not discarded. In exp_31, z_style is entirely
discarded at inference. But z_style encodes subject "type" (amplitude profile,
contraction speed), which can help interpret z_content.

Architecture:
    Same encoder as exp_31: CNN+BiGRU+Attention → z_content + z_style
    FiLM conditioning: z_content_film = FiLM(z_content, style_vector)

    Training:
        - Base path: Classifier(z_content) — regularizer
        - FiLM+MixStyle path: Classifier(FiLM(z_content, z_style_mix))
        - Style bank path: for each anchor s_k, Classifier(FiLM(z_content, s_k))
          → style-averaged logits → GroupDRO per training subject
        - Subject classifier on z_style + distance correlation(z_content, z_style)

    Inference:
        - z_style of test subject NOT computed
        - For each style anchor s_k from training subjects:
            logits_k = Classifier(FiLM(z_content, s_k))
        - final_logits = mean(logits_1, ..., logits_K)

Differences from related experiments:
    vs exp_31: z_style is exploited via style bank, not discarded
    vs exp_60: inference doesn't need test subject's z_style — uses train anchors
    vs exp_28: style embedding comes from training subjects only, not test subject

LOSO guarantee:
    - Style bank contains ONLY EMA anchors from training subjects
    - z_style of test subject never computed or used at inference
    - FiLM conditioning at inference uses only training-subject anchors
    - Complete isolation of test subject
"""

import torch
import torch.nn as nn

from models.disentangled_cnn_gru import (
    SharedEncoder,
    ProjectionHead,
)
from models.mixstyle_disentangled_cnn_gru import FiLMLayer, mix_styles_across_subjects


class DSFEStyleBankCNNGRU(nn.Module):
    """
    DSFE Style Bank CNN-GRU for subject-invariant gesture recognition
    with domain-specific feature exploitation.

    Training: dual path (base + FiLM MixStyle) + style bank GroupDRO.
    Inference: multi-style averaging over training subject style anchors.
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
        shared_dim = self.encoder.gru_output_dim  # 2 * gru_hidden = 256

        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head = ProjectionHead(shared_dim, style_dim, dropout)

        # FiLM: conditions z_content on style vector
        self.film = FiLMLayer(feature_dim=content_dim, style_dim=style_dim)

        # Shared gesture classifier — used by base, FiLM, and style bank paths
        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # Subject classifier (training only)
        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

    def _classify_with_anchors(
        self, z_content: torch.Tensor, style_anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gesture logits for each style anchor.

        Args:
            z_content:     (B, content_dim)
            style_anchors: (K, style_dim) — K style bank anchors

        Returns:
            (K, B, num_gestures) — per-anchor logits
        """
        B = z_content.size(0)
        K = style_anchors.size(0)
        all_logits = []
        for k in range(K):
            anchor_k = style_anchors[k].unsqueeze(0).expand(B, -1)  # (B, style_dim)
            z_film_k = self.film(z_content, anchor_k)
            logits_k = self.gesture_classifier(z_film_k)
            all_logits.append(logits_k)
        return torch.stack(all_logits, dim=0)  # (K, B, num_gestures)

    def forward(
        self,
        x: torch.Tensor,
        subject_labels: torch.Tensor = None,
        style_anchors: torch.Tensor = None,
        return_all: bool = False,
    ):
        """
        Args:
            x:              (B, C, T) standardized EMG windows
            subject_labels: (B,) training subject indices (training only)
            style_anchors:  (K, style_dim) style bank anchors
            return_all:     force full dict output in eval mode

        Returns:
            Training mode or return_all:
                dict with gesture_logits_base, gesture_logits_mix,
                subject_logits, z_content, z_style, z_style_mix,
                style_bank_logits (K, B, num_gestures) or None
            Eval mode + style_anchors:
                (B, num_gestures) averaged logits across style anchors
            Eval mode (no style_anchors):
                (B, num_gestures) base gesture logits (fallback)
        """
        shared = self.encoder(x)                       # (B, shared_dim)
        z_content = self.content_head(shared)           # (B, content_dim)

        if not (self.training or return_all):
            # ── Inference path ──
            if style_anchors is not None and len(style_anchors) > 0:
                # Multi-style inference: average predictions across anchors
                # z_style of test subject is NOT computed (LOSO clean)
                per_anchor_logits = self._classify_with_anchors(
                    z_content, style_anchors
                )
                return per_anchor_logits.mean(dim=0)  # (B, num_gestures)
            # Fallback: base path (no style conditioning)
            return self.gesture_classifier(z_content)

        # ── Training path ──
        z_style = self.style_head(shared)               # (B, style_dim)

        # Base path: pure content, no FiLM
        gesture_logits_base = self.gesture_classifier(z_content)

        # MixStyle FiLM path
        if self.training and subject_labels is not None:
            z_style_mix = mix_styles_across_subjects(
                z_style, subject_labels, self.mix_alpha
            )
        else:
            z_style_mix = z_style.detach().clone()

        z_content_film = self.film(z_content, z_style_mix)
        gesture_logits_mix = self.gesture_classifier(z_content_film)

        # Style bank per-anchor logits (for GroupDRO during training)
        style_bank_logits = None
        if style_anchors is not None and len(style_anchors) > 0:
            style_bank_logits = self._classify_with_anchors(
                z_content, style_anchors
            )  # (K, B, num_gestures)

        # Subject classifier
        subject_logits = self.subject_classifier(z_style)

        return {
            "gesture_logits_base": gesture_logits_base,
            "gesture_logits_mix": gesture_logits_mix,
            "subject_logits": subject_logits,
            "z_content": z_content,
            "z_style": z_style,
            "z_style_mix": z_style_mix,
            "style_bank_logits": style_bank_logits,
        }
