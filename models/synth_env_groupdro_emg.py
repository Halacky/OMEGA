"""
Synthetic Environment Expansion + GroupDRO model (Experiment 103).

Architecture: identical to exp_31/exp_57 DisentangledCNNGRU, with one addition —
a FiLM layer exposed as ``self.film`` and a shared gesture_classifier also exposed
for direct use by SynthEnvGroupDROTrainer to compute virtual-environment losses.

Design rationale
----------------
The model's forward() does NOT call FiLM.  Virtual-environment losses are computed
by the trainer, which accesses ``model.film`` and ``model.gesture_classifier``
directly within the same autograd graph.  This keeps forward() clean and makes
the LOSO guarantee trivially provable by inspection:

    - model.eval() → returns gesture_logits = GestureClassifier(z_content)
    - No style input, no FiLM, no test-subject information anywhere.

Training forward (model.train()):
    shared           = Encoder(x)
    z_content        = ContentHead(shared)           # (B, content_dim)
    z_style          = StyleHead(shared)             # (B, style_dim)
    gesture_logits   = GestureClassifier(z_content)  # base path (real envs)
    subject_logits   = SubjectClassifier(z_style)    # disentanglement auxiliary
    Returns dict.

Inference forward (model.eval()):
    Returns gesture_logits tensor — z_content path only.

Virtual-environment path (computed externally by trainer):
    z_style_mix      = λ·z_style_i + (1-λ)·z_style_j   [Beta(α,α)]
    z_content_film   = model.film(z_content, z_style_mix)
    logits_virtual   = model.gesture_classifier(z_content_film)

References
----------
exp_57  — GroupDRO + Disentanglement (DRO objective)
exp_60  — MixStyle + FiLM (style interpolation mechanism)
"""

import torch
import torch.nn as nn

from models.disentangled_cnn_gru import SharedEncoder, ProjectionHead
from models.mixstyle_disentangled_cnn_gru import FiLMLayer


class SynthEnvGroupDROModel(nn.Module):
    """
    Content-style disentangled CNN-GRU for Synthetic Environment Expansion + GroupDRO.

    Parameters
    ----------
    in_channels  : number of EMG channels (e.g. 8)
    num_gestures : number of gesture classes
    num_subjects : number of training subjects (for subject classifier)
    content_dim  : dimensionality of z_content (task-relevant)
    style_dim    : dimensionality of z_style  (subject-specific)
    cnn_channels : channel sizes for CNN blocks (default: [32, 64, 128])
    gru_hidden   : GRU hidden units per direction (bidirectional → 2× output)
    gru_layers   : stacked GRU layers
    dropout      : dropout probability
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
        shared_dim = self.encoder.gru_output_dim  # 2 × gru_hidden = 256

        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head = ProjectionHead(shared_dim, style_dim, dropout)

        # FiLM: used externally by the trainer for virtual-environment losses.
        # Zero-initialised → identity transform at epoch 0, learned progressively.
        # NOT called inside forward() — only by SynthEnvGroupDROTrainer.
        self.film = FiLMLayer(feature_dim=content_dim, style_dim=style_dim)

        # Gesture classifier — shared between base path and virtual-env path.
        # Accessed directly by the trainer for FiLM-conditioned predictions.
        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # Subject classifier — drives z_style to encode subject identity.
        # Used only during training; output never seen by gesture classifier.
        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_labels: torch.Tensor = None,  # accepted but unused (API compat)
    ):
        """
        Args:
            x:              (B, C, T) — per-channel standardised EMG windows
            subject_labels: ignored; accepted for consistent call signature

        Returns:
            Training mode → dict:
                "gesture_logits_base" : (B, num_gestures)   base path (no FiLM)
                "z_content"           : (B, content_dim)
                "z_style"             : (B, style_dim)
                "subject_logits"      : (B, num_subjects)
            Eval mode → tensor (B, num_gestures) — z_content path only, no FiLM.
        """
        shared = self.encoder(x)               # (B, shared_dim)
        z_content = self.content_head(shared)  # (B, content_dim)
        z_style = self.style_head(shared)      # (B, style_dim)

        gesture_logits_base = self.gesture_classifier(z_content)

        if not self.training:
            # Inference: base path only.  No style input, no FiLM.
            # LOSO guarantee: test-subject information cannot leak here.
            return gesture_logits_base

        subject_logits = self.subject_classifier(z_style)

        return {
            "gesture_logits_base": gesture_logits_base,
            "z_content": z_content,
            "z_style": z_style,
            "subject_logits": subject_logits,
        }
