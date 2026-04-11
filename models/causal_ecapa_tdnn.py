"""
Causal ECAPA-TDNN with Content/Style Disentanglement.

Combines ECAPA-TDNN backbone (SE-Res2Net + Attentive Statistics Pooling)
with causal disentanglement of content (gesture-relevant) and style
(subject-specific) representations.

Disentanglement mechanism (inspired by CDDG, Neural Networks 2024):
  - Shared ECAPA-TDNN encoder produces a joint embedding.
  - Content head projects to gesture-invariant representation.
  - Style head projects to subject-specific representation.
  - Causal aggregation loss: minimizes per-class cross-subject variance
    in content space, treating each subject as an "environment".
  - Barlow Twins redundancy reduction: decorrelates content and style.
  - Reconstruction decoder: content + style → embedding (prevents collapse).

At inference, only the content branch is used — no subject labels needed.

Input format:  (B, C_emg, T)  — channels-first, matching PyTorch Conv1d.
Output format: (B, num_gestures) logits  (inference mode)
               dict with all branches     (training mode, return_all=True)
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn

from models.ecapa_tdnn_emg import SERes2NetBlock, AttentiveStatisticsPooling


# ─────────────────────────── loss functions ──────────────────────────────────

def barlow_twins_redundancy_loss(
    z_content: torch.Tensor,
    z_style: torch.Tensor,
) -> torch.Tensor:
    """
    Barlow Twins redundancy reduction between content and style.

    Computes the cross-correlation matrix between normalized content and
    style representations. For disentanglement, ALL cross-correlations
    should be zero (unlike self-supervised BT where diagonal should be 1).

    Args:
        z_content: (B, D_c) content representations.
        z_style:   (B, D_s) style representations.

    Returns:
        Scalar loss — mean squared cross-correlation.
    """
    batch_size = z_content.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=z_content.device)

    # Normalize to zero mean, unit variance per dimension
    z_c = (z_content - z_content.mean(dim=0)) / (z_content.std(dim=0) + 1e-5)
    z_s = (z_style - z_style.mean(dim=0)) / (z_style.std(dim=0) + 1e-5)

    # Cross-correlation matrix: (D_c, D_s)
    c = z_c.T @ z_s / batch_size

    # Mean squared cross-correlation (normalized by number of elements)
    loss = (c ** 2).sum() / (c.size(0) * c.size(1))

    return loss


def causal_aggregation_loss(
    z_content: torch.Tensor,
    gesture_labels: torch.Tensor,
    subject_labels: torch.Tensor,
    min_subjects_per_class: int = 2,
) -> torch.Tensor:
    """
    Causal aggregation loss (from CDDG).

    For each gesture class, computes per-subject mean content representation
    and minimizes variance across subjects. This encourages content features
    to capture only causal (gesture-relevant) factors invariant to the
    "environment" (subject).

    Args:
        z_content:       (B, D_c) content representations.
        gesture_labels:  (B,) gesture class indices.
        subject_labels:  (B,) subject indices (training subjects only).
        min_subjects_per_class: minimum subjects required to compute loss.

    Returns:
        Scalar loss — average per-class cross-subject variance.
    """
    device = z_content.device
    unique_gestures = gesture_labels.unique()

    loss = torch.tensor(0.0, device=device)
    valid_classes = 0

    for g in unique_gestures:
        mask_g = gesture_labels == g
        z_g = z_content[mask_g]
        subj_g = subject_labels[mask_g]

        unique_subjs = subj_g.unique()
        if len(unique_subjs) < min_subjects_per_class:
            continue

        # Per-subject mean content representation
        means = []
        for s in unique_subjs:
            mask_s = subj_g == s
            if mask_s.sum() >= 1:
                means.append(z_g[mask_s].mean(dim=0))

        if len(means) < min_subjects_per_class:
            continue

        means_stack = torch.stack(means)  # (num_subjects, D_c)
        global_mean = means_stack.mean(dim=0, keepdim=True)  # (1, D_c)

        # Cross-subject variance for this gesture class
        loss = loss + ((means_stack - global_mean) ** 2).mean()
        valid_classes += 1

    if valid_classes > 0:
        loss = loss / valid_classes

    return loss


# ─────────────────────────── model ──────────────────────────────────────────

class CausalECAPATDNN(nn.Module):
    """
    ECAPA-TDNN with Causal Content/Style Disentanglement.

    Architecture
    ────────────
    Encoder (shared):
      1. Initial TDNN     : Conv1d(in_ch, C, k=5) + BN + ReLU
      2. SERes2NetBlock ×3 : dilations [2, 3, 4], scale=4
      3. MFA aggregation   : cat([blk1, blk2, blk3]) → Conv1d(3C, 3C, 1) + BN + ReLU
      4. Attentive stats   : (B, 3C, T) → (B, 6C)
      5. FC embedding      : Linear(6C, E) + BN + ReLU + Dropout

    Content branch (causal / gesture-relevant):
      6. Content head      : Linear(E, D_c) + BN + ReLU
      7. Gesture classifier: Linear(D_c, D_c//2) + ReLU + Dropout + Linear(D_c//2, G)

    Style branch (spurious / subject-specific):
      8. Style head        : Linear(E, D_s) + BN + ReLU
      9. Subject classifier: Linear(D_s, D_s//2) + ReLU + Dropout + Linear(D_s//2, S)

    Reconstruction:
     10. Decoder           : Linear(D_c + D_s, E) + ReLU + Linear(E, E)

    LOSO safety:
      - BatchNorm running stats computed from training subjects only.
      - model.eval() freezes BN → no test-subject adaptation.
      - Inference uses only content branch (return_all=False).

    Args:
        in_channels:   Number of EMG input channels (e.g. 8).
        num_gestures:  Number of gesture classes.
        num_subjects:  Number of training subjects (for subject classifier).
        channels:      C — ECAPA internal feature dimension.
        scale:         Res2Net scale / sub-groups.
        embedding_dim: E — pre-head embedding dimension.
        content_dim:   D_c — content latent dimension.
        style_dim:     D_s — style latent dimension.
        dilations:     Dilation per SERes2NetBlock.
        dropout:       Dropout probability.
        se_reduction:  SE bottleneck reduction factor.
    """

    def __init__(
        self,
        in_channels: int,
        num_gestures: int,
        num_subjects: int,
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        content_dim: int = 128,
        style_dim: int = 64,
        dilations: list = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
    ):
        super().__init__()

        if dilations is None:
            dilations = [2, 3, 4]

        self.num_gestures = num_gestures
        self.num_subjects = num_subjects
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.embedding_dim = embedding_dim
        num_blocks = len(dilations)

        # ── 1. Initial TDNN ─────────────────────────────────────────────
        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # ── 2–4. SE-Res2Net blocks ─────────────────────────────────────
        self.blocks = nn.ModuleList([
            SERes2NetBlock(
                channels,
                kernel_size=3,
                dilation=d,
                scale=scale,
                se_reduction=se_reduction,
            )
            for d in dilations
        ])

        # ── 5. Multi-layer Feature Aggregation ─────────────────────────
        mfa_channels = channels * num_blocks
        self.mfa_conv = nn.Sequential(
            nn.Conv1d(mfa_channels, mfa_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_channels),
            nn.ReLU(inplace=True),
        )

        # ── 6. Attentive Statistics Pooling ────────────────────────────
        self.asp = AttentiveStatisticsPooling(mfa_channels)

        # ── 7. FC embedding ────────────────────────────────────────────
        asp_out_dim = mfa_channels * 2  # mean + std concatenation
        self.fc_embedding = nn.Sequential(
            nn.Linear(asp_out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── Content branch (causal / gesture-relevant) ─────────────────
        self.content_head = nn.Sequential(
            nn.Linear(embedding_dim, content_dim, bias=False),
            nn.BatchNorm1d(content_dim),
            nn.ReLU(inplace=True),
        )

        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # ── Style branch (spurious / subject-specific) ─────────────────
        self.style_head = nn.Sequential(
            nn.Linear(embedding_dim, style_dim, bias=False),
            nn.BatchNorm1d(style_dim),
            nn.ReLU(inplace=True),
        )

        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(style_dim // 2, num_subjects),
        )

        # ── Reconstruction decoder ─────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(content_dim + style_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """He-uniform for conv/linear; constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input through shared ECAPA-TDNN backbone.

        Args:
            x: (B, C_in, T) input tensor.
        Returns:
            embedding: (B, E) shared embedding.
        """
        out = self.init_tdnn(x)  # (B, C, T)

        block_outputs = []
        for block in self.blocks:
            out = block(out)  # (B, C, T)
            block_outputs.append(out)

        # Multi-layer Feature Aggregation
        mfa = torch.cat(block_outputs, dim=1)  # (B, 3C, T)
        mfa = self.mfa_conv(mfa)  # (B, 3C, T)

        # Attentive Statistics Pooling
        pooled = self.asp(mfa)  # (B, 6C)

        # FC embedding
        embedding = self.fc_embedding(pooled)  # (B, E)

        return embedding

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> Union[Dict, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C_in, T) input EMG windows.
            return_all: If True, return all branches and losses (training).
                        If False, return only gesture logits (inference).

        Returns:
            If return_all=False:
                gesture_logits: (B, num_gestures)
            If return_all=True:
                dict with keys:
                    gesture_logits: (B, G)
                    subject_logits: (B, S)
                    z_content:      (B, D_c)
                    z_style:        (B, D_s)
                    embedding:      (B, E)  — detached (reconstruction target)
                    reconstruction: (B, E)  — decoded from content + style
        """
        embedding = self.encode(x)  # (B, E)

        z_content = self.content_head(embedding)  # (B, D_c)
        gesture_logits = self.gesture_classifier(z_content)  # (B, G)

        if not return_all:
            return gesture_logits

        z_style = self.style_head(embedding)  # (B, D_s)
        subject_logits = self.subject_classifier(z_style)  # (B, S)

        # Reconstruction: content + style → embedding
        # Target is detached: encoder learns from classification + causal losses,
        # decoder + heads learn from reconstruction loss.
        z_combined = torch.cat([z_content, z_style], dim=1)  # (B, D_c + D_s)
        reconstruction = self.decoder(z_combined)  # (B, E)

        return {
            "gesture_logits": gesture_logits,
            "subject_logits": subject_logits,
            "z_content": z_content,
            "z_style": z_style,
            "embedding": embedding.detach(),
            "reconstruction": reconstruction,
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
