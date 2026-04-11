"""
Channel-wise Contrastive Disentanglement + Per-Channel Worst-Case model.

Hypothesis H9: Per-channel disentanglement instead of global (exp_31).

Architecture:
    Input (B, C, T) → PerChannelEncoder (C independent 1-ch CNNs) → C × (B, D)
      ├─ Per-channel ContentHead → z_content_c (B, content_dim_per_ch)  × C
      ├─ Per-channel StyleHead   → z_style_c   (B, style_dim_per_ch)   × C
      └─ ChannelAttention (SE-style) on z_content_1..C → weighted fusion
    → concat(weighted z_content_c) → GRU → attention → GestureClassifier

Losses:
    L_gesture (per-channel GroupDRO)
    L_contrastive (InfoNCE per channel on z_content)
    L_MI (distance correlation per channel between z_content_c and z_style_c)
    L_subject (subject classifier on concat z_style)

Differences from exp_31:
    - Per-channel disentanglement vs global
    - Contrastive InfoNCE loss vs distance correlation only
    - Per-channel worst-case (GroupDRO) vs ERM
    - Channel importance attention (SE-block) vs equal weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── Loss functions ────────────────────────────


def info_nce_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss on z_content representations.

    Positive pairs: same gesture class (from different subjects).
    Negative pairs: different gesture classes.

    Args:
        z: (B, D) content representations
        labels: (B,) gesture class labels
        temperature: softmax temperature (lower = sharper)

    Returns:
        Scalar loss.
    """
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    # L2 normalize
    z_norm = F.normalize(z, dim=1)  # (B, D)

    # Cosine similarity matrix
    sim = z_norm @ z_norm.T / temperature  # (B, B)

    # Mask: positive pairs = same gesture label, exclude self
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = labels_eq & self_mask  # (B, B)

    # If no positive pairs exist, return 0
    if not pos_mask.any():
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    # Log-sum-exp over all negatives (all non-self entries)
    # For numerical stability, subtract max
    logits_max, _ = sim.detach().max(dim=1, keepdim=True)
    logits = sim - logits_max  # (B, B)

    # Denominator: sum over all non-self
    exp_logits = torch.exp(logits) * self_mask.float()  # (B, B)
    log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)  # (B, 1)

    # Log probability of positives
    log_prob = logits - log_denom  # (B, B)

    # Mean over positive pairs per anchor, then mean over anchors that have positives
    pos_per_anchor = pos_mask.float().sum(dim=1)  # (B,)
    has_pos = pos_per_anchor > 0

    if not has_pos.any():
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / (pos_per_anchor + 1e-12)
    loss = -mean_log_prob[has_pos].mean()

    return loss


def distance_correlation_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Distance correlation between z1 and z2. Returns scalar in [0, 1]."""
    n = z1.size(0)
    if n < 4:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    a = torch.cdist(z1, z1, p=2)
    b = torch.cdist(z2, z2, p=2)

    a_row = a.mean(dim=1, keepdim=True)
    a_col = a.mean(dim=0, keepdim=True)
    a_grand = a.mean()
    A = a - a_row - a_col + a_grand

    b_row = b.mean(dim=1, keepdim=True)
    b_col = b.mean(dim=0, keepdim=True)
    b_grand = b.mean()
    B_ = b - b_row - b_col + b_grand

    dcov2 = (A * B_).mean()
    dvar_a2 = (A * A).mean()
    dvar_b2 = (B_ * B_).mean()

    denom = torch.sqrt(dvar_a2 * dvar_b2 + 1e-12)
    dcor = torch.sqrt(torch.clamp(dcov2, min=0.0) / denom)
    return dcor


# ────────────────────────── Model components ───────────────────────────


class PerChannelCNN(nn.Module):
    """Single-channel 1D CNN encoder, applied independently to each EMG channel."""

    def __init__(
        self,
        cnn_channels: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64]

        layers = []
        prev_ch = 1  # single input channel
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.5),
            ])
            prev_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.output_dim = cnn_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) single-channel signal
        Returns:
            (B, output_dim) — global average pooled representation
        """
        h = self.cnn(x)  # (B, output_dim, T')
        return h.mean(dim=2)  # (B, output_dim) global average pooling


class PerChannelEncoder(nn.Module):
    """
    Applies C independent PerChannelCNN encoders to each EMG channel.

    Input: (B, C, T)
    Output: (B, C, per_ch_dim)
    """

    def __init__(
        self,
        num_channels: int = 8,
        cnn_channels: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.encoders = nn.ModuleList([
            PerChannelCNN(cnn_channels=cnn_channels, dropout=dropout)
            for _ in range(num_channels)
        ])
        self.per_ch_dim = self.encoders[0].output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, per_ch_dim)
        """
        ch_reps = []
        for c in range(self.num_channels):
            x_c = x[:, c:c+1, :]  # (B, 1, T)
            ch_reps.append(self.encoders[c](x_c))  # (B, per_ch_dim)
        return torch.stack(ch_reps, dim=1)  # (B, C, per_ch_dim)


class PerChannelProjectionHead(nn.Module):
    """
    Per-channel content/style projection heads.
    Applies C independent MLPs to produce per-channel latent vectors.
    """

    def __init__(
        self,
        num_channels: int,
        input_dim: int,
        latent_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, latent_dim * 2),
                nn.BatchNorm1d(latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim * 2, latent_dim),
            )
            for _ in range(num_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, input_dim)
        Returns:
            (B, C, latent_dim)
        """
        out = []
        for c in range(self.num_channels):
            out.append(self.heads[c](x[:, c]))  # (B, latent_dim)
        return torch.stack(out, dim=1)  # (B, C, latent_dim)


class ChannelAttention(nn.Module):
    """
    SE-block-style channel attention on per-channel content representations.

    Input: (B, C, content_dim)
    Output: (B, C, content_dim) — reweighted by learned channel importance
    Also returns attention weights (B, C) for analysis.
    """

    def __init__(self, num_channels: int, content_dim: int, reduction: int = 4):
        super().__init__()
        mid = max(num_channels // reduction, 2)
        self.fc = nn.Sequential(
            nn.Linear(num_channels * content_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, z_content: torch.Tensor):
        """
        Args:
            z_content: (B, C, content_dim)
        Returns:
            weighted: (B, C, content_dim)
            weights: (B, C)
        """
        B, C, D = z_content.shape
        flat = z_content.reshape(B, C * D)  # (B, C*D)
        weights = self.fc(flat)  # (B, C)
        weighted = z_content * weights.unsqueeze(2)  # (B, C, D)
        return weighted, weights


class CrossChannelFusion(nn.Module):
    """
    Fuse per-channel content representations via GRU + attention pooling.

    Input: (B, C, content_dim)
    Output: (B, fusion_dim)
    """

    def __init__(
        self,
        content_dim: int,
        num_channels: int,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=content_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        self.fusion_dim = gru_hidden * 2

        # Attention pooling over channel dimension
        self.attention = nn.Sequential(
            nn.Linear(self.fusion_dim, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )

    def forward(self, z_content_weighted: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_content_weighted: (B, C, content_dim)
        Returns:
            (B, fusion_dim) fused representation
        """
        gru_out, _ = self.gru(z_content_weighted)  # (B, C, fusion_dim)
        attn_w = self.attention(gru_out)  # (B, C, 1)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (attn_w * gru_out).sum(dim=1)  # (B, fusion_dim)
        return context


# ──────────────────────────── Main model ───────────────────────────────


class ChannelContrastiveDisentangled(nn.Module):
    """
    Channel-wise Contrastive Disentanglement model.

    Per-channel independent encoders → per-channel content/style split →
    channel attention → GRU fusion → gesture classifier.

    Training returns dict with all intermediate representations for loss computation.
    Eval returns gesture_logits only.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_gestures: int = 10,
        num_subjects: int = 4,
        content_dim_per_ch: int = 32,
        style_dim_per_ch: int = 16,
        per_ch_cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.content_dim_per_ch = content_dim_per_ch
        self.style_dim_per_ch = style_dim_per_ch

        if per_ch_cnn_channels is None:
            per_ch_cnn_channels = [32, 64]

        # Per-channel encoder
        self.encoder = PerChannelEncoder(
            num_channels=in_channels,
            cnn_channels=per_ch_cnn_channels,
            dropout=dropout,
        )
        per_ch_enc_dim = self.encoder.per_ch_dim

        # Per-channel content and style heads
        self.content_heads = PerChannelProjectionHead(
            num_channels=in_channels,
            input_dim=per_ch_enc_dim,
            latent_dim=content_dim_per_ch,
            dropout=dropout,
        )
        self.style_heads = PerChannelProjectionHead(
            num_channels=in_channels,
            input_dim=per_ch_enc_dim,
            latent_dim=style_dim_per_ch,
            dropout=dropout,
        )

        # Channel attention (SE-block style)
        self.channel_attention = ChannelAttention(
            num_channels=in_channels,
            content_dim=content_dim_per_ch,
        )

        # Cross-channel fusion (GRU + attention)
        self.fusion = CrossChannelFusion(
            content_dim=content_dim_per_ch,
            num_channels=in_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        fusion_dim = self.fusion.fusion_dim  # gru_hidden * 2

        # Gesture classifier (on fused content)
        self.gesture_classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_gestures),
        )

        # Subject classifier (on concatenated style vectors)
        total_style_dim = in_channels * style_dim_per_ch
        self.subject_classifier = nn.Sequential(
            nn.Linear(total_style_dim, total_style_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_style_dim // 2, num_subjects),
        )

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x: (B, C, T)
            return_all: force dict output even in eval mode

        Returns:
            Training / return_all:
                dict with:
                    gesture_logits: (B, num_gestures)
                    subject_logits: (B, num_subjects)
                    z_content: (B, C, content_dim_per_ch) per-channel content
                    z_style: (B, C, style_dim_per_ch) per-channel style
                    channel_weights: (B, C) attention weights
                    fused_content: (B, fusion_dim)
            Eval:
                gesture_logits: (B, num_gestures)
        """
        # Per-channel encoding
        ch_reps = self.encoder(x)  # (B, C, per_ch_enc_dim)

        # Per-channel content/style split
        z_content = self.content_heads(ch_reps)  # (B, C, content_dim_per_ch)
        z_style = self.style_heads(ch_reps)      # (B, C, style_dim_per_ch)

        # Channel attention on content
        z_content_weighted, channel_weights = self.channel_attention(z_content)

        # Cross-channel fusion
        fused = self.fusion(z_content_weighted)  # (B, fusion_dim)

        # Gesture classification
        gesture_logits = self.gesture_classifier(fused)

        if self.training or return_all:
            # Subject classification from concatenated style
            B = x.size(0)
            z_style_flat = z_style.reshape(B, -1)  # (B, C * style_dim_per_ch)
            subject_logits = self.subject_classifier(z_style_flat)

            return {
                "gesture_logits": gesture_logits,
                "subject_logits": subject_logits,
                "z_content": z_content,
                "z_style": z_style,
                "channel_weights": channel_weights,
                "fused_content": fused,
            }

        return gesture_logits
