"""
Disentangled CNN-GRU model for content-style separation in EMG gesture recognition.

Hypothesis H5: Gesture → content (causal signal), Subject → style.
Disentanglement via separate latent spaces + mutual information minimization.

Architecture:
    Input (B, C, T) → SharedEncoder (CNN+BiGRU+Attention) → (B, 256)
      ├─ ContentHead → z_content (B, 128) → GestureClassifier → gesture_logits
      └─ StyleHead   → z_style   (B, 64)  → SubjectClassifier → subject_logits

Loss:
    L_total = L_gesture + α * L_subject + β * L_MI(z_content, z_style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── Loss functions ────────────────────────────


def distance_correlation_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Distance correlation between z1 and z2.

    Measures all (including nonlinear) statistical dependencies.
    Returns scalar in [0, 1]; 0 iff z1 ⊥ z2.

    Complexity: O(B^2 * max(D1, D2)).
    """
    n = z1.size(0)
    if n < 4:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    # Pairwise Euclidean distance matrices
    a = torch.cdist(z1, z1, p=2)  # (B, B)
    b = torch.cdist(z2, z2, p=2)  # (B, B)

    # Double centering
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


def orthogonality_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Penalize linear correlation between z_content and z_style.

    Computes mean squared cross-correlation after column-wise L2 normalization.
    """
    z1_norm = F.normalize(z1, dim=0)  # (B, D1)
    z2_norm = F.normalize(z2, dim=0)  # (B, D2)
    cross = z1_norm.T @ z2_norm       # (D1, D2)
    return (cross ** 2).mean()


# ────────────────────────── Model components ───────────────────────────


class SharedEncoder(nn.Module):
    """CNN → BiGRU → Attention pooling.  Mirrors CNNGRUWithAttention."""

    def __init__(
        self,
        in_channels: int = 8,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        # Build CNN blocks
        layers = []
        prev_ch = in_channels
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

        # BiGRU
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        self.gru_output_dim = gru_hidden * 2  # bidirectional

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(self.gru_output_dim, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, gru_hidden * 2) context vector
        """
        h = self.cnn(x)               # (B, cnn[-1], T')
        h = h.transpose(1, 2)         # (B, T', cnn[-1])
        gru_out, _ = self.gru(h)      # (B, T', gru_hidden*2)

        attn_w = self.attention(gru_out)           # (B, T', 1)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (attn_w * gru_out).sum(dim=1)    # (B, gru_hidden*2)
        return context


class ProjectionHead(nn.Module):
    """Two-layer MLP projecting shared features to a latent subspace."""

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────── Main model ───────────────────────────────


class DisentangledCNNGRU(nn.Module):
    """
    Content-Style Disentangled CNN-GRU for subject-invariant gesture recognition.

    During training:  returns dict with gesture_logits, subject_logits, z_content, z_style.
    During eval:      returns gesture_logits tensor only.
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
        shared_dim = self.encoder.gru_output_dim  # 256

        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head = ProjectionHead(shared_dim, style_dim, dropout)

        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x:          (B, C, T)
            return_all: force returning full dict even in eval mode
        Returns:
            training / return_all:
                dict with gesture_logits, subject_logits, z_content, z_style
            eval (default):
                gesture_logits (B, num_gestures)
        """
        shared = self.encoder(x)                        # (B, 256)
        z_content = self.content_head(shared)            # (B, content_dim)
        z_style = self.style_head(shared)                # (B, style_dim)

        gesture_logits = self.gesture_classifier(z_content)

        if self.training or return_all:
            subject_logits = self.subject_classifier(z_style)
            return {
                "gesture_logits": gesture_logits,
                "subject_logits": subject_logits,
                "z_content": z_content,
                "z_style": z_style,
            }
        return gesture_logits
