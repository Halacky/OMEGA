"""
Xception-style CNN for EMG spectrograms (MFCC / MDCT / Fbanks).

Adapted from Chollet, "Xception: Deep Learning with Depthwise Separable
Convolutions," CVPR 2017 — but scaled down for small EMG spectrogram inputs
(8×39×18 for MFCC, 8×75×15 for MDCT).

Key differences from original Xception (299×299×3 ImageNet):
  - 3 blocks instead of 14 (input is 100-1000x smaller)
  - Fewer channels (32→64→128 vs 128→256→728)
  - Residual 1×1 projections at every block (critical for small inputs)
  - No middle flow repetition (would overfit on ~40K training windows)
  - Squeeze-and-Excitation (SE) channel attention after each block

Architecture:
    Input (B, C, H, W) — C=EMG channels, H=coefficients, W=time frames

    Entry flow:
      Block 0: Conv2d(C→32, 3×3) + BN + ReLU + residual
      Block 1: SepConv(32→64) + SepConv(64→64) + residual(1×1 proj) + SE
      Block 2: SepConv(64→128) + SepConv(128→128) + residual(1×1 proj) + SE

    Exit flow:
      Block 3: SepConv(128→256) + SepConv(256→256) + residual(1×1 proj)
      GAP → Dropout → Linear(256→128) → ReLU → Dropout → Linear(128→num_classes)

LOSO integrity: all parameters trained on training subjects only, frozen at test.
"""

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_relu: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class XceptionBlock(nn.Module):
    """
    Xception residual block: two separable convolutions + skip connection.

    If in_ch != out_ch or stride > 1, a 1×1 projection is used for the residual.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, use_se: bool = True):
        super().__init__()
        self.sep1 = SeparableConv2d(in_ch, out_ch, use_relu=True)
        self.sep2 = SeparableConv2d(out_ch, out_ch, use_relu=False)
        self.pool = nn.MaxPool2d(stride, ceil_mode=True) if stride > 1 else nn.Identity()

        # Residual projection
        if in_ch != out_ch or stride > 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.MaxPool2d(stride, ceil_mode=True) if stride > 1 else nn.Identity(),
            )
        else:
            self.residual = nn.Identity()

        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        h = self.sep1(x)
        h = self.sep2(h)
        h = self.pool(h)
        h = self.relu(h + res)
        h = self.se(h)
        return h


class XceptionEMG(nn.Module):
    """
    Xception-style classifier for EMG spectrograms.

    Args:
        in_channels:   C — number of EMG channels (8 for NinaPro DB2).
        n_coeff:       H — spectrogram coefficient dimension.
        n_frames:      W — spectrogram time frames.
        num_classes:   number of gesture classes.
        channels:      channel sizes for each block [entry0, entry1, entry2, exit].
        dropout:       dropout rate for classifier head.
        use_se:        use Squeeze-and-Excitation attention.
    """

    def __init__(
        self,
        in_channels: int = 8,
        n_coeff: int = 39,
        n_frames: int = 18,
        num_classes: int = 10,
        channels: list = None,
        dropout: float = 0.3,
        use_se: bool = True,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.n_coeff = n_coeff
        self.n_frames = n_frames

        # Entry: standard conv to expand from C EMG channels
        self.entry_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Xception blocks with residual connections
        self.block1 = XceptionBlock(channels[0], channels[1], stride=2, use_se=use_se)
        self.block2 = XceptionBlock(channels[1], channels[2], stride=2, use_se=use_se)
        self.block3 = XceptionBlock(channels[2], channels[3], stride=2, use_se=False)

        # Global average pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[3], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, n_coeff, n_frames) — EMG spectrogram
        Returns:
            logits: (B, num_classes)
        """
        h = self.entry_conv(x)   # (B, 32, H, W)
        h = self.block1(h)       # (B, 64, H/2, W/2)
        h = self.block2(h)       # (B, 128, H/4, W/4)
        h = self.block3(h)       # (B, 256, H/8, W/8)
        h = self.pool(h)         # (B, 256, 1, 1)
        h = h.flatten(1)         # (B, 256)
        return self.classifier(h)
