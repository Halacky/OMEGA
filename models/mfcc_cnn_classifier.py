"""
CNN classifier operating on MFCC spectrograms for EMG gesture recognition.

Architecture:
    Input: (B, n_coeff, T_frames, C) — MFCC spectrogram per channel
      → Reshape to (B, C, n_coeff, T_frames) — treat channels as "batch channels"
      → 2D CNN with depthwise-separable convolutions (Xception-style)
      → Global Average Pooling
      → MLP classifier → (B, num_classes)

The depthwise-separable design processes each EMG channel independently in
the depthwise stage, then fuses cross-channel information in the pointwise
stage.  This is natural for multi-electrode EMG where each sensor captures
slightly different muscle groups.

LOSO integrity:
    - All model parameters trained on training subjects only.
    - MFCC computation is deterministic (no learned parameters in the frontend).
    - Channel standardization applied before MFCC extraction (training stats only).
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class MFCCCNNClassifier(nn.Module):
    """
    2D CNN classifier on MFCC spectrograms.

    Args:
        in_channels:    C — number of EMG channels (electrodes).
        n_coeff:        Number of MFCC coefficients (e.g. 39 with deltas).
        n_frames:       Number of time frames in the MFCC spectrogram.
        num_classes:    Number of gesture classes.
        cnn_channels:   List of output channel sizes for conv blocks.
        dropout:        Dropout rate for classifier head.
    """

    def __init__(
        self,
        in_channels: int = 8,
        n_coeff: int = 39,
        n_frames: int = 19,
        num_classes: int = 10,
        cnn_channels: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        self.in_channels = in_channels
        self.n_coeff = n_coeff
        self.n_frames = n_frames

        # Initial conv to expand from C EMG channels
        layers = []
        prev_ch = in_channels
        for i, out_ch in enumerate(cnn_channels):
            if i == 0:
                # Standard conv for first layer (maps C → cnn_channels[0])
                layers.extend([
                    nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ])
            else:
                # Depthwise separable for subsequent layers
                layers.append(DepthwiseSeparableConv2d(prev_ch, out_ch))
            layers.append(nn.MaxPool2d(2, ceil_mode=True))
            layers.append(nn.Dropout2d(dropout * 0.5))
            prev_ch = out_ch

        self.features = nn.Sequential(*layers)

        # Global average pooling → (B, cnn_channels[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, n_coeff, n_frames) — MFCC spectrograms
               where C = EMG channels, n_coeff = MFCC coefficients,
               n_frames = time frames.

        Returns:
            logits: (B, num_classes)
        """
        h = self.features(x)          # (B, cnn[-1], H', W')
        h = self.pool(h)              # (B, cnn[-1], 1, 1)
        h = h.flatten(1)              # (B, cnn[-1])
        return self.classifier(h)     # (B, num_classes)
