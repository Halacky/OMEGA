"""
ResNet1D for EMG gesture recognition.

Residual networks are effective for time-series: multi-scale receptive fields
via depth, residual connections improve gradient flow and generalization.
Suitable for LOSO where we need robust, generalizable features.
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """Residual block: conv -> BN -> ReLU -> conv -> BN -> (+ residual)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return torch.relu(out)


class ResNet1D(nn.Module):
    """
    1D ResNet for raw EMG windows.

    - Stem conv + pool, then 3 stages of residual blocks with optional pooling.
    - Global average pooling and classifier.
    - Designed for (batch, channels, time) input; good for cross-subject when
      combined with augmentation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        blocks_per_stage: tuple = (2, 2, 2),
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        stages = []
        ch = base_channels
        for i, n_blocks in enumerate(blocks_per_stage):
            next_ch = ch * 2 if i > 0 else ch
            for _ in range(n_blocks):
                stages.append(ResBlock1D(ch, next_ch, kernel_size=kernel_size))
                ch = next_ch
            if i < len(blocks_per_stage) - 1:
                stages.append(nn.MaxPool1d(2))
        self.stages = nn.Sequential(*stages)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
