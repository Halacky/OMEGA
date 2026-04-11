"""
MFCC CNN with MixStyle domain generalization for cross-subject EMG.

Extends MFCCCNNClassifier by injecting MixStyle layers after early CNN blocks.
MixStyle (Zhou et al., ICLR 2021) randomly mixes instance-level feature
statistics (mean, std) between training samples — this breaks subject-specific
style while preserving gesture content.

Key design choices:
  - MixStyle is applied AFTER the first CNN block (early features carry most
    subject-specific style information — electrode impedance, skin properties).
  - Optionally applied after the second block too (dual injection).
  - Zero additional parameters — MixStyle is purely a training-time operation.
  - Disabled at eval() time — no effect on test inference.

LOSO integrity: training batches contain only training subjects. MixStyle
mixes within the batch, so no test-subject information can leak.
"""

import torch
import torch.nn as nn

from models.mfcc_cnn_classifier import DepthwiseSeparableConv2d


class MixStyle2d(nn.Module):
    """
    MixStyle for 2D feature maps (Zhou et al., ICLR 2021).

    Mixes channel-wise statistics (μ, σ) between random pairs in the batch.
    Applied only during training; identity at eval time.

    Args:
        p:     probability of applying per forward pass.
        alpha: Beta distribution concentration parameter.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.shape

        # Per-instance channel statistics over spatial dims
        mu = x.mean(dim=[2, 3], keepdim=True)       # (B, C, 1, 1)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + 1e-6).sqrt()                     # (B, C, 1, 1)

        # Normalize
        x_norm = (x - mu) / sig

        # Mix with random batch permutation
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B,))
        lam = lam.to(x.device).view(B, 1, 1, 1)

        perm = torch.randperm(B, device=x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]

        return x_norm * sig_mix + mu_mix


class MFCCMixStyleCNN(nn.Module):
    """
    MFCC 2D CNN with MixStyle injection for domain generalization.

    Architecture:
        Input (B, C, n_coeff, n_frames)
          → Conv2d block 0 (C → 32)
          → MixStyle  ← subject style mixing
          → MaxPool + Dropout
          → DepthwiseSep block 1 (32 → 64)
          → [optional MixStyle]
          → MaxPool + Dropout
          → DepthwiseSep block 2 (64 → 128)
          → MaxPool + Dropout
          → GAP → MLP → logits

    Args:
        in_channels:    C — EMG channels.
        n_coeff:        MFCC coefficients.
        n_frames:       Time frames.
        num_classes:    Gesture classes.
        cnn_channels:   Channel sizes per block.
        dropout:        Dropout rate.
        mixstyle_p:     MixStyle probability.
        mixstyle_alpha: MixStyle Beta concentration.
        mixstyle_layers: Which layers get MixStyle (e.g. [0], [0,1]).
    """

    def __init__(
        self,
        in_channels: int = 8,
        n_coeff: int = 39,
        n_frames: int = 19,
        num_classes: int = 10,
        cnn_channels: list = None,
        dropout: float = 0.3,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        mixstyle_layers: list = None,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        if mixstyle_layers is None:
            mixstyle_layers = [0]  # default: after first block only

        self.in_channels = in_channels
        self.n_coeff = n_coeff
        self.n_frames = n_frames
        self.mixstyle_layers = set(mixstyle_layers)

        # Build CNN blocks
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.mixstyles = nn.ModuleDict()

        prev_ch = in_channels
        for i, out_ch in enumerate(cnn_channels):
            if i == 0:
                block = nn.Sequential(
                    nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            else:
                block = DepthwiseSeparableConv2d(prev_ch, out_ch)

            self.blocks.append(block)
            self.pools.append(nn.Sequential(
                nn.MaxPool2d(2, ceil_mode=True),
                nn.Dropout2d(dropout * 0.5),
            ))

            if i in self.mixstyle_layers:
                self.mixstyles[str(i)] = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)

            prev_ch = out_ch

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, n_coeff, n_frames)
        Returns:
            logits: (B, num_classes)
        """
        h = x
        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            h = block(h)
            # Inject MixStyle after this block (before pooling)
            if str(i) in self.mixstyles:
                h = self.mixstyles[str(i)](h)
            h = pool(h)

        h = self.pool(h).flatten(1)
        return self.classifier(h)
