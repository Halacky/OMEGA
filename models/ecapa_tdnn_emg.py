"""
ECAPA-TDNN adapted for EMG gesture recognition (LOSO cross-subject setting).

Based on:
  "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
  in TDNN Based Speaker Verification" (Desplanques et al., Interspeech 2020)

Key ideas transferred from speaker verification to EMG:
  - TDNN (Time-Delay Neural Network) = dilated 1D convolution  — captures
    multi-scale temporal context without pooling (preserves resolution).
  - Res2Net branching  — hierarchical, multi-granularity feature reuse inside
    each block; multiple dilation contexts computed cheaply.
  - SE (Squeeze-and-Excitation) attention  — channel recalibration; lets the
    network suppress irrelevant EMG channels while amplifying discriminative ones.
  - Attentive Statistics Pooling (ASP)  — soft-selects the most informative
    temporal frames; outputs both weighted mean AND weighted std, so the
    classifier sees temporal spread, not just average activity.
  - Multi-layer Feature Aggregation (MFA)  — concatenates outputs of all
    SE-Res2Net blocks before pooling; model can pick whichever temporal scale
    is most useful for each gesture class.

Why it may outperform CNN-GRU for cross-subject EMG:
  - ASP is permutation-invariant and robust to local temporal distortions
    (electrode shift, varying gesture speed between subjects).
  - SE channel attention can learn to down-weight noisy channels rather than
    relying on fixed spatial filters.
  - Res2Net's multi-scale context matches EMG's multi-timescale dynamics
    (motor unit firing ~5-50 ms, co-contraction envelopes ~100-500 ms).

Input format:  (B, C_emg, T)   —  channels-first, matching PyTorch Conv1d
Output format: (B, num_classes)

Parameter count (default C=128, scale=4, embed=128, 8 EMG channels):
  ≈ 467 K  — comparable to CNNGRUWithAttention (≈ 524 K).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────── primitives ──────────────────────────────────

class Res2NetBlock(nn.Module):
    """
    Res2Net temporal block (hierarchical multi-scale convolution).

    Splits the C-channel feature map into `scale` equal sub-groups.
    The first sub-group is an identity path (no parameters).
    Each subsequent sub-group i applies a dilated Conv1d to the sum of
    the current group's features and the previous group's output (Res2
    shortcut), enabling hierarchical feature reuse at multiple scales.

    Args:
        C:           total channel width (must be divisible by `scale`)
        kernel_size: kernel size of each per-group dilated conv (default 3)
        dilation:    dilation factor of each per-group conv
        scale:       number of sub-groups / feature scales (default 4)
    """

    def __init__(self, C: int, kernel_size: int, dilation: int, scale: int = 4):
        super().__init__()
        assert C % scale == 0, (
            f"Channel width C={C} must be divisible by scale={scale}"
        )
        self.scale = scale
        K = C // scale   # width of each sub-group

        # (scale - 1) branches with learnable dilated conv + BN
        # The first branch is always the identity — no parameters needed.
        # padding = dilation*(kernel_size-1)//2  ensures same-length output.
        p = dilation * (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            nn.Conv1d(K, K, kernel_size=kernel_size,
                      dilation=dilation, padding=p, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(K) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        chunks = torch.chunk(x, self.scale, dim=1)  # list of (B, K, T)
        out: List[torch.Tensor] = [chunks[0]]        # first branch: identity

        y: Optional[torch.Tensor] = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            xi = chunks[i + 1]
            if y is not None:
                xi = xi + y          # Res2 shortcut from previous branch
            y = F.relu(bn(conv(xi)), inplace=True)
            out.append(y)

        return torch.cat(out, dim=1)  # (B, C, T) — same shape as input


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Computes a per-channel attention weight via global average pooling
    followed by a two-layer bottleneck MLP with Sigmoid activation.
    The input feature map is then element-wise scaled by these weights,
    allowing the network to suppress noisy EMG channels adaptively.

    Args:
        C:         number of input/output channels
        reduction: bottleneck reduction factor (mid = C // reduction)
    """

    def __init__(self, C: int, reduction: int = 8):
        super().__init__()
        mid = max(C // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(C, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, C, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=2)                   # (B, C)  — global avg pool over time
        s = self.fc(s).unsqueeze(2)         # (B, C, 1)
        return x * s                        # channel-wise rescaling, broadcast T


class SERes2NetBlock(nn.Module):
    """
    SE-Res2Net TDNN block  — the core building block of ECAPA-TDNN.

    Flow:
      input
        → 1×1 Conv + BN + ReLU          (pointwise mixing)
        → Res2Net(dilated, scale)        (multi-scale temporal context)
        → 1×1 Conv + BN                  (recombine scale branches)
        → SE attention                   (channel recalibration)
        → residual-add input             (skip connection)
        → ReLU
      output

    The residual connection (+ ReLU after addition) is the standard
    post-activation ResNet shortcut; it keeps gradient flow stable
    across multiple stacked blocks.

    Args:
        C:           channel width (same for input and output)
        kernel_size: per-group dilated conv kernel size in Res2Net
        dilation:    dilation factor for the Res2Net convolutions
        scale:       Res2Net scale (number of feature sub-groups)
        se_reduction: SE bottleneck reduction factor
    """

    def __init__(
        self,
        C: int,
        kernel_size: int,
        dilation: int,
        scale: int = 4,
        se_reduction: int = 8,
    ):
        super().__init__()

        # 1×1 pointwise conv before Res2Net: mixes channels, adds capacity
        self.pointwise_in = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )

        # Res2Net multi-scale temporal block
        self.res2 = Res2NetBlock(C, kernel_size=kernel_size,
                                  dilation=dilation, scale=scale)

        # 1×1 pointwise conv after Res2Net: recombine all scale branches.
        # No ReLU here — activation is applied AFTER adding residual.
        self.pointwise_out = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm1d(C),
        )

        # SE channel recalibration
        self.se = SEBlock(C, reduction=se_reduction)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        out = self.pointwise_in(x)      # (B, C, T)
        out = self.res2(out)            # (B, C, T)
        out = self.pointwise_out(out)   # (B, C, T)
        out = self.se(out)              # (B, C, T)  — channel attention
        out = self.relu(out + residual) # (B, C, T)  — residual + activation
        return out


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP).

    Collapses the time dimension (B, C, T) → (B, 2C) by computing
    per-channel weighted mean AND weighted standard deviation.
    The per-channel attention weights are learned via a small conv MLP.

    Why both mean and std?
      The mean captures the average activation level (gesture type), while
      the std captures temporal spread (gesture dynamics / speed variation).
      Together they are more discriminative than mean-pooling alone and more
      robust to inter-subject speed differences than sequence-level models.

    LOSO integrity: attention weights are computed purely from the input
    features at inference time — no running statistics, no stored state.

    Args:
        C: number of input channels (output will be 2*C)
    """

    def __init__(self, C: int):
        super().__init__()
        # Channel-wise attention: (B, C, T) → (B, C, T) scores
        # Using two 1×1 convs with Tanh nonlinearity (bounded scores)
        hidden = max(C // 4, 16)
        self.attn = nn.Sequential(
            nn.Conv1d(C, hidden, kernel_size=1, bias=False),
            nn.Tanh(),
            nn.Conv1d(hidden, C, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        alpha = self.attn(x)                     # (B, C, T) — raw scores
        alpha = F.softmax(alpha, dim=2)          # (B, C, T) — weights, sum_T=1

        mu = (alpha * x).sum(dim=2)              # (B, C) — weighted mean

        # Weighted variance via E[X²] - (E[X])²  (numerically stable since ≥ 0)
        var = (alpha * x.pow(2)).sum(dim=2) - mu.pow(2)
        sigma = var.clamp(min=1e-8).sqrt()       # (B, C) — weighted std

        return torch.cat([mu, sigma], dim=1)     # (B, 2C)


# ──────────────────────────── full ECAPA-TDNN model ──────────────────────────

class ECAPATDNNEmg(nn.Module):
    """
    ECAPA-TDNN encoder + classifier for EMG gesture recognition.

    Architecture
    ────────────
    1. Initial TDNN        : Conv1d(in_ch, C, k=5) + BN + ReLU
                             Broad receptive field to capture inter-channel
                             dependencies early.
    2. SE-Res2Net block ×3 : (C, k=3, dilation=dilations[i])
                             Stacked blocks with increasing dilation build
                             exponentially growing temporal context.
    3. MFA aggregation     : cat([block_1, block_2, block_3]) → Conv1d + BN + ReLU
                             Multi-layer Feature Aggregation: final repr sees
                             features from ALL temporal scales simultaneously.
    4. Attentive stats pool: (B, 3C, T) → (B, 6C)
    5. FC embedding        : Linear(6C, E) + BN + ReLU + Dropout
    6. Classifier          : Linear(E, num_classes)

    Parameter budget (default C=128, scale=4, embed=128, 8 EMG channels):
      ≈ 467 K  — close to CNNGRUWithAttention (≈ 524 K, same 8 ch input).

    LOSO / data-leakage safety:
      - No per-subject normalization inside the model.
      - BatchNorm running stats are computed from training subjects only
        (model is called with eval() during test-subject inference).
      - AttentiveStatisticsPooling has no stored state — purely input-driven.

    Args:
        in_channels:   Number of EMG input channels (e.g. 8).
        num_classes:   Number of gesture classes.
        channels:      C — internal feature dimension (default 128).
        scale:         Res2Net scale / number of sub-groups (default 4).
        embedding_dim: E — pre-classifier embedding dimension (default 128).
        dilations:     Dilation per SE-Res2Net block (default [2, 3, 4]).
        dropout:       Dropout probability before classifier (default 0.3).
        se_reduction:  SE bottleneck reduction factor (default 8).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
    ):
        super().__init__()

        if dilations is None:
            dilations = [2, 3, 4]
        if len(dilations) != 3:
            raise ValueError(
                f"Exactly 3 dilation values expected (one per block), "
                f"got {len(dilations)}: {dilations}"
            )

        self.channels      = channels
        self.embedding_dim = embedding_dim
        num_blocks         = len(dilations)       # 3

        # ── 1. Initial TDNN ────────────────────────────────────────────────
        # k=5 gives receptive field of 5 samples (2.5 ms at 2 kHz) at dilation=1.
        # Wide enough to capture single motor-unit action potential shapes.
        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # ── 2–4. SE-Res2Net blocks with increasing dilation ───────────────
        # Dilations [2, 3, 4] with k=3 give effective receptive fields of
        # 5, 7, 9 samples respectively (per block, before stacking).
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

        # ── 5. Multi-layer Feature Aggregation ────────────────────────────
        # Concatenate outputs from all blocks (B, C*num_blocks, T).
        # A 1×1 conv mixes cross-block, cross-scale information.
        mfa_in = channels * num_blocks       # 3C
        self.mfa = nn.Sequential(
            nn.Conv1d(mfa_in, mfa_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_in),
            nn.ReLU(inplace=True),
        )

        # ── 6. Attentive Statistics Pooling ───────────────────────────────
        # Input: (B, 3C, T)   Output: (B, 6C)
        self.asp = AttentiveStatisticsPooling(mfa_in)

        # ── 7. FC embedding ───────────────────────────────────────────────
        asp_out_dim = mfa_in * 2             # 6C
        self.embedding = nn.Sequential(
            nn.Linear(asp_out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── 8. Classifier ─────────────────────────────────────────────────
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────
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

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_emg, T) — EMG windows in channels-first format.
        Returns:
            logits: (B, num_classes)
        """
        # 1. Initial TDNN
        out = self.init_tdnn(x)                 # (B, C, T)

        # 2–4. SE-Res2Net blocks; collect all outputs for MFA
        block_outputs = []
        for block in self.blocks:
            out = block(out)                    # (B, C, T) — in-place residual
            block_outputs.append(out)

        # 5. MFA: concatenate all block outputs then mix with 1×1 conv
        mfa_in = torch.cat(block_outputs, dim=1)   # (B, 3C, T)
        mfa_out = self.mfa(mfa_in)                  # (B, 3C, T)

        # 6. Attentive statistics pooling: collapse time
        pooled = self.asp(mfa_out)                  # (B, 6C)

        # 7. Embedding
        emb = self.embedding(pooled)                # (B, E)

        # 8. Classification
        logits = self.classifier(emb)               # (B, num_classes)
        return logits

    # ── utility ───────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
