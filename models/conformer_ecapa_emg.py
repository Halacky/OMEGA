"""
Conformer-ECAPA for EMG gesture recognition (LOSO cross-subject setting).

Hypothesis (exp_70)
───────────────────
ECAPA-TDNN (exp_62) is strong due to multi-scale Res2Net temporal modeling +
Attentive Statistics Pooling.  Adding lightweight Conformer blocks between the
conv stem and the ECAPA backbone explicitly captures both:
  • Local micro-patterns via depthwise convolution (k=31, ~15 ms at 2 kHz).
  • Long-range temporal dependencies via MHSA with learnable relative position
    bias — no absolute sinusoidal PE; remains robust to gesture speed variation
    across subjects (since positional biases are learned from all training subjects
    jointly and applied uniformly at test time).

Architecture
────────────
  1. Conv stem      : Conv1d(in_ch, C, k=5) + BN + ReLU
  2. N×ConformerBlock:
       [DWConvModule → MHSA + rel-pos bias → LayerNorm]  (local → global)
  3. 3×SE-Res2Net   : dilations=[2,3,4], identical to ECAPA-TDNN (exp_62)
  4. MFA            : cat(se_block₁,₂,₃) → Conv1d(3C,3C,k=1) + BN + ReLU
                      (MFA sees only SE-Res2Net outputs, not Conformer outputs,
                       preserving the ECAPA multi-scale aggregation pattern)
  5. ASP            : Attentive Statistics Pooling  (B, 3C, T) → (B, 6C)
  6. Embedding      : Linear(6C, E) + BN + ReLU + Dropout
  7. Classifier     : Linear(E, num_classes)

Key design choices
──────────────────
  Relative position bias:
    Learned table of shape (2*max_rel_dist+1, num_heads), indexed by clipped
    signed relative distance (i−j).  Added to attention logits before softmax
    via attn_mask parameter of F.scaled_dot_product_attention.
    This is a trained *parameter* — no test-subject data is used to compute it.

  Efficient attention (F.scaled_dot_product_attention):
    Dispatches to Flash Attention on CUDA (PyTorch ≥ 2.0), giving O(T) memory.
    Falls back to manual scaled dot-product on older backends (O(T²) memory).
    attn_mask (float tensor) is added to logits before softmax in both paths.

  DropPath (stochastic depth):
    Randomly zeroes the entire residual contribution of each ConformerBlock
    sub-module for each sample in the training batch.  Disabled at model.eval()
    — inference is fully deterministic; no test-subject stochasticity.

  Depthwise conv kernel k=31 (15.5 ms at 2 kHz):
    Covers a single motor-unit action potential train (~5–50 ms), broader than
    the ASR Conformer default (k=7) to match EMG timescales.

  MixStyle (optional, off by default):
    Mixes channel-wise feature statistics (μ, σ) between random pairs of
    training batch samples.  Applied only during model.train() — no-op at
    model.eval().  In LOSO training all batch samples are from training subjects
    only, so no test-subject statistics can be mixed in.

LOSO data-leakage guarantees
─────────────────────────────
  ✓ RelativePositionBias: learned parameter (from train subjects only); not
    computed from test-window values.
  ✓ MHSA: operates within each window independently — no cross-window mixing.
  ✓ DropPath: disabled at model.eval() — deterministic test inference.
  ✓ BatchNorm: uses training running statistics at inference (model.eval()).
  ✓ LayerNorm: operates per-token over channel axis — no running statistics.
  ✓ MixStyle: operates on training batches; the test subject is never in any
    training batch under LOSO — no leakage possible.

Parameter budget (defaults: C=128, E=128, N=2, 4 heads, 8 EMG channels, 10 classes):
  ≈ 709 K  (vs exp_62 ECAPA-TDNN ≈ 467 K;  ~52 % more)
  With N=1 Conformer block: ≈ 588 K  (~26 % more than exp_62)

References
──────────
  Gulati et al. "Conformer: Convolution-augmented Transformer for Speech
    Recognition." INTERSPEECH 2020.
  Desplanques et al. "ECAPA-TDNN." INTERSPEECH 2020.
  Zhou et al. "MixStyle." ICLR 2021.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the proven ECAPA building blocks from exp_62 — avoids duplication and
# keeps SE-Res2Net behaviour identical to the baseline.
from models.ecapa_tdnn_emg import (
    AttentiveStatisticsPooling,
    SERes2NetBlock,
)

# Flash Attention availability check (PyTorch ≥ 2.0).
_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


# ═══════════════════════════ Conformer primitives ════════════════════════════


class DropPath(nn.Module):
    """
    Stochastic depth / DropPath (Huang et al., 2016).

    During training, zeros the entire output tensor for each sample with
    probability `drop_prob`, then rescales by 1/(1-p) to keep expectation.
    Disabled at model.eval() — test inference is fully deterministic.

    Applied to RESIDUAL PATHS only (the output of a sub-module before adding
    to the skip connection), so the main signal is never destroyed.

    Args:
        drop_prob: Probability of zeroing a sample's residual path (0 = off).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape (B, 1, 1, …) — broadcasts over all non-batch dimensions.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class RelativePositionBias(nn.Module):
    """
    Learnable 1-D relative position bias for multi-head self-attention.

    Maintains a compact lookup table of shape (2*max_dist+1, num_heads).
    For query position i and key position j the additive bias for head h is:
      table[clip(i−j, −max_dist, +max_dist) + max_dist, h]

    This encodes proximity (nearby frames share similar position bias) without
    rigid absolute-position assumptions, making the model robust to gesture
    segments of varying speed or temporal alignment across subjects.

    The table is a trained nn.Parameter — initialised near zero (trunc_normal
    σ=0.02) and updated only from training-subject windows.

    LOSO safety:
      ✓ Purely a parameter; not computed from test-window values.
      ✓ Applied identically to all windows regardless of subject.

    Args:
        num_heads: Number of attention heads.
        max_dist:  Maximum relative distance to distinguish.  Distances beyond
                   ±max_dist are folded into the same bucket (all treated equal).
    """

    def __init__(self, num_heads: int, max_dist: int = 64):
        super().__init__()
        self.max_dist = max_dist
        num_buckets = 2 * max_dist + 1
        self.bias_table = nn.Parameter(torch.zeros(num_buckets, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Returns additive attention bias of shape (1, num_heads, T, T).

        Computed on-the-fly per forward pass so that different T values
        (e.g. variable-length windows) are handled correctly.

        Memory: O(T² · num_heads) — for T=600, 4 heads: ~5.5 MB (float32).
        """
        positions = torch.arange(T, device=device)
        # rel_pos[i, j] = i − j; positive → query is later than key.
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # (T, T)
        rel_pos = rel_pos.clamp(-self.max_dist, self.max_dist)     # (T, T)
        idx = rel_pos + self.max_dist                              # (T, T) ∈ [0, 2D]

        # Lookup: (T, T, num_heads) → (1, num_heads, T, T)
        bias = self.bias_table[idx]                                # (T, T, H)
        return bias.permute(2, 0, 1).unsqueeze(0)                 # (1, H, T, T)

    def extra_repr(self) -> str:
        H = self.bias_table.shape[1]
        return f"num_heads={H}, max_dist={self.max_dist}"


class EfficientMHSA(nn.Module):
    """
    Multi-Head Self-Attention with learnable relative position bias.

    Uses F.scaled_dot_product_attention (Flash Attention) when available
    (PyTorch ≥ 2.0 + CUDA), reducing attention memory from O(T²) to O(T).
    Falls back to a manual scaled dot-product for CPU or older PyTorch.

    Input / output format: (B, T, C)  [sequence-first].

    The relative position bias is added to attention logits before softmax
    via the attn_mask argument (float tensor → additive bias, not masking).

    LOSO safety:
      ✓ Processes each window independently — no cross-window information.
      ✓ rel_pos_bias is a trained parameter, not derived from test data.
      ✓ No internal state beyond learned parameters.

    Args:
        C:            Model / channel dimension (must be divisible by num_heads).
        num_heads:    Number of attention heads.
        max_rel_dist: Max relative distance for the position bias table.
        dropout:      Attention-weight dropout probability (0 at eval time).
    """

    def __init__(
        self,
        C: int,
        num_heads: int = 4,
        max_rel_dist: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if C % num_heads != 0:
            raise ValueError(
                f"C={C} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim  = C // num_heads
        self.dropout   = dropout

        # Fused QKV projection — no bias (standard in Conformer/BERT).
        self.qkv  = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)

        self.rel_pos_bias = RelativePositionBias(num_heads, max_rel_dist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        qkv = self.qkv(x)                                         # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)                           # each (B, T, C)

        # Reshape to (B, H, T, head_dim).
        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _to_heads(q), _to_heads(k), _to_heads(v)       # (B, H, T, D)

        # Relative position bias: (1, H, T, T) — broadcast over batch.
        bias = self.rel_pos_bias(T, device=x.device)

        attn_drop = self.dropout if self.training else 0.0

        if _HAS_SDPA:
            # Flash-Attention path: O(T) memory, much faster on CUDA.
            # attn_mask as float tensor → added to logits before softmax.
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=bias,
                dropout_p=attn_drop,
            )                                                      # (B, H, T, D)
        else:
            # Manual fallback: O(T²) memory.
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale + bias  # (B,H,T,T)
            attn_w = F.softmax(scores, dim=-1)
            if attn_drop > 0:
                attn_w = F.dropout(attn_w, p=attn_drop, training=True)
            out = torch.matmul(attn_w, v)                          # (B, H, T, D)

        # Merge heads and project back to C.
        out = out.transpose(1, 2).reshape(B, T, C)                # (B, T, C)
        return self.proj(out)                                      # (B, T, C)

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"dropout={self.dropout}"
        )


class ConformerConvModule(nn.Module):
    """
    Conformer Convolutional Module — channels-first format: (B, C, T).

    Implements the conv sub-block from Gulati et al. (2020) with minor
    adaptations for EMG (wider depthwise kernel):
      LayerNorm → PW Conv (C→2C, GLU gate) → DW Conv (k) → BN → SiLU
               → PW Conv (C→C) → Dropout

    The GLU gate selectively suppresses irrelevant temporal activations while
    the depthwise conv provides local position-sensitive mixing with only
    O(C·k) parameters (vs O(C²) for a dense linear layer).

    Kernel choice k=31 (15.5 ms at 2 kHz):
      Covers one full motor-unit action potential train (~5–50 ms) and the
      fast rising edges of muscle co-contraction patterns.  Odd kernel ensures
      same-length output with symmetric causal padding.

    LOSO safety:
      ✓ BatchNorm uses training running statistics at inference (model.eval()).
      ✓ LayerNorm operates per-position — no running statistics.
      ✓ Dropout disabled at eval().

    Args:
        C:         Channel width (input = output).
        dw_kernel: Depthwise conv kernel size (must be odd; default 31).
        dropout:   Dropout probability after the final pointwise conv.
    """

    def __init__(self, C: int, dw_kernel: int = 31, dropout: float = 0.0):
        super().__init__()
        if dw_kernel % 2 == 0:
            raise ValueError(
                f"dw_kernel must be odd for same-length output, got {dw_kernel}"
            )
        pad = (dw_kernel - 1) // 2

        self.norm       = nn.LayerNorm(C)
        # Pointwise expand: C → 2C (provides the two tensors for GLU)
        self.pw_expand  = nn.Conv1d(C, 2 * C, kernel_size=1, bias=False)
        # Depthwise conv: each channel processed independently
        self.dw_conv    = nn.Conv1d(C, C, kernel_size=dw_kernel,
                                    padding=pad, groups=C, bias=False)
        self.bn         = nn.BatchNorm1d(C)
        # Pointwise project: C → C
        self.pw_project = nn.Conv1d(C, C, kernel_size=1, bias=False)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)

        # LayerNorm over channel dim (requires sequence-first temporarily).
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)   # (B, C, T)

        # PW expand + GLU: (B, 2C, T) → (B, C, T).
        # F.glu splits along dim=1: out = x[:, :C] * sigmoid(x[:, C:])
        out = F.glu(self.pw_expand(x), dim=1)               # (B, C, T)

        # Depthwise conv → BN → SiLU.
        out = F.silu(self.bn(self.dw_conv(out)))             # (B, C, T)

        # PW project + dropout.
        out = self.dropout(self.pw_project(out))             # (B, C, T)
        return out


class ConformerBlock(nn.Module):
    """
    Lightweight Conformer block for EMG (channels-first format: (B, C, T)).

    Structure (in order of application):
      x ──▶ [ConformerConvModule] ──┬──▶ x′
                                   │ DropPath residual
      x′ ──▶ [MHSA + rel-pos bias] ──┬──▶ x″
                                    │ DropPath residual
      x″ ──▶ LayerNorm ──▶ output

    The local depthwise conv module runs FIRST (position-sensitive local
    context), followed by global MHSA (long-range dependencies).  This
    ordering is inspired by the "ConvFirst" variant of the Conformer which
    has shown competitive performance in audio tasks.

    Compared to the original Conformer we omit the two half-FFN modules
    to reduce parameter count while retaining the core local + global
    interaction.  Ablation with FFN can be added via `use_ffn` flag.

    DropPath (stochastic depth):
      Randomly zeros the residual contribution of each sub-module per sample.
      Disabled at model.eval() — deterministic inference; no test stochasticity.

    LOSO safety:
      ✓ LayerNorm: per-position, no running statistics.
      ✓ DropPath: no-op at eval().
      ✓ MHSA: window-local, independent computation.

    Args:
        C:             Channel width.
        num_heads:     MHSA attention heads (C must be divisible by num_heads).
        dw_kernel:     Depthwise conv kernel size (odd; default 31).
        max_rel_dist:  Max relative distance for position bias lookup.
        attn_dropout:  Dropout inside MHSA attention weights.
        drop_path_prob: DropPath probability for stochastic depth.
        conv_dropout:  Dropout after final pointwise conv in ConvModule.
    """

    def __init__(
        self,
        C: int,
        num_heads: int = 4,
        dw_kernel: int = 31,
        max_rel_dist: int = 64,
        attn_dropout: float = 0.1,
        drop_path_prob: float = 0.1,
        conv_dropout: float = 0.0,
    ):
        super().__init__()

        # Sub-module 1: local context via depthwise conv
        self.conv_module    = ConformerConvModule(C, dw_kernel=dw_kernel,
                                                  dropout=conv_dropout)
        self.drop_path_conv = DropPath(drop_path_prob)

        # Sub-module 2: global context via MHSA + relative position bias
        self.norm_mhsa      = nn.LayerNorm(C)
        self.mhsa           = EfficientMHSA(C, num_heads=num_heads,
                                            max_rel_dist=max_rel_dist,
                                            dropout=attn_dropout)
        self.drop_path_mhsa = DropPath(drop_path_prob)

        # Final normalisation (stabilises gradient flow between blocks)
        self.final_norm     = nn.LayerNorm(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)

        # 1. Local: depthwise conv module — channels-first throughout.
        x = x + self.drop_path_conv(self.conv_module(x))          # (B, C, T)

        # 2. Global: MHSA needs sequence-first; transpose in/out.
        x_t = x.transpose(1, 2)                                    # (B, T, C)
        x_t = x_t + self.drop_path_mhsa(
            self.mhsa(self.norm_mhsa(x_t))
        )                                                           # (B, T, C)
        x = x_t.transpose(1, 2)                                    # (B, C, T)

        # 3. Final LayerNorm over the channel axis (channels-first).
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)     # (B, C, T)

        return x


# ───────────────────────────────── MixStyle ───────────────────────────────────

class _MixStyle(nn.Module):
    """
    MixStyle domain generalisation (Zhou et al., ICLR 2021).

    Randomly mixes channel-wise feature statistics (μ, σ) between random
    pairs of samples within a training batch using a Beta-distributed
    coefficient λ.  Applied only during model.train().

    LOSO safety:
      In each LOSO fold all training-batch samples come exclusively from
      training subjects.  The test subject is never part of any training batch,
      so cross-subject style mixing cannot introduce test-subject information.

    Args:
        p:     Probability of applying MixStyle per forward pass (default 0.5).
        alpha: Beta distribution concentration (smaller → extreme λ values).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p    = p
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B = x.size(0)

        # Per-instance channel statistics: (B, C, 1)
        mu  = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        sig = (var + 1e-6).sqrt()

        # Normalise (remove style)
        x_norm = (x - mu) / sig

        # Mixing coefficient λ ∈ (0, 1): (B, 1, 1)
        lam = self.beta.sample((B,)).to(x.device).view(B, 1, 1)

        # Random permutation of batch for style donor
        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        # Mix styles and re-inject into normalised features
        mu_mix  = lam * mu  + (1.0 - lam) * mu2
        sig_mix = lam * sig + (1.0 - lam) * sig2

        return x_norm * sig_mix + mu_mix


# ═══════════════════════════ Full ConformerECAPA model ═══════════════════════


class ConformerECAPA(nn.Module):
    """
    Conformer-ECAPA encoder + classifier for EMG gesture recognition.

    Combines N lightweight Conformer blocks (local DW conv + global MHSA with
    relative position bias) with the ECAPA-TDNN backbone (SE-Res2Net ×3 +
    Multi-layer Feature Aggregation + Attentive Statistics Pooling).

    The Conformer blocks sit between the conv stem and the ECAPA backbone:
      stem → Conformer×N → SE-Res2Net×3 → MFA → ASP → embed → classify

    The MFA aggregates only the three SE-Res2Net block outputs (same as
    exp_62), so the ECAPA multi-scale aggregation pattern is preserved.

    See module docstring for LOSO guarantees, design choices, and parameter
    budget.

    Args:
        in_channels:       EMG input channels (e.g. 8).
        num_classes:       Number of gesture classes.
        channels:          C — internal feature dimension (default 128).
        num_conformer:     N — Conformer blocks before ECAPA backbone (default 2).
        conformer_heads:   MHSA heads per Conformer block (default 4).
        dw_kernel:         DW conv kernel in ConformerConvModule (default 31).
        max_rel_dist:      Position bias clip distance (default 64).
        attn_dropout:      MHSA attention-weight dropout (default 0.1).
        drop_path_prob:    DropPath probability per Conformer sub-module (default 0.1).
        scale:             Res2Net scale / sub-group count (default 4).
        dilations:         Dilation per SE-Res2Net block (default [2, 3, 4]).
        se_reduction:      SE bottleneck reduction factor (default 8).
        embedding_dim:     E — pre-classifier embedding dimension (default 128).
        dropout:           Dropout before classifier (default 0.3).
        use_mixstyle:      Apply MixStyle after Conformer blocks (default False).
        mixstyle_p:        MixStyle application probability (default 0.5).
        mixstyle_alpha:    MixStyle Beta distribution alpha (default 0.1).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: int = 128,
        num_conformer: int = 2,
        conformer_heads: int = 4,
        dw_kernel: int = 31,
        max_rel_dist: int = 64,
        attn_dropout: float = 0.1,
        drop_path_prob: float = 0.1,
        scale: int = 4,
        dilations: Optional[List[int]] = None,
        se_reduction: int = 8,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        use_mixstyle: bool = False,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
    ):
        super().__init__()

        if dilations is None:
            dilations = [2, 3, 4]
        if len(dilations) != 3:
            raise ValueError(
                f"Exactly 3 dilation values required (one per SE-Res2Net block), "
                f"got {len(dilations)}: {dilations}"
            )

        self.channels      = channels
        self.embedding_dim = embedding_dim
        self.use_mixstyle  = use_mixstyle
        _NUM_SE_BLOCKS     = 3  # fixed: matches exp_62 backbone

        # ── 1. Conv stem ────────────────────────────────────────────────────
        # k=5 gives a 5-sample receptive field (2.5 ms at 2 kHz) — enough to
        # mix EMG channels and detect single action-potential onset edges.
        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # ── 2. Conformer blocks ──────────────────────────────────────────────
        # Stochastic depth with linearly increasing probability (deeper block
        # → higher drop probability), following standard practice.
        self.conformer_blocks = nn.ModuleList()
        for i in range(num_conformer):
            dp = drop_path_prob * (i + 1) / max(num_conformer, 1)
            self.conformer_blocks.append(
                ConformerBlock(
                    C=channels,
                    num_heads=conformer_heads,
                    dw_kernel=dw_kernel,
                    max_rel_dist=max_rel_dist,
                    attn_dropout=attn_dropout,
                    drop_path_prob=dp,
                )
            )

        # Optional MixStyle after Conformer blocks (training-only, no-op at eval)
        self.mixstyle = _MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) \
            if use_mixstyle else None

        # ── 3. SE-Res2Net blocks (identical to ECAPA backbone in exp_62) ────
        # Increasing dilation exposes the model to exponentially growing
        # temporal contexts: 5, 7, 9 samples per block at k=3 + residuals.
        self.se_blocks = nn.ModuleList([
            SERes2NetBlock(
                channels,
                kernel_size=3,
                dilation=d,
                scale=scale,
                se_reduction=se_reduction,
            )
            for d in dilations
        ])

        # ── 4. Multi-layer Feature Aggregation (SE-Res2Net outputs only) ────
        # Concatenates outputs of all three SE-Res2Net blocks: (B, 3C, T).
        # A 1×1 conv mixes cross-scale, cross-block information.
        mfa_in = channels * _NUM_SE_BLOCKS  # 3C = 384 for C=128
        self.mfa = nn.Sequential(
            nn.Conv1d(mfa_in, mfa_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_in),
            nn.ReLU(inplace=True),
        )

        # ── 5. Attentive Statistics Pooling ─────────────────────────────────
        # (B, 3C, T) → (B, 6C): weighted mean + weighted std over time.
        self.asp = AttentiveStatisticsPooling(mfa_in)

        # ── 6. FC embedding ─────────────────────────────────────────────────
        asp_out_dim = mfa_in * 2  # 6C = 768 for C=128
        self.embedding = nn.Sequential(
            nn.Linear(asp_out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── 7. Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────

    def _init_weights(self):
        """He-uniform for Conv1d and Linear; constant 1/0 for BatchNorm."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # RelativePositionBias.bias_table uses trunc_normal (own init).
            # LayerNorm default init (ones/zeros) is appropriate.

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_emg, T) — EMG windows in channels-first format.
        Returns:
            logits: (B, num_classes)
        """
        # 1. Conv stem
        out = self.init_tdnn(x)                           # (B, C, T)

        # 2. Conformer blocks (local DW conv → global MHSA)
        for conf_block in self.conformer_blocks:
            out = conf_block(out)                         # (B, C, T)

        # Optional MixStyle (training-only; no-op at eval)
        if self.mixstyle is not None:
            out = self.mixstyle(out)                      # (B, C, T)

        # 3. SE-Res2Net blocks; collect outputs for MFA
        se_outputs = []
        for se_block in self.se_blocks:
            out = se_block(out)                           # (B, C, T)
            se_outputs.append(out)

        # 4. MFA: concatenate all SE-Res2Net block outputs
        mfa_in  = torch.cat(se_outputs, dim=1)            # (B, 3C, T)
        mfa_out = self.mfa(mfa_in)                        # (B, 3C, T)

        # 5. Attentive Statistics Pooling: collapse time dimension
        pooled = self.asp(mfa_out)                        # (B, 6C)

        # 6. Embedding
        emb = self.embedding(pooled)                      # (B, E)

        # 7. Classification
        return self.classifier(emb)                       # (B, num_classes)

    # ── Utility ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
