"""
models/filterbank_mode_adaptive_ecapa.py

Learnable Filterbank + Per-Band Soft AGC + Per-Band ECAPA Encoders
+ Mode-Adaptive Style Erasure + Multi-Band Attentive Fusion.

Architecture (Hypothesis 4):
    Input (B, C, T)  — channel-standardized raw EMG, channels-first
      ↓
    SincFilterbank(K=6 filters)
      → (B, C*K, T)  → reshape → (B, K, C, T)   [K parallel band streams]
      ↓
    PerBandSoftAGC — EMA-based normalization applied independently per (band, channel)
      → (B, K, C, T)   [amplitude-normalized band streams]
      ↓
    K × LightECAPAEncoder (K=6, channels=256 instead of 512, shared architecture but
      separate parameters per band)
      → (B, K, D)      [one D-dim embedding per band, via Attentive Statistics Pooling]
      ↓
    ModeAdaptiveStyleErasure — per-band Gradient Reversal with learnable λ_k weights
      → (B, K, D)      [style-erased band embeddings; GRL applied during training only]
      ↓
    MultiBandAttentiveFusion — task-query attention over K band embeddings
      → (B, D)         [attended fusion]
      ↓
    gesture classifier  → (B, num_classes)
    subject classifier  → (B, num_subjects)  [GRL branch, IGNORED at inference]

LOSO integrity guarantees
─────────────────────────
  ✓ SincFilterbank: per-window linear filter — no cross-window state.
  ✓ PerBandSoftAGC: causal EMA implemented as depthwise conv1d, re-initialized
    on every forward call — no persistent state, no test-subject information used.
  ✓ Per-band ECAPA encoders: BatchNorm running stats accumulated from training
    subjects only; frozen at test time via model.eval().
  ✓ AttentiveStatisticsPooling: purely input-driven (no stored state).
  ✓ ModeAdaptiveStyleErasure: λ_k learned from training data; gradient reversal
    has NO effect at inference (torch.no_grad() → no backward pass).
  ✓ subject_logits: IGNORED at inference — only task_logits are returned/used.
  ✓ Channel standardization (mean/std): computed from training data only by the
    experiment; applied as a fixed affine transform to val and test.

Design rationale
────────────────
  - K=6 light ECAPA (C=256) vs one heavy (C=512): ~comparable total params but
    each encoder specialises on a frequency band, reducing within-band complexity.
  - Per-band λ_k: the network learns HOW MUCH to erase subject style per band.
    High-frequency bands (muscle crosstalk) may need stronger erasure than
    low-frequency bands (neural drive); λ_k adapts automatically.
  - Soft AGC (not PCEN): exp_76 showed Soft AGC (34.32%) outperforms PCEN (17.61%)
    dramatically; the bounded exponent preserves gesture-relevant amplitude cues.
  - Multi-band fusion via task query: same cross-band attention as exp_82 mode
    attention, which showed non-uniform band importance for gesture classification.

References
──────────
  - ECAPA-TDNN: Desplanques et al., Interspeech 2020
  - SincNet: Ravanelli & Bengio, SLT 2018
  - DANN / GRL: Ganin et al., JMLR 2016
  - Exp 76 (Soft AGC), Exp 82 (mode attention), Exp 94 (learnable filterbank GRL)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sinc_pcen_cnn_gru import SincFilterbank
from models.ecapa_tdnn_emg import (
    Res2NetBlock,
    SEBlock,
    SERes2NetBlock,
    AttentiveStatisticsPooling,
)


# ──────────────────────────── Per-Band Soft AGC ───────────────────────────────

class PerBandSoftAGC(nn.Module):
    """
    Soft Automatic Gain Control applied independently to each (band, channel) pair.

    Extends SoftAGCLayer (exp_76) from C channels to K*C (band, channel) pairs.
    Each of the K*C pairs has its own learnable (alpha_raw, log_s).

    EMA formula (causal depthwise conv1d):
        M[k, c, t] = (1 - s[k,c]) * M[k,c,t-1] + s[k,c] * |x[k,c,t]|
        out[k,c,t] = x[k,c,t] / (M[k,c,t]^alpha[k,c] + delta)

    Constraints (vs full PCEN — see exp_76):
        alpha ∈ (0, 0.5) via sigmoid(raw) * 0.5
        delta FIXED (not learned) — avoids noise-floor exploitation
        No root compression

    LOSO integrity:
        - EMA recomputed fresh per forward call (no cross-window state).
        - alpha_raw and log_s receive gradients from training data only.
        - delta is a fixed hyperparameter — cannot adapt to test subject.
        - At inference: model.eval() → all parameters frozen.

    Args:
        num_bands:         K — number of frequency bands
        num_channels:      C — number of EMG channels
        ema_kernel_length: length of truncated causal EMA kernel
        delta:             FIXED additive stabilizer in gain denominator
        eps:               numerical safety floor for EMA clamp
    """

    def __init__(
        self,
        num_bands: int,
        num_channels: int,
        ema_kernel_length: int = 100,
        delta: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_bands    = num_bands
        self.num_channels = num_channels
        self.KC           = num_bands * num_channels
        self.ema_kernel_length = ema_kernel_length
        self.delta = delta  # intentionally NOT nn.Parameter
        self.eps   = eps

        # alpha_raw: sigmoid(alpha_raw) * 0.5 → alpha ∈ (0, 0.5)
        # Init at 0 → alpha = 0.25 (moderate suppression)
        self.alpha_raw = nn.Parameter(torch.zeros(self.KC))

        # log_s: logit of EMA smoothing coefficient s ∈ (0, 1)
        # s_init=0.04 → time-constant 25 samples ≈ 12.5 ms @ 2 kHz
        s_init = 0.04
        self.log_s = nn.Parameter(
            torch.full((self.KC,), math.log(s_init / (1.0 - s_init)))
        )

    def _build_ema_kernel(self, device: torch.device) -> torch.Tensor:
        """Build (KC, 1, L) causal EMA kernel for depthwise conv1d."""
        L = self.ema_kernel_length
        s = torch.sigmoid(self.log_s)                            # (KC,)
        j = torch.arange(L, device=device, dtype=s.dtype)       # lags 0..L-1
        impulse = s.unsqueeze(1) * (1.0 - s.unsqueeze(1)) ** j.unsqueeze(0)  # (KC, L)
        impulse = impulse / (impulse.sum(dim=1, keepdim=True) + 1e-8)
        kernel  = impulse.flip(dims=[1])                         # causal: recent last
        return kernel.unsqueeze(1)                               # (KC, 1, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, C, T) — K band streams, channels-first per band
        Returns:
            out: (B, K, C, T) — AGC-normalized band streams
        """
        B, K, C, T = x.shape
        assert K == self.num_bands and C == self.num_channels, (
            f"PerBandSoftAGC: expected ({self.num_bands}, {self.num_channels}), "
            f"got ({K}, {C})"
        )

        # Flatten bands and channels: (B, K*C, T)
        x_flat = x.reshape(B, self.KC, T)

        alpha = torch.sigmoid(self.alpha_raw) * 0.5  # (KC,)

        x_mag = x_flat.abs()  # (B, KC, T)

        L = min(self.ema_kernel_length, T)
        kernel = self._build_ema_kernel(x_flat.device)[:, :, -L:]  # (KC, 1, L)
        x_padded = F.pad(x_mag, (L - 1, 0))                        # (B, KC, T+L-1)
        M = F.conv1d(x_padded, kernel, groups=self.KC)              # (B, KC, T)

        alpha_bc = alpha.view(1, self.KC, 1)
        gain = M.clamp(min=self.eps).pow(alpha_bc) + self.delta     # (B, KC, T)

        out_flat = x_flat / gain                                    # (B, KC, T)
        return out_flat.reshape(B, K, C, T)


# ─────────────────── Light ECAPA Encoder (reduced channels) ──────────────────

class LightECAPAEncoder(nn.Module):
    """
    Lightweight ECAPA-TDNN encoder for a single frequency band.

    Identical architecture to ECAPATDNNEmg but with reduced internal channels
    (default C=256 instead of 512) to control total parameter count when K=6
    encoders run in parallel.

    Output: (B, embedding_dim) — a fixed-size embedding per window via
    Attentive Statistics Pooling over the temporal dimension.

    LOSO integrity:
        - BatchNorm: running stats accumulated from training data only.
        - AttentiveStatisticsPooling: purely input-driven, no stored state.
        - model.eval() at inference freezes all parameters.

    Args:
        in_channels:   C — number of EMG input channels for this band stream
        channels:      internal ECAPA channel width (default 256, reduced from 512)
        scale:         Res2Net scale (default 4)
        embedding_dim: output embedding dimension (default 128)
        dilations:     dilation per SE-Res2Net block (default [2, 3, 4])
        dropout:       dropout probability
        se_reduction:  SE bottleneck reduction factor
    """

    def __init__(
        self,
        in_channels: int,
        channels: int = 256,
        scale: int = 4,
        embedding_dim: int = 128,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [2, 3, 4]
        assert len(dilations) == 3

        self.channels      = channels
        self.embedding_dim = embedding_dim
        num_blocks         = 3

        # Initial TDNN
        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # Three SE-Res2Net blocks with increasing dilation
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

        # Multi-layer Feature Aggregation: concat all block outputs → 1×1 conv
        mfa_in = channels * num_blocks
        self.mfa = nn.Sequential(
            nn.Conv1d(mfa_in, mfa_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_in),
            nn.ReLU(inplace=True),
        )

        # Attentive Statistics Pooling: (B, 3C, T) → (B, 6C)
        self.asp = AttentiveStatisticsPooling(mfa_in)

        # FC embedding: (B, 6C) → (B, embedding_dim)
        asp_out = mfa_in * 2
        self.embedding = nn.Sequential(
            nn.Linear(asp_out, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — one frequency-band stream, channels-first
        Returns:
            emb: (B, embedding_dim) — band embedding
        """
        out = self.init_tdnn(x)                # (B, channels, T)

        block_outputs = []
        for block in self.blocks:
            out = block(out)
            block_outputs.append(out)

        mfa_in  = torch.cat(block_outputs, dim=1)  # (B, 3*channels, T)
        mfa_out = self.mfa(mfa_in)                  # (B, 3*channels, T)

        pooled = self.asp(mfa_out)                  # (B, 6*channels)
        emb    = self.embedding(pooled)             # (B, embedding_dim)
        return emb


# ──────────────────── Gradient Reversal Function ─────────────────────────────

class _GRLFunction(torch.autograd.Function):
    """Identity forward, negated-and-scaled backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_val * grad_output, None


# ─────────────────── Mode-Adaptive Style Erasure ─────────────────────────────

class ModeAdaptiveStyleErasure(nn.Module):
    """
    Per-band Gradient Reversal with learnable band-wise λ_k weights.

    Idea: different frequency bands contain different amounts of subject-specific
    (style) information.  A single global GRL strength (exp_94) treats all bands
    equally.  Here each band k has its own effective reversal strength:
        λ_eff[k] = λ_global * sigmoid(raw_lambda[k])

    λ_global is set externally via set_lambda() following the DANN warm-up
    schedule.  raw_lambda[k] is a learnable per-band scalar that modulates HOW
    MUCH style erasure is applied to band k:
        - If raw_lambda[k] → +∞ → sigmoid → 1 → band gets full erasure
        - If raw_lambda[k] → -∞ → sigmoid → 0 → band is nearly unaffected

    This lets the network self-discover which frequency bands are more
    subject-contaminated without any prior knowledge.

    LOSO integrity:
        - raw_lambda receives gradients from training data only.
        - At inference (torch.no_grad()): GRL has NO effect because no backward
          pass occurs.  The subject_classifier head is DISCARDED at inference.
        - The λ_k values are frozen at test time (model.eval()).

    Args:
        num_bands:   K — number of frequency bands
        lambda_:     initial global GRL reversal strength (0 at training start)
    """

    def __init__(self, num_bands: int, lambda_: float = 0.0) -> None:
        super().__init__()
        self.lambda_: float = lambda_
        # Learnable per-band modulation: init at 0 → sigmoid → 0.5 (moderate start)
        self.raw_lambda = nn.Parameter(torch.zeros(num_bands))

    def set_lambda(self, lambda_: float) -> None:
        """Update global GRL strength (called once per epoch, DANN schedule)."""
        self.lambda_ = lambda_

    def forward(self, mode_reprs: torch.Tensor) -> torch.Tensor:
        """
        Apply per-band style erasure to band embeddings.

        During training backward pass: gradient flowing back through band k is
        scaled by -λ_eff[k], pushing the encoder to produce subject-invariant
        band-k representations.

        During evaluation (torch.no_grad()): _GRLFunction.backward is never
        called, so this is a pure identity transform at inference.

        Args:
            mode_reprs: (B, K, D) — K band embeddings
        Returns:
            mode_reprs_erased: (B, K, D) — same tensor with per-band GRL applied
        """
        B, K, D = mode_reprs.shape
        # Per-band effective lambda: λ_global * sigmoid(raw_lambda[k])
        per_band_lambda = self.lambda_ * torch.sigmoid(self.raw_lambda)  # (K,)

        # Apply GRL independently to each band
        erased_bands: List[torch.Tensor] = []
        for k in range(K):
            band_emb = mode_reprs[:, k, :]                             # (B, D)
            erased   = _GRLFunction.apply(band_emb, float(per_band_lambda[k]))
            erased_bands.append(erased.unsqueeze(1))                   # (B, 1, D)

        return torch.cat(erased_bands, dim=1)  # (B, K, D)


# ─────────────────────── Multi-Band Attentive Fusion ─────────────────────────

class MultiBandAttentiveFusion(nn.Module):
    """
    Attention-weighted fusion of K band embeddings.

    A learnable task query attends over the K band representations, producing a
    weighted combination that emphasizes gesture-discriminative frequency bands.

    Mechanism:
        Q = task_query      — (B, 1, D)   [shared learnable token]
        K = W_k · bands     — (B, K, D)
        V = W_v · bands     — (B, K, D)
        out = softmax(Q Kᵀ / √D) · V  → (B, 1, D) → (B, D)
        attn_weights = softmax weights → (B, K)

    LOSO integrity: purely input-driven at inference; no stored state.

    Args:
        embed_dim:  D — band embedding dimension
        num_heads:  number of attention heads (must divide embed_dim)
        dropout:    attention dropout
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Learnable task query; small init for stable early training
        self.task_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self, mode_reprs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mode_reprs: (B, K, D) — K band embeddings per sample
        Returns:
            fused:        (B, D)  — attended fusion
            attn_weights: (B, K)  — band importance weights (sums to ~1)
        """
        B = mode_reprs.size(0)
        q = self.task_query.expand(B, -1, -1)                # (B, 1, D)
        out, attn = self.mha(q, mode_reprs, mode_reprs,
                             need_weights=True)               # (B,1,D), (B,1,K)
        return out.squeeze(1), attn.squeeze(1)                # (B,D), (B,K)


# ─────────────────────────── Full Model ──────────────────────────────────────

class FilterbankModeAdaptiveECAPA(nn.Module):
    """
    Learnable Filterbank + Per-Band Soft AGC + Per-Band Light ECAPA Encoders
    + Mode-Adaptive Style Erasure + Multi-Band Attentive Fusion.

    (Hypothesis 4: minimum-viable disentanglement using proven components.)

    Forward returns a 3-tuple:
        (task_logits, subject_logits, attn_weights)
        - task_logits:    (B, num_classes)  ← used for gesture recognition
        - subject_logits: (B, num_subjects) ← GRL branch; IGNORED at inference
        - attn_weights:   (B, K)            ← band importance; for analysis

    Args:
        in_channels:      C — EMG input channels (8 for NinaPro DB2)
        num_classes:      number of gesture classes
        num_subjects:     number of TRAINING subjects in this fold (for GRL head).
                          Must be set fresh per fold. Ignored at inference.
        num_bands:        K — number of Sinc bandpass filters / frequency bands
        sinc_kernel_size: Sinc FIR kernel length (odd, e.g. 51)
        sample_rate:      EMG sampling rate in Hz
        min_freq:         lowest Sinc filter cutoff in Hz
        max_freq:         highest Sinc filter cutoff in Hz
        ecapa_channels:   C_ECAPA — internal channel width per LightECAPAEncoder
        ecapa_scale:      Res2Net scale
        embedding_dim:    D — band embedding dimension (fusion input/output)
        ecapa_dilations:  dilation values per SE-Res2Net block
        num_heads:        attention heads for MultiBandAttentiveFusion
        agc_ema_length:   EMA kernel length for PerBandSoftAGC
        agc_delta:        FIXED stabilizer δ for PerBandSoftAGC
        dropout:          dropout probability
        grl_lambda:       initial global GRL strength (set to 0; warm-up updates it)
        se_reduction:     SE bottleneck reduction factor
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        num_subjects: int = 4,
        num_bands: int = 6,
        sinc_kernel_size: int = 51,
        sample_rate: int = 2000,
        min_freq: float = 5.0,
        max_freq: float = 500.0,
        ecapa_channels: int = 256,
        ecapa_scale: int = 4,
        embedding_dim: int = 128,
        ecapa_dilations: Optional[List[int]] = None,
        num_heads: int = 4,
        agc_ema_length: int = 100,
        agc_delta: float = 0.1,
        dropout: float = 0.3,
        grl_lambda: float = 0.0,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()

        if ecapa_dilations is None:
            ecapa_dilations = [2, 3, 4]

        self.num_bands    = num_bands
        self.in_channels  = in_channels
        self.embedding_dim = embedding_dim

        # ── 1. SincFilterbank: K learnable bandpass filters ───────────────
        # Applied to each channel independently.
        # Output: (B, C*K, T) — channel-then-filter interleaved layout.
        # Reshape in forward: (B, C*K, T) → (B, C, K, T) → perm → (B, K, C, T)
        self.sinc = SincFilterbank(
            num_filters  = num_bands,
            kernel_size  = sinc_kernel_size,
            sample_rate  = sample_rate,
            min_freq     = min_freq,
            max_freq     = max_freq,
            in_channels  = in_channels,
        )

        # ── 2. Per-Band Soft AGC: EMA normalization per (band, channel) ───
        # Reduces inter-subject amplitude variation at earliest possible stage.
        self.per_band_agc = PerBandSoftAGC(
            num_bands         = num_bands,
            num_channels      = in_channels,
            ema_kernel_length = agc_ema_length,
            delta             = agc_delta,
        )

        # ── 3. K Light ECAPA Encoders (separate parameters per band) ──────
        # Each encoder takes (B, C, T) and outputs (B, D).
        # Separate parameters: each specialises on its frequency band.
        self.band_encoders = nn.ModuleList([
            LightECAPAEncoder(
                in_channels   = in_channels,
                channels      = ecapa_channels,
                scale         = ecapa_scale,
                embedding_dim = embedding_dim,
                dilations     = ecapa_dilations,
                dropout       = dropout,
                se_reduction  = se_reduction,
            )
            for _ in range(num_bands)
        ])

        # ── 4. Mode-Adaptive Style Erasure ─────────────────────────────────
        # Per-band learnable GRL: λ_eff[k] = λ_global × sigmoid(raw_λ[k]).
        self.style_erasure = ModeAdaptiveStyleErasure(
            num_bands = num_bands,
            lambda_   = grl_lambda,
        )

        # ── 5. Multi-Band Attentive Fusion ─────────────────────────────────
        self.fusion = MultiBandAttentiveFusion(
            embed_dim = embedding_dim,
            num_heads = num_heads,
            dropout   = 0.1,
        )

        # ── 6. Gesture classification head ─────────────────────────────────
        self.task_classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

        # ── 7. Subject classifier (GRL branch) — IGNORED at inference ──────
        # Receives style-erased embeddings fused via simple mean-pool.
        # We use mean-pool for the adversarial branch (not task-query attention)
        # so the GRL acts on a simpler aggregation, improving gradient signal.
        self.subject_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(embedding_dim // 2, num_subjects),
        )

    # ── External controls ─────────────────────────────────────────────────────

    def set_lambda(self, lambda_: float) -> None:
        """Update global GRL reversal strength (DANN warm-up). Call per epoch."""
        self.style_erasure.set_lambda(lambda_)

    def get_per_band_lambda(self) -> torch.Tensor:
        """
        Return effective per-band GRL strengths for logging/analysis.
        Returns (K,) tensor: λ_global × sigmoid(raw_λ[k]).
        """
        with torch.no_grad():
            return (
                self.style_erasure.lambda_
                * torch.sigmoid(self.style_erasure.raw_lambda)
            ).cpu()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) — channel-standardized raw EMG windows, channels-first
        Returns:
            task_logits:    (B, num_classes)  — gesture classification logits
            subject_logits: (B, num_subjects) — GRL adversarial head (ignored at test)
            attn_weights:   (B, K)            — per-band attention weights for analysis
        """
        B, C, T = x.shape
        K = self.num_bands

        # ── Step 1: Sinc filterbank ─────────────────────────────────────────
        # SincFilterbank output: (B, C*K, T) with layout [c*K + k] per position
        filtered = self.sinc(x)                                   # (B, C*K, T)

        # Reshape to (B, K, C, T): K band streams, each with all C channels
        # SincFilterbank layout: out[b, c*K+k, t] → reshape(B,C,K,T) → perm(B,K,C,T)
        filtered = (
            filtered
            .reshape(B, C, K, T)
            .permute(0, 2, 1, 3)
            .contiguous()
        )                                                          # (B, K, C, T)

        # ── Step 2: Per-Band Soft AGC ───────────────────────────────────────
        # Normalizes amplitude independently per (band, channel) using causal EMA.
        agc_out = self.per_band_agc(filtered)                     # (B, K, C, T)

        # ── Step 3: K Light ECAPA Encoders ─────────────────────────────────
        # Each encoder processes its own band stream: (B, C, T) → (B, D).
        # Collected and stacked: (B, K, D).
        band_embs: List[torch.Tensor] = []
        for k, encoder in enumerate(self.band_encoders):
            band_stream = agc_out[:, k, :, :]                     # (B, C, T)
            band_emb    = encoder(band_stream)                     # (B, D)
            band_embs.append(band_emb.unsqueeze(1))               # (B, 1, D)
        mode_reprs = torch.cat(band_embs, dim=1)                  # (B, K, D)

        # ── Step 4: Mode-Adaptive Style Erasure ────────────────────────────
        # Apply per-band GRL: gradient of subject loss reversed through each band.
        # At inference (torch.no_grad()): purely identity — no effect.
        mode_erased = self.style_erasure(mode_reprs)              # (B, K, D)

        # ── Step 5: Multi-Band Attentive Fusion ────────────────────────────
        # Task query attends over K erased band embeddings → one fused vector.
        fused, attn_weights = self.fusion(mode_erased)            # (B,D), (B,K)

        # ── Step 6: Gesture classification ─────────────────────────────────
        task_logits = self.task_classifier(fused)                 # (B, num_classes)

        # ── Step 7: Subject classification (GRL branch) ────────────────────
        # Mean-pool erased band embeddings → adversarial subject head.
        # The gradient reversal is already embedded in mode_erased tensors.
        # IGNORED at inference; trained only on training subjects.
        erased_mean = mode_erased.mean(dim=1)                     # (B, D)
        subject_logits = self.subject_classifier(erased_mean)     # (B, num_subjects)

        return task_logits, subject_logits, attn_weights

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
