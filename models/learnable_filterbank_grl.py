"""
models/learnable_filterbank_grl.py

Learnable Filterbank with Mode-Level Attention and Gradient Reversal Layer (GRL).

Architecture:
    Input (B, C, T)  — channel-standardized raw EMG
      → SincFilterbank (K learnable bandpass filters, shared across channels)
        → (B, C*K, T)
        → reshape → (B, K, C, T)   [K parallel mode streams]
      → SharedModeGRUEncoder       [shared weights, applied to all K modes]
        → (B, K, D)                [one D-dim representation per mode]
      → MultiHeadModeAttention     [learned task query attends over K modes]
        → fused:       (B, D)      [attended fusion]
        → attn_weights: (B, K)     [interpretable mode importance]
      → task_classifier            → (B, num_classes)
      → GradientReversalLayer
      → subject_classifier         → (B, num_subjects)   [ignored at inference]

LOSO integrity:
    ✓ All parameters trained exclusively on train-subject data; frozen at test time.
    ✓ SincFilterbank: per-window per-channel linear filter — no cross-sample state.
    ✓ SharedModeGRUEncoder: GRU state reset each forward pass — no cross-window memory.
    ✓ GRL trains subject-invariant features from training subjects; generalizes to
      unseen test subjects WITHOUT any test-time adaptation.
    ✓ subject_logits are IGNORED at inference; only task_logits are used.
    ✓ Channel standardization (mean/std) is computed from training data only by the
      experiment and applied as a fixed transform before model input.

Reference:
    Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR 2016.
    Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with SincNet," SLT 2018.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sinc_pcen_cnn_gru import SincFilterbank


# ─────────────────────────── Gradient Reversal Layer ─────────────────────────

class _GRLFunction(torch.autograd.Function):
    """Autograd function: identity forward, scaled-negation backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for domain-adversarial training.

    Forward  : identity  (x → x)
    Backward : gradient multiplied by -lambda_  (reversal)

    When a downstream subject_classifier minimizes CE, the reversed gradient
    propagated to the upstream encoder acts as a *maximizer* — pushing the
    encoder to produce representations from which subject identity cannot be
    predicted, i.e., subject-invariant features.

    lambda_ follows the DANN warm-up schedule (see `get_grl_lambda` in the
    experiment file): starts near 0, increases to lambda_max over training.

    LOSO note: at inference (model.eval(), torch.no_grad()), GRL has no effect
    because no gradients flow. subject_logits are discarded by the evaluator.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.lambda_)


# ───────────────────────── Shared Mode GRU Encoder ───────────────────────────

class SharedModeGRUEncoder(nn.Module):
    """
    Lightweight CNN-GRU encoder with temporal attention pooling.

    Applied independently to each of K mode streams using SHARED weights.
    The same parameters process all frequency bands, forcing the encoder to
    learn a mode-agnostic temporal pattern extractor; frequency-specific
    relevance is captured downstream by MultiHeadModeAttention.

    Input:  (B*K, C, T) — K mode streams flattened into the batch dimension.
    Output: (B*K, D)    — one D-dim representation per (sample, mode) pair.

    Architecture:
        Conv1d (5-tap) → BN → ReLU
        Conv1d (3-tap) → BN → ReLU       [local temporal feature extraction]
        GRU            [temporal sequence modelling]
        Temporal attention pooling        [weighted sum over time steps]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cnn_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B*K, C, T) — one frequency-band stream per batch element
        Returns:
            ctx: (B*K, D)    — temporally pooled mode representation
        """
        h = self.cnn_proj(x)                              # (B*K, D, T)
        h = h.transpose(1, 2)                              # (B*K, T, D)
        gru_out, _ = self.gru(h)                           # (B*K, T, D)
        w = torch.softmax(self.attn_fc(gru_out), dim=1)   # (B*K, T, 1)
        ctx = (w * gru_out).sum(dim=1)                     # (B*K, D)
        return ctx


# ────────────────────────── Multi-Head Mode Attention ────────────────────────

class MultiHeadModeAttention(nn.Module):
    """
    Multi-head attention: a learned task query attends over K mode representations.

    The task query is a learnable parameter (independent of input), representing
    the "question" posed to the K frequency bands: which modes carry
    gesture-discriminative information?

    Mechanism:
        Q = W_q · task_query    — (B, 1, D), shared across batch elements
        K = W_k · mode_reprs   — (B, K, D)
        V = W_v · mode_reprs   — (B, K, D)
        out = softmax(Q Kᵀ / √d) · V   → (B, 1, D)
        fused = out.squeeze(1)          → (B, D)
        attn_weights = softmax weights  → (B, K)

    Constraint: mode_dim must be divisible by num_heads.
    """

    def __init__(
        self,
        mode_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert mode_dim % num_heads == 0, (
            f"mode_dim ({mode_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=mode_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Learnable task query token; initialized near zero for stable start
        self.task_query = nn.Parameter(torch.randn(1, 1, mode_dim) * 0.02)

    def forward(
        self, mode_reprs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mode_reprs: (B, K, D) — K mode representations per sample
        Returns:
            fused:        (B, D)  — attended fusion of K modes
            attn_weights: (B, K)  — mode importance (sums to ~1 per sample)
        """
        B = mode_reprs.size(0)
        q = self.task_query.expand(B, -1, -1)           # (B, 1, D)
        out, attn_weights = self.mha(
            q, mode_reprs, mode_reprs, need_weights=True
        )
        # out: (B, 1, D),  attn_weights: (B, 1, K)  [averaged across heads]
        return out.squeeze(1), attn_weights.squeeze(1)  # (B, D), (B, K)


# ────────────────────────────── Full Model ────────────────────────────────────

class LearnableFilterbankGRL(nn.Module):
    """
    Learnable Filterbank + Mode-Level Attention + Gradient Reversal Layer.

    See module docstring for full architecture and LOSO integrity guarantees.

    Args:
        in_channels:      C — EMG channels (8 for NinaPro DB2)
        num_classes:      number of gesture classes
        num_subjects:     number of TRAINING subjects in this LOSO fold (N-1).
                          Used only for the GRL subject classifier head.
                          **Create a fresh model instance per LOSO fold.**
        num_filters:      K — number of Sinc bandpass filters / frequency modes
        sinc_kernel_size: length of the Sinc FIR kernel (must be odd, e.g. 51)
        sample_rate:      EMG sampling rate in Hz
        min_freq:         lowest filter cutoff frequency in Hz
        max_freq:         highest filter cutoff frequency in Hz
        mode_dim:         D — mode representation dimension
        num_heads:        attention heads (must divide mode_dim evenly)
        gru_layers:       number of GRU layers in mode encoder
        dropout:          dropout probability
        grl_lambda:       initial GRL reversal strength; updated via set_grl_lambda()
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        num_subjects: int = 4,
        num_filters: int = 8,
        sinc_kernel_size: int = 51,
        sample_rate: int = 2000,
        min_freq: float = 5.0,
        max_freq: float = 500.0,
        mode_dim: int = 64,
        num_heads: int = 4,
        gru_layers: int = 1,
        dropout: float = 0.3,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_filters = num_filters
        self.in_channels = in_channels

        # K learnable bandpass Sinc filters; same filter bank applied to all channels.
        # Output layout: out[b, c*K+k, t] = channel c filtered by filter k.
        self.sinc = SincFilterbank(
            num_filters=num_filters,
            kernel_size=sinc_kernel_size,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            in_channels=in_channels,
        )

        # Shared-weight GRU encoder: same parameters for all K mode streams.
        self.mode_encoder = SharedModeGRUEncoder(
            in_channels=in_channels,
            hidden_dim=mode_dim,
            num_layers=gru_layers,
            dropout=dropout,
        )

        # Task query attends over K modes; learns which frequency bands matter.
        self.mode_attention = MultiHeadModeAttention(
            mode_dim=mode_dim,
            num_heads=num_heads,
            dropout=0.1,
        )

        # Gesture classification head.
        self.task_classifier = nn.Sequential(
            nn.LayerNorm(mode_dim),
            nn.Linear(mode_dim, mode_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mode_dim // 2, num_classes),
        )

        # GRL branch: subject classifier whose gradients are reversed upstream.
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.subject_classifier = nn.Sequential(
            nn.Linear(mode_dim, mode_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mode_dim // 2, num_subjects),
        )

    def set_grl_lambda(self, lambda_: float) -> None:
        """Update GRL reversal strength. Call once per epoch with DANN schedule."""
        self.grl.set_lambda(lambda_)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) — channel-standardized raw EMG windows
        Returns:
            task_logits:    (B, num_classes)  — gesture classification
            subject_logits: (B, num_subjects) — GRL branch; ignored at inference
            attn_weights:   (B, K)            — mode importance weights
        """
        B, C, T = x.shape
        K = self.num_filters

        # 1. Apply K Sinc filters to each channel independently → (B, C*K, T).
        filtered = self.sinc(x)  # (B, C*K, T)

        # 2. Reshape to (B, K, C, T): K parallel mode streams, each with all C channels.
        #    Layout from SincFilterbank: out[b, c*K+k, t] → reshape(B,C,K,T) → perm(B,K,C,T)
        filtered = filtered.reshape(B, C, K, T).permute(0, 2, 1, 3).contiguous()

        # 3. Apply shared GRU encoder to all K modes simultaneously.
        #    Flatten modes into the batch dimension so shared weights process all modes.
        filtered_flat = filtered.reshape(B * K, C, T)       # (B*K, C, T)
        mode_reprs_flat = self.mode_encoder(filtered_flat)   # (B*K, D)
        mode_reprs = mode_reprs_flat.reshape(B, K, -1)       # (B, K, D)

        # 4. Task query attends over K mode representations → fused context.
        fused, attn_weights = self.mode_attention(mode_reprs)  # (B,D), (B,K)

        # 5. Gesture classification.
        task_logits = self.task_classifier(fused)  # (B, num_classes)

        # 6. GRL branch: gradient reversed through the encoder → subject-invariant features.
        #    LOSO note: subject_logits are IGNORED at inference; no test-time effect.
        reversed_fused = self.grl(fused)
        subject_logits = self.subject_classifier(reversed_fused)  # (B, num_subjects)

        return task_logits, subject_logits, attn_weights
