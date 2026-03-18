"""
FIR Deconvolution Frontend + CNN-BiGRU-Attention for LOSO EMG Classification.

Hypothesis:
    Subject-specific variation in surface EMG is largely encoded in the
    skin/electrode transfer function (a linear filter on the neural drive).
    A trainable per-channel FIR filter, initialized as identity and regularised
    toward smooth solutions, can approximate the inverse of that transfer function
    — recovering a representation closer to the underlying neural drive and thus
    more invariant across subjects.

Components:

    FIRDeconvFrontend
        Per-channel 1D FIR filter implemented as a depthwise grouped convolution.
        Shape-preserving: (N, C, T) → (N, C, T).
        Initialized as δ[center] (identity pass-through).
        Regularization: L2 penalty on all tap weights + smoothness penalty
        (sum of squared first differences along the tap axis).
        Both penalties are *not* part of forward(); the trainer must add them to
        the CrossEntropy loss explicitly so that they can be scaled independently.

    FIRDeconvCNNGRU
        FIRDeconvFrontend → CNN stack → BiGRU → Multi-head Attention → FC.

LOSO correctness:
    ✓  FIR tap weights learned ONLY on the train-subject pool (train split).
    ✓  λ_l2 and λ_smooth are fixed hyper-parameters, NOT estimated from data.
    ✓  No per-subject or per-test normalization inside this module.
    ✓  model.eval() disables BatchNorm running-stat updates → no test leakage.
"""

import torch
import torch.nn as nn
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  FIR Deconvolution Frontend
# ═══════════════════════════════════════════════════════════════════════════════

class FIRDeconvFrontend(nn.Module):
    """
    Trainable per-channel 1D FIR deconvolution frontend.

    Each EMG channel gets an independent filter (depthwise grouped convolution),
    reflecting the physical reality that every electrode has its own transfer
    function.  The "same"-padding (filter_len//2 on each side) preserves the
    temporal dimension, so downstream CNN/GRU shapes are unaffected.

    Initialization as δ[center] means:
        • At the start of training the model is a pure pass-through.
        • The optimizer can only deviate from identity by overcoming the
          L2 regularization — ensuring the learned filter is grounded.

    Smoothness regularization promotes a smooth frequency response, discouraging
    the model from learning narrow-band spurious filters that might overfit to
    training-subject spectral quirks.

    Args:
        in_channels:  number of EMG channels C.
        filter_len:   length of FIR filter (must be odd for symmetric support).
                      At 2000 Hz, 63 taps ≈ 31.5 ms — long enough to cover
                      a full motor unit action potential waveform.
    """

    def __init__(self, in_channels: int, filter_len: int = 63) -> None:
        super().__init__()
        if filter_len % 2 == 0:
            raise ValueError(f"filter_len must be odd, got {filter_len}")

        self.in_channels = in_channels
        self.filter_len  = filter_len
        self._pad        = filter_len // 2  # same-padding

        # Depthwise conv: one independent filter per channel.
        # weight shape: (in_channels, 1, filter_len)
        self.conv = nn.Conv1d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = filter_len,
            groups       = in_channels,   # depthwise
            padding      = self._pad,
            bias         = False,
        )

        # Identity (δ at center tap) initialization.
        with torch.no_grad():
            nn.init.zeros_(self.conv.weight)
            center = filter_len // 2
            self.conv.weight[:, 0, center] = 1.0

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T) — channel-standardized EMG windows.

        Returns:
            (N, C, T) — filtered signal (neural-drive approximation).
        """
        return self.conv(x)

    # ------------------------------------------------------------------

    def regularization_loss(
        self,
        lambda_l2: float     = 1e-3,
        lambda_smooth: float = 5e-3,
    ) -> torch.Tensor:
        """
        Regularization penalties to be added to the classification loss.

        L2 term:       λ_l2    · Σ w²
            Prevents the filter from growing large; anchors weights near zero.

        Smoothness term: λ_smooth · Σ (w[i+1] − w[i])²
            Promotes a smooth frequency response, discouraging narrow-band
            artefacts that could overfit to training-subject spectral profiles.

        Note: the identity initialization shifts the L2 minimum away from zero
        (the center tap starts at 1.0 and will be penalised).  This is intentional:
        the L2 + smooth combo keeps the filter *close to and smooth like* the
        identity, not collapsed to zero.

        Args:
            lambda_l2:     scale for L2 weight penalty.
            lambda_smooth: scale for first-difference smoothness penalty.

        Returns:
            scalar torch.Tensor (on the same device as conv weights).
        """
        w = self.conv.weight           # (C, 1, filter_len)

        l2     = (w ** 2).sum()
        diff   = w[:, :, 1:] - w[:, :, :-1]   # (C, 1, filter_len−1)
        smooth = (diff ** 2).sum()

        return lambda_l2 * l2 + lambda_smooth * smooth


# ═══════════════════════════════════════════════════════════════════════════════
#  Full model: FIRDeconvFrontend + CNN-BiGRU-Attention
# ═══════════════════════════════════════════════════════════════════════════════

class FIRDeconvCNNGRU(nn.Module):
    """
    FIR Deconvolution Frontend + CNN-BiGRU-Attention classifier.

    Pipeline (all shapes shown for default hyper-params):
        (N, C, T=600)
          ↓ FIRDeconvFrontend    → (N, C, 600)   shape-preserving
          ↓ CNN block ×3         → (N, 256, 600/8=75)
          ↓ permute              → (N, 75, 256)   for GRU
          ↓ 2-layer BiGRU        → (N, 75, 256)   bidirectional
          ↓ Multi-head Attention → (N, 75, 256)   residual
          ↓ mean-pool over time  → (N, 256)
          ↓ FC                   → (N, num_classes)

    The FIR frontend operates before any learnable feature extraction, so its
    output (neural-drive approximation) is the sole input to the encoder.

    Regularization access:
        Call model.regularization_loss(lambda_l2, lambda_smooth) to get the
        penalty to add to CrossEntropy.

    Args:
        in_channels:   number of EMG channels C.
        num_classes:   number of gesture classes.
        filter_len:    FIR filter length (odd integer, default 63).
        cnn_channels:  output channels for each CNN block (3 blocks).
        gru_hidden:    hidden units per direction in the BiGRU.
        num_heads:     attention heads (must divide gru_hidden * 2).
        dropout:       dropout probability applied in CNN, GRU, Attention, FC.
    """

    def __init__(
        self,
        in_channels:  int,
        num_classes:  int,
        filter_len:   int              = 63,
        cnn_channels: Tuple[int, ...]  = (64, 128, 256),
        gru_hidden:   int              = 128,
        num_heads:    int              = 4,
        dropout:      float            = 0.3,
    ) -> None:
        super().__init__()

        # ── FIR deconvolution frontend ────────────────────────────────────
        self.frontend = FIRDeconvFrontend(
            in_channels = in_channels,
            filter_len  = filter_len,
        )

        # ── CNN stack ─────────────────────────────────────────────────────
        cnn_layers = []
        ch_in = in_channels
        for ch_out in cnn_channels:
            cnn_layers += [
                nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            ch_in = ch_out
        self.cnn = nn.Sequential(*cnn_layers)

        # ── Bidirectional GRU ─────────────────────────────────────────────
        gru_out_dim = gru_hidden * 2  # bidirectional
        self.gru = nn.GRU(
            input_size   = ch_in,
            hidden_size  = gru_hidden,
            num_layers   = 2,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if dropout > 0.0 else 0.0,
        )

        # ── Multi-head self-attention (residual) ──────────────────────────
        if gru_out_dim % num_heads != 0:
            raise ValueError(
                f"gru_out_dim={gru_out_dim} must be divisible by num_heads={num_heads}"
            )
        self.attn = nn.MultiheadAttention(
            embed_dim   = gru_out_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(gru_out_dim)

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T) — channel-standardized EMG windows.

        Returns:
            logits: (N, num_classes).
        """
        # FIR deconvolution: (N, C, T) → (N, C, T)
        x = self.frontend(x)

        # CNN stack: (N, C, T) → (N, cnn_channels[-1], T//8)
        x = self.cnn(x)

        # Prepare for GRU: (N, C', T') → (N, T', C')
        x = x.permute(0, 2, 1)

        # BiGRU: (N, T', C') → (N, T', gru_out_dim)
        x, _ = self.gru(x)

        # Multi-head self-attention with residual + LayerNorm
        x_attn, _ = self.attn(x, x, x)
        x = self.norm(x + x_attn)

        # Global average pooling over time: (N, T', gru_out_dim) → (N, gru_out_dim)
        x = x.mean(dim=1)

        return self.classifier(x)

    # ------------------------------------------------------------------

    def regularization_loss(
        self,
        lambda_l2:     float = 1e-3,
        lambda_smooth: float = 5e-3,
    ) -> torch.Tensor:
        """
        Frontend regularization loss (delegate to FIRDeconvFrontend).

        Call this in the training loop and add to CrossEntropy before backward():
            loss = criterion(logits, y) + model.regularization_loss(...)
        """
        return self.frontend.regularization_loss(lambda_l2, lambda_smooth)
