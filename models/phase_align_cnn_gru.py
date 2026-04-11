"""
Phase-Aligned CNN-BiGRU-Attention for cross-subject EMG classification.

This module contains the neural network backbone used in Experiment 66
(temporal phase alignment via TKEO onset detection).

The model is structurally identical to the CNN-BiGRU-Attention encoder
inside FIRDeconvCNNGRU (exp_65) but WITHOUT the FIR deconvolution frontend.
Phase alignment is handled upstream in PhaseAlignTrainer as a deterministic
numpy preprocessing step, so this module is a pure discriminative encoder.

Pipeline (default hyper-params, T=600 after phase alignment):
    (N, C, T)
      ↓ CNN block ×3         → (N, 256, T//8 = 75)
      ↓ permute              → (N, 75, 256)   for GRU input
      ↓ 2-layer BiGRU        → (N, 75, 256)   bidirectional
      ↓ Multi-head Attention → (N, 75, 256)   residual + LayerNorm
      ↓ mean-pool over time  → (N, 256)
      ↓ FC (256→128→K)       → (N, K)

LOSO correctness:
    ✓  All weights learned only from the train-subject pool.
    ✓  No per-subject or per-fold parameters anywhere in this module.
    ✓  model.eval() disables BatchNorm running-stat updates → no test leakage.
    ✓  Phase alignment (upstream) is purely per-window and unsupervised.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PhaseAlignCNNGRU(nn.Module):
    """
    CNN-BiGRU-Attention classifier for phase-aligned EMG windows.

    Receives (N, C, T) windows that have been:
        1. Phase-aligned via TKEO onset/offset detection (in PhaseAlignTrainer).
        2. Per-channel standardized with training-data statistics.

    No preprocessing is performed inside the model — it is a pure function
    from normalized, aligned EMG to class logits.

    Args:
        in_channels:  number of EMG channels C (typically 8).
        num_classes:  number of gesture classes.
        cnn_channels: output channels for each of the 3 CNN blocks.
        gru_hidden:   hidden units per direction in the 2-layer BiGRU.
                      gru_out_dim = gru_hidden * 2 (bidirectional).
        num_heads:    attention heads; must divide gru_out_dim evenly.
        dropout:      dropout probability applied in CNN, GRU, Attention, FC.
    """

    def __init__(
        self,
        in_channels:  int,
        num_classes:  int,
        cnn_channels: Tuple[int, ...] = (64, 128, 256),
        gru_hidden:   int   = 128,
        num_heads:    int   = 4,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()

        # ── CNN stack: (N, C, T) → (N, cnn_channels[-1], T // 2^n_blocks) ──
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
        gru_out_dim = gru_hidden * 2   # 256 with default gru_hidden=128
        self.gru = nn.GRU(
            input_size    = ch_in,
            hidden_size   = gru_hidden,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if dropout > 0.0 else 0.0,
        )

        # ── Multi-head self-attention (residual + LayerNorm) ──────────────
        if gru_out_dim % num_heads != 0:
            raise ValueError(
                f"gru_out_dim={gru_out_dim} must be divisible by "
                f"num_heads={num_heads}."
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T) — channel-standardized, phase-aligned EMG windows.

        Returns:
            logits: (N, num_classes).
        """
        # CNN: (N, C, T) → (N, cnn_channels[-1], T//2^3)
        x = self.cnn(x)

        # Prepare for GRU: (N, C', T') → (N, T', C')
        x = x.permute(0, 2, 1)

        # BiGRU: (N, T', C') → (N, T', gru_out_dim)
        x, _ = self.gru(x)

        # Multi-head self-attention with residual + LayerNorm
        x_attn, _ = self.attn(x, x, x)
        x = self.norm(x + x_attn)

        # Global average pooling over time: (N, T', D) → (N, D)
        x = x.mean(dim=1)

        return self.classifier(x)
