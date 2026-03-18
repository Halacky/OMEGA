"""
models/filterbank_vib_emg.py

Differentiable Mode Decomposition with per-mode Variational Information Bottleneck
(Filterbank-VIB-EMG).

Architecture overview
─────────────────────
  Input (B, C, T)  — channel-standardized raw EMG
    → SincFilterbank (K=6 learnable bandpass filters, shared across channels)
        → (B, C*K, T)
        → reshape → (B, K, C, T)   [K parallel mode streams]
    → SharedModeVIBEncoder (shared weights across modes)
        → μ (B*K, D_vib),  log_var (B*K, D_vib)    [VIB encoder outputs]
    → Reparameterization trick
        → z_k ~ N(μ_k, exp(log_var_k))   [training]  OR  z_k = μ_k  [eval]
    → reshape → (B, K, D_vib)
    → MultiHeadModeAttention  [task query → K mode z vectors]
        → fused: (B, D_vib)
        → attn_weights: (B, K)
    → task_classifier → gesture logits (B, num_classes)
    → GRL + shared_subject_classifier (per mode, shared weights)
        → subject_logits (B*K, num_subjects)

Loss components (training only)
────────────────────────────────
  L_task  = CrossEntropy(task_logits,       y_gesture)    [gesture classification]
  L_kl    = β · KL_per_mode                               [IB compression]
             where KL_per_mode = -0.5 Σ_d(1 + log_var_d - μ_d² - exp(log_var_d))
                                  averaged over (batch × K modes)
  L_adv   = α · CrossEntropy(subject_logits, y_subj_tiled) [per-mode subject adversarial]
             gradients reversed through GRL → encoder pushed toward subject-invariance
  L_total = L_task + L_kl + L_adv

LOSO integrity guarantees
──────────────────────────
  ✓ SincFilterbank: per-window per-channel linear filter — no cross-sample state.
  ✓ SharedModeVIBEncoder: GRU state is reset at every forward call (stateless).
  ✓ Reparameterization: noise ε is i.i.d. per sample — no information from other samples.
  ✓ At inference (model.eval()): reparameterize returns μ (deterministic). No sampling.
  ✓ KL drives encoder to compress modes; adversarial GRL removes subject-specific info.
  ✓ VIB generalizes to unseen subjects because gesture-discriminative information is
    preserved while subject-specific variation is squeezed out during training.
  ✓ num_subjects in subject_classifier = |train_subjects| for this fold; fresh model
    instantiated per fold. Subject classifier logits IGNORED at inference.
  ✓ Channel standardization (mean/std) computed from TRAIN windows only; applied as
    a fixed transform before model input.

References
──────────
  Alemi et al., "Deep Variational Information Bottleneck," ICLR 2017.
  Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR 2016.
  Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with SincNet," SLT 2018.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sinc_pcen_cnn_gru import SincFilterbank
from models.learnable_filterbank_grl import GradientReversalLayer


# ─────────────────── Shared Mode VIB Encoder ─────────────────────────────────

class SharedModeVIBEncoder(nn.Module):
    """
    Shared-weight temporal encoder that produces VIB parameters (μ, log_var) per mode.

    Applied identically to ALL K mode streams using a single set of weights.
    Frequency-specific importance is captured downstream by MultiHeadModeAttention;
    this encoder learns a universal per-mode temporal pattern extractor.

    Input:  (B*K, C, T)   — K mode streams flattened into the batch dimension
    Output: mu      (B*K, vib_dim)   — posterior mean
            log_var (B*K, vib_dim)   — posterior log-variance (log σ²)

    Architecture:
        Conv1d(C → hidden, k=5) → BN → ReLU
        Conv1d(hidden → hidden, k=3) → BN → ReLU   [local temporal features]
        GRU(hidden, hidden)                         [temporal sequence modelling]
        Temporal attention pooling                  [weighted sum over time]
        fc_mu    → μ (B*K, vib_dim)
        fc_log_var → log_var (B*K, vib_dim)         [clamped to [-6, 6] in forward]

    LOSO note: GRU state is not carried across windows (stateless forward).
               BatchNorm running statistics are updated from training data only.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        vib_dim: int,
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
        self.fc_mu      = nn.Linear(hidden_dim, vib_dim)
        self.fc_log_var = nn.Linear(hidden_dim, vib_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B*K, C, T)
        Returns:
            mu:      (B*K, vib_dim)
            log_var: (B*K, vib_dim)  clamped to [-6, 6]
        """
        h = self.cnn_proj(x)                              # (B*K, hidden, T)
        h = h.transpose(1, 2)                             # (B*K, T, hidden)
        gru_out, _ = self.gru(h)                          # (B*K, T, hidden)
        w = torch.softmax(self.attn_fc(gru_out), dim=1)  # (B*K, T, 1)
        pooled = (w * gru_out).sum(dim=1)                 # (B*K, hidden)

        mu      = self.fc_mu(pooled)                      # (B*K, vib_dim)
        log_var = self.fc_log_var(pooled).clamp(-6.0, 6.0)  # (B*K, vib_dim)
        return mu, log_var


# ─────────────────── Multi-Head Mode Attention ───────────────────────────────

class MultiHeadModeAttention(nn.Module):
    """
    Multi-head attention: a learnable task query attends over K mode z-vectors.

    The task query is a learned parameter (independent of input) asking:
    "Which frequency-band mode carries the most gesture-discriminative information?"

    Q = W_q · task_query      (B, 1, D)   [shared across batch]
    K = W_k · z_modes         (B, K, D)
    V = W_v · z_modes         (B, K, D)
    out = softmax(Q Kᵀ / √d) · V  → (B, 1, D)
    fused = out.squeeze(1)         → (B, D)
    attn_weights = softmax weights → (B, K)   [mode importance, interpretable]
    """

    def __init__(self, vib_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert vib_dim % num_heads == 0, (
            f"vib_dim ({vib_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=vib_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.task_query = nn.Parameter(torch.randn(1, 1, vib_dim) * 0.02)

    def forward(
        self, z_modes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_modes: (B, K, D)
        Returns:
            fused:        (B, D)
            attn_weights: (B, K)
        """
        B = z_modes.size(0)
        q = self.task_query.expand(B, -1, -1)               # (B, 1, D)
        out, attn_w = self.mha(q, z_modes, z_modes, need_weights=True)
        # out: (B, 1, D),  attn_w: (B, 1, K) averaged across heads
        return out.squeeze(1), attn_w.squeeze(1)             # (B, D), (B, K)


# ─────────────────── Main Model ──────────────────────────────────────────────

class FilterbankVIBEMG(nn.Module):
    """
    Differentiable Mode Decomposition with per-mode Variational Information Bottleneck.

    Training returns:
        task_logits:       (B, num_classes)
        kl_loss:           scalar — KL(q(z|mode) ‖ N(0,I)) summed over dims, averaged over (B·K)
        subj_logits_flat:  (B*K, num_subjects) — GRL adversarial branch (per mode, shared weights)
        attn_weights:      (B, K)

    Evaluation (model.eval()):
        Same signature, but:
        • reparameterize returns μ (no noise) → deterministic, LOSO-safe
        • kl_loss is still returned (for logging) but ignored in evaluation
        • subj_logits_flat is returned but IGNORED by the experiment's evaluation code

    Args:
        in_channels:      C — EMG channels (8 for NinaPro DB2)
        num_classes:      number of gesture classes
        num_subjects:     number of TRAINING subjects in this LOSO fold (N-1).
                          Subject classifier head size. Create a fresh model per fold.
        num_filters:      K — number of Sinc bandpass filters / frequency modes  (default 6)
        sinc_kernel_size: sinc FIR kernel length (must be odd)
        sample_rate:      EMG sampling rate in Hz
        min_freq:         lowest filter cutoff in Hz
        max_freq:         highest filter cutoff in Hz
        hidden_dim:       GRU / CNN hidden size inside VIB encoder
        vib_dim:          D — VIB latent dimension per mode
        num_heads:        attention heads (must divide vib_dim evenly)
        gru_layers:       GRU layers in mode encoder
        dropout:          dropout probability
        grl_lambda:       initial GRL reversal strength (updated via set_grl_lambda())
    """

    def __init__(
        self,
        in_channels:      int   = 8,
        num_classes:      int   = 10,
        num_subjects:     int   = 4,
        num_filters:      int   = 6,
        sinc_kernel_size: int   = 51,
        sample_rate:      int   = 2000,
        min_freq:         float = 5.0,
        max_freq:         float = 500.0,
        hidden_dim:       int   = 64,
        vib_dim:          int   = 32,
        num_heads:        int   = 4,
        gru_layers:       int   = 1,
        dropout:          float = 0.3,
        grl_lambda:       float = 0.0,
    ) -> None:
        super().__init__()
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.vib_dim     = vib_dim

        # ── Learnable Sinc filterbank ─────────────────────────────────────
        # Same K filters applied independently to each of C channels.
        # Output: (B, C, T) → (B, C*K, T)
        self.sinc = SincFilterbank(
            num_filters   = num_filters,
            kernel_size   = sinc_kernel_size,
            sample_rate   = sample_rate,
            min_freq      = min_freq,
            max_freq      = max_freq,
            in_channels   = in_channels,
        )

        # ── Per-mode VIB encoder (shared weights across K modes) ──────────
        # Applied to each mode stream (B*K, C, T) → μ, log_var (B*K, vib_dim)
        self.mode_vib_encoder = SharedModeVIBEncoder(
            in_channels = in_channels,
            hidden_dim  = hidden_dim,
            vib_dim     = vib_dim,
            num_layers  = gru_layers,
            dropout     = dropout,
        )

        # ── Multi-head mode attention ─────────────────────────────────────
        # Task query attends over K mode z-vectors → fused (B, vib_dim)
        self.mode_attention = MultiHeadModeAttention(
            vib_dim   = vib_dim,
            num_heads = num_heads,
            dropout   = 0.1,
        )

        # ── Gesture classification head ───────────────────────────────────
        self.task_classifier = nn.Sequential(
            nn.LayerNorm(vib_dim),
            nn.Linear(vib_dim, vib_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vib_dim * 2, num_classes),
        )

        # ── GRL + shared per-mode subject classifier ──────────────────────
        # GRL reverses gradients → encoder pushed toward subject-invariance
        # SAME weights applied per mode (flattened into batch): (B*K, vib_dim)
        # At inference: GRL has no effect (no gradient flow). Subject logits ignored.
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.subject_classifier = nn.Sequential(
            nn.Linear(vib_dim, vib_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(vib_dim, num_subjects),
        )

    def set_grl_lambda(self, lambda_: float) -> None:
        """Update GRL reversal strength. Call once per epoch (DANN warm-up schedule)."""
        self.grl.set_lambda(lambda_)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick.

        Training: z = μ + σ * ε,  ε ~ N(0,I)  — stochastic, enables gradient flow
        Eval:     z = μ            — deterministic, LOSO-safe (no subject-specific noise)

        LOSO note: ε is sampled i.i.d. per window — no information leaks across samples.
        Using μ at eval time means the model behaves as a deterministic function of input.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)   # σ = exp(½ log σ²)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def compute_kl_loss(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence KL(N(μ, σ²) ‖ N(0, I)) per element, averaged over (B·K) and D.

            KL = -½ Σ_d (1 + log_var_d - μ_d² - exp(log_var_d))

        Shapes: mu, log_var — (B*K, D)
        Returns: scalar
        """
        kl_per = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())  # (B*K, D)
        return kl_per.sum(dim=-1).mean()                              # scalar

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) — channel-standardized raw EMG, transposed from (N, T, C)

        Returns:
            task_logits:      (B, num_classes)  — gesture logits
            kl_loss:          scalar             — IB compression term (0 during eval)
            subj_logits_flat: (B*K, num_subjects) — per-mode adversarial (IGNORED at eval)
            attn_weights:     (B, K)             — mode importance (interpretable)
        """
        B, C, T = x.shape
        K = self.num_filters

        # ── 1. Learnable filterbank: (B, C, T) → (B, C*K, T) ────────────
        filtered = self.sinc(x)                                    # (B, C*K, T)

        # Reshape to (B, K, C, T): layout is out[b, c*K+k, t], so:
        #   reshape(B, C, K, T) → permute(0,2,1,3) → (B, K, C, T)
        filtered = filtered.reshape(B, C, K, T).permute(0, 2, 1, 3).contiguous()

        # ── 2. VIB encoding (shared weights, flattened batch) ─────────────
        filtered_flat = filtered.reshape(B * K, C, T)              # (B*K, C, T)
        mu_flat, log_var_flat = self.mode_vib_encoder(filtered_flat)  # (B*K, D) each

        # ── 3. KL loss (computed before reparameterization for stability) ─
        kl_loss = self.compute_kl_loss(mu_flat, log_var_flat)      # scalar

        # ── 4. Reparameterize → z ─────────────────────────────────────────
        z_flat = self.reparameterize(mu_flat, log_var_flat)        # (B*K, D)
        z_modes = z_flat.reshape(B, K, self.vib_dim)               # (B, K, D)

        # ── 5. Mode attention fusion ──────────────────────────────────────
        fused, attn_weights = self.mode_attention(z_modes)         # (B,D), (B,K)

        # ── 6. Gesture classification ─────────────────────────────────────
        task_logits = self.task_classifier(fused)                  # (B, num_classes)

        # ── 7. Per-mode adversarial (GRL, shared weights) ─────────────────
        # GRL reverses gradients through the VIB encoder toward subject-invariance.
        # Applied to ALL mode z-vectors simultaneously (flattened as batch).
        # LOSO note: at eval, no gradient flows, subject logits are discarded.
        reversed_z = self.grl(z_flat)                              # (B*K, D), grad reversed
        subj_logits_flat = self.subject_classifier(reversed_z)     # (B*K, num_subjects)

        return task_logits, kl_loss, subj_logits_flat, attn_weights
