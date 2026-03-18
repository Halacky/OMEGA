"""
Hierarchical Conditional β-VAE with Learnable Frequency Decomposition.

Hypothesis (Exp 108):
    Three-level hierarchical disentanglement — frequency → channel → amplitude —
    in a single end-to-end architecture via conditional β-VAE.

Architecture
────────────
    Level 1 — Learnable Mode Decomposition (UVMD):
        Differentiable UVMDBlock decomposes each channel into K=4 frequency modes.
        Parameters (α, τ, ω) are learned end-to-end via backpropagation.
        Reuses UVMDBlock from exp_93 (models/uvmd_classifier.py).
        10× faster than classical VMD at comparable separation quality.

    Level 3 — Soft AGC per Mode (amplitude normalization):
        K separate SoftAGCLayer instances (one per mode), each parameterised by
        in_channels learnable scalars.  Applied to each mode's (B, C, T) signal
        BEFORE per-channel encoding.  Removes domain-specific amplitude variation
        without destroying gesture-discriminative amplitude patterns.
        Reuses SoftAGCLayer from exp_76 (models/soft_agc_cnn_gru.py).

    Level 2 — Per-(mode, channel) β-VAE Disentanglement:
        For each of the K×C (mode, channel) pairs, a shared β-VAE encoder
        produces two latent variables:
          z_content  (B, C, content_dim) — gesture-informative latent
          z_style    (B, C, style_dim)   — subject-specific latent
        Reparameterization trick at training time; deterministic μ at test time.
        Unlike exp_31 (global disentanglement): K×C separate disentanglement axes.
        Unlike exp_106 (C axes, no frequency split): K×C = 4×8 = 32 axes.

    Classifier:
        All z_content from K modes and C channels:
          → stack: (B, K, C, content_dim)
          → reshape: (B, K*C, content_dim)
          → ECAPA-style Attentive Statistics Pooling: (B, 2*content_dim)
          → MLP classification head: (B, num_classes)

Loss (computed in trainer, not in model)
─────────────────────────────────────────
    L = L_class
      + β_content × mean_KL(z_content)          -- content regularisation
      + β_style   × mean_KL(z_style)            -- style regularisation
      + λ_mi      × mean_MI(z_content, z_style) -- content/style independence
      + λ_overlap × spectral_overlap_penalty     -- mode frequency separation

    KL(z) = −½ Σ_d (1 + log σ²_d − μ²_d − σ²_d)   (per latent dimension d)
    MI term = distance correlation (differentiable mutual-information proxy)

LOSO safety
───────────
    ✓  UVMDBlock: deterministic function of (input, params) — no per-subject state.
    ✓  SoftAGCLayer: EMA computed fresh each forward call — no cross-window state.
    ✓  β-VAE encoder: stateless.  Training uses reparameterisation (stochastic).
       Eval uses z = μ only (deterministic) — no test-subject statistics needed.
    ✓  All parameters (UVMD, SoftAGC, CNN, VAE heads, ASP, classifier) trained
       exclusively on train-subject windows.  Frozen at test time (model.eval()).
    ✓  BatchNorm running statistics accumulated from training samples only.
    ✓  No test-subject information is used anywhere in the forward or backward pass.
    ✓  AttentiveStatsPooling: purely a function of its input — no stored state.

References
──────────
    Dragomiretskiy & Zosso (2014) — VMD.
    Chen et al. (2021) — Algorithm Unrolling.
    Kingma & Welling (2014) — VAE.
    Higgins et al. (2017) — β-VAE.
    Desplanques et al. (2020) — ECAPA-TDNN (Attentive Statistics Pooling).
    exp_93 / models/uvmd_classifier.py — UVMDBlock source.
    exp_76 / models/soft_agc_cnn_gru.py — SoftAGCLayer source.
    exp_106 / models/channel_contrastive_disentangled.py — per-channel MI loss.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.uvmd_classifier import UVMDBlock
from models.soft_agc_cnn_gru import SoftAGCLayer


# ═══════════════════════════════════════════════════════════════════════════
# Distance correlation (MI proxy) — duplicated locally to avoid circular
# imports from channel_contrastive_disentangled.py
# ═══════════════════════════════════════════════════════════════════════════

def _distance_correlation(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Distance correlation between two (B, D) tensors.

    Differentiable proxy for mutual information.  Returns value in [0, 1].
    Called once per (mode, channel) pair in the trainer.

    LOSO safety: purely a function of its arguments — no stored state.
    """
    n = z1.size(0)
    if n < 4:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    a = torch.cdist(z1, z1, p=2)
    b = torch.cdist(z2, z2, p=2)

    a_row = a.mean(dim=1, keepdim=True)
    a_col = a.mean(dim=0, keepdim=True)
    a_grand = a.mean()
    A = a - a_row - a_col + a_grand

    b_row = b.mean(dim=1, keepdim=True)
    b_col = b.mean(dim=0, keepdim=True)
    b_grand = b.mean()
    B_ = b - b_row - b_col + b_grand

    dcov2 = (A * B_).mean()
    dvar_a2 = (A * A).mean()
    dvar_b2 = (B_ * B_).mean()
    denom = torch.sqrt(dvar_a2 * dvar_b2 + 1e-12)
    dcor = torch.sqrt(torch.clamp(dcov2, min=0.0) / denom)
    return dcor


# ═══════════════════════════════════════════════════════════════════════════
# Per-channel CNN backbone
# ═══════════════════════════════════════════════════════════════════════════

class _PerChannelCNN(nn.Module):
    """
    Shared 1-channel 1-D CNN backbone applied independently to each EMG channel.

    The single-channel design is intentional: after UVMD decomposition and
    per-mode Soft AGC, each (mode, channel) signal is encoded in isolation,
    preserving the per-channel disentanglement philosophy of exp_106.

    Weight sharing across both modes and channels (not separate weights per pair)
    keeps the total parameter count manageable (K×C = 32 concurrent evaluations
    with a single set of CNN weights). The mode and channel specificity is already
    provided upstream by the UVMD decomposition and mode-specific SoftAGC.

    Input : (B*K*C, 1, T) — all (batch, mode, channel) triples flattened
    Output: (B*K*C, output_dim) — global-average-pooled representation

    LOSO safety: no per-subject or per-window state.  BatchNorm running stats
    are frozen at eval time via model.eval().
    """

    def __init__(
        self,
        cnn_channels: Tuple[int, ...] = (32, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_ch = 1  # single-channel input
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.5),
            ])
            prev_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.output_dim: int = cnn_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*K*C, 1, T)
        Returns:
            (B*K*C, output_dim) — global average pooled
        """
        h = self.cnn(x)       # (B*K*C, output_dim, T')
        return h.mean(dim=2)  # (B*K*C, output_dim) — temporal average


# ═══════════════════════════════════════════════════════════════════════════
# Attentive Statistics Pooling (ECAPA-style)
# ═══════════════════════════════════════════════════════════════════════════

class AttentiveStatsPooling(nn.Module):
    """
    ECAPA-style Attentive Statistics Pooling over a sequence of feature vectors.

    Computes soft-attention weights over the N "position" dimension, then
    returns the concatenation of the attention-weighted mean and the
    corresponding weighted standard deviation.

        Input : (B, N, D) — N vectors of dimension D per batch element
        Output: (B, 2D)   — [weighted_mean; weighted_std]

    The attention is computed by a small 2-layer MLP applied independently to
    each of the N vectors.  There is no recurrence and no cross-sample state.

    Interpretation here: N = K × C (mode × channel pairs), D = content_dim.
    Forcing the classifier to attend to the most informative (mode, channel)
    combinations rather than treating all 32 pairs equally.

    LOSO safety:
        · No stored running statistics — purely a deterministic function of input.
        · attention weights computed independently per batch element.
        · Weights frozen at test time via model.eval().
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, 2D) — [attention-weighted mean; attention-weighted std]
        """
        # x: (B, N, D)
        w = torch.softmax(self.attention(x), dim=1)  # (B, N, 1)
        mean = (w * x).sum(dim=1)                   # (B, D)
        # Numerically stable variance: E[(x - mean)^2] using attention weights
        diff_sq = (x - mean.unsqueeze(1)) ** 2      # (B, N, D)
        var = (w * diff_sq).sum(dim=1)              # (B, D)
        std = torch.sqrt(var + 1e-6)                # (B, D)
        return torch.cat([mean, std], dim=1)        # (B, 2D)


# ═══════════════════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalBetaVAEUVMD(nn.Module):
    """
    Hierarchical Conditional β-VAE with Learnable Frequency Decomposition.

    Three-level invariance for cross-subject EMG gesture recognition:
      Level 1 (UVMD)    — frequency-domain disentanglement
      Level 3 (SoftAGC) — amplitude-domain normalisation per mode
      Level 2 (β-VAE)   — content/style split per (mode, channel) pair

    Input/output convention:
        Input  x: (B, C, T)  — channel-first (trainer transposes from (N,T,C))
        Output  : dict (training) or (B, num_classes) tensor (eval)

    Subject-independence guarantee:
        All parameters are shared across subjects.  The forward pass is a
        deterministic function of (x, parameters) — identical computation for
        every subject.  At eval time reparameterisation uses μ only.

    Parameters
    ──────────
    K              : number of VMD modes (frequency bands)
    L              : number of unrolled ADMM iterations in UVMDBlock
    in_channels    : number of EMG channels (C)
    num_classes    : number of gesture classes
    content_dim    : dimension of z_content per (mode, channel) pair
    style_dim      : dimension of z_style per (mode, channel) pair
    cnn_channels   : hidden channel widths of the per-channel CNN backbone
    asp_bottleneck : attention bottleneck dimension in AttentiveStatsPooling
    clf_hidden     : hidden size of the MLP classification head
    dropout        : dropout probability
    alpha_init     : initial UVMD bandwidth constraint (classic VMD: 2000)
    tau_init       : initial UVMD dual step (near 0 = noise-free regime)
    agc_ema_length : SoftAGC causal EMA kernel length (in samples)
    agc_delta      : SoftAGC fixed additive stabiliser (NOT learned)
    """

    def __init__(
        self,
        K: int = 4,
        L: int = 8,
        in_channels: int = 8,
        num_classes: int = 10,
        content_dim: int = 16,
        style_dim: int = 8,
        cnn_channels: Tuple[int, ...] = (32, 64),
        asp_bottleneck: int = 64,
        clf_hidden: int = 128,
        dropout: float = 0.3,
        alpha_init: float = 2000.0,
        tau_init: float = 0.01,
        agc_ema_length: int = 100,
        agc_delta: float = 0.1,
    ) -> None:
        super().__init__()
        self.K = K
        self.C = in_channels
        self.content_dim = content_dim
        self.style_dim = style_dim

        # ── Level 1: Learnable frequency decomposition ───────────────────
        # UVMDBlock: shared across all subjects and samples.
        # Input: (B, T, C), Output: (B, K, T, C)
        # Parameters: log_alpha (L, K), raw_tau (L,), raw_omega (K,)
        # LOSO safety: trained on train subjects only; deterministic at test time.
        self.uvmd = UVMDBlock(K=K, L=L, alpha_init=alpha_init, tau_init=tau_init)

        # ── Level 3: Per-mode Soft AGC ────────────────────────────────────
        # K separate SoftAGCLayer instances (one per frequency mode).
        # Each has in_channels learnable parameters (alpha_raw, log_s per channel).
        # Applied to (B, C, T) mode signals BEFORE per-channel encoding.
        # LOSO safety: EMA recomputed from scratch each forward call.
        self.soft_agc = nn.ModuleList([
            SoftAGCLayer(
                num_channels=in_channels,
                ema_kernel_length=agc_ema_length,
                delta=agc_delta,
            )
            for _ in range(K)
        ])

        # ── Level 2a: Per-channel CNN backbone (shared across modes & channels) ─
        # Processes K*C single-channel signals simultaneously via batched dim.
        # Weight sharing keeps parameter count tractable; mode/channel specificity
        # is provided by UVMD decomposition and mode-specific SoftAGC upstream.
        # LOSO safety: BatchNorm frozen at eval; no per-sample state.
        self.per_ch_encoder = _PerChannelCNN(
            cnn_channels=cnn_channels,
            dropout=dropout,
        )
        enc_dim = self.per_ch_encoder.output_dim

        # ── Level 2b: β-VAE heads (shared across modes & channels) ───────
        # Content: maps enc_dim → (mu_content, logvar_content) ∈ R^{2*content_dim}
        # Style:   maps enc_dim → (mu_style,   logvar_style)   ∈ R^{2*style_dim}
        # Shared weights: the diversity comes from the different (mode, channel)
        # inputs, not from separate head weights.  This reduces overfitting risk.
        # LOSO safety: linear projections — no stored state.
        self.content_mu = nn.Linear(enc_dim, content_dim)
        self.content_lv = nn.Linear(enc_dim, content_dim)
        self.style_mu   = nn.Linear(enc_dim, style_dim)
        self.style_lv   = nn.Linear(enc_dim, style_dim)

        # ── Attentive Statistics Pooling over K*C z_content vectors ──────
        # Aggregates 32 content representations into a fixed-size descriptor.
        # LOSO safety: stateless, no running statistics.
        self.asp = AttentiveStatsPooling(
            input_dim=content_dim,
            bottleneck_dim=asp_bottleneck,
        )

        # ── Classification head ───────────────────────────────────────────
        # Input: (B, 2*content_dim) from ASP [mean || std concatenation]
        asp_out_dim = 2 * content_dim
        self.classifier = nn.Sequential(
            nn.Linear(asp_out_dim, clf_hidden),
            nn.BatchNorm1d(clf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, num_classes),
        )

    # ── Reparameterisation ───────────────────────────────────────────────────

    def _reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterisation trick for the β-VAE.

        Training: z = μ + ε · exp(½ logvar),  ε ~ N(0, I)
        Eval    : z = μ  (deterministic — no sampling noise at test time)

        LOSO safety:
            · Training samples ε from a fresh N(0, I) on each call — no stored seed.
            · Eval returns μ: z depends only on the input x and model weights,
              not on any test-subject statistics or runtime state.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  # drawn fresh, discarded after call
            return mu + eps * std
        return mu  # deterministic at test time — LOSO safe

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> Union[Dict[str, object], torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C, T) — channel-standardised EMG windows.
                           Channel standardisation is applied by the Trainer
                           using TRAIN-SUBJECT statistics only.

        Returns:
            Training mode (self.training == True):
                dict with keys:
                    "logits"       : (B, num_classes) — gesture predictions
                    "mu_content"   : List[K] of (B, C, content_dim)
                    "lv_content"   : List[K] of (B, C, content_dim)
                    "mu_style"     : List[K] of (B, C, style_dim)
                    "lv_style"     : List[K] of (B, C, style_dim)
                    "z_content"    : List[K] of (B, C, content_dim) — sampled
                    "z_style"      : List[K] of (B, C, style_dim)   — sampled
            Eval mode:
                (B, num_classes) gesture logits — z = μ (deterministic)

        Shape invariants (verified at each stage):
            x              : (B, C, T)
            x_tc           : (B, T, C)   — permuted for UVMDBlock
            modes          : (B, K, T, C)
            mode_k         : (B, C, T)   — for SoftAGC
            mode_k_norm    : (B, C, T)
            x_ch           : (B*C, 1, T) — for per-channel CNN
            enc_kc         : (B, C, enc_dim)
            mu_cont_k      : (B, C, content_dim)
            z_content_all  : (B, K, C, content_dim)
            z_content_seq  : (B, K*C, content_dim)
            pooled         : (B, 2*content_dim)
            logits         : (B, num_classes)
        """
        B, C, T = x.shape
        K = self.K

        # ── Level 1: UVMD frequency decomposition ────────────────────────
        # UVMDBlock expects (B, T, C); our input is (B, C, T).
        x_tc = x.permute(0, 2, 1)       # (B, T, C)
        modes = self.uvmd(x_tc)          # (B, K, T, C)

        # Accumulated per-mode outputs (filled in the loop below)
        all_mu_content: List[torch.Tensor] = []
        all_lv_content: List[torch.Tensor] = []
        all_mu_style:   List[torch.Tensor] = []
        all_lv_style:   List[torch.Tensor] = []
        all_z_content:  List[torch.Tensor] = []
        all_z_style:    List[torch.Tensor] = []

        for k in range(K):
            # mode_k: (B, T, C) → (B, C, T) for SoftAGC and per-channel CNN
            mode_k = modes[:, k].permute(0, 2, 1)        # (B, C, T)

            # ── Level 3: Soft AGC amplitude normalisation ─────────────
            # Mode-specific SoftAGC: learnable EMA-based gain control.
            # Removes subject-specific amplitude drift without destroying
            # gesture-discriminative amplitude structure (unlike PCEN).
            # EMA is computed fresh — no cross-window state.
            mode_k_norm = self.soft_agc[k](mode_k)       # (B, C, T)

            # ── Level 2a: Per-channel CNN encoding ───────────────────
            # Flatten (B, C, T) → (B*C, 1, T) to process all channels
            # with the shared single-channel CNN in one batched call.
            x_ch = mode_k_norm.reshape(B * C, 1, T)      # (B*C, 1, T)
            enc_flat = self.per_ch_encoder(x_ch)          # (B*C, enc_dim)
            enc_kc = enc_flat.reshape(B, C, -1)           # (B, C, enc_dim)

            # ── Level 2b: β-VAE heads ─────────────────────────────────
            # Apply shared linear projections across all (B*C) representations.
            enc_bc = enc_kc.reshape(B * C, -1)            # (B*C, enc_dim)

            mu_cont  = self.content_mu(enc_bc).reshape(B, C, self.content_dim)
            lv_cont  = self.content_lv(enc_bc).reshape(B, C, self.content_dim)
            mu_sty   = self.style_mu(enc_bc).reshape(B, C, self.style_dim)
            lv_sty   = self.style_lv(enc_bc).reshape(B, C, self.style_dim)

            # Clamp logvar to prevent numerical instability in exp(logvar).
            # Range [-10, 10] → σ ∈ [exp(-5), exp(5)] ≈ [0.007, 148].
            lv_cont = lv_cont.clamp(-10.0, 10.0)
            lv_sty  = lv_sty.clamp(-10.0, 10.0)

            # Reparameterise: stochastic during training, deterministic at test
            z_cont = self._reparameterise(mu_cont, lv_cont)  # (B, C, content_dim)
            z_sty  = self._reparameterise(mu_sty,  lv_sty)   # (B, C, style_dim)

            all_mu_content.append(mu_cont)
            all_lv_content.append(lv_cont)
            all_mu_style.append(mu_sty)
            all_lv_style.append(lv_sty)
            all_z_content.append(z_cont)
            all_z_style.append(z_sty)

        # ── Aggregate z_content: (B, K, C, content_dim) ──────────────────
        z_content_all = torch.stack(all_z_content, dim=1)   # (B, K, C, content_dim)
        # Flatten K and C into a single "sequence" dimension for ASP
        z_content_seq = z_content_all.reshape(B, K * C, self.content_dim)

        # ── ECAPA Attentive Statistics Pooling ────────────────────────────
        # Pooling over K*C = 32 (mode, channel) content representations.
        # Learned attention weights identify the most discriminative pairs.
        pooled = self.asp(z_content_seq)                    # (B, 2*content_dim)

        # ── Classification ────────────────────────────────────────────────
        logits = self.classifier(pooled)                    # (B, num_classes)

        if self.training:
            return {
                "logits":      logits,
                "mu_content":  all_mu_content,   # List[K] of (B, C, content_dim)
                "lv_content":  all_lv_content,   # List[K] of (B, C, content_dim)
                "mu_style":    all_mu_style,     # List[K] of (B, C, style_dim)
                "lv_style":    all_lv_style,     # List[K] of (B, C, style_dim)
                "z_content":   all_z_content,    # List[K] of (B, C, content_dim)
                "z_style":     all_z_style,      # List[K] of (B, C, style_dim)
            }

        # Eval: return only logits (z = μ, deterministic — LOSO safe)
        return logits

    # ── Convenience helpers ──────────────────────────────────────────────────

    def spectral_overlap_penalty(self, sigma: float = 0.05) -> torch.Tensor:
        """Spectral overlap penalty from UVMDBlock (mode centre-freq clustering)."""
        return self.uvmd.spectral_overlap_penalty(sigma=sigma)

    def get_learned_uvmd_params(self) -> Dict[str, object]:
        """Return UVMD learnable parameters for post-training analysis."""
        with torch.no_grad():
            return {
                "omega_k":  self.uvmd.omega.cpu().numpy().tolist(),
                "alpha_lk": self.uvmd.alpha.cpu().numpy().tolist(),
                "tau_l":    self.uvmd.tau.cpu().numpy().tolist(),
            }


# ═══════════════════════════════════════════════════════════════════════════
# Loss utilities (used by the trainer)
# ═══════════════════════════════════════════════════════════════════════════

def kl_divergence_gaussian(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """
    KL( q(z|x) || N(0, I) ) for a diagonal Gaussian.

    KL = −½ Σ_d (1 + logvar_d − μ²_d − exp(logvar_d))

    Args:
        mu:     (..., D) — posterior mean
        logvar: (..., D) — posterior log-variance (clamped before call)

    Returns:
        scalar — mean KL over all elements (batch × channels × dim)

    LOSO safety: purely algebraic — no stored state.
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def mean_distance_correlation_loss(
    z_content_list: List[torch.Tensor],
    z_style_list:   List[torch.Tensor],
) -> torch.Tensor:
    """
    Mean distance correlation between z_content and z_style across all
    (mode, channel) pairs.

    Args:
        z_content_list: List[K] of (B, C, content_dim) — reparameterised
        z_style_list:   List[K] of (B, C, style_dim)   — reparameterised

    Returns:
        scalar — mean distance correlation over K×C pairs

    Called from the trainer using only TRAINING-split outputs.
    LOSO safety: purely a function of its arguments, no stored state.
    """
    K = len(z_content_list)
    C = z_content_list[0].size(1)
    dcors: List[torch.Tensor] = []
    for k in range(K):
        for c in range(C):
            zc = z_content_list[k][:, c, :]   # (B, content_dim)
            zs = z_style_list[k][:, c, :]     # (B, style_dim)
            dcors.append(_distance_correlation(zc, zs))
    if not dcors:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(dcors).mean()
