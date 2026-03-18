"""
Short-Time VMD (STVMD) Classifier with Learnable Segmentation
and Mode-Aware Augmentation.

Hypothesis
──────────
EMG gesture signals are non-stationary across a 200 ms window (phases:
onset → sustained → offset).  Global VMD treats the entire window as
stationary, averaging over these distinct dynamical phases.  STVMD applies
a shared Unrolled VMD independently to S short sub-windows (segments),
capturing the local spectral structure of each temporal phase.

Mode-level augmentation (Gaussian perturbation of spectral amplitude and
phase per mode, training only) simulates inter-subject variability in motor-
unit recruitment patterns, expanding the effective training distribution
without explicit domain adaptation.

Architecture
────────────

  Raw EMG  (B, T, C)
      │
  ┌── LearnableSegmenter ────────────────────────────────────────────────┐
  │  Small 1D-CNN encodes each uniform sub-window to predict a scalar   │
  │  attention logit.  Soft attention weights (softmax over S segments) │
  │  determine which temporal phase dominates the final representation. │
  │  Segment BOUNDARIES are fixed (uniform grid); only weights learned. │
  └──────────────────────────────────────────────────────────────────────┘
      │ segments     (B, S, seg_len, C)   — uniform sub-windows
      │ attn_weights (B, S)               — softmax, sums to 1
      │
  Reshape → (B·S, seg_len, C)
      │
  ┌── SharedUVMDBlock ────────────────────────────────────────────────────┐
  │  L unrolled ADMM iterations; learnable: log_alpha (L,K), raw_tau   │
  │  (L,), raw_omega (K,).  Same weights for every segment and subject. │
  └──────────────────────────────────────────────────────────────────────┘
      │ modes  (B·S, K, seg_len, C)
      │
  ┌── ModeAugmenter  [TRAINING ONLY] ────────────────────────────────────┐
  │  For each mode: Gaussian noise on spectral amplitude and phase.     │
  │  Active ONLY during model.train(); no-op during model.eval().      │
  │  noise ~ N(0,σ) with fixed σ — no data-derived statistics.         │
  └──────────────────────────────────────────────────────────────────────┘
      │ modes  (B·S, K, seg_len, C)   (perturbed during training)
      │
  Reshape → (B, S, K, seg_len, C)
      │
  ┌── PerModeCNN ─────────────────────────────────────────────────────────┐
  │  K separate 1D-CNN branches (one per mode).  Each encoder applied  │
  │  to all S segments of its mode.  Output: per-segment features.     │
  └──────────────────────────────────────────────────────────────────────┘
      │ seg_feats  (B, S, K·feat_dim)
      │
  Attention-weighted sum over S segments using attn_weights from segmenter
      │ fused  (B, K·feat_dim)
      │
  Classifier  (Linear → ReLU → Dropout → Linear)
      ↓
  logits  (B, num_classes)

Training objective
──────────────────
  loss = CrossEntropy(logits, y)
       + λ_overlap × SpectralOverlapPenalty(ω_k)

  SpectralOverlapPenalty prevents mode centre frequencies ω_k from
  collapsing to the same value, ensuring diverse spectral coverage.

LOSO Safety Guarantees
───────────────────────
  ✓ LearnableSegmenter:  global CNN weights; per-window operation;
                          no subject-specific state.
  ✓ SharedUVMDBlock:     global ADMM params (alpha, tau, omega);
                          per-window; no subject-specific state.
  ✓ ModeAugmenter:       active ONLY during model.train().
                          Disabled at model.eval() — never active at
                          validation or test time.
                          Uses fixed σ hyperparameters, not data stats.
                          Noise is i.i.d. per batch — no cross-sample info.
  ✓ PerModeCNN:          K global encoders; shared across segments & subjects.
  ✓ BatchNorm stats:     accumulated from training data only;
                          frozen at inference via model.eval().
  ✓ No test-time adaptation of any kind.

References
──────────
  Dragomiretskiy & Zosso (2014) — Variational Mode Decomposition (ADMM).
  Chen et al. (2021) — Algorithm Unrolling: Interpretable, Efficient DL.
  VMD-based data augmentation (mode-level perturbation for domain shift).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.uvmd_classifier import UVMDBlock


# ═══════════════════════════════════════════════════════════════════════════
# Learnable Segmenter
# ═══════════════════════════════════════════════════════════════════════════

class LearnableSegmenter(nn.Module):
    """
    Split raw EMG window into S uniform segments and predict soft attention.

    The segment BOUNDARIES are fixed at uniform positions (no differentiable
    boundary estimation is attempted — this avoids the complexity of
    learnable window extraction while still allowing the network to learn
    WHICH temporal phase is most discriminative).

    A lightweight 1D-CNN is applied independently to each sub-window to
    produce a scalar attention logit.  Softmax over S logits gives
    normalised per-segment importance weights.

    LOSO Safety
    ───────────
    All CNN and linear weights are global — the same network is applied
    to every window from every subject.  No cross-sample statistics are
    used in the forward pass.

    Parameters
    ----------
    in_channels  : Number of EMG channels C.
    num_segments : Number of equal-length sub-windows S.
    seg_len      : Temporal length of each sub-window (T // S).
    hidden       : Width of the internal CNN feature maps.
    """

    def __init__(
        self,
        in_channels:  int,
        num_segments: int,
        seg_len:      int,
        hidden:       int = 32,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.seg_len      = seg_len

        # Lightweight per-segment CNN encoder.
        # No BatchNorm: we want amplitude sensitivity for phase detection.
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (B*S, hidden, 1)
            nn.Flatten(),              # → (B*S, hidden)
        )

        # Maps segment encoding to a scalar attention logit
        self.attn_proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract uniform segments and compute soft attention weights.

        Parameters
        ----------
        x : (B, T, C)

        Returns
        -------
        segments     : (B, S, seg_len, C) — S non-overlapping sub-windows.
        attn_weights : (B, S) — softmax attention (sums to 1 along S dim).
        """
        B, T, C = x.shape
        S  = self.num_segments
        sl = self.seg_len

        # ── Extract S uniform non-overlapping segments ─────────────────────
        # Guaranteed: T == S * sl (enforced by STVMDClassifier constructor).
        segments = x[:, : S * sl, :].reshape(B, S, sl, C)   # (B, S, sl, C)

        # ── Encode each segment for attention ──────────────────────────────
        # Reshape to (B*S, C, sl) for Conv1d (channels-first).
        seg_flat     = segments.reshape(B * S, sl, C).permute(0, 2, 1)   # (B*S, C, sl)
        enc          = self.encoder(seg_flat)                              # (B*S, hidden)
        scores       = self.attn_proj(enc).reshape(B, S)                  # (B, S)
        attn_weights = torch.softmax(scores, dim=1)                       # (B, S)

        return segments, attn_weights


# ═══════════════════════════════════════════════════════════════════════════
# Mode-Aware Augmenter  (training only)
# ═══════════════════════════════════════════════════════════════════════════

class ModeAugmenter(nn.Module):
    """
    Training-only frequency-domain perturbation of VMD mode signals.

    For each mode signal, adds independent zero-mean Gaussian noise to the
    amplitude and phase of every spectral component.  This simulates:
      • Amplitude noise  → inter-subject variability in motor-unit
                           recruitment intensity.
      • Phase noise      → conduction-velocity differences between subjects
                           (shift in spectral phase without energy change).

    Perturbation model (per frequency bin f)
    ─────────────────────────────────────────
      f̂_aug[f] = f̂[f] × (1 + ε_amp[f]) × exp(i ε_phase[f])

    where  ε_amp   ~ N(0, σ_amp),  ε_phase ~ N(0, σ_phase)  are sampled
    independently for every window, mode, channel, and frequency bin.

    This is equivalent to multiplying by the complex scalar

        perturb[f] = (1 + ε_amp[f]) exp(i ε_phase[f])

    so that real magnitude and phase are perturbed independently.

    LOSO Safety
    ───────────
    • Active ONLY when self.training is True (model.train()).
      Returns input unchanged during model.eval() — zero leakage risk.
    • σ_amp and σ_phase are fixed hyperparameters, NOT estimated from data.
    • Noise is sampled i.i.d. — no cross-sample or cross-subject coupling.

    Parameters
    ----------
    amp_noise_std   : σ for amplitude noise (recommended: 0.05 – 0.15).
    phase_noise_std : σ for phase noise in radians (recommended: 0.03 – 0.10).
    """

    def __init__(
        self,
        amp_noise_std:   float = 0.10,
        phase_noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.amp_noise_std   = amp_noise_std
        self.phase_noise_std = phase_noise_std
        # No learnable parameters — augmentation is purely stochastic.

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency-domain perturbation to all VMD modes (training only).

        Parameters
        ----------
        modes : (N, K, seg_len, C)
            Decomposed mode signals.  N = B × S (batch × segments).

        Returns
        -------
        torch.Tensor, same shape as modes.
            Frequency-perturbed during training; unchanged during eval.
        """
        if not self.training:
            return modes            # LOSO guard: no perturbation at test time

        N, K, seg_len, C = modes.shape
        n_flat = N * K * C

        # ── Reshape to (N*K*C, seg_len) for batched rfft ──────────────────
        # modes: (N, K, seg_len, C)
        #  → permute → (N, K, C, seg_len)   [seg_len last for rfft]
        #  → reshape  → (N*K*C, seg_len)
        modes_2d = modes.permute(0, 1, 3, 2).reshape(n_flat, seg_len)

        # ── Forward FFT ────────────────────────────────────────────────────
        f_hat  = torch.fft.rfft(modes_2d, dim=-1)    # (N*K*C, T_rfft), complex64
        T_rfft = f_hat.shape[-1]

        # ── Sample independent noise for amplitude and phase ───────────────
        # Both tensors: (N*K*C, T_rfft), float32 on same device as modes.
        amp_noise    = torch.randn(n_flat, T_rfft, device=modes.device, dtype=torch.float32)
        phase_angles = torch.randn(n_flat, T_rfft, device=modes.device, dtype=torch.float32)

        amp_factors  = 1.0 + amp_noise    * self.amp_noise_std     # ∈ (1-nσ, 1+nσ)
        phase_angles = phase_angles        * self.phase_noise_std   # radians

        # ── Combined complex perturbation ──────────────────────────────────
        # perturb = amp_factor × exp(i × phase_angle)
        #         = amp_factor × (cos(phase) + i·sin(phase))
        perturb = torch.complex(
            amp_factors * torch.cos(phase_angles),
            amp_factors * torch.sin(phase_angles),
        )                                             # (N*K*C, T_rfft), complex64

        f_aug = f_hat * perturb                       # element-wise complex multiply

        # ── Inverse FFT → time domain ──────────────────────────────────────
        modes_aug_2d = torch.fft.irfft(f_aug, n=seg_len, dim=-1)   # (N*K*C, seg_len)

        # ── Reshape back to (N, K, seg_len, C) ────────────────────────────
        modes_aug = (
            modes_aug_2d
            .reshape(N, K, C, seg_len)
            .permute(0, 1, 3, 2)
        )                                             # (N, K, seg_len, C)

        return modes_aug


# ═══════════════════════════════════════════════════════════════════════════
# Per-Mode CNN Encoder
# ═══════════════════════════════════════════════════════════════════════════

class PerModeCNN(nn.Module):
    """
    K separate 1D-CNN branches, one per VMD mode.

    Each encoder specialises to the spectral characteristics of its
    assigned mode (e.g., low-frequency fatigue, mid-frequency recruitment,
    high-frequency transients) without sharing weights across modes.

    The same encoder for mode k is applied to ALL segments and ALL
    subjects — no subject-specific or segment-specific weights.

    Parameters
    ----------
    K          : Number of VMD modes.
    in_channels: Number of EMG channels C.
    feat_dim   : Output feature dimension per mode encoder.
    """

    def __init__(self, K: int, in_channels: int, feat_dim: int) -> None:
        super().__init__()
        self.K        = K
        self.feat_dim = feat_dim
        self.encoders = nn.ModuleList([
            self._make_encoder(in_channels, feat_dim) for _ in range(K)
        ])

    @staticmethod
    def _make_encoder(in_channels: int, feat_dim: int) -> nn.Sequential:
        """Lightweight 1-D CNN branch for one frequency mode."""
        return nn.Sequential(
            nn.Conv1d(in_channels, 32,       kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,          feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (·, feat_dim, 1)
            nn.Flatten(),              # → (·, feat_dim)
        )

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """
        Encode each VMD mode independently across all batch×segments.

        Parameters
        ----------
        modes : (B, S, K, seg_len, C)

        Returns
        -------
        seg_feats : (B, S, K·feat_dim)
            Per-segment features — all K mode encodings concatenated.
        """
        B, S, K, seg_len, C = modes.shape
        if K != self.K:
            raise ValueError(f"PerModeCNN: expected K={self.K} modes, got K={K}")

        per_mode: List[torch.Tensor] = []
        for k in range(K):
            # Mode k for all batch samples and all segments: (B, S, seg_len, C)
            mode_k = modes[:, :, k, :, :]
            # Reshape to (B*S, C, seg_len) for Conv1d (channels-first)
            mode_k_flat = mode_k.reshape(B * S, seg_len, C).permute(0, 2, 1)
            feat_k      = self.encoders[k](mode_k_flat)          # (B*S, feat_dim)
            per_mode.append(feat_k.reshape(B, S, self.feat_dim)) # (B, S, feat_dim)

        # Stack along mode axis, then flatten modes into feature vector
        stacked = torch.stack(per_mode, dim=2)                   # (B, S, K, feat_dim)
        return stacked.reshape(B, S, K * self.feat_dim)          # (B, S, K*feat_dim)


# ═══════════════════════════════════════════════════════════════════════════
# STVMD Classifier  (full pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class STVMDClassifier(nn.Module):
    """
    Short-Time VMD Classifier with Learnable Segmentation and
    Mode-Aware Augmentation.

    Orchestrates four components (LearnableSegmenter, SharedUVMDBlock,
    ModeAugmenter, PerModeCNN) into an end-to-end differentiable pipeline.
    All parameters are global — shared across every subject.

    Subject-independence guarantee
    ───────────────────────────────
    Every parameter (UVMD, segmenter CNN, mode encoders, classifier) is
    shared across ALL subjects.  The model is a stateless function of its
    input — the same weights applied to every window regardless of origin.

    Parameters
    ----------
    K               : Number of VMD modes (default 3).
    L               : Number of unrolled ADMM iterations (default 4).
    num_segments    : Number of equal sub-windows S (default 4).
    in_channels     : EMG channel count C.
    num_classes     : Gesture class count.
    window_size     : Temporal length T of the input window.
                      MUST be divisible by num_segments.
    feat_dim        : Feature dimension output per mode CNN branch.
    hidden_dim      : Hidden units in the classification MLP.
    dropout         : Dropout probability in classifier head.
    alpha_init      : Initial UVMD bandwidth constraint (VMD default: 2000).
    tau_init        : Initial UVMD dual step size (near 0 = noise-free).
    amp_noise_std   : Mode augmentation amplitude noise σ (training only).
    phase_noise_std : Mode augmentation phase noise σ (training only).
    seg_attn_hidden : Hidden size of the segment attention CNN.
    """

    def __init__(
        self,
        K:               int   = 3,
        L:               int   = 4,
        num_segments:    int   = 4,
        in_channels:     int   = 12,
        num_classes:     int   = 10,
        window_size:     int   = 200,
        feat_dim:        int   = 48,
        hidden_dim:      int   = 128,
        dropout:         float = 0.3,
        alpha_init:      float = 2000.0,
        tau_init:        float = 0.01,
        amp_noise_std:   float = 0.10,
        phase_noise_std: float = 0.05,
        seg_attn_hidden: int   = 32,
    ) -> None:
        super().__init__()
        self.K            = K
        self.num_segments = num_segments

        if window_size % num_segments != 0:
            raise ValueError(
                f"window_size={window_size} must be divisible by "
                f"num_segments={num_segments}.  "
                f"Got window_size % num_segments = {window_size % num_segments}."
            )
        self.seg_len = window_size // num_segments

        # ── Components ────────────────────────────────────────────────────

        # 1. Learnable segmenter: predicts attention over uniform segments
        self.segmenter = LearnableSegmenter(
            in_channels  = in_channels,
            num_segments = num_segments,
            seg_len      = self.seg_len,
            hidden       = seg_attn_hidden,
        )

        # 2. Shared UVMD block — applied to each segment independently
        #    Reuses UVMDBlock from uvmd_classifier.py (avoids code duplication).
        #    Parameters (log_alpha, raw_tau, raw_omega) are shared across ALL
        #    segments and ALL subjects.
        self.uvmd = UVMDBlock(
            K          = K,
            L          = L,
            alpha_init = alpha_init,
            tau_init   = tau_init,
        )

        # 3. Mode augmenter — no learnable params; disabled at eval/test time
        self.augmenter = ModeAugmenter(
            amp_noise_std   = amp_noise_std,
            phase_noise_std = phase_noise_std,
        )

        # 4. Per-mode CNN encoders — K separate branches, shared across segments
        self.mode_cnn = PerModeCNN(
            K           = K,
            in_channels = in_channels,
            feat_dim    = feat_dim,
        )

        # 5. Classification head applied after segment attention pooling
        self.classifier = nn.Sequential(
            nn.Linear(K * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass.

        Parameters
        ----------
        x : (B, T, C) — pre-standardised EMG windows.
            T must equal num_segments × seg_len (guaranteed by constructor).

        Returns
        -------
        logits : (B, num_classes)

        Data flow (no cross-sample or cross-subject information)
        ─────────────────────────────────────────────────────────
          1. Segment: split window into S uniform sub-windows + compute
             soft attention weights (both per-window, no cross-sample ops).
          2. UVMD: decompose each sub-window into K modes.
             Purely algebraic on the input spectrum — no running stats.
          3. Augment: perturb mode spectra with i.i.d. Gaussian noise
             (training only; disabled at eval).
          4. Encode: K-branch CNN over each mode's temporal signal.
             BatchNorm running stats are frozen at eval (model.eval()).
          5. Pool: weighted sum over S segments using attention weights.
          6. Classify: two-layer MLP.
        """
        B, T, C = x.shape
        S  = self.num_segments
        K  = self.K
        sl = self.seg_len

        # ── 1. Learnable segmentation ──────────────────────────────────────
        segments, attn_weights = self.segmenter(x)
        # segments:     (B, S, sl, C)
        # attn_weights: (B, S) — softmax, sums to 1 over S

        # ── 2. Shared UVMD — applied per segment independently ─────────────
        # Flatten batch and segment dimensions: (B, S, sl, C) → (B*S, sl, C)
        segs_flat = segments.reshape(B * S, sl, C)              # (B*S, sl, C)
        modes     = self.uvmd(segs_flat)                        # (B*S, K, sl, C)

        # ── 3. Mode-level augmentation (no-op at model.eval()) ─────────────
        modes = self.augmenter(modes)                           # (B*S, K, sl, C)

        # ── 4. Reshape for per-mode CNN ────────────────────────────────────
        modes = modes.reshape(B, S, K, sl, C)                  # (B, S, K, sl, C)

        # ── 5. Per-mode CNN: encode each mode across all batch×segments ────
        seg_feats = self.mode_cnn(modes)                        # (B, S, K*feat_dim)

        # ── 6. Segment attention pooling ───────────────────────────────────
        # Weighted sum over S segments using learned soft attention weights.
        # attn_weights: (B, S) → unsqueeze → (B, S, 1) for broadcasting.
        fused = (seg_feats * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, K*feat_dim)

        # ── 7. Classify ────────────────────────────────────────────────────
        return self.classifier(fused)                           # (B, num_classes)

    # ── Regularisation ──────────────────────────────────────────────────────

    def spectral_overlap_penalty(self, sigma: float = 0.05) -> torch.Tensor:
        """
        Gaussian spectral overlap penalty on UVMD mode centre frequencies.

        Penalises pairwise similarity between omega_k values, preventing
        mode centre frequencies from collapsing to the same spectral region.
        Operates on the model parameter omega — not on any data.

        Delegates to UVMDBlock.spectral_overlap_penalty() — see that class
        for the exact formula.
        """
        return self.uvmd.spectral_overlap_penalty(sigma=sigma)

    # ── Analysis helpers ────────────────────────────────────────────────────

    def get_learned_params(self) -> Dict[str, object]:
        """
        Return current learnable UVMD parameters as plain Python objects.

        Useful for post-training analysis: shows which frequency bands the
        model discovered, whether modes are well-separated, and whether
        alpha/tau converged to meaningful values.

        Returns
        -------
        dict with keys:
          "omega_k"  : list[float], length K — mode centre frequencies.
          "alpha_lk" : list[list[float]], shape (L, K) — bandwidth params.
          "tau_l"    : list[float], length L — dual step sizes.
        """
        with torch.no_grad():
            omega = self.uvmd.omega.cpu().numpy().tolist()
            alpha = self.uvmd.alpha.cpu().numpy().tolist()
            tau   = self.uvmd.tau.cpu().numpy().tolist()
        return {
            "omega_k":  omega,
            "alpha_lk": alpha,
            "tau_l":    tau,
        }
