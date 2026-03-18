"""
Unfolded VMD (UVMD) Classifier for EMG gesture recognition.

Replaces classical iterative VMD with L unrolled ADMM iterations where
bandwidth constraints (alpha), dual step sizes (tau), and mode centre
frequencies (omega_k) are all learnable via backpropagation.

All parameters are GLOBAL — shared across every subject and every channel.
There is no subject-specific state anywhere in this module.

Architecture
────────────
  Raw EMG  (B, T, C)
      │
  UVMDBlock  [K modes, L unrolled ADMM iterations]
      │  learnable: log_alpha (L, K), raw_tau (L,), raw_omega (K,)
      ↓
  modes  (B, K, T, C)
      │
  K × per-mode 1-D CNN encoder  →  feat  (B, K, feat_dim)
      │
  Concatenate  →  (B, K·feat_dim)
      │
  Linear classifier  →  (B, num_classes)

Training loss
─────────────
  L = CrossEntropy + lambda_overlap × spectral_overlap_penalty(omega)

The spectral overlap penalty discourages modes from collapsing to the same
centre frequency by penalising small pairwise distances between omega_k.

LOSO safety
───────────
  ✓  UVMDBlock is a deterministic function of (input, parameters) — no
     cross-sample or cross-subject statistics in the forward pass.
  ✓  All parameters (alpha, tau, omega, CNN weights) are trained from
     TRAIN data only and applied identically to every sample.
  ✓  BatchNorm running stats accumulated only from training samples.
  ✓  model.eval() freezes all stochastic layers at inference.

References
──────────
  Dragomiretskiy & Zosso (2014) — Variational Mode Decomposition (ADMM).
  Chen et al. (2021) — Algorithm Unrolling: Interpretable, Efficient DL.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Unfolded VMD Block
# ═══════════════════════════════════════════════════════════════════════════

class UVMDBlock(nn.Module):
    """
    L unrolled ADMM iterations of VMD with all parameters learnable.

    For each input window x ∈ R^{B × T × C}:

      Step 0 — One-sided DFT:
          f̂(ω)  ∈  C^{B·C × T_rfft}   via torch.fft.rfft

      Step l (for l = 0..L-1) — Jacobi mode update:
          For k = 0..K-1:
              û_k ← (f̂ − Σ_{j≠k} û_j^prev + λ̂/2) / (1 + 2α_lk·(ω − ω_k)²)
          λ̂ ← λ̂ + τ_l · (f̂ − Σ_k û_k)

      Final — IDFT each mode → modes ∈ R^{B × K × T × C}

    Learnable parameters (all shared across subjects and channels):
      log_alpha : (L, K)  — log bandwidth per iteration per mode;
                             alpha = exp(log_alpha) > 0 always
      raw_tau   : (L,)    — dual step size; tau = softplus(raw_tau) > 0
      raw_omega : (K,)    — centre frequencies;
                             omega = sigmoid(raw_omega) × 0.5 ∈ (0, 0.5)

    The Jacobi update (simultaneous, not Gauss-Seidel) avoids any ordering
    dependency between modes and keeps the forward graph clean for autograd.

    Parameters
    ----------
    K : int
        Number of modes.
    L : int
        Number of unrolled iterations (network depth).
    alpha_init : float
        Initial bandwidth constraint (classic VMD default: 2000).
    tau_init : float
        Initial dual step size (near 0 = noise-free regime).
    """

    def __init__(
        self,
        K: int = 4,
        L: int = 8,
        alpha_init: float = 2000.0,
        tau_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.K = K
        self.L = L

        # ── Bandwidth: alpha = exp(log_alpha), shape (L, K) ──────────────
        # Init at log(alpha_init) so alpha starts at the classical VMD value.
        self.log_alpha = nn.Parameter(
            torch.full((L, K), math.log(alpha_init))
        )

        # ── Dual step: tau = softplus(raw_tau), shape (L,) ───────────────
        # softplus_inverse(x) = log(exp(x) - 1) for x > 0
        raw_tau_init = math.log(math.expm1(tau_init) + 1e-8)  # expm1 = exp-1
        self.raw_tau = nn.Parameter(torch.full((L,), raw_tau_init))

        # ── Centre freqs: omega = sigmoid(raw_omega) × 0.5 ∈ (0, 0.5) ──
        # Initialise linearly spaced across the spectrum: 0.05, …, 0.45
        omega_init = torch.linspace(0.05, 0.45, K)
        # Invert: omega = 0.5·sigmoid(raw) ⟹ sigmoid(raw) = 2·omega
        #         raw = logit(2·omega) = log(2ω / (1 − 2ω))
        raw_omega_init = torch.log(
            2.0 * omega_init / (1.0 - 2.0 * omega_init + 1e-8)
        )
        self.raw_omega = nn.Parameter(raw_omega_init)

    # ── Parameter accessors ─────────────────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        """Bandwidth constraints, shape (L, K), always positive."""
        return torch.exp(self.log_alpha)

    @property
    def tau(self) -> torch.Tensor:
        """Dual step sizes, shape (L,), always positive."""
        return F.softplus(self.raw_tau)

    @property
    def omega(self) -> torch.Tensor:
        """Centre frequencies, shape (K,), values in (0, 0.5)."""
        return torch.sigmoid(self.raw_omega) * 0.5

    # ── Spectral overlap penalty ─────────────────────────────────────────────

    def spectral_overlap_penalty(self, sigma: float = 0.05) -> torch.Tensor:
        """
        Gaussian overlap penalty on pairwise mode centre-frequency distances.

        Penalises configurations where two modes cluster at similar centre
        frequencies.  Encourages spectral diversity without hard constraints.

        The penalty is normalised by the number of off-diagonal pairs so
        it stays in [0, 1] regardless of K.

        Parameters
        ----------
        sigma : float
            Width of the Gaussian kernel (default 0.05 = 5% of [0, 0.5]).

        Returns
        -------
        penalty : scalar tensor in [0, 1]
        """
        omega = self.omega                                           # (K,)
        diff = omega.unsqueeze(0) - omega.unsqueeze(1)             # (K, K)
        penalty_mat = torch.exp(-(diff ** 2) / (2.0 * sigma ** 2)) # (K, K)
        mask = 1.0 - torch.eye(self.K, device=omega.device, dtype=omega.dtype)
        n_pairs = max(1.0, mask.sum().item())
        return (penalty_mat * mask).sum() / n_pairs

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose raw EMG windows into K frequency modes.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, C)
            Batch of (optionally pre-standardised) EMG windows.

        Returns
        -------
        modes : torch.Tensor, shape (B, K, T, C)
            K decomposed mode signals per window.

        Gradient flow
        -------------
        The pass is fully differentiable:
          rfft / irfft are differentiable in PyTorch (complex autograd).
          The ADMM algebraic operations (division, addition) are differentiable.
          Gradients flow to log_alpha, raw_tau, raw_omega.

        No cross-sample information is used — each window is processed
        independently, making this operation safe for LOSO.
        """
        B, T, C = x.shape
        K, L = self.K, self.L
        T_rfft = T // 2 + 1

        # Current learnable parameters
        alpha = self.alpha   # (L, K), positive float
        tau   = self.tau     # (L,),   positive float
        omega = self.omega   # (K,),   in (0, 0.5)

        # Normalised frequency axis — matches omega's range [0, 0.5]
        freqs = torch.linspace(0.0, 0.5, T_rfft, device=x.device, dtype=x.dtype)

        # ── One-sided DFT ────────────────────────────────────────────────
        # Flatten (B, C) dimensions for batched rfft: (B, C, T) → (B·C, T)
        x_bc = x.permute(0, 2, 1).reshape(B * C, T)           # (B·C, T)
        f_hat = torch.fft.rfft(x_bc, dim=-1)                   # (B·C, T_rfft), complex64

        # ── Initialise ADMM variables ────────────────────────────────────
        # u_list[k]: mode-k spectrum, shape (B·C, T_rfft), complex
        # Using Python list to avoid in-place tensor mutation (autograd safety).
        u_list: List[torch.Tensor] = [
            torch.zeros(B * C, T_rfft, dtype=f_hat.dtype, device=x.device)
            for _ in range(K)
        ]
        lambda_hat = torch.zeros_like(f_hat)  # (B·C, T_rfft), complex

        # ── L unrolled ADMM iterations (Jacobi update) ───────────────────
        for l_idx in range(L):
            alpha_l = alpha[l_idx]   # (K,) float
            tau_l   = tau[l_idx]     # scalar float

            # Sum of all modes from the PREVIOUS iteration (Jacobi)
            u_stacked_prev = torch.stack(u_list, dim=1)   # (B·C, K, T_rfft)
            u_sum = u_stacked_prev.sum(dim=1)              # (B·C, T_rfft)

            u_new: List[torch.Tensor] = []
            for k in range(K):
                # Residual signal: input minus all modes except k
                sum_other = u_sum - u_list[k]             # (B·C, T_rfft)

                # ADMM spectral update:
                #   û_k = (f̂ − sum_other + λ̂/2) / (1 + 2αₗₖ·(freq − ωₖ)²)
                numerator = f_hat - sum_other + lambda_hat * 0.5   # complex

                # Real denominator: (T_rfft,) — broadcast over B·C
                denom = 1.0 + 2.0 * alpha_l[k] * (freqs - omega[k]) ** 2
                u_k = numerator / denom.unsqueeze(0)      # (B·C, T_rfft)
                u_new.append(u_k)

            u_list = u_new   # replace list (no in-place mutation)

            # Dual variable (Lagrange multiplier) ascent
            u_sum_new = torch.stack(u_list, dim=1).sum(dim=1)   # (B·C, T_rfft)
            lambda_hat = lambda_hat + tau_l * (f_hat - u_sum_new)

        # ── IDFT: frequency → time domain ────────────────────────────────
        # Stack final modes: (B·C, K, T_rfft) → reshape → (B·C·K, T_rfft)
        u_final = torch.stack(u_list, dim=1).reshape(B * C * K, T_rfft)
        modes_flat = torch.fft.irfft(u_final, n=T, dim=-1)        # (B·C·K, T)

        # Reshape to (B, C, K, T), then permute to (B, K, T, C)
        modes = modes_flat.reshape(B, C, K, T).permute(0, 2, 3, 1)
        return modes                                               # (B, K, T, C)


# ═══════════════════════════════════════════════════════════════════════════
# UVMD Classifier
# ═══════════════════════════════════════════════════════════════════════════

class UVMDClassifier(nn.Module):
    """
    End-to-end trainable EMG gesture classifier using Unfolded VMD.

    Combines a differentiable UVMD decomposition with per-mode 1-D CNN
    encoders and a linear classifier.  All weights — including the VMD
    parameters — are trained jointly via:

        loss = CrossEntropy + lambda_overlap × spectral_overlap_penalty

    Subject-independence guarantee
    ───────────────────────────────
    Every parameter (UVMDBlock and CNN) is shared across ALL subjects.
    The model is a stateless function: same weights applied to every
    input window regardless of which subject it came from.

    Parameters
    ----------
    K          : Number of VMD modes.
    L          : Number of unrolled ADMM iterations.
    in_channels: Number of EMG channels in the input.
    num_classes: Number of gesture classes.
    feat_dim   : Feature dimension output by each mode-CNN branch.
    hidden_dim : Hidden units in the classification MLP.
    dropout    : Dropout probability in the classifier.
    alpha_init : Initial VMD bandwidth constraint (classic default: 2000).
    tau_init   : Initial VMD dual step size (near 0 = noise-free start).
    """

    def __init__(
        self,
        K: int = 4,
        L: int = 8,
        in_channels: int = 12,
        num_classes: int = 10,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        alpha_init: float = 2000.0,
        tau_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.K = K

        # ── Differentiable VMD decomposition ─────────────────────────────
        self.uvmd = UVMDBlock(K=K, L=L, alpha_init=alpha_init, tau_init=tau_init)

        # ── Per-mode CNN encoders ─────────────────────────────────────────
        # Separate branches allow each encoder to specialise to its mode's
        # frequency band without weight sharing across modes.
        self.mode_encoders = nn.ModuleList([
            self._make_encoder(in_channels, feat_dim) for _ in range(K)
        ])

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(K * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _make_encoder(in_channels: int, feat_dim: int) -> nn.Sequential:
        """Lightweight 1-D CNN branch for one frequency mode."""
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, C)
            Batch of pre-standardised EMG windows.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
        """
        # Decompose into K frequency modes — fully differentiable
        modes = self.uvmd(x)                                # (B, K, T, C)
        B, K, T, C = modes.shape

        # Encode each mode independently
        mode_features: List[torch.Tensor] = []
        for k in range(K):
            mode_k = modes[:, k].permute(0, 2, 1)          # (B, C, T)
            feat_k = self.mode_encoders[k](mode_k)          # (B, feat_dim)
            mode_features.append(feat_k)

        fused = torch.cat(mode_features, dim=1)             # (B, K·feat_dim)
        return self.classifier(fused)                       # (B, num_classes)

    def spectral_overlap_penalty(self, sigma: float = 0.05) -> torch.Tensor:
        """Scalar overlap penalty on mode centre-frequency clustering."""
        return self.uvmd.spectral_overlap_penalty(sigma=sigma)

    def get_learned_uvmd_params(self) -> Dict[str, object]:
        """
        Return current learnable VMD parameters as plain Python lists.

        Useful for post-training analysis — shows what decomposition the
        model converged to (are modes well-separated? alpha converged?).
        """
        with torch.no_grad():
            omega = self.uvmd.omega.cpu().numpy().tolist()      # (K,)
            alpha = self.uvmd.alpha.cpu().numpy().tolist()      # (L, K)
            tau   = self.uvmd.tau.cpu().numpy().tolist()        # (L,)
        return {
            "omega_k": omega,
            "alpha_lk": alpha,
            "tau_l": tau,
        }
