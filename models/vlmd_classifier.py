"""
Variational Latent Mode Decomposition (VLMD) Classifier for multi-channel EMG.

Architecture
────────────
  Input X ∈ ℝ^{B×T×C}  (C = 12 channels, T time steps)

  1. VLMD Encoder  (global — no subject-specific parameters)
       A ∈ ℝ^{C×M},  M < C — learnable mixing matrix
       Z = X @ A  →  (B, T, M)  latent modes

  2. Per-Mode Feature Extraction  (shared extractor, differentiable end-to-end)
       For each latent mode z_m ∈ ℝ^{B×T}:
         Temporal stats  : mean, std, min, max, peak-to-peak, MAV, RMS, soft-ZCR  (8)
         Spectral bands  : log-compressed FFT power in num_bands bands        (num_bands)
         Feature vector  : (B, 8 + num_bands)  per mode

  3. Classification Head
       Input  : (B, M × (8 + num_bands))
       MLP    : Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Linear
       Output : (B, num_classes)

Regularisers
────────────
  Applied only to model parameters and training-batch signals — zero test leakage.

  Orthogonality   λ_orth  × ‖AᵀA − I_M‖_F² / M²
    Encourages A to have orthonormal columns.
    Ensures modes capture linearly independent (non-redundant) signal components,
    analogous to the independence criterion in ICA/PCA.

  Reconstruction  λ_recon × ‖X − ZAᵀ‖_F² / ‖X‖_F²
    Ensures M latent modes collectively preserve the full C-channel signal.
    Normalised by signal power → amplitude-invariant penalty.

LOSO guarantee
──────────────
  • A and classifier weights are global — trained from pooled train-subjects.
  • No subject-specific parameters exist in the model.
  • Standardisation: mean/std computed from X_train only.
  • model.eval() at inference: BatchNorm running statistics frozen.
    No test-time adaptation of any kind.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# VLMD Encoder — global learnable mixing matrix
# ═══════════════════════════════════════════════════════════════════════════

class VLMDEncoder(nn.Module):
    """
    Global learnable linear mixing matrix A ∈ ℝ^{C×M}.

    Decomposes C-channel EMG into M latent mode signals:
        Z = X @ A,   X: (B, T, C) → Z: (B, T, M)

    A is shared across ALL subjects and folds — it models subject-invariant
    muscle synergies discovered from the pooled training set.

    Initialisation
    ──────────────
    A is initialised via thin QR decomposition of a Gaussian random matrix,
    giving M orthonormal columns of length C.  This ensures:
      1. Initial modes have equal energy (unit-norm columns).
      2. Initial modes are linearly independent (no redundancy).
      3. Starting close to orthonormality speeds up convergence.

    Parameters
    ----------
    in_channels : C — number of EMG input channels.
    num_modes   : M — number of latent modes.  Must satisfy M < C.
    """

    def __init__(self, in_channels: int, num_modes: int) -> None:
        super().__init__()
        if num_modes >= in_channels:
            raise ValueError(
                f"num_modes={num_modes} must be strictly less than "
                f"in_channels={in_channels}.  VLMD requires M < C to achieve "
                "dimensionality reduction from channels to latent modes."
            )
        self.in_channels = in_channels
        self.num_modes   = num_modes

        # Orthonormal initialisation via thin QR:
        # randn(C, M) with C > M  →  QR gives Q: (C, M) with orthonormal cols.
        A_init = torch.randn(in_channels, num_modes)
        Q, _   = torch.linalg.qr(A_init)           # Q: (C, M)
        self.A = nn.Parameter(Q.contiguous())       # shape: (C, M)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Linear mixing: Z = X @ A.

        Parameters
        ----------
        X : (B, T, C) — raw (standardised) EMG windows.

        Returns
        -------
        Z : (B, T, M) — M latent mode signals.
        """
        return X @ self.A

    # ── Regularisation terms ──────────────────────────────────────────────

    def orthogonality_penalty(self) -> torch.Tensor:
        """
        ‖AᵀA − I_M‖_F² / M²

        Penalises deviation of A from having orthonormal columns.
        Normalised by M² so the penalty magnitude is independent of num_modes.

        • When A has orthonormal columns: AᵀA = I_M  →  penalty = 0.
        • As columns become correlated or scale away from 1: penalty grows.

        Encourages the M latent modes to capture linearly independent,
        non-redundant signal components — analogous to ICA independence.
        Operates only on model parameter A — no data involved.
        """
        AtA = self.A.T @ self.A                                       # (M, M)
        eye = torch.eye(self.num_modes, device=self.A.device, dtype=self.A.dtype)
        return (AtA - eye).pow(2).sum() / (self.num_modes ** 2)

    def reconstruction_penalty(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """
        ‖X − ZAᵀ‖_F² / ‖X‖_F²

        Ensures that the M latent modes collectively preserve the original
        C-channel signal.  X_hat = Z @ Aᵀ is the pseudo-inverse reconstruction.
        When A has orthonormal columns, X_hat = X exactly (zero penalty).
        During training, A drifts from orthonormality; this penalty acts as
        a counterforce ensuring signal completeness.

        Normalised by signal power → amplitude-invariant.

        Parameters
        ----------
        X : (B, T, C) — training-batch windows (train data only, never test).
        Z : (B, T, M) — latent modes for the same training batch.
        """
        X_hat   = Z @ self.A.T                                        # (B, T, C)
        sig_pow = X.pow(2).mean().clamp_min(1e-8)
        return (X - X_hat).pow(2).mean() / sig_pow


# ═══════════════════════════════════════════════════════════════════════════
# Per-mode differentiable feature extractor
# ═══════════════════════════════════════════════════════════════════════════

class ModeFeatureExtractor(nn.Module):
    """
    Extract temporal statistics and spectral power features from a 1-D mode.

    All operations are fully differentiable, enabling gradient flow from
    the classification loss back through the features to the mixing matrix A.

    Temporal features (8 per mode)
    ───────────────────────────────
    mean, std, min, max, peak-to-peak, mean absolute value (MAV),
    root-mean-square (RMS), soft zero-crossing rate (soft-ZCR).

    Soft-ZCR uses a sigmoid approximation rather than a hard sign comparison,
    preserving differentiability:
        soft_zcr = sigmoid(−z_t · z_{t+1} · k).mean()
    When consecutive samples have opposite signs (sign change), the product is
    negative, sigmoid returns ≈1.  Same sign → product positive → sigmoid ≈0.

    Spectral features (num_bands per mode)
    ───────────────────────────────────────
    Normalised FFT power averaged in `num_bands` logarithmically-spaced
    frequency bins.  Log-spacing emphasises low-frequency content
    (20–200 Hz, where EMG gesture information concentrates) while still
    covering higher-frequency components.  Values are log-compressed (log1p)
    for numerical stability.

    Total features per mode: 8 + num_bands.

    Parameters
    ----------
    num_bands : number of spectral frequency bands (default 8).
    """

    # Sharpness of the sigmoid approximation for ZCR (unitless).
    # At k=20 and unit-scale signals (std≈1), the transition is very steep.
    _ZCR_K: float = 20.0

    def __init__(self, num_bands: int = 8) -> None:
        super().__init__()
        self.num_bands = num_bands

    @property
    def features_per_mode(self) -> int:
        """Total feature dimension produced for one latent mode."""
        return 8 + self.num_bands

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, T) — one latent mode signal, all samples in a batch.

        Returns
        -------
        feats : (B, 8 + num_bands) — temporal + spectral feature vector.
        """
        B, T = z.shape

        # ── Temporal statistics ───────────────────────────────────────────
        t_mean = z.mean(dim=1, keepdim=True)                          # (B, 1)
        t_std  = z.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
        t_min  = z.min(dim=1, keepdim=True).values
        t_max  = z.max(dim=1, keepdim=True).values
        t_ptp  = t_max - t_min                                        # peak-to-peak
        t_mav  = z.abs().mean(dim=1, keepdim=True)                   # mean abs value
        t_rms  = z.pow(2).mean(dim=1, keepdim=True).sqrt()           # root mean square

        # Soft ZCR — differentiable approximation of zero-crossing rate.
        # sigmoid(-z_t · z_{t+1} · k) ≈ 1 when sign changes, ≈ 0 otherwise.
        zcr_soft = torch.sigmoid(
            -z[:, 1:] * z[:, :-1] * self._ZCR_K
        ).mean(dim=1, keepdim=True)                                   # (B, 1)

        temporal_feats = torch.cat(
            [t_mean, t_std, t_min, t_max, t_ptp, t_mav, t_rms, zcr_soft],
            dim=1,
        )  # (B, 8)

        # ── Spectral features ─────────────────────────────────────────────
        n_fft = T // 2 + 1                                            # rfft output size
        Z_fft = torch.fft.rfft(z, dim=1)                             # (B, n_fft) complex
        power = Z_fft.abs().pow(2)                                    # (B, n_fft) real

        # Logarithmically-spaced band edges on [1, n_fft-1] (skip DC bin 0).
        # logspace(0, log10(n_fft-1), num_bands+1) gives num_bands intervals.
        max_bin   = max(n_fft - 1, 2)
        log_edges = torch.logspace(
            0.0,
            math.log10(max_bin),
            steps=self.num_bands + 1,
            device=z.device,
        )  # (num_bands + 1,)

        band_feats: List[torch.Tensor] = []
        for b in range(self.num_bands):
            lo = max(1, int(log_edges[b].item()))
            hi = min(n_fft, int(log_edges[b + 1].item()) + 1)
            hi = max(hi, lo + 1)                                      # guarantee non-empty
            band_power = power[:, lo:hi].mean(dim=1, keepdim=True)
            band_feats.append(band_power)

        spectral_feats = torch.cat(band_feats, dim=1)                # (B, num_bands)
        spectral_feats = torch.log1p(spectral_feats)                 # log-compress

        return torch.cat([temporal_feats, spectral_feats], dim=1)    # (B, 8+num_bands)


# ═══════════════════════════════════════════════════════════════════════════
# Full VLMD Classifier
# ═══════════════════════════════════════════════════════════════════════════

class VLMDClassifier(nn.Module):
    """
    End-to-end VLMD pipeline:
        VLMD Encoder  →  Per-mode feature extraction  →  MLP classifier.

    See module-level docstring for the full architecture and LOSO guarantees.

    Parameters
    ----------
    in_channels : C — EMG input channels (e.g. 12 for Ninapro DB2).
    num_modes   : M — latent modes, M < C (e.g. 6 for 12-channel input).
    num_classes : K — gesture classes.
    num_bands   : spectral frequency bands per mode (default 8).
    hidden_dim  : MLP hidden layer width (default 128).
    dropout     : dropout probability in the MLP (default 0.3).
    """

    def __init__(
        self,
        in_channels: int,
        num_modes:   int,
        num_classes: int,
        num_bands:   int   = 8,
        hidden_dim:  int   = 128,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder        = VLMDEncoder(in_channels, num_modes)
        self.feat_extractor = ModeFeatureExtractor(num_bands=num_bands)

        feat_dim = num_modes * self.feat_extractor.features_per_mode

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Shared internal step ──────────────────────────────────────────────

    def _encode_and_extract(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode X to latent modes Z, then extract per-mode features.

        This shared step is called by both forward() and forward_with_modes()
        to avoid redundant computation.

        Parameters
        ----------
        X : (B, T, C)

        Returns
        -------
        Z        : (B, T, M) — latent modes (used for regularisation in training).
        combined : (B, M × (8 + num_bands)) — concatenated mode feature vectors.
        """
        Z = self.encoder(X)                                          # (B, T, M)

        mode_feats: List[torch.Tensor] = []
        for m in range(self.encoder.num_modes):
            z_m   = Z[:, :, m]                                       # (B, T)
            feats = self.feat_extractor(z_m)                         # (B, feat_per_mode)
            mode_feats.append(feats)

        combined = torch.cat(mode_feats, dim=1)                      # (B, M*feat_per_mode)
        return Z, combined

    # ── Public API ────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for inference.

        Parameters
        ----------
        X : (B, T, C) — standardised EMG windows.

        Returns
        -------
        logits : (B, num_classes)
        """
        _, combined = self._encode_and_extract(X)
        return self.classifier(combined)

    def forward_with_modes(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns latent modes Z.

        Used in the training loop so that the caller can compute the
        reconstruction penalty using the same Z that produced the logits
        (no redundant encode call).

        Parameters
        ----------
        X : (B, T, C) — standardised training-batch windows.

        Returns
        -------
        logits : (B, num_classes)
        Z      : (B, T, M) — latent modes for regularisation
        """
        Z, combined = self._encode_and_extract(X)
        logits = self.classifier(combined)
        return logits, Z

    def regularisation_loss(
        self,
        X:            torch.Tensor,
        Z:            torch.Tensor,
        lambda_orth:  float = 0.1,
        lambda_recon: float = 0.01,
    ) -> torch.Tensor:
        """
        Combined regularisation:
            reg = lambda_orth  × orthogonality_penalty(A)
                + lambda_recon × reconstruction_penalty(X, Z)

        Operates on model parameter A and training-batch signals only.
        No test-data statistics involved — no data leakage.

        Parameters
        ----------
        X            : (B, T, C) — training batch (standardised).
        Z            : (B, T, M) — latent modes for the same batch.
        lambda_orth  : weight for the orthogonality penalty.
        lambda_recon : weight for the reconstruction penalty.
        """
        orth  = self.encoder.orthogonality_penalty()
        recon = self.encoder.reconstruction_penalty(X, Z)
        return lambda_orth * orth + lambda_recon * recon

    # ── Interpretability ──────────────────────────────────────────────────

    def get_mixing_matrix(self) -> torch.Tensor:
        """Return the learned mixing matrix A: (C, M) — detached, on CPU."""
        return self.encoder.A.detach().cpu()

    def analyse_modes(self) -> Dict:
        """
        Compute interpretability metrics for the current mixing matrix A.

        Returns
        -------
        dict containing:
          mixing_matrix  : A as nested list, shape (C, M).
          gram_matrix    : AᵀA as nested list, shape (M, M).
                           Ideally near identity for orthonormal modes.
          orthogonality  : ‖AᵀA − I‖_F — 0 means perfectly orthonormal.
          mean_col_norm  : mean L2 norm of A's columns.  1.0 = unit columns.
          col_norms      : per-column norms, list of M floats.
        """
        A   = self.encoder.A.detach().cpu()                          # (C, M)
        AtA = A.T @ A                                                # (M, M)
        eye = torch.eye(self.encoder.num_modes)
        return {
            "mixing_matrix":  A.tolist(),
            "gram_matrix":    AtA.tolist(),
            "orthogonality":  float((AtA - eye).norm().item()),
            "mean_col_norm":  float(A.norm(dim=0).mean().item()),
            "col_norms":      A.norm(dim=0).tolist(),
        }
