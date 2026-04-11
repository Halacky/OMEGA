"""
VMD (Variational Mode Decomposition) Multi-Stream CNN for EMG classification.

Decomposes each EMG channel into K intrinsic mode functions via VMD,
then processes each mode through a CNN backbone with mode embeddings
and attention-based fusion for final classification.

Key insight: inter-subject variability may be concentrated in specific
frequency modes (e.g., low-frequency baseline drift), while gesture-
discriminative information resides in others. Learned mode attention
can automatically down-weight subject-specific modes.

Reference: Dragomiretskiy & Zosso (2014) — Variational Mode Decomposition.

No data leakage:
  - VMD is a per-signal operation (each channel of each window independently).
  - No cross-sample or cross-subject statistics are used in decomposition.
  - Model parameters are learned only from training data.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# VMD Implementation (pure numpy — no external dependencies beyond numpy)
# ═══════════════════════════════════════════════════════════════════════

def vmd_decompose_signal(
    signal: np.ndarray,
    K: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Variational Mode Decomposition of a 1D signal.

    Decomposes `signal` into `K` band-limited modes by solving the
    constrained variational problem via ADMM.

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        Input 1D signal.
    K : int
        Number of modes to decompose into.
    alpha : float
        Bandwidth constraint (moderate bandwidth penalty).
        Larger values → narrower frequency bands per mode.
    tau : float
        Noise tolerance for Lagrange multiplier dual ascent.
        0 = exact reconstruction; >0 allows noise tolerance.
    tol : float
        Convergence tolerance on mode change between iterations.
    max_iter : int
        Maximum number of ADMM iterations.

    Returns
    -------
    modes : np.ndarray, shape (K, T)
        Decomposed mode signals in time domain.
    center_freqs : np.ndarray, shape (K,)
        Estimated center frequency for each mode (normalised 0–0.5).
    n_iter : int
        Actual number of iterations performed.
    """
    T_orig = len(signal)

    if T_orig < 4:
        # Signal too short for meaningful decomposition — distribute evenly
        modes = np.tile(signal / max(K, 1), (K, 1))
        return modes, np.linspace(0, 0.5, K, endpoint=False), 0

    # --- Mirror extension for boundary handling ---
    half = T_orig // 2
    f = np.concatenate([
        signal[half - 1::-1],      # reverse first half
        signal,                     # original
        signal[-1:half - 1:-1],     # reverse second half
    ])
    T = len(f)

    # Spectral domain discretisation
    freqs = np.arange(T, dtype=np.float64) / T - 0.5

    # One-sided FFT of mirrored signal
    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T // 2] = 0

    # Initialise mode spectra, center frequencies, Lagrange multiplier
    u_hat = np.zeros((T, K), dtype=np.complex128)
    u_hat_prev = np.zeros_like(u_hat)
    omega = np.array([(0.5 / K) * k for k in range(K)], dtype=np.float64)
    lambda_hat = np.zeros(T, dtype=np.complex128)

    half_T = T // 2
    freq_positive = freqs[half_T:]

    n_iter = 0
    for iteration in range(max_iter):
        u_hat_prev[:] = u_hat

        for k in range(K):
            # Residual: input minus all other modes (using latest updates)
            sum_other = np.sum(u_hat, axis=1) - u_hat[:, k]
            numerator = f_hat_plus - sum_other + lambda_hat / 2.0
            denominator = 1.0 + alpha * (freqs - omega[k]) ** 2

            u_hat[:, k] = numerator / denominator

            # Update center frequency (spectral center of gravity, positive freqs only)
            psd_k = np.abs(u_hat[half_T:, k]) ** 2
            psd_sum = psd_k.sum()
            if psd_sum > 1e-16:
                omega[k] = np.dot(freq_positive, psd_k) / psd_sum

        # Dual ascent step (Lagrange multiplier update)
        lambda_hat += tau * (np.sum(u_hat, axis=1) - f_hat_plus)

        # Convergence check: relative change in mode spectra
        change = np.sum(np.abs(u_hat - u_hat_prev) ** 2) / T
        n_iter = iteration + 1
        if change < tol:
            break

    # --- Reconstruct time-domain modes ---
    modes = np.zeros((K, T_orig), dtype=np.float64)
    for k in range(K):
        # Build full spectrum via Hermitian symmetry from one-sided spectrum
        spec_full = np.zeros(T, dtype=np.complex128)
        spec_full[half_T:] = u_hat[half_T:, k]
        if half_T > 1:
            spec_full[1:half_T] = np.conj(u_hat[-1:half_T:-1, k])
        spec_full[0] = np.conj(u_hat[-1, k])

        # IFFT, multiply by 2 (one-sided → full), extract original-length portion
        mode_time = np.real(np.fft.ifft(np.fft.ifftshift(spec_full))) * 2
        modes[k] = mode_time[half:half + T_orig]

    return modes, omega, n_iter


def decompose_windows_vmd(
    windows: np.ndarray,
    K: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 300,
) -> np.ndarray:
    """
    Apply VMD to all channels of all windows.

    This is a per-window, per-channel operation — no cross-sample
    information is used (safe for LOSO).

    Parameters
    ----------
    windows : np.ndarray, shape (N, T, C)
        Input windows in (samples, time, channels) format.
    K : int
        Number of VMD modes.
    alpha, tau, tol, max_iter :
        VMD algorithm parameters (see vmd_decompose_signal).

    Returns
    -------
    decomposed : np.ndarray, shape (N, K, T, C)
        Decomposed windows: for each sample, K mode signals per channel.
    """
    N, T, C = windows.shape
    decomposed = np.zeros((N, K, T, C), dtype=np.float32)

    total_calls = N * C
    log_interval = max(1, total_calls // 10)
    call_count = 0

    for i in range(N):
        for c in range(C):
            signal = windows[i, :, c].astype(np.float64)
            modes, _, _ = vmd_decompose_signal(
                signal, K=K, alpha=alpha, tau=tau, tol=tol, max_iter=max_iter,
            )
            decomposed[i, :, :, c] = modes.astype(np.float32)

            call_count += 1
            if call_count % log_interval == 0:
                logger.info(
                    f"  VMD progress: {call_count}/{total_calls} "
                    f"({100 * call_count / total_calls:.0f}%)"
                )

    return decomposed


# ═══════════════════════════════════════════════════════════════════════
# Multi-Stream CNN Model
# ═══════════════════════════════════════════════════════════════════════

class VMDMultiStreamCNN(nn.Module):
    """
    Multi-stream CNN that processes VMD-decomposed EMG modes separately,
    then fuses them via learned attention.

    Architecture
    ────────────
    1. Shared CNN backbone extracts features from each mode independently.
    2. Mode embeddings allow the backbone to distinguish between modes.
    3. Attention network learns per-sample importance weights over modes.
    4. Weighted fusion → classification head.

    The attention weights are returned alongside logits for analysis
    (which modes carry gesture-discriminative vs subject-specific info).

    Parameters
    ----------
    num_modes : int
        Number of VMD modes (K).
    in_channels : int
        Number of EMG channels per mode.
    num_classes : int
        Number of gesture classes.
    backbone_type : str
        'shared' — one CNN backbone + mode embeddings (parameter-efficient).
        'separate' — independent CNN per mode (more capacity, more parameters).
    feat_dim : int
        Feature dimension output by each backbone branch.
    hidden_dim : int
        Hidden dimension in the classification head.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        num_modes: int = 4,
        in_channels: int = 12,
        num_classes: int = 10,
        backbone_type: str = "shared",
        feat_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.backbone_type = backbone_type

        if backbone_type == "shared":
            self.backbone = self._make_backbone(in_channels, feat_dim)
            self.mode_embed = nn.Embedding(num_modes, feat_dim)
        else:
            self.branches = nn.ModuleList([
                self._make_backbone(in_channels, feat_dim)
                for _ in range(num_modes)
            ])

        # Attention network over modes
        self.mode_attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _make_backbone(in_channels: int, feat_dim: int) -> nn.Sequential:
        """Lightweight 1D CNN backbone."""
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feat_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, K, C, T)
            B = batch size, K = number of VMD modes,
            C = EMG channels, T = time steps.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
        attn_weights : torch.Tensor, shape (B, K)
            Mode attention weights (sum to 1 per sample).
        """
        B, K, C, T = x.shape

        if self.backbone_type == "shared":
            # Process all modes through shared backbone
            x_flat = x.reshape(B * K, C, T)
            features = self.backbone(x_flat)             # (B*K, feat_dim)
            features = features.view(B, K, -1)           # (B, K, feat_dim)

            # Add mode embeddings so the backbone can distinguish modes
            mode_ids = torch.arange(K, device=x.device)
            embeds = self.mode_embed(mode_ids)            # (K, feat_dim)
            features = features + embeds.unsqueeze(0)     # broadcast: (B, K, feat_dim)
        else:
            # Each mode processed by its own branch
            mode_features = []
            for k in range(K):
                feat = self.branches[k](x[:, k])         # (B, feat_dim)
                mode_features.append(feat)
            features = torch.stack(mode_features, dim=1)  # (B, K, feat_dim)

        # Attention over modes
        attn_scores = self.mode_attn(features)            # (B, K, 1)
        attn_weights = F.softmax(attn_scores, dim=1)      # (B, K, 1)

        # Weighted fusion
        fused = (features * attn_weights).sum(dim=1)      # (B, feat_dim)

        # Classify
        logits = self.classifier(fused)                   # (B, num_classes)

        return logits, attn_weights.squeeze(-1)
