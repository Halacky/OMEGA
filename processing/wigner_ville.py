"""
Pseudo Wigner-Ville Distribution (PWVD) features for EMG signals.

The Wigner-Ville Distribution provides the MAXIMUM time-frequency resolution
— no trade-off between time and frequency like STFT.  However, the pure WVD
suffers from cross-term interference.  The pseudo-WVD uses a frequency
smoothing window to suppress cross-terms at a modest cost in frequency
resolution (still much better than STFT).

For a discrete analytic signal z[n]:
    PWVD[n, k] = Σ_m  h[m] · z[n+m] · z*[n−m] · exp(−j·4π·k·m / N_freq)

where h[m] is a Hamming smoothing window of length L.

Key advantages over STFT for EMG:
  - Captures rapid transients (motor unit firing) with full time resolution
  - No spectral leakage from fixed window length
  - Energy distribution is always real and preserves marginal properties

Output modes (same API as MFCC/MDCT):
  1. Spectrogram (N, n_freq, T_frames, C) for 2D CNN
  2. Flat features (N, F) — statistics over time for SVM

References:
  - Ville, J. "Théorie et applications de la notion de signal analytique," 1948
  - Claasen & Mecklenbräuker, "The Wigner Distribution," Philips J. Res., 1980
  - Cohen, L. "Time-Frequency Analysis," Prentice Hall, 1995
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import hilbert


@dataclass
class PseudoWVDExtractor:
    """
    Pseudo Wigner-Ville Distribution feature extractor for EMG.

    Args:
        sampling_rate:   Hz (NinaPro DB2: 2000).
        n_freq:          Number of frequency bins in output (default 64).
        window_length:   Smoothing window length in samples (default 51, must be odd).
        hop:             Time hop in samples for output frames (default 20 = 10ms).
        fmax:            Maximum frequency to keep in Hz (default 1000 = Nyquist).
        use_deltas:      Append delta + delta-delta.
        logger:          Optional logger.
    """
    sampling_rate: int = 2000
    n_freq: int = 64
    window_length: int = 51
    hop: int = 20
    fmax: float = 1000.0
    use_deltas: bool = False
    logger: Optional[logging.Logger] = None

    _window: np.ndarray = field(init=False, repr=False)
    _n_freq_keep: int = field(init=False, repr=False)

    def __post_init__(self):
        if self.window_length % 2 == 0:
            self.window_length += 1
        self._window = np.hamming(self.window_length).astype(np.float64)
        # How many freq bins to keep (up to fmax)
        self._n_freq_keep = min(
            self.n_freq,
            int(self.n_freq * self.fmax / (self.sampling_rate / 2.0))
        )
        if self._n_freq_keep < 1:
            self._n_freq_keep = self.n_freq

        if self.logger:
            self.logger.info(
                f"[PWVD] n_freq={self.n_freq}, window={self.window_length}, "
                f"hop={self.hop}, fmax={self.fmax}, "
                f"freq_bins_kept={self._n_freq_keep}, deltas={self.use_deltas}"
            )

    @staticmethod
    def _compute_deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
        padded = np.pad(feat, ((0, 0), (width, width)), mode='edge')
        denom = 2.0 * sum(n ** 2 for n in range(1, width + 1))
        delta = np.zeros_like(feat)
        for n in range(1, width + 1):
            delta += n * (padded[:, width + n:width + n + feat.shape[1]]
                          - padded[:, width - n:width - n + feat.shape[1]])
        return delta / denom

    def _compute_pwvd_single_channel(self, x: np.ndarray) -> np.ndarray:
        """
        Compute pseudo-WVD for a single 1D signal.

        Args:
            x: (T,) — single-channel time-domain signal.

        Returns:
            pwvd: (n_coeff, T_frames)
                  n_coeff = n_freq_keep * 3 if use_deltas else n_freq_keep.
        """
        T = len(x)

        # 1. Compute analytic signal via Hilbert transform
        z = hilbert(x.astype(np.float64))  # complex analytic signal, (T,)

        # 2. Determine output time points
        half_win = self.window_length // 2
        # Time centers where we can compute full WVD (need ±half_win margin)
        t_start = half_win
        t_end = T - half_win
        if t_start >= t_end:
            n_coeff = self._n_freq_keep * 3 if self.use_deltas else self._n_freq_keep
            return np.zeros((n_coeff, 1), dtype=np.float32)

        time_centers = np.arange(t_start, t_end, self.hop)
        n_frames = len(time_centers)

        if n_frames == 0:
            n_coeff = self._n_freq_keep * 3 if self.use_deltas else self._n_freq_keep
            return np.zeros((n_coeff, 1), dtype=np.float32)

        # 3. Compute PWVD for all time centers (vectorized)
        # Lag indices: m = -half_win, ..., 0, ..., half_win
        m = np.arange(-half_win, half_win + 1)  # (L,)

        # Build index arrays: (n_frames, L)
        t_plus_m = time_centers[:, None] + m[None, :]   # (n_frames, L)
        t_minus_m = time_centers[:, None] - m[None, :]   # (n_frames, L)

        # Clip to valid range
        t_plus_m = np.clip(t_plus_m, 0, T - 1)
        t_minus_m = np.clip(t_minus_m, 0, T - 1)

        # Instantaneous autocorrelation: R[n, m] = z[n+m] * conj(z[n-m])
        R = z[t_plus_m] * np.conj(z[t_minus_m])  # (n_frames, L)

        # Apply smoothing window
        R = R * self._window[None, :]  # (n_frames, L)

        # 4. FFT along lag axis → frequency
        # Zero-pad to n_freq*2 for better frequency resolution
        nfft = self.n_freq * 2
        S = np.fft.fft(R, n=nfft, axis=1)  # (n_frames, nfft)

        # Take magnitude (PWVD is real but numerical errors give small imag part)
        # Keep only positive frequencies up to fmax
        pwvd = np.abs(S[:, :self._n_freq_keep]).T  # (n_freq_keep, n_frames)

        # Log-scale for better dynamic range
        pwvd = np.log(pwvd + 1e-10).astype(np.float32)

        # 5. Optional deltas
        if self.use_deltas:
            delta = self._compute_deltas(pwvd)
            delta2 = self._compute_deltas(delta)
            pwvd = np.concatenate([pwvd, delta, delta2], axis=0)

        return pwvd

    # ── Public API: spectrogram ──────────────────────────────────────────

    def transform_spectrogram(self, X: np.ndarray) -> np.ndarray:
        """
        Compute PWVD spectrograms for a batch.

        Args:
            X: (N, T, C) — batch of raw EMG windows.
        Returns:
            (N, n_coeff, T_frames, C) — PWVD spectrograms.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected (N, T, C), got {X.shape}")

        N, T, C = X.shape
        sample = self._compute_pwvd_single_channel(X[0, :, 0])
        n_coeff, T_frames = sample.shape

        result = np.zeros((N, n_coeff, T_frames, C), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                result[i, :, :, c] = self._compute_pwvd_single_channel(X[i, :, c])

        return result

    # ── Public API: flat features ────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute flat PWVD features for a batch.

        6 statistics per frequency bin per channel.

        Args:
            X: (N, T, C) — batch of raw EMG windows.
        Returns:
            (N, F) — flat features.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected (N, T, C), got {X.shape}")

        spectrograms = self.transform_spectrogram(X)
        N, n_coeff, T_frames, C = spectrograms.shape

        eps = 1e-8
        mean = spectrograms.mean(axis=2)
        std = spectrograms.std(axis=2) + eps
        vmin = spectrograms.min(axis=2)
        vmax = spectrograms.max(axis=2)

        centered = spectrograms - mean[:, :, None, :]
        m3 = (centered ** 3).mean(axis=2)
        m4 = (centered ** 4).mean(axis=2)
        skew = m3 / (std ** 3 + eps)
        kurt = m4 / (std ** 4 + eps) - 3.0

        stats = np.stack([mean, std, vmin, vmax, skew, kurt], axis=-1)
        features = stats.reshape(N, -1).astype(np.float32)

        if self.logger:
            self.logger.info(
                f"[PWVD] Flat features: {features.shape[1]} "
                f"(n_coeff={n_coeff}, C={C}, 6 stats)"
            )

        return features

    @property
    def n_coeff(self) -> int:
        n = self._n_freq_keep
        return n * 3 if self.use_deltas else n
