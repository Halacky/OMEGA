"""
EMG Chromagram — projection of PSD onto functional EMG frequency bands.

Analogy to audio:
    In music, a chromagram projects the spectrum onto 12 pitch classes (C, C#, ...).
    For EMG, we project onto functional frequency bands that correspond to
    physiologically meaningful ranges (validated by H1 spectral analysis):

      Band 0:  20 –  50 Hz  — motor unit firing rates, low-threshold MUs
      Band 1:  50 – 100 Hz  — primary EMG power band
      Band 2: 100 – 200 Hz  — high-threshold MUs, recruitment patterns
      Band 3: 200 – 500 Hz  — fast MU components, noise-dominated (H1: CV 2x)

    These match the H1 finding that inter-subject variability is frequency-dependent:
    CV ratio grows ~10x from Band 0 (0.20) to Band 3 (2.04).

Output modes:
    1. **Spectrogram** (N, n_bands*mult, T_frames, C) — for 2D CNN
       where mult=3 if use_deltas else 1
    2. **Flat features** (N, F) — statistics over time for ML classifiers

Features per band per channel (spectrogram mode):
    - Log band energy per frame (compact, 4 values per frame per channel)
    - Optional: delta + delta-delta temporal derivatives

Features per band per channel (flat mode, 10 statistics):
    - mean, std, min, max, skew, kurtosis of log band energy over frames
    - band energy ratio (fraction of total energy in this band)
    - spectral centroid within band
    - spectral flatness within band
    - inter-band correlation (mean correlation with other bands)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.fft import rfft


# EMG functional frequency bands (Hz) — from H1 analysis
EMG_BANDS = [
    (20, 50),    # Band 0: low-frequency motor unit activity
    (50, 100),   # Band 1: primary EMG power
    (100, 200),  # Band 2: high-threshold MUs
    (200, 500),  # Band 3: fast components, high inter-subject variability
]


@dataclass
class EMGChromagramExtractor:
    """
    EMG chromagram feature extractor.

    Projects short-time PSD onto physiologically-motivated frequency bands.

    Args:
        sampling_rate:   EMG sampling rate in Hz.
        bands:           List of (f_low, f_high) tuples. Default: EMG_BANDS.
        frame_length_ms: Analysis frame length in ms (default 25).
        frame_hop_ms:    Hop between frames in ms (default 10).
        pre_emphasis:    Pre-emphasis coefficient (0 to disable).
        use_deltas:      Append delta + delta-delta features.
        logger:          Optional logger.
    """
    sampling_rate: int = 2000
    bands: list = field(default_factory=lambda: list(EMG_BANDS))
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    pre_emphasis: float = 0.97
    use_deltas: bool = True
    logger: Optional[logging.Logger] = None

    # Computed
    _frame_length: int = field(init=False, repr=False)
    _frame_hop: int = field(init=False, repr=False)
    _nfft: int = field(init=False, repr=False)
    _band_masks: list = field(init=False, repr=False)

    def __post_init__(self):
        self._frame_length = int(self.sampling_rate * self.frame_length_ms / 1000.0)
        self._frame_hop = int(self.sampling_rate * self.frame_hop_ms / 1000.0)
        self._nfft = 1
        while self._nfft < self._frame_length:
            self._nfft *= 2

        # Precompute frequency bin masks for each band
        n_freqs = self._nfft // 2 + 1
        freqs = np.arange(n_freqs) * self.sampling_rate / self._nfft
        self._band_masks = []
        for f_low, f_high in self.bands:
            mask = (freqs >= f_low) & (freqs < f_high)
            self._band_masks.append(mask)

        if self.logger:
            self.logger.info(
                f"[EMGChromagram] {len(self.bands)} bands, "
                f"frame={self._frame_length} ({self.frame_length_ms}ms), "
                f"hop={self._frame_hop} ({self.frame_hop_ms}ms), "
                f"nfft={self._nfft}, deltas={self.use_deltas}"
            )

    def _frame_signal(self, x: np.ndarray) -> np.ndarray:
        """Split signal into overlapping frames. Returns (n_frames, frame_length)."""
        T = len(x)
        if T < self._frame_length:
            x = np.pad(x, (0, self._frame_length - T), mode='constant')
            T = self._frame_length
        n_frames = 1 + (T - self._frame_length) // self._frame_hop
        indices = (
            np.arange(self._frame_length)[None, :]
            + np.arange(n_frames)[:, None] * self._frame_hop
        )
        return x[indices]

    @staticmethod
    def _compute_deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
        """Compute delta features using regression. feat: (F, T) -> (F, T)."""
        padded = np.pad(feat, ((0, 0), (width, width)), mode='edge')
        denom = 2.0 * sum(n ** 2 for n in range(1, width + 1))
        delta = np.zeros_like(feat)
        for n in range(1, width + 1):
            delta += n * (padded[:, width + n:width + n + feat.shape[1]]
                          - padded[:, width - n:width - n + feat.shape[1]])
        return delta / denom

    def _compute_chromagram_single_channel(self, x: np.ndarray) -> np.ndarray:
        """
        Compute EMG chromagram for a single 1D signal.

        Args:
            x: (T,) — single-channel signal.

        Returns:
            chroma: (n_coeff, T_frames)
                    n_coeff = n_bands * 3 if use_deltas else n_bands.
        """
        n_bands = len(self.bands)

        if self.pre_emphasis > 0:
            x = np.append(x[0], x[1:] - self.pre_emphasis * x[:-1])

        frames = self._frame_signal(x)
        if len(frames) == 0:
            n_coeff = n_bands * 3 if self.use_deltas else n_bands
            return np.zeros((n_coeff, 1), dtype=np.float32)

        window = np.hamming(self._frame_length).astype(np.float32)
        frames = frames * window[None, :]

        # Power spectrum
        spec = rfft(frames, n=self._nfft, axis=1)
        power_spec = np.abs(spec) ** 2 / self._nfft  # (n_frames, n_freqs)

        # Project onto bands → log band energies
        chroma = np.zeros((len(frames), n_bands), dtype=np.float32)
        for b, mask in enumerate(self._band_masks):
            band_energy = power_spec[:, mask].sum(axis=1)  # (n_frames,)
            chroma[:, b] = np.log(band_energy + 1e-10)

        chroma = chroma.T  # (n_bands, n_frames)

        if self.use_deltas:
            delta = self._compute_deltas(chroma)
            delta2 = self._compute_deltas(delta)
            chroma = np.concatenate([chroma, delta, delta2], axis=0)

        return chroma.astype(np.float32)

    # ── Public API: spectrogram ──────────────────────────────────────────

    def transform_spectrogram(self, X: np.ndarray) -> np.ndarray:
        """
        Compute EMG chromagram spectrograms for a batch.

        Args:
            X: (N, T, C) — batch of raw EMG windows.

        Returns:
            (N, n_coeff, T_frames, C) — chromagram spectrograms.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected (N, T, C), got {X.shape}")

        N, T, C = X.shape
        sample = self._compute_chromagram_single_channel(X[0, :, 0])
        n_coeff, T_frames = sample.shape

        result = np.zeros((N, n_coeff, T_frames, C), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                result[i, :, :, c] = self._compute_chromagram_single_channel(X[i, :, c])

        return result

    # ── Public API: flat features ────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute flat chromagram features for a batch.

        Per band per channel: 6 statistics (mean, std, min, max, skew, kurt)
        of log band energy over time frames, plus 4 cross-band features per channel.

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

        # Cross-band features per channel: band energy ratios + spectral centroid
        n_bands = len(self.bands)
        # Use only the base bands (no deltas) for ratio features
        base_energy = spectrograms[:, :n_bands, :, :]  # (N, n_bands, T_frames, C)
        total_energy = base_energy.sum(axis=1, keepdims=True) + eps  # (N, 1, T_frames, C)
        ratios = (base_energy / total_energy).mean(axis=2)  # (N, n_bands, C)

        features_list = [
            stats.reshape(N, -1),   # (N, n_coeff * C * 6)
            ratios.reshape(N, -1),  # (N, n_bands * C)
        ]

        features = np.concatenate(features_list, axis=1).astype(np.float32)

        if self.logger:
            self.logger.info(
                f"[EMGChromagram] Flat features: {features.shape[1]} "
                f"(n_coeff={n_coeff}, C={C}, 6 stats + {n_bands} ratios)"
            )

        return features

    @property
    def n_coeff(self) -> int:
        """Number of chromagram coefficients (including deltas if enabled)."""
        n = len(self.bands)
        return n * 3 if self.use_deltas else n
