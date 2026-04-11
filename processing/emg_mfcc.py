"""
EMG-adapted MFCC (Mel-Frequency Cepstral Coefficients) feature extraction.

Key adaptations for sEMG vs speech:
  - Frequency range: [20, 500] Hz (EMG informative range) vs [300, 8000] Hz (speech).
  - Sampling rate: 2000 Hz (NinaPro DB2) vs 16000 Hz (speech).
  - Number of mel filters: 26 (adequate for 20–500 Hz range).
  - Number of cepstral coefficients: 13 (standard) + delta + delta-delta = 39 total.
  - Frame length: 25 ms (50 samples @ 2000 Hz) — matches speech convention.
  - Frame hop: 10 ms (20 samples) — standard overlap.

Two output modes:
  1. **Spectrogram** (N, n_mfcc, T_frames, C) — for 2D CNN input.
  2. **Flat features** (N, F) — statistics over time for ML classifiers (SVM/RF).

References:
  - Davis & Mermelstein, "Comparison of Parametric Representations for Monosyllabic
    Word Recognition in Continuously Spoken Sentences," IEEE TASSP, 1980.
  - Phinyomark et al., "EMG Feature Evaluation for Improving Myoelectric Pattern
    Recognition Robustness," Expert Systems with Applications, 2013.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.fft import rfft


# ────────────────────────── Mel scale helpers ──────────────────────────────

def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """Convert Hz to Mel scale (O'Shaughnessy, 1987)."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Convert Mel scale back to Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


# ────────────────────────── MFCC Extractor ─────────────────────────────────

@dataclass
class EMGMFCCExtractor:
    """
    EMG-adapted MFCC feature extractor.

    Computes MFCC coefficients from raw EMG windows.  Each channel is processed
    independently (no cross-channel mixing in the MFCC computation).

    Args:
        sampling_rate:  EMG sampling rate in Hz (NinaPro DB2: 2000).
        n_mfcc:         Number of cepstral coefficients to keep (default 13).
        n_mels:         Number of triangular mel filterbank channels (default 26).
        fmin:           Lowest filter center frequency in Hz (default 20).
        fmax:           Highest filter center frequency in Hz (default 500).
        frame_length_ms: Analysis frame length in ms (default 25).
        frame_hop_ms:   Hop between frames in ms (default 10).
        pre_emphasis:   Pre-emphasis coefficient (default 0.97, 0 to disable).
        use_deltas:     Append delta (Δ) and delta-delta (ΔΔ) features (default True).
        use_energy:     Replace c0 with log-energy (default True).
        lifter:         Cepstral liftering coefficient (0 to disable, 22 standard).
        logger:         Optional Python logger.
    """
    sampling_rate: int = 2000
    n_mfcc: int = 13
    n_mels: int = 26
    fmin: float = 20.0
    fmax: float = 500.0
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    pre_emphasis: float = 0.97
    use_deltas: bool = True
    use_energy: bool = True
    lifter: int = 22
    logger: Optional[logging.Logger] = None

    # Computed in __post_init__
    _frame_length: int = field(init=False, repr=False)
    _frame_hop: int = field(init=False, repr=False)
    _nfft: int = field(init=False, repr=False)
    _mel_filterbank: np.ndarray = field(init=False, repr=False)
    _dct_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._frame_length = int(self.sampling_rate * self.frame_length_ms / 1000.0)
        self._frame_hop = int(self.sampling_rate * self.frame_hop_ms / 1000.0)
        # Next power of 2 for FFT efficiency
        self._nfft = 1
        while self._nfft < self._frame_length:
            self._nfft *= 2

        self._mel_filterbank = self._build_mel_filterbank()
        self._dct_matrix = self._build_dct_matrix()

        if self.logger:
            self.logger.info(
                f"[EMGMFCCExtractor] frame={self._frame_length} samples "
                f"({self.frame_length_ms} ms), hop={self._frame_hop} samples "
                f"({self.frame_hop_ms} ms), nfft={self._nfft}, "
                f"n_mels={self.n_mels}, n_mfcc={self.n_mfcc}, "
                f"freq=[{self.fmin}, {self.fmax}] Hz, "
                f"deltas={self.use_deltas}"
            )

    # ── Mel filterbank construction ───────────────────────────────────────

    def _build_mel_filterbank(self) -> np.ndarray:
        """
        Build triangular mel-spaced filterbank matrix.

        Returns:
            (n_mels, nfft//2 + 1) filterbank matrix.
        """
        n_freqs = self._nfft // 2 + 1
        fmax_safe = min(self.fmax, self.sampling_rate / 2.0 - 1.0)

        # Mel-spaced center frequencies
        mel_min = hz_to_mel(self.fmin)
        mel_max = hz_to_mel(fmax_safe)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert Hz to FFT bin indices
        bin_indices = np.floor((self._nfft + 1) * hz_points / self.sampling_rate).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_freqs - 1)

        # Build triangular filters
        filterbank = np.zeros((self.n_mels, n_freqs), dtype=np.float64)
        for m in range(self.n_mels):
            f_left = bin_indices[m]
            f_center = bin_indices[m + 1]
            f_right = bin_indices[m + 2]

            # Rising slope
            if f_center > f_left:
                filterbank[m, f_left:f_center + 1] = (
                    np.arange(f_left, f_center + 1) - f_left
                ) / (f_center - f_left)
            # Falling slope
            if f_right > f_center:
                filterbank[m, f_center:f_right + 1] = (
                    f_right - np.arange(f_center, f_right + 1)
                ) / (f_right - f_center)

        return filterbank.astype(np.float32)

    # ── DCT matrix ────────────────────────────────────────────────────────

    def _build_dct_matrix(self) -> np.ndarray:
        """
        Build type-II DCT matrix for mel-to-cepstral conversion.

        Returns:
            (n_mfcc, n_mels) DCT matrix.
        """
        n = np.arange(self.n_mels)
        k = np.arange(self.n_mfcc)
        # DCT-II: D[k, n] = cos(π k (2n+1) / (2 N))
        dct = np.cos(np.pi * k[:, None] * (2.0 * n[None, :] + 1.0) / (2.0 * self.n_mels))
        # Orthonormal scaling
        dct[0, :] *= 1.0 / np.sqrt(self.n_mels)
        dct[1:, :] *= np.sqrt(2.0 / self.n_mels)
        return dct.astype(np.float32)

    # ── Core: single channel MFCC ─────────────────────────────────────────

    def _compute_mfcc_single_channel(self, x: np.ndarray) -> np.ndarray:
        """
        Compute MFCC for a single 1D signal.

        Args:
            x: (T,) — single-channel time-domain signal.

        Returns:
            mfcc: (n_coeff, T_frames) — MFCC matrix.
                  n_coeff = n_mfcc if not use_deltas else n_mfcc * 3.
        """
        # Pre-emphasis
        if self.pre_emphasis > 0:
            x = np.append(x[0], x[1:] - self.pre_emphasis * x[:-1])

        # Framing
        frames = self._frame_signal(x)  # (n_frames, frame_length)
        if len(frames) == 0:
            n_coeff = self.n_mfcc * 3 if self.use_deltas else self.n_mfcc
            return np.zeros((n_coeff, 1), dtype=np.float32)

        # Windowing (Hamming)
        window = np.hamming(self._frame_length).astype(np.float32)
        frames = frames * window[None, :]

        # Power spectrum
        spec = rfft(frames, n=self._nfft, axis=1)
        power_spec = np.abs(spec) ** 2 / self._nfft  # (n_frames, nfft//2+1)

        # Frame energy (for c0 replacement)
        frame_energy = np.sum(power_spec, axis=1)  # (n_frames,)
        frame_energy = np.maximum(frame_energy, 1e-10)

        # Mel filterbank application
        mel_spec = power_spec @ self._mel_filterbank.T  # (n_frames, n_mels)
        mel_spec = np.maximum(mel_spec, 1e-10)

        # Log mel spectrum
        log_mel = np.log(mel_spec)  # (n_frames, n_mels)

        # DCT → cepstral coefficients
        mfcc = log_mel @ self._dct_matrix.T  # (n_frames, n_mfcc)

        # Replace c0 with log-energy
        if self.use_energy:
            mfcc[:, 0] = np.log(frame_energy)

        # Cepstral liftering
        if self.lifter > 0:
            n = np.arange(self.n_mfcc)
            lift = 1.0 + (self.lifter / 2.0) * np.sin(np.pi * n / self.lifter)
            mfcc *= lift[None, :]

        mfcc = mfcc.T  # (n_mfcc, n_frames)

        # Delta and delta-delta
        if self.use_deltas:
            delta = self._compute_deltas(mfcc)
            delta2 = self._compute_deltas(delta)
            mfcc = np.concatenate([mfcc, delta, delta2], axis=0)  # (3*n_mfcc, n_frames)

        return mfcc.astype(np.float32)

    # ── Framing ───────────────────────────────────────────────────────────

    def _frame_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Split signal into overlapping frames.

        Args:
            x: (T,) signal.

        Returns:
            (n_frames, frame_length) array. Zero-padded if necessary.
        """
        T = len(x)
        if T < self._frame_length:
            # Pad short signals
            x = np.pad(x, (0, self._frame_length - T), mode='constant')
            T = self._frame_length

        n_frames = 1 + (T - self._frame_length) // self._frame_hop
        indices = (
            np.arange(self._frame_length)[None, :]
            + np.arange(n_frames)[:, None] * self._frame_hop
        )
        return x[indices]

    # ── Delta computation ─────────────────────────────────────────────────

    @staticmethod
    def _compute_deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
        """
        Compute delta (differential) features using regression formula.

        Args:
            feat: (F, T) feature matrix.
            width: number of frames on each side for regression.

        Returns:
            (F, T) delta features.
        """
        padded = np.pad(feat, ((0, 0), (width, width)), mode='edge')
        denom = 2.0 * sum(n ** 2 for n in range(1, width + 1))
        delta = np.zeros_like(feat)
        for n in range(1, width + 1):
            delta += n * (padded[:, width + n:width + n + feat.shape[1]]
                          - padded[:, width - n:width - n + feat.shape[1]])
        return delta / denom

    # ── Core: single channel log-mel filterbank ─────────────────────────

    def _compute_fbanks_single_channel(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log-mel filterbank energies for a single 1D signal.

        This is the MFCC pipeline *before* the DCT step — preserves full
        spectral detail in mel space (no cepstral compression).

        Args:
            x: (T,) — single-channel time-domain signal.

        Returns:
            fbanks: (n_coeff, T_frames)
                    n_coeff = n_mels * 3 if use_deltas else n_mels.
        """
        # Pre-emphasis
        if self.pre_emphasis > 0:
            x = np.append(x[0], x[1:] - self.pre_emphasis * x[:-1])

        # Framing
        frames = self._frame_signal(x)  # (n_frames, frame_length)
        if len(frames) == 0:
            n_coeff = self.n_mels * 3 if self.use_deltas else self.n_mels
            return np.zeros((n_coeff, 1), dtype=np.float32)

        # Windowing (Hamming)
        window = np.hamming(self._frame_length).astype(np.float32)
        frames = frames * window[None, :]

        # Power spectrum
        spec = rfft(frames, n=self._nfft, axis=1)
        power_spec = np.abs(spec) ** 2 / self._nfft  # (n_frames, nfft//2+1)

        # Mel filterbank application
        mel_spec = power_spec @ self._mel_filterbank.T  # (n_frames, n_mels)
        mel_spec = np.maximum(mel_spec, 1e-10)

        # Log mel spectrum — this IS the filterbank feature
        log_mel = np.log(mel_spec)  # (n_frames, n_mels)

        fbanks = log_mel.T  # (n_mels, n_frames)

        # Delta and delta-delta
        if self.use_deltas:
            delta = self._compute_deltas(fbanks)
            delta2 = self._compute_deltas(delta)
            fbanks = np.concatenate([fbanks, delta, delta2], axis=0)

        return fbanks.astype(np.float32)

    # ── Public API: fbanks spectrogram mode ──────────────────────────────

    def transform_fbanks_spectrogram(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log-mel filterbank spectrograms for a batch of EMG windows.

        Args:
            X: (N, T, C) — batch of raw EMG windows.

        Returns:
            (N, n_coeff, T_frames, C) — log-mel filterbank spectrograms.
            n_coeff = n_mels * 3 if use_deltas else n_mels.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        N, T, C = X.shape
        sample = self._compute_fbanks_single_channel(X[0, :, 0])
        n_coeff, T_frames = sample.shape

        result = np.zeros((N, n_coeff, T_frames, C), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                result[i, :, :, c] = self._compute_fbanks_single_channel(X[i, :, c])

        return result

    # ── Public API: fbanks flat features mode ────────────────────────────

    def transform_fbanks(self, X: np.ndarray) -> np.ndarray:
        """
        Compute flat log-mel filterbank feature vectors for a batch of EMG windows.

        Same statistics as transform() but on log-mel features (no DCT).

        Args:
            X: (N, T, C) — batch of raw EMG windows.

        Returns:
            (N, F) — flat feature vectors.
            F = C * n_coeff * 6 (6 statistics per coefficient per channel).
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        spectrograms = self.transform_fbanks_spectrogram(X)
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
                f"[EMGMFCCExtractor] Fbanks flat features: {features.shape[1]} "
                f"(n_coeff={n_coeff}, C={C}, 6 stats)"
            )

        return features

    # ── Core: single channel MDCT ────────────────────────────────────────

    def _compute_mdct_single_channel(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Modified Discrete Cosine Transform for a single 1D signal.

        MDCT uses 50%-overlapping frames with a sine analysis window and
        DCT-IV to produce N/2 coefficients per N-sample frame.  This gives
        critically-sampled, alias-free time-frequency representation — the
        standard in MP3/AAC/Vorbis audio codecs.

        Key properties vs STFT:
          - No spectral leakage artifacts at frame boundaries (TDAC)
          - Real-valued output (no phase), N/2 bins (compact)
          - Sine window is optimal for 50% overlap-add reconstruction

        Args:
            x: (T,) — single-channel time-domain signal.

        Returns:
            mdct: (n_coeff, T_frames)
                  n_coeff = (frame_length // 2) * 3 if use_deltas
                            else frame_length // 2.
        """
        N = self._frame_length  # frame length
        M = N // 2              # MDCT output bins per frame
        hop = M                 # 50% overlap (MDCT requirement)

        # Pre-emphasis
        if self.pre_emphasis > 0:
            x = np.append(x[0], x[1:] - self.pre_emphasis * x[:-1])

        T = len(x)
        if T < N:
            x = np.pad(x, (0, N - T), mode='constant')
            T = N

        # Frame the signal with 50% overlap
        n_frames = 1 + (T - N) // hop
        if n_frames <= 0:
            n_coeff = M * 3 if self.use_deltas else M
            return np.zeros((n_coeff, 1), dtype=np.float32)

        indices = (
            np.arange(N)[None, :]
            + np.arange(n_frames)[:, None] * hop
        )
        frames = x[indices]  # (n_frames, N)

        # Sine analysis window (optimal for MDCT with 50% overlap)
        n_idx = np.arange(N)
        window = np.sin(np.pi / N * (n_idx + 0.5)).astype(np.float32)
        frames = frames * window[None, :]

        # DCT-IV: X[k] = Σ_n x[n] cos(π/M (n + 0.5 + M/2)(k + 0.5))
        # Vectorized over all frames
        n = np.arange(N).astype(np.float64)
        k = np.arange(M).astype(np.float64)
        # Basis matrix: (M, N)
        basis = np.cos(
            np.pi / M * np.outer(k + 0.5, n + 0.5 + M / 2.0)
        ).astype(np.float32)

        mdct_coeffs = frames @ basis.T  # (n_frames, M)

        # Log-magnitude (like log-mel in MFCC)
        mdct_log = np.log(np.abs(mdct_coeffs) + 1e-10)  # (n_frames, M)

        mdct_out = mdct_log.T  # (M, n_frames)

        # Delta and delta-delta
        if self.use_deltas:
            delta = self._compute_deltas(mdct_out)
            delta2 = self._compute_deltas(delta)
            mdct_out = np.concatenate([mdct_out, delta, delta2], axis=0)

        return mdct_out.astype(np.float32)

    # ── Public API: MDCT spectrogram mode ────────────────────────────────

    def transform_mdct_spectrogram(self, X: np.ndarray) -> np.ndarray:
        """
        Compute MDCT spectrograms for a batch of multi-channel EMG windows.

        Args:
            X: (N, T, C) — batch of raw EMG windows.

        Returns:
            (N, n_coeff, T_frames, C) — MDCT spectrograms per channel.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        N, T, C = X.shape
        sample = self._compute_mdct_single_channel(X[0, :, 0])
        n_coeff, T_frames = sample.shape

        result = np.zeros((N, n_coeff, T_frames, C), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                result[i, :, :, c] = self._compute_mdct_single_channel(X[i, :, c])

        return result

    # ── Public API: MDCT flat features mode ──────────────────────────────

    def transform_mdct(self, X: np.ndarray) -> np.ndarray:
        """
        Compute flat MDCT-based feature vectors for a batch of EMG windows.

        Args:
            X: (N, T, C) — batch of raw EMG windows.

        Returns:
            (N, F) — flat feature vectors (6 statistics per coeff per channel).
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        spectrograms = self.transform_mdct_spectrogram(X)
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
                f"[EMGMFCCExtractor] MDCT flat features: {features.shape[1]} "
                f"(n_coeff={n_coeff}, C={C}, 6 stats)"
            )

        return features

    # ── Public API: spectrogram mode ──────────────────────────────────────

    def transform_spectrogram(self, X: np.ndarray) -> np.ndarray:
        """
        Compute MFCC spectrograms for a batch of multi-channel EMG windows.

        Args:
            X: (N, T, C) — batch of raw EMG windows.

        Returns:
            (N, n_coeff, T_frames, C) — MFCC spectrograms per channel.
            n_coeff = n_mfcc * 3 if use_deltas else n_mfcc.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        N, T, C = X.shape
        # Compute for first sample to get output shape
        sample_mfcc = self._compute_mfcc_single_channel(X[0, :, 0])
        n_coeff, T_frames = sample_mfcc.shape

        result = np.zeros((N, n_coeff, T_frames, C), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                result[i, :, :, c] = self._compute_mfcc_single_channel(X[i, :, c])

        return result

    # ── Public API: flat features mode ────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute flat MFCC-based feature vectors for a batch of EMG windows.

        For each channel and each MFCC coefficient, computes statistics over
        the time frames: mean, std, min, max, skew, kurtosis.

        Args:
            X: (N, T, C) — batch of raw EMG windows.

        Returns:
            (N, F) — flat feature vectors.
            F = C * n_coeff * 6 (6 statistics per coefficient per channel).
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        # (N, n_coeff, T_frames, C)
        spectrograms = self.transform_spectrogram(X)
        N, n_coeff, T_frames, C = spectrograms.shape

        # Statistics over time axis
        eps = 1e-8
        mean = spectrograms.mean(axis=2)                       # (N, n_coeff, C)
        std = spectrograms.std(axis=2) + eps                   # (N, n_coeff, C)
        vmin = spectrograms.min(axis=2)                        # (N, n_coeff, C)
        vmax = spectrograms.max(axis=2)                        # (N, n_coeff, C)

        centered = spectrograms - mean[:, :, None, :]
        m3 = (centered ** 3).mean(axis=2)
        m4 = (centered ** 4).mean(axis=2)
        skew = m3 / (std ** 3 + eps)                           # (N, n_coeff, C)
        kurt = m4 / (std ** 4 + eps) - 3.0                     # (N, n_coeff, C)

        # Stack and flatten: (N, n_coeff, C, 6) → (N, F)
        stats = np.stack([mean, std, vmin, vmax, skew, kurt], axis=-1)  # (N, n_coeff, C, 6)
        features = stats.reshape(N, -1).astype(np.float32)

        if self.logger:
            self.logger.info(
                f"[EMGMFCCExtractor] Flat features: {features.shape[1]} "
                f"(n_coeff={n_coeff}, C={C}, 6 stats)"
            )

        return features

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def n_coeff(self) -> int:
        """Number of MFCC coefficients (including deltas if enabled)."""
        return self.n_mfcc * 3 if self.use_deltas else self.n_mfcc

    @property
    def n_fbanks_coeff(self) -> int:
        """Number of filterbank coefficients (including deltas if enabled)."""
        return self.n_mels * 3 if self.use_deltas else self.n_mels

    @property
    def n_mdct_coeff(self) -> int:
        """Number of MDCT coefficients (including deltas if enabled)."""
        M = self._frame_length // 2
        return M * 3 if self.use_deltas else M

    def get_n_frames(self, window_length: int) -> int:
        """Compute number of output frames for a given window length."""
        if window_length < self._frame_length:
            return 1
        return 1 + (window_length - self._frame_length) // self._frame_hop
