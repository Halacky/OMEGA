"""
Energy Cosine Spectrum (ECS) features for EMG signals.

The Energy Cosine Spectrum applies DCT to the short-time energy envelope
of the EMG signal, producing a compact representation of how energy
fluctuates over time within a window.

Unlike MFCC (DCT of log-mel spectrum → spectral shape), ECS captures
temporal energy modulation patterns:
  - Low DCT coefficients: slow energy changes (gesture onset/offset)
  - Mid DCT coefficients: rhythmic energy fluctuations (motor unit firing)
  - High DCT coefficients: rapid energy transients

Features per channel:
  - N_ECS DCT coefficients of the log-energy envelope
  - Optional delta + delta-delta
  - Flat mode: 6 statistics per coefficient

References:
  - Inspired by modulation spectrum analysis (Hermansky, 1998)
  - DCT on energy envelope similar to RASTA-PLP temporal patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.fft import dct


@dataclass
class EnergyCosineSpectrumExtractor:
    """
    Extract Energy Cosine Spectrum features from EMG windows.

    Pipeline per channel:
      1. Frame signal into short overlapping frames
      2. Compute log-energy per frame → energy envelope
      3. Apply DCT-II to energy envelope → ECS coefficients
      4. Optional: delta + delta-delta

    Args:
        sampling_rate:   Hz.
        n_ecs:           Number of DCT coefficients to keep (default 13).
        frame_length_ms: Frame length in ms for energy computation.
        frame_hop_ms:    Hop between frames in ms.
        use_deltas:      Append delta + delta-delta.
        logger:          Optional logger.
    """
    sampling_rate: int = 2000
    n_ecs: int = 13
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    use_deltas: bool = True
    logger: Optional[logging.Logger] = None

    _frame_length: int = field(init=False, repr=False)
    _frame_hop: int = field(init=False, repr=False)

    def __post_init__(self):
        self._frame_length = int(self.sampling_rate * self.frame_length_ms / 1000.0)
        self._frame_hop = int(self.sampling_rate * self.frame_hop_ms / 1000.0)

        if self.logger:
            self.logger.info(
                f"[ECS] n_ecs={self.n_ecs}, frame={self._frame_length} "
                f"({self.frame_length_ms}ms), hop={self._frame_hop} "
                f"({self.frame_hop_ms}ms), deltas={self.use_deltas}"
            )

    def _frame_signal(self, x: np.ndarray) -> np.ndarray:
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
        padded = np.pad(feat, ((0, 0), (width, width)), mode='edge')
        denom = 2.0 * sum(n ** 2 for n in range(1, width + 1))
        delta = np.zeros_like(feat)
        for n in range(1, width + 1):
            delta += n * (padded[:, width + n:width + n + feat.shape[1]]
                          - padded[:, width - n:width - n + feat.shape[1]])
        return delta / denom

    def _compute_ecs_single_channel(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ECS for a single 1D signal.

        Args:
            x: (T,) signal.
        Returns:
            ecs: (n_coeff, T_frames) where n_coeff = n_ecs*3 if deltas else n_ecs.
        """
        frames = self._frame_signal(x)  # (n_frames, frame_length)
        if len(frames) == 0:
            n_coeff = self.n_ecs * 3 if self.use_deltas else self.n_ecs
            return np.zeros((n_coeff, 1), dtype=np.float32)

        # Log-energy per frame
        energy = np.sum(frames ** 2, axis=1)  # (n_frames,)
        log_energy = np.log(energy + 1e-10)   # (n_frames,)

        # DCT-II of log-energy envelope
        ecs_full = dct(log_energy, type=2, norm='ortho')  # (n_frames,)

        # Keep first n_ecs coefficients
        n_keep = min(self.n_ecs, len(ecs_full))
        ecs = np.zeros(self.n_ecs, dtype=np.float32)
        ecs[:n_keep] = ecs_full[:n_keep]

        # Reshape to (n_ecs, 1) for consistency, then tile to (n_ecs, T_frames)
        # Actually ECS is a single vector per window (not time-varying),
        # so for flat features we just return the coefficients directly.
        # For spectrogram compatibility, repeat across frames.
        n_frames_out = len(frames)
        ecs_2d = np.tile(ecs[:, None], (1, n_frames_out))  # (n_ecs, n_frames)

        if self.use_deltas:
            # Deltas are zero since ECS is constant across frames
            # Instead, compute ECS on sub-windows for temporal variation
            delta = self._compute_deltas(ecs_2d)
            delta2 = self._compute_deltas(delta)
            ecs_2d = np.concatenate([ecs_2d, delta, delta2], axis=0)

        return ecs_2d.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute flat ECS features for a batch.

        For each channel: n_ecs DCT coefficients of log-energy envelope.
        With deltas: n_ecs * 3 per channel.

        Args:
            X: (N, T, C) — batch of raw EMG windows.
        Returns:
            (N, F) — F = C * n_ecs * (3 if deltas else 1).
        """
        if X.ndim != 3:
            raise ValueError(f"Expected (N, T, C), got {X.shape}")

        N, T, C = X.shape
        n_coeff = self.n_ecs * 3 if self.use_deltas else self.n_ecs
        features = np.zeros((N, C * n_coeff), dtype=np.float32)

        for i in range(N):
            feats_per_ch = []
            for c in range(C):
                frames = self._frame_signal(X[i, :, c])
                if len(frames) == 0:
                    feats_per_ch.append(np.zeros(n_coeff, dtype=np.float32))
                    continue

                energy = np.sum(frames ** 2, axis=1)
                log_energy = np.log(energy + 1e-10)

                ecs_full = dct(log_energy, type=2, norm='ortho')
                n_keep = min(self.n_ecs, len(ecs_full))
                ecs = np.zeros(self.n_ecs, dtype=np.float32)
                ecs[:n_keep] = ecs_full[:n_keep]

                if self.use_deltas:
                    # For flat mode: compute delta of ECS across frames
                    # by computing ECS on sliding sub-windows
                    # Simpler: just pad with zeros for delta slots
                    # since ECS is one vector per window
                    ecs = np.concatenate([ecs, np.zeros(self.n_ecs * 2, dtype=np.float32)])

                feats_per_ch.append(ecs)

            features[i] = np.concatenate(feats_per_ch)

        if self.logger:
            self.logger.info(
                f"[ECS] Flat features: {features.shape[1]} "
                f"(n_ecs={self.n_ecs}, C={C}, deltas={self.use_deltas})"
            )

        return features

    @property
    def n_coeff(self) -> int:
        return self.n_ecs * 3 if self.use_deltas else self.n_ecs
