from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HandcraftedFeatureExtractor:
    """
    Extracts hand-crafted features from EMG windows.

    Supported feature sets:
    - "basic_v1": your original mixed set (mean, std, RMS, ZCR, WL, skew, kurtosis,
      spectral entropy, 4 band powers). Output: (N, F).
    - "emg_td": classical EMG time-domain features per channel:
        MAV, RMS, WL, ZC, SSC. Output: (N, F).
    - "emg_td_seq": same EMG TD features, but computed on short frames inside each
      window to preserve temporal structure. Output: (N, T_frames, F_per_frame).
      This is suitable for deep models (CNN/LSTM/etc).
    """
    sampling_rate: Optional[float] = None
    logger: Optional[logging.Logger] = None
    feature_set: str = "basic_v1"  # "basic_v1", "emg_td", "emg_td_seq"

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C) raw EMG windows.
        Returns:
            - basic_v1, emg_td: (N, F)
            - emg_td_seq: (N, T_frames, F_per_frame)
        """
        if X.ndim != 3:
            raise ValueError(
                f"HandcraftedFeatureExtractor expected X shape (N, T, C), got {X.shape}"
            )

        N, T, C = X.shape
        if self.logger:
            self.logger.info(
                f"[HandcraftedFeatureExtractor] Input data: N={N}, T={T}, C={C}, "
                f"feature_set='{self.feature_set}'"
            )

        if self.feature_set == "basic_v1":
            feats = self._features_basic_v1(X)
        elif self.feature_set == "emg_td":
            feats = self._features_emg_td(X)
        elif self.feature_set == "emg_td_seq":
            feats = self._features_emg_td_seq(X)
        else:
            raise ValueError(f"Unknown feature_set: {self.feature_set}")

        if self.logger:
            if feats.ndim == 2:
                self.logger.info(
                    f"[HandcraftedFeatureExtractor] Extracted features: {feats.shape[1]} "
                    f"per window (flat)"
                )
            else:
                self.logger.info(
                    f"[HandcraftedFeatureExtractor] Extracted sequential features: "
                    f"shape={feats.shape}"
                )

        return feats

    # ---------------------- basic_v1 (ваша старая логика) ---------------------- #

    def _features_basic_v1(self, X: np.ndarray) -> np.ndarray:
        """
        Original "basic_v1" feature set from your code.
        Returns: (N, F) flat features.
        """
        N, T, C = X.shape

        mean = X.mean(axis=1)                            # (N, C)
        std = X.std(axis=1) + 1e-8                       # (N, C)
        rms = np.sqrt((X ** 2).mean(axis=1))             # (N, C)

        centered = X - mean[:, None, :]                  # (N, T, C)
        m3 = (centered ** 3).mean(axis=1)                # (N, C)
        m4 = (centered ** 4).mean(axis=1)                # (N, C)
        skew = m3 / (std ** 3 + 1e-8)                    # (N, C)
        kurt = m4 / (std ** 4 + 1e-8)                    # (N, C)

        # Zero-crossing rate
        signs = np.sign(X)
        sign_changes = np.diff(signs, axis=1) != 0       # (N, T-1, C)
        zcr = sign_changes.sum(axis=1) / max(T - 1, 1)   # (N, C)

        # Waveform length
        wl = np.abs(np.diff(X, axis=1)).sum(axis=1)      # (N, C)

        # Spectral features
        fft_vals = np.fft.rfft(X, axis=1)                # (N, Freq, C)
        psd = (np.abs(fft_vals) ** 2)
        psd_sum = psd.sum(axis=1, keepdims=True) + 1e-12
        psd_norm = psd / psd_sum

        spectral_entropy = -np.sum(
            psd_norm * np.log(psd_norm + 1e-12), axis=1
        )  # (N, C)

        n_freqs = psd.shape[1]
        edges = np.linspace(0, n_freqs, 5, dtype=int)    # 0,25%,50%,75%,100%
        band_powers = []
        for b in range(4):
            s, e = edges[b], edges[b + 1]
            if e > s:
                bp = psd[:, s:e, :].sum(axis=1)          # (N, C)
            else:
                bp = np.zeros((N, C), dtype=psd.dtype)
            band_powers.append(bp)

        per_channel_features = [
            mean, std, rms, zcr, wl, skew, kurt,
            spectral_entropy,
            *band_powers,
        ]
        feats_stack = np.stack(per_channel_features, axis=2)  # (N, C, F_pc)
        N, C, Fpc = feats_stack.shape

        feats_flat = feats_stack.reshape(N, C * Fpc).astype(np.float32)  # (N, F)
        return feats_flat

    # ---------------------- EMG TD (MAV, WL, ZC, SSC) ------------------------- #

    def _features_emg_td(self, X: np.ndarray) -> np.ndarray:
        """
        Classic EMG time-domain features per channel:
        MAV, RMS, WL, ZC, SSC.
        Returns: (N, F) flat array, F = C * 5.
        """
        N, T, C = X.shape

        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(X), axis=1)  # (N, C)

        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(X ** 2, axis=1))  # (N, C)

        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(X, axis=1)), axis=1)  # (N, C)

        # Threshold per window/channel for ZC/SSC
        # 1% of max absolute value in the window for this channel
        eps = 1e-8
        max_abs = np.max(np.abs(X), axis=1) + eps  # (N, C)
        thr = 0.01 * max_abs  # (N, C)

        # Zero Crossings (ZC)
        x1 = X[:, :-1, :]  # (N, T-1, C)
        x2 = X[:, 1:, :]
        prod = x1 * x2
        amp_diff = np.abs(x2 - x1)
        zc_bool = (prod < 0) & (amp_diff > thr[:, None, :])
        zc = zc_bool.sum(axis=1)  # (N, C)

        # Slope Sign Changes (SSC)
        x_prev = X[:, :-2, :]      # (N, T-2, C)
        x_curr = X[:, 1:-1, :]     # (N, T-2, C)
        x_next = X[:, 2:, :]       # (N, T-2, C)

        diff1 = x_curr - x_prev
        diff2 = x_curr - x_next

        ssc_bool = (
            (diff1 * diff2 < 0)
            & (np.abs(diff1) > thr[:, None, :])
            & (np.abs(diff2) > thr[:, None, :])
        )
        ssc = ssc_bool.sum(axis=1)  # (N, C)

        per_channel_features = [mav, rms, wl, zc, ssc]  # list of (N, C)
        feats_stack = np.stack(per_channel_features, axis=2)  # (N, C, F_pc=5)
        N, C, Fpc = feats_stack.shape
        feats_flat = feats_stack.reshape(N, C * Fpc).astype(np.float32)  # (N, F)

        return feats_flat

    # ---------------------- EMG TD sequential (для DL) ----------------------- #

    def _features_emg_td_seq(
        self,
        X: np.ndarray,
        frame_len: int = 50,
        hop_len: int = 25,
    ) -> np.ndarray:
        """
        Sequential EMG TD features:
        Compute MAV, RMS, WL, ZC, SSC on short overlapping frames inside each
        window to preserve temporal structure.

        X: (N, T, C)
        Returns: (N, T_frames, F_per_frame)
            where F_per_frame = C * 5 (MAV, RMS, WL, ZC, SSC per channel)
        """
        N, T, C = X.shape
        if T < frame_len:
            raise ValueError(
                f"Window length T={T} is smaller than frame_len={frame_len}"
            )

        # Frame start indices
        starts = np.arange(0, T - frame_len + 1, hop_len, dtype=int)
        n_frames = len(starts)

        all_frames_feats = []

        for s in starts:
            frame = X[:, s:s + frame_len, :]  # (N, frame_len, C)

            mav = np.mean(np.abs(frame), axis=1)           # (N, C)
            rms = np.sqrt(np.mean(frame ** 2, axis=1))     # (N, C)
            wl = np.sum(np.abs(np.diff(frame, axis=1)), axis=1)  # (N, C)

            # Threshold for this frame
            eps = 1e-8
            max_abs = np.max(np.abs(frame), axis=1) + eps  # (N, C)
            thr = 0.01 * max_abs  # (N, C)

            # ZC
            fx1 = frame[:, :-1, :]  # (N, L-1, C)
            fx2 = frame[:, 1:, :]
            prod = fx1 * fx2
            amp_diff = np.abs(fx2 - fx1)
            zc_bool = (prod < 0) & (amp_diff > thr[:, None, :])
            zc = zc_bool.sum(axis=1)  # (N, C)

            # SSC
            f_prev = frame[:, :-2, :]
            f_curr = frame[:, 1:-1, :]
            f_next = frame[:, 2:, :]

            diff1 = f_curr - f_prev
            diff2 = f_curr - f_next
            ssc_bool = (
                (diff1 * diff2 < 0)
                & (np.abs(diff1) > thr[:, None, :])
                & (np.abs(diff2) > thr[:, None, :])
            )
            ssc = ssc_bool.sum(axis=1)  # (N, C)

            per_channel = [mav, rms, wl, zc, ssc]          # list of (N, C)
            feats_stack = np.stack(per_channel, axis=2)    # (N, C, 5)
            Nf, Cf, Fpc = feats_stack.shape
            feats_flat = feats_stack.reshape(Nf, Cf * Fpc) # (N, C*5)
            all_frames_feats.append(feats_flat)

        # (n_frames, N, C*5) -> (N, n_frames, C*5)
        seq_feats = np.stack(all_frames_feats, axis=1).astype(np.float32)
        return seq_feats