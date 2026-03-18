"""
Experiment 38: Nonlinear Statistics + Channel-Pair Features for Cross-Subject EMG (LOSO)

Hypothesis H38:
    High-level nonlinear statistics (sample entropy, permutation entropy, Higuchi
    fractal dimension, Hjorth parameters, approximate Lyapunov exponent) and
    channel-pair features (cross-correlation, spectral coherence, mutual information)
    capture subject-invariant muscle activation dynamics better than classical
    amplitude/power descriptors (RMS, MAV, PSD).

    Combining these features with the existing PowerfulFeatureExtractor and feeding
    them into SVM (RBF / linear) or LightGBM should outperform exp_4
    (PowerfulFeatureExtractor + SVM alone).

Feature groups:
    Per-channel (7 features × C channels):
        - Hjorth activity, mobility, complexity
        - Sample entropy  (m=2, r=0.2, downsampled ×4)
        - Permutation entropy  (ordinal pattern length m=4)
        - Higuchi fractal dimension  (kmax=8)
        - Approximate largest Lyapunov exponent  (Rosenstein-style, 1-D)

    Channel-pair (8 features × C*(C-1)/2 pairs):
        - Cross-correlation: peak value, normalised lag, mean absolute
        - Spectral coherence: mean in 4 EMG sub-bands (20-80, 80-150, 150-300, 300-500 Hz)
        - Mutual information  (histogram-based, 10 bins)

    Combined with PowerfulFeatureExtractor (use_entropy=False to avoid placeholder zeros).

For C=8, T=600:
    NonlinearEMGExtractor:  7×8  + 8×28  = 56 + 224 = 280 features
    PowerfulFeatureExtractor (no entropy): ~253 features
    CombinedNonlinearExtractor total: ~533 features

Usage:
    python experiments/exp_38_nonlinear_stats_svm_lgbm_loso.py          # CI subjects (default)
    python experiments/exp_38_nonlinear_stats_svm_lgbm_loso.py --ci     # same
    python experiments/exp_38_nonlinear_stats_svm_lgbm_loso.py --full   # 20 subjects
    python experiments/exp_38_nonlinear_stats_svm_lgbm_loso.py --subjects DB2_s1,DB2_s12
"""

import sys
import json
import math
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Subject lists
# ---------------------------------------------------------------------------

_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]

_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]


def parse_subjects_args(argv=None) -> List[str]:
    """Parse --subjects / --ci / --full CLI args. Defaults to CI subjects."""
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject IDs")
    _parser.add_argument("--ci",   action="store_true",
                         help="Use CI test subset (5 subjects)")
    _parser.add_argument("--full", action="store_true",
                         help="Use full 20-subject set")
    _args, _ = _parser.parse_known_args(argv)

    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    # Default: CI subjects (server has symlinks only for these)
    return _CI_SUBJECTS


# ===========================================================================
#  Nonlinear EMG Feature Extractor
# ===========================================================================

class NonlinearEMGExtractor:
    """
    Extracts nonlinear per-channel and cross-channel features from EMG windows.

    Input:  X  (N, T, C)
    Output:     (N, F)
        where F = C * 7  +  C*(C-1)/2 * 8
    """

    # Coherence frequency bands (Hz)
    _COH_BANDS = [(20, 80), (80, 150), (150, 300), (300, 500)]

    def __init__(
        self,
        sampling_rate: int = 2000,
        entropy_downsample: int = 4,
        perm_m: int = 4,
        higuchi_kmax: int = 8,
        mi_bins: int = 10,
        n_jobs: int = 1,
        cross_channel_batch: int = 1000,
    ):
        """
        Args:
            sampling_rate:       EMG sampling rate in Hz.
            entropy_downsample:  Downsample factor before computing sample entropy.
            perm_m:              Ordinal pattern length for permutation entropy.
            higuchi_kmax:        Max k for Higuchi fractal dimension.
            mi_bins:             Number of bins for histogram-based mutual information.
            n_jobs:              Parallelism for slow per-window loops (sample entropy,
                                 Lyapunov). -1 = all cores.
            cross_channel_batch: Process cross-channel features in batches of this many
                                 windows (memory control).
        """
        self.fs = sampling_rate
        self.entropy_downsample = max(1, entropy_downsample)
        self.perm_m = perm_m
        self.higuchi_kmax = higuchi_kmax
        self.mi_bins = mi_bins
        self.n_jobs = n_jobs
        self.cross_channel_batch = cross_channel_batch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (N, T, C) EMG windows
        Returns:
            features: (N, F) float32
        """
        if X.ndim != 3:
            raise ValueError(f"Expected (N, T, C), got {X.shape}")

        per_ch = self._per_channel_features(X)    # (N, C*7)
        cross_ch = self._cross_channel_features(X)  # (N, n_pairs*8)

        out = np.concatenate([per_ch, cross_ch], axis=1)
        out = np.nan_to_num(out.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # ------------------------------------------------------------------
    # Per-channel features
    # ------------------------------------------------------------------

    def _per_channel_features(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C)
        Returns: (N, C * 7)  [act, mob, comp, samp_ent, perm_ent, hfd, lya]
        """
        N, T, C = X.shape
        feats = np.zeros((N, C, 7), dtype=np.float64)

        for c in range(C):
            sig = X[:, :, c]  # (N, T)

            # --- Hjorth parameters (fully vectorised) ---
            d1 = np.diff(sig, axis=1)   # (N, T-1)
            d2 = np.diff(d1,  axis=1)   # (N, T-2)

            act     = np.var(sig, axis=1)               # (N,) activity
            var_d1  = np.var(d1,  axis=1)
            var_d2  = np.var(d2,  axis=1)

            mob  = np.sqrt(var_d1 / (act + 1e-12))      # mobility
            comp = (np.sqrt(var_d2 / (var_d1 + 1e-12))
                    / (mob + 1e-12))                     # complexity

            feats[:, c, 0] = act
            feats[:, c, 1] = mob
            feats[:, c, 2] = comp

            # --- Sample entropy (parallelised over N) ---
            feats[:, c, 3] = self._batch_sample_entropy(sig)

            # --- Permutation entropy (vectorised) ---
            feats[:, c, 4] = self._batch_perm_entropy(sig)

            # --- Higuchi fractal dimension (vectorised) ---
            feats[:, c, 5] = self._batch_higuchi_fd(sig)

            # --- Approximate Lyapunov exponent (parallelised over N) ---
            feats[:, c, 6] = self._batch_lyapunov(sig)

        return feats.reshape(N, -1)

    # ------------------------------------------------------------------
    # Sample entropy
    # ------------------------------------------------------------------

    def _batch_sample_entropy(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T) — one channel
        Returns: (N,) sample entropy values
        """
        X_ds = X[:, ::self.entropy_downsample]  # (N, T_ds)
        N = len(X_ds)

        if self.n_jobs == 1:
            return np.array([self._sample_entropy(X_ds[i]) for i in range(N)])
        return np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._sample_entropy)(X_ds[i]) for i in range(N)
            )
        )

    @staticmethod
    def _sample_entropy(sig: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Sample entropy for a single (downsampled) signal.
        Vectorised Chebyshev-distance count.
        """
        N = len(sig)
        if N <= m + 2:
            return 0.0
        std = np.std(sig)
        if std < 1e-12:
            return 0.0
        r_abs = r * std

        def _phi(order: int) -> float:
            x = np.lib.stride_tricks.sliding_window_view(sig, order)   # (M, order)
            diff = np.abs(x[:, None, :] - x[None, :, :])               # (M, M, order)
            dist = diff.max(axis=2)                                     # (M, M)
            M = len(x)
            C = (dist <= r_abs).sum(axis=1) - 1                        # exclude self
            return float(C.sum()) / (M * (M - 1) + 1e-12)

        phi_m  = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m <= 0 or phi_m1 <= 0:
            return 0.0
        return float(-np.log(phi_m1 / (phi_m + 1e-12)))

    # ------------------------------------------------------------------
    # Permutation entropy
    # ------------------------------------------------------------------

    def _batch_perm_entropy(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T) — one channel
        Returns: (N,) normalised permutation entropy in [0, 1]

        Ordinal patterns of length m are encoded as base-m integers
        (collision-free for valid permutations with distinct ranks).
        Entropy is normalised by log2(m!).
        """
        N, T = X.shape
        m = self.perm_m
        n_patterns = T - m + 1
        if n_patterns < 1:
            return np.zeros(N)

        # Sliding windows: (N, n_patterns, m)
        windows = np.lib.stride_tricks.sliding_window_view(X, m, axis=1)

        # Ordinal ranks within each sub-window: argsort of argsort = rank
        ranks = np.argsort(np.argsort(windows, axis=2), axis=2).astype(np.int32)

        # Encode each permutation as a base-m integer (unique for distinct digits)
        m_powers = (m ** np.arange(m - 1, -1, -1)).astype(np.int32)  # (m,)
        codes = (ranks * m_powers[np.newaxis, np.newaxis, :]).sum(axis=2)  # (N, n_patterns)

        # Shannon entropy per window
        max_entropy = math.log2(max(1.0, float(math.factorial(m))))
        pe = np.zeros(N, dtype=np.float64)
        for i in range(N):
            _, counts = np.unique(codes[i], return_counts=True)
            probs = counts / float(counts.sum())
            pe[i] = -np.sum(probs * np.log2(probs + 1e-12))

        return pe / (max_entropy + 1e-12)

    # ------------------------------------------------------------------
    # Higuchi fractal dimension
    # ------------------------------------------------------------------

    def _batch_higuchi_fd(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T) — one channel
        Returns: (N,) Higuchi FD estimates

        FD = -(slope of log L(k) vs log k).
        Typical EMG values: 1.4–1.9.
        """
        N, T = X.shape
        kmax = self.higuchi_kmax

        log_L = np.zeros((N, kmax), dtype=np.float64)

        for k in range(1, kmax + 1):
            L_k = np.zeros(N, dtype=np.float64)
            for m in range(1, k + 1):
                n_seg = (T - m) // k         # floor((T-m)/k)
                if n_seg < 1:
                    continue
                # Extract sub-series at spacing k starting from index m-1
                idx = np.arange(m - 1, m - 1 + n_seg * k + 1, k)
                if len(idx) < 2:
                    continue
                sub = X[:, idx]              # (N, n_seg+1)
                diff_abs = np.abs(np.diff(sub, axis=1))  # (N, n_seg)
                # Normalised curve length
                length = (diff_abs.sum(axis=1)
                          * (T - 1) / (n_seg * k))      # (N,)
                L_k += length

            # Average over m=1..k
            log_L[:, k - 1] = np.log(np.abs(L_k / k) + 1e-12)

        # Log-log linear regression: FD = -slope
        log_k   = np.log(np.arange(1, kmax + 1))  # (kmax,)
        k_mean  = log_k.mean()
        L_mean  = log_L.mean(axis=1)              # (N,)

        num   = ((log_k - k_mean) * (log_L - L_mean[:, None])).sum(axis=1)
        denom = ((log_k - k_mean) ** 2).sum()

        return -num / (denom + 1e-12)

    # ------------------------------------------------------------------
    # Approximate Lyapunov exponent
    # ------------------------------------------------------------------

    def _batch_lyapunov(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T) — one channel
        Returns: (N,) approximate largest Lyapunov exponent
        """
        N = len(X)
        if self.n_jobs == 1:
            return np.array([self._approx_lyapunov(X[i]) for i in range(N)])
        return np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._approx_lyapunov)(X[i]) for i in range(N)
            )
        )

    @staticmethod
    def _approx_lyapunov(
        sig: np.ndarray,
        w: int = 15,
        T_future: int = 20,
        max_ref: int = 80,
    ) -> float:
        """
        1-D Rosenstein-style approximate largest Lyapunov exponent.

        For each of max_ref reference points, finds the nearest neighbour
        (temporal exclusion window ±w), then tracks log-divergence over
        T_future steps and fits a slope.

        Args:
            sig:      1-D signal
            w:        temporal exclusion half-window
            T_future: divergence tracking horizon
            max_ref:  max reference points (subsampled)
        """
        T = len(sig)
        if T < 2 * w + T_future + 10:
            return 0.0

        step = max(1, (T - T_future - w) // max_ref)
        diverge_log = np.zeros(T_future, dtype=np.float64)
        count = 0

        for i in range(0, T - T_future - w, step):
            dists = np.abs(sig - sig[i])
            # Temporal exclusion
            lo = max(0, i - w)
            hi = min(T, i + w + 1)
            dists[lo:hi] = np.inf

            j = int(np.argmin(dists))
            d0 = dists[j]
            if not np.isfinite(d0) or d0 < 1e-12:
                continue

            # Track divergence for T_future steps
            end = min(T, min(i, j) + T_future)
            futures = end - max(i, j)
            if futures < 2:
                continue

            sep = np.abs(sig[i:i + futures] - sig[j:j + futures])
            log_sep = np.log(sep / d0 + 1e-12)
            diverge_log[:futures] += log_sep
            count += 1

        if count == 0:
            return 0.0

        diverge_log /= count
        t = np.arange(T_future)
        valid = np.isfinite(diverge_log)
        if valid.sum() < 2:
            return 0.0

        slope = float(np.polyfit(t[valid], diverge_log[valid], 1)[0])
        return slope

    # ------------------------------------------------------------------
    # Cross-channel features
    # ------------------------------------------------------------------

    def _cross_channel_features(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C)
        Returns: (N, n_pairs * 8)
            8 = [corr_peak, corr_lag_norm, corr_mean_abs,
                 coh_band0, coh_band1, coh_band2, coh_band3,
                 mutual_info]

        Processes windows in batches of self.cross_channel_batch to
        limit peak memory consumption.
        """
        N, T, C = X.shape
        n_pairs = C * (C - 1) // 2
        n_coh   = len(self._COH_BANDS)
        n_feat  = 3 + n_coh + 1   # 8

        out = np.zeros((N, n_pairs * n_feat), dtype=np.float64)

        for batch_start in range(0, N, self.cross_channel_batch):
            batch = X[batch_start: batch_start + self.cross_channel_batch]
            out[batch_start: batch_start + len(batch)] = (
                self._cross_channel_batch(batch)
            )
        return out

    def _cross_channel_batch(self, X: np.ndarray) -> np.ndarray:
        """
        X: (Nb, T, C)  — one batch of windows.
        Returns: (Nb, n_pairs * 8)
        """
        Nb, T, C = X.shape
        n_pairs = C * (C - 1) // 2
        n_coh   = len(self._COH_BANDS)
        n_feat  = 3 + n_coh + 1

        feats = np.zeros((Nb, n_pairs * n_feat), dtype=np.float64)

        # Centre and normalise for cross-correlation
        X_c   = X - X.mean(axis=1, keepdims=True)             # (Nb, T, C)
        X_std = X_c.std(axis=1) + 1e-12                        # (Nb, C)
        X_n   = X_c / X_std[:, np.newaxis, :]                 # (Nb, T, C)

        # Zero-padded FFT for cross-correlation  (full xcorr length 2T-1)
        nfft    = 2 * T - 1
        X_fft_xc = np.fft.rfft(X_n, n=nfft, axis=1)          # (Nb, nfft//2+1, C)

        # Unpadded FFT for coherence
        X_fft_coh = np.fft.rfft(X, axis=1)                    # (Nb, T//2+1, C)
        freqs_coh  = np.fft.rfftfreq(T, d=1.0 / self.fs)      # (T//2+1,)

        pair_idx = 0
        for i in range(C):
            for j in range(i + 1, C):
                col = pair_idx * n_feat

                # ── Cross-correlation ────────────────────────────────────
                xcorr_fft = (X_fft_xc[:, :, i].conj()
                             * X_fft_xc[:, :, j])             # (Nb, nfft//2+1)
                xcorr = np.fft.irfft(xcorr_fft, n=nfft, axis=1)  # (Nb, nfft)
                xcorr /= T  # normalise

                peak_idx   = np.argmax(xcorr, axis=1)          # (Nb,)
                peak_vals  = xcorr[np.arange(Nb), peak_idx]
                # Map index → lag: indices 0..T-1 → +lags, T..nfft-1 → -lags
                lags       = np.where(peak_idx < T, peak_idx,
                                      peak_idx.astype(int) - nfft)
                lag_norm   = lags / T
                mean_abs   = np.abs(xcorr).mean(axis=1)

                feats[:, col]     = np.clip(peak_vals, -1.0, 1.0)
                feats[:, col + 1] = lag_norm
                feats[:, col + 2] = mean_abs

                # ── Spectral coherence ────────────────────────────────────
                psd_i     = np.abs(X_fft_coh[:, :, i]) ** 2   # (Nb, F_coh)
                psd_j     = np.abs(X_fft_coh[:, :, j]) ** 2
                cross_psd = (X_fft_coh[:, :, i]
                             * np.conj(X_fft_coh[:, :, j]))    # (Nb, F_coh)

                for b_idx, (f_lo, f_hi) in enumerate(self._COH_BANDS):
                    mask = (freqs_coh >= f_lo) & (freqs_coh < f_hi)
                    if not mask.any():
                        continue
                    csd = cross_psd[:, mask]
                    pii = psd_i[:, mask]
                    pjj = psd_j[:, mask]
                    coh = np.abs(csd) ** 2 / (pii * pjj + 1e-12)  # (Nb, F_band)
                    feats[:, col + 3 + b_idx] = coh.mean(axis=1)

                # ── Mutual information ─────────────────────────────────────
                xi = X[:, :, i]  # (Nb, T) — raw (unnormalised)
                xj = X[:, :, j]
                feats[:, col + 3 + n_coh] = self._batch_mutual_information(xi, xj)

                pair_idx += 1

        return feats

    def _batch_mutual_information(
        self, xi: np.ndarray, xj: np.ndarray
    ) -> np.ndarray:
        """
        xi, xj: (N, T)
        Returns: (N,) histogram-based mutual information

        Uses global min/max across all N windows for consistent binning.
        """
        N, T = xi.shape
        bins = self.mi_bins
        eps  = 1e-12

        # Global normalisation to [0, bins-1]
        xi_min, xi_max = xi.min(), xi.max()
        xj_min, xj_max = xj.min(), xj.max()

        mi_vals = np.zeros(N, dtype=np.float64)

        if xi_max <= xi_min or xj_max <= xj_min:
            return mi_vals

        xi_b = np.clip(
            ((xi - xi_min) / (xi_max - xi_min + eps) * bins).astype(int),
            0, bins - 1,
        )  # (N, T)
        xj_b = np.clip(
            ((xj - xj_min) / (xj_max - xj_min + eps) * bins).astype(int),
            0, bins - 1,
        )  # (N, T)

        for k in range(N):
            c_xy = np.zeros((bins, bins), dtype=np.float64)
            np.add.at(c_xy, (xi_b[k], xj_b[k]), 1.0)

            total = c_xy.sum() + eps
            px    = c_xy.sum(axis=1) / total   # (bins,)
            py    = c_xy.sum(axis=0) / total   # (bins,)
            pxy   = c_xy / total               # (bins, bins)

            px_outer = px[:, np.newaxis] * py[np.newaxis, :]
            mask     = pxy > 0
            mi_vals[k] = float(
                (pxy[mask] * np.log(pxy[mask] / (px_outer[mask] + eps))).sum()
            )

        return mi_vals


# ===========================================================================
#  Combined extractor  =  PowerfulFeatureExtractor  +  NonlinearEMGExtractor
# ===========================================================================

class CombinedNonlinearExtractor:
    """
    Concatenates PowerfulFeatureExtractor (with use_entropy=False to avoid
    placeholder-zero entropy features) and NonlinearEMGExtractor.

    Input:  X  (N, T, C)
    Output:     (N, F_powerful + F_nonlinear)
    """

    def __init__(self, sampling_rate: int = 2000, **nonlinear_kwargs):
        from processing.powerful_features import PowerfulFeatureExtractor

        self.powerful = PowerfulFeatureExtractor(
            sampling_rate=sampling_rate,
            use_entropy=False,   # we supply real entropy via NonlinearEMGExtractor
        )
        self.nonlinear = NonlinearEMGExtractor(
            sampling_rate=sampling_rate,
            **nonlinear_kwargs,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C)
        Returns: (N, F) float32
        """
        f_pow = self.powerful.transform(X)    # (N, F1)
        f_nl  = self.nonlinear.transform(X)  # (N, F2)
        return np.concatenate([f_pow, f_nl], axis=1).astype(np.float32)


# ===========================================================================
#  Custom trainer that adds LightGBM support
# ===========================================================================

class NonlinearMLTrainer:
    """
    Thin wrapper around FeatureMLTrainer that adds 'lgbm' as a valid
    ml_model_type (falls back gracefully if LightGBM is not installed).
    """

    def __new__(cls, **kwargs):
        # Import here so we can subclass dynamically
        from training.trainer import FeatureMLTrainer

        class _NonlinearMLTrainer(FeatureMLTrainer):
            def _create_ml_model(self, model_type: str):
                if model_type == "lgbm":
                    try:
                        import lightgbm as lgb
                        return lgb.LGBMClassifier(
                            n_estimators=300,
                            learning_rate=0.05,
                            num_leaves=63,
                            min_child_samples=20,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=-1,
                            random_state=self.cfg.seed,
                            verbose=-1,
                        )
                    except ImportError:
                        import warnings
                        warnings.warn(
                            "LightGBM not installed; falling back to SVM-RBF. "
                            "Install with: pip install lightgbm"
                        )
                        return super()._create_ml_model("svm_rbf")
                return super()._create_ml_model(model_type)

        return _NonlinearMLTrainer(**kwargs)


# ===========================================================================
#  Utilities
# ===========================================================================

def make_json_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ===========================================================================
#  LOSO fold runner
# ===========================================================================

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    proc_cfg,
    split_cfg,
    train_cfg,
    feature_extractor: CombinedNonlinearExtractor,
) -> Dict:
    """Run one LOSO fold with the combined nonlinear feature extractor."""
    import torch
    from config.cross_subject import CrossSubjectConfig
    from data.multi_subject_loader import MultiSubjectLoader
    from evaluation.cross_subject import CrossSubjectExperiment
    from visualization.base import Visualizer
    from visualization.cross_subject import CrossSubjectVisualizer
    from utils.logging import setup_logging, seed_everything
    from utils.artifacts import ArtifactSaver

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type  = "ml_emg_td"
    train_cfg.model_type     = model_type
    train_cfg.ml_model_type  = model_type

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects,
        test_subject=test_subject,
        exercises=exercises,
        base_dir=base_dir,
        pool_train_subjects=True,
        use_separate_val_subject=False,
        val_subject=None,
        val_ratio=0.15,
        seed=train_cfg.seed,
        max_gestures=10,
    )
    cs_cfg.save(output_dir / "cross_subject_config.json")

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=False,           # CPU-only: nonlinear features computed in numpy
        use_improved_processing=True,
    )

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    trainer = NonlinearMLTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )
    # Inject the combined feature extractor so FeatureMLTrainer skips its own
    trainer.feature_extractor = feature_extractor

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=base_viz,
        logger=logger,
    )

    try:
        results = experiment.run()
    except Exception as e:
        print(f"Error in LOSO fold (test={test_subject}, model={model_type}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1  = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test subject {test_subject} | Model: {model_type} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    results_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_save), f, indent=4,
                  ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "model_type":     model_type,
            "approach":       "nonlinear_stats_combined",
            "exercises":      exercises,
            "metrics": {
                "test_accuracy":  test_acc,
                "test_f1_macro":  test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type":   model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ===========================================================================
#  Main
# ===========================================================================

def main():
    EXPERIMENT_NAME = "exp_38_nonlinear_stats_svm_lgbm_loso"
    BASE_DIR    = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()

    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES  = ["E1"]

    # Models to evaluate: svm_rbf, svm_linear, lgbm (if installed)
    MODEL_TYPES = ["svm_rbf", "svm_linear", "lgbm"]

    # Check LightGBM availability early
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print("[exp_38] LightGBM not installed — 'lgbm' will fall back to SVM-RBF.")

    from config.base import ProcessingConfig, SplitConfig, TrainingConfig
    from utils.logging import setup_logging

    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )

    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    train_cfg = TrainingConfig(
        batch_size=4096,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.0,
        early_stopping_patience=1,
        use_class_weights=False,
        seed=42,
        num_workers=0,
        device="cpu",
        use_handcrafted_features=False,
        handcrafted_feature_set="emg_td",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_rbf",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    # Build combined feature extractor once (stateless, reused across folds)
    feature_extractor = CombinedNonlinearExtractor(
        sampling_rate=proc_cfg.sampling_rate,
        entropy_downsample=4,
        perm_m=4,
        higuchi_kmax=8,
        mi_bins=10,
        n_jobs=-1,            # parallelise sample entropy + Lyapunov loops
        cross_channel_batch=1000,
    )

    print(f"[{EXPERIMENT_NAME}] Subjects  : {ALL_SUBJECTS}")
    print(f"[{EXPERIMENT_NAME}] Models    : {MODEL_TYPES}")
    print(f"[{EXPERIMENT_NAME}] Exercises : {EXERCISES}")
    print(f"[{EXPERIMENT_NAME}] Features  : PowerfulFeatureExtractor (no-entropy)")
    print(f"                          + NonlinearEMGExtractor")
    print(f"                            per-channel × 7  +  pairs × 8  = 280 nonlinear")

    all_loso_results: List[Dict] = []

    for model_type in MODEL_TYPES:
        print(f"\n{'=' * 65}")
        print(f"  Model: {model_type}  —  starting LOSO")
        print(f"{'=' * 65}")

        for test_subject in ALL_SUBJECTS:
            train_subjects  = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"

            train_cfg.ml_model_type = model_type

            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                model_type=model_type,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                feature_extractor=feature_extractor,
            )
            all_loso_results.append(fold_res)

    # ---- Aggregate results ----
    aggregate_results: Dict = {}
    for model_type in MODEL_TYPES:
        model_results = [
            r for r in all_loso_results
            if r["model_type"] == model_type and r.get("test_accuracy") is not None
        ]
        if not model_results:
            continue
        accs = [r["test_accuracy"] for r in model_results]
        f1s  = [r["test_f1_macro"] for r in model_results]
        aggregate_results[model_type] = {
            "mean_accuracy":  float(np.mean(accs)),
            "std_accuracy":   float(np.std(accs)),
            "mean_f1_macro":  float(np.mean(f1s)),
            "std_f1_macro":   float(np.std(f1s)),
            "num_subjects":   len(accs),
            "per_subject":    model_results,
        }

    # ---- Print summary ----
    print(f"\n{'=' * 65}")
    print(f"SUMMARY: {EXPERIMENT_NAME}")
    print(f"{'=' * 65}")
    for mt, res in aggregate_results.items():
        acc_m, acc_s = res["mean_accuracy"], res["std_accuracy"]
        f1_m,  f1_s  = res["mean_f1_macro"],  res["std_f1_macro"]
        print(f"  {mt:12s}: Acc={acc_m:.4f} ± {acc_s:.4f},"
              f"  F1={f1_m:.4f} ± {f1_s:.4f}")

    # ---- Save summary JSON ----
    loso_summary = {
        "experiment_name":   EXPERIMENT_NAME,
        "hypothesis":        (
            "Nonlinear statistics (sample entropy, permutation entropy, Higuchi FD, "
            "Hjorth, Lyapunov) and channel-pair features (cross-corr, coherence, MI) "
            "are more subject-invariant than classical RMS/MAV/PSD features."
        ),
        "approach":          "CombinedNonlinearExtractor + SVM / LightGBM",
        "feature_extractor": "CombinedNonlinearExtractor "
                             "(PowerfulFeatureExtractor + NonlinearEMGExtractor)",
        "nonlinear_features_per_channel": 7,
        "cross_channel_features_per_pair": 8,
        "models":            MODEL_TYPES,
        "subjects":          ALL_SUBJECTS,
        "exercises":         EXERCISES,
        "processing_config": asdict(proc_cfg),
        "split_config":      asdict(split_cfg),
        "training_config":   asdict(train_cfg),
        "aggregate_results": aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date":   datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(loso_summary), f,
                  indent=4, ensure_ascii=False)
    print(f"\n[DONE] {EXPERIMENT_NAME} → {summary_path}")

    # ---- Notify hypothesis executor if available ----
    try:
        from hypothesis_executor import mark_hypothesis_verified
        best_model = max(
            aggregate_results,
            key=lambda m: aggregate_results[m]["mean_f1_macro"],
            default=None,
        )
        if best_model is not None:
            best_res = aggregate_results[best_model]
            mark_hypothesis_verified(
                hypothesis_id="H38",
                metrics={
                    "best_model":        best_model,
                    "mean_accuracy":     best_res["mean_accuracy"],
                    "mean_f1_macro":     best_res["mean_f1_macro"],
                    "std_accuracy":      best_res["std_accuracy"],
                    "aggregate_results": aggregate_results,
                },
                experiment_name=EXPERIMENT_NAME,
            )
    except ImportError:
        pass
    except Exception as _he_err:
        print(f"[exp_38] hypothesis_executor notification failed: {_he_err}")


if __name__ == "__main__":
    main()
