"""
Experiment 67: Cyclostationary / Spectral Correlation Features for Cross-Subject EMG (LOSO)

Hypothesis H67:
    Gestures generate quasi-periodic motor unit (MU) firing patterns.
    Cyclostationary features — normalized ACF, spectral cyclic coherence,
    and envelope periodicity — capture this temporal structure while being
    invariant to amplitude distortions (electrode-skin impedance variation),
    analogous to how cyclostationary signal processing in communications
    achieves channel-invariant detection.

    Combining these features with PowerfulFeatureExtractor and compressing
    via PCA (fit on train only, strict LOSO protocol) should improve
    cross-subject generalization compared to amplitude-sensitive baselines.

Theoretical background:
    A cyclostationary signal x(t) has second-order statistics (autocorrelation)
    that are periodic in time with some period T_0. The Spectral Correlation
    Density (SCD) is defined as:

        S^α(f) = lim_{T→∞} (1/T) E[ X_T(f + α/2) · X_T*(f - α/2) ]

    where α is the "cycle frequency" and f is the spectral frequency.
    When α=0, SCD reduces to the ordinary PSD.
    Cyclostationary features (SCD, cyclic autocorrelation) are:
      - Non-zero only when the signal has periodic structure at rate α
      - Amplitude-invariant if properly normalized
    For EMG: MU firing at rates 5–100 Hz → signal is cyclostationary with
    those cycle frequencies. Different gestures → different MU recruitment.

Feature groups (per channel, all amplitude-invariant):
    1. Normalized ACF at K lags — ρ(τ) = R(τ)/R(0) ∈ [-1, 1]
       Captures temporal correlation structure without amplitude sensitivity.
       Uses FFT via the Wiener-Khinchin theorem.
       K = 20 lags covering 2ms – 300ms at 2000 Hz.

    2. Spectral cyclic coherence at n_alpha cycle frequencies × n_bands:
       γ²(α, f) = |E_m[X_m(f+α/2) · X_m*(f-α/2)]|²
                  / (E_m[|X_m(f+α/2)|²] · E_m[|X_m(f-α/2)|²])
       Estimated by dividing each window into M overlapping sub-blocks (Welch
       approach for SCD). γ² ∈ [0, 1] and is amplitude-invariant.
       Averaged over 4 EMG spectral bands [20–80, 80–150, 150–300, 300–500 Hz].
       Cycle frequencies: [20, 30, 40, 60, 80] Hz (resolvable with 200-sample
       sub-blocks at 2000 Hz → 10 Hz/bin, min α = 20 Hz at δ=1 bin).
       Features: 5 α × 4 bands = 20 per channel.

    3. Normalized envelope periodicity at cycle frequencies:
       |FFT{A(t)}[α]| / Σ_f |FFT{A(t)}[f]|
       A(t) = |x(t) + j·H{x(t)}|  (Hilbert envelope via FFT)
       Captures MU firing rate periodicity in the signal envelope.
       Amplitude-invariant (normalized by total spectral mass of envelope).
       Features: 5 per channel.

Per channel total:  20 (ACF) + 20 (SCD coherence) + 5 (envelope) = 45
For C=8 channels:   360 cyclostationary features
Combined total:     ~253 (PowerfulFeatureExtractor, no entropy) + 360 ≈ 613

LOSO protocol (strict, no leakage):
    - Feature extraction: stateless, all normalization within each window
    - z-score standardization: fit on pooled train-subject windows only
    - PCA: fit on train set only (ml_use_pca=True in TrainingConfig)
    - Test subject is NEVER seen during any fitting step

Usage:
    python experiments/exp_67_cyclostationary_spectral_correlation_loso.py
    python experiments/exp_67_cyclostationary_spectral_correlation_loso.py --ci
    python experiments/exp_67_cyclostationary_spectral_correlation_loso.py --full
    python experiments/exp_67_cyclostationary_spectral_correlation_loso.py \\
        --subjects DB2_s1,DB2_s12
"""

import sys
import json
import math
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np

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
#  Cyclostationary EMG Feature Extractor
# ===========================================================================

class CyclostationaryEMGExtractor:
    """
    Cyclostationary / spectral correlation features for cross-subject EMG.

    All three feature groups are amplitude-invariant: each normalization uses
    only values from within the same window, so no cross-window or cross-subject
    statistics are required. This means the extractor is fully stateless and
    introduces no data leakage regardless of how windows are partitioned.

    Input:  X  (N, T, C)   — EMG windows, float32 or float64
    Output:     (N, C × (K_acf + n_alpha × n_bands + n_alpha))  float32
    """

    # Spectral bands for cyclic coherence, matched to EMG frequency content (Hz)
    _SCD_SPECTRAL_BANDS: List[Tuple[float, float]] = [
        (20, 80),
        (80, 150),
        (150, 300),
        (300, 500),
    ]

    def __init__(
        self,
        sampling_rate: int = 2000,
        acf_lags: Optional[List[int]] = None,
        cycle_freqs_hz: Optional[List[float]] = None,
        scd_block_len: int = 200,
        scd_hop: int = 100,
    ):
        """
        Args:
            sampling_rate:  EMG sampling rate in Hz.
            acf_lags:       Lag indices (in samples) for normalized ACF features.
                            Default: 20 lags covering 2 ms – 300 ms.
            cycle_freqs_hz: Cycle frequencies (Hz) for SCD coherence and
                            envelope periodicity. Must satisfy
                            α ≥ 2 × fs / scd_block_len to ensure bin shift δ ≥ 1.
                            Default: [20, 30, 40, 60, 80] Hz.
            scd_block_len:  Sub-block length (samples) for Welch-SCD estimation.
                            Trades frequency resolution vs. stationarity within
                            the block. At fs=2000, len=200 → 10 Hz/bin, so the
                            minimum resolvable cycle frequency is 20 Hz (δ=1 bin).
            scd_hop:        Hop between consecutive sub-blocks (samples).
                            Must be ≤ scd_block_len.
        """
        self.fs = sampling_rate

        if acf_lags is None:
            # 20 lags: 2 ms (4 samp) to 300 ms (600 samp) at 2000 Hz
            self.acf_lags: List[int] = [
                4, 8, 12, 16, 20, 25, 30, 40, 50, 60,
                80, 100, 120, 150, 200, 250, 300, 400, 500, 599,
            ]
        else:
            self.acf_lags = sorted(int(l) for l in acf_lags)

        if cycle_freqs_hz is None:
            # At scd_block_len=200, fs=2000: min resolvable = 2×fs/L = 20 Hz
            # δ for each α = round(α × L / fs / 2):
            #   20 Hz → δ=1, 30 Hz → δ=2 (banker's round), 40 Hz → δ=2,
            #   60 Hz → δ=3, 80 Hz → δ=4  — all ≥ 1 ✓
            self.cycle_freqs_hz: List[float] = [20.0, 30.0, 40.0, 60.0, 80.0]
        else:
            self.cycle_freqs_hz = list(cycle_freqs_hz)

        assert scd_hop <= scd_block_len, "scd_hop must be ≤ scd_block_len"
        self.scd_block_len = scd_block_len
        self.scd_hop = scd_hop
        self.scd_bands = self._SCD_SPECTRAL_BANDS

        # Validate: warn if cycle frequency cannot produce a non-zero bin shift.
        # The bin shift δ = round(α × L / fs / 2) must be ≥ 1.
        # This requires α ≥ 2 × fs / L (= twice the spectral resolution per bin).
        min_resolvable = 2.0 * self.fs / self.scd_block_len
        for alpha in self.cycle_freqs_hz:
            delta_check = int(round(alpha * self.scd_block_len / self.fs / 2.0))
            if delta_check == 0:
                warnings.warn(
                    f"[CyclostationaryEMGExtractor] Cycle frequency {alpha} Hz "
                    f"gives bin shift δ=0 (min resolvable ≈ {min_resolvable:.1f} Hz "
                    f"= 2×fs/scd_block_len). This cycle frequency will be skipped.",
                    stacklevel=2,
                )

    @property
    def n_features_per_channel(self) -> int:
        return (
            len(self.acf_lags)
            + len(self.cycle_freqs_hz) * len(self.scd_bands)
            + len(self.cycle_freqs_hz)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (N, T, C) EMG windows
        Returns:
            features: (N, C × (K_acf + n_alpha × n_bands + n_alpha))  float32
        """
        if X.ndim != 3:
            raise ValueError(f"CyclostationaryEMGExtractor.transform: "
                             f"expected (N, T, C), got {X.shape}")
        N, T, C = X.shape
        n_feat = self.n_features_per_channel

        out = np.zeros((N, C * n_feat), dtype=np.float64)

        n_acf    = len(self.acf_lags)
        n_alpha  = len(self.cycle_freqs_hz)
        n_bands  = len(self.scd_bands)

        for c in range(C):
            sig = X[:, :, c]  # (N, T)

            acf_f = self._batch_normalized_acf(sig)        # (N, n_acf)
            scd_f = self._batch_scd_coherence(sig)         # (N, n_alpha * n_bands)
            env_f = self._batch_envelope_periodicity(sig)  # (N, n_alpha)

            col_start = c * n_feat
            col_mid1  = col_start + n_acf
            col_mid2  = col_mid1  + n_alpha * n_bands
            col_end   = col_mid2  + n_alpha

            out[:, col_start:col_mid1] = acf_f
            out[:, col_mid1 :col_mid2] = scd_f
            out[:, col_mid2 :col_end]  = env_f

        return np.nan_to_num(
            out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

    # ------------------------------------------------------------------
    # Group 1: Normalized autocorrelation function (ACF)
    # ------------------------------------------------------------------

    def _batch_normalized_acf(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalized ACF ρ(τ) = R(τ) / R(0) at self.acf_lags.

        ρ(τ) ∈ [-1, 1] — amplitude-invariant (variance R(0) in denominator).
        R(τ) computed via FFT using the Wiener-Khinchin theorem:
            PSD(f) = |FFT(x)|²  →  ACF(τ) = IFFT(PSD)(τ)

        X: (N, T)
        Returns: (N, K)  — normalized ACF at K lags
        """
        N, T = X.shape
        K = len(self.acf_lags)

        # Demean: converts autocorrelation → autocovariance (removes DC bias)
        Xc = X - X.mean(axis=1, keepdims=True)  # (N, T)

        # Zero-pad to 2T to avoid circular aliasing in the linear ACF
        nfft = 2 * T
        Xfft = np.fft.rfft(Xc, n=nfft, axis=1)          # (N, nfft//2+1)
        psd  = np.abs(Xfft) ** 2                          # (N, nfft//2+1)
        acf  = np.fft.irfft(psd, n=nfft, axis=1)[:, :T]  # (N, T)
        # acf[:, τ] = sum_t Xc[t] * Xc[t+τ]  (unnormalized)

        R0 = acf[:, 0]  # (N,)  lag-0 = N × variance × (something scaling)

        out = np.zeros((N, K), dtype=np.float64)
        for k, lag in enumerate(self.acf_lags):
            if 0 <= lag < T:
                # ρ(τ) = R(τ) / R(0), guaranteed in [-1, 1] by Cauchy-Schwarz
                out[:, k] = np.clip(
                    acf[:, lag] / (R0 + 1e-12), -1.0, 1.0
                )
        return out

    # ------------------------------------------------------------------
    # Group 2: Spectral cyclic coherence (Welch-SCD estimator)
    # ------------------------------------------------------------------

    def _batch_scd_coherence(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate spectral cyclic coherence using overlapping sub-blocks.

        For cycle frequency α and center spectral frequency f:

            γ²(α, f) = |E_m[ X_m(f+α/2) · X_m*(f-α/2) ]|²
                       / (E_m[|X_m(f+α/2)|²] · E_m[|X_m(f-α/2)|²])

        where E_m averages over M sub-blocks extracted from the window.
        γ²(α, f) ∈ [0, 1] and is amplitude-invariant (scaling x cancels).

        Physical interpretation:
          - When α=0: reduces to 1 (trivial).
          - α≠0: measures degree of second-order periodicity at cycle rate α.
          - γ²≈1 means x(t) is nearly cyclostationary at that (α, f) pair.
          - For EMG: MU firing at rate α creates peaks at that cycle frequency.

        Sub-block length L and hop h trade off:
          - Spectral resolution of f: fs/L Hz/bin
          - Number of sub-blocks M ≈ (T - L) / h + 1 (more → better coherence est.)
          - Min resolvable α: fs/L (need k_shift ≥ 1 bin)

        X: (N, T)
        Returns: (N, n_alpha * n_bands)  — band-averaged cyclic coherence
        """
        N, T = X.shape
        L = self.scd_block_len
        h = self.scd_hop
        n_alpha = len(self.cycle_freqs_hz)
        n_bands = len(self.scd_bands)

        # Sub-block start indices
        starts = list(range(0, T - L + 1, h))
        M = len(starts)

        if M < 2:
            # Window too short to form multiple sub-blocks → return zeros
            return np.zeros((N, n_alpha * n_bands), dtype=np.float64)

        # Build sub-blocks with Hann window to reduce spectral leakage
        hann = np.hanning(L).astype(np.float64)          # (L,)
        X_blocks = np.empty((N, M, L), dtype=np.float64)
        for m, s in enumerate(starts):
            X_blocks[:, m, :] = X[:, s:s + L] * hann[np.newaxis, :]

        # FFT of each sub-block
        X_fft = np.fft.rfft(X_blocks, axis=2)            # (N, M, nf)
        nf = X_fft.shape[2]
        freqs = np.fft.rfftfreq(L, d=1.0 / self.fs)      # (nf,)

        out = np.zeros((N, n_alpha * n_bands), dtype=np.float64)

        for ai, alpha_hz in enumerate(self.cycle_freqs_hz):
            # Bin shift δ = round(α × L / fs / 2)
            #   The SCD at cycle freq α between spectral bins k-δ and k+δ
            #   requires a shift of δ bins on each side.
            delta = int(round(alpha_hz * L / self.fs / 2.0))

            if delta == 0:
                # Cycle frequency too low to resolve → skip, leave zeros
                continue

            n_valid = nf - 2 * delta
            if n_valid <= 0:
                continue

            # Vectorized SCD cross-spectrum for all valid center bins k.
            # Center bin k runs from δ to nf-1-δ (n_valid values).
            # For k' = k - δ ∈ [0, n_valid-1]:
            #   upper freq = k + δ = k' + 2δ  →  X_fft[:, :, k'+2δ : nf]
            #   lower freq = k - δ = k'        →  X_fft[:, :, 0 : nf-2δ]
            Xp       = X_fft[:, :, 2*delta:]         # X(f + α/2): (N, M, n_valid)
            Xm_conj  = X_fft[:, :, :n_valid].conj()  # X*(f - α/2): (N, M, n_valid)

            # Cross-spectrum per sub-block
            cross = Xp * Xm_conj                      # (N, M, n_valid)

            # Average over M sub-blocks (expectation approximation)
            cross_mean = cross.mean(axis=1)           # (N, n_valid)

            # Numerator: |E[cross]|²
            num = np.abs(cross_mean) ** 2             # (N, n_valid)

            # Denominator: E[|Xp|²] × E[|Xm|²]
            psd_upper = (np.abs(Xp) ** 2).mean(axis=1)      # (N, n_valid)
            psd_lower = (np.abs(Xm_conj) ** 2).mean(axis=1) # (N, n_valid)
            denom = psd_upper * psd_lower                    # (N, n_valid)

            # Cyclic coherence γ² ∈ [0, 1]
            coh = np.clip(num / (denom + 1e-24), 0.0, 1.0)  # (N, n_valid)

            # Center frequency axis for the n_valid valid bins:
            # k' + δ → freqs[k'+ δ], so freqs[δ : nf-δ]
            freqs_center = freqs[delta : nf - delta]         # (n_valid,)

            # Band-average coherence
            for bi, (f_lo, f_hi) in enumerate(self.scd_bands):
                mask = (freqs_center >= f_lo) & (freqs_center < f_hi)
                if not mask.any():
                    continue
                col = ai * n_bands + bi
                out[:, col] = coh[:, mask].mean(axis=1)

        return out

    # ------------------------------------------------------------------
    # Group 3: Normalized envelope periodicity
    # ------------------------------------------------------------------

    def _batch_envelope_periodicity(self, X: np.ndarray) -> np.ndarray:
        """
        Normalized envelope periodicity at cycle frequencies.

        A(t) = |x(t) + j·H{x(t)}|  — analytic envelope via FFT-Hilbert.
        Feature: |FFT{A}[α]| / Σ_f |FFT{A}[f]|  — fraction of envelope
        spectral energy at each cycle frequency.

        Amplitude-invariant: scaling x by c scales A by c (envelope is
        homogeneous of degree 1), but the ratio |FFT{A}[α]| / Σ|FFT{A}[f]|
        cancels the constant c.

        Physical basis: if a MU fires at rate α, the EMG envelope will have
        a periodic modulation at that frequency, creating a peak in |FFT{A}|
        at α Hz. Different gestures → different dominant MU firing rates.

        X: (N, T)
        Returns: (N, n_alpha)
        """
        N, T = X.shape
        n_alpha = len(self.cycle_freqs_hz)

        # Analytic signal via FFT-based Hilbert transform
        # h: one-sided spectral weighting for Hilbert
        Xfft = np.fft.fft(X, axis=1)   # (N, T) — full DFT
        h = np.zeros(T, dtype=np.float64)
        h[0] = 1.0
        if T % 2 == 0:
            h[1:T // 2]  = 2.0
            h[T // 2]    = 1.0
        else:
            h[1:(T + 1) // 2] = 2.0
        # analytic_fft = Xfft * h: doubles positive freqs, zeros negative freqs
        analytic = np.fft.ifft(Xfft * h[np.newaxis, :], axis=1)  # (N, T) complex
        envelope = np.abs(analytic)                                 # (N, T) float

        # FFT of the envelope signal (real → use rfft)
        env_fft   = np.fft.rfft(envelope, axis=1)         # (N, T//2+1)
        env_amp   = np.abs(env_fft)                        # (N, T//2+1)
        freqs_env = np.fft.rfftfreq(T, d=1.0 / self.fs)   # (T//2+1,)

        # Total L1 spectral mass for normalization (amplitude-invariant ratio)
        total_mass = env_amp.sum(axis=1)                   # (N,)

        out = np.zeros((N, n_alpha), dtype=np.float64)
        for k, alpha_hz in enumerate(self.cycle_freqs_hz):
            bin_idx = int(np.argmin(np.abs(freqs_env - alpha_hz)))
            out[:, k] = env_amp[:, bin_idx] / (total_mass + 1e-12)

        return out


# ===========================================================================
#  Combined extractor = PowerfulFeatureExtractor + CyclostationaryEMGExtractor
# ===========================================================================

class CombinedCyclostationaryExtractor:
    """
    Concatenates:
      - PowerfulFeatureExtractor (use_entropy=False to avoid placeholder zeros;
        real entropy is not required here since cyclostationary features already
        capture nonlinear temporal structure)
      - CyclostationaryEMGExtractor

    Input:  X  (N, T, C)
    Output:     (N, F_powerful + F_cyclo)  float32
    """

    def __init__(self, sampling_rate: int = 2000, **cyclo_kwargs):
        from processing.powerful_features import PowerfulFeatureExtractor

        self.powerful = PowerfulFeatureExtractor(
            sampling_rate=sampling_rate,
            use_entropy=False,   # entropy handled by cyclostationary features
        )
        self.cyclo = CyclostationaryEMGExtractor(
            sampling_rate=sampling_rate,
            **cyclo_kwargs,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C)
        Returns: (N, F_powerful + F_cyclo) float32
        """
        f_pow = self.powerful.transform(X)   # (N, F1)
        f_cyc = self.cyclo.transform(X)      # (N, F2)
        return np.concatenate([f_pow, f_cyc], axis=1).astype(np.float32)


# ===========================================================================
#  LightGBM-aware trainer (same pattern as exp_38)
# ===========================================================================

class CyclostationaryMLTrainer:
    """
    Thin wrapper around FeatureMLTrainer that adds 'lgbm' as a valid
    ml_model_type (falls back gracefully if LightGBM is not installed).
    """

    def __new__(cls, **kwargs):
        from training.trainer import FeatureMLTrainer

        class _CyclostationaryMLTrainer(FeatureMLTrainer):
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
                        warnings.warn(
                            "LightGBM not installed; falling back to SVM-RBF. "
                            "Install with: pip install lightgbm"
                        )
                        return super()._create_ml_model("svm_rbf")
                return super()._create_ml_model(model_type)

        return _CyclostationaryMLTrainer(**kwargs)


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
    feature_extractor: CombinedCyclostationaryExtractor,
) -> Dict:
    """
    Run one LOSO fold.

    LOSO invariance guarantees:
      - feature_extractor.transform() is stateless: all normalization is
        within each window (no cross-window statistics).
      - z-score (feature_mean, feature_std) is computed from X_train only.
      - PCA is fit on X_train only (ml_use_pca=True triggers this in
        FeatureMLTrainer.fit() from trainer.py).
      - test_subject data is NEVER passed to any fitting step.
    """
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

    train_cfg.pipeline_type = "ml_emg_td"
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
        use_gpu=False,          # cyclostationary features computed in numpy
        use_improved_processing=True,
    )

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    trainer = CyclostationaryMLTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )
    # Inject the combined feature extractor so FeatureMLTrainer skips its
    # own creation and uses this one directly. The PCA/standardization are
    # still fit on X_train only inside FeatureMLTrainer.fit().
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
        print(f"[exp_67] Error in LOSO fold "
              f"(test={test_subject}, model={model_type}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type":   model_type,
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
        json.dump(make_json_serializable(results_save), f,
                  indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "model_type":     model_type,
            "approach":       "cyclostationary_spectral_correlation",
            "exercises":      exercises,
            "cyclo_n_features_per_channel": (
                feature_extractor.cyclo.n_features_per_channel
            ),
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
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
    EXPERIMENT_NAME = "exp_67_cyclostationary_spectral_correlation_loso"
    BASE_DIR     = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()

    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES  = ["E1"]

    # Models to evaluate
    MODEL_TYPES = ["svm_rbf", "svm_linear", "lgbm"]

    # Check LightGBM availability early
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print("[exp_67] LightGBM not installed — 'lgbm' will fall back to SVM-RBF.")

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
        # PCA is fit on X_train only inside FeatureMLTrainer.fit() —
        # this is the key LOSO guard against leakage from SCD features.
        ml_use_pca=True,
        ml_pca_var_ratio=0.95,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    # Build combined feature extractor once (stateless across folds).
    # Sub-block length 200 → 10 Hz/bin at fs=2000.
    # Bin shift δ = round(α × 200 / 2000 / 2) ≥ 1 for α ≥ 20 Hz.
    # With T=600 and hop=100: M = (600-200)//100 + 1 = 5 sub-blocks per window.
    feature_extractor = CombinedCyclostationaryExtractor(
        sampling_rate=proc_cfg.sampling_rate,
        # ACF: 20 lags from 2ms to 300ms at 2000 Hz
        acf_lags=[4, 8, 12, 16, 20, 25, 30, 40, 50, 60,
                  80, 100, 120, 150, 200, 250, 300, 400, 500, 599],
        # Cycle frequencies: 20–80 Hz (MU firing range, min ≥ 20 = fs/100)
        cycle_freqs_hz=[20.0, 30.0, 40.0, 60.0, 80.0],
        scd_block_len=200,   # 100ms at 2000 Hz → 10 Hz/bin; min α = 20 Hz (δ=1)
        scd_hop=100,         # 50% overlap → 5 sub-blocks per 600-sample window
    )

    n_cyclo = feature_extractor.cyclo.n_features_per_channel * proc_cfg.num_channels
    print(f"[{EXPERIMENT_NAME}]")
    print(f"  Subjects    : {ALL_SUBJECTS}")
    print(f"  Models      : {MODEL_TYPES}")
    print(f"  Exercises   : {EXERCISES}")
    print(f"  CyclostationaryEMGExtractor:")
    print(f"    ACF lags  : {len(feature_extractor.cyclo.acf_lags)} lags × "
          f"{proc_cfg.num_channels} ch = "
          f"{len(feature_extractor.cyclo.acf_lags) * proc_cfg.num_channels} feats")
    print(f"    SCD bands : {len(feature_extractor.cyclo.cycle_freqs_hz)} α × "
          f"{len(feature_extractor.cyclo.scd_bands)} bands × "
          f"{proc_cfg.num_channels} ch = "
          f"{len(feature_extractor.cyclo.cycle_freqs_hz) * len(feature_extractor.cyclo.scd_bands) * proc_cfg.num_channels} feats")
    print(f"    Envelope  : {len(feature_extractor.cyclo.cycle_freqs_hz)} α × "
          f"{proc_cfg.num_channels} ch = "
          f"{len(feature_extractor.cyclo.cycle_freqs_hz) * proc_cfg.num_channels} feats")
    print(f"  Total cyclo features: {n_cyclo}")
    print(f"  PCA: enabled (var_ratio=0.95, fit on train only)")

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
            if r["model_type"] == model_type
            and r.get("test_accuracy") is not None
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
        acc_m = res["mean_accuracy"]
        acc_s = res["std_accuracy"]
        f1_m  = res["mean_f1_macro"]
        f1_s  = res["std_f1_macro"]
        print(
            f"  {mt:12s}: "
            f"Acc={acc_m:.4f} ± {acc_s:.4f},  "
            f"F1={f1_m:.4f} ± {f1_s:.4f}"
        )

    # ---- Save summary JSON ----
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "H67: Cyclostationary features (normalized ACF, spectral cyclic "
            "coherence, envelope periodicity) are amplitude-invariant and capture "
            "gesture-specific MU firing structure, improving cross-subject "
            "generalization. Analogous to cyclostationary channel-invariance "
            "in communications."
        ),
        "approach": (
            "CombinedCyclostationaryExtractor (PowerfulFeatureExtractor + "
            "CyclostationaryEMGExtractor) → z-score → PCA (train only) → SVM/LightGBM"
        ),
        "feature_groups": {
            "acf_lags": feature_extractor.cyclo.acf_lags,
            "cycle_freqs_hz": feature_extractor.cyclo.cycle_freqs_hz,
            "scd_block_len": feature_extractor.cyclo.scd_block_len,
            "scd_hop": feature_extractor.cyclo.scd_hop,
            "scd_spectral_bands": feature_extractor.cyclo.scd_bands,
            "n_cyclo_features_total": n_cyclo,
        },
        "loso_leakage_prevention": {
            "feature_extraction": "stateless, within-window normalization only",
            "standardization": "fit on X_train (pooled train subjects) only",
            "pca": "fit on X_train only (ml_use_pca=True in FeatureMLTrainer)",
            "test_subject": "never seen in any fitting step",
        },
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
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        best_model = max(
            aggregate_results,
            key=lambda m: aggregate_results[m]["mean_f1_macro"],
            default=None,
        )
        if best_model is not None:
            best_res = aggregate_results[best_model]
            mark_hypothesis_verified(
                hypothesis_id="H67",
                metrics={
                    "best_model":    best_model,
                    "mean_accuracy": best_res["mean_accuracy"],
                    "mean_f1_macro": best_res["mean_f1_macro"],
                    "std_accuracy":  best_res["std_accuracy"],
                    "aggregate_results": aggregate_results,
                },
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                "H67",
                "No successful LOSO folds completed.",
            )
    except ImportError:
        pass
    except Exception as _he_err:
        print(f"[exp_67] hypothesis_executor notification failed: {_he_err}")


if __name__ == "__main__":
    main()
