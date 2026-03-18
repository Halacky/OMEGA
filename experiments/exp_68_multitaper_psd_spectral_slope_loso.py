"""
Experiment 68: Multitaper PSD + Spectral Slope / 1/f Parameters (LOSO)

Hypothesis H68:
    The shape of the EMG power spectrum — specifically its aperiodic 1/f exponent,
    spectral knee, and residual oscillatory peaks above the aperiodic background —
    is more subject-invariant than raw band power or amplitude features.

    Multitaper spectral estimation (DPSS tapers) gives a low-variance PSD estimate
    per window. A FOOOF-style simplified model (linear fit in log-log space) separates
    the aperiodic background from oscillatory peaks. The resulting parameters are more
    robust to subject-specific gain differences (which shift the offset uniformly) while
    still capturing gesture-specific spectral shapes.

Why multitaper vs plain FFT:
    DPSS (Slepian) tapers are the optimal set of orthogonal band-limited windows.
    Averaging K = 2·NW-1 tapered periodograms dramatically reduces spectral variance
    compared to a single periodogram, while the half-bandwidth NW/T·fs controls
    frequency leakage. For EMG windows of 600 samples at 2000 Hz and NW=4, we get
    K=7 tapers and ~13 Hz half-bandwidth — sufficient to resolve EMG spectral bands.

Feature groups per channel C (C=8 → 15 × 8 = 120 multitaper features):
    Aperiodic model (log-log linear fit in 20–450 Hz):
        1.  aperiodic_offset      — log₁₀ PSD level at reference freq (intercept)
        2.  aperiodic_exponent    — spectral slope (positive = falling spectrum)
    Spectral knee:
        3.  knee_freq             — Hz of maximum residual above aperiodic model
        4.  knee_residual         — height of that residual (oscillatory indicator)
    Overall oscillatory content:
        5.  residual_rms          — RMS of log-residual over 20–450 Hz
    Residual peaks per EMG band (3 bands × 2 feats = 6):
        6.  peak_freq_20_100      — Hz of strongest residual peak in 20–100 Hz
        7.  peak_amp_20_100       — amplitude above aperiodic in that band
        8.  peak_freq_100_300
        9.  peak_amp_100_300
        10. peak_freq_300_500
        11. peak_amp_300_500
    Band power ratios vs aperiodic model (3 bands):
        12. ratio_20_80           — mean log(actual/predicted) in 20–80 Hz
        13. ratio_80_200          — mean log(actual/predicted) in 80–200 Hz
        14. ratio_200_500         — mean log(actual/predicted) in 200–500 Hz
    Spectral rolloff:
        15. rolloff_90            — Hz below which 90% of cumulative power falls

Combined with PowerfulFeatureExtractor (use_entropy=False, ~253 features):
    Total ≈ 120 + 253 = ~373 features

LOSO purity:
    - All spectral parameters (NW, K, freq bands) are fixed constants — no data fitting.
    - Feature standardization uses only training-split statistics, handled by
      FeatureMLTrainer.fit() which receives only the LOSO training data.
    - No test-subject data is used in any step of feature parameterisation.

Comparison target: exp_38 (NonlinearEMGExtractor + SVM/LGBM)

Usage:
    python experiments/exp_68_multitaper_psd_spectral_slope_loso.py          # CI (default)
    python experiments/exp_68_multitaper_psd_spectral_slope_loso.py --ci     # same
    python experiments/exp_68_multitaper_psd_spectral_slope_loso.py --full   # 20 subjects
    python experiments/exp_68_multitaper_psd_spectral_slope_loso.py --subjects DB2_s1,DB2_s12
"""

import sys
import json
import argparse
import traceback
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
#  Multitaper PSD Feature Extractor
# ===========================================================================

class MultitaperPSDExtractor:
    """
    Extracts DPSS multitaper PSD features from EMG windows.

    For each channel the following steps are applied independently to each window
    (no cross-window or cross-subject statistics):
        1. Compute multitaper PSD via K DPSS tapers (purely signal-processing).
        2. Fit a log-log linear model (aperiodic / 1/f component).
        3. Compute the residual (actual log PSD minus aperiodic prediction).
        4. Extract peak, ratio, and rolloff features from the residual.

    LOSO purity: all parameters (NW, K, frequency bounds) are fixed constants.
    Feature standardisation is delegated to FeatureMLTrainer.fit().

    Input:  X  (N, T, C)
    Output:     (N, C × N_FEATS_PER_CHANNEL)  float32
    """

    # Fixed frequency analysis range
    FREQ_MIN_HZ: float = 20.0
    FREQ_MAX_HZ: float = 450.0

    # EMG peak-search bands (residual above aperiodic model)
    PEAK_BANDS: List[Tuple[float, float]] = [(20.0, 100.0), (100.0, 300.0), (300.0, 500.0)]

    # Band-power ratio bands (actual vs aperiodic prediction)
    RATIO_BANDS: List[Tuple[float, float]] = [(20.0, 80.0), (80.0, 200.0), (200.0, 500.0)]

    # Features per channel:
    #   aperiodic (2) + knee (2) + residual_rms (1)
    #   + peak bands (3 × 2 = 6) + ratio bands (3) + rolloff (1) = 15
    N_FEATS_PER_CHANNEL: int = 15

    def __init__(
        self,
        sampling_rate: int = 2000,
        nw: float = 4.0,
        rolloff_percentile: float = 0.90,
        window_batch_size: int = 512,
    ):
        """
        Args:
            sampling_rate:      EMG sampling rate (Hz).
            nw:                 DPSS time-bandwidth product NW. K = 2·NW-1 tapers
                                are used (e.g. NW=4 → 7 tapers, half-bandwidth
                                ≈ NW/T·fs = 4/600·2000 ≈ 13 Hz).
            rolloff_percentile: Fraction of cumulative power for rolloff feature
                                (e.g. 0.90 → frequency below which 90% of power lies).
            window_batch_size:  Process this many windows at once to cap memory.
                                Peak memory ≈ batch × C × K × T × 8 bytes
                                (512 × 8 × 7 × 600 × 8 ≈ 137 MB — manageable).
        """
        self.fs = sampling_rate
        self.nw = float(nw)
        self.rolloff_pct = float(rolloff_percentile)
        self.window_batch_size = max(1, int(window_batch_size))

        # Taper cache: invalidated when window size T changes
        self._cached_T: Optional[int] = None
        self._cached_tapers: Optional[np.ndarray] = None  # (K, T)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (N, T, C) EMG windows, float32 or float64.
        Returns:
            features: (N, C × N_FEATS_PER_CHANNEL) float32
        """
        if X.ndim != 3:
            raise ValueError(f"MultitaperPSDExtractor expects (N, T, C), got {X.shape}")

        N, T, C = X.shape
        tapers = self._get_tapers(T)                      # (K, T)
        freqs  = np.fft.rfftfreq(T, d=1.0 / self.fs)     # (F,) Hz

        # Process in batches to bound peak memory.
        # Per-batch cost: B × C × K × T × 8 bytes
        # (B=512, C=8, K=7, T=600 → ~137 MB — safe for both CPU and GPU hosts).
        out = np.zeros((N, C, self.N_FEATS_PER_CHANNEL), dtype=np.float64)

        for batch_start in range(0, N, self.window_batch_size):
            batch = X[batch_start: batch_start + self.window_batch_size]  # (B, T, C)
            B = len(batch)

            # (B, C, 1, T) × (1, 1, K, T) → (B, C, K, T)
            X_chf = batch.transpose(0, 2, 1).astype(np.float64)  # (B, C, T)
            X_tap = X_chf[:, :, np.newaxis, :] * tapers[np.newaxis, np.newaxis, :, :]
            X_fft = np.fft.rfft(X_tap, axis=-1)                   # (B, C, K, F)
            psd_b = (np.abs(X_fft) ** 2).mean(axis=2)             # (B, C, F)

            # One-sided normalisation (double non-DC, non-Nyquist bins)
            n_rfft = psd_b.shape[-1]
            psd_b[:, :, 1: n_rfft - 1] *= 2.0
            psd_b /= (T * self.fs)

            for c in range(C):
                out[batch_start: batch_start + B, c, :] = (
                    self._extract_channel_features(freqs, psd_b[:, c, :])
                )

        feats = out.reshape(N, -1).astype(np.float32)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # DPSS taper caching
    # ------------------------------------------------------------------

    def _get_tapers(self, T: int) -> np.ndarray:
        """Return cached DPSS tapers (K, T), recomputing if T changes."""
        if self._cached_T == T and self._cached_tapers is not None:
            return self._cached_tapers

        from scipy.signal.windows import dpss

        K = max(1, int(2 * self.nw) - 1)   # standard number of tapers
        tapers = dpss(T, self.nw, Kmax=K)  # (K, T) or (T,) when K=1
        if tapers.ndim == 1:
            tapers = tapers[np.newaxis, :]  # (1, T)

        self._cached_T = T
        self._cached_tapers = tapers
        return tapers

    # ------------------------------------------------------------------
    # Per-channel feature extraction (vectorised over N windows)
    # ------------------------------------------------------------------

    def _extract_channel_features(
        self, freqs: np.ndarray, psd: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            freqs: (F,)   frequency bins in Hz (from rfftfreq, identical for all windows).
            psd:   (N, F) one-sided normalised multitaper PSD for one channel.
        Returns:
            (N, N_FEATS_PER_CHANNEL) float64
        """
        N = psd.shape[0]
        feats = np.zeros((N, self.N_FEATS_PER_CHANNEL), dtype=np.float64)

        # Select analysis range
        mask = (freqs >= self.FREQ_MIN_HZ) & (freqs <= self.FREQ_MAX_HZ)
        freqs_s = freqs[mask]           # (F_s,)
        psd_s   = psd[:, mask]          # (N, F_s)

        if freqs_s.size < 4:
            return feats  # pathological — return zeros

        log_f   = np.log10(freqs_s + 1e-12)   # (F_s,)
        log_psd = np.log10(psd_s   + 1e-30)   # (N, F_s)

        # ── Features 0–1: Aperiodic model (log-log linear regression) ────
        offset, exponent = self._batch_loglog_fit(log_f, log_psd)  # (N,), (N,)
        feats[:, 0] = offset    # log₁₀ PSD intercept (aperiodic level)
        feats[:, 1] = exponent  # spectral exponent (positive = falling)

        # ── Features 2–3: Spectral knee ──────────────────────────────────
        # Residual = actual log PSD - aperiodic prediction over the full analysis range.
        # The knee is the frequency where the residual is most positive, capturing
        # where the spectrum "bulges" above the power-law background.
        # Note: this is computed per-window — no cross-window information used.
        lp_aperiodic = (offset[:, np.newaxis]
                        - exponent[:, np.newaxis] * log_f[np.newaxis, :])  # (N, F_s)
        residual = log_psd - lp_aperiodic  # (N, F_s)

        knee_idx = np.argmax(residual, axis=1)                  # (N,)
        feats[:, 2] = freqs_s[knee_idx]                         # knee_freq (Hz)
        feats[:, 3] = residual[np.arange(N), knee_idx]          # knee_residual (log units)

        # ── Feature 4: Residual RMS (total oscillatory content) ──────────
        feats[:, 4] = np.sqrt(np.mean(residual ** 2, axis=1))   # (N,)

        # ── Features 5–10: Residual peak per EMG frequency band ──────────
        # For each band: find the frequency bin with the highest residual above the
        # aperiodic model.  Returns peak frequency (Hz) and its amplitude (log units).
        feat_col = 5
        for f_lo, f_hi in self.PEAK_BANDS:
            band_mask = (freqs_s >= f_lo) & (freqs_s < f_hi)
            if band_mask.sum() < 1:
                feat_col += 2
                continue
            band_freqs    = freqs_s[band_mask]
            band_residual = residual[:, band_mask]          # (N, F_band)
            peak_idx      = np.argmax(band_residual, axis=1)  # (N,)
            feats[:, feat_col]     = band_freqs[peak_idx]
            feats[:, feat_col + 1] = band_residual[np.arange(N), peak_idx]
            feat_col += 2
        # feat_col == 11 after 3 bands

        # ── Features 11–13: Band power ratio vs aperiodic model ──────────
        # Mean residual in each band = mean log(actual / predicted).
        # Subject-specific gain shifts the aperiodic offset uniformly → ratios are
        # invariant to such global amplitude scaling.
        for f_lo, f_hi in self.RATIO_BANDS:
            band_mask = (freqs_s >= f_lo) & (freqs_s < f_hi)
            if band_mask.sum() < 1:
                feat_col += 1
                continue
            feats[:, feat_col] = residual[:, band_mask].mean(axis=1)
            feat_col += 1
        # feat_col == 14 after 3 bands

        # ── Feature 14: Spectral rolloff ──────────────────────────────────
        # Frequency below which rolloff_pct of total power falls.
        # Computed from raw PSD (not log), preserving power distribution shape.
        psd_pos     = np.maximum(psd_s, 0.0)
        cum_power   = np.cumsum(psd_pos, axis=1)           # (N, F_s)
        total_pwr   = cum_power[:, -1:] + 1e-30            # (N, 1)
        cum_norm    = cum_power / total_pwr                 # (N, F_s) in [0, 1]
        rolloff_idx = np.argmax(cum_norm >= self.rolloff_pct, axis=1)  # (N,)
        # If no bin reaches threshold (e.g. all power below FREQ_MIN), use last bin
        no_thresh = ~(cum_norm >= self.rolloff_pct).any(axis=1)
        rolloff_idx[no_thresh] = freqs_s.size - 1
        feats[:, feat_col] = freqs_s[rolloff_idx]
        # feat_col == 14 → last feature slot used

        return feats  # (N, N_FEATS_PER_CHANNEL)

    # ------------------------------------------------------------------
    # Vectorised log-log ordinary least-squares
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_loglog_fit(
        log_f: np.ndarray, log_psd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit  log_psd[i] ≈ offset[i] + slope[i] * log_f  for each of N windows
        using a single batched matrix multiply (no per-window loop).

        The design matrix A = [1 | log_f]  is the same for every window, so we
        precompute (A^T A)^{-1} A^T once and apply it to all N targets.

        Args:
            log_f:   (F,)   log₁₀ frequency values.
            log_psd: (N, F) log₁₀ PSD per window.
        Returns:
            offset:   (N,) intercepts.
            exponent: (N,) −slope  (positive = falling spectrum, i.e. 1/f-like).
        """
        F = log_f.size
        N = log_psd.shape[0]

        # Design matrix: [1, log_f] shape (F, 2)
        A = np.stack([np.ones(F, dtype=np.float64), log_f], axis=1)  # (F, 2)

        try:
            # Precompute pseudo-inverse: (A^T A)^{-1} A^T  shape (2, F)
            AtA_inv = np.linalg.inv(A.T @ A)           # (2, 2)
            proj    = AtA_inv @ A.T                    # (2, F)
        except np.linalg.LinAlgError:
            return np.zeros(N), np.zeros(N)

        # coefs: (2, N) = (2, F) @ (F, N)
        coefs    = proj @ log_psd.T    # (2, N)
        offset   =  coefs[0]           # (N,) intercepts
        exponent = -coefs[1]           # (N,) negate so positive = falling spectrum

        return offset, exponent


# ===========================================================================
#  Combined extractor: MultitaperPSDExtractor + PowerfulFeatureExtractor
# ===========================================================================

class CombinedMultitaperExtractor:
    """
    Concatenates MultitaperPSDExtractor and PowerfulFeatureExtractor outputs.

    PowerfulFeatureExtractor is used with use_entropy=False to avoid the
    placeholder-zero entropy features it includes by default, keeping the
    feature vector dense and informative.

    Input:  X  (N, T, C)
    Output:     (N, F_multitaper + F_powerful)  float32
    """

    def __init__(
        self,
        sampling_rate: int = 2000,
        nw: float = 4.0,
        window_batch_size: int = 512,
    ):
        from processing.powerful_features import PowerfulFeatureExtractor

        self.multitaper = MultitaperPSDExtractor(
            sampling_rate=sampling_rate,
            nw=nw,
            rolloff_percentile=0.90,
            window_batch_size=window_batch_size,
        )
        self.powerful = PowerfulFeatureExtractor(
            sampling_rate=sampling_rate,
            use_entropy=False,   # avoid zero-filled entropy placeholder
            n_jobs=-1,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C)
        Returns: (N, F) float32  where F = F_multitaper + F_powerful
        """
        f_mt  = self.multitaper.transform(X)   # (N, C × 15)
        f_pow = self.powerful.transform(X)     # (N, F_powerful)
        return np.concatenate([f_mt, f_pow], axis=1).astype(np.float32)


# ===========================================================================
#  ML trainer with LightGBM support
# ===========================================================================

class MultitaperMLTrainer:
    """
    Thin wrapper around FeatureMLTrainer that adds 'lgbm' as a valid
    ml_model_type (falls back gracefully if LightGBM is not installed).
    Mirrors the same pattern used in exp_38.
    """

    def __new__(cls, **kwargs):
        from training.trainer import FeatureMLTrainer

        class _MultitaperMLTrainer(FeatureMLTrainer):
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

        return _MultitaperMLTrainer(**kwargs)


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
    feature_extractor: CombinedMultitaperExtractor,
) -> Dict:
    """
    Execute one LOSO fold: train on train_subjects, evaluate on test_subject.

    LOSO purity:
    - feature_extractor is stateless (no learned parameters).
    - train_cfg.ml_model_type determines the classifier.
    - FeatureMLTrainer.fit() standardises features using only training data.
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
        use_gpu=False,          # multitaper runs in NumPy — no GPU needed
        use_improved_processing=True,
    )

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    trainer = MultitaperMLTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )
    # Inject the combined feature extractor so FeatureMLTrainer uses it instead
    # of constructing its own PowerfulFeatureExtractor.
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
        print(f"Error in fold (test={test_subject}, model={model_type}): {e}")
        traceback.print_exc()
        return {
            "test_subject":   test_subject,
            "model_type":     model_type,
            "test_accuracy":  None,
            "test_f1_macro":  None,
            "error":          str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1  = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test={test_subject} | Model={model_type} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )

    results_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "model_type":     model_type,
            "approach":       "multitaper_psd_spectral_slope",
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
        "test_subject":  test_subject,
        "model_type":    model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ===========================================================================
#  Main
# ===========================================================================

def main():
    EXPERIMENT_NAME = "exp_68_multitaper_psd_spectral_slope_loso"
    BASE_DIR     = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()

    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES  = ["E1"]

    # Models to evaluate
    MODEL_TYPES = ["svm_rbf", "svm_linear", "lgbm"]

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print(f"[{EXPERIMENT_NAME}] LightGBM not installed — 'lgbm' falls back to SVM-RBF.")

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

    # Build feature extractor once — stateless, reused across all folds.
    # NW=4 → K=7 tapers, half-bandwidth ≈ 13 Hz at 2000 Hz / 600 samples.
    # window_batch_size=512: peak memory ≈ 512 × 8 × 7 × 600 × 8 B ≈ 137 MB.
    feature_extractor = CombinedMultitaperExtractor(
        sampling_rate=proc_cfg.sampling_rate,
        nw=4.0,
        window_batch_size=512,
    )

    # Compute and print expected feature counts
    n_mt_feats = 8 * MultitaperPSDExtractor.N_FEATS_PER_CHANNEL
    print(f"[{EXPERIMENT_NAME}] Subjects    : {ALL_SUBJECTS}")
    print(f"[{EXPERIMENT_NAME}] Models      : {MODEL_TYPES}")
    print(f"[{EXPERIMENT_NAME}] Exercises   : {EXERCISES}")
    print(f"[{EXPERIMENT_NAME}] Multitaper  : NW=4.0, K=7 tapers, "
          f"half-BW ≈ 13 Hz  →  {n_mt_feats} features (8 ch × 15)")
    print(f"[{EXPERIMENT_NAME}] Combined    : {n_mt_feats} multitaper"
          f" + PowerfulFeatureExtractor(no-entropy) ≈ ~373 total")

    all_loso_results: List[Dict] = []

    for model_type in MODEL_TYPES:
        print(f"\n{'=' * 65}")
        print(f"  Model: {model_type}  —  starting LOSO ({len(ALL_SUBJECTS)} folds)")
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

    # ── Aggregate results ──────────────────────────────────────────────────
    aggregate_results: Dict = {}
    for model_type in MODEL_TYPES:
        model_results = [
            r for r in all_loso_results
            if r["model_type"] == model_type and r.get("test_accuracy") is not None
        ]
        if not model_results:
            continue
        accs = [r["test_accuracy"] for r in model_results]
        f1s  = [r["test_f1_macro"]  for r in model_results]
        aggregate_results[model_type] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_subjects":  len(accs),
            "per_subject":   model_results,
        }

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"SUMMARY: {EXPERIMENT_NAME}")
    print(f"{'=' * 65}")
    for mt, res in aggregate_results.items():
        acc_m = res["mean_accuracy"]
        acc_s = res["std_accuracy"]
        f1_m  = res["mean_f1_macro"]
        f1_s  = res["std_f1_macro"]
        if acc_m is not None:
            print(f"  {mt:12s}: Acc={acc_m:.4f} ± {acc_s:.4f},"
                  f"  F1={f1_m:.4f} ± {f1_s:.4f}")

    # ── Save summary JSON ──────────────────────────────────────────────────
    loso_summary = {
        "experiment_name":   EXPERIMENT_NAME,
        "hypothesis":        (
            "Multitaper PSD aperiodic exponent, spectral knee, and residual oscillatory "
            "peaks (FOOOF-style, log-log linear fit) are more subject-invariant than raw "
            "band power or amplitude. Combined with PowerfulFeatureExtractor for SVM/LGBM."
        ),
        "approach":          "MultitaperPSDExtractor + PowerfulFeatureExtractor + SVM / LGBM",
        "multitaper_params": {
            "nw":             4.0,
            "k_tapers":       7,
            "half_bw_hz":     13.3,
            "freq_range_hz":  [20.0, 450.0],
            "peak_bands_hz":  MultitaperPSDExtractor.PEAK_BANDS,
            "ratio_bands_hz": MultitaperPSDExtractor.RATIO_BANDS,
            "rolloff_pct":    0.90,
        },
        "features_per_channel": MultitaperPSDExtractor.N_FEATS_PER_CHANNEL,
        "multitaper_features_total": 8 * MultitaperPSDExtractor.N_FEATS_PER_CHANNEL,
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
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)
    print(f"\n[DONE] {EXPERIMENT_NAME} → {summary_path}")

    # ── Notify hypothesis executor (optional dependency) ──────────────────
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
                hypothesis_id="H68",
                metrics={
                    "best_model":        best_model,
                    "mean_accuracy":     best_res["mean_accuracy"],
                    "mean_f1_macro":     best_res["mean_f1_macro"],
                    "std_accuracy":      best_res["std_accuracy"],
                    "std_f1_macro":      best_res["std_f1_macro"],
                    "aggregate_results": aggregate_results,
                },
                experiment_name=EXPERIMENT_NAME,
            )
    except ImportError:
        pass
    except Exception as _he_err:
        print(f"[{EXPERIMENT_NAME}] hypothesis_executor notification failed: {_he_err}")


if __name__ == "__main__":
    main()
