"""
Experiment 84: Marginal Kurtosis Frequency Band Selection (LOSO)

Based on: Zeng et al. (IEEE Access 2021) — Marginal Kurtosis and MK-Dominant Frequency

Hypothesis H84:
    Marginal Kurtosis (MK) analysis of the EMG power spectrum identifies frequency
    bands that carry gesture-discriminative information. For each frequency bin,
    MK = kurtosis({P_i(f)}_{i=1}^N) where P_i(f) is the power spectral density
    of training window i at frequency f.

    High MK indicates non-Gaussian (bursty) power fluctuations at that frequency —
    characteristic of transient MUAP activity during gesture transitions.
    Low MK indicates relatively constant power — likely noise floor or tonic baseline.

    The pipeline:
        1. Divide the frequency axis into fixed sub-bands.
        2. Compute MK for each sub-band from training windows ONLY.
        3. Select the top-K bands by MK (highest cross-window variability).
        4. Apply a smooth spectral mask to retain only the selected bands.
        5. Extract PowerfulFeatureExtractor features from the filtered signal.
        6. Extract additional MK-specific spectral features (band powers, energy
           ratio, spectral centroid, peak frequency within selected bands).
        7. Train SVM / LGBM on the combined feature set.

    This is an analytical (non-learned) alternative to Sinc-filter approaches
    (exp_61), providing principled frequency-domain preprocessing before any model.

Related experiments:
    - exp_61 (Sinc-PCEN Frontend): learned Sinc filters end-to-end — MK is analytic
    - exp_64 (Multiclass CSP Filterbank): spatial filtering — MK is spectral filtering
    - exp_68 (Multitaper PSD): spectral features without pre-filtering
    - exp_76 (Soft AGC + PCEN-lite): different normalization approach

LOSO purity:
    - MK profile is computed ONLY from training subjects' windows in each fold.
    - The spectral mask is applied identically to train, val, and test data.
    - No test-subject data participates in MK computation or band selection.
    - Feature standardization uses only training-split statistics (FeatureMLTrainer).
    - All analysis hyperparameters (N sub-bands, analysis range, top-K ratio) are
      fixed constants — no data-dependent tuning.

Usage:
    python experiments/exp_84_marginal_kurtosis_frequency_band_selection_loso.py
    python experiments/exp_84_marginal_kurtosis_frequency_band_selection_loso.py --ci
    python experiments/exp_84_marginal_kurtosis_frequency_band_selection_loso.py --full
    python experiments/exp_84_marginal_kurtosis_frequency_band_selection_loso.py --subjects DB2_s1,DB2_s12
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
#  Marginal Kurtosis Analyzer
# ===========================================================================

class MarginalKurtosisAnalyzer:
    """
    Computes Marginal Kurtosis (MK) profile from EMG training windows and
    builds a frequency-domain spectral mask for band selection.

    MK at sub-band b = kurtosis({BandPower_i(b)}_{i=1}^N) computed across all
    N training windows. High MK indicates gesture-informative frequency content.

    LOSO usage:
        analyzer = MarginalKurtosisAnalyzer(sampling_rate=2000)
        analyzer.fit(X_train)              # training windows ONLY
        X_filtered = analyzer.filter(X)    # apply to any windows
        mk_feats = analyzer.extract_features(X)  # spectral features

    All analysis parameters (sub-band count, frequency range, top-K ratio) are
    fixed constants — no data-dependent tuning.
    """

    # Fixed constants
    ANALYSIS_FREQ_MIN: float = 10.0    # Hz — below this is DC/drift
    ANALYSIS_FREQ_MAX: float = 500.0   # Hz — above this is mostly noise at 2 kHz

    def __init__(
        self,
        sampling_rate: int = 2000,
        n_subbands: int = 30,
        top_k_ratio: float = 0.5,
        transition_width: float = 0.15,
        batch_size: int = 512,
    ):
        """
        Args:
            sampling_rate:    EMG sampling rate (Hz).
            n_subbands:       Number of equal-width sub-bands in analysis range.
                              30 sub-bands at 10–500 Hz → ~16.3 Hz per band.
            top_k_ratio:      Fraction of bands to keep (by MK ranking).
                              0.5 → keep top 15 of 30 bands.
            transition_width: Width of cosine rolloff at band edges, as a fraction
                              of band width. Prevents spectral ringing.
            batch_size:       Process windows in batches to cap memory usage.
        """
        self.fs = sampling_rate
        self.n_subbands = n_subbands
        self.top_k = max(1, int(n_subbands * top_k_ratio))
        self.transition_width = transition_width
        self.batch_size = max(1, batch_size)

        # Computed by fit() — training data dependent
        self._mk_profile: Optional[np.ndarray] = None       # (n_subbands,)
        self._mk_per_channel: Optional[np.ndarray] = None   # (n_subbands, C)
        self._selected_bands: Optional[np.ndarray] = None   # (top_k,) indices
        self._spectral_mask: Optional[np.ndarray] = None    # (n_rfft,) mask
        self._band_edges: Optional[np.ndarray] = None       # (n_subbands+1,) Hz
        self._n_fft: Optional[int] = None
        self._n_channels: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self._spectral_mask is not None

    # ------------------------------------------------------------------
    # fit: compute MK from training windows ONLY
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> None:
        """
        Compute MK profile from training windows and build spectral mask.

        Args:
            X_train: (N, T, C) training EMG windows. MUST be training data only.
        """
        from scipy.stats import kurtosis

        if X_train.ndim != 3:
            raise ValueError(f"Expected (N, T, C), got shape {X_train.shape}")

        N, T, C = X_train.shape
        self._n_fft = T
        self._n_channels = C

        freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)  # (F,)

        # Define sub-band edges (linearly spaced within analysis range)
        self._band_edges = np.linspace(
            self.ANALYSIS_FREQ_MIN, self.ANALYSIS_FREQ_MAX, self.n_subbands + 1
        )

        # Compute band powers for all training windows: (N, n_subbands, C)
        band_powers = np.zeros((N, self.n_subbands, C), dtype=np.float64)

        for start in range(0, N, self.batch_size):
            batch = X_train[start: start + self.batch_size]  # (B, T, C)
            B = len(batch)

            # FFT: (B, T, C) → transpose → (B, C, T) → rfft → (B, C, F)
            X_freq = np.fft.rfft(batch.transpose(0, 2, 1).astype(np.float64), axis=-1)
            psd = np.abs(X_freq) ** 2 / T  # (B, C, F)

            for b in range(self.n_subbands):
                f_lo, f_hi = self._band_edges[b], self._band_edges[b + 1]
                mask = (freqs >= f_lo) & (freqs < f_hi)
                if mask.sum() > 0:
                    # Mean PSD within band, per channel: (B, C)
                    band_powers[start: start + B, b, :] = psd[:, :, mask].mean(axis=-1)

        # Compute MK per sub-band per channel
        # MK(band, channel) = kurtosis of band_power across N training windows
        self._mk_per_channel = np.zeros((self.n_subbands, C), dtype=np.float64)
        for b in range(self.n_subbands):
            for c in range(C):
                bp = band_powers[:, b, c]
                if bp.std() > 1e-12:
                    self._mk_per_channel[b, c] = kurtosis(bp, fisher=True)
                else:
                    self._mk_per_channel[b, c] = 0.0

        # Average MK across channels for global band selection
        self._mk_profile = self._mk_per_channel.mean(axis=1)  # (n_subbands,)

        # Select top-K bands by MK (highest kurtosis = most informative)
        sorted_idx = np.argsort(self._mk_profile)[::-1]
        self._selected_bands = np.sort(sorted_idx[: self.top_k])

        # Build smooth spectral mask
        self._spectral_mask = self._build_spectral_mask(freqs)

    # ------------------------------------------------------------------
    # Spectral mask construction
    # ------------------------------------------------------------------

    def _build_spectral_mask(self, freqs: np.ndarray) -> np.ndarray:
        """
        Build smooth frequency-domain mask from selected bands.

        Each selected band has a flat passband with cosine rolloff transitions
        at the edges. Overlapping transitions from adjacent bands are resolved
        by taking the maximum (not summing), preventing gain > 1.

        Args:
            freqs: (F,) frequency bins in Hz (from rfftfreq).
        Returns:
            mask: (F,) values in [0, 1].
        """
        mask = np.zeros(len(freqs), dtype=np.float64)

        for band_idx in self._selected_bands:
            f_lo = self._band_edges[band_idx]
            f_hi = self._band_edges[band_idx + 1]
            bw = f_hi - f_lo
            trans = bw * self.transition_width

            # Flat passband
            in_pass = (freqs >= f_lo + trans) & (freqs <= f_hi - trans)
            mask[in_pass] = 1.0

            # Lower cosine transition
            if trans > 1e-6:
                in_lower = (freqs >= f_lo) & (freqs < f_lo + trans)
                if in_lower.any():
                    t = (freqs[in_lower] - f_lo) / trans
                    values = 0.5 * (1.0 - np.cos(np.pi * t))
                    mask[in_lower] = np.maximum(mask[in_lower], values)

            # Upper cosine transition
            if trans > 1e-6:
                in_upper = (freqs > f_hi - trans) & (freqs <= f_hi)
                if in_upper.any():
                    t = (freqs[in_upper] - (f_hi - trans)) / trans
                    values = 0.5 * (1.0 + np.cos(np.pi * t))
                    mask[in_upper] = np.maximum(mask[in_upper], values)

        return mask

    # ------------------------------------------------------------------
    # filter: apply spectral mask to any windows
    # ------------------------------------------------------------------

    def filter(self, X: np.ndarray) -> np.ndarray:
        """
        Apply MK-derived spectral mask to filter EMG windows.

        Performs FFT → multiply by mask → IFFT. The mask was computed in fit()
        from training data only. Applied identically to train/val/test.

        Args:
            X: (N, T, C) EMG windows.
        Returns:
            X_filtered: (N, T, C) filtered windows (same shape).
        """
        assert self.is_fitted, "Call fit() before filter()"
        N, T, C = X.shape
        if T != self._n_fft:
            raise ValueError(
                f"Window size mismatch: fit() saw T={self._n_fft}, got T={T}"
            )

        X_filtered = np.zeros_like(X)
        for start in range(0, N, self.batch_size):
            batch = X[start: start + self.batch_size]  # (B, T, C)
            B = len(batch)

            # (B, C, T) → rfft → (B, C, F)
            X_freq = np.fft.rfft(
                batch.transpose(0, 2, 1).astype(np.float64), axis=-1
            )
            # Apply mask: (B, C, F) × (1, 1, F)
            X_freq *= self._spectral_mask[np.newaxis, np.newaxis, :]
            # IFFT back to time domain: (B, C, T)
            X_time = np.fft.irfft(X_freq, n=T, axis=-1)
            X_filtered[start: start + B] = X_time.transpose(0, 2, 1)

        return X_filtered.astype(np.float32)

    # ------------------------------------------------------------------
    # extract_features: MK-specific spectral features
    # ------------------------------------------------------------------

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract MK-related spectral features per window.

        Features per channel (C channels):
            0:              Energy ratio (selected bands / total) — how much of the
                            signal's energy falls within the MK-selected bands.
            1:              Spectral centroid of the MK-selected bands (Hz).
            2:              Peak frequency within MK-selected bands (Hz).
            3 .. 3+top_k-1: Log band power for each of the top-K selected bands.

        Total features: C × (3 + top_k)

        Args:
            X: (N, T, C) EMG windows.
        Returns:
            features: (N, C × (3 + top_k)) float32.
        """
        assert self.is_fitted, "Call fit() before extract_features()"
        N, T, C = X.shape

        freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)
        n_feats_per_ch = 3 + self.top_k
        features = np.zeros((N, C, n_feats_per_ch), dtype=np.float64)

        for start in range(0, N, self.batch_size):
            batch = X[start: start + self.batch_size]  # (B, T, C)
            B = len(batch)

            # PSD: (B, C, F)
            X_freq = np.fft.rfft(
                batch.transpose(0, 2, 1).astype(np.float64), axis=-1
            )
            psd = np.abs(X_freq) ** 2 / T  # (B, C, F)

            for c in range(C):
                psd_c = psd[:, c, :]  # (B, F)

                # Total energy
                total_energy = psd_c.sum(axis=1) + 1e-30  # (B,)

                # Energy in selected bands
                selected_energy = np.zeros(B, dtype=np.float64)
                # Weighted PSD for centroid computation
                weighted_freq_sum = np.zeros(B, dtype=np.float64)
                selected_psd_total = np.zeros(B, dtype=np.float64)

                for band_idx in self._selected_bands:
                    f_lo = self._band_edges[band_idx]
                    f_hi = self._band_edges[band_idx + 1]
                    band_mask = (freqs >= f_lo) & (freqs < f_hi)
                    if band_mask.sum() > 0:
                        bp = psd_c[:, band_mask].sum(axis=1)  # (B,)
                        selected_energy += bp
                        # For spectral centroid: sum(f * P(f)) / sum(P(f))
                        weighted_freq_sum += (
                            psd_c[:, band_mask] * freqs[band_mask][np.newaxis, :]
                        ).sum(axis=1)
                        selected_psd_total += bp

                # Feature 0: Energy ratio
                features[start: start + B, c, 0] = selected_energy / total_energy

                # Feature 1: Spectral centroid of selected bands (Hz)
                safe_total = selected_psd_total + 1e-30
                features[start: start + B, c, 1] = weighted_freq_sum / safe_total

                # Feature 2: Peak frequency within selected bands
                masked_psd = psd_c * self._spectral_mask[np.newaxis, :]
                peak_idx = np.argmax(masked_psd, axis=1)  # (B,)
                features[start: start + B, c, 2] = freqs[peak_idx]

                # Features 3 .. 3+top_k-1: Log band power per selected band
                for ki, band_idx in enumerate(self._selected_bands):
                    f_lo = self._band_edges[band_idx]
                    f_hi = self._band_edges[band_idx + 1]
                    band_mask = (freqs >= f_lo) & (freqs < f_hi)
                    if band_mask.sum() > 0:
                        bp = psd_c[:, band_mask].mean(axis=1)
                        features[start: start + B, c, 3 + ki] = np.log1p(bp)

        feats = features.reshape(N, -1).astype(np.float32)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ===========================================================================
#  Combined Feature Extractor: MK Filtering + PowerfulFeatures + MK Features
# ===========================================================================

class MKFilteredFeatureExtractor:
    """
    Combines MK spectral filtering with feature extraction.

    Pipeline (per call to transform()):
        1. Apply MK spectral filter to input → filtered signal (N, T, C)
        2. Extract PowerfulFeatureExtractor features from filtered signal → (N, F_pow)
        3. Extract MK-specific spectral features from original signal → (N, F_mk)
        4. Concatenate → (N, F_pow + F_mk)

    The MK analyzer must be pre-fitted (on training data) before calling transform().

    Input:  X  (N, T, C)
    Output:     (N, F_pow + F_mk) float32
    """

    def __init__(
        self,
        mk_analyzer: MarginalKurtosisAnalyzer,
        sampling_rate: int = 2000,
    ):
        from processing.powerful_features import PowerfulFeatureExtractor

        assert mk_analyzer.is_fitted, "MK analyzer must be fitted before use"
        self.mk_analyzer = mk_analyzer
        self.powerful = PowerfulFeatureExtractor(
            sampling_rate=sampling_rate,
            use_entropy=False,  # avoid zero-filled entropy placeholder
            n_jobs=-1,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C)
        Returns: (N, F_pow + F_mk) float32
        """
        # 1. Apply MK spectral filter
        X_filtered = self.mk_analyzer.filter(X)  # (N, T, C)

        # 2. Extract PowerfulFeatures from FILTERED signal
        f_pow = self.powerful.transform(X_filtered)  # (N, F_pow)

        # 3. Extract MK-specific features from ORIGINAL signal
        f_mk = self.mk_analyzer.extract_features(X)  # (N, F_mk)

        # 4. Concatenate
        return np.concatenate([f_pow, f_mk], axis=1).astype(np.float32)


# ===========================================================================
#  Custom ML Trainer with MK fitting
# ===========================================================================

def _create_mk_trainer(**kwargs):
    """
    Factory for MK-aware FeatureMLTrainer.

    Creates a trainer subclass that:
    1. Fits the MK analyzer on training windows inside fit().
    2. Creates a MKFilteredFeatureExtractor with the fitted analyzer.
    3. Delegates to FeatureMLTrainer.fit() for standardization + ML training.

    LOSO purity: MK is fitted ONLY on training windows from splits["train"].
    """
    from training.trainer import FeatureMLTrainer

    class _MKFeatureMLTrainer(FeatureMLTrainer):

        def __init__(self, mk_params: Dict, **kw):
            super().__init__(**kw)
            self.mk_params = mk_params
            self.mk_analyzer: Optional[MarginalKurtosisAnalyzer] = None

        def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
            """
            Override to fit MK analyzer on training data before feature extraction.

            Steps:
                1. Extract training arrays from splits["train"].
                2. Fit MarginalKurtosisAnalyzer on training windows ONLY.
                3. Create MKFilteredFeatureExtractor with fitted analyzer.
                4. Delegate to parent FeatureMLTrainer.fit().
            """
            # Step 1: Extract training arrays for MK fitting
            train_d = {
                gid: arr for gid, arr in splits["train"].items()
                if isinstance(arr, np.ndarray) and len(arr) > 0
            }
            if not train_d:
                raise ValueError("No training data in splits['train']")

            X_train_list = [train_d[gid] for gid in sorted(train_d.keys())]
            X_train_raw = np.concatenate(X_train_list, axis=0)  # (N, T, C)

            # Step 2: Fit MK on training data ONLY — no val/test involved
            self.mk_analyzer = MarginalKurtosisAnalyzer(**self.mk_params)
            self.mk_analyzer.fit(X_train_raw)

            n_train = X_train_raw.shape[0]
            del X_train_raw  # free memory

            self.logger.info(
                f"[MKTrainer] MK analyzer fitted on {n_train} training windows. "
                f"Selected {self.mk_analyzer.top_k}/{self.mk_analyzer.n_subbands} "
                f"bands (indices: {self.mk_analyzer._selected_bands.tolist()})."
            )

            # Log MK profile
            for b in range(self.mk_analyzer.n_subbands):
                f_lo = self.mk_analyzer._band_edges[b]
                f_hi = self.mk_analyzer._band_edges[b + 1]
                mk_val = self.mk_analyzer._mk_profile[b]
                selected = "***" if b in self.mk_analyzer._selected_bands else "   "
                self.logger.info(
                    f"  Band {b:2d} [{f_lo:6.1f}–{f_hi:6.1f} Hz]: "
                    f"MK={mk_val:8.2f} {selected}"
                )

            # Step 3: Create combined feature extractor
            self.feature_extractor = MKFilteredFeatureExtractor(
                mk_analyzer=self.mk_analyzer,
                sampling_rate=self.mk_params["sampling_rate"],
            )

            # Step 4: Delegate to parent for standardization + ML training
            return super().fit(splits)

        def _create_ml_model(self, model_type: str):
            """Add LGBM support alongside SVM."""
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

    return _MKFeatureMLTrainer(**kwargs)


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
    mk_params: Dict,
    proc_cfg,
    split_cfg,
    train_cfg,
) -> Dict:
    """
    Execute one LOSO fold with MK-based frequency band selection.

    LOSO purity:
    - MK analyzer is fitted inside trainer.fit() on training windows only.
    - Feature standardization uses only training statistics.
    - No test-subject data participates in any fitting step.
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
    train_cfg.model_type = model_type
    train_cfg.ml_model_type = model_type

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
        use_gpu=False,
        use_improved_processing=True,
    )

    base_viz = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    # Create MK-aware trainer — MK fitting happens inside trainer.fit()
    trainer = _create_mk_trainer(
        mk_params=mk_params,
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )

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
            "test_subject":  test_subject,
            "model_type":    model_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test={test_subject} | Model={model_type} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )

    # Save results (excluding raw subject data)
    results_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_save), f, indent=4, ensure_ascii=False)

    # Save MK analysis info
    mk_info = {
        "n_subbands": mk_params["n_subbands"],
        "top_k_ratio": mk_params["top_k_ratio"],
        "top_k": trainer.mk_analyzer.top_k if trainer.mk_analyzer else None,
        "selected_bands": (
            trainer.mk_analyzer._selected_bands.tolist()
            if trainer.mk_analyzer else None
        ),
        "mk_profile": (
            trainer.mk_analyzer._mk_profile.tolist()
            if trainer.mk_analyzer else None
        ),
        "band_edges": (
            trainer.mk_analyzer._band_edges.tolist()
            if trainer.mk_analyzer else None
        ),
    }
    with open(output_dir / "mk_analysis.json", "w") as f:
        json.dump(make_json_serializable(mk_info), f, indent=4)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "model_type":     model_type,
            "approach":       "marginal_kurtosis_band_selection",
            "exercises":      exercises,
            "mk_params":      mk_params,
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # Cleanup
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
    EXPERIMENT_NAME = "exp_84_marginal_kurtosis_frequency_band_selection_loso"
    BASE_DIR = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()

    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES = ["E1"]

    # Models to evaluate
    MODEL_TYPES = ["svm_rbf", "svm_linear", "lgbm"]

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print(
            f"[{EXPERIMENT_NAME}] LightGBM not installed — "
            f"'lgbm' falls back to SVM-RBF."
        )

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
        handcrafted_feature_set="powerful",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_rbf",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
    )

    # MK analysis parameters — all fixed constants
    mk_params = {
        "sampling_rate": proc_cfg.sampling_rate,
        "n_subbands": 30,          # 30 bands in 10-500 Hz → ~16.3 Hz each
        "top_k_ratio": 0.5,        # keep top 50% (15 bands)
        "transition_width": 0.15,  # cosine rolloff = 15% of band width
        "batch_size": 512,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    # Feature count estimate
    n_mk_feats_per_ch = 3 + max(1, int(mk_params["n_subbands"] * mk_params["top_k_ratio"]))
    n_mk_feats = 8 * n_mk_feats_per_ch
    print(f"[{EXPERIMENT_NAME}] Subjects    : {ALL_SUBJECTS}")
    print(f"[{EXPERIMENT_NAME}] Models      : {MODEL_TYPES}")
    print(f"[{EXPERIMENT_NAME}] Exercises   : {EXERCISES}")
    print(f"[{EXPERIMENT_NAME}] MK params   : {mk_params['n_subbands']} sub-bands, "
          f"top {mk_params['top_k_ratio']*100:.0f}% kept, "
          f"analysis range {MarginalKurtosisAnalyzer.ANALYSIS_FREQ_MIN:.0f}–"
          f"{MarginalKurtosisAnalyzer.ANALYSIS_FREQ_MAX:.0f} Hz")
    print(f"[{EXPERIMENT_NAME}] MK features : {n_mk_feats} "
          f"(8 ch x {n_mk_feats_per_ch} feats/ch)")
    print(f"[{EXPERIMENT_NAME}] Combined    : {n_mk_feats} MK + "
          f"PowerfulFeatureExtractor(no-entropy) on filtered signal")

    all_loso_results: List[Dict] = []

    for model_type in MODEL_TYPES:
        print(f"\n{'=' * 65}")
        print(f"  Model: {model_type}  —  starting LOSO ({len(ALL_SUBJECTS)} folds)")
        print(f"{'=' * 65}")

        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"

            train_cfg.ml_model_type = model_type

            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                model_type=model_type,
                mk_params=mk_params,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
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
        f1s = [r["test_f1_macro"] for r in model_results]
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
        f1_m = res["mean_f1_macro"]
        f1_s = res["std_f1_macro"]
        if acc_m is not None and f1_m is not None:
            print(
                f"  {mt:12s}: Acc={acc_m:.4f} +/- {acc_s:.4f},  "
                f"F1={f1_m:.4f} +/- {f1_s:.4f}"
            )

    # ── Save summary JSON ──────────────────────────────────────────────────
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "Marginal Kurtosis (MK) identifies frequency bands with non-Gaussian "
            "(bursty) power fluctuations across training windows, characteristic of "
            "gesture-related MUAP activity. Filtering to retain only high-MK bands "
            "removes noise and focuses feature extraction on informative content."
        ),
        "approach": (
            "MK band selection (training-only) → spectral filtering → "
            "PowerfulFeatureExtractor + MK spectral features → SVM / LGBM"
        ),
        "mk_params": mk_params,
        "mk_analysis_range_hz": [
            MarginalKurtosisAnalyzer.ANALYSIS_FREQ_MIN,
            MarginalKurtosisAnalyzer.ANALYSIS_FREQ_MAX,
        ],
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False
        )
    print(f"\n[DONE] {EXPERIMENT_NAME} -> {summary_path}")

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
                hypothesis_id="H84",
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
