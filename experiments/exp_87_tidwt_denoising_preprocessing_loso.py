"""
Experiment 87: Translation-Invariant DWT (SWT) Denoising Preprocessing for Cross-Subject EMG

Source: Zeng et al. (IEEE Access 2021)

Hypothesis:
    Using the Stationary Wavelet Transform (SWT / TIDWT / undecimated DWT) for EMG signal
    denoising before feature extraction improves cross-subject classification accuracy.
    Unlike standard DWT, SWT is shift-invariant — critical for EMG where gesture onset
    timing varies within windows. BayesShrink adaptive soft thresholding removes noise
    while preserving discriminative signal structure.

Approach:
    1. Apply SWT denoising to raw EMG windows (per-window, per-channel):
       - Decompose signal using SWT with Daubechies-4 wavelet (4 levels)
       - Estimate noise sigma from finest detail coefficients (MAD estimator)
       - Apply BayesShrink adaptive soft thresholding to detail coefficients
       - Reconstruct denoised signal via inverse SWT
    2. Extract PowerfulFeatures from denoised windows
    3. Classify with SVM (RBF + Linear)

LOSO Compliance:
    - SWT denoising is entirely per-window, per-channel — zero cross-window information
    - Noise sigma estimated independently for each individual window from its own
      finest detail coefficients only (MAD estimator)
    - BayesShrink thresholds computed per-window, per-level — no training data stats
    - Wavelet family and decomposition level are fixed hyperparameters (not tuned on data)
    - Feature standardization in FeatureMLTrainer uses only training data statistics
    - No subject-specific adaptation of any kind

Relation to existing experiments:
    - exp_33: Wavelet scattering transform for feature extraction (not denoising)
    - exp_4: ML with powerful features on raw EMG (baseline without denoising)
    - This experiment: SWT denoising as preprocessing before powerful features + SVM

Dependencies:
    - PyWavelets (pywt): pip install PyWavelets
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch

try:
    import pywt
except ImportError:
    raise ImportError(
        "PyWavelets is required for experiment 87 (TIDWT denoising). "
        "Install with: pip install PyWavelets"
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------- Subject lists ---------------

_FULL_SUBJECTS = [
    "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]

_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]


def parse_subjects_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--full", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.subjects:
        return [s.strip() for s in args.subjects.split(",")]
    if args.full:
        return _FULL_SUBJECTS
    return _CI_SUBJECTS


# =========================================================================
#  TIDWT (Stationary Wavelet Transform) Denoiser
# =========================================================================

class TIDWTDenoiser:
    """
    Translation-Invariant DWT denoiser for EMG signals using the Stationary
    Wavelet Transform (SWT / undecimated DWT).

    Unlike standard DWT, SWT is shift-invariant: the denoised output does not
    depend on the alignment of the gesture within the window. This is critical
    for EMG where gesture onset timing varies across repetitions.

    Denoising is performed per-window, per-channel — fully LOSO-compliant.
    No cross-window, cross-channel, or cross-subject information is used.

    Algorithm per window per channel:
        1. Pad signal to next multiple of 2^level (symmetric padding)
        2. Compute SWT decomposition (no downsampling, shift-invariant)
        3. Estimate noise sigma from finest detail coefficients via MAD
        4. Compute adaptive threshold per level (BayesShrink)
        5. Apply soft thresholding to all detail coefficients
        6. Reconstruct denoised signal via inverse SWT
        7. Trim back to original length

    Frequency bands at 2kHz sampling rate with level=4:
        Level 1 details: 500-1000 Hz  (predominantly noise)
        Level 2 details: 250-500 Hz   (upper EMG band + noise)
        Level 3 details: 125-250 Hz   (core EMG signal)
        Level 4 details: 62.5-125 Hz  (core EMG signal)
        Approximation:   0-62.5 Hz    (low-frequency EMG content)
    BayesShrink adapts thresholds automatically — aggressive on noisy levels,
    gentle on signal-rich levels.
    """

    # Maximum windows processed at once to limit memory (~200 MB per batch)
    _MAX_BATCH = 10_000

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
        threshold_mode: str = "soft",
        threshold_rule: str = "bayes",
    ):
        """
        Args:
            wavelet: Wavelet family. 'db4' is standard for EMG denoising.
            level: SWT decomposition depth (4 covers 0-1000 Hz in 5 bands at 2kHz).
            threshold_mode: 'soft' (smooth, default) or 'hard'.
            threshold_rule: 'bayes' (adaptive BayesShrink, default) or 'universal'
                            (VisuShrink — single global threshold).
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self.threshold_rule = threshold_rule

    # ----- internal helpers -----

    def _pad_signal(self, x: np.ndarray) -> tuple:
        """Pad last axis to next multiple of 2^level for SWT compatibility.

        Uses symmetric padding (signal mirrored at boundary) to avoid
        edge artifacts that zero-padding would introduce.

        Args:
            x: (..., T) array — last axis is time.

        Returns:
            (x_padded, orig_len)
        """
        orig_len = x.shape[-1]
        factor = 2 ** self.level
        remainder = orig_len % factor
        if remainder == 0:
            return x, orig_len
        pad_len = factor - remainder
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad_len)]
        x_padded = np.pad(x, pad_width, mode="symmetric")
        return x_padded, orig_len

    @staticmethod
    def _estimate_noise_sigma(finest_detail: np.ndarray) -> np.ndarray:
        """Robust noise sigma estimate via MAD of finest detail coefficients.

        sigma = median(|d|) / 0.6745

        Computed **per-window** (along time axis) — no cross-window information.

        Args:
            finest_detail: (N, T) — finest level detail coefficients.

        Returns:
            sigma: (N,) — per-window noise standard deviation.
        """
        return np.median(np.abs(finest_detail), axis=-1) / 0.6745

    def _compute_threshold(
        self,
        detail_coeffs: np.ndarray,
        sigma_noise: np.ndarray,
        signal_len: int,
    ) -> np.ndarray:
        """Compute per-window denoising threshold for one detail level.

        BayesShrink:
            sigma_signal^2 = max(var(detail) - sigma_noise^2, 0)
            threshold = sigma_noise^2 / sigma_signal   (if sigma_signal > 0)
                      = max(|detail|)                   (if sigma_signal = 0, kill all)

        Universal (VisuShrink):
            threshold = sigma_noise * sqrt(2 * log(n))

        Args:
            detail_coeffs: (N, T) detail coefficients at one level.
            sigma_noise: (N,) per-window noise estimate.
            signal_len: padded signal length (for universal threshold).

        Returns:
            threshold: (N,) per-window threshold.
        """
        if self.threshold_rule == "bayes":
            var_detail = np.var(detail_coeffs, axis=-1)          # (N,)
            sigma_noise_sq = sigma_noise ** 2                     # (N,)
            sigma_signal_sq = np.maximum(var_detail - sigma_noise_sq, 0.0)

            # Where sigma_signal ~ 0 → all noise → kill all coefficients
            max_abs = np.max(np.abs(detail_coeffs), axis=-1)      # (N,)
            threshold = np.where(
                sigma_signal_sq > 1e-10,
                sigma_noise_sq / np.sqrt(np.maximum(sigma_signal_sq, 1e-10)),
                max_abs,
            )
            return threshold  # (N,)
        else:
            # Universal (VisuShrink)
            return sigma_noise * np.sqrt(2.0 * np.log(max(signal_len, 2)))

    def _apply_threshold(
        self, x: np.ndarray, threshold: np.ndarray
    ) -> np.ndarray:
        """Apply soft or hard thresholding.

        Args:
            x: (N, T) coefficients.
            threshold: (N,) per-window thresholds.

        Returns:
            (N, T) thresholded coefficients.
        """
        t = threshold[:, np.newaxis]  # (N, 1) for broadcasting
        if self.threshold_mode == "soft":
            return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
        else:
            return x * (np.abs(x) > t)

    # ----- main processing -----

    def _denoise_batch(self, x: np.ndarray) -> np.ndarray:
        """Denoise a batch of single-channel signals.

        All thresholds are computed **per-window** from each window's own
        coefficients. No information is shared between windows.

        Args:
            x: (N, T) batch of 1D signals (float64 recommended).

        Returns:
            (N, T) denoised signals (original length, before padding).
        """
        if x.shape[0] == 0:
            return x

        x_padded, orig_len = self._pad_signal(x)
        padded_len = x_padded.shape[-1]

        # SWT decomposition: [(cA_n, cD_n), ..., (cA_1, cD_1)]
        # cA = approximation, cD = detail; all arrays have shape (N, T_padded)
        coeffs = pywt.swt(x_padded, self.wavelet, level=self.level, axis=-1)

        # Noise estimate from finest detail coefficients (cD_1 = last tuple)
        finest_detail = coeffs[-1][1]                       # (N, T_padded)
        sigma_noise = self._estimate_noise_sigma(finest_detail)  # (N,)

        # Threshold detail coefficients at every level
        denoised_coeffs = []
        for cA, cD in coeffs:
            threshold = self._compute_threshold(cD, sigma_noise, padded_len)
            cD_denoised = self._apply_threshold(cD, threshold)
            # Keep approximation coefficients untouched (low-freq signal content)
            denoised_coeffs.append((cA, cD_denoised))

        # Inverse SWT reconstruction
        x_reconstructed = pywt.iswt(denoised_coeffs, self.wavelet, axis=-1)

        return x_reconstructed[:, :orig_len]

    def denoise_channel(self, x: np.ndarray) -> np.ndarray:
        """Denoise a batch of single-channel signals, with memory-safe batching.

        Args:
            x: (N, T) batch of 1D signals.

        Returns:
            (N, T) denoised signals.
        """
        N = x.shape[0]
        if N <= self._MAX_BATCH:
            return self._denoise_batch(x)

        # Process in chunks to avoid excessive memory usage
        result = np.empty_like(x)
        for start in range(0, N, self._MAX_BATCH):
            end = min(start + self._MAX_BATCH, N)
            result[start:end] = self._denoise_batch(x[start:end])
        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Denoise multi-channel EMG windows.

        Processing is per-window, per-channel — fully LOSO-compliant.
        No cross-window, cross-channel, or cross-subject information is used.

        Args:
            X: (N, T, C) raw EMG windows.

        Returns:
            X_denoised: (N, T, C) denoised EMG windows.
        """
        N, T, C = X.shape
        X_denoised = np.empty((N, T, C), dtype=np.float64)

        for c in range(C):
            channel_data = X[:, :, c].astype(np.float64)
            X_denoised[:, :, c] = self.denoise_channel(channel_data)

        return X_denoised.astype(np.float32)


# =========================================================================
#  Combined Feature Extractor: TIDWT Denoise → PowerfulFeatures
# =========================================================================

class TIDWTDenoisedFeatureExtractor:
    """
    Wraps TIDWT denoising + PowerfulFeatureExtractor into a single
    feature extractor compatible with FeatureMLTrainer injection.

    Pipeline:
        raw EMG (N, T, C) → SWT denoise (N, T, C) → PowerfulFeatures (N, F)

    LOSO compliance:
        - Denoising: per-window, per-channel, no learned parameters
        - Feature extraction: per-window, stateless
        - No cross-window or cross-subject information at any stage
    """

    def __init__(self, denoiser: TIDWTDenoiser, feature_extractor):
        self.denoiser = denoiser
        self.feature_extractor = feature_extractor

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (N, T, C) raw EMG windows.

        Returns:
            features: (N, F) feature vectors from denoised windows.
        """
        X_denoised = self.denoiser.transform(X)
        return self.feature_extractor.transform(X_denoised)


# =========================================================================
#  Utilities
# =========================================================================

def make_json_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =========================================================================
#  LOSO fold runner
# =========================================================================

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
    feature_extractor: TIDWTDenoisedFeatureExtractor,
) -> Dict:
    """Run one LOSO fold with TIDWT-denoised feature extractor."""
    from config.cross_subject import CrossSubjectConfig
    from data.multi_subject_loader import MultiSubjectLoader
    from evaluation.cross_subject import CrossSubjectExperiment
    from training.trainer import FeatureMLTrainer
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
        use_gpu=True,
        use_improved_processing=True,
    )

    base_viz = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    trainer = FeatureMLTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )
    # Inject combined TIDWT-denoised feature extractor.
    # FeatureMLTrainer.fit() checks `if self.feature_extractor is None` —
    # since we set it here, it skips creating its own and uses ours.
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
    test_acc = test_metrics.get("accuracy")
    test_f1 = test_metrics.get("f1_macro")

    if test_acc is not None:
        print(
            f"[LOSO] Test subject {test_subject} | Model: {model_type} | "
            f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
        )
    else:
        print(
            f"[LOSO] Test subject {test_subject} | Model: {model_type} | "
            f"Accuracy=None, F1-macro=None"
        )

    results_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": model_type,
        "approach": "tidwt_denoising_powerful_svm",
        "exercises": exercises,
        "metrics": {
            "test_accuracy": float(test_acc) if test_acc is not None else None,
            "test_f1_macro": float(test_f1) if test_f1 is not None else None,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "test_accuracy": float(test_acc) if test_acc is not None else None,
        "test_f1_macro": float(test_f1) if test_f1 is not None else None,
    }


# =========================================================================
#  Main
# =========================================================================

def main():
    EXPERIMENT_NAME = "exp_87_tidwt_denoising_preprocessing_loso"
    BASE_DIR = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()

    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES = ["E1"]
    MODEL_TYPES = ["svm_rbf", "svm_linear"]

    from config.base import ProcessingConfig, SplitConfig, TrainingConfig
    from processing.powerful_features import PowerfulFeatureExtractor
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

    # ---- Build TIDWT denoiser ----
    # db4: Daubechies-4 wavelet — standard for EMG, good time-frequency localization
    # level=4: decomposes 0-1000 Hz into 5 frequency bands at 2kHz sampling rate
    # BayesShrink: adaptive per-level thresholding, gentle on signal-rich bands
    # soft: smooth thresholding, avoids discontinuities in reconstructed signal
    denoiser = TIDWTDenoiser(
        wavelet="db4",
        level=4,
        threshold_mode="soft",
        threshold_rule="bayes",
    )

    # ---- Build PowerfulFeatureExtractor ----
    powerful_extractor = PowerfulFeatureExtractor(
        sampling_rate=proc_cfg.sampling_rate,
        logger=global_logger,
        feature_set="powerful",
        n_jobs=-1,
        use_torch=True,
        device="cuda",
        gpu_batch_size=4096,
    )

    # ---- Combined: TIDWT denoise → PowerfulFeatures ----
    feature_extractor = TIDWTDenoisedFeatureExtractor(
        denoiser=denoiser,
        feature_extractor=powerful_extractor,
    )

    print(f"[{EXPERIMENT_NAME}] TIDWT Denoising + PowerfulFeatures Pipeline:")
    print(f"  Wavelet: {denoiser.wavelet}")
    print(f"  Decomposition level: {denoiser.level}")
    print(f"  Threshold rule: {denoiser.threshold_rule}")
    print(f"  Threshold mode: {denoiser.threshold_mode}")
    print(f"  Frequency bands at {proc_cfg.sampling_rate}Hz:")
    for lv in range(1, denoiser.level + 1):
        lo = proc_cfg.sampling_rate / (2 ** (lv + 1))
        hi = proc_cfg.sampling_rate / (2 ** lv)
        print(f"    Level {lv} detail: {lo:.0f}-{hi:.0f} Hz")
    print(f"    Approximation: 0-{proc_cfg.sampling_rate / (2 ** (denoiser.level + 1)):.0f} Hz")
    print(f"  Subjects: {ALL_SUBJECTS}")

    all_loso_results = []

    for model_type in MODEL_TYPES:
        print(f"\n{'=' * 60}")
        print(f"ML MODEL: {model_type} -- starting LOSO")
        print(f"{'=' * 60}")

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
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                feature_extractor=feature_extractor,
            )
            all_loso_results.append(fold_res)

    # ---- Aggregate results ----
    aggregate_results = {}
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
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
            "per_subject": model_results,
        }

    # ---- Print summary ----
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {EXPERIMENT_NAME}")
    print(f"{'=' * 60}")
    for model_type, res in aggregate_results.items():
        acc_m = res["mean_accuracy"]
        acc_s = res["std_accuracy"]
        f1_m = res["mean_f1_macro"]
        f1_s = res["std_f1_macro"]
        print(f"  {model_type}: Acc={acc_m:.4f} +/- {acc_s:.4f}, F1={f1_m:.4f} +/- {f1_s:.4f}")

    # ---- Save summary JSON ----
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "SWT (shift-invariant undecimated DWT) denoising of raw EMG windows "
            "before feature extraction improves cross-subject classification "
            "by removing noise while preserving gesture-discriminative signal structure."
        ),
        "source": "Zeng et al. (IEEE Access 2021)",
        "approach": "TIDWT_denoising + PowerfulFeatures + SVM",
        "denoising_params": {
            "wavelet": denoiser.wavelet,
            "level": denoiser.level,
            "threshold_rule": denoiser.threshold_rule,
            "threshold_mode": denoiser.threshold_mode,
        },
        "loso_compliance_notes": [
            "Denoising: per-window, per-channel, no cross-window info",
            "Noise sigma: MAD of each window's own finest detail coefficients",
            "BayesShrink threshold: computed per-window from that window only",
            "Feature standardization: training data statistics only",
            "No subject-specific adaptation",
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
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)
    print(f"\n[DONE] {EXPERIMENT_NAME} -> {summary_path}")


if __name__ == "__main__":
    main()
