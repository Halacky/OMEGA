# FILE: experiments/exp_49_emg_cepstral_coefficients_with_muscle_filterbanks_loso.py
"""
Experiment 49: EMG Cepstral Coefficients (EMGCC) with Muscle-Specific Filterbanks

Hypothesis: Speech-inspired cepstral features (log + DCT deconvolution) with
EMG-physiology filterbanks will produce inherently subject-invariant features.
CMVN normalizes electrode/channel effects (like microphone normalization in speech).

Key innovations:
- Muscle-specific triangular filterbanks (20-500Hz, non-linear spacing)
- Cepstral processing: STFT -> filterbank energies -> log -> DCT -> EMGCC
- Delta + Delta-Delta temporal derivatives
- CMVN (Cepstral Mean-Variance Normalization) per channel
- Combined with PowerfulFeatureExtractor features
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.fftpack import dct

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    DEFAULT_SUBJECTS,
    CI_TEST_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import FeatureMLTrainer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from processing.powerful_features import PowerfulFeatureExtractor


EXPERIMENT_NAME = "exp_49_emg_cepstral_coefficients_muscle_filterbanks_loso"
HYPOTHESIS_ID = "h-049-emgcc-cepstral"


# ─── EMGCC Feature Extraction ───────────────────────────────────────────────

def _build_muscle_filterbank(n_filters: int, n_fft: int, sample_rate: int,
                              low_freq: float = 20.0, high_freq: float = 500.0) -> np.ndarray:
    """
    Build triangular filterbank with non-linear (log-like) spacing for EMG.
    Emphasizes low-frequency motor unit firing rates.
    """
    # Non-linear spacing: denser at low frequencies (motor unit territory)
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    # Convert to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        # Rising slope
        if center > left:
            filterbank[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        # Falling slope
        if right > center:
            filterbank[i, center:right] = np.linspace(1, 0, right - center, endpoint=False)
        # Ensure at least center bin is 1
        if center < filterbank.shape[1]:
            filterbank[i, center] = 1.0

    return filterbank


def _compute_emgcc_single_channel(channel_data: np.ndarray, sample_rate: int,
                                    n_fft: int = 256, hop_length: int = 64,
                                    n_filters: int = 26, n_ceps: int = 13,
                                    filterbank: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute EMGCC for a single channel. Returns (n_ceps,) mean cepstral vector."""
    # STFT
    f, t, Zxx = scipy_signal.stft(channel_data, fs=sample_rate, nperseg=n_fft,
                                    noverlap=n_fft - hop_length, window='hann')
    power_spectrum = np.abs(Zxx) ** 2  # (n_freq, n_frames)

    if filterbank is None:
        filterbank = _build_muscle_filterbank(n_filters, n_fft, sample_rate)

    # Apply filterbank
    n_freq = min(filterbank.shape[1], power_spectrum.shape[0])
    filter_energies = filterbank[:, :n_freq] @ power_spectrum[:n_freq, :]  # (n_filters, n_frames)
    filter_energies = np.maximum(filter_energies, 1e-10)

    # Log energies
    log_energies = np.log(filter_energies)  # (n_filters, n_frames)

    # DCT -> cepstral coefficients (per frame)
    cepstral = dct(log_energies, type=2, axis=0, norm='ortho')[:n_ceps, :]  # (n_ceps, n_frames)

    # Mean over frames (summary per window)
    return cepstral.mean(axis=1)  # (n_ceps,)


def _compute_deltas(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Compute delta (1st derivative) of feature matrix along axis 0."""
    # features shape: (n_samples, n_features)
    n = features.shape[0]
    if n < 2 * width + 1:
        return np.zeros_like(features)
    padded = np.pad(features, ((width, width), (0, 0)), mode='edge')
    deltas = np.zeros_like(features)
    denom = 2 * sum(i ** 2 for i in range(1, width + 1))
    if denom == 0:
        return deltas
    for t_idx in range(n):
        for i in range(1, width + 1):
            deltas[t_idx] += i * (padded[t_idx + width + i] - padded[t_idx + width - i])
        deltas[t_idx] /= denom
    return deltas


def extract_emgcc_features(windows: np.ndarray, sample_rate: int = 2000,
                            n_filters: int = 26, n_ceps: int = 13) -> np.ndarray:
    """
    Extract EMGCC features for a batch of EMG windows.

    Args:
        windows: (N, T, C) EMG windows
        sample_rate: sampling rate in Hz

    Returns:
        features: (N, n_features) where n_features = n_ceps * 3 * C
    """
    N, T, C = windows.shape
    n_fft = min(256, T)
    filterbank = _build_muscle_filterbank(n_filters, n_fft, sample_rate)

    all_features = []
    for i in range(N):
        window_feats = []
        for ch in range(C):
            ceps = _compute_emgcc_single_channel(
                windows[i, :, ch], sample_rate, n_fft=n_fft,
                hop_length=max(n_fft // 4, 1), n_filters=n_filters,
                n_ceps=n_ceps, filterbank=filterbank,
            )
            window_feats.append(ceps)
        all_features.append(np.concatenate(window_feats))  # (n_ceps * C,)

    features = np.array(all_features)  # (N, n_ceps * C)

    # Compute deltas and delta-deltas over the sample dimension
    deltas = _compute_deltas(features)
    delta_deltas = _compute_deltas(deltas)

    # Concatenate: static + delta + delta-delta
    result = np.concatenate([features, deltas, delta_deltas], axis=1)  # (N, n_ceps * C * 3)

    # CMVN (Cepstral Mean-Variance Normalization)
    mean = result.mean(axis=0, keepdims=True)
    std = result.std(axis=0, keepdims=True) + 1e-8
    result = (result - mean) / std

    return result.astype(np.float32)


# ─── Custom Trainer ──────────────────────────────────────────────────────────

class EMGCCFeatureMLTrainer(FeatureMLTrainer):
    """FeatureMLTrainer extended with EMGCC features."""

    def __init__(self, *args, sample_rate: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """Override fit to inject EMGCC features before standard ML pipeline."""
        # Convert splits to arrays using parent's method
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        self.class_ids = class_ids
        self.class_names = class_names

        self.logger.info(f"Extracting EMGCC features from windows {X_train.shape}...")

        # Extract EMGCC features  (windows are (N, T, C))
        emgcc_train = extract_emgcc_features(X_train, self.sample_rate)
        emgcc_val = extract_emgcc_features(X_val, self.sample_rate) if len(X_val) > 0 else np.empty((0, emgcc_train.shape[1]))
        emgcc_test = extract_emgcc_features(X_test, self.sample_rate) if len(X_test) > 0 else np.empty((0, emgcc_train.shape[1]))

        self.logger.info(f"EMGCC features: {emgcc_train.shape[1]} dims")

        # Also extract PowerfulFeatures
        pfe = PowerfulFeatureExtractor(sampling_rate=self.sample_rate)
        pf_train = pfe.transform(X_train)
        pf_val = pfe.transform(X_val) if len(X_val) > 0 else np.empty((0, pf_train.shape[1]))
        pf_test = pfe.transform(X_test) if len(X_test) > 0 else np.empty((0, pf_train.shape[1]))

        self.logger.info(f"Powerful features: {pf_train.shape[1]} dims")

        # Combine
        F_train = np.concatenate([emgcc_train, pf_train], axis=1)
        F_val = np.concatenate([emgcc_val, pf_val], axis=1) if len(X_val) > 0 else np.empty((0, F_train.shape[1]))
        F_test = np.concatenate([emgcc_test, pf_test], axis=1) if len(X_test) > 0 else np.empty((0, F_train.shape[1]))

        self.logger.info(f"Combined features: {F_train.shape[1]} dims")

        # Normalize
        self.feature_mean = F_train.mean(axis=0)
        self.feature_std = F_train.std(axis=0) + 1e-8
        F_train = (F_train - self.feature_mean) / self.feature_std
        F_val = (F_val - self.feature_mean) / self.feature_std if len(F_val) > 0 else F_val
        F_test = (F_test - self.feature_mean) / self.feature_std if len(F_test) > 0 else F_test

        # Feature selection
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(F_train, y_train, random_state=42, n_neighbors=5)
        top_k = min(200, F_train.shape[1])
        self.selected_feature_indices = np.argsort(mi)[::-1][:top_k]

        F_train = F_train[:, self.selected_feature_indices]
        F_val = F_val[:, self.selected_feature_indices] if len(F_val) > 0 else F_val
        F_test = F_test[:, self.selected_feature_indices] if len(F_test) > 0 else F_test

        self.logger.info(f"Selected top {top_k} features by MI")

        # Train SVM-RBF with GridSearch
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.001],
        }
        svm_base = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        grid = GridSearchCV(svm_base, param_grid, cv=3, scoring='accuracy',
                            n_jobs=-1, refit=True)
        grid.fit(F_train, y_train)

        self.model = grid.best_estimator_
        self.logger.info(f"Best SVM params: {grid.best_params_}, CV acc: {grid.best_score_:.4f}")

        # Store for evaluate_numpy
        self._feature_extractor_fn = lambda X: self._extract_and_select(X)
        self.pfe = pfe

        val_acc = self.model.score(F_val, y_val) if len(F_val) > 0 else 0.0
        return {"best_val_accuracy": val_acc, "best_params": grid.best_params_}

    def _extract_and_select(self, X: np.ndarray) -> np.ndarray:
        """Extract EMGCC + powerful features, normalize, select."""
        emgcc = extract_emgcc_features(X, self.sample_rate)
        pf = self.pfe.transform(X)
        F = np.concatenate([emgcc, pf], axis=1)
        F = (F - self.feature_mean) / self.feature_std
        F = F[:, self.selected_feature_indices]
        return F

    def evaluate_numpy(self, X: np.ndarray, y: np.ndarray,
                       split_name: str, visualize: bool = True) -> Dict:
        """Evaluate on numpy arrays."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        F = self._extract_and_select(X)
        preds = self.model.predict(F)
        accuracy = (preds == y).mean()

        from sklearn.metrics import f1_score, classification_report, confusion_matrix
        f1_macro = f1_score(y, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(y, preds, average='weighted', zero_division=0)
        report = classification_report(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds)

        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'report': report,
            'confusion_matrix': cm.tolist(),
        }

        if visualize and self.visualizer:
            try:
                self.visualizer.plot_confusion_matrix(
                    cm, class_names=[str(i) for i in range(len(np.unique(y)))],
                    title=f"CM - {split_name}",
                    save_path=self.output_dir / f"cm_{split_name}.png",
                )
            except Exception:
                pass

        return metrics


# ─── LOSO fold ───────────────────────────────────────────────────────────────

def run_single_loso_fold(
    base_dir: Path, output_dir: Path,
    train_subjects: List[str], test_subject: str,
    exercises: List[str], model_type: str,
    proc_cfg: ProcessingConfig, split_cfg: SplitConfig, train_cfg: TrainingConfig,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects, test_subject=test_subject,
        exercises=exercises, base_dir=base_dir,
        pool_train_subjects=True, use_separate_val_subject=False,
        val_subject=None, val_ratio=0.15, seed=train_cfg.seed, max_gestures=10,
    )
    cs_cfg.save(output_dir / "cross_subject_config.json")

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg, logger=logger,
        use_gpu=True, use_improved_processing=False,
    )
    base_viz = Visualizer(output_dir, logger)

    trainer = EMGCCFeatureMLTrainer(
        train_cfg=train_cfg, logger=logger,
        output_dir=output_dir, visualizer=base_viz,
        sample_rate=proc_cfg.sampling_rate,
    )

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg, split_config=split_cfg,
        multi_subject_loader=multi_loader, trainer=trainer,
        visualizer=base_viz, logger=logger,
    )

    try:
        results = experiment.run()
    except Exception as e:
        logger.error(f"Error in LOSO fold (test_subject={test_subject}): {e}")
        traceback.print_exc()
        return {"test_subject": test_subject, "model_type": model_type,
                "test_accuracy": None, "test_f1_macro": None, "error": str(e)}

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(f"[LOSO] Test subject {test_subject} | Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}")

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(make_json_serializable({
        "test_subject": test_subject, "train_subjects": train_subjects,
        "model_type": model_type, "exercises": exercises,
        "metrics": {"test_accuracy": test_acc, "test_f1_macro": test_f1},
    }), filename="fold_metadata.json")

    import gc
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {"test_subject": test_subject, "model_type": model_type,
            "test_accuracy": test_acc, "test_f1_macro": test_f1}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    ALL_SUBJECTS = parse_subjects_args()
    EXERCISES = ["E1"]
    MODEL_TYPES = ["svm_rbf"]

    proc_cfg = ProcessingConfig(
        window_size=600, window_overlap=300, num_channels=8,
        sampling_rate=2000, segment_edge_margin=0.1,
    )
    split_cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        mode="by_segments", shuffle_segments=True, seed=42,
        include_rest_in_splits=False,
    )
    train_cfg = TrainingConfig(
        batch_size=512, epochs=1, learning_rate=1e-3, weight_decay=1e-4,
        dropout=0.3, early_stopping_patience=10, use_class_weights=True,
        seed=42, num_workers=4,
        device="cuda" if __import__('torch').cuda.is_available() else "cpu",
        pipeline_type="ml_emg_td", model_type="svm_rbf",
        ml_model_type="svm_rbf", ml_use_hyperparam_search=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Features: EMGCC + PowerfulFeatures -> SVM-RBF")
    print(f"Subjects: {len(ALL_SUBJECTS)}")

    all_loso_results = []

    for model_type in MODEL_TYPES:
        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"

            try:
                fold_res = run_single_loso_fold(
                    base_dir=BASE_DIR, output_dir=fold_output_dir,
                    train_subjects=train_subjects, test_subject=test_subject,
                    exercises=EXERCISES, model_type=model_type,
                    proc_cfg=proc_cfg, split_cfg=split_cfg, train_cfg=train_cfg,
                )
                all_loso_results.append(fold_res)
                acc_str = f"{fold_res['test_accuracy']:.4f}" if fold_res.get('test_accuracy') is not None else "N/A"
                f1_str = f"{fold_res['test_f1_macro']:.4f}" if fold_res.get('test_f1_macro') is not None else "N/A"
                print(f"  {test_subject}: acc={acc_str}, f1={f1_str}")
            except Exception as e:
                global_logger.error(f"Failed {test_subject} {model_type}: {e}")
                traceback.print_exc()
                all_loso_results.append({
                    "test_subject": test_subject, "model_type": model_type,
                    "test_accuracy": None, "test_f1_macro": None, "error": str(e),
                })

    # Aggregate
    aggregate = {}
    for model_type in MODEL_TYPES:
        model_results = [r for r in all_loso_results
                         if r["model_type"] == model_type and r.get("test_accuracy") is not None]
        if not model_results:
            continue
        accs = [r["test_accuracy"] for r in model_results]
        f1s = [r["test_f1_macro"] for r in model_results]
        aggregate[model_type] = {
            "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)), "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
        }
        print(f"\n{model_type}: Acc={aggregate[model_type]['mean_accuracy']:.4f}+-{aggregate[model_type]['std_accuracy']:.4f}, "
              f"F1={aggregate[model_type]['mean_f1_macro']:.4f}+-{aggregate[model_type]['std_f1_macro']:.4f}")

    # Save
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME, "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "emgcc+powerful", "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS, "exercises": EXERCISES,
        "processing_config": asdict(proc_cfg), "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate, "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")

    # Qdrant callback
    try:
        from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed
        if aggregate:
            best_model = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
            best_metrics = aggregate[best_model].copy()
            best_metrics["best_model"] = best_model
            mark_hypothesis_verified(hypothesis_id=HYPOTHESIS_ID, metrics=best_metrics,
                                     experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed(hypothesis_id=HYPOTHESIS_ID,
                                   error_message="No successful LOSO folds completed")
    except ImportError:
        print("hypothesis_executor not available, skipping Qdrant update")


if __name__ == "__main__":
    main()
