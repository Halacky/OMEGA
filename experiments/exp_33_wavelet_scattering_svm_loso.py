"""
Experiment 33: Wavelet Scattering Transform for Cross-Subject EMG Classification

Hypothesis:
    1D wavelet scattering transform provides mathematically built-in invariance
    to time-warping and scale deformations. In biosignals and audio, scattering
    representations often outperform CNNs, especially in low-data regimes.

Approach:
    1. Compute 1D wavelet scattering coefficients per EMG channel (via kymatio)
    2. Apply log-normalization + global temporal mean pooling
    3. Concatenate across channels → fixed-size feature vector
    4. Classify with SVM (RBF + Linear)

Why:
    Time-warp and scale invariance are built in mathematically — no need
    to learn them from data. This is especially valuable for cross-subject
    generalization where inter-subject variability includes amplitude
    scaling and temporal stretching of muscle activation patterns.
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
#  Wavelet Scattering Feature Extractor
# =========================================================================

class ScatteringFeatureExtractor:
    """
    Extracts wavelet scattering features from multi-channel EMG windows.

    For each channel independently:
      1. Zero-pad signal to next power of 2
      2. Compute 1D wavelet scattering (orders 0, 1, 2) via kymatio
      3. Apply log(|x| + eps) for numerical stability
      4. Global temporal mean pooling → fixed-size vector

    Features from all channels are concatenated into a single vector.

    The scattering transform produces representations that are:
      - Locally translation-invariant (up to scale 2^J)
      - Stable to small deformations (time-warping)
      - Informative (preserves discriminative signal structure)
    """

    def __init__(
        self,
        sampling_rate: int = 2000,
        J: int = 6,
        Q: int = 8,
        signal_length: int = 600,
    ):
        self.sampling_rate = sampling_rate
        self.J = J
        self.Q = Q
        self.signal_length = signal_length
        self.T_padded = 2 ** int(np.ceil(np.log2(signal_length)))

        from kymatio.torch import Scattering1D
        self.scattering = Scattering1D(J=J, shape=self.T_padded, Q=Q)

        # Compute number of scattering coefficients with a dummy forward pass
        dummy = torch.zeros(1, self.T_padded)
        dummy_out = self.scattering(dummy)  # (1, n_paths, T')
        self.n_coeffs = dummy_out.shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract scattering features from multi-channel EMG windows.

        Args:
            X: (N, T, C) — EMG windows (N windows, T time steps, C channels)

        Returns:
            features: (N, C * n_coeffs) — scattering features
        """
        N, T, C = X.shape
        eps = 1e-6

        # Zero-pad time dimension to T_padded
        if T < self.T_padded:
            pad_width = ((0, 0), (0, self.T_padded - T), (0, 0))
            X_padded = np.pad(X, pad_width, mode="constant", constant_values=0.0)
        else:
            X_padded = X[:, :self.T_padded, :]

        all_channel_features = []

        for c in range(C):
            # Extract single channel: (N, T_padded)
            channel_data = X_padded[:, :, c]

            # Convert to torch tensor for kymatio
            channel_tensor = torch.from_numpy(channel_data.astype(np.float32))

            # Compute scattering: (N, n_paths, T')
            with torch.no_grad():
                coeffs = self.scattering(channel_tensor)

            # Convert back to numpy
            coeffs_np = coeffs.numpy()  # (N, n_paths, T')

            # Log-normalization for numerical stability
            coeffs_log = np.log(np.abs(coeffs_np) + eps)

            # Global temporal mean pooling: (N, n_paths)
            channel_features = np.mean(coeffs_log, axis=-1)

            all_channel_features.append(channel_features)

        # Concatenate across channels: (N, C * n_paths)
        features = np.concatenate(all_channel_features, axis=1)
        return features.astype(np.float32)

    def get_feature_names(self, n_channels: int = 8) -> List[str]:
        """Human-readable feature names (for analysis / debugging)."""
        names = []
        for c in range(n_channels):
            for p in range(self.n_coeffs):
                names.append(f"scat_ch{c}_path{p}")
        return names


# =========================================================================
#  LOSO fold runner
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
    feature_extractor: ScatteringFeatureExtractor,
) -> Dict:
    """Run one LOSO fold with wavelet scattering feature extractor."""
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
    # Inject custom scattering feature extractor — FeatureMLTrainer.fit()
    # will skip creating its own extractor and use ours instead.
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
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test subject {test_subject} | Model: {model_type} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    results_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": model_type,
        "approach": "wavelet_scattering",
        "exercises": exercises,
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
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
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# =========================================================================
#  Main
# =========================================================================

def main():
    EXPERIMENT_NAME = "exp_33_wavelet_scattering_svm_loso"
    BASE_DIR = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()

    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES = ["E1"]
    MODEL_TYPES = ["svm_rbf", "svm_linear"]

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

    # Create wavelet scattering feature extractor (stateless, reused across folds)
    # J=6: max averaging scale = 2^6 = 64 samples = 32ms @ 2kHz
    # Q=8: 8 wavelets per octave — good frequency resolution for EMG
    feature_extractor = ScatteringFeatureExtractor(
        sampling_rate=proc_cfg.sampling_rate,
        J=6,
        Q=8,
        signal_length=proc_cfg.window_size,
    )

    total_features = feature_extractor.n_coeffs * proc_cfg.num_channels
    print(f"[{EXPERIMENT_NAME}] Wavelet Scattering Features:")
    print(f"  J={feature_extractor.J}, Q={feature_extractor.Q}")
    print(f"  Signal length: {proc_cfg.window_size} -> padded to {feature_extractor.T_padded}")
    print(f"  Scattering coefficients per channel: {feature_extractor.n_coeffs}")
    print(f"  Total features ({proc_cfg.num_channels} channels): {total_features}")
    print(f"  Subjects: {ALL_SUBJECTS}")

    all_loso_results = []

    for model_type in MODEL_TYPES:
        print(f"\n{'=' * 60}")
        print(f"ML MODEL: {model_type} — starting LOSO")
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
            "1D wavelet scattering transform provides built-in invariance "
            "to time-warping and scale deformations, improving cross-subject "
            "EMG classification without learned features"
        ),
        "approach": "wavelet_scattering + SVM",
        "feature_extractor": "ScatteringFeatureExtractor (kymatio)",
        "scattering_params": {
            "J": feature_extractor.J,
            "Q": feature_extractor.Q,
            "signal_length": feature_extractor.signal_length,
            "T_padded": feature_extractor.T_padded,
            "n_coeffs_per_channel": feature_extractor.n_coeffs,
            "total_features": total_features,
        },
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
