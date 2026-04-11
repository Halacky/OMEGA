"""
Experiment 32: Spectral Band Power Ratio Features for Cross-Subject EMG Classification

Hypothesis:
    Absolute spectral power varies across subjects, but ratios between
    frequency bands are more stable (analogous to EEG band-power ratios).

Approach:
    1. Split EMG spectrum into 6 frequency bands (20-1000 Hz)
    2. Compute per-channel:
       - Band powers (P_i)
       - Relative band powers (P_i / P_total)
       - All pairwise ratios (P_i / P_j)
       - All pairwise log-ratios log(P_i / P_j)
    3. Use these ratio-based features with SVM (RBF + Linear)

Why:
    Ratios suppress amplitude variability between subjects,
    potentially improving cross-subject generalization.
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
#  Spectral Band Ratio Feature Extractor
# =========================================================================

class SpectralBandRatioExtractor:
    """
    Extracts spectral band power ratio features from EMG windows.

    For each channel, computes:
      - Band powers in predefined frequency bands
      - Relative band powers  P_i / P_total
      - All pairwise ratios   P_i / P_j   (i < j)
      - All pairwise log-ratios  log(P_i / P_j)

    These ratio-based features are designed to be invariant to
    absolute amplitude differences between subjects.
    """

    # EMG frequency bands (Hz) — 6 bands covering 20–1000 Hz
    DEFAULT_BANDS = [
        (20, 60),     # Low-frequency motor unit activity
        (60, 120),    # Mid-low frequency
        (120, 250),   # Main EMG energy band
        (250, 500),   # High frequency
        (500, 750),   # Very high frequency
        (750, 1000),  # Near Nyquist for 2 kHz sampling
    ]

    def __init__(self, sampling_rate: int = 2000, bands=None):
        self.sampling_rate = sampling_rate
        self.bands = bands or self.DEFAULT_BANDS
        self.n_bands = len(self.bands)
        self.n_pairs = self.n_bands * (self.n_bands - 1) // 2

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract spectral band ratio features.

        Args:
            X: (N, T, C) — EMG windows

        Returns:
            features: (N, F) where
                F = C * (n_bands + n_bands + n_pairs + n_pairs)
                  = C * (2*n_bands + 2*n_pairs)
                With 6 bands and 8 channels: F = 8*(12 + 30) = 336
        """
        N, T, C = X.shape
        eps = 1e-12

        # Vectorised FFT across all windows and channels
        fft_vals = np.fft.rfft(X, axis=1)                              # (N, T//2+1, C)
        fft_freqs = np.fft.rfftfreq(T, d=1.0 / self.sampling_rate)    # (T//2+1,)
        psd = np.abs(fft_vals) ** 2 / T                                # (N, T//2+1, C)

        # Band powers — one (N, C) array per band
        band_powers_list = []
        for f_low, f_high in self.bands:
            mask = (fft_freqs >= f_low) & (fft_freqs < f_high)
            bp = psd[:, mask, :].sum(axis=1)                           # (N, C)
            band_powers_list.append(bp)

        band_powers = np.stack(band_powers_list, axis=0)               # (n_bands, N, C)
        total_power = band_powers.sum(axis=0) + eps                    # (N, C)
        rel_powers = band_powers / total_power[np.newaxis, :, :]       # (n_bands, N, C)

        all_features = []

        # 1) Raw band powers  →  (N, n_bands*C)
        all_features.append(
            band_powers.transpose(1, 0, 2).reshape(N, -1)
        )

        # 2) Relative band powers  →  (N, n_bands*C)
        all_features.append(
            rel_powers.transpose(1, 0, 2).reshape(N, -1)
        )

        # 3) Pairwise ratios + log-ratios
        ratios_list = []
        log_ratios_list = []
        for bi in range(self.n_bands):
            for bj in range(bi + 1, self.n_bands):
                ratio = (band_powers[bi] + eps) / (band_powers[bj] + eps)  # (N, C)
                ratios_list.append(ratio)
                log_ratios_list.append(np.log(ratio))

        if ratios_list:
            all_features.append(
                np.stack(ratios_list, axis=1).reshape(N, -1)
            )
            all_features.append(
                np.stack(log_ratios_list, axis=1).reshape(N, -1)
            )

        return np.concatenate(all_features, axis=1).astype(np.float32)

    def get_feature_names(self, n_channels: int = 8) -> List[str]:
        """Human-readable feature names (for analysis / debugging)."""
        names = []
        bnames = [f"{lo}-{hi}Hz" for lo, hi in self.bands]

        for b in bnames:
            for c in range(n_channels):
                names.append(f"BP_{b}_ch{c}")
        for b in bnames:
            for c in range(n_channels):
                names.append(f"RelP_{b}_ch{c}")
        for bi in range(self.n_bands):
            for bj in range(bi + 1, self.n_bands):
                for c in range(n_channels):
                    names.append(f"Ratio_{bnames[bi]}/{bnames[bj]}_ch{c}")
        for bi in range(self.n_bands):
            for bj in range(bi + 1, self.n_bands):
                for c in range(n_channels):
                    names.append(f"LogR_{bnames[bi]}/{bnames[bj]}_ch{c}")
        return names


# =========================================================================
#  LOSO fold runner (custom — pre-sets feature extractor on the trainer)
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
    feature_extractor: SpectralBandRatioExtractor,
) -> Dict:
    """Run one LOSO fold with the custom spectral-ratio feature extractor."""
    import torch
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
    # Inject the custom feature extractor so FeatureMLTrainer.fit() skips
    # its own extractor creation and uses ours instead.
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
        "approach": "spectral_band_ratio",
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
    EXPERIMENT_NAME = "exp_32_spectral_band_ratio_svm_loso"
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

    # Create the spectral band-ratio feature extractor (stateless, reused
    # across all folds).
    feature_extractor = SpectralBandRatioExtractor(sampling_rate=2000)

    n_feat_per_ch = 2 * feature_extractor.n_bands + 2 * feature_extractor.n_pairs
    total_features = n_feat_per_ch * proc_cfg.num_channels
    print(f"[{EXPERIMENT_NAME}] Spectral Band Ratio Features:")
    print(f"  Bands: {feature_extractor.bands}")
    print(f"  Features per channel: {n_feat_per_ch}")
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
            "Spectral band power ratios are more stable across subjects "
            "than absolute power"
        ),
        "approach": "spectral_band_ratio + SVM",
        "feature_extractor": "SpectralBandRatioExtractor",
        "bands": feature_extractor.bands,
        "n_features_per_channel": n_feat_per_ch,
        "total_features": total_features,
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
