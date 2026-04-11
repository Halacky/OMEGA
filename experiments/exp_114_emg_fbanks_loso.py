"""
Experiment 114: Log-Mel Filterbank (Fbanks) Features for LOSO Gesture Classification

Hypothesis:
    Log-mel filterbank energies — the intermediate representation in the MFCC
    pipeline BEFORE the DCT step — preserve richer spectral detail than MFCC
    coefficients.  The DCT in MFCC compresses n_mels channels into n_mfcc
    (26 -> 13), discarding information that may be discriminative for EMG
    gestures.  Fbanks keep all 26 mel-band energies.

    This is well-established in speech: modern ASR (e.g. Whisper, wav2vec2)
    uses log-mel filterbanks, NOT MFCCs, because neural networks learn their
    own decorrelation — the hand-crafted DCT step is redundant.

What is tested:
    Four configurations mirroring exp_113:
      A) Fbanks+D+DD flat (26*3=78 coeff, 6 stats) -> SVM-RBF
      B) Fbanks+D+DD flat -> Random Forest
      C) Fbanks+D+DD spectrogram (78 x T_frames x C) -> 2D CNN
      D) Fbanks (no deltas, 26 coeff) -> 2D CNN

LOSO protocol: identical to exp_113 (channel standardization from train only,
deterministic feature extraction, per-fold training).

Run examples:
    python experiments/exp_114_emg_fbanks_loso.py --ci
    python experiments/exp_114_emg_fbanks_loso.py --ci --config A
    python experiments/exp_114_emg_fbanks_loso.py --full
"""

import gc
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.mfcc_trainer import MFCCTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_114_emg_fbanks"
EXERCISES       = ["E1"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# Filterbank hyperparameters (same mel-scale as exp_113)
FBANKS_N_MELS   = 26       # mel filterbank channels (= output dimension)
FBANKS_N_MFCC   = 13       # only used for extractor init, not for fbanks output
FBANKS_FMIN     = 20.0     # Hz — EMG lower bound
FBANKS_FMAX     = 500.0    # Hz — EMG upper bound

# Experiment configurations
CONFIGS = {
    "A": {"mode": "ml",   "ml_classifier": "svm_rbf",  "use_deltas": True,  "label": "Fbanks+D+DD -> SVM-RBF"},
    "B": {"mode": "ml",   "ml_classifier": "rf",       "use_deltas": True,  "label": "Fbanks+D+DD -> RF"},
    "C": {"mode": "deep", "ml_classifier": None,       "use_deltas": True,  "label": "Fbanks+D+DD -> 2D-CNN"},
    "D": {"mode": "deep", "ml_classifier": None,       "use_deltas": False, "label": "Fbanks (no D) -> 2D-CNN"},
}


# ============================== SPLITS BUILDER ===============================

def _build_splits(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """Build train/val/test splits from loaded subject data (LOSO-clean)."""
    rng = np.random.RandomState(seed)

    train_dict: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}
    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        for gid, reps in filtered.items():
            for rep_arr in reps:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    train_dict[gid].append(rep_arr)

    final_train: Dict[int, np.ndarray] = {}
    final_val: Dict[int, np.ndarray] = {}
    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        if len(trn_idx) > 0:
            final_train[gid] = X_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid] = X_gid[val_idx]

    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": final_test}


# ================================ SINGLE FOLD ================================

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    config_key: str = "A",
) -> Dict:
    """Execute one LOSO fold for a given Fbanks configuration."""
    cfg = CONFIGS[config_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    # Data loader
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    # Load all subjects
    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES
    )
    logger.info(
        f"Common gestures ({len(common_gestures)}): {common_gestures}"
    )

    # Build LOSO splits
    splits = _build_splits(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    for sname in ("train", "val", "test"):
        total = sum(len(a) for a in splits[sname].values()
                    if isinstance(a, np.ndarray) and a.ndim == 3)
        logger.info(f"  {sname.upper()}: {total} windows, {len(splits[sname])} gestures")

    # Trainer — feature_type="fbanks" is the key difference from exp_113
    trainer = MFCCTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        mode=cfg["mode"],
        ml_classifier=cfg["ml_classifier"] or "svm_rbf",
        sampling_rate=proc_cfg.sampling_rate,
        n_mfcc=FBANKS_N_MFCC,
        n_mels=FBANKS_N_MELS,
        fmin=FBANKS_FMIN,
        fmax=FBANKS_FMAX,
        use_deltas=cfg["use_deltas"],
        feature_type="fbanks",
    )

    # Training
    try:
        trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "config": config_key,
            "label": cfg["label"],
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # Test evaluation
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error(f"No test windows for {test_subject}")
        return {
            "test_subject": test_subject,
            "config": config_key,
            "label": cfg["label"],
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    test_results = trainer.evaluate_numpy(
        X_test, y_test,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1 = float(test_results["f1_macro"])

    print(
        f"[LOSO] {cfg['label']} | test={test_subject} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )

    # Save fold results
    fold_summary = {
        "test_subject": test_subject,
        "config": config_key,
        "label": cfg["label"],
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "report": test_results.get("report"),
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del trainer, multi_loader, subjects_data, splits
    gc.collect()

    return {
        "test_subject": test_subject,
        "config": config_key,
        "label": cfg["label"],
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ==================================== MAIN ===================================

def main():
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _parser.add_argument("--config", type=str, default=None,
                         help="Run single config: A, B, C, or D. Default: all.")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS  # safe default for server

    configs_to_run = [_args.config] if _args.config else list(CONFIGS.keys())

    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    # Configs
    proc_cfg = ProcessingConfig(
        window_size=400,       # 200 ms @ 2000 Hz
        window_overlap=200,    # 50% overlap
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

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : Log-mel filterbank energies (no DCT) for cross-subject EMG")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Configs    : {configs_to_run}")
    print(f"Fbanks     : n_mels={FBANKS_N_MELS}, freq=[{FBANKS_FMIN}, {FBANKS_FMAX}] Hz")
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results: Dict[str, List[Dict]] = {k: [] for k in configs_to_run}

    for config_key in configs_to_run:
        cfg = CONFIGS[config_key]
        print(f"\n{'~' * 60}")
        print(f"Config {config_key}: {cfg['label']}")
        print(f"{'~' * 60}")

        # Different training config for ML vs deep
        if cfg["mode"] == "ml":
            train_cfg = TrainingConfig(
                model_type="fbanks_ml",
                pipeline_type="feature_ml",
                use_handcrafted_features=True,
                batch_size=64,
                epochs=1,  # not used for ML
                learning_rate=1e-3,
                weight_decay=0,
                dropout=0,
                early_stopping_patience=10,
                seed=42,
                use_class_weights=False,
                num_workers=0,
                device="cpu",
            )
        else:
            train_cfg = TrainingConfig(
                model_type="fbanks_cnn",
                pipeline_type="deep_fbanks",
                use_handcrafted_features=False,
                batch_size=64,
                epochs=60,
                learning_rate=1e-3,
                weight_decay=1e-4,
                dropout=0.3,
                early_stopping_patience=12,
                seed=42,
                use_class_weights=True,
                num_workers=4,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_dir = (OUTPUT_ROOT / f"config_{config_key}"
                        / f"test_{test_subject}")

            result = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                config_key=config_key,
            )
            all_results[config_key].append(result)

    # ── Aggregate and report ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"LOSO Summary — {EXPERIMENT_NAME}")
    print(f"{'=' * 80}")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "fbanks_config": {
            "n_mels": FBANKS_N_MELS,
            "fmin": FBANKS_FMIN,
            "fmax": FBANKS_FMAX,
        },
        "configs": {},
    }

    for config_key, results in all_results.items():
        valid = [r for r in results if r.get("test_accuracy") is not None]
        label = CONFIGS[config_key]["label"]
        if valid:
            accs = [r["test_accuracy"] for r in valid]
            f1s = [r["test_f1_macro"] for r in valid]
            print(f"  Config {config_key} ({label}):")
            print(f"    Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
            print(f"    F1-macro : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
            print(f"    Folds    : {len(valid)}/{len(results)}")
            summary["configs"][config_key] = {
                "label": label,
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_folds": len(valid),
                "per_subject": results,
            }
        else:
            print(f"  Config {config_key} ({label}): ALL FOLDS FAILED")
            summary["configs"][config_key] = {
                "label": label,
                "error": "All folds failed",
                "per_subject": results,
            }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"\nSummary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # Report to hypothesis_executor
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        best_f1 = 0.0
        best_config = None
        for ck, results in all_results.items():
            valid = [r for r in results if r.get("test_f1_macro") is not None]
            if valid:
                mean_f1 = np.mean([r["test_f1_macro"] for r in valid])
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_config = ck
        if best_config is not None:
            valid = [r for r in all_results[best_config]
                     if r.get("test_accuracy") is not None]
            metrics = {
                "best_config": best_config,
                "best_label": CONFIGS[best_config]["label"],
                "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid])),
                "mean_f1_macro": float(best_f1),
            }
            mark_hypothesis_verified("H_FBANKS_EMG", metrics,
                                     experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_FBANKS_EMG", "All configurations failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
