"""
Experiment 61: SincNet-PCEN Learnable Frontend for LOSO EMG Classification

Hypothesis:
    EMG amplitude varies across subjects due to electrode placement and skin impedance,
    analogous to channel/speaker gain variation in speech.  A learnable bandpass
    filterbank (SincNet-style) combined with Per-Channel Energy Normalization (PCEN)
    reduces this variability and should improve cross-subject generalization.

What is tested:
    - SincFilterbank: K learnable (f1, f2) pairs, Mel-initialized over [5, 500] Hz.
      Applied as a grouped 1D convolution (same filters for all C EMG channels).
    - PCENLayer: adaptive AGC with learnable (alpha, delta, root, s) per filter channel.
      EMA smoother implemented as a causal depthwise 1D conv (GPU-efficient, no loops).
    - CNN-BiGRU-Attention encoder on top of the PCEN features.

LOSO protocol (strictly enforced):
    ┌──────────────────────────────────────────────────────────────────┐
    │  For each fold  (test_subject = one of the N subjects):          │
    │    train_subjects = all_subjects \ {test_subject}                │
    │    1. Load windows for ALL subjects.                             │
    │    2. Pool train-subject windows → train split (+ val held out). │
    │    3. Test-subject windows → test split (NEVER seen in training).│
    │    4. Channel mean/std computed from train windows ONLY.         │
    │    5. Train SincPCENCNNGRU end-to-end on train split.            │
    │       All frontend parameters (SincFilterbank, PCENLayer) are    │
    │       trained jointly — they are model parameters, not heuristics.│
    │    6. Apply frozen model to test subject windows.                │
    │       No BN updates, no PCEN parameter updates, no adaptation.   │
    └──────────────────────────────────────────────────────────────────┘

Run examples:
    # 5-subject CI run (fast):
    python experiments/exp_61_sinc_pcen_frontend_loso.py --ci

    # Specific subjects:
    python experiments/exp_61_sinc_pcen_frontend_loso.py --subjects DB2_s1,DB2_s12,DB2_s15

    # Full 20-subject run:
    python experiments/exp_61_sinc_pcen_frontend_loso.py --full
"""

import gc
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
    parse_subjects_args,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.sinc_pcen_trainer import SincPCENTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ═══════════════════════════ EXPERIMENT SETTINGS ════════════════════════════

EXPERIMENT_NAME = "exp_61_sinc_pcen_frontend"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# Frontend hyper-parameters
NUM_SINC_FILTERS  = 32     # K — learnable bandpass filters per EMG channel
SINC_KERNEL_SIZE  = 51     # sinc impulse response length (samples, must be odd)
MIN_FREQ_HZ       = 5.0    # lower bound of filter frequency range
MAX_FREQ_HZ       = 500.0  # upper bound (≤ Nyquist = 1000 Hz for 2000 Hz EMG)
PCEN_EMA_LENGTH   = 128    # truncated IIR kernel length (captures ~99.4% for s=0.04)

# Encoder hyper-parameters
CNN_CHANNELS = [64, 128, 256]  # wider than standard (input has C*K channels)
GRU_HIDDEN   = 128


# ══════════════════════════════ SPLITS BUILDER ══════════════════════════════

def _build_splits(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Construct train / val / test split dictionaries from loaded subject data.

    LOSO contract:
      - train + val windows come exclusively from `train_subjects`.
      - test windows come exclusively from `test_subject`.
      - The val split is carved out of train data via random permutation
        (no test-subject data is ever used to tune hyper-parameters).

    Returns:
        dict with keys "train", "val", "test", each being Dict[int, np.ndarray]
        (gesture_id → (N, T, C) array).
    """
    rng = np.random.RandomState(seed)

    # ── accumulate training windows per gesture ──────────────────────────
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

    # ── concatenate and split train → train / val ────────────────────────
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)  # (N, T, C)
        n = len(X_gid)
        perm    = rng.permutation(n)
        n_val   = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        if len(trn_idx) > 0:
            final_train[gid] = X_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid]   = X_gid[val_idx]

    # ── build test split from test subject only ──────────────────────────
    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": final_test}


# ═══════════════════════════════ SINGLE FOLD ════════════════════════════════

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
) -> Dict:
    """
    Execute one LOSO fold: train on `train_subjects`, evaluate on `test_subject`.

    Returns a dict with at least "test_accuracy" and "test_f1_macro".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "sinc_pcen_cnn_gru"

    # Save configs for reproducibility
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # ── data loader ──────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    # ── load all subjects (train + test) in one call ─────────────────────
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
        f"Common gestures across {len(all_subject_ids)} subjects "
        f"({len(common_gestures)} total): {common_gestures}"
    )

    # ── build LOSO splits ────────────────────────────────────────────────
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
        total_windows = sum(
            len(arr) for arr in splits[sname].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(f"  {sname.upper()}: {total_windows} windows, "
                    f"{len(splits[sname])} gestures")

    # ── trainer ──────────────────────────────────────────────────────────
    trainer = SincPCENTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        sample_rate=proc_cfg.sampling_rate,
        num_sinc_filters=NUM_SINC_FILTERS,
        sinc_kernel_size=SINC_KERNEL_SIZE,
        min_freq=MIN_FREQ_HZ,
        max_freq=MAX_FREQ_HZ,
        pcen_ema_length=PCEN_EMA_LENGTH,
    )

    # ── training ─────────────────────────────────────────────────────────
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "sinc_pcen_cnn_gru",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── cross-subject test evaluation ────────────────────────────────────
    # Assemble flat test arrays from the test split.
    # class_ids ordering comes from trainer.fit() → must use the same mapping.
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error(f"No test windows available for subject {test_subject}.")
        return {
            "test_subject":  test_subject,
            "model_type":    "sinc_pcen_cnn_gru",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    test_results = trainer.evaluate_numpy(
        X_test_concat,
        y_test_concat,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Acc={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── save fold results ─────────────────────────────────────────────────
    fold_summary = {
        "test_subject":    test_subject,
        "train_subjects":  train_subjects,
        "common_gestures": common_gestures,
        "training":        training_results,
        "cross_subject_test": {
            "subject":         test_subject,
            "accuracy":        test_acc,
            "f1_macro":        test_f1,
            "report":          test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":  test_subject,
            "train_subjects": train_subjects,
            "model_type":    "sinc_pcen_cnn_gru",
            "approach":      APPROACH,
            "exercises":     exercises,
            "frontend_config": {
                "num_sinc_filters":  NUM_SINC_FILTERS,
                "sinc_kernel_size":  SINC_KERNEL_SIZE,
                "min_freq_hz":       MIN_FREQ_HZ,
                "max_freq_hz":       MAX_FREQ_HZ,
                "pcen_ema_length":   PCEN_EMA_LENGTH,
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # ── cleanup ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data, splits
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    "sinc_pcen_cnn_gru",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════════════════ MAIN ══════════════════════════════════

def main():
    # ── subject list ──────────────────────────────────────────────────────
    # Codegen rule: default MUST be CI_TEST_SUBJECTS.
    # Use --full to run the complete 20-subject evaluation.
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci",   action="store_true")
    _parser.add_argument("--full", action="store_true")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS   # safe default for server

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    # ── configs ───────────────────────────────────────────────────────────
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
        model_type="sinc_pcen_cnn_gru",
        pipeline_type=APPROACH,
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

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : SincNet-PCEN learnable front-end for channel-invariant EMG")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Exercises  : {EXERCISES}")
    print(f"Frontend   : K={NUM_SINC_FILTERS} sinc filters @ [{MIN_FREQ_HZ},{MAX_FREQ_HZ}] Hz, "
          f"kernel={SINC_KERNEL_SIZE}, PCEN_L={PCEN_EMA_LENGTH}")
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / "sinc_pcen_cnn_gru" / f"test_{test_subject}"

        result = run_single_loso_fold(
            base_dir=BASE_DIR,
            output_dir=fold_dir,
            train_subjects=train_subjects,
            test_subject=test_subject,
            exercises=EXERCISES,
            proc_cfg=proc_cfg,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
        )
        all_results.append(result)

    # ── aggregate ─────────────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"] for r in valid]
        print(f"\n{'=' * 60}")
        print(f"SincPCEN-CNNGRU — LOSO Summary ({len(valid)} folds)")
        print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    # ── save summary JSON ─────────────────────────────────────────────────
    summary = {
        "experiment":  EXPERIMENT_NAME,
        "hypothesis":  "SincNet-PCEN learnable frontend for channel-invariant EMG",
        "timestamp":   TIMESTAMP,
        "subjects":    ALL_SUBJECTS,
        "exercises":   EXERCISES,
        "approach":    APPROACH,
        "frontend_config": {
            "num_sinc_filters":  NUM_SINC_FILTERS,
            "sinc_kernel_size":  SINC_KERNEL_SIZE,
            "min_freq_hz":       MIN_FREQ_HZ,
            "max_freq_hz":       MAX_FREQ_HZ,
            "pcen_ema_length":   PCEN_EMA_LENGTH,
            "cnn_channels":      CNN_CHANNELS,
            "gru_hidden":        GRU_HIDDEN,
        },
        "results": all_results,
    }
    if valid:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_folds":     len(valid),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── report to hypothesis_executor if available ────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed  # noqa
        if valid:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
            }
            mark_hypothesis_verified("H_PCEN", metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_PCEN", "All LOSO folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
