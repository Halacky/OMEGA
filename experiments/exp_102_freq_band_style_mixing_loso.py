"""
Experiment 102: Frequency-Band Style Mixing for LOSO EMG Gesture Recognition.

Hypothesis H102 (= Hypothesis 5 in the research plan):
    "Frequency-Band Style Mixing with channel-wise spectral statistics"

Core idea
---------
Adapt MixStyle to EMG by mixing per-channel statistics band-selectively in the
FREQUENCY DOMAIN rather than the latent space:

  * Low band  (20–150 Hz)  — dominant source of inter-subject variance (fat/
    impedance). Apply AGGRESSIVE mixing λ ~ Beta(0.2, 0.2).
  * Mid band (150–450 Hz) — concentrates gesture-discriminative motor-unit
    recruitment patterns. Apply CONSERVATIVE mixing λ ~ Beta(0.8, 0.8).
  * High band (>450 Hz)   — noise-dominated; NO mixing.

Mechanism (LOSO-safe)
---------------------
For each training batch:
  1. Decompose each window x into band_low, band_mid via FFT masking.
  2. Compute per-sample, per-channel mean/std (instance statistics).
  3. Find a cross-subject partner j ≠ i within the batch.
  4. Mix statistics: μ_mix = λ·μ_i + (1-λ)·μ_j,  σ_mix = …
  5. Apply AdaIN: normalise band with own stats → rescale with mixed stats.
  6. Reconstruct: x_mixed = x + Δ_low + Δ_mid.
At inference: FreqBandStyleMixer is disabled (model.eval()) → raw signal → encoder.

LOSO data-leakage audit
-----------------------
✓ Style mixing uses ONLY samples within the current training batch.
✓ Test subject data goes EXCLUSIVELY into splits["test"] — never into any
  training batch or style pool.
✓ Channel standardization (mean_c, std_c) computed from training windows only.
✓ Validation split comes from training subjects (val_ratio fraction held out).
✓ Early stopping is driven by val_loss (no test information).
✓ No test-time adaptation: inference uses fixed model weights, eval mode.
✓ FFT band masks depend only on sampling rate and frequency cutoffs, never
  on any data statistics.

Expected result
---------------
F1-macro > 36%.  The frequency-domain analysis should also show that the
mid-band class separability is preserved after training (visualised separately).

Usage
-----
    python experiments/exp_102_freq_band_style_mixing_loso.py
    python experiments/exp_102_freq_band_style_mixing_loso.py --ci
    python experiments/exp_102_freq_band_style_mixing_loso.py \\
        --subjects DB2_s1,DB2_s12,DB2_s15,DB2_s28,DB2_s39
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

# ── Project imports ───────────────────────────────────────────────────────────
from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.freq_band_style_mix_trainer import FreqBandStyleMixTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ════════════════════════ EXPERIMENT SETTINGS ════════════════════════════════

EXPERIMENT_NAME = "exp_102_freq_band_style_mixing"
APPROACH        = "deep_raw"
EXERCISES       = ["E1"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES    = 10

# ── Frequency-band mixing hyperparameters ─────────────────────────────────────
LOW_BAND        = (20.0, 150.0)   # Hz — dominant inter-subject variance band
MID_BAND        = (150.0, 450.0)  # Hz — gesture-discriminative band
LOW_MIX_ALPHA   = 0.2             # Beta(0.2, 0.2) → aggressive mixing in low band
MID_MIX_ALPHA   = 0.8             # Beta(0.8, 0.8) → conservative mixing in mid band
CLASSIFIER_DIM  = 128             # hidden units in the 2-layer gesture head
SAMPLING_RATE   = 2000            # Hz (Ninapro DB2)


# ════════════════════════ DATA PREPARATION ════════════════════════════════════


def grouped_to_arrays(grouped_windows: Dict[int, List[np.ndarray]]):
    """
    Convert grouped_windows dict to flat (windows, labels) arrays.

    grouped_windows: {gesture_id: [rep_array_1, rep_array_2, ...]}
        where each rep_array is (N_rep, T, C).

    Returns:
        windows: (N_total, T, C) float32
        labels:  (N_total,) int64  — gesture IDs (NOT class indices)
    """
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        reps = grouped_windows[gid]
        for rep in reps:
            if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0:
                windows_list.append(rep)
                labels_list.append(np.full(len(rep), gid, dtype=np.int64))
    if not windows_list:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(windows_list, axis=0).astype(np.float32),
        np.concatenate(labels_list,  axis=0),
    )


def _build_splits_with_subject_labels(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Build train / val / test splits with subject-provenance tracking.

    Subject labels are required by FreqBandStyleMixTrainer so that the
    in-batch style mixing can find cross-subject partners during training.

    LOSO guarantee
    --------------
    Test-subject data goes EXCLUSIVELY into splits["test"].
    Training-subject data is split into train (1−val_ratio) and val (val_ratio).
    The test subject's windows never appear in train or val.

    Returns
    -------
    {
        "train":                Dict[gesture_id → np.ndarray (N, T, C)]
        "val":                  Dict[gesture_id → np.ndarray (N, T, C)]
        "test":                 Dict[gesture_id → np.ndarray (N, T, C)]
        "train_subject_labels": Dict[gesture_id → np.ndarray (N,) int]
        "num_train_subjects":   int
    }
    """
    rng = np.random.RandomState(seed)

    # Map train subject IDs to consecutive integers [0 .. K-1]
    # Test subject is NOT in this map.
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects   = len(train_subjects)

    # ── Step 1: collect per-gesture arrays from training subjects ─────────
    train_dict:       Dict[int, np.ndarray] = {}
    train_subj_dict:  Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        windows_for_gid = []
        subj_for_gid    = []

        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            # subjects_data values are tuples (emg, segments, grouped_windows)
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid not in filtered:
                continue
            for rep_array in filtered[gid]:
                if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                    windows_for_gid.append(rep_array)
                    subj_for_gid.append(
                        np.full(len(rep_array), train_subject_to_idx[sid],
                                dtype=np.int64)
                    )

        if windows_for_gid:
            train_dict[gid]      = np.concatenate(windows_for_gid, axis=0)
            train_subj_dict[gid] = np.concatenate(subj_for_gid,    axis=0)

    # ── Step 2: split train → train / val per gesture ─────────────────────
    # Permutation applied jointly to windows and subject labels to keep alignment.
    final_train:      Dict[int, np.ndarray] = {}
    final_val:        Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if gid not in train_dict:
            continue
        X_g = train_dict[gid]
        S_g = train_subj_dict[gid]
        n   = len(X_g)

        perm  = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        final_train[gid]      = X_g[train_idx]
        final_val[gid]        = X_g[val_idx]
        final_train_subj[gid] = S_g[train_idx]

    # ── Step 3: test split from test subject ONLY ─────────────────────────
    # Critical LOSO boundary.  Test subject windows go here and NOWHERE else.
    test_dict: Dict[int, np.ndarray] = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
        if valid:
            test_dict[gid] = np.concatenate(valid, axis=0)

    return {
        "train":                final_train,
        "val":                  final_val,
        "test":                 test_dict,
        "train_subject_labels": final_train_subj,
        "num_train_subjects":   num_train_subjects,
    }


# ════════════════════════ SINGLE LOSO FOLD ════════════════════════════════════


def run_single_loso_fold(
    base_dir:       Path,
    output_dir:     Path,
    train_subjects: List[str],
    test_subject:   str,
    exercises:      List[str],
    proc_cfg:       ProcessingConfig,
    split_cfg:      SplitConfig,
    train_cfg:      TrainingConfig,
) -> Dict:
    """
    Execute one LOSO fold using Frequency-Band Style Mixing.

    Returns a result dict with test_accuracy / test_f1_macro, or error info.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "freq_band_style_mix_emg"

    # ── Save configs for reproducibility ──────────────────────────────────
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "fbsm_config.json", "w") as f:
        json.dump({
            "low_band":       LOW_BAND,
            "mid_band":       MID_BAND,
            "low_mix_alpha":  LOW_MIX_ALPHA,
            "mid_mix_alpha":  MID_MIX_ALPHA,
            "classifier_dim": CLASSIFIER_DIM,
            "sampling_rate":  SAMPLING_RATE,
        }, f, indent=4)

    # ── Data loading ───────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

    # Load ALL subjects (train + test) in a single pass for efficiency.
    # The LOSO boundary is enforced in _build_splits_with_subject_labels
    # where test_subject windows go EXCLUSIVELY into splits["test"].
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
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    # ── Build splits with subject provenance ──────────────────────────────
    splits = _build_splits_with_subject_labels(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    for split_name in ["train", "val", "test"]:
        total = sum(
            len(arr)
            for arr in splits[split_name].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"{split_name.upper()}: {total} windows, "
            f"{len(splits[split_name])} gesture classes"
        )

    # ── Create trainer ────────────────────────────────────────────────────
    trainer = FreqBandStyleMixTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        low_band=LOW_BAND,
        mid_band=MID_BAND,
        low_mix_alpha=LOW_MIX_ALPHA,
        mid_mix_alpha=MID_MIX_ALPHA,
        classifier_dim=CLASSIFIER_DIM,
        sampling_rate=SAMPLING_RATE,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject":   test_subject,
            "model_type":     "freq_band_style_mix_emg",
            "approach":       APPROACH,
            "test_accuracy":  None,
            "test_f1_macro":  None,
            "error":          str(e),
        }

    # ── Evaluate on held-out test subject ─────────────────────────────────
    # Uses trainer.evaluate_numpy() which calls model in eval mode —
    # FreqBandStyleMixer is disabled; raw signal goes through encoder.
    # No subject information is needed or used.
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test data available after gesture filtering.")
        return {
            "test_subject":  test_subject,
            "model_type":    "freq_band_style_mix_emg",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy handles transpose + standardisation internally
    test_results = trainer.evaluate_numpy(
        X_test, y_test,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── Save fold results ──────────────────────────────────────────────────
    results_to_save = {
        "test_subject":       test_subject,
        "train_subjects":     train_subjects,
        "common_gestures":    common_gestures,
        "training":           training_results,
        "cross_subject_test": {
            "subject":           test_subject,
            "accuracy":          test_acc,
            "f1_macro":          test_f1,
            "report":            test_results.get("report"),
            "confusion_matrix":  test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f,
                  indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":     test_subject,
            "train_subjects":   train_subjects,
            "model_type":       "freq_band_style_mix_emg",
            "approach":         APPROACH,
            "exercises":        exercises,
            "fbsm_config": {
                "low_band":       LOW_BAND,
                "mid_band":       MID_BAND,
                "low_mix_alpha":  LOW_MIX_ALPHA,
                "mid_mix_alpha":  MID_MIX_ALPHA,
                "classifier_dim": CLASSIFIER_DIM,
                "sampling_rate":  SAMPLING_RATE,
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # ── Memory cleanup ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    "freq_band_style_mix_emg",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════ MAIN ════════════════════════════════════════════════


def main():
    # Codegen rule 24: parse_subjects_args() defaults to CI_TEST_SUBJECTS.
    # Full 20-subject list runs ONLY when --subjects or --full is given explicitly.
    # The server (vast.ai) has symlinks only for CI subjects.
    ALL_SUBJECTS = parse_subjects_args()

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    # ── Processing config ─────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=SAMPLING_RATE,
        segment_edge_margin=0.1,
    )

    # ── Split config ──────────────────────────────────────────────────────
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    # ── Training config ───────────────────────────────────────────────────
    train_cfg = TrainingConfig(
        model_type="freq_band_style_mix_emg",
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

    print(f"{'=' * 80}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Hypothesis H102: Frequency-Band Style Mixing (AdaIN in FFT bands)")
    print(f"Subjects:         {ALL_SUBJECTS}")
    print(f"Exercises:        {EXERCISES}")
    print(f"Low  band:  {LOW_BAND} Hz   alpha={LOW_MIX_ALPHA}  (aggressive)")
    print(f"Mid  band:  {MID_BAND} Hz  alpha={MID_MIX_ALPHA}  (conservative)")
    print(f"High band:  >450 Hz         alpha=none    (no mixing)")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = (
            OUTPUT_ROOT / "freq_band_style_mix_emg" / f"test_{test_subject}"
        )

        result = run_single_loso_fold(
            base_dir=BASE_DIR,
            output_dir=fold_output,
            train_subjects=train_subjects,
            test_subject=test_subject,
            exercises=EXERCISES,
            proc_cfg=proc_cfg,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
        )
        all_loso_results.append(result)

    # ── Aggregate LOSO summary ─────────────────────────────────────────────
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]

    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s  = [r["test_f1_macro"] for r in valid_results]
        print(f"\n{'=' * 60}")
        print(
            f"Freq-Band Style Mix CNN-GRU — LOSO Summary ({len(valid_results)} folds)"
        )
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    summary = {
        "experiment":  EXPERIMENT_NAME,
        "hypothesis":  "H102: Frequency-Band Style Mixing (AdaIN per EMG band)",
        "timestamp":   TIMESTAMP,
        "subjects":    ALL_SUBJECTS,
        "approach":    APPROACH,
        "fbsm_config": {
            "low_band":       LOW_BAND,
            "mid_band":       MID_BAND,
            "low_mix_alpha":  LOW_MIX_ALPHA,
            "mid_mix_alpha":  MID_MIX_ALPHA,
            "classifier_dim": CLASSIFIER_DIM,
            "sampling_rate":  SAMPLING_RATE,
        },
        "results": all_loso_results,
    }

    if valid_results:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_folds":     len(valid_results),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")

    # Report to hypothesis_executor if available (always guard import)
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        if valid_results:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
            }
            mark_hypothesis_verified("H102", metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H102", "All LOSO folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
