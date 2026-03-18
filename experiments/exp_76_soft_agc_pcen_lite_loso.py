"""
Experiment 76: Soft AGC / PCEN-lite Frontend for LOSO EMG Classification

Hypothesis:
    Aggressive PCEN (exp_61) destroyed discriminative amplitude cues.
    A softer dynamic range normalizer should reduce inter-subject amplitude
    variation (electrode placement, skin impedance) without over-suppressing
    gesture-relevant amplitude modulation.

Three frontend variants are compared in the same LOSO protocol:

  "log_affine"
      out[c,t] = log(ε + |x[c,t]|) * scale[c] + bias[c]
      Static log-compression with learnable per-channel affine.
      2*C learnable parameters.  No adaptive component.

  "rms_window"
      out[c,t] = x[c,t] / sqrt( mean_{W-window}( x[c,t]^2 ) + ε )
      Causal local-RMS normalization.  ZERO learnable parameters.
      Amplitude stabilization without any subject-specific adaptation.

  "soft_agc"
      M[c,t] = (1-s[c])*M[c,t-1] + s[c]*|x[c,t]|  (causal EMA)
      out[c,t] = x[c,t] / ( M[c,t]^alpha[c] + delta )
      EMA-based AGC with alpha ∈ (0, 0.5] — half the PCEN range [0,1].
      delta is FIXED.  2*C learnable parameters (alpha_raw, log_s).

LOSO protocol (strictly enforced, zero test leakage):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  For each fold  (test_subject = one subject from ALL_SUBJECTS):      │
    │    train_subjects = ALL_SUBJECTS \ {test_subject}                    │
    │    1. Load windows for ALL subjects via load_multiple_subjects().    │
    │    2. Pool train-subject windows → train split.                      │
    │       Carve val split from train data (val_ratio of train windows).  │
    │    3. Test-subject windows → test split (NEVER used in training).    │
    │    4. Channel mean/std computed from train windows ONLY.             │
    │    5. Train SoftAGCCNNGRU end-to-end on train split.                │
    │       All frontend parameters trained on train subjects only.        │
    │    6. Apply frozen model to test-subject windows.                    │
    │       No BatchNorm updates, no frontend parameter changes.           │
    └──────────────────────────────────────────────────────────────────────┘

Run examples:
    # 5-subject CI run (server-safe default):
    python experiments/exp_76_soft_agc_pcen_lite_loso.py

    # Explicit CI flag:
    python experiments/exp_76_soft_agc_pcen_lite_loso.py --ci

    # Specific subjects:
    python experiments/exp_76_soft_agc_pcen_lite_loso.py --subjects DB2_s1,DB2_s12

    # Full 20-subject run:
    python experiments/exp_76_soft_agc_pcen_lite_loso.py --full
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
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.soft_agc_trainer import SoftAGCTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ═══════════════════════════ EXPERIMENT SETTINGS ════════════════════════════

EXPERIMENT_NAME = "exp_76_soft_agc_pcen_lite"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# Frontend configurations
FRONTEND_TYPES = ["log_affine", "rms_window", "soft_agc"]

# RMSWindowLayer: causal window length in samples (50 ≈ 25 ms @ 2 kHz)
RMS_WINDOW_SIZE = 50

# SoftAGCLayer: EMA kernel length.  100 samples captures ~98% IIR energy at s=0.04
AGC_EMA_LENGTH  = 100

# SoftAGCLayer: FIXED additive stabilizer in gain denominator.
# Not learned → model cannot adapt delta to test-subject noise floor.
AGC_DELTA       = 0.1

# Encoder configuration (same for all frontend types)
CNN_CHANNELS = [64, 128, 256]
GRU_HIDDEN   = 128


# ════════════════════════════ HELPERS ════════════════════════════════════════

def grouped_to_arrays(grouped_windows: Dict[int, List[np.ndarray]]):
    """
    Convert grouped_windows (gesture_id → list of (N_rep, T, C) arrays) to
    flat (X, y) arrays for a single subject.

    Returns:
        windows: (N_total, T, C)
        labels:  (N_total,) — integer class indices (sorted gesture order)
    """
    xs, ys = [], []
    for class_idx, (gid) in enumerate(sorted(grouped_windows.keys())):
        reps = grouped_windows[gid]
        for rep_arr in reps:
            if isinstance(rep_arr, np.ndarray) and rep_arr.ndim == 3 and len(rep_arr) > 0:
                xs.append(rep_arr)
                ys.append(np.full(len(rep_arr), class_idx, dtype=np.int64))
    if not xs:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


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
    Build train / val / test split dictionaries from loaded subject data.

    LOSO contract (enforced here):
      - train + val windows come EXCLUSIVELY from `train_subjects`.
      - test windows come EXCLUSIVELY from `test_subject`.
      - val is carved out of train via random permutation; NO test-subject
        data is ever used to select hyper-parameters or early-stop epochs.

    Returns:
        {
            "train": Dict[int, np.ndarray],   gesture_id → (N, T, C)
            "val":   Dict[int, np.ndarray],
            "test":  Dict[int, np.ndarray],
        }
    """
    rng = np.random.RandomState(seed)

    # ── Accumulate training windows per gesture ──────────────────────────
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

    # ── Concatenate and split train → train / val ────────────────────────
    # The split is performed per-gesture to maintain class balance.
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)  # (N, T, C)
        n        = len(X_gid)
        perm     = rng.permutation(n)
        n_val    = max(1, int(n * val_ratio))
        val_idx  = perm[:n_val]
        trn_idx  = perm[n_val:]
        if len(trn_idx) > 0:
            final_train[gid] = X_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid]   = X_gid[val_idx]

    # ── Build test split from test subject ONLY ──────────────────────────
    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": final_test}


# ════════════════════════════ SINGLE FOLD ════════════════════════════════════

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    frontend_type: str,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
) -> Dict:
    """
    Execute one LOSO fold for a specific frontend type.

    Trains SoftAGCCNNGRU[frontend_type] on `train_subjects`,
    evaluates on `test_subject`.

    Returns:
        dict with at least "test_accuracy" and "test_f1_macro".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = f"soft_agc_cnn_gru_{frontend_type}"

    # Save configs for reproducibility
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # ── Data loader ──────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    # ── Load all subjects (train + test) in one call ─────────────────────
    # Order: train subjects first, test subject last (avoids duplicates).
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
        f"[{frontend_type}] Common gestures across {len(all_subject_ids)} subjects "
        f"({len(common_gestures)} total): {common_gestures}"
    )

    # ── Build LOSO splits ────────────────────────────────────────────────
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
        logger.info(
            f"  {sname.upper()}: {total_windows} windows, "
            f"{len(splits[sname])} gestures"
        )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = SoftAGCTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        frontend_type=frontend_type,
        rms_window_size=RMS_WINDOW_SIZE,
        agc_ema_length=AGC_EMA_LENGTH,
        agc_delta=AGC_DELTA,
        cnn_channels=CNN_CHANNELS,
        gru_hidden=GRU_HIDDEN,
    )

    # ── Training ─────────────────────────────────────────────────────────
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed for frontend={frontend_type}: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "frontend_type": frontend_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── Cross-subject test evaluation ────────────────────────────────────
    # Assemble flat (X_test, y_test) arrays from the test split.
    # class_ids ordering comes from trainer.fit() — must use the same mapping.
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error(
            f"No test windows available for subject {test_subject} "
            f"(frontend={frontend_type})."
        )
        return {
            "test_subject":  test_subject,
            "frontend_type": frontend_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy applies training-statistics standardization (no test leakage)
    test_results = trainer.evaluate_numpy(
        X_test_concat,
        y_test_concat,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] [{frontend_type}] Test subject {test_subject} | "
        f"Acc={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── Save fold results ─────────────────────────────────────────────────
    fold_summary = {
        "test_subject":    test_subject,
        "train_subjects":  train_subjects,
        "frontend_type":   frontend_type,
        "common_gestures": common_gestures,
        "training":        training_results,
        "cross_subject_test": {
            "subject":          test_subject,
            "accuracy":         test_acc,
            "f1_macro":         test_f1,
            "report":           test_results.get("report"),
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
            "frontend_type": frontend_type,
            "exercises":     exercises,
            "frontend_config": {
                "rms_window_size": RMS_WINDOW_SIZE,
                "agc_ema_length":  AGC_EMA_LENGTH,
                "agc_delta":       AGC_DELTA,
                "cnn_channels":    CNN_CHANNELS,
                "gru_hidden":      GRU_HIDDEN,
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # ── Cleanup ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data, splits
    gc.collect()

    return {
        "test_subject":  test_subject,
        "frontend_type": frontend_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ══════════════════════════════════ MAIN ════════════════════════════════════

def main():
    import argparse

    # ── Subject list ──────────────────────────────────────────────────────
    # Default: CI_TEST_SUBJECTS (5 subjects, safe for vast.ai server).
    # Use --full for the complete 20-subject evaluation.
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject IDs")
    _parser.add_argument("--ci",   action="store_true",
                         help="CI test subset (5 subjects)")
    _parser.add_argument("--full", action="store_true",
                         help="Full 20-subject LOSO run")
    # Allow individual frontend selection for partial reruns
    _parser.add_argument("--frontends", type=str, default=None,
                         help="Comma-separated frontend types to run, "
                              "e.g. log_affine,soft_agc")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS   # server-safe default

    if _args.frontends:
        selected_frontends = [f.strip() for f in _args.frontends.split(",")]
        # Validate against known frontend types
        for ft in selected_frontends:
            if ft not in FRONTEND_TYPES:
                raise ValueError(
                    f"Unknown frontend type '{ft}'. "
                    f"Must be one of {FRONTEND_TYPES}."
                )
    else:
        selected_frontends = FRONTEND_TYPES

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    # ── Configs ───────────────────────────────────────────────────────────
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
        model_type="soft_agc_cnn_gru",
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
    print(f"Hypothesis : Soft AGC / PCEN-lite frontend for cross-subject EMG")
    print(f"Frontends  : {selected_frontends}")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Exercises  : {EXERCISES}")
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    # ── LOSO loop over frontend types ─────────────────────────────────────
    all_results: Dict[str, List[Dict]] = {}

    for frontend_type in selected_frontends:
        print(f"\n{'─' * 60}")
        print(f"Frontend: {frontend_type}")
        print(f"{'─' * 60}")
        results_for_type = []

        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_dir = OUTPUT_ROOT / frontend_type / f"test_{test_subject}"

            result = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                frontend_type=frontend_type,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
            )
            results_for_type.append(result)

        all_results[frontend_type] = results_for_type

    # ── Aggregate per-frontend summary ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"LOSO Summary — {EXPERIMENT_NAME}")
    print(f"{'=' * 70}")

    aggregate: Dict[str, Dict] = {}
    for frontend_type in selected_frontends:
        valid = [
            r for r in all_results[frontend_type]
            if r.get("test_accuracy") is not None
        ]
        if valid:
            accs = [r["test_accuracy"] for r in valid]
            f1s  = [r["test_f1_macro"] for r in valid]
            mean_acc = float(np.mean(accs))
            std_acc  = float(np.std(accs))
            mean_f1  = float(np.mean(f1s))
            std_f1   = float(np.std(f1s))
            print(
                f"  [{frontend_type:12s}] "
                f"Acc={mean_acc:.4f}±{std_acc:.4f}  "
                f"F1={mean_f1:.4f}±{std_f1:.4f}  "
                f"({len(valid)} folds)"
            )
            aggregate[frontend_type] = {
                "mean_accuracy": mean_acc,
                "std_accuracy":  std_acc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro":  std_f1,
                "num_folds":     len(valid),
            }
        else:
            print(f"  [{frontend_type:12s}] ALL FOLDS FAILED")
            aggregate[frontend_type] = {"error": "all_folds_failed"}

    print(f"{'=' * 70}\n")

    # ── Save overall summary JSON ─────────────────────────────────────────
    summary = {
        "experiment":  EXPERIMENT_NAME,
        "hypothesis":  (
            "Softer AGC normalization (log+affine, local-RMS, or bounded-alpha EMA) "
            "reduces inter-subject amplitude variation without over-suppressing "
            "gesture-discriminative amplitude cues."
        ),
        "timestamp":   TIMESTAMP,
        "subjects":    ALL_SUBJECTS,
        "exercises":   EXERCISES,
        "approach":    APPROACH,
        "frontend_types": selected_frontends,
        "frontend_config": {
            "rms_window_size": RMS_WINDOW_SIZE,
            "agc_ema_length":  AGC_EMA_LENGTH,
            "agc_delta":       AGC_DELTA,
            "cnn_channels":    CNN_CHANNELS,
            "gru_hidden":      GRU_HIDDEN,
        },
        "aggregate": aggregate,
        "results_by_frontend": {
            ft: all_results[ft] for ft in selected_frontends
        },
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")

    # ── Report to hypothesis_executor if available ────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed  # noqa

        best_acc = -1.0
        best_frontend = None
        for ft, agg in aggregate.items():
            if "mean_accuracy" in agg and agg["mean_accuracy"] > best_acc:
                best_acc    = agg["mean_accuracy"]
                best_frontend = ft

        if best_frontend is not None:
            metrics = {
                "best_frontend":     best_frontend,
                "best_mean_accuracy": best_acc,
                **{
                    f"{ft}_mean_accuracy": aggregate[ft].get("mean_accuracy")
                    for ft in selected_frontends
                    if "mean_accuracy" in aggregate[ft]
                },
                **{
                    f"{ft}_mean_f1": aggregate[ft].get("mean_f1_macro")
                    for ft in selected_frontends
                    if "mean_f1_macro" in aggregate[ft]
                },
            }
            mark_hypothesis_verified(
                "H_SOFT_AGC", metrics, experiment_name=EXPERIMENT_NAME
            )
        else:
            mark_hypothesis_failed(
                "H_SOFT_AGC", "All LOSO folds failed across all frontend types"
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
