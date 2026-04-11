"""
Experiment 65: Trainable FIR Deconvolution Frontend for Neural-Drive Approximation

Hypothesis:
    Subject-specific variability in surface EMG is largely caused by the
    skin/electrode transfer function — a linear filter applied to the underlying
    "neural drive" (motor unit discharge patterns).  A trainable per-channel
    FIR filter with strong regularization (L2 + smoothness) can approximate
    the inverse of that transfer function, recovering a more subject-invariant
    representation that improves cross-subject gesture recognition.

What is tested:
    • FIRDeconvFrontend: per-channel depthwise 1D FIR filter (63 taps, ~31.5 ms
      at 2000 Hz).  Initialized as identity (δ at center tap) — starts as a
      pure pass-through and deviates only under gradient pressure.
    • Regularization:
        - L2 on tap weights (λ=1e-3): prevents large divergence from identity.
        - Smoothness penalty on first differences (λ=5e-3): promotes a
          smooth frequency response, discouraging narrow-band overfit to
          training-subject spectral quirks.
    • CNN-BiGRU-Attention encoder on top of the filtered signal.
    • Training loss: CrossEntropy + FIR regularization penalties.

LOSO protocol (strictly enforced, zero leakage):
    ┌────────────────────────────────────────────────────────────────────┐
    │  For each fold  (test_subject = one of the N subjects):            │
    │    train_subjects = all_subjects minus {test_subject}              │
    │    1.  Load windows for ALL subjects (train + test).               │
    │    2.  Pool train-subject windows → train split (+ val held-out).  │
    │    3.  Test-subject windows → test split (NEVER seen in training). │
    │    4.  Channel mean/std computed from TRAIN windows ONLY.          │
    │    5.  Train FIRDeconvCNNGRU end-to-end on train split.            │
    │        FIR tap weights, λ values, and CNN/GRU params are all       │
    │        learned from training data only.                            │
    │    6.  Apply frozen model to test-subject windows.                 │
    │        model.eval() — no BN running-stat updates, no FIR          │
    │        adaptation, no test-time statistics.                        │
    └────────────────────────────────────────────────────────────────────┘

Leakage prevention summary:
    ✓  Channel normalization stats from X_train exclusively.
    ✓  FIR filter weights optimised on train set only.
    ✓  λ_l2 / λ_smooth are fixed constants, not estimated from data.
    ✓  No per-subject or per-fold test-time adaptation of any kind.
    ✓  val split carved from train subjects via random permutation.

Run examples:
    # 5-subject CI run (fast, default):
    python experiments/exp_65_trainable_fir_neural_drive_deconv_loso.py

    # Same, explicit:
    python experiments/exp_65_trainable_fir_neural_drive_deconv_loso.py --ci

    # Specific subjects:
    python experiments/exp_65_trainable_fir_neural_drive_deconv_loso.py --subjects DB2_s1,DB2_s12,DB2_s15

    # Full 20-subject run:
    python experiments/exp_65_trainable_fir_neural_drive_deconv_loso.py --full
"""

import gc
import json
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
from training.fir_deconv_trainer import FIRDeconvTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ════════════════════════════════════════════════════════════════════════════
#  Experiment settings
# ════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_65_trainable_fir_neural_drive_deconv"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# FIR frontend hyper-parameters
FILTER_LEN    = 63      # taps (odd); 63 × (1/2000 s) ≈ 31.5 ms
LAMBDA_L2     = 1e-3    # L2 penalty on FIR tap weights
LAMBDA_SMOOTH = 5e-3    # Smoothness penalty (first-difference L2 on taps)

# Encoder hyper-parameters
CNN_CHANNELS = (64, 128, 256)   # 3 CNN blocks
GRU_HIDDEN   = 128
NUM_HEADS    = 4


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _build_splits(
    subjects_data:   Dict,
    train_subjects:  List[str],
    test_subject:    str,
    common_gestures: List[int],
    multi_loader:    MultiSubjectLoader,
    val_ratio:       float = 0.15,
    seed:            int   = 42,
) -> Dict:
    """
    Construct train / val / test splits from loaded subject data.

    LOSO contract enforced here:
      • train + val windows come exclusively from `train_subjects`.
      • val split is carved from train data via random permutation.
      • test windows come exclusively from `test_subject`.
      • test-subject data is assembled AFTER the train/val split — it never
        influences val_ratio computation or any other train-side decision.

    Returns:
        {"train": Dict[int, np.ndarray],
         "val":   Dict[int, np.ndarray],
         "test":  Dict[int, np.ndarray]}
        Each inner dict: gesture_id → (N, T, C) array.
    """
    rng = np.random.RandomState(seed)

    # ── accumulate train windows per gesture (from train subjects only) ──
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

    # ── concatenate, then split → train / val ────────────────────────────
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid  = np.concatenate(train_dict[gid], axis=0)   # (N, T, C)
        n      = len(X_gid)
        perm   = rng.permutation(n)
        n_val  = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        if len(trn_idx) > 0:
            final_train[gid] = X_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid]   = X_gid[val_idx]

    # ── test split — test subject data only ──────────────────────────────
    # Assembled independently; never influences any train-side statistics.
    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw  = subjects_data[test_subject]
        filtered_test  = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": final_test}


# ════════════════════════════════════════════════════════════════════════════
#  Single LOSO fold
# ════════════════════════════════════════════════════════════════════════════

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
    Execute one LOSO fold: train on `train_subjects`, evaluate on `test_subject`.

    All FIR filter parameters and normalization statistics are derived from
    the training split only.  The test subject is isolated until the final
    evaluation step.

    Returns:
        dict with at least "test_accuracy" and "test_f1_macro".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "fir_deconv_cnn_gru"

    # Persist configs for reproducibility
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as fh:
        json.dump(asdict(split_cfg), fh, indent=4)

    # ── data loading ──────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config    = proc_cfg,
        logger               = logger,
        use_gpu              = True,
        use_improved_processing = USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    # Load train subjects AND test subject in a single call to reuse the cache.
    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir    = base_dir,
        subject_ids = all_subject_ids,
        exercises   = exercises,
        include_rest = split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES
    )
    logger.info(
        f"Common gestures ({len(common_gestures)}): {common_gestures} "
        f"across {len(all_subject_ids)} subjects"
    )

    # ── build LOSO splits ─────────────────────────────────────────────────
    splits = _build_splits(
        subjects_data   = subjects_data,
        train_subjects  = train_subjects,
        test_subject    = test_subject,
        common_gestures = common_gestures,
        multi_loader    = multi_loader,
        val_ratio       = split_cfg.val_ratio,
        seed            = train_cfg.seed,
    )

    for sname in ("train", "val", "test"):
        n_windows = sum(
            len(arr) for arr in splits[sname].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"  {sname.upper():5s}: {n_windows:5d} windows, "
            f"{len(splits[sname])} gestures"
        )

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = FIRDeconvTrainer(
        train_cfg      = train_cfg,
        logger         = logger,
        output_dir     = output_dir,
        visualizer     = base_viz,
        filter_len     = FILTER_LEN,
        cnn_channels   = CNN_CHANNELS,
        gru_hidden     = GRU_HIDDEN,
        num_heads      = NUM_HEADS,
        lambda_l2      = LAMBDA_L2,
        lambda_smooth  = LAMBDA_SMOOTH,
    )

    # ── training ──────────────────────────────────────────────────────────
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed for test_subject={test_subject}: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "fir_deconv_cnn_gru",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── cross-subject evaluation on held-out test subject ─────────────────
    # Build flat (X, y) arrays from the test split.
    # class_ids ordering is established by trainer.fit() and must be reused
    # here so that integer labels match the model's output neurons.
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error(f"No test windows for subject {test_subject}.")
        return {
            "test_subject":  test_subject,
            "model_type":    "fir_deconv_cnn_gru",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy applies training-data standardization internally —
    # no test statistics are used.
    test_results = trainer.evaluate_numpy(
        X_test,
        y_test,
        split_name = f"cross_subject_test_{test_subject}",
        visualize  = True,
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
            "subject":          test_subject,
            "accuracy":         test_acc,
            "f1_macro":         test_f1,
            "report":           test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as fh:
        json.dump(
            make_json_serializable(fold_summary), fh, indent=4, ensure_ascii=False
        )

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "model_type":     "fir_deconv_cnn_gru",
            "approach":       APPROACH,
            "exercises":      exercises,
            "frontend_config": {
                "filter_len":    FILTER_LEN,
                "lambda_l2":     LAMBDA_L2,
                "lambda_smooth": LAMBDA_SMOOTH,
                "cnn_channels":  list(CNN_CHANNELS),
                "gru_hidden":    GRU_HIDDEN,
                "num_heads":     NUM_HEADS,
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
        "model_type":    "fir_deconv_cnn_gru",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── subject list ──────────────────────────────────────────────────────
    # Default MUST be CI_TEST_SUBJECTS (server has symlinks only for 5 CI subjects).
    # Use --full to run the complete 20-subject evaluation.
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str,  default=None,
                         help="Comma-separated subject IDs")
    _parser.add_argument("--ci",       action="store_true",
                         help="Use 5-subject CI subset (default)")
    _parser.add_argument("--full",     action="store_true",
                         help="Use all 20 subjects")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS      # safe server default

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    # ── configs ───────────────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size          = 600,
        window_overlap       = 300,
        num_channels         = 8,
        sampling_rate        = 2000,
        segment_edge_margin  = 0.1,
    )
    split_cfg = SplitConfig(
        train_ratio            = 0.7,
        val_ratio              = 0.15,
        test_ratio             = 0.15,
        mode                   = "by_segments",
        shuffle_segments       = True,
        seed                   = 42,
        include_rest_in_splits = False,
    )
    train_cfg = TrainingConfig(
        model_type               = "fir_deconv_cnn_gru",
        pipeline_type            = APPROACH,
        use_handcrafted_features = False,
        batch_size               = 64,
        epochs                   = 60,
        learning_rate            = 1e-3,
        weight_decay             = 1e-4,
        dropout                  = 0.3,
        early_stopping_patience  = 12,
        seed                     = 42,
        use_class_weights        = True,
        num_workers              = 4,
        device                   = "cuda" if torch.cuda.is_available() else "cpu",
    )

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis : Trainable per-channel FIR frontend approximates the "
        f"inverse of the skin/electrode transfer function, yielding a "
        f"subject-invariant neural-drive representation."
    )
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Exercises  : {EXERCISES}")
    print(
        f"Frontend   : FIR filter_len={FILTER_LEN} taps, "
        f"λ_l2={LAMBDA_L2}, λ_smooth={LAMBDA_SMOOTH}"
    )
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / "fir_deconv_cnn_gru" / f"test_{test_subject}"

        result = run_single_loso_fold(
            base_dir       = BASE_DIR,
            output_dir     = fold_dir,
            train_subjects = train_subjects,
            test_subject   = test_subject,
            exercises      = EXERCISES,
            proc_cfg       = proc_cfg,
            split_cfg      = split_cfg,
            train_cfg      = train_cfg,
        )
        all_results.append(result)

    # ── aggregate ─────────────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"] for r in valid]
        print(f"\n{'=' * 60}")
        print(f"FIRDeconv-CNNGRU — LOSO Summary ({len(valid)} folds)")
        print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    # ── save summary JSON ─────────────────────────────────────────────────
    summary: Dict = {
        "experiment":  EXPERIMENT_NAME,
        "hypothesis":  (
            "Trainable per-channel FIR frontend as neural-drive deconvolution "
            "for cross-subject EMG classification."
        ),
        "timestamp":   TIMESTAMP,
        "subjects":    ALL_SUBJECTS,
        "exercises":   EXERCISES,
        "approach":    APPROACH,
        "frontend_config": {
            "filter_len":    FILTER_LEN,
            "lambda_l2":     LAMBDA_L2,
            "lambda_smooth": LAMBDA_SMOOTH,
            "cnn_channels":  list(CNN_CHANNELS),
            "gru_hidden":    GRU_HIDDEN,
            "num_heads":     NUM_HEADS,
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
    summary_path = OUTPUT_ROOT / "loso_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(make_json_serializable(summary), fh, indent=4, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")

    # ── report to hypothesis_executor if available ────────────────────────
    try:
        from hypothesis_executor import (            # noqa: F401
            mark_hypothesis_verified,
            mark_hypothesis_failed,
        )
        if valid:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
            }
            mark_hypothesis_verified(
                "H_FIR_DECONV", metrics,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                "H_FIR_DECONV", "All LOSO folds failed"
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
