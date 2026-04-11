"""
Experiment 62: ECAPA-TDNN for Subject-Robust EMG Gesture Recognition (LOSO)

Hypothesis
──────────
CNN-GRU may underperform across subjects because:
  (a) global max-pooling over time loses temporal variance information;
  (b) fixed spatial filters cannot adapt to inter-subject channel differences;
  (c) GRU hidden state is sensitive to sequential ordering / gesture speed variation.

ECAPA-TDNN (originally for speaker verification) addresses these via:
  • Attentive Statistics Pooling — soft frame selection + mean AND std aggregation,
    giving temporal-spread information without being order-sensitive.
  • SE (Squeeze-and-Excitation) channel attention — suppresses noisy EMG channels,
    amplifies discriminative ones, learned from training subjects.
  • Res2Net multi-scale temporal context — hierarchical, multi-granularity features
    at multiple dilations without pooling (preserves all temporal positions).
  • Multi-layer Feature Aggregation — model simultaneously uses short-range and
    long-range temporal context for classification.

LOSO Protocol (strictly enforced — no adaptation)
──────────────────────────────────────────────────
  For each fold (test_subject ∈ ALL_SUBJECTS):
    train_subjects = ALL_SUBJECTS \\ {test_subject}

    1. Load windows for ALL subjects (train + test) in one call.
    2. Build train split: pool all train-subject windows per gesture,
       randomly carve val_ratio fraction as validation (no test data used).
    3. Build test split: test_subject windows only.
    4. Compute per-channel mean/std from TRAIN windows ONLY.
       Apply these same statistics to val and test (no test-stat leakage).
    5. Train ECAPATDNNEmg end-to-end on train split.
    6. Run model.eval() on test split — no BN updates, no adaptation.
       Test subject NEVER seen during training or normalisation.

Data-leakage guards:
  ✓ _build_splits() populates train/val from train_subjects only.
  ✓ Test split derived from test_subject data only.
  ✓ mean_c / std_c computed inside ECAPATDNNTrainer.fit() from X_train only.
  ✓ No common_gestures statistics come from signal values — only gesture ID sets.
  ✓ model.eval() at inference: BatchNorm frozen to training running stats.

Comparison
──────────
  Baseline: CNNGRUWithAttention (cnn=[32,64], gru_hidden=128, 2 layers)  ≈ 524 K params
  Proposed: ECAPATDNNEmg (C=128, scale=4, embed=128, dilations=[2,3,4])  ≈ 467 K params

  Same training config, same data pipeline, same LOSO protocol.
  The only difference is the architecture.

Run examples
────────────
  # 5-subject CI run (fast, safe default for server):
  python experiments/exp_62_ecapa_tdnn_subject_robust_loso.py --ci

  # Specific subjects:
  python experiments/exp_62_ecapa_tdnn_subject_robust_loso.py \\
      --subjects DB2_s1,DB2_s12,DB2_s15

  # Full 20-subject run:
  python experiments/exp_62_ecapa_tdnn_subject_robust_loso.py --full
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
from training.ecapa_tdnn_trainer import ECAPATDNNTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ═══════════════════════════ EXPERIMENT SETTINGS ════════════════════════════

EXPERIMENT_NAME = "exp_62_ecapa_tdnn_subject_robust"
APPROACH        = "deep_raw"
EXERCISES       = ["E1",]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# ECAPA-TDNN architecture hyper-parameters
ECAPA_CHANNELS      = 128    # C — internal TDNN feature width
ECAPA_SCALE         = 4      # Res2Net scale (sub-group count)
ECAPA_EMBEDDING_DIM = 128    # pre-classifier embedding dimension
ECAPA_DILATIONS     = [2, 3, 4]   # dilation per SE-Res2Net block
ECAPA_SE_REDUCTION  = 8      # SE bottleneck ratio


# ══════════════════════════════ HELPER: local grouped_to_arrays ══════════════

def grouped_to_arrays(grouped_windows: Dict[int, List[np.ndarray]]):
    """
    Convert grouped_windows (Dict[int, List[np.ndarray]]) to flat arrays.

    NOTE: grouped_to_arrays does NOT exist in any processing/ module.
    This local helper must be defined in every experiment file that needs it.

    Returns:
        windows: (N, T, C) float32
        labels:  (N,)      int64   (values = sorted gesture IDs, NOT class indices)
    """
    all_windows, all_labels = [], []
    for gid in sorted(grouped_windows.keys()):
        reps = grouped_windows[gid]
        for rep_arr in reps:
            if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                all_windows.append(rep_arr)
                all_labels.append(
                    np.full(len(rep_arr), gid, dtype=np.int64)
                )
    if not all_windows:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(all_windows, axis=0).astype(np.float32),
        np.concatenate(all_labels,  axis=0),
    )


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
    Construct train / val / test splits from loaded subject data.

    LOSO invariants upheld here:
      ─ train + val windows come ONLY from train_subjects.
      ─ val is carved from train data via random permutation (no leakage).
      ─ test windows come ONLY from test_subject.
      ─ No signal statistics (mean, std) are computed in this function;
        normalisation happens later inside ECAPATDNNTrainer.fit().

    Args:
        subjects_data:    Dict returned by MultiSubjectLoader.load_multiple_subjects.
                          Keys: subject_id. Values: (emg, segments, grouped_windows).
        train_subjects:   Subject IDs used for training (≠ test_subject).
        test_subject:     Subject ID held out for evaluation.
        common_gestures:  Gesture IDs common across all subjects.
        multi_loader:     MultiSubjectLoader instance (for filter_by_gestures).
        val_ratio:        Fraction of train windows held out for validation.
        seed:             RNG seed for reproducible val split.

    Returns:
        {"train": Dict[int, np.ndarray],  # gesture_id → (N, T, C)
         "val":   Dict[int, np.ndarray],
         "test":  Dict[int, np.ndarray]}
    """
    rng = np.random.RandomState(seed)

    # ── accumulate train windows per gesture (from train subjects only) ──
    train_dict: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}

    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        # subjects_data values are tuples (emg, segments, grouped_windows)
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        for gid, reps in filtered.items():
            for rep_arr in reps:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    train_dict[gid].append(rep_arr)

    # ── split train windows into train / val (no test data touches this) ─
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)   # (N, T, C)
        n       = len(X_gid)
        perm    = rng.permutation(n)
        n_val   = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        if len(trn_idx) > 0:
            final_train[gid] = X_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid]   = X_gid[val_idx]

    # ── test split: test subject data only ───────────────────────────────
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

    Returns dict with at least "test_accuracy" and "test_f1_macro".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "ecapa_tdnn_emg"

    # Save configs for full reproducibility
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as fh:
        json.dump(asdict(split_cfg), fh, indent=4)

    # ── data loader ──────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    # ── load ALL subjects (train + test) in a single call ────────────────
    # We load the test subject together with train subjects so that
    # get_common_gestures() can find gesture IDs present in ALL subjects.
    # No signal statistics (mean/std) are derived here — only gesture ID sets.
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
        f"Common gestures across {len(all_subject_ids)} subjects: "
        f"{common_gestures}  ({len(common_gestures)} total)"
    )

    # ── build LOSO-clean splits ───────────────────────────────────────────
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
        total_w = sum(
            len(arr)
            for arr in splits[sname].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"  {sname.upper():5s}: {total_w:5d} windows, "
            f"{len(splits[sname])} gestures"
        )

    # ── trainer ──────────────────────────────────────────────────────────
    trainer = ECAPATDNNTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        channels=ECAPA_CHANNELS,
        scale=ECAPA_SCALE,
        embedding_dim=ECAPA_EMBEDDING_DIM,
        dilations=ECAPA_DILATIONS,
        se_reduction=ECAPA_SE_REDUCTION,
    )

    # ── training ─────────────────────────────────────────────────────────
    # fit() receives the splits dict; it computes normalisation from
    # X_train only (see ECAPATDNNTrainer.fit() for LOSO integrity details).
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "ecapa_tdnn_emg",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── cross-subject test evaluation ────────────────────────────────────
    # Assemble flat test arrays.  Use class_ids ordering from trainer so
    # class indices match what the model was trained with.
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
            "model_type":    "ecapa_tdnn_emg",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy() applies transpose + training-stats normalisation;
    # model.eval() — no BatchNorm updates, no test-subject adaptation.
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
        "test_subject":       test_subject,
        "train_subjects":     train_subjects,
        "common_gestures":    common_gestures,
        "training":           training_results,
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
            "test_subject":  test_subject,
            "train_subjects": train_subjects,
            "model_type":    "ecapa_tdnn_emg",
            "approach":      APPROACH,
            "exercises":     exercises,
            "ecapa_config": {
                "channels":      ECAPA_CHANNELS,
                "scale":         ECAPA_SCALE,
                "embedding_dim": ECAPA_EMBEDDING_DIM,
                "dilations":     ECAPA_DILATIONS,
                "se_reduction":  ECAPA_SE_REDUCTION,
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # ── memory cleanup ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data, splits
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    "ecapa_tdnn_emg",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════════════════ MAIN ══════════════════════════════════

def main():
    """
    LOSO evaluation loop.

    Subject list priority:
      1. --subjects DB2_s1,DB2_s12,...  — explicit list
      2. --full                         — all 20 DEFAULT_SUBJECTS
      3. --ci  (or default)             — 5 CI_TEST_SUBJECTS  ← safe server default

    The default (no flags) intentionally uses CI_TEST_SUBJECTS because the
    vast.ai server has symlinks only for those 5 subjects.
    """
    import argparse

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject IDs")
    _parser.add_argument("--ci",   action="store_true",
                         help="Use 5 CI test subjects")
    _parser.add_argument("--full", action="store_true",
                         help="Use all 20 DEFAULT_SUBJECTS")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS      # safe default for server

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = (
        ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    )

    # ── configs ───────────────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,    # drop 10 % at each segment edge
    )
    split_cfg = SplitConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )
    train_cfg = TrainingConfig(
        model_type="ecapa_tdnn_emg",
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
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis  : ECAPA-TDNN (Res2Net + SE + Attentive Stats Pooling)\n"
        f"              vs CNN-GRU-Attention baseline  (equal param budget)"
    )
    print(f"Subjects    : {ALL_SUBJECTS}")
    print(f"Exercises   : {EXERCISES}")
    print(
        f"ECAPA config: C={ECAPA_CHANNELS}, scale={ECAPA_SCALE}, "
        f"embed={ECAPA_EMBEDDING_DIM}, dilations={ECAPA_DILATIONS}"
    )
    print(f"Output      : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / "ecapa_tdnn_emg" / f"test_{test_subject}"

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

    # ── aggregate LOSO summary ────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"]  for r in valid]
        mean_acc  = float(np.mean(accs))
        std_acc   = float(np.std(accs))
        mean_f1   = float(np.mean(f1s))
        std_f1    = float(np.std(f1s))

        print("\n" + "=" * 80)
        print("LOSO SUMMARY — ECAPATDNNEmg")
        print(f"  Subjects evaluated : {len(valid)}")
        print(
            f"  Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}"
            f"  (min={min(accs):.4f}, max={max(accs):.4f})"
        )
        print(
            f"  F1-macro  : {mean_f1:.4f} ± {std_f1:.4f}"
            f"  (min={min(f1s):.4f}, max={max(f1s):.4f})"
        )
        print("=" * 80)

        summary = {
            "experiment":     EXPERIMENT_NAME,
            "model":          "ecapa_tdnn_emg",
            "approach":       APPROACH,
            "subjects":       ALL_SUBJECTS,
            "exercises":      EXERCISES,
            "ecapa_config": {
                "channels":      ECAPA_CHANNELS,
                "scale":         ECAPA_SCALE,
                "embedding_dim": ECAPA_EMBEDDING_DIM,
                "dilations":     ECAPA_DILATIONS,
                "se_reduction":  ECAPA_SE_REDUCTION,
            },
            "loso_metrics": {
                "mean_accuracy":  mean_acc,
                "std_accuracy":   std_acc,
                "mean_f1_macro":  mean_f1,
                "std_f1_macro":   std_f1,
                "per_subject":    all_results,
            },
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_ROOT / "loso_summary.json", "w") as fh:
            json.dump(make_json_serializable(summary), fh, indent=4, ensure_ascii=False)
        print(f"Summary saved → {OUTPUT_ROOT / 'loso_summary.json'}")
    else:
        print("No successful folds to summarise.")

    # Optional: report to hypothesis executor if available
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        hypothesis_id = os.environ.get("HYPOTHESIS_ID", "")
        if hypothesis_id and valid:
            mark_hypothesis_verified(
                hypothesis_id,
                metrics={
                    "mean_accuracy":  mean_acc,
                    "std_accuracy":   std_acc,
                    "mean_f1_macro":  mean_f1,
                    "std_f1_macro":   std_f1,
                    "n_folds":        len(valid),
                },
                experiment_name=EXPERIMENT_NAME,
            )
        elif hypothesis_id and not valid:
            mark_hypothesis_failed(
                hypothesis_id,
                error_message="All LOSO folds failed — no valid results.",
            )
    except ImportError:
        pass   # hypothesis_executor not installed in this environment


if __name__ == "__main__":
    main()
