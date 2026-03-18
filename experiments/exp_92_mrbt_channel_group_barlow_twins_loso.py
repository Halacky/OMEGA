"""
exp_92 — Multi-Resolution Barlow Twins with Channel-Group Factorization (MRBT-CG)

Hypothesis 5:
    EMG channels are grouped by anatomical proximity (K groups of C_emg//K
    channels each).  Per-group ECAPA-TDNN branches at L parallel temporal
    scales (different dilations) produce multi-scale group features.
    Cross-group attention extracts the inter-group consensus (content) and
    the inter-group spread/std (style) at each scale.
    A Barlow-Twins-inspired cross-correlation loss pushes content and style
    apart at every scale without adversarial training.
    The classifier is trained on content features only.

Key design choices:
    • Shared encoder weights across K groups (parameter-efficient).
    • Parallel (not sequential) scale branches — each scale branch truly
      operates at a different temporal resolution.
    • style = std_k(group_features) — physically motivated: electrode
      placement / impedance effects produce inter-group amplitude variance.
    • BT loss = mean-squared cross-correlation(content, style) across batch,
      normalised per scale; no collapse risk because L_cls prevents content
      collapse.

LOSO Protocol (strict — NO adaptation to test subject):
    For each subject s in ALL_SUBJECTS:
        train_subjects = ALL_SUBJECTS \\ {s}
        test_subject   = s
        • All data loading uses the correct train/test partition.
        • mean_c, std_c computed from TRAINING windows only.
        • model.forward_with_style() called only on training batches.
        • model.forward() (content pathway) used for validation and test.
        • No test-subject statistics or windows ever reach the model during
          training.

Usage:
    python experiments/exp_92_mrbt_channel_group_barlow_twins_loso.py
    python experiments/exp_92_mrbt_channel_group_barlow_twins_loso.py --ci
    python experiments/exp_92_mrbt_channel_group_barlow_twins_loso.py --full
    python experiments/exp_92_mrbt_channel_group_barlow_twins_loso.py \\
        --subjects DB2_s1 DB2_s12 DB2_s15
"""

import gc
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.mrbt_trainer import MRBTTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything


# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = "exp_92_mrbt_channel_group_barlow_twins"
APPROACH = "deep_raw"
EXERCISES = ["E1", ]
MAX_GESTURES = 10

# ── Processing ───────────────────────────────────────────────────────────────
NUM_CHANNELS = 8         # NinaPro DB2 8-channel subset used in other experiments.
                         # With 12 channels, set n_groups=4 for 3 ch/group.
WINDOW_SIZE = 600        # 300 ms at 2 kHz
WINDOW_OVERLAP = 300     # 50 % overlap

# ── Architecture ─────────────────────────────────────────────────────────────
# n_groups=4 with 8 channels → 2 channels per group (coarse anatomical split).
# Hypothesis risk note: with 2 ch/group the cross-group agreement signal
# may be noisy.  n_groups=2 (4 ch/group) is a safer alternative if results
# are unstable.
N_GROUPS = 4
GROUP_CHANNELS = 32      # C_g — must be divisible by RES2_SCALE (32 // 4 = 8 ✓)
N_SCALES = 3             # L parallel temporal scales
DILATIONS = [1, 2, 4]   # receptive fields: 5, 9, 17 samples at 2 kHz (k=3)
EMBEDDING_DIM = 128
RES2_SCALE = 4           # Res2Net branching inside each group block
SE_REDUCTION = 8

# ── Barlow Twins loss ─────────────────────────────────────────────────────────
# lambda_bt controls content-style decorrelation strength.
# bt_warmup_frac: linear ramp-up over first 15 % of epochs to avoid early
# instability while the content pathway is still initialising.
LAMBDA_BT = 0.1
BT_WARMUP_FRAC = 0.15

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
EARLY_STOP_PATIENCE = 15
VAL_RATIO = 0.15
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Local helper: grouped_windows → flat (windows, labels) arrays
# NOTE: does NOT exist in any processing/ module — defined locally per
#       MEMORY codegen rule 19.
# ─────────────────────────────────────────────────────────────────────────────

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
    gesture_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten grouped_windows into (X, y) arrays.

    Args:
        grouped_windows: Dict[gesture_id → list of rep arrays (N_rep, T, C)]
        gesture_ids:     ordered gesture IDs (define class indices by position)
    Returns:
        X: (N, T, C), y: (N,) int64 class indices
    """
    X_parts, y_parts = [], []
    for cls_idx, gid in enumerate(gesture_ids):
        if gid not in grouped_windows:
            continue
        for rep in grouped_windows[gid]:
            if isinstance(rep, np.ndarray) and len(rep) > 0:
                X_parts.append(rep)
                y_parts.append(np.full(len(rep), cls_idx, dtype=np.int64))
    if not X_parts:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# LOSO-compliant split builder
# ─────────────────────────────────────────────────────────────────────────────

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
    Build train / val / test splits respecting the LOSO protocol.

    LOSO guarantees:
        • Only train_subjects contribute to train and val splits.
        • test_subject contributes ONLY to the test split.
        • No test-subject windows are ever seen during training.
        • Val split is drawn from training subjects (random hold-out per gesture).

    Returns:
        {
            "train": Dict[int, np.ndarray],   # gesture_id → (N, T, C)
            "val":   Dict[int, np.ndarray],
            "test":  Dict[int, np.ndarray],
        }
    """
    rng = np.random.RandomState(seed)

    # ── Collect windows per gesture from all training subjects ────────────
    raw_train: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}

    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        # subjects_data values are tuples (emg, segments, grouped_windows)
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        for gid in common_gestures:
            if gid not in filtered:
                continue
            for rep in filtered[gid]:
                if isinstance(rep, np.ndarray) and len(rep) > 0:
                    raw_train[gid].append(rep)

    # ── Split each gesture's training windows into train / val ────────────
    final_train: Dict[int, np.ndarray] = {}
    final_val: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not raw_train[gid]:
            continue
        X_gid = np.concatenate(raw_train[gid], axis=0)    # (N, T, C)
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        final_val[gid] = X_gid[perm[:n_val]]
        final_train[gid] = X_gid[perm[n_val:]]

    # ── Test split from test_subject only ─────────────────────────────────
    # LOSO: test-subject data touches ONLY this dict; never seen at training.
    test_dict: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in test_filtered.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                test_dict[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": test_dict}


# ─────────────────────────────────────────────────────────────────────────────
# Single LOSO fold
# ─────────────────────────────────────────────────────────────────────────────

def run_single_loso_fold(
    test_subject: str,
    train_subjects: List[str],
    base_dir: Path,
    output_root: Path,
    fold_idx: int,
    total_folds: int,
) -> Optional[Dict]:
    """
    Run one LOSO fold: train on train_subjects, evaluate on test_subject.

    Returns result dict on success, None on failure.
    """
    fold_dir = output_root / f"fold_{test_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(fold_dir)

    logger.info(
        f"=== Fold {fold_idx}/{total_folds} | test={test_subject} "
        f"| train_n={len(train_subjects)} ==="
    )
    logger.info(f"  Train subjects: {sorted(train_subjects)}")

    try:
        seed_everything(SEED)

        # ── Processing config ─────────────────────────────────────────────
        proc_cfg = ProcessingConfig(
            window_size=WINDOW_SIZE,
            window_overlap=WINDOW_OVERLAP,
            num_channels=NUM_CHANNELS,
            sampling_rate=2000,
            segment_edge_margin=0.1,
        )

        # ── Load data ─────────────────────────────────────────────────────
        # All subjects (train + test) are loaded together; the LOSO partition
        # is applied strictly in _build_splits: test windows go to test only.
        multi_loader = MultiSubjectLoader(
            processing_config=proc_cfg,
            logger=logger,
            use_gpu=False,
            use_improved_processing=True,
        )
        all_subjects = sorted(train_subjects) + [test_subject]
        subjects_data = multi_loader.load_multiple_subjects(
            base_dir=base_dir,
            subject_ids=all_subjects,
            exercises=EXERCISES,
            include_rest=False,
        )

        # ── Common gestures ───────────────────────────────────────────────
        common_gestures = multi_loader.get_common_gestures(
            subjects_data, max_gestures=MAX_GESTURES
        )
        logger.info(f"  Common gestures ({len(common_gestures)}): {common_gestures}")
        if len(common_gestures) < 2:
            logger.error("Fewer than 2 common gestures — skipping fold.")
            return None

        # ── Build LOSO-compliant splits ───────────────────────────────────
        splits = _build_splits(
            subjects_data=subjects_data,
            train_subjects=train_subjects,
            test_subject=test_subject,
            common_gestures=common_gestures,
            multi_loader=multi_loader,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )

        # Sanity check: test split must be non-empty
        n_test = sum(len(v) for v in splits["test"].values())
        if n_test == 0:
            logger.error(f"Empty test split for {test_subject} — skipping fold.")
            return None

        n_train = sum(len(v) for v in splits["train"].values())
        n_val = sum(len(v) for v in splits["val"].values())
        logger.info(
            f"  Windows — train={n_train}, val={n_val}, test={n_test}"
        )

        # ── Training config ───────────────────────────────────────────────
        train_cfg = TrainingConfig(
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            dropout=DROPOUT,
            early_stopping_patience=EARLY_STOP_PATIENCE,
            use_class_weights=True,
            seed=SEED,
            num_workers=0,
        )

        # ── Trainer ───────────────────────────────────────────────────────
        visualizer = Visualizer(fold_dir / "plots", logger)
        trainer = MRBTTrainer(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=fold_dir,
            visualizer=visualizer,
            # Architecture
            n_groups=N_GROUPS,
            group_channels=GROUP_CHANNELS,
            n_scales=N_SCALES,
            dilations=DILATIONS,
            embedding_dim=EMBEDDING_DIM,
            res2_scale=RES2_SCALE,
            se_reduction=SE_REDUCTION,
            # Loss
            lambda_bt=LAMBDA_BT,
            bt_warmup_frac=BT_WARMUP_FRAC,
        )

        # ── Train ─────────────────────────────────────────────────────────
        # fit() trains on splits["train"] (training subjects only).
        # BT loss uses only training-batch content/style features.
        # Val uses content pathway only.
        # LOSO: test-subject windows are held out and never seen here.
        training_results = trainer.fit(splits)

        # ── Evaluate on test subject (content pathway only) ───────────────
        # LOSO: test_subject windows were never seen during training.
        # evaluate_numpy normalises with training mean_c/std_c.
        class_ids = trainer.class_ids

        # Assemble flat test arrays in class_ids order
        X_test_parts: List[np.ndarray] = []
        y_test_parts: List[np.ndarray] = []
        for cls_idx, gid in enumerate(class_ids):
            if gid in splits["test"]:
                arr = splits["test"][gid]
                X_test_parts.append(arr)
                y_test_parts.append(np.full(len(arr), cls_idx, dtype=np.int64))

        if not X_test_parts:
            logger.error("No test windows after assembling — skipping fold.")
            return None

        X_test_np = np.concatenate(X_test_parts, axis=0)
        y_test_np = np.concatenate(y_test_parts, axis=0)

        # evaluate_numpy expects (N, T, C); split arrays are (N, T, C) ✓
        test_metrics = trainer.evaluate_numpy(
            X_test_np, y_test_np,
            split_name="loso_test",
            visualize=True,
        )

        # ── Save fold results ─────────────────────────────────────────────
        val_res = training_results.get("val") or {}
        val_acc = val_res.get("accuracy")
        val_f1 = val_res.get("f1_macro")

        fold_result = {
            "test_subject": test_subject,
            "train_subjects": sorted(train_subjects),
            "num_gestures": len(class_ids),
            "class_ids": class_ids,
            "n_test_windows": int(len(y_test_np)),
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["report"],
        }
        with open(fold_dir / "fold_result.json", "w") as fh:
            json.dump(make_json_serializable(fold_result), fh, indent=2)

        acc = test_metrics["accuracy"]
        f1 = test_metrics["f1_macro"]
        # Guard against None before formatting (rule 2 in codegen rules)
        acc_str = f"{acc:.4f}" if acc is not None else "None"
        f1_str = f"{f1:.4f}" if f1 is not None else "None"
        logger.info(f"  Fold done — test_acc={acc_str}  test_f1={f1_str}")
        return fold_result

    except Exception:
        logger.error(f"Fold failed:\n{traceback.format_exc()}")
        return None
    finally:
        # Release GPU memory between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Main — LOSO loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    _parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    _parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Explicit list of subject IDs (overrides --ci / --full)",
    )
    _parser.add_argument(
        "--ci", action="store_true",
        help="Use CI test subjects (default behaviour when no flag given)",
    )
    _parser.add_argument(
        "--full", action="store_true",
        help="Use full DEFAULT_SUBJECTS list (20 subjects; only for local runs). "
             "NOTE: vast.ai server has symlinks only for CI subjects.",
    )
    _args, _ = _parser.parse_known_args()

    # Subject selection — safe default is CI_TEST_SUBJECTS.
    # --full must be explicit because the server only has CI symlinks.
    if _args.subjects:
        ALL_SUBJECTS = _args.subjects
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS   # safe default

    base_dir = ROOT / "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    main_logger = setup_logging(output_root)
    main_logger.info(f"Experiment : {EXPERIMENT_NAME}")
    main_logger.info(f"Subjects   : {ALL_SUBJECTS}")
    main_logger.info(f"Output     : {output_root}")
    main_logger.info(
        f"Architecture: n_groups={N_GROUPS}, group_channels={GROUP_CHANNELS}, "
        f"n_scales={N_SCALES}, dilations={DILATIONS}, "
        f"embedding_dim={EMBEDDING_DIM}, res2_scale={RES2_SCALE}"
    )
    main_logger.info(
        f"Loss: lambda_bt={LAMBDA_BT}, bt_warmup_frac={BT_WARMUP_FRAC}"
    )
    main_logger.info(
        f"Training: bs={BATCH_SIZE}, epochs={NUM_EPOCHS}, "
        f"lr={LR}, patience={EARLY_STOP_PATIENCE}"
    )

    seed_everything(SEED)

    fold_results = []
    total_folds = len(ALL_SUBJECTS)

    for fold_idx, test_subject in enumerate(ALL_SUBJECTS, start=1):
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        result = run_single_loso_fold(
            test_subject=test_subject,
            train_subjects=train_subjects,
            base_dir=base_dir,
            output_root=output_root,
            fold_idx=fold_idx,
            total_folds=total_folds,
        )
        if result is not None:
            fold_results.append(result)

    # ── Aggregate LOSO summary ─────────────────────────────────────────────
    if fold_results:
        accs = [r["test_accuracy"] for r in fold_results]
        f1s = [r["test_f1_macro"] for r in fold_results]
        summary = {
            "experiment": EXPERIMENT_NAME,
            "approach": APPROACH,
            "num_folds": len(fold_results),
            "subjects": ALL_SUBJECTS,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "per_subject_accuracy": {
                r["test_subject"]: r["test_accuracy"] for r in fold_results
            },
            "per_subject_f1": {
                r["test_subject"]: r["test_f1_macro"] for r in fold_results
            },
            "fold_results": fold_results,
            "hyperparameters": {
                "exercises": EXERCISES,
                "max_gestures": MAX_GESTURES,
                "num_channels": NUM_CHANNELS,
                "window_size": WINDOW_SIZE,
                "window_overlap": WINDOW_OVERLAP,
                "val_ratio": VAL_RATIO,
                "n_groups": N_GROUPS,
                "group_channels": GROUP_CHANNELS,
                "n_scales": N_SCALES,
                "dilations": DILATIONS,
                "embedding_dim": EMBEDDING_DIM,
                "res2_scale": RES2_SCALE,
                "se_reduction": SE_REDUCTION,
                "lambda_bt": LAMBDA_BT,
                "bt_warmup_frac": BT_WARMUP_FRAC,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "early_stopping_patience": EARLY_STOP_PATIENCE,
                "seed": SEED,
            },
        }
        with open(output_root / "loso_summary.json", "w") as fh:
            json.dump(make_json_serializable(summary), fh, indent=2)

        main_logger.info("=" * 60)
        main_logger.info(f"LOSO complete — {len(fold_results)}/{total_folds} folds")
        main_logger.info(
            f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}"
        )
        main_logger.info(
            f"  F1 macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        )
        main_logger.info(f"  Output   : {output_root}")
    else:
        main_logger.error("All folds failed — no results to aggregate.")

    # ── Notify hypothesis_executor if available ────────────────────────────
    # Wrapped in try/except: hypothesis_executor may not be installed
    # (plain LOSO runs without the research system).
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        hyp_id = os.environ.get("HYPOTHESIS_ID", "exp_92_mrbt_channel_group")
        if fold_results:
            best_metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_folds": len(fold_results),
            }
            mark_hypothesis_verified(hyp_id, best_metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed(hyp_id, "All LOSO folds failed.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
