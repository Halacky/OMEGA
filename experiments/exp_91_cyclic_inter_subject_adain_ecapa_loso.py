"""
exp_91 — Cyclic Inter-Subject Reconstruction with multi-scale AdaIN (DSDRNet + ECAPA2)

Hypothesis 4:
    DSDRNet-style cyclic reconstruction combined with per-scale AdaIN on
    ECAPA-TDNN features.  At each SE-Res2Net level the subject-specific
    statistics of a style window are transferred onto the content of a
    different-subject window, creating an augmented view that retains the
    gesture pattern but adopts the style subject's amplitude/noise/tempo
    characteristics.  A cycle-consistency loss (A→B-style→A-style ≈ A)
    provides a self-supervised signal without explicit style labels.

    At inference ONLY the content pathway is active; AdaIN and the decoder
    are discarded — guaranteeing strict LOSO compliance.

LOSO Protocol (strict — NO adaptation to test subject):
    For each subject s in ALL_SUBJECTS:
        train subjects = ALL_SUBJECTS \ {s}
        test subject   = s
        • All data loading, normalisation, and style pairs use train subjects only.
        • The model is evaluated on s's data using the content pathway only.
        • No test-subject windows are ever seen during training.
        • No test-subject statistics are used for AdaIN or normalisation.

Usage:
    python experiments/exp_91_cyclic_inter_subject_adain_ecapa_loso.py
    python experiments/exp_91_cyclic_inter_subject_adain_ecapa_loso.py --ci
    python experiments/exp_91_cyclic_inter_subject_adain_ecapa_loso.py --full
    python experiments/exp_91_cyclic_inter_subject_adain_ecapa_loso.py \\
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
from training.dsdrnet_ecapa_adain_trainer import DSDRNetECAPATrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = "exp_91_cyclic_inter_subject_adain_ecapa"
APPROACH        = "deep_raw"
EXERCISES       = ["E1"]
MAX_GESTURES    = 10

# ── ECAPA architecture ──────────────────────────────────────────────────────
ECAPA_CHANNELS      = 128
ECAPA_SCALE         = 4
ECAPA_EMBEDDING_DIM = 128
ECAPA_DILATIONS     = [2, 3, 4]
ECAPA_SE_REDUCTION  = 8
DECODER_HIDDEN      = 256

# ── Loss weights ────────────────────────────────────────────────────────────
LAMBDA_CYCLE        = 1.0
LAMBDA_SELF_RECON   = 0.5
LAMBDA_PERCEPTUAL   = 0.1
CYCLE_WARMUP_FRAC   = 0.20   # ramp reconstruction losses over first 20% of epochs

# ── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE          = 64     # smaller than 256 default — heavier per-batch computation
NUM_EPOCHS          = 100
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
DROPOUT             = 0.3
EARLY_STOP_PATIENCE = 15     # longer patience — cycle warmup delays early benefits
VAL_RATIO           = 0.15
SEED                = 42


# ─────────────────────────────────────────────────────────────────────────────
# Local helper: grouped_windows → flat (windows, labels) arrays
# NOTE: this helper does NOT exist in any processing/ module — must be defined
#       locally in every experiment that needs it (see MEMORY rule 19).
# ─────────────────────────────────────────────────────────────────────────────

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
    gesture_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten a grouped_windows dict into (X, y) arrays for a given gesture set.

    Args:
        grouped_windows: Dict[gesture_id → list of rep arrays, each (N_rep, T, C)]
        gesture_ids:     ordered list of gesture IDs to include (defines class indices)
    Returns:
        X: (N, T, C)  — all windows concatenated
        y: (N,)       — class indices (position of gesture_id in gesture_ids)
    """
    X_parts, y_parts = [], []
    for cls_idx, gid in enumerate(gesture_ids):
        if gid not in grouped_windows:
            continue
        reps = grouped_windows[gid]
        for rep in reps:
            if isinstance(rep, np.ndarray) and len(rep) > 0:
                X_parts.append(rep)
                y_parts.append(np.full(len(rep), cls_idx, dtype=np.int64))
    if not X_parts:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Split builder — includes subject labels for training data
# ─────────────────────────────────────────────────────────────────────────────

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
    Build train / val / test splits including per-window subject indices for training.

    LOSO compliance:
      • Only train_subjects contribute to train and val splits.
      • test_subject contributes ONLY to the test split.
      • Subject labels (0 … N_train−1) are integer indices into sorted(train_subjects).
        They are used exclusively to form cross-subject style pairs during training.

    Returns:
        {
            "train":                Dict[int, np.ndarray],   # gesture_id → (N, T, C)
            "val":                  Dict[int, np.ndarray],
            "test":                 Dict[int, np.ndarray],
            "train_subject_labels": Dict[int, np.ndarray],   # gesture_id → (N,) int
            "num_train_subjects":   int,
        }
    """
    rng = np.random.RandomState(seed)

    # Integer index for each training subject (used as style-pair labels)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects   = len(train_subjects)

    # ── Collect windows per gesture from all training subjects ────────────
    raw_train:      Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}
    raw_train_subj: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}

    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[sid]   # unpack tuple — NOT a dict
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        subj_idx = train_subject_to_idx[sid]

        for gid in common_gestures:
            if gid not in filtered:
                continue
            for rep in filtered[gid]:
                if isinstance(rep, np.ndarray) and len(rep) > 0:
                    raw_train[gid].append(rep)
                    raw_train_subj[gid].append(
                        np.full(len(rep), subj_idx, dtype=np.int64)
                    )

    # ── Concatenate and split each gesture into train / val ───────────────
    final_train:      Dict[int, np.ndarray] = {}
    final_val:        Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not raw_train[gid]:
            continue
        X_gid = np.concatenate(raw_train[gid],      axis=0)  # (N, T, C)
        S_gid = np.concatenate(raw_train_subj[gid], axis=0)  # (N,)
        n     = len(X_gid)
        perm  = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))

        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        final_train[gid]      = X_gid[train_idx]
        final_val[gid]        = X_gid[val_idx]
        final_train_subj[gid] = S_gid[train_idx]

    # ── Test split from test_subject only ─────────────────────────────────
    # LOSO: test-subject data goes ONLY into the test split and is never
    # seen by the model during training.
    test_dict: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
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


# ─────────────────────────────────────────────────────────────────────────────
# Single LOSO fold
# ─────────────────────────────────────────────────────────────────────────────

def run_single_loso_fold(
    test_subject:   str,
    train_subjects: List[str],
    base_dir:       Path,
    output_root:    Path,
    fold_idx:       int,
    total_folds:    int,
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

        # ── Load data ────────────────────────────────────────────────────
        # All subjects loaded together; subject labels are separated in _build_splits.
        # LOSO compliance: load() is called once for all subjects, but training and
        # test data are then strictly partitioned — the model never sees test_subject
        # data during training.
        proc_cfg = ProcessingConfig(
            window_size        = 600,
            window_overlap     = 300,
            num_channels       = 8,
            sampling_rate      = 2000,
            segment_edge_margin = 0.1,
        )
        multi_loader  = MultiSubjectLoader(
            processing_config       = proc_cfg,
            logger                  = logger,
            use_gpu                 = False,
            use_improved_processing = True,
        )
        all_subjects  = sorted(train_subjects) + [test_subject]
        subjects_data = multi_loader.load_multiple_subjects(
            base_dir    = base_dir,
            subject_ids = all_subjects,
            exercises   = EXERCISES,
            include_rest = False,
        )

        # ── Common gestures across ALL subjects ───────────────────────────
        common_gestures = multi_loader.get_common_gestures(
            subjects_data, max_gestures=MAX_GESTURES
        )
        logger.info(f"  Common gestures ({len(common_gestures)}): {common_gestures}")
        if len(common_gestures) < 2:
            logger.error("Fewer than 2 common gestures — skipping fold.")
            return None

        # ── Build splits with subject labels ─────────────────────────────
        splits = _build_splits_with_subject_labels(
            subjects_data  = subjects_data,
            train_subjects = train_subjects,
            test_subject   = test_subject,
            common_gestures = common_gestures,
            multi_loader   = multi_loader,
            val_ratio      = VAL_RATIO,
            seed           = SEED,
        )

        # Sanity check: test split must be non-empty
        n_test = sum(len(v) for v in splits["test"].values())
        if n_test == 0:
            logger.error(f"Empty test split for {test_subject} — skipping fold.")
            return None

        logger.info(
            f"  Split sizes — "
            f"train_gestures={len(splits['train'])}, "
            f"val_gestures={len(splits['val'])}, "
            f"test_gestures={len(splits['test'])}, "
            f"num_train_subjects={splits['num_train_subjects']}"
        )

        # ── Training configuration ────────────────────────────────────────
        train_cfg = TrainingConfig(
            batch_size             = BATCH_SIZE,
            epochs                 = NUM_EPOCHS,
            learning_rate          = LR,
            weight_decay           = WEIGHT_DECAY,
            dropout                = DROPOUT,
            early_stopping_patience = EARLY_STOP_PATIENCE,
            use_class_weights      = True,
            seed                   = SEED,
            num_workers            = 0,
        )

        # ── Trainer ───────────────────────────────────────────────────────
        visualizer = Visualizer(fold_dir / "plots", logger)
        trainer = DSDRNetECAPATrainer(
            train_cfg         = train_cfg,
            logger            = logger,
            output_dir        = fold_dir,
            visualizer        = visualizer,
            # ECAPA architecture
            channels          = ECAPA_CHANNELS,
            scale             = ECAPA_SCALE,
            embedding_dim     = ECAPA_EMBEDDING_DIM,
            dilations         = ECAPA_DILATIONS,
            se_reduction      = ECAPA_SE_REDUCTION,
            decoder_hidden    = DECODER_HIDDEN,
            # Loss weights
            lambda_cycle      = LAMBDA_CYCLE,
            lambda_self_recon = LAMBDA_SELF_RECON,
            lambda_perceptual = LAMBDA_PERCEPTUAL,
            cycle_warmup_frac = CYCLE_WARMUP_FRAC,
        )

        # ── Train ─────────────────────────────────────────────────────────
        # fit() trains on splits["train"] (training subjects only).
        # It computes mean_c/std_c from training windows, forms cross-subject
        # style pairs within each batch, and evaluates val via content pathway.
        training_results = trainer.fit(splits)

        # ── Evaluate on test subject (content pathway only) ───────────────
        # LOSO compliance: test_subject's windows were held out during training.
        # evaluate_numpy() uses only the content pathway — no AdaIN, no decoder.
        # Normalisation uses training mean_c/std_c (set in fit()).
        class_ids = trainer.class_ids

        # Assemble flat test arrays in class_ids order (same as _prepare_splits)
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

        X_test_concat = np.concatenate(X_test_parts, axis=0)
        y_test_concat = np.concatenate(y_test_parts, axis=0)

        test_metrics = trainer.evaluate_numpy(
            X_test_concat, y_test_concat,
            split_name="loso_test",
            visualize=True,
        )

        # ── Save fold results ─────────────────────────────────────────────
        fold_result = {
            "test_subject":     test_subject,
            "train_subjects":   sorted(train_subjects),
            "num_gestures":     len(class_ids),
            "class_ids":        class_ids,
            "n_test_windows":   len(y_test_concat),
            "test_accuracy":    test_metrics["accuracy"],
            "test_f1_macro":    test_metrics["f1_macro"],
            "val_accuracy":     (training_results["val"] or {}).get("accuracy"),
            "val_f1_macro":     (training_results["val"] or {}).get("f1_macro"),
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["report"],
        }
        with open(fold_dir / "fold_result.json", "w") as fh:
            json.dump(make_json_serializable(fold_result), fh, indent=2)

        acc = test_metrics["accuracy"]
        f1  = test_metrics["f1_macro"]
        logger.info(
            f"  ✓ Fold done — test_acc={acc:.4f}  test_f1={f1:.4f}"
        )
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
        help="Use CI test subjects (default behaviour)",
    )
    _parser.add_argument(
        "--full", action="store_true",
        help="Use full DEFAULT_SUBJECTS list (20 subjects; only for local runs)",
    )
    _args, _ = _parser.parse_known_args()

    # Subject selection — default to CI subjects.
    # _FULL_SUBJECTS requires explicit --full because the server (vast.ai)
    # has symlinks only for CI subjects; loading others causes FileNotFoundError.
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
        f"Architecture: ECAPA channels={ECAPA_CHANNELS}, "
        f"emb={ECAPA_EMBEDDING_DIM}, dilations={ECAPA_DILATIONS}"
    )
    main_logger.info(
        f"Losses: λ_cycle={LAMBDA_CYCLE}, λ_self={LAMBDA_SELF_RECON}, "
        f"λ_perc={LAMBDA_PERCEPTUAL}, warmup_frac={CYCLE_WARMUP_FRAC}"
    )
    main_logger.info(
        f"Training: bs={BATCH_SIZE}, epochs={NUM_EPOCHS}, lr={LR}, "
        f"patience={EARLY_STOP_PATIENCE}"
    )

    seed_everything(SEED)

    fold_results = []
    total_folds  = len(ALL_SUBJECTS)

    for fold_idx, test_subject in enumerate(ALL_SUBJECTS, start=1):
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        result = run_single_loso_fold(
            test_subject   = test_subject,
            train_subjects = train_subjects,
            base_dir       = base_dir,
            output_root    = output_root,
            fold_idx       = fold_idx,
            total_folds    = total_folds,
        )
        if result is not None:
            fold_results.append(result)

    # ── Aggregate LOSO summary ─────────────────────────────────────────────
    if fold_results:
        accs = [r["test_accuracy"] for r in fold_results]
        f1s  = [r["test_f1_macro"] for r in fold_results]
        summary = {
            "experiment":          EXPERIMENT_NAME,
            "approach":            APPROACH,
            "num_folds":           len(fold_results),
            "subjects":            ALL_SUBJECTS,
            "mean_accuracy":       float(np.mean(accs)),
            "std_accuracy":        float(np.std(accs)),
            "mean_f1_macro":       float(np.mean(f1s)),
            "std_f1_macro":        float(np.std(f1s)),
            "per_subject_accuracy": {r["test_subject"]: r["test_accuracy"] for r in fold_results},
            "per_subject_f1":      {r["test_subject"]: r["test_f1_macro"]  for r in fold_results},
            "fold_results":        fold_results,
            "hyperparameters": {
                "exercises":            EXERCISES,
                "max_gestures":         MAX_GESTURES,
                "val_ratio":            VAL_RATIO,
                "ecapa_channels":       ECAPA_CHANNELS,
                "ecapa_scale":          ECAPA_SCALE,
                "ecapa_embedding_dim":  ECAPA_EMBEDDING_DIM,
                "ecapa_dilations":      ECAPA_DILATIONS,
                "ecapa_se_reduction":   ECAPA_SE_REDUCTION,
                "decoder_hidden":       DECODER_HIDDEN,
                "lambda_cycle":         LAMBDA_CYCLE,
                "lambda_self_recon":    LAMBDA_SELF_RECON,
                "lambda_perceptual":    LAMBDA_PERCEPTUAL,
                "cycle_warmup_frac":    CYCLE_WARMUP_FRAC,
                "batch_size":           BATCH_SIZE,
                "num_epochs":           NUM_EPOCHS,
                "learning_rate":        LR,
                "weight_decay":         WEIGHT_DECAY,
                "dropout":              DROPOUT,
                "early_stopping_patience": EARLY_STOP_PATIENCE,
                "seed":                 SEED,
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
    # Wrapped in try/except because hypothesis_executor may not be installed
    # in all environments (e.g. plain LOSO runs without the research system).
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        hyp_id = os.environ.get("HYPOTHESIS_ID", "exp_91_cyclic_adain_ecapa")
        if fold_results:
            best_metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
                "num_folds":     len(fold_results),
            }
            mark_hypothesis_verified(hyp_id, best_metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed(hyp_id, "All LOSO folds failed.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
