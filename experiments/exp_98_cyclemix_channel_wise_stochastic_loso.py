"""
Experiment 98: CycleMix со стохастической магнитудой стилевых компонент

Hypothesis H98:
    Заменить текущее попарное Beta-смешивание стилей на циклическое перемешивание
    (CycleMix), где в каждом минибатче стилевые статистики (μ, σ) каждого канала
    рекомбинируются с рандомизированной магнитудой.

Mechanism (LOSO-safe):
    1. Encoder → z_content (128d) + z_style (64d).
    2. z_style разбивается на 8 групп по 8 dims (по одной на EMG-канал).
    3. Для каждой группы k независимо выбирается донор j_k из других training subjects.
       Виртуальных стилей: 4^8 = 65,536 vs 6 пар в MixStyle (exp_60).
    4. λ_k ~ Beta(α, α), α ~ Uniform(0.1, 0.5) один раз за epoch.
    5. style_mix[k] = λ_k · z_style[k] + (1-λ_k) · z_donor_j_k[k]
    6. FiLM(z_content, z_style_mix) → gesture_classifier (тренировка).
    7. При инференсе: только gesture_classifier(z_content) без FiLM.

Loss:
    L = CE(base_path, y) + γ·CE(mix_path, y) + α_s·CE(subject_head, y_s)
      + β(t)·DistCorr(z_content, z_style)

LOSO data-leakage audit:
    ✓ Стиль-смешение происходит только внутри training batch (train subjects only).
    ✓ Тест-субъект загружается только в splits["test"], никогда в DataLoader обучения.
    ✓ subject_labels — индексы [0, N_train-1] только train-субъектов.
    ✓ Нормализация считается только по train windows.
    ✓ Validation = subset from train subjects; early stopping по val loss.
    ✓ Inference: model(x) в eval mode → gesture_logits_base (без FiLM, без z_style).
    ✓ Нет test-time adaptation.

Baseline comparison:
    exp_60 (MixStyle):             F1 ≈ 35% (6 pairs, fixed mix_alpha=0.4)
    exp_62 (ECAPA-TDNN, текущий лидер): F1 = 35.59%
    Target (H98):                  F1 > 36.5%, std < 5%

Usage:
    python experiments/exp_98_cyclemix_channel_wise_stochastic_loso.py
    python experiments/exp_98_cyclemix_channel_wise_stochastic_loso.py --ci
    python experiments/exp_98_cyclemix_channel_wise_stochastic_loso.py --full
    python experiments/exp_98_cyclemix_channel_wise_stochastic_loso.py \\
        --subjects DB2_s1,DB2_s12,DB2_s15,DB2_s28,DB2_s39
"""

import gc
import json
import os
import sys
import traceback
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
from config.base import ProcessingConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.cyclemix_disentangled_trainer import CycleMixDisentangledTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything

# ════════════════════════ EXPERIMENT CONSTANTS ════════════════════════════

EXPERIMENT_NAME = "exp_98_cyclemix_channel_wise_stochastic"
APPROACH        = "deep_raw"
EXERCISES       = ["E1"]
MAX_GESTURES    = 10

# ── Disentanglement (same as exp_60 baseline for fair comparison) ─────────
CONTENT_DIM = 128
STYLE_DIM   = 64    # 8 channels × 8 dims/channel — divisible by num_channels=8

# ── Loss weights ─────────────────────────────────────────────────────────
ALPHA_SUBJ         = 0.5    # weight of subject classification loss
BETA_MI            = 0.1    # weight of MI minimisation loss (annealed)
GAMMA              = 0.5    # weight of mixed-style gesture path
BETA_ANNEAL_EPOCHS = 10
MI_LOSS_TYPE       = "distance_correlation"

# ── CycleMix-specific ────────────────────────────────────────────────────
ALPHA_LOW  = 0.1    # lower bound: α ~ Uniform(ALPHA_LOW, ALPHA_HIGH) per epoch
ALPHA_HIGH = 0.5    # upper bound

# ── Training ─────────────────────────────────────────────────────────────
BATCH_SIZE          = 64
NUM_EPOCHS          = 60
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
DROPOUT             = 0.3
EARLY_STOP_PATIENCE = 12
VAL_RATIO           = 0.15
SEED                = 42


# ════════════════════════ LOCAL HELPERS ══════════════════════════════════

def grouped_to_arrays(
    grouped_windows: Dict,
    gesture_ids: List[int],
):
    """
    Flatten a grouped_windows dict into (X, y) arrays.

    NOTE: grouped_to_arrays does NOT exist in any processing/ module — must be
    defined locally in every experiment (MEMORY rule 19).

    Args:
        grouped_windows: Dict[gesture_id → list of rep arrays (N_rep, T, C)]
        gesture_ids:     ordered gesture IDs (defines class indices)
    Returns:
        X: (N, T, C), y: (N,) int64
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


# ════════════════════════ SPLIT BUILDER ══════════════════════════════════

def _build_splits_with_subject_labels(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> Dict:
    """
    Build train / val / test splits with per-window subject provenance labels.

    Subject labels are required by CycleMixDisentangledTrainer to form
    cross-subject style pairs within each training batch.

    LOSO guarantee:
        • Only train_subjects contribute to train and val splits.
        • test_subject contributes ONLY to the test split.
        • Subject labels are integer indices into sorted(train_subjects).
          The test subject has no index and cannot appear in any style mix.

    Returns:
        {
            "train":                Dict[gesture_id → (N, T, C) array]
            "val":                  Dict[gesture_id → (N, T, C) array]
            "test":                 Dict[gesture_id → (N, T, C) array]
            "train_subject_labels": Dict[gesture_id → (N,) int array]
            "num_train_subjects":   int
        }
    """
    rng = np.random.RandomState(seed)

    # Integer index per training subject (test subject intentionally absent)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects   = len(train_subjects)

    # ── Collect per-gesture windows from training subjects ────────────────
    raw_train:      Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}
    raw_train_subj: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}

    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[sid]   # tuple, NOT dict (MEMORY rule 9)
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
    # LOSO boundary: test-subject windows go ONLY here, never into train or val.
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


# ════════════════════════ SINGLE LOSO FOLD ═══════════════════════════════

def run_single_loso_fold(
    test_subject:   str,
    train_subjects: List[str],
    base_dir:       Path,
    output_root:    Path,
    fold_idx:       int,
    total_folds:    int,
) -> Optional[Dict]:
    """
    Execute one LOSO fold: train on train_subjects, evaluate on test_subject.

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
            window_size=600,
            window_overlap=300,
            num_channels=8,
            sampling_rate=2000,
            segment_edge_margin=0.1,
        )

        # ── Load data for all subjects in one pass ────────────────────────
        # LOSO: the hard boundary is enforced in _build_splits_with_subject_labels,
        # which routes test_subject windows exclusively into splits["test"].
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

        # ── Common gestures across all subjects ───────────────────────────
        common_gestures = multi_loader.get_common_gestures(
            subjects_data, max_gestures=MAX_GESTURES
        )
        logger.info(f"  Common gestures ({len(common_gestures)}): {common_gestures}")
        if len(common_gestures) < 2:
            logger.error("Fewer than 2 common gestures — skipping fold.")
            return None

        # ── Build LOSO-safe splits with subject provenance labels ─────────
        splits = _build_splits_with_subject_labels(
            subjects_data=subjects_data,
            train_subjects=train_subjects,
            test_subject=test_subject,
            common_gestures=common_gestures,
            multi_loader=multi_loader,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )

        # Sanity check: test split must be non-empty
        n_test = sum(
            len(v) for v in splits["test"].values()
            if isinstance(v, np.ndarray) and v.ndim == 3
        )
        if n_test == 0:
            logger.error(f"Empty test split for {test_subject} — skipping fold.")
            return None

        for split_name in ("train", "val", "test"):
            total = sum(
                len(arr) for arr in splits[split_name].values()
                if isinstance(arr, np.ndarray) and arr.ndim == 3
            )
            logger.info(
                f"  {split_name.upper()}: {total} windows, "
                f"{len(splits[split_name])} gesture classes"
            )

        # ── Training configuration ────────────────────────────────────────
        train_cfg = TrainingConfig(
            model_type="cyclemix_disentangled_cnn_gru",
            pipeline_type=APPROACH,
            use_handcrafted_features=False,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            dropout=DROPOUT,
            early_stopping_patience=EARLY_STOP_PATIENCE,
            seed=SEED,
            use_class_weights=True,
            num_workers=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # ── Create trainer ────────────────────────────────────────────────
        visualizer = Visualizer(fold_dir / "plots", logger)
        trainer = CycleMixDisentangledTrainer(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=fold_dir,
            visualizer=visualizer,
            content_dim=CONTENT_DIM,
            style_dim=STYLE_DIM,
            alpha_subj=ALPHA_SUBJ,
            beta_mi=BETA_MI,
            gamma=GAMMA,
            beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
            mi_loss_type=MI_LOSS_TYPE,
            alpha_low=ALPHA_LOW,
            alpha_high=ALPHA_HIGH,
        )

        # ── Train ─────────────────────────────────────────────────────────
        # fit() trains on splits["train"] (training subjects only).
        # Per-channel CycleMix is applied inside model.forward() during training.
        # Validation uses the base path (no FiLM) — mirrors inference exactly.
        training_results = trainer.fit(splits)

        # ── Evaluate on held-out test subject ─────────────────────────────
        # LOSO: test_subject's windows were never seen during training.
        # evaluate_numpy() uses only z_content (base path, no FiLM, no style).
        # Normalisation uses training mean_c/std_c stored in trainer.fit().
        class_ids = trainer.class_ids

        # Assemble flat test arrays in class_ids order
        X_test_parts: List[np.ndarray] = []
        y_test_parts: List[np.ndarray] = []
        for cls_idx, gid in enumerate(class_ids):
            if gid in splits["test"]:
                arr = splits["test"][gid]
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                    X_test_parts.append(arr)
                    y_test_parts.append(np.full(len(arr), cls_idx, dtype=np.int64))

        if not X_test_parts:
            logger.error("No test windows after assembling — skipping fold.")
            return None

        X_test_concat = np.concatenate(X_test_parts, axis=0)
        y_test_concat = np.concatenate(y_test_parts, axis=0)

        # evaluate_numpy: transpose + standardise + eval mode → base path
        test_metrics = trainer.evaluate_numpy(
            X_test_concat, y_test_concat,
            split_name=f"loso_test_{test_subject}",
            visualize=True,
        )

        test_acc = test_metrics["accuracy"]
        test_f1  = test_metrics["f1_macro"]
        logger.info(
            f"  ✓ Fold done — "
            f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}"
        )

        # ── Save fold result ──────────────────────────────────────────────
        val_results = training_results.get("val") or {}
        fold_result = {
            "test_subject":          test_subject,
            "train_subjects":        sorted(train_subjects),
            "num_gestures":          len(class_ids),
            "class_ids":             class_ids,
            "n_test_windows":        len(y_test_concat),
            "test_accuracy":         test_acc,
            "test_f1_macro":         test_f1,
            "val_accuracy":          val_results.get("accuracy"),
            "val_f1_macro":          val_results.get("f1_macro"),
            "confusion_matrix":      test_metrics["confusion_matrix"],
            "classification_report": test_metrics["report"],
        }
        with open(fold_dir / "fold_result.json", "w") as fh:
            json.dump(make_json_serializable(fold_result), fh, indent=2)

        return fold_result

    except Exception:
        logger.error(f"Fold failed:\n{traceback.format_exc()}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ════════════════════════ MAIN ════════════════════════════════════════════

def main() -> None:
    import argparse

    _parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    _parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs, e.g. DB2_s1,DB2_s12",
    )
    _parser.add_argument(
        "--ci",
        action="store_true",
        help="Use CI test subjects (5 subjects, default behaviour)",
    )
    _parser.add_argument(
        "--full",
        action="store_true",
        help="Use full DEFAULT_SUBJECTS list (20 subjects; local only — server has only CI symlinks)",
    )
    _args, _ = _parser.parse_known_args()

    # Subject selection — default to CI_TEST_SUBJECTS (server-safe).
    # --full requires explicit flag because vast.ai server only has CI symlinks.
    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS   # safe default

    base_dir    = ROOT / "data"
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    main_logger = setup_logging(output_root)
    main_logger.info(f"Experiment    : {EXPERIMENT_NAME}")
    main_logger.info(f"Hypothesis    : H98 — CycleMix channel-wise stochastic style mixing")
    main_logger.info(f"Subjects      : {ALL_SUBJECTS}")
    main_logger.info(f"Exercises     : {EXERCISES}")
    main_logger.info(f"Output        : {output_root}")
    main_logger.info(
        f"Architecture  : content_dim={CONTENT_DIM}, style_dim={STYLE_DIM} "
        f"(8 channels × {STYLE_DIM // 8} dims/channel)"
    )
    main_logger.info(
        f"Loss weights  : α_subj={ALPHA_SUBJ}, β_MI={BETA_MI}, γ={GAMMA}, "
        f"β_anneal={BETA_ANNEAL_EPOCHS}ep"
    )
    main_logger.info(
        f"CycleMix α    : Uniform({ALPHA_LOW}, {ALPHA_HIGH}) per epoch"
    )
    main_logger.info(
        f"Training      : bs={BATCH_SIZE}, epochs={NUM_EPOCHS}, "
        f"lr={LR}, patience={EARLY_STOP_PATIENCE}"
    )

    seed_everything(SEED)

    fold_results = []
    total_folds  = len(ALL_SUBJECTS)

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

    # ── LOSO aggregate summary ─────────────────────────────────────────────
    if fold_results:
        accs = [r["test_accuracy"] for r in fold_results]
        f1s  = [r["test_f1_macro"]  for r in fold_results]

        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        mean_f1  = float(np.mean(f1s))
        std_f1   = float(np.std(f1s))

        print(f"\n{'=' * 60}")
        print(f"CycleMix Disentangled CNN-GRU — LOSO Summary ({len(fold_results)} folds)")
        print(f"  Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  F1-macro : {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Target   : F1 > 36.5% (vs exp_62 leader = 35.59%)")
        print(f"{'=' * 60}\n")

        summary = {
            "experiment":   EXPERIMENT_NAME,
            "hypothesis":   "H98: CycleMix channel-wise stochastic style mixing",
            "timestamp":    timestamp,
            "subjects":     ALL_SUBJECTS,
            "approach":     APPROACH,
            "num_folds":    len(fold_results),
            "hyperparameters": {
                "exercises":            EXERCISES,
                "max_gestures":         MAX_GESTURES,
                "content_dim":          CONTENT_DIM,
                "style_dim":            STYLE_DIM,
                "alpha_subj":           ALPHA_SUBJ,
                "beta_mi":              BETA_MI,
                "gamma":                GAMMA,
                "beta_anneal_epochs":   BETA_ANNEAL_EPOCHS,
                "mi_loss_type":         MI_LOSS_TYPE,
                "alpha_low":            ALPHA_LOW,
                "alpha_high":           ALPHA_HIGH,
                "batch_size":           BATCH_SIZE,
                "num_epochs":           NUM_EPOCHS,
                "learning_rate":        LR,
                "weight_decay":         WEIGHT_DECAY,
                "dropout":              DROPOUT,
                "early_stopping_patience": EARLY_STOP_PATIENCE,
                "seed":                 SEED,
                "val_ratio":            VAL_RATIO,
            },
            "aggregate": {
                "mean_accuracy": mean_acc,
                "std_accuracy":  std_acc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro":  std_f1,
            },
            "per_subject_accuracy": {r["test_subject"]: r["test_accuracy"] for r in fold_results},
            "per_subject_f1":       {r["test_subject"]: r["test_f1_macro"]  for r in fold_results},
            "fold_results":         fold_results,
        }
    else:
        main_logger.error("All folds failed — no results to aggregate.")
        summary = {
            "experiment": EXPERIMENT_NAME,
            "timestamp":  timestamp,
            "subjects":   ALL_SUBJECTS,
            "error":      "All folds failed",
        }

    with open(output_root / "loso_summary.json", "w") as fh:
        json.dump(make_json_serializable(summary), fh, indent=2)
    main_logger.info(f"Summary saved: {output_root / 'loso_summary.json'}")

    # ── Notify hypothesis_executor (optional dependency) ──────────────────
    # Guarded in try/except ImportError per MEMORY rule 23
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        hyp_id = os.environ.get("HYPOTHESIS_ID", "H98")
        if fold_results:
            best_metrics = {
                "mean_accuracy": mean_acc,
                "std_accuracy":  std_acc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro":  std_f1,
                "num_folds":     len(fold_results),
            }
            # mark_hypothesis_verified takes (hypothesis_id, metrics, experiment_name)
            mark_hypothesis_verified(hyp_id, best_metrics, experiment_name=EXPERIMENT_NAME)
        else:
            # mark_hypothesis_failed takes ONLY (hypothesis_id, error_message) — no metrics kwarg
            mark_hypothesis_failed(hyp_id, "All LOSO folds failed.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
