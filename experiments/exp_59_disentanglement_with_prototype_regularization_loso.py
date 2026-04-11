"""
Experiment 59: Disentanglement + Prototype Regularization for LOSO EMG.

Hypothesis:
  In cross-subject gesture recognition, gesture clusters in z_content should be
  compact regardless of subject identity.  EMA prototypes (per class, in z_content)
  computed from TRAIN data serve as stable anchors that regularize the encoder.

Architecture:
  Same DisentangledCNNGRU as exp_31:
    Input (B, C, T) → SharedEncoder → z_content (128-D) + z_style (64-D)
    Gesture classifier operates on z_content (subject-invariant).
    Subject classifier operates on z_style (adversarial disentanglement target).

Additional losses (prototype regularization):
  L_center  = mean_i  ||z_content_i - ema_proto[y_i]||^2
              → pulls samples toward class-specific EMA anchors
  L_push    = mean_{a<b} max(0, margin - ||batch_proto_a - batch_proto_b||_2)
              → pushes mini-batch class prototypes apart

Full objective:
  L = L_gesture + α·L_subject + β(t)·L_MI + γ(t)·L_center + δ(t)·L_push

Why this should work better than exp_36 (pure metric learning):
  - Strong CE + disentanglement provide the primary learning signal.
  - Prototype regularization is additive: it compresses intra-class variance and
    increases inter-class separation in z_content without replacing the classifier.
  - If the hypothesis is wrong, gamma=0 or delta=0 trivially recovers exp_31.

LOSO compliance (no data leakage):
  1. subjects_data is loaded for all subjects, but splits are built so that
     - splits["train"] / splits["val"] contain ONLY train-subject windows.
     - splits["test"] contains ONLY test-subject windows.
  2. EMA prototypes are updated ONLY in the training loop inside ProtoDisentangledTrainer.
     Validation data NEVER triggers a prototype update.
  3. No subject-specific adaptation occurs at test time.
  4. The diagnostic (intra-class variance by subject) is computed using ONLY
     train-subject data (never the held-out test subject).

Diagnostic metric:
  Intra-class variance of z_content across subjects:
    var(c) = mean_s ||mu_{c,s} - mu_c||^2
  Lower value → z_content is more subject-invariant for that gesture.

Usage:
  python experiments/exp_59_disentanglement_with_prototype_regularization_loso.py
  python experiments/exp_59_disentanglement_with_prototype_regularization_loso.py --ci
  python experiments/exp_59_disentanglement_with_prototype_regularization_loso.py \
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

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.proto_disentangled_trainer import ProtoDisentangledTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ═══════════════════════════ EXPERIMENT SETTINGS ════════════════════════════

EXPERIMENT_NAME = "exp_59_disentanglement_prototype_regularization"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# Disentanglement hyperparameters (same defaults as exp_31 for fair comparison)
CONTENT_DIM = 128
STYLE_DIM = 64
ALPHA = 0.5          # subject classifier loss weight
BETA = 0.1           # MI minimization weight
BETA_ANNEAL_EPOCHS = 10
MI_LOSS_TYPE = "distance_correlation"

# Prototype regularization hyperparameters
LAMBDA_CENTER = 0.10   # center loss weight
LAMBDA_PUSH = 0.05     # inter-class push loss weight
PUSH_MARGIN = 4.0      # minimum L2 distance between class prototypes
EMA_DECAY = 0.99       # EMA smoothing for prototype update
PROTO_ANNEAL_EPOCHS = 10  # ramp-up period for prototype losses

# Compute the z_content intra-class variance diagnostic after each fold.
# Set to False to skip (saves time on full 20-subject runs).
COMPUTE_DIAGNOSTICS = True


# ══════════════════════════════ DATA HELPERS ═════════════════════════════════


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
    Build train / val / test splits while tracking subject provenance.

    LOSO compliance:
      - Train and val windows come exclusively from train_subjects.
      - Test windows come exclusively from test_subject.
      - subjects_data may contain both (loaded once for efficiency), but only
        the appropriate keys are accessed in each branch below.

    Returns:
        {
          "train": Dict[gesture_id, (N, T, C) ndarray],
          "val":   Dict[gesture_id, (N, T, C) ndarray],
          "test":  Dict[gesture_id, (N, T, C) ndarray],
          "train_subject_labels": Dict[gesture_id, (N,) int64 ndarray],
          "num_train_subjects": int,
        }
    """
    rng = np.random.RandomState(seed)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}

    # ── Build per-gesture train arrays with subject provenance ───────────────
    raw_train: Dict[int, np.ndarray] = {}
    raw_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        win_parts, subj_parts = [], []
        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, grouped_windows = subjects_data[sid]  # unpack tuple
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid not in filtered:
                continue
            for rep_arr in filtered[gid]:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    win_parts.append(rep_arr)
                    subj_parts.append(
                        np.full(
                            len(rep_arr),
                            train_subject_to_idx[sid],
                            dtype=np.int64,
                        )
                    )
        if win_parts:
            raw_train[gid] = np.concatenate(win_parts, axis=0)
            raw_subj[gid] = np.concatenate(subj_parts, axis=0)

    # ── Random train / val split (same permutation for windows and subj labels)
    final_train: Dict[int, np.ndarray] = {}
    final_val: Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if gid not in raw_train:
            continue
        X = raw_train[gid]
        S = raw_subj[gid]
        n = len(X)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        final_train[gid] = X[train_idx]
        final_val[gid] = X[val_idx]
        final_train_subj[gid] = S[train_idx]

    # ── Test split from test_subject (ONLY) ─────────────────────────────────
    test_dict: Dict[int, np.ndarray] = {}
    _, _, test_gw = subjects_data[test_subject]  # unpack tuple
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
        if valid:
            test_dict[gid] = np.concatenate(valid, axis=0)

    return {
        "train": final_train,
        "val": final_val,
        "test": test_dict,
        "train_subject_labels": final_train_subj,
        "num_train_subjects": len(train_subjects),
    }


def _grouped_to_arrays(
    grouped_windows: Dict[int, list],
    gesture_to_class: Dict[int, int],
) -> tuple:
    """
    Convert grouped_windows (gesture_id -> list of rep arrays) to flat (X, y).

    Returns:
        X: (N, T, C) float32 ndarray
        y: (N,) int64 ndarray of class indices
    """
    X_parts, y_parts = [], []
    for gid in sorted(gesture_to_class.keys()):
        if gid not in grouped_windows:
            continue
        cls_idx = gesture_to_class[gid]
        for rep_arr in grouped_windows[gid]:
            if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                X_parts.append(rep_arr)
                y_parts.append(
                    np.full(len(rep_arr), cls_idx, dtype=np.int64)
                )
    if not X_parts:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def _compute_subject_variance_diagnostic(
    trainer: ProtoDisentangledTrainer,
    subjects_data: Dict,
    train_subjects: List[str],
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
) -> Optional[Dict]:
    """
    Compute intra-class variance of z_content across train subjects.

    LOSO compliance: passes ONLY train-subject data to the diagnostic method.
    The test subject (held out in this fold) is NEVER included here.

    Args:
        trainer:        Fitted ProtoDisentangledTrainer (after fit()).
        subjects_data:  Full subjects_data dict from load_multiple_subjects().
        train_subjects: List of subject IDs used for training in this fold.
        common_gestures: Common gesture IDs.
        multi_loader:   MultiSubjectLoader instance.

    Returns:
        dict with "intra_class_variance_by_class" and "mean_intra_class_variance",
        or None if fewer than 2 subjects are available.
    """
    assert trainer.class_ids is not None, "trainer must be fitted first"

    gesture_to_class = {
        gid: i for i, gid in enumerate(trainer.class_ids)
    }

    subj_windows: Dict[str, np.ndarray] = {}
    subj_labels: Dict[str, np.ndarray] = {}

    for sid in train_subjects:
        if sid not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[sid]  # unpack tuple
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        X, y = _grouped_to_arrays(filtered, gesture_to_class)
        if len(X) == 0:
            continue

        # Apply the same preprocessing as trainer.fit():
        #   (N, T, C) → (N, C, T) if T > C, then channel standardization.
        if X.ndim == 3:
            _N, d1, d2 = X.shape
            if d1 > d2:
                X = np.transpose(X, (0, 2, 1))
        X = trainer._apply_standardization(X, trainer.mean_c, trainer.std_c)

        subj_windows[sid] = X
        subj_labels[sid] = y

    if len(subj_windows) < 2:
        return None

    return trainer.compute_content_variance_by_subject(subj_windows, subj_labels)


# ══════════════════════════════ LOSO FOLD ════════════════════════════════════


def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    compute_diagnostics: bool = True,
) -> Dict:
    """
    Single LOSO fold: train on train_subjects, evaluate on test_subject.

    No adaptation to the test subject occurs at any stage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "proto_disentangled_cnn_gru"

    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # ── Data loading ─────────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

    # Load all subjects in one call (efficient), but splits are built separately
    # so test-subject data never leaks into train/val.
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

    # ── Build splits (LOSO-compliant) ────────────────────────────────────────
    splits = _build_splits_with_subject_labels(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    for sname in ("train", "val", "test"):
        total = sum(
            len(arr)
            for arr in splits[sname].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"{sname.upper()}: {total} windows, "
            f"{len(splits[sname])} gesture classes"
        )

    # ── Create trainer ────────────────────────────────────────────────────────
    base_viz = Visualizer(output_dir, logger)

    trainer = ProtoDisentangledTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        content_dim=CONTENT_DIM,
        style_dim=STYLE_DIM,
        alpha=ALPHA,
        beta=BETA,
        beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
        mi_loss_type=MI_LOSS_TYPE,
        lambda_center=LAMBDA_CENTER,
        lambda_push=LAMBDA_PUSH,
        push_margin=PUSH_MARGIN,
        ema_decay=EMA_DECAY,
        proto_anneal_epochs=PROTO_ANNEAL_EPOCHS,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "proto_disentangled_cnn_gru",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # ── Evaluate on test subject ───────────────────────────────────────────────
    # Build flat test arrays using trainer's class_ids ordering.
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test data available after split construction")
        return {
            "test_subject": test_subject,
            "model_type": "proto_disentangled_cnn_gru",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)  # (N, T, C)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy() is inherited from DisentangledTrainer:
    # it applies (N,T,C)→(N,C,T) transpose and standardization internally,
    # then runs inference using only the gesture classifier.
    test_results = trainer.evaluate_numpy(
        X_test,
        y_test,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1 = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Acc={test_acc:.4f}  F1={test_f1:.4f}"
    )

    # ── Diagnostic: intra-class variance (train subjects only) ───────────────
    # LOSO compliance: _compute_subject_variance_diagnostic receives ONLY
    # train_subjects as input; the test_subject is excluded by construction.
    variance_info = None
    if compute_diagnostics and len(train_subjects) >= 2:
        try:
            variance_info = _compute_subject_variance_diagnostic(
                trainer=trainer,
                subjects_data=subjects_data,
                train_subjects=train_subjects,
                common_gestures=common_gestures,
                multi_loader=multi_loader,
            )
            if variance_info is not None:
                mv = variance_info["mean_intra_class_variance"]
                logger.info(
                    f"Intra-class z_content variance across train subjects: "
                    f"{mv:.4f}" if mv is not None else "n/a"
                )
        except Exception as diag_e:
            logger.warning(f"Diagnostic computation failed (non-fatal): {diag_e}")
            variance_info = None

    # ── Save fold results ─────────────────────────────────────────────────────
    results_to_save = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "common_gestures": common_gestures,
        "training": training_results,
        "cross_subject_test": {
            "subject": test_subject,
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "report": test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
        "variance_diagnostic": variance_info,
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "proto_disentangled_cnn_gru",
        "exercises": exercises,
        "disentanglement_config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
        },
        "prototype_config": {
            "lambda_center": LAMBDA_CENTER,
            "lambda_push": LAMBDA_PUSH,
            "push_margin": PUSH_MARGIN,
            "ema_decay": EMA_DECAY,
            "proto_anneal_epochs": PROTO_ANNEAL_EPOCHS,
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
        "variance_diagnostic": variance_info,
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # ── Memory cleanup ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data, splits
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "proto_disentangled_cnn_gru",
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "mean_intra_class_variance": (
            variance_info["mean_intra_class_variance"]
            if variance_info is not None
            else None
        ),
    }


# ══════════════════════════════════ MAIN ═════════════════════════════════════


def main():
    # parse_subjects_args() defaults to CI_TEST_SUBJECTS — safe for vast.ai server
    ALL_SUBJECTS = parse_subjects_args()

    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    # ── Configs ───────────────────────────────────────────────────────────────
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
        model_type="proto_disentangled_cnn_gru",
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
    print(f"Hypothesis : Disentanglement + Prototype Regularization in z_content")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Exercises  : {EXERCISES}")
    print(f"Arch       : content_dim={CONTENT_DIM}, style_dim={STYLE_DIM}")
    print(f"Disen.     : α={ALPHA} (subj), β={BETA} (MI, {MI_LOSS_TYPE})")
    print(
        f"Proto      : λ_center={LAMBDA_CENTER}, λ_push={LAMBDA_PUSH}, "
        f"margin={PUSH_MARGIN}, ema={EMA_DECAY}"
    )
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = (
            OUTPUT_ROOT / "proto_disentangled_cnn_gru" / f"test_{test_subject}"
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
            compute_diagnostics=COMPUTE_DIAGNOSTICS,
        )
        all_results.append(result)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s = [r["test_f1_macro"] for r in valid]
        variances = [
            r["mean_intra_class_variance"]
            for r in valid
            if r.get("mean_intra_class_variance") is not None
        ]
        print(f"\n{'=' * 60}")
        print(f"Proto-Disentangled CNN-GRU — LOSO Summary ({len(valid)} folds)")
        print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        if variances:
            print(
                f"  z_content intra-class variance (train subjects): "
                f"{np.mean(variances):.4f} ± {np.std(variances):.4f}"
            )
        print(f"{'=' * 60}\n")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary: Dict = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "Disentanglement + Prototype Regularization in z_content",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "disentanglement_config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
        },
        "prototype_config": {
            "lambda_center": LAMBDA_CENTER,
            "lambda_push": LAMBDA_PUSH,
            "push_margin": PUSH_MARGIN,
            "ema_decay": EMA_DECAY,
            "proto_anneal_epochs": PROTO_ANNEAL_EPOCHS,
        },
        "results": all_results,
    }

    if valid:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_folds": len(valid),
        }
        if variances:
            summary["aggregate"]["mean_intra_class_variance"] = float(
                np.mean(variances)
            )
            summary["aggregate"]["std_intra_class_variance"] = float(
                np.std(variances)
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)

    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── Hypothesis executor (optional) ───────────────────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed  # noqa

        if valid:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
            }
            if variances:
                metrics["mean_intra_class_variance"] = float(np.mean(variances))
            mark_hypothesis_verified(
                "H-ProtoDisentangle", metrics, experiment_name=EXPERIMENT_NAME
            )
        else:
            mark_hypothesis_failed("H-ProtoDisentangle", "All folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
