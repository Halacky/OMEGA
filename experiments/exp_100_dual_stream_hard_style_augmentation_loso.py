"""
Experiment 100: Dual-Stream Hard Style Augmentation with Adversarial Perturbation

Hypothesis H100:
    Standard MixStyle (exp_60) creates virtual styles via convex combinations
    of training-subject styles. Such mixtures always lie INSIDE the convex hull
    of observed styles, so a test subject whose style falls OUTSIDE that hull
    remains under-covered.

    Adversarial style perturbation (FGSM-like) generates "hard" virtual styles
    on the BOUNDARY of the plausible style distribution, explicitly targeting the
    failure modes of convex-combination augmentation.

Approach — Dual-Stream (LOSO-safe):

    Stream 1 (Easy — MixStyle):
        z_style_easy = λ·z_style_i + (1−λ)·z_style_j,   λ ~ Beta(0.4, 0.4)
        z_content_easy = FiLM(z_content, z_style_easy)
        L_easy = CE(GestureClassifier(z_content_easy), y)

    Stream 2 (Hard — FGSM adversarial):
        grad = ∇_{z_style} L_gesture(GestureClassifier(FiLM(z_content, z_style)))
        z_style_hard = z_style + ε · sign(grad)
        ε = 0.5 · per-dim std(z_style across batch)          ← training-batch only
        z_style_hard clipped to [μ−3σ, μ+3σ] per dim         ← plausibility bound
        M = σ(UncertaintyNet(z_content))                      ← soft mask
        z_content_hard = FiLM(z_content * (1−M), z_style_hard)
        L_hard = CE(GestureClassifier(z_content_hard), y)

    Base path:
        L_base = CE(GestureClassifier(z_content), y)          ← no FiLM

    Disentanglement:
        L_subject = CE(SubjectClassifier(z_style), y_subject)
        L_MI = DistCorr(z_content, z_style)

    Total loss:
        L_total = L_base + 0.3·L_easy + 0.7·L_hard + α·L_subject + β(t)·L_MI

    Inference (LOSO clean):
        GestureClassifier(z_content)   ← NO FiLM, NO perturbation

LOSO data-leakage audit:
    ✓ Style mixing and adversarial perturbation only from training-batch z_style.
    ✓ ε and clipping [μ−3σ, μ+3σ] computed from training-batch statistics only.
    ✓ Test subject data goes EXCLUSIVELY into splits["test"] — never into any
      style pool, any batch, or any running statistic.
    ✓ Channel standardisation computed on training windows only.
    ✓ torch.autograd.grad inner pass is isolated (detached leaf + detached content);
      model .grad attributes are NOT modified by the inner gradient computation.
    ✓ Validation split comes from training subjects; no test-subject windows.
    ✓ Inference uses fixed weights; no test-time adaptation.

Expected improvement over exp_60 (MixStyle):
    +2–4 pp F1, especially on "hard" subjects (per-subject F1 < 30%) where
    the test style may lie outside the training convex hull.

Usage:
    python experiments/exp_100_dual_stream_hard_style_augmentation_loso.py
    python experiments/exp_100_dual_stream_hard_style_augmentation_loso.py --ci
    python experiments/exp_100_dual_stream_hard_style_augmentation_loso.py \\
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
from typing import Dict, List

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
from training.dual_stream_hard_style_trainer import DualStreamHardStyleTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ════════════════════════ EXPERIMENT SETTINGS ════════════════════════════

EXPERIMENT_NAME = "exp_100_dual_stream_hard_style_augmentation"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# ── Disentanglement ───────────────────────────────────────────────────────
CONTENT_DIM = 128
STYLE_DIM = 64
ALPHA = 0.5              # weight of subject classification loss
BETA = 0.1               # weight of MI minimization loss
BETA_ANNEAL_EPOCHS = 10
MI_LOSS_TYPE = "distance_correlation"

# ── Dual-stream augmentation ──────────────────────────────────────────────
MIX_ALPHA = 0.4          # Beta(MIX_ALPHA, MIX_ALPHA) for easy MixStyle path
EASY_WEIGHT = 0.3        # weight of L_gesture_easy in total loss
HARD_WEIGHT = 0.7        # weight of L_gesture_hard in total loss
EPSILON_FACTOR = 0.5     # ε = EPSILON_FACTOR · per-dim std(z_style in batch)


# ════════════════════════ DATA PREPARATION ═══════════════════════════════


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
    Build train / val / test splits with subject-provenance labels.

    Subject labels are required by DualStreamHardStyleTrainer so that
    the easy-path MixStyle can perform cross-subject mixing within each
    training batch.

    LOSO guarantee:
        test_subject data goes ONLY into splits["test"].
        The function never includes test_subject windows in train or val.

    Returns:
        {
            "train":                Dict[gesture_id, np.ndarray (N,T,C)]
            "val":                  Dict[gesture_id, np.ndarray (N,T,C)]
            "test":                 Dict[gesture_id, np.ndarray (N,T,C)]
            "train_subject_labels": Dict[gesture_id, np.ndarray (N,) int]
            "num_train_subjects":   int
        }
    """
    rng = np.random.RandomState(seed)
    # Map train subject IDs to consecutive integer indices [0, …, K−1]
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # ── Collect per-gesture arrays across training subjects ────────────
    train_dict: Dict[int, np.ndarray] = {}
    train_subj_labels: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        windows_for_gid = []
        subj_labels_for_gid = []

        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid not in filtered:
                continue
            for rep_array in filtered[gid]:
                if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                    windows_for_gid.append(rep_array)
                    subj_labels_for_gid.append(
                        np.full(
                            len(rep_array),
                            train_subject_to_idx[sid],
                            dtype=np.int64,
                        )
                    )

        if windows_for_gid:
            train_dict[gid] = np.concatenate(windows_for_gid, axis=0)
            train_subj_labels[gid] = np.concatenate(subj_labels_for_gid, axis=0)

    # ── Split each gesture's training pool into train / val ───────────
    final_train: Dict[int, np.ndarray] = {}
    final_val: Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if gid not in train_dict:
            continue
        X_gid = train_dict[gid]
        S_gid = train_subj_labels[gid]
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        final_train[gid] = X_gid[train_idx]
        final_val[gid] = X_gid[val_idx]
        final_train_subj[gid] = S_gid[train_idx]

    # ── Test split: test_subject ONLY — LOSO boundary ─────────────────
    test_dict: Dict[int, np.ndarray] = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        valid_reps = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
        if valid_reps:
            test_dict[gid] = np.concatenate(valid_reps, axis=0)

    return {
        "train": final_train,
        "val": final_val,
        "test": test_dict,
        "train_subject_labels": final_train_subj,
        "num_train_subjects": num_train_subjects,
    }


# ════════════════════════ SINGLE LOSO FOLD ═══════════════════════════════


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
    Execute one LOSO fold with Dual-Stream Hard Style training.

    Returns a result dict with test_accuracy / test_f1_macro, or
    error info if the fold fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "dual_stream_hard_style_cnn_gru"

    # Save configs for reproducibility
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "exp100_config.json", "w") as f:
        json.dump(
            {
                "content_dim": CONTENT_DIM,
                "style_dim": STYLE_DIM,
                "alpha": ALPHA,
                "beta": BETA,
                "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
                "mi_loss_type": MI_LOSS_TYPE,
                "mix_alpha": MIX_ALPHA,
                "easy_weight": EASY_WEIGHT,
                "hard_weight": HARD_WEIGHT,
                "epsilon_factor": EPSILON_FACTOR,
            },
            f,
            indent=4,
        )

    # ── Data loading ───────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

    # Load all subjects (train + test) in one pass.
    # LOSO boundary is enforced in _build_splits_with_subject_labels:
    # test_subject windows go EXCLUSIVELY into splits["test"].
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

    # ── Build splits ───────────────────────────────────────────────────
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

    # ── Create trainer ─────────────────────────────────────────────────
    trainer = DualStreamHardStyleTrainer(
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
        mix_alpha=MIX_ALPHA,
        easy_weight=EASY_WEIGHT,
        hard_weight=HARD_WEIGHT,
        epsilon_factor=EPSILON_FACTOR,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "dual_stream_hard_style_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # ── Evaluate on held-out test subject ──────────────────────────────
    # Inference: model.eval() → GestureClassifier(z_content) only — LOSO clean.
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test data available after gesture filtering")
        return {
            "test_subject": test_subject,
            "model_type": "dual_stream_hard_style_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy: transposes + standardises + calls model.eval()
    # Inherited from DisentangledTrainer — uses base-path logits only.
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
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── Save fold results ──────────────────────────────────────────────
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
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable(
            {
                "test_subject": test_subject,
                "train_subjects": train_subjects,
                "model_type": "dual_stream_hard_style_cnn_gru",
                "approach": APPROACH,
                "exercises": exercises,
                "exp100_config": {
                    "content_dim": CONTENT_DIM,
                    "style_dim": STYLE_DIM,
                    "alpha": ALPHA,
                    "beta": BETA,
                    "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
                    "mi_loss_type": MI_LOSS_TYPE,
                    "mix_alpha": MIX_ALPHA,
                    "easy_weight": EASY_WEIGHT,
                    "hard_weight": HARD_WEIGHT,
                    "epsilon_factor": EPSILON_FACTOR,
                },
                "metrics": {
                    "test_accuracy": test_acc,
                    "test_f1_macro": test_f1,
                },
            }
        ),
        filename="fold_metadata.json",
    )

    # ── Memory cleanup ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "dual_stream_hard_style_cnn_gru",
        "approach": APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════ MAIN ════════════════════════════════════════════


def main():
    ALL_SUBJECTS = parse_subjects_args()
    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

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
        model_type="dual_stream_hard_style_cnn_gru",
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
    print(f"Hypothesis H100: Dual-Stream Hard Style Augmentation (FGSM adversarial)")
    print(f"Subjects:   {ALL_SUBJECTS}")
    print(f"Exercises:  {EXERCISES}")
    print(
        f"Disentanglement: content_dim={CONTENT_DIM}, style_dim={STYLE_DIM}"
    )
    print(
        f"Loss weights: alpha={ALPHA} (subject), beta={BETA} (MI), "
        f"easy={EASY_WEIGHT}, hard={HARD_WEIGHT}"
    )
    print(
        f"Style aug: mix_alpha={MIX_ALPHA} (Beta), "
        f"epsilon_factor={EPSILON_FACTOR} (FGSM)"
    )
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = (
            OUTPUT_ROOT
            / "dual_stream_hard_style_cnn_gru"
            / f"test_{test_subject}"
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

    # ── Aggregate LOSO summary ─────────────────────────────────────────
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]

    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        print(f"\n{'=' * 60}")
        print(
            f"Dual-Stream Hard Style — LOSO Summary ({len(valid_results)} folds)"
        )
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "H100: Dual-Stream Hard Style Augmentation with FGSM adversarial perturbation",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
            "mix_alpha": MIX_ALPHA,
            "easy_weight": EASY_WEIGHT,
            "hard_weight": HARD_WEIGHT,
            "epsilon_factor": EPSILON_FACTOR,
        },
        "results": all_loso_results,
    }

    if valid_results:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_folds": len(valid_results),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)

    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # Report to hypothesis_executor if available
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        if valid_results:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
            }
            mark_hypothesis_verified("H100", metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H100", "All folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
