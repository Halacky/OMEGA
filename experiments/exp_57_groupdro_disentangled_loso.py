"""
Experiment 57: GroupDRO + Content-Style Disentanglement for LOSO EMG

Hypothesis H8
-------------
Standard ERM (exp_31) optimises the *average* gesture loss over all train
subjects, which biases z_content towards "easy" subjects.  Optimising the
*worst-case* risk via Group Distributionally Robust Optimization (GroupDRO,
Sagawa et al. 2020) over train subjects forces the encoder to produce
z_content that works for the hardest subjects — which should

    1. increase mean LOSO F1 by closing the tail,
    2. decrease LOSO std (more uniform per-subject performance), and
    3. raise worst-subject F1 (the primary DRO objective).

Difference from exp_31 (ERM disentanglement)
---------------------------------------------
    exp_31 : L_gesture = E_{all train}[ CE(z_content(x), y) ]
    exp_57 : L_gesture = Σ_s q_s · E_{train|s}[ CE(z_content(x), y) ]
               where q_s are updated via exponentiated gradient ascent.
    Subject-classifier and MI losses are identical in both experiments.

LOSO compliance
---------------
- Groups are defined over **train subjects only**.  The test subject is
  withheld before any data loading loop and never influences group weights,
  loss computation, normalization statistics, or model selection.
- Channel standardization is computed from X_train (train subjects) only.
- Early stopping uses val_loss from the train-subject val split.
- Test-subject windows are evaluated ONCE for final metrics only.
- No test-subject information is passed back into the training loop.

What to measure vs exp_31
--------------------------
- mean F1, std F1, worst-subject F1  (primary comparison)
- Per-subject breakdown (which subjects improved / degraded)
- Final group weights per fold (which train subjects were "hard")
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

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    make_json_serializable,
    parse_subjects_args,
)
from training.groupdro_disentangled_trainer import GroupDRODisentangledTrainer
from utils.artifacts import ArtifactSaver
from utils.logging import seed_everything, setup_logging
from visualization.base import Visualizer

# ══════════════════════════════════════════════════════════════════════════════
# Experiment settings
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_57_groupdro_disentangled"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# Disentanglement hyperparameters (same as exp_31 for fair comparison)
CONTENT_DIM = 128
STYLE_DIM = 64
ALPHA = 0.5            # subject classifier weight
BETA = 0.1             # MI minimization weight
BETA_ANNEAL_EPOCHS = 10
MI_LOSS_TYPE = "distance_correlation"

# GroupDRO-specific
DRO_ETA = 0.01         # exponentiated gradient step size for group weights


# ══════════════════════════════════════════════════════════════════════════════
# Split construction (identical to exp_31 — preserves subject provenance)
# ══════════════════════════════════════════════════════════════════════════════

def _build_splits_with_subject_labels(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: "MultiSubjectLoader",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Build train / val / test splits with per-window subject provenance.

    Train and val splits contain ONLY windows from ``train_subjects``.
    Test split contains ONLY windows from ``test_subject``.

    Returns
    -------
    dict with keys:
        "train"               : Dict[gesture_id, np.ndarray]
        "val"                 : Dict[gesture_id, np.ndarray]
        "test"                : Dict[gesture_id, np.ndarray]
        "train_subject_labels": Dict[gesture_id, np.ndarray] of int64 subject indices
        "num_train_subjects"  : int
    """
    rng = np.random.RandomState(seed)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # ── Collect all train windows with subject provenance ─────────────────
    train_dict: Dict[int, List[np.ndarray]] = {}
    train_subj_dict: Dict[int, List[np.ndarray]] = {}

    for gid in common_gestures:
        train_dict[gid] = []
        train_subj_dict[gid] = []

        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid not in filtered:
                continue
            for rep_array in filtered[gid]:
                if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                    train_dict[gid].append(rep_array)
                    train_subj_dict[gid].append(
                        np.full(len(rep_array), train_subject_to_idx[sid], dtype=np.int64)
                    )

    # ── Train/val split (gesture-stratified, consistent permutation) ──────
    final_train: Dict[int, np.ndarray] = {}
    final_val: Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)
        S_gid = np.concatenate(train_subj_dict[gid], axis=0)
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        final_train[gid] = X_gid[train_idx]
        final_val[gid] = X_gid[val_idx]
        final_train_subj[gid] = S_gid[train_idx]

    # ── Test split (test subject only) ────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# Single LOSO fold
# ══════════════════════════════════════════════════════════════════════════════

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
    Run one LOSO fold: train on ``train_subjects``, evaluate on ``test_subject``.

    Returns
    -------
    dict with keys:
        test_subject, model_type, approach,
        test_accuracy, test_f1_macro,
        final_group_weights, train_subjects_sorted,
        error (only if something failed)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "groupdro_disentangled_cnn_gru"

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

    # ── Load subjects (all at once; test subject needed for test split) ───
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

    # ── Build splits (test_subject only appears in splits["test"]) ────────
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
            len(arr) for arr in splits[split_name].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"{split_name.upper()}: {total} windows, {len(splits[split_name])} gestures"
        )

    # ── Create trainer ────────────────────────────────────────────────────
    trainer = GroupDRODisentangledTrainer(
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
        dro_eta=DRO_ETA,
    )

    try:
        trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "groupdro_disentangled_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "final_group_weights": None,
            "train_subjects_sorted": sorted(train_subjects),
            "error": str(e),
        }

    # ── Evaluate on test subject (LOSO final evaluation) ─────────────────
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test data available for final evaluation")
        return {
            "test_subject": test_subject,
            "model_type": "groupdro_disentangled_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "final_group_weights": trainer.final_group_weights,
            "train_subjects_sorted": sorted(train_subjects),
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy() is from DisentangledTrainer — uses z_content only.
    test_results = trainer.evaluate_numpy(
        X_test, y_test,
        split_name=f"loso_test_{test_subject}",
        visualize=True,
    )
    test_acc = float(test_results["accuracy"])
    test_f1 = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f} | "
        f"group_w={[f'{w:.3f}' for w in (trainer.final_group_weights or [])]}"
    )

    # ── Save fold results ─────────────────────────────────────────────────
    fold_result = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "common_gestures": common_gestures,
        "groupdro_config": {
            "dro_eta": DRO_ETA,
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
        },
        "final_group_weights": trainer.final_group_weights,
        "train_subjects_sorted": sorted(train_subjects),
        "cross_subject_test": {
            "subject": test_subject,
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "report": test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_result), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "model_type": "groupdro_disentangled_cnn_gru",
            "approach": APPROACH,
            "exercises": exercises,
            "groupdro_config": fold_result["groupdro_config"],
            "metrics": {"test_accuracy": test_acc, "test_f1_macro": test_f1},
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
        "test_subject": test_subject,
        "model_type": "groupdro_disentangled_cnn_gru",
        "approach": APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "final_group_weights": fold_result["final_group_weights"],
        "train_subjects_sorted": sorted(train_subjects),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

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
        model_type="groupdro_disentangled_cnn_gru",
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
    print(f"Hypothesis H8: GroupDRO + Content-Style Disentanglement")
    print(f"Subjects: {ALL_SUBJECTS}  ({len(ALL_SUBJECTS)} total)")
    print(f"Exercises: {EXERCISES}")
    print(f"Disentanglement: content_dim={CONTENT_DIM}, style_dim={STYLE_DIM}")
    print(f"Loss weights: alpha={ALPHA} (subject), beta={BETA} (MI)")
    print(f"GroupDRO: eta={DRO_ETA}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "groupdro_disentangled_cnn_gru" / f"test_{test_subject}"

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

    # ── Aggregate metrics ─────────────────────────────────────────────────
    valid = [r for r in all_loso_results if r.get("test_f1_macro") is not None]

    print(f"\n{'=' * 60}")
    print(f"GroupDRO Disentangled CNN-GRU — LOSO Summary ({len(valid)} folds)")

    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s = [r["test_f1_macro"] for r in valid]
        worst_f1 = float(np.min(f1s))
        best_f1 = float(np.max(f1s))
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))

        print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Worst-subject F1: {worst_f1:.4f}  (DRO primary target)")
        print(f"  Best-subject F1:  {best_f1:.4f}")

        # Per-subject detail
        print(f"\n  Per-subject results:")
        for r in sorted(valid, key=lambda x: x["test_f1_macro"]):
            gw = r.get("final_group_weights")
            gw_str = f"  | train_gw={[f'{w:.3f}' for w in gw]}" if gw else ""
            f1_val = r['test_f1_macro']
            acc_val = r['test_accuracy']
            print(f"    {r['test_subject']}: acc={acc_val:.4f}, f1={f1_val:.4f}{gw_str}")

    print(f"{'=' * 60}\n")

    # ── Save summary ──────────────────────────────────────────────────────
    summary: Dict = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "H8: GroupDRO + Content-Style Disentanglement",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "groupdro_config": {
            "dro_eta": DRO_ETA,
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
        },
        "results": all_loso_results,
    }

    if valid:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": mean_f1,
            "std_f1_macro": std_f1,
            "worst_f1_macro": worst_f1,
            "best_f1_macro": best_f1,
            "num_folds": len(valid),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── Report to hypothesis_executor (optional) ──────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_failed, mark_hypothesis_verified
        if valid:
            mark_hypothesis_verified(
                "H8",
                {
                    "mean_accuracy": float(np.mean(accs)),
                    "std_accuracy": float(np.std(accs)),
                    "mean_f1_macro": mean_f1,
                    "std_f1_macro": std_f1,
                    "worst_f1_macro": worst_f1,
                },
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed("H8", "All LOSO folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
