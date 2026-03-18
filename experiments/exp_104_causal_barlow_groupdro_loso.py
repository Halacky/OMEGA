"""
Experiment 104: Causal Disentanglement with Barlow Twins + GroupDRO (LOSO)

Hypothesis: Distance correlation (exp_31) blindly removes ALL statistical
dependencies between z_content and z_style. But some dependencies are
causally useful (e.g., gesture intensity correlates with amplitude —
this is subject-invariant and task-relevant). Causal disentanglement
(CDDG-inspired) replaces blind MI minimization with:

1. Barlow Twins redundancy reduction: removes linear cross-correlations
   (O(D^2) vs O(B^2)), but allows nonlinear causal dependencies to persist.

2. Causal consistency loss: for pairs (x_i, x_j) of the same gesture from
   different subjects, minimizes ||z_content(x_i) - z_content(x_j)||.
   This directly pushes z_content toward subject-invariant factors via
   positive pairs (not just penalty on MI).

3. Reconstruction loss: decoder(z_content || z_style) → x_hat prevents
   information loss from over-aggressive disentanglement.

4. GroupDRO on gesture loss (exp_57) for worst-case subject robustness.

LOSO compliance:
- All causal pairs from TRAIN subjects only; test subject never in batch.
- Decoder trained on train data only.
- Channel stats from train only. Early stopping on train-subject val split.
- Test subject evaluated ONLY for final metrics. No adaptation.

Expected: +1-2pp F1 over exp_31.

References:
    - CDDG (Neural Networks 2024): causal disentanglement for time-series
    - Barlow Twins (ICML 2021): redundancy reduction
    - Sagawa et al. 2020: GroupDRO
"""

import gc
import json
import sys
import traceback
from datetime import datetime
from dataclasses import asdict
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
from training.causal_barlow_trainer import CausalBarlowTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ========== EXPERIMENT SETTINGS ==========
EXPERIMENT_NAME = "exp_104_causal_barlow_groupdro"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# Causal disentanglement hyperparameters
CONTENT_DIM = 128
STYLE_DIM = 64
ALPHA = 0.5              # subject classifier weight
BETA_BT = 0.1            # Barlow Twins redundancy reduction weight
BETA_CC = 0.05           # causal consistency weight
DELTA_RECON = 0.01       # reconstruction regularization weight
BETA_ANNEAL_EPOCHS = 10  # anneal auxiliary losses from 0 → full over N epochs
DRO_ETA = 0.01           # GroupDRO step size


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
    Build train/val/test splits with subject provenance tracking.

    Returns augmented splits dict with:
        "train", "val", "test": Dict[int, np.ndarray]  (gesture_id → windows)
        "train_subject_labels": Dict[int, np.ndarray]   (gesture_id → subj indices)
        "num_train_subjects": int

    LOSO guarantee: test subject data goes ONLY into splits["test"].
    Train/val splits contain ONLY train-subject data.
    """
    rng = np.random.RandomState(seed)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # Build per-gesture arrays for training data with subject labels
    train_dict = {}
    train_subj_labels = {}

    for gid in common_gestures:
        windows_for_gid = []
        subj_labels_for_gid = []

        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid in filtered:
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

    # Split train → train/val (same permutation for windows and subject labels)
    final_train = {}
    final_val = {}
    final_train_subj = {}

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

    # Build test split from test subject ONLY
    test_dict = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        if reps:
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
    """Single LOSO fold with causal Barlow Twins disentanglement + GroupDRO."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "causal_barlow_cnn_gru"

    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # Data loader
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

    base_viz = Visualizer(output_dir, logger)

    # Load all subjects
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

    # Build splits with subject provenance (test subject → test split ONLY)
    splits = _build_splits_with_subject_labels(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    # Log split info
    for split_name in ["train", "val", "test"]:
        total = sum(
            len(arr) for arr in splits[split_name].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"{split_name.upper()}: {total} windows, "
            f"{len(splits[split_name])} gestures"
        )

    # Create trainer
    trainer = CausalBarlowTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        content_dim=CONTENT_DIM,
        style_dim=STYLE_DIM,
        alpha=ALPHA,
        beta_bt=BETA_BT,
        beta_cc=BETA_CC,
        delta_recon=DELTA_RECON,
        beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
        dro_eta=DRO_ETA,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "causal_barlow_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # Evaluate on test subject (using only z_content — no subject info)
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test data available")
        return {
            "test_subject": test_subject,
            "model_type": "causal_barlow_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    test_results = trainer.evaluate_numpy(
        X_test_concat, y_test_concat,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1 = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # Save results
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
        json.dump(
            make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False
        )

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "causal_barlow_cnn_gru",
        "approach": APPROACH,
        "exercises": exercises,
        "causal_config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta_bt": BETA_BT,
            "beta_cc": BETA_CC,
            "delta_recon": DELTA_RECON,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "dro_eta": DRO_ETA,
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    if trainer.final_group_weights is not None:
        meta["final_group_weights"] = trainer.final_group_weights
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "causal_barlow_cnn_gru",
        "approach": APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    ALL_SUBJECTS = parse_subjects_args()
    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = (
        ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    )

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
        model_type="causal_barlow_cnn_gru",
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
    print(f"Causal Disentanglement (CDDG-inspired) + Barlow Twins + GroupDRO")
    print(f"Subjects: {ALL_SUBJECTS}")
    print(f"Exercises: {EXERCISES}")
    print(f"Content dim={CONTENT_DIM}, Style dim={STYLE_DIM}")
    print(f"alpha={ALPHA}, beta_bt={BETA_BT}, beta_cc={BETA_CC}, "
          f"delta={DELTA_RECON}, dro_eta={DRO_ETA}")
    print(f"Anneal epochs={BETA_ANNEAL_EPOCHS}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "causal_barlow_cnn_gru" / f"test_{test_subject}"

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

    # Aggregate
    valid_results = [
        r for r in all_loso_results if r.get("test_accuracy") is not None
    ]
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        print(f"\n{'=' * 60}")
        print(
            f"Causal Barlow + GroupDRO — LOSO Summary ({len(valid_results)} folds)"
        )
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    # Save summary
    summary = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "Causal Disentanglement (CDDG) + Barlow Twins + GroupDRO",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "causal_config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta_bt": BETA_BT,
            "beta_cc": BETA_CC,
            "delta_recon": DELTA_RECON,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "dro_eta": DRO_ETA,
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
        json.dump(
            make_json_serializable(summary), f, indent=4, ensure_ascii=False
        )

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
            mark_hypothesis_verified(
                "H_causal_barlow", metrics, experiment_name=EXPERIMENT_NAME
            )
        else:
            mark_hypothesis_failed("H_causal_barlow", "All folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
