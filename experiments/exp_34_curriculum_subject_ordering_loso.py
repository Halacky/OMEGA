"""
Experiment 34: Curriculum Learning — Training from Similar to Distant Subjects

Hypothesis:
    Training the model starting from "similar" subjects and progressively adding
    "distant" ones improves cross-subject invariance, inspired by curriculum
    learning in computer vision (domain difficulty scheduling).

Approach:
    1. Compute a distance matrix between all subjects based on their EMG
       signal statistics (per-channel mean/std + inter-channel correlation).
    2. For each LOSO fold, rank training subjects by similarity to the test subject.
    3. Begin training on k_init nearest subjects.
    4. Every `expand_every` epochs, add the next closest subject.
    5. After all subjects are included, continue for `consolidation_epochs`.

Why this should work:
    - The model first learns from data most similar to the test distribution,
      building a strong initial representation.
    - Gradually adding more diverse (distant) subjects forces the model to
      generalize without the shock of seeing very different data early on.
    - This mimics curriculum learning: easy → hard data ordering.

Model: CNN-GRU with attention (cnn_gru_attention) — proven strong baseline.

Cannot use CrossSubjectExperiment.run() directly because it merges all
training subjects and loses subject identity. Instead we build splits
manually with subject provenance tracking (same pattern as exp_31).
"""

import os
import sys
import json
import gc
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

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
from training.curriculum_trainer import CurriculumTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ========== EXPERIMENT SETTINGS ==========
EXPERIMENT_NAME = "exp_34_curriculum_subject_ordering"
APPROACH = "deep_raw"
MODEL_TYPE = "cnn_gru_attention"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# Curriculum hyperparameters
K_INIT = 1                   # start training on the 1 nearest subject
EXPAND_EVERY = 10            # add next subject every 10 epochs
CONSOLIDATION_EPOCHS = 20    # final training on all subjects
LR_ON_EXPAND = None          # optional LR reset on subject expansion (None = keep current)

# Distance metric
DISTANCE_METRIC = "channel_stats"  # "channel_stats" or "mmd_linear"


# ========== SUBJECT DISTANCE COMPUTATION ==========

def _compute_subject_channel_stats(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> np.ndarray:
    """
    Compute a compact representation of a subject's EMG distribution.

    For each channel, compute: mean, std, RMS, and pairwise channel correlations.
    Returns a flat feature vector characterizing this subject's signal.
    """
    # Collect all windows into one big array
    all_windows = []
    for gid in sorted(grouped_windows.keys()):
        for rep in grouped_windows[gid]:
            if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0:
                all_windows.append(rep)
    if not all_windows:
        return np.zeros(1)

    # (total_windows, T, C)
    X = np.concatenate(all_windows, axis=0)
    N, T, C = X.shape

    # Per-channel statistics: mean, std, RMS — computed across all windows and time
    # Reshape to (N*T, C) for per-channel stats
    X_flat = X.reshape(-1, C)  # (N*T, C)
    ch_mean = X_flat.mean(axis=0)   # (C,)
    ch_std = X_flat.std(axis=0)     # (C,)
    ch_rms = np.sqrt((X_flat ** 2).mean(axis=0))  # (C,)

    # Per-channel skewness and kurtosis (higher-order statistics capture
    # distribution shape differences between subjects)
    eps = 1e-8
    centered = X_flat - ch_mean[None, :]
    ch_skew = (centered ** 3).mean(axis=0) / (ch_std ** 3 + eps)  # (C,)
    ch_kurt = (centered ** 4).mean(axis=0) / (ch_std ** 4 + eps) - 3.0  # (C,)

    # Inter-channel correlation matrix (upper triangle)
    corr = np.corrcoef(X_flat.T)  # (C, C)
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(C, k=1)
    ch_corr = corr[triu_idx]  # (C*(C-1)/2,)

    # Handle NaN in correlations (constant channels)
    ch_corr = np.nan_to_num(ch_corr, nan=0.0)

    # Concatenate all features
    features = np.concatenate([ch_mean, ch_std, ch_rms, ch_skew, ch_kurt, ch_corr])
    return features


def compute_distance_matrix(
    subjects_data: Dict[str, Tuple],
    subject_ids: List[str],
    metric: str = "channel_stats",
) -> np.ndarray:
    """
    Compute pairwise distance matrix between subjects.

    Args:
        subjects_data: dict of subject_id → (emg, segments, grouped_windows)
        subject_ids: ordered list of subject IDs
        metric: "channel_stats" or "mmd_linear"

    Returns:
        distance_matrix: (N_subjects, N_subjects) symmetric matrix
    """
    n = len(subject_ids)

    if metric == "channel_stats":
        # Compute per-subject feature vectors, then Euclidean distance
        features = []
        for sid in subject_ids:
            _, _, gw = subjects_data[sid]
            feat = _compute_subject_channel_stats(gw)
            features.append(feat)
        features = np.array(features)  # (n, D)

        # Normalize features to unit variance for fair distance comparison
        feat_std = features.std(axis=0) + 1e-8
        features_norm = features / feat_std[None, :]

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(features_norm[i] - features_norm[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    elif metric == "mmd_linear":
        # Linear MMD: distance between mean embeddings in channel-time space
        means = []
        for sid in subject_ids:
            _, _, gw = subjects_data[sid]
            all_w = []
            for gid in sorted(gw.keys()):
                for rep in gw[gid]:
                    if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0:
                        all_w.append(rep)
            if all_w:
                X = np.concatenate(all_w, axis=0)  # (N, T, C)
                # Mean embedding: flatten T*C per window, then mean across windows
                X_flat = X.reshape(len(X), -1)  # (N, T*C)
                means.append(X_flat.mean(axis=0))
            else:
                means.append(np.zeros(1))

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(means[i] - means[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return dist_matrix


def get_subject_order_by_similarity(
    dist_matrix: np.ndarray,
    subject_ids: List[str],
    test_subject: str,
    train_subjects: List[str],
) -> List[Tuple[str, float]]:
    """
    Rank training subjects by similarity (closest first) to the test subject.

    Returns:
        List of (subject_id, distance) sorted by distance ascending.
    """
    test_idx = subject_ids.index(test_subject)
    distances = []
    for sid in train_subjects:
        sid_idx = subject_ids.index(sid)
        distances.append((sid, dist_matrix[test_idx, sid_idx]))
    distances.sort(key=lambda x: x[1])
    return distances


# ========== SPLIT BUILDING (with subject provenance) ==========

def _build_splits_with_subject_labels(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    train_subject_to_idx: Dict[str, int],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Build train/val/test splits with subject provenance tracking.

    Returns dict with:
        "train", "val", "test": Dict[int, np.ndarray]  (gesture_id → windows)
        "train_subject_labels": Dict[int, np.ndarray]   (gesture_id → subject indices)
        "num_train_subjects": int
    """
    rng = np.random.RandomState(seed)
    num_train_subjects = len(train_subjects)

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
                            np.full(len(rep_array), train_subject_to_idx[sid], dtype=np.int64)
                        )

        if windows_for_gid:
            train_dict[gid] = np.concatenate(windows_for_gid, axis=0)
            train_subj_labels[gid] = np.concatenate(subj_labels_for_gid, axis=0)

    # Split train → train/val (same random permutation for windows and labels)
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

    # Build test split from test subject
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


# ========== SINGLE LOSO FOLD ==========

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    dist_matrix: np.ndarray,
    all_subject_ids: List[str],
) -> Dict:
    """Single LOSO fold with curriculum subject ordering."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = MODEL_TYPE

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

    # Load all subjects for this fold
    fold_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=fold_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES
    )
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    # --- Compute curriculum order ---
    ranked = get_subject_order_by_similarity(
        dist_matrix, all_subject_ids, test_subject, train_subjects
    )
    # ranked: [(subject_id, distance), ...] sorted nearest first

    # Build a stable subject→index mapping (sorted train subjects as indices)
    sorted_train = sorted(train_subjects)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted_train)}

    # subject_order: list of subject indices in curriculum order (similar → distant)
    subject_order = [train_subject_to_idx[sid] for sid, _dist in ranked]

    logger.info("Curriculum subject order (similar → distant):")
    for sid, dist in ranked:
        logger.info(f"  {sid} (idx={train_subject_to_idx[sid]}, dist={dist:.4f})")

    # Save distance info for this fold
    fold_dist_info = {
        "test_subject": test_subject,
        "train_order": [(sid, float(d)) for sid, d in ranked],
        "subject_order_indices": subject_order,
    }
    with open(output_dir / "curriculum_order.json", "w") as f:
        json.dump(fold_dist_info, f, indent=4)

    # --- Build splits ---
    splits = _build_splits_with_subject_labels(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        train_subject_to_idx=train_subject_to_idx,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )
    # Add curriculum order to splits
    splits["train_subject_order"] = subject_order

    for split_name in ["train", "val", "test"]:
        total = sum(
            len(arr) for arr in splits[split_name].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(f"{split_name.upper()}: {total} windows across {len(splits[split_name])} gestures")

    # --- Create curriculum trainer ---
    trainer = CurriculumTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        k_init=K_INIT,
        expand_every=EXPAND_EVERY,
        consolidation_epochs=CONSOLIDATION_EPOCHS,
        lr_on_expand=LR_ON_EXPAND,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": MODEL_TYPE,
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # --- Evaluate on test subject ---
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
            "model_type": MODEL_TYPE,
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
        "curriculum_order": [(sid, float(d)) for sid, d in ranked],
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
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": MODEL_TYPE,
        "approach": APPROACH,
        "exercises": exercises,
        "curriculum_config": {
            "k_init": K_INIT,
            "expand_every": EXPAND_EVERY,
            "consolidation_epochs": CONSOLIDATION_EPOCHS,
            "lr_on_expand": LR_ON_EXPAND,
            "distance_metric": DISTANCE_METRIC,
            "subject_order": [(sid, float(d)) for sid, d in ranked],
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": MODEL_TYPE,
        "approach": APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ========== MAIN ==========

def main():
    ALL_SUBJECTS = parse_subjects_args()
    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject_tag = "_".join(s.replace("DB2_s", "") for s in ALL_SUBJECTS)
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}_{subject_tag}"

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
        model_type=MODEL_TYPE,
        pipeline_type=APPROACH,
        use_handcrafted_features=False,
        batch_size=64,
        epochs=200,  # will be overridden by curriculum schedule
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=15,
        seed=42,
        use_class_weights=True,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"{'=' * 80}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Hypothesis: Curriculum learning — similar→distant subject ordering")
    print(f"Model: {MODEL_TYPE}")
    print(f"Subjects: {ALL_SUBJECTS}")
    print(f"Exercises: {EXERCISES}")
    print(f"Curriculum: k_init={K_INIT}, expand_every={EXPAND_EVERY}, "
          f"consolidation={CONSOLIDATION_EPOCHS}")
    print(f"Distance metric: {DISTANCE_METRIC}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    # --- Precompute subject distance matrix ---
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("\n[Phase 1] Computing subject distance matrix...")
    logger_global = setup_logging(OUTPUT_ROOT)
    seed_everything(train_cfg.seed)

    multi_loader_global = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger_global,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    all_subjects_data = multi_loader_global.load_multiple_subjects(
        base_dir=BASE_DIR,
        subject_ids=ALL_SUBJECTS,
        exercises=EXERCISES,
        include_rest=split_cfg.include_rest_in_splits,
    )

    dist_matrix = compute_distance_matrix(
        all_subjects_data, ALL_SUBJECTS, metric=DISTANCE_METRIC
    )

    # Save distance matrix
    dist_info = {
        "subjects": ALL_SUBJECTS,
        "metric": DISTANCE_METRIC,
        "distance_matrix": dist_matrix.tolist(),
    }
    with open(OUTPUT_ROOT / "distance_matrix.json", "w") as f:
        json.dump(dist_info, f, indent=4)

    print("Subject distance matrix:")
    for i, si in enumerate(ALL_SUBJECTS):
        row = "  ".join(f"{dist_matrix[i, j]:.3f}" for j in range(len(ALL_SUBJECTS)))
        print(f"  {si}: [{row}]")

    # Free global data (will be reloaded per fold)
    del all_subjects_data, multi_loader_global
    gc.collect()

    # --- Run LOSO folds ---
    print(f"\n[Phase 2] Running {len(ALL_SUBJECTS)} LOSO folds...")
    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / MODEL_TYPE / f"test_{test_subject}"

        result = run_single_loso_fold(
            base_dir=BASE_DIR,
            output_dir=fold_output,
            train_subjects=train_subjects,
            test_subject=test_subject,
            exercises=EXERCISES,
            proc_cfg=proc_cfg,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
            dist_matrix=dist_matrix,
            all_subject_ids=ALL_SUBJECTS,
        )
        all_loso_results.append(result)

    # --- Aggregate ---
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        print(f"\n{'=' * 60}")
        print(f"Curriculum CNN-GRU-Attention — LOSO Summary ({len(valid_results)} folds)")
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    # Save summary
    summary = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "Curriculum learning: training from similar to distant subjects",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "model_type": MODEL_TYPE,
        "approach": APPROACH,
        "curriculum_config": {
            "k_init": K_INIT,
            "expand_every": EXPAND_EVERY,
            "consolidation_epochs": CONSOLIDATION_EPOCHS,
            "lr_on_expand": LR_ON_EXPAND,
            "distance_metric": DISTANCE_METRIC,
        },
        "distance_matrix": dist_matrix.tolist(),
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
            mark_hypothesis_verified("H_curriculum", metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_curriculum", "All folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
