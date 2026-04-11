"""
Experiment 60: Mixture-of-Styles Content Disentanglement for LOSO EMG

Hypothesis H60: "Mixture of styles without the test style"
    Instead of trying to guess the test-subject style (which would violate LOSO),
    train the content representation to be robust to arbitrary *convex combinations*
    of training-subject styles.

Approach — MixStyle + FiLM in latent z-space:
    1. Encoder → z_content (gesture-relevant) + z_style (subject-relevant)
    2. Within each training batch: for each sample i, mix its z_style with z_style_j
       from a different training subject:
           z_style_mix = λ·z_style_i + (1-λ)·z_style_j,  λ ~ Beta(mix_alpha, mix_alpha)
    3. FiLM conditions z_content on z_style_mix:
           z_content_film = FiLM(z_content, z_style_mix)
    4. Dual gesture loss:
           L_gesture = CE(GestureClassifier(z_content),      y)    ← base path
                     + γ · CE(GestureClassifier(z_content_film), y) ← mixed-style path
    5. Disentanglement regularisers:
           + α · CE(SubjectClassifier(z_style), y_subject)
           + β(t) · dCorr(z_content, z_style)
    6. Inference: only GestureClassifier(z_content) — NO FiLM, NO style needed

LOSO data-leakage audit:
    ✓ Style mixing only between samples within the TRAINING batch.
    ✓ Test subject's data is loaded ONLY into splits["test"] and NEVER appears
      in any training batch or style pool.
    ✓ Channel standardisation statistics computed on training windows only.
    ✓ Early stopping monitored on a held-out subset of TRAINING subjects
      (val split comes from train_subjects, not from test_subject).
    ✓ No test-time adaptation: inference uses fixed model weights.
    ✓ Benchmark: identical LOSO splits to exp_31.

Why this may work:
    - Analogous to MixStyle (Zhou et al., ICLR 2021) in computer vision DG.
    - Forcing the classifier to work on "virtual subjects" (convex style combos)
      creates a denser coverage of the style space than just the N training subjects.
    - The base path (no FiLM) is always co-trained, preventing train/infer gap.
    - Validated on val subjects (from train pool) so no test info is used.

Compared to exp_31 (DisentangledCNNGRU):
    + FiLMLayer in latent space (content conditioned on mixed style)
    + Dual gesture loss with mixed-style path weight γ
    + mix_alpha (Beta parameter for style interpolation)
    Everything else identical: same encoder, same MI loss, same LOSO protocol.

Usage:
    python experiments/exp_60_mixstyle_content_disentangled_loso.py
    python experiments/exp_60_mixstyle_content_disentangled_loso.py --ci
    python experiments/exp_60_mixstyle_content_disentangled_loso.py \\
        --subjects DB2_s1,DB2_s12,DB2_s15,DB2_s28,DB2_s39
"""

import os
import sys
import json
import gc
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

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
from training.mixstyle_disentangled_trainer import MixStyleDisentangledTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ════════════════════════ EXPERIMENT SETTINGS ════════════════════════════

EXPERIMENT_NAME = "exp_60_mixstyle_content_disentangled"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# ── Disentanglement (inherited from exp_31 baseline) ──────────────────
CONTENT_DIM = 128
STYLE_DIM = 64
ALPHA = 0.5            # weight of subject classification loss
BETA = 0.1             # weight of MI minimization loss
BETA_ANNEAL_EPOCHS = 10
MI_LOSS_TYPE = "distance_correlation"

# ── MixStyle extensions ────────────────────────────────────────────────
GAMMA = 0.5            # weight of mixed-style gesture path vs base path
MIX_ALPHA = 0.4        # Beta(MIX_ALPHA, MIX_ALPHA) for style interpolation
                       # 0.4 gives λ mostly in [0.2, 0.8] — a broad mix range


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
    Build train / val / test splits with subject-provenance tracking.

    Subject labels are required by MixStyleDisentangledTrainer so that
    style mixing can be cross-subject within each training batch.

    LOSO guarantee: test_subject data goes ONLY into splits["test"].
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
    # Map train subject IDs to consecutive integer indices [0, ..., K-1]
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # ── Step 1: collect per-gesture arrays across training subjects ────
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

    # ── Step 2: split train → train / val per gesture ──────────────────
    # Permutation applied jointly to windows and subject labels so they stay aligned
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

    # ── Step 3: test split from test subject only ──────────────────────
    # Critical LOSO boundary: test_subject windows go ONLY here.
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
    Execute one LOSO fold with MixStyle-disentangled training.

    Returns a result dict with test_accuracy / test_f1_macro, or
    error info if the fold fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "mixstyle_disentangled_cnn_gru"

    # Save configs for reproducibility
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "mixstyle_config.json", "w") as f:
        json.dump({
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
            "mix_alpha": MIX_ALPHA,
        }, f, indent=4)

    # ── Data loading ──────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

    base_viz = Visualizer(output_dir, logger)

    # Load ALL subjects (train + test) in a single pass for efficiency.
    # The boundary is enforced in _build_splits_with_subject_labels where
    # test_subject windows go exclusively into splits["test"].
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

    # ── Build splits with subject provenance ──────────────────────────
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

    # ── Create trainer ────────────────────────────────────────────────
    trainer = MixStyleDisentangledTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        content_dim=CONTENT_DIM,
        style_dim=STYLE_DIM,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
        mi_loss_type=MI_LOSS_TYPE,
        mix_alpha=MIX_ALPHA,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "mixstyle_disentangled_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # ── Evaluate on held-out test subject ─────────────────────────────
    # Inference path uses only z_content (no FiLM, no style conditioning).
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
            "model_type": "mixstyle_disentangled_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy: applies transpose + standardisation then calls model in eval mode
    test_results = trainer.evaluate_numpy(
        X_test, y_test,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1 = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── Save fold results ─────────────────────────────────────────────
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
        make_json_serializable({
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "model_type": "mixstyle_disentangled_cnn_gru",
            "approach": APPROACH,
            "exercises": exercises,
            "mixstyle_config": {
                "content_dim": CONTENT_DIM,
                "style_dim": STYLE_DIM,
                "alpha": ALPHA,
                "beta": BETA,
                "gamma": GAMMA,
                "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
                "mi_loss_type": MI_LOSS_TYPE,
                "mix_alpha": MIX_ALPHA,
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # ── Memory cleanup ────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "mixstyle_disentangled_cnn_gru",
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
        model_type="mixstyle_disentangled_cnn_gru",
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
    print(f"Hypothesis H60: Mixture of Styles (MixStyle + FiLM in latent z-space)")
    print(f"Subjects:  {ALL_SUBJECTS}")
    print(f"Exercises: {EXERCISES}")
    print(f"Disentanglement: content_dim={CONTENT_DIM}, style_dim={STYLE_DIM}")
    print(f"Loss weights:  alpha={ALPHA} (subject), beta={BETA} (MI), gamma={GAMMA} (mix)")
    print(f"Style mixing:  mix_alpha={MIX_ALPHA} (Beta param), MI_type={MI_LOSS_TYPE}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "mixstyle_disentangled_cnn_gru" / f"test_{test_subject}"

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

    # ── Aggregate LOSO summary ────────────────────────────────────────
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]

    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        print(f"\n{'=' * 60}")
        print(
            f"MixStyle Disentangled CNN-GRU — LOSO Summary ({len(valid_results)} folds)"
        )
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "H60: Mixture of Styles — FiLM in latent z_style space",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "mixstyle_config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
            "mix_alpha": MIX_ALPHA,
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
            mark_hypothesis_verified("H60", metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H60", "All folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
