"""
Experiment 58: MI Upper-Bound Subject Adversary for LOSO EMG (Hypothesis H5b)

Motivation
----------
Exp_31 used distance correlation as a surrogate for MI(z_content; subject).
Distance correlation is a useful sanity check but is a global scalar that may
fail to capture asymmetric, high-dimensional dependencies.  The Gradient
Reversal Layer (GRL) in earlier experiments collapsed because it applies an
undifferentiated backward pressure.

CLUB (Contrastive Log-ratio Upper Bound) gives a tighter, differentiable
upper bound on MI that allows the encoder to selectively erase subject
information from z_content while keeping gesture signal intact.

Training objective per batch
----------------------------
    L = L_gesture(z_content)
      + α · L_subject(z_style)              [CE on style branch — makes style discriminative]
      + β · Î_CLUB(z_content; subject)      [minimise → push subject info out of content]
      − γ · Î_CLUB(z_style;   subject)      [maximise → pull subject info into style]

Two-step update (separate optimisers):
    Step 1 — update CLUB q_θ networks on z.detach()
    Step 2 — update main model; gradients flow: Î → z → encoder

Key diagnostic: subject-probe accuracy
---------------------------------------
After training, a linear LogisticRegression is fitted on z_content vectors from
training windows (training subjects only) and evaluated on val windows (also
training subjects only).  The probe val-accuracy measures subject discriminability
remaining in z_content.  Successful disentanglement → probe val-accuracy → chance.

Probe is computed EXCLUSIVELY on training-fold subjects.  The LOSO test subject
is never touched by CLUB, the probe, or any adaptation step.

LOSO invariant
--------------
    - load_multiple_subjects: all_subjects = train_subjects + [test_subject]
    - _build_splits: train/val from train_subjects ONLY
                     test from test_subject ONLY
    - CLUB training: windows from train_subjects, z.detach()
    - Subject probe: z_content from train windows + val windows (train subjects)
    - evaluate_numpy: only gesture_logits from z_content; no subject label used
"""

import gc
import json
import os
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
from training.mi_disentangled_trainer import MIDisentangledTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ═══════════════════════════════════════════════════════════════════════════
# Experiment settings
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_58_mi_subject_adversary"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# ── Architecture ──────────────────────────────────────────────────────────
CONTENT_DIM = 128
STYLE_DIM = 64

# ── Loss weights ──────────────────────────────────────────────────────────
# α: subject CE on z_style (makes style branch predict subject)
ALPHA = 0.5
# β: CLUB upper bound on MI(z_content; subject)  — minimise this
BETA = 0.3
# γ: CLUB upper bound on MI(z_style; subject)   — maximise this (optional)
#    Set to 0.0 to rely only on α for style alignment.
GAMMA = 0.05
# Anneal β and γ from 0 over this many epochs (stabilises early training)
BETA_ANNEAL_EPOCHS = 10
GAMMA_ANNEAL_EPOCHS = 10

# ── CLUB estimator ────────────────────────────────────────────────────────
CLUB_HIDDEN_DIM = 64
CLUB_LR = 5e-4          # Separate LR for CLUB; lower than main model


# ═══════════════════════════════════════════════════════════════════════════
# Split builder (with subject provenance for CLUB and probe)
# ═══════════════════════════════════════════════════════════════════════════


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
    Build LOSO splits with subject-identity tracking.

    Returns a splits dict with:
        "train":               Dict[int, np.ndarray]   gesture_id → windows (N,T,C)
        "val":                 Dict[int, np.ndarray]
        "test":                Dict[int, np.ndarray]   (from test_subject only)
        "train_subject_labels": Dict[int, np.ndarray]  gesture_id → subject index
        "val_subject_labels":   Dict[int, np.ndarray]  gesture_id → subject index
        "num_train_subjects":  int

    LOSO invariant:
        - train / val / train_subject_labels / val_subject_labels
          are built from train_subjects ONLY.
        - test is built from test_subject ONLY.
        - subject index 0 .. N_train−1 corresponds to sorted(train_subjects).
    """
    rng = np.random.RandomState(seed)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # ── Collect all windows + subject labels per gesture ──────────────────
    raw_train_windows: Dict[int, list] = {gid: [] for gid in common_gestures}
    raw_train_subjlbl: Dict[int, list] = {gid: [] for gid in common_gestures}

    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        subj_idx = train_subject_to_idx[sid]

        for gid in common_gestures:
            if gid not in filtered:
                continue
            for rep_array in filtered[gid]:
                if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                    raw_train_windows[gid].append(rep_array)
                    raw_train_subjlbl[gid].append(
                        np.full(len(rep_array), subj_idx, dtype=np.int64)
                    )

    # Concatenate per gesture
    all_windows_by_gest: Dict[int, np.ndarray] = {}
    all_subjlbl_by_gest: Dict[int, np.ndarray] = {}
    for gid in common_gestures:
        if raw_train_windows[gid]:
            all_windows_by_gest[gid] = np.concatenate(raw_train_windows[gid], axis=0)
            all_subjlbl_by_gest[gid] = np.concatenate(raw_train_subjlbl[gid], axis=0)

    # ── Train / val split (per gesture, aligned permutation) ──────────────
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}
    final_val_subj:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if gid not in all_windows_by_gest:
            continue
        X = all_windows_by_gest[gid]
        S = all_subjlbl_by_gest[gid]
        n = len(X)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        final_train[gid]      = X[train_idx]
        final_val[gid]        = X[val_idx]
        final_train_subj[gid] = S[train_idx]
        final_val_subj[gid]   = S[val_idx]

    # ── Test split from test_subject (no subject label needed at test time) ─
    final_test: Dict[int, np.ndarray] = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
        if valid:
            final_test[gid] = np.concatenate(valid, axis=0)

    return {
        "train":                final_train,
        "val":                  final_val,
        "test":                 final_test,
        "train_subject_labels": final_train_subj,
        "val_subject_labels":   final_val_subj,
        "num_train_subjects":   num_train_subjects,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single LOSO fold
# ═══════════════════════════════════════════════════════════════════════════


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
    Execute one LOSO fold.

    Data flow (LOSO-safe):
        load_multiple_subjects(train + [test_subject])
            → _build_splits (train/val from train_subjects; test from test_subject)
            → MIDisentangledTrainer.fit(splits)
                  ↳ CLUB trained on train windows only
                  ↳ subject probe on train/val z_content (training subjects only)
                  ↳ model checkpoint saved
            → evaluate_numpy(X_test, y_test)  — gesture only, no subject label
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "mi_disentangled_cnn_gru"

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # ── Load data ─────────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

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

    # ── Build splits with subject provenance ──────────────────────────────
    splits = _build_splits_with_subject_labels(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    for split_name in ("train", "val", "test"):
        total = sum(
            len(arr)
            for arr in splits[split_name].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"{split_name.upper()}: {total} windows across "
            f"{len(splits[split_name])} gestures"
        )

    # ── Create trainer ────────────────────────────────────────────────────
    base_viz = Visualizer(output_dir, logger)
    trainer = MIDisentangledTrainer(
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
        gamma_anneal_epochs=GAMMA_ANNEAL_EPOCHS,
        club_hidden_dim=CLUB_HIDDEN_DIM,
        club_lr=CLUB_LR,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "mi_disentangled_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # ── Cross-subject evaluation on the LOSO test subject ─────────────────
    # No subject labels used — inference is purely gesture classification
    class_ids = trainer.class_ids
    X_test_parts, y_test_parts = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_parts.append(arr)
                y_test_parts.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_parts:
        logger.error("No test data available after split.")
        return {
            "test_subject": test_subject,
            "model_type": "mi_disentangled_cnn_gru",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test_concat = np.concatenate(X_test_parts, axis=0)   # (N, T, C)
    y_test_concat = np.concatenate(y_test_parts, axis=0)

    test_results = trainer.evaluate_numpy(
        X_test_concat,
        y_test_concat,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    # Subject probe accuracy (from training_results, computed during fit)
    probe = training_results.get("subject_probe", {})
    probe_val_acc = probe.get("val_acc", float("nan"))

    print(
        f"[LOSO] test={test_subject} | "
        f"acc={test_acc:.4f}  f1={test_f1:.4f} | "
        f"subj_probe_val_acc={probe_val_acc:.4f}"
        if probe_val_acc == probe_val_acc  # not NaN
        else
        f"[LOSO] test={test_subject} | acc={test_acc:.4f}  f1={test_f1:.4f}"
    )

    # ── Save fold results ─────────────────────────────────────────────────
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
        "subject_probe": probe,
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "mi_disentangled_cnn_gru",
        "approach": APPROACH,
        "exercises": exercises,
        "disentanglement_config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "gamma_anneal_epochs": GAMMA_ANNEAL_EPOCHS,
            "club_hidden_dim": CLUB_HIDDEN_DIM,
            "club_lr": CLUB_LR,
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "subject_probe_val_acc": probe_val_acc,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # ── Memory cleanup ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data, splits
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "mi_disentangled_cnn_gru",
        "approach": APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "subject_probe_val_acc": probe_val_acc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


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
        model_type="mi_disentangled_cnn_gru",
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
    print(f"Hypothesis H5b: CLUB MI upper-bound subject adversary")
    print(f"Subjects: {ALL_SUBJECTS}")
    print(f"Exercises: {EXERCISES}")
    print(f"content_dim={CONTENT_DIM}, style_dim={STYLE_DIM}")
    print(f"α={ALPHA} (subject CE)  β={BETA} (CLUB content, min)  γ={GAMMA} (CLUB style, max)")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "mi_disentangled_cnn_gru" / f"test_{test_subject}"

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

    # ── Aggregate results ─────────────────────────────────────────────────
    valid = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    if valid:
        accs   = [r["test_accuracy"]   for r in valid]
        f1s    = [r["test_f1_macro"]   for r in valid]
        probes = [
            r["subject_probe_val_acc"]
            for r in valid
            if r.get("subject_probe_val_acc") == r.get("subject_probe_val_acc")  # filter NaN
        ]

        print(f"\n{'=' * 60}")
        print(f"MI-Disentangled CNN-GRU — LOSO Summary ({len(valid)} folds)")
        print(f"  Gesture Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Gesture F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        if probes:
            n_subj = len(ALL_SUBJECTS) - 1  # training subjects per fold
            chance = 1.0 / n_subj if n_subj > 0 else float("nan")
            print(
                f"  Subject probe acc: {np.mean(probes):.4f} ± {np.std(probes):.4f}"
                f"  (chance≈{chance:.4f})"
            )
        print(f"{'=' * 60}\n")

    # ── Save summary ──────────────────────────────────────────────────────
    summary: Dict = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "H5b: CLUB MI upper-bound subject adversary",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "disentanglement_config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "gamma_anneal_epochs": GAMMA_ANNEAL_EPOCHS,
            "club_hidden_dim": CLUB_HIDDEN_DIM,
            "club_lr": CLUB_LR,
        },
        "results": all_loso_results,
    }

    if valid:
        summary["aggregate"] = {
            "mean_accuracy":   float(np.mean(accs)),
            "std_accuracy":    float(np.std(accs)),
            "mean_f1_macro":   float(np.mean(f1s)),
            "std_f1_macro":    float(np.std(f1s)),
            "num_folds":       len(valid),
        }
        if probes:
            summary["aggregate"]["mean_subject_probe_val_acc"] = float(np.mean(probes))
            summary["aggregate"]["std_subject_probe_val_acc"]  = float(np.std(probes))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── Report to hypothesis_executor if available ────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        if valid:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
            }
            if probes:
                metrics["mean_subject_probe_val_acc"] = float(np.mean(probes))
            mark_hypothesis_verified("H5b", metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H5b", "All LOSO folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
