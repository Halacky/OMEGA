"""
Experiment 110: Multi-Resolution Aligned Disentanglement — LOSO.

Hypothesis 3: Two-stage scheme — Alignment first, Disentanglement second —
inspired by "From Consistency to Complementarity" (2026).

Core idea
---------
Treat the EMG signal as multimodal by decomposing it into 3 frequency bands:
  Band 0: 0–200 Hz   — inter-subject variance dominated (fat / impedance)
  Band 1: 200–500 Hz — motor-unit recruitment patterns, gesture-discriminative
  Band 2: 500–1000 Hz — high-frequency residual

Stage 1 — Alignment (contrastive):
  Per-band Mini-ECAPA encoders + NT-Xent loss across all 3 band-pairs.
  "Same window seen through different frequency lenses" = positive pair.
  Creates a shared gesture representation invariant to frequency content.

Stage 2 — Disentanglement (complementarity):
  Per-band specific encoders + Gradient Reversal Layer (GRL).
  GRL ensures specific features become maximally uninformative about gesture →
  they capture band-specific subject variation instead.
  Only the aligned representation is used for classification.

Motivation: resolves the simultaneous-optimization conflict in exp_31/57/59/60/89
by separating alignment and disentanglement objectives.

LOSO data-leakage audit
------------------------
✓ Each fold: test subject goes exclusively to splits["test"]; never in train/val.
✓ Channel standardization (mean_c, std_c) computed from X_train only.
✓ FreqBandSplitter: purely physics-based FFT masking — no data statistics.
✓ SoftAGC parameters: gradients from training subjects only.
✓ NT-Xent: positive/negative pairs within each training batch; test not present.
✓ Adversarial loss: uses gesture labels from training batch only.
✓ Validation loss drives early stopping; no test information.
✓ evaluate_numpy(): model.eval() + stage=1; BN frozen; no test-time adaptation.

Usage
-----
    python experiments/exp_110_multi_res_aligned_disentangle_loso.py
    python experiments/exp_110_multi_res_aligned_disentangle_loso.py --ci
    python experiments/exp_110_multi_res_aligned_disentangle_loso.py \\
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

# ── Project imports ────────────────────────────────────────────────────────────
from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.multi_res_aligned_disentangle_trainer import (
    MultiResAlignedDisentangleTrainer,
)
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ════════════════════════ EXPERIMENT SETTINGS ════════════════════════════════

EXPERIMENT_NAME = "exp_110_multi_res_aligned_disentangle"
APPROACH        = "deep_raw"
EXERCISES       = ["E1",]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES    = 10
SAMPLING_RATE   = 2000  # Hz (NinaPro DB2)

# ── Frequency-band geometry ───────────────────────────────────────────────────
# Band 0: [   0, F_LOW)  Hz  — inter-subject variance dominant
# Band 1: [F_LOW, F_MID) Hz  — gesture-discriminative motor-unit patterns
# Band 2: [F_MID, fs/2]  Hz  — high-frequency residual
F_LOW = 200.0   # Hz
F_MID = 500.0   # Hz

# ── Stage-1: Alignment ────────────────────────────────────────────────────────
STAGE1_EPOCHS = 40          # epochs before adversarial disentanglement starts
ALPHA_ALIGN   = 0.5         # weight for NT-Xent alignment loss
TEMPERATURE   = 0.1         # NT-Xent temperature (lower = sharper contrast)

# ── Stage-2: Adversarial disentanglement ─────────────────────────────────────
BETA_ADV  = 0.3             # weight for adversarial (GRL) loss
GRL_ALPHA = 1.0             # gradient reversal magnitude

# ── Encoder geometry ─────────────────────────────────────────────────────────
CHANNELS       = 64         # MiniECAPAEncoder internal channel width
EMBED_DIM      = 64         # aligned encoder output dimension
PROJ_DIM       = 32         # projection head output for NT-Xent
SPEC_HIDDEN    = 32         # SpecificEncoder hidden conv channels
SPEC_EMBED_DIM = 32         # SpecificEncoder output dimension


# ════════════════════════ DATA PREPARATION ════════════════════════════════════


def grouped_to_arrays(grouped_windows: Dict[int, List[np.ndarray]]):
    """
    Convert grouped_windows dict to flat (windows, labels) arrays.

    grouped_windows: {gesture_id: [rep_array_1, rep_array_2, ...]}
        each rep_array: (N_rep, T, C).

    Returns:
        windows: (N_total, T, C) float32
        labels:  (N_total,) int64  — gesture IDs (not class indices)
    """
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep in grouped_windows[gid]:
            if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0:
                windows_list.append(rep)
                labels_list.append(np.full(len(rep), gid, dtype=np.int64))
    if not windows_list:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(windows_list, axis=0).astype(np.float32),
        np.concatenate(labels_list,  axis=0),
    )


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
    Build train / val / test splits for a single LOSO fold.

    LOSO guarantee
    --------------
    Test-subject windows go EXCLUSIVELY into splits["test"].
    Training-subject windows are split into train (1−val_ratio) and val (val_ratio).
    The test subject's windows NEVER appear in train or val.

    Returns
    -------
    {
        "train": Dict[gesture_id → np.ndarray (N, T, C)],
        "val":   Dict[gesture_id → np.ndarray (N, T, C)],
        "test":  Dict[gesture_id → np.ndarray (N, T, C)],
    }
    """
    rng = np.random.RandomState(seed)

    # ── Step 1: collect windows from training subjects ────────────────────
    train_dict: Dict[int, np.ndarray] = {}
    for gid in common_gestures:
        windows_for_gid = []
        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            # subjects_data values are tuples (emg, segments, grouped_windows)
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid not in filtered:
                continue
            for rep_array in filtered[gid]:
                if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                    windows_for_gid.append(rep_array)
        if windows_for_gid:
            train_dict[gid] = np.concatenate(windows_for_gid, axis=0)

    # ── Step 2: per-gesture permuted train / val split ────────────────────
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}
    for gid, X_g in train_dict.items():
        n     = len(X_g)
        perm  = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        # Val = first n_val indices; train = the rest
        final_val[gid]   = X_g[perm[:n_val]]
        final_train[gid] = X_g[perm[n_val:]]

    # ── Step 3: test split from test subject ONLY (LOSO boundary) ─────────
    test_dict: Dict[int, np.ndarray] = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered  = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
        if valid:
            test_dict[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": test_dict}


# ════════════════════════ SINGLE LOSO FOLD ════════════════════════════════════


def run_single_loso_fold(
    base_dir:       Path,
    output_dir:     Path,
    train_subjects: List[str],
    test_subject:   str,
    exercises:      List[str],
    proc_cfg:       ProcessingConfig,
    split_cfg:      SplitConfig,
    train_cfg:      TrainingConfig,
) -> Dict:
    """
    Execute one LOSO fold of Multi-Resolution Aligned Disentanglement.

    Returns dict with test_accuracy / test_f1_macro, or error info.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "multi_res_aligned_disentangle"

    # ── Save configs ───────────────────────────────────────────────────────
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "mrad_config.json", "w") as f:
        json.dump(
            {
                "f_low":          F_LOW,
                "f_mid":          F_MID,
                "stage1_epochs":  STAGE1_EPOCHS,
                "alpha_align":    ALPHA_ALIGN,
                "temperature":    TEMPERATURE,
                "beta_adv":       BETA_ADV,
                "grl_alpha":      GRL_ALPHA,
                "channels":       CHANNELS,
                "embed_dim":      EMBED_DIM,
                "proj_dim":       PROJ_DIM,
                "spec_hidden":    SPEC_HIDDEN,
                "spec_embed_dim": SPEC_EMBED_DIM,
                "sampling_rate":  SAMPLING_RATE,
            },
            f,
            indent=4,
        )

    # ── Data loading ───────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

    # Load all subjects (train + test) in one pass.
    # The LOSO boundary is enforced in _build_splits() below.
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
    logger.info(
        f"Common gestures ({len(common_gestures)}): {common_gestures}"
    )

    # ── Build splits — LOSO boundary enforced here ─────────────────────────
    splits = _build_splits(
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
            f"{split_name.upper()}: {total} windows, "
            f"{len(splits[split_name])} gesture classes"
        )

    # ── Build trainer ──────────────────────────────────────────────────────
    trainer = MultiResAlignedDisentangleTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        sampling_rate=SAMPLING_RATE,
        f_low=F_LOW,
        f_mid=F_MID,
        channels=CHANNELS,
        embed_dim=EMBED_DIM,
        proj_dim=PROJ_DIM,
        spec_hidden=SPEC_HIDDEN,
        spec_embed_dim=SPEC_EMBED_DIM,
        stage1_epochs=STAGE1_EPOCHS,
        alpha_align=ALPHA_ALIGN,
        beta_adv=BETA_ADV,
        temperature=TEMPERATURE,
        grl_alpha=GRL_ALPHA,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "multi_res_aligned_disentangle",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── Build test arrays from test split ──────────────────────────────────
    # trainer.class_ids gives gesture IDs in the order they were indexed.
    # We must assign class index i to class_ids[i] — same mapping as inside fit().
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for class_idx, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(
                    np.full(len(arr), class_idx, dtype=np.int64)
                )

    if not X_test_list:
        logger.error("No test data available after gesture filtering.")
        return {
            "test_subject":  test_subject,
            "model_type":    "multi_res_aligned_disentangle",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test_np = np.concatenate(X_test_list, axis=0)   # (N, T, C) from splits
    y_test_np = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy() handles (N,T,C)→(N,C,T) transpose + standardisation.
    # model.eval() + stage=1: specific encoders not activated on test data.
    test_results = trainer.evaluate_numpy(
        X_test_np, y_test_np,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── Save fold results ──────────────────────────────────────────────────
    results_to_save = {
        "test_subject":     test_subject,
        "train_subjects":   train_subjects,
        "common_gestures":  common_gestures,
        "training":         training_results,
        "cross_subject_test": {
            "subject":          test_subject,
            "accuracy":         test_acc,
            "f1_macro":         test_f1,
            "report":           test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(
            make_json_serializable(results_to_save), f,
            indent=4, ensure_ascii=False
        )

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable(
            {
                "test_subject":   test_subject,
                "train_subjects": train_subjects,
                "model_type":     "multi_res_aligned_disentangle",
                "approach":       APPROACH,
                "exercises":      exercises,
                "mrad_config": {
                    "f_low":          F_LOW,
                    "f_mid":          F_MID,
                    "stage1_epochs":  STAGE1_EPOCHS,
                    "alpha_align":    ALPHA_ALIGN,
                    "temperature":    TEMPERATURE,
                    "beta_adv":       BETA_ADV,
                    "grl_alpha":      GRL_ALPHA,
                    "channels":       CHANNELS,
                    "embed_dim":      EMBED_DIM,
                    "proj_dim":       PROJ_DIM,
                    "spec_hidden":    SPEC_HIDDEN,
                    "spec_embed_dim": SPEC_EMBED_DIM,
                    "sampling_rate":  SAMPLING_RATE,
                },
                "metrics": {
                    "test_accuracy": test_acc,
                    "test_f1_macro": test_f1,
                },
            }
        ),
        filename="fold_metadata.json",
    )

    # ── Memory cleanup ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    "multi_res_aligned_disentangle",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════ MAIN ════════════════════════════════════════════════


def main():
    # Codegen rule 24: parse_subjects_args() defaults to CI_TEST_SUBJECTS.
    # The server (vast.ai) has symlinks only for CI subjects.
    ALL_SUBJECTS = parse_subjects_args()

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = (
        ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    )

    # ── Processing config ─────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=SAMPLING_RATE,
        segment_edge_margin=0.1,
    )

    # ── Split config ──────────────────────────────────────────────────────
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    # ── Training config ───────────────────────────────────────────────────
    # Total epochs = STAGE1_EPOCHS (alignment) + up to STAGE1_EPOCHS (adv).
    # Early stopping is reset at stage boundary, so each stage gets its own
    # patience budget.
    total_epochs = STAGE1_EPOCHS * 2  # 80 epochs total by default

    train_cfg = TrainingConfig(
        model_type="multi_res_aligned_disentangle",
        pipeline_type=APPROACH,
        use_handcrafted_features=False,
        batch_size=64,
        epochs=total_epochs,
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
    print(
        f"Hypothesis 3: Multi-Resolution Aligned Disentanglement "
        f"(Alignment → Disentanglement)"
    )
    print(f"  Bands:         0–{F_LOW:.0f} Hz | {F_LOW:.0f}–{F_MID:.0f} Hz | {F_MID:.0f}–{SAMPLING_RATE//2} Hz")
    print(f"  Stage 1 epochs: {STAGE1_EPOCHS}  (align α={ALPHA_ALIGN}, T={TEMPERATURE})")
    print(f"  Stage 2 epochs: {total_epochs - STAGE1_EPOCHS}  (adv β={BETA_ADV}, GRL α={GRL_ALPHA})")
    print(f"  Subjects:      {ALL_SUBJECTS}")
    print(f"  Output:        {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output    = (
            OUTPUT_ROOT
            / "multi_res_aligned_disentangle"
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

    # ── LOSO summary ──────────────────────────────────────────────────────
    valid = [r for r in all_loso_results if r.get("test_accuracy") is not None]

    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"] for r in valid]
        print(f"\n{'=' * 60}")
        print(
            f"Multi-Res Aligned Disentanglement — "
            f"LOSO Summary ({len(valid)} folds)"
        )
        print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    summary = {
        "experiment":  EXPERIMENT_NAME,
        "hypothesis":  "H3: Multi-Resolution Aligned Disentanglement",
        "timestamp":   TIMESTAMP,
        "subjects":    ALL_SUBJECTS,
        "approach":    APPROACH,
        "mrad_config": {
            "f_low":          F_LOW,
            "f_mid":          F_MID,
            "stage1_epochs":  STAGE1_EPOCHS,
            "alpha_align":    ALPHA_ALIGN,
            "temperature":    TEMPERATURE,
            "beta_adv":       BETA_ADV,
            "grl_alpha":      GRL_ALPHA,
            "channels":       CHANNELS,
            "embed_dim":      EMBED_DIM,
            "proj_dim":       PROJ_DIM,
            "spec_hidden":    SPEC_HIDDEN,
            "spec_embed_dim": SPEC_EMBED_DIM,
            "sampling_rate":  SAMPLING_RATE,
        },
        "results": all_loso_results,
    }

    if valid:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_folds":     len(valid),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
