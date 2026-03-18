"""
Experiment 90: HyGDL — Analytical Orthogonal Content/Style Projection (LOSO)

Hypothesis
──────────
Learnable disentanglement (adversarial, VAE) suffers from instability and
approximate orthogonality.  Replacing it with a closed-form analytical projection
(inspired by HyGDL — Hyper Geodesic Disentanglement Learning) should:

  1. Guarantee exact mathematical orthogonality between z_content and z_style.
  2. Remove adversarial training dynamics entirely (no min-max instability).
  3. Simplify training: V_style is computed by SVD on training-subject mean
     embeddings — no extra trainable parameters, no posterior collapse.

Architecture
────────────
  Input (B, C, T)
    └─ ECAPA-TDNN Encoder  (SE-Res2Net + MFA + Attentive Stats Pooling + FC)
         → z ∈ R^E (embedding_dim=128)
       ├─ [Analytical projection — no gradients, no trainable params]
       │    V_style ∈ R^{E×k}  — SVD of training-subject mean embeddings
       │    z_style   = z @ V_style @ V_style^T   (projection onto style subspace)
       │    z_content = z − z_style               (guaranteed orthogonal complement)
       │
       ├─ Classifier(z_content)  →  gesture logits   [CE loss]
       └─ Decoder(z)             →  x̂ compressed     [MSE regulariser]

Training phases
───────────────
  Phase 1 (warmup_epochs=15):
    − Train encoder + classifier with CE loss only.
    − Encoder learns meaningful gesture representations without projection overhead.

  Phase 2 (remaining epochs):
    − At first Phase-2 epoch and every subspace_update_interval=5 epochs:
        * Forward-pass all training-subject windows through encoder (no_grad/eval).
        * Compute μ_s = mean_embedding(subject_s)  for each training subject s.
        * Stack {μ_s} → centre → SVD → V_style ∈ R^{E×k}.
        * Update model.V_style buffer (not a Parameter — no backprop path).
    − Train with: CE(classifier(z_content), y) + λ_rec * MSE(decoder(z), x̂_down)

LOSO protocol (strictly enforced — no test-subject adaptation)
──────────────────────────────────────────────────────────────
For each fold (test_subject ∈ ALL_SUBJECTS):
    train_subjects = ALL_SUBJECTS \ {test_subject}

  1. Load windows for ALL subjects (train + test) in one call.
  2. Build train split from train_subjects only.  Val is carved from train.
  3. Build per_subject_windows from train_subjects (for V_style computation).
  4. Compute channel mean/std from X_train only → apply to val/test unchanged.
  5. Phase-1 warmup: train encoder with full z (no projection).
  6. Phase-2 disentangle: update V_style from training-subject embeddings only,
     then train classifier on z_content.
  7. Test: model.eval() on test_subject → frozen V_style, frozen BN stats,
     no adaptation of any kind.

Data-leakage guards:
  ✓ _build_splits() populates train/val from train_subjects only.
  ✓ per_subject_windows populated from train_subjects only (test subject excluded).
  ✓ Channel mean/std from X_train only.
  ✓ V_style SVD: encoder(X_train_subj_s) in no_grad + eval mode — no test data.
  ✓ model.eval() at test inference: BN frozen to training running stats.
  ✓ No per-subject adaptation, no fine-tuning, no threshold tuning on test data.

Comparison baseline
────────────────────
  Baseline : ECAPATDNNEmg (exp_62, C=128, scale=4, embed=128) ≈ 467 K params
  Proposed : HyGDLModel   (same ECAPA encoder + projection + decoder)  ≈ 561 K params
  Same training config, same data pipeline, same LOSO protocol.

Run examples
────────────
  # 5-subject CI run (fast, safe default for server):
  python experiments/exp_90_hygdl_analytical_orthogonal_projection_loso.py --ci

  # Specific subjects:
  python experiments/exp_90_hygdl_analytical_orthogonal_projection_loso.py \\
      --subjects DB2_s1,DB2_s12,DB2_s15

  # Full 20-subject run:
  python experiments/exp_90_hygdl_analytical_orthogonal_projection_loso.py --full
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
    DEFAULT_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.hygdl_trainer import HyGDLTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ══════════════════════════ EXPERIMENT SETTINGS ═════════════════════════════

EXPERIMENT_NAME = "exp_90_hygdl_analytical_orthogonal_projection"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# ECAPA encoder hyper-parameters (same as exp_62 baseline for fair comparison)
ECAPA_CHANNELS      = 128
ECAPA_SCALE         = 4
ECAPA_EMBEDDING_DIM = 128
ECAPA_DILATIONS     = [2, 3, 4]
ECAPA_SE_REDUCTION  = 8

# HyGDL-specific hyper-parameters
STYLE_DIM                 = 4    # k — style subspace rank
T_COMPRESSED              = 75   # reconstruction target time steps (600//8)
WARMUP_EPOCHS             = 15   # Phase-1 epochs before first V_style update
SUBSPACE_UPDATE_INTERVAL  = 5    # Phase-2 epochs between V_style recomputations
LAMBDA_REC                = 0.1  # reconstruction loss weight


# ══════════════════════════ LOCAL HELPER ════════════════════════════════════


def grouped_to_arrays(grouped_windows: Dict[int, List[np.ndarray]]):
    """
    Convert grouped_windows to flat (windows, labels) arrays.

    NOTE: grouped_to_arrays does NOT exist in any processing/ module.
    This local helper must be defined in every experiment file that needs it.

    Returns:
        windows : (N, T, C) float32
        labels  : (N,)      int64  — values are gesture IDs, not class indices
    """
    all_windows, all_labels = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                all_windows.append(rep_arr)
                all_labels.append(np.full(len(rep_arr), gid, dtype=np.int64))
    if not all_windows:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(all_windows, axis=0).astype(np.float32),
        np.concatenate(all_labels,  axis=0),
    )


# ══════════════════════════ SPLITS BUILDER ══════════════════════════════════


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
    Construct train / val / test splits and per-subject windows for V_style.

    LOSO invariants upheld in this function:
      − train + val windows come ONLY from train_subjects.
      − val is carved from the train data by random permutation (no test leakage).
      − test windows come ONLY from test_subject.
      − per_subject_windows: training-subject windows collected before the train/val
        split so the trainer has access to the full per-subject corpus for computing
        more stable mean embeddings for V_style.
      − No signal statistics (mean, std) are computed here; normalisation happens
        inside HyGDLTrainer.fit() from X_train only.

    Args:
        subjects_data:   Dict returned by MultiSubjectLoader.load_multiple_subjects.
                         Keys: subject_id.  Values: (emg, segments, grouped_windows).
        train_subjects:  Subject IDs for training (must exclude test_subject).
        test_subject:    Subject ID held out for cross-subject evaluation.
        common_gestures: Gesture IDs present in all subjects.
        multi_loader:    MultiSubjectLoader instance (for filter_by_gestures).
        val_ratio:       Fraction of train windows held out for validation.
        seed:            RNG seed for reproducible val split.

    Returns:
        {
          "train":                Dict[int, np.ndarray],  gesture_id → (N, T, C)
          "val":                  Dict[int, np.ndarray],
          "test":                 Dict[int, np.ndarray],
          "per_subject_windows":  Dict[int, np.ndarray],  subj_idx → (N_s, T, C)
        }
    """
    rng = np.random.RandomState(seed)

    # ── Accumulate train windows per gesture AND per subject ──────────────
    # LOSO guard: only train_subjects are iterated; test_subject is excluded.
    train_dict: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}
    # per_subject_all: subj_idx → list of windows (all gestures concatenated)
    per_subject_all: Dict[int, List[np.ndarray]] = {}

    for s_idx, sid in enumerate(sorted(train_subjects)):
        if sid not in subjects_data:
            continue
        # subjects_data values are tuples: (emg, segments, grouped_windows)
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)

        subject_wins: List[np.ndarray] = []
        for gid, reps in filtered.items():
            for rep_arr in reps:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    train_dict[gid].append(rep_arr)
                    subject_wins.append(rep_arr)

        if subject_wins:
            per_subject_all[s_idx] = np.concatenate(subject_wins, axis=0).astype(np.float32)

    # ── Split train windows into train / val ─────────────────────────────
    # val is carved from train by random permutation — no test data is seen.
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)   # (N, T, C)
        n     = len(X_gid)
        perm  = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        if len(perm[n_val:]) > 0:
            final_train[gid] = X_gid[perm[n_val:]]
        if n_val > 0:
            final_val[gid]   = X_gid[perm[:n_val]]

    # ── Test split: test_subject data only ───────────────────────────────
    # Only test_subject windows appear here — never mixed with train data.
    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {
        "train":               final_train,
        "val":                 final_val,
        "test":                final_test,
        # Per-subject windows for V_style computation.
        # Contains training-subject data ONLY (test_subject excluded above).
        "per_subject_windows": per_subject_all,
    }


# ════════════════════════════ SINGLE FOLD ═══════════════════════════════════


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
    Execute one LOSO fold: train on train_subjects, evaluate on test_subject.

    Returns dict with at least "test_accuracy" and "test_f1_macro".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "hygdl_ecapa"

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as fh:
        json.dump(asdict(split_cfg), fh, indent=4)

    # ── Data loader ───────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    # ── Load ALL subjects (train + test) in a single call ─────────────────
    # We include test_subject so get_common_gestures() can identify gesture IDs
    # that exist in ALL subjects (including the held-out one).
    # No signal statistics are derived at this stage — only gesture ID sets.
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
        f"Common gestures across {len(all_subject_ids)} subjects: "
        f"{common_gestures}  ({len(common_gestures)} total)"
    )

    # ── Build LOSO-clean splits + per_subject_windows ─────────────────────
    # per_subject_windows is populated from train_subjects only (inside _build_splits).
    splits = _build_splits(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    for sname in ("train", "val", "test"):
        total_w = sum(
            len(arr)
            for arr in splits[sname].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"  {sname.upper():5s}: {total_w:5d} windows, "
            f"{len(splits[sname])} gestures"
        )
    logger.info(
        f"  per_subject_windows: "
        f"{len(splits['per_subject_windows'])} training subjects"
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = HyGDLTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        # HyGDL-specific
        warmup_epochs=WARMUP_EPOCHS,
        subspace_update_interval=SUBSPACE_UPDATE_INTERVAL,
        style_dim=STYLE_DIM,
        t_compressed=T_COMPRESSED,
        lambda_rec=LAMBDA_REC,
        # ECAPA encoder
        channels=ECAPA_CHANNELS,
        scale=ECAPA_SCALE,
        embedding_dim=ECAPA_EMBEDDING_DIM,
        dilations=ECAPA_DILATIONS,
        se_reduction=ECAPA_SE_REDUCTION,
    )

    # ── Training ──────────────────────────────────────────────────────────
    # fit() computes channel normalisation from X_train only, then:
    #   Phase 1 (warmup_epochs): CE loss on full z (no projection).
    #   Phase 2 (remaining):     V_style updated from per_subject_windows (train only),
    #                            CE(z_content) + λ_rec * MSE(decoder(z), x_down).
    try:
        training_results = trainer.fit(splits)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "hygdl_ecapa",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(exc),
        }

    # ── Cross-subject test evaluation ─────────────────────────────────────
    # Assemble flat test arrays ordered by class_ids from the trainer so that
    # class indices are consistent with what the model was trained on.
    class_ids = trainer.class_ids
    X_test_list: List[np.ndarray] = []
    y_test_list: List[np.ndarray] = []

    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error(f"No test windows for subject {test_subject}.")
        return {
            "test_subject":  test_subject,
            "model_type":    "hygdl_ecapa",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy() applies:
    #   (a) transpose to (N, C, T) if needed,
    #   (b) channel standardisation with TRAINING mean/std,
    #   (c) model.eval() — frozen BN running stats, frozen V_style,
    #   (d) analytical projection → classify on z_content.
    # No test-subject statistics or adaptation of any kind.
    test_results = trainer.evaluate_numpy(
        X_test_concat,
        y_test_concat,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Acc={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── Save fold results ─────────────────────────────────────────────────
    fold_summary = {
        "test_subject":       test_subject,
        "train_subjects":     train_subjects,
        "common_gestures":    common_gestures,
        "training":           training_results,
        "cross_subject_test": {
            "subject":          test_subject,
            "accuracy":         test_acc,
            "f1_macro":         test_f1,
            "report":           test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as fh:
        json.dump(
            make_json_serializable(fold_summary), fh, indent=4, ensure_ascii=False
        )

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":  test_subject,
            "train_subjects": train_subjects,
            "model_type":    "hygdl_ecapa",
            "approach":      APPROACH,
            "exercises":     exercises,
            "hygdl_config": {
                "channels":                 ECAPA_CHANNELS,
                "scale":                    ECAPA_SCALE,
                "embedding_dim":            ECAPA_EMBEDDING_DIM,
                "dilations":                ECAPA_DILATIONS,
                "se_reduction":             ECAPA_SE_REDUCTION,
                "style_dim":                STYLE_DIM,
                "t_compressed":             T_COMPRESSED,
                "warmup_epochs":            WARMUP_EPOCHS,
                "subspace_update_interval": SUBSPACE_UPDATE_INTERVAL,
                "lambda_rec":               LAMBDA_REC,
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # ── Memory cleanup ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data, splits
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    "hygdl_ecapa",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ══════════════════════════════ MAIN ════════════════════════════════════════


def main():
    """
    LOSO evaluation loop.

    Subject list priority:
      1. --subjects DB2_s1,DB2_s12,...  — explicit comma-separated list
      2. --full                          — all 20 DEFAULT_SUBJECTS
      3. --ci  (or default)              — 5 CI_TEST_SUBJECTS  ← safe server default

    The default (no flags) intentionally uses CI_TEST_SUBJECTS because the
    vast.ai server has symlinks only for those 5 subjects.
    """
    import argparse

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument(
        "--subjects", type=str, default=None,
        help="Comma-separated subject IDs, e.g. DB2_s1,DB2_s12,DB2_s15",
    )
    _parser.add_argument(
        "--ci", action="store_true",
        help="Use 5 CI test subjects",
    )
    _parser.add_argument(
        "--full", action="store_true",
        help="Use all 20 DEFAULT_SUBJECTS",
    )
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS      # safe default for server

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    # ── Configs ───────────────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )
    split_cfg = SplitConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )
    train_cfg = TrainingConfig(
        model_type="hygdl_ecapa",
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
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis  : Analytical orthogonal content/style projection (HyGDL)\n"
        f"              on ECAPA-TDNN embeddings vs ECAPA-TDNN baseline (exp_62)"
    )
    print(f"Subjects    : {ALL_SUBJECTS}  ({len(ALL_SUBJECTS)} total)")
    print(f"Exercises   : {EXERCISES}")
    print(
        f"ECAPA       : C={ECAPA_CHANNELS}, scale={ECAPA_SCALE}, "
        f"embed={ECAPA_EMBEDDING_DIM}, dilations={ECAPA_DILATIONS}"
    )
    print(
        f"HyGDL       : style_dim={STYLE_DIM}, warmup={WARMUP_EPOCHS}ep, "
        f"update_interval={SUBSPACE_UPDATE_INTERVAL}ep, lambda_rec={LAMBDA_REC}"
    )
    print(f"Output      : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results: List[Dict] = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / "hygdl_ecapa" / f"test_{test_subject}"

        result = run_single_loso_fold(
            base_dir=BASE_DIR,
            output_dir=fold_dir,
            train_subjects=train_subjects,
            test_subject=test_subject,
            exercises=EXERCISES,
            proc_cfg=proc_cfg,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
        )
        all_results.append(result)

    # ── Aggregate LOSO summary ────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"]  for r in valid]
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        mean_f1  = float(np.mean(f1s))
        std_f1   = float(np.std(f1s))

        print("\n" + "=" * 80)
        print("LOSO SUMMARY — HyGDL Analytical Orthogonal Projection")
        print(f"  Subjects evaluated : {len(valid)}")
        print(
            f"  Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}"
            f"  (min={min(accs):.4f}, max={max(accs):.4f})"
        )
        print(
            f"  F1-macro  : {mean_f1:.4f} ± {std_f1:.4f}"
            f"  (min={min(f1s):.4f}, max={max(f1s):.4f})"
        )
        print("=" * 80)

        summary = {
            "experiment": EXPERIMENT_NAME,
            "model":      "hygdl_ecapa",
            "approach":   APPROACH,
            "subjects":   ALL_SUBJECTS,
            "exercises":  EXERCISES,
            "hygdl_config": {
                "channels":                 ECAPA_CHANNELS,
                "scale":                    ECAPA_SCALE,
                "embedding_dim":            ECAPA_EMBEDDING_DIM,
                "dilations":                ECAPA_DILATIONS,
                "se_reduction":             ECAPA_SE_REDUCTION,
                "style_dim":                STYLE_DIM,
                "t_compressed":             T_COMPRESSED,
                "warmup_epochs":            WARMUP_EPOCHS,
                "subspace_update_interval": SUBSPACE_UPDATE_INTERVAL,
                "lambda_rec":               LAMBDA_REC,
            },
            "loso_metrics": {
                "mean_accuracy": mean_acc,
                "std_accuracy":  std_acc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro":  std_f1,
                "per_subject":   all_results,
            },
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_ROOT / "loso_summary.json", "w") as fh:
            json.dump(make_json_serializable(summary), fh, indent=4, ensure_ascii=False)
        print(f"Summary saved → {OUTPUT_ROOT / 'loso_summary.json'}")
    else:
        print("No successful folds to summarise.")

    # ── Optional: report to hypothesis executor ───────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        hypothesis_id = os.environ.get("HYPOTHESIS_ID", "")
        if hypothesis_id and valid:
            mark_hypothesis_verified(
                hypothesis_id,
                metrics={
                    "mean_accuracy": mean_acc,
                    "std_accuracy":  std_acc,
                    "mean_f1_macro": mean_f1,
                    "std_f1_macro":  std_f1,
                    "n_folds":       len(valid),
                },
                experiment_name=EXPERIMENT_NAME,
            )
        elif hypothesis_id and not valid:
            mark_hypothesis_failed(
                hypothesis_id,
                error_message="All LOSO folds failed — no valid results.",
            )
    except ImportError:
        pass   # hypothesis_executor not installed in this environment


if __name__ == "__main__":
    main()
