"""
Experiment 77: Subject-Invariance via Stochastic Hypernetwork FIR Domain Randomization

Hypothesis:
    Different subjects correspond to different electrode-skin transfer functions
    — unknown linear FIR filters on the underlying neural drive.  At test time
    (LOSO), the target transfer function is completely unknown.

    Instead of learning a single fixed deconvolution filter (exp_65), we train a
    hypernetwork that generates per-channel depthwise FIR coefficients from a
    random noise vector u ~ N(0, I).  A fresh random filter realization is applied
    to every training sample, so the CNN-BiGRU-Attention backbone is forced to
    learn features that are invariant to the particular filter realization.

    This is analogous to:
        • Domain randomization in sim-to-real robotics:
          randomize physics parameters (friction, mass) → robust policy.
        • Randomized spectral augmentation in robust ASR:
          random convolutive channel → channel-robust speech features.

    At test time: u = 0 → canonical (near-identity) filter.  The model is fully
    deterministic; no test-subject information is used anywhere.

What is tested:
    • FIRHyperNetwork: 3-layer MLP (noise_dim → hidden → hidden → n_ch × filter_len).
      Initialized so that hypernetwork(0) == identity filter for every channel.
    • StochasticFIRFrontend: per-sample depthwise 1D FIR via grouped Conv1d.
      Training: u ~ N(0, I) per batch element → domain randomization.
      Inference: u = 0  → canonical deterministic path.
    • Regularization (on canonical filter only — no data involvement):
        - Second-order smoothness: penalizes curvature of the u=0 filter.
        - Band-limiting: penalizes high-frequency energy above cutoff_ratio × Nyquist.
    • CNN-BiGRU-Attention backbone: identical architecture to exp_65 for fair comparison.

Why regularizing the canonical filter (u=0) rather than sampled realizations?
    Regularizing sampled filters would constrain the diversity of augmentations —
    the opposite of what we want.  Instead we constrain the MEAN filter to stay
    smooth and physically plausible, while allowing sampled realizations to freely
    explore the neighbourhood.  This decouples "augmentation diversity" (good) from
    "canonical-filter plausibility" (also good).

LOSO protocol (strictly enforced, zero leakage):
    ┌────────────────────────────────────────────────────────────────────┐
    │  For each fold (test_subject = one of the N subjects):            │
    │    train_subjects = all_subjects − {test_subject}                 │
    │    1. Load windows for all subjects.                              │
    │    2. Pool train-subject windows → train (+ val held-out).       │
    │    3. Test-subject windows → test split (NEVER seen in training). │
    │    4. Channel mean/std from TRAIN windows ONLY.                  │
    │    5. Train StochasticFIRCNNGRU:                                  │
    │       − sample u ~ N(0,I) per batch during training              │
    │       − hypernetwork(u) → per-sample depthwise FIR coefficients  │
    │       − loss = CrossEntropy + canonical-filter regularization     │
    │    6. Evaluate frozen model on test-subject windows.              │
    │       − u = 0 (deterministic), model.eval() (frozen BN)          │
    └────────────────────────────────────────────────────────────────────┘

Leakage prevention summary:
    ✓  Channel normalization from X_train exclusively.
    ✓  No subject IDs in the model at any point — only noise vectors.
    ✓  u ~ N(0, I) is independent of subject identity.
    ✓  Regularization on canonical (u=0) filter only — zero data involvement.
    ✓  evaluate_numpy: u=0 → deterministic, no test-subject statistics.
    ✓  model.eval() → frozen BatchNorm running stats, no updates.
    ✓  val split carved from train subjects via random permutation (same as exp_65).

Run examples:
    # 5-subject CI run (fast, default):
    python experiments/exp_77_stochastic_hypernetwork_fir_deconv_loso.py

    # Same, explicit:
    python experiments/exp_77_stochastic_hypernetwork_fir_deconv_loso.py --ci

    # Specific subjects:
    python experiments/exp_77_stochastic_hypernetwork_fir_deconv_loso.py \\
        --subjects DB2_s1,DB2_s12,DB2_s15

    # Full 20-subject run:
    python experiments/exp_77_stochastic_hypernetwork_fir_deconv_loso.py --full
"""

import gc
import json
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
from training.stochastic_fir_trainer import StochasticFIRTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ════════════════════════════════════════════════════════════════════════════
#  Experiment settings
# ════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_77_stochastic_hypernetwork_fir_deconv"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# FIR hyper-parameters
FILTER_LEN   = 63      # taps (odd); 63 × (1/2000 s) ≈ 31.5 ms — same as exp_65
NOISE_DIM    = 16      # noise vector dimension; small to prevent subject memorization
HYPER_HIDDEN = 64      # hypernetwork MLP hidden dim

# Regularization on the canonical (u=0) filter
LAMBDA_SMOOTH = 5e-3   # second-order curvature penalty (same weight as exp_65 smooth)
LAMBDA_BAND   = 1e-3   # high-frequency spectral energy penalty
CUTOFF_RATIO  = 0.5    # penalize energy above 0.5 × Nyquist (above 500 Hz @ 2000 Hz)

# Backbone hyper-parameters (identical to exp_65 for fair comparison)
CNN_CHANNELS = (64, 128, 256)
GRU_HIDDEN   = 128
NUM_HEADS    = 4


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _build_splits(
    subjects_data:   Dict,
    train_subjects:  List[str],
    test_subject:    str,
    common_gestures: List[int],
    multi_loader:    MultiSubjectLoader,
    val_ratio:       float = 0.15,
    seed:            int   = 42,
) -> Dict:
    """
    Construct LOSO train / val / test splits from loaded subject data.

    LOSO contract enforced:
      • train + val windows exclusively from `train_subjects`.
      • val split carved by random permutation from train data.
      • test windows exclusively from `test_subject`.
      • test-subject data assembled AFTER train/val split — never influences
        any train-side statistics or val_ratio computation.

    Returns:
        {"train": Dict[int, np.ndarray],
         "val":   Dict[int, np.ndarray],
         "test":  Dict[int, np.ndarray]}
        Each inner dict: gesture_id → (N, T, C) array.
    """
    rng = np.random.RandomState(seed)

    # ── accumulate train windows per gesture (train subjects only) ────────
    train_dict: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}

    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        for gid, reps in filtered.items():
            for rep_arr in reps:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    train_dict[gid].append(rep_arr)

    # ── concatenate, then split → train / val ─────────────────────────────
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)   # (N, T, C)
        n     = len(X_gid)
        perm  = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        if len(trn_idx) > 0:
            final_train[gid] = X_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid]   = X_gid[val_idx]

    # ── test split — test-subject data only ───────────────────────────────
    # Assembled independently; never influences any train-side decision.
    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": final_test}


# ════════════════════════════════════════════════════════════════════════════
#  Single LOSO fold
# ════════════════════════════════════════════════════════════════════════════

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
    Execute one LOSO fold: train on `train_subjects`, evaluate on `test_subject`.

    All hypernetwork parameters, backbone parameters, and channel normalization
    statistics are derived from the pooled training-subject data only.  The test
    subject is isolated until the final evaluation step, where a frozen model
    with u=0 (canonical deterministic filter) is applied.

    Returns:
        dict with at least "test_accuracy" and "test_f1_macro".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "stochastic_fir_cnn_gru"

    # Persist configs for reproducibility.
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as fh:
        json.dump(asdict(split_cfg), fh, indent=4)

    # ── data loading ──────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config       = proc_cfg,
        logger                  = logger,
        use_gpu                 = True,
        use_improved_processing = USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    # Load train subjects AND test subject in a single call to reuse the cache.
    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir     = base_dir,
        subject_ids  = all_subject_ids,
        exercises    = exercises,
        include_rest = split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES
    )
    logger.info(
        f"Common gestures ({len(common_gestures)}): {common_gestures} "
        f"across {len(all_subject_ids)} subjects"
    )

    # ── build LOSO splits ─────────────────────────────────────────────────
    splits = _build_splits(
        subjects_data   = subjects_data,
        train_subjects  = train_subjects,
        test_subject    = test_subject,
        common_gestures = common_gestures,
        multi_loader    = multi_loader,
        val_ratio       = split_cfg.val_ratio,
        seed            = train_cfg.seed,
    )

    for sname in ("train", "val", "test"):
        n_windows = sum(
            len(arr) for arr in splits[sname].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"  {sname.upper():5s}: {n_windows:5d} windows, "
            f"{len(splits[sname])} gestures"
        )

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = StochasticFIRTrainer(
        train_cfg      = train_cfg,
        logger         = logger,
        output_dir     = output_dir,
        visualizer     = base_viz,
        filter_len     = FILTER_LEN,
        noise_dim      = NOISE_DIM,
        hyper_hidden   = HYPER_HIDDEN,
        cnn_channels   = CNN_CHANNELS,
        gru_hidden     = GRU_HIDDEN,
        num_heads      = NUM_HEADS,
        lambda_smooth  = LAMBDA_SMOOTH,
        lambda_band    = LAMBDA_BAND,
        cutoff_ratio   = CUTOFF_RATIO,
    )

    # ── training ──────────────────────────────────────────────────────────
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed for test_subject={test_subject}: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "stochastic_fir_cnn_gru",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── cross-subject evaluation on held-out test subject ─────────────────
    # Build flat (X, y) arrays from the test split.
    # class_ids ordering is fixed by trainer.fit() and reused here so integer
    # labels align with the model's output neurons.
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
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
            "model_type":    "stochastic_fir_cnn_gru",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy applies training-data standardization and u=0 inference
    # internally — no test statistics used.
    test_results = trainer.evaluate_numpy(
        X_test,
        y_test,
        split_name = f"cross_subject_test_{test_subject}",
        visualize  = True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Acc={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── save fold results ─────────────────────────────────────────────────
    fold_summary = {
        "test_subject":    test_subject,
        "train_subjects":  train_subjects,
        "common_gestures": common_gestures,
        "training":        training_results,
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
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "model_type":     "stochastic_fir_cnn_gru",
            "approach":       APPROACH,
            "exercises":      exercises,
            "frontend_config": {
                "filter_len":    FILTER_LEN,
                "noise_dim":     NOISE_DIM,
                "hyper_hidden":  HYPER_HIDDEN,
                "lambda_smooth": LAMBDA_SMOOTH,
                "lambda_band":   LAMBDA_BAND,
                "cutoff_ratio":  CUTOFF_RATIO,
                "cnn_channels":  list(CNN_CHANNELS),
                "gru_hidden":    GRU_HIDDEN,
                "num_heads":     NUM_HEADS,
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    # ── cleanup ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data, splits
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    "stochastic_fir_cnn_gru",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── subject list ──────────────────────────────────────────────────────
    # Default is CI_TEST_SUBJECTS (server has symlinks only for 5 CI subjects).
    # Use --full to run the complete 20-subject evaluation.
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument(
        "--subjects", type=str, default=None,
        help="Comma-separated subject IDs, e.g. DB2_s1,DB2_s12",
    )
    _parser.add_argument(
        "--ci", action="store_true",
        help="Use 5-subject CI subset (default behaviour)",
    )
    _parser.add_argument(
        "--full", action="store_true",
        help="Use all 20 subjects",
    )
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS      # safe server default

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    # ── configs ───────────────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size         = 600,
        window_overlap      = 300,
        num_channels        = 8,
        sampling_rate       = 2000,
        segment_edge_margin = 0.1,
    )
    split_cfg = SplitConfig(
        train_ratio            = 0.7,
        val_ratio              = 0.15,
        test_ratio             = 0.15,
        mode                   = "by_segments",
        shuffle_segments       = True,
        seed                   = 42,
        include_rest_in_splits = False,
    )
    train_cfg = TrainingConfig(
        model_type               = "stochastic_fir_cnn_gru",
        pipeline_type            = APPROACH,
        use_handcrafted_features = False,
        batch_size               = 64,
        epochs                   = 60,
        learning_rate            = 1e-3,
        weight_decay             = 1e-4,
        dropout                  = 0.3,
        early_stopping_patience  = 12,
        seed                     = 42,
        use_class_weights        = True,
        num_workers              = 4,
        device                   = "cuda" if torch.cuda.is_available() else "cpu",
    )

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis : Stochastic FIR domain randomization — hypernetwork maps\n"
        f"             noise u~N(0,I) to per-sample depthwise FIR coefficients.\n"
        f"             Backbone learns features invariant to filter realization.\n"
        f"             Test-time: u=0 (canonical deterministic filter, no leakage)."
    )
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Exercises  : {EXERCISES}")
    print(
        f"Frontend   : filter_len={FILTER_LEN}, noise_dim={NOISE_DIM}, "
        f"hyper_hidden={HYPER_HIDDEN}"
    )
    print(
        f"Reg        : λ_smooth={LAMBDA_SMOOTH}, λ_band={LAMBDA_BAND}, "
        f"cutoff_ratio={CUTOFF_RATIO}"
    )
    print(f"Backbone   : CNN{CNN_CHANNELS}, GRU_hidden={GRU_HIDDEN}, heads={NUM_HEADS}")
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / "stochastic_fir_cnn_gru" / f"test_{test_subject}"

        result = run_single_loso_fold(
            base_dir       = BASE_DIR,
            output_dir     = fold_dir,
            train_subjects = train_subjects,
            test_subject   = test_subject,
            exercises      = EXERCISES,
            proc_cfg       = proc_cfg,
            split_cfg      = split_cfg,
            train_cfg      = train_cfg,
        )
        all_results.append(result)

    # ── aggregate ─────────────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"] for r in valid]
        print(f"\n{'=' * 60}")
        print(f"StochasticFIR-CNNGRU — LOSO Summary ({len(valid)} folds)")
        print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    # ── save summary JSON ─────────────────────────────────────────────────
    summary: Dict = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": (
            "Domain randomization via noise-conditioned FIR hypernetwork. "
            "Backbone learns subject-invariant features by training on random "
            "filter realizations. Test-time: u=0, fully deterministic."
        ),
        "timestamp": TIMESTAMP,
        "subjects":  ALL_SUBJECTS,
        "exercises": EXERCISES,
        "approach":  APPROACH,
        "frontend_config": {
            "filter_len":    FILTER_LEN,
            "noise_dim":     NOISE_DIM,
            "hyper_hidden":  HYPER_HIDDEN,
            "lambda_smooth": LAMBDA_SMOOTH,
            "lambda_band":   LAMBDA_BAND,
            "cutoff_ratio":  CUTOFF_RATIO,
            "cnn_channels":  list(CNN_CHANNELS),
            "gru_hidden":    GRU_HIDDEN,
            "num_heads":     NUM_HEADS,
        },
        "results": all_results,
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
    with open(summary_path, "w") as fh:
        json.dump(make_json_serializable(summary), fh, indent=4, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")

    # ── report to hypothesis_executor if available ────────────────────────
    try:
        from hypothesis_executor import (          # noqa: F401
            mark_hypothesis_verified,
            mark_hypothesis_failed,
        )
        if valid:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
            }
            mark_hypothesis_verified(
                "H_STOCHASTIC_FIR_HYPERNETWORK", metrics,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                "H_STOCHASTIC_FIR_HYPERNETWORK",
                "All LOSO folds failed",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
