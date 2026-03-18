"""
Experiment 75: EMG Patch Token Transformer + Performer Attention (LOSO)

Hypothesis
──────────
CNN stems impose fixed inductive biases (locality, spatial equivariance) that may
over-fit to subject-specific muscle activation topography, harming cross-subject
generalisation.  Treating each EMG window as a sequence of *patch tokens* (ViT-style
"tokenisation") and modelling their relationships with Performer linear attention —
combined with Attentive Statistics Pooling — gives the model more freedom to discover
gesture-discriminative patterns that are invariant across subjects.

Motivation from other domains
──────────────────────────────
- ViT (Dosovitskiy et al., ICLR 2021): images → 16×16 patches + Transformer.
- Performer (Choromanski et al., ICLR 2021): FAVOR+ for O(L·m) attention.
- HAR-Transformers: several works show Transformers outperform CNNs for
  inertial/HAR data especially when subjects vary widely.
- ECAPA-TDNN (Desplanques et al., IS 2020): Attentive Statistics Pooling.

Architecture
────────────
  1. Patchify : (T=600, C=8) split into P=25-sample patches → 24 tokens.
                Each token: linear projection of (8×25)=200 features → d_model=128.
  2. Performer: 3 layers, 4 heads, 64 random features (FAVOR+ ORF).
                O(L·m) = O(24·64) per head instead of O(L²) = O(576).
  3. Pooling  : Token Attentive Statistics Pooling (weighted mean + std over tokens).
  4. Head     : FC(256→128) + GELU + Dropout → Linear(128→num_classes).

  ≈ 500 K parameters  — comparable to ECAPATDNNEmg (≈ 467 K).

LOSO Protocol (strictly enforced — no adaptation)
──────────────────────────────────────────────────
  For each fold (test_subject ∈ ALL_SUBJECTS):
    train_subjects = ALL_SUBJECTS \\ {test_subject}

    1. Load windows for ALL subjects (train + test) in one call.
       [Loading test data is required for get_common_gestures(); no signal
        statistics are derived from test data at this stage.]
    2. Build train split: pool all train-subject windows per gesture.
       Carve val_ratio fraction from train data only (no test data).
    3. Build test split: test_subject windows only.
    4. Compute per-channel mean/std from TRAIN windows ONLY inside fit().
       Apply these same statistics to val and test (no test-stat leakage).
    5. Train EMGPatchTransformer on train split; early-stop on val_loss.
    6. model.eval() on test split — no parameter updates, no adaptation.
       Test subject NEVER seen during training or normalisation.

Data-leakage guards:
  ✓ _build_splits() accumulates windows ONLY from train_subjects for train+val.
  ✓ val is carved from train windows via random permutation (no test involvement).
  ✓ test split derived from test_subject windows only.
  ✓ mean_c / std_c computed inside EMGPatchTransformerTrainer.fit() from X_train.
  ✓ Performer ω (random features) is sampled at model init — not data-dependent.
  ✓ LayerNorm is per-sample (no running statistics → safe at eval/test time).
  ✓ model.eval() at inference: no LayerNorm updates (already per-sample).
  ✓ No gesture-value statistics come from test subject: only gesture ID sets are used
    for get_common_gestures() (set intersection, not signal aggregation).

Comparison baseline
────────────────────
  Exp 62 — ECAPATDNNEmg (Res2Net + SE + Attentive Stats Pooling):  ≈ 467 K params
  Exp 75 — EMGPatchTransformer (Patch tokens + Performer + ASP):   ≈ 500 K params

  Same training config, same data pipeline, same LOSO protocol.
  The only difference is the architecture (tokenisation vs. convolution stem).

Run examples
────────────
  # 5-subject CI run (fast, safe default for server):
  python experiments/exp_75_emg_patch_tokens_performer_loso.py --ci

  # Specific subjects:
  python experiments/exp_75_emg_patch_tokens_performer_loso.py \\
      --subjects DB2_s1,DB2_s12,DB2_s15

  # Full 20-subject run:
  python experiments/exp_75_emg_patch_tokens_performer_loso.py --full
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
from training.emg_patch_transformer_trainer import EMGPatchTransformerTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ═══════════════════════════ EXPERIMENT SETTINGS ════════════════════════════

EXPERIMENT_NAME = "exp_75_emg_patch_tokens_performer"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# EMGPatchTransformer hyper-parameters
PATCH_SIZE   = 25    # time samples per patch  → 24 patches for T=600
D_MODEL      = 128   # transformer hidden dim
NUM_HEADS    = 4     # attention heads (d_head = 32)
NUM_LAYERS   = 3     # Performer encoder layers
NUM_FEATURES = 64    # random feature count m (FAVOR+ ORF)
FFN_MULT     = 2     # FFN hidden = D_MODEL × FFN_MULT = 256
EMBED_DIM    = 128   # pre-classifier embedding dimension
MAX_PATCHES  = 64    # max sequence length (safe for T≤1600, P=25)


# ══════════════════ LOCAL HELPER: grouped_to_arrays ══════════════════════════

def grouped_to_arrays(grouped_windows: Dict[int, List[np.ndarray]]):
    """
    Convert grouped_windows (Dict[int, List[np.ndarray]]) → flat arrays.

    NOTE: grouped_to_arrays does NOT exist in any processing/ module.
    Must be defined locally in each experiment file (see MEMORY.md rule 19).

    Returns:
        windows: (N, T, C) float32
        labels:  (N,)      int64   — values = gesture IDs (NOT class indices)
    """
    all_windows, all_labels = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                all_windows.append(rep_arr)
                all_labels.append(
                    np.full(len(rep_arr), gid, dtype=np.int64)
                )
    if not all_windows:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(all_windows, axis=0).astype(np.float32),
        np.concatenate(all_labels,  axis=0),
    )


# ══════════════════════════════ SPLITS BUILDER ══════════════════════════════

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
    Construct train / val / test splits from loaded subject data.

    LOSO invariants upheld here:
      ─ train + val windows come ONLY from train_subjects.
      ─ val is carved from train data via random permutation (no leakage).
      ─ test windows come ONLY from test_subject.
      ─ No signal statistics (mean, std) are computed in this function;
        normalisation happens later inside EMGPatchTransformerTrainer.fit().

    Args:
        subjects_data:    Dict returned by MultiSubjectLoader.load_multiple_subjects.
                          Keys: subject_id. Values: tuples (emg, segments, grouped_windows).
        train_subjects:   Subject IDs used for training (≠ test_subject).
        test_subject:     Subject ID held out for evaluation.
        common_gestures:  Gesture IDs common across all subjects.
        multi_loader:     MultiSubjectLoader instance (for filter_by_gestures).
        val_ratio:        Fraction of train windows held out for validation.
        seed:             RNG seed for reproducible val split.

    Returns:
        {"train": Dict[int, np.ndarray],  # gesture_id → (N, T, C)
         "val":   Dict[int, np.ndarray],
         "test":  Dict[int, np.ndarray]}
    """
    rng = np.random.RandomState(seed)

    # ── Accumulate train windows per gesture (train subjects only) ───────
    train_dict: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}

    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        # subjects_data values are tuples (emg, segments, grouped_windows)
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        for gid, reps in filtered.items():
            for rep_arr in reps:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    train_dict[gid].append(rep_arr)

    # ── Split train windows into train / val (no test data touches this) ─
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)  # (N, T, C)
        n      = len(X_gid)
        perm   = rng.permutation(n)
        n_val  = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        if len(trn_idx) > 0:
            final_train[gid] = X_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid]   = X_gid[val_idx]

    # ── Test split: test subject data only ───────────────────────────────
    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": final_test}


# ═══════════════════════════════ SINGLE FOLD ════════════════════════════════

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
    Execute one LOSO fold: train on `train_subjects`, evaluate on `test_subject`.

    Returns dict with at least "test_accuracy" and "test_f1_macro".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "emg_patch_transformer"

    # Save configs for reproducibility
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
    # We load the test subject together with train subjects so that
    # get_common_gestures() can compute gesture ID intersection across all folds.
    # LOSO integrity: no signal-level statistics (mean/std) are derived here —
    # only gesture ID sets (set intersection is not signal-level leakage).
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

    # ── Build LOSO-clean splits ────────────────────────────────────────────
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

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = EMGPatchTransformerTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_features=NUM_FEATURES,
        ffn_mult=FFN_MULT,
        embed_dim=EMBED_DIM,
        max_patches=MAX_PATCHES,
    )

    # ── Training ──────────────────────────────────────────────────────────
    # fit() receives the splits dict; it computes normalisation from X_train only
    # (see EMGPatchTransformerTrainer.fit() for LOSO integrity details).
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "emg_patch_transformer",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── Cross-subject test evaluation ─────────────────────────────────────
    # Assemble flat test arrays using class_ids ordering from trainer so that
    # class indices match what the model was trained with.
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
            "model_type":    "emg_patch_transformer",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy() applies transpose + training-stats normalisation.
    # model.eval() — LayerNorm is per-sample (no running stats); Performer ω
    # is a fixed buffer.  No test-subject adaptation occurs.
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
            "model_type":    "emg_patch_transformer",
            "approach":      APPROACH,
            "exercises":     exercises,
            "model_config": {
                "patch_size":   PATCH_SIZE,
                "d_model":      D_MODEL,
                "num_heads":    NUM_HEADS,
                "num_layers":   NUM_LAYERS,
                "num_features": NUM_FEATURES,
                "ffn_mult":     FFN_MULT,
                "embed_dim":    EMBED_DIM,
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
        "model_type":    "emg_patch_transformer",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════════════════ MAIN ══════════════════════════════════

def main():
    """
    LOSO evaluation loop.

    Subject list priority:
      1. --subjects DB2_s1,DB2_s12,...  — explicit list
      2. --full                         — all 20 DEFAULT_SUBJECTS
      3. --ci  (or default)             — 5 CI_TEST_SUBJECTS  ← safe server default

    The default (no flags) intentionally uses CI_TEST_SUBJECTS because the
    vast.ai server has symlinks only for those 5 subjects.
    """
    import argparse

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject IDs")
    _parser.add_argument("--ci",   action="store_true",
                         help="Use 5 CI test subjects")
    _parser.add_argument("--full", action="store_true",
                         help="Use all 20 DEFAULT_SUBJECTS")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS      # safe default for server

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = (
        ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    )

    # ── Configs ───────────────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,    # drop 10 % at each segment edge
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
        model_type="emg_patch_transformer",
        pipeline_type=APPROACH,
        use_handcrafted_features=False,
        batch_size=64,
        epochs=60,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.1,               # Transformers typically use smaller dropout
        early_stopping_patience=12,
        seed=42,
        use_class_weights=True,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("=" * 80)
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis  : EMG Patch Tokens + Performer linear attention\n"
        f"              vs ECAPATDNNEmg baseline (comparable param budget)"
    )
    print(f"Subjects    : {ALL_SUBJECTS}")
    print(f"Exercises   : {EXERCISES}")
    print(
        f"Model config: patch_size={PATCH_SIZE}, d_model={D_MODEL}, "
        f"heads={NUM_HEADS}, layers={NUM_LAYERS}, "
        f"num_features={NUM_FEATURES}, ffn_mult={FFN_MULT}"
    )
    print(f"Output      : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / "emg_patch_transformer" / f"test_{test_subject}"

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
        print("LOSO SUMMARY — EMGPatchTransformer (Performer)")
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
            "experiment":   EXPERIMENT_NAME,
            "model":        "emg_patch_transformer",
            "approach":     APPROACH,
            "subjects":     ALL_SUBJECTS,
            "exercises":    EXERCISES,
            "model_config": {
                "patch_size":   PATCH_SIZE,
                "d_model":      D_MODEL,
                "num_heads":    NUM_HEADS,
                "num_layers":   NUM_LAYERS,
                "num_features": NUM_FEATURES,
                "ffn_mult":     FFN_MULT,
                "embed_dim":    EMBED_DIM,
                "max_patches":  MAX_PATCHES,
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
            json.dump(
                make_json_serializable(summary), fh, indent=4, ensure_ascii=False
            )
        print(f"Summary saved → {OUTPUT_ROOT / 'loso_summary.json'}")
    else:
        print("No successful folds to summarise.")

    # Optional: report to hypothesis executor if available
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
