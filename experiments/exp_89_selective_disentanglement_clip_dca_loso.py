"""
Experiment 89: Selective Disentanglement ECAPA-TDNN (CLIP-DCA approach, LOSO)

Hypothesis
──────────
The key failure of full content/style separation (exp_31) is that forcing
complete encoder-level disentanglement destroys useful correlations
(e.g., EMG amplitude correlates with both gesture type and subject identity).

CLIP-DCA alternative: strengthen domain AWARENESS in the encoder, but enforce
domain INVARIANCE only at the classifier level (not at the full representation).

Architecture (see models/selective_disentanglement_ecapa.py)
────────────────────────────────────────────────────────────
1. Domain-aware encoder (ECAPA-TDNN backbone):
   Trained with auxiliary subject classification head — encoder explicitly
   learns to encode subject identity, enabling implicit per-subject normalisation.

2. Domain-conditioned attention (FiLM at each SE-Res2Net block):
   Subject embedding modulates SE attention weights via Feature-wise Linear
   Modulation (scale γ, shift β per channel).  Encoder can learn to
   normalise feature amplitudes relative to the known subject.

3. Projection head:
   Maps domain-aware embedding z → gesture-specific representation h.

4. Domain-invariant gesture classifier:
   Gradient Reversal applied to h (not to z) before a domain confusion head.
   Pushes h to be domain-invariant without destroying z's expressiveness.

Loss (training only):
  L = L_cls + λ_subj * L_subj_aux + λ_dom * L_domain_confusion

  λ_subj = 0.5   (auxiliary subject head on z)
  λ_dom  = 0.3   (domain confusion head on GRL(h))

GRL alpha annealing (standard DANN schedule): 0 → 1 over training epochs.

LOSO Protocol (strictly enforced — NO test-subject adaptation)
──────────────────────────────────────────────────────────────
For each fold (test_subject ∈ ALL_SUBJECTS):
  train_subjects = ALL_SUBJECTS \\ {test_subject}

  1. Load windows for ALL subjects in one call (needed to find common gestures).
     No signal statistics derived here — only gesture ID sets.

  2. Build train/val splits from train_subjects only, with subject provenance.
     Val carved from train (random permutation) — test_subject never touched.

  3. Build test split: test_subject windows only.

  4. Channel mean/std computed from X_train ONLY inside trainer.fit().
     Same stats applied to val and test (no leakage).

  5. Train SelectiveDisentanglementECAPA end-to-end on train split.
     Subject labels used only during training.

  6. After training: compute mean_subject_emb from embedding table parameters.
     This sets the FiLM conditioning for inference — NO test data involved.

  7. Evaluate with model.eval(), inference=True.
     Test subject NEVER seen during training, normalisation, or FiLM setup.

Data-leakage guards:
  ✓ _build_splits_with_subject_labels() builds train/val from train_subjects only.
  ✓ Test split: test_subject windows only (after common_gestures filtering).
  ✓ mean_c / std_c: computed in trainer.fit() from X_train only.
  ✓ mean_subject_emb: computed from model parameters, NOT from test-subject data.
  ✓ model.eval() at inference: BatchNorm frozen to training running stats.
  ✓ No subject-specific adaptation, no running statistics from test subject.

Comparison to baselines
────────────────────────
  exp_62: Plain ECAPA-TDNN (no disentanglement)
  exp_88: Causal ECAPA + full content/style disentanglement
  exp_89: Selective disentanglement (this experiment) — domain-aware z, invariant h

Run examples
────────────
  # 5-subject CI run (default and safe for server):
  python experiments/exp_89_selective_disentanglement_clip_dca_loso.py

  python experiments/exp_89_selective_disentanglement_clip_dca_loso.py --ci

  # Specific subjects:
  python experiments/exp_89_selective_disentanglement_clip_dca_loso.py \\
      --subjects DB2_s1,DB2_s12,DB2_s15

  # Full 20-subject run (only on a machine with all subjects available):
  python experiments/exp_89_selective_disentanglement_clip_dca_loso.py --full
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
from training.selective_disentanglement_trainer import SelectiveDisentanglementTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ══════════════════════════ EXPERIMENT SETTINGS ══════════════════════════════

EXPERIMENT_NAME = "exp_89_selective_disentanglement_clip_dca"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# ── ECAPA-TDNN backbone ──────────────────────────────────────────────────────
ECAPA_CHANNELS      = 128
ECAPA_SCALE         = 4
ECAPA_EMBEDDING_DIM = 128
ECAPA_DILATIONS     = [2, 3, 4]
ECAPA_SE_REDUCTION  = 8

# ── Selective disentanglement ────────────────────────────────────────────────
SUBJECT_EMB_DIM = 32    # Dimension of subject embedding (FiLM conditioning)
PROJ_DIM        = 128   # Projection head output (domain-invariant space)
LAMBDA_SUBJ     = 0.5   # Auxiliary subject classification loss weight
LAMBDA_DOM      = 0.3   # Domain confusion loss weight (after GRL)


# ══════════════════════ SPLIT BUILDER WITH SUBJECT PROVENANCE ════════════════

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
    Build train/val/test splits with per-window subject provenance tracking.

    Unlike CrossSubjectExperiment._prepare_splits(), this function preserves
    which training subject each window came from — needed for the auxiliary
    subject classification head and GRL domain confusion training.

    LOSO invariants:
      ─ train + val windows come ONLY from train_subjects.
      ─ val is carved from train via random permutation (no test leakage).
      ─ test windows come ONLY from test_subject.
      ─ Same random permutation applied to windows and subject labels
        so alignment is preserved.
      ─ No signal statistics (mean, std) are computed here.

    Args:
        subjects_data:    Dict[str, Tuple[emg, segments, grouped_windows]].
        train_subjects:   Subject IDs for training (excludes test_subject).
        test_subject:     Subject ID held out for evaluation.
        common_gestures:  Gesture IDs common to all subjects.
        multi_loader:     MultiSubjectLoader instance.
        val_ratio:        Fraction of train windows held out for validation.
        seed:             RNG seed for reproducibility.

    Returns:
        {
          "train":               Dict[int, np.ndarray],  # (N, T, C)
          "val":                 Dict[int, np.ndarray],  # (N, T, C)
          "test":                Dict[int, np.ndarray],  # (N, T, C)
          "train_subject_labels": Dict[int, np.ndarray], # (N,) int64
          "num_train_subjects":  int,
        }
    """
    rng = np.random.RandomState(seed)
    # Assign a deterministic integer index to each training subject
    train_subject_to_idx: Dict[str, int] = {
        sid: i for i, sid in enumerate(sorted(train_subjects))
    }
    num_train_subjects = len(train_subjects)

    # ── Collect train windows with subject provenance ─────────────────────
    train_dict:       Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}
    train_subj_dict:  Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}

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
                    train_subj_dict[gid].append(
                        np.full(len(rep_arr), train_subject_to_idx[sid], dtype=np.int64)
                    )

    # ── Train → train / val per gesture (aligned permutation) ─────────────
    final_train:      Dict[int, np.ndarray] = {}
    final_val:        Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)      # (N, T, C)
        S_gid = np.concatenate(train_subj_dict[gid], axis=0) # (N,)
        n       = len(X_gid)
        perm    = rng.permutation(n)
        n_val   = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]

        if len(trn_idx) > 0:
            final_train[gid]      = X_gid[trn_idx]
            final_train_subj[gid] = S_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid] = X_gid[val_idx]

    # ── Test split: test_subject windows only ─────────────────────────────
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
        "train_subject_labels": final_train_subj,
        "num_train_subjects":  num_train_subjects,
    }


# ══════════════════════════════ SINGLE FOLD ══════════════════════════════════

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
    None values indicate failed folds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "selective_disentanglement_ecapa"

    # Save configs for full reproducibility
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

    # ── Load ALL subjects (train + test) together ─────────────────────────
    # Test subject is included so get_common_gestures() can find gesture IDs
    # present in ALL subjects.  No signal statistics are derived here.
    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES,
    )
    logger.info(
        f"Common gestures across {len(all_subject_ids)} subjects: "
        f"{common_gestures}  ({len(common_gestures)} total)"
    )

    if len(common_gestures) < 2:
        logger.error("Fewer than 2 common gestures — skipping fold.")
        return {
            "test_subject": test_subject,
            "model_type": "selective_disentanglement_ecapa",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "Fewer than 2 common gestures.",
        }

    # ── Build LOSO-clean splits with subject provenance ───────────────────
    splits = _build_splits_with_subject_labels(
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
            len(arr) for arr in splits[sname].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"  {sname.upper():5s}: {total_w:5d} windows, "
            f"{len(splits[sname])} gestures"
        )
    logger.info(
        f"  Training subjects: {splits['num_train_subjects']} "
        f"({sorted(train_subjects)})"
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SelectiveDisentanglementTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        channels=ECAPA_CHANNELS,
        scale=ECAPA_SCALE,
        embedding_dim=ECAPA_EMBEDDING_DIM,
        subject_emb_dim=SUBJECT_EMB_DIM,
        proj_dim=PROJ_DIM,
        dilations=ECAPA_DILATIONS,
        se_reduction=ECAPA_SE_REDUCTION,
        lambda_subj=LAMBDA_SUBJ,
        lambda_dom=LAMBDA_DOM,
    )

    # ── Training ──────────────────────────────────────────────────────────
    # fit() computes channel normalisation from X_train only;
    # subject labels come from splits["train_subject_labels"] (training only).
    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "selective_disentanglement_ecapa",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── Cross-subject test evaluation ─────────────────────────────────────
    # Assemble flat test arrays in class_ids order (as trainer expects).
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
            "model_type":    "selective_disentanglement_ecapa",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data.",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy():
    #   - applies training-stats normalisation
    #   - calls model.forward(inference=True)
    #     → uses mean_subject_emb (model params, no test data)
    #   - model.eval() → BatchNorm frozen to training stats
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

    # ── Save fold results ──────────────────────────────────────────────────
    fold_summary = {
        "test_subject":       test_subject,
        "train_subjects":     train_subjects,
        "common_gestures":    common_gestures,
        "num_train_subjects": splits["num_train_subjects"],
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
        json.dump(make_json_serializable(fold_summary), fh, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject":  test_subject,
            "train_subjects": train_subjects,
            "model_type":    "selective_disentanglement_ecapa",
            "approach":      APPROACH,
            "exercises":     exercises,
            "model_config": {
                "channels":       ECAPA_CHANNELS,
                "scale":          ECAPA_SCALE,
                "embedding_dim":  ECAPA_EMBEDDING_DIM,
                "subject_emb_dim": SUBJECT_EMB_DIM,
                "proj_dim":       PROJ_DIM,
                "dilations":      ECAPA_DILATIONS,
                "se_reduction":   ECAPA_SE_REDUCTION,
            },
            "loss_config": {
                "lambda_subj": LAMBDA_SUBJ,
                "lambda_dom":  LAMBDA_DOM,
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
        "model_type":    "selective_disentanglement_ecapa",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ════════════════════════════════ MAIN ══════════════════════════════════════

def main():
    """
    LOSO evaluation loop over all subjects.

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
                         help="Use 5 CI test subjects (default)")
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
        model_type="selective_disentanglement_ecapa",
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
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(
        f"Hypothesis : Selective Disentanglement ECAPA-TDNN (CLIP-DCA)\n"
        f"             domain-aware encoder z + domain-invariant projection h\n"
        f"             FiLM-conditioned SE attention + GRL at projection level"
    )
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Exercises  : {EXERCISES}")
    print(
        f"Model cfg  : C={ECAPA_CHANNELS}, scale={ECAPA_SCALE}, "
        f"embed={ECAPA_EMBEDDING_DIM}, subj_emb={SUBJECT_EMB_DIM}, "
        f"proj={PROJ_DIM}, dilations={ECAPA_DILATIONS}"
    )
    print(
        f"Loss cfg   : λ_subj={LAMBDA_SUBJ}, λ_dom={LAMBDA_DOM}, "
        f"GRL alpha annealed 0→1"
    )
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_ROOT / "selective_disentanglement" / f"test_{test_subject}"

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

    # ── Aggregate LOSO summary ─────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"]  for r in valid]
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        mean_f1  = float(np.mean(f1s))
        std_f1   = float(np.std(f1s))

        print("\n" + "=" * 80)
        print("LOSO SUMMARY — Selective Disentanglement ECAPA-TDNN")
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
            "model":        "selective_disentanglement_ecapa",
            "approach":     APPROACH,
            "subjects":     ALL_SUBJECTS,
            "exercises":    EXERCISES,
            "model_config": {
                "channels":       ECAPA_CHANNELS,
                "scale":          ECAPA_SCALE,
                "embedding_dim":  ECAPA_EMBEDDING_DIM,
                "subject_emb_dim": SUBJECT_EMB_DIM,
                "proj_dim":       PROJ_DIM,
                "dilations":      ECAPA_DILATIONS,
                "se_reduction":   ECAPA_SE_REDUCTION,
            },
            "loss_config": {
                "lambda_subj": LAMBDA_SUBJ,
                "lambda_dom":  LAMBDA_DOM,
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

    # ── Optional: report to hypothesis executor ────────────────────────────
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
