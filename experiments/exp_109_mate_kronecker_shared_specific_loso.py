"""
Experiment 109: MATE-inspired Shared-Specific with Kronecker Attention (LOSO)

Hypothesis 2: Shared-Specific Mode Interaction without orthogonality.

Motivation:
  Experiments using orthogonal disentanglement (exp_31: 35.28% F1,
  exp_57: 33.37% GroupDRO, exp_59: 32.94% prototype push) show that
  strengthening orthogonality constraints hurts cross-subject generalisation.
  This aligns with the core MATE insight (NeurIPS 2023): the orthogonality
  assumption is too strong — shared and specific variables may be legitimately
  correlated.

Key design choices:
  1. ECAPA-TDNN backbone (proven best in exp_62/70) → global embedding h.
  2. Shared Prior Network: h → z_shared (gesture-wide, adversarially regularised
     to be subject-invariant via GRL — DANN schedule).
  3. Per-channel CNNs → Specific Prior Networks (×8): f_k → z_specific_k.
     Captures channel-specific variance without orthogonality constraint.
  4. Kronecker Attention: fuses Z_specific ∈ ℝ^{8 × D_p} efficiently.
     A_ch ⊗ A_feat — O(K² + D²) vs O((KD)²) for full attention.
  5. NO Barlow Twins / distance correlation / MI penalty between z_shared
     and z_specific (the change that distinguishes this from exp_31/57/59).
  6. Subject adversary on z_shared only (GRL, annealed alpha from DANN).

Expected outcome: F1 > 35.5% (above exp_31 baseline).

LOSO protocol:
  - Each fold: train on N-1 subjects, test on 1 held-out subject.
  - Subject labels used ONLY during training, NOT at val/test/inference.
  - Channel standardisation computed from training fold only.
  - model.eval() freezes BatchNorm — no test-subject stats adaptation.
  - No subject-specific layers, no fine-tuning, no test-time adaptation.

Cannot use CrossSubjectExperiment.run() directly because _prepare_splits()
merges training subjects, losing subject identity needed for GRL. Instead
we build splits manually with subject provenance (same as exp_31/88).
"""

import sys
import json
import gc
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict

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
from training.mate_kronecker_trainer import MATEKroneckerTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ========== EXPERIMENT SETTINGS ==========

EXPERIMENT_NAME = "exp_109_mate_kronecker_shared_specific"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES    = 10

# ECAPA-TDNN backbone
ECAPA_CHANNELS   = 128
ECAPA_SCALE      = 4
ECAPA_DILATIONS  = [2, 3, 4]
ECAPA_SE_REDUCTION = 8
EMBEDDING_DIM    = 128

# Shared-Specific dimensions
SHARED_DIM   = 128   # z_shared size (maps from ECAPA embedding)
SPECIFIC_DIM = 32    # z_specific_k size per channel (8 × 32 = 256 total)
CH_ENC_DIM   = 64    # ChannelEncoder intermediate dimension

# Kronecker Attention
KRON_D_K = 16        # Q/K projection dimension for inter-channel and intra-feature

# Loss
LAMBDA_ADV = 0.5     # weight on adversarial subject loss (GRL-based)
                     # GRL alpha itself is annealed via DANN schedule, not lambda_adv


# ========== SPLIT BUILDER WITH SUBJECT PROVENANCE ==========================

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

    Unlike CrossSubjectExperiment._prepare_splits(), this function preserves
    which training subject each window came from. The subject index is needed
    for the adversarial subject loss (GRL).

    LOSO safety:
      - test_subject windows go ONLY to the test split — never in train or val.
      - train/val split uses a fixed random permutation (per-gesture) derived
        from the seed — fully deterministic and consistent across runs.
      - subject_label[i] == i_subject ∈ {0, ..., N_train-1} for training only.
        Subject labels are NOT created for val or test.

    Returns dict:
        "train": Dict[gesture_id → np.ndarray (N, T, C)]
        "val":   Dict[gesture_id → np.ndarray (N, T, C)]
        "test":  Dict[gesture_id → np.ndarray (N, T, C)]
        "train_subject_labels": Dict[gesture_id → np.ndarray (N,)]  subject indices
        "num_train_subjects": int
    """
    rng = np.random.RandomState(seed)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects   = len(train_subjects)

    # ── Collect training windows + subject labels per gesture ─────────────
    train_dict      = {}
    train_subj_dict = {}

    for gid in common_gestures:
        windows_list   = []
        subj_label_list = []

        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid not in filtered:
                continue
            for rep_array in filtered[gid]:
                if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                    windows_list.append(rep_array)
                    subj_label_list.append(
                        np.full(len(rep_array), train_subject_to_idx[sid],
                                dtype=np.int64)
                    )

        if windows_list:
            train_dict[gid]      = np.concatenate(windows_list,    axis=0)
            train_subj_dict[gid] = np.concatenate(subj_label_list, axis=0)

    # ── Split collected training data into train / val (per gesture) ──────
    # Windows and subject labels are permuted together to maintain alignment.
    final_train      = {}
    final_val        = {}
    final_train_subj = {}

    for gid in common_gestures:
        if gid not in train_dict:
            continue
        X_gid = train_dict[gid]
        S_gid = train_subj_dict[gid]
        n     = len(X_gid)
        perm  = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        final_train[gid]      = X_gid[train_idx]
        final_val[gid]        = X_gid[val_idx]
        final_train_subj[gid] = S_gid[train_idx]

    # ── Build test split from the held-out subject only ───────────────────
    # No subject labels for test (never needed at inference).
    test_dict = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        valid_reps = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
        if valid_reps:
            test_dict[gid] = np.concatenate(valid_reps, axis=0)

    return {
        "train":               final_train,
        "val":                 final_val,
        "test":                test_dict,
        "train_subject_labels": final_train_subj,
        "num_train_subjects":   num_train_subjects,
    }


# ========== SINGLE LOSO FOLD ==============================================

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
    Execute one LOSO fold: train on all-but-one subjects, evaluate on the one.

    LOSO invariants enforced here:
      - Data for test_subject flows ONLY into the test split.
      - Normalisation stats are computed inside trainer.fit() from training
        windows only — this function does not touch raw data statistics.
      - Subject labels are injected only into the "train" split portion used by
        the adversary. Val and test receive no subject labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger      = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "mate_kronecker"

    # Save configs for reproducibility
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # ── Data loading ──────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

    # Load all subjects for this fold in one call
    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data   = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES,
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

    for split_name in ["train", "val", "test"]:
        total = sum(
            len(arr) for arr in splits[split_name].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"{split_name.upper()}: {total} windows, "
            f"{len(splits[split_name])} gestures"
        )

    # ── Build and train ───────────────────────────────────────────────────
    trainer = MATEKroneckerTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        lambda_adv=LAMBDA_ADV,
        ecapa_channels=ECAPA_CHANNELS,
        ecapa_scale=ECAPA_SCALE,
        embedding_dim=EMBEDDING_DIM,
        shared_dim=SHARED_DIM,
        specific_dim=SPECIFIC_DIM,
        ch_enc_dim=CH_ENC_DIM,
        kron_d_k=KRON_D_K,
        dilations=ECAPA_DILATIONS,
        se_reduction=ECAPA_SE_REDUCTION,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject":   test_subject,
            "model_type":     "mate_kronecker",
            "approach":       APPROACH,
            "test_accuracy":  None,
            "test_f1_macro":  None,
            "error":          str(e),
        }

    # ── Evaluate on held-out test subject ─────────────────────────────────
    # Assemble test arrays using class_ids order from trainer.fit()
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test windows available for held-out subject.")
        return {
            "test_subject":   test_subject,
            "model_type":     "mate_kronecker",
            "approach":       APPROACH,
            "test_accuracy":  None,
            "test_f1_macro":  None,
            "error":          "No test data",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    test_results = trainer.evaluate_numpy(
        X_test_concat, y_test_concat,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── Save per-fold results ─────────────────────────────────────────────
    fold_results = {
        "test_subject":    test_subject,
        "train_subjects":  train_subjects,
        "common_gestures": common_gestures,
        "training":        training_results,
        "cross_subject_test": {
            "subject":     test_subject,
            "accuracy":    test_acc,
            "f1_macro":    test_f1,
            "report":      test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(
            make_json_serializable(fold_results),
            f, indent=4, ensure_ascii=False,
        )

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject":   test_subject,
        "train_subjects": train_subjects,
        "model_type":     "mate_kronecker",
        "approach":       APPROACH,
        "exercises":      exercises,
        "architecture": {
            "ecapa_channels": ECAPA_CHANNELS,
            "ecapa_scale":    ECAPA_SCALE,
            "embedding_dim":  EMBEDDING_DIM,
            "shared_dim":     SHARED_DIM,
            "specific_dim":   SPECIFIC_DIM,
            "ch_enc_dim":     CH_ENC_DIM,
            "kron_d_k":       KRON_D_K,
            "dilations":      ECAPA_DILATIONS,
            "se_reduction":   ECAPA_SE_REDUCTION,
        },
        "loss_config": {
            "lambda_adv": LAMBDA_ADV,
            "grl_schedule": "DANN: alpha=2/(1+exp(-10*p))-1, p=epoch/total_epochs",
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # ── Cleanup ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    "mate_kronecker",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ========== MAIN ==========================================================

def main():
    ALL_SUBJECTS = parse_subjects_args()   # defaults to CI_TEST_SUBJECTS
    BASE_DIR     = ROOT / "data"
    TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT  = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

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
        model_type="mate_kronecker",
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

    fused_dim = SHARED_DIM + 8 * SPECIFIC_DIM

    print(f"{'=' * 80}")
    print(f"Experiment:  {EXPERIMENT_NAME}")
    print(f"Hypothesis:  MATE Shared-Specific + Kronecker Attention (no orthogonality)")
    print(f"  Backbone:  ECAPA-TDNN C={ECAPA_CHANNELS}, scale={ECAPA_SCALE}, "
          f"dilations={ECAPA_DILATIONS}")
    print(f"  Shared:    z_shared ∈ R^{SHARED_DIM}  (subject-adversarial via GRL)")
    print(f"  Specific:  z_specific_k ∈ R^{SPECIFIC_DIM}  (×8 channels, "
          f"Kronecker attention, no orthogonality constraint)")
    print(f"  Fused dim: {fused_dim}")
    print(f"  Loss:      L_gesture + {LAMBDA_ADV}·L_subject_adv  (no Barlow/MI between shared/specific)")
    print(f"  GRL alpha: DANN schedule (0→1 over training)")
    print(f"Subjects:    {ALL_SUBJECTS}")
    print(f"Exercises:   {EXERCISES}")
    print(f"Output:      {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = (
            OUTPUT_ROOT / "mate_kronecker" / f"test_{test_subject}"
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

    # ── Aggregate results ─────────────────────────────────────────────────
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s  = [r["test_f1_macro"] for r in valid_results]
        print(f"\n{'=' * 60}")
        print(f"MATE + Kronecker — LOSO Summary ({len(valid_results)} folds)")
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"  Baseline (exp_31): 35.28% F1 — target: >35.5%")
        print(f"{'=' * 60}\n")

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        "experiment":  EXPERIMENT_NAME,
        "hypothesis":  (
            "MATE Shared-Specific without orthogonality: "
            "shared and specific representations allowed to be correlated; "
            "only z_shared is adversarially regularised against subject ID."
        ),
        "timestamp":  TIMESTAMP,
        "subjects":   ALL_SUBJECTS,
        "approach":   APPROACH,
        "architecture": {
            "ecapa_channels": ECAPA_CHANNELS,
            "ecapa_scale":    ECAPA_SCALE,
            "embedding_dim":  EMBEDDING_DIM,
            "shared_dim":     SHARED_DIM,
            "specific_dim":   SPECIFIC_DIM,
            "ch_enc_dim":     CH_ENC_DIM,
            "kron_d_k":       KRON_D_K,
            "dilations":      ECAPA_DILATIONS,
            "se_reduction":   ECAPA_SE_REDUCTION,
        },
        "loss_config": {
            "lambda_adv":   LAMBDA_ADV,
            "grl_schedule": "DANN: alpha=2/(1+exp(-10*p))-1",
            "no_orthogonality_between_shared_specific": True,
        },
        "results": all_loso_results,
    }

    if valid_results:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_folds":     len(valid_results),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(
            make_json_serializable(summary),
            f, indent=4, ensure_ascii=False,
        )
    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── Report to hypothesis_executor if available ────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        if valid_results:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
            }
            mark_hypothesis_verified(
                "H_mate_kronecker",
                metrics,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                "H_mate_kronecker",
                "All LOSO folds failed",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
