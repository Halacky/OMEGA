"""
Experiment 101: XDomainMix — 4-Component Feature Decomposition with
                Cross-Domain Recombination for LOSO EMG Gesture Recognition

Hypothesis H101:
    4-component decomposition (class-generic, class-specific, domain-generic,
    domain-specific) provides finer-grained disentanglement than the 2-component
    (content, style) approach used in exp_60 (MixStyle).

    Independent control over domain-specific (z_ds) and domain-generic (z_dg)
    components allows:
        - More selective augmentation: only z_ds is swapped between subjects,
          z_dg (universal physiological patterns) remains stable.
        - Cleaner orthogonality constraints: 4 cross-type pairs (cs-ds, cs-dg,
          cg-ds, cg-dg) enforce independence between class and domain subspaces.
        - Richer virtual-domain generation: combining z_ds from subject A with
          z_dg from subject B (shared, so same) and z_cs from subject C creates
          more diverse training conditions than simple 2-component mixing.

Architecture (SharedEncoder → 4 projection heads):
    z_cg (32-dim):  class-generic  — common EMG activation across all gestures
    z_cs (96-dim):  class-specific — gesture-discriminative features
    z_dg (32-dim):  domain-generic — stable physiology shared across subjects
    z_ds (64-dim):  domain-specific — individual subject characteristics

    Gesture classifier:  GestureHead(concat(z_cs, z_cg)) = 128-dim input
    Domain classifier:   DomainHead(concat(z_ds, z_dg))  =  96-dim input

Training:
    Base path:   GestureHead(concat(z_cs, z_cg))                            → L_gesture_base
    Aug  path:   GestureHead(concat(FiLM(z_cs, z_ds_swap), z_cg))          → L_gesture_aug
    Domain path: DomainHead(concat(z_ds_orig, z_dg))                       → L_domain
    Orth. loss:  mean(orth(z_cs,z_ds), orth(z_cs,z_dg),
                      orth(z_cg,z_ds), orth(z_cg,z_dg))                    → L_orth (annealed)

    L_total = L_gesture_base + gamma*L_gesture_aug + alpha_d*L_domain + beta(t)*L_orth

Inference:
    encoder(x) → z_cs, z_cg → GestureHead(concat(z_cs, z_cg))
    No domain information used. No test-time adaptation.

LOSO data-leakage audit:
    ✓ Training batches contain only training-subject windows.
    ✓ Domain-specific swap (z_ds → z_ds[j from different subject]) is performed
      exclusively within each training batch; test subject never appears.
    ✓ Channel standardisation (mean_c, std_c) computed on training windows only,
      applied to val/test with the same fixed statistics.
    ✓ Validation split drawn from training subjects only; test split contains
      exclusively the held-out subject's windows.
    ✓ FiLM conditioning (aug path) is only active in model.training=True mode;
      inference always uses the base path.
    ✓ BatchNorm running stats accumulated during training epochs from training data.
    ✓ No test-time batch-norm update, no feature normalisation from test data.
    ✓ Domain classifier trained on training subjects only (0…K-1 indices);
      test subject has no registered domain index.

Expected effect: +1.5-3pp F1 over MixStyle baseline (exp_60, ~34-36% F1).
Success metric:  mean F1-macro > 36.5% on 5 CI subjects.

Comparison baseline: exp_60 (MixStyle, 2-component, same encoder + FiLM structure).

Usage:
    # CI subset (default, safe on vast.ai server)
    python experiments/exp_101_xdomain_mix_4component_decomposition_loso.py

    # Explicit CI flag
    python experiments/exp_101_xdomain_mix_4component_decomposition_loso.py --ci

    # Custom subjects
    python experiments/exp_101_xdomain_mix_4component_decomposition_loso.py \\
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
from typing import Dict, List, Optional

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
from training.xdomain_mix_trainer import XDomainMixTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ════════════════════════ EXPERIMENT SETTINGS ════════════════════════════

EXPERIMENT_NAME = "exp_101_xdomain_mix_4component_decomposition"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# ── 4-component latent dimensions ─────────────────────────────────────────
CG_DIM = 32   # class-generic:  common EMG activation patterns
CS_DIM = 96   # class-specific: gesture-discriminative features
DG_DIM = 32   # domain-generic: stable physiological patterns (NOT swapped)
DS_DIM = 64   # domain-specific: individual subject characteristics (swapped)

# ── Loss weights ──────────────────────────────────────────────────────────
GAMMA = 0.5          # weight of augmented (FiLM) gesture path loss
ALPHA_D = 0.3        # weight of domain (subject) classification loss
BETA_ORTH = 0.1      # max weight of cross-type orthogonality losses
BETA_ANNEAL_EPOCHS = 10  # epochs to ramp beta from 0 → BETA_ORTH


# ════════════════════════ DATA PREPARATION ════════════════════════════════


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
    Build train / val / test splits with subject-provenance labels.

    Subject labels are integer indices 0…K-1 for K training subjects.
    They are required by XDomainMixTrainer so the model can perform
    cross-domain z_ds swaps within each training batch.

    LOSO boundary guarantee:
        - splits["train"] and splits["train_subject_labels"] contain ONLY
          windows from train_subjects.
        - splits["test"] contains ONLY windows from test_subject.
        - The two sets are strictly disjoint; there is no information flow
          from the test subject into any training computation.

    Returns:
        {
            "train":                Dict[gesture_id, np.ndarray (N, T, C)]
            "val":                  Dict[gesture_id, np.ndarray (N, T, C)]
            "test":                 Dict[gesture_id, np.ndarray (N, T, C)]
            "train_subject_labels": Dict[gesture_id, np.ndarray (N,) int]
            "num_train_subjects":   int
        }
    """
    rng = np.random.RandomState(seed)
    # Map train subject IDs to consecutive integers [0, ..., K-1]
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # ── Step 1: collect per-gesture arrays across training subjects ────────
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

    # ── Step 2: split training data → train / val per gesture ─────────────
    # Permutation applied jointly to keep windows and subject labels aligned.
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

    # ── Step 3: test split from test subject only ──────────────────────────
    # Critical LOSO boundary: test_subject windows go ONLY here.
    # No test_subject window is present in final_train or final_val.
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


# ════════════════════════ SINGLE LOSO FOLD ════════════════════════════════


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
    Execute one LOSO fold with XDomainMix 4-component training.

    Returns a result dict with test_accuracy / test_f1_macro, or
    error info if the fold fails.

    LOSO safety is enforced at three levels:
        1. _build_splits_with_subject_labels(): test_subject data goes only to
           splits["test"]; training/validation splits contain only train_subjects.
        2. XDomainMixTrainer.fit(): DataLoader built from training splits only;
           standardisation stats computed on training data only.
        3. model.eval() inference: base path only, no domain info, no test adaptation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "xdomain_mix_emg"

    # Save configs for reproducibility
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "xdomain_mix_config.json", "w") as f:
        json.dump({
            "cg_dim": CG_DIM,
            "cs_dim": CS_DIM,
            "dg_dim": DG_DIM,
            "ds_dim": DS_DIM,
            "gamma": GAMMA,
            "alpha_d": ALPHA_D,
            "beta_orth": BETA_ORTH,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
        }, f, indent=4)

    # ── Data loading ───────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

    base_viz = Visualizer(output_dir, logger)

    # Load ALL subjects (train + test) in one pass for efficiency.
    # The LOSO boundary is enforced in _build_splits_with_subject_labels where
    # test_subject windows are placed EXCLUSIVELY in splits["test"].
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

    # ── Build splits with subject provenance ───────────────────────────────
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

    # ── Create and train XDomainMix model ────────────────────────────────
    trainer = XDomainMixTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        cg_dim=CG_DIM,
        cs_dim=CS_DIM,
        dg_dim=DG_DIM,
        ds_dim=DS_DIM,
        gamma=GAMMA,
        alpha_d=ALPHA_D,
        beta_orth=BETA_ORTH,
        beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "xdomain_mix_emg",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # ── Evaluate on held-out test subject ─────────────────────────────────
    # Inference path: encoder → z_cs, z_cg → GestureHead(concat(z_cs, z_cg)).
    # No domain information from the test subject is used.
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
            "model_type": "xdomain_mix_emg",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy: transposes (N,T,C)→(N,C,T), applies training standardisation,
    # then calls model(xb) in eval mode (base path tensor).
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
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "model_type": "xdomain_mix_emg",
            "approach": APPROACH,
            "exercises": exercises,
            "xdomain_mix_config": {
                "cg_dim": CG_DIM,
                "cs_dim": CS_DIM,
                "dg_dim": DG_DIM,
                "ds_dim": DS_DIM,
                "gamma": GAMMA,
                "alpha_d": ALPHA_D,
                "beta_orth": BETA_ORTH,
                "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
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
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "xdomain_mix_emg",
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
        model_type="xdomain_mix_emg",
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
    print(f"Hypothesis H101: XDomainMix — 4-Component Decomposition + Cross-Domain Recombination")
    print(f"Subjects:   {ALL_SUBJECTS}")
    print(f"Exercises:  {EXERCISES}")
    print(f"Dims: cg={CG_DIM}, cs={CS_DIM}, dg={DG_DIM}, ds={DS_DIM}")
    print(f"Losses: gamma={GAMMA} (aug), alpha_d={ALPHA_D} (domain), beta_orth={BETA_ORTH} (orth)")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "xdomain_mix_emg" / f"test_{test_subject}"

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

    # ── Aggregate LOSO summary ────────────────────────────────────────────
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]

    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        print(f"\n{'=' * 60}")
        print(f"XDomainMix — LOSO Summary ({len(valid_results)} folds)")
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "H101: XDomainMix — 4-Component Decomposition with Cross-Domain Recombination",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "xdomain_mix_config": {
            "cg_dim": CG_DIM,
            "cs_dim": CS_DIM,
            "dg_dim": DG_DIM,
            "ds_dim": DS_DIM,
            "gamma": GAMMA,
            "alpha_d": ALPHA_D,
            "beta_orth": BETA_ORTH,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
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
            mark_hypothesis_verified("H101", metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H101", "All folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
