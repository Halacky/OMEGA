"""
Experiment 107: Progressive Environment Diversification with Adaptive DRO (LOSO)

Hypothesis H107: Combining phased training, progressive virtual domain creation,
and adaptive GroupDRO to improve robustness in cross-subject EMG gesture recognition.

Problem statement:
    - exp_57 (GroupDRO + disentanglement): fixed eta=0.01 and N groups throughout
      training — DRO is noisy early when representations are still forming.
    - exp_69 (V-REx/Fishr): variance penalty unstable with small N.
    - exp_31 (disentanglement only): no worst-case optimisation.
    - exp_60 (MixStyle): virtual domains but no DRO prioritisation.

Solution — three-phase progressive training:
    Phase 1 (epochs 1-15): Disentanglement only (ERM).
        No DRO, no style mixing. Let the encoder learn to separate content/style.
        Loss = L_gest + alpha*L_subj + beta(t)*L_MI

    Phase 2 (epochs 16-30): MixStyle + soft GroupDRO.
        Create M virtual domains from pairwise style interpolation (MixStyle).
        Enable GroupDRO with soft eta=0.003 on N+M environments.
        Loss = DRO(L_base + gamma*L_mix) + alpha*L_subj + beta*L_MI

    Phase 3 (epochs 31-60): Aggressive DRO + style extrapolation.
        Add extrapolation beyond convex hull: z_extrap = z + a*(z - z_partner).
        Increase DRO step size: eta=0.01.  N+M_mix+M_extrap environments.
        Loss = DRO(L_base + gamma*L_mix + delta*L_extrap) + alpha*L_subj + beta*L_MI

Key mechanisms:
    - Adaptive eta: eta(t) = eta_base * (1 + H(q)/H_max) — adjusts DRO step
      based on entropy of group weights (uniform -> explore, concentrated -> stabilise).
    - Anti-collapse: if max(q) > 0.5, reset q <- uniform. Prevents degeneration.
    - Virtual domains: unordered pairs of training subjects define style-mixing
      and extrapolation environments for DRO.

LOSO data-leakage audit:
    [OK] All virtual domains created from TRAINING subjects only.
    [OK] Style extrapolation uses directed perturbation of training z_style;
         no access to test-subject data.
    [OK] Phase transitions determined by epoch number, not test-subject statistics.
    [OK] Channel standardisation computed on training windows only.
    [OK] Early stopping monitored on held-out subset of training subjects.
    [OK] Test subject evaluated ONCE after training completes.
    [OK] Inference uses z_content only (no FiLM, no style needed).

Differences from related experiments:
    vs exp_57 (GroupDRO):  progressive phases + adaptive eta + anti-collapse +
                           virtual domains (not just real subjects).
    vs exp_34 (curriculum): curriculum over training phases (not subjects);
                            no test-subject statistics used.
    vs exp_31 (disent):     adds DRO + MixStyle + extrapolation.
    vs exp_60 (MixStyle):   adds phased DRO, extrapolation beyond convex hull.

Expected improvement: +2-3pp F1 over exp_57, +0.5-1pp over exp_31.

Usage:
    python experiments/exp_107_progressive_env_diversification_adaptive_dro_loso.py
    python experiments/exp_107_progressive_env_diversification_adaptive_dro_loso.py --ci
    python experiments/exp_107_progressive_env_diversification_adaptive_dro_loso.py \\
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
from training.progressive_env_dro_trainer import ProgressiveEnvDROTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ======================== EXPERIMENT SETTINGS ================================

EXPERIMENT_NAME = "exp_107_progressive_env_diversification_adaptive_dro"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# ── Disentanglement (baseline from exp_31) ──────────────────────────────
CONTENT_DIM = 128
STYLE_DIM = 64
ALPHA = 0.5             # subject classifier loss weight
BETA = 0.1              # MI minimisation loss weight
BETA_ANNEAL_EPOCHS = 10
MI_LOSS_TYPE = "distance_correlation"

# ── MixStyle / extrapolation loss weights ────────────────────────────────
GAMMA = 0.5             # mixed-style gesture path weight
DELTA = 0.3             # extrapolated-style gesture path weight

# ── Phase boundaries ────────────────────────────────────────────────────
PHASE1_END = 15         # end of Phase 1 (disentanglement only)
PHASE2_END = 30         # end of Phase 2 (MixStyle + soft DRO)
                        # Phase 3 runs until training ends (epoch 60)

# ── DRO parameters ──────────────────────────────────────────────────────
ETA_PHASE2 = 0.003      # soft DRO step size
ETA_PHASE3 = 0.01       # aggressive DRO step size
NUM_MIX_PAIRS = 6       # max virtual domains from MixStyle
NUM_EXTRAP_PAIRS = 6    # max virtual domains from extrapolation

# ── Style manipulation ──────────────────────────────────────────────────
MIX_ALPHA = 0.4         # Beta(MIX_ALPHA, MIX_ALPHA) for style interpolation
EXTRAP_ALPHA_MIN = 0.1  # extrapolation magnitude range
EXTRAP_ALPHA_MAX = 0.5

# ── Anti-collapse ───────────────────────────────────────────────────────
COLLAPSE_THRESHOLD = 0.5  # max DRO weight before uniform reset


# ======================== DATA PREPARATION ===================================


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

    LOSO guarantee: test_subject data goes ONLY into splits["test"].
    Training subject provenance is tracked via integer indices for
    GroupDRO group assignment and cross-subject style mixing.

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
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # ── Collect per-gesture arrays across training subjects ──────────────
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
                        np.full(len(rep_array), train_subject_to_idx[sid], dtype=np.int64)
                    )

        if windows_for_gid:
            train_dict[gid] = np.concatenate(windows_for_gid, axis=0)
            train_subj_labels[gid] = np.concatenate(subj_labels_for_gid, axis=0)

    # ── Split train -> train / val per gesture ───────────────────────────
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

    # ── Test split from test subject only (LOSO boundary) ────────────────
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


# ======================== SINGLE LOSO FOLD ===================================


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
    Execute one LOSO fold with Progressive Environment Diversification.

    Returns result dict with test_accuracy / test_f1_macro, or error info.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "progressive_env_dro"

    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "progressive_dro_config.json", "w") as f:
        json.dump({
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "delta": DELTA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
            "phase1_end": PHASE1_END,
            "phase2_end": PHASE2_END,
            "eta_phase2": ETA_PHASE2,
            "eta_phase3": ETA_PHASE3,
            "num_mix_pairs": NUM_MIX_PAIRS,
            "num_extrap_pairs": NUM_EXTRAP_PAIRS,
            "mix_alpha": MIX_ALPHA,
            "extrap_alpha_min": EXTRAP_ALPHA_MIN,
            "extrap_alpha_max": EXTRAP_ALPHA_MAX,
            "collapse_threshold": COLLAPSE_THRESHOLD,
        }, f, indent=4)

    # ── Data loading ─────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

    base_viz = Visualizer(output_dir, logger)

    # Load ALL subjects (train + test) in one pass.
    # LOSO boundary enforced in _build_splits_with_subject_labels.
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
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    # ── Build splits with subject provenance ─────────────────────────────
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

    # ── Create trainer ───────────────────────────────────────────────────
    trainer = ProgressiveEnvDROTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        content_dim=CONTENT_DIM,
        style_dim=STYLE_DIM,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        delta=DELTA,
        beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
        mi_loss_type=MI_LOSS_TYPE,
        phase1_end=PHASE1_END,
        phase2_end=PHASE2_END,
        eta_phase2=ETA_PHASE2,
        eta_phase3=ETA_PHASE3,
        num_mix_pairs=NUM_MIX_PAIRS,
        num_extrap_pairs=NUM_EXTRAP_PAIRS,
        mix_alpha=MIX_ALPHA,
        extrap_alpha_min=EXTRAP_ALPHA_MIN,
        extrap_alpha_max=EXTRAP_ALPHA_MAX,
        collapse_threshold=COLLAPSE_THRESHOLD,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "progressive_env_dro",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # ── Evaluate on held-out test subject ────────────────────────────────
    # Inference uses z_content only (no FiLM, no style, no subject info).
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
            "model_type": "progressive_env_dro",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

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

    # ── Save fold results ────────────────────────────────────────────────
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
        json.dump(
            make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False,
        )

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "model_type": "progressive_env_dro",
            "approach": APPROACH,
            "exercises": exercises,
            "progressive_dro_config": {
                "content_dim": CONTENT_DIM,
                "style_dim": STYLE_DIM,
                "alpha": ALPHA,
                "beta": BETA,
                "gamma": GAMMA,
                "delta": DELTA,
                "phase1_end": PHASE1_END,
                "phase2_end": PHASE2_END,
                "eta_phase2": ETA_PHASE2,
                "eta_phase3": ETA_PHASE3,
                "num_mix_pairs": NUM_MIX_PAIRS,
                "num_extrap_pairs": NUM_EXTRAP_PAIRS,
                "collapse_threshold": COLLAPSE_THRESHOLD,
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
            "final_group_weights": trainer.final_group_weights,
        }),
        filename="fold_metadata.json",
    )

    # ── Memory cleanup ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "progressive_env_dro",
        "approach": APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ======================== MAIN ===============================================


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
        model_type="progressive_env_dro",
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
    print(f"Hypothesis H107: Progressive Environment Diversification + Adaptive DRO")
    print(f"Subjects:  {ALL_SUBJECTS}")
    print(f"Exercises: {EXERCISES}")
    print(f"Disentanglement: content_dim={CONTENT_DIM}, style_dim={STYLE_DIM}")
    print(f"Loss weights: alpha={ALPHA} (subj), beta={BETA} (MI), "
          f"gamma={GAMMA} (mix), delta={DELTA} (extrap)")
    print(f"Phases: 1->{PHASE1_END}, 2->{PHASE2_END}, 3->{train_cfg.epochs}")
    print(f"DRO: eta2={ETA_PHASE2}, eta3={ETA_PHASE3}, "
          f"collapse_thresh={COLLAPSE_THRESHOLD}")
    print(f"Virtual domains: {NUM_MIX_PAIRS} mix + {NUM_EXTRAP_PAIRS} extrap")
    print(f"Style: mix_alpha={MIX_ALPHA}, extrap=[{EXTRAP_ALPHA_MIN}, {EXTRAP_ALPHA_MAX}]")
    print(f"MI type: {MI_LOSS_TYPE}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "progressive_env_dro" / f"test_{test_subject}"

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

    # ── Aggregate LOSO summary ───────────────────────────────────────────
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]

    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        print(f"\n{'=' * 60}")
        print(
            f"Progressive Env DRO — LOSO Summary ({len(valid_results)} folds)"
        )
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        if len(f1s) > 1:
            print(f"  Worst-subject F1: {min(f1s):.4f}")
            print(f"  Best-subject F1:  {max(f1s):.4f}")
        print(f"{'=' * 60}\n")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": (
            "H107: Progressive Environment Diversification with Adaptive DRO — "
            "phased training (disentanglement -> MixStyle+DRO -> extrap+aggressive DRO), "
            "adaptive eta, anti-collapse"
        ),
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "progressive_dro_config": {
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "delta": DELTA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
            "phase1_end": PHASE1_END,
            "phase2_end": PHASE2_END,
            "eta_phase2": ETA_PHASE2,
            "eta_phase3": ETA_PHASE3,
            "num_mix_pairs": NUM_MIX_PAIRS,
            "num_extrap_pairs": NUM_EXTRAP_PAIRS,
            "mix_alpha": MIX_ALPHA,
            "extrap_alpha_min": EXTRAP_ALPHA_MIN,
            "extrap_alpha_max": EXTRAP_ALPHA_MAX,
            "collapse_threshold": COLLAPSE_THRESHOLD,
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
        json.dump(
            make_json_serializable(summary), f, indent=4, ensure_ascii=False,
        )

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
            mark_hypothesis_verified(
                "H107", metrics, experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed("H107", "All folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
