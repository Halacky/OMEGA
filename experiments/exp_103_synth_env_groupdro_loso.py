"""
Experiment 103: Synthetic Environment Expansion + Soft GroupDRO

Hypothesis H103
---------------
Fundamental problem with exp_57 (GroupDRO, N=4 real environments):
with only N=4 groups, the exponentiated gradient ascent with η=0.01 quickly
produces a degenerate weight distribution — 1-2 groups accumulate ~80% of the
weight, reducing GroupDRO to a near-worst-case optimisation over a tiny subset.

Fix: expand the number of environments via latent-space style interpolation
(inspired by exp_60 MixStyle), then apply DRO over the expanded environment set.

Mechanism
---------
For N training subjects:
  Real environments   : N      (base path, no FiLM, one per subject)
  Virtual environments: C(N,2) (FiLM path, one per subject pair {i,j})
  Total               : N + C(N, 2)

Example (LOSO fold with N=4 training subjects):
    4 real  +  C(4,2)=6 virtual  =  10 environments
    GroupDRO runs with η=0.005 (reduced because more envs → noisier updates)

For each batch containing samples from subjects i and j:
  1. Compute z_style for all samples via the style encoder.
  2. For samples from i: mix with random partner from j
         z_style_mix = λ · z_style_i + (1-λ) · z_style_j[rand]
         λ ~ Beta(0.4, 0.4)   (independent per sample)
  3. FiLM-condition z_content: z_content_film = FiLM(z_content, z_style_mix)
  4. Virtual-env loss: CE(GestureClassifier(z_content_film), y)
     computed over samples from BOTH i and j.

DRO operates jointly over all N_total environments:
    L_dro = Σ_e q_e · L_e
    q_e ← q_e · exp(η · L_e)   then normalise to simplex.

Differences from exp_57 (GroupDRO)
------------------------------------
  exp_57 : DRO over 4 real environments, η=0.01, no FiLM
  exp_103: DRO over 4+6=10 environments, η=0.005, FiLM on virtual envs

Differences from exp_60 (MixStyle + FiLM)
------------------------------------------
  exp_60 : ERM + dual CE loss (average-risk), FiLM on every sample
  exp_103: worst-case (DRO) across virtual environments

LOSO compliance
---------------
- Virtual environments are constructed solely from training-batch z_style vectors.
  Test-subject data never participates in style mixing, DRO weight updates, or
  FiLM conditioning.
- model.eval() returns GestureClassifier(z_content) with no FiLM: no style
  information is required or used at inference time.
- Channel normalisation statistics are computed from X_train only (no test data).
- Early stopping uses val_loss from the train-subject validation split.
- Test-subject windows are evaluated exactly once, after training completes.

Expected outcome
----------------
  +1–2 pp F1 over exp_57 (GroupDRO)
    — more stable DRO dynamics with 10 environments vs 4
  Lower std(F1) across subjects
    — virtual envs force robustness to unseen style mixtures
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

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    make_json_serializable,
    parse_subjects_args,
)
from training.synth_env_groupdro_trainer import SynthEnvGroupDROTrainer
from utils.artifacts import ArtifactSaver
from utils.logging import seed_everything, setup_logging
from visualization.base import Visualizer

# ══════════════════════════════════════════════════════════════════════════════
# Experiment settings
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_103_synth_env_groupdro"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# Disentanglement hyperparameters (same as exp_31/exp_57 for fair comparison)
CONTENT_DIM = 128
STYLE_DIM = 64
ALPHA = 0.5             # subject classifier loss weight
BETA = 0.1              # MI minimisation weight (annealed)
BETA_ANNEAL_EPOCHS = 10
MI_LOSS_TYPE = "distance_correlation"

# Synthetic environment expansion + GroupDRO specific
DRO_ETA = 0.005         # reduced vs exp_57 (0.01) — more envs → smaller step
MIX_ALPHA = 0.4         # Beta(0.4, 0.4) for style mixing — same as exp_60


# ══════════════════════════════════════════════════════════════════════════════
# Split construction (identical to exp_57 — preserves subject provenance)
# ══════════════════════════════════════════════════════════════════════════════

def _build_splits_with_subject_labels(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: "MultiSubjectLoader",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Build train / val / test splits with per-window subject provenance labels.

    Train and val splits contain ONLY windows from ``train_subjects``.
    Test split contains ONLY windows from ``test_subject``.

    The ``train_subject_labels`` dict maps gesture_id → int64 array of
    per-window subject indices (0 … N_train-1), aligned to the train split.

    LOSO guarantee: test_subject data is stored only in splits["test"].
    It never appears in "train", "val", "train_subject_labels".

    Returns
    -------
    dict with keys:
        "train"                : Dict[gesture_id, np.ndarray]
        "val"                  : Dict[gesture_id, np.ndarray]
        "test"                 : Dict[gesture_id, np.ndarray]
        "train_subject_labels" : Dict[gesture_id, np.ndarray of int64]
        "num_train_subjects"   : int
    """
    rng = np.random.RandomState(seed)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # ── Collect train windows with subject provenance ─────────────────────
    train_dict: Dict[int, List[np.ndarray]] = {}
    train_subj_dict: Dict[int, List[np.ndarray]] = {}

    for gid in common_gestures:
        train_dict[gid] = []
        train_subj_dict[gid] = []
        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid not in filtered:
                continue
            for rep_array in filtered[gid]:
                if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                    train_dict[gid].append(rep_array)
                    train_subj_dict[gid].append(
                        np.full(len(rep_array), train_subject_to_idx[sid], dtype=np.int64)
                    )

    # ── Train / val split (gesture-stratified, consistent permutation) ─────
    final_train: Dict[int, np.ndarray] = {}
    final_val: Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)
        S_gid = np.concatenate(train_subj_dict[gid], axis=0)
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        final_train[gid] = X_gid[train_idx]
        final_val[gid] = X_gid[val_idx]
        final_train_subj[gid] = S_gid[train_idx]

    # ── Test split (test_subject only — never mixed with train) ───────────
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


# ══════════════════════════════════════════════════════════════════════════════
# Single LOSO fold
# ══════════════════════════════════════════════════════════════════════════════

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
    Run one LOSO fold: train on ``train_subjects``, evaluate on ``test_subject``.

    Returns
    -------
    dict with:
        test_subject, model_type, approach,
        test_accuracy, test_f1_macro,
        final_group_weights, env_descriptions,
        train_subjects_sorted,
        error (only on failure)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "synth_env_groupdro_emg"

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

    # ── Load all subjects (test subject needed only for test split) ────────
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

    # ── Build splits (test_subject only appears in splits["test"]) ─────────
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
            f"{split_name.upper()}: {total} windows, "
            f"{len(splits[split_name])} gestures"
        )

    # ── Create trainer ─────────────────────────────────────────────────────
    trainer = SynthEnvGroupDROTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        content_dim=CONTENT_DIM,
        style_dim=STYLE_DIM,
        alpha=ALPHA,
        beta=BETA,
        beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
        mi_loss_type=MI_LOSS_TYPE,
        dro_eta=DRO_ETA,
        mix_alpha=MIX_ALPHA,
    )

    try:
        trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "synth_env_groupdro_emg",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "final_group_weights": None,
            "env_descriptions": None,
            "train_subjects_sorted": sorted(train_subjects),
            "error": str(e),
        }

    # ── LOSO final evaluation (test subject, one-shot) ─────────────────────
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test data available for final evaluation.")
        return {
            "test_subject": test_subject,
            "model_type": "synth_env_groupdro_emg",
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "final_group_weights": trainer.final_group_weights,
            "env_descriptions": trainer.env_descriptions,
            "train_subjects_sorted": sorted(train_subjects),
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy() is inherited from DisentangledTrainer:
    # transposes, standardises, passes through model.eval() (base path only).
    test_results = trainer.evaluate_numpy(
        X_test,
        y_test,
        split_name=f"loso_test_{test_subject}",
        visualize=True,
    )
    test_acc = float(test_results["accuracy"])
    test_f1 = float(test_results["f1_macro"])

    # Log real-env weights for interpretability; virtual weights logged separately.
    gw = trainer.final_group_weights or []
    n_real = len(train_subjects)
    gw_real = gw[:n_real]
    gw_virt_sum = sum(gw[n_real:]) if len(gw) > n_real else 0.0
    print(
        f"[LOSO] Test {test_subject} | "
        f"Acc={test_acc:.4f}  F1={test_f1:.4f} | "
        f"real_gw=[{', '.join(f'{w:.3f}' for w in gw_real)}] "
        f"virt_gw_sum={gw_virt_sum:.3f}"
    )

    # ── Save fold results ──────────────────────────────────────────────────
    num_virt = len(trainer.env_descriptions or []) - n_real
    fold_result = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "common_gestures": common_gestures,
        "config": {
            "dro_eta": DRO_ETA,
            "mix_alpha": MIX_ALPHA,
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
        },
        "num_real_envs": n_real,
        "num_virtual_envs": num_virt,
        "num_total_envs": n_real + num_virt,
        "env_descriptions": trainer.env_descriptions,
        "final_group_weights": trainer.final_group_weights,
        "train_subjects_sorted": sorted(train_subjects),
        "cross_subject_test": {
            "subject": test_subject,
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "report": test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_result), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "model_type": "synth_env_groupdro_emg",
            "approach": APPROACH,
            "exercises": exercises,
            "config": fold_result["config"],
            "metrics": {"test_accuracy": test_acc, "test_f1_macro": test_f1},
        }),
        filename="fold_metadata.json",
    )

    # ── Memory cleanup ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "synth_env_groupdro_emg",
        "approach": APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "final_group_weights": fold_result["final_group_weights"],
        "env_descriptions": fold_result["env_descriptions"],
        "train_subjects_sorted": sorted(train_subjects),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

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
        model_type="synth_env_groupdro_emg",
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

    n_total_envs_example = len(ALL_SUBJECTS) - 1
    from itertools import combinations as _comb
    n_virt_example = len(list(_comb(range(n_total_envs_example), 2)))
    print(f"{'=' * 80}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Hypothesis H103: Synthetic Environment Expansion + Soft GroupDRO")
    print(f"Subjects: {ALL_SUBJECTS}  ({len(ALL_SUBJECTS)} total)")
    print(f"Per-fold environments: {n_total_envs_example} real + "
          f"{n_virt_example} virtual = {n_total_envs_example + n_virt_example} total")
    print(f"Exercises: {EXERCISES}")
    print(f"Disentanglement: content_dim={CONTENT_DIM}, style_dim={STYLE_DIM}")
    print(f"Loss weights: alpha={ALPHA} (subject), beta={BETA} (MI)")
    print(f"GroupDRO: eta={DRO_ETA}  MixStyle: alpha={MIX_ALPHA}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "synth_env_groupdro_emg" / f"test_{test_subject}"

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

    # ── Aggregate metrics ──────────────────────────────────────────────────
    valid = [r for r in all_loso_results if r.get("test_f1_macro") is not None]

    print(f"\n{'=' * 60}")
    print(f"Synth Env GroupDRO — LOSO Summary ({len(valid)} folds)")

    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s = [r["test_f1_macro"] for r in valid]
        worst_f1 = float(np.min(f1s))
        best_f1 = float(np.max(f1s))
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))

        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  F1-macro: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Worst-subject F1: {worst_f1:.4f}  (DRO primary target)")
        print(f"  Best-subject  F1: {best_f1:.4f}")

        print(f"\n  Per-subject results:")
        for r in sorted(valid, key=lambda x: x["test_f1_macro"]):
            gw = r.get("final_group_weights")
            n_real = len(r.get("train_subjects_sorted", []))
            if gw and n_real:
                gw_real = [f"{w:.3f}" for w in gw[:n_real]]
                gw_virt_sum = sum(gw[n_real:]) if len(gw) > n_real else 0.0
                extra = f"  real_gw={gw_real} virt_sum={gw_virt_sum:.3f}"
            else:
                extra = ""
            acc_val = r["test_accuracy"]
            f1_val = r["test_f1_macro"]
            print(f"    {r['test_subject']}: acc={acc_val:.4f}, f1={f1_val:.4f}{extra}")

    print(f"{'=' * 60}\n")

    # ── Save summary ───────────────────────────────────────────────────────
    summary: Dict = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "H103: Synthetic Environment Expansion + Soft GroupDRO",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "config": {
            "dro_eta": DRO_ETA,
            "mix_alpha": MIX_ALPHA,
            "content_dim": CONTENT_DIM,
            "style_dim": STYLE_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
        },
        "results": all_loso_results,
    }

    if valid:
        summary["aggregate"] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_f1_macro": mean_f1,
            "std_f1_macro": std_f1,
            "worst_f1_macro": worst_f1,
            "best_f1_macro": best_f1,
            "num_folds": len(valid),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── Report to hypothesis_executor (optional) ───────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_failed, mark_hypothesis_verified
        if valid:
            mark_hypothesis_verified(
                "H103",
                {
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "mean_f1_macro": mean_f1,
                    "std_f1_macro": std_f1,
                    "worst_f1_macro": worst_f1,
                },
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed("H103", "All LOSO folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
