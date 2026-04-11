"""
Experiment 108: Hierarchical Conditional β-VAE with Learnable Frequency Decomposition

Hypothesis H1_new
─────────────────
Domain shift in cross-subject sEMG has three orthogonal components:
    (a) Spectral: electrode placement alters frequency content of activation.
    (b) Amplitude: impedance and force differences change signal magnitude.
    (c) Morphological: individual motor-unit firing patterns change waveform shape.

This experiment addresses all three simultaneously via hierarchical disentanglement:

    Level 1 (UVMD, exp_93):
        Learnable VMD separates each channel into K=4 frequency modes.
        α, τ, ω are trained end-to-end — no hand-crafted filterbank.

    Level 3 (Soft AGC, exp_76):
        Per-mode amplitude normalisation via EMA-based soft AGC.
        Applied BEFORE per-channel encoding to remove subject-specific gain.
        Softer than PCEN (exp_61): α ∈ (0, 0.5), δ fixed — avoids destroying
        amplitude-based gesture cues.

    Level 2 (β-VAE, inspired by LTG):
        For each of the K×C = 32 (mode, channel) pairs, a shared β-VAE encoder
        produces z_content (gesture-informative) and z_style (subject-specific).
        β annealing prevents premature KL pressure collapsing content codes.
        Distance-correlation MI proxy enforces z_content ⊥ z_style.

Key differences from prior experiments:
    exp_31  (global disentanglement, ERM):       1 content/style axis
    exp_57  (global disentanglement, GroupDRO):  1 axis + DRO
    exp_106 (per-channel, contrastive, GroupDRO): C=8 axes, no freq. split
    exp_93  (UVMD + CNN, no disentanglement):    frequency only
    exp_76  (Soft AGC + CNN-GRU, no VAE):        amplitude only
    exp_108 (THIS):  K×C=32 axes + frequency decomposition + amplitude norm
                     + β-VAE (not contrastive) + no GroupDRO (simpler loss)

LOSO compliance (strictly enforced)
────────────────────────────────────
    ✓  Data loading: test subject is loaded only to populate splits["test"].
       All model parameter updates use ONLY train-subject windows.
    ✓  Channel standardisation computed from X_train (train subjects only).
       mean_c / std_c applied to X_val and X_test without refit.
    ✓  UVMD, SoftAGC, CNN, VAE heads, ASP, classifier — all parameters
       trained on train-subject batches only.
    ✓  Early stopping monitors val_loss (train-subject validation split).
    ✓  Test evaluation: model.eval() + torch.no_grad() + z = μ (no sampling).
    ✓  No test-subject label or window is ever in a gradient computation.
    ✓  No test-time adaptation of any kind.
    ✓  Subject list defaults to CI_TEST_SUBJECTS (5 subjects) to avoid
       FileNotFoundError on server where only CI symlinks exist.

Expected result: F1-macro > 36% (additive invariance from all 3 levels).
Risk: optimisation complexity (K×C VAE heads + annealing schedules).
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
from training.hierarchical_beta_vae_trainer import HierarchicalBetaVAETrainer
from utils.artifacts import ArtifactSaver
from utils.logging import seed_everything, setup_logging
from visualization.base import Visualizer

# ══════════════════════════════════════════════════════════════════════════════
# Experiment settings
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_108_hierarchical_beta_vae_uvmd"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

# ── UVMD ──────────────────────────────────────────────────────────────────────
K_MODES = 4           # frequency modes (aligns with exp_93 optimal)
L_UNROLL = 8          # ADMM unrolled iterations (exp_93 default)

# ── β-VAE latent dimensions ────────────────────────────────────────────────────
# Keep small to avoid latent collapse while still capturing structure.
# Total content per window: K × C × content_dim = 4 × 8 × 16 = 512 → pooled to 2×16=32
CONTENT_DIM = 16      # z_content per (mode, channel)
STYLE_DIM   = 8       # z_style   per (mode, channel)

# ── Per-channel CNN backbone ───────────────────────────────────────────────────
CNN_CHANNELS = (32, 64)  # lightweight; applied K×C times per forward pass

# ── Attentive Statistics Pooling ──────────────────────────────────────────────
ASP_BOTTLENECK = 64

# ── Classification head ───────────────────────────────────────────────────────
CLF_HIDDEN = 128

# ── Loss weights ──────────────────────────────────────────────────────────────
# β_content: low — avoids destroying gesture information in content codes.
# β_style:   moderate — pushes style codes toward standard normal.
# λ_mi:      moderate — separates content from style via distance correlation.
# λ_overlap: small — keeps UVMD modes spectrally separated.
BETA_CONTENT      = 0.05
BETA_STYLE        = 0.05
LAMBDA_MI         = 0.1
LAMBDA_OVERLAP    = 0.01
BETA_ANNEAL_EPOCHS = 10   # anneal β from 0 → target over this many epochs


# ══════════════════════════════════════════════════════════════════════════════
# Helper: grouped_to_arrays (defined locally — not in any processing module)
# ══════════════════════════════════════════════════════════════════════════════

def grouped_to_arrays(grouped_windows: Dict[int, List[np.ndarray]]):
    """
    Convert grouped_windows (Dict[gesture_id, List[ndarray]]) to flat arrays.

    Args:
        grouped_windows: Dict[int, List[np.ndarray]]
            gesture_id → list of rep-level arrays, each shape (N_rep, T, C)

    Returns:
        (windows, labels): Tuple[np.ndarray, np.ndarray]
            windows: (N_total, T, C), labels: (N_total,)
    """
    all_windows: List[np.ndarray] = []
    all_labels:  List[np.ndarray] = []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                all_windows.append(rep_arr)
                all_labels.append(np.full(len(rep_arr), gid, dtype=np.int64))
    if not all_windows:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Split construction (no subject provenance needed — β-VAE, no GroupDRO)
# ══════════════════════════════════════════════════════════════════════════════

def _build_splits(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: "MultiSubjectLoader",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Build standard train / val / test splits for LOSO evaluation.

    Train and val contain ONLY train-subject windows.
    Test contains ONLY test-subject windows.

    No subject provenance tracking required here (unlike exp_106/57 which use
    GroupDRO). The β-VAE is trained with a simple CE + KL + MI objective.

    Returns
    ───────
    dict with keys "train", "val", "test":
        each is Dict[gesture_id (int), np.ndarray (N, T, C)]

    LOSO safety:
        · splits["train"] and splits["val"] ← train-subject data only
        · splits["test"] ← test-subject data only
        · No test-subject windows contaminate training or validation.
    """
    rng = np.random.RandomState(seed)

    # ── Collect train windows per gesture ────────────────────────────────
    train_dict: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}

    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        # subjects_data values are tuples (emg, segments, grouped_windows)
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        for gid in common_gestures:
            if gid not in filtered:
                continue
            for rep_arr in filtered[gid]:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    train_dict[gid].append(rep_arr)

    # ── Stratified train/val split ────────────────────────────────────────
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)  # (N, T, C)
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        final_train[gid] = X_gid[train_idx]
        final_val[gid]   = X_gid[val_idx]

    # ── Test split (test subject only) ────────────────────────────────────
    final_test: Dict[int, np.ndarray] = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
        if valid:
            final_test[gid] = np.concatenate(valid, axis=0)

    return {
        "train": final_train,
        "val":   final_val,
        "test":  final_test,
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
    Execute one LOSO fold: train on train_subjects, evaluate on test_subject.

    The test subject appears ONLY in splits["test"] and is never used during
    parameter optimisation or model selection (early stopping uses val_loss
    which is derived from train-subject data).

    Returns
    ───────
    dict with keys: test_subject, test_accuracy, test_f1_macro, (and error if failed)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "hierarchical_beta_vae_uvmd"

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # ── Load all subjects (train + test, test used only in splits["test"]) ──
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

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

    # ── Build splits ──────────────────────────────────────────────────────
    splits = _build_splits(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    # Log split sizes
    for sname in ("train", "val", "test"):
        total = sum(
            len(arr)
            for arr in splits[sname].values()
            if isinstance(arr, np.ndarray) and arr.ndim == 3
        )
        logger.info(
            f"{sname.upper()}: {total} windows across {len(splits[sname])} gestures"
        )

    # ── Create trainer ────────────────────────────────────────────────────
    trainer = HierarchicalBetaVAETrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        beta_content=BETA_CONTENT,
        beta_style=BETA_STYLE,
        lambda_mi=LAMBDA_MI,
        lambda_overlap=LAMBDA_OVERLAP,
        beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
        K=K_MODES,
        L=L_UNROLL,
        content_dim=CONTENT_DIM,
        style_dim=STYLE_DIM,
        cnn_channels=CNN_CHANNELS,
        asp_bottleneck=ASP_BOTTLENECK,
        clf_hidden=CLF_HIDDEN,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    try:
        trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed for test_subject={test_subject}: {e}")
        traceback.print_exc()
        return {
            "test_subject":   test_subject,
            "model_type":     "hierarchical_beta_vae_uvmd",
            "approach":       APPROACH,
            "test_accuracy":  None,
            "test_f1_macro":  None,
            "error":          str(e),
        }

    # ── Build test arrays (using class_ids from trainer) ──────────────────
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
            "test_subject":   test_subject,
            "model_type":     "hierarchical_beta_vae_uvmd",
            "approach":       APPROACH,
            "test_accuracy":  None,
            "test_f1_macro":  None,
            "error":          "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)   # (N, T, C)
    y_test = np.concatenate(y_test_list, axis=0)   # (N,)

    # ── Evaluate on test subject (LOSO final evaluation) ──────────────────
    # evaluate_numpy: transposes, standardises (train stats), eval mode, z = μ.
    # No adaptation to test subject — LOSO compliant.
    test_results = trainer.evaluate_numpy(
        X_test, y_test,
        split_name=f"loso_test_{test_subject}",
        visualize=True,
    )
    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] {test_subject} | "
        f"acc={test_acc:.4f}, f1={test_f1:.4f}"
    )

    # ── Save fold results ─────────────────────────────────────────────────
    fold_result = {
        "test_subject":   test_subject,
        "train_subjects": train_subjects,
        "common_gestures": common_gestures,
        "config": {
            "K": K_MODES, "L": L_UNROLL,
            "content_dim": CONTENT_DIM, "style_dim": STYLE_DIM,
            "cnn_channels": CNN_CHANNELS,
            "beta_content": BETA_CONTENT, "beta_style": BETA_STYLE,
            "lambda_mi": LAMBDA_MI, "lambda_overlap": LAMBDA_OVERLAP,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
        },
        "cross_subject_test": {
            "subject":   test_subject,
            "accuracy":  test_acc,
            "f1_macro":  test_f1,
            "report":    test_results.get("report"),
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
            "model_type": "hierarchical_beta_vae_uvmd",
            "approach": APPROACH,
            "exercises": exercises,
            "config": fold_result["config"],
            "metrics": {"test_accuracy": test_acc, "test_f1_macro": test_f1},
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
        "test_subject":  test_subject,
        "model_type":    "hierarchical_beta_vae_uvmd",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main: LOSO loop over all subjects
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Subject list: defaults to CI_TEST_SUBJECTS (5 subjects) to prevent
    # FileNotFoundError on server (only CI symlinks exist by default).
    # Override via --subjects DB2_s1,DB2_s2,... or --ci for 5-subject subset.
    ALL_SUBJECTS = parse_subjects_args()

    BASE_DIR   = ROOT / "data"
    TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        model_type="hierarchical_beta_vae_uvmd",
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
    print(f"Hypothesis: Hierarchical Conditional β-VAE + UVMD + SoftAGC")
    print(f"Subjects  : {ALL_SUBJECTS}  ({len(ALL_SUBJECTS)} total)")
    print(f"Exercises : {EXERCISES}")
    print(f"K={K_MODES} modes, L={L_UNROLL} ADMM steps")
    print(f"content_dim={CONTENT_DIM}, style_dim={STYLE_DIM} per (mode, channel)")
    print(f"β_content={BETA_CONTENT}, β_style={BETA_STYLE}, "
          f"λ_mi={LAMBDA_MI}, λ_overlap={LAMBDA_OVERLAP}")
    print(f"Output    : {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results: List[Dict] = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "hierarchical_beta_vae_uvmd" / f"test_{test_subject}"

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

    # ── Aggregate LOSO results ────────────────────────────────────────────
    valid = [r for r in all_loso_results if r.get("test_f1_macro") is not None]

    print(f"\n{'=' * 60}")
    print(f"Hierarchical β-VAE UVMD — LOSO Summary ({len(valid)} folds)")

    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"]  for r in valid]
        mean_f1  = float(np.mean(f1s))
        std_f1   = float(np.std(f1s))
        worst_f1 = float(np.min(f1s))
        best_f1  = float(np.max(f1s))

        print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Worst-subject F1: {worst_f1:.4f}")
        print(f"  Best-subject F1:  {best_f1:.4f}")
        print(f"\n  Per-subject results:")
        for r in sorted(valid, key=lambda x: x["test_f1_macro"]):
            f1_val = r["test_f1_macro"]
            acc_val = r["test_accuracy"]
            f1_str  = f"{f1_val:.4f}"  if f1_val  is not None else "N/A"
            acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
            print(f"    {r['test_subject']}: acc={acc_str}, f1={f1_str}")

    print(f"{'=' * 60}\n")

    # ── Save LOSO summary ─────────────────────────────────────────────────
    summary: Dict = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "H1_new: Hierarchical β-VAE + UVMD + SoftAGC",
        "timestamp":  TIMESTAMP,
        "subjects":   ALL_SUBJECTS,
        "approach":   APPROACH,
        "config": {
            "K": K_MODES, "L": L_UNROLL,
            "content_dim": CONTENT_DIM, "style_dim": STYLE_DIM,
            "cnn_channels": CNN_CHANNELS,
            "asp_bottleneck": ASP_BOTTLENECK,
            "clf_hidden": CLF_HIDDEN,
            "beta_content": BETA_CONTENT,
            "beta_style":   BETA_STYLE,
            "lambda_mi":    LAMBDA_MI,
            "lambda_overlap": LAMBDA_OVERLAP,
            "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
        },
        "results": all_loso_results,
    }

    if valid:
        summary["aggregate"] = {
            "mean_accuracy":  float(np.mean(accs)),
            "std_accuracy":   float(np.std(accs)),
            "mean_f1_macro":  mean_f1,
            "std_f1_macro":   std_f1,
            "worst_f1_macro": worst_f1,
            "best_f1_macro":  best_f1,
            "num_folds":      len(valid),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── Report to hypothesis_executor (optional dependency) ───────────────
    # Wrapped in try/except to avoid crashes when hypothesis_executor is not
    # installed (e.g., on CI or local machines without the agent package).
    try:
        from hypothesis_executor import mark_hypothesis_failed, mark_hypothesis_verified
        if valid:
            mark_hypothesis_verified(
                "H1_new",
                {
                    "mean_accuracy":  float(np.mean(accs)),
                    "std_accuracy":   float(np.std(accs)),
                    "mean_f1_macro":  mean_f1,
                    "std_f1_macro":   std_f1,
                    "worst_f1_macro": worst_f1,
                },
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed("H1_new", "All LOSO folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
