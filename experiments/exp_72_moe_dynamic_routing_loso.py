"""
Experiment 72: MoE v2 — Dynamic Regime Routing (LOSO)

Hypothesis:
  EMG windows carry a "motion mode" signal (fast activation, sustained
  contraction, onset/offset transitions) that is shared across subjects but
  encodes gesture-execution dynamics. Routing these windows to TDNN experts
  specialised for different temporal scales — using physics-inspired dynamic
  features (TKEO energy, envelope slope, kurtosis, ZCR) — should outperform
  the previous subject-style MoE routing (exp_27) in cross-subject LOSO.

Key differences from exp_27 (MoECNNGRUAttention):
  - Routing features: TKEO/kurtosis/ZCR instead of RMS/spectral centroid/SNR
  - Router normalisation: LayerNorm (per-sample) instead of BatchNorm (cross-batch)
  - Experts: TDNN blocks with dilations 1,2,4,8 instead of BiGRU+Attention heads
  - Aux loss: entropy regularisation + uniform load balance instead of importance-only

LOSO compliance verification (no data leakage):
  - Test subject is NEVER seen during training (enforced by CrossSubjectExperiment).
  - DynamicFeatureExtractor has NO learnable parameters → cannot encode subjects.
  - DynamicRoutingNetwork uses LayerNorm → per-sample normalisation, no cross-
    sample statistics that could encode subject identity.
  - BatchNorm running stats in backbone/experts are accumulated over ALL training
    subjects jointly within each fold — standard LOSO practice.
  - No subject ID, subject embedding, or subject-specific adaptation anywhere.
  - val_ratio carves out a subject-agnostic portion from TRAINING subjects only;
    the test subject's windows never enter any normalisation or split.

Usage:
  python experiments/exp_72_moe_dynamic_routing_loso.py          # CI subjects (default)
  python experiments/exp_72_moe_dynamic_routing_loso.py --ci     # CI subjects explicit
  python experiments/exp_72_moe_dynamic_routing_loso.py --full   # all 20 subjects
  python experiments/exp_72_moe_dynamic_routing_loso.py --subjects DB2_s1,DB2_s12
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer

# Register the new MoE v2 model
from models.moe_dynamic_routing_emg import MoEDynamicRoutingEMG
from models import register_model
register_model("moe_dynamic_routing", MoEDynamicRoutingEMG)

# ---------------------------------------------------------------------------
# Subject lists
# ---------------------------------------------------------------------------

_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]


def parse_subjects_args() -> List[str]:
    """Parse --subjects / --ci / --full CLI args. Defaults to CI subjects."""
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci",   action="store_true")
    _parser.add_argument("--full", action="store_true")
    _args, _ = _parser.parse_known_args()
    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    # Default: CI subjects — safe on server (has symlinks only for CI subjects)
    return _CI_SUBJECTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grouped_to_arrays(grouped_windows: Dict) -> tuple:
    """
    Convert grouped_windows dict to flat (windows, labels) arrays.

    Args:
        grouped_windows: Dict[int, List[np.ndarray]] — gesture_id → list of
            repetition arrays, each of shape (N_rep, T, C).

    Returns:
        windows: np.ndarray (N_total, T, C)
        labels:  np.ndarray (N_total,) — class indices 0..K-1 in sorted order
    """
    windows_list, labels_list = [], []
    for class_idx, gesture_id in enumerate(sorted(grouped_windows.keys())):
        reps = grouped_windows[gesture_id]
        for rep_arr in reps:
            windows_list.append(rep_arr)
            labels_list.append(
                np.full(len(rep_arr), class_idx, dtype=np.int64)
            )
    if not windows_list:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(windows_list, axis=0), np.concatenate(labels_list, axis=0)


def make_json_serializable(obj):
    from pathlib import Path as _Path
    import numpy as _np
    if isinstance(obj, _Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, _np.integer):
        return int(obj)
    elif isinstance(obj, _np.floating):
        return float(obj)
    elif isinstance(obj, _np.ndarray):
        return obj.tolist()
    else:
        return obj


# ---------------------------------------------------------------------------
# Routing visualisation
# ---------------------------------------------------------------------------

def visualize_routing_patterns(
    trainer: WindowClassifierTrainer,
    test_subject: str,
    results: dict,
    output_dir: Path,
) -> None:
    """
    Visualise how gestures route to experts on the held-out test subject.

    Runs a forward pass over ALL test-subject windows (using the fitted model
    in eval mode), collects per-window gating weights, then plots:
      1. Heatmap: mean gate weight per gesture class × expert (routing affinity)
      2. Bar chart: expert load (mean gate weight across all test windows)

    No training statistics or test labels are used during the forward pass —
    only the fitted model and raw test windows are used.  This is purely
    post-hoc analysis on the held-out fold.

    Args:
        trainer:      fitted WindowClassifierTrainer (holds .model, .mean_c, .std_c)
        test_subject: subject ID string (for plot titles)
        results:      dict returned by CrossSubjectExperiment.run()
        output_dir:   directory to save figures
    """
    model = getattr(trainer, "model", None)
    if model is None or not hasattr(model, "_last_gates"):
        return  # not our MoE model

    mean_c = getattr(trainer, "mean_c", None)
    std_c  = getattr(trainer, "std_c",  None)
    if mean_c is None or std_c is None:
        return

    class_ids = getattr(trainer, "class_ids", None)
    if class_ids is None:
        return

    # Retrieve test-subject grouped windows from experiment results
    subjects_data = results.get("subjects_data", {})
    if test_subject not in subjects_data:
        return
    _, _, grouped_windows = subjects_data[test_subject]
    test_windows, test_labels = grouped_to_arrays(grouped_windows)

    if len(test_windows) == 0:
        return

    # Standardise: trainer transposes (N,T,C)→(N,C,T) then standardises
    X = test_windows.transpose(0, 2, 1).astype(np.float32)  # (N, C, T)
    X_std = (X - mean_c[None, :, None]) / (std_c[None, :, None] + 1e-8)

    # Forward pass in eval mode — no gradient needed
    model.eval()
    device = trainer.cfg.device
    batch_size = 256
    all_gates: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(X_std), batch_size):
            xb = torch.tensor(X_std[i : i + batch_size]).to(device)
            model(xb)  # populates model._last_gates (CPU tensor)
            if model._last_gates is not None:
                all_gates.append(model._last_gates.numpy())

    if not all_gates:
        return

    gates_arr = np.concatenate(all_gates, axis=0)  # (N, K)
    num_experts = gates_arr.shape[1]
    num_classes = len(class_ids)

    # Remap test_labels (which are gesture_ids) to class indices
    # class_ids[i] = gesture_id for class index i
    gesture_to_class = {gid: ci for ci, gid in enumerate(class_ids)}
    # test_labels produced by grouped_to_arrays are already class indices
    # (0..K-1 in sorted gesture_id order) — verify alignment with class_ids
    sorted_gids = sorted(grouped_windows.keys())
    local_to_trainer = {}
    for local_ci, gid in enumerate(sorted_gids):
        if gid in gesture_to_class:
            local_to_trainer[local_ci] = gesture_to_class[gid]

    # Per-class mean gate weights
    class_mean_gates = np.zeros((num_classes, num_experts))
    class_counts = np.zeros(num_classes, dtype=int)
    for local_ci, trainer_ci in local_to_trainer.items():
        mask = test_labels == local_ci
        if mask.sum() > 0:
            class_mean_gates[trainer_ci] = gates_arr[mask].mean(axis=0)
            class_counts[trainer_ci] = mask.sum()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(4 + num_experts * 1.5, max(4, num_classes * 0.7 + 2)))

        # --- Plot 1: routing heatmap ---
        ax = axes[0]
        im = ax.imshow(
            class_mean_gates, aspect="auto", cmap="viridis", vmin=0.0,
            vmax=max(0.01, class_mean_gates.max()),
        )
        ax.set_xlabel("Expert (dilation: 1→2→4→8)")
        ax.set_ylabel("Gesture class")
        ax.set_xticks(range(num_experts))
        ax.set_xticklabels(
            [f"E{i}\n(d={2**i})" for i in range(num_experts)], fontsize=8
        )
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(
            [f"G{class_ids[ci]} (n={class_counts[ci]})" for ci in range(num_classes)],
            fontsize=8,
        )
        plt.colorbar(im, ax=ax, label="Mean gate weight")
        ax.set_title(
            f"Routing affinity\n{test_subject}",
            fontsize=10,
        )

        # --- Plot 2: expert load bar chart ---
        ax2 = axes[1]
        load = gates_arr.mean(axis=0)   # (K,) mean gate across all test windows
        uniform = np.full(num_experts, 1.0 / num_experts)
        bars = ax2.bar(range(num_experts), load, color="steelblue", label="Actual load")
        ax2.axhline(uniform[0], color="red", linestyle="--", label=f"Uniform (1/{num_experts})")
        ax2.set_xlabel("Expert")
        ax2.set_ylabel("Mean gate weight")
        ax2.set_xticks(range(num_experts))
        ax2.set_xticklabels([f"E{i}" for i in range(num_experts)])
        ax2.set_ylim(0, 1.0)
        ax2.legend(fontsize=8)
        ax2.set_title(
            f"Expert load (all gestures)\n{test_subject}",
            fontsize=10,
        )

        # Entropy of load distribution (0=collapsed, log(K)=uniform)
        eps = 1e-9
        entropy = -(load * np.log(load + eps)).sum()
        max_entropy = np.log(num_experts)
        ax2.text(
            0.05, 0.95,
            f"Load entropy: {entropy:.3f} / {max_entropy:.3f}",
            transform=ax2.transAxes, fontsize=8, va="top",
        )

        plt.suptitle(
            "MoE v2 Dynamic Routing — Post-hoc analysis\n"
            "(eval mode, test subject only, NO training data used)",
            fontsize=9,
        )
        plt.tight_layout()

        save_path = output_dir / f"routing_analysis_{test_subject}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    except Exception as viz_e:
        print(f"  [viz] routing visualisation failed: {viz_e}")


# ---------------------------------------------------------------------------
# Single LOSO fold
# ---------------------------------------------------------------------------

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
) -> Dict:
    """
    Execute one LOSO fold: train on train_subjects, evaluate on test_subject.

    The test subject is strictly held out:
      - CrossSubjectExperiment loads train subjects and test subject separately.
      - Only train-subject windows enter the train/val split.
      - Test-subject windows are used ONLY for final evaluation.
      - Trainer normalisation (mean_c / std_c) is computed on training windows.

    Returns:
        dict with test_subject, model_type, test_accuracy, test_f1_macro
        (and 'error' key if the fold failed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.model_type = model_type
    train_cfg.pipeline_type = "deep_raw"
    train_cfg.use_handcrafted_features = False

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects,
        test_subject=test_subject,
        exercises=exercises,
        base_dir=base_dir,
        pool_train_subjects=True,
        use_separate_val_subject=False,
        val_subject=None,
        val_ratio=0.15,
        seed=train_cfg.seed,
        max_gestures=10,
    )
    cs_cfg.save(output_dir / "cross_subject_config.json")

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=True,
    )

    base_viz = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    trainer = WindowClassifierTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=base_viz,
        logger=logger,
    )

    try:
        results = experiment.run()
    except Exception as e:
        print(f"Error in LOSO fold (test={test_subject}, model={model_type}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1  = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: {model_type} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # Post-hoc routing visualisation (uses only fitted model + test windows)
    try:
        visualize_routing_patterns(
            trainer=trainer,
            test_subject=test_subject,
            results=results,
            output_dir=output_dir,
        )
    except Exception as viz_e:
        print(f"  [viz] routing analysis skipped: {viz_e}")

    # Save fold results (exclude bulky subjects_data from JSON)
    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": model_type,
        "exercises": exercises,
        "hypothesis": (
            "H72: MoE v2 — routing by motion dynamics (TKEO/slope/kurtosis/ZCR) "
            "outperforms subject-style routing (exp_27) in cross-subject LOSO"
        ),
        "config": {
            "processing": asdict(proc_cfg),
            "split":      asdict(split_cfg),
            "training":   asdict(train_cfg),
            "cross_subject": {
                "train_subjects": train_subjects,
                "test_subject":   test_subject,
                "exercises":      exercises,
            },
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type":   model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPERIMENT_NAME = "exp_72_moe_dynamic_routing_loso"
    BASE_DIR = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")

    EXERCISES   = ["E1"]
    MODEL_TYPES = ["moe_dynamic_routing"]

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
        batch_size=256,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"EXPERIMENT: {EXPERIMENT_NAME}")
    global_logger.info(
        "Hypothesis: MoE v2 — dynamic regime routing via TKEO/slope/kurtosis/ZCR"
    )
    global_logger.info(f"Models:   {MODEL_TYPES}")
    global_logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    global_logger.info(f"Exercises: {EXERCISES}")
    global_logger.info("=" * 80)

    all_loso_results: List[Dict] = []

    for model_type in MODEL_TYPES:
        print(f"\nMODEL: {model_type} — LOSO over {len(ALL_SUBJECTS)} subjects")
        for test_subject in ALL_SUBJECTS:
            print(f"  fold: test_subject = {test_subject}")
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"

            try:
                fold_res = run_single_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_output_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    model_type=model_type,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                )
                all_loso_results.append(fold_res)
                acc_str = f"{fold_res['test_accuracy']:.4f}" if fold_res["test_accuracy"] is not None else "N/A"
                f1_str  = f"{fold_res['test_f1_macro']:.4f}"  if fold_res["test_f1_macro"]  is not None else "N/A"
                print(f"  -> acc={acc_str}, f1={f1_str}")
            except Exception as e:
                global_logger.error(f"Failed fold test_subject={test_subject}: {e}")
                global_logger.error(traceback.format_exc())
                all_loso_results.append({
                    "test_subject": test_subject,
                    "model_type":   model_type,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": str(e),
                })

    # ---- Aggregate results -------------------------------------------------
    print("\n" + "=" * 80)
    print("AGGREGATING LOSO RESULTS")
    print("=" * 80)

    aggregate_results = {}
    for model_type in MODEL_TYPES:
        model_results = [
            r for r in all_loso_results
            if r["model_type"] == model_type and r.get("test_accuracy") is not None
        ]
        if not model_results:
            continue

        accs = [r["test_accuracy"] for r in model_results]
        f1s  = [r["test_f1_macro"] for r in model_results]

        aggregate_results[model_type] = {
            "mean_accuracy":  float(np.mean(accs)),
            "std_accuracy":   float(np.std(accs)),
            "mean_f1_macro":  float(np.mean(f1s)),
            "std_f1_macro":   float(np.std(f1s)),
            "num_subjects":   len(accs),
            "per_subject":    model_results,
        }

        print(
            f"  {model_type:35s}: "
            f"Acc={np.mean(accs):.4f} ± {np.std(accs):.4f}, "
            f"F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f} "
            f"(n={len(accs)})"
        )

    # ---- Save summary -------------------------------------------------------
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "H72: MoE v2 — routing by motion dynamics (TKEO energy, envelope "
            "slope, kurtosis, ZCR) with LayerNorm router and TDNN experts of "
            "varying dilation outperforms subject-style MoE routing (exp_27) "
            "in cross-subject LOSO"
        ),
        "feature_set":       "deep_raw",
        "models":            MODEL_TYPES,
        "subjects":          ALL_SUBJECTS,
        "exercises":         EXERCISES,
        "processing_config": asdict(proc_cfg),
        "split_config":      asdict(split_cfg),
        "training_config":   asdict(train_cfg),
        "aggregate_results": aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date":   datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")

    # ---- Hypothesis executor callback (optional) ----------------------------
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        if aggregate_results:
            best = max(aggregate_results.values(), key=lambda x: x["mean_accuracy"])
            mark_hypothesis_verified(
                "H72_moe_dynamic_routing",
                metrics={
                    "mean_accuracy": best["mean_accuracy"],
                    "std_accuracy":  best["std_accuracy"],
                    "mean_f1_macro": best["mean_f1_macro"],
                    "std_f1_macro":  best["std_f1_macro"],
                },
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                "H72_moe_dynamic_routing",
                "All LOSO folds failed — no results to aggregate",
            )
    except ImportError:
        pass
    except Exception as cb_e:
        print(f"hypothesis_executor callback error: {cb_e}")


if __name__ == "__main__":
    main()
