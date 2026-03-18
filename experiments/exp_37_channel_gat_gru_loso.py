"""
Experiment 37: Channel GAT + BiGRU — Spatio-Temporal Graph Network (LOSO)

Hypothesis H37:
    Modelling inter-muscular co-activation via a graph
    (EMG channels as nodes, edges = Pearson correlation + spectral coherence)
    followed by per-channel BiGRU temporal modelling captures subject-invariant
    representations better than purely temporal (CNN/RNN) or purely graph (GAT)
    approaches.

Architecture (models/channel_gat_gru.py → ChannelGATGRU):
    (B, C, T) raw EMG
    → TemporalCNNEncoder       : shared 1-D CNN per channel, retains T' steps
    → SpectralDynamicAdjacency : Pearson corr + spectral coherence + learnable prior
    → SpatioTemporalGAT        : GATLayer × n_gat_layers at every time step
    → Per-channel BiGRU        : temporal modelling per electrode
    → Channel Attention readout
    → MLP Classifier

Baseline comparison:
    exp_1  (SimpleCNN)
    exp_29 (SpectralTransformer)
    exp_30 (ChannelGAT — graph only, no GRU)

Key differences vs exp_30:
    - Keeps temporal resolution T' after CNN (instead of global pooling)
    - Adds BiGRU after GAT for temporal modelling
    - Adds spectral coherence as an additional edge signal
    - Inspiration: EEG GNNs (BCI), Human Pose Estimation via joint graphs
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict

import numpy as np
import torch

# ── Repo root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Project imports ───────────────────────────────────────────────────────────
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ── Register model ────────────────────────────────────────────────────────────
from models.channel_gat_gru import ChannelGATGRU
from models import register_model

register_model("channel_gat_gru", ChannelGATGRU)

# ── Subject lists ─────────────────────────────────────────────────────────────
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]

_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]


def parse_subjects_args() -> List[str]:
    """Parse --subjects / --ci / --full CLI args.  Defaults to CI subjects."""
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated list of subject IDs")
    _parser.add_argument("--ci",   action="store_true",
                         help="Use 5-subject CI test set")
    _parser.add_argument("--full", action="store_true",
                         help="Use full 20-subject set")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    # Default: CI subjects (safe for vast.ai server with limited symlinks)
    return _CI_SUBJECTS


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Single LOSO fold ──────────────────────────────────────────────────────────

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
    """Train on train_subjects, evaluate on test_subject.  Returns metrics dict."""
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

    from visualization.base import Visualizer
    from visualization.cross_subject import CrossSubjectVisualizer

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
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
    f1_str  = f"{test_f1:.4f}"  if test_f1  is not None else "N/A"
    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: {model_type} | "
        f"Accuracy={acc_str}, F1-macro={f1_str}"
    )

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4,
                  ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": model_type,
        "exercises": exercises,
        "hypothesis": (
            "H37: Graph (GAT, Pearson+spectral coherence edges) + BiGRU temporal "
            "modelling captures subject-invariant inter-muscular co-activation."
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
            "test_accuracy":  test_acc,
            "test_f1_macro":  test_f1,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    EXPERIMENT_NAME = "exp_37_channel_gat_gru_loso"
    BASE_DIR    = ROOT / "data"
    OUTPUT_DIR  = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    ALL_SUBJECTS = parse_subjects_args()

    EXERCISES   = ["E1"]
    MODEL_TYPES = ["channel_gat_gru"]

    # ── Processing config ─────────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=400,
        window_overlap=200,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )

    # ── Split config ──────────────────────────────────────────────────────────
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    # ── Training config ───────────────────────────────────────────────────────
    # ChannelGATGRU is heavier than pure ChannelGAT due to the BiGRU.
    # Use a moderate batch size and slightly longer patience.
    train_cfg = TrainingConfig(
        batch_size=128,
        epochs=70,
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=15,
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
    global_logger.info(f"EXPERIMENT : {EXPERIMENT_NAME}")
    global_logger.info(
        "Hypothesis : H37 — GAT (Pearson + spectral coherence edges) + BiGRU "
        "captures subject-invariant inter-muscular co-activation"
    )
    global_logger.info(f"Models     : {MODEL_TYPES}")
    global_logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    global_logger.info(f"Exercises  : {EXERCISES}")
    global_logger.info("=" * 80)

    all_loso_results: List[Dict] = []

    for model_type in MODEL_TYPES:
        print(f"\nMODEL: {model_type} — LOSO over {len(ALL_SUBJECTS)} subjects")
        for test_subject in ALL_SUBJECTS:
            print(f"  LOSO fold: test_subject = {test_subject}")
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
                acc_str = (
                    f"{fold_res['test_accuracy']:.4f}"
                    if fold_res["test_accuracy"] is not None else "N/A"
                )
                f1_str = (
                    f"{fold_res['test_f1_macro']:.4f}"
                    if fold_res["test_f1_macro"] is not None else "N/A"
                )
                print(f"  → acc={acc_str}, f1={f1_str}")

            except Exception as e:
                global_logger.error(
                    f"Failed fold test_subject={test_subject}: {e}"
                )
                global_logger.error(traceback.format_exc())
                all_loso_results.append({
                    "test_subject": test_subject,
                    "model_type":   model_type,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": str(e),
                })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("AGGREGATING LOSO RESULTS")
    print("=" * 80)

    aggregate_results: Dict = {}
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

    # ── Save summary ──────────────────────────────────────────────────────────
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "H37: Spatio-temporal graph network with GAT (Pearson correlation + "
            "spectral coherence as edges) + per-channel BiGRU captures "
            "subject-invariant inter-muscular co-activation patterns."
        ),
        "feature_set":        "deep_raw",
        "models":             MODEL_TYPES,
        "subjects":           ALL_SUBJECTS,
        "exercises":          EXERCISES,
        "processing_config":  asdict(proc_cfg),
        "split_config":       asdict(split_cfg),
        "training_config":    asdict(train_cfg),
        "aggregate_results":  aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date":    datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4,
                  ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")

    # ── Hypothesis executor callback (optional dependency) ─────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed  # noqa
        if aggregate_results:
            best = max(aggregate_results.values(),
                       key=lambda x: x["mean_accuracy"])
            mark_hypothesis_verified(
                "H37_channel_gat_gru",
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
                "H37_channel_gat_gru",
                "No successful LOSO folds — check data and model configuration.",
            )
    except ImportError:
        pass
    except Exception as e:
        print(f"hypothesis_executor callback error: {e}")


if __name__ == "__main__":
    main()
