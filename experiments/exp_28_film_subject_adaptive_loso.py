"""
Experiment 28: FiLM Subject-Adaptive CNN-GRU-Attention (LOSO)

Hypothesis H2: A model with learnable subject-style embedding (without knowing
subject ID!) can adapt to inter-subject variability via FiLM conditioning.

Key idea: Instead of removing subject variability, parameterize it.
- Self-supervised pretext: predict pseudo-subject cluster via K-means on signal features
- StyleEncoder computes z_subject from a few reference windows
- FiLM layers: y = γ(z_subject) * BN(x) + β(z_subject)
- At test time: z_subject is computed from K calibration windows of the test subject

Architecture: CNN-GRU-Attention with FiLM conditioning after each CNN block.
"""

import os
import sys
import json
import gc
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    DEFAULT_SUBJECTS,
    CI_TEST_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.film_trainer import FiLMSubjectAdaptiveTrainer
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ========== EXPERIMENT SETTINGS ==========
EXPERIMENT_NAME = "exp_28_film_subject_adaptive"
MODEL_TYPES = ["film_subject_adaptive"]
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True

# FiLM-specific hyperparameters
STYLE_DIM = 64
NUM_REF_WINDOWS = 5
NUM_PSEUDO_CLUSTERS = 10
AUX_LOSS_WEIGHT = 0.1


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
    Single LOSO fold: train FiLM Subject-Adaptive model, test on test_subject.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = model_type

    # Save configs
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
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

    base_viz = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    # Custom FiLM trainer
    trainer = FiLMSubjectAdaptiveTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        num_ref_windows=NUM_REF_WINDOWS,
        num_pseudo_clusters=NUM_PSEUDO_CLUSTERS,
        aux_loss_weight=AUX_LOSS_WEIGHT,
        style_dim=STYLE_DIM,
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
        print(f"Error in LOSO fold (test_subject={test_subject}, model={model_type}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "approach": APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: {model_type} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": model_type,
        "approach": APPROACH,
        "exercises": exercises,
        "use_improved_processing": USE_IMPROVED_PROCESSING,
        "film_config": {
            "style_dim": STYLE_DIM,
            "num_ref_windows": NUM_REF_WINDOWS,
            "num_pseudo_clusters": NUM_PSEUDO_CLUSTERS,
            "aux_loss_weight": AUX_LOSS_WEIGHT,
        },
        "config": {
            "processing": asdict(proc_cfg),
            "split": asdict(split_cfg),
            "training": asdict(train_cfg),
            "cross_subject": {
                "train_subjects": train_subjects,
                "test_subject": test_subject,
                "exercises": exercises,
            },
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    ALL_SUBJECTS = parse_subjects_args()
    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    proc_cfg = ProcessingConfig()
    split_cfg = SplitConfig()
    train_cfg = TrainingConfig(
        model_type="film_subject_adaptive",
        pipeline_type=APPROACH,
        use_handcrafted_features=False,
        batch_size=64,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        seed=42,
        use_class_weights=True,
    )

    print(f"{'=' * 80}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Subjects: {ALL_SUBJECTS}")
    print(f"Models: {MODEL_TYPES}")
    print(f"Approach: {APPROACH}")
    print(f"FiLM: style_dim={STYLE_DIM}, K={NUM_REF_WINDOWS}, "
          f"clusters={NUM_PSEUDO_CLUSTERS}, aux_weight={AUX_LOSS_WEIGHT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for model_type in MODEL_TYPES:
        model_results = []

        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output = (
                OUTPUT_ROOT / model_type / f"test_{test_subject}"
            )

            result = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                model_type=model_type,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
            )
            model_results.append(result)
            all_loso_results.append(result)

        # Aggregate per-model results
        valid_results = [r for r in model_results if r.get("test_accuracy") is not None]
        if valid_results:
            accs = [r["test_accuracy"] for r in valid_results]
            f1s = [r["test_f1_macro"] for r in valid_results]
            print(f"\n{'=' * 60}")
            print(f"Model: {model_type} — LOSO Summary ({len(valid_results)} folds)")
            print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"  F1-macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"{'=' * 60}\n")

    # Save overall summary
    summary = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "models": MODEL_TYPES,
        "approach": APPROACH,
        "film_config": {
            "style_dim": STYLE_DIM,
            "num_ref_windows": NUM_REF_WINDOWS,
            "num_pseudo_clusters": NUM_PSEUDO_CLUSTERS,
            "aux_loss_weight": AUX_LOSS_WEIGHT,
        },
        "results": all_loso_results,
    }

    valid_all = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    if valid_all:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid_all])),
            "std_accuracy": float(np.std([r["test_accuracy"] for r in valid_all])),
            "mean_f1_macro": float(np.mean([r["test_f1_macro"] for r in valid_all])),
            "std_f1_macro": float(np.std([r["test_f1_macro"] for r in valid_all])),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    # Report to hypothesis executor if available
    try:
        from hypothesis_executor.callbacks import mark_hypothesis_verified, mark_hypothesis_failed
        if valid_all:
            metrics = summary.get("aggregate", {})
            mark_hypothesis_verified("H2", metrics, EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H2", "All LOSO folds failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
