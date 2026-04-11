# FILE: experiments/exp_12_augmented_svm_with_time_domain_features_for_improv_loso.py
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

# добавить корень репо в sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig

from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer, FeatureMLTrainer

from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer

from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

from exp_X_template_loso import run_single_loso_fold, make_json_serializable


def main():
    EXPERIMENT_NAME = "exp_12_augmented_svm_with_time_domain_features_for_improv_loso"
    HYPOTHESIS_ID = "264d86e7-9299-44ed-87ff-f0d8f622ad82"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")

    ALL_SUBJECTS = [
      "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]
    MODEL_TYPES = ["svm_linear"]
    APPROACH = "ml_emg_td"

    proc_cfg = ProcessingConfig(
        window_size=500,
        window_overlap=0,
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
        num_workers=0,
        device="cpu",
        use_handcrafted_features=True,
        handcrafted_feature_set="powerful",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_linear",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
        aug_apply=True,
        aug_noise_std=0.05,
        aug_time_warp_max=0.2,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    print(f"EXPERIMENT: {EXPERIMENT_NAME} | Models: {MODEL_TYPES} | LOSO n={len(ALL_SUBJECTS)} | Augmentation: noise(std=0.05) + time_warp(max=0.2)")

    all_loso_results = []
    for model_type in MODEL_TYPES:
        for test_subject in ALL_SUBJECTS:
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
                    approach=APPROACH,
                    use_improved_processing=True,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                )
                all_loso_results.append(fold_res)
                if fold_res.get("test_accuracy") is not None:
                    print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, f1={fold_res['test_f1_macro']:.4f}")
                else:
                    print(f"  ✗ {test_subject}: {fold_res.get('error', 'Unknown error')}")
            except Exception as e:
                global_logger.error(f"Failed {test_subject} {model_type}: {e}")
                traceback.print_exc()
                all_loso_results.append({
                    "test_subject": test_subject,
                    "model_type": model_type,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": str(e),
                })

    aggregate = {}
    for model_type in MODEL_TYPES:
        model_results = [r for r in all_loso_results if r["model_type"] == model_type and r.get("test_accuracy") is not None]
        if not model_results:
            continue
        accs = [r["test_accuracy"] for r in model_results]
        f1s = [r["test_f1_macro"] for r in model_results]
        aggregate[model_type] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
        }
        print(f"\n{model_type}: Acc = {aggregate[model_type]['mean_accuracy']:.4f} ± {aggregate[model_type]['std_accuracy']:.4f}, "
              f"F1 = {aggregate[model_type]['mean_f1_macro']:.4f} ± {aggregate[model_type]['std_f1_macro']:.4f}")

    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "powerful",
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "noise(std=0.05) + time_warp(max=0.2)",
        "note": "SVM-linear on powerful time-domain features with data augmentation.",
        "baseline_comparison": {
            "baseline_experiment": "exp4_svm_linear_powerful_loso",
            "baseline_accuracy": 0.3524,
            "expected_improvement": "2-5% (target: 0.37-0.38)",
        },
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")

    # === Update hypothesis status in Qdrant ===
    from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed

    # Find the best model metrics for Qdrant
    if aggregate:
        best_model_name = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
        best_metrics = aggregate[best_model_name]
        best_metrics["best_model"] = best_model_name
        mark_hypothesis_verified(
            hypothesis_id=HYPOTHESIS_ID,
            metrics=best_metrics,
            experiment_name=EXPERIMENT_NAME,
        )
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )


if __name__ == "__main__":
    main()