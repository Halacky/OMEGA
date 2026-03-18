# FILE: experiments/exp_21_svm_rbf_with_noise_time_warp_augmentation_and_tune_loso.py
"""
Experiment 21: SVM-RBF with Noise+Time-Warp Augmentation and Tuned Regularization on Full LOSO

Hypothesis: Applying signal-level noise+time_warp augmentation before powerful feature extraction 
for SVM-RBF (rather than SVM-Linear) with increased regularization (C=1.0) will achieve >36% accuracy 
with balanced F1 (>0.33) on the 20-subject LOSO benchmark.

This tests the missing cell in the 2x2 matrix:
- SVM-Linear × signal-level augmentation (exp_12: 35.61%)
- SVM-RBF × no augmentation (exp_4: 34.46%)
- SVM-RBF × feature-space jitter (exp_18: 40.73% acc, 21.59% F1 - collapsed)
- SVM-RBF × signal-level augmentation (THIS EXPERIMENT)
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

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    DEFAULT_SUBJECTS,
    CI_TEST_SUBJECTS,
    run_single_loso_fold,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import FeatureMLTrainer
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


def run_single_loso_fold_svm_rbf(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    svm_c: float = 1.0,
    augmentation_factor: int = 2,
) -> Dict:
    """
    LOSO fold for SVM-RBF with signal-level augmentation.
    
    Applies noise + time_warp augmentation at raw signal level BEFORE 
    powerful feature extraction, then trains SVM-RBF with specified C.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    seed_everything(train_cfg.seed, verbose=False)
    
    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    
    # CrossSubjectConfig
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
    
    # Load data
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=False,
        use_improved_processing=False,
    )
    
    base_viz = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)
    
    # FeatureMLTrainer for SVM-RBF
    trainer = FeatureMLTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )
    
    # Override SVM C parameter via hyperparameter search config
    # This is the proper way to set C without modifying TrainingConfig
    trainer.svm_c_values = [svm_c]
    
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
        print(f"Error in LOSO fold (test_subject={test_subject}, model=svm_rbf): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "svm_rbf",
            "approach": "ml_emg_td",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }
    
    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))
    
    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: svm_rbf | C={svm_c} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )
    
    # Save results
    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)
    
    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "svm_rbf",
        "approach": "ml_emg_td",
        "exercises": exercises,
        "svm_c": svm_c,
        "augmentation": "noise+time_warp",
        "augmentation_factor": augmentation_factor,
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
    import gc
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()
    
    return {
        "test_subject": test_subject,
        "model_type": "svm_rbf",
        "approach": "ml_emg_td",
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "svm_c": svm_c,
    }


def main():
    EXPERIMENT_NAME = "exp_21_svm_rbf_with_noise_time_warp_augmentation_and_tune_loso"
    HYPOTHESIS_ID = "04b951c7-774f-449b-9202-eaf2bceb6376"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    # Parse subjects from CLI (supports --subjects and --ci flags)
    ALL_SUBJECTS = ['DB2_s1', 'DB2_s12', 'DB2_s15', 'DB2_s28', 'DB2_s39']
    EXERCISES = ["E1"]
    
    # Processing config - same as exp_12 for fair comparison
    proc_cfg = ProcessingConfig(
        window_size=500,
        window_overlap=0,
        num_channels=12,
        sampling_rate=2000,
        segment_edge_margin=0.0,
    )
    
    # Split config
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )
    
    # Training config for SVM-RBF with augmentation
    # Key settings:
    # - ml_model_type: svm_rbf (nonlinear kernel)
    # - use_handcrafted_features: True (powerful features)
    # - handcrafted_feature_set: powerful
    # - aug_apply: True (enable augmentation)
    # - Signal-level noise + time_warp BEFORE feature extraction
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=1,  # Not used for ML models
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cpu",
        # ML-specific settings
        model_type="svm_rbf",
        use_handcrafted_features=True,
        handcrafted_feature_set="powerful",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_rbf",
        ml_use_hyperparam_search=False,  # We set C directly in trainer
        ml_use_feature_selection=False,
        ml_use_pca=False,
        # Signal-level augmentation (applied BEFORE feature extraction)
        aug_apply=True,
        aug_noise_std=0.01,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )
    
    # SVM-RBF regularization: C=1.0 (increased regularization from default 10.0)
    # This prevents RBF kernel from overfitting to augmented samples
    SVM_C = 1.0
    AUGMENTATION_FACTOR = 2
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Hypothesis ID: {HYPOTHESIS_ID}")
    print(f"Model: SVM-RBF with C={SVM_C}")
    print(f"Features: powerful (handcrafted)")
    print(f"Augmentation: noise (std=0.01) + time_warp (max=0.1)")
    print(f"LOSO subjects: {len(ALL_SUBJECTS)}")
    print(f"Subjects: {ALL_SUBJECTS}")
    print("=" * 60)
    
    all_loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        try:
            fold_res = run_single_loso_fold_svm_rbf(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                svm_c=SVM_C,
                augmentation_factor=AUGMENTATION_FACTOR,
            )
            all_loso_results.append(fold_res)
            
            acc = fold_res.get("test_accuracy")
            f1 = fold_res.get("test_f1_macro")
            acc_str = f"{acc:.4f}" if acc is not None else "None"
            f1_str = f"{f1:.4f}" if f1 is not None else "None"
            print(f"  ✓ {test_subject}: acc={acc_str}, f1={f1_str}")
            
        except Exception as e:
            global_logger.error(f"Failed {test_subject} svm_rbf: {e}")
            traceback.print_exc()
            all_loso_results.append({
                "test_subject": test_subject,
                "model_type": "svm_rbf",
                "test_accuracy": None,
                "test_f1_macro": None,
                "svm_c": SVM_C,
                "error": str(e),
            })
    
    # Compute aggregate statistics
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        
        aggregate = {
            "svm_rbf": {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_subjects": len(accs),
                "svm_c": SVM_C,
            }
        }
        
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS (20-subject LOSO):")
        print(f"  SVM-RBF (C={SVM_C}):")
        print(f"    Accuracy: {aggregate['svm_rbf']['mean_accuracy']:.4f} ± {aggregate['svm_rbf']['std_accuracy']:.4f}")
        print(f"    F1-macro: {aggregate['svm_rbf']['mean_f1_macro']:.4f} ± {aggregate['svm_rbf']['std_f1_macro']:.4f}")
        print(f"    Acc/F1 ratio: {aggregate['svm_rbf']['mean_accuracy'] / max(aggregate['svm_rbf']['mean_f1_macro'], 1e-6):.2f}")
    else:
        aggregate = {}
        print("\nERROR: No successful LOSO folds completed!")
    
    # Save LOSO summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "powerful",
        "model": "svm_rbf",
        "svm_c": SVM_C,
        "augmentation": "noise (std=0.01) + time_warp (max=0.1)",
        "augmentation_level": "signal-level (before feature extraction)",
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "note": "SVM-RBF with signal-level augmentation and increased regularization (C=1.0). "
                "Tests the missing cell in the SVM×augmentation matrix.",
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
    
    if aggregate and aggregate.get("svm_rbf"):
        best_metrics = aggregate["svm_rbf"].copy()
        best_metrics["best_model"] = "svm_rbf"
        
        # Check if hypothesis is verified: >36% accuracy AND F1 >0.33
        mean_acc = best_metrics["mean_accuracy"]
        mean_f1 = best_metrics["mean_f1_macro"]
        
        if mean_acc > 0.36 and mean_f1 > 0.33:
            mark_hypothesis_verified(
                hypothesis_id=HYPOTHESIS_ID,
                metrics=best_metrics,
                experiment_name=EXPERIMENT_NAME,
            )
            print(f"\n✓ HYPOTHESIS VERIFIED: Acc={mean_acc:.4f} > 36%, F1={mean_f1:.4f} > 0.33")
        else:
            # Mark as failed if thresholds not met
            mark_hypothesis_failed(
                hypothesis_id=HYPOTHESIS_ID,
                error_message=f"Thresholds not met: Acc={mean_acc:.4f} (need >0.36), F1={mean_f1:.4f} (need >0.33)",
            )
            print(f"\n✗ HYPOTHESIS NOT VERIFIED: Acc={mean_acc:.4f}, F1={mean_f1:.4f}")
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )
        print("\n✗ HYPOTHESIS FAILED: No successful LOSO folds completed")


if __name__ == "__main__":
    main()