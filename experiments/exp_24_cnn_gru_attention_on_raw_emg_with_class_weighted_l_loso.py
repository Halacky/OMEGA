# FILE: experiments/exp_24_cnn_gru_attention_on_raw_emg_with_class_weighted_l_loso.py
"""
Experiment 24: CNN-GRU-Attention on Raw EMG with Class-Weighted Loss and Noise+Time-Warp Augmentation

Hypothesis: Applying class-weighted cross-entropy loss combined with noise+time_warp augmentation
to the CNN-GRU-Attention model on the deep_raw pipeline will improve both accuracy AND F1 score
beyond the current best deep learning result (30.85% acc, 28.19% F1), addressing the class imbalance
that causes F1 collapse in augmented/fusion models while leveraging the best deep architecture.

Key differences from previous experiments:
- Uses class-weighted loss (use_class_weights=True) to address class imbalance
- No fusion (avoiding the class bias introduced by fusion in exp_7, exp_20)
- Batch size 256 (avoiding OOM from exp_16 with 4096)
- Noise + time_warp augmentation for regularization
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

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    run_single_loso_fold,
    make_json_serializable,
    parse_subjects_args,
    DEFAULT_SUBJECTS,
)

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from utils.logging import setup_logging, seed_everything


def main():
    # ====== EXPERIMENT CONFIGURATION ======
    EXPERIMENT_NAME = "exp_24_cnn_gru_attention_on_raw_emg_with_class_weighted_l_loso"
    HYPOTHESIS_ID = "2c875e39-5fe0-4026-aa67-f7772e46980e"
    
    # IMPORTANT: Always use ROOT-relative path for data
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    # Subject list - overridable via CLI (--subjects or --ci)
    ALL_SUBJECTS = ['DB2_s1', 'DB2_s12', 'DB2_s15', 'DB2_s28', 'DB2_s39']
    EXERCISES = ["E1"]
    
    # Model configuration - CNN-GRU-Attention (best deep learning baseline)
    MODEL_TYPES = ["cnn_gru_attention"]
    
    # ====== PROCESSING CONFIG ======
    # Using same window parameters as baseline exp_1 for fair comparison
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,  # Avoid transition regions
    )
    
    # ====== SPLIT CONFIG ======
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )
    
    # ====== TRAINING CONFIG ======
    # Key settings for hypothesis:
    # - use_class_weights=True: inverse-frequency class weighting to address imbalance
    # - aug_apply=True + noise + time_warp: mild regularization
    # - batch_size=256: avoid OOM (exp_16 failed with 4096)
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,  # KEY: class-weighted loss for imbalance
        seed=42,
        num_workers=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="cnn_gru_attention",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        # Augmentation settings
        aug_apply=True,
        aug_noise_std=0.01,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )
    
    # ====== RUN EXPERIMENT ======
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"HYPOTHESIS ID: {HYPOTHESIS_ID}")
    print(f"Models: {MODEL_TYPES}")
    print(f"LOSO n={len(ALL_SUBJECTS)} subjects")
    print(f"Class-weighted loss: ENABLED")
    print(f"Augmentation: noise (std=0.01) + time_warp (max=0.1)")
    print(f"Pipeline: deep_raw (no fusion)")
    print(f"Batch size: 256")
    print("-" * 60)
    
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
                    approach="deep_raw",
                    use_improved_processing=True,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                )
                all_loso_results.append(fold_res)
                
                acc = fold_res.get('test_accuracy')
                f1 = fold_res.get('test_f1_macro')
                acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                print(f"  ✓ {test_subject}: acc={acc_str}, f1={f1_str}")
                
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
    
    # ====== AGGREGATE RESULTS ======
    aggregate = {}
    for model_type in MODEL_TYPES:
        model_results = [
            r for r in all_loso_results 
            if r["model_type"] == model_type and r.get("test_accuracy") is not None
        ]
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
            "acc_f1_ratio": float(np.mean(accs) / np.mean(f1s)) if np.mean(f1s) > 0 else None,
        }
        
        print(f"\n{model_type}:")
        print(f"  Accuracy = {aggregate[model_type]['mean_accuracy']:.4f} ± {aggregate[model_type]['std_accuracy']:.4f}")
        print(f"  F1-macro = {aggregate[model_type]['mean_f1_macro']:.4f} ± {aggregate[model_type]['std_f1_macro']:.4f}")
        if aggregate[model_type]['acc_f1_ratio']:
            print(f"  Acc/F1 ratio = {aggregate[model_type]['acc_f1_ratio']:.2f}")
    
    # ====== SAVE SUMMARY ======
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "noise (std=0.01) + time_warp (max=0.1)",
        "class_weighted_loss": True,
        "note": (
            "Testing whether class-weighted loss prevents F1 collapse in augmented models. "
            "Baseline exp_1: 30.85% acc, 28.19% F1 (ratio 1.09). "
            "Fusion experiments (exp_7, exp_20) showed accuracy gains but F1 collapse. "
            "Key insight: addressing class imbalance in loss function, not architecture."
        ),
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
    
    # ====== UPDATE HYPOTHESIS STATUS IN QDRANT ======
    from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed
    
    if aggregate:
        best_model_name = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
        best_metrics = aggregate[best_model_name].copy()
        best_metrics["best_model"] = best_model_name
        
        # Determine if hypothesis is verified
        # Expected: accuracy 31-33%, F1 28-31%, Acc/F1 ratio < 1.15
        mean_acc = best_metrics["mean_accuracy"]
        mean_f1 = best_metrics["mean_f1_macro"]
        acc_f1_ratio = best_metrics.get("acc_f1_ratio", mean_acc / mean_f1 if mean_f1 > 0 else float('inf'))
        
        # Verification criteria:
        # 1. Accuracy > 30.85% (baseline)
        # 2. F1 > 28.19% (baseline)
        # 3. Acc/F1 ratio < 1.15 (healthy balance, not collapsed)
        verified = (
            mean_acc > 0.3085 and
            mean_f1 > 0.2819 and
            acc_f1_ratio < 1.15
        )
        
        if verified:
            print(f"\n✓ HYPOTHESIS VERIFIED: {mean_acc:.4f} acc, {mean_f1:.4f} F1, ratio {acc_f1_ratio:.2f}")
            mark_hypothesis_verified(
                hypothesis_id=HYPOTHESIS_ID,
                metrics=best_metrics,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            print(f"\n✗ HYPOTHESIS NOT VERIFIED: {mean_acc:.4f} acc, {mean_f1:.4f} F1, ratio {acc_f1_ratio:.2f}")
            best_metrics["verification_failed_reason"] = (
                f"Did not meet all criteria: acc>{0.3085:.4f} ({mean_acc:.4f}), "
                f"F1>{0.2819:.4f} ({mean_f1:.4f}), ratio<1.15 ({acc_f1_ratio:.2f})"
            )
            mark_hypothesis_failed(
                hypothesis_id=HYPOTHESIS_ID,
                error_message=best_metrics.get("verification_failed_reason", "Criteria not met"),
            )
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )


if __name__ == "__main__":
    main()