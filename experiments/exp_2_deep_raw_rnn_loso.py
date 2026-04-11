#!/usr/bin/env python3
"""
Experiment 2: Deep Learning with Raw Windows - RNN-based Models
LOSO Cross-Subject Evaluation

Feature Set: raw windows (deep_raw)
Models: BiLSTM, BiLSTM+Attention, BiGRU, CNN-LSTM, CNN-GRU-Attention
Protocol: Leave-One-Subject-Out (40 subjects)
"""

import sys
import os
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_models_improved import compare_models

def main():
    # ============================================================================
    # EXPERIMENT CONFIGURATION
    # ============================================================================
    
    EXPERIMENT_NAME = "exp2_deep_raw_rnn_loso"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    
    ALL_SUBJECTS = [
        "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5", 
        "DB2_s6", "DB2_s7", "DB2_s8", "DB2_s9", "DB2_s10",
        "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
        "DB2_s16", "DB2_s17", "DB2_s18", "DB2_s19", "DB2_s20",
        "DB2_s21", "DB2_s22", "DB2_s23", "DB2_s24", "DB2_s25",
        "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
        "DB2_s31", "DB2_s32", "DB2_s33", "DB2_s34", "DB2_s35",
        "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
    ]
    
    EXERCISE = "E3"
    
    # ============================================================================
    # HYPERPARAMETERS - IDENTICAL TO EXP1 FOR FAIR COMPARISON
    # ============================================================================
    
    PROCESSING_CONFIG = {
        'window_size': 600,
        'window_overlap': 300,
        'num_channels': 8,
        'sampling_rate': 2000,
        'segment_edge_margin': 0.1,
    }
    
    SPLIT_CONFIG = {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'mode': 'by_segments',
        'shuffle_segments': True,
        'seed': 42,
        'include_rest_in_splits': False,
    }
    
    TRAINING_CONFIG = {
        'batch_size': 4096,
        'epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'early_stopping_patience': 7,
        'use_class_weights': True,
        'seed': 42,
        'num_workers': 16,
        'device': 'cuda',
        'use_handcrafted_features': False,  # RAW WINDOWS
        'pipeline_type': 'deep_raw',
    }
    
    # RNN-based models
    MODEL_TYPES = [
        'bilstm',
        'bilstm_attention',
        'bigru',
        'cnn_lstm',
        'cnn_gru_attention',
    ]
    
    USE_IMPROVED_PROCESSING = True
    
    # ============================================================================
    # RUN LOSO EVALUATION
    # ============================================================================
    
    print("=" * 80)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 80)
    print(f"Feature Set: Raw Windows (deep_raw)")
    print(f"Models: {MODEL_TYPES}")
    print(f"Protocol: LOSO ({len(ALL_SUBJECTS)} subjects)")
    print("=" * 80)
    
    loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        print(f"\n{'='*80}")
        print(f"LOSO Iteration: Test Subject = {test_subject}")
        print(f"{'='*80}")
        
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        try:
            summary = compare_models(
                base_dir=BASE_DIR,
                output_base_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercise=EXERCISE,
                model_types=MODEL_TYPES,
                approach='deep_raw',
                use_improved_processing=USE_IMPROVED_PROCESSING,
            )
            
            for result in summary['results']:
                loso_results.append({
                    'test_subject': test_subject,
                    'model_type': result['model_type'],
                    'test_accuracy': result['test_accuracy'],
                    'test_f1_macro': result['test_f1_macro'],
                    'improved_processing': result.get('improved_processing', USE_IMPROVED_PROCESSING),
                })
            
            print(f"✓ Completed: {test_subject}")
            
        except Exception as e:
            print(f"✗ Failed: {test_subject} - {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================================
    # AGGREGATE RESULTS
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("AGGREGATING LOSO RESULTS")
    print("=" * 80)
    
    aggregate_results = {}
    
    for model_type in MODEL_TYPES:
        model_results = [r for r in loso_results 
                        if r['model_type'] == model_type and r['test_accuracy'] is not None]
        
        if model_results:
            accuracies = [r['test_accuracy'] for r in model_results]
            f1_scores = [r['test_f1_macro'] for r in model_results]
            
            aggregate_results[model_type] = {
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'mean_f1_macro': float(np.mean(f1_scores)),
                'std_f1_macro': float(np.std(f1_scores)),
                'num_subjects': len(accuracies),
                'all_accuracies': accuracies,
                'all_f1_scores': f1_scores,
            }
            
            print(f"{model_type:20s}: "
                  f"Acc={aggregate_results[model_type]['mean_accuracy']:.4f} ± "
                  f"{aggregate_results[model_type]['std_accuracy']:.4f}, "
                  f"F1={aggregate_results[model_type]['mean_f1_macro']:.4f} ± "
                  f"{aggregate_results[model_type]['std_f1_macro']:.4f}")
    
    loso_summary = {
        'experiment_name': EXPERIMENT_NAME,
        'feature_set': 'deep_raw',
        'models': MODEL_TYPES,
        'subjects': ALL_SUBJECTS,
        'exercise': EXERCISE,
        'processing_config': PROCESSING_CONFIG,
        'split_config': SPLIT_CONFIG,
        'training_config': TRAINING_CONFIG,
        'use_improved_processing': USE_IMPROVED_PROCESSING,
        'aggregate_results': aggregate_results,
        'individual_results': loso_results,
    }
    
    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(loso_summary, f, indent=4, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE: {EXPERIMENT_NAME}")
    print(f"Results: {OUTPUT_DIR.resolve()}")
    print("=" * 80)
    
    return loso_summary


if __name__ == "__main__":
    main()