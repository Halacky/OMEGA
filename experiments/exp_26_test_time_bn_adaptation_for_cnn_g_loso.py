# FILE: experiments/exp_26_test_time_bn_adaptation_for_cnn_g_loso.py
"""
Test-Time Batch Normalization Adaptation for CNN-GRU-Attention

Hypothesis: Applying test-time adaptation (TTA) by updating batch normalization 
statistics on each test subject's unlabeled data before inference will reduce 
the cross-subject domain gap for CNN-GRU-Attention.

Technique:
1. Train model normally on training subjects
2. Before evaluating on test subject:
   - Set all BN layers to train mode (to update running statistics)
   - Forward-pass all test subject windows (no gradients)
   - Set all BN layers back to eval mode
   - Evaluate on test subject

This requires no labeled target data, no gradient computation, and no 
architectural changes — just a single forward pass to accumulate statistics.
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig

from data.multi_subject_loader import MultiSubjectLoader
from training.trainer import WindowClassifierTrainer
from evaluation.cross_subject import CrossSubjectExperiment

from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer

from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


def adapt_bn_statistics(model: nn.Module, dataloader: DataLoader, device: str):
    """
    Test-Time Batch Normalization Adaptation.
    
    Updates batch normalization running statistics (mean, variance) using
    unlabeled test data. This is the simplest form of test-time adaptation.
    
    Args:
        model: The trained model with BatchNorm layers
        dataloader: DataLoader containing test subject's windows
        device: Device to run on
    
    Returns:
        The model with updated BN statistics
    """
    model.eval()
    
    # Find all BatchNorm modules
    bn_modules = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            bn_modules.append(module)
    
    if not bn_modules:
        print("  [TTA] No BatchNorm layers found, skipping adaptation")
        return model
    
    print(f"  [TTA] Adapting {len(bn_modules)} BatchNorm layer(s) on test subject data...")
    
    # Set only BN layers to train mode (enables running stat updates)
    for bn in bn_modules:
        bn.train()
    
    # Forward pass through all test data to update BN running stats
    with torch.no_grad():
        num_samples = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            model(x)
            num_samples += x.shape[0]
    
    # Set all modules back to eval mode
    model.eval()
    
    print(f"  [TTA] Adapted BN statistics using {num_samples} test windows")
    return model


def run_loso_fold_with_tta(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    use_improved_processing: bool,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    batch_size_for_tta: int = 256,
) -> Dict:
    """
    LOSO fold with Test-Time Batch Normalization Adaptation.
    
    Training proceeds normally, but before evaluation on the test subject,
    we adapt the BN statistics using the test subject's unlabeled windows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)
    
    train_cfg.pipeline_type = "deep_raw"
    train_cfg.model_type = model_type
    
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
    
    # Data loader
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=use_improved_processing,
    )
    
    base_viz = Visualizer(output_dir, logger)
    
    trainer = WindowClassifierTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )
    
    # Standard experiment setup
    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=base_viz,
        logger=logger,
    )
    
    try:
        # Load data
        logger.info(f"Loading data for train subjects: {train_subjects}")
        logger.info(f"Loading data for test subject: {test_subject}")
        
        subjects_data = multi_loader.load_multiple_subjects(
            base_dir=cs_cfg.base_dir,
            subject_ids=cs_cfg.train_subjects + [cs_cfg.test_subject],
            exercises=cs_cfg.exercises,
            include_rest=False,
        )
        
        # Get common gestures
        common_gestures = multi_loader.get_common_gestures(
            subjects_data, max_gestures=cs_cfg.max_gestures
        )
        num_classes = len(common_gestures)
        logger.info(f"Common gestures: {sorted(common_gestures)} ({num_classes} classes)")

        # Prepare splits using experiment's internal method
        splits, split_info = experiment._prepare_splits(subjects_data, sorted(common_gestures))

        # Training
        logger.info("Starting training...")
        trainer.fit(splits)

        # Get trained model and device
        model = trainer.model
        device = train_cfg.device

        # Extract test data from splits (Dict[int, np.ndarray] keyed by gesture_id)
        test_split = splits["test"]
        X_test_list, y_test_list = [], []
        for gid in trainer.class_ids:
            if gid in test_split and len(test_split[gid]) > 0:
                cls_idx = trainer.class_ids.index(gid)
                X_test_list.append(test_split[gid])
                y_test_list.append(np.full(len(test_split[gid]), cls_idx, dtype=np.int64))

        test_windows = np.concatenate(X_test_list, axis=0)  # (N, T, C)
        test_labels = np.concatenate(y_test_list, axis=0)

        # Transpose (N, T, C) -> (N, C, T) and apply channel standardization
        test_windows_c = test_windows.transpose(0, 2, 1)
        test_windows_c = trainer._apply_standardization(test_windows_c, trainer.mean_c, trainer.std_c)

        logger.info(f"Test windows shape: {test_windows_c.shape}")
        logger.info(f"Applying Test-Time BN Adaptation on {len(test_windows_c)} windows...")

        # Create DataLoader for TTA
        test_tensor = torch.tensor(test_windows_c, dtype=torch.float32)
        test_dataset = TensorDataset(test_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_for_tta,
            shuffle=False,
            num_workers=0,
        )

        # Apply TTA: adapt BN statistics
        model = adapt_bn_statistics(model, test_loader, device)

        # Evaluate with adapted model
        logger.info("Evaluating with adapted BN statistics...")

        model.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(test_tensor), batch_size_for_tta):
                batch = test_tensor[i:i + batch_size_for_tta].to(device)
                outputs = model(batch)
                preds = outputs.argmax(dim=1).cpu()
                all_preds.append(preds)

        all_preds = torch.cat(all_preds).numpy()

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        test_acc = accuracy_score(test_labels, all_preds)
        test_f1 = f1_score(test_labels, all_preds, average='macro')

        logger.info(f"Test Accuracy (with TTA): {test_acc:.4f}")
        logger.info(f"Test F1-macro (with TTA): {test_f1:.4f}")

        # Detailed classification report
        class_report = classification_report(
            test_labels, all_preds, output_dict=True, zero_division=0
        )

        results = {
            "cross_subject_test": {
                "accuracy": test_acc,
                "f1_macro": test_f1,
                "classification_report": class_report,
                "num_test_samples": len(test_labels),
            }
        }
        
    except Exception as e:
        logger.error(f"Error in LOSO fold: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "approach": "deep_raw_with_tta",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }
    
    print(
        f"[LOSO-TTA] Test subject {test_subject} | "
        f"Model: {model_type} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )
    
    # Save results
    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)
    
    # Save metadata
    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": model_type,
        "approach": "deep_raw_with_tta",
        "exercises": exercises,
        "use_improved_processing": use_improved_processing,
        "tta_enabled": True,
        "tta_method": "bn_statistics_adaptation",
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
    del experiment, trainer, multi_loader, model
    gc.collect()
    
    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": "deep_raw_with_tta",
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    # ====== Experiment Configuration ======
    EXPERIMENT_NAME = "exp_26_test_time_bn_adaptation_for_cnn_g_loso"
    HYPOTHESIS_ID = "9c0b2f84-8bf2-41f8-9df0-4187086fd96c"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    else:
        ALL_SUBJECTS = ['DB2_s1', 'DB2_s12', 'DB2_s15', 'DB2_s28', 'DB2_s39']
    
    EXERCISES = ["E1"]
    MODEL_TYPES = ["cnn_gru_attention"]  # CNN-GRU-Attention as specified
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Hypothesis ID: {HYPOTHESIS_ID}")
    print(f"Model(s): {MODEL_TYPES}")
    print(f"Subjects: {len(ALL_SUBJECTS)} subjects")
    print(f"Test-Time Adaptation: BN statistics adaptation enabled")
    
    # Processing config
    proc_cfg = ProcessingConfig(
        window_size=500,
        window_overlap=0,
        num_channels=12,
        sampling_rate=2000,
        segment_edge_margin=0.1,
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
    
    # Training config - deep_raw pipeline, no augmentation
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="cnn_gru_attention",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        aug_apply=False,  # No augmentation as per hypothesis
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    all_loso_results = []
    
    for model_type in MODEL_TYPES:
        print(f"\n{'='*60}")
        print(f"Model: {model_type}")
        print(f"{'='*60}")
        
        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"
            
            print(f"\n--- Test subject: {test_subject} ---")
            
            try:
                fold_res = run_loso_fold_with_tta(
                    base_dir=BASE_DIR,
                    output_dir=fold_output_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    model_type=model_type,
                    use_improved_processing=True,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                    batch_size_for_tta=256,
                )
                all_loso_results.append(fold_res)
                
                if fold_res.get("test_accuracy") is not None:
                    print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, "
                          f"f1={fold_res['test_f1_macro']:.4f}")
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
    
    # Aggregate results
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
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
        }
        
        print(f"\n{'='*60}")
        print(f"Results for {model_type}:")
        print(f"  Accuracy: {aggregate[model_type]['mean_accuracy']:.4f} "
              f"± {aggregate[model_type]['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate[model_type]['mean_f1_macro']:.4f} "
              f"± {aggregate[model_type]['std_f1_macro']:.4f}")
        print(f"  Range: [{aggregate[model_type]['min_accuracy']:.4f}, "
              f"{aggregate[model_type]['max_accuracy']:.4f}]")
        print(f"{'='*60}")
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "none",
        "test_time_adaptation": {
            "enabled": True,
            "method": "bn_statistics_adaptation",
            "description": "Update BatchNorm running mean/var on test subject data before inference",
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
    try:
        from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed

        if aggregate:
            best_model_name = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
            best_metrics = aggregate[best_model_name].copy()
            best_metrics["best_model"] = best_model_name
            mark_hypothesis_verified(
                hypothesis_id=HYPOTHESIS_ID,
                metrics=best_metrics,
                experiment_name=EXPERIMENT_NAME,
            )
            print(f"\n✓ Hypothesis {HYPOTHESIS_ID} marked as VERIFIED")
            print(f"  Best model: {best_model_name}")
            print(f"  Mean accuracy: {best_metrics['mean_accuracy']:.4f}")
        else:
            mark_hypothesis_failed(
                hypothesis_id=HYPOTHESIS_ID,
                error_message="No successful LOSO folds completed",
            )
            print(f"\n✗ Hypothesis {HYPOTHESIS_ID} marked as FAILED")
    except ImportError:
        print("hypothesis_executor not available, skipping Qdrant update")


if __name__ == "__main__":
    main()