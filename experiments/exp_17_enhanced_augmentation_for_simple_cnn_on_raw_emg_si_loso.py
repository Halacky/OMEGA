# FILE: experiments/exp_17_enhanced_augmentation_for_simple_cnn_on_raw_emg_si_loso.py
"""
Enhanced Augmentation for Simple CNN on Raw EMG Signals

Hypothesis: Applying a more comprehensive augmentation strategy (noise + time_warp + rotation)
to the simple_cnn model trained on raw EMG signals will improve generalization and accuracy
beyond the current best augmented result of 0.3566.

Rotation augmentation (amplitude scaling) helps models become invariant to amplitude variations
that occur due to electrode-skin contact and muscle contraction strength differences across subjects.
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# добавить корень репо в sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig

from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer

from visualization.base import Visualizer

from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


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


class EnhancedAugmentationTrainer(WindowClassifierTrainer):
    """
    Extended WindowClassifierTrainer with rotation (amplitude scaling) augmentation.
    
    Rotation augmentation applies random amplitude scaling to simulate variations
    in electrode-skin contact and muscle contraction strength across subjects.
    """
    
    def __init__(self, *args, rotation_range: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation_range = rotation_range
    
    def _apply_augmentation(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply comprehensive augmentation: noise + time_warp + rotation (amplitude scaling).
        
        Args:
            X: Input tensor of shape (batch, channels, time)
            
        Returns:
            Augmented tensor
        """
        X_aug = X.clone()
        
        # 1. Noise augmentation
        if self.cfg.aug_apply_noise and self.cfg.aug_noise_std > 0:
            noise = torch.randn_like(X_aug) * self.cfg.aug_noise_std
            X_aug = X_aug + noise
        
        # 2. Time warp augmentation (simple implementation via random stretching)
        if self.cfg.aug_apply_time_warp and self.cfg.aug_time_warp_max > 0:
            batch_size, channels, time_steps = X_aug.shape
            warp_factor = 1.0 + torch.rand(batch_size, device=X_aug.device) * 2 * self.cfg.aug_time_warp_max - self.cfg.aug_time_warp_max
            # Apply slight time warping via interpolation
            for i in range(batch_size):
                if abs(warp_factor[i] - 1.0) > 0.01:
                    # Create new time indices
                    orig_indices = torch.linspace(0, time_steps - 1, time_steps, device=X_aug.device)
                    new_length = int(time_steps * warp_factor[i])
                    new_length = max(time_steps // 2, min(time_steps * 2, new_length))
                    new_indices = torch.linspace(0, time_steps - 1, new_length, device=X_aug.device)
                    # Interpolate
                    warped = torch.nn.functional.interpolate(
                        X_aug[i:i+1],
                        size=new_length,
                        mode='linear',
                        align_corners=True
                    )
                    # Resample back to original size
                    X_aug[i:i+1] = torch.nn.functional.interpolate(
                        warped,
                        size=time_steps,
                        mode='linear',
                        align_corners=True
                    )
        
        # 3. Rotation augmentation (amplitude scaling)
        if self.rotation_range > 0:
            # Random scaling factor per sample and per channel
            batch_size, channels, time_steps = X_aug.shape
            # Generate random scaling factors: uniform in [1-rotation_range, 1+rotation_range]
            scaling_factors = 1.0 + (torch.rand(batch_size, channels, 1, device=X_aug.device) * 2 - 1) * self.rotation_range
            X_aug = X_aug * scaling_factors
        
        return X_aug
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Override train method to inject enhanced augmentation.
        """
        device = self.cfg.device
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        # Apply per-channel standardization (train-time normalization)
        train_mean = X_train_t.mean(dim=(0, 2), keepdim=True)
        train_std = X_train_t.std(dim=(0, 2), keepdim=True) + 1e-8
        X_train_t = (X_train_t - train_mean) / train_std
        X_val_t = (X_val_t - train_mean) / train_std
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True if device == "cuda" else False,
        )
        
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        
        # Create model
        in_channels = X_train.shape[1]
        model = self._create_model(in_channels, num_classes, self.cfg.model_type)
        model = model.to(device)
        
        # Loss function
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.cfg.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Apply enhanced augmentation during training
                if self.cfg.aug_apply:
                    X_batch = self._apply_augmentation(X_batch)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    
                    _, predicted = outputs.max(1)
                    val_total += y_batch.size(0)
                    val_correct += predicted.eq(y_batch).sum().item()
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            scheduler.step(val_acc)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save model
        torch.save(model.state_dict(), self.output_dir / "best_model.pt")
        
        return {
            "best_val_accuracy": best_val_acc,
            "model": model,
            "train_mean": train_mean.cpu().numpy(),
            "train_std": train_std.cpu().numpy(),
        }


def run_single_loso_fold_enhanced_aug(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    approach: str,
    use_improved_processing: bool,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    rotation_range: float = 0.1,
) -> Dict:
    """
    LOSO fold with enhanced augmentation (noise + time_warp + rotation).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    # Fix seed
    seed_everything(train_cfg.seed, verbose=False)

    # Set pipeline type and model
    train_cfg.pipeline_type = approach
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

    # Loader
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=use_improved_processing,
    )

    base_viz = Visualizer(output_dir, logger)

    # Use enhanced augmentation trainer
    trainer = EnhancedAugmentationTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        rotation_range=rotation_range,
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
            "approach": approach,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: {model_type} | Approach: {approach} | "
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
        "model_type": model_type,
        "approach": approach,
        "exercises": exercises,
        "use_improved_processing": use_improved_processing,
        "augmentation": {
            "noise": train_cfg.aug_apply_noise,
            "noise_std": train_cfg.aug_noise_std,
            "time_warp": train_cfg.aug_apply_time_warp,
            "time_warp_max": train_cfg.aug_time_warp_max,
            "rotation_range": rotation_range,
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

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": approach,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    EXPERIMENT_NAME = "exp_17_enhanced_augmentation_for_simple_cnn_on_raw_emg_si_loso"
    HYPOTHESIS_ID = "c7d68e78-1306-4585-a213-4675c9920b82"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")

    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--ci", type=int, default=0)
    _parser.add_argument("--subjects", type=str, default=None)
    _args, _ = _parser.parse_known_args()

    _CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
    _FULL_SUBJECTS = [
        "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
        "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
        "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
        "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
    ]
    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    else:
        # Default to CI subjects — server only has symlinks for these 5
        # Pass --subjects DB2_s1,DB2_s2,... to use a custom/full list
        ALL_SUBJECTS = _CI_SUBJECTS

    EXERCISES = ["E1"]
    MODEL_TYPE = "simple_cnn"
    APPROACH = "deep_raw"
    ROTATION_RANGE = 0.1  # Amplitude scaling range ±10%

    # Processing config (same as exp6 baseline)
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

    # Training config with noise + time_warp + rotation augmentation
    train_cfg = TrainingConfig(
        batch_size=4096,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: {MODEL_TYPE} | Approach: {APPROACH}")
    print(f"Augmentation: noise({train_cfg.aug_noise_std}) + time_warp({train_cfg.aug_time_warp_max}) + rotation({ROTATION_RANGE})")
    print(f"LOSO n={len(ALL_SUBJECTS)} subjects")

    all_loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        try:
            fold_res = run_single_loso_fold_enhanced_aug(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                model_type=MODEL_TYPE,
                approach=APPROACH,
                use_improved_processing=True,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                rotation_range=ROTATION_RANGE,
            )
            all_loso_results.append(fold_res)
            if fold_res.get("test_accuracy") is not None:
                print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, f1={fold_res['test_f1_macro']:.4f}")
            else:
                print(f"  ✗ {test_subject}: {fold_res.get('error', 'Unknown error')}")
        except Exception as e:
            global_logger.error(f"Failed {test_subject}: {e}")
            traceback.print_exc()
            all_loso_results.append({
                "test_subject": test_subject,
                "model_type": MODEL_TYPE,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })

    # Compute aggregate statistics
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        
        aggregate = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
        }
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS:")
        print(f"  Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"  Successful folds: {len(valid_results)}/{len(ALL_SUBJECTS)}")
        print(f"{'='*60}")
    else:
        aggregate = {}
        print("\nNo successful folds completed!")

    # Save LOSO summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "model": MODEL_TYPE,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": {
            "noise": True,
            "noise_std": train_cfg.aug_noise_std,
            "time_warp": True,
            "time_warp_max": train_cfg.aug_time_warp_max,
            "rotation": True,
            "rotation_range": ROTATION_RANGE,
            "description": "Comprehensive augmentation: noise + time_warp + rotation (amplitude scaling)",
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

    if aggregate:
        best_metrics = aggregate.copy()
        best_metrics["best_model"] = MODEL_TYPE
        mark_hypothesis_verified(
            hypothesis_id=HYPOTHESIS_ID,
            metrics=best_metrics,
            experiment_name=EXPERIMENT_NAME,
        )
        print(f"\nHypothesis {HYPOTHESIS_ID} marked as VERIFIED in Qdrant.")
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )
        print(f"\nHypothesis {HYPOTHESIS_ID} marked as FAILED in Qdrant.")


if __name__ == "__main__":
    main()