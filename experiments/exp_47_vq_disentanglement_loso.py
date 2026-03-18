# FILE: experiments/exp_47_vq_disentanglement_loso.py
"""
Experiment 47: Vector Quantization Disentanglement for Content-Style Separation

Hypothesis: Replacing continuous z_content/z_style with discrete VQ codebooks
will create a harder information bottleneck that forces different subjects to
map onto the same canonical gesture codes, improving cross-subject generalization.

Key innovations:
- VQ-Content Codebook (K=128, dim=128): quantizes gesture representations
- VQ-Style Codebook (K=64, dim=64): captures subject variations
- Commitment loss + EMA updates for stable training
- Diversity loss to prevent codebook collapse
- Codebook reset strategy for unused codes
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    DEFAULT_SUBJECTS,
    CI_TEST_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver
from visualization.base import Visualizer

# Import and register the custom model
from models import register_model
from models.vq_disentangle_emg import VQDisentangleEMG

# Register the model
register_model("vq_disentangle_emg", VQDisentangleEMG)


class VQDisentangleTrainer:
    """
    Custom trainer for VQ-based disentanglement model.
    
    Handles:
    - Standard classification loss
    - VQ commitment loss
    - Diversity loss for codebook collapse prevention
    - Periodic codebook reset
    """
    
    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        visualizer,
        codebook_reset_interval: int = 100,
    ):
        self.cfg = train_cfg
        self.logger = logger
        self.output_dir = output_dir
        self.visualizer = visualizer
        self.codebook_reset_interval = codebook_reset_interval
        
        self.device = torch.device(train_cfg.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.class_ids = None
        
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def _splits_to_arrays(self, split_dict: Dict[int, np.ndarray], class_ids: List[int]):
        """Convert {gesture_id: array} split dict to flat (X, y) arrays."""
        X_list, y_list = [], []
        for i, gid in enumerate(class_ids):
            if gid in split_dict and len(split_dict[gid]) > 0:
                X_list.append(split_dict[gid])
                y_list.append(np.full(len(split_dict[gid]), i, dtype=np.int64))
        X = np.concatenate(X_list, axis=0).astype(np.float32)
        y = np.concatenate(y_list, axis=0)
        return X, y

    def fit(self, splits: Dict) -> Dict:
        """
        Train the model using cross-subject splits.

        Args:
            splits: Dict[str, Dict[int, np.ndarray]] — 'train'/'val'/'test',
                    each mapping gesture_id -> windows array (N, T, C)
        """
        # Determine class_ids from training split
        train_d = {gid: arr for gid, arr in splits['train'].items() if len(arr) > 0}
        self.class_ids = sorted(train_d.keys())
        self.class_names = {gid: f"Gesture {gid}" for gid in self.class_ids}
        n_classes = len(self.class_ids)

        # Extract flat arrays
        train_windows, train_labels = self._splits_to_arrays(splits['train'], self.class_ids)
        val_windows, val_labels = self._splits_to_arrays(splits['val'], self.class_ids)

        # Convert to torch format: (N, C, T)
        train_x = torch.from_numpy(train_windows).float().transpose(1, 2)
        train_y = torch.from_numpy(train_labels).long()
        val_x = torch.from_numpy(val_windows).float().transpose(1, 2)
        val_y = torch.from_numpy(val_labels).long()

        # Get dimensions
        n_channels = train_x.shape[1]
        
        self.logger.info(f"VQ Disentangle: {n_channels} channels, {n_classes} classes")
        self.logger.info(f"Train: {len(train_x)}, Val: {len(val_x)}")
        
        # Create model
        self.model = VQDisentangleEMG(
            in_channels=n_channels,
            num_classes=n_classes,
            dropout=self.cfg.dropout,
            content_codebook_size=128,
            content_codebook_dim=128,
            style_codebook_size=64,
            style_codebook_dim=64,
            commitment_cost=0.25,
            diversity_weight=0.1,
        ).to(self.device)
        
        # Compute class weights
        if self.cfg.use_class_weights:
            class_counts = np.bincount(train_labels, minlength=n_classes)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * n_classes
            class_weights = torch.from_numpy(class_weights).float().to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        
        # Scheduler (no verbose param - removed in PyTorch 2.4+)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
        )
        
        # Data loaders
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )
        
        # Training loop
        patience_counter = 0
        global_step = 0
        
        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_aux_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(batch_x)
                
                # Classification loss
                cls_loss = criterion(logits, batch_y)
                
                # Auxiliary losses (commitment + diversity)
                aux_loss = self.model.get_auxiliary_loss()
                
                # Total loss
                total_loss = cls_loss + aux_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_aux_loss += aux_loss.item()
                
                global_step += 1
                
                # Periodic codebook reset
                if global_step % self.codebook_reset_interval == 0:
                    reset_info = self.model.reset_unused_codebooks(threshold=0.001)
                    if reset_info['content_codes_reset'] > 0 or reset_info['style_codes_reset'] > 0:
                        self.logger.info(
                            f"Step {global_step}: Reset {reset_info['content_codes_reset']} content, "
                            f"{reset_info['style_codes_reset']} style codes"
                        )
            
            # Validation
            val_metrics = self.evaluate_numpy(val_x.numpy().transpose(0, 2, 1), val_y.numpy(), "val", visualize=False)
            val_acc = val_metrics['accuracy']
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Log progress
            n_batches = len(train_loader)
            self.logger.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Loss: {epoch_loss/n_batches:.4f} (cls: {epoch_cls_loss/n_batches:.4f}, aux: {epoch_aux_loss/n_batches:.4f}) | "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Log codebook stats
            if (epoch + 1) % 10 == 0:
                stats = self.model.get_codebook_stats()
                self.logger.info(
                    f"  Codebook usage - Content: {stats['content']['used_codes']}/{stats['content']['total_codes']}, "
                    f"Style: {stats['style']['used_codes']}/{stats['style']['total_codes']}"
                )
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
        
        return {'best_val_accuracy': self.best_val_acc}
    
    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str,
        visualize: bool = True
    ) -> Dict:
        """Evaluate model on numpy arrays."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        # Convert to torch: (N, C, T)
        X_tensor = torch.from_numpy(X).float().transpose(1, 2).to(self.device)
        y_tensor = torch.from_numpy(y).long().to(self.device)
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            batch_size = 4096
            for i in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[i:i+batch_size]
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        preds = np.concatenate(all_preds)
        probs = np.concatenate(all_probs)
        
        # Calculate metrics
        accuracy = (preds == y).mean()
        
        # F1 scores
        from sklearn.metrics import f1_score, confusion_matrix, classification_report
        f1_macro = f1_score(y, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(y, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y, preds)
        report = classification_report(y, preds, zero_division=0)

        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'predictions': preds,
            'probabilities': probs,
            'confusion_matrix': cm,
            'report': report,
        }

        if visualize and self.visualizer:
            self.visualizer.plot_confusion_matrix(
                cm,
                class_labels=[str(i) for i in range(len(np.unique(y)))],
                filename=f"cm_{split_name}.png",
            )
        
        return metrics


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
    Single LOSO fold for VQ Disentangle model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    seed_everything(train_cfg.seed, verbose=False)
    
    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    
    # Cross-subject config
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
    
    # Multi-subject loader
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=False,
    )
    
    # Visualizer
    base_viz = Visualizer(output_dir, logger)
    
    # Custom VQ trainer
    trainer = VQDisentangleTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        codebook_reset_interval=100,
    )
    
    # Cross-subject experiment
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
        logger.error(f"Error in LOSO fold (test_subject={test_subject}): {e}")
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
    
    print(
        f"[LOSO] Test subject {test_subject} | "
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
        "exercises": exercises,
        "config": {
            "processing": asdict(proc_cfg),
            "split": asdict(split_cfg),
            "training": asdict(train_cfg),
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
    del experiment, trainer, multi_loader, base_viz
    gc.collect()
    
    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    # ===== Experiment Configuration =====
    EXPERIMENT_NAME = "exp_47_vq_disentanglement_for_content_st_loso"
    HYPOTHESIS_ID = "h-047-vq-disentanglement"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    
    # Subject list via CLI args (default: CI subjects for fast testing)
    ALL_SUBJECTS = parse_subjects_args()
    
    EXERCISES = ["E1"]
    MODEL_TYPES = ["vq_disentangle_emg"]
    
    # ===== Processing Config =====
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )
    
    # ===== Split Config =====
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )
    
    # ===== Training Config =====
    train_cfg = TrainingConfig(
        batch_size=512,
        epochs=60,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        model_type="vq_disentangle_emg",
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )
    
    # ===== Run Experiment =====
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: VQ Disentanglement EMG")
    print(f"Subjects: {len(ALL_SUBJECTS)} ({'CI test' if len(ALL_SUBJECTS) == 5 else 'Full'})")
    print(f"Augmentation: noise + time_warp")
    print(f"VQ: Content K=128/dim=128, Style K=64/dim=64")
    
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
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                )
                all_loso_results.append(fold_res)
                
                acc_str = f"{fold_res['test_accuracy']:.4f}" if fold_res.get('test_accuracy') is not None else "N/A"
                f1_str = f"{fold_res['test_f1_macro']:.4f}" if fold_res.get('test_f1_macro') is not None else "N/A"
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
    
    # ===== Aggregate Results =====
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
        }
        
        print(f"\n{model_type}:")
        print(f"  Accuracy = {aggregate[model_type]['mean_accuracy']:.4f} ± {aggregate[model_type]['std_accuracy']:.4f}")
        print(f"  F1-macro = {aggregate[model_type]['mean_f1_macro']:.4f} ± {aggregate[model_type]['std_f1_macro']:.4f}")
    
    # ===== Save Summary =====
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "noise + time_warp",
        "vq_config": {
            "content_codebook_size": 128,
            "content_codebook_dim": 128,
            "style_codebook_size": 64,
            "style_codebook_dim": 64,
            "commitment_cost": 0.25,
            "diversity_weight": 0.1,
            "codebook_reset_interval": 100,
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
    
    # ===== Update Hypothesis Status in Qdrant =====
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
        print(f"\n✓ Hypothesis {HYPOTHESIS_ID} verified with metrics: {best_metrics}")
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )
        print(f"\n✗ Hypothesis {HYPOTHESIS_ID} failed: No successful LOSO folds")


if __name__ == "__main__":
    main()