# FILE: experiments/exp_22_focal_loss_class_balanced_sampling_for_cnn_gru_att_loso.py
"""
Experiment 22: Focal Loss + Class-Balanced Sampling for CNN-GRU-Attention

Hypothesis: Replacing standard cross-entropy loss with focal loss (gamma=2, alpha per-class 
inverse frequency weighting) and using class-balanced mini-batch sampling during training 
will improve F1 score without sacrificing accuracy, addressing the Acc/F1 divergence pattern.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    CI_TEST_SUBJECTS,
    setup_logging,
    seed_everything,
    ArtifactSaver,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from models import CNNGRUWithAttention


HYPOTHESIS_ID = "212764c1-2890-41d7-ada1-07d9e3e6b76d"


def grouped_to_arrays(grouped_windows):
    """Convert grouped_windows dict to flat (windows, labels) arrays."""
    all_windows = []
    all_labels = []
    for gesture_id in sorted(grouped_windows.keys()):
        for rep_array in grouped_windows[gesture_id]:
            all_windows.append(rep_array)
            all_labels.append(np.full(len(rep_array), gesture_id))
    if not all_windows:
        return np.empty((0,)), np.empty((0,), dtype=int)
    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Per-class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (default 2.0)
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = prob of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha weighting
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class FocalLossTrainer:
    """Custom trainer with focal loss, class-balanced sampling, and F1-based early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device: str,
        output_dir: Path,
        logger,
        gamma: float = 2.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 30,
        early_stopping_patience: int = 7,
    ):
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = None
        self.criterion = None
        self.class_ids = list(range(num_classes))
        
    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        class_counts = np.bincount(labels, minlength=self.num_classes)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        # Inverse frequency normalized to sum to num_classes
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * self.num_classes
        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def _create_balanced_sampler(self, labels: np.ndarray) -> WeightedRandomSampler:
        """Create class-balanced sampler for DataLoader."""
        class_counts = np.bincount(labels, minlength=self.num_classes)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """Train model with focal loss and class-balanced sampling."""
        
        # Compute class weights for focal loss alpha
        class_weights = self._compute_class_weights(y_train)
        self.criterion = FocalLoss(alpha=class_weights, gamma=self.gamma)
        
        # Prepare data
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        # Class-balanced sampler for training
        sampler = self._create_balanced_sampler(y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        
        training_history = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass: (N, T, C) -> (N, C, T) for model
                batch_x_transposed = batch_x.transpose(1, 2)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x_transposed)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())
            
            train_loss /= len(train_labels)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            
            # Validation
            val_metrics = self._evaluate(val_loader)
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1_macro']
            
            self.scheduler.step()
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_f1_macro': train_f1,
                'val_accuracy': val_acc,
                'val_f1_macro': val_f1,
                'lr': self.optimizer.param_groups[0]['lr'],
            })
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
                )
            
            # Early stopping on validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {'training_history': training_history, 'best_val_f1': best_val_f1}
    
    def _evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_x_transposed = batch_x.transpose(1, 2)
                outputs = self.model(batch_x_transposed)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        }
    
    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "test",
        visualize: bool = False,
    ) -> Dict:
        """Evaluate model on numpy arrays."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        metrics = self._evaluate(loader)
        self.logger.info(
            f"{split_name.capitalize()} Results - Accuracy: {metrics['accuracy']:.4f}, "
            f"F1-Macro: {metrics['f1_macro']:.4f}"
        )
        return metrics
    
    def save_model(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
        }, path)


def run_loso_fold_with_focal_loss(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    train_cfg: TrainingConfig,
) -> Dict:
    """Run a single LOSO fold with focal loss training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)
    
    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    
    # Load data
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=False,
    )
    
    # Load training subjects
    train_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=train_subjects,
        exercises=exercises,
        include_rest=False,
    )
    
    # Load test subject
    test_emg, test_segments, test_grouped = multi_loader.load_subject(
        base_dir=base_dir,
        subject_id=test_subject,
        exercise=exercises[0],
        include_rest=False,
    )
    
    # Get common gestures
    common_gestures = multi_loader.get_common_gestures(train_data, max_gestures=10)
    gesture_to_class = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    num_classes = len(gesture_to_class)
    
    logger.info(f"Common gestures: {common_gestures} -> {num_classes} classes")
    
    # Process training data
    X_train_list = []
    y_train_list = []
    for subj_id, (emg, segments, grouped_windows) in train_data.items():
        windows, labels = grouped_to_arrays(grouped_windows)
        # Filter to common gestures
        mask = np.isin(labels, list(common_gestures))
        if mask.sum() == 0:
            continue
        windows = windows[mask]
        labels = labels[mask]
        X_train_list.append(windows)
        y_train_list.extend([gesture_to_class[lbl] for lbl in labels])
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.array(y_train_list)
    
    # Process test data
    test_windows, test_labels = grouped_to_arrays(test_grouped)
    test_mask = np.isin(test_labels, list(common_gestures))
    X_test = test_windows[test_mask]
    y_test = np.array([gesture_to_class[lbl] for lbl in test_labels[test_mask]])
    
    # Standardization (per-channel, computed on training data)
    train_mean = X_train.mean(axis=(0, 1), keepdims=True)
    train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    # Split training data into train/val
    n_samples = len(X_train)
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * 0.15)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    
    logger.info(
        f"Data shapes - Train: {X_train_split.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )
    logger.info(
        f"Class distribution - Train: {dict(Counter(y_train_split))}, "
        f"Val: {dict(Counter(y_val))}, Test: {dict(Counter(y_test))}"
    )
    
    # Create model
    in_channels = X_train.shape[2]  # (N, T, C) -> C channels
    model = CNNGRUWithAttention(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_channels=[32, 64],
        gru_hidden=128,
        gru_layers=2,
        dropout=train_cfg.dropout,
    )
    
    # Create trainer with focal loss settings
    trainer = FocalLossTrainer(
        model=model,
        num_classes=num_classes,
        device=train_cfg.device,
        output_dir=output_dir,
        logger=logger,
        gamma=2.0,
        learning_rate=1e-4,
        weight_decay=train_cfg.weight_decay,
        batch_size=256,
        epochs=30,
        early_stopping_patience=7,
    )
    
    # Train
    train_results = trainer.fit(X_train_split, y_train_split, X_val, y_val)
    
    # Evaluate on test
    test_metrics = trainer.evaluate_numpy(X_test, y_test, split_name="test")
    
    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(train_results['training_history'], f, indent=4)
    
    # Save model
    trainer.save_model(output_dir / "best_model.pt")
    
    # Save fold metadata
    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "cnn_gru_attention",
        "approach": "deep_raw_with_focal_loss",
        "exercises": exercises,
        "config": {
            "processing": asdict(proc_cfg),
            "training": asdict(train_cfg),
        },
        "metrics": {
            "test_accuracy": test_metrics['accuracy'],
            "test_f1_macro": test_metrics['f1_macro'],
            "best_val_f1": train_results['best_val_f1'],
        },
    }
    saver.save_metadata(meta, filename="fold_metadata.json")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model, trainer, multi_loader
    import gc
    gc.collect()
    
    return {
        "test_subject": test_subject,
        "model_type": "cnn_gru_attention",
        "approach": "deep_raw_with_focal_loss",
        "test_accuracy": test_metrics['accuracy'],
        "test_f1_macro": test_metrics['f1_macro'],
    }


def main():
    EXPERIMENT_NAME = "exp_22_focal_loss_class_balanced_sampling_for_cnn_gru_att_loso"
    HYPOTHESIS_ID = "212764c1-2890-41d7-ada1-07d9e3e6b76d"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    ALL_SUBJECTS = parse_subjects_args()
    EXERCISES = ["E1"]
    
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )
    
    train_cfg = TrainingConfig(
        batch_size=256,  # As specified in hypothesis
        epochs=30,
        learning_rate=1e-4,  # Reduced from default
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,  # Will be handled by focal loss alpha
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        model_type="cnn_gru_attention",
        # Light noise augmentation
        aug_apply=True,
        aug_noise_std=0.005,
        aug_time_warp_max=0.0,
        aug_apply_noise=True,
        aug_apply_time_warp=False,
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: cnn_gru_attention with Focal Loss + Class-Balanced Sampling")
    print(f"Subjects: {len(ALL_SUBJECTS)} LOSO folds")
    print(f"Device: {train_cfg.device}")
    
    all_loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        try:
            print(f"\n{'='*60}")
            print(f"LOSO Fold: Test subject = {test_subject}")
            print(f"{'='*60}")
            
            fold_res = run_loso_fold_with_focal_loss(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                train_cfg=train_cfg,
            )
            all_loso_results.append(fold_res)
            
            acc = fold_res.get('test_accuracy')
            f1 = fold_res.get('test_f1_macro')
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            print(f"✓ {test_subject}: Accuracy = {acc_str}, F1-Macro = {f1_str}")
            
        except Exception as e:
            logger.error(f"Failed {test_subject}: {e}")
            traceback.print_exc()
            all_loso_results.append({
                "test_subject": test_subject,
                "model_type": "cnn_gru_attention",
                "approach": "deep_raw_with_focal_loss",
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })
    
    # Aggregate results
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        
        aggregate = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(valid_results),
            "acc_f1_ratio": float(np.mean(accs) / max(np.mean(f1s), 1e-6)),
        }
        
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"F1-Macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"Acc/F1 Ratio: {aggregate['acc_f1_ratio']:.2f}")
    else:
        aggregate = None
        print("\nNo successful folds completed!")
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "model_type": "cnn_gru_attention",
        "approach": "deep_raw_with_focal_loss",
        "training_modifications": {
            "loss_function": "focal_loss",
            "focal_gamma": 2.0,
            "focal_alpha": "inverse_class_frequency",
            "sampling": "class_balanced",
            "scheduler": "cosine_annealing",
            "early_stopping_metric": "val_f1_macro",
        },
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "processing_config": asdict(proc_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(loso_summary, f, indent=4, default=str, ensure_ascii=False)
    
    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")
    
    # === Update hypothesis status in Qdrant ===
    try:
        from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed

        if aggregate:
            best_metrics = {
                "mean_accuracy": aggregate["mean_accuracy"],
                "std_accuracy": aggregate["std_accuracy"],
                "mean_f1_macro": aggregate["mean_f1_macro"],
                "std_f1_macro": aggregate["std_f1_macro"],
                "acc_f1_ratio": aggregate["acc_f1_ratio"],
                "best_model": "cnn_gru_attention_focal_loss",
            }
            mark_hypothesis_verified(
                hypothesis_id=HYPOTHESIS_ID,
                metrics=best_metrics,
                experiment_name=EXPERIMENT_NAME,
            )
            print(f"\n✓ Hypothesis {HYPOTHESIS_ID} marked as VERIFIED in Qdrant")
        else:
            mark_hypothesis_failed(
                hypothesis_id=HYPOTHESIS_ID,
                error_message="No successful LOSO folds completed",
            )
            print(f"\n✗ Hypothesis {HYPOTHESIS_ID} marked as FAILED in Qdrant")
    except ImportError:
        print("hypothesis_executor not available, skipping Qdrant update")


if __name__ == "__main__":
    main()