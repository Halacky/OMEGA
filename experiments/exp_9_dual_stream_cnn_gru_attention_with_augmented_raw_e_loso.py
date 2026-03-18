# FILE: experiments/exp_9_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso.py
"""
Experiment 9: Dual-Stream CNN-GRU-Attention with Augmented Raw EMG and Handcrafted Feature Fusion

Hypothesis: A dual-stream architecture that processes augmented raw EMG signals through a 
CNN-GRU-Attention network while simultaneously processing handcrafted powerful features 
through a lightweight MLP, with late-stage attention-based fusion, will outperform the 
current best models by better leveraging both raw signal temporal patterns and engineered 
feature discriminative power.

Key components:
- Primary stream: Augmented raw EMG through CNN-GRU-Attention
- Secondary stream: Powerful handcrafted features through 2-layer MLP
- Late fusion: Attention mechanism to weight and combine outputs
- Joint training with balanced loss weighting
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from processing.powerful_features import PowerfulFeatureExtractor
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer, FeatureMLTrainer
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# Import and register the dual-stream model
from models.dual_stream_cnn_gru_attention import DualStreamCNNGRUAttention
from models import register_model
register_model("dual_stream_cnn_gru_attention", DualStreamCNNGRUAttention)


def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
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


class DualStreamTrainer:
    """
    Custom trainer for dual-stream model that handles both raw EMG and handcrafted features.
    """
    
    def __init__(
        self,
        train_cfg: TrainingConfig,
        output_dir: Path,
        logger,
        visualizer,
        feature_dim: int = 154,
    ):
        self.cfg = train_cfg
        self.output_dir = output_dir
        self.logger = logger
        self.visualizer = visualizer
        self.feature_dim = feature_dim
        
        self.device = torch.device(train_cfg.device)
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_model_state = None
        
    def _extract_powerful_features(self, windows: np.ndarray) -> np.ndarray:
        """Extract powerful handcrafted features from EMG windows."""
        extractor = PowerfulFeatureExtractor(sampling_rate=2000)

        # PowerfulFeatureExtractor.transform expects (N, T, C), windows are (N, C, T)
        windows_ntc = np.transpose(windows, (0, 2, 1))
        features = extractor.transform(windows_ntc)

        return features.astype(np.float32)
    
    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply noise and time warp augmentation to raw EMG signals."""
        if not self.cfg.aug_apply:
            return x
        
        augmented = x.clone()
        
        # Noise augmentation
        if self.cfg.aug_apply_noise and self.cfg.aug_noise_std > 0:
            noise = torch.randn_like(augmented) * self.cfg.aug_noise_std
            augmented = augmented + noise
        
        # Time warp augmentation
        if self.cfg.aug_apply_time_warp and self.cfg.aug_time_warp_max > 0:
            batch_size, num_channels, seq_len = augmented.shape
            warp_factor = 1.0 + torch.rand(batch_size, 1, 1, device=augmented.device) * self.cfg.aug_time_warp_max * 2 - self.cfg.aug_time_warp_max
            warp_factor = torch.clamp(warp_factor, 1.0 - self.cfg.aug_time_warp_max, 1.0 + self.cfg.aug_time_warp_max)
            
            # Create time indices
            time_idx = torch.linspace(0, 1, seq_len, device=augmented.device).view(1, 1, -1)
            warped_idx = time_idx * warp_factor
            warped_idx = warped_idx.expand(batch_size, num_channels, -1)
            warped_idx = torch.clamp(warped_idx, 0, 1) * (seq_len - 1)
            
            # Interpolate
            warped_idx_floor = warped_idx.long().float()
            warped_idx_ceil = torch.min(warped_idx_floor + 1, torch.tensor(seq_len - 1, device=augmented.device).float())
            weight = warped_idx - warped_idx_floor
            
            augmented = augmented * (1 - weight) + torch.gather(augmented, 2, warped_idx_ceil.long()) * weight
        
        return augmented
    
    def train(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        val_windows: np.ndarray,
        val_labels: np.ndarray,
        num_classes: int,
    ) -> Dict:
        """
        Train the dual-stream model.
        
        Args:
            train_windows: Training windows (N, C, T)
            train_labels: Training labels (N,)
            val_windows: Validation windows (N, C, T)
            val_labels: Validation labels (N,)
            num_classes: Number of gesture classes
        
        Returns:
            Dictionary with training history
        """
        self.logger.info("Extracting handcrafted features for training data...")
        train_features = self._extract_powerful_features(train_windows)
        self.logger.info(f"Train features shape: {train_features.shape}")
        
        self.logger.info("Extracting handcrafted features for validation data...")
        val_features = self._extract_powerful_features(val_windows)
        self.logger.info(f"Val features shape: {val_features.shape}")
        
        # Update feature dimension based on actual extraction
        self.feature_dim = train_features.shape[1]
        
        # Normalize features (per-feature standardization)
        train_mean = train_features.mean(axis=0, keepdims=True)
        train_std = train_features.std(axis=0, keepdims=True) + 1e-8
        train_features = (train_features - train_mean) / train_std
        val_features = (val_features - train_mean) / train_std
        
        # Convert to tensors
        X_train = torch.tensor(train_windows, dtype=torch.float32)
        F_train = torch.tensor(train_features, dtype=torch.float32)
        y_train = torch.tensor(train_labels, dtype=torch.long)
        
        X_val = torch.tensor(val_windows, dtype=torch.float32)
        F_val = torch.tensor(val_features, dtype=torch.float32)
        y_val = torch.tensor(val_labels, dtype=torch.long)
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, F_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        
        val_dataset = TensorDataset(X_val, F_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        
        # Standardize raw signals (per-channel)
        X_train_flat = X_train.permute(0, 2, 1).reshape(-1, X_train.size(1))
        channel_mean = X_train_flat.mean(dim=0)
        channel_std = X_train_flat.std(dim=0) + 1e-8
        
        # Create model
        num_channels = train_windows.shape[1]
        self.model = DualStreamCNNGRUAttention(
            in_channels=num_channels,
            num_classes=num_classes,
            dropout=self.cfg.dropout,
            handcrafted_dim=self.feature_dim,
        ).to(self.device)
        
        self.logger.info(f"Model created: {type(self.model).__name__}")
        self.logger.info(f"  - in_channels: {num_channels}")
        self.logger.info(f"  - num_classes: {num_classes}")
        self.logger.info(f"  - feature_dim: {self.feature_dim}")
        
        # Loss and optimizer
        if self.cfg.use_class_weights:
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / (class_counts + 1e-8)
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
        patience_counter = 0
        
        for epoch in range(self.cfg.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_f, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_f = batch_f.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Apply augmentation to raw signals only
                batch_x = self._apply_augmentation(batch_x)
                
                # Standardize per-channel
                batch_x = (batch_x - channel_mean.view(1, -1, 1).to(self.device)) / \
                          channel_std.view(1, -1, 1).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x, batch_f)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_f, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_f = batch_f.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Standardize (no augmentation during validation)
                    batch_x = (batch_x - channel_mean.view(1, -1, 1).to(self.device)) / \
                              channel_std.view(1, -1, 1).to(self.device)
                    
                    outputs = self.model(batch_x, batch_f)
                    _, predicted = outputs.max(1)
                    
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_acc = val_correct / val_total
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            scheduler.step(val_f1)
            
            # Early stopping check
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.cfg.epochs} - "
                    f"Loss: {train_loss:.4f} - "
                    f"Train Acc: {train_acc:.4f} - "
                    f"Val Acc: {val_acc:.4f} - "
                    f"Val F1: {val_f1:.4f}"
                )
            
            if patience_counter >= self.cfg.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'history': history,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'channel_mean': channel_mean.cpu().numpy(),
            'channel_std': channel_std.cpu().numpy(),
            'feature_mean': train_mean,
            'feature_std': train_std,
        }
    
    def evaluate(
        self,
        test_windows: np.ndarray,
        test_labels: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> Dict:
        """Evaluate the trained model on test data."""
        # Extract features
        test_features = self._extract_powerful_features(test_windows)
        test_features = (test_features - feature_mean) / feature_std
        
        # Convert to tensors
        X_test = torch.tensor(test_windows, dtype=torch.float32)
        F_test = torch.tensor(test_features, dtype=torch.float32)
        y_test = torch.tensor(test_labels, dtype=torch.long)
        
        channel_mean_t = torch.tensor(channel_mean, dtype=torch.float32, device=self.device)
        channel_std_t = torch.tensor(channel_std, dtype=torch.float32, device=self.device)
        
        # Create dataloader
        test_dataset = TensorDataset(X_test, F_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        
        # Evaluate
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_f, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_f = batch_f.to(self.device)
                
                # Standardize
                batch_x = (batch_x - channel_mean_t.view(1, -1, 1)) / \
                          channel_std_t.view(1, -1, 1)
                
                outputs = self.model(batch_x, batch_f)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets),
        }


def run_dual_stream_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
) -> Dict:
    """
    Run a single LOSO fold with the dual-stream model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    seed_everything(train_cfg.seed, verbose=False)
    
    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    
    # Create cross-subject config
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
        use_gpu=True,
        use_improved_processing=True,
    )
    
    base_viz = Visualizer(output_dir, logger)
    
    # Helper: convert grouped_windows to flat arrays
    def grouped_to_arrays(grouped_windows):
        windows_list, labels_list = [], []
        for gesture_id in sorted(grouped_windows.keys()):
            for rep_windows in grouped_windows[gesture_id]:
                if len(rep_windows) > 0:
                    windows_list.append(rep_windows)
                    labels_list.append(np.full(len(rep_windows), gesture_id))
        return np.concatenate(windows_list, axis=0), np.concatenate(labels_list, axis=0)

    # Load training subjects
    train_windows_list = []
    train_labels_list = []
    for subject in train_subjects:
        try:
            emg, segments, grouped_windows = multi_loader.load_subject(
                base_dir=base_dir,
                subject_id=subject,
                exercise=exercises[0],
            )
            w, l = grouped_to_arrays(grouped_windows)
            train_windows_list.append(w)
            train_labels_list.append(l)
            logger.info(f"Loaded {subject}: {len(w)} windows")
        except Exception as e:
            logger.warning(f"Failed to load {subject}: {e}")

    if not train_windows_list:
        raise ValueError("No training data loaded!")

    # Pool training data
    train_windows = np.concatenate(train_windows_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    logger.info(f"Total training windows: {len(train_windows)}")

    # Split train/val
    n_train = len(train_windows)
    indices = np.arange(n_train)
    np.random.seed(train_cfg.seed)
    np.random.shuffle(indices)

    val_size = int(n_train * split_cfg.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = train_windows[train_indices]
    y_train = train_labels[train_indices]
    X_val = train_windows[val_indices]
    y_val = train_labels[val_indices]

    # Load test subject
    emg, segments, grouped_windows = multi_loader.load_subject(
        base_dir=base_dir,
        subject_id=test_subject,
        exercise=exercises[0],
    )
    X_test, y_test = grouped_to_arrays(grouped_windows)
    logger.info(f"Test windows: {len(X_test)}")
    
    # Get number of classes
    num_classes = len(np.unique(np.concatenate([train_labels, y_test])))
    logger.info(f"Number of classes: {num_classes}")
    
    # Create trainer and train
    trainer = DualStreamTrainer(
        train_cfg=train_cfg,
        output_dir=output_dir,
        logger=logger,
        visualizer=base_viz,
    )
    
    try:
        train_results = trainer.train(
            train_windows=X_train,
            train_labels=y_train,
            val_windows=X_val,
            val_labels=y_val,
            num_classes=num_classes,
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate(
            test_windows=X_test,
            test_labels=y_test,
            channel_mean=train_results['channel_mean'],
            channel_std=train_results['channel_std'],
            feature_mean=train_results['feature_mean'],
            feature_std=train_results['feature_std'],
        )
        
        test_acc = test_results['accuracy']
        test_f1 = test_results['f1_macro']
        
        logger.info(f"Test Results - Accuracy: {test_acc:.4f}, F1-macro: {test_f1:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "dual_stream_cnn_gru_attention",
            "approach": "dual_stream",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }
    
    # Save results
    results = {
        "test_subject": test_subject,
        "model_type": "dual_stream_cnn_gru_attention",
        "approach": "dual_stream",
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "train_history": train_results['history'],
        "best_val_acc": float(train_results['best_val_acc']),
        "best_val_f1": float(train_results['best_val_f1']),
    }
    
    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(make_json_serializable(results), f, indent=4, ensure_ascii=False)
    
    # Save metadata
    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "dual_stream_cnn_gru_attention",
        "approach": "dual_stream",
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
    del trainer, multi_loader
    gc.collect()
    
    return results


def main():
    EXPERIMENT_NAME = "exp_9_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    ALL_SUBJECTS = [
  "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]
    
    # Processing config for raw EMG
    proc_cfg = ProcessingConfig(
        window_size=500,
        window_overlap=250,
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
    
    # Training config with augmentation
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=60,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="dual_stream_cnn_gru_attention",
        pipeline_type="deep_raw",
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"="*80)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: dual_stream_cnn_gru_attention")
    print(f"Subjects: {len(ALL_SUBJECTS)} (LOSO)")
    print(f"Augmentation: noise + time_warp")
    print(f"Features: dual-stream (raw + powerful handcrafted)")
    print(f"="*80)
    
    all_loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        print(f"\n{'='*60}")
        print(f"Processing LOSO fold: test_subject={test_subject}")
        print(f"{'='*60}")
        
        try:
            fold_res = run_dual_stream_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
            )
            all_loso_results.append(fold_res)
            
            if fold_res.get('test_accuracy') is not None:
                print(f"  ✓ {test_subject}: Acc={fold_res['test_accuracy']:.4f}, F1={fold_res['test_f1_macro']:.4f}")
            else:
                print(f"  ✗ {test_subject}: {fold_res.get('error', 'Unknown error')}")
                
        except Exception as e:
            global_logger.error(f"Failed {test_subject}: {e}")
            traceback.print_exc()
            all_loso_results.append({
                "test_subject": test_subject,
                "model_type": "dual_stream_cnn_gru_attention",
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })
    
    # Aggregate results
    valid_results = [r for r in all_loso_results if r.get('test_accuracy') is not None]
    
    if valid_results:
        accs = [r['test_accuracy'] for r in valid_results]
        f1s = [r['test_f1_macro'] for r in valid_results]
        
        aggregate = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
            "num_failed": len(all_loso_results) - len(valid_results),
        }
        
        print(f"\n{'='*80}")
        print(f"AGGREGATE RESULTS:")
        print(f"  Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"  Valid folds: {aggregate['num_subjects']}/{len(ALL_SUBJECTS)}")
        print(f"{'='*80}")
    else:
        aggregate = {
            "mean_accuracy": None,
            "std_accuracy": None,
            "mean_f1_macro": None,
            "std_f1_macro": None,
            "num_subjects": 0,
            "num_failed": len(all_loso_results),
        }
        print("\nAll folds failed!")
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "dual_stream_cnn_gru_attention",
        "approach": "dual_stream",
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "noise + time_warp",
        "feature_streams": "raw_emg (CNN-GRU-Attention) + powerful_handcrafted (MLP)",
        "fusion_method": "attention-based late fusion",
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


if __name__ == "__main__":
    main()