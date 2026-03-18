# FILE: experiments/exp_7_cnn_gru_attention_with_noise_and_time_warp_augment_loso.py
"""
Experiment 7: CNN-GRU-Attention with Noise+TimeWarp Augmentation and Feature Fusion

Hypothesis: Combining the successful noise+time_warp augmentation from exp6 
with the cnn_gru_attention architecture, while adding a learnable feature 
fusion layer to incorporate handcrafted powerful features, will improve 
accuracy beyond the current best result of 0.3566.

Key elements:
1. CNN-GRU-Attention architecture (proven architecture)
2. Noise + Time Warp augmentation (from successful exp6)
3. Feature fusion with handcrafted powerful features (novel combination)
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

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# Import and register the custom fusion model
from models.cnn_gru_attention_fusion import CNNGRUAttentionFusion
from models import register_model

# Register the model so trainer can find it
register_model("cnn_gru_attention_fusion", CNNGRUAttentionFusion)


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


class FusionTrainer:
    """
    Custom trainer for the CNN-GRU-Attention Fusion model.
    Handles dual inputs: raw EMG windows and handcrafted features.
    """
    
    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        handcrafted_dim: int = 88,
    ):
        self.cfg = train_cfg
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.handcrafted_dim = handcrafted_dim
        self.device = torch.device(train_cfg.device)
        
        self.model = None
        self.best_model_state = None
        self.best_val_f1 = 0.0
        
    def _compute_handcrafted_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Compute powerful handcrafted features from EMG windows.

        Args:
            windows: (N, C, T) raw EMG windows

        Returns:
            features: (N, feature_dim) handcrafted features
        """
        from processing.powerful_features import PowerfulFeatureExtractor

        extractor = PowerfulFeatureExtractor(sampling_rate=2000)

        # PowerfulFeatureExtractor.transform expects (N, T, C), windows are (N, C, T)
        windows_ntc = np.transpose(windows, (0, 2, 1))
        features = extractor.transform(windows_ntc)

        return features.astype(np.float32)
    
    def _create_model(self, in_channels: int, num_classes: int) -> nn.Module:
        """Create the fusion model."""
        model = CNNGRUAttentionFusion(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=self.cfg.dropout,
            cnn_channels=[32, 64],
            gru_hidden=128,
            gru_layers=2,
            fusion_hidden=256,
            handcrafted_dim=self.handcrafted_dim,
        )
        return model.to(self.device)
    
    def _augment_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to batch if enabled."""
        if not self.cfg.aug_apply:
            return x
        
        x_aug = x.clone()
        
        # Noise augmentation
        if self.cfg.aug_apply_noise and self.cfg.aug_noise_std > 0:
            noise = torch.randn_like(x_aug) * self.cfg.aug_noise_std
            x_aug = x_aug + noise
        
        # Time warp augmentation
        if self.cfg.aug_apply_time_warp and self.cfg.aug_time_warp_max > 0:
            batch_size, channels, time_steps = x_aug.shape
            warp_factor = 1.0 + (torch.rand(batch_size, 1, 1, device=x.device) * 2 - 1) * self.cfg.aug_time_warp_max
            
            # Create time indices
            t_orig = torch.linspace(0, 1, time_steps, device=x.device).view(1, 1, -1)
            t_warped = t_orig * warp_factor
            t_warped = torch.clamp(t_warped, 0, 1)
            
            # Interpolate
            x_list = []
            for i in range(batch_size):
                x_interp = torch.nn.functional.interpolate(
                    x_aug[i:i+1],
                    size=time_steps,
                    mode='linear',
                    align_corners=True
                )
                x_list.append(x_interp)
            x_aug = torch.cat(x_list, dim=0)
        
        return x_aug
    
    def train(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        val_windows: np.ndarray = None,
        val_labels: np.ndarray = None,
        num_classes: int = None,
    ) -> Dict:
        """
        Train the fusion model.
        
        Args:
            train_windows: (N, C, T) training windows
            train_labels: (N,) training labels
            val_windows: (M, C, T) validation windows
            val_labels: (M,) validation labels
            num_classes: Number of classes
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Computing handcrafted features for training data...")
        train_hc_features = self._compute_handcrafted_features(train_windows)
        
        if val_windows is not None:
            self.logger.info("Computing handcrafted features for validation data...")
            val_hc_features = self._compute_handcrafted_features(val_windows)
        
        in_channels = train_windows.shape[1]
        if num_classes is None:
            num_classes = len(np.unique(train_labels))

        # Update handcrafted_dim to match actual feature count
        self.handcrafted_dim = train_hc_features.shape[1]
        self.model = self._create_model(in_channels, num_classes)
        self.logger.info(f"Created CNNGRUAttentionFusion model: in_channels={in_channels}, num_classes={num_classes}")
        
        # Compute class weights
        if self.cfg.use_class_weights:
            class_counts = np.bincount(train_labels.astype(int))
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Create data loaders
        train_x = torch.tensor(train_windows, dtype=torch.float32)
        train_hc = torch.tensor(train_hc_features, dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.long)
        
        train_dataset = TensorDataset(train_x, train_hc, train_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )
        
        if val_windows is not None:
            val_x = torch.tensor(val_windows, dtype=torch.float32)
            val_hc = torch.tensor(val_hc_features, dtype=torch.float32)
            val_y = torch.tensor(val_labels, dtype=torch.long)
            
            val_dataset = TensorDataset(val_x, val_hc, val_y)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
            )
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_f1': [], 'val_acc': []}
        
        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_hc, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_hc = batch_hc.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Apply augmentation
                batch_x = self._augment_batch(batch_x)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x, batch_hc)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_windows is not None:
                val_metrics = self.evaluate(val_loader)
                val_f1 = val_metrics['f1_macro']
                val_acc = val_metrics['accuracy']
                history['val_f1'].append(val_f1)
                history['val_acc'].append(val_acc)
                
                scheduler.step(val_f1)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.best_model_state = self.model.state_dict().copy()
                    self.best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.cfg.epochs}: "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                        f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
                    )
                
                if patience_counter >= self.cfg.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 5 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.cfg.epochs}: "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                    )
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'history': history,
            'best_val_f1': best_val_f1,
            'num_classes': num_classes,
        }
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_hc, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_hc = batch_hc.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x, batch_hc)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        # Compute F1 macro
        from sklearn.metrics import f1_score
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'predictions': all_preds,
            'labels': all_labels,
        }
    
    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Predict labels for windows."""
        self.model.eval()
        
        # Compute handcrafted features
        hc_features = self._compute_handcrafted_features(windows)
        
        x = torch.tensor(windows, dtype=torch.float32).to(self.device)
        hc = torch.tensor(hc_features, dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(x, hc)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch_x, batch_hc in loader:
                outputs = self.model(batch_x, batch_hc)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
        
        return np.array(all_preds)
    
    def save_model(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.cfg),
            'best_val_f1': self.best_val_f1,
        }, path)


def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercise: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    handcrafted_dim: int = 88,
) -> Dict:
    """
    Run single LOSO fold with CNN-GRU-Attention Fusion model.
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
        exercises=exercise,
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
    
    # Helper: convert grouped_windows to flat arrays
    def grouped_to_arrays(grouped_windows):
        windows_list, labels_list = [], []
        for gesture_id in sorted(grouped_windows.keys()):
            for rep_windows in grouped_windows[gesture_id]:
                if len(rep_windows) > 0:
                    windows_list.append(rep_windows)
                    labels_list.append(np.full(len(rep_windows), gesture_id))
        return np.concatenate(windows_list, axis=0), np.concatenate(labels_list, axis=0)

    # Load subjects data
    subjects_windows = {}
    subjects_labels = {}
    for subject_id in train_subjects + [test_subject]:
        try:
            emg, segments, grouped_windows = multi_loader.load_subject(
                base_dir=base_dir,
                subject_id=subject_id,
                exercise=exercise[0],
            )
            w, l = grouped_to_arrays(grouped_windows)
            subjects_windows[subject_id] = w
            subjects_labels[subject_id] = l
        except Exception as e:
            logger.error(f"Failed to load {subject_id}: {e}")
            raise

    # Pool all training data and split train/val
    all_train_windows = np.concatenate([subjects_windows[s] for s in train_subjects], axis=0)
    all_train_labels = np.concatenate([subjects_labels[s] for s in train_subjects], axis=0)

    # Shuffle and split train/val
    n_total = len(all_train_labels)
    indices = np.arange(n_total)
    np.random.seed(train_cfg.seed)
    np.random.shuffle(indices)
    val_size = int(n_total * split_cfg.val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_windows = all_train_windows[train_idx]
    train_labels = all_train_labels[train_idx]

    if val_size > 0:
        val_windows = all_train_windows[val_idx]
        val_labels = all_train_labels[val_idx]
    else:
        val_windows = None
        val_labels = None

    # Prepare test data
    test_windows = subjects_windows[test_subject]
    test_labels = subjects_labels[test_subject]
    
    # Determine number of classes
    all_labels = np.concatenate([train_labels, test_labels])
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    
    train_labels = np.array([label_map[l] for l in train_labels])
    if val_labels is not None:
        val_labels = np.array([label_map[l] for l in val_labels])
    test_labels = np.array([label_map[l] for l in test_labels])
    
    logger.info(f"Train: {train_windows.shape}, Val: {val_windows.shape if val_windows is not None else 'None'}, Test: {test_windows.shape}")
    logger.info(f"Num classes: {num_classes}")
    
    # Create trainer
    trainer = FusionTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        handcrafted_dim=handcrafted_dim,
    )
    
    try:
        # Train
        train_results = trainer.train(
            train_windows=train_windows,
            train_labels=train_labels,
            val_windows=val_windows,
            val_labels=val_labels,
            num_classes=num_classes,
        )
        
        # Evaluate on test
        test_preds = trainer.predict(test_windows)
        test_acc = (test_preds == test_labels).mean()
        
        from sklearn.metrics import f1_score
        test_f1 = f1_score(test_labels, test_preds, average='macro')
        
        # Save model
        trainer.save_model(output_dir / "best_model.pt")
        
    except Exception as e:
        logger.error(f"Error in LOSO fold: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "cnn_gru_attention_fusion",
            "approach": "fusion",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }
    
    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: cnn_gru_attention_fusion | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )
    
    # Save results
    results = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "cnn_gru_attention_fusion",
        "approach": "fusion",
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "num_classes": num_classes,
        "best_val_f1": float(train_results['best_val_f1']),
    }
    
    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(make_json_serializable(results), f, indent=4)
    
    # Save metadata
    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "cnn_gru_attention_fusion",
        "approach": "fusion_with_augmentation",
        "exercises": exercise,
        "augmentation": "noise+time_warp",
        "config": {
            "processing": asdict(proc_cfg),
            "split": asdict(split_cfg),
            "training": asdict(train_cfg),
        },
        "metrics": {
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1),
            "best_val_f1": float(train_results['best_val_f1']),
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
    EXPERIMENT_NAME = "exp_7_cnn_gru_attention_with_noise_and_time_warp_augment_loso"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    ALL_SUBJECTS = [
        "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]
    
    # Processing config: same as exp6 for fair comparison
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
    
    # Training config with augmentation enabled
    train_cfg = TrainingConfig(
        batch_size=256,  # Smaller batch for fusion model
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,  # Increased dropout for regularization
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="cnn_gru_attention_fusion",
        pipeline_type="fusion",
        # Augmentation settings (from exp6)
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: cnn_gru_attention_fusion (custom fusion model)")
    print(f"Augmentation: noise (std=0.02) + time_warp (max=0.1)")
    print(f"Feature Fusion: raw + powerful handcrafted features")
    print(f"LOSO: n={len(ALL_SUBJECTS)} subjects")
    print("=" * 60)
    
    all_loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        try:
            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercise=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                handcrafted_dim=88,  # Powerful feature set dimension
            )
            all_loso_results.append(fold_res)
            
            if fold_res.get("test_accuracy") is not None:
                print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, f1={fold_res['test_f1_macro']:.4f}")
            else:
                print(f"  ✗ {test_subject}: ERROR - {fold_res.get('error', 'unknown')}")
                
        except Exception as e:
            global_logger.error(f"Failed {test_subject}: {e}")
            traceback.print_exc()
            all_loso_results.append({
                "test_subject": test_subject,
                "model_type": "cnn_gru_attention_fusion",
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
            "num_subjects": len(accs),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
        }
        
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS:")
        print(f"Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"F1-macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"Range: [{aggregate['min_accuracy']:.4f}, {aggregate['max_accuracy']:.4f}]")
        print(f"Successful folds: {len(valid_results)}/{len(ALL_SUBJECTS)}")
    else:
        aggregate = {"error": "No valid results"}
        print("\nERROR: No successful folds!")
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": "test-001",
        "model_type": "cnn_gru_attention_fusion",
        "approach": "fusion_with_augmentation",
        "augmentation": "noise+time_warp",
        "feature_fusion": "raw_cnn_gru_attention + powerful_handcrafted",
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
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