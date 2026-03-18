# FILE: experiments/exp_20_augmented_cnn_gru_attention_with_feature_fusion_fo_loso.py
"""
Experiment: Augmented CNN-GRU-Attention with Feature Fusion for EMG Gesture Recognition

Hypothesis: Applying noise+time_warp augmentation to cnn_gru_attention while adding
a learnable feature fusion layer to incorporate handcrafted powerful features will
improve accuracy beyond the current best result of 0.3566.

Key elements:
1. CNN-GRU-Attention architecture for raw EMG processing
2. Noise + time_warp augmentation (from exp6)
3. Learnable fusion layer for handcrafted features
4. Dropout increased to 0.3 for regularization
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
from torch.utils.data import Dataset, DataLoader

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver
from models import register_model
from models.fusion_cnn_gru_attention import FusionCNNGRUAttention

# Register the custom model
register_model("fusion_cnn_gru_attention", FusionCNNGRUAttention)


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


class FusionDataset(Dataset):
    """Dataset that provides both raw windows and handcrafted features."""
    
    def __init__(
        self, 
        windows: np.ndarray, 
        labels: np.ndarray, 
        handcrafted_features: np.ndarray = None,
        augment: bool = False,
        noise_std: float = 0.02,
        time_warp_max: float = 0.1,
    ):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_std = noise_std
        self.time_warp_max = time_warp_max
        
        if handcrafted_features is not None:
            self.handcrafted = torch.FloatTensor(handcrafted_features)
        else:
            self.handcrafted = None
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.labels[idx]
        hc = self.handcrafted[idx] if self.handcrafted is not None else torch.zeros(1)
        
        if self.augment:
            # Apply noise augmentation
            noise = torch.randn_like(window) * self.noise_std
            window = window + noise
            
            # Apply time warp augmentation
            if torch.rand(1).item() < 0.5:
                seq_len = window.size(1)
                warp_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.time_warp_max
                new_len = int(seq_len * warp_factor)
                new_len = max(10, min(seq_len * 2, new_len))
                
                # Interpolate
                window_np = window.numpy()
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, seq_len)
                x_new = np.linspace(0, 1, new_len)
                
                warped = np.zeros_like(window_np)
                for c in range(window_np.shape[0]):
                    f = interp1d(x_old, window_np[c], kind='linear', fill_value='extrapolate')
                    warped_channel = f(x_new)
                    # Resize back to original length
                    if new_len != seq_len:
                        f2 = interp1d(np.linspace(0, 1, new_len), warped_channel, kind='linear', fill_value='extrapolate')
                        warped[c] = f2(x_old)
                    else:
                        warped[c] = warped_channel
                window = torch.FloatTensor(warped)
        
        return window, hc, label


class FusionTrainer:
    """Custom trainer for the fusion model that handles both raw and handcrafted features."""
    
    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        visualizer,
        handcrafted_dim: int = 128,
    ):
        self.cfg = train_cfg
        self.logger = logger
        self.output_dir = output_dir
        self.visualizer = visualizer
        self.handcrafted_dim = handcrafted_dim
        self.device = torch.device(train_cfg.device)
        
    def _extract_handcrafted_features(self, windows: np.ndarray) -> np.ndarray:
        """Extract powerful handcrafted features from EMG windows."""
        # windows: (N, C, T)
        n_samples, n_channels, window_len = windows.shape
        features = []
        
        for ch in range(n_channels):
            ch_data = windows[:, ch, :]  # (N, T)
            
            # Time-domain features
            mav = np.mean(np.abs(ch_data), axis=1)
            var = np.var(ch_data, axis=1)
            std = np.std(ch_data, axis=1)
            rms = np.sqrt(np.mean(ch_data**2, axis=1))
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(ch_data), axis=1) != 0, axis=1)
            zc = zero_crossings / window_len
            
            # Waveform length
            wl = np.sum(np.abs(np.diff(ch_data, axis=1)), axis=1)
            
            # Slope sign changes
            ssc = np.sum(
                np.logical_and(
                    np.diff(ch_data, axis=1) * np.diff(ch_data[:, ::-1], axis=1)[:, ::-1] < 0,
                    np.abs(np.diff(ch_data, axis=1)) > 1e-6
                ),
                axis=1
            )
            
            # Willison amplitude
            wa = np.sum(np.abs(np.diff(ch_data, axis=1)) > 0.01, axis=1)
            
            # Higher-order statistics
            skew = np.apply_along_axis(lambda x: np.mean((x - np.mean(x))**3) / (np.std(x)**3 + 1e-8), 1, ch_data)
            kurt = np.apply_along_axis(lambda x: np.mean((x - np.mean(x))**4) / (np.std(x)**4 + 1e-8) - 3, 1, ch_data)
            
            # Frequency-domain features (approximate)
            from scipy import signal as sig
            freq_features = []
            for i in range(n_samples):
                f, psd = sig.welch(ch_data[i], fs=2000, nperseg=min(64, window_len))
                if len(psd) > 0:
                    mnf = np.sum(f * psd) / (np.sum(psd) + 1e-8)  # Mean frequency
                    mdf_f = np.cumsum(psd)
                    mdf_f = mdf_f / (mdf_f[-1] + 1e-8)
                    mdf_idx = np.searchsorted(mdf_f, 0.5)
                    mdf = f[min(mdf_idx, len(f)-1)]  # Median frequency
                    freq_features.append([mnf, mdf])
                else:
                    freq_features.append([0.0, 0.0])
            freq_features = np.array(freq_features)
            
            ch_features = np.column_stack([mav, var, std, rms, zc, wl, ssc, wa, skew, kurt, freq_features])
            features.append(ch_features)
        
        features = np.hstack(features)  # (N, C * 12)
        return features
    
    def train(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        val_windows: np.ndarray,
        val_labels: np.ndarray,
        num_classes: int,
    ):
        """Train the fusion model."""
        
        # Extract handcrafted features
        self.logger.info("Extracting handcrafted features for training data...")
        train_hc = self._extract_handcrafted_features(train_windows)
        val_hc = self._extract_handcrafted_features(val_windows)
        
        self.handcrafted_dim = train_hc.shape[1]
        self.logger.info(f"Handcrafted feature dimension: {self.handcrafted_dim}")
        
        # Standardize handcrafted features
        hc_mean = train_hc.mean(axis=0)
        hc_std = train_hc.std(axis=0) + 1e-8
        train_hc = (train_hc - hc_mean) / hc_std
        val_hc = (val_hc - hc_mean) / hc_std
        
        # Create datasets
        train_dataset = FusionDataset(
            train_windows, train_labels, train_hc,
            augment=self.cfg.aug_apply,
            noise_std=self.cfg.aug_noise_std,
            time_warp_max=self.cfg.aug_time_warp_max,
        )
        val_dataset = FusionDataset(val_windows, val_labels, val_hc)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )
        
        # Create model
        in_channels = train_windows.shape[1]
        self.model = FusionCNNGRUAttention(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=self.cfg.dropout,
            handcrafted_dim=self.handcrafted_dim,
        ).to(self.device)
        
        # Compute class weights
        if self.cfg.use_class_weights:
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training loop
        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.cfg.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for windows, hc, labels in train_loader:
                windows = windows.to(self.device)
                hc = hc.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(windows, hc)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * windows.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for windows, hc, labels in val_loader:
                    windows = windows.to(self.device)
                    hc = hc.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(windows, hc)
                    _, predicted = outputs.max(1)
                    
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_acc = val_correct / val_total
            
            # Compute F1
            from sklearn.metrics import f1_score
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            
            scheduler.step(val_acc)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} F1: {val_f1:.4f}"
            )
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            "val_accuracy": best_val_acc,
            "val_f1_macro": best_val_f1,
            "best_epoch": epoch - patience_counter + 1,
        }
    
    def predict(self, test_windows: np.ndarray) -> np.ndarray:
        """Predict on test data."""
        self.model.eval()
        
        # Extract handcrafted features for test data
        test_hc = self._extract_handcrafted_features(test_windows)
        test_hc = (test_hc - test_hc.mean(axis=0)) / (test_hc.std(axis=0) + 1e-8)
        
        test_dataset = FusionDataset(test_windows, np.zeros(len(test_windows)), test_hc)
        test_loader = DataLoader(
            test_dataset, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        
        all_preds = []
        with torch.no_grad():
            for windows, hc, _ in test_loader:
                windows = windows.to(self.device)
                hc = hc.to(self.device)
                outputs = self.model(windows, hc)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
        
        return np.array(all_preds)
    
    def evaluate(self, test_windows: np.ndarray, test_labels: np.ndarray) -> Dict:
        """Evaluate on test data."""
        predictions = self.predict(test_windows)
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(test_labels, predictions)
        f1_macro = f1_score(test_labels, predictions, average='macro')
        precision_macro = precision_score(test_labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(test_labels, predictions, average='macro', zero_division=0)
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "predictions": predictions,
        }


def run_single_loso_fold(
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
    Single LOSO fold for the fusion model.
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
        use_improved_processing=True,
    )
    
    # Load subjects data
    def grouped_to_arrays(grouped_windows):
        windows_list, labels_list = [], []
        for gesture_id in sorted(grouped_windows.keys()):
            for rep_windows in grouped_windows[gesture_id]:
                if len(rep_windows) > 0:
                    windows_list.append(rep_windows)
                    labels_list.append(np.full(len(rep_windows), gesture_id))
        return np.concatenate(windows_list, axis=0), np.concatenate(labels_list, axis=0)

    raw_subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=train_subjects + [test_subject],
        exercises=exercises,
        include_rest=True,
    )

    # Get train data
    train_windows_list = []
    train_labels_list = []
    for subj_id, (emg, segments, grouped_windows) in raw_subjects_data.items():
        if subj_id in train_subjects:
            w, l = grouped_to_arrays(grouped_windows)
            train_windows_list.append(w)
            train_labels_list.append(l)

    train_windows = np.concatenate(train_windows_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    # Get test data
    test_emg, test_segments, test_grouped = raw_subjects_data[test_subject]
    test_windows, test_labels = grouped_to_arrays(test_grouped)
    
    # Create validation split from training data
    n_train = len(train_windows)
    indices = np.random.permutation(n_train)
    val_size = int(n_train * split_cfg.val_ratio)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    val_windows = train_windows[val_indices]
    val_labels = train_labels[val_indices]
    train_windows_final = train_windows[train_indices]
    train_labels_final = train_labels[train_indices]
    
    # Normalize data
    train_mean = train_windows_final.mean(axis=(0, 2), keepdims=True)
    train_std = train_windows_final.std(axis=(0, 2), keepdims=True) + 1e-8
    
    train_windows_final = (train_windows_final - train_mean) / train_std
    val_windows = (val_windows - train_mean) / train_std
    test_windows = (test_windows - train_mean) / train_std
    
    # Get number of classes
    num_classes = len(np.unique(train_labels))
    
    # Create trainer
    base_viz = Visualizer(output_dir, logger)
    trainer = FusionTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        handcrafted_dim=128,
    )
    
    try:
        # Train
        train_results = trainer.train(
            train_windows_final, train_labels_final,
            val_windows, val_labels,
            num_classes,
        )
        
        # Evaluate on test
        test_metrics = trainer.evaluate(test_windows, test_labels)
        
        test_acc = float(test_metrics["accuracy"])
        test_f1 = float(test_metrics["f1_macro"])
        
    except Exception as e:
        logger.error(f"Error in training: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "fusion_cnn_gru_attention",
            "approach": "hybrid_fusion",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }
    
    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: fusion_cnn_gru_attention | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )
    
    # Save results
    results = {
        "test_subject": test_subject,
        "model_type": "fusion_cnn_gru_attention",
        "approach": "hybrid_fusion",
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "train_results": train_results,
    }
    
    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(make_json_serializable(results), f, indent=4)
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    del trainer, multi_loader
    gc.collect()
    
    return {
        "test_subject": test_subject,
        "model_type": "fusion_cnn_gru_attention",
        "approach": "hybrid_fusion",
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    EXPERIMENT_NAME = "exp_20_augmented_cnn_gru_attention_with_feature_fusion_fo_loso"
    HYPOTHESIS_ID = "e2f39b15-6605-42fc-ae5c-dcc0a0153653"
    
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

    # Processing config matching exp6 successful settings
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
    
    # Training config with augmentation from exp6 + increased dropout
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=60,  # Slightly more epochs for fusion model
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,  # Increased from default
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="fusion_cnn_gru_attention",
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
    print(f"Hypothesis: CNN-GRU-Attention with feature fusion + augmentation")
    print(f"Subjects: {len(ALL_SUBJECTS)} | LOSO evaluation")
    print(f"Augmentation: noise (std=0.02) + time_warp (max=0.1)")
    print(f"Dropout: 0.3")
    
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
                exercises=EXERCISES,
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
            global_logger.error(f"Failed {test_subject}: {e}")
            traceback.print_exc()
            all_loso_results.append({
                "test_subject": test_subject,
                "model_type": "fusion_cnn_gru_attention",
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
        
        print(f"\n{'='*60}")
        print(f"RESULTS: Acc = {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"         F1  = {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"{'='*60}")
    else:
        aggregate = {}
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "model_type": "fusion_cnn_gru_attention",
        "approach": "hybrid_fusion_with_augmentation",
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "noise + time_warp",
        "dropout": 0.3,
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
        best_metrics = {
            "mean_accuracy": aggregate["mean_accuracy"],
            "std_accuracy": aggregate["std_accuracy"],
            "mean_f1_macro": aggregate["mean_f1_macro"],
            "std_f1_macro": aggregate["std_f1_macro"],
            "num_subjects": aggregate["num_subjects"],
            "best_model": "fusion_cnn_gru_attention",
        }
        mark_hypothesis_verified(
            hypothesis_id=HYPOTHESIS_ID,
            metrics=best_metrics,
            experiment_name=EXPERIMENT_NAME,
        )
        print(f"✓ Hypothesis {HYPOTHESIS_ID} marked as verified in Qdrant")
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )
        print(f"✗ Hypothesis {HYPOTHESIS_ID} marked as failed in Qdrant")


if __name__ == "__main__":
    main()