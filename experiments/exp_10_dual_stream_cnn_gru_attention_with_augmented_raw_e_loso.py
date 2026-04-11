# FILE: experiments/exp_10_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso.py
"""
Experiment: Dual-Stream CNN-GRU-Attention with Augmented Raw EMG and Handcrafted Feature Fusion

Hypothesis: A dual-stream architecture processing augmented raw EMG through CNN-GRU-Attention
while simultaneously processing handcrafted powerful features through a lightweight MLP,
with late-stage attention-based fusion, will outperform current best models.

Architecture:
- Stream 1: Augmented raw EMG through CNN-GRU-Attention
- Stream 2: Handcrafted powerful features through 2-layer MLP
- Late fusion: Attention mechanism for adaptive stream weighting
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

# Register the custom dual-stream model
from models import register_model
from models.dual_stream_cnn_gru_attention import DualStreamCNNGRUAttention

register_model("dual_stream_cnn_gru_attention", DualStreamCNNGRUAttention)


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


class DualStreamTrainer(WindowClassifierTrainer):
    """
    Custom trainer for dual-stream architecture that handles both raw EMG and handcrafted features.
    Extends WindowClassifierTrainer to support dual-input training.
    """
    
    def __init__(self, handcrafted_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.handcrafted_dim = handcrafted_dim
    
    def _extract_handcrafted_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract powerful handcrafted features from EMG windows.
        
        Args:
            windows: numpy array of shape (N, C, T)
        
        Returns:
            features: numpy array of shape (N, handcrafted_dim)
        """
        from processing.powerful_features import PowerfulFeatureExtractor

        extractor = PowerfulFeatureExtractor(sampling_rate=2000)

        # PowerfulFeatureExtractor.transform expects (N, T, C), windows are (N, C, T)
        windows_ntc = np.transpose(windows, (0, 2, 1))
        features_array = extractor.transform(windows_ntc)

        return features_array.astype(np.float32)
    
    def _create_model(self, in_channels: int, num_classes: int, model_type: str = "dual_stream_cnn_gru_attention") -> nn.Module:
        """Create the dual-stream model."""
        return DualStreamCNNGRUAttention(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=self.cfg.dropout,
            handcrafted_dim=self.handcrafted_dim,
            cnn_channels=[32, 64],
            gru_hidden=128,
            gru_layers=2,
        )
    
    def train(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        val_windows: np.ndarray,
        val_labels: np.ndarray,
        num_classes: int,
    ) -> Dict:
        """
        Train the dual-stream model with both raw EMG and handcrafted features.
        """
        # Extract handcrafted features for train and val sets
        self.logger.info("Extracting handcrafted features for dual-stream training...")
        train_handcrafted = self._extract_handcrafted_features(train_windows)
        val_handcrafted = self._extract_handcrafted_features(val_windows)
        
        self.logger.info(f"Train handcrafted features shape: {train_handcrafted.shape}")
        self.logger.info(f"Val handcrafted features shape: {val_handcrafted.shape}")
        
        # Update handcrafted_dim based on actual extracted features
        self.handcrafted_dim = train_handcrafted.shape[1]
        
        # Create dataset and dataloader for dual-stream
        train_dataset = DualStreamDataset(
            windows=train_windows,
            handcrafted_features=train_handcrafted,
            labels=train_labels,
            transform=self._get_train_transform(),
        )
        val_dataset = DualStreamDataset(
            windows=val_windows,
            handcrafted_features=val_handcrafted,
            labels=val_labels,
            transform=None,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        
        # Create model
        in_channels = train_windows.shape[1]
        model = self._create_model(in_channels, num_classes)
        model = model.to(self.device)
        
        self.logger.info(f"Created dual-stream model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Training setup
        criterion = nn.CrossEntropyLoss(weight=self._compute_class_weights(train_labels))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
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
            
            for batch in train_loader:
                raw_windows = batch["raw"].to(self.device)
                handcrafted = batch["handcrafted"].to(self.device)
                labels = batch["label"].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(raw_windows, handcrafted)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    raw_windows = batch["raw"].to(self.device)
                    handcrafted = batch["handcrafted"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = model(raw_windows, handcrafted)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * labels.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            scheduler.step(val_acc)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )
            
            # Early stopping and best model tracking
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save model
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "in_channels": in_channels,
            "handcrafted_dim": self.handcrafted_dim,
        }, self.output_dir / "best_model.pt")
        
        return {
            "best_val_accuracy": best_val_acc,
            "final_epoch": epoch + 1,
        }
    
    def evaluate(
        self,
        test_windows: np.ndarray,
        test_labels: np.ndarray,
        num_classes: int,
    ) -> Dict:
        """Evaluate the dual-stream model on test data."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        
        # Extract handcrafted features for test set
        self.logger.info("Extracting handcrafted features for test evaluation...")
        test_handcrafted = self._extract_handcrafted_features(test_windows)
        
        # Load best model
        checkpoint = torch.load(self.output_dir / "best_model.pt", map_location=self.device)
        in_channels = checkpoint["in_channels"]
        loaded_num_classes = checkpoint["num_classes"]
        self.handcrafted_dim = checkpoint.get("handcrafted_dim", self.handcrafted_dim)
        
        model = self._create_model(in_channels, loaded_num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        
        # Create test dataset
        test_dataset = DualStreamDataset(
            windows=test_windows,
            handcrafted_features=test_handcrafted,
            labels=test_labels,
            transform=None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        
        # Evaluate
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                raw_windows = batch["raw"].to(self.device)
                handcrafted = batch["handcrafted"].to(self.device)
                labels = batch["label"]
                
                outputs = model(raw_windows, handcrafted)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
            "f1_weighted": float(f1_score(all_labels, all_preds, average="weighted")),
            "precision_macro": float(precision_score(all_labels, all_preds, average="macro")),
            "recall_macro": float(recall_score(all_labels, all_preds, average="macro")),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }
        
        self.logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")
        
        return metrics


class DualStreamDataset(Dataset):
    """Dataset for dual-stream training with raw EMG and handcrafted features."""
    
    def __init__(self, windows, handcrafted_features, labels, transform=None):
        self.windows = torch.FloatTensor(windows)
        self.handcrafted_features = torch.FloatTensor(handcrafted_features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        raw = self.windows[idx]
        handcrafted = self.handcrafted_features[idx]
        label = self.labels[idx]
        
        if self.transform:
            raw = self.transform(raw)
        
        return {
            "raw": raw,
            "handcrafted": handcrafted,
            "label": label,
        }


def run_single_loso_fold(
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
) -> Dict:
    """
    One LOSO fold: train dual-stream model on train_subjects, test on test_subject.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = approach
    train_cfg.model_type = model_type

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

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

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=use_improved_processing,
    )

    base_viz = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    # Use custom DualStreamTrainer
    trainer = DualStreamTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        handcrafted_dim=256,  # Will be updated based on actual features
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": approach,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    # ====== EXPERIMENT SETTINGS ======
    EXPERIMENT_NAME = "exp_10_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso"
    HYPOTHESIS_ID = "0d519c5d-fa3f-44d1-b955-1ef224cc74e4"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")

    ALL_SUBJECTS = [
    "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]
    MODEL_TYPE = "dual_stream_cnn_gru_attention"
    APPROACH = "deep_raw"  # Primary pipeline is deep_raw, with handcrafted as secondary stream

    # Processing config: optimized for raw EMG
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )
    
    # Split config: by segments for better generalization
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
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,  # Handled by DualStreamTrainer
        pipeline_type=APPROACH,
        model_type=MODEL_TYPE,
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"HYPOTHESIS ID: {HYPOTHESIS_ID}")
    print(f"Model: {MODEL_TYPE} | Approach: {APPROACH}")
    print(f"LOSO n={len(ALL_SUBJECTS)} subjects")
    print(f"Augmentation: noise (std=0.02) + time_warp (max=0.1)")

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
                model_type=MODEL_TYPE,
                approach=APPROACH,
                use_improved_processing=True,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
            )
            all_loso_results.append(fold_res)
            print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, f1={fold_res['test_f1_macro']:.4f}")
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

    # Aggregate results
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    aggregate = {}
    
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
        print(f"DUAL-STREAM CNN-GRU-Attention LOSO Results:")
        print(f"  Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"  Range: [{aggregate['min_accuracy']:.4f}, {aggregate['max_accuracy']:.4f}]")
        print(f"{'='*60}")

    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "model_type": MODEL_TYPE,
        "approach": APPROACH,
        "feature_set": "dual_stream (raw + powerful handcrafted)",
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "noise + time_warp (raw stream only)",
        "architecture": {
            "stream1": "CNN-GRU-Attention on augmented raw EMG",
            "stream2": "2-layer MLP on powerful handcrafted features",
            "fusion": "Attention-based late fusion",
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
        best_metrics = dict(aggregate)
        best_metrics["best_model"] = MODEL_TYPE
        mark_hypothesis_verified(
            hypothesis_id=HYPOTHESIS_ID,
            metrics=best_metrics,
            experiment_name=EXPERIMENT_NAME,
        )
        print(f"Hypothesis {HYPOTHESIS_ID} marked as VERIFIED")
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )
        print(f"Hypothesis {HYPOTHESIS_ID} marked as FAILED")


if __name__ == "__main__":
    main()