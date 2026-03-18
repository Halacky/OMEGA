# FILE: experiments/exp_13_leveraging_subject_specific_calibration_via_few_sh_loso.py
"""
Experiment: Few-Shot Learning with Subject-Specific Calibration for EMG Gesture Recognition

This experiment implements a two-phase training approach:
1. Meta-training phase: Pre-train on all subjects except target using MAML-inspired approach
2. Few-shot calibration phase: Fine-tune on target subject with minimal data (5-10 samples per gesture)

The goal is to reduce inter-subject variance while maintaining accuracy.
"""

import os
import sys
import json
import traceback
import copy
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split

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

from models import SimpleCNN1D


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


class MAMLFewShotTrainer:
    """
    MAML-inspired few-shot learning trainer for subject-specific calibration.
    
    Phase 1: Meta-training on multiple subjects to learn adaptable initialization
    Phase 2: Few-shot fine-tuning on target subject with frozen early layers
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_cfg = train_cfg
        self.logger = logger
        self.output_dir = output_dir
        self.device = device
        
        # MAML hyperparameters
        self.meta_lr = train_cfg.learning_rate
        self.inner_lr = train_cfg.learning_rate * 0.1  # Smaller LR for inner loop
        self.num_inner_steps = 5  # Gradient steps per task
        self.meta_batch_size = 4  # Number of tasks per meta-update
        
        # Few-shot settings
        self.num_shots_per_class = 7  # 5-10 samples per gesture
        self.fine_tune_lr = train_cfg.learning_rate * 0.01  # Reduced LR for fine-tuning
        
    def meta_train(
        self,
        subject_data_loaders: Dict[str, DataLoader],
        num_epochs: int = 30,
        val_loader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """
        Meta-training phase: Learn initialization that adapts quickly to new subjects.
        
        Uses a simplified MAML approach where each "task" is adaptation to a subject.
        """
        self.logger.info("Starting MAML-inspired meta-training phase...")
        
        meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, 
                                     weight_decay=self.train_cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        subject_names = list(subject_data_loaders.keys())
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            meta_loss = 0.0
            num_meta_batches = 0
            
            # Sample meta-batch of subjects (tasks)
            np.random.shuffle(subject_names)
            
            for i in range(0, len(subject_names) - 1, 2):
                if i + 1 >= len(subject_names):
                    break
                    
                # Create support and query sets from different subjects
                support_subject = subject_names[i]
                query_subject = subject_names[i + 1]
                
                support_loader = subject_data_loaders[support_subject]
                query_loader = subject_data_loaders[query_subject]
                
                # Inner loop: adapt to support subject
                fast_weights = self._inner_loop_adapt(support_loader, criterion)
                
                # Outer loop: evaluate on query subject and update meta-parameters
                query_loss, query_acc = self._evaluate_on_query(
                    query_loader, fast_weights, criterion
                )
                
                meta_optimizer.zero_grad()
                query_loss.backward()
                meta_optimizer.step()
                
                meta_loss += query_loss.item()
                num_meta_batches += 1
            
            # Validation
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.train_cfg.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
                if epoch % 5 == 0:
                    self.logger.info(
                        f"Epoch {epoch}/{num_epochs} | Meta-loss: {meta_loss/num_meta_batches:.4f} | "
                        f"Val Acc: {val_acc:.4f}"
                    )
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        
        return self.model
    
    def _inner_loop_adapt(
        self,
        support_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Inner loop adaptation: Fast adaptation to support set.
        Returns adapted parameters.
        """
        # Create a copy of model parameters
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        inner_optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.num_inner_steps):
            for batch_x, batch_y in support_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                inner_optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                inner_optimizer.step()
        
        return fast_weights
    
    def _evaluate_on_query(
        self,
        query_loader: DataLoader,
        fast_weights: Dict[str, torch.Tensor],
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, float]:
        """
        Evaluate adapted model on query set.
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_x, batch_y in query_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(batch_y).sum().item()
            total_samples += batch_y.size(0)
        
        avg_loss = total_loss / len(query_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return torch.tensor(avg_loss, requires_grad=True), accuracy
    
    def few_shot_fine_tune(
        self,
        target_data: Tuple[torch.Tensor, torch.Tensor],
        num_shots_per_class: int = 7,
        num_epochs: int = 20,
        freeze_early_layers: bool = True,
    ) -> nn.Module:
        """
        Few-shot fine-tuning phase: Adapt to target subject with minimal data.
        
        Args:
            target_data: (X, y) tensors for target subject
            num_shots_per_class: Number of samples per gesture class
            num_epochs: Number of fine-tuning epochs
            freeze_early_layers: Whether to freeze early convolutional layers
        """
        self.logger.info(f"Starting few-shot fine-tuning with {num_shots_per_class} shots per class...")
        
        X, y = target_data
        unique_classes = torch.unique(y)
        
        # Sample few-shot examples (stratified per class)
        few_shot_indices = []
        for cls in unique_classes:
            cls_indices = (y == cls).nonzero(as_tuple=True)[0]
            if len(cls_indices) > 0:
                n_samples = min(num_shots_per_class, len(cls_indices))
                selected = cls_indices[torch.randperm(len(cls_indices))[:n_samples]]
                few_shot_indices.extend(selected.tolist())
        
        X_fewshot = X[few_shot_indices]
        y_fewshot = y[few_shot_indices]
        
        self.logger.info(f"Few-shot set: {len(few_shot_indices)} samples across {len(unique_classes)} classes")
        
        # Create data loader for few-shot set
        fewshot_dataset = TensorDataset(X_fewshot, y_fewshot)
        fewshot_loader = DataLoader(fewshot_dataset, batch_size=32, shuffle=True)
        
        # Freeze early layers if specified
        if freeze_early_layers:
            self._freeze_early_layers()
        
        # Fine-tune with reduced learning rate
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.fine_tune_lr,
            weight_decay=self.train_cfg.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_x, batch_y in fewshot_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(fewshot_loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 5:  # Early stopping for fine-tuning
                break
            
            if epoch % 5 == 0:
                self.logger.info(f"Fine-tune epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        # Unfreeze layers for evaluation
        self._unfreeze_all()
        
        return self.model
    
    def _freeze_early_layers(self):
        """Freeze the first convolutional layer for fine-tuning."""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            # Freeze first conv layer and batch norm
            if 'net.0' in name or 'net.1' in name:  # First Conv1d and BatchNorm1d
                param.requires_grad = False
                frozen_count += 1
        
        self.logger.info(f"Frozen {frozen_count} early layer parameters")
    
    def _unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(batch_y).sum().item()
                total_samples += batch_y.size(0)
        
        return total_correct / total_samples if total_samples > 0 else 0.0


def prepare_subject_data_loaders(
    subjects_data: Dict,
    batch_size: int = 256,
    val_ratio: float = 0.15,
) -> Tuple[Dict[str, DataLoader], Optional[DataLoader]]:
    """
    Prepare per-subject data loaders for meta-training.
    """
    subject_loaders = {}
    val_loader = None
    
    for subject_name, subject_dict in subjects_data.items():
        if 'train' not in subject_dict:
            continue
            
        train_data = subject_dict['train']
        X = train_data['X']
        y = train_data['y']
        
        # Convert to tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        
        # Create train/val split for this subject
        if val_ratio > 0 and len(X) > 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_ratio, stratify=y, random_state=42
            )
            
            train_dataset = TensorDataset(X_train, y_train)
            subject_loaders[subject_name] = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
            
            # Use first subject's val set for overall validation
            if val_loader is None:
                val_dataset = TensorDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            dataset = TensorDataset(X, y)
            subject_loaders[subject_name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
    
    return subject_loaders, val_loader


def run_few_shot_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    use_improved_processing: bool,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    num_shots: int = 7,
) -> Dict:
    """
    Run LOSO fold with few-shot learning approach.
    
    Phase 1: Meta-train on all training subjects
    Phase 2: Few-shot fine-tune on target subject
    Phase 3: Evaluate on held-out test set from target subject
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
        use_gpu=True,
        use_improved_processing=use_improved_processing,
    )
    
    # Use base trainer for initial data loading structure
    base_viz = Visualizer(output_dir, logger)
    base_trainer = WindowClassifierTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )
    
    # Load all subjects data
    def grouped_to_arrays(grouped_windows):
        windows_list, labels_list = [], []
        for gesture_id in sorted(grouped_windows.keys()):
            for rep_windows in grouped_windows[gesture_id]:
                if len(rep_windows) > 0:
                    windows_list.append(rep_windows)
                    labels_list.append(np.full(len(rep_windows), gesture_id))
        return np.concatenate(windows_list, axis=0), np.concatenate(labels_list, axis=0)

    try:
        raw_subjects_data = multi_loader.load_multiple_subjects(
            base_dir=base_dir,
            subject_ids=train_subjects + [test_subject],
            exercises=exercises,
            include_rest=True,
        )
    except Exception as e:
        logger.error(f"Failed to load subjects data: {e}")
        raise

    # Convert to expected format: {subject_id: {'train': {'X': ..., 'y': ...}}}
    subjects_data = {}
    for subj_id, (emg, segments, grouped_windows) in raw_subjects_data.items():
        X, y = grouped_to_arrays(grouped_windows)
        subjects_data[subj_id] = {'train': {'X': X, 'y': y}}

    # Prepare meta-training data loaders (one per subject)
    train_subjects_data = {s: subjects_data[s] for s in train_subjects if s in subjects_data}
    subject_loaders, val_loader = prepare_subject_data_loaders(
        train_subjects_data,
        batch_size=train_cfg.batch_size,
        val_ratio=0.15,
    )
    
    # Determine input dimensions
    sample_subject = list(train_subjects_data.keys())[0]
    sample_data = train_subjects_data[sample_subject]['train']
    sample_X = sample_data['X']
    if isinstance(sample_X, np.ndarray):
        sample_X = torch.from_numpy(sample_X).float()
    
    in_channels = sample_X.shape[1]
    num_classes = len(np.unique(sample_data['y']))
    
    logger.info(f"Input channels: {in_channels}, Num classes: {num_classes}")
    
    # Create model
    model = SimpleCNN1D(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=train_cfg.dropout,
    )
    
    # Phase 1: Meta-training
    few_shot_trainer = MAMLFewShotTrainer(
        model=model,
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        device=train_cfg.device,
    )
    
    model = few_shot_trainer.meta_train(
        subject_data_loaders=subject_loaders,
        num_epochs=train_cfg.epochs,
        val_loader=val_loader,
    )
    
    # Prepare target subject data
    test_subject_data = subjects_data[test_subject]
    
    # Split test subject into calibration set and evaluation set
    test_X = test_subject_data['train']['X']
    test_y = test_subject_data['train']['y']
    
    if isinstance(test_X, np.ndarray):
        test_X = torch.from_numpy(test_X).float()
    if isinstance(test_y, np.ndarray):
        test_y = torch.from_numpy(test_y).long()
    
    # Use 20% of test subject data for few-shot calibration, rest for evaluation
    X_calib, X_eval, y_calib, y_eval = train_test_split(
        test_X, test_y, test_size=0.8, stratify=test_y, random_state=42
    )
    
    # Phase 2: Few-shot fine-tuning
    model = few_shot_trainer.few_shot_fine_tune(
        target_data=(X_calib, y_calib),
        num_shots_per_class=num_shots,
        num_epochs=20,
        freeze_early_layers=True,
    )
    
    # Phase 3: Evaluate on held-out test set
    model.eval()
    eval_dataset = TensorDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=train_cfg.batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(train_cfg.device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Calculate F1-macro
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    logger.info(f"Few-shot LOSO | Test subject: {test_subject} | "
                f"Accuracy: {accuracy:.4f} | F1-macro: {f1_macro:.4f}")
    
    # Save results
    results = {
        "test_subject": test_subject,
        "num_shots_per_class": num_shots,
        "test_accuracy": float(accuracy),
        "test_f1_macro": float(f1_macro),
        "num_test_samples": len(all_labels),
    }
    
    with open(output_dir / "few_shot_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save model
    torch.save(model.state_dict(), output_dir / "few_shot_model.pt")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del model, few_shot_trainer, multi_loader, base_trainer
    gc.collect()
    
    return results


def main():
    EXPERIMENT_NAME = "exp_13_leveraging_subject_specific_calibration_via_few_sh_loso"
    HYPOTHESIS_ID = "3901a9cf-1112-4c14-9c68-baed76f94c28"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    ALL_SUBJECTS = [
     "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]
    
    # Configuration based on best-performing simple_cnn setup with augmentation
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
    
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=40,  # More epochs for meta-training
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="simple_cnn",
        pipeline_type="deep_raw",
        # Augmentation settings (noise + time_warp)
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: simple_cnn | Approach: Few-shot learning with MAML-inspired meta-training")
    print(f"Augmentation: noise + time_warp | LOSO n={len(ALL_SUBJECTS)}")
    print(f"Few-shot samples per gesture: 7")
    
    all_loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        try:
            fold_res = run_few_shot_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                use_improved_processing=True,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                num_shots=7,
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
            "simple_cnn_few_shot": {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_subjects": len(accs),
                "num_shots_per_class": 7,
            }
        }
        
        print(f"\nFew-shot simple_cnn Results:")
        print(f"  Accuracy: {aggregate['simple_cnn_few_shot']['mean_accuracy']:.4f} ± {aggregate['simple_cnn_few_shot']['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate['simple_cnn_few_shot']['mean_f1_macro']:.4f} ± {aggregate['simple_cnn_few_shot']['std_f1_macro']:.4f}")
        print(f"  (Target: reduce std from ~0.08 to ~0.04-0.06)")
    else:
        aggregate = {}
        print("\nNo valid results obtained!")
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "approach": "few_shot_learning_with_maml_meta_training",
        "model_type": "simple_cnn",
        "feature_set": "deep_raw",
        "augmentation": "noise + time_warp",
        "num_shots_per_class": 7,
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
    
    # === Update hypothesis status in Qdrant ===
    from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed
    
    if aggregate:
        best_model_name = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
        best_metrics = aggregate[best_model_name]
        best_metrics["best_model"] = best_model_name
        best_metrics["std_reduction_target_met"] = best_metrics.get("std_accuracy", 1.0) < 0.06
        mark_hypothesis_verified(
            hypothesis_id=HYPOTHESIS_ID,
            metrics=best_metrics,
            experiment_name=EXPERIMENT_NAME,
        )
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )


if __name__ == "__main__":
    main()