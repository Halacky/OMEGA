# FILE: experiments/exp_48_invariant_risk_minimization_for_causal_gesture_fea_loso.py
"""
Experiment: Invariant Risk Minimization for Causal Gesture Features

Hypothesis: Adding IRM regularization on top of content-style disentanglement 
will select only causally gesture-related features, cutting spurious correlations
that work on some subjects but not others.

Key mechanism:
- Each training subject = separate environment
- IRM penalty: ||grad_{w=1} R_e(Phi * w)||^2
- Penalty weight annealed: 0 for first 20% epochs, linear ramp to lambda_irm=1.0
- Combined loss: L_gesture + lambda_irm * L_irm

Expected effect: Accuracy improvement from 38.86% to 40-43%
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

# Add repo root to sys.path
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

# Import and register the IRM model
from models.irm_content_style_emg import IRMContentStyleEMG, compute_irm_penalty
from models import register_model

# Register model for use in trainer factory
register_model("irm_content_style_emg", IRMContentStyleEMG)

# Subject handling
from experiments.exp_X_template_loso import (
    parse_subjects_args,
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)


def grouped_to_arrays(grouped_windows):
    """Convert grouped_windows dict to flat (windows, labels) arrays."""
    windows_list, labels_list = [], []
    for gesture_id in sorted(grouped_windows.keys()):
        for rep_array in grouped_windows[gesture_id]:
            windows_list.append(rep_array)
            labels_list.append(np.full(len(rep_array), gesture_id, dtype=np.int64))
    return np.concatenate(windows_list, axis=0), np.concatenate(labels_list, axis=0)


class IRMTrainer(WindowClassifierTrainer):
    """
    Custom trainer for IRM-based training with per-environment penalty computation.
    
    Each training subject is treated as a separate environment.
    IRM penalty is computed per-environment and averaged.
    Penalty weight is annealed during training.
    """
    
    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        visualizer,
        lambda_irm: float = 1.0,
        irm_warmup_epochs: int = 10,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.lambda_irm = lambda_irm
        self.irm_warmup_epochs = irm_warmup_epochs
        self.class_ids = None
    
    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        num_classes = len(self.class_ids)
        class_counts = np.bincount(labels, minlength=num_classes)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * num_classes
        return torch.tensor(weights, dtype=torch.float32, device=self.cfg.device)

    def _get_irm_weight(self, epoch: int, total_epochs: int) -> float:
        """
        Compute IRM weight with annealing schedule.
        - 0 for first 20% of training (warmup)
        - Linear ramp to lambda_irm over remaining epochs
        """
        warmup_epochs = int(0.2 * total_epochs)
        if epoch < warmup_epochs:
            return 0.0
        # Linear ramp from 0 to lambda_irm
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        return self.lambda_irm * min(progress, 1.0)
    
    def fit(self, splits: Dict) -> Dict:
        """
        Train model with IRM penalty on per-subject environments.
        
        Args:
            splits: Dict with 'train', 'val', 'test' windows and labels
                    Also expects 'train_subjects_data' for environment split
        
        Returns:
            Training results dict
        """
        # Extract standard splits
        train_windows = splits["train_windows"]
        train_labels = splits["train_labels"]
        val_windows = splits["val_windows"]
        val_labels = splits["val_labels"]
        test_windows = splits.get("test_windows")
        test_labels = splits.get("test_labels")
        
        # Get per-subject environment indices if available
        train_subject_indices = splits.get("train_subject_indices", None)
        
        # Infer input dimensions
        in_channels = train_windows.shape[2]  # (N, T, C)
        num_classes = len(np.unique(train_labels))
        self.class_ids = list(range(num_classes))
        
        self.logger.info(f"IRM Training: in_channels={in_channels}, num_classes={num_classes}")
        self.logger.info(f"IRM lambda={self.lambda_irm}, warmup={self.irm_warmup_epochs} epochs")
        
        # Create model
        self.model = self._create_model(in_channels, num_classes, self.cfg.model_type)
        self.model = self.model.to(self.cfg.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(
            weight=self._compute_class_weights(train_labels) if self.cfg.use_class_weights else None
        )
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
        
        # Prepare data loaders
        # Transpose windows from (N, T, C) to (N, C, T) for model
        train_X = torch.tensor(train_windows.transpose(0, 2, 1), dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.long)
        val_X = torch.tensor(val_windows.transpose(0, 2, 1), dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.long)
        
        # Create per-environment data for IRM
        if train_subject_indices is not None:
            # We have per-subject indices, create environment-specific loaders
            environments = self._create_environments(
                train_X, train_y, train_subject_indices
            )
            self.logger.info(f"Created {len(environments)} environments for IRM")
        else:
            # No environment info, treat all as single environment
            environments = [(train_X, train_y)]
            self.logger.warning("No subject indices provided, using single environment")
        
        # Main training loader
        train_dataset = TensorDataset(train_X, train_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = DataLoader(
            TensorDataset(val_X, val_y),
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        
        # Training loop
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            irm_weight = self._get_irm_weight(epoch, self.cfg.epochs)
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.cfg.device)
                batch_y = batch_y.to(self.cfg.device)
                
                optimizer.zero_grad()
                
                # Forward pass with content features
                if hasattr(self.model, 'forward_with_content'):
                    logits, content = self.model.forward_with_content(batch_X)
                else:
                    logits = self.model(batch_X)
                    content = None
                
                # Standard classification loss
                loss_cls = criterion(logits, batch_y)
                loss = loss_cls
                
                # IRM penalty (if weight > 0)
                if irm_weight > 0 and len(environments) > 1:
                    loss_irm = self._compute_irm_penalty_over_environments(
                        environments, criterion, len(train_X)
                    )
                    loss = loss + irm_weight * loss_irm
                    
                    if batch_idx == 0 and epoch % 5 == 0:
                        self.logger.debug(
                            f"Epoch {epoch}: L_cls={loss_cls.item():.4f}, "
                            f"L_irm={loss_irm.item():.4f}, weight={irm_weight:.3f}"
                        )
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_acc = train_correct / train_total
            
            # Validation
            val_acc, val_f1, val_loss = self._evaluate(val_loader, criterion)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
            else:
                patience_counter += 1
            
            if epoch % 5 == 0 or epoch == self.cfg.epochs - 1:
                self.logger.info(
                    f"Epoch {epoch}/{self.cfg.epochs}: "
                    f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
                    f"val_f1={val_f1:.4f}, irm_weight={irm_weight:.3f}"
                )
            
            if patience_counter >= self.cfg.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if (self.output_dir / "best_model.pt").exists():
            self.model.load_state_dict(torch.load(self.output_dir / "best_model.pt"))
        
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        # Final evaluation on all splits
        results = {
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
        }
        
        if test_windows is not None:
            test_X = torch.tensor(test_windows.transpose(0, 2, 1), dtype=torch.float32)
            test_y = torch.tensor(test_labels, dtype=torch.long)
            test_loader = DataLoader(
                TensorDataset(test_X, test_y),
                batch_size=self.cfg.batch_size,
                shuffle=False,
            )
            test_acc, test_f1, _ = self._evaluate(test_loader, criterion)
            results["test_accuracy"] = test_acc
            results["test_f1_macro"] = test_f1
        
        return results
    
    def _create_environments(
        self,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        subject_indices: Dict[str, List[int]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create per-subject environment data for IRM."""
        environments = []
        for subj_id, indices in subject_indices.items():
            if len(indices) > 0:
                env_X = train_X[indices]
                env_y = train_y[indices]
                environments.append((env_X, env_y))
        return environments
    
    def _compute_irm_penalty_over_environments(
        self,
        environments: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        total_samples: int,
    ) -> torch.Tensor:
        """
        Compute IRM penalty averaged over all environments.
        
        For each environment, we compute the IRM penalty and then
        weight by the environment size relative to total samples.
        """
        total_penalty = 0.0
        
        for env_X, env_y in environments:
            env_X = env_X.to(self.cfg.device)
            env_y = env_y.to(self.cfg.device)
            
            # Forward pass
            if hasattr(self.model, 'forward_with_content'):
                logits, _ = self.model.forward_with_content(env_X)
            else:
                logits = self.model(env_X)
            
            # Compute IRM penalty for this environment
            penalty = compute_irm_penalty(logits, env_y, criterion)
            total_penalty = total_penalty + penalty * env_X.size(0)
        
        # Weighted average
        return total_penalty / total_samples
    
    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.cfg.device)
                batch_y = batch_y.to(self.cfg.device)
                
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = (all_preds == all_labels).mean()
        f1 = self._compute_f1_macro(all_labels, all_preds)
        
        return acc, f1, total_loss / len(loader)
    
    def _compute_f1_macro(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute macro F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "test",
        visualize: bool = True,
    ) -> Dict:
        """Evaluate model on numpy arrays."""
        self.model.eval()
        
        # Transpose to (N, C, T)
        X_tensor = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        X_tensor = X_tensor.to(self.cfg.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        
        acc = (preds == y).mean()
        f1 = self._compute_f1_macro(y, preds)
        
        return {
            "accuracy": acc,
            "f1_macro": f1,
            "predictions": preds,
        }


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
    lambda_irm: float = 1.0,
) -> Dict:
    """
    Run a single LOSO fold with IRM training.
    
    Loads each training subject separately to create environments for IRM.
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
    
    # Load data
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=True,
    )
    
    # Load all subjects data
    all_subjects = train_subjects + [test_subject]
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subjects,
        exercises=exercises,
        include_rest=False,
    )
    
    # Get common gestures
    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    gesture_to_class = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    num_classes = len(gesture_to_class)
    logger.info(f"Common gestures: {common_gestures}, num_classes={num_classes}")
    
    # Prepare per-subject data
    train_windows_list = []
    train_labels_list = []
    train_subject_indices = {s: [] for s in train_subjects}
    
    val_windows_list = []
    val_labels_list = []
    
    test_windows_list = []
    test_labels_list = []
    
    current_idx = 0
    
    for subj_id in train_subjects:
        if subj_id not in subjects_data:
            logger.warning(f"Subject {subj_id} not found in loaded data")
            continue
        
        emg, segments, grouped_windows = subjects_data[subj_id]
        windows, labels = grouped_to_arrays(grouped_windows)
        
        # Filter to common gestures
        mask = np.array([l in gesture_to_class for l in labels])
        windows = windows[mask]
        labels = np.array([gesture_to_class[l] for l in labels[mask]])
        
        # Split this subject's data into train/val
        n_samples = len(windows)
        n_val = int(n_samples * cs_cfg.val_ratio)
        n_train = n_samples - n_val
        
        # Shuffle indices
        rng = np.random.default_rng(train_cfg.seed)
        indices = rng.permutation(n_samples)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        # Track subject indices for IRM environments
        start_idx = current_idx
        train_windows_list.append(windows[train_idx])
        train_labels_list.append(labels[train_idx])
        train_subject_indices[subj_id] = list(range(start_idx, start_idx + len(train_idx)))
        current_idx += len(train_idx)
        
        val_windows_list.append(windows[val_idx])
        val_labels_list.append(labels[val_idx])
    
    # Load test subject data
    if test_subject in subjects_data:
        emg, segments, grouped_windows = subjects_data[test_subject]
        windows, labels = grouped_to_arrays(grouped_windows)
        mask = np.array([l in gesture_to_class for l in labels])
        test_windows_list.append(windows[mask])
        test_labels_list.append(np.array([gesture_to_class[l] for l in labels[mask]]))
    
    # Concatenate all
    train_windows = np.concatenate(train_windows_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    val_windows = np.concatenate(val_windows_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    
    if test_windows_list:
        test_windows = np.concatenate(test_windows_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
    else:
        test_windows = None
        test_labels = None
    
    logger.info(
        f"Data shapes: train={train_windows.shape}, val={val_windows.shape}, "
        f"test={test_windows.shape if test_windows is not None else None}"
    )
    logger.info(f"Training environments (subjects): {len(train_subject_indices)}")
    
    # Create trainer
    base_viz = Visualizer(output_dir, logger)
    
    trainer = IRMTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        lambda_irm=lambda_irm,
        irm_warmup_epochs=int(0.2 * train_cfg.epochs),
    )
    
    # Prepare splits with environment info
    splits = {
        "train_windows": train_windows,
        "train_labels": train_labels,
        "val_windows": val_windows,
        "val_labels": val_labels,
        "test_windows": test_windows,
        "test_labels": test_labels,
        "train_subject_indices": train_subject_indices,
    }
    
    try:
        results = trainer.fit(splits)
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
    
    # Final evaluation on test set
    if test_windows is not None:
        test_metrics = trainer.evaluate_numpy(test_windows, test_labels, "test")
        test_acc = float(test_metrics.get("accuracy", 0.0))
        test_f1 = float(test_metrics.get("f1_macro", 0.0))
    else:
        test_acc = None
        test_f1 = None
    
    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: {model_type} | IRM | "
        f"Accuracy={f'{test_acc:.4f}' if test_acc is not None else 'N/A'}, "
        f"F1-macro={f'{test_f1:.4f}' if test_f1 is not None else 'N/A'}"
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
        "approach": "deep_raw_irm",
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
        "irm_settings": {
            "lambda_irm": lambda_irm,
            "warmup_ratio": 0.2,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del trainer, multi_loader, base_viz
    gc.collect()
    
    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    # ========== Experiment Configuration ==========
    EXPERIMENT_NAME = "exp_48_invariant_risk_minimization_for_causal_gesture_fea_loso"
    HYPOTHESIS_ID = "h-048-irm-regularization"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    
    # Subject list (default to CI subjects, override with --subjects or --full)
    ALL_SUBJECTS = parse_subjects_args()
    EXERCISES = ["E1"]
    MODEL_TYPES = ["irm_content_style_emg"]
    
    # Processing config (same as content-style disentanglement baseline)
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
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
    
    # Training config with augmentation
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=60,  # Longer training for IRM to converge
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="irm_content_style_emg",
        pipeline_type="deep_raw",
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )
    
    # IRM settings
    LAMBDA_IRM = 1.0
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"HYPOTHESIS: {HYPOTHESIS_ID}")
    print(f"Model: {MODEL_TYPES}")
    print(f"Subjects: {len(ALL_SUBJECTS)} (CI: {ALL_SUBJECTS == CI_TEST_SUBJECTS})")
    print(f"IRM: lambda={LAMBDA_IRM}, warmup=20%")
    print(f"Augmentation: noise + time_warp")
    
    # Run LOSO
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
                    lambda_irm=LAMBDA_IRM,
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
        }
        
        print(
            f"\n{model_type}: Acc = {aggregate[model_type]['mean_accuracy']:.4f} "
            f"± {aggregate[model_type]['std_accuracy']:.4f}, "
            f"F1 = {aggregate[model_type]['mean_f1_macro']:.4f} "
            f"± {aggregate[model_type]['std_f1_macro']:.4f}"
        )
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "noise + time_warp",
        "irm_settings": {
            "lambda_irm": LAMBDA_IRM,
            "warmup_ratio": 0.2,
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
    
    # ========== Update hypothesis status in Qdrant ==========
    from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed
    
    if aggregate:
        best_model_name = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
        best_metrics = aggregate[best_model_name]
        best_metrics["best_model"] = best_model_name
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