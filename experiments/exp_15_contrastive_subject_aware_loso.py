# FILE: experiments/exp_15_contrastive_subject_aware_loso.py
"""LOSO experiment: Contrastive Learning with Subject-Aware Augmentation.

Hypothesis: Two-stage training (contrastive pre-training + supervised fine-tuning)
with subject-specific augmentation parameters improves accuracy and reduces inter-subject variance.
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
from tqdm import tqdm

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

# Register custom model
from models import register_model
from models.contrastive_cnn import ContrastiveCNN, NTXentLoss

register_model("contrastive_cnn", ContrastiveCNN)


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


def compute_subject_augmentation_params(
    subject_tuple,  # (emg, segments, grouped_windows) — tuple from load_multiple_subjects
    base_noise_std: float = 0.02,
    base_time_warp: float = 0.1,
    num_bins: int = 3,
) -> Dict[str, float]:
    """Compute subject-specific augmentation parameters based on signal statistics.

    Adapts noise and time-warp levels based on the subject's EMG signal characteristics.
    Higher signal variance -> lower augmentation intensity (signal already has diversity).
    Lower signal variance -> higher augmentation intensity (need more variation).
    """
    _, _, grouped_windows = subject_tuple

    # Collect all windows from all gestures/repetitions
    all_signals = []
    for reps in grouped_windows.values():
        for rep_windows in reps:
            if isinstance(rep_windows, np.ndarray) and len(rep_windows) > 0:
                all_signals.append(rep_windows)

    if not all_signals:
        return {"noise_std": base_noise_std, "time_warp_max": base_time_warp}

    all_signals = np.concatenate(all_signals, axis=0)  # (N, T, C)

    # Compute per-channel variance (axes 0=samples, 1=time), then average
    channel_vars = np.var(all_signals, axis=(0, 1))  # (C,)
    avg_variance = float(np.mean(channel_vars))
    
    # Normalize variance to determine augmentation scaling
    # Low variance -> need more augmentation, high variance -> need less
    variance_percentile = min(1.0, avg_variance / 0.5)  # Normalize by typical EMG variance
    
    # Adaptive scaling: inverse relationship
    # Low variance (0.1) -> scale up augmentation by 1.5x
    # High variance (0.9) -> scale down augmentation by 0.7x
    scale_factor = 1.0 + (0.5 - variance_percentile) * 0.8  # Range: [0.6, 1.4]
    
    noise_std = base_noise_std * scale_factor
    time_warp_max = base_time_warp * scale_factor
    
    # Clip to reasonable ranges
    noise_std = np.clip(noise_std, 0.005, 0.05)
    time_warp_max = np.clip(time_warp_max, 0.05, 0.2)
    
    return {"noise_std": float(noise_std), "time_warp_max": float(time_warp_max)}


def augment_batch(
    batch: torch.Tensor,
    noise_std: float,
    time_warp_max: float,
    apply_noise: bool = True,
    apply_time_warp: bool = True,
) -> torch.Tensor:
    """Apply augmentation to a batch of EMG windows."""
    augmented = batch.clone()
    
    if apply_noise and noise_std > 0:
        noise = torch.randn_like(augmented) * noise_std
        augmented = augmented + noise
    
    if apply_time_warp and time_warp_max > 0:
        # Simple time warping: stretch/compress time axis slightly
        batch_size, channels, time_steps = augmented.shape
        warp_factor = 1.0 + (torch.rand(batch_size, device=batch.device) * 2 - 1) * time_warp_max
        
        # Apply warping via interpolation
        warped = []
        for i in range(batch_size):
            # Generate new time indices
            old_indices = torch.linspace(0, time_steps - 1, time_steps, device=batch.device)
            new_length = int(time_steps * warp_factor[i])
            new_length = max(time_steps // 2, min(time_steps * 2, new_length))
            new_indices = torch.linspace(0, time_steps - 1, new_length, device=batch.device)
            
            # Interpolate
            warped_signal = torch.nn.functional.interpolate(
                augmented[i:i+1],
                size=new_length,
                mode='linear',
                align_corners=False
            )
            
            # Resample back to original length
            warped_signal = torch.nn.functional.interpolate(
                warped_signal,
                size=time_steps,
                mode='linear',
                align_corners=False
            )
            warped.append(warped_signal)
        
        augmented = torch.cat(warped, dim=0)
    
    return augmented


def contrastive_pretraining(
    model: ContrastiveCNN,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    epochs: int = 20,
    lr: float = 1e-3,
    temperature: float = 0.5,
    patience: int = 5,
    noise_std: float = 0.02,
    time_warp_max: float = 0.1,
    logger=None,
) -> Dict:
    """Stage 1: Contrastive pre-training with SimCLR."""
    
    model.set_contrastive_mode(True)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = NTXentLoss(temperature=temperature)
    
    best_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, _ in tqdm(train_loader, desc=f"Contrastive Epoch {epoch+1}/{epochs}", leave=False):
            batch_x = batch_x.to(device)
            
            # Create two augmented views
            aug1 = augment_batch(batch_x, noise_std, time_warp_max, apply_noise=True, apply_time_warp=True)
            aug2 = augment_batch(batch_x, noise_std, time_warp_max, apply_noise=True, apply_time_warp=True)
            
            # Forward pass
            z1 = model(aug1, return_projection=True)
            z2 = model(aug2, return_projection=True)
            
            loss = criterion(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)
        scheduler.step()
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    aug1 = augment_batch(batch_x, noise_std, time_warp_max)
                    aug2 = augment_batch(batch_x, noise_std, time_warp_max)
                    z1 = model(aug1, return_projection=True)
                    z2 = model(aug2, return_projection=True)
                    loss = criterion(z1, z2)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(avg_val_loss)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if logger:
                logger.info(f"Contrastive Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            
            if patience_counter >= patience:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if logger:
                logger.info(f"Contrastive Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
    
    model.set_contrastive_mode(False)
    return history


def supervised_finetuning(
    model: ContrastiveCNN,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_classes: int,
    device: str,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    use_class_weights: bool = True,
    noise_std: float = 0.02,
    time_warp_max: float = 0.1,
    logger=None,
) -> Tuple[Dict, nn.Module]:
    """Stage 2: Supervised fine-tuning with frozen encoder initially, then full fine-tuning."""
    
    model.set_contrastive_mode(False)
    model.to(device)
    
    # Compute class weights
    class_weights = None
    if use_class_weights:
        all_labels = []
        for _, y in train_loader:
            all_labels.extend(y.numpy())
        class_counts = np.bincount(all_labels, minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        class_weights = class_weights / class_weights.sum() * num_classes
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Phase 1: Train classifier only (frozen encoder)
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr * 10, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Classifier-only training (5 epochs)
    for epoch in range(5):
        model.train()
        train_losses, train_correct, train_total = [], 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Apply augmentation
            aug_x = augment_batch(batch_x, noise_std, time_warp_max, apply_noise=True, apply_time_warp=True)
            
            outputs = model(aug_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = train_correct / train_total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
    
    # Phase 2: Full fine-tuning
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses, train_correct, train_total = [], 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Apply augmentation
            aug_x = augment_batch(batch_x, noise_std, time_warp_max, apply_noise=True, apply_time_warp=True)
            
            outputs = model(aug_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = train_correct / train_total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_losses, val_correct, val_total = [], 0, 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_losses.append(loss.item())
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
            
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = val_correct / val_total
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(avg_val_acc)
            
            scheduler.step(avg_val_acc)
            
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if logger:
                logger.info(f"Fine-tune Epoch {epoch+1}: train_acc={avg_train_acc:.4f}, val_acc={avg_val_acc:.4f}")
            
            if patience_counter >= patience:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if logger:
                logger.info(f"Fine-tune Epoch {epoch+1}: train_acc={avg_train_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate model on test set."""
    model.eval()
    model.to(device)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    
    # Compute per-class F1
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "num_samples": len(all_labels),
    }


def run_single_loso_fold_contrastive(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    contrastive_epochs: int = 20,
    finetune_epochs: int = 50,
) -> Dict:
    """Run single LOSO fold with contrastive learning + subject-aware augmentation."""
    
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
        use_improved_processing=True,
    )
    
    # Load subjects data directly via multi_loader
    # Returns Dict[str, Tuple[emg, segments, grouped_windows]] — NOT dict of dicts
    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=cs_cfg.base_dir,
        subject_ids=all_subject_ids,
        exercises=cs_cfg.exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    if not subjects_data:
        raise ValueError(f"No subject data loaded for test_subject={test_subject}")

    # Determine common gestures across all loaded subjects
    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=cs_cfg.max_gestures)
    if not common_gestures:
        raise ValueError("No common gestures found across subjects")
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    # Class mapping: gesture_id -> class index
    class_ids = sorted(common_gestures)
    gesture_to_class = {gid: i for i, gid in enumerate(class_ids)}

    # Compute subject-specific augmentation parameters
    subject_aug_params = {}
    for subj_name, subj_tuple in subjects_data.items():
        params = compute_subject_augmentation_params(
            subj_tuple,
            base_noise_std=train_cfg.aug_noise_std,
            base_time_warp=train_cfg.aug_time_warp_max,
        )
        subject_aug_params[subj_name] = params
        logger.info(f"Subject {subj_name}: noise_std={params['noise_std']:.4f}, time_warp={params['time_warp_max']:.4f}")
    
    # Use average of training subjects' augmentation params
    train_params = [subject_aug_params[s] for s in train_subjects if s in subject_aug_params]
    if train_params:
        avg_noise_std = np.mean([p["noise_std"] for p in train_params])
        avg_time_warp = np.mean([p["time_warp_max"] for p in train_params])
    else:
        avg_noise_std = train_cfg.aug_noise_std
        avg_time_warp = train_cfg.aug_time_warp_max
    
    logger.info(f"Using augmentation params: noise_std={avg_noise_std:.4f}, time_warp={avg_time_warp:.4f}")
    
    # Prepare tensors
    train_windows, train_labels = [], []
    val_windows, val_labels = [], []
    test_windows, test_labels = [], []
    
    for subj_name, subj_tuple in subjects_data.items():
        # subjects_data values are tuples (emg, segments, grouped_windows)
        _, _, grouped_windows = subj_tuple

        for gesture_id in sorted(grouped_windows.keys()):
            if gesture_id not in common_gestures:
                continue
            cls_idx = gesture_to_class[gesture_id]

            for rep_windows in grouped_windows[gesture_id]:
                if len(rep_windows) == 0:
                    continue
                rep_labels = np.full(len(rep_windows), cls_idx, dtype=np.int64)

                # Determine split
                if subj_name == test_subject:
                    test_windows.append(rep_windows)
                    test_labels.append(rep_labels)
                else:
                    # Split into train/val per repetition
                    n_samples = len(rep_windows)
                    n_train = int(n_samples * split_cfg.train_ratio)
                    n_val = int(n_samples * split_cfg.val_ratio)

                    indices = np.random.permutation(n_samples)

                    train_windows.append(rep_windows[indices[:n_train]])
                    train_labels.append(rep_labels[indices[:n_train]])
                    val_windows.append(rep_windows[indices[n_train:n_train + n_val]])
                    val_labels.append(rep_labels[indices[n_train:n_train + n_val]])
    
    if not train_windows:
        raise ValueError("No training data")
    if not test_windows:
        raise ValueError("No test data")
    
    # Concatenate
    train_x = np.concatenate(train_windows, axis=0)
    train_y = np.concatenate(train_labels, axis=0)
    val_x = np.concatenate(val_windows, axis=0) if val_windows else np.array([])
    val_y = np.concatenate(val_labels, axis=0) if val_labels else np.array([])
    test_x = np.concatenate(test_windows, axis=0)
    test_y = np.concatenate(test_labels, axis=0)
    
    logger.info(f"Data shapes: train={train_x.shape}, val={val_x.shape if len(val_x) > 0 else 'empty'}, test={test_x.shape}")
    
    # Standardize per channel
    train_mean = train_x.mean(axis=(0, 2), keepdims=True)
    train_std = train_x.std(axis=(0, 2), keepdims=True) + 1e-8
    
    train_x = (train_x - train_mean) / train_std
    if len(val_x) > 0:
        val_x = (val_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std
    
    # Convert to tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.long)
    
    # Create data loaders
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    
    val_loader = None
    if len(val_x) > 0:
        val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
        val_y_tensor = torch.tensor(val_y, dtype=torch.long)
        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)
    
    # Get dimensions
    in_channels = train_x.shape[1]
    num_classes = int(max(train_y.max(), test_y.max()) + 1)
    
    logger.info(f"Model: in_channels={in_channels}, num_classes={num_classes}")
    
    # Create model
    model = ContrastiveCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=train_cfg.dropout,
        projection_dim=128,
    )
    
    device = train_cfg.device
    
    # Stage 1: Contrastive Pre-training
    logger.info("=== Stage 1: Contrastive Pre-training ===")
    contrastive_history = contrastive_pretraining(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=contrastive_epochs,
        lr=train_cfg.learning_rate,
        temperature=0.5,
        patience=7,
        noise_std=avg_noise_std,
        time_warp_max=avg_time_warp,
        logger=logger,
    )
    
    # Stage 2: Supervised Fine-tuning
    logger.info("=== Stage 2: Supervised Fine-tuning ===")
    finetune_history, model = supervised_finetuning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        epochs=finetune_epochs,
        lr=train_cfg.learning_rate * 0.1,
        weight_decay=train_cfg.weight_decay,
        patience=train_cfg.early_stopping_patience,
        use_class_weights=train_cfg.use_class_weights,
        noise_std=avg_noise_std,
        time_warp_max=avg_time_warp,
        logger=logger,
    )
    
    # Evaluate
    logger.info("=== Evaluation ===")
    test_metrics = evaluate_model(model, test_loader, device)
    
    test_acc = test_metrics["accuracy"]
    test_f1 = test_metrics["f1_macro"]
    
    print(
        f"[LOSO-Contrastive] Test subject {test_subject} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )
    
    # Save results
    results = {
        "test_subject": test_subject,
        "model_type": "contrastive_cnn",
        "approach": "contrastive_subject_aware",
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "contrastive_history": contrastive_history,
        "finetune_history": finetune_history,
        "subject_aug_params": subject_aug_params,
        "avg_aug_params": {"noise_std": avg_noise_std, "time_warp_max": avg_time_warp},
    }
    
    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(make_json_serializable(results), f, indent=4)
    
    # Save model
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    del model, multi_loader
    gc.collect()
    
    return {
        "test_subject": test_subject,
        "model_type": "contrastive_cnn",
        "approach": "contrastive_subject_aware",
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
    }


def main():
    EXPERIMENT_NAME = "exp_15_contrastive_subject_aware_loso"
    HYPOTHESIS_ID = "70d5a74e-f695-4d0b-a782-6c001ea6f4be"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    ALL_SUBJECTS = [
        "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]
    
    # Processing config optimized for simple_cnn baseline
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
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        model_type="contrastive_cnn",
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Hypothesis: Contrastive learning with subject-aware augmentation")
    print(f"Model: contrastive_cnn (SimpleCNN + projection head)")
    print(f"LOSO: n={len(ALL_SUBJECTS)} subjects")
    
    all_loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        try:
            fold_res = run_single_loso_fold_contrastive(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                contrastive_epochs=20,
                finetune_epochs=50,
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
                "model_type": "contrastive_cnn",
                "approach": "contrastive_subject_aware",
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
        
        aggregate["contrastive_cnn"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
        }
        
        print(f"\n{'='*60}")
        print(f"LOSO Results for Contrastive CNN + Subject-Aware Augmentation:")
        print(f"  Accuracy: {aggregate['contrastive_cnn']['mean_accuracy']:.4f} ± {aggregate['contrastive_cnn']['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate['contrastive_cnn']['mean_f1_macro']:.4f} ± {aggregate['contrastive_cnn']['std_f1_macro']:.4f}")
        print(f"  Target: accuracy >= 0.38, std <= 0.07")
        print(f"{'='*60}\n")
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "model": "contrastive_cnn",
        "approach": "contrastive_subject_aware",
        "training_stages": ["contrastive_pretraining", "supervised_finetuning"],
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "subject-adaptive (noise + time_warp)",
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to {OUTPUT_DIR.resolve()}")
    
    # === Update hypothesis status in Qdrant ===
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