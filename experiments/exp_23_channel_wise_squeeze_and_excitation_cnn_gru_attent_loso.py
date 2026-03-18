# FILE: experiments/exp_23_channel_wise_squeeze_and_excitation_cnn_gru_attent_loso.py
import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    DEFAULT_SUBJECTS,
    CI_TEST_SUBJECTS,
)

HYPOTHESIS_ID = "2846e57d-5df2-477f-9d5b-9cab62aab2c9"


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


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Per-class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (default 2.0)
        reduction: 'mean' or 'sum' or 'none'
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


class FocalLossTrainer(WindowClassifierTrainer):
    """Custom trainer using Focal Loss and Cosine Annealing with Warm Restarts."""
    
    def __init__(self, focal_gamma: float = 2.0, cosine_t0: int = 10, cosine_t_mult: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.focal_gamma = focal_gamma
        self.cosine_t0 = cosine_t0
        self.cosine_t_mult = cosine_t_mult
        self.focal_alpha = None
        self.class_ids = None
    
    def _compute_focal_alpha(self, labels: np.ndarray, num_classes: int) -> torch.Tensor:
        """Compute per-class alpha weights as inverse frequency normalized."""
        unique_classes, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        n_classes = len(unique_classes)

        # Inverse frequency
        inv_freq = total / (counts * n_classes)
        # Normalize to sum to n_classes
        alpha = inv_freq / inv_freq.sum() * n_classes

        full_alpha = np.ones(num_classes, dtype=np.float32)
        for i, cls in enumerate(unique_classes):
            if cls < num_classes:
                full_alpha[cls] = alpha[i]

        return torch.tensor(full_alpha, dtype=torch.float32, device=self.cfg.device)
    
    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """Fit using focal loss and cosine annealing scheduler."""
        # Unpack splits using parent helper
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        self.class_ids = class_ids
        self.class_names = class_names

        # Transpose (N, T, C) -> (N, C, T) for Conv1d
        X_train_c = X_train.transpose(0, 2, 1)
        X_val_c = X_val.transpose(0, 2, 1)

        # Compute and store channel standardization
        mean_c, std_c = self._compute_channel_standardization(X_train_c)
        self.mean_c = mean_c
        self.std_c = std_c

        X_train_c = self._apply_standardization(X_train_c, mean_c, std_c)
        X_val_c = self._apply_standardization(X_val_c, mean_c, std_c)

        num_classes = len(class_ids)

        # Compute focal alpha
        self.focal_alpha = self._compute_focal_alpha(y_train, num_classes)

        # Create datasets and loaders
        train_dataset = self._create_dataset_from_arrays(X_train_c, y_train)
        val_dataset = self._create_dataset_from_arrays(X_val_c, y_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True
        )

        # Create model
        in_channels = X_train_c.shape[1]  # (N, C, T) -> C
        self.model = self._create_model(in_channels, num_classes, self.cfg.model_type)
        self.model.to(self.cfg.device)

        # Focal loss
        criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )

        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self.cosine_t0, T_mult=self.cosine_t_mult
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []

        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.cfg.device)
                batch_y = batch_y.to(self.cfg.device)

                # Apply noise augmentation if enabled
                if self.cfg.aug_apply and self.cfg.aug_apply_noise:
                    noise = torch.randn_like(batch_x) * self.cfg.aug_noise_std
                    batch_x = batch_x + noise

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            scheduler.step()

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.cfg.device)
                    batch_y = batch_y.to(self.cfg.device)

                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()

            val_loss /= len(val_loader)
            val_acc = correct / total

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            })

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.cfg.epochs} - "
                    f"Train Loss: {train_loss/len(train_loader):.4f} - "
                    f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    if self.logger:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        return {'training_history': training_history, 'best_val_loss': best_val_loss}
    
    def _create_dataset_from_arrays(self, X: np.ndarray, y: np.ndarray):
        """Create PyTorch dataset from already-transposed (N, C, T) arrays."""
        from torch.utils.data import TensorDataset
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return TensorDataset(x_tensor, y_tensor)


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
    focal_gamma: float = 2.0,
    cosine_t0: int = 10,
    cosine_t_mult: int = 2,
) -> Dict:
    """Run single LOSO fold with focal loss and cosine annealing."""
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
    
    # Use custom trainer with focal loss
    trainer = FocalLossTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        focal_gamma=focal_gamma,
        cosine_t0=cosine_t0,
        cosine_t_mult=cosine_t_mult,
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
    test_acc = test_metrics.get("accuracy")
    test_f1 = test_metrics.get("f1_macro")
    
    acc_str = f"{test_acc:.4f}" if test_acc is not None else "None"
    f1_str = f"{test_f1:.4f}" if test_f1 is not None else "None"
    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: {model_type} | Approach: {approach} | "
        f"Accuracy={acc_str}, F1-macro={f1_str}"
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
        "focal_loss_gamma": focal_gamma,
        "cosine_annealing_t0": cosine_t0,
        "cosine_annealing_t_mult": cosine_t_mult,
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
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true", help="Run with full 20 subjects")
    _args, _ = _parser.parse_known_args()
    
    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    else:
        ALL_SUBJECTS = ['DB2_s1', 'DB2_s12', 'DB2_s15', 'DB2_s28', 'DB2_s39']
    
    EXPERIMENT_NAME = "exp_23_channel_wise_squeeze_and_excitation_cnn_gru_attent_loso"
    HYPOTHESIS_ID = "2846e57d-5df2-477f-9d5b-9cab62aab2c9"
    BASE_DIR = Path("/home/kirill/omega_data")
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    EXERCISES = ["E1"]
    MODEL_TYPE = "cnn_gru_attention"
    APPROACH = "deep_raw"
    
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
        batch_size=4096,
        epochs=60,
        learning_rate=5e-4,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type=APPROACH,
        model_type=MODEL_TYPE,
        aug_apply=True,
        aug_noise_std=0.005,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=False,
    )
    
    # Focal loss and cosine annealing params
    FOCAL_GAMMA = 2.0
    COSINE_T0 = 10
    COSINE_T_MULT = 2
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: {MODEL_TYPE} | Approach: {APPROACH}")
    print(f"Focal Loss: gamma={FOCAL_GAMMA}")
    print(f"Scheduler: CosineAnnealingWarmRestarts T_0={COSINE_T0}, T_mult={COSINE_T_MULT}")
    print(f"LOSO n={len(ALL_SUBJECTS)} subjects: {ALL_SUBJECTS}")
    print(f"Augmentation: noise (std={train_cfg.aug_noise_std})")
    
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
                focal_gamma=FOCAL_GAMMA,
                cosine_t0=COSINE_T0,
                cosine_t_mult=COSINE_T_MULT,
            )
            all_loso_results.append(fold_res)
            
            acc_str = f"{fold_res['test_accuracy']:.4f}" if fold_res.get('test_accuracy') is not None else "None"
            f1_str = f"{fold_res['test_f1_macro']:.4f}" if fold_res.get('test_f1_macro') is not None else "None"
            print(f"  ✓ {test_subject}: acc={acc_str}, f1={f1_str}")
            
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
    
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        
        aggregate = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
            "acc_f1_ratio": float(np.mean(accs) / max(np.mean(f1s), 1e-6)),
        }
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS ({len(valid_results)} subjects)")
        print(f"Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"F1-macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"Acc/F1 Ratio: {aggregate['acc_f1_ratio']:.2f}")
        print(f"{'='*60}")
    else:
        aggregate = None
        print("\nNo valid results obtained!")
    
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": APPROACH,
        "model": MODEL_TYPE,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "focal_loss": {
            "gamma": FOCAL_GAMMA,
            "alpha": "inverse_class_frequency_normalized",
        },
        "scheduler": {
            "type": "CosineAnnealingWarmRestarts",
            "T_0": COSINE_T0,
            "T_mult": COSINE_T_MULT,
        },
        "augmentation": f"noise (std={train_cfg.aug_noise_std})",
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
    try:
        from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed

        if aggregate:
            mark_hypothesis_verified(
                hypothesis_id=HYPOTHESIS_ID,
                metrics=aggregate,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                hypothesis_id=HYPOTHESIS_ID,
                error_message="No successful LOSO folds completed",
            )
    except ImportError:
        print("hypothesis_executor not available, skipping Qdrant update")


if __name__ == "__main__":
    main()