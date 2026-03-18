# FILE: experiments/exp_11_enhancing_simple_cnn_robustness_with_subject_speci_loso.py
import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# добавить корень репо в sys.path
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


def compute_subject_calibration_stats(
    windows: np.ndarray,
    calibration_fraction: float = 0.1,
) -> Dict[str, float]:
    """
    Compute EMG signal statistics from calibration set (first fraction of data).
    
    Args:
        windows: Array of shape (N, C, T) - EMG windows
        calibration_fraction: Fraction of data to use for calibration
        
    Returns:
        Dictionary with calibration statistics
    """
    n_windows = windows.shape[0]
    n_calib = max(1, int(n_windows * calibration_fraction))
    calib_windows = windows[:n_calib]  # (n_calib, C, T)
    
    # Compute statistics per channel, then aggregate
    # Signal variance (across time dimension)
    channel_variances = np.var(calib_windows, axis=2)  # (n_calib, C)
    mean_variance = float(np.mean(channel_variances))
    std_variance = float(np.std(channel_variances))
    
    # Signal amplitude (mean absolute value)
    channel_amplitudes = np.mean(np.abs(calib_windows), axis=2)  # (n_calib, C)
    mean_amplitude = float(np.mean(channel_amplitudes))
    
    # Signal smoothness using gradient-based metric
    # Lower smoothness = higher frequency content / more noise
    gradients = np.gradient(calib_windows, axis=2)  # (n_calib, C, T-1)
    gradient_magnitudes = np.mean(np.abs(gradients), axis=2)  # (n_calib, C)
    mean_gradient = float(np.mean(gradient_magnitudes))
    
    # Smoothness ratio: amplitude / gradient magnitude
    # Higher = smoother signal
    smoothness_ratio = mean_amplitude / (mean_gradient + 1e-8)
    
    # Signal-to-noise proxy (using coefficient of variation)
    snr_proxy = mean_amplitude / (np.sqrt(mean_variance) + 1e-8)
    
    return {
        "mean_variance": mean_variance,
        "std_variance": std_variance,
        "mean_amplitude": mean_amplitude,
        "mean_gradient": mean_gradient,
        "smoothness_ratio": smoothness_ratio,
        "snr_proxy": snr_proxy,
        "n_calibration_windows": n_calib,
    }


def calibrate_augmentation_params(
    subject_stats_list: List[Dict[str, float]],
    base_noise_std: float = 0.02,
    base_time_warp_max: float = 0.1,
) -> Dict[str, float]:
    """
    Compute subject-calibrated augmentation parameters.
    
    For each subject, compute calibration factors based on their signal statistics,
    then aggregate to get effective augmentation parameters for the training fold.
    
    Strategy:
    - noise_std: Scale by sqrt(subject_variance / global_mean_variance)
      Higher variance signals get more noise augmentation
    - time_warp_max: Scale by smoothness factor
      Smoother signals can tolerate more time warping
    
    Args:
        subject_stats_list: List of calibration stats for each training subject
        base_noise_std: Base noise standard deviation
        base_time_warp_max: Base time warp maximum
        
    Returns:
        Calibrated augmentation parameters
    """
    if not subject_stats_list:
        return {"noise_std": base_noise_std, "time_warp_max": base_time_warp_max}
    
    # Aggregate statistics across subjects
    all_variances = [s["mean_variance"] for s in subject_stats_list]
    all_smoothness = [s["smoothness_ratio"] for s in subject_stats_list]
    all_snr = [s["snr_proxy"] for s in subject_stats_list]
    
    global_mean_variance = np.mean(all_variances)
    global_mean_smoothness = np.mean(all_smoothness)
    global_mean_snr = np.mean(all_snr)
    
    # Compute calibration factors per subject
    noise_factors = []
    warp_factors = []
    
    for stats in subject_stats_list:
        # Noise factor: subjects with higher variance get proportionally more noise
        # But capped to avoid extreme values
        var_ratio = stats["mean_variance"] / (global_mean_variance + 1e-8)
        noise_factor = np.clip(np.sqrt(var_ratio), 0.5, 2.0)
        noise_factors.append(noise_factor)
        
        # Warp factor: smoother signals can handle more warping
        smooth_ratio = stats["smoothness_ratio"] / (global_mean_smoothness + 1e-8)
        warp_factor = np.clip(smooth_ratio, 0.5, 2.0)
        warp_factors.append(warp_factor)
    
    # Use mean calibration factor (could also use weighted average by data size)
    mean_noise_factor = np.mean(noise_factors)
    mean_warp_factor = np.mean(warp_factors)
    
    # Compute calibrated parameters
    calibrated_noise_std = base_noise_std * mean_noise_factor
    calibrated_time_warp_max = base_time_warp_max * mean_warp_factor
    
    # Ensure reasonable bounds
    calibrated_noise_std = np.clip(calibrated_noise_std, 0.005, 0.05)
    calibrated_time_warp_max = np.clip(calibrated_time_warp_max, 0.05, 0.2)
    
    return {
        "noise_std": float(calibrated_noise_std),
        "time_warp_max": float(calibrated_time_warp_max),
        "mean_noise_factor": float(mean_noise_factor),
        "mean_warp_factor": float(mean_warp_factor),
        "global_mean_variance": float(global_mean_variance),
        "global_mean_smoothness": float(global_mean_smoothness),
    }


class SubjectCalibratedTrainer(WindowClassifierTrainer):
    """
    Extended trainer with subject-specific augmentation calibration.
    
    Overrides augmentation logic to use calibrated parameters based on
    subject-specific EMG signal characteristics.
    """
    
    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        output_dir: Path,
        visualizer: Visualizer,
        calibration_stats: Optional[Dict] = None,
        calibrated_aug_params: Optional[Dict] = None,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.calibration_stats = calibration_stats or {}
        self.calibrated_aug_params = calibrated_aug_params or {}
        
    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply calibrated augmentation to input batch.
        
        Uses subject-calibrated noise and time warp parameters.
        """
        if not self.cfg.aug_apply:
            return x
            
        x_aug = x.clone()
        batch_size, n_channels, seq_len = x.shape
        
        # Get calibrated parameters (fallback to config defaults)
        noise_std = self.calibrated_aug_params.get("noise_std", self.cfg.aug_noise_std)
        time_warp_max = self.calibrated_aug_params.get("time_warp_max", self.cfg.aug_time_warp_max)
        
        # Apply noise augmentation
        if self.cfg.aug_apply_noise:
            noise = torch.randn_like(x_aug) * noise_std
            x_aug = x_aug + noise
        
        # Apply time warp augmentation
        if self.cfg.aug_apply_time_warp and time_warp_max > 0:
            # Simple time warping via random stretching/compression
            for i in range(batch_size):
                warp_factor = 1.0 + np.random.uniform(-time_warp_max, time_warp_max)
                
                # Compute new length
                new_len = int(seq_len * warp_factor)
                new_len = max(seq_len // 2, min(seq_len * 2, new_len))
                
                if new_len != seq_len:
                    # Interpolate
                    x_np = x_aug[i].cpu().numpy()  # (C, T)
                    x_interp = np.zeros((n_channels, seq_len))
                    
                    orig_indices = np.linspace(0, seq_len - 1, seq_len)
                    new_indices = np.linspace(0, seq_len - 1, new_len)
                    
                    for c in range(n_channels):
                        if warp_factor > 1.0:  # Stretch
                            # Pad with edge values
                            padded = np.pad(x_np[c], (0, new_len - seq_len), mode='edge')
                            x_interp[c] = np.interp(orig_indices, np.linspace(0, seq_len - 1, new_len), padded[:new_len])
                        else:  # Compress
                            x_interp[c] = np.interp(orig_indices, new_indices, x_np[c][:new_len] if new_len < seq_len else x_np[c])
                    
                    x_aug[i] = torch.from_numpy(x_interp).to(x.device)
        
        return x_aug
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
    ) -> tuple:
        """
        Override train_epoch to use calibrated augmentation.
        """
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Apply calibrated augmentation
            if self.cfg.aug_apply:
                batch_x = self.apply_augmentation(batch_x)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(batch_y).sum().item()
            total_samples += batch_x.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy


def run_single_loso_fold_calibrated(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    use_improved_processing: bool,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    calibration_fraction: float = 0.1,
) -> Dict:
    """
    One LOSO fold with subject-specific augmentation calibration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = "deep_raw"
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

    # Load data to compute calibration statistics
    def grouped_to_windows(grouped_windows):
        windows_list = []
        for gesture_id in sorted(grouped_windows.keys()):
            for rep_windows in grouped_windows[gesture_id]:
                if len(rep_windows) > 0:
                    windows_list.append(rep_windows)
        return np.concatenate(windows_list, axis=0) if windows_list else np.array([])

    try:
        subjects_data = multi_loader.load_multiple_subjects(
            base_dir=base_dir,
            subject_ids=train_subjects,
            exercises=exercises,
            include_rest=True,
        )
    except Exception as e:
        logger.error(f"Failed to load subjects data: {e}")
        raise

    # Compute calibration statistics for each training subject
    all_subject_stats = {}
    calibration_info = {"subjects": {}}

    for subj_name, (emg, segments, grouped_windows) in subjects_data.items():
        windows = grouped_to_windows(grouped_windows)
        if windows is not None and len(windows) > 0:
            stats = compute_subject_calibration_stats(windows, calibration_fraction)
            all_subject_stats[subj_name] = stats
            calibration_info["subjects"][subj_name] = stats
            logger.info(f"Calibration stats for {subj_name}: var={stats['mean_variance']:.4f}, "
                       f"smooth={stats['smoothness_ratio']:.4f}, snr={stats['snr_proxy']:.4f}")

    # Compute calibrated augmentation parameters
    subject_stats_list = list(all_subject_stats.values())
    calibrated_params = calibrate_augmentation_params(
        subject_stats_list,
        base_noise_std=train_cfg.aug_noise_std,
        base_time_warp_max=train_cfg.aug_time_warp_max,
    )
    calibration_info["calibrated_params"] = calibrated_params
    
    logger.info(f"Calibrated augmentation: noise_std={calibrated_params['noise_std']:.4f}, "
               f"time_warp_max={calibrated_params['time_warp_max']:.4f}")
    
    # Update train_cfg with calibrated parameters
    train_cfg.aug_noise_std = calibrated_params["noise_std"]
    train_cfg.aug_time_warp_max = calibrated_params["time_warp_max"]
    
    # Save calibration info
    with open(output_dir / "calibration_info.json", "w") as f:
        json.dump(make_json_serializable(calibration_info), f, indent=4)

    # Create calibrated trainer
    trainer = SubjectCalibratedTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        calibration_stats=all_subject_stats,
        calibrated_aug_params=calibrated_params,
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
            "approach": "deep_raw",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO-CALIBRATED] Test subject {test_subject} | "
        f"Model: {model_type} | "
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
        "approach": "deep_raw",
        "exercises": exercises,
        "use_improved_processing": use_improved_processing,
        "calibration_fraction": calibration_fraction,
        "calibrated_params": calibrated_params,
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": "deep_raw",
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "calibrated_noise_std": calibrated_params["noise_std"],
        "calibrated_time_warp_max": calibrated_params["time_warp_max"],
    }


def main():
    EXPERIMENT_NAME = "exp_11_enhancing_simple_cnn_robustness_with_subject_speci_loso"
    HYPOTHESIS_ID = "11cfe09b-5d98-4fa9-9bee-16222fae2aab"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")

    ALL_SUBJECTS = [
        "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]
    MODEL_TYPE = "simple_cnn"

    # Processing config matching exp6 baseline for fair comparison
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
    # Base parameters will be calibrated per-fold
    train_cfg = TrainingConfig(
        batch_size=4096,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        model_type=MODEL_TYPE,
        aug_apply=True,
        aug_noise_std=0.02,  # Base value, will be calibrated
        aug_time_warp_max=0.1,  # Base value, will be calibrated
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"HYPOTHESIS: Subject-specific augmentation calibration for {MODEL_TYPE}")
    print(f"LOSO n={len(ALL_SUBJECTS)} subjects")
    print(f"Base augmentation: noise_std=0.02, time_warp_max=0.1")
    print(f"Calibration: first 10% of each subject's data")

    all_loso_results = []
    
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"
        
        try:
            fold_res = run_single_loso_fold_calibrated(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                model_type=MODEL_TYPE,
                use_improved_processing=True,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                calibration_fraction=0.1,
            )
            all_loso_results.append(fold_res)
            
            if fold_res.get("test_accuracy") is not None:
                print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, "
                      f"f1={fold_res['test_f1_macro']:.4f}, "
                      f"calib_noise={fold_res.get('calibrated_noise_std', 0):.4f}, "
                      f"calib_warp={fold_res.get('calibrated_time_warp_max', 0):.4f}")
            else:
                print(f"  ✗ {test_subject}: {fold_res.get('error', 'Unknown error')}")
                
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
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
        }
        
        # Compute mean calibrated parameters
        noise_stds = [r.get("calibrated_noise_std", 0.02) for r in valid_results]
        warp_maxs = [r.get("calibrated_time_warp_max", 0.1) for r in valid_results]
        aggregate["mean_calibrated_noise_std"] = float(np.mean(noise_stds))
        aggregate["mean_calibrated_time_warp_max"] = float(np.mean(warp_maxs))
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS:")
        print(f"  Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"  Range: [{aggregate['min_accuracy']:.4f}, {aggregate['max_accuracy']:.4f}]")
        print(f"  Mean calibrated noise_std: {aggregate['mean_calibrated_noise_std']:.4f}")
        print(f"  Mean calibrated time_warp_max: {aggregate['mean_calibrated_time_warp_max']:.4f}")
        print(f"{'='*60}")
        
        # Check hypothesis targets
        if aggregate['std_accuracy'] < 0.06:
            print(f"✓ TARGET MET: Inter-subject variance (std={aggregate['std_accuracy']:.4f}) < 0.06")
        else:
            print(f"✗ TARGET NOT MET: Inter-subject variance (std={aggregate['std_accuracy']:.4f}) >= 0.06")
            
        if aggregate['mean_accuracy'] >= 0.35:
            print(f"✓ TARGET MET: Mean accuracy ({aggregate['mean_accuracy']:.4f}) >= 0.35")
        else:
            print(f"✗ TARGET NOT MET: Mean accuracy ({aggregate['mean_accuracy']:.4f}) < 0.35")
    else:
        aggregate = {}
        print("\nERROR: No valid LOSO folds completed!")

    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "model": MODEL_TYPE,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "subject-calibrated (noise + time_warp)",
        "calibration_fraction": 0.1,
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
            "mean_calibrated_noise_std": aggregate.get("mean_calibrated_noise_std", 0),
            "mean_calibrated_time_warp_max": aggregate.get("mean_calibrated_time_warp_max", 0),
            "best_model": MODEL_TYPE,
        }
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