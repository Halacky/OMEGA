# FILE: experiments/exp_25_svm_linear_on_powerful_features_with_combined_nois_loso.py
import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    DEFAULT_SUBJECTS,
    CI_TEST_SUBJECTS,
    make_json_serializable,
)

HYPOTHESIS_ID = "3b1480c0-aa08-4cad-b277-966e3654c011"


from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from processing.powerful_features import PowerfulFeatureExtractor
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


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


def apply_noise_augmentation(windows: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """Apply Gaussian noise augmentation to EMG windows.
    
    Args:
        windows: Shape (N, T, C)
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        Augmented windows with same shape
    """
    noise = np.random.randn(*windows.shape).astype(windows.dtype) * noise_std
    return windows + noise


def apply_time_warp_augmentation(windows: np.ndarray, max_warp: float = 0.1) -> np.ndarray:
    """Apply time warping augmentation to EMG windows.
    
    Args:
        windows: Shape (N, T, C)
        max_warp: Maximum time warp factor
        
    Returns:
        Time-warped windows
    """
    N, T, C = windows.shape
    warped = np.zeros_like(windows)
    
    for i in range(N):
        warp_factor = 1.0 + np.random.uniform(-max_warp, max_warp)
        new_length = int(T * warp_factor)
        new_length = max(10, min(T * 2, new_length))
        
        orig_indices = np.linspace(0, T - 1, new_length)
        warped_signal = np.zeros((new_length, C), dtype=windows.dtype)
        
        for c in range(C):
            warped_signal[:, c] = np.interp(orig_indices, np.arange(T), windows[i, :, c])
        
        if new_length >= T:
            start = (new_length - T) // 2
            warped[i] = warped_signal[start:start + T]
        else:
            pad_start = (T - new_length) // 2
            warped[i, :pad_start] = warped_signal[0:1]
            warped[i, pad_start:pad_start + new_length] = warped_signal
            warped[i, pad_start + new_length:] = warped_signal[-1:]
    
    return warped


def apply_rotation_augmentation(windows: np.ndarray, amplitude_factor: float = 0.1) -> np.ndarray:
    """Apply rotation/amplitude scaling augmentation to EMG windows.
    
    This simulates inter-subject electrode placement and skin impedance variations
    by applying per-channel random amplitude scaling.
    
    Args:
        windows: Shape (N, T, C)
        amplitude_factor: Maximum amplitude scaling factor
        
    Returns:
        Amplitude-scaled windows
    """
    N, T, C = windows.shape
    augmented = windows.copy()
    
    for c in range(C):
        scale = 1.0 + np.random.uniform(-amplitude_factor, amplitude_factor)
        augmented[:, :, c] *= scale
    
    return augmented


def apply_combined_augmentation(
    windows: np.ndarray,
    labels: np.ndarray,
    noise_std: float = 0.01,
    time_warp_max: float = 0.1,
    rotation_factor: float = 0.1,
    num_augmented_copies: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply combined noise + time_warp + rotation augmentation.
    
    Generates multiple augmented copies per original sample.
    
    Args:
        windows: Original windows, shape (N, T, C)
        labels: Original labels, shape (N,)
        noise_std: Standard deviation for Gaussian noise
        time_warp_max: Maximum time warp factor
        rotation_factor: Maximum amplitude scaling factor
        num_augmented_copies: Number of augmented copies to generate per original
        
    Returns:
        Tuple of (augmented_windows, augmented_labels) with original + augmented samples
    """
    N, T, C = windows.shape
    
    all_windows = [windows]
    all_labels = [labels]
    
    for _ in range(num_augmented_copies):
        aug_windows = windows.copy()
        
        aug_windows = apply_noise_augmentation(aug_windows, noise_std)
        aug_windows = apply_time_warp_augmentation(aug_windows, time_warp_max)
        aug_windows = apply_rotation_augmentation(aug_windows, rotation_factor)
        
        all_windows.append(aug_windows)
        all_labels.append(labels)
    
    combined_windows = np.concatenate(all_windows, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    indices = np.random.permutation(len(combined_labels))
    return combined_windows[indices], combined_labels[indices]


def extract_powerful_features(windows: np.ndarray, extractor: PowerfulFeatureExtractor) -> np.ndarray:
    """Extract powerful features from EMG windows.
    
    Args:
        windows: Shape (N, T, C)
        extractor: PowerfulFeatureExtractor instance
        
    Returns:
        Features shape (N, num_features)
    """
    features = extractor.transform(windows)
    return features


def run_loso_experiment(
    base_dir: Path,
    output_dir: Path,
    all_subjects: List[str],
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    noise_std: float = 0.01,
    time_warp_max: float = 0.1,
    rotation_factor: float = 0.1,
    num_augmented_copies: int = 2,
) -> Dict:
    """Run LOSO experiment with triple augmentation + powerful features + SVM-Linear."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info(f"Starting LOSO experiment with {len(all_subjects)} subjects")
    logger.info(f"Augmentation: noise_std={noise_std}, time_warp_max={time_warp_max}, "
                f"rotation_factor={rotation_factor}, copies={num_augmented_copies}")
    
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=False,
        use_improved_processing=True,
    )
    
    feature_extractor = PowerfulFeatureExtractor(sampling_rate=proc_cfg.sampling_rate)
    
    all_results = []
    
    for test_subject in all_subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"LOSO Fold: Test subject = {test_subject}")
        logger.info(f"{'='*60}")
        
        fold_output_dir = output_dir / f"test_{test_subject}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        train_subjects = [s for s in all_subjects if s != test_subject]
        
        try:
            subjects_data = multi_loader.load_multiple_subjects(
                base_dir=base_dir,
                subject_ids=train_subjects,
                exercises=exercises,
                include_rest=False,
            )
            
            test_data = multi_loader.load_multiple_subjects(
                base_dir=base_dir,
                subject_ids=[test_subject],
                exercises=exercises,
                include_rest=False,
            )
            
            if not subjects_data:
                logger.error(f"No training data loaded for fold {test_subject}")
                all_results.append({
                    "test_subject": test_subject,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": "No training data loaded",
                })
                continue
            
            common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
            gesture_to_class = {gid: i for i, gid in enumerate(sorted(common_gestures))}
            num_classes = len(gesture_to_class)
            
            logger.info(f"Common gestures: {sorted(common_gestures)}")
            logger.info(f"Number of classes: {num_classes}")
            
            train_windows_list = []
            train_labels_list = []
            
            for subj_id, (emg, segments, grouped_windows) in subjects_data.items():
                windows, labels = grouped_to_arrays(grouped_windows)
                
                mask = np.isin(labels, list(gesture_to_class.keys()))
                if mask.sum() == 0:
                    continue
                
                windows = windows[mask]
                labels = labels[mask]
                
                mapped_labels = np.array([gesture_to_class[int(l)] for l in labels])
                
                train_windows_list.append(windows)
                train_labels_list.append(mapped_labels)
            
            if not train_windows_list:
                logger.error(f"No valid training windows for fold {test_subject}")
                all_results.append({
                    "test_subject": test_subject,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": "No valid training windows",
                })
                continue
            
            train_windows = np.concatenate(train_windows_list, axis=0)
            train_labels = np.concatenate(train_labels_list, axis=0)
            
            logger.info(f"Original training samples: {len(train_labels)}")
            
            val_ratio = split_cfg.val_ratio
            val_size = int(len(train_labels) * val_ratio)
            indices = np.random.permutation(len(train_labels))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            X_train_raw = train_windows[train_indices]
            y_train = train_labels[train_indices]
            X_val_raw = train_windows[val_indices]
            y_val = train_labels[val_indices]
            
            logger.info(f"Applying triple augmentation to training data...")
            X_train_aug, y_train_aug = apply_combined_augmentation(
                X_train_raw, y_train,
                noise_std=noise_std,
                time_warp_max=time_warp_max,
                rotation_factor=rotation_factor,
                num_augmented_copies=num_augmented_copies,
            )
            logger.info(f"Augmented training samples: {len(y_train_aug)}")
            
            logger.info("Extracting powerful features...")
            X_train_features = extract_powerful_features(X_train_aug, feature_extractor)
            X_val_features = extract_powerful_features(X_val_raw, feature_extractor)
            
            logger.info(f"Feature dimension: {X_train_features.shape[1]}")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_features)
            X_val_scaled = scaler.transform(X_val_features)
            
            logger.info("Training SVM-Linear classifier...")
            svm = SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                random_state=train_cfg.seed,
            )
            svm.fit(X_train_scaled, y_train_aug)
            
            val_pred = svm.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='macro')
            logger.info(f"Validation: Acc={val_acc:.4f}, F1={val_f1:.4f}")
            
            test_windows_list = []
            test_labels_list = []
            
            for subj_id, (emg, segments, grouped_windows) in test_data.items():
                windows, labels = grouped_to_arrays(grouped_windows)
                
                mask = np.isin(labels, list(gesture_to_class.keys()))
                if mask.sum() == 0:
                    continue
                
                windows = windows[mask]
                labels = labels[mask]
                
                mapped_labels = np.array([gesture_to_class[int(l)] for l in labels])
                
                test_windows_list.append(windows)
                test_labels_list.append(mapped_labels)
            
            if not test_windows_list:
                logger.error(f"No valid test windows for fold {test_subject}")
                all_results.append({
                    "test_subject": test_subject,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": "No valid test windows",
                })
                continue
            
            test_windows = np.concatenate(test_windows_list, axis=0)
            test_labels = np.concatenate(test_labels_list, axis=0)
            
            logger.info(f"Test samples: {len(test_labels)}")
            
            X_test_features = extract_powerful_features(test_windows, feature_extractor)
            X_test_scaled = scaler.transform(X_test_features)
            
            test_pred = svm.predict(X_test_scaled)
            test_acc = accuracy_score(test_labels, test_pred)
            test_f1 = f1_score(test_labels, test_pred, average='macro')
            
            logger.info(f"Test: Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}")
            
            fold_result = {
                "test_subject": test_subject,
                "train_samples": len(y_train),
                "augmented_train_samples": len(y_train_aug),
                "test_samples": len(test_labels),
                "val_accuracy": float(val_acc),
                "val_f1_macro": float(val_f1),
                "test_accuracy": float(test_acc),
                "test_f1_macro": float(test_f1),
                "feature_dim": X_train_features.shape[1],
                "num_classes": num_classes,
            }
            
            all_results.append(fold_result)
            
            with open(fold_output_dir / "fold_results.json", "w") as f:
                json.dump(make_json_serializable(fold_result), f, indent=4)
            
            print(f"[LOSO] Test subject {test_subject}: Acc={test_acc:.4f}, F1={test_f1:.4f}")
            
        except Exception as e:
            logger.error(f"Error in fold {test_subject}: {e}")
            traceback.print_exc()
            all_results.append({
                "test_subject": test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })
        
        import gc
        gc.collect()
    
    return {"individual_results": all_results}


def main():
    EXPERIMENT_NAME = "exp_25_svm_linear_on_powerful_features_with_combined_nois_loso"
    HYPOTHESIS_ID = "3b1480c0-aa08-4cad-b277-966e3654c011"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")
    
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    else:
        ALL_SUBJECTS = ['DB2_s1', 'DB2_s12', 'DB2_s15', 'DB2_s28', 'DB2_s39']

    EXERCISES = ["E1"]
    
    proc_cfg = ProcessingConfig(
        window_size=500,
        window_overlap=0,
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
        device="cpu",
        use_handcrafted_features=True,
        handcrafted_feature_set="powerful",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_linear",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: SVM-Linear | Features: Powerful | Augmentation: noise+time_warp+rotation")
    print(f"LOSO n={len(ALL_SUBJECTS)} subjects: {ALL_SUBJECTS}")
    
    results = run_loso_experiment(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        all_subjects=ALL_SUBJECTS,
        exercises=EXERCISES,
        proc_cfg=proc_cfg,
        split_cfg=split_cfg,
        train_cfg=train_cfg,
        noise_std=0.01,
        time_warp_max=0.1,
        rotation_factor=0.1,
        num_augmented_copies=2,
    )
    
    individual_results = results["individual_results"]
    
    valid_results = [r for r in individual_results 
                     if r.get("test_accuracy") is not None]
    
    aggregate = {}
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        
        aggregate = {
            "svm_linear_powerful_triple_aug": {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_subjects": len(accs),
                "min_accuracy": float(np.min(accs)),
                "max_accuracy": float(np.max(accs)),
            }
        }
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS")
        print(f"{'='*60}")
        print(f"SVM-Linear + Powerful Features + Triple Augmentation:")
        print(f"  Accuracy: {aggregate['svm_linear_powerful_triple_aug']['mean_accuracy']:.4f} "
              f"+/- {aggregate['svm_linear_powerful_triple_aug']['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate['svm_linear_powerful_triple_aug']['mean_f1_macro']:.4f} "
              f"+/- {aggregate['svm_linear_powerful_triple_aug']['std_f1_macro']:.4f}")
        print(f"  Range: [{aggregate['svm_linear_powerful_triple_aug']['min_accuracy']:.4f}, "
              f"{aggregate['svm_linear_powerful_triple_aug']['max_accuracy']:.4f}]")
        
        acc_f1_ratio = (aggregate['svm_linear_powerful_triple_aug']['mean_accuracy'] / 
                        max(aggregate['svm_linear_powerful_triple_aug']['mean_f1_macro'], 0.01))
        print(f"  Acc/F1 ratio: {acc_f1_ratio:.2f}")
    else:
        print("\nNo successful LOSO folds completed!")
    
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "model_type": "svm_linear",
        "feature_set": "powerful",
        "augmentation": "noise + time_warp + rotation",
        "augmentation_params": {
            "noise_std": 0.01,
            "time_warp_max": 0.1,
            "rotation_factor": 0.1,
            "num_augmented_copies": 2,
        },
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": individual_results,
        "experiment_date": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")
    
    # === Update hypothesis status in Qdrant ===
    try:
        from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed

        if aggregate:
            best_model_name = "svm_linear_powerful_triple_aug"
            best_metrics = aggregate[best_model_name].copy()
            best_metrics["best_model"] = best_model_name
            best_metrics["acc_f1_ratio"] = (
                best_metrics["mean_accuracy"] / max(best_metrics["mean_f1_macro"], 0.01)
            )
            mark_hypothesis_verified(
                hypothesis_id=HYPOTHESIS_ID,
                metrics=best_metrics,
                experiment_name=EXPERIMENT_NAME,
            )
            print(f"\nHypothesis {HYPOTHESIS_ID} marked as VERIFIED")
        else:
            mark_hypothesis_failed(
                hypothesis_id=HYPOTHESIS_ID,
                error_message="No successful LOSO folds completed",
            )
            print(f"\nHypothesis {HYPOTHESIS_ID} marked as FAILED")
    except ImportError:
        print("hypothesis_executor not available, skipping Qdrant update")


if __name__ == "__main__":
    main()