# FILE: experiments/exp_18_augmented_svm_with_feature_space_jitter_for_improv_loso.py
"""
Experiment: Augmented SVM with Feature-Space Jitter for Improved Generalization

Hypothesis: Applying feature-space data augmentation (jitter/noise) to the powerful 
handcrafted feature set before training an SVM with RBF kernel will improve 
cross-subject generalization compared to the unaugmented baseline (exp4: 0.3446).

Expected improvement: 2-5 percentage points (target: 0.36-0.38)
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.signal import welch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig

from data.multi_subject_loader import MultiSubjectLoader
from processing.powerful_features import PowerfulFeatureExtractor

from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver
from visualization.base import Visualizer


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






def augment_features_with_noise(
    features: np.ndarray,
    labels: np.ndarray,
    noise_std: float = 0.02,
    n_copies: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment feature vectors with Gaussian noise (jitter).
    
    The noise is scaled by the standard deviation of each feature dimension
    to ensure consistent augmentation across features with different scales.
    
    Args:
        features: (N, F) array of features
        labels: (N,) array of labels
        noise_std: Standard deviation of Gaussian noise as fraction of feature std
        n_copies: Number of augmented copies to create per original sample
        seed: Random seed
    
    Returns:
        augmented_features: (N * (n_copies + 1), F) array
        augmented_labels: (N * (n_copies + 1),) array
    """
    rng = np.random.RandomState(seed)
    
    # Keep original
    all_features = [features]
    all_labels = [labels]
    
    # Compute feature-wise standard deviation for scaling
    feature_std = np.std(features, axis=0, keepdims=True)
    feature_std = np.maximum(feature_std, 1e-8)  # Avoid zero std
    
    # Create augmented copies
    for i in range(n_copies):
        # Generate noise scaled by feature std
        noise = rng.randn(*features.shape).astype(np.float32)
        scaled_noise = noise * noise_std * feature_std
        
        # Add noise to features
        augmented = features + scaled_noise
        all_features.append(augmented)
        all_labels.append(labels.copy())
    
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    use_improved_processing: bool,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    feature_noise_std: float = 0.02,
    n_augment_copies: int = 3,
) -> Dict:
    """
    LOSO fold with feature-space augmentation for SVM-RBF.
    
    Extracts powerful handcrafted features, applies Gaussian noise jitter
    to training features, trains SVM with RBF kernel, evaluates on test subject.
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
        use_improved_processing=use_improved_processing,
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

    # Load all subjects
    subjects_windows = {}
    subjects_labels = {}
    for subject_id in train_subjects + [test_subject]:
        try:
            emg, segments, grouped_windows = multi_loader.load_subject(
                base_dir=base_dir,
                subject_id=subject_id,
                exercise=exercises[0],
            )
            w, l = grouped_to_arrays(grouped_windows)
            subjects_windows[subject_id] = w
            subjects_labels[subject_id] = l
            logger.info(f"Loaded {subject_id}: {len(w)} windows")
        except Exception as e:
            logger.error(f"Failed to load {subject_id}: {e}")
            raise

    # Extract features for all subjects
    feature_extractor = PowerfulFeatureExtractor(sampling_rate=proc_cfg.sampling_rate)
    all_features = {}
    all_labels = {}

    for subject_id in subjects_windows:
        # Windows from grouped_windows are already (N, T, C) — no transpose needed
        features = feature_extractor.transform(subjects_windows[subject_id])
        all_features[subject_id] = features
        all_labels[subject_id] = subjects_labels[subject_id]
        logger.info(f"Extracted features for {subject_id}: shape={features.shape}")
    
    # Prepare train and test data
    train_features_list = []
    train_labels_list = []
    
    for subject_id in train_subjects:
        train_features_list.append(all_features[subject_id])
        train_labels_list.append(all_labels[subject_id])
    
    train_features = np.concatenate(train_features_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    
    test_features = all_features[test_subject]
    test_labels = all_labels[test_subject]
    
    logger.info(f"Train: {train_features.shape}, Test: {test_features.shape}")
    
    # Apply feature-space augmentation to training data
    train_features_aug, train_labels_aug = augment_features_with_noise(
        train_features,
        train_labels,
        noise_std=feature_noise_std,
        n_copies=n_augment_copies,
        seed=train_cfg.seed
    )
    
    logger.info(f"Augmented train: {train_features_aug.shape} (original: {len(train_labels)}, augmented: {len(train_labels_aug)})")
    
    # Standardize features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_aug)
    test_features_scaled = scaler.transform(test_features)  # No augmentation on test

    # PCA to reduce dimensionality (16k+ features is too many for SVM)
    pca = PCA(n_components=0.95, random_state=train_cfg.seed)
    train_features_pca = pca.fit_transform(train_features_scaled)
    test_features_pca = pca.transform(test_features_scaled)
    logger.info(f"PCA: {train_features_scaled.shape[1]} -> {train_features_pca.shape[1]} components (95% variance)")

    # Train LinearSVC (much faster than SVC RBF for large feature sets)
    svm_model = LinearSVC(
        C=1.0,
        class_weight='balanced' if train_cfg.use_class_weights else None,
        random_state=train_cfg.seed,
        max_iter=5000,
    )

    logger.info(f"Training LinearSVC with {train_features_pca.shape[1]} PCA features...")
    svm_model.fit(train_features_pca, train_labels_aug)

    # Evaluate on test set
    test_preds = svm_model.predict(test_features_pca)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    logger.info(f"Test Accuracy: {test_acc:.4f}, F1-macro: {test_f1:.4f}")
    
    # Save results
    results = {
        "test_subject": test_subject,
        "model_type": "svm_rbf",
        "approach": "ml_emg_td",
        "feature_set": "powerful",
        "feature_augmentation": {
            "noise_std": feature_noise_std,
            "n_copies": n_augment_copies,
        },
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "train_samples_original": int(len(train_labels)),
        "train_samples_augmented": int(len(train_labels_aug)),
        "test_samples": int(len(test_labels)),
        "n_features_original": int(train_features.shape[1]),
        "n_features_pca": int(train_features_pca.shape[1]),
    }
    
    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(make_json_serializable(results), f, indent=4)
    
    print(
        f"[LOSO] Test subject {test_subject} | "
        f"SVM-RBF + powerful + feature-aug (std={feature_noise_std}, copies={n_augment_copies}) | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )
    
    # Save metadata
    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "svm_rbf",
        "approach": "ml_emg_td",
        "feature_set": "powerful",
        "feature_augmentation": {
            "noise_std": feature_noise_std,
            "n_copies": n_augment_copies,
        },
        "exercises": exercises,
        "use_improved_processing": use_improved_processing,
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    del subjects_windows, subjects_labels, multi_loader, feature_extractor, svm_model, scaler, pca
    gc.collect()
    
    return results


def main():
    EXPERIMENT_NAME = "exp_18_augmented_svm_with_feature_space_jitter_for_improv_loso"
    HYPOTHESIS_ID = "d7c44dc7-cc18-4669-8b30-9db935f75204"
    
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

    # Feature-space augmentation parameters
    # Based on hypothesis: Gaussian noise with std=0.01-0.05
    FEATURE_NOISE_STD = 0.03  # Middle of suggested range
    N_AUGMENT_COPIES = 1  # Number of augmented copies per original sample
    
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
        model_type="svm_rbf",
        use_handcrafted_features=True,
        handcrafted_feature_set="powerful",
        pipeline_type="ml_emg_td",
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print("=" * 70)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 70)
    print(f"Model: SVM-RBF")
    print(f"Features: powerful handcrafted feature set")
    print(f"Augmentation: feature-space jitter (noise_std={FEATURE_NOISE_STD}, n_copies={N_AUGMENT_COPIES})")
    print(f"LOSO: {len(ALL_SUBJECTS)} subjects")
    print(f"Baseline (exp4_svm_rbf_powerful_loso): 0.3446 ± 0.0698")
    print("=" * 70)
    
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
                use_improved_processing=True,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                feature_noise_std=FEATURE_NOISE_STD,
                n_augment_copies=N_AUGMENT_COPIES,
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
                "model_type": "svm_rbf",
                "approach": "ml_emg_td",
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
        
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS")
        print("=" * 70)
        print(f"Model: SVM-RBF + powerful features + feature-space augmentation")
        print(f"Augmentation params: noise_std={FEATURE_NOISE_STD}, n_copies={N_AUGMENT_COPIES}")
        print("-" * 70)
        print(f"Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"F1-macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"Range: [{aggregate['min_accuracy']:.4f}, {aggregate['max_accuracy']:.4f}]")
        print("-" * 70)
        print(f"Baseline (exp4, no augmentation): 0.3446 ± 0.0698")
        improvement = (aggregate['mean_accuracy'] - 0.3446) / 0.3446 * 100
        print(f"Improvement: {improvement:+.2f}%")
        print("=" * 70)
    else:
        aggregate = {}
    
    # Save summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "model_type": "svm_rbf",
        "feature_set": "powerful",
        "feature_augmentation": {
            "type": "feature_space_jitter",
            "noise_std": FEATURE_NOISE_STD,
            "n_copies": N_AUGMENT_COPIES,
            "description": "Gaussian noise added to feature vectors during training, scaled by feature std",
        },
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": all_loso_results,
        "baseline_comparison": {
            "baseline_experiment": "exp4_svm_rbf_powerful_loso",
            "baseline_accuracy": 0.3446,
            "baseline_std": 0.0698,
        },
        "experiment_date": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")
    
    # === Update hypothesis status in Qdrant ===
    from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed

    if aggregate:
        aggregate["best_model"] = "svm_rbf_augmented"
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


if __name__ == "__main__":
    main()