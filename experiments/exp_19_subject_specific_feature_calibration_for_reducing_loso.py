# FILE: experiments/exp_19_subject_specific_feature_calibration_for_reducing_loso.py
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
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig

from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment

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


class MMDFeatureCalibrator:
    """
    MMD-based domain adaptation for feature calibration.
    Aligns source (training subjects) features to target (test subject) distribution
    using Maximum Mean Discrepancy minimization.
    """
    
    def __init__(
        self,
        kernel_bandwidth: float = 1.0,
        regularization: float = 1e-4,
        n_components: Optional[int] = None,
        calibration_samples: int = 100,
    ):
        self.kernel_bandwidth = kernel_bandwidth
        self.regularization = regularization
        self.n_components = n_components
        self.calibration_samples = calibration_samples
        self.transformation_matrix = None
        self.source_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float = None) -> np.ndarray:
        """Compute RBF kernel between X and Y."""
        if gamma is None:
            gamma = 1.0 / (2 * self.kernel_bandwidth ** 2)
        
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        XY = np.dot(X, Y.T)
        
        distances = XX - 2 * XY + YY.T
        return np.exp(-gamma * distances)
    
    def _compute_mmd(self, X_source: np.ndarray, X_target: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy between source and target."""
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]
        
        K_ss = self._rbf_kernel(X_source, X_source)
        K_tt = self._rbf_kernel(X_target, X_target)
        K_st = self._rbf_kernel(X_source, X_target)
        
        mmd = (np.sum(K_ss) / (n_source ** 2) 
               - 2 * np.sum(K_st) / (n_source * n_target) 
               + np.sum(K_tt) / (n_target ** 2))
        
        return mmd
    
    def _compute_transformation_matrix(
        self, 
        X_source: np.ndarray, 
        X_target: np.ndarray
    ) -> np.ndarray:
        """
        Compute the transformation matrix for domain adaptation using
        subspace alignment with MMD regularization.
        """
        n_features = X_source.shape[1]
        
        # Compute covariance matrices
        C_source = np.cov(X_source.T) + self.regularization * np.eye(n_features)
        C_target = np.cov(X_target.T) + self.regularization * np.eye(n_features)
        
        # Eigendecomposition
        eig_vals_s, eig_vecs_s = np.linalg.eigh(C_source)
        eig_vals_t, eig_vecs_t = np.linalg.eigh(C_target)
        
        # Sort by eigenvalues (descending)
        idx_s = np.argsort(eig_vals_s)[::-1]
        idx_t = np.argsort(eig_vals_t)[::-1]
        
        eig_vecs_s = eig_vecs_s[:, idx_s]
        eig_vecs_t = eig_vecs_t[:, idx_t]
        
        # Determine number of components
        n_comp = self.n_components if self.n_components else min(n_features, 100)
        n_comp = min(n_comp, n_features)
        
        # Get principal subspaces
        P_source = eig_vecs_s[:, :n_comp]
        P_target = eig_vecs_t[:, :n_comp]
        
        # Compute alignment transformation
        # M = P_source @ P_source.T (project to source subspace)
        # Then align to target subspace
        M = P_target @ P_target.T @ P_source @ P_source.T
        
        # Add identity residual for stability
        M = 0.8 * M + 0.2 * np.eye(n_features)
        
        return M
    
    def fit(
        self, 
        X_source: np.ndarray, 
        X_target: np.ndarray,
        y_source: Optional[np.ndarray] = None
    ) -> 'MMDFeatureCalibrator':
        """
        Fit the calibrator by learning the transformation to align
        source features to target domain.
        """
        # Sample if too many points (for computational efficiency)
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]
        
        if n_source > self.calibration_samples * 10:
            idx = np.random.choice(n_source, self.calibration_samples * 10, replace=False)
            X_source_sample = X_source[idx]
        else:
            X_source_sample = X_source
            
        if n_target > self.calibration_samples:
            idx = np.random.choice(n_target, self.calibration_samples, replace=False)
            X_target_sample = X_target[idx]
        else:
            X_target_sample = X_target
        
        # Standardize both domains
        X_source_scaled = self.source_scaler.fit_transform(X_source_sample)
        X_target_scaled = self.target_scaler.fit_transform(X_target_sample)
        
        # Compute transformation matrix
        self.transformation_matrix = self._compute_transformation_matrix(
            X_source_scaled, X_target_scaled
        )
        
        # Compute initial MMD for logging
        initial_mmd = self._compute_mmd(X_source_scaled, X_target_scaled)
        print(f"  [MMD Calibrator] Initial MMD: {initial_mmd:.6f}")
        
        return self
    
    def transform_source(self, X_source: np.ndarray) -> np.ndarray:
        """Transform source features to target-aligned space."""
        if self.transformation_matrix is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        X_scaled = self.source_scaler.transform(X_source)
        X_transformed = X_scaled @ self.transformation_matrix
        return X_transformed
    
    def transform_target(self, X_target: np.ndarray) -> np.ndarray:
        """Transform target features (standardization only)."""
        return self.target_scaler.transform(X_target)
    
    def fit_transform(
        self, 
        X_source: np.ndarray, 
        X_target: np.ndarray,
        y_source: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform both source and target features."""
        self.fit(X_source, X_target, y_source)
        return self.transform_source(X_source), self.transform_target(X_target)


class CalibratedSVMTrainer:
    """
    Custom trainer that combines MMD-based feature calibration with SVM.
    """
    
    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        visualizer,
    ):
        self.cfg = train_cfg
        self.logger = logger
        self.output_dir = output_dir
        self.visualizer = visualizer
        self.model = None
        self.calibrator = None
        self.feature_scaler = None
        self.feature_extractor = None
        self.class_ids = None
        self.class_names = None

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """Adapter: unpack splits dict, extract features, then call train()."""
        from processing.powerful_features import PowerfulFeatureExtractor

        train_d = {gid: arr for gid, arr in splits["train"].items() if len(arr) > 0}
        val_d   = {gid: arr for gid, arr in splits.get("val", {}).items() if len(arr) > 0}
        test_d  = {gid: arr for gid, arr in splits.get("test", {}).items() if len(arr) > 0}

        self.class_ids = sorted(train_d.keys())
        self.class_names = {
            gid: ("REST" if gid == 0 else f"Gesture {gid}") for gid in self.class_ids
        }
        val_d  = {gid: arr for gid, arr in val_d.items()  if gid in self.class_ids}
        test_d = {gid: arr for gid, arr in test_d.items() if gid in self.class_ids}

        def concat_xy(dct):
            X_list, y_list = [], []
            for i, gid in enumerate(self.class_ids):
                if gid in dct:
                    X_list.append(dct[gid])
                    y_list.append(np.full((len(dct[gid]),), i, dtype=np.int64))
            if not X_list:
                return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)
            return (
                np.concatenate(X_list, axis=0).astype(np.float32),
                np.concatenate(y_list, axis=0),
            )

        X_train, y_train = concat_xy(train_d)
        X_val,   y_val   = concat_xy(val_d)
        X_test,  y_test  = concat_xy(test_d)

        if self.feature_extractor is None:
            self.feature_extractor = PowerfulFeatureExtractor(
                sampling_rate=2000,
                logger=self.logger,
                feature_set="powerful",
                n_jobs=-1,
                use_torch=True,
                device="cuda",
                gpu_batch_size=4096,
            )

        self.logger.info("Extracting features from train/val/test windows...")
        X_train_feat = self.feature_extractor.transform(X_train)
        X_val_feat   = self.feature_extractor.transform(X_val)  if len(X_val)  > 0 else None
        X_test_feat  = self.feature_extractor.transform(X_test) if len(X_test) > 0 else None

        return self.train(
            train_windows=X_train_feat,
            train_labels=y_train,
            val_windows=X_val_feat,
            val_labels=y_val if len(y_val) > 0 else None,
            test_windows=X_test_feat,
            test_labels=y_test if len(y_test) > 0 else None,
            num_classes=len(self.class_ids),
        )

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """Evaluate on raw windows (N, T, C): extract features, calibrate, scale, predict."""
        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report, confusion_matrix
        )

        if self.feature_extractor is None or self.model is None:
            raise ValueError("Model not fitted yet — call fit() first")

        X_feat = self.feature_extractor.transform(X)

        if self.calibrator is not None:
            X_feat = self.calibrator.transform_target(X_feat)

        X_scaled = self.feature_scaler.transform(X_feat)
        y_pred = self.model.predict(X_scaled)

        acc     = accuracy_score(y, y_pred)
        f1_mac  = f1_score(y, y_pred, average="macro")
        report  = classification_report(y, y_pred, output_dict=True, zero_division=0)
        cm      = confusion_matrix(y, y_pred, labels=np.arange(len(self.class_ids)))

        if visualize and self.visualizer is not None:
            class_labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, class_labels, normalize=True, filename=f"cm_{split_name}.png"
            )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_mac),
            "report": report,
            "confusion_matrix": cm.tolist(),
        }

    def train(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        val_windows: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        test_windows: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        """
        Train SVM with MMD-based feature calibration.
        
        Args:
            train_windows: Handcrafted features from training subjects
            train_labels: Labels for training data
            val_windows: Validation features (from train subjects split)
            test_windows: Test features from target subject (used for calibration)
            test_labels: Test labels
        """
        self.logger.info("Starting Calibrated SVM Training with MMD Domain Adaptation")
        
        # Get test data for calibration (few-shot from target domain)
        test_features = kwargs.get('test_features', test_windows)
        
        if test_features is not None and len(test_features) > 0:
            self.logger.info(f"Applying MMD-based feature calibration using {len(test_features)} test samples")
            
            # Initialize and fit calibrator
            self.calibrator = MMDFeatureCalibrator(
                kernel_bandwidth=1.0,
                regularization=1e-4,
                calibration_samples=min(200, len(test_features) // 2),
            )
            
            # Fit calibrator and transform features
            train_calibrated, test_calibrated = self.calibrator.fit_transform(
                train_windows, test_features
            )
            
            # Also calibrate validation set if present
            if val_windows is not None:
                val_calibrated = self.calibrator.transform_source(val_windows)
            else:
                val_calibrated = None
                
        else:
            self.logger.warning("No test features available for calibration, using standard training")
            train_calibrated = train_windows
            test_calibrated = test_features
            val_calibrated = val_windows
        
        # Final scaling
        self.feature_scaler = StandardScaler()
        train_scaled = self.feature_scaler.fit_transform(train_calibrated)
        
        if val_calibrated is not None:
            val_scaled = self.feature_scaler.transform(val_calibrated)
        else:
            val_scaled = None
            
        if test_calibrated is not None:
            test_scaled = self.feature_scaler.transform(test_calibrated)
        else:
            test_scaled = None
        
        # Train SVM with linear kernel
        self.logger.info("Training Linear SVM...")
        
        # Use LinearSVC with calibrated probabilities
        base_svc = LinearSVC(
            C=1.0,
            penalty='l2',
            loss='squared_hinge',
            dual=False,
            multi_class='ovr',
            class_weight='balanced' if self.cfg.use_class_weights else None,
            max_iter=5000,
            random_state=self.cfg.seed,
        )
        
        # Wrap with calibrated classifier for probability estimates
        self.model = CalibratedClassifierCV(base_svc, cv=3, method='sigmoid')
        self.model.fit(train_scaled, train_labels)
        
        # Evaluate on validation set
        val_metrics = {}
        if val_scaled is not None and val_labels is not None:
            val_pred = self.model.predict(val_scaled)
            val_acc = np.mean(val_pred == val_labels)
            val_metrics['accuracy'] = float(val_acc)
            self.logger.info(f"Validation Accuracy: {val_acc:.4f}")
        
        # Evaluate on test set
        test_metrics = {}
        if test_scaled is not None and test_labels is not None:
            test_pred = self.model.predict(test_scaled)
            test_acc = np.mean(test_pred == test_labels)
            
            # Compute per-class metrics
            from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
            
            test_f1 = f1_score(test_labels, test_pred, average='macro')
            test_precision = precision_score(test_labels, test_pred, average='macro')
            test_recall = recall_score(test_labels, test_pred, average='macro')
            
            test_metrics = {
                'accuracy': float(test_acc),
                'f1_macro': float(test_f1),
                'precision_macro': float(test_precision),
                'recall_macro': float(test_recall),
            }
            
            self.logger.info(f"Test Accuracy: {test_acc:.4f}, F1-macro: {test_f1:.4f}")
            
            # Save confusion matrix
            cm = confusion_matrix(test_labels, test_pred)
            test_metrics['confusion_matrix'] = cm.tolist()
        
        return {
            'train_metrics': {'accuracy': float(self.model.score(train_scaled, train_labels))},
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model_type': 'calibrated_svm_linear',
            'calibration_applied': self.calibrator is not None,
        }
    
    def predict(self, windows: np.ndarray, **kwargs) -> np.ndarray:
        """Predict labels for new windows."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Apply calibration transformation if available
        if self.calibrator is not None:
            # Use transform_target for test data
            windows = self.calibrator.transform_target(windows)
        
        windows_scaled = self.feature_scaler.transform(windows)
        return self.model.predict(windows_scaled)
    
    def predict_proba(self, windows: np.ndarray, **kwargs) -> np.ndarray:
        """Predict probabilities for new windows."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        if self.calibrator is not None:
            windows = self.calibrator.transform_target(windows)
            
        windows_scaled = self.feature_scaler.transform(windows)
        return self.model.predict_proba(windows_scaled)


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
) -> Dict:
    """
    LOSO fold with MMD-based feature calibration for SVM.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = "ml_emg_td"
    train_cfg.ml_model_type = "svm_linear"
    train_cfg.use_handcrafted_features = True
    train_cfg.handcrafted_feature_set = "powerful"

    # Save configs
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
        use_gpu=False,  # CPU for ML pipeline
        use_improved_processing=use_improved_processing,
    )

    base_viz = Visualizer(output_dir, logger)

    # Use custom calibrated SVM trainer
    trainer = CalibratedSVMTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
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
        print(f"Error in LOSO fold (test_subject={test_subject}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "approach": "ml_emg_td_calibrated",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO-Calibrated] Test subject {test_subject} | "
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
        "approach": "ml_emg_td_with_mmd_calibration",
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
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": "ml_emg_td_calibrated",
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    EXPERIMENT_NAME = "exp_19_subject_specific_feature_calibration_for_reducing_loso"
    HYPOTHESIS_ID = "df248bc2-7fdc-44c3-a9ff-48e0110bd9bc"
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
    MODEL_TYPE = "calibrated_svm_linear"

    proc_cfg = ProcessingConfig(
        window_size=500,
        window_overlap=250,
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
        aug_apply=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Approach: ML with MMD-based Feature Calibration")
    print(f"Model: Linear SVM with Powerful Features")
    print(f"LOSO n={len(ALL_SUBJECTS)} subjects")

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
            )
            all_loso_results.append(fold_res)
            if fold_res.get("test_accuracy") is not None:
                print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, f1={fold_res['test_f1_macro']:.4f}")
            else:
                print(f"  ✗ {test_subject}: error - {fold_res.get('error', 'unknown')}")
        except Exception as e:
            global_logger.error(f"Failed {test_subject}: {e}")
            traceback.print_exc()
            all_loso_results.append({
                "test_subject": test_subject,
                "model_type": MODEL_TYPE,
                "approach": "ml_emg_td_calibrated",
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })

    # Compute aggregate statistics
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        
        aggregate = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
            "num_subjects": len(accs),
            "num_failed": len(all_loso_results) - len(valid_results),
        }
        
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS:")
        print(f"  Accuracy: {aggregate['mean_accuracy']:.4f} ± {aggregate['std_accuracy']:.4f}")
        print(f"  F1-macro: {aggregate['mean_f1_macro']:.4f} ± {aggregate['std_f1_macro']:.4f}")
        print(f"  Range: [{aggregate['min_accuracy']:.4f}, {aggregate['max_accuracy']:.4f}]")
        print(f"{'='*60}")
    else:
        aggregate = {
            "mean_accuracy": None,
            "std_accuracy": None,
            "mean_f1_macro": None,
            "std_f1_macro": None,
            "num_subjects": 0,
            "num_failed": len(all_loso_results),
        }
        print("\nERROR: No successful LOSO folds completed!")

    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "powerful",
        "model": MODEL_TYPE,
        "approach": "ml_emg_td_with_mmd_calibration",
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "none",
        "calibration_method": "MMD-based domain adaptation",
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

    if aggregate.get("mean_accuracy") is not None:
        best_metrics = {
            "mean_accuracy": aggregate["mean_accuracy"],
            "std_accuracy": aggregate["std_accuracy"],
            "mean_f1_macro": aggregate["mean_f1_macro"],
            "std_f1_macro": aggregate["std_f1_macro"],
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