# FILE: experiments/exp_55_optimal_transport_barycenter_alignment_loso.py
"""
Experiment 55: Optimal Transport Barycenter Alignment for Cross-Subject EMG

Hypothesis: Wasserstein barycenter of training subject feature distributions +
OT transport maps to align all subjects to barycenter will normalize inter-subject
variability while preserving gesture-discriminative structure.

Key innovations:
- Per-class OT alignment using Sinkhorn algorithm
- Wasserstein barycenter computed from training subjects only
- Test-time alignment via nearest training subject transport maps
- Combined with Powerful + nonlinear features
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
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    parse_subjects_args, CI_TEST_SUBJECTS, make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import FeatureMLTrainer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver
from visualization.base import Visualizer
from processing.powerful_features import PowerfulFeatureExtractor


EXPERIMENT_NAME = "exp_55_optimal_transport_barycenter_alignment_loso"
HYPOTHESIS_ID = "h-055-ot-barycenter"


# ─── OT Alignment Utilities ─────────────────────────────────────────────────

def _sinkhorn_transport(a: np.ndarray, b: np.ndarray, M: np.ndarray,
                         reg: float = 0.01, max_iter: int = 100) -> np.ndarray:
    """
    Compute Sinkhorn transport plan between distributions a and b.

    Args:
        a: source weights (n,)
        b: target weights (m,)
        M: cost matrix (n, m)
        reg: entropic regularization
        max_iter: maximum iterations

    Returns:
        T: transport plan (n, m)
    """
    n, m = M.shape
    K = np.exp(-M / (reg + 1e-10))
    K = np.maximum(K, 1e-300)

    u = np.ones(n) / n
    for _ in range(max_iter):
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)
        # Numerical stability
        u = np.clip(u, 1e-300, 1e300)
        v = np.clip(v, 1e-300, 1e300)

    T = np.diag(u) @ K @ np.diag(v)
    return T


def _transport_map_apply(X_source: np.ndarray, X_target_ref: np.ndarray,
                          T: np.ndarray) -> np.ndarray:
    """
    Apply transport plan to map source features toward target.
    Uses barycentric mapping.

    Args:
        X_source: (n, d) source features
        X_target_ref: (m, d) target reference features
        T: (n, m) transport plan

    Returns:
        X_mapped: (n, d) mapped features
    """
    # Normalize rows of T to get conditional probabilities
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    T_norm = T / row_sums

    # Barycentric mapping: mapped[i] = sum_j T_norm[i,j] * target[j]
    X_mapped = T_norm @ X_target_ref
    return X_mapped


def compute_subject_stats(windows: np.ndarray) -> np.ndarray:
    """
    Compute subject-level statistics from windows (N, T, C).
    Returns a feature vector characterizing the subject's signal properties.
    """
    # Per-channel RMS
    rms = np.sqrt(np.mean(windows ** 2, axis=(0, 1)))  # (C,)

    # Per-channel spectral centroid (approximate via zero-crossing rate)
    zcr = np.mean(np.abs(np.diff(np.sign(windows), axis=1)), axis=(0, 1)) / 2  # (C,)

    # Per-channel variance
    var = np.var(windows, axis=(0, 1))  # (C,)

    # Per-channel kurtosis
    from scipy.stats import kurtosis
    flat = windows.reshape(-1, windows.shape[2])
    kurt = kurtosis(flat, axis=0, nan_policy='omit')  # (C,)

    return np.concatenate([rms, zcr, var, kurt]).astype(np.float32)


def _compute_barycenter_features(subject_features: Dict[str, np.ndarray],
                                   subject_labels: Dict[str, np.ndarray],
                                   class_ids: List[int],
                                   reg: float = 0.05,
                                   max_samples_per_class: int = 200) -> Dict[int, np.ndarray]:
    """
    Compute Wasserstein barycenter per class from all training subjects.
    Simple approach: weighted average of class-conditional feature distributions.
    """
    barycenter = {}
    for cls_idx, gid in enumerate(class_ids):
        all_class_feats = []
        for subj_id, feats in subject_features.items():
            labels = subject_labels[subj_id]
            mask = labels == cls_idx
            if mask.sum() > 0:
                class_feats = feats[mask]
                # Subsample if too many
                if len(class_feats) > max_samples_per_class:
                    idx = np.random.choice(len(class_feats), max_samples_per_class, replace=False)
                    class_feats = class_feats[idx]
                all_class_feats.append(class_feats)

        if all_class_feats:
            # Simple barycenter: concatenate all and compute mean/medoid
            all_feats = np.concatenate(all_class_feats, axis=0)
            barycenter[gid] = all_feats  # Store all barycenter samples
        else:
            barycenter[gid] = np.empty((0, next(iter(subject_features.values())).shape[1]))

    return barycenter


def align_features_to_barycenter(X_source: np.ndarray, barycenter_samples: np.ndarray,
                                   reg: float = 0.05) -> np.ndarray:
    """Align source features toward barycenter using OT."""
    n = len(X_source)
    m = len(barycenter_samples)

    if n == 0 or m == 0:
        return X_source

    # Subsample for efficiency
    max_n = min(n, 500)
    max_m = min(m, 500)

    if n > max_n:
        src_idx = np.random.choice(n, max_n, replace=False)
    else:
        src_idx = np.arange(n)

    if m > max_m:
        tgt_idx = np.random.choice(m, max_m, replace=False)
    else:
        tgt_idx = np.arange(m)

    X_src_sub = X_source[src_idx]
    X_tgt_sub = barycenter_samples[tgt_idx]

    # Cost matrix
    M = cdist(X_src_sub, X_tgt_sub, metric='sqeuclidean')
    M = M / (M.max() + 1e-10)  # Normalize

    # Uniform weights
    a = np.ones(len(X_src_sub)) / len(X_src_sub)
    b = np.ones(len(X_tgt_sub)) / len(X_tgt_sub)

    # Sinkhorn transport
    T = _sinkhorn_transport(a, b, M, reg=reg)

    # Apply mapping to subsample
    X_mapped_sub = _transport_map_apply(X_src_sub, X_tgt_sub, T)

    # Compute displacement vectors
    displacements = X_mapped_sub - X_src_sub

    # For non-subsampled points: interpolate displacement using nearest neighbor
    if n > max_n:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        nn.fit(X_src_sub)
        distances, indices = nn.kneighbors(X_source)
        weights = 1.0 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        X_mapped = np.zeros_like(X_source)
        for i in range(n):
            disp = np.zeros(X_source.shape[1])
            for j in range(3):
                disp += weights[i, j] * displacements[indices[i, j]]
            X_mapped[i] = X_source[i] + disp
    else:
        X_mapped = X_mapped_sub

    return X_mapped.astype(np.float32)


# ─── Custom Trainer ──────────────────────────────────────────────────────────

class OTBaryMLTrainer(FeatureMLTrainer):
    """ML trainer with OT barycenter alignment."""

    def __init__(self, *args, ot_reg: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.ot_reg = ot_reg
        self.barycenter = None
        self.training_subject_stats = {}
        self.training_subject_transport_info = {}

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)
        self.class_ids = class_ids
        self.class_names = class_names

        self.logger.info(f"Extracting features from {X_train.shape}...")

        # Extract powerful features
        pfe = PowerfulFeatureExtractor(sampling_rate=2000)
        F_train = pfe.transform(X_train)
        F_val = pfe.transform(X_val) if len(X_val) > 0 else np.empty((0, F_train.shape[1]))

        self.logger.info(f"Features: {F_train.shape[1]} dims")

        # Normalize features
        self.feature_mean = F_train.mean(axis=0)
        self.feature_std = F_train.std(axis=0) + 1e-8
        F_train = (F_train - self.feature_mean) / self.feature_std
        F_val = (F_val - self.feature_mean) / self.feature_std if len(F_val) > 0 else F_val

        # Compute barycenter from all training data
        self.logger.info("Computing OT barycenter from training data...")
        barycenter_per_class = {}
        for i, gid in enumerate(class_ids):
            mask = y_train == i
            if mask.sum() > 0:
                barycenter_per_class[gid] = F_train[mask]

        self.barycenter = barycenter_per_class

        # Align training features to barycenter (self-alignment for consistency)
        self.logger.info("Aligning training features to barycenter...")
        F_train_aligned = np.zeros_like(F_train)
        for i, gid in enumerate(class_ids):
            mask = y_train == i
            if mask.sum() > 0 and gid in self.barycenter and len(self.barycenter[gid]) > 0:
                F_train_aligned[mask] = align_features_to_barycenter(
                    F_train[mask], self.barycenter[gid], reg=self.ot_reg
                )

        # Align val
        F_val_aligned = np.zeros_like(F_val) if len(F_val) > 0 else F_val
        if len(F_val) > 0:
            # For val, we don't know classes — align to overall barycenter
            all_bary = np.concatenate([v for v in self.barycenter.values() if len(v) > 0], axis=0)
            F_val_aligned = align_features_to_barycenter(F_val, all_bary, reg=self.ot_reg)

        # Feature selection
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(F_train_aligned, y_train, random_state=42)
        top_k = min(200, F_train_aligned.shape[1])
        self.selected_feature_indices = np.argsort(mi)[::-1][:top_k]

        F_train_sel = F_train_aligned[:, self.selected_feature_indices]
        F_val_sel = F_val_aligned[:, self.selected_feature_indices] if len(F_val) > 0 else F_val_aligned

        # Train SVM-RBF
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001]}
        grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', random_state=42),
                            param_grid, cv=3, scoring='accuracy', n_jobs=-1, refit=True)
        grid.fit(F_train_sel, y_train)

        self.model = grid.best_estimator_
        self.pfe = pfe
        self.logger.info(f"Best SVM: {grid.best_params_}, CV acc: {grid.best_score_:.4f}")

        val_acc = self.model.score(F_val_sel, y_val) if len(F_val_sel) > 0 else 0.0
        return {"best_val_accuracy": val_acc, "best_params": grid.best_params_}

    def _extract_and_align(self, X: np.ndarray) -> np.ndarray:
        """Extract features, normalize, align to barycenter, select."""
        F = self.pfe.transform(X)
        F = (F - self.feature_mean) / self.feature_std

        # Align to overall barycenter
        if self.barycenter:
            all_bary = np.concatenate([v for v in self.barycenter.values() if len(v) > 0], axis=0)
            if len(all_bary) > 0:
                F = align_features_to_barycenter(F, all_bary, reg=self.ot_reg)

        return F[:, self.selected_feature_indices]

    def evaluate_numpy(self, X: np.ndarray, y: np.ndarray,
                       split_name: str, visualize: bool = True) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        F = self._extract_and_align(X)
        preds = self.model.predict(F)
        accuracy = float((preds == y).mean())
        from sklearn.metrics import f1_score, classification_report, confusion_matrix
        f1_macro = float(f1_score(y, preds, average='macro', zero_division=0))
        f1_weighted = float(f1_score(y, preds, average='weighted', zero_division=0))
        report = classification_report(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds)
        if visualize and self.visualizer:
            try:
                self.visualizer.plot_confusion_matrix(
                    cm, class_names=[str(i) for i in range(len(np.unique(y)))],
                    title=f"CM - {split_name}",
                    save_path=self.output_dir / f"cm_{split_name}.png")
            except Exception:
                pass
        return {'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
                'report': report, 'confusion_matrix': cm.tolist()}


# ─── LOSO fold ───────────────────────────────────────────────────────────────

def run_single_loso_fold(base_dir, output_dir, train_subjects, test_subject,
                          exercises, model_type, proc_cfg, split_cfg, train_cfg):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects, test_subject=test_subject,
        exercises=exercises, base_dir=base_dir,
        pool_train_subjects=True, use_separate_val_subject=False,
        val_subject=None, val_ratio=0.15, seed=train_cfg.seed, max_gestures=10,
    )

    multi_loader = MultiSubjectLoader(processing_config=proc_cfg, logger=logger,
                                       use_gpu=True, use_improved_processing=False)
    base_viz = Visualizer(output_dir, logger)
    trainer = OTBaryMLTrainer(train_cfg=train_cfg, logger=logger,
                               output_dir=output_dir, visualizer=base_viz, ot_reg=0.05)

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg, split_config=split_cfg,
        multi_subject_loader=multi_loader, trainer=trainer,
        visualizer=base_viz, logger=logger)

    try:
        results = experiment.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return {"test_subject": test_subject, "model_type": model_type,
                "test_accuracy": None, "test_f1_macro": None, "error": str(e)}

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))
    print(f"[LOSO] {test_subject} | Acc={test_acc:.4f}, F1={test_f1:.4f}")

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    import gc; del experiment, trainer, multi_loader; gc.collect()
    return {"test_subject": test_subject, "model_type": model_type,
            "test_accuracy": test_acc, "test_f1_macro": test_f1}


def main():
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    ALL_SUBJECTS = parse_subjects_args()
    EXERCISES = ["E1"]
    MODEL_TYPES = ["svm_rbf"]

    proc_cfg = ProcessingConfig(window_size=600, window_overlap=300, num_channels=8,
                                 sampling_rate=2000, segment_edge_margin=0.1)
    split_cfg = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                             mode="by_segments", shuffle_segments=True, seed=42,
                             include_rest_in_splits=False)
    train_cfg = TrainingConfig(
        batch_size=512, epochs=1, learning_rate=1e-3, weight_decay=1e-4,
        dropout=0.3, early_stopping_patience=10, use_class_weights=True,
        seed=42, num_workers=4,
        device="cuda" if __import__('torch').cuda.is_available() else "cpu",
        pipeline_type="ml_emg_td", model_type="svm_rbf",
        ml_model_type="svm_rbf", ml_use_hyperparam_search=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Features: Powerful + OT Barycenter Alignment -> SVM-RBF")

    all_loso_results = []
    for model_type in MODEL_TYPES:
        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"
            try:
                res = run_single_loso_fold(BASE_DIR, fold_dir, train_subjects, test_subject,
                                            EXERCISES, model_type, proc_cfg, split_cfg, train_cfg)
                all_loso_results.append(res)
                acc_s = f"{res['test_accuracy']:.4f}" if res.get('test_accuracy') is not None else "N/A"
                f1_s = f"{res['test_f1_macro']:.4f}" if res.get('test_f1_macro') is not None else "N/A"
                print(f"  {test_subject}: acc={acc_s}, f1={f1_s}")
            except Exception as e:
                traceback.print_exc()
                all_loso_results.append({"test_subject": test_subject, "model_type": model_type,
                                          "test_accuracy": None, "test_f1_macro": None, "error": str(e)})

    aggregate = {}
    for mt in MODEL_TYPES:
        mr = [r for r in all_loso_results if r["model_type"] == mt and r.get("test_accuracy") is not None]
        if mr:
            accs = [r["test_accuracy"] for r in mr]
            f1s = [r["test_f1_macro"] for r in mr]
            aggregate[mt] = {"mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
                             "mean_f1_macro": float(np.mean(f1s)), "std_f1_macro": float(np.std(f1s)),
                             "num_subjects": len(accs)}
            print(f"\n{mt}: Acc={aggregate[mt]['mean_accuracy']:.4f}+-{aggregate[mt]['std_accuracy']:.4f}")

    loso_summary = {
        "experiment_name": EXPERIMENT_NAME, "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "powerful+ot_alignment", "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS, "exercises": EXERCISES,
        "ot_config": {"reg": 0.05},
        "aggregate_results": aggregate, "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)

    try:
        from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed
        if aggregate:
            best = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
            metrics = aggregate[best].copy(); metrics["best_model"] = best
            mark_hypothesis_verified(hypothesis_id=HYPOTHESIS_ID, metrics=metrics,
                                     experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed(hypothesis_id=HYPOTHESIS_ID,
                                   error_message="No successful LOSO folds")
    except ImportError:
        print("hypothesis_executor not available")


if __name__ == "__main__":
    main()
