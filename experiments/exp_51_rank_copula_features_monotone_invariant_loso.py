# FILE: experiments/exp_51_rank_copula_features_monotone_invariant_loso.py
"""
Experiment 51: Rank-Based & Copula Features for Monotone-Invariant EMG Classification

Hypothesis: Rank transforms + copula inter-channel statistics are provably invariant
to ALL monotone transformations (amplitude, impedance, gain), covering most subject
differences.
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
from scipy import stats as scipy_stats

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


EXPERIMENT_NAME = "exp_51_rank_copula_features_monotone_invariant_loso"
HYPOTHESIS_ID = "h-051-rank-copula"


# ─── Rank & Copula Feature Extraction ───────────────────────────────────────

def _rank_transform_channel(x: np.ndarray) -> np.ndarray:
    """Rank-transform a 1D signal to uniform(0,1) marginals."""
    n = len(x)
    if n == 0:
        return x
    ranks = scipy_stats.rankdata(x, method='average')
    return ranks / (n + 1)  # Avoid exact 0 and 1


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau rank correlation (fast O(n log n))."""
    try:
        tau, _ = scipy_stats.kendalltau(x, y)
        return 0.0 if np.isnan(tau) else tau
    except Exception:
        return 0.0


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho rank correlation."""
    try:
        rho, _ = scipy_stats.spearmanr(x, y)
        return 0.0 if np.isnan(rho) else rho
    except Exception:
        return 0.0


def _tail_dependence(u: np.ndarray, v: np.ndarray, q: float = 0.1) -> tuple:
    """Estimate upper and lower tail dependence coefficients."""
    n = len(u)
    if n == 0:
        return 0.0, 0.0
    # Lower tail: P(V <= q | U <= q)
    lower_mask = u <= q
    lower_count = lower_mask.sum()
    lower_dep = (v[lower_mask] <= q).sum() / max(lower_count, 1) if lower_count > 0 else 0.0
    # Upper tail: P(V > 1-q | U > 1-q)
    upper_mask = u >= (1 - q)
    upper_count = upper_mask.sum()
    upper_dep = (v[upper_mask] >= (1 - q)).sum() / max(upper_count, 1) if upper_count > 0 else 0.0
    return float(lower_dep), float(upper_dep)


def _rank_temporal_features(ranked: np.ndarray) -> np.ndarray:
    """Compute rank-based temporal features for one channel."""
    n = len(ranked)
    feats = []
    # 1) Number of runs (runs test statistic)
    if n > 1:
        median = np.median(ranked)
        above = ranked >= median
        runs = 1 + np.sum(above[1:] != above[:-1])
        feats.append(runs / n)
    else:
        feats.append(0.0)
    # 2) Number of inversions (normalized)
    if n > 1:
        inversions = 0
        # Efficient: use Kendall's tau relationship: tau = 1 - 4*inversions/(n*(n-1))
        tau, _ = scipy_stats.kendalltau(np.arange(n), ranked)
        inv_ratio = (1 - (tau if not np.isnan(tau) else 0.0)) / 2
        feats.append(inv_ratio)
    else:
        feats.append(0.0)
    # 3-6) Rank autocorrelation at different lags
    for lag in [1, 5, 10, 50]:
        if n > lag:
            r = np.corrcoef(ranked[:-lag], ranked[lag:])[0, 1]
            feats.append(0.0 if np.isnan(r) else r)
        else:
            feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def _quantile_features(x: np.ndarray) -> np.ndarray:
    """Quantile-based robust features for one channel."""
    if len(x) == 0:
        return np.zeros(7, dtype=np.float32)
    median = np.median(x)
    q10, q25, q75, q90 = np.percentile(x, [10, 25, 75, 90])
    iqr = q75 - q25
    mad = np.median(np.abs(x - median))
    return np.array([median, iqr, mad, q10, q25, q75, q90], dtype=np.float32)


def extract_rank_copula_features(windows: np.ndarray) -> np.ndarray:
    """
    Extract rank-based and copula features from EMG windows.

    Args:
        windows: (N, T, C) array

    Returns:
        features: (N, n_features) array
    """
    N, T, C = windows.shape
    n_pairs = C * (C - 1) // 2

    all_features = []
    for i in range(N):
        window = windows[i]  # (T, C)

        # Step 1: Rank-transform each channel
        ranked = np.zeros_like(window)
        for ch in range(C):
            ranked[:, ch] = _rank_transform_channel(window[:, ch])

        # Step 2: Copula features between channel pairs
        copula_feats = []
        for ch_a in range(C):
            for ch_b in range(ch_a + 1, C):
                tau = _kendall_tau(ranked[:, ch_a], ranked[:, ch_b])
                rho = _spearman_rho(ranked[:, ch_a], ranked[:, ch_b])
                lower_td, upper_td = _tail_dependence(ranked[:, ch_a], ranked[:, ch_b])
                copula_feats.extend([tau, rho, lower_td, upper_td])

        # Step 3: Rank-based temporal features per channel
        temporal_feats = []
        for ch in range(C):
            temporal_feats.extend(_rank_temporal_features(ranked[:, ch]))

        # Step 4: Quantile features per channel (on original signal)
        quantile_feats = []
        for ch in range(C):
            quantile_feats.extend(_quantile_features(window[:, ch]))

        all_features.append(np.concatenate([
            np.array(copula_feats, dtype=np.float32),
            np.array(temporal_feats, dtype=np.float32),
            np.array(quantile_feats, dtype=np.float32),
        ]))

    return np.array(all_features, dtype=np.float32)


# ─── Custom Trainer ──────────────────────────────────────────────────────────

class RankCopulaMLTrainer(FeatureMLTrainer):
    """FeatureMLTrainer with rank/copula + powerful features."""

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)
        self.class_ids = class_ids
        self.class_names = class_names

        self.logger.info(f"Extracting rank/copula features from {X_train.shape}...")
        rc_train = extract_rank_copula_features(X_train)
        rc_val = extract_rank_copula_features(X_val) if len(X_val) > 0 else np.empty((0, rc_train.shape[1]))
        self.logger.info(f"Rank/copula features: {rc_train.shape[1]} dims")

        pfe = PowerfulFeatureExtractor(sampling_rate=2000)
        pf_train = pfe.transform(X_train)
        pf_val = pfe.transform(X_val) if len(X_val) > 0 else np.empty((0, pf_train.shape[1]))
        self.logger.info(f"Powerful features: {pf_train.shape[1]} dims")

        F_train = np.concatenate([rc_train, pf_train], axis=1)
        F_val = np.concatenate([rc_val, pf_val], axis=1) if len(X_val) > 0 else np.empty((0, F_train.shape[1]))

        # Normalize
        self.feature_mean = F_train.mean(axis=0)
        self.feature_std = F_train.std(axis=0) + 1e-8
        F_train = (F_train - self.feature_mean) / self.feature_std
        F_val = (F_val - self.feature_mean) / self.feature_std if len(F_val) > 0 else F_val

        # Feature selection
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(F_train, y_train, random_state=42)
        top_k = min(250, F_train.shape[1])
        self.selected_feature_indices = np.argsort(mi)[::-1][:top_k]
        F_train = F_train[:, self.selected_feature_indices]
        F_val = F_val[:, self.selected_feature_indices] if len(F_val) > 0 else F_val

        # SVM-RBF
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001]}
        grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', random_state=42),
                            param_grid, cv=3, scoring='accuracy', n_jobs=-1, refit=True)
        grid.fit(F_train, y_train)

        self.model = grid.best_estimator_
        self.pfe = pfe
        self.logger.info(f"Best SVM: {grid.best_params_}, CV acc: {grid.best_score_:.4f}")

        val_acc = self.model.score(F_val, y_val) if len(F_val) > 0 else 0.0
        return {"best_val_accuracy": val_acc, "best_params": grid.best_params_}

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        rc = extract_rank_copula_features(X)
        pf = self.pfe.transform(X)
        F = np.concatenate([rc, pf], axis=1)
        F = (F - self.feature_mean) / self.feature_std
        return F[:, self.selected_feature_indices]

    def evaluate_numpy(self, X: np.ndarray, y: np.ndarray,
                       split_name: str, visualize: bool = True) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        F = self._extract_features(X)
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
    trainer = RankCopulaMLTrainer(train_cfg=train_cfg, logger=logger,
                                  output_dir=output_dir, visualizer=base_viz)

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
    print(f"Features: Rank/Copula + Powerful -> SVM-RBF")

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
        "feature_set": "rank_copula+powerful", "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS, "exercises": EXERCISES,
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
