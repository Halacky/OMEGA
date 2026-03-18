#!/usr/bin/env python3
"""
Experiment 45b: Verification of Personalized Window Segmentation
=================================================================

Verifies experiment 45 results by eliminating methodological confounds.

Problems identified in exp_45:
  1. Test sets differ across methods (different window counts per class)
  2. Baseline lacks PCA/feature selection (weaker than exp_4 SVM-RBF)
  3. Accuracy comparison is not apples-to-apples due to different class balance

Verification design (factorial):
  All methods use the SAME fixed test set (baseline 600/300 windows).
  Only training windowing varies.

  | ID | Model name                    | Train windows       | Test windows  | PCA  |
  |----|-------------------------------|---------------------|---------------|------|
  | A  | baseline                      | 600/300 all         | 600/300       | No   |
  | B  | global_best_train             | 800/0.75 all        | 600/300       | No   |
  | C  | personalized_train            | per-gesture optimal | 600/300       | No   |
  | D  | global_best_matched           | 800/0.75 all        | 800/0.75      | No   |
  | E  | personalized_matched          | per-gesture optimal | per-gesture   | No   |
  | F  | baseline_pca                  | 600/300 all         | 600/300       | Yes  |
  | G  | global_best_train_pca         | 800/0.75 all        | 600/300       | Yes  |
  | H  | personalized_train_pca        | per-gesture optimal | 600/300       | Yes  |

  Key comparisons:
    B - A  = effect of more/larger training windows only
    C - A  = effect of per-gesture training optimization only
    E - A  = original exp_45 delta (contains test-set confound)
    E - C  = contribution of test-set difference (confound magnitude)
    F - A  = effect of PCA alone
    H - F  = effect of per-gesture training with PCA

  Also reports per-method:
    - balanced_accuracy (= macro-averaged recall, immune to class imbalance)
    - window count per class (diagnostic)
    - repetition-level majority-vote accuracy (secondary metric)

Visualizations:
  1. factorial_comparison.png    — bar chart of all 8 models (acc + f1 + balanced_acc)
  2. confound_decomposition.png  — stacked bar showing gain sources (train vs test effect)
  3. window_count_per_class.png  — per-class window count across methods (shows imbalance)
  4. per_fold_heatmap.png        — heatmap of all models × folds
  5. repetition_vote_comparison.png — repetition-level majority vote results
"""

import os
import sys
import json
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Subject lists
# ──────────────────────────────────────────────────────────────────────────────
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]

# ──────────────────────────────────────────────────────────────────────────────
# Project imports
# ──────────────────────────────────────────────────────────────────────────────
from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from processing.windowing import WindowExtractor
from processing.powerful_features import PowerfulFeatureExtractor
from utils.logging import setup_logging, seed_everything

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZES: List[int] = [100, 200, 300, 400, 500, 600, 800]
OVERLAP_RATIOS: List[float] = [0.0, 0.25, 0.5, 0.75]
MAX_GESTURES: int = 10
EXERCISES: List[str] = ["E1"]
BASE_DIR: Path = ROOT / "data"
SAMPLING_RATE: int = 2000

BASELINE_WINDOW_SIZE: int = 600
BASELINE_OVERLAP: int = 300


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities (shared with exp_45)
# ══════════════════════════════════════════════════════════════════════════════

def extract_windows_with_config(
    segments: Dict[int, List[np.ndarray]],
    window_size: int,
    overlap: int,
    logger,
    use_gpu: bool = True,
) -> Dict[int, List[np.ndarray]]:
    """Re-extract windows from raw segments with a custom window configuration."""
    cfg = ProcessingConfig(
        window_size=window_size,
        window_overlap=overlap,
        num_channels=None,
    )
    extractor = WindowExtractor(cfg, logger, use_gpu=use_gpu)
    return extractor.process_all_segments_grouped(segments)


def extract_features_from_grouped(
    grouped_windows: Dict[int, List[np.ndarray]],
    feature_extractor: PowerfulFeatureExtractor,
    gesture_order: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from grouped windows. Returns (X_feat, y)."""
    X_parts, y_parts = [], []
    for cls_idx, gid in enumerate(gesture_order):
        reps = grouped_windows.get(gid, [])
        valid_reps = [r for r in reps if len(r) > 0]
        if not valid_reps:
            continue
        windows = np.concatenate(valid_reps, axis=0)  # (N, T, C)
        feats = feature_extractor.transform(windows)   # (N, F)
        X_parts.append(feats)
        y_parts.append(np.full(len(feats), cls_idx, dtype=np.int64))

    if not X_parts:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def train_svm(X_train: np.ndarray, y_train: np.ndarray, use_pca: bool = False):
    """Train balanced SVM-RBF. Returns (clf, scaler, pca_or_None)."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = None
    if use_pca and X_scaled.shape[1] > 10:
        pca = PCA(n_components=0.95, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)

    clf = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', random_state=42,
    )
    clf.fit(X_scaled, y_train)
    return clf, scaler, pca


def evaluate_svm(clf, scaler, pca, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate SVM. Returns accuracy, balanced_accuracy, f1_macro, per-class f1."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score
    )
    X_scaled = scaler.transform(X_test)
    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    y_pred = clf.predict(X_scaled)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        "f1_per_class": f1_score(y_test, y_pred, average=None, zero_division=0).tolist(),
        "y_pred": y_pred,
        "y_true": y_test,
    }


def count_windows_per_class(y: np.ndarray, n_classes: int) -> Dict[int, int]:
    """Count windows per class index."""
    counts = {}
    for c in range(n_classes):
        counts[c] = int(np.sum(y == c))
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# Grid search (reused from exp_45)
# ══════════════════════════════════════════════════════════════════════════════

def search_optimal_window_configs(
    train_segments: Dict[int, List[np.ndarray]],
    val_segments: Dict[int, List[np.ndarray]],
    common_gestures: List[int],
    feature_extractor: PowerfulFeatureExtractor,
    window_sizes: List[int],
    overlap_ratios: List[float],
    logger,
    use_gpu: bool = True,
) -> Dict[Tuple[int, float], Dict]:
    """Grid-search over (window_size, overlap_ratio) pairs on a validation set."""
    results: Dict = {}
    n_configs = len(window_sizes) * len(overlap_ratios)
    logger.info(f"  Grid search: {n_configs} configs")

    for ws in window_sizes:
        for ov_ratio in overlap_ratios:
            overlap = int(ws * ov_ratio)
            config_key = (ws, ov_ratio)
            try:
                train_grouped = extract_windows_with_config(
                    train_segments, ws, overlap, logger, use_gpu
                )
                val_grouped = extract_windows_with_config(
                    val_segments, ws, overlap, logger, use_gpu
                )
                train_filtered = {
                    gid: train_grouped.get(gid, []) for gid in common_gestures
                }
                val_filtered = {
                    gid: val_grouped.get(gid, []) for gid in common_gestures
                }

                X_train, y_train = extract_features_from_grouped(
                    train_filtered, feature_extractor, gesture_order=common_gestures
                )
                X_val, y_val = extract_features_from_grouped(
                    val_filtered, feature_extractor, gesture_order=common_gestures
                )
                if len(X_train) < 10 or len(X_val) < 5:
                    continue

                clf, scaler, _ = train_svm(X_train, y_train, use_pca=False)
                from sklearn.metrics import f1_score, accuracy_score
                X_val_sc = scaler.transform(X_val)
                y_pred = clf.predict(X_val_sc)

                f1_macro = float(f1_score(y_val, y_pred, average='macro', zero_division=0))
                f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
                acc = float(accuracy_score(y_val, y_pred))

                results[config_key] = {
                    "f1_macro": f1_macro,
                    "f1_per_class": f1_per_class,
                    "accuracy": acc,
                    "n_train_windows": int(len(X_train)),
                    "n_val_windows": int(len(X_val)),
                }
                logger.info(
                    f"  ws={ws:4d} ov={ov_ratio:.2f} → "
                    f"F1={f1_macro:.4f}  acc={acc:.4f}  "
                    f"n_train={len(X_train):5d}  n_val={len(X_val):4d}"
                )
            except Exception as e:
                logger.warning(f"  Config ({ws}, {ov_ratio:.2f}) failed: {e}")
    return results


def find_optimal_configs_per_gesture(
    search_results: Dict[Tuple[int, float], Dict],
    common_gestures: List[int],
) -> Dict[int, Optional[Tuple[int, float]]]:
    """For each gesture (by class index), return the config maximizing per-class F1."""
    n_classes = len(common_gestures)
    optimal: Dict[int, Optional[Tuple[int, float]]] = {i: None for i in range(n_classes)}
    best_f1: Dict[int, float] = {i: -1.0 for i in range(n_classes)}

    for config_key, metrics in search_results.items():
        f1_pc = metrics.get("f1_per_class", np.array([]))
        for cls_idx in range(n_classes):
            if cls_idx < len(f1_pc) and f1_pc[cls_idx] > best_f1[cls_idx]:
                best_f1[cls_idx] = f1_pc[cls_idx]
                optimal[cls_idx] = config_key
    return optimal


# ══════════════════════════════════════════════════════════════════════════════
# Per-gesture training with personalized windows
# ══════════════════════════════════════════════════════════════════════════════

def train_personalized(
    all_train_segments: Dict[int, List[np.ndarray]],
    common_gestures: List[int],
    optimal_configs: Dict[int, Optional[Tuple[int, float]]],
    feature_extractor: PowerfulFeatureExtractor,
    logger,
    use_pca: bool = False,
    use_gpu: bool = True,
):
    """
    Train one SVM using per-gesture optimal window configs for training data.
    Returns (clf, scaler, pca_or_None).
    """
    gesture_to_cls = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    X_parts, y_parts = [], []

    for gid in common_gestures:
        cls_idx = gesture_to_cls[gid]
        config = optimal_configs.get(cls_idx)
        ws, ov_ratio = config if config is not None else (BASELINE_WINDOW_SIZE, 0.5)
        overlap = int(ws * ov_ratio)

        if gid not in all_train_segments or not all_train_segments[gid]:
            continue

        single_gest_segs = {gid: all_train_segments[gid]}
        grouped = extract_windows_with_config(single_gest_segs, ws, overlap, logger, use_gpu)

        reps = grouped.get(gid, [])
        valid_reps = [r for r in reps if len(r) > 0]
        if not valid_reps:
            continue

        windows = np.concatenate(valid_reps, axis=0)
        feats = feature_extractor.transform(windows)
        X_parts.append(feats)
        y_parts.append(np.full(len(feats), cls_idx, dtype=np.int64))

    if not X_parts:
        raise ValueError("Personalized: no training features extracted")

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    clf, scaler, pca = train_svm(X_train, y_train, use_pca=use_pca)
    return clf, scaler, pca, X_train.shape[0], count_windows_per_class(y_train, len(common_gestures))


def evaluate_personalized_test(
    test_segments: Dict[int, List[np.ndarray]],
    common_gestures: List[int],
    optimal_configs: Dict[int, Optional[Tuple[int, float]]],
    feature_extractor: PowerfulFeatureExtractor,
    clf, scaler, pca,
    logger,
    use_gpu: bool = True,
) -> Tuple[Dict, Dict[int, int]]:
    """Evaluate personalized model on per-gesture test windows. Returns (metrics, window_counts)."""
    gesture_to_cls = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    X_parts, y_parts = [], []

    for gid in common_gestures:
        cls_idx = gesture_to_cls[gid]
        config = optimal_configs.get(cls_idx)
        ws, ov_ratio = config if config is not None else (BASELINE_WINDOW_SIZE, 0.5)
        overlap = int(ws * ov_ratio)

        if gid not in test_segments or not test_segments[gid]:
            continue

        single_gest_segs = {gid: test_segments[gid]}
        grouped = extract_windows_with_config(single_gest_segs, ws, overlap, logger, use_gpu)

        reps = grouped.get(gid, [])
        valid_reps = [r for r in reps if len(r) > 0]
        if not valid_reps:
            continue

        windows = np.concatenate(valid_reps, axis=0)
        feats = feature_extractor.transform(windows)
        X_parts.append(feats)
        y_parts.append(np.full(len(feats), cls_idx, dtype=np.int64))

    if not X_parts:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "f1_macro": 0.0, "f1_per_class": []}, {}

    X_test = np.concatenate(X_parts, axis=0)
    y_test = np.concatenate(y_parts, axis=0)
    wc = count_windows_per_class(y_test, len(common_gestures))
    metrics = evaluate_svm(clf, scaler, pca, X_test, y_test)
    return metrics, wc


# ══════════════════════════════════════════════════════════════════════════════
# Repetition-level majority vote evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_repetition_vote(
    test_segments: Dict[int, List[np.ndarray]],
    common_gestures: List[int],
    feature_extractor: PowerfulFeatureExtractor,
    clf, scaler, pca,
    window_size: int,
    overlap: int,
    logger,
    use_gpu: bool = True,
) -> Dict:
    """
    Evaluate at the repetition level via majority vote.

    For each gesture's each repetition segment:
      - Extract windows with the given config
      - Classify each window
      - Majority vote → one prediction per repetition

    This normalizes out different window counts across methods.
    """
    from scipy.stats import mode as scipy_mode
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    gesture_to_cls = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    y_true_reps, y_pred_reps = [], []

    for gid in common_gestures:
        cls_idx = gesture_to_cls[gid]
        reps = test_segments.get(gid, [])

        for rep_seg in reps:
            # Extract windows from this single repetition
            single_seg = {gid: [rep_seg]}
            grouped = extract_windows_with_config(single_seg, window_size, overlap, logger, use_gpu)
            rep_windows_list = grouped.get(gid, [])
            valid = [r for r in rep_windows_list if len(r) > 0]
            if not valid:
                continue

            windows = np.concatenate(valid, axis=0)
            if len(windows) == 0:
                continue

            feats = feature_extractor.transform(windows)
            X_sc = scaler.transform(feats)
            if pca is not None:
                X_sc = pca.transform(X_sc)

            preds = clf.predict(X_sc)
            # Majority vote
            vote_result = scipy_mode(preds, keepdims=False)
            voted_class = int(vote_result.mode)

            y_true_reps.append(cls_idx)
            y_pred_reps.append(voted_class)

    if not y_true_reps:
        return {"rep_accuracy": 0.0, "rep_balanced_accuracy": 0.0, "rep_f1_macro": 0.0, "n_reps": 0}

    y_true_reps = np.array(y_true_reps)
    y_pred_reps = np.array(y_pred_reps)

    return {
        "rep_accuracy": float(accuracy_score(y_true_reps, y_pred_reps)),
        "rep_balanced_accuracy": float(balanced_accuracy_score(y_true_reps, y_pred_reps)),
        "rep_f1_macro": float(f1_score(y_true_reps, y_pred_reps, average='macro', zero_division=0)),
        "n_reps": len(y_true_reps),
    }


def evaluate_personalized_repetition_vote(
    test_segments: Dict[int, List[np.ndarray]],
    common_gestures: List[int],
    optimal_configs: Dict[int, Optional[Tuple[int, float]]],
    feature_extractor: PowerfulFeatureExtractor,
    clf, scaler, pca,
    logger,
    use_gpu: bool = True,
) -> Dict:
    """Repetition-level majority vote with per-gesture window configs."""
    from scipy.stats import mode as scipy_mode
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    gesture_to_cls = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    y_true_reps, y_pred_reps = [], []

    for gid in common_gestures:
        cls_idx = gesture_to_cls[gid]
        config = optimal_configs.get(cls_idx)
        ws, ov_ratio = config if config is not None else (BASELINE_WINDOW_SIZE, 0.5)
        overlap = int(ws * ov_ratio)

        reps = test_segments.get(gid, [])
        for rep_seg in reps:
            single_seg = {gid: [rep_seg]}
            grouped = extract_windows_with_config(single_seg, ws, overlap, logger, use_gpu)
            rep_windows_list = grouped.get(gid, [])
            valid = [r for r in rep_windows_list if len(r) > 0]
            if not valid:
                continue
            windows = np.concatenate(valid, axis=0)
            if len(windows) == 0:
                continue

            feats = feature_extractor.transform(windows)
            X_sc = scaler.transform(feats)
            if pca is not None:
                X_sc = pca.transform(X_sc)

            preds = clf.predict(X_sc)
            vote_result = scipy_mode(preds, keepdims=False)
            voted_class = int(vote_result.mode)

            y_true_reps.append(cls_idx)
            y_pred_reps.append(voted_class)

    if not y_true_reps:
        return {"rep_accuracy": 0.0, "rep_balanced_accuracy": 0.0, "rep_f1_macro": 0.0, "n_reps": 0}

    y_true_reps = np.array(y_true_reps)
    y_pred_reps = np.array(y_pred_reps)
    return {
        "rep_accuracy": float(accuracy_score(y_true_reps, y_pred_reps)),
        "rep_balanced_accuracy": float(balanced_accuracy_score(y_true_reps, y_pred_reps)),
        "rep_f1_macro": float(f1_score(y_true_reps, y_pred_reps, average='macro', zero_division=0)),
        "n_reps": len(y_true_reps),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def plot_factorial_comparison(fold_results: List[Dict], output_dir: Path) -> None:
    """Bar chart comparing all 8 models across 3 metrics (mean over folds)."""
    model_ids = ["A", "B", "C", "D", "E", "F", "G", "H"]
    model_names = [
        "Baseline\n600/300",
        "GlobalBest\ntrain only",
        "Personalized\ntrain only",
        "GlobalBest\nmatched",
        "Personalized\nmatched",
        "Baseline\n+PCA",
        "GlobalBest\ntrain+PCA",
        "Personalized\ntrain+PCA",
    ]
    metrics = ["accuracy", "balanced_accuracy", "f1_macro"]
    metric_labels = ["Accuracy", "Balanced Accuracy", "F1 Macro"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        means, stds = [], []
        for mid in model_ids:
            vals = [r[mid][metric] for r in fold_results if mid in r and r[mid] is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c',
                  '#e67e22', '#2980b9', '#27ae60']
        bars = ax.bar(range(len(model_ids)), means, yerr=stds, capsize=3,
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

        for i, (m, s) in enumerate(zip(means, stds)):
            if m > 0:
                ax.text(i, m + s + 0.01, f"{m:.3f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(len(model_ids)))
        ax.set_xticklabels(model_names, fontsize=8)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_ylim(0, 0.65)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(mlabel, fontsize=12, fontweight='bold')

    plt.suptitle(
        "Factorial Comparison: Fixed Test Set (A-C,F-H) vs Matched Test (D,E)\n"
        "(mean ± std over LOSO folds)",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_dir / "factorial_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_confound_decomposition(fold_results: List[Dict], output_dir: Path) -> None:
    """Show how much of the exp_45 gain comes from training vs test-set change."""
    # For each fold: E_acc - A_acc = total; C_acc - A_acc = train effect; E_acc - C_acc = test confound
    total_gains, train_effects, test_confounds = [], [], []
    subjects = []

    for r in fold_results:
        if "A" not in r or "C" not in r or "E" not in r:
            continue
        if r["A"] is None or r["C"] is None or r["E"] is None:
            continue
        a_acc = r["A"]["accuracy"]
        c_acc = r["C"]["accuracy"]
        e_acc = r["E"]["accuracy"]
        total_gains.append(e_acc - a_acc)
        train_effects.append(c_acc - a_acc)
        test_confounds.append(e_acc - c_acc)
        subjects.append(r.get("test_subject", "?"))

    if not total_gains:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-fold stacked bar
    ax = axes[0]
    x = np.arange(len(subjects))
    ax.bar(x, train_effects, label='Train effect (C - A)', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x, test_confounds, bottom=train_effects, label='Test confound (E - C)',
           color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.7)
    for i, tg in enumerate(total_gains):
        ax.text(i, tg + 0.005, f"{tg:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=10)
    ax.set_ylabel("Accuracy Δ", fontsize=11)
    ax.set_title("Decomposition per Fold", fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Mean pie chart
    ax = axes[1]
    mean_train = np.mean(train_effects)
    mean_test = np.mean(test_confounds)
    if mean_train + mean_test > 0:
        sizes = [max(mean_train, 0), max(mean_test, 0)]
        labels_pie = [
            f'Training effect\n{mean_train:.4f}',
            f'Test-set confound\n{mean_test:.4f}',
        ]
        colors_pie = ['#2ecc71', '#e74c3c']
        ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax.set_title(
            f"Mean gain decomposition\n"
            f"Total: {np.mean(total_gains):.4f}",
            fontsize=12,
        )
    else:
        ax.text(0.5, 0.5, 'No positive gain to decompose',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Mean gain decomposition", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "confound_decomposition.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_window_counts(fold_results: List[Dict], output_dir: Path) -> None:
    """Per-class window count comparison for selected models."""
    models_to_show = ["A", "B", "C", "E"]
    model_labels = {
        "A": "Baseline 600/300",
        "B": "GlobalBest train",
        "C": "Personalized train",
        "E": "Personalized matched",
    }

    # Aggregate across folds
    all_counts: Dict[str, Dict[int, List[int]]] = {m: defaultdict(list) for m in models_to_show}
    for r in fold_results:
        for mid in models_to_show:
            wc = r.get(f"{mid}_test_window_counts", {})
            for cls_str, cnt in wc.items():
                all_counts[mid][int(cls_str)].append(cnt)

    if not any(all_counts[m] for m in models_to_show):
        return

    n_classes = max(
        max(all_counts[m].keys()) + 1
        for m in models_to_show if all_counts[m]
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_classes)
    width = 0.2
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#1abc9c']

    for i, (mid, color) in enumerate(zip(models_to_show, colors)):
        means = [np.mean(all_counts[mid].get(c, [0])) for c in range(n_classes)]
        ax.bar(x + i * width, means, width, label=model_labels[mid],
               color=color, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"G{c}" for c in range(n_classes)], fontsize=10)
    ax.set_ylabel("Test windows (mean over folds)", fontsize=11)
    ax.set_title("Test Window Count per Class by Method\n(shows class balance differences)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "window_count_per_class.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_heatmap(fold_results: List[Dict], output_dir: Path) -> None:
    """Heatmap: models × folds for accuracy."""
    model_ids = ["A", "B", "C", "D", "E", "F", "G", "H"]
    model_names = [
        "A: Baseline",
        "B: GlobalBest train",
        "C: Pers. train",
        "D: GlobalBest matched",
        "E: Pers. matched",
        "F: Baseline+PCA",
        "G: GlobalBest+PCA",
        "H: Pers. train+PCA",
    ]
    subjects = [r.get("test_subject", f"fold_{i}") for i, r in enumerate(fold_results)]
    matrix = np.full((len(model_ids), len(fold_results)), np.nan)

    for j, r in enumerate(fold_results):
        for i, mid in enumerate(model_ids):
            if mid in r and r[mid] is not None:
                matrix[i, j] = r[mid]["accuracy"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.15, vmax=0.55)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, fontsize=10)
    ax.set_yticks(range(len(model_ids)))
    ax.set_yticklabels(model_names, fontsize=9)

    for i in range(len(model_ids)):
        for j in range(len(subjects)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if matrix[i, j] < 0.25 else 'black')

    plt.colorbar(im, ax=ax, label='Accuracy')
    ax.set_title("Accuracy: Models × LOSO Folds", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "per_fold_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_repetition_vote(fold_results: List[Dict], output_dir: Path) -> None:
    """Bar chart of repetition-level majority vote accuracy."""
    model_ids = ["A", "C", "E"]
    model_names = [
        "Baseline\n(rep vote)",
        "Pers. train\nfixed test\n(rep vote)",
        "Pers. matched\n(rep vote)",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    means, stds = [], []
    for mid in model_ids:
        vals = [
            r.get(f"{mid}_rep_vote", {}).get("rep_accuracy", 0)
            for r in fold_results
            if r.get(f"{mid}_rep_vote") is not None
        ]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)

    colors = ['#e74c3c', '#2ecc71', '#1abc9c']
    bars = ax.bar(range(len(model_ids)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    for i, (m, s) in enumerate(zip(means, stds)):
        if m > 0:
            ax.text(i, m + s + 0.01, f"{m:.3f}", ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(model_ids)))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Repetition-level Accuracy", fontsize=11)
    ax.set_title("Majority Vote per Repetition\n(same repetitions across all methods)", fontsize=12)
    ax.set_ylim(0, 0.65)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "repetition_vote_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Single LOSO fold
# ══════════════════════════════════════════════════════════════════════════════

def run_loso_fold(
    base_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    output_dir: Path,
    logger,
    val_ratio: float = 0.20,
    max_gestures: int = MAX_GESTURES,
    use_gpu: bool = True,
    seed: int = 42,
) -> Dict:
    """Run a single LOSO fold with all 8 factorial models."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD: test={test_subject}, train={train_subjects}")
    logger.info(f"{'='*60}")

    # ── Load segments ──────────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=BASELINE_WINDOW_SIZE,
        window_overlap=BASELINE_OVERLAP,
        num_channels=8,
    )
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=use_gpu,
        use_improved_processing=True,
    )

    all_subject_ids = train_subjects + [test_subject]
    subjects_segments: Dict[str, Dict[int, List[np.ndarray]]] = {}

    for sid in all_subject_ids:
        logger.info(f"  Loading {sid} ...")
        _emg, segments, _gw = multi_loader.load_subject_multiple_exercises(
            base_dir=base_dir,
            subject_id=sid,
            exercises=exercises,
            include_rest=False,
        )
        subjects_segments[sid] = segments
        logger.info(f"  {sid}: {len(segments)} gestures loaded")

    # ── Common gestures ────────────────────────────────────────────────────
    gesture_sets = [set(subjects_segments[sid].keys()) for sid in all_subject_ids]
    common_set = set.intersection(*gesture_sets) if gesture_sets else set()
    non_rest = sorted(g for g in common_set if g != 0)
    common_gestures = non_rest[-max_gestures:]
    n_classes = len(common_gestures)
    logger.info(f"Common gestures ({n_classes}): {common_gestures}")

    # Filter to common gestures
    for sid in all_subject_ids:
        subjects_segments[sid] = {
            gid: subjects_segments[sid][gid]
            for gid in common_gestures
            if gid in subjects_segments[sid]
        }

    # ── Train/val split (for grid search) ──────────────────────────────────
    train_segments: Dict[int, List[np.ndarray]] = defaultdict(list)
    val_segments: Dict[int, List[np.ndarray]] = defaultdict(list)
    rng = np.random.RandomState(seed)

    for sid in train_subjects:
        for gid, reps in subjects_segments.get(sid, {}).items():
            n_reps = len(reps)
            if n_reps == 0:
                continue
            idx = rng.permutation(n_reps)
            n_val = max(1, int(n_reps * val_ratio))
            for i in idx[:n_val]:
                val_segments[gid].append(reps[i])
            for i in idx[n_val:]:
                train_segments[gid].append(reps[i])

    # All train reps (no val split) for final models
    all_train_segments: Dict[int, List[np.ndarray]] = defaultdict(list)
    for sid in train_subjects:
        for gid, reps in subjects_segments.get(sid, {}).items():
            all_train_segments[gid].extend(reps)

    test_segments = subjects_segments[test_subject]

    # ── Feature extractor ──────────────────────────────────────────────────
    feature_extractor = PowerfulFeatureExtractor(sampling_rate=SAMPLING_RATE)

    # ── Grid search ────────────────────────────────────────────────────────
    logger.info("Grid search for optimal window configs ...")
    search_results = search_optimal_window_configs(
        train_segments=dict(train_segments),
        val_segments=dict(val_segments),
        common_gestures=common_gestures,
        feature_extractor=feature_extractor,
        window_sizes=WINDOW_SIZES,
        overlap_ratios=OVERLAP_RATIOS,
        logger=logger,
        use_gpu=use_gpu,
    )

    optimal_per_gesture = find_optimal_configs_per_gesture(search_results, common_gestures)
    if search_results:
        global_best_key = max(search_results, key=lambda k: search_results[k]["f1_macro"])
        global_best_ws, global_best_ov = global_best_key
    else:
        global_best_ws, global_best_ov = BASELINE_WINDOW_SIZE, 0.5

    gl_overlap = int(global_best_ws * global_best_ov)
    logger.info(f"Global best: ws={global_best_ws}, ov={global_best_ov:.2f}")
    for cls_idx, cfg in optimal_per_gesture.items():
        gid = sorted(common_gestures)[cls_idx]
        logger.info(f"  G{gid} (cls {cls_idx}): {cfg}")

    # ══════════════════════════════════════════════════════════════════════
    # Extract FIXED test windows (baseline 600/300)
    # ══════════════════════════════════════════════════════════════════════
    fixed_test_grouped = extract_windows_with_config(
        test_segments, BASELINE_WINDOW_SIZE, BASELINE_OVERLAP, logger, use_gpu
    )
    fixed_test_filtered = {gid: fixed_test_grouped.get(gid, []) for gid in common_gestures}
    X_fixed_test, y_fixed_test = extract_features_from_grouped(
        fixed_test_filtered, feature_extractor, gesture_order=common_gestures
    )
    fixed_test_wc = count_windows_per_class(y_fixed_test, n_classes)
    logger.info(f"Fixed test set: {len(X_fixed_test)} windows, per-class: {fixed_test_wc}")

    # ══════════════════════════════════════════════════════════════════════
    # Extract GLOBAL-BEST test windows (800/0.75)
    # ══════════════════════════════════════════════════════════════════════
    gl_test_grouped = extract_windows_with_config(
        test_segments, global_best_ws, gl_overlap, logger, use_gpu
    )
    gl_test_filtered = {gid: gl_test_grouped.get(gid, []) for gid in common_gestures}
    X_gl_test, y_gl_test = extract_features_from_grouped(
        gl_test_filtered, feature_extractor, gesture_order=common_gestures
    )
    gl_test_wc = count_windows_per_class(y_gl_test, n_classes)

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN models
    # ══════════════════════════════════════════════════════════════════════
    results = {"test_subject": test_subject}

    # --- A: Baseline (train=600/300, test=600/300, no PCA) ---
    logger.info("Training A: Baseline ...")
    bl_train_grouped = extract_windows_with_config(
        dict(all_train_segments), BASELINE_WINDOW_SIZE, BASELINE_OVERLAP, logger, use_gpu
    )
    bl_train_filtered = {gid: bl_train_grouped.get(gid, []) for gid in common_gestures}
    X_bl_tr, y_bl_tr = extract_features_from_grouped(
        bl_train_filtered, feature_extractor, gesture_order=common_gestures
    )
    bl_tr_wc = count_windows_per_class(y_bl_tr, n_classes)
    clf_a, scaler_a, pca_a = train_svm(X_bl_tr, y_bl_tr, use_pca=False)
    results["A"] = evaluate_svm(clf_a, scaler_a, pca_a, X_fixed_test, y_fixed_test)
    results["A_train_window_counts"] = bl_tr_wc
    results["A_test_window_counts"] = fixed_test_wc
    logger.info(f"  A: acc={results['A']['accuracy']:.4f}  bal_acc={results['A']['balanced_accuracy']:.4f}  f1={results['A']['f1_macro']:.4f}")

    # --- B: Global-Best train / fixed test (no PCA) ---
    logger.info(f"Training B: GlobalBest train (ws={global_best_ws}, ov={global_best_ov:.2f}) ...")
    gl_train_grouped = extract_windows_with_config(
        dict(all_train_segments), global_best_ws, gl_overlap, logger, use_gpu
    )
    gl_train_filtered = {gid: gl_train_grouped.get(gid, []) for gid in common_gestures}
    X_gl_tr, y_gl_tr = extract_features_from_grouped(
        gl_train_filtered, feature_extractor, gesture_order=common_gestures
    )
    gl_tr_wc = count_windows_per_class(y_gl_tr, n_classes)
    clf_b, scaler_b, pca_b = train_svm(X_gl_tr, y_gl_tr, use_pca=False)
    results["B"] = evaluate_svm(clf_b, scaler_b, pca_b, X_fixed_test, y_fixed_test)
    results["B_train_window_counts"] = gl_tr_wc
    results["B_test_window_counts"] = fixed_test_wc
    logger.info(f"  B: acc={results['B']['accuracy']:.4f}  bal_acc={results['B']['balanced_accuracy']:.4f}  f1={results['B']['f1_macro']:.4f}")

    # --- C: Personalized train / fixed test (no PCA) ---
    logger.info("Training C: Personalized train ...")
    clf_c, scaler_c, pca_c, n_pers_tr, pers_tr_wc = train_personalized(
        dict(all_train_segments), common_gestures, optimal_per_gesture,
        feature_extractor, logger, use_pca=False, use_gpu=use_gpu,
    )
    results["C"] = evaluate_svm(clf_c, scaler_c, pca_c, X_fixed_test, y_fixed_test)
    results["C_train_window_counts"] = pers_tr_wc
    results["C_test_window_counts"] = fixed_test_wc
    logger.info(f"  C: acc={results['C']['accuracy']:.4f}  bal_acc={results['C']['balanced_accuracy']:.4f}  f1={results['C']['f1_macro']:.4f}")

    # --- D: Global-Best matched (train=800/0.75, test=800/0.75, no PCA) ---
    logger.info("Training D: GlobalBest matched ...")
    clf_d, scaler_d, pca_d = train_svm(X_gl_tr, y_gl_tr, use_pca=False)
    results["D"] = evaluate_svm(clf_d, scaler_d, pca_d, X_gl_test, y_gl_test)
    results["D_train_window_counts"] = gl_tr_wc
    results["D_test_window_counts"] = gl_test_wc
    logger.info(f"  D: acc={results['D']['accuracy']:.4f}  bal_acc={results['D']['balanced_accuracy']:.4f}  f1={results['D']['f1_macro']:.4f}")

    # --- E: Personalized matched (train=per-gesture, test=per-gesture, no PCA) ---
    logger.info("Training E: Personalized matched (replicates exp_45) ...")
    # Reuse the personalized-trained model (clf_c uses the same training)
    # But we need to re-train because clf_c was trained without PCA
    # and evaluate on per-gesture test windows
    clf_e, scaler_e, pca_e, _, pers_tr_wc_e = train_personalized(
        dict(all_train_segments), common_gestures, optimal_per_gesture,
        feature_extractor, logger, use_pca=False, use_gpu=use_gpu,
    )
    results["E"], pers_test_wc = evaluate_personalized_test(
        test_segments, common_gestures, optimal_per_gesture,
        feature_extractor, clf_e, scaler_e, pca_e, logger, use_gpu,
    )
    results["E_train_window_counts"] = pers_tr_wc_e
    results["E_test_window_counts"] = pers_test_wc
    logger.info(f"  E: acc={results['E']['accuracy']:.4f}  bal_acc={results['E']['balanced_accuracy']:.4f}  f1={results['E']['f1_macro']:.4f}")

    # --- F: Baseline + PCA (train=600/300, test=600/300) ---
    logger.info("Training F: Baseline + PCA ...")
    clf_f, scaler_f, pca_f = train_svm(X_bl_tr, y_bl_tr, use_pca=True)
    results["F"] = evaluate_svm(clf_f, scaler_f, pca_f, X_fixed_test, y_fixed_test)
    results["F_train_window_counts"] = bl_tr_wc
    results["F_test_window_counts"] = fixed_test_wc
    logger.info(f"  F: acc={results['F']['accuracy']:.4f}  bal_acc={results['F']['balanced_accuracy']:.4f}  f1={results['F']['f1_macro']:.4f}")

    # --- G: GlobalBest train + PCA (train=800/0.75, test=600/300) ---
    logger.info("Training G: GlobalBest train + PCA ...")
    clf_g, scaler_g, pca_g = train_svm(X_gl_tr, y_gl_tr, use_pca=True)
    results["G"] = evaluate_svm(clf_g, scaler_g, pca_g, X_fixed_test, y_fixed_test)
    results["G_train_window_counts"] = gl_tr_wc
    results["G_test_window_counts"] = fixed_test_wc
    logger.info(f"  G: acc={results['G']['accuracy']:.4f}  bal_acc={results['G']['balanced_accuracy']:.4f}  f1={results['G']['f1_macro']:.4f}")

    # --- H: Personalized train + PCA (per-gesture train, test=600/300) ---
    logger.info("Training H: Personalized train + PCA ...")
    clf_h, scaler_h, pca_h, _, pers_tr_wc_h = train_personalized(
        dict(all_train_segments), common_gestures, optimal_per_gesture,
        feature_extractor, logger, use_pca=True, use_gpu=use_gpu,
    )
    results["H"] = evaluate_svm(clf_h, scaler_h, pca_h, X_fixed_test, y_fixed_test)
    results["H_train_window_counts"] = pers_tr_wc_h
    results["H_test_window_counts"] = fixed_test_wc
    logger.info(f"  H: acc={results['H']['accuracy']:.4f}  bal_acc={results['H']['balanced_accuracy']:.4f}  f1={results['H']['f1_macro']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Repetition-level majority vote (for A, C, E)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("Computing repetition-level majority vote ...")

    # A: baseline config for both train and test votes
    results["A_rep_vote"] = evaluate_repetition_vote(
        test_segments, common_gestures, feature_extractor,
        clf_a, scaler_a, pca_a,
        BASELINE_WINDOW_SIZE, BASELINE_OVERLAP, logger, use_gpu,
    )
    logger.info(f"  A rep_vote: acc={results['A_rep_vote']['rep_accuracy']:.4f}")

    # C: personalized training, but baseline windows for test votes
    results["C_rep_vote"] = evaluate_repetition_vote(
        test_segments, common_gestures, feature_extractor,
        clf_c, scaler_c, pca_c,
        BASELINE_WINDOW_SIZE, BASELINE_OVERLAP, logger, use_gpu,
    )
    logger.info(f"  C rep_vote: acc={results['C_rep_vote']['rep_accuracy']:.4f}")

    # E: personalized training + personalized test votes
    results["E_rep_vote"] = evaluate_personalized_repetition_vote(
        test_segments, common_gestures, optimal_per_gesture,
        feature_extractor, clf_e, scaler_e, pca_e, logger, use_gpu,
    )
    logger.info(f"  E rep_vote: acc={results['E_rep_vote']['rep_accuracy']:.4f}")

    # ── Store metadata ─────────────────────────────────────────────────────
    results["global_best_config"] = {"ws": global_best_ws, "ov": global_best_ov}
    results["optimal_per_gesture"] = {
        str(k): list(v) if v is not None else None
        for k, v in optimal_per_gesture.items()
    }
    results["common_gestures"] = common_gestures
    results["n_classes"] = n_classes

    # Save fold results
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean = {}
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray):
                    clean[kk] = vv.tolist()
                elif isinstance(vv, (np.integer, np.floating)):
                    clean[kk] = float(vv)
                else:
                    clean[kk] = vv
            serializable[k] = clean
        elif isinstance(v, (list, tuple)):
            serializable[k] = v
        else:
            serializable[k] = v

    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)

    # Cleanup
    import gc
    del clf_a, clf_b, clf_c, clf_d, clf_e, clf_f, clf_g, clf_h
    del scaler_a, scaler_b, scaler_c, scaler_d, scaler_e, scaler_f, scaler_g, scaler_h
    gc.collect()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _parser = argparse.ArgumentParser(
        description="Exp 45b: Verification of Personalized Window Segmentation"
    )
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _parser.add_argument("--val-ratio", type=float, default=0.20)
    _parser.add_argument("--seed", type=int, default=42)
    _parser.add_argument("--no-gpu", action="store_true")
    _parser.add_argument("--max-gestures", type=int, default=MAX_GESTURES)
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = _FULL_SUBJECTS
    else:
        ALL_SUBJECTS = _CI_SUBJECTS

    use_gpu = not _args.no_gpu
    seed = _args.seed

    EXP_NAME = "exp_45b_verification_personalized_window"
    OUTPUT_DIR = ROOT / "experiments_output" / EXP_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 70)
    logger.info("Experiment 45b: Verification of Personalized Window Segmentation")
    logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    logger.info(f"Baseline: ws={BASELINE_WINDOW_SIZE}, overlap={BASELINE_OVERLAP}")
    logger.info(f"Grid: {len(WINDOW_SIZES)} sizes × {len(OVERLAP_RATIOS)} overlaps")
    logger.info("=" * 70)

    seed_everything(seed, verbose=True)

    # ── LOSO loop ──────────────────────────────────────────────────────────
    all_fold_results: List[Dict] = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"fold_{test_subject}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        fold_logger = setup_logging(fold_output_dir)

        try:
            fold_data = run_loso_fold(
                base_dir=BASE_DIR,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                output_dir=fold_output_dir,
                logger=fold_logger,
                val_ratio=_args.val_ratio,
                max_gestures=_args.max_gestures,
                use_gpu=use_gpu,
                seed=seed,
            )
            all_fold_results.append(fold_data)

            # Log comparison
            a = fold_data["A"]["accuracy"]
            c = fold_data["C"]["accuracy"]
            e = fold_data["E"]["accuracy"]
            logger.info(
                f"[FOLD {test_subject}]  "
                f"A(baseline)={a:.4f}  "
                f"C(pers_train)={c:.4f} (Δ={c-a:+.4f})  "
                f"E(pers_matched)={e:.4f} (Δ={e-a:+.4f})  "
                f"confound={e-c:+.4f}"
            )

        except Exception as e:
            logger.error(f"Fold {test_subject} FAILED: {e}")
            traceback.print_exc()
            all_fold_results.append({"test_subject": test_subject, "error": str(e)})

    # ── Aggregate summary ──────────────────────────────────────────────────
    valid_folds = [r for r in all_fold_results if "error" not in r]
    model_ids = ["A", "B", "C", "D", "E", "F", "G", "H"]

    summary: Dict = {
        "experiment": EXP_NAME,
        "subjects": ALL_SUBJECTS,
        "n_folds": len(ALL_SUBJECTS),
        "n_valid_folds": len(valid_folds),
        "baseline_config": {"ws": BASELINE_WINDOW_SIZE, "overlap": BASELINE_OVERLAP},
        "models": {},
    }

    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<30s} {'Accuracy':>10s} {'Bal.Acc':>10s} {'F1 Macro':>10s}")
    logger.info("-" * 62)

    for mid in model_ids:
        accs = [r[mid]["accuracy"] for r in valid_folds if mid in r and r[mid] is not None]
        baccs = [r[mid]["balanced_accuracy"] for r in valid_folds if mid in r and r[mid] is not None]
        f1s = [r[mid]["f1_macro"] for r in valid_folds if mid in r and r[mid] is not None]

        model_summary = {
            "mean_accuracy": float(np.mean(accs)) if accs else None,
            "std_accuracy": float(np.std(accs)) if accs else None,
            "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else None,
            "std_balanced_accuracy": float(np.std(baccs)) if baccs else None,
            "mean_f1_macro": float(np.mean(f1s)) if f1s else None,
            "std_f1_macro": float(np.std(f1s)) if f1s else None,
        }
        summary["models"][mid] = model_summary

        if model_summary["mean_accuracy"] is not None:
            logger.info(
                f"  {mid:<28s} "
                f"{model_summary['mean_accuracy']:.4f}±{model_summary['std_accuracy']:.4f}  "
                f"{model_summary['mean_balanced_accuracy']:.4f}±{model_summary['std_balanced_accuracy']:.4f}  "
                f"{model_summary['mean_f1_macro']:.4f}±{model_summary['std_f1_macro']:.4f}"
            )

    # Key comparisons
    if summary["models"].get("A", {}).get("mean_accuracy") is not None:
        a_acc = summary["models"]["A"]["mean_accuracy"]
        logger.info("\n--- Key Comparisons (mean Δ accuracy) ---")
        for mid, label in [
            ("B", "B-A: GlobalBest train effect"),
            ("C", "C-A: Personalized train effect"),
            ("E", "E-A: Total (exp_45 replication)"),
            ("F", "F-A: PCA effect alone"),
            ("H", "H-A: Personalized + PCA"),
        ]:
            m_acc = summary["models"].get(mid, {}).get("mean_accuracy")
            if m_acc is not None:
                logger.info(f"  {label}: {m_acc - a_acc:+.4f}")

        c_acc = summary["models"].get("C", {}).get("mean_accuracy")
        e_acc = summary["models"].get("E", {}).get("mean_accuracy")
        if c_acc is not None and e_acc is not None:
            logger.info(f"  E-C: Test-set confound:     {e_acc - c_acc:+.4f}")

    # Repetition-vote summary
    logger.info("\n--- Repetition-level Majority Vote ---")
    for mid in ["A", "C", "E"]:
        vals = [
            r.get(f"{mid}_rep_vote", {}).get("rep_accuracy", None)
            for r in valid_folds
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            logger.info(f"  {mid} rep_vote: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save summary
    summary["fold_results_brief"] = [
        {
            "test_subject": r.get("test_subject"),
            **{
                mid: {
                    "accuracy": r[mid]["accuracy"],
                    "balanced_accuracy": r[mid]["balanced_accuracy"],
                    "f1_macro": r[mid]["f1_macro"],
                }
                for mid in model_ids if mid in r and r[mid] is not None
            },
        }
        for r in valid_folds
    ]

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # ── Visualizations ─────────────────────────────────────────────────────
    logger.info("\nGenerating visualizations ...")
    try:
        plot_factorial_comparison(valid_folds, OUTPUT_DIR)
        logger.info("  factorial_comparison.png")
    except Exception as e:
        logger.warning(f"  factorial_comparison failed: {e}")

    try:
        plot_confound_decomposition(valid_folds, OUTPUT_DIR)
        logger.info("  confound_decomposition.png")
    except Exception as e:
        logger.warning(f"  confound_decomposition failed: {e}")

    try:
        plot_window_counts(valid_folds, OUTPUT_DIR)
        logger.info("  window_count_per_class.png")
    except Exception as e:
        logger.warning(f"  window_count_per_class failed: {e}")

    try:
        plot_per_fold_heatmap(valid_folds, OUTPUT_DIR)
        logger.info("  per_fold_heatmap.png")
    except Exception as e:
        logger.warning(f"  per_fold_heatmap failed: {e}")

    try:
        plot_repetition_vote(valid_folds, OUTPUT_DIR)
        logger.info("  repetition_vote_comparison.png")
    except Exception as e:
        logger.warning(f"  repetition_vote_comparison failed: {e}")

    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
