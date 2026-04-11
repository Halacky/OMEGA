#!/usr/bin/env python3
"""
Experiment 45: Personalized Window Segmentation (LOSO)
=======================================================

Hypothesis:
    Different gestures have different optimal window sizes/strides.
    A fixed 600-sample window may be too large for short gestures, causing
    information dilution or loss of discriminative temporal patterns.
    Per-gesture window optimisation found on a validation set can improve
    cross-subject classification accuracy.

Method:
    1. Load raw EMG *segments* (before windowing) for all LOSO subjects.
    2. Grid-search over window_size × overlap_ratio for ALL gestures together
       on a held-out validation set (20 % of train repetitions).
    3. For each gesture independently, select the window config that maximises
       per-class F1 on the validation set → "per-gesture optimal" configs.
    4. Train three SVM (RBF, PowerfulFeatures) models per LOSO fold:
         a. Baseline  — fixed 600 samples / 300 overlap (project default)
         b. GlobalBest — single best config from the search (same for all gestures)
         c. Personalized — per-gesture optimal config
    5. Evaluate all three models on the held-out test subject and aggregate
       metrics over LOSO folds.

Visualizations (9 plots):
    1. segment_length_distribution.png      — box plots of raw segment lengths
                                              per gesture (shows WHY short windows matter)
    2. window_config_search_heatmap.png     — F1 per gesture × window size heat-map
                                              (averaged over overlap ratios & LOSO folds)
    3. window_size_sensitivity_curve.png    — global F1 vs window size curve
    4. overlap_sensitivity.png              — F1 vs overlap ratio per window size
    5. window_yield_vs_accuracy.png         — scatter: #val-windows vs F1
                                              (shows yield / accuracy trade-off)
    6. optimal_window_per_gesture.png       — optimal window size per gesture
                                              (bar chart + distribution)
    7. config_diversity_heatmap.png         — gesture × window size count-matrix
                                              (stability across LOSO folds)
    8. accuracy_comparison_per_fold.png /   — grouped bars per fold + mean comparison
       mean_accuracy_comparison.png
    9. per_gesture_f1_comparison.png        — per-gesture F1: baseline vs personalized
                                              + delta bar chart
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
import seaborn as sns

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
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from processing.windowing import WindowExtractor
from processing.powerful_features import PowerfulFeatureExtractor
from utils.logging import setup_logging, seed_everything

# ──────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ──────────────────────────────────────────────────────────────────────────────
# Window search grid
WINDOW_SIZES: List[int] = [100, 200, 300, 400, 500, 600, 800]   # samples @ 2000 Hz
OVERLAP_RATIOS: List[float] = [0.0, 0.25, 0.5, 0.75]            # fraction of window_size

MAX_GESTURES: int = 10
EXERCISES: List[str] = ["E1"]
BASE_DIR: Path = ROOT / "data"

# Baseline config (project default for this dataset)
BASELINE_WINDOW_SIZE: int = 600
BASELINE_OVERLAP: int = 300   # 50 % overlap

SAMPLING_RATE: int = 2000


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grouped_windows {gid: [arrays (N,T,C)]} → (X, y) flat arrays."""
    X_parts, y_parts = [], []
    for cls_idx, gid in enumerate(sorted(grouped_windows.keys())):
        reps = grouped_windows[gid]
        valid_reps = [r for r in reps if len(r) > 0]
        if not valid_reps:
            continue
        X_g = np.concatenate(valid_reps, axis=0)
        y_g = np.full(len(X_g), cls_idx, dtype=np.int64)
        X_parts.append(X_g)
        y_parts.append(y_g)
    if not X_parts:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


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
        num_channels=None,   # channels already selected before segmentation
    )
    extractor = WindowExtractor(cfg, logger, use_gpu=use_gpu)
    return extractor.process_all_segments_grouped(segments)


def extract_features_from_grouped(
    grouped_windows: Dict[int, List[np.ndarray]],
    feature_extractor: PowerfulFeatureExtractor,
    gesture_order: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extract features from a grouped_windows dict.

    Args:
        grouped_windows: {gesture_id: [array(N,T,C), ...]}
        feature_extractor: fitted-or-fit-free PowerfulFeatureExtractor
        gesture_order: ordered gesture IDs (defines class index mapping);
                       if None, sorted(grouped_windows.keys()) is used.

    Returns:
        X_feat  — shape (N_total, F)
        y       — shape (N_total,) with class indices
        gesture_ids_sorted — list of gesture IDs in class-index order
    """
    if gesture_order is None:
        gesture_order = sorted(grouped_windows.keys())

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
        return np.empty((0,)), np.empty((0,), dtype=np.int64), gesture_order

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        gesture_order,
    )


def train_svm(X_train: np.ndarray, y_train: np.ndarray):
    """Train a balanced SVM-RBF. Returns (clf, scaler)."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', random_state=42,
    )
    clf.fit(X_scaled, y_train)
    return clf, scaler


def evaluate_svm(clf, scaler, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate SVM; returns accuracy, f1_macro and per-class f1."""
    from sklearn.metrics import accuracy_score, f1_score

    X_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": f1_per_class,
        "y_pred": y_pred,
        "y_true": y_test,
    }


def get_segment_lengths(
    segments: Dict[int, List[np.ndarray]],
) -> Dict[int, List[int]]:
    """Return T (number of samples) of each repetition segment per gesture."""
    return {gid: [r.shape[0] for r in reps] for gid, reps in segments.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Core search / training functions
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
    """
    Grid-search over (window_size, overlap_ratio) pairs.

    For each config:
      * Extract windows from train_segments and val_segments
      * Train SVM (PowerfulFeatures) on train windows
      * Evaluate on val windows

    Returns:
        {(window_size, overlap_ratio): {
            "f1_macro": float,
            "f1_per_class": np.ndarray,     # length == len(common_gestures)
            "accuracy": float,
            "n_train_windows": int,
            "n_val_windows": int,
        }}
    """
    results: Dict = {}
    n_configs = len(window_sizes) * len(overlap_ratios)
    logger.info(f"  Grid search: {n_configs} configs "
                f"({len(window_sizes)} sizes × {len(overlap_ratios)} overlaps)")

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

                # Filter to common gestures only
                train_filtered = {
                    gid: train_grouped.get(gid, []) for gid in common_gestures
                }
                val_filtered = {
                    gid: val_grouped.get(gid, []) for gid in common_gestures
                }

                X_train, y_train, _ = extract_features_from_grouped(
                    train_filtered, feature_extractor, gesture_order=common_gestures
                )
                X_val, y_val, _ = extract_features_from_grouped(
                    val_filtered, feature_extractor, gesture_order=common_gestures
                )

                if len(X_train) < 10 or len(X_val) < 5:
                    logger.debug(
                        f"  Config ws={ws} ov={ov_ratio:.2f}: "
                        f"too few windows (train={len(X_train)}, val={len(X_val)}), skip"
                    )
                    continue

                clf, scaler = train_svm(X_train, y_train)
                metrics = evaluate_svm(clf, scaler, X_val, y_val)

                results[config_key] = {
                    "f1_macro": metrics["f1_macro"],
                    "f1_per_class": metrics["f1_per_class"],
                    "accuracy": metrics["accuracy"],
                    "n_train_windows": int(len(X_train)),
                    "n_val_windows": int(len(X_val)),
                }

                logger.info(
                    f"  ws={ws:4d} ov={ov_ratio:.2f} → "
                    f"F1={metrics['f1_macro']:.4f}  acc={metrics['accuracy']:.4f}  "
                    f"n_train={len(X_train):5d}  n_val={len(X_val):4d}"
                )

            except Exception as e:
                logger.warning(f"  Config ({ws}, {ov_ratio:.2f}) failed: {e}")

    return results


def find_optimal_configs_per_gesture(
    search_results: Dict[Tuple[int, float], Dict],
    common_gestures: List[int],
) -> Dict[int, Optional[Tuple[int, float]]]:
    """
    For each gesture (by class index == position in sorted(common_gestures)),
    return the window config that maximises per-class F1 on the validation set.

    Returns:
        {cls_idx: (window_size, overlap_ratio) or None}
    """
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


def train_personalized_model(
    all_train_segments: Dict[int, List[np.ndarray]],
    common_gestures: List[int],
    optimal_configs: Dict[int, Optional[Tuple[int, float]]],
    feature_extractor: PowerfulFeatureExtractor,
    logger,
    use_gpu: bool = True,
):
    """
    Train SVM using per-gesture optimal window configs.

    For each gesture, windows are extracted with the gesture-specific optimal
    (window_size, overlap_ratio).  All features are then concatenated into one
    training set that is passed to a single SVM.

    Returns (clf, scaler, gesture_to_cls)
    """
    gesture_to_cls = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    X_parts, y_parts = [], []

    for gid in common_gestures:
        cls_idx = gesture_to_cls[gid]
        config = optimal_configs.get(cls_idx)

        if config is None:
            ws, ov_ratio = BASELINE_WINDOW_SIZE, 0.5
            logger.warning(
                f"  No optimal config for gesture {gid} (cls {cls_idx}), "
                f"falling back to baseline ({ws}, ov={ov_ratio})"
            )
        else:
            ws, ov_ratio = config

        overlap = int(ws * ov_ratio)

        if gid not in all_train_segments or not all_train_segments[gid]:
            logger.warning(f"  No training segments for gesture {gid}, skipping")
            continue

        # Extract windows for this gesture only
        single_gest_segs = {gid: all_train_segments[gid]}
        grouped = extract_windows_with_config(single_gest_segs, ws, overlap, logger, use_gpu)

        reps = grouped.get(gid, [])
        valid_reps = [r for r in reps if len(r) > 0]
        if not valid_reps:
            logger.warning(f"  Gesture {gid}: 0 windows with ws={ws}, skipping")
            continue

        windows = np.concatenate(valid_reps, axis=0)   # (N, T, C)
        feats = feature_extractor.transform(windows)   # (N, F)
        X_parts.append(feats)
        y_parts.append(np.full(len(feats), cls_idx, dtype=np.int64))

    if not X_parts:
        raise ValueError("Personalized model: no training features extracted")

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    clf, scaler = train_svm(X_train, y_train)
    return clf, scaler, gesture_to_cls


def evaluate_personalized(
    test_segments: Dict[int, List[np.ndarray]],
    common_gestures: List[int],
    optimal_configs: Dict[int, Optional[Tuple[int, float]]],
    feature_extractor: PowerfulFeatureExtractor,
    clf,
    scaler,
    gesture_to_cls: Dict[int, int],
    logger,
    use_gpu: bool = True,
) -> Dict:
    """Evaluate the personalized model on a test subject."""
    X_parts, y_parts = [], []

    for gid in common_gestures:
        cls_idx = gesture_to_cls.get(gid)
        if cls_idx is None:
            continue

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
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_per_class": np.array([])}

    X_test = np.concatenate(X_parts, axis=0)
    y_test = np.concatenate(y_parts, axis=0)
    return evaluate_svm(clf, scaler, X_test, y_test)


# ══════════════════════════════════════════════════════════════════════════════
# Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def _gest_labels(common_gestures: List[int]) -> List[str]:
    return [f"G{gid}" for gid in sorted(common_gestures)]


def plot_segment_length_distribution(
    all_segment_lengths: Dict[str, Dict[int, List[int]]],
    common_gestures: List[int],
    output_dir: Path,
) -> None:
    """
    Box plot of raw segment lengths per gesture, aggregated over all subjects.
    Orange dashed lines mark the window sizes in the search grid.
    Helps to understand which gestures might suffer from a 600-sample window.
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    lengths_by_gesture: Dict[int, List[int]] = defaultdict(list)
    for subj_lengths in all_segment_lengths.values():
        for gid in common_gestures:
            lengths_by_gesture[gid].extend(subj_lengths.get(gid, []))

    data_to_plot, labels = [], []
    for gid in sorted(common_gestures):
        vals = lengths_by_gesture[gid]
        if vals:
            data_to_plot.append(vals)
            med_ms = np.median(vals) / SAMPLING_RATE * 1000
            labels.append(f"G{gid}\n({med_ms:.0f} ms)")

    bp = ax.boxplot(
        data_to_plot, labels=labels, patch_artist=True,
        medianprops=dict(color='crimson', linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(data_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Mark window sizes in search grid
    for ws_idx, ws in enumerate(WINDOW_SIZES):
        ax.axhline(
            ws, color='darkorange', alpha=0.5,
            linestyle='--', linewidth=1.2,
            label=f"{ws}s ({ws/SAMPLING_RATE*1000:.0f}ms)" if ws_idx == 0 else "_nolegend_",
        )
        ax.text(
            len(data_to_plot) + 0.1, ws, f" {ws}",
            va='center', fontsize=7.5, color='darkorange',
        )

    ax.axhline(
        BASELINE_WINDOW_SIZE, color='red', linewidth=2.5, linestyle='-',
        label=f"Baseline ({BASELINE_WINDOW_SIZE}s = {BASELINE_WINDOW_SIZE/SAMPLING_RATE*1000:.0f}ms)",
    )

    ax.set_xlabel("Gesture (median duration)", fontsize=12)
    ax.set_ylabel("Segment length (samples)", fontsize=12)
    ax.set_title(
        "Raw Segment Length Distribution per Gesture\n"
        "(orange dashed = searched window sizes; red solid = baseline)",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "segment_length_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_window_config_search_heatmap(
    search_results_per_fold: List[Dict],
    common_gestures: List[int],
    output_dir: Path,
) -> None:
    """
    Heat-map: F1 per gesture (rows) × window_size (columns).
    Values averaged over overlap_ratios and LOSO folds.
    Shows which gestures prefer which window sizes.
    """
    n_classes = len(common_gestures)
    n_ws = len(WINDOW_SIZES)
    f1_matrix = np.full((n_classes, n_ws), np.nan)
    count_matrix = np.zeros((n_classes, n_ws))

    for fold_results in search_results_per_fold:
        for ws_idx, ws in enumerate(WINDOW_SIZES):
            f1_accum = np.zeros(n_classes)
            cnt = 0
            for ov_ratio in OVERLAP_RATIOS:
                key = (ws, ov_ratio)
                if key in fold_results:
                    f1_pc = fold_results[key].get("f1_per_class", np.array([]))
                    if len(f1_pc) >= n_classes:
                        f1_accum += f1_pc[:n_classes]
                        cnt += 1
            if cnt > 0:
                avg = f1_accum / cnt
                for cls_idx in range(n_classes):
                    if np.isnan(f1_matrix[cls_idx, ws_idx]):
                        f1_matrix[cls_idx, ws_idx] = avg[cls_idx]
                        count_matrix[cls_idx, ws_idx] = 1
                    else:
                        n = count_matrix[cls_idx, ws_idx]
                        f1_matrix[cls_idx, ws_idx] = (
                            f1_matrix[cls_idx, ws_idx] * n + avg[cls_idx]
                        ) / (n + 1)
                        count_matrix[cls_idx, ws_idx] = n + 1

    fig, ax = plt.subplots(figsize=(13, max(5, n_classes * 0.65 + 2)))
    gesture_labels = _gest_labels(common_gestures)
    ws_labels = [f"{ws}\n({ws/SAMPLING_RATE*1000:.0f}ms)" for ws in WINDOW_SIZES]

    im = ax.imshow(
        np.nan_to_num(f1_matrix, nan=0),
        aspect='auto', cmap='RdYlGn', vmin=0.0, vmax=1.0,
        interpolation='nearest',
    )

    ax.set_xticks(range(n_ws))
    ax.set_xticklabels(ws_labels, fontsize=10)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(gesture_labels, fontsize=10)

    for i in range(n_classes):
        for j in range(n_ws):
            if not np.isnan(f1_matrix[i, j]):
                val = f1_matrix[i, j]
                txt_color = 'black' if 0.25 < val < 0.75 else 'white'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=8, color=txt_color)

    # Mark baseline window size column
    bl_idx = WINDOW_SIZES.index(BASELINE_WINDOW_SIZE) if BASELINE_WINDOW_SIZE in WINDOW_SIZES else None
    if bl_idx is not None:
        ax.add_patch(plt.Rectangle(
            (bl_idx - 0.5, -0.5), 1, n_classes,
            linewidth=2.5, edgecolor='red', facecolor='none', clip_on=False,
        ))
        ax.text(bl_idx, n_classes - 0.3, "baseline", ha='center',
                fontsize=9, color='red', style='italic')

    plt.colorbar(im, ax=ax, label='F1 Score', shrink=0.8)
    ax.set_xlabel("Window Size (samples / duration)", fontsize=12)
    ax.set_ylabel("Gesture Class", fontsize=12)
    ax.set_title(
        "F1 Score per Gesture × Window Size\n"
        "(averaged over overlap ratios and LOSO folds; red border = baseline)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "window_config_search_heatmap.png", dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_window_size_sensitivity_curve(
    search_results_per_fold: List[Dict],
    output_dir: Path,
) -> None:
    """
    Global F1 macro vs window size, averaged over overlap ratios and LOSO folds.
    Reveals the project-wide "sweet spot" window duration.
    """
    f1_by_ws: Dict[int, List[float]] = {ws: [] for ws in WINDOW_SIZES}

    for fold_results in search_results_per_fold:
        for ws in WINDOW_SIZES:
            for ov_ratio in OVERLAP_RATIOS:
                key = (ws, ov_ratio)
                if key in fold_results:
                    f1_by_ws[ws].append(fold_results[key]["f1_macro"])

    means = [np.mean(f1_by_ws[ws]) if f1_by_ws[ws] else np.nan for ws in WINDOW_SIZES]
    stds = [np.std(f1_by_ws[ws]) if len(f1_by_ws[ws]) > 1 else 0.0 for ws in WINDOW_SIZES]
    ws_ms = [ws / SAMPLING_RATE * 1000 for ws in WINDOW_SIZES]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ws_ms, means, 'b-o', linewidth=2.5, markersize=9, label='Mean F1')
    ax.fill_between(
        ws_ms,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.18, color='blue', label='±1 std',
    )

    baseline_ms = BASELINE_WINDOW_SIZE / SAMPLING_RATE * 1000
    ax.axvline(
        baseline_ms, color='red', linestyle='--', linewidth=2,
        label=f'Baseline ({BASELINE_WINDOW_SIZE}s = {baseline_ms:.0f}ms)',
    )

    valid_means = [(ms, m) for ms, m in zip(ws_ms, means) if not np.isnan(m)]
    if valid_means:
        best_ms, best_mean = max(valid_means, key=lambda x: x[1])
        ax.scatter(
            [best_ms], [best_mean], s=220, color='gold',
            zorder=5, marker='*',
            label=f'Best: {best_ms:.0f}ms (F1={best_mean:.4f})',
        )

    ax.set_xlabel("Window Size (ms)", fontsize=12)
    ax.set_ylabel("Mean F1 Macro (Validation)", fontsize=12)
    ax.set_title(
        "Window Size Sensitivity Curve\n"
        "(mean ± std over overlap ratios and LOSO folds)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(ws_ms)
    ax.set_xticklabels([f"{ms:.0f}" for ms in ws_ms])
    plt.tight_layout()
    plt.savefig(
        output_dir / "window_size_sensitivity_curve.png", dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_overlap_sensitivity(
    search_results_per_fold: List[Dict],
    output_dir: Path,
) -> None:
    """
    F1 vs overlap ratio for each window size — shows whether more overlap
    consistently helps or whether there is a diminishing-returns effect.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(WINDOW_SIZES)))

    for ws_idx, ws in enumerate(WINDOW_SIZES):
        f1_per_overlap = []
        for ov_ratio in OVERLAP_RATIOS:
            f1s = []
            for fold_results in search_results_per_fold:
                key = (ws, ov_ratio)
                if key in fold_results:
                    f1s.append(fold_results[key]["f1_macro"])
            f1_per_overlap.append(np.mean(f1s) if f1s else np.nan)

        dur_ms = ws / SAMPLING_RATE * 1000
        ax.plot(
            OVERLAP_RATIOS, f1_per_overlap,
            marker='o', color=colors[ws_idx], linewidth=2, markersize=8,
            label=f'{ws}s ({dur_ms:.0f}ms)',
        )

    ax.set_xlabel("Overlap Ratio", fontsize=12)
    ax.set_ylabel("Mean F1 Macro (Validation)", fontsize=12)
    ax.set_title(
        "F1 vs Overlap Ratio for Each Window Size\n"
        "(mean over LOSO folds)",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xticks(OVERLAP_RATIOS)
    ax.set_xticklabels([f"{o:.0%}" for o in OVERLAP_RATIOS])
    plt.tight_layout()
    plt.savefig(output_dir / "overlap_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_window_yield_vs_accuracy(
    search_results_per_fold: List[Dict],
    output_dir: Path,
) -> None:
    """
    Scatter plot: number of validation windows vs F1 macro for each config & fold.
    Color = window size.  Shows whether more windows always improve accuracy.
    """
    all_n, all_f1, all_ws = [], [], []
    for fold_results in search_results_per_fold:
        for (ws, _ov), metrics in fold_results.items():
            all_n.append(metrics.get("n_val_windows", 0))
            all_f1.append(metrics.get("f1_macro", 0))
            all_ws.append(ws)

    if not all_n:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        all_n, all_f1,
        c=all_ws, cmap='viridis',
        alpha=0.7, s=80, edgecolors='black', linewidth=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Window Size (samples)', fontsize=11)

    # Add trend line
    if len(all_n) > 3:
        z = np.polyfit(all_n, all_f1, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(all_n), max(all_n), 100)
        ax.plot(x_range, p(x_range), 'r--', linewidth=1.5, alpha=0.6, label='Trend')
        ax.legend(fontsize=10)

    ax.set_xlabel("Number of Validation Windows", fontsize=12)
    ax.set_ylabel("F1 Macro (Validation)", fontsize=12)
    ax.set_title(
        "Window Yield vs F1 Performance\n"
        "(each point = one config × one LOSO fold; colour = window size)",
        fontsize=13,
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "window_yield_vs_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_optimal_window_per_gesture(
    optimal_per_gesture_all_folds: List[Dict[int, Optional[Tuple[int, float]]]],
    common_gestures: List[int],
    output_dir: Path,
) -> None:
    """
    Two-panel figure:
      Left  — bar chart: mean ± std optimal window size per gesture class.
      Right — histogram: how many gesture classes prefer each window size (mode).
    """
    n_classes = len(common_gestures)
    gesture_labels = _gest_labels(common_gestures)

    opt_ws_per_cls: Dict[int, List[int]] = defaultdict(list)
    for fold_opt in optimal_per_gesture_all_folds:
        for cls_idx, config in fold_opt.items():
            if config is not None and cls_idx < n_classes:
                opt_ws_per_cls[cls_idx].append(config[0])

    mean_ws = [
        np.mean(opt_ws_per_cls[i]) if opt_ws_per_cls[i] else BASELINE_WINDOW_SIZE
        for i in range(n_classes)
    ]
    std_ws = [
        np.std(opt_ws_per_cls[i]) if len(opt_ws_per_cls[i]) > 1 else 0
        for i in range(n_classes)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # ── Left: bar chart ──────────────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(n_classes)
    norm_vals = np.array(mean_ws)
    colors = plt.cm.coolwarm(norm_vals / max(WINDOW_SIZES) if max(WINDOW_SIZES) > 0 else norm_vals)
    bars = ax.bar(x, mean_ws, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.errorbar(x, mean_ws, yerr=std_ws, fmt='none', color='black', capsize=5, linewidth=1.8)

    ax.axhline(
        BASELINE_WINDOW_SIZE, color='red', linestyle='--', linewidth=2.2,
        label=f'Baseline ({BASELINE_WINDOW_SIZE}s / {BASELINE_WINDOW_SIZE/SAMPLING_RATE*1000:.0f}ms)',
    )

    for bar, ws in zip(bars, mean_ws):
        dur_ms = ws / SAMPLING_RATE * 1000
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(std_ws) * 0.1 + 15,
            f"{dur_ms:.0f}ms",
            ha='center', va='bottom', fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(gesture_labels, fontsize=10)
    ax.set_ylabel("Window Size (samples)", fontsize=12)
    ax.set_title(
        "Optimal Window Size per Gesture\n(mean ± std over LOSO folds)", fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(WINDOW_SIZES) + 180)

    # ── Right: distribution ───────────────────────────────────────────────────
    ax = axes[1]
    mode_counts: Dict[int, int] = defaultdict(int)
    for cls_idx in range(n_classes):
        vals = opt_ws_per_cls[cls_idx]
        if vals:
            mode_ws = max(set(vals), key=vals.count)
            mode_counts[mode_ws] += 1

    if mode_counts:
        wss = sorted(mode_counts.keys())
        cnts = [mode_counts[ws] for ws in wss]
        xl = [f"{ws}\n({ws/SAMPLING_RATE*1000:.0f}ms)" for ws in wss]
        ax.bar(xl, cnts, color='steelblue', alpha=0.85, edgecolor='black')
        for i, (cnt, ws) in enumerate(zip(cnts, wss)):
            pct = cnt / n_classes * 100
            ax.text(i, cnt + 0.05, f"{pct:.0f}%", ha='center', va='bottom', fontsize=10)
    ax.set_xlabel("Window Size", fontsize=12)
    ax.set_ylabel("# gesture classes preferring this size", fontsize=12)
    ax.set_title("Distribution of Mode Optimal Window Sizes\nAcross Gesture Classes", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "optimal_window_per_gesture.png", dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_config_diversity_heatmap(
    optimal_per_gesture_all_folds: List[Dict[int, Optional[Tuple[int, float]]]],
    common_gestures: List[int],
    output_dir: Path,
) -> None:
    """
    Heat-map: gesture (rows) × window_size (columns) → count of LOSO folds
    that chose this window size as optimal for this gesture.
    Shows stability and diversity of the optimal configurations.
    """
    n_classes = len(common_gestures)
    n_ws = len(WINDOW_SIZES)
    count_matrix = np.zeros((n_classes, n_ws))

    for fold_opt in optimal_per_gesture_all_folds:
        for cls_idx, config in fold_opt.items():
            if config is not None and cls_idx < n_classes:
                ws, _ = config
                if ws in WINDOW_SIZES:
                    count_matrix[cls_idx, WINDOW_SIZES.index(ws)] += 1

    n_folds = len(optimal_per_gesture_all_folds)
    fig, ax = plt.subplots(figsize=(13, max(5, n_classes * 0.65 + 2)))
    gesture_labels = _gest_labels(common_gestures)
    ws_labels = [f"{ws}\n({ws/SAMPLING_RATE*1000:.0f}ms)" for ws in WINDOW_SIZES]

    im = ax.imshow(
        count_matrix, aspect='auto', cmap='Blues',
        vmin=0, vmax=n_folds if n_folds > 0 else 1,
        interpolation='nearest',
    )

    ax.set_xticks(range(n_ws))
    ax.set_xticklabels(ws_labels, fontsize=10)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(gesture_labels, fontsize=10)

    for i in range(n_classes):
        for j in range(n_ws):
            cnt = int(count_matrix[i, j])
            if cnt > 0:
                txt_color = 'white' if cnt > n_folds * 0.5 else 'black'
                ax.text(j, i, str(cnt), ha='center', va='center',
                        fontsize=11, fontweight='bold', color=txt_color)

    # Mark baseline window size
    if BASELINE_WINDOW_SIZE in WINDOW_SIZES:
        bl_j = WINDOW_SIZES.index(BASELINE_WINDOW_SIZE)
        ax.add_patch(plt.Rectangle(
            (bl_j - 0.5, -0.5), 1, n_classes,
            linewidth=2.5, edgecolor='red', facecolor='none', clip_on=False,
        ))

    plt.colorbar(im, ax=ax, label=f'# LOSO folds preferring this config (max={n_folds})')
    ax.set_xlabel("Window Size (samples / duration)", fontsize=12)
    ax.set_ylabel("Gesture Class", fontsize=12)
    ax.set_title(
        "Configuration Diversity Heatmap\n"
        "(number of LOSO folds that selected each window size per gesture; "
        "red border = baseline)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "config_diversity_heatmap.png", dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_accuracy_comparison(
    fold_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Two-panel grouped bar charts:
      Top    — accuracy per LOSO fold (baseline / global-best / personalized)
      Bottom — mean ± std across all folds (both accuracy and F1 macro)
    """
    subjects = [r["test_subject"] for r in fold_results]
    baseline_accs = [r.get("baseline_accuracy", 0) or 0 for r in fold_results]
    global_accs = [r.get("global_best_accuracy", 0) or 0 for r in fold_results]
    pers_accs = [r.get("personalized_accuracy", 0) or 0 for r in fold_results]

    baseline_f1s = [r.get("baseline_f1_macro", 0) or 0 for r in fold_results]
    global_f1s = [r.get("global_best_f1_macro", 0) or 0 for r in fold_results]
    pers_f1s = [r.get("personalized_f1_macro", 0) or 0 for r in fold_results]

    x = np.arange(len(subjects))
    width = 0.24
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    labels = [
        f'Baseline ({BASELINE_WINDOW_SIZE}s)',
        'Global-Best Window',
        'Per-Gesture Personalized',
    ]

    # ── Figure 1: per-fold ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (vals_b, vals_g, vals_p, metric_name) in enumerate([
        (baseline_accs, global_accs, pers_accs, "Accuracy"),
        (baseline_f1s, global_f1s, pers_f1s, "F1 Macro"),
    ]):
        ax = axes[ax_idx]
        b1 = ax.bar(x - width, vals_b, width, label=labels[0], color=colors[0], alpha=0.85)
        b2 = ax.bar(x,         vals_g, width, label=labels[1], color=colors[1], alpha=0.85)
        b3 = ax.bar(x + width, vals_p, width, label=labels[2], color=colors[2], alpha=0.85)
        for bars in (b1, b2, b3):
            ax.bar_label(bars, fmt='{:.2f}', fontsize=7, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("DB2_", "") for s in subjects], fontsize=10)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name}: Baseline vs Global-Best vs Personalized", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Per-Fold Performance Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(
        output_dir / "accuracy_comparison_per_fold.png", dpi=150, bbox_inches='tight'
    )
    plt.close()

    # ── Figure 2: mean ± std ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    metrics_names = ['Accuracy', 'F1 Macro']
    means_b = [np.mean(baseline_accs), np.mean(baseline_f1s)]
    means_g = [np.mean(global_accs),   np.mean(global_f1s)]
    means_p = [np.mean(pers_accs),     np.mean(pers_f1s)]
    stds_b = [np.std(baseline_accs), np.std(baseline_f1s)]
    stds_g = [np.std(global_accs),   np.std(global_f1s)]
    stds_p = [np.std(pers_accs),     np.std(pers_f1s)]

    x2 = np.arange(2)
    b1 = ax.bar(x2 - width, means_b, width, yerr=stds_b, capsize=6,
                label=labels[0], color=colors[0], alpha=0.85)
    b2 = ax.bar(x2,          means_g, width, yerr=stds_g, capsize=6,
                label=labels[1], color=colors[1], alpha=0.85)
    b3 = ax.bar(x2 + width,  means_p, width, yerr=stds_p, capsize=6,
                label=labels[2], color=colors[2], alpha=0.85)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt='{:.4f}', fontsize=9, padding=3)

    ax.set_xticks(x2)
    ax.set_xticklabels(metrics_names, fontsize=13)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Mean Performance: Baseline vs Global-Best vs Personalized\n"
        "(mean ± std over LOSO folds)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "mean_accuracy_comparison.png", dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_per_gesture_f1_comparison(
    fold_results: List[Dict],
    common_gestures: List[int],
    output_dir: Path,
) -> None:
    """
    Two-panel figure:
      Left  — side-by-side F1 per gesture: baseline vs personalized
      Right — ΔF1 per gesture (improvement or degradation)
    """
    n_classes = len(common_gestures)
    gesture_labels = _gest_labels(common_gestures)

    baseline_f1_pg = np.zeros(n_classes)
    pers_f1_pg = np.zeros(n_classes)
    count = 0

    for r in fold_results:
        bfpg = r.get("baseline_f1_per_class", [])
        pfpg = r.get("personalized_f1_per_class", [])
        if len(bfpg) >= n_classes and len(pfpg) >= n_classes:
            baseline_f1_pg += np.array(bfpg[:n_classes])
            pers_f1_pg += np.array(pfpg[:n_classes])
            count += 1

    if count == 0:
        return

    baseline_f1_pg /= count
    pers_f1_pg /= count
    delta = pers_f1_pg - baseline_f1_pg

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    x = np.arange(n_classes)
    width = 0.38

    # ── Left: side-by-side ───────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x - width / 2, baseline_f1_pg, width,
           label='Baseline', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, pers_f1_pg, width,
           label='Personalized', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(gesture_labels, fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Gesture F1: Baseline vs Personalized\n(mean over LOSO folds)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # ── Right: delta F1 ───────────────────────────────────────────────────────
    ax = axes[1]
    bar_colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in delta]
    ax.bar(x, delta, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1.5)
    mean_delta = np.mean(delta)
    ax.axhline(
        mean_delta, color='navy', linestyle='--', linewidth=1.8,
        label=f'Mean Δ = {mean_delta:+.4f}',
    )

    for i, d in enumerate(delta):
        offset = max(abs(delta)) * 0.04
        va = 'bottom' if d >= 0 else 'top'
        ax.text(
            i, d + offset * (1 if d >= 0 else -1),
            f"{d:+.3f}", ha='center', va=va, fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(gesture_labels, fontsize=10)
    ax.set_ylabel("ΔF1  (Personalized − Baseline)", fontsize=12)
    ax.set_title(
        "Per-Gesture F1 Improvement from Personalization\n"
        "(green = improved, red = degraded)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Per-Gesture Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(
        output_dir / "per_gesture_f1_comparison.png", dpi=150, bbox_inches='tight'
    )
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# LOSO fold
# ══════════════════════════════════════════════════════════════════════════════

def run_loso_fold(
    base_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    output_dir: Path,
    logger,
    val_ratio: float = 0.2,
    max_gestures: int = MAX_GESTURES,
    use_gpu: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Single LOSO fold for the personalized window segmentation experiment.

    Steps:
      1. Load EMG *segments* (before windowing) for all subjects via
         MultiSubjectLoader (improved processing applied).
      2. Find common gestures.
      3. Split train subjects' repetitions into train_segments / val_segments.
      4. Run grid search over (window_size, overlap_ratio) on val set.
      5. Derive per-gesture optimal configs.
      6. Train three SVM models: baseline / global-best / personalized.
      7. Evaluate on test subject.
    """
    seed_everything(seed, verbose=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 70}")
    logger.info(f"LOSO FOLD  test_subject={test_subject}")
    logger.info(f"Train subjects: {train_subjects}")

    # Minimal ProcessingConfig — we will redo windowing inside this experiment.
    # The window_size / overlap here only affects the default grouped_windows
    # returned by the loader, which we discard.
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

    # ── 1. Load segments (discard pre-computed grouped_windows) ───────────────
    all_subject_ids = train_subjects + [test_subject]
    subjects_segments: Dict[str, Dict[int, List[np.ndarray]]] = {}

    for sid in all_subject_ids:
        logger.info(f"  Loading {sid} ...")
        # load_subject_multiple_exercises returns (emg, segments, grouped_windows)
        # segments[gid] = List[np.ndarray(T, C)] — raw gesture repetition segments
        _emg, segments, _gw = multi_loader.load_subject_multiple_exercises(
            base_dir=base_dir,
            subject_id=sid,
            exercises=exercises,
            include_rest=False,   # exclude rest class for gesture classification
        )
        subjects_segments[sid] = segments
        logger.info(f"  {sid}: {len(segments)} gestures loaded")

    # ── 2. Common gestures ────────────────────────────────────────────────────
    gesture_sets = [set(subjects_segments[sid].keys()) for sid in all_subject_ids]
    common_set = set.intersection(*gesture_sets) if gesture_sets else set()
    non_rest = sorted(g for g in common_set if g != 0)
    common_gestures = non_rest[-max_gestures:]   # take last N (highest IDs)
    n_classes = len(common_gestures)

    logger.info(f"Common gestures ({n_classes}): {common_gestures}")
    if n_classes == 0:
        raise ValueError("No common gestures found across subjects")

    # Filter to common gestures only
    for sid in all_subject_ids:
        subjects_segments[sid] = {
            gid: subjects_segments[sid][gid]
            for gid in common_gestures
            if gid in subjects_segments[sid]
        }

    # ── 3. Collect segment length statistics ──────────────────────────────────
    all_segment_lengths: Dict[str, Dict[int, List[int]]] = {
        sid: get_segment_lengths(subjects_segments[sid])
        for sid in all_subject_ids
    }

    # ── 4. Build train / val split by repetitions ─────────────────────────────
    train_segments: Dict[int, List[np.ndarray]] = defaultdict(list)
    val_segments: Dict[int, List[np.ndarray]] = defaultdict(list)
    rng = np.random.RandomState(seed)

    for sid in train_subjects:
        if sid not in subjects_segments:
            continue
        for gid, reps in subjects_segments[sid].items():
            n_reps = len(reps)
            if n_reps == 0:
                continue
            idx = rng.permutation(n_reps)
            n_val = max(1, int(n_reps * val_ratio))
            for i in idx[:n_val]:
                val_segments[gid].append(reps[i])
            for i in idx[n_val:]:
                train_segments[gid].append(reps[i])

    # All train repetitions (no val split) — used to train the final models
    all_train_segments: Dict[int, List[np.ndarray]] = defaultdict(list)
    for sid in train_subjects:
        for gid, reps in subjects_segments.get(sid, {}).items():
            all_train_segments[gid].extend(reps)

    test_segments = subjects_segments[test_subject]

    logger.info(
        f"  Split → train: {dict((k, len(v)) for k, v in train_segments.items())}"
    )
    logger.info(
        f"  Split → val:   {dict((k, len(v)) for k, v in val_segments.items())}"
    )
    logger.info(
        f"  Test segments: {dict((k, len(v)) for k, v in test_segments.items())}"
    )

    # ── 5. Feature extractor ──────────────────────────────────────────────────
    feature_extractor = PowerfulFeatureExtractor(sampling_rate=SAMPLING_RATE)

    # ── 6. Grid search ────────────────────────────────────────────────────────
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
    logger.info(f"Search: {len(search_results)} valid configs evaluated")

    # ── 7. Optimal configs ────────────────────────────────────────────────────
    optimal_per_gesture = find_optimal_configs_per_gesture(search_results, common_gestures)

    if search_results:
        global_best_key = max(search_results, key=lambda k: search_results[k]["f1_macro"])
        global_best_ws, global_best_ov = global_best_key
    else:
        global_best_ws, global_best_ov = BASELINE_WINDOW_SIZE, 0.5
        global_best_key = (global_best_ws, global_best_ov)

    logger.info(f"Global best config: ws={global_best_ws}, ov={global_best_ov:.2f}  "
                f"(val F1={search_results.get(global_best_key, {}).get('f1_macro', 0):.4f})")
    for cls_idx, cfg in optimal_per_gesture.items():
        gid = sorted(common_gestures)[cls_idx]
        logger.info(f"  Gesture G{gid} (cls {cls_idx}): optimal cfg = {cfg}")

    # ── 8a. Baseline model ─────────────────────────────────────────────────────
    logger.info("Training baseline model ...")
    bl_train_grouped = extract_windows_with_config(
        dict(all_train_segments), BASELINE_WINDOW_SIZE, BASELINE_OVERLAP, logger, use_gpu
    )
    bl_test_grouped = extract_windows_with_config(
        test_segments, BASELINE_WINDOW_SIZE, BASELINE_OVERLAP, logger, use_gpu
    )
    bl_train_filtered = {gid: bl_train_grouped.get(gid, []) for gid in common_gestures}
    bl_test_filtered = {gid: bl_test_grouped.get(gid, []) for gid in common_gestures}

    X_bl_tr, y_bl_tr, _ = extract_features_from_grouped(
        bl_train_filtered, feature_extractor, gesture_order=common_gestures
    )
    X_bl_te, y_bl_te, _ = extract_features_from_grouped(
        bl_test_filtered, feature_extractor, gesture_order=common_gestures
    )
    bl_clf, bl_scaler = train_svm(X_bl_tr, y_bl_tr)
    baseline_metrics = evaluate_svm(bl_clf, bl_scaler, X_bl_te, y_bl_te)
    logger.info(
        f"Baseline → acc={baseline_metrics['accuracy']:.4f}  "
        f"f1={baseline_metrics['f1_macro']:.4f}"
    )

    # ── 8b. Global-best model ─────────────────────────────────────────────────
    logger.info(
        f"Training global-best model (ws={global_best_ws}, ov={global_best_ov:.2f}) ..."
    )
    gl_overlap = int(global_best_ws * global_best_ov)
    gl_train_grouped = extract_windows_with_config(
        dict(all_train_segments), global_best_ws, gl_overlap, logger, use_gpu
    )
    gl_test_grouped = extract_windows_with_config(
        test_segments, global_best_ws, gl_overlap, logger, use_gpu
    )
    gl_train_filtered = {gid: gl_train_grouped.get(gid, []) for gid in common_gestures}
    gl_test_filtered = {gid: gl_test_grouped.get(gid, []) for gid in common_gestures}

    X_gl_tr, y_gl_tr, _ = extract_features_from_grouped(
        gl_train_filtered, feature_extractor, gesture_order=common_gestures
    )
    X_gl_te, y_gl_te, _ = extract_features_from_grouped(
        gl_test_filtered, feature_extractor, gesture_order=common_gestures
    )
    gl_clf, gl_scaler = train_svm(X_gl_tr, y_gl_tr)
    global_metrics = evaluate_svm(gl_clf, gl_scaler, X_gl_te, y_gl_te)
    logger.info(
        f"Global-best → acc={global_metrics['accuracy']:.4f}  "
        f"f1={global_metrics['f1_macro']:.4f}"
    )

    # ── 8c. Per-gesture personalized model ────────────────────────────────────
    logger.info("Training per-gesture personalized model ...")
    pers_clf, pers_scaler, gesture_to_cls = train_personalized_model(
        all_train_segments=dict(all_train_segments),
        common_gestures=common_gestures,
        optimal_configs=optimal_per_gesture,
        feature_extractor=feature_extractor,
        logger=logger,
        use_gpu=use_gpu,
    )
    pers_metrics = evaluate_personalized(
        test_segments=test_segments,
        common_gestures=common_gestures,
        optimal_configs=optimal_per_gesture,
        feature_extractor=feature_extractor,
        clf=pers_clf,
        scaler=pers_scaler,
        gesture_to_cls=gesture_to_cls,
        logger=logger,
        use_gpu=use_gpu,
    )
    logger.info(
        f"Personalized → acc={pers_metrics['accuracy']:.4f}  "
        f"f1={pers_metrics['f1_macro']:.4f}"
    )

    # ── 9. Package results ────────────────────────────────────────────────────
    def _f1pc(m: Dict) -> List[float]:
        return m["f1_per_class"].tolist() if len(m.get("f1_per_class", [])) > 0 else []

    fold_result = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "common_gestures": common_gestures,
        "n_classes": n_classes,
        # ── Baseline ──
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_f1_macro": baseline_metrics["f1_macro"],
        "baseline_f1_per_class": _f1pc(baseline_metrics),
        # ── Global-best ──
        "global_best_accuracy": global_metrics["accuracy"],
        "global_best_f1_macro": global_metrics["f1_macro"],
        "global_best_f1_per_class": _f1pc(global_metrics),
        "global_best_window_size": int(global_best_ws),
        "global_best_overlap_ratio": float(global_best_ov),
        # ── Personalized ──
        "personalized_accuracy": pers_metrics["accuracy"],
        "personalized_f1_macro": pers_metrics["f1_macro"],
        "personalized_f1_per_class": _f1pc(pers_metrics),
        "optimal_per_gesture": {
            str(cls_idx): list(cfg) if cfg is not None else None
            for cls_idx, cfg in optimal_per_gesture.items()
        },
        # ── Segment length stats ──
        "segment_lengths": {
            sid: {str(gid): lens for gid, lens in gid_lens.items()}
            for sid, gid_lens in all_segment_lengths.items()
        },
    }

    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(fold_result, f, indent=4, ensure_ascii=False)

    # Per-fold visualizations
    _plot_fold_search_summary(search_results, common_gestures, output_dir)

    import gc
    del bl_clf, bl_scaler, gl_clf, gl_scaler, pers_clf, pers_scaler
    gc.collect()

    return {
        "fold_result": fold_result,
        "search_results": search_results,
        "optimal_per_gesture": optimal_per_gesture,
        "all_segment_lengths": all_segment_lengths,
        "common_gestures": common_gestures,
    }


def _plot_fold_search_summary(
    search_results: Dict,
    common_gestures: List[int],
    output_dir: Path,
) -> None:
    """
    Quick per-fold plot: F1 macro of every (ws, ov) config evaluated in this fold.
    Saved as fold_window_search_summary.png inside the fold output directory.
    """
    if not search_results:
        return

    keys = sorted(search_results.keys())
    labels = [f"ws={ws}\nov={ov:.2f}" for ws, ov in keys]
    f1s = [search_results[k]["f1_macro"] for k in keys]

    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.5), 5))
    bar_colors = plt.cm.RdYlGn(np.array(f1s))
    ax.bar(range(len(keys)), f1s, color=bar_colors, edgecolor='black', linewidth=0.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel("F1 Macro (Validation)", fontsize=11)
    ax.set_title("Window Config Search Results (this fold)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Mark baseline result
    bl_key = (BASELINE_WINDOW_SIZE, 0.5)
    if bl_key in search_results:
        bl_idx = keys.index(bl_key)
        ax.axvline(bl_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Baseline position (F1={f1s[bl_idx]:.4f})')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(
        output_dir / "fold_window_search_summary.png", dpi=130, bbox_inches='tight'
    )
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _parser = argparse.ArgumentParser(
        description="Exp 45: Personalized Window Segmentation (LOSO)"
    )
    _parser.add_argument(
        "--subjects", type=str, default=None,
        help="Comma-separated subject IDs e.g. DB2_s1,DB2_s12",
    )
    _parser.add_argument("--ci", action="store_true", help="Use CI test subset (5 subjects)")
    _parser.add_argument("--full", action="store_true", help="Use full 20-subject set")
    _parser.add_argument(
        "--val-ratio", type=float, default=0.20,
        help="Fraction of train repetitions held out for window-config search (default 0.20)",
    )
    _parser.add_argument("--seed", type=int, default=42)
    _parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    _parser.add_argument(
        "--max-gestures", type=int, default=MAX_GESTURES,
        help="Maximum number of gestures to classify",
    )
    _args, _ = _parser.parse_known_args()

    # Subject list — default to CI subjects (safe for server)
    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = _FULL_SUBJECTS
    else:
        ALL_SUBJECTS = _CI_SUBJECTS   # default = CI (server-safe)

    use_gpu = not _args.no_gpu
    seed = _args.seed

    EXP_NAME = "exp_45_personalized_window_segmentation"
    OUTPUT_DIR = ROOT / "experiments_output" / EXP_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 70)
    logger.info("Experiment 45: Personalized Window Segmentation (LOSO)")
    logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    logger.info(f"Window sizes searched: {WINDOW_SIZES} samples")
    logger.info(f"Overlap ratios searched: {OVERLAP_RATIOS}")
    logger.info(
        f"Grid total: {len(WINDOW_SIZES) * len(OVERLAP_RATIOS)} configs per fold"
    )
    logger.info(f"Baseline: ws={BASELINE_WINDOW_SIZE}, overlap={BASELINE_OVERLAP}")
    logger.info(f"Exercises: {EXERCISES}")
    logger.info(f"Val ratio for search: {_args.val_ratio}")
    logger.info(f"Max gestures: {_args.max_gestures}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info("=" * 70)

    seed_everything(seed, verbose=True)

    # ── LOSO loop ─────────────────────────────────────────────────────────────
    all_fold_results: List[Dict] = []
    all_search_results_per_fold: List[Dict] = []
    all_optimal_per_gesture_folds: List[Dict] = []
    all_segment_lengths_global: Dict[str, Dict[int, List[int]]] = {}
    first_common_gestures: Optional[List[int]] = None

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

            all_fold_results.append(fold_data["fold_result"])
            all_search_results_per_fold.append(fold_data["search_results"])
            all_optimal_per_gesture_folds.append(fold_data["optimal_per_gesture"])

            for sid, lengths in fold_data["all_segment_lengths"].items():
                if sid not in all_segment_lengths_global:
                    all_segment_lengths_global[sid] = lengths

            if first_common_gestures is None:
                first_common_gestures = fold_data["common_gestures"]

            fr = fold_data["fold_result"]
            logger.info(
                f"[LOSO done] {test_subject} | "
                f"baseline acc={fr['baseline_accuracy']:.4f}  "
                f"global acc={fr['global_best_accuracy']:.4f}  "
                f"personalized acc={fr['personalized_accuracy']:.4f}"
            )

        except Exception as e:
            logger.error(f"Fold {test_subject} FAILED: {e}")
            traceback.print_exc()
            all_fold_results.append({
                "test_subject": test_subject,
                "error": str(e),
                "baseline_accuracy": None,
                "global_best_accuracy": None,
                "personalized_accuracy": None,
                "baseline_f1_macro": None,
                "global_best_f1_macro": None,
                "personalized_f1_macro": None,
                "baseline_f1_per_class": [],
                "personalized_f1_per_class": [],
            })

    # ── Aggregate & save summary ──────────────────────────────────────────────
    valid_folds = [r for r in all_fold_results if "error" not in r]
    summary: Dict = {
        "experiment": EXP_NAME,
        "subjects": ALL_SUBJECTS,
        "n_folds_total": len(ALL_SUBJECTS),
        "n_folds_valid": len(valid_folds),
        "window_sizes_searched": WINDOW_SIZES,
        "overlap_ratios_searched": OVERLAP_RATIOS,
        "baseline_config": {"window_size": BASELINE_WINDOW_SIZE, "overlap": BASELINE_OVERLAP},
    }

    if valid_folds:
        def _safe_mean(key: str) -> float:
            vals = [r[key] for r in valid_folds if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float('nan')

        def _safe_std(key: str) -> float:
            vals = [r[key] for r in valid_folds if r.get(key) is not None]
            return float(np.std(vals)) if vals else float('nan')

        for model_key, acc_key, f1_key in [
            ("baseline", "baseline_accuracy", "baseline_f1_macro"),
            ("global_best", "global_best_accuracy", "global_best_f1_macro"),
            ("personalized", "personalized_accuracy", "personalized_f1_macro"),
        ]:
            summary[model_key] = {
                "mean_accuracy": _safe_mean(acc_key),
                "std_accuracy": _safe_std(acc_key),
                "mean_f1_macro": _safe_mean(f1_key),
                "std_f1_macro": _safe_std(f1_key),
            }

        summary["delta_accuracy_pers_vs_baseline"] = (
            summary["personalized"]["mean_accuracy"]
            - summary["baseline"]["mean_accuracy"]
        )
        summary["delta_f1_pers_vs_baseline"] = (
            summary["personalized"]["mean_f1_macro"]
            - summary["baseline"]["mean_f1_macro"]
        )
        summary["delta_accuracy_global_vs_baseline"] = (
            summary["global_best"]["mean_accuracy"]
            - summary["baseline"]["mean_accuracy"]
        )

        logger.info("=" * 70)
        logger.info("LOSO SUMMARY")
        logger.info(
            f"  Baseline:     acc={summary['baseline']['mean_accuracy']:.4f} ± "
            f"{summary['baseline']['std_accuracy']:.4f}   "
            f"F1={summary['baseline']['mean_f1_macro']:.4f}"
        )
        logger.info(
            f"  Global-best:  acc={summary['global_best']['mean_accuracy']:.4f} ± "
            f"{summary['global_best']['std_accuracy']:.4f}   "
            f"F1={summary['global_best']['mean_f1_macro']:.4f}"
        )
        logger.info(
            f"  Personalized: acc={summary['personalized']['mean_accuracy']:.4f} ± "
            f"{summary['personalized']['std_accuracy']:.4f}   "
            f"F1={summary['personalized']['mean_f1_macro']:.4f}"
        )
        logger.info(
            f"  Δ acc (personalized − baseline) = "
            f"{summary['delta_accuracy_pers_vs_baseline']:+.4f}"
        )
        logger.info(
            f"  Δ F1  (personalized − baseline) = "
            f"{summary['delta_f1_pers_vs_baseline']:+.4f}"
        )
        logger.info("=" * 70)

    summary["fold_results"] = all_fold_results
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # ── Global visualizations ─────────────────────────────────────────────────
    logger.info("Generating global visualizations ...")
    common_gestures = first_common_gestures or list(range(_args.max_gestures))

    if all_segment_lengths_global:
        plot_segment_length_distribution(
            all_segment_lengths=all_segment_lengths_global,
            common_gestures=common_gestures,
            output_dir=OUTPUT_DIR,
        )
        logger.info("  ✓ segment_length_distribution.png")

    if all_search_results_per_fold:
        plot_window_config_search_heatmap(
            search_results_per_fold=all_search_results_per_fold,
            common_gestures=common_gestures,
            output_dir=OUTPUT_DIR,
        )
        logger.info("  ✓ window_config_search_heatmap.png")

        plot_window_size_sensitivity_curve(
            search_results_per_fold=all_search_results_per_fold,
            output_dir=OUTPUT_DIR,
        )
        logger.info("  ✓ window_size_sensitivity_curve.png")

        plot_overlap_sensitivity(
            search_results_per_fold=all_search_results_per_fold,
            output_dir=OUTPUT_DIR,
        )
        logger.info("  ✓ overlap_sensitivity.png")

        plot_window_yield_vs_accuracy(
            search_results_per_fold=all_search_results_per_fold,
            output_dir=OUTPUT_DIR,
        )
        logger.info("  ✓ window_yield_vs_accuracy.png")

    if all_optimal_per_gesture_folds:
        plot_optimal_window_per_gesture(
            optimal_per_gesture_all_folds=all_optimal_per_gesture_folds,
            common_gestures=common_gestures,
            output_dir=OUTPUT_DIR,
        )
        logger.info("  ✓ optimal_window_per_gesture.png")

        plot_config_diversity_heatmap(
            optimal_per_gesture_all_folds=all_optimal_per_gesture_folds,
            common_gestures=common_gestures,
            output_dir=OUTPUT_DIR,
        )
        logger.info("  ✓ config_diversity_heatmap.png")

    if valid_folds:
        plot_accuracy_comparison(fold_results=valid_folds, output_dir=OUTPUT_DIR)
        logger.info("  ✓ accuracy_comparison_per_fold.png + mean_accuracy_comparison.png")

        plot_per_gesture_f1_comparison(
            fold_results=valid_folds,
            common_gestures=common_gestures,
            output_dir=OUTPUT_DIR,
        )
        logger.info("  ✓ per_gesture_f1_comparison.png")

    logger.info(f"Experiment 45 complete.  Results → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
