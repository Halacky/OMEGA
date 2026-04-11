"""
Experiment 40: Window Quality Filtering for Cross-Subject EMG (LOSO)

Hypothesis H40:
    Classifying windows by SNR/noise/segmentation quality and discarding
    low-quality windows allows the model to train on cleaner signals,
    improving cross-subject transferability.

Quality metrics computed per window:
    1. SNR estimation      — signal power vs out-of-band residual
    2. Kurtosis score      — impulse-artifact detection
    3. ZCR regularity      — cross-channel zero-crossing consistency
    4. Saturation score    — fraction of clipped samples
    5. Channel correlation — adjacent-channel coherence
    6. RMS energy score    — outlier energy detection

Filtering strategies:
    - none             : baseline (all windows)
    - percentile 10/20/30 : remove worst N% by composite score
    - hard_threshold 0.3/0.4 : remove windows below absolute score

Test windows are NEVER filtered — evaluation is always fair.

Usage:
    python experiments/exp_40_window_quality_filtering_loso.py          # CI subjects
    python experiments/exp_40_window_quality_filtering_loso.py --ci
    python experiments/exp_40_window_quality_filtering_loso.py --full
    python experiments/exp_40_window_quality_filtering_loso.py --subjects DB2_s1,DB2_s12
"""

import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis as scipy_kurtosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Subject lists
# ---------------------------------------------------------------------------

_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]

_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]


def parse_subjects_args(argv=None) -> List[str]:
    """Parse --subjects / --ci / --full CLI args.  Defaults to CI subjects."""
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject IDs")
    _parser.add_argument("--ci",   action="store_true",
                         help="Use CI test subset (5 subjects)")
    _parser.add_argument("--full", action="store_true",
                         help="Use full 20-subject set")
    _args, _ = _parser.parse_known_args(argv)

    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    return _CI_SUBJECTS


def make_json_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ===========================================================================
#  Window Quality Analyzer
# ===========================================================================

class WindowQualityAnalyzer:
    """Compute per-window quality metrics for EMG signals.

    Input:  windows  (N, T, C) — raw EMG windows
    Output: dict of metric_name -> (N,) scores in [0, 1]  (higher = better)
    """

    # Default metric weights for composite score
    DEFAULT_WEIGHTS = {
        "snr": 0.25,
        "kurtosis": 0.20,
        "zcr_regularity": 0.10,
        "saturation": 0.15,
        "channel_corr": 0.15,
        "rms_energy": 0.15,
    }

    def __init__(self, sampling_rate: int = 2000):
        self.fs = sampling_rate
        # Pre-compute bandpass filter coefficients for SNR (20-450 Hz EMG band)
        self._sos_bp = butter(4, [20, 450], btype="band", fs=self.fs, output="sos")

    def analyze(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all quality metrics.

        Args:
            windows: (N, T, C) raw EMG

        Returns:
            dict with keys: snr, kurtosis, zcr_regularity, saturation,
            channel_corr, rms_energy, composite — all (N,) in [0, 1].
        """
        if windows.ndim != 3:
            raise ValueError(f"Expected (N, T, C), got shape {windows.shape}")

        metrics = {
            "snr":            self._compute_snr(windows),
            "kurtosis":       self._compute_kurtosis_score(windows),
            "zcr_regularity": self._compute_zcr_regularity(windows),
            "saturation":     self._compute_saturation_score(windows),
            "channel_corr":   self._compute_channel_correlation(windows),
            "rms_energy":     self._compute_rms_energy_score(windows),
        }
        metrics["composite"] = self.composite_score(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def _compute_snr(self, windows: np.ndarray) -> np.ndarray:
        """SNR via in-band vs out-of-band power.

        Bandpass 20-450 Hz → signal; residual → noise estimate.
        Score = sigmoid((SNR_dB - 10) / 5) so ~10 dB maps to 0.5.
        """
        N, T, C = windows.shape
        # Apply bandpass to all windows at once: sosfiltfilt along axis=1
        signal = sosfiltfilt(self._sos_bp, windows, axis=1)
        noise = windows - signal

        sig_power = np.mean(signal ** 2, axis=(1, 2))  # (N,)
        noise_power = np.mean(noise ** 2, axis=(1, 2))  # (N,)

        snr_db = 10.0 * np.log10(sig_power / (noise_power + 1e-12) + 1e-12)
        # Sigmoid normalization: center at 10 dB, scale 5
        score = 1.0 / (1.0 + np.exp(-(snr_db - 10.0) / 5.0))
        return score.astype(np.float64)

    def _compute_kurtosis_score(self, windows: np.ndarray) -> np.ndarray:
        """Kurtosis-based artifact detection.

        High kurtosis (>6) → impulse artifacts → bad.
        Score = 1 - sigmoid((mean_kurtosis - 5) / 2).
        """
        N, T, C = windows.shape
        # Per-channel kurtosis, averaged across channels
        # scipy kurtosis: Fisher=True (excess kurtosis), so Gaussian=0
        kurt = np.zeros((N, C), dtype=np.float64)
        for c in range(C):
            kurt[:, c] = scipy_kurtosis(windows[:, :, c], axis=1, fisher=True)

        mean_kurt = np.mean(np.abs(kurt), axis=1)  # (N,)
        # Invert: higher kurtosis → lower score
        score = 1.0 / (1.0 + np.exp((mean_kurt - 5.0) / 2.0))
        return score

    def _compute_zcr_regularity(self, windows: np.ndarray) -> np.ndarray:
        """Zero-crossing rate consistency across channels.

        Compute ZCR per channel; high cross-channel variance → inconsistent → bad.
        Score = exp(-cv^2 / 2) where cv = std(zcr) / (mean(zcr) + eps).
        """
        N, T, C = windows.shape
        # Count zero crossings per channel
        signs = np.sign(windows[:, 1:, :]) != np.sign(windows[:, :-1, :])
        zcr = signs.sum(axis=1).astype(np.float64)  # (N, C)

        mean_zcr = zcr.mean(axis=1)  # (N,)
        std_zcr = zcr.std(axis=1)    # (N,)

        cv = std_zcr / (mean_zcr + 1e-12)
        score = np.exp(-cv ** 2 / 2.0)
        return score

    def _compute_saturation_score(self, windows: np.ndarray) -> np.ndarray:
        """Amplitude saturation ratio.

        Fraction of samples within 5% of channel max absolute value.
        Score = 1 - saturation_ratio (capped at ratio=0.3 → score=0).
        """
        N, T, C = windows.shape
        abs_windows = np.abs(windows)
        # Per-channel max over time for each window
        ch_max = abs_windows.max(axis=1, keepdims=True)  # (N, 1, C)
        threshold = 0.95 * ch_max

        saturated = (abs_windows >= threshold).astype(np.float64)
        sat_ratio = saturated.mean(axis=(1, 2))  # (N,)

        # Map: ratio 0 → score 1, ratio 0.3+ → score ~0
        score = np.clip(1.0 - sat_ratio / 0.3, 0.0, 1.0)
        return score

    def _compute_channel_correlation(self, windows: np.ndarray) -> np.ndarray:
        """Adjacent-channel correlation consistency.

        Mean absolute Pearson correlation between adjacent channels.
        EMG from bracelet electrodes should show moderate correlation.
        Too low (<0.05) → dead electrode; Too high (>0.95) → common-mode artifact.
        Score = Gaussian around ideal range [0.1, 0.6].
        """
        N, T, C = windows.shape
        if C < 2:
            return np.ones(N, dtype=np.float64)

        # Compute correlation between adjacent channels
        corrs = np.zeros((N, C - 1), dtype=np.float64)
        for c in range(C - 1):
            # Vectorized Pearson correlation for each window
            x = windows[:, :, c]      # (N, T)
            y = windows[:, :, c + 1]  # (N, T)
            x_c = x - x.mean(axis=1, keepdims=True)
            y_c = y - y.mean(axis=1, keepdims=True)
            num = (x_c * y_c).sum(axis=1)
            den = np.sqrt((x_c ** 2).sum(axis=1) * (y_c ** 2).sum(axis=1)) + 1e-12
            corrs[:, c] = np.abs(num / den)

        mean_corr = corrs.mean(axis=1)  # (N,)

        # Penalize extreme values: ideal ~0.1-0.6
        # Score = 1 if in [0.1, 0.6], decreasing outside
        penalty_low = np.clip((mean_corr - 0.05) / 0.1, 0, 1)
        penalty_high = np.clip((0.9 - mean_corr) / 0.3, 0, 1)
        score = penalty_low * penalty_high
        return score

    def _compute_rms_energy_score(self, windows: np.ndarray) -> np.ndarray:
        """RMS energy outlier detection.

        Windows far from the median energy are suspicious.
        Score = exp(-((log_rms - median_log_rms)^2) / (2 * iqr^2)).
        """
        N, T, C = windows.shape
        rms = np.sqrt(np.mean(windows ** 2, axis=(1, 2)))  # (N,)
        log_rms = np.log(rms + 1e-12)

        median_log = np.median(log_rms)
        q25, q75 = np.percentile(log_rms, [25, 75])
        iqr = q75 - q25 + 1e-12

        # Gaussian kernel around median
        z = (log_rms - median_log) / iqr
        score = np.exp(-z ** 2 / 2.0)
        return score

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    def composite_score(
        self,
        metrics: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Weighted average of all metric scores."""
        w = weights or self.DEFAULT_WEIGHTS
        total_w = sum(w.values())

        N = len(next(iter(metrics.values())))
        composite = np.zeros(N, dtype=np.float64)
        for name, weight in w.items():
            if name in metrics:
                composite += (weight / total_w) * metrics[name]
        return composite


# ===========================================================================
#  Helper: grouped_to_arrays
# ===========================================================================

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
    gesture_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grouped windows dict to flat (windows, labels) arrays.

    Args:
        grouped_windows: {gesture_id: [rep_array, ...]}
        gesture_ids: if given, only include these gestures (sorted order → class idx)

    Returns:
        (windows, labels) where windows is (N_total, T, C) and labels is (N_total,)
        label values are class indices (position in sorted gesture_ids).
    """
    if gesture_ids is None:
        gesture_ids = sorted(grouped_windows.keys())
    else:
        gesture_ids = sorted(gesture_ids)

    gid_to_cls = {gid: i for i, gid in enumerate(gesture_ids)}

    X_parts, y_parts = [], []
    for gid in gesture_ids:
        if gid not in grouped_windows:
            continue
        reps = grouped_windows[gid]
        for rep in reps:
            if len(rep) > 0:
                X_parts.append(rep)
                y_parts.append(np.full(len(rep), gid_to_cls[gid], dtype=np.int64))

    if not X_parts:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def grouped_to_arrays_with_subject_ids(
    subjects_data: Dict[str, Tuple],
    gesture_ids: List[int],
    subject_ids: List[str],
    filter_fn=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like grouped_to_arrays but also returns per-window subject index.

    Returns:
        (windows, labels, subject_indices) all (N_total,)
    """
    gid_to_cls = {gid: i for i, gid in enumerate(sorted(gesture_ids))}
    sid_to_idx = {sid: i for i, sid in enumerate(subject_ids)}

    X_parts, y_parts, s_parts = [], [], []
    for sid in subject_ids:
        if sid not in subjects_data:
            continue
        _, _, grouped = subjects_data[sid]
        if filter_fn is not None:
            grouped = filter_fn(grouped, gesture_ids)
        for gid in sorted(gesture_ids):
            if gid not in grouped:
                continue
            for rep in grouped[gid]:
                if len(rep) > 0:
                    X_parts.append(rep)
                    y_parts.append(np.full(len(rep), gid_to_cls[gid], dtype=np.int64))
                    s_parts.append(np.full(len(rep), sid_to_idx[sid], dtype=np.int64))

    if not X_parts:
        empty_f = np.empty((0,), dtype=np.float32)
        empty_i = np.empty((0,), dtype=np.int64)
        return empty_f, empty_i, empty_i
    return (np.concatenate(X_parts), np.concatenate(y_parts),
            np.concatenate(s_parts))


# ===========================================================================
#  Window filtering
# ===========================================================================

def filter_windows_by_quality(
    windows: np.ndarray,
    labels: np.ndarray,
    quality_scores: np.ndarray,
    strategy: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter training windows based on quality scores.

    Args:
        windows: (N, T, C)
        labels:  (N,)
        quality_scores: (N,) composite quality in [0, 1]
        strategy: "none", "percentile", or "hard_threshold"
        threshold: for percentile — fraction to remove (0.1 = worst 10%);
                   for hard_threshold — minimum acceptable score.

    Returns:
        (filtered_windows, filtered_labels, kept_mask)
    """
    N = len(windows)

    if strategy == "none" or threshold <= 0:
        return windows, labels, np.ones(N, dtype=bool)

    if strategy == "percentile":
        cutoff = np.percentile(quality_scores, threshold * 100)
        mask = quality_scores >= cutoff
    elif strategy == "hard_threshold":
        mask = quality_scores >= threshold
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Safety: don't remove more than 80% of data
    if mask.sum() < 0.2 * N:
        # Keep at least top 20%
        top_k = max(int(0.2 * N), 1)
        indices = np.argsort(quality_scores)[-top_k:]
        mask = np.zeros(N, dtype=bool)
        mask[indices] = True

    return windows[mask], labels[mask], mask


def arrays_to_grouped(
    windows: np.ndarray,
    labels: np.ndarray,
    class_ids: List[int],
) -> Dict[int, List[np.ndarray]]:
    """Reverse of grouped_to_arrays: flat arrays → grouped_windows format.

    Returns Dict[gesture_id, [single_array]] with one "rep" per gesture.
    This is compatible with DatasetSplitter.split_grouped_windows().
    """
    grouped: Dict[int, List[np.ndarray]] = {}
    gid_to_cls = {gid: i for i, gid in enumerate(sorted(class_ids))}

    for gid in sorted(class_ids):
        cls_idx = gid_to_cls[gid]
        mask = labels == cls_idx
        if mask.any():
            arr = windows[mask]
            # Split into pseudo-repetitions of ~50 windows for splitter compatibility
            n = len(arr)
            rep_size = max(50, n // 6)  # at least 6 reps for reasonable splitting
            reps = []
            for i in range(0, n, rep_size):
                reps.append(arr[i:i + rep_size])
            grouped[gid] = reps
        else:
            grouped[gid] = []
    return grouped


# ===========================================================================
#  Visualizations
# ===========================================================================

def _save_fig(fig, path: Path, dpi: int = 150):
    """Save figure and close."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")


def plot_quality_distributions(
    all_metrics: Dict[str, np.ndarray],
    subject_indices: np.ndarray,
    subject_ids: List[str],
    output_dir: Path,
):
    """Per-metric histograms colored by subject."""
    metric_names = [k for k in all_metrics if k != "composite"]
    n_metrics = len(metric_names) + 1  # +1 for composite
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(subject_ids)))

    for idx, name in enumerate(metric_names + ["composite"]):
        ax = axes[idx]
        vals = all_metrics[name]
        for sid_idx, sid in enumerate(subject_ids):
            mask = subject_indices == sid_idx
            if mask.any():
                ax.hist(vals[mask], bins=40, alpha=0.5, color=colors[sid_idx],
                        label=sid, density=True)
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Window Quality Score Distributions by Subject", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir / "quality_distributions_by_subject.png")


def plot_quality_by_subject(
    composite_scores: np.ndarray,
    subject_indices: np.ndarray,
    subject_ids: List[str],
    output_dir: Path,
):
    """Boxplots of composite quality per subject."""
    fig, ax = plt.subplots(figsize=(max(6, len(subject_ids) * 1.2), 5))

    data = []
    for sid_idx in range(len(subject_ids)):
        mask = subject_indices == sid_idx
        data.append(composite_scores[mask])

    bp = ax.boxplot(data, labels=subject_ids, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(subject_ids)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Composite Quality Score")
    ax.set_xlabel("Subject")
    ax.set_title("Window Quality Distribution per Subject")
    ax.axhline(y=np.median(composite_scores), color="gray", linestyle="--",
               alpha=0.5, label=f"Overall median: {np.median(composite_scores):.3f}")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, output_dir / "quality_by_subject_boxplot.png")


def plot_quality_by_gesture(
    composite_scores: np.ndarray,
    labels: np.ndarray,
    class_ids: List[int],
    output_dir: Path,
):
    """Boxplots of composite quality per gesture class."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_ids) * 1.0), 5))

    gid_to_cls = {gid: i for i, gid in enumerate(sorted(class_ids))}
    data = []
    tick_labels = []
    for gid in sorted(class_ids):
        cls_idx = gid_to_cls[gid]
        mask = labels == cls_idx
        data.append(composite_scores[mask])
        name = "REST" if gid == 0 else f"G{gid}"
        tick_labels.append(name)

    bp = ax.boxplot(data, labels=tick_labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=5))

    colors = plt.cm.Paired(np.linspace(0, 1, len(class_ids)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Composite Quality Score")
    ax.set_xlabel("Gesture")
    ax.set_title("Window Quality Distribution per Gesture")
    fig.tight_layout()
    _save_fig(fig, output_dir / "quality_by_gesture_boxplot.png")


def plot_clean_vs_noisy_examples(
    windows: np.ndarray,
    composite_scores: np.ndarray,
    output_dir: Path,
    n_examples: int = 4,
):
    """Show top-quality and worst-quality window EMG traces side by side."""
    sorted_idx = np.argsort(composite_scores)
    worst_idx = sorted_idx[:n_examples]
    best_idx = sorted_idx[-n_examples:][::-1]

    C = windows.shape[2]
    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 6),
                             sharex=True)

    for col, (row_label, indices) in enumerate([
        ("Best (clean)", best_idx),
        ("Worst (noisy)", worst_idx),
    ]):
        for j, widx in enumerate(indices):
            ax = axes[col, j]
            w = windows[widx]  # (T, C)
            for c in range(min(C, 8)):
                offset = c * 0.5
                ax.plot(w[:, c] / (np.abs(w[:, c]).max() + 1e-12) * 0.4 + offset,
                        linewidth=0.5, alpha=0.8)
            ax.set_title(f"Q={composite_scores[widx]:.3f}", fontsize=9)
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")

    fig.suptitle("Clean vs Noisy Window Examples (all channels)", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, output_dir / "clean_vs_noisy_examples.png")


def plot_threshold_sweep(
    sweep_results: Dict[str, List[Dict]],
    output_dir: Path,
):
    """Line plot: filtering threshold vs accuracy/F1 for each model.

    sweep_results: {model_type: [{"strategy": ..., "threshold": ...,
                                   "mean_accuracy": ..., "mean_f1_macro": ...}, ...]}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    model_colors = plt.cm.Set1(np.linspace(0, 1, len(sweep_results)))

    for midx, (model_type, results) in enumerate(sweep_results.items()):
        # Sort by effective removal percentage
        percentile_results = sorted(
            [r for r in results if r["strategy"] in ("none", "percentile")],
            key=lambda r: r["threshold"],
        )
        if not percentile_results:
            continue

        x = [r["threshold"] * 100 for r in percentile_results]
        acc = [r["mean_accuracy"] for r in percentile_results if r["mean_accuracy"] is not None]
        f1 = [r["mean_f1_macro"] for r in percentile_results if r["mean_f1_macro"] is not None]
        x_valid = x[:len(acc)]

        color = model_colors[midx]
        ax1.plot(x_valid, acc, "o-", color=color, label=model_type, linewidth=2, markersize=6)
        ax2.plot(x_valid, f1, "s-", color=color, label=model_type, linewidth=2, markersize=6)

    ax1.set_xlabel("Windows Removed (%)")
    ax1.set_ylabel("Mean Test Accuracy")
    ax1.set_title("Accuracy vs Filtering Threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Windows Removed (%)")
    ax2.set_ylabel("Mean Test F1-macro")
    ax2.set_title("F1-macro vs Filtering Threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Impact of Window Quality Filtering on Model Performance", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, output_dir / "threshold_sweep_accuracy_f1.png")


def plot_quality_correlation_matrix(
    all_metrics: Dict[str, np.ndarray],
    output_dir: Path,
):
    """Heatmap of Pearson correlations between quality metrics."""
    metric_names = [k for k in all_metrics if k != "composite"]
    n = len(metric_names)

    corr_matrix = np.zeros((n, n))
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            corr_matrix[i, j] = np.corrcoef(all_metrics[m1], all_metrics[m2])[0, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    labels = [m.replace("_", "\n") for m in metric_names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Quality Metric Correlation Matrix", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, output_dir / "quality_metric_correlations.png")


def plot_subject_gesture_quality_heatmap(
    composite_scores: np.ndarray,
    labels: np.ndarray,
    subject_indices: np.ndarray,
    subject_ids: List[str],
    class_ids: List[int],
    output_dir: Path,
):
    """Heatmap: subjects x gestures, color = mean quality score."""
    gid_to_cls = {gid: i for i, gid in enumerate(sorted(class_ids))}
    n_subjects = len(subject_ids)
    n_gestures = len(class_ids)

    heatmap = np.full((n_subjects, n_gestures), np.nan)

    for sid_idx in range(n_subjects):
        for gid_idx, gid in enumerate(sorted(class_ids)):
            cls_idx = gid_to_cls[gid]
            mask = (subject_indices == sid_idx) & (labels == cls_idx)
            if mask.any():
                heatmap[sid_idx, gid_idx] = composite_scores[mask].mean()

    fig, ax = plt.subplots(figsize=(max(8, n_gestures * 0.8), max(5, n_subjects * 0.6)))

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto",
                   vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))

    gesture_labels = ["REST" if gid == 0 else f"G{gid}" for gid in sorted(class_ids)]
    ax.set_xticks(range(n_gestures))
    ax.set_xticklabels(gesture_labels, fontsize=9)
    ax.set_yticks(range(n_subjects))
    ax.set_yticklabels(subject_ids, fontsize=9)

    for i in range(n_subjects):
        for j in range(n_gestures):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if val > 0.5 else "white")

    fig.colorbar(im, ax=ax, label="Mean Quality Score")
    ax.set_title("Mean Window Quality: Subject x Gesture", fontsize=13)
    ax.set_xlabel("Gesture")
    ax.set_ylabel("Subject")
    fig.tight_layout()
    _save_fig(fig, output_dir / "subject_gesture_quality_heatmap.png")


def plot_filtering_impact_per_subject(
    baseline_results: List[Dict],
    best_filtered_results: List[Dict],
    best_strategy_label: str,
    output_dir: Path,
):
    """Grouped bar chart: per test-subject accuracy baseline vs best filtered."""
    subjects = [r["test_subject"] for r in baseline_results
                if r.get("test_accuracy") is not None]
    baseline_acc = {r["test_subject"]: r["test_accuracy"] for r in baseline_results
                    if r.get("test_accuracy") is not None}
    filtered_acc = {r["test_subject"]: r["test_accuracy"] for r in best_filtered_results
                    if r.get("test_accuracy") is not None}

    common_subjects = [s for s in subjects if s in filtered_acc]
    if not common_subjects:
        return

    x = np.arange(len(common_subjects))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(common_subjects) * 1.5), 9),
                                    gridspec_kw={"height_ratios": [3, 1]})

    b_vals = [baseline_acc.get(s, 0) for s in common_subjects]
    f_vals = [filtered_acc.get(s, 0) for s in common_subjects]

    ax1.bar(x - width / 2, b_vals, width, label="Baseline", color="#5DA5DA", alpha=0.85)
    ax1.bar(x + width / 2, f_vals, width, label=best_strategy_label, color="#FAA43A", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_subjects, rotation=30, ha="right")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Per-Subject Accuracy: Baseline vs Best Filtered Strategy")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Delta bar
    deltas = [f_vals[i] - b_vals[i] for i in range(len(common_subjects))]
    colors = ["#60BD68" if d >= 0 else "#F15854" for d in deltas]
    ax2.bar(x, deltas, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(common_subjects, rotation=30, ha="right")
    ax2.set_ylabel("Accuracy Delta")
    ax2.set_title("Improvement from Filtering (positive = better)")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, output_dir / "filtering_impact_per_subject.png")


def plot_confusion_matrix_comparison(
    baseline_cm: np.ndarray,
    filtered_cm: np.ndarray,
    class_names: List[str],
    baseline_label: str,
    filtered_label: str,
    output_dir: Path,
):
    """Side-by-side normalized confusion matrices."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    def _plot_cm(ax, cm, title):
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=8)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.5 else "black")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        return im

    _plot_cm(ax1, baseline_cm, f"Baseline ({baseline_label})")
    _plot_cm(ax2, filtered_cm, f"Filtered ({filtered_label})")

    # Difference heatmap
    b_norm = baseline_cm.astype(float) / (baseline_cm.sum(axis=1, keepdims=True) + 1e-12)
    f_norm = filtered_cm.astype(float) / (filtered_cm.sum(axis=1, keepdims=True) + 1e-12)
    diff = f_norm - b_norm

    im3 = ax3.imshow(diff, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax3.set_yticks(range(len(class_names)))
    ax3.set_yticklabels(class_names, fontsize=8)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = diff[i, j]
            ax3.text(j, i, f"{val:+.2f}", ha="center", va="center",
                     fontsize=7, color="black")
    ax3.set_title("Difference (Filtered - Baseline)", fontsize=11)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    fig.colorbar(im3, ax=ax3, label="Change")

    fig.suptitle("Confusion Matrix Comparison", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, output_dir / "confusion_matrix_comparison.png")


def plot_removed_windows_analysis(
    labels: np.ndarray,
    subject_indices: np.ndarray,
    kept_mask: np.ndarray,
    subject_ids: List[str],
    class_ids: List[int],
    all_metrics: Dict[str, np.ndarray],
    output_dir: Path,
):
    """Analyze characteristics of removed windows: gesture/subject distribution + metric breakdown."""
    removed = ~kept_mask
    n_removed = removed.sum()
    n_total = len(kept_mask)

    if n_removed == 0:
        return

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Gesture distribution of removed windows
    ax1 = fig.add_subplot(gs[0, 0])
    gid_to_cls = {gid: i for i, gid in enumerate(sorted(class_ids))}
    gesture_labels = ["REST" if gid == 0 else f"G{gid}" for gid in sorted(class_ids)]
    removed_per_gesture = []
    total_per_gesture = []
    for gid in sorted(class_ids):
        cls_idx = gid_to_cls[gid]
        gesture_mask = labels == cls_idx
        removed_per_gesture.append((gesture_mask & removed).sum())
        total_per_gesture.append(gesture_mask.sum())

    removal_rate = [r / (t + 1e-12) * 100 for r, t in zip(removed_per_gesture, total_per_gesture)]
    ax1.bar(range(len(gesture_labels)), removal_rate, color="#F15854", alpha=0.8)
    ax1.set_xticks(range(len(gesture_labels)))
    ax1.set_xticklabels(gesture_labels, fontsize=8)
    ax1.set_ylabel("Removal Rate (%)")
    ax1.set_title("Removal Rate by Gesture")
    ax1.axhline(y=n_removed / n_total * 100, color="gray", linestyle="--",
                alpha=0.6, label=f"Overall: {n_removed / n_total * 100:.1f}%")
    ax1.legend(fontsize=8)

    # 2. Subject distribution of removed windows
    ax2 = fig.add_subplot(gs[0, 1])
    removed_per_subject = []
    total_per_subject = []
    for sid_idx in range(len(subject_ids)):
        subj_mask = subject_indices == sid_idx
        removed_per_subject.append((subj_mask & removed).sum())
        total_per_subject.append(subj_mask.sum())

    removal_rate_subj = [r / (t + 1e-12) * 100
                         for r, t in zip(removed_per_subject, total_per_subject)]
    ax2.bar(range(len(subject_ids)), removal_rate_subj, color="#5DA5DA", alpha=0.8)
    ax2.set_xticks(range(len(subject_ids)))
    ax2.set_xticklabels(subject_ids, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Removal Rate (%)")
    ax2.set_title("Removal Rate by Subject")

    # 3. Quality metric distributions: removed vs kept
    metric_names = [k for k in all_metrics if k != "composite"]
    ax3 = fig.add_subplot(gs[0, 2])
    kept_means = [all_metrics[m][kept_mask].mean() for m in metric_names]
    removed_means = [all_metrics[m][removed].mean() for m in metric_names]
    x = np.arange(len(metric_names))
    w = 0.35
    ax3.bar(x - w / 2, kept_means, w, label="Kept", color="#60BD68", alpha=0.8)
    ax3.bar(x + w / 2, removed_means, w, label="Removed", color="#F15854", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace("_", "\n") for m in metric_names], fontsize=7)
    ax3.set_ylabel("Mean Score")
    ax3.set_title("Mean Metric Scores: Kept vs Removed")
    ax3.legend(fontsize=8)

    # 4. Composite score histogram: kept vs removed
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(all_metrics["composite"][kept_mask], bins=50, alpha=0.6,
             color="#60BD68", label="Kept", density=True)
    ax4.hist(all_metrics["composite"][removed], bins=50, alpha=0.6,
             color="#F15854", label="Removed", density=True)
    ax4.set_xlabel("Composite Quality Score")
    ax4.set_ylabel("Density")
    ax4.set_title("Composite Score Distribution")
    ax4.legend()
    ax4.axvline(x=all_metrics["composite"][removed].max(), color="red",
                linestyle="--", alpha=0.5, label="Cutoff")

    # 5. Summary stats text
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    summary_text = (
        f"Total windows: {n_total:,}\n"
        f"Removed: {n_removed:,} ({n_removed / n_total * 100:.1f}%)\n"
        f"Kept: {(n_total - n_removed):,} ({(n_total - n_removed) / n_total * 100:.1f}%)\n\n"
        f"Composite quality (kept):    mean={all_metrics['composite'][kept_mask].mean():.4f}, "
        f"std={all_metrics['composite'][kept_mask].std():.4f}\n"
        f"Composite quality (removed): mean={all_metrics['composite'][removed].mean():.4f}, "
        f"std={all_metrics['composite'][removed].std():.4f}\n\n"
    )
    # Per-metric comparison
    for m in metric_names:
        km = all_metrics[m][kept_mask].mean()
        rm = all_metrics[m][removed].mean()
        summary_text += f"  {m:20s}: kept={km:.4f}  removed={rm:.4f}  delta={km - rm:+.4f}\n"

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.suptitle("Analysis of Removed Windows", fontsize=14)
    _save_fig(fig, output_dir / "removed_windows_analysis.png")


# ===========================================================================
#  LOSO fold runner
# ===========================================================================

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    approach: str,
    strategy: str,
    threshold: float,
    proc_cfg,
    split_cfg,
    train_cfg,
    quality_analyzer: WindowQualityAnalyzer,
) -> Dict:
    """Run one LOSO fold with quality-based window filtering on train data.

    Pipeline:
    1. Load all subjects
    2. Get common gestures
    3. Merge train windows → flat arrays
    4. Compute quality scores on train windows
    5. Apply filtering strategy
    6. Rebuild grouped_windows → split via DatasetSplitter
    7. Test = unfiltered test_subject windows
    8. trainer.fit(splits) → evaluate_numpy(X_test, y_test)
    """
    import torch
    from config.cross_subject import CrossSubjectConfig
    from data.multi_subject_loader import MultiSubjectLoader
    from training.trainer import WindowClassifierTrainer, FeatureMLTrainer
    from processing.splitting import DatasetSplitter
    from visualization.base import Visualizer
    from utils.logging import setup_logging, seed_everything
    from utils.artifacts import ArtifactSaver

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    strategy_label = f"{strategy}_{threshold}" if strategy != "none" else "none"

    # ---- 1. Load data ----
    all_subject_ids = list(set(train_subjects + [test_subject]))
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=True,
    )

    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    if len(common_gestures) == 0:
        raise ValueError("No common gestures found")

    # ---- 2. Prepare train data (merged) ----
    train_filtered_data = {}
    for sid in train_subjects:
        if sid not in subjects_data:
            continue
        _, _, grouped = subjects_data[sid]
        train_filtered_data[sid] = multi_loader.filter_by_gestures(grouped, common_gestures)

    # Merge train windows
    train_grouped = multi_loader.merge_grouped_windows(
        {sid: (None, None, train_filtered_data[sid]) for sid in train_filtered_data},
        list(train_filtered_data.keys()),
    )

    # Flatten to arrays for quality analysis
    train_windows, train_labels = grouped_to_arrays(train_grouped, common_gestures)

    logger.info(f"Train windows before filtering: {len(train_windows)}")

    # ---- 3. Compute quality scores ----
    quality_metrics = quality_analyzer.analyze(train_windows)
    composite = quality_metrics["composite"]

    # ---- 4. Apply filtering ----
    filtered_windows, filtered_labels, kept_mask = filter_windows_by_quality(
        train_windows, train_labels, composite, strategy, threshold,
    )

    n_removed = (~kept_mask).sum()
    n_kept = kept_mask.sum()
    logger.info(
        f"Quality filtering ({strategy_label}): "
        f"kept={n_kept}, removed={n_removed} "
        f"({n_removed / len(kept_mask) * 100:.1f}%)"
    )

    # ---- 5. Rebuild grouped format + split into train/val ----
    filtered_grouped = arrays_to_grouped(filtered_windows, filtered_labels, common_gestures)

    splitter = DatasetSplitter(split_cfg, logger)
    train_val_splits, _ = splitter.split_grouped_windows(filtered_grouped)

    # ---- 6. Prepare test data (UNFILTERED) ----
    _, _, test_grouped_raw = subjects_data[test_subject]
    test_grouped = multi_loader.filter_by_gestures(test_grouped_raw, common_gestures)
    test_split = {}
    for gid, reps in test_grouped.items():
        if reps:
            test_split[gid] = np.concatenate(reps, axis=0)
        else:
            test_split[gid] = np.empty((0,), dtype=np.float32)

    splits = {
        "train": train_val_splits["train"],
        "val": train_val_splits["val"],
        "test": test_split,
    }

    # Log split sizes
    for sname in ["train", "val", "test"]:
        total = sum(len(arr) for arr in splits[sname].values()
                    if isinstance(arr, np.ndarray) and arr.ndim == 3)
        logger.info(f"  {sname.upper()}: {total} windows")

    # ---- 7. Create trainer ----
    train_cfg_copy = train_cfg  # configs are shared but overridden fields are same
    train_cfg_copy.pipeline_type = approach
    train_cfg_copy.model_type = model_type

    base_viz = Visualizer(output_dir, logger)

    if approach in ("deep_raw",):
        trainer = WindowClassifierTrainer(
            train_cfg=train_cfg_copy,
            logger=logger,
            output_dir=output_dir,
            visualizer=base_viz,
        )
    elif approach == "ml_emg_td":
        trainer = FeatureMLTrainer(
            train_cfg=train_cfg_copy,
            logger=logger,
            output_dir=output_dir,
            visualizer=base_viz,
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # ---- 8. Train ----
    try:
        training_results = trainer.fit(splits)
        if training_results is None:
            training_results = {}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "strategy": strategy_label,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
            "quality_stats": {
                "n_total": int(len(train_windows)),
                "n_kept": int(n_kept),
                "n_removed": int(n_removed),
            },
        }

    # ---- 9. Evaluate on test subject ----
    X_test_parts, y_test_parts = [], []
    for gid in sorted(common_gestures):
        if gid in test_split and isinstance(test_split[gid], np.ndarray) and test_split[gid].ndim == 3:
            cls_idx = trainer.class_ids.index(gid)
            X_test_parts.append(test_split[gid])
            y_test_parts.append(np.full(len(test_split[gid]), cls_idx, dtype=np.int64))

    X_test_cat = np.concatenate(X_test_parts, axis=0)
    y_test_cat = np.concatenate(y_test_parts, axis=0)

    test_results = trainer.evaluate_numpy(
        X_test_cat, y_test_cat,
        split_name=f"cs_test_{test_subject}_{strategy_label}",
        visualize=False,
    )

    test_acc = float(test_results.get("accuracy", 0.0))
    test_f1 = float(test_results.get("f1_macro", 0.0))

    logger.info(
        f"[LOSO] {test_subject} | {model_type} | {strategy_label} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )

    # ---- Save fold results ----
    fold_result = {
        "test_subject": test_subject,
        "model_type": model_type,
        "strategy": strategy_label,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "confusion_matrix": test_results.get("confusion_matrix"),
        "quality_stats": {
            "n_total": int(len(train_windows)),
            "n_kept": int(n_kept),
            "n_removed": int(n_removed),
            "removal_pct": float(n_removed / len(train_windows) * 100),
            "mean_quality_kept": float(composite[kept_mask].mean()) if n_kept > 0 else None,
            "mean_quality_removed": float(composite[~kept_mask].mean()) if n_removed > 0 else None,
        },
    }

    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(make_json_serializable(fold_result), f, indent=4, ensure_ascii=False)

    # ---- Cleanup ----
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del trainer, multi_loader, subjects_data, base_viz
    gc.collect()

    return fold_result


# ===========================================================================
#  Main
# ===========================================================================

def main():
    EXPERIMENT_NAME = "exp_40_window_quality_filtering_loso"
    BASE_DIR = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()

    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXERCISES = ["E1"]

    # Filtering strategies: (strategy_name, threshold_value)
    STRATEGIES = [
        ("none",           0.0),    # baseline
        ("percentile",     0.10),   # remove worst 10%
        ("percentile",     0.20),   # remove worst 20%
        ("percentile",     0.30),   # remove worst 30%
        ("hard_threshold", 0.3),    # absolute score cutoff 0.3
        ("hard_threshold", 0.4),    # absolute score cutoff 0.4
    ]

    # Models: ML + deep to demonstrate generalizability
    MODEL_CONFIGS = [
        ("svm_rbf",    "ml_emg_td"),
        ("simple_cnn", "deep_raw"),
    ]

    from config.base import ProcessingConfig, SplitConfig, TrainingConfig
    from utils.logging import setup_logging

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

    # Shared training config (overridden per model)
    train_cfg_base = TrainingConfig(
        batch_size=4096,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=False,
        seed=42,
        num_workers=0,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        handcrafted_feature_set="emg_td",
        pipeline_type="deep_raw",
        ml_model_type="svm_rbf",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
    )

    global_logger = setup_logging(OUTPUT_DIR)
    quality_analyzer = WindowQualityAnalyzer(sampling_rate=proc_cfg.sampling_rate)

    print(f"[{EXPERIMENT_NAME}] Subjects  : {ALL_SUBJECTS}")
    print(f"[{EXPERIMENT_NAME}] Strategies: {STRATEGIES}")
    print(f"[{EXPERIMENT_NAME}] Models    : {[m for m, _ in MODEL_CONFIGS]}")
    print(f"[{EXPERIMENT_NAME}] Exercises : {EXERCISES}")

    # =====================================================================
    # Phase 1: Pre-compute quality analysis on all data
    # =====================================================================
    print("\n" + "=" * 70)
    print("  Phase 1: Quality Analysis on All Data")
    print("=" * 70)

    from data.multi_subject_loader import MultiSubjectLoader

    analysis_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=global_logger,
        use_gpu=True,
        use_improved_processing=True,
    )

    all_subjects_data = analysis_loader.load_multiple_subjects(
        base_dir=BASE_DIR,
        subject_ids=ALL_SUBJECTS,
        exercises=EXERCISES,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = analysis_loader.get_common_gestures(all_subjects_data, max_gestures=10)
    print(f"Common gestures: {common_gestures}")

    # Flatten ALL windows with subject tracking
    all_windows, all_labels, all_subj_idx = grouped_to_arrays_with_subject_ids(
        all_subjects_data,
        gesture_ids=common_gestures,
        subject_ids=ALL_SUBJECTS,
        filter_fn=analysis_loader.filter_by_gestures,
    )

    print(f"Total windows across all subjects: {len(all_windows)}")
    print(f"Computing quality metrics...")

    all_quality_metrics = quality_analyzer.analyze(all_windows)

    # Summary stats
    print("\nQuality metric summary:")
    for name, vals in all_quality_metrics.items():
        print(f"  {name:20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")

    # ---- Generate quality analysis visualizations ----
    qa_dir = OUTPUT_DIR / "quality_analysis"
    qa_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating quality analysis visualizations...")

    plot_quality_distributions(all_quality_metrics, all_subj_idx, ALL_SUBJECTS, qa_dir)
    plot_quality_by_subject(all_quality_metrics["composite"], all_subj_idx,
                            ALL_SUBJECTS, qa_dir)
    plot_quality_by_gesture(all_quality_metrics["composite"], all_labels,
                            common_gestures, qa_dir)
    plot_clean_vs_noisy_examples(all_windows, all_quality_metrics["composite"], qa_dir)
    plot_quality_correlation_matrix(all_quality_metrics, qa_dir)
    plot_subject_gesture_quality_heatmap(
        all_quality_metrics["composite"], all_labels, all_subj_idx,
        ALL_SUBJECTS, common_gestures, qa_dir,
    )

    # Save quality stats
    quality_summary = {
        "n_windows": int(len(all_windows)),
        "n_subjects": len(ALL_SUBJECTS),
        "common_gestures": [int(g) for g in common_gestures],
        "metrics": {
            name: {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "percentiles": {
                    str(p): float(np.percentile(vals, p))
                    for p in [5, 10, 25, 50, 75, 90, 95]
                },
            }
            for name, vals in all_quality_metrics.items()
        },
        "per_subject": {
            sid: {
                "n_windows": int((all_subj_idx == i).sum()),
                "mean_composite": float(all_quality_metrics["composite"][all_subj_idx == i].mean()),
                "std_composite": float(all_quality_metrics["composite"][all_subj_idx == i].std()),
            }
            for i, sid in enumerate(ALL_SUBJECTS)
        },
    }
    with open(qa_dir / "quality_analysis_summary.json", "w") as f:
        json.dump(quality_summary, f, indent=4, ensure_ascii=False)

    # Free analysis data
    del all_windows, all_labels, all_subj_idx, all_subjects_data, analysis_loader
    import gc
    gc.collect()

    # =====================================================================
    # Phase 2: Run LOSO experiments for all strategies
    # =====================================================================
    print("\n" + "=" * 70)
    print("  Phase 2: LOSO Experiments with Quality Filtering")
    print("=" * 70)

    all_loso_results: List[Dict] = []

    for model_type, approach in MODEL_CONFIGS:
        # Create per-model training config
        import copy
        train_cfg = copy.deepcopy(train_cfg_base)

        if approach == "ml_emg_td":
            train_cfg.pipeline_type = "ml_emg_td"
            train_cfg.ml_model_type = model_type
            train_cfg.use_handcrafted_features = True
            train_cfg.epochs = 1
            train_cfg.device = "cpu"
        elif approach == "deep_raw":
            train_cfg.pipeline_type = "deep_raw"
            train_cfg.model_type = model_type
            train_cfg.use_handcrafted_features = False

        for strategy, threshold in STRATEGIES:
            strategy_label = f"{strategy}_{threshold}" if strategy != "none" else "none"

            print(f"\n{'─' * 60}")
            print(f"  Model: {model_type} | Strategy: {strategy_label}")
            print(f"{'─' * 60}")

            for test_subject in ALL_SUBJECTS:
                train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
                fold_dir = (OUTPUT_DIR / model_type / strategy_label
                            / f"test_{test_subject}")

                fold_res = run_single_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    model_type=model_type,
                    approach=approach,
                    strategy=strategy,
                    threshold=threshold,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                    quality_analyzer=quality_analyzer,
                )
                all_loso_results.append(fold_res)

    # =====================================================================
    # Phase 3: Aggregate and visualize results
    # =====================================================================
    print("\n" + "=" * 70)
    print("  Phase 3: Aggregation & Visualization")
    print("=" * 70)

    # Aggregate by (model_type, strategy)
    aggregate_results: Dict = {}
    sweep_results: Dict[str, List[Dict]] = {}

    for model_type, _ in MODEL_CONFIGS:
        if model_type not in sweep_results:
            sweep_results[model_type] = []

        for strategy, threshold in STRATEGIES:
            strategy_label = f"{strategy}_{threshold}" if strategy != "none" else "none"
            key = f"{model_type}__{strategy_label}"

            fold_results = [
                r for r in all_loso_results
                if r["model_type"] == model_type
                and r["strategy"] == strategy_label
                and r.get("test_accuracy") is not None
            ]

            if not fold_results:
                aggregate_results[key] = {
                    "model_type": model_type,
                    "strategy": strategy_label,
                    "mean_accuracy": None,
                    "mean_f1_macro": None,
                    "num_subjects": 0,
                }
                continue

            accs = [r["test_accuracy"] for r in fold_results]
            f1s = [r["test_f1_macro"] for r in fold_results]

            agg = {
                "model_type": model_type,
                "strategy": strategy_label,
                "strategy_raw": strategy,
                "threshold": threshold,
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_subjects": len(fold_results),
                "per_subject": fold_results,
            }
            aggregate_results[key] = agg

            sweep_results[model_type].append({
                "strategy": strategy,
                "threshold": threshold,
                "mean_accuracy": float(np.mean(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_accuracy": float(np.std(accs)),
                "std_f1_macro": float(np.std(f1s)),
            })

    # ---- Print summary table ----
    print(f"\n{'Model':15s} {'Strategy':20s} {'Accuracy':>12s} {'F1-macro':>12s}")
    print("─" * 65)
    for key, agg in aggregate_results.items():
        m_acc = agg.get("mean_accuracy")
        s_acc = agg.get("std_accuracy", 0)
        m_f1 = agg.get("mean_f1_macro")
        s_f1 = agg.get("std_f1_macro", 0)
        acc_str = f"{m_acc:.4f}±{s_acc:.4f}" if m_acc is not None else "N/A"
        f1_str = f"{m_f1:.4f}±{s_f1:.4f}" if m_f1 is not None else "N/A"
        print(f"{agg['model_type']:15s} {agg['strategy']:20s} {acc_str:>12s} {f1_str:>12s}")

    # ---- Generate comparison visualizations ----
    viz_dir = OUTPUT_DIR / "comparison_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Threshold sweep
    plot_threshold_sweep(sweep_results, viz_dir)

    # Per-subject impact (baseline vs best filtered, per model)
    for model_type, _ in MODEL_CONFIGS:
        baseline_key = f"{model_type}__none"
        if baseline_key not in aggregate_results:
            continue
        baseline_folds = aggregate_results[baseline_key].get("per_subject", [])
        if not baseline_folds:
            continue

        # Find best non-baseline strategy for this model
        best_key = None
        best_f1 = -1
        for key, agg in aggregate_results.items():
            if (agg["model_type"] == model_type
                    and agg["strategy"] != "none"
                    and agg.get("mean_f1_macro") is not None
                    and agg["mean_f1_macro"] > best_f1):
                best_f1 = agg["mean_f1_macro"]
                best_key = key

        if best_key is not None:
            best_folds = aggregate_results[best_key].get("per_subject", [])
            best_label = aggregate_results[best_key]["strategy"]

            plot_filtering_impact_per_subject(
                baseline_folds, best_folds, best_label,
                viz_dir,
            )

            # Confusion matrix comparison (aggregate across folds)
            baseline_cms = [
                np.array(r["confusion_matrix"])
                for r in baseline_folds
                if r.get("confusion_matrix") is not None
            ]
            filtered_cms = [
                np.array(r["confusion_matrix"])
                for r in best_folds
                if r.get("confusion_matrix") is not None
            ]

            if baseline_cms and filtered_cms:
                # Sum confusion matrices across folds
                avg_base_cm = sum(baseline_cms)
                avg_filt_cm = sum(filtered_cms)
                n_classes = avg_base_cm.shape[0]
                class_names = [f"G{gid}" if gid != 0 else "REST"
                               for gid in sorted(common_gestures)[:n_classes]]

                plot_confusion_matrix_comparison(
                    avg_base_cm, avg_filt_cm, class_names,
                    "none", best_label,
                    viz_dir,
                )

    # =====================================================================
    # Phase 4: Save summary
    # =====================================================================
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "Removing low-quality EMG windows (by SNR/kurtosis/saturation/"
            "ZCR/channel-correlation/RMS-energy) from training data improves "
            "cross-subject classification transferability."
        ),
        "quality_metrics": list(WindowQualityAnalyzer.DEFAULT_WEIGHTS.keys()),
        "quality_weights": WindowQualityAnalyzer.DEFAULT_WEIGHTS,
        "strategies": [
            {"strategy": s, "threshold": t} for s, t in STRATEGIES
        ],
        "models": [m for m, _ in MODEL_CONFIGS],
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "aggregate_results": aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)

    print(f"\n[DONE] {EXPERIMENT_NAME} -> {summary_path}")

    # ---- Hypothesis executor notification ----
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        # Determine if filtering helped: compare best filtered vs baseline
        baseline_f1s = {}
        best_filtered_f1s = {}
        for model_type, _ in MODEL_CONFIGS:
            base_agg = aggregate_results.get(f"{model_type}__none", {})
            if base_agg.get("mean_f1_macro") is not None:
                baseline_f1s[model_type] = base_agg["mean_f1_macro"]

            best_f1_model = -1
            best_strat = "none"
            for key, agg in aggregate_results.items():
                if (agg["model_type"] == model_type
                        and agg["strategy"] != "none"
                        and agg.get("mean_f1_macro") is not None
                        and agg["mean_f1_macro"] > best_f1_model):
                    best_f1_model = agg["mean_f1_macro"]
                    best_strat = agg["strategy"]
            if best_f1_model > -1:
                best_filtered_f1s[model_type] = best_f1_model

        improved = any(
            best_filtered_f1s.get(m, 0) > baseline_f1s.get(m, 0)
            for m in baseline_f1s
        )

        metrics = {
            "baseline_f1": baseline_f1s,
            "best_filtered_f1": best_filtered_f1s,
            "improved_any_model": improved,
            "aggregate_results": {
                k: {kk: vv for kk, vv in v.items() if kk != "per_subject"}
                for k, v in aggregate_results.items()
            },
        }

        if improved:
            mark_hypothesis_verified(
                hypothesis_id="H40",
                metrics=metrics,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                hypothesis_id="H40",
                error_message=(
                    f"Filtering did not improve F1 for any model. "
                    f"Baseline: {baseline_f1s}, Best filtered: {best_filtered_f1s}"
                ),
            )
    except ImportError:
        pass
    except Exception as _he_err:
        print(f"[{EXPERIMENT_NAME}] hypothesis_executor notification failed: {_he_err}")


if __name__ == "__main__":
    main()
