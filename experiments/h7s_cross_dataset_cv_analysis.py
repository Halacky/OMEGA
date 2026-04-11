#!/usr/bin/env python3
"""
H7s: Cross-Dataset CV Gradient Analysis (Exploratory).

Verifies that the frequency-dependent CV gradient (originally shown in H1)
reproduces across different exercises within NinaPro DB2, serving as a
proxy for cross-dataset reproducibility.

Question
--------
  Is the monotonic CV gradient (higher inter-subject CV at lower
  frequencies) consistent across exercises E1, E2, E3?

Method
------
  1. Load raw EMG from DB2 for each exercise (E1, E2, E3)
  2. For each exercise, compute PSD via Welch's method per subject/gesture
  3. Compute normalised inter-subject CV across 6 frequency bands:
     [20-100, 100-200, 200-350, 350-500, 500-700, 700-1000] Hz
  4. Compute Fisher discriminant ratio per band
  5. Compare CV gradients across exercises with Spearman correlation

Protocol
--------
  No LOSO training -- pure signal analysis.
  Each exercise is treated as a separate "dataset".
  Strong Spearman correlation (rho > 0.8) between exercises confirms
  that the CV gradient is a stable property of sEMG, not an artefact
  of specific gestures.

Output
------
  experiments_output/h7s_cross_dataset_cv_analysis_<timestamp>/
    +-- cv_gradient_analysis.json
    +-- experiment.log

Usage
-----
  python experiments/h7s_cross_dataset_cv_analysis.py
  python experiments/h7s_cross_dataset_cv_analysis.py --full
  python experiments/h7s_cross_dataset_cv_analysis.py --exercises E1 E2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import loadmat
from scipy.stats import spearmanr

# -- project imports ----------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logging import setup_logging

# -- constants ----------------------------------------------------------------
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [
    f"DB2_s{i}" for i in
    [1, 2, 3, 4, 5, 11, 12, 13, 14, 15,
     26, 27, 28, 29, 30, 36, 37, 38, 39, 40]
]

SEED = 42
SAMPLING_RATE = 2000
N_CHANNELS = 12
WINDOW_SIZE = 200
K_MODES = 4

EXPERIMENT_NAME = "h7s_cross_dataset_cv_analysis"

# Frequency bands (same as H1)
FREQ_BANDS = {
    "20-100 Hz":   (20, 100),
    "100-200 Hz":  (100, 200),
    "200-350 Hz":  (200, 350),
    "350-500 Hz":  (350, 500),
    "500-700 Hz":  (500, 700),
    "700-1000 Hz": (700, 1000),
}
BAND_NAMES = list(FREQ_BANDS.keys())
BAND_RANGES = list(FREQ_BANDS.values())

# Exercise -> gesture IDs mapping (NinaPro DB2)
EXERCISE_GESTURES = {
    "E1": list(range(1, 18)),   # 17 basic finger movements
    "E2": list(range(1, 24)),   # 23 wrist/hand movements
    "E3": list(range(1, 10)),   # 9 force patterns
}


# =============================================================================
#  Helpers
# =============================================================================

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grouped windows to flat (windows, labels) arrays."""
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            windows_list.append(rep_arr)
            labels_list.extend([gid] * len(rep_arr))
    return np.concatenate(windows_list, axis=0), np.array(labels_list)


def load_raw_segments(
    base_dir: Path,
    subject_id: str,
    exercise: str,
    logger: logging.Logger,
) -> Optional[Dict[int, List[np.ndarray]]]:
    """
    Load raw EMG segments for a single subject and exercise.

    Returns dict {gesture_id: [segment_array(T_i, C), ...]} or None if
    the data file is missing.
    """
    subject_num = subject_id.split("_s")[1]
    ex_num = exercise[1:]  # "E1" -> "1"
    file_path = base_dir / subject_id / f"S{subject_num}_E{ex_num}_A1.mat"

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}, skipping {subject_id}/{exercise}")
        return None

    data = loadmat(str(file_path))
    emg = data["emg"]               # (total_samples, C)
    stimulus = data["stimulus"].flatten()

    gesture_ids = EXERCISE_GESTURES.get(exercise, list(range(1, 18)))

    segments: Dict[int, List[np.ndarray]] = {}
    stim_diff = np.diff(stimulus, prepend=0)
    changes = np.where(stim_diff != 0)[0]

    for i in range(len(changes)):
        start = changes[i]
        end = changes[i + 1] if i + 1 < len(changes) else len(stimulus)
        gid = int(stimulus[start])
        if gid == 0:
            continue
        if gid not in gesture_ids:
            continue
        seg = emg[start:end].copy()
        if gid not in segments:
            segments[gid] = []
        segments[gid].append(seg)

    logger.info(
        f"  {subject_id}/{exercise}: {sum(len(v) for v in segments.values())} "
        f"segments across {len(segments)} gestures"
    )
    return segments


def compute_psd_welch(
    segment: np.ndarray,
    fs: int = SAMPLING_RATE,
    nperseg: int = 512,
    noverlap: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD for each channel using Welch's method.

    Parameters
    ----------
    segment : ndarray, shape (T, C)
    fs : sampling frequency

    Returns
    -------
    freqs : ndarray, shape (F,)
    psd : ndarray, shape (C, F)
    """
    n_channels = segment.shape[1]
    psds = []
    for ch in range(n_channels):
        freqs, pxx = scipy_signal.welch(
            segment[:, ch], fs=fs,
            nperseg=min(nperseg, segment.shape[0]),
            noverlap=min(noverlap, segment.shape[0] // 2),
            window="hann",
        )
        psds.append(pxx)
    return freqs, np.array(psds)


def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float],
) -> np.ndarray:
    """
    Compute mean power in a frequency band for each channel.

    Parameters
    ----------
    freqs : (F,)
    psd : (C, F)
    band : (low_hz, high_hz)

    Returns
    -------
    (C,) mean power per channel in the band.
    """
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not mask.any():
        return np.zeros(psd.shape[0])
    return np.mean(psd[:, mask], axis=1)


def compute_fisher_ratio(
    band_powers: Dict[int, np.ndarray],
) -> float:
    """
    Fisher discriminant ratio: between-class variance / within-class variance.

    Parameters
    ----------
    band_powers : {gesture_id: (n_subjects, C)} array of band powers

    Returns
    -------
    float: Fisher ratio averaged across channels.
    """
    if len(band_powers) < 2:
        return 0.0

    all_powers = []
    for gid in sorted(band_powers.keys()):
        all_powers.append(band_powers[gid])

    # Per-channel Fisher ratio
    # Global mean per channel: average across all subjects and gestures
    all_concat = np.concatenate(all_powers, axis=0)  # (total_obs, C)
    global_mean = all_concat.mean(axis=0)  # (C,)

    between_var = np.zeros(all_concat.shape[1])
    within_var = np.zeros(all_concat.shape[1])

    for class_data in all_powers:
        n_k = class_data.shape[0]
        class_mean = class_data.mean(axis=0)
        between_var += n_k * (class_mean - global_mean) ** 2
        within_var += ((class_data - class_mean) ** 2).sum(axis=0)

    within_var = within_var / max(len(all_concat) - len(all_powers), 1)
    between_var = between_var / max(len(all_powers) - 1, 1)

    fisher = between_var / (within_var + 1e-10)
    return float(fisher.mean())


# =============================================================================
#  Per-Exercise Analysis
# =============================================================================

def analyse_exercise(
    base_dir: Path,
    exercise: str,
    subjects: List[str],
    logger: logging.Logger,
) -> Optional[Dict]:
    """
    Compute per-band inter-subject CV and Fisher ratio for one exercise.

    Returns
    -------
    dict with keys: band_cv, band_fisher, band_mean_power, n_subjects_loaded
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Analysing exercise: {exercise}")
    logger.info(f"{'='*60}")

    # band_power_per_subj[band_idx][gesture_id] = list of (C,) per subject
    band_power_per_subj: Dict[int, Dict[int, List[np.ndarray]]] = {
        b: {} for b in range(len(BAND_RANGES))
    }

    loaded_subjects = []

    for subj_id in subjects:
        segments = load_raw_segments(base_dir, subj_id, exercise, logger)
        if segments is None:
            continue
        loaded_subjects.append(subj_id)

        for gid, segs in segments.items():
            for seg in segs:
                if seg.shape[0] < 128:
                    continue
                freqs, psd = compute_psd_welch(seg)

                for b_idx, band in enumerate(BAND_RANGES):
                    bp = compute_band_power(freqs, psd, band)  # (C,)
                    if gid not in band_power_per_subj[b_idx]:
                        band_power_per_subj[b_idx][gid] = []
                    band_power_per_subj[b_idx][gid].append(bp)

    if len(loaded_subjects) < 2:
        logger.warning(f"  Not enough subjects for {exercise}, skipping")
        return None

    # Compute per-band normalised CV (across subjects, averaged over gestures)
    band_cv = []
    band_fisher = []
    band_mean_power = []

    for b_idx, band_name in enumerate(BAND_NAMES):
        gesture_cvs = []
        gesture_means = []

        # Collect per-gesture: stack subject observations -> (n_obs, C)
        bp_by_gesture: Dict[int, np.ndarray] = {}
        for gid, bp_list in band_power_per_subj[b_idx].items():
            arr = np.array(bp_list)  # (n_obs, C)
            bp_by_gesture[gid] = arr

            # CV across observations (subjects * reps) per channel, then mean
            mean_ch = arr.mean(axis=0)
            std_ch = arr.std(axis=0)
            cv_ch = std_ch / (mean_ch + 1e-10)
            gesture_cvs.append(cv_ch.mean())
            gesture_means.append(mean_ch.mean())

        cv_mean = float(np.mean(gesture_cvs)) if gesture_cvs else 0.0
        power_mean = float(np.mean(gesture_means)) if gesture_means else 0.0
        fisher = compute_fisher_ratio(bp_by_gesture)

        band_cv.append(cv_mean)
        band_fisher.append(fisher)
        band_mean_power.append(power_mean)

        logger.info(
            f"  {band_name:>14s}: CV={cv_mean:.4f}  Fisher={fisher:.4f}  "
            f"MeanPower={power_mean:.6f}"
        )

    return {
        "exercise": exercise,
        "n_subjects_loaded": len(loaded_subjects),
        "subjects_loaded": loaded_subjects,
        "band_names": BAND_NAMES,
        "band_cv": band_cv,
        "band_fisher": band_fisher,
        "band_mean_power": band_mean_power,
    }


# =============================================================================
#  Cross-Exercise Comparison
# =============================================================================

def compare_cv_gradients(
    exercise_results: Dict[str, Dict],
    logger: logging.Logger,
) -> Dict:
    """
    Compare CV gradients across exercises using Spearman correlation.

    Returns
    -------
    dict with pairwise Spearman rho and p-values.
    """
    logger.info(f"\n{'='*60}")
    logger.info("  Cross-Exercise CV Gradient Comparison (Spearman)")
    logger.info(f"{'='*60}")

    exercises = sorted(exercise_results.keys())
    comparisons = {}

    for i in range(len(exercises)):
        for j in range(i + 1, len(exercises)):
            ex1, ex2 = exercises[i], exercises[j]
            cv1 = np.array(exercise_results[ex1]["band_cv"])
            cv2 = np.array(exercise_results[ex2]["band_cv"])

            rho, p_value = spearmanr(cv1, cv2)

            key = f"{ex1}_vs_{ex2}"
            comparisons[key] = {
                "rho": float(rho),
                "p_value": float(p_value),
                "cv_gradient_1": exercise_results[ex1]["band_cv"],
                "cv_gradient_2": exercise_results[ex2]["band_cv"],
                "significant": bool(p_value < 0.05),
            }

            logger.info(
                f"  {key}: rho={rho:.4f}  p={p_value:.4f}  "
                f"{'*' if p_value < 0.05 else 'ns'}"
            )

    # Also compare Fisher gradients
    fisher_comparisons = {}
    for i in range(len(exercises)):
        for j in range(i + 1, len(exercises)):
            ex1, ex2 = exercises[i], exercises[j]
            f1 = np.array(exercise_results[ex1]["band_fisher"])
            f2 = np.array(exercise_results[ex2]["band_fisher"])

            rho, p_value = spearmanr(f1, f2)
            key = f"{ex1}_vs_{ex2}_fisher"
            fisher_comparisons[key] = {
                "rho": float(rho),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
            logger.info(
                f"  {key} (Fisher): rho={rho:.4f}  p={p_value:.4f}  "
                f"{'*' if p_value < 0.05 else 'ns'}"
            )

    return {
        "cv_correlations": comparisons,
        "fisher_correlations": fisher_comparisons,
    }


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    _parser = argparse.ArgumentParser(
        description="H7s: Cross-Dataset CV Gradient Analysis",
    )
    _parser.add_argument("--full", action="store_true",
                         help="Use full 20-subject set")
    _parser.add_argument("--ci", type=int, default=0,
                         help="Use CI subject set (default)")
    _parser.add_argument("--subjects", type=str, default="",
                         help="Comma-separated subject IDs")
    _parser.add_argument("--exercises", nargs="+", default=["E1", "E2", "E3"],
                         help="Exercises to compare (default: E1 E2 E3)")
    _parser.add_argument("--base_dir", type=str, default="data",
                         help="Base data directory")
    _args, _ = _parser.parse_known_args()

    # Subject selection (default to CI)
    if _args.subjects:
        subjects = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        subjects = _FULL_SUBJECTS
    elif _args.ci:
        subjects = _CI_SUBJECTS
    else:
        subjects = _CI_SUBJECTS

    exercises = _args.exercises
    base_dir = Path(_args.base_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments_output") / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("H7s: Cross-Dataset CV Gradient Analysis")
    logger.info(f"Subjects ({len(subjects)}): {subjects}")
    logger.info(f"Exercises: {exercises}")

    # -- Per-exercise analysis ------------------------------------------------
    exercise_results: Dict[str, Dict] = {}
    for ex in exercises:
        result = analyse_exercise(base_dir, ex, subjects, logger)
        if result is not None:
            exercise_results[ex] = result

    if len(exercise_results) < 2:
        logger.warning(
            "Less than 2 exercises had enough data for comparison. "
            "Saving available results only."
        )

    # -- Cross-exercise comparison -------------------------------------------
    cross_comparison = {}
    if len(exercise_results) >= 2:
        cross_comparison = compare_cv_gradients(exercise_results, logger)

    # -- Summary table -------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info(f"{'Band':<16} ", end="")
    for ex in sorted(exercise_results.keys()):
        logger.info(f"{'CV('+ex+')':>12} {'Fisher('+ex+')':>12} ", end="")
    logger.info("")
    logger.info("-" * 70)
    for b_idx, band_name in enumerate(BAND_NAMES):
        line = f"{band_name:<16} "
        for ex in sorted(exercise_results.keys()):
            cv_val = exercise_results[ex]["band_cv"][b_idx]
            f_val = exercise_results[ex]["band_fisher"][b_idx]
            line += f"{cv_val:>12.4f} {f_val:>12.4f} "
        logger.info(line)

    # -- Save results --------------------------------------------------------
    combined = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "exercises": exercises,
        "per_exercise": exercise_results,
        "cross_exercise_comparison": cross_comparison,
        "frequency_bands": {name: list(rng) for name, rng in FREQ_BANDS.items()},
    }

    output_path = output_dir / "cv_gradient_analysis.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # -- Hypothesis tracking -------------------------------------------------
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        if cross_comparison and "cv_correlations" in cross_comparison:
            rhos = [
                v["rho"] for v in cross_comparison["cv_correlations"].values()
            ]
            all_significant = all(
                v["significant"]
                for v in cross_comparison["cv_correlations"].values()
            )
            mean_rho = float(np.mean(rhos)) if rhos else 0.0

            if all_significant and mean_rho > 0.7:
                mark_hypothesis_verified("H7s", {
                    "mean_spearman_rho": mean_rho,
                    "all_significant": True,
                    "n_comparisons": len(rhos),
                }, experiment_name=EXPERIMENT_NAME)
            else:
                mark_hypothesis_failed(
                    "H7s",
                    f"CV gradient not consistent: mean_rho={mean_rho:.4f}, "
                    f"all_significant={all_significant}",
                )
        else:
            mark_hypothesis_failed(
                "H7s",
                "Not enough exercises loaded for cross-exercise comparison",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
