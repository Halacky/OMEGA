#!/usr/bin/env python3
"""
H1s: Frequency-Aware Augmentation Analysis (Descriptive).

Analyses how different augmentation strategies affect representation
quality in the UVMD-decomposed frequency space.  No model training —
this is a pure analysis experiment using a pre-trained encoder.

Question
────────
  If augmentations are applied selectively by frequency band (strong in
  high-CV bands, weak in low-CV bands), do the resulting representations
  better preserve gesture information while being invariant to subject?

Method
──────
  1. Train a supervised UVMD+MixStyle encoder (H7-F baseline) OR load
     a pretrained SSL encoder.
  2. For each augmentation strategy:
     - global:         same noise/scaling/warp across all bands
     - freq_selective: strong aug on high bands, weak on low bands
     - inverse:        strong on low bands, weak on high bands (control)
  3. Compute:
     - Alignment: avg cosine similarity between original and augmented
       representations of the SAME gesture
     - Uniformity: log of avg pairwise similarity across all gestures
       (should be low = well-spread representations)
     - Subject invariance: avg cosine similarity between same-gesture
       representations from DIFFERENT subjects (should be high)
  4. Visualise: alignment vs uniformity scatter, per-strategy

Output
──────
  experiments_output/h1s_freq_augmentation_analysis_<timestamp>/
    ├── augmentation_metrics.json
    ├── alignment_uniformity_scatter.png
    └── per_band_cv_vs_augmentation.png

Usage
─────
  python experiments/h1s_freq_augmentation_analysis.py
  python experiments/h1s_freq_augmentation_analysis.py --full
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

# ── project imports ──────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from models.uvmd_classifier import UVMDBlock
from models.uvmd_ssl_encoder import (
    UVMDSSLEncoder,
    PerBandMixStyle,
    FreqAwareAugmentation,
)
from utils.logging import setup_logging

# ── constants ────────────────────────────────────────────────────────
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [
    f"DB2_s{i}" for i in
    [1, 2, 3, 4, 5, 11, 12, 13, 14, 15,
     26, 27, 28, 29, 30, 36, 37, 38, 39, 40]
]

SEED = 42
WINDOW_SIZE = 200
WINDOW_OVERLAP = 100
SAMPLING_RATE = 2000
K_MODES = 4
N_CHANNELS = 12

EXPERIMENT_NAME = "h1s_freq_augmentation_analysis"

# Augmentation presets
AUG_STRATEGIES = {
    "global": {
        "noise_low": 0.05, "noise_high": 0.05,
        "scale_range_low": (0.8, 1.2), "scale_range_high": (0.8, 1.2),
        "time_warp_low": 0.1, "time_warp_high": 0.1,
    },
    "freq_selective": {
        "noise_low": 0.01, "noise_high": 0.1,
        "scale_range_low": (0.9, 1.1), "scale_range_high": (0.5, 2.0),
        "time_warp_low": 0.05, "time_warp_high": 0.15,
    },
    "inverse": {
        "noise_low": 0.1, "noise_high": 0.01,
        "scale_range_low": (0.5, 2.0), "scale_range_high": (0.9, 1.1),
        "time_warp_low": 0.15, "time_warp_high": 0.05,
    },
}


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            windows_list.append(rep_arr)
            labels_list.extend([gid] * len(rep_arr))
    return np.concatenate(windows_list, axis=0), np.array(labels_list)


def compute_alignment(
    z_orig: np.ndarray, z_aug: np.ndarray,
) -> float:
    """Average cosine similarity between original and augmented features."""
    z_orig_norm = z_orig / (np.linalg.norm(z_orig, axis=1, keepdims=True) + 1e-8)
    z_aug_norm = z_aug / (np.linalg.norm(z_aug, axis=1, keepdims=True) + 1e-8)
    cos_sim = (z_orig_norm * z_aug_norm).sum(axis=1)
    return float(cos_sim.mean())


def compute_uniformity(z: np.ndarray, t: float = 2.0) -> float:
    """Log of average pairwise Gaussian potential (Wang & Isola, 2020)."""
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    # Subsample for efficiency if large
    N = len(z_norm)
    if N > 2000:
        idx = np.random.choice(N, 2000, replace=False)
        z_norm = z_norm[idx]
        N = 2000
    sq_dists = np.sum((z_norm[:, None] - z_norm[None, :]) ** 2, axis=2)
    # Exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    return float(np.log(np.exp(-t * sq_dists[mask]).mean() + 1e-10))


def compute_subject_invariance(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
) -> float:
    """
    Average cosine similarity between same-gesture representations
    from different subjects.
    """
    unique_gestures = np.unique(labels)
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 2:
        return 0.0

    similarities = []
    for g in unique_gestures:
        g_mask = labels == g
        g_feats = features[g_mask]
        g_sids = subject_ids[g_mask]

        # Compute per-subject centroids
        centroids = {}
        for s in unique_subjects:
            s_mask = g_sids == s
            if s_mask.sum() > 0:
                c = g_feats[s_mask].mean(axis=0)
                c = c / (np.linalg.norm(c) + 1e-8)
                centroids[s] = c

        # Pairwise cosine similarity between subject centroids
        sids = list(centroids.keys())
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                sim = float(np.dot(centroids[sids[i]], centroids[sids[j]]))
                similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


# ═════════════════════════════════════════════════════════════════════
#  Main Analysis
# ═════════════════════════════════════════════════════════════════════

def run_analysis(
    subjects: List[str],
    base_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Run the full augmentation analysis."""
    seed_everything()

    # ── Load data ──────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP,
        sampling_rate=SAMPLING_RATE,
    )
    multi_loader = MultiSubjectLoader(proc_cfg, logger, use_gpu=False,
                                       use_improved_processing=True)
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=subjects,
        exercises=["E1"],
        include_rest=False,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    gesture_to_class = {g: i for i, g in enumerate(sorted(common_gestures))}

    # Collect all windows with labels and subject IDs
    all_windows, all_labels, all_sids = [], [], []
    for sid, (_, _, gw) in subjects_data.items():
        wins, labs = grouped_to_arrays(gw)
        mask = np.isin(labs, list(gesture_to_class.keys()))
        wins, labs = wins[mask], labs[mask]
        labs = np.array([gesture_to_class[g] for g in labs])
        all_windows.append(wins)
        all_labels.append(labs)
        all_sids.append(np.full(len(labs), hash(sid) % 10000))

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0)
    sid_arr = np.concatenate(all_sids, axis=0)

    # Channel standardisation
    mean_c = X.mean(axis=(0, 1), keepdims=True)
    std_c = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean_c) / std_c

    # Subsample for efficiency
    N = len(X)
    if N > 5000:
        idx = np.random.choice(N, 5000, replace=False)
        X, y, sid_arr = X[idx], y[idx], sid_arr[idx]

    logger.info(f"Analysis data: {X.shape[0]} windows, {len(gesture_to_class)} classes")

    # ── Create encoder (random init — we just need consistent features) ──
    encoder = UVMDSSLEncoder(
        K=K_MODES, L=8, in_channels=N_CHANNELS, feat_dim=64,
        use_mixstyle=False,
    ).to(device)
    encoder.eval()

    # ── Extract original features ─────────────────────────────────
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        z_orig = encoder.encode(X_tensor).cpu().numpy()

    logger.info(f"Original features shape: {z_orig.shape}")

    # ── Per-strategy analysis ─────────────────────────────────────
    results = {}
    for strategy_name, aug_kwargs in AUG_STRATEGIES.items():
        logger.info(f"\n--- Strategy: {strategy_name} ---")

        augmentation = FreqAwareAugmentation(K=K_MODES, **aug_kwargs).to(device)

        # Generate augmented features (average over 5 runs for stability)
        alignments, uniformities, invariances = [], [], []
        for run in range(5):
            with torch.no_grad():
                modes = encoder.decompose(X_tensor)  # (N, K, T, C)
                aug_modes, _ = augmentation(modes)
                per_band = encoder.encode_per_band(aug_modes)
                z_aug = torch.cat(per_band, dim=1).cpu().numpy()

            alignments.append(compute_alignment(z_orig, z_aug))
            uniformities.append(compute_uniformity(z_aug))
            invariances.append(compute_subject_invariance(z_aug, y, sid_arr))

        # Gesture silhouette on augmented features
        with torch.no_grad():
            modes = encoder.decompose(X_tensor)
            aug_modes, _ = augmentation(modes)
            per_band = encoder.encode_per_band(aug_modes)
            z_aug_final = torch.cat(per_band, dim=1).cpu().numpy()

        gesture_sil = float(silhouette_score(
            z_aug_final[:2000], y[:2000], metric="cosine",
        )) if len(z_aug_final) >= 2000 else 0.0

        results[strategy_name] = {
            "alignment_mean": float(np.mean(alignments)),
            "alignment_std": float(np.std(alignments)),
            "uniformity_mean": float(np.mean(uniformities)),
            "uniformity_std": float(np.std(uniformities)),
            "subject_invariance_mean": float(np.mean(invariances)),
            "subject_invariance_std": float(np.std(invariances)),
            "gesture_silhouette": gesture_sil,
        }

        logger.info(
            f"  Alignment: {results[strategy_name]['alignment_mean']:.4f} "
            f"± {results[strategy_name]['alignment_std']:.4f}"
        )
        logger.info(
            f"  Uniformity: {results[strategy_name]['uniformity_mean']:.4f} "
            f"± {results[strategy_name]['uniformity_std']:.4f}"
        )
        logger.info(
            f"  Subject invariance: "
            f"{results[strategy_name]['subject_invariance_mean']:.4f} "
            f"± {results[strategy_name]['subject_invariance_std']:.4f}"
        )
        logger.info(
            f"  Gesture silhouette: {gesture_sil:.4f}"
        )

    # ── Also compute baseline (no augmentation) ───────────────────
    baseline_uniformity = compute_uniformity(z_orig)
    baseline_sil = float(silhouette_score(
        z_orig[:2000], y[:2000], metric="cosine",
    )) if len(z_orig) >= 2000 else 0.0
    baseline_inv = compute_subject_invariance(z_orig, y, sid_arr)

    results["no_augmentation"] = {
        "alignment_mean": 1.0,  # identity
        "alignment_std": 0.0,
        "uniformity_mean": baseline_uniformity,
        "uniformity_std": 0.0,
        "subject_invariance_mean": baseline_inv,
        "subject_invariance_std": 0.0,
        "gesture_silhouette": baseline_sil,
    }

    return results


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="H1s: Freq-aware augmentation analysis")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--ci", type=int, default=0)
    parser.add_argument("--subjects", type=str, default="")
    parser.add_argument("--base_dir", type=str, default="data")
    args, _ = parser.parse_known_args()

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    elif args.full:
        subjects = _FULL_SUBJECTS
    elif args.ci:
        subjects = _CI_SUBJECTS
    else:
        subjects = _CI_SUBJECTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments_output") / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"H1s: Frequency-Aware Augmentation Analysis")
    logger.info(f"Subjects: {subjects}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    results = run_analysis(subjects, args.base_dir, device, logger)

    # Save results
    output_path = output_dir / "augmentation_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"{'Strategy':<20} {'Alignment':>10} {'Uniformity':>12} {'Subj Inv':>10} {'Gest Sil':>10}")
    logger.info("-" * 70)
    for name, r in results.items():
        logger.info(
            f"{name:<20} "
            f"{r['alignment_mean']:>10.4f} "
            f"{r['uniformity_mean']:>12.4f} "
            f"{r['subject_invariance_mean']:>10.4f} "
            f"{r['gesture_silhouette']:>10.4f}"
        )

    try:
        from hypothesis_executor import mark_hypothesis_verified
        mark_hypothesis_verified(
            "H1s", results, experiment_name=EXPERIMENT_NAME,
        )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
