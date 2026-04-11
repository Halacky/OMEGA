"""
Experiment 83: Subject-Cluster-Specific Models for LOSO EMG Classification.

Source: de Souza et al. (Sensors 2019) — clustering by physiological profile.

Idea:
  Instead of training a single model on ALL train subjects, we:
    1. Compute an unsupervised physiological profile for every subject
       (mean amplitude, spectral centroid, band power ratios, etc.)
       using only raw EMG signals — NO gesture labels involved.
    2. Cluster the TRAIN subjects by these profiles (KMeans).
    3. Train a separate ML model per cluster (SVM / RF on powerful features).
    4. For the test subject, compute the same unsupervised profile,
       find the nearest cluster centroid, and evaluate with
       that cluster's model.

LOSO correctness:
  - Test subject data is NEVER used for clustering or model training.
  - Clustering happens ONLY on train subjects.
  - Test-subject cluster assignment uses raw signal statistics
    (no labels) — analogous to computing features before inference.
  - No model adaptation to the test subject whatsoever.

Comparison baseline:
  - For each fold we also train a standard single model on ALL train
    subjects, so we can directly measure the effect of clustering.
"""

import os
import sys
import gc
import json
import copy
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import torch

# ── repo root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import FeatureMLTrainer
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ── subject lists ──────────────────────────────────────────────────────
from exp_X_template_loso import (
    parse_subjects_args,
    make_json_serializable,
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
)

# ======================================================================
#  UNSUPERVISED SUBJECT PROFILING  (labels are never used)
# ======================================================================


def _spectral_features_per_channel(
    windows: np.ndarray, fs: int
) -> np.ndarray:
    """Compute spectral features for each channel.

    Args:
        windows: (N, T, C)
        fs: sampling rate in Hz

    Returns:
        feats: (C, 5) — [mean_freq, median_freq, low_ratio, mid_ratio, high_ratio]
    """
    N, T, C = windows.shape
    # Average PSD over all windows (Welch, per channel)
    feats = np.zeros((C, 5), dtype=np.float64)

    nperseg = min(256, T)
    for ch in range(C):
        # shape of x_ch: (N, T)
        x_ch = windows[:, :, ch]
        # Welch PSD — average across windows
        freqs, psd = scipy_signal.welch(
            x_ch, fs=fs, nperseg=nperseg, axis=-1
        )  # psd: (N, n_freqs)
        psd_mean = psd.mean(axis=0)  # average over windows

        total_power = np.sum(psd_mean) + 1e-12

        # Mean frequency (spectral centroid)
        feats[ch, 0] = np.sum(freqs * psd_mean) / total_power

        # Median frequency
        cumulative = np.cumsum(psd_mean)
        half_power = total_power / 2.0
        idx_median = np.searchsorted(cumulative, half_power)
        idx_median = min(idx_median, len(freqs) - 1)
        feats[ch, 1] = freqs[idx_median]

        # Band power ratios
        low_mask = (freqs >= 20) & (freqs < 100)
        mid_mask = (freqs >= 100) & (freqs < 250)
        high_mask = (freqs >= 250) & (freqs <= min(500, fs / 2))

        feats[ch, 2] = np.sum(psd_mean[low_mask]) / total_power
        feats[ch, 3] = np.sum(psd_mean[mid_mask]) / total_power
        feats[ch, 4] = np.sum(psd_mean[high_mask]) / total_power

    return feats


def compute_subject_profile(
    windows: np.ndarray, sampling_rate: int = 2000
) -> np.ndarray:
    """Compute an unsupervised physiological profile from raw EMG windows.

    Only raw signal statistics are used — gesture labels are never involved.

    Features computed (per channel, then mean & std across channels):
        - MAV  (mean absolute value)
        - RMS  (root-mean-square)
        - WL   (waveform length, normalised by T)
        - Mean frequency
        - Median frequency
        - Band power ratios (low / mid / high)

    Args:
        windows: (N, T, C) — all windows from one subject, labels ignored
        sampling_rate: Hz

    Returns:
        profile: (16,) 1-D vector
            8 features × 2 (mean across channels, std across channels)
    """
    N, T, C = windows.shape

    # ── time-domain features per channel ──
    abs_vals = np.abs(windows)                              # (N, T, C)
    mav_per_win = abs_vals.mean(axis=1)                     # (N, C)
    mav_mean = mav_per_win.mean(axis=0)                     # (C,)

    rms_per_win = np.sqrt((windows ** 2).mean(axis=1))      # (N, C)
    rms_mean = rms_per_win.mean(axis=0)                     # (C,)

    diffs = np.abs(np.diff(windows, axis=1))                # (N, T-1, C)
    wl_per_win = diffs.sum(axis=1) / T                      # (N, C)
    wl_mean = wl_per_win.mean(axis=0)                       # (C,)

    # ── spectral features per channel ──
    spec = _spectral_features_per_channel(windows, sampling_rate)
    # spec: (C, 5) — mean_freq, median_freq, low_ratio, mid_ratio, high_ratio

    # Stack per-channel features: (C, 8)
    per_channel = np.column_stack([
        mav_mean,           # (C,)
        rms_mean,           # (C,)
        wl_mean,            # (C,)
        spec[:, 0],         # mean_freq        (C,)
        spec[:, 1],         # median_freq      (C,)
        spec[:, 2],         # low_ratio        (C,)
        spec[:, 3],         # mid_ratio        (C,)
        spec[:, 4],         # high_ratio       (C,)
    ])  # (C, 8)

    # Aggregate across channels: mean & std → (16,)
    profile = np.concatenate([
        per_channel.mean(axis=0),   # (8,)
        per_channel.std(axis=0),    # (8,)
    ])

    return profile.astype(np.float64)


# ======================================================================
#  CLUSTERING UTILITIES
# ======================================================================


def cluster_train_subjects(
    train_profiles: Dict[str, np.ndarray],
    n_clusters: int,
    seed: int = 42,
) -> Tuple[Dict[str, int], np.ndarray, StandardScaler]:
    """Cluster train subjects by their physiological profiles.

    Args:
        train_profiles: {subject_id: profile_vector}
        n_clusters:     K for KMeans
        seed:           random seed

    Returns:
        assignments: {subject_id: cluster_id}
        centroids:   (K, F) in scaled space
        scaler:      fitted StandardScaler
    """
    sids = sorted(train_profiles.keys())
    X = np.array([train_profiles[sid] for sid in sids])  # (n_train, F)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)

    assignments = {sid: int(labels[i]) for i, sid in enumerate(sids)}
    centroids = km.cluster_centers_  # (K, F)

    return assignments, centroids, scaler


def select_best_n_clusters(
    train_profiles: Dict[str, np.ndarray],
    candidates: List[int],
    seed: int = 42,
) -> int:
    """Pick the number of clusters with the best silhouette score.

    Falls back to the smallest candidate if silhouette cannot be computed
    (e.g. n_samples <= 3).
    """
    sids = sorted(train_profiles.keys())
    n_subjects = len(sids)

    # Filter candidates: need at least 2 per cluster and n_clusters < n_subjects
    valid = [k for k in candidates if 2 <= k < n_subjects]
    if not valid:
        return min(candidates)

    X = np.array([train_profiles[sid] for sid in sids])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = valid[0]
    best_score = -2.0

    for k in valid:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        # Silhouette requires at least 2 clusters with >0 members each
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            continue
        try:
            sc = silhouette_score(X_scaled, labels)
        except ValueError:
            continue
        if sc > best_score:
            best_score = sc
            best_k = k

    return best_k


def assign_to_cluster(
    profile: np.ndarray,
    centroids: np.ndarray,
    scaler: StandardScaler,
) -> int:
    """Assign a subject to the nearest cluster centroid (unsupervised).

    Uses only signal-level profile — no labels involved.
    """
    profile_scaled = scaler.transform(profile.reshape(1, -1))  # (1, F)
    distances = np.linalg.norm(centroids - profile_scaled, axis=1)  # (K,)
    return int(np.argmin(distances))


# ======================================================================
#  HELPER: concatenate all windows from grouped_windows
# ======================================================================


def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert {gesture_id: [rep_arrays]} → flat (windows, labels).

    Returns:
        windows: (N, T, C)
        labels:  (N,) gesture IDs (NOT class indices)
    """
    all_w, all_l = [], []
    for gid in sorted(grouped_windows.keys()):
        reps = grouped_windows[gid]
        for rep in reps:
            if rep.ndim == 3 and len(rep) > 0:
                all_w.append(rep)
                all_l.append(np.full(len(rep), gid, dtype=np.int64))
    if not all_w:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_w, axis=0), np.concatenate(all_l, axis=0)


def concat_all_windows_no_labels(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> np.ndarray:
    """Concatenate all windows ignoring gesture IDs.

    Returns:
        windows: (N, T, C)
    """
    parts = []
    for gid in sorted(grouped_windows.keys()):
        for rep in grouped_windows[gid]:
            if rep.ndim == 3 and len(rep) > 0:
                parts.append(rep)
    if not parts:
        return np.empty((0,))
    return np.concatenate(parts, axis=0)


# ======================================================================
#  SINGLE LOSO FOLD — clustered approach
# ======================================================================


def run_cluster_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    ml_model_type: str,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    max_n_clusters: int = 3,
    logger=None,
) -> Dict:
    """Run a single LOSO fold with subject-cluster-specific models.

    For comparison, also trains a baseline model on ALL train subjects.

    Returns dict with keys:
        test_subject, ml_model_type,
        baseline_accuracy, baseline_f1,
        cluster_accuracy, cluster_f1,
        n_clusters, test_cluster_id, cluster_subjects,
        profiles, assignments
    """
    fold_dir = output_dir
    fold_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = setup_logging(fold_dir)

    seed_everything(train_cfg.seed, verbose=False)

    all_subject_ids = train_subjects + [test_subject]

    # ------------------------------------------------------------------
    # 1. Load ALL subjects
    # ------------------------------------------------------------------
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

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=10
    )
    logger.info(
        f"[Cluster-LOSO] Common gestures ({len(common_gestures)}): "
        f"{common_gestures}"
    )

    # ------------------------------------------------------------------
    # 2. Compute unsupervised profiles (NO labels used)
    # ------------------------------------------------------------------
    profiles: Dict[str, np.ndarray] = {}
    for sid in all_subject_ids:
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(
            grouped_windows, common_gestures
        )
        all_windows = concat_all_windows_no_labels(filtered)
        if all_windows.ndim != 3 or len(all_windows) == 0:
            logger.warning(
                f"[Cluster-LOSO] Subject {sid}: no valid windows, skipping profile"
            )
            continue
        profiles[sid] = compute_subject_profile(
            all_windows, sampling_rate=proc_cfg.sampling_rate
        )
        logger.info(
            f"[Cluster-LOSO] Profile {sid}: "
            f"MAV={profiles[sid][0]:.4f}, RMS={profiles[sid][1]:.4f}, "
            f"MeanFreq={profiles[sid][3]:.1f}Hz"
        )

    # ------------------------------------------------------------------
    # 3. Cluster TRAIN subjects only (test excluded!)
    # ------------------------------------------------------------------
    train_profiles = {
        sid: profiles[sid] for sid in train_subjects if sid in profiles
    }

    n_train = len(train_profiles)
    if n_train < 4:
        # Not enough subjects for meaningful clustering — use baseline only
        logger.warning(
            f"[Cluster-LOSO] Only {n_train} train subjects — "
            "skipping clustering, using baseline only"
        )
        n_clusters = 1
        assignments = {sid: 0 for sid in train_profiles}
        centroids = None
        scaler = None
        test_cluster = 0
    else:
        candidates = [k for k in range(2, max_n_clusters + 1)]
        n_clusters = select_best_n_clusters(
            train_profiles, candidates, seed=train_cfg.seed
        )
        assignments, centroids, scaler = cluster_train_subjects(
            train_profiles, n_clusters, seed=train_cfg.seed
        )
        # Assign test subject (unsupervised — uses only signal profile)
        test_cluster = assign_to_cluster(
            profiles[test_subject], centroids, scaler
        )

    logger.info(f"[Cluster-LOSO] n_clusters={n_clusters}")
    for sid, cid in sorted(assignments.items()):
        logger.info(f"  Train subject {sid} → cluster {cid}")
    logger.info(
        f"  Test subject {test_subject} → cluster {test_cluster} (unsupervised)"
    )

    # Subjects in the test subject's cluster
    cluster_train_sids = [
        sid for sid, c in assignments.items() if c == test_cluster
    ]
    logger.info(
        f"[Cluster-LOSO] Cluster model will train on: {cluster_train_sids}"
    )

    # ------------------------------------------------------------------
    # 4. BASELINE: train on ALL train subjects
    # ------------------------------------------------------------------
    logger.info("[Cluster-LOSO] === Training BASELINE (all train subjects) ===")
    baseline_dir = fold_dir / "baseline"
    baseline_result = _train_and_evaluate(
        base_dir=base_dir,
        output_dir=baseline_dir,
        train_sids=train_subjects,
        test_subject=test_subject,
        exercises=exercises,
        ml_model_type=ml_model_type,
        proc_cfg=proc_cfg,
        split_cfg=split_cfg,
        train_cfg=train_cfg,
        logger=logger,
        tag="baseline",
    )

    # ------------------------------------------------------------------
    # 5. CLUSTER MODEL: train on cluster's subjects only
    # ------------------------------------------------------------------
    logger.info(
        f"[Cluster-LOSO] === Training CLUSTER model "
        f"(cluster {test_cluster}: {cluster_train_sids}) ==="
    )
    cluster_dir = fold_dir / f"cluster_{test_cluster}"
    cluster_result = _train_and_evaluate(
        base_dir=base_dir,
        output_dir=cluster_dir,
        train_sids=cluster_train_sids,
        test_subject=test_subject,
        exercises=exercises,
        ml_model_type=ml_model_type,
        proc_cfg=proc_cfg,
        split_cfg=split_cfg,
        train_cfg=train_cfg,
        logger=logger,
        tag=f"cluster_{test_cluster}",
    )

    # ------------------------------------------------------------------
    # 6. Assemble fold results
    # ------------------------------------------------------------------
    baseline_acc = baseline_result.get("accuracy")
    baseline_f1 = baseline_result.get("f1_macro")
    cluster_acc = cluster_result.get("accuracy")
    cluster_f1 = cluster_result.get("f1_macro")

    improvement_acc = None
    improvement_f1 = None
    if baseline_acc is not None and cluster_acc is not None:
        improvement_acc = cluster_acc - baseline_acc
    if baseline_f1 is not None and cluster_f1 is not None:
        improvement_f1 = cluster_f1 - baseline_f1

    # Safe formatting
    def fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"

    logger.info(
        f"[Cluster-LOSO] Fold result for test={test_subject}:\n"
        f"  Baseline — Acc={fmt(baseline_acc)}, F1={fmt(baseline_f1)}\n"
        f"  Cluster  — Acc={fmt(cluster_acc)}, F1={fmt(cluster_f1)}\n"
        f"  Delta    — Acc={fmt(improvement_acc)}, F1={fmt(improvement_f1)}"
    )

    # Save profiles for analysis
    profiles_serializable = {
        sid: prof.tolist() for sid, prof in profiles.items()
    }

    fold_result = {
        "test_subject": test_subject,
        "ml_model_type": ml_model_type,
        # Baseline
        "baseline_accuracy": baseline_acc,
        "baseline_f1_macro": baseline_f1,
        # Cluster
        "cluster_accuracy": cluster_acc,
        "cluster_f1_macro": cluster_f1,
        # Improvement
        "improvement_accuracy": improvement_acc,
        "improvement_f1_macro": improvement_f1,
        # Cluster info
        "n_clusters": n_clusters,
        "test_cluster_id": test_cluster,
        "cluster_train_subjects": cluster_train_sids,
        "all_cluster_assignments": assignments,
        # Profiles (for post-hoc analysis)
        "profiles": profiles_serializable,
    }

    with open(fold_dir / "fold_result.json", "w") as f:
        json.dump(make_json_serializable(fold_result), f, indent=4)

    # Cleanup
    del subjects_data, multi_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return fold_result


# ======================================================================
#  Train + Evaluate helper (uses CrossSubjectExperiment)
# ======================================================================


def _train_and_evaluate(
    base_dir: Path,
    output_dir: Path,
    train_sids: List[str],
    test_subject: str,
    exercises: List[str],
    ml_model_type: str,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    logger,
    tag: str,
) -> Dict:
    """Train a FeatureMLTrainer on `train_sids` and evaluate on `test_subject`.

    Uses CrossSubjectExperiment for correctness and consistency with
    the rest of the codebase.

    Returns:
        {"accuracy": float|None, "f1_macro": float|None, "error": str|None}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tcfg = copy.deepcopy(train_cfg)
    tcfg.ml_model_type = ml_model_type
    tcfg.pipeline_type = "ml_emg_td"

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_sids,
        test_subject=test_subject,
        exercises=exercises,
        base_dir=base_dir,
        pool_train_subjects=True,
        use_separate_val_subject=False,
        val_subject=None,
        val_ratio=0.15,
        seed=tcfg.seed,
        max_gestures=10,
    )

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=True,
    )

    viz = Visualizer(output_dir, logger)

    trainer = FeatureMLTrainer(
        train_cfg=tcfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=viz,
    )

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=viz,
        logger=logger,
    )

    try:
        results = experiment.run()
        test_metrics = results.get("cross_subject_test", {})
        acc = float(test_metrics.get("accuracy", 0.0))
        f1 = float(test_metrics.get("f1_macro", 0.0))

        # Save results (exclude subjects_data)
        results_to_save = {
            k: v for k, v in results.items() if k != "subjects_data"
        }
        with open(output_dir / "cross_subject_results.json", "w") as f:
            json.dump(
                make_json_serializable(results_to_save),
                f, indent=4, ensure_ascii=False,
            )

        return {"accuracy": acc, "f1_macro": f1, "error": None}

    except Exception as e:
        logger.error(f"[{tag}] Error: {e}")
        traceback.print_exc()
        return {"accuracy": None, "f1_macro": None, "error": str(e)}

    finally:
        del experiment, trainer, multi_loader, viz
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ======================================================================
#  MAIN
# ======================================================================


def main():
    EXPERIMENT_NAME = "exp_83_subject_cluster_models_loso"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")

    ALL_SUBJECTS = parse_subjects_args()
    EXERCISES = ["E1"]

    # ML model types to evaluate
    MODEL_TYPES = ["svm_rbf", "svm_linear"]
    MAX_N_CLUSTERS = 3  # max number of clusters to consider

    # ── configs ────────────────────────────────────────────────────────
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

    train_cfg = TrainingConfig(
        batch_size=4096,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.0,
        early_stopping_patience=1,
        use_class_weights=False,
        seed=42,
        num_workers=0,
        device="cpu",
        use_handcrafted_features=False,  # FeatureMLTrainer sets internally
        handcrafted_feature_set="powerful",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_rbf",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"Experiment: {EXPERIMENT_NAME}")
    global_logger.info(f"Subjects: {ALL_SUBJECTS}")
    global_logger.info(f"Models: {MODEL_TYPES}")
    global_logger.info(f"Max clusters: {MAX_N_CLUSTERS}")
    global_logger.info("=" * 80)

    all_fold_results = []

    for ml_model_type in MODEL_TYPES:
        print(f"\n{'='*60}")
        print(f"ML MODEL: {ml_model_type}")
        print(f"{'='*60}")

        for test_subject in ALL_SUBJECTS:
            train_subjects = [
                s for s in ALL_SUBJECTS if s != test_subject
            ]
            print(
                f"\n[LOSO] Test: {test_subject} | "
                f"Train: {train_subjects} | Model: {ml_model_type}"
            )

            fold_output_dir = OUTPUT_DIR / ml_model_type / f"test_{test_subject}"

            try:
                fold_result = run_cluster_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_output_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    ml_model_type=ml_model_type,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                    max_n_clusters=MAX_N_CLUSTERS,
                    logger=global_logger,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                fold_result = {
                    "test_subject": test_subject,
                    "ml_model_type": ml_model_type,
                    "baseline_accuracy": None,
                    "baseline_f1_macro": None,
                    "cluster_accuracy": None,
                    "cluster_f1_macro": None,
                    "error": str(e),
                }

            all_fold_results.append(fold_result)

            # Print fold summary
            def fmt(v):
                return f"{v:.4f}" if v is not None else "N/A"

            print(
                f"  Baseline: Acc={fmt(fold_result.get('baseline_accuracy'))}, "
                f"F1={fmt(fold_result.get('baseline_f1_macro'))}"
            )
            print(
                f"  Cluster:  Acc={fmt(fold_result.get('cluster_accuracy'))}, "
                f"F1={fmt(fold_result.get('cluster_f1_macro'))}"
            )
            delta_acc = fold_result.get("improvement_accuracy")
            print(f"  Delta Acc: {fmt(delta_acc)}")

    # ==================================================================
    # Aggregate results
    # ==================================================================
    aggregate = {}
    for ml_model_type in MODEL_TYPES:
        model_folds = [
            r for r in all_fold_results
            if r.get("ml_model_type") == ml_model_type
        ]

        bl_accs = [
            r["baseline_accuracy"] for r in model_folds
            if r.get("baseline_accuracy") is not None
        ]
        bl_f1s = [
            r["baseline_f1_macro"] for r in model_folds
            if r.get("baseline_f1_macro") is not None
        ]
        cl_accs = [
            r["cluster_accuracy"] for r in model_folds
            if r.get("cluster_accuracy") is not None
        ]
        cl_f1s = [
            r["cluster_f1_macro"] for r in model_folds
            if r.get("cluster_f1_macro") is not None
        ]
        imp_accs = [
            r["improvement_accuracy"] for r in model_folds
            if r.get("improvement_accuracy") is not None
        ]
        imp_f1s = [
            r["improvement_f1_macro"] for r in model_folds
            if r.get("improvement_f1_macro") is not None
        ]

        def safe_mean_std(arr):
            if not arr:
                return None, None
            return float(np.mean(arr)), float(np.std(arr))

        bl_acc_mean, bl_acc_std = safe_mean_std(bl_accs)
        bl_f1_mean, bl_f1_std = safe_mean_std(bl_f1s)
        cl_acc_mean, cl_acc_std = safe_mean_std(cl_accs)
        cl_f1_mean, cl_f1_std = safe_mean_std(cl_f1s)
        imp_acc_mean, imp_acc_std = safe_mean_std(imp_accs)
        imp_f1_mean, imp_f1_std = safe_mean_std(imp_f1s)

        aggregate[ml_model_type] = {
            "baseline": {
                "mean_accuracy": bl_acc_mean,
                "std_accuracy": bl_acc_std,
                "mean_f1_macro": bl_f1_mean,
                "std_f1_macro": bl_f1_std,
            },
            "cluster": {
                "mean_accuracy": cl_acc_mean,
                "std_accuracy": cl_acc_std,
                "mean_f1_macro": cl_f1_mean,
                "std_f1_macro": cl_f1_std,
            },
            "improvement": {
                "mean_accuracy": imp_acc_mean,
                "std_accuracy": imp_acc_std,
                "mean_f1_macro": imp_f1_mean,
                "std_f1_macro": imp_f1_std,
            },
            "num_folds": len(model_folds),
        }

        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        print(f"\n{'='*60}")
        print(f"AGGREGATE for {ml_model_type}:")
        print(
            f"  Baseline: Acc={fmt(bl_acc_mean)}±{fmt(bl_acc_std)}, "
            f"F1={fmt(bl_f1_mean)}±{fmt(bl_f1_std)}"
        )
        print(
            f"  Cluster:  Acc={fmt(cl_acc_mean)}±{fmt(cl_acc_std)}, "
            f"F1={fmt(cl_f1_mean)}±{fmt(cl_f1_std)}"
        )
        print(
            f"  Delta:    Acc={fmt(imp_acc_mean)}±{fmt(imp_acc_std)}, "
            f"F1={fmt(imp_f1_mean)}±{fmt(imp_f1_std)}"
        )
        print(f"{'='*60}")

    # ==================================================================
    # Save summary
    # ==================================================================
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "description": (
            "Subject-cluster-specific models: cluster train subjects by "
            "unsupervised EMG physiological profiles, train separate models "
            "per cluster, assign test subject to nearest cluster. "
            "Fully LOSO-correct — no test labels or adaptation used."
        ),
        "approach": "ml_emg_td",
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "max_n_clusters": MAX_N_CLUSTERS,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": make_json_serializable(all_fold_results),
        "experiment_date": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            make_json_serializable(loso_summary),
            f, indent=4, ensure_ascii=False,
        )

    print(f"\n[DONE] {EXPERIMENT_NAME}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
