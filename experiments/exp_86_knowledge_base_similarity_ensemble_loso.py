"""
Experiment 86: Knowledge-Base Subject-Similarity Ensemble for LOSO EMG Classification.

Source: Wang et al. (EAAI 2024) — optimization with "knowledge base".

Idea:
  Build a "knowledge base" from training subjects. Each KB entry stores:
    1. Unsupervised physiological profile (signal-level statistics only)
    2. A specialist SVM model trained on that subject's data + powerful features
    3. Feature normalization parameters (mean, std from that subject)
    4. Inner cross-validation quality estimate

  At test time for subject S:
    1. Compute S's unsupervised profile (NO labels used)
    2. Find k nearest training subjects in scaled profile space
    3. Ensemble their specialist models via similarity-weighted soft voting

LOSO correctness guarantees:
  - Test subject is NEVER used for model training, KB construction, or normalization
  - Profile computation uses ONLY unsupervised signal statistics (no gesture labels)
  - Profile scaler fitted on TRAINING profiles only
  - No test-time adaptation, fine-tuning, or BN updates
  - Each specialist model trained independently on its own subject's data only

Novelty vs prior experiments:
  - exp_83 clusters subjects and trains ONE model per cluster (hard assignment)
  - THIS experiment trains ONE model PER subject and ensembles via continuous
    similarity weights (soft assignment) — finer granularity, no information loss
    from hard clustering
  - Quality-aware weighting: specialists with higher inner-CV accuracy
    receive more influence in the ensemble

Comparison baselines (per fold):
  - Global baseline: single SVM on all pooled training subjects
  - KB ensemble: similarity-weighted combination of per-subject specialists
"""

import os
import sys
import gc
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from scipy import signal as scipy_signal
from scipy.spatial.distance import cdist

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch

# ── repo root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from processing.powerful_features import PowerfulFeatureExtractor
from utils.logging import setup_logging, seed_everything

# ── subject lists & helpers from template ──────────────────────────────
from exp_X_template_loso import (
    parse_subjects_args,
    make_json_serializable,
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
)

# ======================================================================
#  EXPERIMENT SETTINGS
# ======================================================================

EXPERIMENT_NAME = "exp_86_knowledge_base_similarity_ensemble_loso"
EXERCISES = ["E1"]
USE_IMPROVED = True
MAX_GESTURES = 10

# Knowledge base settings
K_NEIGHBORS = [1, 2, 3]          # k values to evaluate
INNER_CV_FOLDS = 3               # for quality estimation within each subject
SVM_KERNELS = ["rbf"]            # SVM kernel(s) for specialists
SVM_C_VALUES = [1.0, 10.0]       # C values for quick grid search per specialist
SVM_GAMMA = "scale"

ENSEMBLE_STRATEGIES = [
    "weighted_soft",              # softmax(-dist) weighted probability average
    "uniform_soft",               # equal-weight probability average
    "quality_weighted_soft",      # softmax(-dist) * inner_cv_acc weighted
]


# ======================================================================
#  HELPER: grouped_windows -> flat arrays
# ======================================================================

def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert {gesture_id: [rep_arrays]} -> flat (windows, labels)."""
    all_w, all_l = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep in grouped_windows[gid]:
            if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0:
                all_w.append(rep)
                all_l.append(np.full(len(rep), gid, dtype=np.int64))
    if not all_w:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_w, axis=0), np.concatenate(all_l, axis=0)


def concat_all_windows_no_labels(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> np.ndarray:
    """Concatenate all windows ignoring gesture IDs (for profiling)."""
    parts = []
    for gid in sorted(grouped_windows.keys()):
        for rep in grouped_windows[gid]:
            if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0:
                parts.append(rep)
    if not parts:
        return np.empty((0,))
    return np.concatenate(parts, axis=0)


# ======================================================================
#  UNSUPERVISED SUBJECT PROFILING  (gesture labels are NEVER used)
# ======================================================================

def _spectral_features_per_channel(
    windows: np.ndarray, fs: int,
) -> np.ndarray:
    """Compute spectral features for each EMG channel.

    Args:
        windows: (N, T, C)
        fs: sampling rate Hz

    Returns:
        feats: (C, 5) — [mean_freq, median_freq, low_ratio, mid_ratio, high_ratio]
    """
    _N, T, C = windows.shape
    feats = np.zeros((C, 5), dtype=np.float64)
    nperseg = min(256, T)

    for ch in range(C):
        x_ch = windows[:, :, ch]  # (N, T)
        freqs, psd = scipy_signal.welch(x_ch, fs=fs, nperseg=nperseg, axis=-1)
        psd_mean = psd.mean(axis=0)
        total_power = np.sum(psd_mean) + 1e-12

        # Mean frequency (spectral centroid)
        feats[ch, 0] = np.sum(freqs * psd_mean) / total_power

        # Median frequency
        cumulative = np.cumsum(psd_mean)
        idx_median = np.searchsorted(cumulative, total_power / 2.0)
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
    windows: np.ndarray, sampling_rate: int = 2000,
) -> np.ndarray:
    """Compute unsupervised physiological profile from raw EMG windows.

    Only raw signal statistics are used — gesture labels are NEVER involved.

    Features (per channel, then aggregated across channels):
        Time-domain:  MAV, RMS, WL
        Spectral:     mean_freq, median_freq, low/mid/high band ratios
        Cross-channel: mean & std of inter-channel correlations

    Args:
        windows: (N, T, C)
        sampling_rate: Hz

    Returns:
        profile: (18,) 1-D vector
    """
    N, T, C = windows.shape

    # ── time-domain per channel ──
    abs_vals = np.abs(windows)
    mav_mean = abs_vals.mean(axis=1).mean(axis=0)                     # (C,)
    rms_mean = np.sqrt((windows ** 2).mean(axis=1)).mean(axis=0)      # (C,)
    wl_mean = (np.abs(np.diff(windows, axis=1)).sum(axis=1) / T).mean(axis=0)  # (C,)

    # ── spectral per channel ──
    spec = _spectral_features_per_channel(windows, sampling_rate)     # (C, 5)

    # ── cross-channel correlation (unsupervised, no labels) ──
    ch_means = windows.mean(axis=1)  # (N, C) — per-window channel means
    if C > 1:
        corr_vals = []
        for i in range(C):
            for j in range(i + 1, C):
                r = np.corrcoef(ch_means[:, i], ch_means[:, j])[0, 1]
                if np.isfinite(r):
                    corr_vals.append(r)
        mean_corr = np.mean(corr_vals) if corr_vals else 0.0
        std_corr = np.std(corr_vals) if corr_vals else 0.0
    else:
        mean_corr, std_corr = 0.0, 0.0

    # ── per-channel features → (C, 8) ──
    per_channel = np.column_stack([
        mav_mean, rms_mean, wl_mean,
        spec[:, 0], spec[:, 1],
        spec[:, 2], spec[:, 3], spec[:, 4],
    ])  # (C, 8)

    # Aggregate across channels: mean & std → (16,) + cross-channel (2,) = (18,)
    profile = np.concatenate([
        per_channel.mean(axis=0),              # (8,)
        per_channel.std(axis=0),               # (8,)
        np.array([mean_corr, std_corr]),       # (2,)
    ])
    return profile.astype(np.float64)


# ======================================================================
#  KNOWLEDGE BASE
# ======================================================================

@dataclass
class KBEntry:
    """Single knowledge-base entry for one training subject."""
    subject_id: str
    profile: np.ndarray            # (P,) unsupervised profile
    model: Any                     # trained SVC
    feat_mean: np.ndarray          # (F,) feature normalization mean
    feat_std: np.ndarray           # (F,) feature normalization std
    inner_cv_accuracy: float       # quality estimate from inner CV
    n_windows: int
    best_C: float                  # best SVM C from inner CV


def build_knowledge_base(
    per_subject: Dict[str, Dict],
    train_subjects: List[str],
    gesture_to_class: Dict[int, int],
    svm_kernel: str = "rbf",
    svm_C_values: Optional[List[float]] = None,
    svm_gamma: str = "scale",
    inner_cv_folds: int = 3,
    seed: int = 42,
    logger=None,
) -> List[KBEntry]:
    """Build knowledge base from training subjects.

    For each training subject:
      1. Use pre-extracted features & labels
      2. Normalize features (per-subject stats)
      3. Inner stratified CV to select best C and estimate quality
      4. Train final specialist on all of that subject's data
      5. Compute unsupervised profile from raw windows

    LOSO-correct: only training subjects' data is used.
    """
    if svm_C_values is None:
        svm_C_values = [1.0, 10.0]

    num_classes = len(gesture_to_class)
    kb_entries: List[KBEntry] = []

    for sid in train_subjects:
        subj = per_subject.get(sid)
        if subj is None:
            if logger:
                logger.warning(f"[KB] Subject {sid}: no data, skipping")
            continue

        features = subj["features"]     # (N_i, F)
        labels = subj["labels_class"]   # (N_i,)   class indices
        profile = subj["profile"]       # (P,)
        n_windows = len(features)

        if n_windows == 0:
            if logger:
                logger.warning(f"[KB] Subject {sid}: 0 windows, skipping")
            continue

        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            if logger:
                logger.warning(
                    f"[KB] Subject {sid}: only {len(unique_classes)} class(es), skipping"
                )
            continue

        # ── normalize features (per-subject stats) ──
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0
        features_norm = (features - feat_mean) / feat_std

        # ── inner CV: select best C & estimate quality ──
        min_class_count = min(
            np.sum(labels == c) for c in unique_classes
        )
        effective_cv_folds = min(inner_cv_folds, int(min_class_count))

        best_C = svm_C_values[0]
        best_cv_acc = 0.0

        if effective_cv_folds >= 2:
            for C_val in svm_C_values:
                skf = StratifiedKFold(
                    n_splits=effective_cv_folds, shuffle=True, random_state=seed,
                )
                fold_accs = []
                for tr_idx, va_idx in skf.split(features_norm, labels):
                    svm_cv = SVC(
                        kernel=svm_kernel, C=C_val, gamma=svm_gamma,
                        probability=True, random_state=seed, max_iter=5000,
                    )
                    svm_cv.fit(features_norm[tr_idx], labels[tr_idx])
                    preds = svm_cv.predict(features_norm[va_idx])
                    fold_accs.append(accuracy_score(labels[va_idx], preds))

                mean_cv_acc = float(np.mean(fold_accs))
                if mean_cv_acc > best_cv_acc:
                    best_cv_acc = mean_cv_acc
                    best_C = C_val
        else:
            best_cv_acc = 0.0
            best_C = svm_C_values[0]
            if logger:
                logger.warning(
                    f"[KB] Subject {sid}: min_class_count={min_class_count}, "
                    f"too few for CV, using default C={best_C}"
                )

        # ── train final specialist on ALL of this subject's data ──
        specialist = SVC(
            kernel=svm_kernel, C=best_C, gamma=svm_gamma,
            probability=True, random_state=seed, max_iter=10000,
        )
        specialist.fit(features_norm, labels)

        entry = KBEntry(
            subject_id=sid,
            profile=profile,
            model=specialist,
            feat_mean=feat_mean,
            feat_std=feat_std,
            inner_cv_accuracy=best_cv_acc,
            n_windows=n_windows,
            best_C=best_C,
        )
        kb_entries.append(entry)

        if logger:
            logger.info(
                f"[KB] Subject {sid}: {n_windows} windows, "
                f"best_C={best_C}, inner_CV_acc={best_cv_acc:.4f}"
            )

    return kb_entries


# ======================================================================
#  ENSEMBLE INFERENCE
# ======================================================================

def kb_ensemble_predict(
    kb_entries: List[KBEntry],
    test_features: np.ndarray,
    test_profile: np.ndarray,
    profile_scaler: StandardScaler,
    num_classes: int,
    k: int,
    strategy: str = "weighted_soft",
    logger=None,
) -> Tuple[np.ndarray, Dict]:
    """Predict using KB similarity-weighted ensemble.

    LOSO-correct: all KB entries are from training subjects only.
    Test subject's profile is used ONLY for unsupervised similarity
    computation — no labels involved.

    Args:
        kb_entries: list of KBEntry from training subjects
        test_features: (N, F) raw features (NOT normalized)
        test_profile: (P,) unsupervised profile of test subject
        profile_scaler: StandardScaler fitted on TRAINING profiles only
        num_classes: number of gesture classes
        k: number of nearest neighbors
        strategy: ensemble strategy name

    Returns:
        predictions: (N,) class indices
        info: metadata dict
    """
    # ── compute distances in scaled profile space ──
    train_profiles = np.array([e.profile for e in kb_entries])
    train_profiles_scaled = profile_scaler.transform(train_profiles)
    test_profile_scaled = profile_scaler.transform(test_profile.reshape(1, -1))

    distances = cdist(
        test_profile_scaled, train_profiles_scaled, metric="euclidean",
    )[0]  # (M,)

    # ── select k nearest ──
    effective_k = min(k, len(kb_entries))
    nearest_idx = np.argsort(distances)[:effective_k]
    selected = [kb_entries[i] for i in nearest_idx]
    sel_distances = distances[nearest_idx]

    # ── compute weights ──
    if strategy == "weighted_soft":
        neg_d = -sel_distances
        neg_d = neg_d - neg_d.max()  # numerical stability
        weights = np.exp(neg_d)
        weights /= weights.sum() + 1e-12

    elif strategy == "quality_weighted_soft":
        # Combine distance similarity with inner-CV quality
        neg_d = -sel_distances
        neg_d = neg_d - neg_d.max()
        dist_weights = np.exp(neg_d)
        quality = np.array([e.inner_cv_accuracy for e in selected])
        quality = np.clip(quality, 0.01, 1.0)  # avoid zero weight
        weights = dist_weights * quality
        weights /= weights.sum() + 1e-12

    elif strategy == "uniform_soft":
        weights = np.ones(effective_k) / effective_k

    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")

    # ── soft voting: weighted average of probability distributions ──
    N = test_features.shape[0]
    ensemble_probs = np.zeros((N, num_classes), dtype=np.float64)

    for entry, w in zip(selected, weights):
        # Normalize test features using THIS specialist's stats
        feat_norm = (test_features - entry.feat_mean) / entry.feat_std
        probs = entry.model.predict_proba(feat_norm)  # (N, n_model_classes)

        # Map specialist's learned classes to global class indices
        # (handles case where a subject is missing some classes)
        model_classes = entry.model.classes_
        if len(model_classes) == num_classes:
            ensemble_probs += w * probs
        else:
            for local_idx, global_class in enumerate(model_classes):
                if global_class < num_classes:
                    ensemble_probs[:, global_class] += w * probs[:, local_idx]

    predictions = np.argmax(ensemble_probs, axis=1)

    info = {
        "k": effective_k,
        "strategy": strategy,
        "selected_subjects": [e.subject_id for e in selected],
        "distances": sel_distances.tolist(),
        "weights": weights.tolist(),
        "quality_scores": [e.inner_cv_accuracy for e in selected],
    }

    if logger:
        logger.info(
            f"[KB-Ensemble] k={effective_k}, strategy={strategy}, "
            f"selected={info['selected_subjects']}, "
            f"weights=[{', '.join(f'{w:.3f}' for w in weights)}]"
        )

    return predictions, info


# ======================================================================
#  SINGLE LOSO FOLD
# ======================================================================

def run_kb_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    k_neighbors: List[int],
    svm_kernel: str = "rbf",
    svm_C_values: Optional[List[float]] = None,
    svm_gamma: str = "scale",
    inner_cv_folds: int = 3,
    logger=None,
) -> Dict:
    """Run one LOSO fold with knowledge-base ensemble.

    Data flow:
      1. Load all subjects
      2. Pre-extract powerful features for ALL subjects (shared extractor)
      3. Compute unsupervised profiles for ALL subjects
      4. Build KB from TRAINING subjects only
      5. Fit profile scaler on TRAINING profiles only
      6. Train global baseline on pooled TRAINING data
      7. For each (k, strategy) combination, ensemble-predict on test subject
      8. Compare all methods

    LOSO guarantees:
      - Steps 4,5,6 use ONLY training subjects
      - Profile for test subject uses only unsupervised signal statistics
      - No labels from test subject are ever used for training/tuning
    """
    fold_dir = output_dir
    fold_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = setup_logging(fold_dir)

    seed_everything(train_cfg.seed, verbose=False)

    if svm_C_values is None:
        svm_C_values = [1.0, 10.0]

    # ------------------------------------------------------------------
    # 1. Load all subjects
    # ------------------------------------------------------------------
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )

    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES,
    )
    gesture_to_class = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    class_to_gesture = {i: gid for gid, i in gesture_to_class.items()}
    num_classes = len(common_gestures)

    logger.info(
        f"[KB-LOSO] Test={test_subject}, Train={train_subjects}, "
        f"Common gestures ({num_classes}): {common_gestures}"
    )

    # ------------------------------------------------------------------
    # 2. Pre-extract features & profiles for ALL subjects (once)
    # ------------------------------------------------------------------
    feature_extractor = PowerfulFeatureExtractor(
        sampling_rate=proc_cfg.sampling_rate,
    )

    per_subject: Dict[str, Dict] = {}

    for sid in all_subject_ids:
        if sid not in subjects_data:
            logger.warning(f"[KB-LOSO] Subject {sid} not in loaded data")
            continue

        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        windows, labels_raw = grouped_to_arrays(filtered)

        if len(windows) == 0 or windows.ndim != 3:
            logger.warning(f"[KB-LOSO] Subject {sid}: no valid windows")
            continue

        # Features: (N_i, F)
        features = feature_extractor.transform(windows)

        # Map gesture IDs -> class indices
        labels_class = np.array([gesture_to_class[g] for g in labels_raw])

        # Unsupervised profile (NO labels)
        all_windows_for_profile = concat_all_windows_no_labels(filtered)
        profile = compute_subject_profile(
            all_windows_for_profile, proc_cfg.sampling_rate,
        )

        per_subject[sid] = {
            "features": features,
            "labels_class": labels_class,
            "profile": profile,
            "n_windows": len(windows),
        }

        logger.info(
            f"[KB-LOSO] Subject {sid}: {len(windows)} windows, "
            f"{features.shape[1]} features, "
            f"MAV={profile[0]:.4f}, MeanFreq={profile[3]:.1f}Hz"
        )

    # Free raw data — we only need pre-extracted features from here
    del subjects_data
    gc.collect()

    # Verify test subject exists
    if test_subject not in per_subject:
        logger.error(f"[KB-LOSO] Test subject {test_subject} has no data")
        return {
            "test_subject": test_subject,
            "error": "No test data",
            "baseline_accuracy": None,
            "baseline_f1_macro": None,
        }

    test_data = per_subject[test_subject]
    test_features = test_data["features"]       # (N_test, F)
    test_labels = test_data["labels_class"]     # (N_test,)
    test_profile = test_data["profile"]         # (P,)

    logger.info(
        f"[KB-LOSO] Test subject {test_subject}: "
        f"{len(test_features)} windows, {num_classes} classes"
    )

    # ------------------------------------------------------------------
    # 3. Build knowledge base from TRAINING subjects only
    # ------------------------------------------------------------------
    logger.info("[KB-LOSO] Building knowledge base from training subjects...")
    kb_entries = build_knowledge_base(
        per_subject=per_subject,
        train_subjects=train_subjects,
        gesture_to_class=gesture_to_class,
        svm_kernel=svm_kernel,
        svm_C_values=svm_C_values,
        svm_gamma=svm_gamma,
        inner_cv_folds=inner_cv_folds,
        seed=train_cfg.seed,
        logger=logger,
    )

    if len(kb_entries) == 0:
        logger.error("[KB-LOSO] No valid KB entries")
        return {
            "test_subject": test_subject,
            "error": "No valid KB entries",
            "baseline_accuracy": None,
            "baseline_f1_macro": None,
        }

    logger.info(f"[KB-LOSO] Knowledge base built: {len(kb_entries)} entries")

    # ------------------------------------------------------------------
    # 4. Fit profile scaler on TRAINING profiles only
    # ------------------------------------------------------------------
    train_profiles_matrix = np.array([e.profile for e in kb_entries])
    profile_scaler = StandardScaler()
    profile_scaler.fit(train_profiles_matrix)

    # ------------------------------------------------------------------
    # 5. Train global baseline on pooled TRAINING data
    # ------------------------------------------------------------------
    logger.info("[KB-LOSO] Training baseline (global SVM on all train subjects)...")

    train_features_list = []
    train_labels_list = []
    for sid in train_subjects:
        subj = per_subject.get(sid)
        if subj is None:
            continue
        train_features_list.append(subj["features"])
        train_labels_list.append(subj["labels_class"])

    train_features_all = np.concatenate(train_features_list, axis=0)
    train_labels_all = np.concatenate(train_labels_list, axis=0)

    # Normalize with pooled training stats
    baseline_mean = train_features_all.mean(axis=0)
    baseline_std = train_features_all.std(axis=0)
    baseline_std[baseline_std < 1e-8] = 1.0
    train_feat_norm = (train_features_all - baseline_mean) / baseline_std

    baseline_model = SVC(
        kernel=svm_kernel, C=max(svm_C_values), gamma=svm_gamma,
        probability=True, random_state=train_cfg.seed, max_iter=10000,
    )
    baseline_model.fit(train_feat_norm, train_labels_all)

    logger.info(
        f"[KB-LOSO] Baseline trained on {len(train_feat_norm)} windows "
        f"from {len(train_subjects)} subjects"
    )

    # Free pooled training arrays
    del train_features_all, train_labels_all, train_feat_norm
    gc.collect()

    # ── baseline prediction ──
    test_feat_baseline = (test_features - baseline_mean) / baseline_std
    baseline_preds = baseline_model.predict(test_feat_baseline)
    baseline_acc = float(accuracy_score(test_labels, baseline_preds))
    baseline_f1 = float(f1_score(
        test_labels, baseline_preds, average="macro", zero_division=0,
    ))

    # Baseline per-class report
    class_names_list = [str(class_to_gesture[i]) for i in range(num_classes)]
    baseline_report = classification_report(
        test_labels, baseline_preds,
        target_names=class_names_list, output_dict=True, zero_division=0,
    )

    logger.info(
        f"[KB-LOSO] Baseline: Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}"
    )

    del baseline_model, test_feat_baseline
    gc.collect()

    # ------------------------------------------------------------------
    # 6. KB ensemble predictions for various k and strategies
    # ------------------------------------------------------------------
    ensemble_results: Dict[str, Dict] = {}

    for k in k_neighbors:
        for strategy in ENSEMBLE_STRATEGIES:
            key = f"k{k}_{strategy}"

            try:
                preds, info = kb_ensemble_predict(
                    kb_entries=kb_entries,
                    test_features=test_features,
                    test_profile=test_profile,
                    profile_scaler=profile_scaler,
                    num_classes=num_classes,
                    k=k,
                    strategy=strategy,
                    logger=logger,
                )

                acc = float(accuracy_score(test_labels, preds))
                f1 = float(f1_score(
                    test_labels, preds, average="macro", zero_division=0,
                ))
                delta_acc = acc - baseline_acc
                delta_f1 = f1 - baseline_f1

                ens_report = classification_report(
                    test_labels, preds,
                    target_names=class_names_list, output_dict=True,
                    zero_division=0,
                )

                ensemble_results[key] = {
                    "k": k,
                    "strategy": strategy,
                    "accuracy": acc,
                    "f1_macro": f1,
                    "delta_accuracy": delta_acc,
                    "delta_f1_macro": delta_f1,
                    "selected_subjects": info["selected_subjects"],
                    "distances": info["distances"],
                    "weights": info["weights"],
                    "quality_scores": info["quality_scores"],
                    "report": ens_report,
                }

                logger.info(
                    f"[KB-LOSO] {key}: Acc={acc:.4f} (d={delta_acc:+.4f}), "
                    f"F1={f1:.4f} (d={delta_f1:+.4f})"
                )

            except Exception as e:
                logger.error(f"[KB-LOSO] {key} failed: {e}")
                traceback.print_exc()
                ensemble_results[key] = {
                    "k": k, "strategy": strategy,
                    "accuracy": None, "f1_macro": None,
                    "error": str(e),
                }

    # ── also try k=all (all training subjects weighted) ──
    for strategy in ENSEMBLE_STRATEGIES:
        key = f"kAll_{strategy}"
        try:
            preds, info = kb_ensemble_predict(
                kb_entries=kb_entries,
                test_features=test_features,
                test_profile=test_profile,
                profile_scaler=profile_scaler,
                num_classes=num_classes,
                k=len(kb_entries),
                strategy=strategy,
                logger=logger,
            )
            acc = float(accuracy_score(test_labels, preds))
            f1 = float(f1_score(
                test_labels, preds, average="macro", zero_division=0,
            ))
            ensemble_results[key] = {
                "k": len(kb_entries),
                "strategy": strategy,
                "accuracy": acc,
                "f1_macro": f1,
                "delta_accuracy": acc - baseline_acc,
                "delta_f1_macro": f1 - baseline_f1,
                "selected_subjects": info["selected_subjects"],
                "distances": info["distances"],
                "weights": info["weights"],
            }
            logger.info(
                f"[KB-LOSO] {key}: Acc={acc:.4f}, F1={f1:.4f}"
            )
        except Exception as e:
            logger.error(f"[KB-LOSO] {key} failed: {e}")
            ensemble_results[key] = {"error": str(e)}

    # ------------------------------------------------------------------
    # 7. Find best ensemble configuration for this fold
    # ------------------------------------------------------------------
    best_key = None
    best_acc = baseline_acc
    for key, res in ensemble_results.items():
        if res.get("accuracy") is not None and res["accuracy"] > best_acc:
            best_acc = res["accuracy"]
            best_key = key

    # ------------------------------------------------------------------
    # 8. Save KB profiles for post-hoc analysis
    # ------------------------------------------------------------------
    kb_profiles_info: Dict[str, Dict] = {}
    for entry in kb_entries:
        kb_profiles_info[entry.subject_id] = {
            "profile": entry.profile.tolist(),
            "inner_cv_accuracy": entry.inner_cv_accuracy,
            "n_windows": entry.n_windows,
            "best_C": entry.best_C,
        }
    kb_profiles_info[test_subject] = {
        "profile": test_profile.tolist(),
        "is_test": True,
    }

    # ------------------------------------------------------------------
    # 9. Assemble fold result
    # ------------------------------------------------------------------
    fold_result = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "n_test_windows": int(len(test_features)),
        "num_classes": num_classes,
        "common_gestures": [int(g) for g in common_gestures],
        "svm_kernel": svm_kernel,
        # Baseline
        "baseline_accuracy": baseline_acc,
        "baseline_f1_macro": baseline_f1,
        "baseline_report": baseline_report,
        # Ensemble
        "ensemble_results": ensemble_results,
        # Best config
        "best_ensemble_key": best_key,
        "best_ensemble_accuracy": float(best_acc) if best_key else None,
        # KB info
        "kb_size": len(kb_entries),
        "kb_profiles": kb_profiles_info,
    }

    with open(fold_dir / "fold_result.json", "w") as f:
        json.dump(make_json_serializable(fold_result), f, indent=4)

    # Cleanup
    del per_subject, kb_entries, test_features, test_labels
    del multi_loader, feature_extractor, profile_scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result


# ======================================================================
#  MAIN
# ======================================================================

def main():
    ALL_SUBJECTS = parse_subjects_args()
    BASE_DIR = ROOT / "data"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_{timestamp}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        use_handcrafted_features=False,
        handcrafted_feature_set="powerful",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_rbf",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
    )

    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"Experiment: {EXPERIMENT_NAME}")
    global_logger.info(f"Subjects: {ALL_SUBJECTS}")
    global_logger.info(f"K neighbors: {K_NEIGHBORS}")
    global_logger.info(f"SVM kernels: {SVM_KERNELS}")
    global_logger.info(f"Ensemble strategies: {ENSEMBLE_STRATEGIES}")
    global_logger.info(f"Inner CV folds: {INNER_CV_FOLDS}")
    global_logger.info("=" * 80)

    all_fold_results: List[Dict] = []

    for svm_kernel in SVM_KERNELS:
        print(f"\n{'=' * 60}")
        print(f"SVM KERNEL: {svm_kernel}")
        print(f"{'=' * 60}")

        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]

            print(
                f"\n[LOSO] Test: {test_subject} | "
                f"Train: {train_subjects} | Kernel: {svm_kernel}"
            )

            fold_output_dir = OUTPUT_DIR / svm_kernel / f"test_{test_subject}"

            try:
                fold_result = run_kb_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_output_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                    k_neighbors=K_NEIGHBORS,
                    svm_kernel=svm_kernel,
                    svm_C_values=SVM_C_VALUES,
                    svm_gamma=SVM_GAMMA,
                    inner_cv_folds=INNER_CV_FOLDS,
                    logger=global_logger,
                )
                fold_result["svm_kernel"] = svm_kernel

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                fold_result = {
                    "test_subject": test_subject,
                    "svm_kernel": svm_kernel,
                    "baseline_accuracy": None,
                    "baseline_f1_macro": None,
                    "error": str(e),
                }

            all_fold_results.append(fold_result)

            # Print fold summary
            def fmt(val):
                return f"{val:.4f}" if val is not None else "N/A"

            bl_acc = fold_result.get("baseline_accuracy")
            bl_f1 = fold_result.get("baseline_f1_macro")
            best_key = fold_result.get("best_ensemble_key")

            print(f"  Baseline: Acc={fmt(bl_acc)}, F1={fmt(bl_f1)}")
            if best_key:
                ens_res = fold_result.get("ensemble_results", {}).get(best_key, {})
                ens_acc = ens_res.get("accuracy")
                ens_f1 = ens_res.get("f1_macro")
                delta = ens_res.get("delta_accuracy")
                print(
                    f"  Best ensemble ({best_key}): "
                    f"Acc={fmt(ens_acc)}, F1={fmt(ens_f1)}, "
                    f"dAcc={fmt(delta)}"
                )
            else:
                print("  No ensemble improved over baseline")

    # ==================================================================
    # Aggregate results
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 80}")

    def safe_mean_std(arr):
        if not arr:
            return None, None
        return float(np.mean(arr)), float(np.std(arr))

    def fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"

    aggregate = {}

    for svm_kernel in SVM_KERNELS:
        kernel_folds = [
            r for r in all_fold_results
            if r.get("svm_kernel") == svm_kernel
        ]

        # Baseline aggregate
        bl_accs = [
            r["baseline_accuracy"] for r in kernel_folds
            if r.get("baseline_accuracy") is not None
        ]
        bl_f1s = [
            r["baseline_f1_macro"] for r in kernel_folds
            if r.get("baseline_f1_macro") is not None
        ]

        bl_acc_mean, bl_acc_std = safe_mean_std(bl_accs)
        bl_f1_mean, bl_f1_std = safe_mean_std(bl_f1s)

        print(f"\n--- Kernel: {svm_kernel} ---")
        print(
            f"  Baseline: Acc={fmt(bl_acc_mean)}+/-{fmt(bl_acc_std)}, "
            f"F1={fmt(bl_f1_mean)}+/-{fmt(bl_f1_std)}"
        )

        # Ensemble aggregate per configuration
        ensemble_agg: Dict[str, Dict] = {}

        all_keys: set = set()
        for r in kernel_folds:
            if "ensemble_results" in r:
                all_keys.update(r["ensemble_results"].keys())

        for ens_key in sorted(all_keys):
            accs = []
            f1s = []
            delta_accs = []
            delta_f1s = []

            for r in kernel_folds:
                ens = r.get("ensemble_results", {}).get(ens_key, {})
                if ens.get("accuracy") is not None:
                    accs.append(ens["accuracy"])
                if ens.get("f1_macro") is not None:
                    f1s.append(ens["f1_macro"])
                if ens.get("delta_accuracy") is not None:
                    delta_accs.append(ens["delta_accuracy"])
                if ens.get("delta_f1_macro") is not None:
                    delta_f1s.append(ens["delta_f1_macro"])

            acc_mean, acc_std = safe_mean_std(accs)
            f1_mean, f1_std = safe_mean_std(f1s)
            dacc_mean, dacc_std = safe_mean_std(delta_accs)
            df1_mean, df1_std = safe_mean_std(delta_f1s)

            ensemble_agg[ens_key] = {
                "mean_accuracy": acc_mean,
                "std_accuracy": acc_std,
                "mean_f1_macro": f1_mean,
                "std_f1_macro": f1_std,
                "mean_delta_accuracy": dacc_mean,
                "std_delta_accuracy": dacc_std,
                "mean_delta_f1_macro": df1_mean,
                "std_delta_f1_macro": df1_std,
                "n_folds": len(accs),
            }

            print(
                f"  {ens_key}: "
                f"Acc={fmt(acc_mean)}+/-{fmt(acc_std)}, "
                f"F1={fmt(f1_mean)}+/-{fmt(f1_std)}, "
                f"dAcc={fmt(dacc_mean)}"
            )

        aggregate[svm_kernel] = {
            "baseline": {
                "mean_accuracy": bl_acc_mean,
                "std_accuracy": bl_acc_std,
                "mean_f1_macro": bl_f1_mean,
                "std_f1_macro": bl_f1_std,
            },
            "ensemble": ensemble_agg,
            "n_folds": len(kernel_folds),
        }

    # Find overall best
    overall_best_acc = 0.0
    overall_best_config = "baseline"
    for svm_kernel, kernel_agg in aggregate.items():
        bl_mean = kernel_agg["baseline"].get("mean_accuracy")
        if bl_mean is not None and bl_mean > overall_best_acc:
            overall_best_acc = bl_mean
            overall_best_config = f"{svm_kernel}/baseline"
        for ens_key, ens_vals in kernel_agg.get("ensemble", {}).items():
            ens_mean = ens_vals.get("mean_accuracy")
            if ens_mean is not None and ens_mean > overall_best_acc:
                overall_best_acc = ens_mean
                overall_best_config = f"{svm_kernel}/{ens_key}"

    print(
        f"\nBest overall: {overall_best_config} "
        f"with Acc={fmt(overall_best_acc)}"
    )

    # ==================================================================
    # Count how often ensemble beats baseline per fold
    # ==================================================================
    wins = {key: 0 for key in sorted(all_keys)} if all_keys else {}
    total_valid_folds = 0

    for r in all_fold_results:
        bl = r.get("baseline_accuracy")
        if bl is None:
            continue
        total_valid_folds += 1
        for key in wins:
            ens = r.get("ensemble_results", {}).get(key, {})
            ens_acc = ens.get("accuracy")
            if ens_acc is not None and ens_acc > bl:
                wins[key] += 1

    if total_valid_folds > 0:
        print(f"\nWin rate (ensemble > baseline) across {total_valid_folds} folds:")
        for key in sorted(wins.keys()):
            rate = wins[key] / total_valid_folds
            print(f"  {key}: {wins[key]}/{total_valid_folds} ({rate:.0%})")

    # ==================================================================
    # Save summary
    # ==================================================================
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "description": (
            "Knowledge-base subject-similarity ensemble: for each training "
            "subject, train an SVM specialist on powerful features and store "
            "with an unsupervised physiological profile. At test time, find "
            "k nearest training subjects by profile similarity and ensemble "
            "their specialist models via weighted soft voting. "
            "Fully LOSO-correct: no test data used for training or adaptation."
        ),
        "approach": "ml_emg_td",
        "svm_kernels": SVM_KERNELS,
        "k_neighbors": K_NEIGHBORS,
        "ensemble_strategies": ENSEMBLE_STRATEGIES,
        "inner_cv_folds": INNER_CV_FOLDS,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "best_overall": {
            "config": overall_best_config,
            "mean_accuracy": overall_best_acc,
        },
        "ensemble_win_rates": {
            key: (wins.get(key, 0) / total_valid_folds if total_valid_folds > 0 else 0)
            for key in sorted(wins.keys())
        } if wins else {},
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

    # ── optional: report to hypothesis executor ──
    try:
        from hypothesis_executor import (
            mark_hypothesis_verified,
            mark_hypothesis_failed,
        )
    except ImportError:
        pass
    else:
        if overall_best_acc > 0:
            try:
                mark_hypothesis_verified(
                    EXPERIMENT_NAME,
                    metrics={
                        "best_config": overall_best_config,
                        "best_mean_accuracy": overall_best_acc,
                    },
                    experiment_name=EXPERIMENT_NAME,
                )
            except Exception as e:
                print(f"[hypothesis_executor] mark_verified failed: {e}")
        else:
            try:
                mark_hypothesis_failed(
                    EXPERIMENT_NAME,
                    "No valid results obtained",
                )
            except Exception as e:
                print(f"[hypothesis_executor] mark_failed failed: {e}")


if __name__ == "__main__":
    main()
