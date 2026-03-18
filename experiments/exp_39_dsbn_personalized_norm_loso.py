"""
Experiment 39: Personalized Normalization via Domain-Specific Batch Normalization
              (DSBN) + Instance Normalization for Cross-Subject EMG LOSO

Hypothesis:
    Combining InstanceNorm at the input (removes per-sample amplitude bias from
    electrode placement) with Domain-Specific BatchNorm in CNN layers (one set of
    affine parameters per cluster of similar subjects) preserves class-discriminative
    temporal patterns while reducing inter-subject distribution shift — without
    destroying discriminative information the way global domain adversarial methods do.

Method:
    1. Cluster training subjects into K groups via K-Means on per-channel RMS stats.
    2. Assign each training window a domain_id (the subject's cluster).
    3. Train DSBNCNNGRUAttention with DomainBN1d (shared running stats, per-domain
       affine) and InstanceNorm1d at the input.
    4. At test time: compute RMS stats for the test subject, find the nearest
       cluster centroid (Euclidean distance), and forward-pass all test windows
       using that domain_id.

Expected improvement over exp_1 baseline (CNN-GRU-Attn: 30.85% / 30.74% CI):
    - InstanceNorm should help the model generalize to unseen amplitude ranges.
    - DSBN allows cluster-specific feature normalization to emerge naturally,
      unlike global BN which conflates all subject distributions.

References:
    - Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation" (2018)
    - Pan et al., "Two at Once: Enhancing Learning and Generalization via IBN-Net" (2018)
    - exp_26 showed test-time BN adaptation matches baseline — DSBN is a training-time
      complement that should stack with it.

Cannot use CrossSubjectExperiment.run() directly because _prepare_splits() merges
all training subjects and loses subject identity (needed for domain_id assignment).
Instead we implement a standalone LOSO loop identical in spirit to exp_31.
"""

import os
import sys
import json
import gc
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from utils.logging import setup_logging, seed_everything

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
EXPERIMENT_NAME = "exp_39_dsbn_personalized_norm"
EXERCISES = ["E1"]
USE_IMPROVED_PROCESSING = True
INCLUDE_REST = False       # use only active gestures (IDs 8-17)
MAX_GESTURES = 10

N_DOMAINS = 3              # max number of subject clusters; auto-reduced for small cohorts
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 80
PATIENCE = 15              # early stopping patience (val accuracy)
GRAD_CLIP = 1.0
NOISE_STD = 0.01           # light augmentation during training
SEED = 42

# ============================================================
# LOCAL HELPER: grouped_to_arrays
# (NOT in any processing/ module — must be defined locally)
# ============================================================
def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten grouped_windows {gesture_id: [rep_array, ...]} into (windows, labels).

    Returns:
        windows: (N, T, C) float32
        labels:  (N,) int64 of gesture_ids (raw, NOT class-index remapped)
    """
    windows_list, labels_list = [], []
    for gesture_id in sorted(grouped_windows.keys()):
        for rep_array in grouped_windows[gesture_id]:
            if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                windows_list.append(rep_array)
                labels_list.append(
                    np.full(len(rep_array), gesture_id, dtype=np.int64)
                )
    if not windows_list:
        return (
            np.empty((0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.concatenate(windows_list, axis=0).astype(np.float32),
        np.concatenate(labels_list, axis=0),
    )


# ============================================================
# SUBJECT STATISTICS  (used for clustering)
# ============================================================
def compute_subject_stats(windows: np.ndarray) -> np.ndarray:
    """
    Compact descriptor of a subject's signal distribution.

    windows: (N, T, C) — raw windows for one subject.

    Returns a 2*C-dimensional vector:
        [mean_rms_per_channel, std_rms_per_channel]

    RMS captures electrode contact quality / muscle strength differences.
    Mean and std over repetitions capture consistency.
    """
    # Per-window, per-channel RMS:  (N, C)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))  # axis=1 is T
    mean_rms = rms.mean(axis=0)                   # (C,)
    std_rms = rms.std(axis=0) + 1e-8              # (C,)
    return np.concatenate([mean_rms, std_rms]).astype(np.float32)


# ============================================================
# SUBJECT CLUSTERING
# ============================================================
def cluster_subjects(
    subj_stats: Dict[str, np.ndarray],
    n_clusters: int,
    seed: int = SEED,
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Cluster training subjects into n_clusters groups using K-Means.

    Args:
        subj_stats: {subject_id: stats_vector}
        n_clusters: desired number of clusters
        seed:       random seed

    Returns:
        domain_map:  {subject_id: cluster_id}
        centroids:   (n_clusters, stats_dim) cluster centroids in stats space
    """
    subjects = sorted(subj_stats.keys())
    X = np.stack([subj_stats[s] for s in subjects], axis=0)  # (n_subj, stats_dim)

    # Ensure n_clusters ≤ number of subjects
    n_clusters = min(n_clusters, len(subjects))

    if n_clusters == 1:
        centroids = X.mean(axis=0, keepdims=True)            # (1, stats_dim)
        return {s: 0 for s in subjects}, centroids

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)
    domain_map = {s: int(l) for s, l in zip(subjects, labels)}
    return domain_map, kmeans.cluster_centers_


def find_nearest_cluster(
    test_stats: np.ndarray,
    centroids: np.ndarray,
) -> int:
    """
    Return the cluster index whose centroid is closest to test_stats (L2 distance).
    """
    dists = np.linalg.norm(centroids - test_stats[None, :], axis=1)  # (n_clusters,)
    return int(np.argmin(dists))


# ============================================================
# DATASET
# ============================================================
class DSBNDataset(Dataset):
    """
    Dataset returning (window, class_label, domain_id) triples.

    windows:    (N, T, C) — will be transposed to (N, C, T) for Conv1d
    labels:     (N,) int64 — class indices (0-indexed, not gesture_ids)
    domain_ids: (N,) int64 — cluster assignment
    augment:    if True, adds Gaussian noise during training
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        domain_ids: np.ndarray,
        augment: bool = False,
        noise_std: float = NOISE_STD,
    ):
        # (N, T, C) → (N, C, T) for Conv1d
        self.X = torch.from_numpy(windows.transpose(0, 2, 1)).float()
        self.y = torch.from_numpy(labels).long()
        self.d = torch.from_numpy(domain_ids.astype(np.int64)).long()
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.y[idx], self.d[idx]


# ============================================================
# STANDARDIZATION  (per-channel, on training data)
# ============================================================
def compute_channel_stats(
    X_train_ct: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std for standardization.

    X_train_ct: (N, C, T) format.
    Returns mean (C,), std (C,).
    """
    mean = X_train_ct.mean(axis=(0, 2)).astype(np.float32)
    std = (X_train_ct.std(axis=(0, 2)) + 1e-8).astype(np.float32)
    return mean, std


def apply_channel_std(
    X_ct: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply per-channel standardization to (N, C, T) array."""
    return (X_ct - mean[None, :, None]) / std[None, :, None]


# ============================================================
# TRAINING LOOP
# ============================================================
def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger,
    fold_output_dir: Path,
) -> nn.Module:
    """
    Train DSBNCNNGRUAttention for one LOSO fold with early stopping.

    Returns the best model (by val accuracy).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Training ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch, d_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            d_batch = d_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x_batch, d_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        train_acc = train_correct / max(train_total, 1)
        train_loss /= max(train_total, 1)

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for x_batch, y_batch, d_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                d_batch = d_batch.to(device, non_blocking=True)
                logits = model(x_batch, d_batch)
                val_preds_all.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels_all.extend(y_batch.numpy())

        val_acc = accuracy_score(val_labels_all, val_preds_all)
        scheduler.step(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        # ── Early stopping ───────────────────────────────────────────────
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"  Best val_acc = {best_val_acc:.4f}")
    return model


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    X_ct: np.ndarray,
    y: np.ndarray,
    test_domain_id: int,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> Tuple[float, float]:
    """
    Evaluate model on numpy arrays (N, C, T).

    All test samples receive test_domain_id (the assigned nearest cluster).

    Returns (accuracy, f1_macro).
    """
    model.eval()
    preds_all = []

    n = len(X_ct)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = torch.from_numpy(X_ct[start:end]).float().to(device)
        d_batch = torch.full((end - start,), test_domain_id, dtype=torch.long, device=device)
        logits = model(x_batch, d_batch)
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())

    acc = accuracy_score(y, preds_all)
    f1 = f1_score(y, preds_all, average="macro", zero_division=0)
    return float(acc), float(f1)


# ============================================================
# LOSO FOLD
# ============================================================
def run_loso_fold(
    test_subject: str,
    all_subjects: List[str],
    subjects_data: Dict,          # {subj_id: (emg, segments, grouped_windows)}
    common_gestures: List[int],
    gesture_to_class: Dict[int, int],
    num_actual_domains: int,
    proc_cfg: ProcessingConfig,
    device: torch.device,
    output_dir: Path,
    logger,
) -> Dict:
    """Run one LOSO fold: train on all-but-test, evaluate on test."""

    train_subjects = [s for s in all_subjects if s != test_subject]
    fold_dir = output_dir / test_subject
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO fold: test={test_subject}, n_train={len(train_subjects)}")

    # ── 1. Per-subject stats for clustering ─────────────────────────────────
    subj_stats: Dict[str, np.ndarray] = {}
    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for sid in train_subjects:
        if sid not in subjects_data:
            logger.warning(f"  {sid} not in subjects_data, skipping")
            continue
        _, _, grouped_windows = subjects_data[sid]
        windows_raw, labels_raw = grouped_to_arrays(grouped_windows)

        # Filter to common gestures
        mask = np.isin(labels_raw, common_gestures)
        windows_raw = windows_raw[mask]
        labels_raw = labels_raw[mask]

        if len(windows_raw) == 0:
            logger.warning(f"  {sid}: no windows after gesture filter, skipping")
            continue

        # Remap gesture_ids → class indices
        labels_cls = np.array([gesture_to_class[g] for g in labels_raw], dtype=np.int64)
        subj_arrays[sid] = (windows_raw, labels_cls)
        subj_stats[sid] = compute_subject_stats(windows_raw)

    if not subj_arrays:
        logger.error("No training subjects have data — skipping fold")
        return {
            "test_subject": test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No training data",
        }

    # ── 2. Cluster training subjects ─────────────────────────────────────────
    n_clusters = min(num_actual_domains, len(subj_arrays))
    domain_map, centroids = cluster_subjects(subj_stats, n_clusters=n_clusters)
    logger.info(f"  Clusters (k={n_clusters}): {domain_map}")

    # ── 3. Build combined training arrays with domain_ids ────────────────────
    win_list, lbl_list, dom_list = [], [], []
    for sid, (windows, labels) in subj_arrays.items():
        domain_id = domain_map[sid]
        win_list.append(windows)
        lbl_list.append(labels)
        dom_list.append(np.full(len(windows), domain_id, dtype=np.int64))

    X_all = np.concatenate(win_list, axis=0)   # (N, T, C)
    y_all = np.concatenate(lbl_list, axis=0)   # (N,)
    d_all = np.concatenate(dom_list, axis=0)   # (N,)

    # ── 4. Train / val split (stratified by class) ───────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, val_idx = next(sss.split(X_all, y_all))

    X_tr, y_tr, d_tr = X_all[train_idx], y_all[train_idx], d_all[train_idx]
    X_va, y_va, d_va = X_all[val_idx],   y_all[val_idx],   d_all[val_idx]

    # ── 5. Per-channel standardization (computed on training split) ──────────
    # Transpose to (N, C, T) for Conv1d and stats computation
    X_tr_ct = X_tr.transpose(0, 2, 1)  # (N, C, T)
    X_va_ct = X_va.transpose(0, 2, 1)
    mean_c, std_c = compute_channel_stats(X_tr_ct)
    X_tr_ct = apply_channel_std(X_tr_ct, mean_c, std_c)
    X_va_ct = apply_channel_std(X_va_ct, mean_c, std_c)

    logger.info(
        f"  Train: {X_tr_ct.shape}, Val: {X_va_ct.shape} | "
        f"Classes: {sorted(set(y_tr.tolist()))}"
    )

    # ── 6. DataLoaders ───────────────────────────────────────────────────────
    # DSBNDataset expects (N, T, C) — it does the transpose internally.
    # But we already have (N, C, T) here, so pass them transposed back to (N, T, C).
    X_tr_tc = X_tr_ct.transpose(0, 2, 1)  # back to (N, T, C) for Dataset
    X_va_tc = X_va_ct.transpose(0, 2, 1)

    from models.dsbn_cnn_gru import DSBNCNNGRUAttention

    ds_train = DSBNDataset(X_tr_tc, y_tr, d_tr, augment=True)
    ds_val = DSBNDataset(X_va_tc, y_va, d_va, augment=False)

    dl_train = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    dl_val = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── 7. Build model ───────────────────────────────────────────────────────
    in_channels = X_tr_ct.shape[1]  # C (number of EMG channels)
    num_classes = len(gesture_to_class)

    seed_everything(SEED)
    model = DSBNCNNGRUAttention(
        in_channels=in_channels,
        num_classes=num_classes,
        num_domains=n_clusters,
        dropout=0.3,
        cnn_channels=[32, 64],
        gru_hidden=128,
        gru_layers=2,
    ).to(device)

    logger.info(
        f"  Model: DSBNCNNGRUAttention | "
        f"in_ch={in_channels}, n_classes={num_classes}, n_domains={n_clusters}"
    )

    # ── 8. Train ─────────────────────────────────────────────────────────────
    model = train_one_fold(model, dl_train, dl_val, device, logger, fold_dir)

    # ── 9. Load test subject and assign cluster ───────────────────────────────
    if test_subject not in subjects_data:
        logger.error(f"  Test subject {test_subject} not found in subjects_data")
        return {
            "test_subject": test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": f"Test subject {test_subject} not in subjects_data",
        }

    _, _, test_grouped = subjects_data[test_subject]
    X_test_raw, y_test_raw = grouped_to_arrays(test_grouped)

    mask = np.isin(y_test_raw, common_gestures)
    X_test_raw = X_test_raw[mask]
    y_test_raw = y_test_raw[mask]
    y_test_cls = np.array([gesture_to_class[g] for g in y_test_raw], dtype=np.int64)

    # Compute test subject stats → find nearest cluster
    test_stats = compute_subject_stats(X_test_raw)
    test_domain_id = find_nearest_cluster(test_stats, centroids)
    logger.info(f"  Test subject assigned to domain {test_domain_id}")

    # Apply standardization (same stats from training data)
    X_test_ct = X_test_raw.transpose(0, 2, 1)  # (N, C, T)
    X_test_ct = apply_channel_std(X_test_ct, mean_c, std_c)

    # ── 10. Evaluate ──────────────────────────────────────────────────────────
    test_acc, test_f1 = evaluate(model, X_test_ct, y_test_cls, test_domain_id, device)

    logger.info(
        f"  [RESULT] test={test_subject} | acc={test_acc:.4f} | f1={test_f1:.4f} | "
        f"domain={test_domain_id}"
    )

    # Save fold result
    fold_result = {
        "test_subject": test_subject,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "assigned_domain": test_domain_id,
        "n_clusters": n_clusters,
        "cluster_map": domain_map,
        "n_train_windows": int(len(X_tr)),
        "n_val_windows": int(len(X_va)),
        "n_test_windows": int(len(X_test_raw)),
    }
    with open(fold_dir / "fold_result.json", "w") as f:
        json.dump(make_json_serializable(fold_result), f, indent=2)

    # Save model checkpoint
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "mean_c": mean_c,
            "std_c": std_c,
            "gesture_to_class": gesture_to_class,
            "test_domain_id": test_domain_id,
            "centroids": centroids.tolist(),
            "domain_map": domain_map,
        },
        fold_dir / "checkpoint.pt",
    )

    # Cleanup
    del model, ds_train, ds_val, dl_train, dl_val
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return fold_result


# ============================================================
# MAIN
# ============================================================
def main():
    # ── Arg parsing ──────────────────────────────────────────────────────────
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--n_domains", type=int, default=N_DOMAINS)
    _args, _ = _parser.parse_known_args()

    _CI_SUBJECTS = CI_TEST_SUBJECTS
    _FULL_SUBJECTS = DEFAULT_SUBJECTS

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.ci:
        ALL_SUBJECTS = _CI_SUBJECTS
    else:
        ALL_SUBJECTS = _CI_SUBJECTS   # default = CI on server (no full subject symlinks)

    num_actual_domains = _args.n_domains

    # ── Setup ─────────────────────────────────────────────────────────────────
    seed_everything(SEED)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Experiment: {EXPERIMENT_NAME}")
    logger.info(f"Subjects: {ALL_SUBJECTS}")
    logger.info(f"N_DOMAINS={num_actual_domains}, EXERCISES={EXERCISES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    BASE_DIR = ROOT / "data"

    # ── Processing config ─────────────────────────────────────────────────────
    proc_cfg = ProcessingConfig()
    proc_cfg.window_size = 600
    proc_cfg.window_step = 300

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=False,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

    # ── Load all subjects' data once ─────────────────────────────────────────
    logger.info("Loading all subjects...")
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=BASE_DIR,
        subject_ids=ALL_SUBJECTS,
        exercises=EXERCISES,
        include_rest=INCLUDE_REST,
    )
    logger.info(f"Loaded {len(subjects_data)} subjects")

    # ── Find common gesture set ───────────────────────────────────────────────
    common_gestures_set = None
    for sid, (_, _, gw) in subjects_data.items():
        gids = set(gw.keys())
        if common_gestures_set is None:
            common_gestures_set = gids
        else:
            common_gestures_set &= gids

    if common_gestures_set is None or len(common_gestures_set) == 0:
        raise RuntimeError("No common gestures found across subjects")

    # Limit to MAX_GESTURES most common (sorted ascending ID)
    common_gestures = sorted(common_gestures_set)[:MAX_GESTURES]
    gesture_to_class = {gid: i for i, gid in enumerate(common_gestures)}
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")
    logger.info(f"gesture_to_class: {gesture_to_class}")

    # ── LOSO loop ─────────────────────────────────────────────────────────────
    all_results = []
    failed_subjects = []

    for test_subject in ALL_SUBJECTS:
        if test_subject not in subjects_data:
            logger.warning(f"Skipping {test_subject}: not in loaded data")
            failed_subjects.append(test_subject)
            continue

        try:
            result = run_loso_fold(
                test_subject=test_subject,
                all_subjects=ALL_SUBJECTS,
                subjects_data=subjects_data,
                common_gestures=common_gestures,
                gesture_to_class=gesture_to_class,
                num_actual_domains=num_actual_domains,
                proc_cfg=proc_cfg,
                device=device,
                output_dir=output_dir,
                logger=logger,
            )
            all_results.append(result)

        except Exception as e:
            logger.error(f"Fold {test_subject} failed: {e}")
            traceback.print_exc()
            all_results.append({
                "test_subject": test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })
            failed_subjects.append(test_subject)

    # ── Aggregate results ─────────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    accs = [r["test_accuracy"] for r in valid]
    f1s  = [r["test_f1_macro"]  for r in valid]

    mean_acc = float(np.mean(accs)) if accs else None
    std_acc  = float(np.std(accs))  if accs else None
    mean_f1  = float(np.mean(f1s))  if f1s  else None
    std_f1   = float(np.std(f1s))   if f1s  else None

    summary = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "subjects": ALL_SUBJECTS,
        "n_domains_requested": num_actual_domains,
        "exercises": EXERCISES,
        "common_gestures": common_gestures,
        "n_valid_folds": len(valid),
        "n_failed_folds": len(failed_subjects),
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_f1_macro": mean_f1,
        "std_f1_macro": std_f1,
        "per_fold": all_results,
    }

    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=2)

    if mean_acc is not None:
        logger.info(
            f"\n{'='*60}\n"
            f"SUMMARY: {EXPERIMENT_NAME}\n"
            f"  Mean Accuracy : {mean_acc:.4f} ± {std_acc:.4f}\n"
            f"  Mean F1-macro : {mean_f1:.4f} ± {std_f1:.4f}\n"
            f"  Valid folds   : {len(valid)} / {len(ALL_SUBJECTS)}\n"
            f"  Results saved : {summary_path}\n"
            f"{'='*60}"
        )
    else:
        logger.error("No valid folds completed.")

    # ── Hypothesis executor callback (optional) ────────────────────────────────
    HYPOTHESIS_ID = None   # set to Qdrant UUID when hypothesis is registered

    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        _HE_AVAILABLE = True
    except ImportError:
        _HE_AVAILABLE = False

    if _HE_AVAILABLE and HYPOTHESIS_ID is not None:
        if mean_acc is not None:
            metrics = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro": std_f1,
                "n_valid_folds": len(valid),
                "n_domains": num_actual_domains,
            }
            mark_hypothesis_verified(HYPOTHESIS_ID, metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed(HYPOTHESIS_ID, "All LOSO folds failed")


if __name__ == "__main__":
    main()
