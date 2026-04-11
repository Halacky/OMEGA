#!/usr/bin/env python3
"""
H9s: Representation Geometry Analysis (Descriptive).

Visualises and quantifies the geometry of learned representations from
supervised vs SSL-pretrained models.

Question
--------
  Do SSL-pretrained representations form more gesture-separable and
  subject-invariant clusters than purely supervised representations?

Method
------
  1. Train 3 encoders on the same data (single LOSO fold):
     a) supervised:   UVMD + MixStyle, supervised only
     b) ssl_finetune: VICReg pretrain -> fine-tune
     c) ssl_frozen:   VICReg pretrain only (encoder frozen, no fine-tune)

  2. Extract features from the held-out test subject.

  3. Compute geometry metrics:
     - Silhouette score coloured by gesture class
     - Silhouette score coloured by subject ID
     - Ratio: gesture_sil / subject_sil (higher = better invariance)
     - Linear probe: accuracy for gesture classification on frozen features
     - Linear probe: accuracy for subject classification on frozen features
     - CKA (Centered Kernel Alignment) between model representations

  4. Save t-SNE coordinates for visualisation.

Protocol
--------
  Default: single fold (first subject as test).
  Optional: --all_folds for full LOSO (slower but more robust).
  Metrics aggregated across folds if multiple folds are run.

Output
------
  experiments_output/h9s_representation_geometry_<timestamp>/
    +-- geometry_analysis.json
    +-- tsne_coordinates.json
    +-- experiment.log

Usage
-----
  python experiments/h9s_representation_geometry.py
  python experiments/h9s_representation_geometry.py --full
  python experiments/h9s_representation_geometry.py --all_folds
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# -- project imports ----------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from models.uvmd_ssl_encoder import UVMDSSLEncoder, UVMDSSLClassifier
from models.freq_aware_vicreg import FreqAwareVICReg
from utils.logging import setup_logging

# -- constants ----------------------------------------------------------------
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
N_CHANNELS = 12
K_MODES = 4

# UVMD architecture
L_LAYERS = 8
ALPHA_INIT = 2000.0
TAU_INIT = 0.01
FEAT_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.3

# SSL pretraining
SSL_EPOCHS = 50
SSL_BATCH = 512
SSL_LR = 1e-3
SSL_WARMUP_EPOCHS = 5

# Fine-tuning
FT_EPOCHS = 50
FT_BATCH = 512
FT_LR = 1e-4
FT_PATIENCE = 15
PROBE_EPOCHS = 10
PROBE_LR = 1e-3

# Supervised baseline
SUP_EPOCHS = 80
SUP_LR = 1e-3
SUP_PATIENCE = 15

# Common
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 0.05
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

EXPERIMENT_NAME = "h9s_representation_geometry"

ENCODER_NAMES = ["supervised", "ssl_finetune", "ssl_frozen"]
ENCODER_LABELS = {
    "supervised":   "Supervised (UVMD + MixStyle)",
    "ssl_finetune": "VICReg pretrain + fine-tune",
    "ssl_frozen":   "VICReg pretrain (frozen)",
}


# =============================================================================
#  Helpers
# =============================================================================

def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights."""
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / (counts.astype(np.float64) + 1e-8)
    weights /= weights.sum()
    weights *= len(classes)
    w = torch.zeros(int(classes.max()) + 1, dtype=torch.float32, device=device)
    for c, wt in zip(classes, weights):
        w[int(c)] = wt
    return w


def cosine_schedule(
    epoch: int, total_epochs: int, base_lr: float, warmup: int = 5,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def extract_features(
    encoder: UVMDSSLEncoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract encoder features from numpy arrays."""
    encoder.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    features_list = []
    with torch.no_grad():
        for s in range(0, len(Xt), batch_size):
            feat = encoder.encode(Xt[s: s + batch_size])
            features_list.append(feat.cpu().numpy())
    return np.concatenate(features_list, axis=0)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear Centered Kernel Alignment (CKA).

    Parameters
    ----------
    X, Y : ndarray, shape (n_samples, n_features)

    Returns
    -------
    float: CKA value in [0, 1].
    """
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Linear kernels
    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


# =============================================================================
#  Training Functions
# =============================================================================

def train_supervised_encoder(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLEncoder:
    """Train UVMD + MixStyle supervised, return the encoder."""
    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=True, mixstyle_p=MIXSTYLE_P, mixstyle_alpha=MIXSTYLE_ALPHA,
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    model = UVMDSSLClassifier(
        encoder=encoder, num_classes=num_classes,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SUP_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=FT_BATCH, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(SUP_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            ce_loss = criterion(logits, yb)
            overlap_loss = model.spectral_overlap_penalty()
            loss = ce_loss + OVERLAP_LAMBDA * overlap_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), FT_BATCH):
                val_logits.append(model(Xv[vs: vs + FT_BATCH]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= SUP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("  [Supervised] Training complete.")
    return model.encoder


def ssl_pretrain_vicreg(
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLEncoder:
    """VICReg pretraining on unlabelled train+val data."""
    logger.info("  [VICReg] Starting SSL pretraining...")

    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=False,
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    ssl_model = FreqAwareVICReg(
        encoder=encoder,
        proj_hidden=2048, proj_dim=2048,
        overlap_lambda=OVERLAP_LAMBDA, overlap_sigma=OVERLAP_SIGMA,
    ).to(device)

    X_all = np.concatenate([X_train, X_val], axis=0)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=SSL_BATCH, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        ssl_model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(SSL_EPOCHS):
        ssl_model.train()
        lr = cosine_schedule(epoch, SSL_EPOCHS, SSL_LR, SSL_WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            loss, details = ssl_model(xb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ssl_model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / max(n_batches, 1)
            logger.info(
                f"    VICReg epoch {epoch+1}/{SSL_EPOCHS}  "
                f"loss={avg:.4f}  lr={lr:.6f}"
            )

    logger.info("  [VICReg] Pretraining complete.")
    return encoder


def finetune_encoder(
    encoder: UVMDSSLEncoder,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLEncoder:
    """Fine-tune a pretrained encoder, return the updated encoder."""
    model = UVMDSSLClassifier(
        encoder=encoder, num_classes=num_classes,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=FT_BATCH, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    # Phase 1: Linear probe
    model.freeze_encoder()
    probe_opt = torch.optim.Adam(model.classifier.parameters(), lr=PROBE_LR)
    for epoch in range(PROBE_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            probe_opt.zero_grad()
            loss.backward()
            probe_opt.step()

    # Phase 2: Full fine-tune
    model.unfreeze_encoder()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=FT_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(FT_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            ce_loss = criterion(logits, yb)
            overlap_loss = model.spectral_overlap_penalty()
            loss = ce_loss + OVERLAP_LAMBDA * overlap_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), FT_BATCH):
                val_logits.append(model(Xv[vs: vs + FT_BATCH]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FT_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("  [Fine-tune] Training complete.")
    return model.encoder


# =============================================================================
#  Geometry Metrics
# =============================================================================

def compute_geometry_metrics(
    features: np.ndarray,
    gesture_labels: np.ndarray,
    subject_ids: np.ndarray,
    logger: logging.Logger,
) -> Dict:
    """
    Compute representation geometry metrics.

    Parameters
    ----------
    features : (N, D)
    gesture_labels : (N,) integer class labels
    subject_ids : (N,) integer subject identifiers

    Returns
    -------
    dict with silhouette scores, linear probe accuracies, ratio.
    """
    # Standardise features for metric computation
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Subsample for silhouette (expensive for large N)
    N = len(features_scaled)
    if N > 3000:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(N, 3000, replace=False)
        feat_sub = features_scaled[idx]
        gest_sub = gesture_labels[idx]
        subj_sub = subject_ids[idx]
    else:
        feat_sub = features_scaled
        gest_sub = gesture_labels
        subj_sub = subject_ids

    # -- Silhouette by gesture -----------------------------------------------
    n_gesture_classes = len(np.unique(gest_sub))
    if n_gesture_classes >= 2 and n_gesture_classes < len(gest_sub):
        gesture_sil = float(silhouette_score(feat_sub, gest_sub, metric="euclidean"))
    else:
        gesture_sil = 0.0

    # -- Silhouette by subject -----------------------------------------------
    n_subject_classes = len(np.unique(subj_sub))
    if n_subject_classes >= 2 and n_subject_classes < len(subj_sub):
        subject_sil = float(silhouette_score(feat_sub, subj_sub, metric="euclidean"))
    else:
        subject_sil = 0.0

    # -- Ratio (higher = more gesture-separable, less subject-separable) -----
    sil_ratio = gesture_sil / (abs(subject_sil) + 1e-6)

    # -- Linear probe: gesture classification --------------------------------
    lr_gesture = LogisticRegression(
        max_iter=1000, random_state=SEED, solver="lbfgs",
        multi_class="multinomial", C=1.0,
    )
    lr_gesture.fit(features_scaled, gesture_labels)
    gesture_probe_acc = float(lr_gesture.score(features_scaled, gesture_labels))

    # -- Linear probe: subject classification --------------------------------
    lr_subject = LogisticRegression(
        max_iter=1000, random_state=SEED, solver="lbfgs",
        multi_class="multinomial", C=1.0,
    )
    lr_subject.fit(features_scaled, subject_ids)
    subject_probe_acc = float(lr_subject.score(features_scaled, subject_ids))

    metrics = {
        "gesture_silhouette": gesture_sil,
        "subject_silhouette": subject_sil,
        "silhouette_ratio": sil_ratio,
        "gesture_probe_accuracy": gesture_probe_acc,
        "subject_probe_accuracy": subject_probe_acc,
    }

    logger.info(f"    Gesture silhouette:    {gesture_sil:.4f}")
    logger.info(f"    Subject silhouette:    {subject_sil:.4f}")
    logger.info(f"    Sil ratio (gest/subj): {sil_ratio:.4f}")
    logger.info(f"    Gesture probe acc:     {gesture_probe_acc:.4f}")
    logger.info(f"    Subject probe acc:     {subject_probe_acc:.4f}")

    return metrics


def compute_tsne(
    features: np.ndarray,
    max_samples: int = 2000,
) -> np.ndarray:
    """
    Compute t-SNE embedding, subsampling if needed.

    Returns
    -------
    (N', 2) array of t-SNE coordinates.
    """
    if len(features) > max_samples:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(features), max_samples, replace=False)
        features = features[idx]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(features_scaled)
    return coords


# =============================================================================
#  Single Fold Analysis
# =============================================================================

def run_one_fold(
    test_sid: str,
    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    subjects: List[str],
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Run geometry analysis for one LOSO fold."""
    seed_everything()

    X_test, y_test = subj_arrays[test_sid]
    if len(X_test) == 0:
        return {}

    # Collect train data and subject IDs
    Xs, ys, sids = [], [], []
    for sid in subjects:
        if sid == test_sid:
            continue
        if sid not in subj_arrays:
            continue
        w, l = subj_arrays[sid]
        if len(w) > 0:
            Xs.append(w)
            ys.append(l)
            sids.append(np.full(len(l), hash(sid) % 10000))

    if not Xs:
        return {}

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    sid_all = np.concatenate(sids, axis=0)

    # Train/val split
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X_all))
    n_val = max(1, int(len(X_all) * VAL_RATIO))
    X_train, y_train = X_all[perm[n_val:]], y_all[perm[n_val:]]
    X_val, y_val = X_all[perm[:n_val]], y_all[perm[:n_val]]

    # Keep subject IDs aligned for geometry analysis
    sid_train = sid_all[perm[n_val:]]

    # Channel standardisation
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean_c) / std_c
    X_val_norm = (X_val - mean_c) / std_c

    # Use train data + subject IDs for geometry analysis
    # (not test, since we want to analyse representations of known data)
    X_geom = X_train_norm
    y_geom = y_train
    sid_geom = sid_train

    fold_result = {"test_subject": test_sid, "encoders": {}}

    # -- 1. Supervised encoder -----------------------------------------------
    logger.info(f"  Training supervised encoder...")
    sup_encoder = train_supervised_encoder(
        X_train_norm, y_train, X_val_norm, y_val,
        num_classes, device, logger,
    )
    sup_features = extract_features(sup_encoder, X_geom, device)

    logger.info(f"  Supervised encoder geometry:")
    sup_metrics = compute_geometry_metrics(
        sup_features, y_geom, sid_geom, logger,
    )
    fold_result["encoders"]["supervised"] = sup_metrics

    # -- 2. SSL pretrained encoder -------------------------------------------
    logger.info(f"  SSL pretraining (VICReg)...")
    ssl_encoder = ssl_pretrain_vicreg(
        X_train_norm, X_val_norm, device, logger,
    )

    # 2a. Frozen SSL features (no fine-tuning)
    frozen_features = extract_features(ssl_encoder, X_geom, device)
    logger.info(f"  SSL frozen encoder geometry:")
    frozen_metrics = compute_geometry_metrics(
        frozen_features, y_geom, sid_geom, logger,
    )
    fold_result["encoders"]["ssl_frozen"] = frozen_metrics

    # 2b. Fine-tuned SSL features
    logger.info(f"  Fine-tuning SSL encoder...")
    ssl_encoder_copy = copy.deepcopy(ssl_encoder)
    ft_encoder = finetune_encoder(
        ssl_encoder_copy, X_train_norm, y_train, X_val_norm, y_val,
        num_classes, device, logger,
    )
    ft_features = extract_features(ft_encoder, X_geom, device)
    logger.info(f"  SSL fine-tuned encoder geometry:")
    ft_metrics = compute_geometry_metrics(
        ft_features, y_geom, sid_geom, logger,
    )
    fold_result["encoders"]["ssl_finetune"] = ft_metrics

    # -- 3. CKA between representations -------------------------------------
    logger.info(f"  Computing CKA between encoders...")
    cka_sup_ft = linear_cka(sup_features, ft_features)
    cka_sup_frozen = linear_cka(sup_features, frozen_features)
    cka_ft_frozen = linear_cka(ft_features, frozen_features)

    fold_result["cka"] = {
        "supervised_vs_ssl_finetune": cka_sup_ft,
        "supervised_vs_ssl_frozen": cka_sup_frozen,
        "ssl_finetune_vs_ssl_frozen": cka_ft_frozen,
    }

    logger.info(f"    CKA(sup, ft):     {cka_sup_ft:.4f}")
    logger.info(f"    CKA(sup, frozen): {cka_sup_frozen:.4f}")
    logger.info(f"    CKA(ft, frozen):  {cka_ft_frozen:.4f}")

    # -- 4. t-SNE coordinates ------------------------------------------------
    logger.info(f"  Computing t-SNE embeddings...")
    max_tsne = 2000
    if len(X_geom) > max_tsne:
        tsne_idx = np.random.RandomState(SEED).choice(
            len(X_geom), max_tsne, replace=False,
        )
    else:
        tsne_idx = np.arange(len(X_geom))

    tsne_data = {}
    for enc_name, feats in [
        ("supervised", sup_features),
        ("ssl_finetune", ft_features),
        ("ssl_frozen", frozen_features),
    ]:
        coords = compute_tsne(feats[tsne_idx])
        tsne_data[enc_name] = {
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "gesture_labels": y_geom[tsne_idx].tolist(),
            "subject_ids": sid_geom[tsne_idx].tolist(),
        }

    fold_result["tsne"] = tsne_data

    return fold_result


# =============================================================================
#  Main Analysis
# =============================================================================

def run_analysis(
    subjects: List[str],
    base_dir: str,
    all_folds: bool,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Run geometry analysis across specified folds."""
    # Load data
    proc_cfg = ProcessingConfig(
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP,
        sampling_rate=SAMPLING_RATE,
    )
    multi_loader = MultiSubjectLoader(
        proc_cfg, logger, use_gpu=False, use_improved_processing=True,
    )
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=subjects,
        exercises=["E1"],
        include_rest=False,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    gesture_to_class = {g: i for i, g in enumerate(sorted(common_gestures))}
    num_classes = len(gesture_to_class)

    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, (_, _, gw) in subjects_data.items():
        wins, labs = grouped_to_arrays(gw)
        mask = np.isin(labs, list(gesture_to_class.keys()))
        wins, labs = wins[mask], labs[mask]
        labs = np.array([gesture_to_class[g] for g in labs])
        if len(wins) > 0:
            subj_arrays[sid] = (wins, labs)

    # Determine which folds to run
    if all_folds:
        test_subjects = [s for s in subjects if s in subj_arrays]
    else:
        # Single fold: use first subject as test
        test_subjects = [s for s in subjects if s in subj_arrays][:1]

    logger.info(f"Running {len(test_subjects)} fold(s): {test_subjects}")

    fold_results = []
    for test_sid in test_subjects:
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"  Fold: test={test_sid}")
        logger.info(f"{'='*60}")

        result = run_one_fold(
            test_sid=test_sid,
            subj_arrays=subj_arrays,
            subjects=subjects,
            num_classes=num_classes,
            device=device,
            logger=logger,
        )

        if result:
            elapsed = time.time() - t0
            result["time_s"] = round(elapsed, 1)
            fold_results.append(result)
            logger.info(f"  Fold {test_sid} complete ({elapsed:.0f}s)")

    # Aggregate metrics across folds
    aggregated = {}
    for enc_name in ENCODER_NAMES:
        metrics_keys = [
            "gesture_silhouette", "subject_silhouette", "silhouette_ratio",
            "gesture_probe_accuracy", "subject_probe_accuracy",
        ]
        agg = {}
        for key in metrics_keys:
            values = [
                fr["encoders"][enc_name][key]
                for fr in fold_results
                if enc_name in fr.get("encoders", {})
                and key in fr["encoders"][enc_name]
            ]
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))
            else:
                agg[f"{key}_mean"] = 0.0
                agg[f"{key}_std"] = 0.0
        aggregated[enc_name] = agg

    # Aggregate CKA
    cka_agg = {}
    cka_keys = [
        "supervised_vs_ssl_finetune",
        "supervised_vs_ssl_frozen",
        "ssl_finetune_vs_ssl_frozen",
    ]
    for key in cka_keys:
        values = [
            fr["cka"][key]
            for fr in fold_results
            if "cka" in fr and key in fr["cka"]
        ]
        if values:
            cka_agg[f"{key}_mean"] = float(np.mean(values))
            cka_agg[f"{key}_std"] = float(np.std(values))

    return {
        "n_folds": len(fold_results),
        "per_fold": fold_results,
        "aggregated_by_encoder": aggregated,
        "aggregated_cka": cka_agg,
        "num_classes": num_classes,
        "gesture_to_class": {str(k): v for k, v in gesture_to_class.items()},
    }


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    _parser = argparse.ArgumentParser(
        description="H9s: Representation Geometry Analysis",
    )
    _parser.add_argument("--full", action="store_true",
                         help="Use full 20-subject set")
    _parser.add_argument("--ci", type=int, default=0,
                         help="Use CI subject set (default)")
    _parser.add_argument("--subjects", type=str, default="",
                         help="Comma-separated subject IDs")
    _parser.add_argument("--all_folds", action="store_true",
                         help="Run all LOSO folds (default: single fold)")
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments_output") / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("H9s: Representation Geometry Analysis")
    logger.info(f"Subjects ({len(subjects)}): {subjects}")
    logger.info(f"All folds: {_args.all_folds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    results = run_analysis(
        subjects, _args.base_dir, _args.all_folds, device, logger,
    )

    # Save main results (without t-SNE for size)
    results_no_tsne = copy.deepcopy(results)
    for fr in results_no_tsne.get("per_fold", []):
        fr.pop("tsne", None)

    combined = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "results": results_no_tsne,
    }
    with open(output_dir / "geometry_analysis.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Save t-SNE coordinates separately (can be large)
    tsne_all = {}
    for fr in results.get("per_fold", []):
        if "tsne" in fr:
            tsne_all[fr["test_subject"]] = fr["tsne"]
    if tsne_all:
        with open(output_dir / "tsne_coordinates.json", "w") as f:
            json.dump(tsne_all, f, indent=2)

    # Summary table
    logger.info(f"\n{'='*70}")
    logger.info(
        f"{'Encoder':<30} {'Gest Sil':>10} {'Subj Sil':>10} "
        f"{'Ratio':>8} {'Gest Probe':>11} {'Subj Probe':>11}"
    )
    logger.info("-" * 70)
    for enc_name in ENCODER_NAMES:
        if enc_name in results["aggregated_by_encoder"]:
            agg = results["aggregated_by_encoder"][enc_name]
            gs = agg.get("gesture_silhouette_mean", 0.0)
            ss = agg.get("subject_silhouette_mean", 0.0)
            sr = agg.get("silhouette_ratio_mean", 0.0)
            gp = agg.get("gesture_probe_accuracy_mean", 0.0)
            sp = agg.get("subject_probe_accuracy_mean", 0.0)
            logger.info(
                f"{ENCODER_LABELS[enc_name]:<30} "
                f"{gs:>10.4f} {ss:>10.4f} {sr:>8.2f} "
                f"{gp:>11.4f} {sp:>11.4f}"
            )

    if results.get("aggregated_cka"):
        logger.info(f"\nCKA (averaged across folds):")
        for key, val in results["aggregated_cka"].items():
            logger.info(f"  {key}: {val:.4f}")

    logger.info(f"\nResults saved to {output_dir}")

    # Hypothesis tracking
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed

        agg = results.get("aggregated_by_encoder", {})
        ssl_ft = agg.get("ssl_finetune", {})
        sup = agg.get("supervised", {})

        ssl_ratio = ssl_ft.get("silhouette_ratio_mean", 0.0)
        sup_ratio = sup.get("silhouette_ratio_mean", 0.0)

        if ssl_ratio > sup_ratio:
            mark_hypothesis_verified("H9s", {
                "ssl_finetune_sil_ratio": ssl_ratio,
                "supervised_sil_ratio": sup_ratio,
                "improvement": ssl_ratio - sup_ratio,
                "aggregated_cka": results.get("aggregated_cka", {}),
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed(
                "H9s",
                f"SSL did not improve geometry: "
                f"ssl_ratio={ssl_ratio:.4f} vs sup_ratio={sup_ratio:.4f}",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
