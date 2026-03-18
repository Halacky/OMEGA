"""
Hypothesis 4: Content-Style Disentanglement for Cross-Subject EMG (LOSO)

Goal: Show that explicitly separating gesture content from subject style
in the learned representation improves LOSO generalization.

Foundation (from H2+H3):
  - Fixed Sinc filterbank (K=4) for frequency decomposition
  - Per-band MixStyle for soft style augmentation

Ablation variants:
  1. baseline       — H3 best: Sinc + MixStyle + K-branch CNN (no disentanglement)
  2. adversarial    — Add subject classifier with GRL on content features
  3. contrastive    — Add InfoNCE pulling same-gesture content together across subjects
  4. mi_min         — Add distance correlation to minimize MI(content, style)
  5. full           — adversarial + contrastive + MI minimization

Protocol: E1 only (gestures 1-10, no REST), strict LOSO, 20 subjects.

Usage:
  python experiments/h4_content_style_disentanglement_loso.py --variant all --full
  python experiments/h4_content_style_disentanglement_loso.py --variant full --ci
"""

import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from utils.logging import setup_logging, seed_everything

EXPERIMENT_NAME = "h4_content_style_disentanglement"
SAMPLING_RATE = 2000

# ── Hyperparameters ──────────────────────────────────────────────────
K = 4                    # Number of frequency bands
FEAT_DIM = 64            # Per-band feature dimension
CONTENT_DIM = 48         # Content representation per band
STYLE_DIM = 16           # Style representation per band
HIDDEN_DIM = 128         # Classifier hidden dim
DROPOUT = 0.3
EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15

# Loss weights
ALPHA_ADV = 0.5          # Adversarial (subject classifier) weight
BETA_MI = 0.1            # MI minimization weight
GAMMA_CL = 0.3           # Contrastive loss weight
CONTRASTIVE_TEMP = 0.1   # InfoNCE temperature
ANNEAL_EPOCHS = 10       # Gradually ramp up aux losses

# MixStyle params (from H3 best)
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

VARIANTS = ["baseline", "adversarial", "contrastive", "mi_min", "full"]

# Subject lists
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [
    "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]


# ═════════════════════════════════════════════════════════════════════════
# Frontend: Fixed Sinc Filterbank (from H2/H3)
# ═════════════════════════════════════════════════════════════════════════

class FixedSincFilterbank(nn.Module):
    """K-band fixed Sinc filterbank. Splits EMG into frequency bands."""

    def __init__(self, K: int = 4, T: int = 200, fs: int = 2000):
        super().__init__()
        self.K = K
        self.T = T
        self.fs = fs
        nyq = fs / 2.0
        edges = np.linspace(20.0, nyq, K + 1)
        filters = []
        for k in range(K):
            lo, hi = edges[k] / nyq, edges[k + 1] / nyq
            hi = min(hi, 0.99)
            if lo >= hi:
                lo = max(0.01, hi - 0.05)
            b = sp_signal.firwin(min(T, 101), [lo, hi], pass_zero=False)
            padded = np.zeros(T)
            padded[:len(b)] = b
            filters.append(padded)
        self.register_buffer("filters", torch.tensor(np.array(filters), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_perm = x.permute(0, 2, 1)  # (B, C, T)
        modes = []
        for k in range(self.K):
            filt = self.filters[k].unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            filt_k = filt.expand(C, 1, -1)
            filtered = F.conv1d(x_perm, filt_k, groups=C,
                                padding=self.T // 2)[:, :, :T]
            modes.append(filtered.permute(0, 2, 1))  # (B, T, C)
        return torch.stack(modes, dim=1)  # (B, K, T, C)


# ═════════════════════════════════════════════════════════════════════════
# Style normalization: Per-band MixStyle (from H3 best)
# ═════════════════════════════════════════════════════════════════════════

class PerBandMixStyle(nn.Module):
    """Per-band MixStyle: mix instance-level style statistics between samples."""

    def __init__(self, K, C, p=0.5, alpha=0.1):
        super().__init__()
        self.K = K
        self.p = p
        self.alpha = alpha

    def forward(self, modes):
        if not self.training or np.random.random() > self.p:
            return modes
        B, K, T, C = modes.shape
        out = []
        for k in range(K):
            x_k = modes[:, k]
            mu = x_k.mean(dim=1, keepdim=True)
            sigma = x_k.std(dim=1, keepdim=True) + 1e-6
            x_normed = (x_k - mu) / sigma
            perm = torch.randperm(B, device=x_k.device)
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x_k.device)
            mu_new = lam * mu + (1 - lam) * mu[perm]
            sigma_new = lam * sigma + (1 - lam) * sigma[perm]
            out.append(x_normed * sigma_new + mu_new)
        return torch.stack(out, dim=1)


class NoStyleNorm(nn.Module):
    def __init__(self, K, C):
        super().__init__()
    def forward(self, modes):
        return modes


# ═════════════════════════════════════════════════════════════════════════
# Gradient Reversal Layer
# ═════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lam)


# ═════════════════════════════════════════════════════════════════════════
# H4 Classifier with Content-Style Disentanglement
# ═════════════════════════════════════════════════════════════════════════

class H4Classifier(nn.Module):
    """
    Sinc filterbank -> MixStyle -> K-branch CNN -> content/style heads -> classifier.

    Content branch: used for gesture classification.
    Style branch: used for subject adversarial + MI minimization.
    """

    def __init__(self, frontend, style_norm, K, in_channels, num_classes,
                 num_subjects, content_dim=CONTENT_DIM, style_dim=STYLE_DIM,
                 feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                 use_adversarial=False, use_contrastive=False, use_mi_min=False):
        super().__init__()
        self.frontend = frontend
        self.style_norm = style_norm
        self.K = K
        self.use_adversarial = use_adversarial
        self.use_contrastive = use_contrastive
        self.use_mi_min = use_mi_min

        # Shared per-band encoders
        self.mode_encoders = nn.ModuleList([
            self._make_encoder(in_channels, feat_dim) for _ in range(K)
        ])

        # Content heads (per band -> content features)
        self.content_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, content_dim),
                nn.ReLU(),
            ) for _ in range(K)
        ])

        # Style heads (per band -> style features)
        self.style_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, style_dim),
                nn.ReLU(),
            ) for _ in range(K)
        ])

        # Gesture classifier (on content only)
        total_content = K * content_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_content, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Subject adversarial classifier (on content with GRL)
        if use_adversarial:
            self.grl = GradientReversal(lam=1.0)
            self.subject_classifier = nn.Sequential(
                nn.Linear(total_content, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_subjects),
            )

    @staticmethod
    def _make_encoder(in_channels, feat_dim):
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x, return_all=False):
        modes = self.frontend(x)       # (B, K, T, C)
        modes = self.style_norm(modes)  # (B, K, T, C)

        content_feats = []
        style_feats = []
        for k in range(self.K):
            mode_k = modes[:, k].permute(0, 2, 1)  # (B, C, T)
            enc_k = self.mode_encoders[k](mode_k)   # (B, feat_dim)
            content_feats.append(self.content_heads[k](enc_k))
            style_feats.append(self.style_heads[k](enc_k))

        z_content = torch.cat(content_feats, dim=1)  # (B, K*content_dim)
        z_style = torch.cat(style_feats, dim=1)      # (B, K*style_dim)

        logits = self.classifier(z_content)

        if not return_all:
            return logits

        result = {
            "logits": logits,
            "z_content": z_content,
            "z_style": z_style,
        }

        if self.use_adversarial:
            result["subject_logits"] = self.subject_classifier(self.grl(z_content))

        return result


# ═════════════════════════════════════════════════════════════════════════
# Auxiliary losses
# ═════════════════════════════════════════════════════════════════════════

def distance_correlation(x, y):
    """Compute distance correlation between x and y (batch-wise).
    Lower = more independent. Used for MI minimization.
    """
    n = x.size(0)
    if n < 4:
        return torch.tensor(0.0, device=x.device)

    # Pairwise distances
    a = torch.cdist(x, x, p=2)
    b = torch.cdist(y, y, p=2)

    # Double centering
    a_row = a.mean(dim=1, keepdim=True)
    a_col = a.mean(dim=0, keepdim=True)
    a_mean = a.mean()
    A = a - a_row - a_col + a_mean

    b_row = b.mean(dim=1, keepdim=True)
    b_col = b.mean(dim=0, keepdim=True)
    b_mean = b.mean()
    B = b - b_row - b_col + b_mean

    dcov_xy = (A * B).mean()
    dcov_xx = (A * A).mean()
    dcov_yy = (B * B).mean()

    denom = (dcov_xx * dcov_yy).sqrt() + 1e-8
    return dcov_xy / denom


def infonce_loss(z, labels, temperature=0.1):
    """InfoNCE contrastive loss. Positive pairs = same gesture class."""
    z_norm = F.normalize(z, dim=1)
    sim = torch.mm(z_norm, z_norm.t()) / temperature  # (B, B)

    # Mask: same label = positive (exclude self)
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.t()).float()
    mask_pos.fill_diagonal_(0)

    # For each anchor, compute log-softmax over all pairs (exclude self)
    mask_self = torch.eye(sim.size(0), device=sim.device).bool()
    sim.masked_fill_(mask_self, -1e9)

    # Sum of positives
    num_pos = mask_pos.sum(dim=1).clamp(min=1)

    # Log-sum-exp over all (denominator)
    log_sum_exp = torch.logsumexp(sim, dim=1)

    # Mean of positive similarities
    pos_sim = (sim * mask_pos).sum(dim=1) / num_pos

    loss = (log_sum_exp - pos_sim).mean()
    return loss


# ═════════════════════════════════════════════════════════════════════════
# Build model for variant
# ═════════════════════════════════════════════════════════════════════════

def build_model(variant: str, num_classes: int, num_subjects: int,
                window_size: int, in_channels: int = 12):
    frontend = FixedSincFilterbank(K=K, T=window_size, fs=SAMPLING_RATE)

    # All variants use per-band MixStyle (H3 best), except baseline uses it too
    style_norm = PerBandMixStyle(K, in_channels, p=MIXSTYLE_P, alpha=MIXSTYLE_ALPHA)

    use_adv = variant in ("adversarial", "full")
    use_cl = variant in ("contrastive", "full")
    use_mi = variant in ("mi_min", "full")

    return H4Classifier(
        frontend=frontend, style_norm=style_norm,
        K=K, in_channels=in_channels, num_classes=num_classes,
        num_subjects=num_subjects,
        content_dim=CONTENT_DIM, style_dim=STYLE_DIM,
        feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        use_adversarial=use_adv, use_contrastive=use_cl, use_mi_min=use_mi,
    )


# ═════════════════════════════════════════════════════════════════════════
# LOSO infrastructure (reused from H3)
# ═════════════════════════════════════════════════════════════════════════

def grouped_to_arrays(grouped_windows):
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for arr in grouped_windows[gid]:
            windows_list.append(arr)
            labels_list.append(np.full(arr.shape[0], gid))
    if not windows_list:
        return np.array([]), np.array([])
    return np.concatenate(windows_list), np.concatenate(labels_list)


def build_loso_splits(subjects_data, train_subjects, test_subject,
                      common_gestures, val_ratio=0.15, seed=42):
    gesture_to_class = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    rng = np.random.RandomState(seed)

    train_wins, train_labels, train_subj_ids = [], [], []
    subject_to_idx = {s: i for i, s in enumerate(sorted(train_subjects))}

    for si, subj in enumerate(sorted(train_subjects)):
        _, _, gw = subjects_data[subj]
        for gid in sorted(common_gestures):
            if gid not in gw:
                continue
            for arr in gw[gid]:
                train_wins.append(arr)
                train_labels.append(np.full(arr.shape[0], gesture_to_class[gid]))
                train_subj_ids.append(np.full(arr.shape[0], subject_to_idx[subj]))

    X_all = np.concatenate(train_wins)
    y_all = np.concatenate(train_labels)
    s_all = np.concatenate(train_subj_ids)

    # Split train/val
    n = len(X_all)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train, s_train = X_all[train_idx], y_all[train_idx], s_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # Test
    _, _, gw_test = subjects_data[test_subject]
    test_wins, test_labels = [], []
    for gid in sorted(common_gestures):
        if gid not in gw_test:
            continue
        for arr in gw_test[gid]:
            test_wins.append(arr)
            test_labels.append(np.full(arr.shape[0], gesture_to_class[gid]))

    X_test = np.concatenate(test_wins)
    y_test = np.concatenate(test_labels)

    return (X_train, y_train, s_train, X_val, y_val, X_test, y_test)


# ═════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════

def train_fold(model, variant, X_train, y_train, s_train, X_val, y_val,
               num_classes, device, batch_size=BATCH_SIZE, epochs=EPOCHS,
               patience=PATIENCE, logger=None):
    """Train one LOSO fold with disentanglement losses."""

    use_adv = variant in ("adversarial", "full")
    use_cl = variant in ("contrastive", "full")
    use_mi = variant in ("mi_min", "full")
    needs_aux = use_adv or use_cl or use_mi

    # Standardize channels
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean_c) / std_c
    X_val = (X_val - mean_c) / std_c

    # Tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    s_tr_t = torch.tensor(s_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t, s_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Anneal auxiliary loss weights
        anneal = min(1.0, epoch / max(1, ANNEAL_EPOCHS))

        for xb, yb, sb in train_loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)

            if needs_aux:
                out = model(xb, return_all=True)
                logits = out["logits"]
                z_content = out["z_content"]
                z_style = out["z_style"]
            else:
                logits = model(xb)

            # Primary gesture loss
            loss = ce_loss_fn(logits, yb)

            # Adversarial: subject classifier on content with GRL
            if use_adv:
                subj_logits = out["subject_logits"]
                loss_adv = ce_loss_fn(subj_logits, sb)
                loss = loss + anneal * ALPHA_ADV * loss_adv

            # Contrastive: InfoNCE on content features
            if use_cl:
                loss_cl = infonce_loss(z_content, yb, temperature=CONTRASTIVE_TEMP)
                loss = loss + anneal * GAMMA_CL * loss_cl

            # MI minimization: distance correlation between content and style
            if use_mi:
                loss_mi = distance_correlation(z_content, z_style)
                loss = loss + anneal * BETA_MI * loss_mi

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = ce_loss_fn(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)

        scheduler.step(val_loss)

        if logger and (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"    Ep {epoch+1:3d}/{epochs} | train={avg_train_loss:.4f} "
                        f"val={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if logger:
                    logger.info(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, mean_c, std_c


def evaluate(model, X_test, y_test, mean_c, std_c, device):
    """Evaluate model on test set."""
    X_test = (X_test - mean_c) / std_c
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    return acc, f1


# ═════════════════════════════════════════════════════════════════════════
# Run one variant
# ═════════════════════════════════════════════════════════════════════════

def run_variant(variant, subjects, base_dir, output_dir, logger,
                batch_size=BATCH_SIZE, num_workers=4):
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# Variant: {variant}")
    logger.info(f"# Subjects: {len(subjects)}")
    logger.info(f"{'#' * 70}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load all subjects
    proc_cfg = ProcessingConfig(
        window_size=200, window_overlap=100, sampling_rate=SAMPLING_RATE,
        segment_edge_margin=0.1)
    multi_loader = MultiSubjectLoader(proc_cfg, logger, use_gpu=False,
                                       use_improved_processing=True)
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir, subject_ids=subjects,
        exercises=["E1"], include_rest=False,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    gesture_to_class = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    num_classes = len(common_gestures)
    num_subjects_train = len(subjects) - 1

    # Window size from first subject
    first_subj = list(subjects_data.keys())[0]
    _, _, gw0 = subjects_data[first_subj]
    first_gid = sorted(gw0.keys())[0]
    window_size = gw0[first_gid][0].shape[1]
    in_channels = gw0[first_gid][0].shape[2]

    logger.info(f"  Classes: {num_classes}, Window: {window_size}, Channels: {in_channels}")

    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()

    for fold_i, test_subject in enumerate(sorted(subjects)):
        logger.info(f"\n  Fold: test={test_subject}, train={len(subjects)-1} subjects")
        seed_everything(42 + fold_i)

        train_subjects = [s for s in sorted(subjects) if s != test_subject]

        X_train, y_train, s_train, X_val, y_val, X_test, y_test = \
            build_loso_splits(subjects_data, train_subjects, test_subject,
                              common_gestures)

        logger.info(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        model = build_model(variant, num_classes, num_subjects_train,
                            window_size, in_channels)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    Model params: {n_params:,}")

        t_fold = time.time()
        model, mean_c, std_c = train_fold(
            model, variant, X_train, y_train, s_train, X_val, y_val,
            num_classes, device, batch_size=batch_size, logger=logger,
        )

        acc, f1 = evaluate(model, X_test, y_test, mean_c, std_c, device)
        fold_time = time.time() - t_fold

        logger.info(f"    -> Acc={acc:.4f}, F1={f1:.4f}, Time={fold_time:.0f}s")

        fold_result = {
            "test_subject": test_subject,
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1),
            "train_time_s": float(fold_time),
        }
        results.append(fold_result)

        # Save fold checkpoint
        fold_dir = variant_dir / f"fold_{test_subject}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "checkpoint.pt")

        # Cleanup
        del model, X_train, y_train, s_train, X_val, y_val, X_test, y_test
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - t0

    # Aggregate
    accs = [r["test_accuracy"] for r in results]
    f1s = [r["test_f1_macro"] for r in results]

    summary = {
        "experiment": EXPERIMENT_NAME,
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "K": K,
        "content_dim": CONTENT_DIM,
        "style_dim": STYLE_DIM,
        "architecture": {
            "frontend": "fixed_sinc_filterbank",
            "style_norm": "per_band_mixstyle",
            "disentanglement": variant,
            "feat_dim": FEAT_DIM,
            "hidden_dim": HIDDEN_DIM,
        },
        "loss_weights": {
            "alpha_adv": ALPHA_ADV if variant in ("adversarial", "full") else 0,
            "beta_mi": BETA_MI if variant in ("mi_min", "full") else 0,
            "gamma_cl": GAMMA_CL if variant in ("contrastive", "full") else 0,
        },
        "results": {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
        },
        "per_subject": results,
        "total_time_s": float(total_time),
    }

    with open(variant_dir / "loso_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  {variant}: F1={np.mean(f1s)*100:.2f}% +/- {np.std(f1s)*100:.2f}% "
                f"({total_time:.0f}s)")
    return summary


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    _parser = argparse.ArgumentParser(description="H4: Content-Style Disentanglement")
    _parser.add_argument("--variant", type=str, default="all")
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--data_dir", type=str, default="data")
    _parser.add_argument("--output_dir", type=str, default=None)
    _parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    _parser.add_argument("--num_workers", type=int, default=0)
    _args, _ = _parser.parse_known_args()

    # Subject selection
    if _args.subjects:
        subjects = _args.subjects.split(",")
    elif _args.full:
        subjects = _FULL_SUBJECTS
    elif _args.ci:
        subjects = _CI_SUBJECTS
    else:
        subjects = _CI_SUBJECTS

    base_dir = ROOT / _args.data_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(_args.output_dir) if _args.output_dir else \
        ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"H4 Content-Style Disentanglement Ablation")
    logger.info(f"  Subjects: {len(subjects)} — {subjects}")
    logger.info(f"  Output:   {output_dir}")
    logger.info(f"  Batch:    {_args.batch_size}, Workers: {_args.num_workers}")

    # Variants
    if _args.variant == "all":
        variants_to_run = VARIANTS
    else:
        variants_to_run = [v.strip() for v in _args.variant.split(",")]
        for v in variants_to_run:
            if v not in VARIANTS:
                logger.error(f"Unknown variant: {v}. Choose from {VARIANTS}")
                sys.exit(1)

    all_summaries = []
    for variant in variants_to_run:
        summary = run_variant(variant, subjects, base_dir, output_dir, logger,
                              batch_size=_args.batch_size,
                              num_workers=_args.num_workers)
        all_summaries.append(summary)
        gc.collect()
        torch.cuda.empty_cache()

    # Comparison
    if len(all_summaries) > 1:
        comp = {}
        for s in all_summaries:
            v = s["variant"]
            r = s["results"]
            comp[v] = {
                "mean_f1": r["mean_f1_macro"],
                "std_f1": r["std_f1_macro"],
                "mean_acc": r["mean_accuracy"],
            }
        comp_path = output_dir / "comparison.json"
        with open(comp_path, "w") as f:
            json.dump(comp, f, indent=2)
        logger.info(f"\nComparison saved to {comp_path}")

    logger.info(f"\nH4 Ablation complete!")


if __name__ == "__main__":
    main()
