"""
Hypothesis 5: Integrated System — Optimizing the Best Architecture (LOSO)

Goal: Push the best architecture from H1-H4 further by testing key design choices:
  - Number of frequency bands K (4, 6, 8)
  - Channel attention (SE-block) on content features
  - GroupDRO (worst-case subject loss) instead of standard CE

Foundation: H4 baseline = Sinc FB (K=4) + per-band MixStyle + content/style heads → F1=39.19%

Ablation variants:
  1. h4_best     — H4 baseline reproduced (K=4, MixStyle, CE loss)
  2. k6          — K=6 frequency bands (finer decomposition)
  3. k8          — K=8 frequency bands
  4. se_attn     — K=4 + Squeeze-and-Excitation on content features
  5. groupdro    — K=4 + GroupDRO loss (worst-case across train subjects)
  6. best_combo  — Best K + SE + GroupDRO combined

Protocol: E1 only (gestures 1-10, no REST), strict LOSO, 20 subjects.

Usage:
  python experiments/h5_integrated_system_loso.py --variant all --full
  python experiments/h5_integrated_system_loso.py --variant all --ci
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

EXPERIMENT_NAME = "h5_integrated_system"
SAMPLING_RATE = 2000

# ── Hyperparameters ──────────────────────────────────────────────────
FEAT_DIM = 64
CONTENT_DIM = 48
STYLE_DIM = 16
HIDDEN_DIM = 128
DROPOUT = 0.3
EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15

# MixStyle
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

# GroupDRO
DRO_ETA = 0.01

VARIANTS = ["h4_best", "k6", "k8", "se_attn", "groupdro", "best_combo"]

# Variant configs
VARIANT_CONFIG = {
    "h4_best":    {"K": 4, "se": False, "dro": False},
    "k6":         {"K": 6, "se": False, "dro": False},
    "k8":         {"K": 8, "se": False, "dro": False},
    "se_attn":    {"K": 4, "se": True,  "dro": False},
    "groupdro":   {"K": 4, "se": False, "dro": True},
    "best_combo": {"K": 6, "se": True,  "dro": True},
}

_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [
    "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]


# ═════════════════════════════════════════════════════════════════════════
# Fixed Sinc Filterbank
# ═════════════════════════════════════════════════════════════════════════

class FixedSincFilterbank(nn.Module):
    def __init__(self, K: int = 4, T: int = 200, fs: int = 2000):
        super().__init__()
        self.K = K
        self.T = T
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

    def forward(self, x):
        B, T, C = x.shape
        x_perm = x.permute(0, 2, 1)
        modes = []
        for k in range(self.K):
            filt = self.filters[k].unsqueeze(0).unsqueeze(0).expand(C, 1, -1)
            filtered = F.conv1d(x_perm, filt, groups=C, padding=self.T // 2)[:, :, :T]
            modes.append(filtered.permute(0, 2, 1))
        return torch.stack(modes, dim=1)  # (B, K, T, C)


# ═════════════════════════════════════════════════════════════════════════
# Per-band MixStyle
# ═════════════════════════════════════════════════════════════════════════

class PerBandMixStyle(nn.Module):
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


# ═════════════════════════════════════════════════════════════════════════
# Squeeze-and-Excitation block
# ═════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Channel-wise attention (SE-Net) applied to content features from K bands."""
    def __init__(self, in_dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


# ═════════════════════════════════════════════════════════════════════════
# H5 Classifier
# ═════════════════════════════════════════════════════════════════════════

class H5Classifier(nn.Module):
    def __init__(self, frontend, style_norm, K, in_channels, num_classes,
                 content_dim=CONTENT_DIM, style_dim=STYLE_DIM,
                 feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                 use_se=False):
        super().__init__()
        self.frontend = frontend
        self.style_norm = style_norm
        self.K = K
        self.use_se = use_se

        self.mode_encoders = nn.ModuleList([
            self._make_encoder(in_channels, feat_dim) for _ in range(K)
        ])
        self.content_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, content_dim), nn.ReLU())
            for _ in range(K)
        ])
        self.style_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, style_dim), nn.ReLU())
            for _ in range(K)
        ])

        total_content = K * content_dim
        if use_se:
            self.se = SEBlock(total_content, reduction=4)

        self.classifier = nn.Sequential(
            nn.Linear(total_content, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _make_encoder(in_channels, feat_dim):
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )

    def forward(self, x):
        modes = self.frontend(x)
        modes = self.style_norm(modes)

        content_feats = []
        for k in range(self.K):
            mode_k = modes[:, k].permute(0, 2, 1)
            enc_k = self.mode_encoders[k](mode_k)
            content_feats.append(self.content_heads[k](enc_k))

        z_content = torch.cat(content_feats, dim=1)
        if self.use_se:
            z_content = self.se(z_content)
        return self.classifier(z_content)


def build_model(cfg, num_classes, window_size, in_channels=12):
    K = cfg["K"]
    frontend = FixedSincFilterbank(K=K, T=window_size, fs=SAMPLING_RATE)
    style_norm = PerBandMixStyle(K, in_channels, p=MIXSTYLE_P, alpha=MIXSTYLE_ALPHA)
    return H5Classifier(
        frontend=frontend, style_norm=style_norm,
        K=K, in_channels=in_channels, num_classes=num_classes,
        use_se=cfg["se"],
    )


# ═════════════════════════════════════════════════════════════════════════
# Data helpers
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
    subject_to_idx = {s: i for i, s in enumerate(sorted(train_subjects))}

    train_wins, train_labels, train_subj_ids = [], [], []
    for subj in sorted(train_subjects):
        _, _, gw = subjects_data[subj]
        for gid in sorted(common_gestures):
            if gid not in gw: continue
            for arr in gw[gid]:
                train_wins.append(arr)
                train_labels.append(np.full(arr.shape[0], gesture_to_class[gid]))
                train_subj_ids.append(np.full(arr.shape[0], subject_to_idx[subj]))

    X_all = np.concatenate(train_wins)
    y_all = np.concatenate(train_labels)
    s_all = np.concatenate(train_subj_ids)

    n = len(X_all)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train, s_train = X_all[train_idx], y_all[train_idx], s_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    _, _, gw_test = subjects_data[test_subject]
    test_wins, test_labels = [], []
    for gid in sorted(common_gestures):
        if gid not in gw_test: continue
        for arr in gw_test[gid]:
            test_wins.append(arr)
            test_labels.append(np.full(arr.shape[0], gesture_to_class[gid]))

    X_test = np.concatenate(test_wins)
    y_test = np.concatenate(test_labels)
    return (X_train, y_train, s_train, X_val, y_val, X_test, y_test)


# ═════════════════════════════════════════════════════════════════════════
# GroupDRO loss
# ═════════════════════════════════════════════════════════════════════════

class GroupDROLoss(nn.Module):
    """Distributionally Robust Optimization — worst-case over groups (subjects)."""
    def __init__(self, n_groups, eta=DRO_ETA):
        super().__init__()
        self.n_groups = n_groups
        self.eta = eta
        self.register_buffer("group_weights",
                             torch.ones(n_groups) / n_groups)

    def forward(self, logits, labels, group_ids):
        ce = F.cross_entropy(logits, labels, reduction="none")
        group_losses = torch.zeros(self.n_groups, device=logits.device)
        group_counts = torch.zeros(self.n_groups, device=logits.device)

        for g in range(self.n_groups):
            mask = group_ids == g
            if mask.sum() > 0:
                group_losses[g] = ce[mask].mean()
                group_counts[g] = mask.sum()

        # Mirror descent update on group weights
        with torch.no_grad():
            valid = group_counts > 0
            if valid.sum() > 0:
                self.group_weights[valid] *= torch.exp(self.eta * group_losses[valid])
                self.group_weights /= self.group_weights.sum()

        # Weighted loss
        loss = (self.group_weights * group_losses).sum()
        return loss


# ═════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════

def train_fold(model, cfg, X_train, y_train, s_train, X_val, y_val,
               num_classes, num_subjects, device, batch_size=BATCH_SIZE,
               logger=None):
    use_dro = cfg["dro"]

    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean_c) / std_c
    X_val = (X_val - mean_c) / std_c

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
    dro_loss_fn = GroupDROLoss(num_subjects).to(device) if use_dro else None

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for xb, yb, sb in train_loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            logits = model(xb)

            if use_dro:
                loss = dro_loss_fn(logits, yb, sb)
            else:
                loss = ce_loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = ce_loss_fn(val_logits, y_val_t).item()
            val_acc = accuracy_score(y_val, val_logits.argmax(dim=1).cpu().numpy())

        scheduler.step(val_loss)

        if logger and (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"    Ep {epoch+1:3d}/{EPOCHS} | train={avg_train_loss:.4f} "
                        f"val={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                if logger:
                    logger.info(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, mean_c, std_c


def evaluate(model, X_test, y_test, mean_c, std_c, device):
    X_test = (X_test - mean_c) / std_c
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    return accuracy_score(y_test, preds), f1_score(y_test, preds, average="macro")


# ═════════════════════════════════════════════════════════════════════════
# Run one variant
# ═════════════════════════════════════════════════════════════════════════

def run_variant(variant, subjects, base_dir, output_dir, logger,
                batch_size=BATCH_SIZE, subjects_data=None, common_gestures=None):
    cfg = VARIANT_CONFIG[variant]
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# Variant: {variant} | K={cfg['K']} SE={cfg['se']} DRO={cfg['dro']}")
    logger.info(f"# Subjects: {len(subjects)}")
    logger.info(f"{'#' * 70}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    gesture_to_class = {gid: i for i, gid in enumerate(sorted(common_gestures))}
    num_classes = len(common_gestures)
    num_subjects_train = len(subjects) - 1

    first_subj = list(subjects_data.keys())[0]
    _, _, gw0 = subjects_data[first_subj]
    first_gid = sorted(gw0.keys())[0]
    window_size = gw0[first_gid][0].shape[1]
    in_channels = gw0[first_gid][0].shape[2]

    logger.info(f"  Classes: {num_classes}, Window: {window_size}, "
                f"Channels: {in_channels}, K={cfg['K']}")

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

        model = build_model(cfg, num_classes, window_size, in_channels)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    Model params: {n_params:,}")

        t_fold = time.time()
        model, mean_c, std_c = train_fold(
            model, cfg, X_train, y_train, s_train, X_val, y_val,
            num_classes, num_subjects_train, device,
            batch_size=batch_size, logger=logger,
        )
        acc, f1 = evaluate(model, X_test, y_test, mean_c, std_c, device)
        fold_time = time.time() - t_fold

        logger.info(f"    -> Acc={acc:.4f}, F1={f1:.4f}, Time={fold_time:.0f}s")

        results.append({
            "test_subject": test_subject,
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1),
            "train_time_s": float(fold_time),
        })

        fold_dir = variant_dir / f"fold_{test_subject}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "checkpoint.pt")

        del model, X_train, y_train, s_train, X_val, y_val, X_test, y_test
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - t0
    accs = [r["test_accuracy"] for r in results]
    f1s = [r["test_f1_macro"] for r in results]

    summary = {
        "experiment": EXPERIMENT_NAME,
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "config": cfg,
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
    _parser = argparse.ArgumentParser(description="H5: Integrated System")
    _parser.add_argument("--variant", type=str, default="all")
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--data_dir", type=str, default="data")
    _parser.add_argument("--output_dir", type=str, default=None)
    _parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    _parser.add_argument("--num_workers", type=int, default=0)
    _args, _ = _parser.parse_known_args()

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
    logger.info(f"H5 Integrated System Ablation")
    logger.info(f"  Subjects: {len(subjects)} — {subjects}")
    logger.info(f"  Output:   {output_dir}")
    logger.info(f"  Batch:    {_args.batch_size}")

    if _args.variant == "all":
        variants_to_run = VARIANTS
    else:
        variants_to_run = [v.strip() for v in _args.variant.split(",")]

    # Load data once, reuse across variants
    logger.info(f"\nLoading subjects data...")
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
    logger.info(f"  Loaded {len(subjects_data)} subjects, {len(common_gestures)} gestures")

    all_summaries = []
    for variant in variants_to_run:
        if variant not in VARIANT_CONFIG:
            logger.error(f"Unknown variant: {variant}"); continue
        summary = run_variant(variant, subjects, base_dir, output_dir, logger,
                              batch_size=_args.batch_size,
                              subjects_data=subjects_data,
                              common_gestures=common_gestures)
        all_summaries.append(summary)
        gc.collect()
        torch.cuda.empty_cache()

    if len(all_summaries) > 1:
        comp = {s["variant"]: {
            "mean_f1": s["results"]["mean_f1_macro"],
            "std_f1": s["results"]["std_f1_macro"],
            "config": s["config"],
        } for s in all_summaries}
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comp, f, indent=2)

    logger.info(f"\nH5 Ablation complete!")


if __name__ == "__main__":
    main()
