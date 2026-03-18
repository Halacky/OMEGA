#!/usr/bin/env python3
"""
Generate confusion matrix for the best model (H7 variant F: UVMD + MixStyle).

Runs a single LOSO fold, collects predictions, and generates:
1. Aggregated confusion matrix across all folds
2. Per-class precision/recall/F1 table
3. Figure: normalized confusion matrix

Usage:
  python scripts/generate_confusion_matrix.py                    # 5 CI subjects
  python scripts/generate_confusion_matrix.py --full             # 20 subjects
  python scripts/generate_confusion_matrix.py --subjects 1 2 3   # specific subjects
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ConfMatrix")

OUT_DIR = PROJECT_ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import model from H7 ────────────────────────────────────────
# We inline the model definition to avoid import issues on server
# This is a copy from h7_uvmd_mixstyle_loso.py

BATCH_SIZE = 512
NUM_EPOCHS = 80
LR = 1e-3
PATIENCE = 15
K = 4
L_ADMM = 8
LAMBDA_OVERLAP = 0.01

_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [f"DB2_s{i}" for i in list(range(1, 6)) + list(range(11, 16)) +
                  list(range(26, 31)) + list(range(36, 41))]


class UVMDBlock(nn.Module):
    def __init__(self, in_channels: int, K: int = 4, L: int = 8):
        super().__init__()
        self.K, self.L, self.C = K, L, in_channels
        self.alpha = nn.Parameter(torch.ones(L, K) * 10.0)
        self.tau = nn.Parameter(torch.ones(L) * 0.1)
        self.omega = nn.Parameter(torch.linspace(0.05, 0.45, K))

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        freqs = torch.linspace(0, 0.5, x_freq.shape[1], device=x.device)
        modes = []
        residual_freq = x_freq.clone()
        for k in range(self.K):
            mode_freq = torch.zeros_like(x_freq)
            for l_iter in range(self.L):
                alpha_kl = torch.abs(self.alpha[l_iter, k])
                bandwidth = 1.0 / (1.0 + alpha_kl * (freqs - self.omega[k]).pow(2))
                mode_freq = residual_freq * bandwidth.unsqueeze(0).unsqueeze(-1)
            residual_freq = residual_freq - mode_freq
            mode_t = torch.fft.irfft(mode_freq, n=T, dim=1)
            modes.append(mode_t)
        return modes, self.omega


class PerBandMixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.p, self.alpha = p, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[1], keepdim=True)
        sig = (x.var(dim=[1], keepdim=True) + 1e-6).sqrt()
        x_normed = (x - mu) / sig
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2
        return x_normed * sig_mix + mu_mix


class PerBandEncoder(nn.Module):
    def __init__(self, in_ch: int = 12, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, feat_dim, 3, padding=1), nn.BatchNorm1d(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class UVMDMixStyleModel(nn.Module):
    def __init__(self, in_ch=12, K=4, L=8, num_classes=10, feat_dim=128, use_mixstyle=True):
        super().__init__()
        self.uvmd = UVMDBlock(in_ch, K, L)
        self.mixstyle = PerBandMixStyle() if use_mixstyle else None
        self.encoders = nn.ModuleList([PerBandEncoder(in_ch, feat_dim) for _ in range(K)])
        self.classifier = nn.Sequential(
            nn.Linear(K * feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_omega=False):
        modes, omega = self.uvmd(x)
        feats = []
        for k, mode_k in enumerate(modes):
            if self.mixstyle is not None:
                mode_k = self.mixstyle(mode_k)
            feats.append(self.encoders[k](mode_k))
        fused = torch.cat(feats, dim=-1)
        logits = self.classifier(fused)
        if return_omega:
            return logits, omega
        return logits


def spectral_overlap_penalty(omega):
    diffs = omega.unsqueeze(0) - omega.unsqueeze(1)
    penalty = torch.exp(-10.0 * diffs.pow(2))
    mask = ~torch.eye(len(omega), dtype=torch.bool, device=omega.device)
    return penalty[mask].mean()


def grouped_to_arrays(grouped_windows, gesture_ids):
    windows_list, labels_list = [], []
    for gid in gesture_ids:
        if gid in grouped_windows:
            for arr in grouped_windows[gid]:
                windows_list.append(arr)
                labels_list.append(np.full(arr.shape[0], gid))
    if not windows_list:
        return np.array([]), np.array([])
    return np.concatenate(windows_list), np.concatenate(labels_list)


def train_and_predict_fold(subjects_data, test_subj, gesture_ids, gesture_to_class, device, batch_size=512):
    """Train one fold, return (y_true, y_pred) for the test subject."""
    # Prepare data
    train_X, train_y = [], []
    for sid, (emg, seg, gw) in subjects_data.items():
        if sid == test_subj:
            continue
        w, l = grouped_to_arrays(gw, gesture_ids)
        if len(w) > 0:
            train_X.append(w)
            train_y.append(np.array([gesture_to_class[g] for g in l]))

    test_emg, test_seg, test_gw = subjects_data[test_subj]
    test_w, test_l = grouped_to_arrays(test_gw, gesture_ids)
    test_y = np.array([gesture_to_class[g] for g in test_l])

    train_X = np.concatenate(train_X).astype(np.float32)
    train_y = np.concatenate(train_y).astype(np.int64)
    test_X = test_w.astype(np.float32)

    # Standardize
    mean = train_X.mean(axis=(0, 1), keepdims=True)
    std = train_X.std(axis=(0, 1), keepdims=True) + 1e-8
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    # Model
    num_classes = len(gesture_to_class)
    model = UVMDMixStyleModel(in_ch=train_X.shape[2], K=K, L=L_ADMM,
                               num_classes=num_classes, use_mixstyle=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Split train into train/val (last 10%)
    n_val = max(1, int(len(train_X) * 0.1))
    idx = np.random.RandomState(42).permutation(len(train_X))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tr_X, tr_y = train_X[tr_idx], train_y[tr_idx]
    vl_X, vl_y = train_X[val_idx], train_y[val_idx]

    tr_ds = TensorDataset(torch.from_numpy(tr_X), torch.from_numpy(tr_y))
    vl_ds = TensorDataset(torch.from_numpy(vl_X), torch.from_numpy(vl_y))

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, omega = model(xb, return_omega=True)
            loss = criterion(logits, yb) + LAMBDA_OVERLAP * spectral_overlap_penalty(omega)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{NUM_EPOCHS}  val_loss={val_loss:.4f}  patience={patience_counter}/{PATIENCE}")

    # Test predictions
    model.load_state_dict(best_state)
    model.eval()
    test_tensor = torch.from_numpy(test_X).to(device)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i+batch_size]
            logits = model(batch)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())

    y_pred = np.concatenate(all_preds)
    return test_y, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--subjects", nargs="+", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    args, _ = parser.parse_known_args()

    if args.subjects:
        subjects = [f"DB2_s{s}" if not s.startswith("DB2") else s for s in args.subjects]
    elif args.full:
        subjects = _FULL_SUBJECTS
    else:
        subjects = _CI_SUBJECTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, Subjects: {len(subjects)}, Batch size: {args.batch_size}")

    # Load data
    proc_cfg = ProcessingConfig()
    loader = MultiSubjectLoader(processing_config=proc_cfg, logger=logger)
    data_dir = PROJECT_ROOT / "data"

    subjects_data = {}
    for sid in subjects:
        emg, seg, gw = loader.load_subject(base_dir=data_dir, subject_id=sid, exercise="E1", include_rest=False)
        subjects_data[sid] = (emg, seg, gw)

    # Use gestures 8-17 (E1, 10 grasping gestures) — matching H6/H7
    gesture_ids = list(range(8, 18))
    gesture_to_class = {g: i for i, g in enumerate(gesture_ids)}
    class_names = [f"G{g}" for g in gesture_ids]

    # Filter grouped_windows to only these gestures
    for sid in list(subjects_data.keys()):
        emg, seg, gw = subjects_data[sid]
        gw_filtered = {g: gw[g] for g in gesture_ids if g in gw}
        subjects_data[sid] = (emg, seg, gw_filtered)

    logger.info(f"Gestures: {gesture_ids}, Classes: {len(gesture_ids)}")

    # LOSO: collect all predictions
    all_y_true, all_y_pred = [], []
    per_subject_cm = {}

    for test_subj in subjects:
        logger.info(f"Fold: test={test_subj}")
        y_true, y_pred = train_and_predict_fold(
            subjects_data, test_subj, gesture_ids, gesture_to_class, device, args.batch_size
        )
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(gesture_ids))))
        per_subject_cm[test_subj] = cm.tolist()
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average="macro")
        logger.info(f"  {test_subj}: F1={f1:.4f}")

    # Aggregate
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    cm_total = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(gesture_ids))))
    report = classification_report(y_true_all, y_pred_all, target_names=class_names, output_dict=True)

    # Print
    print("\n" + "=" * 70)
    print("AGGREGATED CONFUSION MATRIX (all folds)")
    print("=" * 70)
    print(classification_report(y_true_all, y_pred_all, target_names=class_names))

    # Save data
    result = {
        "model": "UVMD+MixStyle (H7 variant F)",
        "n_subjects": len(subjects),
        "subjects": subjects,
        "gesture_ids": gesture_ids,
        "class_names": class_names,
        "confusion_matrix": cm_total.tolist(),
        "classification_report": report,
        "per_subject_cm": per_subject_cm,
    }
    out_json = OUT_DIR / "confusion_matrix_data.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Data saved to {out_json}")

    # ── Figure: Normalized Confusion Matrix ──────────────────────
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})

    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    n_classes = len(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalized Confusion Matrix — UVMD+MixStyle\n"
                 f"(NinaPro DB2 E1, LOSO, {len(subjects)} subjects, "
                 f"F1={report['macro avg']['f1-score']:.1%})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"fig_confusion_matrix_best.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved to {OUT_DIR / 'fig_confusion_matrix_best.pdf'}")


if __name__ == "__main__":
    main()
