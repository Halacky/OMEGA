"""
Experiment 42: Multi-Task Self-Supervised Pretraining for LOSO EMG Classification

Hypothesis: Simultaneous pretraining on 3 self-supervised tasks teaches the encoder
to separate gesture-invariant features from subject-specific style:
  1. MAE reconstruction  — learn EMG temporal structure via masked patch prediction
  2. Subject prediction   — auxiliary task + distance-correlation decorrelation
  3. Cross-subject contrastive — pull same-gesture/different-subject windows together

The disentangled representation enables better LOSO cross-subject generalization.

Architecture:
  Pretraining:
    (B, C, T) → PatchEmbed → Positional Enc
    Pass 1 (masked):  mask 40% → Encoder(visible) → Decoder → MSE(masked patches)
    Pass 2 (full):    Encoder(all) → avg pool → subject_head / projection_head
    Loss = λ_mae * L_mae + λ_subj * L_subject + λ_contr * L_contrastive + β * L_dcor

  Fine-tuning:
    (B, C, T) → PatchEmbed → Encoder(pretrained) → CLS token → Linear → logits

Usage:
    python experiments/exp_42_multi_task_ssl_pretrain_loso.py --ci
    python experiments/exp_42_multi_task_ssl_pretrain_loso.py --subjects DB2_s1,DB2_s12
    python experiments/exp_42_multi_task_ssl_pretrain_loso.py --full
"""

import os
import sys
import json
import gc
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.trainer import WindowClassifierTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

from models.multi_task_ssl import (
    MultiTaskSSLForPretraining,
    MultiTaskSSLForClassification,
    CrossSubjectContrastiveLoss,
)
from models.disentangled_cnn_gru import distance_correlation_loss
from models import register_model

register_model("multi_task_ssl", MultiTaskSSLForClassification)

# ============================================================================
# Experiment settings
# ============================================================================

EXPERIMENT_NAME = "exp_42_multi_task_ssl_pretrain"
EXERCISES = ["E1"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10

SSL_CFG = {
    # Shared encoder
    "patch_size": 20,
    "d_model": 128,
    "encoder_depth": 4,
    "encoder_heads": 4,
    # MAE decoder
    "decoder_depth": 2,
    "decoder_heads": 4,
    "decoder_d_model": 64,
    "mask_ratio": 0.4,
    # Loss weights
    "lambda_mae": 1.0,
    "lambda_subject": 0.3,
    "lambda_contrastive": 0.5,
    "beta_decorrelation": 0.1,
    # Pretraining
    "pretrain_epochs": 30,
    "pretrain_lr": 1e-3,
    "pretrain_batch_size": 256,
    # Contrastive
    "contrastive_temperature": 0.07,
    "projection_dim": 128,
    # Fine-tuning
    "finetune_lr": 5e-4,
}


# ============================================================================
# Dataset with subject + gesture labels
# ============================================================================

class MultiTaskSSLDataset(Dataset):
    """Dataset returning (window, subject_id, gesture_id) triples."""

    def __init__(
        self,
        windows: np.ndarray,      # (N, C, T)
        subject_ids: np.ndarray,   # (N,) int
        gesture_ids: np.ndarray,   # (N,) int
    ):
        self.windows = torch.from_numpy(windows).float()
        self.subject_ids = torch.from_numpy(subject_ids).long()
        self.gesture_ids = torch.from_numpy(gesture_ids).long()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.subject_ids[idx], self.gesture_ids[idx]


# ============================================================================
# Build splits with subject provenance
# ============================================================================

def _build_splits_with_subject_labels(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Build train/val/test splits with subject provenance tracking.

    Returns dict with:
        "train", "val", "test": Dict[int, np.ndarray]  (gesture_id → (N, T, C) windows)
        "train_subject_labels": Dict[int, np.ndarray]   (gesture_id → subject indices)
        "val_subject_labels": Dict[int, np.ndarray]
        "num_train_subjects": int
        "train_subject_mapping": Dict[str, int]  (subject_id → index)
    """
    rng = np.random.RandomState(seed)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # Build per-gesture arrays with subject labels
    train_dict: Dict[int, np.ndarray] = {}
    train_subj_labels: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        windows_for_gid = []
        subj_labels_for_gid = []

        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid in filtered:
                for rep_array in filtered[gid]:
                    if isinstance(rep_array, np.ndarray) and len(rep_array) > 0:
                        windows_for_gid.append(rep_array)
                        subj_labels_for_gid.append(
                            np.full(len(rep_array), train_subject_to_idx[sid], dtype=np.int64)
                        )

        if windows_for_gid:
            train_dict[gid] = np.concatenate(windows_for_gid, axis=0)
            train_subj_labels[gid] = np.concatenate(subj_labels_for_gid, axis=0)

    # Split train → train/val per gesture
    final_train: Dict[int, np.ndarray] = {}
    final_val: Dict[int, np.ndarray] = {}
    final_train_subj: Dict[int, np.ndarray] = {}
    final_val_subj: Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if gid not in train_dict:
            continue
        X_gid = train_dict[gid]
        S_gid = train_subj_labels[gid]
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        final_train[gid] = X_gid[train_idx]
        final_val[gid] = X_gid[val_idx]
        final_train_subj[gid] = S_gid[train_idx]
        final_val_subj[gid] = S_gid[val_idx]

    # Build test split from test subject
    test_dict: Dict[int, np.ndarray] = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        if reps:
            valid_reps = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid_reps:
                test_dict[gid] = np.concatenate(valid_reps, axis=0)

    return {
        "train": final_train,
        "val": final_val,
        "test": test_dict,
        "train_subject_labels": final_train_subj,
        "val_subject_labels": final_val_subj,
        "num_train_subjects": num_train_subjects,
        "train_subject_mapping": train_subject_to_idx,
    }


# ============================================================================
# Visualization functions
# ============================================================================

def _safe_import_matplotlib():
    """Import matplotlib with non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_multitask_loss_curves(history: Dict, output_dir: Path):
    """4-panel plot: MAE loss, Subject CE, Contrastive loss, Combined loss."""
    plt = _safe_import_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-Task SSL Pretraining Loss Curves", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["mae_loss"]) + 1)

    # MAE reconstruction loss
    ax = axes[0, 0]
    ax.plot(epochs, history["mae_loss"], "b-", linewidth=2, label="MAE recon")
    ax.set_title("Task 1: MAE Reconstruction (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subject prediction loss
    ax = axes[0, 1]
    ax.plot(epochs, history["subject_loss"], "r-", linewidth=2, label="Subject CE")
    if "decorrelation_loss" in history:
        ax2 = ax.twinx()
        ax2.plot(epochs, history["decorrelation_loss"], "r--", alpha=0.6, label="Decorrelation")
        ax2.set_ylabel("Decorrelation", color="r", alpha=0.6)
        ax2.legend(loc="upper left")
    ax.set_title("Task 2: Subject Prediction + Decorrelation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Contrastive loss
    ax = axes[1, 0]
    ax.plot(epochs, history["contrastive_loss"], "g-", linewidth=2, label="Contrastive")
    ax.set_title("Task 3: Cross-Subject Contrastive (InfoNCE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined loss
    ax = axes[1, 1]
    ax.plot(epochs, history["total_loss"], "k-", linewidth=2.5, label="Total")
    ax.set_title("Combined Multi-Task Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "pretrain_multitask_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mae_reconstruction(
    model: MultiTaskSSLForPretraining,
    X_sample: np.ndarray,  # (N_sample, C, T)
    output_dir: Path,
    device: str = "cpu",
    n_examples: int = 3,
):
    """Show original vs reconstructed EMG for sample windows with masked regions highlighted."""
    plt = _safe_import_matplotlib()

    model.eval()
    n_show = min(n_examples, len(X_sample))
    X_t = torch.from_numpy(X_sample[:n_show]).float().to(device)

    with torch.no_grad():
        result = model.forward_pretrain(
            X_t,
            subject_ids=torch.zeros(n_show, dtype=torch.long, device=device),
            gesture_ids=torch.zeros(n_show, dtype=torch.long, device=device),
        )
        pred_patches = result["pred_patches"].cpu().numpy()   # (n, L, patch_dim)
        mask = result["mask"].cpu().numpy()                    # (n, L)
        target = result["target_patches"].cpu().numpy()        # (n, L, patch_dim)

    C = X_sample.shape[1]
    patch_size = model.patch_size

    fig, axes = plt.subplots(n_show, 2, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("MAE Reconstruction: Original vs Predicted", fontsize=14, fontweight="bold")

    for i in range(n_show):
        # Reconstruct full signal from patches
        orig_signal = target[i].reshape(-1, C, patch_size)   # (L, C, ps)
        orig_signal = orig_signal.transpose(1, 0, 2).reshape(C, -1)  # (C, T)

        recon_signal = pred_patches[i].reshape(-1, C, patch_size)
        recon_signal = recon_signal.transpose(1, 0, 2).reshape(C, -1)

        # Plot channel 0 (representative)
        T_total = orig_signal.shape[1]
        t = np.arange(T_total)

        # Original
        ax = axes[i, 0]
        ax.plot(t, orig_signal[0], "b-", linewidth=0.8, alpha=0.9)
        # Highlight masked patches
        for p_idx in range(len(mask[i])):
            if mask[i, p_idx]:
                start = p_idx * patch_size
                end = start + patch_size
                ax.axvspan(start, end, alpha=0.2, color="red")
        ax.set_title(f"Example {i+1}: Original (red = masked regions)")
        ax.set_ylabel("Amplitude (ch 0)")
        ax.grid(True, alpha=0.2)

        # Reconstructed
        ax = axes[i, 1]
        ax.plot(t, orig_signal[0], "b-", linewidth=0.8, alpha=0.4, label="Original")
        ax.plot(t, recon_signal[0], "r-", linewidth=0.8, alpha=0.9, label="Reconstructed")
        for p_idx in range(len(mask[i])):
            if mask[i, p_idx]:
                start = p_idx * patch_size
                end = start + patch_size
                ax.axvspan(start, end, alpha=0.1, color="red")
        ax.set_title(f"Example {i+1}: Reconstructed")
        ax.set_ylabel("Amplitude (ch 0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "pretrain_mae_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subject_prediction_accuracy(history: Dict, output_dir: Path):
    """Subject prediction accuracy over pretraining epochs."""
    plt = _safe_import_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["subject_accuracy"]) + 1)
    num_subjects = history.get("num_subjects", 5)
    chance_level = 1.0 / num_subjects

    ax.plot(epochs, history["subject_accuracy"], "r-o", linewidth=2, markersize=4,
            label="Subject prediction acc")
    ax.axhline(y=chance_level, color="gray", linestyle="--", linewidth=1.5,
               label=f"Chance level (1/{num_subjects} = {chance_level:.2f})")

    ax.set_title("Subject Prediction Accuracy During Pretraining\n"
                  "(Lower is better → encoder learns subject-invariant features)",
                  fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "pretrain_subject_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contrastive_similarity_heatmap(
    model: MultiTaskSSLForPretraining,
    X: np.ndarray,           # (N, C, T)
    gesture_ids: np.ndarray,  # (N,)
    subject_ids: np.ndarray,  # (N,)
    output_dir: Path,
    device: str = "cpu",
    max_samples: int = 2000,
):
    """Heatmap: avg cosine similarity between (gesture, subject) groups."""
    plt = _safe_import_matplotlib()

    model.eval()
    # Subsample if too large
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, gesture_ids, subject_ids = X[idx], gesture_ids[idx], subject_ids[idx]

    X_t = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        repr_all = model.get_global_repr(X_t).cpu().numpy()

    # Normalize
    norms = np.linalg.norm(repr_all, axis=1, keepdims=True) + 1e-8
    repr_all = repr_all / norms

    unique_gestures = np.unique(gesture_ids)
    unique_subjects = np.unique(subject_ids)
    n_g = len(unique_gestures)
    n_s = len(unique_subjects)

    # Compute per-group mean representations
    group_reprs = {}
    for g in unique_gestures:
        for s in unique_subjects:
            mask = (gesture_ids == g) & (subject_ids == s)
            if mask.sum() > 0:
                group_reprs[(g, s)] = repr_all[mask].mean(axis=0)

    # Build similarity matrix: gesture vs gesture, averaged over subjects
    sim_matrix = np.zeros((n_g, n_g))
    for i, g1 in enumerate(unique_gestures):
        for j, g2 in enumerate(unique_gestures):
            sims = []
            for s1 in unique_subjects:
                for s2 in unique_subjects:
                    if s1 != s2 and (g1, s1) in group_reprs and (g2, s2) in group_reprs:
                        sim = np.dot(group_reprs[(g1, s1)], group_reprs[(g2, s2)])
                        sims.append(sim)
            sim_matrix[i, j] = np.mean(sims) if sims else 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_g))
    ax.set_yticks(range(n_g))
    ax.set_xticklabels([f"G{g}" for g in unique_gestures], fontsize=9)
    ax.set_yticklabels([f"G{g}" for g in unique_gestures], fontsize=9)
    ax.set_title("Cross-Subject Gesture Similarity\n"
                  "(Cosine similarity of representations, averaged across subject pairs)",
                  fontsize=11)
    ax.set_xlabel("Gesture")
    ax.set_ylabel("Gesture")

    # Annotate cells
    for i in range(n_g):
        for j in range(n_g):
            val = sim_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    fig.savefig(output_dir / "pretrain_contrastive_similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_representations(
    model: nn.Module,
    X: np.ndarray,
    gesture_ids: np.ndarray,
    subject_ids: np.ndarray,
    output_dir: Path,
    device: str = "cpu",
    phase: str = "pretrained",
    max_samples: int = 2000,
):
    """Side-by-side t-SNE: colored by gesture (left) and by subject (right)."""
    plt = _safe_import_matplotlib()
    from sklearn.manifold import TSNE

    model.eval()
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, gesture_ids, subject_ids = X[idx], gesture_ids[idx], subject_ids[idx]

    X_t = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        if hasattr(model, "get_global_repr"):
            repr_all = model.get_global_repr(X_t).cpu().numpy()
        else:
            # For classification model: use encoder output before classifier
            B, C, T = X_t.shape
            num_patches = T // model.patch_size
            tokens = model.patch_embed(X_t) + model.encoder_pos_embed[:, :num_patches, :]
            encoded = model.encoder(tokens)
            repr_all = encoded.mean(dim=1).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    coords = tsne.fit_transform(repr_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"t-SNE of Encoder Representations ({phase})", fontsize=14, fontweight="bold")

    # Left: colored by gesture
    unique_g = np.unique(gesture_ids)
    colors_g = plt.cm.tab10(np.linspace(0, 1, len(unique_g)))
    for i, g in enumerate(unique_g):
        mask = gesture_ids == g
        ax1.scatter(coords[mask, 0], coords[mask, 1], c=[colors_g[i]], s=10, alpha=0.6,
                    label=f"G{g}")
    ax1.set_title("Colored by Gesture\n(Tight clusters = good gesture separation)")
    ax1.legend(fontsize=7, markerscale=2, loc="best", ncol=2)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Right: colored by subject
    unique_s = np.unique(subject_ids)
    colors_s = plt.cm.Set1(np.linspace(0, 1, len(unique_s)))
    for i, s in enumerate(unique_s):
        mask = subject_ids == s
        ax2.scatter(coords[mask, 0], coords[mask, 1], c=[colors_s[i]], s=10, alpha=0.6,
                    label=f"S{s}")
    ax2.set_title("Colored by Subject\n(Mixed within clusters = subject-invariant)")
    ax2.legend(fontsize=7, markerscale=2, loc="best")
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_dir / f"tsne_{phase}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_disentanglement_probing(
    model: nn.Module,
    X: np.ndarray,
    gesture_ids: np.ndarray,
    subject_ids: np.ndarray,
    output_dir: Path,
    device: str = "cpu",
    max_samples: int = 3000,
):
    """Linear probing: gesture acc (should be high) vs subject acc (should be low)."""
    plt = _safe_import_matplotlib()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model.eval()
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, gesture_ids, subject_ids = X[idx], gesture_ids[idx], subject_ids[idx]

    X_t = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        if hasattr(model, "get_global_repr"):
            repr_all = model.get_global_repr(X_t).cpu().numpy()
        else:
            B, C, T = X_t.shape
            num_patches = T // model.patch_size
            tokens = model.patch_embed(X_t) + model.encoder_pos_embed[:, :num_patches, :]
            encoded = model.encoder(tokens)
            repr_all = encoded.mean(dim=1).cpu().numpy()

    # Linear probe: gesture
    clf_gesture = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="multinomial")
    gesture_scores = cross_val_score(clf_gesture, repr_all, gesture_ids, cv=3, scoring="accuracy")
    gesture_acc = gesture_scores.mean()

    # Linear probe: subject
    clf_subject = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="multinomial")
    subject_scores = cross_val_score(clf_subject, repr_all, subject_ids, cv=3, scoring="accuracy")
    subject_acc = subject_scores.mean()

    # Chance levels
    n_gestures = len(np.unique(gesture_ids))
    n_subjects = len(np.unique(subject_ids))
    gesture_chance = 1.0 / n_gestures
    subject_chance = 1.0 / n_subjects

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = [0, 1]
    bars = ax.bar(x_pos, [gesture_acc, subject_acc],
                  color=["#2ecc71", "#e74c3c"], width=0.5, edgecolor="black", linewidth=0.8)

    # Chance lines
    ax.plot([-0.3, 0.3], [gesture_chance, gesture_chance], "k--", linewidth=1)
    ax.plot([0.7, 1.3], [subject_chance, subject_chance], "k--", linewidth=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Gesture\n(should be HIGH)", "Subject\n(should be LOW)"], fontsize=11)
    ax.set_ylabel("Linear Probe Accuracy", fontsize=12)
    ax.set_title("Disentanglement Quality: Linear Probing on Encoder Features", fontsize=13)
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, [gesture_acc, subject_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.text(0, gesture_chance + 0.02, f"chance={gesture_chance:.2f}", ha="center",
            fontsize=8, color="gray")
    ax.text(1, subject_chance + 0.02, f"chance={subject_chance:.2f}", ha="center",
            fontsize=8, color="gray")

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "disentanglement_probing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"gesture_probe_acc": float(gesture_acc), "subject_probe_acc": float(subject_acc)}


def plot_finetuning_curves(history: Dict, output_dir: Path):
    """Train/val loss and accuracy curves for fine-tuning phase."""
    plt = _safe_import_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fine-tuning Training Curves", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train loss")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r-", linewidth=2, label="Val loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", linewidth=2, label="Train acc")
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], "r-", linewidth=2, label="Val acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "finetune_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_contribution_breakdown(history: Dict, output_dir: Path):
    """Stacked area chart: relative contribution of each loss term during pretraining."""
    plt = _safe_import_matplotlib()

    epochs = range(1, len(history["mae_loss"]) + 1)
    cfg = history.get("loss_weights", {"mae": 1.0, "subject": 0.3, "contrastive": 0.5, "dcor": 0.1})

    weighted_mae = np.array(history["mae_loss"]) * cfg["mae"]
    weighted_subj = np.array(history["subject_loss"]) * cfg["subject"]
    weighted_contr = np.array(history["contrastive_loss"]) * cfg["contrastive"]
    weighted_dcor = np.array(history.get("decorrelation_loss", [0.0] * len(history["mae_loss"]))) * cfg["dcor"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        epochs,
        weighted_mae, weighted_subj, weighted_contr, weighted_dcor,
        labels=["MAE Recon", "Subject CE", "Contrastive", "Decorrelation"],
        colors=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"],
        alpha=0.8,
    )
    ax.set_title("Loss Contribution Breakdown During Pretraining", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "pretrain_loss_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_representation_geometry(
    model: nn.Module,
    X: np.ndarray,
    gesture_ids: np.ndarray,
    subject_ids: np.ndarray,
    output_dir: Path,
    device: str = "cpu",
    max_samples: int = 2000,
):
    """Combined t-SNE: gesture-colored points with subject-shaped markers."""
    plt = _safe_import_matplotlib()
    from sklearn.manifold import TSNE

    model.eval()
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, gesture_ids, subject_ids = X[idx], gesture_ids[idx], subject_ids[idx]

    X_t = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        if hasattr(model, "get_global_repr"):
            repr_all = model.get_global_repr(X_t).cpu().numpy()
        else:
            B, C, T = X_t.shape
            num_patches = T // model.patch_size
            tokens = model.patch_embed(X_t) + model.encoder_pos_embed[:, :num_patches, :]
            encoded = model.encoder(tokens)
            repr_all = encoded.mean(dim=1).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    coords = tsne.fit_transform(repr_all)

    unique_g = np.unique(gesture_ids)
    unique_s = np.unique(subject_ids)
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "<"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_g)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, g in enumerate(unique_g):
        for j, s in enumerate(unique_s):
            mask = (gesture_ids == g) & (subject_ids == s)
            if mask.sum() > 0:
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[colors[i]], marker=markers[j % len(markers)],
                    s=20, alpha=0.6,
                    label=f"G{g}/S{s}" if j == 0 else None,
                )

    # Create custom legends
    gesture_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i],
                    markersize=8, label=f"Gesture {g}")
        for i, g in enumerate(unique_g)
    ]
    subject_handles = [
        plt.Line2D([0], [0], marker=markers[j % len(markers)], color="gray",
                    markersize=8, label=f"Subject {s}", linestyle="None")
        for j, s in enumerate(unique_s)
    ]

    leg1 = ax.legend(handles=gesture_handles, title="Gestures", loc="upper left",
                     fontsize=7, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=subject_handles, title="Subjects", loc="upper right",
              fontsize=7, title_fontsize=9)

    ax.set_title("Representation Geometry: Gesture (color) x Subject (marker)\n"
                  "Ideal: tight gesture clusters with mixed subject markers",
                  fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_dir / "representation_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pretraining_ablation_metrics(pretrain_metrics: Dict, output_dir: Path):
    """Bar chart summarizing pretraining outcome metrics."""
    plt = _safe_import_matplotlib()

    metrics = {
        "Final MAE\nRecon Loss": pretrain_metrics.get("final_mae_loss", 0),
        "Subject Pred\nAccuracy": pretrain_metrics.get("final_subject_acc", 0),
        "Contrastive\nLoss": pretrain_metrics.get("final_contrastive_loss", 0),
        "Decorrelation\nLoss": pretrain_metrics.get("final_decorrelation_loss", 0),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(metrics))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    bars = ax.bar(x, list(metrics.values()), color=colors, width=0.5,
                  edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=10)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Pretraining Summary Metrics", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, metrics.values()):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "pretrain_summary_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_subject_accuracy(all_results: List[Dict], output_dir: Path):
    """Bar chart with per-subject test accuracy across LOSO folds."""
    plt = _safe_import_matplotlib()

    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if not valid:
        return

    subjects = [r["test_subject"] for r in valid]
    accs = [r["test_accuracy"] for r in valid]
    f1s = [r["test_f1_macro"] for r in valid]

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle("Per-Subject LOSO Results", fontsize=14, fontweight="bold")

    x = np.arange(len(subjects))

    # Accuracy
    bars1 = ax1.bar(x, accs, color="#3498db", width=0.6, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=mean_acc, color="red", linestyle="--", linewidth=2,
                label=f"Mean = {mean_acc:.4f} (std = {std_acc:.4f})")
    ax1.fill_between([-0.5, len(subjects) - 0.5], mean_acc - std_acc, mean_acc + std_acc,
                     alpha=0.15, color="red")
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Test Accuracy per Subject")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # F1
    mean_f1 = np.mean(f1s)
    bars2 = ax2.bar(x, f1s, color="#2ecc71", width=0.6, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=mean_f1, color="red", linestyle="--", linewidth=2,
                label=f"Mean = {mean_f1:.4f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("F1 Macro", fontsize=11)
    ax2.set_title("Test F1 Macro per Subject")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars2, f1s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "loso_per_subject_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Custom Trainer
# ============================================================================

class MultiTaskSSLTrainer(WindowClassifierTrainer):
    """
    Two-phase trainer for Multi-Task Self-Supervised Pretraining.

    Phase 1: Multi-task SSL pretraining (MAE + Subject + Contrastive + Decorrelation)
    Phase 2: Supervised fine-tuning with pretrained encoder
    """

    def __init__(self, ssl_cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.ssl_cfg = ssl_cfg
        self.pretrain_history: Optional[Dict] = None
        self.finetune_history: Optional[Dict] = None
        self.pretrain_metrics: Optional[Dict] = None

    def _pretrain(
        self,
        X_all: np.ndarray,           # (N, C, T) standardized
        subject_ids_all: np.ndarray,  # (N,) int
        gesture_ids_all: np.ndarray,  # (N,) int
        in_channels: int,
        time_steps: int,
        num_subjects: int,
    ) -> MultiTaskSSLForPretraining:
        """Phase 1: Multi-task self-supervised pretraining."""
        device = self.cfg.device
        cfg = self.ssl_cfg

        # Adjust patch_size if needed
        patch_size = cfg["patch_size"]
        if time_steps % patch_size != 0:
            for ps in range(patch_size, 0, -1):
                if time_steps % ps == 0:
                    self.logger.warning(
                        f"T={time_steps} not divisible by patch_size={patch_size}, "
                        f"adjusted to {ps}"
                    )
                    cfg["patch_size"] = ps
                    break

        model = MultiTaskSSLForPretraining(
            in_channels=in_channels,
            time_steps=time_steps,
            patch_size=cfg["patch_size"],
            d_model=cfg["d_model"],
            encoder_depth=cfg["encoder_depth"],
            encoder_heads=cfg["encoder_heads"],
            decoder_depth=cfg["decoder_depth"],
            decoder_heads=cfg["decoder_heads"],
            decoder_d_model=cfg["decoder_d_model"],
            mask_ratio=cfg["mask_ratio"],
            num_subjects=num_subjects,
            projection_dim=cfg["projection_dim"],
            dropout=self.cfg.dropout,
        ).to(device)

        contrastive_criterion = CrossSubjectContrastiveLoss(
            temperature=cfg["contrastive_temperature"]
        )
        subject_criterion = nn.CrossEntropyLoss()

        dataset = MultiTaskSSLDataset(X_all, subject_ids_all, gesture_ids_all)
        loader = DataLoader(
            dataset,
            batch_size=cfg["pretrain_batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        optimizer = optim.AdamW(
            model.parameters(), lr=cfg["pretrain_lr"], weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["pretrain_epochs"]
        )

        # Loss weights
        w_mae = cfg["lambda_mae"]
        w_subj = cfg["lambda_subject"]
        w_contr = cfg["lambda_contrastive"]
        w_dcor = cfg["beta_decorrelation"]

        self.logger.info(
            f"[SSL Pretraining] {cfg['pretrain_epochs']} epochs, N={len(X_all)}, "
            f"patch={cfg['patch_size']}, mask={cfg['mask_ratio']}, "
            f"subjects={num_subjects}, "
            f"weights: mae={w_mae}, subj={w_subj}, contr={w_contr}, dcor={w_dcor}"
        )

        history = {
            "mae_loss": [], "subject_loss": [], "contrastive_loss": [],
            "decorrelation_loss": [], "total_loss": [], "subject_accuracy": [],
            "num_subjects": num_subjects,
            "loss_weights": {"mae": w_mae, "subject": w_subj, "contrastive": w_contr, "dcor": w_dcor},
        }

        model.train()
        for epoch in range(1, cfg["pretrain_epochs"] + 1):
            ep_mae, ep_subj, ep_contr, ep_dcor, ep_total = 0.0, 0.0, 0.0, 0.0, 0.0
            ep_subj_correct, ep_count = 0, 0

            for batch_x, batch_sid, batch_gid in loader:
                batch_x = batch_x.to(device)
                batch_sid = batch_sid.to(device)
                batch_gid = batch_gid.to(device)

                optimizer.zero_grad()

                result = model.forward_pretrain(batch_x, batch_sid, batch_gid)

                # Loss 1: MAE reconstruction
                loss_mae = result["mae_loss"]

                # Loss 2: Subject prediction
                loss_subj = subject_criterion(result["subject_logits"], batch_sid)

                # Loss 3: Cross-subject contrastive
                loss_contr = contrastive_criterion(
                    result["projections"], batch_gid, batch_sid
                )

                # Loss 4: Distance correlation (decorrelation)
                # Decorrelate global repr from subject-specific info
                global_repr = result["global_repr"]
                # Create subject one-hot as proxy for subject features
                subj_onehot = F.one_hot(batch_sid, num_classes=num_subjects).float()
                loss_dcor = distance_correlation_loss(global_repr, subj_onehot)

                # Combined loss
                total_loss = (
                    w_mae * loss_mae
                    + w_subj * loss_subj
                    + w_contr * loss_contr
                    + w_dcor * loss_dcor
                )

                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                bs = len(batch_x)
                ep_mae += loss_mae.item() * bs
                ep_subj += loss_subj.item() * bs
                ep_contr += loss_contr.item() * bs
                ep_dcor += loss_dcor.item() * bs
                ep_total += total_loss.item() * bs

                # Subject accuracy
                pred_subj = result["subject_logits"].argmax(dim=1)
                ep_subj_correct += (pred_subj == batch_sid).sum().item()
                ep_count += bs

            scheduler.step()

            # Record history
            history["mae_loss"].append(ep_mae / ep_count)
            history["subject_loss"].append(ep_subj / ep_count)
            history["contrastive_loss"].append(ep_contr / ep_count)
            history["decorrelation_loss"].append(ep_dcor / ep_count)
            history["total_loss"].append(ep_total / ep_count)
            history["subject_accuracy"].append(ep_subj_correct / ep_count)

            if epoch % 5 == 0 or epoch == 1:
                self.logger.info(
                    f"[SSL] Epoch {epoch}/{cfg['pretrain_epochs']} "
                    f"mae={ep_mae/ep_count:.5f} "
                    f"subj_ce={ep_subj/ep_count:.4f} "
                    f"contr={ep_contr/ep_count:.4f} "
                    f"dcor={ep_dcor/ep_count:.4f} "
                    f"total={ep_total/ep_count:.4f} "
                    f"subj_acc={ep_subj_correct/ep_count:.3f}"
                )

        self.pretrain_history = history
        self.pretrain_metrics = {
            "final_mae_loss": history["mae_loss"][-1],
            "final_subject_acc": history["subject_accuracy"][-1],
            "final_contrastive_loss": history["contrastive_loss"][-1],
            "final_decorrelation_loss": history["decorrelation_loss"][-1],
        }

        return model

    def _finetune(
        self,
        pretrain_model: MultiTaskSSLForPretraining,
        X_train: np.ndarray,  # (N, C, T) standardized
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        in_channels: int,
        time_steps: int,
        num_classes: int,
    ) -> MultiTaskSSLForClassification:
        """Phase 2: Supervised fine-tuning with pretrained encoder."""
        device = self.cfg.device
        cfg = self.ssl_cfg

        finetune_model = MultiTaskSSLForClassification(
            in_channels=in_channels,
            num_classes=num_classes,
            time_steps=time_steps,
            patch_size=cfg["patch_size"],
            d_model=cfg["d_model"],
            encoder_depth=cfg["encoder_depth"],
            encoder_heads=cfg["encoder_heads"],
            dropout=self.cfg.dropout,
        ).to(device)

        finetune_model.load_pretrained_encoder(pretrain_model)
        del pretrain_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(
            f"[Fine-tuning] {self.cfg.epochs} epochs, LR={cfg['finetune_lr']}, "
            f"classes={num_classes}"
        )

        def _make_loader(X, y, shuffle):
            ds = TensorDataset(
                torch.from_numpy(X).float(),
                torch.from_numpy(y).long(),
            )
            return DataLoader(
                ds, batch_size=self.cfg.batch_size,
                shuffle=shuffle, num_workers=0, pin_memory=True,
            )

        dl_train = _make_loader(X_train, y_train, shuffle=True)
        dl_val = _make_loader(X_val, y_val, shuffle=False) if len(X_val) > 0 else None

        # Class weights
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            weights = counts.sum() / (counts + 1e-8)
            weights /= weights.mean()
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(weights).float().to(device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(
            finetune_model.parameters(),
            lr=cfg["finetune_lr"],
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = self.cfg.early_stopping_patience

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.cfg.epochs + 1):
            # Train
            finetune_model.train()
            train_loss, train_correct = 0.0, 0
            for batch_x, batch_y in dl_train:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = finetune_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(finetune_model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
                train_correct += (logits.argmax(1) == batch_y).sum().item()

            train_loss /= len(y_train)
            train_acc = train_correct / len(y_train)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if dl_val is not None:
                finetune_model.eval()
                val_loss, val_correct = 0.0, 0
                with torch.no_grad():
                    for batch_x, batch_y in dl_val:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        logits = finetune_model(batch_x)
                        val_loss += criterion(logits, batch_y).item() * len(batch_x)
                        val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_loss /= len(y_val)
                val_acc = val_correct / len(y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in finetune_model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 5 == 0 or epoch == 1:
                    self.logger.info(
                        f"[FT] Epoch {epoch}/{self.cfg.epochs} "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )

                if patience_counter >= patience:
                    self.logger.info(f"[FT] Early stopping at epoch {epoch}.")
                    break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    self.logger.info(
                        f"[FT] Epoch {epoch}/{self.cfg.epochs} "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                    )

        if best_state is not None:
            finetune_model.load_state_dict(best_state)
            self.logger.info(f"[FT] Restored best val_loss={best_val_loss:.4f}")

        self.finetune_history = history
        return finetune_model

    # ------------------------------------------------------------------
    # Overridden fit()
    # ------------------------------------------------------------------

    def fit(self, splits: Dict) -> Dict:
        """
        Two-phase fit:
          1. Standardize data
          2. Multi-task SSL pretraining on train+val (with subject/gesture labels)
          3. Fine-tune classifier on labeled train, early stopping on val
          4. Generate visualizations
        """
        seed_everything(self.cfg.seed)

        # --- Prepare flat arrays (N, T, C) from splits ---
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # Extract subject labels from splits
        train_subject_labels = splits.get("train_subject_labels", {})
        val_subject_labels = splits.get("val_subject_labels", {})
        num_train_subjects = splits.get("num_train_subjects", 1)

        # Build flat subject_ids arrays matching X_train/X_val
        def _build_subject_array(split_dict, subj_labels_dict, class_ids_local):
            s_list = []
            for gid in class_ids_local:
                if gid in subj_labels_dict:
                    s_list.append(subj_labels_dict[gid])
            if s_list:
                return np.concatenate(s_list, axis=0)
            return np.zeros(0, dtype=np.int64)

        subj_train = _build_subject_array(splits["train"], train_subject_labels, class_ids)
        subj_val = _build_subject_array(splits["val"], val_subject_labels, class_ids)

        # Transpose (N, T, C) → (N, C, T) — PyTorch convention
        def _t(X):
            if isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[1] > X.shape[2]:
                return np.transpose(X, (0, 2, 1))
            return X

        X_train = _t(X_train)
        X_val = _t(X_val) if len(X_val) > 0 else X_val
        X_test = _t(X_test) if len(X_test) > 0 else X_test

        in_channels = X_train.shape[1]
        time_steps = X_train.shape[2]

        # --- Channel standardization (train only) ---
        mean_c, std_c = self._compute_channel_standardization(X_train)
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = time_steps

        X_train_n = self._apply_standardization(X_train, mean_c, std_c)
        X_val_n = self._apply_standardization(X_val, mean_c, std_c) if len(X_val) > 0 else X_val
        X_test_n = self._apply_standardization(X_test, mean_c, std_c) if len(X_test) > 0 else X_test

        # Save normalization stats
        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # --- Phase 1: Multi-task SSL pretraining (train + val, unlabeled gestures) ---
        X_pretrain = (
            np.concatenate([X_train_n, X_val_n], axis=0)
            if len(X_val_n) > 0 else X_train_n
        )
        # Gesture IDs for pretraining (class indices, not raw gesture IDs)
        gesture_pretrain = (
            np.concatenate([y_train, y_val], axis=0)
            if len(y_val) > 0 else y_train
        )
        # Subject IDs for pretraining
        subject_pretrain = (
            np.concatenate([subj_train, subj_val], axis=0)
            if len(subj_val) > 0 else subj_train
        )

        pretrain_model = self._pretrain(
            X_pretrain, subject_pretrain, gesture_pretrain,
            in_channels, time_steps, num_train_subjects,
        )

        # Save pretrained weights
        pretrain_path = self.output_dir / "ssl_pretrained.pt"
        torch.save(pretrain_model.state_dict(), pretrain_path)
        self.logger.info(f"[SSL] Pretrained weights saved: {pretrain_path}")

        # --- Pretraining visualizations ---
        try:
            plot_multitask_loss_curves(self.pretrain_history, self.output_dir)
            plot_loss_contribution_breakdown(self.pretrain_history, self.output_dir)
            plot_subject_prediction_accuracy(self.pretrain_history, self.output_dir)
            plot_pretraining_ablation_metrics(self.pretrain_metrics, self.output_dir)

            # MAE reconstruction visualization
            sample_idx = np.random.choice(len(X_pretrain), min(5, len(X_pretrain)), replace=False)
            plot_mae_reconstruction(
                pretrain_model, X_pretrain[sample_idx],
                self.output_dir, device=self.cfg.device,
            )

            # t-SNE and similarity (use subset for speed)
            viz_n = min(2000, len(X_pretrain))
            viz_idx = np.random.choice(len(X_pretrain), viz_n, replace=False)
            plot_tsne_representations(
                pretrain_model, X_pretrain[viz_idx],
                gesture_pretrain[viz_idx], subject_pretrain[viz_idx],
                self.output_dir, device=self.cfg.device, phase="pretrained",
            )
            plot_contrastive_similarity_heatmap(
                pretrain_model, X_pretrain[viz_idx],
                gesture_pretrain[viz_idx], subject_pretrain[viz_idx],
                self.output_dir, device=self.cfg.device,
            )
            plot_representation_geometry(
                pretrain_model, X_pretrain[viz_idx],
                gesture_pretrain[viz_idx], subject_pretrain[viz_idx],
                self.output_dir, device=self.cfg.device,
            )

            # Disentanglement probing
            probe_results = plot_disentanglement_probing(
                pretrain_model, X_pretrain[viz_idx],
                gesture_pretrain[viz_idx], subject_pretrain[viz_idx],
                self.output_dir, device=self.cfg.device,
            )
            if self.pretrain_metrics is not None:
                self.pretrain_metrics.update(probe_results)

        except Exception as e:
            self.logger.warning(f"Visualization error (non-fatal): {e}")

        # --- Phase 2: Fine-tuning ---
        num_classes = len(class_ids)
        finetune_model = self._finetune(
            pretrain_model, X_train_n, y_train, X_val_n, y_val,
            in_channels, time_steps, num_classes,
        )

        self.model = finetune_model

        # Fine-tuning visualization
        try:
            plot_finetuning_curves(self.finetune_history, self.output_dir)

            # t-SNE after fine-tuning
            plot_tsne_representations(
                finetune_model, X_pretrain[viz_idx],
                gesture_pretrain[viz_idx], subject_pretrain[viz_idx],
                self.output_dir, device=self.cfg.device, phase="finetuned",
            )
        except Exception as e:
            self.logger.warning(f"Fine-tuning visualization error (non-fatal): {e}")

        # Save SSL config
        with open(self.output_dir / "ssl_config.json", "w") as f:
            json.dump(self.ssl_cfg, f, indent=4)

        return {}


# ============================================================================
# LOSO fold runner
# ============================================================================

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    ssl_cfg: dict,
) -> Dict:
    """Single LOSO fold with multi-task SSL pretraining."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = "deep_raw"
    train_cfg.model_type = "multi_task_ssl"

    # Save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "ssl_config.json", "w") as f:
        json.dump(ssl_cfg, f, indent=4)

    # Data loader
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )

    base_viz = Visualizer(output_dir, logger)

    # Load all subjects
    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES
    )
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    # Build splits with subject labels
    splits = _build_splits_with_subject_labels(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    # Log split sizes
    for split_name in ["train", "val", "test"]:
        total = sum(
            len(arr) for arr in splits[split_name].values()
            if isinstance(arr, np.ndarray) and arr.ndim >= 1
        )
        logger.info(f"Split '{split_name}': {total} windows, {len(splits[split_name])} gestures")

    # Trainer
    trainer = MultiTaskSSLTrainer(
        ssl_cfg=dict(ssl_cfg),
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )

    try:
        trainer.fit(splits)
    except Exception as e:
        logger.error(f"Error in fit(): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # Evaluate on test set
    try:
        # Build flat test arrays
        test_windows, test_labels = [], []
        for i, gid in enumerate(trainer.class_ids):
            if gid in splits["test"]:
                arr = splits["test"][gid]
                test_windows.append(arr)
                test_labels.append(np.full(len(arr), i, dtype=np.int64))

        if test_windows:
            X_test = np.concatenate(test_windows, axis=0).astype(np.float32)
            y_test = np.concatenate(test_labels, axis=0)

            test_results = trainer.evaluate_numpy(
                X_test, y_test, split_name="test", visualize=True
            )
            test_acc = float(test_results.get("accuracy", 0.0))
            test_f1 = float(test_results.get("f1_macro", 0.0))
        else:
            logger.warning("No test data available!")
            test_acc, test_f1 = 0.0, 0.0
    except Exception as e:
        logger.error(f"Error in evaluate: {e}")
        traceback.print_exc()
        test_acc, test_f1 = None, None

    acc_str = f"{test_acc:.4f}" if test_acc is not None else "None"
    f1_str = f"{test_f1:.4f}" if test_f1 is not None else "None"
    logger.info(f"[LOSO] Test={test_subject} | Acc={acc_str}, F1={f1_str}")
    print(f"[LOSO] Test={test_subject} | Acc={acc_str}, F1={f1_str}")

    # Save fold results
    fold_result = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "pretrain_metrics": trainer.pretrain_metrics,
    }
    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(make_json_serializable(fold_result), f, indent=4, ensure_ascii=False)

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, subjects_data, base_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    ALL_SUBJECTS = parse_subjects_args()
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(
        f"./experiments_output/{EXPERIMENT_NAME}_loso_"
        + "_".join(s.split("_s")[1] for s in ALL_SUBJECTS)
    )

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
        batch_size=256,
        epochs=50,
        learning_rate=5e-4,
        weight_decay=1e-4,
        dropout=0.1,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
    )

    ssl_cfg = dict(SSL_CFG)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"EXPERIMENT: {EXPERIMENT_NAME}")
    global_logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    global_logger.info(f"Device: {train_cfg.device}")
    global_logger.info(f"SSL config: {ssl_cfg}")
    global_logger.info("=" * 80)

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        print(f"\nLOSO fold: test_subject={test_subject}")
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"

        try:
            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                ssl_cfg=dict(ssl_cfg),
            )
            all_loso_results.append(fold_res)
            acc = fold_res.get("test_accuracy")
            f1 = fold_res.get("test_f1_macro")
            acc_str = f"{acc:.4f}" if acc is not None else "None"
            f1_str = f"{f1:.4f}" if f1 is not None else "None"
            print(f"  Done: acc={acc_str}, f1={f1_str}")
        except Exception as e:
            global_logger.error(f"Failed fold test={test_subject}: {e}")
            global_logger.error(traceback.format_exc())
            all_loso_results.append({
                "test_subject": test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })

    # --- Aggregate ---
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        aggregate = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_folds": len(valid_results),
            "per_subject": valid_results,
        }
        print(
            f"\nAGGREGATE: "
            f"Acc={aggregate['mean_accuracy']:.4f}+/-{aggregate['std_accuracy']:.4f}, "
            f"F1={aggregate['mean_f1_macro']:.4f}+/-{aggregate['std_f1_macro']:.4f} "
            f"(n={aggregate['num_folds']})"
        )
        global_logger.info(
            f"AGGREGATE: "
            f"Acc={aggregate['mean_accuracy']:.4f}+/-{aggregate['std_accuracy']:.4f}, "
            f"F1={aggregate['mean_f1_macro']:.4f}+/-{aggregate['std_f1_macro']:.4f}"
        )
    else:
        aggregate = {"error": "All folds failed"}

    # --- Summary visualizations ---
    try:
        plot_per_subject_accuracy(all_loso_results, OUTPUT_DIR)
    except Exception as e:
        global_logger.warning(f"Summary visualization error: {e}")

    # --- Save summary JSON ---
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "Multi-task SSL pretraining (MAE + Subject Prediction + Cross-Subject Contrastive "
            "+ Decorrelation) learns gesture-invariant features separated from subject style, "
            "improving LOSO cross-subject generalization."
        ),
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "ssl_config": ssl_cfg,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)

    global_logger.info(f"EXPERIMENT COMPLETE. Results: {OUTPUT_DIR.resolve()}")

    # --- Hypothesis executor integration ---
    try:
        from hypothesis_executor.qdrant_callback import (
            mark_hypothesis_verified,
            mark_hypothesis_failed,
        )

        if valid_results:
            best_metrics = {
                "accuracy": aggregate["mean_accuracy"],
                "f1_macro": aggregate["mean_f1_macro"],
                "std_accuracy": aggregate["std_accuracy"],
                "num_folds": aggregate["num_folds"],
            }
            mark_hypothesis_verified(
                hypothesis_id="H42_multi_task_ssl",
                metrics=best_metrics,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                hypothesis_id="H42_multi_task_ssl",
                error_message="All LOSO folds failed",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
