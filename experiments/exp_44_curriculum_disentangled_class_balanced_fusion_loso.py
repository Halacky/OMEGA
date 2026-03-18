"""
Experiment 44: Curriculum + Disentanglement + Class-Balanced Fusion (LOSO)

Hypothesis H_fusion:
    Combining three complementary mechanisms reduces *both* the class-bias
    and the subject-gap that separately limit cross-subject EMG recognition:

    1. CURRICULUM (exp_34 lineage)
       Train on the k nearest subjects first, expand to more distant ones
       gradually.  The model builds a robust initial representation before
       encountering high-variance subjects.

    2. CONTENT-STYLE DISENTANGLEMENT (exp_31 lineage)
       DisentangledCNNGRU separates z_content (gesture) from z_style (subject).
       Only z_content is used at test time → structural subject-invariance.

    3. CLASS-BALANCED OVERSAMPLING + SUBJECT-AWARE MIXUP (new)
       Per curriculum stage: minority gesture classes are oversampled to
       balance the class distribution; then cross-subject MixUp creates
       interpolated signals from pairs sharing the same gesture label but
       coming from different subjects.  This explicitly teaches z_content
       that the same gesture can look very different across subjects.

Rich visualizations:
  Per fold:
    ● training_multi_loss.png        — 4-panel loss curves + active-subject timeline
    ● class_distribution_stages.png — class counts before/after balancing per stage
    ● curriculum_gantt.png          — Gantt chart of subject inclusion schedule
    ● embedding_tsne.png            — t-SNE of z_content / z_style colored by
                                      gesture class AND subject identity
    ● per_class_f1.png              — bar chart of per-gesture F1 on test subject
    ● confusion_matrix_test.png     — normalized confusion matrix (test subject)

  Global (after all folds):
    ● subject_distance_heatmap.png  — pairwise EMG-distance matrix heat-map
    ● loso_bars.png                 — per-fold accuracy & F1 bar chart
    ● distance_vs_accuracy.png      — scatter: nearest-subject dist → test acc
    ● per_class_f1_heatmap.png      — (n_folds × n_gestures) F1 heat-map
    ● loss_components_boxplots.png  — box-plots of final-epoch loss per fold
"""

import os
import sys
import gc
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.curriculum_disentangled_trainer import CurriculumDisentangledTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "exp_44_curriculum_disentangled_class_balanced_fusion"
APPROACH = "deep_raw"
EXERCISES = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES = 10
VISUALIZE_EMBEDDINGS = True      # enable t-SNE visualization (adds ~10-30 s per fold)

# ── Disentanglement ──────────────────────────────────────────────────────────
CONTENT_DIM = 128
STYLE_DIM = 64
ALPHA = 0.5              # subject-classifier loss weight
BETA = 0.1               # MI loss weight (annealed)
BETA_ANNEAL_EPOCHS = 10
MI_LOSS_TYPE = "distance_correlation"

# ── Curriculum ───────────────────────────────────────────────────────────────
K_INIT = 2               # start with 2 nearest subjects (enables MixUp from step 0)
EXPAND_EVERY = 8         # add one subject every 8 epochs
CONSOLIDATION_EPOCHS = 20
LR_ON_EXPAND = None      # keep LR continuous across stages

# ── Class balancing & MixUp ──────────────────────────────────────────────────
USE_CLASS_BALANCED_OVERSAMPLING = True
USE_SUBJECT_MIXUP = True
MIXUP_ALPHA = 0.2        # Beta(α, α) parameter

# ── Subject distance ──────────────────────────────────────────────────────────
DISTANCE_METRIC = "channel_stats"   # "channel_stats" | "mmd_linear"


# ═══════════════════════════════════════════════════════════════════════════
# SUBJECT DISTANCE COMPUTATION  (identical to exp_34)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_subject_channel_stats(grouped_windows: Dict) -> np.ndarray:
    """
    Per-subject EMG fingerprint: per-channel mean, std, RMS, skewness, kurtosis
    + upper-triangle of inter-channel correlation matrix.
    """
    all_windows = []
    for gid in sorted(grouped_windows.keys()):
        for rep in grouped_windows[gid]:
            if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0:
                all_windows.append(rep)
    if not all_windows:
        return np.zeros(1)

    X = np.concatenate(all_windows, axis=0)   # (N, T, C)
    X_flat = X.reshape(-1, X.shape[-1])       # (N*T, C)

    ch_mean = X_flat.mean(axis=0)
    ch_std  = X_flat.std(axis=0)
    ch_rms  = np.sqrt((X_flat ** 2).mean(axis=0))

    eps = 1e-8
    c = X_flat - ch_mean[None, :]
    ch_skew = (c ** 3).mean(axis=0) / (ch_std ** 3 + eps)
    ch_kurt = (c ** 4).mean(axis=0) / (ch_std ** 4 + eps) - 3.0

    corr = np.corrcoef(X_flat.T)
    triu = np.triu_indices(X.shape[-1], k=1)
    ch_corr = np.nan_to_num(corr[triu], nan=0.0)

    return np.concatenate([ch_mean, ch_std, ch_rms, ch_skew, ch_kurt, ch_corr])


def compute_distance_matrix(
    subjects_data: Dict[str, Tuple],
    subject_ids: List[str],
    metric: str = "channel_stats",
) -> np.ndarray:
    """Pairwise EMG distance matrix between subjects."""
    n = len(subject_ids)

    if metric == "channel_stats":
        feats = []
        for sid in subject_ids:
            _, _, gw = subjects_data[sid]
            feats.append(_compute_subject_channel_stats(gw))
        feats = np.array(feats)
        feat_std = feats.std(axis=0) + 1e-8
        feats_n = feats / feat_std[None, :]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(feats_n[i] - feats_n[j])
                D[i, j] = D[j, i] = d
    elif metric == "mmd_linear":
        means = []
        for sid in subject_ids:
            _, _, gw = subjects_data[sid]
            ws = [rep for gid in sorted(gw) for rep in gw[gid]
                  if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0]
            if ws:
                X = np.concatenate(ws, axis=0).reshape(len(ws[0]), -1)
                means.append(np.concatenate(ws, axis=0).reshape(len(np.concatenate(ws, axis=0)), -1).mean(axis=0))
            else:
                means.append(np.zeros(1))
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if means[i].shape == means[j].shape:
                    d = np.linalg.norm(means[i] - means[j])
                else:
                    d = 0.0
                D[i, j] = D[j, i] = d
    else:
        raise ValueError(f"Unknown distance metric: {metric!r}")

    return D


def get_subject_order_by_similarity(
    dist_matrix: np.ndarray,
    subject_ids: List[str],
    test_subject: str,
    train_subjects: List[str],
) -> List[Tuple[str, float]]:
    """Return train subjects sorted by ascending distance to test subject."""
    ti = subject_ids.index(test_subject)
    ranked = [(sid, float(dist_matrix[ti, subject_ids.index(sid)])) for sid in train_subjects]
    ranked.sort(key=lambda x: x[1])
    return ranked


# ═══════════════════════════════════════════════════════════════════════════
# SPLIT BUILDING  (with subject provenance for val as well)
# ═══════════════════════════════════════════════════════════════════════════

def _build_splits_with_subject_labels(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    train_subject_to_idx: Dict[str, int],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Build train/val/test splits with full subject provenance tracking.

    Returns dict with:
        "train":               Dict[gid, (N, T, C) array]
        "val":                 Dict[gid, (N, T, C) array]
        "test":                Dict[gid, (N, T, C) array]
        "train_subject_labels": Dict[gid, (N,) int array]   — subject index per train window
        "val_subject_labels":   Dict[gid, (N,) int array]   — subject index per val window
        "num_train_subjects":  int
    """
    rng = np.random.RandomState(seed)
    num_train_subjects = len(train_subjects)

    train_dict:      Dict[int, np.ndarray] = {}
    train_subj_lbl:  Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        wins_g, slbls_g = [], []
        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, gw = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(gw, [gid])
            if gid not in filtered:
                continue
            sidx = train_subject_to_idx[sid]
            for rep in filtered[gid]:
                if isinstance(rep, np.ndarray) and len(rep) > 0:
                    wins_g.append(rep)
                    slbls_g.append(np.full(len(rep), sidx, dtype=np.int64))

        if wins_g:
            train_dict[gid] = np.concatenate(wins_g, axis=0)
            train_subj_lbl[gid] = np.concatenate(slbls_g, axis=0)

    # Train / val split (same permutation applied to windows and subject labels)
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}
    final_train_s: Dict[int, np.ndarray] = {}
    final_val_s:   Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if gid not in train_dict:
            continue
        X_g = train_dict[gid]
        S_g = train_subj_lbl[gid]
        n = len(X_g)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        vi, ti = perm[:n_val], perm[n_val:]

        final_train[gid] = X_g[ti]
        final_val[gid]   = X_g[vi]
        final_train_s[gid] = S_g[ti]
        final_val_s[gid]   = S_g[vi]

    # Test split from the held-out test subject
    test_dict: Dict[int, np.ndarray] = {}
    _, _, test_gw = subjects_data[test_subject]
    test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
    for gid, reps in test_filtered.items():
        valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
        if valid:
            test_dict[gid] = np.concatenate(valid, axis=0)

    return {
        "train": final_train,
        "val":   final_val,
        "test":  test_dict,
        "train_subject_labels": final_train_s,
        "val_subject_labels":   final_val_s,
        "num_train_subjects":   num_train_subjects,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL HELPER  (grouped_to_arrays — does not exist in any processing module)
# ═══════════════════════════════════════════════════════════════════════════

def grouped_to_arrays(
    grouped_windows: Dict[int, List],
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten grouped_windows → (windows (N,T,C), labels (N,))."""
    X_parts, y_parts = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep in grouped_windows[gid]:
            if isinstance(rep, np.ndarray) and rep.ndim == 3 and len(rep) > 0:
                X_parts.append(rep)
                y_parts.append(np.full(len(rep), gid, dtype=np.int64))
    if not X_parts:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS — PER FOLD
# ═══════════════════════════════════════════════════════════════════════════

def _plot_curriculum_gantt(
    schedule: List[Tuple[int, int, set]],
    subject_ids_sorted: List[str],
    output_dir: Path,
) -> None:
    """Gantt chart: which subjects are active in each curriculum epoch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        n_subjects = len(subject_ids_sorted)
        colors = plt.cm.Set2(np.linspace(0, 1, n_subjects))

        fig, ax = plt.subplots(figsize=(max(10, schedule[-1][1] // 5), max(4, n_subjects * 0.7)))
        for stage_idx, (s_ep, e_ep, allowed) in enumerate(schedule):
            width = e_ep - s_ep + 1
            for i, sid in enumerate(subject_ids_sorted):
                if i in allowed:
                    ax.barh(
                        sid, width, left=s_ep,
                        height=0.6, color=colors[i], alpha=0.8,
                        edgecolor="white", linewidth=0.5,
                    )
            # Stage boundary line
            ax.axvline(x=s_ep, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Training Subject", fontsize=11)
        ax.set_title("Curriculum Schedule — Subject Inclusion Timeline", fontsize=12, fontweight="bold")
        ax.set_xlim(1, schedule[-1][1])
        plt.tight_layout()
        plt.savefig(output_dir / "curriculum_gantt.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] curriculum_gantt failed: {exc}")


def _plot_multi_loss_curves(history: Dict, output_dir: Path) -> None:
    """4-panel training curves: losses, accuracy, beta, active subjects."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Training Curves — Curriculum Disentangled Fusion", fontsize=13, fontweight="bold")

        # Panel 1: All losses
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="Total (train)", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Gesture CE (val)", linewidth=2, linestyle="--")
        ax.plot(epochs, history["gesture_loss"], label="Gesture CE (train)", alpha=0.7)
        ax.plot(epochs, history["subject_loss"], label="Subject CE (train)", alpha=0.7)
        ax.plot(epochs, history["mi_loss"], label="MI loss (train)", alpha=0.7)
        ax.set_ylabel("Loss")
        ax.set_title("Loss Components")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 2: Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, history["train_acc"], label="Train acc", linewidth=2)
        ax.plot(epochs, history["val_acc"], label="Val acc", linewidth=2, linestyle="--")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

        # Panel 3: Beta annealing
        ax = axes[1, 0]
        ax.plot(epochs, history["beta"], color="purple", linewidth=2)
        ax.set_ylabel("β (MI loss weight)")
        ax.set_xlabel("Epoch")
        ax.set_title("MI Loss Weight Annealing")
        ax.grid(alpha=0.3)

        # Panel 4: Active subjects
        ax = axes[1, 1]
        ax.step(epochs, history["num_active_subjects"], color="green", linewidth=2, where="post")
        ax.set_ylabel("# Active Training Subjects")
        ax.set_xlabel("Epoch")
        ax.set_title("Curriculum Expansion")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(history["num_active_subjects"]) + 1)

        # Stage change markers on all panels
        stage_arr = history["stage"]
        for ax in axes.flat:
            for ep in range(1, len(stage_arr)):
                if stage_arr[ep] != stage_arr[ep - 1]:
                    ax.axvline(x=ep + 1, color="gray", linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / "training_multi_loss.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] training_multi_loss failed: {exc}")


def _plot_class_distribution_stages(
    stage_stats_history: List[Dict],
    class_ids: List[int],
    class_names: Dict,
    output_dir: Path,
) -> None:
    """Bar chart showing class sizes before/after oversampling per curriculum stage."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_stages = len(stage_stats_history)
        if n_stages == 0:
            return

        fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5), sharey=True)
        if n_stages == 1:
            axes = [axes]

        gesture_labels = [class_names.get(gid, str(gid)) for gid in class_ids]
        x = np.arange(len(class_ids))

        for i, (ax, stats) in enumerate(zip(axes, stage_stats_history)):
            raw = np.array(stats.get("class_counts_raw", [0] * len(class_ids)))
            bars_raw = ax.bar(x - 0.2, raw, width=0.35, label="Raw", alpha=0.7, color="steelblue")
            if "class_counts_oversampled" in stats:
                ov = np.array(stats["class_counts_oversampled"])
                ax.bar(x + 0.2, ov, width=0.35, label="Oversampled", alpha=0.7, color="tomato")
            n_subj = len(stats.get("subjects", []))
            n_mix = stats.get("n_mixed_pairs", 0)
            ax.set_title(
                f"Stage {stats.get('stage', i)}\n"
                f"({n_subj} subj | epoch {stats.get('epoch_start', '?')} | +{n_mix} MixUp)",
                fontsize=9,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(gesture_labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Window count")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Class Distribution per Curriculum Stage", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution_stages.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] class_distribution_stages failed: {exc}")


def _plot_per_class_f1(
    report: Dict,
    class_ids: List[int],
    class_names: Dict,
    test_subject: str,
    output_dir: Path,
) -> None:
    """Bar chart of per-gesture F1 for the test subject."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f1s = []
        labels = []
        for i, gid in enumerate(class_ids):
            key = str(i)
            f1 = report.get(key, {}).get("f1-score", 0.0)
            f1s.append(f1)
            labels.append(class_names.get(gid, str(gid)))

        x = np.arange(len(f1s))
        fig, ax = plt.subplots(figsize=(max(8, len(f1s)), 4))
        bars = ax.bar(x, f1s, color=[plt.cm.RdYlGn(v) for v in f1s], edgecolor="gray", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("F1-score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-Gesture F1 — Test Subject {test_subject}", fontsize=11, fontweight="bold")
        ax.axhline(y=np.mean(f1s), color="navy", linestyle="--", linewidth=1.5, label=f"Mean F1={np.mean(f1s):.3f}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        # Value labels on bars
        for bar, f1 in zip(bars, f1s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{f1:.2f}", ha="center", va="bottom", fontsize=7,
            )
        plt.tight_layout()
        plt.savefig(output_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] per_class_f1 failed: {exc}")


def _plot_embedding_tsne(
    trainer: CurriculumDisentangledTrainer,
    X_val: np.ndarray,
    y_val_gesture: np.ndarray,
    y_val_subject: np.ndarray,
    class_ids: List[int],
    class_names: Dict,
    sorted_train: List[str],
    test_subject: str,
    output_dir: Path,
) -> None:
    """
    t-SNE of z_content and z_style from validation windows.

    Left pair: z_content colored by gesture class  (should show clear clusters)
    Right pair: z_content colored by subject        (should show NO clustering → subject-invariant)
    Middle: z_style colored by subject              (should show subject-separated clusters)
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if len(X_val) < 10:
            return

        z_content, z_style = trainer.get_content_style_embeddings(X_val)

        n_gesture = len(class_ids)
        n_subject = len(sorted_train)
        gest_cmap = cm.get_cmap("tab10", n_gesture)
        subj_cmap = cm.get_cmap("Set1", n_subject)

        # t-SNE on content
        tsne_c = TSNE(n_components=2, perplexity=min(30, len(z_content) // 4), random_state=42)
        emb_c = tsne_c.fit_transform(z_content)

        # t-SNE on style
        tsne_s = TSNE(n_components=2, perplexity=min(30, len(z_style) // 4), random_state=42)
        emb_s = tsne_s.fit_transform(z_style)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Embedding Space — Test Subject {test_subject} (Val Windows)",
            fontsize=12, fontweight="bold",
        )

        # Panel 1: z_content colored by gesture
        ax = axes[0]
        for i, gid in enumerate(class_ids):
            mask = y_val_gesture == i
            if mask.any():
                ax.scatter(
                    emb_c[mask, 0], emb_c[mask, 1],
                    c=[gest_cmap(i)], label=class_names.get(gid, str(gid)),
                    s=15, alpha=0.7, edgecolors="none",
                )
        ax.set_title("z_content — colored by GESTURE\n(should cluster by gesture)", fontsize=10)
        ax.legend(fontsize=6, loc="best", markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])

        # Panel 2: z_content colored by subject
        ax = axes[1]
        for si, sname in enumerate(sorted_train):
            mask = y_val_subject == si
            if mask.any():
                ax.scatter(
                    emb_c[mask, 0], emb_c[mask, 1],
                    c=[subj_cmap(si)], label=sname.replace("DB2_", ""),
                    s=15, alpha=0.7, edgecolors="none",
                )
        ax.set_title("z_content — colored by SUBJECT\n(should NOT cluster by subject)", fontsize=10)
        ax.legend(fontsize=7, loc="best", markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])

        # Panel 3: z_style colored by subject
        ax = axes[2]
        for si, sname in enumerate(sorted_train):
            mask = y_val_subject == si
            if mask.any():
                ax.scatter(
                    emb_s[mask, 0], emb_s[mask, 1],
                    c=[subj_cmap(si)], label=sname.replace("DB2_", ""),
                    s=15, alpha=0.7, edgecolors="none",
                )
        ax.set_title("z_style — colored by SUBJECT\n(should cluster by subject)", fontsize=10)
        ax.legend(fontsize=7, loc="best", markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(output_dir / "embedding_tsne.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] embedding_tsne failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS — GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

def _plot_subject_distance_heatmap(
    dist_matrix: np.ndarray,
    subject_ids: List[str],
    output_dir: Path,
) -> None:
    """Annotated heat-map of pairwise subject distances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(subject_ids)
        labels = [s.replace("DB2_", "") for s in subject_ids]

        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        im = ax.imshow(dist_matrix, cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="EMG distance")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        # Annotate cells
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{dist_matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if dist_matrix[i, j] < dist_matrix.max() * 0.7 else "white")
        ax.set_title(f"Pairwise Subject EMG Distance ({DISTANCE_METRIC})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "subject_distance_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] subject_distance_heatmap failed: {exc}")


def _plot_loso_bars(all_results: List[Dict], output_dir: Path) -> None:
    """Bar chart of accuracy and F1 per fold."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [r for r in all_results if r.get("test_accuracy") is not None]
        if not valid:
            return

        subjects = [r["test_subject"].replace("DB2_", "") for r in valid]
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"] for r in valid]
        x = np.arange(len(subjects))

        fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.2), 5))
        bars1 = ax.bar(x - 0.2, accs, width=0.35, label="Accuracy", color="steelblue", alpha=0.85)
        bars2 = ax.bar(x + 0.2, f1s,  width=0.35, label="F1-macro",  color="tomato",   alpha=0.85)

        mean_acc = np.mean(accs)
        mean_f1  = np.mean(f1s)
        ax.axhline(mean_acc, color="steelblue", linestyle="--", linewidth=1.5, label=f"Mean acc={mean_acc:.3f}")
        ax.axhline(mean_f1,  color="tomato",    linestyle="--", linewidth=1.5, label=f"Mean F1={mean_f1:.3f}")

        ax.set_xticks(x)
        ax.set_xticklabels(subjects, fontsize=9)
        ax.set_xlabel("Test Subject")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("LOSO Results — Curriculum + Disentanglement + Class-Balanced Fusion",
                     fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(list(bars1) + list(bars2), accs + f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / "loso_bars.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] loso_bars failed: {exc}")


def _plot_distance_vs_accuracy(
    all_results: List[Dict],
    dist_matrix: np.ndarray,
    subject_ids: List[str],
    output_dir: Path,
) -> None:
    """Scatter: minimum train-subject distance to test subject vs test accuracy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [r for r in all_results if r.get("test_accuracy") is not None]
        if not valid:
            return

        min_dists, accs = [], []
        for r in valid:
            ts = r["test_subject"]
            if ts not in subject_ids:
                continue
            ti = subject_ids.index(ts)
            train_indices = [i for i, s in enumerate(subject_ids) if s != ts]
            if not train_indices:
                continue
            min_d = dist_matrix[ti, train_indices].min()
            min_dists.append(min_d)
            accs.append(r["test_accuracy"])

        if not min_dists:
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(min_dists, accs, s=80, c="steelblue", alpha=0.8, edgecolors="navy", linewidth=0.8)

        for r, md, acc in zip(valid, min_dists, accs):
            ax.annotate(r["test_subject"].replace("DB2_", ""),
                        (md, acc), xytext=(5, 5), textcoords="offset points", fontsize=8)

        # Simple trend line
        if len(min_dists) >= 3:
            z = np.polyfit(min_dists, accs, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(min_dists), max(min_dists), 50)
            ax.plot(xs, p(xs), "r--", alpha=0.7, linewidth=1.5, label="Linear trend")
            ax.legend()

        ax.set_xlabel("Min EMG distance to nearest training subject")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Subject Similarity vs Cross-Subject Test Accuracy", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "distance_vs_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] distance_vs_accuracy failed: {exc}")


def _plot_per_class_f1_heatmap(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """Heat-map: (n_folds × n_gestures) of per-gesture F1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [r for r in all_results if r.get("test_report") is not None]
        if not valid:
            return

        # Collect all gesture keys
        all_keys_sets = [set(r["test_report"].keys()) for r in valid]
        # intersection of numeric keys (gesture indices)
        numeric_keys = sorted(
            set.intersection(*all_keys_sets) - {"accuracy", "macro avg", "weighted avg"},
            key=lambda k: int(k) if k.isdigit() else 0,
        )
        if not numeric_keys:
            return

        subjects = [r["test_subject"].replace("DB2_", "") for r in valid]
        matrix = np.array([
            [r["test_report"].get(k, {}).get("f1-score", 0.0) for k in numeric_keys]
            for r in valid
        ])

        fig, ax = plt.subplots(figsize=(max(8, len(numeric_keys)), max(4, len(subjects) * 0.7)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="F1-score")
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels(subjects, fontsize=9)
        ax.set_xticks(range(len(numeric_keys)))
        ax.set_xticklabels([f"G{k}" for k in numeric_keys], fontsize=8)
        ax.set_xlabel("Gesture index")
        ax.set_ylabel("Test Subject (fold)")
        ax.set_title("Per-Gesture F1 Across All LOSO Folds", fontsize=11, fontweight="bold")
        # Annotate
        for i in range(len(subjects)):
            for j in range(len(numeric_keys)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / "per_class_f1_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] per_class_f1_heatmap failed: {exc}")


def _plot_loss_components_boxplots(
    all_fold_histories: List[Dict],
    output_dir: Path,
) -> None:
    """Box-plots of final-epoch loss components across folds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not all_fold_histories:
            return

        def last_n_mean(hist_key, n=5):
            return [np.mean(h[hist_key][-n:]) for h in all_fold_histories if hist_key in h and h[hist_key]]

        components = {
            "Gesture CE": last_n_mean("gesture_loss"),
            "Subject CE": last_n_mean("subject_loss"),
            "MI loss":    last_n_mean("mi_loss"),
            "Total loss": last_n_mean("train_loss"),
            "Val loss":   last_n_mean("val_loss"),
        }
        labels = [k for k, v in components.items() if v]
        data   = [v for v in components.values() if v]

        if not data:
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
        colors = ["steelblue", "tomato", "goldenrod", "purple", "teal"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Loss value (avg of last 5 epochs)")
        ax.set_title("Loss Component Distribution Across LOSO Folds\n(last 5 training epochs)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "loss_components_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[viz] loss_components_boxplots failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE LOSO FOLD
# ═══════════════════════════════════════════════════════════════════════════

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    dist_matrix: np.ndarray,
    all_subject_ids: List[str],
) -> Dict:
    """Single LOSO fold: load data, build splits, train, evaluate, visualize."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type = "disentangled_cnn_gru"

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # ── Data loader ──────────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

    fold_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=fold_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=MAX_GESTURES)
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    # ── Curriculum order ──────────────────────────────────────────────────────
    ranked = get_subject_order_by_similarity(
        dist_matrix, all_subject_ids, test_subject, train_subjects
    )
    sorted_train = sorted(train_subjects)
    train_subject_to_idx = {sid: i for i, sid in enumerate(sorted_train)}
    subject_order = [train_subject_to_idx[sid] for sid, _ in ranked]

    logger.info("Curriculum order (similar → distant):")
    for sid, d in ranked:
        logger.info(f"  {sid} (idx={train_subject_to_idx[sid]}, dist={d:.4f})")

    with open(output_dir / "curriculum_order.json", "w") as f:
        json.dump({
            "test_subject": test_subject,
            "train_order": [(sid, float(d)) for sid, d in ranked],
            "subject_order_indices": subject_order,
        }, f, indent=4)

    # ── Build splits ──────────────────────────────────────────────────────────
    splits = _build_splits_with_subject_labels(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        train_subject_to_idx=train_subject_to_idx,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )
    splits["train_subject_order"] = subject_order

    for sn in ["train", "val", "test"]:
        total = sum(len(a) for a in splits[sn].values() if isinstance(a, np.ndarray) and a.ndim == 3)
        logger.info(f"  {sn.upper()}: {total} windows, {len(splits[sn])} gestures")

    # ── Create trainer ────────────────────────────────────────────────────────
    trainer = CurriculumDisentangledTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        content_dim=CONTENT_DIM,
        style_dim=STYLE_DIM,
        alpha=ALPHA,
        beta=BETA,
        beta_anneal_epochs=BETA_ANNEAL_EPOCHS,
        mi_loss_type=MI_LOSS_TYPE,
        k_init=K_INIT,
        expand_every=EXPAND_EVERY,
        consolidation_epochs=CONSOLIDATION_EPOCHS,
        lr_on_expand=LR_ON_EXPAND,
        use_class_balanced_oversampling=USE_CLASS_BALANCED_OVERSAMPLING,
        use_subject_mixup=USE_SUBJECT_MIXUP,
        mixup_alpha=MIXUP_ALPHA,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # ── Evaluate on test subject ──────────────────────────────────────────────
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test data available.")
        return {
            "test_subject": test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test_cat = np.concatenate(X_test_list, axis=0)
    y_test_cat = np.concatenate(y_test_list, axis=0)

    test_results = trainer.evaluate_numpy(
        X_test_cat, y_test_cat,
        split_name=f"test_{test_subject}",
        visualize=True,
    )
    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(f"[LOSO] {test_subject} | Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}")

    # ── Per-fold visualizations ───────────────────────────────────────────────
    history = {}
    try:
        hist_path = output_dir / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
    except Exception:
        pass

    stage_stats = training_results.get("stage_stats_history", [])

    # Curriculum schedule (reconstructed from trainer)
    schedule = trainer._compute_schedule(len(train_subjects), subject_order)
    _plot_curriculum_gantt(schedule, sorted_train, output_dir)
    _plot_multi_loss_curves(history, output_dir)
    _plot_class_distribution_stages(
        stage_stats, class_ids,
        trainer.class_names if trainer.class_names else {gid: f"G{gid}" for gid in class_ids},
        output_dir,
    )
    _plot_per_class_f1(
        test_results.get("report", {}),
        class_ids,
        trainer.class_names if trainer.class_names else {gid: f"G{gid}" for gid in class_ids},
        test_subject, output_dir,
    )

    # t-SNE embeddings (on validation set with subject labels)
    if VISUALIZE_EMBEDDINGS:
        try:
            val_subject_labels_dict = splits.get("val_subject_labels", {})
            X_val_list, y_val_g_list, y_val_s_list = [], [], []
            for i, gid in enumerate(class_ids):
                if gid in splits["val"]:
                    arr = splits["val"][gid]
                    if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                        X_val_list.append(arr)
                        y_val_g_list.append(np.full(len(arr), i, dtype=np.int64))
                        if gid in val_subject_labels_dict:
                            y_val_s_list.append(val_subject_labels_dict[gid])
                        else:
                            y_val_s_list.append(np.zeros(len(arr), dtype=np.int64))

            if X_val_list and y_val_s_list:
                X_val_cat = np.concatenate(X_val_list, axis=0)
                y_val_g_cat = np.concatenate(y_val_g_list, axis=0)
                y_val_s_cat = np.concatenate(y_val_s_list, axis=0)
                _plot_embedding_tsne(
                    trainer, X_val_cat, y_val_g_cat, y_val_s_cat,
                    class_ids,
                    trainer.class_names if trainer.class_names else {gid: f"G{gid}" for gid in class_ids},
                    sorted_train, test_subject, output_dir,
                )
        except Exception as exc:
            logger.warning(f"t-SNE visualization failed: {exc}")

    # ── Save fold results ─────────────────────────────────────────────────────
    results_to_save = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "common_gestures": common_gestures,
        "curriculum_order": [(sid, float(d)) for sid, d in ranked],
        "training": {k: v for k, v in training_results.items() if k != "stage_stats_history"},
        "cross_subject_test": {
            "subject": test_subject,
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "report": test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(make_json_serializable({
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "disentangled_cnn_gru",
        "approach": APPROACH,
        "exercises": exercises,
        "hyperparameters": {
            "content_dim": CONTENT_DIM, "style_dim": STYLE_DIM,
            "alpha": ALPHA, "beta": BETA, "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
            "k_init": K_INIT, "expand_every": EXPAND_EVERY,
            "consolidation_epochs": CONSOLIDATION_EPOCHS,
            "use_class_balanced_oversampling": USE_CLASS_BALANCED_OVERSAMPLING,
            "use_subject_mixup": USE_SUBJECT_MIXUP, "mixup_alpha": MIXUP_ALPHA,
            "distance_metric": DISTANCE_METRIC,
        },
        "curriculum_order": [(sid, float(d)) for sid, d in ranked],
        "metrics": {"test_accuracy": test_acc, "test_f1_macro": test_f1},
    }), filename="fold_metadata.json")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "test_report": test_results.get("report"),
        "training_history": history,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ALL_SUBJECTS = parse_subjects_args()
    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject_tag = "_".join(s.replace("DB2_s", "") for s in ALL_SUBJECTS)
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}_{subject_tag}"

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
        model_type="disentangled_cnn_gru",
        pipeline_type=APPROACH,
        use_handcrafted_features=False,
        batch_size=64,
        epochs=200,           # curriculum scheduler controls actual duration
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=12,
        seed=42,
        use_class_weights=True,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"{'=' * 80}")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : Curriculum + Disentanglement + Class-Balanced MixUp Fusion")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Exercises  : {EXERCISES}")
    print(f"Curriculum : k_init={K_INIT}, expand_every={EXPAND_EVERY}, "
          f"consolidation={CONSOLIDATION_EPOCHS}")
    print(f"Disent.    : content_dim={CONTENT_DIM}, style_dim={STYLE_DIM}, "
          f"α={ALPHA}, β={BETA}")
    print(f"Augment.   : oversampling={USE_CLASS_BALANCED_OVERSAMPLING}, "
          f"mixup={USE_SUBJECT_MIXUP} (α={MIXUP_ALPHA})")
    print(f"Output     : {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    # ── Phase 1: Compute subject distance matrix ──────────────────────────────
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("\n[Phase 1] Computing subject distance matrix …")
    logger_global = setup_logging(OUTPUT_ROOT)
    seed_everything(train_cfg.seed)

    multi_loader_global = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger_global,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    all_subjects_data = multi_loader_global.load_multiple_subjects(
        base_dir=BASE_DIR,
        subject_ids=ALL_SUBJECTS,
        exercises=EXERCISES,
        include_rest=split_cfg.include_rest_in_splits,
    )

    dist_matrix = compute_distance_matrix(all_subjects_data, ALL_SUBJECTS, metric=DISTANCE_METRIC)

    with open(OUTPUT_ROOT / "distance_matrix.json", "w") as f:
        json.dump({"subjects": ALL_SUBJECTS, "metric": DISTANCE_METRIC,
                   "distance_matrix": dist_matrix.tolist()}, f, indent=4)

    print("Subject distance matrix:")
    for i, si in enumerate(ALL_SUBJECTS):
        row = "  ".join(f"{dist_matrix[i, j]:.3f}" for j in range(len(ALL_SUBJECTS)))
        print(f"  {si}: [{row}]")

    _plot_subject_distance_heatmap(dist_matrix, ALL_SUBJECTS, OUTPUT_ROOT)

    del all_subjects_data, multi_loader_global
    gc.collect()

    # ── Phase 2: LOSO folds ───────────────────────────────────────────────────
    print(f"\n[Phase 2] Running {len(ALL_SUBJECTS)} LOSO folds …")
    all_loso_results = []
    all_fold_histories = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output = OUTPUT_ROOT / "curriculum_disentangled" / f"test_{test_subject}"

        result = run_single_loso_fold(
            base_dir=BASE_DIR,
            output_dir=fold_output,
            train_subjects=train_subjects,
            test_subject=test_subject,
            exercises=EXERCISES,
            proc_cfg=proc_cfg,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
            dist_matrix=dist_matrix,
            all_subject_ids=ALL_SUBJECTS,
        )
        if "training_history" in result:
            all_fold_histories.append(result.pop("training_history"))

        all_loso_results.append(result)

    # ── Aggregate results ─────────────────────────────────────────────────────
    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    accs = [r["test_accuracy"] for r in valid_results]
    f1s  = [r["test_f1_macro"] for r in valid_results]

    if valid_results:
        print(f"\n{'=' * 60}")
        print(f"Curriculum Disentangled Fusion — LOSO Summary ({len(valid_results)} folds)")
        print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"{'=' * 60}\n")

    # ── Global visualizations ─────────────────────────────────────────────────
    _plot_loso_bars(all_loso_results, OUTPUT_ROOT)
    _plot_distance_vs_accuracy(all_loso_results, dist_matrix, ALL_SUBJECTS, OUTPUT_ROOT)
    _plot_per_class_f1_heatmap(all_loso_results, OUTPUT_ROOT)
    _plot_loss_components_boxplots(all_fold_histories, OUTPUT_ROOT)

    # ── Save summary ──────────────────────────────────────────────────────────
    summary: Dict = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": "H_fusion: Curriculum + Disentanglement + Class-Balanced MixUp",
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "approach": APPROACH,
        "hyperparameters": {
            "content_dim": CONTENT_DIM, "style_dim": STYLE_DIM,
            "alpha": ALPHA, "beta": BETA, "beta_anneal_epochs": BETA_ANNEAL_EPOCHS,
            "mi_loss_type": MI_LOSS_TYPE,
            "k_init": K_INIT, "expand_every": EXPAND_EVERY,
            "consolidation_epochs": CONSOLIDATION_EPOCHS,
            "use_class_balanced_oversampling": USE_CLASS_BALANCED_OVERSAMPLING,
            "use_subject_mixup": USE_SUBJECT_MIXUP, "mixup_alpha": MIXUP_ALPHA,
            "distance_metric": DISTANCE_METRIC,
        },
        "distance_matrix": dist_matrix.tolist(),
        "results": [
            {k: v for k, v in r.items() if k != "test_report"} for r in all_loso_results
        ],
    }
    if valid_results:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_folds":     len(valid_results),
        }

    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── Report to hypothesis_executor (if installed) ───────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        if valid_results:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
            }
            mark_hypothesis_verified("H_fusion", metrics, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_fusion", "All LOSO folds failed.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
