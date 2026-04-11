"""
Experiment 43: TCN-GAT Hybrid — Multi-scale Temporal Conv + Graph Attention (LOSO)

Hypothesis H43:
    Combining local multi-scale temporal feature extraction (causal dilated TCN,
    dilation ∈ {1,2,4,8}) with global inter-muscular co-activation modelling
    (multi-head Graph Attention Network over the channel graph) captures richer,
    more subject-invariant gesture representations than either purely-temporal
    (TCN, CNN-GRU) or purely-graph (ChannelGAT) models.

Architecture (models/tcn_gat_hybrid.py → TCNGATHybrid):
    (B, C, T) raw EMG
    → PerChannelDilatedTCN : causal TCN per channel, dilation {1,2,4,8}, kernel=7
                              effective RF ≈ 184 timesteps (≈92 ms @ 2 kHz)
                              → (B, C, T', d_tcn)
    → DynamicAdjacency     : Pearson corr (feature space) + learnable symmetric prior
                              → (B, C, C) adjacency bias
    → GATLayerExtractable×n: multi-head GAT at every time step
                              → (B, C, T', d_tcn) + attention (B, T', heads, C, C)
    → Per-channel BiGRU    : (B·C, T', d_tcn) → (B, C, T', d_gru·2)
    → TemporalAttention    : soft attention over T' steps per channel
                              → (B, C, d_gru·2),  weights (B, C, T')
    → ChannelAttention     : soft gate over C channels
                              → (B, d_gru·2),     gates (B, C)
    → MLP Classifier       → (B, num_classes)

Baseline comparison:
    exp_1  (SimpleCNN)         — purely temporal, no graph
    exp_37 (ChannelGATGRU)     — graph+GRU but plain CNN encoder, no temporal attn

Key differences vs exp_37 (ChannelGATGRU):
    1. Causal dilated TCN instead of plain 2-block CNN → multi-scale receptive field
    2. Temporal attention in addition to channel attention → learns WHICH time slice
       within the window is most informative per channel
    3. Simplified adjacency (Pearson + prior, no spectral coherence) — cleaner
       inductive bias; spectral info is already captured by dilated TCN

Visualisations produced:
    Per-fold (in fold output directory):
      1. tcn_receptive_field_diagram.png   — static: effective RF per dilation level
      2. adjacency_per_gesture.png         — learned inter-channel adj. matrix per class
      3. feat_corr_vs_prior_weights.png    — mixture weight of Pearson vs. prior over test
      4. gat_attention_heads.png           — per-head attention matrix (time-averaged)
      5. channel_importance_per_gesture.png— channel gate values per gesture class
      6. temporal_attention_heatmap.png    — temporal weights averaged per gesture class
      7. tsne_tcn_vs_gat.png              — t-SNE: raw→TCN features vs post-GAT features

    Global (experiment output directory):
      8. loso_summary_bar.png              — per-subject Acc + F1 bar chart
      9. loso_boxplot.png                  — distribution of Acc / F1 across subjects

Usage:
    python experiments/exp_43_tcn_gat_hybrid_loso.py --ci
    python experiments/exp_43_tcn_gat_hybrid_loso.py --subjects DB2_s1,DB2_s12,DB2_s15
    python experiments/exp_43_tcn_gat_hybrid_loso.py --full
"""

import os
import sys
import gc
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Repo root ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Subject lists ─────────────────────────────────────────────────────────────
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]


def parse_subjects_args() -> List[str]:
    import argparse
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument("--subjects", type=str, default=None)
    _p.add_argument("--ci",   action="store_true")
    _p.add_argument("--full", action="store_true")
    _a, _ = _p.parse_known_args()
    if _a.subjects:
        return [s.strip() for s in _a.subjects.split(",")]
    if _a.full:
        return _FULL_SUBJECTS
    return _CI_SUBJECTS   # safe default — server has only CI symlinks


# ── Project imports ───────────────────────────────────────────────────────────
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ── Register model ────────────────────────────────────────────────────────────
from models.tcn_gat_hybrid import TCNGATHybrid
from models import register_model

register_model("tcn_gat_hybrid", TCNGATHybrid)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_json_serializable(obj):
    from pathlib import Path as _P
    import numpy as _np
    if isinstance(obj, _P):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert grouped_windows (gesture_id → list of rep arrays (N_rep, T, C))
    to flat (windows, labels).
    Windows are returned as (N_total, T, C).
    Labels are gesture class indices (0-based, sorted by gesture_id).
    """
    sorted_ids = sorted(grouped_windows.keys())
    all_windows, all_labels = [], []
    for cls_idx, gid in enumerate(sorted_ids):
        for rep_arr in grouped_windows[gid]:
            n = rep_arr.shape[0]
            all_windows.append(rep_arr)
            all_labels.append(np.full(n, cls_idx, dtype=np.int64))
    windows = np.concatenate(all_windows, axis=0)  # (N, T, C)
    labels  = np.concatenate(all_labels,  axis=0)  # (N,)
    return windows, labels


# ── Custom colour map for adjacency (white → dark-blue) ──────────────────────
_ADJ_CMAP = LinearSegmentedColormap.from_list(
    "adj_white_blue", ["#ffffff", "#084594"]
)


# ═════════════════════════════════════════════════════════════════════════════
# Visualisation functions
# ═════════════════════════════════════════════════════════════════════════════

def _channel_labels(n: int) -> List[str]:
    return [f"CH{i+1}" for i in range(n)]


def plot_tcn_receptive_field(output_dir: Path) -> None:
    """
    Visualisation 1: TCN Receptive Field Pyramid.

    Static diagram showing how the stacked dilated causal convolutions expand
    the effective receptive field at each layer.
    """
    from models.tcn_gat_hybrid import PerChannelDilatedTCN

    # Compute RF per block
    ks = 7
    dilations = [1, 2, 4, 8]
    rfs, cumulative_rf = [], 1
    for d in dilations:
        block_rf = 1 + 2 * (ks - 1) * d
        cumulative_rf = cumulative_rf + block_rf - 1
        rfs.append(cumulative_rf)

    sampling_rate = 2000.0  # Hz

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "TCN Dilated Causal Convolutions — Receptive Field Growth",
        fontsize=14, fontweight="bold"
    )

    # ── Left: block-level RF bar chart ────────────────────────────────────
    ax = axes[0]
    colors = ["#4292c6", "#2166ac", "#084594", "#08306b"]
    bar_labels = [f"Block {i+1}\n(dilation={d})" for i, d in enumerate(dilations)]
    bars = ax.barh(bar_labels, rfs, color=colors, height=0.5, edgecolor="white", linewidth=1.5)
    for bar, rf in zip(bars, rfs):
        ms = rf / sampling_rate * 1000
        ax.text(
            bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            f"{rf} ts  ({ms:.1f} ms)",
            va="center", fontsize=9
        )
    ax.set_xlabel("Cumulative receptive field (timesteps @ 2 kHz)", fontsize=10)
    ax.set_title("Cumulative RF per stack depth", fontsize=11)
    ax.set_xlim(0, max(rfs) * 1.35)
    ax.axvline(200, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="100 ms (typical gesture onset)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # ── Right: schematic of dilated kernel coverage ────────────────────────
    ax2 = axes[1]
    t_axis = np.arange(50)
    for i, (d, color) in enumerate(zip(dilations, colors)):
        # Show positions covered by dilated kernel of size 7 centred at t=49
        positions = [49 - d * j for j in range(ks) if 49 - d * j >= 0]
        y_level = len(dilations) - i - 1
        ax2.scatter(positions, [y_level] * len(positions),
                    s=60, color=color, zorder=3,
                    label=f"d={d}, RF≈{rfs[i]} ts")
        ax2.plot([min(positions), 49], [y_level, y_level],
                 color=color, linewidth=1.5, alpha=0.5)

    ax2.axvline(49, color="black", linestyle="--", linewidth=1.5, label="Current time t")
    ax2.set_yticks(range(len(dilations)))
    ax2.set_yticklabels([f"d={d}" for d in reversed(dilations)])
    ax2.set_xlabel("Time (relative, most recent → right)", fontsize=10)
    ax2.set_title("Kernel coverage per dilation level (kernel=7, causal)", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_xlim(-5, 55)

    plt.tight_layout()
    out_path = output_dir / "tcn_receptive_field_diagram.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {out_path.name}")


def plot_adjacency_per_gesture(
    adj_by_class: np.ndarray,   # (num_classes, C, C)
    class_names: List[str],
    output_dir: Path,
) -> None:
    """
    Visualisation 2: Learned inter-channel adjacency matrix per gesture class.

    A strong hypothesis prediction: different gestures should activate
    different muscle-pair co-activation patterns.
    """
    n_classes = adj_by_class.shape[0]
    n_ch = adj_by_class.shape[1]
    ch_labels = _channel_labels(n_ch)

    ncols = min(5, n_classes)
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows))
    fig.suptitle(
        "Learned Inter-Muscle Adjacency (DynamicAdjacency) per Gesture\n"
        "↑ higher value = stronger predicted co-activation",
        fontsize=13, fontweight="bold"
    )
    axes_flat = np.array(axes).flatten()

    vmin = adj_by_class.min()
    vmax = adj_by_class.max()

    for i, (ax, adj) in enumerate(zip(axes_flat, adj_by_class)):
        im = ax.imshow(adj, cmap=_ADJ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(class_names[i] if i < len(class_names) else f"G{i}", fontsize=9)
        ax.set_xticks(range(n_ch))
        ax.set_yticks(range(n_ch))
        ax.set_xticklabels(ch_labels, rotation=45, fontsize=7)
        ax.set_yticklabels(ch_labels, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes_flat[n_classes:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = output_dir / "adjacency_per_gesture.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {out_path.name}")


def plot_adj_weight_pie(
    adj_mix_weights: np.ndarray,  # (2,) averaged over test set
    output_dir: Path,
) -> None:
    """
    Visualisation 3: Mixture weights of adjacency components.

    Shows how much the model relies on Pearson correlation vs. the
    learnable prior to construct the inter-channel graph.
    """
    labels = ["Pearson\nCorrelation", "Learnable\nPrior"]
    colors = ["#2166ac", "#f4a582"]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        adj_mix_weights, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title(
        "DynamicAdjacency Mixture Weights\n(Pearson Correlation vs. Learnable Prior)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = output_dir / "adj_mixture_weights_pie.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {out_path.name}")


def plot_gat_attention_heads(
    gat_attn_by_head: np.ndarray,  # (n_heads, C, C) time-averaged
    output_dir: Path,
) -> None:
    """
    Visualisation 4: GAT attention per head (time-averaged over test set).

    Shows head specialisation — different heads capture different
    inter-muscular communication patterns.
    """
    n_heads, n_ch, _ = gat_attn_by_head.shape
    ch_labels = _channel_labels(n_ch)

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    fig.suptitle(
        "GAT Attention Weights per Head (time-averaged over test set)\n"
        "Row i → Column j: how much channel j attends to channel i",
        fontsize=12, fontweight="bold"
    )

    for h_idx, ax in enumerate(axes):
        data = gat_attn_by_head[h_idx]
        im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=data.max(), aspect="auto")
        ax.set_title(f"Head {h_idx + 1}", fontsize=10)
        ax.set_xticks(range(n_ch))
        ax.set_yticks(range(n_ch))
        ax.set_xticklabels(ch_labels, rotation=45, fontsize=7)
        ax.set_yticklabels(ch_labels, fontsize=7)
        ax.set_xlabel("Key (source)", fontsize=8)
        ax.set_ylabel("Query (target)", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_path = output_dir / "gat_attention_heads.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {out_path.name}")


def plot_channel_importance_per_gesture(
    chan_gates_by_class: np.ndarray,  # (num_classes, C)
    class_names: List[str],
    output_dir: Path,
) -> None:
    """
    Visualisation 5: Channel (electrode) importance per gesture.

    The channel attention gate values reveal which muscles the model
    focuses on for each gesture class.
    """
    n_classes, n_ch = chan_gates_by_class.shape
    ch_labels = _channel_labels(n_ch)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(5, n_classes * 0.55)),
        gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # ── Left: heatmap (gestures × channels) ──────────────────────────────
    ax = axes[0]
    im = ax.imshow(chan_gates_by_class, cmap="Blues", aspect="auto",
                   vmin=0, vmax=chan_gates_by_class.max())
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(ch_labels, fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names if class_names else [f"G{i}" for i in range(n_classes)],
                       fontsize=9)
    ax.set_title("Channel Gate Value per Gesture\n(↑ = more important)", fontsize=11)
    ax.set_xlabel("EMG Channel", fontsize=10)
    ax.set_ylabel("Gesture Class", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7)
    # annotate cells
    for i in range(n_classes):
        for j in range(n_ch):
            ax.text(j, i, f"{chan_gates_by_class[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if chan_gates_by_class[i, j] > 0.5 * chan_gates_by_class.max() else "black")

    # ── Right: mean channel importance across all gestures ─────────────
    ax2 = axes[1]
    mean_importance = chan_gates_by_class.mean(axis=0)   # (C,)
    std_importance  = chan_gates_by_class.std(axis=0)
    colors = plt.cm.Blues(mean_importance / mean_importance.max())
    ax2.barh(ch_labels, mean_importance, xerr=std_importance,
             color=colors, edgecolor="white", linewidth=1.2)
    ax2.set_xlabel("Mean Channel Gate (± std across gestures)", fontsize=9)
    ax2.set_title("Overall Channel\nImportance", fontsize=11)
    ax2.invert_yaxis()

    plt.suptitle("Channel Attention: Electrode Importance Analysis", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "channel_importance_per_gesture.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {out_path.name}")


def plot_temporal_attention_heatmap(
    temp_attn_by_class: np.ndarray,  # (num_classes, C, T')
    class_names: List[str],
    window_size_ms: float,
    output_dir: Path,
) -> None:
    """
    Visualisation 6: Temporal attention weights per gesture and channel.

    Reveals WHEN within the EMG window the model pays most attention.
    The hypothesis predicts: attention should peak during the active
    phase of the gesture rather than at rest transitions.
    """
    n_classes, n_ch, T_prime = temp_attn_by_class.shape
    ch_labels = _channel_labels(n_ch)
    time_axis = np.linspace(0, window_size_ms, T_prime)

    # Average over channels for a compact view
    temp_mean = temp_attn_by_class.mean(axis=1)  # (n_classes, T')

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # ── Top: heatmap gesture × time ────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(
        temp_mean, cmap="hot", aspect="auto",
        extent=[0, window_size_ms, n_classes - 0.5, -0.5]
    )
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names if class_names else [f"G{i}" for i in range(n_classes)],
                       fontsize=8)
    ax.set_xlabel("Time within window (ms)", fontsize=10)
    ax.set_title("Temporal Attention Weight (channel-averaged) per Gesture Class\n"
                 "↑ brighter = model focuses more on this time step",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="Attention weight")

    # ── Bottom: mean ± std across all gestures ──────────────────────────
    ax2 = axes[1]
    global_mean = temp_mean.mean(axis=0)
    global_std  = temp_mean.std(axis=0)
    ax2.fill_between(time_axis, global_mean - global_std, global_mean + global_std,
                     alpha=0.3, color="#2166ac", label="±1 std across gestures")
    ax2.plot(time_axis, global_mean, color="#2166ac", linewidth=2,
             label="Mean attention")
    ax2.set_xlabel("Time within window (ms)", fontsize=10)
    ax2.set_ylabel("Attention weight", fontsize=10)
    ax2.set_title("Global Temporal Attention Profile", fontsize=11)
    ax2.legend(fontsize=9)

    plt.suptitle("TemporalAttention: When does the model focus?", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "temporal_attention_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {out_path.name}")


def plot_tsne_tcn_vs_gat(
    tcn_features: np.ndarray,   # (N, d_tcn) — channel+time averaged TCN features
    gat_features: np.ndarray,   # (N, d_gru2) — readout features after GAT+GRU+attn
    labels: np.ndarray,          # (N,) class indices
    class_names: List[str],
    output_dir: Path,
    max_samples: int = 2000,
) -> None:
    """
    Visualisation 7: t-SNE comparison — TCN features vs post-GAT readout.

    Shows whether the GAT+GRU+Attention pipeline produces more linearly
    separable class clusters compared to raw TCN features.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [viz] sklearn not available, skipping t-SNE plot")
        return

    # Subsample to keep t-SNE tractable
    if len(labels) > max_samples:
        idx = np.random.choice(len(labels), max_samples, replace=False)
        tcn_features = tcn_features[idx]
        gat_features = gat_features[idx]
        labels = labels[idx]

    n_cls = len(np.unique(labels))
    palette = plt.cm.get_cmap("tab10", n_cls)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "t-SNE: TCN features vs. post-GAT readout\n"
        "(does graph attention improve class separability?)",
        fontsize=13, fontweight="bold"
    )

    for ax, features, title in zip(
        axes,
        [tcn_features, gat_features],
        ["After TCN Encoder\n(before GAT)", "After TCN→GAT→GRU→Attention\n(final readout)"]
    ):
        tsne = TSNE(n_components=2, perplexity=min(30, len(labels) // 4),
                    n_iter=1000, random_state=42)
        emb = tsne.fit_transform(features)

        for cls_idx in np.unique(labels):
            mask = labels == cls_idx
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"G{cls_idx}"
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[palette(cls_idx)], label=name,
                       s=12, alpha=0.6, linewidths=0)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t-SNE dim 1", fontsize=9)
        ax.set_ylabel("t-SNE dim 2", fontsize=9)
        ax.legend(fontsize=7, markerscale=2, ncol=2,
                  loc="upper right", framealpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = output_dir / "tsne_tcn_vs_gat.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {out_path.name}")


def plot_loso_summary(
    all_results: List[Dict],
    output_dir: Path,
    experiment_name: str,
) -> None:
    """
    Visualisations 8 & 9: Per-subject Acc/F1 bar chart + box plot.
    """
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    if not valid:
        return

    subjects = [r["test_subject"] for r in valid]
    accs = np.array([r["test_accuracy"] for r in valid])
    f1s  = np.array([r["test_f1_macro"] for r in valid])
    x = np.arange(len(subjects))

    # ── Bar chart ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.2), 5))
    w = 0.38
    ax.bar(x - w / 2, accs, width=w, label="Accuracy",  color="#2166ac", alpha=0.85)
    ax.bar(x + w / 2, f1s,  width=w, label="F1-macro",  color="#f4a582", alpha=0.85)

    for xi, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(xi - w / 2, a + 0.005, f"{a:.3f}", ha="center", va="bottom",
                fontsize=7, rotation=45)
        ax.text(xi + w / 2, f + 0.005, f"{f:.3f}", ha="center", va="bottom",
                fontsize=7, rotation=45)

    ax.axhline(accs.mean(), color="#2166ac", linestyle="--", linewidth=1.5,
               label=f"Mean Acc={accs.mean():.4f}")
    ax.axhline(f1s.mean(),  color="#f4a582", linestyle="--", linewidth=1.5,
               label=f"Mean F1={f1s.mean():.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"{experiment_name} — LOSO Per-Subject Results\n"
        f"Acc: {accs.mean():.4f}±{accs.std():.4f}   "
        f"F1: {f1s.mean():.4f}±{f1s.std():.4f}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = output_dir / "loso_summary_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {out.name}")

    # ── Box plot ───────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    bp = ax2.boxplot(
        [accs, f1s], labels=["Accuracy", "F1-macro"],
        patch_artist=True, widths=0.45,
        medianprops={"color": "white", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor("#2166ac")
    bp["boxes"][1].set_facecolor("#f4a582")
    for i, (data, color) in enumerate(zip([accs, f1s], ["#2166ac", "#f4a582"])):
        x_jitter = np.random.normal(i + 1, 0.05, len(data))
        ax2.scatter(x_jitter, data, s=40, color=color, alpha=0.7, zorder=3)
    ax2.set_title(f"LOSO Score Distribution\n(n={len(accs)} subjects)", fontsize=12)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_ylim(0, 1.05)
    plt.tight_layout()
    out2 = output_dir / "loso_boxplot.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [viz] Saved: {out2.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Core: extract attention from trained model on test data
# ═════════════════════════════════════════════════════════════════════════════

def visualize_model_internals(
    trainer,
    results: Dict,
    test_subject: str,
    output_dir: Path,
    proc_cfg: "ProcessingConfig",
    batch_size: int = 64,
) -> None:
    """
    Post-fold interpretability visualisation.

    Loads test subject's windows, runs inference with attention extraction,
    then produces all interpretability plots for this fold.

    Expected call: AFTER experiment.run(), BEFORE cleanup.
    """
    import torch
    from models.tcn_gat_hybrid import TCNGATHybrid

    model = trainer.model
    if model is None or not isinstance(model, TCNGATHybrid):
        print("  [viz] Model not TCNGATHybrid — skipping interpretability plots")
        return

    device = trainer.cfg.device
    mean_c = trainer.mean_c   # (C,) or (1, C, 1)
    std_c  = trainer.std_c

    # Get test subject windows
    subjects_data = results.get("subjects_data", {})
    subj_tuple = subjects_data.get(test_subject)
    if subj_tuple is None:
        print(f"  [viz] Test subject {test_subject} not in subjects_data — skipping")
        return

    _, _, grouped_windows = subj_tuple
    if not grouped_windows:
        print("  [viz] Empty grouped_windows — skipping")
        return

    windows, labels = grouped_to_arrays(grouped_windows)
    # windows: (N, T, C)  → trainer uses (N, C, T) after transpose
    # trainer.fit() internally computes mean_c/std_c on (N, C, T)

    if windows is None or len(windows) == 0:
        return

    # Transpose to (N, C, T)
    X = windows.transpose(0, 2, 1)  # (N, C, T)

    # Standardise using the trained trainer's statistics
    if mean_c is not None and std_c is not None:
        mean_ = mean_c if mean_c.ndim == 3 else mean_c.reshape(1, -1, 1)
        std_  = std_c  if std_c.ndim  == 3 else std_c.reshape(1, -1, 1)
        std_safe = np.where(std_ < 1e-8, 1.0, std_)
        X = (X - mean_) / std_safe

    n_classes = len(trainer.class_ids) if trainer.class_ids is not None else int(labels.max()) + 1
    class_ids = trainer.class_ids or list(range(n_classes))
    class_names = [str(trainer.class_names.get(cid, f"G{cid}"))
                   for cid in class_ids] if trainer.class_names else [f"G{i}" for i in range(n_classes)]

    model.eval()

    # ── Collect attention dicts over all batches ─────────────────────────
    all_adj       = []   # (B, C, C)
    all_feat_corr = []   # (B, C, C)
    all_adj_w     = []   # (B, 2)
    all_gat_attn  = []   # (B, T', heads, C, C)
    all_temp_attn = []   # (B, C, T')
    all_chan_gates = []  # (B, C)
    all_tcn_feats = []   # (B, d_tcn)  — time & channel averaged for t-SNE
    all_readout   = []   # (B, d_gru2) — for t-SNE post-GAT
    all_labels_out = []

    n = len(X)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            lb = labels[start:start + batch_size]

            logits, attn_dict = model.forward_with_attention(xb)

            all_adj.append(attn_dict["adjacency"].cpu().numpy())
            all_feat_corr.append(attn_dict["feat_corr"].cpu().numpy())
            all_adj_w.append(attn_dict["adj_mix_weights"].cpu().numpy().reshape(1, -1)
                             .repeat(xb.shape[0], axis=0))
            if attn_dict["gat_attention"] is not None:
                all_gat_attn.append(attn_dict["gat_attention"].cpu().numpy())
            all_temp_attn.append(attn_dict["temporal_weights"].cpu().numpy())
            all_chan_gates.append(attn_dict["channel_gates"].cpu().numpy())

            # For t-SNE: reduce TCN features to (B, d_tcn) by averaging time & channels
            tcn_f = attn_dict["tcn_features"]  # (B, C, T', d_tcn)
            if tcn_f is not None:
                tcn_pooled = tcn_f.mean(dim=(1, 2)).cpu().numpy()  # (B, d_tcn)
                all_tcn_feats.append(tcn_pooled)

            # Readout = after GAT+GRU+attention (proxy: logits' input)
            # We use channel_gates × channel-aggregated features as proxy;
            # or simply use the model's readout via classifier's input.
            # Capture via hook on the classifier:
            all_labels_out.append(lb)

    # ── Also run forward pass to collect readout features via hook ────────
    readout_features = []

    def _hook(module, inp, out):
        readout_features.append(inp[0].detach().cpu().numpy())

    hook_handle = model.classifier.register_forward_hook(_hook)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            model(xb)  # standard forward to trigger hook
    hook_handle.remove()

    # ── Concatenate ───────────────────────────────────────────────────────
    adj_all       = np.concatenate(all_adj,        axis=0)  # (N, C, C)
    feat_corr_all = np.concatenate(all_feat_corr,  axis=0)  # (N, C, C)
    adj_w_all     = np.concatenate(all_adj_w,      axis=0)  # (N, 2)
    temp_attn_all = np.concatenate(all_temp_attn,  axis=0)  # (N, C, T')
    chan_gates_all = np.concatenate(all_chan_gates, axis=0)  # (N, C)
    labels_all    = np.concatenate(all_labels_out, axis=0)  # (N,)

    if all_gat_attn:
        gat_attn_all = np.concatenate(all_gat_attn, axis=0)  # (N, T', heads, C, C)
    else:
        gat_attn_all = None

    tcn_feats_all = (
        np.concatenate(all_tcn_feats, axis=0) if all_tcn_feats else None
    )
    readout_all = (
        np.concatenate(readout_features, axis=0) if readout_features else None
    )

    n_classes_actual = len(np.unique(labels_all))
    class_names_trimmed = class_names[:n_classes_actual]

    # ── Compute per-class averages ────────────────────────────────────────
    adj_by_class      = np.zeros((n_classes_actual, adj_all.shape[1], adj_all.shape[2]))
    chan_gates_by_cls = np.zeros((n_classes_actual, chan_gates_all.shape[1]))
    T_prime = temp_attn_all.shape[2]
    C       = adj_all.shape[1]
    temp_by_cls = np.zeros((n_classes_actual, C, T_prime))

    for cls_idx in range(n_classes_actual):
        mask = labels_all == cls_idx
        if mask.sum() == 0:
            continue
        adj_by_class[cls_idx]      = adj_all[mask].mean(axis=0)
        chan_gates_by_cls[cls_idx] = chan_gates_all[mask].mean(axis=0)
        temp_by_cls[cls_idx]       = temp_attn_all[mask].mean(axis=0)

    # Per-head GAT attention (time-averaged)
    if gat_attn_all is not None:
        # (N, T', heads, C, C) → mean over N and T' → (heads, C, C)
        gat_by_head = gat_attn_all.mean(axis=(0, 1))  # (heads, C, C)
    else:
        gat_by_head = None

    # Mean adjacency mixture weights
    mean_adj_w = adj_w_all.mean(axis=0)  # (2,)

    # ── Window size in ms ─────────────────────────────────────────────────
    ws_ms = proc_cfg.window_size / proc_cfg.sampling_rate * 1000.0

    # ── Produce plots ──────────────────────────────────────────────────────
    print(f"\n  [viz] Generating interpretability plots for {test_subject}...")

    plot_adjacency_per_gesture(adj_by_class, class_names_trimmed, output_dir)
    plot_adj_weight_pie(mean_adj_w, output_dir)

    if gat_by_head is not None:
        plot_gat_attention_heads(gat_by_head, output_dir)

    plot_channel_importance_per_gesture(chan_gates_by_cls, class_names_trimmed, output_dir)
    plot_temporal_attention_heatmap(temp_by_cls, class_names_trimmed, ws_ms, output_dir)

    if tcn_feats_all is not None and readout_all is not None:
        plot_tsne_tcn_vs_gat(
            tcn_features=tcn_feats_all,
            gat_features=readout_all,
            labels=labels_all,
            class_names=class_names_trimmed,
            output_dir=output_dir,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Single LOSO fold
# ═════════════════════════════════════════════════════════════════════════════

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    proc_cfg: "ProcessingConfig",
    split_cfg: "SplitConfig",
    train_cfg: "TrainingConfig",
    visualize_internals: bool = True,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.model_type = model_type
    train_cfg.pipeline_type = "deep_raw"
    train_cfg.use_handcrafted_features = False

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects,
        test_subject=test_subject,
        exercises=exercises,
        base_dir=base_dir,
        pool_train_subjects=True,
        use_separate_val_subject=False,
        val_subject=None,
        val_ratio=0.15,
        seed=train_cfg.seed,
        max_gestures=10,
    )
    cs_cfg.save(output_dir / "cross_subject_config.json")

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=True,
    )

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    trainer = WindowClassifierTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=base_viz,
        logger=logger,
    )

    try:
        results = experiment.run()
    except Exception as e:
        print(f"Error in LOSO fold (test={test_subject}, model={model_type}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1  = float(test_metrics.get("f1_macro", 0.0))

    acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
    f1_str  = f"{test_f1:.4f}"  if test_f1  is not None else "N/A"
    print(
        f"[LOSO] Test={test_subject} | Model={model_type} | "
        f"Acc={acc_str}, F1={f1_str}"
    )

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject":   test_subject,
        "train_subjects": train_subjects,
        "model_type":     model_type,
        "exercises":      exercises,
        "hypothesis": (
            "H43: Multi-scale causal TCN + dynamic channel-graph GAT + BiGRU + "
            "temporal attention captures subject-invariant inter-muscular co-activation."
        ),
        "config": {
            "processing":     asdict(proc_cfg),
            "split":          asdict(split_cfg),
            "training":       asdict(train_cfg),
            "cross_subject":  {
                "train_subjects": train_subjects,
                "test_subject":   test_subject,
                "exercises":      exercises,
            },
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # ── Interpretability visualisations ───────────────────────────────────
    if visualize_internals:
        try:
            visualize_model_internals(
                trainer=trainer,
                results=results,
                test_subject=test_subject,
                output_dir=output_dir,
                proc_cfg=proc_cfg,
            )
        except Exception as viz_err:
            print(f"  [viz] Warning: interpretability plots failed: {viz_err}")
            traceback.print_exc()

    # ── Memory cleanup ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    EXPERIMENT_NAME = "exp_43_tcn_gat_hybrid_loso"
    BASE_DIR     = ROOT / "data"
    OUTPUT_DIR   = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    ALL_SUBJECTS = parse_subjects_args()

    EXERCISES   = ["E1"]
    MODEL_TYPES = ["tcn_gat_hybrid"]

    # ── Processing config ──────────────────────────────────────────────────
    proc_cfg = ProcessingConfig(
        window_size=400,         # 200 ms @ 2 kHz
        window_overlap=200,      # 50% overlap
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )

    # ── Split config ───────────────────────────────────────────────────────
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    # ── Training config ────────────────────────────────────────────────────
    # TCNGATHybrid is deeper than SimpleCNN but lighter than ChannelGATGRU
    # due to the reduced adjacency complexity.  Moderate hyperparams.
    train_cfg = TrainingConfig(
        batch_size=128,
        epochs=80,
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=18,
        use_class_weights=True,
        seed=42,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"EXPERIMENT : {EXPERIMENT_NAME}")
    global_logger.info(
        "Hypothesis : H43 — Multi-scale causal dilated TCN (RF≈184ts) + "
        "channel-graph GAT + BiGRU + temporal attention"
    )
    global_logger.info(f"Models     : {MODEL_TYPES}")
    global_logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    global_logger.info(f"Exercises  : {EXERCISES}")
    global_logger.info("=" * 80)

    # ── Static visualisation: TCN receptive field (does not need data) ─────
    try:
        plot_tcn_receptive_field(OUTPUT_DIR)
    except Exception as e:
        print(f"  [viz] TCN RF diagram failed: {e}")

    all_loso_results: List[Dict] = []

    for model_type in MODEL_TYPES:
        print(f"\nMODEL: {model_type} — LOSO over {len(ALL_SUBJECTS)} subjects")
        for test_subject in ALL_SUBJECTS:
            print(f"  LOSO fold: test_subject={test_subject}")
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"

            try:
                fold_res = run_single_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    model_type=model_type,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                    visualize_internals=True,
                )
                all_loso_results.append(fold_res)
                acc_s = (
                    f"{fold_res['test_accuracy']:.4f}"
                    if fold_res["test_accuracy"] is not None else "N/A"
                )
                f1_s = (
                    f"{fold_res['test_f1_macro']:.4f}"
                    if fold_res["test_f1_macro"] is not None else "N/A"
                )
                print(f"  → acc={acc_s}, f1={f1_s}")

            except Exception as e:
                global_logger.error(f"Failed fold test={test_subject}: {e}")
                global_logger.error(traceback.format_exc())
                all_loso_results.append({
                    "test_subject":  test_subject,
                    "model_type":    model_type,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": str(e),
                })

    # ── Aggregate ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("AGGREGATING LOSO RESULTS")
    print("=" * 80)

    aggregate_results: Dict = {}
    for model_type in MODEL_TYPES:
        model_res = [
            r for r in all_loso_results
            if r["model_type"] == model_type and r.get("test_accuracy") is not None
        ]
        if not model_res:
            continue
        accs = [r["test_accuracy"] for r in model_res]
        f1s  = [r["test_f1_macro"] for r in model_res]
        aggregate_results[model_type] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_subjects":  len(accs),
            "per_subject":   model_res,
        }
        print(
            f"  {model_type:35s}: "
            f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
            f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} "
            f"(n={len(accs)})"
        )

    # ── Save summary ───────────────────────────────────────────────────────
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "H43: Multi-scale causal dilated TCN (dilation {1,2,4,8}, kernel=7, "
            "RF≈184 ts ≈ 92 ms @ 2 kHz) + dynamic channel-graph GAT "
            "(Pearson + learnable prior) + per-channel BiGRU + temporal attention "
            "captures subject-invariant inter-muscular co-activation patterns."
        ),
        "feature_set":        "deep_raw",
        "models":             MODEL_TYPES,
        "subjects":           ALL_SUBJECTS,
        "exercises":          EXERCISES,
        "processing_config":  asdict(proc_cfg),
        "split_config":       asdict(split_cfg),
        "training_config":    asdict(train_cfg),
        "aggregate_results":  aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date":    datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)

    # ── Global summary visualisations ──────────────────────────────────────
    try:
        plot_loso_summary(all_loso_results, OUTPUT_DIR, EXPERIMENT_NAME)
    except Exception as e:
        print(f"  [viz] Summary plot failed: {e}")

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")

    # ── Hypothesis executor callback (optional dependency) ─────────────────
    try:
        from hypothesis_executor import (  # noqa
            mark_hypothesis_verified,
            mark_hypothesis_failed,
        )
        if aggregate_results:
            best = max(aggregate_results.values(), key=lambda x: x["mean_accuracy"])
            mark_hypothesis_verified(
                "H43_tcn_gat_hybrid",
                metrics={
                    "mean_accuracy": best["mean_accuracy"],
                    "std_accuracy":  best["std_accuracy"],
                    "mean_f1_macro": best["mean_f1_macro"],
                    "std_f1_macro":  best["std_f1_macro"],
                },
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                "H43_tcn_gat_hybrid",
                "No successful LOSO folds — check data and model configuration.",
            )
    except ImportError:
        pass
    except Exception as cb_err:
        print(f"hypothesis_executor callback error: {cb_err}")


if __name__ == "__main__":
    main()
