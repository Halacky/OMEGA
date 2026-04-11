#!/usr/bin/env python3
"""
H6 Unified Ablation — Paper Figure Generation

Generates four publication-quality figures from H6 ablation results:
  1. Ablation bar chart (F1 macro with error bars and delta annotations)
  2. Per-subject heatmap (subjects x variants)
  3. Per-subject delta plot (C vs A improvement)
  4. Cumulative contribution stacked bar
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments_output", "h6_unified_ablation_20260312_130226",
)
OUT_DIR = RESULT_DIR

VARIANT_KEYS = ["A", "B", "C", "D"]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Professional color palette (IEEE-friendly, colorblind-safe)
COLORS = {
    "A": "#4C72B0",  # steel blue
    "B": "#55A868",  # sage green
    "C": "#C44E52",  # muted red
    "D": "#8172B2",  # soft purple
}

# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data():
    variants = {}
    for v in VARIANT_KEYS:
        path = os.path.join(RESULT_DIR, f"variant_{v}.json")
        with open(path) as f:
            variants[v] = json.load(f)
    return variants


def _save(fig, name):
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(out, dpi=300)
        print(f"  saved {out}")
    plt.close(fig)


# ── Figure 1: Ablation Bar Chart ─────────────────────────────────────────────

def fig_ablation_bar(variants):
    """Grouped bar chart: mean F1 macro per variant with std error bars."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    labels = []
    means = []
    stds = []
    for v in VARIANT_KEYS:
        d = variants[v]
        labels.append(f"({v}) {d['label']}")
        means.append(d["mean_f1_macro"] * 100)
        stds.append(d["std_f1_macro"] * 100)

    x = np.arange(len(VARIANT_KEYS))
    bar_colors = [COLORS[v] for v in VARIANT_KEYS]

    bars = ax.bar(
        x, means, width=0.55, color=bar_colors, edgecolor="white",
        linewidth=0.5, yerr=stds, capsize=4, error_kw={"linewidth": 0.9},
    )

    # Delta annotations between consecutive bars
    for i in range(1, len(means)):
        delta = means[i] - means[0]
        sign = "+" if delta >= 0 else ""
        y_top = means[i] + stds[i] + 1.0
        ax.annotate(
            f"{sign}{delta:.1f} pp",
            xy=(x[i], y_top),
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
            color=bar_colors[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"({v})" for v in VARIANT_KEYS], fontsize=10)
    ax.set_ylabel("F1 Macro (%)", fontsize=10)
    ax.set_title("H6 Ablation: Cumulative Component Contribution", fontsize=11, pad=8)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend below
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[v], label=f"({v}) {variants[v]['label']}")
        for v in VARIANT_KEYS
    ]
    ax.legend(
        handles=legend_elements, loc="lower right", frameon=False,
        fontsize=7.5, ncol=1,
    )

    ax.set_ylim(0, max(means) + max(stds) + 6)

    _save(fig, "fig_h6_ablation_f1")


# ── Figure 2: Per-Subject Heatmap ────────────────────────────────────────────

def fig_per_subject_heatmap(variants):
    """Heatmap: subjects (rows) x variants (cols), colored by F1 macro."""

    # Build per-subject F1 dict for each variant
    subj_f1 = {}  # {subject: {variant: f1}}
    for v in VARIANT_KEYS:
        for entry in variants[v]["per_subject"]:
            subj = entry["test_subject"]
            subj_f1.setdefault(subj, {})[v] = entry["f1_macro"]

    # Sort subjects by variant A performance (ascending, worst at top)
    subjects_sorted = sorted(subj_f1.keys(), key=lambda s: subj_f1[s]["A"])

    # Build matrix
    matrix = np.zeros((len(subjects_sorted), len(VARIANT_KEYS)))
    for i, subj in enumerate(subjects_sorted):
        for j, v in enumerate(VARIANT_KEYS):
            matrix[i, j] = subj_f1[subj][v] * 100

    fig, ax = plt.subplots(figsize=(5.0, 6.5))

    cmap = plt.cm.YlOrRd
    norm = Normalize(vmin=matrix.min() - 1, vmax=matrix.max() + 1)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val > (matrix.max() + matrix.min()) / 2 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="medium")

    ax.set_xticks(np.arange(len(VARIANT_KEYS)))
    ax.set_xticklabels(
        [f"({v})" for v in VARIANT_KEYS],
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(subjects_sorted)))
    ax.set_yticklabels(subjects_sorted, fontsize=7.5)

    ax.set_title("Per-Subject F1 Macro (%) by Variant", fontsize=10, pad=8)
    ax.set_xlabel(
        "  |  ".join([f"({v}) {variants[v]['label']}" for v in VARIANT_KEYS]),
        fontsize=7, labelpad=6,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("F1 Macro (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "fig_h6_per_subject_heatmap")


# ── Figure 3: Per-Subject Delta (C vs A) ─────────────────────────────────────

def fig_delta_per_subject(variants):
    """Horizontal bar chart: F1 delta (C - A) per subject, sorted."""

    deltas = {}
    f1_a = {e["test_subject"]: e["f1_macro"] for e in variants["A"]["per_subject"]}
    f1_c = {e["test_subject"]: e["f1_macro"] for e in variants["C"]["per_subject"]}

    for subj in f1_a:
        deltas[subj] = (f1_c[subj] - f1_a[subj]) * 100  # percentage points

    # Sort by delta
    sorted_subjects = sorted(deltas.keys(), key=lambda s: deltas[s])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    y_pos = np.arange(len(sorted_subjects))
    delta_vals = [deltas[s] for s in sorted_subjects]
    bar_colors = ["#55A868" if d >= 0 else "#C44E52" for d in delta_vals]

    ax.barh(y_pos, delta_vals, color=bar_colors, edgecolor="white",
            linewidth=0.4, height=0.7)

    # Zero line
    ax.axvline(0, color="black", linewidth=0.6, linestyle="-")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_subjects, fontsize=7.5)
    ax.set_xlabel("F1 Macro Change (pp): Variant C vs A", fontsize=9)
    ax.set_title(
        "Per-Subject Improvement: + Per-band MixStyle vs Raw CNN",
        fontsize=10, pad=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate values on bars
    for i, (subj, dv) in enumerate(zip(sorted_subjects, delta_vals)):
        ha = "left" if dv >= 0 else "right"
        offset = 0.3 if dv >= 0 else -0.3
        ax.text(dv + offset, i, f"{dv:+.1f}", va="center", ha=ha,
                fontsize=7, color="#333333")

    _save(fig, "fig_h6_delta_per_subject")


# ── Figure 4: Cumulative Contribution Stacked Bar ────────────────────────────

def fig_cumulative_contribution(variants):
    """Stacked bar showing cumulative contribution of each component."""

    mean_a = variants["A"]["mean_f1_macro"] * 100
    mean_b = variants["B"]["mean_f1_macro"] * 100
    mean_c = variants["C"]["mean_f1_macro"] * 100
    mean_d = variants["D"]["mean_f1_macro"] * 100

    # Component contributions (incremental)
    base = mean_a
    delta_decomp = mean_b - mean_a       # Sinc Filterbank
    delta_mixstyle = mean_c - mean_b     # Per-band MixStyle
    delta_cs = mean_d - mean_c           # Content/Style heads

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    component_labels = [
        "Baseline\n(Raw CNN)",
        "Sinc Filterbank\n(decomposition)",
        "Per-band\nMixStyle",
        "Content/Style\nheads",
    ]
    component_values = [base, delta_decomp, delta_mixstyle, delta_cs]
    component_colors = [COLORS["A"], COLORS["B"], COLORS["C"], COLORS["D"]]

    # Build stacked bar (single bar, stacked)
    x = [0]
    bottom = 0
    bars = []
    for i, (val, color, label) in enumerate(
        zip(component_values, component_colors, component_labels)
    ):
        b = ax.bar(x, val, bottom=bottom, color=color, edgecolor="white",
                   linewidth=0.5, width=0.5, label=label)
        bars.append(b)

        # Annotate inside each segment
        mid_y = bottom + val / 2
        text = f"{val:.1f} pp" if i > 0 else f"{val:.1f}%"
        text_color = "white" if val > 3 else "black"
        fontsize = 8 if abs(val) > 1.5 else 6.5
        ax.text(0, mid_y, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color)

        bottom += val

    # Total annotation
    ax.text(0, bottom + 0.5, f"Total: {bottom:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0, bottom + 4)
    ax.set_xticks([])
    ax.set_ylabel("F1 Macro (%)", fontsize=10)
    ax.set_title("Cumulative Component Contribution", fontsize=11, pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=8)

    _save(fig, "fig_h6_cumulative")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from {RESULT_DIR}")
    variants = load_data()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[1/4] Ablation bar chart")
    fig_ablation_bar(variants)

    print("\n[2/4] Per-subject heatmap")
    fig_per_subject_heatmap(variants)

    print("\n[3/4] Per-subject delta (C vs A)")
    fig_delta_per_subject(variants)

    print("\n[4/4] Cumulative contribution")
    fig_cumulative_contribution(variants)

    print("\nDone.")


if __name__ == "__main__":
    main()
