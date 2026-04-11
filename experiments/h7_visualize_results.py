#!/usr/bin/env python3
"""
H6 + H7 Combined Visualizations for Paper.

Generates:
  1. Full ablation bar chart (A–F) with error bars
  2. MixStyle delta comparison (Sinc vs UVMD)
  3. Per-subject F1 heatmap (E vs F)
  4. UVMD omega convergence across folds
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load results ─────────────────────────────────────────────────────
H6_DIR = PROJECT_ROOT / "experiments_output" / "h6_unified_ablation_20260312_130226"
H7_DIR = PROJECT_ROOT / "experiments_output" / "h7_uvmd_mixstyle"
OUT_DIR = PROJECT_ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

h6_comp = json.loads((H6_DIR / "comparison.json").read_text())
h7_comp = json.loads((H7_DIR / "comparison.json").read_text())

# Extract H6 per-variant data
h6_variants = {}
for v_name in ["A", "B", "C", "D"]:
    vf = H6_DIR / f"variant_{v_name}.json"
    if vf.exists():
        h6_variants[v_name] = json.loads(vf.read_text())

# H7 variants
h7_variants = {}
for v_name in ["E", "F"]:
    vf = H7_DIR / f"variant_{v_name}.json"
    if vf.exists():
        h7_variants[v_name] = json.loads(vf.read_text())

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

COLORS = {
    "A": "#9e9e9e",   # gray - baseline
    "B": "#42a5f5",   # blue - sinc
    "C": "#66bb6a",   # green - sinc+mixstyle
    "D": "#ef5350",   # red - sinc+mixstyle+CS
    "E": "#ab47bc",   # purple - UVMD
    "F": "#ff7043",   # orange - UVMD+mixstyle
}

LABELS = {
    "A": "Raw CNN\n(baseline)",
    "B": "Sinc FB\n(K=4)",
    "C": "Sinc FB\n+ MixStyle",
    "D": "Sinc FB\n+ MS + CS",
    "E": "UVMD\n(learnable)",
    "F": "UVMD\n+ MixStyle",
}

# ═════════════════════════════════════════════════════════════════════
#  Figure 1: Full Ablation Bar Chart (A–F)
# ═════════════════════════════════════════════════════════════════════
def plot_full_ablation():
    fig, ax = plt.subplots(figsize=(10, 5))

    variants = ["A", "B", "C", "D", "E", "F"]
    f1_means = []
    f1_stds = []

    for v in variants:
        if v in h6_variants:
            d = h6_variants[v]
        else:
            d = h7_variants[v]
        f1_means.append(d["mean_f1_macro"] * 100)
        f1_stds.append(d["std_f1_macro"] * 100)

    x = np.arange(len(variants))
    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=4,
                  color=[COLORS[v] for v in variants],
                  edgecolor="black", linewidth=0.5, width=0.6)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, f1_means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + f1_stds[i] + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Delta annotations
    baseline = f1_means[0]
    for i in range(1, len(variants)):
        delta = f1_means[i] - baseline
        ax.annotate(f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}",
                    xy=(x[i], f1_means[i] - 1.5),
                    ha="center", va="top", fontsize=8, color="white", fontweight="bold")

    # Grouping brackets
    ax.axvline(3.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(1.5, max(f1_means) + max(f1_stds) + 4, "Fixed Sinc FB",
            ha="center", fontsize=9, style="italic", color="gray")
    ax.text(4.5, max(f1_means) + max(f1_stds) + 4, "Learnable UVMD",
            ha="center", fontsize=9, style="italic", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in variants], fontsize=9)
    ax.set_ylabel("F1 Macro (%)")
    ax.set_title("Ablation Study: Cumulative Component Contributions (H6 + H7)")
    ax.set_ylim(25, max(f1_means) + max(f1_stds) + 7)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"fig_h6h7_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_h6h7_ablation.pdf/.png")


# ═════════════════════════════════════════════════════════════════════
#  Figure 2: MixStyle Delta Comparison (Sinc vs UVMD)
# ═════════════════════════════════════════════════════════════════════
def plot_mixstyle_delta():
    fig, ax = plt.subplots(figsize=(6, 4))

    frontends = ["Sinc FB\n(fixed)", "UVMD\n(learnable)"]
    without_ms = [
        h6_variants["B"]["mean_f1_macro"] * 100,
        h7_variants["E"]["mean_f1_macro"] * 100,
    ]
    with_ms = [
        h6_variants["C"]["mean_f1_macro"] * 100,
        h7_variants["F"]["mean_f1_macro"] * 100,
    ]
    deltas = [w - wo for w, wo in zip(with_ms, without_ms)]

    x = np.arange(len(frontends))
    w = 0.3

    bars1 = ax.bar(x - w/2, without_ms, w, label="Without MixStyle",
                   color=["#42a5f5", "#ab47bc"], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, with_ms, w, label="With MixStyle",
                   color=["#66bb6a", "#ff7043"], edgecolor="black", linewidth=0.5)

    # Delta arrows
    for i in range(len(frontends)):
        mid_x = x[i]
        ax.annotate(
            f"+{deltas[i]:.2f} pp",
            xy=(mid_x + w/2, with_ms[i]),
            xytext=(mid_x + 0.55, with_ms[i] + 1),
            fontsize=11, fontweight="bold", color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(frontends, fontsize=11)
    ax.set_ylabel("F1 Macro (%)")
    ax.set_title("Per-band MixStyle Effect Across Frontend Types")
    ax.set_ylim(30, max(with_ms) + 5)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"fig_mixstyle_delta.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_mixstyle_delta.pdf/.png")


# ═════════════════════════════════════════════════════════════════════
#  Figure 3: Per-subject F1 Heatmap (E vs F)
# ═════════════════════════════════════════════════════════════════════
def plot_per_subject_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    # Get subject order sorted by variant F F1
    subjects_f = {r["test_subject"]: r["f1_macro"] for r in h7_variants["F"]["per_subject"]}
    subjects_e = {r["test_subject"]: r["f1_macro"] for r in h7_variants["E"]["per_subject"]}
    sorted_subjects = sorted(subjects_f.keys(), key=lambda s: subjects_f[s], reverse=True)

    e_vals = [subjects_e[s] * 100 for s in sorted_subjects]
    f_vals = [subjects_f[s] * 100 for s in sorted_subjects]
    deltas = [f - e for f, e in zip(f_vals, e_vals)]

    # Left: grouped bar chart
    ax = axes[0]
    y = np.arange(len(sorted_subjects))
    h = 0.35

    ax.barh(y + h/2, e_vals, h, label="E: UVMD only", color=COLORS["E"],
            edgecolor="black", linewidth=0.3)
    ax.barh(y - h/2, f_vals, h, label="F: UVMD + MixStyle", color=COLORS["F"],
            edgecolor="black", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("DB2_", "") for s in sorted_subjects], fontsize=9)
    ax.set_xlabel("F1 Macro (%)")
    ax.set_title("Per-Subject F1: UVMD vs UVMD + MixStyle")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Right: delta bar
    ax2 = axes[1]
    colors_delta = ["#2e7d32" if d > 0 else "#c62828" for d in deltas]
    ax2.barh(y, deltas, 0.6, color=colors_delta, edgecolor="black", linewidth=0.3)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Δ F1 (pp)")
    ax2.set_title("MixStyle Effect")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    # Mean delta line
    mean_d = np.mean(deltas)
    ax2.axvline(mean_d, color="darkgreen", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.text(mean_d + 0.2, len(sorted_subjects) - 0.5,
             f"mean\n+{mean_d:.1f}", fontsize=8, color="darkgreen")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"fig_h7_per_subject.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_h7_per_subject.pdf/.png")


# ═════════════════════════════════════════════════════════════════════
#  Figure 4: UVMD Omega Convergence
# ═════════════════════════════════════════════════════════════════════
def plot_omega_convergence():
    fig, ax = plt.subplots(figsize=(8, 4))

    # Omega init vs learned for each fold
    init_omega = np.array(h7_variants["E"]["per_subject"][0]["init_omega_k"])
    K = len(init_omega)

    # Collect all folds for E and F
    omega_e = np.array([r["final_omega_k"] for r in h7_variants["E"]["per_subject"]])
    omega_f = np.array([r["final_omega_k"] for r in h7_variants["F"]["per_subject"]])

    mode_labels = [f"Mode {k+1}" for k in range(K)]
    x = np.arange(K)
    w = 0.2

    # Init
    ax.bar(x - w, init_omega, w, label="Initial", color="#bdbdbd",
           edgecolor="black", linewidth=0.5)

    # E learned
    ax.bar(x, omega_e.mean(axis=0), w,
           yerr=omega_e.std(axis=0), capsize=3,
           label="E: UVMD only", color=COLORS["E"],
           edgecolor="black", linewidth=0.5)

    # F learned
    ax.bar(x + w, omega_f.mean(axis=0), w,
           yerr=omega_f.std(axis=0), capsize=3,
           label="F: UVMD + MixStyle", color=COLORS["F"],
           edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.set_ylabel("Normalized Frequency (ω)")
    ax.set_title("UVMD Learned Mode Frequencies (Initial → Trained)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Frequency annotations
    fs = 2000  # Hz
    for k in range(K):
        freq_hz = omega_e.mean(axis=0)[k] * fs
        ax.text(x[k], omega_e.mean(axis=0)[k] + omega_e.std(axis=0)[k] + 0.01,
                f"~{freq_hz:.0f} Hz", ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"fig_uvmd_omega.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_uvmd_omega.pdf/.png")


# ═════════════════════════════════════════════════════════════════════
#  Figure 5: Summary comparison table figure
# ═════════════════════════════════════════════════════════════════════
def plot_summary_table():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    rows = [
        ["A", "Raw CNN (baseline)", "32.40%", "—", "H6"],
        ["B", "Sinc FB (K=4)", "36.08%", "+3.68 pp", "H6"],
        ["C", "Sinc FB + MixStyle", "37.19%", "+4.79 pp", "H6"],
        ["D", "Sinc FB + MS + CS heads", "36.37%", "+3.97 pp", "H6"],
        ["E", "UVMD (learnable)", "35.79%", "+3.39 pp", "H7"],
        ["F", "UVMD + MixStyle", "37.58%", "+5.18 pp", "H7"],
    ]

    col_labels = ["Variant", "Configuration", "F1 Macro", "Δ vs Raw", "Source"]
    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Color headers
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#37474f")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color best rows
    for i, row in enumerate(rows, start=1):
        v = row[0]
        table[i, 0].set_facecolor(COLORS.get(v, "white"))
        table[i, 0].set_text_props(color="white" if v in ["A", "D", "E"] else "black",
                                    fontweight="bold")
        if v == "F":
            for j in range(len(col_labels)):
                table[i, j].set_facecolor("#fff3e0")

    ax.set_title("H6 + H7: Ablation Study Summary\n(20 subjects, strict LOSO, NinaPro DB2 E1)",
                 fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"fig_ablation_summary_table.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_ablation_summary_table.pdf/.png")


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating H6+H7 paper figures...")
    plot_full_ablation()
    plot_mixstyle_delta()
    plot_per_subject_heatmap()
    plot_omega_convergence()
    plot_summary_table()
    print(f"\nAll figures saved to {OUT_DIR}")
