#!/usr/bin/env python3
"""
H2 Ablation: Visualization of decomposition variant comparison.
Generates paper-quality figures from H2 experiment results.

Figures:
  1. Bar chart: F1 macro ± std across variants
  2. Per-subject heatmap: F1 by variant × subject
  3. Learned omega_k across LOSO folds (UVMD stability)
  4. Training time comparison
"""

import json
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

def find_latest_h2_dir():
    """Find most recent H2 output directory."""
    output_root = ROOT / "experiments_output"
    dirs = sorted(output_root.glob("h2_ablation_decomposition_*"), reverse=True)
    if not dirs:
        print("ERROR: No H2 output directory found")
        sys.exit(1)
    return dirs[0]


def load_results(h2_dir: Path):
    """Load all variant summaries."""
    variants_order = ["none", "fixed_fb", "uvmd", "uvmd_overlap"]
    results = {}
    for v in variants_order:
        summary_path = h2_dir / v / "loso_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results[v] = json.load(f)
    return results


# ── Nice labels ────────────────────────────────────────────────────────
VARIANT_LABELS = {
    "none": "Raw EMG\n(no decomposition)",
    "fixed_fb": "Fixed Sinc\nFilterbank",
    "uvmd": "UVMD\n(learnable)",
    "uvmd_overlap": "UVMD +\nOverlap Penalty",
}

VARIANT_COLORS = {
    "none": "#b0b0b0",
    "fixed_fb": "#6baed6",
    "uvmd": "#2171b5",
    "uvmd_overlap": "#08519c",
}


def fig1_f1_bar_chart(results, output_dir):
    """Bar chart comparing F1 macro across variants."""
    variants = [v for v in ["none", "fixed_fb", "uvmd", "uvmd_overlap"] if v in results]

    means = [results[v]["results"]["mean_f1_macro"] * 100 for v in variants]
    stds = [results[v]["results"]["std_f1_macro"] * 100 for v in variants]
    labels = [VARIANT_LABELS[v] for v in variants]
    colors = [VARIANT_COLORS[v] for v in variants]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.6,
                  error_kw={"linewidth": 1.5})

    # Value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("F1 Macro (%)", fontsize=12)
    ax.set_title("H2: Ablation of Frequency Decomposition Frontend\n(LOSO, E1 gestures 1-10)", fontsize=13)
    ax.set_ylim(0, max(means) + max(stds) + 5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Delta annotation
    raw_f1 = means[variants.index("none")] if "none" in variants else 0
    if "uvmd" in variants:
        uvmd_f1 = means[variants.index("uvmd")]
        delta = uvmd_f1 - raw_f1
        ax.annotate(f"+{delta:.1f} pp",
                    xy=(variants.index("uvmd"), uvmd_f1),
                    xytext=(variants.index("uvmd") + 0.4, uvmd_f1 + 3),
                    fontsize=10, color="#2171b5", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#2171b5", lw=1.5))

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"h2_f1_comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h2_f1_comparison.png/pdf")


def fig2_per_subject_heatmap(results, output_dir):
    """Heatmap: F1 per subject × variant."""
    variants = [v for v in ["none", "fixed_fb", "uvmd", "uvmd_overlap"] if v in results]
    subjects = [s["test_subject"] for s in results[variants[0]]["per_subject"]]

    data = np.zeros((len(subjects), len(variants)))
    for j, v in enumerate(variants):
        for i, s in enumerate(results[v]["per_subject"]):
            data[i, j] = s["test_f1_macro"] * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=10, vmax=45)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS[v].replace("\n", " ") for v in variants],
                       fontsize=9, rotation=15, ha="right")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=10)

    # Annotate cells
    for i in range(len(subjects)):
        for j in range(len(variants)):
            val = data[i, j]
            color = "white" if val > 30 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_title("H2: Per-Subject F1 Macro by Decomposition Variant", fontsize=12)
    fig.colorbar(im, ax=ax, label="F1 Macro (%)", shrink=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"h2_per_subject_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h2_per_subject_heatmap.png/pdf")


def fig3_omega_k_stability(results, output_dir):
    """Scatter plot of learned omega_k across LOSO folds."""
    uvmd_variants = [v for v in ["uvmd", "uvmd_overlap"] if v in results]
    if not uvmd_variants:
        print("  Skipping omega_k plot (no UVMD results)")
        return

    fig, axes = plt.subplots(1, len(uvmd_variants), figsize=(6 * len(uvmd_variants), 4.5),
                              squeeze=False)

    for ax_idx, variant in enumerate(uvmd_variants):
        ax = axes[0, ax_idx]
        per_subj = results[variant]["per_subject"]
        K = len(per_subj[0]["learned_params"]["omega_k"])
        subjects = [s["test_subject"].replace("DB2_", "") for s in per_subj]

        mode_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

        for k in range(K):
            omegas = [s["learned_params"]["omega_k"][k] for s in per_subj]
            # Convert normalized freq to Hz (fs=2000)
            omegas_hz = [w * 1000 for w in omegas]  # normalized [0,1] → [0, fs/2]
            mean_hz = np.mean(omegas_hz)
            std_hz = np.std(omegas_hz)

            ax.scatter(range(len(subjects)), omegas_hz, color=mode_colors[k],
                      s=60, zorder=3, edgecolors="black", linewidth=0.5)
            ax.axhline(mean_hz, color=mode_colors[k], linestyle="--", alpha=0.5, linewidth=1)
            ax.fill_between(range(len(subjects)),
                           mean_hz - std_hz, mean_hz + std_hz,
                           color=mode_colors[k], alpha=0.1)
            ax.text(len(subjects) - 0.5, mean_hz,
                    f"$\\omega_{k+1}$={mean_hz:.0f}Hz\n$\\pm${std_hz:.1f}",
                    fontsize=8, va="center", color=mode_colors[k])

        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, fontsize=9, rotation=45, ha="right")
        ax.set_ylabel("Center Frequency (Hz)", fontsize=11)
        ax.set_xlabel("Test Subject (LOSO fold)", fontsize=11)
        ax.set_title(f"Learned Mode Frequencies ({VARIANT_LABELS[variant].replace(chr(10), ' ')})",
                     fontsize=11)
        ax.set_ylim(0, 550)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"h2_omega_k_stability.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h2_omega_k_stability.png/pdf")


def fig4_time_comparison(results, output_dir):
    """Bar chart of training time per variant."""
    variants = [v for v in ["none", "fixed_fb", "uvmd", "uvmd_overlap"] if v in results]

    times = [results[v]["total_time_s"] / 60 for v in variants]  # minutes
    labels = [VARIANT_LABELS[v] for v in variants]
    colors = [VARIANT_COLORS[v] for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(variants))
    bars = ax.bar(x, times, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{t:.1f} min", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Total LOSO Time (minutes)", fontsize=12)
    ax.set_title("H2: Computational Cost by Decomposition Variant", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"h2_time_comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h2_time_comparison.png/pdf")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None,
                        help="H2 results directory (auto-detect if not given)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for figures (defaults to results_dir)")
    args, _ = parser.parse_known_args()

    if args.results_dir:
        h2_dir = Path(args.results_dir)
    else:
        h2_dir = find_latest_h2_dir()

    output_dir = Path(args.output_dir) if args.output_dir else h2_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"H2 Visualization")
    print(f"  Results: {h2_dir}")
    print(f"  Output:  {output_dir}")
    print()

    results = load_results(h2_dir)
    print(f"  Loaded {len(results)} variants: {list(results.keys())}")
    print()

    fig1_f1_bar_chart(results, output_dir)
    fig2_per_subject_heatmap(results, output_dir)
    fig3_omega_k_stability(results, output_dir)
    fig4_time_comparison(results, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
