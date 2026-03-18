#!/usr/bin/env python3
"""
H3 Ablation: Visualization of style normalization variant comparison.
Generates paper-quality figures from H3 experiment results.

Figures:
  1. Bar chart: F1 macro +/- std across variants
  2. Per-subject heatmap: F1 by variant x subject
  3. Learned gamma_k for adaptive_in (if available)
  4. Delta F1 (per-band vs global) per subject — shows consistency of improvement
"""

import json
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent


def find_latest_h3_dir():
    output_root = ROOT / "experiments_output"
    dirs = sorted(output_root.glob("h3_style_normalization_*"), reverse=True)
    if not dirs:
        print("ERROR: No H3 output directory found")
        sys.exit(1)
    return dirs[0]


def load_results(h3_dir: Path):
    variants_order = ["baseline", "global_in", "per_band_in", "per_band_mix", "adaptive_in"]
    results = {}
    for v in variants_order:
        summary_path = h3_dir / v / "loso_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results[v] = json.load(f)
    return results


VARIANT_LABELS = {
    "baseline": "Baseline\n(no style norm)",
    "global_in": "Global\nInstanceNorm",
    "per_band_in": "Per-Band\nInstanceNorm",
    "per_band_mix": "Per-Band\nMixStyle",
    "adaptive_in": "Adaptive\nPer-Band IN",
}

VARIANT_COLORS = {
    "baseline": "#b0b0b0",
    "global_in": "#fdae61",
    "per_band_in": "#2171b5",
    "per_band_mix": "#6baed6",
    "adaptive_in": "#08519c",
}


def fig1_f1_bar_chart(results, output_dir):
    variants = [v for v in ["baseline", "global_in", "per_band_in", "per_band_mix", "adaptive_in"]
                if v in results]

    means = [results[v]["results"]["mean_f1_macro"] * 100 for v in variants]
    stds = [results[v]["results"]["std_f1_macro"] * 100 for v in variants]
    labels = [VARIANT_LABELS[v] for v in variants]
    colors = [VARIANT_COLORS[v] for v in variants]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.6,
                  error_kw={"linewidth": 1.5})

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("F1 Macro (%)", fontsize=12)
    ax.set_title("H3: Style Normalization Ablation\n(Fixed Sinc Filterbank + varying norm, LOSO)", fontsize=13)
    ax.set_ylim(0, max(means) + max(stds) + 5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate best per-band vs baseline delta
    baseline_f1 = means[variants.index("baseline")] if "baseline" in variants else 0
    best_v = max(variants, key=lambda v: results[v]["results"]["mean_f1_macro"])
    best_f1 = means[variants.index(best_v)]
    delta = best_f1 - baseline_f1
    if delta > 0:
        ax.annotate(f"+{delta:.1f} pp",
                    xy=(variants.index(best_v), best_f1),
                    xytext=(variants.index(best_v) + 0.4, best_f1 + 3),
                    fontsize=10, color="#2171b5", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#2171b5", lw=1.5))

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"h3_f1_comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h3_f1_comparison.png/pdf")


def fig2_per_subject_heatmap(results, output_dir):
    variants = [v for v in ["baseline", "global_in", "per_band_in", "per_band_mix", "adaptive_in"]
                if v in results]
    subjects = [s["test_subject"] for s in results[variants[0]]["per_subject"]
                if s["test_f1_macro"] is not None]

    data = np.zeros((len(subjects), len(variants)))
    for j, v in enumerate(variants):
        for i, s in enumerate(results[v]["per_subject"]):
            if s["test_f1_macro"] is not None:
                data[i, j] = s["test_f1_macro"] * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=10, vmax=50)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS[v].replace("\n", " ") for v in variants],
                       fontsize=9, rotation=15, ha="right")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=9)

    for i in range(len(subjects)):
        for j in range(len(variants)):
            val = data[i, j]
            color = "white" if val > 35 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_title("H3: Per-Subject F1 Macro by Style Normalization Variant", fontsize=12)
    fig.colorbar(im, ax=ax, label="F1 Macro (%)", shrink=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"h3_per_subject_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h3_per_subject_heatmap.png/pdf")


def fig3_adaptive_gamma(results, output_dir):
    if "adaptive_in" not in results:
        print("  Skipping gamma_k plot (no adaptive_in results)")
        return

    per_subj = results["adaptive_in"]["per_subject"]
    gammas = [s["learned_params"]["gamma_k"] for s in per_subj
              if s.get("learned_params")]
    if not gammas:
        print("  Skipping gamma_k plot (no learned params)")
        return

    arr = np.array(gammas)  # (n_folds, K)
    K = arr.shape[1]
    subjects = [s["test_subject"].replace("DB2_", "") for s in per_subj
                if s.get("learned_params")]

    band_labels = ["20-265 Hz", "265-510 Hz", "510-755 Hz", "755-1000 Hz"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for k in range(K):
        label = band_labels[k] if k < len(band_labels) else f"band_{k}"
        vals = arr[:, k]
        mean_v = np.mean(vals)
        ax.scatter(range(len(subjects)), vals, color=colors[k],
                   s=50, zorder=3, edgecolors="black", linewidth=0.5, label=label)
        ax.axhline(mean_v, color=colors[k], linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Learned IN Strength (gamma_k)", fontsize=11)
    ax.set_xlabel("Test Subject (LOSO fold)", fontsize=11)
    ax.set_title("H3: Adaptive Per-Band IN Strength Across LOSO Folds", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"h3_adaptive_gamma.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h3_adaptive_gamma.png/pdf")


def fig4_delta_per_subject(results, output_dir):
    """Bar chart: F1 delta (best per-band - baseline) per subject."""
    if "baseline" not in results or "per_band_in" not in results:
        print("  Skipping delta plot (missing baseline or per_band_in)")
        return

    subjects = [s["test_subject"] for s in results["baseline"]["per_subject"]
                if s["test_f1_macro"] is not None]
    baseline_f1 = np.array([s["test_f1_macro"] for s in results["baseline"]["per_subject"]
                            if s["test_f1_macro"] is not None])

    # Find best per-band variant per subject
    per_band_variants = [v for v in ["per_band_in", "per_band_mix", "adaptive_in"]
                         if v in results]

    best_perband_f1 = baseline_f1.copy()
    for v in per_band_variants:
        v_f1 = np.array([s["test_f1_macro"] for s in results[v]["per_subject"]
                         if s["test_f1_macro"] is not None])
        best_perband_f1 = np.maximum(best_perband_f1, v_f1)

    deltas = (best_perband_f1 - baseline_f1) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2171b5" if d > 0 else "#d62728" for d in deltas]
    short_names = [s.replace("DB2_", "") for s in subjects]
    ax.bar(range(len(subjects)), deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(short_names, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("F1 Delta (pp)", fontsize=11)
    ax.set_title("H3: Per-Subject F1 Improvement (Best Per-Band Norm vs Baseline)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_better = np.sum(deltas > 0)
    ax.text(0.98, 0.95, f"{n_better}/{len(subjects)} improved",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"h3_delta_per_subject.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h3_delta_per_subject.png/pdf")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    h3_dir = Path(args.results_dir) if args.results_dir else find_latest_h3_dir()
    output_dir = Path(args.output_dir) if args.output_dir else h3_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"H3 Visualization")
    print(f"  Results: {h3_dir}")
    print(f"  Output:  {output_dir}")
    print()

    results = load_results(h3_dir)
    print(f"  Loaded {len(results)} variants: {list(results.keys())}")
    print()

    fig1_f1_bar_chart(results, output_dir)
    fig2_per_subject_heatmap(results, output_dir)
    fig3_adaptive_gamma(results, output_dir)
    fig4_delta_per_subject(results, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
