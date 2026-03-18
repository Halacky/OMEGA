#!/usr/bin/env python3
"""
Large-Scale Benchmark Statistical Analysis for Cross-Subject sEMG.

Performs:
  1. Collects per-subject F1 scores from all experiments
  2. Friedman test (non-parametric, repeated measures)
  3. Post-hoc Nemenyi test for pairwise comparisons
  4. Critical Difference (CD) diagram (Demsar, 2006)
  5. Taxonomy classification of methods
  6. Failure analysis

Usage:
  python scripts/benchmark_statistical_analysis.py
  python scripts/benchmark_statistical_analysis.py --top 20
  python scripts/benchmark_statistical_analysis.py --output_dir paper_figures/
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results_collected"
OUTPUT_DIR = ROOT / "paper_figures" / "benchmark"

# CI subjects used across all experiments
CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]


# ═════════════════════════════════════════════════════════════════════════
# Taxonomy of methods
# ═════════════════════════════════════════════════════════════════════════

TAXONOMY = {
    # Category -> list of experiment number prefixes
    "Baseline (CNN/GRU/LSTM)": [1, 2, 3, 4, 5, 6],
    "Signal Augmentation": [7, 8, 9, 10, 12, 16, 17, 18, 21, 24, 25],
    "Subject Adaptation": [11, 13, 14, 19, 20, 26, 28, 39, 45],
    "Attention / SE": [23, 85],
    "Architecture (Transformer/TCN/GAT)": [27, 29, 30, 37, 43, 52, 70, 72, 75],
    "Contrastive / Metric Learning": [15, 36, 73, 74, 79],
    "Disentanglement": [31, 47, 57, 58, 59, 60, 89, 99, 106],
    "Domain Generalization (DRO/IRM)": [48, 69, 80, 103, 104, 107],
    "Signal Decomposition (VMD/UVMD)": [82, 93, 95, 96, 108, 110],
    "Filterbank / Spectral": [49, 61, 64, 67, 68, 76, 84, 94, 97, 111],
    "Style Mixing / Normalization": [46, 53, 60, 91, 98, 100, 102, 105],
    "SSL Pretraining": [35, 42, 56],
    "Curriculum / Quality": [34, 40, 44],
    "ECAPA-TDNN": [62, 81, 88],
    "Deconvolution / Phase": [65, 66, 77, 78],
    "Hybrid / Fusion": [38, 50, 51, 54, 55, 63, 83, 86, 87, 90, 92, 109, 112],
}


def get_exp_number(exp_name: str) -> Optional[int]:
    """Extract experiment number from name like 'exp_106_...'."""
    m = re.match(r"exp_(\d+)", exp_name)
    return int(m.group(1)) if m else None


def classify_experiment(exp_name: str) -> str:
    """Classify experiment into taxonomy category."""
    num = get_exp_number(exp_name)
    if num is None:
        return "Other"
    for category, numbers in TAXONOMY.items():
        if num in numbers:
            return category
    return "Other"


# ═════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════

def _extract_per_subject_f1(data: dict) -> Dict[str, float]:
    """Extract per-subject F1 scores from any known JSON format.

    Supported formats:
      1. results (list) + aggregate — exp_100+ style
      2. aggregate_results + individual_results — exp_7-25 style
      3. loso_metrics.per_subject — exp_62, exp_75, etc.
      4. per_subject_f1 (dict) — exp_91, exp_92
      5. aggregate + per_fold — exp_73, exp_74
      6. per_subject (list) — H2 style
      7. results_by_frontend — exp_76 (multi-variant, pick best)
    """
    per_subj: Dict[str, float] = {}

    def _extract_from_list(lst):
        out = {}
        for r in lst:
            if not isinstance(r, dict):
                continue
            subj = r.get("test_subject")
            f1 = r.get("test_f1_macro")
            if subj and f1 is not None and isinstance(f1, (int, float)):
                out[subj] = float(f1)
        return out

    # Format 1: results as list (exp_100+ style)
    results = data.get("results", [])
    if isinstance(results, list) and results:
        per_subj = _extract_from_list(results)
        if per_subj:
            return per_subj

    # Format 2: individual_results (exp_7-25 style)
    ind_results = data.get("individual_results", [])
    if isinstance(ind_results, list) and ind_results:
        per_subj = _extract_from_list(ind_results)
        if per_subj:
            return per_subj

    # Format 3: loso_metrics.per_subject (exp_62, exp_75, etc.)
    loso_metrics = data.get("loso_metrics", {})
    if isinstance(loso_metrics, dict):
        lm_per_subj = loso_metrics.get("per_subject", [])
        if isinstance(lm_per_subj, list) and lm_per_subj:
            per_subj = _extract_from_list(lm_per_subj)
            if per_subj:
                return per_subj

    # Format 4: per_subject_f1 dict (exp_91, exp_92)
    psf1 = data.get("per_subject_f1", {})
    if isinstance(psf1, dict) and psf1:
        for subj, f1 in psf1.items():
            if f1 is not None and isinstance(f1, (int, float)):
                per_subj[subj] = float(f1)
        if per_subj:
            return per_subj

    # Format 5: per_fold (exp_73, exp_74)
    per_fold = data.get("per_fold", [])
    if isinstance(per_fold, list) and per_fold:
        per_subj = _extract_from_list(per_fold)
        if per_subj:
            return per_subj

    # Format 6: per_subject list (H2 style)
    per_subject_list = data.get("per_subject", [])
    if isinstance(per_subject_list, list) and per_subject_list:
        per_subj = _extract_from_list(per_subject_list)
        if per_subj:
            return per_subj

    # Format 7: fold_results (exp_98)
    fold_results = data.get("fold_results", [])
    if isinstance(fold_results, list) and fold_results:
        per_subj = _extract_from_list(fold_results)
        if per_subj:
            return per_subj

    # Format 8: results_by_frontend (exp_76 — multi-variant, pick best aggregate)
    rbf = data.get("results_by_frontend", {})
    if isinstance(rbf, dict) and rbf:
        best_f1 = -1
        best_subj = {}
        agg = data.get("aggregate", {})
        for frontend, frontend_results in rbf.items():
            if isinstance(frontend_results, list):
                candidate = _extract_from_list(frontend_results)
                if candidate:
                    candidate_mean = np.mean(list(candidate.values()))
                    if candidate_mean > best_f1:
                        best_f1 = candidate_mean
                        best_subj = candidate
        if best_subj:
            return best_subj

    return per_subj


def _get_mean_f1(data: dict, per_subj: Dict[str, float]) -> float:
    """Get mean F1 from aggregate fields or compute from per-subject."""
    # Try various aggregate locations
    for key in ["aggregate", "aggregate_results"]:
        agg = data.get(key, {})
        if isinstance(agg, dict) and "mean_f1_macro" in agg:
            return float(agg["mean_f1_macro"])

    loso_metrics = data.get("loso_metrics", {})
    if isinstance(loso_metrics, dict) and "mean_f1_macro" in loso_metrics:
        return float(loso_metrics["mean_f1_macro"])

    if "mean_f1_macro" in data:
        return float(data["mean_f1_macro"])

    # Compute from per-subject
    ci_f1 = [per_subj[s] for s in CI_SUBJECTS if s in per_subj]
    return float(np.mean(ci_f1)) if ci_f1 else 0.0


def load_all_results() -> Dict[str, Dict]:
    """Load per-subject F1 from all experiments."""
    experiments = {}
    skipped = {"no_json": 0, "no_per_subj": 0, "missing_ci": 0}

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_path = exp_dir / "loso_summary.json"
        if not summary_path.exists():
            skipped["no_json"] += 1
            continue

        try:
            with open(summary_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception):
            continue

        # Use experiment field or dir name
        exp_name = data.get("experiment", data.get("experiment_name", exp_dir.name))

        per_subj = _extract_per_subject_f1(data)
        if not per_subj:
            skipped["no_per_subj"] += 1
            continue

        if not all(s in per_subj for s in CI_SUBJECTS):
            skipped["missing_ci"] += 1
            continue

        # Use dir name as key to avoid collisions
        short_name = exp_dir.name
        if short_name in experiments:
            old_mean = np.mean([experiments[short_name]["per_subject"][s] for s in CI_SUBJECTS])
            new_mean = np.mean([per_subj[s] for s in CI_SUBJECTS])
            if new_mean <= old_mean:
                continue

        mean_f1 = _get_mean_f1(data, per_subj)

        experiments[short_name] = {
            "per_subject": per_subj,
            "mean_f1": float(mean_f1),
            "category": classify_experiment(short_name),
            "dir": str(exp_dir),
        }

    # Also try loading from per-fold subdirectories (exp_39, exp_45 style)
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        short_name = exp_dir.name
        if short_name in experiments:
            continue
        summary_path = exp_dir / "loso_summary.json"
        if summary_path.exists():
            continue  # Already handled above

        # Check for per-fold subdirs: DB2_sN/fold_result.json or fold_DB2_sN/fold_result.json
        per_subj: Dict[str, float] = {}
        for sub in exp_dir.iterdir():
            if not sub.is_dir():
                continue
            fold_json = sub / "fold_result.json"
            if not fold_json.exists():
                continue
            try:
                with open(fold_json) as f:
                    fold_data = json.load(f)
                subj = fold_data.get("test_subject", sub.name.replace("fold_", ""))
                f1 = fold_data.get("test_f1_macro")
                if subj and f1 is not None:
                    per_subj[subj] = float(f1)
            except Exception:
                pass

        if per_subj and all(s in per_subj for s in CI_SUBJECTS):
            experiments[short_name] = {
                "per_subject": per_subj,
                "mean_f1": float(np.mean([per_subj[s] for s in CI_SUBJECTS])),
                "category": classify_experiment(short_name),
                "dir": str(exp_dir),
            }

    print(f"  Loaded {len(experiments)} experiments")
    print(f"  Skipped: {skipped}")
    return experiments


def build_score_matrix(experiments: Dict[str, Dict]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build (n_experiments, n_subjects) F1 score matrix."""
    exp_names = sorted(experiments.keys(), key=lambda e: experiments[e]["mean_f1"], reverse=True)
    n_exp = len(exp_names)
    n_subj = len(CI_SUBJECTS)

    matrix = np.zeros((n_exp, n_subj))
    for i, name in enumerate(exp_names):
        for j, subj in enumerate(CI_SUBJECTS):
            matrix[i, j] = experiments[name]["per_subject"][subj]

    return matrix, exp_names, CI_SUBJECTS


# ═════════════════════════════════════════════════════════════════════════
# Friedman test + Nemenyi post-hoc
# ═════════════════════════════════════════════════════════════════════════

def friedman_test(matrix: np.ndarray) -> Tuple[float, float]:
    """Friedman test on (n_methods, n_subjects) matrix.
    Tests H0: all methods perform equally.
    """
    # scipy.stats.friedmanchisquare takes each method's scores as separate args
    args = [matrix[i, :] for i in range(matrix.shape[0])]
    stat, pval = stats.friedmanchisquare(*args)
    return float(stat), float(pval)


def compute_ranks(matrix: np.ndarray) -> np.ndarray:
    """Compute per-subject ranks (1=best). Returns (n_methods, n_subjects)."""
    n_methods, n_subjects = matrix.shape
    ranks = np.zeros_like(matrix)
    for j in range(n_subjects):
        # Rank descending (higher F1 = lower rank number)
        order = np.argsort(-matrix[:, j])
        for rank, idx in enumerate(order):
            ranks[idx, j] = rank + 1
    return ranks


def nemenyi_cd(n_methods: int, n_subjects: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi critical difference.
    CD = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_subjects))

    q_alpha values from Demsar (2006) Table 5, using Studentized Range / sqrt(2).
    """
    # q_alpha for alpha=0.05, computed from studentized range distribution
    # For large k, approximate using: q_alpha ≈ stats.norm.ppf(1 - alpha / (k*(k-1)))
    # But for exact values, use the formula from scipy
    from scipy.stats import studentized_range
    q = studentized_range.ppf(1 - alpha, n_methods, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_subjects))
    return cd


# ═════════════════════════════════════════════════════════════════════════
# Critical Difference Diagram
# ═════════════════════════════════════════════════════════════════════════

def plot_cd_diagram(avg_ranks: np.ndarray, names: List[str], cd: float,
                    n_methods: int, output_path: Path, title: str = ""):
    """Plot Critical Difference diagram (Demsar 2006 style)."""
    n = len(names)
    # Sort by average rank
    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.3), max(4, n * 0.18 + 2)))

    # Axis: rank scale
    low_rank = 1
    high_rank = n
    ax.set_xlim(low_rank - 0.5, high_rank + 0.5)
    ax.set_ylim(0, n + 1)

    # Draw rank axis at top
    ax.hlines(n + 0.5, low_rank, high_rank, color="black", linewidth=1)
    for r in range(1, n + 1, max(1, n // 10)):
        ax.vlines(r, n + 0.3, n + 0.7, color="black", linewidth=0.5)
        ax.text(r, n + 0.8, str(r), ha="center", va="bottom", fontsize=8)

    # CD bar
    cd_x = low_rank + 1
    ax.hlines(n + 1.2, cd_x, cd_x + cd, color="red", linewidth=2)
    ax.text(cd_x + cd / 2, n + 1.4, f"CD={cd:.2f}", ha="center", fontsize=9,
            color="red", fontweight="bold")

    # Place methods: left half on left, right half on right
    half = n // 2
    left_methods = list(range(half))
    right_methods = list(range(half, n))

    y_positions = {}
    # Left side (best methods) — evenly spaced
    for i, idx in enumerate(left_methods):
        y = n - i * (n / (half + 1))
        y_positions[idx] = y

    # Right side (worst methods) — evenly spaced
    for i, idx in enumerate(right_methods):
        y = n - i * (n / (len(right_methods) + 1))
        y_positions[idx] = y

    # Draw method lines and labels
    for idx in range(n):
        rank = sorted_ranks[idx]
        name = sorted_names[idx]
        y = n - idx * 0.9 - 0.5

        # Line from rank to label
        ax.plot([rank, rank], [y, y + 0.2], color="black", linewidth=0.8)
        ax.plot(rank, y, 'ko', markersize=3)

        # Label
        if idx < half:
            ax.text(rank - 0.3, y, f"{name} ({rank:.1f})", ha="right", va="center",
                    fontsize=7)
        else:
            ax.text(rank + 0.3, y, f"({rank:.1f}) {name}", ha="left", va="center",
                    fontsize=7)

    # Draw cliques (groups with no significant difference)
    # Methods whose rank difference < CD are not significantly different
    cliques = []
    for i in range(n):
        for j in range(i + 1, n):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                # Check if this extends an existing clique
                merged = False
                for clique in cliques:
                    if i in clique:
                        clique.add(j)
                        merged = True
                        break
                if not merged:
                    cliques.append({i, j})

    # Merge overlapping cliques
    merged_cliques = []
    for clique in cliques:
        merged = False
        for mc in merged_cliques:
            if mc & clique:
                mc |= clique
                merged = True
                break
        if not merged:
            merged_cliques.append(clique)

    # Draw clique bars
    colors_clique = ["#2171b5", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f"]
    for ci, clique in enumerate(merged_cliques):
        if len(clique) < 2:
            continue
        members = sorted(clique)
        lo_rank = sorted_ranks[members[0]]
        hi_rank = sorted_ranks[members[-1]]
        y_bar = -0.3 - ci * 0.4
        color = colors_clique[ci % len(colors_clique)]
        ax.hlines(y_bar, lo_rank, hi_rank, color=color, linewidth=3, alpha=0.6)

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Taxonomy summary figure
# ═════════════════════════════════════════════════════════════════════════

def plot_taxonomy_boxplot(experiments: Dict[str, Dict], output_path: Path):
    """Box plot of F1 scores grouped by taxonomy category."""
    cat_scores = defaultdict(list)
    for name, exp in experiments.items():
        cat = exp["category"]
        cat_scores[cat].append(exp["mean_f1"] * 100)

    # Sort by median F1 descending
    cats = sorted(cat_scores.keys(), key=lambda c: np.median(cat_scores[c]), reverse=True)
    data = [cat_scores[c] for c in cats]
    labels = [f"{c} (n={len(cat_scores[c])})" for c in cats]

    fig, ax = plt.subplots(figsize=(10, max(6, len(cats) * 0.4)))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("F1 Macro (%)", fontsize=11)
    ax.set_title("Distribution of F1 Scores by Method Category\n(5 CI subjects, strict LOSO)", fontsize=12)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_analysis(experiments: Dict[str, Dict], output_path: Path):
    """Scatter: mean F1 vs std F1 across subjects, colored by category."""
    fig, ax = plt.subplots(figsize=(10, 7))

    cat_colors = {}
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0

    for name, exp in sorted(experiments.items(), key=lambda x: x[1]["mean_f1"]):
        cat = exp["category"]
        if cat not in cat_colors:
            cat_colors[cat] = cmap[color_idx % 20]
            color_idx += 1

        f1s = [exp["per_subject"][s] for s in CI_SUBJECTS]
        mean_f1 = np.mean(f1s) * 100
        std_f1 = np.std(f1s) * 100

        ax.scatter(mean_f1, std_f1, color=cat_colors[cat], s=30, alpha=0.7,
                   edgecolors="black", linewidth=0.3)

    # Add category legend (top categories only)
    top_cats = sorted(cat_colors.keys(),
                      key=lambda c: max(experiments[n]["mean_f1"] for n, e in experiments.items()
                                        if e["category"] == c),
                      reverse=True)[:10]
    handles = [mpatches.Patch(color=cat_colors[c], label=c) for c in top_cats]
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=1)

    ax.set_xlabel("Mean F1 Macro (%)", fontsize=11)
    ax.set_ylabel("Std F1 Macro (%) across subjects", fontsize=11)
    ax.set_title("Mean vs Variability of F1 Scores\n(lower std = more consistent)", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark statistical analysis")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top methods for CD diagram")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for Nemenyi")
    args, _ = parser.parse_known_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BENCHMARK STATISTICAL ANALYSIS")
    print("=" * 70)

    # Load all results
    experiments = load_all_results()
    print(f"\nLoaded {len(experiments)} experiments with complete CI subject results")

    # Build score matrix
    matrix, exp_names, subjects = build_score_matrix(experiments)
    print(f"Score matrix: {matrix.shape} (methods x subjects)")

    # Category distribution
    cats = defaultdict(int)
    for name in exp_names:
        cats[experiments[name]["category"]] += 1
    print(f"\nCategories ({len(cats)}):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        cat_f1s = [experiments[n]["mean_f1"] * 100 for n in exp_names
                   if experiments[n]["category"] == cat]
        print(f"  {cat:45s}: n={count:2d}, best F1={max(cat_f1s):.1f}%, "
              f"median={np.median(cat_f1s):.1f}%")

    # ── Top-N summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TOP-{args.top} METHODS BY MEAN F1")
    print("=" * 70)
    print(f"{'Rank':>4s} {'Experiment':50s} {'F1':>6s} {'Category'}")
    print("-" * 90)
    for i, name in enumerate(exp_names[:args.top]):
        f1 = experiments[name]["mean_f1"] * 100
        cat = experiments[name]["category"]
        print(f"{i+1:4d} {name:50s} {f1:5.2f}% {cat}")

    # ── Friedman test ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FRIEDMAN TEST")
    print("=" * 70)

    # Need at least 3 methods
    if len(exp_names) >= 3:
        chi2, p_friedman = friedman_test(matrix)
        print(f"  Chi-square statistic: {chi2:.2f}")
        print(f"  p-value: {p_friedman:.2e}")
        print(f"  Conclusion: {'REJECT H0 — methods differ significantly' if p_friedman < args.alpha else 'FAIL TO REJECT H0'}")
    else:
        print("  Not enough methods for Friedman test")
        p_friedman = 1.0

    # ── Ranks ──────────────────────────────────────────────────────
    ranks = compute_ranks(matrix)
    avg_ranks = ranks.mean(axis=1)

    print(f"\n{'='*70}")
    print(f"AVERAGE RANKS (top-{args.top})")
    print("=" * 70)
    rank_order = np.argsort(avg_ranks)
    for i in range(min(args.top, len(exp_names))):
        idx = rank_order[i]
        name = exp_names[idx]
        f1 = experiments[name]["mean_f1"] * 100
        print(f"  Rank {avg_ranks[idx]:5.1f}: {name:50s} (F1={f1:.2f}%)")

    # ── Nemenyi CD ─────────────────────────────────────────────────
    if p_friedman < args.alpha:
        print(f"\n{'='*70}")
        print(f"NEMENYI POST-HOC (alpha={args.alpha})")
        print("=" * 70)

        n_methods_cd = min(args.top, len(exp_names))
        try:
            cd = nemenyi_cd(n_methods_cd, len(subjects), alpha=args.alpha)
            print(f"  Critical Difference (CD): {cd:.2f}")
            print(f"  Methods within CD of the best are NOT significantly different")

            # List cliques
            top_ranks = avg_ranks[rank_order[:n_methods_cd]]
            best_rank = top_ranks[0]
            print(f"\n  Methods NOT significantly different from the best (rank {best_rank:.1f}):")
            for i in range(n_methods_cd):
                idx = rank_order[i]
                if top_ranks[i] - best_rank < cd:
                    f1 = experiments[exp_names[idx]]["mean_f1"] * 100
                    print(f"    {exp_names[idx]:50s} rank={avg_ranks[idx]:.1f} F1={f1:.2f}%")
                else:
                    break

            # CD diagram for top-N
            print(f"\n  Generating CD diagram for top-{n_methods_cd}...")
            top_names = [exp_names[rank_order[i]] for i in range(n_methods_cd)]
            # Shorten names for display
            short_names = []
            for n in top_names:
                s = n.replace("exp_", "").replace("_loso", "")
                # Keep first 40 chars
                if len(s) > 40:
                    s = s[:40] + "..."
                short_names.append(s)

            top_avg_ranks = np.array([avg_ranks[rank_order[i]] for i in range(n_methods_cd)])

            plot_cd_diagram(
                top_avg_ranks, short_names, cd, n_methods_cd,
                output_dir / "cd_diagram_top",
                title=f"Critical Difference Diagram (top-{n_methods_cd}, alpha={args.alpha})"
            )
            print(f"  Saved cd_diagram_top.png/pdf")

        except Exception as e:
            print(f"  Nemenyi computation failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Taxonomy boxplot ───────────────────────────────────────────
    print(f"\n  Generating taxonomy boxplot...")
    plot_taxonomy_boxplot(experiments, output_dir / "taxonomy_boxplot")
    print(f"  Saved taxonomy_boxplot.png/pdf")

    # ── Failure analysis scatter ───────────────────────────────────
    print(f"\n  Generating failure analysis scatter...")
    plot_failure_analysis(experiments, output_dir / "failure_scatter")
    print(f"  Saved failure_scatter.png/pdf")

    # ── Failure cases ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FAILURE ANALYSIS (F1 < 20%)")
    print("=" * 70)
    failures = [(n, e) for n, e in experiments.items() if e["mean_f1"] < 0.20]
    failures.sort(key=lambda x: x[1]["mean_f1"])
    for name, exp in failures:
        f1 = exp["mean_f1"] * 100
        cat = exp["category"]
        print(f"  {name:55s} F1={f1:5.2f}% [{cat}]")

    print(f"\n  Total failures: {len(failures)}/{len(experiments)} "
          f"({100*len(failures)/len(experiments):.0f}%)")

    # ── LOSO audit violations ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXPERIMENTS WITH ANOMALOUS PATTERNS (potential LOSO violations)")
    print("=" * 70)
    for name in exp_names:
        exp = experiments[name]
        f1s = np.array([exp["per_subject"][s] for s in CI_SUBJECTS])
        # Flag if any subject has F1 > 0.8 (suspiciously high for cross-subject)
        if np.max(f1s) > 0.8:
            print(f"  {name}: max per-subject F1={np.max(f1s):.3f} (suspiciously high)")
        # Flag if accuracy >> F1 (class imbalance issue)
        # (would need accuracy data for this check)

    # ── Category-level rank table ──────────────────────────────────
    print(f"\n{'='*70}")
    print("CATEGORY-LEVEL SUMMARY (best method per category)")
    print("=" * 70)
    cat_best = {}
    for name in exp_names:
        cat = experiments[name]["category"]
        f1 = experiments[name]["mean_f1"]
        if cat not in cat_best or f1 > cat_best[cat]["f1"]:
            cat_best[cat] = {"name": name, "f1": f1, "rank": float(avg_ranks[exp_names.index(name)])}
    for cat, info in sorted(cat_best.items(), key=lambda x: x[1]["rank"]):
        print(f"  {cat:45s}: F1={info['f1']*100:5.2f}% rank={info['rank']:5.1f} ({info['name'][:40]})")

    # ── Per-subject breakdown for top-10 ─────────────────────────
    print(f"\n{'='*70}")
    print(f"PER-SUBJECT F1 FOR TOP-10 (% macro)")
    print("=" * 70)
    header = f"{'Method':40s}" + "".join(f" {s.replace('DB2_',''):>6s}" for s in CI_SUBJECTS) + f" {'Mean':>6s}"
    print(header)
    print("-" * len(header))
    for i in range(min(10, len(exp_names))):
        idx = rank_order[i]
        name = exp_names[idx][:40]
        f1s = [experiments[exp_names[idx]]["per_subject"][s] * 100 for s in CI_SUBJECTS]
        row = f"{name:40s}" + "".join(f" {f1:6.1f}" for f1 in f1s) + f" {np.mean(f1s):6.1f}"
        print(row)

    # ── Save summary JSON ──────────────────────────────────────────
    summary = {
        "n_experiments": len(experiments),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "friedman_chi2": float(chi2) if len(exp_names) >= 3 else None,
        "friedman_pval": float(p_friedman) if len(exp_names) >= 3 else None,
        "top_methods": [
            {
                "rank": i + 1,
                "name": exp_names[rank_order[i]],
                "avg_rank": float(avg_ranks[rank_order[i]]),
                "mean_f1": float(experiments[exp_names[rank_order[i]]]["mean_f1"]),
                "category": experiments[exp_names[rank_order[i]]]["category"],
            }
            for i in range(min(args.top, len(exp_names)))
        ],
        "categories": {
            cat: {
                "count": count,
                "best_f1": float(max(experiments[n]["mean_f1"] for n in exp_names
                                     if experiments[n]["category"] == cat)),
                "median_f1": float(np.median([experiments[n]["mean_f1"] for n in exp_names
                                              if experiments[n]["category"] == cat])),
            }
            for cat, count in cats.items()
        },
        "n_failures": len(failures),
    }

    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'benchmark_summary.json'}")
    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
