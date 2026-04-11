#!/usr/bin/env python3
"""
H2 Ablation: Statistical significance analysis.
Performs paired statistical tests between decomposition variants.

Tests:
  - Paired Wilcoxon signed-rank test (non-parametric, n>=5)
  - Paired t-test (parametric, for reference)
  - Effect size (Cohen's d)
  - Confidence intervals for mean differences
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def find_latest_h2_dir():
    output_root = ROOT / "experiments_output"
    dirs = sorted(output_root.glob("h2_ablation_decomposition_*"), reverse=True)
    if not dirs:
        print("ERROR: No H2 output directory found")
        sys.exit(1)
    return dirs[0]


def load_per_subject_f1(h2_dir: Path):
    """Load per-subject F1 scores for each variant."""
    variants_order = ["none", "fixed_fb", "uvmd", "uvmd_overlap"]
    data = {}
    for v in variants_order:
        path = h2_dir / v / "loso_summary.json"
        if path.exists():
            with open(path) as f:
                summary = json.load(f)
            f1s = [s["test_f1_macro"] for s in summary["per_subject"]]
            subjects = [s["test_subject"] for s in summary["per_subject"]]
            data[v] = {"f1": np.array(f1s), "subjects": subjects}
    return data


def cohens_d(a, b):
    """Paired Cohen's d effect size."""
    diff = a - b
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0


def bootstrap_ci(a, b, n_bootstrap=10000, alpha=0.05):
    """Bootstrap confidence interval for mean difference."""
    rng = np.random.default_rng(42)
    diffs = a - b
    n = len(diffs)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


def run_analysis(h2_dir: Path):
    data = load_per_subject_f1(h2_dir)
    n_subjects = len(next(iter(data.values()))["f1"])

    print(f"H2 Statistical Analysis")
    print(f"  Results dir: {h2_dir}")
    print(f"  N subjects: {n_subjects}")
    print(f"  Variants: {list(data.keys())}")
    print()

    # Summary stats
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for v, d in data.items():
        f1 = d["f1"] * 100
        print(f"  {v:16s}: F1 = {np.mean(f1):5.2f} +/- {np.std(f1):5.2f}% "
              f"(median={np.median(f1):.2f}, min={np.min(f1):.2f}, max={np.max(f1):.2f})")
    print()

    # Pairwise comparisons
    pairs = [
        ("uvmd", "none"),         # Key: learnable vs raw
        ("uvmd", "fixed_fb"),     # Key: learnable vs fixed
        ("fixed_fb", "none"),     # Fixed vs raw
        ("uvmd", "uvmd_overlap"), # Effect of overlap penalty
    ]

    print("=" * 70)
    print("PAIRWISE COMPARISONS")
    print("=" * 70)
    results_list = []

    for v1, v2 in pairs:
        if v1 not in data or v2 not in data:
            continue

        a = data[v1]["f1"]
        b = data[v2]["f1"]
        diff = a - b

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(a, b)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pval = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        # Effect size
        d = cohens_d(a, b)

        # Bootstrap CI
        ci_lo, ci_hi = bootstrap_ci(a, b)

        # How many subjects improved
        n_better = np.sum(diff > 0)
        n_worse = np.sum(diff < 0)
        n_tie = np.sum(diff == 0)

        result = {
            "pair": f"{v1} vs {v2}",
            "mean_diff_pp": np.mean(diff) * 100,
            "median_diff_pp": np.median(diff) * 100,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pval": w_pval,
            "cohens_d": d,
            "ci_95_lo": ci_lo * 100,
            "ci_95_hi": ci_hi * 100,
            "n_better": int(n_better),
            "n_worse": int(n_worse),
            "n_tie": int(n_tie),
        }
        results_list.append(result)

        sig_t = "*" if t_pval < 0.05 else ""
        sig_w = "*" if w_pval < 0.05 else ""

        print(f"\n  {v1} vs {v2}")
        print(f"    Mean diff:    {np.mean(diff)*100:+.2f} pp  (median: {np.median(diff)*100:+.2f} pp)")
        print(f"    95% CI:       [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp")
        print(f"    Paired t:     t={t_stat:.3f}, p={t_pval:.4f} {sig_t}")
        print(f"    Wilcoxon:     W={w_stat:.1f}, p={w_pval:.4f} {sig_w}" if not np.isnan(w_pval)
              else f"    Wilcoxon:     N/A (ties)")
        print(f"    Cohen's d:    {d:.3f} ({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'})")
        print(f"    Winners:      {n_better}/{n_subjects} better, {n_worse}/{n_subjects} worse")

    # Bonferroni correction
    print(f"\n{'='*70}")
    print("BONFERRONI-CORRECTED p-values (m={len(results_list)} comparisons)")
    print("=" * 70)
    for r in results_list:
        corr_t = min(r["t_pval"] * len(results_list), 1.0)
        corr_w = min(r["wilcoxon_pval"] * len(results_list), 1.0) if not np.isnan(r["wilcoxon_pval"]) else np.nan
        sig = "***" if corr_w < 0.001 else "**" if corr_w < 0.01 else "*" if corr_w < 0.05 else "ns"
        print(f"  {r['pair']:25s}: Wilcoxon p_corr={corr_w:.4f} {sig}, "
              f"diff={r['mean_diff_pp']:+.2f} pp")

    # Save
    output_path = h2_dir / "statistical_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_path}")

    return results_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    h2_dir = Path(args.results_dir) if args.results_dir else find_latest_h2_dir()
    run_analysis(h2_dir)


if __name__ == "__main__":
    main()
