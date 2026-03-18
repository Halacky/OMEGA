#!/usr/bin/env python3
"""
H3 Ablation: Statistical significance analysis.
Performs paired statistical tests between style normalization variants.
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def find_latest_h3_dir():
    output_root = ROOT / "experiments_output"
    dirs = sorted(output_root.glob("h3_style_normalization_*"), reverse=True)
    if not dirs:
        print("ERROR: No H3 output directory found")
        sys.exit(1)
    return dirs[0]


def load_per_subject_f1(h3_dir: Path):
    variants_order = ["baseline", "global_in", "per_band_in", "per_band_mix", "adaptive_in"]
    data = {}
    for v in variants_order:
        path = h3_dir / v / "loso_summary.json"
        if path.exists():
            with open(path) as f:
                summary = json.load(f)
            f1s = [s["test_f1_macro"] for s in summary["per_subject"]
                   if s["test_f1_macro"] is not None]
            subjects = [s["test_subject"] for s in summary["per_subject"]
                        if s["test_f1_macro"] is not None]
            data[v] = {"f1": np.array(f1s), "subjects": subjects}
    return data


def cohens_d(a, b):
    diff = a - b
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / sd if sd > 0 else 0.0


def bootstrap_ci(a, b, n_bootstrap=10000, alpha=0.05):
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


def run_analysis(h3_dir: Path):
    data = load_per_subject_f1(h3_dir)
    if not data:
        print("ERROR: No variant results found")
        sys.exit(1)

    n_subjects = len(next(iter(data.values()))["f1"])

    print(f"H3 Statistical Analysis")
    print(f"  Results dir: {h3_dir}")
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

    # Key pairwise comparisons
    pairs = [
        # Per-band vs global (main hypothesis)
        ("per_band_in", "global_in"),
        ("per_band_mix", "global_in"),
        ("adaptive_in", "global_in"),
        # Any style norm vs baseline
        ("per_band_in", "baseline"),
        ("global_in", "baseline"),
        ("per_band_mix", "baseline"),
        ("adaptive_in", "baseline"),
        # Per-band variants vs each other
        ("per_band_mix", "per_band_in"),
        ("adaptive_in", "per_band_in"),
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

        t_stat, t_pval = stats.ttest_rel(a, b)

        try:
            w_stat, w_pval = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        d = cohens_d(a, b)
        ci_lo, ci_hi = bootstrap_ci(a, b)

        n_better = np.sum(diff > 0)
        n_worse = np.sum(diff < 0)

        result = {
            "pair": f"{v1} vs {v2}",
            "mean_diff_pp": np.mean(diff) * 100,
            "median_diff_pp": np.median(diff) * 100,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_pval": float(w_pval) if not np.isnan(w_pval) else None,
            "cohens_d": d,
            "ci_95_lo": ci_lo * 100,
            "ci_95_hi": ci_hi * 100,
            "n_better": int(n_better),
            "n_worse": int(n_worse),
        }
        results_list.append(result)

        sig_w = "*" if (not np.isnan(w_pval) and w_pval < 0.05) else ""

        print(f"\n  {v1} vs {v2}")
        print(f"    Mean diff:    {np.mean(diff)*100:+.2f} pp  (median: {np.median(diff)*100:+.2f} pp)")
        print(f"    95% CI:       [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp")
        print(f"    Paired t:     t={t_stat:.3f}, p={t_pval:.4f}")
        if not np.isnan(w_pval):
            print(f"    Wilcoxon:     W={w_stat:.1f}, p={w_pval:.4f} {sig_w}")
        else:
            print(f"    Wilcoxon:     N/A")
        effect = 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'
        print(f"    Cohen's d:    {d:.3f} ({effect})")
        print(f"    Winners:      {n_better}/{n_subjects} better, {n_worse}/{n_subjects} worse")

    # Bonferroni correction
    m = len(results_list)
    print(f"\n{'='*70}")
    print(f"BONFERRONI-CORRECTED p-values (m={m} comparisons)")
    print("=" * 70)
    for r in results_list:
        wp = r["wilcoxon_pval"]
        if wp is not None:
            corr_w = min(wp * m, 1.0)
            sig = "***" if corr_w < 0.001 else "**" if corr_w < 0.01 else "*" if corr_w < 0.05 else "ns"
        else:
            corr_w = float("nan")
            sig = "N/A"
        print(f"  {r['pair']:35s}: Wilcoxon p_corr={corr_w:.4f} {sig}, "
              f"diff={r['mean_diff_pp']:+.2f} pp")

    # Check adaptive_in learned gamma_k
    adaptive_path = h3_dir / "adaptive_in" / "loso_summary.json"
    if adaptive_path.exists():
        with open(adaptive_path) as f:
            adaptive_data = json.load(f)
        gammas = [s["learned_params"]["gamma_k"] for s in adaptive_data["per_subject"]
                  if s.get("learned_params")]
        if gammas:
            arr = np.array(gammas)
            print(f"\n{'='*70}")
            print("ADAPTIVE IN — LEARNED gamma_k (IN strength per band)")
            print("=" * 70)
            band_labels = ["20-265 Hz", "265-510 Hz", "510-755 Hz", "755-1000 Hz"]
            for k in range(arr.shape[1]):
                label = band_labels[k] if k < len(band_labels) else f"band_{k}"
                print(f"  gamma_{k+1} ({label:12s}): "
                      f"mean={arr[:,k].mean():.3f} +/- {arr[:,k].std():.3f} "
                      f"(range [{arr[:,k].min():.3f}, {arr[:,k].max():.3f}])")
            print(f"\n  Interpretation: gamma=0 -> no IN, gamma=1 -> full IN")
            print(f"  H1 predicts: gamma should increase with band index (more norm for high-freq)")

    # Save
    output_path = h3_dir / "statistical_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_path}")

    return results_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    h3_dir = Path(args.results_dir) if args.results_dir else find_latest_h3_dir()
    run_analysis(h3_dir)


if __name__ == "__main__":
    main()
