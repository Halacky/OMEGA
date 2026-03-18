#!/usr/bin/env python3
"""
Statistical significance analysis for H6 + H7 ablation results.

Tests:
  1. Wilcoxon signed-rank tests between consecutive variants (paired per-subject)
  2. Cohen's d effect sizes
  3. Bootstrap 95% confidence intervals for mean F1 differences
  4. Bonferroni-corrected p-values

Uses existing per-subject F1 data from variant JSON files — no GPU needed.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

H6_DIR = PROJECT_ROOT / "experiments_output" / "h6_unified_ablation_20260312_130226"
H7_DIR = PROJECT_ROOT / "experiments_output" / "h7_uvmd_mixstyle"
OUT_DIR = PROJECT_ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_per_subject_f1(variant_file: Path) -> dict:
    """Load per-subject F1 from variant JSON. Returns {subject: f1}."""
    data = json.loads(variant_file.read_text())
    return {
        r["test_subject"]: r["f1_macro"]
        for r in data["per_subject"]
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Cohen's d (within-subjects)."""
    diff = x - y
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000,
                  ci: float = 0.95, seed: int = 42) -> tuple:
    """Bootstrap CI for mean(x - y)."""
    rng = np.random.RandomState(seed)
    diff = x - y
    n = len(diff)
    boot_means = np.array([
        rng.choice(diff, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_means, alpha * 100)),
        float(np.percentile(boot_means, (1 - alpha) * 100)),
    )


# ── Load all variant data ────────────────────────────────────────────
print("Loading per-subject F1 data...")

variants = {}
for v_name in ["A", "B", "C", "D"]:
    f = H6_DIR / f"variant_{v_name}.json"
    if f.exists():
        variants[v_name] = load_per_subject_f1(f)
        print(f"  {v_name}: {len(variants[v_name])} subjects")

for v_name in ["E", "F"]:
    f = H7_DIR / f"variant_{v_name}.json"
    if f.exists():
        variants[v_name] = load_per_subject_f1(f)
        print(f"  {v_name}: {len(variants[v_name])} subjects")

# ── Define comparisons ───────────────────────────────────────────────
# Key comparisons for the paper
comparisons = [
    # Consecutive H6 ablation
    ("A", "B", "Decomposition effect (Sinc FB)"),
    ("B", "C", "MixStyle effect (on Sinc FB)"),
    ("C", "D", "Content/Style heads effect"),
    # UVMD
    ("E", "F", "MixStyle effect (on UVMD)"),
    # Cross-frontend
    ("B", "E", "Sinc FB vs UVMD (no MixStyle)"),
    ("C", "F", "Sinc+MixStyle vs UVMD+MixStyle"),
    # Baseline vs best
    ("A", "C", "Raw → Sinc+MixStyle (total H6 gain)"),
    ("A", "F", "Raw → UVMD+MixStyle (total H7 gain)"),
]

n_tests = len(comparisons)
bonferroni = n_tests

print(f"\n{'='*80}")
print(f"STATISTICAL ANALYSIS: {n_tests} comparisons, Bonferroni correction (k={bonferroni})")
print(f"{'='*80}\n")

results = []

for v1, v2, description in comparisons:
    if v1 not in variants or v2 not in variants:
        print(f"  SKIP: {v1} vs {v2} — missing data")
        continue

    # Get common subjects
    common = sorted(set(variants[v1].keys()) & set(variants[v2].keys()))
    n = len(common)

    x = np.array([variants[v1][s] for s in common])  # baseline
    y = np.array([variants[v2][s] for s in common])   # improved

    # Paired Wilcoxon signed-rank test (two-sided)
    stat_w, p_wilcoxon = stats.wilcoxon(x, y, alternative="two-sided")

    # One-sided: y > x (improvement)
    _, p_onesided = stats.wilcoxon(x, y, alternative="less")  # x < y

    # Bonferroni correction
    p_bonferroni = min(p_wilcoxon * bonferroni, 1.0)
    p_onesided_bonf = min(p_onesided * bonferroni, 1.0)

    # Effect size
    d = cohens_d(y, x)

    # Bootstrap CI
    ci_lo, ci_hi = bootstrap_ci(y, x)

    # Mean difference
    mean_diff = (y - x).mean()
    std_diff = (y - x).std()

    # How many subjects improved
    n_improved = int(np.sum(y > x))
    n_same = int(np.sum(y == x))
    n_worse = int(np.sum(y < x))

    result = {
        "comparison": f"{v1} → {v2}",
        "description": description,
        "n_subjects": n,
        "mean_f1_v1": float(x.mean()),
        "mean_f1_v2": float(y.mean()),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "cohens_d": d,
        "wilcoxon_stat": float(stat_w),
        "p_twosided": float(p_wilcoxon),
        "p_onesided": float(p_onesided),
        "p_bonferroni_twosided": float(p_bonferroni),
        "p_bonferroni_onesided": float(p_onesided_bonf),
        "bootstrap_95ci": [ci_lo, ci_hi],
        "n_improved": n_improved,
        "n_same": n_same,
        "n_worse": n_worse,
    }
    results.append(result)

    # Significance markers
    def sig_marker(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "n.s."

    print(f"  {v1} → {v2}: {description}")
    print(f"    ΔF1 = {mean_diff*100:+.2f} pp ± {std_diff*100:.2f}")
    print(f"    Cohen's d = {d:.3f} ({['negligible','small','medium','large'][min(3, int(abs(d)/0.2+0.5) if abs(d) < 0.8 else 3)]})")
    print(f"    Wilcoxon p = {p_wilcoxon:.4f} (two-sided) {sig_marker(p_wilcoxon)}")
    print(f"    Wilcoxon p = {p_onesided:.4f} (one-sided) {sig_marker(p_onesided)}")
    print(f"    Bonferroni p = {p_bonferroni:.4f} (two-sided) {sig_marker(p_bonferroni)}")
    print(f"    Bonferroni p = {p_onesided_bonf:.4f} (one-sided) {sig_marker(p_onesided_bonf)}")
    print(f"    95% CI: [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp")
    print(f"    Subjects: {n_improved}↑ {n_same}= {n_worse}↓ (of {n})")
    print()


# ── Summary table for paper ──────────────────────────────────────────
print("\n" + "="*80)
print("LATEX-READY SUMMARY TABLE")
print("="*80)
print()
print("| Comparison | ΔF1 (pp) | Cohen's d | p (Wilcoxon) | p (Bonf.) | 95% CI | ↑/↓ |")
print("|------------|----------|-----------|-------------|-----------|--------|-----|")

for r in results:
    ci = r["bootstrap_95ci"]
    p_raw = r["p_twosided"]
    p_bonf = r["p_bonferroni_twosided"]

    def fmt_p(p):
        if p < 0.001: return "<0.001***"
        if p < 0.01: return f"{p:.3f}**"
        if p < 0.05: return f"{p:.3f}*"
        return f"{p:.3f}"

    print(
        f"| {r['comparison']:<10s} | "
        f"{r['mean_diff']*100:+5.2f}    | "
        f"{r['cohens_d']:+.3f}    | "
        f"{fmt_p(p_raw):<13s} | "
        f"{fmt_p(p_bonf):<9s} | "
        f"[{ci[0]*100:+.1f},{ci[1]*100:+.1f}] | "
        f"{r['n_improved']}↑{r['n_worse']}↓ |"
    )

# ── Save results ─────────────────────────────────────────────────────
out_file = OUT_DIR / "statistical_analysis_h6h7.json"
with open(out_file, "w") as f:
    json.dump({
        "description": "Wilcoxon signed-rank tests for H6+H7 ablation",
        "n_comparisons": n_tests,
        "bonferroni_k": bonferroni,
        "comparisons": results,
    }, f, indent=2)

print(f"\nResults saved to {out_file}")


# ── Additional: Effect size interpretation ───────────────────────────
print("\n" + "="*80)
print("EFFECT SIZE INTERPRETATION (Cohen's d)")
print("="*80)
print("  |d| < 0.2  : negligible")
print("  0.2 ≤ |d| < 0.5 : small")
print("  0.5 ≤ |d| < 0.8 : medium")
print("  |d| ≥ 0.8  : large")
print()
for r in results:
    d = abs(r["cohens_d"])
    if d < 0.2: cat = "negligible"
    elif d < 0.5: cat = "SMALL"
    elif d < 0.8: cat = "MEDIUM"
    else: cat = "LARGE"
    print(f"  {r['comparison']:<10s}: d={r['cohens_d']:+.3f} → {cat}")
