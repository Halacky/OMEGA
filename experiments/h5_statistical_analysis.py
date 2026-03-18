"""H5 Statistical Analysis: Integrated System Ablation."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Find latest H5 results
H5_DIRS = sorted(ROOT.glob("experiments_output/h5_integrated_system_*"))
# Pick the one with comparison.json
RESULTS_DIR = None
for d in reversed(H5_DIRS):
    if (d / "comparison.json").exists():
        RESULTS_DIR = d
        break
if not RESULTS_DIR:
    print("No H5 results found"); sys.exit(1)

VARIANTS = ["h4_best", "k6", "k8", "se_attn", "groupdro", "best_combo"]
VARIANT_LABELS = {
    "h4_best": "Baseline (K=4)",
    "k6": "K=6 bands",
    "k8": "K=8 bands",
    "se_attn": "K=4 + SE attention",
    "groupdro": "K=4 + GroupDRO",
    "best_combo": "K=6 + SE + GroupDRO",
}

print(f"H5 Statistical Analysis")
print(f"  Results dir: {RESULTS_DIR}")

# Load per-subject F1
data = {}
for v in VARIANTS:
    p = RESULTS_DIR / v / "loso_summary.json"
    if not p.exists():
        print(f"  SKIP {v}: no results"); continue
    d = json.load(open(p))
    subjects = [r["test_subject"] for r in d["per_subject"]]
    f1s = [r["test_f1_macro"] for r in d["per_subject"]]
    data[v] = {"subjects": subjects, "f1": np.array(f1s)}

baseline = "h4_best"
n_subj = len(data[baseline]["f1"])
print(f"  N subjects: {n_subj}")
print(f"  Variants: {list(data.keys())}")

# Summary stats
print(f"\n{'='*70}")
print("SUMMARY STATISTICS")
print(f"{'='*70}")
for v in VARIANTS:
    if v not in data: continue
    f = data[v]["f1"]
    print(f"  {v:15s}: F1 = {f.mean()*100:.2f} +/- {f.std()*100:.2f}% "
          f"(median={np.median(f)*100:.2f}, min={f.min()*100:.2f}, max={f.max()*100:.2f})")

# Friedman test (all variants)
print(f"\n{'='*70}")
print("FRIEDMAN TEST (all 6 variants)")
print(f"{'='*70}")
arrays = [data[v]["f1"] for v in VARIANTS if v in data]
chi2, p_friedman = stats.friedmanchisquare(*arrays)
print(f"  Friedman chi2 = {chi2:.4f}, p = {p_friedman:.6f}")
if p_friedman < 0.05:
    print(f"  -> Significant difference between variants (p < 0.05)")
else:
    print(f"  -> No significant difference between variants (p >= 0.05)")

# Pairwise comparisons vs baseline
print(f"\n{'='*70}")
print(f"PAIRWISE COMPARISONS (vs {baseline})")
print(f"{'='*70}")

comparisons = [
    ("k6", baseline),
    ("k8", baseline),
    ("se_attn", baseline),
    ("groupdro", baseline),
    ("best_combo", baseline),
    ("se_attn", "k6"),  # cross-comparison
]

results_stat = []
for va, vb in comparisons:
    if va not in data or vb not in data: continue
    a, b = data[va]["f1"], data[vb]["f1"]
    diff = a - b
    mean_diff = diff.mean()
    ci_lo = mean_diff - 1.96 * diff.std() / np.sqrt(len(diff))
    ci_hi = mean_diff + 1.96 * diff.std() / np.sqrt(len(diff))

    t_stat, t_pval = stats.ttest_rel(a, b)
    try:
        w_stat, w_pval = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_pval = 0, 1.0

    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    n_better = (diff > 0).sum()
    n_worse = (diff < 0).sum()

    print(f"\n  {VARIANT_LABELS.get(va, va)} vs {VARIANT_LABELS.get(vb, vb)}")
    print(f"    Mean diff:    {mean_diff*100:+.2f} pp  (median: {np.median(diff)*100:+.2f} pp)")
    print(f"    95% CI:       [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp")
    print(f"    Wilcoxon:     W={w_stat:.0f}, p={w_pval:.4f} {'*' if w_pval < 0.05 else ''}")
    print(f"    t-test:       t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"    Cohen's d:    {cohens_d:.3f}")
    print(f"    Winners:      {n_better}/{n_subj} better, {n_worse}/{n_subj} worse")

    results_stat.append({
        "comparison": f"{va} vs {vb}",
        "mean_diff": float(mean_diff),
        "wilcoxon_p": float(w_pval),
        "ttest_p": float(t_pval),
        "cohens_d": float(cohens_d),
        "n_better": int(n_better),
        "n_worse": int(n_worse),
    })

# Bonferroni
print(f"\n{'='*70}")
print(f"BONFERRONI-CORRECTED p-values (m={len(results_stat)} comparisons)")
print(f"{'='*70}")
m = len(results_stat)
for r in results_stat:
    p_corr = min(1.0, r["wilcoxon_p"] * m)
    sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
    print(f"  {r['comparison']:35s}: p_corr={p_corr:.4f} {sig}, diff={r['mean_diff']*100:+.2f} pp, d={r['cohens_d']:.3f}")

# Variance analysis — does any variant reduce inter-subject variance?
print(f"\n{'='*70}")
print("INTER-SUBJECT VARIANCE ANALYSIS")
print(f"{'='*70}")
for v in VARIANTS:
    if v not in data: continue
    f = data[v]["f1"]
    iqr = np.percentile(f, 75) - np.percentile(f, 25)
    print(f"  {v:15s}: std={f.std()*100:.2f}%, IQR={iqr*100:.2f}%, range=[{f.min()*100:.1f}, {f.max()*100:.1f}]")

baseline_std = data[baseline]["f1"].std()
print(f"\n  GroupDRO variance reduction: {(data['groupdro']['f1'].std() - baseline_std)*100:+.2f} pp std")
print(f"  (GroupDRO designed to reduce worst-case, std: {data['groupdro']['f1'].std()*100:.2f} vs baseline {baseline_std*100:.2f})")

# Save
out = {
    "experiment": "h5_integrated_system_ablation",
    "n_subjects": n_subj,
    "friedman": {"chi2": float(chi2), "p": float(p_friedman)},
    "variants": {v: {"mean_f1": float(data[v]["f1"].mean()),
                      "std_f1": float(data[v]["f1"].std())}
                 for v in data},
    "comparisons": results_stat,
}
out_path = RESULTS_DIR / "statistical_analysis.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to {out_path}")
