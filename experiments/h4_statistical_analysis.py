"""H4 Statistical Analysis: Content-Style Disentanglement."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Find latest H4 results
H4_DIRS = sorted(ROOT.glob("experiments_output/h4_content_style_disentanglement_*"))
if not H4_DIRS:
    print("No H4 results found"); sys.exit(1)
RESULTS_DIR = H4_DIRS[-1]

VARIANTS = ["baseline", "adversarial", "contrastive", "mi_min", "full"]
VARIANT_LABELS = {
    "baseline": "Baseline (MixStyle only)",
    "adversarial": "Adversarial (GRL)",
    "contrastive": "Contrastive (InfoNCE)",
    "mi_min": "MI Minimization (DistCorr)",
    "full": "Full (Adv+CL+MI)",
}

print(f"H4 Statistical Analysis")
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

n_subj = len(data["baseline"]["f1"])
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

# Pairwise comparisons
print(f"\n{'='*70}")
print("PAIRWISE COMPARISONS (vs baseline)")
print(f"{'='*70}")

comparisons = [
    ("contrastive", "baseline"),
    ("adversarial", "baseline"),
    ("mi_min", "baseline"),
    ("full", "baseline"),
    ("full", "contrastive"),
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

    print(f"\n  {va} vs {vb}")
    print(f"    Mean diff:    {mean_diff*100:+.2f} pp  (median: {np.median(diff)*100:+.2f} pp)")
    print(f"    95% CI:       [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp")
    print(f"    Wilcoxon:     W={w_stat:.0f}, p={w_pval:.4f} {'*' if w_pval < 0.05 else ''}")
    print(f"    Cohen's d:    {cohens_d:.3f}")
    print(f"    Winners:      {n_better}/20 better, {n_worse}/20 worse")

    results_stat.append({
        "comparison": f"{va} vs {vb}",
        "mean_diff": float(mean_diff),
        "wilcoxon_p": float(w_pval),
        "cohens_d": float(cohens_d),
    })

# Bonferroni
print(f"\n{'='*70}")
print(f"BONFERRONI-CORRECTED p-values (m={len(results_stat)} comparisons)")
print(f"{'='*70}")
m = len(results_stat)
for r in results_stat:
    p_corr = min(1.0, r["wilcoxon_p"] * m)
    sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
    print(f"  {r['comparison']:35s}: p_corr={p_corr:.4f} {sig}, diff={r['mean_diff']*100:+.2f} pp")

# Save
out = {
    "experiment": "h4_content_style_disentanglement",
    "n_subjects": n_subj,
    "variants": {v: {"mean_f1": float(data[v]["f1"].mean()),
                      "std_f1": float(data[v]["f1"].std())}
                 for v in data},
    "comparisons": results_stat,
}
out_path = RESULTS_DIR / "statistical_analysis.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to {out_path}")
