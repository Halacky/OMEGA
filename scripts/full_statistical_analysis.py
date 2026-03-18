#!/usr/bin/env python3
"""
Comprehensive statistical analysis for ALL hypotheses H2-H7.

Wilcoxon signed-rank tests with Bonferroni correction,
Cohen's d effect sizes, bootstrap 95% CIs.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_per_subject(json_path: Path) -> dict:
    """Load per-subject F1 from loso_summary.json or variant_X.json."""
    d = json.loads(json_path.read_text())
    result = {}
    for entry in d["per_subject"]:
        sid = entry.get("test_subject", entry.get("subject", ""))
        # Try multiple key names
        f1 = entry.get("f1_macro",
             entry.get("test_f1_macro",
             entry.get("f1", 0)))
        result[sid] = f1
    return result


def cohens_d(x, y):
    diff = x - y
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def bootstrap_ci(x, y, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    diff = x - y
    n = len(diff)
    boot = np.array([rng.choice(diff, n, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100))


def sig_marker(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


# ═══════════════════════════════════════════════════════════════
# Load all per-subject data
# ═══════════════════════════════════════════════════════════════
EXP = PROJECT_ROOT / "experiments_output"

data = {}

# H2
for v in ["none", "fixed_fb", "uvmd"]:
    import glob
    files = sorted(glob.glob(str(EXP / "h2_ablation_decomposition_20260307_095328" / v / "*.json")))
    if files:
        data[f"H2_{v}"] = load_per_subject(Path(files[0]))

# H3
for v in ["baseline", "per_band_mix", "global_in", "adaptive_in"]:
    files = sorted(glob.glob(str(EXP / "h3_style_normalization_20260308_014819" / v / "*.json")))
    if files:
        data[f"H3_{v}"] = load_per_subject(Path(files[0]))

# H4
for v in ["baseline", "adversarial", "contrastive", "mi_min", "full"]:
    files = sorted(glob.glob(str(EXP / "h4_content_style_disentanglement_20260309_020154" / v / "*.json")))
    if files:
        data[f"H4_{v}"] = load_per_subject(Path(files[0]))

# H5
for v in ["h4_best", "k6", "k8", "se_attn", "groupdro", "best_combo"]:
    files = sorted(glob.glob(str(EXP / "h5_integrated_system_20260310_184555" / v / "*.json")))
    if files:
        data[f"H5_{v}"] = load_per_subject(Path(files[0]))

# H6
for v in ["A", "B", "C", "D"]:
    f = EXP / "h6_unified_ablation_20260312_130226" / f"variant_{v}.json"
    if f.exists():
        d = json.loads(f.read_text())
        data[f"H6_{v}"] = {r["test_subject"]: r["f1_macro"] for r in d["per_subject"]}

# H7
for v in ["E", "F"]:
    f = EXP / "h7_uvmd_mixstyle" / f"variant_{v}.json"
    if f.exists():
        d = json.loads(f.read_text())
        data[f"H7_{v}"] = {r["test_subject"]: r["f1_macro"] for r in d["per_subject"]}

print(f"Loaded {len(data)} variants")
for k, v in sorted(data.items()):
    vals = list(v.values())
    print(f"  {k}: n={len(v)}, mean_F1={np.mean(vals):.4f} ± {np.std(vals):.4f}")

# ═══════════════════════════════════════════════════════════════
# Define comparisons
# ═══════════════════════════════════════════════════════════════
comparisons = [
    # H2: Decomposition
    ("H2_none", "H2_fixed_fb", "H2", "Raw → Sinc FB"),
    ("H2_none", "H2_uvmd", "H2", "Raw → UVMD"),
    ("H2_fixed_fb", "H2_uvmd", "H2", "Sinc FB vs UVMD"),

    # H3: Normalization
    ("H3_baseline", "H3_per_band_mix", "H3", "Baseline → Per-band MixStyle"),
    ("H3_baseline", "H3_global_in", "H3", "Baseline → Global IN"),
    ("H3_baseline", "H3_adaptive_in", "H3", "Baseline → Adaptive IN"),

    # H4: Disentanglement
    ("H4_baseline", "H4_adversarial", "H4", "MixStyle → +Adversarial"),
    ("H4_baseline", "H4_contrastive", "H4", "MixStyle → +Contrastive"),
    ("H4_baseline", "H4_mi_min", "H4", "MixStyle → +MI-min"),
    ("H4_baseline", "H4_full", "H4", "MixStyle → +All losses"),

    # H5: Architecture (Friedman for multiple)
    # Pairwise against h4_best
    ("H5_h4_best", "H5_k6", "H5", "K=4 vs K=6"),
    ("H5_h4_best", "H5_k8", "H5", "K=4 vs K=8"),
    ("H5_h4_best", "H5_se_attn", "H5", "K=4 vs K=4+SE"),
    ("H5_h4_best", "H5_groupdro", "H5", "K=4 vs K=4+GroupDRO"),

    # H6: Ablation
    ("H6_A", "H6_B", "H6", "Raw → Sinc FB (clean)"),
    ("H6_B", "H6_C", "H6", "Sinc → +MixStyle"),
    ("H6_C", "H6_D", "H6", "+MixStyle → +CS heads"),
    ("H6_A", "H6_C", "H6", "Raw → Sinc+MS (total)"),

    # H7: UVMD+MixStyle
    ("H7_E", "H7_F", "H7", "UVMD → +MixStyle"),
    ("H6_A", "H7_F", "H7", "Raw → UVMD+MS (total)"),
    ("H6_B", "H7_E", "cross", "Sinc vs UVMD (no MS)"),
    ("H6_C", "H7_F", "cross", "Sinc+MS vs UVMD+MS"),
]

n_tests = len(comparisons)
print(f"\n{'='*90}")
print(f"STATISTICAL ANALYSIS: {n_tests} comparisons, Bonferroni k={n_tests}")
print(f"{'='*90}\n")

results = []

for v1_key, v2_key, hyp, description in comparisons:
    if v1_key not in data or v2_key not in data:
        print(f"  SKIP: {v1_key} vs {v2_key} — missing")
        continue

    common = sorted(set(data[v1_key].keys()) & set(data[v2_key].keys()))
    n = len(common)
    x = np.array([data[v1_key][s] for s in common])
    y = np.array([data[v2_key][s] for s in common])

    stat_w, p_two = stats.wilcoxon(x, y, alternative="two-sided")
    _, p_one = stats.wilcoxon(x, y, alternative="less")

    p_bonf_two = min(p_two * n_tests, 1.0)
    p_bonf_one = min(p_one * n_tests, 1.0)

    d = cohens_d(y, x)
    ci_lo, ci_hi = bootstrap_ci(y, x)
    mean_diff = (y - x).mean()
    n_up = int(np.sum(y > x))
    n_down = int(np.sum(y < x))

    result = {
        "hypothesis": hyp,
        "comparison": f"{v1_key} → {v2_key}",
        "description": description,
        "n_subjects": n,
        "mean_f1_v1": float(x.mean()),
        "mean_f1_v2": float(y.mean()),
        "mean_diff": float(mean_diff),
        "cohens_d": d,
        "p_twosided": float(p_two),
        "p_bonferroni": float(p_bonf_two),
        "bootstrap_95ci": [ci_lo, ci_hi],
        "n_improved": n_up,
        "n_worse": n_down,
    }
    results.append(result)

    print(f"  [{hyp}] {description}")
    print(f"    {v1_key} ({x.mean()*100:.1f}%) → {v2_key} ({y.mean()*100:.1f}%)")
    print(f"    ΔF1 = {mean_diff*100:+.2f} pp, d = {d:.3f}, p = {p_two:.4f} {sig_marker(p_two)}, "
          f"p_bonf = {p_bonf_two:.4f} {sig_marker(p_bonf_two)}, "
          f"CI [{ci_lo*100:+.1f}, {ci_hi*100:+.1f}], ↑{n_up}/↓{n_down}")
    print()

# ═══════════════════════════════════════════════════════════════
# Friedman test for H5
# ═══════════════════════════════════════════════════════════════
print("=" * 90)
print("FRIEDMAN TEST — H5 Architecture Optimization")
print("=" * 90)

h5_variants = ["H5_h4_best", "H5_k6", "H5_k8", "H5_se_attn", "H5_groupdro", "H5_best_combo"]
h5_data_present = [v for v in h5_variants if v in data]
common_h5 = sorted(set.intersection(*[set(data[v].keys()) for v in h5_data_present]))

h5_matrix = np.array([[data[v][s] for s in common_h5] for v in h5_data_present])
chi2, p_friedman = stats.friedmanchisquare(*h5_matrix)
print(f"  Variants: {h5_data_present}")
print(f"  n={len(common_h5)} subjects, chi2={chi2:.4f}, p={p_friedman:.4f} {sig_marker(p_friedman)}")
print()

# ═══════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════
print("=" * 120)
print("PAPER-READY SUMMARY TABLE")
print("=" * 120)
print(f"| {'Hyp':<5s} | {'Comparison':<35s} | {'ΔF1 (pp)':>9s} | {'d':>7s} | {'p':>8s} | {'p_bonf':>8s} | {'95% CI':>14s} | {'↑/↓':>5s} |")
print("|" + "-"*7 + "|" + "-"*37 + "|" + "-"*11 + "|" + "-"*9 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*16 + "|" + "-"*7 + "|")

for r in results:
    ci = r["bootstrap_95ci"]
    print(f"| {r['hypothesis']:<5s} | {r['description']:<35s} | "
          f"{r['mean_diff']*100:+6.2f}    | {r['cohens_d']:+6.3f} | "
          f"{r['p_twosided']:.4f}  | {r['p_bonferroni']:.4f}  | "
          f"[{ci[0]*100:+5.1f},{ci[1]*100:+5.1f}] | "
          f"{r['n_improved']}↑{r['n_worse']}↓ |")

# ═══════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════
out_data = {
    "description": "Full statistical analysis for H2-H7",
    "n_comparisons": n_tests,
    "bonferroni_k": n_tests,
    "comparisons": results,
    "friedman_h5": {
        "chi2": float(chi2),
        "p_value": float(p_friedman),
        "n_subjects": len(common_h5),
        "variants": h5_data_present,
    },
}

out_path = OUT_DIR / "full_statistical_analysis.json"
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2)
print(f"\nSaved to {out_path}")
