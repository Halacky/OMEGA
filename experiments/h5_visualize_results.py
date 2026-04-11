"""H5 Visualization: Integrated System Ablation."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
H5_DIRS = sorted(ROOT.glob("experiments_output/h5_integrated_system_*"))
RESULTS_DIR = None
for d in reversed(H5_DIRS):
    if (d / "comparison.json").exists():
        RESULTS_DIR = d
        break
if not RESULTS_DIR:
    print("No H5 results found"); sys.exit(1)
OUTPUT_DIR = RESULTS_DIR

VARIANTS = ["h4_best", "k6", "k8", "se_attn", "groupdro", "best_combo"]
VARIANT_LABELS = {
    "h4_best": "Baseline\n(K=4)",
    "k6": "K=6\nbands",
    "k8": "K=8\nbands",
    "se_attn": "K=4\n+ SE",
    "groupdro": "K=4\n+ GroupDRO",
    "best_combo": "K=6\n+ SE + DRO",
}
COLORS = {
    "h4_best": "#2ecc71",
    "k6": "#3498db",
    "k8": "#9b59b6",
    "se_attn": "#e67e22",
    "groupdro": "#e74c3c",
    "best_combo": "#1abc9c",
}

print(f"H5 Visualization")
print(f"  Results: {RESULTS_DIR}")

# Load data
data = {}
for v in VARIANTS:
    p = RESULTS_DIR / v / "loso_summary.json"
    if not p.exists(): continue
    d = json.load(open(p))
    subjects = [r["test_subject"] for r in d["per_subject"]]
    f1s = np.array([r["test_f1_macro"] for r in d["per_subject"]])
    accs = np.array([r["test_accuracy"] for r in d["per_subject"]])
    data[v] = {"subjects": subjects, "f1": f1s, "acc": accs}

print(f"  Loaded {len(data)} variants")

# ── Fig 1: Bar chart with error bars ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(VARIANTS))
means = [data[v]["f1"].mean() * 100 for v in VARIANTS]
stds = [data[v]["f1"].std() * 100 for v in VARIANTS]
colors = [COLORS[v] for v in VARIANTS]
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

best_idx = np.argmax(means)
bars[best_idx].set_edgecolor("gold")
bars[best_idx].set_linewidth(2.5)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.8, f"{m:.1f}%", ha="center", fontweight="bold", fontsize=11)

# Delta annotation from baseline
baseline_mean = means[0]
for i in range(1, len(means)):
    delta = means[i] - baseline_mean
    ax.annotate(f"{delta:+.1f} pp", xy=(i, means[i] - stds[i] - 1.5),
                ha="center", fontsize=9, color="red" if delta < 0 else "green")

ax.set_xticks(x)
ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=10)
ax.set_ylabel("F1 Macro (%)", fontsize=12)
ax.set_title("H5: Integrated System Ablation\n(Sinc FB + MixStyle, varying K / SE / GroupDRO, LOSO 20 subjects)", fontsize=13)
ax.set_ylim(0, max(means) + max(stds) + 5)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h5_f1_comparison.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h5_f1_comparison.png/pdf")

# ── Fig 2: Per-subject heatmap ──────────────────────────────────
subjects = data["h4_best"]["subjects"]
n_subj = len(subjects)
short_subj = [s.replace("DB2_", "") for s in subjects]

matrix = np.zeros((len(VARIANTS), n_subj))
for i, v in enumerate(VARIANTS):
    matrix[i] = data[v]["f1"] * 100

fig, ax = plt.subplots(figsize=(14, 4.5))
im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=20, vmax=55)
ax.set_xticks(range(n_subj))
ax.set_xticklabels(short_subj, fontsize=8, rotation=45)
ax.set_yticks(range(len(VARIANTS)))
ax.set_yticklabels([VARIANT_LABELS[v].replace("\n", " ") for v in VARIANTS], fontsize=9)
ax.set_title("H5: Per-Subject F1 Macro (%)", fontsize=12)
plt.colorbar(im, ax=ax, label="F1 (%)")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h5_per_subject_heatmap.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h5_per_subject_heatmap.png/pdf")

# ── Fig 3: Boxplot comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bp_data = [data[v]["f1"] * 100 for v in VARIANTS]
bp = ax.boxplot(bp_data, labels=[VARIANT_LABELS[v].replace("\n", " ") for v in VARIANTS],
                patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], [COLORS[v] for v in VARIANTS]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for i, v in enumerate(VARIANTS):
    y = data[v]["f1"] * 100
    jitter = np.random.normal(0, 0.05, len(y))
    ax.scatter(np.ones(len(y)) * (i + 1) + jitter, y, alpha=0.4, s=15, color="black", zorder=3)
ax.set_ylabel("F1 Macro (%)", fontsize=12)
ax.set_title("H5: Per-Subject F1 Distribution by Variant", fontsize=13)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(fontsize=9)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h5_boxplot.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h5_boxplot.png/pdf")

# ── Fig 4: Delta per subject vs baseline ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes_flat = axes.flatten()
compare_variants = ["k6", "k8", "se_attn", "groupdro", "best_combo"]
for idx, v in enumerate(compare_variants):
    ax = axes_flat[idx]
    delta = (data[v]["f1"] - data["h4_best"]["f1"]) * 100
    colors_bar = ["green" if d > 0 else "red" for d in delta]
    ax.bar(range(n_subj), delta, color=colors_bar, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(delta.mean(), color="blue", linewidth=1, linestyle="--",
               label=f"mean={delta.mean():+.1f} pp")
    ax.set_xticks(range(n_subj))
    ax.set_xticklabels(short_subj, fontsize=6, rotation=45)
    ax.set_ylabel("ΔF1 (pp)")
    ax.set_title(f"{VARIANT_LABELS[v].replace(chr(10), ' ')} vs Baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

# Hide last subplot
axes_flat[5].set_visible(False)
fig.suptitle("H5: Per-Subject F1 Change vs Baseline (K=4)", fontsize=13)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h5_delta_per_subject.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h5_delta_per_subject.png/pdf")

# ── Fig 5: Ablation factor analysis ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
factors = {
    "Baseline (K=4)": data["h4_best"]["f1"].mean() * 100,
    "+ More bands (K=6)": data["k6"]["f1"].mean() * 100,
    "+ Even more (K=8)": data["k8"]["f1"].mean() * 100,
    "+ SE attention": data["se_attn"]["f1"].mean() * 100,
    "+ GroupDRO loss": data["groupdro"]["f1"].mean() * 100,
    "+ All combined": data["best_combo"]["f1"].mean() * 100,
}
names = list(factors.keys())
vals = list(factors.values())
colors_f = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#1abc9c"]
bars = ax.barh(range(len(names)), vals, color=colors_f, edgecolor="black", linewidth=0.5)
for i, val in enumerate(vals):
    ax.text(val + 0.2, i, f"{val:.2f}%", va="center", fontweight="bold")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("F1 Macro (%)")
ax.set_title("H5: Impact of Each System Component")
ax.set_xlim(36, 42)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h5_component_impact.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h5_component_impact.png/pdf")

print(f"\nAll figures saved to {OUTPUT_DIR}")
