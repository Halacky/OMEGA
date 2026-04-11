"""H4 Visualization: Content-Style Disentanglement."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
H4_DIRS = sorted(ROOT.glob("experiments_output/h4_content_style_disentanglement_*"))
if not H4_DIRS:
    print("No H4 results found"); sys.exit(1)
RESULTS_DIR = H4_DIRS[-1]
OUTPUT_DIR = RESULTS_DIR

VARIANTS = ["baseline", "adversarial", "contrastive", "mi_min", "full"]
VARIANT_LABELS = {
    "baseline": "Baseline\n(MixStyle only)",
    "adversarial": "Adversarial\n(GRL)",
    "contrastive": "Contrastive\n(InfoNCE)",
    "mi_min": "MI Min\n(DistCorr)",
    "full": "Full\n(Adv+CL+MI)",
}
COLORS = {
    "baseline": "#888888",
    "adversarial": "#e74c3c",
    "contrastive": "#2ecc71",
    "mi_min": "#3498db",
    "full": "#9b59b6",
}

print(f"H4 Visualization")
print(f"  Results: {RESULTS_DIR}")

# Load data
data = {}
for v in VARIANTS:
    p = RESULTS_DIR / v / "loso_summary.json"
    if not p.exists(): continue
    d = json.load(open(p))
    subjects = [r["test_subject"] for r in d["per_subject"]]
    f1s = np.array([r["test_f1_macro"] for r in d["per_subject"]])
    data[v] = {"subjects": subjects, "f1": f1s, "results": d["results"]}

print(f"  Loaded {len(data)} variants")

# ── Fig 1: Bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(VARIANTS))
means = [data[v]["f1"].mean() * 100 for v in VARIANTS]
stds = [data[v]["f1"].std() * 100 for v in VARIANTS]
colors = [COLORS[v] for v in VARIANTS]
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

best_idx = np.argmax(means)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 1, f"{m:.1f}%", ha="center", fontweight="bold", fontsize=11)

# Delta annotation from baseline
baseline_mean = means[0]
for i in range(1, len(means)):
    delta = means[i] - baseline_mean
    ax.annotate(f"{delta:+.1f} pp", xy=(i, means[i] - stds[i] - 1),
                ha="center", fontsize=9, color="red" if delta < 0 else "green")

ax.set_xticks(x)
ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=10)
ax.set_ylabel("F1 Macro (%)", fontsize=12)
ax.set_title("H4: Content-Style Disentanglement Ablation\n(Sinc FB + MixStyle + varying disentanglement, LOSO)", fontsize=13)
ax.set_ylim(0, max(means) + max(stds) + 5)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h4_f1_comparison.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h4_f1_comparison.png/pdf")

# ── Fig 2: Per-subject heatmap ──────────────────────────────────
subjects = data["baseline"]["subjects"]
n_subj = len(subjects)
short_subj = [s.replace("DB2_", "") for s in subjects]

matrix = np.zeros((len(VARIANTS), n_subj))
for i, v in enumerate(VARIANTS):
    matrix[i] = data[v]["f1"] * 100

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=15, vmax=55)
ax.set_xticks(range(n_subj))
ax.set_xticklabels(short_subj, fontsize=8, rotation=45)
ax.set_yticks(range(len(VARIANTS)))
ax.set_yticklabels([VARIANT_LABELS[v].replace("\n", " ") for v in VARIANTS], fontsize=9)
ax.set_title("H4: Per-Subject F1 Macro (%)", fontsize=12)
plt.colorbar(im, ax=ax, label="F1 (%)")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h4_per_subject_heatmap.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h4_per_subject_heatmap.png/pdf")

# ── Fig 3: Delta per subject vs baseline ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for idx, v in enumerate(["adversarial", "contrastive", "mi_min", "full"]):
    ax = axes[idx // 2][idx % 2]
    delta = (data[v]["f1"] - data["baseline"]["f1"]) * 100
    colors_bar = ["green" if d > 0 else "red" for d in delta]
    ax.bar(range(n_subj), delta, color=colors_bar, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(delta.mean(), color="blue", linewidth=1, linestyle="--",
               label=f"mean={delta.mean():+.1f} pp")
    ax.set_xticks(range(n_subj))
    ax.set_xticklabels(short_subj, fontsize=7, rotation=45)
    ax.set_ylabel("ΔF1 (pp)")
    ax.set_title(f"{VARIANT_LABELS[v].replace(chr(10), ' ')} vs Baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.suptitle("H4: Per-Subject F1 Change vs Baseline", fontsize=13)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h4_delta_per_subject.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h4_delta_per_subject.png/pdf")

# ── Fig 4: Loss component impact ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
components = {
    "No disentanglement": data["baseline"]["f1"].mean() * 100,
    "+ Adversarial (GRL)": data["adversarial"]["f1"].mean() * 100,
    "+ Contrastive (InfoNCE)": data["contrastive"]["f1"].mean() * 100,
    "+ MI Min (DistCorr)": data["mi_min"]["f1"].mean() * 100,
    "+ All three": data["full"]["f1"].mean() * 100,
}
names = list(components.keys())
vals = list(components.values())
colors_c = ["#888888", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
bars = ax.barh(range(len(names)), vals, color=colors_c, edgecolor="black", linewidth=0.5)
for i, val in enumerate(vals):
    ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontweight="bold")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("F1 Macro (%)")
ax.set_title("H4: Impact of Each Disentanglement Component")
ax.set_xlim(30, 42)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUTPUT_DIR / f"h4_component_impact.{ext}", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved h4_component_impact.png/pdf")

print(f"\nAll figures saved to {OUTPUT_DIR}")
