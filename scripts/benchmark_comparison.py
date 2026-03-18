#!/usr/bin/env python3
"""
Comparison of our H6/H7 results with published baselines from the benchmark paper.

Source: Golovan & Zvereva, "A Reproducible Benchmark for Zero-Shot Cross-Subject
Hand Gesture Recognition Using sEMG Signals" — Table I (LOSO, N=20 subjects).

Our experiments use the same dataset (NinaPro DB2), protocol (strict LOSO),
preprocessing pipeline, and 20 subjects.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  Published baselines (Table I, LOSO N=20)
# ═══════════════════════════════════════════════════════════════════
PUBLISHED = [
    # (name, category, features, acc_mean, acc_std, f1_mean, f1_std)
    ("CNN/LSTM BiGRU",    "Raw-Signal DL",   "Raw EMG", 35.7, 7.8, 33.5, 8.0),
    ("Multiscale CNN",    "Raw-Signal DL",   "Raw EMG", 35.5, 8.1, 33.4, 8.8),
    ("Simple CNN",        "Raw-Signal DL",   "Raw EMG", 35.9, 8.5, 33.3, 9.1),
    ("CNN-GRU-Attn",      "Raw-Signal DL",   "Raw EMG", 34.8, 8.8, 33.1, 9.1),
    ("Attention CNN",     "Raw-Signal DL",   "Raw EMG", 34.2, 8.0, 31.1, 8.7),
    ("BiLSTM-Attn",       "Raw-Signal DL",   "Raw EMG", 34.0, 7.8, 31.9, 8.1),
    ("TCN",               "Raw-Signal DL",   "Raw EMG", 26.7, 7.4, 23.1, 6.9),
    ("CNN/LSTM BiGRU",    "Feature-Based DL","TD Feat.", 32.7, 7.2, 30.7, 8.0),
    ("TCN-Attn",          "Feature-Based DL","TD Feat.", 31.9, 6.7, 30.2, 7.0),
    ("SVM-Linear",        "Classical ML",    "TD Feat.", 35.2, 8.9, 32.5, 9.4),
    ("SVM-RBF",           "Classical ML",    "TD Feat.", 34.5, 7.2, 32.6, 7.7),
    ("Random Forest",     "Classical ML",    "TD Feat.", 32.0, 6.5, 30.3, 7.0),
    ("Hybrid Powerful DL","Hybrid DL",       "TD Feat.", 33.9, 7.2, 31.7, 7.0),
    ("MLP Powerful",      "Hybrid DL",       "TD Feat.", 33.8, 7.4, 31.6, 7.3),
]

# ═══════════════════════════════════════════════════════════════════
#  Our results (H6 + H7, batch=512, 20 subjects)
# ═══════════════════════════════════════════════════════════════════
H6_DIR = PROJECT_ROOT / "experiments_output" / "h6_unified_ablation_20260312_130226"
H7_DIR = PROJECT_ROOT / "experiments_output" / "h7_uvmd_mixstyle"

OURS = []
for v_name, v_dir, label in [
    ("A", H6_DIR, "Raw CNN (ours, baseline)"),
    ("B", H6_DIR, "Sinc FB K=4 (ours)"),
    ("C", H6_DIR, "Sinc FB + MixStyle (ours)"),
    ("E", H7_DIR, "UVMD (ours)"),
    ("F", H7_DIR, "UVMD + MixStyle (ours)"),
]:
    f = v_dir / f"variant_{v_name}.json"
    if f.exists():
        d = json.loads(f.read_text())
        OURS.append((
            label, "Ours", "Raw EMG",
            d["mean_accuracy"] * 100, d["std_accuracy"] * 100,
            d["mean_f1_macro"] * 100, d["std_f1_macro"] * 100,
            v_name,
        ))

# ═══════════════════════════════════════════════════════════════════
#  Print comparison table
# ═══════════════════════════════════════════════════════════════════
print("=" * 95)
print("COMPARISON WITH PUBLISHED BASELINES (LOSO, N=20 subjects, NinaPro DB2 E1)")
print("=" * 95)
print(f"{'Method':<35s} {'Category':<18s} {'Acc (%)':<16s} {'F1 (%)':<16s}")
print("-" * 95)

# Sort all by F1 descending
all_methods = []
for name, cat, feat, acc_m, acc_s, f1_m, f1_s in PUBLISHED:
    all_methods.append((name, cat, feat, acc_m, acc_s, f1_m, f1_s, False, ""))
for name, cat, feat, acc_m, acc_s, f1_m, f1_s, vname in OURS:
    all_methods.append((name, cat, feat, acc_m, acc_s, f1_m, f1_s, True, vname))

all_methods.sort(key=lambda x: x[5], reverse=True)

for name, cat, feat, acc_m, acc_s, f1_m, f1_s, is_ours, vname in all_methods:
    marker = " ★" if is_ours else ""
    print(
        f"  {name:<33s} {cat:<18s} "
        f"{acc_m:5.1f} ± {acc_s:4.1f}   "
        f"{f1_m:5.1f} ± {f1_s:4.1f}{marker}"
    )

# ═══════════════════════════════════════════════════════════════════
#  Key comparisons
# ═══════════════════════════════════════════════════════════════════
best_published_f1 = max(r[5] for r in PUBLISHED)
best_published_name = [r[0] for r in PUBLISHED if r[5] == best_published_f1][0]
best_ours_f1 = max(r[5] for r in OURS)
best_ours_name = [r[0] for r in OURS if r[5] == best_ours_f1][0]

print(f"\n{'='*60}")
print(f"KEY COMPARISONS")
print(f"{'='*60}")
print(f"  Best published:  {best_published_name} — F1 = {best_published_f1:.1f}%")
print(f"  Best ours:       {best_ours_name} — F1 = {best_ours_f1:.1f}%")
print(f"  Improvement:     +{best_ours_f1 - best_published_f1:.1f} pp")
print()

# Our baseline vs their best (fair comparison)
our_baseline_f1 = [r[5] for r in OURS if r[7] == "A"][0]
print(f"  Our raw baseline (A): F1 = {our_baseline_f1:.1f}%")
print(f"  Their best raw CNN:   F1 = {best_published_f1:.1f}%")
print(f"  Delta:                {our_baseline_f1 - best_published_f1:+.1f} pp")
print()
print(f"  → Our baseline is {'comparable to' if abs(our_baseline_f1 - best_published_f1) < 2 else 'different from'} published best")
print(f"  → Our best method (UVMD+MixStyle) adds +{best_ours_f1 - our_baseline_f1:.1f} pp on top")

# ═══════════════════════════════════════════════════════════════════
#  Figure: Comparison bar chart
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(12, 7))

# Select representative baselines + all ours
selected_published = [
    ("CNN/LSTM BiGRU\n(Raw)", 33.5, 8.0),
    ("Simple CNN\n(Raw)", 33.3, 9.1),
    ("CNN-GRU-Attn\n(Raw)", 33.1, 9.1),
    ("SVM-Linear\n(TD Feat.)", 32.5, 9.4),
    ("SVM-RBF\n(TD Feat.)", 32.6, 7.7),
]

selected_ours = [
    ("Raw CNN\n(A, ours)", OURS[0][5], OURS[0][6]),
    ("Sinc FB\n(B, ours)", OURS[1][5], OURS[1][6]),
    ("Sinc+MS\n(C, ours)", OURS[2][5], OURS[2][6]),
    ("UVMD\n(E, ours)", OURS[3][5], OURS[3][6]),
    ("UVMD+MS\n(F, ours)", OURS[4][5], OURS[4][6]),
]

labels = [s[0] for s in selected_published] + [""] + [s[0] for s in selected_ours]
f1s = [s[1] for s in selected_published] + [0] + [s[1] for s in selected_ours]
stds = [s[2] for s in selected_published] + [0] + [s[2] for s in selected_ours]

n = len(labels)
x = np.arange(n)

colors = []
for i in range(len(selected_published)):
    colors.append("#78909c")  # gray-blue for published
colors.append("white")  # spacer
ours_colors = ["#9e9e9e", "#42a5f5", "#66bb6a", "#ab47bc", "#ff7043"]
colors.extend(ours_colors)

bars = ax.bar(x, f1s, yerr=stds, capsize=3,
              color=colors, edgecolor="black", linewidth=0.5, width=0.7)

# Make spacer invisible
bars[len(selected_published)].set_visible(False)

# Value labels
for i, (bar, val, std) in enumerate(zip(bars, f1s, stds)):
    if val == 0:
        continue
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
            fontweight="bold" if i > len(selected_published) else "normal")

# Divider
sep = len(selected_published) + 0.5
ax.axvline(sep, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.text(len(selected_published) / 2, max(f1s) + max(stds) + 3,
        "Published Baselines\n(Golovan & Zvereva, 2025)",
        ha="center", fontsize=9, style="italic", color="#546e7a")
ax.text(sep + len(selected_ours) / 2 + 0.5, max(f1s) + max(stds) + 3,
        "Our Methods\n(Frequency-Aware Processing)",
        ha="center", fontsize=9, style="italic", color="#1b5e20")

# Best published line
ax.axhline(best_published_f1, color="#b71c1c", linestyle=":", alpha=0.5, linewidth=1)
ax.text(n - 0.5, best_published_f1 + 0.3,
        f"Best published: {best_published_f1:.1f}%",
        ha="right", fontsize=8, color="#b71c1c")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("F1 Macro (%)")
ax.set_title("Cross-Subject Gesture Recognition: Published Baselines vs. Our Method\n"
             "(NinaPro DB2, LOSO, 20 subjects, 10 gestures E1)",
             fontsize=12)
ax.set_ylim(20, max(f1s) + max(stds) + 7)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"fig_benchmark_comparison.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {OUT_DIR / 'fig_benchmark_comparison.pdf'}")

# ═══════════════════════════════════════════════════════════════════
#  Save comparison data
# ═══════════════════════════════════════════════════════════════════
comparison_data = {
    "description": "Comparison with published baselines from benchmark paper",
    "source": "Golovan & Zvereva (2025), Table I, LOSO N=20",
    "dataset": "NinaPro DB2, Exercise E1, 10 gestures",
    "protocol": "Strict LOSO, 20 subjects",
    "published_baselines": [
        {"name": n, "category": c, "features": f, "f1_mean": f1, "f1_std": s}
        for n, c, f, _, _, f1, s in PUBLISHED
    ],
    "our_results": [
        {"variant": vn, "name": n, "f1_mean": round(f1, 2), "f1_std": round(s, 2)}
        for n, _, _, _, _, f1, s, vn in OURS
    ],
    "key_findings": {
        "best_published": {"name": best_published_name, "f1": best_published_f1},
        "best_ours": {"name": best_ours_name, "f1": round(best_ours_f1, 2)},
        "improvement_pp": round(best_ours_f1 - best_published_f1, 2),
        "our_baseline_vs_published": round(our_baseline_f1 - best_published_f1, 2),
    },
}

with open(OUT_DIR / "benchmark_comparison_data.json", "w") as f:
    json.dump(comparison_data, f, indent=2)
print(f"Data saved to {OUT_DIR / 'benchmark_comparison_data.json'}")
