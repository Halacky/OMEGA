#!/usr/bin/env python3
"""
Analysis: Why does MixStyle help some subjects but hurt others?

Correlates MixStyle F1 delta with:
1. Baseline F1 (does MixStyle help weak subjects more?)
2. Per-subject spectral profile deviation from population mean
3. UVMD omega consistency (do atypical subjects benefit more/less?)

Output: correlation figure + statistical summary.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "paper_figures"
FS = 2000  # sampling rate
OUT_DIR.mkdir(parents=True, exist_ok=True)

H6_DIR = PROJECT_ROOT / "experiments_output" / "h6_unified_ablation_20260312_130226"
H7_DIR = PROJECT_ROOT / "experiments_output" / "h7_uvmd_mixstyle"

# ── Load per-subject F1 data ──────────────────────────────────────
def load_variant(path):
    d = json.loads(path.read_text())
    return {r["test_subject"]: r for r in d["per_subject"]}

variants = {}
for vname, vdir in [("A", H6_DIR), ("B", H6_DIR), ("C", H6_DIR), ("E", H7_DIR), ("F", H7_DIR)]:
    f = vdir / f"variant_{vname}.json"
    if f.exists():
        variants[vname] = load_variant(f)

# ── Analysis 1: MixStyle delta vs baseline F1 ─────────────────────
# For both Sinc FB (B→C) and UVMD (E→F)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})

for idx, (v_base, v_mix, title) in enumerate([
    ("B", "C", "Sinc Filterbank: B → C (+MixStyle)"),
    ("E", "F", "UVMD: E → F (+MixStyle)"),
]):
    if v_base not in variants or v_mix not in variants:
        continue

    ax = axes[idx]
    common = sorted(set(variants[v_base].keys()) & set(variants[v_mix].keys()))

    base_f1 = np.array([variants[v_base][s]["f1_macro"] for s in common])
    mix_f1 = np.array([variants[v_mix][s]["f1_macro"] for s in common])
    delta = (mix_f1 - base_f1) * 100  # in pp

    # Color by sign
    colors = ["#43a047" if d > 0 else "#e53935" for d in delta]

    ax.scatter(base_f1 * 100, delta, c=colors, s=60, edgecolors="black", linewidth=0.5, zorder=5)

    # Regression line
    slope, intercept, r_val, p_val, se = stats.linregress(base_f1 * 100, delta)
    x_line = np.linspace(base_f1.min() * 100, base_f1.max() * 100, 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.5, linewidth=1)

    # Subject labels for outliers
    for i, s in enumerate(common):
        if abs(delta[i]) > np.percentile(np.abs(delta), 75):
            ax.annotate(s.replace("DB2_", ""), (base_f1[i] * 100, delta[i]),
                       fontsize=7, alpha=0.7, ha="center", va="bottom" if delta[i] > 0 else "top")

    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Baseline F1 (%)")
    ax.set_ylabel("ΔF1 with MixStyle (pp)")
    ax.set_title(title)

    n_up = np.sum(delta > 0)
    n_down = np.sum(delta < 0)
    ax.text(0.05, 0.95,
            f"r = {r_val:.3f}, p = {p_val:.3f}\n"
            f"↑{n_up} / ↓{n_down} subjects\n"
            f"Mean Δ = {delta.mean():+.2f} pp",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.grid(alpha=0.2)

fig.suptitle("MixStyle Effect vs. Baseline Performance", fontsize=13, y=1.02)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"fig_mixstyle_delta_correlation.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"Figure 1 saved: {OUT_DIR / 'fig_mixstyle_delta_correlation.pdf'}")

# ── Analysis 2: Per-subject profile — who benefits from UVMD vs Sinc? ──
if "B" in variants and "E" in variants:
    common = sorted(set(variants["B"].keys()) & set(variants["E"].keys()))
    sinc_f1 = np.array([variants["B"][s]["f1_macro"] for s in common])
    uvmd_f1 = np.array([variants["E"][s]["f1_macro"] for s in common])

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.scatter(sinc_f1 * 100, uvmd_f1 * 100, c="#42a5f5", s=60,
                edgecolors="black", linewidth=0.5, zorder=5)

    # Diagonal (equal performance)
    lims = [min(sinc_f1.min(), uvmd_f1.min()) * 100 - 2,
            max(sinc_f1.max(), uvmd_f1.max()) * 100 + 2]
    ax2.plot(lims, lims, "k--", alpha=0.3, label="Equal line")

    for i, s in enumerate(common):
        ax2.annotate(s.replace("DB2_", ""), (sinc_f1[i] * 100, uvmd_f1[i] * 100),
                    fontsize=7, alpha=0.6, ha="left")

    n_uvmd_better = np.sum(uvmd_f1 > sinc_f1)
    stat, p = stats.wilcoxon(uvmd_f1, sinc_f1)

    ax2.set_xlabel("Sinc Filterbank F1 (%)")
    ax2.set_ylabel("UVMD F1 (%)")
    ax2.set_title("Sinc Filterbank (B) vs. UVMD (E): Per-Subject Comparison")
    ax2.text(0.05, 0.95,
             f"UVMD better: {n_uvmd_better}/{len(common)} subjects\n"
             f"Wilcoxon p = {p:.4f}",
             transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="lower right")

    fig2.tight_layout()
    for ext in ["pdf", "png"]:
        fig2.savefig(OUT_DIR / f"fig_sinc_vs_uvmd_scatter.{ext}", bbox_inches="tight")
    plt.close(fig2)
    print(f"Figure 2 saved: {OUT_DIR / 'fig_sinc_vs_uvmd_scatter.pdf'}")

# ── Analysis 3: UVMD omega variability vs F1 ──────────────────────
if "E" in variants:
    subjects = sorted(variants["E"].keys())
    f1_vals = []
    omega_devs = []

    # Population mean omega
    all_omegas = np.array([variants["E"][s]["final_omega_k"] for s in subjects])
    pop_mean = all_omegas.mean(axis=0)

    for s in subjects:
        r = variants["E"][s]
        f1_vals.append(r["f1_macro"])
        omega = np.array(r["final_omega_k"])
        dev = np.sqrt(((omega - pop_mean) ** 2).sum())
        omega_devs.append(dev)

    f1_vals = np.array(f1_vals)
    omega_devs = np.array(omega_devs)

    r_val, p_val = stats.pearsonr(omega_devs, f1_vals)
    print(f"\nOmega deviation vs F1: r={r_val:.3f}, p={p_val:.3f}")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.scatter(omega_devs * FS, f1_vals * 100, c="#ab47bc", s=60,
                edgecolors="black", linewidth=0.5)
    for i, s in enumerate(subjects):
        ax3.annotate(s.replace("DB2_", ""), (omega_devs[i] * FS, f1_vals[i] * 100),
                    fontsize=7, alpha=0.6, ha="left")

    ax3.set_xlabel("UVMD Omega Deviation from Population Mean (Hz)")
    ax3.set_ylabel("F1 Macro (%)")
    ax3.set_title("UVMD Frequency Adaptation vs. Classification Performance")
    ax3.text(0.05, 0.95, f"Pearson r = {r_val:.3f}, p = {p_val:.3f}",
             transform=ax3.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax3.grid(alpha=0.2)

    fig3.tight_layout()
    for ext in ["pdf", "png"]:
        fig3.savefig(OUT_DIR / f"fig_omega_deviation_vs_f1.{ext}", bbox_inches="tight")
    plt.close(fig3)
    print(f"Figure 3 saved: {OUT_DIR / 'fig_omega_deviation_vs_f1.pdf'}")

# ── Summary statistics ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("MIXSTYLE DELTA ANALYSIS SUMMARY")
print("=" * 70)

for v_base, v_mix, label in [("B", "C", "Sinc+MixStyle"), ("E", "F", "UVMD+MixStyle")]:
    if v_base not in variants or v_mix not in variants:
        continue
    common = sorted(set(variants[v_base].keys()) & set(variants[v_mix].keys()))
    base_f1 = np.array([variants[v_base][s]["f1_macro"] for s in common])
    mix_f1 = np.array([variants[v_mix][s]["f1_macro"] for s in common])
    delta = (mix_f1 - base_f1) * 100

    r_val, p_val = stats.pearsonr(base_f1 * 100, delta)
    print(f"\n{label}:")
    print(f"  Mean delta: {delta.mean():+.2f} pp ± {delta.std():.2f}")
    print(f"  Subjects improved: {np.sum(delta > 0)}/{len(delta)}")
    print(f"  Correlation with baseline: r={r_val:.3f}, p={p_val:.3f}")
    print(f"  Largest gain: {common[np.argmax(delta)]} ({delta.max():+.2f} pp)")
    print(f"  Largest loss: {common[np.argmin(delta)]} ({delta.min():+.2f} pp)")
