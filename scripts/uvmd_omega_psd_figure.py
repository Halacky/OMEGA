#!/usr/bin/env python3
"""
Figure: UVMD learned center frequencies overlaid on EMG power spectral density.

Shows how UVMD discovers frequency bands that align with EMG spectral structure:
- Mode 1 (~87 Hz): Motor unit firing / low-frequency power
- Mode 2 (~385 Hz): Mid-frequency EMG activity
- Mode 3 (~671 Hz): High-frequency EMG components
- Mode 4 (~889 Hz): Near-Nyquist residual

Uses raw EMG from one representative subject to compute PSD.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 2000  # Sampling rate

# ── Load learned omega values ──────────────────────────────────────
variant_e = json.loads(
    (PROJECT_ROOT / "experiments_output" / "h7_uvmd_mixstyle" / "variant_E.json").read_text()
)

# Collect all per-subject final omega values
all_omegas = []
for r in variant_e["per_subject"]:
    all_omegas.append(r["final_omega_k"])

all_omegas = np.array(all_omegas)  # (N_subjects, K)
mean_omega = all_omegas.mean(axis=0)
std_omega = all_omegas.std(axis=0)

# Convert to Hz: omega is normalized [0, 0.5] -> [0, Nyquist]
mean_freq_hz = mean_omega * FS
std_freq_hz = std_omega * FS

# Init frequencies for comparison
init_omega = np.array(variant_e["per_subject"][0]["init_omega_k"])
init_freq_hz = init_omega * FS

print("UVMD Center Frequencies (Hz):")
print(f"  Init:    {init_freq_hz}")
print(f"  Learned: {mean_freq_hz} ± {std_freq_hz}")

# ── Compute PSD from raw EMG data ─────────────────────────────────
from data.multi_subject_loader import MultiSubjectLoader
from config.base import ProcessingConfig
import logging

logger = logging.getLogger("PSD")
proc_cfg = ProcessingConfig()
loader = MultiSubjectLoader(processing_config=proc_cfg, logger=logger)
data_dir = PROJECT_ROOT / "data"

# Use subject 1 as representative
subj_id = "DB2_s1"
emg, segments, grouped_windows = loader.load_subject(
    base_dir=data_dir,
    subject_id=subj_id,
    exercise="E1",
    include_rest=False,
)

print(f"\nLoaded {subj_id}: EMG shape = {emg.shape}")

# Compute PSD using Welch's method on raw EMG (all channels averaged)
freqs, psd_all = signal.welch(emg, fs=FS, nperseg=1024, noverlap=512, axis=0)
psd_mean = psd_all.mean(axis=1)  # Average across channels
psd_db = 10 * np.log10(psd_mean + 1e-12)

# Also compute per-channel PSD for envelope
psd_channels = 10 * np.log10(psd_all + 1e-12)
psd_lo = np.percentile(psd_channels, 10, axis=1)
psd_hi = np.percentile(psd_channels, 90, axis=1)

# ── Create figure ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(10, 5.5))

# PSD background
ax.fill_between(freqs, psd_lo, psd_hi, alpha=0.15, color="#90caf9", label="PSD range (10-90th %ile)")
ax.plot(freqs, psd_db, color="#1565c0", linewidth=1.5, alpha=0.8, label=f"Mean PSD ({subj_id})")

# Learned omega bands — vertical lines + shaded regions
mode_colors = ["#e53935", "#fb8c00", "#43a047", "#8e24aa"]
mode_labels = [
    "Mode 1: Motor unit firing",
    "Mode 2: Mid-frequency EMG",
    "Mode 3: High-frequency EMG",
    "Mode 4: Near-Nyquist",
]

for k in range(len(mean_freq_hz)):
    f_c = mean_freq_hz[k]
    f_std = std_freq_hz[k]
    color = mode_colors[k]

    # Shaded ±1σ band
    ax.axvspan(f_c - f_std, f_c + f_std, alpha=0.12, color=color)

    # Center frequency line
    ax.axvline(f_c, color=color, linewidth=2, linestyle="-", alpha=0.9,
               label=f"{mode_labels[k]}\n  f={f_c:.0f} ± {f_std:.0f} Hz")

    # Init frequency (dashed)
    ax.axvline(init_freq_hz[k], color=color, linewidth=1, linestyle="--", alpha=0.4)

    # Arrow from init to learned
    y_arrow = psd_db.max() - 2 - k * 3
    ax.annotate("", xy=(f_c, y_arrow), xytext=(init_freq_hz[k], y_arrow),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.6))

# Physiological annotations
ax.axvspan(20, 150, alpha=0.04, color="green")
ax.text(85, psd_db.min() + 2, "MU firing\n(20-150 Hz)", ha="center",
        fontsize=7, color="green", alpha=0.7, style="italic")

ax.axvspan(150, 500, alpha=0.04, color="orange")
ax.text(325, psd_db.min() + 2, "Peak EMG\npower", ha="center",
        fontsize=7, color="darkorange", alpha=0.7, style="italic")

# Labels
ax.set_xlabel("Frequency (Hz)", fontsize=11)
ax.set_ylabel("Power Spectral Density (dB)", fontsize=11)
ax.set_title("Learned UVMD Center Frequencies vs. EMG Power Spectrum\n"
             "(NinaPro DB2, 12-channel sEMG, 2 kHz)", fontsize=12)
ax.set_xlim(0, 1000)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.grid(alpha=0.2)

# Add text about convergence
ax.text(0.02, 0.02,
        f"Dashed lines = init (uniform), solid = learned (mean over {len(all_omegas)} subjects)\n"
        f"Arrows show frequency shift during training",
        transform=ax.transAxes, fontsize=7, color="gray", va="bottom")

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"fig_uvmd_omega_psd.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {OUT_DIR / 'fig_uvmd_omega_psd.pdf'}")
