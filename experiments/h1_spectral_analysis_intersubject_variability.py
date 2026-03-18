"""
Hypothesis 1: Spectral Analysis of Inter-Subject Variability in sEMG

Goal: Show that inter-subject variability is heterogeneous across frequency bands.
- Low-frequency components (20-150 Hz, MU firing rates) have higher amplitude CV
- Mid/high-frequency components carry more subject-invariant gesture information
- This motivates frequency-aware processing (UVMD, Freq Band Style Mixing, etc.)

Protocol: E1 only (gestures 1-10, no REST), all 40 subjects, Ninapro DB2.
No model training — pure signal analysis. LOSO-compatible: we characterize
the data, not fit anything.

Outputs (saved to experiments_output/h1_spectral_analysis/):
  - fig1_cv_heatmap_band_gesture.pdf   — CV(band × gesture), main result
  - fig2_cv_heatmap_band_channel.pdf   — CV(band × channel)
  - fig3_cv_by_band_summary.pdf        — bar chart: mean CV per band
  - fig4_psd_overlay_per_gesture.pdf    — PSD overlays showing per-subject spread
  - fig5_fisher_ratio_by_band.pdf       — between-gesture / within-subject variance ratio
  - fig6_normalized_cv_comparison.pdf   — raw vs normalized power CV
  - summary_statistics.json             — all numerical results
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants ────────────────────────────────────────────────────────
SAMPLING_RATE = 2000
NUM_CHANNELS = 12  # Ninapro DB2 has 12 EMG channels

GESTURE_IDS = list(range(1, 11))  # E1: gestures 1-10
GESTURE_NAMES = {
    1: "Index flexion",
    2: "Index extension",
    3: "Middle flexion",
    4: "Middle extension",
    5: "Ring flexion",
    6: "Ring extension",
    7: "Little flexion",
    8: "Little extension",
    9: "Thumb adduction",
    10: "Thumb abduction",
}

FREQ_BANDS = {
    "20-100 Hz":   (20, 100),
    "100-200 Hz":  (100, 200),
    "200-350 Hz":  (200, 350),
    "350-500 Hz":  (350, 500),
    "500-700 Hz":  (500, 700),
    "700-1000 Hz": (700, 1000),
}
BAND_NAMES = list(FREQ_BANDS.keys())
BAND_RANGES = list(FREQ_BANDS.values())

ALL_SUBJECTS = [f"DB2_s{i}" for i in range(1, 41)]

# ── Logging ──────────────────────────────────────────────────────────
def setup_logger():
    logger = logging.getLogger("h1_spectral")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# ── Data loading (minimal, no windowing needed) ──────────────────────
def load_raw_segments(base_dir: Path, subject_id: str, logger: logging.Logger):
    """
    Load E1 raw EMG segments for a single subject.
    Returns dict {gesture_id: [segment_array(T_i, C), ...]}.
    No windowing, no preprocessing — raw signal segments.
    """
    subject_num = subject_id.split("_s")[1]
    file_path = base_dir / subject_id / f"S{subject_num}_E1_A1.mat"

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}, skipping {subject_id}")
        return None

    data = loadmat(str(file_path))
    emg = data["emg"]           # (total_samples, 12)
    stimulus = data["stimulus"].flatten()  # (total_samples,)

    # Segment by gestures (no REST)
    segments = {}
    stim_diff = np.diff(stimulus, prepend=0)
    changes = np.where(stim_diff != 0)[0]

    for i in range(len(changes)):
        start = changes[i]
        end = changes[i + 1] if i + 1 < len(changes) else len(stimulus)
        gid = int(stimulus[start])
        if gid == 0:  # skip REST
            continue
        if gid not in GESTURE_IDS:
            continue
        seg = emg[start:end].copy()
        if gid not in segments:
            segments[gid] = []
        segments[gid].append(seg)

    logger.info(f"{subject_id}: {emg.shape[1]} channels, "
                f"{sum(len(v) for v in segments.values())} segments across "
                f"{len(segments)} gestures")
    return segments


# ── PSD computation ──────────────────────────────────────────────────
def compute_psd_welch(segment: np.ndarray, fs: int = SAMPLING_RATE,
                      nperseg: int = 512, noverlap: int = 256):
    """
    Compute PSD for each channel using Welch's method.
    segment: (T, C)
    Returns: freqs (F,), psd (C, F)
    """
    n_channels = segment.shape[1]
    psds = []
    for ch in range(n_channels):
        freqs, pxx = signal.welch(segment[:, ch], fs=fs,
                                  nperseg=min(nperseg, segment.shape[0]),
                                  noverlap=min(noverlap, segment.shape[0] // 2),
                                  window='hann')
        psds.append(pxx)
    return freqs, np.array(psds)  # (C, F)


def compute_band_power(freqs: np.ndarray, psd: np.ndarray,
                       band: tuple) -> np.ndarray:
    """
    Compute mean power in a frequency band for each channel.
    psd: (C, F)
    Returns: (C,) — mean power per channel in the band.
    """
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not mask.any():
        return np.zeros(psd.shape[0])
    return np.mean(psd[:, mask], axis=1)


# ── Main analysis ────────────────────────────────────────────────────
def run_analysis(base_dir: Path, subjects: list, output_dir: Path, logger: logging.Logger):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data and compute band power per (subject, gesture, channel, band)
    logger.info("=" * 60)
    logger.info("Step 1: Loading data and computing PSD band powers")
    logger.info("=" * 60)

    # Storage: band_power[subj_idx][gesture_id] = (C, B) array
    # where C = channels, B = len(BAND_RANGES)
    all_band_powers = {}  # subject_id -> {gesture_id -> (C, B)}
    all_psds = {}         # subject_id -> {gesture_id -> (C, F)} averaged across reps
    common_freqs = None
    loaded_subjects = []
    n_channels_actual = None

    for subj_id in subjects:
        segments = load_raw_segments(base_dir, subj_id, logger)
        if segments is None:
            continue

        subj_bp = {}
        subj_psd = {}

        for gid in GESTURE_IDS:
            if gid not in segments or len(segments[gid]) == 0:
                continue

            # Average PSD across repetitions for this gesture
            rep_psds = []
            rep_bps = []
            for seg in segments[gid]:
                if seg.shape[0] < 128:  # too short
                    continue
                freqs, psd = compute_psd_welch(seg)
                if common_freqs is None:
                    common_freqs = freqs
                if n_channels_actual is None:
                    n_channels_actual = psd.shape[0]

                bp = np.array([compute_band_power(freqs, psd, b) for b in BAND_RANGES]).T  # (C, B)
                rep_psds.append(psd)
                rep_bps.append(bp)

            if len(rep_psds) == 0:
                continue

            subj_psd[gid] = np.mean(rep_psds, axis=0)   # (C, F)
            subj_bp[gid] = np.mean(rep_bps, axis=0)      # (C, B)

        if len(subj_bp) > 0:
            all_band_powers[subj_id] = subj_bp
            all_psds[subj_id] = subj_psd
            loaded_subjects.append(subj_id)

    logger.info(f"Loaded {len(loaded_subjects)} subjects successfully")
    n_bands = len(BAND_RANGES)
    C = n_channels_actual

    # ── Step 2: Compute CV across subjects for each (gesture, channel, band)
    logger.info("=" * 60)
    logger.info("Step 2: Computing Coefficient of Variation across subjects")
    logger.info("=" * 60)

    # cv_raw[gesture][channel][band] — CV of raw power
    # cv_norm[gesture][channel][band] — CV of normalized (relative) power
    cv_raw = np.full((len(GESTURE_IDS), C, n_bands), np.nan)
    cv_norm = np.full((len(GESTURE_IDS), C, n_bands), np.nan)
    mean_power = np.full((len(GESTURE_IDS), C, n_bands), np.nan)

    for gi, gid in enumerate(GESTURE_IDS):
        # Collect band powers across subjects: (N_subj, C, B)
        powers_list = []
        for subj_id in loaded_subjects:
            if gid in all_band_powers[subj_id]:
                powers_list.append(all_band_powers[subj_id][gid])

        if len(powers_list) < 3:
            continue

        powers = np.array(powers_list)  # (N, C, B)
        mean_p = np.mean(powers, axis=0)  # (C, B)
        std_p = np.std(powers, axis=0)    # (C, B)

        # Raw CV
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.where(mean_p > 0, std_p / mean_p, np.nan)
        cv_raw[gi] = cv
        mean_power[gi] = mean_p

        # Normalized power: proportion of total power per subject
        total_power = np.sum(powers, axis=2, keepdims=True)  # (N, C, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_powers = np.where(total_power > 0, powers / total_power, 0)
        mean_np = np.mean(norm_powers, axis=0)
        std_np = np.std(norm_powers, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_n = np.where(mean_np > 0, std_np / mean_np, np.nan)
        cv_norm[gi] = cv_n

    # ── Step 3: Compute Fisher ratio per band
    # Fisher ratio = between-gesture variance / within-gesture (across-subject) variance
    # High Fisher ratio = band is discriminative AND subject-invariant
    logger.info("=" * 60)
    logger.info("Step 3: Computing Fisher discriminant ratio per frequency band")
    logger.info("=" * 60)

    fisher_ratio = np.full((C, n_bands), np.nan)

    for ch in range(C):
        for bi in range(n_bands):
            # Collect: for each gesture, the list of per-subject band powers
            gesture_means = []
            within_vars = []
            for gi, gid in enumerate(GESTURE_IDS):
                vals = []
                for subj_id in loaded_subjects:
                    if gid in all_band_powers[subj_id]:
                        vals.append(all_band_powers[subj_id][gid][ch, bi])
                if len(vals) < 2:
                    continue
                vals = np.array(vals)
                gesture_means.append(np.mean(vals))
                within_vars.append(np.var(vals))

            if len(gesture_means) < 2:
                continue

            between_var = np.var(gesture_means)
            mean_within_var = np.mean(within_vars)
            if mean_within_var > 0:
                fisher_ratio[ch, bi] = between_var / mean_within_var

    # ── Step 4: Compute LOSO-compatible discriminability
    # For each left-out subject, compute classification-relevant band statistics
    # using ONLY the remaining subjects (strict LOSO)
    logger.info("=" * 60)
    logger.info("Step 4: LOSO cross-validated band discriminability")
    logger.info("=" * 60)

    loso_fisher = np.full((len(loaded_subjects), C, n_bands), np.nan)

    for si, test_subj in enumerate(loaded_subjects):
        train_subjects = [s for s in loaded_subjects if s != test_subj]
        for ch in range(C):
            for bi in range(n_bands):
                gesture_means = []
                within_vars = []
                for gid in GESTURE_IDS:
                    vals = []
                    for subj_id in train_subjects:
                        if gid in all_band_powers[subj_id]:
                            vals.append(all_band_powers[subj_id][gid][ch, bi])
                    if len(vals) < 2:
                        continue
                    vals = np.array(vals)
                    gesture_means.append(np.mean(vals))
                    within_vars.append(np.var(vals))
                if len(gesture_means) < 2:
                    continue
                bv = np.var(gesture_means)
                wv = np.mean(within_vars)
                if wv > 0:
                    loso_fisher[si, ch, bi] = bv / wv

    loso_fisher_mean = np.nanmean(loso_fisher, axis=0)  # (C, B)
    loso_fisher_std = np.nanstd(loso_fisher, axis=0)

    # ── Step 5: Visualizations ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 5: Generating visualizations")
    logger.info("=" * 60)

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    gesture_labels = [f"G{gid}" for gid in GESTURE_IDS]
    band_labels = BAND_NAMES
    channel_labels = [f"Ch{i+1}" for i in range(C)]

    # --- Fig 1: CV heatmap (band × gesture) — MAIN RESULT ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Raw CV: average across channels
    cv_raw_avg = np.nanmean(cv_raw, axis=1)  # (G, B)
    im1 = axes1[0].imshow(cv_raw_avg.T, aspect='auto', cmap='YlOrRd',
                           vmin=0, vmax=np.nanpercentile(cv_raw_avg, 95))
    axes1[0].set_xticks(range(len(GESTURE_IDS)))
    axes1[0].set_xticklabels(gesture_labels, rotation=45)
    axes1[0].set_yticks(range(n_bands))
    axes1[0].set_yticklabels(band_labels)
    axes1[0].set_title("Raw Power CV (across subjects)")
    axes1[0].set_xlabel("Gesture")
    axes1[0].set_ylabel("Frequency Band")
    plt.colorbar(im1, ax=axes1[0], label="CV")

    # Annotate cells
    for i in range(n_bands):
        for j in range(len(GESTURE_IDS)):
            val = cv_raw_avg[j, i]
            if not np.isnan(val):
                axes1[0].text(j, i, f"{val:.2f}", ha='center', va='center',
                              fontsize=8, color='black' if val < 1.0 else 'white')

    # Normalized CV: average across channels
    cv_norm_avg = np.nanmean(cv_norm, axis=1)  # (G, B)
    im2 = axes1[1].imshow(cv_norm_avg.T, aspect='auto', cmap='YlOrRd',
                           vmin=0, vmax=np.nanpercentile(cv_norm_avg, 95))
    axes1[1].set_xticks(range(len(GESTURE_IDS)))
    axes1[1].set_xticklabels(gesture_labels, rotation=45)
    axes1[1].set_yticks(range(n_bands))
    axes1[1].set_yticklabels(band_labels)
    axes1[1].set_title("Normalized Power CV (across subjects)")
    axes1[1].set_xlabel("Gesture")
    plt.colorbar(im2, ax=axes1[1], label="CV")

    for i in range(n_bands):
        for j in range(len(GESTURE_IDS)):
            val = cv_norm_avg[j, i]
            if not np.isnan(val):
                axes1[1].text(j, i, f"{val:.2f}", ha='center', va='center',
                              fontsize=8, color='black' if val < 0.5 else 'white')

    fig1.suptitle("Inter-Subject Variability by Frequency Band and Gesture (E1, 40 subjects)",
                  fontsize=14, y=1.02)
    fig1.tight_layout()
    fig1.savefig(output_dir / "fig1_cv_heatmap_band_gesture.pdf")
    fig1.savefig(output_dir / "fig1_cv_heatmap_band_gesture.png")
    plt.close(fig1)
    logger.info("Saved fig1_cv_heatmap_band_gesture")

    # --- Fig 2: CV heatmap (band × channel) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    cv_raw_by_ch = np.nanmean(cv_raw, axis=0)  # (C, B) — avg across gestures
    im3 = axes2[0].imshow(cv_raw_by_ch.T, aspect='auto', cmap='YlOrRd',
                           vmin=0, vmax=np.nanpercentile(cv_raw_by_ch, 95))
    axes2[0].set_xticks(range(C))
    axes2[0].set_xticklabels(channel_labels, rotation=45)
    axes2[0].set_yticks(range(n_bands))
    axes2[0].set_yticklabels(band_labels)
    axes2[0].set_title("Raw Power CV (avg across gestures)")
    axes2[0].set_xlabel("Channel")
    axes2[0].set_ylabel("Frequency Band")
    plt.colorbar(im3, ax=axes2[0], label="CV")

    for i in range(n_bands):
        for j in range(C):
            val = cv_raw_by_ch[j, i]
            if not np.isnan(val):
                axes2[0].text(j, i, f"{val:.2f}", ha='center', va='center',
                              fontsize=7, color='black' if val < 1.0 else 'white')

    cv_norm_by_ch = np.nanmean(cv_norm, axis=0)
    im4 = axes2[1].imshow(cv_norm_by_ch.T, aspect='auto', cmap='YlOrRd',
                           vmin=0, vmax=np.nanpercentile(cv_norm_by_ch, 95))
    axes2[1].set_xticks(range(C))
    axes2[1].set_xticklabels(channel_labels, rotation=45)
    axes2[1].set_yticks(range(n_bands))
    axes2[1].set_yticklabels(band_labels)
    axes2[1].set_title("Normalized Power CV (avg across gestures)")
    axes2[1].set_xlabel("Channel")
    plt.colorbar(im4, ax=axes2[1], label="CV")

    for i in range(n_bands):
        for j in range(C):
            val = cv_norm_by_ch[j, i]
            if not np.isnan(val):
                axes2[1].text(j, i, f"{val:.2f}", ha='center', va='center',
                              fontsize=7, color='black' if val < 0.5 else 'white')

    fig2.suptitle("Inter-Subject Variability by Frequency Band and Channel",
                  fontsize=14, y=1.02)
    fig2.tight_layout()
    fig2.savefig(output_dir / "fig2_cv_heatmap_band_channel.pdf")
    fig2.savefig(output_dir / "fig2_cv_heatmap_band_channel.png")
    plt.close(fig2)
    logger.info("Saved fig2_cv_heatmap_band_channel")

    # --- Fig 3: Summary bar chart — mean CV per band ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    cv_raw_summary = np.nanmean(cv_raw_avg, axis=0)  # (B,) — avg across gestures
    cv_norm_summary = np.nanmean(cv_norm_avg, axis=0)

    x = np.arange(n_bands)
    width = 0.35
    bars1 = ax3.bar(x - width / 2, cv_raw_summary, width, label='Raw Power CV',
                    color='#e74c3c', alpha=0.8)
    bars2 = ax3.bar(x + width / 2, cv_norm_summary, width, label='Normalized Power CV',
                    color='#3498db', alpha=0.8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(band_labels, rotation=30, ha='right')
    ax3.set_ylabel("Coefficient of Variation")
    ax3.set_title("Inter-Subject CV by Frequency Band\n(averaged across gestures and channels)")
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f"{h:.2f}", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f"{h:.2f}", ha='center', va='bottom', fontsize=9)

    fig3.tight_layout()
    fig3.savefig(output_dir / "fig3_cv_by_band_summary.pdf")
    fig3.savefig(output_dir / "fig3_cv_by_band_summary.png")
    plt.close(fig3)
    logger.info("Saved fig3_cv_by_band_summary")

    # --- Fig 4: PSD overlay per gesture (showing per-subject spread) ---
    fig4, axes4 = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axes4 = axes4.flatten()

    # Use channel 0 (representative)
    representative_ch = 0

    for gi, gid in enumerate(GESTURE_IDS):
        ax = axes4[gi]
        psd_list = []
        for subj_id in loaded_subjects:
            if gid in all_psds[subj_id]:
                psd_list.append(all_psds[subj_id][gid][representative_ch])

        if len(psd_list) == 0:
            continue

        psd_arr = np.array(psd_list)  # (N_subj, F)
        mean_psd = np.mean(psd_arr, axis=0)
        std_psd = np.std(psd_arr, axis=0)

        # Plot individual subjects as thin lines
        for p in psd_arr:
            ax.semilogy(common_freqs, p, color='gray', alpha=0.15, linewidth=0.5)

        # Mean ± std
        ax.semilogy(common_freqs, mean_psd, color='#e74c3c', linewidth=2, label='Mean')
        ax.fill_between(common_freqs,
                        np.maximum(mean_psd - std_psd, 1e-12),
                        mean_psd + std_psd,
                        color='#e74c3c', alpha=0.2)

        # Band boundaries
        for _, (f_lo, f_hi) in FREQ_BANDS.items():
            ax.axvline(f_lo, color='#2c3e50', alpha=0.2, linestyle='--', linewidth=0.5)

        ax.set_title(f"G{gid}: {GESTURE_NAMES.get(gid, '')}", fontsize=10)
        ax.set_xlim(10, 1000)
        if gi >= 5:
            ax.set_xlabel("Frequency (Hz)")
        if gi % 5 == 0:
            ax.set_ylabel("PSD (V²/Hz)")

    fig4.suptitle(f"PSD per Subject (Ch{representative_ch+1}, gray=individual, red=mean±std)",
                  fontsize=14, y=1.02)
    fig4.tight_layout()
    fig4.savefig(output_dir / "fig4_psd_overlay_per_gesture.pdf")
    fig4.savefig(output_dir / "fig4_psd_overlay_per_gesture.png")
    plt.close(fig4)
    logger.info("Saved fig4_psd_overlay_per_gesture")

    # --- Fig 5: Fisher ratio by band (LOSO cross-validated) ---
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))

    # Full-data Fisher ratio averaged across channels
    fisher_avg = np.nanmean(fisher_ratio, axis=0)  # (B,)
    axes5[0].bar(range(n_bands), fisher_avg, color='#27ae60', alpha=0.8)
    axes5[0].set_xticks(range(n_bands))
    axes5[0].set_xticklabels(band_labels, rotation=30, ha='right')
    axes5[0].set_ylabel("Fisher Ratio\n(between-gesture var / within-gesture subject var)")
    axes5[0].set_title("Fisher Discriminant Ratio by Band\n(higher = more discriminative & subject-invariant)")
    axes5[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(fisher_avg):
        if not np.isnan(v):
            axes5[0].text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=9)

    # LOSO Fisher: mean ± std across folds
    loso_f_avg = np.nanmean(loso_fisher_mean, axis=0)  # (B,)
    loso_f_std = np.nanmean(loso_fisher_std, axis=0)
    axes5[1].bar(range(n_bands), loso_f_avg, yerr=loso_f_std,
                 color='#2980b9', alpha=0.8, capsize=3)
    axes5[1].set_xticks(range(n_bands))
    axes5[1].set_xticklabels(band_labels, rotation=30, ha='right')
    axes5[1].set_ylabel("Fisher Ratio (LOSO mean ± std)")
    axes5[1].set_title("LOSO Cross-Validated Fisher Ratio\n(computed on N-1 subjects per fold)")
    axes5[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(loso_f_avg):
        if not np.isnan(v):
            axes5[1].text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=9)

    fig5.tight_layout()
    fig5.savefig(output_dir / "fig5_fisher_ratio_by_band.pdf")
    fig5.savefig(output_dir / "fig5_fisher_ratio_by_band.png")
    plt.close(fig5)
    logger.info("Saved fig5_fisher_ratio_by_band")

    # --- Fig 6: Raw vs Normalized CV comparison ---
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    for gi, gid in enumerate(GESTURE_IDS):
        ax6.scatter(cv_raw_avg[gi], cv_norm_avg[gi],
                    label=f"G{gid}", alpha=0.7, s=60)
    ax6.set_xlabel("Raw Power CV")
    ax6.set_ylabel("Normalized (Relative) Power CV")
    ax6.set_title("Raw vs Normalized Inter-Subject Variability per Band\n(each point = one gesture-band pair)")
    ax6.plot([0, 2], [0, 2], 'k--', alpha=0.3, label='y=x')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax6.grid(alpha=0.3)
    fig6.tight_layout()
    fig6.savefig(output_dir / "fig6_normalized_cv_comparison.pdf")
    fig6.savefig(output_dir / "fig6_normalized_cv_comparison.png")
    plt.close(fig6)
    logger.info("Saved fig6_normalized_cv_comparison")

    # --- Fig 7: Per-band CV distribution across subjects (boxplots) ---
    fig7, ax7 = plt.subplots(figsize=(10, 5))

    # For each band, compute per-subject total band power (avg across gestures and channels)
    band_power_per_subj = np.full((len(loaded_subjects), n_bands), np.nan)
    for si, subj_id in enumerate(loaded_subjects):
        for bi in range(n_bands):
            vals = []
            for gid in GESTURE_IDS:
                if gid in all_band_powers[subj_id]:
                    vals.append(np.mean(all_band_powers[subj_id][gid][:, bi]))
            if vals:
                band_power_per_subj[si, bi] = np.mean(vals)

    # Normalize per subject (relative power)
    total_per_subj = np.nansum(band_power_per_subj, axis=1, keepdims=True)
    rel_power = band_power_per_subj / total_per_subj

    bp_data = [rel_power[:, bi][~np.isnan(rel_power[:, bi])] for bi in range(n_bands)]
    bp = ax7.boxplot(bp_data, labels=band_labels, patch_artist=True)
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax7.set_ylabel("Relative Band Power (proportion of total)")
    ax7.set_title("Distribution of Relative Band Power Across Subjects\n(each box = 40 subjects)")
    ax7.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    fig7.tight_layout()
    fig7.savefig(output_dir / "fig7_band_power_distribution.pdf")
    fig7.savefig(output_dir / "fig7_band_power_distribution.png")
    plt.close(fig7)
    logger.info("Saved fig7_band_power_distribution")

    # ── Step 6: Save summary statistics ─────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 6: Saving summary statistics")
    logger.info("=" * 60)

    summary = {
        "n_subjects": len(loaded_subjects),
        "subjects": loaded_subjects,
        "n_channels": C,
        "n_gestures": len(GESTURE_IDS),
        "gesture_ids": GESTURE_IDS,
        "frequency_bands": {name: list(rng) for name, rng in FREQ_BANDS.items()},
        "raw_cv_per_band": {
            band_labels[bi]: float(np.nanmean(cv_raw_avg[:, bi]))
            for bi in range(n_bands)
        },
        "normalized_cv_per_band": {
            band_labels[bi]: float(np.nanmean(cv_norm_avg[:, bi]))
            for bi in range(n_bands)
        },
        "fisher_ratio_per_band": {
            band_labels[bi]: float(fisher_avg[bi]) if not np.isnan(fisher_avg[bi]) else None
            for bi in range(n_bands)
        },
        "loso_fisher_per_band_mean": {
            band_labels[bi]: float(loso_f_avg[bi]) if not np.isnan(loso_f_avg[bi]) else None
            for bi in range(n_bands)
        },
        "loso_fisher_per_band_std": {
            band_labels[bi]: float(loso_f_std[bi]) if not np.isnan(loso_f_std[bi]) else None
            for bi in range(n_bands)
        },
        "key_findings": [],
    }

    # Determine key findings
    max_cv_band = band_labels[np.nanargmax(cv_raw_summary)]
    min_cv_band = band_labels[np.nanargmin(cv_raw_summary)]
    max_fisher_band = band_labels[np.nanargmax(fisher_avg)]
    min_fisher_band = band_labels[np.nanargmin(fisher_avg)]

    summary["key_findings"] = [
        f"Highest raw CV: {max_cv_band} (CV={np.nanmax(cv_raw_summary):.3f})",
        f"Lowest raw CV: {min_cv_band} (CV={np.nanmin(cv_raw_summary):.3f})",
        f"Most discriminative band (Fisher): {max_fisher_band} (ratio={np.nanmax(fisher_avg):.4f})",
        f"Least discriminative band (Fisher): {min_fisher_band} (ratio={np.nanmin(fisher_avg):.4f})",
        f"CV ratio (max/min across bands): {np.nanmax(cv_raw_summary)/np.nanmin(cv_raw_summary):.2f}x",
    ]

    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary_statistics.json")

    # Print key findings
    logger.info("=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)
    for finding in summary["key_findings"]:
        logger.info(f"  * {finding}")

    logger.info(f"\nAll outputs saved to: {output_dir}")
    return summary, all_band_powers, loaded_subjects


# ── Spectral deviation vs classification F1 ─────────────────────────
def run_correlation_analysis(all_band_powers, loaded_subjects, output_dir, logger):
    """
    Correlate per-subject spectral profile deviation with classification F1.
    Uses results from exp_93 (UVMD, 20 subj), exp_102, and baseline exp_1.
    """
    from scipy.stats import spearmanr

    output_dir = Path(output_dir)
    project_root = Path(__file__).resolve().parent.parent

    # ── Load per-subject F1 from experiments ─────────────────────────
    experiments = {
        "UVMD (exp_93)": project_root / "experiments_output" / "exp_93_unfolded_vmd_uvmd_20260305_190049" / "loso_summary.json",
        "FreqBand Mix (exp_102)": project_root / "experiments_output" / "exp_102_freq_band_style_mixing_20260304_183809" / "loso_summary.json",
        "CNN-GRU Baseline (exp_1)": project_root / "experiments_output" / "exp1_deep_raw_cnn_gru_attention_loso_isolated_v2" / "loso_summary.json",
    }

    exp_f1 = {}  # {exp_name: {subject_id: f1}}
    for exp_name, path in experiments.items():
        if not path.exists():
            logger.warning(f"Results not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)

        subj_f1 = {}
        # Format 1: exp_93 style — loso_metrics.per_subject
        if "loso_metrics" in data and "per_subject" in data["loso_metrics"]:
            for entry in data["loso_metrics"]["per_subject"]:
                subj_f1[entry["test_subject"]] = entry["test_f1_macro"]
        # Format 2: exp_1/exp_102 style — individual_results or results as list
        elif "individual_results" in data and isinstance(data["individual_results"], list):
            for entry in data["individual_results"]:
                subj_f1[entry["test_subject"]] = entry["test_f1_macro"]
        elif "results" in data and isinstance(data["results"], list):
            for entry in data["results"]:
                subj_f1[entry["test_subject"]] = entry["test_f1_macro"]
        # Format 3: results as dict of model -> per_subject
        elif "results" in data and isinstance(data["results"], dict):
            for model_name, model_results in data["results"].items():
                if isinstance(model_results, dict) and "per_subject" in model_results:
                    for entry in model_results["per_subject"]:
                        subj_f1[entry["test_subject"]] = entry.get("test_f1_macro", 0)
                    break

        if subj_f1:
            exp_f1[exp_name] = subj_f1
            logger.info(f"Loaded F1 for {exp_name}: {len(subj_f1)} subjects")
        else:
            logger.warning(f"Could not parse per-subject F1 from {path}")

    if not exp_f1:
        logger.warning("No experiment results found, skipping correlation analysis")
        return

    # ── Compute spectral deviation per subject ───────────────────────
    n_bands = len(BAND_RANGES)

    # Build relative spectral profile for each subject (avg across gestures and channels)
    subj_profiles = {}  # {subj_id: (B,) relative power profile}
    for subj_id in loaded_subjects:
        band_vals = np.zeros(n_bands)
        count = 0
        for gid in GESTURE_IDS:
            if gid in all_band_powers[subj_id]:
                # all_band_powers[subj_id][gid] is (C, B)
                band_vals += np.mean(all_band_powers[subj_id][gid], axis=0)
                count += 1
        if count > 0:
            band_vals /= count
            total = np.sum(band_vals)
            if total > 0:
                subj_profiles[subj_id] = band_vals / total

    # Mean profile across all subjects
    all_profiles = np.array(list(subj_profiles.values()))
    mean_profile = np.mean(all_profiles, axis=0)

    # Spectral deviation: L2 distance from mean profile (normalized)
    subj_deviation = {}
    for subj_id, profile in subj_profiles.items():
        subj_deviation[subj_id] = np.linalg.norm(profile - mean_profile)

    # ── Fig 8: Scatter plot — spectral deviation vs F1 ───────────────
    n_exp = len(exp_f1)
    fig8, axes8 = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5), squeeze=False)
    axes8 = axes8.flatten()

    correlation_results = {}

    for idx, (exp_name, subj_f1) in enumerate(exp_f1.items()):
        ax = axes8[idx]
        common_subjs = sorted(set(subj_deviation.keys()) & set(subj_f1.keys()))
        if len(common_subjs) < 5:
            continue

        devs = np.array([subj_deviation[s] for s in common_subjs])
        f1s = np.array([subj_f1[s] for s in common_subjs])

        ax.scatter(devs, f1s, c='#2980b9', s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

        # Label each point with subject ID
        for i, s in enumerate(common_subjs):
            short = s.replace("DB2_", "")
            ax.annotate(short, (devs[i], f1s[i]), fontsize=7,
                        xytext=(4, 4), textcoords='offset points', alpha=0.7)

        # Spearman correlation
        rho, pval = spearmanr(devs, f1s)
        correlation_results[exp_name] = {"rho": float(rho), "p_value": float(pval), "n": len(common_subjs)}

        # Trend line
        z = np.polyfit(devs, f1s, 1)
        p = np.poly1d(z)
        x_line = np.linspace(devs.min(), devs.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='#e74c3c', alpha=0.6, linewidth=1.5)

        ax.set_xlabel("Spectral Profile Deviation\n(L2 dist from mean relative power)")
        ax.set_ylabel("Test F1-macro")
        sig_str = f"p={pval:.3f}" if pval >= 0.001 else f"p={pval:.1e}"
        ax.set_title(f"{exp_name}\n(Spearman rho={rho:.3f}, {sig_str}, n={len(common_subjs)})")
        ax.grid(alpha=0.3)

    fig8.suptitle("Spectral Deviation vs Classification Performance per Subject",
                  fontsize=14, y=1.02)
    fig8.tight_layout()
    fig8.savefig(output_dir / "fig8_spectral_deviation_vs_f1.pdf")
    fig8.savefig(output_dir / "fig8_spectral_deviation_vs_f1.png")
    plt.close(fig8)
    logger.info("Saved fig8_spectral_deviation_vs_f1")

    # ── Fig 9: Per-band profile of "easy" vs "hard" subjects ─────────
    # Use UVMD F1 to split subjects
    uvmd_f1 = exp_f1.get("UVMD (exp_93)", {})
    if len(uvmd_f1) >= 10:
        common_subjs = sorted(set(subj_profiles.keys()) & set(uvmd_f1.keys()))
        sorted_by_f1 = sorted(common_subjs, key=lambda s: uvmd_f1[s])
        n_group = max(len(sorted_by_f1) // 3, 3)
        hard_subjs = sorted_by_f1[:n_group]
        easy_subjs = sorted_by_f1[-n_group:]

        fig9, ax9 = plt.subplots(figsize=(8, 5))
        x = np.arange(len(BAND_NAMES))

        hard_profiles = np.array([subj_profiles[s] for s in hard_subjs])
        easy_profiles = np.array([subj_profiles[s] for s in easy_subjs])

        hard_mean = np.mean(hard_profiles, axis=0)
        hard_std = np.std(hard_profiles, axis=0)
        easy_mean = np.mean(easy_profiles, axis=0)
        easy_std = np.std(easy_profiles, axis=0)

        width = 0.35
        bars1 = ax9.bar(x - width/2, easy_mean, width, yerr=easy_std,
                        label=f'Easy subjects (top {n_group}, F1>{uvmd_f1[easy_subjs[0]]:.0%})',
                        color='#27ae60', alpha=0.7, capsize=3)
        bars2 = ax9.bar(x + width/2, hard_mean, width, yerr=hard_std,
                        label=f'Hard subjects (bottom {n_group}, F1<{uvmd_f1[hard_subjs[-1]]:.0%})',
                        color='#e74c3c', alpha=0.7, capsize=3)

        ax9.set_xticks(x)
        ax9.set_xticklabels(BAND_NAMES, rotation=30, ha='right')
        ax9.set_ylabel("Relative Band Power (proportion)")
        ax9.set_title("Spectral Profile: Easy vs Hard Subjects\n(grouped by UVMD F1-macro)")
        ax9.legend()
        ax9.grid(axis='y', alpha=0.3)

        # Add subject labels
        easy_labels = ", ".join(s.replace("DB2_", "") for s in easy_subjs)
        hard_labels = ", ".join(s.replace("DB2_", "") for s in hard_subjs)
        ax9.text(0.02, 0.98, f"Easy: {easy_labels}\nHard: {hard_labels}",
                 transform=ax9.transAxes, fontsize=7, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig9.tight_layout()
        fig9.savefig(output_dir / "fig9_easy_vs_hard_spectral_profile.pdf")
        fig9.savefig(output_dir / "fig9_easy_vs_hard_spectral_profile.png")
        plt.close(fig9)
        logger.info("Saved fig9_easy_vs_hard_spectral_profile")

    # ── Save correlation results ─────────────────────────────────────
    corr_summary = {
        "spectral_deviation_metric": "L2 norm of (subject relative power profile - mean profile)",
        "correlations": correlation_results,
        "subject_deviations": {s: float(d) for s, d in subj_deviation.items()},
    }
    with open(output_dir / "correlation_statistics.json", "w") as f:
        json.dump(corr_summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved correlation_statistics.json")

    # Print results
    logger.info("=" * 60)
    logger.info("CORRELATION: Spectral Deviation vs Classification F1")
    logger.info("=" * 60)
    for exp_name, res in correlation_results.items():
        sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "ns"
        logger.info(f"  {exp_name}: rho={res['rho']:.3f}, p={res['p_value']:.4f} {sig} (n={res['n']})")

    return correlation_results


# ── Entry point ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="H1: Spectral analysis of inter-subject variability")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data directory containing DB2_sN folders")
    parser.add_argument("--output_dir", type=str,
                        default="experiments_output/h1_spectral_analysis",
                        help="Output directory for figures and stats")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated subject IDs (default: all 40)")
    args, _ = parser.parse_known_args()

    logger = setup_logger()

    base_dir = Path(args.data_dir)
    if not base_dir.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent / args.data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / args.output_dir

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    else:
        subjects = ALL_SUBJECTS

    logger.info(f"Data dir: {base_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Subjects: {len(subjects)}")

    summary, all_band_powers, loaded_subjects = run_analysis(base_dir, subjects, output_dir, logger)

    # Correlation analysis: spectral deviation vs classification F1
    logger.info("")
    logger.info("=" * 60)
    logger.info("Running correlation analysis (spectral deviation vs F1)")
    logger.info("=" * 60)
    run_correlation_analysis(all_band_powers, loaded_subjects, output_dir, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
