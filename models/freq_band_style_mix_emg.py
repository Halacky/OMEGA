"""
Frequency-Band Style Mixing EMG model (Hypothesis 5 / Experiment 102).

Core idea: Subject-invariant EMG gesture recognition via AdaIN-style mixing of
per-channel statistics applied band-selectively in the frequency domain.

Motivation
----------
EMG inter-subject variance is highly frequency-band-dependent:
  * 20–150 Hz   (low band):  maximal subject variance — subcutaneous fat
                              thickness, impedance variability.
  * 150–450 Hz  (mid band):  moderate variance; concentrates gesture-discriminative
                              motor-unit recruitment patterns.
  * >450 Hz     (high band): noise-dominated, no gesture information.

The mixer applies AdaIN (Adaptive Instance Normalization) to each band:
  * Low band  → aggressive mixing  λ ~ Beta(0.2, 0.2):  broad style coverage.
  * Mid band  → conservative mixing λ ~ Beta(0.8, 0.8):  preserve gesture content.
  * High band → no mixing: noise is not mixed (it would add artefacts).

The effect is that the CNN-GRU encoder is trained on a virtually unlimited variety
of "virtual subjects" while gesture-discriminative mid-band patterns are preserved.

Architecture
------------
Input  (B, C, T)
  └─ FreqBandStyleMixer  [training only; eval → identity]
       ├─ FFT → mask_low  → irfft → AdaIN-low  (α=0.2)
       ├─ FFT → mask_mid  → irfft → AdaIN-mid  (α=0.8)
       └─ high + dc: unchanged
  └─ SharedEncoder (CNN → BiGRU → attention)  →  (B, 256)
  └─ GestureClassifier  →  (B, num_gestures)

LOSO data-leakage audit
-----------------------
✓ FreqBandStyleMixer is ONLY active when model.training=True and subject_labels
  is not None; in model.eval() the input passes through unchanged.
✓ subject_labels contains ONLY indices of TRAINING subjects (0..K-1 where K is
  the number of training subjects in the current LOSO fold). The test subject
  has NO index and is never present in any training batch.
✓ FFT band masks are computed from the signal length and sampling rate alone —
  no data statistics are fitted, no subject information flows into the masks.
✓ AdaIN statistics (μ, σ) are per-sample (instance statistics).  At inference
  the mixer is disabled so NO population statistics from training data are used
  for the test subject.
✓ No test-time adaptation of any kind.
✓ Channel standardization (mean_c, std_c) is computed in the Trainer from
  X_train only and applied identically to val/test windows.

Reference
---------
Inspired by:
  * AdaIN: Huang & Belongie, "Arbitrary Style Transfer in Real-time with
    Adaptive Instance Normalization", ICCV 2017.
  * MixStyle: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
  * MixStyle for SED (Sound Event Detection) — frequency-band selective mixing.
"""

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.disentangled_cnn_gru import SharedEncoder


# ─────────────────────────── helpers ────────────────────────────────────


def _rfft_band_mask(T: int, low_hz: float, high_hz: float,
                    sampling_rate: float, device: torch.device) -> torch.Tensor:
    """
    Create a real-valued mask for torch.fft.rfft output of length T.

    The rfft of a real signal of length T has T//2+1 non-redundant frequency
    bins. This function returns a float mask that is 1.0 inside [low_hz, high_hz]
    and 0.0 outside, with no learnable parameters.

    Args:
        T:             signal length (time samples)
        low_hz:        lower frequency bound (Hz), inclusive
        high_hz:       upper frequency bound (Hz), inclusive
        sampling_rate: sampling rate of the EMG signal (Hz)
        device:        torch device

    Returns:
        mask: (T//2 + 1,) float32 tensor with values in {0.0, 1.0}
    """
    freqs = torch.fft.rfftfreq(T, d=1.0 / sampling_rate).to(device)
    mask = ((freqs >= low_hz) & (freqs <= high_hz)).float()
    return mask


def _extract_band(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Extract a frequency band from a time-domain signal via FFT masking.

    The operation is:
        band = irfft(rfft(x) * mask)

    Gradient flows through both the FFT and the mask multiplication (mask is
    not a learnable parameter but the gradient w.r.t. x still propagates).

    Args:
        x:    (B, C, T) real-valued time-domain signal
        mask: (T//2+1,) float32 — output of _rfft_band_mask

    Returns:
        (B, C, T) time-domain signal containing only frequencies in the band
    """
    T = x.shape[-1]
    X_freq = torch.fft.rfft(x, n=T, dim=-1)              # (B, C, T//2+1) complex
    X_band = X_freq * mask.unsqueeze(0).unsqueeze(0)      # broadcast over B and C
    return torch.fft.irfft(X_band, n=T, dim=-1)           # (B, C, T)


def _adain_mix(
    band: torch.Tensor,
    subject_labels: torch.Tensor,
    alpha: float,
    rng_device: torch.device,
) -> torch.Tensor:
    """
    AdaIN-based style mixing for one frequency band.

    For each sample i in the batch:
      1. Compute per-channel mean μ_i and std σ_i over time (instance statistics).
      2. Find a random partner j from a DIFFERENT training subject.
      3. Sample λ ~ Beta(alpha, alpha).
      4. Mix statistics: μ_mix = λ·μ_i + (1−λ)·μ_j,  σ_mix = λ·σ_i + (1−λ)·σ_j.
      5. Apply AdaIN: normalize band with own stats, rescale with mixed stats.

    LOSO guarantee:
      subject_labels contains ONLY training subject indices.  The test subject's
      data is in a separate DataLoader that is never passed to this function.

    If the entire batch comes from a single subject (possible for small batches),
    the function falls back to identity (partner j = i → λ·stats + (1−λ)·stats =
    own stats → no change).

    Args:
        band:           (B, C, T) time-domain band-filtered signal (float32)
        subject_labels: (B,)      integer training-subject indices
        alpha:          Beta distribution concentration parameter
        rng_device:     torch device for sampling λ

    Returns:
        (B, C, T) signal with mixed per-channel statistics
    """
    B, C, T = band.shape

    # ── Per-sample, per-channel instance statistics ─────────────────────
    # mean / std over TIME only (not across batch or channels)
    mu    = band.mean(dim=-1, keepdim=True)           # (B, C, 1)
    sigma = band.std(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, C, 1)

    # Normalize to zero-mean, unit-variance per sample per channel
    normalized = (band - mu) / sigma                  # (B, C, T)

    # ── Cross-subject partner selection ──────────────────────────────────
    subj_list = subject_labels.cpu().tolist()
    perm_indices = []
    for i in range(B):
        s_i = subj_list[i]
        candidates = [j for j in range(B) if subj_list[j] != s_i]
        if candidates:
            perm_indices.append(random.choice(candidates))
        else:
            # All samples from the same subject → identity mix
            perm_indices.append(i)
    perm = torch.tensor(perm_indices, dtype=torch.long, device=band.device)

    # ── Sample λ ~ Beta(alpha, alpha) for each sample ────────────────────
    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32, device=rng_device),
        torch.tensor(alpha, dtype=torch.float32, device=rng_device),
    )
    lam = beta_dist.sample((B,)).to(band.device).view(B, 1, 1)  # (B, 1, 1)

    # ── Mix statistics (detach partner to stop gradient from arbitrary pairs) ─
    # Gradient flows only through sample i's own statistics, not j's.
    # This is the standard MixStyle training stabiliser.
    mu_partner    = mu[perm].detach()                 # (B, C, 1)
    sigma_partner = sigma[perm].detach()              # (B, C, 1)

    mu_mix    = lam * mu    + (1.0 - lam) * mu_partner    # (B, C, 1)
    sigma_mix = lam * sigma + (1.0 - lam) * sigma_partner  # (B, C, 1)

    # ── Reconstruct with mixed statistics ────────────────────────────────
    return sigma_mix * normalized + mu_mix            # (B, C, T)


# ─────────────────────────── FreqBandStyleMixer ─────────────────────────


class FreqBandStyleMixer(nn.Module):
    """
    Frequency-band selective AdaIN style mixer for EMG signals.

    Applies AdaIN with cross-subject style mixing independently to each
    frequency band. Each band can have a different mixing aggressiveness
    controlled by the Beta distribution concentration parameter alpha:
      * small alpha (e.g. 0.2) → λ distributed broadly over [0,1] → aggressive
      * large alpha (e.g. 0.8) → λ concentrated near 0.5 → conservative

    Training vs. inference behaviour
    ---------------------------------
    Training (model.train(), subject_labels provided):
        1. Decompose x into band_low, band_mid via FFT bandpass.
        2. Apply band-selective AdaIN mixing to band_low and band_mid.
        3. Reconstruct: x_mixed = x + Δ_low + Δ_mid  (residual update).
    Inference (model.eval() or no subject_labels):
        Return x UNCHANGED.  No mixing, no population statistics used.

    The high band (>high_band[1] Hz) and the dc/very-low component (<low_band[0] Hz)
    are never touched — they are implicitly included in the reconstruction residual
    without modification.

    Args:
        sampling_rate:   EMG sampling rate in Hz (default 2000)
        low_band:        (f_lo, f_hi) Hz boundaries for the low frequency band
        mid_band:        (f_lo, f_hi) Hz boundaries for the mid frequency band
        low_mix_alpha:   Beta concentration for low-band mixing (aggressive)
        mid_mix_alpha:   Beta concentration for mid-band mixing (conservative)
    """

    def __init__(
        self,
        sampling_rate: int = 2000,
        low_band: Tuple[float, float] = (20.0, 150.0),
        mid_band: Tuple[float, float] = (150.0, 450.0),
        low_mix_alpha: float = 0.2,
        mid_mix_alpha: float = 0.8,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.low_band = low_band
        self.mid_band = mid_band
        self.low_mix_alpha = low_mix_alpha
        self.mid_mix_alpha = mid_mix_alpha

        # No learnable parameters — all operations are data-driven per-sample.

    def forward(
        self,
        x: torch.Tensor,
        subject_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, C, T) — per-channel standardised EMG windows
            subject_labels: (B,) int  — training subject indices (0..K-1)
                            Required during training; ignored at inference.

        Returns:
            (B, C, T) — x with band-selectively mixed statistics in training,
                        or x unchanged at inference.

        LOSO safety:
            This function is called ONLY from FreqBandStyleMixEMG.forward() which
            gates the call on self.training.  At evaluation time this code is
            never reached and the test subject's signal is never modified.
        """
        if not self.training or subject_labels is None:
            return x

        T = x.shape[-1]
        device = x.device

        # ── Band extraction via FFT bandpass ─────────────────────────────
        mask_low = _rfft_band_mask(T, self.low_band[0], self.low_band[1],
                                   self.sampling_rate, device)
        mask_mid = _rfft_band_mask(T, self.mid_band[0], self.mid_band[1],
                                   self.sampling_rate, device)

        band_low = _extract_band(x, mask_low)   # (B, C, T) — 20–150 Hz component
        band_mid = _extract_band(x, mask_mid)   # (B, C, T) — 150–450 Hz component

        # ── Band-selective AdaIN mixing ───────────────────────────────────
        band_low_mixed = _adain_mix(
            band_low, subject_labels, self.low_mix_alpha, device
        )
        band_mid_mixed = _adain_mix(
            band_mid, subject_labels, self.mid_mix_alpha, device
        )

        # ── Residual reconstruction ───────────────────────────────────────
        # The high-band and dc/sub-20 Hz components are implicitly preserved:
        #   x_mixed = (x - band_low - band_mid + high_band)
        #             + band_low_mixed + band_mid_mixed
        #           = x + (band_low_mixed - band_low) + (band_mid_mixed - band_mid)
        x_mixed = x + (band_low_mixed - band_low) + (band_mid_mixed - band_mid)
        return x_mixed


# ─────────────────────────── Full model ─────────────────────────────────


class FreqBandStyleMixEMG(nn.Module):
    """
    EMG gesture classifier with frequency-band selective style mixing.

    The FreqBandStyleMixer is applied as a differentiable pre-processing step
    only during training, making the CNN-GRU encoder robust to subject-specific
    low-frequency statistics without corrupting gesture-discriminative mid-band
    patterns.

    Architecture
    ------------
    Training forward (model.train(), subject_labels provided):
        x (B,C,T) → FreqBandStyleMixer → SharedEncoder (CNN+BiGRU+Attn)
                  → GestureClassifier → logits (B, num_gestures)

    Inference forward (model.eval()):
        x (B,C,T) → SharedEncoder → GestureClassifier → logits (B, num_gestures)
        [FreqBandStyleMixer is skipped entirely]

    Args:
        in_channels:     number of EMG channels (default 8)
        num_gestures:    number of gesture classes
        sampling_rate:   EMG sampling rate in Hz (default 2000)
        classifier_dim:  hidden size of the 2-layer gesture classifier head
        low_band:        (f_lo, f_hi) Hz for low-frequency band
        mid_band:        (f_lo, f_hi) Hz for mid-frequency band
        low_mix_alpha:   Beta α for low-band (aggressive, e.g. 0.2)
        mid_mix_alpha:   Beta α for mid-band (conservative, e.g. 0.8)
        cnn_channels:    CNN block channel sizes (default [32, 64, 128])
        gru_hidden:      GRU hidden size per direction (bidirectional → ×2 output)
        gru_layers:      number of GRU layers
        dropout:         dropout probability
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_gestures: int = 10,
        sampling_rate: int = 2000,
        classifier_dim: int = 128,
        low_band: Tuple[float, float] = (20.0, 150.0),
        mid_band: Tuple[float, float] = (150.0, 450.0),
        low_mix_alpha: float = 0.2,
        mid_mix_alpha: float = 0.8,
        cnn_channels: Optional[list] = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.band_mixer = FreqBandStyleMixer(
            sampling_rate=sampling_rate,
            low_band=low_band,
            mid_band=mid_band,
            low_mix_alpha=low_mix_alpha,
            mid_mix_alpha=mid_mix_alpha,
        )

        # Shared CNN-BiGRU-Attention encoder (reused from exp_31 / exp_60)
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # 2 × gru_hidden = 256

        # Two-layer gesture classifier
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, classifier_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_gestures),
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, C, T) — channel-standardised EMG windows
            subject_labels: (B,) int  — training subject indices.
                            Must be provided when model.training=True and
                            style mixing is desired; otherwise ignored.

        Returns:
            (B, num_gestures) — gesture class logits

        LOSO guarantee:
            At model.eval(), FreqBandStyleMixer is skipped entirely.
            subject_labels is not used and does not need to be provided.
        """
        # Apply band-selective style mixing (no-op in eval mode)
        x = self.band_mixer(x, subject_labels)

        # Encode + classify
        features = self.encoder(x)           # (B, shared_dim)
        return self.classifier(features)     # (B, num_gestures)
