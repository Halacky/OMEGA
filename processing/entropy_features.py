"""
Rényi Entropy + Statistical Complexity features for EMG signals.

Implements the Complexity-Entropy (C-H) plane analysis from:
    Rosso et al., "Distinguishing Noise from Chaos," Physical Review Letters, 2007.

The C-H plane provides a 2D characterization of signal dynamics:
    - H: Normalized Rényi entropy (disorder measure, 0=deterministic, 1=uniform noise)
    - C: Statistical complexity = Q_JS * H  (structure measure)
      where Q_JS is the Jensen-Shannon divergence from the uniform distribution,
      normalized so C ∈ [0, C_max].

For EMG classification, the C-H plane captures:
    - Different gesture types produce different muscle recruitment patterns
      with distinct entropy/complexity signatures
    - Rest (low entropy, low complexity) vs active gesture (higher entropy)
    - Fine finger movements (high complexity) vs power grasp (lower complexity)

Features extracted per channel:
    - Rényi entropy H_α for α ∈ {0.5, 1.0 (Shannon), 2.0, 3.0}
    - Normalized Shannon entropy H_norm
    - Jensen-Shannon divergence Q_JS from uniform
    - Statistical complexity C = Q_JS * H_norm
    - Spectral variants of all above (on PSD instead of ordinal pattern PDF)

Total: 7 features per channel in time domain + 7 in frequency domain = 14 per channel.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ──────────────────────── Ordinal Patterns (Bandt-Pompe) ─────────────────────

def _ordinal_pattern_distribution(x: np.ndarray, order: int = 3, delay: int = 1) -> np.ndarray:
    """
    Compute the ordinal pattern probability distribution (Bandt & Pompe, 2002).

    Vectorized: builds all sub-sequences at once, batch-argsort, then converts
    permutations to Lehmer code indices via precomputed factorial weights.

    Args:
        x:      (T,) — 1D time series.
        order:  D — embedding dimension (pattern length = D+1). Default 3.
        delay:  τ — time delay between elements. Default 1.

    Returns:
        prob: ((D+1)!,) — probability distribution over ordinal patterns.
    """
    D = order
    L = D + 1
    n_patterns = math.factorial(L)
    T = len(x)

    n_sub = T - D * delay
    if n_sub <= 0:
        return np.ones(n_patterns) / n_patterns

    # Build embedding matrix: (n_sub, L)
    col_offsets = np.arange(L) * delay
    row_starts = np.arange(n_sub)[:, None]
    embedded = x[row_starts + col_offsets]  # (n_sub, L)

    # Batch argsort → permutation matrix (n_sub, L)
    perms = np.argsort(embedded, axis=1, kind='mergesort')

    # Vectorized Lehmer code: for each position i, count elements in
    # positions i+1..L-1 that are smaller than the element at position i
    fact_weights = np.array([math.factorial(L - 1 - i) for i in range(L)])
    smaller_counts = np.zeros((n_sub, L), dtype=np.int32)
    for i in range(L - 1):
        smaller_counts[:, i] = np.sum(
            perms[:, i + 1:] < perms[:, i:i + 1], axis=1
        )

    indices = (smaller_counts * fact_weights).sum(axis=1)  # (n_sub,)
    counts = np.bincount(indices.astype(np.intp), minlength=n_patterns)
    return counts.astype(np.float64) / counts.sum()


# ──────────────────────── Entropy Functions ──────────────────────────────────

def renyi_entropy(prob: np.ndarray, alpha: float) -> float:
    """
    Rényi entropy of order α.

    H_α(P) = 1/(1-α) * log₂(Σ p_i^α)

    Special cases:
        α → 1: Shannon entropy H₁ = -Σ p_i log₂(p_i)
        α = 0: Hartley entropy (log of support size)
        α = 2: collision entropy -log₂(Σ p_i²)
        α → ∞: min-entropy -log₂(max p_i)

    Args:
        prob:  probability distribution (sums to 1).
        alpha: order parameter (> 0, ≠ 1 for general formula).

    Returns:
        H_α in bits.
    """
    prob = prob[prob > 0]  # remove zeros
    if len(prob) == 0:
        return 0.0

    if abs(alpha - 1.0) < 1e-10:
        # Shannon entropy (limit as α → 1)
        return float(-np.sum(prob * np.log2(prob)))
    else:
        return float(1.0 / (1.0 - alpha) * np.log2(np.sum(prob ** alpha)))


def normalized_shannon_entropy(prob: np.ndarray) -> float:
    """Normalized Shannon entropy H_norm ∈ [0, 1], divided by log₂(N)."""
    N = len(prob)
    if N <= 1:
        return 0.0
    H = renyi_entropy(prob, alpha=1.0)
    H_max = np.log2(N)
    return H / H_max if H_max > 0 else 0.0


def jensen_shannon_divergence(prob: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between prob and the uniform distribution.

    Q_JS = H((P + U)/2) - H(P)/2 - H(U)/2

    where U is the uniform distribution over the same support.
    Result is normalized to [0, 1] by dividing by Q_max.
    """
    N = len(prob)
    if N <= 1:
        return 0.0

    uniform = np.ones(N) / N
    mixture = 0.5 * (prob + uniform)

    H_mix = renyi_entropy(mixture, alpha=1.0)
    H_p = renyi_entropy(prob, alpha=1.0)
    H_u = np.log2(N)  # entropy of uniform

    Q_JS = H_mix - 0.5 * H_p - 0.5 * H_u

    # Normalize by maximum possible Q_JS
    # Q_max occurs for a distribution concentrated on one symbol:
    # Q_max = -0.5 * [(N+1)/N * log₂(N+1) - 2*log₂(2N) + log₂(N)]
    Q_max = _q_max(N)
    if Q_max > 0:
        return float(Q_JS / Q_max)
    return 0.0


def _q_max(N: int) -> float:
    """Maximum Jensen-Shannon divergence for N symbols (Rosso et al., 2007)."""
    if N <= 1:
        return 0.0
    # Q_max = -0.5 * { (N+1)/N * log₂(N+1) - 2*log₂(2*N) + log₂(N) }
    val = -0.5 * (
        (N + 1.0) / N * np.log2(N + 1.0)
        - 2.0 * np.log2(2.0 * N)
        + np.log2(N)
    )
    return float(val)


def statistical_complexity(prob: np.ndarray) -> float:
    """
    Statistical complexity C = Q_JS * H_norm (Rosso et al., 2007).

    Combines disorder (entropy) with structure (divergence from uniform).
    C = 0 for both perfectly ordered and perfectly random signals.
    C > 0 for signals with non-trivial structure.
    """
    H = normalized_shannon_entropy(prob)
    Q = jensen_shannon_divergence(prob)
    return float(Q * H)


# ──────────────────────── Feature Extractor ───────────────────────────────────

@dataclass
class EntropyComplexityExtractor:
    """
    Extract Rényi entropy + Complexity-Entropy plane features from EMG windows.

    For each channel, computes features in two domains:
      1. Time domain: ordinal pattern distribution (Bandt-Pompe)
      2. Frequency domain: normalized PSD as probability distribution

    Features per channel per domain (7):
      - H_0.5 (Rényi α=0.5)
      - H_1.0 (Shannon)
      - H_2.0 (collision entropy)
      - H_3.0 (Rényi α=3)
      - H_norm (normalized Shannon)
      - Q_JS (Jensen-Shannon divergence from uniform)
      - C (statistical complexity = Q_JS * H_norm)

    Total: 14 features per channel.

    Args:
        sampling_rate:  EMG sampling rate (used for PSD computation).
        order:          Ordinal pattern embedding dimension D (default 3, patterns of length 4).
        delay:          Ordinal pattern time delay τ (default 1).
        alpha_values:   Rényi entropy orders to compute.
        logger:         Optional Python logger.
    """
    sampling_rate: int = 2000
    order: int = 3
    delay: int = 1
    alpha_values: tuple = (0.5, 1.0, 2.0, 3.0)
    logger: Optional[logging.Logger] = None

    def _extract_entropy_features(self, prob: np.ndarray) -> np.ndarray:
        """Extract 7 features from a probability distribution."""
        feats = []
        for alpha in self.alpha_values:
            feats.append(renyi_entropy(prob, alpha))
        feats.append(normalized_shannon_entropy(prob))
        feats.append(jensen_shannon_divergence(prob))
        feats.append(statistical_complexity(prob))
        return np.array(feats, dtype=np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract entropy + complexity features from a batch of EMG windows.

        Args:
            X: (N, T, C) — batch of raw EMG windows.

        Returns:
            features: (N, F) where F = C * 14 (7 time-domain + 7 freq-domain per channel).
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        N, T, C = X.shape
        n_feat_per_domain = len(self.alpha_values) + 3  # 4 Rényi + H_norm + Q_JS + C
        n_feat_per_channel = n_feat_per_domain * 2  # time + freq
        features = np.zeros((N, C * n_feat_per_channel), dtype=np.float32)

        for i in range(N):
            feat_list = []
            for c in range(C):
                signal = X[i, :, c]

                # --- Time domain: ordinal pattern distribution ---
                prob_ord = _ordinal_pattern_distribution(signal, self.order, self.delay)
                td_feats = self._extract_entropy_features(prob_ord)

                # --- Frequency domain: normalized PSD ---
                fft_vals = np.fft.rfft(signal)
                psd = np.abs(fft_vals) ** 2
                psd_sum = psd.sum()
                if psd_sum > 0:
                    prob_psd = psd / psd_sum
                else:
                    prob_psd = np.ones_like(psd) / len(psd)
                fd_feats = self._extract_entropy_features(prob_psd)

                feat_list.append(np.concatenate([td_feats, fd_feats]))

            features[i] = np.concatenate(feat_list)

        if self.logger:
            self.logger.info(
                f"[EntropyComplexityExtractor] Extracted {n_feat_per_channel} features "
                f"per channel × {C} channels = {C * n_feat_per_channel} total"
            )

        return features

    @property
    def feature_names(self) -> list:
        """Return list of feature names for documentation."""
        names = []
        alpha_names = [f"H_{a}" for a in self.alpha_values]
        base = alpha_names + ["H_norm", "Q_JS", "C_complexity"]
        for domain in ["time", "freq"]:
            for name in base:
                names.append(f"{domain}_{name}")
        return names
