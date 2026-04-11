"""
Learnable Fourier Series Filterbank + PCEN + CNN-GRU-Attention for EMG.

Key idea:
    Instead of SincNet's 2-parameter (f1, f2) bandpass filters, each filter is
    parameterized as a **learnable Fourier series** — a weighted sum of cosine
    and sine harmonics with trainable amplitudes and phases:

        h_k[n] = Σ_{m=1}^{M} a_{k,m} cos(2π f_{k,m} n / sr) + b_{k,m} sin(2π f_{k,m} n / sr)

    where both the amplitudes (a, b) AND the frequencies (f) are learnable.

    This gives strictly more expressive power than SincNet:
      - SincNet: 2 params per filter (f1, f2 cutoffs) → forced bandpass shape
      - Fourier: 3M params per filter (a_m, b_m, f_m) → arbitrary frequency response

    The filters can learn bandpass, notch, comb, or any smooth frequency response
    that the data requires — not just the rectangular-in-freq bandpass of sinc.

    Initialization: M harmonics per filter, uniformly spaced in [fmin, fmax],
    with random amplitudes. Hamming window applied to final kernel.

Architecture:
    Input (B, C, T)
      → LearnableFourierFilterbank → (B, C*K, T)
      → PCENLayer                  → (B, C*K, T)
      → CNN-BiGRU-Attention
      → Linear classifier → (B, num_classes)

LOSO integrity: same as SincPCENCNNGRU — all parameters trained on training
subjects only, frozen at test time. PCEN EMA recomputed per window.

References:
    - FNet: Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms,"
      NAACL 2022 (Fourier as learned transformation)
    - LEAF: Zeghidour et al., "LEAF: A Learnable Frontend for Audio
      Classification," ICLR 2021 (learnable filterbank concept)
    - SincNet: Ravanelli & Bengio, 2018 (baseline comparison)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sinc_pcen_cnn_gru import PCENLayer, SincPCENEncoder


class LearnableFourierFilterbank(nn.Module):
    """
    Filterbank where each filter is a learnable Fourier series.

    Each of K filters is defined by M harmonics with learnable:
      - frequencies f_{k,m} (stored in log-space for positivity)
      - amplitudes  a_{k,m} (cosine coefficients)
      - amplitudes  b_{k,m} (sine coefficients)

    The impulse response of filter k:
        h_k[n] = w[n] * Σ_m (a_{k,m} cos(2π f_{k,m} n) + b_{k,m} sin(2π f_{k,m} n))

    where w[n] is a Hamming window and frequencies are normalized (f/sr).

    Applied to all C channels independently via grouped conv1d.
    Output: (B, C, T) → (B, C*K, T).

    Args:
        num_filters:    K — number of output filters
        num_harmonics:  M — Fourier harmonics per filter
        kernel_size:    filter length in samples (odd)
        sample_rate:    EMG sampling rate in Hz
        min_freq:       minimum initialization frequency in Hz
        max_freq:       maximum initialization frequency in Hz
        in_channels:    C — number of raw EMG channels
    """

    def __init__(
        self,
        num_filters: int = 32,
        num_harmonics: int = 8,
        kernel_size: int = 51,
        sample_rate: int = 2000,
        min_freq: float = 5.0,
        max_freq: float = 500.0,
        in_channels: int = 8,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert min_freq > 0 and max_freq < sample_rate / 2

        self.num_filters = num_filters
        self.num_harmonics = num_harmonics
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.in_channels = in_channels

        K, M = num_filters, num_harmonics

        # Initialize frequencies: each filter gets M harmonics spread across
        # a sub-band. Filter k covers [fk_low, fk_high] with M evenly spaced
        # harmonics inside that range.
        f_init = torch.zeros(K, M)
        band_edges = torch.linspace(min_freq, max_freq, K + 1)
        for k in range(K):
            f_low = band_edges[k].item()
            f_high = band_edges[k + 1].item()
            f_init[k] = torch.linspace(f_low, f_high, M)

        # Normalize to [0, 0.5] (Nyquist-relative)
        f_init_norm = f_init / sample_rate

        # Store in log-space for positivity
        self.log_freq = nn.Parameter(torch.log(f_init_norm.clamp(min=1e-6)))  # (K, M)

        # Fourier amplitudes: random init with small magnitude
        self.cos_amp = nn.Parameter(torch.randn(K, M) * 0.1)  # (K, M)
        self.sin_amp = nn.Parameter(torch.randn(K, M) * 0.1)  # (K, M)

        # Time axis centered at 0
        n = torch.arange(kernel_size) - (kernel_size - 1) // 2
        self.register_buffer("n_", n.float())  # (L,)

        # Hamming window
        self.register_buffer("window_", torch.hamming_window(kernel_size, periodic=False))

    def _get_filters(self) -> torch.Tensor:
        """
        Compute (K, 1, L) filter kernels from current parameters.

        Returns:
            kernels: (K, 1, kernel_size) ready for F.conv1d
        """
        K, M = self.num_filters, self.num_harmonics

        # Recover frequencies, clamp to valid range
        max_f = 0.5 - 1e-4  # just below Nyquist
        freq = torch.exp(self.log_freq).clamp(1e-6, max_f)  # (K, M)

        # n: (1, 1, L), freq: (K, M, 1) → broadcast
        n = self.n_.unsqueeze(0).unsqueeze(0)          # (1, 1, L)
        f = freq.unsqueeze(2)                           # (K, M, 1)

        # Phase arguments: 2π f n
        phase = 2.0 * math.pi * f * n                  # (K, M, L)

        # Weighted sum of harmonics
        a = self.cos_amp.unsqueeze(2)                   # (K, M, 1)
        b = self.sin_amp.unsqueeze(2)                   # (K, M, 1)

        h = (a * torch.cos(phase) + b * torch.sin(phase)).sum(dim=1)  # (K, L)

        # Apply Hamming window
        h = h * self.window_.unsqueeze(0)               # (K, L)

        # L2-normalize each filter
        h = h / (h.norm(dim=1, keepdim=True) + 1e-8)

        return h.unsqueeze(1)  # (K, 1, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T) — raw EMG
        Returns:
            out: (B, C*K, T) — filtered signal
        """
        B, C, T = x.shape
        kernels = self._get_filters()  # (K, 1, L)
        pad = self.kernel_size // 2

        x_flat = x.reshape(B * C, 1, T)
        out_flat = F.conv1d(x_flat, kernels, padding=pad)
        out = out_flat.reshape(B, C * self.num_filters, T)
        return out

    def get_frequency_response(self, n_fft: int = 512) -> torch.Tensor:
        """Return magnitude frequency response for visualization. Shape: (K, n_fft//2+1)."""
        kernels = self._get_filters().squeeze(1)  # (K, L)
        H = torch.fft.rfft(kernels, n=n_fft, dim=1)
        return H.abs().detach()


class FourierPCENCNNGRU(nn.Module):
    """
    Learnable Fourier filterbank + PCEN + CNN-GRU-Attention classifier.

    Drop-in replacement for SincPCENCNNGRU with a more expressive frontend.

    Args:
        in_channels:      C — number of EMG channels
        num_classes:      number of gesture classes
        num_filters:      K — filters in the Fourier filterbank
        num_harmonics:    M — Fourier harmonics per filter
        kernel_size:      filter length (odd)
        sample_rate:      Hz
        min_freq:         Hz — minimum init frequency
        max_freq:         Hz — maximum init frequency
        pcen_ema_length:  PCEN EMA kernel length
        cnn_channels:     encoder CNN channels
        gru_hidden:       GRU hidden size
        gru_layers:       number of GRU layers
        dropout:          dropout rate
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        num_filters: int = 32,
        num_harmonics: int = 8,
        kernel_size: int = 51,
        sample_rate: int = 2000,
        min_freq: float = 5.0,
        max_freq: float = 500.0,
        pcen_ema_length: int = 128,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        frontend_channels = in_channels * num_filters

        self.fourier_fb = LearnableFourierFilterbank(
            num_filters=num_filters,
            num_harmonics=num_harmonics,
            kernel_size=kernel_size,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            in_channels=in_channels,
        )

        self.pcen = PCENLayer(
            num_channels=frontend_channels,
            ema_kernel_length=pcen_ema_length,
        )

        self.encoder = SincPCENEncoder(
            in_channels=frontend_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.gru_dim, self.encoder.gru_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.gru_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, C, T) — channel-standardized raw EMG windows
        Returns:
            logits: (B, num_classes)
        """
        h = self.fourier_fb(x)    # (B, C*K, T)
        h = self.pcen(h)          # (B, C*K, T)
        ctx = self.encoder(h)     # (B, gru_dim)
        return self.classifier(ctx)
