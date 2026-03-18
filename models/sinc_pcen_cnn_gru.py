"""
SincNet-PCEN Frontend + CNN-GRU-Attention for subject-invariant EMG gesture recognition.

Hypothesis (Exp 61):
    EMG amplitude variability across subjects is analogous to channel gain / speaker
    variability in ASR.  Two speech-inspired techniques are adapted here:

    1. SincFilterbank — learnable bandpass filters parametrized by (f1, f2) cutoff
       frequencies, initialized with Mel-spaced values over [min_freq, max_freq].
       Replaces fixed-frequency pre-processing with a data-driven frequency front-end.

    2. PCENLayer — Per-Channel Energy Normalization with learnable (alpha, delta, root, s).
       Implements an adaptive AGC: X_pcen = (|X| / (eps + EMA(|X|))^alpha + delta)^r - delta^r.
       Makes the representation invariant to channel gain changes (electrode/skin impedance).

Architecture:
    Input (B, C, T)
      → SincFilterbank  → (B, C*K, T)   [K learnable bandpass filters, shared across channels]
      → PCENLayer        → (B, C*K, T)   [per-filter adaptive gain normalization]
      → CNN-BiGRU-Attention              [temporal pattern extractor]
      → Linear classifier → (B, num_classes)

LOSO integrity:
    - ALL parameters (SincFilterbank, PCENLayer, encoder, classifier) are part of the
      PyTorch model and receive gradient updates ONLY from training-subject data.
    - At test time the model is switched to eval() mode and parameters are FROZEN.
    - PCENLayer EMA smoother is re-initialized from each individual window (no
      cross-window or cross-subject state).  The smoother is implemented as a
      causal depthwise 1D convolution — no stateful RNN cells, no leakage.
    - Channel standardization (mean/std) is computed from training windows only
      and applied as a fixed affine transform at inference (in the Trainer).

Reference inspirations:
    - SincNet: Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with
      SincNet," SLT 2018, arXiv:1808.00158
    - PCEN:    Wang et al., "Trainable Frontend For Robust and Far-Field Keyword
      Spotting," ICASSP 2017; Lostanlen et al., "Per-Channel Energy Normalization:
      Why and How," IEEE SPL 2019
    - LEAF:    Zeghidour et al., "LEAF: A Learnable Frontend for Audio Classification,"
      ICLR 2021
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────── helpers ───────────────────────────────────────

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ──────────────────────────── SincFilterbank ────────────────────────────────

class SincFilterbank(nn.Module):
    """
    Learnable bandpass filterbank using parametric sinc functions.

    Each of the K filters is defined by two learnable cutoff frequencies (f1, f2) in Hz.
    The impulse response of filter k is:
        h_k[n] = 2*f2_k * sinc(2*f2_k*n) - 2*f1_k * sinc(2*f1_k*n)
    where sinc is the *normalized* sinc function: sinc(x) = sin(π x) / (π x).

    The SAME filter bank is applied to every EMG channel independently via a
    grouped convolution (same weights across all C channels).  Output expands
    the channel dimension: (B, C, T) → (B, C*K, T).

    Parameters are initialized using Mel-spaced frequencies over [min_freq, max_freq]
    and stored in log-space to guarantee positivity throughout training.

    Args:
        num_filters:   K — number of bandpass filters
        kernel_size:   sinc kernel length in samples (must be odd, e.g. 51)
        sample_rate:   EMG sampling rate in Hz (NinaPro DB2: 2000 Hz)
        min_freq:      lowest representable cutoff frequency in Hz
        max_freq:      highest representable cutoff frequency in Hz
        in_channels:   C — number of raw EMG channels
    """

    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 51,
        sample_rate: int = 2000,
        min_freq: float = 5.0,
        max_freq: float = 500.0,
        in_channels: int = 8,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for a symmetric sinc kernel"
        assert min_freq > 0 and max_freq < sample_rate / 2, (
            f"Frequencies must satisfy 0 < min_freq={min_freq} < max_freq={max_freq} < Nyquist={sample_rate/2}"
        )
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.in_channels = in_channels
        self.min_freq_hz = min_freq
        self.max_freq_hz = max_freq

        # Mel-spaced initialization of filter cutoff frequencies.
        # We use num_filters+2 Mel points so that adjacent filters share edges
        # (like a triangular Mel filter bank), giving a clean tiling of [min, max].
        mel_min = _hz_to_mel(min_freq)
        mel_max = _hz_to_mel(max_freq)
        mel_pts = torch.linspace(mel_min, mel_max, num_filters + 2)
        hz_pts = torch.tensor([_mel_to_hz(m.item()) for m in mel_pts])

        # Normalize to [0, 0.5] (Nyquist = 0.5 in normalized frequency)
        f1_init = hz_pts[:-2] / sample_rate  # (K,) lower cutoffs
        f2_init = hz_pts[2:]  / sample_rate  # (K,) upper cutoffs

        # Store in log-frequency space: f = exp(log_f) stays positive during gradient descent
        self.log_f1 = nn.Parameter(torch.log(f1_init))  # (K,)
        self.log_f2 = nn.Parameter(torch.log(f2_init))  # (K,)

        # Sample positions centered at 0: n ∈ {-(L-1)//2, ..., 0, ..., (L-1)//2}
        n = torch.arange(kernel_size) - (kernel_size - 1) // 2
        self.register_buffer("n_", n.float())

        # Hamming window to taper filter edges (reduces spectral leakage / ringing)
        self.register_buffer("window_", torch.hamming_window(kernel_size, periodic=False))

    def _get_filters(self) -> torch.Tensor:
        """
        Compute (K, 1, kernel_size) filter kernels from current learnable parameters.

        Cutoff frequencies are clamped to the valid range [min_freq, max_freq] and
        the constraint f1 < f2 is enforced by clamping.  Filters are L2-normalized
        to unit energy so that the amplitude of the output does not depend on the
        frequency band width.
        """
        min_f = self.min_freq_hz / self.sample_rate
        max_f = self.max_freq_hz / self.sample_rate

        f1 = torch.exp(self.log_f1).clamp(min_f, max_f - 1e-4)  # (K,)
        f2 = torch.exp(self.log_f2).clamp(min_f + 1e-4, max_f)  # (K,)
        f2 = torch.max(f2, f1 + 1e-4)                            # enforce f2 > f1

        # n: (1, L),  f1/f2: (K, 1)  → broadcast to (K, L)
        n  = self.n_.unsqueeze(0)   # (1, L)
        f1 = f1.unsqueeze(1)        # (K, 1)
        f2 = f2.unsqueeze(1)        # (K, 1)

        # h[n] = 2*f2*sinc(2*f2*n) - 2*f1*sinc(2*f1*n)
        # torch.sinc uses the normalized convention: sinc(x) = sin(π x) / (π x), sinc(0)=1
        h = (2.0 * f2 * torch.sinc(2.0 * f2 * n)
           - 2.0 * f1 * torch.sinc(2.0 * f1 * n))  # (K, L)

        # Taper with Hamming window
        h = h * self.window_.unsqueeze(0)  # (K, L)

        # L2-normalize each filter to unit energy
        h = h / (h.norm(dim=1, keepdim=True) + 1e-8)

        return h.unsqueeze(1)  # (K, 1, L) — ready for F.conv1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the learned filterbank to raw EMG.

        The same K filters are applied to each of the C channels independently
        via a reshape + grouped convolution trick:
            (B, C, T) → (B*C, 1, T) → conv1d(kernels) → (B*C, K, T) → (B, C*K, T)

        Symmetric (same) padding is used to preserve the temporal length T.

        Args:
            x:   (B, C, T) — channel-standardized raw EMG windows
        Returns:
            out: (B, C*K, T) — filtered signal, same length T
        """
        B, C, T = x.shape
        kernels = self._get_filters()  # (K, 1, L)
        pad = self.kernel_size // 2

        x_flat = x.reshape(B * C, 1, T)                           # (B*C, 1, T)
        out_flat = F.conv1d(x_flat, kernels, padding=pad)          # (B*C, K, T)
        out = out_flat.reshape(B, C * self.num_filters, T)         # (B, C*K, T)
        return out


# ──────────────────────────── PCENLayer ─────────────────────────────────────

class PCENLayer(nn.Module):
    """
    Per-Channel Energy Normalization (PCEN) with fully learnable parameters.

    For each channel c at time step t in a window:

        M_c[t] = (1 - s_c) * M_c[t-1] + s_c * |X_c[t]|     (EMA smoother)
        PCEN_c[t] = (|X_c[t]| / (ε + M_c[t])^α_c + δ_c)^r_c  - δ_c^r_c

    All four parameters (α, δ, r, s) are:
        - Learnable scalars per frequency channel (one value per C*K channel)
        - Shared across all subjects — subject-specific amplitude variation is
          *normalized away* by the gain term (ε + M)^α, not remembered per-subject
        - Updated ONLY via gradients from training data

    The EMA is approximated by a finite causal depthwise 1D convolution:
        M ≈ F.conv1d(|X|, ema_kernel, groups=C, padding=L-1)
    where ema_kernel[c, L-1-j] = s_c * (1-s_c)^j  (j lags, older → smaller weight).
    This is equivalent to the recursive IIR (up to truncation error) and is
    fully differentiable with respect to s.

    LOSO integrity:
        - The EMA smoother is recomputed fresh for every forward pass window.
          There is NO recurrent state carried across windows, subjects, or batches.
        - The causal kernel ensures that M[t] depends only on |X[0..t]| — no
          future samples contaminate the normalization.
        - Channel standardization (applied before this layer by the Trainer)
          is computed from training data only and does not use test statistics.

    Args:
        num_channels:      number of filter channels to normalize (C * num_sinc_filters)
        ema_kernel_length: length of the truncated EMA convolution kernel.
                           For s=0.04, L=128 captures >99.4% of the infinite IIR energy.
        eps:               numerical stability constant
    """

    def __init__(
        self,
        num_channels: int,
        ema_kernel_length: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.ema_kernel_length = ema_kernel_length
        self.eps = eps

        # ---------- learnable per-channel PCEN parameters ----------
        # Stored in transformed space to keep them in valid ranges during training.

        # alpha ∈ (0, 1): gain suppression exponent.
        # alpha=1 → full division by gain (CMVN-like); alpha=0 → no normalization.
        # Initialized to logit(0.98) so sigmoid → 0.98 (aggressive, from PCEN paper).
        self.log_alpha = nn.Parameter(
            torch.full((num_channels,), math.log(0.98 / (1.0 - 0.98)))
        )

        # delta > 0: additive bias for stability (avoids x/0 at silence).
        # softplus(log_delta) → delta > 0. Init log(2.0) → delta ≈ 2.0.
        self.log_delta = nn.Parameter(
            torch.full((num_channels,), math.log(2.0))
        )

        # root r > 0: compression exponent. r=0.5 → square-root compression.
        # softplus(log_root) + 0.1 → r > 0.1.  Init log(0.5).
        self.log_root = nn.Parameter(
            torch.full((num_channels,), math.log(0.5))
        )

        # s ∈ (0, 1): EMA smoothing coefficient.
        # s=0.04 → time constant ≈ 1/s = 25 samples @ 2000 Hz ≈ 12.5 ms.
        # sigmoid(log_s_raw) = s.  Init logit(0.04).
        s_init = 0.04
        self.log_s = nn.Parameter(
            torch.full((num_channels,), math.log(s_init / (1.0 - s_init)))
        )

    def _build_ema_kernel(self, device: torch.device) -> torch.Tensor:
        """
        Build per-channel truncated exponential EMA kernel for depthwise conv1d.

        The IIR smoother M[t] = (1-s)*M[t-1] + s*x[t] has impulse response:
            h[j] = s * (1-s)^j   for lag j = 0 (most recent), 1, 2, ...

        For a causal convolution output[t] = Σ_j coeff_j * input[t-j]:
            conv1d kernel[L-1-j] = h[j]  (most recent sample → last kernel position)
            ↔  kernel = flip(h)   in the time dimension

        Kernel is normalized by its sum to compensate for IIR truncation.

        Returns:
            kernel: (C, 1, L) — per-channel causal EMA kernels
        """
        C = self.num_channels
        L = self.ema_kernel_length
        s = torch.sigmoid(self.log_s)  # (C,), s ∈ (0, 1)

        j = torch.arange(L, device=device, dtype=s.dtype)  # lags 0..L-1

        # impulse[c, j] = s_c * (1-s_c)^j  — most recent lag first
        impulse = (s.unsqueeze(1)
                   * (1.0 - s.unsqueeze(1)) ** j.unsqueeze(0))  # (C, L)

        # Truncation normalization: Σ h[j] = 1 - (1-s)^L.  Renormalize to 1.
        impulse = impulse / (impulse.sum(dim=1, keepdim=True) + 1e-8)

        # Flip so that the most recent sample aligns with the last kernel position.
        # This is required by F.conv1d's causal padding scheme (see module docstring).
        kernel = impulse.flip(dims=[1])  # (C, L)

        return kernel.unsqueeze(1)  # (C, 1, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PCEN normalization to a batch of filtered windows.

        Args:
            x:   (B, C, T) — output of SincFilterbank (real-valued)
        Returns:
            out: (B, C, T) — PCEN-normalized signal
        """
        B, C, T = x.shape
        assert C == self.num_channels, (
            f"PCENLayer expects {self.num_channels} channels, got {C}"
        )

        # Constrain parameters to their valid ranges
        alpha = torch.sigmoid(self.log_alpha)        # (C,) ∈ (0, 1)
        delta = F.softplus(self.log_delta) + 1e-6    # (C,) > 0
        root  = F.softplus(self.log_root) + 0.1      # (C,) > 0.1

        # Take absolute value — PCEN operates on the magnitude of the signal
        x_mag = x.abs()  # (B, C, T)

        # EMA smoother via causal depthwise 1D convolution.
        # We pad L-1 zeros on the LEFT so that the first output position t=0
        # uses only x_mag[:, :, 0] (and L-1 zeros), making it fully causal.
        L = min(self.ema_kernel_length, T)
        kernel = self._build_ema_kernel(x.device)[:, :, -L:]  # (C, 1, L) — truncate if T < L

        x_padded = F.pad(x_mag, (L - 1, 0))            # (B, C, T + L - 1)
        M = F.conv1d(x_padded, kernel, groups=C)        # (B, C, T)

        # PCEN formula
        # (1) Normalize by adaptive gain: x_mag / (eps + M)^alpha
        # (2) Apply bias + root compression: (·+ delta)^r - delta^r
        alpha_bc = alpha.view(1, C, 1)
        delta_bc = delta.view(1, C, 1)
        root_bc  = root.view(1, C, 1)

        gain    = (self.eps + M).pow(alpha_bc)          # (B, C, T)
        normed  = x_mag / gain                           # (B, C, T)
        pcen    = (normed + delta_bc).pow(root_bc) - delta_bc.pow(root_bc)

        return pcen


# ──────────────────────────── Encoder ───────────────────────────────────────

class SincPCENEncoder(nn.Module):
    """
    CNN-BiGRU with attention pooling — operates on PCEN feature maps.

    Architecture mirrors SharedEncoder from exp_31 (DisentangledCNNGRU) but
    uses wider CNN channels to handle the expanded C*K input.
    """

    def __init__(
        self,
        in_channels: int,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        layers = []
        prev_ch = in_channels
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.5),
            ])
            prev_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_dim = gru_hidden * 2  # bidirectional

        self.attn = nn.Sequential(
            nn.Linear(self.gru_dim, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C*K, T) — PCEN features
        Returns:
            ctx: (B, gru_dim) — attended context vector
        """
        h = self.cnn(x)              # (B, cnn[-1], T')
        h = h.transpose(1, 2)        # (B, T', cnn[-1])
        gru_out, _ = self.gru(h)     # (B, T', gru_dim)

        w = torch.softmax(self.attn(gru_out), dim=1)  # (B, T', 1)
        ctx = (w * gru_out).sum(dim=1)                 # (B, gru_dim)
        return ctx


# ──────────────────────────── Full model ────────────────────────────────────

class SincPCENCNNGRU(nn.Module):
    """
    SincNet-PCEN frontend + CNN-GRU-Attention classifier for EMG gesture recognition.

    Motivation:
        Raw EMG amplitude varies substantially across subjects due to electrode
        placement, skin impedance, and muscle anatomy.  In speech, analogous
        channel/speaker gain variability is tackled with PCEN + learnable front-ends
        (SincNet, LEAF).  This model applies the same idea to sEMG.

    Pipeline:
        raw EMG (B, C, T)
          → channel standardization [done by Trainer, from training stats only]
          → SincFilterbank  → (B, C*K, T)
          → PCENLayer        → (B, C*K, T)
          → CNN-BiGRU-Attention
          → gesture logits (B, num_classes)

    All parameters are trained on training subjects and FROZEN at test time.

    Args:
        in_channels:      C — number of EMG channels (8 for NinaPro DB2)
        num_classes:      number of gesture classes
        num_sinc_filters: K — filters per channel
        sinc_kernel_size: sinc impulse response length (odd)
        sample_rate:      EMG sampling rate in Hz
        min_freq:         minimum filter cutoff in Hz
        max_freq:         maximum filter cutoff in Hz
        pcen_ema_length:  length of the truncated PCEN EMA kernel
        cnn_channels:     list of CNN channel sizes in the encoder
        gru_hidden:       GRU hidden size (bidirectional → gru_hidden*2 output)
        gru_layers:       number of GRU layers
        dropout:          dropout probability
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        num_sinc_filters: int = 32,
        sinc_kernel_size: int = 51,
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

        frontend_channels = in_channels * num_sinc_filters

        self.sinc = SincFilterbank(
            num_filters=num_sinc_filters,
            kernel_size=sinc_kernel_size,
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
        h = self.sinc(x)       # (B, C*K, T)  learnable bandpass filtering
        h = self.pcen(h)       # (B, C*K, T)  adaptive gain normalization
        ctx = self.encoder(h)  # (B, gru_dim) temporal encoding
        return self.classifier(ctx)
