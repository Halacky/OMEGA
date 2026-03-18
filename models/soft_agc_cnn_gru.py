"""
Soft AGC Frontend + CNN-BiGRU-Attention for subject-invariant EMG classification.

Hypothesis (Exp 76):
    Aggressive PCEN (exp_61) may destroy amplitude-based discriminative cues that
    vary across gestures.  A softer dynamic range normalizer — bounded exponent,
    no root compression, or purely logarithmic — can reduce inter-subject amplitude
    variation while preserving gesture-relevant amplitude patterns.

Three frontend variants (selected via `frontend_type`):

    "log_affine"
        out[c, t] = log(ε + |x[c, t]|) * scale[c] + bias[c]
        Static log-compression (no adaptive component).  2*C learnable params.
        Initialization: scale=1, bias=0  →  identity = log(ε + |x|).

    "rms_window"
        out[c, t] = x[c, t] / sqrt( mean_{j in [t-W+1, t]}( x[c, j]^2 ) + ε )
        Causal local RMS normalization.  ZERO learnable parameters.
        Implemented as a causal depthwise average conv over x^2.

    "soft_agc"
        M[c, t] = (1-s[c])*M[c, t-1] + s[c]*|x[c, t]|       (causal EMA)
        out[c, t] = x[c, t] / ( M[c, t]^alpha[c] + delta )
        EMA-based AGC.  Key constraints vs PCENLayer (exp_61):
          · alpha ∈ (0, 0.5) via sigmoid(·)*0.5  — HALF the PCEN range
          · delta is FIXED (not learned) — avoids noise-floor exploitation
          · No root compression  — simpler formula, fewer instabilities
          · 2*C learnable params (alpha_raw, log_s)

LOSO integrity (strictly enforced):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Learnable params (scale, bias, alpha_raw, log_s) receive gradients ONLY  │
    │ from training-subject data.  At test time the model is in eval() mode   │
    │ with ALL parameters frozen.                                              │
    │                                                                          │
    │ SoftAGCLayer / LogAffineLayer: EMA/computation is stateless — reset on  │
    │ every forward pass.  No cross-window or cross-subject state.             │
    │                                                                          │
    │ RMSWindowLayer: no parameters, purely deterministic → zero leakage.     │
    │                                                                          │
    │ CNN encoder BatchNorm: uses running stats from training; frozen in eval. │
    └──────────────────────────────────────────────────────────────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────── LogAffineLayer ────────────────────────────────

class LogAffineLayer(nn.Module):
    """
    Static log-compression with learnable per-channel affine transform.

    For each channel c at time t:
        out[c, t] = log(ε + |x[c, t]|) * scale[c] + bias[c]

    Properties:
      - Log compression reduces dynamic range: large amplitudes are compressed
        more than small ones, analogous to dB-scale representation.
      - Learnable (scale, bias) per channel let the model tune the output range;
        initialized to (1, 0) so the untrained model outputs raw log-magnitude.
      - Fully time-invariant: no adaptive/recurrent component within a window.
      - Only 2*C learnable parameters.

    LOSO integrity: scale and bias receive gradients from training data only.
    No running statistics, no state, no test-time adaptation.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        # scale=1, bias=0 → identity: log(ε + |x|)
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.bias  = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T) — channel-standardized EMG windows
        Returns:
            out: (B, C, T) — log-compressed and affine-scaled signal
        """
        log_mag = torch.log(self.eps + x.abs())  # (B, C, T)
        s = self.scale.view(1, -1, 1)
        b = self.bias.view(1, -1, 1)
        return log_mag * s + b


# ──────────────────────────── RMSWindowLayer ────────────────────────────────

class RMSWindowLayer(nn.Module):
    """
    Causal RMS normalization over a sliding window — no learnable parameters.

    For each channel c at time t:
        rms[c, t] = sqrt( (1/W) * sum_{j=0}^{W-1} x[c, t-j]^2  +  ε )
        out[c, t] = x[c, t] / rms[c, t]

    Implemented as a causal depthwise average convolution on x^2:
        x_sq    = x ** 2                             (B, C, T)
        padded  = F.pad(x_sq, (W-1, 0))             (B, C, T + W - 1)
        mean_sq = conv1d(padded, 1/W kernel, groups=C)  (B, C, T)
        rms     = sqrt(mean_sq + ε)
        output  = x / rms

    Window length W = `window_size` (default 50 samples ≈ 25 ms @ 2 kHz).
    This is long enough to capture typical EMG burst durations but short
    relative to gesture segments (600-sample windows).

    LOSO integrity: NO parameters → zero gradient-based or statistic-based
    leakage.  This layer is purely deterministic.
    """

    def __init__(
        self,
        num_channels: int,
        window_size: int = 50,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.window_size = window_size
        self.eps = eps
        # Fixed uniform averaging kernel (1/W per time step, independent per channel)
        kernel = torch.ones(num_channels, 1, window_size) / window_size
        self.register_buffer("kernel_", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T) — channel-standardized EMG windows
        Returns:
            out: (B, C, T) — local-RMS-normalized signal
        """
        B, C, T = x.shape
        x_sq = x ** 2
        # Causal padding: (W-1) zeros on the left → output[t] uses x[max(0, t-W+1)..t]
        x_padded = F.pad(x_sq, (self.window_size - 1, 0))
        mean_sq  = F.conv1d(x_padded, self.kernel_, groups=C)  # (B, C, T)
        rms      = torch.sqrt(mean_sq + self.eps)
        return x / rms


# ──────────────────────────── SoftAGCLayer ──────────────────────────────────

class SoftAGCLayer(nn.Module):
    """
    Soft Automatic Gain Control: x / (EMA(|x|)^alpha + delta), alpha ∈ (0, 0.5].

    Key differences from PCENLayer (exp_61):
      1. alpha ∈ (0, 0.5) via sigmoid(raw)*0.5  — half the PCEN range [0,1]
         → maximum suppression is sqrt(EMA) instead of full EMA normalization.
         At alpha=0.5 gain = sqrt(local energy + delta); at alpha≈0 ≈ passthrough.
      2. delta is FIXED (not learned)  — avoids the model exploiting silence/noise
         floors to effectively invert the normalization.
      3. No root compression (PCEN applies an additional r-th root after division).
         Removing this term simplifies the gradient landscape.
      4. s (EMA coefficient) remains learnable (2*C parameters: alpha_raw, log_s).

    EMA is a causal depthwise 1D convolution (same approach as PCENLayer in exp_61):
        impulse[c, j] = s[c] * (1 - s[c])^j     (most-recent lag first)
        M[c, t]       = sum_j impulse[c, j] * |x[c, t-j]|   (causal)

    This is differentiable w.r.t. s and requires no recurrent state — the
    computation is reset fresh on every forward call.

    LOSO integrity:
      - alpha_raw and log_s receive gradients from training subjects only.
      - delta is fixed → cannot adapt to test subject.
      - EMA is stateless: no cross-window or cross-subject leakage.
      - At test time model.eval() leaves all parameters frozen.

    Args:
        num_channels:      C — number of input EMG channels
        ema_kernel_length: length of truncated EMA kernel (causal convolution)
        delta:             FIXED additive stabilizer in the gain denominator
        eps:               numerical stability constant for EMA clamp
    """

    def __init__(
        self,
        num_channels: int,
        ema_kernel_length: int = 100,
        delta: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.ema_kernel_length = ema_kernel_length
        self.delta = delta   # intentionally NOT an nn.Parameter
        self.eps = eps

        # alpha_raw: sigmoid(alpha_raw) * 0.5  → alpha ∈ (0, 0.5)
        # Init at 0 → alpha = 0.25 (moderate suppression, between PCEN and identity)
        self.alpha_raw = nn.Parameter(torch.zeros(num_channels))

        # log_s: logit of the EMA coefficient s ∈ (0, 1)
        # Init: s = 0.04 → time constant 1/s = 25 samples ≈ 12.5 ms @ 2 kHz
        s_init = 0.04
        self.log_s = nn.Parameter(
            torch.full((num_channels,), math.log(s_init / (1.0 - s_init)))
        )

    def _build_ema_kernel(self, device: torch.device) -> torch.Tensor:
        """
        Build a (C, 1, L) causal EMA kernel for depthwise conv1d.

        The IIR smoother M[t] = (1-s)*M[t-1] + s*|x[t]| has impulse response
            h[j] = s * (1-s)^j  for lag j = 0, 1, 2, ...  (most recent first)

        For conv1d (output[t] = sum_j weight[j] * input[t-j]) we need to
        FLIP h so the most recent sample aligns with the LAST kernel position:
            kernel_flipped[k] = h[L-1-k]

        Kernel is normalized by sum(h) to compensate for IIR truncation error.
        """
        C = self.num_channels
        L = self.ema_kernel_length
        s = torch.sigmoid(self.log_s)  # (C,), s ∈ (0, 1)
        j = torch.arange(L, device=device, dtype=s.dtype)  # lags 0..L-1

        # impulse[c, j] = s_c * (1 - s_c)^j
        impulse = s.unsqueeze(1) * (1.0 - s.unsqueeze(1)) ** j.unsqueeze(0)  # (C, L)
        # Normalize for IIR truncation
        impulse = impulse / (impulse.sum(dim=1, keepdim=True) + 1e-8)
        # Causal flip: last position = most recent sample
        kernel = impulse.flip(dims=[1])  # (C, L)
        return kernel.unsqueeze(1)       # (C, 1, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, T) — channel-standardized EMG windows
        Returns:
            out: (B, C, T) — softly AGC-normalized signal
        """
        B, C, T = x.shape
        assert C == self.num_channels, (
            f"SoftAGCLayer: expected {self.num_channels} channels, got {C}"
        )

        # alpha ∈ (0, 0.5) — bounded gain exponent
        alpha = torch.sigmoid(self.alpha_raw) * 0.5  # (C,)

        x_mag = x.abs()  # (B, C, T)

        # Causal EMA smoother via depthwise conv1d
        L = min(self.ema_kernel_length, T)
        kernel = self._build_ema_kernel(x.device)[:, :, -L:]  # (C, 1, L) — truncate to T
        x_padded = F.pad(x_mag, (L - 1, 0))                    # (B, C, T + L - 1)
        M = F.conv1d(x_padded, kernel, groups=C)                # (B, C, T)

        # Gain: M^alpha + delta  (soft, bounded)
        alpha_bc = alpha.view(1, C, 1)
        gain = M.clamp(min=self.eps).pow(alpha_bc) + self.delta  # (B, C, T)

        return x / gain


# ──────────────────────────── Encoder ───────────────────────────────────────

class CNNGRUEncoder(nn.Module):
    """
    CNN-BiGRU with attention pooling — temporal feature extractor.

    Architecture:
        Conv1d → BN → ReLU → MaxPool  (×len(cnn_channels))
        BiGRU (multi-layer)
        Attention pooling over time → fixed-size context vector

    Same structure as SincPCENEncoder from exp_61 (models/sinc_pcen_cnn_gru.py),
    defined here to decouple this experiment from that module.
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
            x:   (B, C, T) — frontend output (same channel count as raw input)
        Returns:
            ctx: (B, gru_dim) — attended context vector
        """
        h = self.cnn(x)               # (B, cnn_channels[-1], T')
        h = h.transpose(1, 2)         # (B, T', cnn_channels[-1])
        gru_out, _ = self.gru(h)      # (B, T', gru_dim)
        w = torch.softmax(self.attn(gru_out), dim=1)  # (B, T', 1)
        ctx = (w * gru_out).sum(dim=1)                # (B, gru_dim)
        return ctx


# ──────────────────────────── Full model ────────────────────────────────────

class SoftAGCCNNGRU(nn.Module):
    """
    Soft AGC Frontend + CNN-BiGRU-Attention for EMG gesture recognition.

    Experiment 76 hypothesis: a softer dynamic range normalizer reduces inter-subject
    amplitude variation (electrode placement, skin impedance) without destroying the
    amplitude-based gesture cues that PCEN (exp_61) may have over-suppressed.

    Frontend types:
      "log_affine" — log(ε + |x|) * scale + bias  [2*C learnable params]
      "rms_window" — divide by causal local RMS    [0 params, fixed W=rms_window_size]
      "soft_agc"   — x / (EMA(|x|)^α + δ), α ∈ (0,0.5)  [2*C learnable params]

    Pipeline:
        raw EMG (B, C, T)
          → [channel standardization by Trainer, from train stats only]
          → SoftAGC frontend   → (B, C, T)   [same shape, normalized amplitude]
          → CNNGRUEncoder      → (B, gru_dim) [temporal context vector]
          → Linear classifier  → (B, num_classes)

    LOSO integrity:
      - Channel mean/std: computed from training windows only (in the Trainer).
      - Frontend parameters: gradients from training subjects only.
      - RMSWindowLayer: no parameters → zero leakage.
      - model.eval() at inference → BatchNorm uses frozen training running stats.
      - SoftAGCLayer EMA: recomputed fresh per batch, no cross-window state.

    Args:
        in_channels:     C — number of EMG channels (8 for NinaPro DB2)
        num_classes:     number of gesture classes
        frontend_type:   one of {"log_affine", "rms_window", "soft_agc"}
        rms_window_size: causal window length in samples for RMSWindowLayer
        agc_ema_length:  EMA kernel length for SoftAGCLayer
        agc_delta:       FIXED additive stabilizer δ for SoftAGCLayer
        cnn_channels:    list of CNN channel widths in the encoder
        gru_hidden:      GRU hidden size (BiGRU output = 2 × gru_hidden)
        gru_layers:      number of GRU layers
        dropout:         dropout probability
    """

    FRONTEND_TYPES = ("log_affine", "rms_window", "soft_agc")

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        frontend_type: str = "soft_agc",
        rms_window_size: int = 50,
        agc_ema_length: int = 100,
        agc_delta: float = 0.1,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]
        assert frontend_type in self.FRONTEND_TYPES, (
            f"frontend_type must be one of {self.FRONTEND_TYPES}, got '{frontend_type}'"
        )
        self.frontend_type = frontend_type

        if frontend_type == "log_affine":
            self.frontend = LogAffineLayer(in_channels)
        elif frontend_type == "rms_window":
            self.frontend = RMSWindowLayer(in_channels, window_size=rms_window_size)
        elif frontend_type == "soft_agc":
            self.frontend = SoftAGCLayer(
                num_channels=in_channels,
                ema_kernel_length=agc_ema_length,
                delta=agc_delta,
            )

        self.encoder = CNNGRUEncoder(
            in_channels=in_channels,
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
        h = self.frontend(x)       # (B, C, T) — amplitude-normalized
        ctx = self.encoder(h)      # (B, gru_dim) — temporal encoding
        return self.classifier(ctx)

    def count_frontend_params(self) -> int:
        """Return the number of learnable parameters in the frontend."""
        return sum(p.numel() for p in self.frontend.parameters())
