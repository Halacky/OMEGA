"""
Stochastic Hypernetwork FIR Deconvolution for Subject-Invariant EMG Classification.

Hypothesis:
    Different subjects correspond to different electrode-skin transfer functions
    (unknown linear FIR filters on the underlying neural drive). By training a
    hypernetwork that generates per-channel depthwise FIR coefficients from a
    random noise vector u ~ N(0, I) — NOT subject IDs — and applying a fresh
    random filter realization to each training sample, the backbone learns features
    that are invariant to any particular transfer function realization.

    This is domain randomization (robotics / ASR) applied to the EMG acquisition
    physics level.

Key design properties:
    FIRHyperNetwork
        - Input: noise u ~ N(0, I) of shape (B, noise_dim).  No subject identity.
        - Output: (B, n_channels, filter_len) per-sample FIR coefficients.
        - Initialization: last-layer weights=0, bias=identity filter.
          → hypernetwork(zeros) == identity for every channel at the start.

    StochasticFIRFrontend
        - Per-sample depthwise 1D FIR via grouped Conv1d (batch × channel → groups).
        - Training: sample_noise=True  → u ~ N(0, I) → different filter per sample.
        - Inference: sample_noise=False → u = 0       → canonical near-identity filter.
        - Regularization (on canonical filter only):
            * Second-order smoothness: penalizes curvature of the mean filter.
            * Band-limiting:           penalizes spectral energy above cutoff_ratio × Nyquist.

    StochasticFIRCNNGRU
        - StochasticFIRFrontend → CNN stack → BiGRU → MHA → FC.
        - Architecture identical to FIRDeconvCNNGRU for fair comparison.

LOSO integrity (never violated):
    ✓  No subject IDs anywhere in the forward pass — only noise vectors.
    ✓  Hypernetwork and backbone trained only on pooled train-subject data.
    ✓  u=0 at test time → fully deterministic, no test-subject information.
    ✓  Regularization targets the canonical (u=0) filter, NOT sampled realizations.
    ✓  model.eval() at test time → BatchNorm uses frozen running stats.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  FIR Hypernetwork: noise → filter coefficients
# ══════════════════════════════════════════════════════════════════════════════

class FIRHyperNetwork(nn.Module):
    """
    3-layer MLP that maps a noise vector to per-channel FIR filter coefficients.

    Initialization ensures hypernetwork(zeros) == identity filter for all channels:
        - Last-layer weight initialized to zero (no sensitivity to u at startup).
        - Last-layer bias set to flattened identity filters (center tap = 1.0).

    As training progresses, the MLP learns which filter perturbations are useful
    for the cross-subject task, while the noise input prevents encoding any fixed
    subject-specific transfer function.

    Args:
        noise_dim:   dimension of the input noise vector.
        n_channels:  number of EMG channels (one independent filter per channel).
        filter_len:  FIR tap length (must be odd for symmetric same-padding).
        hidden_dim:  hidden dimension of the MLP.
    """

    def __init__(
        self,
        noise_dim:  int,
        n_channels: int,
        filter_len: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if filter_len % 2 == 0:
            raise ValueError(f"filter_len must be odd, got {filter_len}")

        self.noise_dim  = noise_dim
        self.n_channels = n_channels
        self.filter_len = filter_len

        # 3-layer MLP with Tanh activations.
        # Tanh bounds the output → filter coefficients stay in a controlled range.
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim,  hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_channels * filter_len),
        )

        # Identity initialization: hypernetwork(zeros) == identity filter.
        with torch.no_grad():
            # Zero all weights in the output layer → no sensitivity to u at startup.
            nn.init.zeros_(self.mlp[-1].weight)

            # Bias = flattened identity filters: center tap=1, all others=0.
            bias = torch.zeros(n_channels * filter_len)
            center = filter_len // 2
            for c in range(n_channels):
                bias[c * filter_len + center] = 1.0
            self.mlp[-1].bias.copy_(bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, noise_dim) noise vectors.  u=0 → identity filter.
        Returns:
            filters: (B, n_channels, filter_len).
        """
        return self.mlp(u).view(-1, self.n_channels, self.filter_len)


# ══════════════════════════════════════════════════════════════════════════════
#  Stochastic FIR Frontend: per-sample depthwise filtering
# ══════════════════════════════════════════════════════════════════════════════

class StochasticFIRFrontend(nn.Module):
    """
    Stochastic depthwise FIR frontend implementing domain randomization at the
    electrode-skin physics level.

    Per-sample depthwise filtering trick (grouped Conv1d):
        Standard Conv1d cannot apply a different filter per batch element.
        We work around this by merging the batch and channel dimensions into a
        single groups axis:
            x:       (B, C, T) → x_r:  (1, B*C, T)   [one "batch", B*C groups]
            filters: (B, C, L) → w_r:  (B*C, 1,  L)   [depthwise weight format]
            F.conv1d(x_r, w_r, groups=B*C, padding=L//2) → (1, B*C, T)
            reshape → (B, C, T)
        This implements truly independent per-sample depthwise filtering with
        full gradient flow through the filter coefficients to the hypernetwork.

    Regularization (on canonical filter only, never on test data):
        - Smoothness: second-order finite-difference curvature penalty on f(u=0).
        - Band-limit: high-frequency spectral energy penalty on f(u=0).
        Both target the mean/canonical filter, not sampled realizations.

    Args:
        n_channels:   number of EMG channels C.
        filter_len:   FIR length in taps (odd, default 63 ≈ 31.5 ms @ 2000 Hz).
        noise_dim:    noise vector dimension for the hypernetwork.
        hyper_hidden: hidden dim of the hypernetwork MLP.
    """

    def __init__(
        self,
        n_channels:   int,
        filter_len:   int = 63,
        noise_dim:    int = 16,
        hyper_hidden: int = 64,
    ) -> None:
        super().__init__()
        if filter_len % 2 == 0:
            raise ValueError(f"filter_len must be odd, got {filter_len}")

        self.n_channels = n_channels
        self.filter_len = filter_len
        self.noise_dim  = noise_dim
        self._pad       = filter_len // 2   # same-padding for odd kernel

        self.hypernetwork = FIRHyperNetwork(
            noise_dim  = noise_dim,
            n_channels = n_channels,
            filter_len = filter_len,
            hidden_dim = hyper_hidden,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _apply_per_sample_depthwise(
        self,
        x:       torch.Tensor,
        filters: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-sample depthwise FIR filters via grouped 1D convolution.

        Preserves the temporal dimension (same-padding with odd filter_len).

        Args:
            x:       (B, C, T) channel-standardized EMG windows.
            filters: (B, C, L) per-sample filter coefficients.
        Returns:
            (B, C, T) filtered signal.
        """
        B, C, T = x.shape

        # Merge (batch, channel) → single groups dimension.
        x_r = x.reshape(1, B * C, T)       # (1, B*C, T)
        # Depthwise weight format: (out_channels, in_channels/groups, kernel_len).
        w_r = filters.reshape(B * C, 1, filters.shape[-1])  # (B*C, 1, L)

        # Grouped depthwise conv: each (sample, channel) pair has its own filter.
        # output_len = T for odd kernel with padding = kernel_len // 2.
        out = F.conv1d(x_r, w_r, padding=self._pad, groups=B * C)  # (1, B*C, T')

        # Safety trim to T (handles any off-by-one from edge cases).
        out = out[:, :, :T]

        return out.reshape(B, C, T)

    # ──────────────────────────────────────────────────────────────────────────

    def mean_filter(self) -> torch.Tensor:
        """
        Return the canonical filter corresponding to u = 0.

        Shape: (1, n_channels, filter_len).
        Used for regularization — never computed from test-subject data.
        """
        device = next(self.hypernetwork.parameters()).device
        dtype  = next(self.hypernetwork.parameters()).dtype
        u_zero = torch.zeros(1, self.noise_dim, device=device, dtype=dtype)
        return self.hypernetwork(u_zero)   # (1, n_channels, filter_len)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:            torch.Tensor,
        sample_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply stochastic depthwise FIR filtering.

        Args:
            x:            (B, C, T) channel-standardized EMG signal.
            sample_noise: True (training) → u ~ N(0, I) per batch element.
                          False (test)    → u = 0 for all batch elements.
        Returns:
            out:     (B, C, T) filtered signal (same shape as input).
            filters: (B, n_channels, filter_len) sampled coefficients.
        """
        B = x.shape[0]
        if sample_noise:
            u = torch.randn(B, self.noise_dim, device=x.device, dtype=x.dtype)
        else:
            u = torch.zeros(B, self.noise_dim, device=x.device, dtype=x.dtype)

        filters = self.hypernetwork(u)                        # (B, C, L)
        out     = self._apply_per_sample_depthwise(x, filters)  # (B, C, T)
        return out, filters

    # ──────────────────────────────────────────────────────────────────────────

    def regularization_loss(
        self,
        lambda_smooth: float = 5e-3,
        lambda_band:   float = 1e-3,
        cutoff_ratio:  float = 0.5,
    ) -> torch.Tensor:
        """
        Regularization loss evaluated on the canonical (u=0) filter.

        Only the mean filter is constrained — sampled realizations during training
        can deviate freely within the bounds imposed by the canonical constraint.
        This decouples "domain diversity" (good for generalization) from
        "canonical plausibility" (good for physical interpretability).

        Smoothness (second-order curvature):
            Penalizes rapid oscillations in the canonical filter's tap sequence.
            Encourages a smooth frequency response, similar to FIRDeconvFrontend.

        Band-limiting (high-frequency spectral energy):
            Penalizes energy in the canonical filter's DFT above cutoff_ratio × Nyquist.
            Prevents the mean filter from introducing high-frequency noise artifacts.

        Args:
            lambda_smooth: weight for curvature penalty.
            lambda_band:   weight for high-frequency energy penalty.
            cutoff_ratio:  fraction of Nyquist above which energy is penalized.
        Returns:
            scalar loss tensor.
        """
        f = self.mean_filter()    # (1, C, L)   — no test data involved

        # Second-order finite difference (curvature) of the canonical filter.
        d1 = f[:, :, 1:] - f[:, :, :-1]       # (1, C, L-1)  first difference
        d2 = d1[:, :, 1:] - d1[:, :, :-1]     # (1, C, L-2)  second difference
        smooth_loss = d2.pow(2).mean()

        # High-frequency spectral energy of the canonical filter.
        F_spec  = torch.fft.rfft(f, dim=-1)    # (1, C, L//2+1)
        n_bins  = F_spec.shape[-1]
        cutoff  = max(1, int(cutoff_ratio * n_bins))
        band_loss = F_spec[:, :, cutoff:].abs().pow(2).mean()

        return lambda_smooth * smooth_loss + lambda_band * band_loss


# ══════════════════════════════════════════════════════════════════════════════
#  Full model: Stochastic FIR Frontend + CNN-BiGRU-Attention
# ══════════════════════════════════════════════════════════════════════════════

class StochasticFIRCNNGRU(nn.Module):
    """
    Stochastic Hypernetwork FIR Frontend + CNN-BiGRU-Multi-head Attention classifier.

    Architecture mirrors FIRDeconvCNNGRU for a controlled comparison; the only
    difference is the FIR frontend: fixed learned weights (exp_65) vs.
    noise-conditioned hypernetwork (exp_77).

    Pipeline (default hyper-params, T=600 input):
        (N, C, T=600)
          ↓ StochasticFIRFrontend  → (N, C, 600)    per-sample random FIR
          ↓ CNN block ×3           → (N, 256, 75)    temporal downsampling ×8
          ↓ permute                → (N, 75, 256)    for GRU
          ↓ 2-layer BiGRU          → (N, 75, 256)    bidirectional
          ↓ Multi-head Attention   → (N, 75, 256)    residual + LayerNorm
          ↓ global avg pool        → (N, 256)
          ↓ FC (256→128→n_classes) → (N, n_classes)

    Training: sample_noise=True  → domain randomization via filter diversity.
    Test:     sample_noise=False → canonical (u=0) deterministic inference.

    Args:
        in_channels:  number of EMG channels C.
        num_classes:  number of gesture classes.
        filter_len:   FIR tap length (odd, default 63).
        noise_dim:    noise vector dimension for the hypernetwork.
        hyper_hidden: hidden dim of the hypernetwork MLP.
        cnn_channels: output channels per CNN block (3 blocks).
        gru_hidden:   BiGRU hidden units per direction.
        num_heads:    multi-head attention heads (must divide gru_hidden * 2).
        dropout:      dropout probability in CNN, GRU, Attention, FC.
    """

    def __init__(
        self,
        in_channels:  int,
        num_classes:  int,
        filter_len:   int             = 63,
        noise_dim:    int             = 16,
        hyper_hidden: int             = 64,
        cnn_channels: Tuple[int, ...] = (64, 128, 256),
        gru_hidden:   int             = 128,
        num_heads:    int             = 4,
        dropout:      float           = 0.3,
    ) -> None:
        super().__init__()

        # ── Stochastic FIR frontend ───────────────────────────────────────
        self.frontend = StochasticFIRFrontend(
            n_channels   = in_channels,
            filter_len   = filter_len,
            noise_dim    = noise_dim,
            hyper_hidden = hyper_hidden,
        )

        # ── CNN stack (same as FIRDeconvCNNGRU) ──────────────────────────
        cnn_layers = []
        ch_in = in_channels
        for ch_out in cnn_channels:
            cnn_layers += [
                nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            ch_in = ch_out
        self.cnn = nn.Sequential(*cnn_layers)

        # ── Bidirectional GRU ─────────────────────────────────────────────
        gru_out_dim = gru_hidden * 2
        self.gru = nn.GRU(
            input_size    = ch_in,
            hidden_size   = gru_hidden,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if dropout > 0.0 else 0.0,
        )

        # ── Multi-head self-attention (residual + LayerNorm) ──────────────
        if gru_out_dim % num_heads != 0:
            raise ValueError(
                f"gru_out_dim={gru_out_dim} must be divisible by num_heads={num_heads}"
            )
        self.attn = nn.MultiheadAttention(
            embed_dim   = gru_out_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(gru_out_dim)

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:            torch.Tensor,
        sample_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:            (N, C, T) channel-standardized EMG windows.
            sample_noise: True  → u ~ N(0, I) per sample (training domain randomization).
                          False → u = 0 for all samples (deterministic test inference).
        Returns:
            logits:  (N, num_classes).
            filters: (N, n_channels, filter_len) — sampled FIR coefficients for logging.
        """
        # Domain randomization: random filter per sample (train) or u=0 (test).
        x_fir, filters = self.frontend(x, sample_noise=sample_noise)  # (N, C, T)

        # CNN temporal feature extraction.
        x_cnn = self.cnn(x_fir)                    # (N, cnn_out, T//8)

        # BiGRU: expects (N, T', C').
        x_gru, _ = self.gru(x_cnn.permute(0, 2, 1))  # (N, T//8, gru_out_dim)

        # Multi-head self-attention with residual + LayerNorm.
        x_attn, _ = self.attn(x_gru, x_gru, x_gru)
        x_out = self.norm(x_gru + x_attn)          # (N, T//8, gru_out_dim)

        # Global average pool over the time axis.
        pooled = x_out.mean(dim=1)                 # (N, gru_out_dim)

        return self.classifier(pooled), filters    # (N, num_classes), (N, C, L)

    # ──────────────────────────────────────────────────────────────────────────

    def regularization_loss(
        self,
        lambda_smooth: float = 5e-3,
        lambda_band:   float = 1e-3,
        cutoff_ratio:  float = 0.5,
    ) -> torch.Tensor:
        """
        Canonical-filter regularization loss (delegated to StochasticFIRFrontend).

        Add to CrossEntropy in the training loop:
            loss = criterion(logits, y) + model.regularization_loss(...)

        The regularization targets u=0 (mean/canonical filter), never test data.
        """
        return self.frontend.regularization_loss(
            lambda_smooth = lambda_smooth,
            lambda_band   = lambda_band,
            cutoff_ratio  = cutoff_ratio,
        )
