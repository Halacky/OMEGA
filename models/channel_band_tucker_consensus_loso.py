"""
Channel-Band Factorized Representation with Temporal Consensus
(Hypothesis 5: Tucker Decomposition + Multi-Resolution Temporal Agreement)

Architecture:
  Raw EMG (B, C, T)
  → STFT per channel → log-magnitude spectrogram (B, C, F, T')
  → Soft Freq AGC:   per-freq temporal mean removal + learnable scale
  → Tucker Channel Factor U_ch: projects C → r_c
      * intermediate h_ch (B, r_c, F, T') fed to subject adversary via GRL
  → Tucker Freq Factor U_f: projects F → r_f
      * core tensor h_cf (B, r_c, r_f, T')
  → Temporal Consensus: classify from full T' and individual temporal quarters;
      minimise KL divergence between full-window and quarter predictions
  → Classifier on globally-pooled core: h_cf.mean(-1) → (B, r_c * r_f) → MLP

Three-axis subject-invariance:
  (1) Channel axis: gradient reversal on h_ch    (DANN schedule)
  (2) Freq axis:    Soft AGC per-freq normalization
  (3) Temporal axis: quarter-window KL consistency loss

LOSO Safety:
  - STFT uses a fixed Hann window — no statistics estimated from data.
  - Soft AGC uses per-sample temporal mean only (no cross-sample stats).
  - BatchNorm uses running statistics at eval() — frozen, no test-subject leakage.
  - Subject adversary is computed only when return_all=True (training).
  - Temporal consensus loss computed only when return_all=True (training).
  - No test-time adaptation, no subject-specific branches.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer (DANN)
# ─────────────────────────────────────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    """Autograd function: forward identity, backward negates gradient by alpha."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (alpha,) = ctx.saved_tensors
        return -alpha.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for domain-adversarial training (DANN)."""

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return _GRLFunction.apply(x, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# STFT 3D Encoder
# ─────────────────────────────────────────────────────────────────────────────

class STFT3DEncoder(nn.Module):
    """
    Convert raw EMG (B, C, T) → log-magnitude spectrogram (B, C, F, T').

    Uses a fixed Hann window — no learnable parameters.
    LOSO-safe: transform is data-independent.

    Args:
        n_fft:       STFT window length (default 64 → F = 33 freq bins)
        hop_length:  STFT hop length   (default 16 → T' = (600-64)//16+1 = 34)
    """

    def __init__(self, n_fft: int = 64, hop_length: int = 16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1
        # Fixed Hann window — non-trainable buffer.
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — raw EMG, channels first
        Returns:
            (B, C, F, T') — log-magnitude spectrogram
        """
        B, C, T = x.shape
        # Reshape for batched stft: (B*C, T)
        x_flat = x.reshape(B * C, T)

        # torch.stft — real input, onesided=True → F = n_fft//2+1
        X_complex = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # → (B*C, F, T')

        X_mag = X_complex.abs()          # magnitude (B*C, F, T')
        X_log = torch.log1p(X_mag)       # log(1 + |STFT|) — better dynamic range

        T_prime = X_log.shape[-1]
        return X_log.reshape(B, C, self.freq_bins, T_prime)  # (B, C, F, T')


# ─────────────────────────────────────────────────────────────────────────────
# Soft Frequency AGC
# ─────────────────────────────────────────────────────────────────────────────

class SoftFreqAGC(nn.Module):
    """
    Per-frequency-band amplitude normalization.

    Removes per-sample, per-(channel, frequency) temporal mean, then applies
    a learnable per-frequency scale γ_f.

    Motivation: subjects differ in absolute EMG amplitude per frequency band
    (fatigue, electrode pressure). Subtracting the temporal mean within each
    (channel, freq) position reduces this inter-subject variability.

    LOSO-safe:
      - Normalization uses per-sample statistics (temporal mean of the sample
        itself) — no cross-sample statistics. Identical at train and test time.
      - γ_f is trained on training-fold data; at test time the same γ_f is
        applied without update.

    Args:
        n_freq_bins: number of frequency bins (F = n_fft//2+1)
    """

    def __init__(self, n_freq_bins: int):
        super().__init__()
        # Learnable per-frequency scale (analogous to LayerNorm gamma)
        self.gamma = nn.Parameter(torch.ones(n_freq_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, F, T') — log-magnitude spectrogram
        Returns:
            (B, C, F, T') — amplitude-normalized spectrogram
        """
        # Per-sample temporal mean subtraction for each (C, F) position
        mean_t = x.mean(dim=-1, keepdim=True)   # (B, C, F, 1)
        x_norm = x - mean_t

        # Apply learnable per-frequency scale: γ (F,) → broadcast (B, C, F, T')
        x_scaled = x_norm * self.gamma[None, None, :, None]
        return x_scaled


# ─────────────────────────────────────────────────────────────────────────────
# Tucker Channel-Freq Factorization Network
# ─────────────────────────────────────────────────────────────────────────────

class TuckerChannelFreqNet(nn.Module):
    """
    Mode-wise Tucker factorization along Channel and Frequency axes.

    Applies two sequential mode-wise linear projections to the 3D spectrogram:

      Mode-1 (Channel, C → r_c):
        x (B, C, F, T') → h_ch (B, r_c, F, T')

      Mode-2 (Frequency, F → r_f):
        h_ch (B, r_c, F, T') → h_cf (B, r_c, r_f, T')

    Returns both intermediate (h_ch) and final (h_cf) tensors so the caller
    can apply different regularizers per axis:
      - h_ch → subject adversary via GRL
      - h_cf → temporal consensus loss

    BatchNorm after each projection stabilizes training. BN uses running
    statistics at eval() — no test-subject leakage.

    Args:
        n_channels: raw EMG channel count (C = 8)
        n_freq:     STFT frequency bins   (F = 33 for n_fft=64)
        r_c:        channel rank (default 8)
        r_f:        frequency rank (default 16)
    """

    def __init__(
        self,
        n_channels: int,
        n_freq: int,
        r_c: int,
        r_f: int,
    ):
        super().__init__()
        # Mode-1: project C → r_c (applied along channel dimension)
        self.U_ch = nn.Linear(n_channels, r_c, bias=False)
        self.bn_ch = nn.BatchNorm2d(r_c)

        # Mode-2: project F → r_f (applied along frequency dimension)
        self.U_f = nn.Linear(n_freq, r_f, bias=False)
        self.bn_cf = nn.BatchNorm2d(r_c)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, F, T') — normalized spectrogram
        Returns:
            h_ch: (B, r_c, F, T') — channel-factorized representation
            h_cf: (B, r_c, r_f, T') — channel+freq-factorized representation (core)
        """
        # ── Mode-1: channel projection ────────────────────────────────────
        # Permute C to last dim, apply linear, permute back
        x_perm = x.permute(0, 2, 3, 1)          # (B, F, T', C)
        h_ch   = self.U_ch(x_perm)              # (B, F, T', r_c)
        h_ch   = h_ch.permute(0, 3, 1, 2)       # (B, r_c, F, T')
        h_ch   = F.relu(self.bn_ch(h_ch))

        # ── Mode-2: frequency projection ──────────────────────────────────
        # Permute F to last dim, apply linear, permute back
        h_perm = h_ch.permute(0, 1, 3, 2)       # (B, r_c, T', F)
        h_cf   = self.U_f(h_perm)               # (B, r_c, T', r_f)
        h_cf   = h_cf.permute(0, 1, 3, 2)       # (B, r_c, r_f, T')
        h_cf   = F.relu(self.bn_cf(h_cf))

        return h_ch, h_cf


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class ChannelBandTuckerConsensusEMG(nn.Module):
    """
    Channel-Band Factorized EMG Model with Temporal Consensus.

    Three-axis factorization of EMG spectrogram with per-axis subject-invariance:

      Axis 1 — Channel (C):
        Tucker U_ch projects C → r_c.  h_ch is regularized via GRL + subject
        adversary so U_ch learns subject-invariant channel patterns.

      Axis 2 — Frequency (F):
        Soft AGC removes per-frequency amplitude variability (subject-specific
        electrode placement and skin impedance affect amplitude per freq band).
        Tucker U_f then projects F → r_f.

      Axis 3 — Temporal (T'):
        Temporal consensus: classifier applied to full temporal span vs
        individual temporal quarters. KL divergence between full and quarter
        predictions regularizes temporal robustness.

    LOSO Safety:
      - STFT: fixed Hann window, no statistics from data.
      - Soft AGC: per-sample normalization, no cross-sample statistics.
      - Channel standardization: applied BEFORE model (in trainer, training data only).
      - BatchNorm: frozen at model.eval() — no test-subject stats update.
      - Subject adversary: enabled only when return_all=True (training loop).
      - Temporal consensus loss: returned only when return_all=True (training).
      - No test-time adaptation, no subject-specific layers.

    Args:
        n_classes:        number of gesture classes
        n_subjects_train: number of training subjects (for adversary output dim)
        n_channels:       EMG channel count (default 8)
        t_samples:        samples per window (default 600)
        n_fft:            STFT window size  (default 64 → F=33)
        hop_length:       STFT hop length   (default 16 → T'=34)
        r_c:              channel Tucker rank (default 8)
        r_f:              frequency Tucker rank (default 16)
        hidden_dim:       classifier MLP hidden units (default 128)
        dropout:          dropout rate (default 0.3)
    """

    def __init__(
        self,
        n_classes: int,
        n_subjects_train: int,
        n_channels: int = 8,
        t_samples: int = 600,
        n_fft: int = 64,
        hop_length: int = 16,
        r_c: int = 8,
        r_f: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_subjects_train = n_subjects_train
        self.r_c = r_c
        self.r_f = r_f

        n_freq  = n_fft // 2 + 1                             # 33 for n_fft=64
        t_prime = (t_samples - n_fft) // hop_length + 1      # 34 for T=600

        # ── Step 1: STFT (fixed transform, no learnable params) ───────────
        self.stft_enc = STFT3DEncoder(n_fft=n_fft, hop_length=hop_length)

        # ── Step 2: Soft Freq AGC (per-freq amplitude normalization) ──────
        self.freq_agc = SoftFreqAGC(n_freq_bins=n_freq)

        # ── Step 3: Tucker channel + frequency factorization ──────────────
        self.tucker_net = TuckerChannelFreqNet(
            n_channels=n_channels, n_freq=n_freq, r_c=r_c, r_f=r_f,
        )

        # ── Step 4: Gesture classifier (on temporally-pooled core tensor) ─
        # Input dim: r_c * r_f (e.g. 8 × 16 = 128)
        core_dim = r_c * r_f
        self.classifier = nn.Sequential(
            nn.Linear(core_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # LayerNorm: per-sample, no batch stats
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        # ── Step 5: Subject adversary on channel factors (training only) ──
        # Receives temporally-pooled h_ch: (B, r_c, F) → flatten → (B, r_c*F)
        adv_in_dim = r_c * n_freq   # 8 × 33 = 264
        self.grl = GradientReversalLayer()
        self.subject_adversary = nn.Sequential(
            nn.Linear(adv_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_subjects_train),
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _classify_from_hcf(self, h_cf: torch.Tensor) -> torch.Tensor:
        """
        Pool h_cf temporally and classify.

        Args:
            h_cf: (B, r_c, r_f, T_local) — any temporal length
        Returns:
            logits: (B, n_classes)
        """
        z = h_cf.mean(dim=-1)           # (B, r_c, r_f) — global temporal pool
        return self.classifier(z.flatten(1))   # (B, n_classes)

    def _temporal_consensus_loss(self, h_cf: torch.Tensor) -> torch.Tensor:
        """
        Quarter-window temporal consensus loss.

        Encourages the model to produce consistent gesture predictions
        regardless of which temporal quarter of the spectrogram is observed.

        Implementation:
          1. Classify from full temporal span → logits_full (reference, detached
             during KL computation to stabilize training).
          2. Split T' into 4 equal quarters; classify each quarter.
          3. KL( softmax(logits_full / τ) || softmax(logits_qk / τ) ) averaged
             over all quarters.

        Note on correctness:
          Because global temporal pooling is LINEAR, logits_full equals the
          weighted mean of quarter logits. However, KL divergence in softmax
          space is NONLINEAR, so the loss is non-trivial and provides meaningful
          gradient signal: it encourages each temporal view to produce a
          probability distribution close to the full-window distribution.

        LOSO-safe: uses only within-sample statistics, no cross-sample leakage.

        Args:
            h_cf: (B, r_c, r_f, T') — core tensor, full temporal extent
        Returns:
            scalar temporal consensus loss
        """
        T = h_cf.shape[-1]
        if T < 4:
            return h_cf.new_tensor(0.0)

        # Full-window logits (reference — detached to give a stable target)
        logits_full = self._classify_from_hcf(h_cf)
        log_p_full  = F.log_softmax(logits_full.detach() / 2.0, dim=-1)

        quarter = T // 4
        total_kl = h_cf.new_tensor(0.0)
        n_valid  = 0

        for k in range(4):
            start = k * quarter
            end   = (k + 1) * quarter if k < 3 else T
            if end - start < 1:
                continue
            logits_q = self._classify_from_hcf(h_cf[:, :, :, start:end])
            p_q      = F.softmax(logits_q / 2.0, dim=-1)
            total_kl = total_kl + F.kl_div(
                log_p_full, p_q, reduction="batchmean",
            )
            n_valid += 1

        return total_kl / max(n_valid, 1)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor = None,
        grl_alpha: float = 1.0,
        return_all: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            x:           (B, C, T) — channel-standardized raw EMG
            subject_ids: (B,) — subject indices 0…N_train-1 (training only,
                         not used in computation; retained for API symmetry)
            grl_alpha:   GRL reversal strength (DANN schedule, training only)
            return_all:  if True → also compute subject adversary and temporal
                         consensus loss (training mode only)

        Returns:
            dict with:
              "logits"      — (B, n_classes), always present
              "adv_logits"  — (B, n_subjects_train), only if return_all=True
              "cons_loss"   — scalar tensor, only if return_all=True
        """
        # ── 1. STFT ───────────────────────────────────────────────────────
        X_stft = self.stft_enc(x)           # (B, C, F, T')

        # ── 2. Soft Freq AGC ──────────────────────────────────────────────
        X_norm = self.freq_agc(X_stft)      # (B, C, F, T')

        # ── 3. Tucker factorization ───────────────────────────────────────
        h_ch, h_cf = self.tucker_net(X_norm)
        # h_ch: (B, r_c, F, T') — after channel projection
        # h_cf: (B, r_c, r_f, T') — core tensor (channel + freq projected)

        # ── 4. Gesture classification from full temporal span ─────────────
        logits = self._classify_from_hcf(h_cf)     # (B, n_classes)

        if not return_all:
            # ── Inference path: return logits tensor directly ──────────────
            # Consistent with the rest of the codebase (evaluate_numpy calls
            # model(x, return_all=False) and expects a plain tensor).
            return logits

        # ── Training path: compute all auxiliary outputs ───────────────────

        # ── 5. Temporal consensus loss ─────────────────────────────────────
        # KL between full-window prediction and each quarter-window prediction.
        cons_loss = self._temporal_consensus_loss(h_cf)

        # ── 6. Subject adversary on channel factors ────────────────────────
        # Pool h_ch along T': (B, r_c, F, T') → mean over T' → (B, r_c, F)
        # Flatten → (B, r_c * F) → GRL → adversary MLP.
        h_ch_pooled = h_ch.mean(dim=-1).flatten(1)    # (B, r_c * F)
        h_ch_rev    = self.grl(h_ch_pooled, grl_alpha)
        adv_logits  = self.subject_adversary(h_ch_rev) # (B, n_subjects_train)

        return {
            "logits":     logits,
            "adv_logits": adv_logits,
            "cons_loss":  cons_loss,
        }
