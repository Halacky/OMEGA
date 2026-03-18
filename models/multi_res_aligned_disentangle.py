"""
Multi-Resolution Aligned Disentanglement model for LOSO EMG gesture recognition.

Hypothesis 3: Two-stage scheme — Alignment first, then Disentanglement —
inspired by "From Consistency to Complementarity" (2026).

Motivation
----------
Previous disentanglement experiments (exp_31, 57, 59, 60, 89) simultaneously
optimize CE + reconstruction + disentanglement losses, causing well-known VAE
gradient conflicts.  This model separates the two objectives:

  Stage 1 — Alignment (contrastive):
    Treat the EMG signal as multimodal: 3 frequency bands are treated as 3
    "modalities" of the same gesture event.  Per-band ECAPA-TDNN encoders +
    NT-Xent contrastive loss align the representations so that bands of the
    SAME window are close in embedding space — creating a shared, frequency-
    invariant gesture representation.

  Stage 2 — Disentanglement (complementarity):
    Per-band specific encoders learn what is UNIQUE to each band.  A Gradient
    Reversal Layer (GRL) forces specific features to be maximally uninformative
    about gesture class — they capture band-specific subject variation instead
    (electrode impedance at different frequency ranges, skin-tissue filtering).
    Only the aligned representation is used for classification.

Architecture
------------
  Input (B, C, T) — EMG, channels-first

  1. BandSplitter — FFT masking (differentiable, no parameters):
       Band 0: [  0, 200) Hz  — dominant inter-subject variance
       Band 1: [200, 500) Hz  — gesture-discriminative motor-unit recruitment
       Band 2: [500, fs/2) Hz — high-frequency residual

  2. SoftAGC per band — causal EMA amplitude normalization (2×C params/band)
       Learnable alpha ∈ (0, 0.5), fixed delta — prevents noise-floor exploit.

  3. MiniECAPAEncoder per band — lightweight 2-block ECAPA (C=64):
       → aligned_emb_i  (B, embed_dim)

  4. Projection head per band — Linear + L2-normalize → (B, proj_dim)
       Used only for Stage-1 contrastive alignment loss.

  5. Aligned aggregate: element-wise mean of 3 embeddings → z_aligned (B, E)

  6. GestureClassifier: Linear(E, num_classes)

  Stage-2 components (active only during training, Stage 2):
  7. SpecificEncoder per band — shallow 2-layer CNN → (B, spec_embed_dim)
  8. GradientReversalLayer — negates gradients flowing into SpecificEncoder
  9. AdversarialClassifier per band — Linear(spec_embed_dim, num_classes)
       CE on adv_logits trains adversarial classifier; GRL pushes SpecificEncoder
       to maximise adversarial classifier loss → specific feats uninformative.

LOSO Integrity
--------------
  BandSplitter: FFT bins from physics (sampling_rate, cutoffs), no data stats.
  SoftAGC: alpha/s parameters receive gradients from training subjects only.
  All BN running stats: computed from training subjects; frozen at eval().
  Channel standardization (mean_c, std_c): trainer computes from X_train only.
  Contrastive loss: pairs from within-batch training windows only; test subject
    is never present in any training batch.
  Adversarial loss: uses gesture labels (not subject identities); test subject
    never in batch → zero subject-identity leakage.
  Inference: model.eval(), stage=1 forward only. No test-time adaptation.

Input format:  (B, C_emg, T)  — channels-first (transposed by trainer)
Output format: dict with "logits" (B, num_classes), "projections" (list of 3),
               optionally "adv_logits" (list of 3) when stage=2.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.soft_agc_cnn_gru import SoftAGCLayer
from models.ecapa_tdnn_emg import SERes2NetBlock, AttentiveStatisticsPooling


# ─────────────────────────── Band Splitter ───────────────────────────────────

class BandSplitter(nn.Module):
    """
    Differentiable FFT-based frequency-band splitter (no learnable parameters).

    Decomposes an EMG signal (B, C, T) into 3 non-overlapping frequency bands
    via multiplication with binary masks in the RFFT domain:

        Band 0: [ 0,   f_low) Hz
        Band 1: [f_low, f_mid) Hz
        Band 2: [f_mid, fs/2] Hz

    The masks are pre-computed as buffers and moved to the correct device with
    the model.  Because torch.fft is fully differentiable, this layer can be
    used inside a gradient-tracked forward pass.

    LOSO integrity: mask cutoffs depend only on (sampling_rate, f_low, f_mid),
    never on any data statistics.  No learnable parameters → zero leakage.
    """

    def __init__(
        self,
        window_size: int,
        sampling_rate: int = 2000,
        f_low: float = 200.0,
        f_mid: float = 500.0,
    ):
        super().__init__()
        self.window_size  = window_size
        self.sampling_rate = sampling_rate
        self.f_low = f_low
        self.f_mid = f_mid

        freqs = torch.fft.rfftfreq(window_size, d=1.0 / sampling_rate)  # (T//2+1,)

        mask_low  = (freqs < f_low).float()
        mask_mid  = ((freqs >= f_low) & (freqs < f_mid)).float()
        mask_high = (freqs >= f_mid).float()

        self.register_buffer("mask_low",  mask_low)
        self.register_buffer("mask_mid",  mask_mid)
        self.register_buffer("mask_high", mask_high)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) — input signal
        Returns:
            x_low, x_mid, x_high: each (B, C, T) — band-filtered signals
        """
        T  = x.shape[-1]
        X  = torch.fft.rfft(x, n=T, dim=-1)           # (B, C, T//2+1)

        # Broadcast masks over (B, C) dimensions
        ml = self.mask_low.view(1, 1, -1)
        mm = self.mask_mid.view(1, 1, -1)
        mh = self.mask_high.view(1, 1, -1)

        x_low  = torch.fft.irfft(X * ml, n=T, dim=-1)
        x_mid  = torch.fft.irfft(X * mm, n=T, dim=-1)
        x_high = torch.fft.irfft(X * mh, n=T, dim=-1)

        return x_low, x_mid, x_high


# ──────────────────────── Gradient Reversal Layer ────────────────────────────

class _GradReverseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Return None for the `alpha` parameter (non-tensor)
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward pass: identity transform.
    Backward pass: multiply gradient by -alpha.

    When placed before an adversarial classifier that predicts gesture class:
      - The adversarial classifier is trained to predict gestures (normal grad).
      - The GRL reverses gradients into the upstream SpecificEncoder, pushing it
        to MAXIMISE the adversarial classifier's loss — i.e., to produce
        features that are uniformly distributed over gesture classes.

    LOSO integrity: no parameters, no state.  Pure gradient manipulation.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradReverseFunc.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha


# ─────────────────────── Mini-ECAPA Aligned Encoder ──────────────────────────

class MiniECAPAEncoder(nn.Module):
    """
    Lightweight ECAPA-TDNN (2 SE-Res2Net blocks, C=64) for per-band alignment.

    Reuses SERes2NetBlock and AttentiveStatisticsPooling from ecapa_tdnn_emg.py.
    The smaller capacity keeps the total parameter budget comparable to the
    baseline (~467K for full ECAPA with 3 bands × full encoder).

    Architecture:
        Conv1d(in_ch, C, k=5) + BN + ReLU            — initial TDNN
        SERes2NetBlock(C, k=3, dilation=dilations[0])  — block 1
        SERes2NetBlock(C, k=3, dilation=dilations[1])  — block 2
        MFA: cat([b1, b2]) → Conv1d(2C, 2C, k=1) + BN + ReLU
        AttentiveStatisticsPooling: (2C, T) → (4C,)
        Linear(4C, embed_dim) + BN + ReLU + Dropout

    LOSO integrity: BN running stats from training subjects only (eval() freezes).
    """

    def __init__(
        self,
        in_channels: int,
        channels: int = 64,
        embed_dim: int = 64,
        scale: int = 4,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2, 4]
        assert len(dilations) == 2, "MiniECAPAEncoder requires exactly 2 dilation values"

        self.channels  = channels
        self.embed_dim = embed_dim
        num_blocks     = 2

        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            SERes2NetBlock(
                channels,
                kernel_size=3,
                dilation=d,
                scale=scale,
                se_reduction=se_reduction,
            )
            for d in dilations
        ])

        mfa_in = channels * num_blocks
        self.mfa = nn.Sequential(
            nn.Conv1d(mfa_in, mfa_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_in),
            nn.ReLU(inplace=True),
        )

        # ASP output = 2 * mfa_in (mean + std)
        self.asp = AttentiveStatisticsPooling(mfa_in)
        asp_out  = mfa_in * 2

        self.embedding = nn.Sequential(
            nn.Linear(asp_out, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            emb: (B, embed_dim)
        """
        out = self.init_tdnn(x)
        block_outs = []
        for blk in self.blocks:
            out = blk(out)
            block_outs.append(out)

        mfa_in  = torch.cat(block_outs, dim=1)   # (B, 2C, T)
        mfa_out = self.mfa(mfa_in)               # (B, 2C, T)
        pooled  = self.asp(mfa_out)              # (B, 4C)
        return self.embedding(pooled)            # (B, embed_dim)


# ─────────────────────────── Specific Encoder ────────────────────────────────

class SpecificEncoder(nn.Module):
    """
    Shallow CNN encoder for band-specific (modality-specific) features.

    Captures residual information unique to each frequency band.  Expected to
    encode band-specific subject variation (e.g., electrode impedance differs
    at low vs high frequencies; skin acts as a low-pass filter).

    At inference this encoder is NOT activated — only aligned encoders are used.
    During Stage-2 training, the GRL applied after this encoder ensures its
    features become uninformative about gesture class.

    Architecture:
        Conv1d(in_ch, H, k=5) + BN + ReLU
        Conv1d(H, H, k=5) + BN + ReLU
        AdaptiveAvgPool1d(1)  → (B, H, 1)
        Flatten → Linear(H, embed_dim) + BN + ReLU + Dropout

    LOSO integrity: BN running stats from training subjects only.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),   # (B, H, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            emb: (B, embed_dim)
        """
        return self.head(self.cnn(x))


# ─────────────────────────── Full Model ──────────────────────────────────────

class MultiResAlignedDisentangle(nn.Module):
    """
    Multi-Resolution Aligned Disentanglement — full model.

    See module docstring for complete description.

    Args:
        in_channels:    C — number of EMG channels (8 for NinaPro DB2)
        num_classes:    number of gesture classes
        window_size:    T — samples per window (600 with default proc_cfg)
        sampling_rate:  Hz (2000 for NinaPro DB2)
        f_low:          lower split frequency in Hz (default 200.0)
        f_mid:          mid split frequency in Hz (default 500.0)
        channels:       MiniECAPAEncoder internal width C_enc (default 64)
        embed_dim:      aligned encoder output dimension E (default 64)
        proj_dim:       projection head output for contrastive loss (default 32)
        spec_hidden:    SpecificEncoder hidden conv width (default 32)
        spec_embed_dim: SpecificEncoder output dimension (default 32)
        dropout:        dropout probability (default 0.3)
        grl_alpha:      GRL gradient-reversal scale (default 1.0)
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        window_size: int = 600,
        sampling_rate: int = 2000,
        f_low: float = 200.0,
        f_mid: float = 500.0,
        channels: int = 64,
        embed_dim: int = 64,
        proj_dim: int = 32,
        spec_hidden: int = 32,
        spec_embed_dim: int = 32,
        dropout: float = 0.3,
        grl_alpha: float = 1.0,
    ):
        super().__init__()
        self.in_channels    = in_channels
        self.num_classes    = num_classes
        self.embed_dim      = embed_dim
        self.proj_dim       = proj_dim
        self.spec_embed_dim = spec_embed_dim

        # ── 1. Band splitting (physics-based, no parameters) ─────────────
        self.band_splitter = BandSplitter(
            window_size=window_size,
            sampling_rate=sampling_rate,
            f_low=f_low,
            f_mid=f_mid,
        )

        # ── 2. Per-band Soft AGC (2*C learnable params per band) ─────────
        # SoftAGCLayer from exp_76: causal EMA-based amplitude normalization.
        # alpha ∈ (0, 0.5), delta fixed — LOSO-safe (no test-time adaptation).
        self.agc_layers = nn.ModuleList([
            SoftAGCLayer(num_channels=in_channels)
            for _ in range(3)
        ])

        # ── 3. Per-band aligned encoder (Stage 1) ────────────────────────
        self.aligned_encoders = nn.ModuleList([
            MiniECAPAEncoder(
                in_channels=in_channels,
                channels=channels,
                embed_dim=embed_dim,
                dropout=dropout,
            )
            for _ in range(3)
        ])

        # ── 4. Per-band projection head for NT-Xent contrastive loss ─────
        # L2-normalization is applied in forward(), not as a module.
        self.projections = nn.ModuleList([
            nn.Linear(embed_dim, proj_dim, bias=False)
            for _ in range(3)
        ])

        # ── 5. Gesture classifier on averaged aligned representation ──────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # ── 6. Stage-2 components: specific encoders + GRL + adv classif ─
        self.specific_encoders = nn.ModuleList([
            SpecificEncoder(
                in_channels=in_channels,
                hidden_dim=spec_hidden,
                embed_dim=spec_embed_dim,
                dropout=dropout,
            )
            for _ in range(3)
        ])

        # Single shared GRL (no parameters, alpha is just a float)
        self.grl = GradientReversalLayer(alpha=grl_alpha)

        self.adv_classifiers = nn.ModuleList([
            nn.Linear(spec_embed_dim, num_classes)
            for _ in range(3)
        ])

        self._init_output_layers()

    def _init_output_layers(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for lin in self.adv_classifiers:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    # ── helpers ───────────────────────────────────────────────────────────

    def _split_and_agc(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split x into 3 bands and apply per-band Soft AGC."""
        bands = list(self.band_splitter(x))      # 3 × (B, C, T)
        return [agc(b) for agc, b in zip(self.agc_layers, bands)]

    def _aligned_embeddings(
        self, bands: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Compute per-band aligned embeddings: 3 × (B, embed_dim)."""
        return [enc(b) for enc, b in zip(self.aligned_encoders, bands)]

    def _projections(
        self, aligned_embs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Project + L2-normalize for contrastive loss: 3 × (B, proj_dim)."""
        return [
            F.normalize(proj(e), dim=-1)
            for proj, e in zip(self.projections, aligned_embs)
        ]

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        stage: int = 1,
    ) -> dict:
        """
        Args:
            x:     (B, C, T) — channel-standardized EMG (channels-first)
            stage: 1 = alignment-only forward (no specific encoders).
                   2 = full forward including specific encoders + GRL.
                   At inference always use stage=1 (model.eval() + stage=1).

        Returns dict:
            "logits":      (B, num_classes) — gesture classifier output
            "projections": List[3 × (B, proj_dim)] — L2-normalized projections
                           for NT-Xent alignment loss (always present)
            "adv_logits":  List[3 × (B, num_classes)] — adversarial classifier
                           output (only present when stage == 2)
        """
        # 1–2. Band split + per-band AGC
        bands = self._split_and_agc(x)                    # 3 × (B, C, T)

        # 3. Per-band aligned encodings
        aligned_embs = self._aligned_embeddings(bands)    # 3 × (B, E)

        # 4. Projection heads (for contrastive loss)
        projs = self._projections(aligned_embs)           # 3 × (B, P)

        # 5. Aggregate: element-wise mean → gesture classifier
        z_aligned = torch.stack(aligned_embs, dim=0).mean(dim=0)  # (B, E)
        logits    = self.classifier(z_aligned)                     # (B, K)

        out = {"logits": logits, "projections": projs}

        if stage == 2:
            # 6. Per-band specific encodings + GRL + adversarial classifier
            adv_logits = []
            for spec_enc, adv_cls, b in zip(
                self.specific_encoders, self.adv_classifiers, bands
            ):
                # SpecificEncoder produces (B, spec_embed_dim)
                s      = spec_enc(b)
                # GRL: identity in forward, negated gradient in backward
                s_grl  = self.grl(s)
                # Adversarial classifier
                adv_logits.append(adv_cls(s_grl))
            out["adv_logits"] = adv_logits

        return out

    # ── utilities ─────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_aligned_parameters(self) -> int:
        """Parameters belonging to Stage-1 components only."""
        stage1_mods = [
            *self.agc_layers,
            *self.aligned_encoders,
            *self.projections,
            self.classifier,
        ]
        return sum(
            p.numel()
            for m in stage1_mods
            for p in m.parameters()
            if p.requires_grad
        )

    def count_specific_parameters(self) -> int:
        """Parameters belonging to Stage-2 components only."""
        stage2_mods = [
            *self.specific_encoders,
            *self.adv_classifiers,
        ]
        return sum(
            p.numel()
            for m in stage2_mods
            for p in m.parameters()
            if p.requires_grad
        )
