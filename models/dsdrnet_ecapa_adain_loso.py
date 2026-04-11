"""
DSDRNet + ECAPA-TDNN with per-scale AdaIN for subject-robust EMG gesture recognition.

Hypothesis: Cyclic Inter-Subject Reconstruction with multi-scale AdaIN.
  - Shared multi-scale encoder (ECAPA-TDNN): 3 SE-Res2Net blocks → {f¹, f², f³}
  - Per-scale AdaIN transfers subject-style statistics at each temporal resolution,
    correcting subject-specific patterns from low-level (amplitude, noise) to
    high-level (movement tempo).
  - DSDRNet-style cyclic reconstruction: encode content from A, style from B →
    reconstruct "A in B's style" → cycle back → should reconstruct A.
  - Self-supervised cycle consistency signal does not require explicit style labels.

LOSO Compliance (strict, no leakage):
  - forward(x):  CONTENT PATHWAY ONLY → returns logits.
                 No AdaIN, no decoder, no test subject statistics accessed.
  - encode(), apply_multi_scale_adain(), decode():
                 Exposed for the trainer to use DURING TRAINING on training data.
                 The trainer never passes test subject windows to these methods.
  - At inference the decoder and AdaIN modules are entirely bypassed.

Reference:
  ECAPA-TDNN: Desplanques et al., 2020 (speaker verification)
  DSDRNet:    Zhang et al. (cyclic disentangled style-content reconstruction)
  AdaIN:      Huang & Belongie, 2017 (Arbitrary Style Transfer in Real-time)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks (same topology as models/ecapa_tdnn_emg.py)
# ─────────────────────────────────────────────────────────────────────────────

class Res2NetBlock(nn.Module):
    """Hierarchical multi-scale 1-D convolution block."""

    def __init__(self, C: int, kernel_size: int = 3, dilation: int = 1, scale: int = 4):
        super().__init__()
        assert C % scale == 0, f"C={C} must be divisible by scale={scale}"
        self.scale = scale
        self.width = C // scale
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.width, self.width, kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2,
            )
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.scale, dim=1)
        outs = [chunks[0]]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = chunks[i + 1] if i == 0 else chunks[i + 1] + outs[-1]
            outs.append(F.relu(bn(conv(h))))
        return torch.cat(outs, dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, C: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(C, C // reduction),
            nn.ReLU(),
            nn.Linear(C // reduction, C),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = x.mean(dim=-1)                    # (B, C)
        scale = self.fc(gap).unsqueeze(-1)      # (B, C, 1)
        return x * scale


class SERes2NetBlock(nn.Module):
    """SE-Res2Net block — core ECAPA-TDNN building block."""

    def __init__(
        self, C: int, kernel_size: int = 3, dilation: int = 1,
        scale: int = 4, se_reduction: int = 8,
    ):
        super().__init__()
        self.pw_in  = nn.Sequential(nn.Conv1d(C, C, 1), nn.BatchNorm1d(C), nn.ReLU())
        self.res2   = Res2NetBlock(C, kernel_size, dilation, scale)
        self.pw_out = nn.Sequential(nn.Conv1d(C, C, 1), nn.BatchNorm1d(C))
        self.se     = SEBlock(C, se_reduction)
        self.relu   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.pw_in(x)
        h = self.res2(h)
        h = self.pw_out(h)
        h = self.se(h)
        return self.relu(h + residual)


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling: weighted μ ‖ weighted σ → 2C."""

    def __init__(self, C: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(C, C // 2, 1), nn.Tanh(),
            nn.Conv1d(C // 2, C, 1), nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w     = self.attn(x)                                            # (B, C, T)
        mu    = (w * x).sum(dim=-1)                                     # (B, C)
        sigma = (w * (x - mu.unsqueeze(-1)).pow(2)).sum(dim=-1)         # (B, C)
        sigma = sigma.clamp(min=1e-8).sqrt()
        return torch.cat([mu, sigma], dim=1)                            # (B, 2C)


# ─────────────────────────────────────────────────────────────────────────────
# New modules for DSDRNet-style cyclic reconstruction
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleAdaIN(nn.Module):
    """
    Instance-wise Adaptive Instance Normalisation at one temporal scale.

    Formula (Huang & Belongie, 2017):
        AdaIN(f_A, f_B) = σ(f_B) · (f_A − μ(f_A)) / σ(f_A) + μ(f_B)

    Statistics are computed per-sample along the temporal axis (T).
    This is a purely functional module — no learnable parameters.

    LOSO note: called only during training with training-subject tensors.
    """

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: (B, C, T) features from subject A
            style:   (B, C, T) features from subject B  [training subjects only]
        Returns:
            (B, C, T) — content normalised then rescaled with style statistics
        """
        mu_c    = content.mean(dim=-1, keepdim=True)                    # (B, C, 1)
        sigma_c = content.std(dim=-1, keepdim=True).clamp(min=1e-5)
        mu_s    = style.mean(dim=-1, keepdim=True)
        sigma_s = style.std(dim=-1, keepdim=True).clamp(min=1e-5)
        return sigma_s * (content - mu_c) / sigma_c + mu_s


class SignalDecoder(nn.Module):
    """
    Lightweight 1-D CNN decoder: reconstructs an EMG signal from multi-scale features.

    Input:  concatenation of 3 encoder feature maps → (B, 3·encoder_channels, T)
    Output: reconstructed EMG signal                → (B, in_channels, T)

    Properties:
      - Output layer initialised to zero → training starts from near-zero reconstruction
        (allows stable convergence before the encoder has learned useful features).
      - TRAINING ONLY — discarded at inference, never receives test-subject data.
    """

    def __init__(self, encoder_channels: int, out_channels: int, hidden: int = 256):
        super().__init__()
        in_ch = 3 * encoder_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,      hidden,      kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden,     hidden // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Conv1d(hidden // 2, out_channels, kernel_size=5, padding=2),
        )
        # Zero-init output to stabilise early training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of 3 tensors, each (B, encoder_channels, T)
        Returns:
            (B, out_channels, T) reconstructed signal
        """
        return self.net(torch.cat(features, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class DSDRNetECAPA(nn.Module):
    """
    DSDRNet + ECAPA-TDNN with multi-scale AdaIN.

    Architecture (at inference — content pathway only):
        x (B, C_emg, T)
        → init_tdnn  : Conv1d(C_emg, C, k=5)  → (B, C, T)
        → block[0]   : SERes2NetBlock(dil=d₀) → f¹ (B, C, T)
        → block[1]   : SERes2NetBlock(dil=d₁) → f² (B, C, T)
        → block[2]   : SERes2NetBlock(dil=d₂) → f³ (B, C, T)
        → mfa        : cat(f¹,f²,f³)+Conv1d   → (B, 3C, T)
        → asp        : AttentiveStatisticsPool → (B, 6C)
        → embedding  : Linear+BN+ReLU+Drop    → (B, E)
        → classifier : Linear                 → (B, num_classes)

    Additional modules used ONLY during training:
        adain_modules: one MultiScaleAdaIN per encoder block (stateless)
        decoder      : SignalDecoder for cyclic / self-reconstruction

    LOSO Compliance:
        forward(x) → CONTENT PATHWAY ONLY.
          No AdaIN, no decoder called. No test-subject statistics accessed.
        encode(), apply_multi_scale_adain(), decode():
          Used by DSDRNetECAPATrainer exclusively with training-subject windows.
          The trainer guarantees test-subject data never flows through these paths.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
        decoder_hidden: int = 256,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2, 3, 4]
        if len(dilations) != 3:
            raise ValueError(f"Exactly 3 dilations required, got {len(dilations)}")

        self.channels      = channels
        self.embedding_dim = embedding_dim

        # ── Content pathway ────────────────────────────────────────────
        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            SERes2NetBlock(
                channels, kernel_size=3, dilation=d,
                scale=scale, se_reduction=se_reduction,
            )
            for d in dilations
        ])
        self.mfa = nn.Sequential(
            nn.Conv1d(3 * channels, 3 * channels, kernel_size=1),
            nn.BatchNorm1d(3 * channels),
            nn.ReLU(),
        )
        self.asp = AttentiveStatisticsPooling(3 * channels)
        self.embedding = nn.Sequential(
            nn.Linear(6 * channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # ── Training-only modules ──────────────────────────────────────
        # Stateless: no learnable parameters, so no weight contamination
        self.adain_modules = nn.ModuleList([MultiScaleAdaIN() for _ in range(3)])
        # Learnable: used only on training-subject data, then discarded at inference
        self.decoder = SignalDecoder(channels, in_channels, hidden=decoder_hidden)

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Re-zero the decoder output layer (overrides kaiming above)
        nn.init.zeros_(self.decoder.net[-1].weight)
        nn.init.zeros_(self.decoder.net[-1].bias)

    # ── Building-block methods (trainer-facing, training data only) ────────

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale feature maps.

        LOSO note: called by trainer exclusively on TRAINING-subject windows.

        Args:
            x: (B, C_emg, T) standardised EMG windows from training subjects
        Returns:
            list of 3 tensors, each (B, channels, T)  — features {f¹, f², f³}
        """
        h = self.init_tdnn(x)
        features: List[torch.Tensor] = []
        for block in self.blocks:
            h = block(h)
            features.append(h)
        return features

    def features_to_embedding(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multi-scale features into a fixed-size embedding vector.

        Args:
            scale_features: list of 3 (B, channels, T) tensors
        Returns:
            (B, embedding_dim)
        """
        mfa_out = self.mfa(torch.cat(scale_features, dim=1))   # (B, 3C, T)
        pooled  = self.asp(mfa_out)                             # (B, 6C)
        return self.embedding(pooled)                           # (B, E)

    def apply_multi_scale_adain(
        self,
        content_features: List[torch.Tensor],
        style_features:   List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply per-scale AdaIN: transfer style statistics to content at each scale.

        Mathematical effect at scale l:
            f̃ˡ_A = σ(fˡ_B) · (fˡ_A − μ(fˡ_A)) / σ(fˡ_A) + μ(fˡ_B)

        Low scales (l=1): corrects amplitude/noise patterns.
        High scales (l=3): corrects higher-level tempo/shape patterns.

        LOSO note: both content_features and style_features must originate from
        TRAINING subjects. The trainer enforces this — test-subject data never
        flows through this path.

        Args:
            content_features: list of 3 (B, C, T) tensors from subject A
            style_features:   list of 3 (B, C, T) tensors from subject B
                              (B subjects must all be training subjects)
        Returns:
            list of 3 stylised (B, C, T) tensors
        """
        return [
            adain(cf, sf)
            for adain, cf, sf in zip(self.adain_modules, content_features, style_features)
        ]

    def decode(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct an EMG signal from multi-scale feature maps.

        LOSO note: TRAINING ONLY. Never called at inference. The trainer
        only passes training-subject features here.

        Args:
            features: list of 3 (B, channels, T) tensors
                      (may be plain encoder features or AdaIN-stylised)
        Returns:
            (B, in_channels, T) reconstructed signal
        """
        return self.decoder(features)

    # ── Standard inference forward ─────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CONTENT PATHWAY ONLY — the only path active at inference.

        Sequence: encode → features_to_embedding → classifier → logits
        AdaIN modules and decoder are NOT called here.
        No test-subject statistics are accessed.

        Args:
            x: (B, C_emg, T) standardised EMG windows
        Returns:
            logits (B, num_classes)
        """
        features = self.encode(x)
        emb      = self.features_to_embedding(features)
        return self.classifier(emb)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
