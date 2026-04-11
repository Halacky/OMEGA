"""
Frequency-Aware Contrastive Predictive Coding (Per-Band CPC).

Key idea: apply CPC independently to each UVMD frequency band.  Each band
has its own temporal encoder, autoregressive context model, and prediction
heads.  This captures band-specific temporal dynamics:
  - Low bands: slow motor-unit recruitment patterns (gesture-related)
  - High bands: fast electrode/skin artefact patterns (noise-related)

Optionally, cross-band prediction is supported: use context from band_k
to predict future features in band_j.  This captures cross-frequency
temporal dependencies.

Architecture
────────────
  Raw EMG (B, T, C)
      │
  UVMDBlock → modes (B, K, T, C)
      │
  Per-band temporal encoder:  K × StridedConv1d → (B, K, T', d_enc)
      │
  Per-band context GRU:  K × CausalGRU → (B, K, T', d_ctx)
      │
  Per-band predictors:  K × K_steps × Linear(d_ctx → d_enc)
      │
  InfoNCE loss: sum over K bands × K_steps prediction steps

LOSO safety
───────────
  ✓  CPC is causal (GRU context only from past, no future leakage).
  ✓  InfoNCE negatives are from same batch (training subjects only).
  ✓  GroupNorm in temporal encoder.
  ✓  No cross-sample statistics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.uvmd_ssl_encoder import UVMDSSLEncoder


# ═════════════════════════════════════════════════════════════════════════════
# Per-Band Temporal Encoder (strided Conv1d)
# ═════════════════════════════════════════════════════════════════════════════

class BandTemporalEncoder(nn.Module):
    """
    Strided 1D-CNN encoder for a single frequency band.

    Reduces temporal resolution while extracting features:
      Conv1d(C→64, k=7, s=2) → GN → GELU →
      Conv1d(64→128, k=5, s=2) → GN → GELU →
      Conv1d(128→d_enc, k=3, s=1) → GN → GELU

    For T=200, C=12: output T'=50 (stride factor ~4).
    """

    def __init__(self, in_channels: int = 12, d_enc: int = 128):
        super().__init__()
        self.d_enc = d_enc
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, d_enc, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, d_enc),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T) — one band, channels first

        Returns
        -------
        z : (B, T', d_enc) — temporal features, time-last
        """
        return self.net(x).permute(0, 2, 1)  # (B, d_enc, T') → (B, T', d_enc)


# ═════════════════════════════════════════════════════════════════════════════
# InfoNCE Loss (CPC-style)
# ═════════════════════════════════════════════════════════════════════════════

def info_nce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.

    For each (batch, time) prediction, the target at that (batch, time)
    is positive and all other batch elements at the same time are negatives.

    Parameters
    ----------
    predictions : (B, T', d_enc) — predicted future features
    targets : (B, T', d_enc) — actual future features

    Returns
    -------
    loss : scalar
    """
    B, T_prime, D = predictions.shape
    if B < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # Flatten time: (B*T', D)
    pred_flat = predictions.reshape(B * T_prime, D)
    tgt_flat = targets.reshape(B * T_prime, D)

    # Normalise for cosine similarity
    pred_norm = F.normalize(pred_flat, dim=-1)
    tgt_norm = F.normalize(tgt_flat, dim=-1)

    # Similarity matrix: (B*T', B*T')
    sim = torch.mm(pred_norm, tgt_norm.T) / temperature

    # Labels: diagonal = positive pairs
    labels = torch.arange(B * T_prime, device=sim.device)
    return F.cross_entropy(sim, labels)


# ═════════════════════════════════════════════════════════════════════════════
# Per-Band CPC Model
# ═════════════════════════════════════════════════════════════════════════════

class FreqAwareCPC(nn.Module):
    """
    Per-band Contrastive Predictive Coding for SSL pretraining.

    Each of K frequency bands has:
      1. Temporal encoder: Conv1d → z_k (B, T', d_enc)
      2. Context GRU: causal → c_k (B, T', d_ctx)
      3. K_steps linear predictors: c_k[t] → pred_k[t+s] for s=1..K_steps

    Loss = Σ_k Σ_s InfoNCE(pred_k[t, s], z_k[t+s])

    Optionally, cross-band prediction adds:
      c_k[t] → pred_{k→j}[t+s]  for j ≠ k
    This captures temporal cross-frequency dependencies.

    Parameters
    ----------
    encoder : UVMDSSLEncoder
        Shared backbone (only UVMD is used; CPC has its own temporal encoder).
    d_enc : int
        Temporal encoder output dimension.
    d_ctx : int
        GRU context dimension.
    k_steps : int
        Number of future prediction steps.
    in_channels : int
        EMG channels per band.
    cross_band : bool
        Enable cross-band prediction.
    overlap_lambda : float
        UVMD spectral overlap penalty weight.
    """

    def __init__(
        self,
        encoder: UVMDSSLEncoder,
        d_enc: int = 128,
        d_ctx: int = 128,
        k_steps: int = 8,
        in_channels: int = 12,
        cross_band: bool = False,
        temperature: float = 0.1,
        overlap_lambda: float = 0.01,
        overlap_sigma: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.K = encoder.K
        self.d_enc = d_enc
        self.d_ctx = d_ctx
        self.k_steps = k_steps
        self.cross_band = cross_band
        self.temperature = temperature
        self.overlap_lambda = overlap_lambda
        self.overlap_sigma = overlap_sigma

        # Per-band temporal encoders
        self.temporal_encoders = nn.ModuleList([
            BandTemporalEncoder(in_channels, d_enc)
            for _ in range(self.K)
        ])

        # Per-band causal context GRU
        self.context_grus = nn.ModuleList([
            nn.GRU(d_enc, d_ctx, batch_first=True)
            for _ in range(self.K)
        ])

        # Per-band prediction heads: K bands × k_steps predictors
        self.predictors = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d_ctx, d_enc) for _ in range(k_steps)
            ])
            for _ in range(self.K)
        ])

        # Optional cross-band predictors: band_k context → band_j future
        if cross_band:
            self.cross_predictors = nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        nn.Linear(d_ctx, d_enc) for _ in range(k_steps)
                    ])
                    for _ in range(self.K)  # target band j
                ])
                for _ in range(self.K)  # source band k
            ])

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        x : (B, T, C)

        Returns
        -------
        loss : scalar
        details : dict
        """
        # 1. UVMD decomposition
        modes = self.encoder.uvmd(x)  # (B, K, T, C)

        # 2. Per-band temporal encoding
        z_bands = []  # K × (B, T', d_enc)
        c_bands = []  # K × (B, T', d_ctx)

        for k in range(self.K):
            band_k = modes[:, k].permute(0, 2, 1)  # (B, C, T)
            z_k = self.temporal_encoders[k](band_k)  # (B, T', d_enc)
            c_k, _ = self.context_grus[k](z_k)       # (B, T', d_ctx)
            z_bands.append(z_k)
            c_bands.append(c_k)

        T_prime = z_bands[0].shape[1]

        # 3. Intra-band CPC loss
        total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        n_terms = 0
        per_band_losses = {}

        for k in range(self.K):
            band_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            for s in range(self.k_steps):
                # Predict z_k[t+s+1] from c_k[t]
                t_max = T_prime - s - 1
                if t_max <= 0:
                    continue
                pred = self.predictors[k][s](c_bands[k][:, :t_max])  # (B, t_max, d_enc)
                target = z_bands[k][:, s + 1: s + 1 + t_max]         # (B, t_max, d_enc)
                band_loss = band_loss + info_nce_loss(pred, target, self.temperature)
                n_terms += 1

            per_band_losses[f"band_{k}_cpc"] = band_loss.item()
            total_loss = total_loss + band_loss

        # 4. Optional cross-band CPC loss
        if self.cross_band:
            cross_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            for k in range(self.K):
                for j in range(self.K):
                    if k == j:
                        continue
                    for s in range(self.k_steps):
                        t_max = T_prime - s - 1
                        if t_max <= 0:
                            continue
                        pred = self.cross_predictors[k][j][s](
                            c_bands[k][:, :t_max],
                        )
                        target = z_bands[j][:, s + 1: s + 1 + t_max]
                        cross_loss = cross_loss + info_nce_loss(
                            pred, target, self.temperature,
                        )
                        n_terms += 1
            total_loss = total_loss + 0.5 * cross_loss  # lower weight for cross-band
            per_band_losses["cross_band_cpc"] = cross_loss.item()

        # Normalise by number of terms
        if n_terms > 0:
            total_loss = total_loss / n_terms

        # 5. Overlap penalty
        overlap = self.encoder.spectral_overlap_penalty(sigma=self.overlap_sigma)
        total_loss = total_loss + self.overlap_lambda * overlap

        details = {
            **per_band_losses,
            "overlap_penalty": overlap.item(),
            "total_loss": total_loss.item(),
            "n_terms": n_terms,
        }
        return total_loss, details

    def get_learned_uvmd_params(self) -> Dict:
        return self.encoder.get_learned_uvmd_params()
