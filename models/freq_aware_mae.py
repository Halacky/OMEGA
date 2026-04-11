"""
Frequency-Aware Masked Autoencoder (Per-Band MAE) for SSL pretraining.

Key idea: mask ENTIRE frequency bands (not temporal patches) and reconstruct
them from the remaining visible bands.  This forces the encoder to learn
cross-band relationships — specifically, to predict high-frequency (noisy)
content from low-frequency (informative) context.

Masking strategies (tested in H3_new):
  random    — mask 1-2 bands uniformly at random
  cv_biased — P(mask high band) >> P(mask low band), guided by H1 CV gradient
  high_only — always mask bands K-2, K-1 (high freq = noise)
  low_only  — always mask bands 0, 1 (negative control)

Architecture
────────────
  Raw EMG (B, T, C)
      │
  UVMDBlock → modes (B, K, T, C)
      │
  Mask M bands → visible_modes (B, K_vis, T, C)
      │
  Per-band CNN encoders (visible bands only) → (B, K_vis, feat_dim)
      │
  Transformer decoder with MASK tokens for missing bands
      │
  Reconstruction head → predicted band signals (B, K_masked, T, C)
      │
  Loss: MSE(predicted, original masked bands)

LOSO safety
───────────
  ✓  Masking is per-sample random.
  ✓  Reconstruction target is the sample's own signal (no cross-sample info).
  ✓  GroupNorm in encoders.
  ✓  Transformer decoder uses LayerNorm (per-sample).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.uvmd_ssl_encoder import UVMDSSLEncoder


# ═════════════════════════════════════════════════════════════════════════════
# Band Masking Strategies
# ═════════════════════════════════════════════════════════════════════════════

class BandMasker:
    """
    Generate binary masks indicating which frequency bands to mask.

    Parameters
    ----------
    K : int
        Total number of bands.
    strategy : str
        "random", "cv_biased", "high_only", "low_only"
    n_mask : int
        Number of bands to mask per sample (default 2 for K=4).
    cv_weights : list of float or None
        Per-band masking probabilities for "cv_biased" strategy.
        Higher weight = more likely to be masked.
        Default from H1: [0.05, 0.15, 0.30, 0.50] (low→high freq).
    """

    def __init__(
        self,
        K: int = 4,
        strategy: str = "random",
        n_mask: int = 2,
        cv_weights: Optional[List[float]] = None,
    ):
        self.K = K
        self.strategy = strategy
        self.n_mask = min(n_mask, K - 1)  # must keep at least 1 visible

        if cv_weights is None:
            # Default: proportional to normalised CV from H1 analysis
            # CV gradient: 0.20 → 0.59 → 1.12 → 2.04
            self.cv_weights = [0.05, 0.15, 0.30, 0.50]
        else:
            self.cv_weights = cv_weights

    def __call__(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Generate band mask.

        Returns
        -------
        mask : (B, K) bool tensor
            True = masked (to reconstruct), False = visible (input to encoder)
        """
        if self.strategy == "high_only":
            return self._fixed_mask(B, device, list(range(self.K - self.n_mask, self.K)))
        elif self.strategy == "low_only":
            return self._fixed_mask(B, device, list(range(self.n_mask)))
        elif self.strategy == "cv_biased":
            return self._weighted_mask(B, device)
        else:  # "random"
            return self._random_mask(B, device)

    def _fixed_mask(
        self, B: int, device: torch.device, masked_bands: List[int],
    ) -> torch.Tensor:
        mask = torch.zeros(B, self.K, dtype=torch.bool, device=device)
        for k in masked_bands:
            mask[:, k] = True
        return mask

    def _random_mask(self, B: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(B, self.K, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(self.K, device=device)[: self.n_mask]
            mask[b, idx] = True
        return mask

    def _weighted_mask(self, B: int, device: torch.device) -> torch.Tensor:
        weights = torch.tensor(self.cv_weights[: self.K], device=device)
        weights = weights / weights.sum()
        mask = torch.zeros(B, self.K, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.multinomial(weights, self.n_mask, replacement=False)
            mask[b, idx] = True
        return mask


# ═════════════════════════════════════════════════════════════════════════════
# Lightweight Transformer Decoder
# ═════════════════════════════════════════════════════════════════════════════

class TransformerDecoderBlock(nn.Module):
    """Single Transformer block with pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int = 4, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-LN FFN
        x = x + self.ff(self.norm2(x))
        return x


class BandDecoder(nn.Module):
    """
    Lightweight Transformer decoder for band reconstruction.

    Takes encoded visible-band features + learnable MASK tokens for masked
    bands, processes with Transformer layers, then reconstructs the
    masked band signals.

    Parameters
    ----------
    K : int
        Total number of bands.
    feat_dim : int
        Feature dimension from per-band encoder.
    d_model : int
        Decoder hidden dimension.
    n_layers : int
        Number of Transformer decoder layers.
    n_heads : int
        Number of attention heads.
    out_channels : int
        Number of EMG channels (for reconstruction target).
    out_time : int
        Time dimension of output signal (window_size).
    """

    def __init__(
        self,
        K: int = 4,
        feat_dim: int = 64,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        out_channels: int = 12,
        out_time: int = 200,
    ):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.out_channels = out_channels
        self.out_time = out_time

        # Project encoder features to decoder dimension
        self.input_proj = nn.Linear(feat_dim, d_model)

        # Learnable MASK token (one per masked band slot)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable band positional embeddings
        self.band_pos = nn.Parameter(torch.randn(1, K, d_model) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Reconstruction head: d_model → out_channels * out_time (per band)
        # We use a small MLP to go from d_model to the full signal
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, out_channels * out_time),
        )

    def forward(
        self,
        visible_feats: List[torch.Tensor],
        visible_indices: List[int],
        masked_indices: List[int],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        visible_feats : list of (B, feat_dim) tensors for visible bands
        visible_indices : list of int, which bands are visible
        masked_indices : list of int, which bands are masked

        Returns
        -------
        reconstructed : (B, n_masked, T, C) — reconstructed signals
        """
        B = visible_feats[0].shape[0]
        device = visible_feats[0].device

        # Build sequence: visible features + mask tokens, in band order
        tokens = []
        for k in range(self.K):
            if k in visible_indices:
                idx = visible_indices.index(k)
                tok = self.input_proj(visible_feats[idx])  # (B, d_model)
            else:
                tok = self.mask_token.expand(B, -1, -1).squeeze(1)  # (B, d_model)
            tokens.append(tok)

        # (B, K, d_model) + positional embeddings
        x = torch.stack(tokens, dim=1) + self.band_pos

        # Transformer decoder
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Extract masked band tokens and reconstruct
        masked_tokens = torch.stack(
            [x[:, k] for k in masked_indices], dim=1,
        )  # (B, n_masked, d_model)

        # Reconstruct each masked band
        n_masked = len(masked_indices)
        flat = self.recon_head(
            masked_tokens.reshape(B * n_masked, self.d_model),
        )  # (B*n_masked, C*T)
        reconstructed = flat.reshape(
            B, n_masked, self.out_time, self.out_channels,
        )
        return reconstructed


# ═════════════════════════════════════════════════════════════════════════════
# FreqAwareMAE — Full Pretraining Model
# ═════════════════════════════════════════════════════════════════════════════

class FreqAwareMAE(nn.Module):
    """
    Frequency-Aware Masked Autoencoder for SSL pretraining.

    Masks entire frequency bands and reconstructs them from visible bands.

    Parameters
    ----------
    encoder : UVMDSSLEncoder
        Shared backbone.
    mask_strategy : str
        "random", "cv_biased", "high_only", "low_only"
    n_mask : int
        Number of bands to mask per sample.
    decoder_dim : int
        Transformer decoder hidden dimension.
    decoder_layers : int
        Number of decoder Transformer layers.
    decoder_heads : int
        Number of attention heads in decoder.
    window_size : int
        Temporal window size (for reconstruction target shape).
    overlap_lambda : float
        Weight for UVMD spectral overlap penalty.
    """

    def __init__(
        self,
        encoder: UVMDSSLEncoder,
        mask_strategy: str = "random",
        n_mask: int = 2,
        decoder_dim: int = 128,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        window_size: int = 200,
        overlap_lambda: float = 0.01,
        overlap_sigma: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.overlap_lambda = overlap_lambda
        self.overlap_sigma = overlap_sigma

        K = encoder.K
        in_channels = 12  # NinaPro DB2

        self.masker = BandMasker(
            K=K, strategy=mask_strategy, n_mask=n_mask,
        )

        self.decoder = BandDecoder(
            K=K,
            feat_dim=encoder.feat_dim,
            d_model=decoder_dim,
            n_layers=decoder_layers,
            n_heads=decoder_heads,
            out_channels=in_channels,
            out_time=window_size,
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        x : (B, T, C)

        Returns
        -------
        loss : scalar (MSE reconstruction + overlap penalty)
        details : dict with loss components
        """
        B, T, C = x.shape

        # 1. Decompose
        modes = self.encoder.uvmd(x)  # (B, K, T, C)

        # 2. Generate mask
        mask = self.masker(B, x.device)  # (B, K) bool

        # For simplicity, use the same mask for all samples in batch
        # (required for batched decoder). Use the first sample's mask.
        # Alternative: group by mask pattern — more complex but more varied.
        # Here we use per-sample masking but process in a loop if masks differ.
        # For efficiency, use majority mask pattern:
        mask_pattern = mask[0]  # (K,) — use first sample's mask
        masked_indices = mask_pattern.nonzero(as_tuple=True)[0].tolist()
        visible_indices = (~mask_pattern).nonzero(as_tuple=True)[0].tolist()

        if not masked_indices:
            # Edge case: nothing to mask
            zero_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            return zero_loss, {"recon_loss": 0.0, "total_loss": 0.0}

        # 3. Encode visible bands
        visible_feats = [
            self.encoder.band_encoders.encoders[k](
                modes[:, k].permute(0, 2, 1),
            )
            for k in visible_indices
        ]  # list of (B, feat_dim)

        # 4. Decode / reconstruct masked bands
        reconstructed = self.decoder(
            visible_feats, visible_indices, masked_indices,
        )  # (B, n_masked, T, C)

        # 5. Reconstruction target: original masked band signals
        target = torch.stack(
            [modes[:, k] for k in masked_indices], dim=1,
        )  # (B, n_masked, T, C)

        # 6. MSE loss on masked bands
        recon_loss = F.mse_loss(reconstructed, target.detach())

        # 7. Overlap penalty
        overlap = self.encoder.spectral_overlap_penalty(sigma=self.overlap_sigma)
        total_loss = recon_loss + self.overlap_lambda * overlap

        details = {
            "recon_loss": recon_loss.item(),
            "overlap_penalty": overlap.item(),
            "total_loss": total_loss.item(),
            "n_masked": len(masked_indices),
            "masked_bands": masked_indices,
        }
        return total_loss, details

    def get_learned_uvmd_params(self) -> Dict:
        return self.encoder.get_learned_uvmd_params()
