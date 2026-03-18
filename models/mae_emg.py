"""
MAE-EMG: Masked Autoencoder for EMG signals.

Hypothesis H9: Self-supervised reconstruction pretraining улучшит representation.

Approach (MAE-style):
  1. Pretraining — mask 40% of temporal patches → Transformer encoder over visible
     patches → lightweight Transformer decoder → reconstruct masked patches (MSE).
  2. Fine-tuning — load pretrained encoder weights → attach linear classifier head
     → supervised training on gesture labels.

Architecture overview:
  Input (B, C, T)  [PyTorch convention, C=8 channels, T=600 time steps]
  ↓ PatchEmbed: split T into num_patches patches, flatten C×patch_size → d_model
  ↓ Add learnable positional embeddings
  ↓ [Pretraining]  mask 40% patches, prepend MASK tokens in decoder
  ↓ Transformer Encoder (visible patches only → encoded tokens)
  ↓ Transformer Decoder (encoded + MASK tokens → reconstruct all patches)
  ↓ MSE loss on masked patches only

  [Fine-tuning]
  ↓ Transformer Encoder (all patches) → global average pool → Linear classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Split temporal signal into non-overlapping patches and project each patch
    to d_model dimensions.

    Input:  (B, C, T)
    Output: (B, num_patches, d_model)
    """

    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        # Each patch has in_channels * patch_size values
        self.proj = nn.Linear(in_channels * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert T % self.patch_size == 0, (
            f"Time dimension T={T} must be divisible by patch_size={self.patch_size}"
        )
        num_patches = T // self.patch_size
        # (B, C, num_patches, patch_size) → (B, num_patches, C * patch_size)
        x = x.reshape(B, C, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, num_patches, C * self.patch_size)
        return self.proj(x)  # (B, num_patches, d_model)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block: MHA + FFN."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MAEEncoder(nn.Module):
    """
    Transformer encoder.  During pretraining it receives only visible (unmasked)
    patch tokens.  During fine-tuning it receives all patch tokens.

    Args:
        d_model:      embedding dimension
        depth:        number of Transformer blocks
        n_heads:      attention heads
        dropout:      dropout probability
    """

    def __init__(self, d_model: int, depth: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model) → (B, seq_len, d_model)"""
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full MAE model (pretraining)
# ---------------------------------------------------------------------------

class MAEEmgForPretraining(nn.Module):
    """
    Full Masked Autoencoder for EMG pretraining.

    Input:  (B, C, T)
    Output: (reconstruction_loss, pred_patches, mask)
      - reconstruction_loss: scalar MSE loss on masked patches
      - pred_patches: (B, num_patches, patch_size*C) reconstructed signal
      - mask: (B, num_patches) bool tensor — True = masked
    """

    def __init__(
        self,
        in_channels: int = 8,
        time_steps: int = 600,
        patch_size: int = 20,
        d_model: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        decoder_d_model: int = 64,
        mask_ratio: float = 0.4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.num_patches = time_steps // patch_size
        self.patch_dim = in_channels * patch_size  # raw patch dimension

        # --- Encoder ---
        self.patch_embed = PatchEmbed(in_channels, patch_size, d_model)
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        self.encoder = MAEEncoder(d_model, encoder_depth, encoder_heads, dropout)

        # --- Decoder ---
        # Project encoder tokens to decoder dimension
        self.enc_to_dec = nn.Linear(d_model, decoder_d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_d_model)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder = MAEEncoder(decoder_d_model, decoder_depth, decoder_heads, dropout)
        # Reconstruction head: decoder_d_model → patch_dim
        self.reconstruction_head = nn.Linear(decoder_d_model, self.patch_dim)

    def _random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking: keep (1-mask_ratio) patches visible.

        Args:
            x: (B, L, d_model) patch embeddings
            mask_ratio: fraction of patches to mask

        Returns:
            x_visible: (B, L_visible, d_model)
            ids_restore: (B, L) permutation to restore original order
            mask: (B, L) bool — True = masked
        """
        B, L, D = x.shape
        num_mask = int(L * mask_ratio)
        num_keep = L - num_mask

        # Random shuffle per sample
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)          # ascending → keep lowest
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]               # first num_keep are visible
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Build binary mask: 1 = masked, 0 = kept
        mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return x_visible, ids_restore, mask

    def forward(
        self, x: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, C, T = x.shape

        # --- Patchify input for target reconstruction ---
        # target shape: (B, num_patches, patch_dim)
        num_patches = T // self.patch_size
        target = x.reshape(B, C, num_patches, self.patch_size)
        target = target.permute(0, 2, 1, 3).reshape(B, num_patches, self.patch_dim)

        # --- Encoder ---
        tokens = self.patch_embed(x)                           # (B, L, d_model)
        tokens = tokens + self.encoder_pos_embed[:, :num_patches, :]

        tokens_vis, ids_restore, mask = self._random_masking(tokens, mask_ratio)
        encoded = self.encoder(tokens_vis)                     # (B, L_vis, d_model)

        # --- Decoder ---
        dec_tokens = self.enc_to_dec(encoded)                  # (B, L_vis, dec_d)

        # Expand mask tokens to fill masked positions
        num_mask = num_patches - dec_tokens.shape[1]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)

        # Concatenate [encoded visible | mask tokens] then restore original order
        full_tokens = torch.cat([dec_tokens, mask_tokens], dim=1)  # (B, L, dec_d)
        full_tokens = torch.gather(
            full_tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1])
        )
        full_tokens = full_tokens + self.decoder_pos_embed[:, :num_patches, :]
        decoded = self.decoder(full_tokens)                    # (B, L, dec_d)
        pred = self.reconstruction_head(decoded)               # (B, L, patch_dim)

        # --- MSE loss on masked patches only ---
        loss = F.mse_loss(pred[mask], target[mask])

        return loss, pred, mask

    def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode all patches (no masking).  Used during fine-tuning feature
        extraction.  Returns (B, num_patches, d_model).
        """
        B, C, T = x.shape
        num_patches = T // self.patch_size
        tokens = self.patch_embed(x) + self.encoder_pos_embed[:, :num_patches, :]
        return self.encoder(tokens)


# ---------------------------------------------------------------------------
# Fine-tuning classifier
# ---------------------------------------------------------------------------

class MAEEmgForClassification(nn.Module):
    """
    Fine-tuning model for gesture classification.

    The encoder weights can be initialised from a pretrained
    MAEEmgForPretraining instance via `load_pretrained_encoder()`.

    Input:  (B, C, T)  — same as rest of the codebase
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        time_steps: int = 600,
        patch_size: int = 20,
        d_model: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        dropout: float = 0.1,
        **kwargs,  # absorb extra kwargs from trainer factory
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model
        num_patches = time_steps // patch_size

        self.patch_embed = PatchEmbed(in_channels, patch_size, d_model)
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, d_model)
        )
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        self.encoder = MAEEncoder(d_model, encoder_depth, encoder_heads, dropout)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def load_pretrained_encoder(self, pretrain_model: MAEEmgForPretraining) -> None:
        """Copy patch embedding + positional embeddings + encoder weights."""
        self.patch_embed.load_state_dict(pretrain_model.patch_embed.state_dict())
        with torch.no_grad():
            L = self.encoder_pos_embed.shape[1]
            self.encoder_pos_embed.copy_(
                pretrain_model.encoder_pos_embed[:, :L, :]
            )
        self.encoder.load_state_dict(pretrain_model.encoder.state_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        num_patches = T // self.patch_size

        tokens = self.patch_embed(x)                                # (B, L, d_model)
        tokens = tokens + self.encoder_pos_embed[:, :num_patches, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                     # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)                    # (B, L+1, d_model)

        encoded = self.encoder(tokens)                              # (B, L+1, d_model)
        cls_out = encoded[:, 0, :]                                  # (B, d_model)
        return self.classifier(cls_out)                             # (B, num_classes)
