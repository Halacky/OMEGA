"""
Multi-Task Self-Supervised Pretraining model for EMG gesture classification.

Hypothesis: Pretrain on 3 self-supervised tasks simultaneously:
  1. MAE reconstruction  — mask 40% temporal patches, reconstruct via decoder
  2. Subject prediction   — auxiliary head + distance-correlation decorrelation
  3. Cross-subject contrastive — pull same-gesture/different-subject, push different gestures

The encoder learns gesture-invariant features separated from subject-specific style,
improving LOSO cross-subject generalization.

Architecture:
  Pretraining:
    Input (B, C, T) → PatchEmbed → positional embeddings
    Pass 1 (masked):  mask 40% → Encoder(visible) → Decoder → reconstruct → MSE
    Pass 2 (full):    Encoder(all) → avg pool → (B, d_model)
      ├─ Subject head → CE(subject_id)
      ├─ Projection head → cross-subject contrastive loss
      └─ Distance correlation(gesture_repr, subject_repr) → minimize

  Fine-tuning:
    (B, C, T) → PatchEmbed → Encoder(pretrained) → CLS token → Linear → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

from models.mae_emg import PatchEmbed, MAEEncoder


# ---------------------------------------------------------------------------
# Cross-Subject Contrastive Loss
# ---------------------------------------------------------------------------

class CrossSubjectContrastiveLoss(nn.Module):
    """
    InfoNCE variant for cross-subject gesture alignment.

    Positive pairs: same gesture_id, different subject_id.
    Negative pairs: different gesture_id (regardless of subject).
    Falls back gracefully when no valid positive pairs exist in a batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        projections: torch.Tensor,   # (B, D), L2-normalized
        gesture_ids: torch.Tensor,    # (B,) int
        subject_ids: torch.Tensor,    # (B,) int
    ) -> torch.Tensor:
        B = projections.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=projections.device, requires_grad=True)

        # Cosine similarity matrix (already L2-normalized)
        sim = torch.mm(projections, projections.t()) / self.temperature  # (B, B)

        # Positive mask: same gesture AND different subject
        gesture_match = gesture_ids.unsqueeze(0) == gesture_ids.unsqueeze(1)   # (B, B)
        subject_diff = subject_ids.unsqueeze(0) != subject_ids.unsqueeze(1)    # (B, B)
        pos_mask = gesture_match & subject_diff                                 # (B, B)

        # Exclude self-similarity
        eye = torch.eye(B, device=projections.device, dtype=torch.bool)
        pos_mask = pos_mask & ~eye

        # Check if any sample has at least one positive
        has_pos = pos_mask.any(dim=1)  # (B,)
        if not has_pos.any():
            return torch.tensor(0.0, device=projections.device, requires_grad=True)

        # Mask self from similarity
        sim = sim.masked_fill(eye, float("-inf"))

        # For each anchor with positives, compute loss
        # log(sum_pos(exp(sim))) - log(sum_all(exp(sim)))  → equivalent to InfoNCE
        exp_sim = torch.exp(sim)  # (B, B)
        pos_exp = (exp_sim * pos_mask.float()).sum(dim=1)   # (B,)
        all_exp = exp_sim.sum(dim=1)                         # (B,)

        # Only compute for anchors that have valid positives
        loss = -torch.log(pos_exp[has_pos] / (all_exp[has_pos] + 1e-8))
        return loss.mean()


# ---------------------------------------------------------------------------
# Multi-Task SSL Pretraining Model
# ---------------------------------------------------------------------------

class MultiTaskSSLForPretraining(nn.Module):
    """
    Multi-task self-supervised pretraining model.

    Three simultaneous tasks from a shared Transformer encoder:
      1. MAE patch reconstruction (masked patches only)
      2. Subject ID prediction (auxiliary)
      3. Cross-subject gesture contrastive alignment

    Input:  (B, C, T) — PyTorch convention
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
        num_subjects: int = 5,
        projection_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.num_patches = time_steps // patch_size
        self.patch_dim = in_channels * patch_size

        # ── Shared encoder ──────────────────────────────────────────────
        self.patch_embed = PatchEmbed(in_channels, patch_size, d_model)
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        self.encoder = MAEEncoder(d_model, encoder_depth, encoder_heads, dropout)

        # ── Task 1: MAE decoder ─────────────────────────────────────────
        self.enc_to_dec = nn.Linear(d_model, decoder_d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_d_model)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder = MAEEncoder(decoder_d_model, decoder_depth, decoder_heads, dropout)
        self.reconstruction_head = nn.Linear(decoder_d_model, self.patch_dim)

        # ── Task 2: Subject prediction head ─────────────────────────────
        self.subject_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_subjects),
        )

        # ── Task 3: Contrastive projection head ────────────────────────
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, projection_dim),
        )

    # ── Masking utility ─────────────────────────────────────────────────

    def _random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking of patch tokens.

        Returns:
            x_visible: (B, L_vis, D)
            ids_restore: (B, L) permutation for original order
            mask: (B, L) bool — True = masked
        """
        B, L, D = x.shape
        num_mask = int(L * mask_ratio)
        num_keep = L - num_mask

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, L, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return x_visible, ids_restore, mask

    # ── Forward: pretraining ────────────────────────────────────────────

    def forward_pretrain(
        self,
        x: torch.Tensor,             # (B, C, T)
        subject_ids: torch.Tensor,    # (B,)
        gesture_ids: torch.Tensor,    # (B,)
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task pretraining forward pass.

        Returns dict with keys:
            "mae_loss", "subject_logits", "projections",
            "gesture_repr", "subject_repr"
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, C, T = x.shape
        num_patches = T // self.patch_size

        # ── Patchify target for MAE ──
        target = x.reshape(B, C, num_patches, self.patch_size)
        target = target.permute(0, 2, 1, 3).reshape(B, num_patches, self.patch_dim)

        # ── Embed patches + positional encoding ──
        tokens = self.patch_embed(x)  # (B, L, d_model)
        tokens = tokens + self.encoder_pos_embed[:, :num_patches, :]

        # ━━━ Pass 1: Masked encoding → MAE reconstruction ━━━
        tokens_vis, ids_restore, mask = self._random_masking(tokens, mask_ratio)
        encoded_vis = self.encoder(tokens_vis)  # (B, L_vis, d_model)

        # Decode
        dec_tokens = self.enc_to_dec(encoded_vis)
        num_mask = num_patches - dec_tokens.shape[1]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        full_tokens = torch.cat([dec_tokens, mask_tokens], dim=1)
        full_tokens = torch.gather(
            full_tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1])
        )
        full_tokens = full_tokens + self.decoder_pos_embed[:, :num_patches, :]
        decoded = self.decoder(full_tokens)
        pred = self.reconstruction_head(decoded)  # (B, L, patch_dim)

        mae_loss = F.mse_loss(pred[mask], target[mask])

        # ━━━ Pass 2: Full encoding → subject + contrastive ━━━
        encoded_full = self.encoder(
            self.patch_embed(x) + self.encoder_pos_embed[:, :num_patches, :]
        )  # (B, L, d_model)
        global_repr = encoded_full.mean(dim=1)  # (B, d_model)

        # Task 2: Subject prediction
        subject_logits = self.subject_head(global_repr)  # (B, num_subjects)

        # Task 3: Contrastive projection
        projections = self.projection_head(global_repr)   # (B, projection_dim)
        projections = F.normalize(projections, dim=1)

        return {
            "mae_loss": mae_loss,
            "subject_logits": subject_logits,
            "projections": projections,
            "global_repr": global_repr,
            "pred_patches": pred,
            "mask": mask,
            "target_patches": target,
        }

    # ── Forward: full encoding (no masking) ─────────────────────────────

    def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all patches (no masking). Returns (B, num_patches, d_model)."""
        B, C, T = x.shape
        num_patches = T // self.patch_size
        tokens = self.patch_embed(x) + self.encoder_pos_embed[:, :num_patches, :]
        return self.encoder(tokens)

    def get_global_repr(self, x: torch.Tensor) -> torch.Tensor:
        """Global average pooled encoder representation. Returns (B, d_model)."""
        return self.get_encoder_output(x).mean(dim=1)


# ---------------------------------------------------------------------------
# Fine-tuning classifier
# ---------------------------------------------------------------------------

class MultiTaskSSLForClassification(nn.Module):
    """
    Fine-tuning model: pretrained encoder + CLS token + linear classifier.

    Input:  (B, C, T)
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
        **kwargs,
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

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def load_pretrained_encoder(
        self, pretrain_model: MultiTaskSSLForPretraining
    ) -> None:
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

        tokens = self.patch_embed(x)
        tokens = tokens + self.encoder_pos_embed[:, :num_patches, :]

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]
        return self.classifier(cls_out)
