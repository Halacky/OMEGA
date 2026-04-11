"""
Frequency-Selective Contrastive Learning for cross-subject sEMG.

Key insight: H4 (original) showed that adversarial content-style
disentanglement HARMS sEMG gesture recognition (-2.6 pp).  However,
the same goal (subject-invariant representations) can be achieved
through contrastive learning with frequency-selective projection.

This module provides supervised contrastive (SupCon) loss that operates
on selected frequency bands.  The hypothesis (H4_new) is that
contrasting only on LOW-frequency bands (gesture-informative, low CV)
yields better cross-subject features than contrasting on all bands.

Architecture
────────────
  Raw EMG (B, T, C)
      │
  UVMDSSLEncoder → per-band features (B, K, feat_dim)
      │
  Band selection: keep only bands in `band_indices`
      │
  Projection head: MLP → (B, proj_dim)
      │
  SupCon / SimCLR / VICReg loss on selected-band projections

Variants (tested in H4_new):
  A: all_bands     — project all K bands
  B: low_bands     — project only bands [0, 1]
  C: high_bands    — project only bands [K-2, K-1] (control)
  D: weighted      — per-band VICReg with inverse-CV weights

LOSO safety
───────────
  ✓  Contrastive pairs from training subjects only.
  ✓  Augmented views via FreqAwareAugmentation (per-sample).
  ✓  No test-subject data in loss computation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.uvmd_ssl_encoder import UVMDSSLEncoder, FreqAwareAugmentation


# ═════════════════════════════════════════════════════════════════════════════
# Supervised Contrastive Loss (SupCon)
# ═════════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Positive pairs: same gesture class.
    Negative pairs: different gesture class.

    Unlike cross-subject contrastive in multi_task_ssl.py, this does NOT
    require subject labels — only gesture labels.  The frequency-selective
    projection handles subject invariance implicitly.

    Parameters
    ----------
    temperature : float
        Softmax temperature.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        projections: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        projections : (B, D) — L2-normalised
        labels : (B,) — gesture class indices

        Returns
        -------
        loss : scalar
        """
        B = projections.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=projections.device, requires_grad=True)

        # Cosine similarity
        sim = torch.mm(projections, projections.T) / self.temperature  # (B, B)

        # Positive mask: same class
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye = torch.eye(B, device=projections.device, dtype=torch.bool)
        pos_mask = pos_mask & ~eye  # exclude self

        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=projections.device, requires_grad=True)

        # Mask self from similarity
        sim = sim.masked_fill(eye, float("-inf"))

        # Log-softmax (numerically stable via logsumexp)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Mean of log-probability over positive pairs
        # Mask -inf entries to avoid NaN propagation
        log_prob = log_prob.clamp(min=-100.0)
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
        loss = -(pos_log_prob[has_pos] / n_pos[has_pos]).mean()

        return loss


# ═════════════════════════════════════════════════════════════════════════════
# Projection Head
# ═════════════════════════════════════════════════════════════════════════════

class ContrastiveProjectionHead(nn.Module):
    """2-layer MLP projection head with L2 normalisation."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ═════════════════════════════════════════════════════════════════════════════
# FreqSelectiveContrastive — Full Pretraining Model
# ═════════════════════════════════════════════════════════════════════════════

class FreqSelectiveContrastive(nn.Module):
    """
    Frequency-selective contrastive pretraining.

    Supports two modes:
      1. Self-supervised (VICReg-like): two augmented views, no labels.
      2. Supervised contrastive (SupCon): gesture labels available.

    In both modes, only selected frequency bands are used for the
    contrastive objective.

    Parameters
    ----------
    encoder : UVMDSSLEncoder
        Shared backbone.
    band_indices : list of int
        Which bands to include for contrastive loss.
        [0, 1] = low bands, [2, 3] = high bands, [0,1,2,3] = all.
    proj_hidden : int
        Projection head hidden dimension.
    proj_dim : int
        Projection head output dimension.
    temperature : float
        Contrastive loss temperature.
    mode : str
        "supcon" for supervised contrastive, "vicreg" for self-supervised.
    overlap_lambda : float
        UVMD spectral overlap penalty weight.
    """

    def __init__(
        self,
        encoder: UVMDSSLEncoder,
        band_indices: Optional[List[int]] = None,
        proj_hidden: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.07,
        mode: str = "supcon",
        overlap_lambda: float = 0.01,
        overlap_sigma: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.band_indices = band_indices or list(range(encoder.K))
        self.mode = mode
        self.overlap_lambda = overlap_lambda
        self.overlap_sigma = overlap_sigma

        in_dim = len(self.band_indices) * encoder.feat_dim
        self.projection = ContrastiveProjectionHead(in_dim, proj_hidden, proj_dim)

        if mode == "supcon":
            self.criterion = SupConLoss(temperature=temperature)
        # VICReg mode uses FreqAwareVICReg instead — this class is for SupCon

        self.augmentation = FreqAwareAugmentation(K=encoder.K)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        x : (B, T, C)
        labels : (B,) gesture class indices (required for supcon mode)

        Returns
        -------
        loss : scalar
        details : dict
        """
        # 1. Decompose
        modes = self.encoder.uvmd(x)  # (B, K, T, C)

        # 2. Augment → two views
        view1, view2 = self.augmentation(modes)

        # 3. Encode selected bands
        feats1 = self.encoder.encode_per_band(view1)
        feats2 = self.encoder.encode_per_band(view2)

        sel1 = torch.cat([feats1[k] for k in self.band_indices], dim=1)
        sel2 = torch.cat([feats2[k] for k in self.band_indices], dim=1)

        # 4. Project
        p1 = self.projection(sel1)  # (B, proj_dim), L2-normalised
        p2 = self.projection(sel2)

        # 5. Compute loss
        if self.mode == "supcon" and labels is not None:
            # Concatenate both views for SupCon
            projections = torch.cat([p1, p2], dim=0)     # (2B, proj_dim)
            labels_dup = torch.cat([labels, labels], dim=0)  # (2B,)
            loss = self.criterion(projections, labels_dup)
        else:
            # Fallback: VICReg-style invariance loss (MSE between views)
            loss = F.mse_loss(p1, p2)

        # 6. Overlap penalty
        overlap = self.encoder.spectral_overlap_penalty(sigma=self.overlap_sigma)
        total_loss = loss + self.overlap_lambda * overlap

        details = {
            "contrastive_loss": loss.item(),
            "overlap_penalty": overlap.item(),
            "total_loss": total_loss.item(),
            "bands_used": self.band_indices,
        }
        return total_loss, details

    def get_learned_uvmd_params(self) -> Dict:
        return self.encoder.get_learned_uvmd_params()
