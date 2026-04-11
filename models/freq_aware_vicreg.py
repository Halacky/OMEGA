"""
Frequency-Aware VICReg for self-supervised EMG representation learning.

VICReg (Variance-Invariance-Covariance Regularization) is chosen over
SimCLR/BYOL because:
  1. No negative samples needed (unlike SimCLR) — works with small batches.
  2. No momentum encoder (unlike BYOL) — simpler, fewer hyperparameters.
  3. Interpretable loss components: variance (prevent collapse), invariance
     (pull augmented views), covariance (decorrelate dimensions).

Integration with OMEGA's frequency decomposition
─────────────────────────────────────────────────
  Raw EMG → UVMD → FreqAwareAugmentation (two views) → PerBandEncoder →
  Projector → VICReg loss

The key novelty is FreqAwareAugmentation: augmentation intensity is
proportional to the inter-subject CV of each frequency band (H1 insight).
Low-frequency bands (gesture info) get light augmentation; high-frequency
bands (subject noise) get aggressive augmentation.

LOSO safety
───────────
  ✓  No target/test subject data used in pretraining.
  ✓  Augmentations are per-sample, no cross-sample statistics.
  ✓  GroupNorm in encoder — no running statistics.
  ✓  VICReg loss computed on batch statistics of training subjects only.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.uvmd_ssl_encoder import UVMDSSLEncoder, FreqAwareAugmentation


# ═════════════════════════════════════════════════════════════════════════════
# VICReg Loss
# ═════════════════════════════════════════════════════════════════════════════

def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Hinge loss on per-dimension standard deviation.

    Prevents representation collapse by requiring each dimension to have
    std >= gamma.

    Parameters
    ----------
    z : (B, D)
    gamma : float
        Target minimum std per dimension (default 1.0).

    Returns
    -------
    loss : scalar
    """
    std_z = torch.sqrt(z.var(dim=0) + 1e-4)  # (D,)
    return F.relu(gamma - std_z).mean()


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Off-diagonal covariance regularisation.

    Decorrelates dimensions by penalising off-diagonal entries of the
    covariance matrix.  Prevents information collapse where many
    dimensions encode the same information.

    Parameters
    ----------
    z : (B, D)

    Returns
    -------
    loss : scalar
    """
    B, D = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(B - 1, 1)  # (D, D)
    # Off-diagonal elements
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


def vicreg_invariance_loss(
    z1: torch.Tensor, z2: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared distance between representations of two views.

    Parameters
    ----------
    z1, z2 : (B, D)

    Returns
    -------
    loss : scalar
    """
    return F.mse_loss(z1, z2)


class VICRegLoss(nn.Module):
    """
    Combined VICReg loss: λ_inv * invariance + λ_var * variance + λ_cov * covariance.

    Default weights from Bardes et al. (2022): λ_inv=25, λ_var=25, λ_cov=1.

    Parameters
    ----------
    lambda_inv : float
        Weight for invariance (MSE between views).
    lambda_var : float
        Weight for variance (hinge loss on std).
    lambda_cov : float
        Weight for covariance (off-diagonal penalty).
    gamma : float
        Target minimum std for variance loss.
    """

    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        z1, z2 : (B, D) — projected representations of two augmented views.

        Returns
        -------
        total_loss : scalar tensor
        details : dict with individual loss components (for logging)
        """
        inv_loss = vicreg_invariance_loss(z1, z2)
        var_loss = vicreg_variance_loss(z1, self.gamma) + vicreg_variance_loss(z2, self.gamma)
        cov_loss = vicreg_covariance_loss(z1) + vicreg_covariance_loss(z2)

        total = (
            self.lambda_inv * inv_loss
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
        )

        details = {
            "inv_loss": inv_loss.item(),
            "var_loss": var_loss.item(),
            "cov_loss": cov_loss.item(),
            "total_loss": total.item(),
        }
        return total, details


# ═════════════════════════════════════════════════════════════════════════════
# Projector / Expander MLP
# ═════════════════════════════════════════════════════════════════════════════

class ProjectorMLP(nn.Module):
    """
    3-layer MLP projector for VICReg.

    Maps encoder features to a high-dimensional space where VICReg loss
    is computed.  The projector is discarded after pretraining.

    Architecture: Linear → BN → ReLU → Linear → BN → ReLU → Linear

    Parameters
    ----------
    in_dim : int
        Input dimension (encoder output: K * feat_dim).
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output dimension (projection space).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
# FreqAwareVICReg — Full Pretraining Model
# ═════════════════════════════════════════════════════════════════════════════

class FreqAwareVICReg(nn.Module):
    """
    Frequency-aware VICReg pretraining model.

    Pipeline:
      x (B, T, C)
      → UVMDSSLEncoder.decompose() → modes (B, K, T, C)
      → FreqAwareAugmentation → view1, view2 (B, K, T, C) each
      → UVMDSSLEncoder.encode_per_band(view_i) → features (B, K*feat_dim)
      → ProjectorMLP → projections (B, proj_dim)
      → VICRegLoss(proj1, proj2)

    The UVMD decomposition is shared between both views (decompose once,
    augment the modes).  This means the decomposition learns from the
    reconstruction signal, and the encoder learns augmentation-invariant
    features in the decomposed space.

    Parameters
    ----------
    encoder : UVMDSSLEncoder
        Shared backbone (UVMD + per-band CNN).
    proj_hidden : int
        Hidden dimension of projector MLP.
    proj_dim : int
        Output dimension of projector.
    aug_kwargs : dict
        Keyword arguments for FreqAwareAugmentation.
    vicreg_kwargs : dict
        Keyword arguments for VICRegLoss (lambda weights).
    overlap_lambda : float
        Weight for UVMD spectral overlap penalty.
    overlap_sigma : float
        Width for spectral overlap penalty.
    contrastive_bands : list of int or None
        If set, VICReg is computed only on features from these bands.
        None = use all bands.  [0, 1] = low-freq only (H4_new variant B).
    """

    def __init__(
        self,
        encoder: UVMDSSLEncoder,
        proj_hidden: int = 2048,
        proj_dim: int = 2048,
        aug_kwargs: Optional[Dict] = None,
        vicreg_kwargs: Optional[Dict] = None,
        overlap_lambda: float = 0.01,
        overlap_sigma: float = 0.05,
        contrastive_bands: Optional[list] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.contrastive_bands = contrastive_bands

        # Determine projector input dimension
        if contrastive_bands is not None:
            proj_in = len(contrastive_bands) * encoder.feat_dim
        else:
            proj_in = encoder.total_feat_dim

        self.projector = ProjectorMLP(proj_in, proj_hidden, proj_dim)
        self.augmentation = FreqAwareAugmentation(
            K=encoder.K, **(aug_kwargs or {}),
        )
        self.vicreg_loss = VICRegLoss(**(vicreg_kwargs or {}))
        self.overlap_lambda = overlap_lambda
        self.overlap_sigma = overlap_sigma

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VICReg loss for a batch of raw EMG windows.

        Parameters
        ----------
        x : (B, T, C)

        Returns
        -------
        loss : scalar tensor (VICReg + overlap penalty)
        details : dict with per-component losses for logging
        """
        # 1. Decompose (shared, no augmentation yet)
        modes = self.encoder.uvmd(x)  # (B, K, T, C)

        # 2. Generate two augmented views of the modes
        view1, view2 = self.augmentation(modes)

        # 3. Encode both views through per-band CNN
        feats1 = self.encoder.encode_per_band(view1)  # list of K × (B, feat_dim)
        feats2 = self.encoder.encode_per_band(view2)

        # 4. Select bands (all or subset)
        if self.contrastive_bands is not None:
            feats1 = [feats1[k] for k in self.contrastive_bands]
            feats2 = [feats2[k] for k in self.contrastive_bands]

        z1 = torch.cat(feats1, dim=1)  # (B, selected_K * feat_dim)
        z2 = torch.cat(feats2, dim=1)

        # 5. Project to VICReg space
        p1 = self.projector(z1)  # (B, proj_dim)
        p2 = self.projector(z2)

        # 6. VICReg loss
        vicreg_total, details = self.vicreg_loss(p1, p2)

        # 7. Spectral overlap penalty
        overlap = self.encoder.spectral_overlap_penalty(sigma=self.overlap_sigma)
        total_loss = vicreg_total + self.overlap_lambda * overlap
        details["overlap_penalty"] = overlap.item()

        return total_loss, details

    def get_learned_uvmd_params(self) -> Dict:
        return self.encoder.get_learned_uvmd_params()


# ═════════════════════════════════════════════════════════════════════════════
# Per-Band VICReg (separate VICReg per frequency band)
# ═════════════════════════════════════════════════════════════════════════════

class PerBandVICReg(nn.Module):
    """
    Variant: separate VICReg loss per frequency band with band-specific weights.

    Instead of concatenating all bands and computing one VICReg loss, this
    computes K independent VICReg losses with weights proportional to
    1/CV_k (inverse of inter-subject variability).

    Bands with low inter-subject variability (low CV → high weight) are
    trained more strongly for invariance, while noisy high-CV bands get
    lower weight.

    Parameters
    ----------
    encoder : UVMDSSLEncoder
    band_weights : list of float or None
        Per-band loss weights.  None = equal weights.
        Recommended from H1: [1.0, 0.8, 0.4, 0.2] (low→high CV).
    proj_dim_per_band : int
        Projector output dim per band (smaller than full VICReg).
    """

    def __init__(
        self,
        encoder: UVMDSSLEncoder,
        band_weights: Optional[list] = None,
        proj_dim_per_band: int = 512,
        vicreg_kwargs: Optional[Dict] = None,
        overlap_lambda: float = 0.01,
        overlap_sigma: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        K = encoder.K

        if band_weights is not None:
            assert len(band_weights) == K
            self.register_buffer(
                "band_weights", torch.tensor(band_weights, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "band_weights", torch.ones(K, dtype=torch.float32),
            )

        # Per-band projectors (smaller than full VICReg)
        self.projectors = nn.ModuleList([
            ProjectorMLP(encoder.feat_dim, 512, proj_dim_per_band)
            for _ in range(K)
        ])

        self.augmentation = FreqAwareAugmentation(K=K)
        self.vicreg_loss = VICRegLoss(**(vicreg_kwargs or {}))
        self.overlap_lambda = overlap_lambda
        self.overlap_sigma = overlap_sigma

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        modes = self.encoder.uvmd(x)
        view1, view2 = self.augmentation(modes)

        feats1 = self.encoder.encode_per_band(view1)
        feats2 = self.encoder.encode_per_band(view2)

        total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        details: Dict[str, float] = {}

        for k in range(self.encoder.K):
            p1 = self.projectors[k](feats1[k])
            p2 = self.projectors[k](feats2[k])
            band_loss, band_details = self.vicreg_loss(p1, p2)
            weighted = self.band_weights[k] * band_loss
            total_loss = total_loss + weighted
            details[f"band_{k}_loss"] = band_loss.item()

        # Spectral overlap
        overlap = self.encoder.spectral_overlap_penalty(sigma=self.overlap_sigma)
        total_loss = total_loss + self.overlap_lambda * overlap
        details["overlap_penalty"] = overlap.item()
        details["total_loss"] = total_loss.item()

        return total_loss, details
