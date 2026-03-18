"""
SubcenterArcFaceEMG — ECAPA-TDNN backbone with Sub-center ArcFace loss head.

Hypothesis
──────────
Gesture classes are multi-modal across subjects: different subjects execute the
same gesture with different electrode placements, grip styles, and muscular
activation patterns.  Standard softmax (one centroid per class) blurs all
intra-class sub-clusters into a single mean direction, which is particularly
harmful for cross-subject generalization.

Sub-center ArcFace (K prototype vectors per class) explicitly models up to K
distinct intra-class clusters.  The K sub-centers are learned from training
subjects only; at inference, classification is the argmax of the maximum
cosine similarity over K sub-centers — no per-subject adaptation.

Why this differs from standard (single-center) ArcFace (exp_36)
───────────────────────────────────────────────────────────────
  exp_36 used one learnable prototype per class with ArcFace margin.
  If different subjects occupy distinctly different regions of the sphere
  for the same gesture, a single centroid is pulled in all directions at
  once, degrading the angular separability that ArcFace tries to create.
  With K=3–5 sub-centers, each sub-center can specialise to a different
  "subject style", and the decision boundary only needs to push the best
  matching sub-center past the margin — not all subjects simultaneously.

LOSO / data-leakage safety
──────────────────────────
  ✓ ECAPA BatchNorm running stats: estimated from training subjects only.
    model.eval() at inference → frozen running stats (no test updates).
  ✓ Sub-center weight matrix W: learned from training subjects only.
    At inference, W is fixed — nearest sub-center lookup only.
  ✓ No per-subject normalization inside the model at any stage.
  ✓ ArcFace margin (passed labels in forward) is pure training-time loss
    shaping — it does NOT use test-subject information.

Architecture
────────────
  Input  : (B, C, T)  — EMG windows, channels-first (PyTorch Conv1d format)
  ↓  ECAPA-TDNN encoder
       init_tdnn → 3 × SE-Res2Net blocks → MFA → Attentive Stats Pooling
       → Linear(6C, E) → BN → ReLU → Dropout → L2-normalise
  ↓  (B, E)  — unit-norm embedding on the unit hypersphere
  ↓  SubcenterArcFaceHead   W ∈ ℝ^{num_classes × K × E}
       Training : logit_t = scale · cos(θ_{t,k*} + m)
                  logit_c = scale · max_k cos(θ_{c,k})   (c ≠ t)
                  where k* = argmax_k cos(θ_{t,k})
       Inference: logit_c = scale · max_k cos(θ_{c,k})   ∀ c

Reference
─────────
  Deng et al., "Sub-center ArcFace: Boosting Face Recognition by Large-Scale
  Noisy Web Faces", ECCV 2020.  https://arxiv.org/abs/2004.10448
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ecapa_tdnn_emg import ECAPATDNNEmg


# ─────────────────────────── Sub-center ArcFace head ─────────────────────────


class SubcenterArcFaceHead(nn.Module):
    """
    Sub-center ArcFace classification head.

    Maintains K learnable prototype vectors per class:
      W ∈ ℝ^{num_classes × K × embed_dim}
    Each row W[c, k] is L2-normalised to the unit hypersphere during forward.

    Training forward (self.training=True, labels provided):
        For each sample, for every class c:
          max_cos_c = max_k( embed · W[c, k] )         (best sub-center)
        For the TARGET class t:
          Apply ArcFace margin to max_cos_t:
          logit_t = scale · cos(θ_{t,k*} + m)
        For all NON-TARGET classes c ≠ t:
          logit_c = scale · max_cos_c                  (no margin)

    Inference forward (self.training=False OR labels=None):
        logit_c = scale · max_k cos(θ_{c,k})   ∀ c    (no margin)

    Numerical stability
    ───────────────────
        When θ + m > π (i.e. cos_theta > th = cos(π − m)):
            use  cos_theta − mm  (linear lower bound)  instead of  cos(θ + m).

    Args:
        embed_dim  : embedding dimension (matches backbone output)
        num_classes: number of gesture classes
        K          : number of sub-centers per class (default 3)
        margin     : ArcFace additive angular margin, radians (default 0.35)
        scale      : cosine similarity temperature / logit scale (default 32.0)
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        K: int = 3,
        margin: float = 0.35,
        scale: float = 32.0,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_classes = num_classes
        self.K           = K
        self.margin      = margin
        self.scale       = scale

        # Prototype matrix: (num_classes * K, embed_dim).
        # Stored flat for efficient batched matrix multiply; reshaped to (C, K, d)
        # only when needed.
        self.weight = nn.Parameter(torch.empty(num_classes * K, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute margin constants (re-computed only if margin changes)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Stability threshold: cos(π − m) = −cos(m)
        # If cos_theta > th, θ < π − m, so θ + m < π → safe to use cos(θ + m)
        self.th = math.cos(math.pi - margin)
        # Linear fallback: cos_theta − sin(π − m) · m  = cos_theta − sin(m) · m
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings : (B, embed_dim)  L2-normalised
            labels     : (B,) integer class indices [0, num_classes), or None
        Returns:
            logits     : (B, num_classes)
        """
        # Normalise all sub-center vectors to the unit hypersphere: (C*K, d)
        w = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity between embeddings and every sub-center: (B, C*K)
        cos_all = (embeddings @ w.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Reshape to (B, num_classes, K), take max over K sub-centers per class
        cos_all   = cos_all.view(-1, self.num_classes, self.K)   # (B, C, K)
        cos_theta, _ = cos_all.max(dim=2)                        # (B, C)

        # Pure inference / eval mode: no margin applied
        if labels is None or not self.training:
            return self.scale * cos_theta

        # ── ArcFace margin for the target class (training only) ─────────────
        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        sin_theta   = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=1e-12))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Numerical stability: if θ + m > π, use linear lower bound
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm,
        )

        # One-hot mask: apply margin to target class column only
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = self.scale * (
            one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        )
        return logits


# ────────────────────────────── Full model ───────────────────────────────────


class SubcenterArcFaceEMG(nn.Module):
    """
    ECAPA-TDNN encoder + Sub-center ArcFace head for EMG gesture recognition.

    The ECAPA-TDNN part is identical to ECAPATDNNEmg (exp_62) — the only
    structural change is that the final linear classifier is replaced by a
    SubcenterArcFaceHead with K prototype vectors per class.

    LOSO / data-leakage safety:
      - No per-subject normalization inside the model.
      - BatchNorm running statistics: computed from training subjects only.
        model.eval() at test time freezes these stats — no test-batch updates.
      - Attentive Statistics Pooling: purely input-driven, no stored state.
      - Sub-center weight matrix W: learned from training data; fixed at test.
      - ArcFace forward (with labels) is only ever called during training.
        Inference always uses the no-margin path (labels=None or eval mode).

    Args:
        in_channels   : Number of EMG input channels (e.g. 8).
        num_classes   : Number of gesture classes.
        K             : Sub-centers per class (default 3; range 2–5 suggested).
        channels      : ECAPA internal feature width C (default 128).
        scale         : Res2Net scale / number of sub-groups (default 4).
        embedding_dim : Pre-head embedding dimension E (default 128).
        dilations     : SE-Res2Net block dilation list (default [2, 3, 4]).
        dropout       : Dropout probability before embedding FC (default 0.3).
        se_reduction  : SE bottleneck reduction factor (default 8).
        margin        : ArcFace angular margin in radians (default 0.35).
        arc_scale     : ArcFace logit temperature (default 32.0).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        K: int = 3,
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
        margin: float = 0.35,
        arc_scale: float = 32.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.K             = K

        # ECAPA-TDNN encoder — backbone for multi-scale temporal feature extraction.
        # num_classes=1 is a dummy value; the backbone linear classifier is never
        # called (we run the forward manually, stopping at the embedding layer).
        self._encoder = ECAPATDNNEmg(
            in_channels=in_channels,
            num_classes=1,
            channels=channels,
            scale=scale,
            embedding_dim=embedding_dim,
            dilations=dilations,
            dropout=dropout,
            se_reduction=se_reduction,
        )

        # Sub-center ArcFace classification head
        self.head = SubcenterArcFaceHead(
            embed_dim=embedding_dim,
            num_classes=num_classes,
            K=K,
            margin=margin,
            scale=arc_scale,
        )

    # ── encoder helpers ───────────────────────────────────────────────────────

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalised embeddings from raw EMG windows.

        Runs the ECAPA pipeline up to the embedding FC layer, then applies
        L2 normalisation.  The backbone linear classifier is NOT called.

        Args:
            x: (B, C_emg, T) — EMG windows in channels-first format
        Returns:
            emb: (B, embedding_dim) — unit-norm embedding on unit hypersphere
        """
        enc = self._encoder

        # Initial TDNN: (B, C_emg, T) → (B, C, T)
        out = enc.init_tdnn(x)

        # SE-Res2Net blocks + collect for MFA: [(B, C, T), ...]
        block_outputs = []
        for block in enc.blocks:
            out = block(out)
            block_outputs.append(out)

        # Multi-layer Feature Aggregation: (B, 3C, T)
        mfa_out = enc.mfa(torch.cat(block_outputs, dim=1))

        # Attentive Statistics Pooling: (B, 6C)
        pooled = enc.asp(mfa_out)

        # Embedding FC (Linear → BN → ReLU → Dropout): (B, E)
        emb = enc.embedding(pooled)

        # L2-normalise to unit hypersphere (required for ArcFace cosine geometry)
        return F.normalize(emb, p=2, dim=1)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Training (labels provided, self.training=True):
            ArcFace angular margin applied to the maximum-similarity sub-center
            of the target class.  Non-target classes use plain max cosine sim.

        Inference (labels=None  OR  self.training=False):
            logit_c = scale · max_k cos(θ_{c,k})  — no margin, no adaptation.

        Args:
            x      : (B, C_emg, T) — EMG windows in channels-first format
            labels : (B,) integer class indices [0, num_classes), or None
        Returns:
            logits : (B, num_classes)
        """
        emb = self.get_embeddings(x)      # (B, embedding_dim), L2-normalised
        return self.head(emb, labels)     # (B, num_classes)

    # ── utility ───────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
