"""
PrototypicalArcFaceEMG — EMG gesture classification via prototypical networks.

Hypothesis H10: Classes are better modelled as distributions in embedding space
than as SVM decision boundaries. Prototypical networks are more robust to
domain shift because the embedding distance metric generalises better across
subjects.

Design:
  Input  : (B, C, T)  — PyTorch conv format (C channels, T time steps)
  ↓  EMGEmbeddingNet  — CNN blocks → BiGRU → FC → L2-normalised embedding
  ↓  (B, embed_dim)
  ↓  ArcFaceHead      — cosine similarity against class prototypes
                         + additive angular margin during training

Training:   ArcFace logits = scale · cos(θ + m)  for target class
                           = scale · cos(θ)        for non-target classes
Inference:  logits = scale · cos(θ)  against stored class prototypes

After training, call update_prototypes() to replace the learned weight matrix
with the empirical mean of training embeddings per class.  This step realises
the "class = mean embedding" interpretation from the hypothesis.

Reference:
  Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
  CVPR 2019. https://arxiv.org/abs/1801.07698
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embedding network
# ---------------------------------------------------------------------------

class EMGEmbeddingNet(nn.Module):
    """
    CNN + BiGRU embedding network for raw EMG windows.

    Architecture:
      3 × Conv1d stages (64 → 128 → 128 channels, BN, ReLU)  → MaxPool(2)
      2-layer BiGRU  (128 → 256 hidden, bidirectional)
      GlobalAvgPool → Linear(256, embed_dim) → BN → L2-normalise

    Input  : (B, C, T)     — B samples, C EMG channels, T time steps
    Output : (B, embed_dim) — unit-norm embeddings on the unit hypersphere
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim

        # ---- CNN feature extraction ----
        self.cnn = nn.Sequential(
            # Stage 1: capture local EMG activation patterns
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Stage 2: wider receptive field
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),          # T → T // 2
            # Stage 3: deep features + dropout regularisation
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- BiGRU temporal aggregation ----
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # ---- Projection to embedding space ----
        # 256 = 2 × 128 from bidirectional GRU
        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EMG windows in PyTorch format
        Returns:
            embed: (B, embed_dim) L2-normalised embeddings
        """
        feat = self.cnn(x)                   # (B, 128, T//2)
        feat = feat.permute(0, 2, 1)         # (B, T//2, 128)
        gru_out, _ = self.gru(feat)          # (B, T//2, 256)
        pooled = gru_out.mean(dim=1)         # (B, 256) — global average pooling
        embed = self.proj(pooled)            # (B, embed_dim)
        return F.normalize(embed, p=2, dim=1)


# ---------------------------------------------------------------------------
# ArcFace head
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace) classification head.

    Maintains a learnable weight matrix W ∈ ℝ^{num_classes × embed_dim}.
    Each row Wᵢ is L2-normalised, representing the i-th class prototype
    on the unit hypersphere.

    Training forward  (labels provided, self.training=True):
        cos_θᵢ = embed · Wᵢᵀ
        For the target class t: logit_t  = scale · cos(θ_t + m)
        For all other classes:  logit_k  = scale · cos(θ_k)

    Inference forward  (labels=None OR self.training=False):
        logit_i = scale · cos_θᵢ   (nearest prototype)

    Numerical stability:
        If θ_t + m > π, use  cos_θ_t − m·sin(π − m)  (linear approximation).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        margin: float = 0.3,
        scale: float = 32.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Learnable class prototype matrix
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute margin constants (avoid recomputing each forward)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)       # cos(π − m): stability threshold
        self.mm = math.sin(math.pi - margin) * margin  # m·sin(π − m): linear fallback

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings : (B, embed_dim) — L2-normalised (from EMGEmbeddingNet)
            labels     : (B,) integer class indices, or None for inference
        Returns:
            logits     : (B, num_classes) — scaled cosine similarities
        """
        # Row-normalise class prototypes (unit hypersphere)
        w = F.normalize(self.weight, p=2, dim=1)          # (num_classes, embed_dim)

        # Cosine similarity between embeddings and all class prototypes
        cos_theta = embeddings @ w.T                       # (B, num_classes)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Inference: no margin, pure cosine similarity
        if labels is None or not self.training:
            return self.scale * cos_theta

        # --- ArcFace angular margin (training only) ---
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # cos(θ + m) = cos(θ)·cos(m) − sin(θ)·sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Numerical stability: if θ + m > π, switch to linear lower bound
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm,
        )

        # Apply margin only to the ground-truth class via one-hot mask
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = self.scale * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
        return logits


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PrototypicalArcFaceEMG(nn.Module):
    """
    End-to-end prototypical network with ArcFace loss.

    Usage:
        model = PrototypicalArcFaceEMG(in_channels=8, num_classes=10)

        # ---- Training ----
        model.train()
        logits = model(x, labels=y)          # ArcFace logits with angular margin
        loss   = F.cross_entropy(logits, y)

        # ---- Post-training prototype update ----
        model.update_prototypes(X_train_tensor, y_train_tensor, num_classes, device)

        # ---- Inference ----
        model.eval()
        logits = model(x)                    # cosine similarity against prototypes
        pred   = logits.argmax(dim=1)        # nearest prototype classification
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 128,
        dropout: float = 0.3,
        margin: float = 0.3,
        scale: float = 32.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding_net = EMGEmbeddingNet(in_channels, embed_dim, dropout)
        self.arcface_head = ArcFaceHead(embed_dim, num_classes, margin, scale)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised embeddings.  (B, C, T) → (B, embed_dim)."""
        return self.embedding_net(x)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Training (labels provided + self.training=True):
            Returns ArcFace logits — angular margin applied to target class.
        Inference (labels=None  OR  self.training=False):
            Returns cosine similarity × scale — nearest prototype classification.
        """
        embeddings = self.embedding_net(x)           # (B, embed_dim)
        return self.arcface_head(embeddings, labels)  # (B, num_classes)

    @torch.no_grad()
    def update_prototypes(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_classes: int,
        device: str,
        batch_size: int = 512,
    ) -> None:
        """
        Replace the ArcFace weight matrix with empirical class mean embeddings.

        This operation grounds the classification in the prototypical interpretation:
        prototype_c = L2_normalise( mean{ embed(x) : class(x) = c } )

        The model transitions from "learned linear boundary in embedding space"
        (ArcFace training) to "nearest mean embedding" (prototypical inference).

        Args:
            X         : (N, C, T) normalised training windows (tensor or ndarray)
            y         : (N,)  integer class labels
            num_classes: number of gesture classes
            device    : torch device string (e.g. "cuda" or "cpu")
            batch_size: batch size for embedding extraction (memory management)
        """
        import numpy as _np

        self.eval()

        if isinstance(X, _np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, _np.ndarray):
            y = torch.from_numpy(y).long()

        # --- Extract embeddings in batches ---
        all_embeddings: list = []
        for start in range(0, len(X), batch_size):
            xb = X[start: start + batch_size].to(device)
            emb = self.embedding_net(xb)     # (batch, embed_dim), already L2-normalised
            all_embeddings.append(emb.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)   # (N, embed_dim)

        # --- Compute per-class mean and L2-normalise to unit sphere ---
        prototypes = torch.zeros(num_classes, self.embed_dim)
        for c in range(num_classes):
            mask = (y == c)
            if mask.sum() > 0:
                proto = all_embeddings[mask].mean(dim=0)
                prototypes[c] = F.normalize(proto, p=2, dim=0)
            else:
                # No training samples → random unit-norm fallback
                prototypes[c] = F.normalize(torch.randn(self.embed_dim), p=2, dim=0)

        # --- Overwrite ArcFace weight matrix with mean prototypes ---
        self.arcface_head.weight.data.copy_(prototypes)
