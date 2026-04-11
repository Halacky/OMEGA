"""
EMG Patch Token Transformer (ViT-for-EMG) — LOSO cross-subject gesture recognition.

Hypothesis
──────────
CNN-stem approaches may lose useful temporal structure by imposing fixed inductive
biases (locality, channel equivariance).  Treating an EMG window as a sequence of
*patch tokens* — analogous to ViT in vision or sequence models in NLP — gives the
model more freedom to discover long-range and cross-channel patterns that generalize
across subjects.

Architecture
────────────
1. PatchEmbedding      : Split (B, C_emg, T) into non-overlapping time patches of
                         size P → each patch flattened to (C_emg × P) → linear
                         projection to d_model.  Positional encoding is learnable.
2. Performer Encoder   : N layers of {PerformerAttention + FFN + LayerNorm}.
                         Performer = FAVOR+ random feature approximation of softmax
                         attention, giving O(L·m) complexity instead of O(L²).
3. TokenAttentivePooling: Soft-select informative tokens; output weighted mean +
                          weighted std → (2 × d_model).  Same idea as ECAPA ASP.
4. FC Classifier       : Linear(2·d_model, embed_dim) → GELU → Dropout →
                         Linear(embed_dim, num_classes).

Input format : (B, C_emg, T)  — channels-first, matching Conv1d convention.
Output format: (B, num_classes).

LOSO data-leakage guards
─────────────────────────
  ✓ No per-subject normalization inside the model.
  ✓ LayerNorm operates per-sample (no shared running statistics → safe at eval).
  ✓ Performer ω (random features) is sampled once at __init__; not data-dependent.
  ✓ Positional embeddings are learned from training subjects only; frozen at eval.
  ✓ model.eval() at inference: no parameter updates from test-subject batches.
  ✓ TokenAttentivePooling has no stored state — purely input-driven.

Parameter count (default: d_model=128, num_heads=4, num_layers=3, ffn_mult=2,
                          embed_dim=128, 8 EMG channels):
  ≈ 500 K — close to ECAPATDNNEmg (≈ 467 K) for a fair comparison.

References
──────────
- ViT: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
- Performer: Choromanski et al., "Rethinking Attention with Performers" (ICLR 2021)
- ECAPA-TDNN: Desplanques et al., Interspeech 2020
- HAR-Transformer: Yao et al., ACM MobiSys 2023
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────── Patch Embedding ───────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Patchify a 1D EMG signal and project each patch linearly to d_model.

    Input : (B, C_emg, T)
    Output: (B, num_patches, d_model)

    Patches are non-overlapping windows of `patch_size` time steps.
    If T is not divisible by patch_size, the tail is zero-padded.

    This is the 1-D analogue of the ViT "patch + linear projection" stem.
    Unlike a CNN stem, it imposes no locality bias within a patch — the linear
    layer can freely combine any (channel, time) element within the patch.

    Args:
        in_channels : Number of EMG input channels (e.g. 8).
        patch_size  : Length of each patch in time samples (e.g. 25).
        d_model     : Token embedding dimension.
    """

    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels * patch_size      # raw patch dimension
        self.proj = nn.Linear(patch_dim, d_model) # token projection (no bias norm)
        self.norm = nn.LayerNorm(d_model)         # post-projection per-token norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        P = self.patch_size

        # Zero-pad T to the next multiple of P
        pad = (P - T % P) % P
        if pad > 0:
            x = F.pad(x, (0, pad))               # right-pad on time axis
        T_pad = x.shape[2]
        L = T_pad // P                            # number of patches

        # Reshape into patches: (B, C, T) → (B, L, C*P)
        x = x.reshape(B, C, L, P)                # (B, C, L, P)
        x = x.permute(0, 2, 1, 3)                # (B, L, C, P)
        x = x.reshape(B, L, C * P)               # (B, L, C*P)  — flattened patch

        # Linear projection + LayerNorm: (B, L, C*P) → (B, L, d_model)
        return self.norm(self.proj(x))


# ─────────────────────────── Performer Attention ─────────────────────────────

class PerformerAttention(nn.Module):
    """
    FAVOR+ (Fast Attention Via positive Orthogonal Random features).

    Approximates multi-head softmax attention in O(L·m·d) instead of O(L²·d)
    by replacing the softmax kernel with random feature maps φ(x) such that:

        E[φ(q)ᵀ φ(k)] ≈ exp(qᵀk / √d)

    The feature map used (from the Performer paper, positive variant):

        φ(x) = exp(xᵀω / d^{1/4}  −  ‖x‖² / (2√d)) / √m

    where ω ∈ ℝ^{d_head × m} is sampled once at init (Orthogonal Random Features,
    ORF variant).  After normalization (dividing by key feature sums) this gives an
    unbiased approximation of softmax attention.

    Linear attention formula:
        KV = K_feat^T V                       (B, H, m, d)  — precompute key-value
        O_unnorm = Q_feat · KV                (B, H, L, d)  — O(L·m·d)
        norm = Q_feat · K_feat.sum(L)         (B, H, L)
        O = O_unnorm / norm                   (B, H, L, d)

    LOSO safety:
      ✓ ω is sampled at init (not from any data) — registered as a buffer.
      ✓ No running statistics; each forward is independent.
      ✓ model.eval() does not change ω or any parameter.

    Args:
        d_model     : Total model dimension.
        num_heads   : Number of attention heads (d_model must be divisible).
        num_features: m — random feature projection count (more = lower variance).
        dropout     : Output projection dropout.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_features: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
        self.num_heads    = num_heads
        self.d_head       = d_model // num_heads
        self.num_features = num_features

        # Query, key, value projections — no bias (standard Transformer practice)
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

        # Random projection matrix ω ∈ ℝ^{d_head × m} — fixed, not learned.
        # Registered as a buffer so it moves to the correct device with the model.
        omega = self._sample_omega(self.d_head, num_features)
        self.register_buffer("omega", omega)   # (d_head, m)

    @staticmethod
    def _sample_omega(d: int, m: int) -> torch.Tensor:
        """
        Orthogonal Random Features (ORF) projection matrix ω ∈ ℝ^{d×m}.

        Each block of d columns is a random orthogonal d×d matrix whose rows are
        scaled by chi(d)-distributed norms.  This gives each column the same
        marginal distribution as N(0, I_d) while pairwise orthogonality within
        a block reduces estimation variance vs. iid Gaussian features.

        For m ≤ d: a single block is built and the first m columns are taken.
        For m > d: ceil(m/d) independent blocks are stacked and trimmed to m.
        """
        num_blocks = math.ceil(m / d)
        blocks = []
        for _ in range(num_blocks):
            G = torch.randn(d, d)              # iid N(0, 1) — Gaussian matrix
            Q, _ = torch.linalg.qr(G)          # Q: (d, d) — random orthogonal
            # Scale row i by ‖G[i, :]‖ ~ chi(d) so each row ~ N(0, I_d)
            row_norms = G.norm(dim=1)           # (d,)
            S = Q * row_norms.unsqueeze(1)      # S[i, :] = ‖G[i]‖ · Q[i, :]
            blocks.append(S.T)                  # (d, d) — columns are the ωᵢ
        omega = torch.cat(blocks, dim=1)[:, :m] # (d, m)
        return omega

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Positive random feature map (FAVOR+ variant):

            φ(x) = exp(xᵀω / d^{1/4}  −  ‖x‖² / (2√d)) / √m

        The d^{1/4} denominator ensures that for two vectors q, k:
            E[φ(q)ᵀ φ(k)] ≈ exp(qᵀk / √d)
        matching the standard attention scaling.

        The norm_sq term makes the feature map always positive (no cancellations),
        which is necessary for the O(L·m) linear attention re-ordering to hold.

        After normalisation by the key feature sums the norm terms cancel,
        so this is an unbiased estimator of softmax attention.

        Args:
            x: (B, H, L, d_head)
        Returns:
            (B, H, L, m) — positive features
        """
        scale = self.d_head ** 0.25
        x_s   = x / scale                                          # (B, H, L, d)
        proj  = torch.einsum("bhld,dm->bhlm", x_s, self.omega)    # (B, H, L, m)
        norm_sq = (x_s ** 2).sum(dim=-1, keepdim=True) / 2        # (B, H, L, 1)
        feat  = torch.exp(proj - norm_sq) / (self.num_features ** 0.5)
        return feat                                                 # (B, H, L, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model) — Performer attention output
        """
        B, L, D = x.shape
        H, d = self.num_heads, self.d_head

        # Project and split into heads: (B, H, L, d_head)
        Q = self.q_proj(x).reshape(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, L, H, d).permute(0, 2, 1, 3)

        # Random feature maps
        Q_feat = self._phi(Q)   # (B, H, L, m)
        K_feat = self._phi(K)   # (B, H, L, m)

        # ── Linear attention: O(L·m·d) ──────────────────────────────────
        # Precompute K^T V:  (B, H, m, d)
        KV = torch.einsum("bhlm,bhld->bhmd", K_feat, V)

        # Unnormalized output:  (B, H, L, d)
        out = torch.einsum("bhlm,bhmd->bhld", Q_feat, KV)

        # Normalise by sum of key features (ensures rows sum to 1 like softmax)
        K_sum = K_feat.sum(dim=2)                                  # (B, H, m)
        norm  = torch.einsum("bhlm,bhm->bhl", Q_feat, K_sum)      # (B, H, L)
        norm  = norm.unsqueeze(-1).clamp(min=1e-6)                 # (B, H, L, 1)
        out   = out / norm                                         # (B, H, L, d)

        # Recombine heads: (B, H, L, d) → (B, L, D)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        out = self.out_proj(out)
        return self.drop(out)


# ──────────────────────────── Performer Encoder Layer ────────────────────────

class PerformerLayer(nn.Module):
    """
    One Transformer encoder layer using Performer (linear) attention.

    Uses the Pre-LN (LayerNorm before sub-layer) residual layout, which is more
    stable to train than Post-LN without warm-up — recommended when num_layers > 2.

    Flow:
        x → LayerNorm → PerformerAttention → residual → x′
        x′ → LayerNorm → FFN (Linear–GELU–Drop–Linear–Drop) → residual → output

    LOSO safety:
      ✓ LayerNorm is per-sample — no batch-level running statistics.
      ✓ No BatchNorm (which would require training-set running stats at eval).

    Args:
        d_model     : Model dimension.
        num_heads   : Number of attention heads.
        ffn_dim     : FFN hidden dimension (default: 2 × d_model).
        num_features: Performer random feature count.
        dropout     : Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        num_features: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        ffn_dim = ffn_dim or d_model * 2

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = PerformerAttention(d_model, num_heads, num_features, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # Pre-LN attention + residual
        x = x + self.ffn(self.norm2(x))    # Pre-LN FFN + residual
        return x


# ─────────────────── Token Attentive Statistics Pooling ──────────────────────

class TokenAttentivePooling(nn.Module):
    """
    Attentive Statistics Pooling over the *token* (sequence) dimension.

    Analogous to ECAPA-TDNN's AttentiveStatisticsPooling but operating on the
    patch-token axis of a Transformer instead of the time axis of a TDNN.

    Computes a soft attention weight per token via a small score network, then
    outputs both the weighted mean and weighted standard deviation, giving a
    2 × d_model fixed-size representation.

    Why mean + std?
      The mean captures the average token activation (gesture type), while the std
      captures how much token responses vary across patches (gesture dynamics and
      speed variation across subjects).  Together they are more discriminative than
      mean-only pooling (CLS token or global average) and avoid sensitivity to
      temporal ordering.

    LOSO integrity:
      Attention weights are computed purely from the input at inference time —
      no running statistics, no stored per-subject state.

    Args:
        d_model: Input / output feature dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        hidden = max(d_model // 4, 16)
        # Score network: (B, L, d_model) → (B, L, 1) scalar per token
        self.score_net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, 2 * d_model) — [weighted_mean ‖ weighted_std]
        """
        alpha = self.score_net(x)           # (B, L, 1) — raw scores
        alpha = F.softmax(alpha, dim=1)     # (B, L, 1) — weights, Σ_L = 1

        mu    = (alpha * x).sum(dim=1)                       # (B, d_model)
        var   = (alpha * x.pow(2)).sum(dim=1) - mu.pow(2)    # (B, d_model)
        sigma = var.clamp(min=1e-8).sqrt()                   # (B, d_model)

        return torch.cat([mu, sigma], dim=1)                 # (B, 2*d_model)


# ─────────────────────────── Full Model ──────────────────────────────────────

class EMGPatchTransformer(nn.Module):
    """
    EMG Patch Token Transformer for cross-subject gesture recognition (LOSO).

    Input : (B, C_emg, T)  — channels-first (matching Conv1d / ECAPATDNNEmg convention)
    Output: (B, num_classes)

    Architecture summary
    ────────────────────
    1. PatchEmbedding       : T → L = ⌊T/P⌋ tokens, each of dim d_model.
    2. Positional Encoding  : Learnable position embedding (max_patches × d_model).
    3. Performer Encoder    : num_layers × PerformerLayer (Pre-LN, linear attention).
    4. Post-encoder LayerNorm
    5. TokenAttentivePooling: (B, L, d_model) → (B, 2·d_model).
    6. FC Embedding         : Linear(2·d_model, embed_dim) + GELU + Dropout.
    7. Classifier           : Linear(embed_dim, num_classes).

    Why patchify + Performer instead of CNN-GRU?
    ─────────────────────────────────────────────
    • CNN stems impose locality + translation-equivariance which may over-fit to
      subject-specific muscle activation topography.
    • Patch tokens allow the attention mechanism to freely attend to any
      (channel × time-step) sub-window — a more flexible inductive bias.
    • Performer's linear attention is O(L·m) so it scales if patch_size is
      reduced (more tokens) in future experiments.
    • TokenAttentivePooling mirrors ECAPA-TDNN's ASP: robust to gesture-speed
      variation across subjects since it is permutation-aware via learned weights.

    LOSO data-leakage notes
    ─────────────────────────
      ✓ No per-subject normalization inside the model.
      ✓ LayerNorm (Pre-LN everywhere): per-sample, no batch-level running stats.
      ✓ Performer ω: sampled at __init__, not data-dependent.
      ✓ Positional embeddings: learned from train subjects; frozen at eval.
      ✓ model.eval(): no weight updates, no test-subject adaptation.
      ✓ TokenAttentivePooling: purely feedforward, no stored state.

    Args:
        in_channels  : Number of EMG input channels (e.g. 8).
        num_classes  : Number of gesture classes.
        patch_size   : Time samples per patch (default 25 → 24 patches for T=600).
        d_model      : Transformer hidden dimension (default 128).
        num_heads    : Attention heads per Performer layer (default 4).
        num_layers   : Number of stacked Performer layers (default 3).
        num_features : Random feature count m for Performer (default 64).
        ffn_mult     : FFN hidden = d_model * ffn_mult (default 2).
        embed_dim    : Pre-classifier embedding dimension (default 128).
        dropout      : Dropout probability (default 0.1).
        max_patches  : Max sequence length supported (default 64, safe for T≤1600
                       with P=25).  Increase if using smaller patch_size or longer T.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        patch_size: int = 25,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        num_features: int = 64,
        ffn_mult: int = 2,
        embed_dim: int = 128,
        dropout: float = 0.1,
        max_patches: int = 64,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.d_model     = d_model
        self.max_patches = max_patches

        # 1. Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, patch_size, d_model)

        # 2. Learnable positional encoding
        # Smaller init than the default (following ViT's std=0.02) so position
        # information doesn't dominate patch content at the start of training.
        self.pos_embed = nn.Embedding(max_patches, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        # 3. Performer encoder: num_layers stacked layers
        ffn_dim = d_model * ffn_mult
        self.encoder = nn.Sequential(*[
            PerformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                num_features=num_features,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 4. Post-encoder LayerNorm (Pre-LN architecture needs this after last block)
        self.post_norm = nn.LayerNorm(d_model)

        # 5. Attentive statistics pooling
        self.pool = TokenAttentivePooling(d_model)

        # 6. FC embedding (collapsed to embed_dim)
        pool_dim = d_model * 2
        self.embedding = nn.Sequential(
            nn.Linear(pool_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 7. Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Weight init: kaiming for linear layers, ones/zeros for norms
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # pos_embed already initialised with std=0.02 in __init__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_emg, T) — EMG windows in channels-first format.
        Returns:
            logits: (B, num_classes)
        """
        # 1. Patch embedding: (B, C, T) → (B, L, d_model)
        tokens = self.patch_embed(x)                       # (B, L, d_model)
        B, L, _ = tokens.shape

        assert L <= self.max_patches, (
            f"num_patches L={L} > max_patches={self.max_patches}.  "
            f"Reduce patch_size or increase max_patches."
        )

        # 2. Positional encoding: add learnable embedding for each position
        pos = torch.arange(L, device=x.device)            # (L,)
        tokens = tokens + self.pos_embed(pos)              # (B, L, d_model)

        # 3. Performer encoder (Pre-LN residual Transformer)
        tokens = self.encoder(tokens)                      # (B, L, d_model)

        # 4. Post-encoder LayerNorm
        tokens = self.post_norm(tokens)                    # (B, L, d_model)

        # 5. Attentive statistics pooling: collapse L → fixed-size vector
        pooled = self.pool(tokens)                         # (B, 2*d_model)

        # 6. Embedding projection
        emb = self.embedding(pooled)                       # (B, embed_dim)

        # 7. Classification
        return self.classifier(emb)                        # (B, num_classes)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
