"""
HyGDL: Analytical Orthogonal Content/Style Projection on ECAPA-TDNN Embeddings.

Hypothesis 3: Use closed-form (analytical) projection for content/style
decomposition instead of learnable adversarial or VAE-based disentanglement.

Key idea (inspired by HyGDL, "Hyper Geodesic Disentanglement Learning"):
  1. ECAPA-TDNN encoder:  x → z ∈ R^E
  2. Style subspace (training-time, no gradients):
       For each training subject s, compute μ_s = E[encoder(x_s)].
       Stack {μ_s} → centre → SVD → V_style ∈ R^{E×k}  (top-k columns capture
       inter-subject embedding variance).
  3. Analytical projection:
       z_style   = z @ V_style @ V_style^T   (projection onto style subspace)
       z_content = z − z_style               (orthogonal complement — guaranteed)
  4. Classifier on z_content only  [CE loss]
  5. Decoder (z → x̂ compressed)   [MSE regulariser — ensures encoder is not degenerate]

Analytical projection properties:
  • Guaranteed mathematical orthogonality: z_content ⊥ z_style (not approximate)
  • No extra trainable parameters for disentanglement
  • No adversarial dynamics — stable training
  • V_style is differentiable w.r.t. z  (it is a fixed buffer, not a Parameter)
    → encoder receives proper gradient signal through z_content

LOSO data-leakage checklist:
  ✓ BN running stats: accumulated from training batches (model.train()) only.
  ✓ Channel normalisation: mean/std from X_train only.
  ✓ V_style: computed via encoder(X_train_subj_i) in no_grad+eval mode.
    Only training-subject windows flow into the SVD; test subject never enters.
  ✓ Test evaluation: model.eval() + frozen V_style → pure analytical projection.
    No adaptation, no BN updates from test-subject data.

Input format:  (B, C_emg, T)   — channels-first, matching PyTorch Conv1d.
Encoder output: (B, embedding_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# Reuse the proven ECAPA-TDNN building blocks from experiment 62.
# This avoids code duplication and guarantees identical low-level behaviour.
from models.ecapa_tdnn_emg import (
    SERes2NetBlock,
    AttentiveStatisticsPooling,
)


# ──────────────────────────── Encoder backbone ────────────────────────────────


class HyGDLEncoderECAPA(nn.Module):
    """
    ECAPA-TDNN encoder without the final classification head.

    Produces embedding z ∈ R^embedding_dim from raw EMG (B, C_emg, T).
    Architecture is identical to ECAPATDNNEmg up to and including the FC
    embedding layer; the final nn.Linear classifier is absent.

    LOSO note:
      BN running statistics are accumulated during model.train() forward passes
      (training batches only).  When called with model.eval() — e.g. during the
      periodic V_style estimation — BN uses frozen running stats from training
      data, producing no information leakage from test data.

    Args:
        in_channels:   Number of EMG input channels (e.g. 8).
        channels:      C — internal TDNN feature dimension.
        scale:         Res2Net scale (number of sub-groups per block).
        embedding_dim: E — output embedding dimension.
        dilations:     Dilation per SE-Res2Net block (exactly 3 values).
        dropout:       Dropout before/after the FC embedding.
        se_reduction:  SE bottleneck reduction factor.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2, 3, 4]
        if len(dilations) != 3:
            raise ValueError(
                f"Exactly 3 dilation values required, got {len(dilations)}: {dilations}"
            )

        self.embedding_dim = embedding_dim
        num_blocks = len(dilations)

        # ── 1. Initial TDNN ────────────────────────────────────────────────
        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # ── 2. SE-Res2Net blocks (increasing dilation) ────────────────────
        self.blocks = nn.ModuleList([
            SERes2NetBlock(
                channels,
                kernel_size=3,
                dilation=d,
                scale=scale,
                se_reduction=se_reduction,
            )
            for d in dilations
        ])

        # ── 3. Multi-layer Feature Aggregation ────────────────────────────
        # Concatenate all block outputs → 1×1 conv mixes cross-scale features.
        mfa_in = channels * num_blocks
        self.mfa = nn.Sequential(
            nn.Conv1d(mfa_in, mfa_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_in),
            nn.ReLU(inplace=True),
        )

        # ── 4. Attentive Statistics Pooling ───────────────────────────────
        # (B, 3C, T) → (B, 6C)  [weighted mean + weighted std]
        self.asp = AttentiveStatisticsPooling(mfa_in)

        # ── 5. FC embedding (no classifier here) ──────────────────────────
        asp_out_dim = mfa_in * 2  # 6C
        self.embedding = nn.Sequential(
            nn.Linear(asp_out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_emg, T) — EMG windows, channels-first.
        Returns:
            z: (B, embedding_dim)
        """
        out = self.init_tdnn(x)

        block_outputs: List[torch.Tensor] = []
        for block in self.blocks:
            out = block(out)
            block_outputs.append(out)

        mfa_in = torch.cat(block_outputs, dim=1)   # (B, 3C, T)
        mfa_out = self.mfa(mfa_in)                  # (B, 3C, T)
        pooled = self.asp(mfa_out)                  # (B, 6C)
        z = self.embedding(pooled)                  # (B, E)
        return z


# ──────────────────────────── Full HyGDL model ────────────────────────────────


class HyGDLModel(nn.Module):
    """
    HyGDL full model: ECAPA encoder + analytical projection + classifier + decoder.

    The style subspace V_style ∈ R^{E×k} is a non-trainable buffer (registered
    via register_buffer).  It is updated externally by the trainer after each
    periodic SVD of inter-subject mean embeddings — never via backpropagation.

    Projection is differentiable w.r.t. z (V_style is fixed), so the encoder
    receives a meaningful gradient: "produce z_content that is informative for
    gesture classification and simultaneously orthogonal to the style subspace."

    Training phases (managed by HyGDLTrainer):
      Phase 1 — Warmup:       projection_valid=False → classifier sees full z.
      Phase 2 — Disentangle:  V_style updated from training subjects → projection
                               active → classifier sees only z_content.

    Args:
        in_channels:   Number of EMG input channels (e.g. 8).
        num_classes:   Number of gesture classes.
        embedding_dim: E — encoder output dimension (default 128).
        style_dim:     k — style subspace rank (default 4).
                       Effective rank = min(k, n_train_subjects − 1) when
                       training subjects are fewer than k.
        t_compressed:  Time steps in reconstruction target (default 75).
                       Should satisfy T // t_compressed ≥ 1 (e.g. T=600, tc=75 → stride=8).
        channels, scale, dilations, dropout, se_reduction: ECAPA encoder params.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embedding_dim: int = 128,
        style_dim: int = 4,
        t_compressed: int = 75,
        channels: int = 128,
        scale: int = 4,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.style_dim = style_dim
        self.t_compressed = t_compressed
        self.in_channels = in_channels

        # ── ECAPA-TDNN encoder ────────────────────────────────────────────
        self.encoder = HyGDLEncoderECAPA(
            in_channels=in_channels,
            channels=channels,
            scale=scale,
            embedding_dim=embedding_dim,
            dilations=dilations,
            dropout=dropout,
            se_reduction=se_reduction,
        )

        # ── Gesture classifier (operates on z_content only) ───────────────
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # ── Reconstruction decoder (operates on full z = z_content + z_style)
        # Purpose: regularise encoder — ensures z retains enough information
        # to reconstruct the signal, preventing degenerate collapse.
        # Lightweight two-layer MLP: comparable total param count to baseline.
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, in_channels * t_compressed),
        )

        # ── Style projection buffer (NOT a trainable Parameter) ───────────
        # V_style: (E, k) — orthonormal columns spanning inter-subject mean
        #          embedding variance (top-k right singular vectors).
        # projection_valid: False until the first update_style_subspace() call.
        self.register_buffer("V_style", torch.zeros(embedding_dim, style_dim))
        self.register_buffer("projection_valid", torch.tensor(False))

        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Public API: style subspace update ─────────────────────────────────

    def update_style_subspace(self, V_new: torch.Tensor) -> None:
        """
        Replace the V_style buffer with a freshly computed orthonormal basis.

        Must be called by the trainer using training-subject embeddings only.
        V_new must have shape (E, k) with orthonormal columns (from SVD).

        This operation does NOT participate in autograd.  Copying a float32
        tensor into a registered buffer has no effect on the computation graph
        of any in-progress backward pass, and creates no gradient history.

        LOSO guard: The trainer calls this only from within fit(), after
        forward-passing training-subject windows (in no_grad + eval mode).
        Test-subject data never reaches this method.

        Args:
            V_new: (embedding_dim, style_dim) orthonormal float32 tensor.
        """
        if V_new.shape != (self.embedding_dim, self.style_dim):
            raise ValueError(
                f"V_style shape mismatch: expected "
                f"({self.embedding_dim}, {self.style_dim}), got {tuple(V_new.shape)}"
            )
        self.V_style.copy_(V_new.to(self.V_style.device))
        self.projection_valid.fill_(True)

    # ── Core projection ────────────────────────────────────────────────────

    def project(
        self, z: torch.Tensor
    ):
        """
        Analytical orthogonal projection of z onto content / style subspaces.

        Math:
          P_style   = V_style @ V_style^T       (symmetric, idempotent projector)
          z_style   = z @ P_style               (projection onto style subspace)
          z_content = z − z_style               (orthogonal complement — exact)

        This operation is differentiable w.r.t. z — V_style is a constant buffer.
        The encoder therefore receives gradient signal:
          ∂L_cls / ∂z_content  →  (I - P_style)^T  →  ∂z (through encoder).

        When projection_valid is False (pre-warmup, no V_style yet), the method
        returns z_content = z and z_style = 0, so the classifier trains on the
        full embedding without any projection overhead.

        Args:
            z: (B, E)
        Returns:
            z_content: (B, E)  — gesture-informative, orthogonal to style
            z_style:   (B, E)  — inter-subject variance component
        """
        if not self.projection_valid.item():
            return z, torch.zeros_like(z)

        # V_style: (E, k)
        # z_style_coeff = z @ V_style       : (B, k) — coords in style subspace
        # z_style       = z_style_coeff @ V_style.T : (B, E) — back to ambient
        z_style_coeff = z @ self.V_style          # (B, k)
        z_style       = z_style_coeff @ self.V_style.T  # (B, E)
        z_content     = z - z_style                     # (B, E)
        return z_content, z_style

    # ── Forward passes ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward for inference — returns classification logits.

        Args:
            x: (B, C, T) EMG windows, channels-first, already standardised.
        Returns:
            logits: (B, num_classes)
        """
        z = self.encoder(x)
        z_content, _ = self.project(z)
        return self.classifier(z_content)

    def forward_with_reconstruction(self, x: torch.Tensor):
        """
        Extended forward returning logits, reconstruction output, and embeddings.

        Used during Phase 2 training to compute both classification and
        reconstruction losses in a single forward pass.

        Reconstruction target: average-pool the input x over time to t_compressed
        steps.  The decoder maps the full embedding z (not just z_content) to the
        target shape; this ensures the encoder has an incentive to retain all
        signal information — style information is needed for reconstruction even
        though the classifier ignores it.

        Args:
            x: (B, C, T) input EMG windows (standardised, channels-first).
        Returns:
            logits:    (B, num_classes)
            x_hat:     (B, C, t_compressed) — decoder reconstruction
            x_target:  (B, C, t_compressed) — avg-pooled reconstruction target
            z_content: (B, E)
            z_style:   (B, E)
        """
        z = self.encoder(x)
        z_content, z_style = self.project(z)
        logits = self.classifier(z_content)

        # Reconstruction from full z (z_content + z_style)
        x_hat_flat = self.decoder(z)                              # (B, C * t_compressed)
        x_hat      = x_hat_flat.view(x.size(0), self.in_channels, self.t_compressed)

        # Reconstruction target: average-pool x to exactly t_compressed steps.
        # stride = T // t_compressed (integer, at least 1).
        T = x.size(2)
        stride = max(1, T // self.t_compressed)
        x_down = F.avg_pool1d(x, kernel_size=stride, stride=stride)  # (B, C, T')

        # Trim or zero-pad to exactly t_compressed
        if x_down.size(2) > self.t_compressed:
            x_target = x_down[:, :, : self.t_compressed]
        elif x_down.size(2) < self.t_compressed:
            x_target = F.pad(x_down, (0, self.t_compressed - x_down.size(2)))
        else:
            x_target = x_down

        return logits, x_hat, x_target, z_content, z_style

    # ── Utility ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
