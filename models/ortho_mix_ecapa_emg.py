"""
OrthoMixECAPAEmg: ECAPATDNNEmg with orthogonal channel-mixing augmentation.

Hypothesis
──────────
Inter-subject electrode variability is partly a linear mixing phenomenon:
surface EMG electrodes capture weighted sums of signals from neighbouring
muscles, and those weights shift between subjects as a function of anatomy
and electrode placement.  A random orthogonal matrix A close to the identity
(A = exp(εS), S skew-symmetric) simulates this during training without
arbitrary permutation, encouraging the backbone to learn representations
invariant to soft channel mixing.

Augmentation scheme
───────────────────
For each training batch X ∈ ℝ^{B×C×T} (channels-first):

    R  ~ N(0, I_C)                       # random C×C matrix
    S  = ε · (R − Rᵀ) / 2               # skew-symmetric: Sᵀ = −S
    A  = matrix_exp(S) ∈ SO(C)           # orthogonal: AᵀA = I, det A = ±1
    X' = einsum('ij,bjt→bit', A, X)      # mix channels for every time step

Properties:
  ─ ‖A − I‖_F = O(ε): soft mixing for small ε (default ε = 0.1)
  ─ ‖X'‖_2 = ‖X‖_2: orthogonal transform preserves L2 norm
  ─ Probabilistic (mix_prob): not every batch is augmented
  ─ Per-batch sampling: one A per batch, constant across windows in that batch
    (biologically: one electrode placement per "recording session / subject")
  ─ A has no trainable parameters — freshly sampled at forward time
  ─ Augmentation gated by self.training: OFF at eval() / test time

Why NOT arbitrary channel permutation?
  SetTransformer-style random permutation destroys anatomical ordering.
  Here we use a nearly-diagonal orthogonal matrix, so adjacent muscles blend
  gently rather than swap arbitrarily.  This is closer to the physical model
  of electrode cross-talk between neighbouring muscles.

LOSO data-leakage guards
────────────────────────
  ✓ A is sampled from N(0,1) — no subject statistics, no test data.
  ✓ Augmentation is disabled (self.training = False) at val/test inference.
  ✓ Backbone ECAPATDNNEmg inherits all LOSO guarantees from exp_62.
  ✓ No per-subject adaptive normalisation anywhere in this module.
  ✓ A is detached (torch.no_grad) — no gradient through random sampling.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from models.ecapa_tdnn_emg import ECAPATDNNEmg


class OrthoMixECAPAEmg(nn.Module):
    """
    ECAPATDNNEmg backbone with stochastic orthogonal channel-mixing augmentation.

    Wraps ECAPATDNNEmg and applies _ortho_mix() inside forward() only when
    self.training is True.  When self.training is False (model.eval() called),
    forward() is identical to plain ECAPATDNNEmg — zero test-time overhead.

    The mixing module has no learnable parameters; the only trainable weights
    are those of the ECAPATDNNEmg backbone.

    Args:
        in_channels:   Number of EMG input channels (e.g. 8).
        num_classes:   Number of gesture classes.
        channels:      C — internal TDNN feature width (default 128).
        scale:         Res2Net scale / sub-group count (default 4).
        embedding_dim: E — pre-classifier embedding dimension (default 128).
        dilations:     Dilation per SE-Res2Net block (default [2, 3, 4]).
        dropout:       Dropout probability before classifier (default 0.3).
        se_reduction:  SE bottleneck reduction factor (default 8).
        mix_epsilon:   ε — scale of skew-symmetric perturbation (default 0.1).
                       Controls ‖A − I‖_F.  Smaller → closer to identity.
        mix_prob:      Per-batch probability of applying mixing (default 0.7).
                       Prevents over-regularisation while maintaining diversity.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
        mix_epsilon: float = 0.1,
        mix_prob: float = 0.7,
    ):
        super().__init__()

        if not (0.0 < mix_epsilon <= 1.0):
            raise ValueError(
                f"mix_epsilon should be in (0, 1], got {mix_epsilon}. "
                "Larger values produce orthogonal matrices far from identity."
            )
        if not (0.0 < mix_prob <= 1.0):
            raise ValueError(f"mix_prob must be in (0, 1], got {mix_prob}.")

        self.mix_epsilon = mix_epsilon
        self.mix_prob = mix_prob

        # Backbone: full ECAPATDNNEmg — unchanged architecture
        self.backbone = ECAPATDNNEmg(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=channels,
            scale=scale,
            embedding_dim=embedding_dim,
            dilations=dilations,
            dropout=dropout,
            se_reduction=se_reduction,
        )

    # ── orthogonal channel mixing ─────────────────────────────────────────────

    def _ortho_mix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Sample and apply a random orthogonal channel-mixing matrix.

        Constructs A = matrix_exp(ε · S) where S is a fresh random
        skew-symmetric matrix (S = (R − Rᵀ)/2, R ~ N(0,1)).

        Properties of A:
          • AᵀA = I  (orthogonal)               — guaranteed by exp(skew-sym)
          • det A = ±1                           — Lie group SO(C)
          • ‖A − I‖_F = O(ε)                    — close to identity for small ε
          • A has no requires_grad               — no gradient through sampling

        Gradients flow through X only (correct augmentation semantics):
        the downstream backbone receives dL/dX' and the chain rule gives
        dL/dX = Aᵀ · (dL/dX') since A is detached.

        One A is sampled per batch (not per sample), modelling the assumption
        that all windows in a batch share the same electrode placement
        (consistent within a subject's recording session).

        Args:
            X: (B, C, T) — training batch, on the correct device.

        Returns:
            X_mix: (B, C, T) — channel-mixed batch, same device and dtype.
        """
        C = X.shape[1]

        # Construct orthogonal matrix under no_grad so A has no gradient
        with torch.no_grad():
            R = torch.randn(C, C, device=X.device, dtype=X.dtype)
            # Skew-symmetric: S = ε · (R − Rᵀ) / 2  →  Sᵀ = −S
            S = self.mix_epsilon * (R - R.t()) / 2.0
            # Matrix exponential of skew-symmetric ∈ SO(C):  AᵀA = I
            A = torch.matrix_exp(S)  # (C, C)

        # Channel mixing: X'[b, :, t] = A @ X[b, :, t]  for all (b, t)
        # einsum is cleaner than batched matmul here and avoids reshape
        return torch.einsum("ij,bjt->bit", A, X)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  Augmentation active ONLY during model.train().

        The self.training flag is set externally by model.train() / model.eval()
        calls in the trainer.  During validation and test inference the
        trainer calls model.eval() before iterating batches, ensuring that
        _ortho_mix() is never applied to val or test data.

        LOSO safety:
          • self.training == False at inference → no augmentation applied.
          • mix_prob gate → not every training batch is augmented either,
            avoiding over-regularisation.

        Args:
            x: (B, C_emg, T) — channels-first EMG windows.

        Returns:
            logits: (B, num_classes)
        """
        # Augmentation guard: gated by training mode AND random draw
        if self.training and torch.rand(1, device=x.device).item() < self.mix_prob:
            x = self._ortho_mix(x)

        return self.backbone(x)

    # ── utility ───────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Trainable parameter count.

        Mixing module is parameter-free — count equals backbone only.
        Same as ECAPATDNNEmg.count_parameters().
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
