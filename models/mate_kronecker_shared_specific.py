"""
MATE-inspired Shared-Specific EMG Model with Kronecker Attention.

Hypothesis: Shared-Specific Mode Interaction without orthogonality.

Key departure from exp_31/57/59 (which all force orthogonality via distance
correlation or MI penalties between content and style):
  - Shared and specific representations are ALLOWED to be correlated (MATE insight).
  - Subject-invariance enforced ONLY on z_shared via adversarial gradient reversal.
  - NO MI/distance-correlation/Barlow-Twins between z_shared and z_specific.

Architecture
────────────
1. Per-channel CNN (×C_emg): lightweight independent temporal encoder per channel.
     ChannelEncoder(1 → ch_enc_dim)  — Conv1d stack + global avg pool.

2. Shared ECAPA-TDNN backbone: processes the full multi-channel signal.
     (B, C_emg, T) → h ∈ (B, embedding_dim)

3. Shared Prior Network: h → z_shared ∈ (B, shared_dim)
     Captures gesture-wide patterns; adversarially regularized to be
     subject-invariant (GRL → subject adversary).

4. Specific Prior Networks (×C_emg): f_k → z_specific_k ∈ (B, specific_dim)
     Each network processes the per-channel CNN feature for channel k.
     Captures channel-k-specific patterns (electrode placement, impedance).
     NOT forced to be orthogonal to z_shared.

5. Kronecker Attention: Z_specific ∈ (B, C_emg, specific_dim) → Z_attended
     Factorises attention as A_ch ⊗ A_feat (inter-channel × intra-feature).
     Complexity: O(K² + D²) vs O((K·D)²) for full attention.
     X' = A_ch @ X @ A_feat^T  (Kronecker product application in matrix form).

6. Gesture Classifier: concat(z_shared, flatten(Z_attended)) → G logits.

7. Subject Adversary (training only, NOT at inference):
     gradient_reversal(z_shared) → subject logits.
     Forces z_shared to be unable to predict subject → subject-invariant.

LOSO Safety
───────────
- Per-channel standardization computed from training data only (stored in
  self.mean_c, self.std_c). Applied with training stats to val/test.
- BatchNorm running stats accumulated only during training. model.eval()
  freezes them — no subject-specific batch statistics at test time.
- Inference path (return_all=False): uses only gesture classifier. Subject
  labels, GRL, and adversary are completely excluded at test time.
- Per-channel CNNs are trained per fold from scratch — no cross-fold leakage.

Input format:  (B, C_emg, T)  — channels-first, compatible with PyTorch Conv1d.
Output format: (B, num_gestures) logits       when return_all=False (inference)
               dict with all branches          when return_all=True  (training)
"""

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from models.ecapa_tdnn_emg import SERes2NetBlock, AttentiveStatisticsPooling


# ─────────────────────────── Gradient Reversal ──────────────────────────────


class _GradRevFn(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward pass: identity.
    Backward pass: negates and scales the gradient by alpha.

    This allows the shared encoder to minimise subject prediction accuracy
    while the subject adversary maximises it — a minimax game.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (alpha,) = ctx.saved_tensors
        # Return -alpha * grad for x; None for alpha (not differentiable w.r.t. scalar)
        return -alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with scaling factor alpha."""
    a = torch.tensor(alpha, dtype=x.dtype, device=x.device)
    return _GradRevFn.apply(x, a)


# ───────────────────────── Kronecker Attention ───────────────────────────────


class KroneckerAttention(nn.Module):
    """
    Kronecker-factored attention for 2D token grids X ∈ ℝ^{B × K × D}.

    Standard full attention over K·D tokens has complexity O((K·D)²).
    Kronecker factorisation splits this into two smaller attention matrices:

        A_ch   ∈ ℝ^{B × K × K}   (inter-channel attention)
        A_feat ∈ ℝ^{B × D × D}   (intra-feature attention)

    and applies them jointly as the Kronecker product:

        X' = A_ch @ X @ A_feat^T

    Complexity: O(K² · D + K · D²) — much cheaper than O((K·D)²).
    For K=8, D=32: 2048 + 8192 = 10 240 vs 65 536.

    Inspired by "Hi-TS: High-Resolution Time Series Transformer" (2024).

    Args:
        K:   Number of channels (e.g. 8 for NinaPro DB2 EMG).
        D:   Feature dimension per channel (specific_dim).
        d_k: Attention projection dimension for computing Q/K weights.
    """

    def __init__(self, K: int, D: int, d_k: int = 16):
        super().__init__()
        self.K = K
        self.D = D
        self.scale_ch = math.sqrt(d_k)
        self.scale_feat = math.sqrt(d_k)

        # Inter-channel: project each channel's D-dim feature vector → d_k
        # Applied to X ∈ (B, K, D): nn.Linear(D, d_k) operates on last dim
        self.W_q_ch = nn.Linear(D, d_k, bias=False)
        self.W_k_ch = nn.Linear(D, d_k, bias=False)

        # Intra-feature: project each feature dim's K-channel vector → d_k
        # Applied to X.T ∈ (B, D, K): nn.Linear(K, d_k) operates on last dim
        self.W_q_feat = nn.Linear(K, d_k, bias=False)
        self.W_k_feat = nn.Linear(K, d_k, bias=False)

        # Output projection (applied per-channel) and residual normalisation
        self.out_proj = nn.Linear(D, D, bias=False)
        self.norm = nn.LayerNorm([K, D])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, K, D) — batch × channels × features per channel.

        Returns:
            out: (B, K, D) — Kronecker-attended output.
        """
        residual = X  # (B, K, D)

        # ── Inter-channel attention ──────────────────────────────────────
        # Q_ch, K_ch: (B, K, D) → (B, K, d_k)
        Q_ch = self.W_q_ch(X)
        K_ch = self.W_k_ch(X)
        # A_ch: (B, K, K) — each channel attends to all other channels
        A_ch = torch.softmax(
            Q_ch @ K_ch.transpose(-1, -2) / self.scale_ch, dim=-1
        )

        # ── Intra-feature attention ──────────────────────────────────────
        # Transpose to (B, D, K) so each feature dim's K-channel vector
        # becomes the token representation.
        X_t = X.transpose(-1, -2)          # (B, D, K)
        Q_feat = self.W_q_feat(X_t)        # (B, D, d_k)
        K_feat = self.W_k_feat(X_t)        # (B, D, d_k)
        # A_feat: (B, D, D) — each feature dim attends to all feature dims
        A_feat = torch.softmax(
            Q_feat @ K_feat.transpose(-1, -2) / self.scale_feat, dim=-1
        )

        # ── Kronecker application: X' = A_ch @ X @ A_feat^T ─────────────
        # This equals (A_ch ⊗ A_feat) applied to vec(X^T) in vectorised form.
        # Step 1: (B,K,K) @ (B,K,D) = (B,K,D)  — inter-channel mixing
        # Step 2: (B,K,D) @ (B,D,D) = (B,K,D)  — intra-feature mixing
        out = A_ch @ X @ A_feat.transpose(-1, -2)  # (B, K, D)

        # Output projection applied independently to each channel's D-dim vector
        out = self.out_proj(out)            # (B, K, D)

        # Residual connection + layer norm over (K, D)
        return self.norm(out + residual)    # (B, K, D)


# ──────────────────────────── Channel Encoder ────────────────────────────────


class ChannelEncoder(nn.Module):
    """
    Lightweight per-channel temporal CNN.

    Processes a single EMG channel (B, 1, T) independently to extract
    channel-specific temporal features. Used to build Z_specific.

    Architecture:
        Conv1d(1 → 32, k=7) → BN → ReLU
        Conv1d(32 → out_dim, k=5) → BN → ReLU
        AdaptiveAvgPool1d(1)  [global temporal pooling]

    Args:
        out_dim: Feature dimension per channel (specific_dim equivalent).
    """

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, out_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),   # (B, out_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) → (B, out_dim)"""
        return self.net(x).squeeze(-1)


# ───────────────────────────── Main Model ────────────────────────────────────


class MATEKroneckerEMG(nn.Module):
    """
    MATE-inspired Shared-Specific EMG classifier with Kronecker Attention.

    Core design principle (from MATE, NeurIPS 2023):
      Enforcing strict orthogonality between shared and specific representations
      is too strong an assumption — they may legitimately share information.
      Instead, only the shared representation is subject-adversarially
      regularised. Specific representations are free to be correlated with
      the shared one.

    This addresses the failure mode observed in exp_31 (35.28%), exp_57
    (33.37%), exp_59 (32.94%): adding stronger orthogonality constraints
    (GroupDRO + prototype push) degraded rather than improved performance.

    LOSO safety:
      - ChannelEncoder BatchNorm: training stats frozen via model.eval().
      - ECAPA backbone BatchNorm: training stats frozen via model.eval().
      - Subject adversary: excluded from forward pass at inference
        (return_all=False). No subject labels required at test time.
      - Gradient reversal only reverses gradients during training backprop.
        At eval, model.eval() + torch.no_grad() makes it irrelevant.

    Args:
        in_channels:    Number of EMG channels (8 for NinaPro DB2).
        num_gestures:   Number of gesture classes.
        num_subjects:   Number of training subjects (adversary head size).
        ecapa_channels: C — ECAPA internal feature dimension.
        ecapa_scale:    Res2Net scale / sub-groups.
        embedding_dim:  E — ECAPA embedding (ASP → FC) dimension.
        shared_dim:     D_s — shared prior network output dimension.
        specific_dim:   D_p — per-channel specific network output dimension.
        ch_enc_dim:     Intermediate dim of per-channel CNN encoder.
        kron_d_k:       Kronecker attention Q/K projection dimension.
        dilations:      Dilations for ECAPA SE-Res2Net blocks.
        se_reduction:   SE bottleneck reduction factor.
        dropout:        Dropout probability in classifier and adversary heads.
    """

    def __init__(
        self,
        in_channels: int,
        num_gestures: int,
        num_subjects: int,
        ecapa_channels: int = 128,
        ecapa_scale: int = 4,
        embedding_dim: int = 128,
        shared_dim: int = 128,
        specific_dim: int = 32,
        ch_enc_dim: int = 64,
        kron_d_k: int = 16,
        dilations: Optional[List[int]] = None,
        se_reduction: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        if dilations is None:
            dilations = [2, 3, 4]

        self.in_channels = in_channels
        self.num_gestures = num_gestures
        self.num_subjects = num_subjects
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        num_blocks = len(dilations)

        # ── 1. Per-channel encoders (independent, one per EMG channel) ───
        # Each encoder processes its channel independently → channel-specific
        # features that capture electrode-placement and impedance effects.
        self.channel_encoders = nn.ModuleList([
            ChannelEncoder(out_dim=ch_enc_dim)
            for _ in range(in_channels)
        ])

        # ── 2. Shared ECAPA-TDNN backbone ────────────────────────────────
        # Processes the full multi-channel signal to capture global
        # gesture-discriminative patterns across all channels jointly.
        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, ecapa_channels, kernel_size=5, padding=2,
                      bias=False),
            nn.BatchNorm1d(ecapa_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            SERes2NetBlock(
                ecapa_channels,
                kernel_size=3,
                dilation=d,
                scale=ecapa_scale,
                se_reduction=se_reduction,
            )
            for d in dilations
        ])
        mfa_channels = ecapa_channels * num_blocks
        self.mfa_conv = nn.Sequential(
            nn.Conv1d(mfa_channels, mfa_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_channels),
            nn.ReLU(inplace=True),
        )
        self.asp = AttentiveStatisticsPooling(mfa_channels)
        asp_out_dim = mfa_channels * 2
        self.fc_embedding = nn.Sequential(
            nn.Linear(asp_out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── 3. Shared Prior Network ──────────────────────────────────────
        # Maps the global embedding h → z_shared.
        # z_shared is adversarially constrained to be subject-invariant.
        # NOT constrained to be orthogonal to z_specific (MATE key insight).
        self.shared_prior = nn.Sequential(
            nn.Linear(embedding_dim, shared_dim, bias=False),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(inplace=True),
        )

        # ── 4. Specific Prior Networks (one per channel) ─────────────────
        # Each maps the corresponding ChannelEncoder output → z_specific_k.
        # Captures channel-specific variance (electrode layout, impedance).
        # Allowed to be correlated with z_shared (unlike exp_31/57/59).
        self.specific_priors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch_enc_dim, specific_dim, bias=False),
                nn.BatchNorm1d(specific_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(in_channels)
        ])

        # ── 5. Kronecker Attention ────────────────────────────────────────
        # Fuses channel-specific representations Z_specific ∈ (B, K, D_p)
        # via factorised attention: inter-channel ⊗ intra-feature.
        # Output: Z_attended ∈ (B, K, D_p)  → flattened to (B, K·D_p).
        self.kronecker_attn = KroneckerAttention(
            K=in_channels, D=specific_dim, d_k=kron_d_k,
        )

        # ── 6. Gesture Classifier ─────────────────────────────────────────
        # Takes concatenation of shared and attended-specific representations.
        fused_dim = shared_dim + in_channels * specific_dim
        self.gesture_classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim // 2, num_gestures),
        )

        # ── 7. Subject Adversary ──────────────────────────────────────────
        # Applied via GRL to z_shared. Drives z_shared to be unable to
        # predict subject → shared features become subject-invariant.
        # Only used during training (return_all=True). Not present in the
        # inference path (return_all=False).
        self.subject_adversary = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(shared_dim // 2, num_subjects),
        )

        self._init_weights()

    def _init_weights(self):
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

    def _encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run ECAPA-TDNN backbone → shared embedding h.

        Args:
            x: (B, C_in, T)
        Returns:
            h: (B, embedding_dim)
        """
        out = self.init_tdnn(x)              # (B, C, T)
        block_outputs = []
        for block in self.blocks:
            out = block(out)                 # (B, C, T)
            block_outputs.append(out)
        mfa = torch.cat(block_outputs, dim=1)  # (B, 3C, T)
        mfa = self.mfa_conv(mfa)            # (B, 3C, T)
        pooled = self.asp(mfa)              # (B, 6C)
        h = self.fc_embedding(pooled)       # (B, E)
        return h

    def _encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run per-channel encoders independently.

        Args:
            x: (B, C_in, T)
        Returns:
            channel_feats: (B, C_in, ch_enc_dim)
        """
        feats = []
        for k, enc in enumerate(self.channel_encoders):
            f_k = enc(x[:, k:k + 1, :])    # slice: (B, 1, T) → (B, ch_enc_dim)
            feats.append(f_k)
        return torch.stack(feats, dim=1)    # (B, C_in, ch_enc_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
        grl_alpha: float = 1.0,
    ) -> Union[torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            x:          (B, C_in, T) — channels-first EMG windows.
            return_all: False → return only gesture logits (inference, LOSO-safe).
                        True  → return all branches incl. subject adversary (training).
            grl_alpha:  Gradient reversal scale. Annealed from 0→1 during training
                        (DANN schedule). Irrelevant at inference since return_all=False.

        Returns:
            return_all=False (inference):
                gesture_logits: (B, G)

            return_all=True (training):
                dict {
                    "gesture_logits": (B, G),
                    "subject_logits": (B, S),
                    "z_shared":       (B, D_s),
                    "z_specific":     (B, C_in, D_p),
                    "z_fused_flat":   (B, C_in * D_p),
                }
        """
        B = x.size(0)

        # ── Shared ECAPA embedding ────────────────────────────────────────
        h = self._encode_shared(x)                   # (B, E)

        # ── Shared prior network ──────────────────────────────────────────
        z_shared = self.shared_prior(h)              # (B, D_s)

        # ── Per-channel features + specific prior networks ────────────────
        channel_feats = self._encode_channels(x)     # (B, C_in, ch_enc_dim)
        z_specific_list = []
        for k, prior_net in enumerate(self.specific_priors):
            z_k = prior_net(channel_feats[:, k, :]) # (B, D_p)
            z_specific_list.append(z_k)
        Z_specific = torch.stack(z_specific_list, dim=1)  # (B, C_in, D_p)

        # ── Kronecker Attention ───────────────────────────────────────────
        # Fuses per-channel specific representations via factorised attention.
        # No orthogonality loss applied — z_shared and Z_specific may overlap.
        Z_attended = self.kronecker_attn(Z_specific)  # (B, C_in, D_p)
        z_fused_flat = Z_attended.reshape(B, -1)      # (B, C_in * D_p)

        # ── Gesture classification ────────────────────────────────────────
        h_combined = torch.cat([z_shared, z_fused_flat], dim=1)  # (B, D_s + C_in*D_p)
        gesture_logits = self.gesture_classifier(h_combined)     # (B, G)

        if not return_all:
            # Inference path: subject adversary excluded entirely.
            return gesture_logits

        # ── Subject adversary (training only) ─────────────────────────────
        # GRL reverses gradients → encoder learns to make z_shared
        # indistinguishable across subjects (adversarial subject-invariance).
        # This is the ONLY subject-related constraint; no shared/specific
        # orthogonality constraint is applied (MATE key insight).
        z_shared_rev = gradient_reversal(z_shared, alpha=grl_alpha)
        subject_logits = self.subject_adversary(z_shared_rev)    # (B, S)

        return {
            "gesture_logits": gesture_logits,
            "subject_logits": subject_logits,
            "z_shared":       z_shared,
            "z_specific":     Z_specific,
            "z_fused_flat":   z_fused_flat,
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
