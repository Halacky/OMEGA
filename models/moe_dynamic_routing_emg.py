"""
MoE v2: Dynamic Regime Routing for EMG Gesture Recognition

Hypothesis: EMG signals exhibit distinct motion modes (fast activation, sustained
contraction, onset/offset transitions) that cut across subject boundaries.
Routing EMG windows to specialized experts based on physics-inspired signal
dynamics — rather than subject-style features — should yield more generalizable
routing and improve cross-subject transfer.

Core idea:
  - Previous MoE (exp_27) routes via amplitude/spectral features that implicitly
    correlate with subject identity → poor LOSO transfer.
  - This model routes via TKEO energy, envelope slope, kurtosis, ZCR — features
    that describe the *motion regime* of a window, not *who* produced it.

Architecture:
  1. DynamicFeatureExtractor  — non-parametric physics features (N,C,T)→(N,6C)
  2. DynamicRoutingNetwork    — LayerNorm MLP: (N,6C)→(N,K) soft gates
  3. Shared CNN backbone      — (N,C,T)→(N,F,T')
  4. K TDNNExpertBlocks       — each with a different dilation (temporal scale)
  5. Gated mixture            — (N,K,F')×(N,K,1)→(N,F') weighted sum
  6. Classifier               — (N,F')→(N,num_classes)

Auxiliary losses (training only, stored in self._aux_loss for trainer hook):
  - Entropy regularisation: maximize per-sample routing entropy → soft routing
  - Load balancing:         mean batch gate → uniform across experts

LOSO compliance:
  - DynamicFeatureExtractor has NO learnable parameters (purely deterministic).
  - DynamicRoutingNetwork uses LayerNorm (per-sample normalisation, no cross-
    sample statistics → cannot encode subject identity through normalisation).
  - BatchNorm in backbone/experts uses running stats accumulated over ALL training
    subjects jointly → standard LOSO practice, no test-subject leakage.
  - No subject ID, subject embedding, or subject-specific branch anywhere.

Compatible with trainer model registry interface:
    __init__(in_channels, num_classes, dropout=0.3)
    forward(x) -> (B, num_classes) logits
    self._aux_loss set during training forward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dynamic feature extractor (non-parametric)
# ---------------------------------------------------------------------------

class DynamicFeatureExtractor(nn.Module):
    """
    Non-parametric extractor of physics-inspired dynamic summary features.

    Computes four types of signal-dynamic descriptors per EMG channel:

      1. TKEO energy (mean, std):
         tkeo[t] = x[t]² - x[t-1]·x[t+1]
         Captures instantaneous muscular activation energy.
         High mean  → sustained strong contraction.
         High std   → variable / transitioning activation.

      2. Envelope slope (mean |slope|, std of slope):
         Slope of the smoothed absolute-value envelope.
         High mean  → fast activation or deactivation.
         Low mean   → steady-state contraction.

      3. Kurtosis per channel:
         E[(x-μ)⁴] / E[(x-μ)²]²
         High → impulsive / spike-like contractions.
         ≈3   → sustained Gaussian-like contraction.

      4. Zero-crossing rate per channel:
         Fraction of samples where sign changes.
         High → high-frequency oscillations / tremor.
         Low  → smooth sustained movement.

    All 6 features per channel are concatenated → output shape (N, 6·C).
    No learnable parameters.

    Args:
        envelope_smooth_k (int): kernel size for moving-average envelope
            smoothing. Default 31 ≈ 15 ms at 2 kHz.

    Input:  (N, C, T)
    Output: (N, 6·C)
    """

    def __init__(self, envelope_smooth_k: int = 31):
        super().__init__()
        self._ks = envelope_smooth_k  # plain attribute, not nn.Parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,C,T) → (N,6C)
        N, C, T = x.shape
        feats: List[torch.Tensor] = []

        # ---- 1. TKEO energy ------------------------------------------------
        # x_mid: x[t], x_pre: x[t-1], x_nxt: x[t+1]
        x_mid = x[:, :, 1:-1]   # (N, C, T-2)
        x_pre = x[:, :, :-2]    # (N, C, T-2)
        x_nxt = x[:, :, 2:]     # (N, C, T-2)
        tkeo = (x_mid ** 2 - x_pre * x_nxt).clamp(min=0.0)  # (N, C, T-2)
        feats.append(tkeo.mean(dim=-1))   # (N, C) TKEO mean
        feats.append(tkeo.std(dim=-1))    # (N, C) TKEO std  (≥0)

        # ---- 2. Envelope slope ---------------------------------------------
        # Reshape to (N·C, 1, T) for avg_pool1d
        env = x.abs().reshape(N * C, 1, T)
        # "same" padding: output length = T for odd kernel, stride=1
        # With k=31, pad=15: out_len = T + 2·15 - 31 + 1 = T  ✓
        smooth = F.avg_pool1d(
            env,
            kernel_size=self._ks,
            stride=1,
            padding=self._ks // 2,
        )  # (N·C, 1, T)
        smooth = smooth.reshape(N, C, T)
        slope = smooth[:, :, 1:] - smooth[:, :, :-1]   # (N, C, T-1)
        feats.append(slope.abs().mean(dim=-1))           # (N, C) mean |slope|
        feats.append(slope.std(dim=-1))                  # (N, C) slope variability

        # ---- 3. Kurtosis ---------------------------------------------------
        mu = x.mean(dim=-1, keepdim=True)               # (N, C, 1)
        xc = x - mu                                      # (N, C, T) centred
        var = (xc ** 2).mean(dim=-1)                    # (N, C)
        kurt = (xc ** 4).mean(dim=-1) / (var ** 2 + 1e-8)  # (N, C)
        feats.append(kurt)

        # ---- 4. Zero-crossing rate -----------------------------------------
        sign_diff = (x[:, :, 1:].sign() != x[:, :, :-1].sign()).float()
        zcr = sign_diff.mean(dim=-1)                    # (N, C)
        feats.append(zcr)

        return torch.cat(feats, dim=-1)  # (N, 6·C)


# ---------------------------------------------------------------------------
# Routing network
# ---------------------------------------------------------------------------

class DynamicRoutingNetwork(nn.Module):
    """
    Lightweight MLP that maps dynamic summary features → soft expert weights.

    Uses LayerNorm (per-sample, feature-dimension normalisation) deliberately:
      - LayerNorm does NOT share statistics across samples in a batch.
      - Therefore it cannot encode inter-subject differences through running
        statistics — no implicit subject information leak via normalisation.
      - Contrast: BatchNorm1d shares batch mean/var, which can correlate with
        subject identity when batches contain subject-specific sub-distributions.

    Learnable affine parameters in LayerNorm are shared across all subjects
    (trained jointly on all training subjects in each LOSO fold) — LOSO-safe.

    Args:
        in_dim      (int): input dimension (6·C)
        num_experts (int): number of experts K
        hidden_dim  (int): MLP hidden dimension

    Input:  (N, in_dim)
    Output: (N, K) — soft gating weights summing to 1
    """

    def __init__(self, in_dim: int, num_experts: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,D)→(N,K)
        return F.softmax(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# TDNN Expert block
# ---------------------------------------------------------------------------

class TDNNExpertBlock(nn.Module):
    """
    Time-Delay Neural Network (TDNN) expert block with residual connection.

    Each expert uses a different dilation, giving a different temporal
    receptive field:
      dilation=1 → RF ≈ 5 samples  (~2.5 ms)  for fast transient activation
      dilation=2 → RF ≈ 9 samples  (~4.5 ms)  for short transitions
      dilation=4 → RF ≈ 17 samples (~8.5 ms)  for medium-scale dynamics
      dilation=8 → RF ≈ 33 samples (~16.5 ms) for sustained contractions

    Uses "same" padding (no causal constraint) since windows are processed
    offline — no causal requirement in this setting.

    Input:  (N, in_channels, T')
    Output: (N, out_channels, T')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2  # "same" padding for odd k
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=pad,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,F,T')→(N,F',T')
        return self.act(self.conv_block(x) + self.skip(x))


# ---------------------------------------------------------------------------
# Full MoE model
# ---------------------------------------------------------------------------

class MoEDynamicRoutingEMG(nn.Module):
    """
    Mixture-of-Experts EMG classifier with dynamic-regime routing.

    See module docstring for full description of the architecture and
    LOSO compliance rationale.

    Args:
        in_channels       (int):   number of EMG channels (e.g. 8)
        num_classes       (int):   number of gesture classes
        dropout           (float): dropout rate (applied in backbone + classifier)
        num_experts       (int):   number of expert blocks K (default 4)
        backbone_channels (int):   output channels of the shared CNN backbone
        expert_channels   (int):   output channels of each TDNN expert
        entropy_weight    (float): weight for entropy regularisation aux loss
        balance_weight    (float): weight for load-balancing aux loss
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        num_experts: int = 4,
        backbone_channels: int = 128,
        expert_channels: int = 128,
        entropy_weight: float = 0.01,
        balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.entropy_weight = entropy_weight
        self.balance_weight = balance_weight

        # ---- Routing components --------------------------------------------
        self.dynamic_extractor = DynamicFeatureExtractor()
        routing_in_dim = 6 * in_channels
        self.routing_net = DynamicRoutingNetwork(
            in_dim=routing_in_dim,
            num_experts=num_experts,
            hidden_dim=max(64, routing_in_dim),
        )

        # ---- Shared CNN backbone -------------------------------------------
        # Deliberately mirrors the CNN backbone in CNNGRUWithAttention / exp_27
        # to isolate the routing strategy as the only variable.
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, backbone_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(backbone_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2),
        )

        # ---- Expert TDNN blocks (varying dilation = varying temporal scale) -
        # 4 experts cover dilation = 1,2,4,8; for K≠4 dilations cycle
        _dilations = [1, 2, 4, 8]
        self.experts = nn.ModuleList([
            TDNNExpertBlock(
                in_channels=backbone_channels,
                out_channels=expert_channels,
                dilation=_dilations[i % len(_dilations)],
            )
            for i in range(num_experts)
        ])

        # ---- Temporal pooling + classifier ---------------------------------
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(expert_channels, num_classes),
        )

        # ---- Internal state ------------------------------------------------
        # _aux_loss: picked up by trainer's loss hook (line ~595 in trainer.py)
        # _last_gates: stored for post-hoc routing analysis / visualisation
        self._aux_loss: Optional[torch.Tensor] = None
        self._last_gates: Optional[torch.Tensor] = None  # CPU tensor, (N,K)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T) — standardised raw EMG windows (trainer handles std)

        Returns:
            (N, num_classes) — classification logits
        """
        N = x.size(0)

        # -- 1. Physics-based dynamic features for routing --
        dyn_feats = self.dynamic_extractor(x)   # (N, 6C)
        gates = self.routing_net(dyn_feats)      # (N, K) soft, sums to 1

        # Store gates on CPU for external analysis (no gradient stored)
        self._last_gates = gates.detach().cpu()

        # -- 2. Shared backbone --
        feat = self.backbone(x)                  # (N, F, T')

        # -- 3. Expert processing: each expert processes the SAME backbone
        #        output but with a different temporal receptive field --
        expert_pooled = torch.stack(
            [self.pool(expert(feat)).squeeze(-1) for expert in self.experts],
            dim=1,
        )  # (N, K, F')

        # -- 4. Gated mixture --
        mixture = (expert_pooled * gates.unsqueeze(-1)).sum(dim=1)  # (N, F')

        # -- 5. Classify --
        logits = self.classifier(mixture)        # (N, num_classes)

        # -- 6. Auxiliary losses (training only) --
        if self.training:
            # Entropy regularisation:
            #   H = -Σ g_i log(g_i+ε) per sample; maximise → minimise -mean(H)
            entropy = -(gates * (gates + 1e-9).log()).sum(dim=-1)  # (N,)
            entropy_loss = -entropy.mean()

            # Load balancing:
            #   Penalise deviation of batch-mean gates from uniform [1/K, …, 1/K]
            mean_gates = gates.mean(dim=0)                          # (K,)
            uniform = torch.full_like(mean_gates, 1.0 / self.num_experts)
            balance_loss = F.mse_loss(mean_gates, uniform)

            self._aux_loss = (
                self.entropy_weight * entropy_loss
                + self.balance_weight * balance_loss
            )
        else:
            self._aux_loss = None

        return logits
