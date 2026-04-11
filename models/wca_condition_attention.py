"""
WCA-NARX-inspired Working-Condition-Aware Attention for EMG Gesture Recognition

Source: Wang et al. (EAAI 2024) — WCA-NARX

Idea:
  Attention mechanism conditioned on "working conditions" — unsupervised
  per-window signal characteristics (RMS, zero-crossing rate, spectral centroid
  per channel).  The model first computes a non-parametric condition vector from
  the raw EMG window, then uses it to modulate both channel and temporal
  attention weights inside a CNN-GRU backbone.

Relation to prior experiments:
  exp_23 (SE CNN-GRU)         — channel attention without condition context
  exp_28 (FiLM Subject-Adaptive) — condition on subject embeddings (breaks LOSO)
  exp_72 (MoE Dynamic Routing)   — routing by input dynamics, no explicit
                                    condition-modulated attention

What's new:
  Explicit "condition vector" computed from unsupervised EMG characteristics
  (RMS, ZCR, spectral centroid per channel) used to modulate attention weights
  through a lightweight conditioning MLP.  Strictly LOSO-correct: condition
  is computed from the input window itself, no subject IDs, no test-subject
  statistics.

Architecture:
  1. ConditionExtractor     — non-parametric: (N,C,T) → (N,3C) [RMS, ZCR, SC]
  2. CNN backbone           — (N,C,T) → (N,F,T')
  3. ConditionedChannelAttn — SE-style channel attention whose squeeze-excite
                              MLP receives the condition vector as side input
  4. BiGRU                  — (N,T',F) → (N,T',2H)
  5. ConditionedTemporalAttn — temporal attention whose query is conditioned
                              on the condition vector
  6. Classifier             — (N,2H) → (N,num_classes)

LOSO compliance:
  - ConditionExtractor has NO learnable parameters (deterministic signal metrics)
  - Condition vectors are computed per-window from raw input — no cross-sample
    or cross-subject statistics
  - All learnable parameters (CNN, GRU, attention MLPs, classifier) are trained
    jointly on all training subjects — standard LOSO practice
  - BatchNorm in backbone uses running stats from training subjects only
  - No subject ID, subject embedding, or test-time adaptation anywhere

Compatible with trainer model registry interface:
    __init__(in_channels, num_classes, dropout=0.3)
    forward(x) -> (B, num_classes) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Non-parametric condition extractor
# ---------------------------------------------------------------------------

class ConditionExtractor(nn.Module):
    """
    Non-parametric extractor of per-window "working condition" features.

    Computes three unsupervised signal descriptors per EMG channel:

      1. RMS (Root Mean Square) per channel:
         sqrt(mean(x²))
         Captures overall signal amplitude / contraction intensity.

      2. Zero-Crossing Rate (ZCR) per channel:
         Fraction of consecutive samples with sign change.
         Proxy for dominant frequency content and muscle activation type.

      3. Spectral Centroid per channel:
         Weighted mean of FFT frequency bins.
         Captures the "center of mass" of the power spectrum — higher values
         indicate more high-frequency content (fast-twitch fibres, tremor).

    All features are deterministic functions of the input window.
    No learnable parameters → no possibility of data leakage.

    Input:  (N, C, T)
    Output: (N, 3·C)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T = x.shape
        feats = []

        # ---- 1. RMS per channel -----------------------------------------------
        rms = torch.sqrt((x ** 2).mean(dim=-1) + 1e-8)  # (N, C)
        feats.append(rms)

        # ---- 2. Zero-Crossing Rate per channel --------------------------------
        sign_changes = (x[:, :, 1:].sign() != x[:, :, :-1].sign()).float()
        zcr = sign_changes.mean(dim=-1)  # (N, C)
        feats.append(zcr)

        # ---- 3. Spectral Centroid per channel ----------------------------------
        # Use rfft for real-valued signals
        # x: (N, C, T) → fft over last dim
        X_fft = torch.fft.rfft(x, dim=-1)  # (N, C, T//2+1) complex
        magnitude = X_fft.abs()  # (N, C, T//2+1)

        num_bins = magnitude.shape[-1]
        # Frequency bin indices normalized to [0, 1]
        freq_bins = torch.linspace(0, 1, num_bins, device=x.device, dtype=x.dtype)
        # freq_bins: (num_bins,)

        # Weighted mean of frequency bins
        # magnitude: (N, C, num_bins), freq_bins: (num_bins,)
        total_energy = magnitude.sum(dim=-1, keepdim=True) + 1e-8  # (N, C, 1)
        spectral_centroid = (magnitude * freq_bins).sum(dim=-1) / total_energy.squeeze(-1)
        # spectral_centroid: (N, C)
        feats.append(spectral_centroid)

        return torch.cat(feats, dim=-1)  # (N, 3C)


# ---------------------------------------------------------------------------
# Conditioned channel attention (SE-style + condition side-input)
# ---------------------------------------------------------------------------

class ConditionedChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention augmented with a condition vector.

    Standard SE block:
      squeeze: GAP → (N, F)
      excite:  FC→ReLU→FC→Sigmoid → (N, F)  channel weights

    This module concatenates the condition vector to the squeezed representation
    before the excitation MLP, allowing the gating to depend on the signal's
    working conditions (amplitude level, frequency content, etc.).

    Args:
        feature_channels: number of feature channels F from CNN backbone
        condition_dim:    dimension of condition vector (3·C)
        reduction:        SE reduction ratio (default 8)

    Input:
        features: (N, F, T')  — CNN backbone output
        condition: (N, D)     — condition vector from ConditionExtractor

    Output:
        (N, F, T') — channel-reweighted features
    """

    def __init__(self, feature_channels: int, condition_dim: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(feature_channels // reduction, 16)
        # Input to excitation: squeezed features (F) + condition (D)
        self.excite = nn.Sequential(
            nn.Linear(feature_channels + condition_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, feature_channels),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # features: (N, F, T'), condition: (N, D)
        squeezed = features.mean(dim=-1)  # (N, F) — global average pooling
        combined = torch.cat([squeezed, condition], dim=-1)  # (N, F+D)
        gates = self.excite(combined).unsqueeze(-1)  # (N, F, 1)
        return features * gates  # broadcast: (N, F, T')


# ---------------------------------------------------------------------------
# Conditioned temporal attention
# ---------------------------------------------------------------------------

class ConditionedTemporalAttention(nn.Module):
    """
    Temporal attention over GRU hidden states, conditioned on the working
    condition vector.

    Standard temporal attention:
      score_t = v^T · tanh(W·h_t + b)
      weights = softmax(scores)
      output  = sum(weights * h)

    Conditioned version:
      score_t = v^T · tanh(W_h·h_t + W_c·condition + b)

    The condition vector shifts the attention query, biasing the model to
    focus on different temporal positions depending on the signal's
    characteristics (e.g., focusing on onset for fast activations,
    or averaging for sustained contractions).

    Args:
        hidden_dim:    GRU output dim (2*H for bidirectional)
        condition_dim: dimension of condition vector (3·C)

    Input:
        gru_output: (N, T', 2H)
        condition:  (N, D)

    Output:
        context: (N, 2H) — attention-weighted temporal summary
    """

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c = nn.Linear(condition_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_output: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # gru_output: (N, T', 2H), condition: (N, D)
        # Project hidden states: (N, T', 2H) → (N, T', 2H)
        h_proj = self.W_h(gru_output)
        # Project condition and broadcast: (N, D) → (N, 2H) → (N, 1, 2H)
        c_proj = self.W_c(condition).unsqueeze(1)
        # Combined attention energy
        energy = torch.tanh(h_proj + c_proj + self.bias)  # (N, T', 2H)
        scores = self.v(energy).squeeze(-1)  # (N, T')
        weights = F.softmax(scores, dim=-1)  # (N, T')
        # Weighted sum
        context = (gru_output * weights.unsqueeze(-1)).sum(dim=1)  # (N, 2H)
        return context


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class WCAConditionAttention(nn.Module):
    """
    Working-Condition-Aware CNN-GRU with conditioned channel and temporal
    attention for EMG gesture recognition.

    Architecture:
      Input (N, C, T)
        |
        ├─→ ConditionExtractor → (N, 3C)  [non-parametric]
        |
        └─→ CNN backbone → (N, F, T')
              |
              └─→ ConditionedChannelAttn(cond) → (N, F, T')
                    |
                    └─→ Permute → (N, T', F)
                          |
                          └─→ BiGRU → (N, T', 2H)
                                |
                                └─→ ConditionedTemporalAttn(cond) → (N, 2H)
                                      |
                                      └─→ Classifier → (N, num_classes)

    Args:
        in_channels:       number of EMG channels (e.g. 8)
        num_classes:       number of gesture classes
        dropout:           dropout rate (default 0.3)
        backbone_channels: CNN backbone output channels (default 128)
        gru_hidden:        GRU hidden size per direction (default 128)
        gru_layers:        number of GRU layers (default 2)
        se_reduction:      SE reduction ratio for channel attention (default 8)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        backbone_channels: int = 128,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        condition_dim = 3 * in_channels  # RMS + ZCR + spectral centroid

        # ---- Non-parametric condition extraction ----------------------------
        self.condition_extractor = ConditionExtractor()

        # ---- Condition projection (LayerNorm for LOSO safety) ---------------
        # Small MLP to project raw condition features into a learned space.
        # Uses LayerNorm (per-sample normalization) to avoid leaking
        # cross-sample / cross-subject statistics through BatchNorm.
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ---- CNN backbone ---------------------------------------------------
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, backbone_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(backbone_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
        )

        # ---- Conditioned channel attention ----------------------------------
        self.channel_attn = ConditionedChannelAttention(
            feature_channels=backbone_channels,
            condition_dim=condition_dim,
            reduction=se_reduction,
        )

        # ---- BiGRU for temporal modelling -----------------------------------
        self.gru = nn.GRU(
            input_size=backbone_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        gru_output_dim = gru_hidden * 2  # bidirectional

        # ---- Conditioned temporal attention ---------------------------------
        self.temporal_attn = ConditionedTemporalAttention(
            hidden_dim=gru_output_dim,
            condition_dim=condition_dim,
        )

        # ---- Classifier -----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T) — standardised raw EMG windows

        Returns:
            (N, num_classes) — classification logits
        """
        # -- 1. Extract condition vector (non-parametric) --------------------
        raw_cond = self.condition_extractor(x)  # (N, 3C)
        condition = self.condition_proj(raw_cond)  # (N, 3C) projected

        # -- 2. CNN backbone -------------------------------------------------
        feat = self.backbone(x)  # (N, F, T')

        # -- 3. Conditioned channel attention --------------------------------
        feat = self.channel_attn(feat, condition)  # (N, F, T')

        # -- 4. BiGRU -------------------------------------------------------
        # Reshape for GRU: (N, F, T') → (N, T', F)
        feat = feat.permute(0, 2, 1)
        gru_out, _ = self.gru(feat)  # (N, T', 2H)

        # -- 5. Conditioned temporal attention -------------------------------
        context = self.temporal_attn(gru_out, condition)  # (N, 2H)

        # -- 6. Classify ----------------------------------------------------
        logits = self.classifier(context)  # (N, num_classes)

        return logits
