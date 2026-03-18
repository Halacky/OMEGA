"""
Domain-Specific Batch Normalization CNN-GRU-Attention (DSBN-CNN-GRU).

Architecture hypothesis: Combining per-sample InstanceNorm (removes amplitude
bias from electrode placement) with Domain-Specific BatchNorm (separate affine
scale/bias per subject cluster) preserves class-discriminative temporal patterns
while absorbing inter-subject distribution shift.

Design choices:
- InstanceNorm1d at the input: decouples gesture shape from absolute amplitude
- DomainBN1d uses SHARED running statistics for gradient stability during training
  with SEPARATE learnable affine (weight, bias) per domain cluster
- BiGRU + Attention: unchanged from baseline CNNGRUWithAttention (exp_1 best)
- During inference: caller must pass domain_ids = assigned cluster for test subject

References:
- DSBN: Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation" (ICLR 2018)
- IBN-Net: Pan et al., "Two at Once: Enhancing Learning and Generalization
  Capacities via IBN-Net" (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainBN1d(nn.Module):
    """
    Domain-Specific Batch Normalization for 1-D feature maps.

    Uses a SINGLE shared BatchNorm (affine=False) for stable running statistics,
    then applies PER-DOMAIN learnable scale (weight) and shift (bias).

    This is more stable than K separate BatchNorm layers when per-domain batch
    sizes are small (e.g., 4 training subjects → only ~25% of batch per domain).

    Works for both (B, C) and (B, C, T) inputs.
    """

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains

        # Shared normalization: accumulates running mean/var over all domains
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=False)

        # Per-domain affine parameters: shape (num_domains, num_features)
        self.weight = nn.Parameter(torch.ones(num_domains, num_features))
        self.bias = nn.Parameter(torch.zeros(num_domains, num_features))

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (B, C) or (B, C, T)
            domain_ids: (B,) long tensor; values in [0, num_domains)

        Returns:
            Normalized tensor with domain-specific affine applied, same shape as x.
        """
        x_norm = self.bn(x)  # shared normalization, same shape as x

        w = self.weight[domain_ids]  # (B, C)
        b = self.bias[domain_ids]    # (B, C)

        if x.dim() == 3:             # (B, C, T) → broadcast over T
            w = w.unsqueeze(-1)      # (B, C, 1)
            b = b.unsqueeze(-1)      # (B, C, 1)

        return x_norm * w + b

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"num_domains={self.num_domains}"
        )


class DSBNCNNGRUAttention(nn.Module):
    """
    CNN-GRU-Attention with Instance Normalization input + Domain-Specific BN.

    Forward pass (training):
        x (B, C, T), domain_ids (B,) → logits (B, num_classes)

    Forward pass (inference on test subject):
        x (B, C, T), domain_ids = torch.full((B,), nearest_cluster) → logits

    Architecture:
        InstanceNorm1d  → removes per-sample amplitude bias (electrode placement)
        Conv1d + DomainBN1d × L → local feature extraction with domain-aware BN
        BiGRU                   → temporal sequence modeling
        Attention               → focus on discriminative time steps
        Linear                  → classification
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_domains: int = 2,
        dropout: float = 0.3,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64]

        self.in_channels = in_channels
        self.num_domains = num_domains
        self.gru_hidden = gru_hidden

        # ── Input normalization ──────────────────────────────────────────────
        # affine=True: model can learn to undo normalization if needed
        # This normalizes each (sample, channel) pair over T independently
        self.input_norm = nn.InstanceNorm1d(in_channels, affine=True)

        # ── CNN with Domain-Specific BN ──────────────────────────────────────
        self.conv_layers = nn.ModuleList()
        self.domain_bns = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_ch = in_channels
        for out_ch in cnn_channels:
            self.conv_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
            )
            self.domain_bns.append(DomainBN1d(out_ch, num_domains))
            self.pool_layers.append(nn.MaxPool1d(2))
            in_ch = out_ch

        # ── BiGRU ────────────────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ── Attention ────────────────────────────────────────────────────────
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          Raw EMG windows  (B, C, T)
            domain_ids: Cluster index    (B,) long tensor.
                        If None, all samples use domain 0 (single-domain fallback).

        Returns:
            Logits  (B, num_classes)
        """
        B = x.size(0)

        if domain_ids is None:
            domain_ids = torch.zeros(B, dtype=torch.long, device=x.device)

        # 1. Instance normalization: remove per-sample, per-channel amplitude offset
        x = self.input_norm(x)  # (B, C, T) — unchanged shape

        # 2. CNN with domain-specific BN
        for conv, dbn, pool in zip(self.conv_layers, self.domain_bns, self.pool_layers):
            x = conv(x)               # (B, out_ch, T')
            x = dbn(x, domain_ids)    # domain-specific affine, same shape
            x = F.relu(x, inplace=True)
            x = pool(x)               # (B, out_ch, T'//2)

        # 3. BiGRU: (B, C, T') → (B, T', C) → (B, T', gru_hidden*2)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)

        # 4. Attention pooling
        attn_weights = F.softmax(self.attention(gru_out), dim=1)  # (B, T', 1)
        context = (gru_out * attn_weights).sum(dim=1)             # (B, gru_hidden*2)

        # 5. Classify
        context = self.dropout(context)
        return self.classifier(context)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"num_domains={self.num_domains}, "
            f"gru_hidden={self.gru_hidden}"
        )
