"""
Mixture-of-Experts CNN-GRU-Attention model for EMG gesture recognition.

Hypothesis H1: Subject-as-Domain + MoE
Each subject is treated as a separate domain. Instead of enforcing domain invariance (GRL),
we model subject-specific signal styles as a latent factor via learned expert routing.

Architecture:
- Shared CNN backbone extracts low-level temporal features
- Signal-style gating network routes inputs based on:
  - Per-channel RMS (amplitude profile)
  - Per-channel spectral centroid (frequency content)
  - Global log-SNR (signal quality)
- 4 lightweight expert heads (BiGRU + Attention + Classifier) specialize
- Soft routing: final output = weighted sum of expert outputs
- Load-balancing auxiliary loss prevents expert collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalGatingNetwork(nn.Module):
    """
    Gating network that computes expert routing weights from signal-level features.

    Extracts domain-agnostic characteristics directly from raw EMG input:
    - Per-channel RMS (amplitude profile)
    - Per-channel spectral centroid via FFT (frequency content)
    - Global log-SNR estimate (signal quality)

    All computations use torch ops for GPU compatibility and gradient flow.
    """

    def __init__(self, in_channels: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        # Gating feature dimension: per-channel RMS + per-channel spectral centroid + global SNR
        gating_dim = 2 * in_channels + 1
        self.feature_bn = nn.BatchNorm1d(gating_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gating_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_experts),
        )

    def _compute_gating_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract signal-level features for gating.

        Args:
            x: (B, C, T) raw EMG input

        Returns:
            (B, 2*C + 1) feature vector
        """
        # Per-channel RMS — amplitude profile
        rms = torch.sqrt(x.pow(2).mean(dim=2) + 1e-8)  # (B, C)

        # Per-channel spectral centroid via FFT
        fft_mag = torch.fft.rfft(x, dim=2).abs()  # (B, C, T//2+1)
        n_freq = fft_mag.shape[2]
        freq_bins = torch.linspace(0, 1, n_freq, device=x.device, dtype=x.dtype)
        spectral_centroid = (
            (fft_mag * freq_bins[None, None, :]).sum(dim=2)
            / (fft_mag.sum(dim=2) + 1e-8)
        )  # (B, C)

        # Global log-SNR estimate: signal power / high-freq noise power
        signal_power = x.pow(2).mean(dim=(1, 2))  # (B,)
        hf_start = n_freq * 3 // 4
        noise_power = fft_mag[:, :, hf_start:].pow(2).mean(dim=(1, 2))  # (B,)
        snr = torch.log1p(signal_power / (noise_power + 1e-8))  # (B,)

        return torch.cat([rms, spectral_centroid, snr.unsqueeze(1)], dim=1)  # (B, 2C+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EMG input

        Returns:
            (B, num_experts) soft routing weights (sums to 1)
        """
        features = self._compute_gating_features(x)  # (B, 2C+1)
        features = self.feature_bn(features)
        gate_logits = self.gate_mlp(features)  # (B, num_experts)
        return F.softmax(gate_logits, dim=1)


class ExpertHead(nn.Module):
    """
    Lightweight expert head: BiGRU + Attention + Classifier.

    Each expert processes shared CNN features independently, potentially
    specializing in different signal styles / subject domains.
    """

    def __init__(
        self,
        cnn_output_dim: int,
        num_classes: int,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cnn_output_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_classes),
        )

    def forward(self, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_features: (B, T', cnn_dim) — output of shared CNN backbone

        Returns:
            (B, num_classes) logits
        """
        gru_out, _ = self.gru(cnn_features)  # (B, T', gru_hidden*2)
        attn_weights = self.attention(gru_out)  # (B, T', 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (attn_weights * gru_out).sum(dim=1)  # (B, gru_hidden*2)
        return self.classifier(context)  # (B, num_classes)


class MoECNNGRUAttention(nn.Module):
    """
    Mixture-of-Experts CNN-GRU-Attention for EMG gesture recognition.

    Architecture:
    - Shared CNN backbone (Conv1d blocks with BN, ReLU, MaxPool)
    - Signal-style gating network (amplitude + spectral + SNR features)
    - N expert heads (each: BiGRU + Attention + Classifier)
    - Soft routing: output = sum(gate_weight_i * expert_output_i)
    - Load-balancing auxiliary loss stored in self._aux_loss during training

    Compatible with the trainer's model registry interface:
        __init__(in_channels, num_classes, dropout=0.3)
        forward(x) -> (B, num_classes) logits
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        num_experts: int = 4,
        cnn_channels: list = None,
        expert_gru_hidden: int = 64,
        expert_gru_layers: int = 2,
        load_balance_weight: float = 0.1,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64]

        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        self._aux_loss = None  # populated during forward in training mode

        # Shared CNN backbone — same architecture as CNNGRUWithAttention
        cnn_layers = []
        prev_ch = in_channels
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(prev_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.5),
            ])
            prev_ch = out_ch
        self.shared_cnn = nn.Sequential(*cnn_layers)

        # Signal-style gating network
        self.gating = SignalGatingNetwork(in_channels, num_experts, dropout=0.1)

        # Expert heads
        self.experts = nn.ModuleList([
            ExpertHead(
                cnn_output_dim=cnn_channels[-1],
                num_classes=num_classes,
                gru_hidden=expert_gru_hidden,
                gru_layers=expert_gru_layers,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

    def _compute_load_balance_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Importance-based load balancing loss.

        Encourages uniform expert utilization by penalizing the squared mean
        of gate weights across the batch. Minimum when all experts receive
        equal weight (1/num_experts).

        Args:
            gate_weights: (B, num_experts) soft routing weights

        Returns:
            scalar loss
        """
        # Mean gate weight per expert across batch
        f = gate_weights.mean(dim=0)  # (num_experts,)
        # Penalize deviation from uniform: minimum at f_i = 1/N for all i
        return self.num_experts * (f ** 2).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EMG input

        Returns:
            (B, num_classes) weighted logits
        """
        # Compute expert routing weights from raw signal characteristics
        gate_weights = self.gating(x)  # (B, num_experts)

        # Shared CNN feature extraction
        cnn_out = self.shared_cnn(x)  # (B, cnn_ch, T')
        cnn_features = cnn_out.transpose(1, 2)  # (B, T', cnn_ch)

        # Run all expert heads
        expert_outputs = torch.stack(
            [expert(cnn_features) for expert in self.experts], dim=1
        )  # (B, num_experts, num_classes)

        # Soft routing: weighted sum of expert outputs
        weighted_logits = (
            expert_outputs * gate_weights.unsqueeze(2)
        ).sum(dim=1)  # (B, num_classes)

        # Compute auxiliary load-balancing loss during training
        if self.training:
            self._aux_loss = self.load_balance_weight * self._compute_load_balance_loss(gate_weights)
        else:
            self._aux_loss = None

        return weighted_logits
