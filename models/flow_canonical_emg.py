"""
Subject-Conditional Normalizing Flow for Canonical EMG Representation.

Maps subject-specific EMG feature distributions to a shared canonical
latent space via invertible affine coupling layers.

Architecture:
    Input (B, C, T) -> CNN-GRU Encoder -> z (B, hidden_dim)
    z -> Affine Coupling Flow (K layers) -> z_canonical (B, hidden_dim)
    z_canonical -> Classifier -> gesture logits (B, num_classes)

Training:
    L = L_gesture + lambda_flow * L_flow
    L_flow = negative log-likelihood under the flow (change of variables)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flow.

    Splits the input into two halves: x1, x2.
    x1 stays the same; x2 is transformed via an affine function
    conditioned on x1: y2 = x2 * exp(s(x1)) + t(x1).

    The log-determinant of the Jacobian = sum(s(x1)).
    """

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        half_dim = dim // 2
        self.split_dim = half_dim

        # s and t networks
        self.net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.s_out = nn.Linear(hidden_dim, dim - half_dim)
        self.t_out = nn.Linear(hidden_dim, dim - half_dim)

        # Initialize near identity
        nn.init.zeros_(self.s_out.weight)
        nn.init.zeros_(self.s_out.bias)
        nn.init.zeros_(self.t_out.weight)
        nn.init.zeros_(self.t_out.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (encoding direction).
        Returns: (y, log_det_jacobian)
        """
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        h = self.net(x1)
        s = torch.tanh(self.s_out(h)) * 0.5  # clamp scale
        t = self.t_out(h)

        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)

        log_det = s.sum(dim=1)  # (B,)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass (decoding direction)."""
        y1, y2 = y[:, :self.split_dim], y[:, self.split_dim:]
        h = self.net(y1)
        s = torch.tanh(self.s_out(h)) * 0.5
        t = self.t_out(h)

        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=1)


class FlowCanonicalEMG(nn.Module):
    """
    EMG model with normalizing flow for canonical representation.

    Args:
        in_channels: Number of EMG channels (C).
        num_classes: Number of gesture classes.
        dropout: Dropout rate.
        hidden_dim: Latent dimension for encoder and flow.
        n_flow_layers: Number of affine coupling layers.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        hidden_dim: int = 128,
        n_flow_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_flow_layers = n_flow_layers

        # ===== CNN Encoder =====
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ===== GRU for temporal context =====
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        # Projection from GRU output to hidden_dim
        self.encoder_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ===== Normalizing Flow =====
        self.flow_layers = nn.ModuleList()
        for i in range(n_flow_layers):
            self.flow_layers.append(AffineCouplingLayer(hidden_dim, hidden_dim * 2))

        # Permutation: flip dimensions between coupling layers
        self.register_buffer(
            "flip_idx",
            torch.arange(hidden_dim - 1, -1, -1, dtype=torch.long),
        )

        # ===== Gesture Classifier (on canonical representation) =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw EMG to latent representation.
        Args: x (B, C, T)
        Returns: z (B, hidden_dim)
        """
        # CNN features
        cnn_out = self.encoder_cnn(x).squeeze(-1)  # (B, hidden_dim)

        # GRU features from intermediate CNN
        # Use the second conv block output for GRU
        h = x
        for layer in list(self.encoder_cnn.children())[:5]:
            h = layer(h)
        # h is (B, 64, T/2) -- transpose for GRU
        h2 = h
        for layer in list(self.encoder_cnn.children())[5:10]:
            h2 = layer(h2)
        # h2 is (B, 128, T/4)
        gru_input = h2.permute(0, 2, 1)  # (B, T/4, 128)
        gru_out, _ = self.gru(gru_input)  # (B, T/4, hidden_dim)
        gru_pooled = gru_out.mean(dim=1)  # (B, hidden_dim)

        # Combine CNN and GRU
        z = self.encoder_proj(cnn_out + gru_pooled)
        return z

    def flow_forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass latent z through normalizing flow.
        Returns: (z_canonical, total_log_det)
        """
        total_log_det = torch.zeros(z.size(0), device=z.device)

        for i, layer in enumerate(self.flow_layers):
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
            # Permute dimensions between layers
            if i < len(self.flow_layers) - 1:
                z = z[:, self.flip_idx]

        return z, total_log_det

    def flow_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood under standard normal prior.

        NLL = -log p(z_canonical) - log |det J|
            = 0.5 * ||z_canonical||^2 + 0.5 * D * log(2pi) - log_det
        """
        z_canonical, log_det = self.flow_forward(z)

        # Log-probability under standard normal
        D = z_canonical.shape[1]
        log_prob = -0.5 * (z_canonical.pow(2).sum(dim=1) + D * math.log(2 * math.pi))

        # NLL = -(log_prob + log_det)
        nll = -(log_prob + log_det)
        return nll.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass: encode -> flow -> classify.
        Args: x (B, C, T)
        Returns: logits (B, num_classes)
        """
        z = self.encode(x)
        z_canonical, _ = self.flow_forward(z)
        logits = self.classifier(z_canonical)
        return logits

    def forward_all(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning all components for training.
        Returns dict with: logits, z, z_canonical, log_det
        """
        z = self.encode(x)
        z_canonical, log_det = self.flow_forward(z)
        logits = self.classifier(z_canonical)
        return {
            "logits": logits,
            "z": z,
            "z_canonical": z_canonical,
            "log_det": log_det,
        }
