"""
Channel Graph Attention Network (ChannelGAT) for EMG gesture classification.

Hypothesis: Inter-electrode (inter-channel) correlations are more important
than temporal structure. In EEG/fMRI, GNNs outperform CNNs by modelling
spatial relationships between sensors.

Architecture:
    Input (B, C, T) raw EMG
    → TemporalEncoder: shared 1D CNN per channel → (B, C, d_node)
    → DynamicAdjacency: correlation + learnable prior → (B, C, C)
    → GATLayer × N: multi-head graph attention with adjacency bias → (B, C, d_node)
    → Attention readout over nodes → (B, d_node)
    → MLP classifier → (B, num_classes)

Pure PyTorch — no torch_geometric dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Shared 1D CNN that extracts a feature vector per channel."""

    def __init__(self, d_node: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, d_node, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_node),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # → (*, d_node, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EMG
        Returns:
            (B, C, d_node) node features — one vector per channel
        """
        B, C, T = x.shape
        # Process each channel independently with shared weights
        h = x.reshape(B * C, 1, T)      # (B*C, 1, T)
        h = self.net(h)                   # (B*C, d_node, 1)
        h = h.squeeze(-1)                 # (B*C, d_node)
        return h.reshape(B, C, -1)        # (B, C, d_node)


class DynamicAdjacency(nn.Module):
    """Combines data-driven correlation with a learnable adjacency prior."""

    def __init__(self, n_channels: int):
        super().__init__()
        # Learnable adjacency prior (symmetric via A + A^T)
        self.adj_prior = nn.Parameter(torch.zeros(n_channels, n_channels))
        nn.init.xavier_uniform_(self.adj_prior)
        # Learnable mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, C, d_node) node features
        Returns:
            (B, C, C) adjacency bias (not normalized — added to attention logits)
        """
        # Data-driven: Pearson correlation between node feature vectors
        h_centered = h - h.mean(dim=-1, keepdim=True)
        h_norm = h_centered / (h_centered.norm(dim=-1, keepdim=True) + 1e-8)
        corr = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (B, C, C) in [-1, 1]

        # Learnable symmetric prior
        prior = self.adj_prior + self.adj_prior.t()  # (C, C)

        # Mix
        alpha = torch.sigmoid(self.alpha)
        adj = alpha * corr + (1 - alpha) * prior.unsqueeze(0)
        return adj


class GATLayer(nn.Module):
    """Multi-head graph attention layer with adjacency bias, pre-norm, residual, FFN."""

    def __init__(self, d_node: int, n_heads: int, dropout: float = 0.1,
                 ffn_expansion: int = 2):
        super().__init__()
        assert d_node % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_node // n_heads
        self.scale = self.d_k ** -0.5

        # Attention projections
        self.W_qkv = nn.Linear(d_node, 3 * d_node)
        self.W_out = nn.Linear(d_node, d_node)
        self.attn_drop = nn.Dropout(dropout)

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(d_node)

        # FFN with pre-norm
        self.norm2 = nn.LayerNorm(d_node)
        self.ffn = nn.Sequential(
            nn.Linear(d_node, d_node * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_node * ffn_expansion, d_node),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, adj_bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, C, d_node) node features
            adj_bias: (B, C, C) adjacency bias added to attention logits
        Returns:
            (B, C, d_node) updated node features
        """
        B, C, D = h.shape

        # Pre-norm attention
        h_norm = self.norm1(h)
        qkv = self.W_qkv(h_norm).reshape(B, C, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, C, d_k)
        q, k, v = qkv.unbind(0)

        # Attention scores with adjacency bias
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, C, C)
        attn = attn + adj_bias.unsqueeze(1)               # broadcast over heads
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate
        out = (attn @ v).transpose(1, 2).reshape(B, C, D)  # (B, C, d_node)
        out = self.W_out(out)
        h = h + out  # residual

        # Pre-norm FFN
        h = h + self.ffn(self.norm2(h))
        return h


class ChannelGAT(nn.Module):
    """
    Channel Graph Attention Network for EMG gesture classification.

    Nodes = EMG channels. Edges = learned + correlation-based adjacency.
    Temporal features extracted per-channel, then GAT reasons over
    inter-channel relationships.

    Args:
        in_channels: number of EMG channels (= number of graph nodes)
        num_classes: number of gesture classes
        dropout: dropout rate
        d_node: node feature dimension
        n_heads: number of GAT attention heads
        n_gat_layers: number of stacked GAT layers
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3,
                 d_node: int = 64, n_heads: int = 4, n_gat_layers: int = 3,
                 **kwargs):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(d_node=d_node)
        self.dynamic_adj = DynamicAdjacency(n_channels=in_channels)

        self.gat_layers = nn.ModuleList([
            GATLayer(d_node=d_node, n_heads=n_heads, dropout=dropout * 0.5,
                     ffn_expansion=2)
            for _ in range(n_gat_layers)
        ])

        self.final_norm = nn.LayerNorm(d_node)

        # Attention readout: learned gate per node
        self.readout_gate = nn.Sequential(
            nn.Linear(d_node, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_node, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw standardized EMG
        Returns:
            (B, num_classes) logits
        """
        # 1. Extract per-channel temporal features → graph node embeddings
        h = self.temporal_encoder(x)           # (B, C, d_node)

        # 2. Compute dynamic adjacency from node features
        adj_bias = self.dynamic_adj(h)          # (B, C, C)

        # 3. GAT layers: inter-channel reasoning
        for gat in self.gat_layers:
            h = gat(h, adj_bias)                # (B, C, d_node)

        h = self.final_norm(h)                  # (B, C, d_node)

        # 4. Attention-weighted readout over nodes
        gate = self.readout_gate(h)             # (B, C, 1)
        h = (h * gate).sum(dim=1) / (gate.sum(dim=1) + 1e-8)  # (B, d_node)

        # 5. Classify
        return self.classifier(h)               # (B, num_classes)
