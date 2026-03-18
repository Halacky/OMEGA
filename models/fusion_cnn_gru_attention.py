# FILE: models/fusion_cnn_gru_attention.py
"""CNN-GRU-Attention with learnable feature fusion for handcrafted features."""

import torch
import torch.nn as nn


class FusionCNNGRUAttention(nn.Module):
    """
    Hybrid model combining CNN-GRU-Attention for raw EMG with learnable fusion
    for handcrafted features.
    
    Architecture:
    1. CNN feature extractor from raw EMG
    2. BiGRU for temporal modeling
    3. Attention mechanism
    4. Fusion layer combining deep features with handcrafted features
    5. Classification head
    """
    
    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        dropout: float = 0.3,
        handcrafted_dim: int = 128,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        fusion_hidden: int = 256,
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64]
        
        self.in_channels = in_channels
        self.handcrafted_dim = handcrafted_dim
        
        # === CNN Feature Extractor ===
        cnn_layers = []
        in_ch = in_channels
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        # === BiGRU for Temporal Modeling ===
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        
        # === Multi-Head Self-Attention ===
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_hidden * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        
        # === Feature Fusion Layer ===
        # Combines deep features with handcrafted features
        deep_feature_dim = gru_hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(deep_feature_dim + handcrafted_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # === Classification Head ===
        self.classifier = nn.Linear(fusion_hidden // 2, num_classes)
        
    def forward(self, x: torch.Tensor, handcrafted: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Raw EMG windows of shape (B, C, T)
            handcrafted: Handcrafted features of shape (B, handcrafted_dim)
                        If None, uses zero tensor (degrades to standard cnn_gru_attention)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        batch_size = x.size(0)
        
        # Handle missing handcrafted features
        if handcrafted is None:
            handcrafted = torch.zeros(
                batch_size, self.handcrafted_dim, 
                device=x.device, dtype=x.dtype
            )
        
        # CNN feature extraction: (B, C, T) -> (B, cnn_channels[-1], T')
        cnn_out = self.cnn(x)
        
        # Reshape for GRU: (B, T', cnn_channels[-1])
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # BiGRU: (B, T', 2*gru_hidden)
        gru_out, _ = self.gru(cnn_out)
        
        # Self-attention over time steps
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Global pooling: (B, 2*gru_hidden)
        deep_features = attn_out.mean(dim=1)
        
        # Feature fusion
        combined = torch.cat([deep_features, handcrafted], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits