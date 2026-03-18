# FILE: models/dual_stream_cnn_gru_attention.py
"""
Dual-Stream CNN-GRU-Attention with Handcrafted Feature Fusion

Architecture:
- Stream 1: Raw EMG through CNN-GRU-Attention (same as proven CNNGRUWithAttention)
- Stream 2: Handcrafted powerful features through 2-layer MLP
- Late fusion: Attention mechanism to weight and combine outputs from both streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamCNNGRUAttention(nn.Module):
    """
    Dual-stream architecture for EMG gesture recognition:
    - Primary stream: augmented raw EMG through CNN-GRU-Attention
    - Secondary stream: handcrafted powerful features through lightweight MLP
    - Late fusion: attention-based weighting to combine streams
    """
    
    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        dropout: float = 0.3,
        handcrafted_dim: int = 256,  # dimension of handcrafted feature vector
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64]
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.handcrafted_dim = handcrafted_dim
        self.gru_hidden = gru_hidden
        
        # ========== STREAM 1: CNN-GRU-Attention for Raw EMG ==========
        
        # CNN feature extractor
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
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        
        # Attention mechanism for GRU output
        self.gru_attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )
        
        # Projection for stream 1 output
        self.stream1_proj = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ========== STREAM 2: MLP for Handcrafted Features ==========
        
        self.mlp = nn.Sequential(
            nn.Linear(handcrafted_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ========== LATE FUSION WITH ATTENTION ==========
        
        # Fusion attention: learns to weight each stream's contribution
        # Input: concatenated [stream1_feat, stream2_feat] -> weights for each stream
        self.fusion_attention = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # 2 weights: one for each stream
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x, handcrafted_features=None):
        """
        Forward pass.
        
        Args:
            x: Raw EMG signal tensor of shape (B, C, T)
            handcrafted_features: Handcrafted feature tensor of shape (B, handcrafted_dim)
                                  If None, uses zeros (degrades to single stream)
        
        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        batch_size = x.size(0)
        
        # ===== Stream 1: CNN-GRU-Attention =====
        # CNN feature extraction: (B, C, T) -> (B, cnn_channels[-1], T')
        cnn_out = self.cnn(x)
        
        # Prepare for GRU: (B, C, T) -> (B, T, C)
        gru_in = cnn_out.permute(0, 2, 1)
        
        # GRU forward: (B, T, C) -> (B, T, gru_hidden*2)
        gru_out, _ = self.gru(gru_in)
        
        # Attention pooling over time
        # (B, T, gru_hidden*2) -> (B, T, 1) -> (B, T)
        attn_weights = self.gru_attention(gru_out).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum: (B, T, gru_hidden*2) * (B, T, 1) -> (B, gru_hidden*2)
        stream1_feat = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)
        
        # Project to common dimension
        stream1_feat = self.stream1_proj(stream1_feat)  # (B, 128)
        
        # ===== Stream 2: MLP for Handcrafted Features =====
        if handcrafted_features is None:
            # If no handcrafted features, use zeros (degraded mode)
            handcrafted_features = torch.zeros(
                batch_size, self.handcrafted_dim, 
                device=x.device, dtype=x.dtype
            )
        
        stream2_feat = self.mlp(handcrafted_features)  # (B, 64)
        
        # ===== Late Fusion with Attention =====
        # Concatenate features from both streams
        combined = torch.cat([stream1_feat, stream2_feat], dim=1)  # (B, 192)
        
        # Compute attention weights for each stream
        fusion_logits = self.fusion_attention(combined)  # (B, 2)
        fusion_weights = F.softmax(fusion_logits, dim=1)  # (B, 2)
        
        # Weight each stream's contribution
        weighted_stream1 = stream1_feat * fusion_weights[:, 0:1]  # (B, 128)
        weighted_stream2 = stream2_feat * fusion_weights[:, 1:2]  # (B, 64)
        
        # Final fused representation
        fused = torch.cat([weighted_stream1, weighted_stream2], dim=1)  # (B, 192)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits