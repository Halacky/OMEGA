# FILE: models/cnn_gru_attention_fusion.py
"""
CNN-GRU with Attention and Feature Fusion for EMG gesture recognition.

This model combines:
1. CNN-GRU-Attention processing of raw EMG windows
2. Learnable fusion with handcrafted powerful features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGRUAttentionFusion(nn.Module):
    """
    Hybrid model that fuses CNN-GRU-Attention raw signal features 
    with handcrafted EMG features via a learnable fusion layer.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        dropout: float = 0.3,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        fusion_hidden: int = 256,
        handcrafted_dim: int = 88,  # Default for 'powerful' feature set
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64]
        
        self.in_channels = in_channels
        self.gru_hidden = gru_hidden
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
        
        # === GRU Sequence Model ===
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        
        # === Attention Mechanism ===
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )
        
        # === Handcrafted Feature Encoder ===
        self.handcrafted_encoder = nn.Sequential(
            nn.Linear(handcrafted_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # === Feature Fusion Layer ===
        # Raw features: gru_hidden * 2 (bidirectional)
        # Handcrafted features: 64
        fusion_input_dim = gru_hidden * 2 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # === Classifier ===
        self.classifier = nn.Linear(fusion_hidden // 2, num_classes)
        
    def forward(self, x: torch.Tensor, handcrafted: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Raw EMG window (B, C, T)
            handcrafted: Handcrafted features (B, handcrafted_dim)
                         If None, uses zeros (fallback for inference)
        
        Returns:
            Logits (B, num_classes)
        """
        batch_size = x.size(0)
        
        # --- CNN Feature Extraction ---
        # x: (B, C, T) -> (B, cnn_channels[-1], T')
        cnn_out = self.cnn(x)
        
        # --- Reshape for GRU ---
        # (B, C, T') -> (B, T', C)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # --- GRU Processing ---
        # gru_out: (B, T', gru_hidden * 2)
        gru_out, _ = self.gru(cnn_out)
        
        # --- Attention Mechanism ---
        # attention_weights: (B, T', 1)
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        
        # Weighted sum: (B, gru_hidden * 2)
        raw_features = (gru_out * attention_weights).sum(dim=1)
        
        # --- Handcrafted Feature Processing ---
        if handcrafted is None:
            # Fallback: use zeros (allows inference without handcrafted features)
            device = x.device
            handcrafted = torch.zeros(batch_size, self.handcrafted_dim, device=device)
        
        hc_features = self.handcrafted_encoder(handcrafted)  # (B, 64)
        
        # --- Feature Fusion ---
        fused = torch.cat([raw_features, hc_features], dim=1)  # (B, gru_hidden*2 + 64)
        fused = self.fusion(fused)  # (B, fusion_hidden // 2)
        
        # --- Classification ---
        logits = self.classifier(fused)
        
        return logits