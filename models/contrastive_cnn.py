# FILE: models/contrastive_cnn.py
"""Contrastive CNN with projection head for SimCLR-style pre-training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveCNN(nn.Module):
    """SimpleCNN1D with optional projection head for contrastive learning.
    
    Two-stage training:
    1. Contrastive pre-training: use_projection=True, returns projection embeddings
    2. Supervised fine-tuning: use_projection=False, returns class logits
    """
    
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3, projection_dim: int = 128):
        super().__init__()
        
        # Encoder backbone (same architecture as SimpleCNN1D)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        
        # Projection head for contrastive learning (SimCLR style)
        self.projection_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )
        
        # Classifier head for supervised learning
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        self.use_projection = True  # Switch between contrastive and supervised mode
        self.feature_dim = 128
        self.projection_dim = projection_dim
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings from encoder."""
        return self.encoder(x)
    
    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to contrastive space."""
        return self.projection_head(features)
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify features to logits."""
        return self.classifier(features)
    
    def forward(self, x: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, T)
            return_projection: If True and use_projection=True, return projection.
                              If False or use_projection=False, return logits.
        
        Returns:
            Projection embeddings (B, projection_dim) or logits (B, num_classes)
        """
        features = self.encode(x)
        
        if return_projection and self.use_projection:
            return self.project(features)
        else:
            return self.classify(features)
    
    def set_contrastive_mode(self, enabled: bool = True):
        """Enable or disable contrastive mode."""
        self.use_projection = enabled


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR."""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss for a batch of paired embeddings.
        
        Args:
            z_i: First set of augmented embeddings (B, D)
            z_j: Second set of augmented embeddings (B, D)
        
        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Concatenate and normalize
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        z = F.normalize(z, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
        
        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss