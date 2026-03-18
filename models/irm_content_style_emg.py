# FILE: models/irm_content_style_emg.py
"""
IRM Content-Style EMG Model for Invariant Risk Minimization.

Architecture:
- Shared CNN encoder extracts features from raw EMG windows
- Content encoder maps to gesture-relevant features
- Gesture classifier operates on content features
- IRM penalty computed during training to ensure invariant predictions

The key insight: IRM selects features whose predictive relationship 
with gesture labels is INVARIANT across subjects (environments).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class IRMContentStyleEMG(nn.Module):
    """
    Content-style disentanglement model with IRM support for EMG gesture recognition.
    
    Args:
        in_channels: Number of input EMG channels
        num_classes: Number of gesture classes
        dropout: Dropout rate
        content_dim: Dimension of content feature space
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        content_dim: int = 128,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.content_dim = content_dim
        
        # Shared CNN encoder for raw EMG
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
        )
        
        # Content encoder (gesture-relevant features)
        self.content_encoder = nn.Sequential(
            nn.Linear(128, content_dim),
            nn.LayerNorm(content_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Gesture classifier on content features
        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.LayerNorm(content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            x: Input tensor of shape (B, C, T) where C is channels, T is time
            
        Returns:
            Gesture logits of shape (B, num_classes)
        """
        features = self.encoder(x)  # (B, 128, 1)
        features = features.squeeze(-1)  # (B, 128)
        content = self.content_encoder(features)  # (B, content_dim)
        logits = self.gesture_classifier(content)  # (B, num_classes)
        return logits
    
    def forward_with_content(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and content features.
        Used during training for IRM penalty computation.
        
        Args:
            x: Input tensor of shape (B, C, T)
            
        Returns:
            Tuple of (logits, content_features)
        """
        features = self.encoder(x)  # (B, 128, 1)
        features = features.squeeze(-1)  # (B, 128)
        content = self.content_encoder(features)  # (B, content_dim)
        logits = self.gesture_classifier(content)  # (B, num_classes)
        return logits, content
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract content features only."""
        features = self.encoder(x).squeeze(-1)
        return self.content_encoder(features)


def compute_irm_penalty(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """
    Compute IRMv1 penalty: ||grad_{w=1} L(w * logits)||^2
    
    The penalty measures how much the loss changes when we scale logits
    by a scalar w near w=1. If the classifier is optimal for all
    environments, this gradient should be near zero.
    
    Args:
        logits: Model output logits (B, num_classes)
        labels: Ground truth labels (B,)
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        
    Returns:
        IRM penalty scalar
    """
    # Create dummy scalar w = 1.0 with gradients enabled
    w = logits.new_tensor(1.0, requires_grad=True)
    
    # Scale logits by w
    scaled_logits = logits * w
    
    # Compute loss with scaled logits
    loss = loss_fn(scaled_logits, labels)
    
    # Compute gradient of loss w.r.t. w
    grad = torch.autograd.grad(loss, w, create_graph=True)[0]
    
    # Penalty is squared gradient
    return grad ** 2