# FILE: models/vq_disentangle_emg.py
"""
Vector Quantization Disentanglement Model for EMG Gesture Recognition.

Implements content-style disentanglement through discrete VQ codebooks,
forcing gesture representations into canonical forms while separating subject style.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantization layer with EMA (Exponential Moving Average) updates.
    Based on VQ-VAE-2 and EnCodec architectures.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize embeddings uniformly
        self.register_buffer(
            'embeddings', 
            torch.randn(num_embeddings, embedding_dim) * 0.1
        )
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embeddings.clone())
        
        # Track usage for reset strategy
        self.register_buffer('code_usage_count', torch.zeros(num_embeddings))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: Input tensor of shape (B, D) or (B, ..., D)
        
        Returns:
            quantized: Quantized tensor
            loss: Commitment loss
            info: Dict with perplexity, encodings, usage info
        """
        original_shape = x.shape
        flat_x = x.view(-1, self.embedding_dim)
        
        # Calculate distances to all codebook entries
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 * x^T e
        x_sq = (flat_x ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.embeddings ** 2).sum(dim=1)
        dist = x_sq + e_sq - 2 * torch.matmul(flat_x, self.embeddings.t())
        
        # Get nearest code
        encoding_indices = torch.argmin(dist, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings)
        quantized = quantized.view(original_shape)
        
        # EMA updates (only during training)
        if self.training:
            # Update cluster size
            cluster_size = encodings.sum(dim=0)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size
            
            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon) / 
                (n + self.num_embeddings * self.epsilon) * n
            )
            
            # Update embeddings
            dw = torch.matmul(encodings.t(), flat_x)
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
            
            # Normalize
            self.embeddings = self.ema_w / self.ema_cluster_size.unsqueeze(1)
            
            # Track usage for reset strategy
            self.code_usage_count = self.code_usage_count + cluster_size
        
        # Straight-through estimator
        quantized_st = x + (quantized - x).detach()
        
        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(x, quantized.detach())
        
        # Perplexity
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        info = {
            'perplexity': perplexity,
            'encoding_indices': encoding_indices.view(original_shape[:-1]),
            'encodings': encodings,
            'commitment_loss': commitment_loss,
        }
        
        return quantized_st, commitment_loss, info
    
    def get_usage_stats(self) -> Dict:
        """Get codebook usage statistics."""
        usage = self.code_usage_count / self.code_usage_count.sum()
        used_codes = (usage > 0.001).sum().item()
        return {
            'used_codes': used_codes,
            'total_codes': self.num_embeddings,
            'usage_distribution': usage.cpu().numpy(),
        }
    
    def reset_unused_codes(self, threshold: float = 0.001) -> int:
        """Reset codes with usage below threshold."""
        usage = self.code_usage_count / (self.code_usage_count.sum() + 1e-8)
        unused_mask = usage < threshold
        num_reset = unused_mask.sum().item()
        
        if num_reset > 0:
            # Re-initialize unused codes randomly
            unused_indices = torch.where(unused_mask)[0]
            self.embeddings[unused_indices] = torch.randn(
                num_reset, self.embedding_dim, device=self.embeddings.device
            ) * 0.1
            self.ema_w[unused_indices] = self.embeddings[unused_indices]
            self.ema_cluster_size[unused_indices] = 0
            self.code_usage_count[unused_indices] = 0
        
        return num_reset


class VQDisentangleEMG(nn.Module):
    """
    VQ-based Content-Style Disentanglement Model for EMG.
    
    Architecture:
    - Encoder extracts embeddings from EMG windows
    - VQ-Content Codebook quantizes gesture patterns into discrete codes
    - VQ-Style Codebook captures subject variations
    - Classifier uses ONLY content codes for gesture prediction
    
    Training objectives:
    - Classification loss on content codes
    - Commitment loss for VQ
    - Diversity loss to prevent codebook collapse
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        # VQ parameters
        content_codebook_size: int = 128,
        content_codebook_dim: int = 128,
        style_codebook_size: int = 64,
        style_codebook_dim: int = 64,
        commitment_cost: float = 0.25,
        # Architecture
        encoder_hidden: int = 128,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        
        self.content_codebook_dim = content_codebook_dim
        self.style_codebook_dim = style_codebook_dim
        self.diversity_weight = diversity_weight
        
        # ===== ENCODER =====
        # Input: (B, C, T) -> Output: (B, hidden_dim)
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            
            # Block 3
            nn.Conv1d(128, encoder_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # ===== CONTENT BRANCH =====
        self.content_proj = nn.Sequential(
            nn.Linear(encoder_hidden, content_codebook_dim),
            nn.LayerNorm(content_codebook_dim),
            nn.ReLU(),
        )
        
        # ===== STYLE BRANCH =====
        self.style_proj = nn.Sequential(
            nn.Linear(encoder_hidden, style_codebook_dim),
            nn.LayerNorm(style_codebook_dim),
            nn.ReLU(),
        )
        
        # ===== VQ LAYERS =====
        self.vq_content = VectorQuantizerEMA(
            num_embeddings=content_codebook_size,
            embedding_dim=content_codebook_dim,
            commitment_cost=commitment_cost,
            decay=0.99,
        )
        
        self.vq_style = VectorQuantizerEMA(
            num_embeddings=style_codebook_size,
            embedding_dim=style_codebook_dim,
            commitment_cost=commitment_cost,
            decay=0.99,
        )
        
        # ===== CLASSIFIER (uses ONLY content codes) =====
        self.classifier = nn.Sequential(
            nn.Linear(content_codebook_dim, content_codebook_dim // 2),
            nn.LayerNorm(content_codebook_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_codebook_dim // 2, num_classes),
        )
        
        # Store for auxiliary losses
        self.auxiliary_losses = {}
        self.quantization_info = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with straight-through estimation.
        
        Args:
            x: Input tensor of shape (B, C, T) - EMG windows
            
        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Encode
        encoded = self.encoder(x)  # (B, encoder_hidden)
        encoded = encoded.squeeze(-1)  # (B, encoder_hidden)
        
        # Project to content and style spaces
        z_content = self.content_proj(encoded)  # (B, content_dim)
        z_style = self.style_proj(encoded)  # (B, style_dim)
        
        # Vector Quantization
        q_content, content_commit_loss, content_info = self.vq_content(z_content)
        q_style, style_commit_loss, style_info = self.vq_style(z_style)
        
        # Store auxiliary losses
        total_commit_loss = content_commit_loss + style_commit_loss
        diversity_loss = self._compute_diversity_loss(content_info['encodings'], style_info['encodings'])
        
        self.auxiliary_losses = {
            'commitment_loss': total_commit_loss,
            'diversity_loss': diversity_loss,
            'total_aux_loss': total_commit_loss + self.diversity_weight * diversity_loss,
        }
        
        self.quantization_info = {
            'content_perplexity': content_info['perplexity'],
            'style_perplexity': style_info['perplexity'],
            'content_indices': content_info['encoding_indices'],
            'style_indices': style_info['encoding_indices'],
        }
        
        # Classification using ONLY content codes
        logits = self.classifier(q_content)
        
        return logits
    
    def _compute_diversity_loss(
        self, 
        content_encodings: torch.Tensor, 
        style_encodings: torch.Tensor
    ) -> torch.Tensor:
        """
        Diversity loss to encourage uniform codebook usage.
        Prevents codebook collapse by penalizing peaked distributions.
        """
        # Content diversity
        content_probs = content_encodings.mean(dim=0)
        content_entropy = -torch.sum(content_probs * torch.log(content_probs + 1e-10))
        content_diversity = -content_entropy / math.log(content_encodings.shape[1])
        
        # Style diversity
        style_probs = style_encodings.mean(dim=0)
        style_entropy = -torch.sum(style_probs * torch.log(style_probs + 1e-10))
        style_diversity = -style_entropy / math.log(style_encodings.shape[1])
        
        # Negative entropy (we want to maximize entropy = minimize negative entropy)
        return content_diversity + style_diversity
    
    def get_auxiliary_loss(self) -> torch.Tensor:
        """Get combined auxiliary loss for training."""
        return self.auxiliary_losses.get('total_aux_loss', torch.tensor(0.0, device=next(self.parameters()).device))
    
    def reset_unused_codebooks(self, threshold: float = 0.001) -> Dict[str, int]:
        """Reset unused codes in both codebooks."""
        content_reset = self.vq_content.reset_unused_codes(threshold)
        style_reset = self.vq_style.reset_unused_codes(threshold)
        return {'content_codes_reset': content_reset, 'style_codes_reset': style_reset}
    
    def get_codebook_stats(self) -> Dict:
        """Get usage statistics for both codebooks."""
        return {
            'content': self.vq_content.get_usage_stats(),
            'style': self.vq_style.get_usage_stats(),
        }


def create_vq_disentangle_emg(
    in_channels: int,
    num_classes: int,
    dropout: float = 0.3,
    **kwargs
) -> VQDisentangleEMG:
    """Factory function for VQ Disentangle EMG model."""
    return VQDisentangleEMG(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs
    )