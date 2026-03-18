import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCnnvitWithContrastiveLearning(nn.Module):
    """
    Hybrid CNN-ViT architecture for EMG gesture recognition, trained with contrastive learning.
    Combines CNN layers for local feature extraction and a ViT for global context.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 sequence_length: int = 200, dropout: float = 0.5,
                 cnn_filters: int = 32, vit_patch_size: int = 8,
                 vit_dim: int = 64, vit_depth: int = 2, vit_heads: int = 4,
                 mlp_dim: int = 128, **kwargs):
        super().__init__()

        self.cnn_filters = cnn_filters
        self.vit_patch_size = vit_patch_size
        self.vit_dim = vit_dim
        self.vit_depth = vit_depth
        self.vit_heads = vit_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # CNN layers for local feature extraction
        self.conv1 = nn.Conv1d(in_channels, cnn_filters, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Adaptive pooling to handle variable sequence lengths
        # Use integer division to ensure divisible by patch_size
        target_length = vit_patch_size * 8
        # Replace AdaptiveAvgPool1d with fixed pooling to avoid deterministic algorithm issues
        self.fixed_pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Will be applied conditionally

        # ViT for global context
        self.patchify = Patchify(patch_size=vit_patch_size)
        num_patches = target_length // vit_patch_size  # Fixed after pooling
        patch_dim = cnn_filters * 2 * vit_patch_size
        
        self.vit = ViT(
            dim=patch_dim,  # Input dimension to ViT is patch_dim
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=mlp_dim,
            num_patches=num_patches,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Hybrid CNN-ViT model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_classes).
        """
        # CNN layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Fixed pooling instead of adaptive pooling to avoid deterministic algorithm issues
        # Apply pooling until we reach target length
        target_length = self.vit_patch_size * 8
        current_length = x.shape[2]
        
        # Apply fixed pooling to reach target length
        while current_length > target_length:
            x = self.fixed_pool(x)
            current_length = x.shape[2]
        
        # If we're still not at target length, use interpolation
        if current_length != target_length:
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)

        # ViT
        x = self.patchify(x)
        x = self.vit(x)  # (batch, num_classes)

        return x


class Patchify(nn.Module):
    """
    Patchify layer for ViT.
    """
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Patchify layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_patches, patch_dim).
        """
        batch_size, channels, seq_len = x.shape
        patch_size = self.patch_size
        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        # Reshape the input tensor to (batch, num_patches, patch_dim)
        x = x.unfold(dimension=2, size=patch_size, step=patch_size)
        x = x.transpose(1, 2).reshape(batch_size, num_patches, patch_dim)
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) module.
    """
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, num_patches: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.to_embedding = nn.Linear(dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout_layer = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViT module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_patches, patch_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_classes).
        """
        x = self.to_embedding(x)
        x += self.pos_embedding
        x = self.dropout_layer(x)

        for block in self.transformer_blocks:
            x = block(x)

        # Use torch.mean instead of x.mean() to avoid deterministic algorithm issues
        x = torch.mean(x, dim=1)  # Global average pooling over the patches
        return self.mlp_head(x)


class TransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_patches, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_patches, dim).
        """
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention layer.
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_patches, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_patches, dim).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    FeedForward layer (MLP).
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, num_patches, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_patches, dim).
        """
        return self.net(x)