import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWindowLearningWithVitForDynamicGestures(nn.Module):
    """
    Adaptive Window Learning with Vision Transformer for Dynamic Gestures.

    This model dynamically adjusts window sizes based on signal characteristics
    and uses a Vision Transformer (ViT) for classification. It incorporates
    attention-based weighting to optimize for agility and accuracy.
    """

    def __init__(self, in_channels: int, num_classes: int,
                 sequence_length: int = 200, dropout: float = 0.5,
                 base_window_size: int = 50, max_window_size: int = 200,
                 patch_size: int = 10, num_layers: int = 6, num_heads: int = 8,
                 hidden_dim: int = 256, mlp_dim: int = 512, **kwargs):
        """
        Initializes the AdaptiveWindowLearningWithVitForDynamicGestures model.

        Args:
            in_channels (int): Number of input channels (e.g., EMG sensors).
            num_classes (int): Number of output classes (gestures).
            sequence_length (int): Length of the input sequence.
            dropout (float): Dropout probability.
            base_window_size (int): Base window size in milliseconds.
            max_window_size (int): Maximum window size in milliseconds.
            patch_size (int): Size of the patches for the Vision Transformer.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension of the transformer.
            mlp_dim (int): Dimension of the MLP in the transformer.
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.base_window_size = base_window_size
        self.max_window_size = max_window_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        # Adaptive Window Selection (simplified - replace with actual algorithm)
        self.window_size = min(self.base_window_size + (sequence_length // 10), self.max_window_size)

        # Patchify (using conv1d as a patch extraction layer)
        self.patchify = nn.Conv1d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AdaptiveWindowLearningWithVitForDynamicGestures model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_classes).
        """
        # x shape: (batch, in_channels, sequence_length)

        # Adaptive Windowing (using a fixed window size for simplicity)
        # In a real implementation, this would involve a dynamic window selection algorithm
        # based on signal characteristics.
        windowed_x = x[:, :, :self.window_size]  # (batch, in_channels, window_size)

        # Patchify
        patches = self.patchify(windowed_x)  # (batch, hidden_dim, num_patches)
        patches = patches.transpose(1, 2)  # (batch, num_patches, hidden_dim)

        # Transformer Encoder
        transformer_out = self.transformer_encoder(patches)  # (batch, num_patches, hidden_dim)

        # Global Average Pooling
        transformer_out = transformer_out.transpose(1, 2) # (batch, hidden_dim, num_patches)
        pooled = self.adaptive_pool(transformer_out).squeeze(-1)  # (batch, hidden_dim)

        # Dropout
        pooled = self.dropout_layer(pooled)

        # Classification head
        output = self.fc(pooled)  # (batch, num_classes)

        return output