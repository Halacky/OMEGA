import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """
    A residual block with dilated causal convolutions.
    Uses weight normalization and residual connections for stable training.
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)  # Remove excess padding for causal convolution
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # 1x1 convolution for residual connection if dimensions don't match
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Removes the last elements of a time series to make convolution causal.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with residual connections.
    Better at capturing long-range temporal dependencies compared to standard CNNs.
    Uses dilated convolutions to increase receptive field exponentially.
    """
    def __init__(self, in_channels: int, num_classes: int, 
                 num_channels: list = None, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        if num_channels is None:
            num_channels = [32, 64, 128]
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1,
                dilation=dilation_size, padding=padding, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        x = self.network(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class TemporalConvNetWithAttention(nn.Module):
    """
    Temporal Convolutional Network with multi-head self-attention on top.
    This model:
      1) Applies standard TCN layers to extract temporal features.
      2) Applies multi-head self-attention to capture long-range dependencies.
      3) Applies global pooling (mean) on the attention output.
    Input:
      - x: (N, C_in, T)
    Output:
      - logits: (N, num_classes)
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 128]

        # Reuse original TemporalConvNet backbone as feature extractor
        self.tcn = TemporalConvNet(
            in_channels=in_channels,
            num_classes=num_classes,  # this is not used directly, but we will not use its fc
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Replace original head: we only use its network part
        self.tcn.fc = nn.Identity()
        self.tcn.global_pool = nn.Identity()

        d_model = num_channels[-1]
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=False,  # we will use (T, N, C)
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (N, C_in, T)
        x = self.tcn.network(x)  # (N, C_out, T)

        # Prepare for multi-head attention: (T, N, C_out)
        x = x.permute(2, 0, 1).contiguous()

        # Self-attention (query=key=value=x)
        attn_out, _ = self.attn(x, x, x)
        x = x + self.attn_dropout(attn_out)
        x = self.out_norm(x)

        # Global average over time: (T, N, C) -> (N, C)
        x = x.mean(dim=0)

        logits = self.fc(x)
        return logits