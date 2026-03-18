import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention mechanism.
    Learns to reweight channels dynamically, helping with sensor rotation invariance.
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Attention over temporal dimension to focus on important time steps.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, T)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class AttentionCNN1D(nn.Module):
    """
    CNN with channel and spatial attention mechanisms (CBAM-style).
    Designed to handle sensor rotation by learning rotation-invariant channel representations.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.ca1 = ChannelAttention(32, reduction=4)
        self.sa1 = SpatialAttention(kernel_size=7)
        self.pool1 = nn.MaxPool1d(2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.ca2 = ChannelAttention(64, reduction=4)
        self.sa2 = SpatialAttention(kernel_size=7)
        self.pool2 = nn.MaxPool1d(2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.ca3 = ChannelAttention(128, reduction=4)
        self.sa3 = SpatialAttention(kernel_size=7)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.ca1(x)
        x = self.sa1(x)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.ca2(x)
        x = self.sa2(x)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.ca3(x)
        x = self.sa3(x)
        
        # Classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x