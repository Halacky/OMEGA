import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    Inception-style module that captures features at multiple temporal scales.
    Each branch uses different kernel sizes to capture patterns of different durations.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Ensure out_channels is divisible by 4 for equal split
        branch_channels = out_channels // 4
        
        # Branch 1: 1x1 conv (point-wise)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        # Branch 2: 1x1 -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        # Branch 3: 1x1 -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        # Branch 4: 3x3 max pooling -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate along channel dimension
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs


class MultiScaleCNN1D(nn.Module):
    """
    Multi-scale CNN with Inception-style modules.
    Captures EMG features at different temporal scales simultaneously.
    More robust to variations in gesture execution speed.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Inception modules
        self.inception1 = InceptionModule(32, 64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.inception2 = InceptionModule(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.inception3 = InceptionModule(128, 256)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv_init(x)
        
        x = self.inception1(x)
        x = self.pool1(x)
        
        x = self.inception2(x)
        x = self.pool2(x)
        
        x = self.inception3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x