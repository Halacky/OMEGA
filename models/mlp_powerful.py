import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPForFeatures(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # ожидает (N, C, T) или (N, F, 1)
        if x.ndim == 3:
            # (N, C, T) -> (N, C*T) но для powerful T=1, можно просто squeeze
            x = x.squeeze(-1)  # (N, C)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x