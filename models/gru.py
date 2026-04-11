import torch
import torch.nn as nn


class BiGRU(nn.Module):
    """
    Bidirectional GRU for EMG gesture recognition.
    GRU is a lighter alternative to LSTM with fewer parameters,
    often achieving comparable performance with faster training.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        # GRU expects: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, time, channels)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch, time, hidden*2)
        
        # Use the last time step output
        gru_out = gru_out[:, -1, :]  # (batch, hidden*2)
        
        # Classifier
        x = self.dropout(gru_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x