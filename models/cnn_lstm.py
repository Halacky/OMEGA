import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model combining spatial feature extraction with temporal modeling.
    CNN extracts local patterns, LSTM models temporal dependencies.
    
    This architecture is popular for EMG because:
    - CNN captures spatial patterns across channels
    - LSTM captures temporal dynamics of gesture execution
    """
    def __init__(self, in_channels: int, num_classes: int,
                 cnn_channels: list = None, lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64]
        
        # CNN feature extractor
        cnn_layers = []
        prev_channels = in_channels
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(prev_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.5)  # Lighter dropout in CNN
            ])
            prev_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x):
        # CNN feature extraction
        # x: (batch, channels, time)
        x = self.cnn(x)  # (batch, cnn_channels[-1], reduced_time)
        
        # Prepare for LSTM: (batch, time, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # (batch, time, lstm_hidden*2)
        
        # Use last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Classifier
        x = self.dropout(lstm_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNNGRUWithAttention(nn.Module):
    """
    Hybrid CNN-GRU with attention mechanism.
    Combines CNN feature extraction, GRU temporal modeling, and attention.
    
    Attention helps identify critical phases in gesture execution.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 cnn_channels: list = None, gru_hidden: int = 128,
                 gru_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64]
        
        # CNN feature extractor
        cnn_layers = []
        prev_channels = in_channels
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(prev_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.5)
            ])
            prev_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1)
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_hidden * 2, gru_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(gru_hidden, num_classes)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # (batch, cnn_channels[-1], reduced_time)
        
        # Prepare for GRU
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # GRU temporal modeling
        gru_out, _ = self.gru(x)  # (batch, time, gru_hidden*2)
        
        # Attention
        attention_weights = self.attention(gru_out)  # (batch, time, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * gru_out, dim=1)  # (batch, gru_hidden*2)
        
        # Classifier
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x