import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for EMG gesture recognition.
    LSTMs are well-suited for temporal sequence modeling and can capture
    long-term dependencies in EMG signals.
    
    Bidirectional processing allows the model to use both past and future context,
    which is appropriate for offline gesture recognition.
    """
    def __init__(self, in_channels: int, num_classes: int, 
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
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
        # LSTM expects: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, time, channels)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)
        
        # Use the last time step output
        lstm_out = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Classifier
        x = self.dropout(lstm_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class BiLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with attention mechanism.
    Attention allows the model to focus on important time steps,
    which can be crucial for distinguishing gesture phases.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, time, channels)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, time, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classifier
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x