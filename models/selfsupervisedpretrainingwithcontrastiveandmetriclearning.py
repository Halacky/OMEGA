import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfsupervisedPretrainingWithContrastiveAndMetricLearning(nn.Module):
    """
    Combines contrastive learning and metric learning for self-supervised pre-training,
    followed by fine-tuning for EMG gesture classification.
    """

    def __init__(self, in_channels: int, num_classes: int, sequence_length: int = 200,
                 dropout: float = 0.5, embedding_size: int = 128, num_filters: int = 64):
        """
        Initializes the SelfsupervisedPretrainingWithContrastiveAndMetricLearning model.

        Args:
            in_channels (int): Number of input channels (EMG sensors).
            num_classes (int): Number of gesture classes.
            sequence_length (int): Length of the input sequence.
            dropout (float): Dropout probability.
            embedding_size (int): Size of the embedding vector for contrastive and metric learning.
            num_filters (int): Number of filters in the convolutional layers.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_filters = num_filters

        # Encoder (CNN)
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        # Adaptive average pooling to reduce sequence length
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Embedding layer
        self.fc_embedding = nn.Linear(num_filters * 4, embedding_size)

        # Classification layer
        self.fc_classifier = nn.Linear(embedding_size, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the convolutional and linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SelfsupervisedPretrainingWithContrastiveAndMetricLearning model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_classes).
        """
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Adaptive average pooling
        x = self.avgpool(x)  # Shape: (batch, num_filters * 4, 1)
        x = torch.flatten(x, 1)  # Shape: (batch, num_filters * 4)

        # Embedding
        embedding = self.fc_embedding(x)  # Shape: (batch, embedding_size)

        # Classification
        output = self.fc_classifier(embedding)  # Shape: (batch, num_classes)

        return output