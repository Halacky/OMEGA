import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveFeatureDisentanglementWithUnsupervisedPersonalization(nn.Module):
    """
    Combines contrastive learning and adversarial feature disentanglement for EMG gesture recognition.
    This model uses a shared CNN feature extractor, followed by branches for pattern-specific and
    subject-specific features. It is trained with contrastive learning for cross-subject alignment
    and adversarial loss for disentanglement, then fine-tuned with pseudo-labels in a LOSO setting.
    """

    def __init__(self, in_channels: int, num_classes: int, sequence_length: int = 200,
                 dropout: float = 0.5, hidden_size: int = 64, lambda_adversarial: float = 0.1, **kwargs):
        """
        Initializes the ContrastiveFeatureDisentanglementWithUnsupervisedPersonalization model.

        Args:
            in_channels (int): Number of input channels (e.g., EMG channels).
            num_classes (int): Number of gesture classes.
            sequence_length (int): Length of the input sequence.
            dropout (float): Dropout probability.
            hidden_size (int): Size of the hidden layers.
            lambda_adversarial (float): Weight for the adversarial loss.
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout_rate = dropout
        self.hidden_size = hidden_size
        self.lambda_adversarial = lambda_adversarial

        # Shared CNN feature extractor
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Pattern-specific branch
        self.pattern_fc1 = nn.Linear(128, hidden_size)
        self.pattern_bn1 = nn.BatchNorm1d(hidden_size)
        self.pattern_dropout = nn.Dropout(dropout)
        self.pattern_fc2 = nn.Linear(hidden_size, num_classes)

        # Subject-specific branch
        self.subject_fc1 = nn.Linear(128, hidden_size)
        self.subject_bn1 = nn.BatchNorm1d(hidden_size)
        self.subject_dropout = nn.Dropout(dropout)
        self.subject_fc2 = nn.Linear(hidden_size, hidden_size)  # Subject embedding

        # Adversarial discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ContrastiveFeatureDisentanglementWithUnsupervisedPersonalization model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_classes).
        """
        # Shared feature extractor
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 128)

        # Pattern-specific branch
        pattern_features = F.relu(self.pattern_bn1(self.pattern_fc1(x)))
        pattern_features = self.pattern_dropout(pattern_features)
        pattern_output = self.pattern_fc2(pattern_features)  # (batch, num_classes)

        # Subject-specific branch
        subject_features = F.relu(self.subject_bn1(self.subject_fc1(x)))
        subject_features = self.subject_dropout(subject_features)
        subject_embedding = self.subject_fc2(subject_features)  # (batch, hidden_size)

        # Adversarial discriminator (for training, not used in inference)
        # The discriminator is used during training to disentangle features.
        # During inference, we only use the pattern_output for classification.

        return pattern_output

    def get_subject_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the subject embedding for a given input.  Useful for personalization tasks.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, sequence_length).

        Returns:
            torch.Tensor: Subject embedding of shape (batch, hidden_size).
        """
        # Shared feature extractor
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)

        # Subject-specific branch
        subject_features = F.relu(self.subject_bn1(self.subject_fc1(x)))
        subject_features = self.subject_dropout(subject_features)
        subject_embedding = self.subject_fc2(subject_features)

        return subject_embedding