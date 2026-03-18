import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiorientationFeatureEnsembleWithAdversarialDisentanglement(nn.Module):
    """
    Ensemble model for EMG gesture recognition using multiple feature extraction methods
    and adversarial disentanglement to separate orientation-invariant and gesture-specific features.
    """

    def __init__(self, in_channels: int, num_classes: int, sequence_length: int = 200,
                 feature_dim: int = 64, ensemble_size: int = 3, dropout: float = 0.5,
                 num_orientation: int = 4,  # Number of orientations for adversarial disentanglement
                 **kwargs):
        """
        Initializes the MultiorientationFeatureEnsembleWithAdversarialDisentanglement model.

        Args:
            in_channels (int): Number of input channels (e.g., EMG electrodes).
            num_classes (int): Number of gesture classes.
            sequence_length (int): Length of the input sequence.
            feature_dim (int): Dimension of the extracted features.
            ensemble_size (int): Number of models in the ensemble.
            dropout (float): Dropout probability.
            num_orientation (int): Number of orientations for adversarial disentanglement.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.ensemble_size = ensemble_size
        self.dropout = dropout
        self.num_orientation = num_orientation

        # Shared Encoder (e.g., CNN)
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(in_channels, feature_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(feature_dim, feature_dim * 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

        # Calculate the output size of the shared encoder
        test_input = torch.randn(1, in_channels, sequence_length)
        with torch.no_grad():
            encoder_output = self.shared_encoder(test_input)
            # Get the shape after convolutions: (batch, channels, length)
            self.encoder_output_channels = encoder_output.shape[1]
            self.encoder_output_length = encoder_output.shape[2]
            self.encoder_output_size = self.encoder_output_channels * self.encoder_output_length

        # Flatten layer
        self.flatten = nn.Flatten()

        # Gesture Classifier Heads (Ensemble)
        self.classifier_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.encoder_output_size, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            ) for _ in range(ensemble_size)
        ])

        # Orientation Discriminator
        self.orientation_discriminator = nn.Sequential(
            nn.Linear(self.encoder_output_size, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_orientation)
        )

        # Initialize weights (optional)
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, orientation_labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the MultiorientationFeatureEnsembleWithAdversarialDisentanglement model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, sequence_length).
            orientation_labels (torch.Tensor, optional): Orientation labels for adversarial training.
                Shape (batch,). Defaults to None.

        Returns:
            torch.Tensor: Ensemble output (batch, num_classes).
        """
        # Shared Encoder
        encoded_features = self.shared_encoder(x)
        
        # Flatten the output
        encoded_features = self.flatten(encoded_features)

        # Gesture Classification
        ensemble_outputs = [head(encoded_features) for head in self.classifier_heads]
        # Average the ensemble outputs
        ensemble_output = torch.mean(torch.stack(ensemble_outputs), dim=0)

        # Adversarial Disentanglement (if orientation labels are provided)
        if orientation_labels is not None:
            orientation_predictions = self.orientation_discriminator(encoded_features)
            # Calculate adversarial loss (e.g., cross-entropy)
            adversarial_loss = F.cross_entropy(orientation_predictions, orientation_labels)
            # This loss should be used during training to update the shared encoder
            # to remove orientation-specific information.
            self.adversarial_loss = adversarial_loss  # Store for logging

        return ensemble_output