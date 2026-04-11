import torch
import torch.nn as nn
import torch.nn.functional as F


class SyntheticDataAugmentationWithSeqemgganForLosoTraining(nn.Module):
    """
    Synthetic Data Augmentation with SeqEMG-GAN for LOSO Training.

    This model uses a simplified SeqEMG-GAN approach to generate synthetic EMG data
    conditioned on gesture labels. It includes a generator and discriminator,
    and is designed to augment training data for each user in a Leave-One-Subject-Out (LOSO)
    protocol to increase data diversity and reduce overfitting.

    Args:
        in_channels (int): Number of input EMG channels.
        num_classes (int): Number of gesture classes.
        sequence_length (int): Length of the input EMG sequence (default: 200).
        dropout (float): Dropout probability (default: 0.5).
        latent_dim (int): Dimension of the latent space for the generator (default: 100).
        generator_hidden_dim (int): Hidden dimension for the generator (default: 128).
        discriminator_hidden_dim (int): Hidden dimension for the discriminator (default: 128).
    """

    def __init__(self, in_channels: int, num_classes: int,
                 sequence_length: int = 200, dropout: float = 0.5,
                 latent_dim: int = 100, generator_hidden_dim: int = 128,
                 discriminator_hidden_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + num_classes, generator_hidden_dim),
            nn.ReLU(),
            nn.Linear(generator_hidden_dim, generator_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(generator_hidden_dim * 2, in_channels * sequence_length),
            nn.Tanh()  # Output range [-1, 1]
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(in_channels * sequence_length + num_classes, discriminator_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(discriminator_hidden_dim * 2, discriminator_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(discriminator_hidden_dim, 1),
            nn.Sigmoid()  # Output probability (real or fake)
        )

        # Classifier (for downstream task)
        # Calculate the flattened input size dynamically
        self.flattened_size = in_channels * sequence_length
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, generator_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(generator_hidden_dim, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classifier.

        Args:
            x (torch.Tensor): Input EMG data of shape (batch, channels, sequence_length).

        Returns:
            torch.Tensor: Classification output of shape (batch, num_classes).
        """
        batch_size = x.size(0)
        # Flatten to (batch, channels * sequence_length)
        x = x.reshape(batch_size, -1)
        output = self.classifier(x)
        return output

    def generate_synthetic_data(self, num_samples: int, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generates synthetic EMG data conditioned on gesture labels.

        Args:
            num_samples (int): Number of synthetic samples to generate.
            labels (torch.Tensor): Gesture labels for conditioning (one-hot encoded).
            device (torch.device): Device to run the generation on.

        Returns:
            torch.Tensor: Synthetic EMG data of shape (num_samples, in_channels, sequence_length).
        """
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=device)
            # Concatenate noise and labels
            gen_input = torch.cat((noise, labels), dim=1)
            # Generate data
            synthetic_data = self.generator(gen_input)
            synthetic_data = synthetic_data.view(num_samples, self.in_channels, self.sequence_length)
        return synthetic_data

    def discriminate(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Discriminates between real and synthetic EMG data.

        Args:
            x (torch.Tensor): Input EMG data of shape (batch, channels, sequence_length).
            labels (torch.Tensor): Gesture labels for conditioning (one-hot encoded).

        Returns:
            torch.Tensor: Probability of the input being real (batch, 1).
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        # Concatenate data and labels
        disc_input = torch.cat((x, labels), dim=1)
        # Discriminate
        validity = self.discriminator(disc_input)
        return validity