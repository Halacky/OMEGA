"""
FiLM Subject-Adaptive CNN-GRU-Attention model for EMG gesture recognition.

Hypothesis H2: Learnable Subject-Style Embedding with FiLM Conditioning.
Instead of removing subject variability, we parameterize it via a compact
style embedding z_subject, computed from a few reference windows.
FiLM layers condition the backbone: out = gamma(z) * BN(x) + beta(z).

Inspired by speaker embeddings in speech recognition, where speaker
embedding dramatically improves cross-speaker generalization.

Training:
- Pseudo-subject clusters (K-means on signal features) provide style groups
- Reference windows are sampled from the same cluster as each training sample
- Auxiliary loss encourages style encoder to capture subject-specific information

Test time:
- Compute z_subject once from K calibration windows of the test subject
- Reuse z_subject for all test windows (no subject ID required)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: out = gamma * x + beta."""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) feature map after BatchNorm
            gamma: (B, C) per-channel scale
            beta: (B, C) per-channel shift
        Returns:
            (B, C, T) modulated features
        """
        return gamma.unsqueeze(2) * x + beta.unsqueeze(2)


class StyleEncoder(nn.Module):
    """
    Computes a permutation-invariant style embedding from K reference windows.

    Architecture:
        Per-window CNN → GlobalAvgPool → (K, feat_dim)
        MeanPool over K → (feat_dim,)
        FC → z_subject (style_dim,)
    """

    def __init__(self, in_channels: int = 12, style_dim: int = 64):
        super().__init__()
        self.style_dim = style_dim

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B*K, 64, 1)
        )
        self.fc = nn.Linear(64, style_dim)

    def forward(self, ref_windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_windows: (B, K, C, T) — K reference windows per sample
                         OR (K, C, T) — single set of reference windows (test time)
        Returns:
            z_subject: (B, style_dim) or (style_dim,) style embedding
        """
        squeeze_batch = False
        if ref_windows.dim() == 3:
            # Single set: (K, C, T) → (1, K, C, T)
            ref_windows = ref_windows.unsqueeze(0)
            squeeze_batch = True

        B, K, C, T = ref_windows.shape
        # Process all windows through shared CNN
        x = ref_windows.reshape(B * K, C, T)  # (B*K, C, T)
        x = self.cnn(x)  # (B*K, 64, 1)
        x = x.squeeze(2)  # (B*K, 64)
        x = x.reshape(B, K, -1)  # (B, K, 64)

        # Permutation-invariant aggregation: mean over K
        x = x.mean(dim=1)  # (B, 64)
        z = self.fc(x)  # (B, style_dim)

        if squeeze_batch:
            z = z.squeeze(0)  # (style_dim,)
        return z


class FiLMGenerator(nn.Module):
    """
    Generates per-layer FiLM parameters (gamma, beta) from style embedding.
    Identity initialization: gamma=1, beta=0 at start (no modulation initially).
    """

    def __init__(self, style_dim: int, layer_channels: list):
        """
        Args:
            style_dim: dimension of z_subject
            layer_channels: list of channel counts for each FiLM layer, e.g. [32, 64, 128]
        """
        super().__init__()
        self.num_layers = len(layer_channels)
        self.layer_channels = layer_channels

        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(style_dim, 128),
            nn.ReLU(),
        )

        # Per-layer gamma and beta heads
        self.gamma_heads = nn.ModuleList()
        self.beta_heads = nn.ModuleList()
        for ch in layer_channels:
            gamma_head = nn.Linear(128, ch)
            beta_head = nn.Linear(128, ch)
            # Identity init: gamma → 1, beta → 0
            nn.init.ones_(gamma_head.bias)
            nn.init.zeros_(gamma_head.weight)
            nn.init.zeros_(beta_head.bias)
            nn.init.zeros_(beta_head.weight)
            self.gamma_heads.append(gamma_head)
            self.beta_heads.append(beta_head)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, style_dim) style embedding
        Returns:
            gammas: list of (B, C_i) tensors
            betas: list of (B, C_i) tensors
        """
        h = self.shared(z)  # (B, 128)
        gammas = [head(h) for head in self.gamma_heads]
        betas = [head(h) for head in self.beta_heads]
        return gammas, betas


class FiLMSubjectAdaptiveCNNGRU(nn.Module):
    """
    CNN-GRU-Attention backbone conditioned via FiLM layers using a subject-style embedding.

    Architecture:
        StyleEncoder(ref_windows) → z_subject
        FiLMGenerator(z_subject) → (γ₁,β₁), (γ₂,β₂), (γ₃,β₃)
        Conv1d → BN → FiLM₁ → ReLU → MaxPool
        Conv1d → BN → FiLM₂ → ReLU → MaxPool
        Conv1d → BN → FiLM₃ → ReLU
        BiGRU → Attention → FC → logits
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 10,
        dropout: float = 0.3,
        style_dim: int = 64,
        num_pseudo_clusters: int = 10,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.style_dim = style_dim
        self.num_pseudo_clusters = num_pseudo_clusters
        self.cnn_channels = cnn_channels

        # Style embedding
        self.style_encoder = StyleEncoder(in_channels, style_dim)
        self.film_generator = FiLMGenerator(style_dim, cnn_channels)

        # CNN backbone blocks
        self.conv1 = nn.Conv1d(in_channels, cnn_channels[0], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        self.film1 = FiLMLayer(cnn_channels[0])

        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        self.film2 = FiLMLayer(cnn_channels[1])

        self.conv3 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_channels[2])
        self.film3 = FiLMLayer(cnn_channels[2])

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        # BiGRU
        self.gru = nn.GRU(
            input_size=cnn_channels[2],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Attention pooling
        gru_out_dim = gru_hidden * 2  # bidirectional
        self.attn_fc = nn.Linear(gru_out_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Auxiliary cluster predictor (for training)
        self.cluster_predictor = nn.Linear(style_dim, num_pseudo_clusters)

        # Storage for auxiliary loss
        self._aux_loss = None

    def _attention_pool(self, gru_out: torch.Tensor) -> torch.Tensor:
        """
        Attention pooling over GRU output sequence.
        Args:
            gru_out: (B, T', gru_out_dim)
        Returns:
            (B, gru_out_dim) context vector
        """
        attn_weights = torch.softmax(self.attn_fc(gru_out), dim=1)  # (B, T', 1)
        context = (attn_weights * gru_out).sum(dim=1)  # (B, gru_out_dim)
        return context

    def forward(
        self,
        x: torch.Tensor,
        ref_windows: torch.Tensor = None,
        z_subject: torch.Tensor = None,
        cluster_labels: torch.Tensor = None,
        aux_loss_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) main EMG windows
            ref_windows: (B, K, C, T) reference windows for style extraction (training)
            z_subject: (B, style_dim) precomputed style embedding (test time)
            cluster_labels: (B,) pseudo-cluster labels for auxiliary loss (training)
            aux_loss_weight: weight for auxiliary cluster prediction loss
        Returns:
            logits: (B, num_classes)
        """
        B = x.size(0)
        device = x.device

        # Compute style embedding
        if z_subject is not None:
            # Precomputed (test time) — expand if needed
            if z_subject.dim() == 1:
                z = z_subject.unsqueeze(0).expand(B, -1)
            else:
                z = z_subject
        elif ref_windows is not None:
            z = self.style_encoder(ref_windows)  # (B, style_dim)
        else:
            # No conditioning — use zero embedding (identity FiLM)
            z = torch.zeros(B, self.style_dim, device=device)

        # Generate FiLM parameters
        gammas, betas = self.film_generator(z)

        # CNN backbone with FiLM conditioning
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.film1(h, gammas[0], betas[0])
        h = F.relu(h)
        h = self.pool(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.film2(h, gammas[1], betas[1])
        h = F.relu(h)
        h = self.pool(h)
        h = self.dropout(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.film3(h, gammas[2], betas[2])
        h = F.relu(h)
        h = self.dropout(h)

        # BiGRU: (B, C_out, T') → (B, T', C_out)
        h = h.permute(0, 2, 1)
        gru_out, _ = self.gru(h)  # (B, T', gru_out_dim)

        # Attention pooling
        context = self._attention_pool(gru_out)  # (B, gru_out_dim)

        # Classification
        logits = self.classifier(context)  # (B, num_classes)

        # Auxiliary loss: cluster prediction from style embedding
        self._aux_loss = None
        if cluster_labels is not None and self.training:
            cluster_logits = self.cluster_predictor(z)  # (B, num_clusters)
            self._aux_loss = aux_loss_weight * F.cross_entropy(cluster_logits, cluster_labels)

        return logits
