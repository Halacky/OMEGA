"""
Conditional VAE for EMG Synthetic Subject Generation.

Disentangles gesture content from subject style in the latent space.
Generates "virtual subjects" by sampling novel style vectors while
conditioning on gesture identity.

Architecture:
  Encoder: (B, C, T) → CNN → [μ_content, logvar_content, μ_style, logvar_style]
  Decoder: [z_content, z_style, gesture_emb] → TransposedCNN → (B, C, T)

Losses:
  L = MSE(x, x_recon) + β_c * KL(q(z_c|x) || p(z_c)) + β_s * KL(q(z_s|x) || p(z_s))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class EMGConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for EMG signals with
    disentangled content (gesture) and style (subject) latent spaces.

    Args:
        in_channels: Number of EMG channels (e.g. 8 or 12).
        time_steps: Temporal length of each window (e.g. 400 or 600).
        num_classes: Number of gesture classes (for conditioning).
        content_dim: Dimensionality of content (gesture) latent space.
        style_dim: Dimensionality of style (subject) latent space.
        gesture_embed_dim: Dimensionality of gesture label embedding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        time_steps: int,
        num_classes: int,
        content_dim: int = 64,
        style_dim: int = 32,
        gesture_embed_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.gesture_embed_dim = gesture_embed_dim

        # Gesture label embedding (used in both encoder and decoder)
        self.gesture_embedding = nn.Embedding(num_classes, gesture_embed_dim)

        # ---- Encoder ----
        # Input: (B, C, T) → conv layers → flatten → latent heads
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        # Compute encoder output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, time_steps)
            enc_out = self.encoder(dummy)
            self._enc_channels = enc_out.shape[1]  # 256
            self._enc_time = enc_out.shape[2]       # T // 16 approx
            self._enc_flat = self._enc_channels * self._enc_time

        # Projection: encoder output + gesture_embedding → latent
        enc_input_dim = self._enc_flat + gesture_embed_dim

        # Content latent heads
        self.fc_mu_content = nn.Linear(enc_input_dim, content_dim)
        self.fc_logvar_content = nn.Linear(enc_input_dim, content_dim)

        # Style latent heads
        self.fc_mu_style = nn.Linear(enc_input_dim, style_dim)
        self.fc_logvar_style = nn.Linear(enc_input_dim, style_dim)

        # ---- Decoder ----
        # Input: z_content + z_style + gesture_embedding → transposed conv → (B, C, T)
        dec_input_dim = content_dim + style_dim + gesture_embed_dim
        self.fc_decode = nn.Linear(dec_input_dim, self._enc_flat)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

        # Adaptive layer to match exact time_steps (transposed convs may not be exact)
        self._decoder_out_time = self._enc_time * 16  # rough estimate
        # We'll use adaptive pooling / interpolation in forward

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(
        self, x: torch.Tensor, gesture_label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input into content and style latent distributions.

        Args:
            x: (B, C, T) EMG windows.
            gesture_label: (B,) integer gesture class indices.

        Returns:
            mu_c, logvar_c, mu_s, logvar_s — each (B, dim).
        """
        h = self.encoder(x)                         # (B, 256, T')
        h_flat = h.view(h.size(0), -1)              # (B, enc_flat)
        g_emb = self.gesture_embedding(gesture_label)  # (B, gesture_embed_dim)
        h_cond = torch.cat([h_flat, g_emb], dim=1)  # (B, enc_flat + embed)

        mu_c = self.fc_mu_content(h_cond)
        logvar_c = self.fc_logvar_content(h_cond)
        mu_s = self.fc_mu_style(h_cond)
        logvar_s = self.fc_logvar_style(h_cond)

        return mu_c, logvar_c, mu_s, logvar_s

    def decode(
        self,
        z_content: torch.Tensor,
        z_style: torch.Tensor,
        gesture_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent vectors back to EMG signal.

        Args:
            z_content: (B, content_dim)
            z_style: (B, style_dim)
            gesture_label: (B,) integer gesture class indices.

        Returns:
            x_recon: (B, C, T) reconstructed EMG.
        """
        g_emb = self.gesture_embedding(gesture_label)
        z_cat = torch.cat([z_content, z_style, g_emb], dim=1)

        h = self.fc_decode(z_cat)                    # (B, enc_flat)
        h = h.view(-1, self._enc_channels, self._enc_time)  # (B, 256, T')
        x_recon = self.decoder(h)                    # (B, C, T_approx)

        # Ensure exact time_steps via interpolation
        if x_recon.shape[2] != self.time_steps:
            x_recon = F.interpolate(
                x_recon, size=self.time_steps, mode="linear", align_corners=False
            )

        return x_recon

    def forward(
        self, x: torch.Tensor, gesture_label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → reparameterize → decode.

        Returns:
            x_recon, mu_c, logvar_c, mu_s, logvar_s
        """
        mu_c, logvar_c, mu_s, logvar_s = self.encode(x, gesture_label)
        z_c = self.reparameterize(mu_c, logvar_c)
        z_s = self.reparameterize(mu_s, logvar_s)
        x_recon = self.decode(z_c, z_s, gesture_label)
        return x_recon, mu_c, logvar_c, mu_s, logvar_s

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I)) per sample, summed over latent dims."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu_c: torch.Tensor,
        logvar_c: torch.Tensor,
        mu_s: torch.Tensor,
        logvar_s: torch.Tensor,
        beta_content: float = 1.0,
        beta_style: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss = Recon + β_c * KL_content + β_s * KL_style.

        Returns:
            total_loss, recon_loss, kl_content, kl_style — all scalar tensors.
        """
        recon = F.mse_loss(x_recon, x, reduction="mean")
        kl_c = self.kl_divergence(mu_c, logvar_c).mean()
        kl_s = self.kl_divergence(mu_s, logvar_s).mean()
        total = recon + beta_content * kl_c + beta_style * kl_s
        return total, recon, kl_c, kl_s

    @torch.no_grad()
    def generate(
        self,
        gesture_labels: torch.Tensor,
        n_per_label: int,
        device: torch.device,
        style_scale: float = 1.5,
    ) -> torch.Tensor:
        """
        Generate synthetic EMG windows by sampling from the prior.

        For each gesture label, generates n_per_label windows with random
        content and style vectors. Uses increased style_scale to encourage
        diverse synthetic subjects.

        Args:
            gesture_labels: (K,) unique gesture class indices.
            n_per_label: Number of windows to generate per gesture.
            device: Torch device.
            style_scale: Scaling factor for style prior std (>1 = more diverse).

        Returns:
            windows: (K * n_per_label, C, T) synthetic EMG data.
            labels: (K * n_per_label,) corresponding gesture labels.
        """
        self.eval()
        all_windows = []
        all_labels = []

        for g in gesture_labels:
            g_tensor = torch.full((n_per_label,), g.item(), dtype=torch.long, device=device)
            z_c = torch.randn(n_per_label, self.content_dim, device=device)
            z_s = torch.randn(n_per_label, self.style_dim, device=device) * style_scale
            x_gen = self.decode(z_c, z_s, g_tensor)
            all_windows.append(x_gen)
            all_labels.append(g_tensor)

        windows = torch.cat(all_windows, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return windows, labels

    @torch.no_grad()
    def generate_from_real_styles(
        self,
        x_real: torch.Tensor,
        y_real: torch.Tensor,
        n_synthetic_subjects: int,
        windows_per_gesture: int,
        device: torch.device,
        style_noise_scale: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic subjects by perturbing real subject styles.

        1. Encode all real data → extract style vectors per subject cluster.
        2. For each synthetic subject: sample a base style from real distribution,
           add noise to create a novel but plausible style.
        3. Decode with the new style + content sampled from prior, conditioned on gesture.

        Args:
            x_real: (N, C, T) real training windows.
            y_real: (N,) gesture labels (class indices, 0-based).
            n_synthetic_subjects: Number of virtual subjects to create.
            windows_per_gesture: Windows per gesture per virtual subject.
            device: Torch device.
            style_noise_scale: Noise std added to real style means.

        Returns:
            synthetic_windows: (M, C, T) generated data.
            synthetic_labels: (M,) gesture labels.
        """
        self.eval()

        # Encode all real data to get style distribution
        mu_c, logvar_c, mu_s, logvar_s = self.encode(x_real.to(device), y_real.to(device))
        real_style_mean = mu_s.mean(dim=0)   # (style_dim,)
        real_style_std = mu_s.std(dim=0)     # (style_dim,)

        unique_gestures = torch.unique(y_real)
        all_windows = []
        all_labels = []

        for subj_idx in range(n_synthetic_subjects):
            # Create a novel style vector for this synthetic subject
            style_base = real_style_mean + real_style_std * torch.randn_like(real_style_mean)
            style_noise = torch.randn_like(real_style_mean) * style_noise_scale
            subj_style = style_base + style_noise  # (style_dim,)

            for g in unique_gestures:
                g_val = g.item()
                g_tensor = torch.full(
                    (windows_per_gesture,), g_val, dtype=torch.long, device=device
                )
                # Content: sample from prior (gesture-independent)
                z_c = torch.randn(windows_per_gesture, self.content_dim, device=device)
                # Style: replicate this synthetic subject's style
                z_s = subj_style.unsqueeze(0).expand(windows_per_gesture, -1)

                x_gen = self.decode(z_c, z_s, g_tensor)
                all_windows.append(x_gen.cpu())
                all_labels.append(
                    torch.full((windows_per_gesture,), g_val, dtype=torch.long)
                )

        synthetic_windows = torch.cat(all_windows, dim=0)
        synthetic_labels = torch.cat(all_labels, dim=0)
        return synthetic_windows, synthetic_labels

    @torch.no_grad()
    def interpolate_styles(
        self,
        x_a: torch.Tensor,
        y_a: torch.Tensor,
        x_b: torch.Tensor,
        y_b: torch.Tensor,
        gesture_label: int,
        steps: int = 8,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Interpolate in style space between two sets of windows (two subjects).

        Args:
            x_a, x_b: (N_a, C, T) and (N_b, C, T) windows from two subjects.
            y_a, y_b: (N_a,) and (N_b,) labels.
            gesture_label: Gesture class to generate.
            steps: Number of interpolation steps.
            device: Torch device.

        Returns:
            interpolated: (steps, C, T) windows transitioning from style_a → style_b.
        """
        self.eval()

        # Encode both sets and average styles
        _, _, mu_s_a, _ = self.encode(x_a.to(device), y_a.to(device))
        _, _, mu_s_b, _ = self.encode(x_b.to(device), y_b.to(device))

        style_a = mu_s_a.mean(dim=0)  # (style_dim,)
        style_b = mu_s_b.mean(dim=0)

        alphas = torch.linspace(0, 1, steps, device=device)
        g_tensor = torch.full((steps,), gesture_label, dtype=torch.long, device=device)
        z_c = torch.randn(steps, self.content_dim, device=device) * 0.3  # low variance

        z_s = torch.stack([
            (1 - a) * style_a + a * style_b for a in alphas
        ])

        interpolated = self.decode(z_c, z_s, g_tensor)
        return interpolated.cpu()
