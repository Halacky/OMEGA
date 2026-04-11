"""
Latent Diffusion Model for Subject-Style Removal in EMG.

Trains a denoising diffusion process in the encoder's latent space.
The diffusion process learns to map noisy subject-specific latents
to a clean, canonical (subject-invariant) representation.

Architecture:
    Input (B, C, T) -> CNN-GRU Encoder -> z (B, hidden_dim)
    z -> Gesture Classifier -> logits (standard path)
    z + noise -> Denoiser(z_noisy, t) -> z_denoised (diffusion path)

Training:
    L = L_gesture + lambda_diff * L_diffusion
    L_diffusion = ||denoiser(z + eps, t) - eps||^2 (predict noise)

Inference:
    Encode -> iterative denoising (T steps) -> classify denoised latent
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal time embeddings for diffusion timestep.
    Args: t (B,) float timesteps, dim: embedding dimension
    Returns: (B, dim)
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
    )
    args = t.unsqueeze(1) * freqs.unsqueeze(0)  # (B, half_dim)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class Denoiser(nn.Module):
    """
    MLP denoiser that predicts noise given noisy latent + timestep.

    Takes z_noisy (B, D) and timestep embedding (B, D),
    predicts the added noise eps (B, D).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_noisy: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_noisy: (B, D) noisy latent
            t_embed: (B, D) sinusoidal time embedding
        Returns: (B, D) predicted noise
        """
        t_h = self.time_embed(t_embed)
        h = torch.cat([z_noisy, t_h], dim=1)
        return self.net(h)


class LatentDiffusionEMG(nn.Module):
    """
    EMG model with latent diffusion for subject-style removal.

    Args:
        in_channels: Number of EMG channels.
        num_classes: Number of gesture classes.
        dropout: Dropout rate.
        hidden_dim: Latent dimension.
        n_diffusion_steps: Number of diffusion timesteps.
        beta_start: Starting beta for linear schedule.
        beta_end: Ending beta for linear schedule.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        hidden_dim: int = 128,
        n_diffusion_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_diffusion_steps = n_diffusion_steps

        # ===== CNN-GRU Encoder =====
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.encoder_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ===== Gesture Classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # ===== Denoiser =====
        self.denoiser = Denoiser(hidden_dim, hidden_dim * 2)

        # ===== Diffusion schedule (linear beta) =====
        betas = torch.linspace(beta_start, beta_end, n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode EMG input to latent.
        Args: x (B, C, T)
        Returns: z (B, hidden_dim)
        """
        h = self.cnn(x)                       # (B, hidden_dim, T')
        h = h.permute(0, 2, 1)                # (B, T', hidden_dim)
        gru_out, _ = self.gru(h)              # (B, T', hidden_dim)
        pooled = gru_out.mean(dim=1)          # (B, hidden_dim)
        z = self.encoder_proj(pooled)         # (B, hidden_dim)
        return z

    def diffusion_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion training loss: predict noise at random timestep.

        Args: z (B, D) clean latent
        Returns: scalar MSE loss
        """
        B = z.size(0)
        device = z.device

        # Sample random timesteps
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=device)

        # Sample noise
        eps = torch.randn_like(z)

        # Create noisy latent: z_t = sqrt(alpha_bar_t) * z + sqrt(1 - alpha_bar_t) * eps
        sqrt_ab = self.sqrt_alpha_cumprod[t].unsqueeze(1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_cumprod[t].unsqueeze(1)
        z_noisy = sqrt_ab * z + sqrt_1_ab * eps

        # Time embedding
        t_embed = sinusoidal_embedding(t.float(), self.hidden_dim)

        # Predict noise
        eps_pred = self.denoiser(z_noisy, t_embed)

        # MSE loss
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def denoise(self, z: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """
        Iterative denoising: z_T -> z_0 (canonical latent).

        For inference, we add noise to z and then denoise,
        which removes subject-specific style while preserving content.
        """
        if n_steps is None:
            n_steps = self.n_diffusion_steps

        B = z.size(0)
        device = z.device

        # Start from noisy version of z
        z_t = z + torch.randn_like(z) * 0.5  # partial noise

        for i in reversed(range(n_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_embed = sinusoidal_embedding(t.float(), self.hidden_dim)

            eps_pred = self.denoiser(z_t, t_embed)

            # DDPM reverse step
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alpha_cumprod[i]
            beta_t = self.betas[i]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                z_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )

            if i > 0:
                noise = torch.randn_like(z_t)
                sigma = torch.sqrt(beta_t)
                z_t = mean + sigma * noise
            else:
                z_t = mean

        return z_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward: encode -> classify (no denoising at forward time).
        Args: x (B, C, T)
        Returns: logits (B, num_classes)
        """
        z = self.encode(x)
        logits = self.classifier(z)
        return logits

    def forward_denoised(self, x: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """
        Inference with denoising: encode -> denoise -> classify.
        Uses fewer steps than training for efficiency.
        """
        z = self.encode(x)
        z_clean = self.denoise(z, n_steps=n_steps)
        logits = self.classifier(z_clean)
        return logits

    def forward_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward returning all components for training."""
        z = self.encode(x)
        logits = self.classifier(z)
        diff_loss = self.diffusion_loss(z)
        return {
            "logits": logits,
            "z": z,
            "diffusion_loss": diff_loss,
        }
