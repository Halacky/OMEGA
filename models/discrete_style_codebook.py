"""
Discrete Style Codebook with EMA Quantization for EMG Gesture Recognition.

Hypothesis: Replace continuous style space with discrete VQ codebook (K=16-32
entries). Styles are quantized via EMA-updated codebook. During training,
random code replacement (p=0.5) forces content features to be style-invariant.
At inference, only content features are used (no VQ/FiLM).

Architecture:
    Input (B, C, T) → SharedEncoder (CNN+BiGRU+Attention) → shared (B, 256)
      ├─ ContentProjection → z_content (B, 128)
      └─ StyleProjection   → z_style   (B, 64) → VQ → c_k ∈ {c_1,...,c_K}

    Training:
      c_sampled = c_k or random code (p=0.5 per sample)
      FiLM(z_content, c_sampled) → GestureClassifier → logits
      Loss: L_cls + L_vq_commit + λ_div * L_diversity

    Inference:
      z_content → GestureClassifier → logits  (no VQ/FiLM)

LOSO-safe: no subject-specific adaptation at inference.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vq_disentangle_emg import VectorQuantizerEMA


# ──────────────────────── Encoder ────────────────────────


class _SharedEncoder(nn.Module):
    """CNN → BiGRU → Attention pooling.  Same topology as DisentangledCNNGRU."""

    def __init__(
        self,
        in_channels: int = 8,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        layers = []
        prev_ch = in_channels
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.5),
            ])
            prev_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        self.gru_output_dim = gru_hidden * 2

        self.attention = nn.Sequential(
            nn.Linear(self.gru_output_dim, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, gru_hidden*2)"""
        h = self.cnn(x)
        h = h.transpose(1, 2)
        gru_out, _ = self.gru(h)
        attn_w = torch.softmax(self.attention(gru_out), dim=1)
        context = (attn_w * gru_out).sum(dim=1)
        return context


# ──────────────────────── Model ──────────────────────────


class DiscreteStyleCodebookModel(nn.Module):
    """
    Content-style disentanglement via discrete VQ codebook for style.

    During training the style code (possibly replaced with a random codebook
    entry) modulates content via FiLM.  During inference only z_content is
    used, making prediction fully subject-agnostic.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        dropout: float = 0.3,
        content_dim: int = 128,
        style_dim: int = 64,
        style_codebook_size: int = 32,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        style_augment_prob: float = 0.5,
        diversity_weight: float = 0.1,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
    ):
        super().__init__()

        self.content_dim = content_dim
        self.style_dim = style_dim
        self.style_codebook_size = style_codebook_size
        self.style_augment_prob = style_augment_prob
        self.diversity_weight = diversity_weight

        # ── shared encoder ──
        self.encoder = _SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # 256

        # ── content branch ──
        self.content_proj = nn.Sequential(
            nn.Linear(shared_dim, content_dim),
            nn.LayerNorm(content_dim),
            nn.ReLU(),
        )

        # ── style branch ──
        self.style_proj = nn.Sequential(
            nn.Linear(shared_dim, style_dim),
            nn.LayerNorm(style_dim),
            nn.ReLU(),
        )

        # ── VQ codebook for style ──
        self.vq_style = VectorQuantizerEMA(
            num_embeddings=style_codebook_size,
            embedding_dim=style_dim,
            commitment_cost=commitment_cost,
            decay=ema_decay,
        )

        # ── FiLM: style code → (γ, β) for content modulation ──
        # Identity init: γ=1, β=0 so initially FiLM is a no-op.
        self.film_gamma = nn.Linear(style_dim, content_dim)
        self.film_beta = nn.Linear(style_dim, content_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # ── gesture classifier (operates on content_dim) ──
        self.classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.LayerNorm(content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_classes),
        )

        # loss / info storage (set each forward pass)
        self.auxiliary_losses: Dict[str, torch.Tensor] = {}
        self.quantization_info: Dict = {}

    # ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            logits: (B, num_classes)
        """
        B = x.size(0)

        # ── encode ──
        shared = self.encoder(x)                      # (B, 256)
        z_content = self.content_proj(shared)          # (B, content_dim)
        z_style = self.style_proj(shared)              # (B, style_dim)

        if self.training:
            # ── quantize style ──
            c_k, commitment_loss, vq_info = self.vq_style(z_style)

            # ── per-sample random code replacement ──
            mask = (torch.rand(B, device=x.device) < self.style_augment_prob)
            random_idx = torch.randint(
                0, self.style_codebook_size, (B,), device=x.device
            )
            random_codes = self.vq_style.embeddings[random_idx].detach()
            c_sampled = torch.where(mask.unsqueeze(1), random_codes, c_k)

            # ── FiLM conditioning ──
            gamma = self.film_gamma(c_sampled)         # (B, content_dim)
            beta = self.film_beta(c_sampled)           # (B, content_dim)
            z_modulated = gamma * z_content + beta

            logits = self.classifier(z_modulated)

            # ── auxiliary losses ──
            diversity_loss = self._diversity_loss(vq_info["encodings"])
            self.auxiliary_losses = {
                "commitment_loss": commitment_loss,
                "diversity_loss": diversity_loss,
                "total_aux_loss": commitment_loss + self.diversity_weight * diversity_loss,
            }
            self.quantization_info = {
                "style_perplexity": vq_info["perplexity"],
                "style_indices": vq_info["encoding_indices"],
            }
            return logits

        # ── inference: content only, no VQ / FiLM ──
        logits = self.classifier(z_content)
        return logits

    # ────────────────────────────────────────────────────

    def _diversity_loss(self, encodings: torch.Tensor) -> torch.Tensor:
        """Negative normalised entropy — minimise to encourage uniform codebook usage."""
        probs = encodings.mean(dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return -entropy / math.log(encodings.shape[1])

    def get_auxiliary_loss(self) -> torch.Tensor:
        return self.auxiliary_losses.get(
            "total_aux_loss",
            torch.tensor(0.0, device=next(self.parameters()).device),
        )

    def reset_unused_codebooks(self, threshold: float = 0.001) -> Dict[str, int]:
        return {"style_codes_reset": self.vq_style.reset_unused_codes(threshold)}

    def get_codebook_stats(self) -> Dict:
        return {"style": self.vq_style.get_usage_stats()}
