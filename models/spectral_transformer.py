"""
Spectral Transformer for EMG gesture classification.

Hypothesis H3: Self-attention over frequency bands + channels > CNN-RNN.
STFT transforms raw EMG to time-frequency representation, then axial
attention (time + channel) with relative positional encoding models
long-range dependencies that CNNs miss.

Architecture:
    Input (B, C, T) raw EMG
    → STFT per channel → log-magnitude spectrogram (B*C, F, T')
    → Conv2d embedding + pool over frequency → (B, C, T', d_model)
    → N axial attention blocks (time-attn + channel-attn + FFN)
    → global pool → MLP classifier → (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head self-attention with learnable relative position bias (Swin-style)."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Relative position bias table: (2*max_seq_len - 1) possible offsets
        self.max_seq_len = max_seq_len
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_seq_len - 1, n_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index for max_seq_len
        coords = torch.arange(max_seq_len)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # (L, L)
        relative_coords += max_seq_len - 1  # shift to [0, 2*L-2]
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, d_k)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, L, L)

        # Add relative position bias
        rel_idx = self.relative_position_index[:L, :L]  # (L, L)
        rel_bias = self.relative_position_bias_table[rel_idx]  # (L, L, heads)
        attn = attn + rel_bias.permute(2, 0, 1).unsqueeze(0)  # broadcast over B

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SpectralEmbedding(nn.Module):
    """STFT + Conv2d embedding of magnitude spectrogram."""

    def __init__(self, d_model: int = 64, n_fft: int = 64, hop_length: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EMG
        Returns:
            (B, C, T', d_model) spectral embeddings
        """
        B, C, T = x.shape
        xr = x.reshape(B * C, T).contiguous()

        stft_out = torch.stft(
            xr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        spec = torch.log1p(stft_out.abs())  # (B*C, F, T')

        h = self.conv(spec.unsqueeze(1))  # (B*C, d_model, F, T')
        h = h.mean(dim=2)  # pool over frequency → (B*C, d_model, T')
        h = h.permute(0, 2, 1)  # (B*C, T', d_model)

        Tp = h.shape[1]
        h = h.reshape(B, C, Tp, -1)  # (B, C, T', d_model)
        return h


class SpectralAxialBlock(nn.Module):
    """One layer of axial attention: time-attn + channel-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int, max_time_len: int,
                 n_channels: int, ffn_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        # Time attention (pre-norm)
        self.norm_t = nn.LayerNorm(d_model)
        self.time_attn = RelativeMultiHeadAttention(
            d_model, n_heads, max_seq_len=max_time_len, dropout=dropout,
        )

        # Channel attention (pre-norm)
        self.norm_c = nn.LayerNorm(d_model)
        self.channel_attn = RelativeMultiHeadAttention(
            d_model, n_heads, max_seq_len=n_channels, dropout=dropout,
        )

        # FFN (pre-norm)
        self.norm_f = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T', d_model)
        Returns:
            (B, C, T', d_model)
        """
        B, C, Tp, D = x.shape

        # Time attention: attend across T' for each channel
        xt = self.norm_t(x).reshape(B * C, Tp, D)
        xt = self.time_attn(xt).reshape(B, C, Tp, D)
        x = x + xt

        # Channel attention: attend across C for each time step
        xc = self.norm_c(x).permute(0, 2, 1, 3).reshape(B * Tp, C, D)
        xc = self.channel_attn(xc).reshape(B, Tp, C, D).permute(0, 2, 1, 3)
        x = x + xc

        # FFN
        x = x + self.ffn(self.norm_f(x))

        return x


class SpectralTransformer(nn.Module):
    """
    Spectral Transformer for EMG gesture classification.

    STFT → Conv2d spectral embedding → axial attention (time + channel)
    with relative positional encoding → global pool → classifier.

    Args:
        in_channels: number of EMG channels
        num_classes: number of gesture classes
        dropout: dropout rate
        d_model: transformer hidden dimension
        n_heads: number of attention heads
        n_layers: number of axial attention blocks
        n_fft: STFT window size
        hop_length: STFT hop size
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_fft: int = 64, hop_length: int = 32, **kwargs):
        super().__init__()

        self.spectral_embed = SpectralEmbedding(d_model, n_fft, hop_length)

        self.layers = nn.ModuleList([
            SpectralAxialBlock(
                d_model=d_model,
                n_heads=n_heads,
                max_time_len=64,  # safe upper bound for T'
                n_channels=in_channels,
                dropout=dropout * 0.5,  # lighter dropout in attention
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw standardized EMG
        Returns:
            (B, num_classes) logits
        """
        h = self.spectral_embed(x)  # (B, C, T', d_model)

        for layer in self.layers:
            h = layer(h)  # (B, C, T', d_model)

        h = self.final_norm(h)
        h = h.mean(dim=2)  # pool over time → (B, C, d_model)
        h = h.mean(dim=1)  # pool over channels → (B, d_model)

        return self.classifier(h)
