"""
TCN-GAT Hybrid: Multi-scale Temporal Convolution + Channel Graph Attention Network.

Hypothesis H43:
    Combining local multi-scale temporal feature extraction (causal dilated TCN)
    with global inter-muscular co-activation modelling (GAT over channel graph)
    captures richer subject-invariant representations than either approach alone
    or simple CNN-based graph networks.

Architecture:
    Input (B, C, T)  — raw standardised EMG, C=channels, T=window timesteps
    ↓
    PerChannelDilatedTCN    : dilated causal TCN per channel (weight-shared),
                              dilation ∈ {1, 2, 4, 8}, kernel_size=7
                              → (B, C, T', d_tcn)  where T' = T // pool_factor
    ↓
    DynamicAdjacency        : Pearson corr (feature space)
                              + learnable symmetric prior
                              → (B, C, C) adjacency bias
    ↓
    GATLayerExtractable ×n  : multi-head GAT at every time step
                              → (B, C, T', d_node), attention (B·T', heads, C, C)
    ↓
    Per-channel BiGRU       : temporal modelling per electrode
                              → (B, C, T', d_gru·2)
    ↓
    TemporalAttention       : soft attention over T' time steps (per channel)
                              → (B, C, d_gru·2)
    ↓
    ChannelAttention        : soft attention over C channels
                              → (B, d_gru·2)
    ↓
    MLP Classifier          → (B, num_classes)

Key innovations vs exp_37 (ChannelGATGRU):
    1. **Causal dilated TCN** instead of plain CNN — captures temporal causality
       and multi-scale dynamics (ms to 50ms receptive field) via dilation {1,2,4,8}.
    2. **Temporal attention** in addition to channel attention — the model learns
       WHICH time slice within the window is most informative.
    3. **Simplified adjacency** (feature Pearson + learnable prior, no spectral
       coherence) — cleaner inductive bias for inter-channel coupling.
    4. `forward_with_attention()` returns full interpretability dict for rich
       post-hoc analysis and visualisation.

Pure PyTorch — no torch_geometric dependency.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------

class CausalTCNBlock(nn.Module):
    """
    Dilated causal residual TCN block for 1-D sequence modelling.

    Causal: output at time t depends only on inputs at times ≤ t.
    Implemented via symmetric padding + chomping the right-side excess.

    Effective receptive field per block: 1 + 2 * (kernel_size - 1) * dilation.

    Input  : (N, in_ch, T)
    Output : (N, out_ch, T)  — same length as input
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # amount to chomp on right

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # 1×1 projection for residual if channel dims differ
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        """Remove right-side excess padding to maintain causal property."""
        if self.padding > 0:
            return x[:, :, :-self.padding].contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._chomp(self.conv1(x))
        h = self.act(self.bn1(h))
        h = self.drop(h)
        h = self._chomp(self.conv2(h))
        h = self.bn2(h)
        res = x if self.proj is None else self.proj(x)
        return self.act(h + res)


class PerChannelDilatedTCN(nn.Module):
    """
    Multi-scale dilated causal TCN applied independently to each EMG channel
    (weights are shared across channels — electrode-agnostic feature extraction).

    Dilation schedule: {1, 2, 4, 8}
    Effective receptive field (kernel=7):
        d=1:  RF = 13 ts  ≈  6.5 ms  @ 2 kHz
        d=2:  RF = 25 ts  ≈ 12.5 ms
        d=4:  RF = 49 ts  ≈ 24.5 ms
        d=8:  RF = 97 ts  ≈ 48.5 ms  (covers ~50% of 200ms window)
    Combined stacked RF: ≈ 184 timesteps (nearly the full 200ms gesture window).

    Input  : (B, C, T)
    Output : (B, C, T', d_tcn)  where T' = T // pool_factor
    """

    DILATIONS = [1, 2, 4, 8]

    def __init__(
        self,
        d_tcn: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.1,
        pool_factor: int = 4,
    ):
        super().__init__()

        channel_dims = [32, d_tcn, d_tcn, d_tcn]  # 4 blocks
        blocks = []
        in_ch = 1
        for out_ch, dil in zip(channel_dims, self.DILATIONS):
            blocks.append(
                CausalTCNBlock(in_ch, out_ch, kernel_size, dil, dropout)
            )
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool1d(pool_factor)
        self.pool_factor = pool_factor
        self.d_tcn = d_tcn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        h = x.reshape(B * C, 1, T)          # (B·C, 1, T)
        h = self.blocks(h)                   # (B·C, d_tcn, T)
        h = self.pool(h)                     # (B·C, d_tcn, T')
        T_prime = h.shape[-1]
        h = h.permute(0, 2, 1)              # (B·C, T', d_tcn)
        return h.reshape(B, C, T_prime, -1)  # (B, C, T', d_tcn)

    @property
    def receptive_fields(self) -> Dict[int, int]:
        """Cumulative receptive field of each block (in timesteps)."""
        rf = {}
        cumulative = 1
        ks = 7  # hard-coded to match default
        for i, d in enumerate(self.DILATIONS):
            block_rf = 1 + 2 * (ks - 1) * d
            cumulative = cumulative + block_rf - 1
            rf[i] = cumulative
        return rf


# ---------------------------------------------------------------------------
# Dynamic Adjacency
# ---------------------------------------------------------------------------

class DynamicAdjacency(nn.Module):
    """
    Constructs an inter-channel adjacency matrix from:
      1. Pearson correlation of time-averaged TCN node features (feature space).
      2. A learnable symmetric prior (optimised jointly with the rest of the model).

    The two components are combined via softmax-normalised learnable log-weights.

    Args:
        n_channels: number of EMG channels (graph nodes).
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.adj_prior = nn.Parameter(torch.zeros(n_channels, n_channels))
        nn.init.xavier_uniform_(self.adj_prior.unsqueeze(0)).squeeze(0)
        # log-weights for [feat_corr, prior]
        self.log_weights = nn.Parameter(torch.zeros(2))

    def forward(
        self, h_avg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_avg : (B, C, d_tcn)  time-averaged TCN features.

        Returns:
            adj          : (B, C, C)  adjacency bias tensor.
            feat_corr    : (B, C, C)  Pearson correlation component.
            mix_weights  : (2,)       softmax mixture weights [corr, prior].
        """
        # Feature-space Pearson correlation
        h_c = h_avg - h_avg.mean(dim=-1, keepdim=True)
        h_n = h_c / (h_c.norm(dim=-1, keepdim=True) + 1e-8)
        feat_corr = torch.bmm(h_n, h_n.transpose(1, 2))  # (B, C, C)

        # Symmetric learnable prior
        prior = (self.adj_prior + self.adj_prior.t()) / 2.0  # (C, C)
        prior = prior.unsqueeze(0)                            # (1, C, C)

        # Mixture
        w = F.softmax(self.log_weights, dim=0)               # (2,)
        adj = w[0] * feat_corr + w[1] * prior

        return adj, feat_corr, w


# ---------------------------------------------------------------------------
# GAT layer with optional attention extraction
# ---------------------------------------------------------------------------

class GATLayerExtractable(nn.Module):
    """
    Multi-head graph attention layer with:
      - Pre-LayerNorm
      - Adjacency bias (added to raw attention logits)
      - Residual connection
      - Position-wise FFN

    Compared to the standard GATLayer, this version can return the raw
    attention weight tensors for interpretability analysis.

    Input  : (B, C, d_node)
    Output : (B, C, d_node) [, (B, heads, C, C) attention weights]
    """

    def __init__(
        self,
        d_node: int,
        n_heads: int,
        dropout: float = 0.1,
        ffn_expansion: int = 2,
    ):
        super().__init__()
        assert d_node % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_node // n_heads
        self.scale = self.d_k ** -0.5

        self.W_qkv = nn.Linear(d_node, 3 * d_node)
        self.W_out = nn.Linear(d_node, d_node)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_node)
        self.norm2 = nn.LayerNorm(d_node)
        self.ffn = nn.Sequential(
            nn.Linear(d_node, d_node * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_node * ffn_expansion, d_node),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        adj_bias: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            h          : (B, C, d_node)
            adj_bias   : (B, C, C)       — added to logits before softmax.
            return_attention: if True, also return (B, heads, C, C) attention.

        Returns:
            h_out [, attn_weights]
        """
        B, C, D = h.shape

        h_n = self.norm1(h)
        qkv = self.W_qkv(h_n).reshape(B, C, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, C, d_k)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, C, C)
        attn = attn + adj_bias.unsqueeze(1)              # broadcast over heads
        attn_w = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_w)

        out = (attn @ v).transpose(1, 2).reshape(B, C, D)
        out = self.W_out(out)
        h = h + out

        h = h + self.ffn(self.norm2(h))

        if return_attention:
            return h, attn_w.detach()
        return h


# ---------------------------------------------------------------------------
# Attention readout modules
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Soft additive attention over T' time steps, applied independently per channel.

    Input  : (B, C, T', d_in)
    Output : (B, C, d_in),  attention weights (B, C, T')
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.Tanh(),
            nn.Linear(d_in // 2, 1),
        )

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(h).squeeze(-1)   # (B, C, T')
        weights = logits.softmax(dim=-1)     # (B, C, T')
        out = (h * weights.unsqueeze(-1)).sum(dim=2)  # (B, C, d_in)
        return out, weights


class ChannelAttention(nn.Module):
    """
    Soft attention (sigmoid gate) over C channels.

    Input  : (B, C, d_in)
    Output : (B, d_in),  gate values (B, C)
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_in, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(h).squeeze(-1)   # (B, C)
        out = (h * g.unsqueeze(-1)).sum(dim=1) / (g.sum(dim=1, keepdim=True) + 1e-8)
        return out, g


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TCNGATHybrid(nn.Module):
    """
    TCN-GAT Hybrid model for EMG gesture classification.

    Treats each EMG electrode as a graph node.
    1. Per-channel multi-scale dilated causal TCN extracts temporal features.
    2. Dynamic adjacency (feature-space Pearson + learnable prior) defines the graph.
    3. Multi-head GAT propagates information across channels at every time step.
    4. Per-channel BiGRU models post-GAT temporal dynamics.
    5. Temporal attention selects the informative time slice per channel.
    6. Channel attention weights electrode contributions to the final representation.
    7. MLP classifier produces class logits.

    Args:
        in_channels  : number of EMG channels (graph nodes).
        num_classes  : number of gesture classes.
        dropout      : base dropout probability.
        d_tcn        : TCN hidden dimensionality (= d_node for GAT).
        n_heads      : number of GAT attention heads.
        n_gat_layers : number of stacked GAT layers.
        d_gru        : GRU hidden size (output is 2·d_gru for bidirectional).
        n_gru_layers : number of GRU layers.
        tcn_kernel   : TCN convolution kernel size.
        pool_factor  : TCN temporal downsampling factor (T' = T // pool_factor).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        d_tcn: int = 64,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        d_gru: int = 128,
        n_gru_layers: int = 2,
        tcn_kernel: int = 7,
        pool_factor: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels

        # ── 1. Per-channel dilated causal TCN ─────────────────────────────
        self.tcn_encoder = PerChannelDilatedTCN(
            d_tcn=d_tcn,
            kernel_size=tcn_kernel,
            dropout=dropout * 0.5,
            pool_factor=pool_factor,
        )

        # ── 2. Dynamic inter-channel adjacency ────────────────────────────
        self.dynamic_adj = DynamicAdjacency(n_channels=in_channels)

        # ── 3. Stacked GAT layers ─────────────────────────────────────────
        self.gat_layers = nn.ModuleList([
            GATLayerExtractable(
                d_node=d_tcn,
                n_heads=n_heads,
                dropout=dropout * 0.5,
                ffn_expansion=2,
            )
            for _ in range(n_gat_layers)
        ])
        self.gat_norm = nn.LayerNorm(d_tcn)

        # ── 4. Per-channel BiGRU ──────────────────────────────────────────
        gru_drop = dropout if n_gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=d_tcn,
            hidden_size=d_gru,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=gru_drop,
            bidirectional=True,
        )
        self.d_gru_out = d_gru * 2

        # ── 5. Temporal attention ─────────────────────────────────────────
        self.temporal_attn = TemporalAttention(d_in=self.d_gru_out)

        # ── 6. Channel attention ──────────────────────────────────────────
        self.channel_attn = ChannelAttention(d_in=self.d_gru_out)

        # ── 7. MLP classifier ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_gru_out),
            nn.Linear(self.d_gru_out, self.d_gru_out // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_gru_out // 2, num_classes),
        )

        # Metadata for visualisation
        self._d_tcn = d_tcn
        self._n_heads = n_heads

    # ------------------------------------------------------------------

    def _gat_forward(
        self,
        h: torch.Tensor,
        adj_bias: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Apply stacked GAT layers at every time step.

        Args:
            h         : (B, C, T', d_node)
            adj_bias  : (B, C, C)
            return_attention : if True, collect attention from the LAST GAT layer.

        Returns:
            h_out     : (B, C, T', d_node)
            attn_last : (B, T', heads, C, C) or None
        """
        B, C, T_prime, D = h.shape

        # Fold time into batch: (B, T', C, D) → (B·T', C, D)
        h_t = h.permute(0, 2, 1, 3).reshape(B * T_prime, C, D)

        # Expand adjacency to all time steps
        adj_t = adj_bias.unsqueeze(1).expand(-1, T_prime, -1, -1)  # (B, T', C, C)
        adj_t = adj_t.reshape(B * T_prime, C, C)

        attn_last = None
        for i, gat in enumerate(self.gat_layers):
            is_last = (i == len(self.gat_layers) - 1)
            if return_attention and is_last:
                h_t, attn_last = gat(h_t, adj_t, return_attention=True)
                # attn_last: (B·T', heads, C, C)
            else:
                h_t = gat(h_t, adj_t)

        h_t = self.gat_norm(h_t)  # (B·T', C, D)

        # Restore shape
        h_out = h_t.reshape(B, T_prime, C, D).permute(0, 2, 1, 3)  # (B, C, T', D)

        if return_attention and attn_last is not None:
            # Reshape: (B·T', heads, C, C) → (B, T', heads, C, C)
            attn_last = attn_last.reshape(B, T_prime, self._n_heads, C, C)

        return h_out, attn_last

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass — returns class logits only.

        Args:
            x: (B, C, T)  raw standardised EMG.
        Returns:
            logits: (B, num_classes)
        """
        logits, _ = self._forward_impl(x, return_attention=False)
        return logits

    def forward_with_attention(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with full attention extraction for interpretability.

        Args:
            x: (B, C, T)  raw standardised EMG.

        Returns:
            logits : (B, num_classes)
            attn   : dict with interpretability tensors:
                'adjacency'        : (B, C, C)          — final adj bias
                'feat_corr'        : (B, C, C)          — Pearson component
                'adj_mix_weights'  : (2,)                — [feat_corr, prior] weights
                'gat_attention'    : (B, T', heads, C, C)  — last GAT layer attention
                'temporal_weights' : (B, C, T')          — temporal attention weights
                'channel_gates'    : (B, C)              — channel gate values
                'tcn_features'     : (B, C, T', d_tcn)  — TCN output (before GAT)
        """
        return self._forward_impl(x, return_attention=True)

    def _forward_impl(
        self, x: torch.Tensor, return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        B, C, T = x.shape

        # ── Step 1: Per-channel dilated TCN ───────────────────────────────
        h = self.tcn_encoder(x)              # (B, C, T', d_tcn)
        T_prime = h.shape[2]

        # ── Step 2: Dynamic adjacency ──────────────────────────────────────
        h_avg = h.mean(dim=2)                # (B, C, d_tcn) — time-average
        adj, feat_corr, adj_w = self.dynamic_adj(h_avg)  # (B,C,C), (B,C,C), (2,)

        # ── Step 3: Spatio-temporal GAT ───────────────────────────────────
        h, gat_attn = self._gat_forward(h, adj, return_attention=return_attention)
        # h: (B, C, T', d_tcn) | gat_attn: (B, T', heads, C, C) or None

        tcn_out = h.detach() if return_attention else None

        # ── Step 4: Per-channel BiGRU ──────────────────────────────────────
        # (B, C, T', d_tcn) → (B·C, T', d_tcn)
        h_seq = h.reshape(B * C, T_prime, self._d_tcn)
        gru_out, _ = self.gru(h_seq)         # (B·C, T', d_gru·2)
        h_gru = gru_out.reshape(B, C, T_prime, self.d_gru_out)

        # ── Step 5: Temporal attention ────────────────────────────────────
        h_chan, temp_w = self.temporal_attn(h_gru)  # (B, C, d_gru·2), (B, C, T')

        # ── Step 6: Channel attention ──────────────────────────────────────
        readout, chan_g = self.channel_attn(h_chan)  # (B, d_gru·2), (B, C)

        # ── Step 7: Classify ───────────────────────────────────────────────
        logits = self.classifier(readout)    # (B, num_classes)

        if return_attention:
            attn_dict = {
                "adjacency":        adj.detach(),
                "feat_corr":        feat_corr.detach(),
                "adj_mix_weights":  adj_w.detach(),
                "gat_attention":    gat_attn,            # (B, T', heads, C, C)
                "temporal_weights": temp_w.detach(),     # (B, C, T')
                "channel_gates":    chan_g.detach(),     # (B, C)
                "tcn_features":     tcn_out,             # (B, C, T', d_tcn)
            }
            return logits, attn_dict

        return logits, None
