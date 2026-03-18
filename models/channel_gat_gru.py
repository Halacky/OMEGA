"""
Channel GAT + BiGRU: Spatio-Temporal Graph Network for EMG Gesture Classification.

Hypothesis H37: Modelling inter-muscular co-activation via a graph (channels as
nodes, edges = Pearson correlation + spectral coherence) followed by temporal
sequence modelling (BiGRU) captures subject-invariant representations better than
purely temporal or purely graph approaches.

Architecture:
    Input (B, C, T) raw standardised EMG
    → TemporalCNNEncoder : shared 1-D CNN per channel, retains T' steps
                           → (B, C, T', d_node)
    → SpectralDynamicAdjacency : Pearson corr (feature space)
                                 + spectral coherence (frequency space)
                                 + learnable prior
                                 → (B, C, C) adjacency bias
    → SpatioTemporalGAT  : GATLayer applied at every time step
                           → (B, C, T', d_node)
    → Per-channel BiGRU  : temporal modelling per channel
                           → (B, C, 2·d_gru)
    → Channel Attention readout → (B, 2·d_gru)
    → MLP Classifier     → (B, num_classes)

Pure PyTorch – no torch_geometric dependency.
Inspiration: EEG GNNs (BCI), Human Pose Estimation via joint graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class TemporalCNNEncoder(nn.Module):
    """
    Shared 1-D CNN per EMG channel that retains spatial resolution T'.

    Input  : (B, C, T)
    Output : (B, C, T', d_node)  where T' = T // 16
    """

    def __init__(self, d_node: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),                              # T → T // 4
            nn.Conv1d(32, d_node, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_node),
            nn.GELU(),
            nn.MaxPool1d(4),                              # T → T // 16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        h = x.reshape(B * C, 1, T)          # (B·C, 1, T)
        h = self.net(h)                       # (B·C, d_node, T')
        T_prime = h.shape[-1]
        h = h.permute(0, 2, 1)               # (B·C, T', d_node)
        return h.reshape(B, C, T_prime, -1)  # (B, C, T', d_node)


class SpectralDynamicAdjacency(nn.Module):
    """
    Combines three sources of inter-channel relationship:
      1. Pearson correlation of node feature vectors (learned embedding space).
      2. Spectral coherence – cosine similarity of normalised power spectra
         (captures frequency-domain co-activation of muscles).
      3. Learnable symmetric adjacency prior.

    All three are mixed via softmax-normalised learnable weights.

    Args:
        n_channels: number of EMG electrodes (= graph nodes).
    """

    def __init__(self, n_channels: int):
        super().__init__()
        # Learnable prior (symmetric via A + Aᵀ)
        self.adj_prior = nn.Parameter(torch.zeros(n_channels, n_channels))
        nn.init.xavier_uniform_(self.adj_prior)
        # Log-weights for the three adjacency components
        self.log_weights = nn.Parameter(torch.zeros(3))  # softmax → (w1, w2, w3)

    def forward(self, h_avg: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_avg : (B, C, d_node)  time-averaged node features from CNN encoder.
            x_raw : (B, C, T)       raw EMG signal used for spectral coherence.
        Returns:
            (B, C, C)  adjacency bias added to GAT attention logits.
        """
        # 1. Pearson correlation in feature space --------------------------------
        h_c = h_avg - h_avg.mean(dim=-1, keepdim=True)          # centre
        h_n = h_c / (h_c.norm(dim=-1, keepdim=True) + 1e-8)     # normalise
        feat_corr = torch.bmm(h_n, h_n.transpose(1, 2))         # (B, C, C)

        # 2. Spectral coherence in frequency space --------------------------------
        # rfft along time axis → power spectrum per channel
        X_fft = torch.fft.rfft(x_raw, dim=-1)                   # (B, C, F) complex
        X_pow = X_fft.abs()                                       # (B, C, F) real
        X_norm = X_pow / (X_pow.norm(dim=-1, keepdim=True) + 1e-8)
        spec_coh = torch.bmm(X_norm, X_norm.transpose(1, 2))    # (B, C, C)

        # 3. Learnable symmetric prior --------------------------------------------
        prior = (self.adj_prior + self.adj_prior.t())             # (C, C) symmetric
        prior = prior.unsqueeze(0)                                # (1, C, C) broadcast

        # Softmax-normalised mixing
        w = F.softmax(self.log_weights, dim=0)                   # (3,)
        adj = w[0] * feat_corr + w[1] * spec_coh + w[2] * prior
        return adj                                                # (B, C, C)


class GATLayer(nn.Module):
    """
    Multi-head graph attention layer with adjacency bias, pre-norm, residual, FFN.
    (Same design as in channel_gat.py – replicated here to keep the model
    self-contained and allow independent tuning.)
    """

    def __init__(self, d_node: int, n_heads: int, dropout: float = 0.1,
                 ffn_expansion: int = 2):
        super().__init__()
        assert d_node % n_heads == 0, "d_node must be divisible by n_heads"
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

    def forward(self, h: torch.Tensor, adj_bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h        : (B, C, d_node)
            adj_bias : (B, C, C)  – added to attention logits before softmax
        Returns:
            (B, C, d_node)
        """
        B, C, D = h.shape

        # Pre-norm → Q K V projections
        h_n = self.norm1(h)
        qkv = self.W_qkv(h_n).reshape(B, C, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, C, d_k)
        q, k, v = qkv.unbind(0)

        # Attention with adjacency bias
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, C, C)
        attn = attn + adj_bias.unsqueeze(1)               # broadcast over heads
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, C, D)
        out = self.W_out(out)
        h = h + out                                       # residual

        # Pre-norm FFN
        h = h + self.ffn(self.norm2(h))
        return h


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ChannelGATGRU(nn.Module):
    """
    Channel Graph Attention + BiGRU network for EMG gesture classification.

    Treats each EMG electrode as a graph node.  A dynamic adjacency matrix
    (Pearson correlation + spectral coherence + learnable prior) captures
    inter-muscular co-activation.  GAT layers propagate information across
    the channel graph at every time step; a per-channel BiGRU then models
    the resulting spatio-temporal features over the temporal dimension;
    finally a soft-attention readout aggregates across channels.

    Args:
        in_channels   : number of EMG channels / graph nodes.
        num_classes   : number of gesture classes.
        dropout       : dropout probability.
        d_node        : node feature dimensionality in CNN encoder & GAT.
        n_heads       : number of GAT attention heads (d_node % n_heads == 0).
        n_gat_layers  : number of stacked GAT layers.
        d_gru         : hidden size of GRU (output is 2·d_gru for bidirectional).
        n_gru_layers  : number of GRU layers.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        d_node: int = 64,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        d_gru: int = 128,
        n_gru_layers: int = 2,
        **kwargs,
    ):
        super().__init__()

        # 1. Per-channel temporal CNN (retains T' steps)
        self.temporal_cnn = TemporalCNNEncoder(d_node=d_node)

        # 2. Dynamic adjacency (Pearson + spectral coherence + prior)
        self.dynamic_adj = SpectralDynamicAdjacency(n_channels=in_channels)

        # 3. Stacked GAT layers (applied at every time step)
        self.gat_layers = nn.ModuleList([
            GATLayer(
                d_node=d_node,
                n_heads=n_heads,
                dropout=dropout * 0.5,
                ffn_expansion=2,
            )
            for _ in range(n_gat_layers)
        ])
        self.gat_norm = nn.LayerNorm(d_node)

        # 4. Per-channel BiGRU for temporal modelling
        gru_dropout = dropout if n_gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=d_node,
            hidden_size=d_gru,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=True,
        )
        d_gru_out = d_gru * 2  # bidirectional

        # 5. Channel attention readout
        self.channel_gate = nn.Sequential(
            nn.Linear(d_gru_out, 1),
            nn.Sigmoid(),
        )

        # 6. MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_gru_out, d_gru_out // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_gru_out // 2, num_classes),
        )

        self._d_gru_out = d_gru_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)  raw standardised EMG, C = in_channels, T = window size.
        Returns:
            (B, num_classes)  classification logits.
        """
        B, C, T = x.shape

        # ------------------------------------------------------------------
        # Step 1 – Per-channel CNN: extract d_node features at T' time steps
        # ------------------------------------------------------------------
        h = self.temporal_cnn(x)            # (B, C, T', d_node)
        T_prime = h.shape[2]
        D = h.shape[3]

        # ------------------------------------------------------------------
        # Step 2 – Dynamic adjacency from time-averaged features + raw signal
        # ------------------------------------------------------------------
        h_avg = h.mean(dim=2)               # (B, C, d_node) – collapse time
        adj_bias = self.dynamic_adj(h_avg, x)   # (B, C, C)

        # ------------------------------------------------------------------
        # Step 3 – Spatio-temporal GAT: apply GAT at every time step
        #
        # Strategy: fold B and T' into a single batch dimension so that
        # each time step is treated as an independent graph instance.
        # Adjacency is shared across time (derived from the full sequence).
        # ------------------------------------------------------------------
        # (B, C, T', D) → (B, T', C, D) → (B·T', C, D)
        h_t = h.permute(0, 2, 1, 3).reshape(B * T_prime, C, D)

        # Expand adjacency to match the folded batch: (B, C, C) → (B·T', C, C)
        adj_t = adj_bias.unsqueeze(1).expand(-1, T_prime, -1, -1)   # (B, T', C, C)
        adj_t = adj_t.reshape(B * T_prime, C, C)

        for gat in self.gat_layers:
            h_t = gat(h_t, adj_t)          # (B·T', C, D) → (B·T', C, D)
        h_t = self.gat_norm(h_t)           # (B·T', C, D)

        # Restore: (B·T', C, D) → (B, T', C, D) → (B, C, T', D)
        h = h_t.reshape(B, T_prime, C, D).permute(0, 2, 1, 3)  # (B, C, T', D)

        # ------------------------------------------------------------------
        # Step 4 – Per-channel BiGRU: model temporal dynamics per electrode
        # ------------------------------------------------------------------
        # (B, C, T', D) → (B·C, T', D)
        h_seq = h.reshape(B * C, T_prime, D)

        _, hidden = self.gru(h_seq)
        # hidden: (n_layers·2, B·C, d_gru) for bidirectional GRU
        # Concatenate last-layer forward and backward hidden states
        h_out = torch.cat([hidden[-2], hidden[-1]], dim=-1)   # (B·C, d_gru·2)
        h_out = h_out.reshape(B, C, self._d_gru_out)          # (B, C, d_gru·2)

        # ------------------------------------------------------------------
        # Step 5 – Soft channel attention readout
        # ------------------------------------------------------------------
        gate = self.channel_gate(h_out)    # (B, C, 1)
        readout = (h_out * gate).sum(dim=1) / (gate.sum(dim=1) + 1e-8)  # (B, d_gru·2)

        # ------------------------------------------------------------------
        # Step 6 – Classify
        # ------------------------------------------------------------------
        return self.classifier(readout)    # (B, num_classes)
