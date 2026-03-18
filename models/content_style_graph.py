"""
Content-Style Graph Network for subject-invariant EMG gesture recognition.

Hypothesis: Combining content-style disentanglement with graph-based style
encoding of inter-muscular coordination patterns strengthens subject-invariance.

Architecture:
    Input (B, C, T) raw EMG
      ├─ Content Branch: CNN+BiGRU+Attention → z_content → GestureClassifier
      ├─ Style Branch:   PerChannelCNN+GAT+BiGRU+ChannelAttn → z_style → SubjectClassifier
      └─ Fusion:         gate([z_content, z_style]) → FusionClassifier

Training: returns dict with all logits, latent vectors, adjacency, channel weights.
Eval:     returns gesture_logits only (from content branch — subject-invariant).

Pure PyTorch — no torch_geometric dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── Loss functions ────────────────────────────


def distance_correlation_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Distance correlation between z1 and z2.

    Measures all (including nonlinear) statistical dependencies.
    Returns scalar in [0, 1]; 0 iff z1 ⊥ z2.
    """
    n = z1.size(0)
    if n < 4:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    a = torch.cdist(z1, z1, p=2)
    b = torch.cdist(z2, z2, p=2)

    a_row = a.mean(dim=1, keepdim=True)
    a_col = a.mean(dim=0, keepdim=True)
    a_grand = a.mean()
    A = a - a_row - a_col + a_grand

    b_row = b.mean(dim=1, keepdim=True)
    b_col = b.mean(dim=0, keepdim=True)
    b_grand = b.mean()
    B_ = b - b_row - b_col + b_grand

    dcov2 = (A * B_).mean()
    dvar_a2 = (A * A).mean()
    dvar_b2 = (B_ * B_).mean()

    denom = torch.sqrt(dvar_a2 * dvar_b2 + 1e-12)
    dcor = torch.sqrt(torch.clamp(dcov2, min=0.0) / denom)
    return dcor


def orthogonality_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Penalize linear correlation between z_content and z_style."""
    z1_norm = F.normalize(z1, dim=0)
    z2_norm = F.normalize(z2, dim=0)
    cross = z1_norm.T @ z2_norm
    return (cross ** 2).mean()


# ──────────────────── Content Branch Components ────────────────────────


class ContentEncoder(nn.Module):
    """
    CNN → BiGRU → Attention pooling for gesture-discriminative temporal features.

    Input:  (B, C, T)
    Output: (B, gru_hidden * 2) context vector
    """

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
        h = self.cnn(x)                              # (B, cnn[-1], T')
        h = h.transpose(1, 2)                        # (B, T', cnn[-1])
        gru_out, _ = self.gru(h)                     # (B, T', gru_hidden*2)
        attn_w = self.attention(gru_out)              # (B, T', 1)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (attn_w * gru_out).sum(dim=1)      # (B, gru_hidden*2)
        return context


# ───────────────────── Style Branch Components ─────────────────────────


class StyleTemporalCNNEncoder(nn.Module):
    """
    Shared 1-D CNN per EMG channel that retains temporal resolution.

    Input:  (B, C, T)
    Output: (B, C, T', d_node)  where T' = T // 16
    """

    def __init__(self, d_node: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, d_node, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_node),
            nn.GELU(),
            nn.MaxPool1d(4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        h = x.reshape(B * C, 1, T)
        h = self.net(h)                               # (B*C, d_node, T')
        T_prime = h.shape[-1]
        h = h.permute(0, 2, 1)                        # (B*C, T', d_node)
        return h.reshape(B, C, T_prime, -1)            # (B, C, T', d_node)


class StyleSpectralDynamicAdjacency(nn.Module):
    """
    Three-source inter-channel adjacency:
      1. Pearson correlation of node features (learned embedding space)
      2. Spectral coherence (frequency-domain co-activation)
      3. Learnable symmetric prior

    Mixed via softmax-normalised learnable weights.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.adj_prior = nn.Parameter(torch.zeros(n_channels, n_channels))
        nn.init.xavier_uniform_(self.adj_prior)
        self.log_weights = nn.Parameter(torch.zeros(3))

    def forward(self, h_avg: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_avg: (B, C, d_node) time-averaged node features.
            x_raw: (B, C, T) raw EMG for spectral coherence.
        Returns:
            (B, C, C) adjacency bias.
        """
        # 1. Pearson correlation in feature space
        h_c = h_avg - h_avg.mean(dim=-1, keepdim=True)
        h_n = h_c / (h_c.norm(dim=-1, keepdim=True) + 1e-8)
        feat_corr = torch.bmm(h_n, h_n.transpose(1, 2))

        # 2. Spectral coherence in frequency space
        X_fft = torch.fft.rfft(x_raw, dim=-1)
        X_pow = X_fft.abs()
        X_norm = X_pow / (X_pow.norm(dim=-1, keepdim=True) + 1e-8)
        spec_coh = torch.bmm(X_norm, X_norm.transpose(1, 2))

        # 3. Learnable symmetric prior
        prior = (self.adj_prior + self.adj_prior.t()).unsqueeze(0)

        # Softmax-normalised mixing
        w = F.softmax(self.log_weights, dim=0)
        adj = w[0] * feat_corr + w[1] * spec_coh + w[2] * prior
        return adj


class StyleGATLayer(nn.Module):
    """
    Multi-head graph attention layer with adjacency bias,
    pre-norm, residual connection, and FFN.
    """

    def __init__(self, d_node: int, n_heads: int, dropout: float = 0.1,
                 ffn_expansion: int = 2):
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

    def forward(self, h: torch.Tensor, adj_bias: torch.Tensor) -> torch.Tensor:
        """
        h:        (B, C, d_node)
        adj_bias: (B, C, C)
        returns:  (B, C, d_node)
        """
        B, C, D = h.shape
        h_n = self.norm1(h)
        qkv = self.W_qkv(h_n).reshape(B, C, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, C, d_k)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + adj_bias.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, C, D)
        out = self.W_out(out)
        h = h + out
        h = h + self.ffn(self.norm2(h))
        return h


class StyleEncoder(nn.Module):
    """
    Full style encoder: TemporalCNN → DynamicAdj → GAT → BiGRU → ChannelAttn.
    Captures inter-muscular coordination patterns (subject-specific).

    Input:  (B, C, T)
    Output: (readout (B, d_gru*2), adjacency (B, C, C), channel_weights (B, C, 1))
    """

    def __init__(
        self,
        in_channels: int = 8,
        d_node: int = 64,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        d_gru: int = 64,
        n_gru_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.temporal_cnn = StyleTemporalCNNEncoder(d_node=d_node)
        self.dynamic_adj = StyleSpectralDynamicAdjacency(n_channels=in_channels)

        self.gat_layers = nn.ModuleList([
            StyleGATLayer(
                d_node=d_node,
                n_heads=n_heads,
                dropout=dropout * 0.5,
                ffn_expansion=2,
            )
            for _ in range(n_gat_layers)
        ])
        self.gat_norm = nn.LayerNorm(d_node)

        gru_dropout = dropout if n_gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=d_node,
            hidden_size=d_gru,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )

        self._d_out = d_gru * 2
        self.channel_gate = nn.Sequential(
            nn.Linear(self._d_out, 1),
            nn.Sigmoid(),
        )

    @property
    def output_dim(self) -> int:
        return self._d_out

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T)
        returns: (readout, adj, channel_weights)
            readout:         (B, d_gru*2)
            adj:             (B, C, C)
            channel_weights: (B, C, 1)
        """
        B, C, T = x.shape

        # Step 1: Per-channel CNN → (B, C, T', d_node)
        h = self.temporal_cnn(x)
        T_prime = h.shape[2]
        D = h.shape[3]

        # Step 2: Dynamic adjacency from time-averaged features + raw signal
        h_avg = h.mean(dim=2)                          # (B, C, d_node)
        adj_bias = self.dynamic_adj(h_avg, x)          # (B, C, C)

        # Step 3: Spatio-temporal GAT — fold B*T' into batch dim
        h_t = h.permute(0, 2, 1, 3).reshape(B * T_prime, C, D)
        adj_t = adj_bias.unsqueeze(1).expand(-1, T_prime, -1, -1)
        adj_t = adj_t.reshape(B * T_prime, C, C)

        for gat in self.gat_layers:
            h_t = gat(h_t, adj_t)
        h_t = self.gat_norm(h_t)

        # Restore: (B*T', C, D) → (B, C, T', D)
        h = h_t.reshape(B, T_prime, C, D).permute(0, 2, 1, 3)

        # Step 4: Per-channel BiGRU → (B, C, d_gru*2)
        h_seq = h.reshape(B * C, T_prime, D)
        _, hidden = self.gru(h_seq)
        h_out = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (B*C, d_gru*2)
        h_out = h_out.reshape(B, C, self._d_out)

        # Step 5: Channel attention readout
        gate = self.channel_gate(h_out)                # (B, C, 1)
        readout = (h_out * gate).sum(dim=1) / (gate.sum(dim=1) + 1e-8)

        return readout, adj_bias, gate


# ──────────────────────── Projection Head ──────────────────────────────


class ProjectionHead(nn.Module):
    """Two-layer MLP projecting encoder features to a latent subspace."""

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────── Fusion Module ────────────────────────────────


class ContentStyleFusion(nn.Module):
    """
    Attention-gated fusion of content and style representations.

    Concatenates z_content and z_style, then applies a learnable sigmoid gate
    to weight how much of each representation flows to the fusion classifier.
    """

    def __init__(self, content_dim: int, style_dim: int, dropout: float = 0.3):
        super().__init__()
        total_dim = content_dim + style_dim
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(total_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_content: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([z_content, z_style], dim=-1)
        gate = self.gate_net(cat)
        fused = gate * cat
        return self.norm(self.dropout(fused))


# ──────────────────────── Main Model ───────────────────────────────────


class ContentStyleGraphNet(nn.Module):
    """
    Content-Style Graph Network for subject-invariant EMG gesture recognition.

    Content branch: CNN+BiGRU+Attention → z_content → gesture classifier
    Style branch:   PerChannelCNN+GAT+BiGRU+ChannelAttn → z_style → subject classifier
    Fusion:         gated concat → fusion classifier

    During training:  returns dict with all logits, latent vectors, adjacency, channel weights.
    During eval:      returns gesture_logits tensor only (content branch — subject-invariant).
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_gestures: int = 10,
        num_subjects: int = 4,
        content_dim: int = 128,
        style_dim: int = 64,
        d_node: int = 64,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        gru_hidden_content: int = 128,
        gru_hidden_style: int = 64,
        dropout: float = 0.3,
        use_fusion_at_eval: bool = False,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.use_fusion_at_eval = use_fusion_at_eval

        # ── Content branch ──
        self.content_encoder = ContentEncoder(
            in_channels=in_channels,
            gru_hidden=gru_hidden_content,
            dropout=dropout,
        )
        content_enc_dim = self.content_encoder.gru_output_dim  # 256

        self.content_proj = ProjectionHead(content_enc_dim, content_dim, dropout)

        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # ── Style branch ──
        self.style_encoder = StyleEncoder(
            in_channels=in_channels,
            d_node=d_node,
            n_heads=n_heads,
            n_gat_layers=n_gat_layers,
            d_gru=gru_hidden_style,
            dropout=dropout,
        )
        style_enc_dim = self.style_encoder.output_dim  # gru_hidden_style * 2

        self.style_proj = ProjectionHead(style_enc_dim, style_dim, dropout)

        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

        # ── Fusion ──
        self.fusion = ContentStyleFusion(content_dim, style_dim, dropout)
        fused_dim = content_dim + style_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_gestures),
        )

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x:          (B, C, T)
            return_all: force returning full dict even in eval mode
        Returns:
            training / return_all:
                dict with gesture_logits, fusion_logits, subject_logits,
                z_content, z_style, adjacency, channel_weights
            eval (default):
                gesture_logits (B, num_gestures)
        """
        # Content branch
        content_enc = self.content_encoder(x)              # (B, 256)
        z_content = self.content_proj(content_enc)         # (B, content_dim)
        gesture_logits = self.gesture_classifier(z_content)

        if self.training or return_all:
            # Style branch
            style_enc, adj, channel_w = self.style_encoder(x)
            z_style = self.style_proj(style_enc)           # (B, style_dim)
            subject_logits = self.subject_classifier(z_style)

            # Fusion
            fused = self.fusion(z_content, z_style)
            fusion_logits = self.fusion_classifier(fused)

            return {
                "gesture_logits": gesture_logits,
                "fusion_logits": fusion_logits,
                "subject_logits": subject_logits,
                "z_content": z_content,
                "z_style": z_style,
                "adjacency": adj,
                "channel_weights": channel_w,
            }

        # Eval mode: content branch only (subject-invariant)
        if self.use_fusion_at_eval:
            style_enc, adj, channel_w = self.style_encoder(x)
            z_style = self.style_proj(style_enc)
            fused = self.fusion(z_content, z_style)
            return self.fusion_classifier(fused)

        return gesture_logits
