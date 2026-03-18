"""
Multi-Resolution Barlow Twins with Channel-Group Factorization (MRBT-CG).

Hypothesis 5:
    EMG channels are anatomically grouped; electrode-placement noise is
    per-group (local), while gesture-related muscle activation is
    inter-group consistent.  ECAPA-TDNN blocks at L parallel temporal
    scales produce per-group multi-scale features.  Cross-group attention
    extracts content (consistent across groups) and style (inter-group
    variance) at each scale.  A Barlow-Twins-inspired cross-correlation
    decorrelation loss, computed in the trainer over (content, style)
    pairs at each scale, pushes the two representations apart without
    adversarial training.

Architecture (input: (B, C_emg, T), e.g. C_emg=8):
    1.  Channel splitting:  C_emg → K groups of C_emg//K channels.
    2.  SharedGroupInitTDNN:
            k-th group (B, C_g_in, T) → shared Conv1d(5) → (B, C_g, T).
            Implemented as single Conv1d via (B*K, C_g_in, T) reshape trick.
    3.  Parallel multi-scale processing:
            For each scale l (dilation d_l):
              SharedGroupSERes2NetBlock(h0, dilation=d_l) → h_l  (B, K*C_g, T).
            All L branches receive the same h0; they are independent.
    4.  CrossGroupAttention at each scale:
            content_l = Σ_k attn_k * g_k^l       (B, C_g, T)  — soft consensus
            style_l   = std_k(g_k^l)              (B, C_g, T)  — inter-group spread
    5.  MFA: cat([content_0, …, content_{L-1}], dim=1) → 1×1 Conv → (B, L*C_g, T).
    6.  Attentive Statistics Pooling: (B, L*C_g, T) → (B, 2*L*C_g).
    7.  Embedding: Linear(2*L*C_g, E) + BN + ReLU + Dropout → (B, E).
    8.  Classifier: Linear(E, num_classes).

Inference path (forward):
    Steps 1-8 using content only — no style branch, no BT loss.

Training path (forward_with_style):
    Same as inference but returns content_list and style_list per scale
    so the trainer can compute BT cross-correlation losses.

LOSO Compliance (strict — no adaptation to test subject):
    ┌───────────────────────────────────────────────────────────────┐
    │  forward(x)                                                   │
    │    → content pathway ONLY.                                    │
    │    → No style computation, no BT loss, no stored subject      │
    │      statistics accessed at inference time.                   │
    │                                                               │
    │  forward_with_style(x)                                        │
    │    → Called ONLY by the trainer on training-subject windows.  │
    │    → Returns per-scale (content, style) for BT loss.          │
    │    → Test-subject windows are never passed here.              │
    │                                                               │
    │  BatchNorm running stats                                       │
    │    → Set from training subjects only (model.eval() at test).  │
    │    → SharedGroupInitTDNN / SharedGroupSERes2NetBlock use the  │
    │      (B*K, …) reshape, so BN sees a richer batch but still    │
    │      only training-subject data.                              │
    └───────────────────────────────────────────────────────────────┘
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Standard ECAPA building blocks
# ─────────────────────────────────────────────────────────────────────────────

class Res2NetBlock(nn.Module):
    """
    Hierarchical multi-scale 1-D convolution (Res2Net branching).

    Splits C channels into `scale` equal sub-groups.  The first sub-group
    is an identity path; each subsequent sub-group applies a dilated Conv1d
    to the sum of its input and the previous branch's output (Res2 shortcut).
    """

    def __init__(self, C: int, kernel_size: int = 3, dilation: int = 1, scale: int = 4):
        super().__init__()
        assert C % scale == 0, f"C={C} must be divisible by scale={scale}"
        self.scale = scale
        self.width = C // scale
        pad = dilation * (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size,
                      dilation=dilation, padding=pad, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.scale, dim=1)
        outs = [chunks[0]]
        prev: Optional[torch.Tensor] = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = chunks[i + 1] if prev is None else chunks[i + 1] + prev
            prev = F.relu(bn(conv(h)), inplace=True)
            outs.append(prev)
        return torch.cat(outs, dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, C: int, reduction: int = 8):
        super().__init__()
        mid = max(C // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(C, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, C, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.fc(x.mean(dim=2)).unsqueeze(2)   # (B, C, 1)
        return x * s


class SERes2NetBlock(nn.Module):
    """
    SE-Res2Net TDNN block — core building block of ECAPA-TDNN.

    Flow:  input → 1×1 PW + BN + ReLU
                 → Res2Net(dilation)
                 → 1×1 PW + BN
                 → SE attention
                 → residual-add + ReLU
    """

    def __init__(
        self,
        C: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 4,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.pw_in = nn.Sequential(
            nn.Conv1d(C, C, 1, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )
        self.res2 = Res2NetBlock(C, kernel_size, dilation, scale)
        self.pw_out = nn.Sequential(
            nn.Conv1d(C, C, 1, bias=False),
            nn.BatchNorm1d(C),
        )
        self.se = SEBlock(C, se_reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.pw_in(x)
        out = self.res2(out)
        out = self.pw_out(out)
        out = self.se(out)
        return self.relu(out + residual)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP): (B, C, T) → (B, 2C).

    Learns per-channel, per-timestep attention weights; outputs the
    weighted mean and weighted standard deviation over time.
    LOSO-safe: purely input-driven, no stored statistics.
    """

    def __init__(self, C: int):
        super().__init__()
        hidden = max(C // 4, 16)
        self.attn = nn.Sequential(
            nn.Conv1d(C, hidden, 1, bias=False),
            nn.Tanh(),
            nn.Conv1d(hidden, C, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.attn(x), dim=2)         # (B, C, T)
        mu = (alpha * x).sum(dim=2)                    # (B, C)
        var = (alpha * x.pow(2)).sum(dim=2) - mu.pow(2)
        sigma = var.clamp(min=1e-8).sqrt()             # (B, C)
        return torch.cat([mu, sigma], dim=1)           # (B, 2C)


# ─────────────────────────────────────────────────────────────────────────────
# Channel-group specific modules
# ─────────────────────────────────────────────────────────────────────────────

class SharedGroupInitTDNN(nn.Module):
    """
    Shared initial TDNN applied independently to each EMG channel group.

    Takes K groups of C_g_in channels each (packed as (B, K*C_g_in, T))
    and maps each group to C_g feature channels using SHARED Conv1d weights.

    Implementation: (B, K*C_g_in, T) is reshaped to (B*K, C_g_in, T),
    processed through a single Conv1d+BN+ReLU, then reshaped back to
    (B, K*C_g, T).  This batches all K groups through one forward pass
    and ensures ZERO cross-group information flow while sharing parameters.

    Physical motivation: the same short-range temporal features (motor-unit
    action potentials, ≈2.5 ms receptive field at 2 kHz) are relevant for
    all electrode groups.
    """

    def __init__(self, n_groups: int, in_ch_per_group: int, group_channels: int):
        super().__init__()
        self.n_groups = n_groups
        self.in_ch = in_ch_per_group
        self.out_ch = group_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch_per_group, group_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(group_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K*C_g_in, T) — channel groups packed along dim 1
        Returns:
            (B, K*C_g, T) — per-group features packed along dim 1
        """
        B, KCin, T = x.shape
        K = self.n_groups
        x_r = x.contiguous().view(B * K, self.in_ch, T)   # (B*K, C_g_in, T)
        h = self.conv(x_r)                                  # (B*K, C_g, T)
        return h.view(B, K * self.out_ch, T)               # (B, K*C_g, T)


class SharedGroupSERes2NetBlock(nn.Module):
    """
    Shared SERes2Net block applied independently to each channel group.

    Same reshape trick as SharedGroupInitTDNN:
        (B, K*C_g, T) → (B*K, C_g, T) → SERes2Net → (B, K*C_g, T)

    Shared weights across K groups:
      - Group independence is strictly maintained (no cross-group Conv1d).
      - The same temporal feature extractor is applied to each group,
        giving a group-agnostic representation before cross-group attention.
      - Cross-group stylistic differences emerge purely from input data.
    """

    def __init__(
        self,
        n_groups: int,
        group_channels: int,
        dilation: int,
        scale: int = 4,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.C_g = group_channels
        self.block = SERes2NetBlock(
            group_channels, kernel_size=3,
            dilation=dilation, scale=scale, se_reduction=se_reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K*C_g, T)
        Returns:
            (B, K*C_g, T) — each group processed independently via shared block
        """
        B, KCg, T = x.shape
        K = self.n_groups
        C_g = self.C_g
        x_r = x.contiguous().view(B * K, C_g, T)   # (B*K, C_g, T)
        out = self.block(x_r)                        # (B*K, C_g, T)
        return out.view(B, K * C_g, T)              # (B, K*C_g, T)


class CrossGroupAttention(nn.Module):
    """
    Cross-group attention for content/style disentanglement at one temporal scale.

    Given K per-group feature maps packed as (B, K*C_g, T), computes:

        content_l = Σ_k attn_k(t) * g_k^l(t)   — soft consensus (B, C_g, T)
        style_l   = std_{k}(g_k^l)              — inter-group spread (B, C_g, T)

    Physical interpretation:
        • Gestures activate multiple muscle groups SIMULTANEOUSLY and CONSISTENTLY
          → content captures this cross-group agreement.
        • Electrode placement / skin impedance effects are GROUP-LOCAL
          → style captures the per-group deviation (inter-group std).

    Attention scoring:
        A shared Conv1d(C_g → 1, kernel=1) is applied to each group via the
        (B*K, C_g, T) reshape trick, producing a per-group per-timestep scalar.
        Softmax over K normalises the weights to sum to 1 at each timestep.

    LOSO compliance:
        • No subject-specific parameters.
        • Attention weights are purely input-driven at inference time.
        • `style` is only used during training (for BT loss); the model's
          forward() discards it.
    """

    def __init__(self, n_groups: int, group_channels: int):
        super().__init__()
        self.n_groups = n_groups
        self.C_g = group_channels
        # Shared attention query: C_g features → 1 scalar per group per timestep
        self.attn_query = nn.Conv1d(group_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, K*C_g, T)
        Returns:
            content: (B, C_g, T) — attention-weighted mean across K groups
            style:   (B, C_g, T) — standard deviation across K groups
        """
        B, KCg, T = x.shape
        K = self.n_groups
        C_g = self.C_g

        # Per-group attention score via shared query (reshape trick)
        x_r = x.contiguous().view(B * K, C_g, T)         # (B*K, C_g, T)
        scores = self.attn_query(x_r).view(B, K, T)      # (B, K, T)
        attn = F.softmax(scores, dim=1)                   # (B, K, T) — sums to 1 over K

        # Reshape K groups for group-wise operations
        x_g = x.view(B, K, C_g, T)                       # (B, K, C_g, T)

        # Content: attention-weighted sum — what is consistent across groups
        # attn[:, :, None, :] broadcasts over C_g dimension
        content = (attn.unsqueeze(2) * x_g).sum(dim=1)   # (B, C_g, T)

        # Style: inter-group standard deviation — what differs across groups
        # std_k(g_k) captures electrode-placement and impedance variability.
        # Using unbiased=False for gradient stability with small K.
        # The clamp prevents NaN gradients when K=2 and both groups are identical.
        style = x_g.var(dim=1, unbiased=False).clamp(min=0.0).sqrt()  # (B, C_g, T)

        return content, style


# ─────────────────────────────────────────────────────────────────────────────
# BT cross-correlation decorrelation loss (used by trainer)
# ─────────────────────────────────────────────────────────────────────────────

def barlow_twins_decor_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    """
    Barlow-Twins-inspired cross-correlation decorrelation between z_a and z_b.

    Goal: minimise ALL pairwise cross-correlations between dimensions of
    z_a (content) and z_b (style) so that they are linearly independent.

    Unlike the original Barlow Twins (which also enforces diagonal = 1 to
    prevent collapse of a self-supervised representation), here we ONLY
    minimise cross-correlations — the content representation is prevented
    from collapsing by the classification loss in the trainer.

    Loss = mean_{i,j}( corr(z_a[:,i], z_b[:,j])^2 )

    Args:
        z_a: (B, D_a) — content features (time-pooled per scale)
        z_b: (B, D_b) — style  features (time-pooled per scale)

    Returns:
        Scalar loss. Returns 0 if batch size < 2 (undefined correlations).

    LOSO note: called only on training-batch tensors; test windows never seen.
    """
    B = z_a.shape[0]
    if B < 2:
        return z_a.new_zeros(1).squeeze()

    # Normalise each feature dimension across batch (z-score)
    z_a_n = (z_a - z_a.mean(0, keepdim=True)) / (z_a.std(0, keepdim=True) + 1e-8)
    z_b_n = (z_b - z_b.mean(0, keepdim=True)) / (z_b.std(0, keepdim=True) + 1e-8)

    # Cross-correlation matrix: (D_a, D_b)
    C = (z_a_n.T @ z_b_n) / B

    # All elements should approach 0 (content ⊥ style)
    return C.pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class MRBTChannelGroupModel(nn.Module):
    """
    Multi-Resolution Barlow Twins with Channel-Group Factorization.

    Parameters
    ----------
    in_channels : int
        Number of EMG input channels (e.g., 8 for NinaPro DB2 8-ch subset).
        Must be divisible by n_groups.
    num_classes : int
        Number of gesture classes.
    n_groups : int
        K — number of channel groups.  With in_channels=8, n_groups=4 gives
        2 EMG channels per group.  With in_channels=12, n_groups=4 gives 3.
    group_channels : int
        C_g — feature dimension per group.  Must be divisible by `scale`.
    n_scales : int
        L — number of parallel temporal scales (one dilation value each).
    dilations : list of int
        Dilation factors for the L parallel SE-Res2Net branches.
    embedding_dim : int
        E — dimension of the final embedding before the classifier.
    dropout : float
        Dropout rate before the classifier.
    scale : int
        Res2Net sub-group branching factor (inside each SE-Res2Net block).
    se_reduction : int
        Squeeze-and-Excitation bottleneck reduction factor.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_groups: int = 4,
        group_channels: int = 32,
        n_scales: int = 3,
        dilations: Optional[List[int]] = None,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        scale: int = 4,
        se_reduction: int = 8,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4]
        if len(dilations) != n_scales:
            raise ValueError(
                f"len(dilations)={len(dilations)} must equal n_scales={n_scales}"
            )
        if in_channels % n_groups != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by n_groups={n_groups}"
            )
        if group_channels % scale != 0:
            raise ValueError(
                f"group_channels={group_channels} must be divisible by scale={scale} "
                f"(Res2Net sub-group requirement)"
            )

        self.n_groups = n_groups
        self.C_g = group_channels
        self.n_scales = n_scales
        in_ch_per_group = in_channels // n_groups

        # ── 1. Shared initial TDNN per channel group ────────────────────────
        # k=5 (2.5 ms at 2 kHz) — captures single MUAP shapes.
        self.group_init = SharedGroupInitTDNN(n_groups, in_ch_per_group, group_channels)

        # ── 2. Parallel multi-scale SE-Res2Net branches ─────────────────────
        # All L branches receive the SAME h0 (output of group_init).
        # Each branch uses a different dilation → different temporal receptive field.
        # Shared weights within each branch (across K groups, via reshape trick).
        self.scale_blocks = nn.ModuleList([
            SharedGroupSERes2NetBlock(n_groups, group_channels, d, scale, se_reduction)
            for d in dilations
        ])

        # ── 3. Cross-group attention — one module per scale ─────────────────
        self.cross_attn = nn.ModuleList([
            CrossGroupAttention(n_groups, group_channels)
            for _ in range(n_scales)
        ])

        # ── 4. Multi-layer Feature Aggregation ──────────────────────────────
        # Concatenates content from all L scales → C_g * L channels.
        # 1×1 Conv mixes cross-scale information.
        mfa_in = group_channels * n_scales
        self.mfa = nn.Sequential(
            nn.Conv1d(mfa_in, mfa_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_in),
            nn.ReLU(inplace=True),
        )

        # ── 5. Attentive Statistics Pooling ─────────────────────────────────
        self.asp = AttentiveStatisticsPooling(mfa_in)

        # ── 6. Embedding + Classifier ───────────────────────────────────────
        asp_out = mfa_in * 2
        self.embedding = nn.Sequential(
            nn.Linear(asp_out, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """He-uniform initialisation for conv/linear; constant 1/0 for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Core encoder: group init → parallel scale branches → cross-group attention.

        Args:
            x: (B, C_emg, T)
        Returns:
            content_list : L tensors (B, C_g, T) — cross-group consensus per scale
            style_list   : L tensors (B, C_g, T) — inter-group std per scale
            mfa_out      : (B, L*C_g, T)          — concatenated content after MFA
        """
        # Shared initial features for all channel groups
        h0 = self.group_init(x)                     # (B, K*C_g, T)

        # Parallel multi-scale processing + cross-group attention
        content_list: List[torch.Tensor] = []
        style_list: List[torch.Tensor] = []

        for block, attn in zip(self.scale_blocks, self.cross_attn):
            h_l = block(h0)                          # (B, K*C_g, T) — scale-l features
            content_l, style_l = attn(h_l)           # (B, C_g, T) each
            content_list.append(content_l)
            style_list.append(style_l)

        # MFA: concatenate content from all scales, then mix with 1×1 conv
        mfa_in_t = torch.cat(content_list, dim=1)   # (B, L*C_g, T)
        mfa_out = self.mfa(mfa_in_t)                # (B, L*C_g, T)

        return content_list, style_list, mfa_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference: content pathway only.

        Args:
            x: (B, C_emg, T) — EMG windows in channels-first format
        Returns:
            logits: (B, num_classes)

        LOSO compliance:
            No style computation, no BT loss, no subject-specific statistics.
            BatchNorm running stats were set from training subjects only.
        """
        _, _, mfa_out = self._encode(x)
        pooled = self.asp(mfa_out)             # (B, 2*L*C_g)
        emb = self.embedding(pooled)           # (B, E)
        return self.classifier(emb)            # (B, num_classes)

    def forward_with_style(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Training path: returns logits + per-scale (content, style) tensors.

        Called ONLY by the trainer on training-subject windows to compute
        the Barlow Twins decorrelation loss.  Test-subject windows are
        NEVER passed to this method.

        Args:
            x: (B, C_emg, T)
        Returns:
            logits:       (B, num_classes)
            content_list: list of n_scales tensors (B, C_g, T)
            style_list:   list of n_scales tensors (B, C_g, T)
        """
        content_list, style_list, mfa_out = self._encode(x)
        pooled = self.asp(mfa_out)
        emb = self.embedding(pooled)
        logits = self.classifier(emb)
        return logits, content_list, style_list

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
