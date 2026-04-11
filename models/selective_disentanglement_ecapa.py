"""
Selective Disentanglement ECAPA-TDNN (CLIP-DCA approach).

Hypothesis (exp_89):
  The key failure of full content/style separation (exp_31) is that forcing
  complete disentanglement at the encoder level destroys useful correlations
  (e.g., EMG amplitude correlates with both gesture and subject).

  CLIP-DCA alternative: strengthen domain awareness in the encoder, but
  enforce domain invariance only at the classifier level.

Architecture
────────────
1. Domain-aware encoder (ECAPA-TDNN backbone):
   - Trained with auxiliary subject classification head.
   - Encoder explicitly encodes subject identity.

2. Domain-conditioned attention (FiLM):
   - Subject embedding modulates SE attention weights at each SE-Res2Net block
     via Feature-wise Linear Modulation (scale γ, shift β per channel).
   - Allows encoder to implicitly normalise features relative to the subject.
   - At inference: subject embedding = mean of ALL learned training-subject
     embeddings (computed from model parameters only — LOSO-clean).

3. Projection head:
   - Maps domain-aware embedding z → gesture-specific representation h.

4. Domain-invariant gesture classifier:
   - Trained with gradient reversal on h only (not on full z).
   - Gradient reversal pushes h to be domain-invariant without destroying
     the expressiveness of the full encoder representation z.

5. Loss (training only):
   L = L_cls + λ_subj * L_subj_aux + λ_dom * L_domain_confusion

   At inference: subject head and domain head are skipped entirely.

LOSO invariants
───────────────
  ✓ Subject labels used during training only — dropped at inference.
  ✓ FiLM at test time: mean of learned embedding table (model parameters,
    NOT derived from any test-subject signal data).
  ✓ Channel normalisation: computed from training data only (in trainer).
  ✓ model.eval() at inference: BatchNorm frozen to training running stats.
  ✓ No per-subject adaptation, no running statistics from test subject.
  ✓ GRL operates only on projection h — encoder z retains full expressiveness.

Input:  (B, C_emg, T)   — channels-first, matching PyTorch Conv1d convention.
Output: gesture_logits (B, num_classes)  [inference]
        + subject_logits, domain_logits  [training only]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ═══════════════════════════ Gradient Reversal Layer ════════════════════════

class _GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from DANN (Ganin et al., 2016).

    Forward:  identity (x → x).
    Backward: negates and scales gradients by alpha → -alpha * grad.

    Applying this before a domain classifier and minimising the domain
    classification loss is equivalent to maximising domain confusion
    (the representation becomes domain-invariant w.r.t. domain_head).

    Alpha is annealed from 0 → 1 during training so that domain confusion
    pressure is gradually introduced after the gesture classifier has found
    a reasonable feature space.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply Gradient Reversal with strength alpha."""
    return _GradientReversalFunction.apply(x, alpha)


# ═══════════════════════════════ FiLM Layer ═════════════════════════════════

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioning layer.

    Produces per-channel scale (γ) and shift (β) from a subject embedding
    vector.  The conditioned output is:

        out = (1 + γ) * x + β

    The "1 +" initialisation trick (γ_linear initialised to zero) ensures that
    at the start of training, FiLM acts as identity conditioning — the model
    begins identical to a standard ECAPA-TDNN and gradually learns to exploit
    subject identity.

    LOSO safety at test time
    ────────────────────────
    subject_emb = mean of all rows in SelectiveDisentanglementECAPA.subject_embeddings.weight
    This is computed from learned model parameters AFTER training, never from
    test-subject signal data.  A single fixed modulation is applied to all
    test windows — no per-subject adaptation.

    Args:
        subject_emb_dim: Dimension of subject embedding vector (D).
        channels:        Number of feature channels to modulate (C).
    """

    def __init__(self, subject_emb_dim: int, channels: int):
        super().__init__()
        self.gamma_linear = nn.Linear(subject_emb_dim, channels, bias=True)
        self.beta_linear  = nn.Linear(subject_emb_dim, channels, bias=True)

        # Zero-init → identity at start of training
        nn.init.zeros_(self.gamma_linear.weight)
        nn.init.zeros_(self.gamma_linear.bias)
        nn.init.zeros_(self.beta_linear.weight)
        nn.init.zeros_(self.beta_linear.bias)

    def forward(
        self, subject_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            subject_emb: (B, D) — batched embeddings, or (D,) for broadcast.
        Returns:
            gamma: (B, C) or (1, C) — scale modulation (additive delta)
            beta:  (B, C) or (1, C) — shift modulation
        """
        if subject_emb.dim() == 1:
            subject_emb = subject_emb.unsqueeze(0)  # (1, D) for broadcast
        gamma = self.gamma_linear(subject_emb)  # (B, C) or (1, C)
        beta  = self.beta_linear(subject_emb)   # (B, C) or (1, C)
        return gamma, beta


# ══════════════════════ ECAPA-TDNN Building Blocks ══════════════════════════

class Res2NetBlock(nn.Module):
    """
    Res2Net temporal block — hierarchical multi-scale dilated convolution.

    Splits the C-channel feature map into `scale` equal sub-groups.
    The first sub-group is an identity path.  Each subsequent sub-group i
    applies a dilated Conv1d to the sum of its own features and the previous
    group's output (Res2 shortcut), enabling hierarchical feature reuse.

    Args:
        C:           total channel width (must be divisible by `scale`)
        kernel_size: per-group dilated conv kernel size (default 3)
        dilation:    dilation factor for each per-group conv
        scale:       number of sub-groups (default 4)
    """

    def __init__(self, C: int, kernel_size: int, dilation: int, scale: int = 4):
        super().__init__()
        assert C % scale == 0, f"C={C} must be divisible by scale={scale}"
        self.scale = scale
        K = C // scale
        p = dilation * (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            nn.Conv1d(K, K, kernel_size=kernel_size,
                      dilation=dilation, padding=p, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(K) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.scale, dim=1)
        out: List[torch.Tensor] = [chunks[0]]
        y: Optional[torch.Tensor] = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            xi = chunks[i + 1]
            if y is not None:
                xi = xi + y
            y = F.relu(bn(conv(xi)), inplace=True)
            out.append(y)
        return torch.cat(out, dim=1)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Learns a per-channel attention weight via global average pooling + MLP.
    The FiLM layer is applied AFTER SE (see FiLMConditionedSERes2NetBlock),
    so FiLM modulates the channel-recalibrated features, not the raw block output.

    Args:
        C:         number of channels
        reduction: SE bottleneck reduction factor
    """

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
        s = x.mean(dim=2)               # (B, C)
        s = self.fc(s).unsqueeze(2)     # (B, C, 1)
        return x * s                    # (B, C, T)


class FiLMConditionedSERes2NetBlock(nn.Module):
    """
    SE-Res2Net TDNN block with FiLM subject-conditioning.

    Flow:
      x  →  1×1 Conv+BN+ReLU  →  Res2Net  →  1×1 Conv+BN  →  SE
        →  FiLM(γ * · + β)  →  residual-add  →  ReLU  →  output

    FiLM is applied after SE attention so that the subject embedding
    modulates the channel-recalibrated features (not the raw residual).
    This ordering preserves the SE block's learned suppression of noisy
    EMG channels while allowing the encoder to adapt relative amplitude
    levels across subjects.

    At inference (test time): subject_emb = mean training embedding.
    FiLM then applies a fixed global modulation — NOT per-subject adaptation.

    Args:
        C:              channel width
        kernel_size:    Res2Net per-group conv kernel size
        dilation:       Res2Net dilation factor
        subject_emb_dim: dimension of subject embedding vector
        scale:          Res2Net scale (number of sub-groups)
        se_reduction:   SE bottleneck reduction factor
    """

    def __init__(
        self,
        C: int,
        kernel_size: int,
        dilation: int,
        subject_emb_dim: int,
        scale: int = 4,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.pointwise_in = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )
        self.res2          = Res2NetBlock(C, kernel_size=kernel_size,
                                          dilation=dilation, scale=scale)
        self.pointwise_out = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm1d(C),
        )
        self.se   = SEBlock(C, reduction=se_reduction)
        self.film = FiLMLayer(subject_emb_dim, C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, subject_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (B, C, T)
            subject_emb: (B, D) — one embedding per sample in batch,
                         or (D,) — broadcast over all samples,
                         or (1, D) — also broadcast.
        Returns:
            out: (B, C, T)
        """
        residual = x
        out = self.pointwise_in(x)       # (B, C, T)
        out = self.res2(out)             # (B, C, T)
        out = self.pointwise_out(out)    # (B, C, T)
        out = self.se(out)               # (B, C, T)

        # FiLM modulation: (B, C) or (1, C) → unsqueeze to (B, C, 1) or (1, C, 1)
        gamma, beta = self.film(subject_emb)   # each: (B, C) or (1, C)
        out = (1.0 + gamma).unsqueeze(-1) * out + beta.unsqueeze(-1)  # (B, C, T)

        out = self.relu(out + residual)  # (B, C, T)
        return out


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP).

    Collapses (B, C, T) → (B, 2C) via per-channel soft attention.
    Returns weighted mean AND weighted std, giving both temporal centre
    and temporal spread — more informative than mean-pooling alone and
    robust to inter-subject gesture-speed variation.

    LOSO safety: attention weights computed purely from input features —
    no running statistics, no stored per-subject state.
    """

    def __init__(self, C: int):
        super().__init__()
        hidden = max(C // 4, 16)
        self.attn = nn.Sequential(
            nn.Conv1d(C, hidden, kernel_size=1, bias=False),
            nn.Tanh(),
            nn.Conv1d(hidden, C, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.attn(x), dim=2)          # (B, C, T)
        mu    = (alpha * x).sum(dim=2)                  # (B, C)
        var   = (alpha * x.pow(2)).sum(dim=2) - mu.pow(2)
        sigma = var.clamp(min=1e-8).sqrt()               # (B, C)
        return torch.cat([mu, sigma], dim=1)             # (B, 2C)


# ══════════════════════ Full Selective Disentanglement Model ═════════════════

class SelectiveDisentanglementECAPA(nn.Module):
    """
    Selective Disentanglement ECAPA-TDNN for cross-subject EMG recognition.

    Key design decisions
    ─────────────────────
    • z (embedding): domain-AWARE — encoder learns to encode subject identity
      via auxiliary subject head + FiLM conditioning.  This preserves domain-
      correlated but task-relevant features (e.g., EMG amplitude envelope).

    • h = projection(z): domain-INVARIANT — trained with gradient reversal
      applied only to h, not to z.  The projection acts as a bottleneck that
      selectively forgets subject-specific information.

    • At inference: subject head and domain head are skipped (inference=True).
      FiLM uses mean_subject_emb (average of training embedding table).
      This is a fixed, global modulation — NOT per-subject adaptation.
      No test-subject signal data is ever used to compute mean_subject_emb.

    Parameters
    ──────────
    in_channels:      Number of EMG input channels (e.g. 8).
    num_classes:      Number of gesture classes.
    num_subjects_train: Number of training subjects in this LOSO fold.
    channels:         C — internal TDNN feature width (default 128).
    scale:            Res2Net sub-group count (default 4).
    embedding_dim:    E — encoder embedding dimension (default 128).
    subject_emb_dim:  D — subject embedding dimension for FiLM (default 32).
    proj_dim:         P — projection head output dimension (default 128).
    dilations:        Dilation per SE-Res2Net block (default [2, 3, 4]).
    dropout:          Dropout before projection (default 0.3).
    se_reduction:     SE bottleneck reduction factor (default 8).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_subjects_train: int,
        channels: int = 128,
        scale: int = 4,
        embedding_dim: int = 128,
        subject_emb_dim: int = 32,
        proj_dim: int = 128,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
        se_reduction: int = 8,
    ):
        super().__init__()

        if dilations is None:
            dilations = [2, 3, 4]
        if len(dilations) != 3:
            raise ValueError(
                f"Exactly 3 dilation values required (one per block), "
                f"got {len(dilations)}: {dilations}"
            )

        self.channels          = channels
        self.embedding_dim     = embedding_dim
        self.subject_emb_dim   = subject_emb_dim
        self.proj_dim          = proj_dim
        self.num_subjects_train = num_subjects_train
        num_blocks             = len(dilations)

        # ── Subject embedding lookup table ────────────────────────────────
        # Learnable: one D-dim vector per training subject.
        # At test time we use the mean of all rows — no test data involved.
        self.subject_embeddings = nn.Embedding(num_subjects_train, subject_emb_dim)
        nn.init.normal_(self.subject_embeddings.weight, mean=0.0, std=0.1)

        # ── Initial TDNN ──────────────────────────────────────────────────
        self.init_tdnn = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # ── FiLM-conditioned SE-Res2Net blocks ────────────────────────────
        self.blocks = nn.ModuleList([
            FiLMConditionedSERes2NetBlock(
                channels,
                kernel_size=3,
                dilation=d,
                subject_emb_dim=subject_emb_dim,
                scale=scale,
                se_reduction=se_reduction,
            )
            for d in dilations
        ])

        # ── Multi-layer Feature Aggregation (MFA) ─────────────────────────
        mfa_in = channels * num_blocks   # 3C
        self.mfa = nn.Sequential(
            nn.Conv1d(mfa_in, mfa_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(mfa_in),
            nn.ReLU(inplace=True),
        )

        # ── Attentive Statistics Pooling ──────────────────────────────────
        self.asp = AttentiveStatisticsPooling(mfa_in)

        # ── Domain-aware embedding z ──────────────────────────────────────
        asp_out_dim = mfa_in * 2   # 6C
        self.embedding = nn.Sequential(
            nn.Linear(asp_out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── Auxiliary subject head (training only) ────────────────────────
        # Applied to z — encourages encoder to capture subject identity.
        # Not used at inference; not involved in LOSO test data at all.
        self.subject_head = nn.Linear(embedding_dim, num_subjects_train)

        # ── Projection head → domain-invariant features h ─────────────────
        # GRL is applied to h before domain_head, keeping z unrestricted.
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # ── Gesture classifier (on h, domain-invariant space) ─────────────
        self.classifier = nn.Linear(proj_dim, num_classes)

        # ── Domain confusion head (GRL applied to h before this) ──────────
        # Forces h to lose subject discriminability while z retains it.
        self.domain_head = nn.Linear(proj_dim, num_subjects_train)

        # ── Mean subject embedding buffer (for test-time FiLM) ────────────
        # Initialised to zeros (→ identity FiLM since γ_linear is zero-init).
        # Populated by compute_mean_subject_embedding() after training.
        self.register_buffer(
            "mean_subject_emb",
            torch.zeros(subject_emb_dim),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear,)):
                # Skip FiLM linear layers — they have their own zero-init
                if not isinstance(m.weight, torch.Tensor):
                    continue
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Compute mean subject embedding ────────────────────────────────────────

    def compute_mean_subject_embedding(self) -> None:
        """
        Compute and store the mean of all learned training subject embeddings.

        Called once after training completes.

        LOSO safety:
          Uses ONLY model parameters (subject_embeddings.weight) — a (N_subj, D)
          table learned during training.  No test-subject signal data is involved.
          The resulting mean vector is a fixed global representation of the
          "average training subject" used for FiLM conditioning at inference.
        """
        with torch.no_grad():
            mean_emb = self.subject_embeddings.weight.mean(dim=0)  # (D,)
        self.mean_subject_emb.copy_(mean_emb)

    # ── Internal encoder ──────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor, subject_emb: torch.Tensor) -> torch.Tensor:
        """
        Run ECAPA-TDNN backbone with FiLM conditioning.

        Args:
            x:           (B, C_emg, T)
            subject_emb: (B, D) — per-sample embeddings during training,
                         or (D,) / (1, D) — broadcast during inference.
        Returns:
            z: (B, embedding_dim) — domain-aware embedding.
        """
        out = self.init_tdnn(x)                       # (B, C, T)
        block_outputs = []
        for block in self.blocks:
            out = block(out, subject_emb)             # (B, C, T)
            block_outputs.append(out)
        mfa_in = torch.cat(block_outputs, dim=1)      # (B, 3C, T)
        mfa_out = self.mfa(mfa_in)                    # (B, 3C, T)
        pooled  = self.asp(mfa_out)                   # (B, 6C)
        z       = self.embedding(pooled)              # (B, E)
        return z

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        inference: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x:           (B, C_emg, T) — EMG windows, channels-first.
            subject_ids: (B,) int64 — subject indices [0, num_subjects_train).
                         Required when inference=False (training mode).
                         Ignored when inference=True.
            alpha:       GRL reversal strength (annealed during training,
                         irrelevant at inference).
            inference:   If True, use mean_subject_emb for FiLM; skip
                         auxiliary heads.  Set to True for all evaluation.

        Returns:
            gesture_logits: (B, num_classes)
            subject_logits: (B, num_subjects_train) or None at inference.
            domain_logits:  (B, num_subjects_train) or None at inference.

        LOSO safety:
          When inference=True, mean_subject_emb is used for FiLM.
          This tensor was computed solely from model parameters after training.
          No test-subject signal data ever flows into this path.
        """
        if inference:
            # Test time: broadcast mean training embedding over batch
            subj_emb = self.mean_subject_emb  # (D,) — will be broadcast by FiLMLayer
        else:
            if subject_ids is None:
                raise ValueError("subject_ids must be provided during training (inference=False).")
            subj_emb = self.subject_embeddings(subject_ids)  # (B, D)

        # ── Domain-aware encoding ─────────────────────────────────────────
        z = self._encode(x, subj_emb)    # (B, E)

        # ── Projection → domain-invariant space ───────────────────────────
        h = self.projection(z)           # (B, P)

        # ── Gesture classification ─────────────────────────────────────────
        gesture_logits = self.classifier(h)   # (B, num_classes)

        if inference:
            return gesture_logits, None, None

        # ── Auxiliary subject classification on z (training only) ─────────
        subject_logits = self.subject_head(z)              # (B, num_subjects)

        # ── Domain confusion on h with GRL (training only) ────────────────
        # Gradient reversal on h (not on z): encoder remains expressive,
        # only the projection learns to suppress domain information.
        h_rev         = grad_reverse(h, alpha)
        domain_logits = self.domain_head(h_rev)            # (B, num_subjects)

        return gesture_logits, subject_logits, domain_logits

    # ── Utility ───────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
