"""
CPC-EMG: Contrastive Predictive Coding for EMG signals.

Reference: van den Oord et al., "Representation Learning with Contrastive
Predictive Coding" (NeurIPS 2018).
Extended with optional Gumbel-softmax VQ quantization (wav2vec2-style targets).

Architecture overview
---------------------
Pretrain phase:
    (B, C, T) → CPCEncoder  → z : (B, d_enc, T')   # strided CNN
    z.T       → causal GRU  → c : (B, T', d_ctx)   # AR context
    (optional) z → GumbelVQ → q: (B, T', d_enc)    # discrete targets
    c_t → W_k → pred_{t,k}  → InfoNCE(pred, z_{t+k})  for k=1..K

Fine-tune phase:
    (B, C, T) → CPCEncoder → global average pool → Linear classifier

Key LOSO invariant
------------------
Both phases operate only on training-subject data.  The test subject
is NEVER passed to any of these models during training.

Shapes
------
All PyTorch convention: (B, C, T)  i.e.  channels first.
For T=600, C=8 with default strides: T' = 60.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CPCEncoder(nn.Module):
    """
    Strided 1-D CNN feature encoder.

    Input:  (B, C, T)       — C=8 EMG channels, T=600 time steps
    Output: (B, d_enc, T')  — T'=60 with default configuration

    Dimension computation for T=600 (default):
        Layer 1: Conv1d(C → 64,   k=10, s=5, p=3)  → T = (600+6-10)//5+1 = 120
        Layer 2: Conv1d(64 → 128, k=4,  s=2, p=1)  → T = (120+2-4)//2+1  =  60
        Layer 3: Conv1d(128→d_enc,k=3,  s=1, p=1)  → T =  60  (unchanged)

    GroupNorm(1, C) is used instead of BatchNorm so that the encoder
    statistics do not depend on batch composition (important for LOSO).
    """

    def __init__(self, in_channels: int, d_enc: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_enc = d_enc
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=10, stride=5, padding=3),
            nn.GroupNorm(1, 64),   # LayerNorm-equivalent per sample
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, d_enc, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, d_enc),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, d_enc, T')"""
        return self.layers(x)

    def output_length(self, input_length: int) -> int:
        """Compute T' for a given input length T (for sanity checks)."""
        t = (input_length + 6 - 10) // 5 + 1   # layer 1
        t = (t + 2 - 4) // 2 + 1               # layer 2
        # layer 3: stride=1, same padding → same length
        return t


# ---------------------------------------------------------------------------
# Gumbel-softmax VQ quantizer  (optional)
# ---------------------------------------------------------------------------

class GumbelVQQuantizer(nn.Module):
    """
    Product-quantization with Gumbel-softmax relaxation (wav2vec2 style).

    Maps continuous encoder representations to discrete codebook vectors,
    using Gumbel noise + softmax for differentiable training.  During
    inference the hard argmax is used.

    Input:  (B, T', d_enc)
    Output: (B, T', d_enc)  — projected back to encoder dimension
            diversity_loss  — scalar, encourages uniform codebook usage

    Args:
        d_enc:        encoder dimension (input and output)
        num_vars:     codebook size per group  (V)
        num_groups:   number of codebook groups  (G)
        temperature:  initial Gumbel temperature
        min_temp:     minimum temperature after annealing
    """

    def __init__(
        self,
        d_enc: int,
        num_vars: int = 320,
        num_groups: int = 2,
        temperature: float = 2.0,
        min_temp: float = 0.5,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.num_groups = num_groups
        self.temperature = temperature
        self.min_temp = min_temp

        d_group = max(d_enc // num_groups, 1)
        self.d_group = d_group

        # Project encoder output → group logits
        self.proj_in = nn.Linear(d_enc, num_groups * num_vars, bias=False)
        # Codebooks: one row per (group, var) pair
        self.codebooks = nn.Parameter(
            torch.randn(num_groups, num_vars, d_group) * 0.1
        )
        # Project concatenated group codes → d_enc
        self.proj_out = nn.Linear(num_groups * d_group, d_enc, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T', d_enc)
        Returns:
            quantized: (B, T', d_enc)
            diversity_loss: scalar (positive → minimize to encourage usage)
        """
        B, T, D = x.shape

        # Compute raw logits for all groups/vars
        logits = self.proj_in(x)                          # (B, T', G*V)
        logits = logits.view(B, T, self.num_groups, self.num_vars)

        if self.training:
            # Gumbel noise: u ~ Uniform(0,1), gumbel = -log(-log(u))
            u = torch.zeros_like(logits).uniform_().clamp(1e-9, 1 - 1e-9)
            gumbel_noise = -torch.log(-torch.log(u))
            logits = (logits + gumbel_noise) / self.temperature
        else:
            logits = logits / self.temperature

        # Soft probabilities (used for gradient flow)
        soft = logits.softmax(dim=-1)                     # (B, T', G, V)

        # Hard one-hot via argmax + straight-through estimator
        indices = soft.detach().argmax(dim=-1, keepdim=True)  # (B, T', G, 1)
        hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
        probs = hard - soft.detach() + soft               # (B, T', G, V) — STE

        # Lookup codewords per group: einsum bTGV, GVd -> bTGd
        quantized_g = torch.einsum('btgv,gvd->btgd', probs, self.codebooks)
        # Flatten groups: (B, T', G*d_group)
        quantized_g = quantized_g.reshape(B, T, self.num_groups * self.d_group)
        # Project back to encoder dimension
        quantized = self.proj_out(quantized_g)            # (B, T', d_enc)

        # Diversity loss: encourage all codebook entries to be used equally.
        # avg_probs[g, v] = mean probability mass on code v in group g.
        avg_probs = soft.mean(dim=(0, 1))                 # (G, V)
        # Negative entropy (per group), averaged: lower = more uniform usage
        neg_entropy = (avg_probs * (avg_probs + 1e-9).log()).sum(dim=-1).mean()
        max_entropy = math.log(self.num_vars)
        # Normalise to [0, 1]: 0 = perfectly uniform, 1 = all mass on one code
        diversity_loss = (max_entropy + neg_entropy) / max_entropy

        return quantized, diversity_loss

    def decay_temperature(self) -> None:
        """Call once per training step to anneal Gumbel temperature."""
        self.temperature = max(self.min_temp, self.temperature * 0.999995)


# ---------------------------------------------------------------------------
# CPC pretraining model
# ---------------------------------------------------------------------------

class CPCPretrainModel(nn.Module):
    """
    Full CPC pretraining model.

    Pretraining objective: given the causal context c_t, predict the
    encoder representation z_{t+k} (or its quantized version q_{t+k})
    for k = 1 … K using an InfoNCE loss.  Negatives are drawn from
    OTHER sequences in the same mini-batch (batch negatives).

    No gesture labels are used.  Only train-subject data is ever fed
    to this model; the test subject is completely held out (LOSO).

    Args:
        in_channels:    number of EMG channels  (C)
        d_enc:          encoder latent dimension
        d_ctx:          GRU context dimension
        K:              number of future steps to predict
        use_quantizer:  if True, quantize targets with GumbelVQ
        num_vars:       VQ codebook size per group
        num_groups:     VQ product quantization groups
        dropout:        dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        d_enc: int = 256,
        d_ctx: int = 256,
        K: int = 12,
        use_quantizer: bool = False,
        num_vars: int = 320,
        num_groups: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.d_ctx = d_ctx
        self.K = K
        self.use_quantizer = use_quantizer

        self.encoder = CPCEncoder(in_channels, d_enc, dropout)

        if use_quantizer:
            self.quantizer: Optional[GumbelVQQuantizer] = GumbelVQQuantizer(
                d_enc=d_enc,
                num_vars=num_vars,
                num_groups=num_groups,
            )
        else:
            self.quantizer = None

        # Causal autoregressive context model.
        # Processes z_1 … z_t → c_t (no look-ahead → valid predictive setup).
        self.ar_model = nn.GRU(
            input_size=d_enc,
            hidden_size=d_ctx,
            num_layers=1,
            batch_first=True,
            bidirectional=False,   # causal: only past
        )

        # K separate linear predictors W_1 … W_K: c_t → pred of z_{t+k}
        self.predictors = nn.ModuleList([
            nn.Linear(d_ctx, d_enc, bias=False) for _ in range(K)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C, T)  — batch of EMG windows from train subjects only
        Returns:
            cpc_loss:       scalar InfoNCE loss
            diversity_loss: scalar VQ diversity loss, or None if no quantizer
        """
        # Step 1: Encode
        z = self.encoder(x)          # (B, d_enc, T')
        z = z.transpose(1, 2)        # (B, T', d_enc)

        # Step 2: Compute targets (optionally quantized)
        diversity_loss: Optional[torch.Tensor] = None
        if self.use_quantizer and self.quantizer is not None:
            targets, diversity_loss = self.quantizer(z)  # (B, T', d_enc), scalar
        else:
            targets = z                                   # continuous targets

        # Step 3: Causal context with GRU
        # context[:, t, :] = GRU(z[:, :t+1, :]) — uses only past positions
        context, _ = self.ar_model(z)   # (B, T', d_ctx)

        # Step 4: InfoNCE loss
        cpc_loss = self._cpc_infonce(context, targets)

        return cpc_loss, diversity_loss

    def _cpc_infonce(
        self, context: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE loss using batch negatives.

        For each prediction step k and each sampled time position t:
          - positive pair : (W_k(c_t),  z_{t+k})  — same sequence
          - negative pairs: (W_k(c_t),  z'_{t+k}) — other sequences in batch

        The score matrix is (n_sample × B × B); the diagonal holds
        all positive scores.

        Memory: O(n_sample × B²) per prediction step.
        With n_sample=16, B=256: ~4 MB per step → negligible.

        LOSO guarantee: `targets` and `context` are built from train-subject
        data only; the test subject is never passed here.
        """
        B, T, _ = context.shape
        # Sample at most 16 time positions per k to keep memory bounded
        n_sample = min(16, max(1, T // 2))

        total_loss = torch.zeros(1, device=context.device, dtype=context.dtype)
        count = 0

        for k, predictor in enumerate(self.predictors, start=1):
            T_avail = T - k     # valid source positions: 0 … T_avail-1
            if T_avail <= 0:
                break

            actual_n = min(n_sample, T_avail)
            # Random subset of time positions (without replacement)
            perm = torch.randperm(T_avail, device=context.device)[:actual_n]

            c_t   = context[:, perm, :]       # (B, actual_n, d_ctx)
            z_tk  = targets[:, perm + k, :]   # (B, actual_n, d_enc) — positive

            # Predict: W_k(c_t)
            pred  = predictor(c_t)             # (B, actual_n, d_enc)

            # L2-normalise for cosine similarity scores
            pred_n = F.normalize(pred,  dim=-1)   # (B, actual_n, d_enc)
            z_n    = F.normalize(z_tk,  dim=-1)   # (B, actual_n, d_enc)

            # Reorder to (actual_n, B, d_enc) for batched matmul
            pred_t = pred_n.permute(1, 0, 2)      # (actual_n, B, d_enc)
            z_t    = z_n.permute(1, 0, 2)         # (actual_n, B, d_enc)

            # scores[s, b, b'] = pred[b, s] · z[b', s]
            scores = torch.bmm(
                pred_t, z_t.transpose(1, 2)
            )                                      # (actual_n, B, B)

            # Correct match: diagonal → labels = [0, 1, 2, ..., B-1]
            labels = torch.arange(B, device=context.device)          # (B,)
            labels = labels.unsqueeze(0).expand(actual_n, B)         # (actual_n, B)

            loss_k = F.cross_entropy(
                scores.reshape(actual_n * B, B),   # (actual_n*B, B)
                labels.reshape(actual_n * B),      # (actual_n*B,)
            )
            total_loss = total_loss + loss_k
            count += 1

        return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Classifier for fine-tuning
# ---------------------------------------------------------------------------

class CPCClassifier(nn.Module):
    """
    Classifier head on top of a pretrained CPCEncoder.

    Fine-tuning only uses the encoder (no AR model): the encoder is
    initialised from CPC pretrained weights and fine-tuned end-to-end.
    The GRU context network is discarded — global average pooling over
    the temporal dimension serves as the aggregation step instead.

    Input:  (B, C, T)
    Output: (B, num_classes) logits

    Args:
        in_channels:  number of EMG channels  (C)
        num_classes:  number of gesture classes
        d_enc:        encoder latent dimension (must match pretrained encoder)
        dropout:      dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_enc: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = CPCEncoder(in_channels, d_enc, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_enc, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, num_classes)"""
        z = self.encoder(x)   # (B, d_enc, T')
        z = z.mean(dim=-1)    # (B, d_enc) — global average pool over time
        return self.classifier(z)

    def load_pretrained_encoder(self, pretrain_model: CPCPretrainModel) -> None:
        """
        Copy encoder weights from a pretrained CPCPretrainModel.

        Called after pretraining to transfer learned representations to
        the fine-tuning classifier without leaking any test-subject info.
        """
        self.encoder.load_state_dict(pretrain_model.encoder.state_dict())
