"""
JigsawTemporalEMGNet — Dual-head model for Temporal Order Invariance training.

Architecture overview
---------------------
    Shared encoder:  Conv1D blocks  (N, C, T) → (N, d_enc, T')
    Context:         Bidirectional GRU         → (N, T', 2·d_ctx)
    Attention pool:                            → (N, 2·d_ctx)
    Gesture head:    Linear                    → (N, num_classes)
    Jigsaw head:     Linear                    → (N, num_perms)   [training only]

Training protocol
-----------------
    At every training step, each window is permuted with a random element
    from a fixed vocabulary of constrained permutations (adjacent-swap only).
    The model must simultaneously:
        (1) predict the gesture class  [primary task, weight α]
        (2) predict which permutation was applied  [aux task, weight 1-α]

    Auxiliary loss encourages the encoder to explicitly separate
    "what gesture" information from "what temporal ordering" information,
    making the gesture head more robust to subject-specific timing variations.

Evaluation (strictly LOSO-compliant)
--------------------------------------
    At inference time NO permutation is applied.
    Only gesture_head output is used.
    No test-subject data is ever seen during fit().

Permutation vocabulary
----------------------
    `generate_constrained_permutations(n_chunks, n_perms, max_swaps, seed)` builds
    a fixed, reproducible set of permutations where each is obtained from the
    identity by at most `max_swaps` adjacent transpositions.
    Index 0 is always the identity (no reordering).

    With n_chunks=8 and max_swaps=2, each chunk is ~37.5 ms (at 2 kHz, T=600),
    so permutations represent local ±37–75 ms temporal shifts — subtle enough
    not to destroy muscle-activation content but sufficient to vary timing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Permutation vocabulary utilities
# ---------------------------------------------------------------------------

def generate_constrained_permutations(
    n_chunks: int,
    n_perms: int,
    max_swaps: int = 2,
    seed: int = 0,
) -> List[List[int]]:
    """
    Build a fixed, reproducible vocabulary of constrained permutations.

    Each permutation is derived from the identity [0, 1, …, n_chunks-1]
    by applying 1..max_swaps randomly chosen *adjacent* transpositions.
    This keeps temporal disruption local: neighbouring chunks swap places,
    while chunks far apart in time remain in their original relative order.

    Permutation index 0 is always the identity (no reordering).

    Args:
        n_chunks:  number of temporal segments the window is divided into.
        n_perms:   target vocabulary size (including identity).
        max_swaps: maximum number of adjacent swaps per permutation.
        seed:      fixed seed for reproducibility across all LOSO folds.

    Returns:
        List of n_perms permutations, each a list of length n_chunks.
        If fewer than n_perms distinct permutations can be generated within
        the constraint, the function returns what it found.
    """
    rng = np.random.RandomState(seed)
    identity = list(range(n_chunks))
    perms: List[List[int]] = [list(identity)]
    seen = {tuple(identity)}

    max_attempts = 20_000
    attempts = 0
    while len(perms) < n_perms and attempts < max_attempts:
        attempts += 1
        n_swaps = rng.randint(1, max_swaps + 1)
        perm = list(identity)
        for _ in range(n_swaps):
            i = int(rng.randint(0, n_chunks - 1))
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            perms.append(perm)

    return perms


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _CNNEncoder(nn.Module):
    """
    Three-block 1-D CNN encoder.

    Input:  (N, C, T)
    Output: (N, d_enc, T')  where T' ≈ T / 4  (two MaxPool2 layers)
    """

    def __init__(self, in_channels: int, d_enc: int = 128, dropout: float = 0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, d_enc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_enc),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class _AttentionPool(nn.Module):
    """
    Soft attention pooling: (N, T', d) → (N, d).

    A single linear layer produces unnormalised attention scores over the
    time axis; softmax turns them into a probability distribution that is
    used to form a weighted sum of the time-step representations.
    """

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T', d)
        w = torch.softmax(self.score(x), dim=1)   # (N, T', 1)
        return (x * w).sum(dim=1)                  # (N, d)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class JigsawTemporalEMGNet(nn.Module):
    """
    Dual-head EMG classifier with temporal-order invariance training.

    At training time the model receives windows that have been permuted
    by one of the `num_perms` constrained permutations and must predict:
        1. the gesture class  (gesture_head)
        2. which permutation was applied  (jigsaw_head)

    At inference time call with `return_jigsaw=False` (default).
    Only gesture_logits are returned; no permutation is needed.

    Args:
        in_channels: number of EMG channels (C).
        num_classes:  number of gesture classes.
        num_perms:    size of the permutation vocabulary.
        d_enc:        number of channels in the CNN encoder output.
        d_ctx:        hidden size of the GRU (one direction).
        dropout:      dropout probability applied throughout.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_perms: int = 30,
        d_enc: int = 128,
        d_ctx: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = _CNNEncoder(in_channels, d_enc=d_enc, dropout=dropout)

        self.rnn = nn.GRU(
            input_size=d_enc,
            hidden_size=d_ctx,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        d_total = d_ctx * 2   # bidirectional output
        self.pool = _AttentionPool(d_total)
        self.drop = nn.Dropout(dropout)

        self.gesture_head = nn.Linear(d_total, num_classes)
        self.jigsaw_head  = nn.Linear(d_total, num_perms)

    def forward(
        self,
        x: torch.Tensor,
        return_jigsaw: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:             (N, C, T) input windows (already normalised).
            return_jigsaw: if True, also compute and return jigsaw logits.

        Returns:
            gesture_logits: (N, num_classes)
            jigsaw_logits:  (N, num_perms) when return_jigsaw=True, else None.

        LOSO compliance note:
            During evaluation `return_jigsaw` is always False and no
            permutation is applied to the input.  The jigsaw_head is never
            used to influence predictions on the test (held-out) subject.
        """
        enc = self.encoder(x)             # (N, d_enc, T')
        enc_t = enc.permute(0, 2, 1)      # (N, T', d_enc)
        rnn_out, _ = self.rnn(enc_t)      # (N, T', 2*d_ctx)
        feat = self.pool(rnn_out)         # (N, 2*d_ctx)
        feat = self.drop(feat)

        gesture_logits = self.gesture_head(feat)  # (N, num_classes)

        jigsaw_logits: Optional[torch.Tensor] = None
        if return_jigsaw:
            jigsaw_logits = self.jigsaw_head(feat)  # (N, num_perms)

        return gesture_logits, jigsaw_logits
