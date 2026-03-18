"""
MI-Disentangled CNN-GRU: Subject-adversarial training via CLUB MI upper bound.

Hypothesis H5b: Replace distance correlation (exp_31) with CLUB (Contrastive
Log-ratio Upper Bound) for more principled MI minimization between z_content
and subject identity.

Core idea (Cheng et al., ICML 2020 — «CLUB»):
    Model q_θ(z | subject) as a diagonal Gaussian (per-subject embedding).
    MI upper bound:
        Î_CLUB(z; y) = E_{(z,y)}[log q(z|y)] - E_{z, ỹ~shuffle}[log q(z|ỹ)]
    Minimizing Î_CLUB over z pushes z to be independent of subject.

Two CLUB instances are used:
    club_content: Î(z_content; subject)  — MINIMIZE → subject-invariant content
    club_style:   Î(z_style;   subject)  — MAXIMIZE → style encodes subject

Full training objective per batch:
    L = L_gesture(z_content) + α·L_subject(z_style)
      + β·Î(z_content; subject)          ← minimize (β ≥ 0)
      − γ·Î(z_style;   subject)          ← maximize (γ ≥ 0)

Two-step update per batch (separate optimizers):
    Step 1 — Update q_θ networks on z.detach():
        maximize E[log q(z|y)]  ↔  minimize –E[log q(z|y)]
    Step 2 — Update main model (CLUB frozen):
        minimize L above; gradients flow from Î through z to encoder

LOSO safety:
    - CLUB is trained on training-fold subjects only
    - test subject never touches CLUB training or subject probe
    - Subject probe (linear classifier on z_content) uses only training subjects

Architecture is identical to DisentangledCNNGRU (exp_31):
    Input (B, C, T) → SharedEncoder (CNN + BiGRU + Attention) → (B, 256)
        ├── ContentHead → z_content (B, content_dim) → GestureClassifier
        └── StyleHead   → z_style   (B, style_dim)   → SubjectClassifier

Reference:
    Cheng, P. et al. "CLUB: A Contrastive Log-ratio Upper Bound of Mutual
    Information Estimation." ICML 2020. https://arxiv.org/abs/2006.12013
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# CLUB Estimator
# ═══════════════════════════════════════════════════════════════════════════


class CLUBEstimator(nn.Module):
    """
    Contrastive Log-ratio Upper Bound (CLUB) of MI for discrete conditioning.

    Models p(z | subject_id) as a diagonal Gaussian:
        q_θ(z | y) = N(μ_θ(y), diag(σ_θ²(y)))

    where μ_θ, σ_θ are two-layer MLPs on top of a shared subject embedding.

    Two operations (called from the trainer's two-step update):

        .learning_loss(z_detached, y)
            Train q_θ: minimize −E[log q_θ(z|y)].
            Call with z.detach() — gradients MUST NOT flow to the main encoder.

        .mi_upper_bound(z, y)
            Compute Î_CLUB(z; y) with gradients flowing through z.
            CLUB parameters are frozen during this step (separate optimizer).

    Args:
        z_dim:          Dimensionality of the continuous latent vector z.
        num_conditions: Number of discrete conditions (= number of training
                        subjects in the current LOSO fold).
        hidden_dim:     Width of the MLP layers in the Gaussian head.
    """

    _LOG_2PI: float = math.log(2.0 * math.pi)

    def __init__(self, z_dim: int, num_conditions: int, hidden_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.num_conditions = num_conditions

        # Shared subject embedding (learned independently for μ and logvar)
        self.embed = nn.Embedding(num_conditions, hidden_dim)

        # μ head: embedding → z_dim
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

        # log σ² head: embedding → z_dim (clamped at forward time)
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_params(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y: (B,) LongTensor of subject indices in [0, num_conditions).
        Returns:
            mu:     (B, z_dim)
            logvar: (B, z_dim) — clamped to [−6, 6] for numerical stability.
        """
        h = self.embed(y)                               # (B, hidden_dim)
        mu = self.mu_head(h)                            # (B, z_dim)
        logvar = torch.clamp(self.logvar_head(h), -6.0, 6.0)
        return mu, logvar

    def _log_prob(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        log N(z ; μ(y), diag(exp(logvar(y)))) for each sample.

        Returns:
            (B,) per-sample log-probability.
        """
        mu, logvar = self._get_params(y)
        # log N = −0.5 · [D·log2π + Σ logvar_d + Σ (z_d − μ_d)²/exp(logvar_d)]
        log_p = -0.5 * (
            self.z_dim * self._LOG_2PI
            + logvar.sum(dim=-1)
            + ((z - mu).pow(2) / logvar.exp().clamp(min=1e-8)).sum(dim=-1)
        )
        return log_p  # (B,)

    # ------------------------------------------------------------------
    # Public API called from the trainer
    # ------------------------------------------------------------------

    def learning_loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss for training q_θ.

        MUST be called with z = z.detach() so that gradients do NOT
        propagate to the main encoder during the CLUB update step.

        Returns:
            Scalar; minimize this to improve the variational approximation.
        """
        return -self._log_prob(z.detach(), y).mean()

    def mi_upper_bound(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        CLUB MI upper bound Î_CLUB(z; y).

        Î = E_{(z,y)}[log q(z|y)] − E_{z, ỹ}[log q(z|ỹ)]
        where ỹ is a random permutation of y within the batch.

        Gradients flow through z → main encoder.  CLUB parameters accumulate
        gradients here but are not updated (separate optimizer handles that
        only in the learning_loss step).

        Returns:
            Scalar (can be negative — the bound is not constrained positive,
            which is fine; minimizing it still reduces subject information).
        """
        B = z.size(0)
        if B < 2:
            # Differentiable zero — allows backward() without error
            return (z * 0.0).sum()

        log_paired = self._log_prob(z, y)               # (B,)

        # Random permutation of labels within the batch
        perm_idx = torch.randperm(B, device=z.device)
        log_shuffled = self._log_prob(z, y[perm_idx])   # (B,)

        return (log_paired - log_shuffled).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Model Components (identical to exp_31 for fair comparison)
# ═══════════════════════════════════════════════════════════════════════════


class SharedEncoder(nn.Module):
    """
    CNN → BiGRU → Attention pooling.

    Input:  (B, C, T)   — channels-first, same as exp_31.
    Output: (B, gru_hidden * 2)  — bidirectional context vector.
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
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_output_dim = gru_hidden * 2

        self.attention = nn.Sequential(
            nn.Linear(self.gru_output_dim, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)                         # (B, cnn[-1], T')
        h = h.transpose(1, 2)                   # (B, T', cnn[-1])
        gru_out, _ = self.gru(h)                # (B, T', gru_hidden*2)

        attn_w = self.attention(gru_out)         # (B, T', 1)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (attn_w * gru_out).sum(dim=1)  # (B, gru_hidden*2)
        return context


class ProjectionHead(nn.Module):
    """Two-layer MLP projecting the shared representation to a latent subspace."""

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


# ═══════════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════════


class MIDisentangledCNNGRU(nn.Module):
    """
    Content-Style Disentangled CNN-GRU for subject-invariant gesture recognition.

    Architecture (identical to exp_31's DisentangledCNNGRU for fair comparison):
        Input (B, C, T)
            → SharedEncoder (CNN + BiGRU + Attention) → shared (B, 256)
            ├── ContentHead → z_content (B, content_dim) → GestureClassifier
            └── StyleHead   → z_style   (B, style_dim)   → SubjectClassifier

    The CLUB estimators live OUTSIDE this model (in the trainer) to keep
    model parameters and CLUB parameters on separate optimizers.

    Training mode (model.training or return_all=True):
        Returns dict with keys: gesture_logits, subject_logits, z_content, z_style.

    Inference mode (model.eval(), return_all=False):
        Returns gesture_logits only — no subject information needed at test time.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_gestures: int = 10,
        num_subjects: int = 4,
        content_dim: int = 128,
        style_dim: int = 64,
        cnn_channels: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim

        self.encoder = SharedEncoder(
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        shared_dim = self.encoder.gru_output_dim  # gru_hidden * 2

        self.content_head = ProjectionHead(shared_dim, content_dim, dropout)
        self.style_head = ProjectionHead(shared_dim, style_dim, dropout)

        self.gesture_classifier = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim // 2, num_gestures),
        )

        # Subject classifier on z_style — used only during training
        self.subject_classifier = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, num_subjects),
        )

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x:          (B, C, T) — channels-first tensor.
            return_all: If True, return full dict even in eval mode.

        Returns:
            training or return_all=True:
                {
                    "gesture_logits": (B, num_gestures),
                    "subject_logits": (B, num_subjects),
                    "z_content":      (B, content_dim),
                    "z_style":        (B, style_dim),
                }
            eval (default):
                gesture_logits: (B, num_gestures)
        """
        shared = self.encoder(x)                         # (B, shared_dim)
        z_content = self.content_head(shared)            # (B, content_dim)
        z_style = self.style_head(shared)                # (B, style_dim)
        gesture_logits = self.gesture_classifier(z_content)

        if self.training or return_all:
            subject_logits = self.subject_classifier(z_style)
            return {
                "gesture_logits": gesture_logits,
                "subject_logits": subject_logits,
                "z_content": z_content,
                "z_style": z_style,
            }
        return gesture_logits
