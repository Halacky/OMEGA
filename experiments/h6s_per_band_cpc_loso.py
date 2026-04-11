#!/usr/bin/env python3
"""
H6s: Per-Band CPC (Confirmatory, Family A).

Tests whether per-band Contrastive Predictive Coding improves cross-subject
sEMG gesture recognition over (a) supervised baseline and (b) global CPC
on concatenated features.

Variants
────────
  no_cpc:          Supervised baseline (UVMD + MixStyle, no SSL)
  global_cpc:      CPC pretrain on concatenated encoder output -> fine-tune
  per_band_cpc:    FreqAwareCPC pretrain with cross_band=False -> fine-tune
  cross_band_cpc:  FreqAwareCPC pretrain with cross_band=True -> fine-tune

Protocol (per LOSO fold)
────────────────────────
  no_cpc:
    Supervised training: 80 epochs, lr=1e-3, early stopping

  global_cpc / per_band_cpc / cross_band_cpc:
    Phase 1 — SSL Pretrain: 50 ep, batch=512, lr=1e-3, cosine decay
    Phase 2 — Fine-tune: linear probe (10 ep) + full (50 ep, lr=1e-4)

  All variants:
    Evaluate on held-out test subject: accuracy, macro-F1

Statistical design
──────────────────
  Family A (k=2): per_band_cpc vs no_cpc, per_band_cpc vs global_cpc
  Wilcoxon signed-rank + Holm-Bonferroni correction (alpha=0.05)

Usage
─────
  python experiments/h6s_per_band_cpc_loso.py                       # CI
  python experiments/h6s_per_band_cpc_loso.py --full                 # 20 subj
  python experiments/h6s_per_band_cpc_loso.py --variants no_cpc per_band_cpc
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

# ── project imports ──────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.base import ProcessingConfig
from data.multi_subject_loader import MultiSubjectLoader
from models.uvmd_ssl_encoder import UVMDSSLEncoder, UVMDSSLClassifier
from models.freq_aware_cpc import FreqAwareCPC
from models.cpc_emg import CPCEncoder
from utils.logging import setup_logging

# ── constants ────────────────────────────────────────────────────────
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]
_FULL_SUBJECTS = [
    f"DB2_s{i}" for i in
    [1, 2, 3, 4, 5, 11, 12, 13, 14, 15,
     26, 27, 28, 29, 30, 36, 37, 38, 39, 40]
]

SEED = 42
WINDOW_SIZE = 200
WINDOW_OVERLAP = 100
SAMPLING_RATE = 2000
N_CHANNELS = 12

# UVMD architecture
K_MODES = 4
L_LAYERS = 8
ALPHA_INIT = 2000.0
TAU_INIT = 0.01
FEAT_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.3

# SSL pretraining
SSL_EPOCHS = 50
SSL_BATCH = 512
SSL_LR = 1e-3
SSL_WARMUP_EPOCHS = 5

# Fine-tuning
FT_EPOCHS = 50
FT_BATCH = 512
FT_LR = 1e-4
FT_PATIENCE = 15
PROBE_EPOCHS = 10
PROBE_LR = 1e-3

# Supervised baseline
SUP_EPOCHS = 80
SUP_LR = 1e-3
SUP_PATIENCE = 15

# Common
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
OVERLAP_LAMBDA = 0.01
OVERLAP_SIGMA = 0.05
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1

# CPC
CPC_D_ENC = 128
CPC_D_CTX = 128
CPC_K_STEPS = 8

VARIANT_NAMES = ["no_cpc", "global_cpc", "per_band_cpc", "cross_band_cpc"]
VARIANT_LABELS = {
    "no_cpc": "Supervised baseline (UVMD + MixStyle)",
    "global_cpc": "Global CPC pretrain -> fine-tune",
    "per_band_cpc": "Per-band CPC pretrain -> fine-tune",
    "cross_band_cpc": "Cross-band CPC pretrain -> fine-tune",
}

EXPERIMENT_NAME = "h6s_per_band_cpc_loso"


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def grouped_to_arrays(
    grouped_windows: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grouped_windows dict to flat (windows, labels) arrays."""
    windows_list, labels_list = [], []
    for gid in sorted(grouped_windows.keys()):
        for rep_arr in grouped_windows[gid]:
            windows_list.append(rep_arr)
            labels_list.extend([gid] * len(rep_arr))
    return np.concatenate(windows_list, axis=0), np.array(labels_list)


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / (counts.astype(np.float64) + 1e-8)
    weights /= weights.sum()
    weights *= len(classes)
    w = torch.zeros(int(classes.max()) + 1, dtype=torch.float32, device=device)
    for c, wt in zip(classes, weights):
        w[int(c)] = wt
    return w


def cosine_schedule(epoch: int, total_epochs: int, base_lr: float, warmup: int = 5) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate_model(
    model: nn.Module,
    X: np.ndarray, y: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_list = []
        for s in range(0, len(Xt), batch_size):
            logits_list.append(model(Xt[s: s + batch_size]))
        preds = torch.cat(logits_list).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}


# ═════════════════════════════════════════════════════════════════════
#  Global CPC Wrapper
# ═════════════════════════════════════════════════════════════════════

class GlobalCPCWrapper(nn.Module):
    """
    Global CPC: apply CPC on the concatenated encoder output.

    UVMDSSLEncoder encodes (B, T, C) -> per-band features via K parallel CNNs.
    For global CPC, we need temporal features, so we use a separate CPCEncoder
    on the concatenated UVMD modes.

    Pipeline:
      x (B, T, C) -> UVMDBlock -> modes (B, K, T, C) ->
      reshape to (B, K*C, T) -> CPCEncoder -> z (B, d_enc, T') ->
      GRU context -> CPC prediction -> InfoNCE loss
    """

    def __init__(self, encoder: UVMDSSLEncoder, d_enc: int = 128,
                 d_ctx: int = 128, k_steps: int = 8,
                 temperature: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.K = encoder.K
        self.d_enc = d_enc
        self.k_steps = k_steps
        self.temperature = temperature

        # Temporal encoder on concatenated K*C channels
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(self.K * N_CHANNELS, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, d_enc, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, d_enc),
            nn.GELU(),
        )

        # Causal GRU context
        self.context_gru = nn.GRU(d_enc, d_ctx, batch_first=True)

        # Prediction heads
        self.predictors = nn.ModuleList([
            nn.Linear(d_ctx, d_enc) for _ in range(k_steps)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """(B, T, C) -> (loss, details)"""
        # UVMD decomposition
        modes = self.encoder.uvmd(x)  # (B, K, T, C)
        B, K, T, C = modes.shape

        # Concatenate bands: (B, K*C, T)
        modes_cat = modes.permute(0, 1, 3, 2).reshape(B, K * C, T)

        # Temporal encoding
        z = self.temporal_encoder(modes_cat)  # (B, d_enc, T')
        z = z.permute(0, 2, 1)  # (B, T', d_enc)

        # Context
        c, _ = self.context_gru(z)  # (B, T', d_ctx)
        T_prime = z.shape[1]

        # CPC loss
        total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        n_terms = 0

        for s in range(self.k_steps):
            t_max = T_prime - s - 1
            if t_max <= 0:
                continue
            pred = self.predictors[s](c[:, :t_max])  # (B, t_max, d_enc)
            target = z[:, s + 1: s + 1 + t_max]       # (B, t_max, d_enc)

            # InfoNCE
            B_t, T_t, D = pred.shape
            if B_t < 2:
                continue
            pred_flat = F.normalize(pred.reshape(-1, D), dim=-1)
            tgt_flat = F.normalize(target.reshape(-1, D), dim=-1)
            sim = torch.mm(pred_flat, tgt_flat.T) / self.temperature
            labels = torch.arange(B_t * T_t, device=sim.device)
            total_loss = total_loss + F.cross_entropy(sim, labels)
            n_terms += 1

        if n_terms > 0:
            total_loss = total_loss / n_terms

        # Overlap penalty
        overlap = self.encoder.spectral_overlap_penalty(sigma=OVERLAP_SIGMA)
        total_loss = total_loss + OVERLAP_LAMBDA * overlap

        details = {
            "cpc_loss": total_loss.item(),
            "overlap_penalty": overlap.item(),
            "n_terms": n_terms,
        }
        return total_loss, details


# ═════════════════════════════════════════════════════════════════════
#  SSL Pretraining
# ═════════════════════════════════════════════════════════════════════

def ssl_pretrain_per_band_cpc(
    X_train: np.ndarray,
    X_val: np.ndarray,
    cross_band: bool,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLEncoder:
    """FreqAwareCPC pretraining (per-band or cross-band)."""
    tag = "CrossBandCPC" if cross_band else "PerBandCPC"
    logger.info(f"  [{tag}] Starting SSL pretraining...")

    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=False,
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    ssl_model = FreqAwareCPC(
        encoder=encoder,
        d_enc=CPC_D_ENC, d_ctx=CPC_D_CTX,
        k_steps=CPC_K_STEPS,
        in_channels=N_CHANNELS,
        cross_band=cross_band,
        overlap_lambda=OVERLAP_LAMBDA, overlap_sigma=OVERLAP_SIGMA,
    ).to(device)

    X_all = np.concatenate([X_train, X_val], axis=0)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=SSL_BATCH, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        ssl_model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(SSL_EPOCHS):
        ssl_model.train()
        lr = cosine_schedule(epoch, SSL_EPOCHS, SSL_LR, SSL_WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            loss, details = ssl_model(xb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ssl_model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / max(n_batches, 1)
            logger.info(
                f"    {tag} epoch {epoch+1}/{SSL_EPOCHS}  "
                f"loss={avg:.4f}  lr={lr:.6f}"
            )

    logger.info(f"  [{tag}] Pretraining complete.")
    return encoder


def ssl_pretrain_global_cpc(
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLEncoder:
    """Global CPC pretraining on concatenated UVMD features."""
    logger.info("  [GlobalCPC] Starting SSL pretraining...")

    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=False,
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    ssl_model = GlobalCPCWrapper(
        encoder=encoder,
        d_enc=CPC_D_ENC, d_ctx=CPC_D_CTX,
        k_steps=CPC_K_STEPS,
    ).to(device)

    X_all = np.concatenate([X_train, X_val], axis=0)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=SSL_BATCH, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        ssl_model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(SSL_EPOCHS):
        ssl_model.train()
        lr = cosine_schedule(epoch, SSL_EPOCHS, SSL_LR, SSL_WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            loss, details = ssl_model(xb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ssl_model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / max(n_batches, 1)
            logger.info(
                f"    GlobalCPC epoch {epoch+1}/{SSL_EPOCHS}  "
                f"loss={avg:.4f}  lr={lr:.6f}"
            )

    logger.info("  [GlobalCPC] Pretraining complete.")
    return encoder


# ═════════════════════════════════════════════════════════════════════
#  Fine-Tuning (after SSL pretrain)
# ═════════════════════════════════════════════════════════════════════

def finetune(
    encoder: UVMDSSLEncoder,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLClassifier:
    """Two-phase fine-tuning: linear probe then full fine-tune."""
    model = UVMDSSLClassifier(
        encoder=encoder, num_classes=num_classes,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=FT_BATCH, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    # ── Phase 1: Linear probe (frozen encoder) ────────────────────
    model.freeze_encoder()
    probe_optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=PROBE_LR,
    )
    for epoch in range(PROBE_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            probe_optimizer.zero_grad()
            loss.backward()
            probe_optimizer.step()

    # ── Phase 2: Full fine-tune (unfrozen) ────────────────────────
    model.unfreeze_encoder()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=FT_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(FT_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            ce_loss = criterion(logits, yb)
            overlap_loss = model.spectral_overlap_penalty()
            loss = ce_loss + OVERLAP_LAMBDA * overlap_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), FT_BATCH):
                val_logits.append(model(Xv[vs: vs + FT_BATCH]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FT_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


# ═════════════════════════════════════════════════════════════════════
#  Supervised Training (baseline)
# ═════════════════════════════════════════════════════════════════════

def train_supervised_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> UVMDSSLClassifier:
    """Train UVMD + MixStyle supervised (H7 variant F)."""
    encoder = UVMDSSLEncoder(
        K=K_MODES, L=L_LAYERS, in_channels=N_CHANNELS, feat_dim=FEAT_DIM,
        use_mixstyle=True, mixstyle_p=MIXSTYLE_P, mixstyle_alpha=MIXSTYLE_ALPHA,
        alpha_init=ALPHA_INIT, tau_init=TAU_INIT,
    ).to(device)

    model = UVMDSSLClassifier(
        encoder=encoder, num_classes=num_classes,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)

    class_w = compute_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=SUP_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=FT_BATCH, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(SUP_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            ce_loss = criterion(logits, yb)
            overlap_loss = model.spectral_overlap_penalty()
            loss = ce_loss + OVERLAP_LAMBDA * overlap_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = []
            for vs in range(0, len(Xv), FT_BATCH):
                val_logits.append(model(Xv[vs: vs + FT_BATCH]))
            val_logits = torch.cat(val_logits)
            val_loss = F.cross_entropy(val_logits, yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= SUP_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


# ═════════════════════════════════════════════════════════════════════
#  Single LOSO Fold
# ═════════════════════════════════════════════════════════════════════

def train_one_fold(
    variant: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    seed_everything()

    if variant == "no_cpc":
        model = train_supervised_baseline(
            X_train, y_train, X_val, y_val, num_classes, device, logger,
        )
    elif variant == "global_cpc":
        encoder = ssl_pretrain_global_cpc(X_train, X_val, device, logger)
        model = finetune(
            encoder, X_train, y_train, X_val, y_val,
            num_classes, device, logger,
        )
    elif variant == "per_band_cpc":
        encoder = ssl_pretrain_per_band_cpc(
            X_train, X_val, cross_band=False, device=device, logger=logger,
        )
        model = finetune(
            encoder, X_train, y_train, X_val, y_val,
            num_classes, device, logger,
        )
    elif variant == "cross_band_cpc":
        encoder = ssl_pretrain_per_band_cpc(
            X_train, X_val, cross_band=True, device=device, logger=logger,
        )
        model = finetune(
            encoder, X_train, y_train, X_val, y_val,
            num_classes, device, logger,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    test_metrics = evaluate_model(model, X_test, y_test, device)
    learned = model.get_learned_uvmd_params()

    return {
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "final_omega_k": learned["omega_k"],
    }


# ═════════════════════════════════════════════════════════════════════
#  LOSO Loop
# ═════════════════════════════════════════════════════════════════════

def run_variant(
    variant: str,
    subjects: List[str],
    base_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"  Variant: {VARIANT_LABELS[variant]}")
    logger.info(f"{'='*60}")

    proc_cfg = ProcessingConfig(
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP,
        sampling_rate=SAMPLING_RATE,
    )
    multi_loader = MultiSubjectLoader(proc_cfg, logger, use_gpu=False,
                                       use_improved_processing=True)
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=subjects,
        exercises=["E1"],
        include_rest=False,
    )

    common_gestures = multi_loader.get_common_gestures(subjects_data, max_gestures=10)
    gesture_to_class = {g: i for i, g in enumerate(sorted(common_gestures))}
    num_classes = len(gesture_to_class)

    subj_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, (_, _, gw) in subjects_data.items():
        wins, labs = grouped_to_arrays(gw)
        mask = np.isin(labs, list(gesture_to_class.keys()))
        wins, labs = wins[mask], labs[mask]
        labs = np.array([gesture_to_class[g] for g in labs])
        subj_arrays[sid] = (wins, labs)

    fold_results = []
    for test_sid in subjects:
        t0 = time.time()
        logger.info(f"  Fold: test={test_sid}")

        X_test, y_test = subj_arrays[test_sid]
        if len(X_test) == 0:
            logger.warning(f"    Skipping {test_sid}: no test data")
            continue

        Xs, ys = [], []
        for sid in subjects:
            if sid == test_sid:
                continue
            w, l = subj_arrays[sid]
            if len(w) > 0:
                Xs.append(w)
                ys.append(l)
        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)

        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(X_all))
        n_val = max(1, int(len(X_all) * VAL_RATIO))
        X_train, y_train = X_all[perm[n_val:]], y_all[perm[n_val:]]
        X_val, y_val = X_all[perm[:n_val]], y_all[perm[:n_val]]

        mean_c = X_train.mean(axis=(0, 1), keepdims=True)
        std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train = (X_train - mean_c) / std_c
        X_val = (X_val - mean_c) / std_c
        X_test_norm = (X_test - mean_c) / std_c

        metrics = train_one_fold(
            variant=variant,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test_norm, y_test=y_test,
            num_classes=num_classes,
            device=device, logger=logger,
        )

        elapsed = time.time() - t0
        metrics["test_subject"] = test_sid
        metrics["time_s"] = round(elapsed, 1)
        fold_results.append(metrics)

        logger.info(
            f"    Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}  ({elapsed:.0f}s)"
        )

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]

    summary = {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "n_subjects": len(fold_results),
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "std_f1_macro": float(np.std(f1s)) if f1s else 0.0,
        "per_subject": fold_results,
    }

    logger.info(
        f"\n  >>> {VARIANT_LABELS[variant]}: "
        f"Acc={summary['mean_accuracy']*100:.2f}+-{summary['std_accuracy']*100:.2f}  "
        f"F1={summary['mean_f1_macro']*100:.2f}+-{summary['std_f1_macro']*100:.2f}"
    )
    return summary


# ═════════════════════════════════════════════════════════════════════
#  Statistical Tests
# ═════════════════════════════════════════════════════════════════════

def statistical_comparison(
    results: Dict[str, Dict],
    logger: logging.Logger,
) -> Dict:
    """Wilcoxon signed-rank tests with Holm-Bonferroni correction (Family A, k=2)."""
    from scipy import stats

    comparisons = [
        ("per_band_cpc", "no_cpc"),
        ("per_band_cpc", "global_cpc"),
    ]

    stats_results = {}
    p_values = []

    for v1, v2 in comparisons:
        if v1 not in results or v2 not in results:
            continue

        f1_v1 = {r["test_subject"]: r["f1_macro"] for r in results[v1]["per_subject"]}
        f1_v2 = {r["test_subject"]: r["f1_macro"] for r in results[v2]["per_subject"]}
        common = sorted(set(f1_v1.keys()) & set(f1_v2.keys()))

        if len(common) < 5:
            continue

        x = np.array([f1_v1[s] for s in common])
        y = np.array([f1_v2[s] for s in common])
        diff = x - y

        stat, p = stats.wilcoxon(diff, alternative="two-sided")
        d = diff.mean() / (diff.std() + 1e-8)

        key = f"{v1}_vs_{v2}"
        stats_results[key] = {
            "statistic": float(stat),
            "p_uncorrected": float(p),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std()),
            "cohens_d": float(d),
            "n_subjects": len(common),
            "n_v1_wins": int((diff > 0).sum()),
        }
        p_values.append((key, p))

    # Holm-Bonferroni correction
    if p_values:
        p_values.sort(key=lambda x: x[1])
        k = len(p_values)
        for rank, (key, p_raw) in enumerate(p_values):
            p_corrected = min(1.0, p_raw * (k - rank))
            stats_results[key]["p_holm"] = float(p_corrected)
            sig = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*" if p_corrected < 0.05 else "ns"
            stats_results[key]["significance"] = sig

            logger.info(
                f"  {key}: Delta={stats_results[key]['mean_diff']*100:+.2f} pp  "
                f"d={stats_results[key]['cohens_d']:.2f}  "
                f"p_Holm={p_corrected:.4f} ({sig})"
            )

    return stats_results


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="H6s: Per-Band CPC")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--ci", type=int, default=0)
    parser.add_argument("--subjects", type=str, default="")
    parser.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                        choices=VARIANT_NAMES)
    parser.add_argument("--batch_size", type=int, default=FT_BATCH)
    parser.add_argument("--base_dir", type=str, default="data")
    args, _ = parser.parse_known_args()

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    elif args.full:
        subjects = _FULL_SUBJECTS
    elif args.ci:
        subjects = _CI_SUBJECTS
    else:
        subjects = _CI_SUBJECTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments_output") / f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"H6s: Per-Band CPC")
    logger.info(f"Subjects ({len(subjects)}): {subjects}")
    logger.info(f"Variants: {args.variants}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    all_results = {}
    for variant in args.variants:
        summary = run_variant(
            variant=variant,
            subjects=subjects,
            base_dir=args.base_dir,
            device=device,
            logger=logger,
        )
        all_results[variant] = summary

        variant_dir = output_dir / variant
        variant_dir.mkdir(exist_ok=True)
        with open(variant_dir / "loso_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Statistical comparison
    logger.info(f"\n{'='*60}")
    logger.info("  Statistical Comparison (Family A, Holm-Bonferroni, k=2)")
    logger.info(f"{'='*60}")
    stats_results = statistical_comparison(all_results, logger)

    # Save combined results
    combined = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "variants": {v: all_results[v] for v in args.variants if v in all_results},
        "statistical_tests": stats_results,
    }
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"{'Variant':<45} {'Acc':>10} {'F1':>10}")
    logger.info("-" * 68)
    for v in args.variants:
        if v in all_results:
            r = all_results[v]
            acc_str = f"{r['mean_accuracy']*100:.2f}+-{r['std_accuracy']*100:.2f}"
            f1_str = f"{r['mean_f1_macro']*100:.2f}+-{r['std_f1_macro']*100:.2f}"
            logger.info(f"{VARIANT_LABELS[v]:<45} {acc_str:>10} {f1_str:>10}")

    logger.info(f"\nResults saved to {output_dir}")

    # Hypothesis tracking
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        best_variant = max(
            [v for v in args.variants if v in all_results],
            key=lambda v: all_results[v]["mean_f1_macro"],
        )
        baseline_f1 = all_results.get("no_cpc", {}).get("mean_f1_macro", 0)
        if best_variant in ("per_band_cpc", "cross_band_cpc") and all_results[best_variant]["mean_f1_macro"] > baseline_f1:
            mark_hypothesis_verified("H6s", {
                "best_variant": best_variant,
                "mean_f1_macro": all_results[best_variant]["mean_f1_macro"],
                "delta_vs_no_cpc": all_results[best_variant]["mean_f1_macro"] - baseline_f1,
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H6s", f"Per-band CPC did not improve over supervised baseline (best: {best_variant})")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
