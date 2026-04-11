"""
Trainer for PhaseAlignCNNGRU (Experiment 66).

Extends WindowClassifierTrainer with a custom fit() that:

    1. Converts splits (Dict[str, Dict[int, np.ndarray]]) → (N, T, C) arrays
       using the parent's _prepare_splits_arrays() helper.
    2. Transposes (N, T, C) → (N, C, T) for PyTorch Conv1d convention.
    3. Applies TKEO-based phase alignment to every window independently.
       This step uses ONLY per-window statistics — zero data leakage.
    4. Computes per-channel mean/std EXCLUSIVELY from the aligned training data.
    5. Standardizes all splits with training-data statistics.
    6. Trains PhaseAlignCNNGRU (CNN-BiGRU-Attention, no FIR frontend).
    7. Saves checkpoint and training history.

evaluate_numpy() applies the same pipeline (transpose → phase align →
standardize) and runs frozen inference.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSO integrity checklist
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase alignment (TKEO onset detection):
    ✓  Uses ONLY the SINGLE window being processed.
    ✓  Threshold = percentile_10(window_energy)
                 + alpha * (max(window_energy) - percentile_10(window_energy))
       Both P_LOW=10 and alpha=0.20 are FIXED CONSTANTS, not data-estimated.
    ✓  smooth_len=30 (energy envelope kernel) is a FIXED CONSTANT.
    ✓  Same algorithm applied identically to train, val, and test windows.
    ✓  No global threshold, no subject statistics, no fold information.

Channel standardization:
    ✓  mean_c, std_c computed from aligned X_train ONLY (train-subject pool).
    ✓  X_val and X_test standardized with TRAIN statistics.
    ✓  No test-set statistics used anywhere.

Model training:
    ✓  Model weights optimized only on train split.
    ✓  Val split used for early stopping (loss only, no param estimation).
    ✓  Test split is fully isolated until final evaluation.
    ✓  model.eval() at inference disables BN running-stat updates.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from training.trainer import (
    WindowClassifierTrainer, WindowDataset, get_worker_init_fn, seed_everything,
)
from models.phase_align_cnn_gru import PhaseAlignCNNGRU


class PhaseAlignTrainer(WindowClassifierTrainer):
    """
    Trainer for temporal phase alignment + CNN-BiGRU-Attention.

    The phase alignment step detects the active (high-energy) portion of each
    EMG window using the Teager-Kaiser Energy Operator (TKEO) and resamples it
    to fill the full window length.  This canonicalizes gesture timing across
    subjects without any training-data–derived parameters.

    The CNN-BiGRU-Attention encoder then learns gesture-discriminative features
    from the uniformly-timed representation.

    Args:
        train_cfg:              TrainingConfig dataclass.
        logger:                 Python logger.
        output_dir:             Directory for checkpoints and logs.
        visualizer:             Optional Visualizer for plots.
        cnn_channels:           CNN block output channels (3 blocks).
        gru_hidden:             BiGRU hidden units per direction.
        num_heads:              Multi-head attention heads.
        smooth_len:             TKEO energy envelope smoothing kernel length
                                (samples).  Default 30 = 15 ms at 2000 Hz.
        energy_percentile_low:  Percentile used as the energy baseline.
                                Default 10 (10th percentile).
        energy_alpha:           Threshold factor: threshold = baseline
                                + alpha * (peak - baseline).  Default 0.20.
        min_active_ratio:       Minimum fraction of T that the detected active
                                region must span; windows with shorter detections
                                are left unaligned.  Default 0.05 (3% = 18 smp).
    """

    def __init__(
        self,
        train_cfg,
        logger:                logging.Logger,
        output_dir:            Path,
        visualizer             = None,
        cnn_channels:          Tuple[int, ...]  = (64, 128, 256),
        gru_hidden:            int              = 128,
        num_heads:             int              = 4,
        smooth_len:            int              = 30,
        energy_percentile_low: int              = 10,
        energy_alpha:          float            = 0.20,
        min_active_ratio:      float            = 0.05,
    ) -> None:
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.cnn_channels          = cnn_channels
        self.gru_hidden            = gru_hidden
        self.num_heads             = num_heads
        self.smooth_len            = smooth_len
        self.energy_percentile_low = energy_percentile_low
        self.energy_alpha          = energy_alpha
        self.min_active_ratio      = min_active_ratio

    # ──────────────────────────────────────────────────────────────────────────
    #  TKEO phase alignment helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _tkeo_energy_per_sample(self, window_ct: np.ndarray) -> np.ndarray:
        """
        Compute mean-over-channels Teager-Kaiser energy per time sample.

        TKEO for a discrete signal x:
            ψ[n] = x[n]² − x[n−1] · x[n+1]

        Borders are replicated from the nearest interior sample to avoid
        boundary artifacts.  The result is the mean absolute TKEO across
        all C channels, giving a single (T,) energy envelope that reflects
        global muscle activation.

        Args:
            window_ct: (C, T) float32 EMG window, channel-first.

        Returns:
            energy: (T,) float64 non-negative energy envelope.
        """
        C, T = window_ct.shape
        energy = np.zeros(T, dtype=np.float64)

        for c in range(C):
            x = window_ct[c].astype(np.float64)
            tkeo = np.empty(T, dtype=np.float64)
            # Interior samples
            tkeo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
            # Border replication (avoids zeros at edges)
            tkeo[0]    = tkeo[1]
            tkeo[-1]   = tkeo[-2]
            energy += np.abs(tkeo)

        energy /= max(C, 1)
        return energy

    def _detect_active_bounds(
        self, energy: np.ndarray
    ) -> Tuple[int, int]:
        """
        Detect onset and offset of the active (high-energy) region.

        Algorithm (all parameters are FIXED CONSTANTS — no data estimation):
            1. Smooth energy with a rectangular kernel of length smooth_len.
            2. baseline = percentile(smoothed_energy, energy_percentile_low)
            3. peak     = max(smoothed_energy)
            4. threshold = baseline + energy_alpha * (peak − baseline)
            5. onset  = first sample ≥ threshold
               offset = last  sample ≥ threshold

        Returns:
            (onset, offset) inclusive sample indices.
            Falls back to (0, T−1) if no meaningful activity is found.
        """
        T = len(energy)

        # Smooth energy envelope
        kernel        = np.ones(self.smooth_len, dtype=np.float64) / self.smooth_len
        energy_smooth = np.convolve(energy, kernel, mode="same")

        baseline  = np.percentile(energy_smooth, self.energy_percentile_low)
        peak      = energy_smooth.max()

        # Flat signal — cannot detect onset/offset
        if peak <= baseline + 1e-12:
            return 0, T - 1

        threshold = baseline + self.energy_alpha * (peak - baseline)
        above     = energy_smooth >= threshold

        if not above.any():
            return 0, T - 1

        onset  = int(np.argmax(above))
        offset = int(T - 1 - np.argmax(above[::-1]))

        return onset, offset

    def _phase_align_single(self, window_ct: np.ndarray) -> np.ndarray:
        """
        Phase-align a single (C, T) EMG window → (C, T).

        Steps:
            1. Compute TKEO energy envelope (T,) from the window itself.
            2. Detect onset and offset of the active region.
            3. If the active region is too short (< min_active_ratio * T),
               return the window unmodified (alignment would be degenerate).
            4. Resample each channel's active segment to T using linear
               interpolation (scipy-free, numerically stable).

        The resampled output has the same shape (C, T) as the input.  The
        entire time axis is filled with the resampled active phase — there is
        no zero-padding; every output sample carries information.

        LOSO guarantee:
            All decisions (threshold, onset, offset) derive from this single
            window only.  No global, subject-specific, or fold-specific state
            is used.

        Args:
            window_ct: (C, T) float32 channel-first window.

        Returns:
            (C, T) float32 phase-aligned window.
        """
        C, T = window_ct.shape

        energy        = self._tkeo_energy_per_sample(window_ct)
        onset, offset = self._detect_active_bounds(energy)

        active_len = offset - onset + 1   # inclusive
        min_active = max(2, int(self.min_active_ratio * T))

        # Degenerate detection: return unmodified window
        if active_len < min_active:
            return window_ct.copy()

        # Already fills the whole window — nothing to do
        if active_len == T:
            return window_ct.copy()

        # Linear-interpolation resampling of the active segment to length T.
        # t_src in [0, 1] with active_len points;
        # t_dst in [0, 1] with T points.
        t_src  = np.linspace(0.0, 1.0, active_len,  dtype=np.float64)
        t_dst  = np.linspace(0.0, 1.0, T,            dtype=np.float64)

        out = np.empty((C, T), dtype=np.float32)
        for c in range(C):
            segment = window_ct[c, onset : offset + 1].astype(np.float64)
            out[c]  = np.interp(t_dst, t_src, segment).astype(np.float32)

        return out

    def _phase_align_batch(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply phase alignment to a batch of windows.

        Args:
            X: (N, C, T) float32, channel-first.

        Returns:
            X_aligned: (N, C, T) float32 — each window independently aligned.
            stats: diagnostic dict with alignment statistics.
        """
        N, C, T = X.shape
        X_aligned = np.empty_like(X)

        n_aligned     = 0
        active_ratios = []

        for i in range(N):
            window_ct = X[i]   # (C, T)

            energy          = self._tkeo_energy_per_sample(window_ct)
            onset, offset   = self._detect_active_bounds(energy)
            active_len      = offset - onset + 1
            min_active      = max(2, int(self.min_active_ratio * T))

            if active_len >= min_active and active_len < T:
                X_aligned[i] = self._phase_align_single(window_ct)
                n_aligned   += 1
                active_ratios.append(active_len / T)
            else:
                X_aligned[i] = window_ct.copy()
                active_ratios.append(active_len / T)

        stats = {
            "n_total":         N,
            "n_aligned":       n_aligned,
            "pct_aligned":     100.0 * n_aligned / max(1, N),
            "mean_active_ratio": float(np.mean(active_ratios)) if active_ratios else 0.0,
            "std_active_ratio":  float(np.std(active_ratios))  if active_ratios else 0.0,
        }
        return X_aligned, stats

    # ──────────────────────────────────────────────────────────────────────────
    #  fit
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train PhaseAlignCNNGRU on the provided LOSO splits.

        Preprocessing pipeline (strictly LOSO-safe):
            1. _prepare_splits_arrays()  → (N, T, C)
            2. transpose                 → (N, C, T)
            3. phase alignment           → (N, C, T)  [per-window, no leakage]
            4. channel standardization   → train stats only

        Args:
            splits: {
                "train": Dict[int, np.ndarray],   # gesture_id → (N, T, C)
                "val":   Dict[int, np.ndarray],
                "test":  Dict[int, np.ndarray],
            }

        Returns:
            dict with in-fold val / test metrics and alignment stats.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Splits → flat (N, T, C) numpy arrays ──────────────────────
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T) ───────────────────────────
        # T (600) >> C (8); detect by comparing axis sizes.
        if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
            X_train = X_train.transpose(0, 2, 1)
            if X_val.ndim  == 3 and len(X_val)  > 0:
                X_val  = X_val.transpose(0, 2, 1)
            if X_test.ndim == 3 and len(X_test) > 0:
                X_test = X_test.transpose(0, 2, 1)
            self.logger.info(
                f"Transposed to (N, C, T): X_train={X_train.shape}"
            )

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 3. TKEO phase alignment — PER-WINDOW, ZERO LEAKAGE ────────────
        # Alignment parameters are fixed constants; no global statistics.
        # The same algorithm is applied to every window regardless of split.
        self.logger.info(
            f"Phase alignment: smooth_len={self.smooth_len}, "
            f"energy_alpha={self.energy_alpha}, "
            f"energy_percentile_low={self.energy_percentile_low}, "
            f"min_active_ratio={self.min_active_ratio}"
        )

        X_train, align_stats_train = self._phase_align_batch(X_train)
        self.logger.info(
            f"[Train] {align_stats_train['n_aligned']}/{align_stats_train['n_total']} "
            f"windows aligned ({align_stats_train['pct_aligned']:.1f}%); "
            f"mean active ratio = {align_stats_train['mean_active_ratio']:.3f}"
        )

        if len(X_val) > 0:
            X_val, align_stats_val = self._phase_align_batch(X_val)
            self.logger.info(
                f"[Val]   {align_stats_val['n_aligned']}/{align_stats_val['n_total']} "
                f"windows aligned ({align_stats_val['pct_aligned']:.1f}%)"
            )
        else:
            align_stats_val = {}

        if len(X_test) > 0:
            X_test, align_stats_test = self._phase_align_batch(X_test)
            self.logger.info(
                f"[Test]  {align_stats_test['n_aligned']}/{align_stats_test['n_total']} "
                f"windows aligned ({align_stats_test['pct_aligned']:.1f}%)"
            )
        else:
            align_stats_test = {}

        # Save alignment stats for diagnostic purposes
        align_report = {
            "train": align_stats_train,
            "val":   align_stats_val,
            "test":  align_stats_test,
        }
        with open(self.output_dir / "phase_alignment_stats.json", "w") as fh:
            json.dump(align_report, fh, indent=4)

        # ── 4. Per-channel standardization — TRAIN STATS ONLY ────────────
        # LOSO contract: mean_c and std_c are computed from the phase-aligned
        # training data only.  Val and test windows are standardized with
        # these same training statistics.
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val  = self._apply_standardization(X_val,  mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean      = mean_c,
            std       = std_c,
            class_ids = np.array(class_ids, dtype=np.int32),
        )
        self.logger.info(
            "Per-channel standardization applied (training statistics only). "
            f"in_ch={in_channels}, T={window_size}, classes={num_classes}"
        )

        # ── 5. Build model ────────────────────────────────────────────────
        model = PhaseAlignCNNGRU(
            in_channels  = in_channels,
            num_classes  = num_classes,
            cnn_channels = self.cnn_channels,
            gru_hidden   = self.gru_hidden,
            num_heads    = self.num_heads,
            dropout      = self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"PhaseAlignCNNGRU built: in_ch={in_channels}, T={window_size}, "
            f"classes={num_classes}, total_params={total_params:,}"
        )

        # ── 6. DataLoaders ────────────────────────────────────────────────
        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val,   y_val)   if len(X_val)  > 0 else None
        ds_test  = WindowDataset(X_test,  y_test)  if len(X_test) > 0 else None

        worker_init = get_worker_init_fn(self.cfg.seed)
        g           = torch.Generator().manual_seed(self.cfg.seed)

        dl_train = DataLoader(
            ds_train,
            batch_size     = self.cfg.batch_size,
            shuffle        = True,
            num_workers    = self.cfg.num_workers,
            pin_memory     = True,
            worker_init_fn = worker_init if self.cfg.num_workers > 0 else None,
            generator      = g,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size  = self.cfg.batch_size,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size  = self.cfg.batch_size,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
        ) if ds_test else None

        # ── 7. Loss function (class-weighted CrossEntropy) ─────────────────
        if self.cfg.use_class_weights:
            counts    = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw        = counts.sum() / (counts + 1e-8)
            cw       /= cw.mean()
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(self.cfg.device)
            )
            self.logger.info(f"Class weights applied: {cw.round(3).tolist()}")
        else:
            criterion = nn.CrossEntropyLoss()

        # ── 8. Optimizer + LR scheduler ───────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr           = self.cfg.learning_rate,
            weight_decay = self.cfg.weight_decay,
        )
        # verbose removed in PyTorch ≥ 2.4
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 9. Training loop ──────────────────────────────────────────────
        history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }
        best_val_loss         = float("inf")
        best_state: Optional[Dict] = None
        no_improve            = 0
        device                = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            ep_loss, ep_correct, ep_total = 0.0, 0, 0

            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()

                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs          = xb.size(0)
                ep_loss    += loss.item() * bs
                ep_correct += (logits.argmax(1) == yb).sum().item()
                ep_total   += bs

            train_loss = ep_loss    / max(1, ep_total)
            train_acc  = ep_correct / max(1, ep_total)

            # ── validation ────────────────────────────────────────────────
            if dl_val is not None:
                model.eval()
                vl_sum, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits  = model(xb)
                        vl_sum += criterion(logits, yb).item() * yb.size(0)
                        vc     += (logits.argmax(1) == yb).sum().item()
                        vt     += yb.size(0)
                val_loss = vl_sum / max(1, vt)
                val_acc  = vc    / max(1, vt)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

            # ── early stopping ─────────────────────────────────────────────
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state    = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}.")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        # ── 10. Store trainer state (needed by evaluate_numpy) ────────────
        self.model       = model
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # ── 11. Persist training history ──────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(
                history, filename="training_curves.png"
            )

        # ── 12. In-fold evaluation on val / internal test split ───────────
        results: Dict = {
            "class_ids":    class_ids,
            "class_names":  class_names,
            "alignment":    align_report,
        }

        def _eval_loader(dl, split_name: str):
            if dl is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(device)
                    all_logits.append(model(xb).cpu().numpy())
                    all_y.append(yb.numpy())
            logits_np = np.concatenate(all_logits, axis=0)
            y_true    = np.concatenate(all_y,      axis=0)
            y_pred    = logits_np.argmax(axis=1)
            acc       = accuracy_score(y_true, y_pred)
            f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
            rep       = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            cm        = confusion_matrix(
                y_true, y_pred, labels=np.arange(num_classes)
            )
            if self.visualizer is not None:
                labels = [class_names.get(gid, str(gid)) for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, labels, normalize=True,
                    filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report":   rep,
                "confusion_matrix": cm.tolist(),
            }

        results["val"]  = _eval_loader(dl_val,  "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── 13. Save checkpoint ───────────────────────────────────────────
        torch.save(
            {
                "state_dict":  model.state_dict(),
                "in_channels": in_channels,
                "num_classes": num_classes,
                "class_ids":   class_ids,
                "mean":        mean_c,
                "std":         std_c,
                "window_size": window_size,
                "cnn_channels": list(self.cnn_channels),
                "gru_hidden":   self.gru_hidden,
                "num_heads":    self.num_heads,
                # Phase alignment hyper-parameters (constants, not learned)
                "smooth_len":            self.smooth_len,
                "energy_percentile_low": self.energy_percentile_low,
                "energy_alpha":          self.energy_alpha,
                "min_active_ratio":      self.min_active_ratio,
                "training_config":       asdict(self.cfg),
            },
            self.output_dir / "phase_align_cnn_gru.pt",
        )
        self.logger.info(
            f"Checkpoint saved: {self.output_dir / 'phase_align_cnn_gru.pt'}"
        )

        with open(self.output_dir / "classification_results.json", "w") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    #  evaluate_numpy — called by the experiment for cross-subject test
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        split_name: str  = "custom",
        visualize:  bool = False,
    ) -> Dict:
        """
        Evaluate the trained model on arbitrary (X, y) numpy arrays.

        Applies the SAME preprocessing pipeline as fit():
            1. Transpose (N, T, C) → (N, C, T) if needed.
            2. Phase alignment — per-window TKEO, no external state.
            3. Per-channel standardization with training-data statistics.

        Args:
            X:          (N, T, C) or (N, C, T) raw EMG windows.
            y:          (N,) integer class labels (matching class_ids from fit()).
            split_name: prefix for saved confusion-matrix plot.
            visualize:  if True, save confusion-matrix image.

        Returns:
            dict with "accuracy", "f1_macro", "report", "confusion_matrix",
            "logits", "alignment_stats".
        """
        assert self.model     is not None,  "Call fit() before evaluate_numpy()."
        assert self.mean_c    is not None and self.std_c     is not None
        assert self.class_ids is not None and self.class_names is not None

        X_in = X.copy().astype(np.float32)

        # Transpose (N, T, C) → (N, C, T) using same heuristic as fit()
        if X_in.ndim == 3 and X_in.shape[1] > X_in.shape[2]:
            X_in = X_in.transpose(0, 2, 1)

        # Phase alignment — deterministic per-window, no train-set info
        X_in, align_stats = self._phase_align_batch(X_in)
        self.logger.info(
            f"[{split_name}] Phase alignment: "
            f"{align_stats['n_aligned']}/{align_stats['n_total']} windows "
            f"aligned ({align_stats['pct_aligned']:.1f}%)"
        )

        # Standardize with training-data statistics (no test stats)
        X_in = self._apply_standardization(X_in, self.mean_c, self.std_c)

        ds = WindowDataset(X_in, y)
        dl = DataLoader(
            ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
        )

        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                all_logits.append(self.model(xb).cpu().numpy())
                all_y.append(yb.numpy())

        logits  = np.concatenate(all_logits, axis=0)
        y_true  = np.concatenate(all_y,      axis=0)
        y_pred  = logits.argmax(axis=1)

        acc     = accuracy_score(y_true, y_pred)
        f1_mac  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rep     = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        cm      = confusion_matrix(
            y_true, y_pred, labels=np.arange(len(self.class_ids))
        )

        if visualize and self.visualizer is not None:
            labels = [
                self.class_names.get(gid, str(gid)) for gid in self.class_ids
            ]
            self.visualizer.plot_confusion_matrix(
                cm, labels, normalize=True,
                filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy":         float(acc),
            "f1_macro":         float(f1_mac),
            "report":           rep,
            "confusion_matrix": cm.tolist(),
            "logits":           logits,
            "alignment_stats":  align_stats,
        }
