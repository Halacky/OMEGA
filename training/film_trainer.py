"""
Custom trainer for FiLM Subject-Adaptive model.

Handles:
1. Pseudo-subject clustering via K-means on signal features
2. Custom dataset with reference window sampling per cluster
3. Training loop with FiLM conditioning + auxiliary cluster loss
4. Evaluation with calibration windows for style embedding computation
"""

import json
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

from config.base import TrainingConfig
from training.trainer import WindowClassifierTrainer
from training.datasets import WindowDataset
from utils.logging import seed_everything, get_worker_init_fn


class FiLMWindowDataset(Dataset):
    """
    Dataset that returns (window, label, cluster_id, ref_windows).

    For each sample, K reference windows are randomly sampled from the
    same pseudo-subject cluster, providing the style encoder with
    examples of the "same style" during training.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cluster_ids: np.ndarray,
        num_ref_windows: int = 5,
    ):
        """
        Args:
            X: (N, C, T) standardized windows
            y: (N,) class labels
            cluster_ids: (N,) pseudo-subject cluster assignments
            num_ref_windows: K — number of reference windows to sample per item
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.cluster_ids = torch.from_numpy(cluster_ids).long()
        self.num_ref_windows = num_ref_windows

        # Pre-compute per-cluster indices for fast sampling
        self.cluster_indices = {}
        unique_clusters = np.unique(cluster_ids)
        for cid in unique_clusters:
            self.cluster_indices[int(cid)] = np.where(cluster_ids == cid)[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        window = self.X[idx]
        label = self.y[idx]
        cid = int(self.cluster_ids[idx].item())

        # Sample K reference windows from the same cluster
        pool = self.cluster_indices[cid]
        K = self.num_ref_windows
        if len(pool) <= K:
            # Not enough windows — repeat with replacement
            ref_idx = np.random.choice(pool, size=K, replace=True)
        else:
            ref_idx = np.random.choice(pool, size=K, replace=False)

        ref_windows = self.X[ref_idx]  # (K, C, T)
        return window, label, self.cluster_ids[idx], ref_windows


def extract_signal_features_numpy(X: np.ndarray) -> np.ndarray:
    """
    Extract simple signal statistics from EMG windows for clustering.

    Args:
        X: (N, C, T) EMG windows

    Returns:
        (N, 2*C + 1) feature matrix — per-channel RMS, per-channel
        spectral centroid, global log-SNR.
    """
    N, C, T = X.shape

    # Per-channel RMS
    rms = np.sqrt(np.mean(X ** 2, axis=2) + 1e-8)  # (N, C)

    # Per-channel spectral centroid via FFT
    fft_mag = np.abs(np.fft.rfft(X, axis=2))  # (N, C, T//2+1)
    n_freq = fft_mag.shape[2]
    freq_bins = np.linspace(0, 1, n_freq)
    spectral_centroid = (
        np.sum(fft_mag * freq_bins[None, None, :], axis=2)
        / (np.sum(fft_mag, axis=2) + 1e-8)
    )  # (N, C)

    # Global log-SNR
    signal_power = np.mean(X ** 2, axis=(1, 2))  # (N,)
    hf_start = n_freq * 3 // 4
    noise_power = np.mean(fft_mag[:, :, hf_start:] ** 2, axis=(1, 2))  # (N,)
    snr = np.log1p(signal_power / (noise_power + 1e-8))  # (N,)

    features = np.concatenate([rms, spectral_centroid, snr[:, None]], axis=1)
    return features.astype(np.float32)


class FiLMSubjectAdaptiveTrainer(WindowClassifierTrainer):
    """
    Trainer for FiLM Subject-Adaptive CNN-GRU-Attention model.

    Extends WindowClassifierTrainer with:
    - Pseudo-subject clustering of training data
    - Custom training loop with reference window sampling
    - Evaluation with calibration-based style embedding
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        num_ref_windows: int = 5,
        num_pseudo_clusters: int = 10,
        aux_loss_weight: float = 0.1,
        style_dim: int = 64,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.num_ref_windows = num_ref_windows
        self.num_pseudo_clusters = num_pseudo_clusters
        self.aux_loss_weight = aux_loss_weight
        self.style_dim = style_dim

    def _cluster_training_data(self, X_train: np.ndarray) -> np.ndarray:
        """
        Cluster training windows into pseudo-subject groups using K-means
        on signal features (RMS, spectral centroid, SNR).

        Args:
            X_train: (N, C, T) training windows (already standardized)

        Returns:
            cluster_ids: (N,) integer cluster assignments
        """
        self.logger.info(
            f"Extracting signal features for pseudo-subject clustering "
            f"(N={X_train.shape[0]})..."
        )
        features = extract_signal_features_numpy(X_train)

        # Normalize features for K-means
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0) + 1e-8
        features_norm = (features - feat_mean) / feat_std

        n_clusters = min(self.num_pseudo_clusters, X_train.shape[0] // 10)
        n_clusters = max(n_clusters, 2)
        self.logger.info(f"Running K-means with {n_clusters} pseudo-subject clusters...")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.cfg.seed,
            batch_size=min(1024, X_train.shape[0]),
            n_init=3,
        )
        cluster_ids = kmeans.fit_predict(features_norm)

        # Log cluster distribution
        unique, counts = np.unique(cluster_ids, return_counts=True)
        for cid, cnt in zip(unique, counts):
            self.logger.info(f"  Cluster {cid}: {cnt} windows ({100*cnt/len(cluster_ids):.1f}%)")

        return cluster_ids.astype(np.int64)

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """Train FiLM Subject-Adaptive model with pseudo-subject clustering."""
        seed_everything(self.cfg.seed)

        # Prepare arrays from splits
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # Transpose (N, T, C) → (N, C, T) for CNN models
        if X_train.ndim == 3:
            N, dim1, dim2 = X_train.shape
            if dim1 > dim2:
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        # Channel standardization
        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardization.")

        # Save normalization stats
        norm_path = self.output_dir / "normalization_stats.npz"
        np.savez_compressed(
            norm_path, mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # Pseudo-subject clustering
        cluster_ids = self._cluster_training_data(X_train)
        actual_n_clusters = len(np.unique(cluster_ids))

        # Create model
        from models.film_subject_adaptive import FiLMSubjectAdaptiveCNNGRU
        model = FiLMSubjectAdaptiveCNNGRU(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=self.cfg.dropout,
            style_dim=self.style_dim,
            num_pseudo_clusters=actual_n_clusters,
        ).to(self.cfg.device)
        self.logger.info(
            f"Created FiLMSubjectAdaptiveCNNGRU: "
            f"in_channels={in_channels}, num_classes={num_classes}, "
            f"style_dim={self.style_dim}, clusters={actual_n_clusters}"
        )
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")

        # Datasets
        ds_train = FiLMWindowDataset(
            X_train, y_train, cluster_ids,
            num_ref_windows=self.num_ref_windows,
        )
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)

        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        ) if ds_test else None

        # Optimizer and loss
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            class_weights = class_counts.sum() / (class_counts + 1e-8)
            class_weights = class_weights / class_weights.mean()
            weight_tensor = torch.from_numpy(class_weights).float().to(self.cfg.device)
            self.logger.info(f"Class weights: {class_weights.round(3).tolist()}")
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            for batch in dl_train:
                windows, labels, cluster_labels, ref_windows = batch
                windows = windows.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                cluster_labels = cluster_labels.to(self.cfg.device)
                ref_windows = ref_windows.to(self.cfg.device)

                optimizer.zero_grad()

                logits = model(
                    windows,
                    ref_windows=ref_windows,
                    cluster_labels=cluster_labels,
                    aux_loss_weight=self.aux_loss_weight,
                )

                loss = criterion(logits, labels)
                if model._aux_loss is not None:
                    loss = loss + model._aux_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                train_loss_sum += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_loss = train_loss_sum / max(1, train_total)
            train_acc = train_correct / max(1, train_total)

            # Validation — use zero z_subject (no conditioning) for val
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(self.cfg.device)
                        yb = yb.to(self.cfg.device)
                        # No ref_windows for val — model uses zero embedding
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_loss_sum += loss.item() * yb.size(0)
                        preds = logits.argmax(dim=1)
                        val_correct += (preds == yb).sum().item()
                        val_total += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f}"
            )
            print(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f}"
            )

            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.cfg.device)

        # Store trainer state
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # Save training history
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # Evaluate on val/test splits
        results = {"class_ids": class_ids, "class_names": class_names}

        def eval_loader(dloader, split_name):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(self.cfg.device)
                    logits = model(xb)  # No conditioning for in-training eval
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits, axis=0)
            y_true = np.concatenate(all_y, axis=0)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                class_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, class_labels, normalize=True, filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report": report,
                "confusion_matrix": cm.tolist(),
            }

        results["val"] = eval_loader(dl_val, "val")
        results["test"] = eval_loader(dl_test, "test")

        # Save model
        model_path = self.output_dir / "film_subject_adaptive.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "class_ids": class_ids,
            "mean": mean_c,
            "std": std_c,
            "window_size": window_size,
            "style_dim": self.style_dim,
            "num_pseudo_clusters": actual_n_clusters,
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        results_path = self.output_dir / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate with style-conditioned inference.

        Uses K random windows from X as calibration to compute z_subject,
        then classifies ALL windows with that style embedding.
        """
        assert self.model is not None, "Model is not trained/loaded"
        assert self.mean_c is not None and self.std_c is not None, "Normalization stats missing"
        assert self.class_ids is not None and self.class_names is not None, "Class info missing"

        # Transpose (N, T, C) → (N, C, T) if needed
        X_input = X.copy()
        if X_input.ndim == 3:
            N, dim1, dim2 = X_input.shape
            if dim1 > dim2:
                X_input = np.transpose(X_input, (0, 2, 1))

        # Standardize
        Xs = self._apply_standardization(X_input, self.mean_c, self.std_c)

        # Compute z_subject from calibration windows
        K = min(self.num_ref_windows, len(Xs))
        rng = np.random.RandomState(self.cfg.seed)
        cal_idx = rng.choice(len(Xs), size=K, replace=K > len(Xs))
        cal_windows = torch.from_numpy(Xs[cal_idx]).float().to(self.cfg.device)  # (K, C, T)

        self.model.eval()
        with torch.no_grad():
            z_subject = self.model.style_encoder(cal_windows)  # (style_dim,)

        # Classify all windows with fixed z_subject
        ds = WindowDataset(Xs, y)
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                logits = self.model(xb, z_subject=z_subject)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y, axis=0)
        y_pred = logits.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(self.class_ids)))

        if visualize and self.visualizer is not None:
            class_labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, class_labels, normalize=True, filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "logits": logits,
        }
