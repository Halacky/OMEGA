"""
Trainer for MFCC-based EMG gesture classification (Experiment 113).

Two modes:
  1. **Deep mode** — MFCC spectrogram → 2D CNN (MFCCCNNClassifier).
  2. **ML mode** — MFCC flat features → SVM-RBF / Random Forest.

Extends WindowClassifierTrainer.

LOSO integrity:
  ✓ Channel standardization from training data only (before MFCC extraction).
  ✓ MFCC extractor has no learned parameters (deterministic).
  ✓ PCA/normalization fitted on training features only (ML mode).
  ✓ Model trained on training subjects only.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from torch.utils.data import Dataset as _Dataset

from training.trainer import (
    WindowClassifierTrainer, WindowDataset, get_worker_init_fn, seed_everything,
)
from processing.emg_mfcc import EMGMFCCExtractor
from models.mfcc_cnn_classifier import MFCCCNNClassifier


class _TensorDataset(_Dataset):
    """Generic dataset for arbitrary-dim numpy arrays (supports 4D MFCC)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MFCCTrainer(WindowClassifierTrainer):
    """
    Trainer for MFCC-based EMG classification.

    Args:
        train_cfg:      TrainingConfig dataclass.
        logger:         Python logger.
        output_dir:     Directory for checkpoints and logs.
        visualizer:     Optional Visualizer.
        mode:           "deep" (CNN on spectrogram) or "ml" (SVM/RF on flat features).
        ml_classifier:  "svm_rbf", "svm_linear", or "rf" (only for mode="ml").
        sampling_rate:  EMG sampling rate in Hz.
        n_mfcc:         Number of cepstral coefficients.
        n_mels:         Number of mel filterbank channels.
        fmin:           Lowest frequency in Hz.
        fmax:           Highest frequency in Hz.
        use_deltas:     Include delta and delta-delta features.
        cnn_channels:   Channel sizes for CNN blocks (deep mode).
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        mode: str = "deep",
        ml_classifier: str = "svm_rbf",
        sampling_rate: int = 2000,
        n_mfcc: int = 13,
        n_mels: int = 26,
        fmin: float = 20.0,
        fmax: float = 500.0,
        use_deltas: bool = True,
        cnn_channels: list = None,
        feature_type: str = "mfcc",
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.mode = mode
        self.ml_classifier_type = ml_classifier
        self.sampling_rate = sampling_rate
        self.feature_type = feature_type  # "mfcc" or "fbanks"

        self.mfcc_extractor = EMGMFCCExtractor(
            sampling_rate=sampling_rate,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            use_deltas=use_deltas,
            logger=logger,
        )
        self.cnn_channels = cnn_channels or [32, 64, 128]

        # ML mode state
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._ml_model = None

    # ──────────────────────────── fit ──────────────────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train on LOSO splits.

        Args:
            splits: {"train": Dict[int, np.ndarray], "val": ..., "test": ...}
                    Arrays are (N_gesture, T, C).
        Returns:
            Results dict.
        """
        seed_everything(self.cfg.seed)

        # 1. Flatten splits → arrays
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        self.class_ids = class_ids
        self.class_names = class_names

        num_classes = len(class_ids)
        self.logger.info(
            f"[MFCCTrainer] mode={self.mode}, classes={num_classes}, "
            f"X_train={X_train.shape}"
        )

        # 2. Channel standardization (N, T, C) — training stats only
        # Transpose to (N, C, T) for standardization, then back
        X_train_ct = X_train.transpose(0, 2, 1)  # (N, C, T)
        mean_c = X_train_ct.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        std_c = X_train_ct.std(axis=(0, 2), keepdims=True) + 1e-8

        self.mean_c = mean_c.squeeze()  # (C,)
        self.std_c = std_c.squeeze()    # (C,)

        def _standardize_ntc(X):
            """Standardize (N, T, C) using training channel stats."""
            return (X - self.mean_c[None, None, :]) / self.std_c[None, None, :]

        X_train = _standardize_ntc(X_train)
        if len(X_val) > 0:
            X_val = _standardize_ntc(X_val)
        if len(X_test) > 0:
            X_test = _standardize_ntc(X_test)

        self.logger.info("Channel standardization applied (training stats only).")

        if self.mode == "deep":
            return self._fit_deep(X_train, y_train, X_val, y_val,
                                  X_test, y_test, class_ids, class_names)
        else:
            return self._fit_ml(X_train, y_train, X_val, y_val,
                                X_test, y_test, class_ids, class_names)

    # ──────────────────────── Deep mode (CNN) ─────────────────────────────

    def _fit_deep(self, X_train, y_train, X_val, y_val,
                  X_test, y_test, class_ids, class_names) -> Dict:
        """Train 2D CNN on MFCC spectrograms."""
        # Extract spectrograms (MFCC, log-mel filterbank, or MDCT)
        if self.feature_type == "fbanks":
            _spec_fn = self.mfcc_extractor.transform_fbanks_spectrogram
        elif self.feature_type == "mdct":
            _spec_fn = self.mfcc_extractor.transform_mdct_spectrogram
        else:
            _spec_fn = self.mfcc_extractor.transform_spectrogram
        self.logger.info(f"Computing {self.feature_type} spectrograms for training data...")
        mfcc_train = _spec_fn(X_train)  # (N, n_coeff, T_f, C)
        # Reshape to (N, C, n_coeff, T_f) for 2D CNN
        mfcc_train = mfcc_train.transpose(0, 3, 1, 2)  # (N, C, n_coeff, T_f)

        mfcc_val = None
        if len(X_val) > 0:
            mfcc_val = _spec_fn(X_val)
            mfcc_val = mfcc_val.transpose(0, 3, 1, 2)

        mfcc_test = None
        if len(X_test) > 0:
            mfcc_test = _spec_fn(X_test)
            mfcc_test = mfcc_test.transpose(0, 3, 1, 2)

        in_channels = mfcc_train.shape[1]  # C
        n_coeff = mfcc_train.shape[2]
        n_frames = mfcc_train.shape[3]
        num_classes = len(class_ids)

        self.logger.info(
            f"MFCC spectrogram shape: ({in_channels}, {n_coeff}, {n_frames}), "
            f"classes={num_classes}"
        )

        # Build model
        model = MFCCCNNClassifier(
            in_channels=in_channels,
            n_coeff=n_coeff,
            n_frames=n_frames,
            num_classes=num_classes,
            cnn_channels=self.cnn_channels,
            dropout=self.cfg.dropout,
        ).to(self.cfg.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"MFCCCNNClassifier: {total_params:,} parameters")

        # Datasets
        ds_train = _TensorDataset(mfcc_train, y_train)
        ds_val = _TensorDataset(mfcc_val, y_val) if mfcc_val is not None else None
        ds_test = _TensorDataset(mfcc_test, y_test) if mfcc_test is not None else None

        g = torch.Generator().manual_seed(self.cfg.seed)
        worker_init = get_worker_init_fn(self.cfg.seed)

        dl_train = DataLoader(
            ds_train, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
            generator=g,
        )
        dl_val = DataLoader(
            ds_val, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
        ) if ds_val else None

        # Loss
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = counts.sum() / (counts + 1e-8)
            cw /= cw.mean()
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cw).float().to(self.cfg.device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer + scheduler
        optimizer = optim.Adam(
            model.parameters(), lr=self.cfg.learning_rate,
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
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            ep_loss, ep_correct, ep_total = 0.0, 0, 0

            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = xb.size(0)
                ep_loss += loss.item() * bs
                ep_correct += (logits.argmax(1) == yb).sum().item()
                ep_total += bs

            train_loss = ep_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)

            # Validation
            if dl_val is not None:
                model.eval()
                vl, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        vl += criterion(logits, yb).item() * yb.size(0)
                        vc += (logits.argmax(1) == yb).sum().item()
                        vt += yb.size(0)
                val_loss = vl / max(1, vt)
                val_acc = vc / max(1, vt)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if epoch % 5 == 0 or epoch == 1:
                self.logger.info(
                    f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                    f"loss={train_loss:.4f}, acc={train_acc:.3f} | "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
                )

            # Early stopping
            if dl_val is not None:
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}.")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        self.model = model
        self.in_channels = mfcc_train.shape[1]
        self.window_size = X_train.shape[1]

        # Save
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": mfcc_train.shape[1],
            "n_coeff": n_coeff,
            "n_frames": n_frames,
            "num_classes": num_classes,
            "class_ids": class_ids,
            "mean_c": self.mean_c,
            "std_c": self.std_c,
        }, self.output_dir / "mfcc_cnn.pt")

        results = {"class_ids": class_ids, "class_names": class_names}
        return results

    # ──────────────────────── ML mode (SVM/RF) ────────────────────────────

    def _fit_ml(self, X_train, y_train, X_val, y_val,
                X_test, y_test, class_ids, class_names) -> Dict:
        """Train SVM/RF on flat MFCC features."""
        if self.feature_type == "fbanks":
            _flat_fn = self.mfcc_extractor.transform_fbanks
        elif self.feature_type == "mdct":
            _flat_fn = self.mfcc_extractor.transform_mdct
        else:
            _flat_fn = self.mfcc_extractor.transform
        self.logger.info(f"Computing flat {self.feature_type} features for training data...")
        feat_train = _flat_fn(X_train)  # (N, F)
        self.logger.info(f"MFCC features shape: {feat_train.shape}")

        # Standardize features
        self._scaler = StandardScaler()
        feat_train = self._scaler.fit_transform(feat_train)

        # PCA if feature dim is large
        feat_dim = feat_train.shape[1]
        if feat_dim > 200:
            n_components = min(200, feat_train.shape[0] - 1, feat_dim)
            self._pca = PCA(n_components=n_components, random_state=self.cfg.seed)
            feat_train = self._pca.fit_transform(feat_train)
            self.logger.info(
                f"PCA: {feat_dim} → {n_components} "
                f"(explained variance: {self._pca.explained_variance_ratio_.sum():.3f})"
            )

        # Train ML classifier
        if self.ml_classifier_type == "svm_rbf":
            self._ml_model = svm.SVC(kernel="rbf", C=10.0, gamma="scale",
                                     decision_function_shape="ovr",
                                     random_state=self.cfg.seed)
        elif self.ml_classifier_type == "svm_linear":
            self._ml_model = svm.LinearSVC(C=1.0, max_iter=5000,
                                           random_state=self.cfg.seed)
        elif self.ml_classifier_type == "rf":
            self._ml_model = RandomForestClassifier(
                n_estimators=300, max_depth=None,
                random_state=self.cfg.seed, n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown ml_classifier: {self.ml_classifier_type}")

        self.logger.info(f"Training {self.ml_classifier_type} on {feat_train.shape}...")
        self._ml_model.fit(feat_train, y_train)
        self.logger.info("ML model fitted.")

        results = {"class_ids": class_ids, "class_names": class_names}
        return results

    # ──────────────────── evaluate_numpy ──────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate on (X, y) numpy arrays.

        Applies training-data channel standardization, computes MFCC,
        then runs inference with the trained model.

        Args:
            X: (N, T, C) raw EMG windows.
            y: (N,) class indices.
            split_name: prefix for saved plots.
            visualize: whether to save confusion matrix.

        Returns:
            dict with "accuracy", "f1_macro", "report", "confusion_matrix".
        """
        assert self.class_ids is not None, "Model not trained yet."

        X_std = (X - self.mean_c[None, None, :]) / self.std_c[None, None, :]

        if self.mode == "deep":
            return self._evaluate_deep(X_std, y, split_name, visualize)
        else:
            return self._evaluate_ml(X_std, y, split_name, visualize)

    def _evaluate_deep(self, X_std, y, split_name, visualize) -> Dict:
        """Evaluate CNN model."""
        assert self.model is not None

        if self.feature_type == "fbanks":
            _spec_fn = self.mfcc_extractor.transform_fbanks_spectrogram
        elif self.feature_type == "mdct":
            _spec_fn = self.mfcc_extractor.transform_mdct_spectrogram
        else:
            _spec_fn = self.mfcc_extractor.transform_spectrogram
        mfcc = _spec_fn(X_std)
        mfcc = mfcc.transpose(0, 3, 1, 2)  # (N, C, n_coeff, T_f)

        ds = _TensorDataset(mfcc, y)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                        num_workers=self.cfg.num_workers, pin_memory=True)

        self.model.eval()
        all_logits, all_y = [], []
        device = self.cfg.device
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                all_logits.append(self.model(xb).cpu().numpy())
                all_y.append(yb.numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_y, axis=0)
        y_pred = logits.argmax(axis=1)

        return self._format_results(y_true, y_pred, logits, split_name, visualize)

    def _evaluate_ml(self, X_std, y, split_name, visualize) -> Dict:
        """Evaluate ML model."""
        assert self._ml_model is not None

        if self.feature_type == "fbanks":
            _flat_fn = self.mfcc_extractor.transform_fbanks
        elif self.feature_type == "mdct":
            _flat_fn = self.mfcc_extractor.transform_mdct
        else:
            _flat_fn = self.mfcc_extractor.transform
        feat = _flat_fn(X_std)
        feat = self._scaler.transform(feat)
        if self._pca is not None:
            feat = self._pca.transform(feat)

        y_pred = self._ml_model.predict(feat)

        return self._format_results(y, y_pred, None, split_name, visualize)

    def _format_results(self, y_true, y_pred, logits, split_name, visualize) -> Dict:
        """Format evaluation results."""
        num_classes = len(self.class_ids)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        if visualize and self.visualizer is not None:
            labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm, labels, normalize=True, filename=f"cm_{split_name}.png"
            )

        result = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "report": report,
            "confusion_matrix": cm.tolist(),
        }
        if logits is not None:
            result["logits"] = logits
        return result
