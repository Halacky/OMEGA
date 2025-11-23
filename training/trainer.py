import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import asdict
import logging
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

from models.cnn1d import SimpleCNN1D
from training.datasets import WindowDataset
from config.base import TrainingConfig
from utils.logging import seed_everything

class WindowClassifierTrainer:
    def __init__(self, train_cfg: TrainingConfig, logger: logging.Logger, output_dir: Path, visualizer: Optional['Visualizer'] = None):
        self.cfg = train_cfg
        self.logger = logger
        self.output_dir = output_dir
        self.visualizer = visualizer

        self.model: Optional[nn.Module] = None
        self.mean_c: Optional[np.ndarray] = None
        self.std_c: Optional[np.ndarray] = None
        self.class_ids: Optional[List[int]] = None
        self.class_names: Optional[Dict[int, str]] = None
        self.in_channels: Optional[int] = None
        self.window_size: Optional[int] = None

    def _filter_nonempty(self, d: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        return {
            gid: arr for gid, arr in d.items()
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0
        }

    def _prepare_splits_arrays(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict[int, str]]:
        """
            Converts split dictionaries into X,y arrays and class metadata.
            Returns:
            X_train (N, T, C), y_train (N,), X_val, y_val, X_test, y_test,
            class_ids (sorted gesture IDs), class_names {gid: "Gesture gid"}
        """
        train_d = self._filter_nonempty(splits["train"])
        val_d   = self._filter_nonempty(splits["val"])
        test_d  = self._filter_nonempty(splits["test"])

        class_ids = sorted(train_d.keys())
        assert len(class_ids) > 1, "There must be >=2 classes in training"
        for d in [val_d, test_d]:
            extra = set(d.keys()) - set(class_ids)
            if extra:
                self.logger.warning(f"Validation/test contains classes that are not in train: {extra}. They will be ignored.")
        val_d  = {gid: arr for gid, arr in val_d.items() if gid in class_ids}
        test_d = {gid: arr for gid, arr in test_d.items() if gid in class_ids}

        def concat_xy(dct: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            X_list, y_list = [], []
            for i, gid in enumerate(class_ids):
                if gid in dct:
                    X_list.append(dct[gid])
                    y_list.append(np.full((len(dct[gid]),), i, dtype=np.int64))  
            if len(X_list) == 0:
                return np.empty((0,)), np.empty((0,), dtype=np.int64)
            X = np.concatenate(X_list, axis=0)  # (N, T, C)
            y = np.concatenate(y_list, axis=0)  # (N,)
            return X.astype(np.float32), y

        X_train, y_train = concat_xy(train_d)
        X_val, y_val     = concat_xy(val_d)
        X_test, y_test   = concat_xy(test_d)

        class_names = {gid: ("REST" if gid == 0 else f"Gesture {gid}") for gid in class_ids}
        self.logger.info(f"Classes (train): {class_ids} -> {list(class_names.values())}")
        self.logger.info(f"Train: X={X_train.shape}, Val: X={X_val.shape}, Test: X={X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names

    def _compute_channel_standardization(self, X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the mean and std over the channels on the training set.
        X_train: (N, T, C)
        Returns mean(C,), std(C,)
        """
        Xc = np.transpose(X_train, (0, 2, 1))  # (N, C, T)
        mean = Xc.mean(axis=(0, 2))
        std = Xc.std(axis=(0, 2)) + 1e-8
        return mean.astype(np.float32), std.astype(np.float32)

    def _apply_standardization(self, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Applies channel standardization to (N, T, C)"""
        Xc = np.transpose(X, (0, 2, 1))  # (N, C, T)
        Xc = (Xc - mean[None, :, None]) / std[None, :, None]
        return np.transpose(Xc, (0, 2, 1))

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        seed_everything(self.cfg.seed)

        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = self._prepare_splits_arrays(splits)
        num_classes = len(class_ids)
        in_channels = X_train.shape[2]  # C
        window_size = X_train.shape[1]  # T

        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)

        norm_path = self.output_dir / "normalization_stats.npz"
        np.savez_compressed(norm_path, mean=mean_c, std=std_c, class_ids=np.array(class_ids, dtype=np.int32))
        self.logger.info(f"Normalization parameters saved: {norm_path}")

        ds_train = WindowDataset(X_train, y_train)
        ds_val   = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test  = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        dl_train = DataLoader(ds_train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)
        dl_val   = DataLoader(ds_val, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True) if ds_val else None
        dl_test  = DataLoader(ds_test, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True) if ds_test else None

        model = SimpleCNN1D(in_channels=in_channels, num_classes=num_classes, dropout=self.cfg.dropout).to(self.cfg.device)

        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            class_weights = (class_counts.sum() / (class_counts + 1e-8))
            class_weights = class_weights / class_weights.mean()
            weight_tensor = torch.from_numpy(class_weights).float().to(self.cfg.device)
            self.logger.info(f"Class weights: {class_weights.round(3).tolist()}")
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float('inf')
        best_state = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for xb, yb in dl_train:
                xb = xb.to(self.cfg.device)
                yb = yb.to(self.cfg.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * yb.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == yb).sum().item()
                train_total += yb.size(0)

            train_loss /= max(1, train_total)
            train_acc = train_correct / max(1, train_total)

            if dl_val is not None:
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(self.cfg.device)
                        yb = yb.to(self.cfg.device)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_loss += loss.item() * yb.size(0)
                        preds = logits.argmax(dim=1)
                        val_correct += (preds == yb).sum().item()
                        val_total += yb.size(0)
                val_loss /= max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss, val_acc = np.nan, np.nan

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            self.logger.info(f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                             f"Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
                             f"Val: loss={val_loss:.4f}, acc={val_acc:.3f}")

            if dl_val is not None:
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.cfg.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)
        self.logger.info(f"Training history saved: {hist_path}")

        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        results = {"class_ids": class_ids, "class_names": class_names}

        def eval_loader(dloader: Optional[DataLoader], split_name: str) -> Optional[Dict]:
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(self.cfg.device)
                    logits = model(xb)
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits = np.concatenate(all_logits, axis=0)
            y_true = np.concatenate(all_y, axis=0)
            y_pred = logits.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

            if self.visualizer is not None:
                class_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(cm, class_labels, normalize=True, filename=f"cm_{split_name}.png")
                self.visualizer.plot_per_class_f1(report, class_labels, filename=f"f1_{split_name}.png")

                if num_classes >= 2:
                    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
                    self.visualizer.plot_roc_ovr(y_true, probs, class_labels, filename=f"roc_{split_name}.png")

            return {
                "accuracy": float(acc),
                "f1_macro": float(f1_macro),
                "report": report,
                "confusion_matrix": cm.tolist(),
            }

        results["val"] = eval_loader(dl_val, "val")
        results["test"] = eval_loader(dl_test, "test")

        model_path = self.output_dir / "window_cnn1d.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "class_ids": class_ids,
            "mean": mean_c,
            "std": std_c,
            "window_size": window_size,
            "training_config": asdict(self.cfg),
        }, model_path)
        self.logger.info(f"Model saved: {model_path}")

        results_path = self.output_dir / "classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Classification results saved: {results_path}")

        return results

    def evaluate_numpy(self, X: np.ndarray, y: np.ndarray, split_name: str = "custom", visualize: bool = False) -> Dict:
        assert self.model is not None, "Model is not trained/loaded"
        assert self.mean_c is not None and self.std_c is not None, "Normalization stats missing"
        assert self.class_ids is not None and self.class_names is not None, "Class info missing"

        # Standardize with train stats
        Xs = self._apply_standardization(X, self.mean_c, self.std_c)
        ds = WindowDataset(Xs, y)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                        num_workers=self.cfg.num_workers, pin_memory=True)
        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                logits = self.model(xb)
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
            self.visualizer.plot_confusion_matrix(cm, class_labels, normalize=True, filename=f"cm_{split_name}.png")
        return {"accuracy": float(acc), "f1_macro": float(f1_macro),
                "report": report, "confusion_matrix": cm.tolist(), "logits": logits}