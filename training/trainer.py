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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import joblib
from models.cnn1d import SimpleCNN1D
from training.datasets import WindowDataset, FeatureDataset, AugmentedWindowDataset
from config.base import TrainingConfig
from utils.logging import seed_everything
from utils.logging import get_worker_init_fn
from processing.features import HandcraftedFeatureExtractor
from utils.data_quality_check import DataQualityDiagnostic
from processing.powerful_features import PowerfulFeatureExtractor
from sklearn.decomposition import PCA  # NEW
from models.hybrid_powerful_deep import HybridPowerfulDeepNet
from models.tcn import TemporalConvNet, TemporalConvNetWithAttention


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

        self.feature_extractor: Optional[HandcraftedFeatureExtractor] = None
        self.using_features: bool = getattr(train_cfg, "use_handcrafted_features", False)
        self.diagnostic_enabled = True

    def _create_model(self, in_channels: int, num_classes: int, model_type: str = "simple_cnn") -> nn.Module:
        """
        Factory method to create different model architectures.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            model_type: Type of model to create
                - "simple_cnn": Original SimpleCNN1D
                - "attention_cnn": CNN with channel and spatial attention
                - "tcn": Temporal Convolutional Network
                - "multiscale_cnn": Multi-scale CNN with Inception modules
                - "bilstm": Bidirectional LSTM
                - "bilstm_attention": Bidirectional LSTM with attention
                - "bigru": Bidirectional GRU
                - "cnn_lstm": Hybrid CNN-LSTM
                - "cnn_gru_attention": Hybrid CNN-GRU with attention
        
        Returns:
            Initialized model
        """
        from models import (SimpleCNN1D, AttentionCNN1D, TemporalConvNet, MultiScaleCNN1D,
                            BiLSTM, BiLSTMWithAttention, BiGRU, CNNLSTM, CNNGRUWithAttention,
                            ResNet1D)
        
        model_type = model_type.lower()
        
        if model_type == "resnet1d":
            model = ResNet1D(
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=32,
                blocks_per_stage=(2, 2, 2),
                kernel_size=5,
                dropout=self.cfg.dropout,
            )
        elif model_type == "simple_cnn":
            model = SimpleCNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=self.cfg.dropout
            )
        elif model_type == "attention_cnn":
            model = AttentionCNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=self.cfg.dropout
            )
        elif model_type == "tcn":
            model = TemporalConvNet(
                in_channels=in_channels,
                num_classes=num_classes,
                num_channels=[32, 64, 128],
                kernel_size=3,
                dropout=self.cfg.dropout
            )
        elif model_type == "tcn_attn":
            model = TemporalConvNetWithAttention(
                in_channels=in_channels,
                num_classes=num_classes,
                num_channels=[32, 64, 128],
                kernel_size=3,
                dropout=self.cfg.dropout,
                attn_heads=4,
                attn_dropout=0.1,
            )
        elif model_type == "multiscale_cnn":
            model = MultiScaleCNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=self.cfg.dropout
            )
        elif model_type == "bilstm":
            model = BiLSTM(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=128,
                num_layers=2,
                dropout=self.cfg.dropout
            )
        elif model_type == "bilstm_attention":
            model = BiLSTMWithAttention(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=128,
                num_layers=2,
                dropout=self.cfg.dropout
            )
        elif model_type == "bigru":
            model = BiGRU(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=128,
                num_layers=2,
                dropout=self.cfg.dropout
            )
        elif model_type == "cnn_lstm":
            model = CNNLSTM(
                in_channels=in_channels,
                num_classes=num_classes,
                cnn_channels=[32, 64],
                lstm_hidden=128,
                lstm_layers=2,
                dropout=self.cfg.dropout
            )
        elif model_type == "cnn_gru_attention":
            model = CNNGRUWithAttention(
                in_channels=in_channels,
                num_classes=num_classes,
                cnn_channels=[32, 64],
                gru_hidden=128,
                gru_layers=2,
                dropout=self.cfg.dropout
            )
        elif model_type == "hybrid_powerful_deep":
            # Note: here "in_channels" is actually "feature dimension" when using handcrafted features
            hidden_dim = getattr(self.cfg, "hybrid_hidden_dim", 256)
            use_da = getattr(self.cfg, "hybrid_use_domain_adaptation", True)
            num_domains = getattr(self.cfg, "hybrid_num_domains", 2)
            grl_lambda = getattr(self.cfg, "hybrid_grl_lambda", 1.0)

            model = HybridPowerfulDeepNet(
                in_features=in_channels,
                num_classes=num_classes,
                num_domains=num_domains,
                hidden_dim=hidden_dim,
                dropout=self.cfg.dropout,
                use_domain_adaptation=use_da,
                grl_lambda=grl_lambda,
            )
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Choose from: resnet1d, simple_cnn, attention_cnn, tcn, multiscale_cnn, "
                f"bilstm, bilstm_attention, bigru, cnn_lstm, cnn_gru_attention, "
                f"hybrid_powerful_deep"
            )
        return model

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
        Compute mean and std per channel for standardization.
        Expects input in PyTorch format: (N, C, T) where C is channels.
        """
        if X_train.ndim != 3:
            raise ValueError(f"Expected 3D input, got {X_train.shape}")
        
        N, C, T = X_train.shape
        
        # For PyTorch Conv1d: (N, C, T)
        # Compute stats over samples and time: axes (0, 2)
        mean = X_train.mean(axis=(0, 2))  # shape: (C,)
        std = X_train.std(axis=(0, 2)) + 1e-8  # shape: (C,)
        
        self.logger.info(
            f"[Standardization] Input shape: (N={N}, C={C}, T={T}). "
            f"Computed stats: mean shape={mean.shape}, std shape={std.shape}"
        )
        
        return mean.astype(np.float32), std.astype(np.float32)


    def _apply_standardization(self, Xc: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Apply per-channel standardization to input data.
        Expects input in PyTorch format: (N, C, T) where C is channels.
        """
        if Xc.ndim != 3:
            raise ValueError(f"Expected 3D input, got {Xc.shape}")
        
        if mean.ndim > 1:
            mean = mean.reshape(-1)
            std = std.reshape(-1)
        
        N, C, T = Xc.shape
        expected_C = mean.shape[0]
        
        if C != expected_C:
            raise ValueError(
                f"Channel dimension mismatch: data has C={C} channels, "
                f"but normalization stats are for C={expected_C} channels. "
                f"Data shape: {Xc.shape}"
            )
        
        # Apply standardization: (N, C, T) format
        Xs = (Xc - mean[None, :, None]) / std[None, :, None]
        
        return Xs

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        seed_everything(self.cfg.seed)
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)
        if self.diagnostic_enabled and hasattr(self, 'output_dir'):
            diagnostic = DataQualityDiagnostic(self.logger)
            issues = diagnostic.check_all(splits, class_names)
            
            # Сохранить отчет
            diagnostic_path = self.output_dir / "data_quality_report.txt"
            with open(diagnostic_path, 'w', encoding='utf-8') as f:
                f.write("ДИАГНОСТИКА КАЧЕСТВА ДАННЫХ\n")
                f.write("="*80 + "\n")
                f.write(f"Классы: {class_ids}\n")
                f.write("\nПРОБЛЕМЫ:\n")
                for i, issue in enumerate(issues, 1):
                    f.write(f"{i}. {issue}\n")
            
            if any("❌ КРИТИЧНО" in issue for issue in issues):
                self.logger.error("Обнаружены КРИТИЧЕСКИЕ проблемы с данными!")



        use_features = getattr(self.cfg, "use_handcrafted_features", False)
        model_type = getattr(self.cfg, "model_type", "simple_cnn")
        is_hybrid = (model_type == "hybrid_powerful_deep")
        orig_train_shape = X_train.shape 
        if use_features:
            feat_set = getattr(self.cfg, "handcrafted_feature_set", "basic_v1")
            self.logger.info(
                f"Using hand-crafted features (set='{feat_set}') instead of raw windows"
            )
            if self.feature_extractor is None:
                if feat_set == "powerful":
                    self.feature_extractor = PowerfulFeatureExtractor(
                        sampling_rate=self.window_size or 2000,
                        logger=self.logger,
                        feature_set="powerful",
                        n_jobs=-1,
                        # use_torch=False,
                        use_torch=True,
                        device='cuda'
                    )
                else:
                    self.feature_extractor = HandcraftedFeatureExtractor(
                        sampling_rate=None,
                        logger=self.logger,
                        feature_set=feat_set,
                    )
            
            # IMPORTANT: Store original shape before feature extraction
            orig_train_shape = X_train.shape  # (N, T, C)
            
            X_train = self.feature_extractor.transform(X_train)
            if len(X_val) > 0:
                X_val = self.feature_extractor.transform(X_val)
            if len(X_test) > 0:
                X_test = self.feature_extractor.transform(X_test)
            
            self.logger.info(
                f"After feature extraction: X_train shape={X_train.shape}, "
                f"X_val shape={X_val.shape if isinstance(X_val, np.ndarray) else 'empty'}, "
                f"X_test shape={X_test.shape if isinstance(X_test, np.ndarray) else 'empty'}"
            )

        if not is_hybrid:
            if X_train.ndim == 2:
                # Features are flat (N, F) → add dummy time dimension
                X_train = X_train[:, :, None]  # (N, F, 1)
                if isinstance(X_val, np.ndarray) and len(X_val) > 0:
                    X_val = X_val[:, :, None]
                if isinstance(X_test, np.ndarray) and len(X_test) > 0:
                    X_test = X_test[:, :, None]
                self.logger.info(f"Reshaped flat features: X_train={X_train.shape}")
                
            elif X_train.ndim == 3:
                # Check if we need to transpose to (N, C, T) format
                N, dim1, dim2 = X_train.shape
                
                # Heuristic: channels dimension is typically smaller
                if use_features and feat_set == "emg_td_seq":
                    # Sequential features: (N, n_frames, n_features) → (N, n_features, n_frames)
                    X_train = np.transpose(X_train, (0, 2, 1))
                    if isinstance(X_val, np.ndarray) and len(X_val) > 0:
                        X_val = np.transpose(X_val, (0, 2, 1))
                    if isinstance(X_test, np.ndarray) and len(X_test) > 0:
                        X_test = np.transpose(X_test, (0, 2, 1))
                    self.logger.info(
                        f"Transposed sequential features to (N, C, T): X_train={X_train.shape}"
                    )
                elif dim1 > dim2:
                    # Likely (N, T, C) → transpose to (N, C, T)
                    X_train = np.transpose(X_train, (0, 2, 1))
                    if isinstance(X_val, np.ndarray) and len(X_val) > 0:
                        X_val = np.transpose(X_val, (0, 2, 1))
                    if isinstance(X_test, np.ndarray) and len(X_test) > 0:
                        X_test = np.transpose(X_test, (0, 2, 1))
                    self.logger.info(
                        f"Transposed raw data to (N, C, T): X_train={X_train.shape}"
                    )
                else:
                    self.logger.info(
                        f"Data already in (N, C={dim1}, T={dim2}) format"
                    )

        # Determine input dimensions for model
        num_classes = len(class_ids)
        if is_hybrid:
            # Hybrid model uses flat features
            in_channels = X_train.shape[1]
            window_size = 1
        else:
            # CNN models expect (N, C, T)
            in_channels = X_train.shape[1]  # C
            window_size = X_train.shape[2]   # T

        self.logger.info(
            f"Model input dimensions: in_channels={in_channels}, window_size={window_size}, "
            f"num_classes={num_classes}"
        )
        if use_features and feat_set == "emg_td_seq":
            # Sequential features: compute stats over (N, T) for each feature channel
            mean_c = X_train.mean(axis=(0, 2)).astype(np.float32)   
            std_c  = (X_train.std(axis=(0, 2)) + 1e-8).astype(np.float32)
        else:
            # Standard case: compute per-channel stats
            if is_hybrid:
                # Flat features (N, F)
                mean_c = X_train.mean(axis=0).astype(np.float32)
                std_c  = (X_train.std(axis=0) + 1e-8).astype(np.float32)
            else:
                # CNN format (N, C, T)
                mean_c, std_c = self._compute_channel_standardization(X_train)

        self.logger.info(
            f"Computed normalization: mean shape={mean_c.shape}, std shape={std_c.shape}"
        )


        norm_path = self.output_dir / "normalization_stats.npz"
        np.savez_compressed(norm_path, mean=mean_c, std=std_c, class_ids=np.array(class_ids, dtype=np.int32))
        self.logger.info(f"Normalization parameters saved: {norm_path}")

        # Apply same standardization to train/val/test so training and evaluation see same distribution.
        if is_hybrid:
            X_train = (X_train - mean_c[None, :]) / std_c[None, :]
            if len(X_val) > 0:
                X_val = (X_val - mean_c[None, :]) / std_c[None, :]
            if len(X_test) > 0:
                X_test = (X_test - mean_c[None, :]) / std_c[None, :]
        else:
            X_train = self._apply_standardization(X_train, mean_c, std_c)
            if len(X_val) > 0:
                X_val = self._apply_standardization(X_val, mean_c, std_c)
            if len(X_test) > 0:
                X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardization to train/val/test.")

        # On-the-fly augmentation only for raw windows (not hybrid).
        use_aug = getattr(self.cfg, "aug_apply", False) and not is_hybrid
        if use_aug:
            self.logger.info(
                f"Using on-the-fly EMG augmentation: noise_std={self.cfg.aug_noise_std}, "
                f"time_warp_max={self.cfg.aug_time_warp_max}, "
                f"noise={self.cfg.aug_apply_noise}, warp={self.cfg.aug_apply_time_warp}"
            )

        if is_hybrid:
            # Hybrid model uses 2D feature data
            ds_train = FeatureDataset(X_train, y_train)
            ds_val   = FeatureDataset(X_val, y_val) if len(X_val) > 0 else None
            ds_test  = FeatureDataset(X_test, y_test) if len(X_test) > 0 else None
        else:
            # Standard models use 3D window data; train with augmentation when enabled
            if use_aug:
                ds_train = AugmentedWindowDataset(
                    X_train, y_train,
                    noise_std=getattr(self.cfg, "aug_noise_std", 0.02),
                    max_warp=getattr(self.cfg, "aug_time_warp_max", 0.1),
                    apply_noise=getattr(self.cfg, "aug_apply_noise", True),
                    apply_time_warp=getattr(self.cfg, "aug_apply_time_warp", False),
                )
            else:
                ds_train = WindowDataset(X_train, y_train)
            ds_val   = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
            ds_test  = WindowDataset(X_test, y_test) if len(X_test) > 0 else None
        worker_init_fn = get_worker_init_fn(self.cfg.seed)

        dl_train = DataLoader(
            ds_train, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=self.cfg.num_workers, 
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed)  # For shuffle reproducibility
        )

        dl_val = DataLoader(
            ds_val, 
            batch_size=self.cfg.batch_size, 
            shuffle=False, 
            num_workers=self.cfg.num_workers, 
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None
        ) if ds_val else None

        dl_test = DataLoader(
            ds_test, 
            batch_size=self.cfg.batch_size, 
            shuffle=False, 
            num_workers=self.cfg.num_workers, 
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None
        ) if ds_test else None
        model_type = getattr(self.cfg, 'model_type', 'simple_cnn')
        model = self._create_model(in_channels, num_classes, model_type).to(self.cfg.device)
        self.logger.info(f"Created model: {model_type}")

        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            class_weights = (class_counts.sum() / (class_counts + 1e-8))
            class_weights = class_weights / class_weights.mean()
            weight_tensor = torch.from_numpy(class_weights).float().to(self.cfg.device)
            self.logger.info(f"Class weights: {class_weights.round(3).tolist()}")
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        use_domain_adaptation = (
            model_type == "hybrid_powerful_deep"
            and getattr(self.cfg, "hybrid_use_domain_adaptation", False)
        )
        domain_loss_weight = float(getattr(self.cfg, "hybrid_domain_loss_weight", 0.2))
        if use_domain_adaptation:
            # For now we assume domain labels are not provided -> disable DA loss
            # This is left here for future extension when domain labels become available.
            self.logger.info(
                "[Hybrid] Domain adaptation is enabled in config, "
                "but no domain labels are passed. Domain loss will be zero."
            )
            domain_criterion = nn.CrossEntropyLoss()
        else:
            domain_criterion = None

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

                # Support both standard models and hybrid_powerful_deep
                if model_type == "hybrid_powerful_deep":
                    # Model returns (logits_class, logits_domain)
                    logits_class, logits_domain = model(xb, return_domain=use_domain_adaptation)
                    loss_class = criterion(logits_class, yb)

                    loss_domain = 0.0
                    if use_domain_adaptation and domain_criterion is not None and logits_domain is not None:
                        # NOTE: for now we do not have domain labels in this trainer.
                        # When domain labels are available, replace zeros with true labels.
                        # domain_labels = ...
                        # loss_domain = domain_criterion(logits_domain, domain_labels)
                        loss_domain = 0.0

                    loss = loss_class + domain_loss_weight * loss_domain
                    logits = logits_class  # for accuracy computation
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

                        if model_type == "hybrid_powerful_deep":
                            logits_class, logits_domain = model(xb, return_domain=False)
                            logits = logits_class
                            loss = criterion(logits_class, yb)
                        else:
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
            print(f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
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

                    if model_type == "hybrid_powerful_deep":
                        logits_class, _ = model(xb, return_domain=False)
                        logits = logits_class
                    else:
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
                    logits_tensor = torch.from_numpy(logits)
                    logits_tensor = torch.clamp(logits_tensor, -1e6, 1e6)
                    probs = torch.softmax(logits_tensor, dim=1).numpy()

                    if not np.all(np.isfinite(probs)):
                        self.logger.warning(
                            f"[eval_loader:{split_name}] ROC не построен: "
                            f"probs содержит NaN/Inf "
                            f"(min={np.nanmin(probs)}, max={np.nanmax(probs)})"
                        )
                    else:
                        self.visualizer.plot_roc_ovr(
                            y_true, probs, class_labels, filename=f"roc_{split_name}.png"
                        )
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
        
        use_features = getattr(self.cfg, "use_handcrafted_features", False)
        model_type = getattr(self.cfg, "model_type", "simple_cnn")
        is_hybrid = (model_type == "hybrid_powerful_deep")
        
        X_input = X
        
        # Extract features if needed
        if use_features:
            feat_set = getattr(self.cfg, "handcrafted_feature_set", "basic_v1")
            if self.feature_extractor is None:
                if feat_set == "powerful":
                    self.feature_extractor = PowerfulFeatureExtractor(
                        sampling_rate=self.window_size or 2000,
                        logger=self.logger,
                        feature_set="powerful",
                        n_jobs=-1,
                        # use_torch=False,
                        use_torch=True,
                        device='cuda'
                    )
                else:
                    self.feature_extractor = HandcraftedFeatureExtractor(
                        sampling_rate=None,
                        logger=self.logger,
                        feature_set=feat_set,
                    )
            
            if X_input.ndim == 3:
                X_input = self.feature_extractor.transform(X_input)
            elif X_input.ndim == 2:
                pass  # Already features
        
        # Reshape to correct format BEFORE standardization
        if not is_hybrid:
            if X_input.ndim == 2:
                X_input = X_input[:, :, None]  # (N, F, 1)
            elif X_input.ndim == 3:
                N, dim1, dim2 = X_input.shape
                # Transpose if needed to get (N, C, T) format
                if use_features and getattr(self.cfg, "handcrafted_feature_set", "") == "emg_td_seq":
                    X_input = np.transpose(X_input, (0, 2, 1))
                elif dim1 > dim2:
                    X_input = np.transpose(X_input, (0, 2, 1))
        
        # Apply standardization (now data is in correct format)
        if is_hybrid:
            if X_input.ndim != 2:
                raise ValueError(f"[Hybrid] Expected 2D data (N, F), got {X_input.shape}")
            Xs = (X_input - self.mean_c[None, :]) / self.std_c[None, :]
        else:
            if X_input.ndim != 3:
                raise ValueError(f"Expected 3D data (N, C, T), got {X_input.shape}")
            Xs = self._apply_standardization(X_input, self.mean_c, self.std_c)
        
        # Create dataset and dataloader
        if is_hybrid:
            ds = FeatureDataset(Xs, y)
        else:
            ds = WindowDataset(Xs, y)
        
        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        dl = DataLoader(
            ds, 
            batch_size=self.cfg.batch_size, 
            shuffle=False,
            num_workers=self.cfg.num_workers, 
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None
        )
        
        # Evaluate
        self.model.eval()
        all_logits, all_y = [], []
        model_type = getattr(self.cfg, "model_type", "simple_cnn")
        
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                if model_type == "hybrid_powerful_deep":
                    logits_class, _ = self.model(xb, return_domain=False)
                    logits = logits_class
                else:
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
        
        return {
            "accuracy": float(acc), 
            "f1_macro": float(f1_macro),
            "report": report, 
            "confusion_matrix": cm.tolist(), 
            "logits": logits
        }

class FeatureMLTrainer(WindowClassifierTrainer):
    """
    Trainer for classical ML models (SVM, Random Forest, etc.) on top of
    hand-crafted EMG features.

    It reuses the split preparation and class metadata logic from
    WindowClassifierTrainer but replaces the PyTorch model with a
    scikit-learn estimator.
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        output_dir: Path,
        visualizer: Optional['Visualizer'] = None,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.ml_model = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None

    # --------- ML model factory --------- #

    def _create_ml_model(self, model_type: str):
        """
        Creates a scikit-learn model for the given model_type.
        """
        model_type = model_type.lower()
        if model_type == "svm_rbf":
            return svm.SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=self.cfg.seed,
            )
        elif model_type == "svm_linear":
            return svm.SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=self.cfg.seed,
            )
        elif model_type in ("rf", "random_forest"):
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=self.cfg.seed,
            )
        else:
            raise ValueError(
                f"Unknown ML model type: {model_type}. "
                f"Choose from: svm_rbf, svm_linear, rf"
            )

    def _hyperparam_search(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """
        Simple manual hyperparameter search for SVM/RF.

        We keep it lightweight and fully deterministic, without depending on
        sklearn's GridSearchCV, to stay close to the existing code structure.

        Returns:
            best_model, best_params_dict or (None, None) if something failed.
        """
        max_configs = int(getattr(self.cfg, "ml_max_search_configs", 20))
        model_type = model_type.lower()

        search_space = []
        if model_type == "svm_rbf":
            C_list = [0.1, 1.0, 10.0, 100.0]
            gamma_list = ["scale", 0.01, 0.001]
            class_weight_list = ["balanced"]
            for C in C_list:
                for gamma in gamma_list:
                    for cw in class_weight_list:
                        search_space.append(
                            {"C": C, "gamma": gamma, "class_weight": cw}
                        )
        elif model_type == "svm_linear":
            C_list = [0.1, 1.0, 10.0, 100.0]
            class_weight_list = ["balanced"]
            for C in C_list:
                for cw in class_weight_list:
                    search_space.append({"C": C, "class_weight": cw})
        elif model_type in ("rf", "random_forest"):
            max_depth_list = [None, 20, 40]
            min_samples_leaf_list = [1, 5, 10]
            max_features_list = ["sqrt", 0.5, 0.3]
            for md in max_depth_list:
                for msl in min_samples_leaf_list:
                    for mf in max_features_list:
                        search_space.append(
                            {
                                "max_depth": md,
                                "min_samples_leaf": msl,
                                "max_features": mf,
                            }
                        )
        else:
            self.logger.warning(
                f"[FeatureMLTrainer] Hyperparam search not implemented for model_type='{model_type}'"
            )
            return None, None

        if len(search_space) > max_configs:
            self.logger.info(
                f"[FeatureMLTrainer] Reducing search space from {len(search_space)} "
                f"to {max_configs} configurations"
            )
            search_space = search_space[:max_configs]

        best_model = None
        best_params = None
        best_score = -np.inf

        for i, params in enumerate(search_space, 1):
            self.logger.info(
                f"[FeatureMLTrainer] Hyperparam config {i}/{len(search_space)}: {params}"
            )
            try:
                if model_type.startswith("svm"):
                    if model_type == "svm_rbf":
                        model = svm.SVC(
                            kernel="rbf",
                            probability=True,
                            random_state=self.cfg.seed,
                            **params,
                        )
                    else:
                        model = svm.SVC(
                            kernel="linear",
                            probability=True,
                            random_state=self.cfg.seed,
                            **params,
                        )
                else:
                    # RandomForest
                    model = RandomForestClassifier(
                        n_estimators=300,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        random_state=self.cfg.seed,
                        **params,
                    )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average="macro")
                self.logger.info(
                    f"[FeatureMLTrainer] Config {i}: val F1-macro={score:.4f}"
                )

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
            except Exception as e:
                self.logger.error(
                    f"[FeatureMLTrainer] Failed to evaluate config {params}: {e}"
                )

        if best_model is None:
            return None, None
        return best_model, best_params

    # --------- main fit --------- #

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """
        Fit classical ML model on hand-crafted EMG features.

        - Uses HandcraftedFeatureExtractor or PowerfulFeatureExtractor depending
          on feature_set in config (e.g. 'emg_td', 'powerful').
        - Standardizes features (per feature dimension).
        - Optionally applies feature selection (RandomForest importances).
        - Optionally applies PCA.
        - Optionally performs hyperparameter search for SVM/RF.
        - Trains ML model (SVM / RF).
        """
        seed_everything(self.cfg.seed)

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            class_ids,
            class_names,
        ) = self._prepare_splits_arrays(splits)

        num_classes = len(class_ids)

        # ----- 1) Feature extraction -----
        feat_set = getattr(self.cfg, "handcrafted_feature_set", "emg_td")
        self.logger.info(
            f"[FeatureMLTrainer] Using hand-crafted features for ML "
            f"(feature_set='{feat_set}')"
        )

        if self.feature_extractor is None:
            if feat_set == "powerful":
                # Use optimized powerful EMG features
                # self.feature_extractor = PowerfulFeatureExtractor(
                #     sampling_rate=2000,
                #     logger=self.logger,
                #     feature_set="powerful",
                #     n_jobs=-1,
                #     # use_torch=False,
                #     use_torch=True,
                #     device='cuda'
                # )
                self.feature_extractor = PowerfulFeatureExtractor(
                    sampling_rate=2000,
                    logger=self.logger,
                    feature_set="powerful",
                    n_jobs=-1,
                    use_torch=True,          # ← GPU режим
                    device='cuda',            # ← Используем CUDA
                    gpu_batch_size=4096,     # ← Размер батча для GPU
                )
            else:
                self.feature_extractor = HandcraftedFeatureExtractor(
                    sampling_rate=None,
                    logger=self.logger,
                    feature_set=feat_set,
                )

        X_train = self.feature_extractor.transform(X_train)  # (N, F)
        if len(X_val) > 0:
            X_val = self.feature_extractor.transform(X_val)
        if len(X_test) > 0:
            X_test = self.feature_extractor.transform(X_test)

        self.logger.info(
            f"[FeatureMLTrainer] Raw features: "
            f"X_train={X_train.shape}, "
            f"X_val={X_val.shape if isinstance(X_val, np.ndarray) else X_val}, "
            f"X_test={X_test.shape if isinstance(X_test, np.ndarray) else X_test}"
        )

        # ----- 2) Standardization (per feature dimension) -----
        self.feature_mean = X_train.mean(axis=0).astype(np.float32)  # (F_raw,)
        self.feature_std = (X_train.std(axis=0) + 1e-8).astype(np.float32)

        def standardize(arr: np.ndarray) -> np.ndarray:
            return (arr - self.feature_mean[None, :]) / self.feature_std[None, :]

        X_train = standardize(X_train)
        if len(X_val) > 0:
            X_val = standardize(X_val)
        if len(X_test) > 0:
            X_test = standardize(X_test)

        # ----- 3) Optional feature selection (RandomForest importances) -----
        if getattr(self.cfg, "ml_use_feature_selection", False):
            self.logger.info("[FeatureMLTrainer] Applying feature selection using RandomForest importances")
            fs_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=self.cfg.seed,
            )
            fs_rf.fit(X_train, y_train)
            importances = fs_rf.feature_importances_
            # Decide which features to keep
            top_k = getattr(self.cfg, "ml_feature_selection_top_k", None)
            if top_k is not None and top_k > 0:
                k = min(top_k, importances.shape[0])
                idx = np.argsort(importances)[-k:]
                self.logger.info(
                    f"[FeatureMLTrainer] Keeping top-{k} features by importance "
                    f"(out of {importances.shape[0]})"
                )
            else:
                # Threshold at median importance as a simple heuristic
                thr = np.median(importances)
                idx = np.where(importances >= thr)[0]
                self.logger.info(
                    f"[FeatureMLTrainer] Keeping {idx.size} features "
                    f"(importance >= median {thr:.5f}) out of {importances.shape[0]}"
                )

            self.selected_feature_indices = np.sort(idx)

            X_train = X_train[:, self.selected_feature_indices]
            if len(X_val) > 0:
                X_val = X_val[:, self.selected_feature_indices]
            if len(X_test) > 0:
                X_test = X_test[:, self.selected_feature_indices]
        else:
            self.selected_feature_indices = None

        # ----- 4) Optional PCA -----
        if getattr(self.cfg, "ml_use_pca", False):
            var_ratio = max(0.0, min(1.0, float(getattr(self.cfg, "ml_pca_var_ratio", 0.99))))
            self.logger.info(f"[FeatureMLTrainer] Applying PCA with target variance ratio={var_ratio}")
            self.pca = PCA(n_components=var_ratio, svd_solver="full", random_state=self.cfg.seed)
            self.pca.fit(X_train)
            X_train = self.pca.transform(X_train)
            if len(X_val) > 0:
                X_val = self.pca.transform(X_val)
            if len(X_test) > 0:
                X_test = self.pca.transform(X_test)
            self.logger.info(
                f"[FeatureMLTrainer] PCA output dim={X_train.shape[1]} "
                f"(components={self.pca.n_components_})"
            )
        else:
            self.pca = None

        self.logger.info(
            f"[FeatureMLTrainer] Final ML features: "
            f"X_train={X_train.shape}, "
            f"X_val={X_val.shape if isinstance(X_val, np.ndarray) else X_val}, "
            f"X_test={X_test.shape if isinstance(X_test, np.ndarray) else X_test}"
        )

        # ----- 5) Create and train ML model (with optional hyperparam search) -----
        ml_model_type = getattr(self.cfg, "ml_model_type", "svm_rbf")
        use_search = getattr(self.cfg, "ml_use_hyperparam_search", False)

        if use_search and isinstance(X_val, np.ndarray) and len(X_val) > 0:
            self.logger.info(
                f"[FeatureMLTrainer] Running hyperparameter search for ML model type='{ml_model_type}'"
            )
            best_model, best_params = self._hyperparam_search(
                ml_model_type, X_train, y_train, X_val, y_val
            )
            if best_model is not None:
                self.ml_model = best_model
                self.logger.info(
                    f"[FeatureMLTrainer] Best params for '{ml_model_type}': {best_params}"
                )
            else:
                self.logger.warning(
                    "[FeatureMLTrainer] Hyperparam search returned no model, "
                    "falling back to default configuration"
                )
                self.ml_model = self._create_ml_model(ml_model_type)
                self.ml_model.fit(X_train, y_train)
        else:
            if use_search:
                self.logger.warning(
                    "[FeatureMLTrainer] Hyperparam search requested but no validation "
                    "data available. Falling back to default configuration."
                )
            self.ml_model = self._create_ml_model(ml_model_type)
            self.logger.info(f"[FeatureMLTrainer] Training ML model: {ml_model_type}")
            self.ml_model.fit(X_train, y_train)

        # ----- 6) Save normalization and model -----
        norm_path = self.output_dir / "ml_features_normalization.npz"
        np.savez_compressed(
            norm_path,
            mean=self.feature_mean,
            std=self.feature_std,
            class_ids=np.array(class_ids, dtype=np.int32),
            selected_indices=(
                self.selected_feature_indices
                if self.selected_feature_indices is not None
                else np.array([], dtype=np.int64)
            ),
            pca_components=(
                self.pca.components_ if self.pca is not None else np.array([], dtype=np.float32)
            ),
            pca_mean=(
                self.pca.mean_ if self.pca is not None else np.array([], dtype=np.float32)
            ),
            pca_explained_variance_ratio=(
                self.pca.explained_variance_ratio_
                if self.pca is not None
                else np.array([], dtype=np.float32)
            ),
        )
        self.logger.info(f"[FeatureMLTrainer] Feature normalization saved: {norm_path}")

        model_path = self.output_dir / "ml_model.joblib"
        joblib.dump(self.ml_model, model_path)
        self.logger.info(f"[FeatureMLTrainer] ML model saved: {model_path}")

        # For compatibility with other parts of the pipeline
        self.class_ids = class_ids
        self.class_names = class_names

        # ----- 7) Evaluation helper -----
        def eval_split(X: np.ndarray, y: np.ndarray, split_name: str) -> Optional[Dict]:
            if not isinstance(X, np.ndarray) or X.size == 0:
                return None

            # Predict probabilities if available
            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(X)
            else:
                # Fall back to pseudo-probabilities from decision function
                if hasattr(self.ml_model, "decision_function"):
                    dec = self.ml_model.decision_function(X)  # (N, C) or (N,)
                    if dec.ndim == 1:
                        # Binary case -> expand to 2 classes
                        dec = np.stack([-dec, dec], axis=1)
                    # Softmax over decision values
                    exp_dec = np.exp(dec - dec.max(axis=1, keepdims=True))
                    probs = exp_dec / exp_dec.sum(axis=1, keepdims=True)
                else:
                    probs = None

            y_pred = self.ml_model.predict(X)
            acc = accuracy_score(y, y_pred)
            f1_macro = f1_score(y, y_pred, average="macro")
            report = classification_report(
                y, y_pred, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y, y_pred, labels=np.arange(num_classes))

            if self.visualizer is not None:
                class_labels = [class_names[gid] for gid in class_ids]
                if not np.all(np.isfinite(probs)):
                    self.logger.warning(
                        f"[eval_loader:{split_name}] ROC не построен: "
                        f"probs содержит NaN/Inf "
                        f"(min={np.nanmin(probs)}, max={np.nanmax(probs)})"
                    )
                self.visualizer.plot_confusion_matrix(
                    cm,
                    class_labels,
                    normalize=True,
                    filename=f"cm_{split_name}.png",
                )
                self.visualizer.plot_per_class_f1(
                    report,
                    class_labels,
                    filename=f"f1_{split_name}.png",
                )
                if probs is not None and num_classes >= 2:
                    self.visualizer.plot_roc_ovr(
                        y,
                        probs,
                        class_labels,
                        filename=f"roc_{split_name}.png",
                    )

            return {
                "accuracy": float(acc),
                "f1_macro": float(f1_macro),
                "report": report,
                "confusion_matrix": cm.tolist(),
            }

        results = {
            "class_ids": class_ids,
            "class_names": class_names,
            "val": eval_split(X_val, y_val, "val") if len(X_val) > 0 else None,
            "test": eval_split(X_test, y_test, "test") if len(X_test) > 0 else None,
        }

        results_path = self.output_dir / "ml_classification_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        self.logger.info(f"[FeatureMLTrainer] Classification results saved: {results_path}")

        return results


    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate ML model on numpy arrays (X: (N, T, C) raw windows).

        The evaluation pipeline mirrors the training pipeline:
        - feature extraction (HandcraftedFeatureExtractor / PowerfulFeatureExtractor)
        - standardization
        - optional feature selection
        - optional PCA
        - ML prediction
        """
        assert self.ml_model is not None, "ML model is not trained/loaded"
        assert self.feature_mean is not None and self.feature_std is not None, \
            "Feature normalization stats missing"
        assert self.class_ids is not None and self.class_names is not None, \
            "Class info missing"

        feat_set = getattr(self.cfg, "handcrafted_feature_set", "emg_td")
        if self.feature_extractor is None:
            if feat_set == "powerful":
                self.feature_extractor = PowerfulFeatureExtractor(
                    sampling_rate=2000,
                    logger=self.logger,
                    feature_set="powerful",
                    n_jobs=-1,
                    # use_torch=False,
                    use_torch=True,
                    device='cuda'
                )
            else:
                self.feature_extractor = HandcraftedFeatureExtractor(
                    sampling_rate=None,
                    logger=self.logger,
                    feature_set=feat_set,
                )

        if X.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

        # 1) Features
        X_feats = self.feature_extractor.transform(X)  # (N, F_raw)

        # 2) Standardization
        Xs = (X_feats - self.feature_mean[None, :]) / self.feature_std[None, :]

        # 3) Optional feature selection
        if self.selected_feature_indices is not None and self.selected_feature_indices.size > 0:
            Xs = Xs[:, self.selected_feature_indices]

        # 4) Optional PCA
        if self.pca is not None:
            Xs = self.pca.transform(Xs)

        # 5) Predict
        y_pred = self.ml_model.predict(Xs)
        if hasattr(self.ml_model, "predict_proba"):
            probs = self.ml_model.predict_proba(Xs)
        else:
            probs = None

        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro")
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=np.arange(len(self.class_ids)))

        if visualize and self.visualizer is not None:
            class_labels = [self.class_names[gid] for gid in self.class_ids]
            self.visualizer.plot_confusion_matrix(
                cm,
                class_labels,
                normalize=True,
                filename=f"cm_{split_name}.png",
            )

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "probs": probs,
            "y_pred": y_pred,
        }