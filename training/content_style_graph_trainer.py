"""
Trainer for Content-Style Graph Network (Experiment 41).

Extends WindowClassifierTrainer with:
- Subject-aware training using real subject labels
- 4-loss training: gesture CE + fusion CE + subject CE + MI minimization
- Rich post-training visualizations (t-SNE, adjacency, disentanglement probes)
- Inference uses z_content only (subject-invariant)
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from training.trainer import WindowClassifierTrainer, WindowDataset, get_worker_init_fn, seed_everything
from training.disentangled_trainer import DisentangledWindowDataset
from models.content_style_graph import (
    ContentStyleGraphNet,
    distance_correlation_loss,
    orthogonality_loss,
)

# Lazy matplotlib import to avoid backend issues on headless servers
_MPL_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    _MPL_AVAILABLE = False


class ContentStyleGraphTrainer(WindowClassifierTrainer):
    """
    Trainer for ContentStyleGraphNet: content-style disentanglement with
    graph-based style encoding.

    Expects splits dict to contain:
        "train_subject_labels": Dict[int, np.ndarray]
        "num_train_subjects": int
    (Same contract as DisentangledTrainer)

    Loss = L_gesture + lambda_fusion*L_fusion + alpha*L_subject + beta(t)*L_MI
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        # Architecture dims
        content_dim: int = 128,
        style_dim: int = 64,
        d_node: int = 64,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        gru_hidden_content: int = 128,
        gru_hidden_style: int = 64,
        # Loss weights
        alpha: float = 0.5,
        beta: float = 0.1,
        lambda_fusion: float = 0.5,
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",
        # Eval mode
        use_fusion_at_eval: bool = False,
    ):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.d_node = d_node
        self.n_heads = n_heads
        self.n_gat_layers = n_gat_layers
        self.gru_hidden_content = gru_hidden_content
        self.gru_hidden_style = gru_hidden_style
        self.alpha = alpha
        self.beta = beta
        self.lambda_fusion = lambda_fusion
        self.beta_anneal_epochs = beta_anneal_epochs
        self.mi_loss_type = mi_loss_type
        self.use_fusion_at_eval = use_fusion_at_eval

    # ──────────────────── Subject label helper ─────────────────────────

    def _build_subject_labels_array(
        self,
        subject_labels_dict: Dict[int, np.ndarray],
        class_ids: List[int],
    ) -> np.ndarray:
        """Build flat subject label array aligned with _prepare_splits_arrays output."""
        parts = []
        for gid in class_ids:
            if gid in subject_labels_dict:
                parts.append(subject_labels_dict[gid])
        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    # ──────────────────── Main training loop ───────────────────────────

    def fit(self, splits: Dict) -> Dict:
        """Train ContentStyleGraphNet with 4-loss formulation."""
        seed_everything(self.cfg.seed)

        # 1. Prepare arrays (standard)
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # 2. Extract subject labels
        if "train_subject_labels" not in splits:
            raise ValueError(
                "ContentStyleGraphTrainer requires 'train_subject_labels' in splits."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        assert len(y_subject_train) == len(y_train), (
            f"Subject labels ({len(y_subject_train)}) != gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Subject labels: {num_train_subjects} subjects, "
            f"distribution: {np.bincount(y_subject_train).tolist()}"
        )

        # 3. Transpose (N, T, C) → (N, C, T) if needed
        if X_train.ndim == 3:
            N, dim1, dim2 = X_train.shape
            if dim1 > dim2:
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # 4. Channel standardization
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardization.")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # 5. Create model
        device = self.cfg.device
        model = ContentStyleGraphNet(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            d_node=self.d_node,
            n_heads=self.n_heads,
            n_gat_layers=self.n_gat_layers,
            gru_hidden_content=self.gru_hidden_content,
            gru_hidden_style=self.gru_hidden_style,
            dropout=self.cfg.dropout,
            use_fusion_at_eval=self.use_fusion_at_eval,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"ContentStyleGraphNet: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, content_dim={self.content_dim}, "
            f"style_dim={self.style_dim}, d_node={self.d_node}, "
            f"params={total_params:,}"
        )

        # 6. Datasets
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        dl_train = DataLoader(
            ds_train, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        dl_val = DataLoader(
            ds_val, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        ) if ds_val else None
        dl_test = DataLoader(
            ds_test, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        ) if ds_test else None

        # 7. Loss functions
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            gesture_criterion = nn.CrossEntropyLoss()
        subject_criterion = nn.CrossEntropyLoss()

        # 8. Optimizer + scheduler
        optimizer = optim.Adam(
            model.parameters(), lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # 9. Training loop
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "gesture_loss": [], "fusion_loss": [], "subject_loss": [], "mi_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))

            ep = {k: 0.0 for k in ["total", "gesture", "fusion", "subject", "mi"]}
            ep_correct, ep_total = 0, 0

            for windows, gesture_labels, subject_labels in dl_train:
                windows = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()
                outputs = model(windows, return_all=True)

                L_gesture = gesture_criterion(outputs["gesture_logits"], gesture_labels)
                L_fusion = gesture_criterion(outputs["fusion_logits"], gesture_labels)
                L_subject = subject_criterion(outputs["subject_logits"], subject_labels)

                if self.mi_loss_type == "distance_correlation":
                    L_MI = distance_correlation_loss(outputs["z_content"], outputs["z_style"])
                elif self.mi_loss_type == "orthogonal":
                    L_MI = orthogonality_loss(outputs["z_content"], outputs["z_style"])
                elif self.mi_loss_type == "both":
                    L_MI = (
                        distance_correlation_loss(outputs["z_content"], outputs["z_style"])
                        + 0.1 * orthogonality_loss(outputs["z_content"], outputs["z_style"])
                    )
                else:
                    L_MI = distance_correlation_loss(outputs["z_content"], outputs["z_style"])

                total_loss = (
                    L_gesture
                    + self.lambda_fusion * L_fusion
                    + self.alpha * L_subject
                    + current_beta * L_MI
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep["total"] += total_loss.item() * bs
                ep["gesture"] += L_gesture.item() * bs
                ep["fusion"] += L_fusion.item() * bs
                ep["subject"] += L_subject.item() * bs
                ep["mi"] += L_MI.item() * bs
                preds = outputs["gesture_logits"].argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total += bs

            train_loss = ep["total"] / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)

            # Validation (gesture classification only)
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)  # eval → gesture_logits only
                        loss = gesture_criterion(logits, yb)
                        val_loss_sum += loss.item() * yb.size(0)
                        val_correct += (logits.argmax(1) == yb).sum().item()
                        val_total += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            for key in ["gesture", "fusion", "subject", "mi"]:
                history[f"{key}_loss"].append(ep[key] / max(1, ep_total))

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} (gest={history['gesture_loss'][-1]:.4f}, "
                f"fus={history['fusion_loss'][-1]:.4f}, "
                f"subj={history['subject_loss'][-1]:.4f}, MI={history['mi_loss'][-1]:.4f}), "
                f"acc={train_acc:.3f} | Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"beta={current_beta:.4f}"
            )

            # Early stopping on val gesture loss
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
            model.to(device)

        # Store trainer state (CRITICAL for evaluate_numpy)
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size

        # Save history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        # ── Visualizations ──
        if _MPL_AVAILABLE:
            self._plot_extended_training_curves(history)
            self._collect_and_visualize(
                model, dl_train, dl_test, device,
                y_train, y_subject_train, class_ids, class_names,
                num_train_subjects,
            )

        # Evaluate on val/test
        results = {"class_ids": class_ids, "class_names": class_names}

        def eval_loader(dloader, split_name):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    xb = xb.to(device)
                    logits = model(xb)
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

        # Save checkpoint
        torch.save({
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "num_subjects": num_train_subjects,
            "class_ids": class_ids,
            "mean": mean_c,
            "std": std_c,
            "window_size": window_size,
            "content_dim": self.content_dim,
            "style_dim": self.style_dim,
            "d_node": self.d_node,
            "alpha": self.alpha,
            "beta": self.beta,
            "lambda_fusion": self.lambda_fusion,
            "training_config": asdict(self.cfg),
        }, self.output_dir / "content_style_graph.pt")
        self.logger.info(f"Model saved: {self.output_dir / 'content_style_graph.pt'}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # ──────────────────── evaluate_numpy ───────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """Evaluate using gesture classification from z_content (subject-invariant)."""
        assert self.model is not None, "Model is not trained/loaded"
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None and self.class_names is not None

        X_input = X.copy()
        if X_input.ndim == 3:
            N, dim1, dim2 = X_input.shape
            if dim1 > dim2:
                X_input = np.transpose(X_input, (0, 2, 1))

        Xs = self._apply_standardization(X_input, self.mean_c, self.std_c)

        ds = WindowDataset(Xs, y)
        dl = DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )

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

    # ══════════════════════ VISUALIZATIONS ══════════════════════════════

    def _plot_extended_training_curves(self, history: Dict):
        """5 loss curves + train/val accuracy over epochs."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        epochs = range(1, len(history["train_loss"]) + 1)

        # Top: losses
        ax = axes[0]
        ax.plot(epochs, history["train_loss"], "k-", linewidth=2, label="Total")
        ax.plot(epochs, history["gesture_loss"], "b--", label="Gesture")
        ax.plot(epochs, history["fusion_loss"], "c--", label="Fusion")
        ax.plot(epochs, history["subject_loss"], "r--", label="Subject")
        ax.plot(epochs, history["mi_loss"], "m--", label="MI")
        if any(not np.isnan(v) for v in history["val_loss"]):
            ax.plot(epochs, history["val_loss"], "g-", linewidth=1.5, label="Val (gesture)")
        ax.set_ylabel("Loss")
        ax.set_title("Content-Style Graph Network — Training Losses")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom: accuracy
        ax = axes[1]
        ax.plot(epochs, history["train_acc"], "b-", label="Train Acc")
        if any(not np.isnan(v) for v in history["val_acc"]):
            ax.plot(epochs, history["val_acc"], "g-", label="Val Acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "training_curves_extended.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved: {path}")

    def _collect_and_visualize(
        self, model, dl_train, dl_test, device,
        y_train, y_subject_train, class_ids, class_names,
        num_train_subjects,
    ):
        """Collect latent representations and produce all visualizations."""
        model.eval()

        # Collect from training data (subsample for speed)
        max_samples = 2000
        z_content_list, z_style_list = [], []
        adj_list, cw_list = [], []
        y_gesture_list, y_subject_list = [], []
        fusion_gate_list, fusion_gesture_list = [], []
        collected = 0

        with torch.no_grad():
            for xb, yb_g, yb_s in dl_train:
                if collected >= max_samples:
                    break
                xb = xb.to(device)
                out = model(xb, return_all=True)

                n = min(xb.size(0), max_samples - collected)
                z_content_list.append(out["z_content"][:n].cpu().numpy())
                z_style_list.append(out["z_style"][:n].cpu().numpy())
                adj_list.append(out["adjacency"][:n].cpu().numpy())
                cw_list.append(out["channel_weights"][:n].cpu().numpy())
                y_gesture_list.append(yb_g[:n].numpy())
                y_subject_list.append(yb_s[:n].numpy())

                # Collect fusion gate values
                cat = torch.cat([out["z_content"][:n], out["z_style"][:n]], dim=-1)
                gate_vals = model.fusion.gate_net(cat).cpu().numpy()
                fusion_gate_list.append(gate_vals)
                fusion_gesture_list.append(yb_g[:n].numpy())

                collected += n

        z_content = np.concatenate(z_content_list, axis=0)
        z_style = np.concatenate(z_style_list, axis=0)
        adj_all = np.concatenate(adj_list, axis=0)
        cw_all = np.concatenate(cw_list, axis=0)
        y_gesture = np.concatenate(y_gesture_list, axis=0)
        y_subject = np.concatenate(y_subject_list, axis=0)
        fusion_gates = np.concatenate(fusion_gate_list, axis=0)
        fusion_gestures = np.concatenate(fusion_gesture_list, axis=0)

        # Generate all visualizations
        gesture_names = [class_names.get(gid, f"G{gid}") for gid in class_ids]
        subject_names = [f"Subj_{i}" for i in range(num_train_subjects)]

        # 1-4: t-SNE plots
        self._plot_tsne_latent_space(
            z_content, y_gesture, gesture_names,
            "z_content colored by gesture", "tsne_content_by_gesture.png",
        )
        self._plot_tsne_latent_space(
            z_content, y_subject, subject_names,
            "z_content colored by subject (should be mixed)",
            "tsne_content_by_subject.png",
        )
        self._plot_tsne_latent_space(
            z_style, y_subject, subject_names,
            "z_style colored by subject", "tsne_style_by_subject.png",
        )
        self._plot_tsne_latent_space(
            z_style, y_gesture, gesture_names,
            "z_style colored by gesture (should be mixed)",
            "tsne_style_by_gesture.png",
        )

        # 5: Adjacency heatmap
        adj_mean = adj_all.mean(axis=0)  # (C, C)
        self._plot_adjacency_heatmap(adj_mean)

        # 6: Channel importance
        cw_mean = cw_all.mean(axis=0).squeeze()  # (C,)
        self._plot_channel_importance(cw_mean)

        # 7: Disentanglement quality scores
        self._compute_and_save_disentanglement_scores(
            z_content, z_style, y_gesture, y_subject,
            num_train_subjects, len(class_ids),
        )

        # 8: Content-style correlation
        self._plot_content_style_correlation(z_content, z_style)

        # 9: Edge weight distribution
        self._plot_edge_weight_distribution(adj_all)

        # 10: Fusion gate by gesture
        self._plot_fusion_gate_by_gesture(
            fusion_gates, fusion_gestures, class_ids, class_names,
        )

    # ──────────────────── t-SNE ────────────────────────────────────────

    def _plot_tsne_latent_space(
        self, z: np.ndarray, labels: np.ndarray, label_names: list,
        title: str, filename: str,
    ):
        """t-SNE visualization of a latent space."""
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            self.logger.warning("sklearn.manifold.TSNE not available, skipping t-SNE")
            return

        n = min(len(z), 2000)
        if n < 10:
            return
        idx = np.random.RandomState(42).choice(len(z), n, replace=False)
        z_sub = z[idx]
        labels_sub = labels[idx]

        perplexity = min(30, n // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        emb = tsne.fit_transform(z_sub)

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(labels_sub)
        cmap = plt.cm.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")

        for i, lbl in enumerate(unique_labels):
            mask = labels_sub == lbl
            name = label_names[lbl] if lbl < len(label_names) else f"{lbl}"
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=[cmap(i / max(1, len(unique_labels) - 1))],
                label=name, s=15, alpha=0.6, edgecolors="none",
            )
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc="best", markerscale=2, ncol=2)
        ax.set_xticks([])
        ax.set_yticks([])

        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved: {path}")

    # ──────────────────── Adjacency heatmap ────────────────────────────

    def _plot_adjacency_heatmap(self, adj_mean: np.ndarray):
        """Average adjacency matrix (C x C) as heatmap."""
        C = adj_mean.shape[0]
        ch_labels = [f"CH{i+1}" for i in range(C)]

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(adj_mean, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(ch_labels, fontsize=9)
        ax.set_yticklabels(ch_labels, fontsize=9)
        ax.set_title("Learned Inter-Channel Adjacency (avg over training data)", fontsize=10)

        # Annotate
        for i in range(C):
            for j in range(C):
                ax.text(j, i, f"{adj_mean[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(adj_mean[i, j]) > 0.5 * np.abs(adj_mean).max() else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

        path = self.output_dir / "adjacency_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved: {path}")

    # ──────────────────── Channel importance ───────────────────────────

    def _plot_channel_importance(self, cw_mean: np.ndarray):
        """Bar chart of channel attention weights."""
        C = len(cw_mean)
        ch_labels = [f"CH{i+1}" for i in range(C)]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.viridis(cw_mean / (cw_mean.max() + 1e-8))
        ax.bar(ch_labels, cw_mean, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Attention Gate Weight")
        ax.set_title("Style Branch — Channel Importance (GAT readout gate)")
        ax.set_ylim(0, max(cw_mean.max() * 1.2, 0.01))

        for i, v in enumerate(cw_mean):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        path = self.output_dir / "channel_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved: {path}")

    # ──────────────────── Disentanglement probes ───────────────────────

    def _compute_and_save_disentanglement_scores(
        self, z_content, z_style, y_gesture, y_subject,
        num_subjects, num_gestures,
    ):
        """
        Linear probe evaluation of disentanglement quality.
        - z_content → predict subject: should be near random (1/num_subjects)
        - z_style → predict gesture: should be near random (1/num_gestures)
        """
        scores = {}

        # Probe 1: z_content → subject (should fail)
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs",
                                     multi_class="multinomial")
            clf.fit(z_content, y_subject)
            content_subject_acc = float(clf.score(z_content, y_subject))
            content_subject_random = 1.0 / max(1, num_subjects)
            scores["content_to_subject_accuracy"] = content_subject_acc
            scores["content_to_subject_random_baseline"] = content_subject_random
            scores["content_subject_invariance"] = max(0.0, 1.0 - (content_subject_acc - content_subject_random) / (1.0 - content_subject_random + 1e-8))
            self.logger.info(
                f"Disentanglement probe: z_content → subject = {content_subject_acc:.3f} "
                f"(random={content_subject_random:.3f}, invariance={scores['content_subject_invariance']:.3f})"
            )
        except Exception as e:
            self.logger.warning(f"Content→subject probe failed: {e}")
            scores["content_to_subject_accuracy"] = None

        # Probe 2: z_style → gesture (should fail)
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs",
                                     multi_class="multinomial")
            clf.fit(z_style, y_gesture)
            style_gesture_acc = float(clf.score(z_style, y_gesture))
            style_gesture_random = 1.0 / max(1, num_gestures)
            scores["style_to_gesture_accuracy"] = style_gesture_acc
            scores["style_to_gesture_random_baseline"] = style_gesture_random
            scores["style_gesture_invariance"] = max(0.0, 1.0 - (style_gesture_acc - style_gesture_random) / (1.0 - style_gesture_random + 1e-8))
            self.logger.info(
                f"Disentanglement probe: z_style → gesture = {style_gesture_acc:.3f} "
                f"(random={style_gesture_random:.3f}, invariance={scores['style_gesture_invariance']:.3f})"
            )
        except Exception as e:
            self.logger.warning(f"Style→gesture probe failed: {e}")
            scores["style_to_gesture_accuracy"] = None

        path = self.output_dir / "disentanglement_scores.json"
        with open(path, "w") as f:
            json.dump(scores, f, indent=4)
        self.logger.info(f"Saved: {path}")

    # ──────────────────── Content-style correlation ────────────────────

    def _plot_content_style_correlation(self, z_content: np.ndarray, z_style: np.ndarray):
        """Pearson correlation heatmap between content and style dimensions."""
        # Subsample dimensions for readability
        cd = z_content.shape[1]
        sd = z_style.shape[1]

        # Compute correlation matrix
        corr = np.zeros((cd, sd))
        for i in range(cd):
            for j in range(sd):
                c = np.corrcoef(z_content[:, i], z_style[:, j])[0, 1]
                corr[i, j] = c if not np.isnan(c) else 0.0

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xlabel(f"z_style dimensions (0..{sd-1})")
        ax.set_ylabel(f"z_content dimensions (0..{cd-1})")
        ax.set_title(
            f"Content-Style Correlation (mean |r| = {np.abs(corr).mean():.4f})",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, shrink=0.8)

        path = self.output_dir / "content_style_correlation.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved: {path}")

    # ──────────────────── Edge weight distribution ─────────────────────

    def _plot_edge_weight_distribution(self, adj_all: np.ndarray):
        """Histogram of adjacency matrix values."""
        # Flatten upper triangle (exclude diagonal)
        C = adj_all.shape[1]
        triu_idx = np.triu_indices(C, k=1)
        values = adj_all[:, triu_idx[0], triu_idx[1]].flatten()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values, bins=50, color="steelblue", edgecolor="black", linewidth=0.3, alpha=0.8)
        ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean={values.mean():.3f}")
        ax.axvline(np.median(values), color="orange", linestyle="--", label=f"Median={np.median(values):.3f}")
        ax.set_xlabel("Edge Weight")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Inter-Channel Edge Weights")
        ax.legend()

        path = self.output_dir / "edge_weight_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved: {path}")

    # ──────────────────── Fusion gate by gesture ───────────────────────

    def _plot_fusion_gate_by_gesture(
        self, fusion_gates: np.ndarray, y_gesture: np.ndarray,
        class_ids: list, class_names: dict,
    ):
        """
        Show how fusion gate weights content vs style per gesture class.

        fusion_gates: (N, content_dim + style_dim) — gate values after sigmoid
        Split into content portion and style portion, compute mean per gesture.
        """
        content_dim = self.content_dim
        style_dim = self.style_dim

        content_gate = fusion_gates[:, :content_dim].mean(axis=1)  # (N,) mean gate for content dims
        style_gate = fusion_gates[:, content_dim:].mean(axis=1)    # (N,) mean gate for style dims

        gesture_names = []
        content_means = []
        style_means = []

        for i, gid in enumerate(class_ids):
            mask = y_gesture == i
            if mask.sum() == 0:
                continue
            gesture_names.append(class_names.get(gid, f"G{gid}"))
            content_means.append(float(content_gate[mask].mean()))
            style_means.append(float(style_gate[mask].mean()))

        if not gesture_names:
            return

        x = np.arange(len(gesture_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, content_means, width, label="Content gate", color="steelblue")
        ax.bar(x + width / 2, style_means, width, label="Style gate", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(gesture_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Gate Activation")
        ax.set_title("Fusion Gate: Content vs Style Contribution per Gesture")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        path = self.output_dir / "fusion_gate_by_gesture.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved: {path}")
