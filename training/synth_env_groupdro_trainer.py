"""
Synthetic Environment Expansion + Soft GroupDRO Trainer (Experiment 103).

Extends DisentangledTrainer by replacing the plain ERM gesture loss with a
GroupDRO objective over a mixture of real and style-interpolated (virtual)
environments:

    Real environments   : one per training subject (base path, no FiLM)
    Virtual environments: one per unordered pair {i, j} of subjects (FiLM path)

For N training subjects, the environment count is:
    N_real    = N
    N_virtual = C(N, 2)  =  N·(N-1)/2
    N_total   = N + C(N, 2)

Example (N=4, LOSO fold with 5 subjects):
    Real:    env 0..3  (subjects 0..3)
    Virtual: env 4..9  (pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    Total:   10 environments

Virtual environment loss for pair (i, j)
-----------------------------------------
For each batch containing samples from BOTH subject i and j:
  1. Take samples from subject i  → z_content_i, z_style_i, y_i
     Take samples from subject j  → z_content_j, z_style_j, y_j
  2. For each sample k from subject i:
       pick random partner m from subject j;
       lam_k ~ Beta(mix_alpha, mix_alpha)
       z_style_mix_k = lam_k · z_style_k + (1 - lam_k) · z_style_j[m].detach()
  3. Symmetric for samples from subject j (partner picked from i).
  4. Apply FiLM:  z_content_film = model.film(z_content, z_style_mix)
  5. Gesture logits via shared classifier:
       logits_virt = model.gesture_classifier(z_content_film)
  6. L_{i,j} = gesture_criterion(logits_virt, y)   averaged over both groups

GroupDRO update (mirror descent, Sagawa et al. 2020)
-----------------------------------------------------
For every environment e (real or virtual):
    q_e ← q_e · exp(η · L_e_batch)
Normalize: q ← q / Σ q_e

Environments absent in a batch have L_e_batch = 0:
    exp(η · 0) = 1  →  weight unchanged  (correct GroupDRO semantics)

Total training objective
------------------------
L_total = Σ_e q_e · L_e        (DRO gesture loss over all N_total envs)
         + α · L_subject        (subject classifier on z_style)
         + β(t) · L_MI          (MI minimisation, annealed)

LOSO compliance guarantees
--------------------------
- All z_style values used for mixing originate from training-batch samples ONLY.
  No test-subject z_style is ever included.
- model.eval() returns gesture_logits_base = GestureClassifier(z_content) with
  no FiLM involvement.  Inference requires no subject information.
- Group weights are never updated from test-subject losses.
- Channel normalisation statistics are computed from X_train only.
- Model selection (early stopping) uses val_loss from the train-subject val split.
- Test-subject windows are evaluated ONCE at the end for final metrics only.

References
----------
Sagawa et al. (2020). "Distributionally Robust Neural Networks."  (GroupDRO)
Zhou et al.  (2021). "Domain Generalization with MixStyle."        (style mixing)
exp_57 — GroupDRO + Disentanglement (N=4 real envs, no virtual envs)
exp_60 — MixStyle + FiLM (style mixing, ERM objective)
"""

import json
import logging
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from models.disentangled_cnn_gru import distance_correlation_loss, orthogonality_loss
from models.synth_env_groupdro_emg import SynthEnvGroupDROModel
from training.disentangled_trainer import DisentangledTrainer, DisentangledWindowDataset
from training.trainer import WindowDataset, get_worker_init_fn, seed_everything


class SynthEnvGroupDROTrainer(DisentangledTrainer):
    """
    Synthetic Environment Expansion + GroupDRO trainer.

    Compared to GroupDRODisentangledTrainer (exp_57), adds C(N, 2) virtual
    environments created via per-batch style interpolation.  DRO operates on
    all N + C(N, 2) environments jointly with a reduced step size.

    Parameters
    ----------
    dro_eta : float
        Exponentiated gradient step size for group-weight updates.
        Smaller than exp_57 (0.01) because more environments → noisier gradients.
        Default: 0.005.
    mix_alpha : float
        Beta distribution parameter for style interpolation: λ ~ Beta(α, α).
        mix_alpha=0.4 gives moderately strong mixing (same as exp_60).
    """

    def __init__(
        self,
        train_cfg,
        logger: logging.Logger,
        output_dir: Path,
        visualizer=None,
        content_dim: int = 128,
        style_dim: int = 64,
        alpha: float = 0.5,
        beta: float = 0.1,
        beta_anneal_epochs: int = 10,
        mi_loss_type: str = "distance_correlation",
        dro_eta: float = 0.005,
        mix_alpha: float = 0.4,
    ):
        super().__init__(
            train_cfg,
            logger,
            output_dir,
            visualizer,
            content_dim,
            style_dim,
            alpha,
            beta,
            beta_anneal_epochs,
            mi_loss_type,
        )
        self.dro_eta = dro_eta
        self.mix_alpha = mix_alpha
        # Populated after fit() — available for experiment-level logging.
        self.final_group_weights: Optional[List[float]] = None
        self.env_descriptions: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(self, splits: Dict) -> Dict:  # noqa: C901
        """
        Train with Synthetic Environment Expansion + GroupDRO.

        Args:
            splits: dict produced by ``_build_splits_with_subject_labels()``
                Must contain:
                    "train"                : Dict[gesture_id, np.ndarray]
                    "val"                  : Dict[gesture_id, np.ndarray]
                    "test"                 : Dict[gesture_id, np.ndarray]
                    "train_subject_labels" : Dict[gesture_id, np.ndarray of int64]
                    "num_train_subjects"   : int

        Returns:
            dict with val/test accuracy, F1, confusion matrices.
        """
        seed_everything(self.cfg.seed)

        # ── 1. Split arrays ───────────────────────────────────────────────
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        # ── 2. Subject labels ─────────────────────────────────────────────
        if "train_subject_labels" not in splits:
            raise ValueError(
                "SynthEnvGroupDROTrainer requires 'train_subject_labels' in splits. "
                "Use the experiment file that injects subject provenance."
            )
        y_subject_train = self._build_subject_labels_array(
            splits["train_subject_labels"], class_ids
        )
        num_train_subjects = splits["num_train_subjects"]

        assert len(y_subject_train) == len(y_train), (
            f"Subject labels ({len(y_subject_train)}) must match "
            f"gesture labels ({len(y_train)})"
        )
        self.logger.info(
            f"Training: {num_train_subjects} subjects | "
            f"subject distribution: {np.bincount(y_subject_train).tolist()}"
        )

        # ── 3. Transpose (N, T, C) → (N, C, T) if needed ─────────────────
        if X_train.ndim == 3:
            _, dim1, dim2 = X_train.shape
            if dim1 > dim2:  # T > C → needs transpose
                X_train = np.transpose(X_train, (0, 2, 1))
                if len(X_val) > 0:
                    X_val = np.transpose(X_val, (0, 2, 1))
                if len(X_test) > 0:
                    X_test = np.transpose(X_test, (0, 2, 1))
                self.logger.info(f"Transposed to (N, C, T): X_train={X_train.shape}")

        in_channels = X_train.shape[1]
        window_size = X_train.shape[2]
        num_classes = len(class_ids)

        # ── 4. Per-channel standardisation (train stats only — LOSO clean) ─
        mean_c, std_c = self._compute_channel_standardization(X_train)
        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)
        if len(X_test) > 0:
            X_test = self._apply_standardization(X_test, mean_c, std_c)
        self.logger.info("Applied per-channel standardisation (train stats only).")

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c,
            std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 5. Enumerate environments ──────────────────────────────────────
        # Real envs: index 0 … N-1  (one per training subject)
        # Virtual envs: index N … N+C(N,2)-1  (one per unordered pair {i,j})
        real_env_count = num_train_subjects
        pairs: List[tuple] = list(combinations(range(num_train_subjects), 2))
        virtual_env_count = len(pairs)
        total_envs = real_env_count + virtual_env_count
        pair_to_venv_idx: Dict[tuple, int] = {p: k for k, p in enumerate(pairs)}

        env_descriptions = (
            [f"real_s{s}" for s in range(real_env_count)]
            + [f"virt_({i},{j})" for (i, j) in pairs]
        )
        self.logger.info(
            f"Environments: {real_env_count} real + {virtual_env_count} virtual "
            f"= {total_envs} total | dro_eta={self.dro_eta} | mix_alpha={self.mix_alpha}"
        )

        # ── 6. Model ──────────────────────────────────────────────────────
        device = self.cfg.device
        model = SynthEnvGroupDROModel(
            in_channels=in_channels,
            num_gestures=num_classes,
            num_subjects=num_train_subjects,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            dropout=self.cfg.dropout,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"SynthEnvGroupDROModel: in_ch={in_channels}, gestures={num_classes}, "
            f"subjects={num_train_subjects}, content_dim={self.content_dim}, "
            f"style_dim={self.style_dim}, params={total_params:,}"
        )

        # ── 7. Datasets & DataLoaders ─────────────────────────────────────
        ds_train = DisentangledWindowDataset(X_train, y_train, y_subject_train)
        ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
        ds_test = WindowDataset(X_test, y_test) if len(X_test) > 0 else None

        worker_init_fn = get_worker_init_fn(self.cfg.seed)
        _dl_kw = dict(
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            **_dl_kw,
        )
        dl_val = (
            DataLoader(ds_val, batch_size=self.cfg.batch_size, shuffle=False, **_dl_kw)
            if ds_val else None
        )
        dl_test = (
            DataLoader(ds_test, batch_size=self.cfg.batch_size, shuffle=False, **_dl_kw)
            if ds_test else None
        )

        # ── 8. Loss functions ─────────────────────────────────────────────
        if self.cfg.use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = class_counts.sum() / (class_counts + 1e-8)
            cw = cw / cw.mean()
            weight_tensor = torch.from_numpy(cw).float().to(device)
            self.logger.info(f"Gesture class weights: {cw.round(3).tolist()}")
            gesture_criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction="mean")
        else:
            gesture_criterion = nn.CrossEntropyLoss(reduction="mean")

        subject_criterion = nn.CrossEntropyLoss()

        # ── 9. Optimizer & scheduler ──────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # ── 10. GroupDRO group weights (uniform init) ─────────────────────
        # One weight per environment; maintained across entire training.
        # Shape: (total_envs,)  — first N are real, rest are virtual.
        group_weights = torch.ones(total_envs, device=device) / total_envs

        # Beta distribution for style mixing — reused each batch.
        beta_conc = torch.tensor(self.mix_alpha, dtype=torch.float32, device=device)
        beta_dist = torch.distributions.Beta(beta_conc, beta_conc)

        # ── 11. Training loop ─────────────────────────────────────────────
        history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "dro_loss": [], "subject_loss": [], "mi_loss": [],
            "group_weights": [],        # List[List[float]] — (epoch, total_envs)
            "real_env_weights": [],     # List[List[float]] — real envs only for readability
        }
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            current_beta = self.beta * min(1.0, epoch / max(1, self.beta_anneal_epochs))

            ep_total_loss = 0.0
            ep_dro_loss = 0.0
            ep_subject_loss = 0.0
            ep_mi_loss = 0.0
            ep_correct = 0
            ep_total = 0

            for windows, gesture_labels, subject_labels in dl_train:
                windows = windows.to(device)
                gesture_labels = gesture_labels.to(device)
                subject_labels = subject_labels.to(device)

                optimizer.zero_grad()

                # ── Forward (training mode → returns dict) ────────────────
                outputs = model(windows)
                z_content = outputs["z_content"]          # (B, content_dim)
                z_style = outputs["z_style"]              # (B, style_dim)
                gest_logits_base = outputs["gesture_logits_base"]  # (B, num_classes)
                subject_logits = outputs["subject_logits"]  # (B, num_subjects)

                # ── Per-environment losses ────────────────────────────────
                batch_env_losses = torch.zeros(total_envs, device=device)

                # Real environments: base path (no FiLM)
                for s in range(num_train_subjects):
                    mask_s = subject_labels == s
                    if mask_s.any():
                        batch_env_losses[s] = gesture_criterion(
                            gest_logits_base[mask_s],
                            gesture_labels[mask_s],
                        )

                # Virtual environments: FiLM-conditioned path
                # Only computed for pairs where BOTH subjects are in the batch.
                subjects_in_batch = set(subject_labels.cpu().tolist())

                for (si, sj), venv_idx in pair_to_venv_idx.items():
                    if si not in subjects_in_batch or sj not in subjects_in_batch:
                        # Absent pair: L_e_batch = 0 → exp(η·0) = 1 → weight unchanged.
                        continue

                    mask_i = subject_labels == si
                    mask_j = subject_labels == sj

                    z_c_i = z_content[mask_i]   # (ni, content_dim)
                    z_c_j = z_content[mask_j]   # (nj, content_dim)
                    z_s_i = z_style[mask_i]      # (ni, style_dim)
                    z_s_j = z_style[mask_j]      # (nj, style_dim)
                    y_i = gesture_labels[mask_i]
                    y_j = gesture_labels[mask_j]

                    ni = z_c_i.size(0)
                    nj = z_c_j.size(0)

                    # ── Style mixing: samples from subject i ──────────────
                    # Each sample from i is paired with a random sample from j.
                    # Gradient flows through z_s_i (anchor); partner is detached.
                    lam_i = beta_dist.sample((ni,)).unsqueeze(1)   # (ni, 1)
                    perm_j = torch.randint(0, nj, (ni,), device=device)
                    z_s_mix_i = (
                        lam_i * z_s_i + (1.0 - lam_i) * z_s_j[perm_j].detach()
                    )   # (ni, style_dim)

                    # ── Style mixing: samples from subject j ──────────────
                    lam_j = beta_dist.sample((nj,)).unsqueeze(1)   # (nj, 1)
                    perm_i = torch.randint(0, ni, (nj,), device=device)
                    z_s_mix_j = (
                        lam_j * z_s_j + (1.0 - lam_j) * z_s_i[perm_i].detach()
                    )   # (nj, style_dim)

                    # ── FiLM conditioning ─────────────────────────────────
                    # model.film and model.gesture_classifier are in the same
                    # computation graph — gradients flow back through them.
                    z_cf_i = model.film(z_c_i, z_s_mix_i)   # (ni, content_dim)
                    z_cf_j = model.film(z_c_j, z_s_mix_j)   # (nj, content_dim)

                    # ── Virtual-env gesture logits ─────────────────────────
                    logits_virt = torch.cat(
                        [
                            model.gesture_classifier(z_cf_i),
                            model.gesture_classifier(z_cf_j),
                        ],
                        dim=0,
                    )
                    y_virt = torch.cat([y_i, y_j], dim=0)

                    batch_env_losses[real_env_count + venv_idx] = gesture_criterion(
                        logits_virt, y_virt
                    )

                # ── GroupDRO weight update (mirror descent, no_grad) ──────
                # Absent-env losses are 0 → exp(η·0)=1 → weight ratio preserved.
                with torch.no_grad():
                    group_weights = group_weights * torch.exp(
                        self.dro_eta * batch_env_losses
                    )
                    group_weights = group_weights / (group_weights.sum() + 1e-12)

                # ── DRO gesture loss ──────────────────────────────────────
                L_dro = (group_weights.detach() * batch_env_losses).sum()

                # ── Auxiliary losses ──────────────────────────────────────
                L_subject = subject_criterion(subject_logits, subject_labels)

                if self.mi_loss_type == "distance_correlation":
                    L_MI = distance_correlation_loss(z_content, z_style)
                elif self.mi_loss_type == "orthogonal":
                    L_MI = orthogonality_loss(z_content, z_style)
                else:  # "both"
                    L_MI = (
                        distance_correlation_loss(z_content, z_style)
                        + 0.1 * orthogonality_loss(z_content, z_style)
                    )

                total_loss = L_dro + self.alpha * L_subject + current_beta * L_MI

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = windows.size(0)
                ep_total_loss += total_loss.item() * bs
                ep_dro_loss += L_dro.item() * bs
                ep_subject_loss += L_subject.item() * bs
                ep_mi_loss += L_MI.item() * bs
                # Accuracy from base path (inference-time metric)
                preds = gest_logits_base.argmax(dim=1)
                ep_correct += (preds == gesture_labels).sum().item()
                ep_total += bs

            # ── Epoch averages ─────────────────────────────────────────────
            train_loss = ep_total_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            avg_dro = ep_dro_loss / max(1, ep_total)
            avg_subj = ep_subject_loss / max(1, ep_total)
            avg_mi = ep_mi_loss / max(1, ep_total)

            gw_list = group_weights.cpu().tolist()
            gw_real = gw_list[:real_env_count]
            gw_virt = gw_list[real_env_count:]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["dro_loss"].append(avg_dro)
            history["subject_loss"].append(avg_subj)
            history["mi_loss"].append(avg_mi)
            history["group_weights"].append(gw_list)
            history["real_env_weights"].append(gw_real)

            # ── Validation (base path only — mirrors inference) ────────────
            if dl_val is not None:
                model.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)  # eval mode → base-path tensor
                        val_loss_sum += gesture_criterion(logits, yb).item() * yb.size(0)
                        val_correct += (logits.argmax(1) == yb).sum().item()
                        val_total += yb.size(0)
                val_loss = val_loss_sum / max(1, val_total)
                val_acc = val_correct / max(1, val_total)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Log with real-env weights (virtual env count can be large)
            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: total={train_loss:.4f} (dro={avg_dro:.4f}, "
                f"subj={avg_subj:.4f}, MI={avg_mi:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | "
                f"beta={current_beta:.4f} | "
                f"real_gw=[{', '.join(f'{w:.3f}' for w in gw_real)}] | "
                f"virt_gw_sum={sum(gw_virt):.3f}"
            )

            # ── Early stopping on val loss (base path = inference metric) ──
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

        # ── Store trainer state (required by evaluate_numpy()) ────────────
        self.model = model
        self.mean_c = mean_c
        self.std_c = std_c
        self.class_ids = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = window_size
        self.final_group_weights = group_weights.cpu().tolist()
        self.env_descriptions = env_descriptions

        # ── Save training history ─────────────────────────────────────────
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        # ── In-fold evaluation (val + test from training-time splits) ──────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        def _eval_loader(dloader, split_name: str):
            if dloader is None:
                return None
            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in dloader:
                    logits = model(xb.to(device))  # eval mode → base path
                    all_logits.append(logits.cpu().numpy())
                    all_y.append(yb.numpy())
            logits_arr = np.concatenate(all_logits)
            y_true = np.concatenate(all_y)
            y_pred = logits_arr.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            if self.visualizer is not None:
                cls_labels = [class_names[gid] for gid in class_ids]
                self.visualizer.plot_confusion_matrix(
                    cm, cls_labels, normalize=True, filename=f"cm_{split_name}.png",
                )
            return {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "report": report,
                "confusion_matrix": cm.tolist(),
            }

        results["val"] = _eval_loader(dl_val, "val")
        results["test"] = _eval_loader(dl_test, "test")

        # ── Save model checkpoint ─────────────────────────────────────────
        model_path = self.output_dir / "synth_env_groupdro_emg.pt"
        torch.save(
            {
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
                "alpha": self.alpha,
                "beta": self.beta,
                "dro_eta": self.dro_eta,
                "mix_alpha": self.mix_alpha,
                "total_envs": total_envs,
                "real_env_count": real_env_count,
                "virtual_env_count": virtual_env_count,
                "final_group_weights": self.final_group_weights,
                "env_descriptions": env_descriptions,
                "training_config": asdict(self.cfg),
            },
            model_path,
        )
        self.logger.info(f"Model saved: {model_path}")

        with open(self.output_dir / "classification_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    # evaluate_numpy() is inherited from DisentangledTrainer unchanged.
    # model.eval() returns gesture_logits_base (base path tensor) — same
    # interface as DisentangledCNNGRU, so the parent implementation works.
