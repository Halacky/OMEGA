"""
Experiment 53: Latent Diffusion for Subject-Style Removal (LOSO)

Hypothesis (h-053-diffusion-canonical):
    A denoising diffusion process in the encoder's latent space can learn
    to remove subject-specific style while preserving gesture-discriminative
    content. The diffusion process maps noisy (subject-contaminated) latents
    to clean canonical representations.

Architecture: LatentDiffusionEMG (models/latent_diffusion_emg.py)
Training:     L_gesture + lambda_diff * L_diffusion (noise prediction MSE)
Inference:    Encode -> optional denoise -> classify
"""

import os
import sys
import json
import gc
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from training.datasets import WindowDataset, AugmentedWindowDataset
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything, get_worker_init_fn
from utils.artifacts import ArtifactSaver

from models.latent_diffusion_emg import LatentDiffusionEMG
from models import register_model

register_model("latent_diffusion_emg", LatentDiffusionEMG)

EXPERIMENT_NAME = "exp_53_latent_diffusion_subject_style_removal_loso"
HYPOTHESIS_ID = "h-053-diffusion-canonical"
LAMBDA_DIFF = 0.3
HIDDEN_DIM = 128
N_DIFFUSION_STEPS = 50
DENOISE_STEPS_EVAL = 10  # fewer steps at evaluation for speed


class DiffusionTrainer(WindowClassifierTrainer):
    """Custom trainer for LatentDiffusionEMG with diffusion loss."""

    def __init__(self, train_cfg, logger, output_dir, visualizer=None,
                 lambda_diff=0.3, hidden_dim=128, n_diff_steps=50, denoise_eval=10):
        super().__init__(train_cfg, logger, output_dir, visualizer)
        self.lambda_diff = lambda_diff
        self.hidden_dim = hidden_dim
        self.n_diff_steps = n_diff_steps
        self.denoise_eval = denoise_eval

    def fit(self, splits: Dict) -> Dict:
        seed_everything(self.cfg.seed)

        (X_train, y_train, X_val, y_val, X_test, y_test,
         class_ids, class_names) = self._prepare_splits_arrays(splits)

        num_classes = len(class_ids)
        self.class_ids = class_ids
        self.class_names = class_names

        # Transpose (N, T, C) -> (N, C, T)
        X_train = np.transpose(X_train, (0, 2, 1))
        if len(X_val) > 0 and X_val.ndim == 3:
            X_val = np.transpose(X_val, (0, 2, 1))
        if len(X_test) > 0 and X_test.ndim == 3:
            X_test = np.transpose(X_test, (0, 2, 1))

        in_channels = X_train.shape[1]
        self.in_channels = in_channels
        self.window_size = X_train.shape[2]

        mean_c, std_c = self._compute_channel_standardization(X_train)
        self.mean_c = mean_c
        self.std_c = std_c

        X_train = self._apply_standardization(X_train, mean_c, std_c)
        if X_val.ndim == 3 and len(X_val) > 0:
            X_val = self._apply_standardization(X_val, mean_c, std_c)

        self.logger.info(f"DiffusionTrainer: in_ch={in_channels}, classes={num_classes}, "
                         f"diff_steps={self.n_diff_steps}, lambda_diff={self.lambda_diff}")

        # Datasets
        use_aug = getattr(self.cfg, "aug_apply", False)
        if use_aug:
            ds_train = AugmentedWindowDataset(
                X_train, y_train,
                noise_std=getattr(self.cfg, "aug_noise_std", 0.02),
                max_warp=getattr(self.cfg, "aug_time_warp_max", 0.1),
                apply_noise=getattr(self.cfg, "aug_apply_noise", True),
                apply_time_warp=getattr(self.cfg, "aug_apply_time_warp", True),
            )
        else:
            ds_train = WindowDataset(X_train, y_train)

        ds_val = WindowDataset(X_val, y_val) if (X_val.ndim == 3 and len(X_val) > 0) else None

        worker_init = get_worker_init_fn(self.cfg.seed)
        dl_train = DataLoader(
            ds_train, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        dl_val = DataLoader(
            ds_val, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=worker_init if self.cfg.num_workers > 0 else None,
        ) if ds_val else None

        model = LatentDiffusionEMG(
            in_channels=in_channels, num_classes=num_classes,
            dropout=self.cfg.dropout, hidden_dim=self.hidden_dim,
            n_diffusion_steps=self.n_diff_steps,
        ).to(self.cfg.device)

        self.logger.info(f"LatentDiffusionEMG: {sum(p.numel() for p in model.parameters()):,} params")

        # Loss
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            cw = counts.sum() / (counts + 1e-8)
            cw = cw / cw.mean()
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(cw).float().to(self.cfg.device))
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate,
                               weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                    "gesture_loss": [], "diff_loss": []}
        best_val_acc = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            ep_total, ep_gest, ep_diff = 0.0, 0.0, 0.0
            ep_correct, ep_count = 0, 0

            for xb, yb in dl_train:
                xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)
                optimizer.zero_grad()

                out = model.forward_all(xb)
                L_gesture = criterion(out["logits"], yb)
                L_diff = out["diffusion_loss"]
                loss = L_gesture + self.lambda_diff * L_diff

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = yb.size(0)
                ep_total += loss.item() * bs
                ep_gest += L_gesture.item() * bs
                ep_diff += L_diff.item() * bs
                ep_correct += (out["logits"].argmax(1) == yb).sum().item()
                ep_count += bs

            train_loss = ep_total / max(1, ep_count)
            train_acc = ep_correct / max(1, ep_count)

            val_loss, val_acc = float("nan"), float("nan")
            if dl_val is not None:
                model.eval()
                vl, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)
                        logits = model(xb)
                        vl += criterion(logits, yb).item() * yb.size(0)
                        vc += (logits.argmax(1) == yb).sum().item()
                        vt += yb.size(0)
                val_loss = vl / max(1, vt)
                val_acc = vc / max(1, vt)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["gesture_loss"].append(ep_gest / max(1, ep_count))
            history["diff_loss"].append(ep_diff / max(1, ep_count))

            self.logger.info(
                f"[Epoch {epoch:02d}/{self.cfg.epochs}] "
                f"Train: loss={train_loss:.4f} (gest={history['gesture_loss'][-1]:.4f}, "
                f"diff={history['diff_loss'][-1]:.4f}), acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.3f}"
            )

            if dl_val is not None and not np.isnan(val_acc):
                scheduler.step(val_acc)
                if val_acc > best_val_acc + 1e-6:
                    best_val_acc = val_acc
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
        self.model = model

        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        torch.save({
            "state_dict": model.state_dict(), "in_channels": in_channels,
            "num_classes": num_classes, "class_ids": class_ids,
            "mean": mean_c, "std": std_c, "hidden_dim": self.hidden_dim,
        }, self.output_dir / "diffusion_model.pt")

        return {"class_ids": class_ids, "best_val_acc": float(best_val_acc)}

    def evaluate_numpy(self, X, y, split_name="custom", visualize=False):
        assert self.model is not None
        assert self.mean_c is not None and self.std_c is not None
        assert self.class_ids is not None

        if X.ndim == 3:
            N, d1, d2 = X.shape
            X_input = np.transpose(X, (0, 2, 1)) if d1 > d2 else X.copy()
        else:
            raise ValueError(f"Expected 3D, got {X.shape}")

        Xs = self._apply_standardization(X_input, self.mean_c, self.std_c)

        ds = WindowDataset(Xs, y)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                        num_workers=self.cfg.num_workers, pin_memory=True)

        self.model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.cfg.device)
                # Use denoised forward for evaluation
                logits = self.model.forward_denoised(xb, n_steps=self.denoise_eval)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())

        logits_np = np.concatenate(all_logits)
        y_true = np.concatenate(all_y)
        y_pred = logits_np.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(self.class_ids)))

        if visualize and self.visualizer is not None:
            labels = [self.class_names.get(g, f"G{g}") for g in self.class_ids]
            self.visualizer.plot_confusion_matrix(cm, labels, normalize=True,
                                                   filename=f"cm_{split_name}.png")

        return {"accuracy": float(acc), "f1_macro": float(f1_m),
                "report": report, "confusion_matrix": cm.tolist()}


# ───────────── LOSO fold ─────────────

def run_single_loso_fold(base_dir, output_dir, train_subjects, test_subject,
                          exercises, proc_cfg, split_cfg, train_cfg):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = "deep_raw"
    train_cfg.model_type = "latent_diffusion_emg"
    train_cfg.use_handcrafted_features = False

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects, test_subject=test_subject,
        exercises=exercises, base_dir=base_dir,
        pool_train_subjects=True, use_separate_val_subject=False,
        val_subject=None, val_ratio=0.15, seed=train_cfg.seed, max_gestures=10,
    )
    cs_cfg.save(output_dir / "cross_subject_config.json")

    multi_loader = MultiSubjectLoader(processing_config=proc_cfg, logger=logger,
                                       use_gpu=True, use_improved_processing=True)
    base_viz = Visualizer(output_dir, logger)

    trainer = DiffusionTrainer(
        train_cfg=train_cfg, logger=logger, output_dir=output_dir, visualizer=base_viz,
        lambda_diff=LAMBDA_DIFF, hidden_dim=HIDDEN_DIM,
        n_diff_steps=N_DIFFUSION_STEPS, denoise_eval=DENOISE_STEPS_EVAL,
    )

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg, split_config=split_cfg,
        multi_subject_loader=multi_loader, trainer=trainer,
        visualizer=base_viz, logger=logger,
    )

    try:
        results = experiment.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return {"test_subject": test_subject, "model_type": "latent_diffusion_emg",
                "test_accuracy": None, "test_f1_macro": None, "error": str(e)}

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(f"[LOSO] Test {test_subject} | Acc={test_acc:.4f}, F1={test_f1:.4f}")

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(make_json_serializable({
        "test_subject": test_subject, "train_subjects": train_subjects,
        "model_type": "latent_diffusion_emg", "exercises": exercises,
        "metrics": {"test_accuracy": test_acc, "test_f1_macro": test_f1},
    }), filename="fold_metadata.json")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {"test_subject": test_subject, "model_type": "latent_diffusion_emg",
            "test_accuracy": test_acc, "test_f1_macro": test_f1}


def main():
    ALL_SUBJECTS = parse_subjects_args()
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")

    proc_cfg = ProcessingConfig(window_size=600, window_overlap=300, num_channels=12,
                                 sampling_rate=2000, segment_edge_margin=0.1)
    split_cfg = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                             mode="by_segments", shuffle_segments=True, seed=42,
                             include_rest_in_splits=False)
    train_cfg = TrainingConfig(
        batch_size=256, epochs=60, learning_rate=1e-3, weight_decay=1e-4,
        dropout=0.3, early_stopping_patience=10, use_class_weights=True,
        seed=42, num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False, pipeline_type="deep_raw",
        model_type="latent_diffusion_emg",
        aug_apply=True, aug_noise_std=0.02, aug_time_warp_max=0.1,
        aug_apply_noise=True, aug_apply_time_warp=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nEXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: LatentDiffusionEMG, diff_steps={N_DIFFUSION_STEPS}, lambda={LAMBDA_DIFF}")
    print(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")

    all_results = []
    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_DIR / "latent_diffusion_emg" / f"test_{test_subject}"

        result = run_single_loso_fold(
            base_dir=BASE_DIR, output_dir=fold_dir,
            train_subjects=train_subjects, test_subject=test_subject,
            exercises=["E1"], proc_cfg=proc_cfg, split_cfg=split_cfg, train_cfg=train_cfg,
        )
        all_results.append(result)
        acc_s = f"{result['test_accuracy']:.4f}" if result.get('test_accuracy') is not None else "N/A"
        f1_s = f"{result['test_f1_macro']:.4f}" if result.get('test_f1_macro') is not None else "N/A"
        print(f"  -> {test_subject}: acc={acc_s}, f1={f1_s}")

    # Aggregate
    valid = [r for r in all_results if r.get("test_accuracy") is not None]
    aggregate = {}
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s = [r["test_f1_macro"] for r in valid]
        aggregate["latent_diffusion_emg"] = {
            "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)), "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
        }
        print(f"\nDiffusion: Acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}, "
              f"F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}")

    summary = {
        "experiment_name": EXPERIMENT_NAME, "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw", "model": "latent_diffusion_emg",
        "lambda_diff": LAMBDA_DIFF, "n_diffusion_steps": N_DIFFUSION_STEPS,
        "subjects": ALL_SUBJECTS, "exercises": ["E1"],
        "aggregate_results": aggregate, "individual_results": all_results,
        "experiment_date": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)

    try:
        from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed
        if aggregate:
            mark_hypothesis_verified(HYPOTHESIS_ID, metrics=aggregate["latent_diffusion_emg"],
                                     experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed(HYPOTHESIS_ID, "No successful LOSO folds")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
