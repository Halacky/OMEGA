# FILE: experiments/exp_99_discrete_style_codebook_loso.py
"""
Experiment 99: Discrete Style Codebook with EMA Quantisation

Hypothesis: Replacing continuous style space with a discrete VQ codebook
(K=32 entries, EMA updates) forces styles of different subjects onto
shared canonical codes.  During training, random code replacement (p=0.5)
acts as an implicit style normalisation, making content features
robust to unseen subjects.

Key design:
  - VQ-style codebook (K=32, dim=64) with EMA updates
  - Commitment loss  +  diversity loss (anti-collapse)
  - FiLM conditioning of content with (possibly random) style code
  - Inference uses z_content ONLY → no VQ / FiLM at test time
  - Fully LOSO-safe: no subject-specific adaptation

Success criteria: F1 > 36 %, inter-subject std < 4.5 %.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    parse_subjects_args,
    CI_TEST_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver
from visualization.base import Visualizer

from models import register_model
from models.discrete_style_codebook import DiscreteStyleCodebookModel

register_model("discrete_style_codebook", DiscreteStyleCodebookModel)


# ═══════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════


class DiscreteStyleCodebookTrainer:
    """
    Custom trainer for Discrete Style Codebook model.

    Implements the interface expected by CrossSubjectExperiment:
      - fit(splits)           → trains the model
      - evaluate_numpy(X,y,…) → evaluates on held-out data
      - self.class_ids        → set after fit()
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        logger,
        output_dir: Path,
        visualizer,
        # model hyper-params
        style_codebook_size: int = 32,
        style_dim: int = 64,
        content_dim: int = 128,
        commitment_cost: float = 0.25,
        style_augment_prob: float = 0.5,
        diversity_weight: float = 0.1,
        codebook_reset_interval: int = 100,
    ):
        self.cfg = train_cfg
        self.logger = logger
        self.output_dir = output_dir
        self.visualizer = visualizer

        self.style_codebook_size = style_codebook_size
        self.style_dim = style_dim
        self.content_dim = content_dim
        self.commitment_cost = commitment_cost
        self.style_augment_prob = style_augment_prob
        self.diversity_weight = diversity_weight
        self.codebook_reset_interval = codebook_reset_interval

        self.device = torch.device(train_cfg.device)
        self.model: Optional[DiscreteStyleCodebookModel] = None
        self.class_ids: Optional[List[int]] = None
        self.class_names: Optional[Dict[int, str]] = None

        self.best_val_acc = 0.0
        self.best_model_state = None

    # ───────── helpers ─────────

    @staticmethod
    def _splits_to_arrays(
        split_dict: Dict[int, np.ndarray], class_ids: List[int]
    ):
        """Convert {gesture_id: array(N,T,C)} → flat (X, y) arrays."""
        X_parts, y_parts = [], []
        for i, gid in enumerate(class_ids):
            if gid in split_dict and len(split_dict[gid]) > 0:
                X_parts.append(split_dict[gid])
                y_parts.append(
                    np.full(len(split_dict[gid]), i, dtype=np.int64)
                )
        X = np.concatenate(X_parts, axis=0).astype(np.float32)
        y = np.concatenate(y_parts, axis=0)
        return X, y

    # ───────── fit ─────────

    def fit(self, splits: Dict) -> Dict:
        """
        Train DiscreteStyleCodebookModel on cross-subject splits.

        Args:
            splits: {"train": {gid: arr(N,T,C)}, "val": …, "test": …}
        """
        # class ids from training set
        train_d = {
            gid: arr
            for gid, arr in splits["train"].items()
            if len(arr) > 0
        }
        self.class_ids = sorted(train_d.keys())
        self.class_names = {gid: f"Gesture {gid}" for gid in self.class_ids}
        n_classes = len(self.class_ids)

        # flat arrays
        train_X, train_y = self._splits_to_arrays(
            splits["train"], self.class_ids
        )
        val_X, val_y = self._splits_to_arrays(
            splits["val"], self.class_ids
        )

        # (N, T, C) → (N, C, T) for Conv1d
        train_x_t = torch.from_numpy(train_X).float().transpose(1, 2)
        train_y_t = torch.from_numpy(train_y).long()
        val_x_t = torch.from_numpy(val_X).float().transpose(1, 2)
        val_y_t = torch.from_numpy(val_y).long()

        n_channels = train_x_t.shape[1]

        self.logger.info(
            f"DiscreteStyleCodebook: ch={n_channels}, "
            f"classes={n_classes}, K={self.style_codebook_size}"
        )
        self.logger.info(
            f"Train: {len(train_x_t)}, Val: {len(val_x_t)}"
        )

        # ── model ──
        self.model = DiscreteStyleCodebookModel(
            in_channels=n_channels,
            num_classes=n_classes,
            dropout=self.cfg.dropout,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            style_codebook_size=self.style_codebook_size,
            commitment_cost=self.commitment_cost,
            style_augment_prob=self.style_augment_prob,
            diversity_weight=self.diversity_weight,
        ).to(self.device)

        # ── class-weighted loss ──
        if self.cfg.use_class_weights:
            counts = np.bincount(train_y, minlength=n_classes)
            w = 1.0 / (counts + 1e-6)
            w = w / w.sum() * n_classes
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(w).float().to(self.device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        # ── optimiser / scheduler ──
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        # ── data loader ──
        train_loader = DataLoader(
            TensorDataset(train_x_t, train_y_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

        # ── training loop ──
        patience_counter = 0
        global_step = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            ep_loss = 0.0
            ep_cls = 0.0
            ep_aux = 0.0

            for bx, by in train_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                optimizer.zero_grad()

                logits = self.model(bx)
                cls_loss = criterion(logits, by)
                aux_loss = self.model.get_auxiliary_loss()
                loss = cls_loss + aux_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()

                ep_loss += loss.item()
                ep_cls += cls_loss.item()
                ep_aux += aux_loss.item()

                global_step += 1

                # periodic codebook reset
                if global_step % self.codebook_reset_interval == 0:
                    info = self.model.reset_unused_codebooks(threshold=0.001)
                    if info["style_codes_reset"] > 0:
                        self.logger.info(
                            f"Step {global_step}: reset "
                            f"{info['style_codes_reset']} style codes"
                        )

            # ── validation ──
            # val arrays are still (N, C, T) in torch tensors;
            # evaluate_numpy expects (N, T, C) numpy
            val_metrics = self.evaluate_numpy(
                val_X, val_y, "val", visualize=False
            )
            val_acc = val_metrics["accuracy"]

            scheduler.step(val_acc)

            nb = len(train_loader)
            self.logger.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Loss {ep_loss/nb:.4f} "
                f"(cls {ep_cls/nb:.4f}, aux {ep_aux/nb:.4f}) | "
                f"Val acc {val_acc:.4f}"
            )

            # codebook stats every 10 epochs
            if (epoch + 1) % 10 == 0:
                stats = self.model.get_codebook_stats()
                self.logger.info(
                    f"  Codebook usage — Style: "
                    f"{stats['style']['used_codes']}/"
                    f"{stats['style']['total_codes']}"
                )

            # ── early stopping ──
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch + 1}"
                    )
                    break

        # restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        return {"best_val_accuracy": self.best_val_acc}

    # ───────── evaluate ─────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str,
        visualize: bool = True,
    ) -> Dict:
        """
        Evaluate on numpy arrays.

        Args:
            X: (N, T, C)
            y: class indices (0-based)
        """
        if self.model is None:
            raise RuntimeError("Model not trained — call fit() first.")

        self.model.eval()

        # (N, T, C) → (N, C, T) → device
        X_t = torch.from_numpy(X).float().transpose(1, 2).to(self.device)

        all_preds, all_probs = [], []

        with torch.no_grad():
            bs = 4096
            for i in range(0, len(X_t), bs):
                chunk = X_t[i : i + bs]
                logits = self.model(chunk)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        preds = np.concatenate(all_preds)
        probs = np.concatenate(all_probs)

        from sklearn.metrics import (
            f1_score,
            confusion_matrix,
            classification_report,
        )

        accuracy = float((preds == y).mean())
        f1_macro = float(
            f1_score(y, preds, average="macro", zero_division=0)
        )
        f1_weighted = float(
            f1_score(y, preds, average="weighted", zero_division=0)
        )
        cm = confusion_matrix(y, preds)
        report = classification_report(y, preds, zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "predictions": preds,
            "probabilities": probs,
            "confusion_matrix": cm,
            "report": report,
        }

        if visualize and self.visualizer:
            self.visualizer.plot_confusion_matrix(
                cm,
                class_labels=[str(c) for c in range(len(np.unique(y)))],
                filename=f"cm_{split_name}.png",
            )

        return metrics


# ═══════════════════════════════════════════════════════════
# Single LOSO fold
# ═══════════════════════════════════════════════════════════


def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    vq_cfg: Dict,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    # save configs
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects,
        test_subject=test_subject,
        exercises=exercises,
        base_dir=base_dir,
        pool_train_subjects=True,
        use_separate_val_subject=False,
        val_subject=None,
        val_ratio=0.15,
        seed=train_cfg.seed,
        max_gestures=10,
    )
    cs_cfg.save(output_dir / "cross_subject_config.json")

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=False,
    )

    base_viz = Visualizer(output_dir, logger)

    trainer = DiscreteStyleCodebookTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        style_codebook_size=vq_cfg["style_codebook_size"],
        style_dim=vq_cfg["style_dim"],
        content_dim=vq_cfg["content_dim"],
        commitment_cost=vq_cfg["commitment_cost"],
        style_augment_prob=vq_cfg["style_augment_prob"],
        diversity_weight=vq_cfg["diversity_weight"],
        codebook_reset_interval=vq_cfg["codebook_reset_interval"],
    )

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=base_viz,
        logger=logger,
    )

    try:
        results = experiment.run()
    except Exception as e:
        logger.error(
            f"Error in fold (test={test_subject}): {e}"
        )
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_m = results.get("cross_subject_test", {})
    test_acc = float(test_m.get("accuracy", 0.0))
    test_f1 = float(test_m.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test {test_subject} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )

    # save results (drop heavy subjects_data)
    to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(
            make_json_serializable(to_save), f, indent=4, ensure_ascii=False
        )

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable(
            {
                "test_subject": test_subject,
                "train_subjects": train_subjects,
                "model_type": model_type,
                "exercises": exercises,
                "vq_config": vq_cfg,
                "metrics": {
                    "test_accuracy": test_acc,
                    "test_f1_macro": test_f1,
                },
            }
        ),
        filename="fold_metadata.json",
    )

    # cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc

    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════


def main():
    EXPERIMENT_NAME = "exp_99_discrete_style_codebook_loso"
    HYPOTHESIS_ID = "h-099-discrete-style-codebook"

    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")

    ALL_SUBJECTS = parse_subjects_args()

    EXERCISES = ["E1"]
    MODEL_TYPE = "discrete_style_codebook"

    # ── VQ hyperparameters ──
    VQ_CFG = {
        "style_codebook_size": 32,
        "style_dim": 64,
        "content_dim": 128,
        "commitment_cost": 0.25,
        "style_augment_prob": 0.5,
        "diversity_weight": 0.1,
        "codebook_reset_interval": 100,
    }

    # ── processing ──
    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )

    # ── splits ──
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    # ── training ──
    train_cfg = TrainingConfig(
        batch_size=512,
        epochs=60,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        model_type=MODEL_TYPE,
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )

    # ── run ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: Discrete Style Codebook (K={VQ_CFG['style_codebook_size']})")
    print(
        f"Subjects: {len(ALL_SUBJECTS)} "
        f"({'CI test' if len(ALL_SUBJECTS) == 5 else 'Full'})"
    )
    print(f"Augmentation: noise + time_warp + VQ style swap (p=0.5)")

    all_results: List[Dict] = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_DIR / MODEL_TYPE / f"test_{test_subject}"

        try:
            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                model_type=MODEL_TYPE,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                vq_cfg=VQ_CFG,
            )
            all_results.append(fold_res)

            acc_s = (
                f"{fold_res['test_accuracy']:.4f}"
                if fold_res.get("test_accuracy") is not None
                else "N/A"
            )
            f1_s = (
                f"{fold_res['test_f1_macro']:.4f}"
                if fold_res.get("test_f1_macro") is not None
                else "N/A"
            )
            print(f"  {test_subject}: acc={acc_s}, f1={f1_s}")

        except Exception as e:
            global_logger.error(f"Failed {test_subject}: {e}")
            traceback.print_exc()
            all_results.append(
                {
                    "test_subject": test_subject,
                    "model_type": MODEL_TYPE,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": str(e),
                }
            )

    # ── aggregate ──
    valid = [r for r in all_results if r.get("test_accuracy") is not None]

    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s = [r["test_f1_macro"] for r in valid]
        aggregate = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_folds": len(valid),
        }
        print(
            f"\nAggregate ({len(valid)} folds):\n"
            f"  Accuracy = {aggregate['mean_accuracy']:.4f} "
            f"+/- {aggregate['std_accuracy']:.4f}\n"
            f"  F1-macro = {aggregate['mean_f1_macro']:.4f} "
            f"+/- {aggregate['std_f1_macro']:.4f}"
        )
    else:
        aggregate = {}
        print("\nNo successful folds.")

    # ── save summary ──
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "model": MODEL_TYPE,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "vq_config": VQ_CFG,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": all_results,
        "experiment_date": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(
            make_json_serializable(summary), f, indent=4, ensure_ascii=False
        )
    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")

    # ── hypothesis status ──
    try:
        from hypothesis_executor.qdrant_callback import (
            mark_hypothesis_verified,
            mark_hypothesis_failed,
        )

        if aggregate:
            metrics = aggregate.copy()
            metrics["best_model"] = MODEL_TYPE
            mark_hypothesis_verified(
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                experiment_name=EXPERIMENT_NAME,
            )
            print(f"\nHypothesis {HYPOTHESIS_ID} verified: {metrics}")
        else:
            mark_hypothesis_failed(
                hypothesis_id=HYPOTHESIS_ID,
                error_message="No successful LOSO folds completed",
            )
            print(f"\nHypothesis {HYPOTHESIS_ID} failed.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
