"""
Experiment 36: Prototypical Networks with ArcFace Loss for EMG (LOSO)

Hypothesis H10: Classes are better modelled as distributions than SVM boundaries.

Intuition:
  Prototypical networks learn a metric embedding space where gestures from
  different subjects cluster around shared prototypes.  Distance-based
  classification is more robust to domain shift than linear SVM boundaries
  because the metric generalises across subjects.

Implementation:
  - Embedding network: CNN (3 stages) → BiGRU (2 layers, bidirectional) →
    FC → L2-normalised embedding on unit hypersphere
  - Class prototypes: mean embedding per gesture class
  - Training loss: ArcFace (Additive Angular Margin Softmax)
      logit_target  = scale · cos(θ + m)
      logit_others  = scale · cos(θ)
  - Post-training prototype update: replace learned weight matrix with
    empirical class mean embeddings from training set
  - Inference: nearest prototype (cosine similarity)

Baseline comparison:
  exp_1  (CNN-GRU-Attention, deep_raw) — best deep baseline
  exp_4  (SVM, powerful features)      — best classical baseline
"""

import os
import sys
import json
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

from models.prototypical_arcface_emg import PrototypicalArcFaceEMG
from models import register_model

register_model("prototypical_arcface", PrototypicalArcFaceEMG)

# ---------------------------------------------------------------------------
# Subject lists
# ---------------------------------------------------------------------------

_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]


def parse_subjects_args() -> List[str]:
    """Parse --subjects / --ci / --full CLI args.  Defaults to CI subjects."""
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci",   action="store_true")
    _parser.add_argument("--full", action="store_true")
    _args, _ = _parser.parse_known_args()
    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    # Default: CI subjects (server has symlinks only for these)
    return _CI_SUBJECTS


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def make_json_serializable(obj):
    from pathlib import Path as _Path
    import numpy as _np
    if isinstance(obj, _Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, _np.integer):
        return int(obj)
    elif isinstance(obj, _np.floating):
        return float(obj)
    elif isinstance(obj, _np.ndarray):
        return obj.tolist()
    else:
        return obj


# ---------------------------------------------------------------------------
# Prototypical ArcFace hyperparameters
# ---------------------------------------------------------------------------

PROTO_CFG = {
    "embed_dim":         128,    # embedding dimensionality
    "margin":            0.3,    # ArcFace angular margin (radians)
    "scale":             32.0,   # cosine similarity temperature
    "proto_batch_size":  512,    # batch size for prototype extraction pass
}


# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------

class PrototypicalArcFaceTrainer(WindowClassifierTrainer):
    """
    Two-phase training strategy for Prototypical ArcFace networks:

    Phase 1 — Metric learning via ArcFace:
        Train EMGEmbeddingNet + ArcFaceHead with additive angular margin loss.
        Labels are passed to model.forward() so the margin is applied to the
        target class.  Cross-entropy over ArcFace logits.
        Early stopping on validation cross-entropy (eval mode, no margin).

    Phase 2 — Prototype anchoring:
        After gradient training, extract embeddings for all training samples.
        Replace the ArcFace weight matrix with L2-normalised class mean embeddings.
        From this point, inference = nearest-prototype cosine similarity.

    The trainer inherits evaluate_numpy() from WindowClassifierTrainer.
    Because model.forward(x) in eval mode returns scale·cos(θ) against stored
    prototypes, evaluate_numpy produces nearest-prototype predictions without
    any modification.
    """

    def __init__(self, proto_cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.proto_cfg = proto_cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_transpose(X: np.ndarray) -> np.ndarray:
        """
        Ensure data is in (N, C, T) PyTorch format.
        _prepare_splits_arrays returns (N, T, C); heuristic: if dim1 > dim2 → transpose.
        """
        if X.ndim == 3 and X.shape[1] > X.shape[2]:
            return np.transpose(X, (0, 2, 1))
        return X

    @staticmethod
    def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # Overridden fit()
    # ------------------------------------------------------------------

    def fit(self, splits: Dict) -> Dict:
        """
        Custom two-phase fit:
          1. Prepare arrays, standardise channel-wise (train stats only).
          2. ArcFace training loop with early stopping on val CE loss.
          3. Restore best weights.
          4. Compute mean embeddings per class from training data.
          5. Update model prototype matrix with those mean embeddings.
        """
        seed_everything(self.cfg.seed)

        # --- Data preparation -------------------------------------------------
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # Transpose (N, T, C) → (N, C, T)  [parent returns (N, T, C)]
        X_train = self._maybe_transpose(X_train)
        X_val   = self._maybe_transpose(X_val)   if len(X_val)  > 0 else X_val
        X_test  = self._maybe_transpose(X_test)  if len(X_test) > 0 else X_test

        in_channels = X_train.shape[1]   # C
        time_steps  = X_train.shape[2]   # T
        num_classes = len(class_ids)
        device      = self.cfg.device

        # --- Channel standardisation (train statistics only) -----------------
        mean_c, std_c = self._compute_channel_standardization(X_train)

        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = time_steps

        X_train_n = self._apply_standardization(X_train, mean_c, std_c)
        X_val_n   = self._apply_standardization(X_val, mean_c, std_c)   if len(X_val)  > 0 else X_val
        X_test_n  = self._apply_standardization(X_test, mean_c, std_c)  if len(X_test) > 0 else X_test

        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # --- Model + optimiser -----------------------------------------------
        model = PrototypicalArcFaceEMG(
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=self.proto_cfg["embed_dim"],
            dropout=self.cfg.dropout,
            margin=self.proto_cfg["margin"],
            scale=self.proto_cfg["scale"],
        ).to(device)

        # Class-balanced cross-entropy (ArcFace logits are passed to CE)
        if self.cfg.use_class_weights:
            counts  = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            weights = counts.sum() / (counts + 1e-8)
            weights /= weights.mean()
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(weights).float().to(device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        # --- DataLoaders -----------------------------------------------------
        dl_train = self._make_loader(X_train_n, y_train, self.cfg.batch_size, shuffle=True)
        dl_val   = self._make_loader(X_val_n, y_val, self.cfg.batch_size, shuffle=False) \
                   if len(X_val_n) > 0 else None

        # --- Phase 1: ArcFace training loop ----------------------------------
        self.logger.info(
            f"[ProtoArcFace] Training  epochs={self.cfg.epochs}  "
            f"classes={num_classes}  embed_dim={self.proto_cfg['embed_dim']}  "
            f"margin={self.proto_cfg['margin']}  scale={self.proto_cfg['scale']}"
        )

        best_val_loss    = float("inf")
        best_state       = None
        patience_counter = 0
        patience         = self.cfg.early_stopping_patience
        history          = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, self.cfg.epochs + 1):
            # ---- train ----
            model.train()
            train_loss = 0.0
            train_correct = 0
            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                # Pass labels so ArcFace margin is applied to target class
                logits = model(xb, labels=yb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss    += loss.item() * len(yb)
                train_correct += (logits.argmax(1) == yb).sum().item()
            train_loss /= len(y_train)
            train_acc   = train_correct / len(y_train)

            # ---- validation ----
            val_loss = float("nan")
            val_acc  = float("nan")
            if dl_val is not None:
                model.eval()
                val_loss_sum = 0.0
                val_correct  = 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        # Eval mode → no margin → pure cosine similarity logits
                        logits = model(xb)
                        val_loss_sum += criterion(logits, yb).item() * len(yb)
                        val_correct  += (logits.argmax(1) == yb).sum().item()
                val_loss = val_loss_sum / len(y_val)
                val_acc  = val_correct  / len(y_val)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss    = val_loss
                    best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"[ProtoArcFace] Early stopping at epoch {epoch}.")
                        break

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss) if not isinstance(val_loss, float) or not (val_loss != val_loss) else None)
            history["train_acc"].append(float(train_acc))
            history["val_acc"].append(float(val_acc) if not isinstance(val_acc, float) or not (val_acc != val_acc) else None)

            if epoch % 5 == 0 or epoch == 1:
                val_str = f"{val_loss:.4f}" if val_loss == val_loss else "n/a"
                self.logger.info(
                    f"[ProtoArcFace] Epoch {epoch:03d}/{self.cfg.epochs} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_str}"
                )
                print(
                    f"[ProtoArcFace] Epoch {epoch:03d}/{self.cfg.epochs} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_str}"
                )

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
            self.logger.info(f"[ProtoArcFace] Restored best val_loss={best_val_loss:.4f}")

        # --- Phase 2: Prototype update ----------------------------------------
        # Replace ArcFace weight matrix with mean training embeddings per class.
        # From this point model.forward(x) in eval mode = nearest-prototype search.
        self.logger.info("[ProtoArcFace] Computing class prototypes from training embeddings …")
        model.update_prototypes(
            X=X_train_n,
            y=y_train,
            num_classes=num_classes,
            device=device,
            batch_size=self.proto_cfg["proto_batch_size"],
        )
        self.logger.info("[ProtoArcFace] Prototype update complete.")

        # --- Persist model + metadata ----------------------------------------
        self.model = model

        model_path = self.output_dir / "prototypical_arcface.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_channels": in_channels,
                "num_classes": num_classes,
                "embed_dim": self.proto_cfg["embed_dim"],
                "class_ids": class_ids,
                "mean": mean_c,
                "std": std_c,
                "window_size": time_steps,
                "proto_cfg": self.proto_cfg,
            },
            model_path,
        )
        self.logger.info(f"[ProtoArcFace] Model saved: {model_path}")

        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        if self.visualizer is not None and any(
            v is not None for v in history["val_loss"]
        ):
            self.visualizer.plot_training_curves(history, filename="training_curves.png")

        return {"class_ids": class_ids, "class_names": class_names}


# ---------------------------------------------------------------------------
# LOSO fold runner
# ---------------------------------------------------------------------------

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    proto_cfg: dict,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.model_type    = "prototypical_arcface"
    train_cfg.pipeline_type = "deep_raw"
    train_cfg.use_handcrafted_features = False

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "proto_config.json", "w") as f:
        json.dump(proto_cfg, f, indent=4)

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
        use_improved_processing=True,
    )

    from visualization.base import Visualizer
    from visualization.cross_subject import CrossSubjectVisualizer

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    trainer = PrototypicalArcFaceTrainer(
        proto_cfg=proto_cfg,
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
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
        logger.error(f"Error in LOSO fold (test={test_subject}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc     = float(test_metrics.get("accuracy", 0.0))
    test_f1      = float(test_metrics.get("f1_macro", 0.0))

    logger.info(f"[LOSO] Test={test_subject} | Acc={test_acc:.4f}, F1={test_f1:.4f}")
    print(      f"[LOSO] Test={test_subject} | Acc={test_acc:.4f}, F1={test_f1:.4f}")

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "model_type": "prototypical_arcface",
            "exercises": exercises,
            "proto_cfg": proto_cfg,
            "metrics": {"test_accuracy": test_acc, "test_f1_macro": test_f1},
        }),
        filename="fold_metadata.json",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPERIMENT_NAME = "exp_36_prototypical_arcface_loso"
    BASE_DIR        = ROOT / "data"
    ALL_SUBJECTS    = parse_subjects_args()
    OUTPUT_DIR      = Path(
        f"./experiments_output/{EXPERIMENT_NAME}_"
        + "_".join(s.split("_s")[1] for s in ALL_SUBJECTS)
    )
    EXERCISES = ["E1"]

    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )

    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=80,
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=10,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
    )

    proto_cfg = dict(PROTO_CFG)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"EXPERIMENT: {EXPERIMENT_NAME}")
    global_logger.info(f"Hypothesis H10: Prototypical Networks with ArcFace")
    global_logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    global_logger.info(f"Device: {train_cfg.device}")
    global_logger.info(f"Proto config: {proto_cfg}")
    global_logger.info("=" * 80)

    all_loso_results: List[Dict] = []

    for test_subject in ALL_SUBJECTS:
        print(f"\n  LOSO fold: test_subject={test_subject}")
        train_subjects  = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"

        try:
            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                proto_cfg=dict(proto_cfg),
            )
            all_loso_results.append(fold_res)
            acc = fold_res["test_accuracy"]
            f1  = fold_res["test_f1_macro"]
            acc_str = f"{acc:.4f}" if acc is not None else "None"
            f1_str  = f"{f1:.4f}"  if f1  is not None else "None"
            print(f"  ✓ acc={acc_str}, f1={f1_str}")
        except Exception as e:
            global_logger.error(f"✗ Failed fold test={test_subject}: {e}")
            global_logger.error(traceback.format_exc())
            all_loso_results.append({
                "test_subject": test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })

    # --- Aggregate results ---
    valid = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"]  for r in valid]
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        mean_f1  = float(np.mean(f1s))
        std_f1   = float(np.std(f1s))
        print(
            f"\n[LOSO Summary] prototypical_arcface: "
            f"Acc={mean_acc:.4f}±{std_acc:.4f}, "
            f"F1={mean_f1:.4f}±{std_f1:.4f} (n={len(valid)})"
        )
        global_logger.info(
            f"[LOSO Summary] prototypical_arcface: "
            f"Acc={mean_acc:.4f}±{std_acc:.4f}, "
            f"F1={mean_f1:.4f}±{std_f1:.4f} (n={len(valid)})"
        )
    else:
        mean_acc = mean_f1 = None
        global_logger.warning("No successful folds — all failed.")

    # --- Save summary ---
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": "H10",
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "proto_cfg": proto_cfg,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": {
            "mean_accuracy": mean_acc,
            "std_accuracy":  std_acc  if valid else None,
            "mean_f1_macro": mean_f1,
            "std_f1_macro":  std_f1   if valid else None,
            "num_subjects":  len(valid),
        } if valid else {},
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    global_logger.info(f"EXPERIMENT COMPLETE. Results: {OUTPUT_DIR.resolve()}")

    # --- Optional: report to hypothesis executor ---
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        if valid and mean_acc is not None:
            mark_hypothesis_verified(
                "H10",
                {
                    "mean_accuracy":  mean_acc,
                    "std_accuracy":   std_acc,
                    "mean_f1_macro":  mean_f1,
                    "std_f1_macro":   std_f1,
                    "num_subjects":   len(valid),
                },
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed("H10", "All LOSO folds failed — no results produced.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
