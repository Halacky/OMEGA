"""
Experiment 35: MAE Self-Supervised Pretraining for EMG Gesture Classification (LOSO)

Hypothesis H9: Self-supervised reconstruction pretraining improves representation.

Intuition (MAE-style):
  - Mask 40% of temporal patches from each EMG window
  - Pretrain a Transformer encoder to reconstruct the masked parts
  - Fine-tune the encoder + classification head on gesture labels
  - The model learns EMG structure, not just gesture boundaries

Architecture:
  Pretrain:  (B, C, T) → PatchEmbed → mask 40% → Encoder → Decoder → MSE(masked)
  Finetune:  (B, C, T) → PatchEmbed → Encoder (pretrained) → CLS token → FC → logits

Baseline comparison: exp_1 (CNN-GRU-Attention, deep_raw), exp_29 (SpectralTransformer)

Key implementation decisions:
  - patch_size=20 → num_patches=30 for T=600
  - Encoder: d_model=128, depth=4, heads=4
  - Decoder: d_model=64, depth=2, heads=4 (lightweight)
  - Pretraining: 30 epochs on train+val data (unlabeled)
  - Fine-tuning: 50 epochs with early stopping, encoder fully trainable
"""

import os
import sys
import json
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# Our model
from models.mae_emg import MAEEmgForPretraining, MAEEmgForClassification
from models import register_model

register_model("mae_emg", MAEEmgForClassification)

# ---------------------------------------------------------------------------
# Subject lists
# ---------------------------------------------------------------------------

_FULL_SUBJECTS = [
    "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]


def parse_subjects_args() -> List[str]:
    """Parse --subjects / --ci / --full CLI args. Defaults to CI subjects."""
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _args, _ = _parser.parse_known_args()
    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    # Default: CI subjects (safe for server with limited symlinks)
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
# MAE-specific hyperparameters (injected alongside TrainingConfig)
# ---------------------------------------------------------------------------

MAE_CFG = {
    "patch_size": 20,           # temporal patch size (T=600 → 30 patches)
    "d_model": 128,             # encoder embedding dimension
    "encoder_depth": 4,
    "encoder_heads": 4,
    "decoder_depth": 2,
    "decoder_heads": 4,
    "decoder_d_model": 64,
    "mask_ratio": 0.4,          # 40% masking during pretraining
    "pretrain_epochs": 30,
    "pretrain_lr": 1e-3,
    "pretrain_batch_size": 512,
    "finetune_lr": 5e-4,        # lower LR for fine-tuning pretrained encoder
    "freeze_encoder_epochs": 0, # 0 = fully trainable from the start of fine-tune
}


# ---------------------------------------------------------------------------
# Custom MAE trainer
# ---------------------------------------------------------------------------

class MAETrainer(WindowClassifierTrainer):
    """
    Two-phase trainer:
      Phase 1 — pretraining with MAEEmgForPretraining (self-supervised, unlabeled).
      Phase 2 — fine-tuning MAEEmgForClassification (supervised, with pretrained encoder).

    Inherits from WindowClassifierTrainer so evaluate_numpy() and all parent
    utilities work without changes.
    """

    def __init__(self, mae_cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.mae_cfg = mae_cfg

    # ------------------------------------------------------------------
    # Phase 1: pretraining
    # ------------------------------------------------------------------

    def _pretrain(
        self,
        X_all: np.ndarray,      # (N, C, T) — combined train+val, standardised
        in_channels: int,
        time_steps: int,
    ) -> MAEEmgForPretraining:
        """Run MAE pretraining on unlabeled data. Returns trained model."""
        device = self.cfg.device
        cfg = self.mae_cfg

        pretrain_model = MAEEmgForPretraining(
            in_channels=in_channels,
            time_steps=time_steps,
            patch_size=cfg["patch_size"],
            d_model=cfg["d_model"],
            encoder_depth=cfg["encoder_depth"],
            encoder_heads=cfg["encoder_heads"],
            decoder_depth=cfg["decoder_depth"],
            decoder_heads=cfg["decoder_heads"],
            decoder_d_model=cfg["decoder_d_model"],
            mask_ratio=cfg["mask_ratio"],
            dropout=self.cfg.dropout,
        ).to(device)

        X_tensor = torch.from_numpy(X_all).float()
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(
            dataset,
            batch_size=cfg["pretrain_batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        optimizer = optim.AdamW(
            pretrain_model.parameters(),
            lr=cfg["pretrain_lr"],
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["pretrain_epochs"]
        )

        self.logger.info(
            f"[MAE Pretraining] {cfg['pretrain_epochs']} epochs, "
            f"N={len(X_all)}, patch_size={cfg['patch_size']}, "
            f"mask_ratio={cfg['mask_ratio']}"
        )

        pretrain_model.train()
        for epoch in range(1, cfg["pretrain_epochs"] + 1):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                loss, _, _ = pretrain_model(batch_x)
                loss.backward()
                nn.utils.clip_grad_norm_(pretrain_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            scheduler.step()
            avg_loss = epoch_loss / len(X_all)
            if epoch % 5 == 0 or epoch == 1:
                self.logger.info(
                    f"[MAE Pretraining] Epoch {epoch}/{cfg['pretrain_epochs']} "
                    f"— recon loss: {avg_loss:.6f}"
                )

        return pretrain_model

    # ------------------------------------------------------------------
    # Overridden fit()
    # ------------------------------------------------------------------

    def fit(self, splits: Dict) -> Dict:
        """
        Two-phase fit:
          1. Standardise data (channel-wise, computed on train only).
          2. Pretrain MAE encoder on train + val data (no labels).
          3. Initialise classifier from pretrained encoder.
          4. Fine-tune classifier on labeled train data with early stopping on val.
        """
        seed_everything(self.cfg.seed)

        # --- Prepare flat arrays (N, T, C) → transpose to (N, C, T) -------
        X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # Transpose (N, T, C) → (N, C, T) — trainer convention
        def _t(X):
            if isinstance(X, np.ndarray) and X.ndim == 3 and X.shape[1] > X.shape[2]:
                return np.transpose(X, (0, 2, 1))
            return X

        X_train = _t(X_train)
        X_val   = _t(X_val) if len(X_val) > 0 else X_val
        X_test  = _t(X_test) if len(X_test) > 0 else X_test

        in_channels = X_train.shape[1]   # C
        time_steps  = X_train.shape[2]   # T

        # Verify T is divisible by patch_size (adjust patch_size if needed)
        patch_size = self.mae_cfg["patch_size"]
        if time_steps % patch_size != 0:
            # Find largest divisor ≤ patch_size
            for ps in range(patch_size, 0, -1):
                if time_steps % ps == 0:
                    self.logger.warning(
                        f"T={time_steps} not divisible by patch_size={patch_size}. "
                        f"Adjusted patch_size to {ps}."
                    )
                    self.mae_cfg["patch_size"] = ps
                    break

        # --- Channel standardisation (on train only) ----------------------
        mean_c, std_c = self._compute_channel_standardization(X_train)
        self.mean_c = mean_c
        self.std_c  = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = time_steps

        X_train_n = self._apply_standardization(X_train, mean_c, std_c)
        X_val_n   = self._apply_standardization(X_val, mean_c, std_c) if len(X_val) > 0 else X_val
        X_test_n  = self._apply_standardization(X_test, mean_c, std_c) if len(X_test) > 0 else X_test

        # Save normalisation stats
        norm_path = self.output_dir / "normalization_stats.npz"
        np.savez_compressed(
            norm_path, mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32)
        )

        # --- Phase 1: Pretraining on train + val (unlabeled) ---------------
        X_pretrain = (
            np.concatenate([X_train_n, X_val_n], axis=0)
            if len(X_val_n) > 0
            else X_train_n
        )
        pretrain_model = self._pretrain(X_pretrain, in_channels, time_steps)

        # Optionally save pretrained weights
        pretrain_path = self.output_dir / "mae_pretrained.pt"
        torch.save(pretrain_model.state_dict(), pretrain_path)
        self.logger.info(f"[MAE] Pretrained weights saved: {pretrain_path}")

        # --- Phase 2: Fine-tuning -----------------------------------------
        num_classes = len(class_ids)
        cfg = self.mae_cfg
        device = self.cfg.device

        finetune_model = MAEEmgForClassification(
            in_channels=in_channels,
            num_classes=num_classes,
            time_steps=time_steps,
            patch_size=cfg["patch_size"],
            d_model=cfg["d_model"],
            encoder_depth=cfg["encoder_depth"],
            encoder_heads=cfg["encoder_heads"],
            dropout=self.cfg.dropout,
        ).to(device)

        # Load pretrained encoder into the classifier
        finetune_model.load_pretrained_encoder(pretrain_model)
        del pretrain_model  # free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(
            f"[MAE Fine-tuning] {self.cfg.epochs} epochs, "
            f"LR={cfg['finetune_lr']}, classes={num_classes}"
        )

        # DataLoaders
        def _make_loader(X, y, shuffle: bool) -> DataLoader:
            ds = TensorDataset(
                torch.from_numpy(X).float(),
                torch.from_numpy(y).long(),
            )
            return DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=True,
            )

        dl_train = _make_loader(X_train_n, y_train, shuffle=True)
        dl_val   = _make_loader(X_val_n, y_val, shuffle=False) if len(X_val_n) > 0 else None

        # Class weights
        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            weights = counts.sum() / (counts + 1e-8)
            weights /= weights.mean()
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(weights).float().to(device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(
            finetune_model.parameters(),
            lr=cfg["finetune_lr"],
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = self.cfg.early_stopping_patience

        for epoch in range(1, self.cfg.epochs + 1):
            # Train
            finetune_model.train()
            train_loss = 0.0
            train_correct = 0
            for batch_x, batch_y in dl_train:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = finetune_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(finetune_model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
                train_correct += (logits.argmax(1) == batch_y).sum().item()

            train_loss /= len(y_train)
            train_acc = train_correct / len(y_train)

            # Validation
            if dl_val is not None:
                finetune_model.eval()
                val_loss = 0.0
                val_correct = 0
                with torch.no_grad():
                    for batch_x, batch_y in dl_val:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        logits = finetune_model(batch_x)
                        val_loss += criterion(logits, batch_y).item() * len(batch_x)
                        val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_loss /= len(y_val)
                val_acc = val_correct / len(y_val)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in finetune_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 5 == 0 or epoch == 1:
                    self.logger.info(
                        f"[MAE FT] Epoch {epoch}/{self.cfg.epochs} "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )

                if patience_counter >= patience:
                    self.logger.info(f"[MAE FT] Early stopping at epoch {epoch}.")
                    break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    self.logger.info(
                        f"[MAE FT] Epoch {epoch}/{self.cfg.epochs} "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                    )

        # Restore best weights
        if best_state is not None:
            finetune_model.load_state_dict(best_state)
            self.logger.info(f"[MAE FT] Restored best val_loss={best_val_loss:.4f}")

        self.model = finetune_model
        return {}


# ---------------------------------------------------------------------------
# LOSO fold runner
# ---------------------------------------------------------------------------

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
    mae_cfg: dict,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.model_type = model_type
    train_cfg.pipeline_type = "deep_raw"
    train_cfg.use_handcrafted_features = False

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "mae_config.json", "w") as f:
        json.dump(mae_cfg, f, indent=4)

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

    base_viz = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    trainer = MAETrainer(
        mae_cfg=mae_cfg,
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
            "model_type": model_type,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1  = float(test_metrics.get("f1_macro", 0.0))

    logger.info(
        f"[LOSO] Test={test_subject} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )
    print(
        f"[LOSO] Test={test_subject} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        make_json_serializable({
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "model_type": model_type,
            "exercises": exercises,
            "mae_cfg": mae_cfg,
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
        "model_type": model_type,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPERIMENT_NAME = "exp_35_mae_ssl_pretrain_loso"
    BASE_DIR   = ROOT / "data"
    ALL_SUBJECTS = parse_subjects_args()
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_" +
                      "_".join(s.split("_s")[1] for s in ALL_SUBJECTS))

    EXERCISES  = ["E1"]
    MODEL_TYPES = ["mae_emg"]   # registered in models/__init__ via register_model

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
        epochs=50,
        learning_rate=5e-4,      # overridden per-phase inside MAETrainer
        weight_decay=1e-4,
        dropout=0.1,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
    )

    # MAE-specific hyperparams
    mae_cfg = dict(MAE_CFG)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"EXPERIMENT: {EXPERIMENT_NAME}")
    global_logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    global_logger.info(f"Device: {train_cfg.device}")
    global_logger.info(f"MAE config: {mae_cfg}")
    global_logger.info("=" * 80)

    all_loso_results = []

    for model_type in MODEL_TYPES:
        print(f"\nMODEL: {model_type} — starting LOSO over {len(ALL_SUBJECTS)} subjects")
        for test_subject in ALL_SUBJECTS:
            print(f"  LOSO fold: test_subject={test_subject}")
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"

            try:
                fold_res = run_single_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_output_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    model_type=model_type,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                    mae_cfg=dict(mae_cfg),  # pass a fresh copy per fold
                )
                all_loso_results.append(fold_res)
                acc = fold_res["test_accuracy"]
                f1  = fold_res["test_f1_macro"]
                acc_str = f"{acc:.4f}" if acc is not None else "None"
                f1_str  = f"{f1:.4f}" if f1 is not None else "None"
                print(f"  ✓ acc={acc_str}, f1={f1_str}")
            except Exception as e:
                global_logger.error(f"✗ Failed fold test={test_subject}: {e}")
                global_logger.error(traceback.format_exc())
                all_loso_results.append({
                    "test_subject": test_subject,
                    "model_type": model_type,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": str(e),
                })

    # --- Aggregate ---
    aggregate_results = {}
    for model_type in MODEL_TYPES:
        model_results = [
            r for r in all_loso_results
            if r["model_type"] == model_type and r.get("test_accuracy") is not None
        ]
        if not model_results:
            continue
        accs = [r["test_accuracy"] for r in model_results]
        f1s  = [r["test_f1_macro"]  for r in model_results]
        aggregate_results[model_type] = {
            "mean_accuracy":  float(np.mean(accs)),
            "std_accuracy":   float(np.std(accs)),
            "mean_f1_macro":  float(np.mean(f1s)),
            "std_f1_macro":   float(np.std(f1s)),
            "num_subjects":   len(accs),
            "per_subject":    model_results,
        }
        r = aggregate_results[model_type]
        print(
            f"\n{model_type}: "
            f"Acc={r['mean_accuracy']:.4f}±{r['std_accuracy']:.4f}, "
            f"F1={r['mean_f1_macro']:.4f}±{r['std_f1_macro']:.4f} "
            f"(n={r['num_subjects']})"
        )
        global_logger.info(
            f"{model_type}: "
            f"Acc={r['mean_accuracy']:.4f}±{r['std_accuracy']:.4f}, "
            f"F1={r['mean_f1_macro']:.4f}±{r['std_f1_macro']:.4f}"
        )

    # --- Save summary ---
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "mae_cfg": mae_cfg,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)

    global_logger.info(f"EXPERIMENT COMPLETE. Results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
