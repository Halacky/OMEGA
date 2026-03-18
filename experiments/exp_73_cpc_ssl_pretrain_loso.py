"""
Experiment 73: CPC / wav2vec2-style Self-Supervised Pretraining for EMG (LOSO)

Hypothesis:
    Contrastive Predictive Coding (CPC) — predicting future latent
    representations from causal context — learns temporally discriminative
    EMG representations better than MAE-style reconstruction (exp 35),
    because the prediction target is closer to the discriminative temporal
    structure of muscle activation sequences.

Method (strictly LOSO, no test-subject adaptation):
    For each LOSO fold:
        Phase 1  — CPC Pretraining (self-supervised, no gesture labels):
            · SSL train on ALL train-subject data (train + val splits pooled)
            · Encoder: strided Conv1D  (B, C, T) → (B, d_enc, T')
            · Context: causal GRU      (B, T', d_enc) → (B, T', d_ctx)
            · K predictors: W_k(c_t) → pred z_{t+k}
            · InfoNCE loss with batch negatives
            · Optional: Gumbel-VQ for discrete targets  (wav2vec2 style)

        Phase 2  — Supervised Fine-tuning (gesture labels):
            · Load pretrained encoder weights
            · Fine-tune encoder + classification head on labeled train split
            · Early stopping on labeled val split
            · Test split = held-out subject (never seen during any training)

Data-leakage audit:
    ✓ Channel standardisation: μ/σ computed on train windows only,
      applied to val/test.
    ✓ CPC pretraining: uses only train + val windows (from train subjects).
      No gesture labels consumed.  Test-subject windows never seen.
    ✓ Fine-tuning: uses train windows + labels; val used for early stopping.
      Test-subject windows never seen until evaluation.
    ✓ Evaluation: test-subject windows processed with the frozen μ/σ from
      training.  No parameters are updated after evaluating the test set.

Baseline: exp_35 (MAE SSL pretrain) — both are two-phase SSL + fine-tune.
Expected improvement: CPC's discriminative predictive task > MAE's
    reconstructive task for building gesture-relevant representations.

Architecture parameters (CPC_CFG):
    d_enc            encoder dimension               (256)
    d_ctx            GRU context dimension            (256)
    K                prediction steps                  (12)
    use_quantizer    Gumbel-VQ on targets           (False)
    num_vars         VQ codebook size per group       (320)
    num_groups       VQ product quantization groups     (2)
    pretrain_epochs  CPC pretraining epochs            (50)
    pretrain_lr      CPC Adam learning rate           (1e-3)
    pretrain_batch   CPC batch size                   (256)
    vq_diversity_w   weight for VQ diversity loss     (0.1)
    finetune_lr      fine-tune learning rate          (5e-4)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

from models.cpc_emg import CPCPretrainModel, CPCClassifier
from models import register_model

register_model("cpc_emg", CPCClassifier)


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


def _parse_subjects() -> List[str]:
    """Parse --subjects / --ci / --full CLI args. Defaults to CI subjects."""
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci",   action="store_true")
    _parser.add_argument("--full", action="store_true")
    _args, _ = _parser.parse_known_args()
    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    # Default: CI subjects (server has symlinks only for these 5)
    return _CI_SUBJECTS


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# CPC default hyperparameters
# ---------------------------------------------------------------------------

CPC_CFG: Dict = {
    # Encoder / context
    "d_enc":          256,    # encoder latent dimension
    "d_ctx":          256,    # GRU context dimension
    "K":               12,    # prediction steps (k=1..K)

    # Quantizer (wav2vec2-style targets) — disabled by default
    "use_quantizer":  False,
    "num_vars":        320,   # codebook size per group
    "num_groups":        2,   # product quantization groups
    "vq_diversity_w":  0.1,   # weight for VQ diversity loss

    # Pretraining
    "pretrain_epochs":  50,
    "pretrain_lr":    1e-3,
    "pretrain_batch": 256,

    # Fine-tuning
    "finetune_lr":    5e-4,
}


# ---------------------------------------------------------------------------
# CPC Trainer
# ---------------------------------------------------------------------------

class CPCTrainer(WindowClassifierTrainer):
    """
    Two-phase LOSO trainer implementing CPC pretraining + supervised fine-tune.

    Phase 1 — CPC Pretraining (self-supervised, no labels):
        · Data:  X_train + X_val  (both from train subjects ONLY)
        · Model: CPCPretrainModel  (encoder + AR context + K predictors)
        · Loss:  InfoNCE with batch negatives
                 + optional VQ diversity penalty

    Phase 2 — Supervised Fine-tuning:
        · Data:  X_train (with labels), early stop on X_val
        · Model: CPCClassifier  (encoder initialised from Phase 1)
        · Loss:  CrossEntropyLoss (optionally class-weighted)

    Standardisation computed on X_train only, applied to val and test.
    Test-subject windows are never passed to any training method.
    """

    def __init__(self, cpc_cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.cpc_cfg = cpc_cfg

    # ------------------------------------------------------------------
    # Phase 1: CPC pretraining
    # ------------------------------------------------------------------

    def _pretrain_cpc(
        self,
        X_pretrain: np.ndarray,   # (N, C, T) — train+val, already normalised
        in_channels: int,
        time_steps: int,
    ) -> CPCPretrainModel:
        """
        Train CPCPretrainModel on unlabelled data from train subjects.

        X_pretrain must contain windows from TRAIN SUBJECTS ONLY.
        The test-subject split is NEVER passed here.

        Args:
            X_pretrain:  (N, C, T)  train+val windows (normalised, no labels)
            in_channels: C
            time_steps:  T

        Returns:
            Trained CPCPretrainModel (to be used for weight transfer).
        """
        cfg = self.cpc_cfg
        device = self.cfg.device

        pretrain_model = CPCPretrainModel(
            in_channels=in_channels,
            d_enc=cfg["d_enc"],
            d_ctx=cfg["d_ctx"],
            K=cfg["K"],
            use_quantizer=cfg["use_quantizer"],
            num_vars=cfg["num_vars"],
            num_groups=cfg["num_groups"],
            dropout=self.cfg.dropout,
        ).to(device)

        # Verify encoder output length is at least K steps
        T_prime = pretrain_model.encoder.output_length(time_steps)
        if T_prime <= cfg["K"]:
            # Reduce K to fit the available temporal resolution
            new_K = max(1, T_prime - 1)
            self.logger.warning(
                f"[CPC Pretrain] T'={T_prime} ≤ K={cfg['K']}. "
                f"Reducing K to {new_K}."
            )
            # Rebuild with adjusted K
            pretrain_model = CPCPretrainModel(
                in_channels=in_channels,
                d_enc=cfg["d_enc"],
                d_ctx=cfg["d_ctx"],
                K=new_K,
                use_quantizer=cfg["use_quantizer"],
                num_vars=cfg["num_vars"],
                num_groups=cfg["num_groups"],
                dropout=self.cfg.dropout,
            ).to(device)
            self.cpc_cfg = dict(cfg)
            self.cpc_cfg["K"] = new_K

        dataset = TensorDataset(torch.from_numpy(X_pretrain).float())
        loader = DataLoader(
            dataset,
            batch_size=cfg["pretrain_batch"],
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,   # keep batch size constant for InfoNCE score matrix
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
            f"[CPC Pretrain] epochs={cfg['pretrain_epochs']}, "
            f"N={len(X_pretrain)}, T'={T_prime}, K={self.cpc_cfg['K']}, "
            f"use_quantizer={cfg['use_quantizer']}, "
            f"batch={cfg['pretrain_batch']}"
        )

        pretrain_model.train()
        for epoch in range(1, cfg["pretrain_epochs"] + 1):
            epoch_loss = 0.0
            n_batches = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()
                cpc_loss, div_loss = pretrain_model(batch_x)

                loss = cpc_loss
                if div_loss is not None:
                    loss = loss + cfg["vq_diversity_w"] * div_loss

                loss.backward()
                nn.utils.clip_grad_norm_(pretrain_model.parameters(), 1.0)
                optimizer.step()

                # Anneal VQ temperature if quantizer is active
                if pretrain_model.quantizer is not None:
                    pretrain_model.quantizer.decay_temperature()

                epoch_loss += cpc_loss.item()
                n_batches += 1

            scheduler.step()

            if epoch % 10 == 0 or epoch == 1:
                avg_loss = epoch_loss / max(n_batches, 1)
                self.logger.info(
                    f"[CPC Pretrain] Epoch {epoch}/{cfg['pretrain_epochs']} "
                    f"— InfoNCE: {avg_loss:.4f}"
                )

        return pretrain_model

    # ------------------------------------------------------------------
    # Overridden fit()
    # ------------------------------------------------------------------

    def fit(self, splits: Dict) -> Dict:
        """
        Two-phase fit following strict LOSO protocol.

        Data-leakage guarantees:
          1. μ/σ computed from X_train ONLY → never from val/test.
          2. CPC pretrain on X_train + X_val (train subjects), no labels.
             Test-subject data (splits["test"]) is NOT used here.
          3. Fine-tune classifier on X_train with labels.
             Val used for early stopping ONLY.
          4. Test data processed with the frozen μ/σ from step 1.

        Args:
            splits: Dict[str, Dict[int, np.ndarray]]
                "train" / "val" / "test" → {gesture_id: (N, T, C) array}
                (test only used later in evaluate_numpy, not here)

        Returns:
            {} (empty dict; metrics are collected by CrossSubjectExperiment)
        """
        seed_everything(self.cfg.seed)

        # ── 1. Convert split dicts to flat arrays (N, T, C) ─────────────────
        X_train, y_train, X_val, y_val, _X_test, _y_test, class_ids, class_names = \
            self._prepare_splits_arrays(splits)

        # ── 2. Transpose (N, T, C) → (N, C, T)  [PyTorch convention] ────────
        def _to_nct(arr: np.ndarray) -> np.ndarray:
            # _prepare_splits_arrays returns (N, T, C); T=600, C=8 → dim1>dim2
            if arr.ndim == 3 and arr.shape[1] > arr.shape[2]:
                return np.transpose(arr, (0, 2, 1))   # (N, C, T)
            return arr

        X_train_nct = _to_nct(X_train)   # (N, C, T)
        X_val_nct   = _to_nct(X_val)     # (M, C, T) or empty

        in_channels = X_train_nct.shape[1]   # C
        time_steps  = X_train_nct.shape[2]   # T

        # ── 3. Channel-wise standardisation — computed on train ONLY ─────────
        # This is the single source of normalisation statistics.  Val and test
        # are normalised with these same statistics so no test info leaks.
        mean_c, std_c = self._compute_channel_standardization(X_train_nct)
        self.mean_c      = mean_c
        self.std_c       = std_c
        self.class_ids   = class_ids
        self.class_names = class_names
        self.in_channels = in_channels
        self.window_size = time_steps

        X_train_n = self._apply_standardization(X_train_nct, mean_c, std_c)
        X_val_n   = (self._apply_standardization(X_val_nct, mean_c, std_c)
                     if len(X_val_nct) > 0 else X_val_nct)

        # Save normalisation stats for reproducibility
        np.savez_compressed(
            self.output_dir / "normalization_stats.npz",
            mean=mean_c, std=std_c,
            class_ids=np.array(class_ids, dtype=np.int32),
        )

        # ── 4. Phase 1: CPC Pretraining ───────────────────────────────────────
        # Pool train + val windows (both from train subjects, no labels used).
        # The test subject's windows are in splits["test"] but are NOT included.
        if len(X_val_n) > 0:
            X_pretrain = np.concatenate([X_train_n, X_val_n], axis=0)
        else:
            X_pretrain = X_train_n

        pretrain_model = self._pretrain_cpc(X_pretrain, in_channels, time_steps)

        # Save pretrained weights
        pretrain_path = self.output_dir / "cpc_pretrained.pt"
        torch.save(pretrain_model.state_dict(), pretrain_path)
        self.logger.info(f"[CPC] Pretrained weights saved: {pretrain_path}")

        # ── 5. Phase 2: Supervised fine-tuning ───────────────────────────────
        cfg    = self.cpc_cfg
        device = self.cfg.device
        num_classes = len(class_ids)

        finetune_model = CPCClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            d_enc=cfg["d_enc"],
            dropout=self.cfg.dropout,
        ).to(device)

        # Transfer pretrained encoder weights
        finetune_model.load_pretrained_encoder(pretrain_model)
        del pretrain_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(
            f"[CPC Fine-tune] classes={num_classes}, "
            f"epochs={self.cfg.epochs}, LR={cfg['finetune_lr']}"
        )

        def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
            ds = TensorDataset(
                torch.from_numpy(X).float(),
                torch.from_numpy(y).long(),
            )
            return DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

        dl_train = _loader(X_train_n, y_train, shuffle=True)
        dl_val   = (_loader(X_val_n, y_val, shuffle=False)
                    if len(X_val_n) > 0 else None)

        # Optional class weighting
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
            finetune_model.parameters(),
            lr=cfg["finetune_lr"],
            weight_decay=self.cfg.weight_decay,
        )
        # ReduceLROnPlateau without verbose= (removed in PyTorch 2.4+)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        best_state    = None
        patience_ctr  = 0
        patience      = self.cfg.early_stopping_patience

        for epoch in range(1, self.cfg.epochs + 1):
            # ── Train step ──────────────────────────────────────────────────
            finetune_model.train()
            train_loss = 0.0
            train_correct = 0
            for batch_x, batch_y in dl_train:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = finetune_model(batch_x)
                loss   = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(finetune_model.parameters(), 1.0)
                optimizer.step()
                train_loss    += loss.item() * len(batch_x)
                train_correct += (logits.argmax(1) == batch_y).sum().item()

            train_loss /= len(y_train)
            train_acc   = train_correct / len(y_train)

            # ── Validation step ─────────────────────────────────────────────
            if dl_val is not None:
                finetune_model.eval()
                val_loss    = 0.0
                val_correct = 0
                with torch.no_grad():
                    for batch_x, batch_y in dl_val:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        logits   = finetune_model(batch_x)
                        val_loss += criterion(logits, batch_y).item() * len(batch_x)
                        val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_loss /= len(y_val)
                val_acc   = val_correct / len(y_val)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in finetune_model.state_dict().items()
                    }
                    patience_ctr = 0
                else:
                    patience_ctr += 1

                if epoch % 5 == 0 or epoch == 1:
                    self.logger.info(
                        f"[CPC FT] Epoch {epoch}/{self.cfg.epochs} "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )

                if patience_ctr >= patience:
                    self.logger.info(f"[CPC FT] Early stopping at epoch {epoch}.")
                    break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    self.logger.info(
                        f"[CPC FT] Epoch {epoch}/{self.cfg.epochs} "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                    )

        # Restore best checkpoint
        if best_state is not None:
            finetune_model.load_state_dict(best_state)
            self.logger.info(
                f"[CPC FT] Restored best checkpoint "
                f"(val_loss={best_val_loss:.4f})"
            )

        self.model = finetune_model
        return {}


# ---------------------------------------------------------------------------
# Single LOSO fold runner
# ---------------------------------------------------------------------------

def run_single_loso_fold(
    base_dir:       Path,
    output_dir:     Path,
    train_subjects: List[str],
    test_subject:   str,
    exercises:      List[str],
    proc_cfg:       ProcessingConfig,
    split_cfg:      SplitConfig,
    train_cfg:      TrainingConfig,
    cpc_cfg:        dict,
) -> Dict:
    """
    Run one LOSO fold with CPC pretraining + supervised fine-tuning.

    train_subjects — subjects used for SSL pretraining and supervised training.
    test_subject   — held-out subject evaluated ONLY after training completes.

    No data from test_subject influences any model parameter or normalisation
    statistic during this function.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.model_type      = "cpc_emg"
    train_cfg.pipeline_type   = "deep_raw"
    train_cfg.use_handcrafted_features = False

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)
    with open(output_dir / "cpc_config.json", "w") as f:
        json.dump(cpc_cfg, f, indent=4)

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

    trainer = CPCTrainer(
        cpc_cfg=cpc_cfg,
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
            "test_subject":   test_subject,
            "test_accuracy":  None,
            "test_f1_macro":  None,
            "error":          str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy",  0.0))
    test_f1  = float(test_metrics.get("f1_macro",  0.0))

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
        json.dump(_make_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        _make_serializable({
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "exercises":      exercises,
            "cpc_cfg":        cpc_cfg,
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
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
        "test_subject":  test_subject,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPERIMENT_NAME = "exp_73_cpc_ssl_pretrain_loso"
    BASE_DIR     = ROOT / "data"
    ALL_SUBJECTS = _parse_subjects()
    OUTPUT_DIR   = Path(
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
        epochs=60,
        learning_rate=5e-4,        # overridden per phase inside CPCTrainer
        weight_decay=1e-4,
        dropout=0.1,
        early_stopping_patience=8,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
    )

    cpc_cfg = dict(CPC_CFG)   # fresh copy; modified per fold if K auto-adjusted

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    global_logger.info("=" * 80)
    global_logger.info(f"EXPERIMENT: {EXPERIMENT_NAME}")
    global_logger.info(f"Subjects ({len(ALL_SUBJECTS)}): {ALL_SUBJECTS}")
    global_logger.info(f"Device: {train_cfg.device}")
    global_logger.info(f"CPC config: {cpc_cfg}")
    global_logger.info("=" * 80)

    all_loso_results: List[Dict] = []

    for test_subject in ALL_SUBJECTS:
        print(f"\n  LOSO fold: test_subject={test_subject}")
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_dir = OUTPUT_DIR / f"test_{test_subject}"

        try:
            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                cpc_cfg=dict(cpc_cfg),   # fresh copy per fold
            )
            all_loso_results.append(fold_res)
            acc = fold_res["test_accuracy"]
            f1  = fold_res["test_f1_macro"]
            acc_str = f"{acc:.4f}" if acc is not None else "None"
            f1_str  = f"{f1:.4f}"  if f1  is not None else "None"
            print(f"  ✓ test={test_subject}  acc={acc_str}  f1={f1_str}")
        except Exception as e:
            global_logger.error(f"✗ Failed fold test={test_subject}: {e}")
            global_logger.error(traceback.format_exc())
            all_loso_results.append({
                "test_subject":  test_subject,
                "test_accuracy": None,
                "test_f1_macro": None,
                "error":         str(e),
            })

    # ── Aggregate across folds ──────────────────────────────────────────────
    valid = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"]  for r in valid]
        summary_line = (
            f"LOSO summary: "
            f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
            f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} "
            f"(n={len(valid)} folds)"
        )
        print(f"\n{summary_line}")
        global_logger.info(summary_line)
    else:
        global_logger.warning("No successful folds to aggregate.")

    summary = {
        "experiment_name":   EXPERIMENT_NAME,
        "subjects":          ALL_SUBJECTS,
        "exercises":         EXERCISES,
        "cpc_cfg":           cpc_cfg,
        "processing_config": asdict(proc_cfg),
        "split_config":      asdict(split_cfg),
        "training_config":   asdict(train_cfg),
        "aggregate": {
            "mean_accuracy": float(np.mean(accs)) if valid else None,
            "std_accuracy":  float(np.std(accs))  if valid else None,
            "mean_f1_macro": float(np.mean(f1s))  if valid else None,
            "std_f1_macro":  float(np.std(f1s))   if valid else None,
            "n_folds":       len(valid),
        },
        "per_fold":        all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_make_serializable(summary), f, indent=4, ensure_ascii=False)

    global_logger.info(
        f"EXPERIMENT COMPLETE. Results saved to: {OUTPUT_DIR.resolve()}"
    )

    # Optional: report to hypothesis executor if available
    try:
        from hypothesis_executor import mark_hypothesis_verified
        if valid:
            mark_hypothesis_verified(
                "H_CPC_SSL",
                metrics={
                    "mean_accuracy": float(np.mean(accs)),
                    "mean_f1_macro": float(np.mean(f1s)),
                    "n_folds":       len(valid),
                },
                experiment_name=EXPERIMENT_NAME,
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
