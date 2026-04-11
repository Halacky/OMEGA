"""
Experiment 115: Learnable Fourier Series Filterbank for LOSO EMG Classification

Hypothesis:
    A learnable Fourier series filterbank provides strictly more expressive
    frequency decomposition than SincNet's 2-parameter bandpass filters.
    Each filter is a weighted sum of M cosine/sine harmonics with learnable
    amplitudes AND frequencies — it can learn bandpass, notch, comb, or any
    smooth frequency response the data requires.

What is tested:
    Three configurations compared head-to-head:
      A) Fourier FB (K=32, M=8)  + PCEN + CNN-GRU  — main config
      B) Fourier FB (K=16, M=12) + PCEN + CNN-GRU  — fewer filters, more harmonics
      C) Fourier FB (K=32, M=4)  + PCEN + CNN-GRU  — more filters, fewer harmonics

    Baseline comparison: SincPCENCNNGRU (exp_61) with same encoder architecture.

LOSO protocol:
    - Channel standardization from training subjects only
    - All model parameters (filterbank + PCEN + encoder) trained on training data
    - Test subject sees frozen model, no adaptation

Run examples:
    python experiments/exp_115_learnable_fourier_loso.py --ci
    python experiments/exp_115_learnable_fourier_loso.py --ci --config A
    python experiments/exp_115_learnable_fourier_loso.py --full
"""

import gc
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_115_learnable_fourier"
EXERCISES       = ["E1"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

# Configurations: vary num_filters (K) and num_harmonics (M)
CONFIGS = {
    "A": {
        "num_filters": 32, "num_harmonics": 8,  "kernel_size": 51,
        "label": "Fourier K=32 M=8 + PCEN + CNN-GRU",
    },
    "B": {
        "num_filters": 16, "num_harmonics": 12, "kernel_size": 51,
        "label": "Fourier K=16 M=12 + PCEN + CNN-GRU",
    },
    "C": {
        "num_filters": 32, "num_harmonics": 4,  "kernel_size": 51,
        "label": "Fourier K=32 M=4 + PCEN + CNN-GRU",
    },
}

MIN_FREQ = 5.0
MAX_FREQ = 500.0
PCEN_EMA_LENGTH = 128


# ============================== SPLITS BUILDER ===============================

def _build_splits(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """Build train/val/test splits from loaded subject data (LOSO-clean)."""
    rng = np.random.RandomState(seed)

    train_dict: Dict[int, List[np.ndarray]] = {gid: [] for gid in common_gestures}
    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        _, _, grouped_windows = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(grouped_windows, common_gestures)
        for gid, reps in filtered.items():
            for rep_arr in reps:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    train_dict[gid].append(rep_arr)

    final_train: Dict[int, np.ndarray] = {}
    final_val: Dict[int, np.ndarray] = {}
    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        if len(trn_idx) > 0:
            final_train[gid] = X_gid[trn_idx]
        if len(val_idx) > 0:
            final_val[gid] = X_gid[val_idx]

    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": final_test}


# ================================ SINGLE FOLD ================================

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    config_key: str = "A",
) -> Dict:
    """Execute one LOSO fold for a given Fourier filterbank configuration."""
    cfg = CONFIGS[config_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    # Lazy import to avoid circular deps
    from training.sinc_pcen_trainer import SincPCENTrainer
    from models.learnable_fourier_filterbank import FourierPCENCNNGRU

    # Data loader
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    # Load all subjects
    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES
    )
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    # Build LOSO splits
    splits = _build_splits(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    for sname in ("train", "val", "test"):
        total = sum(len(a) for a in splits[sname].values()
                    if isinstance(a, np.ndarray) and a.ndim == 3)
        logger.info(f"  {sname.upper()}: {total} windows, {len(splits[sname])} gestures")

    # --- Use SincPCENTrainer but monkey-patch the model builder ---
    # SincPCENTrainer.fit() calls SincPCENCNNGRU internally.
    # We subclass inline to override the model creation.

    class _FourierTrainer(SincPCENTrainer):
        """SincPCENTrainer with FourierPCENCNNGRU instead of SincPCENCNNGRU."""

        def __init__(self, *args, num_harmonics: int = 8, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_harmonics = num_harmonics

        def fit(self, splits_arg: Dict) -> Dict:
            """Override fit to use FourierPCENCNNGRU."""
            seed_everything(self.cfg.seed)

            # 1. Splits -> arrays
            (X_train, y_train,
             X_val, y_val,
             X_test, y_test,
             class_ids, class_names) = self._prepare_splits_arrays(splits_arg)

            # 2. Transpose (N, T, C) -> (N, C, T)
            if X_train.ndim == 3 and X_train.shape[1] > X_train.shape[2]:
                X_train = X_train.transpose(0, 2, 1)
                if X_val.ndim == 3 and len(X_val) > 0:
                    X_val = X_val.transpose(0, 2, 1)
                if X_test.ndim == 3 and len(X_test) > 0:
                    X_test = X_test.transpose(0, 2, 1)

            in_channels = X_train.shape[1]
            window_size = X_train.shape[2]
            num_classes = len(class_ids)

            # 3. Channel standardization (train only)
            mean_c, std_c = self._compute_channel_standardization(X_train)
            X_train = self._apply_standardization(X_train, mean_c, std_c)
            if len(X_val) > 0:
                X_val = self._apply_standardization(X_val, mean_c, std_c)
            if len(X_test) > 0:
                X_test = self._apply_standardization(X_test, mean_c, std_c)

            self.logger.info("Channel standardization applied (training stats only).")

            # 4. Build Fourier model
            model = FourierPCENCNNGRU(
                in_channels=in_channels,
                num_classes=num_classes,
                num_filters=self.num_sinc_filters,
                num_harmonics=self.num_harmonics,
                kernel_size=self.sinc_kernel_size,
                sample_rate=self.sample_rate,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                pcen_ema_length=self.pcen_ema_length,
                dropout=self.cfg.dropout,
            ).to(self.cfg.device)

            total_params = sum(p.numel() for p in model.parameters())
            fb_params = sum(p.numel() for p in model.fourier_fb.parameters())
            pcen_params = sum(p.numel() for p in model.pcen.parameters())
            self.logger.info(
                f"FourierPCENCNNGRU: K={self.num_sinc_filters}, M={self.num_harmonics}, "
                f"total={total_params:,}, fourier_fb={fb_params:,}, pcen={pcen_params:,}"
            )

            # 5. Datasets
            from training.trainer import WindowDataset, get_worker_init_fn
            ds_train = WindowDataset(X_train, y_train)
            ds_val = WindowDataset(X_val, y_val) if len(X_val) > 0 else None

            g = torch.Generator().manual_seed(self.cfg.seed)
            worker_init = get_worker_init_fn(self.cfg.seed)

            from torch.utils.data import DataLoader
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

            # 6. Loss
            import torch.nn as nn
            if self.cfg.use_class_weights:
                counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
                cw = counts.sum() / (counts + 1e-8)
                cw /= cw.mean()
                criterion = nn.CrossEntropyLoss(
                    weight=torch.from_numpy(cw).float().to(self.cfg.device)
                )
            else:
                criterion = nn.CrossEntropyLoss()

            # 7. Optimizer + scheduler
            import torch.optim as optim
            optimizer = optim.Adam(
                model.parameters(), lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5,
            )

            # 8. Training loop
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

            # 9. Store state
            self.model = model
            self.mean_c = mean_c
            self.std_c = std_c
            self.class_ids = class_ids
            self.class_names = class_names
            self.in_channels = in_channels
            self.window_size = window_size

            # 10. Save
            with open(self.output_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=4)

            torch.save({
                "state_dict": model.state_dict(),
                "in_channels": in_channels,
                "num_classes": num_classes,
                "class_ids": class_ids,
                "mean": mean_c,
                "std": std_c,
            }, self.output_dir / "fourier_pcen_cnn_gru.pt")

            return {"class_ids": class_ids, "class_names": class_names}

    # --- Create trainer and run ---
    trainer = _FourierTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
        sample_rate=proc_cfg.sampling_rate,
        num_sinc_filters=cfg["num_filters"],
        sinc_kernel_size=cfg["kernel_size"],
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        pcen_ema_length=PCEN_EMA_LENGTH,
        num_harmonics=cfg["num_harmonics"],
    )

    try:
        trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "config": config_key,
            "label": cfg["label"],
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    # Test evaluation
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error(f"No test windows for {test_subject}")
        return {
            "test_subject": test_subject,
            "config": config_key,
            "label": cfg["label"],
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": "No test data",
        }

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    test_results = trainer.evaluate_numpy(
        X_test, y_test,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1 = float(test_results["f1_macro"])

    print(
        f"[LOSO] {cfg['label']} | test={test_subject} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )

    fold_summary = {
        "test_subject": test_subject,
        "config": config_key,
        "label": cfg["label"],
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "report": test_results.get("report"),
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del trainer, multi_loader, subjects_data, splits
    gc.collect()

    return {
        "test_subject": test_subject,
        "config": config_key,
        "label": cfg["label"],
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ==================================== MAIN ===================================

def main():
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _parser.add_argument("--config", type=str, default=None,
                         help="Run single config: A, B, or C. Default: all.")
    _args, _ = _parser.parse_known_args()

    if _args.subjects:
        ALL_SUBJECTS = [s.strip() for s in _args.subjects.split(",")]
    elif _args.full:
        ALL_SUBJECTS = DEFAULT_SUBJECTS
    else:
        ALL_SUBJECTS = CI_TEST_SUBJECTS

    configs_to_run = [_args.config] if _args.config else list(CONFIGS.keys())

    BASE_DIR = ROOT / "data"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"

    proc_cfg = ProcessingConfig(
        window_size=400,
        window_overlap=200,
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

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : Learnable Fourier series filterbank vs SincNet")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Configs    : {configs_to_run}")
    print(f"Freq range : [{MIN_FREQ}, {MAX_FREQ}] Hz")
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results: Dict[str, List[Dict]] = {k: [] for k in configs_to_run}

    for config_key in configs_to_run:
        cfg = CONFIGS[config_key]
        print(f"\n{'~' * 60}")
        print(f"Config {config_key}: {cfg['label']}")
        print(f"{'~' * 60}")

        train_cfg = TrainingConfig(
            model_type="fourier_pcen_cnn_gru",
            pipeline_type="deep_raw",
            use_handcrafted_features=False,
            batch_size=64,
            epochs=60,
            learning_rate=1e-3,
            weight_decay=1e-4,
            dropout=0.3,
            early_stopping_patience=12,
            seed=42,
            use_class_weights=True,
            num_workers=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_dir = OUTPUT_ROOT / f"config_{config_key}" / f"test_{test_subject}"

            result = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                config_key=config_key,
            )
            all_results[config_key].append(result)

    # ── Aggregate and report ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"LOSO Summary — {EXPERIMENT_NAME}")
    print(f"{'=' * 80}")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "configs": {},
    }

    for config_key, results in all_results.items():
        valid = [r for r in results if r.get("test_accuracy") is not None]
        label = CONFIGS[config_key]["label"]
        if valid:
            accs = [r["test_accuracy"] for r in valid]
            f1s = [r["test_f1_macro"] for r in valid]
            print(f"  Config {config_key} ({label}):")
            print(f"    Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
            print(f"    F1-macro : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
            print(f"    Folds    : {len(valid)}/{len(results)}")
            summary["configs"][config_key] = {
                "label": label,
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_folds": len(valid),
                "per_subject": results,
            }
        else:
            print(f"  Config {config_key} ({label}): ALL FOLDS FAILED")
            summary["configs"][config_key] = {
                "label": label,
                "error": "All folds failed",
                "per_subject": results,
            }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"\nSummary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        best_f1 = 0.0
        best_config = None
        for ck, results in all_results.items():
            valid = [r for r in results if r.get("test_f1_macro") is not None]
            if valid:
                mean_f1 = np.mean([r["test_f1_macro"] for r in valid])
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_config = ck
        if best_config is not None:
            valid = [r for r in all_results[best_config]
                     if r.get("test_accuracy") is not None]
            metrics = {
                "best_config": best_config,
                "best_label": CONFIGS[best_config]["label"],
                "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid])),
                "mean_f1_macro": float(best_f1),
            }
            mark_hypothesis_verified("H_FOURIER_FB", metrics,
                                     experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_FOURIER_FB", "All configurations failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
