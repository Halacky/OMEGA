"""
Experiment 121: MFCC Fullband + MixStyle for LOSO EMG Classification

Hypothesis:
    Combining the two strongest findings so far:
      1. Full spectrum (fmax=1000) gives +4.5pp for CNN (exp_119: 43.72%)
      2. Per-band MixStyle gives +1.7pp for domain generalization (H3/H5)

    MixStyle mixes subject-specific feature statistics at training time,
    breaking the association between style (subject identity) and content
    (gesture class). Applied after early CNN layers where subject style
    is most present.

What is tested:
    Four configurations varying MixStyle injection point and strength:
      A) MixStyle after block 0 only (p=0.5, α=0.1)
      B) MixStyle after blocks 0+1 (p=0.5, α=0.1)
      C) MixStyle after block 0 only (p=0.5, α=0.3) — softer mixing
      D) No MixStyle baseline — reproduction of exp_119 B for fair comparison

    All use MFCC fmax=1000, 26 mels, E1+E2.

    Target: exp_119 B (43.72%) + MixStyle boost → 44-45%+

Run examples:
    python experiments/exp_121_mfcc_fullband_mixstyle_loso.py --ci
    python experiments/exp_121_mfcc_fullband_mixstyle_loso.py --full
    python experiments/exp_121_mfcc_fullband_mixstyle_loso.py --full --config A
"""

import gc
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from processing.emg_mfcc import EMGMFCCExtractor
from models.mfcc_mixstyle_cnn import MFCCMixStyleCNN
from models.mfcc_cnn_classifier import MFCCCNNClassifier
from training.mfcc_trainer import _TensorDataset
from training.trainer import get_worker_init_fn, seed_everything
from visualization.base import Visualizer
from utils.logging import setup_logging
from utils.artifacts import ArtifactSaver

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_121_mfcc_fullband_mixstyle"
EXERCISES       = ["E1",]
USE_IMPROVED    = True
MAX_GESTURES    = 10

MFCC_N_COEFF   = 13
MFCC_N_MELS    = 26
MFCC_FMIN      = 20.0
MFCC_FMAX      = 1000.0  # full Nyquist

CONFIGS = {
    "A": {
        "use_mixstyle": True, "mixstyle_layers": [0], "mixstyle_p": 0.5,
        "mixstyle_alpha": 0.1,
        "label": "MFCC fmax=1000 + MixStyle@block0 (α=0.1)",
    },
    "B": {
        "use_mixstyle": True, "mixstyle_layers": [0, 1], "mixstyle_p": 0.5,
        "mixstyle_alpha": 0.1,
        "label": "MFCC fmax=1000 + MixStyle@block0+1 (α=0.1)",
    },
    "C": {
        "use_mixstyle": True, "mixstyle_layers": [0], "mixstyle_p": 0.5,
        "mixstyle_alpha": 0.3,
        "label": "MFCC fmax=1000 + MixStyle@block0 (α=0.3)",
    },
    "D": {
        "use_mixstyle": False, "mixstyle_layers": [], "mixstyle_p": 0.0,
        "mixstyle_alpha": 0.1,
        "label": "MFCC fmax=1000 baseline (no MixStyle)",
    },
}


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
        final_val[gid] = X_gid[perm[:n_val]]
        final_train[gid] = X_gid[perm[n_val:]]

    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in filtered_test.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)

    return {"train": final_train, "val": final_val, "test": final_test}


def _splits_to_arrays(splits, gesture_ids):
    arrays = {}
    for sname in ("train", "val", "test"):
        X_list, y_list = [], []
        for i, gid in enumerate(gesture_ids):
            if gid in splits[sname]:
                arr = splits[sname][gid]
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                    X_list.append(arr)
                    y_list.append(np.full(len(arr), i, dtype=np.int64))
        if X_list:
            arrays[f"X_{sname}"] = np.concatenate(X_list)
            arrays[f"y_{sname}"] = np.concatenate(y_list)
        else:
            arrays[f"X_{sname}"] = np.empty((0,))
            arrays[f"y_{sname}"] = np.empty((0,), dtype=np.int64)
    return arrays


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
    cfg = CONFIGS[config_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg, logger=logger,
        use_gpu=True, use_improved_processing=USE_IMPROVED,
    )

    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir, subject_ids=all_subject_ids,
        exercises=exercises, include_rest=split_cfg.include_rest_in_splits,
    )
    common_gestures = sorted(multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES,
    ))
    logger.info(f"Gestures ({len(common_gestures)}): {common_gestures}")

    splits = _build_splits(
        subjects_data, train_subjects, test_subject,
        common_gestures, multi_loader,
        val_ratio=split_cfg.val_ratio, seed=train_cfg.seed,
    )
    data = _splits_to_arrays(splits, common_gestures)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if len(X_test) == 0:
        logger.error(f"No test data for {test_subject}")
        return {"test_subject": test_subject, "config": config_key,
                "label": cfg["label"], "test_accuracy": None, "test_f1_macro": None,
                "error": "No test data"}

    # Channel standardization (train only)
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean_c) / std_c
    X_val = (X_val - mean_c) / std_c if len(X_val) > 0 else X_val
    X_test = (X_test - mean_c) / std_c

    # MFCC extraction
    mfcc_ext = EMGMFCCExtractor(
        sampling_rate=proc_cfg.sampling_rate, n_mfcc=MFCC_N_COEFF,
        n_mels=MFCC_N_MELS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
        use_deltas=True, logger=logger,
    )
    logger.info(f"MFCC: n_mfcc={MFCC_N_COEFF}, n_mels={MFCC_N_MELS}, fmax={MFCC_FMAX}")

    # (N, n_coeff, T_f, C) → (N, C, n_coeff, T_f)
    spec_train = mfcc_ext.transform_spectrogram(X_train).transpose(0, 3, 1, 2)
    spec_val = mfcc_ext.transform_spectrogram(X_val).transpose(0, 3, 1, 2) if len(X_val) > 0 else None
    spec_test = mfcc_ext.transform_spectrogram(X_test).transpose(0, 3, 1, 2)

    in_ch = spec_train.shape[1]
    n_coeff = spec_train.shape[2]
    n_frames = spec_train.shape[3]
    num_classes = len(common_gestures)

    logger.info(f"Spectrograms: ({in_ch}, {n_coeff}, {n_frames}), classes={num_classes}")

    # Build model
    if cfg["use_mixstyle"]:
        model = MFCCMixStyleCNN(
            in_channels=in_ch, n_coeff=n_coeff, n_frames=n_frames,
            num_classes=num_classes, cnn_channels=[32, 64, 128],
            dropout=train_cfg.dropout,
            mixstyle_p=cfg["mixstyle_p"],
            mixstyle_alpha=cfg["mixstyle_alpha"],
            mixstyle_layers=cfg["mixstyle_layers"],
        ).to(train_cfg.device)
    else:
        model = MFCCCNNClassifier(
            in_channels=in_ch, n_coeff=n_coeff, n_frames=n_frames,
            num_classes=num_classes, cnn_channels=[32, 64, 128],
            dropout=train_cfg.dropout,
        ).to(train_cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.__class__.__name__}, params={total_params:,}")

    # Datasets
    ds_train = _TensorDataset(spec_train, y_train)
    ds_val = _TensorDataset(spec_val, y_val) if spec_val is not None else None

    g = torch.Generator().manual_seed(train_cfg.seed)
    dl_train = DataLoader(
        ds_train, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, pin_memory=True,
        worker_init_fn=get_worker_init_fn(train_cfg.seed) if train_cfg.num_workers > 0 else None,
        generator=g,
    )
    dl_val = DataLoader(
        ds_val, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    ) if ds_val else None

    # Loss with class weights
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    cw = counts.sum() / (counts + 1e-8)
    cw /= cw.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(cw).float().to(train_cfg.device)
    )

    optimizer = optim.Adam(model.parameters(), lr=train_cfg.learning_rate,
                           weight_decay=train_cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # Training loop
    best_val_loss, best_state, no_improve = float("inf"), None, 0
    device = train_cfg.device
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, train_cfg.epochs + 1):
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

            ep_loss += loss.item() * xb.size(0)
            ep_correct += (logits.argmax(1) == yb).sum().item()
            ep_total += xb.size(0)

        train_loss = ep_loss / max(1, ep_total)
        train_acc = ep_correct / max(1, ep_total)

        if dl_val:
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
            logger.info(
                f"[Epoch {epoch:02d}/{train_cfg.epochs}] "
                f"loss={train_loss:.4f}, acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

        if dl_val:
            scheduler.step(val_loss)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= train_cfg.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # Evaluate test
    model.eval()
    ds_test = _TensorDataset(spec_test, y_test)
    dl_test = DataLoader(ds_test, batch_size=train_cfg.batch_size, shuffle=False)
    all_preds, all_y = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_y.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_y)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"[LOSO] {cfg['label']} | test={test_subject} | Acc={acc:.4f}, F1={f1:.4f}")

    fold_summary = {
        "test_subject": test_subject, "config": config_key,
        "label": cfg["label"], "test_accuracy": float(acc),
        "test_f1_macro": float(f1), "report": report,
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model, subjects_data, splits
    gc.collect()

    return {
        "test_subject": test_subject, "config": config_key,
        "label": cfg["label"],
        "test_accuracy": float(acc), "test_f1_macro": float(f1),
    }


# ==================================== MAIN ===================================

def main():
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None)
    _parser.add_argument("--ci", action="store_true")
    _parser.add_argument("--full", action="store_true")
    _parser.add_argument("--config", type=str, default=None)
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
        window_size=400, window_overlap=200, num_channels=8,
        sampling_rate=2000, segment_edge_margin=0.1,
    )
    split_cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        mode="by_segments", shuffle_segments=True, seed=42,
        include_rest_in_splits=False,
    )
    train_cfg = TrainingConfig(
        model_type="mfcc_mixstyle_cnn", pipeline_type="deep_mfcc",
        use_handcrafted_features=False, batch_size=64, epochs=60,
        learning_rate=1e-3, weight_decay=1e-4, dropout=0.3,
        early_stopping_patience=12, seed=42, use_class_weights=True,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : MFCC fullband (fmax=1000) + MixStyle domain generalization")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Exercises  : {EXERCISES}")
    print(f"Configs    : {configs_to_run}")
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results: Dict[str, List[Dict]] = {k: [] for k in configs_to_run}

    for config_key in configs_to_run:
        cfg = CONFIGS[config_key]
        print(f"\n{'~' * 60}")
        print(f"Config {config_key}: {cfg['label']}")
        print(f"{'~' * 60}")

        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_dir = OUTPUT_ROOT / f"config_{config_key}" / f"test_{test_subject}"
            result = run_single_loso_fold(
                base_dir=BASE_DIR, output_dir=fold_dir,
                train_subjects=train_subjects, test_subject=test_subject,
                exercises=EXERCISES, proc_cfg=proc_cfg, split_cfg=split_cfg,
                train_cfg=train_cfg, config_key=config_key,
            )
            all_results[config_key].append(result)

    # ── Aggregate ──
    print(f"\n{'=' * 80}")
    print(f"LOSO Summary — {EXPERIMENT_NAME}")
    print(f"{'=' * 80}")

    summary = {
        "experiment": EXPERIMENT_NAME, "timestamp": TIMESTAMP,
        "subjects": ALL_SUBJECTS, "exercises": EXERCISES,
        "mfcc_config": {"n_mfcc": MFCC_N_COEFF, "n_mels": MFCC_N_MELS,
                        "fmin": MFCC_FMIN, "fmax": MFCC_FMAX},
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
            print(f"    vs exp_119 B (43.72%): {np.mean(accs)*100 - 43.72:+.2f}pp")
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
            summary["configs"][config_key] = {"label": label, "error": "All failed",
                                               "per_subject": results}

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"\nSummary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        best_f1, best_config = 0.0, None
        for ck, results in all_results.items():
            valid = [r for r in results if r.get("test_f1_macro") is not None]
            if valid:
                mf1 = np.mean([r["test_f1_macro"] for r in valid])
                if mf1 > best_f1: best_f1, best_config = mf1, ck
        if best_config:
            valid = [r for r in all_results[best_config] if r.get("test_accuracy") is not None]
            mark_hypothesis_verified("H_MFCC_FULLBAND_MIXSTYLE", {
                "best_config": best_config,
                "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid])),
                "mean_f1_macro": float(best_f1),
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_MFCC_FULLBAND_MIXSTYLE", "All configs failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
