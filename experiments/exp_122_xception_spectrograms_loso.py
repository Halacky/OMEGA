"""
Experiment 122: Xception Backbone for EMG Spectrograms (MFCC / MDCT)

Hypothesis:
    The current MFCCCNNClassifier is a simple 3-block CNN without residual
    connections. Xception (Chollet, CVPR 2017) — depthwise separable convolutions
    with residual connections + SE attention — should extract better features
    from the same spectrograms.

    Current best CNN results (E1 only, 20-subject LOSO):
      - MFCC fmax=1000 + simple CNN: 36.73% (exp_119 B) / 36.89% (exp_121 D)
      - MDCT + simple CNN: 36.87% (exp_117 D)

    If the CNN architecture is the bottleneck, Xception should improve these.

What is tested:
    Four configurations:
      A) MFCC fmax=1000 + Xception [32,64,128,256] — main test
      B) MDCT + Xception [32,64,128,256] — second best representation
      C) MFCC fmax=500  + Xception — does Xception help even without fullband?
      D) MFCC fmax=1000 + Xception [64,128,256,512] — wider channels

Run examples:
    python experiments/exp_122_xception_spectrograms_loso.py --ci
    python experiments/exp_122_xception_spectrograms_loso.py --full
    python experiments/exp_122_xception_spectrograms_loso.py --full --config A
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
from models.xception_emg import XceptionEMG
from training.mfcc_trainer import _TensorDataset
from training.trainer import get_worker_init_fn, seed_everything
from visualization.base import Visualizer
from utils.logging import setup_logging

from sklearn.metrics import accuracy_score, f1_score, classification_report

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_122_xception_spectrograms"
EXERCISES       = ["E1"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

CONFIGS = {
    "A": {
        "feature_type": "mfcc", "fmax": 1000.0, "n_mels": 26,
        "channels": [32, 64, 128, 256], "use_deltas": True,
        "label": "MFCC fmax=1000 + Xception [32,64,128,256]",
    },
    "B": {
        "feature_type": "mdct", "fmax": 1000.0, "n_mels": 26,
        "channels": [32, 64, 128, 256], "use_deltas": False,
        "label": "MDCT (no Δ) + Xception [32,64,128,256]",
    },
    "C": {
        "feature_type": "mfcc", "fmax": 500.0, "n_mels": 26,
        "channels": [32, 64, 128, 256], "use_deltas": True,
        "label": "MFCC fmax=500 + Xception [32,64,128,256]",
    },
    "D": {
        "feature_type": "mfcc", "fmax": 1000.0, "n_mels": 26,
        "channels": [64, 128, 256, 512], "use_deltas": True,
        "label": "MFCC fmax=1000 + Xception [64,128,256,512]",
    },
}


# ============================== SPLITS BUILDER ===============================

def _build_splits(
    subjects_data, train_subjects, test_subject, common_gestures,
    multi_loader, val_ratio=0.15, seed=42,
):
    rng = np.random.RandomState(seed)
    train_dict = {gid: [] for gid in common_gestures}
    for sid in sorted(train_subjects):
        if sid not in subjects_data:
            continue
        _, _, gw = subjects_data[sid]
        filtered = multi_loader.filter_by_gestures(gw, common_gestures)
        for gid, reps in filtered.items():
            for r in reps:
                if isinstance(r, np.ndarray) and len(r) > 0:
                    train_dict[gid].append(r)

    final_train, final_val = {}, {}
    for gid in common_gestures:
        if not train_dict[gid]:
            continue
        X = np.concatenate(train_dict[gid], axis=0)
        n = len(X)
        perm = rng.permutation(n)
        nv = max(1, int(n * val_ratio))
        final_val[gid] = X[perm[:nv]]
        final_train[gid] = X[perm[nv:]]

    final_test = {}
    if test_subject in subjects_data:
        _, _, tgw = subjects_data[test_subject]
        ft = multi_loader.filter_by_gestures(tgw, common_gestures)
        for gid, reps in ft.items():
            valid = [r for r in reps if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                final_test[gid] = np.concatenate(valid, axis=0)
    return {"train": final_train, "val": final_val, "test": final_test}


def _to_arrays(splits, gesture_ids):
    out = {}
    for sn in ("train", "val", "test"):
        Xs, ys = [], []
        for i, gid in enumerate(gesture_ids):
            if gid in splits[sn]:
                a = splits[sn][gid]
                if isinstance(a, np.ndarray) and a.ndim == 3 and len(a) > 0:
                    Xs.append(a)
                    ys.append(np.full(len(a), i, dtype=np.int64))
        out[f"X_{sn}"] = np.concatenate(Xs) if Xs else np.empty((0,))
        out[f"y_{sn}"] = np.concatenate(ys) if ys else np.empty((0,), dtype=np.int64)
    return out


# ================================ SINGLE FOLD ================================

def run_single_loso_fold(
    base_dir, output_dir, train_subjects, test_subject,
    exercises, proc_cfg, split_cfg, train_cfg, config_key="A",
):
    cfg = CONFIGS[config_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg, logger=logger,
        use_gpu=True, use_improved_processing=USE_IMPROVED,
    )

    all_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir, subject_ids=all_ids,
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
    data = _to_arrays(splits, common_gestures)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if len(X_test) == 0:
        logger.error(f"No test data for {test_subject}")
        return {"test_subject": test_subject, "config": config_key,
                "label": cfg["label"], "test_accuracy": None,
                "test_f1_macro": None, "error": "No test data"}

    # Channel standardization
    mc = X_train.mean(axis=(0, 1), keepdims=True)
    sc = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mc) / sc
    X_val = (X_val - mc) / sc if len(X_val) > 0 else X_val
    X_test = (X_test - mc) / sc

    # Feature extraction
    ext = EMGMFCCExtractor(
        sampling_rate=proc_cfg.sampling_rate, n_mfcc=13,
        n_mels=cfg["n_mels"], fmin=20.0, fmax=cfg["fmax"],
        use_deltas=cfg["use_deltas"], logger=logger,
    )

    if cfg["feature_type"] == "mdct":
        spec_fn = ext.transform_mdct_spectrogram
    else:
        spec_fn = ext.transform_spectrogram

    # (N, n_coeff, T_f, C) → (N, C, n_coeff, T_f)
    spec_train = spec_fn(X_train).transpose(0, 3, 1, 2)
    spec_val = spec_fn(X_val).transpose(0, 3, 1, 2) if len(X_val) > 0 else None
    spec_test = spec_fn(X_test).transpose(0, 3, 1, 2)

    in_ch, n_coeff, n_frames = spec_train.shape[1], spec_train.shape[2], spec_train.shape[3]
    num_classes = len(common_gestures)
    logger.info(f"Input: ({in_ch}, {n_coeff}, {n_frames}), classes={num_classes}")

    # Build Xception model
    model = XceptionEMG(
        in_channels=in_ch, n_coeff=n_coeff, n_frames=n_frames,
        num_classes=num_classes, channels=cfg["channels"],
        dropout=train_cfg.dropout, use_se=True,
    ).to(train_cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"XceptionEMG: {total_params:,} params, channels={cfg['channels']}")

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

    # Loss
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    cw = counts.sum() / (counts + 1e-8); cw /= cw.mean()
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
        model.load_state_dict(best_state); model.to(device)

    # Evaluate
    model.eval()
    ds_test = _TensorDataset(spec_test, y_test)
    dl_test = DataLoader(ds_test, batch_size=train_cfg.batch_size, shuffle=False)
    all_preds, all_y = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            all_preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
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
        "model_params": total_params,
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model, subjects_data, splits; gc.collect()

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
        model_type="xception_emg", pipeline_type="deep_mfcc",
        use_handcrafted_features=False, batch_size=64, epochs=80,
        learning_rate=5e-4, weight_decay=1e-4, dropout=0.3,
        early_stopping_patience=15, seed=42, use_class_weights=True,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : Xception backbone vs simple CNN on EMG spectrograms")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Configs    : {configs_to_run}")
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results: Dict[str, List[Dict]] = {k: [] for k in configs_to_run}

    for config_key in configs_to_run:
        cfg = CONFIGS[config_key]
        print(f"\n{'~' * 60}\nConfig {config_key}: {cfg['label']}\n{'~' * 60}")

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

    # Aggregate
    print(f"\n{'=' * 80}\nLOSO Summary — {EXPERIMENT_NAME}\n{'=' * 80}")
    summary = {"experiment": EXPERIMENT_NAME, "timestamp": TIMESTAMP,
               "subjects": ALL_SUBJECTS, "exercises": EXERCISES, "configs": {}}

    for config_key, results in all_results.items():
        valid = [r for r in results if r.get("test_accuracy") is not None]
        label = CONFIGS[config_key]["label"]
        if valid:
            accs = [r["test_accuracy"] for r in valid]
            f1s = [r["test_f1_macro"] for r in valid]
            print(f"  Config {config_key} ({label}):")
            print(f"    Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
            print(f"    F1-macro : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
            summary["configs"][config_key] = {
                "label": label, "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_folds": len(valid), "per_subject": results,
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
            mark_hypothesis_verified("H_XCEPTION_EMG", {
                "best_config": best_config,
                "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid])),
                "mean_f1_macro": float(best_f1),
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_XCEPTION_EMG", "All configs failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
