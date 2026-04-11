"""
Experiment 125: Pseudo Wigner-Ville Distribution for LOSO EMG Classification

Hypothesis:
    PWVD provides maximum time-frequency resolution (no STFT trade-off).
    Tests two uses:
      1. Standalone: PWVD spectrogram → 2D CNN / PWVD flat → SVM
      2. Kitchen sink: add PWVD flat features to best combo (exp_124 A: 40.45%)

    PWVD advantages over STFT-based methods (MFCC, MDCT):
      - No window length / frequency resolution trade-off
      - Captures rapid motor unit transients with full time resolution
      - Energy-preserving (Moyal's theorem)

What is tested:
      A) PWVD flat → SVM-RBF (standalone)
      B) PWVD spectrogram → 2D CNN (standalone)
      C) ALL + PWVD flat → SVM-RBF (kitchen sink + PWVD)
      D) HC + ENT + PWVD flat → SVM-RBF (PWVD replaces MFCC/MDCT)

Run:
    python experiments/exp_125_wigner_ville_loso.py --ci
    python experiments/exp_125_wigner_ville_loso.py --full
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
    CI_TEST_SUBJECTS, DEFAULT_SUBJECTS, make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from processing.features import HandcraftedFeatureExtractor
from processing.entropy_features import EntropyComplexityExtractor
from processing.emg_mfcc import EMGMFCCExtractor
from processing.emg_chromagram import EMGChromagramExtractor
from processing.energy_cosine_spectrum import EnergyCosineSpectrumExtractor
from processing.wigner_ville import PseudoWVDExtractor
from models.mfcc_cnn_classifier import MFCCCNNClassifier
from training.mfcc_trainer import _TensorDataset
from training.trainer import get_worker_init_fn, seed_everything
from visualization.base import Visualizer
from utils.logging import setup_logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_125_wigner_ville"
EXERCISES       = ["E1"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

CONFIGS = {
    "A": {
        "mode": "ml", "features": ["pwvd"],
        "label": "PWVD flat → SVM-RBF",
    },
    "B": {
        "mode": "deep", "features": ["pwvd"],
        "label": "PWVD spectrogram → 2D-CNN",
    },
    "C": {
        "mode": "ml", "features": ["hc", "entropy", "mfcc_1000", "chroma", "mdct", "ecs", "pwvd"],
        "label": "ALL + PWVD → SVM-RBF (kitchen sink+)",
    },
    "D": {
        "mode": "ml", "features": ["hc", "entropy", "pwvd"],
        "label": "HC + ENT + PWVD → SVM-RBF",
    },
}


# ============================== HELPERS ======================================

def _build_extractors(feature_list, sr):
    exts = {}
    if "hc" in feature_list:
        exts["hc"] = HandcraftedFeatureExtractor(sampling_rate=sr, feature_set="basic_v1")
    if "entropy" in feature_list:
        exts["entropy"] = EntropyComplexityExtractor(sampling_rate=sr, order=3, delay=1)
    if "mfcc_1000" in feature_list:
        exts["mfcc_1000"] = EMGMFCCExtractor(sampling_rate=sr, n_mfcc=13, n_mels=26,
                                              fmin=20.0, fmax=1000.0, use_deltas=True)
    if "chroma" in feature_list:
        exts["chroma"] = EMGChromagramExtractor(sampling_rate=sr, use_deltas=True)
    if "mdct" in feature_list:
        exts["mdct"] = EMGMFCCExtractor(sampling_rate=sr, n_mfcc=13, n_mels=26,
                                         fmin=20.0, fmax=1000.0, use_deltas=False)
    if "ecs" in feature_list:
        exts["ecs"] = EnergyCosineSpectrumExtractor(sampling_rate=sr, n_ecs=13, use_deltas=False)
    if "pwvd" in feature_list:
        exts["pwvd"] = PseudoWVDExtractor(sampling_rate=sr, n_freq=64, window_length=51,
                                           hop=20, fmax=1000.0, use_deltas=False)
    return exts


def _extract_flat(exts, X, logger):
    parts = []
    for name, ext in exts.items():
        if name == "mdct":
            feat = ext.transform_mdct(X)
        elif name == "mfcc_1000":
            feat = ext.transform(X)
        else:
            feat = ext.transform(X)
        parts.append(feat)
        logger.info(f"  {name}: {feat.shape[1]} features")
    return np.concatenate(parts, axis=1)


def _build_splits_flat(subjects_data, train_subjects, test_subject,
                       common_gestures, multi_loader, val_ratio=0.15, seed=42):
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

    X_tr, y_tr, X_te, y_te = [], [], [], []
    for i, gid in enumerate(sorted(common_gestures)):
        if train_dict[gid]:
            X_gid = np.concatenate(train_dict[gid], axis=0)
            n = len(X_gid)
            perm = rng.permutation(n)
            nv = max(1, int(n * val_ratio))
            X_tr.append(X_gid[perm[nv:]])
            y_tr.append(np.full(n - nv, i, dtype=np.int64))

        if test_subject in subjects_data:
            _, _, tgw = subjects_data[test_subject]
            ft = multi_loader.filter_by_gestures(tgw, common_gestures)
            if gid in ft:
                valid = [r for r in ft[gid] if isinstance(r, np.ndarray) and len(r) > 0]
                if valid:
                    arr = np.concatenate(valid, axis=0)
                    X_te.append(arr)
                    y_te.append(np.full(len(arr), i, dtype=np.int64))

    return (np.concatenate(X_tr) if X_tr else np.empty((0,)),
            np.concatenate(y_tr) if y_tr else np.empty((0,), dtype=np.int64),
            np.concatenate(X_te) if X_te else np.empty((0,)),
            np.concatenate(y_te) if y_te else np.empty((0,), dtype=np.int64))


def _build_splits_dict(subjects_data, train_subjects, test_subject,
                       common_gestures, multi_loader, val_ratio=0.15, seed=42):
    """Return dict splits for deep mode (needs val set)."""
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

    X_tr, y_tr, X_val, y_val, X_te, y_te = [], [], [], [], [], []
    for i, gid in enumerate(sorted(common_gestures)):
        if train_dict[gid]:
            X_gid = np.concatenate(train_dict[gid], axis=0)
            n = len(X_gid)
            perm = rng.permutation(n)
            nv = max(1, int(n * val_ratio))
            X_val.append(X_gid[perm[:nv]])
            y_val.append(np.full(nv, i, dtype=np.int64))
            X_tr.append(X_gid[perm[nv:]])
            y_tr.append(np.full(n - nv, i, dtype=np.int64))
        if test_subject in subjects_data:
            _, _, tgw = subjects_data[test_subject]
            ft = multi_loader.filter_by_gestures(tgw, common_gestures)
            if gid in ft:
                valid = [r for r in ft[gid] if isinstance(r, np.ndarray) and len(r) > 0]
                if valid:
                    arr = np.concatenate(valid, axis=0)
                    X_te.append(arr)
                    y_te.append(np.full(len(arr), i, dtype=np.int64))

    return {
        "X_train": np.concatenate(X_tr) if X_tr else np.empty((0,)),
        "y_train": np.concatenate(y_tr) if y_tr else np.empty((0,), dtype=np.int64),
        "X_val": np.concatenate(X_val) if X_val else np.empty((0,)),
        "y_val": np.concatenate(y_val) if y_val else np.empty((0,), dtype=np.int64),
        "X_test": np.concatenate(X_te) if X_te else np.empty((0,)),
        "y_test": np.concatenate(y_te) if y_te else np.empty((0,), dtype=np.int64),
    }


# ================================ SINGLE FOLD ================================

def run_single_loso_fold(base_dir, output_dir, train_subjects, test_subject,
                         exercises, proc_cfg, split_cfg, train_cfg,
                         config_key="A", seed=42):
    cfg = CONFIGS[config_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(seed, verbose=False)

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

    if cfg["mode"] == "ml":
        return _run_ml_fold(cfg, config_key, subjects_data, train_subjects,
                            test_subject, common_gestures, multi_loader,
                            proc_cfg, split_cfg, logger, output_dir, seed)
    else:
        return _run_deep_fold(cfg, config_key, subjects_data, train_subjects,
                              test_subject, common_gestures, multi_loader,
                              proc_cfg, split_cfg, train_cfg, logger, output_dir, seed)


def _run_ml_fold(cfg, config_key, subjects_data, train_subjects, test_subject,
                 common_gestures, multi_loader, proc_cfg, split_cfg, logger,
                 output_dir, seed):
    X_train, y_train, X_test, y_test = _build_splits_flat(
        subjects_data, train_subjects, test_subject,
        common_gestures, multi_loader, val_ratio=split_cfg.val_ratio, seed=seed,
    )
    if len(X_test) == 0:
        return {"test_subject": test_subject, "config": config_key,
                "label": cfg["label"], "test_accuracy": None,
                "test_f1_macro": None, "error": "No test data"}

    mc = X_train.mean(axis=(0, 1), keepdims=True)
    sc = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_std = (X_train - mc) / sc
    X_test_std = (X_test - mc) / sc

    exts = _build_extractors(cfg["features"], proc_cfg.sampling_rate)
    feat_train = np.nan_to_num(_extract_flat(exts, X_train_std, logger))
    feat_test = np.nan_to_num(_extract_flat(exts, X_test_std, logger))
    logger.info(f"Total: {feat_train.shape[1]} features")

    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    if feat_train.shape[1] > 200:
        n_comp = min(200, feat_train.shape[0] - 1, feat_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        feat_train = pca.fit_transform(feat_train)
        feat_test = pca.transform(feat_test)

    model = svm.SVC(kernel="rbf", C=10.0, gamma="scale",
                    decision_function_shape="ovr", random_state=seed)
    model.fit(feat_train, y_train)
    y_pred = model.predict(feat_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"[LOSO] {cfg['label'][:50]} | test={test_subject} | Acc={acc:.4f}, F1={f1:.4f}")
    _save_fold(output_dir, test_subject, config_key, cfg["label"], acc, f1, report)
    del model, subjects_data; gc.collect()
    return {"test_subject": test_subject, "config": config_key,
            "label": cfg["label"], "test_accuracy": float(acc), "test_f1_macro": float(f1)}


def _run_deep_fold(cfg, config_key, subjects_data, train_subjects, test_subject,
                   common_gestures, multi_loader, proc_cfg, split_cfg, train_cfg,
                   logger, output_dir, seed):
    data = _build_splits_dict(
        subjects_data, train_subjects, test_subject,
        common_gestures, multi_loader, val_ratio=split_cfg.val_ratio, seed=seed,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if len(X_test) == 0:
        return {"test_subject": test_subject, "config": config_key,
                "label": cfg["label"], "test_accuracy": None,
                "test_f1_macro": None, "error": "No test data"}

    mc = X_train.mean(axis=(0, 1), keepdims=True)
    sc = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mc) / sc
    X_val = (X_val - mc) / sc if len(X_val) > 0 else X_val
    X_test = (X_test - mc) / sc

    pwvd_ext = PseudoWVDExtractor(sampling_rate=proc_cfg.sampling_rate,
                                   n_freq=64, window_length=51, hop=20, fmax=1000.0)

    spec_train = pwvd_ext.transform_spectrogram(X_train).transpose(0, 3, 1, 2)
    spec_val = pwvd_ext.transform_spectrogram(X_val).transpose(0, 3, 1, 2) if len(X_val) > 0 else None
    spec_test = pwvd_ext.transform_spectrogram(X_test).transpose(0, 3, 1, 2)

    in_ch, n_coeff, n_frames = spec_train.shape[1], spec_train.shape[2], spec_train.shape[3]
    num_classes = len(common_gestures)
    logger.info(f"PWVD CNN input: ({in_ch}, {n_coeff}, {n_frames}), classes={num_classes}")

    model = MFCCCNNClassifier(
        in_channels=in_ch, n_coeff=n_coeff, n_frames=n_frames,
        num_classes=num_classes, cnn_channels=[32, 64, 128],
        dropout=train_cfg.dropout,
    ).to(train_cfg.device)

    ds_train = _TensorDataset(spec_train, y_train)
    ds_val = _TensorDataset(spec_val, y_val) if spec_val is not None else None
    g = torch.Generator().manual_seed(seed)
    dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, shuffle=True,
                          num_workers=train_cfg.num_workers, pin_memory=True, generator=g)
    dl_val = DataLoader(ds_val, batch_size=train_cfg.batch_size, shuffle=False,
                        num_workers=train_cfg.num_workers, pin_memory=True) if ds_val else None

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    cw = counts.sum() / (counts + 1e-8); cw /= cw.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(cw).float().to(train_cfg.device))
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.learning_rate,
                           weight_decay=train_cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    device = train_cfg.device

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        el, ec, et = 0.0, 0, 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            el += loss.item() * xb.size(0)
            ec += (logits.argmax(1) == yb).sum().item()
            et += xb.size(0)
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
            scheduler.step(val_loss)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= train_cfg.early_stopping_patience:
                    break

    if best_state:
        model.load_state_dict(best_state); model.to(device)

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

    print(f"[LOSO] {cfg['label'][:50]} | test={test_subject} | Acc={acc:.4f}, F1={f1:.4f}")
    _save_fold(output_dir, test_subject, config_key, cfg["label"], acc, f1, report)

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    del model, subjects_data; gc.collect()
    return {"test_subject": test_subject, "config": config_key,
            "label": cfg["label"], "test_accuracy": float(acc), "test_f1_macro": float(f1)}


def _save_fold(output_dir, test_subject, config_key, label, acc, f1, report):
    fold_summary = {"test_subject": test_subject, "config": config_key,
                    "label": label, "test_accuracy": float(acc),
                    "test_f1_macro": float(f1), "report": report}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)


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
        model_type="pwvd_cnn", pipeline_type="deep_raw",
        use_handcrafted_features=False, batch_size=64, epochs=60,
        learning_rate=1e-3, weight_decay=1e-4, dropout=0.3,
        early_stopping_patience=12, seed=42, use_class_weights=True,
        num_workers=4, device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : Pseudo Wigner-Ville Distribution for EMG")
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
                train_cfg=train_cfg, config_key=config_key, seed=42,
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
    for config_key in configs_to_run:
        cfg_path = OUTPUT_ROOT / f"loso_summary_config_{config_key}.json"
        cfg_summary = {"experiment": EXPERIMENT_NAME, "timestamp": TIMESTAMP,
                       "subjects": ALL_SUBJECTS, "exercises": EXERCISES,
                       "configs": {config_key: summary["configs"].get(config_key, {})}}
        with open(cfg_path, "w") as f:
            json.dump(make_json_serializable(cfg_summary), f, indent=4, ensure_ascii=False)
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
            mark_hypothesis_verified("H_PWVD_EMG", {
                "best_config": best_config,
                "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid])),
                "mean_f1_macro": float(best_f1),
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_PWVD_EMG", "All configs failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
