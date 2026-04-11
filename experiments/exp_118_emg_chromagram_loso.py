"""
Experiment 118: EMG Chromagram — Functional Band Projection for LOSO EMG

Hypothesis:
    Projecting the PSD onto 4 physiologically-motivated frequency bands
    (20-50, 50-100, 100-200, 200-500 Hz) — the EMG equivalent of a musical
    chromagram — provides a compact, interpretable representation that
    captures the H1 frequency-dependent variability structure.

    With only 4 bands (vs 13 MFCC, 25 MDCT, 26 Fbanks), the chromagram
    is extremely compact. The question: does this compactness help
    generalization (less overfitting to subject-specific spectral details)
    or hurt (too little information)?

What is tested:
      A) Chromagram+D+DD flat → SVM-RBF
      B) Chromagram+D+DD flat → RF
      C) Chromagram+D+DD spectrogram (12 x T_frames x C) → 2D-CNN
      D) Chromagram+D+DD flat + Handcrafted features → SVM-RBF

Run examples:
    python experiments/exp_118_emg_chromagram_loso.py --ci
    python experiments/exp_118_emg_chromagram_loso.py --full
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
from processing.emg_chromagram import EMGChromagramExtractor
from processing.features import HandcraftedFeatureExtractor
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from training.mfcc_trainer import MFCCTrainer
from training.trainer import WindowDataset, get_worker_init_fn

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_118_emg_chromagram"
EXERCISES       = ["E1"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

CONFIGS = {
    "A": {"mode": "ml",   "classifier": "svm_rbf", "use_deltas": True,
           "use_handcrafted": False, "label": "Chroma+D+DD -> SVM-RBF"},
    "B": {"mode": "ml",   "classifier": "rf",      "use_deltas": True,
           "use_handcrafted": False, "label": "Chroma+D+DD -> RF"},
    "C": {"mode": "deep", "classifier": None,       "use_deltas": True,
           "use_handcrafted": False, "label": "Chroma+D+DD -> 2D-CNN"},
    "D": {"mode": "ml",   "classifier": "svm_rbf", "use_deltas": True,
           "use_handcrafted": True,  "label": "Handcrafted+Chroma -> SVM-RBF"},
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


def _splits_to_arrays(splits, common_gestures):
    """Convert dict-of-arrays splits to flat (X, y) arrays."""
    result = {}
    for split_name in ("train", "val", "test"):
        X_list, y_list = [], []
        for i, gid in enumerate(sorted(common_gestures)):
            if gid in splits[split_name]:
                arr = splits[split_name][gid]
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                    X_list.append(arr)
                    y_list.append(np.full(len(arr), i, dtype=np.int64))
        if X_list:
            result[f"X_{split_name}"] = np.concatenate(X_list, axis=0)
            result[f"y_{split_name}"] = np.concatenate(y_list, axis=0)
        else:
            result[f"X_{split_name}"] = np.empty((0,))
            result[f"y_{split_name}"] = np.empty((0,), dtype=np.int64)
    return result


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
    """Execute one LOSO fold."""
    cfg = CONFIGS[config_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg, logger=logger,
        use_gpu=True, use_improved_processing=USE_IMPROVED,
    )
    base_viz = Visualizer(output_dir, logger)

    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data = multi_loader.load_multiple_subjects(
        base_dir=base_dir, subject_ids=all_subject_ids,
        exercises=exercises, include_rest=split_cfg.include_rest_in_splits,
    )
    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES,
    )
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    splits = _build_splits(
        subjects_data, train_subjects, test_subject,
        common_gestures, multi_loader,
        val_ratio=split_cfg.val_ratio, seed=train_cfg.seed,
    )

    for sname in ("train", "val", "test"):
        total = sum(len(a) for a in splits[sname].values()
                    if isinstance(a, np.ndarray) and a.ndim == 3)
        logger.info(f"  {sname.upper()}: {total} windows, {len(splits[sname])} gestures")

    # ── Deep mode: use MFCCTrainer infrastructure with chromagram ──
    if cfg["mode"] == "deep":
        # Chromagram spectrogram → 2D CNN (via MFCCTrainer with custom extractor)
        # We need a thin adapter: extract chromagram, then feed to CNN
        return _run_deep_fold(
            splits, common_gestures, cfg, config_key,
            proc_cfg, train_cfg, logger, output_dir, base_viz, test_subject,
        )

    # ── ML mode: flat features ──
    data = _splits_to_arrays(splits, common_gestures)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    if len(X_test) == 0:
        logger.error(f"No test data for {test_subject}")
        return {"test_subject": test_subject, "config": config_key,
                "label": cfg["label"], "test_accuracy": None, "test_f1_macro": None,
                "error": "No test data"}

    # Channel standardization
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean_c) / std_c
    X_test = (X_test - mean_c) / std_c

    # Feature extraction
    chroma_ext = EMGChromagramExtractor(
        sampling_rate=proc_cfg.sampling_rate,
        use_deltas=cfg["use_deltas"], logger=logger,
    )
    feat_parts_tr, feat_parts_te = [], []

    feat_parts_tr.append(chroma_ext.transform(X_train))
    feat_parts_te.append(chroma_ext.transform(X_test))

    if cfg["use_handcrafted"]:
        hc = HandcraftedFeatureExtractor(
            sampling_rate=proc_cfg.sampling_rate, feature_set="basic_v1",
        )
        feat_parts_tr.append(hc.transform(X_train))
        feat_parts_te.append(hc.transform(X_test))

    feat_train = np.nan_to_num(np.concatenate(feat_parts_tr, axis=1))
    feat_test = np.nan_to_num(np.concatenate(feat_parts_te, axis=1))

    logger.info(f"Feature dim: {feat_train.shape[1]}")

    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    if feat_train.shape[1] > 200:
        n_comp = min(200, feat_train.shape[0] - 1, feat_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=train_cfg.seed)
        feat_train = pca.fit_transform(feat_train)
        feat_test = pca.transform(feat_test)

    if cfg["classifier"] == "svm_rbf":
        model = svm.SVC(kernel="rbf", C=10.0, gamma="scale",
                        decision_function_shape="ovr", random_state=train_cfg.seed)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=train_cfg.seed, n_jobs=-1)

    logger.info(f"Training {cfg['classifier']} on {feat_train.shape}...")
    model.fit(feat_train, y_train)
    y_pred = model.predict(feat_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    num_classes = len(common_gestures)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"[LOSO] {cfg['label']} | test={test_subject} | Acc={acc:.4f}, F1={f1:.4f}")

    fold_summary = {"test_subject": test_subject, "config": config_key,
                    "label": cfg["label"], "test_accuracy": float(acc),
                    "test_f1_macro": float(f1), "report": report}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    del model, subjects_data; gc.collect()
    return {"test_subject": test_subject, "config": config_key,
            "label": cfg["label"], "test_accuracy": float(acc), "test_f1_macro": float(f1)}


def _run_deep_fold(splits, common_gestures, cfg, config_key,
                   proc_cfg, train_cfg, logger, output_dir, base_viz, test_subject):
    """Deep mode: chromagram spectrogram → 2D CNN."""
    from training.mfcc_trainer import _TensorDataset
    from models.mfcc_cnn_classifier import MFCCCNNClassifier
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim

    data = _splits_to_arrays(splits, common_gestures)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if len(X_test) == 0:
        return {"test_subject": test_subject, "config": config_key,
                "label": cfg["label"], "test_accuracy": None, "test_f1_macro": None,
                "error": "No test data"}

    mean_c = X_train.mean(axis=(0, 1), keepdims=True)
    std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean_c) / std_c
    X_val = (X_val - mean_c) / std_c if len(X_val) > 0 else X_val
    X_test = (X_test - mean_c) / std_c

    chroma_ext = EMGChromagramExtractor(
        sampling_rate=proc_cfg.sampling_rate,
        use_deltas=cfg["use_deltas"], logger=logger,
    )

    # (N, n_coeff, T_frames, C) → (N, C, n_coeff, T_frames)
    spec_train = chroma_ext.transform_spectrogram(X_train).transpose(0, 3, 1, 2)
    spec_val = chroma_ext.transform_spectrogram(X_val).transpose(0, 3, 1, 2) if len(X_val) > 0 else None
    spec_test = chroma_ext.transform_spectrogram(X_test).transpose(0, 3, 1, 2)

    in_ch, n_coeff, n_frames = spec_train.shape[1], spec_train.shape[2], spec_train.shape[3]
    num_classes = len(common_gestures)
    logger.info(f"Chromagram CNN input: ({in_ch}, {n_coeff}, {n_frames}), classes={num_classes}")

    model = MFCCCNNClassifier(
        in_channels=in_ch, n_coeff=n_coeff, n_frames=n_frames,
        num_classes=num_classes, cnn_channels=[32, 64, 128],
        dropout=train_cfg.dropout,
    ).to(train_cfg.device)

    ds_train = _TensorDataset(spec_train, y_train)
    ds_val = _TensorDataset(spec_val, y_val) if spec_val is not None else None

    g = torch.Generator().manual_seed(train_cfg.seed)
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

    # Evaluate
    model.eval()
    ds_test = _TensorDataset(spec_test, y_test)
    dl_test = DataLoader(ds_test, batch_size=train_cfg.batch_size, shuffle=False)
    all_preds, all_y = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.append(preds); all_y.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_y)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"[LOSO] {cfg['label']} | test={test_subject} | Acc={acc:.4f}, F1={f1:.4f}")

    fold_summary = {"test_subject": test_subject, "config": config_key,
                    "label": cfg["label"], "test_accuracy": float(acc),
                    "test_f1_macro": float(f1), "report": report}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return {"test_subject": test_subject, "config": config_key,
            "label": cfg["label"], "test_accuracy": float(acc), "test_f1_macro": float(f1)}


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

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : EMG chromagram (4 functional frequency bands)")
    print(f"Subjects   : {ALL_SUBJECTS}")
    print(f"Configs    : {configs_to_run}")
    print(f"Output     : {OUTPUT_ROOT}")
    print("=" * 80)

    all_results: Dict[str, List[Dict]] = {k: [] for k in configs_to_run}

    for config_key in configs_to_run:
        cfg = CONFIGS[config_key]
        print(f"\n{'~' * 60}\nConfig {config_key}: {cfg['label']}\n{'~' * 60}")

        if cfg["mode"] == "deep":
            train_cfg = TrainingConfig(
                model_type="chroma_cnn", pipeline_type="deep_chroma",
                use_handcrafted_features=False, batch_size=64, epochs=60,
                learning_rate=1e-3, weight_decay=1e-4, dropout=0.3,
                early_stopping_patience=12, seed=42, use_class_weights=True,
                num_workers=4, device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            train_cfg = TrainingConfig(
                model_type="chroma_ml", pipeline_type="feature_ml",
                use_handcrafted_features=True, batch_size=64, epochs=1,
                learning_rate=1e-3, weight_decay=0, dropout=0,
                early_stopping_patience=10, seed=42, use_class_weights=False,
                num_workers=0, device="cpu",
            )

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
            summary["configs"][config_key] = {"label": label, "error": "All folds failed",
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
            mark_hypothesis_verified("H_CHROMAGRAM_EMG", {
                "best_config": best_config,
                "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid])),
                "mean_f1_macro": float(best_f1),
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_CHROMAGRAM_EMG", "All configs failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
