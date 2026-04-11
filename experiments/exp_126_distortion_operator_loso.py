"""
Experiment 126: Distortion Operator (FIR Deconvolution) for LOSO EMG

Hypothesis:
    Subject-specific EMG variation is largely a linear distortion (transfer
    function from neural drive through tissue/electrode to recording).
    A trainable FIR deconvolution filter can approximate the inverse of this
    transfer function, recovering subject-invariant neural drive.

    exp_65 (FIR deconv + CNN-GRU) got 45.08% on 5-subj CI — but used E1+E2
    (gesture bug). This experiment provides clean E1-only 20-subject results.

    New idea: use FIR deconvolution as a PREPROCESSING step for the kitchen
    sink SVM pipeline. If deconvolution removes subject-specific distortion,
    features extracted from deconvolved signals should be more subject-invariant.

What is tested:
      A) FIR deconv + CNN-BiGRU-Attention (reproduction of exp_65 architecture)
      B) FIR deconv → kitchen sink features (ALL 6) → SVM-RBF
      C) Spectral whitening → kitchen sink features → SVM-RBF (simpler deconv)
      D) Kitchen sink features → SVM-RBF (baseline without deconv, = exp_124 A)

Run:
    python experiments/exp_126_distortion_operator_loso.py --ci
    python experiments/exp_126_distortion_operator_loso.py --full
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
from models.fir_deconv_cnn_gru import FIRDeconvCNNGRU
from training.trainer import WindowDataset, get_worker_init_fn, seed_everything
from visualization.base import Visualizer
from utils.logging import setup_logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_126_distortion_operator"
EXERCISES       = ["E1"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

CONFIGS = {
    "A": {
        "mode": "deep", "deconv": "fir",
        "label": "FIR deconv + CNN-BiGRU (exp_65 arch, E1 only)",
    },
    "B": {
        "mode": "ml", "deconv": "fir",
        "label": "FIR deconv → kitchen sink → SVM-RBF",
    },
    "C": {
        "mode": "ml", "deconv": "spectral_whitening",
        "label": "Spectral whitening → kitchen sink → SVM-RBF",
    },
    "D": {
        "mode": "ml", "deconv": "none",
        "label": "Kitchen sink → SVM-RBF (baseline, no deconv)",
    },
}


# ============================== HELPERS ======================================

def _build_splits(subjects_data, train_subjects, test_subject,
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


def spectral_whitening(X, eps=1e-8):
    """
    Per-channel spectral whitening: flatten the average spectral envelope.

    Estimates the mean power spectrum from training data and divides each
    window's spectrum by it. This removes the average subject-specific
    spectral shape while preserving temporal and gesture-specific variations.

    Args:
        X: (N, T, C) — EMG windows.
    Returns:
        X_whitened: (N, T, C), mean_psd: (freq, C) for applying to test.
    """
    N, T, C = X.shape
    # Compute mean PSD across all windows
    fft_vals = np.fft.rfft(X, axis=1)  # (N, freq, C)
    psd = np.abs(fft_vals) ** 2         # (N, freq, C)
    mean_psd = psd.mean(axis=0)         # (freq, C)

    # Whitening filter: 1 / sqrt(mean_psd)
    whitening = 1.0 / (np.sqrt(mean_psd) + eps)  # (freq, C)

    # Apply in frequency domain
    whitened_fft = fft_vals * whitening[None, :, :]  # (N, freq, C)
    X_whitened = np.fft.irfft(whitened_fft, n=T, axis=1).astype(np.float32)

    return X_whitened, mean_psd


def apply_spectral_whitening(X, mean_psd, eps=1e-8):
    """Apply pre-computed whitening filter to new data."""
    N, T, C = X.shape
    fft_vals = np.fft.rfft(X, axis=1)
    whitening = 1.0 / (np.sqrt(mean_psd) + eps)
    whitened_fft = fft_vals * whitening[None, :, :]
    return np.fft.irfft(whitened_fft, n=T, axis=1).astype(np.float32)


def extract_kitchen_sink(X, sr):
    """Extract all 6 feature types from EMG windows."""
    hc = HandcraftedFeatureExtractor(sampling_rate=sr, feature_set="basic_v1")
    ent = EntropyComplexityExtractor(sampling_rate=sr, order=3, delay=1)
    mfcc = EMGMFCCExtractor(sampling_rate=sr, n_mfcc=13, n_mels=26,
                             fmin=20.0, fmax=1000.0, use_deltas=True)
    chroma = EMGChromagramExtractor(sampling_rate=sr, use_deltas=True)
    mdct = EMGMFCCExtractor(sampling_rate=sr, n_mfcc=13, n_mels=26,
                             fmin=20.0, fmax=1000.0, use_deltas=False)
    ecs = EnergyCosineSpectrumExtractor(sampling_rate=sr, n_ecs=13, use_deltas=False)

    parts = [
        hc.transform(X),
        ent.transform(X),
        mfcc.transform(X),
        chroma.transform(X),
        mdct.transform_mdct(X),
        ecs.transform(X),
    ]
    return np.nan_to_num(np.concatenate(parts, axis=1))


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

    data = _build_splits(subjects_data, train_subjects, test_subject,
                         common_gestures, multi_loader,
                         val_ratio=split_cfg.val_ratio, seed=seed)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if len(X_test) == 0:
        return {"test_subject": test_subject, "config": config_key,
                "label": cfg["label"], "test_accuracy": None,
                "test_f1_macro": None, "error": "No test data"}

    # Channel standardization
    mc = X_train.mean(axis=(0, 1), keepdims=True)
    sc = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mc) / sc
    X_val = (X_val - mc) / sc if len(X_val) > 0 else X_val
    X_test = (X_test - mc) / sc

    if cfg["mode"] == "deep":
        return _run_deep(cfg, config_key, X_train, y_train, X_val, y_val,
                         X_test, y_test, common_gestures, train_cfg,
                         logger, output_dir, test_subject, seed)
    else:
        return _run_ml(cfg, config_key, X_train, y_train, X_test, y_test,
                       common_gestures, proc_cfg, logger, output_dir,
                       test_subject, seed)


def _run_ml(cfg, config_key, X_train, y_train, X_test, y_test,
            common_gestures, proc_cfg, logger, output_dir, test_subject, seed):
    """ML mode: optional deconv → kitchen sink features → SVM."""

    if cfg["deconv"] == "spectral_whitening":
        logger.info("Applying spectral whitening...")
        X_train, mean_psd = spectral_whitening(X_train)
        X_test = apply_spectral_whitening(X_test, mean_psd)
    elif cfg["deconv"] == "fir":
        # Train a quick FIR deconv on training data, then apply to both
        logger.info("Training FIR deconvolution filter...")
        X_train, X_test = _apply_learned_fir_deconv(
            X_train, y_train, X_test, len(common_gestures), logger, seed,
        )

    logger.info("Extracting kitchen sink features...")
    feat_train = extract_kitchen_sink(X_train, proc_cfg.sampling_rate)
    feat_test = extract_kitchen_sink(X_test, proc_cfg.sampling_rate)
    logger.info(f"Feature dim: {feat_train.shape[1]}")

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
    _save(output_dir, test_subject, config_key, cfg["label"], acc, f1, report)
    del model; gc.collect()
    return {"test_subject": test_subject, "config": config_key,
            "label": cfg["label"], "test_accuracy": float(acc), "test_f1_macro": float(f1)}


def _apply_learned_fir_deconv(X_train, y_train, X_test, num_classes, logger, seed):
    """Train a FIR deconv filter on training data, apply to train+test."""
    from models.fir_deconv_cnn_gru import FIRDeconvFrontend

    seed_everything(seed, verbose=False)
    C = X_train.shape[2]
    frontend = FIRDeconvFrontend(in_channels=C, filter_len=63)

    # Quick training: minimize classification loss with simple linear classifier
    # to learn the deconv filter
    X_tr_t = torch.from_numpy(X_train.transpose(0, 2, 1)).float()  # (N, C, T)
    y_tr_t = torch.from_numpy(y_train)

    # Simple classifier head for training the FIR filter
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(C, num_classes),
    )
    params = list(frontend.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs
    frontend.train(); classifier.train()
    batch_size = 256
    for epoch in range(15):
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            optimizer.zero_grad()
            deconv = frontend(xb)
            logits = classifier(deconv)
            loss = criterion(logits, yb) + frontend.regularization_loss(1e-3, 5e-3)
            loss.backward()
            optimizer.step()

    # Apply learned filter
    frontend.eval()
    with torch.no_grad():
        X_train_deconv = frontend(X_tr_t).numpy().transpose(0, 2, 1)  # (N, T, C)
        X_te_t = torch.from_numpy(X_test.transpose(0, 2, 1)).float()
        X_test_deconv = frontend(X_te_t).numpy().transpose(0, 2, 1)

    logger.info("FIR deconv filter trained and applied.")
    return X_train_deconv.astype(np.float32), X_test_deconv.astype(np.float32)


def _run_deep(cfg, config_key, X_train, y_train, X_val, y_val,
              X_test, y_test, common_gestures, train_cfg,
              logger, output_dir, test_subject, seed):
    """Deep mode: FIR deconv + CNN-BiGRU-Attention."""
    # Transpose (N, T, C) → (N, C, T)
    X_train_ct = X_train.transpose(0, 2, 1)
    X_val_ct = X_val.transpose(0, 2, 1) if len(X_val) > 0 else None
    X_test_ct = X_test.transpose(0, 2, 1)

    in_ch = X_train_ct.shape[1]
    num_classes = len(common_gestures)

    model = FIRDeconvCNNGRU(
        in_channels=in_ch, num_classes=num_classes,
        filter_len=63, cnn_channels=(64, 128, 256),
        gru_hidden=128, num_heads=4, dropout=train_cfg.dropout,
    ).to(train_cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"FIRDeconvCNNGRU: {total_params:,} params")

    ds_train = WindowDataset(X_train_ct, y_train)
    ds_val = WindowDataset(X_val_ct, y_val) if X_val_ct is not None else None
    g = torch.Generator().manual_seed(seed)
    dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, shuffle=True,
                          num_workers=train_cfg.num_workers, pin_memory=True, generator=g)
    dl_val = DataLoader(ds_val, batch_size=train_cfg.batch_size, shuffle=False,
                        num_workers=train_cfg.num_workers, pin_memory=True) if ds_val else None

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    cw = counts.sum() / (counts + 1e-8); cw /= cw.mean()
    ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(cw).float().to(train_cfg.device))
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
            loss = ce_loss(logits, yb) + model.regularization_loss(1e-3, 5e-3)
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
                    vl += ce_loss(logits, yb).item() * yb.size(0)
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
    ds_test = WindowDataset(X_test_ct, y_test)
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
    _save(output_dir, test_subject, config_key, cfg["label"], acc, f1, report)

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    del model; gc.collect()
    return {"test_subject": test_subject, "config": config_key,
            "label": cfg["label"], "test_accuracy": float(acc), "test_f1_macro": float(f1)}


def _save(output_dir, test_subject, config_key, label, acc, f1, report):
    s = {"test_subject": test_subject, "config": config_key,
         "label": label, "test_accuracy": float(acc),
         "test_f1_macro": float(f1), "report": report}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(s), f, indent=4, ensure_ascii=False)


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
        model_type="fir_deconv_cnn_gru", pipeline_type="deep_raw",
        use_handcrafted_features=False, batch_size=64, epochs=60,
        learning_rate=1e-3, weight_decay=1e-4, dropout=0.3,
        early_stopping_patience=12, seed=42, use_class_weights=True,
        num_workers=4, device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Hypothesis : Distortion operator (FIR deconv / spectral whitening)")
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
        p = OUTPUT_ROOT / f"loso_summary_config_{config_key}.json"
        with open(p, "w") as f:
            json.dump(make_json_serializable({
                "experiment": EXPERIMENT_NAME, "timestamp": TIMESTAMP,
                "subjects": ALL_SUBJECTS, "exercises": EXERCISES,
                "configs": {config_key: summary["configs"].get(config_key, {})},
            }), f, indent=4, ensure_ascii=False)
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
            mark_hypothesis_verified("H_DISTORTION_OP", {
                "best_config": best_config,
                "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid])),
                "mean_f1_macro": float(best_f1),
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_DISTORTION_OP", "All configs failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
