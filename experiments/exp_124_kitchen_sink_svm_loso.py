"""
Experiment 124: Kitchen Sink Features → SVM for LOSO EMG Classification

Hypothesis:
    Each feature type captures different aspects of the EMG signal.
    Combining ALL available feature extractors may push beyond the current
    best of 38.54% (exp_123 B: HC + Entropy + MFCC fullband).

    Available feature types:
      - HC:    96 — handcrafted TD (MAV, RMS, WL, ZCR, skew, kurt, spectral)
      - ENT:  112 — Rényi entropy + C-E plane (time + freq domains)
      - MFCC: 1872 — MFCC flat (13 coeff * 3 deltas * 8 ch * 6 stats), fmax=1000
      - CHROMA: ~608 — chromagram flat (4 bands * 3 deltas * 8 ch * 6 stats + ratios)
      - MDCT: ~1200 — MDCT flat (25 bins * 8 ch * 6 stats), no deltas
      - ECS:  104 — energy cosine spectrum (13 coeff * 8 ch)

What is tested:
      A) ALL features → SVM-RBF  (kitchen sink)
      B) HC + ENT + MFCC(1000) → SVM-RBF  (baseline = exp_123 B, 38.54%)
      C) HC + ENT + MFCC(1000) + CHROMA → SVM-RBF  (+ chromagram)
      D) HC + ENT + MFCC(1000) + MDCT + ECS → SVM-RBF  (+ MDCT + ECS, no chroma)

Run:
    python experiments/exp_124_kitchen_sink_svm_loso.py --ci
    python experiments/exp_124_kitchen_sink_svm_loso.py --full
"""

import gc
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

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
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_124_kitchen_sink_svm"
EXERCISES       = ["E1"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

CONFIGS = {
    "A": {
        "features": ["hc", "entropy", "mfcc_1000", "chroma", "mdct", "ecs"],
        "label": "ALL (HC+ENT+MFCC+CHROMA+MDCT+ECS) → SVM-RBF",
    },
    "B": {
        "features": ["hc", "entropy", "mfcc_1000"],
        "label": "HC+ENT+MFCC(1000) → SVM-RBF (baseline)",
    },
    "C": {
        "features": ["hc", "entropy", "mfcc_1000", "chroma"],
        "label": "HC+ENT+MFCC(1000)+CHROMA → SVM-RBF",
    },
    "D": {
        "features": ["hc", "entropy", "mfcc_1000", "mdct", "ecs"],
        "label": "HC+ENT+MFCC(1000)+MDCT+ECS → SVM-RBF",
    },
}


# ============================== FEATURE EXTRACTORS ===========================

def build_extractors(feature_list, sampling_rate, logger):
    """Build dict of feature extractors based on requested features."""
    extractors = {}

    if "hc" in feature_list:
        extractors["hc"] = HandcraftedFeatureExtractor(
            sampling_rate=sampling_rate, feature_set="basic_v1",
        )

    if "entropy" in feature_list:
        extractors["entropy"] = EntropyComplexityExtractor(
            sampling_rate=sampling_rate, order=3, delay=1,
        )

    if "mfcc_1000" in feature_list:
        extractors["mfcc_1000"] = EMGMFCCExtractor(
            sampling_rate=sampling_rate, n_mfcc=13, n_mels=26,
            fmin=20.0, fmax=1000.0, use_deltas=True,
        )

    if "chroma" in feature_list:
        extractors["chroma"] = EMGChromagramExtractor(
            sampling_rate=sampling_rate, use_deltas=True,
        )

    if "mdct" in feature_list:
        extractors["mdct"] = EMGMFCCExtractor(
            sampling_rate=sampling_rate, n_mfcc=13, n_mels=26,
            fmin=20.0, fmax=1000.0, use_deltas=False,
        )

    if "ecs" in feature_list:
        extractors["ecs"] = EnergyCosineSpectrumExtractor(
            sampling_rate=sampling_rate, n_ecs=13, use_deltas=False,
        )

    return extractors


def extract_all(extractors, X, logger):
    """Extract features from all extractors and concatenate."""
    parts = []
    for name, ext in extractors.items():
        if name == "mdct":
            feat = ext.transform_mdct(X)
        elif name == "mfcc_1000":
            feat = ext.transform(X)
        elif name in ("hc", "entropy", "chroma", "ecs"):
            feat = ext.transform(X)
        else:
            raise ValueError(f"Unknown extractor: {name}")
        parts.append(feat)
        logger.info(f"  {name}: {feat.shape[1]} features")
    return np.concatenate(parts, axis=1)


# ============================== SPLITS =======================================

def _build_splits_flat(
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

    X_tr, y_tr, X_te, y_te = [], [], [], []
    for i, gid in enumerate(sorted(common_gestures)):
        if train_dict[gid]:
            X_gid = np.concatenate(train_dict[gid], axis=0)
            n = len(X_gid)
            perm = rng.permutation(n)
            nv = max(1, int(n * val_ratio))
            X_tr.append(X_gid[perm[nv:]])
            y_tr.append(np.full(len(X_gid) - nv, i, dtype=np.int64))

        if test_subject in subjects_data:
            _, _, tgw = subjects_data[test_subject]
            ft = multi_loader.filter_by_gestures(tgw, common_gestures)
            if gid in ft:
                valid = [r for r in ft[gid] if isinstance(r, np.ndarray) and len(r) > 0]
                if valid:
                    arr = np.concatenate(valid, axis=0)
                    X_te.append(arr)
                    y_te.append(np.full(len(arr), i, dtype=np.int64))

    X_train = np.concatenate(X_tr) if X_tr else np.empty((0,))
    y_train = np.concatenate(y_tr) if y_tr else np.empty((0,), dtype=np.int64)
    X_test = np.concatenate(X_te) if X_te else np.empty((0,))
    y_test = np.concatenate(y_te) if y_te else np.empty((0,), dtype=np.int64)
    return X_train, y_train, X_test, y_test


# ================================ SINGLE FOLD ================================

def run_single_loso_fold(
    base_dir, output_dir, train_subjects, test_subject,
    exercises, proc_cfg, split_cfg, config_key="A", seed=42,
):
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

    X_train, y_train, X_test, y_test = _build_splits_flat(
        subjects_data, train_subjects, test_subject,
        common_gestures, multi_loader,
        val_ratio=split_cfg.val_ratio, seed=seed,
    )

    if len(X_test) == 0:
        logger.error(f"No test data for {test_subject}")
        return {"test_subject": test_subject, "config": config_key,
                "label": cfg["label"], "test_accuracy": None,
                "test_f1_macro": None, "error": "No test data"}

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Channel standardization
    mc = X_train.mean(axis=(0, 1), keepdims=True)
    sc = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_std = (X_train - mc) / sc
    X_test_std = (X_test - mc) / sc

    # Feature extraction
    extractors = build_extractors(cfg["features"], proc_cfg.sampling_rate, logger)
    logger.info(f"Extracting features: {list(extractors.keys())}")

    feat_train = np.nan_to_num(extract_all(extractors, X_train_std, logger))
    feat_test = np.nan_to_num(extract_all(extractors, X_test_std, logger))
    logger.info(f"Total feature dim: {feat_train.shape[1]}")

    # Standardize + PCA
    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    if feat_train.shape[1] > 200:
        n_comp = min(200, feat_train.shape[0] - 1, feat_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        feat_train = pca.fit_transform(feat_train)
        feat_test = pca.transform(feat_test)
        logger.info(f"PCA → {n_comp} components (var={pca.explained_variance_ratio_.sum():.3f})")

    # SVM-RBF
    model = svm.SVC(kernel="rbf", C=10.0, gamma="scale",
                    decision_function_shape="ovr", random_state=seed)
    logger.info(f"Training SVM-RBF on {feat_train.shape}...")
    model.fit(feat_train, y_train)

    y_pred = model.predict(feat_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"[LOSO] {cfg['label'][:50]} | test={test_subject} | Acc={acc:.4f}, F1={f1:.4f}")

    fold_summary = {
        "test_subject": test_subject, "config": config_key,
        "label": cfg["label"], "test_accuracy": float(acc),
        "test_f1_macro": float(f1), "feature_dim": feat_train.shape[1],
        "report": report,
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    del model, subjects_data; gc.collect()
    return {"test_subject": test_subject, "config": config_key,
            "label": cfg["label"],
            "test_accuracy": float(acc), "test_f1_macro": float(f1)}


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
    print(f"Hypothesis : Kitchen sink features → SVM-RBF")
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
                config_key=config_key, seed=42,
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
            print(f"    vs exp_123 B (38.54%): {np.mean(accs)*100 - 38.54:+.2f}pp")
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
    # Save per-config summaries (safe for parallel runs)
    for config_key in configs_to_run:
        cfg_summary = {
            "experiment": EXPERIMENT_NAME, "timestamp": TIMESTAMP,
            "subjects": ALL_SUBJECTS, "exercises": EXERCISES,
            "configs": {config_key: summary["configs"].get(config_key, {})},
        }
        cfg_path = OUTPUT_ROOT / f"loso_summary_config_{config_key}.json"
        with open(cfg_path, "w") as f:
            json.dump(make_json_serializable(cfg_summary), f, indent=4, ensure_ascii=False)
        print(f"Config {config_key} summary: {cfg_path}")
    # Also save combined summary
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Combined summary: {OUTPUT_ROOT / 'loso_summary.json'}")

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
            mark_hypothesis_verified("H_KITCHEN_SINK", {
                "best_config": best_config,
                "mean_accuracy": float(np.mean([r["test_accuracy"] for r in valid])),
                "mean_f1_macro": float(best_f1),
            }, experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_KITCHEN_SINK", "All configs failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
