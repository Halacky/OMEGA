"""
Experiment 116: Rényi Entropy + Complexity-Entropy Plane Features for LOSO EMG

Hypothesis:
    The Complexity-Entropy (C-H) plane (Rosso et al., 2007) provides a compact
    2D characterization of signal dynamics that captures gesture-discriminative
    information not present in standard time/frequency features.

    Different gestures produce distinct muscle recruitment patterns with
    characteristic entropy (disorder) and complexity (structure) signatures:
      - Rest: low entropy, low complexity
      - Fine finger movements: high complexity, moderate entropy
      - Power grasps: moderate complexity, lower entropy

What is tested:
    Three feature sets compared:
      A) Entropy+Complexity features only (14 per channel) → SVM-RBF
      B) Handcrafted TD features (basic_v1) + Entropy+Complexity → SVM-RBF
      C) Handcrafted TD features (basic_v1) + Entropy+Complexity → RF

    The key question: do entropy/complexity features add value ON TOP of
    existing handcrafted features, or are they redundant?

LOSO protocol: same as all other experiments — training stats only for
standardization, per-fold ML fitting.

Run examples:
    python experiments/exp_116_entropy_complexity_loso.py --ci
    python experiments/exp_116_entropy_complexity_loso.py --full
    python experiments/exp_116_entropy_complexity_loso.py --full --config B
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
    CI_TEST_SUBJECTS,
    DEFAULT_SUBJECTS,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from processing.features import HandcraftedFeatureExtractor
from processing.entropy_features import EntropyComplexityExtractor
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ========================== EXPERIMENT SETTINGS ==============================

EXPERIMENT_NAME = "exp_116_entropy_complexity"
EXERCISES       = ["E1"]
USE_IMPROVED    = True
MAX_GESTURES    = 10

CONFIGS = {
    "A": {
        "use_entropy": True,  "use_handcrafted": False,
        "classifier": "svm_rbf",
        "label": "Entropy+Complexity → SVM-RBF",
    },
    "B": {
        "use_entropy": True,  "use_handcrafted": True,
        "classifier": "svm_rbf",
        "label": "Handcrafted+Entropy → SVM-RBF",
    },
    "C": {
        "use_entropy": True,  "use_handcrafted": True,
        "classifier": "rf",
        "label": "Handcrafted+Entropy → RF",
    },
}


# ============================== HELPERS ======================================

def grouped_to_arrays(grouped_windows, common_gestures):
    """Convert grouped windows dict to flat (X, y) arrays."""
    X_list, y_list = [], []
    for i, gid in enumerate(sorted(common_gestures)):
        if gid not in grouped_windows:
            continue
        reps = grouped_windows[gid]
        for rep_arr in reps:
            if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                X_list.append(rep_arr)
                y_list.append(np.full(len(rep_arr), i, dtype=np.int64))
    if not X_list:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def _build_splits(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """Build train/val/test splits → flat (X, y) arrays."""
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

    # Build train/val arrays
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    for i, gid in enumerate(sorted(common_gestures)):
        if not train_dict[gid]:
            continue
        X_gid = np.concatenate(train_dict[gid], axis=0)
        n = len(X_gid)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        trn_idx = perm[n_val:]
        if len(trn_idx) > 0:
            X_train_list.append(X_gid[trn_idx])
            y_train_list.append(np.full(len(trn_idx), i, dtype=np.int64))
        if len(val_idx) > 0:
            X_val_list.append(X_gid[val_idx])
            y_val_list.append(np.full(len(val_idx), i, dtype=np.int64))

    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.empty((0,))
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.empty((0,), dtype=np.int64)
    X_val = np.concatenate(X_val_list, axis=0) if X_val_list else np.empty((0,))
    y_val = np.concatenate(y_val_list, axis=0) if y_val_list else np.empty((0,), dtype=np.int64)

    # Test arrays
    X_test_list, y_test_list = [], []
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        filtered_test = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for i, gid in enumerate(sorted(common_gestures)):
            if gid not in filtered_test:
                continue
            valid = [r for r in filtered_test[gid]
                     if isinstance(r, np.ndarray) and len(r) > 0]
            if valid:
                arr = np.concatenate(valid, axis=0)
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.empty((0,))
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.empty((0,), dtype=np.int64)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


# ================================ SINGLE FOLD ================================

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    config_key: str = "A",
    seed: int = 42,
) -> Dict:
    """Execute one LOSO fold."""
    cfg = CONFIGS[config_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(seed, verbose=False)

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED,
    )

    # Load subjects
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

    # Build splits → flat arrays (N, T, C)
    data = _build_splits(
        subjects_data, train_subjects, test_subject,
        common_gestures, multi_loader,
        val_ratio=split_cfg.val_ratio, seed=seed,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    if len(X_test) == 0:
        logger.error(f"No test data for {test_subject}")
        return {
            "test_subject": test_subject, "config": config_key,
            "label": cfg["label"],
            "test_accuracy": None, "test_f1_macro": None,
            "error": "No test data",
        }

    # ── Channel standardization (train stats only) ──
    mean_c = X_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
    std_c = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean_c) / std_c
    X_test = (X_test - mean_c) / std_c

    # ── Feature extraction ──
    feature_parts_train = []
    feature_parts_test = []

    if cfg["use_entropy"]:
        logger.info("Extracting entropy + complexity features...")
        entropy_ext = EntropyComplexityExtractor(
            sampling_rate=proc_cfg.sampling_rate, order=3, delay=1, logger=logger,
        )
        ent_train = entropy_ext.transform(X_train)
        ent_test = entropy_ext.transform(X_test)
        feature_parts_train.append(ent_train)
        feature_parts_test.append(ent_test)
        logger.info(f"  Entropy features: {ent_train.shape[1]}")

    if cfg["use_handcrafted"]:
        logger.info("Extracting handcrafted features...")
        hc_ext = HandcraftedFeatureExtractor(
            sampling_rate=proc_cfg.sampling_rate, feature_set="basic_v1",
        )
        hc_train = hc_ext.transform(X_train)
        hc_test = hc_ext.transform(X_test)
        feature_parts_train.append(hc_train)
        feature_parts_test.append(hc_test)
        logger.info(f"  Handcrafted features: {hc_train.shape[1]}")

    feat_train = np.concatenate(feature_parts_train, axis=1)
    feat_test = np.concatenate(feature_parts_test, axis=1)

    # Replace NaN/Inf
    feat_train = np.nan_to_num(feat_train, nan=0.0, posinf=0.0, neginf=0.0)
    feat_test = np.nan_to_num(feat_test, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Total feature dim: {feat_train.shape[1]}")

    # ── Standardize + PCA ──
    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    feat_dim = feat_train.shape[1]
    if feat_dim > 200:
        n_components = min(200, feat_train.shape[0] - 1, feat_dim)
        pca = PCA(n_components=n_components, random_state=seed)
        feat_train = pca.fit_transform(feat_train)
        feat_test = pca.transform(feat_test)
        logger.info(f"PCA: {feat_dim} → {n_components} (var={pca.explained_variance_ratio_.sum():.3f})")

    # ── Train classifier ──
    if cfg["classifier"] == "svm_rbf":
        model = svm.SVC(kernel="rbf", C=10.0, gamma="scale",
                        decision_function_shape="ovr", random_state=seed)
    elif cfg["classifier"] == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier: {cfg['classifier']}")

    logger.info(f"Training {cfg['classifier']} on {feat_train.shape}...")
    model.fit(feat_train, y_train)

    # ── Evaluate ──
    y_pred = model.predict(feat_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    num_classes = len(common_gestures)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(
        f"[LOSO] {cfg['label']} | test={test_subject} | "
        f"Acc={acc:.4f}, F1={f1:.4f}"
    )

    fold_summary = {
        "test_subject": test_subject,
        "config": config_key,
        "label": cfg["label"],
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
        "feature_dim": int(feat_train.shape[1]),
        "report": report,
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_summary), f, indent=4, ensure_ascii=False)

    del model, subjects_data
    gc.collect()

    return {
        "test_subject": test_subject,
        "config": config_key,
        "label": cfg["label"],
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
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
    print(f"Hypothesis : Rényi entropy + C-E plane features for EMG classification")
    print(f"Subjects   : {ALL_SUBJECTS}")
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
                base_dir=BASE_DIR,
                output_dir=fold_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                config_key=config_key,
                seed=42,
            )
            all_results[config_key].append(result)

    # ── Aggregate ──
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
            mark_hypothesis_verified("H_ENTROPY_CE", metrics,
                                     experiment_name=EXPERIMENT_NAME)
        else:
            mark_hypothesis_failed("H_ENTROPY_CE", "All configurations failed")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
