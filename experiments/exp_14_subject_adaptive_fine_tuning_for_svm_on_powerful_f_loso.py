# FILE: experiments/exp_14_subject_adaptive_fine_tuning_for_svm_on_powerful_f_loso.py
import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import FeatureMLTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit


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


def run_subject_adaptive_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    calibration_samples_per_class: int = 15,
) -> Dict:
    """
    LOSO fold with subject-adaptive fine-tuning:
    1. Pre-train SVM on train_subjects
    2. Extract calibration samples from test_subject
    3. Fine-tune SVM with calibration data
    4. Evaluate on remaining test data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = "ml_emg_td"
    train_cfg.model_type = "svm_linear"
    train_cfg.use_handcrafted_features = True
    train_cfg.handcrafted_feature_set = "powerful"

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

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

    base_viz = Visualizer(output_dir, logger)

    trainer = FeatureMLTrainer(
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
        print(f"Error in LOSO fold (test_subject={test_subject}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": "svm_linear",
            "approach": "subject_adaptive_fine_tuning",
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    subjects_data = results.get("subjects_data", {})
    common_gestures = results.get("config", {}).get("common_gestures", [])
    class_ids = trainer.class_ids or []

    # subjects_data values are tuples (emg, segments, grouped_windows) — NOT dicts
    test_windows = None
    test_labels = None
    test_features = None
    if test_subject in subjects_data:
        _, _, test_grouped_windows = subjects_data[test_subject]
        w_list, l_list = [], []
        for gesture_id in sorted(test_grouped_windows.keys()):
            if common_gestures and gesture_id not in common_gestures:
                continue
            if gesture_id not in class_ids:
                continue
            cls_idx = class_ids.index(gesture_id)
            for rep in test_grouped_windows[gesture_id]:
                if len(rep) > 0:
                    w_list.append(rep)
                    l_list.append(np.full(len(rep), cls_idx, dtype=np.int64))
        if w_list:
            test_windows = np.concatenate(w_list, axis=0)
            test_labels = np.concatenate(l_list, axis=0)
            if trainer.feature_extractor is not None and trainer.feature_mean is not None:
                try:
                    test_features = trainer.feature_extractor.transform(test_windows)
                    test_features = (test_features - trainer.feature_mean) / trainer.feature_std
                    if trainer.selected_feature_indices is not None:
                        test_features = test_features[:, trainer.selected_feature_indices]
                    if trainer.pca is not None:
                        test_features = trainer.pca.transform(test_features)
                except Exception as feat_e:
                    logger.warning(f"Failed to extract test features: {feat_e}")
                    test_features = None

    pretrain_metrics = results.get("cross_subject_test", {})
    pretrain_acc = float(pretrain_metrics.get("accuracy", 0.0))
    pretrain_f1 = float(pretrain_metrics.get("f1_macro", 0.0))

    fine_tuned_acc = pretrain_acc
    fine_tuned_f1 = pretrain_f1

    if test_features is not None and test_labels is not None:
        test_features = np.array(test_features)
        test_labels = np.array(test_labels)

        unique_classes = np.unique(test_labels)
        min_class_count = min(np.sum(test_labels == c) for c in unique_classes)
        actual_cal_samples = min(calibration_samples_per_class, min_class_count)

        if actual_cal_samples >= 5 and len(test_labels) >= actual_cal_samples * len(unique_classes) * 2:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=None,
                train_size=actual_cal_samples * len(unique_classes),
                random_state=train_cfg.seed
            )
            cal_idx, remaining_idx = next(sss.split(test_features, test_labels))

            cal_features = test_features[cal_idx]
            cal_labels = test_labels[cal_idx]
            remaining_features = test_features[remaining_idx]
            remaining_labels = test_labels[remaining_idx]

            # Extract train features from train subjects in subjects_data
            tr_w_list, tr_l_list = [], []
            for subj_id in train_subjects:
                if subj_id in subjects_data:
                    _, _, tr_gw = subjects_data[subj_id]
                    for gesture_id in sorted(tr_gw.keys()):
                        if common_gestures and gesture_id not in common_gestures:
                            continue
                        if gesture_id not in class_ids:
                            continue
                        cls_idx = class_ids.index(gesture_id)
                        for rep in tr_gw[gesture_id]:
                            if len(rep) > 0:
                                tr_w_list.append(rep)
                                tr_l_list.append(np.full(len(rep), cls_idx, dtype=np.int64))
            train_features = None
            train_labels = None
            if tr_w_list and trainer.feature_extractor is not None and trainer.feature_mean is not None:
                try:
                    all_tr_w = np.concatenate(tr_w_list, axis=0)
                    train_features = trainer.feature_extractor.transform(all_tr_w)
                    train_features = (train_features - trainer.feature_mean) / trainer.feature_std
                    if trainer.selected_feature_indices is not None:
                        train_features = train_features[:, trainer.selected_feature_indices]
                    if trainer.pca is not None:
                        train_features = trainer.pca.transform(train_features)
                    train_labels = np.concatenate(tr_l_list, axis=0)
                except Exception as feat_e:
                    logger.warning(f"Failed to extract train features: {feat_e}")
                    train_features = None
                    train_labels = None

            if train_features is not None and train_labels is not None:

                combined_features = np.vstack([train_features, cal_features])
                combined_labels = np.concatenate([train_labels, cal_labels])

                try:
                    from sklearn.svm import SVC

                    combined_train_data = {
                        "features": combined_features,
                        "labels": combined_labels,
                    }

                    ft_trainer = FeatureMLTrainer(
                        train_cfg=train_cfg,
                        logger=logger,
                        output_dir=output_dir / "fine_tuned",
                        visualizer=base_viz,
                    )
                    ft_trainer.train(combined_train_data, val_data=None)
                    ft_trainer.save_model(output_dir / "fine_tuned_model")

                    ft_predictions = ft_trainer.predict(remaining_features)
                    fine_tuned_acc = float(accuracy_score(remaining_labels, ft_predictions))
                    fine_tuned_f1 = float(f1_score(remaining_labels, ft_predictions, average="macro"))

                    print(
                        f"[Subject-Adaptive] {test_subject}: "
                        f"Pre-train Acc={pretrain_acc:.4f}, F1={pretrain_f1:.4f} | "
                        f"Fine-tuned Acc={fine_tuned_acc:.4f}, F1={fine_tuned_f1:.4f} | "
                        f"Calibration samples: {len(cal_labels)}"
                    )

                except Exception as ft_error:
                    logger.warning(f"Fine-tuning failed for {test_subject}: {ft_error}")
                    fine_tuned_acc = pretrain_acc
                    fine_tuned_f1 = pretrain_f1
        else:
            print(f"[Subject-Adaptive] {test_subject}: Insufficient test data for calibration, using pre-train results")
    else:
        print(f"[Subject-Adaptive] {test_subject}: No test features available, using pre-train results")

    final_acc = fine_tuned_acc
    final_f1 = fine_tuned_f1

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: svm_linear | Approach: subject_adaptive_fine_tuning | "
        f"Accuracy={final_acc:.4f}, F1-macro={final_f1:.4f}"
    )

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    results_to_save["subject_adaptive"] = {
        "pretrain_accuracy": pretrain_acc,
        "pretrain_f1_macro": pretrain_f1,
        "fine_tuned_accuracy": fine_tuned_acc,
        "fine_tuned_f1_macro": fine_tuned_f1,
        "calibration_samples_per_class": calibration_samples_per_class,
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": "svm_linear",
        "approach": "subject_adaptive_fine_tuning",
        "exercises": exercises,
        "config": {
            "processing": asdict(proc_cfg),
            "split": asdict(split_cfg),
            "training": asdict(train_cfg),
            "cross_subject": {
                "train_subjects": train_subjects,
                "test_subject": test_subject,
                "exercises": exercises,
            },
        },
        "metrics": {
            "pretrain_accuracy": pretrain_acc,
            "pretrain_f1_macro": pretrain_f1,
            "fine_tuned_accuracy": fine_tuned_acc,
            "fine_tuned_f1_macro": fine_tuned_f1,
            "test_accuracy": final_acc,
            "test_f1_macro": final_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": "svm_linear",
        "approach": "subject_adaptive_fine_tuning",
        "test_accuracy": final_acc,
        "test_f1_macro": final_f1,
        "pretrain_accuracy": pretrain_acc,
        "pretrain_f1_macro": pretrain_f1,
    }


def main():
    EXPERIMENT_NAME = "exp_14_subject_adaptive_fine_tuning_for_svm_on_powerful_f_loso"
    HYPOTHESIS_ID = "639423ce-b142-41c5-a9a0-8838cfc030e5"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")

    ALL_SUBJECTS = [
      "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]

    proc_cfg = ProcessingConfig(
        window_size=500,
        window_overlap=0,
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
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cpu",
        model_type="svm_linear",
        use_handcrafted_features=True,
        handcrafted_feature_set="powerful",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_linear",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
        aug_apply=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: svm_linear | Approach: subject_adaptive_fine_tuning")
    print(f"Features: powerful | LOSO n={len(ALL_SUBJECTS)}")
    print(f"Calibration samples per class: 15")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output_dir = OUTPUT_DIR / f"test_{test_subject}"

        try:
            fold_res = run_subject_adaptive_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                calibration_samples_per_class=15,
            )
            all_loso_results.append(fold_res)
            if fold_res.get("test_accuracy") is not None:
                print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, f1={fold_res['test_f1_macro']:.4f}")
            else:
                print(f"  ✗ {test_subject}: {fold_res.get('error', 'Unknown error')}")
        except Exception as e:
            global_logger.error(f"Failed {test_subject}: {e}")
            traceback.print_exc()
            all_loso_results.append({
                "test_subject": test_subject,
                "model_type": "svm_linear",
                "approach": "subject_adaptive_fine_tuning",
                "test_accuracy": None,
                "test_f1_macro": None,
                "error": str(e),
            })

    valid_results = [r for r in all_loso_results if r.get("test_accuracy") is not None]

    if valid_results:
        accs = [r["test_accuracy"] for r in valid_results]
        f1s = [r["test_f1_macro"] for r in valid_results]
        pretrain_accs = [r.get("pretrain_accuracy", r["test_accuracy"]) for r in valid_results]
        pretrain_f1s = [r.get("pretrain_f1_macro", r["test_f1_macro"]) for r in valid_results]

        aggregate = {
            "svm_linear_subject_adaptive": {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s)),
                "num_subjects": len(accs),
                "pretrain_mean_accuracy": float(np.mean(pretrain_accs)),
                "pretrain_std_accuracy": float(np.std(pretrain_accs)),
                "improvement_mean": float(np.mean(accs) - np.mean(pretrain_accs)),
            }
        }

        print(f"\n{'='*60}")
        print(f"Subject-Adaptive SVM-Linear Results (powerful features):")
        print(f"  Pre-train:   Acc = {aggregate['svm_linear_subject_adaptive']['pretrain_mean_accuracy']:.4f} ± {aggregate['svm_linear_subject_adaptive']['pretrain_std_accuracy']:.4f}")
        print(f"  Fine-tuned:  Acc = {aggregate['svm_linear_subject_adaptive']['mean_accuracy']:.4f} ± {aggregate['svm_linear_subject_adaptive']['std_accuracy']:.4f}")
        print(f"  Improvement: {aggregate['svm_linear_subject_adaptive']['improvement_mean']:+.4f}")
        print(f"{'='*60}")
    else:
        aggregate = {}

    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "powerful",
        "model": "svm_linear",
        "approach": "subject_adaptive_fine_tuning",
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "calibration_samples_per_class": 15,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_DIR.resolve()}")

    # === Update hypothesis status in Qdrant ===
    from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed

    if aggregate:
        best_model_name = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
        best_metrics = aggregate[best_model_name]
        best_metrics["best_model"] = best_model_name
        mark_hypothesis_verified(
            hypothesis_id=HYPOTHESIS_ID,
            metrics=best_metrics,
            experiment_name=EXPERIMENT_NAME,
        )
    else:
        mark_hypothesis_failed(
            hypothesis_id=HYPOTHESIS_ID,
            error_message="No successful LOSO folds completed",
        )


if __name__ == "__main__":
    main()