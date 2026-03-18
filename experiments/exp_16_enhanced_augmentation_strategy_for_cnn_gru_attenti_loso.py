# FILE: experiments/exp_16_enhanced_augmentation_strategy_for_cnn_gru_attenti_loso.py
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

# добавить корень репо в sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig

from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer

from visualization.base import Visualizer

from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


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


def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    approach: str,
    use_improved_processing: bool,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
) -> Dict:
    """
    Один LOSO-фолд: обучаем модель `model_type` в рамках подхода `approach`,
    тестируем на `test_subject`. Возвращаем основные метрики.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    # фиксируем сид
    seed_everything(train_cfg.seed, verbose=False)

    # записываем выбранный подход и модель в конфиг обучения
    train_cfg.pipeline_type = approach
    train_cfg.model_type = model_type

    # сохраняем конфиги
    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # CrossSubjectConfig
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

    # лоадер
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=use_improved_processing,
    )

    base_viz = Visualizer(output_dir, logger)

    # выбор тренера для deep_raw подхода
    if approach in ("deep_raw", "deep_emg_seq", "deep_powerful", "hybrid_powerful_deep"):
        trainer = WindowClassifierTrainer(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=base_viz,
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")

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
        print(f"Error in LOSO fold (test_subject={test_subject}, model={model_type}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "approach": approach,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Model: {model_type} | Approach: {approach} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # сохраняем урезанные результаты
    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": model_type,
        "approach": approach,
        "exercises": exercises,
        "use_improved_processing": use_improved_processing,
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
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # очистка памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": approach,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    # ====== НАСТРОЙКИ ЭКСПЕРИМЕНТА ======
    EXPERIMENT_NAME = "exp_16_enhanced_augmentation_strategy_for_cnn_gru_attenti_loso"
    HYPOTHESIS_ID = "95984882-baad-4008-a419-050bd00b07a9"
    
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}_1_12_15_28_39")

    ALL_SUBJECTS = [
        "DB2_s1", "DB2_s12", "DB2_s15",  "DB2_s28", "DB2_s39"
    ]
    EXERCISES = ["E1"]
    
    # CNN-GRU-Attention model - best performing unaugmented deep architecture
    MODEL_TYPES = ["cnn_gru_attention"]

    # Processing config matching exp1_deep_raw_cnn_gru_attention_loso_isolated_v2
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
    
    # Enhanced augmentation strategy:
    # - noise augmentation (std=0.02) - from exp6 parameters
    # - time_warp augmentation (max=0.1) - from exp6 parameters
    # - Note: rotation augmentation requires additional framework support
    #   and is planned for future implementation
    train_cfg = TrainingConfig(
        batch_size=4096,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        model_type="cnn_gru_attention",
        # Augmentation settings - enhanced strategy
        aug_apply=True,
        aug_noise_std=0.02,
        aug_time_warp_max=0.1,
        aug_apply_noise=True,
        aug_apply_time_warp=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)
    
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: cnn_gru_attention | Pipeline: deep_raw | LOSO n={len(ALL_SUBJECTS)}")
    print(f"Augmentation: noise(std=0.02) + time_warp(max=0.1)")
    print(f"Hypothesis: Enhanced augmentation will improve accuracy from 0.3085 to ~0.34-0.36")

    all_loso_results = []
    
    for model_type in MODEL_TYPES:
        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"
            
            try:
                fold_res = run_single_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_output_dir,
                    train_subjects=train_subjects,
                    test_subject=test_subject,
                    exercises=EXERCISES,
                    model_type=model_type,
                    approach="deep_raw",
                    use_improved_processing=True,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                )
                all_loso_results.append(fold_res)
                if fold_res.get("test_accuracy") is not None:
                    print(f"  ✓ {test_subject}: acc={fold_res['test_accuracy']:.4f}, f1={fold_res['test_f1_macro']:.4f}")
                else:
                    print(f"  ✗ {test_subject}: {fold_res.get('error', 'Unknown error')}")
            except Exception as e:
                global_logger.error(f"Failed {test_subject} {model_type}: {e}")
                traceback.print_exc()
                all_loso_results.append({
                    "test_subject": test_subject,
                    "model_type": model_type,
                    "approach": "deep_raw",
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": str(e),
                })

    # Aggregate results
    aggregate = {}
    for model_type in MODEL_TYPES:
        model_results = [
            r for r in all_loso_results 
            if r["model_type"] == model_type and r.get("test_accuracy") is not None
        ]
        if not model_results:
            continue
        
        accs = [r["test_accuracy"] for r in model_results]
        f1s = [r["test_f1_macro"] for r in model_results]
        aggregate[model_type] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
        }
        print(
            f"\n{model_type}: Acc = {aggregate[model_type]['mean_accuracy']:.4f} ± {aggregate[model_type]['std_accuracy']:.4f}, "
            f"F1 = {aggregate[model_type]['mean_f1_macro']:.4f} ± {aggregate[model_type]['std_f1_macro']:.4f}"
        )
        print(
            f"  Range: [{aggregate[model_type]['min_accuracy']:.4f}, {aggregate[model_type]['max_accuracy']:.4f}]"
        )

    # Save LOSO summary
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "feature_set": "deep_raw",
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "augmentation": "noise(std=0.02) + time_warp(max=0.1)",
        "hypothesis": "Enhanced augmentation strategy for CNN-GRU-Attention on raw EMG",
        "baseline_comparison": {
            "baseline_experiment": "exp1_deep_raw_cnn_gru_attention_loso_isolated_v2",
            "baseline_accuracy": 0.3085,
            "target_accuracy": "0.34-0.36",
        },
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

    # Find the best model metrics for Qdrant
    if aggregate:
        best_model_name = max(aggregate, key=lambda m: aggregate[m]["mean_accuracy"])
        best_metrics = aggregate[best_model_name].copy()
        best_metrics["best_model"] = best_model_name
        best_metrics["augmentation_applied"] = "noise+time_warp"
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