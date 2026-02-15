from pathlib import Path
import sys
import json
from dataclasses import asdict
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from utils.logging import setup_logging
from .exp_X_template_loso import run_single_loso_fold, make_json_serializable


def main():
    EXPERIMENT_NAME = "exp4_ml_powerful_loso"
    BASE_DIR = ROOT / "data"
    OUTPUT_DIR = Path(f"./experiments_output/{EXPERIMENT_NAME}")

    ALL_SUBJECTS = [
        "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
        "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
        "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
        "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
    ]

    EXERCISES = ["E1"]

    MODEL_TYPES = ["svm_rbf", "svm_linear", "rf"]

    APPROACH = "ml_emg_td"
    USE_IMPROVED_PROCESSING = True

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

    # ВНИМАНИЕ: для ML важно явно прописать use_handcrafted_features и выбор feature_set.
    # При use_improved_processing=True в вашем compare_models_improved
    # ml_emg_td переопределяет feature_set="powerful".
    train_cfg = TrainingConfig(
        batch_size=4096,  # для ML не критично, но нужно для evaluate_numpy
        epochs=1,         # не используется в MLTrainer, но пусть будет фиксировано
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.0,
        early_stopping_patience=1,
        use_class_weights=False,
        seed=42,
        num_workers=0,
        device="cpu",
        use_handcrafted_features=False,  # будет переопределено внутри ml_emg_td-ветки
        handcrafted_feature_set="emg_td",
        pipeline_type=APPROACH,
        ml_model_type="svm_rbf",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    all_loso_results = []

    for model_type in MODEL_TYPES:
        print(f"ML MODEL: {model_type} — starting LOSO (powerful features)")
        for test_subject in ALL_SUBJECTS:
            train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / model_type / f"test_{test_subject}"

            # важно: ml_model_type задаем через train_cfg
            train_cfg.ml_model_type = model_type

            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                model_type=model_type,
                approach=APPROACH,
                use_improved_processing=USE_IMPROVED_PROCESSING,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
            )
            all_loso_results.append(fold_res)

    aggregate_results = {}
    for model_type in MODEL_TYPES:
        model_results = [
            r for r in all_loso_results
            if r["model_type"] == model_type and r.get("test_accuracy") is not None
        ]
        if not model_results:
            continue
        accs = [r["test_accuracy"] for r in model_results]
        f1s = [r["test_f1_macro"] for r in model_results]
        aggregate_results[model_type] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "num_subjects": len(accs),
            "per_subject": model_results,
        }

    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "approach": APPROACH,
        "feature_set": "powerful (ml_emg_td + improved_processing=True)",
        "use_improved_processing": USE_IMPROVED_PROCESSING,
        "models": MODEL_TYPES,
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "training_config": asdict(train_cfg),
        "aggregate_results": aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)
    print(f"[DONE] {EXPERIMENT_NAME} -> {summary_path}")


if __name__ == "__main__":
    main()