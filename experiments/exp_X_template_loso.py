import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch

# добавить корень репо в sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Default 20 subjects for full LOSO
DEFAULT_SUBJECTS = [
    "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]

# Subset for quick testing / CI runs
CI_TEST_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]


def parse_subjects_args(argv: Optional[List[str]] = None) -> List[str]:
    """Parse --subjects CLI argument. Returns subject list or CI_TEST_SUBJECTS."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs, e.g. DB2_s1,DB2_s12,DB2_s15",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Use CI test subset (5 subjects)",
    )
    args, _ = parser.parse_known_args(argv)
    if args.subjects:
        return [s.strip() for s in args.subjects.split(",")]
    if args.ci:
        return CI_TEST_SUBJECTS
    else:
        return DEFAULT_SUBJECTS
    return CI_TEST_SUBJECTS

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig

from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer, FeatureMLTrainer

from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer

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
        max_gestures=10,  # можно вынести в параметры, если нужно
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
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    # выбор тренера в зависимости от подхода
    if approach in ("deep_raw", "deep_emg_seq", "deep_powerful", "hybrid_powerful_deep"):
        trainer = WindowClassifierTrainer(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=base_viz,
        )
    elif approach == "ml_emg_td":
        trainer = FeatureMLTrainer(
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
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": approach,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


def main():
    # ====== НАСТРОЙКИ ЭКСПЕРИМЕНТА (заполняются в конкретных файлах) ======
    # IMPORTANT: Always use ROOT-relative path for data, NEVER hardcoded absolute paths:
    #   BASE_DIR = ROOT / "data"      <-- CORRECT (portable)
    #   BASE_DIR = Path("/home/...")   <-- WRONG (breaks on other machines)
    #
    # To run with a subset of subjects:
    #   python experiments/exp_N_*.py --subjects DB2_s1,DB2_s12,DB2_s15
    #   python experiments/exp_N_*.py --ci   # uses 5 CI test subjects
    #
    # In generated experiments, use:
    #   ALL_SUBJECTS = parse_subjects_args()
    raise NotImplementedError("Этот файл — шаблон. Используйте конкретные exp_*.py файлы.")


if __name__ == "__main__":
    main()