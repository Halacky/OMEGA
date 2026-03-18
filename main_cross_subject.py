import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime 
from dataclasses import asdict

from utils.logging import setup_logging, seed_everything
from config.base import ProcessingConfig, TrainingConfig, SplitConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.trainer import WindowClassifierTrainer
from visualization.base import Visualizer
from visualization.cross_subject import CrossSubjectVisualizer
from evaluation.cross_subject import CrossSubjectExperiment
from utils.artifacts import ArtifactSaver
from config.base import ProcessingConfig, TrainingConfig, SplitConfig, RotationConfig
from visualization.rotation import RotationVisualizer
from evaluation.cross_subject_rotation import CrossSubjectRotationExperiment

def make_json_serializable(obj):
    """Convert object to JSON-serializable format"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    BASE_DIR = Path(__file__).resolve().parent / "data"
    OUTPUT_DIR = Path("./output_cross_subject")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Processing configuration (same as before)
    proc_cfg = ProcessingConfig(
        window_size=500,
        window_overlap=0,
        num_channels=8,  # Use 8 bracelet sensors
        sampling_rate=2000
    )
    
    # Split configuration
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",          # Split by segments for better generalization
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False  # Exclude REST from splits
    )
    # Training configuration
    train_cfg = TrainingConfig(
        batch_size=256,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Cross-subject configuration
    cross_subject_cfg = CrossSubjectConfig(
        train_subjects=["DB2_s1", ],
        test_subject="DB2_s10",
        exercise="E3",
        base_dir=BASE_DIR,
        pool_train_subjects=True,
        use_separate_val_subject=False,
        val_subject=None,
        val_ratio=0.15,
        seed=42
    )
    
    # ========================================================================
    # Setup
    # ========================================================================
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 80)
    logger.info("Cross-Subject Gesture Recognition Experiment")
    logger.info("=" * 80)
    
    seed_everything(train_cfg.seed)
    
    # Save configurations
    cross_subject_cfg.save(OUTPUT_DIR / "cross_subject_config.json")
    proc_cfg.save(OUTPUT_DIR / "processing_config.json")
    train_cfg.save(OUTPUT_DIR / "training_config.json")
    
    # ========================================================================
    # Initialize components
    # ========================================================================
    
    # Multi-subject loader
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True
    )
    
    # Visualizers
    base_viz = Visualizer(OUTPUT_DIR, logger)
    cross_viz = CrossSubjectVisualizer(OUTPUT_DIR, logger)
    
    # Trainer
    trainer = WindowClassifierTrainer(
        train_cfg=train_cfg,
        logger=logger,
        output_dir=OUTPUT_DIR,
        visualizer=base_viz
    )
    
    # ========================================================================
    # Run Cross-Subject Experiment
    # ========================================================================
    
    experiment = CrossSubjectExperiment(
        cross_subject_config=cross_subject_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=base_viz,
        logger=logger
    )

    results = experiment.run()

    # ========================================================================
    # NEW: Cross-subject rotation experiment on test subject
    # ========================================================================
    try:
        logger.info("Starting cross-subject rotation experiment on test subject ...")

        # Достаём данные тестового субъекта и common_gestures
        test_subject_id = cross_subject_cfg.test_subject
        subjects_data = results["subjects_data"]
        common_gestures = results["config"]["common_gestures"]

        _, _, grouped_windows_test = subjects_data[test_subject_id]

        # Rotation config (можно вынести в конфиг/CLI)
        C = trainer.in_channels if trainer.in_channels is not None else grouped_windows_test[next(iter(grouped_windows_test))][0].shape[2]
        rot_cfg = RotationConfig(
            rotations=[-3, -2, -1, 0, 1, 2, 3],
            bracelet_size=C,
            channel_order=list(range(C)),
            single_segment=None
        )

        rot_viz = RotationVisualizer(OUTPUT_DIR, logger)
        cs_rot_experiment = CrossSubjectRotationExperiment(
            trainer=trainer,
            logger=logger,
            rot_cfg=rot_cfg,
            rot_visualizer=rot_viz,
        )

        rotation_to_metrics = cs_rot_experiment.run_full_rotation_on_test_subject(
            grouped_windows_test=grouped_windows_test,
            common_gestures=common_gestures,
            experiment_name=f"cross_subject_{test_subject_id}",
            visualize_per_rotation=False,  # True, если хочешь отдельные CM/ROC на каждую ротацию
        )

        # Сохраняем метрики ротации в общий results
        results["rotation_experiment"] = {
            "config": {
                "rotations": rot_cfg.rotations,
                "bracelet_size": rot_cfg.bracelet_size,
                "channel_order": rot_cfg.channel_order,
            },
            "test_metrics_by_rotation": rotation_to_metrics,
        }

        # Пересохраняем результаты (без subjects_data)
        saver = ArtifactSaver(OUTPUT_DIR, logger)
        results_to_save = {k: v for k, v in results.items() if k != 'subjects_data'}

        results_path = OUTPUT_DIR / "cross_subject_results_with_rotation.json"
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Cross-subject + rotation results saved: {results_path}")

    except Exception as e:
        logger.error(f"Cross-subject rotation experiment failed: {e}")
    
    # ========================================================================
    # Visualizations
    # ========================================================================
    
    logger.info("Creating cross-subject visualizations...")
    
    try:
        # Per-subject comparison
        if results.get('per_subject_analysis'):
            cross_viz.plot_per_subject_comparison(
                per_subject_results=results['per_subject_analysis'],
                train_subjects=cross_subject_cfg.train_subjects,
                test_subject=cross_subject_cfg.test_subject,
                filename="per_subject_comparison.png"
            )
    except Exception as e:
        logger.error(f"Failed to create per-subject comparison plot: {e}")
    
    try:
        # Train vs test comparison
        if results.get('cross_subject_test'):
            cross_viz.plot_train_vs_test_comparison(
                training_results=results.get('training'),
                test_results=results['cross_subject_test'],
                filename="train_vs_test_comparison.png"
            )
    except Exception as e:
        logger.error(f"Failed to create train vs test comparison plot: {e}")
    
    try:
        # Comprehensive summary
        cross_viz.plot_cross_subject_summary(
            results=results,
            filename="cross_subject_summary.png"
        )
    except Exception as e:
        logger.error(f"Failed to create cross-subject summary plot: {e}")
    
    try:
        # Data split schema
        cross_viz.plot_data_split_schema(
            cross_subject_config=cross_subject_cfg,
            split_info=results.get('split_info', {}),
            filename="data_split_schema.png"
        )
    except Exception as e:
        logger.error(f"Failed to create data split schema: {e}")
    
    try:
        # Gesture comparison across subjects
        if 'subjects_data' in results:
            common_gestures = results.get('config', {}).get('common_gestures', [])
            # Select 3-4 representative gestures for comparison
            gestures_to_compare = common_gestures[:min(4, len(common_gestures))]
            
            cross_viz.plot_gesture_comparison_across_subjects(
                subjects_data=results['subjects_data'],
                gesture_ids=gestures_to_compare,
                num_windows_per_gesture=1,
                channel_idx=3,  # First channel
                filename="gesture_comparison_subjects.png"
            )
    except Exception as e:
        logger.error(f"Failed to create gesture comparison plot: {e}")
    
    try:
        # Detailed split breakdown
        cross_viz.plot_detailed_split_breakdown(
            split_info=results.get('split_info', {}),
            cross_subject_config=cross_subject_cfg,
            filename="detailed_split_breakdown.png"
        )
    except Exception as e:
        logger.error(f"Failed to create detailed split breakdown: {e}")
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    saver = ArtifactSaver(OUTPUT_DIR, logger)
    
    # Remove subjects_data from results before saving (too large for JSON)
    results_to_save = {k: v for k, v in results.items() if k != 'subjects_data'}
    
    # Save comprehensive results
    results_path = OUTPUT_DIR / "cross_subject_results.json"
    with open(results_path, "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)
    logger.info(f"Results saved: {results_path}")
    
    # Create metadata
    metadata = {
        "experiment_type": "cross_subject",
        "experiment_date": datetime.now().isoformat(),
        "config": {
            "cross_subject": {
                "train_subjects": cross_subject_cfg.train_subjects,
                "test_subject": cross_subject_cfg.test_subject,
                "exercise": cross_subject_cfg.exercise,
                "base_dir": str(cross_subject_cfg.base_dir),
                "pool_train_subjects": cross_subject_cfg.pool_train_subjects,
                "use_separate_val_subject": cross_subject_cfg.use_separate_val_subject,
                "val_subject": cross_subject_cfg.val_subject,
                "val_ratio": cross_subject_cfg.val_ratio,
                "seed": cross_subject_cfg.seed,
            },
            "processing": asdict(proc_cfg),
            "training": asdict(train_cfg),
            "split": asdict(split_cfg),
        },
        "results_summary": {
            "train_subjects": cross_subject_cfg.train_subjects,
            "test_subject": cross_subject_cfg.test_subject,
            "test_accuracy": results.get('cross_subject_test', {}).get('accuracy', 0.0),
            "test_f1_macro": results.get('cross_subject_test', {}).get('f1_macro', 0.0),
            "common_gestures": results.get('config', {}).get('common_gestures', []),
        },
        "per_subject_summary": {
            subject_id: {
                "accuracy": data.get("accuracy", 0.0),
                "f1_macro": data.get("f1_macro", 0.0),
                "role": "train" if data.get("is_train", False) else ("test" if data.get("is_test", False) else "val")
            }
            for subject_id, data in results.get('per_subject_analysis', {}).items()
        }
    }
    
    # Convert to JSON-serializable format
    metadata = make_json_serializable(metadata)
    
    saver.save_metadata(metadata, filename="cross_subject_metadata.json")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("Cross-Subject Experiment Completed Successfully")
    logger.info("=" * 80)
    logger.info(f"Training Subjects: {', '.join(cross_subject_cfg.train_subjects)}")
    logger.info(f"Test Subject: {cross_subject_cfg.test_subject}")
    logger.info(f"Exercise: {cross_subject_cfg.exercise}")
    logger.info("-" * 80)
    
    test_acc = results.get('cross_subject_test', {}).get('accuracy', 0.0)
    test_f1 = results.get('cross_subject_test', {}).get('f1_macro', 0.0)
    
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test F1-Macro: {test_f1:.4f}")
    logger.info("-" * 80)
    logger.info(f"Output directory: {OUTPUT_DIR.resolve()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()