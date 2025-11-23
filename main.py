import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime 
from dataclasses import asdict

from utils.logging import setup_logging, seed_everything
from config.base import ProcessingConfig, TrainingConfig, SplitConfig, RotationConfig
from data.loaders import NinaProLoader
from processing.segmentation import GestureSegmenter
from processing.windowing import WindowExtractor
from processing.splitting import DatasetSplitter
from training.trainer import WindowClassifierTrainer
from visualization.base import Visualizer
from visualization.rotation import RotationVisualizer
from evaluation.rotation import RotationExperiment
from utils.artifacts import ArtifactSaver

def main():
    # Configs
    BASE_DIR = Path("/home/kirill/projects_2/folium/NIR/OMEGA/data")
    SUBJECT = "DB2_s1"
    EXERCISE = "E3"
    OUTPUT_DIR = Path("./output_ninapro_rotation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 80)
    logger.info("Start NinaPro DB2 Pipeline + Rotation Experiment")
    logger.info("=" * 80)

    # Load data (reuse your loader)
    file_path = BASE_DIR / SUBJECT / f"S{SUBJECT.split('_s')[1]}_{EXERCISE}_A1.mat"
    loader = NinaProLoader(logger)
    raw_data = loader.load_mat_file(file_path)
    data_fields = loader.extract_fields(raw_data)
    emg = data_fields['emg']
    stimulus = data_fields['stimulus']

    # Processing config (unchanged)
    proc_cfg = ProcessingConfig(window_size=500, window_overlap=0, num_channels=8)  # use 8 bracelet sensors
    proc_cfg.save(OUTPUT_DIR / "config.json")

    # Select channels (use your implementation)
    selected_channels = proc_cfg.get_selected_channel_indices(total_channels=emg.shape[1], logger=logger)
    emg = emg[:, selected_channels]
    logger.info(f"Using channels: {selected_channels}. EMG shape: {emg.shape}, Stimulus: {stimulus.shape}")

    # Segment + window extraction (reuse your code)
    segmenter = GestureSegmenter(logger, use_gpu=True)
    segments = segmenter.segment_by_gestures(emg, stimulus, include_rest=True)

    extractor = WindowExtractor(proc_cfg, logger, use_gpu=True)
    windows_dict = extractor.process_all_segments(segments)
    grouped_windows = extractor.process_all_segments_grouped(segments)

    # Split by segments (as in your current main)
    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False
    )
    splitter = DatasetSplitter(split_cfg, logger)
    splits, assignments = splitter.split_grouped_windows(grouped_windows)

    # Visualizations (reuse your existing)
    viz = Visualizer(OUTPUT_DIR, logger)
    viz.plot_signal_overview(emg, stimulus)
    viz.plot_gesture_segments(segments)
    viz.plot_windows(windows_dict)
    viz.plot_statistics(windows_dict)
    viz.plot_windows_timeline(windows_dict, segments, emg, stimulus, proc_cfg)
    viz.plot_two_gestures_full_canvas(emg, stimulus, proc_cfg, grouped_windows, assignments, g1=45, g2=46)
    viz.plot_segments_split_timeline(emg, stimulus, assignments, filename="segments_split_timeline.png",
                                     sampling_rate=proc_cfg.sampling_rate)

    # Save base artifacts (reuse your saver)
    saver = ArtifactSaver(OUTPUT_DIR, logger)
    saver.save_segments(segments)
    saver.save_windows(windows_dict)

    # Train model (reuse your trainer.fit; but trainer now stores model/stats/class info)
    train_cfg = TrainingConfig(
        batch_size=256, epochs=50, learning_rate=1e-3, weight_decay=1e-4, dropout=0.3,
        early_stopping_patience=7, use_class_weights=True, seed=42, num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    trainer = WindowClassifierTrainer(train_cfg, logger, OUTPUT_DIR, viz)
    train_cfg.save(OUTPUT_DIR / "train_config.json")
    clf_results = trainer.fit(splits)

    # Rotation experiment configuration
    C = len(selected_channels)
    rot_cfg = RotationConfig(
        rotations=[-3, -2, -1, 0, 1, 2, 3],
        bracelet_size=C,           # set to C; set to 8 or any desired total sensor count
        channel_order=list(range(C)),  # define mapping model-channel -> bracelet position
        single_segment=None        # or set e.g., (gesture_id=45, occurrence=0)
    )
    rot_viz = RotationVisualizer(OUTPUT_DIR, logger)
    experiment = RotationExperiment(trainer, viz, rot_viz, logger, rot_cfg)

    # 1) Full test: baseline vs rotations
    rotation_to_metrics = experiment.evaluate_full_test_with_rotations(splits)
    if rotation_to_metrics:
        # Save summary JSON
        with open(OUTPUT_DIR / "rotation_test_metrics.json", "w") as f:
            json.dump({str(k): v for k, v in rotation_to_metrics.items()}, f, indent=4, ensure_ascii=False)

        # Plots
        class_labels = [("REST" if gid == 0 else f"Gesture {gid}") for gid in trainer.class_ids]
        rot_viz.plot_accuracy_vs_rotation(rotation_to_metrics, filename="acc_vs_rotation.png")

        # Show selected CMs
        show_rots = [r for r in sorted(rotation_to_metrics.keys()) if r in (-3, -1, 0, 1, 3)]
        if not show_rots:
            show_rots = sorted(rotation_to_metrics.keys())[:4]
        rot_viz.plot_cm_grid_for_rotations(rotation_to_metrics, class_labels, rotations_to_show=show_rots,
                                           filename="cm_grid_rotations.png", normalize=True)

    # 2) Single test segment: per-window probability heatmap vs rotation
    probs_by_rot, gid_sel, occ_sel = experiment.evaluate_single_test_segment_with_rotations(
        grouped_windows=grouped_windows,
        assignments=assignments,
        gesture_id=(rot_cfg.single_segment[0] if rot_cfg.single_segment else None),
        occurrence=(rot_cfg.single_segment[1] if rot_cfg.single_segment else None)
    )
    if probs_by_rot:
        true_cls_idx = trainer.class_ids.index(gid_sel) if gid_sel in trainer.class_ids else 0
        rot_viz.plot_trueclass_prob_heatmap_for_segment(
            probs_by_rotation=probs_by_rot,
            true_class_index=true_cls_idx,
            filename=f"segment_gid{gid_sel}_occ{occ_sel}_trueclass_prob_heatmap.png"
        )

    # Extend metadata
    metadata = {
        "subject": SUBJECT,
        "exercise": EXERCISE,
        "file_path": str(file_path),
        "processing_date": datetime.now().isoformat(),
        "config": asdict(proc_cfg),
        "emg_shape": emg.shape,
        "selected_channels": selected_channels,
        "num_gestures": len(segments),
        "total_windows": sum(len(w) for w in windows_dict.values()),
        "splits_summary": {
            split: {str(gid): int(len(arr)) for gid, arr in splits[split].items()
                    if isinstance(arr, np.ndarray) and arr.ndim == 3}
            for split in ["train", "val", "test"]
        },
        "classification": {
            "config": asdict(train_cfg),
            "val": clf_results.get("val", {}),
            "test": clf_results.get("test", {}),
            "classes": {
                "ids": trainer.class_ids,
                "names": [("REST" if gid == 0 else f"Gesture {gid}") for gid in trainer.class_ids],
            }
        },
        "rotation_experiment": {
            "config": asdict(rot_cfg),
            "test_metrics_by_rotation": rotation_to_metrics
        }
    }
    saver.save_metadata(metadata)

    logger.info("=" * 80)
    logger.info("Rotation experiment completed")
    logger.info(f"Artifacts saved to: {OUTPUT_DIR.resolve()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()