# FILE: evaluation/cross_subject.py

from typing import Dict, Tuple, List
import numpy as np
import logging

from config.cross_subject import CrossSubjectConfig  
from config.base import SplitConfig                  
from data.multi_subject_loader import MultiSubjectLoader  
from training.trainer import WindowClassifierTrainer       
from visualization.base import Visualizer                 
from processing.splitting import DatasetSplitter          
from .augmentation import augment_grouped_windows_with_rotations, create_virtual_rotated_subjects  

class CrossSubjectExperiment:
    def __init__(self,
                 cross_subject_config: CrossSubjectConfig,
                 split_config: SplitConfig,
                 multi_subject_loader: MultiSubjectLoader,
                 trainer: WindowClassifierTrainer,
                 visualizer: Visualizer,
                 logger: logging.Logger):
        self.cs_cfg = cross_subject_config
        self.split_cfg = split_config
        self.loader = multi_subject_loader
        self.trainer = trainer
        self.visualizer = visualizer
        self.logger = logger

    def run(self) -> Dict:
        self.logger.info("=" * 80)
        self.logger.info("Starting Cross-Subject Experiment")
        self.logger.info(f"Train subjects: {self.cs_cfg.train_subjects}")
        self.logger.info(f"Test subject: {self.cs_cfg.test_subject}")
        self.logger.info("=" * 80)

        all_subject_ids = []
        for sid in (
            self.cs_cfg.train_subjects
            + [self.cs_cfg.test_subject]
            + ([self.cs_cfg.val_subject] if self.cs_cfg.val_subject else [])
        ):
            if sid not in all_subject_ids:
                all_subject_ids.append(sid)

        subjects_data = self.loader.load_multiple_subjects(
            base_dir=self.cs_cfg.base_dir,
            subject_ids=all_subject_ids,
            exercises=self.cs_cfg.exercises,
            include_rest=self.split_cfg.include_rest_in_splits
        )
        
        # ОБНОВЛЕННЫЙ ВЫЗОВ с max_gestures
        common_gestures = self.loader.get_common_gestures(
            subjects_data,
            max_gestures=self.cs_cfg.max_gestures  # Передаем параметр
        )
        
        if len(common_gestures) == 0:
            raise ValueError("No common gestures found across subjects")
        
        self.logger.info(
            f"Using {len(common_gestures)} common gestures: {common_gestures}"
        )
        if self.cs_cfg.augment_train_subjects_as_virtual_rotated_subjects and \
           self.cs_cfg.virtual_subject_rotation_config is not None:
            self.logger.info(
                "Augmenting train subjects by creating virtual rotated subjects "
                "(subject-level augmentation)."
            )
            subjects_data = create_virtual_rotated_subjects(
                subjects_data=subjects_data,
                train_subject_ids=self.cs_cfg.train_subjects,
                rot_cfg=self.cs_cfg.virtual_subject_rotation_config,
                logger=self.logger,
            )
        else:
            self.logger.info(
                "Subject-level rotation augmentation is disabled for this experiment"
            )

        splits, split_info = self._prepare_splits(subjects_data, common_gestures)

        self.logger.info("Training model on train subjects...")
        training_results = self.trainer.fit(splits)
        if training_results is None:
            self.logger.warning("Training returned None results, using empty dict")
            training_results = {}

        self.logger.info(f"Evaluating on test subject: {self.cs_cfg.test_subject}")
        X_test = splits["test"]
        y_test_dict = {}
        for gid in common_gestures:
            if gid in X_test and len(X_test[gid]) > 0:
                cls_idx = self.trainer.class_ids.index(gid)
                y_test_dict[gid] = np.full((len(X_test[gid]),), cls_idx, dtype=np.int64)
        X_test_concat = np.concatenate(
            [X_test[gid] for gid in common_gestures if gid in X_test and len(X_test[gid]) > 0],
            axis=0
        )
        y_test_concat = np.concatenate(
            [y_test_dict[gid] for gid in common_gestures if gid in y_test_dict],
            axis=0
        )
        test_results = self.trainer.evaluate_numpy(
            X_test_concat,
            y_test_concat,
            split_name=f"cross_subject_test_{self.cs_cfg.test_subject}",
            visualize=True
        )

        per_subject_results = self._evaluate_per_subject(subjects_data, common_gestures)

        results = {
            "config": {
                "train_subjects": self.cs_cfg.train_subjects,
                "test_subject": self.cs_cfg.test_subject,
                "val_subject": self.cs_cfg.val_subject,
                "exercises": self.cs_cfg.exercises,
                "common_gestures": common_gestures,
            },
            "split_info": split_info,
            "training": training_results,
            "cross_subject_test": {
                "subject": self.cs_cfg.test_subject,
                "accuracy": test_results["accuracy"],
                "f1_macro": test_results["f1_macro"],
                "report": test_results["report"],
                "confusion_matrix": test_results["confusion_matrix"],
            },
            "per_subject_analysis": per_subject_results,
            "subjects_data": subjects_data,
        }
        self.logger.info("=" * 80)
        self.logger.info("Cross-Subject Experiment Complete")
        self.logger.info(f"Test Subject: {self.cs_cfg.test_subject}")
        self.logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        self.logger.info(f"Test F1-Macro: {test_results['f1_macro']:.4f}")
        self.logger.info("=" * 80)
        return results

    def _prepare_splits(self,
                        subjects_data: Dict[str, Tuple],
                        common_gestures: List[int]) -> Tuple[Dict, Dict]:
        # train subjects could now include
        # virtual rotated subjects, but we still call merge_grouped_windows by explicit ids
        filtered_data = {}
        for subject_id, (emg, segments, grouped_windows) in subjects_data.items():
            filtered_grouped = self.loader.filter_by_gestures(grouped_windows, common_gestures)
            filtered_data[subject_id] = filtered_grouped

        # Build final list of train subject IDs:
        # It should include original train subjects AND any virtual subjects we created
        # that start with "<orig_id>_rot_".
        train_ids_for_merge: List[str] = []
        for sid in subjects_data.keys():
            if sid in self.cs_cfg.train_subjects:
                train_ids_for_merge.append(sid)
            else:
                # virtual subject detection: sid like "DB2_s1_rot_-3"
                for orig_sid in self.cs_cfg.train_subjects:
                    if sid.startswith(f"{orig_sid}_rot_"):
                        train_ids_for_merge.append(sid)
                        break

        self.logger.info(f"Train subjects for merge (including virtual): {train_ids_for_merge}")

        train_grouped = self.loader.merge_grouped_windows(
            {sid: (None, None, filtered_data[sid]) for sid in filtered_data
             if sid in train_ids_for_merge},
            train_ids_for_merge
        )

        if self.cs_cfg.use_rotation_augmentation and self.cs_cfg.rotation_config is not None:
            self.logger.info("Applying rotation-based data augmentation to train subjects (window-level)...")
            train_grouped = augment_grouped_windows_with_rotations(
                train_grouped=train_grouped,
                rot_cfg=self.cs_cfg.rotation_config,
                logger=self.logger
            )
        else:
            self.logger.info("Rotation-based window-level augmentation is disabled for this experiment")

        if self.cs_cfg.use_separate_val_subject and self.cs_cfg.val_subject:
            val_grouped = filtered_data[self.cs_cfg.val_subject]
            self.logger.info(f"Using separate validation subject: {self.cs_cfg.val_subject}")
            train_splits = self._grouped_to_single_split(train_grouped, "train")
            val_splits = self._grouped_to_single_split(val_grouped, "val")
            splits = {
                "train": train_splits,
                "val": val_splits,
                "test": self._grouped_to_single_split(filtered_data[self.cs_cfg.test_subject], "test")
            }
        else:
            self.logger.info(f"Splitting train subjects with val_ratio={self.cs_cfg.val_ratio}")
            splitter = DatasetSplitter(self.split_cfg, self.logger)
            train_val_splits, _ = splitter.split_grouped_windows(train_grouped)
            splits = {
                "train": train_val_splits["train"],
                "val": train_val_splits["val"],
                "test": self._grouped_to_single_split(filtered_data[self.cs_cfg.test_subject], "test")
            }

        split_info = {}
        for split_name in ["train", "val", "test"]:
            total_windows = sum(len(arr) for arr in splits[split_name].values()
                                if isinstance(arr, np.ndarray) and arr.ndim == 3)
            split_info[split_name] = {
                "total_windows": total_windows,
                "gestures": sorted(list(splits[split_name].keys())),
                "per_gesture": {
                    int(gid): int(len(arr)) for gid, arr in splits[split_name].items()
                    if isinstance(arr, np.ndarray) and arr.ndim == 3
                }
            }
            self.logger.info(f"{split_name.upper()}: {total_windows} windows across "
                             f"{len(split_info[split_name]['gestures'])} gestures")
        return splits, split_info

    def _grouped_to_single_split(self, grouped_windows: Dict[int, List[np.ndarray]], split_name: str) -> Dict[int, np.ndarray]:
        # unchanged
        result = {}
        for gesture_id, repetitions in grouped_windows.items():
            if len(repetitions) > 0:
                result[gesture_id] = np.concatenate(repetitions, axis=0)
            else:
                result[gesture_id] = np.empty((0,), dtype=np.float32)
        return result

    def _evaluate_per_subject(self,
                              subjects_data: Dict[str, Tuple],
                              common_gestures: List[int]) -> Dict:
        self.logger.info("Performing per-subject evaluation...")
        per_subject_results = {}
        for subject_id in subjects_data.keys():
            _, _, grouped_windows = subjects_data[subject_id]
            filtered_grouped = self.loader.filter_by_gestures(grouped_windows, common_gestures)
            X_list, y_list = [], []
            for gid in sorted(common_gestures):
                if gid in filtered_grouped:
                    reps = filtered_grouped[gid]
                    if len(reps) > 0:
                        X_concat = np.concatenate(reps, axis=0)
                        cls_idx = self.trainer.class_ids.index(gid)
                        y_concat = np.full((len(X_concat),), cls_idx, dtype=np.int64)
                        X_list.append(X_concat)
                        y_list.append(y_concat)
            if len(X_list) == 0:
                self.logger.warning(f"No data for subject {subject_id}, skipping")
                continue
            X_subject = np.concatenate(X_list, axis=0)
            y_subject = np.concatenate(y_list, axis=0)
            results = self.trainer.evaluate_numpy(
                X_subject,
                y_subject,
                split_name=f"subject_{subject_id}",
                visualize=False
            )
            per_subject_results[subject_id] = {
                "accuracy": results["accuracy"],
                "f1_macro": results["f1_macro"],
                "num_samples": len(X_subject),
                "is_train": subject_id in self.cs_cfg.train_subjects,
                "is_test": subject_id == self.cs_cfg.test_subject,
            }
            self.logger.info(
                f"Subject {subject_id}: accuracy={results['accuracy']:.4f}, "
                f"f1_macro={results['f1_macro']:.4f}"
            )
        return per_subject_results