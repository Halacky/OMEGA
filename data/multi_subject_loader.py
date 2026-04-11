import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from data.loaders import NinaProLoader
from processing.segmentation import GestureSegmenter
from processing.windowing import WindowExtractor
from config.base import ProcessingConfig
from processing.improved_processing import ImprovedSegmentProcessor

class MultiSubjectLoader:
    def __init__(self, 
                processing_config: ProcessingConfig,
                logger: logging.Logger,
                use_gpu: bool = True,
                use_improved_processing: bool = False):  # <- ADD THIS
        self.proc_cfg = processing_config
        self.logger = logger
        self.use_gpu = use_gpu
        self.use_improved_processing = use_improved_processing  # <- ADD THIS
        
        self.loader = NinaProLoader(logger)
        self.segmenter = GestureSegmenter(logger, use_gpu=use_gpu)
        self.extractor = WindowExtractor(processing_config, logger, use_gpu=use_gpu)
        
        # ADD THIS BLOCK:
        if use_improved_processing:
            self.segment_processor = ImprovedSegmentProcessor(
                sampling_rate=processing_config.sampling_rate,
                logger=logger,
                use_saturation_fix=True,
                normalization_method='percentile'
            )
    
    def load_subject(self, 
                     base_dir: Path,
                     subject_id: str,
                     exercise: str,
                     include_rest: bool = True) -> Tuple[np.ndarray, Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
        """
        Load and process a single subject
        
        Returns:
            emg: raw EMG data (for reference)
            segments: segmented data by gesture
            grouped_windows: windows grouped by gesture and repetition
        """
        self.logger.info(f"Loading subject: {subject_id}, exercise: {exercise}")
        
        # Construct file path
        subject_num = subject_id.split('_s')[1]
        file_path = base_dir / subject_id / f"S{subject_num}_{exercise}_A1.mat"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        raw_data = self.loader.load_mat_file(file_path)
        data_fields = self.loader.extract_fields(raw_data)
        emg = data_fields['emg']
        stimulus = data_fields['stimulus']
        
        # Select channels
        selected_channels = self.proc_cfg.get_selected_channel_indices(
            total_channels=emg.shape[1], 
            logger=self.logger
        )
        emg = emg[:, selected_channels]
        
        self.logger.info(f"Subject {subject_id}: EMG shape={emg.shape}, channels={selected_channels}")
        
        # Segment by gestures
        segments = self.segmenter.segment_by_gestures(emg, stimulus, include_rest=include_rest)
        if self.use_improved_processing:
            self.logger.info(f"Applying improved processing for subject {subject_id}")
            segments = self.segment_processor.process_all_segments(segments)

            bad_segments = 0
            for gid, reps in segments.items():
                for r_idx, seg in enumerate(reps):
                    if not np.all(np.isfinite(seg)):
                        bad_segments += 1
                        self.logger.warning(
                            f"[ImprovedProcessing] NaN/Inf в сегменте: "
                            f"subject={subject_id}, gesture={gid}, rep={r_idx}, "
                            f"shape={seg.shape}, "
                            f"min={np.nanmin(seg)}, max={np.nanmax(seg)}"
                        )
                        # можно либо выкинуть сегмент, либо починить:
                        segments[gid][r_idx] = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
            if bad_segments > 0:
                self.logger.warning(
                    f"[ImprovedProcessing] Обнаружено и исправлено {bad_segments} сегментов с NaN/Inf"
                )
        
        grouped_windows = self.extractor.process_all_segments_grouped(segments)
        
        total_windows = sum(sum(len(w) for w in segs) for segs in grouped_windows.values())
        self.logger.info(f"Subject {subject_id}: {len(segments)} gestures, {total_windows} windows total")
        
        return emg, segments, grouped_windows
    

    def load_subject_multiple_exercises(
        self,
        base_dir: Path,
        subject_id: str,
        exercises: List[str],
        include_rest: bool = True,
    ) -> Tuple[np.ndarray, Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
        """
        Загрузить и объединить данные по нескольким упражнениям для одного субъекта.
        
        Данные по упражнениям конкатенируются по времени.
        """
        self.logger.info(
            f"Loading subject: {subject_id}, exercises: {exercises}"
        )

        base_dir = Path(base_dir) if not isinstance(base_dir, Path) else base_dir
        subject_num = subject_id.split('_s')[1]

        emg_list = []
        stim_list = []

        try:
            for exercise in exercises:
                file_path = base_dir / subject_id / f"S{subject_num}_{exercise}_A1.mat"
                if not file_path.exists():
                    raise FileNotFoundError(f"Data file not found: {file_path}")

                raw_data = self.loader.load_mat_file(file_path)
                data_fields = self.loader.extract_fields(raw_data)
                emg_part = data_fields['emg']
                stimulus_part = data_fields['stimulus']

                emg_list.append(emg_part)
                stim_list.append(stimulus_part)
        except Exception as e:
            self.logger.error(f"Error loading {subject_id} exercises {exercises}: {e}")
            raise
        # конкатенация по времени
        emg = np.concatenate(emg_list, axis=0)
        stimulus = np.concatenate(stim_list, axis=0)

        # Select channels
        selected_channels = self.proc_cfg.get_selected_channel_indices(
            total_channels=emg.shape[1],
            logger=self.logger
        )
        emg = emg[:, selected_channels]

        self.logger.info(
            f"Subject {subject_id}: EMG shape={emg.shape}, "
            f"channels={selected_channels}, exercises={exercises}"
        )

        # Segment by gestures
        segments = self.segmenter.segment_by_gestures(
            emg, stimulus, include_rest=include_rest
        )

        if self.use_improved_processing:
            self.logger.info(f"Applying improved processing for subject {subject_id}")
            segments = self.segment_processor.process_all_segments(segments)

            bad_segments = 0
            for gid, reps in segments.items():
                for r_idx, seg in enumerate(reps):
                    if not np.all(np.isfinite(seg)):
                        bad_segments += 1
                        self.logger.warning(
                            f"[ImprovedProcessing] NaN/Inf в сегменте: "
                            f"subject={subject_id}, gesture={gid}, rep={r_idx}, "
                            f"shape={seg.shape}, "
                            f"min={np.nanmin(seg)}, max={np.nanmax(seg)}"
                        )
                        segments[gid][r_idx] = np.nan_to_num(
                            seg, nan=0.0, posinf=0.0, neginf=0.0
                        )
            if bad_segments > 0:
                self.logger.warning(
                    f"[ImprovedProcessing] Обнаружено и исправлено "
                    f"{bad_segments} сегментов с NaN/Inf"
                )

        grouped_windows = self.extractor.process_all_segments_grouped(segments)

        total_windows = sum(
            sum(len(w) for w in segs) for segs in grouped_windows.values()
        )
        self.logger.info(
            f"Subject {subject_id}: {len(segments)} gestures, "
            f"{total_windows} windows total (exercises={exercises})"
        )

        return emg, segments, grouped_windows
    
    def load_multiple_subjects(self,
                               base_dir: Path,
                               subject_ids: List[str],
                               exercises: List[str],
                               include_rest: bool = True
                               ) -> Dict[str, Tuple[np.ndarray, Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]]:
        """
        Load multiple subjects (possibly with multiple exercises per subject).

        Args:
            base_dir: base data directory
            subject_ids: list of subject IDs
            exercises: list of exercise codes, e.g. ["E1", "E2"]
            include_rest: whether to keep rest class

        Returns:
            Dict mapping subject_id -> (emg, segments, grouped_windows)
        """
        self.logger.info(
            f"Loading {len(subject_ids)} subjects: {subject_ids}, "
            f"exercises={exercises}"
        )
        
        subjects_data = {}
        for subject_id in subject_ids:
            try:
                data = self.load_subject_multiple_exercises(
                    base_dir=base_dir,
                    subject_id=subject_id,
                    exercises=exercises,
                    include_rest=include_rest,
                )
                subjects_data[subject_id] = data
            except Exception as e:
                self.logger.error(f"Failed to load subject {subject_id}: {e}")
                raise
        
        self.logger.info(f"Successfully loaded {len(subjects_data)} subjects")
        return subjects_data
    
    def merge_grouped_windows(self,
                             subjects_data: Dict[str, Tuple],
                             subject_ids: List[str]) -> Dict[int, List[np.ndarray]]:
        """
        Merge grouped windows from multiple subjects into a single structure
        
        Args:
            subjects_data: output from load_multiple_subjects
            subject_ids: which subjects to merge
        
        Returns:
            merged_grouped_windows: {gesture_id: [windows_rep0, windows_rep1, ...]}
            All repetitions from all subjects concatenated
        """
        self.logger.info(f"Merging grouped windows from subjects: {subject_ids}")
        
        merged = {}
        
        for subject_id in subject_ids:
            if subject_id not in subjects_data:
                self.logger.warning(f"Subject {subject_id} not in loaded data, skipping")
                continue
            
            _, _, grouped_windows = subjects_data[subject_id]
            
            for gesture_id, repetitions in grouped_windows.items():
                if gesture_id not in merged:
                    merged[gesture_id] = []
                
                # Add all repetitions from this subject
                merged[gesture_id].extend(repetitions)
        
        # Log statistics
        for gesture_id in sorted(merged.keys()):
            num_reps = len(merged[gesture_id])
            total_windows = sum(len(rep) for rep in merged[gesture_id])
            self.logger.info(f"Gesture {gesture_id}: {num_reps} repetitions, {total_windows} windows")
        
        return merged
    
    def filter_top_gestures(
        self,
        grouped_windows: Dict[int, List[np.ndarray]],
        max_gestures: int,
        exclude_rest: bool = True
    ) -> Dict[int, List[np.ndarray]]:
        """
        Оставляет только последние N жестов по ID (наибольшие номера).
        
        Args:
            grouped_windows: Словарь с окнами по жестам
            max_gestures: Максимальное количество жестов
            exclude_rest: Исключать ли REST (жест 0) из подсчета
        
        Returns:
            Отфильтрованный словарь
        """
        all_gesture_ids = sorted(grouped_windows.keys())
        
        if exclude_rest and 0 in all_gesture_ids:
            # REST не учитываем в лимите
            non_rest_ids = [gid for gid in all_gesture_ids if gid != 0]
            selected_ids = non_rest_ids[-max_gestures:]  # Берем последние N
            if 0 in all_gesture_ids:
                selected_ids = [0] + selected_ids  # Добавляем REST обратно
        else:
            selected_ids = all_gesture_ids[-max_gestures:]
        
        filtered = {
            gid: grouped_windows[gid] 
            for gid in selected_ids 
            if gid in grouped_windows
        }
        
        self.logger.info(
            f"Filtered gestures: kept {len(filtered)} out of {len(grouped_windows)} "
            f"(IDs: {sorted(filtered.keys())})"
        )
        
        return filtered
    
    def get_common_gestures(
        self, 
        subjects_data: Dict[str, Tuple],
        max_gestures: Optional[int] = None
    ) -> List[int]:
        """
        Находит общие жесты для всех субъектов, опционально ограничивая количество.
        """
        if not subjects_data:
            return []
        
        gesture_sets = []
        for subject_id, (_, _, grouped_windows) in subjects_data.items():
            gesture_sets.append(set(grouped_windows.keys()))
        
        common_gestures = set.intersection(*gesture_sets) if gesture_sets else set()
        common_gestures_list = sorted(list(common_gestures))
        
        # НОВАЯ ЛОГИКА: фильтрация по max_gestures
        if max_gestures is not None and max_gestures > 0:
            # Отделяем REST от остальных жестов
            has_rest = 0 in common_gestures_list
            non_rest = [g for g in common_gestures_list if g != 0]
            
            # Берем последние N жестов (наибольшие ID)
            selected_non_rest = non_rest[-max_gestures:]
            
            # Добавляем REST обратно, если он был
            if has_rest:
                common_gestures_list = [0] + selected_non_rest
            else:
                common_gestures_list = selected_non_rest
            
            self.logger.info(
                f"Limited to {max_gestures} gestures (excluding REST): "
                f"{common_gestures_list}"
            )
        
        self.logger.info(
            f"Common gestures across all subjects: {common_gestures_list}"
        )
        
        return common_gestures_list
    
    def filter_by_gestures(self,
                          grouped_windows: Dict[int, List[np.ndarray]],
                          gesture_ids: List[int]) -> Dict[int, List[np.ndarray]]:
        """
        Filter grouped_windows to only include specified gestures
        """
        filtered = {gid: reps for gid, reps in grouped_windows.items() if gid in gesture_ids}
        self.logger.info(f"Filtered to {len(filtered)} gestures: {sorted(filtered.keys())}")
        return filtered
    

