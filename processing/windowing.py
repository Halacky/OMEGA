import numpy as np
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False
from typing import Dict, List
import logging
from config.base import ProcessingConfig

class WindowExtractor:
    """Extract window from signal"""

    def __init__(self, config: ProcessingConfig, logger: logging.Logger, use_gpu: bool = True):
        self.config = config
        self.logger = logger
        self.use_gpu = use_gpu and _HAS_CUPY and cp.cuda.is_available()

    def _apply_segment_edge_margin(self, segment: np.ndarray) -> np.ndarray:
        """
        Optionally drop a small fraction of samples at the beginning and the end
        of the segment to avoid transition regions (rest <-> gesture).

        The amount is controlled by ProcessingConfig.segment_edge_margin:
        - 0.0: disable trimming (backward compatible)
        - 0.1: drop first 10% and last 10% of samples
        """
        margin = getattr(self.config, "segment_edge_margin", 0.0)
        if margin <= 0.0:
            return segment

        T = segment.shape[0]
        cut = int(T * margin)
        # If the segment is too short, skip trimming
        if cut <= 0 or cut * 2 >= T:
            return segment

        return segment[cut:-cut, :]

    def process_all_segments_grouped(self, segments: Dict[int, List[np.ndarray]]) -> Dict[int, List[np.ndarray]]:
        """
        Return window, grouped by gesture and repeats.
        {gesture_id: [windows_seg0, windows_seg1, ...]}
        """
        self.logger.info("Extracting windows while preserving repetition structure (by segment)")
        grouped: Dict[int, List[np.ndarray]] = {}

        for gesture_id, gesture_segments in segments.items():
            grouped[gesture_id] = []
            for seg_idx, segment in enumerate(gesture_segments):
                # Apply segment edge margin before window extraction
                segment_trimmed = self._apply_segment_edge_margin(segment)
                windows = self.extract_windows(segment_trimmed)
                grouped[gesture_id].append(windows)
                if len(windows) > 0:
                    self.logger.info(
                        f"Gesture {gesture_id}, segment {seg_idx}: {len(windows)} windows retrieved (grouped)"
                    )
        return grouped
    
    def extract_windows(self, segment: np.ndarray) -> np.ndarray:
        """
        Extracting windows from an overlapping segment
        
        Args:
            segment: Segment EMG (samples, channels)
        
        Returns:
            Window array (num_windows, window_size, channels)
        """
        num_samples = segment.shape[0]
        num_channels = segment.shape[1]
        
        step = self.config.window_size - self.config.window_overlap
        
        num_windows = (num_samples - self.config.window_size) // step + 1
        
        if num_windows <= 0:
            self.logger.warning(f"The segment is too short ({num_samples} samples), skipping")
            return np.empty((0, self.config.window_size, segment.shape[1]))
        
        windows = []
        
        if self.use_gpu:
            segment_gpu = cp.asarray(segment)
            
            for i in range(num_windows):
                start_idx = i * step
                end_idx = start_idx + self.config.window_size
                window = segment_gpu[start_idx:end_idx]
                windows.append(cp.asnumpy(window))
        else:
            for i in range(num_windows):
                start_idx = i * step
                end_idx = start_idx + self.config.window_size
                window = segment[start_idx:end_idx]
                windows.append(window)
        
        return np.array(windows)
    
    def process_all_segments(self, segments: Dict[int, List[np.ndarray]]) -> Dict[int, np.ndarray]:
        """
        Processing all segments and extracting windows
        
        Returns:
            Dict {gesture_id: windows_array}
        """
        self.logger.info("Start extracting windows from segments")
        
        all_windows = {}
        
        for gesture_id, gesture_segments in segments.items():
            gesture_windows = []
            
            for seg_idx, segment in enumerate(gesture_segments):
                # Apply segment edge margin before window extraction
                segment_trimmed = self._apply_segment_edge_margin(segment)
                windows = self.extract_windows(segment_trimmed)
                
                if len(windows) > 0:
                    gesture_windows.append(windows)
                    self.logger.info(f"Gesture {gesture_id}, segment {seg_idx}: {len(windows)} windows retrieved")
            
            if gesture_windows:
                all_windows[gesture_id] = np.concatenate(gesture_windows, axis=0)
                self.logger.info(f"Gesture {gesture_id}: total {len(all_windows[gesture_id])} windows")
        
        return all_windows