import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cupy as cp
from typing import Dict, List
import logging
from config.base import ProcessingConfig

class WindowExtractor:
    """Extract window from signal"""
    
    def __init__(self, config: ProcessingConfig, logger: logging.Logger, use_gpu: bool = True):
        self.config = config
        self.logger = logger
        self.use_gpu = use_gpu and cp.cuda.is_available()

    def process_all_segments_grouped(self, segments: Dict[int, List[np.ndarray]]) -> Dict[int, List[np.ndarray]]:
        """
        Retuen window, grouped by gesture and repeats.
        {gesture_id: [windows_seg0, windows_seg1, ...]}
        """
        self.logger.info("Extracting windows while preserving repetition structure (by segment)")
        grouped: Dict[int, List[np.ndarray]] = {}

        for gesture_id, gesture_segments in segments.items():
            grouped[gesture_id] = []
            for seg_idx, segment in enumerate(gesture_segments):
                windows = self.extract_windows(segment)
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
            return np.array([])
        
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
                windows = self.extract_windows(segment)
                
                if len(windows) > 0:
                    gesture_windows.append(windows)
                    self.logger.info(f"Gesture {gesture_id}, segment {seg_idx}: {len(windows)} windows retrieved")
            
            if gesture_windows:
                all_windows[gesture_id] = np.concatenate(gesture_windows, axis=0)
                self.logger.info(f"Gesture {gesture_id}: total {len(all_windows[gesture_id])} windows")
        
        return all_windows