import logging
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False
import numpy as np
from typing import Dict, List

class GestureSegmenter:
    """Segmentation of signal by gestures"""

    def __init__(self, logger: logging.Logger, use_gpu: bool = True):
        self.logger = logger
        self.use_gpu = use_gpu and _HAS_CUPY and cp.cuda.is_available()
        
        if self.use_gpu:
            self.logger.info("GPU enable, use CuPy")
        else:
            self.logger.info("GPU disabled, using NumPy")
    
    def segment_by_gestures(self, emg: np.ndarray, stimulus: np.ndarray, include_rest: bool = True) -> Dict[int, List[np.ndarray]]:
        """
        Segmentation EMG signal per gesture
        
        Args:
            emg: EMG data (samples, channels)
            stimulus: Gesture lables(samples, 1)
            include_rest: Could be apply rest (gesture_id = 0) like own class
        
        Returns:
            Dict {gesture_id: [segment1, segment2, ...]}
        """
        self.logger.info(f"Start segmentation by gesture (include_rest={include_rest})")
        
        stimulus = stimulus.flatten()
        
        stimulus_diff = np.diff(stimulus, prepend=0)
        gesture_changes = np.where(stimulus_diff != 0)[0]
        
        segments = {}
        
        for i in range(len(gesture_changes) - 1):
            start_idx = gesture_changes[i]
            end_idx = gesture_changes[i + 1]
            gesture_id = int(stimulus[start_idx])
            
            if gesture_id == 0 and not include_rest:
                continue
            
            segment = emg[start_idx:end_idx]
            
            if gesture_id not in segments:
                segments[gesture_id] = []
            
            segments[gesture_id].append(segment)
        
        if len(gesture_changes) > 0:
            start_idx = gesture_changes[-1]
            gesture_id = int(stimulus[start_idx])
            if include_rest or gesture_id != 0:
                segment = emg[start_idx:]
                if gesture_id not in segments:
                    segments[gesture_id] = []
                segments[gesture_id].append(segment)
        
        self.logger.info(f"Finded {len(segments)} different classes")
        for gesture_id, segs in segments.items():
            label = "REST" if gesture_id == 0 else f"Gesture {gesture_id}"
            self.logger.info(f"  {label}: {len(segs)} segments")
        
        return segments