import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import logging

class ArtifactSaver:
    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_segments(self, segments: Dict[int, List[np.ndarray]], filename: str = "segments.npz"):
        save_dict = {}
        for gesture_id, segs in segments.items():
            for idx, seg in enumerate(segs):
                key = f"gesture_{gesture_id}_segment_{idx}"
                save_dict[key] = seg
        
        save_path = self.output_dir / filename
        np.savez_compressed(save_path, **save_dict)
        self.logger.info(f"Segments saved: {save_path}")
    
    def save_windows(self, windows_dict: Dict[int, np.ndarray], filename: str = "windows.npz"):
        save_dict = {f"gesture_{gid}": windows for gid, windows in windows_dict.items()}
        
        save_path = self.output_dir / filename
        np.savez_compressed(save_path, **save_dict)
        self.logger.info(f"Windows saved: {save_path}")
    
    def save_metadata(self, data: Dict, filename: str = "metadata.json"):
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        self.logger.info(f"Metadata saved: {save_path}")