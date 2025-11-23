import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict
import logging

class NinaProLoader:
    """Loader for data NinaPro DB2"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def load_mat_file(self, file_path: Path) -> Dict:
        """Load .mat file"""
        self.logger.info(f"Load file: {file_path}")
        
        try:
            data = sio.loadmat(str(file_path))
            self.logger.info(f"File loaded. Keys: {[k for k in data.keys() if not k.startswith('__')]}")
            return data
        except Exception as e:
            self.logger.error(f"Error during file loading {file_path}: {e}")
            raise
    
    def extract_fields(self, data: Dict) -> Dict:
        """Extract relevant fields from data"""
        fields = {}
        
        for key in ['emg', 'acc', 'stimulus', 'force', 'repetition', 'restimulus', 'rerepetition']:
            if key in data:
                fields[key] = data[key]
                if isinstance(fields[key], np.ndarray):
                    self.logger.info(f"Filed '{key}': shape={fields[key].shape}, dtype={fields[key].dtype}")
            else:
                self.logger.warning(f"Filed '{key}' not find")
        
        return fields