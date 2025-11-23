from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import json
from pathlib import Path
import logging
import torch
import numpy as np

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    window_size: int = 500
    window_overlap: int = 0
    sampling_rate: int = 2000

    # If channel_indices is specified, it takes precedence.
    # Otherwise, if num_channels is specified, take the first K channels.
    # If both are None, take all channels.
    num_channels: Optional[int] = None
    channel_indices: Optional[List[int]] = None
    
    def save(self, path: Path):
        """Saving configuration"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    def get_selected_channel_indices(self, total_channels: int, logger: Optional[logging.Logger] = None) -> List[int]:
        """Determines which channels to use."""
        if self.channel_indices is not None:
            # filter and validate indexes
            idx = sorted(set(i for i in self.channel_indices if 0 <= i < total_channels))
            if logger:
                if len(idx) == 0:
                    raise ValueError("channel_indices is empty or out of range")
                invalid = set(self.channel_indices) - set(idx)
                if invalid:
                    logger.warning(f"Ignoring invalid channel indices: {sorted(invalid)}")
            return idx

        if self.num_channels is not None:
            if self.num_channels <= 0:
                raise ValueError("num_channels must be >= 1")
            k = min(self.num_channels, total_channels)
            if logger and k < self.num_channels:
                logger.warning(f"Requested {self.num_channels} channels available {total_channels}. "
                               f"The first {k} channels will be used.")
            return list(range(k))

        return list(range(total_channels))  

@dataclass
class TrainingConfig:
    """Training configuration. Window classification"""
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    early_stopping_patience: int = 7
    use_class_weights: bool = True
    seed: int = 42
    num_workers: int = 0  # for DataLoader
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)


@dataclass
class SplitConfig:
    """Spliting config on train/val/test"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Variant of split
    # "by_windows" - sequential division of windows within each segment
    # "by_segments" - entire segments (repetitions) are assigned entirely to one of the splits
    mode: str = "by_windows"

    # Params for by_segments variant
    shuffle_segments: bool = True
    seed: int = 42

    # whether to include the REST class (gid==0) in the split
    include_rest_in_splits: bool = False

    def validate(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"The sum of the shares should be 1.0, now {total}")
        if any(r < 0 for r in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("Shares cannot be negative")
        if self.mode not in ("by_windows", "by_segments"):
            raise ValueError("mode must be 'by_windows' or 'by_segments'")
        
@dataclass
class RotationConfig:
    rotations: List[int]                 # e.g. [-3,-2,-1,0,1,2,3]
    bracelet_size: Optional[int] = None  # default == number of channels C
    channel_order: Optional[List[int]] = None  # mapping input-channel-index -> bracelet position [0..bracelet_size-1]
    single_segment: Optional[Tuple[int, int]] = None  # (gesture_id, occurrence_idx) for per-segment analysis
