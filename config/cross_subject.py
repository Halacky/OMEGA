# FILE: config/cross_subject.py

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json

from .base import RotationConfig

@dataclass
class CrossSubjectConfig:
    train_subjects: List[str]
    test_subject: str
    exercises: List[str]
    base_dir: Path
    pool_train_subjects: bool = True
    use_separate_val_subject: bool = False
    val_subject: Optional[str] = None
    val_ratio: float = 0.15
    seed: int = 42
    use_rotation_augmentation: bool = False
    rotation_config: Optional[RotationConfig] = None
    augment_train_subjects_as_virtual_rotated_subjects: bool = False
    virtual_subject_rotation_config: Optional[RotationConfig] = None
    
    # НОВЫЙ ПАРАМЕТР
    max_gestures: Optional[int] = None  # None = все жесты, иначе берем последние N
    
    def __post_init__(self):
        if not isinstance(self.train_subjects, list) or len(self.train_subjects) == 0:
            raise ValueError("train_subjects must be a non-empty list")

        if self.use_rotation_augmentation and self.rotation_config is None:
            raise ValueError("rotation_config must be provided when use_rotation_augmentation=True")

        # validation for virtual rotated subjects
        if self.augment_train_subjects_as_virtual_rotated_subjects and self.virtual_subject_rotation_config is None:
            raise ValueError(
                "virtual_subject_rotation_config must be provided when "
                "augment_train_subjects_as_virtual_rotated_subjects=True"
            )

        if not self.test_subject:
            raise ValueError("test_subject must be specified")
        if self.test_subject in self.train_subjects:
            raise ValueError(f"test_subject {self.test_subject} cannot be in train_subjects")

        if self.use_separate_val_subject:
            if not self.val_subject:
                raise ValueError("val_subject must be specified when use_separate_val_subject=True")
            if self.val_subject in self.train_subjects:
                raise ValueError(f"val_subject {self.val_subject} cannot be in train_subjects")
            if self.val_subject == self.test_subject:
                raise ValueError("val_subject cannot be the same as test_subject")

        if not (0.0 < self.val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0 and 1")

    def save(self, path: Path):
        data = asdict(self)
        data['base_dir'] = str(self.base_dir)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: Path) -> 'CrossSubjectConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        data['base_dir'] = Path(data['base_dir'])
        return cls(**data)