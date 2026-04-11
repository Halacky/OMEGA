import torch
from torch.utils.data import Dataset
import numpy as np


class AugmentedWindowDataset(Dataset):
    """
    Window dataset with on-the-fly EMG augmentation in __getitem__.
    Expects X in (N, C, T) format; augmentation uses (N, T, C) internally.
    Use for training only; each epoch sees different augmented samples.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_std: float = 0.02,
        max_warp: float = 0.1,
        apply_noise: bool = True,
        apply_time_warp: bool = True,
    ):
        assert X.ndim == 3, f"Expect X shape (N, C, T), got {X.shape}"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.noise_std = noise_std
        self.max_warp = max_warp
        self.apply_noise = apply_noise
        self.apply_time_warp = apply_time_warp

    def __len__(self):
        return self.y.shape[0]

    def _augment_one(self, x_ct: np.ndarray) -> np.ndarray:
        # x_ct: (C, T) -> (1, T, C) for emg_augment
        from evaluation.emg_augmentation import emg_augment
        one = np.asarray(x_ct, dtype=np.float32)
        if one.ndim == 2:
            one = one[np.newaxis, :, :]  # (1, C, T) -> need (1, T, C)
        one = np.transpose(one, (0, 2, 1))  # (1, T, C)
        aug = emg_augment(
            one,
            noise_std=self.noise_std,
            max_warp=self.max_warp,
            apply_noise=self.apply_noise,
            apply_time_warp=self.apply_time_warp,
        )
        aug = np.transpose(aug, (0, 2, 1))  # (1, C, T)
        return aug[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (C, T)
        x_aug = self._augment_one(x)
        return torch.from_numpy(x_aug).float(), torch.tensor(self.y[idx], dtype=torch.long)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: array of shape (N, C, T) where C is channels, T is time/sequence length
            y: array of shape (N,) with class indices
        """
        assert X.ndim == 3, f"Expect X shape (N, C, T), got {X.shape}"
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeatureDataset(Dataset):
    """Dataset for feature-based models (2D input: N, F)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, f"Expect X shape (N, F), got {X.shape}"
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.astype(np.int64))
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]