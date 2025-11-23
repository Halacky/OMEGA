import torch
from torch.utils.data import Dataset
import numpy as np

class WindowDataset(Dataset):
    """PyTorch Dataset for window EMG"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, T, C) float32
        y: (N,) int64
        Transform в (N, C, T) for Conv1d
        """
        assert X.ndim == 3, "Expect X shape (N, T, C)"
        self.X = torch.from_numpy(np.transpose(X, (0, 2, 1))).float()
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]