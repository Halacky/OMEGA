import numpy as np
from typing import Tuple


def add_gaussian_noise(
    X: np.ndarray,
    noise_std: float = 0.01,
    per_sample: bool = False,
) -> np.ndarray:
    """
    Additive Gaussian noise augmentation for EMG windows.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, T, C).
    noise_std : float
        Standard deviation of noise relative to per-channel std (if per_sample=False)
        or relative to each sample value (if per_sample=True) in normalized units.
    per_sample : bool
        If True, noise std is scaled by |x_ij|; otherwise scaled by global channel std.

    Returns
    -------
    X_noisy : np.ndarray
        Array with the same shape as X.
    """
    if not (isinstance(X, np.ndarray) and X.ndim == 3):
        raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

    X_noisy = X.copy().astype(np.float32)
    N, T, C = X_noisy.shape

    if per_sample:
        # Noise std proportional to absolute value of sample
        eps = 1e-6
        scale = np.abs(X_noisy) + eps
        noise = np.random.randn(N, T, C).astype(np.float32) * (noise_std * scale)
    else:
        # Noise std proportional to channel-wise std across the dataset
        ch_std = X_noisy.std(axis=(0, 1), keepdims=True) + 1e-6
        noise = np.random.randn(N, T, C).astype(np.float32) * (noise_std * ch_std)

    X_noisy += noise
    return X_noisy


def random_time_warp(
    X: np.ndarray,
    max_warp: float = 0.1,
) -> np.ndarray:
    """
    Simple time-warping augmentation using random time scaling.

    For each window, we:
      1) Sample a random scale factor s in [1 - max_warp, 1 + max_warp]
      2) Resample the time axis using linear interpolation
      3) Keep the original window length T

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, T, C).
    max_warp : float
        Maximum relative warp factor; e.g. 0.1 -> scale in [0.9, 1.1].

    Returns
    -------
    X_warped : np.ndarray
        Array with the same shape as X.
    """
    if not (isinstance(X, np.ndarray) and X.ndim == 3):
        raise ValueError(f"Expected X shape (N, T, C), got {X.shape}")

    N, T, C = X.shape
    X_warped = np.empty_like(X, dtype=np.float32)

    original_t = np.arange(T, dtype=np.float32)

    for i in range(N):
        scale = 1.0 + (2.0 * np.random.rand() - 1.0) * max_warp
        # Compressed or stretched time axis
        new_t = np.linspace(0, T - 1, int(T * scale), dtype=np.float32)
        # Interpolate each channel separately, then resample back to length T
        warped = np.empty((new_t.shape[0], C), dtype=np.float32)
        for c in range(C):
            warped[:, c] = np.interp(new_t, original_t, X[i, :, c])

        # Now resample warped signal back to original T
        target_t = np.linspace(0, warped.shape[0] - 1, T, dtype=np.float32)
        for c in range(C):
            X_warped[i, :, c] = np.interp(target_t, np.arange(warped.shape[0], dtype=np.float32), warped[:, c])

    return X_warped


def emg_augment(
    X: np.ndarray,
    noise_std: float = 0.01,
    max_warp: float = 0.1,
    apply_noise: bool = True,
    apply_time_warp: bool = True,
) -> np.ndarray:
    """
    Combined EMG augmentation (noise + time warping).

    Parameters
    ----------
    X : np.ndarray
        Input EMG windows (N, T, C).
    noise_std : float
        Noise strength for add_gaussian_noise.
    max_warp : float
        Max time warp factor for random_time_warp.
    apply_noise : bool
        If True, apply Gaussian noise.
    apply_time_warp : bool
        If True, apply time warping.

    Returns
    -------
    X_aug : np.ndarray
        Augmented EMG windows with the same shape as X.
    """
    X_aug = X.astype(np.float32)
    if apply_time_warp:
        X_aug = random_time_warp(X_aug, max_warp=max_warp)
    if apply_noise:
        X_aug = add_gaussian_noise(X_aug, noise_std=noise_std, per_sample=False)
    return X_aug