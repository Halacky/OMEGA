"""
Experiment 78: Cross-Spectral Density (CPSD) Riemannian Features for Cross-Subject EMG (LOSO)

Hypothesis H78:
    Inter-subject EMG amplitude differences partially derive from broadband power
    variation (different electrode impedance, adipose tissue, muscle size).  The
    STRUCTURE OF INTER-CHANNEL INTERACTIONS IN FREQUENCY (CPSD matrix per band)
    may be more subject-invariant than broadband time-domain covariance (exp_63),
    because phase/coherence relationships are independent of absolute amplitude.

    Inspired by EEG/BCI literature where cross-spectral covariance matrices in
    narrowband (alpha/beta) outperform broadband covariance (Barachant 2013;
    Yger 2017), and imaginary coherence is immune to volume conduction / amplitude
    scaling.  In EMG: muscle coupling at characteristic frequencies may be more
    stable across subjects than raw amplitude covariations.

Method:
    1. Per window (N, T, C):
         For each frequency band b ∈ {B1, ..., B4}  (EMG-motivated bands, 4 bands):
           a. Apply Hann taper + FFT along time axis
           b. Select FFT bins in band b
           c. Form Hermitian CPSD: S_b = (1/F_b) * X_b^H X_b  — (C×C complex PSD)
           d. Convert to real SPD:
                "real_part"  → Re(S_b) + ε·I          shape (C, C)
                "block_real" → [[A,-B],[B,A]] + ε·I   shape (2C, 2C)
                              where S_b = A + iB

    2. fit() on X_train ONLY  ← LOSO critical:
         Per band: log-Euclidean Riemannian mean M_b from training windows only.
         Feature standardisation (μ, σ) from X_train statistics only.

    3. transform() (no refitting):
         Per band: tangent-space projection at M_b → upper-triangle vectorisation
         Concatenate across bands → (N, sum_b F_b)

    4. Classifier: SVM-RBF, SVM-linear (or LightGBM as optional drop-in).

Mathematics (LOSO safety):
    Re(S) is real PSD:
        For real vector v:  v^T Re(S) v = Re(v^H S v) = (1/F) ‖X_b v‖² ≥ 0  ✓
    Block-real is real PSD:
        [u;v]^T [[A,-B],[B,A]] [u;v] = Re((u+iv)^H (A+iB)(u+iv)) = Re(z^H H z) ≥ 0  ✓
    Both become strictly SPD after adding ε·I  ✓

Feature dimensions (C=8, 4 bands):
    real_part  → 36  per band  × 4 = 144 features
    block_real → 136 per band  × 4 = 544 features
    real_part + PowerfulFeatures ≈ 144 + ~253 = ~397 features

Variants tested:
    cpsd_real_4band_svm_rbf      — Re(CPSD), 4 bands, SVM-RBF
    cpsd_real_4band_svm_linear   — Re(CPSD), 4 bands, SVM-linear (EEG-Riemannian canon)
    cpsd_block_4band_svm_rbf     — block-real CPSD, 4 bands, SVM-RBF
    cpsd_real_combined_svm_rbf   — Re(CPSD) + PowerfulFeatureExtractor, SVM-RBF

LOSO compliance (critical, checked at every step):
    • CPSD Riemannian mean M_b fitted on X_train (train-subjects' train windows) ONLY.
    • Feature standardisation (μ, σ) from X_train statistics ONLY.
    • evaluate_numpy() uses fitted M_b — zero refitting on test-subject data.
    • No subject-specific adaptation at test time whatsoever.

Usage:
    python experiments/exp_78_cpsd_riemannian_spectral_loso.py          # CI (default)
    python experiments/exp_78_cpsd_riemannian_spectral_loso.py --ci     # same
    python experiments/exp_78_cpsd_riemannian_spectral_loso.py --full   # 20 subjects
    python experiments/exp_78_cpsd_riemannian_spectral_loso.py --subjects DB2_s1,DB2_s12
"""

import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sk_cm,
    f1_score,
)
from sklearn import svm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Subject lists
# ---------------------------------------------------------------------------

_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]

_FULL_SUBJECTS = [
    "DB2_s1",  "DB2_s2",  "DB2_s3",  "DB2_s4",  "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]


def parse_subjects_args(argv=None) -> List[str]:
    """Parse --subjects / --ci / --full CLI args.  Defaults to CI subjects."""
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject IDs")
    _parser.add_argument("--ci",   action="store_true",
                         help="Use CI test subset (5 subjects)")
    _parser.add_argument("--full", action="store_true",
                         help="Use full 20-subject set")
    _args, _ = _parser.parse_known_args(argv)

    if _args.subjects:
        return [s.strip() for s in _args.subjects.split(",")]
    if _args.full:
        return _FULL_SUBJECTS
    return _CI_SUBJECTS   # default: CI-safe subjects


# ===========================================================================
#  CPSDRiemannianExtractor
# ===========================================================================

class CPSDRiemannianExtractor:
    """
    Cross-Power Spectral Density Riemannian feature extractor.

    Pipeline (per window)
    ---------------------
    For each frequency band b:
      1. Hann taper + FFT along time axis.
      2. Select FFT bins in [f_low, f_high].
      3. CPSD matrix: S_b = (1/F_b) * X_b^H X_b   — (C×C) complex Hermitian PSD.
      4. Convert to real SPD:
           "real_part"  → Re(S_b) + ε·I,  shape (C, C)
           "block_real" → [[A,-B],[B,A]] + ε·I,  shape (2C, 2C)
      5. Tangent-space projection at M_b (Riemannian reference point).
      6. Upper-triangle vectorisation with √2 off-diagonal scaling.
    Concatenate features across all bands → (N, sum_b F_b).

    LOSO safety
    -----------
    Call fit(X_train) with training-split windows ONLY.
    transform() uses the fitted band means M_b — never refits.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    bands : list of (float, float)
        Frequency bands (f_low, f_high) in Hz.
    representation : "real_part" | "block_real"
        How to convert complex Hermitian CPSD to real SPD.
    regularization : float
        Ridge term ε added to each matrix diagonal.  Ensures strict SPD.
    mean_method : "log_euclidean" | "riemannian"
        Reference-point estimation method.
        "log_euclidean" — fast, closed-form approximation (default).
        "riemannian"    — true Fréchet mean via gradient descent (slower).
    spectral_method : "fft" | "welch"
        "fft"   — single Hann-windowed FFT per window (fast, default).
        "welch" — Welch PSD with 50% overlap, nperseg samples.
    nperseg : int | None
        Segment length for Welch; ignored for "fft".
        Defaults to min(T // 4, 256) at fit time.
    riemannian_max_iter : int
    riemannian_tol : float
    """

    def __init__(
        self,
        fs: int = 2000,
        bands: Optional[List[Tuple[float, float]]] = None,
        representation: str = "real_part",
        regularization: float = 1e-4,
        mean_method: str = "log_euclidean",
        spectral_method: str = "fft",
        nperseg: Optional[int] = None,
        riemannian_max_iter: int = 30,
        riemannian_tol: float = 1e-7,
    ):
        self.fs = fs
        self.bands: List[Tuple[float, float]] = bands or [
            (10.0,   50.0),
            (50.0,  150.0),
            (150.0, 350.0),
            (350.0, 1000.0),
        ]
        self.representation    = representation
        self.regularization    = regularization
        self.mean_method       = mean_method
        self.spectral_method   = spectral_method
        self.nperseg           = nperseg
        self.riemannian_max_iter = riemannian_max_iter
        self.riemannian_tol    = riemannian_tol

        # Set by fit()
        self._band_means:      List[np.ndarray] = []
        self._band_sqrts:      List[np.ndarray] = []
        self._band_invsqrts:   List[np.ndarray] = []
        self._fitted:          bool = False
        self._feature_dim:     Optional[int] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "CPSDRiemannianExtractor":
        """
        Estimate per-band Riemannian reference points from training windows.

        MUST be called with training-split windows only (LOSO requirement).
        X: (N, T, C) float32.
        """
        N, T, C = X.shape
        _nperseg = self.nperseg or min(T // 4, 256)

        self._band_means    = []
        self._band_sqrts    = []
        self._band_invsqrts = []
        total_feat_dim = 0

        for band in self.bands:
            covs = self._compute_spd_batch(X, band, T, C, _nperseg)  # (N, C', C')
            Cp = covs.shape[-1]

            if self.mean_method == "riemannian":
                M = self._riemannian_mean_iterative(covs)
            else:
                M = self._log_euclidean_mean(covs)

            sqrt_M, invsqrt_M = self._spd_sqrt_and_invsqrt(M)
            self._band_means.append(M)
            self._band_sqrts.append(sqrt_M)
            self._band_invsqrts.append(invsqrt_M)
            total_feat_dim += Cp * (Cp + 1) // 2

        self._feature_dim = total_feat_dim
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project windows to tangent-space feature vectors using fitted means.

        X: (N, T, C) → (N, F) float32
            F = n_bands * C'*(C'+1)/2
            where C' = C (real_part) or 2C (block_real)

        Uses fitted band means M_b — no refitting.  LOSO safe.
        """
        if not self._fitted:
            raise RuntimeError(
                "CPSDRiemannianExtractor.transform() called before fit(). "
                "Call fit(X_train) with training data only (LOSO requirement)."
            )
        N, T, C = X.shape
        _nperseg = self.nperseg or min(T // 4, 256)

        band_features = []
        for b_idx, band in enumerate(self.bands):
            covs = self._compute_spd_batch(X, band, T, C, _nperseg)  # (N, C', C')
            tang = self._tangent_project(covs, b_idx)                 # (N, C', C')
            feat = self._vectorize_upper(tang)                        # (N, F_b)
            band_features.append(feat)

        features = np.concatenate(band_features, axis=1).astype(np.float32)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # ── CPSD computation ────────────────────────────────────────────────────

    def _compute_spd_batch(
        self,
        X: np.ndarray,
        band: Tuple[float, float],
        T: int,
        C: int,
        nperseg: int,
    ) -> np.ndarray:
        """
        Compute real SPD matrices from CPSD for a given frequency band.

        X:      (N, T, C) float
        Returns (N, C', C') float64, where C' = C (real_part) or 2C (block_real).
        """
        f_low, f_high = band

        if self.spectral_method == "welch":
            S_complex = self._welch_cpsd(X, T, C, nperseg, f_low, f_high)
        else:
            S_complex = self._fft_cpsd(X, T, C, f_low, f_high)

        # S_complex: (N, C, C) Hermitian complex PSD
        return self._hermitian_to_spd(S_complex, C)  # (N, C', C')

    def _fft_cpsd(
        self,
        X: np.ndarray,
        T: int,
        C: int,
        f_low: float,
        f_high: float,
    ) -> np.ndarray:
        """
        Single Hann-windowed FFT-based CPSD (fully vectorised).

        X:      (N, T, C)
        Returns (N, C, C) complex Hermitian:
            S_nij = (1/F_b) sum_{f in band} conj(X_nfi) * X_nfj
                  = (1/F_b) * (X_b_n)^H  @  (X_b_n)
        """
        win = np.hanning(T).astype(np.float32)
        X_win = X * win[np.newaxis, :, np.newaxis]       # (N, T, C)
        X_fft = np.fft.rfft(X_win, axis=1)               # (N, T//2+1, C) complex
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)      # (T//2+1,)

        mask = (freqs >= f_low) & (freqs < f_high)
        n_bins = int(mask.sum())

        if n_bins == 0:
            return np.zeros((X.shape[0], C, C), dtype=np.complex128)

        X_band = X_fft[:, mask, :]                       # (N, F_b, C)
        # CPSD: S_n = (1/F_b) * X_band_n^H @ X_band_n
        S = np.einsum("nfi,nfj->nij", X_band.conj(), X_band)
        S /= n_bins
        return S   # (N, C, C) complex Hermitian

    def _welch_cpsd(
        self,
        X: np.ndarray,
        T: int,
        C: int,
        nperseg: int,
        f_low: float,
        f_high: float,
    ) -> np.ndarray:
        """
        Welch-based CPSD with 50% overlap (fully vectorised).

        X:      (N, T, C)
        Returns (N, C, C) complex Hermitian.
        """
        N = X.shape[0]
        step = max(nperseg // 2, 1)
        starts = list(range(0, T - nperseg + 1, step))
        n_segs = len(starts)

        if n_segs == 0:
            return self._fft_cpsd(X, T, C, f_low, f_high)

        # Build (N, n_segs, nperseg, C) without a Python loop for small n_segs
        segs = np.stack(
            [X[:, s : s + nperseg, :] for s in starts], axis=1
        )  # (N, n_segs, nperseg, C)

        win_seg = np.hanning(nperseg).astype(np.float32)
        segs_win = segs * win_seg[np.newaxis, np.newaxis, :, np.newaxis]
        segs_fft = np.fft.rfft(segs_win, axis=2)         # (N, n_segs, nperseg//2+1, C)

        freqs = np.fft.rfftfreq(nperseg, d=1.0 / self.fs)
        mask = (freqs >= f_low) & (freqs < f_high)
        n_bins = int(mask.sum())

        if n_bins == 0:
            return np.zeros((N, C, C), dtype=np.complex128)

        segs_band = segs_fft[:, :, mask, :]              # (N, n_segs, F_b, C)
        # Average CPSD over segments and frequency bins
        S = np.einsum("nsfi,nsfj->nij", segs_band.conj(), segs_band)
        S /= float(n_segs * n_bins)
        return S   # (N, C, C) complex Hermitian

    def _hermitian_to_spd(self, S: np.ndarray, C: int) -> np.ndarray:
        """
        Convert (N, C, C) complex Hermitian CPSD matrices to real SPD.

        "real_part":
            result = Re(S) + ε·I,  shape (N, C, C)
            Re(S) is real PSD because for real v:
                v^T Re(S) v = Re(v^H S v) = (1/F) ‖X_b v‖² ≥ 0

        "block_real":
            result = [[A, -B], [B, A]] + ε·I,  shape (N, 2C, 2C)
            where S = A + iB  (A symmetric, B skew-symmetric)
            PSD because [u;v]^T R [u;v] = Re((u+iv)^H S (u+iv)) ≥ 0
        """
        N = S.shape[0]
        A = S.real  # (N, C, C) symmetric

        if self.representation == "real_part":
            result = A + self.regularization * np.eye(C, dtype=np.float64)[np.newaxis]

        elif self.representation == "block_real":
            B = S.imag  # (N, C, C) skew-symmetric
            # Block matrix [[A, -B], [B, A]]:  (N, 2C, 2C)
            top    = np.concatenate([ A, -B], axis=-1)    # (N, C, 2C)
            bottom = np.concatenate([ B,  A], axis=-1)    # (N, C, 2C)
            result = np.concatenate([top, bottom], axis=-2)  # (N, 2C, 2C)
            result += self.regularization * np.eye(2 * C, dtype=np.float64)[np.newaxis]

        else:
            raise ValueError(
                f"Unknown representation '{self.representation}'. "
                "Choose 'real_part' or 'block_real'."
            )

        return result.astype(np.float64)

    # ── Riemannian geometry ─────────────────────────────────────────────────

    def _log_euclidean_mean(self, covs: np.ndarray) -> np.ndarray:
        """
        Log-Euclidean mean:  M = expm( mean_i logm(Σ_i) ).

        Closed-form approximation to the Riemannian Fréchet mean.
        Complexity: O(N·C'^3).  Reference: Arsigny et al. 2006.
        """
        log_covs = self._batch_logm(covs)    # (N, C', C')
        return self._single_expm(log_covs.mean(axis=0))

    def _riemannian_mean_iterative(self, covs: np.ndarray) -> np.ndarray:
        """
        True Riemannian (Fréchet) mean via gradient descent on SPD manifold.

        Algorithm (Moakher 2005; Bhatia 2007):
            M_0  ← log-Euclidean mean  (warm start)
            For t = 0, 1, ...:
                S_i  = logm( M_t^{-1/2} Σ_i M_t^{-1/2} )
                grad = mean_i(S_i)
                M_{t+1} = M_t^{1/2} expm(grad) M_t^{1/2}
            Stop when  ||grad||_F < tol
        """
        M = self._log_euclidean_mean(covs)   # warm start

        for _ in range(self.riemannian_max_iter):
            sqrt_M, invsqrt_M = self._spd_sqrt_and_invsqrt(M)
            S = invsqrt_M[np.newaxis] @ covs @ invsqrt_M[np.newaxis]
            L = self._batch_logm(S)
            grad = L.mean(axis=0)
            M = sqrt_M @ self._single_expm(grad) @ sqrt_M
            if np.linalg.norm(grad, "fro") < self.riemannian_tol:
                break

        return M

    def _tangent_project(self, covs: np.ndarray, band_idx: int) -> np.ndarray:
        """
        Map (N, C', C') SPD matrices to tangent space at M_b (fitted mean).

            T_i = M^{1/2}  logm( M^{-1/2} Σ_i M^{-1/2} )  M^{1/2}

        Uses pre-computed sqrt and invsqrt of M_b from fit().
        No re-estimation — LOSO safe.
        """
        invsq = self._band_invsqrts[band_idx]   # (C', C')
        sqrt  = self._band_sqrts[band_idx]       # (C', C')

        S = invsq[np.newaxis] @ covs @ invsq[np.newaxis]   # (N, C', C') whitened
        L = self._batch_logm(S)                             # (N, C', C') sym log
        T = sqrt[np.newaxis] @ L @ sqrt[np.newaxis]         # (N, C', C') tangent
        return T

    # ── Matrix operations (batched via eigendecomposition) ──────────────────

    @staticmethod
    def _batch_logm(Ms: np.ndarray) -> np.ndarray:
        """
        Batched matrix log for SPD matrices.

        Ms: (N, C, C) SPD → (N, C, C)
            logm(M) = V · diag(log λ) · V^T
        """
        eigvals, eigvecs = np.linalg.eigh(Ms)                   # (N,C), (N,C,C)
        log_eigvals = np.log(np.maximum(eigvals, 1e-12))        # (N, C)
        # logm_n = V_n diag(log λ_n) V_n^T
        return np.einsum("nij,nj,nkj->nik", eigvecs, log_eigvals, eigvecs)

    @staticmethod
    def _spd_sqrt_and_invsqrt(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute M^{1/2} and M^{-1/2} for a single (C, C) SPD matrix.
        """
        eigvals, eigvecs = np.linalg.eigh(M)                    # (C,), (C, C)
        eigvals_pos = np.maximum(eigvals, 1e-12)
        sqrt_M    = (eigvecs * np.sqrt(eigvals_pos))   @ eigvecs.T
        invsqrt_M = (eigvecs / np.sqrt(eigvals_pos))   @ eigvecs.T
        return sqrt_M, invsqrt_M

    @staticmethod
    def _single_expm(M: np.ndarray) -> np.ndarray:
        """
        Matrix exponent for a single symmetric (C, C) matrix.
            expm(M) = V · diag(exp λ) · V^T
        """
        eigvals, eigvecs = np.linalg.eigh(M)
        return (eigvecs * np.exp(eigvals)) @ eigvecs.T

    @staticmethod
    def _vectorize_upper(T_mats: np.ndarray) -> np.ndarray:
        """
        Vectorise (N, C, C) symmetric matrices: upper triangle with √2 scaling.

        T_mats: (N, C, C) → (N, C*(C+1)/2)

        Convention (Barachant et al. 2010):
            diagonal elements  × 1
            off-diagonal i < j × √2  (preserves Frobenius inner product)
        """
        N, C, _ = T_mats.shape
        idx_i, idx_j = np.triu_indices(C)
        scale = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0))   # (F_b,)
        return T_mats[:, idx_i, idx_j] * scale[np.newaxis, :]  # (N, F_b)


# ===========================================================================
#  CPSDRiemannianMLTrainer
#  Implements the full trainer interface expected by CrossSubjectExperiment:
#    fit(splits) → Dict
#    evaluate_numpy(X, y, split_name, visualize) → Dict
#    self.class_ids: List[int]
# ===========================================================================

class CPSDRiemannianMLTrainer:
    """
    ML trainer using per-band CPSD Riemannian tangent-space features.

    LOSO correctness
    ----------------
    • CPSDRiemannianExtractor.fit() is called with X_train ONLY.
    • Feature standardisation (μ, σ) from X_train statistics ONLY.
    • evaluate_numpy() uses the fitted extractor — no refitting on test data.

    Args
    ----
    extractor_cfg   : kwargs passed verbatim to CPSDRiemannianExtractor().
    combine_powerful: If True, concatenate PowerfulFeatureExtractor features
                      with the CPSD-Riemannian tangent vector before the SVM.
    train_cfg       : TrainingConfig (we use .seed and .ml_model_type).
    logger          : Python logger.
    output_dir      : Path for artefacts.
    visualizer      : Optional Visualizer (unused here, kept for API compat).
    """

    def __init__(
        self,
        extractor_cfg: Dict,
        combine_powerful: bool,
        train_cfg,
        logger,
        output_dir: Path,
        visualizer=None,
    ):
        self.extractor_cfg    = extractor_cfg
        self.combine_powerful = combine_powerful
        self.cfg              = train_cfg
        self.logger           = logger
        self.output_dir       = Path(output_dir)
        self.visualizer       = visualizer

        # Set during fit()
        self._extractor: Optional[CPSDRiemannianExtractor] = None
        self._powerful_extractor = None   # PowerfulFeatureExtractor, if combine_powerful
        self.ml_model            = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std:  Optional[np.ndarray] = None
        self.class_ids:    Optional[List[int]]  = None
        self.class_names:  Optional[Dict]       = None

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """
        Train the classifier on CPSD-Riemannian tangent-space features.

        splits["train"] ← used for ALL fitting (extractor + scaler + SVM)
        splits["val"]   ← transform only (validation monitoring)
        splits["test"]  ← NEVER used for fitting  ← test subject data

        LOSO compliance ensured at each step:
          (a) extractor.fit(X_train)  — mean M_b from train subjects only
          (b) feature_mean/std from F_train only
          (c) SVM fit on F_train only
        """
        from utils.logging import seed_everything
        seed_everything(self.cfg.seed)

        # 1. Unpack splits → flat (N, T, C) arrays ─────────────────────────
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        num_classes = len(class_ids)

        # 2. Fit CPSD extractor on X_train ONLY  ← LOSO critical ───────────
        self._extractor = CPSDRiemannianExtractor(**self.extractor_cfg)
        self.logger.info(
            f"[CPSDRiemannianMLTrainer] Fitting CPSD extractor on "
            f"X_train={X_train.shape}  (train data only — LOSO safe)"
        )
        self._extractor.fit(X_train)

        # Build PowerfulFeatureExtractor (stateless; init once, reuse in eval)
        if self.combine_powerful:
            from processing.powerful_features import PowerfulFeatureExtractor
            self._powerful_extractor = PowerfulFeatureExtractor(sampling_rate=2000)

        # 3. Extract features (transform only — extractor is already fitted) ─
        F_train = self._extract_features(X_train)   # (N_tr, F)
        F_val = (
            self._extract_features(X_val)
            if len(X_val) > 0
            else np.empty((0, F_train.shape[1]), dtype=np.float32)
        )
        F_test = (
            self._extract_features(X_test)
            if len(X_test) > 0
            else np.empty((0, F_train.shape[1]), dtype=np.float32)
        )
        self.logger.info(
            f"[CPSDRiemannianMLTrainer] Feature dim={F_train.shape[1]}  "
            f"(train={F_train.shape[0]}, val={F_val.shape[0]}, "
            f"test={F_test.shape[0]})"
        )

        # 4. Standardise — stats from X_train ONLY  ← LOSO critical ────────
        self.feature_mean = F_train.mean(axis=0).astype(np.float32)
        self.feature_std  = (F_train.std(axis=0) + 1e-8).astype(np.float32)

        F_train = self._standardize(F_train)
        F_val   = self._standardize(F_val)
        F_test  = self._standardize(F_test)

        # 5. Train classifier ────────────────────────────────────────────────
        ml_model_type = getattr(self.cfg, "ml_model_type", "svm_rbf")
        self.logger.info(
            f"[CPSDRiemannianMLTrainer] Training {ml_model_type} on "
            f"{F_train.shape[0]} samples, {num_classes} classes"
        )

        if ml_model_type == "svm_rbf":
            self.ml_model = svm.SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=self.cfg.seed,
            )
        elif ml_model_type == "svm_linear":
            self.ml_model = svm.SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=self.cfg.seed,
            )
        elif ml_model_type == "lgbm":
            try:
                from lightgbm import LGBMClassifier
                self.ml_model = LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    random_state=self.cfg.seed,
                    n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                self.logger.warning(
                    "[CPSDRiemannianMLTrainer] LightGBM not installed; "
                    "falling back to SVM-RBF"
                )
                self.ml_model = svm.SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=self.cfg.seed,
                )
        else:
            raise ValueError(
                f"[CPSDRiemannianMLTrainer] Unknown ml_model_type='{ml_model_type}'. "
                "Choose: svm_rbf, svm_linear, lgbm."
            )

        self.ml_model.fit(F_train, y_train)

        # 6. Store class metadata ────────────────────────────────────────────
        self.class_ids   = class_ids
        self.class_names = class_names

        # 7. Evaluate splits and build results dict ───────────────────────────
        results: Dict = {"class_ids": class_ids, "class_names": class_names}

        for split_name, F_sp, y_sp in [
            ("val",  F_val,  y_val),
            ("test", F_test, y_test),
        ]:
            if len(F_sp) == 0:
                results[split_name] = None
                continue
            y_pred = self.ml_model.predict(F_sp)
            acc    = float(accuracy_score(y_sp, y_pred))
            f1_mac = float(f1_score(y_sp, y_pred, average="macro", zero_division=0))
            cm     = sk_cm(y_sp, y_pred, labels=np.arange(num_classes))
            report = classification_report(
                y_sp, y_pred,
                target_names=[class_names.get(gid, str(gid)) for gid in class_ids],
                zero_division=0,
                output_dict=True,
            )
            self.logger.info(
                f"[CPSDRiemannianMLTrainer] {split_name}: "
                f"acc={acc:.4f}, f1={f1_mac:.4f}"
            )
            results[split_name] = {
                "accuracy":         acc,
                "f1_macro":         f1_mac,
                "report":           report,
                "confusion_matrix": cm.tolist(),
            }

        results_path = self.output_dir / "cpsd_riemannian_results.json"
        with open(results_path, "w") as fp:
            json.dump(_make_json_serializable(results), fp, indent=4,
                      ensure_ascii=False)

        return results

    # ── evaluate_numpy ─────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate on raw windows (N, T, C) using the fitted CPSD pipeline.

        Uses fitted Riemannian means M_b (from training data) for projection.
        No refitting of any kind.  LOSO safe.
        """
        assert self.ml_model is not None, \
            "Call fit() before evaluate_numpy()"
        assert self.feature_mean is not None and self.feature_std is not None, \
            "Feature stats missing — call fit() first"
        assert self.class_ids is not None, \
            "class_ids not set — call fit() first"

        F      = self._extract_features(X)
        F_std  = self._standardize(F)
        y_pred = self.ml_model.predict(F_std)

        acc    = float(accuracy_score(y, y_pred))
        f1_mac = float(f1_score(y, y_pred, average="macro", zero_division=0))
        cm     = sk_cm(y, y_pred)
        report = classification_report(y, y_pred, zero_division=0, output_dict=True)

        self.logger.info(
            f"[CPSDRiemannianMLTrainer] evaluate_numpy({split_name}): "
            f"acc={acc:.4f}, f1={f1_mac:.4f}"
        )
        return {
            "accuracy":         acc,
            "f1_macro":         f1_mac,
            "report":           report,
            "confusion_matrix": cm.tolist(),
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract CPSD-Riemannian [+ Powerful] features from X: (N,T,C) → (N,F).
        The extractor must already be fitted.
        PowerfulFeatureExtractor is stateless (no fitting needed).
        """
        F = self._extractor.transform(X)   # (N, F_cpsd)
        if self.combine_powerful and self._powerful_extractor is not None:
            P = self._powerful_extractor.transform(X)   # (N, F_pow)
            F = np.concatenate([F, P], axis=1)
        return F.astype(np.float32)

    def _standardize(self, F: np.ndarray) -> np.ndarray:
        """Apply stored (train-derived) mean/std standardisation."""
        if len(F) == 0:
            return F
        return ((F - self.feature_mean) / self.feature_std).astype(np.float32)

    def _prepare_splits_arrays(
        self,
        splits: Dict[str, Dict[int, np.ndarray]],
    ) -> Tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        List[int], Dict[int, str],
    ]:
        """
        Convert splits dict → flat (N, T, C) numpy arrays.

        Mirrors WindowClassifierTrainer._prepare_splits_arrays().

        splits["train"] = Dict[gesture_id → ndarray (N, T, C)] — train subjects
        splits["val"]   = Dict[gesture_id → ndarray (N, T, C)] — train subjects
        splits["test"]  = Dict[gesture_id → ndarray (N, T, C)] — TEST SUBJECT only

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test,
            class_ids (sorted gesture IDs),
            class_names {gid → str}
        """
        def _filter_nonempty(d: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
            return {
                gid: arr
                for gid, arr in d.items()
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0
            }

        train_d = _filter_nonempty(splits["train"])
        val_d   = _filter_nonempty(splits["val"])
        test_d  = _filter_nonempty(splits["test"])

        class_ids = sorted(train_d.keys())
        assert len(class_ids) >= 2, (
            f"Need at least 2 classes in training split, got {len(class_ids)}"
        )

        # Drop classes absent from training split
        val_d  = {gid: arr for gid, arr in val_d.items()  if gid in class_ids}
        test_d = {gid: arr for gid, arr in test_d.items() if gid in class_ids}

        def _concat_xy(dct: Dict[int, np.ndarray]):
            X_list, y_list = [], []
            for i, gid in enumerate(class_ids):
                if gid in dct:
                    X_list.append(dct[gid])
                    y_list.append(np.full(len(dct[gid]), i, dtype=np.int64))
            if not X_list:
                return (
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                )
            return (
                np.concatenate(X_list, axis=0).astype(np.float32),
                np.concatenate(y_list, axis=0),
            )

        X_train, y_train = _concat_xy(train_d)
        X_val,   y_val   = _concat_xy(val_d)
        X_test,  y_test  = _concat_xy(test_d)

        class_names = {
            gid: ("REST" if gid == 0 else f"Gesture {gid}")
            for gid in class_ids
        }
        self.logger.info(
            f"[CPSDRiemannianMLTrainer] Split sizes — "
            f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test, class_ids, class_names


# ===========================================================================
#  Utilities
# ===========================================================================

def _make_json_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ===========================================================================
#  LOSO fold runner
# ===========================================================================

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    variant_name: str,
    extractor_cfg: Dict,
    combine_powerful: bool,
    ml_model_type: str,
    proc_cfg,
    split_cfg,
    train_cfg,
) -> Dict:
    """Run one LOSO fold for the given CPSD-Riemannian variant."""
    import torch
    import gc
    from config.cross_subject import CrossSubjectConfig
    from data.multi_subject_loader import MultiSubjectLoader
    from evaluation.cross_subject import CrossSubjectExperiment
    from visualization.base import Visualizer
    from visualization.cross_subject import CrossSubjectVisualizer
    from utils.logging import setup_logging, seed_everything
    from utils.artifacts import ArtifactSaver

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = "ml_emg_td"
    train_cfg.ml_model_type = ml_model_type

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as fp:
        json.dump(asdict(split_cfg), fp, indent=4)

    cs_cfg = CrossSubjectConfig(
        train_subjects=train_subjects,
        test_subject=test_subject,
        exercises=exercises,
        base_dir=base_dir,
        pool_train_subjects=True,
        use_separate_val_subject=False,
        val_subject=None,
        val_ratio=0.15,
        seed=train_cfg.seed,
        max_gestures=10,
    )
    cs_cfg.save(output_dir / "cross_subject_config.json")

    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=False,          # numpy-only pipeline
        use_improved_processing=True,
    )

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    # Fresh trainer per fold — extractor is stateful (band Riemannian means)
    trainer = CPSDRiemannianMLTrainer(
        extractor_cfg=extractor_cfg,
        combine_powerful=combine_powerful,
        train_cfg=train_cfg,
        logger=logger,
        output_dir=output_dir,
        visualizer=base_viz,
    )

    experiment = CrossSubjectExperiment(
        cross_subject_config=cs_cfg,
        split_config=split_cfg,
        multi_subject_loader=multi_loader,
        trainer=trainer,
        visualizer=base_viz,
        logger=logger,
    )

    try:
        results = experiment.run()
    except Exception as e:
        print(f"Error in LOSO fold (test={test_subject}, variant={variant_name}): {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "variant":       variant_name,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1  = float(test_metrics.get("f1_macro",  0.0))

    print(
        f"[LOSO] test={test_subject} | variant={variant_name} | "
        f"Acc={test_acc:.4f}, F1={test_f1:.4f}"
    )

    results_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as fp:
        json.dump(_make_json_serializable(results_save), fp, indent=4,
                  ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    saver.save_metadata(
        _make_json_serializable({
            "test_subject":     test_subject,
            "train_subjects":   train_subjects,
            "variant":          variant_name,
            "extractor_cfg":    extractor_cfg,
            "combine_powerful": combine_powerful,
            "ml_model_type":    ml_model_type,
            "exercises":        exercises,
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
            },
        }),
        filename="fold_metadata.json",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del experiment, trainer, multi_loader, base_viz, cross_viz
    gc.collect()

    return {
        "test_subject":  test_subject,
        "variant":       variant_name,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ===========================================================================
#  Main
# ===========================================================================

def main():
    EXPERIMENT_NAME = "exp_78_cpsd_riemannian_spectral_loso"
    BASE_DIR        = ROOT / "data"
    ALL_SUBJECTS    = parse_subjects_args()
    OUTPUT_DIR      = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES       = ["E1"]

    # ── EMG frequency bands ─────────────────────────────────────────────────
    # fs = 2000 Hz → Nyquist = 1000 Hz
    # EMG power is concentrated in 10–500 Hz; we include a high-freq band too.
    # 4 bands chosen to match muscle physiology:
    #   B1  10– 50 Hz  : very-low / slow motor-unit recruitment oscillations
    #   B2  50–150 Hz  : primary slow motor units, main power region
    #   B3 150–350 Hz  : fast motor units, most discriminative EMG band
    #   B4 350–1000 Hz : high-frequency near-fibre potentials
    EMG_BANDS_4: List[Tuple[float, float]] = [
        (10.0,   50.0),
        (50.0,  150.0),
        (150.0, 350.0),
        (350.0, 1000.0),
    ]

    # ── Variant definitions ─────────────────────────────────────────────────
    #
    # Feature dimensions (C=8, 4 bands):
    #   real_part  → C*(C+1)/2 = 36 per band × 4 = 144
    #   block_real → 2C*(2C+1)/2 = 136 per band × 4 = 544
    #   real_part + Powerful ≈ 144 + ~253 = ~397
    #
    # extractor_cfg is passed verbatim to CPSDRiemannianExtractor(**extractor_cfg).
    #
    VARIANTS: List[Dict] = [
        {
            "name": "cpsd_real_4band_svm_rbf",
            # Re(CPSD) per band + log-Euclidean Riemannian tangent + SVM-RBF
            "extractor_cfg": {
                "fs":              2000,
                "bands":           EMG_BANDS_4,
                "representation":  "real_part",
                "regularization":  1e-4,
                "mean_method":     "log_euclidean",
                "spectral_method": "fft",
            },
            "combine_powerful": False,
            "ml_model_type":    "svm_rbf",
        },
        {
            "name": "cpsd_real_4band_svm_linear",
            # SVM-linear is the canonical classifier in EEG Riemannian BCI
            "extractor_cfg": {
                "fs":              2000,
                "bands":           EMG_BANDS_4,
                "representation":  "real_part",
                "regularization":  1e-4,
                "mean_method":     "log_euclidean",
                "spectral_method": "fft",
            },
            "combine_powerful": False,
            "ml_model_type":    "svm_linear",
        },
        {
            "name": "cpsd_block_4band_svm_rbf",
            # Full complex information via block-real SPD (2C×2C matrices)
            # Captures both co-spectral (Re) and quad-spectral (Im) structure
            "extractor_cfg": {
                "fs":              2000,
                "bands":           EMG_BANDS_4,
                "representation":  "block_real",
                "regularization":  1e-4,
                "mean_method":     "log_euclidean",
                "spectral_method": "fft",
            },
            "combine_powerful": False,
            "ml_model_type":    "svm_rbf",
        },
        {
            "name": "cpsd_real_combined_svm_rbf",
            # CPSD-Riemannian (spectral coupling) + PowerfulFeatureExtractor
            # (time-domain amplitude/nonlinearity) — complementary information
            "extractor_cfg": {
                "fs":              2000,
                "bands":           EMG_BANDS_4,
                "representation":  "real_part",
                "regularization":  1e-4,
                "mean_method":     "log_euclidean",
                "spectral_method": "fft",
            },
            "combine_powerful": True,
            "ml_model_type":    "svm_rbf",
        },
    ]

    from config.base import ProcessingConfig, SplitConfig, TrainingConfig
    from utils.logging import setup_logging

    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=2000,
        segment_edge_margin=0.1,
    )

    split_cfg = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="by_segments",
        shuffle_segments=True,
        seed=42,
        include_rest_in_splits=False,
    )

    train_cfg = TrainingConfig(
        batch_size=4096,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.0,
        early_stopping_patience=1,
        use_class_weights=False,
        seed=42,
        num_workers=0,
        device="cpu",
        use_handcrafted_features=False,
        handcrafted_feature_set="emg_td",
        pipeline_type="ml_emg_td",
        ml_model_type="svm_rbf",
        ml_use_hyperparam_search=False,
        ml_use_feature_selection=False,
        ml_use_pca=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUTPUT_DIR)

    print(f"[{EXPERIMENT_NAME}] Subjects  : {ALL_SUBJECTS}")
    print(f"[{EXPERIMENT_NAME}] Exercises : {EXERCISES}")
    print(f"[{EXPERIMENT_NAME}] Variants  :")
    for v in VARIANTS:
        rep    = v["extractor_cfg"]["representation"]
        n_bands = len(v["extractor_cfg"]["bands"])
        C = 8
        Cp = C if rep == "real_part" else 2 * C
        feat_dim = n_bands * Cp * (Cp + 1) // 2
        print(
            f"    {v['name']:42s}  "
            f"feat_dim={feat_dim:4d}  "
            f"{'+ Powerful' if v['combine_powerful'] else '':10s}  "
            f"clf={v['ml_model_type']}"
        )

    all_loso_results: List[Dict] = []

    for variant in VARIANTS:
        vname   = variant["name"]
        ext_cfg = variant["extractor_cfg"]
        combine = variant["combine_powerful"]
        ml_type = variant["ml_model_type"]

        print(f"\n{'=' * 70}")
        print(f"  Variant: {vname}")
        print(f"  repr={ext_cfg['representation']}, clf={ml_type}, "
              f"combine_powerful={combine}")
        print(f"  Bands: {ext_cfg['bands']}")
        print(f"  Starting LOSO ({len(ALL_SUBJECTS)} subjects)")
        print(f"{'=' * 70}")

        for test_subject in ALL_SUBJECTS:
            train_subjects  = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / vname / f"test_{test_subject}"

            train_cfg.ml_model_type = ml_type

            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                variant_name=vname,
                extractor_cfg=ext_cfg,
                combine_powerful=combine,
                ml_model_type=ml_type,
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
            )
            all_loso_results.append(fold_res)

    # ── Aggregate results ────────────────────────────────────────────────────
    aggregate_results: Dict = {}
    for variant in VARIANTS:
        vname = variant["name"]
        vr = [
            r for r in all_loso_results
            if r["variant"] == vname and r.get("test_accuracy") is not None
        ]
        if not vr:
            continue
        accs = [r["test_accuracy"] for r in vr]
        f1s  = [r["test_f1_macro"]  for r in vr]
        aggregate_results[vname] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_subjects":  len(accs),
            "per_subject":   vr,
        }

    # ── Summary print ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {EXPERIMENT_NAME}")
    print(f"{'=' * 70}")
    for vn, res in aggregate_results.items():
        acc_m = res["mean_accuracy"]
        acc_s = res["std_accuracy"]
        f1_m  = res["mean_f1_macro"]
        f1_s  = res["std_f1_macro"]
        n_sub = res["num_subjects"]
        print(
            f"  {vn:42s}  "
            f"Acc={acc_m:.4f}±{acc_s:.4f}  "
            f"F1={f1_m:.4f}±{f1_s:.4f}  "
            f"(N={n_sub})"
        )

    # ── Save summary JSON ─────────────────────────────────────────────────────
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "CPSD (Cross-Power Spectral Density) matrices in frequency bands, "
            "converted to real SPD and projected via log-Euclidean Riemannian "
            "tangent space, capture cross-subject-invariant muscle coupling "
            "structure.  The spectral inter-channel interaction pattern is more "
            "stable across subjects than broadband time-domain covariance "
            "(cf. exp_63), because phase/coherence is independent of absolute "
            "signal amplitude."
        ),
        "approach": (
            "Per-band CPSD (Hermitian) → real SPD → "
            "log-Euclidean Riemannian tangent features → SVM"
        ),
        "loso_compliance": (
            "CPSD Riemannian mean M_b fitted on X_train (train subjects' "
            "training windows) ONLY per band, per fold.  "
            "Feature standardisation (μ, σ) from X_train ONLY.  "
            "evaluate_numpy() uses fitted M_b — no test-subject data "
            "influences any fitted parameter."
        ),
        "variants":           VARIANTS,
        "subjects":           ALL_SUBJECTS,
        "exercises":          EXERCISES,
        "processing_config":  asdict(proc_cfg),
        "split_config":       asdict(split_cfg),
        "training_config":    asdict(train_cfg),
        "aggregate_results":  aggregate_results,
        "individual_results": all_loso_results,
        "experiment_date":    datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(_make_json_serializable(loso_summary), fp, indent=4,
                  ensure_ascii=False)
    print(f"\n[DONE] {EXPERIMENT_NAME} → {summary_path}")

    # ── Notify hypothesis executor (optional dependency) ─────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified
        best_variant = max(
            aggregate_results,
            key=lambda v: aggregate_results[v]["mean_f1_macro"],
            default=None,
        )
        if best_variant is not None:
            best_res = aggregate_results[best_variant]
            mark_hypothesis_verified(
                hypothesis_id="H78",
                metrics={
                    "best_variant":      best_variant,
                    "mean_accuracy":     best_res["mean_accuracy"],
                    "mean_f1_macro":     best_res["mean_f1_macro"],
                    "std_accuracy":      best_res["std_accuracy"],
                    "aggregate_results": aggregate_results,
                },
                experiment_name=EXPERIMENT_NAME,
            )
    except ImportError:
        pass
    except Exception as _he_err:
        print(f"[exp_78] hypothesis_executor notification failed: {_he_err}")


if __name__ == "__main__":
    main()
