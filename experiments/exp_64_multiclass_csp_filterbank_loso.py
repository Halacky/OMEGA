"""
Experiment 64: Multi-class CSP + Filter-Bank CSP for Cross-Subject EMG (LOSO)

Hypothesis H64:
    Common Spatial Pattern (CSP) filters learned from multi-subject training data
    produce log-variance features that are more cross-subject robust than classical
    power/amplitude features.  CSP has been the dominant feature extraction method
    in cross-subject EEG BCI for decades; combined with shrinkage-regularised
    covariance estimation and a frequency filter bank it may generalise well to EMG.

    Unlike Riemannian geometry (exp_63), CSP explicitly learns spatial filters that
    maximise the signal-to-noise ratio per gesture class.  The log-variance of CSP
    components is invariant to multiplicative gain shifts — a known source of
    inter-subject variability in surface EMG (electrode placement, skin impedance).

Method:
    One-vs-Rest (OvR) multi-class CSP:
        For each class c ∈ {0,...,K-1}:
            Σ_c    = mean regularised covariance of class-c windows (train only)
            Σ_rest = mean regularised covariance of all non-c windows (train only)
            Solve:  Σ_c · W = λ · (Σ_c + Σ_rest) · W  (generalised eigenproblem)
            Select top-p and bottom-p eigenvectors → W_c ∈ R^{C × 2p}
        Features: f = concat_c [ log(var_t(W_c^T · x_t)) ]  ∈ R^{K × 2p}

    Filter-Bank CSP (FBCSP):
        Apply B bandpass filters (Butterworth order-4) to the raw windows, then
        apply OvR-CSP per band independently, then concatenate all log-variance
        features.  EMG frequency bands used:
            [20–100 Hz], [100–300 Hz], [300–600 Hz], [600–900 Hz]

    Covariance regularisation choices:
        "fixed"     — Σ_reg = Σ + ε·I  (Tikhonov, ε = 1e-4)
        "shrinkage" — Oracle Approximating Shrinkage (OAS, vectorised closed-form)
                      followed by a small fixed ε·I for numerical stability.

Variants tested:
    ovr_csp_svm_rbf     — OvR CSP (p=4, fixed reg)    + SVM-RBF
    ovr_csp_svm_linear  — OvR CSP (p=4, fixed reg)    + SVM-linear
    ovr_csp_shrink_svm  — OvR CSP (p=4, OAS shrinkage) + SVM-RBF
    fbcsp_svm_rbf       — FBCSP  (4 bands, p=4, fixed) + SVM-RBF

LOSO compliance (critical — no data leakage):
    • CSP spatial filters W_c computed from X_train (train-subjects' training
      windows) ONLY.  Covariances Σ_c, Σ_rest derived exclusively from X_train
      and y_train.
    • Feature standardisation (μ, σ) from X_train statistics ONLY.
    • transform() applies the FITTED filters — no re-estimation on val/test data.
    • No subject-specific adaptation, calibration, or fine-tuning at test time.

Usage:
    python experiments/exp_64_multiclass_csp_filterbank_loso.py          # CI (default)
    python experiments/exp_64_multiclass_csp_filterbank_loso.py --ci     # same
    python experiments/exp_64_multiclass_csp_filterbank_loso.py --full   # 20 subjects
    python experiments/exp_64_multiclass_csp_filterbank_loso.py --subjects DB2_s1,DB2_s12
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
    return _CI_SUBJECTS   # server-safe default: only CI symlinks guaranteed


# ===========================================================================
#  MulticlassCSPExtractor
# ===========================================================================

class MulticlassCSPExtractor:
    """
    One-vs-Rest multiclass CSP for EMG.

    For K classes, fits K sets of spatial filters W_c by solving the
    generalised eigenvalue problem:

        Σ_c · W = λ · (Σ_c + Σ_rest) · W

    where Σ_c = mean regularised covariance of class-c windows (train only),
    Σ_rest = mean regularised covariance of all other-class windows (train only).

    The top-p eigenvectors (largest λ) maximise the variance ratio for class c;
    the bottom-p eigenvectors (smallest λ) are most discriminative for the rest.
    Both are kept, giving W_c ∈ R^{C × 2p} per class.

    Features:
        f = [ log(var_t(W_c^T · x_t)) ]_{c=0..K-1}  ∈ R^{K · 2p}
        Log-variance is invariant to multiplicative gain shifts.

    LOSO correctness:
        fit(X_train, y_train) — called with TRAINING data only.
        transform(X)          — uses fitted filters; no re-estimation.

    Args
    ----
    n_components  : p — number of top/bottom eigenvectors per class (default 4).
    regularization: ε — Tikhonov ridge added to each covariance (default 1e-4).
    cov_estimator : "fixed" (Tikhonov only) or "shrinkage" (OAS + Tikhonov).
    """

    def __init__(
        self,
        n_components: int = 4,
        regularization: float = 1e-4,
        cov_estimator: str = "fixed",
    ):
        self.n_components  = n_components
        self.regularization = regularization
        self.cov_estimator  = cov_estimator

        # Set by fit()
        self.filters_: List[np.ndarray] = []   # K items, each (C, 2p)
        self.classes_: Optional[np.ndarray] = None
        self._p: int = 0
        self._fitted: bool = False

    # ── public API ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassCSPExtractor":
        """
        Fit OvR CSP filters from TRAINING windows.

        X: (N, T, C) float32  — training windows ONLY
        y: (N,)      int64    — 0-indexed class labels (from train split)
        """
        from scipy.linalg import eigh

        classes = np.unique(y)
        C = X.shape[2]
        p = max(1, min(self.n_components, C // 2))

        # (N, C, C) regularised covariances — TRAIN DATA ONLY
        covs = self._compute_covs(X)

        self.filters_ = []
        self.classes_ = classes
        self._p = p

        for c in classes:
            mask_c    = (y == c)
            mask_rest = ~mask_c

            Σ_c    = self._class_mean_cov(covs, mask_c)     # (C, C)
            Σ_rest = self._class_mean_cov(covs, mask_rest)  # (C, C)
            Σ_total = Σ_c + Σ_rest                           # (C, C)

            try:
                # eigh solves the symmetric generalised eigenproblem.
                # eigvecs[:,i] is the i-th eigenvector; eigenvalues ascending.
                eigvals, eigvecs = eigh(Σ_c, Σ_total)
                # bottom-p: smallest λ → most "rest-like" → discriminative for rest
                # top-p:    largest λ  → most "class-c-like"
                W_c = np.concatenate(
                    [eigvecs[:, :p], eigvecs[:, -p:]],
                    axis=1,
                )  # (C, 2p)
            except Exception:
                # Fallback: identity projection (safe but non-informative)
                W_c = np.eye(C, 2 * p, dtype=np.float64)

            self.filters_.append(W_c.astype(np.float64))

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted CSP filters and compute log-variance features.

        X: (N, T, C) → (N, K * 2p) float32.
        Uses fitted filters only — no re-estimation.
        """
        if not self._fitted:
            raise RuntimeError(
                "MulticlassCSPExtractor.transform() called before fit().  "
                "Call fit(X_train, y_train) with training data only."
            )
        feats = []
        for W_c in self.filters_:
            # Spatial filter: (N, T, C) @ (C, 2p) → (N, T, 2p)
            X_filt = X @ W_c
            # Log-variance along time axis: (N, 2p)
            var    = X_filt.var(axis=1)
            log_var = np.log(np.maximum(var, 1e-12))
            feats.append(log_var)
        return np.concatenate(feats, axis=1).astype(np.float32)

    # ── covariance computation ───────────────────────────────────────────────

    def _compute_covs(self, X: np.ndarray) -> np.ndarray:
        """
        Compute regularised sample covariance per window.

        X: (N, T, C) → (N, C, C)

        Steps:
          1. Demean along time axis
          2. Σ_i = X_c_i^T X_c_i / (T-1)
          3. (Optional) OAS shrinkage toward scaled identity
          4. Σ_reg = Σ + ε·I  (always applied for numerical stability)
        """
        N, T, C = X.shape
        X_c  = X - X.mean(axis=1, keepdims=True)            # (N, T, C) demeaned
        covs = np.einsum("nti,ntj->nij", X_c, X_c) / max(T - 1, 1)  # (N, C, C)

        if self.cov_estimator == "shrinkage":
            # ── OAS shrinkage (vectorised, Chen et al. 2010) ─────────────────
            # Shrinks each sample covariance toward (tr(Σ)/C) · I.
            # Uses the window length T as the effective sample count.
            tr_S  = np.trace(covs, axis1=1, axis2=2)           # (N,)
            # For symmetric matrices: tr(Σ²) = ||Σ||_F²
            tr_S2 = (covs ** 2).sum(axis=(1, 2))               # (N,)
            mu    = tr_S / C                                    # (N,) target scale

            num   = ((T - 2) / max(T, 1)) * tr_S2 + tr_S ** 2
            denom = (T + 1 - 2.0 / C) * (tr_S2 - tr_S ** 2 / C)
            alpha = np.where(np.abs(denom) < 1e-12, 1.0, num / denom)
            alpha = np.clip(alpha, 0.0, 1.0)                   # (N,)

            I = np.eye(C, dtype=covs.dtype)[np.newaxis]        # (1, C, C)
            a = alpha[:, np.newaxis, np.newaxis]                # (N, 1, 1)
            m = mu[:, np.newaxis, np.newaxis]                   # (N, 1, 1)
            covs = (1 - a) * covs + a * m * I

        # Tikhonov regularisation: always applied (numerical safety)
        covs += self.regularization * np.eye(C, dtype=covs.dtype)[np.newaxis]
        return covs

    def _class_mean_cov(
        self, covs: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Mean covariance matrix for the masked subset of windows."""
        if mask.sum() == 0:
            C = covs.shape[1]
            return np.eye(C, dtype=np.float64)
        return covs[mask].mean(axis=0).astype(np.float64)


# ===========================================================================
#  FilterBankCSPExtractor
# ===========================================================================

class FilterBankCSPExtractor:
    """
    Filter-Bank CSP: apply B bandpass filters independently, fit OvR-CSP on
    each filtered band (training data only), and concatenate log-variance
    features from all bands at transform time.

    Default EMG frequency bands (Hz):
        [20–100], [100–300], [300–600], [600–900]
    (Nyquist = 1000 Hz at 2000 Hz sampling rate — all bands are valid.)

    Feature dimension = B × K × 2p  (B bands, K classes, 2p filters/class).

    LOSO correctness:
        fit(X_train, y_train) — bandpass filters are deterministic (scipy Butter-
          worth), no fitting from data.  CSP filters inside each band extractor are
          fitted from X_train only.
        transform(X) — applies the same deterministic bandpass and the FITTED CSP
          filters.  No re-estimation on test data.

    Args
    ----
    freq_bands     : List of (low_hz, high_hz) tuples.  Default: 4 EMG bands.
    n_components   : p per band (top + bottom eigenvectors per class).
    regularization : Tikhonov ε for covariance SPD regularisation.
    cov_estimator  : "fixed" or "shrinkage" — passed to MulticlassCSPExtractor.
    sampling_rate  : Data sampling rate in Hz (default 2000).
    """

    DEFAULT_FREQ_BANDS: List[Tuple[float, float]] = [
        (20,  100),
        (100, 300),
        (300, 600),
        (600, 900),
    ]

    def __init__(
        self,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        n_components: int = 4,
        regularization: float = 1e-4,
        cov_estimator: str = "fixed",
        sampling_rate: int = 2000,
    ):
        self.freq_bands    = freq_bands or self.DEFAULT_FREQ_BANDS
        self.n_components  = n_components
        self.regularization = regularization
        self.cov_estimator  = cov_estimator
        self.sampling_rate  = sampling_rate

        # Set by fit()
        self.band_extractors_: List[MulticlassCSPExtractor] = []
        self._fitted: bool = False

    # ── public API ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FilterBankCSPExtractor":
        """
        Fit one MulticlassCSPExtractor per frequency band.

        X: (N, T, C) — TRAINING DATA ONLY
        y: (N,)      — 0-indexed class labels
        """
        self.band_extractors_ = []
        for low_hz, high_hz in self.freq_bands:
            X_filt = self._bandpass(X, low_hz, high_hz)
            ext = MulticlassCSPExtractor(
                n_components=self.n_components,
                regularization=self.regularization,
                cov_estimator=self.cov_estimator,
            )
            ext.fit(X_filt, y)
            self.band_extractors_.append(ext)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted FBCSP pipeline.

        X: (N, T, C) → (N, B × K × 2p) float32.
        Uses fitted filters only.
        """
        if not self._fitted:
            raise RuntimeError(
                "FilterBankCSPExtractor.transform() called before fit()."
            )
        feats = []
        for (low_hz, high_hz), ext in zip(self.freq_bands, self.band_extractors_):
            X_filt = self._bandpass(X, low_hz, high_hz)
            feats.append(ext.transform(X_filt))
        return np.concatenate(feats, axis=1).astype(np.float32)

    # ── bandpass filtering ───────────────────────────────────────────────────

    def _bandpass(self, X: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        """
        Butterworth order-4 bandpass filter along the time axis (axis=1).

        scipy.signal.sosfilt supports N-D arrays with explicit axis parameter,
        so no looping over windows or channels is needed.

        X: (N, T, C) → (N, T, C) float32 (same shape, filtered).
        """
        from scipy.signal import butter, sosfilt

        nyq  = self.sampling_rate / 2.0
        low  = max(low_hz  / nyq, 1e-6)
        high = min(high_hz / nyq, 0.9999)
        sos  = butter(4, [low, high], btype="bandpass", output="sos")
        # axis=1 → filter along time dimension; (N, T, C) ND array supported
        return sosfilt(sos, X, axis=1).astype(np.float32)


# ===========================================================================
#  CSPMLTrainer
#  Full trainer interface required by CrossSubjectExperiment:
#    fit(splits)                           → Dict
#    evaluate_numpy(X, y, name, visualize) → Dict
#    self.class_ids: List[int]
# ===========================================================================

class CSPMLTrainer:
    """
    ML trainer using Multi-class CSP (OvR) or FilterBank CSP features + SVM.

    LOSO correctness
    ----------------
    • CSP filters fitted on X_train (train subjects' training windows) ONLY.
    • Feature standardisation (μ, σ) derived from X_train ONLY.
    • evaluate_numpy() uses the fitted extractor — no re-fitting on test data.
    • No subject-specific adaptation at test time.

    Args
    ----
    use_filterbank : If True, use FilterBankCSPExtractor; else MulticlassCSPExtractor.
    n_components   : Number of top/bottom CSP eigenvectors per class (p).
    regularization : Tikhonov ε for covariance regularisation.
    cov_estimator  : "fixed" or "shrinkage".
    freq_bands     : For FBCSP only — list of (low_hz, high_hz) tuples.
    sampling_rate  : Data sampling rate in Hz.
    train_cfg      : TrainingConfig (used for .seed and .ml_model_type).
    logger         : Python logger.
    output_dir     : Path for saving artefacts.
    visualizer     : Optional Visualizer (kept for API compat, not used here).
    """

    def __init__(
        self,
        use_filterbank: bool,
        n_components: int,
        regularization: float,
        cov_estimator: str,
        freq_bands: Optional[List[Tuple[float, float]]],
        sampling_rate: int,
        train_cfg,
        logger,
        output_dir: Path,
        visualizer=None,
    ):
        self.use_filterbank  = use_filterbank
        self.n_components    = n_components
        self.regularization  = regularization
        self.cov_estimator   = cov_estimator
        self.freq_bands      = freq_bands
        self.sampling_rate   = sampling_rate
        self.cfg             = train_cfg
        self.logger          = logger
        self.output_dir      = Path(output_dir)
        self.visualizer      = visualizer

        # Set during fit()
        self._extractor                       = None
        self.ml_model                         = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std:  Optional[np.ndarray] = None
        self.class_ids:    Optional[List[int]]   = None
        self.class_names:  Optional[Dict]         = None

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """
        Fit CSP extractor + SVM on training data.

        splits["train"] — train-subjects training windows   ← CSP fitted here ONLY
        splits["val"]   — train-subjects validation windows
        splits["test"]  — TEST SUBJECT windows              ← NEVER used for fitting

        CSP spatial filters W_c and feature statistics (μ, σ) are derived
        exclusively from splits["train"].
        """
        from utils.logging import seed_everything
        seed_everything(self.cfg.seed)

        # ── 1. Unpack splits → flat (N, T, C) arrays ─────────────────────
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        num_classes = len(class_ids)

        # ── 2. Fit extractor on X_train ONLY ──────────────────────────── ← key
        if self.use_filterbank:
            self._extractor = FilterBankCSPExtractor(
                freq_bands=self.freq_bands,
                n_components=self.n_components,
                regularization=self.regularization,
                cov_estimator=self.cov_estimator,
                sampling_rate=self.sampling_rate,
            )
            extractor_name = "FilterBankCSP"
        else:
            self._extractor = MulticlassCSPExtractor(
                n_components=self.n_components,
                regularization=self.regularization,
                cov_estimator=self.cov_estimator,
            )
            extractor_name = "MulticlassCSP"

        self.logger.info(
            f"[CSPMLTrainer] Fitting {extractor_name} on "
            f"X_train={X_train.shape}  "
            f"(n_classes={num_classes}, cov={self.cov_estimator})  "
            f"← train data only — LOSO safe"
        )
        self._extractor.fit(X_train, y_train)  # ← ONLY X_train used

        # ── 3. Extract features (transform only, no re-fitting) ───────────
        F_train = self._extract_features(X_train)           # (N_train, F)
        F_val   = (
            self._extract_features(X_val)
            if len(X_val) > 0
            else np.empty((0, F_train.shape[1]), dtype=np.float32)
        )
        F_test  = (
            self._extract_features(X_test)
            if len(X_test) > 0
            else np.empty((0, F_train.shape[1]), dtype=np.float32)
        )
        self.logger.info(
            f"[CSPMLTrainer] Feature dim={F_train.shape[1]}  "
            f"(train={F_train.shape[0]}, val={F_val.shape[0]}, "
            f"test={F_test.shape[0]})"
        )

        # ── 4. Feature standardisation — stats from X_train ONLY ─────── ← key
        self.feature_mean = F_train.mean(axis=0).astype(np.float32)
        self.feature_std  = (F_train.std(axis=0) + 1e-8).astype(np.float32)

        F_train = self._standardize(F_train)
        F_val   = self._standardize(F_val)
        F_test  = self._standardize(F_test)

        # ── 5. Train SVM ─────────────────────────────────────────────────
        ml_model_type = getattr(self.cfg, "ml_model_type", "svm_rbf")
        self.logger.info(
            f"[CSPMLTrainer] Training SVM ({ml_model_type}) "
            f"on {F_train.shape[0]} samples, {num_classes} classes"
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
        else:
            raise ValueError(
                f"[CSPMLTrainer] Unknown ml_model_type='{ml_model_type}'. "
                f"Supported: svm_rbf, svm_linear."
            )
        self.ml_model.fit(F_train, y_train)

        # ── 6. Store class metadata ───────────────────────────────────────
        self.class_ids   = class_ids
        self.class_names = class_names

        # ── 7. Evaluate splits and build results dict ────────────────────
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
                f"[CSPMLTrainer] {split_name}: acc={acc:.4f}, f1={f1_mac:.4f}"
            )
            results[split_name] = {
                "accuracy": acc,
                "f1_macro": f1_mac,
                "report":   report,
                "confusion_matrix": cm.tolist(),
            }

        results_path = self.output_dir / "csp_results.json"
        with open(results_path, "w") as fp:
            json.dump(_make_json_serializable(results), fp, indent=4,
                      ensure_ascii=False)

        return results

    # ── evaluate_numpy ───────────────────────────────────────────────────────

    def evaluate_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "custom",
        visualize: bool = False,
    ) -> Dict:
        """
        Evaluate on raw windows (N, T, C) using the fitted CSP pipeline.

        The fitted CSP spatial filters and feature statistics (from training data)
        are used — no re-fitting of any kind.  Safe for test-subject evaluation.
        """
        assert self.ml_model is not None, "Call fit() before evaluate_numpy()"
        assert self.feature_mean is not None and self.feature_std is not None, \
            "Feature statistics missing — call fit() first"
        assert self.class_ids is not None, "class_ids not set — call fit() first"

        F     = self._extract_features(X)
        F_std = self._standardize(F)
        y_pred = self.ml_model.predict(F_std)

        acc    = float(accuracy_score(y, y_pred))
        f1_mac = float(f1_score(y, y_pred, average="macro", zero_division=0))
        cm     = sk_cm(y, y_pred)
        report = classification_report(y, y_pred, zero_division=0, output_dict=True)

        return {
            "accuracy": acc,
            "f1_macro": f1_mac,
            "report":   report,
            "confusion_matrix": cm.tolist(),
        }

    # ── internal helpers ─────────────────────────────────────────────────────

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted CSP extractor to raw windows.

        X: (N, T, C) → (N, F) float32.
        The extractor must already be fitted (no re-estimation here).
        """
        return self._extractor.transform(X).astype(np.float32)

    def _standardize(self, F: np.ndarray) -> np.ndarray:
        """Apply train-derived (μ, σ) standardisation.  Empty arrays pass through."""
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

        splits["train"] = Dict[gesture_id, ndarray (N, T, C)] — train subjects
        splits["val"]   = Dict[gesture_id, ndarray (N, T, C)] — train subjects
        splits["test"]  = Dict[gesture_id, ndarray (N, T, C)] — TEST SUBJECT

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test,
            class_ids (sorted gesture IDs), class_names {gid → str}.
        """
        def _filter_nonempty(d: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
            return {
                gid: arr for gid, arr in d.items()
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0
            }

        train_d = _filter_nonempty(splits["train"])
        val_d   = _filter_nonempty(splits["val"])
        test_d  = _filter_nonempty(splits["test"])

        class_ids = sorted(train_d.keys())
        assert len(class_ids) >= 2, (
            f"Need at least 2 classes in training split, got {len(class_ids)}"
        )

        # Exclude classes not seen in training (avoid leakage of class info)
        val_d  = {gid: arr for gid, arr in val_d.items()  if gid in class_ids}
        test_d = {gid: arr for gid, arr in test_d.items() if gid in class_ids}

        def _concat_xy(dct: Dict[int, np.ndarray]):
            X_list, y_list = [], []
            for i, gid in enumerate(class_ids):
                if gid in dct:
                    X_list.append(dct[gid])
                    y_list.append(np.full(len(dct[gid]), i, dtype=np.int64))
            if not X_list:
                return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
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
            f"[CSPMLTrainer] Split sizes — "
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
    use_filterbank: bool,
    n_components: int,
    regularization: float,
    cov_estimator: str,
    freq_bands: Optional[List[Tuple[float, float]]],
    sampling_rate: int,
    ml_model_type: str,
    proc_cfg,
    split_cfg,
    train_cfg,
) -> Dict:
    """Run one LOSO fold for the given CSP variant."""
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

    train_cfg.pipeline_type  = "ml_emg_td"
    train_cfg.ml_model_type  = ml_model_type

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
        use_gpu=False,       # numpy-only pipeline
        use_improved_processing=True,
    )

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    # Fresh trainer per fold — CSP extractor is stateful (fitted filters)
    trainer = CSPMLTrainer(
        use_filterbank=use_filterbank,
        n_components=n_components,
        regularization=regularization,
        cov_estimator=cov_estimator,
        freq_bands=freq_bands,
        sampling_rate=sampling_rate,
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
        print(f"Error in fold (test={test_subject}, variant={variant_name}): {e}")
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
            "test_subject":   test_subject,
            "train_subjects": train_subjects,
            "variant":        variant_name,
            "use_filterbank": use_filterbank,
            "n_components":   n_components,
            "regularization": regularization,
            "cov_estimator":  cov_estimator,
            "freq_bands":     freq_bands,
            "ml_model_type":  ml_model_type,
            "exercises":      exercises,
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
    EXPERIMENT_NAME = "exp_64_multiclass_csp_filterbank_loso"
    BASE_DIR        = ROOT / "data"
    ALL_SUBJECTS    = parse_subjects_args()
    OUTPUT_DIR      = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES       = ["E1"]
    SAMPLING_RATE   = 2000

    # ── Variant definitions ───────────────────────────────────────────────
    #
    # Feature dimensions for C=8 channels, K=10 classes, p=4:
    #   OvR CSP:  K * 2p = 10 * 8 = 80 features
    #   FBCSP:    B * K * 2p = 4 * 10 * 8 = 320 features  (4 bands)
    #
    VARIANTS: List[Dict] = [
        {
            "name":           "ovr_csp_svm_rbf",
            "use_filterbank": False,
            "n_components":   4,
            "regularization": 1e-4,
            "cov_estimator":  "fixed",
            "freq_bands":     None,
            "ml_model_type":  "svm_rbf",
            # Fixed Tikhonov reg, OvR CSP + SVM-RBF — baseline CSP variant
        },
        {
            "name":           "ovr_csp_svm_linear",
            "use_filterbank": False,
            "n_components":   4,
            "regularization": 1e-4,
            "cov_estimator":  "fixed",
            "freq_bands":     None,
            "ml_model_type":  "svm_linear",
            # SVM-linear: canonical in EEG CSP BCI literature
        },
        {
            "name":           "ovr_csp_shrink_svm",
            "use_filterbank": False,
            "n_components":   4,
            "regularization": 1e-4,
            "cov_estimator":  "shrinkage",   # OAS closed-form shrinkage
            "freq_bands":     None,
            "ml_model_type":  "svm_rbf",
            # Shrinkage regularisation: more robust with limited training data
        },
        {
            "name":           "fbcsp_svm_rbf",
            "use_filterbank": True,
            "n_components":   4,
            "regularization": 1e-4,
            "cov_estimator":  "fixed",
            # 4 EMG frequency bands: [20-100], [100-300], [300-600], [600-900] Hz
            "freq_bands":     [(20, 100), (100, 300), (300, 600), (600, 900)],
            "ml_model_type":  "svm_rbf",
            # FBCSP: captures frequency-specific spatial patterns
        },
    ]

    from config.base import ProcessingConfig, SplitConfig, TrainingConfig
    from utils.logging import setup_logging

    proc_cfg = ProcessingConfig(
        window_size=600,
        window_overlap=300,
        num_channels=8,
        sampling_rate=SAMPLING_RATE,
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
        K, p = 10, v["n_components"]
        if v["use_filterbank"]:
            B = len(v["freq_bands"])
            feat_note = f"B={B} bands × K={K} × 2p={2*p} = {B*K*2*p} features"
        else:
            feat_note = f"K={K} × 2p={2*p} = {K*2*p} features"
        print(
            f"    {v['name']:28s}  {feat_note:38s}  "
            f"cov={v['cov_estimator']:10s}  clf={v['ml_model_type']}"
        )

    all_loso_results: List[Dict] = []

    for variant in VARIANTS:
        vname    = variant["name"]
        print(f"\n{'=' * 68}")
        print(f"  Variant: {vname}  —  starting LOSO ({len(ALL_SUBJECTS)} subjects)")
        print(f"  filterbank={variant['use_filterbank']}, "
              f"cov={variant['cov_estimator']}, "
              f"clf={variant['ml_model_type']}, "
              f"p={variant['n_components']}")
        print(f"{'=' * 68}")

        for test_subject in ALL_SUBJECTS:
            train_subjects  = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / vname / f"test_{test_subject}"

            # Set ml_model_type for this variant in train_cfg
            train_cfg.ml_model_type = variant["ml_model_type"]

            fold_res = run_single_loso_fold(
                base_dir=BASE_DIR,
                output_dir=fold_output_dir,
                train_subjects=train_subjects,
                test_subject=test_subject,
                exercises=EXERCISES,
                variant_name=vname,
                use_filterbank=variant["use_filterbank"],
                n_components=variant["n_components"],
                regularization=variant["regularization"],
                cov_estimator=variant["cov_estimator"],
                freq_bands=variant["freq_bands"],
                sampling_rate=SAMPLING_RATE,
                ml_model_type=variant["ml_model_type"],
                proc_cfg=proc_cfg,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
            )
            all_loso_results.append(fold_res)

    # ── Aggregate per variant ─────────────────────────────────────────────
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
            "mean_accuracy":  float(np.mean(accs)),
            "std_accuracy":   float(np.std(accs)),
            "mean_f1_macro":  float(np.mean(f1s)),
            "std_f1_macro":   float(np.std(f1s)),
            "num_subjects":   len(accs),
            "per_subject":    vr,
        }

    # ── Summary print ─────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print(f"SUMMARY: {EXPERIMENT_NAME}")
    print(f"{'=' * 68}")
    for vn, res in aggregate_results.items():
        acc_m = res["mean_accuracy"]
        acc_s = res["std_accuracy"]
        f1_m  = res["mean_f1_macro"]
        f1_s  = res["std_f1_macro"]
        n_sub = res["num_subjects"]
        print(
            f"  {vn:28s}  "
            f"Acc={acc_m:.4f}±{acc_s:.4f}  "
            f"F1={f1_m:.4f}±{f1_s:.4f}  "
            f"(N={n_sub})"
        )

    # ── Save summary JSON ─────────────────────────────────────────────────
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "CSP spatial filters learned from multi-subject training data produce "
            "log-variance features that are cross-subject robust.  "
            "Log-variance of CSP components is invariant to multiplicative gain shifts "
            "(electrode placement, skin impedance variation) — a key source of "
            "inter-subject EMG variability.  OAS shrinkage and a frequency filter bank "
            "further improve generalisation."
        ),
        "approach": (
            "One-vs-Rest Multiclass CSP (+ optional FilterBank) + SVM. "
            "CSP filters fitted on train subjects only (LOSO safe)."
        ),
        "loso_compliance": (
            "CSP spatial filters W_c fitted on X_train (train-subjects' training windows) ONLY. "
            "Feature standardisation (mu, sigma) from X_train ONLY. "
            "transform() applies fitted filters — no re-estimation on test subject data. "
            "No subject-specific adaptation at test time."
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

    # ── Notify hypothesis executor (optional dependency) ──────────────────
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
                hypothesis_id="H64",
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
        print(f"[exp_64] hypothesis_executor notification failed: {_he_err}")


if __name__ == "__main__":
    main()
