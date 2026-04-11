"""
Experiment 63: Riemannian Geometry Features (SPD Covariance Manifold) for Cross-Subject EMG (LOSO)

Hypothesis H63:
    Covariance-matrix features projected onto the Riemannian tangent space are more
    subject-invariant than classical amplitude/power features.  The SPD manifold of
    channel covariance matrices captures inter-muscle co-activation patterns that are
    robust to inter-subject amplitude shifts — analogous to EEG BCI where Riemannian
    approaches achieve SOTA cross-subject performance (Barachant 2013, Congedo 2017).
    The exp_38 result (channel-pair features helped) suggests the covariance geometry
    is informative; Riemannian geometry provides a principled framework for it.

Method:
    1. Per window (N, T, C) → regularised sample covariance Σ ∈ SPD(C)   [C=8 → 8×8]
    2. Reference point M = log-Euclidean mean of TRAINING covariances ONLY.
       No information from validation, test split, or test subject leaks into M.
    3. Tangent space projection at M:
           T_i = M^{1/2}  logm(M^{-1/2} Σ_i M^{-1/2})  M^{1/2}
    4. Vectorise upper triangle with √2 off-diagonal scaling:
           f_i ∈ R^{C(C+1)/2}  (= 36 for C=8)

Variants tested:
    pure_svm_rbf     – tangent(36) + SVM-RBF
    pure_svm_linear  – tangent(36) + SVM-linear  [canonical in EEG Riemannian lit]
    combined_svm_rbf – tangent(36) + PowerfulFeatureExtractor(~253) + SVM-RBF
    hankel_svm_rbf   – Hankel time-embedded cov (2 lags × 20 samp → C'=24, feat=300)
                       tangent(300) + SVM-RBF

LOSO compliance (critical):
    • Riemannian mean M computed from X_train (train-subjects' training windows) ONLY.
    • Feature standardisation (μ, σ) from X_train statistics ONLY.
    • evaluate_numpy() uses the fitted M — no re-fitting on test-subject data.
    • No subject-specific adaptation at test time whatsoever.

Usage:
    python experiments/exp_63_riemannian_spd_covariance_loso.py          # CI (default)
    python experiments/exp_63_riemannian_spd_covariance_loso.py --ci     # same
    python experiments/exp_63_riemannian_spd_covariance_loso.py --full   # 20 subjects
    python experiments/exp_63_riemannian_spd_covariance_loso.py --subjects DB2_s1,DB2_s12
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
#  RiemannianSPDExtractor
# ===========================================================================

class RiemannianSPDExtractor:
    """
    Stateful Riemannian covariance feature extractor.

    Pipeline
    --------
    1. (Optional) Hankel time-embedding: (N, T, C) → (N, T', C*(n_lags+1))
    2. Regularised sample covariance per window:  (N, T', C') → (N, C', C')
    3. fit()      : reference point M = log-Euclidean mean of TRAIN covs only.
                    (True Riemannian Fréchet mean also available via mean_method.)
    4. transform(): tangent-space projection at M + upper-triangle vectorisation.

    LOSO correctness
    ----------------
    Call fit(X_train) with training windows ONLY.
    transform() then uses the fitted M without any re-estimation.

    Args
    ----
    regularization :
        Ridge term ε added to each covariance diagonal. Ensures SPD and
        numerical stability.  Typical range: 1e-5 – 1e-3.
    n_lags :
        Number of temporal lags for Hankel augmentation.
        0  → standard C×C channel covariance.
        k  → augmented signal has C*(k+1) channels, covariance is C*(k+1) square.
    lag_step :
        Lag step in samples between consecutive embeddings.
        (e.g. lag_step=20 at 2000 Hz ≈ 10 ms per lag)
    mean_method :
        "log_euclidean" (fast, good approximation, default) or
        "riemannian"    (true Fréchet mean via gradient descent, slower).
    riemannian_max_iter :
        Maximum Riemannian gradient-descent iterations.
    riemannian_tol :
        Convergence threshold (Frobenius norm of tangent gradient).
    include_log_diag :
        If True, append C' log-eigenvalue features (log-spectrum) per window.
    """

    def __init__(
        self,
        regularization: float = 1e-4,
        n_lags: int = 0,
        lag_step: int = 20,
        mean_method: str = "log_euclidean",
        riemannian_max_iter: int = 30,
        riemannian_tol: float = 1e-7,
        include_log_diag: bool = False,
    ):
        self.regularization = regularization
        self.n_lags = n_lags
        self.lag_step = lag_step
        self.mean_method = mean_method
        self.riemannian_max_iter = riemannian_max_iter
        self.riemannian_tol = riemannian_tol
        self.include_log_diag = include_log_diag

        # Set by fit()
        self.reference_mean_: Optional[np.ndarray] = None   # (C', C') SPD
        self.mean_sqrt_: Optional[np.ndarray] = None        # (C', C')
        self.mean_invsqrt_: Optional[np.ndarray] = None     # (C', C')
        self._fitted: bool = False

    # ── public API ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "RiemannianSPDExtractor":
        """
        Estimate the Riemannian reference point from training windows.

        MUST be called with training-split windows only (LOSO requirement).
        X: (N, T, C) float32.
        """
        covs = self._compute_covariances(X)             # (N, C', C')
        if self.mean_method == "riemannian":
            M = self._riemannian_mean_iterative(covs)
        else:
            M = self._log_euclidean_mean(covs)          # fast default

        self.reference_mean_ = M
        self.mean_sqrt_, self.mean_invsqrt_ = self._spd_sqrt_and_invsqrt(M)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project windows to tangent-space feature vectors.

        X: (N, T, C) → (N, F) float32
            F = C'*(C'+1)/2  [+ C' if include_log_diag]
            where C' = C * (n_lags + 1)
        """
        if not self._fitted:
            raise RuntimeError(
                "RiemannianSPDExtractor.transform() called before fit(). "
                "Call fit(X_train) with training data only."
            )
        covs  = self._compute_covariances(X)        # (N, C', C')
        tang  = self._tangent_project(covs)          # (N, C', C') symmetric
        feats = self._vectorize_upper(tang)          # (N, F)

        if self.include_log_diag:
            log_diag = self._log_eigenvalue_features(covs)   # (N, C')
            feats = np.concatenate([feats, log_diag], axis=1)

        return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # ── covariance computation ──────────────────────────────────────────────

    def _compute_covariances(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C) → (N, C', C') regularised sample covariance.

        Steps:
          1. (Optional) Hankel time-embedding → C' = C*(n_lags+1)
          2. Demean each window along time axis
          3. Σ = X_c^T X_c / (T-1)  +  ε·I
        """
        if self.n_lags > 0:
            X = self._hankel_augment(X)         # (N, T_valid, C')

        N, T, C = X.shape
        X_c  = X - X.mean(axis=1, keepdims=True)   # demean per window
        # Sample covariance (fully vectorised via einsum)
        covs = np.einsum("nti,ntj->nij", X_c, X_c) / max(T - 1, 1)  # (N, C, C)
        # Tikhonov regularisation: Σ_reg = Σ + ε·I
        covs += self.regularization * np.eye(C, dtype=covs.dtype)[np.newaxis]
        return covs

    def _hankel_augment(self, X: np.ndarray) -> np.ndarray:
        """
        Augment signal with n_lags time-lagged copies.

        X: (N, T, C) → (N, T - n_lags*lag_step, C*(n_lags+1))

        The augmented signal at time t:
            [ x(t),  x(t - lag_step),  ...,  x(t - n_lags*lag_step) ]
        Only the T_valid = T - n_lags*lag_step central time points are kept,
        so no boundary artifacts leak into the covariance estimate.
        """
        N, T, C = X.shape
        max_lag = self.n_lags * self.lag_step
        T_valid = T - max_lag
        if T_valid < 4:
            raise ValueError(
                f"Hankel augmentation: n_lags={self.n_lags}, lag_step={self.lag_step} "
                f"reduces T={T} to only {T_valid} valid time points.  "
                f"Reduce n_lags or lag_step."
            )
        channels = [X[:, max_lag:, :]]             # current time: (N, T_valid, C)
        for k in range(1, self.n_lags + 1):
            shift = k * self.lag_step
            channels.append(X[:, max_lag - shift: T - shift, :])   # lagged copy
        return np.concatenate(channels, axis=2)     # (N, T_valid, C*(n_lags+1))

    # ── reference-point (mean) computation ─────────────────────────────────

    def _log_euclidean_mean(self, covs: np.ndarray) -> np.ndarray:
        """
        Log-Euclidean mean:  M = expm( mean_i logm(Σ_i) ).

        This is a closed-form approximation to the true Riemannian mean,
        commonly used in BCI applications.  Complexity: O(N·C'^3).
        """
        log_covs = self._batch_logm(covs)           # (N, C', C')
        mean_log = log_covs.mean(axis=0)            # (C', C') — mean in log-space
        return self._single_expm(mean_log)          # back to SPD space

    def _riemannian_mean_iterative(self, covs: np.ndarray) -> np.ndarray:
        """
        True Riemannian (Fréchet) mean via gradient descent on SPD manifold.

        Algorithm (Moakher 2005; Bhatia 2007):
            M_0  ← log-Euclidean mean  (warm start)
            For t = 0, 1, ...:
                S_i  = logm( M_t^{-1/2} Σ_i M_t^{-1/2} )   [whitened log]
                grad = mean_i(S_i)                            [tangent gradient]
                M_{t+1} = M_t^{1/2} expm(grad) M_t^{1/2}    [geodesic step]
            Stop when  ||grad||_F < tol
        """
        M = self._log_euclidean_mean(covs)          # warm start

        for _ in range(self.riemannian_max_iter):
            M_sqrt, M_invsqrt = self._spd_sqrt_and_invsqrt(M)
            # Whiten: S_i = M^{-1/2} Σ_i M^{-1/2}  (N, C', C')
            S = M_invsqrt[np.newaxis] @ covs @ M_invsqrt[np.newaxis]
            # Log in whitened space
            L = self._batch_logm(S)                 # (N, C', C')
            grad = L.mean(axis=0)                   # (C', C') tangent gradient
            # Geodesic update
            M = M_sqrt @ self._single_expm(grad) @ M_sqrt
            if np.linalg.norm(grad, "fro") < self.riemannian_tol:
                break

        return M

    # ── tangent-space projection ────────────────────────────────────────────

    def _tangent_project(self, covs: np.ndarray) -> np.ndarray:
        """
        Map SPD matrices to tangent space at self.reference_mean_ M.

            T_i = M^{1/2}  logm( M^{-1/2} Σ_i M^{-1/2} )  M^{1/2}

        covs:   (N, C', C') SPD  →  tangent:  (N, C', C') symmetric
        """
        invsq = self.mean_invsqrt_      # (C', C')  M^{-1/2}
        sqrt  = self.mean_sqrt_         # (C', C')  M^{1/2}

        # Whiten: S_i = M^{-1/2} Σ_i M^{-1/2}
        S = invsq[np.newaxis] @ covs @ invsq[np.newaxis]    # (N, C', C')
        # Symmetric matrix log in whitened space
        L = self._batch_logm(S)                             # (N, C', C')
        # Map back to tangent space at M
        T = sqrt[np.newaxis] @ L @ sqrt[np.newaxis]         # (N, C', C')
        return T

    # ── vectorisation ───────────────────────────────────────────────────────

    @staticmethod
    def _vectorize_upper(T: np.ndarray) -> np.ndarray:
        """
        Vectorise symmetric matrices using upper triangle with √2 scaling.

        T: (N, C', C') → (N, C'*(C'+1)/2)

        Convention (Barachant et al. 2010):
            diagonal elements  × 1
            off-diagonal i < j × √2    (preserves Frobenius inner product)
        """
        N, C, _ = T.shape
        idx_i, idx_j = np.triu_indices(C)                       # (F,) each
        scale = np.where(idx_i == idx_j, 1.0, np.sqrt(2.0))    # (F,)
        vecs  = T[:, idx_i, idx_j] * scale[np.newaxis, :]      # (N, F)
        return vecs

    def _log_eigenvalue_features(self, covs: np.ndarray) -> np.ndarray:
        """
        Append log-spectrum (log-eigenvalues) of each covariance.

        covs: (N, C', C') → (N, C')
        These represent log-power in principal co-activation directions.
        """
        eigvals, _ = np.linalg.eigh(covs)                       # (N, C')
        return np.log(np.maximum(eigvals, 1e-12))               # (N, C')

    # ── batched matrix operations via eigendecomposition ───────────────────
    #   For SPD matrices, eigdecomp is O(C^3) and numpy ≥ 1.14 supports
    #   batched eigh(), so the full pipeline is O(N · C^3) — no Python loops.

    @staticmethod
    def _batch_logm(Ms: np.ndarray) -> np.ndarray:
        """
        Batched matrix logarithm for SPD matrices via eigendecomposition.

        Ms: (N, C, C) SPD → (N, C, C)
            logm(M) = V · diag(log λ) · V^T
        """
        eigvals, eigvecs = np.linalg.eigh(Ms)               # (N,C), (N,C,C)
        log_eigvals = np.log(np.maximum(eigvals, 1e-12))    # (N, C)
        # logm(M)_n = eigvecs_n @ diag(log_eigvals_n) @ eigvecs_n^T
        return np.einsum("nij,nj,nkj->nik", eigvecs, log_eigvals, eigvecs)

    @staticmethod
    def _spd_sqrt_and_invsqrt(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute M^{1/2} and M^{-1/2} for a single SPD matrix.
        M: (C, C) → (sqrt_M, invsqrt_M), each (C, C).
        """
        eigvals, eigvecs = np.linalg.eigh(M)                # (C,), (C, C)
        eigvals_pos  = np.maximum(eigvals, 1e-12)
        sqrt_vals    = np.sqrt(eigvals_pos)
        invsqrt_vals = 1.0 / sqrt_vals
        sqrt_M    = (eigvecs * sqrt_vals)    @ eigvecs.T    # V diag(√λ) V^T
        invsqrt_M = (eigvecs * invsqrt_vals) @ eigvecs.T   # V diag(1/√λ) V^T
        return sqrt_M, invsqrt_M

    @staticmethod
    def _single_expm(M: np.ndarray) -> np.ndarray:
        """
        Matrix exponent for a single symmetric matrix.
        M: (C, C) symmetric → (C, C) SPD.
            expm(M) = V · diag(exp λ) · V^T
        """
        eigvals, eigvecs = np.linalg.eigh(M)
        return (eigvecs * np.exp(eigvals)) @ eigvecs.T


# ===========================================================================
#  RiemannianMLTrainer
#  Implements the full trainer interface expected by CrossSubjectExperiment:
#    fit(splits) → Dict
#    evaluate_numpy(X, y, split_name, visualize) → Dict
#    self.class_ids: List[int]
# ===========================================================================

class RiemannianMLTrainer:
    """
    ML trainer using Riemannian tangent-space features.

    LOSO correctness
    ----------------
    • RiemannianSPDExtractor.fit() is called with X_train ONLY (train subjects'
      training windows).  X_val and X_test are transformed, never used for fitting.
    • Feature standardisation (μ, σ) is derived from X_train ONLY.
    • evaluate_numpy() uses the fitted extractor — no re-fitting on test data.

    Args
    ----
    extractor_cfg   : kwargs passed verbatim to RiemannianSPDExtractor().
    combine_powerful: If True, concatenate PowerfulFeatureExtractor features
                      with the Riemannian tangent vector before the SVM.
    train_cfg       : TrainingConfig (we use .seed and .ml_model_type).
    logger          : Python logger.
    output_dir      : Path for artefacts.
    visualizer      : Optional Visualizer (not used here, kept for API compat).
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
        self.extractor_cfg   = extractor_cfg
        self.combine_powerful = combine_powerful
        self.cfg             = train_cfg
        self.logger          = logger
        self.output_dir      = Path(output_dir)
        self.visualizer      = visualizer

        # Set during fit()
        self._extractor: Optional[RiemannianSPDExtractor] = None
        self._powerful_extractor = None     # PowerfulFeatureExtractor if combine_powerful
        self.ml_model       = None
        self.feature_mean:   Optional[np.ndarray] = None
        self.feature_std:    Optional[np.ndarray] = None
        self.class_ids:      Optional[List[int]]  = None
        self.class_names:    Optional[Dict]        = None

    # ── fit ────────────────────────────────────────────────────────────────

    def fit(self, splits: Dict[str, Dict[int, np.ndarray]]) -> Dict:
        """
        Train the SVM on Riemannian tangent-space features.

        splits["train"] — train-subjects training windows  ← used for fitting
        splits["val"]   — train-subjects validation windows
        splits["test"]  — TEST SUBJECT windows             ← NEVER used for fitting

        The Riemannian mean is estimated from X_train only.
        Feature standardisation stats are from X_train only.
        """
        from utils.logging import seed_everything
        seed_everything(self.cfg.seed)

        # ── 1. Unpack splits → flat (N, T, C) arrays ──────────────────────
        (
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            class_ids, class_names,
        ) = self._prepare_splits_arrays(splits)

        num_classes = len(class_ids)

        # ── 2. Fit Riemannian extractor on X_train ONLY ─────────────────── ← key
        self._extractor = RiemannianSPDExtractor(**self.extractor_cfg)
        self.logger.info(
            f"[RiemannianMLTrainer] Fitting Riemannian extractor on "
            f"X_train={X_train.shape}  (train data only — LOSO safe)"
        )
        self._extractor.fit(X_train)

        # Build PowerfulFeatureExtractor once (stateless, reused in evaluate_numpy)
        if self.combine_powerful:
            from processing.powerful_features import PowerfulFeatureExtractor
            self._powerful_extractor = PowerfulFeatureExtractor(sampling_rate=2000)

        # ── 3. Extract features for all splits (transform only — no fitting) ──
        F_train = self._extract_features(X_train)       # (N_train, F)
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
            f"[RiemannianMLTrainer] Feature dim={F_train.shape[1]}  "
            f"(train={F_train.shape[0]}, val={F_val.shape[0]}, "
            f"test={F_test.shape[0]})"
        )

        # ── 4. Standardise — stats from X_train ONLY ──────────────────────── ← key
        self.feature_mean = F_train.mean(axis=0).astype(np.float32)    # (F,)
        self.feature_std  = (F_train.std(axis=0) + 1e-8).astype(np.float32)

        F_train = self._standardize(F_train)
        F_val   = self._standardize(F_val)
        F_test  = self._standardize(F_test)

        # ── 5. Train SVM ───────────────────────────────────────────────────
        ml_model_type = getattr(self.cfg, "ml_model_type", "svm_rbf")
        self.logger.info(f"[RiemannianMLTrainer] Training SVM ({ml_model_type}) "
                         f"on {F_train.shape[0]} samples, {num_classes} classes")

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
                f"[RiemannianMLTrainer] Unknown ml_model_type='{ml_model_type}'. "
                f"Choose: svm_rbf, svm_linear."
            )
        self.ml_model.fit(F_train, y_train)

        # ── 6. Store class metadata ─────────────────────────────────────────
        self.class_ids   = class_ids
        self.class_names = class_names

        # ── 7. Evaluate splits and build results dict ───────────────────────
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
                f"[RiemannianMLTrainer] {split_name}: "
                f"acc={acc:.4f}, f1={f1_mac:.4f}"
            )
            results[split_name] = {
                "accuracy": acc,
                "f1_macro": f1_mac,
                "report":   report,
                "confusion_matrix": cm.tolist(),
            }

        # Save results JSON
        results_path = self.output_dir / "riemannian_results.json"
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
        Evaluate on raw windows (N, T, C) using the fitted Riemannian pipeline.

        The fitted Riemannian mean M (from training data) is used for projection.
        No re-fitting of any kind.  Safe for test-subject evaluation in LOSO.
        """
        assert self.ml_model is not None, "Call fit() before evaluate_numpy()"
        assert self.feature_mean is not None and self.feature_std is not None, \
            "Feature stats missing — call fit() first"
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

    # ── internal helpers ────────────────────────────────────────────────────

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract Riemannian tangent features [+ PowerfulFeatureExtractor] from X.

        X: (N, T, C) → (N, F) float32.
        The Riemannian extractor must already be fitted.
        PowerfulFeatureExtractor is stateless (no fitting needed).
        """
        F = self._extractor.transform(X)                        # (N, F_riem)
        if self.combine_powerful and self._powerful_extractor is not None:
            P = self._powerful_extractor.transform(X)           # (N, F_pow)
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

        splits["train"] = Dict[gesture_id, ndarray (N, T, C)] — train subjects
        splits["val"]   = Dict[gesture_id, ndarray (N, T, C)] — train subjects
        splits["test"]  = Dict[gesture_id, ndarray (N, T, C)] — TEST SUBJECT

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test,
            class_ids (sorted gesture IDs),
            class_names {gid → str}
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

        # Drop classes absent from train
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
            f"[RiemannianMLTrainer] Split sizes — "
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
    """Run one LOSO fold for the given Riemannian variant."""
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

    # Record variant in training config for artefact traceability
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
        use_gpu=False,          # numpy-only pipeline — GPU not needed
        use_improved_processing=True,
    )

    base_viz  = Visualizer(output_dir, logger)
    cross_viz = CrossSubjectVisualizer(output_dir, logger)

    # Fresh trainer per fold — extractor is stateful (Riemannian mean)
    trainer = RiemannianMLTrainer(
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
            "test_subject":    test_subject,
            "train_subjects":  train_subjects,
            "variant":         variant_name,
            "extractor_cfg":   extractor_cfg,
            "combine_powerful": combine_powerful,
            "ml_model_type":   ml_model_type,
            "exercises":       exercises,
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
    EXPERIMENT_NAME = "exp_63_riemannian_spd_covariance_loso"
    BASE_DIR         = ROOT / "data"
    ALL_SUBJECTS     = parse_subjects_args()
    OUTPUT_DIR       = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES        = ["E1"]

    # ── Variant definitions ──────────────────────────────────────────────
    #
    # extractor_cfg is passed verbatim to RiemannianSPDExtractor(**extractor_cfg).
    # combine_powerful = True concatenates PowerfulFeatureExtractor features.
    #
    # Feature dimensions (C=8):
    #   pure tangent   : C*(C+1)/2 = 36
    #   combined       : 36 + ~253 = ~289  (depends on PowerfulFeatureExtractor)
    #   hankel (k=2)   : C'=24 → C'*(C'+1)/2 = 300
    #
    VARIANTS: List[Dict] = [
        {
            "name":             "pure_svm_rbf",
            "extractor_cfg":    {"regularization": 1e-4, "n_lags": 0},
            "combine_powerful": False,
            "ml_model_type":    "svm_rbf",
        },
        {
            "name":             "pure_svm_linear",
            # SVM-linear is the canonical classifier in EEG Riemannian BCI literature
            "extractor_cfg":    {"regularization": 1e-4, "n_lags": 0},
            "combine_powerful": False,
            "ml_model_type":    "svm_linear",
        },
        {
            "name":             "combined_svm_rbf",
            # Riemannian tangent + classic PowerfulFeatureExtractor features
            "extractor_cfg":    {"regularization": 1e-4, "n_lags": 0},
            "combine_powerful": True,
            "ml_model_type":    "svm_rbf",
        },
        {
            "name":             "hankel_svm_rbf",
            # Hankel time-embedded covariance: n_lags=2, lag_step=20 samp (10 ms @ 2 kHz)
            # Augmented channels: 8 × 3 = 24  →  feature dim = 24×25/2 = 300
            "extractor_cfg":    {"regularization": 1e-4, "n_lags": 2, "lag_step": 20},
            "combine_powerful": False,
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
        feat_dim_note = (
            "C*(C+1)/2=36" if v["extractor_cfg"]["n_lags"] == 0 else
            f"C'={8*(v['extractor_cfg']['n_lags']+1)} → feat={8*(v['extractor_cfg']['n_lags']+1)*(8*(v['extractor_cfg']['n_lags']+1)+1)//2}"
        )
        print(
            f"    {v['name']:30s}  feat={feat_dim_note}  "
            f"{'+ Powerful' if v['combine_powerful'] else '':10s}  "
            f"clf={v['ml_model_type']}"
        )

    all_loso_results: List[Dict] = []

    for variant in VARIANTS:
        vname    = variant["name"]
        ext_cfg  = variant["extractor_cfg"]
        combine  = variant["combine_powerful"]
        ml_type  = variant["ml_model_type"]

        print(f"\n{'=' * 65}")
        print(f"  Variant: {vname}  —  starting LOSO ({len(ALL_SUBJECTS)} subjects)")
        print(f"  extractor_cfg={ext_cfg}, combine_powerful={combine}, clf={ml_type}")
        print(f"{'=' * 65}")

        for test_subject in ALL_SUBJECTS:
            train_subjects  = [s for s in ALL_SUBJECTS if s != test_subject]
            fold_output_dir = OUTPUT_DIR / vname / f"test_{test_subject}"

            # Copy train_cfg and set ml_model_type for this variant
            # (TrainingConfig is a dataclass — copy via replace or just mutate,
            #  since each fold re-reads the attributes it needs)
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

    # ── Aggregate results ─────────────────────────────────────────────────
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

    # ── Summary print ────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"SUMMARY: {EXPERIMENT_NAME}")
    print(f"{'=' * 65}")
    for vn, res in aggregate_results.items():
        acc_m = res["mean_accuracy"]
        acc_s = res["std_accuracy"]
        f1_m  = res["mean_f1_macro"]
        f1_s  = res["std_f1_macro"]
        n_sub = res["num_subjects"]
        print(
            f"  {vn:30s}  "
            f"Acc={acc_m:.4f}±{acc_s:.4f}  "
            f"F1={f1_m:.4f}±{f1_s:.4f}  "
            f"(N={n_sub})"
        )

    # ── Save summary JSON ────────────────────────────────────────────────
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis": (
            "Riemannian tangent-space covariance features are more cross-subject "
            "invariant than classical EMG amplitude/power descriptors.  "
            "SPD covariance captures inter-channel muscle co-activation patterns "
            "robustly to per-subject amplitude shifts (cf. EEG Riemannian BCI)."
        ),
        "approach": "RiemannianSPDExtractor (log-Euclidean mean) + SVM",
        "loso_compliance": (
            "Riemannian mean M fitted on X_train (train subjects' training windows) ONLY.  "
            "Feature standardisation stats from X_train ONLY.  "
            "evaluate_numpy() uses fitted M — no test-subject data influences any parameter."
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

    # ── Notify hypothesis executor (optional dependency) ─────────────────
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
                hypothesis_id="H63",
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
        print(f"[exp_63] hypothesis_executor notification failed: {_he_err}")


if __name__ == "__main__":
    main()
