"""
Exp 71: Implicit Disentanglement via Structured Dropout (ChannelDrop + BandDrop + TemporalDrop)
================================================================================================

Hypothesis:
    EMG models overfit on subject-specific patterns localised in particular channels
    and/or frequency bands ("style shortcuts").  Forcing the model to classify correctly
    even when those cues are randomly masked during training pushes it to learn
    gesture content rather than subject style — analogous to SpecAugment (ASR),
    ChannelDrop (EEG/BCI), and Cutout (CV).

Augmentation pipeline (training only; inference is CLEAN — no masking):
    1. ChannelDrop  – zero out k∈[1, channel_drop_max] random EMG channels per window.
    2. BandDrop     – zero out n∈[1, band_drop_max] random frequency bands via FFT
                      (spectral masking, 5–20 % of the spectrum per band).
    3. TemporalDrop – zero out m∈[1, temporal_drop_max] random contiguous time segments
                      (each 5–20 % of window length, CutOut style).

    Each drop type is applied independently with probability p (default 0.5).
    Augmentation is in Dataset.__getitem__ so every epoch sees a different mask.

LOSO compliance (verified):
    ✓ Normalization stats (mean/std) computed from train-split of training subjects only.
    ✓ StructuredDropDataset used ONLY for the training split; val/test use WindowDataset.
    ✓ Augmentation parameters (p, max) are fixed global hyperparameters — no fitting
      to any subject's data, not even the training subjects.
    ✓ Test-subject windows are never seen before CrossSubjectExperiment.evaluate_numpy().
    ✓ No subject identity information flows into the augmentation decisions.

Experiment design:
    - Model: cnn_gru_attention (best single deep model from earlier experiments).
    - Two configurations per LOSO fold:
        A) Baseline  : aug_apply=False → plain WindowDataset (no augmentation).
        B) StructDrop: aug_apply=True  → StructuredDropDataset (channel+band+temporal).
    - Dataset: NinaPro DB2, E1, 5 CI subjects by default.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Subject lists and CLI parsing
# ---------------------------------------------------------------------------

_FULL_SUBJECTS = [
    "DB2_s1", "DB2_s2", "DB2_s3", "DB2_s4", "DB2_s5",
    "DB2_s11", "DB2_s12", "DB2_s13", "DB2_s14", "DB2_s15",
    "DB2_s26", "DB2_s27", "DB2_s28", "DB2_s29", "DB2_s30",
    "DB2_s36", "DB2_s37", "DB2_s38", "DB2_s39", "DB2_s40",
]
_CI_SUBJECTS = ["DB2_s1", "DB2_s12", "DB2_s15", "DB2_s28", "DB2_s39"]


def _parse_subjects_args(argv: Optional[List[str]] = None) -> List[str]:
    """Parse --subjects / --ci / --full CLI args. Defaults to CI subjects."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--ci",   action="store_true")
    parser.add_argument("--full", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.subjects:
        return [s.strip() for s in args.subjects.split(",")]
    if args.full:
        return _FULL_SUBJECTS
    # Default: CI subjects — safe on the vast.ai server which has only CI symlinks
    return _CI_SUBJECTS


# ---------------------------------------------------------------------------
# Imports from OMEGA core
# ---------------------------------------------------------------------------

from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from config.cross_subject import CrossSubjectConfig
from data.multi_subject_loader import MultiSubjectLoader
from evaluation.cross_subject import CrossSubjectExperiment
from training.trainer import WindowClassifierTrainer
from training.datasets import WindowDataset
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ---------------------------------------------------------------------------
# StructuredDropDataset
# ---------------------------------------------------------------------------

class StructuredDropDataset(Dataset):
    """
    Training-only dataset implementing three structured dropout augmentations.

    LOSO compliance:
        - All augmentation is purely random: no subject-specific statistics are
          estimated or used.  The same distribution of masks applies equally to
          every training subject.
        - Used only for the training split.  Val and test always use WindowDataset.

    Args:
        X: (N, C, T) normalised training windows (float32, already standardised).
        y: (N,) integer class labels.

        p_channel:            Probability of applying ChannelDrop to a window.
        p_band:               Probability of applying BandDrop (spectral mask).
        p_temporal:           Probability of applying TemporalDrop.

        channel_drop_max:     Max number of channels zeroed per application.
                              Clamped to C-1 so at least one channel survives.
        band_drop_max:        Max number of frequency bands zeroed per application.
        temporal_drop_max:    Max number of contiguous time segments zeroed.
        temporal_drop_min_frac: Minimum length of each temporal mask as fraction of T.
        temporal_drop_max_frac: Maximum length of each temporal mask as fraction of T.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        p_channel: float = 0.5,
        p_band: float = 0.5,
        p_temporal: float = 0.5,
        channel_drop_max: int = 2,
        band_drop_max: int = 2,
        temporal_drop_max: int = 2,
        temporal_drop_min_frac: float = 0.05,
        temporal_drop_max_frac: float = 0.20,
    ):
        assert X.ndim == 3, f"Expected X shape (N, C, T), got {X.shape}"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

        self.p_channel = p_channel
        self.p_band = p_band
        self.p_temporal = p_temporal
        self.channel_drop_max = channel_drop_max
        self.band_drop_max = band_drop_max
        self.temporal_drop_max = temporal_drop_max
        self.temporal_drop_min_frac = temporal_drop_min_frac
        self.temporal_drop_max_frac = temporal_drop_max_frac

    # ------------------------------------------------------------------
    # Individual augmentation transforms  (all operate on (C, T) arrays)
    # ------------------------------------------------------------------

    def _channel_drop(self, x: np.ndarray) -> np.ndarray:
        """
        Zero out 1..min(channel_drop_max, C-1) randomly chosen channels.

        Guarantees at least one channel survives.
        No subject-specific information involved — pure random channel selection.
        """
        C = x.shape[0]
        max_drop = min(self.channel_drop_max, C - 1)
        if max_drop < 1:
            return x
        n_drop = np.random.randint(1, max_drop + 1)
        channels = np.random.choice(C, n_drop, replace=False)
        x = x.copy()
        x[channels, :] = 0.0
        return x

    def _band_drop(self, x: np.ndarray) -> np.ndarray:
        """
        Zero out 1..band_drop_max random frequency bands via rFFT.

        Each band occupies 5–20 % of the one-sided spectrum.
        Zeroing is done in the complex FFT domain: both magnitude and phase are
        removed, so the reconstructed signal carries no information from that band.

        Why FFT and not a learnable filter: FFT-based masking is independent of
        any subject's signal statistics — the masking positions are drawn uniformly
        over the spectrum, not calibrated to any subject.

        Edge cases:
            - If n_freqs < 4 (window too short) the transform is skipped.
            - Band boundaries are clamped to [0, n_freqs].
        """
        C, T = x.shape
        X_fft = np.fft.rfft(x, axis=1)   # (C, n_freqs),  n_freqs = T//2 + 1
        n_freqs = X_fft.shape[1]

        if n_freqs < 4:
            return x  # degenerate window — skip

        n_bands = np.random.randint(1, self.band_drop_max + 1)
        result = X_fft.copy()

        for _ in range(n_bands):
            # Band width: 5 % – 20 % of one-sided spectrum
            min_w = max(1, int(n_freqs * 0.05))
            max_w = max(min_w + 1, int(n_freqs * 0.20))
            band_w = np.random.randint(min_w, max_w + 1)
            # Clamp start so the band fits inside the spectrum
            band_start = np.random.randint(0, max(1, n_freqs - band_w))
            result[:, band_start:band_start + band_w] = 0.0

        # irfft with explicit n=T restores the original length
        return np.fft.irfft(result, n=T, axis=1).astype(np.float32)

    def _temporal_drop(self, x: np.ndarray) -> np.ndarray:
        """
        Zero out 1..temporal_drop_max random contiguous time segments (CutOut).

        Each segment length is drawn from [min_frac*T, max_frac*T].
        Segment positions are drawn uniformly — no alignment to gesture boundaries,
        no subject-specific timing adaptation.
        """
        C, T = x.shape
        n_cuts = np.random.randint(1, self.temporal_drop_max + 1)
        x = x.copy()

        for _ in range(n_cuts):
            min_len = max(1, int(T * self.temporal_drop_min_frac))
            max_len = min(T - 1, max(min_len + 1, int(T * self.temporal_drop_max_frac)))
            cut_len = (
                min_len if min_len >= max_len
                else np.random.randint(min_len, max_len + 1)
            )
            # Ensure there is always room for the cut
            cut_start = np.random.randint(0, max(1, T - cut_len))
            x[:, cut_start:cut_start + cut_len] = 0.0

        return x

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (C, T) — local copy to avoid modifying the store

        # Each augmentation is applied independently with probability p.
        # Ordering: channel → band → temporal (order does not matter for zeroing ops).
        if np.random.random() < self.p_channel:
            x = self._channel_drop(x)
        if np.random.random() < self.p_band:
            x = self._band_drop(x)
        if np.random.random() < self.p_temporal:
            x = self._temporal_drop(x)

        return torch.from_numpy(x).float(), torch.tensor(self.y[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# StructuredDropTrainer
# ---------------------------------------------------------------------------

class StructuredDropTrainer(WindowClassifierTrainer):
    """
    Extends WindowClassifierTrainer by injecting StructuredDropDataset as the
    training dataset.  All other logic (normalisation, model creation, training
    loop, evaluation) is inherited unchanged.

    LOSO compliance:
        - _make_train_dataset() is called inside fit() AFTER normalisation stats
          have been computed from X_train.  The dataset receives already-normalised
          windows and applies purely random masks — no subject identity leaks.
        - Val and test datasets are always plain WindowDataset (no augmentation),
          ensured by the parent class.
        - evaluate_numpy() is inherited verbatim — no augmentation at inference.

    Args:
        train_cfg, logger, output_dir, visualizer: forwarded to parent.
        p_channel, p_band, p_temporal: probability of each drop type.
        channel_drop_max: max channels to zero out (≥1, clamped to C-1).
        band_drop_max: max frequency bands to zero out (≥1).
        temporal_drop_max: max temporal segments to zero out (≥1).
        temporal_drop_min_frac: min fractional length of temporal mask.
        temporal_drop_max_frac: max fractional length of temporal mask.
    """

    def __init__(
        self,
        *args,
        p_channel: float = 0.5,
        p_band: float = 0.5,
        p_temporal: float = 0.5,
        channel_drop_max: int = 2,
        band_drop_max: int = 2,
        temporal_drop_max: int = 2,
        temporal_drop_min_frac: float = 0.05,
        temporal_drop_max_frac: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p_channel = p_channel
        self.p_band = p_band
        self.p_temporal = p_temporal
        self.channel_drop_max = channel_drop_max
        self.band_drop_max = band_drop_max
        self.temporal_drop_max = temporal_drop_max
        self.temporal_drop_min_frac = temporal_drop_min_frac
        self.temporal_drop_max_frac = temporal_drop_max_frac

    # Override the factory hook — everything else stays as-is in the parent.
    def _make_train_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_aug: bool,
    ):
        """
        Returns StructuredDropDataset when use_aug=True; plain WindowDataset otherwise.

        Called by parent fit() with X_train already normalised and in (N, C, T) format.
        No test-subject data flows here — guaranteed by CrossSubjectExperiment.
        """
        if not use_aug:
            self.logger.info("[StructuredDrop] aug_apply=False → using plain WindowDataset")
            return WindowDataset(X_train, y_train)

        self.logger.info(
            f"[StructuredDrop] Augmentations enabled:\n"
            f"  ChannelDrop : p={self.p_channel}, max_channels={self.channel_drop_max}\n"
            f"  BandDrop    : p={self.p_band},    max_bands={self.band_drop_max}\n"
            f"  TemporalDrop: p={self.p_temporal}, max_segs={self.temporal_drop_max}, "
            f"frac=[{self.temporal_drop_min_frac}, {self.temporal_drop_max_frac}]\n"
            f"  Training set size: {len(X_train)} windows — NO augmentation on val/test."
        )
        return StructuredDropDataset(
            X_train, y_train,
            p_channel=self.p_channel,
            p_band=self.p_band,
            p_temporal=self.p_temporal,
            channel_drop_max=self.channel_drop_max,
            band_drop_max=self.band_drop_max,
            temporal_drop_max=self.temporal_drop_max,
            temporal_drop_min_frac=self.temporal_drop_min_frac,
            temporal_drop_max_frac=self.temporal_drop_max_frac,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_json_serializable(obj):
    from pathlib import Path as _Path
    import numpy as _np

    if isinstance(obj, _Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, _np.integer):
        return int(obj)
    elif isinstance(obj, _np.floating):
        return float(obj)
    elif isinstance(obj, _np.ndarray):
        return obj.tolist()
    else:
        return obj


# ---------------------------------------------------------------------------
# Single LOSO fold
# ---------------------------------------------------------------------------

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    model_type: str,
    approach: str,
    use_improved_processing: bool,
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
    # Structured-drop hyperparameters
    use_structured_drop: bool = True,
    p_channel: float = 0.5,
    p_band: float = 0.5,
    p_temporal: float = 0.5,
    channel_drop_max: int = 2,
    band_drop_max: int = 2,
    temporal_drop_max: int = 2,
    temporal_drop_min_frac: float = 0.05,
    temporal_drop_max_frac: float = 0.20,
) -> Dict:
    """
    Run one LOSO fold.

    LOSO leak audit:
        train_subjects  — subjects used for training (test_subject excluded by caller).
        test_subject    — held-out subject; data loaded and evaluated at the very end.
        CrossSubjectExperiment loads test_subject data inside run() and passes it only
        to evaluate_numpy() — never to fit().
        Normalisation stats are computed inside fit() from train-split windows of
        train_subjects only.
        Augmentation (StructuredDropDataset) receives only those normalised train-split
        windows and applies random masks — zero subject-specific state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = approach
    train_cfg.model_type = model_type

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

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
        use_gpu=True,
        use_improved_processing=use_improved_processing,
    )
    base_viz = Visualizer(output_dir, logger)

    # -----------------------------------------------------------------------
    # Trainer selection
    # -----------------------------------------------------------------------
    # Baseline: standard WindowClassifierTrainer (no aug or existing aug only).
    # StructDrop: StructuredDropTrainer injects StructuredDropDataset for training.
    #
    # LOSO note: both trainers receive data via CrossSubjectExperiment.run(), which
    # guarantees that test_subject windows are isolated from fit().
    # -----------------------------------------------------------------------
    if use_structured_drop:
        trainer = StructuredDropTrainer(
            train_cfg=train_cfg,
            logger=logger,
            output_dir=output_dir,
            visualizer=base_viz,
            p_channel=p_channel,
            p_band=p_band,
            p_temporal=p_temporal,
            channel_drop_max=channel_drop_max,
            band_drop_max=band_drop_max,
            temporal_drop_max=temporal_drop_max,
            temporal_drop_min_frac=temporal_drop_min_frac,
            temporal_drop_max_frac=temporal_drop_max_frac,
        )
    else:
        trainer = WindowClassifierTrainer(
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
        logger.error(f"LOSO fold failed (test={test_subject}): {e}")
        traceback.print_exc()
        return {
            "test_subject": test_subject,
            "model_type": model_type,
            "approach": approach,
            "use_structured_drop": use_structured_drop,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error": str(e),
        }

    test_metrics = results.get("cross_subject_test", {})
    test_acc = float(test_metrics.get("accuracy", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))

    tag = "StructDrop" if use_structured_drop else "Baseline"
    print(
        f"[LOSO] {tag:10s} | test={test_subject} | model={model_type} | "
        f"Acc={test_acc:.4f}  F1={test_f1:.4f}"
    )

    results_to_save = {k: v for k, v in results.items() if k != "subjects_data"}
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(results_to_save), f, indent=4, ensure_ascii=False)

    saver = ArtifactSaver(output_dir, logger)
    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "model_type": model_type,
        "approach": approach,
        "use_structured_drop": use_structured_drop,
        "exercises": exercises,
        "use_improved_processing": use_improved_processing,
        "drop_params": {
            "p_channel": p_channel,
            "p_band": p_band,
            "p_temporal": p_temporal,
            "channel_drop_max": channel_drop_max,
            "band_drop_max": band_drop_max,
            "temporal_drop_max": temporal_drop_max,
            "temporal_drop_min_frac": temporal_drop_min_frac,
            "temporal_drop_max_frac": temporal_drop_max_frac,
        },
        "config": {
            "processing": asdict(proc_cfg),
            "split": asdict(split_cfg),
            "training": asdict(train_cfg),
            "cross_subject": {
                "train_subjects": train_subjects,
                "test_subject": test_subject,
                "exercises": exercises,
            },
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    del experiment, trainer, multi_loader, base_viz
    gc.collect()

    return {
        "test_subject": test_subject,
        "model_type": model_type,
        "approach": approach,
        "use_structured_drop": use_structured_drop,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def _aggregate(results: List[Dict], tag: str) -> Dict:
    """Compute mean/std/min/max accuracy and F1 over successful LOSO folds."""
    ok = [r for r in results if r.get("test_accuracy") is not None]
    if not ok:
        return {}
    accs = [r["test_accuracy"] for r in ok]
    f1s  = [r["test_f1_macro"] for r in ok]
    return {
        "tag": tag,
        "num_subjects": len(ok),
        "mean_accuracy":  float(np.mean(accs)),
        "std_accuracy":   float(np.std(accs)),
        "min_accuracy":   float(np.min(accs)),
        "max_accuracy":   float(np.max(accs)),
        "mean_f1_macro":  float(np.mean(f1s)),
        "std_f1_macro":   float(np.std(f1s)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPERIMENT_NAME = "exp_71_structured_channel_band_temporal_drop_loso"
    HYPOTHESIS_ID   = ""  # fill in when registered

    ALL_SUBJECTS = _parse_subjects_args()
    BASE_DIR     = ROOT / "data"
    OUTPUT_DIR   = Path(f"./experiments_output/{EXPERIMENT_NAME}")
    EXERCISES    = ["E1"]

    # -----------------------------------------------------------------------
    # Shared processing / split / training configs
    # -----------------------------------------------------------------------
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

    # Base training config — identical for both configurations.
    # aug_apply=True activates the dataset factory path in the trainer.
    # For the baseline, we pass aug_apply=False to run_single_loso_fold.
    _base_train_cfg = dict(
        batch_size=4096,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=7,
        use_class_weights=True,
        seed=42,
        num_workers=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_handcrafted_features=False,
        pipeline_type="deep_raw",
        model_type="cnn_gru_attention",
    )

    # Configurations to sweep
    # -----------------------------------------------------------------------
    # A) Baseline: no augmentation at all.
    # B) StructDrop: ChannelDrop + BandDrop + TemporalDrop.
    #
    # Why not also test partial combinations?
    # A proper ablation (channel-only, band-only, temporal-only) would require
    # 5× as many LOSO folds.  We run the full combination first to test the
    # hypothesis; ablations can follow in exp_72+ if the result is positive.
    # -----------------------------------------------------------------------
    CONFIGS = [
        {
            "tag": "baseline",
            "use_structured_drop": False,
            "train_cfg_overrides": {"aug_apply": False},
            # drop params unused for baseline
        },
        {
            "tag": "struct_drop",
            "use_structured_drop": True,
            "train_cfg_overrides": {"aug_apply": True},
            # Structured-drop hyperparameters (see StructuredDropDataset docstring)
            "drop_params": {
                "p_channel":            0.5,   # 50 % chance per window
                "p_band":               0.5,
                "p_temporal":           0.5,
                "channel_drop_max":     2,     # at most 2 of 8 channels zeroed
                "band_drop_max":        2,     # at most 2 spectral bands zeroed
                "temporal_drop_max":    2,     # at most 2 time segments zeroed
                "temporal_drop_min_frac": 0.05,  # min 5 % of T (~30 samples)
                "temporal_drop_max_frac": 0.20,  # max 20 % of T (~120 samples)
            },
        },
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(OUTPUT_DIR)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"Model: cnn_gru_attention | LOSO n={len(ALL_SUBJECTS)} subjects")
    print(f"Configurations: {[c['tag'] for c in CONFIGS]}")
    print(f"{'='*70}\n")

    all_results_by_tag: Dict[str, List[Dict]] = {c["tag"]: [] for c in CONFIGS}

    for cfg_spec in CONFIGS:
        tag = cfg_spec["tag"]
        use_sd = cfg_spec["use_structured_drop"]
        drop_params = cfg_spec.get("drop_params", {})

        # Build TrainingConfig for this configuration
        tc_kwargs = {**_base_train_cfg, **cfg_spec["train_cfg_overrides"]}
        train_cfg = TrainingConfig(**tc_kwargs)

        print(f"\n--- Configuration: {tag.upper()} ---")
        for test_subj in ALL_SUBJECTS:
            train_subjs = [s for s in ALL_SUBJECTS if s != test_subj]
            fold_dir = OUTPUT_DIR / tag / f"test_{test_subj}"

            try:
                fold_res = run_single_loso_fold(
                    base_dir=BASE_DIR,
                    output_dir=fold_dir,
                    train_subjects=train_subjs,
                    test_subject=test_subj,
                    exercises=EXERCISES,
                    model_type="cnn_gru_attention",
                    approach="deep_raw",
                    use_improved_processing=True,
                    proc_cfg=proc_cfg,
                    split_cfg=split_cfg,
                    train_cfg=train_cfg,
                    use_structured_drop=use_sd,
                    **drop_params,
                )
                all_results_by_tag[tag].append(fold_res)
                acc = fold_res.get("test_accuracy")
                f1  = fold_res.get("test_f1_macro")
                if acc is not None and f1 is not None:
                    print(f"  ✓ {test_subj}: acc={acc:.4f}  f1={f1:.4f}")
                else:
                    print(f"  ✗ {test_subj}: {fold_res.get('error', 'unknown error')}")
            except Exception as e:
                global_logger.error(f"Fold failed ({tag}, {test_subj}): {e}")
                traceback.print_exc()
                all_results_by_tag[tag].append({
                    "test_subject": test_subj,
                    "model_type": "cnn_gru_attention",
                    "approach": "deep_raw",
                    "use_structured_drop": use_sd,
                    "test_accuracy": None,
                    "test_f1_macro": None,
                    "error": str(e),
                })

    # -----------------------------------------------------------------------
    # Aggregate and print summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LOSO SUMMARY")
    print(f"{'='*70}")

    aggregates = {}
    for tag, results in all_results_by_tag.items():
        agg = _aggregate(results, tag)
        aggregates[tag] = agg
        if agg:
            mean_acc = agg.get("mean_accuracy")
            std_acc  = agg.get("std_accuracy")
            mean_f1  = agg.get("mean_f1_macro")
            std_f1   = agg.get("std_f1_macro")
            print(
                f"  {tag:12s}: "
                f"Acc = {mean_acc:.4f} ± {std_acc:.4f}  "
                f"F1 = {mean_f1:.4f} ± {std_f1:.4f}  "
                f"(n={agg['num_subjects']})"
                if (mean_acc is not None and mean_f1 is not None)
                else f"  {tag:12s}: no successful folds"
            )
        else:
            print(f"  {tag:12s}: no successful folds")

    # Delta analysis — only if both configs have results
    if "baseline" in aggregates and "struct_drop" in aggregates:
        b = aggregates["baseline"]
        s = aggregates["struct_drop"]
        if b and s:
            delta_acc = s.get("mean_accuracy", 0) - b.get("mean_accuracy", 0)
            delta_f1  = s.get("mean_f1_macro",  0) - b.get("mean_f1_macro",  0)
            print(
                f"\n  Delta (StructDrop - Baseline): "
                f"ΔAcc = {delta_acc:+.4f}  ΔF1 = {delta_f1:+.4f}"
            )

    # -----------------------------------------------------------------------
    # Save LOSO summary JSON
    # -----------------------------------------------------------------------
    loso_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_date": datetime.now().isoformat(),
        "subjects": ALL_SUBJECTS,
        "exercises": EXERCISES,
        "model": "cnn_gru_attention",
        "approach": "deep_raw",
        "hypothesis": (
            "Structured dropout (ChannelDrop + BandDrop + TemporalDrop) applied "
            "only during training forces the model to learn gesture content rather "
            "than subject-specific style, improving LOSO generalisation."
        ),
        "loso_compliance_notes": [
            "Normalisation computed from train-split of training subjects only.",
            "StructuredDropDataset is used exclusively for training split.",
            "Val/test splits always use plain WindowDataset (no augmentation).",
            "Test subject data first accessed in CrossSubjectExperiment.evaluate_numpy().",
            "Augmentation params are fixed global hyperparameters, not fit to any data.",
        ],
        "processing_config": asdict(proc_cfg),
        "split_config": asdict(split_cfg),
        "configurations": CONFIGS,
        "aggregate_results": aggregates,
        "individual_results": {
            tag: make_json_serializable(results)
            for tag, results in all_results_by_tag.items()
        },
    }

    summary_path = OUTPUT_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(loso_summary), f, indent=4, ensure_ascii=False)

    print(f"\nResults saved → {OUTPUT_DIR.resolve()}")

    # -----------------------------------------------------------------------
    # Hypothesis executor callback (optional — guarded against missing package)
    # -----------------------------------------------------------------------
    try:
        from hypothesis_executor.qdrant_callback import (
            mark_hypothesis_verified,
            mark_hypothesis_failed,
        )

        if HYPOTHESIS_ID:
            best_tag = max(
                aggregates,
                key=lambda t: aggregates[t].get("mean_accuracy", -1) if aggregates[t] else -1,
            )
            best_agg = aggregates.get(best_tag, {})
            if best_agg and best_agg.get("mean_accuracy") is not None:
                mark_hypothesis_verified(
                    hypothesis_id=HYPOTHESIS_ID,
                    metrics={**best_agg, "best_config": best_tag},
                    experiment_name=EXPERIMENT_NAME,
                )
            else:
                mark_hypothesis_failed(
                    hypothesis_id=HYPOTHESIS_ID,
                    error_message="No successful LOSO folds in any configuration.",
                )
    except ImportError:
        pass  # hypothesis_executor not installed — skip silently


if __name__ == "__main__":
    main()
