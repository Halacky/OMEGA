"""
Experiment 112: Channel-Band Factorized Representation with Temporal Consensus
(Hypothesis 5: Tucker Decomposition + Multi-Resolution Agreement)

Motivation:
  Previous approaches disentangle into 2 factors (content/style). But EMG
  windows have THREE axes of variability:
    (1) Channel — electrode placement, inter-electrode distance (subject-specific)
    (2) Frequency — MU recruitment patterns (exercise- and subject-specific)
    (3) Temporal — gesture dynamics (should be gesture-specific, NOT subject-specific)

  Subject variability manifests differently along each axis, so a factorized
  representation with per-axis invariance is better than a single shared/specific split.

Architecture (ChannelBandTuckerConsensusEMG):
  1. STFT per channel → log-magnitude spectrogram (B, C, F, T')
     n_fft=64, hop=16 → F=33 freq bins, T'=34 temporal frames for T=600
  2. Soft Freq AGC: per-freq temporal mean removal + learnable scale γ_f
     (reduces subject-specific amplitude per frequency band)
  3. Tucker Channel Factor U_ch: linear C → r_c projection
     → h_ch (B, r_c, F, T') — subject adversary via GRL
  4. Tucker Freq Factor U_f: linear F → r_f projection
     → h_cf (B, r_c, r_f, T') — Tucker core tensor
  5. Temporal Consensus: classify full T' + individual T'//4 quarters;
     KL(logits_full || logits_quarter) regularizes temporal robustness
  6. Classifier: h_cf.mean(-1) → flatten → LayerNorm → MLP → logits

Loss:
    L_total = L_gesture + λ_adv * L_subject_adv + λ_cons * L_temporal_cons

LOSO Protocol:
  - Each fold: train on N-1 subjects, test on 1 held-out subject.
  - Cannot use CrossSubjectExperiment.run() because it loses subject identity
    needed for GRL. Instead, builds splits with subject provenance manually.
  - Subject labels used ONLY in the training loop, never at val/test.
  - Channel standardisation computed from training fold only.
  - model.eval() freezes BatchNorm — no test-subject stats.
  - No test-time adaptation, no subject-specific layers.
  - Test subject windows never appear in train or val splits.

Expected outcome: F1 > 35.5% (target from hypothesis statement).
Baseline references: exp_31 (35.28%), exp_109 (MATE-Kronecker).
"""

import sys
import json
import gc
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_X_template_loso import (
    CI_TEST_SUBJECTS,
    parse_subjects_args,
    make_json_serializable,
)
from config.base import ProcessingConfig, SplitConfig, TrainingConfig
from data.multi_subject_loader import MultiSubjectLoader
from training.tucker_consensus_trainer import TuckerConsensusTrainer
from visualization.base import Visualizer
from utils.logging import setup_logging, seed_everything
from utils.artifacts import ArtifactSaver


# ========== EXPERIMENT SETTINGS ===========================================

EXPERIMENT_NAME = "exp_112_channel_band_tucker_consensus"
APPROACH        = "deep_raw"
EXERCISES       = ["E1", "E2"]
USE_IMPROVED_PROCESSING = True
MAX_GESTURES    = 10

# Tucker factorization ranks
R_C = 8    # channel rank  (C=8 input → r_c=8: retain all channel info, factorize)
R_F = 16   # frequency rank (F=33 bins → r_f=16: ~half the freq bins retained)

# STFT parameters
N_FFT      = 64   # window size → F = 33 freq bins
HOP_LENGTH = 16   # hop → T' = (600-64)//16+1 = 34 temporal frames

# Classifier
HIDDEN_DIM = 128  # core_dim = r_c * r_f = 8*16 = 128 → hidden_dim = 128

# Loss weights
LAMBDA_ADV  = 0.3   # adversarial subject loss (GRL on channel Tucker factors)
LAMBDA_CONS = 0.1   # temporal consensus KL loss


# ========== SPLIT BUILDER WITH SUBJECT PROVENANCE =========================

def _build_splits_with_subject_labels(
    subjects_data: Dict,
    train_subjects: List[str],
    test_subject: str,
    common_gestures: List[int],
    multi_loader: MultiSubjectLoader,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Build train/val/test splits preserving subject provenance for the GRL.

    Unlike CrossSubjectExperiment._prepare_splits(), this function tracks
    which training subject contributed each window. The subject index is
    required for the adversarial subject loss.

    LOSO safety:
      - test_subject windows go ONLY to the test split — never in train/val.
      - train/val split: fixed per-gesture permutation from seed — reproducible.
      - Subject labels created ONLY for train split (not val, not test).

    Returns:
        "train":                Dict[gesture_id → np.ndarray (N, T, C)]
        "val":                  Dict[gesture_id → np.ndarray (N, T, C)]
        "test":                 Dict[gesture_id → np.ndarray (N, T, C)]
        "train_subject_labels": Dict[gesture_id → np.ndarray (N,)]  subject indices
        "num_train_subjects":   int
    """
    rng = np.random.RandomState(seed)
    train_subj_to_idx = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    num_train_subjects = len(train_subjects)

    # ── Collect all training windows and subject labels per gesture ────────
    raw_train: Dict[int, np.ndarray]  = {}
    raw_subj:  Dict[int, np.ndarray]  = {}

    for gid in common_gestures:
        win_parts  = []
        subj_parts = []
        for sid in sorted(train_subjects):
            if sid not in subjects_data:
                continue
            _, _, grouped_windows = subjects_data[sid]
            filtered = multi_loader.filter_by_gestures(grouped_windows, [gid])
            if gid not in filtered:
                continue
            for rep_arr in filtered[gid]:
                if isinstance(rep_arr, np.ndarray) and len(rep_arr) > 0:
                    win_parts.append(rep_arr)
                    subj_parts.append(
                        np.full(len(rep_arr), train_subj_to_idx[sid], dtype=np.int64)
                    )
        if win_parts:
            raw_train[gid] = np.concatenate(win_parts,  axis=0)
            raw_subj[gid]  = np.concatenate(subj_parts, axis=0)

    # ── Split collected data into train / val per gesture ─────────────────
    # Windows and subject labels are permuted together — alignment maintained.
    final_train: Dict[int, np.ndarray] = {}
    final_val:   Dict[int, np.ndarray] = {}
    final_subj:  Dict[int, np.ndarray] = {}

    for gid in common_gestures:
        if gid not in raw_train:
            continue
        X_gid = raw_train[gid]
        S_gid = raw_subj[gid]
        n     = len(X_gid)
        perm  = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))

        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        if len(train_idx) == 0:
            # Edge case: too few windows — put everything in train
            train_idx = perm
            val_idx   = perm[:0]

        final_train[gid] = X_gid[train_idx]
        final_val[gid]   = X_gid[val_idx]
        final_subj[gid]  = S_gid[train_idx]

    # ── Build test split from held-out subject only ────────────────────────
    # No subject labels for test — subject information never used at inference.
    final_test: Dict[int, np.ndarray] = {}
    if test_subject in subjects_data:
        _, _, test_gw = subjects_data[test_subject]
        test_filtered = multi_loader.filter_by_gestures(test_gw, common_gestures)
        for gid, reps in test_filtered.items():
            valid_reps = [
                r for r in reps if isinstance(r, np.ndarray) and len(r) > 0
            ]
            if valid_reps:
                final_test[gid] = np.concatenate(valid_reps, axis=0)

    return {
        "train":                final_train,
        "val":                  final_val,
        "test":                 final_test,
        "train_subject_labels": final_subj,
        "num_train_subjects":   num_train_subjects,
    }


# ========== SINGLE LOSO FOLD ==============================================

def run_single_loso_fold(
    base_dir: Path,
    output_dir: Path,
    train_subjects: List[str],
    test_subject: str,
    exercises: List[str],
    proc_cfg: ProcessingConfig,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
) -> Dict:
    """
    Execute one LOSO fold.

    LOSO invariants enforced here:
      - test_subject data goes only to the test split.
      - Channel standardisation is computed inside the trainer from training
        windows — this function never inspects data statistics.
      - Subject labels injected only into the train portion of the splits dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(train_cfg.seed, verbose=False)

    train_cfg.pipeline_type = APPROACH
    train_cfg.model_type    = "tucker_consensus"

    proc_cfg.save(output_dir / "processing_config.json")
    train_cfg.save(output_dir / "training_config.json")
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(asdict(split_cfg), f, indent=4)

    # ── Data loading ───────────────────────────────────────────────────────
    multi_loader = MultiSubjectLoader(
        processing_config=proc_cfg,
        logger=logger,
        use_gpu=True,
        use_improved_processing=USE_IMPROVED_PROCESSING,
    )
    base_viz = Visualizer(output_dir, logger)

    all_subject_ids = list(dict.fromkeys(train_subjects + [test_subject]))
    subjects_data   = multi_loader.load_multiple_subjects(
        base_dir=base_dir,
        subject_ids=all_subject_ids,
        exercises=exercises,
        include_rest=split_cfg.include_rest_in_splits,
    )

    common_gestures = multi_loader.get_common_gestures(
        subjects_data, max_gestures=MAX_GESTURES,
    )
    logger.info(f"Common gestures ({len(common_gestures)}): {common_gestures}")

    # ── Build splits with subject provenance ───────────────────────────────
    splits = _build_splits_with_subject_labels(
        subjects_data=subjects_data,
        train_subjects=train_subjects,
        test_subject=test_subject,
        common_gestures=common_gestures,
        multi_loader=multi_loader,
        val_ratio=split_cfg.val_ratio,
        seed=train_cfg.seed,
    )

    for split_name in ["train", "val", "test"]:
        total = sum(
            len(arr) for arr in splits[split_name].values()
            if isinstance(arr, np.ndarray) and arr.ndim >= 2
        )
        logger.info(
            f"{split_name.upper()}: {total} windows, "
            f"{len(splits[split_name])} gestures"
        )

    # ── Build and train ────────────────────────────────────────────────────
    trainer = TuckerConsensusTrainer(
        train_cfg   = train_cfg,
        logger      = logger,
        output_dir  = output_dir,
        visualizer  = base_viz,
        lambda_adv  = LAMBDA_ADV,
        lambda_cons = LAMBDA_CONS,
        r_c         = R_C,
        r_f         = R_F,
        n_fft       = N_FFT,
        hop_length  = HOP_LENGTH,
        hidden_dim  = HIDDEN_DIM,
    )

    try:
        training_results = trainer.fit(splits)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return {
            "test_subject":  test_subject,
            "model_type":    "tucker_consensus",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         str(e),
        }

    # ── Evaluate on held-out test subject ──────────────────────────────────
    # Assemble test arrays in class_ids order (same order trainer used for y_train).
    class_ids = trainer.class_ids
    X_test_list, y_test_list = [], []
    for i, gid in enumerate(class_ids):
        if gid in splits["test"]:
            arr = splits["test"][gid]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and len(arr) > 0:
                X_test_list.append(arr)
                y_test_list.append(np.full(len(arr), i, dtype=np.int64))

    if not X_test_list:
        logger.error("No test windows for held-out subject.")
        return {
            "test_subject":  test_subject,
            "model_type":    "tucker_consensus",
            "approach":      APPROACH,
            "test_accuracy": None,
            "test_f1_macro": None,
            "error":         "No test data",
        }

    X_test_concat = np.concatenate(X_test_list, axis=0)
    y_test_concat = np.concatenate(y_test_list, axis=0)

    # evaluate_numpy handles (N, T, C) → (N, C, T) transpose and standardisation
    # internally using training-fold mean_c / std_c (no test-subject leakage).
    test_results = trainer.evaluate_numpy(
        X_test_concat,
        y_test_concat,
        split_name=f"cross_subject_test_{test_subject}",
        visualize=True,
    )

    test_acc = float(test_results["accuracy"])
    test_f1  = float(test_results["f1_macro"])

    print(
        f"[LOSO] Test subject {test_subject} | "
        f"Accuracy={test_acc:.4f}, F1-macro={test_f1:.4f}"
    )

    # ── Save per-fold results ──────────────────────────────────────────────
    fold_results = {
        "test_subject":    test_subject,
        "train_subjects":  train_subjects,
        "common_gestures": common_gestures,
        "training":        training_results,
        "cross_subject_test": {
            "subject":          test_subject,
            "accuracy":         test_acc,
            "f1_macro":         test_f1,
            "report":           test_results.get("report"),
            "confusion_matrix": test_results.get("confusion_matrix"),
        },
    }
    with open(output_dir / "cross_subject_results.json", "w") as f:
        json.dump(make_json_serializable(fold_results), f, indent=4, ensure_ascii=False)

    saver  = ArtifactSaver(output_dir, logger)
    n_freq = N_FFT // 2 + 1
    t_prime = (proc_cfg.window_size - N_FFT) // HOP_LENGTH + 1
    meta   = {
        "test_subject":  test_subject,
        "train_subjects": train_subjects,
        "model_type":    "tucker_consensus",
        "approach":      APPROACH,
        "exercises":     exercises,
        "architecture": {
            "r_c":         R_C,
            "r_f":         R_F,
            "n_fft":       N_FFT,
            "hop_length":  HOP_LENGTH,
            "hidden_dim":  HIDDEN_DIM,
            "core_dim":    R_C * R_F,
            "n_freq_bins": n_freq,
            "t_prime":     t_prime,
        },
        "loss_config": {
            "lambda_adv":    LAMBDA_ADV,
            "lambda_cons":   LAMBDA_CONS,
            "grl_schedule":  "DANN: alpha=2/(1+exp(-10*p))-1",
            "cons_schedule": "quarter-window KL vs full-window (τ=2)",
        },
        "metrics": {
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
        },
    }
    saver.save_metadata(make_json_serializable(meta), filename="fold_metadata.json")

    # ── Cleanup ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del trainer, multi_loader, base_viz, subjects_data
    gc.collect()

    return {
        "test_subject":  test_subject,
        "model_type":    "tucker_consensus",
        "approach":      APPROACH,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }


# ========== MAIN ==========================================================

def main():
    # parse_subjects_args() defaults to CI_TEST_SUBJECTS (5 subjects).
    # Server (vast.ai) has symlinks only for CI subjects — NEVER default to
    # the 20-subject full list.
    ALL_SUBJECTS = parse_subjects_args()

    BASE_DIR    = ROOT / "data"
    TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = (
        ROOT / "experiments_output" / f"{EXPERIMENT_NAME}_{TIMESTAMP}"
    )

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
        model_type="tucker_consensus",
        pipeline_type=APPROACH,
        use_handcrafted_features=False,
        batch_size=64,
        epochs=80,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=15,
        seed=42,
        use_class_weights=True,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    n_freq  = N_FFT // 2 + 1
    t_prime = (proc_cfg.window_size - N_FFT) // HOP_LENGTH + 1
    core_dim = R_C * R_F

    print(f"{'=' * 80}")
    print(f"Experiment:  {EXPERIMENT_NAME}")
    print(f"Hypothesis:  Channel-Band Tucker Decomposition + Temporal Consensus")
    print(f"  STFT:      n_fft={N_FFT}, hop={HOP_LENGTH} → F={n_freq}, T'={t_prime}")
    print(f"  Tucker:    U_ch: C→{R_C},  U_f: F→{R_F}")
    print(f"  Core dim:  r_c × r_f = {R_C} × {R_F} = {core_dim}")
    print(f"  Axis reg.: (1) Channel: GRL adversary (λ_adv={LAMBDA_ADV})")
    print(f"             (2) Frequency: Soft AGC per-freq normalization")
    print(f"             (3) Temporal: quarter-window KL (λ_cons={LAMBDA_CONS})")
    print(f"  Loss:      L_gesture + {LAMBDA_ADV}·L_subject_adv + {LAMBDA_CONS}·L_cons")
    print(f"Subjects:    {ALL_SUBJECTS}")
    print(f"Exercises:   {EXERCISES}")
    print(f"Output:      {OUTPUT_ROOT}")
    print(f"{'=' * 80}")

    all_loso_results = []

    for test_subject in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != test_subject]
        fold_output    = (
            OUTPUT_ROOT / "tucker_consensus" / f"test_{test_subject}"
        )

        result = run_single_loso_fold(
            base_dir=BASE_DIR,
            output_dir=fold_output,
            train_subjects=train_subjects,
            test_subject=test_subject,
            exercises=EXERCISES,
            proc_cfg=proc_cfg,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
        )
        all_loso_results.append(result)

    # ── Aggregate results ──────────────────────────────────────────────────
    valid = [r for r in all_loso_results if r.get("test_accuracy") is not None]
    if valid:
        accs = [r["test_accuracy"] for r in valid]
        f1s  = [r["test_f1_macro"] for r in valid]
        print(f"\n{'=' * 60}")
        print(f"Tucker Consensus — LOSO Summary ({len(valid)} folds)")
        print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"  Baseline (exp_31): 35.28% F1 — target: >35.5%")
        print(f"{'=' * 60}\n")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": (
            "Three-axis Tucker factorization of EMG spectrogram: "
            "channel (GRL adversary), frequency (Soft AGC), "
            "temporal (quarter-window KL consensus)."
        ),
        "timestamp": TIMESTAMP,
        "subjects":  ALL_SUBJECTS,
        "approach":  APPROACH,
        "architecture": {
            "r_c":        R_C,
            "r_f":        R_F,
            "n_fft":      N_FFT,
            "hop_length": HOP_LENGTH,
            "hidden_dim": HIDDEN_DIM,
            "core_dim":   core_dim,
            "n_freq_bins": n_freq,
            "t_prime":    t_prime,
        },
        "loss_config": {
            "lambda_adv":  LAMBDA_ADV,
            "lambda_cons": LAMBDA_CONS,
            "grl_schedule": "DANN: alpha=2/(1+exp(-10*p))-1",
        },
        "results": all_loso_results,
    }

    if valid:
        summary["aggregate"] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
            "num_folds":     len(valid),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "loso_summary.json", "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4, ensure_ascii=False)
    print(f"Summary saved: {OUTPUT_ROOT / 'loso_summary.json'}")

    # ── Report to hypothesis_executor if available ─────────────────────────
    try:
        from hypothesis_executor import mark_hypothesis_verified, mark_hypothesis_failed
        if valid:
            metrics = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro":  float(np.std(f1s)),
            }
            mark_hypothesis_verified(
                "H_tucker_consensus",
                metrics,
                experiment_name=EXPERIMENT_NAME,
            )
        else:
            mark_hypothesis_failed(
                "H_tucker_consensus",
                "All LOSO folds failed",
            )
    except ImportError:
        pass


if __name__ == "__main__":
    main()
