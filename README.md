# OMEGA — Cross-Subject sEMG Gesture Recognition

Research framework for subject-invariant hand gesture recognition from surface EMG signals, with focus on frequency-selective processing and inter-subject domain generalization.

**Key finding**: subject invariance is better achieved through frequency-selective processing (preserving low frequencies, normalizing/randomizing high frequencies) than through global invariance methods.

## Results

- **112 experiments** covering CNN, RNN, Transformer, SVM, disentanglement, style mixing, self-supervised learning, and more
- **Best accuracy**: 37.7% macro-F1 on 5-subject LOSO (NinaPro DB2, 49 gestures + rest)
- **+4.2 p.p.** relative to the best baseline (Random Forest, 32.0%)
- Best approaches: per-band MixStyle with Sinc/UVMD decomposition (SCG-Net family)

## Dataset

[NinaPro DB2](http://ninapro.hevs.ch/) — 40 subjects, 49 hand gestures + rest, 12 sEMG channels (8 used), 2 kHz sampling rate.

## Project Structure

```
OMEGA/
├── main.py                     # Single-subject pipeline
├── main_cross_subject.py       # Cross-subject LOSO pipeline
├── config/                     # Configuration (base, cross-subject)
├── data/
│   ├── loaders.py              # NinaPro .mat file loader
│   └── multi_subject_loader.py # Multi-subject data loader
├── processing/
│   ├── segmentation.py         # Gesture segmentation (CPU/GPU)
│   ├── windowing.py            # Sliding window extraction
│   ├── splitting.py            # Train/val/test splitting
│   ├── features.py             # Handcrafted time-domain features
│   └── powerful_features.py    # Extended feature extractor
├── models/                     # 80+ model architectures
│   ├── cnn1d.py                # SimpleCNN1D baseline
│   ├── sinc_pcen_cnn_gru.py    # Sinc filterbank + PCEN
│   ├── uvmd_classifier.py      # UVMD decomposition
│   ├── mixstyle_disentangled_cnn_gru.py
│   ├── ecapa_tdnn_emg.py       # ECAPA-TDNN adapted for EMG
│   └── ...
├── training/
│   ├── trainer.py              # Base WindowClassifierTrainer
│   ├── datasets.py             # PyTorch datasets
│   └── ...                     # 37 specialized trainers
├── experiments/                # 112+ experiment scripts
│   ├── exp_1_deep_raw_cnn_loso.py
│   ├── h1_spectral_analysis_intersubject_variability.py
│   ├── h6_unified_ablation_loso.py
│   ├── h7_uvmd_mixstyle_loso.py
│   └── ...
├── evaluation/                 # Cross-subject & rotation evaluation
├── visualization/              # Plotting and paper figures
├── scripts/                    # Analysis, deployment, statistical tests
├── hypothesis_executor/        # LLM-based experiment generation
├── research_agent/             # Multi-agent research automation
└── docs/                       # Research logs and reports
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data Setup

Download NinaPro DB2 and place subject directories as `data/DB2_sN/`:

```bash
data/
├── DB2_s1/
│   ├── DB2_s1_e1.mat
│   ├── DB2_s1_e2.mat
│   └── DB2_s1_e3.mat
├── DB2_s2/
└── ...
```

### Run Experiments

```bash
# Single subject pipeline
python main.py

# Cross-subject LOSO evaluation
python main_cross_subject.py

# Run a specific experiment (e.g., UVMD + MixStyle)
python experiments/h7_uvmd_mixstyle_loso.py --ci 1

# Run on server with specific subjects
python experiments/h7_uvmd_mixstyle_loso.py --subjects DB2_s1,DB2_s12,DB2_s15,DB2_s28,DB2_s39
```

## Research Hypotheses

| ID | Hypothesis | Status |
|----|-----------|--------|
| H1 | Inter-subject variability is frequency-dependent | Confirmed |
| H2 | Frequency decomposition (Sinc/UVMD) improves cross-subject accuracy | Confirmed |
| H3 | Per-band style normalization (MixStyle/IN) is effective | Confirmed |
| H4 | Content-style disentanglement adds value | Rejected |
| H6 | Ablation: each SCG-Net component contributes | Confirmed |
| H7 | UVMD + per-band MixStyle is the best combination | Confirmed |
| H8 | Adaptive MixStyle strength improves over fixed | Partially confirmed |

## Key Technologies

- **Sinc filterbank**: learnable bandpass filters for frequency decomposition
- **UVMD (Unfolded VMD)**: differentiable variational mode decomposition
- **Per-band MixStyle**: frequency-selective style augmentation
- **ECAPA-TDNN**: speaker verification architecture adapted for EMG
- **LOSO cross-validation**: Leave-One-Subject-Out for unbiased evaluation

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- NumPy, SciPy, scikit-learn
- matplotlib, seaborn
- h5py (for NinaPro .mat files)
- CuPy (optional, GPU-accelerated segmentation)

## Infrastructure

Experiments are designed to run on GPU servers (tested on vast.ai). Deployment via:

```bash
make deploy    # rsync to server
make run       # run experiment remotely
```

## License

Research code. If you use this work, please cite the NinaPro dataset and relevant publications on cross-subject EMG recognition.

---

*Last updated: March 2026*
