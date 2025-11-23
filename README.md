# EMG Gesture Recognition

## Project Overview

This project implements a comprehensive pipeline for EMG-based gesture recognition using the NinaPro DB2 dataset, with a special focus on analyzing the impact of sensor on classification performance. The system processes raw EMG signals, segments them by gestures, extracts windows, trains a deep learning model, and evaluates performance under simulated sensor moves.

## Current Experiment Status

### Latest Experiment (November 23, 2025)

**Objective**: Train baseline models and test the hypothesis that sensor rotation significantly impacts classification accuracy.

**Key Findings**:
- ✅ High accuracy (>90%) achieved for within-subject gesture recognition
- ✅ Confirmed hypothesis: Sensor rotation causes exponential drop in accuracy
- ✅ Validated pipeline for rotation experiments
- ✅ Established baseline performance metrics

**Experimental Results**:
- **8 sensors**: High accuracy with full sensor set
- **4 sensors**: Maintained good performance with reduced sensors  
- **2 sensors**: Surprisingly good accuracy even with minimal sensors
- **Rotation impact**: Significant performance degradation with simulated rotations

## Project Structure

```
OMEGA/
├── main.py                          # Main pipeline execution
├── config/
│   └── base.py                      # Configuration classes
├── data/
│   └── loaders.py                   # NinaPro data loader
├── processing/
│   ├── segmentation.py              # Gesture segmentation
│   ├── windowing.py                 # Window extraction
│   └── splitting.py                 # Dataset splitting
├── training/
│   ├── trainer.py                   # Model training
│   └── datasets.py                  # PyTorch datasets
├── models/
│   └── cnn1d.py                     # 1D CNN architecture
├── evaluation/
│   └── rotation.py                  # Rotation experiments
├── visualization/
│   ├── base.py                      # Main visualizer
│   └── rotation.py                  # Rotation-specific plots
├── utils/
│   ├── logging.py                   # Logging setup
│   └── artifacts.py                 # Artifact saving
└── docs/
    └── exp_23_11_2025.md            # Experiment documentation
```

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
- Scikit-learn
- CuPy (optional, for GPU acceleration)

### Basic Usage

```python
from main import main

# Run the complete pipeline
main()
```

### Configuration
The pipeline is highly configurable through dataclasses:
- `ProcessingConfig`: Window size, overlap, channel selection
- `TrainingConfig`: Model hyperparameters, training settings  
- `SplitConfig`: Train/val/test splitting strategy
- `RotationConfig`: Rotation experiment parameters

## Key Components

### Data Processing
- **NinaProLoader**: Loads and parses .mat files from NinaPro DB2
- **GestureSegmenter**: Segments continuous EMG by gesture labels
- **WindowExtractor**: Extracts overlapping windows from segments
- **DatasetSplitter**: Splits data by segments or windows

### Model Architecture
```python
SimpleCNN1D(
    in_channels=8,      # Number of EMG channels
    num_classes=10,     # Gesture classes (including rest)
    dropout=0.3         # Regularization
)
```

### Rotation Experiment
Simulates sensor bracelet rotations by permuting channel indices:
- Models real-world scenario of misaligned sensor placement
- Tests model robustness to positional changes
- Provides insights for transfer learning challenges

## Recent Results

### Baseline Performance
- **Subject**: DB2_s1
- **Exercise**: E3 (10 gestures including rest)
- **Best Accuracy**: >90% with 8 sensors
- **Rotation Impact**: Up to 50% accuracy drop with 3-position rotation

### Critical Insights
1. **Within-subject recognition** works well with simple models
2. **Sensor reduction** is possible with minimal performance loss
3. **Rotation sensitivity** is a major challenge for practical deployment
4. **Segment-based splitting** provides more realistic evaluation

## Next Steps

Based on current findings, future work will focus on:
1. **Cross-subject evaluation** testing the transfer learning hypothesis
2. **Rotation-invariant models** developing robust architectures
3. **Advanced preprocessing** addressing labeling inconsistencies

## Citation

If you use this code in your research, please reference the NinaPro dataset and consider citing relevant publications on EMG gesture recognition and sensor robustness.

---

*Last Updated: November 2025*  
*Experiment Status: Baseline established, rotation hypothesis confirmed*  
*Next Phase: Cross-subject evaluation and robustness improvements*