# OMEGA Full Research Report

**Generated**: 2026-03-18 19:56

Total experiments analyzed: 150

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Baselines](#baselines)
3. [Detailed Experiment Reports](#detailed-experiment-reports)
4. [Cross-Experiment Analysis](#cross-experiment-analysis)
5. [Conclusions](#conclusions)

---

## Executive Summary

This report covers 150 experiments testing various approaches to cross-subject EMG gesture recognition on NinaPro DB2.

**Best result**: 50.64% accuracy (multi_res_aligned_disentangle, exp_110_multi_res_aligned_disentangle)

**27/105** experiments exceeded the best baseline (32.0%).

---

## Baselines

| Model | Accuracy | F1 |
|-------|----------|-----|
| rf | 32.00% | 30.33% |
| cnn_gru_attention | 30.85% | 28.19% |
| svm_rbf | 27.82% | 26.63% |
| simple_cnn | 25.99% | 24.11% |
| svm_linear | 25.51% | 24.43% |

---

## Detailed Experiment Reports

### EXP_0: h2_ablation_decomposition

- **Date**: 20260308_080619
- **Approach**: ``
- **Subjects**: 20 (DB2_s1, DB2_s11, DB2_s12, DB2_s13, DB2_s14...)

**Results:**

---

### EXP_0: h3_style_normalization

- **Date**: 20260308_203852
- **Approach**: ``
- **Subjects**: 20 (DB2_s1, DB2_s11, DB2_s12, DB2_s13, DB2_s14...)

**Results:**

---

### EXP_0: h4_content_style_disentanglement

- **Date**: 2026-03-09T15:43:35.563551
- **Approach**: ``
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)

**Results:**

---

### EXP_0: h5_integrated_system

- **Date**: 2026-03-12T04:55:39.256148
- **Approach**: ``
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)

**Results:**

---

### EXP_0: test_bigru_spec

- **Date**: 2026-02-16T13:31:32.681915
- **Approach**: `deep_raw`
- **Subjects**: 2 (DB2_s2, DB2_s3)

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bigru | 12.20% | 10.70% | 0.0055 | 2 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s2 | bigru | 12.75% | 12.07% |
| DB2_s3 | bigru | 11.66% | 9.34% |

**vs Baseline**: BELOW best baseline by -19.8pp

---

### EXP_1: exp1_deep_raw_attention_cnn_loso_1_12_15_28_39

- **Date**: 2026-02-19T11:41:35.022065
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: attention_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| attention_cnn | 25.31% | 23.28% | 0.0324 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | attention_cnn | 26.52% | 25.34% |
| DB2_s12 | attention_cnn | 20.04% | 16.71% |
| DB2_s15 | attention_cnn | 24.35% | 22.12% |
| DB2_s28 | attention_cnn | 30.04% | 28.17% |
| DB2_s39 | attention_cnn | 25.61% | 24.06% |

**vs Baseline**: BELOW best baseline by -6.7pp

---

### EXP_1: exp1_deep_raw_attention_cnn_loso_isolated

- **Date**: 2026-01-18T16:16:52.008352
- **Approach**: `deep_raw`
- **Subjects**: 40 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: attention_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| attention_cnn | 21.68% | 18.30% | 0.0579 | 40 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | attention_cnn | 32.68% | 30.52% |
| DB2_s2 | attention_cnn | 17.65% | 13.19% |
| DB2_s3 | attention_cnn | 17.44% | 13.47% |
| DB2_s4 | attention_cnn | 21.47% | 18.00% |
| DB2_s5 | attention_cnn | 14.22% | 10.10% |
| DB2_s6 | attention_cnn | 16.40% | 12.71% |
| DB2_s7 | attention_cnn | 13.12% | 8.03% |
| DB2_s8 | attention_cnn | 27.11% | 20.46% |
| DB2_s9 | attention_cnn | 27.65% | 23.58% |
| DB2_s10 | attention_cnn | 23.78% | 22.14% |
| DB2_s11 | attention_cnn | 31.06% | 27.46% |
| DB2_s12 | attention_cnn | 16.04% | 13.12% |
| DB2_s13 | attention_cnn | 25.52% | 21.00% |
| DB2_s14 | attention_cnn | 18.38% | 16.05% |
| DB2_s15 | attention_cnn | 27.33% | 23.99% |
| DB2_s16 | attention_cnn | 6.72% | 4.59% |
| DB2_s17 | attention_cnn | 19.05% | 14.15% |
| DB2_s18 | attention_cnn | 23.88% | 23.67% |
| DB2_s19 | attention_cnn | 26.51% | 22.08% |
| DB2_s20 | attention_cnn | 17.15% | 13.33% |
| DB2_s21 | attention_cnn | 22.31% | 19.09% |
| DB2_s22 | attention_cnn | 19.33% | 17.85% |
| DB2_s23 | attention_cnn | 23.34% | 20.65% |
| DB2_s24 | attention_cnn | 22.87% | 24.40% |
| DB2_s25 | attention_cnn | 31.23% | 25.92% |
| DB2_s26 | attention_cnn | 19.82% | 14.40% |
| DB2_s27 | attention_cnn | 14.95% | 11.03% |
| DB2_s28 | attention_cnn | 24.64% | 21.14% |
| DB2_s29 | attention_cnn | 24.92% | 20.92% |
| DB2_s30 | attention_cnn | 27.64% | 23.96% |
| DB2_s31 | attention_cnn | 17.36% | 12.12% |
| DB2_s32 | attention_cnn | 19.57% | 17.88% |
| DB2_s33 | attention_cnn | 25.91% | 22.36% |
| DB2_s34 | attention_cnn | 25.63% | 22.09% |
| DB2_s35 | attention_cnn | 24.27% | 23.64% |
| DB2_s36 | attention_cnn | 29.60% | 26.69% |
| DB2_s37 | attention_cnn | 8.85% | 6.31% |
| DB2_s38 | attention_cnn | 21.38% | 18.02% |
| DB2_s39 | attention_cnn | 18.95% | 14.35% |
| DB2_s40 | attention_cnn | 21.71% | 17.60% |

**vs Baseline**: BELOW best baseline by -10.3pp

---

### EXP_1: exp1_deep_raw_attention_cnn_loso_isolated_v2

- **Date**: 2026-01-18T17:22:35.429929
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: attention_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| attention_cnn | 27.33% | 23.70% | 0.0641 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | attention_cnn | 29.61% | 26.63% |
| DB2_s2 | attention_cnn | 29.50% | 21.14% |
| DB2_s3 | attention_cnn | 16.90% | 14.84% |
| DB2_s4 | attention_cnn | 29.68% | 26.79% |
| DB2_s5 | attention_cnn | 28.17% | 22.61% |
| DB2_s11 | attention_cnn | 42.86% | 41.87% |
| DB2_s12 | attention_cnn | 24.26% | 21.86% |
| DB2_s13 | attention_cnn | 26.29% | 24.07% |
| DB2_s14 | attention_cnn | 26.98% | 21.91% |
| DB2_s15 | attention_cnn | 19.76% | 14.86% |
| DB2_s26 | attention_cnn | 24.68% | 19.51% |
| DB2_s27 | attention_cnn | 25.64% | 20.83% |
| DB2_s28 | attention_cnn | 31.01% | 28.07% |
| DB2_s29 | attention_cnn | 27.65% | 23.79% |
| DB2_s30 | attention_cnn | 31.72% | 27.22% |
| DB2_s36 | attention_cnn | 34.78% | 32.50% |
| DB2_s37 | attention_cnn | 12.66% | 8.30% |
| DB2_s38 | attention_cnn | 24.65% | 23.63% |
| DB2_s39 | attention_cnn | 24.76% | 19.89% |
| DB2_s40 | attention_cnn | 35.15% | 33.69% |

**vs Baseline**: BELOW best baseline by -4.7pp

---

### EXP_1: exp1_deep_raw_bigru_loso_isolated_v2

- **Date**: 2026-01-19T16:37:30.935033
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: bigru
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bigru | 28.60% | 25.43% | 0.0663 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bigru | 29.67% | 26.84% |
| DB2_s2 | bigru | 27.35% | 21.56% |
| DB2_s3 | bigru | 24.60% | 20.99% |
| DB2_s4 | bigru | 29.56% | 24.63% |
| DB2_s5 | bigru | 30.08% | 27.02% |
| DB2_s11 | bigru | 41.43% | 37.69% |
| DB2_s12 | bigru | 23.13% | 22.62% |
| DB2_s13 | bigru | 25.70% | 21.59% |
| DB2_s14 | bigru | 24.54% | 19.82% |
| DB2_s15 | bigru | 18.93% | 15.93% |
| DB2_s26 | bigru | 31.32% | 26.54% |
| DB2_s27 | bigru | 22.78% | 20.67% |
| DB2_s28 | bigru | 34.68% | 33.75% |
| DB2_s29 | bigru | 29.29% | 26.93% |
| DB2_s30 | bigru | 37.66% | 34.47% |
| DB2_s36 | bigru | 38.81% | 37.53% |
| DB2_s37 | bigru | 13.15% | 8.16% |
| DB2_s38 | bigru | 27.39% | 25.64% |
| DB2_s39 | bigru | 27.13% | 24.18% |
| DB2_s40 | bigru | 34.72% | 32.05% |

**vs Baseline**: BELOW best baseline by -3.4pp

---

### EXP_1: exp1_deep_raw_bilstm_attention_loso

- **Date**: 2026-02-20T02:08:41.067220
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: bilstm_attention
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bilstm_attention | 26.20% | 23.22% | 0.0467 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bilstm_attention | 31.15% | 25.88% |
| DB2_s12 | bilstm_attention | 19.80% | 18.28% |
| DB2_s15 | bilstm_attention | 22.14% | 21.11% |
| DB2_s28 | bilstm_attention | 31.38% | 27.22% |
| DB2_s39 | bilstm_attention | 26.52% | 23.63% |

**vs Baseline**: BELOW best baseline by -5.8pp

---

### EXP_1: exp1_deep_raw_bilstm_attention_loso_isolated_v2

- **Date**: 2026-01-19T11:06:44.786427
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: bilstm_attention
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bilstm_attention | 28.10% | 24.88% | 0.0622 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bilstm_attention | 31.57% | 26.41% |
| DB2_s2 | bilstm_attention | 30.69% | 24.29% |
| DB2_s3 | bilstm_attention | 23.16% | 21.29% |
| DB2_s4 | bilstm_attention | 28.13% | 24.76% |
| DB2_s5 | bilstm_attention | 29.96% | 25.10% |
| DB2_s11 | bilstm_attention | 38.10% | 35.12% |
| DB2_s12 | bilstm_attention | 22.59% | 21.30% |
| DB2_s13 | bilstm_attention | 27.31% | 23.51% |
| DB2_s14 | bilstm_attention | 25.07% | 20.64% |
| DB2_s15 | bilstm_attention | 21.07% | 19.02% |
| DB2_s26 | bilstm_attention | 28.74% | 24.86% |
| DB2_s27 | bilstm_attention | 28.32% | 24.75% |
| DB2_s28 | bilstm_attention | 36.81% | 34.20% |
| DB2_s29 | bilstm_attention | 21.19% | 17.67% |
| DB2_s30 | bilstm_attention | 37.78% | 34.54% |
| DB2_s36 | bilstm_attention | 36.86% | 35.28% |
| DB2_s37 | bilstm_attention | 13.15% | 9.04% |
| DB2_s38 | bilstm_attention | 25.14% | 23.59% |
| DB2_s39 | bilstm_attention | 25.85% | 23.48% |
| DB2_s40 | bilstm_attention | 30.43% | 28.70% |

**vs Baseline**: BELOW best baseline by -3.9pp

---

### EXP_1: exp1_deep_raw_bilstm_loso_isolated_v2

- **Date**: 2026-01-19T07:05:49.849361
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: bilstm
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bilstm | 19.32% | 13.97% | 0.0915 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bilstm | 18.37% | 10.54% |
| DB2_s2 | bilstm | 10.01% | 1.82% |
| DB2_s3 | bilstm | 19.88% | 17.63% |
| DB2_s4 | bilstm | 26.64% | 20.78% |
| DB2_s5 | bilstm | 11.97% | 6.09% |
| DB2_s11 | bilstm | 38.45% | 36.24% |
| DB2_s12 | bilstm | 9.99% | 1.82% |
| DB2_s13 | bilstm | 21.00% | 14.82% |
| DB2_s14 | bilstm | 9.98% | 1.82% |
| DB2_s15 | bilstm | 14.64% | 8.56% |
| DB2_s26 | bilstm | 24.92% | 20.22% |
| DB2_s27 | bilstm | 20.22% | 15.56% |
| DB2_s28 | bilstm | 33.03% | 32.04% |
| DB2_s29 | bilstm | 11.08% | 3.34% |
| DB2_s30 | bilstm | 15.04% | 8.04% |
| DB2_s36 | bilstm | 33.19% | 30.85% |
| DB2_s37 | bilstm | 10.08% | 7.56% |
| DB2_s38 | bilstm | 9.82% | 3.03% |
| DB2_s39 | bilstm | 14.09% | 8.01% |
| DB2_s40 | bilstm | 33.93% | 30.54% |

**vs Baseline**: BELOW best baseline by -12.7pp

---

### EXP_1: exp1_deep_raw_cnn_gru_attention_loso_isolated_v2

- **Date**: 2026-01-19T18:45:18.828320
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: cnn_gru_attention
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_gru_attention | 30.85% | 28.19% | 0.0645 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention | 32.10% | 30.42% |
| DB2_s2 | cnn_gru_attention | 33.13% | 26.74% |
| DB2_s3 | cnn_gru_attention | 23.46% | 21.21% |
| DB2_s4 | cnn_gru_attention | 28.19% | 25.29% |
| DB2_s5 | cnn_gru_attention | 27.75% | 23.83% |
| DB2_s11 | cnn_gru_attention | 43.69% | 40.87% |
| DB2_s12 | cnn_gru_attention | 24.79% | 22.55% |
| DB2_s13 | cnn_gru_attention | 30.93% | 27.24% |
| DB2_s14 | cnn_gru_attention | 27.39% | 22.73% |
| DB2_s15 | cnn_gru_attention | 27.98% | 25.75% |
| DB2_s26 | cnn_gru_attention | 29.23% | 25.85% |
| DB2_s27 | cnn_gru_attention | 29.90% | 28.19% |
| DB2_s28 | cnn_gru_attention | 42.61% | 41.32% |
| DB2_s29 | cnn_gru_attention | 28.01% | 26.95% |
| DB2_s30 | cnn_gru_attention | 38.99% | 36.04% |
| DB2_s36 | cnn_gru_attention | 38.75% | 37.28% |
| DB2_s37 | cnn_gru_attention | 15.30% | 12.35% |
| DB2_s38 | cnn_gru_attention | 32.82% | 30.55% |
| DB2_s39 | cnn_gru_attention | 29.09% | 25.86% |
| DB2_s40 | cnn_gru_attention | 32.76% | 32.74% |

**vs Baseline**: BELOW best baseline by -1.2pp

---

### EXP_1: exp1_deep_raw_cnn_loso_isolated

- **Date**: 2026-01-18T04:31:02.942825
- **Approach**: `deep_raw`
- **Subjects**: 40 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| simple_cnn | 23.14% | 19.06% | 0.0540 | 40 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | simple_cnn | 27.30% | 25.81% |
| DB2_s2 | simple_cnn | 17.02% | 10.93% |
| DB2_s3 | simple_cnn | 20.39% | 15.88% |
| DB2_s4 | simple_cnn | 22.24% | 19.07% |
| DB2_s5 | simple_cnn | 18.21% | 16.24% |
| DB2_s6 | simple_cnn | 24.81% | 21.38% |
| DB2_s7 | simple_cnn | 16.05% | 11.11% |
| DB2_s8 | simple_cnn | 26.34% | 20.54% |
| DB2_s9 | simple_cnn | 27.76% | 22.73% |
| DB2_s10 | simple_cnn | 25.77% | 21.22% |
| DB2_s11 | simple_cnn | 32.88% | 27.23% |
| DB2_s12 | simple_cnn | 16.70% | 12.78% |
| DB2_s13 | simple_cnn | 19.05% | 15.21% |
| DB2_s14 | simple_cnn | 20.02% | 15.74% |
| DB2_s15 | simple_cnn | 24.63% | 20.46% |
| DB2_s16 | simple_cnn | 14.66% | 8.84% |
| DB2_s17 | simple_cnn | 22.00% | 17.20% |
| DB2_s18 | simple_cnn | 26.05% | 24.03% |
| DB2_s19 | simple_cnn | 31.17% | 25.45% |
| DB2_s20 | simple_cnn | 12.34% | 10.57% |
| DB2_s21 | simple_cnn | 27.37% | 23.55% |
| DB2_s22 | simple_cnn | 15.69% | 14.70% |
| DB2_s23 | simple_cnn | 22.60% | 17.29% |
| DB2_s24 | simple_cnn | 27.23% | 22.91% |
| DB2_s25 | simple_cnn | 29.76% | 24.57% |
| DB2_s26 | simple_cnn | 18.84% | 13.73% |
| DB2_s27 | simple_cnn | 22.94% | 20.70% |
| DB2_s28 | simple_cnn | 30.25% | 25.52% |
| DB2_s29 | simple_cnn | 23.63% | 20.72% |
| DB2_s30 | simple_cnn | 29.17% | 25.13% |
| DB2_s31 | simple_cnn | 19.19% | 15.85% |
| DB2_s32 | simple_cnn | 18.53% | 13.97% |
| DB2_s33 | simple_cnn | 28.71% | 24.22% |
| DB2_s34 | simple_cnn | 26.38% | 21.16% |
| DB2_s35 | simple_cnn | 27.24% | 25.43% |
| DB2_s36 | simple_cnn | 31.14% | 27.33% |
| DB2_s37 | simple_cnn | 11.69% | 9.28% |
| DB2_s38 | simple_cnn | 20.92% | 16.68% |
| DB2_s39 | simple_cnn | 21.24% | 16.29% |
| DB2_s40 | simple_cnn | 27.55% | 20.86% |

**vs Baseline**: BELOW best baseline by -8.9pp

---

### EXP_1: exp1_deep_raw_cnn_lstm_loso

- **Date**: 2026-02-20T02:49:42.776308
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_lstm
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_lstm | 20.78% | 16.54% | 0.0433 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_lstm | 15.16% | 7.90% |
| DB2_s12 | cnn_lstm | 15.81% | 14.44% |
| DB2_s15 | cnn_lstm | 24.17% | 21.43% |
| DB2_s28 | cnn_lstm | 24.30% | 18.05% |
| DB2_s39 | cnn_lstm | 24.45% | 20.89% |

**vs Baseline**: BELOW best baseline by -11.2pp

---

### EXP_1: exp1_deep_raw_multiscale_cnn_loso_isolated_v2

- **Date**: 2026-01-18T22:52:02.416288
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: multiscale_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| multiscale_cnn | 26.81% | 23.44% | 0.0571 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | multiscale_cnn | 30.26% | 28.91% |
| DB2_s2 | multiscale_cnn | 29.74% | 20.55% |
| DB2_s3 | multiscale_cnn | 18.87% | 13.67% |
| DB2_s4 | multiscale_cnn | 21.99% | 20.44% |
| DB2_s5 | multiscale_cnn | 24.96% | 21.34% |
| DB2_s11 | multiscale_cnn | 38.39% | 34.87% |
| DB2_s12 | multiscale_cnn | 20.75% | 18.83% |
| DB2_s13 | multiscale_cnn | 27.84% | 24.01% |
| DB2_s14 | multiscale_cnn | 30.36% | 24.85% |
| DB2_s15 | multiscale_cnn | 24.76% | 21.17% |
| DB2_s26 | multiscale_cnn | 30.15% | 24.58% |
| DB2_s27 | multiscale_cnn | 23.51% | 21.36% |
| DB2_s28 | multiscale_cnn | 28.69% | 29.03% |
| DB2_s29 | multiscale_cnn | 26.92% | 23.27% |
| DB2_s30 | multiscale_cnn | 32.08% | 28.83% |
| DB2_s36 | multiscale_cnn | 36.19% | 35.09% |
| DB2_s37 | multiscale_cnn | 13.09% | 9.99% |
| DB2_s38 | multiscale_cnn | 28.13% | 23.50% |
| DB2_s39 | multiscale_cnn | 22.13% | 19.48% |
| DB2_s40 | multiscale_cnn | 27.43% | 24.94% |

**vs Baseline**: BELOW best baseline by -5.2pp

---

### EXP_1: exp1_deep_raw_simple_cnn_loso_1_12_15_28_39

- **Date**: 2026-02-19T10:48:02.753763
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| simple_cnn | 25.99% | 24.11% | 0.0217 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | simple_cnn | 25.03% | 23.25% |
| DB2_s12 | simple_cnn | 22.29% | 19.11% |
| DB2_s15 | simple_cnn | 27.14% | 25.29% |
| DB2_s28 | simple_cnn | 28.63% | 25.28% |
| DB2_s39 | simple_cnn | 26.83% | 27.59% |

**vs Baseline**: BELOW best baseline by -6.0pp

---

### EXP_1: exp1_deep_raw_simple_cnn_loso_isolated_v2

- **Date**: 2026-01-18T15:42:36.331413
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| simple_cnn | 29.30% | 25.51% | 0.0779 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | simple_cnn | 29.13% | 23.98% |
| DB2_s2 | simple_cnn | 36.47% | 29.21% |
| DB2_s3 | simple_cnn | 15.04% | 11.21% |
| DB2_s4 | simple_cnn | 36.53% | 31.31% |
| DB2_s5 | simple_cnn | 30.55% | 27.59% |
| DB2_s11 | simple_cnn | 39.58% | 33.87% |
| DB2_s12 | simple_cnn | 22.00% | 20.48% |
| DB2_s13 | simple_cnn | 23.08% | 18.68% |
| DB2_s14 | simple_cnn | 28.76% | 24.67% |
| DB2_s15 | simple_cnn | 22.38% | 19.01% |
| DB2_s26 | simple_cnn | 28.62% | 23.17% |
| DB2_s27 | simple_cnn | 32.64% | 27.87% |
| DB2_s28 | simple_cnn | 41.51% | 37.00% |
| DB2_s29 | simple_cnn | 22.66% | 20.41% |
| DB2_s30 | simple_cnn | 39.42% | 34.72% |
| DB2_s36 | simple_cnn | 38.14% | 37.81% |
| DB2_s37 | simple_cnn | 12.91% | 9.00% |
| DB2_s38 | simple_cnn | 27.39% | 26.41% |
| DB2_s39 | simple_cnn | 29.76% | 27.69% |
| DB2_s40 | simple_cnn | 29.45% | 26.22% |

**vs Baseline**: BELOW best baseline by -2.7pp

---

### EXP_1: exp1_deep_raw_tcn_attn_loso_1_12_15_28_39

- **Date**: 2026-02-19T19:21:30.021207
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: tcn_attn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| tcn_attn | 24.66% | 22.05% | 0.0268 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tcn_attn | 20.69% | 17.20% |
| DB2_s12 | tcn_attn | 23.66% | 22.82% |
| DB2_s15 | tcn_attn | 23.63% | 20.60% |
| DB2_s28 | tcn_attn | 27.59% | 24.12% |
| DB2_s39 | tcn_attn | 27.74% | 25.50% |

**vs Baseline**: BELOW best baseline by -7.3pp

---

### EXP_1: exp1_deep_raw_tcn_attn_loso_isolated_v2

- **Date**: 2026-01-19T06:52:24.507212
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: tcn_attn
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| tcn_attn | 22.92% | 17.82% | 0.1128 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tcn_attn | 27.05% | 23.53% |
| DB2_s2 | tcn_attn | 29.38% | 22.72% |
| DB2_s3 | tcn_attn | 10.03% | 1.82% |
| DB2_s4 | tcn_attn | 24.79% | 21.85% |
| DB2_s5 | tcn_attn | 26.27% | 19.56% |
| DB2_s11 | tcn_attn | 46.85% | 44.24% |
| DB2_s12 | tcn_attn | 26.69% | 25.41% |
| DB2_s13 | tcn_attn | 10.05% | 1.83% |
| DB2_s14 | tcn_attn | 28.16% | 20.96% |
| DB2_s15 | tcn_attn | 10.00% | 1.82% |
| DB2_s26 | tcn_attn | 10.03% | 1.82% |
| DB2_s27 | tcn_attn | 9.99% | 1.82% |
| DB2_s28 | tcn_attn | 32.54% | 31.95% |
| DB2_s29 | tcn_attn | 25.21% | 23.28% |
| DB2_s30 | tcn_attn | 38.27% | 34.64% |
| DB2_s36 | tcn_attn | 36.98% | 34.23% |
| DB2_s37 | tcn_attn | 14.01% | 11.07% |
| DB2_s38 | tcn_attn | 10.01% | 1.82% |
| DB2_s39 | tcn_attn | 10.06% | 1.83% |
| DB2_s40 | tcn_attn | 32.09% | 30.28% |

**vs Baseline**: BELOW best baseline by -9.1pp

---

### EXP_1: exp1_deep_raw_tcn_loso_1_12_15_28_39

- **Date**: 2026-02-19T13:28:59.568297
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: tcn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| tcn | 23.11% | 18.58% | 0.0399 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tcn | 25.09% | 17.17% |
| DB2_s12 | tcn | 18.43% | 16.04% |
| DB2_s15 | tcn | 25.00% | 23.17% |
| DB2_s28 | tcn | 28.51% | 21.87% |
| DB2_s39 | tcn | 18.54% | 14.65% |

**vs Baseline**: BELOW best baseline by -8.9pp

---

### EXP_1: exp1_deep_raw_tcn_loso_isolated_v2

- **Date**: 2026-01-20T03:21:25.729483
- **Approach**: `deep_raw`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: tcn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| tcn | 26.91% | 23.57% | 0.0779 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tcn | 29.79% | 28.01% |
| DB2_s2 | tcn | 33.13% | 25.90% |
| DB2_s3 | tcn | 18.15% | 16.46% |
| DB2_s4 | tcn | 30.15% | 25.82% |
| DB2_s5 | tcn | 27.64% | 26.26% |
| DB2_s11 | tcn | 30.36% | 28.74% |
| DB2_s12 | tcn | 20.21% | 20.15% |
| DB2_s13 | tcn | 35.63% | 31.90% |
| DB2_s14 | tcn | 25.43% | 22.03% |
| DB2_s15 | tcn | 17.08% | 13.40% |
| DB2_s26 | tcn | 26.77% | 23.42% |
| DB2_s27 | tcn | 23.26% | 18.31% |
| DB2_s28 | tcn | 36.39% | 34.73% |
| DB2_s29 | tcn | 20.77% | 16.45% |
| DB2_s30 | tcn | 42.27% | 36.49% |
| DB2_s36 | tcn | 32.64% | 28.55% |
| DB2_s37 | tcn | 7.19% | 6.75% |
| DB2_s38 | tcn | 21.48% | 17.81% |
| DB2_s39 | tcn | 30.30% | 24.35% |
| DB2_s40 | tcn | 29.64% | 25.96% |

**vs Baseline**: BELOW best baseline by -5.1pp

---

### EXP_2: exp2_deep_emg_td_seq_attention_cnn_loso

- **Date**: 2026-01-20T14:46:24.875802
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: attention_cnn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| attention_cnn | 10.00% | 4.90% | 0.0370 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | attention_cnn | 9.39% | 3.83% |
| DB2_s2 | attention_cnn | 9.89% | 3.15% |
| DB2_s3 | attention_cnn | 9.31% | 1.81% |
| DB2_s4 | attention_cnn | 6.38% | 2.02% |
| DB2_s5 | attention_cnn | 11.79% | 8.85% |
| DB2_s11 | attention_cnn | 6.01% | 2.50% |
| DB2_s12 | attention_cnn | 9.99% | 1.86% |
| DB2_s13 | attention_cnn | 12.02% | 7.18% |
| DB2_s14 | attention_cnn | 10.52% | 4.11% |
| DB2_s15 | attention_cnn | 17.20% | 11.00% |
| DB2_s26 | attention_cnn | 11.08% | 5.31% |
| DB2_s27 | attention_cnn | 13.82% | 5.34% |
| DB2_s28 | attention_cnn | 11.48% | 9.98% |
| DB2_s29 | attention_cnn | 8.22% | 1.80% |
| DB2_s30 | attention_cnn | 17.47% | 11.60% |
| DB2_s36 | attention_cnn | 5.56% | 3.26% |
| DB2_s37 | attention_cnn | 12.85% | 8.26% |
| DB2_s38 | attention_cnn | 2.44% | 1.30% |
| DB2_s39 | attention_cnn | 9.15% | 1.73% |
| DB2_s40 | attention_cnn | 5.39% | 3.15% |

**vs Baseline**: BELOW best baseline by -22.0pp

---

### EXP_2: exp2_deep_emg_td_seq_bigru_loso

- **Date**: 2026-01-21T01:08:21.164215
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: bigru
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bigru | 25.15% | 20.67% | 0.0659 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bigru | 22.95% | 14.78% |
| DB2_s2 | bigru | 34.15% | 25.14% |
| DB2_s3 | bigru | 23.76% | 20.67% |
| DB2_s4 | bigru | 26.76% | 20.15% |
| DB2_s5 | bigru | 28.83% | 26.21% |
| DB2_s11 | bigru | 31.96% | 26.68% |
| DB2_s12 | bigru | 19.32% | 15.18% |
| DB2_s13 | bigru | 27.72% | 22.89% |
| DB2_s14 | bigru | 19.49% | 16.32% |
| DB2_s15 | bigru | 15.48% | 12.85% |
| DB2_s26 | bigru | 20.25% | 19.08% |
| DB2_s27 | bigru | 31.24% | 25.36% |
| DB2_s28 | bigru | 34.62% | 28.52% |
| DB2_s29 | bigru | 20.89% | 14.20% |
| DB2_s30 | bigru | 35.48% | 30.72% |
| DB2_s36 | bigru | 32.64% | 28.78% |
| DB2_s37 | bigru | 13.71% | 9.51% |
| DB2_s38 | bigru | 18.61% | 18.40% |
| DB2_s39 | bigru | 18.48% | 14.36% |
| DB2_s40 | bigru | 26.70% | 23.66% |

**vs Baseline**: BELOW best baseline by -6.8pp

---

### EXP_2: exp2_deep_emg_td_seq_bilstm_attention_loso

- **Date**: 2026-01-20T17:48:26.614455
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: bilstm_attention
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bilstm_attention | 25.10% | 19.99% | 0.0540 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bilstm_attention | 24.97% | 16.52% |
| DB2_s2 | bilstm_attention | 27.29% | 18.89% |
| DB2_s3 | bilstm_attention | 22.21% | 19.62% |
| DB2_s4 | bilstm_attention | 28.43% | 22.40% |
| DB2_s5 | bilstm_attention | 26.44% | 21.40% |
| DB2_s11 | bilstm_attention | 26.85% | 25.12% |
| DB2_s12 | bilstm_attention | 25.68% | 23.02% |
| DB2_s13 | bilstm_attention | 26.47% | 18.32% |
| DB2_s14 | bilstm_attention | 19.25% | 15.96% |
| DB2_s15 | bilstm_attention | 18.87% | 14.75% |
| DB2_s26 | bilstm_attention | 18.03% | 18.98% |
| DB2_s27 | bilstm_attention | 29.42% | 23.12% |
| DB2_s28 | bilstm_attention | 27.90% | 20.76% |
| DB2_s29 | bilstm_attention | 16.69% | 7.96% |
| DB2_s30 | bilstm_attention | 34.63% | 26.58% |
| DB2_s36 | bilstm_attention | 34.54% | 31.74% |
| DB2_s37 | bilstm_attention | 13.64% | 9.45% |
| DB2_s38 | bilstm_attention | 25.81% | 22.55% |
| DB2_s39 | bilstm_attention | 25.30% | 17.44% |
| DB2_s40 | bilstm_attention | 29.52% | 25.16% |

**vs Baseline**: BELOW best baseline by -6.9pp

---

### EXP_2: exp2_deep_emg_td_seq_bilstm_loso

- **Date**: 2026-01-20T17:21:14.167834
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: bilstm
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bilstm | 21.77% | 16.81% | 0.0474 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bilstm | 21.22% | 15.20% |
| DB2_s2 | bilstm | 27.65% | 23.59% |
| DB2_s3 | bilstm | 15.46% | 11.94% |
| DB2_s4 | bilstm | 24.31% | 17.88% |
| DB2_s5 | bilstm | 26.62% | 23.12% |
| DB2_s11 | bilstm | 27.32% | 19.36% |
| DB2_s12 | bilstm | 17.60% | 13.53% |
| DB2_s13 | bilstm | 24.03% | 18.48% |
| DB2_s14 | bilstm | 16.52% | 11.64% |
| DB2_s15 | bilstm | 17.50% | 13.72% |
| DB2_s26 | bilstm | 17.85% | 15.20% |
| DB2_s27 | bilstm | 24.18% | 18.01% |
| DB2_s28 | bilstm | 25.70% | 19.06% |
| DB2_s29 | bilstm | 22.66% | 14.02% |
| DB2_s30 | bilstm | 26.86% | 22.39% |
| DB2_s36 | bilstm | 26.34% | 21.28% |
| DB2_s37 | bilstm | 13.52% | 9.42% |
| DB2_s38 | bilstm | 14.58% | 11.62% |
| DB2_s39 | bilstm | 18.23% | 13.23% |
| DB2_s40 | bilstm | 27.19% | 23.60% |

**vs Baseline**: BELOW best baseline by -10.2pp

---

### EXP_2: exp2_deep_emg_td_seq_cnn_gru_attention_loso

- **Date**: 2026-01-21T12:06:28.024690
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: cnn_gru_attention
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_gru_attention | 15.27% | 8.83% | 0.0342 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention | 10.34% | 2.55% |
| DB2_s2 | cnn_gru_attention | 13.47% | 6.95% |
| DB2_s3 | cnn_gru_attention | 17.61% | 10.41% |
| DB2_s4 | cnn_gru_attention | 19.61% | 12.46% |
| DB2_s5 | cnn_gru_attention | 15.78% | 10.09% |
| DB2_s11 | cnn_gru_attention | 13.81% | 5.44% |
| DB2_s12 | cnn_gru_attention | 11.95% | 6.06% |
| DB2_s13 | cnn_gru_attention | 15.29% | 7.57% |
| DB2_s14 | cnn_gru_attention | 19.25% | 12.17% |
| DB2_s15 | cnn_gru_attention | 14.29% | 9.05% |
| DB2_s26 | cnn_gru_attention | 13.42% | 6.67% |
| DB2_s27 | cnn_gru_attention | 16.99% | 11.62% |
| DB2_s28 | cnn_gru_attention | 22.16% | 13.69% |
| DB2_s29 | cnn_gru_attention | 12.42% | 6.13% |
| DB2_s30 | cnn_gru_attention | 19.65% | 11.82% |
| DB2_s36 | cnn_gru_attention | 11.25% | 5.27% |
| DB2_s37 | cnn_gru_attention | 10.57% | 7.69% |
| DB2_s38 | cnn_gru_attention | 11.47% | 5.81% |
| DB2_s39 | cnn_gru_attention | 16.77% | 9.90% |
| DB2_s40 | cnn_gru_attention | 19.29% | 15.18% |

**vs Baseline**: BELOW best baseline by -16.7pp

---

### EXP_2: exp2_deep_emg_td_seq_cnn_lstm_loso

- **Date**: 2026-01-21T00:33:44.995780
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: cnn_lstm
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_lstm | 13.93% | 7.13% | 0.0389 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_lstm | 14.27% | 7.48% |
| DB2_s2 | cnn_lstm | 15.08% | 7.15% |
| DB2_s3 | cnn_lstm | 9.49% | 4.87% |
| DB2_s4 | cnn_lstm | 10.13% | 2.68% |
| DB2_s5 | cnn_lstm | 20.73% | 16.19% |
| DB2_s11 | cnn_lstm | 10.60% | 2.98% |
| DB2_s12 | cnn_lstm | 12.84% | 7.05% |
| DB2_s13 | cnn_lstm | 12.91% | 4.33% |
| DB2_s14 | cnn_lstm | 12.06% | 4.47% |
| DB2_s15 | cnn_lstm | 11.79% | 6.18% |
| DB2_s26 | cnn_lstm | 10.46% | 2.79% |
| DB2_s27 | cnn_lstm | 19.85% | 13.43% |
| DB2_s28 | cnn_lstm | 19.54% | 12.05% |
| DB2_s29 | cnn_lstm | 20.10% | 11.80% |
| DB2_s30 | cnn_lstm | 21.10% | 13.47% |
| DB2_s36 | cnn_lstm | 11.19% | 4.80% |
| DB2_s37 | cnn_lstm | 12.85% | 4.33% |
| DB2_s38 | cnn_lstm | 10.31% | 3.14% |
| DB2_s39 | cnn_lstm | 11.59% | 6.81% |
| DB2_s40 | cnn_lstm | 11.82% | 6.62% |

**vs Baseline**: BELOW best baseline by -18.1pp

---

### EXP_2: exp2_deep_emg_td_seq_multiscale_cnn_loso

- **Date**: 2026-01-20T16:03:20.516438
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: multiscale_cnn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| multiscale_cnn | 11.99% | 5.09% | 0.0213 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | multiscale_cnn | 17.12% | 8.90% |
| DB2_s2 | multiscale_cnn | 14.00% | 5.46% |
| DB2_s3 | multiscale_cnn | 11.64% | 3.74% |
| DB2_s4 | multiscale_cnn | 10.19% | 4.35% |
| DB2_s5 | multiscale_cnn | 10.84% | 8.05% |
| DB2_s11 | multiscale_cnn | 14.76% | 6.77% |
| DB2_s12 | multiscale_cnn | 15.28% | 10.63% |
| DB2_s13 | multiscale_cnn | 9.99% | 1.82% |
| DB2_s14 | multiscale_cnn | 11.82% | 4.90% |
| DB2_s15 | multiscale_cnn | 7.86% | 3.76% |
| DB2_s26 | multiscale_cnn | 9.97% | 1.81% |
| DB2_s27 | multiscale_cnn | 11.94% | 3.95% |
| DB2_s28 | multiscale_cnn | 14.41% | 6.47% |
| DB2_s29 | multiscale_cnn | 11.94% | 5.01% |
| DB2_s30 | multiscale_cnn | 11.10% | 4.15% |
| DB2_s36 | multiscale_cnn | 10.21% | 2.19% |
| DB2_s37 | multiscale_cnn | 11.25% | 4.71% |
| DB2_s38 | multiscale_cnn | 12.51% | 5.25% |
| DB2_s39 | multiscale_cnn | 10.79% | 3.02% |
| DB2_s40 | multiscale_cnn | 12.12% | 6.87% |

**vs Baseline**: BELOW best baseline by -20.0pp

---

### EXP_2: exp2_deep_emg_td_seq_simple_cnn_loso

- **Date**: 2026-01-20T12:38:22.424405
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| simple_cnn | 11.44% | 5.25% | 0.0232 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | simple_cnn | 13.14% | 7.97% |
| DB2_s2 | simple_cnn | 10.61% | 3.43% |
| DB2_s3 | simple_cnn | 10.03% | 1.90% |
| DB2_s4 | simple_cnn | 10.01% | 1.86% |
| DB2_s5 | simple_cnn | 10.42% | 5.62% |
| DB2_s11 | simple_cnn | 11.96% | 4.09% |
| DB2_s12 | simple_cnn | 10.58% | 4.05% |
| DB2_s13 | simple_cnn | 15.23% | 6.86% |
| DB2_s14 | simple_cnn | 13.73% | 6.87% |
| DB2_s15 | simple_cnn | 10.12% | 4.87% |
| DB2_s26 | simple_cnn | 12.98% | 9.49% |
| DB2_s27 | simple_cnn | 8.10% | 2.61% |
| DB2_s28 | simple_cnn | 14.96% | 8.99% |
| DB2_s29 | simple_cnn | 9.32% | 2.82% |
| DB2_s30 | simple_cnn | 11.46% | 6.99% |
| DB2_s36 | simple_cnn | 8.13% | 3.34% |
| DB2_s37 | simple_cnn | 17.03% | 11.51% |
| DB2_s38 | simple_cnn | 10.62% | 4.40% |
| DB2_s39 | simple_cnn | 10.67% | 4.51% |
| DB2_s40 | simple_cnn | 9.74% | 2.89% |

**vs Baseline**: BELOW best baseline by -20.6pp

---

### EXP_2: exp2_deep_emg_td_seq_tcn_attn_loso

- **Date**: 2026-01-20T15:34:05.052595
- **Approach**: `deep_emg_seq`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: tcn_attn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| tcn_attn | 21.57% | 15.98% | 0.0513 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tcn_attn | 28.06% | 18.86% |
| DB2_s2 | tcn_attn | 31.35% | 25.22% |
| DB2_s3 | tcn_attn | 18.27% | 14.11% |
| DB2_s4 | tcn_attn | 18.36% | 11.11% |
| DB2_s5 | tcn_attn | 16.32% | 10.78% |
| DB2_s11 | tcn_attn | 29.11% | 21.47% |
| DB2_s12 | tcn_attn | 14.33% | 8.92% |
| DB2_s13 | tcn_attn | 30.52% | 25.87% |
| DB2_s14 | tcn_attn | 24.48% | 20.54% |
| DB2_s15 | tcn_attn | 15.71% | 11.17% |
| DB2_s26 | tcn_attn | 21.91% | 16.93% |
| DB2_s27 | tcn_attn | 17.30% | 11.94% |
| DB2_s28 | tcn_attn | 24.05% | 19.60% |
| DB2_s29 | tcn_attn | 21.50% | 15.09% |
| DB2_s30 | tcn_attn | 18.86% | 13.29% |
| DB2_s36 | tcn_attn | 25.43% | 20.13% |
| DB2_s37 | tcn_attn | 20.41% | 15.47% |
| DB2_s38 | tcn_attn | 14.09% | 11.10% |
| DB2_s39 | tcn_attn | 21.71% | 12.29% |
| DB2_s40 | tcn_attn | 19.72% | 15.78% |

**vs Baseline**: BELOW best baseline by -10.4pp

---

### EXP_2: exp_2_deep_emg_td_seq_attention_cnn_loso

- **Date**: 2026-02-20T03:51:37.335262
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: attention_cnn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| attention_cnn | 23.62% | 22.20% | 0.0402 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | attention_cnn | 20.21% | 20.13% |
| DB2_s12 | attention_cnn | 21.46% | 20.70% |
| DB2_s15 | attention_cnn | 28.51% | 25.76% |
| DB2_s28 | attention_cnn | 28.45% | 25.31% |
| DB2_s39 | attention_cnn | 19.45% | 19.11% |

**vs Baseline**: BELOW best baseline by -8.4pp

---

### EXP_2: exp_2_deep_emg_td_seq_bigru_loso

- **Date**: 2026-02-20T04:53:16.641405
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: bigru
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bigru | 24.03% | 22.56% | 0.0519 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bigru | 22.65% | 19.71% |
| DB2_s12 | bigru | 17.42% | 16.61% |
| DB2_s15 | bigru | 29.29% | 28.46% |
| DB2_s28 | bigru | 30.77% | 28.83% |
| DB2_s39 | bigru | 20.00% | 19.22% |

**vs Baseline**: BELOW best baseline by -8.0pp

---

### EXP_2: exp_2_deep_emg_td_seq_bilstm_attention_loso

- **Date**: 2026-02-20T04:39:37.344080
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: bilstm_attention
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bilstm_attention | 24.70% | 23.44% | 0.0406 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bilstm_attention | 21.76% | 20.08% |
| DB2_s12 | bilstm_attention | 21.64% | 21.00% |
| DB2_s15 | bilstm_attention | 27.86% | 26.81% |
| DB2_s28 | bilstm_attention | 31.14% | 28.96% |
| DB2_s39 | bilstm_attention | 21.10% | 20.38% |

**vs Baseline**: BELOW best baseline by -7.3pp

---

### EXP_2: exp_2_deep_emg_td_seq_bilstm_loso

- **Date**: 2026-02-20T04:27:45.286324
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: bilstm
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| bilstm | 23.82% | 22.70% | 0.0586 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | bilstm | 20.10% | 18.81% |
| DB2_s12 | bilstm | 19.32% | 18.75% |
| DB2_s15 | bilstm | 26.85% | 26.09% |
| DB2_s28 | bilstm | 34.00% | 30.89% |
| DB2_s39 | bilstm | 18.84% | 18.94% |

**vs Baseline**: BELOW best baseline by -8.2pp

---

### EXP_2: exp_2_deep_emg_td_seq_cnn_gru_attention_loso

- **Date**: 2026-02-20T05:14:39.368459
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_gru_attention
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_gru_attention | 23.78% | 22.89% | 0.0448 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention | 21.94% | 21.38% |
| DB2_s12 | cnn_gru_attention | 17.90% | 17.83% |
| DB2_s15 | cnn_gru_attention | 29.23% | 28.91% |
| DB2_s28 | cnn_gru_attention | 28.82% | 26.22% |
| DB2_s39 | cnn_gru_attention | 21.04% | 20.11% |

**vs Baseline**: BELOW best baseline by -8.2pp

---

### EXP_2: exp_2_deep_emg_td_seq_cnn_lstm_loso

- **Date**: 2026-02-20T05:03:53.512776
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_lstm
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_lstm | 25.46% | 24.29% | 0.0260 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_lstm | 24.73% | 23.61% |
| DB2_s12 | cnn_lstm | 21.88% | 21.13% |
| DB2_s15 | cnn_lstm | 26.49% | 25.16% |
| DB2_s28 | cnn_lstm | 29.73% | 28.05% |
| DB2_s39 | cnn_lstm | 24.45% | 23.51% |

**vs Baseline**: BELOW best baseline by -6.5pp

---

### EXP_2: exp_2_deep_emg_td_seq_multiscale_cnn_loso

- **Date**: 2026-02-20T04:16:29.576233
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: multiscale_cnn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| multiscale_cnn | 23.41% | 22.44% | 0.0476 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | multiscale_cnn | 21.34% | 21.26% |
| DB2_s12 | multiscale_cnn | 22.18% | 21.86% |
| DB2_s15 | multiscale_cnn | 29.23% | 27.91% |
| DB2_s28 | multiscale_cnn | 28.08% | 25.70% |
| DB2_s39 | multiscale_cnn | 16.22% | 15.45% |

**vs Baseline**: BELOW best baseline by -8.6pp

---

### EXP_2: exp_2_deep_emg_td_seq_simple_cnn_loso

- **Date**: 2026-02-20T03:24:39.308441
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| simple_cnn | 23.37% | 22.51% | 0.0650 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | simple_cnn | 17.90% | 18.56% |
| DB2_s12 | simple_cnn | 18.97% | 18.67% |
| DB2_s15 | simple_cnn | 31.25% | 28.89% |
| DB2_s28 | simple_cnn | 31.38% | 29.30% |
| DB2_s39 | simple_cnn | 17.38% | 17.11% |

**vs Baseline**: BELOW best baseline by -8.6pp

---

### EXP_2: exp_2_deep_emg_td_seq_tcn_attn_loso

- **Date**: 2026-02-20T04:07:45.165404
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: tcn_attn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| tcn_attn | 23.50% | 22.42% | 0.0498 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tcn_attn | 16.88% | 15.98% |
| DB2_s12 | tcn_attn | 21.11% | 19.44% |
| DB2_s15 | tcn_attn | 27.44% | 28.30% |
| DB2_s28 | tcn_attn | 30.83% | 29.52% |
| DB2_s39 | tcn_attn | 21.22% | 18.84% |

**vs Baseline**: BELOW best baseline by -8.5pp

---

### EXP_2: exp_2_deep_emg_td_seq_tcn_loso

- **Date**: 2026-02-20T03:54:43.142516
- **Approach**: `deep_emg_seq`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: tcn
- **Epochs**: 50
- **Batch size**: 2048
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| tcn | 24.63% | 23.29% | 0.0532 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tcn | 23.13% | 21.76% |
| DB2_s12 | tcn | 20.45% | 20.03% |
| DB2_s15 | tcn | 24.82% | 23.58% |
| DB2_s28 | tcn | 34.68% | 32.15% |
| DB2_s39 | tcn | 20.06% | 18.93% |

**vs Baseline**: BELOW best baseline by -7.4pp

---

### EXP_3: exp3_deep_powerful_mlp_powerful_loso

- **Date**: 2026-01-21T14:26:49.030510
- **Approach**: `deep_powerful`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: mlp_powerful
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| mlp_powerful | 10.27% | 2.21% | 0.0117 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | mlp_powerful | 9.99% | 2.13% |
| DB2_s2 | mlp_powerful | 10.01% | 1.82% |
| DB2_s3 | mlp_powerful | 9.91% | 1.80% |
| DB2_s4 | mlp_powerful | 9.95% | 1.82% |
| DB2_s5 | mlp_powerful | 10.01% | 1.86% |
| DB2_s11 | mlp_powerful | 10.00% | 1.82% |
| DB2_s12 | mlp_powerful | 9.99% | 1.82% |
| DB2_s13 | mlp_powerful | 9.99% | 1.82% |
| DB2_s14 | mlp_powerful | 10.04% | 1.83% |
| DB2_s15 | mlp_powerful | 10.00% | 1.82% |
| DB2_s26 | mlp_powerful | 10.09% | 1.83% |
| DB2_s27 | mlp_powerful | 8.40% | 3.24% |
| DB2_s28 | mlp_powerful | 10.07% | 1.84% |
| DB2_s29 | mlp_powerful | 14.56% | 5.75% |
| DB2_s30 | mlp_powerful | 9.95% | 1.81% |
| DB2_s36 | mlp_powerful | 10.02% | 1.82% |
| DB2_s37 | mlp_powerful | 10.14% | 1.84% |
| DB2_s38 | mlp_powerful | 10.07% | 1.83% |
| DB2_s39 | mlp_powerful | 12.38% | 3.93% |
| DB2_s40 | mlp_powerful | 9.92% | 1.81% |

**vs Baseline**: BELOW best baseline by -21.7pp

---

### EXP_4: exp4_rf_powerful_loso

- **Date**: 2026-01-21T15:41:44.910382
- **Approach**: `ml_emg_td`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: rf
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| rf | 32.00% | 30.33% | 0.0632 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | rf | 29.73% | 26.81% |
| DB2_s2 | rf | 36.35% | 32.83% |
| DB2_s3 | rf | 20.90% | 20.69% |
| DB2_s4 | rf | 34.27% | 32.70% |
| DB2_s5 | rf | 35.02% | 33.63% |
| DB2_s11 | rf | 42.98% | 42.18% |
| DB2_s12 | rf | 28.83% | 27.71% |
| DB2_s13 | rf | 32.00% | 28.80% |
| DB2_s14 | rf | 33.16% | 32.29% |
| DB2_s15 | rf | 29.64% | 29.23% |
| DB2_s26 | rf | 28.18% | 27.26% |
| DB2_s27 | rf | 34.47% | 34.75% |
| DB2_s28 | rf | 39.38% | 38.35% |
| DB2_s29 | rf | 24.91% | 20.45% |
| DB2_s30 | rf | 40.08% | 37.61% |
| DB2_s36 | rf | 39.00% | 38.41% |
| DB2_s37 | rf | 16.04% | 12.03% |
| DB2_s38 | rf | 30.87% | 30.07% |
| DB2_s39 | rf | 30.61% | 29.29% |
| DB2_s40 | rf | 33.50% | 31.45% |

**vs Baseline**: BELOW best baseline by -0.0pp

---

### EXP_4: exp4_svm_linear_powerful_loso

- **Date**: 2026-01-21T15:33:41.963056
- **Approach**: `ml_emg_td`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: svm_linear
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_linear | 35.24% | 32.50% | 0.0870 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_linear | 35.37% | 29.76% |
| DB2_s2 | svm_linear | 37.25% | 32.56% |
| DB2_s3 | svm_linear | 19.04% | 16.72% |
| DB2_s4 | svm_linear | 40.05% | 37.18% |
| DB2_s5 | svm_linear | 35.91% | 35.93% |
| DB2_s11 | svm_linear | 49.76% | 48.35% |
| DB2_s12 | svm_linear | 31.69% | 30.12% |
| DB2_s13 | svm_linear | 30.99% | 27.57% |
| DB2_s14 | svm_linear | 32.56% | 30.01% |
| DB2_s15 | svm_linear | 30.65% | 31.33% |
| DB2_s26 | svm_linear | 35.38% | 34.67% |
| DB2_s27 | svm_linear | 43.12% | 40.80% |
| DB2_s28 | svm_linear | 39.68% | 36.65% |
| DB2_s29 | svm_linear | 24.79% | 21.30% |
| DB2_s30 | svm_linear | 45.72% | 41.70% |
| DB2_s36 | svm_linear | 46.21% | 44.29% |
| DB2_s37 | svm_linear | 12.91% | 7.25% |
| DB2_s38 | svm_linear | 40.33% | 37.05% |
| DB2_s39 | svm_linear | 38.11% | 35.59% |
| DB2_s40 | svm_linear | 35.21% | 31.20% |

**vs Baseline**: ABOVE best baseline by +3.2pp

---

### EXP_4: exp4_svm_rbf_powerful_loso

- **Date**: 2026-01-21T14:10:00.401949
- **Approach**: `ml_emg_td`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: svm_rbf
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 34.46% | 32.60% | 0.0701 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 39.06% | 34.86% |
| DB2_s2 | svm_rbf | 35.82% | 30.66% |
| DB2_s3 | svm_rbf | 23.10% | 23.12% |
| DB2_s4 | svm_rbf | 40.58% | 38.99% |
| DB2_s5 | svm_rbf | 30.20% | 28.78% |
| DB2_s11 | svm_rbf | 45.12% | 43.89% |
| DB2_s12 | svm_rbf | 30.56% | 29.72% |
| DB2_s13 | svm_rbf | 33.19% | 30.87% |
| DB2_s14 | svm_rbf | 34.52% | 33.29% |
| DB2_s15 | svm_rbf | 30.71% | 29.73% |
| DB2_s26 | svm_rbf | 28.62% | 27.88% |
| DB2_s27 | svm_rbf | 44.82% | 44.03% |
| DB2_s28 | svm_rbf | 39.32% | 39.03% |
| DB2_s29 | svm_rbf | 29.66% | 25.74% |
| DB2_s30 | svm_rbf | 45.36% | 43.21% |
| DB2_s36 | svm_rbf | 36.67% | 37.07% |
| DB2_s37 | svm_rbf | 17.09% | 11.59% |
| DB2_s38 | svm_rbf | 34.11% | 33.52% |
| DB2_s39 | svm_rbf | 32.44% | 30.83% |
| DB2_s40 | svm_rbf | 38.21% | 35.29% |

**vs Baseline**: ABOVE best baseline by +2.5pp

---

### EXP_4: exp_4_ml_powerful_loso

- **Date**: 2026-02-20T05:40:23.977820
- **Approach**: `ml_emg_td`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: rf
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 27.82% | 26.63% | 0.0483 | 5 |
| rf | 25.12% | 23.47% | 0.0487 | 5 |
| svm_linear | 25.04% | 23.78% | 0.0454 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 23.54% | 22.23% |
| DB2_s12 | svm_rbf | 27.17% | 26.10% |
| DB2_s15 | svm_rbf | 33.33% | 32.01% |
| DB2_s28 | svm_rbf | 33.33% | 31.67% |
| DB2_s39 | svm_rbf | 21.71% | 21.13% |
| DB2_s1 | svm_linear | 22.71% | 22.15% |
| DB2_s12 | svm_linear | 17.48% | 17.28% |
| DB2_s15 | svm_linear | 29.52% | 29.04% |
| DB2_s28 | svm_linear | 29.43% | 24.33% |
| DB2_s39 | svm_linear | 26.04% | 26.09% |
| DB2_s1 | rf | 20.93% | 19.96% |
| DB2_s12 | rf | 23.13% | 21.87% |
| DB2_s15 | rf | 28.33% | 27.75% |
| DB2_s28 | rf | 33.03% | 28.95% |
| DB2_s39 | rf | 20.18% | 18.83% |

**vs Baseline**: BELOW best baseline by -4.2pp

---

### EXP_5: exp5_hybrid_powerful_deep_loso

- **Date**: 2026-01-22T19:01:09.525608
- **Approach**: `hybrid_powerful_deep`
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)
- **Model type**: hybrid_powerful_deep
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| hybrid_powerful_deep | 10.74% | 2.67% | 0.0142 | 20 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | hybrid_powerful_deep | 9.99% | 1.82% |
| DB2_s2 | hybrid_powerful_deep | 10.01% | 1.82% |
| DB2_s3 | hybrid_powerful_deep | 10.03% | 1.83% |
| DB2_s4 | hybrid_powerful_deep | 9.95% | 1.81% |
| DB2_s5 | hybrid_powerful_deep | 10.01% | 1.91% |
| DB2_s11 | hybrid_powerful_deep | 10.00% | 1.82% |
| DB2_s12 | hybrid_powerful_deep | 9.99% | 1.82% |
| DB2_s13 | hybrid_powerful_deep | 15.41% | 5.45% |
| DB2_s14 | hybrid_powerful_deep | 12.54% | 4.90% |
| DB2_s15 | hybrid_powerful_deep | 10.00% | 2.20% |
| DB2_s26 | hybrid_powerful_deep | 10.22% | 2.16% |
| DB2_s27 | hybrid_powerful_deep | 10.11% | 1.84% |
| DB2_s28 | hybrid_powerful_deep | 10.74% | 4.69% |
| DB2_s29 | hybrid_powerful_deep | 9.81% | 1.79% |
| DB2_s30 | hybrid_powerful_deep | 10.25% | 2.41% |
| DB2_s36 | hybrid_powerful_deep | 12.96% | 4.27% |
| DB2_s37 | hybrid_powerful_deep | 10.14% | 1.84% |
| DB2_s38 | hybrid_powerful_deep | 10.07% | 1.83% |
| DB2_s39 | hybrid_powerful_deep | 10.00% | 1.82% |
| DB2_s40 | hybrid_powerful_deep | 12.49% | 5.38% |

**vs Baseline**: BELOW best baseline by -21.3pp

---

### EXP_7: exp_7_cnn_gru_attention_with_noise_and_time_warp_augment_loso

- **Date**: 2026-02-18T11:12:40.977168
- **Approach**: `fusion_with_augmentation`
- **Hypothesis**: test-001
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_gru_attention_fusion
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention_fusion | 44.13% | 19.52% |
| DB2_s12 | cnn_gru_attention_fusion | 24.83% | 10.96% |
| DB2_s15 | cnn_gru_attention_fusion | 36.87% | 14.86% |
| DB2_s28 | cnn_gru_attention_fusion | 35.52% | 15.95% |
| DB2_s39 | cnn_gru_attention_fusion | 30.46% | 15.64% |

---

### EXP_8: exp_8_augmented_svm_with_time_domain_features_for_improv_loso

- **Date**: 2026-02-19T09:35:48.068776
- **Approach**: `powerful`
- **Hypothesis**: 264d86e7-9299-44ed-87ff-f0d8f622ad82
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_linear
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_linear | 22.90% | 22.30% |
| DB2_s12 | svm_linear | 18.85% | 18.45% |
| DB2_s15 | svm_linear | 28.50% | 27.07% |
| DB2_s28 | svm_linear | 33.50% | 31.10% |
| DB2_s39 | svm_linear | 23.82% | 23.22% |

---

### EXP_9: exp_9_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso

- **Date**: 2026-02-19T03:25:51.492881
- **Approach**: `dual_stream`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: dual_stream_cnn_gru_attention
- **Epochs**: 60
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | dual_stream_cnn_gru_attention | 46.71% | 22.27% |
| DB2_s12 | dual_stream_cnn_gru_attention | 23.91% | 11.43% |
| DB2_s15 | dual_stream_cnn_gru_attention | 36.34% | 13.40% |
| DB2_s28 | dual_stream_cnn_gru_attention | 32.57% | 19.50% |
| DB2_s39 | dual_stream_cnn_gru_attention | 30.83% | 15.78% |

---

### EXP_10: exp_10_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso

- **Date**: 2026-02-19T07:26:29.665874
- **Approach**: `deep_raw`
- **Hypothesis**: 0d519c5d-fa3f-44d1-b955-1ef224cc74e4
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: dual_stream_cnn_gru_attention
- **Epochs**: 60
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | dual_stream_cnn_gru_attention | 26.75% | 25.17% |
| DB2_s12 | dual_stream_cnn_gru_attention | 17.06% | 16.57% |
| DB2_s15 | dual_stream_cnn_gru_attention | 23.57% | 21.13% |
| DB2_s28 | dual_stream_cnn_gru_attention | 26.86% | 24.74% |
| DB2_s39 | dual_stream_cnn_gru_attention | 23.60% | 22.06% |

---

### EXP_11: exp_11_enhancing_simple_cnn_robustness_with_subject_speci_loso

- **Date**: 2026-02-19T11:00:59.843879
- **Approach**: `deep_raw`
- **Hypothesis**: 11cfe09b-5d98-4fa9-9bee-16222fae2aab
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|

---

### EXP_12: exp_12_augmented_svm_with_time_domain_features_for_improv_loso

- **Date**: 2026-02-19T12:17:02.261235
- **Approach**: `powerful`
- **Hypothesis**: 264d86e7-9299-44ed-87ff-f0d8f622ad82
- **Note**: SVM-linear on powerful time-domain features with data augmentation.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_linear
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_linear | 25.51% | 24.43% | 0.0504 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_linear | 22.90% | 22.30% |
| DB2_s12 | svm_linear | 18.85% | 18.45% |
| DB2_s15 | svm_linear | 28.50% | 27.07% |
| DB2_s28 | svm_linear | 33.50% | 31.10% |
| DB2_s39 | svm_linear | 23.82% | 23.22% |

**vs Baseline**: BELOW best baseline by -6.5pp

---

### EXP_13: exp_13_leveraging_subject_specific_calibration_via_few_sh_loso

- **Date**: 2026-02-19T05:32:48.336475
- **Approach**: `few_shot_learning_with_maml_meta_training`
- **Hypothesis**: 3901a9cf-1112-4c14-9c68-baed76f94c28
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 40
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| simple_cnn_few_shot | 38.25% | 6.13% | 0.0046 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 38.62% | 6.96% |
| DB2_s12 | ? | 37.85% | 5.79% |
| DB2_s15 | ? | 37.69% | 4.98% |
| DB2_s28 | ? | 38.92% | 6.96% |
| DB2_s39 | ? | 38.18% | 5.93% |

**vs Baseline**: ABOVE best baseline by +6.3pp

---

### EXP_14: exp_14_subject_adaptive_fine_tuning_for_svm_on_powerful_f_loso

- **Date**: 2026-02-19T12:42:03.902586
- **Approach**: `subject_adaptive_fine_tuning`
- **Hypothesis**: 639423ce-b142-41c5-a9a0-8838cfc030e5
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_linear
- **Epochs**: 1
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|

---

### EXP_15: exp_15_contrastive_subject_aware_loso

- **Date**: 2026-02-19T13:52:00.060075
- **Approach**: `contrastive_subject_aware`
- **Hypothesis**: 70d5a74e-f695-4d0b-a782-6c001ea6f4be
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: contrastive_cnn
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|

---

### EXP_16: exp_16_enhanced_augmentation_strategy_for_cnn_gru_attenti_loso

- **Date**: 2026-02-19T08:37:36.791778
- **Approach**: `deep_raw`
- **Hypothesis**: Enhanced augmentation strategy for CNN-GRU-Attention on raw EMG
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_gru_attention
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|

---

### EXP_17: exp_17_enhanced_augmentation_for_simple_cnn_on_raw_emg_si_loso

- **Date**: 2026-02-19T09:03:10.401592
- **Approach**: `deep_raw`
- **Hypothesis**: c7d68e78-1306-4585-a213-4675c9920b82
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | simple_cnn | 26.22% | 23.87% |
| DB2_s12 | simple_cnn | 22.12% | 19.45% |
| DB2_s15 | simple_cnn | 22.50% | 17.81% |
| DB2_s28 | simple_cnn | 28.21% | 23.88% |
| DB2_s39 | simple_cnn | 27.20% | 27.27% |

---

### EXP_18: exp_18_augmented_svm_with_feature_space_jitter_for_improv_loso

- **Date**: 2026-02-19T12:07:32.706542
- **Approach**: `powerful`
- **Hypothesis**: d7c44dc7-cc18-4669-8b30-9db935f75204
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_rbf
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 43.00% | 21.52% |
| DB2_s12 | svm_rbf | 23.83% | 16.24% |
| DB2_s15 | svm_rbf | 47.09% | 22.03% |
| DB2_s28 | svm_rbf | 52.80% | 28.83% |
| DB2_s39 | svm_rbf | 36.93% | 19.33% |

---

### EXP_19: exp_19_subject_specific_feature_calibration_for_reducing_loso

- **Date**: 2026-02-19T09:55:49.246645
- **Approach**: `ml_emg_td_with_mmd_calibration`
- **Hypothesis**: df248bc2-7fdc-44c3-a9ff-48e0110bd9bc
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | calibrated_svm_linear | 16.73% | 16.15% |
| DB2_s12 | calibrated_svm_linear | 14.78% | 14.59% |
| DB2_s15 | calibrated_svm_linear | 22.46% | 21.49% |
| DB2_s28 | calibrated_svm_linear | 22.52% | 21.94% |
| DB2_s39 | calibrated_svm_linear | 17.69% | 17.44% |

---

### EXP_20: exp_20_augmented_cnn_gru_attention_with_feature_fusion_fo_loso

- **Date**: 2026-02-19T18:13:57.137366
- **Approach**: `hybrid_fusion_with_augmentation`
- **Hypothesis**: e2f39b15-6605-42fc-ae5c-dcc0a0153653
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: fusion_cnn_gru_attention
- **Epochs**: 60
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | fusion_cnn_gru_attention | 41.75% | 16.68% |
| DB2_s12 | fusion_cnn_gru_attention | 32.73% | 11.24% |
| DB2_s15 | fusion_cnn_gru_attention | 37.71% | 14.35% |
| DB2_s28 | fusion_cnn_gru_attention | 41.44% | 16.99% |
| DB2_s39 | fusion_cnn_gru_attention | 34.50% | 14.74% |

---

### EXP_21: exp_21_svm_rbf_with_noise_time_warp_augmentation_and_tune_loso

- **Date**: 2026-02-20T16:31:33.406896
- **Approach**: `powerful`
- **Hypothesis**: 04b951c7-774f-449b-9202-eaf2bceb6376
- **Note**: SVM-RBF with signal-level augmentation and increased regularization (C=1.0). Tests the missing cell in the SVM×augmentation matrix.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_rbf
- **Epochs**: 1
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 19.44% | 16.49% | 0.0316 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 15.18% | 11.36% |
| DB2_s12 | svm_rbf | 23.04% | 21.84% |
| DB2_s15 | svm_rbf | 16.17% | 10.44% |
| DB2_s28 | svm_rbf | 21.03% | 20.58% |
| DB2_s39 | svm_rbf | 21.80% | 18.23% |

**vs Baseline**: BELOW best baseline by -12.6pp

---

### EXP_22: exp_22_focal_loss_class_balanced_sampling_for_cnn_gru_att_loso

- **Date**: 2026-02-20T16:51:05.938087
- **Approach**: `deep_raw_with_focal_loss`
- **Hypothesis**: 212764c1-2890-41d7-ada1-07d9e3e6b76d
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_gru_attention
- **Epochs**: 30
- **Batch size**: 256
- **Learning rate**: 0.0001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention | 21.88% | 17.43% |
| DB2_s12 | cnn_gru_attention | 21.58% | 17.74% |
| DB2_s15 | cnn_gru_attention | 14.40% | 8.19% |
| DB2_s28 | cnn_gru_attention | 26.07% | 21.98% |
| DB2_s39 | cnn_gru_attention | 26.71% | 25.52% |

---

### EXP_23: exp_23_channel_wise_squeeze_and_excitation_cnn_gru_attent_loso

- **Date**: 2026-02-20T17:02:05.540759
- **Approach**: `deep_raw`
- **Hypothesis**: 2846e57d-5df2-477f-9d5b-9cab62aab2c9
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_gru_attention
- **Epochs**: 60
- **Batch size**: 4096
- **Learning rate**: 0.0005
- **Window size**: 600

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|

---

### EXP_24: exp_24_cnn_gru_attention_on_raw_emg_with_class_weighted_l_loso

- **Date**: 2026-02-20T18:29:31.487906
- **Approach**: `deep_raw`
- **Hypothesis**: 2c875e39-5fe0-4026-aa67-f7772e46980e
- **Note**: Testing whether class-weighted loss prevents F1 collapse in augmented models. Baseline exp_1: 30.85% acc, 28.19% F1 (ratio 1.09). Fusion experiments (exp_7, exp_20) showed accuracy gains but F1 collapse. Key insight: addressing class imbalance in loss function, not architecture.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_gru_attention
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_gru_attention | 26.30% | 25.22% | 0.0426 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention | 27.29% | 25.45% |
| DB2_s12 | cnn_gru_attention | 20.04% | 20.52% |
| DB2_s15 | cnn_gru_attention | 25.95% | 24.38% |
| DB2_s28 | cnn_gru_attention | 33.27% | 31.20% |
| DB2_s39 | cnn_gru_attention | 24.94% | 24.56% |

**vs Baseline**: BELOW best baseline by -5.7pp

---

### EXP_25: exp_25_svm_linear_on_powerful_features_with_combined_nois_loso

- **Date**: 2026-02-20T18:34:33.900282
- **Approach**: `powerful`
- **Hypothesis**: 3b1480c0-aa08-4cad-b277-966e3654c011
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_linear_powerful_triple_aug | 26.08% | 24.89% | 0.0517 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 20.65% | 20.53% |
| DB2_s12 | ? | 21.29% | 19.74% |
| DB2_s15 | ? | 29.87% | 27.52% |
| DB2_s28 | ? | 34.09% | 33.09% |
| DB2_s39 | ? | 24.51% | 23.55% |

**vs Baseline**: BELOW best baseline by -5.9pp

---

### EXP_26: exp_26_test_time_bn_adaptation_for_cnn_g_loso

- **Date**: 2026-02-20T18:39:05.717379
- **Approach**: `deep_raw`
- **Hypothesis**: 9c0b2f84-8bf2-41f8-9df0-4187086fd96c
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_gru_attention
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 500

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_gru_attention | 30.74% | 30.14% | 0.0714 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention | 33.46% | 30.46% |
| DB2_s12 | cnn_gru_attention | 17.09% | 16.91% |
| DB2_s15 | cnn_gru_attention | 31.34% | 32.21% |
| DB2_s28 | cnn_gru_attention | 37.85% | 37.62% |
| DB2_s39 | cnn_gru_attention | 33.96% | 33.53% |

**vs Baseline**: BELOW best baseline by -1.3pp

---

### EXP_27: exp_27_moe_cnn_gru_attention_loso

- **Date**: 2026-02-21T10:22:52.835381
- **Approach**: `deep_raw`
- **Hypothesis**: H1: Subject-as-Domain MoE — signal-style gating over CNN-GRU-Attention experts
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: moe_cnn_gru_attention
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| moe_cnn_gru_attention | 26.33% | 24.71% | 0.0499 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | moe_cnn_gru_attention | 26.63% | 23.53% |
| DB2_s12 | moe_cnn_gru_attention | 23.60% | 21.45% |
| DB2_s15 | moe_cnn_gru_attention | 22.98% | 22.58% |
| DB2_s28 | moe_cnn_gru_attention | 35.90% | 34.08% |
| DB2_s39 | moe_cnn_gru_attention | 22.56% | 21.93% |

**vs Baseline**: BELOW best baseline by -5.7pp

---

### EXP_28: exp_28_film_subject_adaptive

- **Date**: 20260221_104924
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | film_subject_adaptive | 16.16% | 12.69% |
| DB2_s12 | film_subject_adaptive | 29.62% | 28.07% |
| DB2_s15 | film_subject_adaptive | 35.09% | 32.44% |
| DB2_s28 | film_subject_adaptive | 37.10% | 34.24% |
| DB2_s39 | film_subject_adaptive | 23.38% | 19.05% |

**vs Baseline**: ABOVE best baseline by +5.1pp

---

### EXP_29: exp_29_spectral_transformer_loso

- **Date**: 2026-02-21T11:27:21.892853
- **Approach**: `deep_raw`
- **Hypothesis**: H3: Spectral self-attention over frequency bands + channels > CNN-RNN
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: spectral_transformer
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.0005
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| spectral_transformer | 21.63% | 17.89% | 0.0443 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | spectral_transformer | 21.05% | 16.73% |
| DB2_s12 | spectral_transformer | 18.01% | 15.02% |
| DB2_s15 | spectral_transformer | 27.86% | 24.51% |
| DB2_s28 | spectral_transformer | 15.93% | 11.71% |
| DB2_s39 | spectral_transformer | 25.30% | 21.50% |

**vs Baseline**: BELOW best baseline by -10.4pp

---

### EXP_30: exp_30_channel_gat_loso

- **Date**: 2026-02-21T12:01:26.235181
- **Approach**: `deep_raw`
- **Hypothesis**: H30: Inter-electrode correlations > temporal structure. GAT on channel graph with dynamic adjacency.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: channel_gat
- **Epochs**: 60
- **Batch size**: 256
- **Learning rate**: 0.0003
- **Window size**: 400

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| channel_gat | 22.60% | 20.30% | 0.0319 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | channel_gat | 19.74% | 17.09% |
| DB2_s12 | channel_gat | 22.13% | 19.36% |
| DB2_s15 | channel_gat | 27.88% | 25.12% |
| DB2_s28 | channel_gat | 24.13% | 22.24% |
| DB2_s39 | channel_gat | 19.11% | 17.71% |

**vs Baseline**: BELOW best baseline by -9.4pp

---

### EXP_31: exp_31_disentangled_content_style

- **Date**: 20260221_132041
- **Approach**: `deep_raw`
- **Hypothesis**: H5: Content-Style Disentanglement
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | disentangled_cnn_gru | 38.55% | 33.29% |
| DB2_s12 | disentangled_cnn_gru | 32.56% | 28.73% |
| DB2_s15 | disentangled_cnn_gru | 35.77% | 32.87% |
| DB2_s28 | disentangled_cnn_gru | 42.39% | 37.20% |
| DB2_s39 | disentangled_cnn_gru | 45.05% | 44.33% |

**vs Baseline**: ABOVE best baseline by +13.1pp

---

### EXP_32: exp_32_spectral_band_ratio_svm_loso

- **Date**: 2026-02-21T13:53:21.165053
- **Approach**: `spectral_band_ratio + SVM`
- **Hypothesis**: Spectral band power ratios are more stable across subjects than absolute power
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_linear
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 27.40% | 25.96% | 0.0404 | 5 |
| svm_linear | 24.11% | 22.86% | 0.0432 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 26.16% | 26.04% |
| DB2_s12 | svm_rbf | 28.36% | 28.29% |
| DB2_s15 | svm_rbf | 25.83% | 22.10% |
| DB2_s28 | svm_rbf | 34.43% | 32.59% |
| DB2_s39 | svm_rbf | 22.20% | 20.80% |
| DB2_s1 | svm_linear | 19.50% | 19.38% |
| DB2_s12 | svm_linear | 20.99% | 18.89% |
| DB2_s15 | svm_linear | 23.75% | 22.70% |
| DB2_s28 | svm_linear | 31.99% | 30.27% |
| DB2_s39 | svm_linear | 24.33% | 23.09% |

**vs Baseline**: BELOW best baseline by -4.6pp

---

### EXP_33: exp_33_wavelet_scattering_svm_loso

- **Date**: 2026-02-21T14:31:19.035667
- **Approach**: `wavelet_scattering + SVM`
- **Hypothesis**: 1D wavelet scattering transform provides built-in invariance to time-warping and scale deformations, improving cross-subject EMG classification without learned features
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_linear
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 27.60% | 26.60% | 0.0632 | 5 |
| svm_linear | 24.40% | 23.34% | 0.0589 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 22.89% | 22.87% |
| DB2_s12 | svm_rbf | 21.28% | 20.92% |
| DB2_s15 | svm_rbf | 34.40% | 32.36% |
| DB2_s28 | svm_rbf | 36.14% | 34.88% |
| DB2_s39 | svm_rbf | 23.29% | 21.97% |
| DB2_s1 | svm_linear | 22.83% | 22.29% |
| DB2_s12 | svm_linear | 15.81% | 15.70% |
| DB2_s15 | svm_linear | 28.75% | 27.43% |
| DB2_s28 | svm_linear | 32.84% | 29.94% |
| DB2_s39 | svm_linear | 21.77% | 21.32% |

**vs Baseline**: BELOW best baseline by -4.4pp

---

### EXP_34: exp_34_curriculum_subject_ordering

- **Date**: 20260221_153442
- **Approach**: `deep_raw`
- **Hypothesis**: Curriculum learning: training from similar to distant subjects
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention | 30.70% | 26.21% |
| DB2_s12 | cnn_gru_attention | 32.33% | 30.93% |
| DB2_s15 | cnn_gru_attention | 33.16% | 31.49% |
| DB2_s28 | cnn_gru_attention | 37.74% | 32.90% |
| DB2_s39 | cnn_gru_attention | 35.96% | 36.87% |

**vs Baseline**: ABOVE best baseline by +5.7pp

---

### EXP_35: exp_35_mae_ssl_pretrain_loso

- **Date**: 2026-02-21T16:13:44.101830
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: mae_emg
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.0005
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| mae_emg | 25.61% | 24.49% | 0.0363 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | mae_emg | 24.91% | 25.05% |
| DB2_s12 | mae_emg | 19.80% | 18.68% |
| DB2_s15 | mae_emg | 25.54% | 23.68% |
| DB2_s28 | mae_emg | 31.14% | 28.59% |
| DB2_s39 | mae_emg | 26.65% | 26.43% |

**vs Baseline**: BELOW best baseline by -6.4pp

---

### EXP_36: exp_36_prototypical_arcface_loso

- **Date**: 2026-02-21T16:29:38.016695
- **Approach**: ``
- **Hypothesis**: H10
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: prototypical_arcface
- **Epochs**: 80
- **Batch size**: 256
- **Learning rate**: 0.0003
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 25.45% | 20.55% |
| DB2_s12 | ? | 18.49% | 15.26% |
| DB2_s15 | ? | 16.79% | 14.41% |
| DB2_s28 | ? | 21.98% | 20.93% |
| DB2_s39 | ? | 17.74% | 15.85% |

---

### EXP_37: exp_37_channel_gat_gru_loso

- **Date**: 2026-02-22T09:49:04.867729
- **Approach**: `deep_raw`
- **Hypothesis**: H37: Spatio-temporal graph network with GAT (Pearson correlation + spectral coherence as edges) + per-channel BiGRU captures subject-invariant inter-muscular co-activation patterns.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: channel_gat_gru
- **Epochs**: 70
- **Batch size**: 128
- **Learning rate**: 0.0003
- **Window size**: 400

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| channel_gat_gru | 22.59% | 21.33% | 0.0412 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | channel_gat_gru | 23.97% | 20.46% |
| DB2_s12 | channel_gat_gru | 19.11% | 19.05% |
| DB2_s15 | channel_gat_gru | 24.53% | 22.67% |
| DB2_s28 | channel_gat_gru | 28.46% | 27.51% |
| DB2_s39 | channel_gat_gru | 16.87% | 16.97% |

**vs Baseline**: BELOW best baseline by -9.4pp

---

### EXP_38: exp_38_nonlinear_stats_svm_lgbm_loso

- **Date**: 2026-02-22T10:49:14.361454
- **Approach**: `CombinedNonlinearExtractor + SVM / LightGBM`
- **Hypothesis**: Nonlinear statistics (sample entropy, permutation entropy, Higuchi FD, Hjorth, Lyapunov) and channel-pair features (cross-corr, coherence, MI) are more subject-invariant than classical RMS/MAV/PSD features.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: lgbm
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 30.07% | 28.77% | 0.0469 | 5 |
| lgbm | 30.07% | 28.77% | 0.0469 | 5 |
| svm_linear | 23.82% | 22.18% | 0.0523 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 26.16% | 24.27% |
| DB2_s12 | svm_rbf | 28.48% | 28.49% |
| DB2_s15 | svm_rbf | 32.08% | 29.39% |
| DB2_s28 | svm_rbf | 38.22% | 36.78% |
| DB2_s39 | svm_rbf | 25.43% | 24.91% |
| DB2_s1 | svm_linear | 19.80% | 16.63% |
| DB2_s12 | svm_linear | 17.18% | 17.33% |
| DB2_s15 | svm_linear | 25.30% | 23.77% |
| DB2_s28 | svm_linear | 32.42% | 29.87% |
| DB2_s39 | svm_linear | 24.39% | 23.31% |
| DB2_s1 | lgbm | 26.16% | 24.27% |
| DB2_s12 | lgbm | 28.48% | 28.49% |
| DB2_s15 | lgbm | 32.08% | 29.39% |
| DB2_s28 | lgbm | 38.22% | 36.78% |
| DB2_s39 | lgbm | 25.43% | 24.91% |

**vs Baseline**: BELOW best baseline by -1.9pp

---

### EXP_40: exp_40_window_quality_filtering_loso

- **Date**: 2026-02-22T12:08:29.712482
- **Approach**: ``
- **Hypothesis**: Removing low-quality EMG windows (by SNR/kurtosis/saturation/ZCR/channel-correlation/RMS-energy) from training data improves cross-subject classification transferability.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| simple_cnn__percentile_0.2 | 28.67% | 25.47% | 0.0401 | 5 |
| svm_rbf__percentile_0.2 | 26.70% | 25.51% | 0.0418 | 5 |
| svm_rbf__none | 26.49% | 24.32% | 0.0447 | 5 |
| svm_rbf__hard_threshold_0.3 | 26.49% | 24.32% | 0.0447 | 5 |
| svm_rbf__hard_threshold_0.4 | 26.49% | 24.32% | 0.0447 | 5 |
| svm_rbf__percentile_0.1 | 26.03% | 25.20% | 0.0387 | 5 |
| simple_cnn__percentile_0.1 | 25.92% | 22.23% | 0.0395 | 5 |
| simple_cnn__percentile_0.3 | 24.60% | 20.33% | 0.0555 | 5 |
| svm_rbf__percentile_0.3 | 24.52% | 22.72% | 0.0464 | 5 |
| simple_cnn__none | 18.01% | 11.85% | 0.0499 | 5 |
| simple_cnn__hard_threshold_0.3 | 18.01% | 11.85% | 0.0499 | 5 |
| simple_cnn__hard_threshold_0.4 | 18.01% | 11.85% | 0.0499 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 21.34% | 20.83% |
| DB2_s12 | svm_rbf | 25.98% | 20.88% |
| DB2_s15 | svm_rbf | 32.14% | 30.29% |
| DB2_s28 | svm_rbf | 31.01% | 28.95% |
| DB2_s39 | svm_rbf | 21.95% | 20.63% |
| DB2_s1 | svm_rbf | 21.17% | 21.53% |
| DB2_s12 | svm_rbf | 24.38% | 23.75% |
| DB2_s15 | svm_rbf | 30.00% | 29.26% |
| DB2_s28 | svm_rbf | 31.14% | 29.34% |
| DB2_s39 | svm_rbf | 23.48% | 22.11% |
| DB2_s1 | svm_rbf | 20.33% | 19.84% |
| DB2_s12 | svm_rbf | 25.27% | 24.93% |
| DB2_s15 | svm_rbf | 32.20% | 30.57% |
| DB2_s28 | svm_rbf | 30.28% | 28.64% |
| DB2_s39 | svm_rbf | 25.43% | 23.59% |
| DB2_s1 | svm_rbf | 20.63% | 18.99% |
| DB2_s12 | svm_rbf | 22.83% | 21.30% |
| DB2_s15 | svm_rbf | 30.89% | 29.63% |
| DB2_s28 | svm_rbf | 29.06% | 26.43% |
| DB2_s39 | svm_rbf | 19.21% | 17.26% |
| DB2_s1 | svm_rbf | 21.34% | 20.83% |
| DB2_s12 | svm_rbf | 25.98% | 20.88% |
| DB2_s15 | svm_rbf | 32.14% | 30.29% |
| DB2_s28 | svm_rbf | 31.01% | 28.95% |
| DB2_s39 | svm_rbf | 21.95% | 20.63% |
| DB2_s1 | svm_rbf | 21.34% | 20.83% |
| DB2_s12 | svm_rbf | 25.98% | 20.88% |
| DB2_s15 | svm_rbf | 32.14% | 30.29% |
| DB2_s28 | svm_rbf | 31.01% | 28.95% |
| DB2_s39 | svm_rbf | 21.95% | 20.63% |
| DB2_s1 | simple_cnn | 26.10% | 20.27% |
| DB2_s12 | simple_cnn | 20.99% | 17.38% |
| DB2_s15 | simple_cnn | 11.85% | 5.25% |
| DB2_s28 | simple_cnn | 15.57% | 7.73% |
| DB2_s39 | simple_cnn | 15.55% | 8.60% |
| DB2_s1 | simple_cnn | 19.80% | 13.80% |
| DB2_s12 | simple_cnn | 23.07% | 20.06% |
| DB2_s15 | simple_cnn | 27.26% | 24.25% |
| DB2_s28 | simple_cnn | 28.88% | 25.87% |
| DB2_s39 | simple_cnn | 30.61% | 27.16% |

**vs Baseline**: BELOW best baseline by -3.3pp

---

### EXP_41: exp_41_content_style_graph

- **Date**: 20260222_122335
- **Approach**: `deep_raw`
- **Hypothesis**: Content-Style Graph: GRU/Attention content + GNN style for subject-invariance
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | content_style_graph | 31.12% | 28.10% |
| DB2_s12 | content_style_graph | 28.73% | 27.36% |
| DB2_s15 | content_style_graph | 29.72% | 28.79% |
| DB2_s28 | content_style_graph | 41.28% | 36.64% |
| DB2_s39 | content_style_graph | 37.18% | 36.58% |

**vs Baseline**: ABOVE best baseline by +9.3pp

---

### EXP_42: exp_42_multi_task_ssl_pretrain

- **Date**: 2026-02-22T13:08:48.105891
- **Approach**: ``
- **Hypothesis**: Multi-task SSL pretraining (MAE + Subject Prediction + Cross-Subject Contrastive + Decorrelation) learns gesture-invariant features separated from subject style, improving LOSO cross-subject generalization.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: multi_task_ssl
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.0005
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 24.02% | 23.20% |
| DB2_s12 | ? | 22.65% | 20.68% |
| DB2_s15 | ? | 24.76% | 24.68% |
| DB2_s28 | ? | 34.86% | 33.27% |
| DB2_s39 | ? | 23.90% | 23.77% |

---

### EXP_43: exp_43_tcn_gat_hybrid_loso

- **Date**: 2026-02-22T13:57:22.848732
- **Approach**: `deep_raw`
- **Hypothesis**: H43: Multi-scale causal dilated TCN (dilation {1,2,4,8}, kernel=7, RF≈184 ts ≈ 92 ms @ 2 kHz) + dynamic channel-graph GAT (Pearson + learnable prior) + per-channel BiGRU + temporal attention captures subject-invariant inter-muscular co-activation patterns.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: tcn_gat_hybrid
- **Epochs**: 80
- **Batch size**: 128
- **Learning rate**: 0.0003
- **Window size**: 400

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| tcn_gat_hybrid | 25.62% | 23.47% | 0.0267 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tcn_gat_hybrid | 23.93% | 20.64% |
| DB2_s12 | tcn_gat_hybrid | 27.18% | 26.35% |
| DB2_s15 | tcn_gat_hybrid | 27.88% | 25.80% |
| DB2_s28 | tcn_gat_hybrid | 27.94% | 26.00% |
| DB2_s39 | tcn_gat_hybrid | 21.15% | 18.57% |

**vs Baseline**: BELOW best baseline by -6.4pp

---

### EXP_44: exp_44_curriculum_disentangled_class_balanced_fusion

- **Date**: 20260222_152434
- **Approach**: `deep_raw`
- **Hypothesis**: H_fusion: Curriculum + Disentanglement + Class-Balanced MixUp
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 36.35% | 33.48% |
| DB2_s12 | ? | 35.59% | 31.91% |
| DB2_s15 | ? | 30.12% | 28.56% |
| DB2_s28 | ? | 40.24% | 35.44% |
| DB2_s39 | ? | 36.69% | 35.30% |

**vs Baseline**: ABOVE best baseline by +8.2pp

---

### EXP_46: exp_46_synthetic_subjects_domain_expansion_loso

- **Date**: 2026-02-22T17:00:19.267625
- **Approach**: ``
- **Hypothesis**: Expanding subject space with VAE-generated synthetic subjects reduces cross-subject gap in LOSO evaluation.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cnn_gru_attention
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 400

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cnn_gru_attention | 0.00% | 0.00% | 0.0000 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | cnn_gru_attention | 25.45% | 24.37% |
| DB2_s12 | cnn_gru_attention | 18.29% | 17.83% |
| DB2_s15 | cnn_gru_attention | 29.47% | 27.82% |
| DB2_s28 | cnn_gru_attention | 31.07% | 28.04% |
| DB2_s39 | cnn_gru_attention | 23.35% | 23.17% |

---

### EXP_47: exp_47_vq_disentanglement_for_content_st_loso

- **Date**: 2026-02-23T15:32:06.651969
- **Approach**: `deep_raw`
- **Hypothesis**: h-047-vq-disentanglement
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: vq_disentangle_emg
- **Epochs**: 60
- **Batch size**: 512
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| vq_disentangle_emg | 9.96% | 1.81% | 0.0009 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | vq_disentangle_emg | 10.05% | 1.83% |
| DB2_s12 | vq_disentangle_emg | 9.99% | 1.82% |
| DB2_s15 | vq_disentangle_emg | 10.06% | 1.83% |
| DB2_s28 | vq_disentangle_emg | 9.89% | 1.80% |
| DB2_s39 | vq_disentangle_emg | 9.82% | 1.79% |

**vs Baseline**: BELOW best baseline by -22.0pp

---

### EXP_48: exp_48_invariant_risk_minimization_for_causal_gesture_fea_loso

- **Date**: 2026-02-23T16:16:35.421603
- **Approach**: `deep_raw`
- **Hypothesis**: h-048-irm-regularization
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: irm_content_style_emg
- **Epochs**: 60
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| irm_content_style_emg | 27.96% | 26.19% | 0.0525 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | irm_content_style_emg | 21.64% | 21.59% |
| DB2_s12 | irm_content_style_emg | 25.56% | 22.65% |
| DB2_s15 | irm_content_style_emg | 30.54% | 27.67% |
| DB2_s28 | irm_content_style_emg | 36.81% | 33.87% |
| DB2_s39 | irm_content_style_emg | 25.24% | 25.18% |

**vs Baseline**: BELOW best baseline by -4.0pp

---

### EXP_49: exp_49_emg_cepstral_coefficients_muscle_filterbanks_loso

- **Date**: 2026-02-23T14:16:54.792380
- **Approach**: `emgcc+powerful`
- **Hypothesis**: h-049-emgcc-cepstral
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_rbf
- **Epochs**: 1
- **Batch size**: 512
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 24.92% | 23.29% | 0.0496 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 25.45% | 22.03% |
| DB2_s12 | svm_rbf | 29.01% | 29.35% |
| DB2_s15 | svm_rbf | 16.19% | 13.54% |
| DB2_s28 | svm_rbf | 30.22% | 30.71% |
| DB2_s39 | svm_rbf | 23.72% | 20.82% |

**vs Baseline**: BELOW best baseline by -7.1pp

---

### EXP_50: exp_50_subject_conditional_normalizing_flows_loso

- **Date**: 2026-02-23T14:23:16.368517
- **Approach**: `deep_raw`
- **Hypothesis**: h-050-normalizing-flows
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| flow_canonical_emg | 29.06% | 27.34% | 0.0564 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | flow_canonical_emg | 24.61% | 24.64% |
| DB2_s12 | flow_canonical_emg | 24.91% | 24.77% |
| DB2_s15 | flow_canonical_emg | 32.32% | 28.82% |
| DB2_s28 | flow_canonical_emg | 38.71% | 34.44% |
| DB2_s39 | flow_canonical_emg | 24.76% | 24.06% |

**vs Baseline**: BELOW best baseline by -2.9pp

---

### EXP_51: exp_51_rank_copula_features_monotone_invariant_loso

- **Date**: 2026-02-23T14:53:32.733921
- **Approach**: `rank_copula+powerful`
- **Hypothesis**: h-051-rank-copula
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 20.54% | 18.25% | 0.0487 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 18.31% | 15.87% |
| DB2_s12 | svm_rbf | 23.25% | 21.62% |
| DB2_s15 | svm_rbf | 12.50% | 6.60% |
| DB2_s28 | svm_rbf | 26.86% | 25.83% |
| DB2_s39 | svm_rbf | 21.77% | 21.32% |

**vs Baseline**: BELOW best baseline by -11.5pp

---

### EXP_52: exp_52_channel_permutation_equivariant_set_transformer_loso

- **Date**: 2026-02-23T15:03:21.635095
- **Approach**: `deep_raw`
- **Hypothesis**: h-052-channel-permutation-equivariant
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| set_transformer_emg | 20.96% | 15.86% | 0.0545 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | set_transformer_emg | 16.23% | 10.20% |
| DB2_s12 | set_transformer_emg | 22.95% | 16.19% |
| DB2_s15 | set_transformer_emg | 21.01% | 15.29% |
| DB2_s28 | set_transformer_emg | 14.59% | 10.44% |
| DB2_s39 | set_transformer_emg | 30.00% | 27.18% |

**vs Baseline**: BELOW best baseline by -11.0pp

---

### EXP_53: exp_53_latent_diffusion_subject_style_removal_loso

- **Date**: 2026-02-23T15:08:17.548274
- **Approach**: `deep_raw`
- **Hypothesis**: h-053-diffusion-canonical
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| latent_diffusion_emg | 26.39% | 24.94% | 0.0511 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | latent_diffusion_emg | 22.35% | 21.35% |
| DB2_s12 | latent_diffusion_emg | 18.43% | 17.84% |
| DB2_s15 | latent_diffusion_emg | 31.79% | 30.07% |
| DB2_s28 | latent_diffusion_emg | 29.61% | 26.56% |
| DB2_s39 | latent_diffusion_emg | 29.76% | 28.90% |

**vs Baseline**: BELOW best baseline by -5.6pp

---

### EXP_54: exp_54_multi_resolution_temporal_consensus_loso

- **Date**: 2026-02-23T15:12:43.360686
- **Approach**: `deep_raw`
- **Hypothesis**: h-054-multi-resolution-consensus
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| multi_resolution_emg | 29.37% | 26.70% | 0.0506 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | multi_resolution_emg | 22.18% | 20.19% |
| DB2_s12 | multi_resolution_emg | 25.15% | 23.55% |
| DB2_s15 | multi_resolution_emg | 34.64% | 32.20% |
| DB2_s28 | multi_resolution_emg | 34.86% | 30.06% |
| DB2_s39 | multi_resolution_emg | 30.00% | 27.48% |

**vs Baseline**: BELOW best baseline by -2.6pp

---

### EXP_55: exp_55_optimal_transport_barycenter_alignment_loso

- **Date**: 2026-02-23T15:15:50.096848
- **Approach**: `powerful+ot_alignment`
- **Hypothesis**: h-055-ot-barycenter
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 13.08% | 11.54% | 0.0143 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 13.26% | 12.68% |
| DB2_s12 | svm_rbf | 15.76% | 14.32% |
| DB2_s15 | svm_rbf | 12.02% | 11.49% |
| DB2_s28 | svm_rbf | 11.78% | 9.32% |
| DB2_s39 | svm_rbf | 12.56% | 9.88% |

**vs Baseline**: BELOW best baseline by -18.9pp

---

### EXP_56: exp_56_test_time_training_masked_channel_ssl_loso

- **Date**: 2026-02-23T15:24:24.084376
- **Approach**: `deep_raw`
- **Hypothesis**: h-056-test-time-training
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| ttt_content_style_emg | 30.27% | 30.35% | 0.0479 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ttt_content_style_emg | 29.72% | 28.69% |
| DB2_s12 | ttt_content_style_emg | 21.23% | 23.35% |
| DB2_s15 | ttt_content_style_emg | 33.36% | 34.11% |
| DB2_s28 | ttt_content_style_emg | 34.52% | 33.13% |
| DB2_s39 | ttt_content_style_emg | 32.51% | 32.45% |

**vs Baseline**: BELOW best baseline by -1.7pp

---

### EXP_57: exp_57_groupdro_disentangled

- **Date**: 20260223_182152
- **Approach**: `deep_raw`
- **Hypothesis**: H8: GroupDRO + Content-Style Disentanglement
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | groupdro_disentangled_cnn_gru | 30.40% | 27.34% |
| DB2_s12 | groupdro_disentangled_cnn_gru | 38.79% | 35.45% |
| DB2_s15 | groupdro_disentangled_cnn_gru | 30.12% | 27.82% |
| DB2_s28 | groupdro_disentangled_cnn_gru | 41.47% | 35.78% |
| DB2_s39 | groupdro_disentangled_cnn_gru | 40.72% | 40.46% |

**vs Baseline**: ABOVE best baseline by +9.5pp

---

### EXP_58: exp_58_mi_subject_adversary

- **Date**: 20260223_183350
- **Approach**: `deep_raw`
- **Hypothesis**: H5b: CLUB MI upper-bound subject adversary
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | mi_disentangled_cnn_gru | 32.18% | 26.49% |
| DB2_s12 | mi_disentangled_cnn_gru | 33.21% | 29.43% |
| DB2_s15 | mi_disentangled_cnn_gru | 21.63% | 19.35% |
| DB2_s28 | mi_disentangled_cnn_gru | 33.03% | 27.70% |
| DB2_s39 | mi_disentangled_cnn_gru | 24.05% | 20.92% |

**vs Baseline**: ABOVE best baseline by +1.2pp

---

### EXP_59: exp_59_disentanglement_prototype_regularization

- **Date**: 20260223_184618
- **Approach**: `deep_raw`
- **Hypothesis**: Disentanglement + Prototype Regularization in z_content
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | proto_disentangled_cnn_gru | 31.53% | 29.04% |
| DB2_s12 | proto_disentangled_cnn_gru | 35.77% | 33.93% |
| DB2_s15 | proto_disentangled_cnn_gru | 27.63% | 23.97% |
| DB2_s28 | proto_disentangled_cnn_gru | 41.90% | 34.77% |
| DB2_s39 | proto_disentangled_cnn_gru | 42.86% | 43.00% |

**vs Baseline**: ABOVE best baseline by +10.9pp

---

### EXP_60: exp_60_mixstyle_content_disentangled

- **Date**: 20260223_185851
- **Approach**: `deep_raw`
- **Hypothesis**: H60: Mixture of Styles — FiLM in latent z_style space
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | mixstyle_disentangled_cnn_gru | 38.96% | 36.01% |
| DB2_s12 | mixstyle_disentangled_cnn_gru | 34.34% | 31.84% |
| DB2_s15 | mixstyle_disentangled_cnn_gru | 27.57% | 26.87% |
| DB2_s28 | mixstyle_disentangled_cnn_gru | 41.90% | 37.44% |
| DB2_s39 | mixstyle_disentangled_cnn_gru | 43.04% | 41.44% |

**vs Baseline**: ABOVE best baseline by +11.0pp

---

### EXP_61: exp_61_sinc_pcen_frontend

- **Date**: 20260223_191318
- **Approach**: `deep_raw`
- **Hypothesis**: SincNet-PCEN learnable frontend for channel-invariant EMG
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | sinc_pcen_cnn_gru | 24.93% | 26.47% |
| DB2_s12 | sinc_pcen_cnn_gru | 19.04% | 15.63% |
| DB2_s15 | sinc_pcen_cnn_gru | 30.90% | 27.84% |
| DB2_s28 | sinc_pcen_cnn_gru | 18.17% | 11.94% |
| DB2_s39 | sinc_pcen_cnn_gru | 13.86% | 6.16% |

**vs Baseline**: BELOW best baseline by -1.1pp

---

### EXP_62: exp_62_ecapa_tdnn_subject_robust

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_63: exp_63_riemannian_spd_covariance_loso

- **Date**: 2026-02-23T21:28:23.776845
- **Approach**: `RiemannianSPDExtractor (log-Euclidean mean) + SVM`
- **Hypothesis**: Riemannian tangent-space covariance features are more cross-subject invariant than classical EMG amplitude/power descriptors.  SPD covariance captures inter-channel muscle co-activation patterns robustly to per-subject amplitude shifts (cf. EEG Riemannian BCI).
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| combined_svm_rbf | 30.28% | 29.45% | 0.0503 | 5 |
| pure_svm_rbf | 25.30% | 24.47% | 0.0402 | 5 |
| hankel_svm_rbf | 25.07% | 24.14% | 0.0381 | 5 |
| pure_svm_linear | 22.68% | 21.93% | 0.0352 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 26.58% | 24.62% |
| DB2_s12 | ? | 18.49% | 16.89% |
| DB2_s15 | ? | 30.18% | 29.39% |
| DB2_s28 | ? | 27.72% | 27.56% |
| DB2_s39 | ? | 23.54% | 23.89% |
| DB2_s1 | ? | 23.48% | 22.65% |
| DB2_s12 | ? | 17.18% | 15.40% |
| DB2_s15 | ? | 24.64% | 23.78% |
| DB2_s28 | ? | 27.47% | 26.75% |
| DB2_s39 | ? | 20.61% | 21.08% |
| DB2_s1 | ? | 26.69% | 26.16% |
| DB2_s12 | ? | 29.96% | 29.98% |
| DB2_s15 | ? | 32.68% | 30.36% |
| DB2_s28 | ? | 38.34% | 37.90% |
| DB2_s39 | ? | 23.72% | 22.86% |
| DB2_s1 | ? | 25.98% | 24.60% |
| DB2_s12 | ? | 18.73% | 16.66% |
| DB2_s15 | ? | 29.17% | 28.62% |
| DB2_s28 | ? | 28.39% | 27.90% |
| DB2_s39 | ? | 23.11% | 22.95% |

**vs Baseline**: BELOW best baseline by -1.7pp

---

### EXP_64: exp_64_multiclass_csp_filterbank_loso

- **Date**: 2026-02-23T21:51:35.756900
- **Approach**: `One-vs-Rest Multiclass CSP (+ optional FilterBank) + SVM. CSP filters fitted on train subjects only (LOSO safe).`
- **Hypothesis**: CSP spatial filters learned from multi-subject training data produce log-variance features that are cross-subject robust.  Log-variance of CSP components is invariant to multiplicative gain shifts (electrode placement, skin impedance variation) — a key source of inter-subject EMG variability.  OAS shrinkage and a frequency filter bank further improve generalisation.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| fbcsp_svm_rbf | 28.75% | 27.99% | 0.0487 | 5 |
| ovr_csp_svm_rbf | 25.67% | 24.92% | 0.0552 | 5 |
| ovr_csp_shrink_svm | 25.59% | 24.86% | 0.0550 | 5 |
| ovr_csp_svm_linear | 23.22% | 22.10% | 0.0608 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 23.96% | 22.25% |
| DB2_s12 | ? | 22.95% | 20.86% |
| DB2_s15 | ? | 24.70% | 24.30% |
| DB2_s28 | ? | 36.32% | 35.36% |
| DB2_s39 | ? | 20.43% | 21.85% |
| DB2_s1 | ? | 25.15% | 22.63% |
| DB2_s12 | ? | 16.71% | 14.93% |
| DB2_s15 | ? | 28.10% | 26.62% |
| DB2_s28 | ? | 30.65% | 30.45% |
| DB2_s39 | ? | 15.49% | 15.86% |
| DB2_s1 | ? | 23.84% | 22.09% |
| DB2_s12 | ? | 22.83% | 20.87% |
| DB2_s15 | ? | 24.46% | 24.15% |
| DB2_s28 | ? | 36.26% | 35.26% |
| DB2_s39 | ? | 20.55% | 21.95% |
| DB2_s1 | ? | 24.08% | 23.18% |
| DB2_s12 | ? | 25.45% | 25.12% |
| DB2_s15 | ? | 28.99% | 27.69% |
| DB2_s28 | ? | 37.91% | 37.62% |
| DB2_s39 | ? | 27.32% | 26.33% |

**vs Baseline**: BELOW best baseline by -3.3pp

---

### EXP_65: exp_65_trainable_fir_neural_drive_deconv

- **Date**: 20260223_220733
- **Approach**: `deep_raw`
- **Hypothesis**: Trainable per-channel FIR frontend as neural-drive deconvolution for cross-subject EMG classification.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | fir_deconv_cnn_gru | 29.74% | 25.76% |
| DB2_s12 | fir_deconv_cnn_gru | 25.03% | 22.75% |
| DB2_s15 | fir_deconv_cnn_gru | 34.88% | 33.49% |
| DB2_s28 | fir_deconv_cnn_gru | 45.08% | 40.21% |
| DB2_s39 | fir_deconv_cnn_gru | 36.94% | 36.72% |

**vs Baseline**: ABOVE best baseline by +13.1pp

---

### EXP_66: exp_66_temporal_phase_alignment

- **Date**: 20260223_222844
- **Approach**: `deep_raw`
- **Hypothesis**: TKEO-based temporal phase alignment canonicalizes gesture timing across subjects, reducing inter-subject variability for cross-subject EMG gesture recognition.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | phase_align_cnn_gru | 33.37% | 30.33% |
| DB2_s12 | phase_align_cnn_gru | 31.02% | 29.81% |
| DB2_s15 | phase_align_cnn_gru | 27.21% | 24.35% |
| DB2_s28 | phase_align_cnn_gru | 39.08% | 34.13% |
| DB2_s39 | phase_align_cnn_gru | 36.02% | 35.72% |

**vs Baseline**: ABOVE best baseline by +7.1pp

---

### EXP_67: exp_67_cyclostationary_spectral_correlation_loso

- **Date**: 2026-02-24T11:31:27.389151
- **Approach**: `CombinedCyclostationaryExtractor (PowerfulFeatureExtractor + CyclostationaryEMGExtractor) → z-score → PCA (train only) → SVM/LightGBM`
- **Hypothesis**: H67: Cyclostationary features (normalized ACF, spectral cyclic coherence, envelope periodicity) are amplitude-invariant and capture gesture-specific MU firing structure, improving cross-subject generalization. Analogous to cyclostationary channel-invariance in communications.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: lgbm
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 30.72% | 29.81% | 0.0474 | 5 |
| lgbm | 30.72% | 29.81% | 0.0474 | 5 |
| svm_linear | 24.57% | 23.33% | 0.0486 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 27.82% | 27.24% |
| DB2_s12 | svm_rbf | 28.83% | 28.95% |
| DB2_s15 | svm_rbf | 35.00% | 33.47% |
| DB2_s28 | svm_rbf | 37.36% | 35.84% |
| DB2_s39 | svm_rbf | 24.57% | 23.55% |
| DB2_s1 | svm_linear | 21.58% | 20.28% |
| DB2_s12 | svm_linear | 18.61% | 17.89% |
| DB2_s15 | svm_linear | 27.98% | 27.64% |
| DB2_s28 | svm_linear | 32.17% | 29.24% |
| DB2_s39 | svm_linear | 22.50% | 21.61% |
| DB2_s1 | lgbm | 27.82% | 27.24% |
| DB2_s12 | lgbm | 28.83% | 28.95% |
| DB2_s15 | lgbm | 35.00% | 33.47% |
| DB2_s28 | lgbm | 37.36% | 35.84% |
| DB2_s39 | lgbm | 24.57% | 23.55% |

**vs Baseline**: BELOW best baseline by -1.3pp

---

### EXP_68: exp_68_multitaper_psd_spectral_slope_loso

- **Date**: 2026-02-24T12:00:09.510084
- **Approach**: `MultitaperPSDExtractor + PowerfulFeatureExtractor + SVM / LGBM`
- **Hypothesis**: Multitaper PSD aperiodic exponent, spectral knee, and residual oscillatory peaks (FOOOF-style, log-log linear fit) are more subject-invariant than raw band power or amplitude. Combined with PowerfulFeatureExtractor for SVM/LGBM.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: lgbm
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 28.52% | 27.74% | 0.0463 | 5 |
| lgbm | 28.52% | 27.74% | 0.0463 | 5 |
| svm_linear | 24.37% | 23.45% | 0.0542 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 24.55% | 24.66% |
| DB2_s12 | svm_rbf | 28.24% | 28.50% |
| DB2_s15 | svm_rbf | 30.42% | 28.11% |
| DB2_s28 | svm_rbf | 36.20% | 35.15% |
| DB2_s39 | svm_rbf | 23.17% | 22.28% |
| DB2_s1 | svm_linear | 19.74% | 18.68% |
| DB2_s12 | svm_linear | 18.61% | 17.49% |
| DB2_s15 | svm_linear | 24.94% | 25.20% |
| DB2_s28 | svm_linear | 33.94% | 31.92% |
| DB2_s39 | svm_linear | 24.63% | 23.97% |
| DB2_s1 | lgbm | 24.55% | 24.66% |
| DB2_s12 | lgbm | 28.24% | 28.50% |
| DB2_s15 | lgbm | 30.42% | 28.11% |
| DB2_s28 | lgbm | 36.20% | 35.15% |
| DB2_s39 | lgbm | 23.17% | 22.28% |

**vs Baseline**: BELOW best baseline by -3.5pp

---

### EXP_69: exp_69_vrex_fishr_irm_v2_loso

- **Date**: 2026-02-24T12:21:55.725590
- **Approach**: ``
- **Hypothesis**: h-069-vrex-fishr-irm-v2
- **Note**: Same backbone as exp_48 (IRMv1) for fair comparison
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: irm_content_style_emg
- **Epochs**: 60
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| vrex | 28.70% | 26.69% | 0.0431 | 5 |
| fishr | 28.09% | 26.70% | 0.0367 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | irm_content_style_emg | 23.96% | 21.93% |
| DB2_s12 | irm_content_style_emg | 28.12% | 25.98% |
| DB2_s15 | irm_content_style_emg | 30.42% | 27.98% |
| DB2_s28 | irm_content_style_emg | 36.02% | 32.96% |
| DB2_s39 | irm_content_style_emg | 25.00% | 24.58% |
| DB2_s1 | irm_content_style_emg | 23.78% | 22.15% |
| DB2_s12 | irm_content_style_emg | 27.59% | 25.34% |
| DB2_s15 | irm_content_style_emg | 29.05% | 29.05% |
| DB2_s28 | irm_content_style_emg | 34.49% | 31.74% |
| DB2_s39 | irm_content_style_emg | 25.55% | 25.23% |

**vs Baseline**: BELOW best baseline by -3.3pp

---

### EXP_70: exp_70_conformer_ecapa

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_72: exp_72_moe_dynamic_routing_loso

- **Date**: 2026-02-24T11:01:31.771916
- **Approach**: `deep_raw`
- **Hypothesis**: H72: MoE v2 — routing by motion dynamics (TKEO energy, envelope slope, kurtosis, ZCR) with LayerNorm router and TDNN experts of varying dilation outperforms subject-style MoE routing (exp_27) in cross-subject LOSO
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: moe_dynamic_routing
- **Epochs**: 50
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| moe_dynamic_routing | 28.06% | 25.70% | 0.0438 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | moe_dynamic_routing | 23.31% | 22.18% |
| DB2_s12 | moe_dynamic_routing | 23.42% | 21.47% |
| DB2_s15 | moe_dynamic_routing | 30.00% | 26.89% |
| DB2_s28 | moe_dynamic_routing | 34.98% | 31.21% |
| DB2_s39 | moe_dynamic_routing | 28.60% | 26.74% |

**vs Baseline**: BELOW best baseline by -3.9pp

---

### EXP_73: exp_73_cpc_ssl_pretrain_loso

- **Date**: 2026-02-24T11:43:04.465431
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: cpc_emg
- **Epochs**: 60
- **Batch size**: 256
- **Learning rate**: 0.0005
- **Window size**: 600

**Results:**

---

### EXP_74: exp_74_temporal_order_invariance_jigsaw_loso

- **Date**: 2026-02-24T11:47:54.400807
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: jigsaw_temporal_emg
- **Epochs**: 80
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

---

### EXP_75: exp_75_emg_patch_tokens_performer

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_76: exp_76_soft_agc_pcen_lite

- **Date**: 20260224_191400
- **Approach**: `deep_raw`
- **Hypothesis**: Softer AGC normalization (log+affine, local-RMS, or bounded-alpha EMA) reduces inter-subject amplitude variation without over-suppressing gesture-discriminative amplitude cues.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_77: exp_77_stochastic_hypernetwork_fir_deconv

- **Date**: 20260224_123423
- **Approach**: `deep_raw`
- **Hypothesis**: Domain randomization via noise-conditioned FIR hypernetwork. Backbone learns subject-invariant features by training on random filter realizations. Test-time: u=0, fully deterministic.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | stochastic_fir_cnn_gru | 27.07% | 24.60% |
| DB2_s12 | stochastic_fir_cnn_gru | 28.41% | 26.97% |
| DB2_s15 | stochastic_fir_cnn_gru | 33.75% | 32.75% |
| DB2_s28 | stochastic_fir_cnn_gru | 42.14% | 37.71% |
| DB2_s39 | stochastic_fir_cnn_gru | 35.65% | 35.95% |

**vs Baseline**: ABOVE best baseline by +10.1pp

---

### EXP_78: exp_78_cpsd_riemannian_spectral_loso

- **Date**: 2026-02-24T20:28:14.885112
- **Approach**: `Per-band CPSD (Hermitian) → real SPD → log-Euclidean Riemannian tangent features → SVM`
- **Hypothesis**: CPSD (Cross-Power Spectral Density) matrices in frequency bands, converted to real SPD and projected via log-Euclidean Riemannian tangent space, capture cross-subject-invariant muscle coupling structure.  The spectral inter-channel interaction pattern is more stable across subjects than broadband time-domain covariance (cf. exp_63), because phase/coherence is independent of absolute signal amplitude.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| cpsd_real_combined_svm_rbf | 29.83% | 29.18% | 0.0577 | 5 |
| cpsd_real_4band_svm_rbf | 28.38% | 26.84% | 0.0433 | 5 |
| cpsd_block_4band_svm_rbf | 27.00% | 25.05% | 0.0521 | 5 |
| cpsd_real_4band_svm_linear | 23.18% | 22.42% | 0.0506 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 25.45% | 24.88% |
| DB2_s12 | ? | 22.35% | 20.84% |
| DB2_s15 | ? | 33.99% | 31.25% |
| DB2_s28 | ? | 32.48% | 30.76% |
| DB2_s39 | ? | 27.62% | 26.49% |
| DB2_s1 | ? | 24.14% | 22.60% |
| DB2_s12 | ? | 13.97% | 14.18% |
| DB2_s15 | ? | 27.92% | 26.46% |
| DB2_s28 | ? | 27.53% | 26.92% |
| DB2_s39 | ? | 22.32% | 21.97% |
| DB2_s1 | ? | 24.97% | 23.99% |
| DB2_s12 | ? | 20.27% | 18.38% |
| DB2_s15 | ? | 34.58% | 31.50% |
| DB2_s28 | ? | 31.32% | 28.96% |
| DB2_s39 | ? | 23.84% | 22.40% |
| DB2_s1 | ? | 25.86% | 25.81% |
| DB2_s12 | ? | 27.59% | 27.87% |
| DB2_s15 | ? | 34.94% | 32.65% |
| DB2_s28 | ? | 38.10% | 37.48% |
| DB2_s39 | ? | 22.68% | 22.11% |

**vs Baseline**: BELOW best baseline by -2.2pp

---

### EXP_79: exp_79_subcenter_arcface_emg

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_80: exp_80_synthetic_env_vrex_fishr_loso

- **Date**: 2026-02-25T21:50:56.907614
- **Approach**: ``
- **Hypothesis**: h-080-synthetic-env-vrex-fishr
- **Note**: exp_80 (synthetic transform environments) vs exp_69 (subject-based environments) — same backbone and penalty formulas, different environment source.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: irm_content_style_emg
- **Epochs**: 60
- **Batch size**: 256
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| subject_transform/vrex | 28.04% | 26.47% | 0.0406 | 5 |
| transforms_only/fishr | 27.90% | 26.67% | 0.0486 | 5 |
| transforms_only/vrex | 26.86% | 25.74% | 0.0446 | 5 |
| subject_transform/fishr | 26.62% | 25.45% | 0.0441 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | irm_content_style_emg | 22.77% | 21.76% |
| DB2_s12 | irm_content_style_emg | 26.93% | 26.79% |
| DB2_s15 | irm_content_style_emg | 30.00% | 30.51% |
| DB2_s28 | irm_content_style_emg | 33.33% | 29.33% |
| DB2_s39 | irm_content_style_emg | 21.28% | 20.33% |
| DB2_s1 | irm_content_style_emg | 22.89% | 21.36% |
| DB2_s12 | irm_content_style_emg | 27.41% | 26.36% |
| DB2_s15 | irm_content_style_emg | 29.64% | 29.14% |
| DB2_s28 | irm_content_style_emg | 36.20% | 33.37% |
| DB2_s39 | irm_content_style_emg | 23.35% | 23.11% |
| DB2_s1 | irm_content_style_emg | 24.91% | 23.57% |
| DB2_s12 | irm_content_style_emg | 26.46% | 25.63% |
| DB2_s15 | irm_content_style_emg | 29.70% | 28.08% |
| DB2_s28 | irm_content_style_emg | 35.16% | 31.26% |
| DB2_s39 | irm_content_style_emg | 23.96% | 23.81% |
| DB2_s1 | irm_content_style_emg | 24.44% | 22.86% |
| DB2_s12 | irm_content_style_emg | 25.74% | 25.40% |
| DB2_s15 | irm_content_style_emg | 26.96% | 26.87% |
| DB2_s28 | irm_content_style_emg | 34.62% | 31.41% |
| DB2_s39 | irm_content_style_emg | 21.34% | 20.72% |

**vs Baseline**: BELOW best baseline by -4.0pp

---

### EXP_81: exp_81_ortho_channel_mix_ecapa

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_82: exp_82_vmd_imf_decomposition

- **Date**: N/A
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_83: exp_83_subject_cluster_models_loso

- **Date**: 2026-02-26T21:15:37.790840
- **Approach**: `ml_emg_td`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 0.00% | 0.00% | 0.0000 | ? |
| svm_linear | 0.00% | 0.00% | 0.0000 | ? |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 0.00% | 0.00% |
| DB2_s12 | ? | 0.00% | 0.00% |
| DB2_s15 | ? | 0.00% | 0.00% |
| DB2_s28 | ? | 0.00% | 0.00% |
| DB2_s39 | ? | 0.00% | 0.00% |
| DB2_s1 | ? | 0.00% | 0.00% |
| DB2_s12 | ? | 0.00% | 0.00% |
| DB2_s15 | ? | 0.00% | 0.00% |
| DB2_s28 | ? | 0.00% | 0.00% |
| DB2_s39 | ? | 0.00% | 0.00% |

---

### EXP_84: exp_84_marginal_kurtosis_frequency_band_selection_loso

- **Date**: 2026-02-26T22:07:09.365236
- **Approach**: `MK band selection (training-only) → spectral filtering → PowerfulFeatureExtractor + MK spectral features → SVM / LGBM`
- **Hypothesis**: Marginal Kurtosis (MK) identifies frequency bands with non-Gaussian (bursty) power fluctuations across training windows, characteristic of gesture-related MUAP activity. Filtering to retain only high-MK bands removes noise and focuses feature extraction on informative content.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: lgbm
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 26.11% | 25.18% | 0.0463 | 5 |
| lgbm | 26.11% | 25.18% | 0.0463 | 5 |
| svm_linear | 23.15% | 21.87% | 0.0558 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 23.13% | 22.06% |
| DB2_s12 | svm_rbf | 22.41% | 23.03% |
| DB2_s15 | svm_rbf | 28.99% | 26.89% |
| DB2_s28 | svm_rbf | 33.88% | 32.15% |
| DB2_s39 | svm_rbf | 22.13% | 21.76% |
| DB2_s1 | svm_linear | 14.92% | 14.54% |
| DB2_s12 | svm_linear | 19.14% | 18.48% |
| DB2_s15 | svm_linear | 26.85% | 25.37% |
| DB2_s28 | svm_linear | 30.71% | 28.48% |
| DB2_s39 | svm_linear | 24.15% | 22.48% |
| DB2_s1 | lgbm | 23.13% | 22.06% |
| DB2_s12 | lgbm | 22.41% | 23.03% |
| DB2_s15 | lgbm | 28.99% | 26.89% |
| DB2_s28 | lgbm | 33.88% | 32.15% |
| DB2_s39 | lgbm | 22.13% | 21.76% |

**vs Baseline**: BELOW best baseline by -5.9pp

---

### EXP_85: exp_85_wca_condition_attention_cnn_gru_loso

- **Date**: 2026-02-27T21:30:39.524796
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: wca_condition_attention
- **Epochs**: 60
- **Batch size**: 1024
- **Learning rate**: 0.0005
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | wca_condition_attention | 29.31% | 28.67% |
| DB2_s12 | wca_condition_attention | 23.19% | 21.43% |
| DB2_s15 | wca_condition_attention | 28.27% | 27.19% |
| DB2_s28 | wca_condition_attention | 34.13% | 31.09% |
| DB2_s39 | wca_condition_attention | 25.12% | 24.85% |

---

### EXP_86: exp_86_knowledge_base_similarity_ensemble_loso

- **Date**: 2026-02-27T21:42:17.987283
- **Approach**: `ml_emg_td`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: simple_cnn
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| rbf | 0.00% | 0.00% | 0.0000 | ? |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | ? | 0.00% | 0.00% |
| DB2_s12 | ? | 0.00% | 0.00% |
| DB2_s15 | ? | 0.00% | 0.00% |
| DB2_s28 | ? | 0.00% | 0.00% |
| DB2_s39 | ? | 0.00% | 0.00% |

---

### EXP_87: exp_87_tidwt_denoising_preprocessing_loso

- **Date**: 2026-02-27T22:09:14.750831
- **Approach**: `TIDWT_denoising + PowerfulFeatures + SVM`
- **Hypothesis**: SWT (shift-invariant undecimated DWT) denoising of raw EMG windows before feature extraction improves cross-subject classification by removing noise while preserving gesture-discriminative signal structure.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: svm_linear
- **Epochs**: 1
- **Batch size**: 4096
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|
| svm_rbf | 29.75% | 28.91% | 0.0596 | 5 |
| svm_linear | 25.24% | 23.87% | 0.0599 | 5 |

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | svm_rbf | 25.80% | 25.26% |
| DB2_s12 | svm_rbf | 30.32% | 30.61% |
| DB2_s15 | svm_rbf | 32.20% | 29.35% |
| DB2_s28 | svm_rbf | 39.01% | 38.44% |
| DB2_s39 | svm_rbf | 21.40% | 20.85% |
| DB2_s1 | svm_linear | 21.28% | 20.62% |
| DB2_s12 | svm_linear | 17.18% | 16.07% |
| DB2_s15 | svm_linear | 27.26% | 26.37% |
| DB2_s28 | svm_linear | 34.98% | 31.97% |
| DB2_s39 | svm_linear | 25.49% | 24.30% |

**vs Baseline**: BELOW best baseline by -2.3pp

---

### EXP_88: exp_88_causal_ecapa_disentanglement

- **Date**: 20260228_105330
- **Approach**: `deep_raw`
- **Hypothesis**: Causal ECAPA-TDNN content/style disentanglement
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | causal_ecapa_tdnn | 34.38% | 31.16% |
| DB2_s12 | causal_ecapa_tdnn | 33.16% | 30.76% |
| DB2_s15 | causal_ecapa_tdnn | 24.30% | 22.28% |
| DB2_s28 | causal_ecapa_tdnn | 40.00% | 34.08% |
| DB2_s39 | causal_ecapa_tdnn | 34.19% | 34.32% |

**vs Baseline**: ABOVE best baseline by +8.0pp

---

### EXP_89: exp_89_selective_disentanglement_clip_dca

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_90: exp_90_hygdl_analytical_orthogonal_projection

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_91: exp_91_cyclic_inter_subject_adain_ecapa

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_92: exp_92_mrbt_channel_group_barlow_twins

- **Date**: N/A
- **Approach**: `deep_raw`
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_93: exp_93_unfolded_vmd_uvmd

- **Date**: N/A
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_94: exp_94_learnable_filterbank_grl

- **Date**: N/A
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_95: exp_95_vlmd_latent_mode_decomposition

- **Date**: N/A
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_96: exp_96_stvmd_learnable_seg_mode_aug

- **Date**: N/A
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_97: exp_97_filterbank_vib_information_bottleneck

- **Date**: N/A
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_98: exp_98_cyclemix_channel_wise_stochastic

- **Date**: 20260303_195644
- **Approach**: `deep_raw`
- **Hypothesis**: H98: CycleMix channel-wise stochastic style mixing
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_99: exp_99_discrete_style_codebook_loso

- **Date**: 2026-02-28T23:57:29.093716
- **Approach**: `deep_raw`
- **Hypothesis**: h-099-discrete-style-codebook
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)
- **Model type**: discrete_style_codebook
- **Epochs**: 60
- **Batch size**: 512
- **Learning rate**: 0.001
- **Window size**: 600

**Results:**

| Model | Mean Acc | Mean F1 | Std Acc | N Subjects |
|-------|---------|---------|---------|------------|

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | discrete_style_codebook | 10.05% | 1.83% |
| DB2_s12 | discrete_style_codebook | 11.71% | 4.57% |
| DB2_s15 | discrete_style_codebook | 10.00% | 1.82% |
| DB2_s28 | discrete_style_codebook | 10.07% | 1.83% |
| DB2_s39 | discrete_style_codebook | 10.06% | 1.83% |

---

### EXP_100: exp_100_dual_stream_hard_style_augmentation

- **Date**: 20260301_091054
- **Approach**: `deep_raw`
- **Hypothesis**: H100: Dual-Stream Hard Style Augmentation with FGSM adversarial perturbation
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | dual_stream_hard_style_cnn_gru | 32.00% | 29.11% |
| DB2_s12 | dual_stream_hard_style_cnn_gru | 35.05% | 32.75% |
| DB2_s15 | dual_stream_hard_style_cnn_gru | 32.32% | 30.60% |
| DB2_s28 | dual_stream_hard_style_cnn_gru | 44.28% | 37.83% |
| DB2_s39 | dual_stream_hard_style_cnn_gru | 42.80% | 41.93% |

**vs Baseline**: ABOVE best baseline by +12.3pp

---

### EXP_101: exp_101_xdomain_mix_4component_decomposition

- **Date**: 20260301_092840
- **Approach**: `deep_raw`
- **Hypothesis**: H101: XDomainMix — 4-Component Decomposition with Cross-Domain Recombination
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | xdomain_mix_emg | 34.74% | 30.85% |
| DB2_s12 | xdomain_mix_emg | 36.48% | 32.21% |
| DB2_s15 | xdomain_mix_emg | 30.36% | 29.80% |
| DB2_s28 | xdomain_mix_emg | 41.90% | 36.53% |
| DB2_s39 | xdomain_mix_emg | 34.37% | 35.05% |

**vs Baseline**: ABOVE best baseline by +9.9pp

---

### EXP_102: exp_102_freq_band_style_mixing

- **Date**: 20260304_183809
- **Approach**: `deep_raw`
- **Hypothesis**: H102: Frequency-Band Style Mixing (AdaIN per EMG band)
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | freq_band_style_mix_emg | 39.71% | 40.74% |
| DB2_s2 | freq_band_style_mix_emg | 31.11% | 28.24% |
| DB2_s3 | freq_band_style_mix_emg | 30.09% | 31.02% |
| DB2_s4 | freq_band_style_mix_emg | 34.27% | 32.40% |
| DB2_s5 | freq_band_style_mix_emg | 38.71% | 37.26% |
| DB2_s11 | freq_band_style_mix_emg | 41.61% | 40.01% |
| DB2_s12 | freq_band_style_mix_emg | 35.79% | 32.93% |
| DB2_s13 | freq_band_style_mix_emg | 39.50% | 38.39% |
| DB2_s14 | freq_band_style_mix_emg | 37.08% | 37.65% |
| DB2_s15 | freq_band_style_mix_emg | 33.57% | 33.02% |
| DB2_s26 | freq_band_style_mix_emg | 32.31% | 31.86% |
| DB2_s27 | freq_band_style_mix_emg | 41.17% | 39.49% |
| DB2_s28 | freq_band_style_mix_emg | 46.40% | 44.45% |
| DB2_s29 | freq_band_style_mix_emg | 30.88% | 28.14% |
| DB2_s30 | freq_band_style_mix_emg | 49.42% | 48.31% |
| DB2_s36 | freq_band_style_mix_emg | 36.43% | 34.52% |
| DB2_s37 | freq_band_style_mix_emg | 22.07% | 18.98% |
| DB2_s38 | freq_band_style_mix_emg | 41.18% | 40.01% |
| DB2_s39 | freq_band_style_mix_emg | 42.74% | 42.82% |
| DB2_s40 | freq_band_style_mix_emg | 45.32% | 44.33% |

**vs Baseline**: ABOVE best baseline by +17.4pp

---

### EXP_103: exp_103_synth_env_groupdro

- **Date**: 20260301_102228
- **Approach**: `deep_raw`
- **Hypothesis**: H103: Synthetic Environment Expansion + Soft GroupDRO
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | synth_env_groupdro_emg | 32.00% | 28.45% |
| DB2_s12 | synth_env_groupdro_emg | 32.80% | 29.05% |
| DB2_s15 | synth_env_groupdro_emg | 21.69% | 20.04% |
| DB2_s28 | synth_env_groupdro_emg | 41.71% | 37.65% |
| DB2_s39 | synth_env_groupdro_emg | 42.86% | 41.65% |

**vs Baseline**: ABOVE best baseline by +10.9pp

---

### EXP_104: exp_104_causal_barlow_groupdro

- **Date**: 20260301_104013
- **Approach**: `deep_raw`
- **Hypothesis**: Causal Disentanglement (CDDG) + Barlow Twins + GroupDRO
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | causal_barlow_cnn_gru | 33.91% | 30.80% |
| DB2_s12 | causal_barlow_cnn_gru | 34.52% | 31.08% |
| DB2_s15 | causal_barlow_cnn_gru | 34.52% | 31.79% |
| DB2_s28 | causal_barlow_cnn_gru | 41.53% | 35.41% |
| DB2_s39 | causal_barlow_cnn_gru | 38.77% | 39.14% |

**vs Baseline**: ABOVE best baseline by +9.5pp

---

### EXP_105: exp_105_dsfe_style_bank_exploitation

- **Date**: 20260301_111603
- **Approach**: `deep_raw`
- **Hypothesis**: DSFE: Domain-Specific Feature Exploitation via Style Bank
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | dsfe_style_bank_cnn_gru | 33.85% | 30.60% |
| DB2_s12 | dsfe_style_bank_cnn_gru | 34.34% | 31.02% |
| DB2_s15 | dsfe_style_bank_cnn_gru | 36.24% | 33.57% |
| DB2_s28 | dsfe_style_bank_cnn_gru | 38.59% | 34.32% |
| DB2_s39 | dsfe_style_bank_cnn_gru | 41.33% | 40.03% |

**vs Baseline**: ABOVE best baseline by +9.3pp

---

### EXP_106: exp_106_channel_contrastive_disentanglement_per_ch_dro

- **Date**: 20260303_222358
- **Approach**: `deep_raw`
- **Hypothesis**: H9: Channel-wise Contrastive Disentanglement + Per-Channel DRO
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | channel_contrastive_disentangled | 42.45% | 39.58% |
| DB2_s2 | channel_contrastive_disentangled | 40.05% | 37.65% |
| DB2_s3 | channel_contrastive_disentangled | 22.27% | 21.08% |
| DB2_s4 | channel_contrastive_disentangled | 41.78% | 39.70% |
| DB2_s5 | channel_contrastive_disentangled | 36.15% | 35.70% |
| DB2_s11 | channel_contrastive_disentangled | 38.15% | 36.49% |
| DB2_s12 | channel_contrastive_disentangled | 38.29% | 37.66% |
| DB2_s13 | channel_contrastive_disentangled | 39.74% | 38.77% |
| DB2_s14 | channel_contrastive_disentangled | 36.60% | 34.04% |
| DB2_s15 | channel_contrastive_disentangled | 35.83% | 33.27% |
| DB2_s27 | channel_contrastive_disentangled | 43.91% | 43.67% |
| DB2_s28 | channel_contrastive_disentangled | 46.21% | 44.71% |
| DB2_s29 | channel_contrastive_disentangled | 30.39% | 27.50% |
| DB2_s30 | channel_contrastive_disentangled | 41.54% | 40.27% |
| DB2_s36 | channel_contrastive_disentangled | 36.55% | 38.08% |
| DB2_s37 | channel_contrastive_disentangled | 19.30% | 12.62% |
| DB2_s38 | channel_contrastive_disentangled | 36.79% | 36.39% |
| DB2_s39 | channel_contrastive_disentangled | 38.60% | 36.63% |
| DB2_s40 | channel_contrastive_disentangled | 43.05% | 42.34% |

**vs Baseline**: ABOVE best baseline by +14.2pp

---

### EXP_107: exp_107_progressive_env_diversification_adaptive_dro

- **Date**: 20260301_115542
- **Approach**: `deep_raw`
- **Hypothesis**: H107: Progressive Environment Diversification with Adaptive DRO — phased training (disentanglement -> MixStyle+DRO -> extrap+aggressive DRO), adaptive eta, anti-collapse
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | progressive_env_dro | 33.91% | 30.06% |
| DB2_s12 | progressive_env_dro | 34.58% | 31.79% |
| DB2_s15 | progressive_env_dro | 33.39% | 31.12% |
| DB2_s28 | progressive_env_dro | 40.67% | 34.86% |
| DB2_s39 | progressive_env_dro | 42.67% | 41.18% |

**vs Baseline**: ABOVE best baseline by +10.7pp

---

### EXP_108: exp_108_hierarchical_beta_vae_uvmd

- **Date**: 20260301_134036
- **Approach**: `deep_raw`
- **Hypothesis**: H1_new: Hierarchical β-VAE + UVMD + SoftAGC
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | hierarchical_beta_vae_uvmd | 12.14% | 4.91% |
| DB2_s12 | hierarchical_beta_vae_uvmd | 15.24% | 9.56% |
| DB2_s15 | hierarchical_beta_vae_uvmd | 9.51% | 5.82% |
| DB2_s28 | hierarchical_beta_vae_uvmd | 17.25% | 10.65% |
| DB2_s39 | hierarchical_beta_vae_uvmd | 13.55% | 6.55% |

**vs Baseline**: BELOW best baseline by -14.8pp

---

### EXP_109: exp_109_mate_kronecker_shared_specific

- **Date**: 20260301_135424
- **Approach**: `deep_raw`
- **Hypothesis**: MATE Shared-Specific without orthogonality: shared and specific representations allowed to be correlated; only z_shared is adversarially regularised against subject ID.
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | mate_kronecker | 38.13% | 34.03% |
| DB2_s12 | mate_kronecker | 35.88% | 34.33% |
| DB2_s15 | mate_kronecker | 27.57% | 25.19% |
| DB2_s28 | mate_kronecker | 49.60% | 43.47% |
| DB2_s39 | mate_kronecker | 38.77% | 37.90% |

**vs Baseline**: ABOVE best baseline by +17.6pp

---

### EXP_110: exp_110_multi_res_aligned_disentangle

- **Date**: 20260304_184123
- **Approach**: `deep_raw`
- **Hypothesis**: H3: Multi-Resolution Aligned Disentanglement
- **Subjects**: 20 (DB2_s1, DB2_s2, DB2_s3, DB2_s4, DB2_s5...)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | multi_res_aligned_disentangle | 42.69% | 39.32% |
| DB2_s2 | multi_res_aligned_disentangle | 34.33% | 31.34% |
| DB2_s3 | multi_res_aligned_disentangle | 28.24% | 27.08% |
| DB2_s4 | multi_res_aligned_disentangle | 41.72% | 39.40% |
| DB2_s5 | multi_res_aligned_disentangle | 37.64% | 36.09% |
| DB2_s11 | multi_res_aligned_disentangle | 41.01% | 39.85% |
| DB2_s12 | multi_res_aligned_disentangle | 32.34% | 32.07% |
| DB2_s13 | multi_res_aligned_disentangle | 38.61% | 36.81% |
| DB2_s14 | multi_res_aligned_disentangle | 36.42% | 35.33% |
| DB2_s15 | multi_res_aligned_disentangle | 31.85% | 30.22% |
| DB2_s27 | multi_res_aligned_disentangle | 40.07% | 39.57% |
| DB2_s28 | multi_res_aligned_disentangle | 48.05% | 46.27% |
| DB2_s29 | multi_res_aligned_disentangle | 29.48% | 27.43% |
| DB2_s30 | multi_res_aligned_disentangle | 50.64% | 49.36% |
| DB2_s36 | multi_res_aligned_disentangle | 37.90% | 37.20% |
| DB2_s37 | multi_res_aligned_disentangle | 16.16% | 12.37% |
| DB2_s38 | multi_res_aligned_disentangle | 36.73% | 36.93% |
| DB2_s39 | multi_res_aligned_disentangle | 37.01% | 36.83% |
| DB2_s40 | multi_res_aligned_disentangle | 41.70% | 39.65% |

**vs Baseline**: ABOVE best baseline by +18.6pp

---

### EXP_111: exp_111_filterbank_mode_adaptive_ecapa

- **Date**: N/A
- **Approach**: ``
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

---

### EXP_112: exp_112_channel_band_tucker_consensus

- **Date**: 20260301_201853
- **Approach**: `deep_raw`
- **Hypothesis**: Three-axis Tucker factorization of EMG spectrogram: channel (GRL adversary), frequency (Soft AGC), temporal (quarter-window KL consensus).
- **Subjects**: 5 (DB2_s1, DB2_s12, DB2_s15, DB2_s28, DB2_s39)

**Results:**

**Per-subject results:**

| Subject | Model | Accuracy | F1 |
|---------|-------|----------|-----|
| DB2_s1 | tucker_consensus | 35.04% | 33.24% |
| DB2_s12 | tucker_consensus | 34.10% | 32.20% |
| DB2_s15 | tucker_consensus | 33.75% | 33.12% |
| DB2_s28 | tucker_consensus | 38.17% | 33.39% |
| DB2_s39 | tucker_consensus | 31.93% | 31.61% |

**vs Baseline**: ABOVE best baseline by +6.2pp

---

## Cross-Experiment Analysis

### Approaches ranked by best accuracy:

- **deep_raw**: 50.64% (multi_res_aligned_disentangle, exp_110_multi_res_aligned_disentangle)
- **few_shot_learning_with_maml_meta_training**: 38.25% (simple_cnn_few_shot, exp_13_leveraging_subject_specific_calibration_via_few_sh_loso)
- **ml_emg_td**: 35.24% (svm_linear, exp4_svm_linear_powerful_loso)
- **CombinedCyclostationaryExtractor (PowerfulFeatureExtractor + CyclostationaryEMGExtractor) → z-score → PCA (train only) → SVM/LightGBM**: 30.72% (svm_rbf, exp_67_cyclostationary_spectral_correlation_loso)
- **RiemannianSPDExtractor (log-Euclidean mean) + SVM**: 30.28% (combined_svm_rbf, exp_63_riemannian_spd_covariance_loso)
- **CombinedNonlinearExtractor + SVM / LightGBM**: 30.07% (svm_rbf, exp_38_nonlinear_stats_svm_lgbm_loso)
- **Per-band CPSD (Hermitian) → real SPD → log-Euclidean Riemannian tangent features → SVM**: 29.83% (cpsd_real_combined_svm_rbf, exp_78_cpsd_riemannian_spectral_loso)
- **TIDWT_denoising + PowerfulFeatures + SVM**: 29.75% (svm_rbf, exp_87_tidwt_denoising_preprocessing_loso)
- **One-vs-Rest Multiclass CSP (+ optional FilterBank) + SVM. CSP filters fitted on train subjects only (LOSO safe).**: 28.75% (fbcsp_svm_rbf, exp_64_multiclass_csp_filterbank_loso)
- **unknown**: 28.70% (vrex, exp_69_vrex_fishr_irm_v2_loso)
- **MultitaperPSDExtractor + PowerfulFeatureExtractor + SVM / LGBM**: 28.52% (svm_rbf, exp_68_multitaper_psd_spectral_slope_loso)
- **wavelet_scattering + SVM**: 27.60% (svm_rbf, exp_33_wavelet_scattering_svm_loso)
- **spectral_band_ratio + SVM**: 27.40% (svm_rbf, exp_32_spectral_band_ratio_svm_loso)
- **MK band selection (training-only) → spectral filtering → PowerfulFeatureExtractor + MK spectral features → SVM / LGBM**: 26.11% (svm_rbf, exp_84_marginal_kurtosis_frequency_band_selection_loso)
- **powerful**: 26.08% (svm_linear_powerful_triple_aug, exp_25_svm_linear_on_powerful_features_with_combined_nois_loso)
- **deep_emg_seq**: 25.46% (cnn_lstm, exp_2_deep_emg_td_seq_cnn_lstm_loso)
- **emgcc+powerful**: 24.92% (svm_rbf, exp_49_emg_cepstral_coefficients_muscle_filterbanks_loso)
- **rank_copula+powerful**: 20.54% (svm_rbf, exp_51_rank_copula_features_monotone_invariant_loso)
- **powerful+ot_alignment**: 13.08% (svm_rbf, exp_55_optimal_transport_barycenter_alignment_loso)
- **hybrid_powerful_deep**: 10.74% (hybrid_powerful_deep, exp5_hybrid_powerful_deep_loso)
- **deep_powerful**: 10.27% (mlp_powerful, exp3_deep_powerful_mlp_powerful_loso)

## Conclusions

1. **150 experiments** were conducted in total
2. **27 experiments** exceeded the best baseline of 32.0%
3. Best result: **50.64%** accuracy (multi_res_aligned_disentangle, exp_110_multi_res_aligned_disentangle)
4. Second best: **49.60%** (mate_kronecker, exp_109_mate_kronecker_shared_specific)
