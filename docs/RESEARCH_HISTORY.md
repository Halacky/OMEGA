# OMEGA Research History

**Last updated**: 2026-03-18

Comprehensive record of 150 LOSO cross-subject experiments (303 model evaluations).

---

## Leaderboard

### Full LOSO (≥15 subjects)

| # | Experiment | Model | Acc | F1 | Std | N |
|---|-----------|-------|-----|-----|-----|---|
| 1 | exp_4 | svm_linear | 35.24% | 32.50% | 0.0870 | 20 |
| 2 | exp_4 | svm_rbf | 34.46% | 32.60% | 0.0701 | 20 |
| 3 | exp_4 | rf | 32.00% | 30.33% | 0.0632 | 20 |
| 4 | exp_1 | cnn_gru_attention | 30.85% | 28.19% | 0.0645 | 20 |
| 5 | exp_1 | simple_cnn | 29.30% | 25.51% | 0.0779 | 20 |
| 6 | exp_1 | bigru | 28.60% | 25.43% | 0.0663 | 20 |
| 7 | exp_1 | bilstm_attention | 28.10% | 24.88% | 0.0622 | 20 |
| 8 | exp_1 | attention_cnn | 27.33% | 23.70% | 0.0641 | 20 |
| 9 | exp_1 | tcn | 26.91% | 23.57% | 0.0779 | 20 |
| 10 | exp_1 | multiscale_cnn | 26.81% | 23.44% | 0.0571 | 20 |
| 11 | exp_2 | bigru | 25.15% | 20.67% | 0.0659 | 20 |
| 12 | exp_2 | bilstm_attention | 25.10% | 19.99% | 0.0540 | 20 |
| 13 | exp_1 | simple_cnn | 23.14% | 19.06% | 0.0540 | 40 |
| 14 | exp_1 | tcn_attn | 22.92% | 17.82% | 0.1128 | 20 |
| 15 | exp_2 | bilstm | 21.77% | 16.81% | 0.0474 | 20 |
| 16 | exp_1 | attention_cnn | 21.68% | 18.30% | 0.0579 | 40 |
| 17 | exp_2 | tcn_attn | 21.57% | 15.98% | 0.0513 | 20 |
| 18 | exp_1 | bilstm | 19.32% | 13.97% | 0.0915 | 20 |
| 19 | exp_2 | cnn_gru_attention | 15.27% | 8.83% | 0.0342 | 20 |
| 20 | exp_2 | cnn_lstm | 13.93% | 7.13% | 0.0389 | 20 |
| 21 | exp_2 | multiscale_cnn | 11.99% | 5.09% | 0.0213 | 20 |
| 22 | exp_2 | simple_cnn | 11.44% | 5.25% | 0.0232 | 20 |
| 23 | exp_5 | hybrid_powerful_deep | 10.74% | 2.67% | 0.0142 | 20 |
| 24 | exp_3 | mlp_powerful | 10.27% | 2.21% | 0.0117 | 20 |
| 25 | exp_2 | attention_cnn | 10.00% | 4.90% | 0.0370 | 20 |

### CI Subset LOSO

| # | Experiment | Model | Acc | F1 | Std | N |
|---|-----------|-------|-----|-----|-----|---|
| 1 | exp_110 | multi_res_aligned_disentangle | 50.64% | 49.36% | 0.0000 | 5 |
| 2 | exp_109 | mate_kronecker | 49.60% | 43.47% | 0.0000 | 5 |
| 3 | exp_102 | freq_band_style_mix_emg | 49.42% | 48.31% | 0.0000 | 5 |
| 4 | exp_110 | multi_res_aligned_disentangle | 48.05% | 46.27% | 0.0000 | 5 |
| 5 | exp_102 | freq_band_style_mix_emg | 46.40% | 44.45% | 0.0000 | 5 |
| 6 | exp_106 | channel_contrastive_disentangled | 46.21% | 44.71% | 0.0000 | 5 |
| 7 | exp_102 | freq_band_style_mix_emg | 45.32% | 44.33% | 0.0000 | 5 |
| 8 | exp_65 | fir_deconv_cnn_gru | 45.08% | 40.21% | 0.0000 | 5 |
| 9 | exp_31 | disentangled_cnn_gru | 45.05% | 44.33% | 0.0000 | 5 |
| 10 | exp_100 | dual_stream_hard_style_cnn_gru | 44.28% | 37.83% | 0.0000 | 5 |
| 11 | exp_106 | channel_contrastive_disentangled | 43.91% | 43.67% | 0.0000 | 5 |
| 12 | exp_106 | channel_contrastive_disentangled | 43.05% | 42.34% | 0.0000 | 5 |
| 13 | exp_60 | mixstyle_disentangled_cnn_gru | 43.04% | 41.44% | 0.0000 | 5 |
| 14 | exp_103 | synth_env_groupdro_emg | 42.86% | 41.65% | 0.0000 | 5 |
| 15 | exp_59 | proto_disentangled_cnn_gru | 42.86% | 43.00% | 0.0000 | 5 |
| 16 | exp_100 | dual_stream_hard_style_cnn_gru | 42.80% | 41.93% | 0.0000 | 5 |
| 17 | exp_102 | freq_band_style_mix_emg | 42.74% | 42.82% | 0.0000 | 5 |
| 18 | exp_110 | multi_res_aligned_disentangle | 42.69% | 39.32% | 0.0000 | 5 |
| 19 | exp_107 | progressive_env_dro | 42.67% | 41.18% | 0.0000 | 5 |
| 20 | exp_106 | channel_contrastive_disentangled | 42.45% | 39.58% | 0.0000 | 5 |
| 21 | exp_31 | disentangled_cnn_gru | 42.39% | 37.20% | 0.0000 | 5 |
| 22 | exp_77 | stochastic_fir_cnn_gru | 42.14% | 37.71% | 0.0000 | 5 |
| 23 | exp_101 | xdomain_mix_emg | 41.90% | 36.53% | 0.0000 | 5 |
| 24 | exp_60 | mixstyle_disentangled_cnn_gru | 41.90% | 37.44% | 0.0000 | 5 |
| 25 | exp_59 | proto_disentangled_cnn_gru | 41.90% | 34.77% | 0.0000 | 5 |
| 26 | exp_106 | channel_contrastive_disentangled | 41.78% | 39.70% | 0.0000 | 5 |
| 27 | exp_110 | multi_res_aligned_disentangle | 41.72% | 39.40% | 0.0000 | 5 |
| 28 | exp_103 | synth_env_groupdro_emg | 41.71% | 37.65% | 0.0000 | 5 |
| 29 | exp_110 | multi_res_aligned_disentangle | 41.70% | 39.65% | 0.0000 | 5 |
| 30 | exp_102 | freq_band_style_mix_emg | 41.61% | 40.01% | 0.0000 | 5 |
| 31 | exp_106 | channel_contrastive_disentangled | 41.54% | 40.27% | 0.0000 | 5 |
| 32 | exp_104 | causal_barlow_cnn_gru | 41.53% | 35.41% | 0.0000 | 5 |
| 33 | exp_57 | groupdro_disentangled_cnn_gru | 41.47% | 35.78% | 0.0000 | 5 |
| 34 | exp_105 | dsfe_style_bank_cnn_gru | 41.33% | 40.03% | 0.0000 | 5 |
| 35 | exp_41 | content_style_graph | 41.28% | 36.64% | 0.0000 | 5 |
| 36 | exp_102 | freq_band_style_mix_emg | 41.18% | 40.01% | 0.0000 | 5 |
| 37 | exp_102 | freq_band_style_mix_emg | 41.17% | 39.49% | 0.0000 | 5 |
| 38 | exp_110 | multi_res_aligned_disentangle | 41.01% | 39.85% | 0.0000 | 5 |
| 39 | exp_18 | svm_rbf | 40.73% | 21.59% | 0.0991 | 5 |
| 40 | exp_57 | groupdro_disentangled_cnn_gru | 40.72% | 40.46% | 0.0000 | 5 |
| 41 | exp_107 | progressive_env_dro | 40.67% | 34.86% | 0.0000 | 5 |
| 42 | exp_44 | unknown | 40.24% | 35.44% | 0.0000 | 5 |
| 43 | exp_110 | multi_res_aligned_disentangle | 40.07% | 39.57% | 0.0000 | 5 |
| 44 | exp_106 | channel_contrastive_disentangled | 40.05% | 37.65% | 0.0000 | 5 |
| 45 | exp_88 | causal_ecapa_tdnn | 40.00% | 34.08% | 0.0000 | 5 |
| 46 | exp_106 | channel_contrastive_disentangled | 39.74% | 38.77% | 0.0000 | 5 |
| 47 | exp_102 | freq_band_style_mix_emg | 39.71% | 40.74% | 0.0000 | 5 |
| 48 | exp_102 | freq_band_style_mix_emg | 39.50% | 38.39% | 0.0000 | 5 |
| 49 | exp_66 | phase_align_cnn_gru | 39.08% | 34.13% | 0.0000 | 5 |
| 50 | exp_60 | mixstyle_disentangled_cnn_gru | 38.96% | 36.01% | 0.0000 | 5 |
| 51 | exp_57 | groupdro_disentangled_cnn_gru | 38.79% | 35.45% | 0.0000 | 5 |
| 52 | exp_109 | mate_kronecker | 38.77% | 37.90% | 0.0000 | 5 |
| 53 | exp_104 | causal_barlow_cnn_gru | 38.77% | 39.14% | 0.0000 | 5 |
| 54 | exp_102 | freq_band_style_mix_emg | 38.71% | 37.26% | 0.0000 | 5 |
| 55 | exp_110 | multi_res_aligned_disentangle | 38.61% | 36.81% | 0.0000 | 5 |
| 56 | exp_106 | channel_contrastive_disentangled | 38.60% | 36.63% | 0.0000 | 5 |
| 57 | exp_105 | dsfe_style_bank_cnn_gru | 38.59% | 34.32% | 0.0000 | 5 |
| 58 | exp_31 | disentangled_cnn_gru | 38.55% | 33.29% | 0.0000 | 5 |
| 59 | exp_106 | channel_contrastive_disentangled | 38.29% | 37.66% | 0.0000 | 5 |
| 60 | exp_13 | simple_cnn_few_shot | 38.25% | 6.13% | 0.0046 | 5 |
| 61 | exp_112 | tucker_consensus | 38.17% | 33.39% | 0.0000 | 5 |
| 62 | exp_106 | channel_contrastive_disentangled | 38.15% | 36.49% | 0.0000 | 5 |
| 63 | exp_109 | mate_kronecker | 38.13% | 34.03% | 0.0000 | 5 |
| 64 | exp_110 | multi_res_aligned_disentangle | 37.90% | 37.20% | 0.0000 | 5 |
| 65 | exp_34 | cnn_gru_attention | 37.74% | 32.90% | 0.0000 | 5 |
| 66 | exp_110 | multi_res_aligned_disentangle | 37.64% | 36.09% | 0.0000 | 5 |
| 67 | exp_20 | fusion_cnn_gru_attention | 37.63% | 14.80% | 0.0361 | 5 |
| 68 | exp_41 | content_style_graph | 37.18% | 36.58% | 0.0000 | 5 |
| 69 | exp_28 | film_subject_adaptive | 37.10% | 34.24% | 0.0000 | 5 |
| 70 | exp_102 | freq_band_style_mix_emg | 37.08% | 37.65% | 0.0000 | 5 |
| 71 | exp_110 | multi_res_aligned_disentangle | 37.01% | 36.83% | 0.0000 | 5 |
| 72 | exp_65 | fir_deconv_cnn_gru | 36.94% | 36.72% | 0.0000 | 5 |
| 73 | exp_106 | channel_contrastive_disentangled | 36.79% | 36.39% | 0.0000 | 5 |
| 74 | exp_110 | multi_res_aligned_disentangle | 36.73% | 36.93% | 0.0000 | 5 |
| 75 | exp_44 | unknown | 36.69% | 35.30% | 0.0000 | 5 |
| 76 | exp_106 | channel_contrastive_disentangled | 36.60% | 34.04% | 0.0000 | 5 |
| 77 | exp_106 | channel_contrastive_disentangled | 36.55% | 38.08% | 0.0000 | 5 |
| 78 | exp_101 | xdomain_mix_emg | 36.48% | 32.21% | 0.0000 | 5 |
| 79 | exp_102 | freq_band_style_mix_emg | 36.43% | 34.52% | 0.0000 | 5 |
| 80 | exp_110 | multi_res_aligned_disentangle | 36.42% | 35.33% | 0.0000 | 5 |
| 81 | exp_44 | unknown | 36.35% | 33.48% | 0.0000 | 5 |
| 82 | exp_105 | dsfe_style_bank_cnn_gru | 36.24% | 33.57% | 0.0000 | 5 |
| 83 | exp_106 | channel_contrastive_disentangled | 36.15% | 35.70% | 0.0000 | 5 |
| 84 | exp_66 | phase_align_cnn_gru | 36.02% | 35.72% | 0.0000 | 5 |
| 85 | exp_34 | cnn_gru_attention | 35.96% | 36.87% | 0.0000 | 5 |
| 86 | exp_109 | mate_kronecker | 35.88% | 34.33% | 0.0000 | 5 |
| 87 | exp_106 | channel_contrastive_disentangled | 35.83% | 33.27% | 0.0000 | 5 |
| 88 | exp_102 | freq_band_style_mix_emg | 35.79% | 32.93% | 0.0000 | 5 |
| 89 | exp_31 | disentangled_cnn_gru | 35.77% | 32.87% | 0.0000 | 5 |
| 90 | exp_59 | proto_disentangled_cnn_gru | 35.77% | 33.93% | 0.0000 | 5 |
| 91 | exp_77 | stochastic_fir_cnn_gru | 35.65% | 35.95% | 0.0000 | 5 |
| 92 | exp_44 | unknown | 35.59% | 31.91% | 0.0000 | 5 |
| 93 | exp_28 | film_subject_adaptive | 35.09% | 32.44% | 0.0000 | 5 |
| 94 | exp_100 | dual_stream_hard_style_cnn_gru | 35.05% | 32.75% | 0.0000 | 5 |
| 95 | exp_112 | tucker_consensus | 35.04% | 33.24% | 0.0000 | 5 |
| 96 | exp_65 | fir_deconv_cnn_gru | 34.88% | 33.49% | 0.0000 | 5 |
| 97 | exp_101 | xdomain_mix_emg | 34.74% | 30.85% | 0.0000 | 5 |
| 98 | exp_107 | progressive_env_dro | 34.58% | 31.79% | 0.0000 | 5 |
| 99 | exp_104 | causal_barlow_cnn_gru | 34.52% | 31.79% | 0.0000 | 5 |
| 100 | exp_104 | causal_barlow_cnn_gru | 34.52% | 31.08% | 0.0000 | 5 |
| 101 | exp_88 | causal_ecapa_tdnn | 34.38% | 31.16% | 0.0000 | 5 |
| 102 | exp_101 | xdomain_mix_emg | 34.37% | 35.05% | 0.0000 | 5 |
| 103 | exp_7 | cnn_gru_attention_fusion | 34.36% | 15.39% | 0.0647 | 5 |
| 104 | exp_60 | mixstyle_disentangled_cnn_gru | 34.34% | 31.84% | 0.0000 | 5 |
| 105 | exp_105 | dsfe_style_bank_cnn_gru | 34.34% | 31.02% | 0.0000 | 5 |
| 106 | exp_110 | multi_res_aligned_disentangle | 34.33% | 31.34% | 0.0000 | 5 |
| 107 | exp_102 | freq_band_style_mix_emg | 34.27% | 32.40% | 0.0000 | 5 |
| 108 | exp_88 | causal_ecapa_tdnn | 34.19% | 34.32% | 0.0000 | 5 |
| 109 | exp_112 | tucker_consensus | 34.10% | 32.20% | 0.0000 | 5 |
| 110 | exp_9 | dual_stream_cnn_gru_attention | 34.07% | 16.48% | 0.0750 | 5 |
| 111 | exp_107 | progressive_env_dro | 33.91% | 30.06% | 0.0000 | 5 |
| 112 | exp_104 | causal_barlow_cnn_gru | 33.91% | 30.80% | 0.0000 | 5 |
| 113 | exp_105 | dsfe_style_bank_cnn_gru | 33.85% | 30.60% | 0.0000 | 5 |
| 114 | exp_112 | tucker_consensus | 33.75% | 33.12% | 0.0000 | 5 |
| 115 | exp_77 | stochastic_fir_cnn_gru | 33.75% | 32.75% | 0.0000 | 5 |
| 116 | exp_102 | freq_band_style_mix_emg | 33.57% | 33.02% | 0.0000 | 5 |
| 117 | exp_107 | progressive_env_dro | 33.39% | 31.12% | 0.0000 | 5 |
| 118 | exp_66 | phase_align_cnn_gru | 33.37% | 30.33% | 0.0000 | 5 |
| 119 | exp_58 | mi_disentangled_cnn_gru | 33.21% | 29.43% | 0.0000 | 5 |
| 120 | exp_88 | causal_ecapa_tdnn | 33.16% | 30.76% | 0.0000 | 5 |

---

## Experiment Registry

#### EXP_1: exp1_deep_raw_attention_cnn_loso_1_12_15_28_39
- **Pipeline**: `deep_raw`
- **Models**: attention_cnn
- **Best**: 25.31% acc, 23.28% F1 (-6.7pp, 5 subj)

#### EXP_1: exp1_deep_raw_attention_cnn_loso_isolated
- **Pipeline**: `deep_raw`
- **Models**: attention_cnn
- **Best**: 21.68% acc, 18.30% F1 (-10.3pp, 40 subj)

#### EXP_1: exp1_deep_raw_attention_cnn_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: attention_cnn
- **Best**: 27.33% acc, 23.70% F1 (-4.7pp, 20 subj)

#### EXP_1: exp1_deep_raw_bigru_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: bigru
- **Best**: 28.60% acc, 25.43% F1 (-3.4pp, 20 subj)

#### EXP_1: exp1_deep_raw_bilstm_attention_loso
- **Pipeline**: `deep_raw`
- **Models**: bilstm_attention
- **Best**: 26.20% acc, 23.22% F1 (-5.8pp, 5 subj)

#### EXP_1: exp1_deep_raw_bilstm_attention_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: bilstm_attention
- **Best**: 28.10% acc, 24.88% F1 (-3.9pp, 20 subj)

#### EXP_1: exp1_deep_raw_bilstm_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: bilstm
- **Best**: 19.32% acc, 13.97% F1 (-12.7pp, 20 subj)

#### EXP_1: exp1_deep_raw_cnn_gru_attention_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: cnn_gru_attention
- **Best**: 30.85% acc, 28.19% F1 (-1.2pp, 20 subj)

#### EXP_1: exp1_deep_raw_cnn_loso_isolated
- **Pipeline**: `deep_raw`
- **Models**: simple_cnn
- **Best**: 23.14% acc, 19.06% F1 (-8.9pp, 40 subj)

#### EXP_1: exp1_deep_raw_cnn_lstm_loso
- **Pipeline**: `deep_raw`
- **Models**: cnn_lstm
- **Best**: 20.78% acc, 16.54% F1 (-11.2pp, 5 subj)

#### EXP_1: exp1_deep_raw_multiscale_cnn_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: multiscale_cnn
- **Best**: 26.81% acc, 23.44% F1 (-5.2pp, 20 subj)

#### EXP_1: exp1_deep_raw_simple_cnn_loso_1_12_15_28_39
- **Pipeline**: `deep_raw`
- **Models**: simple_cnn
- **Best**: 25.99% acc, 24.11% F1 (-6.0pp, 5 subj)

#### EXP_1: exp1_deep_raw_simple_cnn_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: simple_cnn
- **Best**: 29.30% acc, 25.51% F1 (-2.7pp, 20 subj)

#### EXP_1: exp1_deep_raw_tcn_attn_loso_1_12_15_28_39
- **Pipeline**: `deep_raw`
- **Models**: tcn_attn
- **Best**: 24.66% acc, 22.05% F1 (-7.3pp, 5 subj)

#### EXP_1: exp1_deep_raw_tcn_attn_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: tcn_attn
- **Best**: 22.92% acc, 17.82% F1 (-9.1pp, 20 subj)

#### EXP_1: exp1_deep_raw_tcn_loso_1_12_15_28_39
- **Pipeline**: `deep_raw`
- **Models**: tcn
- **Best**: 23.11% acc, 18.58% F1 (-8.9pp, 5 subj)

#### EXP_1: exp1_deep_raw_tcn_loso_isolated_v2
- **Pipeline**: `deep_raw`
- **Models**: tcn
- **Best**: 26.91% acc, 23.57% F1 (-5.1pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_attention_cnn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: attention_cnn
- **Best**: 10.00% acc, 4.90% F1 (-22.0pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_bigru_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: bigru
- **Best**: 25.15% acc, 20.67% F1 (-6.8pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_bilstm_attention_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: bilstm_attention
- **Best**: 25.10% acc, 19.99% F1 (-6.9pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_bilstm_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: bilstm
- **Best**: 21.77% acc, 16.81% F1 (-10.2pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_cnn_gru_attention_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: cnn_gru_attention
- **Best**: 15.27% acc, 8.83% F1 (-16.7pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_cnn_lstm_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: cnn_lstm
- **Best**: 13.93% acc, 7.13% F1 (-18.1pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_multiscale_cnn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: multiscale_cnn
- **Best**: 11.99% acc, 5.09% F1 (-20.0pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_simple_cnn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: simple_cnn
- **Best**: 11.44% acc, 5.25% F1 (-20.6pp, 20 subj)

#### EXP_2: exp2_deep_emg_td_seq_tcn_attn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: tcn_attn
- **Best**: 21.57% acc, 15.98% F1 (-10.4pp, 20 subj)

#### EXP_2: exp_2_deep_emg_td_seq_attention_cnn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: attention_cnn
- **Best**: 23.62% acc, 22.20% F1 (-8.4pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_bigru_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: bigru
- **Best**: 24.03% acc, 22.56% F1 (-8.0pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_bilstm_attention_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: bilstm_attention
- **Best**: 24.70% acc, 23.44% F1 (-7.3pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_bilstm_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: bilstm
- **Best**: 23.82% acc, 22.70% F1 (-8.2pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_cnn_gru_attention_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: cnn_gru_attention
- **Best**: 23.78% acc, 22.89% F1 (-8.2pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_cnn_lstm_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: cnn_lstm
- **Best**: 25.46% acc, 24.29% F1 (-6.5pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_multiscale_cnn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: multiscale_cnn
- **Best**: 23.41% acc, 22.44% F1 (-8.6pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_simple_cnn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: simple_cnn
- **Best**: 23.37% acc, 22.51% F1 (-8.6pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_tcn_attn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: tcn_attn
- **Best**: 23.50% acc, 22.42% F1 (-8.5pp, 5 subj)

#### EXP_2: exp_2_deep_emg_td_seq_tcn_loso
- **Pipeline**: `deep_emg_seq`
- **Models**: tcn
- **Best**: 24.63% acc, 23.29% F1 (-7.4pp, 5 subj)

#### EXP_3: exp3_deep_powerful_mlp_powerful_loso
- **Pipeline**: `deep_powerful`
- **Models**: mlp_powerful
- **Best**: 10.27% acc, 2.21% F1 (-21.7pp, 20 subj)

#### EXP_4: exp4_rf_powerful_loso
- **Pipeline**: `ml_emg_td`
- **Models**: rf
- **Best**: 32.00% acc, 30.33% F1 (-0.0pp, 20 subj)

#### EXP_4: exp4_svm_linear_powerful_loso
- **Pipeline**: `ml_emg_td`
- **Models**: svm_linear
- **Best**: 35.24% acc, 32.50% F1 (+3.2pp, 20 subj)

#### EXP_4: exp4_svm_rbf_powerful_loso
- **Pipeline**: `ml_emg_td`
- **Models**: svm_rbf
- **Best**: 34.46% acc, 32.60% F1 (+2.5pp, 20 subj)

#### EXP_4: exp_4_ml_powerful_loso
- **Pipeline**: `ml_emg_td`
- **Models**: svm_linear, rf, svm_rbf
- **Best**: 27.82% acc, 26.63% F1 (-4.2pp, 5 subj)

#### EXP_5: exp5_hybrid_powerful_deep_loso
- **Pipeline**: `hybrid_powerful_deep`
- **Models**: hybrid_powerful_deep
- **Best**: 10.74% acc, 2.67% F1 (-21.3pp, 20 subj)

#### EXP_7: exp_7_cnn_gru_attention_with_noise_and_time_warp_augment_loso
- **Hypothesis**: test-001
- **Pipeline**: `fusion_with_augmentation`
- **Models**: cnn_gru_attention_fusion
- **Best**: 34.36% acc, 15.39% F1 (+2.4pp, 5 subj)

#### EXP_8: exp_8_augmented_svm_with_time_domain_features_for_improv_loso
- **Hypothesis**: 264d86e7-9299-44ed-87ff-f0d8f622ad82
- **Pipeline**: `powerful`
- **Models**: svm_linear
- **Best**: 25.51% acc, 24.43% F1 (-6.5pp, 5 subj)

#### EXP_9: exp_9_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso
- **Pipeline**: `dual_stream`
- **Models**: dual_stream_cnn_gru_attention
- **Best**: 34.07% acc, 16.48% F1 (+2.1pp, 5 subj)

#### EXP_10: exp_10_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso
- **Hypothesis**: 0d519c5d-fa3f-44d1-b955-1ef224cc74e4
- **Pipeline**: `deep_raw`
- **Models**: dual_stream_cnn_gru_attention
- **Best**: 23.57% acc, 21.93% F1 (-8.4pp, 5 subj)

#### EXP_12: exp_12_augmented_svm_with_time_domain_features_for_improv_loso
- **Hypothesis**: 264d86e7-9299-44ed-87ff-f0d8f622ad82
- **Note**: SVM-linear on powerful time-domain features with data augmentation.
- **Pipeline**: `powerful`
- **Models**: svm_linear
- **Best**: 25.51% acc, 24.43% F1 (-6.5pp, 5 subj)

#### EXP_13: exp_13_leveraging_subject_specific_calibration_via_few_sh_loso
- **Hypothesis**: 3901a9cf-1112-4c14-9c68-baed76f94c28
- **Pipeline**: `few_shot_learning_with_maml_meta_training`
- **Models**: simple_cnn_few_shot
- **Best**: 38.25% acc, 6.13% F1 (+6.3pp, 5 subj)

#### EXP_17: exp_17_enhanced_augmentation_for_simple_cnn_on_raw_emg_si_loso
- **Hypothesis**: c7d68e78-1306-4585-a213-4675c9920b82
- **Pipeline**: `deep_raw`
- **Models**: simple_cnn
- **Best**: 25.25% acc, 22.45% F1 (-6.8pp, 5 subj)

#### EXP_18: exp_18_augmented_svm_with_feature_space_jitter_for_improv_loso
- **Hypothesis**: d7c44dc7-cc18-4669-8b30-9db935f75204
- **Pipeline**: `powerful`
- **Models**: svm_rbf
- **Best**: 40.73% acc, 21.59% F1 (+8.7pp, 5 subj)

#### EXP_19: exp_19_subject_specific_feature_calibration_for_reducing_loso
- **Hypothesis**: df248bc2-7fdc-44c3-a9ff-48e0110bd9bc
- **Pipeline**: `ml_emg_td_with_mmd_calibration`
- **Models**: calibrated_svm_linear
- **Best**: 18.84% acc, 18.32% F1 (-13.2pp, 5 subj)

#### EXP_20: exp_20_augmented_cnn_gru_attention_with_feature_fusion_fo_loso
- **Hypothesis**: e2f39b15-6605-42fc-ae5c-dcc0a0153653
- **Pipeline**: `hybrid_fusion_with_augmentation`
- **Models**: fusion_cnn_gru_attention
- **Best**: 37.63% acc, 14.80% F1 (+5.6pp, 5 subj)

#### EXP_21: exp_21_svm_rbf_with_noise_time_warp_augmentation_and_tune_loso
- **Hypothesis**: 04b951c7-774f-449b-9202-eaf2bceb6376
- **Note**: SVM-RBF with signal-level augmentation and increased regularization (C=1.0). Tests the missing cell in the SVM×augmentation matrix.
- **Pipeline**: `powerful`
- **Models**: svm_rbf
- **Best**: 19.44% acc, 16.49% F1 (-12.6pp, 5 subj)

#### EXP_22: exp_22_focal_loss_class_balanced_sampling_for_cnn_gru_att_loso
- **Hypothesis**: 212764c1-2890-41d7-ada1-07d9e3e6b76d
- **Pipeline**: `deep_raw_with_focal_loss`
- **Models**: cnn_gru_attention
- **Best**: 22.13% acc, 18.17% F1 (-9.9pp, 5 subj)

#### EXP_24: exp_24_cnn_gru_attention_on_raw_emg_with_class_weighted_l_loso
- **Hypothesis**: 2c875e39-5fe0-4026-aa67-f7772e46980e
- **Note**: Testing whether class-weighted loss prevents F1 collapse in augmented models. Baseline exp_1: 30.85% acc, 28.19% F1 (ratio 1.09). Fusion experiments (exp_7, exp_20) showed accuracy gains but F1 collapse. Key insight: addressing class imbalance in loss function, not architecture.
- **Pipeline**: `deep_raw`
- **Models**: cnn_gru_attention
- **Best**: 26.30% acc, 25.22% F1 (-5.7pp, 5 subj)

#### EXP_25: exp_25_svm_linear_on_powerful_features_with_combined_nois_loso
- **Hypothesis**: 3b1480c0-aa08-4cad-b277-966e3654c011
- **Pipeline**: `powerful`
- **Models**: svm_linear_powerful_triple_aug
- **Best**: 26.08% acc, 24.89% F1 (-5.9pp, 5 subj)

#### EXP_26: exp_26_test_time_bn_adaptation_for_cnn_g_loso
- **Hypothesis**: 9c0b2f84-8bf2-41f8-9df0-4187086fd96c
- **Pipeline**: `deep_raw`
- **Models**: cnn_gru_attention
- **Best**: 30.74% acc, 30.14% F1 (-1.3pp, 5 subj)

#### EXP_27: exp_27_moe_cnn_gru_attention_loso
- **Hypothesis**: H1: Subject-as-Domain MoE — signal-style gating over CNN-GRU-Attention experts
- **Pipeline**: `deep_raw`
- **Models**: moe_cnn_gru_attention
- **Best**: 26.33% acc, 24.71% F1 (-5.7pp, 5 subj)

#### EXP_28: exp_28_film_subject_adaptive
- **Pipeline**: `deep_raw`
- **Models**: film_subject_adaptive
- **Best**: 37.10% acc, 34.24% F1 (+5.1pp, 5 subj)

#### EXP_29: exp_29_spectral_transformer_loso
- **Hypothesis**: H3: Spectral self-attention over frequency bands + channels > CNN-RNN
- **Pipeline**: `deep_raw`
- **Models**: spectral_transformer
- **Best**: 21.63% acc, 17.89% F1 (-10.4pp, 5 subj)

#### EXP_30: exp_30_channel_gat_loso
- **Hypothesis**: H30: Inter-electrode correlations > temporal structure. GAT on channel graph with dynamic adjacency.
- **Pipeline**: `deep_raw`
- **Models**: channel_gat
- **Best**: 22.60% acc, 20.30% F1 (-9.4pp, 5 subj)

#### EXP_31: exp_31_disentangled_content_style
- **Hypothesis**: H5: Content-Style Disentanglement
- **Pipeline**: `deep_raw`
- **Models**: disentangled_cnn_gru
- **Best**: 45.05% acc, 44.33% F1 (+13.1pp, 5 subj)

#### EXP_32: exp_32_spectral_band_ratio_svm_loso
- **Hypothesis**: Spectral band power ratios are more stable across subjects than absolute power
- **Pipeline**: `spectral_band_ratio + SVM`
- **Models**: svm_linear, svm_rbf
- **Best**: 27.40% acc, 25.96% F1 (-4.6pp, 5 subj)

#### EXP_33: exp_33_wavelet_scattering_svm_loso
- **Hypothesis**: 1D wavelet scattering transform provides built-in invariance to time-warping and scale deformations, improving cross-subject EMG classification without learned features
- **Pipeline**: `wavelet_scattering + SVM`
- **Models**: svm_linear, svm_rbf
- **Best**: 27.60% acc, 26.60% F1 (-4.4pp, 5 subj)

#### EXP_34: exp_34_curriculum_subject_ordering
- **Hypothesis**: Curriculum learning: training from similar to distant subjects
- **Pipeline**: `deep_raw`
- **Models**: cnn_gru_attention
- **Best**: 37.74% acc, 32.90% F1 (+5.7pp, 5 subj)

#### EXP_35: exp_35_mae_ssl_pretrain_loso
- **Pipeline**: ``
- **Models**: mae_emg
- **Best**: 25.61% acc, 24.49% F1 (-6.4pp, 5 subj)

#### EXP_36: exp_36_prototypical_arcface_loso
- **Hypothesis**: H10
- **Pipeline**: ``
- **Models**: unknown
- **Best**: 20.09% acc, 17.40% F1 (-11.9pp, 5 subj)

#### EXP_37: exp_37_channel_gat_gru_loso
- **Hypothesis**: H37: Spatio-temporal graph network with GAT (Pearson correlation + spectral coherence as edges) + per-channel BiGRU captures subject-invariant inter-muscular co-activation patterns.
- **Pipeline**: `deep_raw`
- **Models**: channel_gat_gru
- **Best**: 22.59% acc, 21.33% F1 (-9.4pp, 5 subj)

#### EXP_38: exp_38_nonlinear_stats_svm_lgbm_loso
- **Hypothesis**: Nonlinear statistics (sample entropy, permutation entropy, Higuchi FD, Hjorth, Lyapunov) and channel-pair features (cross-corr, coherence, MI) are more subject-invariant than classical RMS/MAV/PSD features.
- **Pipeline**: `CombinedNonlinearExtractor + SVM / LightGBM`
- **Models**: svm_linear, svm_rbf, lgbm
- **Best**: 30.07% acc, 28.77% F1 (-1.9pp, 5 subj)

#### EXP_40: exp_40_window_quality_filtering_loso
- **Hypothesis**: Removing low-quality EMG windows (by SNR/kurtosis/saturation/ZCR/channel-correlation/RMS-energy) from training data improves cross-subject classification transferability.
- **Pipeline**: ``
- **Models**: simple_cnn__percentile_0.2, svm_rbf__hard_threshold_0.4, svm_rbf__hard_threshold_0.3, svm_rbf__percentile_0.2, simple_cnn__percentile_0.3, simple_cnn__hard_threshold_0.3, simple_cnn__none, simple_cnn__hard_threshold_0.4, svm_rbf__none, svm_rbf__percentile_0.1, simple_cnn__percentile_0.1, svm_rbf__percentile_0.3
- **Best**: 28.67% acc, 25.47% F1 (-3.3pp, 5 subj)

#### EXP_41: exp_41_content_style_graph
- **Hypothesis**: Content-Style Graph: GRU/Attention content + GNN style for subject-invariance
- **Pipeline**: `deep_raw`
- **Models**: content_style_graph
- **Best**: 41.28% acc, 36.64% F1 (+9.3pp, 5 subj)

#### EXP_42: exp_42_multi_task_ssl_pretrain
- **Hypothesis**: Multi-task SSL pretraining (MAE + Subject Prediction + Cross-Subject Contrastive + Decorrelation) learns gesture-invariant features separated from subject style, improving LOSO cross-subject generalization.
- **Pipeline**: ``
- **Models**: unknown
- **Best**: 26.04% acc, 25.12% F1 (-6.0pp, 5 subj)

#### EXP_43: exp_43_tcn_gat_hybrid_loso
- **Hypothesis**: H43: Multi-scale causal dilated TCN (dilation {1,2,4,8}, kernel=7, RF≈184 ts ≈ 92 ms @ 2 kHz) + dynamic channel-graph GAT (Pearson + learnable prior) + per-channel BiGRU + temporal attention captures subject-invariant inter-muscular co-activation patterns.
- **Pipeline**: `deep_raw`
- **Models**: tcn_gat_hybrid
- **Best**: 25.62% acc, 23.47% F1 (-6.4pp, 5 subj)

#### EXP_44: exp_44_curriculum_disentangled_class_balanced_fusion
- **Hypothesis**: H_fusion: Curriculum + Disentanglement + Class-Balanced MixUp
- **Pipeline**: `deep_raw`
- **Models**: unknown
- **Best**: 40.24% acc, 35.44% F1 (+8.2pp, 5 subj)

#### EXP_46: exp_46_synthetic_subjects_domain_expansion_loso
- **Hypothesis**: Expanding subject space with VAE-generated synthetic subjects reduces cross-subject gap in LOSO evaluation.
- **Pipeline**: ``
- **Models**: cnn_gru_attention
- **Best**: 25.53% acc, 24.25% F1 (-6.5pp, 5 subj)

#### EXP_47: exp_47_vq_disentanglement_for_content_st_loso
- **Hypothesis**: h-047-vq-disentanglement
- **Pipeline**: `deep_raw`
- **Models**: vq_disentangle_emg
- **Best**: 9.96% acc, 1.81% F1 (-22.0pp, 5 subj)

#### EXP_48: exp_48_invariant_risk_minimization_for_causal_gesture_fea_loso
- **Hypothesis**: h-048-irm-regularization
- **Pipeline**: `deep_raw`
- **Models**: irm_content_style_emg
- **Best**: 27.96% acc, 26.19% F1 (-4.0pp, 5 subj)

#### EXP_49: exp_49_emg_cepstral_coefficients_muscle_filterbanks_loso
- **Hypothesis**: h-049-emgcc-cepstral
- **Pipeline**: `emgcc+powerful`
- **Models**: svm_rbf
- **Best**: 24.92% acc, 23.29% F1 (-7.1pp, 5 subj)

#### EXP_50: exp_50_subject_conditional_normalizing_flows_loso
- **Hypothesis**: h-050-normalizing-flows
- **Pipeline**: `deep_raw`
- **Models**: flow_canonical_emg
- **Best**: 29.06% acc, 27.34% F1 (-2.9pp, 5 subj)

#### EXP_51: exp_51_rank_copula_features_monotone_invariant_loso
- **Hypothesis**: h-051-rank-copula
- **Pipeline**: `rank_copula+powerful`
- **Models**: svm_rbf
- **Best**: 20.54% acc, 18.25% F1 (-11.5pp, 5 subj)

#### EXP_52: exp_52_channel_permutation_equivariant_set_transformer_loso
- **Hypothesis**: h-052-channel-permutation-equivariant
- **Pipeline**: `deep_raw`
- **Models**: set_transformer_emg
- **Best**: 20.96% acc, 15.86% F1 (-11.0pp, 5 subj)

#### EXP_53: exp_53_latent_diffusion_subject_style_removal_loso
- **Hypothesis**: h-053-diffusion-canonical
- **Pipeline**: `deep_raw`
- **Models**: latent_diffusion_emg
- **Best**: 26.39% acc, 24.94% F1 (-5.6pp, 5 subj)

#### EXP_54: exp_54_multi_resolution_temporal_consensus_loso
- **Hypothesis**: h-054-multi-resolution-consensus
- **Pipeline**: `deep_raw`
- **Models**: multi_resolution_emg
- **Best**: 29.37% acc, 26.70% F1 (-2.6pp, 5 subj)

#### EXP_55: exp_55_optimal_transport_barycenter_alignment_loso
- **Hypothesis**: h-055-ot-barycenter
- **Pipeline**: `powerful+ot_alignment`
- **Models**: svm_rbf
- **Best**: 13.08% acc, 11.54% F1 (-18.9pp, 5 subj)

#### EXP_56: exp_56_test_time_training_masked_channel_ssl_loso
- **Hypothesis**: h-056-test-time-training
- **Pipeline**: `deep_raw`
- **Models**: ttt_content_style_emg
- **Best**: 30.27% acc, 30.35% F1 (-1.7pp, 5 subj)

#### EXP_57: exp_57_groupdro_disentangled
- **Hypothesis**: H8: GroupDRO + Content-Style Disentanglement
- **Pipeline**: `deep_raw`
- **Models**: groupdro_disentangled_cnn_gru
- **Best**: 41.47% acc, 35.78% F1 (+9.5pp, 5 subj)

#### EXP_58: exp_58_mi_subject_adversary
- **Hypothesis**: H5b: CLUB MI upper-bound subject adversary
- **Pipeline**: `deep_raw`
- **Models**: mi_disentangled_cnn_gru
- **Best**: 33.21% acc, 29.43% F1 (+1.2pp, 5 subj)

#### EXP_59: exp_59_disentanglement_prototype_regularization
- **Hypothesis**: Disentanglement + Prototype Regularization in z_content
- **Pipeline**: `deep_raw`
- **Models**: proto_disentangled_cnn_gru
- **Best**: 42.86% acc, 43.00% F1 (+10.9pp, 5 subj)

#### EXP_60: exp_60_mixstyle_content_disentangled
- **Hypothesis**: H60: Mixture of Styles — FiLM in latent z_style space
- **Pipeline**: `deep_raw`
- **Models**: mixstyle_disentangled_cnn_gru
- **Best**: 43.04% acc, 41.44% F1 (+11.0pp, 5 subj)

#### EXP_61: exp_61_sinc_pcen_frontend
- **Hypothesis**: SincNet-PCEN learnable frontend for channel-invariant EMG
- **Pipeline**: `deep_raw`
- **Models**: sinc_pcen_cnn_gru
- **Best**: 30.90% acc, 27.84% F1 (-1.1pp, 5 subj)

#### EXP_63: exp_63_riemannian_spd_covariance_loso
- **Hypothesis**: Riemannian tangent-space covariance features are more cross-subject invariant than classical EMG amplitude/power descriptors.  SPD covariance captures inter-channel muscle co-activation patterns robustly to per-subject amplitude shifts (cf. EEG Riemannian BCI).
- **Pipeline**: `RiemannianSPDExtractor (log-Euclidean mean) + SVM`
- **Models**: pure_svm_linear, hankel_svm_rbf, pure_svm_rbf, combined_svm_rbf
- **Best**: 30.28% acc, 29.45% F1 (-1.7pp, 5 subj)

#### EXP_64: exp_64_multiclass_csp_filterbank_loso
- **Hypothesis**: CSP spatial filters learned from multi-subject training data produce log-variance features that are cross-subject robust.  Log-variance of CSP components is invariant to multiplicative gain shifts (electrode placement, skin impedance variation) — a key source of inter-subject EMG variability.  OAS shrinkage and a frequency filter bank further improve generalisation.
- **Pipeline**: `One-vs-Rest Multiclass CSP (+ optional FilterBank) + SVM. CSP filters fitted on train subjects only (LOSO safe).`
- **Models**: fbcsp_svm_rbf, ovr_csp_svm_rbf, ovr_csp_shrink_svm, ovr_csp_svm_linear
- **Best**: 28.75% acc, 27.99% F1 (-3.3pp, 5 subj)

#### EXP_65: exp_65_trainable_fir_neural_drive_deconv
- **Hypothesis**: Trainable per-channel FIR frontend as neural-drive deconvolution for cross-subject EMG classification.
- **Pipeline**: `deep_raw`
- **Models**: fir_deconv_cnn_gru
- **Best**: 45.08% acc, 40.21% F1 (+13.1pp, 5 subj)

#### EXP_66: exp_66_temporal_phase_alignment
- **Hypothesis**: TKEO-based temporal phase alignment canonicalizes gesture timing across subjects, reducing inter-subject variability for cross-subject EMG gesture recognition.
- **Pipeline**: `deep_raw`
- **Models**: phase_align_cnn_gru
- **Best**: 39.08% acc, 34.13% F1 (+7.1pp, 5 subj)

#### EXP_67: exp_67_cyclostationary_spectral_correlation_loso
- **Hypothesis**: H67: Cyclostationary features (normalized ACF, spectral cyclic coherence, envelope periodicity) are amplitude-invariant and capture gesture-specific MU firing structure, improving cross-subject generalization. Analogous to cyclostationary channel-invariance in communications.
- **Pipeline**: `CombinedCyclostationaryExtractor (PowerfulFeatureExtractor + CyclostationaryEMGExtractor) → z-score → PCA (train only) → SVM/LightGBM`
- **Models**: svm_linear, svm_rbf, lgbm
- **Best**: 30.72% acc, 29.81% F1 (-1.3pp, 5 subj)

#### EXP_68: exp_68_multitaper_psd_spectral_slope_loso
- **Hypothesis**: Multitaper PSD aperiodic exponent, spectral knee, and residual oscillatory peaks (FOOOF-style, log-log linear fit) are more subject-invariant than raw band power or amplitude. Combined with PowerfulFeatureExtractor for SVM/LGBM.
- **Pipeline**: `MultitaperPSDExtractor + PowerfulFeatureExtractor + SVM / LGBM`
- **Models**: svm_linear, svm_rbf, lgbm
- **Best**: 28.52% acc, 27.74% F1 (-3.5pp, 5 subj)

#### EXP_69: exp_69_vrex_fishr_irm_v2_loso
- **Hypothesis**: h-069-vrex-fishr-irm-v2
- **Note**: Same backbone as exp_48 (IRMv1) for fair comparison
- **Pipeline**: ``
- **Models**: fishr, vrex
- **Best**: 28.70% acc, 26.69% F1 (-3.3pp, 5 subj)

#### EXP_72: exp_72_moe_dynamic_routing_loso
- **Hypothesis**: H72: MoE v2 — routing by motion dynamics (TKEO energy, envelope slope, kurtosis, ZCR) with LayerNorm router and TDNN experts of varying dilation outperforms subject-style MoE routing (exp_27) in cross-subject LOSO
- **Pipeline**: `deep_raw`
- **Models**: moe_dynamic_routing
- **Best**: 28.06% acc, 25.70% F1 (-3.9pp, 5 subj)

#### EXP_77: exp_77_stochastic_hypernetwork_fir_deconv
- **Hypothesis**: Domain randomization via noise-conditioned FIR hypernetwork. Backbone learns subject-invariant features by training on random filter realizations. Test-time: u=0, fully deterministic.
- **Pipeline**: `deep_raw`
- **Models**: stochastic_fir_cnn_gru
- **Best**: 42.14% acc, 37.71% F1 (+10.1pp, 5 subj)

#### EXP_78: exp_78_cpsd_riemannian_spectral_loso
- **Hypothesis**: CPSD (Cross-Power Spectral Density) matrices in frequency bands, converted to real SPD and projected via log-Euclidean Riemannian tangent space, capture cross-subject-invariant muscle coupling structure.  The spectral inter-channel interaction pattern is more stable across subjects than broadband time-domain covariance (cf. exp_63), because phase/coherence is independent of absolute signal amplitude.
- **Pipeline**: `Per-band CPSD (Hermitian) → real SPD → log-Euclidean Riemannian tangent features → SVM`
- **Models**: cpsd_real_4band_svm_linear, cpsd_real_4band_svm_rbf, cpsd_real_combined_svm_rbf, cpsd_block_4band_svm_rbf
- **Best**: 29.83% acc, 29.18% F1 (-2.2pp, 5 subj)

#### EXP_80: exp_80_synthetic_env_vrex_fishr_loso
- **Hypothesis**: h-080-synthetic-env-vrex-fishr
- **Note**: exp_80 (synthetic transform environments) vs exp_69 (subject-based environments) — same backbone and penalty formulas, different environment source.
- **Pipeline**: ``
- **Models**: subject_transform/fishr, transforms_only/fishr, subject_transform/vrex, transforms_only/vrex
- **Best**: 28.04% acc, 26.47% F1 (-4.0pp, 5 subj)

#### EXP_84: exp_84_marginal_kurtosis_frequency_band_selection_loso
- **Hypothesis**: Marginal Kurtosis (MK) identifies frequency bands with non-Gaussian (bursty) power fluctuations across training windows, characteristic of gesture-related MUAP activity. Filtering to retain only high-MK bands removes noise and focuses feature extraction on informative content.
- **Pipeline**: `MK band selection (training-only) → spectral filtering → PowerfulFeatureExtractor + MK spectral features → SVM / LGBM`
- **Models**: svm_linear, svm_rbf, lgbm
- **Best**: 26.11% acc, 25.18% F1 (-5.9pp, 5 subj)

#### EXP_85: exp_85_wca_condition_attention_cnn_gru_loso
- **Pipeline**: `deep_raw`
- **Models**: wca_condition_attention
- **Best**: 28.00% acc, 26.65% F1 (-4.0pp, 5 subj)

#### EXP_87: exp_87_tidwt_denoising_preprocessing_loso
- **Hypothesis**: SWT (shift-invariant undecimated DWT) denoising of raw EMG windows before feature extraction improves cross-subject classification by removing noise while preserving gesture-discriminative signal structure.
- **Pipeline**: `TIDWT_denoising + PowerfulFeatures + SVM`
- **Models**: svm_linear, svm_rbf
- **Best**: 29.75% acc, 28.91% F1 (-2.3pp, 5 subj)

#### EXP_88: exp_88_causal_ecapa_disentanglement
- **Hypothesis**: Causal ECAPA-TDNN content/style disentanglement
- **Pipeline**: `deep_raw`
- **Models**: causal_ecapa_tdnn
- **Best**: 40.00% acc, 34.08% F1 (+8.0pp, 5 subj)

#### EXP_99: exp_99_discrete_style_codebook_loso
- **Hypothesis**: h-099-discrete-style-codebook
- **Pipeline**: `deep_raw`
- **Models**: discrete_style_codebook
- **Best**: 10.38% acc, 2.38% F1 (-21.6pp, 5 subj)

#### EXP_100: exp_100_dual_stream_hard_style_augmentation
- **Hypothesis**: H100: Dual-Stream Hard Style Augmentation with FGSM adversarial perturbation
- **Pipeline**: `deep_raw`
- **Models**: dual_stream_hard_style_cnn_gru
- **Best**: 44.28% acc, 37.83% F1 (+12.3pp, 5 subj)

#### EXP_101: exp_101_xdomain_mix_4component_decomposition
- **Hypothesis**: H101: XDomainMix — 4-Component Decomposition with Cross-Domain Recombination
- **Pipeline**: `deep_raw`
- **Models**: xdomain_mix_emg
- **Best**: 41.90% acc, 36.53% F1 (+9.9pp, 5 subj)

#### EXP_102: exp_102_freq_band_style_mixing
- **Hypothesis**: H102: Frequency-Band Style Mixing (AdaIN per EMG band)
- **Pipeline**: `deep_raw`
- **Models**: freq_band_style_mix_emg
- **Best**: 49.42% acc, 48.31% F1 (+17.4pp, 5 subj)

#### EXP_103: exp_103_synth_env_groupdro
- **Hypothesis**: H103: Synthetic Environment Expansion + Soft GroupDRO
- **Pipeline**: `deep_raw`
- **Models**: synth_env_groupdro_emg
- **Best**: 42.86% acc, 41.65% F1 (+10.9pp, 5 subj)

#### EXP_104: exp_104_causal_barlow_groupdro
- **Hypothesis**: Causal Disentanglement (CDDG) + Barlow Twins + GroupDRO
- **Pipeline**: `deep_raw`
- **Models**: causal_barlow_cnn_gru
- **Best**: 41.53% acc, 35.41% F1 (+9.5pp, 5 subj)

#### EXP_105: exp_105_dsfe_style_bank_exploitation
- **Hypothesis**: DSFE: Domain-Specific Feature Exploitation via Style Bank
- **Pipeline**: `deep_raw`
- **Models**: dsfe_style_bank_cnn_gru
- **Best**: 41.33% acc, 40.03% F1 (+9.3pp, 5 subj)

#### EXP_106: exp_106_channel_contrastive_disentanglement_per_ch_dro
- **Hypothesis**: H9: Channel-wise Contrastive Disentanglement + Per-Channel DRO
- **Pipeline**: `deep_raw`
- **Models**: channel_contrastive_disentangled
- **Best**: 46.21% acc, 44.71% F1 (+14.2pp, 5 subj)

#### EXP_107: exp_107_progressive_env_diversification_adaptive_dro
- **Hypothesis**: H107: Progressive Environment Diversification with Adaptive DRO — phased training (disentanglement -> MixStyle+DRO -> extrap+aggressive DRO), adaptive eta, anti-collapse
- **Pipeline**: `deep_raw`
- **Models**: progressive_env_dro
- **Best**: 42.67% acc, 41.18% F1 (+10.7pp, 5 subj)

#### EXP_108: exp_108_hierarchical_beta_vae_uvmd
- **Hypothesis**: H1_new: Hierarchical β-VAE + UVMD + SoftAGC
- **Pipeline**: `deep_raw`
- **Models**: hierarchical_beta_vae_uvmd
- **Best**: 17.25% acc, 10.65% F1 (-14.8pp, 5 subj)

#### EXP_109: exp_109_mate_kronecker_shared_specific
- **Hypothesis**: MATE Shared-Specific without orthogonality: shared and specific representations allowed to be correlated; only z_shared is adversarially regularised against subject ID.
- **Pipeline**: `deep_raw`
- **Models**: mate_kronecker
- **Best**: 49.60% acc, 43.47% F1 (+17.6pp, 5 subj)

#### EXP_110: exp_110_multi_res_aligned_disentangle
- **Hypothesis**: H3: Multi-Resolution Aligned Disentanglement
- **Pipeline**: `deep_raw`
- **Models**: multi_res_aligned_disentangle
- **Best**: 50.64% acc, 49.36% F1 (+18.6pp, 5 subj)

#### EXP_112: exp_112_channel_band_tucker_consensus
- **Hypothesis**: Three-axis Tucker factorization of EMG spectrogram: channel (GRL adversary), frequency (Soft AGC), temporal (quarter-window KL consensus).
- **Pipeline**: `deep_raw`
- **Models**: tucker_consensus
- **Best**: 38.17% acc, 33.39% F1 (+6.2pp, 5 subj)

#### EXP_test_bigru_spec: test_bigru_spec
- **Pipeline**: `deep_raw`
- **Models**: bigru
- **Best**: 12.20% acc, 10.70% F1 (-19.8pp, 2 subj)

---

## Key Findings

1. **multi_res_aligned_disentangle** (exp_110): 50.64% acc (+18.6pp)
2. **mate_kronecker** (exp_109): 49.60% acc (+17.6pp)
3. **freq_band_style_mix_emg** (exp_102): 49.42% acc (+17.4pp)
4. **multi_res_aligned_disentangle** (exp_110): 48.05% acc (+16.0pp)
5. **freq_band_style_mix_emg** (exp_102): 46.40% acc (+14.4pp)
6. **channel_contrastive_disentangled** (exp_106): 46.21% acc (+14.2pp)
7. **freq_band_style_mix_emg** (exp_102): 45.32% acc (+13.3pp)
8. **fir_deconv_cnn_gru** (exp_65): 45.08% acc (+13.1pp)
9. **disentangled_cnn_gru** (exp_31): 45.05% acc (+13.1pp)
10. **dual_stream_hard_style_cnn_gru** (exp_100): 44.28% acc (+12.3pp)
11. **channel_contrastive_disentangled** (exp_106): 43.91% acc (+11.9pp)
12. **channel_contrastive_disentangled** (exp_106): 43.05% acc (+11.0pp)
13. **mixstyle_disentangled_cnn_gru** (exp_60): 43.04% acc (+11.0pp)
14. **synth_env_groupdro_emg** (exp_103): 42.86% acc (+10.9pp)
15. **proto_disentangled_cnn_gru** (exp_59): 42.86% acc (+10.9pp)

**31/150** experiments exceeded the best baseline (32.0%).
