# OMEGA Research Progress

**Last updated**: 2026-03-18

**Total experiments**: 150

- Best accuracy: 50.64%
- Mean accuracy: 28.58%
- Median accuracy: 27.33%

---

## Progress by Research Direction

### Advanced Architectures (7 experiments)

**Best**: spectral_transformer — 21.63% acc, 17.89% F1 (exp_29_spectral_transformer_loso)

- ✗ **exp_29_spectral_transformer_loso**: 21.63% acc, 17.89% F1
- ✗ **exp_52_channel_permutation_equivariant_set_transformer_loso**: 20.96% acc, 15.86% F1
- ✗ **exp_111_filterbank_mode_adaptive_ecapa**: 0.00% acc, 0.00% F1
- ✗ **exp_62_ecapa_tdnn_subject_robust**: 0.00% acc, 0.00% F1
- ✗ **exp_70_conformer_ecapa**: 0.00% acc, 0.00% F1
- ✗ **exp_81_ortho_channel_mix_ecapa**: 0.00% acc, 0.00% F1
- ✗ **exp_91_cyclic_inter_subject_adain_ecapa**: 0.00% acc, 0.00% F1

### Augmentation Strategies (11 experiments)

**Best**: dual_stream_hard_style_cnn_gru — 44.28% acc, 37.83% F1 (exp_100_dual_stream_hard_style_augmentation)

- ✓ **exp_100_dual_stream_hard_style_augmentation**: 44.28% acc, 37.83% F1
- ✗ **exp_12_augmented_svm_with_time_domain_features_for_improv_loso**: 25.51% acc, 24.43% F1
- ✗ **exp_21_svm_rbf_with_noise_time_warp_augmentation_and_tune_loso**: 19.44% acc, 16.49% F1
- ✗ **exp_10_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_16_enhanced_augmentation_strategy_for_cnn_gru_attenti_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_17_enhanced_augmentation_for_simple_cnn_on_raw_emg_si_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_18_augmented_svm_with_feature_space_jitter_for_improv_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_20_augmented_cnn_gru_attention_with_feature_fusion_fo_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_7_cnn_gru_attention_with_noise_and_time_warp_augment_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_8_augmented_svm_with_time_domain_features_for_improv_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_9_dual_stream_cnn_gru_attention_with_augmented_raw_e_loso**: 0.00% acc, 0.00% F1

### Disentanglement & Style (11 experiments)

**Best**: multi_res_aligned_disentangle — 50.64% acc, 49.36% F1 (exp_110_multi_res_aligned_disentangle)

- ✓ **exp_110_multi_res_aligned_disentangle**: 50.64% acc, 49.36% F1
- ✓ **exp_106_channel_contrastive_disentanglement_per_ch_dro**: 46.21% acc, 44.71% F1
- ✓ **exp_31_disentangled_content_style**: 45.05% acc, 44.33% F1
- ✓ **exp_60_mixstyle_content_disentangled**: 43.04% acc, 41.44% F1
- ✓ **exp_59_disentanglement_prototype_regularization**: 42.86% acc, 43.00% F1
- ✓ **exp_57_groupdro_disentangled**: 41.47% acc, 35.78% F1
- ✓ **exp_41_content_style_graph**: 41.28% acc, 36.64% F1
- ✓ **exp_44_curriculum_disentangled_class_balanced_fusion**: 40.24% acc, 35.44% F1
- ✓ **exp_88_causal_ecapa_disentanglement**: 40.00% acc, 34.08% F1
- ✗ **exp_47_vq_disentanglement_for_content_st_loso**: 9.96% acc, 1.81% F1
- ✗ **exp_89_selective_disentanglement_clip_dca**: 0.00% acc, 0.00% F1

### Frequency Decomposition (9 experiments)

**Best**: sinc_pcen_cnn_gru — 30.90% acc, 27.84% F1 (exp_61_sinc_pcen_frontend)

- ○ **exp_61_sinc_pcen_frontend**: 30.90% acc, 27.84% F1
- ○ **exp_64_multiclass_csp_filterbank_loso**: 28.75% acc, 27.99% F1
- ✗ **exp_49_emg_cepstral_coefficients_muscle_filterbanks_loso**: 24.92% acc, 23.29% F1
- ✗ **exp_108_hierarchical_beta_vae_uvmd**: 17.25% acc, 10.65% F1
- ✗ **exp_82_vmd_imf_decomposition**: 0.00% acc, 0.00% F1
- ✗ **exp_93_unfolded_vmd_uvmd**: 0.00% acc, 0.00% F1
- ✗ **exp_94_learnable_filterbank_grl**: 0.00% acc, 0.00% F1
- ✗ **exp_96_stvmd_learnable_seg_mode_aug**: 0.00% acc, 0.00% F1
- ✗ **exp_97_filterbank_vib_information_bottleneck**: 0.00% acc, 0.00% F1

### Hypothesis Testing (H1-H3) (2 experiments)

**Best**: ? — 0.00% acc, 0.00% F1 (h2_ablation_decomposition)

- ✗ **h2_ablation_decomposition**: 0.00% acc, 0.00% F1
- ✗ **h3_style_normalization**: 0.00% acc, 0.00% F1

### Hypothesis Testing (H4-H5) (2 experiments)

**Best**: ? — 0.00% acc, 0.00% F1 (h4_content_style_disentanglement)

- ✗ **h4_content_style_disentanglement**: 0.00% acc, 0.00% F1
- ✗ **h5_integrated_system**: 0.00% acc, 0.00% F1

### ML Baselines (8 experiments)

**Best**: svm_linear — 35.24% acc, 32.50% F1 (exp4_svm_linear_powerful_loso)

- ✓ **exp4_svm_linear_powerful_loso**: 35.24% acc, 32.50% F1
- ✓ **exp4_svm_rbf_powerful_loso**: 34.46% acc, 32.60% F1
- ○ **exp_38_nonlinear_stats_svm_lgbm_loso**: 30.07% acc, 28.77% F1
- ✗ **exp_4_ml_powerful_loso**: 27.82% acc, 26.63% F1
- ✗ **exp_33_wavelet_scattering_svm_loso**: 27.60% acc, 26.60% F1
- ✗ **exp_32_spectral_band_ratio_svm_loso**: 27.40% acc, 25.96% F1
- ✗ **exp_25_svm_linear_on_powerful_features_with_combined_nois_loso**: 26.08% acc, 24.89% F1
- ✗ **exp_14_subject_adaptive_fine_tuning_for_svm_on_powerful_f_loso**: 0.00% acc, 0.00% F1

### Other Approaches (96 experiments)

**Best**: mate_kronecker — 49.60% acc, 43.47% F1 (exp_109_mate_kronecker_shared_specific)

- ✓ **exp_109_mate_kronecker_shared_specific**: 49.60% acc, 43.47% F1
- ✓ **exp_102_freq_band_style_mixing**: 49.42% acc, 48.31% F1
- ✓ **exp_65_trainable_fir_neural_drive_deconv**: 45.08% acc, 40.21% F1
- ✓ **exp_103_synth_env_groupdro**: 42.86% acc, 41.65% F1
- ✓ **exp_107_progressive_env_diversification_adaptive_dro**: 42.67% acc, 41.18% F1
- ✓ **exp_77_stochastic_hypernetwork_fir_deconv**: 42.14% acc, 37.71% F1
- ✓ **exp_101_xdomain_mix_4component_decomposition**: 41.90% acc, 36.53% F1
- ✓ **exp_104_causal_barlow_groupdro**: 41.53% acc, 35.41% F1
- ✓ **exp_105_dsfe_style_bank_exploitation**: 41.33% acc, 40.03% F1
- ✓ **exp_66_temporal_phase_alignment**: 39.08% acc, 34.13% F1
- ✓ **exp_13_leveraging_subject_specific_calibration_via_few_sh_loso**: 38.25% acc, 6.13% F1
- ✓ **exp_112_channel_band_tucker_consensus**: 38.17% acc, 33.39% F1
- ✓ **exp_34_curriculum_subject_ordering**: 37.74% acc, 32.90% F1
- ✓ **exp_28_film_subject_adaptive**: 37.10% acc, 34.24% F1
- ✓ **exp_58_mi_subject_adversary**: 33.21% acc, 29.43% F1
- ○ **exp4_rf_powerful_loso**: 32.00% acc, 30.33% F1
- ○ **exp1_deep_raw_cnn_gru_attention_loso_isolated_v2**: 30.85% acc, 28.19% F1
- ○ **exp_26_test_time_bn_adaptation_for_cnn_g_loso**: 30.74% acc, 30.14% F1
- ○ **exp_67_cyclostationary_spectral_correlation_loso**: 30.72% acc, 29.81% F1
- ○ **exp_63_riemannian_spd_covariance_loso**: 30.28% acc, 29.45% F1
- ○ **exp_78_cpsd_riemannian_spectral_loso**: 29.83% acc, 29.18% F1
- ○ **exp_87_tidwt_denoising_preprocessing_loso**: 29.75% acc, 28.91% F1
- ○ **exp_54_multi_resolution_temporal_consensus_loso**: 29.37% acc, 26.70% F1
- ○ **exp1_deep_raw_simple_cnn_loso_isolated_v2**: 29.30% acc, 25.51% F1
- ○ **exp_50_subject_conditional_normalizing_flows_loso**: 29.06% acc, 27.34% F1
- ○ **exp_69_vrex_fishr_irm_v2_loso**: 28.70% acc, 26.69% F1
- ○ **exp_40_window_quality_filtering_loso**: 28.67% acc, 25.47% F1
- ○ **exp1_deep_raw_bigru_loso_isolated_v2**: 28.60% acc, 25.43% F1
- ○ **exp_68_multitaper_psd_spectral_slope_loso**: 28.52% acc, 27.74% F1
- ○ **exp1_deep_raw_bilstm_attention_loso_isolated_v2**: 28.10% acc, 24.88% F1
- ○ **exp_72_moe_dynamic_routing_loso**: 28.06% acc, 25.70% F1
- ○ **exp_80_synthetic_env_vrex_fishr_loso**: 28.04% acc, 26.47% F1
- ✗ **exp_48_invariant_risk_minimization_for_causal_gesture_fea_loso**: 27.96% acc, 26.19% F1
- ✗ **exp1_deep_raw_attention_cnn_loso_isolated_v2**: 27.33% acc, 23.70% F1
- ✗ **exp1_deep_raw_tcn_loso_isolated_v2**: 26.91% acc, 23.57% F1
- ✗ **exp1_deep_raw_multiscale_cnn_loso_isolated_v2**: 26.81% acc, 23.44% F1
- ✗ **exp_53_latent_diffusion_subject_style_removal_loso**: 26.39% acc, 24.94% F1
- ✗ **exp_27_moe_cnn_gru_attention_loso**: 26.33% acc, 24.71% F1
- ✗ **exp_24_cnn_gru_attention_on_raw_emg_with_class_weighted_l_loso**: 26.30% acc, 25.22% F1
- ✗ **exp1_deep_raw_bilstm_attention_loso**: 26.20% acc, 23.22% F1
- ✗ **exp_84_marginal_kurtosis_frequency_band_selection_loso**: 26.11% acc, 25.18% F1
- ✗ **exp1_deep_raw_simple_cnn_loso_1_12_15_28_39**: 25.99% acc, 24.11% F1
- ✗ **exp_43_tcn_gat_hybrid_loso**: 25.62% acc, 23.47% F1
- ✗ **exp_2_deep_emg_td_seq_cnn_lstm_loso**: 25.46% acc, 24.29% F1
- ✗ **exp1_deep_raw_attention_cnn_loso_1_12_15_28_39**: 25.31% acc, 23.28% F1
- ✗ **exp2_deep_emg_td_seq_bigru_loso**: 25.15% acc, 20.67% F1
- ✗ **exp2_deep_emg_td_seq_bilstm_attention_loso**: 25.10% acc, 19.99% F1
- ✗ **exp_2_deep_emg_td_seq_bilstm_attention_loso**: 24.70% acc, 23.44% F1
- ✗ **exp1_deep_raw_tcn_attn_loso_1_12_15_28_39**: 24.66% acc, 22.05% F1
- ✗ **exp_2_deep_emg_td_seq_tcn_loso**: 24.63% acc, 23.29% F1
- ✗ **exp_2_deep_emg_td_seq_bigru_loso**: 24.03% acc, 22.56% F1
- ✗ **exp_2_deep_emg_td_seq_bilstm_loso**: 23.82% acc, 22.70% F1
- ✗ **exp_2_deep_emg_td_seq_cnn_gru_attention_loso**: 23.78% acc, 22.89% F1
- ✗ **exp_2_deep_emg_td_seq_attention_cnn_loso**: 23.62% acc, 22.20% F1
- ✗ **exp_2_deep_emg_td_seq_tcn_attn_loso**: 23.50% acc, 22.42% F1
- ✗ **exp_2_deep_emg_td_seq_multiscale_cnn_loso**: 23.41% acc, 22.44% F1
- ✗ **exp_2_deep_emg_td_seq_simple_cnn_loso**: 23.37% acc, 22.51% F1
- ✗ **exp1_deep_raw_cnn_loso_isolated**: 23.14% acc, 19.06% F1
- ✗ **exp1_deep_raw_tcn_loso_1_12_15_28_39**: 23.11% acc, 18.58% F1
- ✗ **exp1_deep_raw_tcn_attn_loso_isolated_v2**: 22.92% acc, 17.82% F1
- ✗ **exp_30_channel_gat_loso**: 22.60% acc, 20.30% F1
- ✗ **exp_37_channel_gat_gru_loso**: 22.59% acc, 21.33% F1
- ✗ **exp2_deep_emg_td_seq_bilstm_loso**: 21.77% acc, 16.81% F1
- ✗ **exp1_deep_raw_attention_cnn_loso_isolated**: 21.68% acc, 18.30% F1
- ✗ **exp2_deep_emg_td_seq_tcn_attn_loso**: 21.57% acc, 15.98% F1
- ✗ **exp1_deep_raw_cnn_lstm_loso**: 20.78% acc, 16.54% F1
- ✗ **exp_51_rank_copula_features_monotone_invariant_loso**: 20.54% acc, 18.25% F1
- ✗ **exp1_deep_raw_bilstm_loso_isolated_v2**: 19.32% acc, 13.97% F1
- ✗ **exp2_deep_emg_td_seq_cnn_gru_attention_loso**: 15.27% acc, 8.83% F1
- ✗ **exp2_deep_emg_td_seq_cnn_lstm_loso**: 13.93% acc, 7.13% F1
- ✗ **exp_55_optimal_transport_barycenter_alignment_loso**: 13.08% acc, 11.54% F1
- ✗ **test_bigru_spec**: 12.20% acc, 10.70% F1
- ✗ **exp2_deep_emg_td_seq_multiscale_cnn_loso**: 11.99% acc, 5.09% F1
- ✗ **exp2_deep_emg_td_seq_simple_cnn_loso**: 11.44% acc, 5.25% F1
- ✗ **exp5_hybrid_powerful_deep_loso**: 10.74% acc, 2.67% F1
- ✗ **exp3_deep_powerful_mlp_powerful_loso**: 10.27% acc, 2.21% F1
- ✗ **exp2_deep_emg_td_seq_attention_cnn_loso**: 10.00% acc, 4.90% F1
- ✗ **exp_11_enhancing_simple_cnn_robustness_with_subject_speci_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_15_contrastive_subject_aware_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_19_subject_specific_feature_calibration_for_reducing_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_22_focal_loss_class_balanced_sampling_for_cnn_gru_att_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_23_channel_wise_squeeze_and_excitation_cnn_gru_attent_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_36_prototypical_arcface_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_46_synthetic_subjects_domain_expansion_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_74_temporal_order_invariance_jigsaw_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_75_emg_patch_tokens_performer**: 0.00% acc, 0.00% F1
- ✗ **exp_76_soft_agc_pcen_lite**: 0.00% acc, 0.00% F1
- ✗ **exp_79_subcenter_arcface_emg**: 0.00% acc, 0.00% F1
- ✗ **exp_83_subject_cluster_models_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_85_wca_condition_attention_cnn_gru_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_86_knowledge_base_similarity_ensemble_loso**: 0.00% acc, 0.00% F1
- ✗ **exp_90_hygdl_analytical_orthogonal_projection**: 0.00% acc, 0.00% F1
- ✗ **exp_92_mrbt_channel_group_barlow_twins**: 0.00% acc, 0.00% F1
- ✗ **exp_95_vlmd_latent_mode_decomposition**: 0.00% acc, 0.00% F1
- ✗ **exp_98_cyclemix_channel_wise_stochastic**: 0.00% acc, 0.00% F1
- ✗ **exp_99_discrete_style_codebook_loso**: 0.00% acc, 0.00% F1

### Self-Supervised Learning (4 experiments)

**Best**: ttt_content_style_emg — 30.27% acc, 30.35% F1 (exp_56_test_time_training_masked_channel_ssl_loso)

- ○ **exp_56_test_time_training_masked_channel_ssl_loso**: 30.27% acc, 30.35% F1
- ✗ **exp_35_mae_ssl_pretrain_loso**: 25.61% acc, 24.49% F1
- ✗ **exp_42_multi_task_ssl_pretrain**: 0.00% acc, 0.00% F1
- ✗ **exp_73_cpc_ssl_pretrain_loso**: 0.00% acc, 0.00% F1

---

## Experiment Timeline

- **2026-01-**: 25 experiments, best=35.24% (exp4_svm_linear_powerful_loso)
- **2026-02-**: 77 experiments, best=38.25% (exp_13_leveraging_subject_specific_calibration_via_few_sh_loso)
- **2026-03-**: 2 experiments, best=0.00% (h5_integrated_system)
- **20260221**: 3 experiments, best=45.05% (exp_31_disentangled_content_style)
- **20260222**: 2 experiments, best=41.28% (exp_41_content_style_graph)
- **20260223**: 7 experiments, best=45.08% (exp_65_trainable_fir_neural_drive_deconv)
- **20260224**: 2 experiments, best=42.14% (exp_77_stochastic_hypernetwork_fir_deconv)
- **20260228**: 1 experiments, best=40.00% (exp_88_causal_ecapa_disentanglement)
- **20260301**: 9 experiments, best=49.60% (exp_109_mate_kronecker_shared_specific)
- **20260303**: 2 experiments, best=46.21% (exp_106_channel_contrastive_disentanglement_per_ch_dro)
- **20260304**: 2 experiments, best=50.64% (exp_110_multi_res_aligned_disentangle)
- **20260308**: 2 experiments, best=0.00% (h3_style_normalization)

## Key Milestones

### Top 5 Results

1. **50.64%** acc / 49.36% F1 — multi_res_aligned_disentangle (exp_110_multi_res_aligned_disentangle)
2. **49.60%** acc / 43.47% F1 — mate_kronecker (exp_109_mate_kronecker_shared_specific)
3. **49.42%** acc / 48.31% F1 — freq_band_style_mix_emg (exp_102_freq_band_style_mixing)
4. **46.21%** acc / 44.71% F1 — channel_contrastive_disentangled (exp_106_channel_contrastive_disentanglement_per_ch_dro)
5. **45.08%** acc / 40.21% F1 — fir_deconv_cnn_gru (exp_65_trainable_fir_neural_drive_deconv)
