"""Static registry of valid models, features, and augmentations from the OMEGA pipeline.

These values are extracted from:
- training/trainer.py: _create_model() factory
- config/base.py: TrainingConfig dataclass
- models/__init__.py: model registry
"""

VALID_MODEL_TYPES = [
    "simple_cnn",
    "attention_cnn",
    "tcn",
    "tcn_attn",
    "multiscale_cnn",
    "bilstm",
    "bilstm_attention",
    "bigru",
    "cnn_lstm",
    "cnn_gru_attention",
    "resnet1d",
]

VALID_PIPELINE_TYPES = [
    "deep_raw",
    "deep_emg_td_seq",
    "ml_emg_td",
    "hybrid",
]

VALID_FEATURE_SETS = [
    "raw",
    "basic_v1",
    "powerful",
]

VALID_AUGMENTATIONS = [
    "none",
    "noise",
    "time_warp",
    "rotation",
    "noise+time_warp",
    "noise+rotation",
    "time_warp+rotation",
    "noise+time_warp+rotation",
]

VALID_ML_MODELS = [
    "svm_rbf",
    "svm_linear",
    "rf",
]

BASELINE_EXPERIMENTS = [
    "exp1_deep_raw_attention_cnn_loso_isolated_v2",
    "exp1_deep_raw_bilstm_attention_loso_isolated_v2",
    "exp1_deep_raw_bilstm_loso_isolated_v2",
    "exp1_deep_raw_multiscale_cnn_loso_isolated_v2",
    "exp1_deep_raw_simple_cnn_loso_isolated_v2",
    "exp1_deep_raw_tcn_attn_loso_isolated_v2",
    "exp1_deep_raw_tcn_loso_isolated_v2",
    "exp1_deep_raw_bigru_loso_isolated_v2",
    "exp1_deep_raw_cnn_gru_attention_loso_isolated_v2",
    "exp1_deep_raw_cnn_lstm_loso_isolated_v2",
    "exp2_deep_emg_td_seq_attention_cnn_loso",
    "exp2_deep_emg_td_seq_bigru_loso",
    "exp2_deep_emg_td_seq_bilstm_attention_loso",
    "exp2_deep_emg_td_seq_bilstm_loso",
    "exp2_deep_emg_td_seq_cnn_gru_attention_loso",
    "exp2_deep_emg_td_seq_cnn_lstm_loso",
    "exp2_deep_emg_td_seq_multiscale_cnn_loso",
    "exp2_deep_emg_td_seq_simple_cnn_loso",
    "exp2_deep_emg_td_seq_tcn_attn_loso",
    "exp2_deep_emg_td_seq_tcn_loso",
    "exp3_deep_powerful_mlp_powerful_loso",
    "exp4_svm_linear_powerful_loso",
    "exp4_svm_rbf_powerful_loso",
    "exp4_rf_powerful_loso",
    "exp5_hybrid_powerful_deep_loso",
    "exp6_sota_best_models_aug_loso",
]


def validate_proposed_changes(proposed_changes: dict) -> list[str]:
    """Validate that proposed changes use only valid pipeline components.

    Returns a list of error messages (empty if valid).
    """
    errors = []

    model_type = proposed_changes.get("model_type", "")
    if model_type and model_type not in VALID_MODEL_TYPES + VALID_ML_MODELS:
        errors.append(
            f"Unknown model_type '{model_type}'. "
            f"Valid: {VALID_MODEL_TYPES + VALID_ML_MODELS}"
        )

    features = proposed_changes.get("features", "")
    valid_features = VALID_PIPELINE_TYPES + VALID_FEATURE_SETS
    if features and features not in valid_features:
        errors.append(
            f"Unknown features/pipeline '{features}'. Valid: {valid_features}"
        )

    augmentation = proposed_changes.get("augmentation", "")
    if augmentation and augmentation not in VALID_AUGMENTATIONS:
        errors.append(
            f"Unknown augmentation '{augmentation}'. Valid: {VALID_AUGMENTATIONS}"
        )

    return errors


def get_constraints_text() -> str:
    """Return a text description of valid pipeline constraints for LLM prompts."""
    return (
        f"VALID model_types (deep): {VALID_MODEL_TYPES}\n"
        f"VALID model_types (ML): {VALID_ML_MODELS}\n"
        f"VALID pipeline_types: {VALID_PIPELINE_TYPES}\n"
        f"VALID feature_sets: {VALID_FEATURE_SETS}\n"
        f"VALID augmentations: {VALID_AUGMENTATIONS}\n"
        "You MUST only use values from these lists in proposed_changes.\n"
        "model_type must be from VALID model_types.\n"
        "features must be from VALID pipeline_types or feature_sets.\n"
        "augmentation must be from VALID augmentations."
    )
