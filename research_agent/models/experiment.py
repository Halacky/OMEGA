"""Pydantic models for experiment data parsed from loso_summary.json."""

from typing import Optional

from pydantic import BaseModel, Field


class SubjectResult(BaseModel):
    """Per-subject LOSO fold result."""
    test_subject: str
    test_accuracy: Optional[float] = None
    test_f1_macro: Optional[float] = None


class AggregateResult(BaseModel):
    """Aggregate metrics across all subjects for one model."""
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_f1_macro: float = 0.0
    std_f1_macro: float = 0.0
    num_subjects: int = 0
    per_subject: list[SubjectResult] = Field(default_factory=list)


class ExperimentSummary(BaseModel):
    """Parsed loso_summary.json for one experiment."""
    experiment_name: str
    experiment_id: Optional[int] = None
    feature_set: str = ""
    approach: str = ""
    models: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    exercises: list[str] = Field(default_factory=list)
    processing_config: dict = Field(default_factory=dict)
    split_config: dict = Field(default_factory=dict)
    training_config: dict = Field(default_factory=dict)
    aggregate_results: dict[str, AggregateResult] = Field(default_factory=dict)
    # Metadata for history updater
    hypothesis_id_str: Optional[str] = None
    note: str = ""
    augmentation_desc: str = ""
    training_modifications_desc: dict = Field(default_factory=dict)
    test_time_adaptation_desc: dict = Field(default_factory=dict)

    @property
    def hypothesis_id_short(self) -> str:
        """Return first 8 chars of hypothesis UUID, or empty string."""
        if self.hypothesis_id_str:
            return self.hypothesis_id_str[:8]
        return ""

    def get_best_model(self) -> tuple[str, float]:
        """Return (model_name, best_accuracy) for this experiment."""
        best_model = ""
        best_acc = 0.0
        for model_name, result in self.aggregate_results.items():
            if result.mean_accuracy > best_acc:
                best_acc = result.mean_accuracy
                best_model = model_name
        return best_model, best_acc

    def to_summary_text(self) -> str:
        """Create text summary for embedding / LLM context."""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Feature set: {self.feature_set or self.approach}",
            f"Pipeline: {self.training_config.get('pipeline_type', 'unknown')}",
            f"Subjects: {len(self.subjects)}",
        ]
        for model_name, result in self.aggregate_results.items():
            lines.append(
                f"  Model {model_name}: accuracy={result.mean_accuracy:.4f} "
                f"(±{result.std_accuracy:.4f}), "
                f"f1={result.mean_f1_macro:.4f} (±{result.std_f1_macro:.4f})"
            )
        aug = self.training_config.get("aug_apply", False)
        if aug:
            lines.append(
                f"  Augmentation: noise_std={self.training_config.get('aug_noise_std')}, "
                f"time_warp={self.training_config.get('aug_apply_time_warp', False)}"
            )
        else:
            lines.append("  Augmentation: none")
        return "\n".join(lines)
