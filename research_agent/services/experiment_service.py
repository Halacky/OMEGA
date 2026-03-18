"""Service for loading and analyzing experiment results from experiments_output/ and results_collected/."""

import json
import logging
import re
from pathlib import Path

from research_agent.config import AgentConfig
from research_agent.models.experiment import (
    AggregateResult,
    ExperimentSummary,
    SubjectResult,
)

logger = logging.getLogger("research_agent.experiments")


def _normalize_augmentation_desc(raw: object) -> str:
    """Convert augmentation field to string (may be str or dict in JSON)."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        parts = []
        for k, v in raw.items():
            if isinstance(v, bool) and v:
                parts.append(k)
            elif isinstance(v, str):
                parts.append(f"{k}: {v}")
        return ", ".join(parts) if parts else str(raw)
    return str(raw) if raw else ""


class ExperimentService:
    """Loads and analyzes experiment results from loso_summary.json files."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.output_dir = config.experiments_output_path
        self._cache: list[ExperimentSummary] | None = None

    def load_all_experiments(self) -> list[ExperimentSummary]:
        """Load all experiments that have a loso_summary.json."""
        if self._cache is not None:
            return self._cache

        experiments = []
        if not self.output_dir.exists():
            logger.warning("Experiments output dir not found: %s", self.output_dir)
            return experiments

        for exp_dir in sorted(self.output_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            summary_file = exp_dir / "loso_summary.json"
            if summary_file.exists():
                try:
                    exp = self._parse_summary(summary_file)
                    experiments.append(exp)
                    logger.debug("Loaded experiment: %s", exp.experiment_name)
                except Exception as e:
                    logger.warning("Failed to parse %s: %s", summary_file, e)
            else:
                # Fallback: reconstruct from per-fold fold_metadata.json files
                try:
                    exp = self._reconstruct_from_folds(exp_dir)
                    if exp is not None:
                        experiments.append(exp)
                        logger.debug(
                            "Reconstructed experiment from folds: %s",
                            exp.experiment_name,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to reconstruct %s: %s", exp_dir.name, e
                    )

        logger.info("Loaded %d experiments total", len(experiments))
        self._cache = experiments
        return experiments

    def load_baseline_experiments(
        self, baseline_names: list[str]
    ) -> list[ExperimentSummary]:
        """Load only the specified baseline experiments."""
        all_exps = self.load_all_experiments()
        baselines = [e for e in all_exps if e.experiment_name in baseline_names]
        logger.info(
            "Loaded %d/%d baseline experiments",
            len(baselines),
            len(baseline_names),
        )
        return baselines

    def get_best_experiments(self, top_k: int = 5) -> list[tuple[str, str, float]]:
        """Return top-k (experiment_name, model_name, mean_accuracy) tuples."""
        results = []
        for exp in self.load_all_experiments():
            for model_name, agg in exp.aggregate_results.items():
                results.append((exp.experiment_name, model_name, agg.mean_accuracy))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def get_worst_subjects(
        self, top_k: int = 10
    ) -> list[tuple[str, str, str, float]]:
        """Return top-k worst (experiment, model, subject, accuracy) tuples."""
        results = []
        for exp in self.load_all_experiments():
            for model_name, agg in exp.aggregate_results.items():
                for subj in agg.per_subject:
                    if subj.test_accuracy is not None:
                        results.append(
                            (
                                exp.experiment_name,
                                model_name,
                                subj.test_subject,
                                subj.test_accuracy,
                            )
                        )
        results.sort(key=lambda x: x[3])
        return results[:top_k]

    def get_untested_combinations(self) -> list[dict]:
        """Identify model+pipeline combinations not yet tested."""
        from research_agent.knowledge.codebase_registry import (
            VALID_MODEL_TYPES,
            VALID_PIPELINE_TYPES,
        )

        tested = set()
        for exp in self.load_all_experiments():
            pipeline = exp.training_config.get("pipeline_type", "")
            for model_name in exp.aggregate_results:
                tested.add((model_name, pipeline))

        untested = []
        for model in VALID_MODEL_TYPES:
            for pipeline in VALID_PIPELINE_TYPES:
                if (model, pipeline) not in tested:
                    untested.append({"model_type": model, "pipeline": pipeline})
        return untested

    def get_all_experiments_summary_text(self) -> str:
        """Create a combined text summary of all experiments for LLM context."""
        summaries = []
        for exp in self.load_all_experiments():
            summaries.append(exp.to_summary_text())
        return "\n\n".join(summaries)

    def _reconstruct_from_folds(self, exp_dir: Path) -> ExperimentSummary | None:
        """Reconstruct experiment summary from per-fold fold_metadata.json files.

        Used when loso_summary.json is missing but individual fold results exist.
        """
        import numpy as np

        # Find model subdirectories (they contain test_DB2_s* folders)
        aggregate_results = {}
        training_config = {}
        processing_config = {}
        split_config = {}
        all_subjects = set()
        exercises = []

        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("pipeline"):
                continue

            model_name = model_dir.name
            per_subject = []

            for fold_dir in sorted(model_dir.iterdir()):
                if not fold_dir.is_dir():
                    continue
                meta_file = fold_dir / "fold_metadata.json"
                if not meta_file.exists():
                    continue

                with open(meta_file) as f:
                    meta = json.load(f)

                metrics = meta.get("metrics", {})
                test_subject = meta.get("test_subject", fold_dir.name)
                all_subjects.add(test_subject)

                per_subject.append(
                    SubjectResult(
                        test_subject=test_subject,
                        test_accuracy=metrics.get("test_accuracy"),
                        test_f1_macro=metrics.get("test_f1_macro"),
                    )
                )

                # Grab config from first fold
                if not training_config:
                    config = meta.get("config", {})
                    training_config = config.get("training", {})
                    processing_config = config.get("processing", {})
                    split_config = config.get("split", {})
                    exercises = meta.get("exercises", [])

            if per_subject:
                accs = [
                    s.test_accuracy
                    for s in per_subject
                    if s.test_accuracy is not None
                ]
                f1s = [
                    s.test_f1_macro
                    for s in per_subject
                    if s.test_f1_macro is not None
                ]
                aggregate_results[model_name] = AggregateResult(
                    mean_accuracy=float(np.mean(accs)) if accs else 0.0,
                    std_accuracy=float(np.std(accs)) if accs else 0.0,
                    mean_f1_macro=float(np.mean(f1s)) if f1s else 0.0,
                    std_f1_macro=float(np.std(f1s)) if f1s else 0.0,
                    num_subjects=len(per_subject),
                    per_subject=per_subject,
                )

        if not aggregate_results:
            return None

        pipeline_type = training_config.get("pipeline_type", "")
        feature_set = pipeline_type or "unknown"

        return ExperimentSummary(
            experiment_name=exp_dir.name,
            feature_set=feature_set,
            models=list(aggregate_results.keys()),
            subjects=sorted(all_subjects),
            exercises=exercises,
            processing_config=processing_config,
            split_config=split_config,
            training_config=training_config,
            aggregate_results=aggregate_results,
        )

    def load_collected_experiments(
        self, results_dir: Path | None = None
    ) -> list[ExperimentSummary]:
        """Load experiments from results_collected directory (flat JSON format)."""
        if results_dir is None:
            results_dir = self.config.results_collected_path

        experiments = []
        if not results_dir.exists():
            logger.warning("Results collected dir not found: %s", results_dir)
            return experiments

        for item in sorted(results_dir.iterdir()):
            if not item.is_dir():
                continue
            summary_file = item / "loso_summary.json"
            if not summary_file.exists():
                continue
            try:
                exp = self._parse_collected_summary(summary_file)
                experiments.append(exp)
                logger.debug(
                    "Loaded collected experiment: %s (exp_id=%s)",
                    exp.experiment_name,
                    exp.experiment_id,
                )
            except Exception as e:
                logger.warning("Failed to parse collected %s: %s", summary_file, e)

        logger.info("Loaded %d collected experiments", len(experiments))
        return experiments

    @staticmethod
    def extract_experiment_id(name: str) -> int | None:
        """Extract experiment number from name like 'exp_7_...' or 'exp7_...'."""
        match = re.match(r"exp_?(\d+)_", name)
        return int(match.group(1)) if match else None

    def _parse_collected_summary(self, path: Path) -> ExperimentSummary:
        """Parse a collected loso_summary.json.

        Handles two JSON schemas:
          Schema A (standard): experiment_name, aggregate_results, individual_results
          Schema B (alt):      experiment, aggregate, results
        """
        with open(path) as f:
            data = json.load(f)

        # --- Experiment name: "experiment_name" (A) or "experiment" (B) ---
        experiment_name = (
            data.get("experiment_name")
            or data.get("experiment")
            or path.parent.name
        )
        experiment_id = self.extract_experiment_id(experiment_name)

        # --- Model type ---
        model_type = (
            data.get("model_type")
            or data.get("model")
            or "unknown"
        )
        # Schema B: model_type may be inside individual results
        individual_results = data.get("individual_results") or data.get("results", [])
        if model_type == "unknown" and individual_results:
            model_type = individual_results[0].get("model_type", "unknown")

        # --- Per-subject results ---
        per_subject = []
        for r in individual_results:
            per_subject.append(
                SubjectResult(
                    test_subject=r.get("test_subject", ""),
                    test_accuracy=r.get("test_accuracy"),
                    test_f1_macro=r.get("test_f1_macro"),
                )
            )

        # --- Aggregate results: "aggregate_results" (A) or "aggregate" (B) ---
        raw_agg = data.get("aggregate_results") or data.get("aggregate", {})

        if "mean_accuracy" in raw_agg:
            # Flat format — wrap into {model_type: AggregateResult}
            num_subjects = (
                raw_agg.get("num_subjects")
                or raw_agg.get("num_folds")
                or len(per_subject)
            )
            aggregate_results = {
                model_type: AggregateResult(
                    mean_accuracy=raw_agg.get("mean_accuracy", 0.0),
                    std_accuracy=raw_agg.get("std_accuracy", 0.0),
                    mean_f1_macro=raw_agg.get("mean_f1_macro", 0.0),
                    std_f1_macro=raw_agg.get("std_f1_macro", 0.0),
                    num_subjects=num_subjects,
                    per_subject=per_subject,
                )
            }
        else:
            # Already keyed by model — use standard parser logic
            aggregate_results = {}
            for mn, md in raw_agg.items():
                ps = [
                    SubjectResult(
                        test_subject=s.get("test_subject", ""),
                        test_accuracy=s.get("test_accuracy"),
                        test_f1_macro=s.get("test_f1_macro"),
                    )
                    for s in md.get("per_subject", [])
                ]
                aggregate_results[mn] = AggregateResult(
                    mean_accuracy=md.get("mean_accuracy", 0.0),
                    std_accuracy=md.get("std_accuracy", 0.0),
                    mean_f1_macro=md.get("mean_f1_macro", 0.0),
                    std_f1_macro=md.get("std_f1_macro", 0.0),
                    num_subjects=md.get("num_subjects", 0),
                    per_subject=ps,
                )

        # --- Approach / pipeline ---
        approach = data.get("approach", "")
        training_config = data.get("training_config", {})
        # Schema B often puts pipeline info in "approach" rather than training_config
        if not training_config.get("pipeline_type") and approach:
            training_config = {**training_config, "pipeline_type": approach}
        if not training_config.get("model_type") and model_type != "unknown":
            training_config = {**training_config, "model_type": model_type}

        # --- Hypothesis: "hypothesis_id" (UUID, A) or "hypothesis" (text, B) ---
        hypothesis_id = data.get("hypothesis_id")
        note = data.get("note", "")
        # Schema B may have "hypothesis" as a text description, not a UUID
        hypothesis_text = data.get("hypothesis", "")
        if hypothesis_text and not hypothesis_id:
            # Use as note if it looks like a description
            if not note:
                note = hypothesis_text

        models_list = data.get("models", [model_type] if model_type != "unknown" else [])

        return ExperimentSummary(
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            feature_set=data.get("feature_set", ""),
            approach=approach,
            models=models_list,
            subjects=data.get("subjects", []),
            exercises=data.get("exercises", []),
            processing_config=data.get("processing_config", {}),
            split_config=data.get("split_config", {}),
            training_config=training_config,
            aggregate_results=aggregate_results,
            hypothesis_id_str=hypothesis_id,
            note=note,
            augmentation_desc=_normalize_augmentation_desc(data.get("augmentation", "")),
            training_modifications_desc=data.get("training_modifications", {}),
            test_time_adaptation_desc=data.get("test_time_adaptation", {}),
        )

    def _parse_summary(self, path: Path) -> ExperimentSummary:
        """Parse a single loso_summary.json into ExperimentSummary."""
        with open(path) as f:
            data = json.load(f)

        aggregate_results = {}
        raw_agg = data.get("aggregate_results", {})
        for model_name, model_data in raw_agg.items():
            per_subject = []
            for s in model_data.get("per_subject", []):
                per_subject.append(
                    SubjectResult(
                        test_subject=s.get("test_subject", ""),
                        test_accuracy=s.get("test_accuracy"),
                        test_f1_macro=s.get("test_f1_macro"),
                    )
                )
            aggregate_results[model_name] = AggregateResult(
                mean_accuracy=model_data.get("mean_accuracy", 0.0),
                std_accuracy=model_data.get("std_accuracy", 0.0),
                mean_f1_macro=model_data.get("mean_f1_macro", 0.0),
                std_f1_macro=model_data.get("std_f1_macro", 0.0),
                num_subjects=model_data.get("num_subjects", 0),
                per_subject=per_subject,
            )

        return ExperimentSummary(
            experiment_name=data.get("experiment_name", path.parent.name),
            feature_set=data.get("feature_set", ""),
            approach=data.get("approach", ""),
            models=data.get("models", []),
            subjects=data.get("subjects", []),
            exercises=data.get("exercises", []),
            processing_config=data.get("processing_config", {}),
            split_config=data.get("split_config", {}),
            training_config=data.get("training_config", {}),
            aggregate_results=aggregate_results,
        )
