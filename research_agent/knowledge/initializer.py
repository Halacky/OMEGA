"""Knowledge base initializer: populate Qdrant from baseline experiments."""

import logging
from datetime import datetime

from research_agent.config import AgentConfig
from research_agent.knowledge.codebase_registry import BASELINE_EXPERIMENTS
from research_agent.models.hypothesis import (
    ExperimentReference,
    Hypothesis,
    Motivation,
    ProposedChanges,
)
from research_agent.services.embedding_service import EmbeddingService
from research_agent.services.experiment_service import ExperimentService
from research_agent.services.vector_db import VectorStore

logger = logging.getLogger("research_agent.initializer")


def initialize_knowledge_base(
    config: AgentConfig,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    experiment_service: ExperimentService,
) -> int:
    """Populate Qdrant with baseline experiment knowledge nodes.

    Returns the number of nodes created.
    """
    if not vector_store.is_hypotheses_empty():
        logger.info("Knowledge base already populated, skipping initialization")
        return 0

    logger.info("Initializing knowledge base from %d baseline experiments", len(BASELINE_EXPERIMENTS))
    baselines = experiment_service.load_baseline_experiments(BASELINE_EXPERIMENTS)

    count = 0
    for exp in baselines:
        for model_name, agg_result in exp.aggregate_results.items():
            summary = exp.to_summary_text()

            pipeline_type = exp.training_config.get("pipeline_type", "unknown")
            feature_set = exp.feature_set or exp.approach or pipeline_type
            aug_apply = exp.training_config.get("aug_apply", False)
            aug_desc = "none"
            if aug_apply:
                parts = []
                if exp.training_config.get("aug_apply_noise", False):
                    parts.append("noise")
                if exp.training_config.get("aug_apply_time_warp", False):
                    parts.append("time_warp")
                aug_desc = "+".join(parts) if parts else "noise"

            observation = (
                f"Baseline experiment with {model_name} on {feature_set} pipeline. "
                f"Mean accuracy: {agg_result.mean_accuracy:.4f} "
                f"(±{agg_result.std_accuracy:.4f}), "
                f"F1-macro: {agg_result.mean_f1_macro:.4f} "
                f"(±{agg_result.std_f1_macro:.4f}). "
                f"{agg_result.num_subjects} subjects tested."
            )

            knowledge_node = Hypothesis(
                title=f"Baseline: {model_name} on {feature_set}",
                hypothesis_text=observation,
                motivation=Motivation(
                    based_on_experiments=[
                        ExperimentReference(
                            experiment_name=exp.experiment_name,
                            observation=observation,
                        )
                    ],
                    based_on_papers=[],
                ),
                proposed_changes=ProposedChanges(
                    model_type=model_name,
                    features=feature_set,
                    augmentation=aug_desc,
                    training_modifications="baseline configuration",
                ),
                expected_effect="Baseline measurement",
                novelty_explanation="Baseline experiment — not a hypothesis",
                strategy="baseline",
                status="verified",
                source_type="baseline",
                verification_metrics={
                    "mean_accuracy": agg_result.mean_accuracy,
                    "std_accuracy": agg_result.std_accuracy,
                    "mean_f1_macro": agg_result.mean_f1_macro,
                    "std_f1_macro": agg_result.std_f1_macro,
                    "num_subjects": agg_result.num_subjects,
                },
            )

            embedding = embedding_service.embed_query(
                f"{summary}\n{observation}"
            )
            vector_store.store_hypothesis(knowledge_node, embedding)
            count += 1
            logger.debug(
                "Created knowledge node: %s / %s (acc=%.4f)",
                exp.experiment_name,
                model_name,
                agg_result.mean_accuracy,
            )

    logger.info("Knowledge base initialized with %d baseline nodes", count)
    return count
