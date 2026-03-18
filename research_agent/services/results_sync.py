"""Service for syncing collected experiment results into Qdrant knowledge base
and updating docs/RESEARCH_HISTORY.md."""

import json
import logging
import re
from pathlib import Path

from research_agent.config import AgentConfig
from research_agent.models.hypothesis import (
    ExperimentReference,
    Hypothesis,
    Motivation,
    ProposedChanges,
)
from research_agent.services.embedding_service import EmbeddingService
from research_agent.services.experiment_service import ExperimentService
from research_agent.services.history_updater import update_research_history
from research_agent.services.vector_db import VectorStore

logger = logging.getLogger("research_agent.results_sync")


def sync_collected_results(
    config: AgentConfig,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    experiment_service: ExperimentService,
    results_dir: Path | None = None,
) -> dict:
    """Sync collected experiment results into Qdrant as verified nodes.

    Returns dict with stats: {created, skipped}.
    """
    if results_dir is None:
        results_dir = config.results_collected_path

    experiments = experiment_service.load_collected_experiments(results_dir)
    existing_ids = vector_store.get_all_experiment_ids()

    created = 0
    skipped = 0
    newly_synced_ids: set[int] = set()

    for exp in experiments:
        if exp.experiment_id is not None and exp.experiment_id in existing_ids:
            logger.debug("Skipping exp_%d (already in Qdrant)", exp.experiment_id)
            skipped += 1
            continue

        if exp.experiment_id is not None:
            newly_synced_ids.add(exp.experiment_id)

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

            exp_label = f"Exp #{exp.experiment_id}" if exp.experiment_id else exp.experiment_name
            observation = (
                f"{exp_label}: {model_name} on {feature_set} pipeline. "
                f"Mean accuracy: {agg_result.mean_accuracy:.4f} "
                f"(±{agg_result.std_accuracy:.4f}), "
                f"F1-macro: {agg_result.mean_f1_macro:.4f} "
                f"(±{agg_result.std_f1_macro:.4f}). "
                f"{agg_result.num_subjects} subjects tested."
            )

            node = Hypothesis(
                title=f"Exp #{exp.experiment_id}: {model_name} on {feature_set}"
                if exp.experiment_id
                else f"{exp.experiment_name}: {model_name}",
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
                    training_modifications=exp.training_config.get(
                        "pipeline_type", "unknown"
                    ),
                ),
                expected_effect="Verified experiment result",
                novelty_explanation=f"Collected result from experiment {exp_label}",
                strategy="collected",
                status="verified",
                source_type="collected",
                experiment_id=exp.experiment_id,
                verification_metrics={
                    "mean_accuracy": agg_result.mean_accuracy,
                    "std_accuracy": agg_result.std_accuracy,
                    "mean_f1_macro": agg_result.mean_f1_macro,
                    "std_f1_macro": agg_result.std_f1_macro,
                    "num_subjects": agg_result.num_subjects,
                },
            )

            embedding = embedding_service.embed_query(f"{summary}\n{observation}")
            vector_store.store_hypothesis(node, embedding)
            created += 1
            logger.info(
                "Created node: %s (exp_id=%s, acc=%.4f)",
                node.title,
                exp.experiment_id,
                agg_result.mean_accuracy,
            )

    # Mark existing hypothesis nodes as verified by hypothesis_id
    hypotheses_verified = _verify_hypotheses_by_id(vector_store, experiments)

    # Process _pending_qdrant_updates fallback JSONs
    pending_processed = _process_pending_qdrant_updates(vector_store, results_dir)

    # Update RESEARCH_HISTORY.md with newly synced experiments
    newly_synced = [
        exp for exp in experiments
        if exp.experiment_id in newly_synced_ids
    ]
    if newly_synced:
        history_stats = update_research_history(
            history_path=config.research_history_path,
            new_experiments=newly_synced,
        )
    else:
        history_stats = {"leaderboard_added": 0, "registry_added": 0, "hypotheses_added": 0}

    stats = {
        "created": created,
        "skipped": skipped,
        "hypotheses_verified": hypotheses_verified,
        "pending_processed": pending_processed,
        "history_updated": history_stats,
    }
    logger.info("Sync complete: %s", stats)
    return stats


def _verify_hypotheses_by_id(
    vector_store: VectorStore,
    experiments: list,
) -> int:
    """For each experiment with a hypothesis_id, update the original
    hypothesis node from 'unverified' to 'verified' with metrics."""
    verified_count = 0

    for exp in experiments:
        hyp_id = exp.hypothesis_id_str
        if not hyp_id:
            continue

        # Get best model metrics
        best_model, best_acc = exp.get_best_model()
        best_agg = exp.aggregate_results.get(best_model)
        if not best_agg:
            continue

        metrics = {
            "mean_accuracy": best_agg.mean_accuracy,
            "std_accuracy": best_agg.std_accuracy,
            "mean_f1_macro": best_agg.mean_f1_macro,
            "std_f1_macro": best_agg.std_f1_macro,
            "num_subjects": best_agg.num_subjects,
        }

        try:
            vector_store.update_hypothesis_status(hyp_id, "verified", metrics)
            verified_count += 1
            logger.info(
                "Marked hypothesis %s as verified (acc=%.4f)",
                hyp_id[:8],
                best_agg.mean_accuracy,
            )
        except Exception as e:
            logger.debug(
                "Could not update hypothesis %s (may not exist in Qdrant): %s",
                hyp_id[:8],
                e,
            )

    return verified_count


def _process_pending_qdrant_updates(
    vector_store: VectorStore,
    results_dir: Path,
) -> int:
    """Process _pending_qdrant_updates fallback JSONs from results_collected."""
    pending_dir = results_dir / "_pending_qdrant_updates"
    if not pending_dir.exists():
        return 0

    json_files = sorted(pending_dir.glob("*.json"))
    if not json_files:
        return 0

    processed = 0
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read fallback JSON %s: %s", jf.name, e)
            continue

        hypothesis_id = data.get("hypothesis_id", "")
        status = data.get("status", "")

        if not hypothesis_id or not status:
            continue

        try:
            if status == "verified":
                metrics = data.get("metrics", {})
                vector_store.update_hypothesis_status(hypothesis_id, "verified", metrics)
                logger.info("Pending update: %s marked verified", hypothesis_id[:8])
            elif status == "failed":
                error_msg = data.get("error_message", "Unknown error")
                vector_store.update_hypothesis_status(
                    hypothesis_id, "failed", {"error": error_msg}
                )
                logger.info("Pending update: %s marked failed", hypothesis_id[:8])

            # Move to _processed
            processed_dir = pending_dir / "_processed"
            processed_dir.mkdir(exist_ok=True)
            jf.rename(processed_dir / jf.name)
            processed += 1
        except Exception as e:
            logger.warning(
                "Failed to process pending update %s: %s", jf.name, e
            )

    return processed


def scan_unverified_experiments(
    experiments_dir: Path, verified_ids: set[int]
) -> list[dict]:
    """Scan experiment code files and return those without results.

    Returns list of {experiment_id, name, file_path} for experiments
    that have code but no collected results.
    """
    unverified = []
    if not experiments_dir.exists():
        return unverified

    for py_file in sorted(experiments_dir.glob("exp*_loso.py")):
        if py_file.name == "exp_X_template_loso.py":
            continue
        match = re.match(r"exp_?(\d+)_", py_file.name)
        if not match:
            continue
        exp_id = int(match.group(1))
        if exp_id not in verified_ids:
            # Extract short name from filename
            name = py_file.stem  # e.g. exp_11_enhancing_simple_cnn_...
            unverified.append({
                "experiment_id": exp_id,
                "name": name,
                "file_path": str(py_file),
            })

    return unverified
