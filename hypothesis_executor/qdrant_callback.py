"""Callback for generated experiment scripts to update hypothesis status in Qdrant.

Usage in generated experiments:
    from hypothesis_executor.qdrant_callback import mark_hypothesis_verified, mark_hypothesis_failed

    # At the end of main(), after loso_summary.json is saved:
    mark_hypothesis_verified(
        hypothesis_id="<uuid>",
        metrics=aggregate["model_name"],  # dict with mean_accuracy, std_accuracy, etc.
        experiment_name=EXPERIMENT_NAME,
    )

    # Or on failure:
    mark_hypothesis_failed(
        hypothesis_id="<uuid>",
        error_message="description of what went wrong",
    )
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("hypothesis_executor.qdrant_callback")

# Default Qdrant data path (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_QDRANT_PATH = _PROJECT_ROOT / "research_agent" / "qdrant_data"
_COLLECTION = "hypotheses"


def _get_client():
    """Create a Qdrant client. Returns None if qdrant_client is not installed."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        logger.warning("qdrant-client not installed, skipping Qdrant update")
        return None

    if not _QDRANT_PATH.exists():
        logger.warning("Qdrant data path does not exist: %s", _QDRANT_PATH)
        return None

    return QdrantClient(path=str(_QDRANT_PATH))


def _save_fallback_json(hypothesis_id: str, status: str, data: dict) -> None:
    """Save result to fallback JSON when Qdrant is unavailable (e.g. remote server)."""
    fallback_dir = _PROJECT_ROOT / "experiments_output" / "_pending_qdrant_updates"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "hypothesis_id": hypothesis_id,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        **data,
    }
    fname = f"{hypothesis_id[:8]}_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = fallback_dir / fname
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    print(f"\n[Qdrant Fallback] Saved to {path}")


def mark_hypothesis_verified(
    hypothesis_id: str,
    metrics: dict,
    experiment_name: str = "",
) -> bool:
    """Mark a hypothesis as verified in Qdrant with experiment metrics.

    Safe to call even if Qdrant is unavailable — will save a fallback JSON
    that can be ingested later by collect_results.py on the host machine.

    Args:
        hypothesis_id: UUID string of the hypothesis.
        metrics: Dict with mean_accuracy, std_accuracy, mean_f1_macro, std_f1_macro, num_subjects.
        experiment_name: Name of the experiment for the log entry.

    Returns:
        True if Qdrant was updated, False otherwise.
    """
    if not hypothesis_id:
        logger.warning("No hypothesis_id provided, skipping Qdrant update")
        return False

    client = _get_client()
    if client is None:
        _save_fallback_json(hypothesis_id, "verified", {
            "metrics": metrics,
            "experiment_name": experiment_name,
        })
        return False

    try:
        # Update status and metrics
        client.set_payload(
            collection_name=_COLLECTION,
            payload={
                "status": "verified",
                "verification_metrics": metrics,
            },
            points=[hypothesis_id],
        )

        # Add execution log comment
        _add_comment(
            client,
            hypothesis_id,
            f"Experiment '{experiment_name}' completed successfully: "
            f"acc={metrics.get('mean_accuracy', 0):.4f}±{metrics.get('std_accuracy', 0):.4f}, "
            f"f1={metrics.get('mean_f1_macro', 0):.4f}±{metrics.get('std_f1_macro', 0):.4f} "
            f"(n={metrics.get('num_subjects', '?')} subjects)",
        )

        logger.info(
            "Hypothesis %s marked as VERIFIED (acc=%.4f)",
            hypothesis_id[:8],
            metrics.get("mean_accuracy", 0),
        )
        print(f"\n[Qdrant] Hypothesis {hypothesis_id[:8]}... marked as VERIFIED")
        return True

    except Exception as e:
        logger.error("Failed to update Qdrant: %s", e)
        print(f"\n[Qdrant] WARNING: Failed to update hypothesis status: {e}")
        _save_fallback_json(hypothesis_id, "verified", {
            "metrics": metrics,
            "experiment_name": experiment_name,
        })
        return False


def mark_hypothesis_failed(
    hypothesis_id: str,
    error_message: str,
) -> bool:
    """Mark a hypothesis as failed in Qdrant.

    Safe to call even if Qdrant is unavailable.

    Args:
        hypothesis_id: UUID string of the hypothesis.
        error_message: Description of what went wrong.

    Returns:
        True if Qdrant was updated, False otherwise.
    """
    if not hypothesis_id:
        logger.warning("No hypothesis_id provided, skipping Qdrant update")
        return False

    client = _get_client()
    if client is None:
        _save_fallback_json(hypothesis_id, "failed", {
            "error_message": error_message[:1000],
        })
        return False

    try:
        client.set_payload(
            collection_name=_COLLECTION,
            payload={
                "status": "failed",
                "verification_metrics": {"error": error_message[:1000]},
            },
            points=[hypothesis_id],
        )

        _add_comment(
            client,
            hypothesis_id,
            f"FAILED: {error_message[:500]}",
        )

        logger.info("Hypothesis %s marked as FAILED", hypothesis_id[:8])
        print(f"\n[Qdrant] Hypothesis {hypothesis_id[:8]}... marked as FAILED")
        return True

    except Exception as e:
        logger.error("Failed to update Qdrant: %s", e)
        print(f"\n[Qdrant] WARNING: Failed to update hypothesis status: {e}")
        _save_fallback_json(hypothesis_id, "failed", {
            "error_message": error_message[:1000],
        })
        return False


def _add_comment(client, hypothesis_id: str, text: str) -> None:
    """Append a comment to the hypothesis in Qdrant."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = client.scroll(
            collection_name=_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="id",
                        match=MatchValue(value=hypothesis_id),
                    )
                ]
            ),
            limit=1,
            with_vectors=False,
        )

        existing_comments = []
        if results[0]:
            existing_comments = results[0][0].payload.get("comments", [])

        existing_comments.append({
            "author": "experiment_script",
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })

        client.set_payload(
            collection_name=_COLLECTION,
            payload={"comments": existing_comments},
            points=[hypothesis_id],
        )
    except Exception as e:
        logger.debug("Failed to add comment: %s", e)
