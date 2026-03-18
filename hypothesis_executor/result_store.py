"""Stores experiment results and updates hypothesis status in Qdrant."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hypothesis_executor.result_store")


class ResultStore:
    """Lightweight Qdrant client for updating hypothesis status after experiments.

    Uses qdrant_client directly — no dependency on research_agent's VectorStore
    class, but writes to the same Qdrant database so the research agent sees updates.
    """

    COLLECTION = "hypotheses"

    def __init__(self, project_root: Path):
        self.qdrant_path = project_root / "research_agent" / "qdrant_data"
        self._client = None

    @property
    def client(self):
        """Lazy-init Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient
            if not self.qdrant_path.exists():
                logger.warning(
                    "Qdrant data path does not exist: %s. "
                    "Will create on first write.",
                    self.qdrant_path,
                )
            self._client = QdrantClient(path=str(self.qdrant_path))
            logger.info("Connected to Qdrant at: %s", self.qdrant_path)
        return self._client

    def mark_verified(self, hypothesis_id: str, metrics: dict) -> bool:
        """Mark a hypothesis as verified with experiment metrics.

        Args:
            hypothesis_id: The hypothesis UUID string.
            metrics: Dict with mean_accuracy, std_accuracy, mean_f1_macro, etc.

        Returns:
            True if update succeeded, False otherwise.
        """
        try:
            self.client.set_payload(
                collection_name=self.COLLECTION,
                payload={
                    "status": "verified",
                    "verification_metrics": metrics,
                },
                points=[hypothesis_id],
            )
            logger.info(
                "Hypothesis %s marked as verified (acc=%.4f)",
                hypothesis_id,
                metrics.get("mean_accuracy", 0.0),
            )
            return True
        except Exception as e:
            logger.error("Failed to mark hypothesis verified: %s", e)
            return False

    def mark_failed(self, hypothesis_id: str, error_message: str) -> bool:
        """Mark a hypothesis as failed with error details.

        Args:
            hypothesis_id: The hypothesis UUID string.
            error_message: Description of what went wrong.

        Returns:
            True if update succeeded, False otherwise.
        """
        try:
            self.client.set_payload(
                collection_name=self.COLLECTION,
                payload={
                    "status": "failed",
                    "verification_metrics": {"error": error_message},
                },
                points=[hypothesis_id],
            )
            self.add_execution_log(
                hypothesis_id,
                f"FAILED: {error_message[:500]}",
            )
            logger.info("Hypothesis %s marked as failed", hypothesis_id)
            return True
        except Exception as e:
            logger.error("Failed to mark hypothesis failed: %s", e)
            return False

    def add_execution_log(
        self, hypothesis_id: str, text: str
    ) -> bool:
        """Add an execution log entry as a comment on the hypothesis.

        Args:
            hypothesis_id: The hypothesis UUID string.
            text: Log text to add.

        Returns:
            True if update succeeded, False otherwise.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Get existing comments
            results = self.client.scroll(
                collection_name=self.COLLECTION,
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

            new_comment = {
                "author": "hypothesis_executor",
                "text": text,
                "timestamp": datetime.now().isoformat(),
            }
            existing_comments.append(new_comment)

            self.client.set_payload(
                collection_name=self.COLLECTION,
                payload={"comments": existing_comments},
                points=[hypothesis_id],
            )
            return True

        except Exception as e:
            logger.error("Failed to add execution log: %s", e)
            return False

    def is_available(self) -> bool:
        """Check if the Qdrant store is accessible and has the hypotheses collection."""
        try:
            collections = [
                c.name for c in self.client.get_collections().collections
            ]
            return self.COLLECTION in collections
        except Exception:
            return False
