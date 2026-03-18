"""Qdrant vector database service for hypotheses and papers."""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from research_agent.config import AgentConfig
from research_agent.models.hypothesis import Hypothesis
from research_agent.models.paper import Paper

logger = logging.getLogger("research_agent.vector_db")

COLLECTION_HYPOTHESES = "hypotheses"
COLLECTION_PAPERS = "papers"


class VectorStore:
    """Qdrant-backed vector store for hypotheses and papers."""

    def __init__(self, config: AgentConfig):
        self.config = config
        if config.qdrant_mode == "embedded":
            logger.info("Connecting to Qdrant in embedded mode: %s", config.qdrant_path)
            self.client = QdrantClient(path=config.qdrant_path)
        else:
            logger.info("Connecting to Qdrant server: %s", config.qdrant_url)
            self.client = QdrantClient(url=config.qdrant_url)

    def init_collections(self, vector_size: int) -> None:
        """Create collections if they don't exist."""
        for name in [COLLECTION_HYPOTHESES, COLLECTION_PAPERS]:
            existing = [c.name for c in self.client.get_collections().collections]
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created collection: %s (dim=%d)", name, vector_size)
            else:
                logger.debug("Collection already exists: %s", name)

    def is_hypotheses_empty(self) -> bool:
        """Check if the hypotheses collection is empty."""
        try:
            info = self.client.get_collection(COLLECTION_HYPOTHESES)
            return info.points_count == 0
        except Exception:
            return True

    # ---- Hypotheses ----

    def store_hypothesis(self, hypothesis: Hypothesis, embedding: list[float]) -> None:
        """Store a hypothesis with its embedding vector."""
        point = PointStruct(
            id=hypothesis.id,
            vector=embedding,
            payload=hypothesis.model_dump(),
        )
        self.client.upsert(
            collection_name=COLLECTION_HYPOTHESES,
            points=[point],
        )
        logger.info("Stored hypothesis: %s [%s]", hypothesis.title, hypothesis.id)

    def find_similar_hypotheses(
        self, embedding: list[float], threshold: float = 0.85, limit: int = 5
    ) -> list[tuple[Hypothesis, float]]:
        """Find hypotheses similar to the given embedding.

        Returns list of (hypothesis, similarity_score) tuples.
        """
        results = self.client.query_points(
            collection_name=COLLECTION_HYPOTHESES,
            query=embedding,
            limit=limit,
        )
        similar = []
        for point in results.points:
            score = point.score
            if score >= threshold:
                hyp = Hypothesis(**point.payload)
                similar.append((hyp, score))
        return similar

    def get_max_similarity(self, embedding: list[float]) -> float:
        """Get the maximum similarity score for an embedding."""
        results = self.client.query_points(
            collection_name=COLLECTION_HYPOTHESES,
            query=embedding,
            limit=1,
        )
        if results.points:
            return results.points[0].score
        return 0.0

    def update_hypothesis_status(
        self, hypothesis_id: str, status: str, metrics: Optional[dict] = None
    ) -> None:
        """Update a hypothesis status (e.g., mark as verified with metrics)."""
        payload_update = {"status": status}
        if metrics:
            payload_update["verification_metrics"] = metrics
        self.client.set_payload(
            collection_name=COLLECTION_HYPOTHESES,
            payload=payload_update,
            points=[hypothesis_id],
        )
        logger.info("Updated hypothesis %s: status=%s", hypothesis_id, status)

    def delete_hypothesis(self, hypothesis_id: str) -> None:
        """Delete a hypothesis from the collection."""
        self.client.delete(
            collection_name=COLLECTION_HYPOTHESES,
            points_selector=[hypothesis_id],
        )
        logger.info("Deleted hypothesis: %s", hypothesis_id)

    def add_comment_to_hypothesis(
        self, hypothesis_id: str, author: str, text: str
    ) -> None:
        """Add a comment to a hypothesis."""
        from datetime import datetime

        # First get existing comments
        results = self.client.scroll(
            collection_name=COLLECTION_HYPOTHESES,
            scroll_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=hypothesis_id))]
            ),
            limit=1,
            with_vectors=False,
        )
        existing_comments = []
        if results[0]:
            existing_comments = results[0][0].payload.get("comments", [])

        new_comment = {
            "author": author,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }
        existing_comments.append(new_comment)

        self.client.set_payload(
            collection_name=COLLECTION_HYPOTHESES,
            payload={"comments": existing_comments},
            points=[hypothesis_id],
        )
        logger.info("Added comment to hypothesis %s by %s", hypothesis_id, author)

    def get_all_hypotheses(self) -> list[Hypothesis]:
        """Retrieve all hypotheses from the collection."""
        all_points = []
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=COLLECTION_HYPOTHESES,
                limit=100,
                offset=offset,
                with_vectors=False,
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        hypotheses = []
        for point in all_points:
            try:
                hypotheses.append(Hypothesis(**point.payload))
            except Exception as e:
                logger.warning("Failed to parse hypothesis %s: %s", point.id, e)
        return hypotheses

    def get_hypotheses_by_status(self, status: str) -> list[Hypothesis]:
        """Get hypotheses filtered by status."""
        result = self.client.scroll(
            collection_name=COLLECTION_HYPOTHESES,
            scroll_filter=Filter(
                must=[FieldCondition(key="status", match=MatchValue(value=status))]
            ),
            limit=1000,
            with_vectors=False,
        )
        return [Hypothesis(**p.payload) for p in result[0]]

    def get_all_experiment_ids(self) -> set[int]:
        """Return all experiment_ids stored in Qdrant hypotheses collection."""
        all_hyps = self.get_all_hypotheses()
        return {
            h.experiment_id
            for h in all_hyps
            if h.experiment_id is not None
        }

    def get_hypotheses_by_experiment_id(self, experiment_id: int) -> list[Hypothesis]:
        """Get hypotheses filtered by experiment_id."""
        result = self.client.scroll(
            collection_name=COLLECTION_HYPOTHESES,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="experiment_id",
                        match=MatchValue(value=experiment_id),
                    )
                ]
            ),
            limit=100,
            with_vectors=False,
        )
        return [Hypothesis(**p.payload) for p in result[0]]

    # ---- Papers ----

    def store_paper(self, paper: Paper, embedding: list[float]) -> None:
        """Store a paper with its embedding vector."""
        point = PointStruct(
            id=paper.arxiv_id.replace(".", "_").replace("/", "_"),
            vector=embedding,
            payload=paper.model_dump(),
        )
        self.client.upsert(
            collection_name=COLLECTION_PAPERS,
            points=[point],
        )
        logger.info("Stored paper: %s [%s]", paper.title[:60], paper.arxiv_id)

    def is_paper_indexed(self, arxiv_id: str) -> bool:
        """Check if a paper is already indexed."""
        try:
            result = self.client.scroll(
                collection_name=COLLECTION_PAPERS,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="arxiv_id", match=MatchValue(value=arxiv_id)
                        )
                    ]
                ),
                limit=1,
                with_vectors=False,
            )
            return len(result[0]) > 0
        except Exception:
            return False

    def find_similar_papers(
        self, embedding: list[float], limit: int = 10
    ) -> list[Paper]:
        """Find papers similar to the given embedding."""
        results = self.client.query_points(
            collection_name=COLLECTION_PAPERS,
            query=embedding,
            limit=limit,
        )
        return [Paper(**p.payload) for p in results.points]

    def get_all_papers(self) -> list[Paper]:
        """Retrieve all indexed papers."""
        result = self.client.scroll(
            collection_name=COLLECTION_PAPERS,
            limit=1000,
            with_vectors=False,
        )
        return [Paper(**p.payload) for p in result[0]]
