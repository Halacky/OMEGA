"""arXiv paper search and indexing service."""

import logging
import time

import arxiv

from research_agent.config import AgentConfig
from research_agent.models.paper import Paper
from research_agent.services.embedding_service import EmbeddingService
from research_agent.services.vector_db import VectorStore

logger = logging.getLogger("research_agent.papers")

STRATEGY_QUERIES = {
    "exploitation": [
        "sEMG gesture recognition deep learning",
        "EMG hand gesture classification CNN",
    ],
    "exploration": [
        "sEMG gesture recognition",
        "EMG signal classification novel architecture",
        "attention mechanism EMG signal",
        "data augmentation EMG classification",
    ],
    "literature": [
        "sEMG gesture recognition deep learning LOSO",
        "cross-subject EMG transfer learning",
        "temporal convolutional network electromyography",
        "domain adaptation electromyography",
        "contrastive learning EMG signals",
        "EMG signal processing feature extraction",
    ],
    "error": [
        "cross-subject variability EMG",
        "EMG subject adaptation",
        "robust EMG classification",
    ],
}


class PaperService:
    """Searches arXiv and indexes papers in the vector store."""

    def __init__(
        self,
        config: AgentConfig,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ):
        self.config = config
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.arxiv_client = arxiv.Client(
            page_size=10,
            delay_seconds=3.0,
            num_retries=3,
        )

    def search_and_index(
        self,
        queries: list[str] | None = None,
        strategy: str = "literature",
        max_results_per_query: int | None = None,
    ) -> list[Paper]:
        """Search arXiv and index new papers. Returns all found papers."""
        if queries is None:
            queries = STRATEGY_QUERIES.get(strategy, STRATEGY_QUERIES["literature"])
        if max_results_per_query is None:
            max_results_per_query = self.config.max_papers_per_query

        all_papers = []
        seen_ids = set()

        for query in queries:
            logger.info("Searching arXiv: '%s' (max=%d)", query, max_results_per_query)
            try:
                papers = self._search_query(query, max_results_per_query)
                for paper in papers:
                    if paper.arxiv_id in seen_ids:
                        continue
                    seen_ids.add(paper.arxiv_id)

                    if self.vector_store.is_paper_indexed(paper.arxiv_id):
                        logger.debug("Paper already indexed: %s", paper.arxiv_id)
                        all_papers.append(paper)
                        continue

                    embedding = self.embedding_service.embed_query(
                        paper.to_embedding_text()
                    )
                    self.vector_store.store_paper(paper, embedding)
                    all_papers.append(paper)
                    logger.info("Indexed new paper: %s", paper.title[:60])

            except Exception as e:
                logger.error("arXiv search failed for '%s': %s", query, e)

            time.sleep(1)

        logger.info(
            "Paper search complete: %d papers found, %d queries",
            len(all_papers),
            len(queries),
        )
        return all_papers

    def search_by_user_query(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search arXiv by a user-provided query."""
        return self.search_and_index(
            queries=[query], max_results_per_query=max_results
        )

    def get_papers_for_context(
        self, topic_embedding: list[float], limit: int = 10
    ) -> list[Paper]:
        """Get relevant papers from the index by similarity."""
        return self.vector_store.find_similar_papers(topic_embedding, limit=limit)

    def _search_query(self, query: str, max_results: int) -> list[Paper]:
        """Execute a single arXiv search query."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        for result in self.arxiv_client.results(search):
            paper = Paper(
                arxiv_id=result.entry_id.split("/abs/")[-1],
                title=result.title,
                abstract=result.summary,
                authors=[a.name for a in result.authors[:10]],
                published_date=result.published.isoformat() if result.published else "",
                categories=result.categories,
                pdf_url=result.pdf_url,
            )
            papers.append(paper)

        return papers
