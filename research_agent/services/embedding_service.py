"""Embedding service with local sentence-transformers or DeepSeek API."""

import logging
from typing import Protocol

from research_agent.config import AgentConfig

logger = logging.getLogger("research_agent.embeddings")


class EmbeddingService(Protocol):
    """Protocol for embedding services."""
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def create_embedding_service(config: AgentConfig) -> EmbeddingService:
    """Create an embedding service based on config."""
    if config.use_local_embeddings:
        logger.info(
            "Using local embeddings: %s (dim=%d)",
            config.embedding_model,
            config.embedding_dim,
        )
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        logger.info("Using DeepSeek embeddings API")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
