"""Configuration for the research agent using Pydantic Settings."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class AgentConfig(BaseSettings):
    """All configuration for the research agent."""

    # LLM provider: "anthropic" or "deepseek"
    llm_provider: str = "anthropic"

    # Anthropic (Claude) LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-6"

    # DeepSeek LLM (legacy fallback)
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    temperature: float = 0.7
    max_tokens: int = 4096

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    use_local_embeddings: bool = True

    # Qdrant
    qdrant_mode: str = "embedded"
    qdrant_path: str = ""
    qdrant_url: str = "http://localhost:6333"

    # Agent
    similarity_threshold: float = 0.85
    max_retries: int = 5
    default_strategy: str = "exploitation"
    max_papers_per_query: int = 10

    # Paths
    project_root: str = ""
    experiments_output_dir: str = "experiments_output"
    results_collected_dir: str = "results_collected"

    # Logging
    log_level: str = "INFO"
    log_dir: str = ""

    # Streamlit
    streamlit_port: int = 8501

    model_config = {
        "env_file": ".env",
        "env_prefix": "OMEGA_AGENT_",
        "extra": "ignore",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.project_root:
            self.project_root = str(
                Path(__file__).resolve().parent.parent
            )
        if not self.qdrant_path:
            self.qdrant_path = str(
                Path(self.project_root) / "research_agent" / "qdrant_data"
            )
        if not self.log_dir:
            self.log_dir = str(
                Path(self.project_root) / "research_agent" / "logs"
            )

    @property
    def research_history_path(self) -> Path:
        return Path(self.project_root) / "docs" / "RESEARCH_HISTORY.md"

    @property
    def experiments_output_path(self) -> Path:
        return Path(self.project_root) / self.experiments_output_dir

    @property
    def results_collected_path(self) -> Path:
        return Path(self.project_root) / self.results_collected_dir

    @property
    def experiments_dir_path(self) -> Path:
        return Path(self.project_root) / "experiments"

    @property
    def env_file_path(self) -> Path:
        return Path(self.project_root) / ".env"
