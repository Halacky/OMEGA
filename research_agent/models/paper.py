"""Pydantic models for arXiv paper data."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """An arXiv paper indexed by the paper service."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published_date: str = ""
    categories: list[str] = Field(default_factory=list)
    pdf_url: Optional[str] = None
    insights: list[str] = Field(default_factory=list)
    indexed_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_embedding_text(self) -> str:
        """Create text representation for embedding."""
        return f"{self.title}. {self.abstract}"
