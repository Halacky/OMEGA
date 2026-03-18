"""Data models for the research agent."""

from .hypothesis import Hypothesis, Motivation, ProposedChanges, ExperimentReference, PaperReference
from .experiment import ExperimentSummary, AggregateResult, SubjectResult
from .paper import Paper

__all__ = [
    "Hypothesis", "Motivation", "ProposedChanges",
    "ExperimentReference", "PaperReference",
    "ExperimentSummary", "AggregateResult", "SubjectResult",
    "Paper",
]
