"""LangGraph agent state definition."""

from typing import Annotated, Optional

from typing_extensions import TypedDict


class TraceEntry(TypedDict):
    """A single trace entry for agent decision logging."""
    node: str
    summary: str
    timestamp: str


class AgentState(TypedDict):
    """Full state for the LangGraph research agent."""

    # Input parameters
    strategy: str
    num_hypotheses: int

    # Loaded context
    research_history: str
    experiments: list[dict]
    experiment_analysis: str
    papers: list[dict]
    paper_insights: str
    error_analysis: str
    untested_combinations: list[dict]

    # Current hypothesis generation
    candidate_hypothesis: Optional[dict]
    similar_existing: list[dict]
    similarity_score: float
    validation_errors: list[str]

    # Output
    accepted_hypotheses: list[dict]
    rejected_hypotheses: list[dict]

    # Logging
    trace: list[dict]
    iteration: int


def create_initial_state(
    strategy: str = "exploitation", num_hypotheses: int = 1
) -> AgentState:
    """Create initial agent state with defaults."""
    return AgentState(
        strategy=strategy,
        num_hypotheses=num_hypotheses,
        research_history="",
        experiments=[],
        experiment_analysis="",
        papers=[],
        paper_insights="",
        error_analysis="",
        untested_combinations=[],
        candidate_hypothesis=None,
        similar_existing=[],
        similarity_score=0.0,
        validation_errors=[],
        accepted_hypotheses=[],
        rejected_hypotheses=[],
        trace=[],
        iteration=0,
    )
