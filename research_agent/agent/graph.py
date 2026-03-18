"""LangGraph state machine for the research agent."""

import json
import logging
from datetime import datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from research_agent.agent.nodes import (
    NodeDependencies,
    check_done,
    decide_accept_or_reject,
    make_analyze_errors,
    make_analyze_experiments,
    make_check_similarity,
    make_extract_paper_insights,
    make_generate_hypothesis,
    make_handle_rejection,
    make_load_experiments,
    make_load_research_history,
    make_search_papers,
    make_store_hypothesis,
    make_validate_hypothesis,
    route_by_strategy,
)
from research_agent.agent.state import AgentState, create_initial_state
from research_agent.config import AgentConfig
from research_agent.knowledge.initializer import initialize_knowledge_base

logger = logging.getLogger("research_agent.graph")


def build_graph(deps: NodeDependencies) -> StateGraph:
    """Build the LangGraph research agent graph."""

    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("load_research_history", make_load_research_history(deps))
    graph.add_node("load_experiments", make_load_experiments(deps))
    graph.add_node("analyze_experiments", make_analyze_experiments(deps))
    graph.add_node("search_papers", make_search_papers(deps))
    graph.add_node("extract_paper_insights", make_extract_paper_insights(deps))
    graph.add_node("analyze_errors", make_analyze_errors(deps))
    graph.add_node("generate_hypothesis", make_generate_hypothesis(deps))
    graph.add_node("check_similarity", make_check_similarity(deps))
    graph.add_node("validate_hypothesis", make_validate_hypothesis(deps))
    graph.add_node("store_hypothesis", make_store_hypothesis(deps))
    graph.add_node("handle_rejection", make_handle_rejection(deps))

    # Entry point: first load research history, then experiments
    graph.set_entry_point("load_research_history")

    # Linear edges
    graph.add_edge("load_research_history", "load_experiments")
    graph.add_edge("load_experiments", "analyze_experiments")

    # Strategy routing
    graph.add_conditional_edges(
        "analyze_experiments",
        route_by_strategy,
        {
            "generate_hypothesis": "generate_hypothesis",
            "search_papers": "search_papers",
            "analyze_errors": "analyze_errors",
        },
    )

    # Paper branch
    graph.add_edge("search_papers", "extract_paper_insights")
    graph.add_edge("extract_paper_insights", "generate_hypothesis")

    # Error branch
    graph.add_edge("analyze_errors", "generate_hypothesis")

    # Hypothesis validation pipeline
    graph.add_edge("generate_hypothesis", "check_similarity")
    graph.add_edge("check_similarity", "validate_hypothesis")

    # Accept / reject decision
    graph.add_conditional_edges(
        "validate_hypothesis",
        decide_accept_or_reject,
        {
            "accept": "store_hypothesis",
            "reject": "handle_rejection",
        },
    )

    # Loop or finish
    graph.add_conditional_edges(
        "store_hypothesis",
        check_done,
        {"continue": "generate_hypothesis", "done": END},
    )
    graph.add_conditional_edges(
        "handle_rejection",
        check_done,
        {"continue": "generate_hypothesis", "done": END},
    )

    return graph


def run_agent(
    config: AgentConfig,
    strategy: str = "exploitation",
    num_hypotheses: int = 1,
    init_db: bool = False,
    vector_store=None,
    embedding_service=None,
    experiment_service=None,
) -> dict:
    """Run the research agent and return the final state.

    Args:
        config: Agent configuration
        strategy: Generation strategy (exploitation/exploration/literature/error)
        num_hypotheses: Number of hypotheses to generate
        init_db: Force knowledge base initialization
        vector_store: Optional pre-existing VectorStore (avoids Qdrant lock conflicts)
        embedding_service: Optional pre-existing embedding service
        experiment_service: Optional pre-existing experiment service

    Returns:
        Final agent state dict with accepted_hypotheses, rejected_hypotheses, trace
    """
    logger.info(
        "Starting agent: strategy=%s, num_hypotheses=%d", strategy, num_hypotheses
    )

    # Initialize dependencies, reusing provided services
    deps = NodeDependencies(
        config,
        vector_store=vector_store,
        embedding_service=embedding_service,
        experiment_service=experiment_service,
    )

    # Initialize knowledge base if needed
    if init_db or deps.vector_store.is_hypotheses_empty():
        logger.info("Initializing knowledge base...")
        count = initialize_knowledge_base(
            config,
            deps.vector_store,
            deps.embedding_service,
            deps.experiment_service,
        )
        logger.info("Knowledge base initialized with %d nodes", count)

    # Build and compile graph
    graph = build_graph(deps)
    compiled = graph.compile()

    # Create initial state
    initial_state = create_initial_state(strategy, num_hypotheses)
    initial_state["_max_retries"] = config.max_retries

    # Run the graph
    logger.info("Executing agent graph...")
    final_state = compiled.invoke(initial_state)

    # Save trace
    _save_trace(config, final_state)

    accepted = final_state.get("accepted_hypotheses", [])
    rejected = final_state.get("rejected_hypotheses", [])
    logger.info(
        "Agent complete: %d accepted, %d rejected hypotheses",
        len(accepted),
        len(rejected),
    )

    return final_state


def _save_trace(config: AgentConfig, state: dict) -> None:
    """Save the agent trace to a JSON file."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = log_dir / f"trace_{timestamp}.json"

    trace_data = {
        "strategy": state.get("strategy", ""),
        "num_hypotheses_requested": state.get("num_hypotheses", 0),
        "num_accepted": len(state.get("accepted_hypotheses", [])),
        "num_rejected": len(state.get("rejected_hypotheses", [])),
        "iterations": state.get("iteration", 0),
        "trace": state.get("trace", []),
        "accepted_hypotheses": state.get("accepted_hypotheses", []),
        "rejected_hypotheses": state.get("rejected_hypotheses", []),
        "timestamp": timestamp,
    }

    with open(trace_file, "w") as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)

    logger.info("Trace saved to %s", trace_file)
