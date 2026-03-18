"""LangGraph node functions for the research agent."""

import json
import logging
from datetime import datetime
from pathlib import Path

from research_agent.agent.prompts import (
    ANALYZE_ERRORS_PROMPT,
    ANALYZE_EXPERIMENTS_PROMPT,
    EXTRACT_PAPER_INSIGHTS_PROMPT,
    GENERATE_SEARCH_QUERIES_PROMPT,
)
from research_agent.agent.state import AgentState
from research_agent.agent.strategies import build_generation_prompt
from research_agent.config import AgentConfig
from research_agent.knowledge.codebase_registry import (
    get_constraints_text,
    validate_proposed_changes,
)
from research_agent.models.hypothesis import Hypothesis
from research_agent.services.embedding_service import EmbeddingService, create_embedding_service
from research_agent.services.experiment_service import ExperimentService
from research_agent.services.llm_service import create_llm
from research_agent.services.paper_service import PaperService
from research_agent.services.vector_db import VectorStore

logger = logging.getLogger("research_agent.nodes")


class NodeDependencies:
    """Container for shared dependencies injected into node functions."""

    def __init__(
        self,
        config: AgentConfig,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        experiment_service: ExperimentService | None = None,
    ):
        self.config = config
        self.llm = create_llm(config)
        self.embedding_service = embedding_service or create_embedding_service(config)
        self.vector_store = vector_store or VectorStore(config)
        self.experiment_service = experiment_service or ExperimentService(config)
        self.paper_service = PaperService(
            config, self.vector_store, self.embedding_service
        )
        self.vector_store.init_collections(config.embedding_dim)


def _add_trace(state: AgentState, node: str, summary: str) -> None:
    """Append a trace entry to the state."""
    state["trace"].append(
        {"node": node, "summary": summary, "timestamp": datetime.now().isoformat()}
    )


def make_load_research_history(deps: NodeDependencies):
    """Create the load_research_history node.

    Reads the RESEARCH_HISTORY.md file and loads it into state
    so all subsequent nodes have access to the full research context.
    """

    def load_research_history(state: AgentState) -> dict:
        history_path = deps.config.research_history_path
        logger.info("Loading research history from %s", history_path)

        research_history = ""
        if history_path.exists():
            try:
                research_history = history_path.read_text(encoding="utf-8")
                logger.info(
                    "Research history loaded: %d chars from %s",
                    len(research_history),
                    history_path.name,
                )
            except Exception as e:
                logger.error("Failed to read research history: %s", e)
                research_history = ""
        else:
            logger.warning("Research history file not found: %s", history_path)

        _add_trace(
            state,
            "load_research_history",
            f"Loaded research history ({len(research_history)} chars)"
            if research_history
            else "No research history file found",
        )
        return {"research_history": research_history, "trace": state["trace"]}

    return load_research_history


def make_load_experiments(deps: NodeDependencies):
    """Create the load_experiments node."""

    def load_experiments(state: AgentState) -> dict:
        logger.info("Loading experiments from %s", deps.config.experiments_output_path)
        experiments = deps.experiment_service.load_all_experiments()
        exp_dicts = [e.model_dump() for e in experiments]
        untested = deps.experiment_service.get_untested_combinations()

        _add_trace(state, "load_experiments", f"Loaded {len(experiments)} experiments")
        return {
            "experiments": exp_dicts,
            "untested_combinations": untested,
            "trace": state["trace"],
        }

    return load_experiments


def make_analyze_experiments(deps: NodeDependencies):
    """Create the analyze_experiments node."""

    def analyze_experiments(state: AgentState) -> dict:
        logger.info("Analyzing experiments with LLM")
        experiments_text = deps.experiment_service.get_all_experiments_summary_text()
        research_history = state.get("research_history", "")

        # Truncate research history for analysis prompt if needed
        max_history = 20000
        history_for_prompt = research_history
        if len(history_for_prompt) > max_history:
            history_for_prompt = history_for_prompt[:max_history] + "\n\n[... truncated ...]"

        prompt = ANALYZE_EXPERIMENTS_PROMPT.format(
            research_history=history_for_prompt,
            experiments_text=experiments_text,
        )
        response = deps.llm.invoke(prompt)
        analysis = response.content

        _add_trace(
            state,
            "analyze_experiments",
            f"Generated analysis ({len(analysis)} chars)",
        )
        logger.info("Experiment analysis complete (%d chars)", len(analysis))
        return {"experiment_analysis": analysis, "trace": state["trace"]}

    return analyze_experiments


def make_search_papers(deps: NodeDependencies):
    """Create the search_papers node.

    Uses LLM to generate targeted search queries based on research history
    and current analysis, then searches arXiv.
    """

    def search_papers(state: AgentState) -> dict:
        strategy = state["strategy"]
        logger.info("Generating search queries for strategy: %s", strategy)

        # Generate targeted search queries using LLM
        queries = _generate_search_queries(deps, state)

        logger.info("Searching papers with %d LLM-generated queries", len(queries))
        papers = deps.paper_service.search_and_index(
            queries=queries, strategy=strategy
        )
        paper_dicts = [p.model_dump() for p in papers]

        _add_trace(
            state,
            "search_papers",
            f"Generated {len(queries)} queries, found {len(papers)} papers (strategy={strategy})",
        )
        return {"papers": paper_dicts, "trace": state["trace"]}

    return search_papers


def _generate_search_queries(deps: NodeDependencies, state: AgentState) -> list[str]:
    """Use LLM to generate targeted arXiv search queries based on research context."""
    from research_agent.services.paper_service import STRATEGY_QUERIES

    research_history = state.get("research_history", "")
    experiment_analysis = state.get("experiment_analysis", "")

    # If no research history available, fall back to hardcoded queries
    if not research_history and not experiment_analysis:
        strategy = state["strategy"]
        return STRATEGY_QUERIES.get(strategy, STRATEGY_QUERIES["literature"])

    # Truncate for prompt
    max_history = 15000
    if len(research_history) > max_history:
        research_history = research_history[:max_history] + "\n\n[... truncated ...]"

    max_analysis = 5000
    if len(experiment_analysis) > max_analysis:
        experiment_analysis = experiment_analysis[:max_analysis] + "\n\n[... truncated ...]"

    prompt = GENERATE_SEARCH_QUERIES_PROMPT.format(
        research_history=research_history,
        experiment_analysis=experiment_analysis,
        strategy=state["strategy"],
    )

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(
            content="You are a research assistant. Return ONLY a valid JSON array of strings. "
            "No explanation, no markdown, just the JSON array."
        ),
        HumanMessage(content=prompt),
    ]

    try:
        response = deps.llm.invoke(messages)
        raw_text = response.content.strip()

        # Parse JSON array from response
        json_text = raw_text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        queries = json.loads(json_text)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            logger.info("LLM generated %d search queries: %s", len(queries), queries)
            return queries[:8]  # Cap at 8 queries
        else:
            logger.warning("LLM returned invalid query format, falling back to defaults")

    except Exception as e:
        logger.error("Failed to generate search queries via LLM: %s", e)

    # Fallback to hardcoded queries
    strategy = state["strategy"]
    return STRATEGY_QUERIES.get(strategy, STRATEGY_QUERIES["literature"])


def make_extract_paper_insights(deps: NodeDependencies):
    """Create the extract_paper_insights node."""

    def extract_paper_insights(state: AgentState) -> dict:
        papers = state.get("papers", [])
        if not papers:
            _add_trace(state, "extract_paper_insights", "No papers to analyze")
            return {"paper_insights": "No papers available.", "trace": state["trace"]}

        logger.info("Extracting insights from %d papers", len(papers))
        papers_text = "\n\n".join(
            f"Title: {p.get('title', '')}\n"
            f"arXiv ID: {p.get('arxiv_id', '')}\n"
            f"Abstract: {p.get('abstract', '')[:500]}"
            for p in papers[:15]
        )

        best_results = deps.experiment_service.get_best_experiments(top_k=5)
        best_text = "\n".join(
            f"- {name}/{model}: acc={acc:.4f}" for name, model, acc in best_results
        )

        research_history = state.get("research_history", "")
        max_history = 15000
        if len(research_history) > max_history:
            research_history = research_history[:max_history] + "\n\n[... truncated ...]"

        prompt = EXTRACT_PAPER_INSIGHTS_PROMPT.format(
            research_history=research_history,
            papers_text=papers_text,
            best_results=best_text,
            constraints=get_constraints_text(),
        )
        response = deps.llm.invoke(prompt)
        insights = response.content

        _add_trace(
            state,
            "extract_paper_insights",
            f"Extracted insights ({len(insights)} chars) from {len(papers)} papers",
        )
        return {"paper_insights": insights, "trace": state["trace"]}

    return extract_paper_insights


def make_analyze_errors(deps: NodeDependencies):
    """Create the analyze_errors node."""

    def analyze_errors(state: AgentState) -> dict:
        logger.info("Analyzing error patterns")
        worst = deps.experiment_service.get_worst_subjects(top_k=15)
        worst_text = "\n".join(
            f"- {exp}/{model} on {subj}: acc={acc:.4f}"
            for exp, model, subj, acc in worst
        )

        research_history = state.get("research_history", "")
        max_history = 15000
        if len(research_history) > max_history:
            research_history = research_history[:max_history] + "\n\n[... truncated ...]"

        prompt = ANALYZE_ERRORS_PROMPT.format(
            research_history=research_history,
            worst_subjects=worst_text,
            experiment_analysis=state.get("experiment_analysis", ""),
            constraints=get_constraints_text(),
        )
        response = deps.llm.invoke(prompt)
        analysis = response.content

        _add_trace(
            state, "analyze_errors", f"Error analysis complete ({len(analysis)} chars)"
        )
        return {"error_analysis": analysis, "trace": state["trace"]}

    return analyze_errors


def _normalize_hypothesis_data(data: dict) -> dict:
    """Normalize LLM output to match Hypothesis schema.

    Handles cases where the LLM returns strings instead of objects
    in motivation arrays.
    """
    motivation = data.get("motivation", {})
    if isinstance(motivation, str):
        motivation = {"based_on_experiments": [], "based_on_papers": []}
        data["motivation"] = motivation

    # Normalize based_on_experiments
    experiments = motivation.get("based_on_experiments", [])
    normalized_exps = []
    for item in experiments:
        if isinstance(item, str):
            normalized_exps.append(
                {"experiment_name": item, "observation": "Referenced by LLM"}
            )
        elif isinstance(item, dict):
            item.setdefault("experiment_name", "unknown")
            item.setdefault("observation", "")
            normalized_exps.append(item)
    motivation["based_on_experiments"] = normalized_exps

    # Normalize based_on_papers
    papers = motivation.get("based_on_papers", [])
    normalized_papers = []
    for item in papers:
        if isinstance(item, str):
            normalized_papers.append(
                {"paper_title": item, "arxiv_id": "N/A", "insight_used": item}
            )
        elif isinstance(item, dict):
            item.setdefault("paper_title", "unknown")
            item.setdefault("arxiv_id", "N/A")
            item.setdefault("insight_used", "")
            normalized_papers.append(item)
    motivation["based_on_papers"] = normalized_papers

    # Normalize proposed_changes
    changes = data.get("proposed_changes", {})
    if isinstance(changes, str):
        data["proposed_changes"] = {
            "model_type": "unknown",
            "features": "unknown",
            "augmentation": "none",
            "training_modifications": changes,
        }
    else:
        changes.setdefault("model_type", "unknown")
        changes.setdefault("features", "unknown")
        changes.setdefault("augmentation", "none")
        changes.setdefault("training_modifications", "")

    # Ensure top-level fields
    data.setdefault("title", "Untitled hypothesis")
    data.setdefault("hypothesis_text", "")
    data.setdefault("expected_effect", "")
    data.setdefault("novelty_explanation", "")

    return data


def make_generate_hypothesis(deps: NodeDependencies):
    """Create the generate_hypothesis node."""

    def generate_hypothesis(state: AgentState) -> dict:
        iteration = state.get("iteration", 0) + 1
        strategy = state["strategy"]
        logger.info(
            "Generating hypothesis (strategy=%s, iteration=%d)", strategy, iteration
        )

        prompt = build_generation_prompt(state)

        system_msg = (
            "You are a research hypothesis generator for EMG gesture recognition. "
            "You have access to the full research history and must use it to generate "
            "novel, non-redundant hypotheses. "
            "You MUST respond with valid JSON matching the exact schema below. "
            "Do NOT include any text outside the JSON object.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "title": "string",\n'
            '  "hypothesis_text": "string",\n'
            '  "motivation": {\n'
            '    "based_on_experiments": [\n'
            '      {"experiment_name": "string", "observation": "string"}\n'
            "    ],\n"
            '    "based_on_papers": [\n'
            '      {"paper_title": "string", "arxiv_id": "string", "insight_used": "string"}\n'
            "    ]\n"
            "  },\n"
            '  "proposed_changes": {\n'
            '    "model_type": "string",\n'
            '    "features": "string",\n'
            '    "augmentation": "string",\n'
            '    "training_modifications": "string"\n'
            "  },\n"
            '  "expected_effect": "string",\n'
            '  "novelty_explanation": "string"\n'
            "}\n\n"
            "IMPORTANT: based_on_experiments MUST be an array of objects with "
            '"experiment_name" and "observation" keys. '
            "based_on_papers MUST be an array of objects with "
            '"paper_title", "arxiv_id", and "insight_used" keys. '
            "Do NOT use plain strings in these arrays."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system_msg), HumanMessage(content=prompt)]
        response = deps.llm.invoke(messages)
        raw_text = response.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        json_text = raw_text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        try:
            hypothesis_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse hypothesis JSON: %s\nRaw: %s", e, raw_text[:500])
            hypothesis_data = {
                "title": "Parse error",
                "hypothesis_text": raw_text[:200],
                "motivation": {"based_on_experiments": [], "based_on_papers": []},
                "proposed_changes": {
                    "model_type": "unknown",
                    "features": "unknown",
                    "augmentation": "none",
                    "training_modifications": "",
                },
                "expected_effect": "Parse error",
                "novelty_explanation": "Parse error",
            }

        # Normalize motivation structure (LLM sometimes returns strings instead of objects)
        hypothesis_data = _normalize_hypothesis_data(hypothesis_data)

        # Ensure required fields
        hypothesis_data.setdefault("strategy", strategy)
        hyp = Hypothesis(**hypothesis_data)
        candidate = hyp.model_dump()

        _add_trace(
            state,
            "generate_hypothesis",
            f"Generated: '{hyp.title}' (strategy={strategy}, iter={iteration})",
        )
        return {
            "candidate_hypothesis": candidate,
            "iteration": iteration,
            "trace": state["trace"],
        }

    return generate_hypothesis


def make_check_similarity(deps: NodeDependencies):
    """Create the check_similarity node."""

    def check_similarity(state: AgentState) -> dict:
        candidate = state.get("candidate_hypothesis")
        if not candidate:
            return {
                "similarity_score": 0.0,
                "similar_existing": [],
                "trace": state["trace"],
            }

        hyp = Hypothesis(**candidate)
        embedding_text = hyp.to_embedding_text()
        embedding = deps.embedding_service.embed_query(embedding_text)

        max_sim = deps.vector_store.get_max_similarity(embedding)
        similar = deps.vector_store.find_similar_hypotheses(
            embedding, threshold=0.5, limit=3
        )
        similar_dicts = [
            {"title": h.title, "score": s, "id": h.id} for h, s in similar
        ]

        _add_trace(
            state,
            "check_similarity",
            f"Max similarity: {max_sim:.3f} (threshold={deps.config.similarity_threshold})",
        )
        logger.info(
            "Similarity check: max=%.3f, threshold=%.3f",
            max_sim,
            deps.config.similarity_threshold,
        )
        return {
            "similarity_score": max_sim,
            "similar_existing": similar_dicts,
            "trace": state["trace"],
        }

    return check_similarity


def make_validate_hypothesis(deps: NodeDependencies):
    """Create the validate_hypothesis node."""

    def validate_hypothesis(state: AgentState) -> dict:
        candidate = state.get("candidate_hypothesis")
        if not candidate:
            return {"validation_errors": ["No candidate"], "trace": state["trace"]}

        changes = candidate.get("proposed_changes", {})
        errors = validate_proposed_changes(changes)

        if errors:
            candidate["rejection_reason"] = f"Validation errors: {errors}"
            logger.warning("Validation failed: %s", errors)
        elif state["similarity_score"] > deps.config.similarity_threshold:
            candidate["rejection_reason"] = (
                f"Too similar to existing hypothesis "
                f"(score={state['similarity_score']:.3f} > "
                f"threshold={deps.config.similarity_threshold})"
            )
            logger.warning(
                "Rejected: too similar (%.3f)", state["similarity_score"]
            )

        _add_trace(
            state,
            "validate_hypothesis",
            f"Errors: {errors}, Similarity: {state['similarity_score']:.3f}",
        )
        return {
            "validation_errors": errors,
            "candidate_hypothesis": candidate,
            "trace": state["trace"],
        }

    return validate_hypothesis


def make_store_hypothesis(deps: NodeDependencies):
    """Create the store_hypothesis node."""

    def store_hypothesis(state: AgentState) -> dict:
        candidate = state["candidate_hypothesis"]
        hyp = Hypothesis(**candidate)
        embedding = deps.embedding_service.embed_query(hyp.to_embedding_text())
        deps.vector_store.store_hypothesis(hyp, embedding)

        accepted = list(state.get("accepted_hypotheses", []))
        accepted.append(candidate)

        _add_trace(
            state,
            "store_hypothesis",
            f"Stored: '{hyp.title}' (id={hyp.id})",
        )
        logger.info("Hypothesis accepted and stored: %s", hyp.title)
        return {
            "accepted_hypotheses": accepted,
            "candidate_hypothesis": None,
            "trace": state["trace"],
        }

    return store_hypothesis


def make_handle_rejection(deps: NodeDependencies):
    """Create the handle_rejection node."""

    def handle_rejection(state: AgentState) -> dict:
        candidate = state.get("candidate_hypothesis", {})
        reason = candidate.get("rejection_reason", "Unknown")
        title = candidate.get("title", "Unknown")

        rejected = list(state.get("rejected_hypotheses", []))
        rejected.append(candidate)

        _add_trace(
            state,
            "handle_rejection",
            f"Rejected: '{title}' — {reason}",
        )
        logger.info("Hypothesis rejected: %s — %s", title, reason)
        return {
            "rejected_hypotheses": rejected,
            "candidate_hypothesis": None,
            "trace": state["trace"],
        }

    return handle_rejection


# ---- Routing functions ----


def route_by_strategy(state: AgentState) -> str:
    """Route to the appropriate branch based on strategy."""
    strategy = state["strategy"]
    if strategy == "exploitation":
        return "generate_hypothesis"
    elif strategy in ("exploration", "literature"):
        return "search_papers"
    elif strategy == "error":
        return "analyze_errors"
    return "generate_hypothesis"


def decide_accept_or_reject(state: AgentState) -> str:
    """Decide whether to accept or reject the candidate hypothesis."""
    candidate = state.get("candidate_hypothesis", {})
    if candidate.get("rejection_reason"):
        return "reject"
    return "accept"


def check_done(state: AgentState) -> str:
    """Check if we've generated enough hypotheses or exhausted retries."""
    accepted = len(state.get("accepted_hypotheses", []))
    target = state.get("num_hypotheses", 1)
    iteration = state.get("iteration", 0)
    max_iter = target + state.get("_max_retries", 5)

    if accepted >= target:
        return "done"
    if iteration >= max_iter:
        logger.warning(
            "Max iterations reached (%d), stopping with %d/%d hypotheses",
            iteration,
            accepted,
            target,
        )
        return "done"
    return "continue"
