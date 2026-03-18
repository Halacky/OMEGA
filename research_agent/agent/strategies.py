"""Strategy implementations for hypothesis generation."""

from research_agent.agent.prompts import (
    ERROR_DRIVEN_PROMPT,
    EXPLOITATION_PROMPT,
    EXPLORATION_PROMPT,
    LITERATURE_PROMPT,
)
from research_agent.agent.state import AgentState
from research_agent.knowledge.codebase_registry import get_constraints_text


def build_generation_prompt(state: AgentState) -> str:
    """Build the full generation prompt based on the selected strategy."""
    strategy = state["strategy"]
    constraints = get_constraints_text()
    research_history = state.get("research_history", "")

    # Truncate research history if too long to fit in context
    max_history_chars = 30000
    if len(research_history) > max_history_chars:
        research_history = research_history[:max_history_chars] + "\n\n[... truncated ...]"

    rejected_text = ""
    if state["rejected_hypotheses"]:
        rejected_items = []
        for r in state["rejected_hypotheses"]:
            title = r.get("title", "unknown")
            reason = r.get("rejection_reason", "unknown reason")
            rejected_items.append(f"- {title}: {reason}")
        rejected_text = "\n".join(rejected_items)
    else:
        rejected_text = "None yet."

    if strategy == "exploitation":
        best_results = _format_best_results(state)
        return EXPLOITATION_PROMPT.format(
            research_history=research_history,
            best_results=best_results,
            experiment_analysis=state["experiment_analysis"],
            rejected=rejected_text,
            constraints=constraints,
        )

    elif strategy == "exploration":
        untested = _format_untested(state)
        return EXPLORATION_PROMPT.format(
            research_history=research_history,
            untested_combinations=untested,
            experiment_analysis=state["experiment_analysis"],
            paper_insights=state.get("paper_insights", "Not available."),
            rejected=rejected_text,
            constraints=constraints,
        )

    elif strategy == "literature":
        return LITERATURE_PROMPT.format(
            research_history=research_history,
            paper_insights=state.get("paper_insights", "No papers found."),
            experiment_analysis=state["experiment_analysis"],
            rejected=rejected_text,
            constraints=constraints,
        )

    elif strategy == "error":
        return ERROR_DRIVEN_PROMPT.format(
            research_history=research_history,
            error_analysis=state.get("error_analysis", "Not available."),
            experiment_analysis=state["experiment_analysis"],
            rejected=rejected_text,
            constraints=constraints,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def _format_best_results(state: AgentState) -> str:
    """Format the best experiment results for the prompt."""
    experiments = state["experiments"]
    ranked = []
    for exp in experiments:
        for model_name, agg in exp.get("aggregate_results", {}).items():
            if isinstance(agg, dict):
                acc = agg.get("mean_accuracy", 0)
            else:
                acc = agg.mean_accuracy if hasattr(agg, "mean_accuracy") else 0
            ranked.append((exp.get("experiment_name", ""), model_name, acc))
    ranked.sort(key=lambda x: x[2], reverse=True)

    lines = []
    for exp_name, model, acc in ranked[:10]:
        lines.append(f"- {exp_name} / {model}: accuracy={acc:.4f}")
    return "\n".join(lines) if lines else "No results available."


def _format_untested(state: AgentState) -> str:
    """Format untested combinations for the prompt."""
    combinations = state.get("untested_combinations", [])
    if not combinations:
        return "All basic combinations have been tested."
    lines = []
    for combo in combinations[:20]:
        lines.append(f"- {combo['model_type']} + {combo['pipeline']}")
    return "\n".join(lines)
