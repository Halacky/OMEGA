"""Streamlit UI components for the research agent dashboard."""

import streamlit as st

from research_agent.models.hypothesis import Hypothesis


def render_hypothesis_card(
    hyp: Hypothesis, index: int, vector_store=None
) -> bool:
    """Render a single hypothesis as an expandable card.

    Returns True if the hypothesis was deleted (caller should rerun).
    """
    if hyp.source_type == "baseline":
        icon = ":blue_square:"
        status_text = "Baseline"
    elif hyp.source_type == "collected":
        icon = ":large_orange_circle:"
        status_text = "Collected"
    elif hyp.status == "verified":
        icon = ":green_circle:"
        status_text = "Verified"
    else:
        icon = ":white_circle:"
        status_text = "Unverified"

    # Build title with experiment number badge
    exp_badge = f"Exp #{hyp.experiment_id} | " if hyp.experiment_id is not None else ""
    with st.expander(f"{icon} {exp_badge}{hyp.title} [{status_text}]", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Hypothesis:**")
            st.write(hyp.hypothesis_text)
            st.markdown("**Expected Effect:**")
            st.write(hyp.expected_effect)
            st.markdown("**Novelty:**")
            st.write(hyp.novelty_explanation)

        with col2:
            st.markdown("**Proposed Changes:**")
            st.json(hyp.proposed_changes.model_dump())

            if hyp.verification_metrics:
                st.markdown("**Metrics:**")
                st.json(hyp.verification_metrics)

            st.markdown(f"**Strategy:** {hyp.strategy}")
            if hyp.experiment_id is not None:
                st.markdown(f"**Experiment #:** {hyp.experiment_id}")
            st.markdown(f"**Generated:** {hyp.generation_timestamp}")

        if hyp.motivation.based_on_experiments:
            st.markdown("**Based on Experiments:**")
            for ref in hyp.motivation.based_on_experiments:
                st.markdown(f"- **{ref.experiment_name}**: {ref.observation}")

        if hyp.motivation.based_on_papers:
            st.markdown("**Based on Papers:**")
            for ref in hyp.motivation.based_on_papers:
                st.markdown(
                    f"- **{ref.paper_title}** ({ref.arxiv_id}): {ref.insight_used}"
                )

        # --- Comments section ---
        comments = hyp.comments or []
        if comments:
            st.markdown("**Comments:**")
            for c in comments:
                author = c.get("author", "anonymous")
                text = c.get("text", "")
                ts = c.get("timestamp", "")
                st.markdown(f"> **{author}** ({ts[:16]}): {text}")

        if vector_store is not None:
            st.divider()
            col_comment, col_delete = st.columns([3, 1])

            with col_comment:
                comment_key = f"comment_text_{hyp.id}_{index}"
                comment_text = st.text_input(
                    "Add comment",
                    key=comment_key,
                    placeholder="Write a comment...",
                    label_visibility="collapsed",
                )
                if st.button("Add Comment", key=f"btn_comment_{hyp.id}_{index}"):
                    if comment_text.strip():
                        vector_store.add_comment_to_hypothesis(
                            hyp.id, "user", comment_text.strip()
                        )
                        st.success("Comment added!")
                        return True

            with col_delete:
                if hyp.source_type != "baseline":
                    if st.button(
                        "Delete",
                        key=f"btn_delete_{hyp.id}_{index}",
                        type="secondary",
                    ):
                        st.session_state[f"confirm_delete_{hyp.id}"] = True

                    if st.session_state.get(f"confirm_delete_{hyp.id}", False):
                        st.warning("Confirm deletion?")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Yes", key=f"confirm_yes_{hyp.id}_{index}"):
                                vector_store.delete_hypothesis(hyp.id)
                                st.session_state.pop(
                                    f"confirm_delete_{hyp.id}", None
                                )
                                st.success("Deleted!")
                                return True
                        with c2:
                            if st.button("No", key=f"confirm_no_{hyp.id}_{index}"):
                                st.session_state.pop(
                                    f"confirm_delete_{hyp.id}", None
                                )
                                st.rerun()

    return False


def render_experiment_heatmap(experiments: list[dict]) -> None:
    """Render a model x pipeline accuracy heatmap."""
    import pandas as pd

    rows = []
    for exp in experiments:
        pipeline = exp.get("training_config", {}).get("pipeline_type", "unknown")
        for model_name, agg in exp.get("aggregate_results", {}).items():
            if isinstance(agg, dict):
                acc = agg.get("mean_accuracy", 0)
            else:
                acc = agg.mean_accuracy
            rows.append(
                {"model": model_name, "pipeline": pipeline, "accuracy": acc}
            )

    if not rows:
        st.info("No experiment data available for heatmap.")
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        values="accuracy", index="model", columns="pipeline", aggfunc="max"
    )
    pivot = pivot.fillna(0)

    st.dataframe(
        pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=0.5).format(
            "{:.4f}"
        ),
        use_container_width=True,
    )


def render_trace_timeline(trace: list[dict]) -> None:
    """Render agent trace as a timeline."""
    if not trace:
        st.info("No trace data available. Run the agent first.")
        return

    for i, entry in enumerate(trace):
        node = entry.get("node", "unknown")
        summary = entry.get("summary", "")
        timestamp = entry.get("timestamp", "")

        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"**{i + 1}. {node}**")
        with col2:
            st.markdown(f"{summary}")
            st.caption(timestamp)
        st.divider()
