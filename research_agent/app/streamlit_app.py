"""Streamlit dashboard for the OMEGA Research Agent."""

import json
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_agent.app.components import (
    render_experiment_heatmap,
    render_hypothesis_card,
    render_trace_timeline,
)
from research_agent.app.graph_viz import build_knowledge_graph
from research_agent.config import AgentConfig
from research_agent.models.hypothesis import Hypothesis
from research_agent.services.experiment_service import ExperimentService
from research_agent.services.vector_db import VectorStore

st.set_page_config(
    page_title="OMEGA Research Agent",
    page_icon=":microscope:",
    layout="wide",
)


from research_agent.services.embedding_service import create_embedding_service


@st.cache_resource
def get_config():
    return AgentConfig()


@st.cache_resource
def get_services(_config):
    """Create all shared services once to avoid Qdrant lock conflicts."""
    embedding_service = create_embedding_service(_config)
    vs = VectorStore(_config)
    vs.init_collections(_config.embedding_dim)
    exp_service = ExperimentService(_config)
    return vs, embedding_service, exp_service


def main():
    st.title("OMEGA Research Agent Dashboard")

    config = get_config()
    vector_store, embedding_service, experiment_service = get_services(config)

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Agent Controls")

        strategy = st.selectbox(
            "Strategy",
            ["exploitation", "exploration", "literature", "error"],
            index=0,
        )
        num_hypotheses = st.number_input(
            "Number of hypotheses", min_value=1, max_value=10, value=1
        )

        if st.button("Generate Hypotheses", type="primary"):
            with st.spinner("Running agent..."):
                try:
                    from research_agent.agent.graph import run_agent

                    result = run_agent(
                        config,
                        strategy=strategy,
                        num_hypotheses=num_hypotheses,
                        vector_store=vector_store,
                        embedding_service=embedding_service,
                        experiment_service=experiment_service,
                    )
                    st.session_state["last_trace"] = result.get("trace", [])
                    st.session_state["last_accepted"] = result.get(
                        "accepted_hypotheses", []
                    )
                    st.session_state["last_rejected"] = result.get(
                        "rejected_hypotheses", []
                    )
                    accepted_count = len(result.get("accepted_hypotheses", []))
                    st.success(f"Generated {accepted_count} hypothesis(es)!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent error: {e}")

        st.divider()
        st.header("Results Sync")
        results_dir_input = st.text_input(
            "Results directory",
            value=str(config.results_collected_path),
        )
        if st.button("Sync Results"):
            with st.spinner("Syncing collected results..."):
                try:
                    from research_agent.services.results_sync import (
                        sync_collected_results,
                    )

                    results_path = Path(results_dir_input)
                    stats = sync_collected_results(
                        config,
                        vector_store,
                        embedding_service,
                        experiment_service,
                        results_dir=results_path,
                    )
                    history_stats = stats.get("history_updated", {})
                    hyp_verified = stats.get("hypotheses_verified", 0)
                    pending = stats.get("pending_processed", 0)
                    st.success(
                        f"Sync complete: {stats['created']} Qdrant nodes created, "
                        f"{stats['skipped']} skipped, "
                        f"{hyp_verified} hypotheses verified, "
                        f"{pending} pending updates processed. "
                        f"History: {history_stats.get('leaderboard_added', 0)} leaderboard rows, "
                        f"{history_stats.get('registry_added', 0)} registry entries added."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Sync error: {e}")

        st.divider()
        st.header("Filters")
        status_filter = st.selectbox(
            "Status",
            ["all", "verified", "unverified", "baseline", "collected"],
        )
        strategy_filter = st.selectbox(
            "Strategy Filter",
            [
                "all",
                "exploitation",
                "exploration",
                "literature",
                "error",
                "baseline",
                "collected",
            ],
        )

        st.divider()
        st.header("Database Info")
        try:
            all_hyps = vector_store.get_all_hypotheses()
            baselines = [h for h in all_hyps if h.source_type == "baseline"]
            collected = [h for h in all_hyps if h.source_type == "collected"]
            generated = [h for h in all_hyps if h.source_type == "generated"]
            verified = [h for h in generated if h.status == "verified"]
            st.metric("Baseline Nodes", len(baselines))
            st.metric("Collected Experiments", len(collected))
            st.metric("Generated Hypotheses", len(generated))
            st.metric("Verified", len(verified))
        except Exception:
            all_hyps = []
            st.warning("Cannot connect to Qdrant")

    # ---- Main Content ----
    tab_graph, tab_list, tab_experiments, tab_trace = st.tabs(
        ["Knowledge Graph", "Hypothesis List", "Experiment Overview", "Agent Trace"]
    )

    # Apply filters
    filtered_hyps = all_hyps
    if status_filter == "baseline":
        filtered_hyps = [h for h in all_hyps if h.source_type == "baseline"]
    elif status_filter == "collected":
        filtered_hyps = [h for h in all_hyps if h.source_type == "collected"]
    elif status_filter != "all":
        filtered_hyps = [
            h
            for h in all_hyps
            if h.status == status_filter
            and h.source_type not in ("baseline", "collected")
        ]
    if strategy_filter != "all":
        filtered_hyps = [h for h in filtered_hyps if h.strategy == strategy_filter]

    # Collect unverified experiments for graph
    unverified_experiments = []
    try:
        from research_agent.services.results_sync import scan_unverified_experiments

        verified_ids = {
            h.experiment_id for h in all_hyps if h.experiment_id is not None
        }
        # Also include baseline experiment IDs (1-6)
        for h in all_hyps:
            if h.source_type == "baseline":
                from research_agent.services.experiment_service import (
                    ExperimentService,
                )

                eid = ExperimentService.extract_experiment_id(
                    h.motivation.based_on_experiments[0].experiment_name
                    if h.motivation.based_on_experiments
                    else ""
                )
                if eid is not None:
                    verified_ids.add(eid)
        unverified_experiments = scan_unverified_experiments(
            config.experiments_dir_path, verified_ids
        )
    except Exception:
        pass

    # Tab 1: Knowledge Graph
    with tab_graph:
        st.subheader("Knowledge Graph")
        if all_hyps or unverified_experiments:
            graph_html = build_knowledge_graph(
                filtered_hyps,
                unverified_experiments=unverified_experiments,
            )
            st.components.v1.html(graph_html, height=750, scrolling=True)
        else:
            st.info(
                "No data in the knowledge base. "
                "Click 'Sync Results' to load collected experiments, "
                "or run the agent with --init-db to populate baseline experiments."
            )

    # Tab 2: Hypothesis List
    with tab_list:
        st.subheader(f"Hypotheses ({len(filtered_hyps)})")

        sort_by = st.selectbox(
            "Sort by", ["date (newest)", "date (oldest)", "strategy", "experiment #"]
        )

        sorted_hyps = list(filtered_hyps)
        if sort_by == "date (newest)":
            sorted_hyps.sort(key=lambda h: h.generation_timestamp, reverse=True)
        elif sort_by == "date (oldest)":
            sorted_hyps.sort(key=lambda h: h.generation_timestamp)
        elif sort_by == "strategy":
            sorted_hyps.sort(key=lambda h: h.strategy)
        elif sort_by == "experiment #":
            sorted_hyps.sort(
                key=lambda h: (h.experiment_id if h.experiment_id is not None else 9999)
            )

        for i, hyp in enumerate(sorted_hyps):
            needs_rerun = render_hypothesis_card(hyp, i, vector_store=vector_store)
            if needs_rerun:
                st.rerun()

    # Tab 3: Experiment Overview
    with tab_experiments:
        st.subheader("Experiment Results Overview")

        # Load from both experiments_output and results_collected
        experiments = experiment_service.load_all_experiments()
        collected_experiments = experiment_service.load_collected_experiments()

        # Merge, avoiding duplicates by experiment_name
        seen_names = {e.experiment_name for e in experiments}
        for ce in collected_experiments:
            if ce.experiment_name not in seen_names:
                experiments.append(ce)
                seen_names.add(ce.experiment_name)

        if experiments:
            st.markdown("### Model x Pipeline Accuracy Heatmap")
            render_experiment_heatmap([e.model_dump() for e in experiments])

            # Collected experiments summary
            if collected_experiments:
                st.markdown("### Collected Experiment Results")
                for exp in collected_experiments:
                    exp_label = (
                        f"**Exp #{exp.experiment_id}**"
                        if exp.experiment_id
                        else f"**{exp.experiment_name}**"
                    )
                    for model_name, agg in exp.aggregate_results.items():
                        acc_str = f"{agg.mean_accuracy:.4f}" if agg.mean_accuracy is not None else "N/A"
                        f1_str = f"{agg.mean_f1_macro:.4f}" if agg.mean_f1_macro is not None else "N/A"
                        st.markdown(
                            f"- {exp_label} | {model_name}: "
                            f"acc={acc_str} (±{agg.std_accuracy:.4f}), "
                            f"f1={f1_str} (±{agg.std_f1_macro:.4f}), "
                            f"subjects={agg.num_subjects}"
                        )

            st.markdown("### Top 10 Results")
            best = experiment_service.get_best_experiments(top_k=10)
            for rank, (exp, model, acc) in enumerate(best, 1):
                st.markdown(f"{rank}. **{model}** ({exp}): accuracy={acc:.4f}")

            st.markdown("### Worst 10 Subject Results")
            worst = experiment_service.get_worst_subjects(top_k=10)
            for exp, model, subj, acc in worst:
                st.markdown(f"- **{subj}** / {model} ({exp}): accuracy={acc:.4f}")
        else:
            st.info("No experiments found. Click 'Sync Results' to load collected experiments.")

    # Tab 4: Agent Trace
    with tab_trace:
        st.subheader("Last Agent Run Trace")

        # Try session state first, then load from log files
        trace = st.session_state.get("last_trace", [])
        if not trace:
            log_dir = Path(config.log_dir)
            if log_dir.exists():
                trace_files = sorted(log_dir.glob("trace_*.json"), reverse=True)
                if trace_files:
                    with open(trace_files[0]) as f:
                        trace_data = json.load(f)
                    trace = trace_data.get("trace", [])
                    st.caption(f"Loaded from: {trace_files[0].name}")

        render_trace_timeline(trace)

        # Show accepted/rejected from last run
        last_accepted = st.session_state.get("last_accepted", [])
        last_rejected = st.session_state.get("last_rejected", [])

        if last_accepted:
            st.markdown("### Accepted Hypotheses")
            for h in last_accepted:
                st.json(h)

        if last_rejected:
            st.markdown("### Rejected Hypotheses")
            for h in last_rejected:
                st.json(h)


if __name__ == "__main__":
    main()
