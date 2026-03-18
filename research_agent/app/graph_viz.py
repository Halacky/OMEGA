"""Pyvis graph builder for experiment/hypothesis visualization."""

import re
import tempfile
from pathlib import Path

from pyvis.network import Network

from research_agent.models.hypothesis import Hypothesis


# Experiment evolution relationships (parent -> children).
# Derived from the causal chain in docs/RESEARCH_HISTORY.md.
# Format: {child_experiment_prefix: [(parent_experiment_prefix, edge_label), ...]}
EXPERIMENT_RELATIONSHIPS = {
    # Phase 1-2: Baselines
    # exp_2 (td_seq features) was motivated by exp_1 (raw) performance
    "exp_2": [("exp_1", "test handcrafted features instead of raw")],
    # exp_3 (deep powerful) tried MLP on features after exp_4 showed ML works
    "exp_3": [("exp_4", "test deep learning on powerful features")],
    # exp_4 (ML powerful) was motivated by deep learning underperformance
    "exp_4": [("exp_1", "try classical ML as alternative to DL")],
    # exp_5 (hybrid GRL) combined features + domain adaptation
    "exp_5": [
        ("exp_4", "combine powerful features with domain adaptation"),
        ("exp_1", "add GRL to bridge subject gap"),
    ],
    # exp_6 (SOTA aug) tested augmentation on baselines
    "exp_6": [("exp_1", "add augmentation to baselines")],

    # Phase 3: Augmentation experiments
    # exp_7 (fusion + aug) combined raw stream + powerful features
    "exp_7": [
        ("exp_1", "CNN-GRU-Attention as raw stream"),
        ("exp_4", "powerful features as fusion input"),
    ],
    # exp_8 (augmented SVM) added augmentation to SVM baseline
    "exp_8": [("exp_4", "add signal augmentation before feature extraction")],
    # exp_9 (dual-stream) extended fusion idea with attention
    "exp_9": [("exp_7", "more principled dual-stream fusion")],
    # exp_10 (improved dual-stream) refined exp_9
    "exp_10": [("exp_9", "integrate with base trainer")],
    # exp_11 (subject-calibrated aug) adjusted augmentation per-subject
    "exp_11": [("exp_1", "calibrate augmentation per subject signal stats")],
    # exp_12 (augmented SVM 20subj) scaled up exp_8
    "exp_12": [("exp_8", "scale to 20 subjects")],

    # Phase 4: Domain adaptation & meta-learning
    # exp_13 (MAML) applied meta-learning for few-shot adaptation
    "exp_13": [("exp_1", "meta-learning for rapid subject adaptation")],
    # exp_14 (adaptive SVM) two-stage SVM fine-tuning
    "exp_14": [("exp_4", "fine-tune SVM with calibration data")],
    # exp_15 (contrastive) contrastive pre-training for invariance
    "exp_15": [("exp_1", "contrastive pre-training for subject invariance")],

    # Phase 5: Enhanced approaches
    # exp_16 (enhanced aug CNN-GRU) generic augmentation test
    "exp_16": [("exp_7", "enhanced augmentation strategy")],
    # exp_17 (triple aug) added rotation on top of noise+warp
    "exp_17": [("exp_11", "add rotation augmentation")],
    # exp_18 (feature jitter SVM) augmented in feature space
    "exp_18": [("exp_8", "augment in feature space instead of signal space")],
    # exp_19 (MMD calibration) domain adaptation via MMD
    "exp_19": [("exp_4", "MMD feature calibration for domain shift")],
    # exp_20 (fusion CNN-GRU + aug) refined fusion with augmentation
    "exp_20": [
        ("exp_7", "improve fusion with learnable layer"),
        ("exp_9", "add augmentation to fusion"),
    ],
}


def _extract_exp_prefix(experiment_name: str) -> str:
    """Extract the experiment prefix (e.g., 'exp_7', 'exp_1') from a full name.

    Handles both 'exp1_...' and 'exp_1_...' naming conventions.
    """
    m = re.match(r"(exp_?\d+)", experiment_name)
    return m.group(1) if m else ""


def _normalize_prefix(prefix: str) -> str:
    """Normalize exp prefix to 'exp_N' form for consistent matching.

    'exp1' -> 'exp_1', 'exp_12' -> 'exp_12', etc.
    """
    m = re.match(r"exp_?(\d+)", prefix)
    return f"exp_{m.group(1)}" if m else prefix


def build_knowledge_graph(
    hypotheses: list[Hypothesis],
    unverified_experiments: list[dict] | None = None,
    height: str = "700px",
    width: str = "100%",
) -> str:
    """Build a Pyvis network graph and return the HTML string.

    Node colors:
    - Blue (#4a90d9): baseline experiments (verified, source_type=baseline)
    - Amber (#FF9800): collected experiments (verified, source_type=collected)
    - Green (#4CAF50): verified generated hypotheses
    - Gray (#9E9E9E): unverified hypotheses
    - Light gray (#E0E0E0) dashed: unverified experiments (code exists, no results)

    Edges:
    - Dark gray: hypothesis -> source experiment (inspiration)
    - Blue dashed: experiment -> experiment (evolution/causal chain from research history)
    """
    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
        select_menu=False,
        filter_menu=False,
    )
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -4000,
                "centralGravity": 0.3,
                "springLength": 250,
                "springConstant": 0.03,
                "damping": 0.09
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        },
        "nodes": {
            "font": {"size": 14}
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "color": {"color": "#cccccc"},
            "smooth": {"type": "continuous"}
        }
    }
    """)

    added_nodes: set[str] = set()

    # Index: experiment_name -> list of hypothesis node IDs (baselines + collected)
    exp_to_source: dict[str, list[str]] = {}
    # Index: normalized experiment prefix -> node_id (for evolution edges)
    # We store the FIRST node_id per prefix to avoid duplicates
    prefix_to_node: dict[str, str] = {}

    for hyp in hypotheses:
        if hyp.source_type in ("baseline", "collected"):
            for ref in hyp.motivation.based_on_experiments:
                exp_to_source.setdefault(ref.experiment_name, []).append(
                    f"hyp:{hyp.id}"
                )

    # Phase 1: Add all hypothesis nodes
    for hyp in hypotheses:
        if hyp.source_type == "baseline":
            color = "#4a90d9"
            shape = "box"
            metrics = hyp.verification_metrics or {}
            acc = metrics.get("mean_accuracy", 0)
            f1 = metrics.get("mean_f1_macro", 0)
            label = (
                f"{hyp.proposed_changes.model_type}\n"
                f"{hyp.proposed_changes.features}\n"
                f"acc={acc:.3f}"
            )
            title = (
                f"<b>{hyp.title}</b><br>"
                f"Accuracy: {acc:.4f}<br>"
                f"F1-macro: {f1:.4f}<br>"
                f"Type: Baseline"
            )
        elif hyp.source_type == "collected":
            color = "#FF9800"
            shape = "box"
            metrics = hyp.verification_metrics or {}
            acc = metrics.get("mean_accuracy", 0)
            f1 = metrics.get("mean_f1_macro", 0)
            exp_num = f"Exp #{hyp.experiment_id}" if hyp.experiment_id else "Exp"
            label = (
                f"{exp_num}\n"
                f"{hyp.proposed_changes.model_type}\n"
                f"acc={acc:.3f}"
            )
            title = (
                f"<b>{exp_num}: {hyp.title}</b><br>"
                f"Accuracy: {acc:.4f}<br>"
                f"F1-macro: {f1:.4f}<br>"
                f"Model: {hyp.proposed_changes.model_type}<br>"
                f"Features: {hyp.proposed_changes.features}<br>"
                f"Augmentation: {hyp.proposed_changes.augmentation}<br>"
                f"Type: Collected Experiment"
            )
        else:
            color = "#4CAF50" if hyp.status == "verified" else "#9E9E9E"
            shape = "ellipse"
            exp_prefix = f"Exp #{hyp.experiment_id}: " if hyp.experiment_id else ""
            label = f"{exp_prefix}{hyp.title[:40]}"
            title = (
                f"<b>{exp_prefix}{hyp.title}</b><br>"
                f"Strategy: {hyp.strategy}<br>"
                f"Status: {hyp.status}<br>"
                f"Model: {hyp.proposed_changes.model_type}<br>"
                f"Features: {hyp.proposed_changes.features}<br>"
                f"Expected: {hyp.expected_effect[:100]}"
            )
            if hyp.status == "verified" and hyp.verification_metrics:
                acc = hyp.verification_metrics.get("mean_accuracy", 0)
                title += f"<br>Accuracy: {acc:.4f}"

        node_id = f"hyp:{hyp.id}"
        net.add_node(
            node_id,
            label=label,
            color=color,
            shape=shape,
            title=title,
            size=25 if hyp.source_type in ("baseline", "collected") else 20,
        )
        added_nodes.add(node_id)

        # Build prefix -> node_id index for evolution edges
        if hyp.source_type in ("baseline", "collected"):
            for ref in hyp.motivation.based_on_experiments:
                prefix = _extract_exp_prefix(ref.experiment_name)
                if prefix:
                    norm = _normalize_prefix(prefix)
                    # Keep the node with highest accuracy per prefix
                    if norm not in prefix_to_node:
                        prefix_to_node[norm] = node_id

    # Phase 2: Add hypothesis-to-source edges (generated -> baseline/collected)
    for hyp in hypotheses:
        if hyp.source_type in ("baseline", "collected"):
            continue

        node_id = f"hyp:{hyp.id}"
        for ref in hyp.motivation.based_on_experiments:
            exp_name = ref.experiment_name
            source_ids = exp_to_source.get(exp_name, [])

            if source_ids:
                for source_id in source_ids:
                    if source_id in added_nodes and source_id != node_id:
                        net.add_edge(
                            source_id,
                            node_id,
                            title=ref.observation[:80],
                        )
                        break  # one edge per experiment reference is enough
            else:
                # No matching source — create a standalone experiment node
                exp_node_id = f"exp:{exp_name}"
                if exp_node_id not in added_nodes:
                    net.add_node(
                        exp_node_id,
                        label=exp_name[:30],
                        color="#e0e0e0",
                        shape="box",
                        title=f"Experiment: {exp_name}",
                        size=15,
                    )
                    added_nodes.add(exp_node_id)
                net.add_edge(
                    exp_node_id,
                    node_id,
                    title=ref.observation[:80],
                )

    # Phase 3: Add experiment-to-experiment evolution edges (causal chain)
    # These show the research progression: which experiment inspired which
    _added_evolution_edges: set[tuple[str, str]] = set()

    for child_prefix, parents in EXPERIMENT_RELATIONSHIPS.items():
        child_norm = _normalize_prefix(child_prefix)
        child_node = prefix_to_node.get(child_norm)
        if not child_node or child_node not in added_nodes:
            continue

        for parent_prefix, edge_label in parents:
            parent_norm = _normalize_prefix(parent_prefix)
            parent_node = prefix_to_node.get(parent_norm)
            if not parent_node or parent_node not in added_nodes:
                continue
            if parent_node == child_node:
                continue

            edge_key = (parent_node, child_node)
            if edge_key in _added_evolution_edges:
                continue
            _added_evolution_edges.add(edge_key)

            net.add_edge(
                parent_node,
                child_node,
                title=edge_label,
                color="#4a90d9",
                dashes=True,
                width=1.5,
            )

    # Phase 4: Add unverified experiment nodes (code exists, no results yet)
    if unverified_experiments:
        for uexp in unverified_experiments:
            exp_id = uexp["experiment_id"]
            node_id = f"unverified:{exp_id}"
            if node_id in added_nodes:
                continue
            short_name = uexp["name"]
            if len(short_name) > 35:
                short_name = short_name[:35] + "..."
            label = f"Exp #{exp_id}\n(pending)"
            title = (
                f"<b>Exp #{exp_id}</b><br>"
                f"File: {uexp['name']}.py<br>"
                f"Status: Pending (no results yet)"
            )
            net.add_node(
                node_id,
                label=label,
                color="#E0E0E0",
                shape="box",
                title=title,
                size=15,
                borderWidth=2,
                borderWidthSelected=3,
                shapeProperties={"borderDashes": [5, 5]},
            )
            added_nodes.add(node_id)

            # Try to connect unverified experiments to their parents via evolution map
            prefix = _normalize_prefix(f"exp_{exp_id}")
            if prefix in EXPERIMENT_RELATIONSHIPS:
                for parent_prefix, edge_label in EXPERIMENT_RELATIONSHIPS[prefix]:
                    parent_norm = _normalize_prefix(parent_prefix)
                    parent_node = prefix_to_node.get(parent_norm)
                    if parent_node and parent_node in added_nodes:
                        edge_key = (parent_node, node_id)
                        if edge_key not in _added_evolution_edges:
                            _added_evolution_edges.add(edge_key)
                            net.add_edge(
                                parent_node,
                                node_id,
                                title=edge_label,
                                color="#E0E0E0",
                                dashes=True,
                                width=1,
                            )

    # Generate HTML
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False
    ) as tmp:
        net.save_graph(tmp.name)
        return Path(tmp.name).read_text()
