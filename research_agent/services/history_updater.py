"""Service for auto-updating docs/RESEARCH_HISTORY.md when results are synced.

Template-based (no LLM needed): generates leaderboard rows, registry entries,
and hypothesis tracker rows from ExperimentSummary data, then inserts them
into the existing markdown structure.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from research_agent.models.experiment import AggregateResult, ExperimentSummary

logger = logging.getLogger("research_agent.history_updater")

# ---------------------------------------------------------------------------
# Known baselines for key-finding comparison
# ---------------------------------------------------------------------------
CI_BASELINES: dict[str, dict[str, float]] = {
    "cnn_gru_attention": {"acc": 0.3085, "f1": 0.2819},
    "simple_cnn": {"acc": 0.2599, "f1": 0.2411},
    "svm_rbf": {"acc": 0.2782, "f1": 0.2663},
    "svm_linear": {"acc": 0.2551, "f1": 0.2443},
    "bilstm_attention": {"acc": 0.2620, "f1": 0.2322},
    "attention_cnn": {"acc": 0.2531, "f1": 0.2328},
    "bigru": {"acc": 0.2860, "f1": 0.2543},
    "rf": {"acc": 0.3200, "f1": 0.3033},
}

FULL_LOSO_BASELINES: dict[str, dict[str, float]] = {
    "cnn_gru_attention": {"acc": 0.3085, "f1": 0.2819},
    "svm_linear": {"acc": 0.3524, "f1": 0.3250},
    "svm_rbf": {"acc": 0.3446, "f1": 0.3260},
    "rf": {"acc": 0.3200, "f1": 0.3033},
    "simple_cnn": {"acc": 0.2930, "f1": 0.2551},
}

# Fallback goals extracted from experiment source code for experiments
# that don't have a "note" field in their loso_summary.json
_EXPERIMENT_GOALS: dict[int, str] = {
    22: "Replace cross-entropy with focal loss (gamma=2, per-class inverse frequency weighting) "
        "and class-balanced sampling to improve F1 on CNN-GRU-Attention",
    23: "Enhance CNN-GRU-Attention with channel-wise Squeeze-and-Excitation (SE) blocks "
        "combined with focal loss and cosine annealing warm restarts",
    25: "Apply combined noise + time-warp + amplitude-scaling augmentation before "
        "powerful feature extraction for SVM-Linear with balanced class weights",
    26: "Test-time batch normalization adaptation: update BN running statistics on "
        "unlabeled test subject data to reduce cross-subject domain gap",
    28: "FiLM conditioning layers for subject-style adaptation via learnable embeddings, "
        "computing z_subject from calibration windows to modulate CNN-GRU-Attention features",
    35: "Pretrain a Transformer encoder via masked temporal patch reconstruction (MAE-style) "
        "on unlabeled EMG data, then fine-tune for gesture classification",
}


@dataclass
class LeaderboardRow:
    """One row in a leaderboard table."""

    rank: int
    experiment_label: str
    model: str
    pipeline: str
    mean_acc: float
    mean_f1: float
    std_acc: float
    hypothesis_id_short: str = "---"


@dataclass
class RegistryEntry:
    """One experiment entry for Section 3."""

    experiment_id: int
    title: str
    goal: str
    pipeline: str
    models_tested: str
    result: str
    key_finding: str
    hypothesis_id_short: str = "---"
    relationship: str = ""


@dataclass
class HypothesisRow:
    """One row for Section 8 hypothesis tracker."""

    id_short: str
    hypothesis_text: str
    result_text: str
    verdict: str


# ============================================================================
# Public API
# ============================================================================


def update_research_history(
    history_path: Path,
    new_experiments: list[ExperimentSummary],
) -> dict:
    """Update RESEARCH_HISTORY.md with new experiment results.

    Returns dict with stats: {leaderboard_added, registry_added, hypotheses_added}.
    """
    if not history_path.exists():
        logger.warning("RESEARCH_HISTORY.md not found at %s", history_path)
        return {"leaderboard_added": 0, "registry_added": 0, "hypotheses_added": 0}

    content = history_path.read_text(encoding="utf-8")
    documented_ids = _find_documented_exp_ids(content)

    leaderboard_rows: list[tuple[str, LeaderboardRow]] = []
    registry_entries: list[RegistryEntry] = []
    hypothesis_rows: list[HypothesisRow] = []

    for exp in new_experiments:
        if exp.experiment_id is None:
            continue
        if exp.experiment_id in documented_ids:
            logger.debug("Exp %d already in RESEARCH_HISTORY, skipping", exp.experiment_id)
            continue

        # Leaderboard rows (one per model)
        for model_name, agg in exp.aggregate_results.items():
            section = _determine_leaderboard_section(agg)
            if section:
                row = _build_leaderboard_row(exp, model_name, agg)
                leaderboard_rows.append((section, row))

        # Registry entry (one per experiment)
        entry = _build_registry_entry(exp)
        registry_entries.append(entry)

        # Hypothesis tracker row
        hyp_row = _build_hypothesis_row(exp)
        if hyp_row is not None:
            hypothesis_rows.append(hyp_row)

    if not leaderboard_rows and not registry_entries and not hypothesis_rows:
        logger.info("No new experiments to add to RESEARCH_HISTORY.md")
        return {"leaderboard_added": 0, "registry_added": 0, "hypotheses_added": 0}

    # Apply modifications
    # Group leaderboard rows by section
    rows_by_section: dict[str, list[LeaderboardRow]] = {}
    for section, row in leaderboard_rows:
        rows_by_section.setdefault(section, []).append(row)
    for section, rows in rows_by_section.items():
        content = _insert_leaderboard_rows(content, rows, section)

    if registry_entries:
        content = _insert_registry_section(content, registry_entries)

    if hypothesis_rows:
        content = _insert_hypothesis_rows(content, hypothesis_rows)

    # Update "Last updated" date
    today = date.today().isoformat()
    content = re.sub(
        r"\*\*Last updated\*\*: \d{4}-\d{2}-\d{2}",
        f"**Last updated**: {today}",
        content,
    )

    history_path.write_text(content, encoding="utf-8")

    stats = {
        "leaderboard_added": len(leaderboard_rows),
        "registry_added": len(registry_entries),
        "hypotheses_added": len(hypothesis_rows),
    }
    logger.info("Updated RESEARCH_HISTORY.md: %s", stats)
    return stats


# ============================================================================
# Parsing helpers
# ============================================================================


def _find_documented_exp_ids(content: str) -> set[int]:
    """Find experiment IDs already documented in the registry (Section 3).

    Only scans EXP_N headers to avoid false matches from casual mentions.
    """
    ids: set[int] = set()
    for match in re.finditer(r"####\s+EXP_(\d+)", content):
        ids.add(int(match.group(1)))
    return ids


def _determine_leaderboard_section(agg: AggregateResult) -> str | None:
    """Return '2.1' or '2.2' based on num_subjects."""
    if agg.num_subjects >= 15:
        return "2.1"
    elif agg.num_subjects >= 3:
        return "2.2"
    return None


# ============================================================================
# Builders: create data structures from ExperimentSummary
# ============================================================================


def _build_leaderboard_row(
    exp: ExperimentSummary, model_name: str, agg: AggregateResult
) -> LeaderboardRow:
    """Create a LeaderboardRow from experiment data."""
    pipeline = exp.training_config.get("pipeline_type", "unknown")

    # Enrich pipeline label with augmentation/TTA info
    aug = exp.augmentation_desc
    if aug and aug != "none":
        pipeline += f" ({aug})"
    tta = exp.test_time_adaptation_desc
    if tta and tta.get("enabled"):
        pipeline += " (TTA)"
    mods = exp.training_modifications_desc
    if mods:
        if mods.get("loss_function") == "focal_loss":
            pipeline += " (focal)"
        if mods.get("sampling") == "class_balanced":
            pipeline += " (balanced)"

    # Short experiment label
    exp_label = _short_experiment_label(exp)

    hyp_short = exp.hypothesis_id_short or "---"

    return LeaderboardRow(
        rank=0,
        experiment_label=exp_label,
        model=model_name,
        pipeline=pipeline,
        mean_acc=agg.mean_accuracy,
        mean_f1=agg.mean_f1_macro,
        std_acc=agg.std_accuracy,
        hypothesis_id_short=hyp_short,
    )


def _build_registry_entry(exp: ExperimentSummary) -> RegistryEntry:
    """Create a registry entry for Section 3."""
    title = _readable_title(exp)
    pipeline = exp.training_config.get("pipeline_type", "unknown")
    model_names = ", ".join(exp.aggregate_results.keys())

    best_model, best_acc = exp.get_best_model()
    best_agg = exp.aggregate_results.get(best_model)
    if best_agg:
        result = (
            f"{best_acc * 100:.2f}% accuracy, {best_agg.mean_f1_macro * 100:.2f}% F1 "
            f"({best_agg.num_subjects} CI subjects)"
        )
        key_finding = _generate_key_finding(exp, best_model, best_agg)
    else:
        result = "N/A"
        key_finding = "No results available"

    goal = (
        exp.note
        or _EXPERIMENT_GOALS.get(exp.experiment_id or 0, "")
        or _infer_goal(exp)
    )
    relationship = _infer_relationship(exp)
    hyp_short = exp.hypothesis_id_short or "---"

    return RegistryEntry(
        experiment_id=exp.experiment_id or 0,
        title=title,
        goal=goal,
        pipeline=pipeline,
        models_tested=model_names,
        result=result,
        key_finding=key_finding,
        hypothesis_id_short=hyp_short,
        relationship=relationship,
    )


def _build_hypothesis_row(exp: ExperimentSummary) -> HypothesisRow | None:
    """Create a hypothesis tracker row, or None if no hypothesis_id."""
    if not exp.hypothesis_id_str:
        return None

    best_model, best_acc = exp.get_best_model()
    best_agg = exp.aggregate_results.get(best_model)
    if not best_agg:
        return None

    id_short = exp.hypothesis_id_short
    hypothesis_text = _readable_title(exp)

    acc_pct = best_acc * 100
    f1_pct = best_agg.mean_f1_macro * 100
    result_text = f"{acc_pct:.1f}% acc, {f1_pct:.1f}% F1"

    verdict = _generate_verdict(best_model, best_agg)

    return HypothesisRow(
        id_short=id_short,
        hypothesis_text=hypothesis_text,
        result_text=result_text,
        verdict=verdict,
    )


# ============================================================================
# Key finding & verdict generation (template-based)
# ============================================================================


def _get_baseline(model_name: str, num_subjects: int) -> dict[str, float] | None:
    """Find the best-matching baseline for a model."""
    baselines = CI_BASELINES if num_subjects <= 10 else FULL_LOSO_BASELINES

    # Exact match
    if model_name in baselines:
        return baselines[model_name]

    # Fuzzy: strip suffixes/prefixes
    for base_key in baselines:
        if base_key in model_name or model_name in base_key:
            return baselines[base_key]

    # Category match
    if "svm" in model_name:
        if "rbf" in model_name:
            return baselines.get("svm_rbf")
        return baselines.get("svm_linear")
    if "cnn_gru" in model_name:
        return baselines.get("cnn_gru_attention")
    if "attention" in model_name:
        return baselines.get("cnn_gru_attention")
    if "simple_cnn" in model_name:
        return baselines.get("simple_cnn")

    # Default: best deep baseline
    return baselines.get("cnn_gru_attention")


def _generate_key_finding(
    exp: ExperimentSummary, model_name: str, agg: AggregateResult
) -> str:
    """Generate a key finding by comparing to known baselines."""
    acc = agg.mean_accuracy
    f1 = agg.mean_f1_macro
    ratio = acc / f1 if f1 > 0.001 else float("inf")

    baseline = _get_baseline(model_name, agg.num_subjects)
    if baseline is None:
        return f"Accuracy: {acc * 100:.2f}%, F1: {f1 * 100:.2f}%"

    delta_pp = (acc - baseline["acc"]) * 100

    if delta_pp < -5:
        finding = f"DEGRADED from baseline ({baseline['acc'] * 100:.1f}%) by {abs(delta_pp):.1f}pp"
    elif delta_pp < -1:
        finding = f"Worse than baseline ({baseline['acc'] * 100:.1f}%) by {abs(delta_pp):.1f}pp"
    elif delta_pp < 1:
        finding = f"Matches baseline ({baseline['acc'] * 100:.1f}%)"
    elif delta_pp < 3:
        finding = f"Marginal improvement (+{delta_pp:.1f}pp) over baseline ({baseline['acc'] * 100:.1f}%)"
    else:
        finding = f"Improved over baseline by +{delta_pp:.1f}pp"

    # Acc/F1 balance analysis
    if ratio < 1.15:
        finding += f"; excellent Acc/F1 balance (ratio {ratio:.2f})"
    elif ratio > 2.0:
        finding += f"; severe class bias (Acc/F1 ratio {ratio:.2f})"
    elif ratio > 1.5:
        finding += f"; moderate class bias (Acc/F1 ratio {ratio:.2f})"

    return finding


def _generate_verdict(model_name: str, agg: AggregateResult) -> str:
    """Generate REJECTED / CONFIRMED / PARTIALLY CONFIRMED verdict."""
    baseline = _get_baseline(model_name, agg.num_subjects)
    if baseline is None:
        return "INCONCLUSIVE"

    delta_acc = agg.mean_accuracy - baseline["acc"]
    delta_f1 = agg.mean_f1_macro - baseline["f1"]

    if delta_acc > 0.02 and delta_f1 > 0.01:
        return "CONFIRMED"
    if delta_acc > 0.01 and delta_f1 < -0.02:
        return "PARTIALLY CONFIRMED -- accuracy up but F1 down"
    if abs(delta_acc) < 0.01:
        return "REJECTED -- no significant change"
    if delta_acc < -0.02:
        return f"REJECTED -- degraded by {abs(delta_acc * 100):.1f}pp"
    return "INCONCLUSIVE"


# ============================================================================
# Markdown insertion helpers
# ============================================================================


def _insert_leaderboard_rows(
    content: str, new_rows: list[LeaderboardRow], section: str
) -> str:
    """Parse existing leaderboard table, merge new rows, re-sort, re-rank."""
    if section == "2.2":
        header_pattern = r"### 2\.2 CI Subset LOSO"
    else:
        header_pattern = r"### 2\.1 Full LOSO"

    header_match = re.search(header_pattern, content)
    if not header_match:
        logger.warning("Could not find leaderboard section %s", section)
        return content

    # Find the markdown table after the header
    after_header = content[header_match.start():]
    lines = after_header.split("\n")

    table_header_idx = None
    table_sep_idx = None
    table_end_idx = None
    header_line = ""
    sep_line = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("| Rank"):
            table_header_idx = i
            header_line = line
        elif table_header_idx is not None and table_sep_idx is None and stripped.startswith("|---"):
            table_sep_idx = i
            sep_line = line
        elif table_sep_idx is not None and stripped.startswith("|"):
            table_end_idx = i + 1  # keep extending as long as we see table rows
        elif table_sep_idx is not None and not stripped.startswith("|"):
            break

    if table_header_idx is None or table_sep_idx is None:
        logger.warning("Could not parse leaderboard table in section %s", section)
        return content

    # Parse existing data rows
    existing_rows: list[LeaderboardRow] = []
    data_start = table_sep_idx + 1
    data_end = table_end_idx if table_end_idx else data_start

    for i in range(data_start, data_end):
        row = _parse_leaderboard_row(lines[i])
        if row:
            existing_rows.append(row)

    # Merge and sort
    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda r: r.mean_acc, reverse=True)
    for i, row in enumerate(all_rows, 1):
        row.rank = i

    # Rebuild table lines
    rebuilt = [header_line, sep_line]
    for row in all_rows:
        acc_str = f"**{row.mean_acc:.4f}**" if row.rank == 1 else f"{row.mean_acc:.4f}"
        rebuilt.append(
            f"| {row.rank} | {row.experiment_label} | {row.model} | "
            f"{row.pipeline} | {acc_str} | {row.mean_f1:.4f} | "
            f"{row.std_acc:.4f} | {row.hypothesis_id_short} |"
        )

    # Replace old table lines with rebuilt
    new_lines = lines[:table_header_idx] + rebuilt + lines[data_end:]
    new_after_header = "\n".join(new_lines)
    content = content[: header_match.start()] + new_after_header

    return content


def _parse_leaderboard_row(line: str) -> LeaderboardRow | None:
    """Parse a single markdown table row into a LeaderboardRow."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None

    cells = [c.strip() for c in stripped.split("|")]
    # Split on | gives ['', cell1, cell2, ..., '']
    cells = [c for c in cells if c != ""]

    if len(cells) < 7:
        return None

    try:
        return LeaderboardRow(
            rank=int(cells[0]),
            experiment_label=cells[1].strip(),
            model=cells[2].strip(),
            pipeline=cells[3].strip(),
            mean_acc=float(cells[4].replace("**", "").strip()),
            mean_f1=float(cells[5].strip()),
            std_acc=float(cells[6].strip()),
            hypothesis_id_short=cells[7].strip() if len(cells) > 7 else "---",
        )
    except (ValueError, IndexError):
        return None


def _insert_registry_section(
    content: str, entries: list[RegistryEntry]
) -> str:
    """Insert a new ### 3.N subsection into the experiment registry."""
    # Find the last existing subsection number
    last_subsection = 0
    for match in re.finditer(r"### 3\.(\d+)", content):
        last_subsection = max(last_subsection, int(match.group(1)))

    new_section_num = last_subsection + 1

    # Determine range label
    exp_ids = sorted(e.experiment_id for e in entries)
    if len(exp_ids) > 1:
        range_label = f"exp_{exp_ids[0]} through exp_{exp_ids[-1]}"
    else:
        range_label = f"exp_{exp_ids[0]}"

    # Auto-generate section title from experiment themes
    section_title = _infer_section_title(entries)

    lines = [
        f"### 3.{new_section_num} {section_title} ({range_label})",
        "",
    ]

    for entry in sorted(entries, key=lambda e: e.experiment_id):
        lines.extend([
            f"#### EXP_{entry.experiment_id}: {entry.title}",
            f"- **Goal**: {entry.goal}",
            f"- **Pipeline**: `{entry.pipeline}`",
            f"- **Models tested**: {entry.models_tested}",
            f"- **Result**: {entry.result}",
            f"- **Key finding**: {entry.key_finding}",
            f"- **Hypothesis ID**: {entry.hypothesis_id_short}",
            f"- **Relationship**: {entry.relationship}",
            "",
        ])

    new_section_text = "\n".join(lines)

    # Insert before "---\n\n## 4."
    insert_pos = content.find("\n## 4.")
    if insert_pos == -1:
        logger.warning("Could not find Section 4 marker to insert registry")
        return content

    # Find the preceding "---" separator
    sep_pos = content.rfind("---", 0, insert_pos)
    if sep_pos != -1 and insert_pos - sep_pos < 20:
        insert_at = sep_pos
    else:
        insert_at = insert_pos

    content = content[:insert_at] + new_section_text + "\n" + content[insert_at:]
    return content


def _insert_hypothesis_rows(content: str, new_rows: list[HypothesisRow]) -> str:
    """Append new rows to the Verified Hypotheses table in Section 8."""
    marker = "### Verified Hypotheses"
    pos = content.find(marker)
    if pos == -1:
        logger.warning("Could not find Verified Hypotheses table")
        return content

    # Find the table bounded by the first non-table line after the header
    table_section = content[pos:]
    lines = table_section.split("\n")
    last_table_line_idx = 0
    in_table = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("| ") and "|" in stripped[1:]:
            in_table = True
            last_table_line_idx = i
        elif in_table and not stripped.startswith("|"):
            # End of table reached
            break

    if not in_table:
        logger.warning("Could not find table rows in hypothesis tracker")
        return content

    # Calculate character offset of the line after the last table row
    insert_offset = sum(len(lines[j]) + 1 for j in range(last_table_line_idx + 1))
    insert_pos = pos + insert_offset

    new_lines = []
    for row in new_rows:
        new_lines.append(
            f"| {row.id_short} | {row.hypothesis_text} | {row.result_text} | {row.verdict} |"
        )

    new_text = "\n".join(new_lines) + "\n"
    content = content[:insert_pos] + new_text + content[insert_pos:]
    return content


# ============================================================================
# Text generation helpers
# ============================================================================


def _short_experiment_label(exp: ExperimentSummary) -> str:
    """Create a short label like 'exp_21 aug SVM-RBF'."""
    name = exp.experiment_name.replace("_loso", "")
    # e.g. "exp_21_svm_rbf_with_noise_time_warp_augmentation_and_tune"
    parts = name.split("_")
    # Skip "exp" and the number
    if len(parts) > 2:
        title_words = parts[2:6]
        short_desc = " ".join(w for w in title_words)
        return f"exp_{exp.experiment_id} {short_desc}"
    return f"exp_{exp.experiment_id}"


def _readable_title(exp: ExperimentSummary) -> str:
    """Create a readable title from experiment name."""
    name = exp.experiment_name.replace("_loso", "")
    parts = name.split("_")
    if len(parts) > 2:
        title_words = parts[2:]
        return " ".join(w.capitalize() for w in title_words)
    return name


def _infer_goal(exp: ExperimentSummary) -> str:
    """Infer experiment goal from configuration when no note is available."""
    model_type = exp.training_config.get("model_type", "unknown")
    pipeline = exp.training_config.get("pipeline_type", "unknown")
    aug = exp.augmentation_desc or ""
    tta = exp.test_time_adaptation_desc
    mods = exp.training_modifications_desc

    parts = [f"Test {model_type} on {pipeline} pipeline"]

    if aug and aug != "none":
        parts.append(f"with {aug} augmentation")
    if tta and tta.get("enabled"):
        parts.append(f"with {tta.get('method', 'test-time adaptation')}")
    if mods:
        if mods.get("loss_function") == "focal_loss":
            parts.append("using focal loss")
        if mods.get("sampling") == "class_balanced":
            parts.append("with class-balanced sampling")

    return "; ".join(parts)


def _infer_relationship(exp: ExperimentSummary) -> str:
    """Infer experiment relationship from configuration."""
    parts = []
    model_type = exp.training_config.get("model_type", "")
    pipeline = exp.training_config.get("pipeline_type", "")
    tta = exp.test_time_adaptation_desc
    mods = exp.training_modifications_desc

    # Also check actual model names in results
    actual_models = list(exp.aggregate_results.keys())
    all_model_hints = [model_type] + actual_models

    # Determine base experiment from model/pipeline
    if any("svm" in m for m in all_model_hints) or pipeline == "ml_emg_td":
        parts.append("Builds on exp_4 (SVM baseline)")
    elif any("cnn_gru_attention" in m for m in all_model_hints):
        parts.append("Builds on exp_1 (CNN-GRU-Attention baseline)")
    elif any("simple_cnn" in m for m in all_model_hints):
        parts.append("Builds on exp_1 (SimpleCNN baseline)")
    else:
        parts.append("Extension of baseline experiments")

    aug = exp.augmentation_desc or ""
    if aug and aug != "none":
        parts.append(f"with {aug}")
    if tta and tta.get("enabled"):
        parts.append(f"+ test-time adaptation (Direction B)")
    if mods:
        if mods.get("loss_function") == "focal_loss":
            parts.append("+ focal loss (Direction E)")
        if mods.get("sampling") == "class_balanced":
            parts.append("+ class-balanced sampling (Direction E)")

    return "; ".join(parts)


def _infer_section_title(entries: list[RegistryEntry]) -> str:
    """Infer a short section title from the batch of experiments."""
    themes: list[str] = []
    for entry in entries:
        title_lower = entry.title.lower()
        if ("focal" in title_lower or "class" in title_lower
                or "balanced" in title_lower or "weighted" in title_lower):
            if "Class Balancing" not in themes:
                themes.append("Class Balancing")
        if ("test" in title_lower and "time" in title_lower
                or "adaptation" in title_lower or "bn" in title_lower):
            if "Test-Time Adaptation" not in themes:
                themes.append("Test-Time Adaptation")
        if ("squeeze" in title_lower or "se_" in title_lower
                or "architecture" in title_lower):
            if "Architecture" not in themes:
                themes.append("Architecture")
        if "augment" in title_lower or "noise" in title_lower or "jitter" in title_lower:
            if "Augmentation" not in themes:
                themes.append("Augmentation")

    if not themes:
        return "Additional Experiments"

    # Keep at most 2 themes for a readable title
    return " & ".join(themes[:2])
