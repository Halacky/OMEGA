"""CLI entry point for the research agent."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_agent.config import AgentConfig


def setup_logging(config: AgentConfig) -> None:
    """Configure logging for the agent."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("research_agent")
    root_logger.setLevel(getattr(logging, config.log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File handler
    from datetime import datetime

    log_file = log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initialized. Log file: %s", log_file)


def main():
    parser = argparse.ArgumentParser(
        description="OMEGA Research Hypothesis Generation Agent"
    )
    parser.add_argument(
        "--strategy",
        choices=["exploitation", "exploration", "literature", "error"],
        default="exploitation",
        help="Hypothesis generation strategy (default: exploitation)",
    )
    parser.add_argument(
        "--num-hypotheses",
        type=int,
        default=1,
        help="Number of hypotheses to generate (default: 1)",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Force initialization of the knowledge base from baseline experiments",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to .env config file",
    )

    args = parser.parse_args()

    # Load config
    config = AgentConfig()
    setup_logging(config)

    logger = logging.getLogger("research_agent.main")
    logger.info("=" * 60)
    logger.info("OMEGA Research Agent")
    logger.info("Strategy: %s", args.strategy)
    logger.info("Hypotheses to generate: %d", args.num_hypotheses)
    logger.info("Project root: %s", config.project_root)
    logger.info("=" * 60)

    # Run agent
    from research_agent.agent.graph import run_agent

    result = run_agent(
        config,
        strategy=args.strategy,
        num_hypotheses=args.num_hypotheses,
        init_db=args.init_db,
    )

    # Print results
    accepted = result.get("accepted_hypotheses", [])
    rejected = result.get("rejected_hypotheses", [])

    print("\n" + "=" * 60)
    print(f"RESULTS: {len(accepted)} accepted, {len(rejected)} rejected")
    print("=" * 60)

    for i, hyp in enumerate(accepted, 1):
        print(f"\n--- Hypothesis {i} ---")
        print(json.dumps(hyp, indent=2, ensure_ascii=False))

    if rejected:
        print(f"\n--- Rejected ({len(rejected)}) ---")
        for r in rejected:
            title = r.get("title", "unknown")
            reason = r.get("rejection_reason", "unknown")
            print(f"  - {title}: {reason}")


if __name__ == "__main__":
    main()
