#!/usr/bin/env python3
"""
Collect experiment results from remote servers and update Qdrant on the host.

Usage:
    python scripts/collect_results.py                    # collect + update Qdrant
    python scripts/collect_results.py --collect-only     # only rsync from servers
    python scripts/collect_results.py --update-qdrant    # only process fallback JSONs
    python scripts/collect_results.py --summary          # print results summary
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "servers.yaml"
FALLBACK_DIR = PROJECT_ROOT / "experiments_output" / "_pending_qdrant_updates"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def collect_from_servers(config: dict, experiments: List[str] = None, server_filter: str = None) -> None:
    """rsync results from all configured servers.

    Args:
        config: Server configuration dict.
        experiments: Optional list of experiment prefixes to collect (e.g. ["exp_12", "exp_15"]).
                     If None, collects everything.
        server_filter: If set, collect only from the server with this name.
    """
    host_results_dir = Path(config.get("host_results_dir", "./results_collected"))
    host_results_dir.mkdir(parents=True, exist_ok=True)

    if experiments:
        print(f"Collecting specific experiments: {', '.join(experiments)}")

    for server in config.get("servers", []):
        if server_filter and server.get("name") != server_filter:
            continue
        ssh_key = str(Path(server.get("ssh_key", "~/.ssh/id_rsa")).expanduser())
        port = server.get("port", 22)
        user = server["user"]
        host = server["host"]
        work_dir = server["work_dir"]
        name = server["name"]

        print(f"[{name}] Collecting results from {user}@{host}...")

        if experiments:
            # Collect only specified experiment directories
            for exp in experiments:
                rsync_cmd = [
                    "rsync", "-avz", "--progress",
                    "-e", f"ssh -i {ssh_key} -p {port}",
                    f"{user}@{host}:{work_dir}/experiments_output/{exp}*",
                    str(host_results_dir) + "/",
                ]
                result = subprocess.run(rsync_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    # rsync returns 23 when source doesn't exist — not a fatal error
                    if result.returncode == 23:
                        print(f"  [{name}] No results for {exp}* (not found on server)")
                    else:
                        print(f"  [{name}] ERROR syncing {exp}: {result.stderr.strip()}")
                else:
                    print(f"  [{name}] Synced {exp}*")
        else:
            # Collect everything
            rsync_cmd = [
                "rsync", "-avz", "--progress",
                "-e", f"ssh -i {ssh_key} -p {port}",
                f"{user}@{host}:{work_dir}/experiments_output/",
                str(host_results_dir) + "/",
            ]
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [{name}] ERROR: {result.stderr.strip()}")
            else:
                print(f"  [{name}] Synced successfully")

        # Also collect any fallback Qdrant JSONs
        rsync_qdrant = [
            "rsync", "-avz",
            "-e", f"ssh -i {ssh_key} -p {port}",
            f"{user}@{host}:{work_dir}/experiments_output/_pending_qdrant_updates/",
            str(FALLBACK_DIR) + "/",
        ]
        subprocess.run(rsync_qdrant, capture_output=True, text=True)

    print(f"\nResults collected to: {host_results_dir.resolve()}")


def process_fallback_jsons() -> int:
    """Read fallback JSONs and update Qdrant."""
    if not FALLBACK_DIR.exists():
        print("No pending Qdrant updates found.")
        return 0

    json_files = sorted(FALLBACK_DIR.glob("*.json"))
    if not json_files:
        print("No pending Qdrant updates found.")
        return 0

    print(f"\nFound {len(json_files)} pending Qdrant updates:")

    from hypothesis_executor.qdrant_callback import (
        mark_hypothesis_verified,
        mark_hypothesis_failed,
    )

    processed = 0
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        hypothesis_id = data.get("hypothesis_id", "")
        status = data.get("status", "")

        print(f"  Processing: {jf.name} (status={status}, id={hypothesis_id[:8]}...)")

        ok = False
        if status == "verified":
            ok = mark_hypothesis_verified(
                hypothesis_id=hypothesis_id,
                metrics=data.get("metrics", {}),
                experiment_name=data.get("experiment_name", ""),
            )
        elif status == "failed":
            ok = mark_hypothesis_failed(
                hypothesis_id=hypothesis_id,
                error_message=data.get("error_message", "Unknown error"),
            )

        if ok:
            # Move processed file to avoid re-processing
            processed_dir = FALLBACK_DIR / "_processed"
            processed_dir.mkdir(exist_ok=True)
            jf.rename(processed_dir / jf.name)
            processed += 1
        else:
            print(f"  WARNING: Failed to update Qdrant for {jf.name}")

    print(f"\nUpdated {processed}/{len(json_files)} hypotheses in Qdrant")
    return processed


def print_summary(config: dict) -> None:
    """Print summary of all collected results."""
    host_results_dir = Path(config.get("host_results_dir", "./results_collected"))

    # Also check local experiments_output
    search_dirs = [host_results_dir, PROJECT_ROOT / "experiments_output"]

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    results = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for summary_file in sorted(search_dir.glob("*/loso_summary.json")):
            with open(summary_file) as f:
                summary = json.load(f)

            exp_name = summary.get("experiment_name", summary_file.parent.name)
            aggregate = summary.get("aggregate_results", {})

            if not aggregate:
                results.append((exp_name, "NO RESULTS", 0, 0, 0, str(summary_file.parent)))
                continue

            # Find best model
            if isinstance(aggregate, dict):
                for model_name, metrics in aggregate.items():
                    if isinstance(metrics, dict) and "mean_accuracy" in metrics:
                        results.append((
                            exp_name,
                            model_name,
                            metrics.get("mean_accuracy", 0),
                            metrics.get("std_accuracy", 0),
                            metrics.get("mean_f1_macro", 0),
                            str(summary_file.parent),
                        ))

    if not results:
        print("No results found.")
        return

    # Sort by accuracy descending
    results.sort(key=lambda x: x[2] if isinstance(x[2], (int, float)) else 0, reverse=True)

    print(f"\n{'Experiment':<55} {'Model':<25} {'Acc':>8} {'±':>6} {'F1':>8}")
    print("-" * 110)
    for exp_name, model, acc, std, f1, path in results:
        if isinstance(acc, (int, float)) and acc > 0:
            print(f"{exp_name:<55} {model:<25} {acc:>7.4f} ±{std:>5.4f} {f1:>7.4f}")
        else:
            print(f"{exp_name:<55} {'NO RESULTS':<25}")


def main():
    parser = argparse.ArgumentParser(description="Collect results and update Qdrant")
    parser.add_argument("--collect-only", action="store_true", help="Only collect from servers (no Qdrant update)")
    parser.add_argument("--update-qdrant", action="store_true", help="Only process fallback JSONs into Qdrant")
    parser.add_argument("--summary", action="store_true", help="Print results summary")
    parser.add_argument("--exp", type=str, default="",
                        help="Comma-separated experiment prefixes to collect (e.g. exp_12,exp_15)")
    parser.add_argument("--server", type=str, default="",
                        help="Collect only from this server (by name)")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.exp.split(",") if e.strip()] or None
    server_filter = args.server or None

    config = load_config()

    if args.summary:
        print_summary(config)
        return

    if args.update_qdrant:
        process_fallback_jsons()
        return

    if args.collect_only:
        collect_from_servers(config, experiments, server_filter)
        return

    # Default: collect + update Qdrant + summary
    collect_from_servers(config, experiments, server_filter)
    process_fallback_jsons()
    print_summary(config)


if __name__ == "__main__":
    main()
