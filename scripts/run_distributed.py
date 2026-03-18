#!/usr/bin/env python3
"""
Distributed experiment runner — launches experiments on remote GPU servers via SSH.
Designed for vast.ai and similar bare-metal/container GPU providers (no Docker on remote).

Usage:
    python scripts/run_distributed.py --experiments exp_12,exp_13,exp_14
    python scripts/run_distributed.py --experiments exp_12 --server gpu-server-1
    python scripts/run_distributed.py --all-pending --ci
    python scripts/run_distributed.py --experiments exp_12 --subjects DB2_s1,DB2_s12
    python scripts/run_distributed.py --status
    python scripts/run_distributed.py --logs gpu-server-1
    python scripts/run_distributed.py --collect
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "servers.yaml"


@dataclass
class ServerConfig:
    name: str
    host: str
    port: int
    user: str
    ssh_key: str
    gpu_ids: List[int]
    work_dir: str
    data_dir: str
    max_concurrent: int = 1


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_servers(config: dict, target_server: Optional[str] = None) -> List[ServerConfig]:
    servers = []
    for s in config.get("servers", []):
        if target_server and s["name"] != target_server:
            continue
        servers.append(ServerConfig(
            name=s["name"],
            host=s["host"],
            port=s.get("port", 22),
            user=s["user"],
            ssh_key=str(Path(s.get("ssh_key", "~/.ssh/id_rsa")).expanduser()),
            gpu_ids=s.get("gpu_ids", [0]),
            work_dir=s["work_dir"],
            data_dir=s["data_dir"],
            max_concurrent=s.get("max_concurrent", 1),
        ))
    return servers


def find_experiment_files(experiment_names: List[str]) -> List[Path]:
    """Find experiment .py files by short names (e.g. 'exp_12')."""
    exp_dir = PROJECT_ROOT / "experiments"
    found = []
    for name in experiment_names:
        name = name.strip()
        matches = sorted(exp_dir.glob(f"{name}_*_loso.py"))
        if not matches:
            print(f"WARNING: No experiment file found for '{name}'")
        else:
            found.append(matches[0])
    return found


def find_pending_experiments() -> List[Path]:
    """Find experiments that don't have results yet."""
    exp_dir = PROJECT_ROOT / "experiments"
    output_dir = PROJECT_ROOT / "experiments_output"
    pending = []
    for exp_file in sorted(exp_dir.glob("exp_[0-9]*_*_loso.py")):
        if exp_file.name == "exp_X_template_loso.py":
            continue
        exp_name = exp_file.stem
        if not (output_dir / exp_name / "loso_summary.json").exists():
            pending.append(exp_file)
    return pending


def ssh_cmd(server: ServerConfig) -> List[str]:
    return [
        "ssh",
        "-i", server.ssh_key,
        "-p", str(server.port),
        "-o", "StrictHostKeyChecking=no",
        f"{server.user}@{server.host}",
    ]


def run_experiment_on_server(
    server: ServerConfig,
    exp_file: Path,
    subjects: Optional[str] = None,
    ci: bool = False,
    detach: bool = True,
) -> bool:
    """Launch an experiment on the remote server via SSH (no Docker)."""
    exp_filename = exp_file.name
    log_name = exp_file.stem

    # Build subject args
    subject_args = ""
    if subjects:
        subject_args = f" --subjects {subjects}"
    elif ci:
        subject_args = " --ci"

    # Build the remote command
    # Use nohup + redirect to run in background and survive SSH disconnect
    gpu_str = ",".join(str(g) for g in server.gpu_ids)

    if detach:
        remote_cmd = (
            f"cd {server.work_dir} && "
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"nohup python3 experiments/{exp_filename}{subject_args} "
            f"> experiments_output/{log_name}.log 2>&1 &"
            f" echo $!"
        )
    else:
        remote_cmd = (
            f"cd {server.work_dir} && "
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"python3 experiments/{exp_filename}{subject_args}"
        )

    full_cmd = ssh_cmd(server) + [remote_cmd]
    print(f"  [{server.name}] Launching: {exp_filename}{subject_args}")

    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{server.name}] ERROR: {result.stderr.strip()}")
        return False

    if detach and result.stdout.strip():
        pid = result.stdout.strip().split("\n")[-1]
        print(f"  [{server.name}] Started (PID: {pid}), log: experiments_output/{log_name}.log")
    return True


def check_status(servers: List[ServerConfig]):
    """Check running experiment processes on all servers."""
    print("\n=== Experiment Status ===\n")
    for server in servers:
        # Check for running Python experiment processes
        cmd = ssh_cmd(server) + [
            f"ps aux | grep 'python3 experiments/exp_' | grep -v grep || echo 'No running experiments'"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"[{server.name}] ({server.host}):")
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                print(f"  {line.strip()}")

        # Show recent log files
        cmd2 = ssh_cmd(server) + [
            f"ls -lt {server.work_dir}/experiments_output/*.log 2>/dev/null | head -5 || echo '  No log files'"
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.stdout.strip():
            print(f"  Recent logs:")
            for line in result2.stdout.strip().split("\n"):
                print(f"    {line.strip()}")
        print()


def get_logs(servers: List[ServerConfig], server_name: str, tail: int = 50, exp_filter: Optional[str] = None):
    """Get experiment logs from a specific server.

    Args:
        exp_filter: If set, show only logs matching this experiment prefix (e.g. "exp_12").
    """
    for server in servers:
        if server.name != server_name:
            continue

        # Build glob pattern — filter by experiment prefix if requested
        log_glob = f"{server.work_dir}/experiments_output/{exp_filter}_*.log" if exp_filter \
            else f"{server.work_dir}/experiments_output/*.log"

        cmd = ssh_cmd(server) + [f"ls -t {log_glob} 2>/dev/null"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        log_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        if not log_files:
            msg = f"[{server.name}] No log files found"
            if exp_filter:
                msg += f" for '{exp_filter}'"
            print(msg)
            return

        max_logs = 1 if exp_filter else 3
        for log_file in log_files[:max_logs]:
            print(f"\n=== {Path(log_file).name} ===")
            cmd = ssh_cmd(server) + [f"tail -{tail} {log_file}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)


def kill_experiments(servers: List[ServerConfig], experiment_names: Optional[List[str]] = None):
    """Kill running experiment processes on servers."""
    print("\n=== Killing experiments ===\n")
    for server in servers:
        if experiment_names:
            # Kill specific experiments by name
            for name in experiment_names:
                pattern = f"experiments/{name}_"
                cmd = ssh_cmd(server) + [
                    f"pkill -f '{pattern}' && echo 'Killed {name}' || echo '{name}: not running'"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(f"  [{server.name}] {result.stdout.strip()}")
        else:
            # Kill all experiments
            cmd = ssh_cmd(server) + [
                "pkill -f 'python3 experiments/exp_' && echo 'Killed all experiments' || echo 'No running experiments'"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"  [{server.name}] {result.stdout.strip()}")


def collect_results_from_server(server: ServerConfig, host_results_dir: Path):
    """Collect experiment results from a remote server via rsync."""
    host_results_dir.mkdir(parents=True, exist_ok=True)

    rsync_ssh = f"ssh -i {server.ssh_key} -p {server.port} -o StrictHostKeyChecking=no"

    rsync_cmd = [
        "rsync", "-avz", "--progress",
        "-e", rsync_ssh,
        f"{server.user}@{server.host}:{server.work_dir}/experiments_output/",
        str(host_results_dir) + "/",
    ]

    print(f"  [{server.name}] Collecting results...")
    result = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{server.name}] rsync ERROR: {result.stderr.strip()}")
    else:
        # Count collected dirs
        print(f"  [{server.name}] Results collected successfully")


def main():
    parser = argparse.ArgumentParser(description="Distributed experiment runner (SSH, no Docker)")
    parser.add_argument("--experiments", type=str, help="Comma-separated experiment names (e.g. exp_12,exp_13)")
    parser.add_argument("--all-pending", action="store_true", help="Run all experiments without results")
    parser.add_argument("--server", type=str, help="Target specific server by name")
    parser.add_argument("--subjects", type=str, help="Comma-separated subject IDs")
    parser.add_argument("--ci", action="store_true", help="Use CI test subset (5 subjects)")
    parser.add_argument("--status", action="store_true", help="Check status of running experiments")
    parser.add_argument("--logs", type=str, metavar="SERVER", help="Get logs from a server")
    parser.add_argument("--log-exp", type=str, metavar="EXP", help="Filter logs by experiment (e.g. exp_12)")
    parser.add_argument("--collect", action="store_true", help="Collect results from all servers")
    parser.add_argument("--kill", type=str, metavar="EXPS", help="Kill specific experiments (e.g. exp_7,exp_11)")
    parser.add_argument("--kill-all", action="store_true", help="Kill ALL running experiments on servers")
    parser.add_argument("--no-detach", action="store_true", help="Run in foreground (wait for completion)")
    args = parser.parse_args()

    config = load_config()
    host_results_dir = Path(config.get("host_results_dir", "./results_collected"))
    servers = load_servers(config, args.server)

    if not servers:
        print("ERROR: No servers configured. Edit config/servers.yaml")
        sys.exit(1)

    # --- Status check ---
    if args.status:
        check_status(servers)
        return

    # --- Logs ---
    if args.logs:
        get_logs(servers, args.logs, exp_filter=args.log_exp)
        return

    # --- Kill experiments ---
    if args.kill:
        names = [e.strip() for e in args.kill.split(",")]
        kill_experiments(servers, names)
        return

    if args.kill_all:
        kill_experiments(servers)
        return

    # --- Collect results ---
    if args.collect:
        print("=== Collecting results from servers ===")
        for server in servers:
            collect_results_from_server(server, host_results_dir)
        print(f"\nResults collected to: {host_results_dir.resolve()}")
        return

    # --- Run experiments ---
    if args.experiments:
        exp_names = [e.strip() for e in args.experiments.split(",")]
        exp_files = find_experiment_files(exp_names)
    elif args.all_pending:
        exp_files = find_pending_experiments()
        if not exp_files:
            print("No pending experiments found.")
            return
        print(f"Found {len(exp_files)} pending experiments:")
        for f in exp_files:
            print(f"  {f.stem}")
    else:
        parser.print_help()
        return

    if not exp_files:
        print("No experiment files to run.")
        return

    # Distribute experiments across servers (round-robin)
    print(f"\n=== Distributing {len(exp_files)} experiments across {len(servers)} servers ===\n")
    launched = 0

    for i, exp_file in enumerate(exp_files):
        server = servers[i % len(servers)]
        ok = run_experiment_on_server(
            server=server,
            exp_file=exp_file,
            subjects=args.subjects,
            ci=args.ci,
            detach=not args.no_detach,
        )
        if ok:
            launched += 1

    print(f"\n=== Launched {launched}/{len(exp_files)} experiments ===")
    print("\nTo check status:    python scripts/run_distributed.py --status")
    print("To get logs:        python scripts/run_distributed.py --logs <server-name>")
    print("To collect results: python scripts/run_distributed.py --collect")


if __name__ == "__main__":
    main()
