"""Runs generated experiment scripts and parses their results."""

import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hypothesis_executor.runner")


@dataclass
class RunResult:
    """Result of running an experiment script."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


class ExperimentRunner:
    """Runs experiment scripts as subprocesses and parses their results."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.output_dir = project_root / "experiments_output"

    def run_experiment(
        self, experiment_path: str, timeout: int = 7200
    ) -> RunResult:
        """Run an experiment script as a subprocess.

        Args:
            experiment_path: Absolute path to the experiment .py file.
            timeout: Max seconds before killing the process.

        Returns:
            RunResult with stdout/stderr and success flag.
        """
        experiment_path = str(experiment_path)
        logger.info("Running experiment: %s (timeout=%ds)", experiment_path, timeout)

        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, experiment_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.root),
            )
            duration = time.monotonic() - start
            success = proc.returncode == 0

            if success:
                logger.info("Experiment completed in %.1fs", duration)
            else:
                logger.error(
                    "Experiment failed (code=%d) in %.1fs:\n%s",
                    proc.returncode, duration, proc.stderr[-2000:],
                )

            return RunResult(
                success=success,
                return_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            logger.error("Experiment timed out after %.1fs", duration)
            return RunResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Timed out after {timeout}s",
                duration_seconds=duration,
                timed_out=True,
            )

        except Exception as e:
            duration = time.monotonic() - start
            logger.error("Experiment execution error: %s", e)
            return RunResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
            )

    def run_mini_test(
        self, experiment_path: str, num_subjects: int = 2
    ) -> RunResult:
        """Run experiment with reduced subject count for quick validation.

        Creates a temporary copy of the experiment with a smaller ALL_SUBJECTS
        list, runs it, then cleans up.

        Args:
            experiment_path: Path to the experiment .py file.
            num_subjects: Number of subjects to use for mini test.

        Returns:
            RunResult from running the mini test.
        """
        experiment_path = Path(experiment_path)
        logger.info(
            "Running mini test (%d subjects): %s",
            num_subjects, experiment_path.name,
        )

        source = experiment_path.read_text(encoding="utf-8")

        # Find ALL_SUBJECTS list and replace with first N subjects
        modified = self._reduce_subjects(source, num_subjects)
        if modified is None:
            logger.warning(
                "Could not find ALL_SUBJECTS in experiment. Running full experiment."
            )
            return self.run_experiment(str(experiment_path), timeout=600)

        # Also change experiment name to avoid overwriting full results
        modified = modified.replace(
            'EXPERIMENT_NAME = "',
            'EXPERIMENT_NAME = "mini_test_',
            1,
        )

        # Write to temp file in same directory (so relative imports work)
        temp_path = experiment_path.parent / f"_mini_test_{experiment_path.name}"
        try:
            temp_path.write_text(modified, encoding="utf-8")
            result = self.run_experiment(str(temp_path), timeout=600)
            return result
        finally:
            if temp_path.exists():
                temp_path.unlink()
            # Clean up mini test output directory
            for d in self.output_dir.iterdir():
                if d.is_dir() and d.name.startswith("mini_test_"):
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)

    def parse_results(self, experiment_name: str) -> Optional[dict]:
        """Parse loso_summary.json for the given experiment.

        Args:
            experiment_name: Name of the experiment (directory name in experiments_output).

        Returns:
            Dict with metrics or None if not found:
            {
                "mean_accuracy": float,
                "std_accuracy": float,
                "mean_f1_macro": float,
                "std_f1_macro": float,
                "num_subjects": int,
            }
        """
        # Try exact match first
        summary_path = self.output_dir / experiment_name / "loso_summary.json"
        if not summary_path.exists():
            # Try scanning for partial match (experiment names may vary)
            for d in self.output_dir.iterdir():
                if d.is_dir() and experiment_name in d.name:
                    candidate = d / "loso_summary.json"
                    if candidate.exists():
                        summary_path = candidate
                        break

        if not summary_path.exists():
            logger.warning("loso_summary.json not found for: %s", experiment_name)
            return None

        logger.info("Parsing results from: %s", summary_path)

        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        aggregate = data.get("aggregate_results", {})
        if not aggregate:
            logger.warning("No aggregate_results in loso_summary.json")
            return None

        # Collect metrics across all models in the experiment
        all_accs = []
        all_f1s = []
        total_subjects = 0
        per_model: dict = {}

        for model_name, model_data in aggregate.items():
            if isinstance(model_data, dict):
                mean_acc = model_data.get("mean_accuracy", 0.0)
                std_acc = model_data.get("std_accuracy", 0.0)
                mean_f1 = model_data.get("mean_f1_macro", 0.0)
                std_f1 = model_data.get("std_f1_macro", 0.0)
                n = model_data.get("num_subjects", 0)

                per_model[model_name] = {
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "mean_f1_macro": mean_f1,
                    "std_f1_macro": std_f1,
                    "num_subjects": n,
                }
                all_accs.append(mean_acc)
                all_f1s.append(mean_f1)
                total_subjects = max(total_subjects, n)

        if not all_accs:
            logger.warning("No valid model results in aggregate_results")
            return None

        # If single model, use its metrics directly
        # If multiple models, use best accuracy model
        if len(per_model) == 1:
            metrics = list(per_model.values())[0]
        else:
            best_model = max(per_model, key=lambda m: per_model[m]["mean_accuracy"])
            metrics = per_model[best_model]
            metrics["best_model"] = best_model
            metrics["all_models"] = per_model

        logger.info(
            "Parsed results: acc=%.4f±%.4f, f1=%.4f±%.4f (n=%d)",
            metrics["mean_accuracy"],
            metrics["std_accuracy"],
            metrics["mean_f1_macro"],
            metrics["std_f1_macro"],
            metrics["num_subjects"],
        )
        return metrics

    @staticmethod
    def _reduce_subjects(source: str, num_subjects: int) -> Optional[str]:
        """Replace ALL_SUBJECTS list in source code with first N subjects.

        Returns modified source or None if ALL_SUBJECTS not found.
        """
        # Match ALL_SUBJECTS = [ ... ] across multiple lines
        pattern = re.compile(
            r"(ALL_SUBJECTS\s*=\s*\[)(.*?)(\])",
            re.DOTALL,
        )
        match = pattern.search(source)
        if not match:
            return None

        # Extract individual subject strings
        subjects_str = match.group(2)
        subjects = re.findall(r'"(DB2_s\d+)"', subjects_str)
        if not subjects:
            subjects = re.findall(r"'(DB2_s\d+)'", subjects_str)

        if len(subjects) <= num_subjects:
            return source  # already small enough

        # Take first N subjects
        reduced = subjects[:num_subjects]
        new_list = ", ".join(f'"{s}"' for s in reduced)
        new_subjects = f"ALL_SUBJECTS = [{new_list}]"

        # Replace in source
        return pattern.sub(new_subjects, source, count=1)
