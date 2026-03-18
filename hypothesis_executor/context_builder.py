"""Builds codebase context for the GLM-5 prompt from key project files."""

import re
import logging
from pathlib import Path

logger = logging.getLogger("hypothesis_executor.context_builder")

# Files to include in full
FULL_FILES = [
    "experiments/exp_X_template_loso.py",
    "config/base.py",
    "models/__init__.py",
    "research_agent/knowledge/codebase_registry.py",
]

# Files to include as summaries (partial reads)
SUMMARY_FILES = [
    ("training/trainer.py", "_create_model_factory"),
    ("models/cnn1d.py", "full"),
    ("experiments/exp_6_sota_cnn_bigru_aug_loso.py", "main_only"),
]

# Codegen rules file — read separately and injected into the system prompt
# as a dedicated section so the LLM treats them as mandatory constraints.
CODEGEN_RULES_FILE = "hypothesis_executor/CODEGEN_RULES.md"


class CodebaseContextBuilder:
    """Reads key project files and assembles them into a single context string."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self._cache: str | None = None
        self._rules_cache: str | None = None

    def build_context(self) -> str:
        """Build and cache the codebase context string."""
        if self._cache is not None:
            return self._cache

        sections: list[str] = []

        # Full files
        for rel_path in FULL_FILES:
            content = self._read_file(rel_path)
            if content:
                sections.append(f"=== FILE: {rel_path} ===\n{content}\n===\n")

        # Summaries
        for rel_path, mode in SUMMARY_FILES:
            content = self._read_summary(rel_path, mode)
            if content:
                sections.append(f"=== FILE: {rel_path} (summary) ===\n{content}\n===\n")

        raw = "\n".join(sections)
        # Escape curly braces so str.format() passes them through literally —
        # codebase files contain Python dicts, f-strings, etc.
        self._cache = raw.replace("{", "{{").replace("}", "}}")
        logger.info(
            "Codebase context built: %d chars, %d sections",
            len(self._cache),
            len(sections),
        )
        return self._cache

    def build_codegen_rules(self) -> str:
        """Read and cache the codegen rules from CODEGEN_RULES.md.

        Curly braces in the file are escaped so that str.format() does not
        treat them as placeholders (the rules contain Python code snippets
        with dicts and f-strings).

        Returns the rules text, or an empty string if the file is missing.
        """
        if self._rules_cache is not None:
            return self._rules_cache

        content = self._read_file(CODEGEN_RULES_FILE)
        if content:
            # Escape curly braces so str.format() passes them through literally
            self._rules_cache = content.replace("{", "{{").replace("}", "}}")
            logger.info(
                "Codegen rules loaded: %d chars from %s",
                len(content),
                CODEGEN_RULES_FILE,
            )
        else:
            self._rules_cache = ""
            logger.warning("Codegen rules file not found: %s", CODEGEN_RULES_FILE)

        return self._rules_cache

    def _read_file(self, rel_path: str) -> str | None:
        """Read a file fully."""
        path = self.root / rel_path
        if not path.exists():
            logger.warning("File not found: %s", path)
            return None
        return path.read_text(encoding="utf-8")

    def _read_summary(self, rel_path: str, mode: str) -> str | None:
        """Read a file partially according to the mode."""
        full = self._read_file(rel_path)
        if full is None:
            return None

        if mode == "full":
            return full

        if mode == "_create_model_factory":
            return self._extract_create_model(full)

        if mode == "main_only":
            return self._extract_main_function(full)

        return full

    @staticmethod
    def _extract_create_model(source: str) -> str:
        """Extract the _create_model method from trainer.py source."""
        lines = source.splitlines()
        start = None
        end = None
        indent = None
        for i, line in enumerate(lines):
            if "def _create_model(" in line:
                start = i
                indent = len(line) - len(line.lstrip())
                continue
            if start is not None and i > start:
                stripped = line.lstrip()
                if stripped and not stripped.startswith("#"):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent and stripped.startswith("def "):
                        end = i
                        break
        if start is None:
            return source[:3000]
        return "\n".join(lines[start : end or (start + 150)])

    @staticmethod
    def _extract_main_function(source: str) -> str:
        """Extract the main() function from an experiment file."""
        match = re.search(r"^def main\(\):", source, re.MULTILINE)
        if match:
            return source[match.start():]
        return source
