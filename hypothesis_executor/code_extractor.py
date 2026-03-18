"""Extracts Python code blocks from GLM-5 LLM response text."""

import re
import logging
from typing import Optional

logger = logging.getLogger("hypothesis_executor.code_extractor")


class CodeExtractor:
    """Parses LLM response to extract model and experiment code blocks."""

    # Labeled block patterns
    _RE_MODEL = re.compile(
        r"```python\s+model\s*\n(.*?)```", re.DOTALL
    )
    _RE_EXPERIMENT = re.compile(
        r"```python\s+experiment\s*\n(.*?)```", re.DOTALL
    )
    # Generic python block fallback
    _RE_GENERIC = re.compile(
        r"```python\s*\n(.*?)```", re.DOTALL
    )
    # FILE comment pattern
    _RE_FILE_COMMENT = re.compile(
        r"^#\s*FILE:\s*(.+\.py)\s*$", re.MULTILINE
    )

    @classmethod
    def extract(cls, llm_response: str) -> dict:
        """Extract code blocks from the LLM response.

        Returns:
            {
                "model_code": str | None,
                "model_filename": str | None,
                "experiment_code": str,
                "experiment_filename": str | None,
            }

        Raises:
            ValueError: If no experiment code block is found.
        """
        model_code: Optional[str] = None
        model_filename: Optional[str] = None
        experiment_code: Optional[str] = None
        experiment_filename: Optional[str] = None

        # Try labeled blocks first
        model_match = cls._RE_MODEL.search(llm_response)
        exp_match = cls._RE_EXPERIMENT.search(llm_response)

        if model_match:
            model_code = cls._clean(model_match.group(1))
            model_filename = cls._extract_filename(model_code)

        if exp_match:
            experiment_code = cls._clean(exp_match.group(1))
            experiment_filename = cls._extract_filename(experiment_code)

        # Fallback: try generic python blocks if experiment not found
        if experiment_code is None:
            generic_blocks = cls._RE_GENERIC.findall(llm_response)
            for block in generic_blocks:
                block = cls._clean(block)
                if cls._looks_like_experiment(block):
                    experiment_code = block
                    experiment_filename = cls._extract_filename(block)
                elif cls._looks_like_model(block) and model_code is None:
                    model_code = block
                    model_filename = cls._extract_filename(block)

        if experiment_code is None:
            raise ValueError(
                "No experiment code block found in the LLM response. "
                "Expected a ```python experiment``` or ```python``` block "
                "containing run_single_loso_fold or EXPERIMENT_NAME."
            )

        logger.info(
            "Extracted: experiment=%d chars%s, model=%s",
            len(experiment_code),
            f" ({experiment_filename})" if experiment_filename else "",
            f"{len(model_code)} chars ({model_filename})"
            if model_code
            else "None",
        )

        return {
            "model_code": model_code,
            "model_filename": cls._basename(model_filename),
            "experiment_code": experiment_code,
            "experiment_filename": cls._basename(experiment_filename),
        }

    @classmethod
    def _extract_filename(cls, code: str) -> Optional[str]:
        """Extract filename from `# FILE: path/to/file.py` comment."""
        match = cls._RE_FILE_COMMENT.search(code[:300])
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _basename(path: Optional[str]) -> Optional[str]:
        """Extract just the filename from a path like models/foo.py -> foo.py."""
        if path is None:
            return None
        parts = path.replace("\\", "/").split("/")
        return parts[-1]

    @staticmethod
    def _clean(code: str) -> str:
        """Strip whitespace and remove stray fence markers."""
        code = code.strip()
        # Remove any leftover backtick fences
        if code.endswith("```"):
            code = code[:-3].rstrip()
        return code

    @staticmethod
    def _looks_like_experiment(code: str) -> bool:
        indicators = [
            "EXPERIMENT_NAME",
            "run_single_loso_fold",
            "loso_summary",
            "ALL_SUBJECTS",
            "def main(",
        ]
        return sum(1 for ind in indicators if ind in code) >= 2

    @staticmethod
    def _looks_like_model(code: str) -> bool:
        return "nn.Module" in code and "def forward(" in code
