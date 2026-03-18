"""Validates generated experiment and model code for correctness."""

import ast
import re
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("hypothesis_executor.code_validator")


class CodeValidator:
    """Static analysis validator for generated experiment and model code."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self._training_config_fields: set[str] | None = None

    def validate_experiment(self, code: str) -> List[str]:
        """Validate an experiment script. Returns list of error messages."""
        errors: List[str] = []
        errors.extend(self._check_syntax(code))
        if errors:
            return errors  # no point checking further if syntax is broken
        errors.extend(self._check_required_pattern(code))
        errors.extend(self._check_no_hardcoded_paths(code))
        errors.extend(self._check_training_config_fields(code))
        errors.extend(self._check_model_type_validity(code))
        return errors

    def validate_model(self, code: str) -> List[str]:
        """Validate a model file. Returns list of error messages."""
        errors: List[str] = []
        errors.extend(self._check_syntax(code))
        if errors:
            return errors
        errors.extend(self._check_model_pattern(code))
        return errors

    # ── Syntax ──────────────────────────────────────────────────────

    @staticmethod
    def _check_syntax(code: str) -> List[str]:
        try:
            compile(code, "<generated>", "exec")
            return []
        except SyntaxError as e:
            return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    # ── Required experiment pattern ─────────────────────────────────

    @staticmethod
    def _check_required_pattern(code: str) -> List[str]:
        errors: List[str] = []
        checks = {
            "ROOT path setup": (
                r"ROOT\s*=\s*Path\(",
                "Missing ROOT = Path(__file__).resolve().parents[1]",
            ),
            "sys.path insert": (
                r"sys\.path\.insert\(0",
                "Missing sys.path.insert(0, str(ROOT))",
            ),
            "EXPERIMENT_NAME": (
                r"EXPERIMENT_NAME\s*=",
                "Missing EXPERIMENT_NAME variable assignment",
            ),
            "main() function": (
                r"def main\(\)",
                "Missing def main() function",
            ),
            "__name__ guard": (
                r'if\s+__name__\s*==\s*["\']__main__["\']',
                "Missing if __name__ == '__main__' guard",
            ),
            "HYPOTHESIS_ID": (
                r"HYPOTHESIS_ID\s*=",
                "Missing HYPOTHESIS_ID variable (needed for Qdrant status update)",
            ),
            "qdrant_callback import": (
                r"from hypothesis_executor\.qdrant_callback import",
                "Missing import of qdrant_callback (mark_hypothesis_verified/mark_hypothesis_failed)",
            ),
            "mark_hypothesis call": (
                r"mark_hypothesis_(verified|failed)\(",
                "Missing call to mark_hypothesis_verified() or mark_hypothesis_failed() at end of main()",
            ),
            "ProcessingConfig": (
                r"ProcessingConfig\(",
                "Missing ProcessingConfig instantiation",
            ),
            "TrainingConfig": (
                r"TrainingConfig\(",
                "Missing TrainingConfig instantiation",
            ),
            "loso_summary save": (
                r"loso_summary",
                "Missing loso_summary results saving",
            ),
        }
        for name, (pattern, msg) in checks.items():
            if not re.search(pattern, code):
                errors.append(msg)
        return errors

    # ── No hardcoded paths ──────────────────────────────────────────

    @staticmethod
    def _check_no_hardcoded_paths(code: str) -> List[str]:
        errors: List[str] = []
        # Check that BASE_DIR uses ROOT, not absolute path
        base_dir_match = re.search(r"BASE_DIR\s*=\s*(.+)", code)
        if base_dir_match:
            value = base_dir_match.group(1).strip()
            if value.startswith('Path("/') or value.startswith("Path('/"):
                errors.append(
                    f"BASE_DIR uses hardcoded absolute path: {value}. "
                    "Must use ROOT / 'data'"
                )
        return errors

    # ── TrainingConfig field validation ─────────────────────────────

    def _check_training_config_fields(self, code: str) -> List[str]:
        valid_fields = self._get_training_config_fields()
        if not valid_fields:
            return []  # can't validate without knowing valid fields

        errors: List[str] = []
        # Find TrainingConfig(...) calls and extract keyword args
        pattern = re.compile(r"TrainingConfig\((.*?)\)", re.DOTALL)
        for match in pattern.finditer(code):
            kwargs_str = match.group(1)
            kwarg_names = re.findall(r"(\w+)\s*=", kwargs_str)
            for kw in kwarg_names:
                if kw not in valid_fields:
                    errors.append(
                        f"Unknown TrainingConfig field: '{kw}'. "
                        f"Not found in config/base.py dataclass."
                    )
        return errors

    def _get_training_config_fields(self) -> set[str]:
        """Parse config/base.py to extract TrainingConfig field names."""
        if self._training_config_fields is not None:
            return self._training_config_fields

        config_path = self.root / "config" / "base.py"
        if not config_path.exists():
            logger.warning("config/base.py not found, skipping field validation")
            self._training_config_fields = set()
            return self._training_config_fields

        source = config_path.read_text(encoding="utf-8")
        fields: set[str] = set()

        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name == "TrainingConfig"
                ):
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            fields.add(item.target.id)
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    fields.add(target.id)
        except SyntaxError:
            logger.warning("Failed to parse config/base.py")

        self._training_config_fields = fields
        logger.info("TrainingConfig fields: %s", fields)
        return fields

    # ── Model type validation ───────────────────────────────────────

    @staticmethod
    def _check_model_type_validity(code: str) -> List[str]:
        """Check that model_type strings used are valid or register_model is called."""
        from research_agent.knowledge.codebase_registry import (
            VALID_MODEL_TYPES,
            VALID_ML_MODELS,
        )

        errors: List[str] = []
        all_valid = set(m.lower() for m in VALID_MODEL_TYPES + VALID_ML_MODELS)

        # Check MODEL_TYPES list
        mt_match = re.search(
            r"MODEL_TYPES\s*=\s*\[(.*?)\]", code, re.DOTALL
        )
        if mt_match:
            model_names = re.findall(r'"(\w+)"', mt_match.group(1))
            model_names += re.findall(r"'(\w+)'", mt_match.group(1))
            for name in model_names:
                if name.lower() not in all_valid:
                    # Check if register_model is used for this name
                    if f'register_model("{name}"' not in code and f"register_model('{name}'" not in code:
                        errors.append(
                            f"Model type '{name}' is not in VALID_MODEL_TYPES "
                            f"and register_model() is not called for it."
                        )
        return errors

    # ── Model file pattern ──────────────────────────────────────────

    @staticmethod
    def _check_model_pattern(code: str) -> List[str]:
        errors: List[str] = []
        if "nn.Module" not in code:
            errors.append("Model class does not inherit from nn.Module")
        if "def __init__(" not in code:
            errors.append("Model class missing __init__ method")
        if "def forward(" not in code:
            errors.append("Model class missing forward method")

        # Check constructor signature includes in_channels and num_classes
        init_match = re.search(r"def __init__\(self,\s*(.*?)\):", code, re.DOTALL)
        if init_match:
            params = init_match.group(1)
            if "in_channels" not in params:
                errors.append(
                    "Model __init__ missing 'in_channels' parameter"
                )
            if "num_classes" not in params:
                errors.append(
                    "Model __init__ missing 'num_classes' parameter"
                )
        return errors
