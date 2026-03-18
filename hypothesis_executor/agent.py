"""Main Hypothesis Executor Agent — orchestrates code generation from hypotheses.

Generated experiment scripts contain built-in Qdrant callbacks that update
hypothesis status (verified/failed) when the experiment is run manually.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .config import ExecutorConfig
from .glm_client import GLMClient
from .context_builder import CodebaseContextBuilder
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .code_extractor import CodeExtractor
from .code_validator import CodeValidator

logger = logging.getLogger("hypothesis_executor.agent")


class HypothesisExecutorAgent:
    """Takes an unverified hypothesis and generates validated experiment code via GLM-5.

    Generated scripts include Qdrant callbacks — when run manually, they
    automatically mark the hypothesis as verified/failed with metrics.
    """

    def __init__(self, config: Optional[ExecutorConfig] = None):
        self.config = config or ExecutorConfig()
        self.root = self.config.project_root
        self.glm = GLMClient(self.config)
        self.context_builder = CodebaseContextBuilder(self.root)
        self.validator = CodeValidator(self.root)

    # ── Public API ──────────────────────────────────────────────────

    def execute(self, hypothesis: dict) -> dict:
        """Generate experiment code from a hypothesis dict.

        Args:
            hypothesis: Dict matching the Hypothesis pydantic schema
                (id, title, hypothesis_text, proposed_changes, etc.)

        Returns:
            {
                "success": bool,
                "experiment_path": str | None,
                "model_path": str | None,
                "experiment_name": str,
                "experiment_code": str | None,
                "model_code": str | None,
                "validation_errors": list[str],
                "attempt": int,
            }
        """
        logger.info("Executing hypothesis: %s", hypothesis.get("title", "?"))

        # Step 1: Build codebase context and codegen rules
        codebase_context = self.context_builder.build_context()
        codegen_rules = self.context_builder.build_codegen_rules()

        # Step 2: Determine next experiment number
        next_num = self._get_next_experiment_number()
        logger.info("Next experiment number: %d", next_num)

        # Step 3: Analyze if new model is needed
        proposed = hypothesis.get("proposed_changes", {})
        if isinstance(proposed, dict):
            model_type = proposed.get("model_type", "")
        else:
            model_type = getattr(proposed, "model_type", "")

        needs_new_model = self._needs_new_model(model_type)
        extra_instructions = ""
        if needs_new_model:
            extra_instructions = (
                "IMPORTANT: This hypothesis requires a NEW model architecture "
                f"'{model_type}' that is NOT in VALID_MODEL_TYPES. "
                "You MUST generate BOTH a model file (```python model```) "
                "AND an experiment file (```python experiment```)."
            )
            logger.info("New model required: %s", model_type)

        # Step 4: Format prompts
        system_prompt = SYSTEM_PROMPT.format(
            codebase_context=codebase_context,
            codegen_rules=codegen_rules,
        )
        user_prompt = self._format_user_prompt(
            hypothesis, next_num, extra_instructions
        )

        # Step 5: Generate → extract → validate loop
        all_errors: list[str] = []
        for attempt in range(1, self.config.max_generation_attempts + 1):
            logger.info("Generation attempt %d/%d", attempt, self.config.max_generation_attempts)

            # Call GLM-5
            try:
                raw_response = self.glm.generate(system_prompt, user_prompt)
            except Exception as e:
                logger.error("GLM-5 API call failed: %s", e)
                return {
                    "success": False,
                    "experiment_path": None,
                    "model_path": None,
                    "experiment_name": "",
                    "experiment_code": None,
                    "model_code": None,
                    "validation_errors": [f"API error: {e}"],
                    "attempt": attempt,
                }

            # Extract code blocks
            try:
                extracted = CodeExtractor.extract(raw_response)
            except ValueError as e:
                logger.warning("Extraction failed: %s", e)
                user_prompt += (
                    f"\n\nPREVIOUS ATTEMPT FAILED — could not extract code: {e}\n"
                    "Please ensure you wrap code in properly labeled fences."
                )
                all_errors = [str(e)]
                continue

            # Validate experiment
            exp_errors = self.validator.validate_experiment(
                extracted["experiment_code"]
            )

            # Validate model if present
            model_errors: list[str] = []
            if extracted.get("model_code"):
                model_errors = self.validator.validate_model(
                    extracted["model_code"]
                )

            all_errors = exp_errors + model_errors

            if not all_errors:
                # Success — save files
                logger.info("Validation passed on attempt %d", attempt)
                return self._save_results(extracted, hypothesis, next_num, attempt)

            # Feed errors back for re-generation
            error_text = "\n".join(f"- {e}" for e in all_errors)
            logger.warning(
                "Validation failed (attempt %d): %s", attempt, error_text
            )
            user_prompt += (
                f"\n\nPREVIOUS ATTEMPT HAD VALIDATION ERRORS:\n{error_text}\n\n"
                "Fix ALL errors listed above and regenerate the complete code."
            )

        # All attempts exhausted
        logger.error(
            "Code generation failed after %d attempts. Errors: %s",
            self.config.max_generation_attempts,
            all_errors,
        )
        return {
            "success": False,
            "experiment_path": None,
            "model_path": None,
            "experiment_name": "",
            "experiment_code": None,
            "model_code": None,
            "validation_errors": all_errors,
            "attempt": self.config.max_generation_attempts,
        }

    def execute_from_json(self, json_path: str) -> dict:
        """Load a hypothesis from a JSON file and generate code."""
        logger.info("Loading hypothesis from: %s", json_path)
        with open(json_path, encoding="utf-8") as f:
            hypothesis = json.load(f)
        return self.execute(hypothesis)

    def execute_from_vector_store(self) -> dict:
        """Load the first unverified hypothesis from Qdrant and generate code."""
        hypothesis = self._fetch_unverified(limit=1)
        if hypothesis is None:
            return {"success": False, "validation_errors": ["No unverified hypotheses or Qdrant unavailable"]}
        return self.execute(hypothesis)

    # ── Batch mode: generate code for all unverified hypotheses ────

    def generate_all_unverified(self) -> dict:
        """Generate experiment scripts for ALL unverified hypotheses from Qdrant.

        Each generated script contains built-in Qdrant callbacks — when run
        manually, it will automatically mark the hypothesis as verified/failed.

        Returns:
            {
                "total": int,
                "succeeded": int,
                "failed": int,
                "results": list[dict],
            }
        """
        all_hypotheses = self._fetch_all_unverified()
        if not all_hypotheses:
            logger.info("No unverified hypotheses found in vector store")
            return {"total": 0, "succeeded": 0, "failed": 0, "results": []}

        total = len(all_hypotheses)
        logger.info("Found %d unverified hypotheses. Generating scripts...", total)

        succeeded = 0
        failed = 0
        all_results = []

        for i, hypothesis in enumerate(all_hypotheses, 1):
            title = hypothesis.get("title", "?")
            hyp_id = hypothesis.get("id", "")

            separator = "=" * 60
            logger.info(
                "\n%s\n[%d/%d] Generating code for: %s (id=%s)\n%s",
                separator, i, total, title, hyp_id[:8] if hyp_id else "?", separator,
            )

            try:
                result = self.execute(hypothesis)
                result["hypothesis_title"] = title
                result["hypothesis_id"] = hyp_id
                result["batch_index"] = i
                all_results.append(result)

                if result["success"]:
                    succeeded += 1
                    logger.info(
                        "[%d/%d] CODE GENERATED: %s -> %s",
                        i, total, title, result.get("experiment_path", "?"),
                    )
                else:
                    failed += 1
                    logger.warning(
                        "[%d/%d] GENERATION FAILED: %s — %s",
                        i, total, title, result.get("validation_errors", []),
                    )

            except Exception as e:
                logger.error("[%d/%d] UNEXPECTED ERROR for '%s': %s", i, total, title, e)
                failed += 1
                all_results.append({
                    "hypothesis_title": title,
                    "hypothesis_id": hyp_id,
                    "batch_index": i,
                    "success": False,
                    "validation_errors": [str(e)],
                })

        logger.info(
            "\nBatch code generation complete: %d total, %d succeeded, %d failed",
            total, succeeded, failed,
        )

        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "results": all_results,
        }

    # ── Qdrant helpers ──────────────────────────────────────────────

    def _fetch_unverified(self, limit: int = 1) -> Optional[dict]:
        """Fetch one unverified hypothesis from Qdrant. Returns payload dict or None."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError:
            logger.error("qdrant-client not installed. Use: pip install qdrant-client")
            return None

        qdrant_path = self.root / "research_agent" / "qdrant_data"
        if not qdrant_path.exists():
            logger.error("Qdrant data not found at: %s", qdrant_path)
            return None

        client = QdrantClient(path=str(qdrant_path))
        try:
            results = client.scroll(
                collection_name="hypotheses",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="status",
                            match=MatchValue(value="unverified"),
                        )
                    ]
                ),
                limit=limit,
            )
        except Exception as e:
            logger.error("Qdrant query failed: %s", e)
            return None

        points = results[0] if results else []
        if not points:
            logger.info("No unverified hypotheses found")
            return None

        hypothesis = points[0].payload
        logger.info("Found unverified hypothesis: %s", hypothesis.get("title", "?"))
        return hypothesis

    def _fetch_all_unverified(self) -> list[dict]:
        """Fetch all unverified hypotheses from Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError:
            logger.error("qdrant-client not installed")
            return []

        qdrant_path = self.root / "research_agent" / "qdrant_data"
        if not qdrant_path.exists():
            logger.error("Qdrant data not found at: %s", qdrant_path)
            return []

        client = QdrantClient(path=str(qdrant_path))

        try:
            all_points = []
            offset = None
            while True:
                results, next_offset = client.scroll(
                    collection_name="hypotheses",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="status",
                                match=MatchValue(value="unverified"),
                            )
                        ]
                    ),
                    limit=50,
                    offset=offset,
                )
                all_points.extend(results)
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.error("Qdrant query failed: %s", e)
            return []

        return [p.payload for p in all_points]

    # ── Internal helpers ────────────────────────────────────────────

    def _needs_new_model(self, model_type: str) -> bool:
        """Check if the model_type requires creating a new model file."""
        try:
            from research_agent.knowledge.codebase_registry import (
                VALID_MODEL_TYPES,
                VALID_ML_MODELS,
            )
            all_valid = set(m.lower() for m in VALID_MODEL_TYPES + VALID_ML_MODELS)
        except ImportError:
            all_valid = {
                "simple_cnn", "attention_cnn", "tcn", "tcn_attn",
                "multiscale_cnn", "bilstm", "bilstm_attention", "bigru",
                "cnn_lstm", "cnn_gru_attention", "resnet1d",
                "svm_rbf", "svm_linear", "rf",
            }
        return model_type.lower() not in all_valid

    def _get_next_experiment_number(self) -> int:
        """Scan experiments/ directory for exp_N_*.py and return max N + 1."""
        exp_dir = self.root / self.config.experiments_dir
        max_n = 0
        for f in exp_dir.glob("exp_*_*.py"):
            match = re.match(r"exp_(\d+)_", f.name)
            if match:
                max_n = max(max_n, int(match.group(1)))
        return max_n + 1

    def _format_user_prompt(
        self, h: dict, next_num: int, extra_instructions: str
    ) -> str:
        """Format the user prompt template with hypothesis data."""
        proposed = h.get("proposed_changes", {})
        if isinstance(proposed, dict):
            model_type = proposed.get("model_type", "")
            features = proposed.get("features", "")
            augmentation = proposed.get("augmentation", "")
            training_modifications = proposed.get("training_modifications", "")
        else:
            model_type = getattr(proposed, "model_type", "")
            features = getattr(proposed, "features", "")
            augmentation = getattr(proposed, "augmentation", "")
            training_modifications = getattr(proposed, "training_modifications", "")

        motivation_text = ""
        motivation = h.get("motivation", {})
        if isinstance(motivation, dict):
            for exp_ref in motivation.get("based_on_experiments", []):
                if isinstance(exp_ref, dict):
                    motivation_text += (
                        f"- Experiment: {exp_ref.get('experiment_name', '')} "
                        f"— {exp_ref.get('observation', '')}\n"
                    )
            for paper_ref in motivation.get("based_on_papers", []):
                if isinstance(paper_ref, dict):
                    motivation_text += (
                        f"- Paper: {paper_ref.get('paper_title', '')} "
                        f"({paper_ref.get('arxiv_id', '')}) "
                        f"— {paper_ref.get('insight_used', '')}\n"
                    )

        sanitized = re.sub(r"[^a-z0-9_]", "_", h.get("title", "unnamed").lower())
        sanitized = re.sub(r"_+", "_", sanitized)[:50].rstrip("_")

        return USER_PROMPT_TEMPLATE.format(
            title=h.get("title", ""),
            id=h.get("id", ""),
            status=h.get("status", "unverified"),
            strategy=h.get("strategy", ""),
            hypothesis_text=h.get("hypothesis_text", ""),
            model_type=model_type,
            features=features,
            augmentation=augmentation,
            training_modifications=training_modifications,
            expected_effect=h.get("expected_effect", ""),
            novelty_explanation=h.get("novelty_explanation", ""),
            motivation_text=motivation_text or "None provided",
            next_exp_number=next_num,
            sanitized_title=sanitized,
            extra_instructions=extra_instructions,
        )

    def _save_results(
        self, extracted: dict, hypothesis: dict, exp_num: int, attempt: int
    ) -> dict:
        """Save generated experiment and model files to the project."""
        exp_dir = self.root / self.config.experiments_dir
        model_dir = self.root / self.config.models_dir

        exp_filename = extracted.get("experiment_filename")
        if not exp_filename:
            sanitized = re.sub(
                r"[^a-z0-9_]", "_",
                hypothesis.get("title", "unnamed").lower()
            )
            sanitized = re.sub(r"_+", "_", sanitized)[:40].rstrip("_")
            exp_filename = f"exp_{exp_num}_{sanitized}_loso.py"

        exp_path = exp_dir / exp_filename
        exp_path.write_text(extracted["experiment_code"], encoding="utf-8")
        logger.info("Saved experiment: %s", exp_path)

        model_path = None
        if extracted.get("model_code"):
            model_filename = extracted.get("model_filename")
            if not model_filename:
                proposed = hypothesis.get("proposed_changes", {})
                mt = proposed.get("model_type", "custom_model") if isinstance(proposed, dict) else "custom_model"
                model_filename = f"{mt.lower()}.py"

            model_path_obj = model_dir / model_filename
            model_path_obj.write_text(
                extracted["model_code"], encoding="utf-8"
            )
            model_path = str(model_path_obj)
            logger.info("Saved model: %s", model_path)

        experiment_name = exp_filename.replace(".py", "")
        logger.info("Generation complete: %s (attempt %d)", experiment_name, attempt)

        return {
            "success": True,
            "experiment_path": str(exp_path),
            "model_path": model_path,
            "experiment_name": experiment_name,
            "experiment_code": extracted["experiment_code"],
            "model_code": extracted.get("model_code"),
            "validation_errors": [],
            "attempt": attempt,
        }
