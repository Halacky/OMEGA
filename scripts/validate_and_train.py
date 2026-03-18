#!/usr/bin/env python3
"""
Integrated script: validate model before training.
This prevents wasting time on incompatible models.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.model_validator import validate_model_before_training
from utils.context_extractor import ContextExtractor


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        print("Usage: python validate_and_train.py <model_file.py>")
        print("\nExample:")
        print("  python validate_and_train.py models/new_model.py")
        print("\nOr run full context extraction:")
        print("  python validate_and_train.py --extract-context")
        sys.exit(1)

    if sys.argv[1] == "--extract-context":
        logger.info("Extracting LLM context...")
        extractor = ContextExtractor(project_root, logger)
        output_file = project_root / 'docs' / 'llm_context.json'
        extractor.export_for_llm(output_file)

        # Also print summary
        print("\n" + extractor.get_integration_summary())

        logger.info(f"✅ Context exported to {output_file}")
        logger.info("You can now use this file as context for LLM code generation!")
        sys.exit(0)

    model_file = Path(sys.argv[1])

    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        sys.exit(1)

    logger.info(f"Validating model: {model_file}")
    logger.info("=" * 80)

    is_valid = validate_model_before_training(model_file, logger)

    logger.info("=" * 80)

    if is_valid:
        logger.info("✅ Model validation PASSED")
        logger.info("\nYou can now proceed with training:")
        logger.info(f"  1. Add model to models/__init__.py")
        logger.info(f"  2. Register in trainer._create_model()")
        logger.info(f"  3. Run training script")
        sys.exit(0)
    else:
        logger.error("❌ Model validation FAILED")
        logger.error("\nPlease fix the errors above before training.")
        logger.error("See docs/INTEGRATION_GUIDE.md for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
