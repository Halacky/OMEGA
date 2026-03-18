"""
Automatic model validator for OMEGA project.
Checks if a new model conforms to the project's integration standards.
"""

import torch
import torch.nn as nn
import inspect
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import importlib.util


class ModelValidator:
    """Validates PyTorch models for integration compatibility"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.errors = []
        self.warnings = []

    def validate_model_class(self, model_class: type) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a model class for compatibility.

        Args:
            model_class: The model class to validate

        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Check 1: Inherits from nn.Module
        if not issubclass(model_class, nn.Module):
            self.errors.append(f"Model {model_class.__name__} must inherit from nn.Module")
            return False, self.errors, self.warnings

        # Check 2: __init__ signature
        self._validate_init_signature(model_class)

        # Check 3: forward method exists
        self._validate_forward_method(model_class)

        # Check 4: Try instantiating with dummy parameters
        self._validate_instantiation(model_class)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _validate_init_signature(self, model_class: type):
        """Check __init__ parameters"""
        try:
            sig = inspect.signature(model_class.__init__)
            params = list(sig.parameters.keys())

            # Remove 'self'
            if 'self' in params:
                params.remove('self')

            # Required parameters
            required_params = {'in_channels', 'num_classes'}

            # Check for in_channels or in_features (both acceptable)
            has_input_param = 'in_channels' in params or 'in_features' in params
            has_num_classes = 'num_classes' in params

            if not has_input_param:
                self.errors.append(
                    f"Model __init__ must have 'in_channels' or 'in_features' parameter. "
                    f"Found: {params}"
                )

            if not has_num_classes:
                self.errors.append(
                    f"Model __init__ must have 'num_classes' parameter. "
                    f"Found: {params}"
                )

            # Recommended parameters
            if 'dropout' not in params:
                self.warnings.append(
                    "Model __init__ should have 'dropout' parameter for regularization"
                )

        except Exception as e:
            self.errors.append(f"Failed to inspect __init__ signature: {e}")

    def _validate_forward_method(self, model_class: type):
        """Check forward method"""
        if not hasattr(model_class, 'forward'):
            self.errors.append("Model must implement forward() method")
            return

        try:
            sig = inspect.signature(model_class.forward)
            params = list(sig.parameters.keys())

            # Should have at least 'self' and 'x' (or similar)
            if len(params) < 2:
                self.warnings.append(
                    f"forward() has unusual signature: {params}. "
                    "Expected at least (self, x)"
                )
        except Exception as e:
            self.warnings.append(f"Could not inspect forward() signature: {e}")

    def _validate_instantiation(self, model_class: type):
        """Try to instantiate model with dummy parameters"""
        try:
            # Try common parameter combinations
            test_configs = [
                {'in_channels': 8, 'num_classes': 10, 'dropout': 0.3},
                {'in_features': 269, 'num_classes': 10, 'dropout': 0.3},
            ]

            instantiated = False
            for config in test_configs:
                try:
                    # Filter parameters that exist in __init__
                    sig = inspect.signature(model_class.__init__)
                    valid_params = {
                        k: v for k, v in config.items()
                        if k in sig.parameters
                    }

                    model = model_class(**valid_params)
                    instantiated = True

                    # Try a forward pass
                    self._validate_forward_pass(model, config)
                    break

                except Exception as e:
                    continue

            if not instantiated:
                self.warnings.append(
                    "Could not instantiate model with standard parameters. "
                    "Model may require special initialization."
                )

        except Exception as e:
            self.warnings.append(f"Instantiation test failed: {e}")

    def _validate_forward_pass(self, model: nn.Module, config: Dict[str, Any]):
        """Test forward pass with dummy data"""
        try:
            model.eval()

            # Try different input shapes
            test_shapes = [
                (2, config.get('in_channels', 8), 100),  # (N, C, T) for CNN
                (2, config.get('in_features', 269)),      # (N, F) for features
            ]

            for shape in test_shapes:
                try:
                    x = torch.randn(*shape)
                    with torch.no_grad():
                        output = model(x)

                    # Check output
                    if isinstance(output, torch.Tensor):
                        # Single output
                        if output.shape[0] != shape[0]:
                            self.warnings.append(
                                f"Output batch size {output.shape[0]} != input batch size {shape[0]}"
                            )
                        if output.shape[1] != config['num_classes']:
                            self.warnings.append(
                                f"Output classes {output.shape[1]} != num_classes {config['num_classes']}"
                            )
                    elif isinstance(output, tuple):
                        # Multiple outputs (e.g., hybrid models)
                        self.warnings.append(
                            f"Model returns tuple output: {len(output)} elements. "
                            "Ensure trainer handles this correctly."
                        )

                    # Success - no need to try other shapes
                    return

                except Exception:
                    continue

            self.warnings.append("Could not complete forward pass with standard input shapes")

        except Exception as e:
            self.warnings.append(f"Forward pass test failed: {e}")

    def validate_model_file(
        self, model_file: Path, target_class: str | None = None
    ) -> Dict[str, Any]:
        """
        Validate model classes in a Python file.

        Args:
            model_file: Path to model .py file
            target_class: If provided, only validate this class (by name).
                Helper nn.Module subclasses (Attention, FeedForward, etc.)
                are skipped — they have different __init__ signatures and
                would produce false-positive validation errors.

        Returns:
            Dictionary with validation results
        """
        results = {
            'file': str(model_file),
            'models': [],
            'overall_valid': True
        }

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location("temp_module", model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find all nn.Module subclasses
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj != nn.Module:
                    # Skip helper classes when target is specified
                    if target_class and name != target_class:
                        continue

                    self.logger.info(f"Validating model class: {name}")

                    is_valid, errors, warnings = self.validate_model_class(obj)

                    results['models'].append({
                        'name': name,
                        'valid': is_valid,
                        'errors': errors,
                        'warnings': warnings
                    })

                    if not is_valid:
                        results['overall_valid'] = False

        except Exception as e:
            results['overall_valid'] = False
            results['error'] = str(e)
            self.logger.error(f"Failed to load model file {model_file}: {e}")

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("=" * 80)
        report.append(f"MODEL VALIDATION REPORT: {results['file']}")
        report.append("=" * 80)

        if 'error' in results:
            report.append(f"\n❌ CRITICAL ERROR: {results['error']}")
            return "\n".join(report)

        if results['overall_valid']:
            report.append("\n✅ ALL MODELS VALID")
        else:
            report.append("\n❌ VALIDATION FAILED")

        for model_info in results['models']:
            report.append(f"\n{'─' * 80}")
            report.append(f"Model: {model_info['name']}")
            report.append(f"Status: {'✅ VALID' if model_info['valid'] else '❌ INVALID'}")

            if model_info['errors']:
                report.append("\nErrors:")
                for error in model_info['errors']:
                    report.append(f"  ❌ {error}")

            if model_info['warnings']:
                report.append("\nWarnings:")
                for warning in model_info['warnings']:
                    report.append(f"  ⚠️  {warning}")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


def validate_model_before_training(model_file: Path, logger: Optional[logging.Logger] = None) -> bool:
    """
    Convenience function to validate model before training.

    Args:
        model_file: Path to model file
        logger: Optional logger

    Returns:
        True if all models in file are valid, False otherwise
    """
    validator = ModelValidator(logger)
    results = validator.validate_model_file(model_file)
    report = validator.generate_report(results)

    if logger:
        logger.info(report)
    else:
        print(report)

    return results['overall_valid']


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test with existing models
    models_dir = Path(__file__).parent.parent / "models"

    test_files = [
        models_dir / "cnn1d.py",
        models_dir / "hybrid_powerful_deep.py",
        models_dir / "tcn.py",
    ]

    for model_file in test_files:
        if model_file.exists():
            logger.info(f"\n{'='*80}\nTesting: {model_file.name}\n{'='*80}")
            validate_model_before_training(model_file, logger)
