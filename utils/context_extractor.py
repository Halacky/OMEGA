"""
Context extraction system for LLM-assisted development.
Provides compact, relevant code context without loading entire codebase.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import re
import ast


class ContextExtractor:
    """Extract relevant code context for LLM consumption"""

    def __init__(self, project_root: Path, logger: Optional[logging.Logger] = None):
        self.project_root = Path(project_root)
        self.logger = logger or logging.getLogger(__name__)
        self.index = self._build_index()

    def _build_index(self) -> Dict:
        """Build searchable index of project components"""
        index = {
            'models': {},
            'configs': {},
            'trainers': {},
            'datasets': {},
            'functions': {},
            'classes': {}
        }

        # Index model files
        models_dir = self.project_root / 'models'
        if models_dir.exists():
            for model_file in models_dir.glob('*.py'):
                if model_file.name.startswith('_'):
                    continue
                model_info = self._extract_model_info(model_file)
                index['models'][model_file.stem] = model_info

        # Index config files
        config_dir = self.project_root / 'config'
        if config_dir.exists():
            for config_file in config_dir.glob('*.py'):
                if config_file.name.startswith('_'):
                    continue
                config_info = self._extract_config_info(config_file)
                index['configs'][config_file.stem] = config_info

        return index

    def _extract_model_info(self, file_path: Path) -> Dict:
        """Extract model class signatures and docstrings"""
        info = {
            'file': str(file_path),
            'classes': []
        }

        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'methods': [],
                        'init_params': []
                    }

                    # Extract __init__ signature
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name == '__init__':
                                class_info['init_params'] = [
                                    arg.arg for arg in item.args.args if arg.arg != 'self'
                                ]
                            elif item.name == 'forward':
                                class_info['methods'].append({
                                    'name': 'forward',
                                    'docstring': ast.get_docstring(item)
                                })

                    info['classes'].append(class_info)

        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {e}")

        return info

    def _extract_config_info(self, file_path: Path) -> Dict:
        """Extract config dataclass fields"""
        info = {
            'file': str(file_path),
            'configs': []
        }

        try:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a dataclass
                    is_dataclass = any(
                        isinstance(dec, ast.Name) and dec.id == 'dataclass'
                        for dec in node.decorator_list
                    )

                    if is_dataclass:
                        config_info = {
                            'name': node.name,
                            'docstring': ast.get_docstring(node),
                            'fields': []
                        }

                        # Extract annotated fields
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign):
                                field_name = item.target.id if isinstance(item.target, ast.Name) else None
                                if field_name:
                                    field_info = {'name': field_name}

                                    # Try to get default value
                                    if item.value:
                                        try:
                                            field_info['default'] = ast.literal_eval(item.value)
                                        except:
                                            field_info['default'] = '<complex>'

                                    config_info['fields'].append(field_info)

                        info['configs'].append(config_info)

        except Exception as e:
            self.logger.warning(f"Failed to parse config {file_path}: {e}")

        return info

    def get_model_context(self, model_name: str) -> str:
        """Get compact context for a specific model"""
        if model_name not in self.index['models']:
            return f"Model '{model_name}' not found in index."

        model_info = self.index['models'][model_name]
        context = []

        context.append(f"# Model: {model_name}")
        context.append(f"File: {model_info['file']}\n")

        for cls in model_info['classes']:
            context.append(f"## Class: {cls['name']}")
            if cls['docstring']:
                context.append(f"Description: {cls['docstring']}\n")

            context.append(f"### __init__ parameters:")
            for param in cls['init_params']:
                context.append(f"  - {param}")

            if cls['methods']:
                context.append(f"\n### Methods:")
                for method in cls['methods']:
                    context.append(f"  - {method['name']}")
                    if method['docstring']:
                        context.append(f"    {method['docstring']}")

            context.append("")

        return "\n".join(context)

    def get_config_context(self, config_name: str = "base") -> str:
        """Get compact context for config classes"""
        if config_name not in self.index['configs']:
            return f"Config '{config_name}' not found in index."

        config_info = self.index['configs'][config_name]
        context = []

        context.append(f"# Config: {config_name}")
        context.append(f"File: {config_info['file']}\n")

        for cfg in config_info['configs']:
            context.append(f"## {cfg['name']}")
            if cfg['docstring']:
                context.append(f"Description: {cfg['docstring']}\n")

            context.append("### Fields:")
            for field in cfg['fields']:
                default = f" = {field.get('default', 'required')}"
                context.append(f"  - {field['name']}{default}")

            context.append("")

        return "\n".join(context)

    def get_integration_summary(self) -> str:
        """Get complete integration summary for LLM context"""
        summary = []

        summary.append("=" * 80)
        summary.append("OMEGA PROJECT INTEGRATION SUMMARY")
        summary.append("=" * 80)
        summary.append("")

        # Models summary
        summary.append("## Available Models")
        summary.append("")
        for model_name, model_info in self.index['models'].items():
            for cls in model_info['classes']:
                params = ', '.join(cls['init_params'])
                summary.append(f"- **{cls['name']}** ({model_name}.py)")
                summary.append(f"  Parameters: {params}")
                summary.append("")

        # Configs summary
        summary.append("## Configuration Classes")
        summary.append("")
        for config_name, config_info in self.index['configs'].items():
            for cfg in config_info['configs']:
                summary.append(f"- **{cfg['name']}** ({config_name}.py)")
                key_fields = [f['name'] for f in cfg['fields'][:5]]
                summary.append(f"  Key fields: {', '.join(key_fields)}")
                summary.append("")

        return "\n".join(summary)

    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """Search index by keyword"""
        results = []
        keyword_lower = keyword.lower()

        # Search in models
        for model_name, model_info in self.index['models'].items():
            if keyword_lower in model_name.lower():
                results.append({
                    'type': 'model',
                    'name': model_name,
                    'context': self.get_model_context(model_name)
                })

        # Search in model classes
        for model_name, model_info in self.index['models'].items():
            for cls in model_info['classes']:
                if keyword_lower in cls['name'].lower():
                    results.append({
                        'type': 'model_class',
                        'name': f"{model_name}.{cls['name']}",
                        'context': self.get_model_context(model_name)
                    })

        return results

    def export_for_llm(self, output_file: Path):
        """Export complete context in LLM-friendly format"""
        context = {
            'integration_guide': str(self.project_root / 'docs' / 'INTEGRATION_GUIDE.md'),
            'models': {},
            'configs': {},
            'quick_reference': self.get_integration_summary()
        }

        # Add model contexts
        for model_name in self.index['models']:
            context['models'][model_name] = self.get_model_context(model_name)

        # Add config contexts
        for config_name in self.index['configs']:
            context['configs'][config_name] = self.get_config_context(config_name)

        with open(output_file, 'w') as f:
            json.dump(context, f, indent=2)

        self.logger.info(f"Exported LLM context to {output_file}")


def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    project_root = Path(__file__).parent.parent
    extractor = ContextExtractor(project_root, logger)

    # Export complete context
    output_file = project_root / 'docs' / 'llm_context.json'
    extractor.export_for_llm(output_file)

    # Print integration summary
    print(extractor.get_integration_summary())

    # Search example
    results = extractor.search_by_keyword("cnn")
    print(f"\nFound {len(results)} results for 'cnn':")
    for result in results[:3]:
        print(f"- {result['name']} ({result['type']})")


if __name__ == "__main__":
    main()
