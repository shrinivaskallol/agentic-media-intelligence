# Prompt Registry

Prompts are stored in **separate files per node** (industry standard for modularity, composability, and version control).

## Structure

```
prompts/
├── _registry.yaml    # Version, schema, changelog
├── extraction.yaml   # Gatekeeper: router, coreference
├── grader.yaml       # Pre-synthesis quality check
├── rewrite.yaml      # Query refinement for second retrieval
├── synthesis.yaml    # Report generation: personas, closed_world, entity_rules, human_template
├── critique.yaml     # Red Team auditor
├── evaluator.yaml    # RAGAS Judge
└── evaluation.yaml   # Golden dataset Judge (run_evaluation.py)
```

## Usage

```python
from app.prompts import get_prompt, get_system, get_version

# Template with variable substitution
prompt = get_prompt("grader", "main", query="...", context_str="...")

# System prompt (no substitution)
system = get_system("extraction", "router")

# Nested keys
persona = get_system("synthesis.personas", "RESEARCH")
```

## Editing

1. Edit the relevant YAML file (e.g. `grader.yaml`)
2. Bump `version` in that file and in `_registry.yaml`
3. Add a changelog entry in `_registry.yaml`
