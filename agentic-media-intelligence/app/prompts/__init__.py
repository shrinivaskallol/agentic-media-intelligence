"""
Prompt registry loader. Loads versioned prompts from prompts/*.yaml (one file per node).
Industry standard: separate files per task for modularity, composability, and version control.
Use get_prompt() for templates with variable substitution.
"""

from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parents[1].parent / "prompts"
_REGISTRY_PATH = _PROMPTS_DIR / "_registry.yaml"
_registry_cache: dict | None = None
_section_cache: dict[str, dict] = {}


def _load_registry() -> dict:
    global _registry_cache
    if _registry_cache is None:
        with open(_REGISTRY_PATH) as f:
            _registry_cache = yaml.safe_load(f)
    return _registry_cache


def _load_section(top: str) -> dict:
    """Load prompts from YAML file (e.g. extraction -> extraction.yaml). Cached by file."""
    if top in _section_cache:
        return _section_cache[top]
    path = _PROMPTS_DIR / f"{top}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    _section_cache[top] = data
    return data


def _resolve(section: str, key: str) -> dict:
    """Resolve section.key to prompt definition dict."""
    parts = section.split(".")
    top = parts[0]
    data = _load_section(top)
    obj = data
    for p in parts[1:]:
        obj = obj.get(p, {})
    prompt_def = obj.get(key) if isinstance(obj, dict) else None
    if prompt_def is None:
        raise KeyError(f"Prompt not found: {section}.{key}")
    return prompt_def


def get_prompt(section: str, key: str, **kwargs) -> str:
    """
    Get a prompt by section.key and substitute template variables.
    Example: get_prompt("grader", "main", query="...", context_str="...")
    For nested keys: get_prompt("synthesis.personas", "RESEARCH")
    """
    prompt_def = _resolve(section, key)
    template = prompt_def.get("template") or prompt_def.get("system", "")
    template = template.strip()
    if kwargs:
        template = template.format(**kwargs)
    return template


def get_system(section: str, key: str) -> str:
    """Get a system prompt (no substitution)."""
    prompt_def = _resolve(section, key)
    return (prompt_def.get("system") or prompt_def.get("template", "")).strip()


def get_version() -> str:
    """Return registry version."""
    return _load_registry().get("version", "0.0.0")
