"""
Model configuration loader from model.yaml.
Provides model fallback chains and API keys for Gemini and Groq.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _find_model_yaml() -> Path:
    """Locate model.yaml relative to project root (next to pyproject.toml)."""
    candidates = [
        Path(__file__).resolve().parents[2] / "model.yaml",
        Path(__file__).resolve().parents[1] / "model.yaml",
        Path.cwd() / "model.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "model.yaml not found. Searched: " + ", ".join(str(c) for c in candidates)
    )


def _load_yaml() -> dict[str, Any]:
    """Load model.yaml as dict."""
    import yaml

    path = _find_model_yaml()
    with open(path) as f:
        return yaml.safe_load(f) or {}


_cached_config: dict[str, Any] | None = None


def get_model_config() -> dict[str, Any]:
    """Load and cache model.yaml config."""
    global _cached_config
    if _cached_config is None:
        _cached_config = _load_yaml()
    return _cached_config


def _get_api_key(locations: list[dict] | None) -> str | None:
    """Resolve API key from config locations. Supports env: and root: keys."""
    if not locations:
        return None
    for loc in locations:
        if isinstance(loc, str):
            return os.getenv(loc)
        if isinstance(loc, dict):
            if "env" in loc:
                return os.getenv(loc["env"])
            if "root" in loc:
                return os.getenv(loc["root"])
    return None


def get_gemini_api_key() -> str | None:
    """Primary Gemini API key from model.yaml locations."""
    cfg = get_model_config()
    locs = cfg.get("gemini", {}).get("api_key_locations")
    for loc in locs or []:
        key = _env_key_from_loc(loc)
        if key and (val := os.getenv(key)):
            return val
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _env_key_from_loc(loc: dict | str) -> str | None:
    """Extract env var name from a location spec (root/env/section)."""
    if isinstance(loc, str):
        return loc
    if isinstance(loc, dict):
        return loc.get("env") or loc.get("root")  # section is for file config, skip
    return None


def get_gemini_fallback_key() -> str | None:
    """Fallback Gemini API key from model.yaml locations."""
    cfg = get_model_config()
    locs = cfg.get("gemini", {}).get("fallback_key_locations")
    for loc in locs or []:
        env_key = loc.get("env") if isinstance(loc, dict) else loc
        if env_key:
            val = os.getenv(env_key)
            if val:
                return val
        root_key = loc.get("root") if isinstance(loc, dict) else None
        if root_key:
            val = os.getenv(root_key)
            if val:
                return val
    return os.getenv("GOOGLE_API_KEY_FALLBACK") or os.getenv("GEMINI_API_KEY_FALLBACK")


def get_groq_api_key() -> str | None:
    """Groq API key from model.yaml locations."""
    cfg = get_model_config()
    locs = cfg.get("groq", {}).get("api_key_locations")
    for loc in locs or []:
        env_key = loc.get("env") if isinstance(loc, dict) else loc
        if env_key:
            val = os.getenv(env_key)
            if val:
                return val
        root_key = loc.get("root") if isinstance(loc, dict) else None
        if root_key:
            val = os.getenv(root_key)
            if val:
                return val
    return os.getenv("GROQ_API_KEY")


def get_gemini_models() -> list[str]:
    """Gemini model fallback chain from model.yaml."""
    cfg = get_model_config()
    chain = cfg.get("gemini", {}).get("fallback_chain")
    if isinstance(chain, list):
        return list(chain)
    return ["models/gemini-2.5-flash-lite", "models/gemini-2.5-flash", "models/gemini-2.5-pro"]


def get_groq_models() -> list[str]:
    """Groq model fallback chain from model.yaml."""
    cfg = get_model_config()
    chain = cfg.get("groq", {}).get("fallback_chain")
    if isinstance(chain, list):
        return list(chain)
    return ["llama-3.1-8b-instant"]


def get_gemini_temperature() -> float:
    """Gemini default temperature from model.yaml."""
    cfg = get_model_config()
    return float(cfg.get("gemini", {}).get("default_config", {}).get("temperature", 0.1))


def get_groq_temperature() -> float:
    """Groq default temperature from model.yaml."""
    cfg = get_model_config()
    return float(cfg.get("groq", {}).get("default_config", {}).get("temperature", 0.1))


def get_provider_order() -> list[str]:
    """Provider fallback order: gemini, groq."""
    cfg = get_model_config()
    order = cfg.get("fallback_strategy", {}).get("providers")
    if isinstance(order, list):
        return list(order)
    return ["gemini", "groq"]


def get_prioritize_groq() -> bool:
    """When True, try Groq first and Gemini as fallback (use when Gemini rate limits)."""
    cfg = get_model_config()
    return bool(cfg.get("fallback_strategy", {}).get("prioritize_groq", False))
