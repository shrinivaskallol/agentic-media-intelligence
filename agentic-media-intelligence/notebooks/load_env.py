"""Load .env from project root. Run at start of notebooks that need DB credentials."""
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


def load_env() -> None:
    """Load .env from project root. Call at start of notebooks that need DB credentials."""
    load_dotenv(_ENV_PATH)


def get_project_root() -> Path:
    """Return project root path for resolving data/scripts paths."""
    return _PROJECT_ROOT
