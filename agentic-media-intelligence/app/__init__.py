# Agentic Media Intelligence app package

import logging


def configure_logging(
    level: int = logging.INFO,
    format: str = "%(levelname)s %(name)s: %(message)s",
) -> None:
    """Configure root logger for the application. Call from scripts/entry points."""
    logging.basicConfig(level=level, format=format)
