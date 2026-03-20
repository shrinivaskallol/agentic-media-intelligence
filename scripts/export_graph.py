#!/usr/bin/env python3
"""
Export the entity workflow LangGraph as a PNG image for architecture documentation.
Saves to docs/architecture_graph.png.
"""

import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from app import configure_logging
from app.graph.entity_workflow import build_workflow

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    app = build_workflow()
    runnable = app.get_graph()

    out_path = _root / "docs" / "architecture_graph.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    png_bytes = runnable.draw_mermaid_png(output_file_path=str(out_path))
    if png_bytes and not out_path.exists():
        with open(out_path, "wb") as f:
            f.write(png_bytes)

    logger.info("Exported graph to %s", out_path)


if __name__ == "__main__":
    main()
