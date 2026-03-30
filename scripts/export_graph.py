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

    out_dir = _root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    mermaid_path = out_dir / "architecture_graph.mmd"
    out_path = out_dir / "architecture_graph.png"

    mermaid_src = runnable.draw_mermaid()
    mermaid_path.write_text(mermaid_src, encoding="utf-8")
    logger.info("Wrote Mermaid source to %s (diffable; open in mermaid.live)", mermaid_path)

    if "diversity_gate" not in mermaid_src:
        logger.warning(
            "Mermaid output has no 'diversity_gate' — check app/graph/entity_workflow.py"
        )
    else:
        logger.info("Diagram includes diversity_gate (HITL / MMR gate node)")

    png_bytes = runnable.draw_mermaid_png(output_file_path=str(out_path))
    if png_bytes and not out_path.exists():
        with open(out_path, "wb") as f:
            f.write(png_bytes)

    logger.info("Exported graph PNG to %s", out_path)


if __name__ == "__main__":
    main()
