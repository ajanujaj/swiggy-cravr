"""Render the compiled LangGraph workflow to Mermaid.

Generated from `graph.py` itself, so it can never drift from the real wiring
the way a hand-drawn diagram would.

    python scripts/render_graph.py                 # print to stdout
    python scripts/render_graph.py -o docs/graph.mermaid
    python scripts/render_graph.py --png docs/graph.png

The conditional-edge labels ("feedback", "new_request") come from
the path-map keys in `build_graph()`, which is why those keys are named for
the reason a branch is taken rather than for its destination node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cravr.graph import app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, help="write Mermaid source here")
    parser.add_argument("--png", type=Path, help="also render a PNG (needs network access)")
    args = parser.parse_args()

    graph = app.get_graph()
    mermaid = graph.draw_mermaid()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(mermaid, encoding="utf-8")
        print(f"wrote {args.out} ({len(mermaid.splitlines())} lines)")
    else:
        print(mermaid)

    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        args.png.write_bytes(graph.draw_mermaid_png())
        print(f"wrote {args.png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
