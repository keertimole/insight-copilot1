"""Render the compiled LangGraph as Mermaid so the diagram always matches the code:
    python scripts/export_graph.py > docs/graph.mmd
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insight_copilot.graph import build_graph  # noqa: E402

print(build_graph().get_graph().draw_mermaid())
