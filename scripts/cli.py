"""Terminal runner for quick testing:  python scripts/cli.py "Top 3 sub-categories by sales in 2018" """
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insight_copilot.graph import build_graph, stream_turn  # noqa: E402


def main():
    graph, tid = build_graph(), str(uuid.uuid4())
    questions = sys.argv[1:] or [input("You: ")]
    for q in questions:
        print(f"\n=== {q}")
        shown, answer = 0, ""
        for _node, upd in stream_turn(graph, q, tid):
            trace = upd.get("trace", [])
            for e in trace[shown:]:
                tag = {"plan": "PLAN", "thought": "THINK", "tool": f"TOOL:{e.get('tool')}"}[e["kind"]]
                print(f"[{tag}] {e.get('text') or e.get('args', '')}")
            shown = max(shown, len(trace)) if trace else shown
            answer = upd.get("answer") or answer
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
