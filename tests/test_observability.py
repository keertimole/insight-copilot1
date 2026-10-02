"""Offline tests for per-turn observability (pure Python, no LLM)."""
import json
import os
import tempfile
from pathlib import Path

from insight_copilot.observability import aggregate, log_turn, summarize_turn

TRACE = [
    {"kind": "plan", "route": "tools", "tools": ["analyze_stats", "create_chart"], "fallback": False},
    {"kind": "tool", "tool": "analyze_stats", "args": {}, "output": "{}", "chart": False, "error": False},
    {"kind": "tool", "tool": "create_chart", "args": {}, "output": "{}", "chart": True, "error": False},
    {"kind": "revise", "text": "Revising"},
    {"kind": "critic", "passed": True, "skipped": False, "attempt": 2, "issues": [],
     "checks": [{"label": "ok", "ok": True}, {"label": "chart mentioned", "ok": False}]},
]


def test_summarize_turn_captures_route_tools_and_check():
    r = summarize_turn("Compare West and South", TRACE, "answer text", 3.456, "gemini-3.6-flash", "google")
    assert r["route"] == "tools" and r["tools_run"] == ["analyze_stats", "create_chart"]
    assert r["planned_tools"] == ["analyze_stats", "create_chart"] and r["charts"] == 1 and r["revisions"] == 1
    assert r["self_check"] == "passed" and r["checks_failed"] == ["chart mentioned"]
    assert r["latency_s"] == 3.46 and r["model"] == "google/gemini-3.6-flash" and r["error"] is None


def test_summarize_turn_handles_empty_trace_and_errors():
    r = summarize_turn("hi", [], "", 0.2, error="RateLimitError")
    assert r["self_check"] == "n/a" and r["route"] is None and r["tools_run"] == [] and r["error"] == "RateLimitError"


def test_log_turn_writes_jsonl_and_can_be_disabled():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "turns.jsonl"
        os.environ["TURN_LOG"] = str(path)
        try:
            log_turn(summarize_turn("q1", TRACE, "a", 1.0), "abcdef123456")
            log_turn(summarize_turn("q2", [], "a", 2.0), "abcdef123456")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert [r["question"] for r in rows] == ["q1", "q2"] and rows[0]["thread"] == "abcdef12"
            os.environ["TURN_LOG"] = "off"
            log_turn(summarize_turn("q3", [], "a", 1.0))
            assert len(path.read_text(encoding="utf-8").splitlines()) == 2
        finally:
            os.environ.pop("TURN_LOG", None)


def test_log_turn_never_raises_on_bad_path():
    with tempfile.TemporaryDirectory() as d:
        blocker = Path(d) / "iam_a_file"
        blocker.write_text("x")
        os.environ["TURN_LOG"] = str(blocker / "child" / "turns.jsonl")     # a file cannot be a directory
        try:
            log_turn(summarize_turn("q", [], "a", 1.0))                      # must not raise
        finally:
            os.environ.pop("TURN_LOG", None)


def test_aggregate_metrics():
    recs = [summarize_turn("a", TRACE, "x", 2.0), summarize_turn("b", [], "x", 4.0, error="Boom"),
            summarize_turn("c", [{"kind": "plan", "route": "unsupported", "tools": []}], "x", 6.0)]
    m = aggregate(recs)
    assert m["turns"] == 3 and m["avg_latency_s"] == 4.0 and m["errors"] == 1 and m["revisions"] == 1
    assert m["tool_calls"] == {"analyze_stats": 1, "create_chart": 1} and m["routes"]["unsupported"] == 1
    assert m["self_check_pass_rate"] == 100 and aggregate([]) == {"turns": 0}
