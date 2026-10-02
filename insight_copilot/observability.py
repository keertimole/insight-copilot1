"""Per-turn observability: one structured record per question, an optional JSONL log, and session aggregates.

Pure Python (no LangChain / Streamlit imports) so it is unit-testable. The record is built from the agent's
own `trace`, so it needs no extra instrumentation inside the graph.

  - `summarize_turn()`  -> dict (route, tools planned/run, tool errors, revisions, self-check result, latency, model)
  - `log_turn()`        -> appends the dict to a JSONL file (default logs/turns.jsonl; TURN_LOG=off disables) and to
                           the standard `logging` logger. Never raises: logging must not break the chat.
  - `aggregate()`       -> metrics shown in the sidebar (avg/p95 latency, tool usage, self-check pass rate, errors)

For full LLM-level tracing (prompts, tokens, latency per node) set LANGSMITH_TRACING=true and LANGSMITH_API_KEY:
LangGraph/LangChain pick these up automatically - no code change needed.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

log = logging.getLogger("insight_copilot")


def summarize_turn(question: str, trace: list, answer: str, latency_s: float, model: str = "", provider: str = "",
                   error: str | None = None) -> dict:
    plan = next((e for e in trace if e.get("kind") == "plan"), {})
    calls = [e for e in trace if e.get("kind") == "tool"]
    crit = [e for e in trace if e.get("kind") == "critic"]
    last = crit[-1] if crit else {}
    if not crit:
        check = "n/a"
    elif last.get("skipped"):
        check = "skipped"
    else:
        check = "passed" if last.get("passed") else "failed"
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "question": (question or "")[:300],
        "route": plan.get("route"),
        "planner_fallback": bool(plan.get("fallback")),
        "planned_tools": list(plan.get("tools", [])),
        "tools_run": [e.get("tool") for e in calls],
        "tool_errors": sum(1 for e in calls if e.get("error")),
        "charts": sum(1 for e in calls if e.get("chart")),
        "revisions": sum(1 for e in trace if e.get("kind") == "revise"),
        "self_check": check,
        "checks_failed": [c.get("label", "") for c in last.get("checks", []) if not c.get("ok")],
        "latency_s": round(float(latency_s), 2),
        "answer_chars": len(answer or ""),
        "model": f"{provider}/{model}".strip("/"),
        "error": error,
    }


def log_turn(record: dict, thread_id: str = "") -> None:
    """Append the record to the JSONL log and the logger. Silent on any failure."""
    rec = {**record, "thread": (thread_id or "")[:8]}
    try:
        log.info("turn route=%s tools=%s check=%s latency=%ss error=%s", rec.get("route"), rec.get("tools_run"),
                 rec.get("self_check"), rec.get("latency_s"), rec.get("error"))
        path = os.getenv("TURN_LOG", "logs/turns.jsonl").strip()
        if path.lower() in ("", "0", "off", "none", "false", "no"):
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - observability must never break the app
        pass


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    if not n:
        return {"turns": 0}
    lat = sorted(r.get("latency_s", 0.0) for r in records)
    checked = [r for r in records if r.get("self_check") in ("passed", "failed")]
    return {
        "turns": n,
        "avg_latency_s": round(sum(lat) / n, 1),
        "p95_latency_s": round(lat[min(n - 1, round(0.95 * (n - 1)))], 1),
        "routes": dict(Counter(r.get("route") or "n/a" for r in records)),
        "tool_calls": dict(Counter(t for r in records for t in r.get("tools_run", []))),
        "self_check_pass_rate": round(100 * sum(r["self_check"] == "passed" for r in checked) / len(checked)) if checked else None,
        "revisions": sum(r.get("revisions", 0) for r in records),
        "tool_errors": sum(r.get("tool_errors", 0) for r in records),
        "errors": sum(1 for r in records if r.get("error")),
        "planner_fallbacks": sum(1 for r in records if r.get("planner_fallback")),
    }
