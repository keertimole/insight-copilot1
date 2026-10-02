"""Agent evaluation harness: does the agent make the RIGHT DECISIONS?  (unit tests cover the maths; this covers routing.)

For every labelled question it runs the real LangGraph agent and checks
    question -> actual route -> actual tool calls -> expected route / tools -> PASS / FAIL
plus, where a `truth` key is given, that the final answer contains a figure computed independently with pandas.

    python evals/run_evals.py                    # full agent run (needs an LLM key); ~15 questions
    python evals/run_evals.py --route-only       # planner only: cheap, checks routing + planned tools
    python evals/run_evals.py --ids q04 q08 -v   # a subset, with the tool trace and answer
    python evals/run_evals.py --delay 3          # sleep between questions (free-tier rate limits)
    python evals/run_evals.py --out evals/last_run.json --threshold 0.8      # non-zero exit if pass-rate < 80%

Files:  questions.json (question, expected_tools = ALL must be called, forbidden_tools = NONE may be called,
        optional `setup` turns and `truth`)  ·  expected_routes.json (id -> tools | direct | clarify | unsupported)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from insight_copilot.data import load_df  # noqa: E402

HERE = Path(__file__).parent


# ------------------------------------------------------------------ ground truth computed with plain pandas
def _last_quarter(df: pd.DataFrame) -> pd.DataFrame:
    q = df["Order Date"].max().to_period("Q")
    return df[df["Order Date"].dt.to_period("Q") == q]


def truth_top3_subcat_last_quarter(df) -> list[str]:
    return list(_last_quarter(df).groupby("Sub-Category")["Sales"].sum().nlargest(3).index)


def truth_west_south_gap(df) -> list[str]:
    s = df.groupby("Region")["Sales"].sum()
    return [f"{s['West'] - s['South']:,.0f}"]              # e.g. "321,068" (matches "$321,068" and "$321,068.23")


TRUTH = {"top3_subcat_last_quarter": truth_top3_subcat_last_quarter, "west_south_gap": truth_west_south_gap}


# ------------------------------------------------------------------ running one case
def run_case(graph, case: dict, route_only: bool) -> dict:
    from langchain_core.messages import HumanMessage
    from insight_copilot.graph import planner, stream_turn

    if route_only:
        hist = [HumanMessage(content=q) for q in case.get("setup", [])] + [HumanMessage(content=case["question"])]
        out = planner({"messages": hist})
        return {"route": out["route"], "tools": out["plan"]["tools"], "answer": "", "critic_passed": None, "trace": out["trace"]}

    tid, state = str(uuid.uuid4()), {}
    for q in case.get("setup", []):                                   # earlier turns on the same thread
        for _ in stream_turn(graph, q, tid):
            pass
    for _node, upd in stream_turn(graph, case["question"], tid):
        state.update(upd)
    tools = [e["tool"] for e in state.get("trace", []) if e["kind"] == "tool"]
    critics = [e for e in state.get("trace", []) if e["kind"] == "critic"]
    return {"route": state.get("route", "?"), "tools": tools, "answer": state.get("answer", ""),
            "critic_passed": (critics[-1]["passed"] if critics else None), "revisions": state.get("revisions", 0),
            "trace": state.get("trace", [])}


def is_infra_error(res: dict) -> bool:
    """True when the run tells us nothing about the agent: the planner/LLM call itself failed (rate limit, bad key...)."""
    trace = res.get("trace", [])
    return (str(res["route"]).startswith("ERROR")
            or any(e.get("fallback") for e in trace if e["kind"] == "plan")
            or any(e.get("llm_error") for e in trace if e["kind"] == "thought"))


def grade(case: dict, expected_route: str, res: dict, df) -> dict:
    called = set(res["tools"])
    missing = [t for t in case["expected_tools"] if t not in called]
    forbidden = [t for t in case.get("forbidden_tools", []) if t in called]
    route_ok = res["route"] == expected_route
    tools_ok = not missing and not forbidden
    truth_ok, truth_note = None, ""
    if case.get("truth") and res["answer"]:
        needles = TRUTH[case["truth"]](df)
        absent = [n for n in needles if n.lower() not in res["answer"].lower()]
        truth_ok, truth_note = not absent, ("missing " + ", ".join(absent)) if absent else ""
    passed = route_ok and tools_ok and truth_ok is not False
    why = []
    if not route_ok:
        why.append(f"route={res['route']} (want {expected_route})")
    if missing:
        why.append("missing tool(s): " + ", ".join(missing))
    if forbidden:
        why.append("forbidden tool(s) used: " + ", ".join(forbidden))
    if truth_ok is False:
        why.append("ground truth " + truth_note)
    return {"passed": passed, "route_ok": route_ok, "tools_ok": tools_ok, "truth_ok": truth_ok, "why": "; ".join(why)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route-only", action="store_true", help="run only the planner (no tools / no answer)")
    ap.add_argument("--ids", nargs="*", help="run only these question ids")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between questions")
    ap.add_argument("--threshold", type=float, default=0.0, help="exit 1 if overall pass-rate is below this (0-1)")
    ap.add_argument("--out", help="write full results as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    from insight_copilot.config import has_api_key, model_name, provider
    if not has_api_key():
        print("No LLM API key found (set GROQ_API_KEY or your provider's key in .env). The eval needs a live model.")
        return 2

    cases = json.loads((HERE / "questions.json").read_text())
    routes = json.loads((HERE / "expected_routes.json").read_text())
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    df = load_df()
    graph = None
    if not args.route_only:
        from insight_copilot.graph import build_graph
        graph = build_graph()

    print(f"Model: {provider()}/{model_name()} · mode: {'route-only' if args.route_only else 'full agent'} · {len(cases)} questions\n")
    rows = []
    for i, case in enumerate(cases):
        t0 = time.time()
        try:
            res = run_case(graph, case, args.route_only)
        except Exception as e:  # noqa: BLE001 - one flaky call must not abort the whole run
            res = {"route": f"ERROR {type(e).__name__}", "tools": [], "answer": "", "critic_passed": None, "trace": []}
        if is_infra_error(res):
            g = {"passed": False, "route_ok": False, "tools_ok": False, "truth_ok": None, "error": True,
                 "why": "LLM call failed (rate limit / key / model name?) - not an agent decision, excluded from scores"}
        else:
            g = grade(case, routes[case["id"]], res, df)
        rows.append({"id": case["id"], "question": case["question"], "expected_route": routes[case["id"]],
                     "expected_tools": case["expected_tools"], **{k: v for k, v in res.items() if k != "trace"}, **g,
                     "seconds": round(time.time() - t0, 1)})
        tools = ", ".join(res["tools"]) or "-"
        print(f"{'ERR ' if g.get('error') else 'PASS' if g['passed'] else 'FAIL'}  {case['id']}  [{res['route']:<11}] tools: {tools:<38} "
              f"{case['question'][:60]}" + (f"\n        ↳ {g['why']}" if g["why"] else ""))
        if args.verbose:
            for e in res["trace"]:
                if e["kind"] in ("thought", "critic", "revise"):
                    print(f"        · {e['kind']}: {e['text'][:140]}")
            if res["answer"]:
                print("        answer: " + res["answer"].replace("\n", "\n                ")[:700])
        if args.delay and i < len(cases) - 1:
            time.sleep(args.delay)

    errors = [r for r in rows if r.get("error")]
    rows_all, rows = rows, [r for r in rows if not r.get("error")]
    if errors:
        print(f"\n⚠️  {len(errors)} of {len(rows_all)} questions hit an LLM error (rate limit / key) and are NOT scored.")
    if not rows:
        print("No question reached the model successfully - nothing to score. Wait for the quota to reset or change "
              "LLM_MODEL / LLM_PROVIDER in .env, then re-run.")
        return 2
    n = len(rows)
    pct = lambda k: 100 * sum(bool(r[k]) for r in rows) / n  # noqa: E731
    checked = [r for r in rows if r["truth_ok"] is not None]
    critic = [r for r in rows if r.get("critic_passed") is not None]
    print(f"\n{'=' * 78}\nOverall PASS  {pct('passed'):5.1f}%  ({sum(r['passed'] for r in rows)}/{n})")
    print(f"Route accuracy {pct('route_ok'):5.1f}%   Tool-selection accuracy {pct('tools_ok'):5.1f}%")
    if checked:
        print(f"Ground-truth figures present in answer: {sum(r['truth_ok'] for r in checked)}/{len(checked)}")
    if critic:
        print(f"Critic passed on: {sum(r['critic_passed'] for r in critic)}/{len(critic)} tool-answers "
              f"(revisions used: {sum(r.get('revisions', 0) for r in critic)})")
    if args.out:
        Path(args.out).write_text(json.dumps({"model": f"{provider()}/{model_name()}", "results": rows_all}, indent=2))
        print(f"Wrote {args.out}")
    return 0 if pct("passed") / 100 >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
