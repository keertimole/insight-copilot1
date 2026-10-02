"""Offline graph tests: a scripted fake LLM drives the REAL graph, tools and data (no API key needed).

They verify wiring and control flow - routing, tool execution, the critic -> revise -> critic loop and its budget -
not model quality (that is what evals/run_evals.py measures against a live model).
"""
import pytest
from langchain_core.messages import AIMessage

from insight_copilot import graph as G
from insight_copilot.graph import Plan, Verdict, build_graph, stream_turn

GOOD = ("**Answer:** West outsold South by $321,068.23 (82.5% more).\n**Supporting numbers:**\n"
        "- West: $710,219.68\n- South: $389,151.46\n- Furniture explains 40.1% of the gap\n"
        "**Why it matters:** The gap is spread across all categories, but Furniture alone is 40.1% of it. Chart shown below.")
BAD = GOOD.replace("$710,219.68", "$999,999.99")


class FakeLLM:
    def __init__(self, script):
        self.s, self.schema, self.tools = script, None, False

    def with_structured_output(self, schema):
        c = FakeLLM(self.s); c.schema = schema; return c

    def bind_tools(self, tools):
        c = FakeLLM(self.s); c.tools = True; return c

    def invoke(self, messages):
        if self.schema is Plan:
            return self.s["plan"]
        if self.schema is Verdict:
            return self.s.get("verdict", Verdict(answers_question=True, takeaway_is_analytical=True))
        if self.tools:                                            # executor
            if any(m.__class__.__name__ == "ToolMessage" for m in messages):
                return AIMessage(content="DONE")
            return AIMessage(content="Comparing the two regions.", tool_calls=[
                {"name": "analyze_stats", "id": "c1", "args": {"analysis": "compare_groups", "group_by": "region", "groups": "West,South"}}])
        if "REVISING" in messages[0].content:
            self.s["revise_calls"] += 1
            return AIMessage(content=GOOD)
        return AIMessage(content=self.s["first_draft"])


def run(monkeypatch, question, script):
    script.setdefault("revise_calls", 0)
    llm = FakeLLM(script)
    monkeypatch.setattr(G, "get_llm", lambda temperature=0.0: llm)
    state = {}
    for _node, upd in stream_turn(build_graph(), question, "t"):
        state.update(upd)
        state.setdefault("_nodes", []).append(_node)
    return state


def tools_plan():
    return Plan(goal="Compare West and South", reasoning="Need a regional comparison.", route="tools",
                tools=["analyze_stats"], steps=["analyze_stats compare_groups West vs South"])


def test_happy_path_critic_passes_first_time(monkeypatch):
    st = run(monkeypatch, "Compare West and South", {"plan": tools_plan(), "first_draft": GOOD})
    assert st["_nodes"] == ["planner", "executor", "tool_node", "executor", "synthesizer", "critic"]
    assert st["critique"]["passed"] and st["answer"] == GOOD
    assert [e["kind"] for e in st["trace"]][-1] == "critic"


def test_critic_catches_fabricated_number_and_revises(monkeypatch):
    script = {"plan": tools_plan(), "first_draft": BAD}
    st = run(monkeypatch, "Compare West and South", script)
    assert st["_nodes"][-3:] == ["critic", "revise", "critic"]
    assert script["revise_calls"] == 1
    assert "999,999.99" not in st["answer"] and st["answer"] == GOOD
    kinds = [e["kind"] for e in st["trace"]]
    assert kinds.count("critic") == 2 and "revise" in kinds


def test_revision_budget_is_bounded_and_caveat_added(monkeypatch):
    class Stubborn(FakeLLM):
        def invoke(self, messages):
            if not (self.schema or self.tools) and "REVISING" in messages[0].content:
                return AIMessage(content=BAD)
            return super().invoke(messages)
    script = {"plan": tools_plan(), "first_draft": BAD, "revise_calls": 0}
    monkeypatch.setattr(G, "get_llm", lambda temperature=0.0: Stubborn(script))
    final = {}
    for _n, upd in stream_turn(build_graph(), "Compare West and South", "t2"):
        final.update(upd)
    assert final["revisions"] == G.MAX_REVISIONS
    assert "Could not verify" in final["answer"] and "$999,999.99" in final["answer"]


def test_unsupported_route_skips_tools_and_critic(monkeypatch):
    plan = Plan(goal="profit margin", reasoning="No profit data.", route="unsupported", unsupported_reason="no profit column")
    st = run(monkeypatch, "What was our profit margin?", {"plan": plan, "first_draft": ""})
    assert st["_nodes"] == ["planner", "unsupported"]


def test_multi_turn_history_accumulates(monkeypatch):
    llm = FakeLLM({"plan": tools_plan(), "first_draft": GOOD, "revise_calls": 0})
    monkeypatch.setattr(G, "get_llm", lambda temperature=0.0: llm)
    g = build_graph()
    for q in ("Compare West and South", "and why?"):
        list(stream_turn(g, q, "same-thread"))
    msgs = g.get_state({"configurable": {"thread_id": "same-thread"}}).values["messages"]
    assert [m.__class__.__name__ for m in msgs] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]


@pytest.mark.parametrize("answer,ok", [(GOOD, True), (BAD, False)])
def test_critic_unit(answer, ok):
    import json
    from insight_copilot import analytics as A
    from insight_copilot.critic import check_answer
    from insight_copilot.data import load_df
    out = A.analyze(load_df(), "compare_groups", "region", "West,South", "category", "none")
    obs = [{"tool": "analyze_stats", "args": {}, "output": json.dumps(out)}]
    assert check_answer(answer, obs, 1).passed is ok


def test_llm_outage_is_friendly_and_selfcheck_is_marked_skipped(monkeypatch):
    class RateLimitError(Exception):
        pass

    class Down(FakeLLM):
        def invoke(self, messages):
            if not (self.schema or self.tools):
                raise RateLimitError("Error code: 429 - org_SECRET tokens per day")
            return super().invoke(messages)
    script = {"plan": tools_plan(), "first_draft": GOOD, "revise_calls": 0}
    monkeypatch.setattr(G, "get_llm", lambda temperature=0.0: Down(script))
    st = {}
    for _n, upd in stream_turn(build_graph(), "Compare West and South", "t3"):
        st.update(upd)
    assert "rate limit" in st["answer"].lower() and "org_SECRET" not in st["answer"]
    crit = [e for e in st["trace"] if e["kind"] == "critic"][-1]
    assert crit["skipped"] is True and crit["passed"] is False


def test_critic_rejects_unsupported_recommendation():
    from insight_copilot.critic import check_answer
    ans = ("**Answer:** West leads.\n**Supporting numbers:**\n- Furniture gap: $128,817\n"
           "**Why it matters:** Furniture drives the gap, so boosting Furniture in the South could close most of it.")
    res = check_answer(ans, [{"tool": "x", "args": {}, "output": "128816.77"}])
    assert not res.passed and any("recommendation" in i for i in res.issues)


def test_llm_fallback_used_when_primary_fails():
    from insight_copilot.config import WithFallback

    class Boom:
        def invoke(self, m): raise RuntimeError("429 rate limit")
        def bind_tools(self, t): return self
        def with_structured_output(self, s): return self

    class Ok:
        def invoke(self, m): return "from-backup"
        def bind_tools(self, t): return self
        def with_structured_output(self, s): return self

    assert WithFallback(Boom(), Ok()).bind_tools([]).with_structured_output(dict).invoke("x") == "from-backup"
    with pytest.raises(RuntimeError):
        WithFallback(Boom(), None).invoke("x")


def test_fallback_spec_defaults(monkeypatch):
    from insight_copilot import config as C
    for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACK_MODEL", "LLM_FALLBACK_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    assert C.fallback_spec() == ("groq", C.DEFAULT_GROQ_FALLBACK)
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "none")
    assert C.fallback_spec() is None


def test_fallback_failure_reports_primary_error_and_disables_missing_backup(monkeypatch):
    from insight_copilot import config as C
    monkeypatch.setattr(C, "RATE_LIMIT_WAIT_S", 0.0)

    class Primary:
        calls = 0
        def invoke(self, m):
            Primary.calls += 1
            raise RuntimeError("Error code: 429 - rate limit reached")
        def bind_tools(self, t): return self
        def with_structured_output(self, s): return self

    class Missing:
        calls = 0
        def invoke(self, m):
            Missing.calls += 1
            raise RuntimeError("Error code: 404 - The model `x` does not exist")
        def bind_tools(self, t): return self
        def with_structured_output(self, s): return self

    llm = C.WithFallback(Primary(), Missing())
    with pytest.raises(RuntimeError, match="429"):          # the real cause, not the backup's 404
        llm.invoke("x")
    assert Missing.calls == 1 and llm.state["backup_dead"]
    with pytest.raises(RuntimeError, match="429"):
        llm.invoke("x")
    assert Missing.calls == 1                                # dead backup is not retried


def test_fallback_retries_primary_once_after_rate_limit(monkeypatch):
    from insight_copilot import config as C
    monkeypatch.setattr(C, "RATE_LIMIT_WAIT_S", 0.0)

    class Flaky:
        n = 0
        def invoke(self, m):
            Flaky.n += 1
            if Flaky.n == 1:
                raise RuntimeError("429 rate limit")
            return "primary-ok"
        def bind_tools(self, t): return self
        def with_structured_output(self, s): return self

    class AlsoLimited:
        def invoke(self, m): raise RuntimeError("429 rate limit")
        def bind_tools(self, t): return self
        def with_structured_output(self, s): return self

    assert C.WithFallback(Flaky(), AlsoLimited()).invoke("x") == "primary-ok"


def test_chitchat_detection_for_planner_fallback():
    from insight_copilot.graph import _looks_like_chitchat
    for q in ("Hi! What can you do?", "hello", "Hey there", "thanks!", "Who are you?", "what can you do"):
        assert _looks_like_chitchat(q), q
    for q in ("Show me monthly sales", "Compare West and South", "What was sales in Hillsdale?",
              "Hi, show me sales by region"):
        assert not _looks_like_chitchat(q), q
