"""The eval harness's own grading logic is deterministic, so it gets offline tests too."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import run_evals as R  # noqa: E402

from insight_copilot.data import load_df  # noqa: E402

EV = Path(__file__).resolve().parents[1] / "evals"


def test_eval_files_are_consistent_and_tools_exist():
    from insight_copilot.tools import TOOLS_BY_NAME
    cases = json.loads((EV / "questions.json").read_text())
    routes = json.loads((EV / "expected_routes.json").read_text())
    assert {c["id"] for c in cases} == set(routes)
    assert set(routes.values()) <= {"tools", "direct", "clarify", "unsupported"}
    for c in cases:
        assert set(c["expected_tools"] + c.get("forbidden_tools", [])) <= set(TOOLS_BY_NAME), c["id"]
        assert not set(c["expected_tools"]) & set(c.get("forbidden_tools", [])), c["id"]
        if routes[c["id"]] != "tools":
            assert c["expected_tools"] == [], c["id"]


def test_grade_pass_and_each_failure_mode():
    df = load_df()
    case = {"expected_tools": ["analyze_stats", "create_chart"], "forbidden_tools": ["web_search"], "truth": "west_south_gap"}
    ok = {"route": "tools", "tools": ["analyze_stats", "create_chart"], "answer": "West leads by $321,068"}
    assert R.grade(case, "tools", ok, df)["passed"]
    assert "missing tool" in R.grade(case, "tools", {**ok, "tools": ["analyze_stats"]}, df)["why"]
    assert "forbidden" in R.grade(case, "tools", {**ok, "tools": ok["tools"] + ["web_search"]}, df)["why"]
    assert "route=direct" in R.grade(case, "tools", {**ok, "route": "direct"}, df)["why"]
    assert "ground truth" in R.grade(case, "tools", {**ok, "answer": "West leads by $999"}, df)["why"]


def test_ground_truth_helpers():
    df = load_df()
    assert R.truth_top3_subcat_last_quarter(df) == ["Phones", "Chairs", "Tables"]
    assert R.truth_west_south_gap(df) == ["321,068"]


def test_infra_errors_are_not_scored_as_agent_failures():
    fallback = {"route": "tools", "tools": [], "trace": [{"kind": "plan", "fallback": True}]}
    normal = {"route": "tools", "tools": [], "trace": [{"kind": "plan", "fallback": False}]}
    assert R.is_infra_error(fallback) and not R.is_infra_error(normal)
    assert R.is_infra_error({"route": "ERROR RateLimitError", "tools": [], "trace": []})


def test_executor_llm_failure_is_infra_error_not_agent_fail():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("run_evals", pathlib.Path(__file__).parents[1] / "evals" / "run_evals.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    res = {"route": "tools", "trace": [{"kind": "plan"}, {"kind": "thought", "text": "x", "llm_error": "RateLimitError: 429"}]}
    assert m.is_infra_error(res)
    assert not m.is_infra_error({"route": "tools", "trace": [{"kind": "plan"}, {"kind": "thought", "text": "ok"}]})
