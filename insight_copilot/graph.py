r"""Insight Copilot agent - a LangGraph StateGraph.

    START -> planner --(route)--> executor <-> tool_node -> synthesizer -> critic --(verdict)--> END      route = "tools"
                     \\-> direct_answer -> END                                  \\-> revise -> critic (loop)
                     \\-> clarify       -> END                                                          route = "clarify"
                     \\-> unsupported   -> END                                                          route = "unsupported"

Conditional edges:
  1. after `planner`  : routes on the plan's decision (needs tools? ambiguous? impossible with this data?)
  2. after `executor` : loops to `tool_node` while the LLM requests tools (bounded by MAX_STEPS), else -> synthesizer
  3. after `critic`   : PASS (or revision budget spent) -> END, FAIL -> `revise` -> `critic` again (bounded by MAX_REVISIONS)

Reasoning is *explicit state*: the planner emits a structured plan, every executor turn records its "thought",
each tool call / observation is appended to `trace`, and the critic's verdict is too. The UI renders all of it.
"""
import json
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .config import get_llm, secret
from .critic import check_answer, forecast_note
from .data import data_card, load_df, using_sample_data
from .tools import ALL_TOOLS, TOOLS_BY_NAME

MAX_STEPS = 6          # max executor turns per question (guards against tool loops)
HISTORY_TURNS = 8      # chat messages shown to the LLM for follow-up resolution
MAX_REVISIONS = 2      # critic -> revise -> critic rounds before we ship the best draft with a caveat


# ------------------------------------------------------------------------------------------ state
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # chat history: human questions + final answers
    plan: dict            # structured plan from the planner (this turn)
    route: str            # tools | direct | clarify | unsupported
    work: list            # executor scratchpad: AI tool-call messages + ToolMessages (this turn)
    observations: list    # [{tool, args, output}] gathered this turn
    trace: list           # visible reasoning trace (this turn)
    charts: list          # plotly JSON strings produced this turn
    steps: int            # executor turns used
    draft: str            # synthesizer / reviser output awaiting the critic (this turn)
    draft_ok: bool        # False when the draft is an LLM-outage message (nothing to verify)
    critique: dict        # latest critic verdict {passed, issues, ungrounded, n_figures, llm_*}
    revisions: int        # critic -> revise rounds used
    answer: str           # final text (this turn)


class Plan(BaseModel):
    goal: str = Field("", description="ONE line: the user's goal in plain words, incl. period/filters.")
    reasoning: str = Field(description="2-4 sentences, first person: what the user wants, what data/tool that needs, why.")
    route: Literal["tools", "direct", "clarify", "unsupported"]
    tools: list[str] = Field(default_factory=list, description="Only the tools actually needed.")
    steps: list[str] = Field(default_factory=list, description="1-4 short imperative steps, naming the tool for each.")
    clarifying_question: str = Field("", description="Only for route=clarify.")
    unsupported_reason: str = Field("", description="Only for route=unsupported: what data/capability is missing.")


class Verdict(BaseModel):
    """LLM-judge half of the critic (the numeric/structural half is deterministic - see critic.py)."""
    answers_question: bool = Field(description="Does the Answer part directly and fully answer the user's question?")
    takeaway_is_analytical: bool = Field(description="Does 'Why it matters' state something specific that stands out "
                                                     "in THIS data (a driver, concentration, trend, anomaly) rather than a platitude?")
    claims_supported: bool = Field(True, description="Are 'mainly / concentrated / driven by' claims and any recommendation or "
                                   "what-if backed by the numbers in the reply? False e.g. if a 40% share is called 'concentrated', or "
                                   "if it says fixing one segment 'would close most of the gap' without support.")
    issues: list[str] = Field(default_factory=list, description="Short, concrete fixes. Empty if none.")


# ------------------------------------------------------------------------------------------ helpers
def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _history(messages, n=HISTORY_TURNS) -> str:
    lines = []
    for m in messages[-n:]:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {_text(m.content)}")
        elif isinstance(m, AIMessage) and _text(m.content).strip():
            lines.append(f"Assistant: {_text(m.content)[:1500]}")
    return "\n".join(lines) or "(no earlier messages)"


def _time_context() -> str:
    d = load_df()["Order Date"].max()
    q = d.to_period("Q")
    return (f"Latest order date in data: {d:%Y-%m-%d}. Latest calendar quarter in data: {q} "
            f"({q.start_time:%Y-%m-%d} to {q.end_time:%Y-%m-%d}); latest year: {d.year}. "
            f"'Last quarter' / 'last year' / 'recently' mean these periods in the DATA, not today's date.")


def _friendly_error(e: Exception) -> str:
    """Short, user-safe description of an LLM/API failure (no org ids or raw payloads)."""
    name, msg = type(e).__name__, str(e)
    if ("rate" in name.lower() or "429" in msg or "rate limit" in msg.lower()
            or "ratelimit" in msg.lower()):
        return ("the model provider's rate limit was reached (free tiers have per-minute and per-day token caps). "
                "Wait a few minutes, or set a different LLM_MODEL / LLM_PROVIDER")
    if "auth" in name.lower() or "401" in msg or "api key" in msg.lower():
        return "the API key was rejected - check GROQ_API_KEY / your provider key"
    return f"{name}: {msg[:120]}"


def _clip(s, n=1800) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + " ...[truncated]"


# ------------------------------------------------------------------------------------------ prompts
def planner_prompt() -> str:
    return f"""You are the PLANNER of "Insight Copilot", an analyst agent that explores ONE dataset. You do not answer the
question; you decide how it will be handled and explain your reasoning.

DATA CARD
{data_card()}
{_time_context()}

TOOLS AVAILABLE TO THE EXECUTOR
- query_data: rankings, totals, breakdowns by dimension and/or month/quarter/year.
- analyze_stats: growth_rate, trend, seasonality, anomalies, share_of_total, compare_groups (with gap drivers), summary.
- create_chart: line/bar/pie/heatmap of aggregated data.
- forecast_sales: forecast future monthly sales from the dataset's latest available month.
- web_search: outside context only (events, industry background) - never for numbers in the dataset.

ROUTES
- "tools": answering requires computing from the dataset (or outside context via web_search).
- "direct": greetings, questions about the agent or dataset structure answerable from the DATA CARD, or a follow-up that
  only needs re-explaining earlier answers with no new numbers.
- "clarify": ONLY if the request is so ambiguous that different readings give materially different answers and no
  sensible default exists. Prefer stating an assumption and proceeding.
- "unsupported": needs data or capability this dataset/toolbox lacks (profit, margin, cost, quantity, discounts,
  returns, customer demographics, other datasets, real-time data, predictions other than aggregate sales forecasts (e.g. stock prices, market data, what an individual customer or
  product will do in future), or custom
  computation the tools above cannot express). Say so rather than improvising.

RULES
- Resolve follow-ups ("and for the West?", "why?") using the conversation history.
- For a follow-up that changes only one dimension (e.g. "now Furniture"), carry forward the prior request's
  metric, time grain/date range, and all still-applicable filters; change only the dimension the user changed.
  If the inherited period/filter cannot be determined safely, state the assumption or ask a concise clarification.
- Choose only the tools that are truly needed. A simple ranking needs only query_data; do not add a chart or stats
  unless the question calls for them.
- For seasonality, trend, growth, anomaly, and comparison questions, use analyze_stats with the relevant analysis type.
- Preserve all explicit user filters, including category, sub-category, region, segment, state, and date range, when
  selecting tools. For example, seasonality in Furniture must pass the Furniture category filter.
- "Show me / plot / graph / chart / visualize <a metric over time or by group>" (e.g. "Show me monthly sales") IS a
  request for a visual: plan query_data for the numbers AND create_chart (line for a time series, bar for groups).
- Multi-part asks ("compare X and Y, explain the driver, and show a chart") get ONE step per part, in order, each
  naming its tool (e.g. 1. analyze_stats compare_groups  2. create_chart bar of the same groups  3. explain).
- For forecasting, use forecast_sales and interpret the horizon relative to the latest date in the dataset, not today's
  date. Do not present a historical-data forecast as a current real-world prediction.
- goal = one line restating what the user wants. reasoning must be specific to THIS question and mention the
  period/filters you will use."""


def executor_prompt() -> str:
    return f"""You are the EXECUTOR of "Insight Copilot". Carry out the plan by calling tools, then stop.

DATA CARD
{data_card()}
{_time_context()}

RULES
- Before EVERY tool call, write ONE short sentence saying why you are calling it (it is shown to the user).
- Use the fewest tools needed; never call a tool whose output you will not use. Never invent numbers.
- Tool choice: query_data = rankings/totals/breakdowns. analyze_stats = growth, trend, seasonality, anomalies, shares,
  comparisons, dataset overview. create_chart = only if the user asked for a visual (plot, chart, graph, visualize,
  or "show me <metric> over time / by group" - always call it for these, after the numbers), or a trend/seasonality/comparison
  is clearer visually (call it in addition to the numbers, not instead). forecast_sales = future monthly sales from
  the latest available data month. web_search = outside context only.
- Preserve every explicit filter in tool arguments. For example, if the user asks about Furniture seasonality, pass
  category='Furniture' to analyze_stats. Do not silently drop filters when making follow-up calls or charts.
- On follow-ups, inherit the previous relevant question's metric, period, grouping and filters unless the user explicitly
  replaces them. A follow-up such as "now for Furniture" changes the category only; it must not reset the date range.
- When charting a comparison of specific groups (e.g. West vs South), pass group_by plus groups='West,South' so the chart
  shows ONLY those groups. This comma-separated groups argument is for tools that explicitly define a `groups` parameter;
  it is NOT a categorical filter value.
- A categorical filter such as region, category, sub_category, segment, or state accepts ONE value only. NEVER pass
  comma-separated values as one filter, e.g. region='West,South'.
- ANY "X vs Y" / "compare" / "difference between" request - including a follow-up like "and what about Central vs East?" -
  MUST call analyze_stats compare_groups for those groups (same metric/filters as the previous comparison). Do not
  substitute several query_data calls: only compare_groups returns the verified gap drivers.
- To compare multiple groups, use analyze_stats with analysis='compare_groups' and groups='West,South'. Reuse its
  gap_drivers output directly when it provides the requested breakdown.
- If additional query_data evidence is needed for a group comparison, call query_data separately for each group with
  the same metric, group_by, and other filters, changing only that group's categorical filter. Never retry by silently
  removing a failed filter or substituting an unfiltered query.
- If a tool returns ERROR, read it, fix the arguments and retry (at most twice). If the required filtered evidence cannot
  be obtained, state that limitation rather than using unfiltered results.
- Always pass explicit start_date / end_date for relative periods ("last quarter", "in 2017"). Resolve relative periods
  against the dataset's available dates and state that assumption in the final answer.
- For forecasts, explain that forecast dates follow the dataset cutoff. If the dataset ends in 2018, a 2019 forecast is
  a forecast based on historical data, not a prediction for the current calendar year. Report backtest error only when
  the tool provides it.
- If a tool returns ERROR, read it, fix the arguments and retry (at most twice), otherwise move on.
- Multi-step questions: call tools in sequence and use earlier results to choose later calls.
- When you have enough information, reply with exactly: DONE (no tool call)."""


def synthesizer_prompt() -> str:
    sample = ("\nNOTE: the app is running on a SYNTHETIC SAMPLE dataset (the real train.csv was not found). "
              "Say once, briefly, that figures come from sample data." if using_sample_data() else "")
    return f"""You are the ANALYST voice of "Insight Copilot". Write the final reply from the tool observations.

Format (Markdown, exactly these three parts):
**Answer:** 1-2 sentences that directly answer the question.
**Supporting numbers:**
- 3-6 bullets with concrete labelled figures ($ with thousands separators, percentages to 1 decimal).
**Why it matters:** ONE analytical sentence - what specifically stands out in THIS data: where a gap or change is
concentrated (or evenly spread), which segment drives it, how big it is relative to the whole, or what is unusual.
Name the segment and give a figure. Not a generic sentence like "this shows sales performance" or "this is important".

RULES
- Use ONLY numbers found in the observations; derived numbers (differences, ratios) must be correct.
- Synthesize - compare, rank, explain drivers. Never paste raw JSON. A small markdown table is OK only for >= 4 ranked rows.
- If the observations say a chart was created, open the **Answer** with "The chart below shows ..." (one clause).
- Name the period EXACTLY as queried. If no start_date/end_date was used, say "across all years (2015-2018)"; never
  invent a quarter or year the tool calls did not use.
- State assumptions briefly (e.g. "last quarter = Q4 2018, the latest in the data").
- For forecast results, explicitly identify the forecast period and the dataset's latest available date. Make clear that
  the forecast extends from the historical dataset cutoff; do not imply it describes current or future real-world sales
  relative to today's date. Include forecast error only if supplied by the tool.
- Do not infer acceleration, causes, or future business conditions from forecast values alone. Describe projected values
  neutrally. For example: "The model projects sales above the same period in 2018; this is a historical model estimate,
  not a guarantee."
- If a chart was created, mention it with the word 'chart' (e.g. "The chart below shows ...").
- Only claim a driver is 'the main' / 'concentrated' if it is > 50% of the gap; a 30-40% share among 3 similar parts means
  the gap is broad-based - say so. Also say whether a gap is about VOLUME (orders) or BASKET SIZE (avg order value).
- No recommendations or what-if claims ("boosting X would close the gap"): describe what the data shows only.
- Do not assume a driver: if the gap is spread evenly across categories, say so; if concentrated, name it.
- For "why / what's driving" questions describe what the data shows; label any causal explanation as a hypothesis.
- For anomaly results, report the identified period, metric, and anomaly score or threshold when available. An outlier is
  a statistical signal, not proof of a structural shift, business event, error, or causal explanation. Do not infer a
  structural change or explain why an anomaly occurred unless the observations independently support that claim.
- For seasonality, preserve the requested category, region, segment, state, and date filters in your interpretation.
  Do not generalize a filtered result to the entire dataset.
- If a tool failed or data is missing, say what could not be computed instead of guessing; answer what you can.
- Be concise (roughly <= 180 words).{sample}
DATA CONTEXT: {_time_context()}"""


_CHITCHAT_RE = re.compile(
    r"^\s*(hi|hello|hey|hiya|good (morning|afternoon|evening)|thanks?|thank you|ok(ay)?|cool)\b\W*(there|everyone|team|claude|copilot)?\W*$"
    r"|\bwhat (can|could) you do\b|\bwhat do you do\b|\bwho are you\b|\bhow (do|does) (you|this) work\b|\bhelp me (get started|understand)\b",
    re.I)


def _looks_like_chitchat(question: str) -> bool:
    """Greeting / 'what can you do?' - only used to pick a sane route when the planner model is unavailable."""
    return bool(_CHITCHAT_RE.search(question or ""))


# ------------------------------------------------------------------------------------------ nodes
def planner(state: AgentState) -> dict:
    """Decide route + write the visible plan. Also resets all per-turn state."""
    msgs = state["messages"]
    question = _text(msgs[-1].content)
    fallback = False
    try:
        plan = get_llm(0.0).with_structured_output(Plan).invoke([
            SystemMessage(content=planner_prompt()),
            HumanMessage(content=f"Conversation so far:\n{_history(msgs[:-1])}\n\nLatest user message:\n{question}")])
        plan = plan if isinstance(plan, Plan) else Plan(**plan)
    except Exception as e:  # noqa: BLE001 - model/API hiccup: degrade gracefully instead of crashing
        fallback = True
        if _looks_like_chitchat(question):
            plan = Plan(reasoning=f"(Planner unavailable: {_friendly_error(e)}.) This looks like a greeting or a question "
                                  f"about what I can do, so I'll answer directly.", route="direct")
        else:
            plan = Plan(reasoning=f"(Planner unavailable: {_friendly_error(e)}.) I'll treat this as a data question and "
                                  f"pick tools as I go.", route="tools", steps=["Use the most relevant data tools."])
    plan.tools = [t for t in plan.tools if t in TOOLS_BY_NAME]
    entry = {"kind": "plan", "route": plan.route, "goal": plan.goal, "text": plan.reasoning, "steps": plan.steps,
             "tools": plan.tools, "fallback": fallback}
    return {"plan": plan.model_dump(), "route": plan.route, "work": [], "observations": [], "charts": [],
            "steps": 0, "answer": "", "draft": "", "draft_ok": True, "critique": {}, "revisions": 0,
            "trace": [entry]}


def route_after_planner(state: AgentState) -> str:
    return state.get("route", "tools")


def _invoke_with_retry(llm, convo, attempts=2):
    last = None
    for _ in range(attempts):
        try:
            return llm.invoke(convo), None
        except Exception as e:  # noqa: BLE001 - e.g. malformed tool call / rate limit
            last = e
    return AIMessage(content=""), f"{type(last).__name__}: {last}"


_FORECAST_RE = re.compile(r"\b(forecast|predict|projection|project)\w*\b", re.I)
_REGIONS = ("West", "East", "Central", "South")
_CATEGORIES = ("Furniture", "Office Supplies", "Technology")
_SEGMENTS = ("Consumer", "Corporate", "Home Office")


def _offline_forecast_args(question: str) -> dict | None:
    """When the LLM is unavailable, a clear 'forecast sales ...' request can still be served: no model is needed to
    pick forecast_sales or to read the horizon / region / category / segment out of the question."""
    if not _FORECAST_RE.search(question or "") or not re.search(r"\bsales?\b|\brevenue\b", question, re.I):
        return None
    m = re.search(r"(\d+)\s*(month|quarter|year)", question, re.I)
    horizon = 6
    if m:
        horizon = int(m.group(1)) * {"month": 1, "quarter": 3, "year": 12}[m.group(2).lower()]
    elif re.search(r"next\s+quarter", question, re.I):
        horizon = 3
    elif re.search(r"next\s+year", question, re.I):
        horizon = 12
    args = {"horizon_months": max(1, min(horizon, 24))}
    for key, values in (("region", _REGIONS), ("category", _CATEGORIES), ("segment", _SEGMENTS)):
        hit = next((v for v in values if re.search(rf"\b{re.escape(v)}\b", question, re.I)), None)
        if hit:
            args[key] = hit
    return args


def executor(state: AgentState) -> dict:
    """LLM (with tools bound) decides the next tool call(s) - or DONE. Records its 'thought' in the trace."""
    plan, msgs = state["plan"], state["messages"]
    brief = (f"Conversation so far:\n{_history(msgs[:-1])}\n\nCurrent question: {_text(msgs[-1].content)}\n\n"
             f"Plan: {plan.get('reasoning', '')}\nSteps:\n" + "\n".join(f"- {s}" for s in plan.get("steps", [])) +
             f"\nSuggested tools: {plan.get('tools', [])}")
    convo = [SystemMessage(content=executor_prompt()), HumanMessage(content=brief), *state.get("work", [])]
    # The structured plan's `tools` field decides which tool schemas the executor may use (saves ~2-3k tokens/call).
    # Falls back to all tools if the plan named none (e.g. planner outage). Disable: ENFORCE_PLANNED_TOOLS=0.
    planned = [TOOLS_BY_NAME[t] for t in plan.get("tools", []) if t in TOOLS_BY_NAME]
    enforce = str(secret("ENFORCE_PLANNED_TOOLS", "1")).lower() not in ("0", "false", "no", "off")
    obs_now = state.get("observations") or []
    forecast_only = (obs_now and set(plan.get("tools") or []) <= {"forecast_sales"}
                     and obs_now[-1]["tool"] == "forecast_sales" and not obs_now[-1]["output"].startswith("ERROR"))
    if forecast_only or (obs_now and any(e.get("offline") for e in state.get("trace", []))):
        # a forecast-only plan is complete once the tool ran: skip the extra executor call (saves tokens, avoids rate limits)
        return {"steps": state.get("steps", 0) + 1,
                "trace": state.get("trace", []) + [{"kind": "thought", "text": "I have enough information - writing the answer."}]}
    llm = get_llm(0.0).bind_tools(planned if (enforce and planned) else ALL_TOOLS)
    resp, error = _invoke_with_retry(llm, convo)
    offline_args = None
    if error and not state.get("observations"):         # LLM outage: a clear forecast request needs no model to route
        offline_args = _offline_forecast_args(_text(msgs[-1].content))
        if offline_args:
            resp = AIMessage(content="", tool_calls=[{"name": "forecast_sales", "args": offline_args,
                                                      "id": "offline-forecast", "type": "tool_call"}])
    if not error and not resp.tool_calls and not state.get("observations"):   # must gather data at least once
        resp, error = _invoke_with_retry(llm, convo + [AIMessage(content=_text(resp.content) or "DONE"), HumanMessage(
            content="You have not called any tool yet. Call the tool(s) needed to answer the question.")])
    calls = resp.tool_calls if offline_args else ([] if error else (resp.tool_calls or []))
    thought = _text(resp.content).strip()
    if thought.upper().startswith("DONE"):
        thought = ""
    if calls:
        names = ", ".join(c["name"] for c in calls)
        plan_steps = plan.get("steps", [])
        hint = plan_steps[min(state.get("steps", 0), len(plan_steps) - 1)] if plan_steps else ""
        text = thought or (f"{hint} (calling {names})" if hint else f"Calling {names}.")
    elif error:
        text = "The model call failed - I'll answer with what I have."
    else:
        text = "I have enough information - writing the answer."
    if offline_args:
        text = ("The model is unavailable, but this is clearly a sales-forecast request, so I'm running forecast_sales "
                f"directly (horizon {offline_args['horizon_months']} months).")
    entry = {"kind": "thought", "text": text}
    if offline_args:
        entry["offline"] = True
    if error:
        entry["llm_error"] = error[:300]          # lets the eval runner tell provider outages from agent mistakes
    update = {"steps": state.get("steps", 0) + 1, "trace": state.get("trace", []) + [entry]}
    if calls:
        update["work"] = state.get("work", []) + [resp]
    return update


def tool_node(state: AgentState) -> dict:
    """Runs every tool call requested by the last AI message; stores observations, charts and trace entries."""
    last = state["work"][-1]
    work, obs, trace, charts = list(state["work"]), list(state.get("observations", [])), list(state["trace"]), list(state.get("charts", []))
    seen = {(o["tool"], json.dumps(o["args"], sort_keys=True, default=str)) for o in obs}
    for call in last.tool_calls:
        name, args = call["name"], call.get("args", {})
        key = (name, json.dumps(args, sort_keys=True, default=str))
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            msg = ToolMessage(content=f"ERROR: unknown tool '{name}'. Valid: {list(TOOLS_BY_NAME)}",
                              tool_call_id=call["id"], name=name)
        elif key in seen:
            msg = ToolMessage(content="Already executed with identical arguments - reuse the earlier result.",
                              tool_call_id=call["id"], name=name)
        else:
            try:
                msg = tool.invoke(call)
            except Exception as e:  # noqa: BLE001 - e.g. schema validation error
                msg = ToolMessage(content=f"ERROR: invalid arguments ({type(e).__name__}): {e}",
                                  tool_call_id=call["id"], name=name)
        seen.add(key)
        content = _text(msg.content)
        if getattr(msg, "artifact", None):
            charts.append(msg.artifact)
        work.append(msg)
        obs.append({"tool": name, "args": args, "output": content[:6000]})
        trace.append({"kind": "tool", "tool": name, "args": args, "output": _clip(content),
                      "chart": bool(getattr(msg, "artifact", None)), "error": content.startswith("ERROR")})
    return {"work": work, "observations": obs, "trace": trace, "charts": charts}


def route_after_executor(state: AgentState) -> str:
    work = state.get("work") or []
    wants_tools = bool(work) and isinstance(work[-1], AIMessage) and bool(work[-1].tool_calls)
    return "tools" if wants_tools and state.get("steps", 0) < MAX_STEPS else "synthesize"


def _final(text: str, trace_note: str | None = None, state: AgentState | None = None) -> dict:
    out = {"messages": [AIMessage(content=text)], "answer": text}
    if trace_note:
        out["trace"] = (state or {}).get("trace", []) + [{"kind": "thought", "text": trace_note}]
    return out


def _observation_block(state: AgentState) -> str:
    return "\n\n".join(f"[{i}] tool={o['tool']} args={json.dumps(o['args'], default=str)}\nresult: {o['output']}"
                        for i, o in enumerate(state.get("observations", []), 1)) or "(no tool observations)"


def _forecast_answer(state: AgentState) -> str | None:
    """Deterministic 3-part answer for a successful forecast_sales result (no extra LLM call, so a rate limit
    cannot discard a forecast that was already computed). Returns None when there is no usable forecast."""
    obs = next((o for o in reversed(state.get("observations", []))
                if o.get("tool") == "forecast_sales" and not str(o.get("output", "")).startswith("ERROR")), None)
    if not obs:
        return None
    try:
        data = json.loads(obs.get("output", "{}"))
    except Exception:  # noqa: BLE001 - truncated/invalid JSON: let the LLM synthesizer handle it
        return None
    rows = data.get("forecast") if isinstance(data, dict) else None
    if not rows:
        return None

    filters = data.get("filters_applied") or {}
    scope = (" for " + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in filters.items())) if filters else ""
    lines = [f"**Answer:** The sales forecast{scope} covers {data.get('forecast_period', 'the requested horizon')}, "
             f"following the latest month in the dataset ({data.get('history_ends', 'n/a')})."
             + (f" Total forecast: ${data['forecast_total']:,.2f}." if data.get("forecast_total") is not None else ""),
             "", "**Supporting numbers:**"]
    if data.get("method"):
        lines.append(f"- Method: {data['method']}")
    if data.get("training_months") is not None:
        lines.append(f"- Training history: {data['training_months']} months")
    if data.get("pct_vs_prior_year") is not None:
        lines.append(f"- Versus the same months a year earlier: {data['pct_vs_prior_year']}%")
    if data.get("backtest_mape_pct_last_6_months") is not None:
        lines.append(f"- Backtest MAPE (last 6 months): {data['backtest_mape_pct_last_6_months']}%")
    if data.get("baseline_seasonal_naive_mape_pct") is not None:
        lines.append(f"- Seasonal-naive baseline MAPE: {data['baseline_seasonal_naive_mape_pct']}%")
    if data.get("beats_seasonal_naive_baseline") is not None:
        lines.append("- The model " + ("beat" if data["beats_seasonal_naive_baseline"] else "did not beat")
                     + " the seasonal-naive baseline in the backtest.")
    if state.get("charts"):                     # the critic requires an answer to mention a chart it was given
        lines += ["", "The chart below shows the monthly sales history together with this forecast."]
    lines += ["", "| Month | Forecast sales |", "|---|---:|"]
    lines += [f"| {r.get('month', '')} | ${r['forecast_sales']:,.2f} |" for r in rows if r.get("forecast_sales") is not None]
    lines += ["", "**Why it matters:** This is a statistical estimate extrapolated from the dataset's own history "
                  "(not a real-world prediction), so treat it as a planning baseline rather than a guarantee."]
    if data.get("caveat"):
        lines += ["", f"_{data['caveat']}_"]
    return "\n".join(lines)


def synthesizer(state: AgentState) -> dict:
    """Turns tool observations into a DRAFT analyst-style insight; the critic verifies it before it is shown.
    A successful forecast is formatted deterministically (saves an LLM call and survives provider outages)."""
    msgs = state["messages"]
    direct = _forecast_answer(state)
    if direct:
        return {"draft": direct, "draft_ok": True,
                "trace": state.get("trace", []) + [{"kind": "thought",
                         "text": "The forecast was computed by the tool - presenting its verified result directly."}]}
    if state.get("route", "tools") == "tools" and not state.get("observations"):
        # No tool ran, so there is nothing to report. Letting the LLM write anyway made it claim "the dataset has no data".
        errs = [e["llm_error"] for e in state.get("trace", []) if e.get("llm_error")]
        why = _friendly_error(RuntimeError(errs[-1])) if errs else "no analysis tool could be run for this question"
        return {"draft": f"⚠️ I couldn't run the analysis: {why}. No data was retrieved, so I won't guess - "
                         "please try again in a few minutes.", "draft_ok": False}
    charts = f"\nCharts created for the user: {len(state.get('charts', []))}" if state.get("charts") else ""
    try:
        resp = get_llm(0.2).invoke([
            SystemMessage(content=synthesizer_prompt()),
            HumanMessage(content=f"Conversation so far:\n{_history(msgs[:-1])}\n\nQuestion: {_text(msgs[-1].content)}\n\n"
                                 f"Plan: {state['plan'].get('reasoning', '')}\n\nTOOL OBSERVATIONS\n{_observation_block(state)}{charts}")])
        draft = _text(resp.content).strip()
        if draft:
            return {"draft": draft, "draft_ok": True}
        return {"draft": "The analysis completed, but the model returned no written response. Please try again.",
                "draft_ok": False}
    except Exception as e:  # noqa: BLE001
        text = f"⚠️ I couldn't produce an answer: {_friendly_error(e)}. Please try again."
        return {"draft": text, "draft_ok": False,
                "trace": state.get("trace", []) + [{"kind": "thought", "text": "Final answer synthesis was unavailable.",
                                                     "llm_error": f"{type(e).__name__}: {str(e)[:300]}"}]}


def critic(state: AgentState) -> dict:
    """Self-check: verify the draft against the tool observations before showing it.

    Deterministic checks (critic.py): every figure grounded in observations or a simple derivation of them, the three
    required sections, a non-generic takeaway, chart mentioned. LLM-judge check: answers the question + analytical takeaway.
    PASS -> publish the draft (appended to chat history). FAIL -> `revise` (bounded by MAX_REVISIONS).
    """
    draft, obs, n = state.get("draft", ""), state.get("observations", []), state.get("revisions", 0)
    if not state.get("draft_ok", True):                      # LLM outage message: nothing to verify
        verdict = {"passed": True, "issues": [], "ungrounded": [], "n_figures": 0, "skipped": True}
    else:
        res = check_answer(draft, obs, len(state.get("charts", [])))
        verdict = res.as_dict()
        try:
            if str(secret("CRITIC_LLM_REVIEW", "0")).lower() in ("0", "false", "no", "off"):
                raise RuntimeError("disabled")          # saves 1 LLM call per answer on tight free tiers
            j = get_llm(0.0).with_structured_output(Verdict).invoke([
                SystemMessage(content="You are a strict reviewer of a data analyst's reply. Judge ONLY the two questions "
                                      "in the schema. Do not re-check numbers (another check does that)."),
                HumanMessage(content=f"Question: {_text(state['messages'][-1].content)}\n\nReply:\n{draft}")])
            j = j if isinstance(j, Verdict) else Verdict(**j)
            verdict.update(llm_answers_question=j.answers_question, llm_takeaway_ok=j.takeaway_is_analytical)
            verdict["llm_claims_supported"] = j.claims_supported
            if not (j.answers_question and j.takeaway_is_analytical and j.claims_supported):
                verdict["issues"] += j.issues or ["Reviewer: answer incomplete or takeaway not analytical."]
                verdict["passed"] = False
        except Exception as e:  # noqa: BLE001 - reviewer unavailable: rely on the deterministic checks
            verdict["llm_review"] = f"skipped ({type(e).__name__})"
    exhausted = n >= MAX_REVISIONS
    done = verdict["passed"] or exhausted
    if verdict["passed"]:
        text = f"Self-check passed - {verdict['n_figures']} figure(s) verified against tool output." if not verdict.get("skipped") \
            else "Self-check skipped - no answer was generated (model unavailable)."
    elif exhausted:
        text = "Self-check still failing after " + f"{n} revision(s) - showing the best draft with a caveat: " + "; ".join(verdict["issues"])
    else:
        text = "Self-check failed - sending back for revision: " + "; ".join(verdict["issues"])
    entry = {"kind": "critic", "passed": verdict["passed"] and not verdict.get("skipped"), "skipped": bool(verdict.get("skipped")),
             "attempt": n + 1, "issues": verdict["issues"], "checks": verdict.get("checks", []), "text": text}
    out = {"critique": verdict, "trace": state.get("trace", []) + [entry]}
    if done:
        final = draft
        if not verdict["passed"] and verdict.get("ungrounded"):
            final += ("\n\n> \u26a0\ufe0f Could not verify against the data: " + ", ".join(verdict["ungrounded"]) +
                      ". Treat these figures with caution.")
        note = forecast_note(obs) if state.get("draft_ok", True) else None
        if note:                                    # method / validation / dates: written by code, never skipped by the LLM
            final += "\n\n" + note
        out.update(answer=final, messages=[AIMessage(content=final)])
    return out


def route_after_critic(state: AgentState) -> str:
    v = state.get("critique") or {}
    return "done" if v.get("passed") or state.get("revisions", 0) >= MAX_REVISIONS else "revise"


def revise(state: AgentState) -> dict:
    """Rewrite the draft to fix the critic's issues, using only the tool observations."""
    issues = (state.get("critique") or {}).get("issues", [])
    try:
        resp = get_llm(0.1).invoke([
            SystemMessage(content=synthesizer_prompt() + "\n\nYou are REVISING a draft that failed review. Fix every listed "
                                                          "issue, keep the same 3-part format, and use only numbers from the observations."),
            HumanMessage(content=f"Question: {_text(state['messages'][-1].content)}\n\nTOOL OBSERVATIONS\n{_observation_block(state)}"
                                 f"\n\nDRAFT:\n{state.get('draft', '')}\n\nISSUES TO FIX:\n" + "\n".join(f"- {i}" for i in issues))])
        new, rev = _text(resp.content).strip(), state.get("revisions", 0) + 1
    except Exception:  # noqa: BLE001 - can't revise: keep the draft, burn the budget so we don't loop
        new, rev = state.get("draft", ""), MAX_REVISIONS
    return {"draft": new, "revisions": rev,
            "trace": state.get("trace", []) + [{"kind": "revise", "text": f"Revising the answer (round {rev}) to fix: " + "; ".join(issues)}]}


def direct_answer(state: AgentState) -> dict:
    msgs = state["messages"]
    sys = ("You are Insight Copilot, a friendly data analyst chatbot for the Superstore sales dataset. Answer using only "
           "the DATA CARD and conversation history; do not quote sales figures that are not in the history. If asked what "
           "you can do, briefly list: rankings/aggregations, growth & trend, seasonality, anomalies, region/segment "
           "comparisons with drivers, charts, sales forecasting, web context - and give 3 example "
           f"questions. Be brief.\n\nDATA CARD\n{data_card()}")
    try:
        resp = get_llm(0.3).invoke([SystemMessage(content=sys), HumanMessage(
            content=f"Conversation so far:\n{_history(msgs[:-1])}\n\nUser: {_text(msgs[-1].content)}")])
        text = _text(resp.content).strip()
    except Exception as e:  # noqa: BLE001
        text = f"⚠️ Sorry - {_friendly_error(e)}. Please try again."
    return _final(text, "No tools needed - answering directly from the data card / conversation.", state)


def unsupported(state: AgentState) -> dict:
    msgs, plan = state["messages"], state["plan"]
    sys = ("You are Insight Copilot. The user asked for something this dataset/toolbox cannot do. In 3-5 sentences: say "
           "plainly that you don't have the data/tool for it and why (do NOT guess or invent numbers), then offer the "
           "closest 2 things you CAN do with the available data. Available data: sales by date, region, state, city, "
           "category, sub-category, product, customer, segment, ship mode.\n\nDATA CARD\n" + data_card())
    try:
        resp = get_llm(0.2).invoke([SystemMessage(content=sys), HumanMessage(
            content=f"User: {_text(msgs[-1].content)}\nMissing capability (from planner): {plan.get('unsupported_reason', '')}")])
        text = _text(resp.content).strip()
    except Exception:  # noqa: BLE001
        text = ("I don't have the data or a tool for that. " + plan.get("unsupported_reason", "") +
                " I can still analyse sales by time, region, category, product and customer.")
    return _final(text, "Declining to guess: no tool/data can answer this.", state)


def clarify(state: AgentState) -> dict:
    q = state["plan"].get("clarifying_question") or "Could you tell me a bit more about what you'd like to look at?"
    return _final(q, "The question is ambiguous - asking the user to clarify before running any tools.", state)


# ------------------------------------------------------------------------------------------ graph
def default_checkpointer():
    """Conversation memory store. In-memory by default; a SQLite file if CHECKPOINT_DB is set, so the agent's memory
    survives server restarts (needs `pip install langgraph-checkpoint-sqlite`). Any problem -> falls back to memory."""
    path = str(secret("CHECKPOINT_DB", "") or "").strip()
    if path and path.lower() not in ("0", "off", "none", "false", "no"):
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        except Exception:  # noqa: BLE001 - package missing / unwritable path: never block the app
            pass
    return MemorySaver()


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    for name, fn in [("planner", planner), ("executor", executor), ("tool_node", tool_node),
                     ("synthesizer", synthesizer), ("critic", critic), ("revise", revise),
                     ("direct_answer", direct_answer),
                     ("clarify", clarify), ("unsupported", unsupported)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner, {
        "tools": "executor", "direct": "direct_answer", "clarify": "clarify", "unsupported": "unsupported"})
    g.add_conditional_edges("executor", route_after_executor, {"tools": "tool_node", "synthesize": "synthesizer"})
    g.add_edge("tool_node", "executor")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"done": END, "revise": "revise"})
    g.add_edge("revise", "critic")
    for terminal in ("direct_answer", "clarify", "unsupported"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer or default_checkpointer())


def stream_turn(graph, question: str, thread_id: str):
    """Yield (node_name, state_update) as each node finishes - lets the UI show reasoning live."""
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    for update in graph.stream({"messages": [HumanMessage(content=question)]}, cfg, stream_mode="updates"):
        yield from update.items()
