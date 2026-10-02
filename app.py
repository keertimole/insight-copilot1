"""Streamlit chat UI for Insight Copilot. Run: streamlit run app.py"""
import json
import re
import time
import uuid

import plotly.io as pio
import streamlit as st

from insight_copilot.config import has_api_key, model_name, provider
from insight_copilot.data import dataset_badge, load_df, using_sample_data, validate_dataset
from insight_copilot.observability import aggregate, log_turn, summarize_turn

st.set_page_config(page_title="Insight Copilot", page_icon="📊", layout="wide")

HERO = "Compare West and South sales, explain what is driving the difference, and show me a chart."
QUESTION_GROUPS = {
    "⭐ Featured": [HERO],
    "📊 Rankings & totals": [
        "What were the top 3 sub-categories by sales last quarter?",
        "Who are our top 5 customers by total sales in 2017?",
        "What are the top 10 products by sales?",
        "Which 5 states have the highest sales?",
        "Which cities generated the most sales?",
        "What are the bottom 5 sub-categories by sales?",
        "What is total sales by year?",
        "How many orders did we have each year?",
        "What is the average order value by segment?",
    ],
    "⚖️ Compare": [
        "Compare the West and South regions and tell me what's driving the difference.",
        "Compare Consumer and Corporate segments and explain the gap.",
        "Compare Furniture and Technology sales by region.",
        "Compare East and Central sales.",
        "Compare 2016 and 2017 sales by category.",
        "How do California and Texas compare?",
    ],
    "📈 Trends & charts": [
        "Is there a seasonal trend in sales for Furniture?",
        "Plot monthly sales by category",
        "What is the year-over-year growth in Technology sales?",
        "Plot monthly sales for the West region",
        "Show each category's share of total sales as a pie chart",
        "Show a seasonality heatmap of sales",
        "Which months are the strongest and weakest for sales?",
        "What is the quarterly sales trend for Office Supplies?",
        "Show sales by segment as a bar chart",
        "Is overall sales growing or declining?",
    ],
    "🔮 Forecast": [
        "Forecast sales for the next 6 months",
        "Forecast Technology sales for the next 12 months",
        "Forecast West region sales for the next 3 months",
        "How accurate is the sales forecast?",
    ],
    "🔍 Unusual patterns": [
        "Summarize anything unusual in this data.",
        "Which months had unusually high or low sales?",
        "Are there any outlier orders with very large sales?",
        "Give me an overview of the dataset.",
    ],
    "🧩 Customers & products": [
        "Which segment brings in the most sales?",
        "How many unique customers do we have each year?",
        "Which product has the highest sales in Technology?",
        "What are the top 5 products in Furniture?",
        "Which ship mode is used most often?",
        "Who are the top 5 Corporate customers?",
    ],
    "🌎 Regions & states": [
        "What share of total sales does each region contribute?",
        "Which region grew fastest year over year?",
        "What are the top 5 states in the West region?",
        "Which sub-category sells best in the South?",
        "Plot yearly sales by region",
    ],
    "🌐 Outside context (web)": [
        "According to industry sources, how much do retail sales typically rise around Black Friday?",
        "What are typical retail seasonality patterns in the US?",
    ],
    "💬 Follow-ups (ask right after a comparison)": [
        "And what about Central vs East?",
        "Show that as a chart.",
        "Which sub-category is driving that gap?",
        "Break that down by segment.",
        "Now show it by year.",
    ],
    "🚫 Edge cases": [
        "What was our profit margin last year?",
        "What will Apple's stock price be tomorrow?",
        "Hi! What can you do?",
        "Delete all the 2015 orders from the dataset",
        "Which customer will buy the most next year?",
        "What about profit?",
    ],
}
EXAMPLES = [q for qs in QUESTION_GROUPS.values() for q in qs]


@st.cache_resource(show_spinner="Loading agent...")
def get_graph():
    from insight_copilot.graph import build_graph
    return build_graph()


def esc(t: str) -> str:
    """Streamlit renders $...$ as LaTeX; escape dollar signs so money displays normally."""
    return str(t).replace("$", r"\$")


def _tool_chips(trace: list) -> str:
    return " · ".join(f"{'⚠️' if e.get('error') else '✓'} `{e['tool']}`" for e in trace if e["kind"] == "tool")


def _crit_label(e: dict) -> str:
    return "– skipped (no answer to verify)" if e.get("skipped") else ("✓ passed" if e["passed"] else "⚠️ flagged")


def render_summary(trace: list) -> None:
    """Agent plan / tools selected / execution / self-check at a glance."""
    plan = next((e for e in trace if e["kind"] == "plan"), None)
    if not plan:
        return
    st.markdown("**🧠 Agent plan**")
    if plan.get("goal"):
        st.markdown(f"**Goal:** {esc(plan['goal'])}")
    if plan.get("steps"):
        st.markdown("**Steps:**\n" + esc("\n".join(f"{i}. {s}" for i, s in enumerate(plan["steps"], 1))))
    st.markdown("**🔧 Tools selected:** " + (", ".join(f"`{t}`" for t in plan["tools"]) or
                                            ("_none needed_" if plan["route"] != "tools" else "_decided step by step_")))
    chips = _tool_chips(trace)
    st.markdown("**⚙️ Execution:** " + (chips or f"_no tools run (route → `{plan['route']}`)_"))
    crit = [e for e in trace if e["kind"] == "critic"]
    if crit:
        st.markdown(f"**🛡️ Self-check:** {_crit_label(crit[-1])} after {len(crit)} review(s)")
        for c in crit[-1].get("checks", []):
            st.markdown(f"- {'✓' if c['ok'] else '⚠️'} {esc(c['label'])}")


def render_entry(e: dict) -> None:
    """One line of the step-by-step log. (No expanders here: Streamlit forbids nesting them.)"""
    kind = e["kind"]
    if kind == "plan":
        st.markdown(f"**🧭 Plan** · route → `{e['route']}`")
        st.markdown(esc(e["text"]))
        if e.get("steps"):
            st.markdown(esc("\n".join(f"{i}. {s}" for i, s in enumerate(e["steps"], 1))))
        if e.get("tools"):
            st.caption("Tools planned: " + ", ".join(f"`{t}`" for t in e["tools"]))
        elif e["route"] == "tools":
            st.caption("Tools planned: (decided step by step)")
    elif kind == "thought":
        st.markdown(f"💭 {esc(e['text'])}")
    elif kind == "tool":
        st.markdown(f"{'⚠️' if e.get('error') else '🔧'} **Tool call:** `{e['tool']}`" + (" · 📈 chart" if e.get("chart") else ""))
        st.code(json.dumps({k: v for k, v in e["args"].items() if v not in ("", None)}, indent=2), language="json")
        out = e["output"]
        st.caption("Observation")
        st.code(out[:700] + (" …" if len(out) > 700 else ""), language="json")
    elif kind == "critic":
        st.markdown(f"🛡️ **Self-check** (review {e['attempt']}) · {_crit_label(e)}")
        st.markdown(esc(e["text"]))
    elif kind == "revise":
        st.markdown(f"✏️ {esc(e['text'])}")


def render_trace_panel(trace: list) -> None:
    with st.expander("🧠 Show reasoning", expanded=bool(st.session_state.get("auto_reason", False))):
        render_summary(trace)
        st.divider()
        st.caption("Step-by-step log")
        for e in trace:
            render_entry(e)
            st.divider()


def render_tool_line(trace: list) -> None:
    chips = _tool_chips(trace)
    crit = [e for e in trace if e["kind"] == "critic"]
    bits = ([f"🔧 {chips}"] if chips else []) + ([f"🛡️ self-check {_crit_label(crit[-1])}"] if crit else [])
    if bits:
        st.caption("  ·  ".join(bits))


def render_charts(charts: list, key: str) -> None:
    """Render Plotly JSON artifacts and show a visible diagnostic if one is invalid."""
    for i, cj in enumerate(charts):
        try:
            fig = pio.from_json(cj) if isinstance(cj, str) else cj
            st.plotly_chart(fig, width="stretch", key=f"{key}_{i}")
        except Exception as exc:  # noqa: BLE001 - keep the chat answer usable if a chart artifact is malformed
            st.warning(
                "The analysis completed, but this chart could not be displayed. "
                f"Chart rendering error: {type(exc).__name__}. Try asking for the chart again."
            )


def _sync_url() -> None:
    """Keep the conversation id in the page URL (?t=...) so a refresh can reattach to the agent's saved memory."""
    try:
        st.query_params["t"] = st.session_state.thread_id
    except Exception:  # noqa: BLE001 - older Streamlit / no query params: persistence just becomes session-only
        pass


def reset_chat() -> None:
    st.session_state.messages, st.session_state.thread_id = [], str(uuid.uuid4())
    st.session_state.turn_records = []
    _sync_url()


def _content_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def restore_chat(thread_id: str) -> bool:
    """Rebuild the transcript (text only; traces and charts are not stored) from the agent's checkpointed memory."""
    try:
        values = get_graph().get_state({"configurable": {"thread_id": thread_id}}).values or {}
        msgs = [{"role": "user" if m.__class__.__name__ == "HumanMessage" else "assistant", "content": _content_text(m.content)}
                for m in values.get("messages", []) if m.__class__.__name__ in ("HumanMessage", "AIMessage")]
    except Exception:  # noqa: BLE001 - unknown thread / graph unavailable: start fresh
        return False
    if not msgs:
        return False
    st.session_state.messages, st.session_state.thread_id, st.session_state.turn_records = msgs, thread_id, []
    return True


def stream_text(text: str, delay: float = 0.008):
    """Reveal an already-verified answer word by word (the critic approves the text BEFORE it is shown)."""
    for tok in re.split(r"(\s+)", text):
        yield tok
        time.sleep(delay)


def show_answer(answer: str) -> None:
    try:
        st.write_stream(stream_text(esc(answer)))
    except Exception:  # noqa: BLE001 - fall back to a plain render
        st.markdown(esc(answer))


def render_metrics(slot) -> None:
    """Session metrics from the per-turn observability records (re-rendered after every turn)."""
    recs = st.session_state.get("turn_records", [])
    with slot.container():
        with st.expander(f"📈 Session metrics ({len(recs)} turn{'' if len(recs) == 1 else 's'})"):
            if not recs:
                st.caption("Ask a question to see latency, tool usage and self-check results.")
                return
            m = aggregate(recs)
            st.markdown(f"**Latency:** avg {m['avg_latency_s']}s · p95 {m['p95_latency_s']}s")
            rate = m["self_check_pass_rate"]
            st.markdown(f"**Self-check pass rate:** {'n/a' if rate is None else f'{rate}%'} · **Revisions:** {m['revisions']}")
            st.markdown("**Routes:** " + (", ".join(f"{k} ×{v}" for k, v in m["routes"].items()) or "–"))
            st.markdown("**Tool calls:** " + (", ".join(f"`{k}` ×{v}" for k, v in m["tool_calls"].items()) or "none"))
            st.markdown(f"**Errors:** {m['errors']} · **Tool errors:** {m['tool_errors']} · **Planner fallbacks:** {m['planner_fallbacks']}")


# ------------------------------------------------------------------ session
if "messages" not in st.session_state:
    try:
        _tid = st.query_params.get("t")
    except Exception:  # noqa: BLE001
        _tid = None
    if not (_tid and restore_chat(_tid)):
        reset_chat()
st.session_state.setdefault("turn_records", [])
st.session_state.setdefault("pending", None)

# ------------------------------------------------------------------ sidebar
df = load_df()
with st.sidebar:
    st.title("📊 Insight Copilot")
    st.caption("A LangGraph reasoning agent for exploring the Superstore Sales dataset.")
    if using_sample_data():
        st.warning("Running on a **synthetic sample** (data/train.csv not found). Numbers are illustrative.")
    badge = dataset_badge()
    st.markdown(f"**📊 Dataset:** {badge['name']}  \n{badge['rows']:,} rows · {badge['start']}–{badge['end']}")
    checks = validate_dataset()
    with st.expander(f"{'✅' if all(c['ok'] for c in checks) else '⚠️'} Data quality ({sum(c['ok'] for c in checks)}/{len(checks)} checks)"):
        for c in checks:
            st.markdown(f"{'✓' if c['ok'] else '✗'} **{c['check']}** — {c['detail']}")
    st.markdown(f"**Model:** `{provider()}` / `{model_name()}`")
    if st.button("🗑️ New conversation (clears memory)", width="stretch"):
        reset_chat()
        st.rerun()
    st.caption("🧠 The agent remembers this conversation, so follow-ups like “and for Central?” work. "
               "Refreshing the page keeps it (text only).")
    metrics_slot = st.empty()
    render_metrics(metrics_slot)
    st.divider()
    st.markdown("**💡 Pick a question** (or type your own in the box below)")
    group = st.selectbox("Question type", list(QUESTION_GROUPS), key="q_group", label_visibility="collapsed")
    for q in QUESTION_GROUPS[group]:
        if st.button(q, key=f"ex_{q}", width="stretch"):
            st.session_state.pending = q
    st.toggle("🧠 Open reasoning panel automatically", key="auto_reason", value=False)
    st.caption(f"Dates are relative to the dataset (latest order: {df['Order Date'].max():%b %Y}), not today's date - "
               "so 'last quarter' and 'next 6 months' follow the data.")
    st.divider()
    st.caption("First load after inactivity can take ~30s (free-tier hosting wakes up from sleep).")

# ------------------------------------------------------------------ history
st.header("Ask me about the sales data")
if not st.session_state.messages:
    st.info("Type your own question in the box at the bottom, or pick one from the sidebar. "
            "Open **🧠 Show reasoning** under any answer to see the plan, tools and self-check.")
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(esc(m["content"]))
        if m["role"] == "assistant":
            render_charts(m.get("charts", []), key=f"hist_{i}")
            if m.get("trace"):
                render_tool_line(m["trace"])
                render_trace_panel(m["trace"])

# ------------------------------------------------------------------ new turn
prompt = st.chat_input("Type a question, e.g. Compare West and East and tell me what's driving the gap")
if not prompt and st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not has_api_key():
            st.error("No LLM API key configured. Add `GROQ_API_KEY` (free at console.groq.com) to Streamlit secrets "
                     "or a local `.env` file - see the README.")
            st.stop()
        from insight_copilot.graph import stream_turn
        trace, charts, answer, shown = [], [], "", 0
        err, t0 = None, time.perf_counter()
        status = st.status("🧠 Thinking...", expanded=True)
        try:
            for node, upd in stream_turn(get_graph(), prompt, st.session_state.thread_id):
                trace = upd.get("trace", trace)
                charts = upd.get("charts", charts) if node != "planner" else []
                answer = upd.get("answer", answer) or answer
                with status:
                    for e in trace[shown:]:
                        render_entry(e)
                        st.divider()
                shown = len(trace)
            status.update(label=f"🧠 Reasoning · {sum(e['kind'] == 'tool' for e in trace)} tool call(s)"
                                f"{' · self-check ✓' if any(e['kind'] == 'critic' and e['passed'] and not e.get('skipped') for e in trace) else ''}",
                          state="complete", expanded=False)
        except Exception as ex:  # noqa: BLE001 - never crash the UI
            err = type(ex).__name__
            status.update(label="Something went wrong", state="error", expanded=False)
            answer = (f"Sorry - I hit an error while working on that (`{type(ex).__name__}`). This is usually a "
                      f"rate limit or a transient API issue; please try again in a few seconds.")
        rec = summarize_turn(prompt, trace, answer, time.perf_counter() - t0, model_name(), provider(), err)
        st.session_state.turn_records.append(rec)
        log_turn(rec, st.session_state.thread_id)
        render_metrics(metrics_slot)
        show_answer(answer)
        render_charts(charts, key=f"live_{len(st.session_state.messages)}")
        render_tool_line(trace)
        if trace:
            render_trace_panel(trace)
    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace, "charts": charts})