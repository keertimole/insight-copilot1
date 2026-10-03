"""Streamlit chat UI for Insight Copilot. Run: streamlit run app.py"""

import json
import re
import time
import uuid

import plotly.io as pio
import streamlit as st

from landing import render_landing

from insight_copilot.config import (
    has_api_key,
    model_name,
    provider,
)

from insight_copilot.data import (
    dataset_badge,
    load_df,
    using_sample_data,
    validate_dataset,
)

from insight_copilot.observability import (
    aggregate,
    log_turn,
    summarize_turn,
)


st.set_page_config(
    page_title="Insight Copilot",
    page_icon="📊",
    layout="wide",
)
st.markdown(
    """
    <style>
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 1rem !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: min(900px, calc(100vw - 2rem)) !important;
        z-index: 999999 !important;
    }

    .main .block-container {
        padding-bottom: 7rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

HERO = (
    "Compare West and South sales, explain what is driving "
    "the difference, and show me a chart."
)


QUESTION_GROUPS = {
    "⭐ Featured": [
        HERO,
    ],
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


EXAMPLES = [
    q
    for qs in QUESTION_GROUPS.values()
    for q in qs
]


@st.cache_resource(show_spinner="Loading agent...")
def get_graph():
    from insight_copilot.graph import build_graph

    return build_graph()


def esc(t: str) -> str:
    """Streamlit renders $...$ as LaTeX; escape dollar signs."""
    return str(t).replace("$", r"\$")


def _tool_chips(trace: list) -> str:
    return " · ".join(
        f"{'⚠️' if e.get('error') else '✓'} `{e['tool']}`"
        for e in trace
        if e["kind"] == "tool"
    )


def _crit_label(e: dict) -> str:
    return (
        "– skipped (no answer to verify)"
        if e.get("skipped")
        else (
            "✓ passed"
            if e["passed"]
            else "⚠️ flagged"
        )
    )


def render_summary(trace: list) -> None:
    """Agent plan / tools selected / execution / self-check."""

    plan = next(
        (
            e
            for e in trace
            if e["kind"] == "plan"
        ),
        None,
    )

    if not plan:
        return

    st.markdown("**🧠 Agent plan**")

    if plan.get("goal"):
        st.markdown(
            f"**Goal:** {esc(plan['goal'])}"
        )

    if plan.get("steps"):
        st.markdown(
            "**Steps:**\n"
            + esc(
                "\n".join(
                    f"{i}. {s}"
                    for i, s in enumerate(
                        plan["steps"],
                        1,
                    )
                )
            )
        )

    st.markdown(
        "**🔧 Tools selected:** "
        + (
            ", ".join(
                f"`{t}`"
                for t in plan["tools"]
            )
            or (
                "_none needed_"
                if plan["route"] != "tools"
                else "_decided step by step_"
            )
        )
    )

    chips = _tool_chips(trace)

    st.markdown(
        "**⚙️ Execution:** "
        + (
            chips
            or f"_no tools run "
            f"(route → `{plan['route']}`)_"
        )
    )

    crit = [
        e
        for e in trace
        if e["kind"] == "critic"
    ]

    if crit:
        st.markdown(
            f"**🛡️ Self-check:** "
            f"{_crit_label(crit[-1])} "
            f"after {len(crit)} review(s)"
        )

        for c in crit[-1].get("checks", []):
            st.markdown(
                f"- "
                f"{'✓' if c['ok'] else '⚠️'} "
                f"{esc(c['label'])}"
            )


def render_entry(e: dict) -> None:
    """Render one line of the step-by-step log."""

    kind = e["kind"]

    if kind == "plan":
        st.markdown(
            f"**🧭 Plan** · route → `{e['route']}`"
        )

        st.markdown(
            esc(e["text"])
        )

        if e.get("steps"):
            st.markdown(
                esc(
                    "\n".join(
                        f"{i}. {s}"
                        for i, s in enumerate(
                            e["steps"],
                            1,
                        )
                    )
                )
            )

        if e.get("tools"):
            st.caption(
                "Tools planned: "
                + ", ".join(
                    f"`{t}`"
                    for t in e["tools"]
                )
            )

        elif e["route"] == "tools":
            st.caption(
                "Tools planned: "
                "(decided step by step)"
            )

    elif kind == "thought":
        st.markdown(
            f"💭 {esc(e['text'])}"
        )

    elif kind == "tool":
        st.markdown(
            f"{'⚠️' if e.get('error') else '🔧'} "
            f"**Tool call:** "
            f"`{e['tool']}`"
            + (
                " · 📈 chart"
                if e.get("chart")
                else ""
            )
        )

        st.code(
            json.dumps(
                {
                    k: v
                    for k, v in e["args"].items()
                    if v not in ("", None)
                },
                indent=2,
            ),
            language="json",
        )

        out = e["output"]

        st.caption("Observation")

        st.code(
            out[:700]
            + (
                " …"
                if len(out) > 700
                else ""
            ),
            language="json",
        )

    elif kind == "critic":
        st.markdown(
            f"🛡️ **Self-check** "
            f"(review {e['attempt']}) · "
            f"{_crit_label(e)}"
        )

        st.markdown(
            esc(e["text"])
        )

    elif kind == "revise":
        st.markdown(
            f"✏️ {esc(e['text'])}"
        )


def render_trace_panel(trace: list) -> None:
    with st.expander(
        "🧠 Show reasoning",
        expanded=bool(
            st.session_state.get(
                "auto_reason",
                False,
            )
        ),
    ):
        render_summary(trace)

        st.divider()

        st.caption(
            "Step-by-step log"
        )

        for e in trace:
            render_entry(e)
            st.divider()


def render_tool_line(trace: list) -> None:
    chips = _tool_chips(trace)

    crit = [
        e
        for e in trace
        if e["kind"] == "critic"
    ]

    bits = (
        [f"🔧 {chips}"]
        if chips
        else []
    ) + (
        [
            f"🛡️ self-check "
            f"{_crit_label(crit[-1])}"
        ]
        if crit
        else []
    )

    if bits:
        st.caption(
            " · ".join(bits)
        )


def render_charts(
    charts: list,
    key: str,
) -> None:
    """Render Plotly JSON artifacts."""

    for i, cj in enumerate(charts):
        try:
            fig = (
                pio.from_json(cj)
                if isinstance(cj, str)
                else cj
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{key}_{i}",
            )

        except Exception as exc:
            st.warning(
                "The analysis completed, but this chart "
                "could not be displayed. "
                f"Chart rendering error: "
                f"{type(exc).__name__}. "
                "Try asking for the chart again."
            )


def _sync_url() -> None:
    """Keep the conversation id in the page URL."""

    try:
        st.query_params["t"] = (
            st.session_state.thread_id
        )
    except Exception:
        pass


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = str(
        uuid.uuid4()
    )
    st.session_state.turn_records = []

    _sync_url()


def _content_text(c) -> str:
    if isinstance(c, str):
        return c

    if isinstance(c, list):
        return "".join(
            p.get("text", "")
            if isinstance(p, dict)
            else str(p)
            for p in c
        )

    return str(c)


def restore_chat(thread_id: str) -> bool:
    """Rebuild transcript from checkpointed memory."""

    try:
        values = (
            get_graph()
            .get_state(
                {
                    "configurable": {
                        "thread_id": thread_id
                    }
                }
            )
            .values
            or {}
        )

        msgs = [
            {
                "role": (
                    "user"
                    if m.__class__.__name__
                    == "HumanMessage"
                    else "assistant"
                ),
                "content": _content_text(
                    m.content
                ),
            }
            for m in values.get(
                "messages",
                [],
            )
            if m.__class__.__name__
            in (
                "HumanMessage",
                "AIMessage",
            )
        ]

    except Exception:
        return False

    if not msgs:
        return False

    st.session_state.messages = msgs
    st.session_state.thread_id = thread_id
    st.session_state.turn_records = []

    return True


def stream_text(
    text: str,
    delay: float = 0.008,
):
    """Reveal an already-verified answer word by word."""

    for tok in re.split(
        r"(\s+)",
        text,
    ):
        yield tok
        time.sleep(delay)


def show_answer(answer: str) -> None:
    try:
        st.write_stream(
            stream_text(
                esc(answer)
            )
        )
    except Exception:
        st.markdown(
            esc(answer)
        )


def render_metrics(slot) -> None:
    """Session metrics from per-turn observability records."""

    recs = st.session_state.get(
        "turn_records",
        [],
    )

    with slot.container():
        with st.expander(
            f"📈 Session metrics "
            f"({len(recs)} turn"
            f"{'' if len(recs) == 1 else 's'})"
        ):
            if not recs:
                st.caption(
                    "Ask a question to see latency, "
                    "tool usage and self-check results."
                )
                return

            m = aggregate(recs)

            st.markdown(
                f"**Latency:** "
                f"avg {m['avg_latency_s']}s · "
                f"p95 {m['p95_latency_s']}s"
            )

            rate = m[
                "self_check_pass_rate"
            ]

            st.markdown(
                f"**Self-check pass rate:** "
                f"{'n/a' if rate is None else f'{rate}%'} · "
                f"**Revisions:** {m['revisions']}"
            )

            st.markdown(
                "**Routes:** "
                + (
                    ", ".join(
                        f"{k} ×{v}"
                        for k, v in m[
                            "routes"
                        ].items()
                    )
                    or "–"
                )
            )

            st.markdown(
                "**Tool calls:** "
                + (
                    ", ".join(
                        f"`{k}` ×{v}"
                        for k, v in m[
                            "tool_calls"
                        ].items()
                    )
                    or "none"
                )
            )

            st.markdown(
                f"**Errors:** {m['errors']} · "
                f"**Tool errors:** "
                f"{m['tool_errors']} · "
                f"**Planner fallbacks:** "
                f"{m['planner_fallbacks']}"
            )


# ------------------------------------------------------------------
# session
# ------------------------------------------------------------------

if "messages" not in st.session_state:
    try:
        _tid = st.query_params.get("t")
    except Exception:
        _tid = None

    if not (
        _tid
        and restore_chat(_tid)
    ):
        reset_chat()


st.session_state.setdefault(
    "turn_records",
    [],
)

st.session_state.setdefault(
    "pending",
    None,
)

st.session_state.setdefault(
    "show_landing",
    True,
)


# ------------------------------------------------------------------
# sidebar
# ------------------------------------------------------------------

df = load_df()

with st.sidebar:

    st.title(
        "📊 Insight Copilot"
    )

    st.caption(
        "A LangGraph reasoning agent for "
        "exploring the Superstore Sales dataset."
    )

    if using_sample_data():
        st.warning(
            "Running on a **synthetic sample** "
            "(data/train.csv not found). "
            "Numbers are illustrative."
        )

    badge = dataset_badge()

    st.markdown(
        f"**📊 Dataset:** "
        f"{badge['name']}  \n"
        f"{badge['rows']:,} rows · "
        f"{badge['start']}–{badge['end']}"
    )

    checks = validate_dataset()

    with st.expander(
        f"{'✅' if all(c['ok'] for c in checks) else '⚠️'} "
        f"Data quality "
        f"({sum(c['ok'] for c in checks)}/{len(checks)} checks)"
    ):
        for c in checks:
            st.markdown(
                f"{'✓' if c['ok'] else '✗'} "
                f"**{c['check']}** — "
                f"{c['detail']}"
            )

    st.markdown(
        f"**Model:** "
        f"`{provider()}` / "
        f"`{model_name()}`"
    )

    if st.button(
        "🗑️ New conversation (clears memory)",
        width="stretch",
    ):
        reset_chat()
        st.session_state.show_landing = True
        st.rerun()

    st.caption(
        "🧠 The agent remembers this conversation, "
        "so follow-ups like “and for Central?” work. "
        "Refreshing the page keeps it (text only)."
    )

    metrics_slot = st.empty()

    render_metrics(
        metrics_slot
    )

    st.divider()

    st.markdown(
        "**💡 Pick a question** "
        "(or type your own in the box below)"
    )

    group = st.selectbox(
        "Question type",
        list(QUESTION_GROUPS),
        key="q_group",
        label_visibility="collapsed",
    )

    for q in QUESTION_GROUPS[group]:

        if st.button(
            q,
            key=f"ex_{q}",
            width="stretch",
        ):
            st.session_state.pending = q

    st.toggle(
        "🧠 Open reasoning panel automatically",
        key="auto_reason",
        value=False,
    )

    st.caption(
        f"Dates are relative to the dataset "
        f"(latest order: "
        f"{df['Order Date'].max():%b %Y}), "
        "not today's date - so 'last quarter' "
        "and 'next 6 months' follow the data."
    )

    st.divider()

    st.caption(
        "First load after inactivity can take "
        "~30s (free-tier hosting wakes up from sleep)."
    )


# ------------------------------------------------------------------
# landing
# ------------------------------------------------------------------

if st.session_state.show_landing:

    choice = render_landing(
        badge=badge,
        examples=EXAMPLES,
        has_chat=bool(
            st.session_state.messages
        ),
    )

    if choice is None:
        st.stop()

    if choice in (
        "start",
        "resume",
    ):
        st.session_state.show_landing = False
        st.rerun()

    if choice in EXAMPLES:
        st.session_state.pending = choice
        st.session_state.show_landing = False
        st.rerun()


# ------------------------------------------------------------------
# history
# ------------------------------------------------------------------

st.header(
    "Ask me about the sales data"
)

if not st.session_state.messages:
    st.info(
        "Type your own question in the box at the bottom, "
        "or pick one from the sidebar. Open "
        "**🧠 Show reasoning** under any answer to see "
        "the plan, tools and self-check."
    )


for i, m in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        m["role"]
    ):

        st.markdown(
            esc(m["content"])
        )

        if m["role"] == "assistant":

            render_charts(
                m.get("charts", []),
                key=f"hist_{i}",
            )

            if m.get("trace"):
                render_tool_line(
                    m["trace"]
                )

                render_trace_panel(
                    m["trace"]
                )


# ------------------------------------------------------------------
# new turn
# ------------------------------------------------------------------

prompt = st.chat_input(
    "Type a question, e.g. Compare West and East "
    "and tell me what's driving the gap"
)


if (
    not prompt
    and st.session_state.pending
):

    prompt, st.session_state.pending = (
        st.session_state.pending,
        None,
    )


if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        if not has_api_key():

            st.error(
                "No LLM API key configured. "
                "Add `GROQ_API_KEY` "
                "(free at console.groq.com) "
                "to Streamlit secrets or a local "
                "`.env` file - see the README."
            )

            st.stop()

        from insight_copilot.graph import stream_turn

        trace = []
        charts = []
        answer = ""
        shown = 0
        err = None

        t0 = time.perf_counter()

        status = st.status(
            "🧠 Thinking...",
            expanded=True,
        )

        try:

            for node, upd in stream_turn(
                get_graph(),
                prompt,
                st.session_state.thread_id,
            ):

                trace = upd.get(
                    "trace",
                    trace,
                )

                charts = (
                    upd.get(
                        "charts",
                        charts,
                    )
                    if node != "planner"
                    else []
                )

                answer = (
                    upd.get(
                        "answer",
                        answer,
                    )
                    or answer
                )

                with status:

                    for e in trace[shown:]:

                        render_entry(e)

                        st.divider()

                shown = len(trace)

            self_check = (
                " · self-check ✓"
                if any(
                    e["kind"] == "critic"
                    and e["passed"]
                    and not e.get("skipped")
                    for e in trace
                )
                else ""
            )

            status.update(
                label=(
                    f"🧠 Reasoning · "
                    f"{sum(e['kind'] == 'tool' for e in trace)} "
                    f"tool call(s)"
                    f"{self_check}"
                ),
                state="complete",
                expanded=False,
            )

        except Exception as ex:

            err = type(ex).__name__

            # ----------------------------------------------------------
            # IMPORTANT:
            # A tool may have completed successfully even though a
            # later LLM call failed because of a rate limit.
            #
            # Do NOT discard the successful tool result.
            # ----------------------------------------------------------

            status.update(
                label="⚠️ Final response unavailable",
                state="complete",
                expanded=False,
            )

            successful_tools = [
                e
                for e in trace
                if e.get("kind") == "tool"
                and not e.get("error")
                and e.get("output")
            ]

            if successful_tools:

                last_tool = successful_tools[-1]

                tool_name = last_tool.get(
                    "tool",
                    "analysis tool",
                )

                raw_output = last_tool.get(
                    "output",
                    "",
                )

                # ------------------------------------------------------
                # Parse the successful tool output.
                # ------------------------------------------------------

                try:

                    data = (
                        json.loads(raw_output)
                        if isinstance(
                            raw_output,
                            str,
                        )
                        else raw_output
                    )

                except Exception:

                    data = None

                # ------------------------------------------------------
                # Forecast-specific deterministic fallback.
                # ------------------------------------------------------

                if (
                    tool_name == "forecast_sales"
                    and isinstance(
                        data,
                        dict,
                    )
                ):

                    forecast = data.get(
                        "forecast",
                        [],
                    )

                    lines = [
                        "### Sales forecast",
                        "",
                        (
                            "The forecast was successfully "
                            "calculated, but the language "
                            "model was temporarily "
                            "rate-limited while writing "
                            "the final explanation."
                        ),
                        "",
                    ]

                    if data.get(
                        "forecast_period"
                    ):

                        lines.append(
                            "**Forecast period:** "
                            f"{data['forecast_period']}"
                        )

                    if data.get(
                        "history_ends"
                    ):

                        lines.append(
                            "**Historical data through:** "
                            f"{data['history_ends']}"
                        )

                    if data.get(
                        "method"
                    ):

                        lines.append(
                            "**Method:** "
                            f"{data['method']}"
                        )

                    if data.get(
                        "training_months"
                    ) is not None:

                        lines.append(
                            "**Training history:** "
                            f"{data['training_months']} months"
                        )

                    if data.get(
                        "backtest_mape_pct_last_6_months"
                    ) is not None:

                        lines.append(
                            "**Backtest MAPE:** "
                            f"{data['backtest_mape_pct_last_6_months']}%"
                        )

                    if data.get(
                        "baseline_seasonal_naive_mape"
                    ) is not None:

                        lines.append(
                            "**Seasonal-naive baseline MAPE:** "
                            f"{data['baseline_seasonal_naive_mape']}%"
                        )

                    if data.get(
                        "beats_seasonal_naive_baseline"
                    ) is not None:

                        comparison = (
                            "better than"
                            if data[
                                "beats_seasonal_naive_baseline"
                            ]
                            else "not better than"
                        )

                        lines.append(
                            "**Backtest comparison:** "
                            "The forecasting method performed "
                            f"{comparison} the seasonal-naive "
                            "baseline on the available "
                            "backtest window."
                        )

                    if forecast:

                        lines.extend(
                            [
                                "",
                                "**Monthly forecast:**",
                                "",
                                "| Month | Forecast sales |",
                                "|---|---:|",
                            ]
                        )

                        for row in forecast:

                            month = row.get(
                                "month",
                                "",
                            )

                            value = row.get(
                                "forecast_sales"
                            )

                            if value is not None:

                                lines.append(
                                    f"| {month} | "
                                    f"${value:,.2f} |"
                                )

                    if data.get(
                        "caveat"
                    ):

                        lines.extend(
                            [
                                "",
                                f"_{data['caveat']}_",
                            ]
                        )

                    answer = "\n".join(
                        lines
                    )

                else:

                    # --------------------------------------------------
                    # Generic successful-tool fallback.
                    # --------------------------------------------------

                    answer = (
                        f"The `{tool_name}` analysis "
                        "completed successfully, but "
                        "the language model was temporarily "
                        "unavailable while generating the "
                        "final explanation.\n\n"
                        "The completed tool result is available "
                        "in the **Show reasoning** panel below."
                    )

            else:

                # ------------------------------------------------------
                # No tool completed, so this really was a model/API
                # failure before useful analysis was produced.
                # ------------------------------------------------------

                answer = (
                    f"Sorry - I hit an error while "
                    f"working on that "
                    f"(`{type(ex).__name__}`). "
                    "This is usually a temporary "
                    "model/rate-limit issue. "
                    "Please try again in a few seconds."
                )

        # --------------------------------------------------------------
        # Observability
        # --------------------------------------------------------------

        rec = summarize_turn(
            prompt,
            trace,
            answer,
            time.perf_counter() - t0,
            model_name(),
            provider(),
            err,
        )

        st.session_state.turn_records.append(
            rec
        )

        log_turn(
            rec,
            st.session_state.thread_id,
        )

        render_metrics(
            metrics_slot
        )

        # --------------------------------------------------------------
        # Final answer
        # --------------------------------------------------------------

        show_answer(
            answer
        )

        # --------------------------------------------------------------
        # Charts generated by the successful tool remain visible.
        # --------------------------------------------------------------

        render_charts(
            charts,
            key=f"live_"
            f"{len(st.session_state.messages)}",
        )

        # --------------------------------------------------------------
        # Tool chips + reasoning panel
        # --------------------------------------------------------------

        render_tool_line(
            trace
        )

        if trace:
            render_trace_panel(
                trace
            )

    # --------------------------------------------------------------
    # Persist assistant response in the current conversation.
    # --------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "trace": trace,
            "charts": charts,
        }
    )