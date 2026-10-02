"""Landing page for Insight Copilot - pure Streamlit UI, no agent/LangGraph imports.

`render_landing()` draws the page and returns what the visitor chose:
    None      -> nothing yet (the app should st.stop() after calling this)
    "start"   -> open the chat
    "resume"  -> go back to the existing conversation
    <text>    -> open the chat and ask this sample question
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* hide the chat sidebar while the landing page is showing */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none; }
.block-container { max-width: 1080px; padding-top: 3rem; }

.ic-eyebrow { letter-spacing: .14em; font-size: .78rem; font-weight: 700; color: var(--primary-color, #4f46e5); margin-bottom: .6rem; }
.ic-hero h1 { font-size: 2.9rem; line-height: 1.12; font-weight: 800; margin: 0 0 1rem 0; padding: 0; }
.ic-hero h1 span { color: var(--primary-color, #4f46e5); }
.ic-sub { font-size: 1.15rem; line-height: 1.55; opacity: .8; max-width: 720px; margin-bottom: 1.4rem; }

.ic-stats { display: flex; gap: 2.5rem; flex-wrap: wrap; margin: 2rem 0 .5rem 0; padding: 1.1rem 0;
            border-top: 1px solid rgba(128,128,128,.25); border-bottom: 1px solid rgba(128,128,128,.25); }
.ic-stat b { display: block; font-size: 1.5rem; font-weight: 800; }
.ic-stat span { font-size: .85rem; opacity: .7; }

.ic-h2 { font-size: 1.6rem; font-weight: 750; margin: 2.6rem 0 .3rem 0; }
.ic-lead { opacity: .7; margin-bottom: 1.1rem; }

.ic-card { background: var(--secondary-background-color, rgba(128,128,128,.08)); border: 1px solid rgba(128,128,128,.2);
           border-radius: 14px; padding: 1.15rem 1.2rem; height: 100%; min-height: 148px; }
.ic-card .ic-ico { font-size: 1.5rem; margin-bottom: .35rem; }
.ic-card h4 { margin: 0 0 .3rem 0; padding: 0; font-size: 1.05rem; font-weight: 700; }
.ic-card p { margin: 0; font-size: .93rem; line-height: 1.5; opacity: .78; }

.ic-step { border-left: 3px solid var(--primary-color, #4f46e5); padding: .1rem 0 .1rem .9rem; }
.ic-step b { display: block; font-size: 1rem; }
.ic-step span { font-size: .9rem; opacity: .75; line-height: 1.45; }

.ic-foot { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,.25); font-size: .85rem; opacity: .65; }
@media (max-width: 640px) { .ic-hero h1 { font-size: 2.1rem; } }
</style>
"""

_FEATURES = [
    ("🧭", "Plans before it acts", "Reads your question, decides which tools it needs, and shows its plan, so you can see how it got there."),
    ("🛡️", "Checks every number", "A built-in reviewer compares each figure in the answer with the tool output and sends weak drafts back for revision."),
    ("🧠", "Remembers the conversation", "Ask a follow-up like \u201cand what about Central?\u201d and it keeps your metric, period and filters."),
    ("📈", "Charts on request", "Ask it to plot or compare and it builds the chart, then explains what the chart shows."),
    ("🔮", "Honest forecasts", "Projects the next months from the data and states the method, the period and its limits, with no guarantees."),
    ("🌐", "Outside context", "Can look up public sources for context the dataset doesn't hold, clearly separated from your own numbers."),
]

_STEPS = [
    ("1. Ask", "Type a question in plain English, or pick a sample."),
    ("2. Plan", "The agent chooses the right tools and shows its plan."),
    ("3. Analyze", "It queries the data, runs statistics and builds charts."),
    ("4. Verify", "Figures are checked against tool output before you see them."),
]


def render_landing(badge: dict, examples: list[str], has_chat: bool = False) -> str | None:
    st.markdown(_CSS, unsafe_allow_html=True)
    choice: str | None = None

    # ---- hero
    st.markdown(
        '<div class="ic-hero"><div class="ic-eyebrow">AI DATA ANALYST</div>'
        '<h1>Ask your sales data anything.<br><span>Get answers you can verify.</span></h1>'
        '<div class="ic-sub">Insight Copilot is a reasoning agent for the Superstore Sales dataset. It plans its approach, '
        'runs the analysis, and checks every figure before it answers.</div></div>', unsafe_allow_html=True)
    b1, b2, _ = st.columns([1.3, 1.3, 3])
    if b1.button("Start analyzing →", type="primary", key="land_start", width="stretch"):
        choice = "start"
    if has_chat and b2.button("Resume conversation", key="land_resume", width="stretch"):
        choice = "resume"

    # ---- stats strip (dynamic from the loaded dataset)
    st.markdown(
        '<div class="ic-stats">'
        f'<div class="ic-stat"><b>{badge["rows"]:,}</b><span>order lines analysed</span></div>'
        f'<div class="ic-stat"><b>{badge["start"]}–{badge["end"]}</b><span>order history</span></div>'
        '<div class="ic-stat"><b>Plan → Analyze → Verify</b><span>every answer follows this loop</span></div>'
        '</div>', unsafe_allow_html=True)

    # ---- features
    st.markdown('<div class="ic-h2">What it does</div><div class="ic-lead">Built to be useful and checkable, not just fluent.</div>',
                unsafe_allow_html=True)
    for row in range(0, len(_FEATURES), 3):
        for col, (ico, title, text) in zip(st.columns(3, gap="medium"), _FEATURES[row:row + 3]):
            col.markdown(f'<div class="ic-card"><div class="ic-ico">{ico}</div><h4>{title}</h4><p>{text}</p></div>',
                         unsafe_allow_html=True)
        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

    # ---- how it works
    st.markdown('<div class="ic-h2">How it works</div>', unsafe_allow_html=True)
    for col, (title, text) in zip(st.columns(4, gap="medium"), _STEPS):
        col.markdown(f'<div class="ic-step"><b>{title}</b><span>{text}</span></div>', unsafe_allow_html=True)

    # ---- sample questions
    if examples:
        st.markdown('<div class="ic-h2">Try a question</div><div class="ic-lead">One click opens the chat and runs it.</div>',
                    unsafe_allow_html=True)
        for i, q in enumerate(examples):
            if st.button(q, key=f"land_ex_{i}", width="stretch"):
                choice = q

    st.markdown('<div class="ic-foot">Built with LangGraph and Streamlit · Data: Superstore Sales · '
                'Figures come from the dataset and are checked against tool output.</div>', unsafe_allow_html=True)
    return choice
