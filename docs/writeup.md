# Write-up - design decisions, trade-offs, next steps

**Goal & data.** An analyst-style chatbot whose behaviour is an inspectable LangGraph, not one hidden prompt. I used the
*Superstore Sales (time-series)* dataset: dates, region, category and product make trend, seasonality, anomaly and
region-comparison questions answerable with real numbers.

**Graph.** `planner` writes a structured plan (goal, steps, tools) and picks a route: `tools`, `direct`, `clarify` or
`unsupported`. `executor <-> tool_node` is a bounded ReAct loop (max 6 turns) that records a one-line thought before every
call. `synthesizer` drafts *Answer / Supporting numbers / Why it matters*; `critic` verifies the draft and loops through
`revise` on failure (max 2 rounds). Three conditional edges (planner route, executor loop, critic verdict), a typed
`AgentState`, and a `trace` field that the UI renders as "Show reasoning" (plan, tools selected, execution, self-check).
A LangGraph checkpointer keyed by `thread_id` gives multi-turn memory; the planner resets per-turn fields so follow-ups
("and for Central vs East?") work without state leaking.

**Tools are structured, not free-form.** `query_data`, `analyze_stats`, `create_chart`, `forecast_sales` and `web_search`
take typed arguments and run vetted pandas/statsmodels code; errors return messages listing valid values so the executor
can self-correct. I deliberately dropped a free-form `run_python` tool: running model-written code on a public server needs
a real sandbox, and the structured tools cover the brief. The trade-off is less flexibility for unusual questions - those
route to `unsupported` instead of improvising.

**Honesty.** The data card states what the dataset lacks (profit, quantity, discount), so "what was our profit margin?"
routes to `unsupported` with alternatives. Relative dates resolve against the dataset's latest date, and the answer says so.

**Verification & evaluation.** The critic makes "numbers only from tool output" enforceable: every figure in the answer must
match an observation or a simple derivation (gap, share, ratio); it also rejects generic takeaways and unsupported
recommendations, flags overclaiming about anomalies, and checks forecast horizon and dates - all in Python; an LLM reviewer
is optional. `evals/` scores the agent's *decisions* (route and tool choice on
19 labelled questions, plus pandas ground-truth figures), and the harness excludes API failures from scores so a rate limit
is never mistaken for a wrong decision. Offline tests drive the real graph with a scripted fake LLM.

**Production touches.** Conversation memory is checkpointed per chat and can be saved to SQLite (`CHECKPOINT_DB`); a page refresh restores the
transcript. Reasoning steps stream live and the verified answer is revealed progressively. Every question writes a structured record
(route, tools, self-check, latency) to a JSONL log and a sidebar metrics panel; LangSmith tracing works via environment variables.

**Trade-offs.** The extra planner and critic calls add latency and free-tier token pressure; mitigations are a Python-only critic by default, tool schemas
limited to the planner's chosen tools, and an automatic fallback model on errors. The critic checks figures, not causality.

**With more time.** Grow the eval set to ~40 questions with numeric ground truth and run it in CI; make the critic check
dates and causal wording; stream the final answer; persist checkpoints (SQLite/Postgres); extend to the relational Olist
dataset with a SQL tool and a properly sandboxed code tool.
