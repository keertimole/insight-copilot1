# Assignment requirement checklist

Maps every requirement in the brief to where it is met in this repo.

| # | Requirement (brief section) | Where / evidence |
|---|---|---|
| 1 | Pick one dataset from the shortlist (§3-4) | Superstore Sales (time-series) - `data/train.csv`, 9,800 rows, 2015-2018 |
| 2 | LangGraph `StateGraph`, not a bare AgentExecutor (§5.1) | `insight_copilot/graph.py` -> `build_graph()` |
| 3 | At least one conditional edge (§5.1) | 3: `route_after_planner`, `route_after_executor`, `route_after_critic` |
| 4 | Typed state object carried through the graph (§5.1) | `AgentState` (TypedDict) in `graph.py` |
| 5 | Plan/rationale before acting (§5.2) | `planner` node -> structured `Plan` (goal, reasoning, steps, tools, route) |
| 6 | Reasoning visible in the UI (§5.2) | `state["trace"]` rendered live + in the "Show reasoning" panel (`app.py`) |
| 7 | At least 3 tools, chosen per query (§5.3) | 5 tools in `tools.py`: `query_data`, `analyze_stats`, `create_chart`, `forecast_sales`, `web_search` |
| 8 | Not always calling every tool (§5.3) | Planner picks tools; `direct`/`clarify`/`unsupported` routes call none; checked by `evals/` |
| 9 | Bonus: multi-tool sequences (§5.3) | `executor <-> tool_node` loop, e.g. `analyze_stats` -> `create_chart` |
| 10 | Bonus: "I don't have a tool for that" (§5.3) | `unsupported` route (e.g. profit margin - not in the data) |
| 11 | Insight, not raw dumps (§5.4) | `synthesizer` prompt: Answer -> Supporting numbers -> Why it matters; `critic` verifies figures |
| 12 | Multi-turn chat (§5.5) | `MemorySaver` checkpointer keyed by `thread_id`; history passed to planner/executor/synthesizer |
| 13 | Chat UI (§5.5) | Streamlit - `app.py`: type a question or pick one from the grouped sidebar; collapsible **Show reasoning** panel (optional auto-open) |
| 14 | Public hosted URL (§7-8) | **TODO (you):** deploy on Streamlit Community Cloud, paste URL in README |
| 15 | GitHub repo with dependency file (§7) | **TODO (you):** push; `requirements.txt` included |
| 16 | No API keys committed (§8) | `.gitignore`, `.env.example`, `secrets.toml.example`, `tests/test_repo_hygiene.py` |
| 17 | Mention free-tier wake-up (§8) | README top note |
| 18 | README: setup, architecture, tools, limitations, assumptions (§7) | `README.md` |
| 19 | Architecture diagram (§7) | Mermaid in `README.md` and `docs/graph.mmd` |
| 20 | Short write-up (§7) | `docs/writeup.md` (paste into the submission email) |
