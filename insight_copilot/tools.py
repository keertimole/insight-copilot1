"""The agent's toolbox (5 tools). Thin LangChain wrappers around analytics.py / charts.py.

Design notes
- All args are flat primitives with safe defaults (no Optional/null) - much more reliable for open-source
  tool-calling models. A `None` sent by the model is dropped by `Args._drop_none`, so defaults apply.
- Tools never raise: errors come back as "ERROR: ..." strings that name the valid options, so the
  executor LLM can read them and self-correct on the next step.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from . import analytics as A
from . import charts
from .data import load_df

Grain = Literal["none", "month", "quarter", "year"]


class Args(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _drop_none(cls, data):
        return {k: v for k, v in data.items() if v is not None} if isinstance(data, dict) else data


class Filters(Args):
    region: str = Field("", description="Filter: East | West | Central | South. Empty = all regions.")
    category: str = Field("", description="Filter: Furniture | Office Supplies | Technology. Empty = all.")
    sub_category: str = Field("", description="Filter by sub-category, e.g. 'Chairs', 'Phones'. Empty = all.")
    segment: str = Field("", description="Filter: Consumer | Corporate | Home Office. Empty = all.")
    state: str = Field("", description="Filter by US state name, e.g. 'California'. Empty = all.")
    start_date: str = Field("", description="Inclusive start, YYYY-MM-DD (order date). Empty = no lower bound.")
    end_date: str = Field("", description="Inclusive end, YYYY-MM-DD (order date). Empty = no upper bound.")

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in Filters.model_fields}


class QueryArgs(Filters):
    metric: str = Field("total_sales", description=f"One of {A.METRICS}. avg_order_value = sales / distinct orders.")
    group_by: str = Field("", description="Comma-separated dimensions from: region, category, sub_category, segment, "
                                          "state, city, ship_mode, product_name, customer_name. Empty = overall total.")
    time_grain: Grain = Field("none", description="Also group by time period (month | quarter | year), or 'none'.")
    top_n: int = Field(10, description="Rows to keep when there is no time_grain (ranking).")
    sort: Literal["desc", "asc"] = Field("desc", description="desc = highest first (top N), asc = lowest first.")


class StatsArgs(Filters):
    analysis: Literal["summary", "growth_rate", "trend", "seasonality", "anomalies", "share_of_total",
                      "compare_groups"] = Field(..., description="Which statistical analysis to run.")
    group_by: str = Field("", description="ONE dimension; required for share_of_total and compare_groups "
                                          "(e.g. 'region').")
    groups: str = Field("", description="compare_groups only: comma-separated values to compare, e.g. 'West,South'. "
                                        "Empty = the top two by sales.")
    driver_dim: str = Field("category", description="compare_groups only: dimension used to explain the gap "
                                                    "(category | sub_category | segment | state ...).")
    time_grain: Grain = Field("none", description="growth_rate only: month | quarter | year (default year).")


class ChartArgs(QueryArgs):
    groups: str = Field("", description="Keep ONLY these values of the group_by dimension, comma-separated, e.g. 'West,South' "
                                        "when comparing two regions. Empty = all values.")
    chart_type: Literal["line", "bar", "pie", "seasonality_heatmap"] = Field(
        "bar", description="line = trend over time (needs time_grain); bar = compare groups; pie = share of total; "
                           "seasonality_heatmap = month x year grid of sales.")
    title: str = Field("", description="Chart title.")


class ForecastArgs(Filters):
    horizon_months: int = Field(6, description="Months to forecast beyond the last month in the data (1-24).")


class SearchArgs(Args):
    query: str = Field(..., description="Web search query for context OUTSIDE the dataset.")
    max_results: int = Field(4, description="Number of results (1-6).")


def _err(e: Exception) -> str:
    return f"ERROR: {e}" if isinstance(e, ValueError) else f"ERROR ({type(e).__name__}): {e}"


# ------------------------------------------------------------------------------------------------ tools
@tool("query_data", args_schema=QueryArgs)
def query_data(**kw) -> str:
    """Aggregate the sales data: rankings (top/bottom N), totals, breakdowns by dimension(s) and/or by
    month/quarter/year. Use for 'top 3 products by revenue', 'sales by region per year', 'how many orders'.
    Returns JSON rows. Not for growth %, seasonality, anomalies or comparisons (use analyze_stats)."""
    try:
        a = QueryArgs(**kw)
        return A.to_json(A.query(load_df(), a.as_dict(), a.metric, a.group_by, a.time_grain, a.top_n, a.sort))
    except Exception as e:  # noqa: BLE001
        return _err(e)


@tool("analyze_stats", args_schema=StatsArgs)
def analyze_stats(**kw) -> str:
    """Statistical analysis that produces insight-ready numbers: growth_rate (YoY / period change), trend
    (direction + slope), seasonality (monthly index, peak/trough), anomalies (unusual months + outlier orders),
    share_of_total, compare_groups (A vs B with a breakdown of what drives the gap), summary (dataset overview).
    Use for trend, seasonality, 'what's unusual', 'compare region A and B', growth questions."""
    try:
        a = StatsArgs(**kw)
        df, applied = A.apply_filters(load_df(), **a.as_dict())
        out = A.analyze(df, a.analysis, a.group_by, a.groups, a.driver_dim, a.time_grain)
        return A.to_json({"analysis": a.analysis, "filters_applied": applied, **out})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@tool("create_chart", args_schema=ChartArgs, response_format="content_and_artifact")
def create_chart(**kw):
    """Draw an interactive chart (line / bar / pie / seasonality heatmap) shown to the user under the answer.
    Use when the user asks to plot/show/visualize, or when a trend, seasonality or comparison is clearer visually.
    Takes the same aggregation arguments as query_data. Returns a text digest of the plotted data."""
    try:
        a = ChartArgs(**kw)
        df, applied = A.apply_filters(load_df(), **a.as_dict())
        scope = ", ".join(f"{v}" for v in applied.values()) or "all data"
        if a.chart_type == "seasonality_heatmap":
            title = a.title or f"Monthly sales heatmap ({scope})"
            return f"Heatmap created: {title}", charts.seasonality_heatmap(df, title)
        if a.groups.strip():
            col = A.parse_dims(a.group_by)[0] if a.group_by.strip() else None
            if col is None:
                raise ValueError("'groups' needs group_by (e.g. group_by='region', groups='West,South').")
            wanted = {g.strip().lower() for g in a.groups.split(",") if g.strip()}
            df = df[df[col].str.lower().isin(wanted)]
            if df.empty:
                raise ValueError(f"No rows where {col} is in {sorted(wanted)}. Check the spelling of the values.")
            scope = f"{scope}, {' vs '.join(sorted(df[col].unique()))}"
        grain = a.time_grain
        if a.chart_type == "line" and grain == "none":
            grain = "month"
        res = A.aggregate(df, a.metric, a.group_by, grain, a.top_n, a.sort)
        title = a.title or f"{a.metric.replace('_', ' ').title()} ({scope})"
        fig = charts.from_aggregate(res, a.chart_type, a.metric, title)
        return f"Chart created: {title}. Data plotted (first 30 rows): {A.to_json(A.records(res.head(30)))}", fig
    except Exception as e:  # noqa: BLE001
        return _err(e), None


@tool("forecast_sales", args_schema=ForecastArgs, response_format="content_and_artifact")
def forecast_sales(**kw):
    """Forecast future monthly sales (Holt-Winters with seasonality; reports a backtest error). Use for
    'what will sales look like next quarter / next 6 months'. Also draws a history+forecast chart."""
    try:
        a = ForecastArgs(**kw)
        df, applied = A.apply_filters(load_df(), **a.as_dict())
        fc = A.forecast(df, a.horizon_months)
        fig = charts.forecast_chart(fc, f"Sales forecast - next {fc['horizon_months']} months")
        digest = {k: v for k, v in fc.items() if not k.startswith("_")}
        return A.to_json({"filters_applied": applied, **digest}), fig
    except Exception as e:  # noqa: BLE001
        return _err(e), None


@tool("web_search", args_schema=SearchArgs)
def web_search(**kw) -> str:
    """Search the public web for context the dataset cannot provide (industry trends, holidays/events, economic
    background). Never use it for numbers that exist in the dataset."""
    try:
        a = SearchArgs(**kw)
        try:
            from ddgs import DDGS
        except ImportError:
            from ddgs import DDGS
        hits = list(DDGS().text(a.query, max_results=max(1, min(a.max_results, 6))))
        if not hits:
            return "No web results found."
        return A.to_json([{"title": h.get("title"), "snippet": (h.get("body") or "")[:300], "url": h.get("href")}
                          for h in hits])
    except Exception as e:  # noqa: BLE001
        return f"ERROR: web search unavailable ({type(e).__name__}). Answer from the dataset only and say so."


ALL_TOOLS = [query_data, analyze_stats, create_chart, forecast_sales, web_search]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
