"""Plotly figure builders. Figures are returned as JSON strings so they can travel through LangGraph state."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .analytics import METRICS


def _style(fig, title: str):
    fig.update_layout(title=title, template="plotly_white", margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
    return fig


def from_aggregate(res: pd.DataFrame, chart_type: str, metric: str, title: str) -> str:
    dims = [c for c in res.columns if c not in METRICS + ["Period"]]
    label = metric.replace("_", " ").title()
    if chart_type == "line":
        if "Period" not in res.columns:
            raise ValueError("line charts need a time_grain (month/quarter/year).")
        fig = px.line(res, x="Period", y=metric, color=dims[0] if dims else None, markers=True,
                      labels={metric: label})
    elif chart_type == "pie":
        if not dims:
            raise ValueError("pie charts need a group_by dimension.")
        fig = px.pie(res, names=dims[0], values=metric)
    else:  # bar
        x = dims[0] if dims else "Period"
        if x not in res.columns:
            raise ValueError("bar charts need a group_by dimension or a time_grain.")
        color = dims[1] if len(dims) > 1 else (dims[0] if dims and "Period" in res.columns else None)
        horizontal = res[x].nunique() > 8 and "Period" not in res.columns
        fig = (px.bar(res, y=x, x=metric, color=color, orientation="h", labels={metric: label})
               if horizontal else px.bar(res, x=x, y=metric, color=color, barmode="group", labels={metric: label}))
        if horizontal:
            fig.update_yaxes(categoryorder="total ascending")
    return _style(fig, title).to_json()


def seasonality_heatmap(df: pd.DataFrame, title: str) -> str:
    pv = (df.assign(Year=df["Order Date"].dt.year, Month=df["Order Date"].dt.month)
          .pivot_table(index="Year", columns="Month", values="Sales", aggfunc="sum", fill_value=0))
    pv = pv.reindex(columns=range(1, 13), fill_value=0)
    names = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pv.columns]
    fig = go.Figure(go.Heatmap(z=pv.values, x=names, y=[str(y) for y in pv.index], colorscale="Blues",
                               colorbar=dict(title="Sales")))
    return _style(fig, title).to_json()


def forecast_chart(fc: dict, title: str) -> str:
    fig = go.Figure()
    fig.add_scatter(x=fc["_history"]["x"], y=fc["_history"]["y"], mode="lines", name="History")
    hx, hy = fc["_history"]["x"][-1], fc["_history"]["y"][-1]
    fig.add_scatter(x=[hx] + fc["_forecast"]["x"], y=[hy] + fc["_forecast"]["y"], mode="lines+markers",
                    name="Forecast", line=dict(dash="dash"))
    return _style(fig, title).to_json()
