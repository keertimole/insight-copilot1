"""Pure-pandas analytics used by the agent's tools. No LLM / LangChain imports -> easy to unit test.

Every public function returns JSON-serialisable dicts (or raises ValueError with a helpful message
that the agent can read and use to self-correct).
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

from .data import DIMS

FILTER_COLS = {"region": "Region", "category": "Category", "sub_category": "Sub-Category",
               "segment": "Segment", "state": "State"}
METRICS = ["total_sales", "order_count", "avg_order_value", "customer_count", "line_items", "avg_line_sale"]
GRAINS = {"month": "M", "quarter": "Q", "year": "Y"}


# ----------------------------------------------------------------------------- helpers
def to_json(obj) -> str:
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o), ensure_ascii=False)


def records(df: pd.DataFrame, nd: int = 2) -> list[dict]:
    return json.loads(df.round(nd).to_json(orient="records"))


def apply_filters(df: pd.DataFrame, region="", category="", sub_category="", segment="", state="",
                  start_date="", end_date="") -> tuple[pd.DataFrame, dict]:
    """Case-insensitive exact-match filters. Raises ValueError listing valid values on a miss."""
    applied: dict = {}
    vals = dict(region=region, category=category, sub_category=sub_category, segment=segment, state=state)
    for key, val in vals.items():
        val = (val or "").strip()
        if not val:
            continue
        col = FILTER_COLS[key]
        mask = df[col].str.lower() == val.lower()
        if not mask.any():
            opts = sorted(df[col].unique())
            raise ValueError(f"No rows where {col} = '{val}'. Valid {col} values: {opts[:30]}")
        df, applied[key] = df[mask], df.loc[mask, col].iloc[0]
    for key, val, op in (("start_date", start_date, ">="), ("end_date", end_date, "<=")):
        val = (val or "").strip()
        if val:
            ts = pd.to_datetime(val, errors="coerce")
            if pd.isna(ts):
                raise ValueError(f"Could not parse {key}='{val}'. Use YYYY-MM-DD.")
            df = df[df["Order Date"] >= ts] if op == ">=" else df[df["Order Date"] <= ts]
            applied[key] = str(ts.date())
    if df.empty:
        raise ValueError(f"No rows match filters {applied}. The data spans "
                         f"{df['Order Date'].min() if len(df) else 'n/a'} - check dates/values.")
    return df, applied


def parse_dims(csv: str) -> list[str]:
    out = []
    for token in (csv or "").replace(";", ",").split(","):
        t = token.strip().lower().replace(" ", "_").replace("-", "_")
        if t:
            if t not in DIMS:
                raise ValueError(f"Unknown dimension '{token}'. Valid: {sorted(DIMS)}")
            out.append(DIMS[t])
    return out


def period_label(df: pd.DataFrame, grain: str) -> pd.Series:
    if grain not in GRAINS:
        raise ValueError(f"time_grain must be one of {list(GRAINS)} (or 'none')")
    return df["Order Date"].dt.to_period(GRAINS[grain]).astype(str)


def monthly(df: pd.DataFrame) -> pd.Series:
    """Total sales per calendar month with gaps filled by 0 (PeriodIndex)."""
    s = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
    return s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"), fill_value=0.0)


def _agg(g) -> pd.DataFrame:
    out = g.agg(total_sales=("Sales", "sum"), order_count=("Order ID", "nunique"),
                customer_count=("Customer ID", "nunique"), line_items=("Row ID", "count"))
    out["avg_order_value"] = out["total_sales"] / out["order_count"]
    out["avg_line_sale"] = out["total_sales"] / out["line_items"]
    return out


# ----------------------------------------------------------------------------- query
def aggregate(df: pd.DataFrame, metric: str = "total_sales", group_by: str = "", time_grain: str = "none",
              top_n: int = 10, sort: str = "desc") -> pd.DataFrame:
    """Group + aggregate. Returns a flat DataFrame with the requested metric as the last column."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Valid: {METRICS}")
    dims = parse_dims(group_by)
    work = df.copy()
    keys = list(dims)
    if time_grain and time_grain != "none":
        work["Period"] = period_label(work, time_grain)
        keys = ["Period"] + keys
    if not keys:
        work["_all"] = "all"
        keys = ["_all"]
    res = _agg(work.groupby(keys)).reset_index()
    if keys == ["_all"]:
        res = res.drop(columns="_all")
        return res[[c for c in METRICS if c != metric] + [metric]]
    if "Period" in keys:
        res = res.sort_values(keys)
    else:
        res = res.sort_values(metric, ascending=(sort == "asc")).head(max(1, int(top_n)))
    return res[keys + [c for c in METRICS if c != metric] + [metric]]


def query(df: pd.DataFrame, filters: dict, metric: str, group_by: str, time_grain: str, top_n: int, sort: str,
          max_rows: int = 120) -> dict:
    fdf, applied = apply_filters(df, **filters)
    res = aggregate(fdf, metric, group_by, time_grain, top_n, sort)
    truncated = len(res) > max_rows
    return {"filters_applied": applied, "rows_matched": int(len(fdf)), "metric": metric,
            "overall_total_sales_in_filter": round(float(fdf["Sales"].sum()), 2),
            "result": records(res.head(max_rows)), "truncated": truncated}


# ----------------------------------------------------------------------------- stats
def _series(df: pd.DataFrame, grain: str) -> pd.Series:
    if grain == "none":
        grain = "year"
    s = df.groupby(df["Order Date"].dt.to_period(GRAINS[grain]))["Sales"].sum()
    return s.reindex(pd.period_range(s.index.min(), s.index.max(), freq=GRAINS[grain]), fill_value=0.0)


def growth(df: pd.DataFrame, grain: str = "year") -> dict:
    grain = "year" if grain == "none" else grain
    s = _series(df, grain)
    out = pd.DataFrame({"period": s.index.astype(str), "sales": s.values,
                        "pct_change_vs_previous_period": (s.pct_change() * 100).values})
    lag = {"month": 12, "quarter": 4, "year": 1}[grain]
    if lag > 1:
        out["pct_change_vs_same_period_last_year"] = (s.pct_change(lag) * 100).values
    out = out.replace([np.inf, -np.inf], np.nan)
    note = None
    if grain == "year":
        cnt = df.groupby(df["Order Date"].dt.year)["Order Date"].apply(lambda x: x.dt.month.nunique())
        partial = {int(y): int(c) for y, c in cnt.items() if c < 12}
        note = f"Partial years (months of data): {partial}" if partial else None
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    return {"grain": grain, "table": records(out.tail(24)), "first_period": str(s.index[0]),
            "last_period": str(s.index[-1]), "overall_change_pct_first_to_last":
                round((last / first - 1) * 100, 2) if first else None, "note": note}


def trend(df: pd.DataFrame) -> dict:
    m = monthly(df)
    if len(m) < 6:
        raise ValueError("Need at least 6 months of data for a trend.")
    x = np.arange(len(m))
    slope, intercept = np.polyfit(x, m.values, 1)
    fitted = slope * x + intercept
    ss_res, ss_tot = ((m.values - fitted) ** 2).sum(), ((m.values - m.values.mean()) ** 2).sum()
    yearly = m.groupby(m.index.year).sum()
    return {"months": len(m), "slope_usd_per_month": round(float(slope), 2),
            "slope_pct_of_avg_month_per_year": round(float(slope * 12 / m.mean() * 100), 2),
            "r_squared": round(float(1 - ss_res / ss_tot), 3) if ss_tot else None,
            "direction": "upward" if slope > 0 else "downward",
            "yearly_totals": {str(k): round(float(v), 2) for k, v in yearly.items()},
            "first_month": str(m.index[0]), "last_month": str(m.index[-1]),
            "note": "r_squared is low when seasonality dominates; a linear fit ignores seasonal swings."}


def seasonality_index(df: pd.DataFrame) -> pd.Series:
    """Index by calendar month (100 = average month), from years with all 12 months when possible."""
    m = monthly(df)
    complete = m.groupby(m.index.year).transform("size") == 12
    base = m[complete] if complete.any() else m
    ratio = base / base.groupby(base.index.year).transform("mean")
    idx = ratio.groupby(ratio.index.month).mean() * 100
    return idx.reindex(range(1, 13))


def seasonality(df: pd.DataFrame) -> dict:
    idx = seasonality_index(df).dropna()
    names = {i: pd.Timestamp(2000, i, 1).strftime("%b") for i in range(1, 13)}
    ranked = idx.sort_values(ascending=False)
    q = lambda ms: float(idx.reindex(ms).mean())
    return {"seasonal_index_100_is_average_month": {names[i]: round(float(v), 1) for i, v in idx.items()},
            "peak_months": [names[i] for i in ranked.index[:3]], "trough_months": [names[i] for i in ranked.index[-3:]],
            "peak_vs_trough_ratio": round(float(ranked.iloc[0] / ranked.iloc[-1]), 2),
            "q4_avg_index": round(q([10, 11, 12]), 1), "q1_avg_index": round(q([1, 2, 3]), 1),
            "strength": "strong" if ranked.iloc[0] - ranked.iloc[-1] > 60 else "moderate" if ranked.iloc[0] - ranked.iloc[-1] > 30 else "weak"}


def anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> dict:
    m = monthly(df)
    idx = (seasonality_index(df) / 100).fillna(1.0)
    adj = m / idx.reindex(m.index.month).values
    med = adj.median()
    mad = (adj - med).abs().median()
    z = 0.6745 * (adj - med) / mad if mad else adj * 0
    flagged = [{"month": str(p), "sales": round(float(m[p]), 2), "seasonally_adjusted_z": round(float(z[p]), 2),
                "direction": "spike" if z[p] > 0 else "dip"} for p in m.index if abs(z[p]) >= z_threshold]
    orders = df.groupby("Order ID").agg(total=("Sales", "sum"), date=("Order Date", "first"),
                                        customer=("Customer Name", "first"), state=("State", "first"))
    top_line = df.loc[df.groupby("Order ID")["Sales"].idxmax()].set_index("Order ID")["Product Name"]
    orders["top_product"] = top_line.reindex(orders.index)
    q1, q3 = orders["total"].quantile([.25, .75])
    fence = q3 + 3 * (q3 - q1)
    big = orders[orders["total"] > fence].sort_values("total", ascending=False).head(5).reset_index()
    big["date"] = big["date"].dt.strftime("%Y-%m-%d")
    return {"method": f"Monthly sales divided by seasonal index, then robust z-score (|z| >= {z_threshold}); "
                      f"single orders flagged above Q3 + 3*IQR = ${fence:,.0f}",
            "anomalous_months": flagged, "outlier_orders": records(big),
            "median_order_value": round(float(orders["total"].median()), 2),
            "interpretation_note": "These are statistical flags: months unusual relative to the seasonal baseline and "
                                   "unusually large single orders. They do not by themselves show a structural business "
                                   "change or its cause; that needs further investigation."}


def share(df: pd.DataFrame, dim: str, top_n: int = 15) -> dict:
    cols = parse_dims(dim)
    if len(cols) != 1:
        raise ValueError("share_of_total needs exactly one group_by dimension.")
    s = df.groupby(cols[0])["Sales"].sum().sort_values(ascending=False)
    out = pd.DataFrame({cols[0]: s.index, "sales": s.values, "share_pct": s.values / s.sum() * 100})
    top3 = float(out["share_pct"].head(3).sum())
    return {"total_sales": round(float(s.sum()), 2), "table": records(out.head(top_n)),
            "top3_share_pct": round(top3, 1), "n_groups": int(len(s))}


def compare(df: pd.DataFrame, dim: str, groups: str = "", driver_dim: str = "category") -> dict:
    cols = parse_dims(dim)
    if len(cols) != 1:
        raise ValueError("compare_groups needs exactly one group_by dimension (e.g. 'region').")
    col = cols[0]
    dcols = parse_dims(driver_dim or "category")
    dcol = dcols[0] if dcols[0] != col else ("Sub-Category" if col != "Sub-Category" else "Category")
    g = _agg(df.groupby(col)).sort_values("total_sales", ascending=False)
    wanted = [x.strip() for x in (groups or "").split(",") if x.strip()]
    if wanted:
        lookup = {str(k).lower(): k for k in g.index}
        miss = [w for w in wanted if w.lower() not in lookup]
        if miss:
            raise ValueError(f"Unknown {col} value(s) {miss}. Valid: {list(g.index)[:30]}")
        sel = [lookup[w.lower()] for w in wanted]
    else:
        sel = list(g.index[:2])
    if len(sel) < 2:
        raise ValueError("Need at least two groups to compare.")
    a, b = sel[0], sel[1]
    pv = df[df[col].isin([a, b])].pivot_table(index=dcol, columns=col, values="Sales", aggfunc="sum", fill_value=0)
    pv["gap"] = pv[a] - pv[b]
    total_gap = float(pv["gap"].sum())
    pv["pct_of_total_gap"] = pv["gap"] / total_gap * 100 if total_gap else np.nan
    pv = pv.reindex(pv["gap"].abs().sort_values(ascending=False).index).reset_index().head(8)
    yearly = df[df[col].isin([a, b])].groupby([df["Order Date"].dt.year.rename("year"), col])["Sales"].sum().unstack()
    ga, gb = g.loc[a], g.loc[b]
    return {"dimension": col, "compared": [a, b], "totals": records(g.loc[sel].reset_index()),
            "gap_usd_a_minus_b": round(total_gap, 2),
            "gap_pct_of_b": round(total_gap / float(gb.total_sales) * 100, 1) if gb.total_sales else None,
            "orders_a_vs_b_pct": round((ga.order_count / gb.order_count - 1) * 100, 1),
            "avg_order_value_a_vs_b_pct": round((ga.avg_order_value / gb.avg_order_value - 1) * 100, 1),
            "gap_drivers_by": dcol, "gap_drivers": records(pv), "yearly_sales": records(yearly.reset_index())}


def summary(df: pd.DataFrame) -> dict:
    yearly = df.groupby(df["Order Date"].dt.year)["Sales"].sum()
    top = lambda c: (lambda s: {"name": s.index[0], "share_pct": round(float(s.iloc[0] / s.sum() * 100), 1)})(
        df.groupby(c)["Sales"].sum().sort_values(ascending=False))
    orders = df.groupby("Order ID")["Sales"].sum()
    return {"date_range": [str(df["Order Date"].min().date()), str(df["Order Date"].max().date())],
            "total_sales": round(float(df["Sales"].sum()), 2), "orders": int(orders.size),
            "customers": int(df["Customer ID"].nunique()), "avg_order_value": round(float(orders.mean()), 2),
            "median_order_value": round(float(orders.median()), 2),
            "yearly_sales": {str(k): round(float(v), 2) for k, v in yearly.items()},
            "top_region": top("Region"), "top_category": top("Category"), "top_sub_category": top("Sub-Category"),
            "top_state": top("State"), "top_segment": top("Segment")}


ANALYSES = ["summary", "growth_rate", "trend", "seasonality", "anomalies", "share_of_total", "compare_groups"]


def analyze(df: pd.DataFrame, analysis: str, group_by: str = "", groups: str = "", driver_dim: str = "category",
            time_grain: str = "none") -> dict:
    if analysis == "summary":
        return summary(df)
    if analysis == "growth_rate":
        return growth(df, time_grain)
    if analysis == "trend":
        return trend(df)
    if analysis == "seasonality":
        return seasonality(df)
    if analysis == "anomalies":
        return anomalies(df)
    if analysis == "share_of_total":
        return share(df, group_by)
    if analysis == "compare_groups":
        return compare(df, group_by, groups, driver_dim)
    raise ValueError(f"Unknown analysis '{analysis}'. Valid: {ANALYSES}")


# ----------------------------------------------------------------------------- forecast
def _fit_predict(train: pd.Series, h: int) -> tuple[np.ndarray, str]:
    y = train.values.astype(float)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(y) >= 24 else None
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=seasonal,
                                   seasonal_periods=12 if seasonal else None,
                                   initialization_method="estimated").fit()
        return np.clip(fit.forecast(h), 0, None), "Holt-Winters (damped additive trend" + (" + seasonality)" if seasonal else ")")
    except Exception:  # statsmodels missing or fit failed -> seasonal-naive fallback
        last = y[-12:] if len(y) >= 12 else y
        return np.array([last[i % len(last)] for i in range(h)]), "Seasonal naive (repeat last 12 months)"


def forecast(df: pd.DataFrame, horizon: int = 6) -> dict:
    """Holt-Winters forecast with a hold-out backtest (last 6 months) compared against a seasonal-naive baseline.

    Dates are RELATIVE TO THE DATASET: the forecast covers the months after the last month in the data.
    """
    m = monthly(df)
    if len(m) < 12:
        raise ValueError("Need at least 12 months of history to forecast.")
    h = max(1, min(int(horizon), 24))
    mape = mae = base_mape = None
    if len(m) >= 30:                                   # train on everything except the last 6 months, forecast them
        pred, _ = _fit_predict(m.iloc[:-6], 6)
        act = m.iloc[-6:].values
        ok = act > 0
        mape = round(float(np.mean(np.abs(act[ok] - pred[ok]) / act[ok]) * 100), 1) if ok.any() else None
        mae = round(float(np.mean(np.abs(act - pred))), 2)
        naive = m.iloc[-18:-12].values                 # baseline: the same 6 months one year earlier
        base_mape = round(float(np.mean(np.abs(act[ok] - naive[ok]) / act[ok]) * 100), 1) if ok.any() else None
    fc, method = _fit_predict(m, h)
    fut = pd.period_range(m.index[-1] + 1, periods=h, freq="M")
    prior = m.reindex(fut - 12).dropna()
    fc_total = float(fc.sum())
    return {"method": method, "horizon_months": h,
            "history_ends": str(m.index[-1]), "history_last_month": str(m.index[-1]), "training_months": int(len(m)),
            "forecast_period": f"{fut[0]} to {fut[-1]}",
            "forecast_basis": "The months following the latest month in the dataset (not relative to today's date).",
            "backtest_window_months": 6 if mape is not None else None,
            "backtest_mape_pct_last_6_months": mape, "backtest_mae_last_6_months": mae,
            "baseline_seasonal_naive_mape_pct": base_mape,
            "beats_seasonal_naive_baseline": (mape < base_mape) if mape is not None and base_mape is not None else None,
            "caveat": "Estimate from historical patterns, not a guarantee. One backtest window does not bound future error.",
            "forecast": [{"month": str(p), "forecast_sales": round(float(v), 2)} for p, v in zip(fut, fc)],
            "forecast_total": round(fc_total, 2),
            "same_months_prior_year_total": round(float(prior.sum()), 2) if len(prior) == h else None,
            "pct_vs_prior_year": round((fc_total / float(prior.sum()) - 1) * 100, 1) if len(prior) == h and prior.sum() else None,
            "_history": {"x": [str(p) for p in m.index], "y": [round(float(v), 2) for v in m.values]},
            "_forecast": {"x": [str(p) for p in fut], "y": [round(float(v), 2) for v in fc]}}
