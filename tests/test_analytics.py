"""Offline tests (no LLM / API key needed). Run: pytest -q"""
import pytest

from insight_copilot import analytics as A
from insight_copilot.data import load_df


@pytest.fixture(scope="module")
def df():
    return load_df()


def test_top_n_query_is_sorted_and_limited(df):
    out = A.query(df, {}, "total_sales", "sub_category", "none", 3, "desc")
    vals = [r["total_sales"] for r in out["result"]]
    assert len(vals) == 3 and vals == sorted(vals, reverse=True)


def test_filters_case_insensitive_and_helpful_error(df):
    out = A.query(df, {"region": "west"}, "total_sales", "", "none", 5, "desc")
    assert out["filters_applied"]["region"] == "West"
    with pytest.raises(ValueError, match="Valid Region values"):
        A.query(df, {"region": "Atlantis"}, "total_sales", "", "none", 5, "desc")


def test_time_grain_sorted_by_period(df):
    out = A.query(df, {}, "total_sales", "", "quarter", 10, "desc")
    periods = [r["Period"] for r in out["result"]]
    assert periods == sorted(periods) and len(periods) >= 8


def test_growth_and_trend(df):
    g = A.analyze(df, "growth_rate", time_grain="year")
    assert g["overall_change_pct_first_to_last"] is not None
    t = A.analyze(df, "trend")
    assert t["direction"] in ("upward", "downward") and "yearly_totals" in t


def test_seasonality_has_12_months(df):
    s = A.analyze(df, "seasonality")
    assert len(s["seasonal_index_100_is_average_month"]) == 12 and s["peak_months"]


def test_anomalies_structure(df):
    a = A.analyze(df, "anomalies")
    assert "anomalous_months" in a and "outlier_orders" in a


def test_compare_groups_gap_adds_up(df):
    c = A.analyze(df, "compare_groups", group_by="region", groups="West,South", driver_dim="category")
    assert c["compared"] == ["West", "South"]
    assert abs(sum(d["gap"] for d in c["gap_drivers"]) - c["gap_usd_a_minus_b"]) < 1.0


def test_share_sums_to_100(df):
    s = A.analyze(df, "share_of_total", group_by="category")
    assert abs(sum(r["share_pct"] for r in s["table"]) - 100) < 0.1


def test_forecast_shape(df):
    f = A.forecast(df, 6)
    assert len(f["forecast"]) == 6 and f["forecast_total"] > 0


def test_bad_inputs_raise_readable_errors(df):
    with pytest.raises(ValueError, match="Unknown dimension"):
        A.aggregate(df, "total_sales", "profit")
    with pytest.raises(ValueError, match="Unknown metric"):
        A.aggregate(df, "profit")
