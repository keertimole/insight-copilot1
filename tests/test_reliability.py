"""Offline tests for the reliability fixes: chart wording, anomaly caution, forecast validation and its checks.

Pure pandas/Python - no LLM, no LangGraph needed.
"""
import json

from insight_copilot import analytics as A
from insight_copilot.critic import check_answer, forecast_note, stated_horizons
from insight_copilot.data import load_df

ANSWER = ("**Answer:** Furniture sales are strongly seasonal, peaking in December at index 200.9.\n"
          "**Supporting numbers:**\n- December index: 200.9\n- February index: 24.8\n"
          "**Why it matters:** Q4 averages an index of 164.1, more than triple the Q1 average of 51.9, so the "
          "category depends on year-end volume.")


def _obs(payload: dict, tool: str = "analyze_stats") -> list[dict]:
    return [{"tool": tool, "args": {}, "output": json.dumps(payload)}]


SEAS = _obs({"seasonal_index": {"Dec": 200.9, "Feb": 24.8}, "q4_avg_index": 164.1, "q1_avg_index": 51.9})


def test_chart_reference_accepts_natural_wording():
    # This exact phrasing (no word 'chart' or 'plot') used to fail the self-check three times in a row.
    natural = ANSWER.replace("Furniture sales are", "The monthly sales trend is shown below; Furniture sales are")
    assert check_answer(natural, SEAS, n_charts=1).passed
    assert check_answer(ANSWER + " The chart below shows the monthly pattern.", SEAS, n_charts=1).passed


def test_chart_reference_still_required_when_a_chart_exists():
    res = check_answer(ANSWER, SEAS, n_charts=1)
    assert not res.passed and any("chart" in i.lower() for i in res.issues)
    assert check_answer(ANSWER, SEAS, n_charts=0).passed          # no chart made -> nothing to mention


def test_structural_claims_flagged_only_for_anomaly_output():
    anomalies = _obs({"anomalous_months": [{"month": "2018-11", "sales": 118447.5}], "median_order_value": 100.0})
    bad = check_answer("**Answer:** Sales spiked in 2018-11 to 118447.5.\n**Supporting numbers:**\n- Nov 2018: 118447.5\n"
                       "**Why it matters:** This reflects a structural shift in sales volume during 2018 versus earlier years.",
                       anomalies)
    assert not bad.passed and any("statistical" in i for i in bad.issues)
    ok = check_answer("**Answer:** Sales spiked in 2018-11 to 118447.5.\n**Supporting numbers:**\n- Nov 2018: 118447.5\n"
                      "**Why it matters:** November 2018 is unusually high versus its seasonal baseline; the cause needs "
                      "further investigation.", anomalies)
    assert ok.passed, ok.issues


def test_forecast_output_has_validation_and_dataset_relative_dates():
    fc = A.forecast(load_df(), 6)
    assert fc["history_ends"] == "2018-12" and fc["forecast_period"] == "2019-01 to 2019-06"
    assert "not relative to today" in fc["forecast_basis"]
    assert fc["backtest_window_months"] == 6
    assert fc["backtest_mape_pct_last_6_months"] is not None and fc["backtest_mae_last_6_months"] > 0
    assert fc["baseline_seasonal_naive_mape_pct"] is not None
    assert isinstance(fc["beats_seasonal_naive_baseline"], bool)
    assert fc["caveat"] and len(fc["forecast"]) == 6


def _fc_obs(h=6):
    fc = A.forecast(load_df(), h)
    return _obs({k: v for k, v in fc.items() if not k.startswith("_")}, "forecast_sales"), fc


def test_forecast_checks_horizon_and_period():
    obs, fc = _fc_obs(6)
    total = f"{fc['forecast_total']:,.2f}"
    good = (f"**Answer:** The forecast for the next 6 months (Jan-Jun 2019) totals ${total}.\n**Supporting numbers:**\n"
            f"- Forecast total: ${total}\n- Backtest MAPE: {fc['backtest_mape_pct_last_6_months']}%\n"
            f"**Why it matters:** The forecast is {fc['pct_vs_prior_year']}% above the same months of 2018, "
            f"concentrated in the seasonal peak months.")
    res = check_answer(good, obs)
    assert res.passed, res.issues
    wrong_h = good.replace("next 6 months", "next 12 months")
    assert any("horizon" in i for i in check_answer(wrong_h, obs).issues)
    no_period = good.replace(" (Jan-Jun 2019)", "")
    assert any("months the forecast covers" in i for i in check_answer(no_period, obs).issues)


def test_stated_horizons_parsing():
    assert stated_horizons("Forecast for the next six months") == [6]
    assert stated_horizons("a 12-month forecast") == [12]
    assert stated_horizons("MAPE over the last 6 months") == []          # backtest window is not a horizon claim


def test_forecast_note_is_code_written_and_complete():
    obs, fc = _fc_obs(6)
    note = forecast_note(obs)
    assert fc["method"] in note and "2019-01 to 2019-06" in note and "not relative to today" in note
    assert "estimate" in note and "seasonal-naive baseline" in note
    assert forecast_note(SEAS) is None


def test_checks_list_is_exposed_for_the_ui():
    res = check_answer(ANSWER, SEAS)
    labels = [c["label"] for c in res.as_dict()["checks"]]
    assert any("verified against tool output" in l for l in labels)
    assert all(c["ok"] for c in res.checks)


# ---- regression: K/M/B abbreviations must be checked against the rounding the abbreviation implies
_KM_OBS = [{"tool": "analyze_stats", "args": {}, "output":
            '{"a_sales": 725457.82, "b_sales": 391721.91, "gap": 333735.91, "gap_pct": 85.2}'}]
_KM_ANSWER = ("**Answer:** West was $725,457.82 vs South $391,721.91, a gap of $333,735.91 (85.2%).\n"
              "**Supporting numbers:** West {w}; South {s}; gap {g}.\n"
              "**Why it matters:** West leads South by 85.2%, with the gap concentrated in Technology rather than volume.")


def test_abbreviated_figures_that_round_correctly_are_grounded():
    for w, s, g in [("$725.5K", "$391.7K", "$333.7K"), ("$0.7M", "$0.4M", "$0.3M")]:
        r = check_answer(_KM_ANSWER.format(w=w, s=s, g=g), _KM_OBS)
        assert r.passed and not r.ungrounded, (w, r.issues)


def test_abbreviated_figures_that_are_wrong_are_still_flagged():
    for w, s, g, bad in [("$725.5K", "$412.3K", "$333.7K", "$412.3K"), ("$726.9K", "$391.7K", "$333.7K", "$726.9K")]:
        r = check_answer(_KM_ANSWER.format(w=w, s=s, g=g), _KM_OBS)
        assert not r.passed and bad in r.ungrounded


# ---- regression: answer labelled "2018 Q4" although the tool calls had no date filter (all-years totals)
def _period_answer(period_phrase: str) -> str:
    return (f"**Answer:** East generated $176,871.82 more in total sales than Central {period_phrase}.\n"
            "**Supporting numbers:**\n- East $669,518.73 vs Central $492,646.91\n"
            "**Why it matters:** East's lead of $176,871.82 is about 36% above Central's total, so the gap is large.")


_PERIOD_OBS = [{"tool": "query_data", "args": {"metric": "total_sales", "region": "East"}, "output": '{"value": 669518.73}'},
               {"tool": "query_data", "args": {"metric": "total_sales", "region": "Central"}, "output": '{"value": 492646.91, "gap": 176871.82}'}]


def test_quarter_label_without_date_filter_is_flagged():
    r = check_answer(_period_answer("in 2018 Q4"), _PERIOD_OBS)
    assert not r.passed and any("period" in i.lower() for i in r.issues)


def test_all_years_phrasing_and_real_date_filters_pass():
    assert not any("period" in i.lower() for i in check_answer(_period_answer("across all years (2015-2018)"), _PERIOD_OBS).issues)
    filtered = [{"tool": "query_data", "args": {"start_date": "2018-10-01", "end_date": "2018-12-31"}, "output": _PERIOD_OBS[0]["output"]}]
    assert not any("period" in i.lower() for i in check_answer(_period_answer("in Q4 2018"), filtered).issues)
