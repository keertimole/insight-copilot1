"""Deterministic answer verification used by the graph's `critic` node. No LLM / LangChain imports -> unit-testable.

The critic asks two kinds of questions:
  1. Hard, checkable facts (this module): is every figure in the answer present in - or correctly derivable
     from - the tool observations? Does the reply have the required Answer / Supporting numbers / Why it matters
     parts? Does it mention the chart that was created? Is the takeaway more than a generic platitude?
  2. Soft judgement (an LLM call in graph.py): does it actually answer the question, and is the takeaway analytical?

A figure counts as *grounded* if it matches (within rounding) a number in the observations, OR is a simple
derivation of two of them (difference, sum, ratio, percentage, percentage change) - because the synthesizer is
allowed to compute gaps and shares.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_NUM = re.compile(r"(?<![\w.])(\$)?\s?(-?\d[\d,]*(?:\.\d+)?)\s?(%|[KkMmBb](?![a-zA-Z]))?")
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}

GENERIC_TAKEAWAYS = (
    "this is important", "important to monitor", "shows sales performance", "highlights the importance",
    "it is important to", "worth monitoring", "provides valuable insight", "valuable insights", "keep an eye on",
    "this shows how", "sales performance is important",
)

RECOMMENDATION_MARKERS = ("boosting", "boost ", "could close", "would close", "should focus", "you should", "we should",
                          "recommend", "consider increasing", "needs to")

# Wording that counts as "the answer points the reader at the chart" (models often say "shown below", not "chart").
CHART_PHRASES = ("chart", "plot", "graph", "visual", "figure", "diagram", "shown below", "displayed below",
                 "see below", "illustrated below", "pictured below", "shown above")

# Anomaly detection only flags statistically unusual months/orders; it cannot establish a structural business change.
STRUCTURAL_CLAIMS = ("structural shift", "structural change", "structural break", "structurally", "permanent shift",
                     "fundamental shift")

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12}
_NW = r"(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")"
_HORIZON_PATTERNS = (re.compile(r"\b(?:next|coming|following)\s+" + _NW + r"[- ]months?\b", re.I),
                     re.compile(r"\b" + _NW + r"[- ]months?\s+forecast\b", re.I))

MAX_DERIVATION_POOL = 300     # cap on observation numbers considered for pairwise derivations


@dataclass
class Figure:
    raw: str
    value: float
    decimals: int
    is_pct: bool
    is_money: bool
    unit: float = 1.0          # 1000 for "$725.5K": the last shown digit is worth `unit * 10**-decimals`


@dataclass
class CheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    n_figures: int = 0
    checks: list[dict] = field(default_factory=list)      # [{"label": ..., "ok": bool}] shown in the UI's self-check

    def as_dict(self) -> dict:
        return {"passed": self.passed, "issues": self.issues, "ungrounded": self.ungrounded, "n_figures": self.n_figures,
                "checks": self.checks}


# ---------------------------------------------------------------------------------------------- extraction
def extract_figures(text: str) -> list[Figure]:
    """Pull out the numbers a reader would treat as claims: $ amounts, percentages, decimals, big/comma numbers.
    Skips bare years (2015-2100), list numbering and small ranks/counts ("top 3", "4 regions")."""
    out: list[Figure] = []
    for m in _NUM.finditer(text):
        dollar, num, suffix = m.group(1), m.group(2), m.group(3)
        digits = num.replace(",", "")
        try:
            val = float(digits)
        except ValueError:
            continue
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        is_pct, is_money = suffix == "%", bool(dollar)
        unit = 1.0
        if suffix and suffix.lower() in _SUFFIX:
            unit = _SUFFIX[suffix.lower()]
            val *= unit
        plain_int = decimals == 0 and not is_pct and not is_money and not suffix and "," not in num
        if plain_int and (1900 <= abs(val) <= 2100 or abs(val) < 13):   # years, quarters, ranks, small counts
            continue
        if text[max(0, m.start() - 1):m.start()] == "Q":              # Q4 style labels
            continue
        out.append(Figure(m.group(0).strip(), val, decimals, is_pct, is_money, unit))
    return out


def observation_numbers(observations: list[dict]) -> list[float]:
    """Every number that appears in tool outputs or their arguments (dedup, stable order)."""
    seen: dict[float, None] = {}
    for o in observations:
        blob = f"{o.get('output', '')} {o.get('args', '')}"
        for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", blob):
            try:
                seen.setdefault(float(m.group(0).replace(",", "")), None)
            except ValueError:
                pass
    return list(seen)


# ---------------------------------------------------------------------------------------------- grounding
def _close(x: float, v: float, decimals: int, unit: float = 1.0) -> bool:
    tol = 0.5 * 10 ** (-decimals) * unit + 1e-9      # "$725.5K" may stand for anything in 725,450-725,550
    if decimals == 0 and abs(v) >= 1000:               # "$12,345" vs 12345.67 - allow rounding to the unit / abbreviation
        tol = max(tol, 0.005 * abs(v))
    return abs(x - v) <= tol + 1e-9 * max(1.0, abs(v))


def _candidates(nums: list[float]) -> list[float]:
    pool = nums[:MAX_DERIVATION_POOL]
    cands: list[float] = []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            hi, lo = (a, b) if abs(a) >= abs(b) else (b, a)
            cands += [a - b, b - a, a + b]
            if lo:
                cands += [hi / lo, hi / lo * 100, (hi - lo) / abs(lo) * 100, lo / hi * 100 if hi else 0.0,
                          (lo - hi) / abs(hi) * 100 if hi else 0.0]
    return cands


def is_grounded(fig: Figure, nums: set[float] | list[float], derived: list[float] | None = None) -> bool:
    xs = [fig.value] + ([fig.value / 100] if fig.is_pct else [])
    for x in xs:
        if any(_close(x, v, fig.decimals + (2 if x is not fig.value else 0), fig.unit if x is fig.value else 1.0)
               for v in nums):
            return True
    if derived is not None:
        return any(_close(fig.value, d, fig.decimals, fig.unit) for d in derived)
    return False


def ungrounded_figures(answer: str, observations: list[dict]) -> tuple[list[Figure], int]:
    figs = extract_figures(answer)
    nums = observation_numbers(observations)
    derived = None
    bad: list[Figure] = []
    for f in figs:
        if is_grounded(f, nums):
            continue
        if derived is None:
            derived = _candidates(nums)
        if not is_grounded(f, [], derived):
            bad.append(f)
    return bad, len(figs)


# ---------------------------------------------------------------------------------------------- structure
def split_sections(answer: str) -> dict[str, str]:
    """Split the mandated Markdown format into its three parts (case-insensitive, tolerant of bold/colon variants)."""
    pat = re.compile(r"\*{0,2}\s*(Answer|Supporting numbers|Why it matters)\s*:?\s*\*{0,2}\s*:?", re.I)
    marks = [(m.start(), m.end(), m.group(1).lower()) for m in pat.finditer(answer)]
    parts: dict[str, str] = {}
    for i, (_, end, name) in enumerate(marks):
        nxt = marks[i + 1][0] if i + 1 < len(marks) else len(answer)
        parts.setdefault(name, answer[end:nxt].strip())
    return parts


def _mentions_chart(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in CHART_PHRASES)


def _obs_text(observations: list[dict]) -> str:
    return " ".join(str(o.get("output", "")) for o in observations)


def _as_int(tok: str) -> int | None:
    tok = tok.lower()
    return _NUMBER_WORDS.get(tok) if tok in _NUMBER_WORDS else (int(tok) if tok.isdigit() else None)


def forecast_facts(observations: list[dict]) -> dict | None:
    """Parsed forecast_sales output (or None if no forecast was made this turn)."""
    for o in observations:
        out = str(o.get("output", ""))
        if '"forecast_total"' not in out:
            continue
        try:
            return json.loads(out)
        except ValueError:                                    # clipped/invalid JSON: recover the two fields we need
            h = re.search(r'"horizon_months":\s*(\d+)', out)
            per = re.search(r'"forecast_period":\s*"([^"]+)"', out)
            return {"horizon_months": int(h.group(1)) if h else None, "forecast_period": per.group(1) if per else None}
    return None


def stated_horizons(answer: str) -> list[int]:
    """Forecast horizons the answer claims, e.g. 'next six months', '6-month forecast'."""
    found: list[int] = []
    for pat in _HORIZON_PATTERNS:
        for m in pat.finditer(answer):
            v = _as_int(m.group(1))
            if v is not None:
                found.append(v)
    return found


def forecast_note(observations: list[dict]) -> str | None:
    """Fixed, code-written method/validation/period disclaimer appended to forecast answers (no LLM tokens, cannot be
    forgotten or hallucinated)."""
    d = forecast_facts(observations)
    if not d:
        return None
    bits = [f"Method: {d.get('method', 'time-series model')}."]
    mape = d.get("backtest_mape_pct_last_6_months")
    if mape is not None:
        v = f"Validation: the last {d.get('backtest_window_months', 6)} months were held out; backtest MAPE {mape}%"
        if d.get("baseline_seasonal_naive_mape_pct") is not None:
            v += f" (seasonal-naive baseline {d['baseline_seasonal_naive_mape_pct']}%)"
        bits.append(v + ".")
    if d.get("forecast_period"):
        bits.append(f"Period covered: {d['forecast_period']} - the months after the dataset's last month "
                    f"({d.get('history_ends', 'end of data')}), not relative to today's date.")
    bits.append("This is an estimate from historical patterns, not a guarantee of future sales; a single backtest window "
                "does not bound future error.")
    return "*" + " ".join(bits) + "*"


# A quarter named in the answer ("2018 Q4", "Q4 2018", "last quarter", "fourth quarter") must correspond to a date filter
# or quarterly grouping that was actually applied; otherwise the model is mislabelling an all-years result.
_QUARTER_RE = re.compile(r"\bQ[1-4]\s*,?\s*20\d\d\b|\b20\d\d\s*,?\s*Q[1-4]\b|\b(?:last|latest|previous)\s+quarter\b", re.I)
# (bare "Q4" is NOT matched: "Q4 is the seasonal peak" is a legitimate pattern statement, not a period label)


def _unbacked_quarter_claim(answer: str, observations: list[dict]) -> str | None:
    m = _QUARTER_RE.search(answer)
    if not m:
        return None
    data_obs = [o for o in observations if o.get("tool") in ("query_data", "analyze_stats")]
    if not data_obs or any(o.get("tool") == "forecast_sales" for o in observations):
        return None
    for o in data_obs:
        a = o.get("args") or {}
        if a.get("start_date") or a.get("end_date") or a.get("time_grain") in ("quarter", "month"):
            return None
    return m.group(0)


def check_answer(answer: str, observations: list[dict], n_charts: int = 0) -> CheckResult:
    issues: list[str] = []
    checks: list[dict] = []
    parts = split_sections(answer)
    missing = [n for n in ("answer", "supporting numbers", "why it matters") if not parts.get(n)]
    for name in missing:
        issues.append(f"Missing or empty section: '{name}'. Use exactly: Answer / Supporting numbers / Why it matters.")
    checks.append({"label": "Answer / Supporting numbers / Why it matters all present", "ok": not missing})
    takeaway = parts.get("why it matters", "")
    generic = bool(takeaway) and (any(g in takeaway.lower() for g in GENERIC_TAKEAWAYS) or len(takeaway) < 40)
    if generic:
        issues.append("'Why it matters' is generic. State what specifically stands out (a driver, concentration, "
                      "trend or anomaly) with a concrete figure or named segment.")
    recommends = bool(takeaway) and any(m in takeaway.lower() for m in RECOMMENDATION_MARKERS)
    if recommends:
        issues.append("'Why it matters' contains a recommendation / what-if that the data does not support. "
                      "Describe what the data shows instead (where the gap sits, its size, volume vs basket size).")
    checks.append({"label": "Takeaway is specific and data-based (no generic filler or unsupported advice)",
                   "ok": not (generic or recommends)})
    if n_charts:
        ok = _mentions_chart(answer)
        if not ok:
            issues.append("A chart was created but the answer does not mention it. Say 'the chart below shows ...'.")
        checks.append({"label": "Chart created and referenced in the answer", "ok": ok})

    bad_period = _unbacked_quarter_claim(answer, observations)
    if bad_period:
        issues.append(f"The answer refers to a period ('{bad_period}') but no date filter was applied to the data, so the "
                      "figures cover ALL years in the dataset. State the period exactly as queried (e.g. 'across 2015-2018').")
    checks.append({"label": "Period named in the answer matches the filters actually applied", "ok": not bad_period})

    otext = _obs_text(observations)
    if '"anomalous_months"' in otext:
        overclaim = [c for c in STRUCTURAL_CLAIMS if c in answer.lower()]
        if overclaim:
            issues.append("Anomaly flags are statistical (unusual vs the seasonal baseline). Do not claim a "
                          f"'{overclaim[0]}' or a business cause; say the cause needs further investigation.")
        checks.append({"label": "Anomalies described as statistical flags, not proven business changes", "ok": not overclaim})

    fc = forecast_facts(observations)
    if fc:
        h = fc.get("horizon_months")
        wrong = [v for v in stated_horizons(answer) if h is not None and v != h]
        if wrong:
            issues.append(f"The forecast horizon is {h} months but the answer says {wrong[0]}. Use the tool's horizon.")
        checks.append({"label": f"Forecast horizon matches the tool output ({h} months)", "ok": not wrong})
        per = str(fc.get("forecast_period") or "")
        year = per[:4] if per[:4].isdigit() else ""
        if year:
            ok = year in answer
            if not ok:
                issues.append(f"State the months the forecast covers (e.g. Jan-Jun {year}); they follow the last month "
                              "in the data, not today's date.")
            checks.append({"label": "Forecast period stated (relative to the data, not today)", "ok": ok})

    bad, n = ungrounded_figures(answer, observations)
    ungrounded = [f.raw for f in bad]
    if bad:
        issues.append("These figures are not in the tool observations and are not a simple derivation of them: "
                      + ", ".join(ungrounded) + ". Remove or correct them.")
    checks.insert(0, {"label": (f"{n} numerical claim(s) verified against tool output" if not bad
                                else f"{len(bad)} of {n} figure(s) could not be verified"), "ok": not bad})
    return CheckResult(passed=not issues, issues=issues, ungrounded=ungrounded, n_figures=n, checks=checks)
