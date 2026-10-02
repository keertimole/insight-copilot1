"""Run from the insight-copilot folder:  python scripts/diag_llm.py
Tests the primary and fallback models separately and prints the REAL error (no keys printed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from insight_copilot import config as c

print("primary :", c.provider(), c.model_name(), "| key set:", c.has_api_key())
spec = c.fallback_spec()
print("fallback:", spec if spec else "NONE (fallback is switched off - see why below)")
if not spec:
    print("  LLM_FALLBACK_PROVIDER =", c.secret("LLM_FALLBACK_PROVIDER"))
    print("  LLM_FALLBACK_MODEL    =", c.secret("LLM_FALLBACK_MODEL"))
    print("  GOOGLE_API_KEY set    =", bool(c.secret("GOOGLE_API_KEY")))

def probe(label, p, m):
    try:
        r = c._build(p, m, 0.0).invoke("Reply with the single word: ok")
        print(f"[OK]   {label} {p}/{m} ->", str(r.content)[:40])
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {label} {p}/{m} -> {type(e).__name__}: {str(e)[:300]}")

probe("primary ", c.provider(), c.model_name())
if spec:
    probe("fallback", *spec)
