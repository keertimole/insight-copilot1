"""LLM factory + secrets access. Keys come from env vars or Streamlit secrets - never from the repo."""
from __future__ import annotations

import os
import time
from functools import lru_cache

try:  # local development convenience
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULTS = {"groq": "openai/gpt-oss-120b", "google": "gemini-2.5-flash", "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5-20251001"}
# Second Groq model used when the main one errors. Groq rate limits are per model, so a sibling model is a free safety net.
# (llama-3.3-70b-versatile was the old default but is no longer available to every account - it returned HTTP 404.)
DEFAULT_GROQ_FALLBACK = "openai/gpt-oss-20b"
KEY_NAMES = {"groq": "GROQ_API_KEY", "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY",
             "anthropic": "ANTHROPIC_API_KEY"}


def secret(name: str, default: str | None = None) -> str | None:
    if os.getenv(name):
        return os.getenv(name)
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:  # noqa: BLE001 - no streamlit / no secrets file
        return default


def provider() -> str:
    return (secret("LLM_PROVIDER", "groq") or "groq").lower()


def model_name() -> str:
    return secret("LLM_MODEL", DEFAULTS.get(provider(), "")) or DEFAULTS[provider()]


def has_api_key() -> bool:
    return bool(secret(KEY_NAMES.get(provider(), "GROQ_API_KEY")))


def _build(p: str, m: str, temperature: float):
    if p == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=m, temperature=temperature, api_key=secret("GROQ_API_KEY"), max_retries=1)
    if p == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=m, temperature=temperature, google_api_key=secret("GOOGLE_API_KEY"))
    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=m, temperature=temperature, api_key=secret("OPENAI_API_KEY"))
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=m, temperature=temperature, api_key=secret("ANTHROPIC_API_KEY"))
    raise ValueError(f"Unsupported LLM_PROVIDER '{p}'. Use groq | google | openai | anthropic.")


def fallback_spec() -> tuple[str, str] | None:
    """(provider, model) to use when the primary model errors (e.g. free-tier rate limit), or None.
    Groq rate limits are per model, so a second Groq model is a zero-config safety net."""
    p, m = provider(), model_name()
    fp = (secret("LLM_FALLBACK_PROVIDER", "") or "").lower() or p
    fm = secret("LLM_FALLBACK_MODEL", "")
    if (fm or "").lower() in ("none", "off", "0", "false"):
        return None
    if not fm:
        if fp != p or p != "groq":
            return None
        fm = DEFAULT_GROQ_FALLBACK
    if (fp, fm) == (p, m) or not secret(KEY_NAMES.get(fp, ""), None):
        return None
    return fp, fm


RATE_LIMIT_WAIT_S = 12.0      # one patient retry of the primary when BOTH models fail on a rate limit (free-tier per-minute caps)


def _is_rate_limit(exc: Exception) -> bool:
    t = f"{type(exc).__name__} {exc}".lower()
    return "429" in t or "rate limit" in t or "ratelimit" in t or "rate_limit" in t


def _is_model_missing(exc: Exception) -> bool:
    t = f"{type(exc).__name__} {exc}".lower()
    return "404" in t or "notfound" in t or "does not exist" in t or "model_not_found" in t


class WithFallback:
    """Minimal LLM wrapper: try the primary chain, on ANY exception retry once on the backup chain.
    Supports the two builders the graph uses (`bind_tools`, `with_structured_output`) plus `invoke`.

    Error handling that keeps failures diagnosable:
      * if the backup model is missing for this account (404) it is switched off for the rest of the session;
      * if both models fail, the PRIMARY model's error is raised (the backup's error would hide the real cause);
      * if both failed on a rate limit, wait briefly and retry the primary once (free-tier limits are per minute)."""

    def __init__(self, primary, backup=None, ops: tuple = (), state: dict | None = None):
        self.primary, self.backup, self.ops = primary, backup, ops
        self.state = state if state is not None else {"backup_dead": False}

    def _chain(self, llm):
        for name, arg in self.ops:
            llm = getattr(llm, name)(arg)
        return llm

    def _clone(self, op):
        return WithFallback(self.primary, self.backup, self.ops + (op,), self.state)

    def bind_tools(self, tools):
        return self._clone(("bind_tools", tools))

    def with_structured_output(self, schema):
        return self._clone(("with_structured_output", schema))

    def invoke(self, messages):
        try:
            return self._chain(self.primary).invoke(messages)
        except Exception as primary_exc:  # noqa: BLE001 - rate limit, timeout, malformed tool call ...
            if self.backup is None or self.state["backup_dead"]:
                raise
            try:
                return self._chain(self.backup).invoke(messages)
            except Exception as backup_exc:  # noqa: BLE001
                if _is_model_missing(backup_exc):
                    self.state["backup_dead"] = True
                self.state["last_backup_error"] = f"{type(backup_exc).__name__}: {str(backup_exc)[:200]}"
                if _is_rate_limit(primary_exc) or _is_rate_limit(backup_exc):
                    time.sleep(RATE_LIMIT_WAIT_S)
                    try:
                        return self._chain(self.primary).invoke(messages)
                    except Exception:  # noqa: BLE001
                        pass
                if hasattr(primary_exc, "add_note"):      # Python 3.11+: show BOTH failures in the traceback
                    primary_exc.add_note(f"backup model also failed -> {self.state['last_backup_error']}")
                raise primary_exc


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0):
    primary = _build(provider(), model_name(), temperature)
    spec = fallback_spec()
    try:
        backup = _build(spec[0], spec[1], temperature) if spec else None
    except Exception:  # noqa: BLE001 - fallback package missing etc.: run without it
        backup = None
    return WithFallback(primary, backup)
