"""LLM factory + secrets access.

Keys come from environment variables or Streamlit secrets.
Never store API keys in the repository.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache


# Local development convenience.
# Streamlit Cloud does not require a .env file because secrets are
# provided through Streamlit Secrets.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Primary model.
# The smaller model is used first to reduce free-tier rate-limit pressure.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

# Backup model.
# Used if the primary model fails.
DEFAULT_GROQ_FALLBACK = "openai/gpt-oss-120b"

KEY_NAME = "GROQ_API_KEY"


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def secret(name: str, default: str | None = None) -> str | None:
    """Read a value from environment variables or Streamlit secrets."""

    # Local environment / .env
    value = os.getenv(name)

    if value:
        return value

    # Streamlit Cloud secrets
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:  # noqa: BLE001
        return default


def provider() -> str:
    """Return the configured LLM provider."""

    return (
        secret("LLM_PROVIDER", "groq")
        or "groq"
    ).lower()


def model_name() -> str:
    """Return the primary Groq model."""

    return (
        secret("LLM_MODEL", DEFAULT_GROQ_MODEL)
        or DEFAULT_GROQ_MODEL
    )


def has_api_key() -> bool:
    """Return whether a Groq API key is available."""

    return bool(secret(KEY_NAME))


# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------

def _build(
    provider_name: str,
    model: str,
    temperature: float,
):
    """Build the configured LLM."""

    if provider_name != "groq":
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider_name}'. "
            "This application currently supports only 'groq'."
        )

    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=secret(KEY_NAME),
        # Retry/fallback behavior is handled by WithFallback.
        max_retries=0,
    )


# ---------------------------------------------------------------------------
# Fallback configuration
# ---------------------------------------------------------------------------

def fallback_spec() -> tuple[str, str] | None:
    """Return the provider/model used as the fallback.

    By default:

        Primary  -> Groq openai/gpt-oss-20b
        Fallback -> Groq openai/gpt-oss-120b

    The fallback can be disabled with:

        LLM_FALLBACK_MODEL = "none"

    or explicitly configured through:

        LLM_FALLBACK_PROVIDER
        LLM_FALLBACK_MODEL
    """

    primary_provider = provider()
    primary_model = model_name()

    # This application currently supports Groq only.
    if primary_provider != "groq":
        return None

    fallback_provider = (
        secret(
            "LLM_FALLBACK_PROVIDER",
            "groq",
        )
        or "groq"
    ).lower()

    fallback_model = secret(
        "LLM_FALLBACK_MODEL",
        DEFAULT_GROQ_FALLBACK,
    )

    # Allow the fallback to be disabled.
    if (fallback_model or "").lower() in {
        "none",
        "off",
        "0",
        "false",
    }:
        return None

    if not fallback_model:
        fallback_model = DEFAULT_GROQ_FALLBACK

    # Never use the exact same model as its own fallback.
    if (
        fallback_provider == primary_provider
        and fallback_model == primary_model
    ):
        return None

    # We only support Groq in this configuration.
    if fallback_provider != "groq":
        return None

    # Both models use the same Groq API key.
    if not secret(KEY_NAME):
        return None

    return fallback_provider, fallback_model


# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------

# Give a temporary free-tier rate limit a little time to reset before
# retrying the primary model after both models have failed.
RATE_LIMIT_WAIT_S = 12.0


def _is_rate_limit(exc: Exception) -> bool:
    """Detect common rate-limit errors."""

    text_value = (
        f"{type(exc).__name__} {exc}"
    ).lower()

    return (
        "429" in text_value
        or "rate limit" in text_value
        or "ratelimit" in text_value
        or "rate_limit" in text_value
    )


def _is_model_missing(exc: Exception) -> bool:
    """Detect model-not-found errors."""

    text_value = (
        f"{type(exc).__name__} {exc}"
    ).lower()

    return (
        "404" in text_value
        or "notfound" in text_value
        or "does not exist" in text_value
        or "model_not_found" in text_value
    )


# ---------------------------------------------------------------------------
# Primary + fallback wrapper
# ---------------------------------------------------------------------------

class WithFallback:
    """LLM wrapper that automatically tries a backup model.

    Supported operations:

    - bind_tools()
    - with_structured_output()
    - invoke()

    Invocation flow:

        Primary model
             |
             | failure
             v
        Fallback model
             |
             | rate-limit failure
             v
        Wait briefly
             |
             v
        Primary model retry
    """

    def __init__(
        self,
        primary,
        backup=None,
        ops: tuple = (),
        state: dict | None = None,
    ):
        self.primary = primary
        self.backup = backup
        self.ops = ops

        self.state = (
            state
            if state is not None
            else {
                "backup_dead": False,
            }
        )

    def _chain(self, llm):
        """Apply deferred LangChain operations."""

        for name, argument in self.ops:
            llm = getattr(llm, name)(argument)

        return llm

    def _clone(self, operation):
        """Create a new wrapper preserving the current operations."""

        return WithFallback(
            primary=self.primary,
            backup=self.backup,
            ops=self.ops + (operation,),
            state=self.state,
        )

    def bind_tools(self, tools):
        """Bind tools to both primary and fallback models."""

        return self._clone(
            ("bind_tools", tools)
        )

    def with_structured_output(self, schema):
        """Enable structured output on both models."""

        return self._clone(
            ("with_structured_output", schema)
        )

    def invoke(self, messages):
        """Invoke the primary model and handle fallback failures."""

        # ---------------------------------------------------------------
        # 1. Try primary model
        # ---------------------------------------------------------------
        try:
            return self._chain(
                self.primary
            ).invoke(messages)

        except Exception as primary_exc:  # noqa: BLE001

            # No fallback available.
            if (
                self.backup is None
                or self.state["backup_dead"]
            ):
                raise

            # -----------------------------------------------------------
            # 2. Try fallback model
            # -----------------------------------------------------------
            try:
                return self._chain(
                    self.backup
                ).invoke(messages)

            except Exception as backup_exc:  # noqa: BLE001

                # If the fallback model does not exist or is unavailable,
                # don't repeatedly attempt it during this session.
                if _is_model_missing(backup_exc):
                    self.state["backup_dead"] = True

                self.state["last_backup_error"] = (
                    f"{type(backup_exc).__name__}: "
                    f"{str(backup_exc)[:200]}"
                )

                # -------------------------------------------------------
                # 3. If rate-limited, wait and retry primary once
                # -------------------------------------------------------
                if (
                    _is_rate_limit(primary_exc)
                    or _is_rate_limit(backup_exc)
                ):
                    time.sleep(
                        RATE_LIMIT_WAIT_S
                    )

                    try:
                        return self._chain(
                            self.primary
                        ).invoke(messages)

                    except Exception:  # noqa: BLE001
                        pass

                # -------------------------------------------------------
                # 4. Preserve the original primary error
                # -------------------------------------------------------
                if hasattr(primary_exc, "add_note"):
                    primary_exc.add_note(
                        "Backup model also failed -> "
                        f"{self.state['last_backup_error']}"
                    )

                raise primary_exc


# ---------------------------------------------------------------------------
# Cached LLM factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0):
    """Create and cache the primary LLM and optional fallback."""

    primary = _build(
        provider(),
        model_name(),
        temperature,
    )

    spec = fallback_spec()

    backup = None

    if spec:
        try:
            backup = _build(
                spec[0],
                spec[1],
                temperature,
            )
        except Exception:  # noqa: BLE001
            # If the fallback cannot be constructed, continue using
            # the primary model.
            backup = None

    return WithFallback(
        primary=primary,
        backup=backup,
    )