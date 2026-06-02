"""
LLM factory for Market Intelligence workflows.
Uses gemini-2.5-flash-lite as reliable primary; falls back to Groq on 429/404.
Global circuit breaker: when Gemini hits 429, skip Gemini for future calls.
"""

import logging
import time

from langchain_core.runnables import Runnable

from app.errors.exceptions import LLMInvocationError
from app.errors.helpers import classify_exception
from app.config.models import (
    get_gemini_api_key,
    get_gemini_fallback_key,
    get_gemini_models,
    get_gemini_temperature,
    get_groq_api_key,
    get_groq_models,
    get_groq_temperature,
    get_prioritize_groq,
)

logger = logging.getLogger(__name__)

# Global circuit breaker: flip to False when Gemini hits 429; then skip Gemini entirely
GEMINI_AVAILABLE = True


def _backoff_before_fallback(exc: BaseException, attempt: int) -> None:
    """Sleep when ``is_retryable`` on the classified error (rate limit, timeout, etc.)."""
    err = classify_exception(exc, component="llm_factory", operation="fallback")
    if not err.is_retryable:
        return
    delay = min(0.6 * (2**attempt), 8.0)
    logger.info(
        "[Fallback] retryable %s — backing off %.1fs (attempt %d)",
        err.error_category,
        delay,
        attempt + 1,
    )
    time.sleep(delay)


def _raise_llm_failure(exc: BaseException, *, operation: str) -> None:
    """Raise LLMInvocationError with classified ToolError for graph nodes to capture."""
    if isinstance(exc, LLMInvocationError):
        raise exc
    raise LLMInvocationError(
        classify_exception(exc, component="llm_factory", operation=operation)
    ) from exc


def _is_fallback_error(exc: BaseException) -> bool:
    """Check if we should try next candidate: 429, 413, 404, 401, 400, model_decommissioned, etc."""
    s = str(exc).upper()
    if any(
        k in s
        for k in (
            "429",
            "413",
            "REQUEST TOO LARGE",
            "TPM",
            "404",
            "401",
            "400",
            "NOT_FOUND",
            "RESOURCE_EXHAUSTED",
            "QUOTA",
            "RATE_LIMIT",
            "RATELIMIT",
            "INVALID_API_KEY",
            "EXPIRED_API_KEY",
            "DECOMMISSIONED",
        )
    ):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_fallback_error(cause)
    return False


def _filter_gemini_models() -> list[str]:
    """Remove ghost gemini-3-flash; use gemini-2.5-flash-lite as reliable primary (1.5-flash 404s)."""
    raw = get_gemini_models()
    # Drop any gemini-3-flash (ghost/unreliable)
    filtered = [m for m in raw if "gemini-3-flash" not in m.lower()]
    # Use gemini-2.5-flash-lite first (gemini-1.5-flash returns 404 for many API versions)
    reliable = ["models/gemini-2.5-flash-lite", "models/gemini-2.5-flash"]
    ordered = [p for p in reliable if p in filtered]
    for m in filtered:
        if m not in ordered:
            ordered.append(m)
    return ordered if ordered else reliable[:1]


def _build_groq_candidates() -> list[tuple]:
    """Build Groq-only candidates for circuit-breaker bypass."""
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        return []

    groq_models = get_groq_models()
    groq_temp = get_groq_temperature()
    groq_key = get_groq_api_key()
    if not groq_key:
        return []

    candidates = []
    for model in groq_models:
        try:
            llm = ChatGroq(
                model=model,
                temperature=groq_temp,
                groq_api_key=groq_key,
            )
            candidates.append((llm, f"groq {model}"))
        except Exception:
            continue
    return candidates


def get_groq_llm():
    """Return a Groq-only FallbackLLM when circuit breaker is open."""
    candidates = _build_groq_candidates()
    if not candidates:
        raise RuntimeError("Groq fallback not configured. Set GROQ_API_KEY in .env.")
    return FallbackLLM(candidates=candidates)


def _build_candidates(
    skip_gemini: bool = False, temperature_override: float | None = None
) -> list[tuple]:
    """Build (llm, label) candidates. Order: Groq first when prioritize_groq=True.
    temperature_override: When set (e.g. 0 for critique), use instead of config temps.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        ChatGroq = None

    gemini_temp = (
        temperature_override if temperature_override is not None else get_gemini_temperature()
    )
    groq_temp = temperature_override if temperature_override is not None else get_groq_temperature()
    primary_key = get_gemini_api_key()
    fallback_key = get_gemini_fallback_key()
    groq_key = get_groq_api_key()
    gemini_candidates: list[tuple] = []
    groq_candidates: list[tuple] = []

    if not skip_gemini:
        gemini_models = _filter_gemini_models()
        for key, key_label in [(primary_key, "primary"), (fallback_key, "fallback")]:
            if not key:
                continue
            for model in gemini_models:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=gemini_temp,
                        google_api_key=key,
                    )
                    gemini_candidates.append((llm, f"gemini {model} ({key_label})"))
                except Exception:
                    continue

    if ChatGroq and groq_key:
        for model in get_groq_models():
            try:
                llm = ChatGroq(
                    model=model,
                    temperature=groq_temp,
                    groq_api_key=groq_key,
                )
                groq_candidates.append((llm, f"groq {model}"))
            except Exception:
                continue

    if get_prioritize_groq():
        return groq_candidates + gemini_candidates
    return gemini_candidates + groq_candidates


class FallbackLLM(Runnable):
    """
    Wrapper that on invoke() tries each candidate; on 429 flips circuit breaker
    and falls back to next (e.g. Gemini → Groq).
    """

    def __init__(self, candidates: list[tuple] | None = None):
        global GEMINI_AVAILABLE
        if candidates is not None:
            self._candidates = candidates
        else:
            self._candidates = _build_candidates(skip_gemini=not GEMINI_AVAILABLE)
        if not self._candidates:
            raise RuntimeError("No LLM configured. Set GOOGLE_API_KEY or GROQ_API_KEY in .env.")

    def invoke(self, input, config=None, **kwargs):
        global GEMINI_AVAILABLE
        last_error = None
        for i, (llm, label) in enumerate(self._candidates):
            try:
                out = llm.invoke(input, config=config, **kwargs)
                if i > 0:
                    logger.info("[Fallback OK] Using %s after rate limit", label)
                return out
            except Exception as e:
                last_error = e
                if _is_fallback_error(e):
                    logger.warning("[Fallback] 429/413 on %s, trying next", label)
                    if "gemini" in label.lower():
                        GEMINI_AVAILABLE = False  # Only skip Gemini when Gemini itself fails
                    _backoff_before_fallback(e, i)
                    continue
                _raise_llm_failure(e, operation="invoke")
        if last_error:
            _raise_llm_failure(last_error, operation="invoke")
        raise LLMInvocationError(
            classify_exception(
                RuntimeError("All LLM candidates failed"),
                component="llm_factory",
                operation="invoke",
            )
        )

    def with_structured_output(self, schema, **kwargs):
        """
        Return a runnable that invokes each candidate's with_structured_output with fallback.
        All candidates (including Groq) use the same Pydantic schema—no raw string to Extractor.
        """
        return FallbackStructuredRunnable(self._candidates, schema, **kwargs)


class FallbackStructuredRunnable(Runnable):
    """
    Runnable that invokes each candidate's with_structured_output; falls back on 429.
    Standardized fallbacks: Groq also uses with_structured_output(schema) so Extractor
    always receives a Pydantic object, never a raw string.
    """

    def __init__(self, candidates, schema, **kwargs):
        # Each candidate (Gemini and Groq) gets same schema—consistent structured output
        self._candidates = [
            (llm.with_structured_output(schema, **kwargs), label) for llm, label in candidates
        ]
        self._schema = schema

    def invoke(self, input, config=None, **kwargs):
        global GEMINI_AVAILABLE
        last_error = None
        for i, (structured_llm, label) in enumerate(self._candidates):
            try:
                out = structured_llm.invoke(input, config=config, **kwargs)
                if i > 0:
                    logger.info("[Fallback OK] Structured output via %s", label)
                return out
            except Exception as e:
                last_error = e
                if _is_fallback_error(e):
                    logger.warning("[Fallback] 429/413 on %s (structured), trying next", label)
                    if "gemini" in label.lower():
                        GEMINI_AVAILABLE = False  # Only skip Gemini when Gemini itself fails
                    _backoff_before_fallback(e, i)
                    continue
                _raise_llm_failure(e, operation="invoke_structured")
        if last_error:
            _raise_llm_failure(last_error, operation="invoke_structured")
        raise LLMInvocationError(
            classify_exception(
                RuntimeError("All LLM candidates failed"),
                component="llm_factory",
                operation="invoke_structured",
            )
        )


def get_llm(structured: bool = False):
    """
    Return an LLM with invoke-time fallback on 429 (Gemini → Groq).
    When GEMINI_AVAILABLE is False (circuit breaker open), immediately jump to Groq.
    structured: When True, callers use with_structured_output(schema)—Groq fallback uses same schema.
    """
    global GEMINI_AVAILABLE
    if not GEMINI_AVAILABLE:
        return get_groq_llm()
    return FallbackLLM()


def get_critique_llm():
    """
    Return an LLM for the Critique node with temperature=0 (deterministic auditor).
    Uses same fallback chain as get_llm.
    """
    global GEMINI_AVAILABLE
    candidates = _build_candidates(skip_gemini=not GEMINI_AVAILABLE, temperature_override=0.0)
    if not candidates:
        return get_llm()
    return FallbackLLM(candidates=candidates)


def get_ragas_llm():
    """
    Return a Ragas-compatible LLM for evaluation (faithfulness, etc.).
    Uses the first LangChain LLM from our fallback chain so Ragas can call
    generate_prompt/agenerate_prompt. Fallback on 429 is handled by retrying
    in the evaluation node if needed.
    """
    from langchain_core.language_models import BaseLanguageModel
    from ragas.llms import LangchainLLMWrapper

    global GEMINI_AVAILABLE
    candidates = _build_candidates(skip_gemini=not GEMINI_AVAILABLE, temperature_override=0.01)
    if not candidates:
        raise RuntimeError("No LLM configured for Ragas. Set GOOGLE_API_KEY or GROQ_API_KEY.")
    first_llm = candidates[0][0]
    if not isinstance(first_llm, BaseLanguageModel):
        raise RuntimeError(
            "First candidate is not a BaseLanguageModel; Ragas requires LangChain LLM."
        )
    return LangchainLLMWrapper(first_llm, bypass_temperature=True)
