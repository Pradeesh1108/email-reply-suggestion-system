"""
Thin wrapper around the Groq SDK.

Provides retry/backoff and integrates with the response cache.  This is the
ONLY place in the codebase that makes actual network calls to Groq — all other
modules go through this wrapper, which makes it the natural mock boundary for
tests (spec §8: tests run with no network access).
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from groq import Groq, APIStatusError, RateLimitError

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.infra.cache import ResponseCache

logger = get_logger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds


class GroqClient:
    """
    Managed Groq API client with caching and retry logic.

    Design note (cost-control §6): the cache is checked BEFORE any network call,
    and successful responses are cached so duplicate requests don't re-spend tokens.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        settings = get_settings()
        self._client = Groq(api_key=api_key or settings.groq_api_key)
        self._cache = cache

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        max_tokens: int = 1024,
        cache_key: str | None = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Send a chat completion request to Groq.

        Args:
            model: The Groq model identifier.
            messages: Chat messages (system/user/assistant).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            cache_key: If provided, look up / store in cache.
            response_format: Optional response format specification (e.g. JSON mode).

        Returns:
            The assistant's response text.
        """
        # ── Cache lookup (saves an API call if we've seen this before) ────
        if cache_key and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit — skipping Groq API call")
                return cached

        # ── API call with retry ───────────────────────────────────────────
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = self._client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""

                # ── Cache store ───────────────────────────────────────────
                if cache_key and self._cache:
                    self._cache.set(cache_key, text)

                return text

            except RateLimitError as e:
                last_error = e
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning("Rate limited (attempt %d/%d), waiting %.1fs", attempt, MAX_RETRIES, wait)
                time.sleep(wait)

            except APIStatusError as e:
                last_error = e
                if e.status_code and e.status_code >= 500:
                    wait = BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning("Server error %d (attempt %d/%d), waiting %.1fs", e.status_code, attempt, MAX_RETRIES, wait)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Groq API failed after {MAX_RETRIES} retries: {last_error}")
