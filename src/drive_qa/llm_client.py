"""
LLM client abstraction — protocol and implementations for multiple LLM providers.
"""

from __future__ import annotations

import random
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

from drive_qa.config import RetryConfig
from drive_qa.logging_config import get_logger
from drive_qa.model_registry import ModelSpec, get_model_spec

logger = get_logger("llm_client")


class LLMClient(ABC):
    """Protocol for LLM clients used by the pipeline."""

    @abstractmethod
    def generate_sql(self, prompt: str) -> str:
        """Generate SQL from a constructed prompt. Returns clean SQL string."""
        ...

    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """Generate a natural language answer from results prompt."""
        ...


class GeminiLLMClient(LLMClient):
    """Adapter that wraps the existing GeminiClient to conform to LLMClient."""

    def __init__(self, gemini_client):
        from drive_qa.gemini_client import GeminiClient

        self._client: GeminiClient = gemini_client

    def generate_sql(self, prompt: str) -> str:
        return self._client.generate_sql(prompt)

    def generate_answer(self, prompt: str) -> str:
        return self._client.generate_answer(prompt)


class AzureOpenAIClient(LLMClient):
    """
    LLM client for Azure AI Foundry models using the OpenAI-compatible API.

    Supports Kimi, Deepseek, and GPT models via chat.completions.create().
    """

    def __init__(self, spec: ModelSpec, api_key: str, retry: Optional[RetryConfig] = None):
        from openai import OpenAI

        self._spec = spec
        self._retry = retry or RetryConfig()
        self._client = OpenAI(
            base_url=spec.base_url,
            api_key=api_key,
            timeout=self._retry.request_timeout_seconds,
        )
        logger.info(
            "AzureOpenAIClient initialized (model=%s, deployment=%s, timeout=%.0fs)",
            spec.model_id,
            spec.deployment_name,
            self._retry.request_timeout_seconds,
        )

    def generate_sql(self, prompt: str) -> str:
        from drive_qa.gemini_client import SQL_SYSTEM_PROMPT, _strip_code_fences

        raw = self._call_with_retry(
            system_prompt=SQL_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=self._spec.max_sql_tokens,
            purpose="generate_sql",
        )
        return _strip_code_fences(raw)

    def generate_answer(self, prompt: str) -> str:
        from drive_qa.gemini_client import ANSWER_SYSTEM_PROMPT

        return self._call_with_retry(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=self._spec.max_answer_tokens,
            purpose="generate_answer",
        )

    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        *,
        purpose: str = "generate",
    ) -> str:
        """Call the OpenAI-compatible endpoint with retry logic."""
        from drive_qa.errors import (
            LLMEmptyResponseError,
            LLMError,
            LLMRateLimitError,
            LLMUnavailableError,
        )

        last_error: Optional[Exception] = None

        for attempt in range(self._retry.max_retries + 1):
            try:
                logger.debug(
                    "Azure OpenAI call attempt %d/%d (purpose=%s, model=%s)",
                    attempt + 1,
                    self._retry.max_retries + 1,
                    purpose,
                    self._spec.deployment_name,
                )

                kwargs: dict = {
                    "model": self._spec.deployment_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_completion_tokens": max_tokens,
                }

                if self._spec.supports_temperature and self._spec.temperature is not None:
                    kwargs["temperature"] = self._spec.temperature

                response = self._client.chat.completions.create(**kwargs)

                if not response or not response.choices:
                    if attempt < self._retry.max_retries:
                        logger.warning(
                            "Azure OpenAI returned empty response (attempt %d), retrying...",
                            attempt + 1,
                        )
                        time.sleep(self._compute_delay(attempt, None))
                        continue
                    raise LLMEmptyResponseError(
                        f"{self._spec.display_name} returned an empty response."
                    )

                content = response.choices[0].message.content
                if not content or not content.strip():
                    if attempt < self._retry.max_retries:
                        logger.warning(
                            "Azure OpenAI returned empty content (attempt %d), retrying...",
                            attempt + 1,
                        )
                        time.sleep(self._compute_delay(attempt, None))
                        continue
                    raise LLMEmptyResponseError(
                        f"{self._spec.display_name} returned empty content."
                    )

                logger.debug(
                    "Azure OpenAI call succeeded (purpose=%s, response_length=%d)",
                    purpose,
                    len(content),
                )
                return content.strip()

            except (LLMEmptyResponseError, LLMError):
                raise
            except Exception as exc:
                classified = self._classify_error(exc)
                last_error = classified

                if not classified.retryable or attempt >= self._retry.max_retries:
                    logger.error(
                        "Azure OpenAI permanent error (purpose=%s, attempt=%d): %s",
                        purpose,
                        attempt + 1,
                        classified,
                    )
                    raise classified from exc

                delay = self._compute_delay(attempt, classified.retry_after)
                logger.warning(
                    "Azure OpenAI transient error (purpose=%s, attempt=%d, "
                    "retrying in %.1fs): %s",
                    purpose,
                    attempt + 1,
                    delay,
                    type(classified).__name__,
                )
                time.sleep(delay)

        if last_error:
            raise last_error
        from drive_qa.errors import LLMError

        raise LLMError("Exhausted retries with no error captured.", stage="llm")

    def _compute_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        """Compute delay for a given retry attempt."""
        rc = self._retry
        if retry_after is not None and retry_after > 0:
            delay = retry_after
        else:
            delay = min(
                rc.base_delay * (rc.exponential_base**attempt),
                rc.max_delay,
            )
        if rc.jitter:
            delay += random.uniform(0, delay * 0.25)
        return delay

    def _classify_error(self, exc: Exception):
        """Classify an OpenAI SDK exception into our error hierarchy."""
        from drive_qa.errors import LLMError, LLMRateLimitError, LLMUnavailableError

        msg = str(exc)
        status_code = getattr(exc, "status_code", None)

        msg_lower = msg.lower()

        if (
            status_code == 503
            or "503" in msg
            or "unavailable" in msg_lower
            or "timeout" in msg_lower
            or "timed out" in msg_lower
        ):
            retry_after = self._parse_retry_header(exc)
            return LLMUnavailableError(msg, retry_after=retry_after)

        if status_code == 429 or "429" in msg or "rate" in msg.lower():
            retry_after = self._parse_retry_header(exc)
            return LLMRateLimitError(msg, retryable=True, retry_after=retry_after)

        return LLMError(msg, stage="llm", retryable=False)

    @staticmethod
    def _parse_retry_header(exc: Exception) -> Optional[float]:
        """Try to extract retry-after from exception headers or message."""
        # Check for retry-after header in OpenAI SDK exceptions
        headers = getattr(exc, "headers", None) or {}
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

        # Fallback: parse from message
        msg = str(exc)
        match = re.search(r"retry.{0,10}?(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None


def create_llm_client(model_id: str, api_key: str) -> LLMClient:
    """
    Factory: create the appropriate LLMClient for a given model_id.

    Args:
        model_id: Model identifier from the registry.
        api_key: API key provided by the user.

    Returns:
        An LLMClient instance ready to use.
    """
    spec = get_model_spec(model_id)

    if spec.provider == "gemini":
        from drive_qa.config import GeminiConfig
        from drive_qa.gemini_client import GeminiClient

        gemini_config = GeminiConfig(
            model_name=spec.deployment_name,
            temperature=spec.temperature if spec.temperature is not None else 0.0,
            max_output_tokens=spec.max_sql_tokens,
            max_answer_tokens=spec.max_answer_tokens,
            _api_key=api_key,
        )
        return GeminiLLMClient(GeminiClient(config=gemini_config))

    if spec.provider == "azure_openai":
        return AzureOpenAIClient(spec=spec, api_key=api_key)

    raise ValueError(f"Unsupported provider '{spec.provider}' for model '{model_id}'")
