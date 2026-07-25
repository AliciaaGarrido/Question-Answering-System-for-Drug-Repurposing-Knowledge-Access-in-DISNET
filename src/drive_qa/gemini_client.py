"""
Gemini API client with retry logic, error classification, and structured logging.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from drive_qa.config import GeminiConfig, RetryConfig
from drive_qa.errors import (
    GeminiEmptyResponseError,
    GeminiError,
    GeminiRateLimitError,
    GeminiUnavailableError,
)
from drive_qa.logging_config import get_logger

logger = get_logger("gemini_client")


# ─── Response prompts ─────────────────────────────────────────────────────────

SQL_SYSTEM_PROMPT = """\
You are an expert MySQL SQL query generator for the DRIVE drug repurposing database.
All tables with dr_ prefix are related to drug repurposing evidence models. Other tables are for reference (e.g., gene information).

STRICT RULES:
1. Output ONLY a single SQL SELECT query. No explanations, no markdown, no comments.
2. Use ONLY the tables and columns provided in the schema below.
3. NEVER invent or hallucinate table names or column names.
4. Use explicit JOIN syntax with the join conditions provided.
5. Use table aliases when joining multiple tables.
6. Use GROUP BY when using aggregate functions.
7. Use ORDER BY and LIMIT only for explicit top-k / sorted ranking queries.
8. For max/min questions without an explicit top-N number, filter rows to the computed MAX/MIN metric value using a subquery or CTE, preserving ties.
9. Generate MySQL-compatible SQL only.
10. NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE, GRANT, or REVOKE.
11. Output ONLY the SQL query, nothing else.
12. ALL table references MUST be qualified with the database prefix exactly as shown in the schema (e.g., dr.disease, dr.dr_gnns). NEVER omit the database prefix.
"""

ANSWER_SYSTEM_PROMPT = """\
You are a helpful data analyst assistant for the DRIVE drug repurposing database.

STRICT RULES:
1. Answer the user's question based ONLY on the SQL query results provided.
2. Be concise and precise. Do not add information that is not in the data.
3. If no rows were returned, say so clearly. Do NOT invent or guess data.
4. If the results are truncated, mention that only a subset of rows is shown.
5. Use the column names to understand what each value represents.
6. Respond in the same language as the user's question.
7. Format numbers clearly (e.g., round floats to a reasonable number of decimals).
8. If the results contain multiple metric values, do not describe all rows as sharing the maximum/minimum unless the SQL filtered them that way.
"""


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from Gemini response."""
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text


def _parse_retry_delay(error_message: str) -> Optional[float]:
    """Extract retry delay from Gemini error messages."""
    # Pattern: "Please retry in X.XXs"
    match = re.search(r"[Pp]lease retry in (\d+(?:\.\d+)?)s", error_message)
    if match:
        return float(match.group(1))
    # Pattern: retryDelay: "5s"
    match = re.search(r"retryDelay.*?(\d+)s", error_message)
    if match:
        return float(match.group(1))
    return None


def _is_quota_exhausted(error_message: str) -> bool:
    """Detect if error indicates daily quota exhaustion (not just rate limiting)."""
    indicators = [
        "limit: 0",
        "QuotaExceeded",
        "billing",
    ]
    return any(ind in error_message for ind in indicators)


def _classify_api_error(exc: Exception) -> GeminiError:
    """Classify a raw google.genai exception into our error hierarchy."""
    msg = str(exc)
    msg_lower = msg.lower()

    if "timeout" in msg_lower or "timed out" in msg_lower:
        return GeminiUnavailableError(msg)

    if "503" in msg or "UNAVAILABLE" in msg:
        retry_after = _parse_retry_delay(msg)
        return GeminiUnavailableError(msg, retry_after=retry_after)

    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        retry_after = _parse_retry_delay(msg)
        quota_exhausted = _is_quota_exhausted(msg)
        return GeminiRateLimitError(
            msg,
            retryable=not quota_exhausted,
            retry_after=retry_after,
        )

    # Generic/unexpected Gemini error — not retryable by default
    return GeminiError(msg, stage="gemini", retryable=False)


class GeminiClient:
    """
    Gemini API client with automatic retry for transient errors.

    Handles:
    - 503 UNAVAILABLE → retry with exponential backoff
    - 429 RESOURCE_EXHAUSTED (rate limit) → retry respecting RetryInfo delay
    - 429 (quota exhausted with limit=0) → fail fast, no infinite retry
    - Empty responses → retry once
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()
        timeout_ms = int(self.config.retry.request_timeout_seconds * 1000)
        self._client = genai.Client(
            api_key=self.config.api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        logger.info(
            "GeminiClient initialized (model=%s, temperature=%.1f, timeout=%.0fs)",
            self.config.model_name,
            self.config.temperature,
            self.config.retry.request_timeout_seconds,
        )

    def _compute_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        """Compute delay for a given retry attempt."""
        rc = self.config.retry
        if retry_after is not None and retry_after > 0:
            delay = retry_after
        else:
            delay = min(
                rc.base_delay * (rc.exponential_base ** attempt),
                rc.max_delay,
            )
        if rc.jitter:
            delay += random.uniform(0, delay * 0.25)
        return delay

    def _call_with_retry(
        self,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int,
        *,
        purpose: str = "generate",
    ) -> str:
        """
        Internal method: call Gemini with retry logic.

        Returns the text response on success.
        Raises GeminiError subclass on permanent failure.
        """
        rc = self.config.retry
        last_error: Optional[Exception] = None

        for attempt in range(rc.max_retries + 1):
            try:
                logger.debug(
                    "Gemini call attempt %d/%d (purpose=%s, model=%s)",
                    attempt + 1,
                    rc.max_retries + 1,
                    purpose,
                    self.config.model_name,
                )

                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.config.temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )

                if not response or not response.text:
                    # Empty response — try once more
                    if attempt < rc.max_retries:
                        logger.warning(
                            "Gemini returned empty response (attempt %d), retrying...",
                            attempt + 1,
                        )
                        time.sleep(self._compute_delay(attempt, None))
                        continue
                    raise GeminiEmptyResponseError()

                logger.debug(
                    "Gemini call succeeded (purpose=%s, response_length=%d)",
                    purpose,
                    len(response.text),
                )
                return response.text.strip()

            except (GeminiEmptyResponseError, GeminiError):
                raise
            except Exception as exc:
                classified = _classify_api_error(exc)
                last_error = classified

                if not classified.retryable or attempt >= rc.max_retries:
                    logger.error(
                        "Gemini permanent error (purpose=%s, attempt=%d): %s",
                        purpose,
                        attempt + 1,
                        classified,
                    )
                    raise classified from exc

                delay = self._compute_delay(attempt, classified.retry_after)
                logger.warning(
                    "Gemini transient error (purpose=%s, attempt=%d, "
                    "retrying in %.1fs): %s",
                    purpose,
                    attempt + 1,
                    delay,
                    type(classified).__name__,
                )
                time.sleep(delay)

        # Should not reach here, but safety
        if last_error:
            raise last_error
        raise GeminiError("Exhausted retries with no error captured.", stage="gemini")

    def generate_sql(self, prompt: str) -> str:
        """
        Call Gemini to generate SQL from a constructed prompt.

        Returns cleaned SQL string.
        Raises GeminiError on failure.
        """
        raw = self._call_with_retry(
            system_instruction=SQL_SYSTEM_PROMPT,
            prompt=prompt,
            max_output_tokens=self.config.max_output_tokens,
            purpose="generate_sql",
        )
        return _strip_code_fences(raw)

    def generate_answer(self, prompt: str) -> str:
        """
        Call Gemini to verbalize SQL results into natural language.

        Returns the answer string.
        Raises GeminiError on failure.
        """
        return self._call_with_retry(
            system_instruction=ANSWER_SYSTEM_PROMPT,
            prompt=prompt,
            max_output_tokens=self.config.max_answer_tokens,
            purpose="generate_answer",
        )
