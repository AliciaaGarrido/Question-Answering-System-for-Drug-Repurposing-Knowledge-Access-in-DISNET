"""
Error hierarchy for the DRIVE QA system.

Provides structured, classifiable errors that preserve backward compatibility
with existing dict-based responses while adding rich metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ErrorInfo:
    """Structured error information attached to pipeline responses."""

    error: str
    error_type: str
    stage: str
    retryable: bool = False
    retry_after_seconds: Optional[float] = None
    error_code: Optional[int] = None
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "error": self.error,
            "error_type": self.error_type,
            "stage": self.stage,
            "retryable": self.retryable,
        }
        if self.retry_after_seconds is not None:
            d["retry_after_seconds"] = self.retry_after_seconds
        if self.error_code is not None:
            d["error_code"] = self.error_code
        if self.details is not None:
            d["details"] = self.details
        return d


# ─── Base exception ───────────────────────────────────────────────────────────

class DriveQAError(Exception):
    """Base exception for the drive_qa package."""

    def __init__(self, message: str, *, stage: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.stage = stage
        self.retryable = retryable


# ─── Configuration / startup ─────────────────────────────────────────────────

class ConfigurationError(DriveQAError):
    """Missing or invalid configuration (e.g. no API key)."""

    def __init__(self, message: str):
        super().__init__(message, stage="configuration", retryable=False)


# ─── Generic LLM errors ───────────────────────────────────────────────────────

class LLMError(DriveQAError):
    """Base for all LLM-related errors (provider-agnostic)."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "llm",
        retryable: bool = False,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, stage=stage, retryable=retryable)
        self.status_code = status_code
        self.retry_after = retry_after


class LLMUnavailableError(LLMError):
    """503 UNAVAILABLE — transient, should retry with backoff."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(
            message,
            stage="llm",
            retryable=True,
            status_code=503,
            retry_after=retry_after,
        )


class LLMRateLimitError(LLMError):
    """429 rate limit — may be transient or quota exhausted."""

    def __init__(
        self,
        message: str,
        retryable: bool = True,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            stage="llm",
            retryable=retryable,
            status_code=429,
            retry_after=retry_after,
        )


class LLMEmptyResponseError(LLMError):
    """LLM returned an empty or null response."""

    def __init__(self, message: str = "LLM returned an empty response."):
        super().__init__(message, stage="llm", retryable=True, status_code=None)


# ─── Gemini API errors (backward-compatible, now subclasses of LLMError) ──────

class GeminiError(LLMError):
    """Base for Gemini-related errors."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "gemini",
        retryable: bool = False,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message, stage=stage, retryable=retryable,
            status_code=status_code, retry_after=retry_after,
        )


class GeminiUnavailableError(GeminiError):
    """503 UNAVAILABLE — transient, should retry with backoff."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(
            message,
            stage="gemini",
            retryable=True,
            status_code=503,
            retry_after=retry_after,
        )


class GeminiRateLimitError(GeminiError):
    """429 RESOURCE_EXHAUSTED — may be transient or quota exhausted."""

    def __init__(
        self,
        message: str,
        retryable: bool = True,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            stage="gemini",
            retryable=retryable,
            status_code=429,
            retry_after=retry_after,
        )


class GeminiEmptyResponseError(GeminiError):
    """Gemini returned an empty or null response."""

    def __init__(self, message: str = "Gemini returned an empty response."):
        super().__init__(message, stage="gemini", retryable=True, status_code=None)


# ─── SQL errors ───────────────────────────────────────────────────────────────

class SQLValidationError(DriveQAError):
    """Generated SQL failed validation (forbidden keywords, unknown tables, etc.)."""

    def __init__(self, message: str):
        super().__init__(message, stage="sql_validation", retryable=False)


class SQLExecutionError(DriveQAError):
    """Error executing SQL against the database."""

    def __init__(self, message: str):
        super().__init__(message, stage="sql_execution", retryable=False)


class DatabaseConnectionError(DriveQAError):
    """Cannot connect to the database."""

    def __init__(self, message: str):
        super().__init__(message, stage="database", retryable=True)


# ─── Retriever errors ─────────────────────────────────────────────────────────

class RetrieverError(DriveQAError):
    """Error in the schema retriever (parsing, entity resolution, scoring)."""

    def __init__(self, message: str, *, stage: str = "retriever"):
        super().__init__(message, stage=stage, retryable=False)


class InsufficientContextError(RetrieverError):
    """Retriever could not identify relevant tables."""

    def __init__(self, message: str = "Insufficient schema context."):
        super().__init__(message, stage="retriever")


# ─── Answer generation errors ─────────────────────────────────────────────────

class AnswerGenerationError(DriveQAError):
    """Error during SQL→NL verbalization step."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message, stage="answer_generation", retryable=retryable)


# ─── Helper to build ErrorInfo from exceptions ────────────────────────────────

def error_info_from_exception(exc: Exception, *, stage: str = "unknown") -> ErrorInfo:
    """Convert an exception into a structured ErrorInfo."""
    if isinstance(exc, DriveQAError):
        info = ErrorInfo(
            error=str(exc),
            error_type=type(exc).__name__,
            stage=exc.stage,
            retryable=exc.retryable,
        )
        if isinstance(exc, LLMError):
            info.error_code = exc.status_code
            info.retry_after_seconds = exc.retry_after
        return info

    return ErrorInfo(
        error=str(exc),
        error_type=type(exc).__name__,
        stage=stage,
        retryable=False,
    )
