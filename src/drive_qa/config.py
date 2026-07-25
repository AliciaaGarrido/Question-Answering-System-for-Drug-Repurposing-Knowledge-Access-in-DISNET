"""
Configuration — centralised settings for the DRIVE QA system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for transient errors."""

    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    request_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class GeminiConfig:
    """Configuration for the Gemini API client."""

    model_name: str = "gemini-3.1-flash-lite"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    max_answer_tokens: int = 2048
    retry: RetryConfig = field(default_factory=RetryConfig)
    _api_key: str = ""

    @property
    def api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it or provide the key via the web interface."
            )
        return key


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for the schema retriever."""

    top_k: int = 4
    min_score: int = 2
    expand_relations: bool = True
    limit_per_type: int = 10
    max_entities_per_type: int = 5
    prefer_single_exact: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    db_url: str = ""
    db_schema: str = "dr"
    max_answer_rows: int = 50
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    log_level: str = "INFO"
