"""
Model registry — metadata and factory for supported LLM models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a supported LLM model."""

    model_id: str
    display_name: str
    provider: str  # "gemini" | "azure_openai"
    base_url: str
    deployment_name: str
    temperature: Optional[float]  # None = do not send
    max_sql_tokens: int
    max_answer_tokens: int
    supports_temperature: bool

    @property
    def ui_label(self) -> str:
        return self.display_name


# ─── Registry ────────────────────────────────────────────────────────────────

AZURE_BASE_URL = "https://ali-resource.services.ai.azure.com/openai/v1"

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gemini-3.5-flash": ModelSpec(
        model_id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        provider="gemini",
        base_url="",  # uses google-genai SDK, no base_url needed
        deployment_name="gemini-3.5-flash",
        temperature=0.0,
        max_sql_tokens=4096,
        max_answer_tokens=126000,
        supports_temperature=True,
    ),
    "gemini-3.1-flash-lite": ModelSpec(
        model_id="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash Lite",
        provider="gemini",
        base_url="",
        deployment_name="gemini-3.1-flash-lite",
        temperature=0.0,
        max_sql_tokens=4096,
        max_answer_tokens=126000,
        supports_temperature=True,
    ),
    "Kimi-K2.6": ModelSpec(
        model_id="Kimi-K2.6",
        display_name="Kimi-K2.6",
        provider="azure_openai",
        base_url=AZURE_BASE_URL,
        deployment_name="Kimi-K2.6",
        temperature=0.0,
        max_sql_tokens=4096,
        max_answer_tokens=126000,
        supports_temperature=True,
    ),
    "gpt-5.4-mini": ModelSpec(
        model_id="gpt-5.4-mini",
        display_name="GPT 5.4-mini",
        provider="azure_openai",
        base_url=AZURE_BASE_URL,
        deployment_name="gpt-5.4-mini",
        temperature=None,
        max_sql_tokens=4096,
        max_answer_tokens=126000,
        supports_temperature=False,
    ),
    "gpt-5.4": ModelSpec(
        model_id="gpt-5.4",
        display_name="GPT 5.4",
        provider="azure_openai",
        base_url=AZURE_BASE_URL,
        deployment_name="gpt-5.4",
        temperature=None,
        max_sql_tokens=4096,
        max_answer_tokens=126000,
        supports_temperature=False,
    ),
    "deepseek-v4-flash": ModelSpec(
        model_id="deepseek-v4-flash",
        display_name="Deepseek v4-Flash",
        provider="azure_openai",
        base_url=AZURE_BASE_URL,
        deployment_name="deepseek-v4-flash",
        temperature=0.0,
        max_sql_tokens=4096,
        max_answer_tokens=126000,
        supports_temperature=True,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    """Get model specification by ID. Raises ValueError if unknown."""
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available: {available}")
    return spec


def list_models() -> List[ModelSpec]:
    """Return all registered model specs (ordered)."""
    return list(MODEL_REGISTRY.values())


def is_gemini_model(model_id: str) -> bool:
    """Check if a model_id corresponds to a Gemini provider."""
    spec = MODEL_REGISTRY.get(model_id)
    return spec is not None and spec.provider == "gemini"
