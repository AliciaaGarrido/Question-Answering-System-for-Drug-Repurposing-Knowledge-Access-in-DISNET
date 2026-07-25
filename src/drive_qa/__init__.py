"""
DRIVE QA – Drug Repurposing Natural Language Query System
=========================================================
Package for querying the DRIVE drug repurposing database using natural language.
"""

from drive_qa.pipelines import DriveQAPipeline, NLtoSQLPipeline, create_pipeline
from drive_qa.gemini_client import GeminiClient
from drive_qa.errors import DriveQAError, LLMError
from drive_qa.llm_client import LLMClient, create_llm_client
from drive_qa.model_registry import list_models, get_model_spec, is_gemini_model

__all__ = [
    "DriveQAPipeline",
    "NLtoSQLPipeline",
    "create_pipeline",
    "GeminiClient",
    "DriveQAError",
    "LLMError",
    "LLMClient",
    "create_llm_client",
    "list_models",
    "get_model_spec",
    "is_gemini_model",
]
