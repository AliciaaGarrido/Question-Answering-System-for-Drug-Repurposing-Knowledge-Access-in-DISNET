"""
Pipelines — end-to-end NL→SQL→Answer orchestration.
"""
# DriveQAPipeline is the canonical name; NLtoSQLPipeline is kept as alias.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from drive_qa.config import GeminiConfig
from drive_qa.errors import (
    DatabaseConnectionError,
    DriveQAError,
    LLMError,
    SQLExecutionError,
    error_info_from_exception,
)
from drive_qa.gemini_client import GeminiClient
from drive_qa.llm_client import LLMClient, create_llm_client, GeminiLLMClient
from drive_qa.logging_config import get_logger, setup_logging
from drive_qa.prompt_builder import (
    DEFAULT_DB_SCHEMA,
    build_answer_prompt,
    build_join_context,
    build_prompt,
    qualify_schema_context,
)
from drive_qa.retriever import build_entity_context, build_schema_context, retrieve_schema
from drive_qa.schema_catalog import SCHEMA_CATALOG
from drive_qa.sql_execution import execute_sql
from drive_qa.sql_validation import validate_sql

logger = get_logger("pipeline")


class DriveQAPipeline:
    """
    Full NL → SQL → Execution → Answer pipeline with structured error handling.

    Combines: retriever → prompt → Gemini → SQL validation → execution → answer.
    """

    def __init__(
        self,
        engine: Engine,
        gemini_client=None,
        top_k: int = 4,
        min_score: int = 2,
        expand_relations: bool = True,
        db_schema: str = DEFAULT_DB_SCHEMA,
        max_answer_rows: int = 50,
        *,
        llm_client: Optional[LLMClient] = None,
    ):
        self.engine = engine
        # Accept either llm_client (new) or gemini_client (backward compat)
        if llm_client is not None:
            self._llm_client = llm_client
        elif gemini_client is not None:
            self._llm_client = GeminiLLMClient(gemini_client)
        else:
            raise ValueError("Either llm_client or gemini_client must be provided.")
        self.top_k = top_k
        self.min_score = min_score
        self.expand_relations = expand_relations
        self.db_schema = db_schema
        self.max_answer_rows = max_answer_rows

    @property
    def gemini_client(self):
        """Backward-compatible property."""
        return self._llm_client

    def generate(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL from a natural language question.

        Returns dict with keys:
            question, retrieved_tables, sql, valid, error, error_info
        """
        logger.info("Pipeline.generate() called: %s", question[:120])

        # Step 1: Retrieve relevant tables
        logger.info("Stage: retriever")
        try:
            retriever_result = retrieve_schema(
                question=question,
                catalog=SCHEMA_CATALOG,
                engine=self.engine,
                top_k=self.top_k,
                min_score=self.min_score,
                expand_relations=self.expand_relations,
            )
        except OperationalError:
            message = (
                "Database connection failed. Check the database username, password, "
                "network/VPN access, and that the MySQL user is allowed from your current host."
            )
            logger.error("Retriever failed: %s", message)
            ei = error_info_from_exception(DatabaseConnectionError(message), stage="database")
            return {
                "question": question,
                "retrieved_tables": [],
                "sql": None,
                "valid": False,
                "error": ei.error,
                "error_info": ei.to_dict(),
            }
        except Exception as exc:
            logger.error("Retriever failed: %s", exc, exc_info=True)
            ei = error_info_from_exception(exc, stage="retriever")
            return {
                "question": question,
                "retrieved_tables": [],
                "sql": None,
                "valid": False,
                "error": ei.error,
                "error_info": ei.to_dict(),
            }

        selected_tables = retriever_result["selected_tables"]
        semantic_parse = retriever_result["semantic_parse"]
        detected_entities = retriever_result["detected_entities"]
        entity_strategies = retriever_result["entity_strategies"]

        # Step 2: Check sufficient context
        if not selected_tables:
            logger.warning("No tables selected by retriever.")
            return {
                "question": question,
                "retrieved_tables": [],
                "sql": None,
                "valid": False,
                "error": "Insufficient schema context",
            }

        # Step 3: Build prompt
        logger.info("Stage: prompt construction (tables=%s)", selected_tables)
        schema_context = build_schema_context(selected_tables, SCHEMA_CATALOG)
        schema_context = qualify_schema_context(schema_context, self.db_schema)
        entity_context = build_entity_context(detected_entities, entity_strategies)
        join_context = build_join_context(selected_tables, self.db_schema)

        prompt = build_prompt(
            question=question,
            selected_tables=selected_tables,
            schema_context=schema_context,
            entity_context=entity_context,
            join_context=join_context,
            semantic_parse=semantic_parse,
        )

        # Step 4: Call LLM for SQL generation
        logger.info("Stage: LLM SQL generation")
        try:
            sql = self._llm_client.generate_sql(prompt)
        except LLMError as exc:
            logger.error("LLM SQL generation failed: %s", exc)
            ei = error_info_from_exception(exc)
            return {
                "question": question,
                "retrieved_tables": selected_tables,
                "sql": None,
                "valid": False,
                "error": f"LLM API error: {exc}",
                "error_info": ei.to_dict(),
            }
        except Exception as exc:
            logger.error("Unexpected error during LLM call: %s", exc, exc_info=True)
            ei = error_info_from_exception(exc, stage="llm")
            return {
                "question": question,
                "retrieved_tables": selected_tables,
                "sql": None,
                "valid": False,
                "error": f"LLM API error: {exc}",
                "error_info": ei.to_dict(),
            }

        # Step 5: Validate SQL
        logger.info("Stage: SQL validation")
        # Restrict to tables the retriever selected + base entity tables (always joinable)
        _BASE_TABLES = {"disease", "drug", "gene", "pathway", "encodes"}
        allowed_tables = set(selected_tables) | _BASE_TABLES
        validation = validate_sql(sql, allowed_tables)

        if not validation["valid"]:
            logger.warning("SQL validation failed: %s", validation["error"])
            return {
                "question": question,
                "retrieved_tables": selected_tables,
                "sql": sql,
                "valid": False,
                "error": validation["error"],
            }

        logger.info("SQL generation successful.")
        logger.debug("Generated SQL: %s", sql[:300])
        return {
            "question": question,
            "retrieved_tables": selected_tables,
            "sql": sql,
            "valid": True,
        }

    def answer(self, question: str, max_answer_rows: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate SQL, execute it, and return a natural language answer.

        If answer generation fails, still returns SQL and rows if available.
        """
        max_rows = max_answer_rows or self.max_answer_rows
        logger.info("Pipeline.answer() called: %s", question[:120])

        gen = self.generate(question)

        base: Dict[str, Any] = {
            "question": gen["question"],
            "retrieved_tables": gen["retrieved_tables"],
            "sql": gen.get("sql"),
            "valid": gen["valid"],
            "rows": [],
            "row_count": 0,
            "columns": [],
            "truncated": False,
            "answer": None,
            "error": gen.get("error"),
        }
        if gen.get("error_info"):
            base["error_info"] = gen["error_info"]

        if not gen["valid"]:
            return base

        # Execute SQL
        logger.info("Stage: SQL execution")
        try:
            exec_result = execute_sql(self.engine, gen["sql"], max_rows=max_rows)
        except (SQLExecutionError, DatabaseConnectionError) as exc:
            logger.error("SQL execution failed: %s", exc, exc_info=True)
            ei = error_info_from_exception(exc)
            base["error"] = str(exc)
            base["error_info"] = ei.to_dict()
            return base

        base["rows"] = exec_result["rows"]
        base["columns"] = exec_result["columns"]
        base["row_count"] = exec_result["row_count"]
        base["truncated"] = exec_result["truncated"]

        # Generate natural language answer
        logger.info("Stage: answer generation")
        answer_prompt = build_answer_prompt(
            question=question,
            sql=gen["sql"],
            columns=exec_result["columns"],
            rows=exec_result["rows"],
            row_count=exec_result["row_count"],
            truncated=exec_result["truncated"],
            max_rows=max_rows,
            nl_row_limit=20,
        )

        try:
            base["answer"] = self._llm_client.generate_answer(answer_prompt)
            logger.info("Answer generation successful.")
        except LLMError as exc:
            # Answer generation failed but we still have SQL + rows
            logger.error("Answer generation failed: %s", exc)
            base["error"] = f"Answer generation error: {exc}"
            ei = error_info_from_exception(exc, stage="answer_generation")
            base["error_info"] = ei.to_dict()
        except Exception as exc:
            logger.error("Unexpected error during answer generation: %s", exc, exc_info=True)
            base["error"] = f"Answer generation error: {exc}"

        return base


def create_pipeline(
    db_url: str,
    gemini_model: str = None,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
    top_k: int = 4,
    min_score: int = 2,
    expand_relations: bool = True,
    temperature: float = 0.0,
    db_schema: str = DEFAULT_DB_SCHEMA,
    log_level: Optional[str] = None,
    api_key: Optional[str] = None,
) -> DriveQAPipeline:
    """
    Factory function to create a configured DriveQA pipeline.

    Args:
        db_url: SQLAlchemy database URL for the DRIVE database.
        gemini_model: Gemini model name to use (backward compat).
        model_name: Backward-compatible alias for gemini_model.
        model_id: Model identifier from the model registry. If provided,
            overrides gemini_model and uses the registry for configuration.
        top_k: Max tables the retriever returns.
        min_score: Minimum retriever score threshold.
        expand_relations: Whether to expand related tables.
        temperature: Deprecated — ignored when model_id is used.
        db_schema: Database schema/name to qualify table references.
        log_level: Optional log level override (DEBUG, INFO, WARNING, ERROR).
        api_key: API key for the selected model.

    Returns:
        Configured DriveQAPipeline instance.
    """
    setup_logging(level=log_level)

    if model_name is not None and model_id is None:
        gemini_model = model_name

    engine = create_engine(db_url)

    # New path: use model registry
    if model_id is not None:
        logger.info(
            "Creating pipeline via registry (model_id=%s, top_k=%d, min_score=%d, db_schema=%s)",
            model_id,
            top_k,
            min_score,
            db_schema,
        )
        llm_client = create_llm_client(model_id=model_id, api_key=api_key or "")
        return DriveQAPipeline(
            engine=engine,
            top_k=top_k,
            min_score=min_score,
            expand_relations=expand_relations,
            db_schema=db_schema,
            llm_client=llm_client,
        )

    # Legacy path: direct Gemini configuration
    logger.info(
        "Creating pipeline (model=%s, top_k=%d, min_score=%d, db_schema=%s)",
        gemini_model,
        top_k,
        min_score,
        db_schema,
    )

    gemini_config = GeminiConfig(
        model_name=gemini_model, temperature=temperature, _api_key=api_key or ""
    )
    gemini_client = GeminiClient(config=gemini_config)

    return DriveQAPipeline(
        engine=engine,
        gemini_client=gemini_client,
        top_k=top_k,
        min_score=min_score,
        expand_relations=expand_relations,
        db_schema=db_schema,
    )


# Backward-compatible alias
NLtoSQLPipeline = DriveQAPipeline
