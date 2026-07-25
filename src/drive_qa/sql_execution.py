"""
SQL execution — safe execution of validated SELECT queries.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.engine import Engine

from drive_qa.errors import DatabaseConnectionError, SQLExecutionError
from drive_qa.logging_config import get_logger

logger = get_logger("sql_execution")


def execute_sql(engine: Engine, sql: str, max_rows: int = 50) -> Dict[str, Any]:
    """
    Execute a validated SELECT query and return serialized results.

    Returns a dict with keys: rows, columns, row_count, truncated, error.
    """
    logger.info("Executing SQL (max_rows=%d)", max_rows)
    logger.debug("SQL: %s", sql[:200])

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            # Fetch only max_rows+1 to detect truncation without loading entire result set
            batch = result.mappings().fetchmany(max_rows + 1)
            row_count = len(batch)
            truncated = row_count > max_rows
            rows = [_serialize_row(dict(r)) for r in batch[:max_rows]]

        if truncated:
            logger.warning(
                "Results truncated: %d rows returned, showing %d.",
                row_count,
                max_rows,
            )

        logger.info("SQL execution successful: %d rows returned.", row_count)
        return {
            "rows": rows,
            "columns": columns,
            "row_count": row_count,
            "truncated": truncated,
            "error": None,
        }
    except Exception as e:
        error_msg = str(e)
        # Distinguish connection errors from execution errors
        if "Can't connect" in error_msg or "Connection refused" in error_msg:
            logger.error("Database connection error: %s", error_msg)
            raise DatabaseConnectionError(f"Database connection error: {error_msg}") from e
        logger.error("SQL execution error: %s", error_msg, exc_info=True)
        raise SQLExecutionError(f"SQL execution error: {error_msg}") from e


# ─── Serialization helpers ────────────────────────────────────────────────────

def _serialize_value(val: Any) -> Any:
    """Convert a single value to a JSON-serializable type."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, date):
        return val.isoformat()
    if isinstance(val, timedelta):
        return str(val)
    if isinstance(val, bytes):
        return val[:200].hex()
    return str(val)


def _serialize_row(row: dict) -> dict:
    """Convert all values in a row dict to JSON-serializable types."""
    return {k: _serialize_value(v) for k, v in row.items()}
