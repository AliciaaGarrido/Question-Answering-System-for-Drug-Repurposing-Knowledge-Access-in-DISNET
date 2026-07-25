"""
SQL validation — safety checks for generated SQL queries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from drive_qa.errors import SQLValidationError
from drive_qa.logging_config import get_logger

logger = get_logger("sql_validation")

FORBIDDEN_SQL_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "RENAME", "LOAD",
    "CALL", "EXECUTE", "EXEC",
}


def validate_sql(sql: str, allowed_tables: Set[str]) -> Dict[str, Any]:
    """
    Validate generated SQL for safety and correctness.

    Returns:
        {"valid": True} or {"valid": False, "error": str}
    """
    if not sql or not sql.strip():
        logger.warning("Empty SQL received for validation.")
        return {"valid": False, "error": "Empty SQL generated."}

    sql_clean = sql.strip().rstrip(";")

    # Check for multiple statements
    statements = _split_statements(sql_clean)
    if len(statements) > 1:
        logger.warning("Multiple SQL statements detected.")
        return {"valid": False, "error": "Multiple SQL statements detected. Only single SELECT allowed."}

    sql_upper = sql_clean.upper()

    # Must start with SELECT or WITH
    first_keyword = sql_upper.lstrip().split()[0] if sql_upper.strip() else ""
    if first_keyword not in ("SELECT", "WITH"):
        logger.warning("SQL does not start with SELECT/WITH: %s", first_keyword)
        return {"valid": False, "error": f"SQL must start with SELECT or WITH. Found: {first_keyword}"}

    # If WITH, must end with a SELECT
    if first_keyword == "WITH":
        if not _with_ends_in_select(sql_upper):
            return {"valid": False, "error": "WITH clause must end with a SELECT statement."}

    # Check for forbidden keywords
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, sql_upper):
            logger.warning("Forbidden SQL keyword detected: %s", keyword)
            return {"valid": False, "error": f"Forbidden SQL keyword detected: {keyword}"}

    # Check for unknown tables
    used_tables = _extract_table_references(sql_clean)
    unknown = used_tables - allowed_tables
    if unknown:
        logger.warning("Unknown tables referenced: %s", sorted(unknown))
        return {"valid": False, "error": f"Unknown tables referenced: {sorted(unknown)}"}

    # Warn about unqualified table references (informational, does not block)
    unqualified = re.findall(
        r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\.)',
        re.sub(r'\s+', ' ', sql_clean),
        re.IGNORECASE,
    )
    for name in unqualified:
        if name.lower() in allowed_tables:
            logger.warning("Table '%s' referenced without schema prefix.", name)

    logger.debug("SQL validation passed.")
    return {"valid": True}


def _split_statements(sql: str) -> List[str]:
    """Split SQL on semicolons that are not inside string literals."""
    parts: List[str] = []
    current: List[str] = []
    in_single_quote = False
    in_double_quote = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif ch == ';' and not in_single_quote and not in_double_quote:
            stmt = ''.join(current).strip()
            if stmt:
                parts.append(stmt)
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1

    stmt = ''.join(current).strip()
    if stmt:
        parts.append(stmt)

    return parts


def _with_ends_in_select(sql_upper: str) -> bool:
    """Check that a WITH...AS(...) block ends in SELECT.

    Limitations:
    - May give false positives if forbidden keywords appear inside string literals.
    - Does not fully parse nested CTEs.
    - Sufficient for the DRIVE schema where LLM-generated CTEs are simple.
    """
    last_select_pos = sql_upper.rfind("SELECT")
    if last_select_pos == -1:
        return False
    after_select = sql_upper[last_select_pos:]
    for kw in FORBIDDEN_SQL_KEYWORDS:
        if re.search(r'\b' + kw + r'\b', after_select):
            return False
    return True


# Keywords that may follow FROM/JOIN but are NOT table names
_SQL_KEYWORDS_NOT_TABLES = {
    'SELECT', 'WHERE', 'ON', 'AND', 'OR', 'AS',
    'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS',
    'NATURAL', 'FULL', 'LATERAL', 'GROUP', 'ORDER',
    'HAVING', 'UNION', 'LIMIT', 'SET', 'CASE', 'WHEN',
    'THEN', 'ELSE', 'END', 'EXISTS', 'NOT', 'IN',
    'BETWEEN', 'LIKE', 'IS', 'NULL', 'TRUE', 'FALSE',
    'DISTINCT', 'ALL', 'ANY', 'SOME', 'VALUES',
}


def _extract_table_references(sql: str) -> Set[str]:
    """
    Extract table names referenced in FROM and JOIN clauses.
    Excludes CTE names defined in WITH clauses.
    """
    tables: Set[str] = set()
    sql_normalized = re.sub(r'\s+', ' ', sql)

    # Extract CTE names to exclude them
    cte_names: Set[str] = set()
    cte_defs = re.findall(r'(\w+)\s+AS\s*\(', sql_normalized, re.IGNORECASE)
    cte_names = {name.lower() for name in cte_defs}

    simple_cte = re.findall(r'\bWITH\s+(\w+)\s+AS\b', sql_normalized, re.IGNORECASE)
    cte_names.update(name.lower() for name in simple_cte)

    # Match FROM/JOIN table or schema.table patterns (skip subqueries via negative lookahead)
    from_join_pattern = r'(?:FROM|JOIN)\s+(?!\()(?:[a-zA-Z_]\w*\.)?([a-zA-Z_][a-zA-Z0-9_]*)'
    matches = re.findall(from_join_pattern, sql_normalized, re.IGNORECASE)
    for match in matches:
        if match.upper() not in _SQL_KEYWORDS_NOT_TABLES:
            name_lower = match.lower()
            if name_lower not in cte_names:
                tables.add(name_lower)

    return tables
