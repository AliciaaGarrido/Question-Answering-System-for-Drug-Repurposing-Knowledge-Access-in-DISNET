"""
Entity resolution: searching and scoring entities in the DRIVE database.

Responsibilities:
- Scoring entity matches (string similarity)
- Fetching candidates from DB via SQL with SQL-level match ordering
- Single global entity search across all entity types
- Single-pass search across every supported entity type
- Strategy inference (exact_match, best_candidate, ambiguous, etc.)
- Filtering detected entities
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from drive_qa.schema_catalog import ENTITY_TABLE_CONFIG
from drive_qa.semantic_parser import (
    detect_pattern_search_targets,
    extract_candidate_spans,
)
from drive_qa.text_normalization import normalize_text

# Default limits for global entity search (derived from original two-stage limits)
# Original Stage 1: limit_per_type=10, max_entities_per_type=5
# Original Stage 2: limit_per_type=5-8, max_entities_per_type=2-3
# Combined: limit_per_type=15 covers both passes, max_entities_per_type=5 preserves quality
DEFAULT_LIMIT_PER_TYPE = 15
DEFAULT_MAX_ENTITIES_PER_TYPE = 5

_VALID_SQL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _validate_sql_identifier(name: str, context: str) -> None:
    """Validate that a string is a safe SQL identifier (defense in depth)."""
    if not _VALID_SQL_IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier for {context}: {name!r}")


def score_entity_match(span: str, row_name: str) -> int:
    span_norm = normalize_text(span)
    row_norm = normalize_text(row_name)

    if not span_norm or not row_norm:
        return 0
    if span_norm == row_norm:
        return 100
    if row_norm in span_norm:
        return 95
    if span_norm in row_norm:
        return 90

    span_tokens = set(span_norm.split())
    row_tokens = set(row_norm.split())
    overlap = len(span_tokens & row_tokens)
    if overlap == 0:
        return 0

    jacc = overlap / max(1, len(span_tokens | row_tokens))
    score = int(60 + 30 * jacc)

    if row_norm.startswith(span_norm) or span_norm.startswith(row_norm):
        score += 5

    return min(score, 89)


def fetch_candidates_for_span(
    conn,
    table_name: str,
    id_col: str,
    name_col: str,
    span: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fetch entity candidates from a single table for a given span.

    SQL-level ordering prioritises rows before LIMIT is applied:
    1. Exact normalised match
    2. Name starts with the candidate span
    3. Name contains the candidate span
    4. Comma-stripped exact match
    5. Comma-stripped containment match
    6. Other matches
    Within each tier, shorter names are preferred.
    """
    _validate_sql_identifier(table_name, "table_name")
    _validate_sql_identifier(id_col, "id_col")
    _validate_sql_identifier(name_col, "name_col")

    span_norm = normalize_text(span)
    if not span_norm or len(span_norm) < 5:
        return []

    single_token = len(span_norm.split()) == 1
    if single_token and len(span_norm) < 6:
        return []

    # SQL with explicit ORDER BY CASE to prioritise match quality before LIMIT
    sql = text(f"""
        SELECT {id_col} AS entity_id, {name_col} AS entity_name
        FROM {table_name}
        WHERE LOWER({name_col}) = :exact
           OR LOWER({name_col}) LIKE :starts_with
           OR LOWER({name_col}) LIKE :contains
           OR REPLACE(REPLACE(LOWER({name_col}), ',', ''), '  ', ' ') = :exact_no_comma
           OR REPLACE(REPLACE(LOWER({name_col}), ',', ''), '  ', ' ') LIKE :contains_no_comma
        ORDER BY
            CASE
                WHEN LOWER({name_col}) = :exact THEN 1
                WHEN LOWER({name_col}) LIKE :starts_with THEN 2
                WHEN LOWER({name_col}) LIKE :contains THEN 3
                WHEN REPLACE(REPLACE(LOWER({name_col}), ',', ''), '  ', ' ') = :exact_no_comma THEN 4
                WHEN REPLACE(REPLACE(LOWER({name_col}), ',', ''), '  ', ' ') LIKE :contains_no_comma THEN 5
                ELSE 6
            END,
            LENGTH({name_col})
        LIMIT :limit
    """)

    rows = conn.execute(sql, {
        "exact": span_norm,
        "starts_with": f"{span_norm}%",
        "contains": f"%{span_norm}%",
        "exact_no_comma": span_norm,
        "contains_no_comma": f"%{span_norm}%",
        "limit": limit,
    }).mappings().all()

    return [{"id": r["entity_id"], "name": r["entity_name"]} for r in rows]


def search_entities_in_db(
    engine: Engine,
    question: str,
    entity_types: List[str],
    limit_per_type: int = 10,
    max_entities_per_type: int = 5,
    prefer_single_exact: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search entities of specified types in the database.

    .. deprecated::
        Use search_entities_global() instead for single-pass global search
        with global coverage of every supported entity type.
    """
    results: Dict[str, List[Dict[str, Any]]] = {"diseases": [], "drugs": [], "genes": [], "pathways": []}
    candidate_spans = extract_candidate_spans(question)

    if not candidate_spans or not entity_types:
        return results

    low_signal_spans = {
        "estan", "están", "esta", "está", "presentes", "aparecen", "aparece",
        "tienen", "tiene", "cuantas", "cuántas", "cuantos", "cuántos",
        "todas", "todos", "hipotesis", "hipótesis", "tablas", "metodos", "métodos",
        "threshold", "thresholds", "umbral", "umbrales", "value", "values", "valor", "valores",
        "model", "modelo", "method", "metodo", "método", "based",
    }

    with engine.connect() as conn:
        for entity_type in entity_types:
            if entity_type not in ENTITY_TABLE_CONFIG:
                continue

            table_name, id_col, name_col = ENTITY_TABLE_CONFIG[entity_type]
            scored: List[Dict[str, Any]] = []

            for span in candidate_spans:
                span_norm = normalize_text(span)
                if len(span_norm) < 5:
                    continue
                if span_norm in low_signal_spans:
                    continue
                if any(tok in low_signal_spans for tok in span_norm.split()) and len(span_norm.split()) == 1:
                    continue
                if entity_type == "diseases" and re.search(r"(threshold|umbral|values?|valores?)", span_norm):
                    continue
                if len(span_norm.split()) == 1 and len(span_norm) < 6:
                    continue

                rows = fetch_candidates_for_span(conn, table_name, id_col, name_col, span, limit=limit_per_type)
                for row in rows:
                    row["matched_span"] = span
                    row["match_score"] = score_entity_match(span, row["name"])
                    if row["match_score"] >= 70:
                        scored.append(row)

            best_by_id: Dict[str, Dict[str, Any]] = {}
            for row in scored:
                prev = best_by_id.get(row["id"])
                if prev is None or row["match_score"] > prev["match_score"]:
                    best_by_id[row["id"]] = row

            final = sorted(best_by_id.values(), key=lambda x: x["match_score"], reverse=True)

            if prefer_single_exact and final and final[0]["match_score"] >= 95:
                results[entity_type] = [final[0]]
            else:
                results[entity_type] = final[:max_entities_per_type]

    return results


def search_entities_global(
    engine: Engine,
    question: str,
    predicted_entity_types: Optional[List[str]] = None,
    limit_per_type: int = DEFAULT_LIMIT_PER_TYPE,
    max_entities_per_type: int = DEFAULT_MAX_ENTITIES_PER_TYPE,
    prefer_single_exact: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Single-pass global entity search across all supported entity types.

    This replaces the previous two-stage search (targeted + broad) with a unified
    approach that:
    1. Searches all entity types in a single pass
    2. Uses SQL-level ordering to prioritise better matches before LIMIT
    3. Preserves parser-predicted entity types for diagnostics/future ranking

    Args:
        engine: SQLAlchemy database engine
        question: Natural language question to extract entity mentions from
        predicted_entity_types: Entity types predicted by the Semantic Parser.
            The current result structure is grouped by entity type, so this
            value is not a hard filter and does not change cross-type ranking.
        limit_per_type: Maximum candidates to fetch per entity type per span
        max_entities_per_type: Maximum entities to retain per type after scoring
        prefer_single_exact: If True and top candidate has score >= 95,
            return only that candidate for the entity type

    Returns:
        Dict mapping entity types to lists of candidate dicts with keys:
        id, name, match_score, matched_span

    Scoring and ranking:
        - Lexical similarity score (0-100) is the primary ranking criterion
        - Predicted entity types are currently grouped with their own type
        - Among equal-score candidates, the one matched by the longer span is
          preferred, because a longer matched span indicates a more specific
          entity mention (e.g. "Acquired Immunodeficiency Syndrome" should
          win over "Syndrome" when the question explicitly names the full term)
        - The lexical score is preserved unchanged for strategy classification

    Preserved behaviours:
        - exact_match score of 100
        - containment scores of 95 and 90
        - token-overlap scoring (60-89)
        - minimum score threshold of 70
        - score cap of 89 for approximate matches
        - deduplication by database identifier
        - preservation of highest lexical score for duplicate records
    """
    all_entity_types = list(ENTITY_TABLE_CONFIG.keys())
    predicted_set = set(predicted_entity_types or [])

    results: Dict[str, List[Dict[str, Any]]] = {et: [] for et in all_entity_types}
    candidate_spans = extract_candidate_spans(question)

    if not candidate_spans:
        return results

    # Low-signal spans that should be skipped to avoid spurious matches
    low_signal_spans = {
        "estan", "están", "esta", "está", "presentes", "aparecen", "aparece",
        "tienen", "tiene", "cuantas", "cuántas", "cuantos", "cuántos",
        "todas", "todos", "hipotesis", "hipótesis", "tablas", "metodos", "métodos",
        "threshold", "thresholds", "umbral", "umbrales", "value", "values", "valor", "valores",
        "model", "modelo", "method", "metodo", "método", "based",
    }

    with engine.connect() as conn:
        for entity_type in all_entity_types:
            if entity_type not in ENTITY_TABLE_CONFIG:
                continue

            table_name, id_col, name_col = ENTITY_TABLE_CONFIG[entity_type]
            scored: List[Dict[str, Any]] = []

            for span in candidate_spans:
                span_norm = normalize_text(span)
                if len(span_norm) < 5:
                    continue
                if span_norm in low_signal_spans:
                    continue
                if any(tok in low_signal_spans for tok in span_norm.split()) and len(span_norm.split()) == 1:
                    continue
                # Skip disease-specific threshold-related terms
                if entity_type == "diseases" and re.search(r"(threshold|umbral|values?|valores?)", span_norm):
                    continue
                if len(span_norm.split()) == 1 and len(span_norm) < 6:
                    continue

                rows = fetch_candidates_for_span(conn, table_name, id_col, name_col, span, limit=limit_per_type)
                for row in rows:
                    row["matched_span"] = span
                    row["entity_type"] = entity_type
                    row["match_score"] = score_entity_match(span, row["name"])
                    if row["match_score"] >= 70:
                        scored.append(row)

            # Deduplication: keep highest-scoring entry per entity ID.
            # Also track the length of the span that produced the best score,
            # so that among equal-score candidates the one matched by a longer
            # (more specific) span is preferred over one matched by a shorter span.
            best_by_id: Dict[str, Dict[str, Any]] = {}
            for row in scored:
                prev = best_by_id.get(row["id"])
                if prev is None or row["match_score"] > prev["match_score"]:
                    best_by_id[row["id"]] = row

            # Sort candidates inside this entity-type bucket:
            # Primary: lexical score (higher is better)
            # Secondary: parser-predicted type flag. Since candidates are
            # grouped by entity type, this is constant within the bucket and
            # currently has no observable ranking effect.
            # Tertiary: length of the matched span (longer span = more specific mention)
            # Using matched-span length rather than database-name length avoids
            # preferring generic shorter names over specific multi-word entities
            # when both are matched with the same lexical score.
            is_predicted = entity_type in predicted_set
            final = sorted(
                best_by_id.values(),
                key=lambda x: (
                    -x["match_score"],                       # Primary: higher score first
                    0 if is_predicted else 1,                # Secondary: predicted type first
                    -len(normalize_text(x.get("matched_span", "")))  # Tertiary: longer span first
                )
            )

            if prefer_single_exact and final and final[0]["match_score"] >= 95:
                results[entity_type] = [final[0]]
            else:
                results[entity_type] = final[:max_entities_per_type]

    return results


def infer_entity_strategy(
    question: str,
    detected_entities: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    pattern_targets = detect_pattern_search_targets(question)
    strategies: Dict[str, Dict[str, Any]] = {}

    for entity_type, candidates in detected_entities.items():
        if pattern_targets.get(entity_type, False):
            strategies[entity_type] = {
                "strategy": "pattern_search",
                "selected": None,
                "candidates": candidates,
            }
            continue

        if not candidates:
            strategies[entity_type] = {
                "strategy": "no_entity",
                "selected": None,
                "candidates": [],
            }
            continue

        if len(candidates) == 1:
            c = candidates[0]
            strategies[entity_type] = {
                "strategy": "exact_match" if c["match_score"] >= 95 else "best_candidate",
                "selected": c,
                "candidates": candidates,
            }
            continue

        high_score_candidates = [c for c in candidates if c["match_score"] >= 90]
        if len(high_score_candidates) >= 2:
            strategies[entity_type] = {
                "strategy": "multiple_exact",
                "selected": high_score_candidates,
                "candidates": candidates,
            }
            continue

        top1, top2 = candidates[0], candidates[1]
        if top1["match_score"] - top2["match_score"] >= 10:
            strategies[entity_type] = {
                "strategy": "best_candidate",
                "selected": top1,
                "candidates": candidates,
            }
        else:
            strategies[entity_type] = {
                "strategy": "ambiguous",
                "selected": top1,
                "candidates": candidates,
            }

    return strategies


def merge_detected_entities(
    primary: Dict[str, List[Dict[str, Any]]],
    secondary: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Merge entity results from two search passes.

    .. deprecated::
        This function was used by the two-stage search approach.
        With search_entities_global(), merging is no longer needed.
        Kept for backward compatibility.
    """
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for entity_type in {"diseases", "drugs", "genes", "pathways"}:
        combined = (primary.get(entity_type, []) or []) + (secondary.get(entity_type, []) or [])
        best_by_id: Dict[str, Dict[str, Any]] = {}
        for row in combined:
            row_id = row.get("id")
            if row_id is None:
                continue
            prev = best_by_id.get(row_id)
            if prev is None or row.get("match_score", 0) > prev.get("match_score", 0):
                best_by_id[row_id] = row
        merged[entity_type] = sorted(
            best_by_id.values(),
            key=lambda x: (-x.get("match_score", 0), len(x.get("name", "")))
        )
    return merged


def is_confident_entity_strategy(strategy_info: Dict[str, Any]) -> bool:
    if not strategy_info:
        return False
    strategy = strategy_info.get("strategy")
    selected = strategy_info.get("selected")
    if strategy == "exact_match":
        return True
    if strategy == "multiple_exact":
        return True
    if strategy == "best_candidate" and selected and selected.get("match_score", 0) >= 95:
        return True
    return False


def filter_detected_entities_for_scoring(
    detected_entities: Dict[str, List[Dict[str, Any]]],
    entity_strategies: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for entity_type, rows in detected_entities.items():
        info = entity_strategies.get(entity_type, {})
        strategy = info.get("strategy")
        selected = info.get("selected")

        if strategy == "exact_match":
            filtered[entity_type] = rows
        elif strategy == "multiple_exact":
            filtered[entity_type] = rows
        elif strategy == "best_candidate" and selected and selected.get("match_score", 0) >= 95:
            filtered[entity_type] = [selected]
        else:
            filtered[entity_type] = []
    return filtered
