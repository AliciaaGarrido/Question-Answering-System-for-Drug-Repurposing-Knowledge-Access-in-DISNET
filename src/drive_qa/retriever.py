"""
Schema retriever for the DRIVE QA pipeline.

Orchestrates semantic parsing, entity resolution, table scoring, and context
building to select the most relevant database tables for a natural language question.

Public interface (compatible with scripts/evaluation/retriever_evaluator_v2.py):
- SCHEMA_CATALOG: Dict[str, TableInfo]
- retrieve_schema(question, catalog, engine, ...) -> Dict[str, Any]
- build_schema_context(selected_tables, catalog) -> str
- build_entity_context(detected_entities, entity_strategies) -> str
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from sqlalchemy.engine import Engine

from drive_qa.entity_resolution import (
    DEFAULT_LIMIT_PER_TYPE,
    DEFAULT_MAX_ENTITIES_PER_TYPE,
    filter_detected_entities_for_scoring,
    infer_entity_strategy,
    is_confident_entity_strategy,
    search_entities_global,
)
from drive_qa.schema_catalog import (
    ENTITY_TABLE_CONFIG,
    SCHEMA_CATALOG,
    TableInfo,
)
from drive_qa.semantic_parser import (
    SemanticParse,
    _detect_comparison_pattern,
    parse_question,
)
from drive_qa.table_scoring import (
    apply_cross_model_exclusion_gating,
    apply_db_entity_boosts,
    apply_intent_boosts,
    apply_lexical_scores,
    apply_metric_gating,
    apply_metric_model_boosts,
    apply_metric_tiebreak,
    apply_method_primary_tiebreak,
    apply_primary_table_intent_boost,
    apply_result_bridge_boosts,
    apply_result_type_boosts,
    apply_template_boosts,
    enforce_required_tables,
    expand_related_tables_semantic,
    init_scores,
)
from drive_qa.text_normalization import normalize_text

logger = logging.getLogger(__name__)


def retrieve_schema(
    question: str,
    catalog: Dict[str, TableInfo],
    engine: Engine,
    top_k: int = 5,
    min_score: int = 2,
    expand_relations: bool = True,
    limit_per_type: int = DEFAULT_LIMIT_PER_TYPE,
    max_entities_per_type: int = DEFAULT_MAX_ENTITIES_PER_TYPE,
    prefer_single_exact: bool = True,
) -> Dict[str, Any]:
    """
    Main retrieval function. Selects relevant tables for a NL question.

    Uses single-pass global entity search across all supported entity types.
    Parser-predicted entity types are retained as semantic signals for scoring,
    diagnostics, and evaluation, not as hard filters.

    Returns a dict with:
    - question, semantic_parse, detected_entities, entity_strategies
    - ranking (sorted table scores)
    - selected_tables (final selection)
    - scores_debug (detailed reasons)
    """
    parse = parse_question(question)

    # Detect comparative/enumeration patterns
    is_comparison = _detect_comparison_pattern(question)
    effective_prefer_single = False if is_comparison else prefer_single_exact
    effective_max_entities = max(max_entities_per_type, 10) if is_comparison else max_entities_per_type

    scores, reasons_map = init_scores(catalog)
    apply_lexical_scores(question, catalog, scores, reasons_map)

    # Single-pass global entity search with parser-predicted entity types
    detected_entities = search_entities_global(
        engine=engine,
        question=question,
        predicted_entity_types=parse.predicted_entity_types,
        limit_per_type=limit_per_type,
        max_entities_per_type=effective_max_entities,
        prefer_single_exact=effective_prefer_single,
    )

    entity_strategies = infer_entity_strategy(question, detected_entities)

    # Expand predicted_entity_types with confidently found entities
    confident_types = [
        entity_type
        for entity_type, info in entity_strategies.items()
        if is_confident_entity_strategy(info)
    ]

    if confident_types:
        parse.predicted_entity_types = list(dict.fromkeys(parse.predicted_entity_types + confident_types))

    # If both disease and drug confidently resolved, treat as pair grounding
    if "diseases" in confident_types and "drugs" in confident_types:
        parse.predicted_entity_types = ["diseases", "drugs"]
        if parse.operator is None:
            parse.result_types_requested = []
            parse.query_template = "association_lookup"

    # Use only confident entities for scoring
    detected_entities_for_scoring = filter_detected_entities_for_scoring(
        detected_entities,
        entity_strategies,
    )

    # Apply all scoring layers
    apply_db_entity_boosts(detected_entities_for_scoring, entity_strategies, scores, reasons_map)
    apply_result_type_boosts(parse.result_types_requested, scores, reasons_map)
    apply_intent_boosts(parse.intents, scores, reasons_map)
    apply_primary_table_intent_boost(parse, scores, reasons_map, entity_strategies)
    apply_metric_model_boosts(parse, scores, reasons_map)
    apply_cross_model_exclusion_gating(parse, scores, reasons_map)
    apply_template_boosts(parse, scores, reasons_map)
    apply_result_bridge_boosts(parse.result_types_requested, parse.predicted_entity_types, scores, reasons_map)
    apply_metric_gating(parse, scores, reasons_map)
    apply_metric_tiebreak(parse, scores, reasons_map)
    apply_method_primary_tiebreak(parse, scores, reasons_map)

    # Rank and select
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_tables = [table for table, score in ranked if score >= min_score][:top_k]
    selected = enforce_required_tables(parse, entity_strategies, ranked_tables)

    if expand_relations:
        selected = expand_related_tables_semantic(selected, catalog, scores)

    selected = list(dict.fromkeys(selected))

    return {
        "question": question,
        "semantic_parse": {
            "intents": parse.intents,
            "metric": parse.metric,
            "operator": parse.operator,
            "result_types_requested": parse.result_types_requested,
            "predicted_entity_types": parse.predicted_entity_types,
            "query_template": parse.query_template,
            "candidate_spans": parse.candidate_spans,
            "pattern_targets": parse.pattern_targets,
            "gnn_subtype": parse.method_subtype,
            "is_cross_method": parse.is_cross_method,
        },
        "detected_entities": detected_entities,
        "entity_strategies": entity_strategies,
        "ranking": ranked,
        "selected_tables": selected,
        "scores_debug": reasons_map,
    }


def build_schema_context(selected_tables: List[str], catalog: Dict[str, TableInfo]) -> str:
    """Build a textual schema context for the LLM prompt."""
    lines = []

    for table_name in selected_tables:
        if table_name not in catalog:
            continue

        table = catalog[table_name]
        lines.append(f"TABLE {table.name}")
        lines.append(f"Description: {table.description}")
        lines.append(f"Columns: {', '.join(table.columns)}")

        related_known = [t for t in table.related_tables if t in catalog]
        if related_known:
            lines.append(f"Related tables: {', '.join(related_known)}")

        lines.append("")

    return "\n".join(lines)


def build_entity_context(
    detected_entities: Dict[str, List[Dict[str, Any]]],
    entity_strategies: Dict[str, Dict[str, Any]],
) -> str:
    """Build a textual entity context for the LLM prompt."""
    lines = []

    for entity_type, candidates in detected_entities.items():
        strategy = entity_strategies.get(entity_type, {}).get("strategy", "no_entity")
        selected = entity_strategies.get(entity_type, {}).get("selected")

        lines.append(f"{entity_type.upper()}:")
        lines.append(f"  strategy: {strategy}")

        if strategy == "multiple_exact" and isinstance(selected, list):
            lines.append("  selected (multiple):")
            for s in selected:
                lines.append(
                    f"    - id={s['id']} | name={s['name']} | match_score={s['match_score']}"
                )
        elif selected and isinstance(selected, dict):
            lines.append(
                f"  selected: id={selected['id']} | name={selected['name']} | match_score={selected['match_score']}"
            )

        if candidates:
            lines.append("  candidates:")
            for c in candidates:
                lines.append(
                    f"    - id={c['id']} | name={c['name']} | match_score={c['match_score']}"
                )
        else:
            lines.append("  candidates: []")

        lines.append("")

    return "\n".join(lines)
