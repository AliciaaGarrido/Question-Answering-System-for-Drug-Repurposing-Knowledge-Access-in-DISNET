"""
Table scoring: lexical scoring, boosts, penalties, gating, and table selection.

Responsibilities:
- Lexical scoring of tables against a question
- Entity-based boosts
- Intent-based boosts
- Metric-based model boosts
- Cross-model exclusion gating
- Tiebreaks
- Required table enforcement
- Related table expansion
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from drive_qa.entity_resolution import is_confident_entity_strategy
from drive_qa.schema_catalog import (
    ENTITY_BASE_TABLES,
    RESULT_INTENT_BRIDGES,
    RESULT_TYPE_TABLES,
    SCHEMA_CATALOG,
    TableInfo,
)
from drive_qa.semantic_parser import SemanticParse
from drive_qa.text_normalization import normalize_text, tokenize
from drive_qa.vocabulary import RE_GENE_REFERENCE, RE_NETWORK_PROXIMITY_IMPLICIT


# =========================================================
# Base scoring
# =========================================================

def score_table(question: str, table: TableInfo) -> Tuple[int, List[str]]:
    q_norm = normalize_text(question)
    tokens = set(tokenize(question))

    score = 0
    reasons = []

    table_name_tokens = set(table.name.split("_"))
    overlap_table_name = tokens.intersection(table_name_tokens)
    if overlap_table_name:
        score += 4 * len(overlap_table_name)
        reasons.append(f"Coincidencia con nombre de tabla: {sorted(overlap_table_name)}")

    matched_columns = []
    for col in table.columns:
        col_tokens = set(normalize_text(col).split("_"))
        overlap_col = tokens.intersection(col_tokens)
        if overlap_col:
            score += 2 * len(overlap_col)
            matched_columns.extend(list(overlap_col))
    if matched_columns:
        reasons.append(f"Coincidencia con columnas: {sorted(set(matched_columns))}")

    matched_synonyms = []
    for synonym in table.synonyms:
        if normalize_text(synonym) in q_norm:
            score += 5
            matched_synonyms.append(synonym)
    if matched_synonyms:
        reasons.append(f"Coincidencia con sinónimos: {matched_synonyms}")

    desc_tokens = set(tokenize(table.description))
    overlap_desc = tokens.intersection(desc_tokens)
    if overlap_desc:
        score += len(overlap_desc)
        reasons.append(f"Coincidencia con descripción: {sorted(overlap_desc)}")

    return score, reasons


def init_scores(catalog: Dict[str, TableInfo]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    scores = {table_name: 0 for table_name in catalog}
    reasons = {table_name: [] for table_name in catalog}
    return scores, reasons


def add_boost(
    table_name: str,
    boost: int,
    reason: str,
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    if table_name in scores and boost != 0:
        scores[table_name] += boost
        reasons_map[table_name].append(reason)


# =========================================================
# Lexical scores
# =========================================================

def apply_lexical_scores(
    question: str,
    catalog: Dict[str, TableInfo],
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    for table_name, table_info in catalog.items():
        s, reasons = score_table(question, table_info)
        scores[table_name] += s
        reasons_map[table_name].extend(reasons)


# =========================================================
# Entity boosts
# =========================================================

def apply_db_entity_boosts(
    detected_entities: Dict[str, List[Dict[str, Any]]],
    entity_strategies: Dict[str, Dict[str, Any]],
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    for entity_type, base_tables in ENTITY_BASE_TABLES.items():
        candidates = detected_entities.get(entity_type, [])
        strategy_info = entity_strategies.get(entity_type, {"strategy": "no_entity", "selected": None})

        if not candidates:
            continue

        strategy = strategy_info["strategy"]
        selected = strategy_info.get("selected")

        if strategy == "pattern_search":
            boost = 2
            reason = f"Boost suave por pattern_search en {entity_type}"
        elif strategy == "exact_match" and selected:
            boost = 8
            reason = f"Boost por exact_match en {entity_type}: {selected['name']} ({selected['match_score']})"
        elif strategy == "best_candidate" and selected:
            boost = 5
            reason = f"Boost por best_candidate en {entity_type}: {selected['name']} ({selected['match_score']})"
        elif strategy == "ambiguous":
            boost = 2
            reason = f"Boost suave por ambigüedad en {entity_type}"
        else:
            continue

        for table_name in base_tables:
            add_boost(table_name, boost, reason, scores, reasons_map)


# =========================================================
# Result type boosts
# =========================================================

def apply_result_type_boosts(
    result_types_requested: List[str],
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    for result_type in result_types_requested:
        for table_name in RESULT_TYPE_TABLES.get(result_type, []):
            add_boost(table_name, 4, f"Boost por tipo de resultado '{result_type}'", scores, reasons_map)


# =========================================================
# Intent boosts
# =========================================================

def apply_intent_boosts(
    intent_scores: Dict[str, int],
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    neutral_intent_tables = {
        "ranking": ["disease", "drug"],
        "count": ["disease", "drug"],
        "cross_method_aggregation": ["disease", "drug"],
        "protein_encoding": ["encodes", "gene"],
        "information_paths": ["disease", "drug", "gene"],
        "pathways_method": ["disease", "drug", "pathway", "gene"],
        "disease_pathways": ["disease", "drug", "pathway"],
        "threshold_values": ["disease", "drug", "gene"],
        "gnn_score": ["disease", "drug"],
        "network_proximity": ["disease", "drug"],
    }

    for intent, score in intent_scores.items():
        boost = min(6, 1 + score)
        for table_name in neutral_intent_tables.get(intent, []):
            add_boost(table_name, boost, f"Boost neutro por intención '{intent}' (+{boost})", scores, reasons_map)


# =========================================================
# Primary table intent boost
# =========================================================

def apply_primary_table_intent_boost(
    parse: SemanticParse,
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
    entity_strategies: Dict[str, Dict[str, Any]],
) -> None:
    primary_table = None

    if "gnn_score" in parse.intents:
        asks_for_genes = "genes" in (parse.result_types_requested or [])
        asks_for_pathways = "pathways" in (parse.result_types_requested or [])
        gene_confident = is_confident_entity_strategy(entity_strategies.get("genes", {}))
        pathway_confident = is_confident_entity_strategy(entity_strategies.get("pathways", {}))

        if not asks_for_genes and not gene_confident and "gene" in scores:
            scores["gene"] -= 2
            reasons_map["gene"].append(
                "Penalización ligera: query GNN sin evidencia fuerte de genes (-2)"
            )
        if not asks_for_pathways and not pathway_confident and "pathway" in scores:
            scores["pathway"] -= 2
            reasons_map["pathway"].append(
                "Penalización ligera: query GNN sin evidencia fuerte de pathways (-2)"
            )

    if "threshold_values" in parse.intents:
        primary_table = "dr_threshold_values"
    elif "network_proximity" in parse.intents:
        primary_table = "dr_network_proximity"
    elif "gnn_score" in parse.intents:
        primary_table = "dr_gnns"
    elif "information_paths" in parse.intents:
        primary_table = "dr_information_paths"
    elif "disease_pathways" in parse.intents:
        primary_table = "dr_diseasepathways"
    elif "pathways_method" in parse.intents:
        q_norm = normalize_text(parse.question)
        if "dr_pathways_count" in q_norm:
            primary_table = "dr_pathways_count"
        elif parse.metric in {"count", "association_type"}:
            primary_table = "dr_pathways_count"
        else:
            primary_table = "dr_pathways"

    if primary_table:
        add_boost(
            primary_table,
            5,
            f"Boost semántico suave por intent principal '{primary_table}'",
            scores,
            reasons_map,
        )

    if "network_proximity" in parse.intents:
        q_norm = normalize_text(" ".join(parse.candidate_spans)) if parse.candidate_spans else ""
        if RE_NETWORK_PROXIMITY_IMPLICIT.search(q_norm):
            if "dr_network_proximity" in scores:
                scores["dr_network_proximity"] += 2
                reasons_map["dr_network_proximity"].append(
                    "Boost extra por lenguaje implícito de network proximity (+2)"
                )

    if "threshold_values" in parse.intents:
        q_norm = normalize_text(" ".join(parse.candidate_spans)) if parse.candidate_spans else ""
        asks_for_genes = "genes" in (parse.result_types_requested or [])
        gene_info = entity_strategies.get("genes", {})
        gene_confident = is_confident_entity_strategy(gene_info)

        if asks_for_genes or gene_confident or RE_GENE_REFERENCE.search(q_norm):
            if "dr_threshold_values" in scores:
                scores["dr_threshold_values"] += 3
                reasons_map["dr_threshold_values"].append(
                    "Boost explícito por lenguaje de genes en threshold_values (+3)"
                )

        if not asks_for_genes and not gene_confident:
            if "gene" in scores:
                scores["gene"] -= 2
                reasons_map["gene"].append(
                    "Penalización ligera: query threshold_values sin evidencia fuerte de genes (-2)"
                )
            if "encodes" in scores:
                scores["encodes"] -= 2
                reasons_map["encodes"].append(
                    "Penalización ligera: query threshold_values sin evidencia fuerte de genes (-2)"
                )

    if "disease_pathways" in parse.intents:
        if "dr_pathways" in scores:
            scores["dr_pathways"] -= 3
            reasons_map["dr_pathways"].append(
                "Penalización: query disease_pathways, dr_pathways no es relevante (-3)"
            )
        if "dr_pathways_count" in scores:
            scores["dr_pathways_count"] -= 3
            reasons_map["dr_pathways_count"].append(
                "Penalización: query disease_pathways, dr_pathways_count no es relevante (-3)"
            )

    if "information_paths" in parse.intents:
        for rival in ("dr_pathways", "dr_pathways_count", "dr_diseasepathways"):
            if rival in scores:
                scores[rival] -= 3
                reasons_map[rival].append(
                    "Penalización: query information_paths, tabla pathway no relevante (-3)"
                )
        if "pathway" in scores:
            scores["pathway"] -= 2
            reasons_map["pathway"].append(
                "Penalización ligera: query information_paths sin evidencia de pathways (-2)"
            )

    if "pathways_method" in parse.intents and "disease_pathways" not in parse.intents:
        if "dr_diseasepathways" in scores:
            scores["dr_diseasepathways"] -= 3
            reasons_map["dr_diseasepathways"].append(
                "Penalización: query pathways_method, dr_diseasepathways es un método diferente (-3)"
            )


# =========================================================
# Metric model boosts
# =========================================================

def apply_metric_model_boosts(
    parse: SemanticParse,
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    gnn_metrics = {"redirection", "dmsr", "dmsrc", "behor", "behorc"}
    network_metrics = {"proximity", "closest_distance", "dc_mean", "dc_std"}
    threshold_metrics = {"score", "action_type"}

    if parse.metric in gnn_metrics:
        metric_boost = 10
        if parse.metric in {"dmsrc", "behorc"}:
            metric_boost = 12
        add_boost("dr_gnns", metric_boost, f"Boost explícito por métrica '{parse.metric}'", scores, reasons_map)
        if "drug" in scores:
            scores["drug"] -= 2
            reasons_map["drug"].append(f"Penalización ligera por métrica metodológica explícita '{parse.metric}' (-2)")
        if "disease" in scores:
            scores["disease"] -= 2
            reasons_map["disease"].append(f"Penalización ligera por métrica metodológica explícita '{parse.metric}' (-2)")

    elif parse.metric in network_metrics:
        metric_boost = 10
        if parse.metric in {"dc_mean", "dc_std"}:
            metric_boost = 14
        elif parse.metric == "closest_distance":
            metric_boost = 12
        add_boost("dr_network_proximity", metric_boost, f"Boost explícito por métrica '{parse.metric}'", scores, reasons_map)
        if "drug" in scores:
            scores["drug"] -= 2
            reasons_map["drug"].append(f"Penalización ligera por métrica metodológica explícita '{parse.metric}' (-2)")
        if "disease" in scores:
            scores["disease"] -= 2
            reasons_map["disease"].append(f"Penalización ligera por métrica metodológica explícita '{parse.metric}' (-2)")

    elif parse.metric in threshold_metrics:
        metric_boost = 12 if parse.metric == "score" else 14
        add_boost("dr_threshold_values", metric_boost, f"Boost explícito por métrica '{parse.metric}'", scores, reasons_map)
        for base in ("drug", "disease", "gene"):
            if base in scores:
                scores[base] -= 2
                reasons_map[base].append(f"Penalización ligera por métrica metodológica explícita '{parse.metric}' (-2)")

    elif parse.metric == "path_id":
        add_boost("dr_information_paths", 12, f"Boost explícito por métrica '{parse.metric}'", scores, reasons_map)
        for rival in ("dr_pathways", "dr_pathways_count", "dr_diseasepathways"):
            if rival in scores:
                scores[rival] -= 4
                reasons_map[rival].append(f"Penalización por métrica 'path_id' exclusiva de information_paths (-4)")

    elif parse.metric == "approach":
        add_boost("dr_pathways", 10, f"Boost explícito por métrica '{parse.metric}'", scores, reasons_map)
        add_boost("dr_pathways_count", 6, f"Boost secundario por métrica '{parse.metric}'", scores, reasons_map)
        if "dr_diseasepathways" in scores:
            scores["dr_diseasepathways"] -= 4
            reasons_map["dr_diseasepathways"].append(f"Penalización: 'approach' no pertenece a disease_pathways (-4)")

    elif parse.metric == "association_type":
        add_boost("dr_pathways_count", 12, f"Boost explícito por métrica '{parse.metric}'", scores, reasons_map)
        if "dr_pathways" in scores:
            scores["dr_pathways"] -= 3
            reasons_map["dr_pathways"].append(f"Penalización: 'association_type' es exclusivo de dr_pathways_count (-3)")
        if "dr_diseasepathways" in scores:
            scores["dr_diseasepathways"] -= 4
            reasons_map["dr_diseasepathways"].append(f"Penalización: 'association_type' no pertenece a disease_pathways (-4)")

    elif parse.metric == "count":
        add_boost("dr_pathways_count", 10, f"Boost explícito por métrica '{parse.metric}'", scores, reasons_map)
        if "dr_diseasepathways" in scores:
            scores["dr_diseasepathways"] -= 4
            reasons_map["dr_diseasepathways"].append(f"Penalización: 'count' no pertenece a disease_pathways (-4)")


# =========================================================
# Metric tiebreak
# =========================================================

def apply_metric_tiebreak(
    parse: SemanticParse,
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    metric_to_primary = {
        "redirection": "dr_gnns",
        "dmsr": "dr_gnns",
        "dmsrc": "dr_gnns",
        "behor": "dr_gnns",
        "behorc": "dr_gnns",
        "proximity": "dr_network_proximity",
        "closest_distance": "dr_network_proximity",
        "dc_mean": "dr_network_proximity",
        "dc_std": "dr_network_proximity",
        "score": "dr_threshold_values",
        "action_type": "dr_threshold_values",
        "path_id": "dr_information_paths",
        "approach": "dr_pathways",
        "association_type": "dr_pathways_count",
        "count": "dr_pathways_count",
    }
    if parse.metric == "score" and "pathways_method" in parse.intents and "threshold_values" not in parse.intents:
        primary_table = "dr_pathways"
    else:
        primary_table = metric_to_primary.get(parse.metric)
    if not primary_table:
        return

    base_tables = ["drug", "disease"]
    max_base = max(scores.get(t, -10**9) for t in base_tables)
    primary_score = scores.get(primary_table, 0)
    if primary_score <= max_base:
        delta = (max_base - primary_score) + 1
        scores[primary_table] += delta
        reasons_map[primary_table].append(
            f"Desempate a favor de tabla metodológica por métrica explícita '{parse.metric}' (+{delta})"
        )


# =========================================================
# Metric gating
# =========================================================

def apply_metric_gating(
    parse: SemanticParse,
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    metric_to_primary = {
        "redirection": "dr_gnns",
        "dmsr": "dr_gnns",
        "dmsrc": "dr_gnns",
        "behor": "dr_gnns",
        "behorc": "dr_gnns",
        "proximity": "dr_network_proximity",
        "closest_distance": "dr_network_proximity",
        "dc_mean": "dr_network_proximity",
        "dc_std": "dr_network_proximity",
        "score": "dr_threshold_values",
        "action_type": "dr_threshold_values",
        "path_id": "dr_information_paths",
        "approach": "dr_pathways",
        "association_type": "dr_pathways_count",
        "count": "dr_pathways_count",
    }

    if parse.metric == "score" and "pathways_method" in parse.intents and "threshold_values" not in parse.intents:
        primary_table = "dr_pathways"
    else:
        primary_table = metric_to_primary.get(parse.metric)
    if not primary_table:
        return

    metric_tables = {
        "dr_gnns",
        "dr_network_proximity",
        "dr_information_paths",
        "dr_threshold_values",
        "dr_pathways",
        "dr_diseasepathways",
        "dr_pathways_count",
    }

    for table_name in metric_tables:
        if table_name == primary_table:
            continue
        if primary_table == "dr_pathways" and table_name == "dr_pathways_count":
            continue
        if primary_table == "dr_pathways_count" and table_name == "dr_pathways":
            continue
        if table_name in scores:
            scores[table_name] -= 4
            reasons_map[table_name].append(
                f"Gating suave por métrica explícita '{parse.metric}' (-4)"
            )


# =========================================================
# Result bridge boosts
# =========================================================

def apply_result_bridge_boosts(
    result_types_requested: List[str],
    predicted_entity_types: List[str],
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    neutral_allowed = {"disease", "drug", "gene", "pathway", "encodes"}

    for result_type in result_types_requested:
        for entity_type in predicted_entity_types:
            for table_name in RESULT_INTENT_BRIDGES.get((result_type, entity_type), []):
                if table_name not in neutral_allowed:
                    continue
                add_boost(
                    table_name,
                    3,
                    f"Boost estructural neutro ({result_type} ← {entity_type})",
                    scores,
                    reasons_map,
                )


# =========================================================
# Template boosts
# =========================================================

def apply_template_boosts(
    parse: SemanticParse,
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    template_tables = {
        "method_ranking_by_disease": ["disease", "drug"],
        "method_ranking_by_drug": ["disease", "drug"],
        "cross_method_overlap": [
            "disease", "drug",
            "dr_gnns", "dr_network_proximity", "dr_information_paths",
            "dr_threshold_values", "dr_pathways", "dr_diseasepathways"
        ],
        "aggregation_count": ["disease", "drug"],
        "association_lookup": ["disease", "drug"],
        "entity_pattern_search": [],
        "generic_lookup": [],
    }

    chosen = list(template_tables.get(parse.query_template, []))

    for table_name in dict.fromkeys(chosen):
        add_boost(table_name, 3, f"Boost por plantilla '{parse.query_template}'", scores, reasons_map)

    # For cross_method_overlap, ensure dr_* method tables rank above base tables
    if parse.query_template == "cross_method_overlap":
        dr_method_tables = [
            "dr_gnns", "dr_network_proximity", "dr_information_paths",
            "dr_threshold_values", "dr_pathways", "dr_diseasepathways",
        ]
        for table_name in dr_method_tables:
            add_boost(table_name, 18, "Boost cross-method: method table priority", scores, reasons_map)
        # Demote base tables so method tables rank first
        for base_table in ["disease", "drug", "gene", "pathway", "encodes"]:
            add_boost(base_table, -6, "Penalización cross-method: base table demotion", scores, reasons_map)


# =========================================================
# Method primary tiebreak
# =========================================================

def apply_method_primary_tiebreak(
    parse: SemanticParse,
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    primary_table = None
    base_tables = ["disease", "drug", "gene", "pathway"]

    if "threshold_values" in parse.intents:
        primary_table = "dr_threshold_values"
    elif "network_proximity" in parse.intents:
        primary_table = "dr_network_proximity"
    elif "gnn_score" in parse.intents:
        primary_table = "dr_gnns"
    elif "information_paths" in parse.intents:
        primary_table = "dr_information_paths"
    elif "disease_pathways" in parse.intents:
        primary_table = "dr_diseasepathways"
    elif "pathways_method" in parse.intents:
        q_norm = normalize_text(parse.question)
        if "dr_pathways_count" in q_norm:
            primary_table = "dr_pathways_count"
        elif parse.metric in {"count", "association_type"}:
            primary_table = "dr_pathways_count"
        else:
            primary_table = "dr_pathways"

    if not primary_table or primary_table not in scores:
        return

    primary_score = scores[primary_table]
    max_base_score = max((scores.get(b, -999) for b in base_tables), default=-999)
    sibling_dr_tables: List[str] = []
    if primary_table == "dr_pathways":
        sibling_dr_tables = ["dr_pathways_count"]
    elif primary_table == "dr_pathways_count":
        sibling_dr_tables = ["dr_pathways"]
    max_competitor = max(max_base_score, max((scores.get(s, -999) for s in sibling_dr_tables), default=-999))

    if max_competitor >= primary_score - 3:
        delta = max(4, max_competitor - primary_score + 4)
        scores[primary_table] += delta
        reasons_map[primary_table].append(
            f"Desempate: tabla metodológica debe superar competidores (+{delta})"
        )


# =========================================================
# Cross-model exclusion gating
# =========================================================

def apply_cross_model_exclusion_gating(
    parse: SemanticParse,
    scores: Dict[str, int],
    reasons_map: Dict[str, List[str]],
) -> None:
    if parse.is_cross_method:
        return

    model_to_penalize = None
    strength = -3

    if "gnn_score" in parse.intents:
        model_to_penalize = "gnn_score"
        strength = -8 if parse.intents.get("gnn_score", 0) >= 4 else -3
    elif "threshold_values" in parse.intents:
        model_to_penalize = "threshold_values"
        strength = -8 if parse.intents.get("threshold_values", 0) >= 4 else -3
    elif "network_proximity" in parse.intents:
        model_to_penalize = "network_proximity"
        strength = -8 if parse.intents.get("network_proximity", 0) >= 4 else -3
    elif "information_paths" in parse.intents:
        model_to_penalize = "information_paths"
        strength = -8 if parse.intents.get("information_paths", 0) >= 4 else -3
    elif "disease_pathways" in parse.intents:
        model_to_penalize = "disease_pathways"
        strength = -8 if parse.intents.get("disease_pathways", 0) >= 4 else -3
    elif "pathways_method" in parse.intents:
        model_to_penalize = "pathways_method"
        strength = -6 if parse.intents.get("pathways_method", 0) >= 4 else -3

    _gating_targets = {
        "gnn_score": ("dr_threshold_values", "dr_network_proximity", "dr_information_paths", "dr_diseasepathways", "dr_pathways", "dr_pathways_count"),
        "threshold_values": ("dr_gnns", "dr_network_proximity", "dr_information_paths", "dr_diseasepathways", "dr_pathways", "dr_pathways_count"),
        "network_proximity": ("dr_gnns", "dr_threshold_values", "dr_information_paths", "dr_diseasepathways", "dr_pathways", "dr_pathways_count"),
        "information_paths": ("dr_gnns", "dr_threshold_values", "dr_network_proximity", "dr_diseasepathways", "dr_pathways", "dr_pathways_count"),
        "disease_pathways": ("dr_gnns", "dr_threshold_values", "dr_network_proximity", "dr_information_paths", "dr_pathways", "dr_pathways_count"),
        "pathways_method": ("dr_gnns", "dr_threshold_values", "dr_network_proximity", "dr_information_paths", "dr_diseasepathways"),
    }

    tables = _gating_targets.get(model_to_penalize, ())

    for table in tables:
        if table in scores:
            scores[table] += strength
            reasons_map[table].append(
                f"Gating cruzado híbrido: consulta {model_to_penalize}, penalización {strength}"
            )


# =========================================================
# Table enforcement and expansion
# =========================================================

def enforce_required_tables(
    parse: SemanticParse,
    entity_strategies: Dict[str, Dict[str, Any]],
    ranked_tables: List[str],
) -> List[str]:
    required = set()

    if parse.query_template in {"method_ranking_by_disease", "method_ranking_by_drug"}:
        required.update({"drug", "disease"})
        if parse.metric in {"redirection", "dmsr", "dmsrc", "behor", "behorc"} or "gnn_score" in parse.intents:
            required.add("dr_gnns")
        if parse.metric in {"proximity", "closest_distance", "dc_mean", "dc_std"} or "network_proximity" in parse.intents:
            required.add("dr_network_proximity")
        if parse.metric in {"score", "action_type"} or "threshold_values" in parse.intents:
            if parse.metric == "score" and "pathways_method" in parse.intents and "threshold_values" not in parse.intents:
                required.add("dr_pathways")
            else:
                required.add("dr_threshold_values")
            if "genes" in parse.result_types_requested or "gene" in parse.candidate_spans:
                required.add("gene")
        if parse.metric == "path_id" or "information_paths" in parse.intents:
            required.add("dr_information_paths")
        if parse.metric in {"approach", "association_type", "count"} or "pathways_method" in parse.intents:
            q_norm = normalize_text(parse.question)
            if "dr_pathways_count" in q_norm:
                required.add("dr_pathways_count")
            elif parse.metric in {"count", "association_type"}:
                required.add("dr_pathways_count")
            else:
                required.add("dr_pathways")
        if "disease_pathways" in parse.intents:
            required.add("dr_diseasepathways")

    if parse.query_template == "association_lookup" and "gnn_score" in parse.intents:
        required.update({"dr_gnns", "disease", "drug"})
    if parse.query_template == "association_lookup" and "network_proximity" in parse.intents:
        required.update({"dr_network_proximity", "disease", "drug"})
    if parse.query_template == "association_lookup" and "threshold_values" in parse.intents:
        required.update({"dr_threshold_values", "disease", "drug"})
    if parse.query_template == "association_lookup" and "information_paths" in parse.intents:
        required.update({"dr_information_paths", "disease", "drug"})
    if parse.query_template == "association_lookup" and "disease_pathways" in parse.intents:
        required.update({"dr_diseasepathways", "disease", "drug"})
    if parse.query_template == "association_lookup" and "pathways_method" in parse.intents:
        required.update({"dr_pathways", "disease", "drug"})

    if parse.query_template == "cross_method_overlap":
        required.update({
            "disease",
            "dr_gnns", "dr_network_proximity", "dr_information_paths",
            "dr_threshold_values", "dr_pathways", "dr_diseasepathways"
        })
        if "drugs" in parse.result_types_requested or entity_strategies.get("drugs", {}).get("selected") is not None:
            required.add("drug")

    for etype, info in entity_strategies.items():
        if info.get("selected") is not None:
            if etype == "diseases":
                required.add("disease")
            elif etype == "drugs":
                required.add("drug")
            elif etype == "genes" and parse.query_template != "cross_method_overlap":
                required.add("gene")
            elif etype == "pathways" and parse.query_template != "cross_method_overlap":
                required.add("pathway")

    return list(dict.fromkeys(list(required) + ranked_tables))


def expand_related_tables_semantic(
    selected_tables: List[str],
    catalog: Dict[str, TableInfo],
    scores: Dict[str, int],
) -> List[str]:
    expanded = set(selected_tables)

    def maybe_add(table_name: str) -> None:
        if table_name in catalog:
            expanded.add(table_name)

    if "dr_gnns" in expanded:
        maybe_add("disease")
        maybe_add("drug")
    if "dr_network_proximity" in expanded:
        maybe_add("disease")
        maybe_add("drug")
    if "dr_information_paths" in expanded:
        maybe_add("disease")
        maybe_add("drug")
        if "gene" in selected_tables:
            maybe_add("gene")
    if "dr_threshold_values" in expanded:
        maybe_add("disease")
        maybe_add("drug")
        if "gene" in selected_tables:
            maybe_add("gene")
    if "dr_pathways" in expanded or "dr_diseasepathways" in expanded:
        maybe_add("disease")
        maybe_add("drug")
        if "pathway" in selected_tables:
            maybe_add("pathway")
    if "dr_pathways_count" in expanded:
        maybe_add("disease")
        maybe_add("drug")

    for table_name in list(expanded):
        if table_name not in catalog:
            continue

        related = catalog[table_name].related_tables
        for rel in related:
            if rel in catalog and scores.get(rel, 0) >= 4:
                if rel in {"gene", "pathway"} and rel not in selected_tables:
                    continue
                # Skip tables that received gating penalties (negative score)
                if scores.get(rel, 0) < 0:
                    continue
                expanded.add(rel)

    return list(expanded)
