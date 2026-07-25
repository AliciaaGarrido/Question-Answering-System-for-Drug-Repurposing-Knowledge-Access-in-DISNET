"""
Prompt builder — constructs prompts for Gemini SQL generation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from drive_qa.logging_config import get_logger
from drive_qa.schema_catalog import SCHEMA_CATALOG

logger = get_logger("prompt_builder")

DEFAULT_DB_SCHEMA = "dr"

# Join constraints for the DRIVE schema
JOIN_CONSTRAINTS = {
    ("dr_gnns", "disease"): "dr_gnns.disease_id = disease.disease_id",
    ("dr_gnns", "drug"): "dr_gnns.drug_id = drug.drug_id",
    ("dr_network_proximity", "disease"): "dr_network_proximity.disease_id = disease.disease_id",
    ("dr_network_proximity", "drug"): "dr_network_proximity.drug_id = drug.drug_id",
    ("dr_information_paths", "disease"): "dr_information_paths.disease_id = disease.disease_id",
    ("dr_information_paths", "drug"): "dr_information_paths.drug_id = drug.drug_id",
    ("dr_diseasepathways", "disease"): "dr_diseasepathways.disease_id_original = disease.disease_id (use when disease is the SOURCE of existing drugs) OR dr_diseasepathways.disease_id_new = disease.disease_id (use when disease is the TARGET for repurposing)",
    ("dr_diseasepathways", "drug"): "dr_diseasepathways.drug_id = drug.drug_id",
    ("dr_diseasepathways", "pathway"): "dr_diseasepathways.pathway_id = pathway.pathway_id",
    ("dr_threshold_values", "disease"): "dr_threshold_values.disease_id = disease.disease_id",
    ("dr_threshold_values", "drug"): "dr_threshold_values.drug_id = drug.drug_id",
    ("dr_threshold_values", "gene"): "dr_threshold_values.gene_id = gene.gene_id",
    ("dr_pathways", "disease"): "dr_pathways.disease_id = disease.disease_id",
    ("dr_pathways", "drug"): "dr_pathways.drug_id = drug.drug_id",
    ("dr_pathways", "gene"): "dr_pathways.gene_id = gene.gene_id",
    ("dr_pathways", "pathway"): "dr_pathways.pathway_id = pathway.pathway_id",
    ("dr_pathways_count", "disease"): "dr_pathways_count.disease_id = disease.disease_id",
    ("dr_pathways_count", "drug"): "dr_pathways_count.drug_id = drug.drug_id",
    ("encodes", "gene"): "encodes.gene_id = gene.gene_id",
}


def _qualify_table(name: str, db_schema: str) -> str:
    """Qualify a table name with the database schema prefix."""
    return f"{db_schema}.{name}"


def _qualify_condition(condition: str, tables: Set[str], db_schema: str) -> str:
    """Add database prefix to table references in a join condition string."""
    result = condition
    for table in sorted(tables, key=len, reverse=True):
        result = re.sub(
            rf'\b{re.escape(table)}\.',
            f'{db_schema}.{table}.',
            result,
        )
    return result


def build_join_context(selected_tables: List[str], db_schema: str = DEFAULT_DB_SCHEMA) -> str:
    """Build join constraints text for the selected tables."""
    lines = []
    table_set = set(selected_tables)
    all_table_names: Set[str] = set()
    for (t1, t2) in JOIN_CONSTRAINTS:
        all_table_names.add(t1)
        all_table_names.add(t2)

    for (t1, t2), condition in JOIN_CONSTRAINTS.items():
        if t1 in table_set and t2 in table_set:
            qualified_condition = _qualify_condition(condition, all_table_names, db_schema)
            lines.append(f"  {db_schema}.{t1} ↔ {db_schema}.{t2}: {qualified_condition}")

    if not lines:
        return "No direct joins needed (single table query)."
    return "\n".join(lines)


def qualify_schema_context(schema_context: str, db_schema: str) -> str:
    """Add database prefix to table names in the schema context block."""
    catalog_tables = list(SCHEMA_CATALOG.keys())
    result = schema_context
    for table in sorted(catalog_tables, key=len, reverse=True):
        result = result.replace(f"TABLE {table}", f"TABLE {db_schema}.{table}")
    for table in sorted(catalog_tables, key=len, reverse=True):
        result = re.sub(
            rf'(?<![.\w])\b{re.escape(table)}\b(?!\.)',
            f'{db_schema}.{table}',
            result,
        )
    return result


def build_prompt(
    question: str,
    selected_tables: List[str],
    schema_context: str,
    entity_context: str,
    join_context: str,
    semantic_parse: Dict[str, Any],
) -> str:
    """Construct the full user prompt for Gemini SQL generation."""
    parts = []

    parts.append(f"QUESTION: {question}")
    parts.append("")

    # Schema
    parts.append("AVAILABLE SCHEMA:")
    parts.append(schema_context)

    # Joins
    parts.append("JOIN CONDITIONS:")
    parts.append(join_context)
    parts.append("")

    # Detected entities (grounding)
    if entity_context.strip():
        parts.append("DETECTED ENTITIES (use these exact IDs/names for filtering):")
        parts.append(entity_context)

    # Semantic hints
    if semantic_parse.get("metric"):
        parts.append(f"METRIC TO USE: {semantic_parse['metric']}")
    if semantic_parse.get("operator"):
        parts.append(f"OPERATION: {semantic_parse['operator']}")
    if semantic_parse.get("result_types_requested"):
        parts.append(f"EXPECTED RESULT TYPE: {', '.join(semantic_parse['result_types_requested'])}")

    operation_guidance = build_operation_guidance(question, semantic_parse)
    if operation_guidance:
        parts.append("")
        parts.append("SQL OPERATION GUIDANCE:")
        parts.append(operation_guidance)

    # Directional rule for dr_diseasepathways
    if "dr_diseasepathways" in selected_tables:
        parts.append("")
        parts.append("CRITICAL RULE FOR dr_diseasepathways:")
        parts.append("  - disease_id_original = source disease (has existing approved drugs)")
        parts.append("  - disease_id_new = target disease (candidate for drug repurposing)")
        parts.append("  - To find drugs repurposable FOR disease X: WHERE disease_id_new = X")
        parts.append("  - To find diseases that could RECEIVE drugs from disease Y: WHERE disease_id_original = Y")
        parts.append("  - NEVER confuse source and target: 'repurposed for X' means disease_id_new = X")

    parts.append("")
    parts.append("Generate the SQL query:")

    prompt = "\n".join(parts)
    logger.debug("Prompt built (length=%d chars, tables=%s)", len(prompt), selected_tables)
    return prompt


def build_operation_guidance(question: str, semantic_parse: Dict[str, Any]) -> str:
    """Build precise SQL rules for aggregate/ranking operators."""
    operator = semantic_parse.get("operator")
    metric = semantic_parse.get("metric")
    query_template = semantic_parse.get("query_template", "")
    is_cross_method = semantic_parse.get("is_cross_method", False)

    q_norm = question.lower()
    parts = []

    # Max/min with explicit metric
    if operator in {"max", "min"} and metric:
        has_explicit_top_n = bool(
            re.search(r"\b(top\s+\d+|\d+\s+(?:primer[oa]s?|mejores|mayores|menores|top|drugs?|f[aá]rmacos?|medicamentos?))\b", q_norm)
        )

        if has_explicit_top_n:
            direction = "DESC" if operator == "max" else "ASC"
            parts.append(
                f"The question asks for an explicit top-N/ranking. Order by the metric `{metric}` "
                f"{direction} and use LIMIT N if N is stated."
            )
        else:
            aggregate = "MAX" if operator == "max" else "MIN"
            parts.append(
                f"The question asks for the {'highest' if operator == 'max' else 'lowest'} `{metric}` value, "
                f"not merely a sorted list. Compute the {aggregate}({metric}) over the same filters from "
                f"the question and return only rows whose `{metric}` equals that computed value. Use a "
                "subquery or CTE and preserve ties. Do not require the user to know the numeric value."
            )
        return "\n".join(parts)

    # Count operator
    if operator == "count":
        parts.append("Use COUNT(DISTINCT ...) to avoid duplicates introduced by JOINs.")
        if is_cross_method:
            parts.append(
                "Count how many distinct repurposing methods contain the drug-disease pair. "
                "Use UNION ALL across relevant dr_* tables with a method label column, "
                "then COUNT DISTINCT methods."
            )
        return "\n".join(parts)

    # Sort operator
    if operator == "sort":
        if metric:
            parts.append(
                f"Order results by `{metric}` DESC. Use LIMIT only if the question specifies an explicit number."
            )
        else:
            parts.append(
                "Order results by the relevant score/metric column DESC. "
                "Use LIMIT only if the question specifies an explicit number."
            )
        return "\n".join(parts)

    # Cross-method without explicit operator
    if is_cross_method:
        parts.append(
            "This is a cross-method query. Check presence of the drug-disease pair across "
            "multiple evidence tables (dr_gnns, dr_network_proximity, dr_information_paths, "
            "dr_threshold_values, dr_pathways, dr_diseasepathways). "
            "Use UNION ALL with a method label column, then aggregate as needed."
        )
        return "\n".join(parts)

    # Pattern search
    if query_template == "entity_pattern_search":
        parts.append("Use LIKE '%pattern%' for name-based filtering. Match is case-insensitive (use LOWER).")
        return "\n".join(parts)

    return ""


def build_answer_prompt(
    question: str,
    sql: str,
    columns: List[str],
    rows: List[Dict[str, Any]],
    row_count: int,
    truncated: bool,
    max_rows: int,
    nl_row_limit: int = 20,
) -> str:
    """Build the prompt for SQL→NL verbalization."""
    import json

    rows_for_answer = rows[:nl_row_limit]
    answer_truncated = len(rows) > nl_row_limit

    parts = []
    parts.append(f"QUESTION: {question}")
    parts.append(f"SQL EXECUTED: {sql}")
    parts.append(f"COLUMNS: {', '.join(columns)}")
    parts.append(f"TOTAL ROWS: {row_count}")
    if truncated:
        parts.append(f"(Showing first {max_rows} of {row_count} rows)")
    if answer_truncated:
        parts.append(
            f"(Natural-language answer receives only the first {nl_row_limit} rows; "
            f"do not enumerate all {row_count} rows.)"
        )
    parts.append("")
    parts.append("RESULTS:")
    parts.append(json.dumps(rows_for_answer, ensure_ascii=False, default=str))
    parts.append("")
    if answer_truncated:
        parts.append(
            "Please answer the question based on the data above. Summarize the result "
            "and explicitly mention that only the first rows are shown in the natural-language answer."
        )
    else:
        parts.append("Please answer the question based on the data above.")

    return "\n".join(parts)
