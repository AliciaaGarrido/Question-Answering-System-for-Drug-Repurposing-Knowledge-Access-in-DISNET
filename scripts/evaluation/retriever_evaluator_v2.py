from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    if isinstance(data, list):
        return data
    raise ValueError("El dataset debe ser una lista o un objeto con clave 'cases'.")


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def get_predicted_entity_types(data: Dict[str, Any]) -> List[Any]:
    """Read the renamed semantic field, accepting legacy reports/banks."""
    if "predicted_entity_types" in data:
        return safe_list(data.get("predicted_entity_types"))
    return safe_list(data.get("entity_types_to_search"))


def import_retriever_module(module_name: str):
    """
    Importa dinámicamente el módulo del retriever.
    Debe exponer:
      - SCHEMA_CATALOG
      - retrieve_schema(...)
    """
    import importlib

    mod = importlib.import_module(module_name)
    if not hasattr(mod, "SCHEMA_CATALOG"):
        raise AttributeError(f"El módulo '{module_name}' no expone SCHEMA_CATALOG.")
    if not hasattr(mod, "retrieve_schema"):
        raise AttributeError(f"El módulo '{module_name}' no expone retrieve_schema(...).")
    return mod


def infer_primary_table(required_tables: Set[str], gold: Dict[str, Any] | None = None) -> str | None:
    """
    Intenta identificar la tabla principal esperada dentro de required_tables.
    Si el gold incluye 'primary_table' explícito, se usa directamente.
    Para cross_method_aggregation (múltiples dr_*), se devuelve None
    y top1_ok se evalúa de forma especial.
    """
    if gold and gold.get("primary_table"):
        return gold["primary_table"]

    dr_tables = sorted(t for t in required_tables if t.startswith("dr_"))
    # Si hay exactamente 1 dr_* table, es la primary
    if len(dr_tables) == 1:
        return dr_tables[0]
    # Si hay múltiples dr_* (cross-method), no podemos inferir una sola
    if len(dr_tables) > 1:
        return None
    ordered = sorted(required_tables)
    return ordered[0] if ordered else None


def evaluate_case(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    selected_tables = set(pred.get("selected_tables", []))
    ranking = pred.get("ranking", [])
    semantic = pred.get("semantic_parse", {}) or {}
    intents = semantic.get("intents", {}) or {}
    metric = semantic.get("metric")
    operator = semantic.get("operator")
    result_types = set(safe_list(semantic.get("result_types_requested")))
    predicted_entity_types = set(get_predicted_entity_types(semantic))

    required_tables = set(gold.get("required_tables", []))
    forbidden_tables = set(gold.get("forbidden_tables", []))

    top1_table = ranking[0][0] if ranking else None
    expected_primary_table = infer_primary_table(required_tables, gold)

    contains_required = required_tables.issubset(selected_tables)
    missing_required = sorted(required_tables - selected_tables)
    forbidden_present = sorted(selected_tables & forbidden_tables)

    gold_intent = gold.get("intent")
    gold_metric = gold.get("metric")
    gold_operator = gold.get("operator")
    gold_result_types = set(gold.get("result_types_requested", []))
    gold_predicted_entity_types = set(get_predicted_entity_types(gold))

    intent_ok = gold_intent in intents if gold_intent is not None else True
    metric_ok = (gold_metric == metric) if gold_metric is not None else True
    operator_ok = (gold_operator == operator) if gold_operator is not None else True
    result_types_ok = gold_result_types == result_types
    predicted_entity_types_ok = gold_predicted_entity_types == predicted_entity_types

    # Evaluación genérica: el top-1 debe coincidir con la tabla principal esperada.
    # Para cross-method (expected_primary_table is None), top1_ok es True si
    # top-1 es cualquier dr_* table en required_tables.
    if expected_primary_table is not None:
        top1_ok = (top1_table == expected_primary_table)
    else:
        dr_required = {t for t in required_tables if t.startswith("dr_")}
        top1_ok = top1_table in dr_required if dr_required else True

    exact_semantic_ok = all([intent_ok, metric_ok, operator_ok, result_types_ok, predicted_entity_types_ok])
    contamination_ok = len(forbidden_present) == 0
    exact_retrieval_ok = contains_required and top1_ok and contamination_ok and exact_semantic_ok

    # Precision / Recall / F1 sobre tablas seleccionadas vs required_tables
    tp = len(selected_tables & required_tables)
    precision = tp / len(selected_tables) if selected_tables else 0.0
    recall = tp / len(required_tables) if required_tables else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # MRR: posición recíproca de la tabla principal esperada en el ranking
    mrr = 0.0
    if expected_primary_table and ranking:
        for rank_idx, (tbl, _score) in enumerate(ranking, start=1):
            if tbl == expected_primary_table:
                mrr = 1.0 / rank_idx
                break

    n_required_tables = len(required_tables)

    return {
        "top1_table": top1_table,
        "expected_primary_table": expected_primary_table,
        "selected_tables": sorted(selected_tables),
        "contains_required_tables": contains_required,
        "missing_required_tables": missing_required,
        "forbidden_tables_present": forbidden_present,
        "intent_ok": intent_ok,
        "metric_ok": metric_ok,
        "operator_ok": operator_ok,
        "result_types_ok": result_types_ok,
        "predicted_entity_types_ok": predicted_entity_types_ok,
        "top1_ok": top1_ok,
        "contamination_ok": contamination_ok,
        "exact_semantic_ok": exact_semantic_ok,
        "exact_retrieval_ok": exact_retrieval_ok,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "n_required_tables": n_required_tables,
        "predicted_semantic": {
            "intents": intents,
            "metric": metric,
            "operator": operator,
            "result_types_requested": sorted(result_types),
            "predicted_entity_types": sorted(predicted_entity_types),
        },
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "n_cases": 0,
            "top1_accuracy": 0.0,
            "required_tables_accuracy": 0.0,
            "contamination_free_rate": 0.0,
            "intent_accuracy": 0.0,
            "metric_accuracy": 0.0,
            "operator_accuracy": 0.0,
            "result_types_accuracy": 0.0,
            "predicted_entity_types_accuracy": 0.0,
            "semantic_exact_accuracy": 0.0,
            "exact_retrieval_accuracy": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "avg_mrr": 0.0,
            "avg_missing_required_tables": 0.0,
            "avg_forbidden_tables_selected": 0.0,
            "per_metric": {},
            "per_intent": {},
            "per_primary_table": {},
            "per_n_required_tables": {},
        }

    contamination_counts = [len(r["eval"]["forbidden_tables_present"]) for r in results]
    missing_counts = [len(r["eval"]["missing_required_tables"]) for r in results]

    def block_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        m = len(rows)
        return {
            "n": m,
            "top1_accuracy": sum(1 for x in rows if x["eval"]["top1_ok"]) / m,
            "required_tables_accuracy": sum(1 for x in rows if x["eval"]["contains_required_tables"]) / m,
            "intent_accuracy": sum(1 for x in rows if x["eval"]["intent_ok"]) / m,
            "metric_accuracy": sum(1 for x in rows if x["eval"]["metric_ok"]) / m,
            "operator_accuracy": sum(1 for x in rows if x["eval"]["operator_ok"]) / m,
            "result_types_accuracy": sum(1 for x in rows if x["eval"]["result_types_ok"]) / m,
            "predicted_entity_types_accuracy": sum(1 for x in rows if x["eval"]["predicted_entity_types_ok"]) / m,
            "semantic_exact_accuracy": sum(1 for x in rows if x["eval"]["exact_semantic_ok"]) / m,
            "exact_retrieval_accuracy": sum(1 for x in rows if x["eval"]["exact_retrieval_ok"]) / m,
            "avg_precision": statistics.mean(x["eval"]["precision"] for x in rows),
            "avg_recall": statistics.mean(x["eval"]["recall"] for x in rows),
            "avg_f1": statistics.mean(x["eval"]["f1"] for x in rows),
            "avg_mrr": statistics.mean(x["eval"]["mrr"] for x in rows),
        }

    by_metric: Dict[str, List[Dict[str, Any]]] = {}
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    by_primary_table: Dict[str, List[Dict[str, Any]]] = {}
    by_n_required: Dict[str, List[Dict[str, Any]]] = {}

    for r in results:
        gold = r["gold"]
        metric = gold.get("metric", "UNKNOWN")
        intent = gold.get("intent", "UNKNOWN")
        primary_table = r["eval"].get("expected_primary_table") or "UNKNOWN"
        n_req = r["eval"].get("n_required_tables", 0)
        n_req_key = f"{n_req}_tables" if n_req <= 3 else "4+_tables"

        by_metric.setdefault(metric, []).append(r)
        by_intent.setdefault(intent, []).append(r)
        by_primary_table.setdefault(primary_table, []).append(r)
        by_n_required.setdefault(n_req_key, []).append(r)

    per_metric = {k: block_summary(v) for k, v in sorted(by_metric.items(), key=lambda x: (x[0] is None, x[0] or ""))}
    per_intent = {k: block_summary(v) for k, v in sorted(by_intent.items(), key=lambda x: (x[0] is None, x[0] or ""))}
    per_primary_table = {k: block_summary(v) for k, v in sorted(by_primary_table.items(), key=lambda x: (x[0] is None, x[0] or ""))}
    per_n_required = {k: block_summary(v) for k, v in sorted(by_n_required.items(), key=lambda x: (x[0] is None, x[0] or ""))}

    return {
        "n_cases": n,
        "top1_accuracy": sum(1 for r in results if r["eval"]["top1_ok"]) / n,
        "required_tables_accuracy": sum(1 for r in results if r["eval"]["contains_required_tables"]) / n,
        "contamination_free_rate": sum(1 for r in results if r["eval"]["contamination_ok"]) / n,
        "intent_accuracy": sum(1 for r in results if r["eval"]["intent_ok"]) / n,
        "metric_accuracy": sum(1 for r in results if r["eval"]["metric_ok"]) / n,
        "operator_accuracy": sum(1 for r in results if r["eval"]["operator_ok"]) / n,
        "result_types_accuracy": sum(1 for r in results if r["eval"]["result_types_ok"]) / n,
        "predicted_entity_types_accuracy": sum(1 for r in results if r["eval"]["predicted_entity_types_ok"]) / n,
        "semantic_exact_accuracy": sum(1 for r in results if r["eval"]["exact_semantic_ok"]) / n,
        "exact_retrieval_accuracy": sum(1 for r in results if r["eval"]["exact_retrieval_ok"]) / n,
        "avg_precision": statistics.mean(r["eval"]["precision"] for r in results),
        "avg_recall": statistics.mean(r["eval"]["recall"] for r in results),
        "avg_f1": statistics.mean(r["eval"]["f1"] for r in results),
        "avg_mrr": statistics.mean(r["eval"]["mrr"] for r in results),
        "avg_missing_required_tables": statistics.mean(missing_counts) if missing_counts else 0.0,
        "avg_forbidden_tables_selected": statistics.mean(contamination_counts) if contamination_counts else 0.0,
        "per_metric": per_metric,
        "per_intent": per_intent,
        "per_primary_table": per_primary_table,
        "per_n_required_tables": per_n_required,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluator genérico para retrievers de tablas del esquema DRIVE.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Ruta al JSON de evaluación.",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="SQLAlchemy database URL.",
    )
    parser.add_argument(
        "--retriever-module",
        default="drive_qa.retriever",
        help="Nombre del módulo Python que expone SCHEMA_CATALOG y retrieve_schema.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=int, default=2)
    parser.add_argument("--expand-relations", action="store_true", default=False)
    parser.add_argument("--output", default="retriever_eval_results.json")
    parser.add_argument("--log-level", default="WARNING",
                        help="Log level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_path = resolve_project_path(args.dataset)
    output_path = resolve_project_path(args.output)

    cases = load_dataset(dataset_path)
    retriever_module = import_retriever_module(args.retriever_module)
    engine = create_engine(args.db_url)

    results = []
    for case in cases:
        gold = case.get("gold", case.get("ground_truth"))
        if gold is None:
            raise ValueError(f"El caso {case.get('id')} no tiene clave 'gold' ni 'ground_truth'.")

        pred = retriever_module.retrieve_schema(
            question=case["question"],
            catalog=retriever_module.SCHEMA_CATALOG,
            engine=engine,
            top_k=args.top_k,
            min_score=args.min_score,
            expand_relations=args.expand_relations,
            # limit_per_type and max_entities_per_type use function defaults
            prefer_single_exact=True,
        )
        ev = evaluate_case(pred, gold)
        results.append({
            "id": case.get("id"),
            "question": case.get("question"),
            "gold": gold,
            "eval": ev,
            "predicted_selected_tables": pred.get("selected_tables", []),
            "predicted_ranking": pred.get("ranking", []),
            "predicted_semantic_parse": pred.get("semantic_parse", {}),
            "detected_entities": pred.get("detected_entities", {}),
            "entity_strategies": pred.get("entity_strategies", {}),
        })

    report = {
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT) if dataset_path.is_relative_to(PROJECT_ROOT) else dataset_path),
        "retriever_module": args.retriever_module,
        "config": {
            "top_k": args.top_k,
            "min_score": args.min_score,
            "expand_relations": args.expand_relations,
        },
        "summary": summarize(results),
        "cases": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
