"""
Semantic parser for natural language questions about the DRIVE database.

Responsibilities:
- Intent detection (which computational method is being asked about)
- Metric detection (which specific metric)
- Operator detection (max/min/count/sort)
- Result type detection (what entities are being requested)
- Entity type inference (what entity types are expected in the question)
- Query template classification
- Candidate span extraction for entity search
- Subtype classification (disease_to_drug, pair_lookup, etc.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from drive_qa.text_normalization import (
    ENTITY_QUERY_STOPWORDS,
    normalize_text,
    sort_dict_desc,
    tokenize,
)
from drive_qa.vocabulary import (
    INTENT_PATTERNS,
    METRIC_ALIASES,
    PATTERN_SEARCH_RULES,
    RESULT_TYPE_SYNONYMS,
)


@dataclass
class SemanticParse:
    intents: Dict[str, int]
    metric: Optional[str]
    operator: Optional[str]
    result_types_requested: List[str]
    predicted_entity_types: List[str]
    query_template: str
    candidate_spans: List[str]
    pattern_targets: Dict[str, bool] = field(default_factory=dict)
    method_subtype: Optional[str] = None
    is_cross_method: bool = False
    question: str = ""


# =========================================================
# Pattern search target detection
# =========================================================

def detect_pattern_search_targets(question: str) -> Dict[str, bool]:
    q_norm = normalize_text(question)
    targets = {
        "diseases": False,
        "drugs": False,
        "genes": False,
        "pathways": False,
    }

    for entity_type, rules in PATTERN_SEARCH_RULES.items():
        if any(normalize_text(rule) in q_norm for rule in rules):
            targets[entity_type] = True

    return targets


# =========================================================
# Intent detection
# =========================================================

def detect_intent(question: str) -> Dict[str, int]:
    q_norm = normalize_text(question)
    scores: Dict[str, int] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pat in patterns:
            matches = re.findall(pat, q_norm)
            if matches:
                score += 2 * len(matches)
        if score > 0:
            scores[intent] = score

    if re.search(r"\b(redirection|dmsr|dmsrc|behor|behorc|gnn|gnns|graph neural network|link prediction)\b", q_norm):
        scores["gnn_score"] = scores.get("gnn_score", 0) + 2

    if re.search(
        r"\b(network proximity|proximity|closest distance|dc_mean|dc_std|"
        r"proximidad en red|cercania en red|cercanía en red|"
        r"distancia media|media de distancia|promedio de distancia|"
        r"desviacion estandar de la distancia|desviación estándar de la distancia|"
        r"desviacion tipica de la distancia|desviación típica de la distancia|"
        r"variabilidad de la distancia|dispersion de la distancia|dispersión de la distancia)\b",
        q_norm,
    ):
        scores["network_proximity"] = scores.get("network_proximity", 0) + 2

    if re.search(r"\b(dr_pathways_count|dr_pathways|conteo de pathways|pathways contados?)\b", q_norm):
        scores["pathways_method"] = scores.get("pathways_method", 0) + 2

    return sort_dict_desc(scores)


# =========================================================
# Metric detection
# =========================================================

def detect_metric(question: str) -> Optional[str]:
    q_norm = normalize_text(question)
    alias_hits: List[Tuple[int, int, str]] = []

    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm in q_norm:
                alias_hits.append((len(alias_norm.split()), len(alias_norm), metric))

    if alias_hits:
        alias_hits.sort(reverse=True)
        return alias_hits[0][2]

    return None


# =========================================================
# Operator detection
# =========================================================

def detect_operator(question: str) -> Optional[str]:
    q_norm = normalize_text(question)

    if re.search(r"\b(how many|cuantos|cuántos|count|numero de|número de)\b", q_norm):
        return "count"

    if re.search(r"\b(ordena|rank|sort by|list the top|more than \d+)\b", q_norm):
        return "sort"

    if re.search(
        r"\b("
        r"most|top|highest|max|maximum|maximo|máximo|mayor|best|ranking|"
        r"puntuacion maxima|puntuación máxima|valor maximo|valor máximo|"
        r"mas alto|más alto|score mas alto|score más alto|"
        r"score mas elevado|score más elevado|highest score|"
        r"mejor puntuad[oa]s?|mejor valorad[oa]s?|"
        r"mayor cercania|mayor cercanía|mas cercano en la red|más cercano en la red|"
        r"mas proximo en la red|más próximo en la red|mayor proximidad|mayor proximity"
        r")\b",
        q_norm,
    ):
        return "max"

    if re.search(
        r"\b("
        r"lowest|min|minimum|menor|"
        r"mas cercano|más cercano|mas proximo|más próximo|"
        r"closest|nearest|menor proximity|lower proximity|"
        r"menor distancia|distancia minima|distancia mínima|distancia mas corta|distancia más corta|"
        r"menor closest distance|menor dc_mean|menor dc_std|"
        r"menor distancia media|menor promedio de distancia|"
        r"menor desviacion estandar|menor desviación estándar|"
        r"menor desviacion tipica|menor desviación típica|"
        r"menor variabilidad|menor dispersion|menor dispersión"
        r")\b",
        q_norm,
    ):
        return "min"

    return None


# =========================================================
# Leading result type (header-based)
# =========================================================

def _leading_result_type(question: str) -> Optional[str]:
    q = normalize_text(question)

    patterns = [
        (r"^\s*(que|qué|cual|cuál|cuales|cuáles|which)\s+(es\s+el\s+|son\s+los\s+|son\s+las\s+)?(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)\b", "drugs"),
        (r"^\s*(lista|list|dame|muestrame|muéstrame|indica)\s+(los|las)?\s*(farmacos|fármacos|medicamentos|drug|drugs)\b", "drugs"),
        (r"^\s*ordena\s+(los|las)?\s*(farmacos|fármacos|medicamentos|drug|drugs)\b", "drugs"),
        (r"^\s*(rank|compare|list the top \d+)\s+(the\s+)?(drugs?|medications?)\b", "drugs"),
        (r"^\s*(que|qué|cual|cuál|cuales|cuáles|which)\s+(es\s+la\s+|son\s+las\s+|son\s+los\s+)?(enfermedad|enfermedades|disease|diseases)\b", "diseases"),
        (r"^\s*para\s+(que|qué)\s+(enfermedad|enfermedades|disease|diseases)\b", "diseases"),
        (r"^\s*(lista|list|dame|muestrame|muéstrame|indica)\s+(las|los)?\s*(enfermedades|disease|diseases)\b", "diseases"),
        (r"^\s*ordena\s+(los|las)?\s*(enfermedades|disease|diseases)\b", "diseases"),
        (r"^\s*(rank|compare|list the top \d+)\s+(the\s+)?(diseases?)\b", "diseases"),
        (r"^\s*(for which|for what)\s+(disease|diseases)\b", "diseases"),
        (r"^\s*(what|which)\s+genes?\b", "genes"),
    ]
    for pat, result in patterns:
        if re.search(pat, q):
            return result
    return None


# =========================================================
# Pair metric lookup detection
# =========================================================

def is_gnn_pair_metric_lookup(question: str, intents: Dict[str, int]) -> bool:
    q = normalize_text(question)

    if "gnn_score" not in intents and not re.search(r"\b(redirection|dmsr|dmsrc|behor|behorc)\b", q):
        return False

    metric_value_pattern = r"\b(score|puntuacion|puntuación|valor)\b"
    metric_name_pattern = r"\b(redirection|dmsr|dmsrc|behor|behorc)\b"

    has_value_word = bool(re.search(metric_value_pattern, q))
    has_metric_name = bool(re.search(metric_name_pattern, q))
    has_between_connector = bool(re.search(r"\b(entre|between)\b", q))
    has_both_entity_markers = bool(
        re.search(r"\b(enfermedad|disease)\b", q)
        and re.search(r"\b(farmaco|fármaco|medicamento|drug)\b", q)
    )

    if _leading_result_type(question) is not None:
        return False

    if has_metric_name and has_value_word and (has_between_connector or has_both_entity_markers):
        return True

    explicit_pair_patterns = [
        r"\b(cual es|cuál es|que valor|qué valor|que puntuacion|qué puntuación|que score|qué score)\b.*\b(redirection|dmsr|dmsrc|behor|behorc)\b.*\b(entre|between)\b",
        r"\bvalor\s+de\s+(redirection|dmsr|dmsrc|behor|behorc)\b.*\b(enfermedad|disease)\b.*\b(farmaco|fármaco|medicamento|drug)\b",
        r"\b(score|puntuacion|puntuación|valor)\b.*\b(redirection|dmsr|dmsrc|behor|behorc)\b.*\bentre\b.+\by\b.+",
    ]
    return any(re.search(pat, q) for pat in explicit_pair_patterns)


def is_network_pair_metric_lookup(question: str, intents: Optional[Dict[str, int]] = None) -> bool:
    q = normalize_text(question)
    intents = intents or {}

    network_metric_words = (
        r"proximity|network proximity|proximidad|proximidad en red|cercania en red|"
        r"closest distance|distance|distancia|distancia minima|distancia mas corta|"
        r"dc_mean|distancia media|media de distancia|promedio de distancia|"
        r"dc_std|desviacion estandar de la distancia|desviacion tipica de la distancia|"
        r"variabilidad de la distancia|dispersion de la distancia"
    )

    if "network_proximity" not in intents and not re.search(rf"\b({network_metric_words})\b", q):
        return False

    if _leading_result_type(question) is not None:
        return False

    has_metric_word = bool(re.search(rf"\b({network_metric_words})\b", q))
    has_between_connector = bool(re.search(r"\b(entre|between)\b", q))
    has_both_entity_markers = bool(
        re.search(r"\b(enfermedad|disease)\b", q)
        and re.search(r"\b(farmaco|fármaco|medicamento|drug)\b", q)
    )

    explicit_pair_patterns = [
        rf"\b(cual es|cuál es|que valor|qué valor)\b.*\b({network_metric_words})\b.*\b(entre|between)\b",
        rf"\bvalor\s+de\s+({network_metric_words})\b.*\b(enfermedad|disease)\b.*\b(farmaco|fármaco|medicamento|drug)\b",
        rf"\b({network_metric_words})\b.*\bentre\b.+\by\b.+",
        rf"\b(que tan cerca en la red|qué tan cerca en la red|proximidad en red|cercania en red|cercanía en red)\b.*\bentre\b",
    ]
    if any(re.search(pat, q) for pat in explicit_pair_patterns):
        return True

    return has_metric_word and (has_between_connector or has_both_entity_markers)


def is_threshold_pair_metric_lookup(question: str, intents: Optional[Dict[str, int]] = None) -> bool:
    q = normalize_text(question)
    intents = intents or {}

    threshold_words = (
        r"threshold|threshold value|threshold values|threshold-based|"
        r"umbral|valor umbral|valores umbral|modelo de valores umbral|modelo basado en umbral"
    )
    metric_words = r"score|scores|valor|valores|puntuacion|puntuación|action type|accion|acción|inhibitor|agonist|antagonist|binding|weak inhibitor"

    if "threshold_values" not in intents and not re.search(rf"({threshold_words})", q):
        return False

    if _leading_result_type(question) is not None:
        return False

    has_metric_word = bool(re.search(rf"({metric_words})", q))
    has_pair_connector = bool(re.search(r"(entre|between|par|caso)", q) or "–" in question or "-" in question)
    has_explicit_pair = bool(
        re.search(r"(enfermedad|disease)", q) and re.search(r"(farmaco|fármaco|medicamento|drug)", q)
    )

    has_two_named_chunks = bool(
        re.search(r"para", q)
        and re.search(r"(score|valor|valores|puntuacion|puntuación)", q)
    )

    threshold_para_pattern = bool(
        has_metric_word
        and re.search(r"(que|qué|cual|cuál)", q)
        and re.search(r"(para|de)", q)
        and not re.search(r"(genes|gene|enfermedades|diseases|farmacos|fármacos|medicamentos|drugs)", q)
    )

    return has_metric_word and (has_pair_connector or has_explicit_pair or threshold_para_pattern or has_two_named_chunks)


# =========================================================
# Subtype classification
# =========================================================

def classify_gnn_subtype(question: str, intents: Dict[str, int]) -> Optional[str]:
    q = normalize_text(question)

    if "gnn_score" not in intents:
        return None

    if is_gnn_pair_metric_lookup(question, intents):
        return "pair_lookup"

    leading = _leading_result_type(question)
    if leading == "drugs":
        return "disease_to_drug"
    if leading == "diseases":
        return "drug_to_disease"

    if re.search(r"\b(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)\b", q):
        if re.search(r"\bpara\b", q) or re.search(r"\b(asociados?|relacionados?)\b", q):
            return "disease_to_drug"

    if re.search(r"\b(enfermedad|enfermedades|disease|diseases)\b", q):
        if re.search(r"\bpara\b", q):
            return "drug_to_disease"

    if re.search(r"\bordena\b", q) and re.search(r"\benfermedades\b", q):
        return "drug_to_disease"
    if re.search(r"\bordena\b", q) and re.search(r"\b(farmacos|fármacos|medicamentos|drug|drugs)\b", q):
        return "disease_to_drug"

    return None


def classify_network_subtype(question: str, intents: Dict[str, int]) -> Optional[str]:
    q = normalize_text(question)

    if "network_proximity" not in intents:
        return None

    if is_network_pair_metric_lookup(question, intents):
        return "pair_lookup"

    leading = _leading_result_type(question)
    if leading == "drugs":
        return "disease_to_drug"
    if leading == "diseases":
        return "drug_to_disease"

    if re.search(r"\b(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)\b", q):
        if re.search(r"\bpara\b", q) or re.search(r"\b(a|de)\b", q):
            return "disease_to_drug"

    if re.search(r"\b(enfermedad|enfermedades|disease|diseases)\b", q):
        if re.search(r"\bpara\b", q):
            return "drug_to_disease"

    if re.search(r"\bordena\b", q) and re.search(r"\benfermedades\b", q):
        return "drug_to_disease"
    if re.search(r"\bordena\b", q) and re.search(r"\b(farmacos|fármacos|medicamentos|drug|drugs)\b", q):
        return "disease_to_drug"

    return None


def classify_threshold_subtype(question: str, intents: Dict[str, int]) -> Optional[str]:
    q = normalize_text(question)

    if "threshold_values" not in intents:
        return None

    if is_threshold_pair_metric_lookup(question, intents):
        return "pair_lookup"

    leading = _leading_result_type(question)
    if leading == "drugs":
        return "disease_to_drug"
    if leading == "diseases":
        return "drug_to_disease"

    if leading == "genes":
        if re.search(r"\b(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)\b", q) and re.search(r"\b(enfermedad|enfermedades|disease|diseases)\b", q):
            return "pair_to_gene"
        if re.search(r"\b(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)\b", q):
            return "drug_to_gene"
        return "disease_to_gene"

    if re.search(r"\bordena\b", q) and re.search(r"\benfermedades\b", q):
        return "drug_to_disease"
    if re.search(r"\bordena\b", q) and re.search(r"\b(farmacos|fármacos|medicamentos|drug|drugs)\b", q):
        return "disease_to_drug"
    if re.search(r"\bordena\b", q) and re.search(r"\b(genes|gene|gen)\b", q):
        if re.search(r"\b(farmaco|fármaco|medicamento|drug)\b", q):
            return "drug_to_gene"
        return "disease_to_gene"

    return None


# =========================================================
# Intent inference from metric and language
# =========================================================

def infer_intents_from_metric_and_language(
    question: str,
    intents: Dict[str, int],
    metric: Optional[str],
) -> Dict[str, int]:
    q = normalize_text(question)
    inferred = dict(intents)

    if metric in {"redirection", "dmsr", "dmsrc", "behor", "behorc"}:
        inferred["gnn_score"] = max(inferred.get("gnn_score", 0), 4)

    if metric in {"closest_distance", "dc_mean", "dc_std", "proximity"}:
        inferred["network_proximity"] = max(inferred.get("network_proximity", 0), 4)

    if (
        "network_proximity" not in inferred
        and re.search(
            r"\b(distancia mas corta|distancia más corta|menor distancia|closest distance|proximidad|proximity|dc_mean|dc_std)\b",
            q,
        )
    ):
        inferred["network_proximity"] = max(inferred.get("network_proximity", 0), 3)

    if (
        "threshold_values" not in inferred
        and (
            metric == "action_type"
            or re.search(
                r"\b(threshold values|threshold value|threshold-based|modelo de valores umbral|método de valores umbral|metodo de valores umbral|modelo basado en umbral|método basado en umbral|metodo basado en umbral|valores umbral|valor umbral)\b",
                q,
            )
        )
    ):
        threshold_evidence = 0
        if metric == "action_type":
            threshold_evidence += 2
        if metric == "score" and not re.search(r"\b(pathway|pathways|approach|método de pathways|metodo de pathways)\b", q):
            threshold_evidence += 1
        if re.search(
            r"\b(threshold values|threshold value|threshold-based|modelo de valores umbral|método de valores umbral|metodo de valores umbral|modelo basado en umbral|método basado en umbral|metodo basado en umbral|valores umbral|valor umbral)\b",
            q,
        ):
            threshold_evidence += 2
        if re.search(r"\b(gen|genes|gene|shared target gene|gen diana compartido|gen diana compartidos?)\b", q):
            threshold_evidence += 1
        if threshold_evidence >= 2:
            inferred["threshold_values"] = max(inferred.get("threshold_values", 0), 3)

    if metric == "path_id":
        inferred["information_paths"] = max(inferred.get("information_paths", 0), 4)
    if (
        "information_paths" not in inferred
        and re.search(
            r"\b(caminos? de informacion|vias? de informacion|rutas? de informacion|information path|path_id)\b",
            q,
        )
    ):
        inferred["information_paths"] = max(inferred.get("information_paths", 0), 3)

    if metric in {"approach", "association_type", "count"}:
        inferred["pathways_method"] = max(inferred.get("pathways_method", 0), 4)
    if (
        metric == "score"
        and "threshold_values" not in inferred
        and re.search(
            r"\b(pathway|pathways|approach|método de pathways|metodo de pathways|score en pathways|puntuacion en pathways|puntuación en pathways)\b",
            q,
        )
    ):
        inferred["pathways_method"] = max(inferred.get("pathways_method", 0), 4)
    if (
        "pathways_method" not in inferred
        and "disease_pathways" not in inferred
        and not re.search(r"\b(enfermedades? similares?|transferir|reposicionar|enfermedades? comparten?|enfermedades? relacionadas|enfermedades? ya tratadas?|mecanismos? biologicos?|mecanismos? biológicos?)\b", q)
        and re.search(
            r"\b(approach|conteo de pathways|numero de pathways|número de pathways|pathways compartidos|vias biologicas|vías biológicas|metodo de pathways|método de pathways|score en pathways|puntuacion en pathways|puntuación en pathways|dr_pathways|dr_pathways_count|asociaciones por pathways|tipo de asociacion|tipo de asociación)\b",
            q,
        )
    ):
        inferred["pathways_method"] = max(inferred.get("pathways_method", 0), 3)

    if (
        "disease_pathways" not in inferred
        and re.search(
            r"\b(disease pathways|enfermedad original|enfermedad nueva|nueva enfermedad|indicaciones? terapeuticas|indicaciones? terapéuticas|enfermedades? candidatas?|transferencia de tratamiento|similaridad entre enfermedades|similitud entre enfermedades|reposicionamiento entre enfermedades|dr_diseasepathways|nueva indicacion|nueva indicación)\b",
            q,
        )
    ):
        inferred["disease_pathways"] = max(inferred.get("disease_pathways", 0), 3)

    disease_similarity_signals = [
        r"\benfermedades? similares?\b",
        r"\benfermedades?\b.{0,20}\bsimilares?\b",
        r"\bsimilitud (con|de)\b",
        r"\bsimilaridad (con|de|entre)\b",
        r"\btransferir.*tratamientos?\b",
        r"\btratamientos? transferibles?\b",
        r"\bcompartir? tratamientos?\b",
        r"\bmecanismos? (biologicos?|biológicos?)\b",
        r"\bcandidatos? (terapeuticos?|terapéuticos?)\b",
        r"\bcandidatas? a.*reposicionamiento\b",
        r"\bdesde enfermedades?\b",
        r"\ba partir de enfermedades?\b",
        r"\benfermedades? comparten?\b",
        r"\benfermedades? (que|con) comparten?\b",
        r"\breutilizar.*enfermedades?\b",
        r"\breusar.*tratamientos?\b",
        r"\bbeneficiarse del tratamiento\b",
        r"\benfermedades? relacionadas\b",
        r"\benfermedades? ya tratadas?\b",
        r"\breposicionar\b",
        r"\breposicionarse\b",
        r"\breposicionamiento\b",
        r"\bpathways? (similares|comunes|compartidos|en comun|en común)\b",
        r"\bvias? (biologicas?|biológicas?)\b.*\b(transferir|compartir|reusar|reposicionar)\b",
        r"\b(transferir|compartir|reusar|reposicionar)\b.*\bvias? (biologicas?|biológicas?)\b",
        r"\bvias? (biologicas?|biológicas?) (similares|comunes|compartidas)\b",
        r"\bvías? (biológicas?|biologicas?) (similares|comunes|compartidas)\b",
        # English disease-pathway similarity signals
        r"\bsimilar diseases?\b",
        r"\bdiseases?.*similar\b",
        r"\bshares? pathways?\b",
        r"\bshare pathways?\b",
        r"\bpathway overlap\b",
        r"\bpathway similarity\b",
        r"\bshared biological\b",
        r"\bshared.*mechanisms?\b",
        r"\btreatment transfer\b",
        r"\bshare treatments?\b",
        r"\bcould be repurposed\b",
        r"\brepurposed.*based on.*pathway\b",
        r"\bdiseases? that share\b",
        r"\bdiseases? with shared\b",
    ]
    ds_signal_count = sum(1 for pat in disease_similarity_signals if re.search(pat, q))
    if ds_signal_count >= 1:
        inferred["disease_pathways"] = max(inferred.get("disease_pathways", 0), 3 + ds_signal_count)
        # Only apply mutual exclusion when NOT a cross-method query
        if ("cross_method_aggregation" not in inferred
            and "pathways_method" in inferred
            and inferred.get("pathways_method", 0) <= inferred.get("disease_pathways", 0)):
            del inferred["pathways_method"]

    # Reciprocal exclusion: if pathways_method clearly dominates and no cross-method
    if ("cross_method_aggregation" not in inferred
        and "pathways_method" in inferred
        and "disease_pathways" in inferred
        and inferred["pathways_method"] > inferred["disease_pathways"] + 2):
        del inferred["disease_pathways"]

    # English cross-method reinforcement: explicit method mentions + cross-method keywords
    if "cross_method_aggregation" not in inferred or inferred.get("cross_method_aggregation", 0) < 3:
        cross_method_english_signals = [
            r"\bhow many methods\b",
            r"\bin how many methods\b",
            r"\bin both\b",
            r"\bappears? in both\b",
            r"\bat least \d+ (?:different )?methods?\b",
            r"\bmultiple methods\b",
            r"\bdifferent methods\b",
            r"\bcomputational methods?\b",
            r"\brepurposing methods?\b",
            r"\bmore methods\b",
            r"\bacross.*methods\b",
            r"\bin all.*methods\b",
            r"\boverlap in the same\b",
            r"\bsame repurposing methods\b",
        ]
        cm_signal_count = sum(1 for pat in cross_method_english_signals if re.search(pat, q))
        if cm_signal_count >= 1:
            inferred["cross_method_aggregation"] = max(inferred.get("cross_method_aggregation", 0), 3 + cm_signal_count)

    # If 2+ distinct method intents are detected AND "both" or similar combinator keywords present,
    # infer cross_method_aggregation
    if "cross_method_aggregation" not in inferred or inferred.get("cross_method_aggregation", 0) < 3:
        method_intents = {"gnn_score", "network_proximity", "information_paths", "threshold_values", "pathways_method", "disease_pathways"}
        detected_methods = [k for k in method_intents if k in inferred]
        if len(detected_methods) >= 2:
            if re.search(r"\b(both|ambos|y|and)\b", q):
                inferred["cross_method_aggregation"] = max(inferred.get("cross_method_aggregation", 0), 4)

    return inferred


# =========================================================
# Anchor entity type detection
# =========================================================

def detect_anchor_entity_type(question: str, intents: Optional[Dict[str, int]] = None) -> Optional[str]:
    q = normalize_text(question)
    gnn_subtype = classify_gnn_subtype(question, intents or {}) if intents is not None else None
    network_subtype = classify_network_subtype(question, intents or {}) if intents is not None else None
    threshold_subtype = classify_threshold_subtype(question, intents or {}) if intents is not None else None

    if (
        is_gnn_pair_metric_lookup(question, intents or {})
        or is_network_pair_metric_lookup(question, intents or {})
        or is_threshold_pair_metric_lookup(question, intents or {})
    ):
        return None

    subtype = gnn_subtype or network_subtype or threshold_subtype
    if subtype == "disease_to_drug":
        return "diseases"
    if subtype == "drug_to_disease":
        return "drugs"
    if subtype == "pair_lookup":
        return None
    if subtype == "disease_to_gene":
        return "diseases"
    if subtype == "drug_to_gene":
        return "drugs"
    if subtype == "pair_to_gene":
        return None

    if _leading_result_type(question) == "drugs":
        return "diseases"
    if _leading_result_type(question) == "diseases":
        return "drugs"

    disease_anchor_patterns = [
        r"\bpara la enfermedad\s+([a-z0-9_\-\s]+)$",
        r"\bfor the disease\s+([a-z0-9_\-\s]+)$",
        r"\bpara\s+([a-z0-9_\-\s]+)$",
        r"\bfor\s+([a-z0-9_\-\s]+)$",
    ]
    for pat in disease_anchor_patterns:
        if re.search(pat, q):
            if re.search(r"\b(score|puntuacion|puntuación)\b", q) and re.search(r"\b(entre|between)\b", q):
                return None
            return "diseases"

    return None


# =========================================================
# Threshold surface helpers
# =========================================================

def _threshold_surface_result_type(question: str) -> Optional[str]:
    q = normalize_text(question)

    if re.search(r"\b(genes|gene|gen diana|shared target gene|genes diana compartidos?)\b", q):
        return "genes"
    if re.search(r"\b(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)\b", q):
        return "drugs"
    if re.search(r"\b(enfermedad|enfermedades|disease|diseases)\b", q):
        return "diseases"
    return None


def _threshold_surface_entity_types(question: str, result_types_requested: List[str]) -> Optional[List[str]]:
    q = normalize_text(question)

    if is_threshold_pair_metric_lookup(question, {"threshold_values": 1}):
        return ["diseases", "drugs"]

    if "genes" in result_types_requested:
        has_drug_words = bool(re.search(r"\b(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)\b", q))
        has_disease_words = bool(re.search(r"\b(enfermedad|enfermedades|disease|diseases)\b", q))
        if has_drug_words and has_disease_words:
            return ["diseases", "drugs"]
        if has_drug_words:
            return ["drugs"]
        return ["diseases"]

    if result_types_requested == ["drugs"]:
        return ["diseases"]

    if result_types_requested == ["diseases"]:
        return ["drugs"]

    return None


# =========================================================
# Result types requested
# =========================================================

def detect_result_types_requested(question: str, intents: Optional[Dict[str, int]] = None) -> List[str]:
    q_norm = normalize_text(question)

    if is_gnn_pair_metric_lookup(question, intents or {}) or is_network_pair_metric_lookup(question, intents or {}) or is_threshold_pair_metric_lookup(question, intents or {}):
        return []

    if "threshold_values" in (intents or {}):
        threshold_surface = _threshold_surface_result_type(question)
        if threshold_surface == "genes":
            return ["genes"]
        if threshold_surface == "drugs":
            return ["drugs"]
        if threshold_surface == "diseases":
            return ["diseases"]

    gnn_subtype = classify_gnn_subtype(question, intents or {})
    network_subtype = classify_network_subtype(question, intents or {})
    threshold_subtype = classify_threshold_subtype(question, intents or {})
    subtype = gnn_subtype or network_subtype or threshold_subtype
    if subtype == "disease_to_drug":
        return ["drugs"]
    if subtype == "drug_to_disease":
        return ["diseases"]
    if subtype == "pair_lookup":
        return []
    if subtype in {"disease_to_gene", "drug_to_gene", "pair_to_gene"}:
        return ["genes"]

    if re.search(r"\b(score|puntuacion|puntuación|valor)\b", q_norm) and re.search(r"\b(entre|between|con)\b", q_norm):
        return []

    leading = _leading_result_type(question)
    if leading == "drugs":
        return ["drugs"]
    if leading == "diseases":
        return ["diseases"]

    results: Set[str] = set()

    header_patterns = [
        r"^(que|qué|cual|cuál|cuales|cuáles|how many|which)\s+([a-z0-9_\-\s]+)",
        r"^(dame|muestrame|muéstrame|lista|list|indica)\s+([a-z0-9_\-\s]+)",
        r"^(ordena)\s+([a-z0-9_\-\s]+)",
    ]

    header_text = q_norm
    for pat in header_patterns:
        m = re.search(pat, q_norm)
        if m:
            header_text = m.group(2)
            break

    filter_cut_patterns = [
        r"\bpara la enfermedad\b",
        r"\bpara el disease\b",
        r"\bfor the disease\b",
        r"\bcon nombre\b",
        r"\bwith name\b",
        r"\bsegun la metrica\b",
        r"\bsegún la métrica\b",
        r"\bsegun\b",
        r"\bsegún\b",
        r"\baccording to\b",
        r"\bfor\b",
        r"\bpara\b",
        r"\bwith\b",
        r"\bcontra\b",
        r"\bde la enfermedad\b",
        r"\bdel disease\b",
    ]
    cut_pos = len(header_text)
    for pat in filter_cut_patterns:
        m = re.search(pat, header_text)
        if m:
            cut_pos = min(cut_pos, m.start())
    header_text = header_text[:cut_pos].strip()

    for result_type, synonyms in RESULT_TYPE_SYNONYMS.items():
        for s in synonyms:
            s_norm = normalize_text(s)
            if re.search(rf"\b{s_norm}\b", header_text):
                results.add(result_type)

    if not results:
        prefix = " ".join(q_norm.split()[:6])
        for result_type, synonyms in RESULT_TYPE_SYNONYMS.items():
            for s in synonyms:
                s_norm = normalize_text(s)
                if re.search(rf"^(que|qué|cual|cuál|which|how many|ordena)\s+.*\b{s_norm}\b", prefix):
                    results.add(result_type)

    # When pathways_method or disease_pathways intent is active, "pathways" in the question
    # refers to the method, not to the desired result type, unless explicitly asking for pathway list
    if "pathways" in results and (intents or {}):
        if any(k in intents for k in ["pathways_method", "disease_pathways"]):
            if not re.search(r"^\s*(what|which|que|qué|cuales|cuáles)\s+(pathways?|rutas?|vias?|vías?)\b", q_norm):
                results.discard("pathways")

    return sorted(results)


# =========================================================
# Predicted entity types
# =========================================================

def infer_predicted_entity_types(
    question: str,
    intent_scores: Dict[str, int],
    result_types_requested: List[str],
) -> List[str]:
    q_norm = normalize_text(question)
    pattern_targets = detect_pattern_search_targets(question)

    if is_gnn_pair_metric_lookup(question, intent_scores) or is_network_pair_metric_lookup(question, intent_scores) or is_threshold_pair_metric_lookup(question, intent_scores):
        return ["diseases", "drugs"]

    if "threshold_values" in intent_scores:
        threshold_entity_types = _threshold_surface_entity_types(question, result_types_requested)
        if threshold_entity_types is not None:
            return threshold_entity_types

    gnn_subtype = classify_gnn_subtype(question, intent_scores)
    network_subtype = classify_network_subtype(question, intent_scores)
    threshold_subtype = classify_threshold_subtype(question, intent_scores)
    subtype = gnn_subtype or network_subtype or threshold_subtype
    if subtype == "disease_to_drug":
        return ["diseases"]
    if subtype == "drug_to_disease":
        return ["drugs"]
    if subtype == "pair_lookup":
        return ["diseases", "drugs"]
    if subtype == "disease_to_gene":
        return ["diseases"]
    if subtype == "drug_to_gene":
        return ["drugs"]
    if subtype == "pair_to_gene":
        return ["diseases", "drugs"]

    if "threshold_values" in intent_scores and "genes" in result_types_requested:
        if re.search(r"(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)", q_norm) and re.search(r"(enfermedad|enfermedades|disease|diseases)", q_norm):
            return ["diseases", "drugs"]
        if re.search(r"(farmaco|fármaco|farmacos|fármacos|medicamento|medicamentos|drug|drugs)", q_norm):
            return ["drugs"]
        return ["diseases"]

    # disease_pathways: entity to search is always the disease mentioned
    if "disease_pathways" in intent_scores:
        return ["diseases"]

    explicit_pattern_types = [k for k, v in pattern_targets.items() if v]
    if explicit_pattern_types:
        return explicit_pattern_types

    explicit_named_entity = bool(
        re.search(r"\b(con nombre|with name|called|llamad[oa]|named)\b", q_norm)
        or re.search(r"\bpara la enfermedad\b", q_norm)
        or re.search(r"\bfor the disease\b", q_norm)
    )
    if "cross_method_aggregation" in intent_scores and not explicit_named_entity:
        # Cross-method queries: infer entity types from result type and question context
        if result_types_requested == ["drugs"]:
            # Asking which drugs → need to ground the disease(s) mentioned
            if re.search(r"\b(between|entre)\b.*\b(drug|drugs|farmaco|medicamento)\b", q_norm):
                return ["drugs", "diseases"]
            return ["diseases"]
        if result_types_requested == ["diseases"]:
            # Asking which diseases → need to ground drugs or diseases mentioned
            if re.search(r"\b(drug|drugs|farmaco|medicamento|candidate)\b", q_norm):
                return ["drugs"]
            return ["diseases"]
        # No specific result type → pair lookup style
        return ["diseases", "drugs"]

    types_to_search: Set[str] = set()

    anchor_entity_type = detect_anchor_entity_type(question, intent_scores)
    if anchor_entity_type:
        types_to_search.add(anchor_entity_type)

    if re.search(r"\b(for|para|de|sobre|associated with|with|contra|entre)\b", q_norm):
        if "drugs" in result_types_requested:
            types_to_search.add("diseases")
        if "diseases" in result_types_requested:
            types_to_search.add("drugs")

    if re.search(r"\b(enfermedad|disease)\s+(con nombre|named|called)\b", q_norm):
        types_to_search.add("diseases")
    if re.search(r"\b(drug|farmaco|fármaco|medicamento)\s+(con nombre|named|called)\b", q_norm):
        types_to_search.add("drugs")

    if re.search(r"\b(score|puntuacion|puntuación|valor)\b", q_norm) and re.search(r"\b(entre|between|con)\b", q_norm):
        types_to_search.update(["diseases", "drugs"])

    # Pathway method pair: "between DRUG and DISEASE" implies looking up both entities
    if "pathways_method" in intent_scores and re.search(r"\b(between|entre)\b", q_norm):
        types_to_search.update(["diseases", "drugs"])

    if re.search(r"\b(disease|enfermedad|enfermedades)\b", q_norm) and "diseases" not in result_types_requested:
        types_to_search.add("diseases")
    if re.search(r"\b(drug|drugs|farmaco|farmacos|fármaco|fármacos|medicamento|medicamentos)\b", q_norm) and "drugs" not in result_types_requested:
        types_to_search.add("drugs")
    if re.search(r"\b(gene|genes|gen)\b", q_norm) and "genes" not in result_types_requested:
        types_to_search.add("genes")
    # Only add "pathways" entity search if no method intent is dominant
    if re.search(r"\b(pathway|pathways|ruta|rutas|via|vias|vía|vías)\b", q_norm) and "pathways" not in result_types_requested:
        if not any(k in intent_scores for k in ["pathways_method", "disease_pathways", "information_paths"]):
            types_to_search.add("pathways")

    if any(k in intent_scores for k in ["gnn_score", "network_proximity", "information_paths", "threshold_values", "pathways_method", "disease_pathways"]):
        if "drugs" in result_types_requested and "diseases" not in result_types_requested:
            types_to_search.add("diseases")
        if "diseases" in result_types_requested and "drugs" not in result_types_requested:
            types_to_search.add("drugs")

    if "information_paths" in intent_scores and not result_types_requested:
        types_to_search.update(["diseases", "drugs"])

    if "disease_pathways" in intent_scores and not result_types_requested:
        types_to_search.update(["diseases", "drugs"])

    if "pathways_method" in intent_scores and not result_types_requested:
        types_to_search.update(["diseases", "drugs"])

    if "protein_encoding" in intent_scores:
        types_to_search.add("genes")

    if not types_to_search and result_types_requested:
        if "drugs" in result_types_requested:
            types_to_search.add("diseases")
        if "diseases" in result_types_requested:
            types_to_search.add("drugs")
        if "genes" in result_types_requested or "pathways" in result_types_requested:
            types_to_search.add("diseases")

    return sorted(types_to_search)


# =========================================================
# Query template classification
# =========================================================

def classify_query_template(
    intents: Dict[str, int],
    result_types_requested: List[str],
    operator: Optional[str],
    model_subtype: Optional[str] = None,
) -> str:
    if model_subtype == "disease_to_drug":
        return "method_ranking_by_disease"
    if model_subtype == "drug_to_disease":
        return "method_ranking_by_drug"
    if model_subtype == "pair_lookup":
        return "association_lookup"
    if model_subtype in {"disease_to_gene", "drug_to_gene", "pair_to_gene"}:
        return "entity_pattern_search"

    if "threshold_values" in intents:
        if result_types_requested == ["drugs"]:
            return "method_ranking_by_disease"
        if result_types_requested == ["diseases"]:
            return "method_ranking_by_drug"
        if result_types_requested == ["genes"]:
            return "entity_pattern_search"

    if "cross_method_aggregation" in intents:
        return "cross_method_overlap"
    if operator == "count":
        return "aggregation_count"
    if "drugs" in result_types_requested and any(k in intents for k in ["gnn_score", "network_proximity", "information_paths", "threshold_values", "pathways_method", "disease_pathways"]):
        return "method_ranking_by_disease"
    if "diseases" in result_types_requested and "drugs" in result_types_requested:
        return "association_lookup"
    if result_types_requested:
        return "entity_pattern_search"
    return "generic_lookup"


# =========================================================
# Enumeration / comparison detection
# =========================================================

def _split_enumeration(text: str) -> List[str]:
    if not re.search(r'\b(o|or|y|and|versus|vs)\b', text):
        return []

    parts = re.split(r'\s*,\s*|\s+(?:o|or|y|and|versus|vs)\s+', text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3]

    if len(parts) >= 2:
        return parts
    return []


def _detect_comparison_pattern(question: str) -> bool:
    q_norm = normalize_text(question)
    patterns = [
        r'\w+\s*,\s*\w+\s+(?:o|or|y|and)\s+\w+',
        r'\w+\s+(?:o|or|versus|vs)\s+\w+',
        r'\bcompared?\s+(?:to|with|con)\b',
        r'\bcomparad[oa]s?\s+con\b',
        r'\bcompara\b',
    ]
    if re.search(r'\bentre\s+.+\s+y\s+', q_norm):
        if not re.search(r'\b(se genera|conecta|respalda|sustenta|soporta|path|camino|via|ruta)\b', q_norm):
            return True
    if re.search(r'\bbetween\s+.+\s+and\s+', q_norm):
        if not re.search(r'\b(path|connect|link|support)\b', q_norm):
            return True

    # Detect enumeration of proper-noun entities (e.g. "Asthma and Epilepsy")
    proper_noun_enum = re.findall(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:and|y|or|o)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        question,
    )
    if proper_noun_enum:
        return True

    return any(re.search(p, q_norm) for p in patterns)


# =========================================================
# Candidate span extraction
# =========================================================

def extract_candidate_spans(question: str, max_ngram: int = 5) -> List[str]:
    q_norm = normalize_text(question)
    spans: List[str] = []

    banned_single_tokens = {
        "estan", "están", "esta", "está", "tienen", "tiene", "existen", "existe",
        "presentes", "presente", "aparecen", "aparece", "alcanzan", "alcanza",
        "obtienen", "obtiene", "segun", "según", "metrica", "métrica", "valor",
        "score", "modelo", "nombre", "all", "methods", "techniques", "methodologies",
        "todos", "todas", "tablas", "hipotesis", "hipótesis",
    }

    connector_patterns = [
        r"\bpara la enfermedad\s+(.+)$",
        r"\bfor the disease\s+(.+)$",
        r"\bcon nombre\s+(.+)$",
        r"\bwith name\s+(.+)$",
        r"\bnamed\s+(.+)$",
        r"\bcalled\s+(.+)$",
        r"\bpara\s+(.+)$",
        r"\bfor\s+(.+)$",
        r"\bwith\s+(.+)$",
        r"\bcontra\s+(.+)$",
        r"\bassociated with\s+(.+)$",
        r"\bentre\s+(.+)$",
    ]
    for pat in connector_patterns:
        m = re.search(pat, q_norm)
        if m:
            tail = m.group(1).strip(" ?.")
            tail = re.sub(
                r"\b("
                r"en el modelo|in the model|segun|según|using|con score|score|"
                r"valor maximo|valor máximo|maximo|máximo|mayor|menor|mas alto|más alto|"
                r"segun la metrica|según la métrica|according to|metric|metrica|métrica"
                r")\b.*$",
                "",
                tail
            ).strip(" ?.")
            if len(tail) >= 5 and tail not in banned_single_tokens:
                spans.append(tail)
                enum_items = _split_enumeration(tail)
                for item in enum_items:
                    if len(item) >= 5 and item not in banned_single_tokens:
                        spans.append(item)

    clean_tokens = [
        t for t in normalize_text(question).split()
        if t not in ENTITY_QUERY_STOPWORDS
        and t not in banned_single_tokens
        and len(t) >= 4
        and not t.isdigit()
    ]

    for n in range(min(max_ngram, len(clean_tokens)), 0, -1):
        for i in range(0, len(clean_tokens) - n + 1):
            ng = " ".join(clean_tokens[i:i + n])
            if len(ng) < 5:
                continue
            if n == 1 and ng in banned_single_tokens:
                continue
            spans.append(ng)

    seen = set()
    final_spans = []
    for s in spans:
        s = s.strip()
        if not s:
            continue
        if len(s) < 5:
            continue
        if s in seen:
            continue
        seen.add(s)
        final_spans.append(s)

    return final_spans[:30]


# =========================================================
# Main parse_question orchestrator
# =========================================================

def parse_question(question: str) -> SemanticParse:
    intents = detect_intent(question)
    metric = detect_metric(question)
    if metric is None and "threshold_values" in intents:
        q_norm = normalize_text(question)
        if re.search(r"(action type|tipo de accion|tipo de acción|accion|acción|inhibitor|agonist|antagonist|binding|weak inhibitor)", q_norm):
            metric = "action_type"
        else:
            metric = "score"
    if metric is None and "pathways_method" in intents:
        q_norm = normalize_text(question)
        if re.search(r"(tipo de asociacion|tipos de asociacion|association.type)", q_norm):
            metric = "association_type"
        elif re.search(r"(conteo de pathways|conteos de pathways|pathways contados?|dr_pathways_count|conteo total|conteo maximo|conteo máximo|conteos de associations|pathway count|pathway counts|total pathway count|number of pathways|total pathways)", q_norm):
            metric = "count"
        elif re.search(r"(cuantos pathways|numero de pathways)", q_norm) and not re.search(r"\bdr_pathways\b", q_norm):
            metric = "count"
        elif re.search(r"\b(how many)\b.*\bpathways?\b", q_norm) and not re.search(r"\bdr_pathways\b", q_norm):
            metric = "count"
        elif re.search(r"\b(more|most|fewer|less|\d+)\s+pathways?\b", q_norm):
            metric = "count"
        else:
            metric = "score"

    intents = infer_intents_from_metric_and_language(question, intents, metric)
    operator = detect_operator(question)

    # When pathways_method metric is "count", "how many" refers to the metric not the operator
    if "pathways_method" in intents and metric == "count" and operator == "count":
        # Only suppress if no stronger operator (max/sort/min) would apply
        q_check = normalize_text(question)
        if re.search(r"\b(most|top|highest|max|maximum|best|ranking|mayor|maximo|m\u00e1ximo)\b", q_check):
            operator = "max"
        elif re.search(r"\b(ordena|rank|sort by|list the top|more than \d+)\b", q_check):
            operator = "sort"
        else:
            operator = None

    # For cross_method queries, "more methods" / "at least N methods" implies counting
    if "cross_method_aggregation" in intents and operator is None:
        q_check = normalize_text(question)
        if re.search(r"\b(more|at least \d+|\d+ or more|\d+ different)\b.*\b(methods?|metodos?|m\u00e9todos?|techniques?|tecnicas?|t\u00e9cnicas?)\b", q_check):
            operator = "count"

    gnn_subtype = classify_gnn_subtype(question, intents)
    network_subtype = classify_network_subtype(question, intents)
    threshold_subtype = classify_threshold_subtype(question, intents)
    model_subtype = gnn_subtype or network_subtype or threshold_subtype
    result_types_requested = detect_result_types_requested(question, intents)
    predicted_entity_types = infer_predicted_entity_types(question, intents, result_types_requested)

    # For cross_method queries, metric from individual methods is not relevant
    if "cross_method_aggregation" in intents:
        metric = None

    if "threshold_values" in intents and is_threshold_pair_metric_lookup(question, intents):
        result_types_requested = []
        predicted_entity_types = ["diseases", "drugs"]
        model_subtype = "pair_lookup"

    query_template = classify_query_template(intents, result_types_requested, operator, model_subtype)
    candidate_spans = extract_candidate_spans(question)
    pattern_targets = detect_pattern_search_targets(question)

    is_cross_method = "cross_method_aggregation" in intents

    return SemanticParse(
        intents=intents,
        metric=metric,
        operator=operator,
        result_types_requested=result_types_requested,
        predicted_entity_types=predicted_entity_types,
        query_template=query_template,
        candidate_spans=candidate_spans,
        pattern_targets=pattern_targets,
        method_subtype=model_subtype,
        is_cross_method=is_cross_method,
        question=question,
    )
