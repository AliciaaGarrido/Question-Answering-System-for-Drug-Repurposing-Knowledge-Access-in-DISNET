"""
Text normalization utilities for the DRIVE QA retriever.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

STOPWORDS = {
    "s", "of", "the", "and", "a", "an", "in", "on", "for", "to", "from", "with",
    "de", "del", "la", "las", "el", "los", "y", "al", "por", "para", "con", "sin",
    "what", "which", "who", "whose", "cuál", "cual", "qué", "que", "cuáles", "cuales",
    "según", "segun", "is", "are", "was", "were", "this", "that", "those",
    "en", "sobre", "entre", "all",
}

ENTITY_QUERY_STOPWORDS = STOPWORDS | {
    "drug", "drugs", "fármaco", "fármacos", "farmaco", "farmacos", "medicamento", "medicamentos",
    "disease", "diseases", "enfermedad", "enfermedades",
    "gene", "genes", "gen",
    "pathway", "pathways", "ruta", "rutas", "vía", "vías", "via", "vias",
    "score", "metric", "métrica", "metrica", "valor", "puntuacion", "puntuación",
    "highest", "lowest", "top", "maximum", "minimum", "best", "worst",
    "mayor", "menor", "máximo", "maximo", "mínimo", "minimo",
    "associated", "association", "asociado", "asociada", "asociados", "asociadas",
    "related", "linked", "relacionado", "relacionada", "vinculado", "vinculada",
    "closest", "nearest", "cercano", "cercana", "cercanos", "cercanas", "próximo", "proximo",
    "count", "counts", "cuántas", "cuantos", "número", "numero",
    "name", "nombre", "word", "palabra",
    "have", "has", "tienen", "tiene", "according", "show", "list", "listar", "mostrar",
    "modelo", "method", "methods", "metodo", "método", "tecnica", "técnica", "tecnicas", "técnicas",
}


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_text(text: str) -> str:
    text = strip_accents(text.lower().strip())
    text = re.sub(r"[^\w\s\-\/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    tokens = normalize_text(text).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def sort_dict_desc(d: Dict[str, int]) -> Dict[str, int]:
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
