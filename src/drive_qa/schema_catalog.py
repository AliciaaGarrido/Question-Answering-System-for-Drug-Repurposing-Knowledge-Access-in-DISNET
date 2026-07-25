"""
Schema catalog: table definitions and structural mappings for the DRIVE database.

This is the single source of truth for:
- Table metadata (columns, descriptions, synonyms, relationships)
- Entity-to-table mappings
- Intent-to-table mappings
- Result type bridges
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TableInfo:
    name: str
    description: str
    columns: List[str]
    synonyms: List[str] = field(default_factory=list)
    related_tables: List[str] = field(default_factory=list)


SCHEMA_CATALOG: Dict[str, TableInfo] = {
    "disease": TableInfo(
        name="disease",
        description="Diseases in DRIVE",
        columns=["disease_id", "disease_name", "ddf_type"],
        synonyms=["disease", "diseases", "enfermedad", "enfermedades", "pathology"],
        related_tables=[
            "dr_gnns",
            "dr_network_proximity",
            "dr_information_paths",
            "dr_diseasepathways",
            "dr_threshold_values",
            "dr_pathways",
            "dr_pathways_count",
        ],
    ),
    "drug": TableInfo(
        name="drug",
        description="Drugs in DRIVE",
        columns=["drug_id", "drug_name"],
        synonyms=["drug", "drugs", "fármaco", "fármacos", "farmaco", "farmacos", "medicamento", "medicamentos"],
        related_tables=[
            "dr_gnns",
            "dr_network_proximity",
            "dr_information_paths",
            "dr_diseasepathways",
            "dr_threshold_values",
            "dr_pathways",
            "dr_pathways_count",
        ],
    ),
    "gene": TableInfo(
        name="gene",
        description="Genes in DRIVE",
        columns=["gene_id", "gene_name", "gene_symbol"],
        synonyms=["gene", "genes", "gen", "genetic"],
        related_tables=["dr_threshold_values", "dr_pathways", "encodes"],
    ),
    "pathway": TableInfo(
        name="pathway",
        description="Biological pathways",
        columns=["pathway_id", "pathway_name"],
        synonyms=["pathway", "pathways", "ruta", "rutas", "vía", "vías", "via", "vias"],
        related_tables=["dr_pathways", "dr_diseasepathways"],
    ),
    "encodes": TableInfo(
        name="encodes",
        description="Relation between genes and proteins",
        columns=["gene_id", "protein_id"],
        synonyms=["encodes", "encode", "codifica", "protein", "proteína", "proteins"],
        related_tables=["gene"],
    ),
    "dr_gnns": TableInfo(
        name="dr_gnns",
        description="Drug repurposing scores from GNN-based methods",
        columns=["disease_id", "drug_id", "redirection", "dmsr", "dmsrc", "behor", "behorc"],
        synonyms=["gnn", "gnns", "redirection", "dmsr", "dmsrc", "behor", "behorc", "score"],
        related_tables=["disease", "drug"],
    ),
    "dr_network_proximity": TableInfo(
        name="dr_network_proximity",
        description="Network proximity metrics between disease and drug",
        columns=["disease_id", "drug_id", "closest_distance", "dc_mean", "dc_std", "proximity"],
        synonyms=["network proximity", "proximity", "closest distance", "distance", "distancia", "proximidad", "cercania", "dc_mean", "dc_std"],
        related_tables=["disease", "drug"],
    ),
    "dr_information_paths": TableInfo(
        name="dr_information_paths",
        description="Information paths supporting disease-drug hypotheses via biological reasoning chains",
        columns=["disease_id", "drug_id", "path_id"],
        synonyms=["information path", "information paths", "path id", "caminos de información",
                  "rutas de información", "caminos informativos", "vías de información",
                  "hipótesis por caminos", "cadenas biológicas"],
        related_tables=["disease", "drug"],
    ),
    "dr_diseasepathways": TableInfo(
        name="dr_diseasepathways",
        description="Disease similarity via shared pathways for treatment transfer between diseases. Identifies candidate diseases that share biological mechanisms and could benefit from repurposing drugs used in related diseases. disease_id_original = source disease with existing approved drugs; disease_id_new = target disease for which drugs are repurposed. To find drugs repurposable FOR a disease, filter on disease_id_new.",
        columns=["disease_id_original (source disease with known treatments)", "disease_id_new (target disease for repurposing)", "pathway_id", "drug_id"],
        synonyms=["disease pathways", "enfermedad original", "enfermedad nueva",
                  "enfermedad candidata", "enfermedades candidatas", "nueva indicación",
                  "indicación terapéutica", "transferencia de tratamiento",
                  "similaridad entre enfermedades", "reposicionamiento entre enfermedades",
                  "enfermedades similares", "mecanismos biológicos compartidos",
                  "compartir tratamientos", "reutilizar fármacos", "enfermedades relacionadas",
                  "candidatos terapéuticos", "pathways compartidos entre enfermedades"],
        related_tables=["disease", "drug", "pathway"],
    ),
    "dr_threshold_values": TableInfo(
        name="dr_threshold_values",
        description="Threshold-related disease-drug-gene values",
        columns=["disease_id", "drug_id", "gene_id", "score", "action_type"],
        synonyms=["threshold", "threshold value", "threshold values", "umbral", "action type"],
        related_tables=["disease", "drug", "gene"],
    ),
    "dr_pathways": TableInfo(
        name="dr_pathways",
        description="Disease-drug-pathway-gene scores by approach in the pathways method",
        columns=["disease_id", "drug_id", "approach", "pathway_id", "gene_id", "score"],
        synonyms=["approach", "gene pathway", "pathways method", "método de pathways",
                  "score en pathways", "puntuación en pathways", "score por pathway",
                  "gen en pathway", "gen involucrado"],
        related_tables=["disease", "drug", "gene", "pathway"],
    ),
    "dr_pathways_count": TableInfo(
        name="dr_pathways_count",
        description="Aggregated counts of disease-drug pathway associations by approach and association type",
        columns=["disease_id", "drug_id", "approach", "association_type", "count"],
        synonyms=["conteo de pathways", "número de pathways", "cuántos pathways",
                  "total de asociaciones", "association type", "tipo de asociación",
                  "conteo de asociaciones", "número de asociaciones"],
        related_tables=["disease", "drug"],
    ),
}

# =========================================================
# Entity-to-table mappings
# =========================================================

ENTITY_TABLE_CONFIG: Dict[str, Tuple[str, str, str]] = {
    "diseases": ("disease", "disease_id", "disease_name"),
    "drugs": ("drug", "drug_id", "drug_name"),
    "genes": ("gene", "gene_id", "gene_name"),
    "pathways": ("pathway", "pathway_id", "pathway_name"),
}

ENTITY_BASE_TABLES: Dict[str, List[str]] = {
    "diseases": ["disease"],
    "drugs": ["drug"],
    "genes": ["gene"],
    "pathways": ["pathway"],
    "proteins": ["encodes", "gene"],
}

RESULT_TYPE_TABLES: Dict[str, List[str]] = {
    "diseases": ["disease"],
    "drugs": ["drug"],
    "genes": ["gene"],
    "pathways": ["pathway"],
    "proteins": ["encodes", "gene"],
}

# =========================================================
# Intent-to-table mappings
# =========================================================

INTENT_TABLES: Dict[str, List[str]] = {
    "gnn_score": ["dr_gnns", "disease", "drug"],
    "network_proximity": ["dr_network_proximity", "disease", "drug"],
    "information_paths": ["dr_information_paths", "disease", "drug"],
    "threshold_values": ["dr_threshold_values", "disease", "drug", "gene"],
    "pathways_method": ["dr_pathways", "dr_pathways_count", "pathway", "disease", "drug", "gene"],
    "disease_pathways": ["dr_diseasepathways", "disease", "drug", "pathway"],
    "ranking": ["dr_gnns", "dr_network_proximity", "drug"],
    "count": ["disease", "drug"],
    "cross_method_aggregation": [
        "disease", "drug",
        "dr_gnns", "dr_network_proximity", "dr_information_paths",
        "dr_threshold_values", "dr_pathways", "dr_diseasepathways"
    ],
    "protein_encoding": ["encodes", "gene"],
}

RESULT_INTENT_BRIDGES: Dict[tuple, List[str]] = {
    ("pathways", "diseases"): ["dr_pathways", "dr_diseasepathways"],
    ("genes", "diseases"): ["dr_pathways", "dr_threshold_values"],
    ("drugs", "diseases"): [
        "dr_gnns", "dr_network_proximity", "dr_information_paths",
        "dr_diseasepathways", "dr_threshold_values", "dr_pathways", "dr_pathways_count"
    ],
    ("diseases", "drugs"): [
        "dr_gnns", "dr_network_proximity", "dr_information_paths",
        "dr_diseasepathways", "dr_threshold_values", "dr_pathways", "dr_pathways_count"
    ],
    ("proteins", "genes"): ["encodes"],
}

DOMINANT_INTENT_TO_PRIMARY_TABLE: Dict[str, str] = {
    "gnn_score": "dr_gnns",
    "network_proximity": "dr_network_proximity",
    "information_paths": "dr_information_paths",
    "threshold_values": "dr_threshold_values",
    "pathways_method": "dr_pathways",
    "disease_pathways": "dr_diseasepathways",
}
