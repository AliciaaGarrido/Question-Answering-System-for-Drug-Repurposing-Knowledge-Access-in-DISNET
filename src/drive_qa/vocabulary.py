"""
Vocabulary definitions: regex patterns, metric aliases, and lookup tables
for the DRIVE QA retriever.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List


# ---------------------------------------------------------------------------
# String-enum constants for intents, metrics, and operators
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    GNN_SCORE = "gnn_score"
    NETWORK_PROXIMITY = "network_proximity"
    INFORMATION_PATHS = "information_paths"
    THRESHOLD_VALUES = "threshold_values"
    PATHWAYS_METHOD = "pathways_method"
    DISEASE_PATHWAYS = "disease_pathways"
    RANKING = "ranking"
    COUNT = "count"
    CROSS_METHOD = "cross_method_aggregation"
    PROTEIN_ENCODING = "protein_encoding"


class Metric(str, Enum):
    REDIRECTION = "redirection"
    DMSR = "dmsr"
    DMSRC = "dmsrc"
    BEHOR = "behor"
    BEHORC = "behorc"
    PROXIMITY = "proximity"
    CLOSEST_DISTANCE = "closest_distance"
    DC_MEAN = "dc_mean"
    DC_STD = "dc_std"
    SCORE = "score"
    ACTION_TYPE = "action_type"
    PATH_ID = "path_id"
    APPROACH = "approach"
    ASSOCIATION_TYPE = "association_type"
    COUNT = "count"


class Operator(str, Enum):
    MAX = "max"
    MIN = "min"
    SORT = "sort"
    COUNT = "count"

# ---------------------------------------------------------------------------
# Shared compiled regex patterns (used by semantic_parser and table_scoring)
# ---------------------------------------------------------------------------
RE_NETWORK_PROXIMITY_IMPLICIT = re.compile(
    r"\b(distancia mas corta|distancia más corta|menor distancia|closest distance|proximidad|proximity|dc_mean|dc_std)\b"
)
RE_GENE_REFERENCE = re.compile(
    r"\b(gen|genes|gene|shared target gene|gen diana compartido|gen diana compartidos?)\b"
)

RELATION_KEYWORDS = [
    "associated with", "associated",
    "related to", "related",
    "linked to", "linked",
    "asociado", "asociada", "asociados", "asociadas",
    "relacionado", "relacionada", "relacionados", "relacionadas",
    "vinculado", "vinculada", "vinculados", "vinculadas",
]

RESULT_TYPE_SYNONYMS: Dict[str, List[str]] = {
    "diseases": ["disease", "diseases", "enfermedad", "enfermedades"],
    "drugs": ["drug", "drugs", "fármaco", "fármacos", "farmaco", "farmacos", "medicamento", "medicamentos"],
    "genes": ["gene", "genes", "gen"],
    "pathways": ["pathway", "pathways", "ruta", "rutas", "vía", "vías", "via", "vias"],
    "proteins": ["protein", "proteins", "proteína", "proteínas"],
}

PATTERN_SEARCH_RULES: Dict[str, List[str]] = {
    "diseases": [
        "disease name", "diseases name", "enfermedades tienen", "enfermedad tiene",
        "en el nombre", "palabra", "contiene", "contengan", "whose name", "with the word"
    ],
    "drugs": [
        "drug name", "drugs name", "drug names", "fármaco tiene", "fármacos tienen"
    ],
    "genes": [
        "gene name", "genes name", "gene names", "gen tiene", "genes tienen"
    ],
    "pathways": [
        "pathway name", "pathways name", "pathway names", "ruta tiene", "rutas tienen"
    ],
}

INTENT_PATTERNS: Dict[str, List[str]] = {
    "gnn_score": [
        r"\bredirection\b", r"\bdmsr\b", r"\bdmsrc\b", r"\bbehor\b", r"\bbehorc\b",
        r"\bgnn\b", r"\bgnns\b", r"\bgraph neural network\b", r"\blink prediction\b",
    ],
    "network_proximity": [
        r"\bnetwork proximity\b", r"\bproximity\b", r"\bclosest distance\b",
        r"\bdc_mean\b", r"\bdc_std\b",
        r"\bproximidad en red\b", r"\bcercania en red\b", r"\bcercanía en red\b",
        r"\bdistancia media\b", r"\bmedia de distancia\b", r"\bpromedio de distancia\b",
        r"\bdesviacion estandar de la distancia\b", r"\bdesviación estándar de la distancia\b",
        r"\bdesviacion tipica de la distancia\b", r"\bdesviación típica de la distancia\b",
        r"\bvariabilidad de la distancia\b", r"\bdispersion de la distancia\b", r"\bdispersión de la distancia\b",
    ],
    "information_paths": [
        r"\binformation path\b", r"\binformation paths\b", r"\bpath id\b", r"\bpath_id\b",
        r"\bcaminos? de informacion\b", r"\bvias? de informacion\b",
        r"\brutas? de informacion\b",
        r"\bcaminos? informativos?\b", r"\brutas? informativas?\b",
        r"\binformation path method\b", r"\bmetodo de caminos\b",
        r"\bcaminos? que conectan?\b", r"\brutas? que conectan?\b",
        r"\bvias? que conectan?\b", r"\bpath que respalda\b",
        r"\bcadenas? biologicas?\b", r"\bcadenas? biológicas?\b",
        r"\bhipotesis por caminos\b", r"\bhipótesis por caminos\b",
        r"\bsoportan? la relacion\b", r"\bsoportan? la relación\b",
        r"\brespaldan? la relacion\b", r"\brespaldan? la relación\b",
    ],
    "threshold_values": [
        r"\bthreshold\b", r"\bthreshold value\b", r"\bthreshold values\b",
        r"\bthreshold-based\b", r"\bthreshold-based method\b", r"\bthreshold-based model\b",
        r"\bumbral\b", r"\bvalores umbral\b", r"\bvalor umbral\b",
        r"\bmodelo de valores umbral\b", r"\bmetodo de valores umbral\b",
        r"\bmétodo de valores umbral\b", r"\bmodelo basado en umbral\b",
        r"\bmetodo basado en umbral\b", r"\bmétodo basado en umbral\b",
    ],
    "pathways_method": [
        r"\bshared pathway\b", r"\bshared pathways\b",
        r"\bpathways method\b", r"\bvias? biologicas?\b",
        r"\bpathways? compartidos?\b", r"\bpathways? compartidas?\b",
        r"\bapproach\b", r"\bapproaches\b",
        r"\bconteo de pathways\b", r"\bnumero de pathways\b",
        r"\bnúmero de pathways\b", r"\bcuantos pathways\b",
        r"\bmetodo de pathways\b", r"\bmétodo de pathways\b",
        r"\bscore en pathways\b", r"\bpuntuacion en pathways\b",
        r"\bpuntuación en pathways\b",
        r"\bdr_pathways\b", r"\bdr_pathways_count\b",
        r"\basociaciones por pathways\b", r"\basociaciones de pathways\b",
        r"\bconteo de asociaciones\b", r"\btipo de asociacion\b",
        r"\btipo de asociación\b", r"\bassociation.type\b",
        r"\ben pathways\b", r"\ben el metodo de pathways\b",
        r"\ben el método de pathways\b",
        r"\bparticipan en.*pathways?\b", r"\binvolucrados? en.*pathways?\b",
        r"\bscore.*en pathways\b", r"\bscore.*pathways\b",
        r"\bpathways? con.*score\b", r"\bpathways? con mayor\b",
        r"\bpathways para\b", r"\bpathways de\b",
        r"\bsegun dr_pathways\b", r"\bsegún dr_pathways\b",
        # English patterns for pathway score/count
        r"\bpathway score\b", r"\bpathway.based score\b",
        r"\bpathway.based\b", r"\bpathway hypothesis\b",
        r"\bby pathway\b", r"\bvia pathway\b", r"\bvia pathways\b",
        r"\bpathway count\b", r"\bpathway counts\b",
        r"\btotal pathway\b", r"\bpathways? connect\b",
        r"\bpathways? score\b", r"\bpathways? scores\b",
        # English patterns for "N pathways" / "more pathways" / "most pathways"
        r"\bmore pathways\b", r"\bmost pathways\b",
        r"\bfewer pathways\b", r"\bless pathways\b",
        r"\b\d+ pathways\b",
        r"\bhave pathways\b", r"\bhas pathways\b",
    ],
    "disease_pathways": [
        r"\bdisease pathways\b", r"\bdisease pathway\b",
        r"\bpathways? de enfermedad\b", r"\bpathways? de enfermedades\b",
        r"\benfermedad original\b", r"\benfermedad nueva\b",
        r"\bnueva enfermedad\b", r"\bnuevas enfermedades\b",
        r"\bnew disease\b", r"\boriginal disease\b",
        r"\bdisease_id_original\b", r"\bdisease_id_new\b",
        r"\benfermedades? candidatas?\b", r"\bindicaciones? terapeuticas?\b",
        r"\bindicaciones? terapéuticas?\b",
        r"\bhipotesis? basadas? en pathways\b", r"\bhipótesis? basadas? en pathways\b",
        r"\btransferencia de tratamiento\b", r"\btransferir tratamiento\b",
        r"\bsimilaridad entre enfermedades\b", r"\bsimilitud entre enfermedades\b",
        r"\breposicionamiento entre enfermedades\b",
        r"\bdr_diseasepathways\b",
        r"\bnueva indicacion\b", r"\bnueva indicación\b",
        r"\bnuevas indicaciones\b",
        r"\benfermedades? similares?\b",
        r"\bsimilitud (con|de)\b",
        r"\bsimilaridad (con|de|entre)\b",
        r"\btransferir.*tratamiento\b", r"\btransferir.*farmaco\b", r"\btransferir.*fármaco\b",
        r"\btratamientos? transferibles?\b",
        r"\bcompartir tratamientos?\b", r"\bcompartir? tratamientos?\b",
        r"\bmecanismos? biologicos? (similares?|compartidos?)\b",
        r"\bmecanismos? biológicos? (similares?|compartidos?)\b",
        r"\bmecanismos? biologicos?\b", r"\bmecanismos? biológicos?\b",
        r"\breposicionar.*enfermedades?\b", r"\breposicionarse.*enfermedades?\b",
        r"\breposicionamiento.*a partir de\b",
        r"\bcandidatos? terapeuticos?\b", r"\bcandidatos? terapéuticos?\b",
        r"\bcandidatas? a.*reposicionamiento\b",
        r"\bdesde enfermedades?\b", r"\ba partir de enfermedades?\b",
        r"\benfermedades? (que|con) comparten?\b",
        r"\bcompartir? mecanismos?\b", r"\bcomparten? mecanismos?\b",
        r"\breutilizar.*enfermedades?\b", r"\breusar.*tratamientos?\b",
        r"\bbeneficiarse del tratamiento\b",
        r"\bpathways? (similares|comunes|compartidos|en comun|en común)\b",
        r"\bvias? biologicas? (similares|comunes|compartidas)\b",
        r"\bvías? biológicas? (similares|comunes|compartidas)\b",
        r"\benfermedades? relacionadas\b",
        r"\benfermedades? ya tratadas?\b",
        # English disease-pathway similarity patterns
        r"\bsimilar diseases?\b", r"\bdisease similarity\b",
        r"\bshares? pathways?\b", r"\bshare pathways?\b",
        r"\bshared biological\b", r"\bshared.*mechanisms?\b",
        r"\bpathway similarity\b", r"\bpathway overlap\b",
        r"\bshare treatments?\b", r"\bshared treatments?\b",
        r"\btreatment transfer\b",
        r"\brepurposed.*based on.*pathway\b",
        r"\bdiseases? that share\b", r"\bdiseases? with shared\b",
        r"\bdiseases?.*(similar|related).*pathways?\b",
        r"\bcandidate drugs?\b",
        r"\bpathways?\s+(?:are\s+)?shared\b",
        r"\bshared between\b",
        r"\bcould be repurposed\b", r"\brepurposed for\b",
        r"\bbased on.*pathway\b",
    ],
    "ranking": [
        r"\btop\b", r"\bhighest\b", r"\bmax\b", r"\bmaximum\b", r"\bbest\b",
        r"\bmayor\b", r"\bmáximo\b", r"\bmaximo\b", r"\branking\b",
        r"\bmas alto\b", r"\bpuntuacion maxima\b", r"\bvalor maximo\b",
    ],
    "count": [
        r"\bhow many\b", r"\bcount\b", r"\bcounts\b", r"\bcuantos\b", r"\bcuántos\b",
        r"\bnumero de\b", r"\bnúmero de\b",
    ],
    "cross_method_aggregation": [
        r"\ball methods\b", r"\ball techniques\b", r"\ball methodologies\b",
        r"\btodos los metodos\b", r"\btodos los métodos\b",
        r"\btodas las tecnicas\b", r"\btodas las técnicas\b",
        r"\btodas las metodologias\b", r"\btodas las metodologías\b",
        r"\btodas las tablas de hipotesis\b", r"\btodas las tablas de hipótesis\b",
        r"\ball repurposing tables\b", r"\bmulti search\b",
        r"\bpor todos los metodos\b", r"\bpor todas las tecnicas\b",
        r"\bcomparar entre metodos\b", r"\bcomparar entre métodos\b",
        r"\ben al menos \d+ metodos?\b", r"\ben al menos \d+ métodos?\b",
        r"\ben al menos \d+ modelos?\b",
        r"\bcoinciden en\b", r"\binterseccion entre\b", r"\bintersección entre\b",
        r"\ben comun entre\b", r"\ben común entre\b",
        r"\bcuantos metodos\b", r"\bcuántos métodos\b",
        r"\bcuantos modelos\b", r"\bcuántos modelos\b",
        r"\bcuantas tecnicas\b", r"\bcuántas técnicas\b",
        r"\bmas metodos\b", r"\bmás métodos\b",
        r"\bmismos metodos\b", r"\bmismos métodos\b",
        r"\btecnicas de reposicionamiento\b", r"\btécnicas de reposicionamiento\b",
        r"\bmetodos de reposicionamiento\b", r"\bmétodos de reposicionamiento\b",
        r"\bmodelos distintos\b",
        r"\bning[uú]n otro m[eé]todo\b",
        r"\bmayor cantidad de m[eé]todos\b",
        r"\bconsenso\b", r"\bconsenso multi-?metodo\b",
        r"\bmultiples metodos\b", r"\bmúltiples métodos\b",
        # English cross-method patterns
        r"\bhow many methods\b", r"\bin how many methods\b",
        r"\bin \d+ methods?\b", r"\bin \d+ different methods?\b",
        r"\bin both\b", r"\bappear in both\b", r"\bappears? in both\b",
        r"\bat least \d+ (?:different )?methods?\b",
        r"\bmultiple methods\b", r"\bdifferent methods\b",
        r"\bcomputational methods?\b", r"\brepurposing methods?\b",
        r"\bmore methods\b", r"\bmore.*methods\b",
        r"\bacross.*methods\b", r"\bin all.*methods\b",
        r"\boverlap in.*methods?\b", r"\boverlap in the same\b",
        r"\bsame repurposing methods\b", r"\bsame.*methods\b",
        r"\bwhich method\b", r"\bwhich methods\b",
    ],
    "protein_encoding": [
        r"\bprotein\b", r"\bproteins\b", r"\bproteina\b", r"\bproteínas\b", r"\bencodes\b", r"\bcodifica\b",
    ],
}

METRIC_ALIASES: Dict[str, List[str]] = {
    "redirection": [
        "redirection", "gnn redirection", "modelo redirection", "modelo gnn redirection"
    ],
    "dmsr": [
        "dmsr", "gnn dmsr", "modelo dmsr", "modelo gnn dmsr"
    ],
    "dmsrc": [
        "dmsrc", "gnn dmsrc", "modelo dmsrc", "modelo gnn dmsrc", "dmsr c", "dmsr-c"
    ],
    "behor": [
        "behor", "gnn behor", "modelo behor", "modelo gnn behor"
    ],
    "behorc": [
        "behorc", "gnn behorc", "modelo behorc", "modelo gnn behorc", "behor c", "behor-c"
    ],
    "proximity": [
        "proximity", "network proximity", "proximidad", "proximidad en red",
        "cercania en red", "cercanía en red", "que tan cerca en la red", "qué tan cerca en la red",
        "mas proximo en la red", "más próximo en la red", "mas cercano en la red", "más cercano en la red",
    ],
    "closest_distance": [
        "closest distance", "distance", "distancia", "distancia minima", "distancia mínima",
        "distancia mas corta", "distancia más corta", "menor distancia",
        "shortest distance", "nearest distance", "closest", "nearest",
        "mas cercano", "más cercano", "mas proximo", "más próximo",
    ],
    "dc_mean": [
        "dc_mean", "distancia media", "media de distancia", "promedio de distancia", "valor medio de distancia"
    ],
    "dc_std": [
        "dc_std", "desviacion estandar de la distancia", "desviación estándar de la distancia",
        "desviacion tipica de la distancia", "desviación típica de la distancia",
        "variabilidad de la distancia", "dispersion de la distancia", "dispersión de la distancia"
    ],
    "score": [
        "threshold score", "score en threshold values", "score en threshold-based model",
        "score en threshold-based method", "score en el modelo de valores umbral",
        "score en el modelo basado en umbral", "valor en threshold values",
        "valor en el modelo de valores umbral", "valor en el modelo basado en umbral",
        "score en pathways", "puntuacion en pathways", "puntuación en pathways",
        "score por pathway", "score del método de pathways", "score del metodo de pathways",
        "pathway score", "pathway-based score", "pathway scores",
    ],
    "action_type": [
        "action type", "tipo de accion", "tipo de acción", "accion", "acción",
        "inhibitor", "agonist", "antagonist", "binding", "weak inhibitor"
    ],
    "path_id": [
        "path id", "path_id", "identificador de camino", "identificador de ruta",
        "camino de informacion", "ruta de informacion",
        "caminos de informacion", "rutas de informacion",
        "caminos informativos", "rutas informativas",
    ],
    "approach": [
        "approach", "enfoque", "aproximacion", "aproximación",
        "metodo de pathways", "método de pathways",
    ],
    "association_type": [
        "association type", "tipo de asociacion", "tipo de asociación",
        "association_type", "tipos de asociacion", "tipos de asociación",
        "asociaciones tipo", "asociacion tipo",
    ],
    "count": [
        "conteo de pathways", "conteos de pathways", "pathways contados",
        "conteo de asociaciones", "conteos de associations",
        "número de asociaciones", "numero de asociaciones",
        "total de asociaciones", "mayor conteo", "conteo total",
        "conteo máximo", "conteo maximo", "mayor count",
        "pathway count", "pathway counts", "total pathway count",
        "number of pathways", "total pathways",
    ],
}
