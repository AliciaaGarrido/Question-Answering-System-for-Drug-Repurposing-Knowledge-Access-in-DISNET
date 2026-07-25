#!/usr/bin/env python
"""
Run the retriever evaluator on all DRIVE QA evaluation banks.

Usage:
    python scripts/evaluation/run_eval.py --preset all
    python scripts/evaluation/run_eval.py --datasets gnns network_proximity
    python scripts/evaluation/run_eval.py --preset bilingual
"""

from __future__ import annotations

import argparse
from getpass import getpass
import logging
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote_plus

# Ensure src layout is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
EVALUATION_BANKS_DIR = PROJECT_ROOT / "data" / "evaluation_banks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation_results"
EVALUATOR_SCRIPT = Path(__file__).with_name("retriever_evaluator_v2.py")
sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = "dr"
DEFAULT_RETRIEVER_MODULE = "drive_qa.retriever"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))


def prompt_db_url() -> str:
    db_host = input("Database host: ").strip()
    if not db_host:
        raise SystemExit("Database host is required.")

    db_port_str = input("Database port: ").strip()
    if not db_port_str:
        raise SystemExit("Database port is required.")
    try:
        db_port = int(db_port_str)
        if not (1 <= db_port <= 65535):
            raise ValueError
    except ValueError:
        raise SystemExit("Database port must be a number between 1 and 65535.")

    db_user = input("Database username: ").strip()
    if not db_user:
        raise SystemExit("Database username is required.")

    db_password = getpass("Database password: ")
    if not db_password:
        raise SystemExit("Database password is required.")

    return (
        "mysql+pymysql://"
        f"{quote_plus(db_user)}:{quote_plus(db_password)}"
        f"@{db_host}:{db_port}/{DEFAULT_DB_NAME}"
    )

# All available datasets: (name, spanish_bank_path, english_bank_path)
DATASETS = [
    ("gnns", EVALUATION_BANKS_DIR / "gnns" / "dr_gnns_eval_bank_v2.json", EVALUATION_BANKS_DIR / "gnns" / "dr_gnns_eval_bank_en.json"),
    ("network_proximity", EVALUATION_BANKS_DIR / "network_proximity" / "dr_network_proximity_eval_bank.json", EVALUATION_BANKS_DIR / "network_proximity" / "dr_network_proximity_eval_bank_en.json"),
    ("threshold_values", EVALUATION_BANKS_DIR / "threshold_values" / "dr_threshold_values_eval_bank.json", EVALUATION_BANKS_DIR / "threshold_values" / "dr_threshold_values_eval_bank_en.json"),
    ("information_paths", EVALUATION_BANKS_DIR / "information_paths" / "dr_information_paths_eval_bank.json", EVALUATION_BANKS_DIR / "information_paths" / "dr_information_paths_eval_bank_en.json"),
    ("diseasepathways", EVALUATION_BANKS_DIR / "diseasepathways" / "dr_diseasepathways_eval_bank.json", EVALUATION_BANKS_DIR / "diseasepathways" / "dr_diseasepathways_eval_bank_en.json"),
    ("pathways", EVALUATION_BANKS_DIR / "pathways" / "dr_pathways_eval_bank.json", EVALUATION_BANKS_DIR / "pathways" / "dr_pathways_eval_bank_en.json"),
    ("pathways_count", EVALUATION_BANKS_DIR / "pathways_count" / "dr_pathways_count_eval_bank.json", EVALUATION_BANKS_DIR / "pathways_count" / "dr_pathways_count_eval_bank_en.json"),
    ("cross_method", EVALUATION_BANKS_DIR / "cross_method" / "dr_cross_method_eval_bank.json", EVALUATION_BANKS_DIR / "cross_method" / "dr_cross_method_eval_bank_en.json"),
]

DATASET_NAMES = [d[0] for d in DATASETS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retriever evaluation across DRIVE QA datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        help="Specific datasets to evaluate (default: all).",
    )
    parser.add_argument(
        "--preset",
        choices=["all", "bilingual", "spanish", "english"],
        default="bilingual",
        help="Preset: 'all'/'bilingual' = both languages, 'spanish' or 'english' only.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy database URL. If omitted, username and password are requested.",
    )
    parser.add_argument("--retriever-module", default=DEFAULT_RETRIEVER_MODULE, help="Retriever module name.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=int, default=2)
    parser.add_argument("--expand-relations", action="store_true", default=True)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for result JSON files.",
    )
    parser.add_argument("--version-tag", default="v11", help="Version tag appended to output filenames.")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Path to Python interpreter.")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def run_eval(
    *,
    python: str,
    dataset_path: Path,
    db_url: str,
    retriever_module: str,
    top_k: int,
    min_score: int,
    expand_relations: bool,
    output: str,
    log_level: str,
) -> bool:
    """Run a single evaluation and return True on success."""
    cmd = [
        python,
        str(EVALUATOR_SCRIPT),
        "--dataset", str(dataset_path),
        "--db-url", db_url,
        "--retriever-module", retriever_module,
        "--top-k", str(top_k),
        "--min-score", str(min_score),
        "--output", output,
        "--log-level", log_level,
    ]
    if expand_relations:
        cmd.append("--expand-relations")

    log_cmd = ["<db-url hidden>" if part == db_url else part for part in cmd]
    logger.info("Running: %s", " ".join(log_cmd))
    try:
        display_dataset = dataset_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_dataset = dataset_path
    print(f"  → {display_dataset} → {output}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ✗ FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"      {result.stderr[:500]}")
        return False
    # Print summary line from stdout
    if result.stdout.strip():
        lines = result.stdout.strip().split("\n")
        # Show just the first few summary lines
        for line in lines[:5]:
            print(f"    {line}")
    return True


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = DATASETS
    if args.datasets:
        selected_datasets = [d for d in DATASETS if d[0] in args.datasets]

    db_url = args.db_url or prompt_db_url()

    run_spanish = args.preset in ("all", "bilingual", "spanish")
    run_english = args.preset in ("all", "bilingual", "english")

    total = 0
    success = 0

    if run_spanish:
        print("\n===== SPANISH BANKS =====\n")
        for name, es_path, _ in selected_datasets:
            output_file = str(output_dir / f"dr_{name}_eval_results_{args.version_tag}.json")
            total += 1
            if run_eval(
                python=args.python,
                dataset_path=es_path,
                db_url=db_url,
                retriever_module=args.retriever_module,
                top_k=args.top_k,
                min_score=args.min_score,
                expand_relations=args.expand_relations,
                output=output_file,
                log_level=args.log_level,
            ):
                success += 1

    if run_english:
        print("\n===== ENGLISH BANKS =====\n")
        for name, _, en_path in selected_datasets:
            output_file = str(output_dir / f"dr_{name}_eval_results_en_{args.version_tag}.json")
            total += 1
            if run_eval(
                python=args.python,
                dataset_path=en_path,
                db_url=db_url,
                retriever_module=args.retriever_module,
                top_k=args.top_k,
                min_score=args.min_score,
                expand_relations=args.expand_relations,
                output=output_file,
                log_level=args.log_level,
            ):
                success += 1

    print(f"\n===== DONE: {success}/{total} evaluations succeeded =====")
    sys.exit(0 if success == total else 1)


if __name__ == "__main__":
    main()
