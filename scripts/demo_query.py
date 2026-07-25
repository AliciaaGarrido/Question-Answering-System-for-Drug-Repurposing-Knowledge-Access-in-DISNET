#!/usr/bin/env python
"""
Interactive demo — run a single natural-language question through the DRIVE QA pipeline.

Usage:
    python scripts/demo_query.py "¿Qué fármacos se predicen para Alzheimer?"
    python scripts/demo_query.py "Which drugs target BRCA1?" --model gemini-2.5-flash --max-rows 20
"""

from __future__ import annotations

import argparse
from getpass import getpass
import json
import logging
import sys
from pathlib import Path
from urllib.parse import quote_plus

# Ensure src layout is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drive_qa import DriveQAPipeline, create_pipeline  # noqa: E402


DEFAULT_DB_NAME = "dr"


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


def prompt_question() -> str:
    question = input("Question: ").strip()
    if not question:
        raise SystemExit("Question is required.")
    return question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single NL question through the DRIVE QA pipeline.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural-language question to answer. If omitted, it is requested interactively.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy database URL. If omitted, username and password are requested.",
    )
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="Gemini model name.")
    parser.add_argument("--max-rows", type=int, default=50000, help="Maximum rows to display.")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    db_url = args.db_url or prompt_db_url()
    question = args.question or prompt_question()

    pipeline: DriveQAPipeline = create_pipeline(
        db_url=db_url,
        gemini_model=args.model,
        log_level=args.log_level,
    )

    print(f"\n{'─' * 60}")
    print(f"  Question: {question}")
    print(f"{'─' * 60}\n")

    result = pipeline.answer(question=question, max_answer_rows=args.max_rows)

    print("-- Retriever Output --")
    print(json.dumps(result.get("retrieved_tables", []), ensure_ascii=False, indent=2))
    print()

    print("── Generated SQL ──")
    print(result.get("sql") or "(no SQL generated)")
    print()
    

    if result.get("error"):
        print(f"⚠ Error: {result['error']}")
        sys.exit(1)

    answer = result.get("answer", "")
    if answer:
        print("── Answer ──")
        print(answer)
        print()

    rows = result.get("rows", [])
    if rows:
        print(f"── Results ({len(rows)} rows) ──")
        print(json.dumps(rows[:args.max_rows], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
