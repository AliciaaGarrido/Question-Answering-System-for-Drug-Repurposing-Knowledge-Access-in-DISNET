"""
Logging configuration — centralised setup for the DRIVE QA system.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Optional[str] = None,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> None:
    """
    Configure logging for the entire drive_qa package.

    Priority for level resolution:
      1. Explicit `level` argument.
      2. DRIVE_QA_LOG_LEVEL environment variable.
      3. Default: INFO.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = (
        level
        or os.environ.get("DRIVE_QA_LOG_LEVEL")
        or "INFO"
    ).upper()

    numeric_level = getattr(logging, resolved_level, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root_logger = logging.getLogger("drive_qa")
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)
    root_logger.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped under the drive_qa namespace."""
    return logging.getLogger(f"drive_qa.{name}")
