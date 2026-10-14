"""Structured logging shared by the pipeline entry points.

Every run logs key/value events (``event=run_start command=train device=cpu``)
to stderr through a single ``bijli`` logger, so ``--log-level`` can dial the
verbosity up (DEBUG details) or down (WARNING only) without touching the
human-facing report that scripts print on stdout.
"""

from __future__ import annotations

import logging
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_ROOT_NAME = "bijli"


def setup_logging(level: str | int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Configure and return the ``bijli`` logger at the requested level."""
    if isinstance(level, str):
        level = level.upper()
    logger = logging.getLogger(_ROOT_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str = _ROOT_NAME) -> logging.Logger:
    """Return a child of the ``bijli`` logger (names are namespaced)."""
    if name == _ROOT_NAME or name.startswith(f"{_ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def add_log_level_argument(parser, default: str = DEFAULT_LOG_LEVEL) -> None:
    """Attach the shared ``--log-level`` option to an argparse parser."""
    parser.add_argument(
        "--log-level",
        default=default,
        choices=LOG_LEVELS,
        help="logging verbosity (default: %(default)s)",
    )
