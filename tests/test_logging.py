"""Tests for the shared logging setup (logging_utils.py)."""

from __future__ import annotations

import argparse
import logging

from logging_utils import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVELS,
    add_log_level_argument,
    get_logger,
    setup_logging,
)


def test_setup_logging_sets_level_and_handler():
    logger = setup_logging("WARNING")
    assert logger.name == "bijli"
    assert logger.level == logging.WARNING
    assert logger.handlers
    assert logger.propagate is False


def test_setup_logging_is_idempotent():
    setup_logging("INFO")
    setup_logging("DEBUG")
    logger = logging.getLogger("bijli")
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG


def test_get_logger_namespaces_children():
    root = get_logger()
    child = get_logger("train")
    assert root.name == "bijli"
    assert child.name == "bijli.train"
    assert get_logger("bijli.train") is child


def test_messages_reach_handler():
    logger = setup_logging("DEBUG")

    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    sink = _Capture()
    logger.addHandler(sink)
    get_logger("evaluate").info("event=run_start command=evaluate")
    assert any("event=run_start" in rec.getMessage() for rec in sink.records)


def test_add_log_level_argument_defaults_and_choices():
    parser = argparse.ArgumentParser()
    add_log_level_argument(parser)
    args = parser.parse_args([])
    assert args.log_level == DEFAULT_LOG_LEVEL
    args = parser.parse_args(["--log-level", "DEBUG"])
    assert args.log_level == "DEBUG"
    assert set(LOG_LEVELS) == {"DEBUG", "INFO", "WARNING", "ERROR"}
