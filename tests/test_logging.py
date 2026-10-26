"""日志初始化测试。"""

from __future__ import annotations

import logging

import pytest

from app.core.logging import configure_logging


def test_configure_logging_sets_level_and_single_handler() -> None:
    configure_logging("DEBUG")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("WARNING")

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING


def test_configure_logging_accepts_lowercase() -> None:
    configure_logging("error")

    assert logging.getLogger().level == logging.ERROR


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        configure_logging("LOUD")
