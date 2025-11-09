from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from src.core import logging_setup


def test_json_formatter_includes_extra_fields() -> None:
    formatter = logging_setup.JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.tool = "cli"
    payload = json.loads(formatter.format(record))

    assert payload["message"] == "hello"
    assert payload["tool"] == "cli"


def test_configure_logging_sets_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = StringIO()

    class StubStreamHandler(logging.StreamHandler):
        def __init__(self) -> None:
            super().__init__(stream=stream)

    monkeypatch.setattr(logging_setup, "_configured", False, raising=False)
    monkeypatch.setattr(logging_setup, "_handler", None, raising=False)
    monkeypatch.setattr(logging, "StreamHandler", StubStreamHandler)

    logging_setup.configure_logging({"level": "DEBUG"})

    logger = logging.getLogger("sample")
    logger.debug("test", extra={"tool": "unit"})

    output = stream.getvalue().strip()
    assert output
    payload = json.loads(output)
    assert payload["tool"] == "unit"
    assert logging_setup._handler is not None


def test_set_runtime_level_updates_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream=stream)

    monkeypatch.setattr(logging_setup, "_handler", handler, raising=False)
    logger = logging.getLogger()
    logger.handlers = [handler]

    logging_setup.set_runtime_level("WARNING")
    assert handler.level == logging.WARNING

    with pytest.raises(ValueError):
        logging_setup.set_runtime_level("not-a-level")
