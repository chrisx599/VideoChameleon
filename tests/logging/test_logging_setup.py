import logging

from univa.utils.logging_setup import (
    ContextFilter,
    LOG_FORMAT,
    LOG_PROJECT_ID,
    LOG_SESSION_ID,
    LOG_TOOL,
)


def _record():
    return logging.LogRecord("test.logger", logging.INFO, __file__, 1, "hello", (), None)


def test_context_filter_defaults():
    record = _record()
    ContextFilter().filter(record)
    assert record.session_id == "-"
    assert record.project_id == "-"
    assert record.tool == "-"


def test_context_filter_injects_values():
    session_token = LOG_SESSION_ID.set("session_1")
    project_token = LOG_PROJECT_ID.set("project_1")
    tool_token = LOG_TOOL.set("tool_1")
    try:
        record = _record()
        ContextFilter().filter(record)
        assert record.session_id == "session_1"
        assert record.project_id == "project_1"
        assert record.tool == "tool_1"
    finally:
        LOG_TOOL.reset(tool_token)
        LOG_PROJECT_ID.reset(project_token)
        LOG_SESSION_ID.reset(session_token)


def test_formatting_uses_expected_fields():
    record = _record()
    ContextFilter().filter(record)
    formatted = logging.Formatter(LOG_FORMAT).format(record)
    assert "test.logger" in formatted
    assert "hello" in formatted
    assert "|" in formatted
