from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(session_id)s | %(project_id)s | %(tool)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

LOG_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("log_session_id", default=None)
LOG_PROJECT_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("log_project_id", default=None)
LOG_TOOL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("log_tool", default=None)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = LOG_SESSION_ID.get() or "-"
        record.project_id = LOG_PROJECT_ID.get() or "-"
        record.tool = LOG_TOOL.get() or "-"
        return True


@contextmanager
def log_context(
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    tool: Optional[str] = None,
) -> Iterator[None]:
    tokens = []
    if session_id is not None:
        tokens.append((LOG_SESSION_ID, LOG_SESSION_ID.set(session_id)))
    if project_id is not None:
        tokens.append((LOG_PROJECT_ID, LOG_PROJECT_ID.set(project_id)))
    if tool is not None:
        tokens.append((LOG_TOOL, LOG_TOOL.set(tool)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def configure_logging(
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
    enable_console: bool = False,
    force: bool = False,
) -> logging.Logger:
    root = logging.getLogger()
    if getattr(root, "_univa_logging_configured", False) and not force:
        return root

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for log_filter in list(root.filters):
            root.removeFilter(log_filter)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    if enable_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    root.addFilter(ContextFilter())
    root.setLevel(level)
    logging.captureWarnings(True)
    root._univa_logging_configured = True
    return root
