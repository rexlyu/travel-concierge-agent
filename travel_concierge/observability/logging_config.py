"""Structured Cloud-Logging-compatible JSON logging with mandatory PII redaction.

Uses python-json-logger when installed alongside a Cloud Logging JSON formatter,
automatically injecting active OpenTelemetry trace/span correlation identifiers and
passing every log message and metadata payload through PiiRedactor via a logging.Filter.
"""

from __future__ import annotations

import collections
import copy
import datetime
import json
import logging
import os
import sys
import threading
from collections.abc import Mapping
from typing import Any

from travel_concierge.observability.pii_redactor import PiiRedactor, get_default_redactor
from travel_concierge.observability.tracing import get_current_trace_context

try:
    from pythonjsonlogger import jsonlogger  # type: ignore[import-not-found]

    _BaseJsonFormatter = jsonlogger.JsonFormatter
    HAS_PYTHON_JSON_LOGGER = True
except ImportError:
    jsonlogger = None  # type: ignore[assignment]
    _BaseJsonFormatter = logging.Formatter
    HAS_PYTHON_JSON_LOGGER = False


_MAX_CAPTURED_LOGS = 2000
_LOG_BUFFER_LOCK = threading.Lock()
_CAPTURED_JSON_LOGS: collections.deque[dict[str, Any]] = collections.deque(
    maxlen=_MAX_CAPTURED_LOGS
)

_SEVERITY_TO_LEVEL: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "NOTICE": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "ALERT": logging.CRITICAL,
    "EMERGENCY": logging.CRITICAL,
}


def _resolve_project_id() -> str:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
    if project_id:
        return project_id
    try:
        from travel_concierge.config import get_config  # type: ignore[import-not-found]

        cfg = get_config()
        cfg_project = getattr(cfg, "project_id", None)
        if cfg_project:
            return str(cfg_project)
    except Exception:
        pass
    return "travel-concierge"


class PiiRedactingFilter(logging.Filter):
    """Logging filter that scrubs PII from every message and structured metadata field."""

    def __init__(self, redactor: PiiRedactor | None = None, name: str = "") -> None:
        super().__init__(name=name)
        self._redactor = redactor or get_default_redactor()

    def filter(self, record: logging.LogRecord) -> bool:
        raw_message = record.getMessage()
        record.msg = self._redactor.redact_text(raw_message)
        record.args = ()

        raw_metadata = getattr(record, "metadata", None)
        if isinstance(raw_metadata, Mapping):
            record.metadata = self._redactor.redact_payload(dict(raw_metadata))
        elif raw_metadata is not None:
            record.metadata = {"value": self._redactor.redact_payload(raw_metadata)}
        else:
            record.metadata = {}

        for str_attr in ("event_type", "agent_name", "phase"):
            val = getattr(record, str_attr, None)
            if isinstance(val, str):
                setattr(record, str_attr, self._redactor.redact_text(val))

        session_val = getattr(record, "session_id", None)
        if isinstance(session_val, str) and "@" in session_val:
            record.session_id = self._redactor.redact_text(session_val)

        return True


class CloudLoggingJsonFormatter(_BaseJsonFormatter):  # type: ignore[misc]
    """Emits single-line JSON log records conforming to Google Cloud Logging schema."""

    def __init__(self, project_id: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._project_id = project_id

    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        severity = str(getattr(record, "cloud_severity", None) or record.levelname).upper()
        if severity == "WARN":
            severity = "WARNING"

        trace_ctx = get_current_trace_context()
        trace_id = getattr(record, "trace_id_override", None) or trace_ctx.get("trace_id")
        span_id = getattr(record, "span_id_override", None) or trace_ctx.get("span_id")

        project_id = self._project_id or _resolve_project_id()
        cloud_trace_value = f"projects/{project_id}/traces/{trace_id}" if trace_id else ""
        cloud_span_value = span_id or ""

        raw_meta = getattr(record, "metadata", None)
        metadata: dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}

        if record.exc_info:
            redactor = get_default_redactor()
            exc_text = logging.Formatter.formatException(self, record.exc_info)
            metadata["exception"] = redactor.redact_text(exc_text)

        payload: dict[str, Any] = {
            "timestamp": ts,
            "severity": severity,
            "logger": record.name,
            "message": record.getMessage(),
            "event_type": str(getattr(record, "event_type", "log") or "log"),
            "session_id": str(getattr(record, "session_id", "") or ""),
            "agent_name": str(getattr(record, "agent_name", "") or ""),
            "logging.googleapis.com/trace": cloud_trace_value,
            "logging.googleapis.com/spanId": cloud_span_value,
            "metadata": metadata,
        }

        phase = getattr(record, "phase", None) or metadata.get("phase")
        if phase is not None:
            payload["phase"] = phase

        correlation_id = getattr(record, "correlation_id", None) or metadata.get("correlation_id")
        if correlation_id is not None:
            payload["correlation_id"] = correlation_id

        return json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":"))


class _RingBufferJsonHandler(logging.Handler):
    """In-memory ring buffer handler storing parsed JSON log dicts for tests and evals."""

    def __init__(self, formatter: CloudLoggingJsonFormatter) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted_line = self.format(record)
            parsed = json.loads(formatted_line)
            if isinstance(parsed, dict):
                with _LOG_BUFFER_LOCK:
                    _CAPTURED_JSON_LOGS.append(parsed)
                record._emitted_json_dict = parsed  # type: ignore[attr-defined]
        except Exception:
            self.handleError(record)


_LOGGER_CONFIG_LOCK = threading.Lock()
_SHARED_FORMATTER = CloudLoggingJsonFormatter()
_SHARED_RING_HANDLER = _RingBufferJsonHandler(formatter=_SHARED_FORMATTER)


def _ensure_structured_logger(logger: logging.Logger) -> logging.Logger:
    if getattr(logger, "_travel_concierge_structured", False):
        return logger

    with _LOGGER_CONFIG_LOCK:
        if getattr(logger, "_travel_concierge_structured", False):
            return logger

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        has_pii_filter = any(isinstance(flt, PiiRedactingFilter) for flt in logger.filters)
        if not has_pii_filter:
            logger.addFilter(PiiRedactingFilter())

        if _SHARED_RING_HANDLER not in logger.handlers:
            logger.addHandler(_SHARED_RING_HANDLER)

        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, _RingBufferJsonHandler)
            for h in logger.handlers
        )
        if not has_stream_handler:
            stream_level_name = os.environ.get("TRAVEL_CONCIERGE_LOG_LEVEL", "WARNING").upper()
            stream_level = _SEVERITY_TO_LEVEL.get(stream_level_name, logging.WARNING)
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setLevel(stream_level)
            stream_handler.setFormatter(_SHARED_FORMATTER)
            logger.addHandler(stream_handler)

        logger._travel_concierge_structured = True  # type: ignore[attr-defined]
    return logger


def get_structured_logger(name: str = "travel_concierge") -> logging.Logger:
    """Returns a configured structured JSON logger with active PII redaction."""
    logger = logging.getLogger(name)
    return _ensure_structured_logger(logger)


def log_structured_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    severity: str = "INFO",
    **metadata: Any,
) -> dict[str, Any]:
    """Emits a PII-scrubbed structured JSON log event and returns the parsed JSON dict."""
    configured_logger = _ensure_structured_logger(logger)
    normalized_severity = str(severity or "INFO").upper()
    level_no = _SEVERITY_TO_LEVEL.get(normalized_severity, logging.INFO)

    extra_copy = dict(metadata)
    session_id = str(extra_copy.pop("session_id", "") or "")
    agent_name = str(extra_copy.pop("agent_name", "") or "")

    nested_meta = extra_copy.pop("metadata", None)
    combined_metadata: dict[str, Any] = {}
    if isinstance(nested_meta, Mapping):
        combined_metadata.update(nested_meta)
    elif nested_meta is not None:
        combined_metadata["metadata"] = nested_meta
    combined_metadata.update(extra_copy)

    phase = combined_metadata.get("phase")
    correlation_id = combined_metadata.get("correlation_id")

    record = configured_logger.makeRecord(
        name=configured_logger.name,
        level=level_no,
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None,
        extra={
            "event_type": event_type,
            "session_id": session_id,
            "agent_name": agent_name,
            "cloud_severity": "WARNING" if normalized_severity == "WARN" else normalized_severity,
            "metadata": combined_metadata,
            "phase": phase,
            "correlation_id": correlation_id,
        },
    )
    configured_logger.handle(record)

    emitted = getattr(record, "_emitted_json_dict", None)
    if isinstance(emitted, dict):
        return copy.deepcopy(emitted)

    formatted = _SHARED_FORMATTER.format(record)
    return json.loads(formatted)


def get_captured_json_logs() -> list[dict[str, Any]]:
    """Returns a snapshot list of parsed JSON log dicts from the in-memory ring buffer."""
    with _LOG_BUFFER_LOCK:
        return [copy.deepcopy(entry) for entry in _CAPTURED_JSON_LOGS]


def clear_captured_json_logs() -> None:
    """Clears all captured JSON log records from the in-memory ring buffer."""
    with _LOG_BUFFER_LOCK:
        _CAPTURED_JSON_LOGS.clear()
