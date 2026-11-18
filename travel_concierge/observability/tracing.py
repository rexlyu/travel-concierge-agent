"""OpenTelemetry distributed tracing with contextvars propagation and Cloud Trace export.

Supports hierarchical span propagation across synchronous and asynchronous execution
boundaries (query -> coordinator -> parallel/sequential sub-agents -> domain tools),
working seamlessly with OpenTelemetry SDK/Cloud Trace in production and deterministic
in-memory span recording in local/CI environments.
"""

from __future__ import annotations

import collections
import contextvars
import copy
import functools
import inspect
import os
import secrets
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from travel_concierge.observability.pii_redactor import PiiRedactor, get_default_redactor

try:
    from opentelemetry import trace as otel_trace  # type: ignore[import-not-found]
    from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-not-found]

    HAS_OPENTELEMETRY = True
except ImportError:
    otel_trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    HAS_OPENTELEMETRY = False

try:
    from opentelemetry.exporter.cloud_trace import (  # type: ignore[import-not-found]
        CloudTraceSpanExporter,
    )

    HAS_CLOUD_TRACE_EXPORTER = True
except ImportError:
    CloudTraceSpanExporter = None  # type: ignore[assignment]
    HAS_CLOUD_TRACE_EXPORTER = False


F = TypeVar("F", bound=Callable[..., Any])

_MAX_EXPORTED_SPANS = 2000
_SPAN_LOCK = threading.Lock()
_EXPORTED_SPANS: collections.deque[dict[str, Any]] = collections.deque(maxlen=_MAX_EXPORTED_SPANS)

_ACTIVE_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "travel_concierge_trace_id", default=None
)
_ACTIVE_SPAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "travel_concierge_span_id", default=None
)

_PROVIDER_CONFIG_LOCK = threading.Lock()
_PROVIDER_INITIALIZED = False
_CLOUD_TRACE_ATTACHED = False


def configure_tracing(
    service_name: str = "travel-concierge-agent",
    project_id: str | None = None,
    enable_cloud_trace: bool | None = None,
) -> Any:
    """Configures OpenTelemetry TracerProvider and optional Google Cloud Trace exporter."""
    global _PROVIDER_INITIALIZED, _CLOUD_TRACE_ATTACHED

    resolved_cloud_trace = enable_cloud_trace
    resolved_project = (
        project_id
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT_ID")
    )
    if resolved_cloud_trace is None:
        resolved_cloud_trace = os.environ.get("ENABLE_CLOUD_TRACE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        try:
            from travel_concierge.config import get_config  # type: ignore[import-not-found]

            cfg = get_config()
            resolved_cloud_trace = resolved_cloud_trace or bool(
                getattr(cfg, "enable_cloud_trace", False)
            )
            resolved_project = resolved_project or getattr(cfg, "project_id", None)
        except Exception:
            pass

    if not HAS_OPENTELEMETRY or otel_trace is None or TracerProvider is None:
        return None

    with _PROVIDER_CONFIG_LOCK:
        if not _PROVIDER_INITIALIZED:
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            try:
                otel_trace.set_tracer_provider(provider)
            except Exception:
                pass
            _PROVIDER_INITIALIZED = True

        current_provider = otel_trace.get_tracer_provider()
        if (
            resolved_cloud_trace
            and not _CLOUD_TRACE_ATTACHED
            and HAS_CLOUD_TRACE_EXPORTER
            and CloudTraceSpanExporter is not None
            and BatchSpanProcessor is not None
            and hasattr(current_provider, "add_span_processor")
        ):
            try:
                exporter = (
                    CloudTraceSpanExporter(project_id=resolved_project)
                    if resolved_project
                    else CloudTraceSpanExporter()
                )
                current_provider.add_span_processor(BatchSpanProcessor(exporter))
                _CLOUD_TRACE_ATTACHED = True
            except Exception:
                pass
        return current_provider


def get_current_trace_context() -> dict[str, str | None]:
    """Returns the currently active trace_id and span_id from contextvars (or OTel context)."""
    trace_id = _ACTIVE_TRACE_ID.get()
    span_id = _ACTIVE_SPAN_ID.get()

    if (trace_id is None or span_id is None) and HAS_OPENTELEMETRY and otel_trace is not None:
        try:
            current_span = otel_trace.get_current_span()
            span_ctx = current_span.get_span_context() if current_span is not None else None
            if span_ctx is not None and getattr(span_ctx, "is_valid", False):
                trace_id = trace_id or f"{span_ctx.trace_id:032x}"
                span_id = span_id or f"{span_ctx.span_id:016x}"
        except Exception:
            pass

    return {
        "trace_id": trace_id,
        "span_id": span_id,
    }


class ActiveSpan:
    """Represents an in-flight distributed trace span with PII-safe attribute attachment."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        start_time_ns: int,
        attributes: dict[str, Any] | None = None,
        otel_span: Any = None,
        redactor: PiiRedactor | None = None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_time_ns = start_time_ns
        self.end_time_ns: int | None = None
        self.duration_ms: float | None = None
        self.status: str = "UNSET"
        self._otel_span = otel_span
        self._redactor = redactor or get_default_redactor()
        self.attributes: dict[str, Any] = {}
        if attributes:
            self.set_attributes(attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Sets a single PII-scrubbed attribute on the span."""
        scrubbed_dict = self._redactor.redact_payload({key: value})
        scrubbed_val = scrubbed_dict.get(key)
        self.attributes[key] = scrubbed_val
        if self._otel_span is not None:
            try:
                if isinstance(scrubbed_val, (bool, int, float, str)):
                    self._otel_span.set_attribute(key, scrubbed_val)
                else:
                    self._otel_span.set_attribute(key, str(scrubbed_val))
            except Exception:
                pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Sets multiple PII-scrubbed attributes on the span."""
        scrubbed_map = self._redactor.redact_payload(dict(attributes))
        for key, value in scrubbed_map.items():
            self.attributes[str(key)] = value
            if self._otel_span is not None:
                try:
                    if isinstance(value, (bool, int, float, str)):
                        self._otel_span.set_attribute(str(key), value)
                    else:
                        self._otel_span.set_attribute(str(key), str(value))
                except Exception:
                    pass

    def record_exception(self, exc: BaseException) -> None:
        """Records an exception with PII-redacted error details on the span."""
        self.status = "ERROR"
        self.set_attribute("error.type", type(exc).__name__)
        self.set_attribute("error.message", self._redactor.redact_text(str(exc)))
        if self._otel_span is not None:
            try:
                self._otel_span.record_exception(exc)
            except Exception:
                pass

    def to_dict(self) -> dict[str, Any]:
        """Serializes the completed span into an OpenTelemetry-compatible dictionary."""
        end_ns = self.end_time_ns if self.end_time_ns is not None else time.time_ns()
        duration = (
            self.duration_ms
            if self.duration_ms is not None
            else round(max(0, end_ns - self.start_time_ns) / 1_000_000.0, 4)
        )
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": end_ns,
            "duration_ms": duration,
            "attributes": copy.deepcopy(self.attributes),
            "status": self.status if self.status != "UNSET" else "OK",
        }


class _SpanActivation:
    """Single lifecycle activation of a span across sync or async context blocks."""

    def __init__(
        self,
        span_name: str,
        attributes: dict[str, Any] | None = None,
        tracer_name: str = "travel_concierge",
        redactor: PiiRedactor | None = None,
    ) -> None:
        self._span_name = span_name
        self._initial_attributes = dict(attributes) if attributes else None
        self._tracer_name = tracer_name
        self._redactor = redactor or get_default_redactor()
        self._otel_cm: Any = None
        self._trace_token: contextvars.Token[str | None] | None = None
        self._span_token: contextvars.Token[str | None] | None = None
        self._active_span: ActiveSpan | None = None
        self._start_perf_ns: int = 0

    def start(self) -> ActiveSpan:
        parent_ctx = get_current_trace_context()
        parent_trace_id = parent_ctx["trace_id"]
        parent_span_id = parent_ctx["span_id"]

        otel_span = None
        trace_id: str | None = None
        span_id: str | None = None

        if HAS_OPENTELEMETRY and otel_trace is not None:
            configure_tracing()
            try:
                otel_tracer = otel_trace.get_tracer(self._tracer_name)
                self._otel_cm = otel_tracer.start_as_current_span(self._span_name)
                otel_span = self._otel_cm.__enter__()
                span_ctx = otel_span.get_span_context()
                if span_ctx is not None and getattr(span_ctx, "is_valid", False):
                    trace_id = parent_trace_id or f"{span_ctx.trace_id:032x}"
                    span_id = f"{span_ctx.span_id:016x}"
            except Exception:
                self._otel_cm = None
                otel_span = None

        resolved_trace_id = trace_id or parent_trace_id or secrets.token_hex(16)
        resolved_span_id = span_id or secrets.token_hex(8)

        start_wall_ns = time.time_ns()
        self._start_perf_ns = time.perf_counter_ns()

        self._trace_token = _ACTIVE_TRACE_ID.set(resolved_trace_id)
        self._span_token = _ACTIVE_SPAN_ID.set(resolved_span_id)

        self._active_span = ActiveSpan(
            name=self._span_name,
            trace_id=resolved_trace_id,
            span_id=resolved_span_id,
            parent_span_id=parent_span_id,
            start_time_ns=start_wall_ns,
            attributes=self._initial_attributes,
            otel_span=otel_span,
            redactor=self._redactor,
        )
        return self._active_span

    def finish(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        elapsed_ns = max(0, time.perf_counter_ns() - self._start_perf_ns)
        end_wall_ns = (
            self._active_span.start_time_ns + elapsed_ns
            if self._active_span is not None
            else time.time_ns()
        )

        if self._active_span is not None:
            self._active_span.end_time_ns = end_wall_ns
            self._active_span.duration_ms = round(elapsed_ns / 1_000_000.0, 4)
            if exc_val is not None:
                self._active_span.record_exception(exc_val)
            elif self._active_span.status == "UNSET":
                self._active_span.status = "OK"

            exported = self._active_span.to_dict()
            with _SPAN_LOCK:
                _EXPORTED_SPANS.append(exported)

        if self._span_token is not None:
            try:
                _ACTIVE_SPAN_ID.reset(self._span_token)
            except ValueError:
                _ACTIVE_SPAN_ID.set(None)
            self._span_token = None

        if self._trace_token is not None:
            try:
                _ACTIVE_TRACE_ID.reset(self._trace_token)
            except ValueError:
                _ACTIVE_TRACE_ID.set(None)
            self._trace_token = None

        if self._otel_cm is not None:
            try:
                self._otel_cm.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
            self._otel_cm = None


class _TraceOperation:
    """Dual sync/async context manager AND sync/async function decorator for tracing."""

    def __init__(
        self,
        span_name: str,
        attributes: dict[str, Any] | None = None,
        tracer_name: str = "travel_concierge",
    ) -> None:
        self.span_name = span_name
        self.attributes = dict(attributes) if attributes else None
        self.tracer_name = tracer_name
        self._activation_stack: list[_SpanActivation] = []

    def __enter__(self) -> ActiveSpan:
        activation = _SpanActivation(
            span_name=self.span_name,
            attributes=self.attributes,
            tracer_name=self.tracer_name,
        )
        self._activation_stack.append(activation)
        return activation.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if self._activation_stack:
            activation = self._activation_stack.pop()
            activation.finish(exc_type, exc_val, exc_tb)
        return False

    async def __aenter__(self) -> ActiveSpan:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                activation = _SpanActivation(
                    span_name=self.span_name,
                    attributes=self.attributes,
                    tracer_name=self.tracer_name,
                )
                activation.start()
                try:
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    activation.finish(type(exc), exc, exc.__traceback__)
                    raise
                else:
                    activation.finish(None, None, None)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            activation = _SpanActivation(
                span_name=self.span_name,
                attributes=self.attributes,
                tracer_name=self.tracer_name,
            )
            activation.start()
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                activation.finish(type(exc), exc, exc.__traceback__)
                raise
            else:
                activation.finish(None, None, None)
                return result

        return sync_wrapper  # type: ignore[return-value]


class TravelConciergeTracer:
    """OpenTelemetry-compatible tracer propagating contextvars and recording spans."""

    def __init__(self, name: str = "travel_concierge") -> None:
        self.name = name
        configure_tracing()

    def start_as_current_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> _TraceOperation:
        """Starts a span as the active context manager (sync or async)."""
        return _TraceOperation(
            span_name=name,
            attributes=attributes,
            tracer_name=self.name,
        )


def get_tracer(name: str = "travel_concierge") -> TravelConciergeTracer:
    """Returns an OpenTelemetry-backed TravelConciergeTracer instance."""
    return TravelConciergeTracer(name=name)


def trace_operation(
    span_name: str,
    attributes: dict[str, Any] | None = None,
) -> _TraceOperation:
    """Traces a block (`with` / `async with`) or function (`@trace_operation(...)`)."""
    return _TraceOperation(span_name=span_name, attributes=attributes)


def get_exported_spans() -> list[dict[str, Any]]:
    """Returns a snapshot copy of all recorded spans in export order."""
    with _SPAN_LOCK:
        return [copy.deepcopy(span) for span in _EXPORTED_SPANS]


def clear_exported_spans() -> None:
    """Clears the in-memory exported spans buffer."""
    with _SPAN_LOCK:
        _EXPORTED_SPANS.clear()
