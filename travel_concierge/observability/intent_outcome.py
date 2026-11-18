"""Intent vs. Outcome tracking and ADK observability plugin.

Captures paired pre-execution INTENT records and post-execution OUTCOME records
linked via correlation_id and OpenTelemetry trace/span context, scrubbing all PII
before persistence and emitting structured Cloud Logging JSON audit events.
"""

from __future__ import annotations

import collections
import copy
import datetime
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

from travel_concierge.observability.logging_config import (
    get_structured_logger,
    log_structured_event,
)
from travel_concierge.observability.pii_redactor import PiiRedactor, get_default_redactor
from travel_concierge.observability.tracing import get_current_trace_context

try:
    from google.adk.plugins.base_plugin import BasePlugin as _AdkBasePlugin  # type: ignore[import-not-found]

    HAS_ADK_PLUGIN = True
except ImportError:

    class _AdkBasePlugin:  # type: ignore[no-redef]
        """Lightweight ADK-compatible BasePlugin fallback for offline execution."""

        def __init__(self, name: str = "intent_outcome_plugin") -> None:
            self.name = name

    HAS_ADK_PLUGIN = False


_MAX_INTENT_OUTCOME_RECORDS = 2000
_RECORDS_LOCK = threading.Lock()
_GLOBAL_RECORDS: collections.deque[dict[str, Any]] = collections.deque(
    maxlen=_MAX_INTENT_OUTCOME_RECORDS
)


def _utc_iso_now() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_session_id(ctx: Any) -> str:
    if ctx is None:
        return ""
    for attr in ("session_id", "invocation_id"):
        val = getattr(ctx, attr, None)
        if isinstance(val, str) and val:
            return val
    session = getattr(ctx, "session", None)
    if session is not None:
        sid = getattr(session, "id", None) or getattr(session, "session_id", None)
        if isinstance(sid, str) and sid:
            return sid
    if isinstance(ctx, Mapping):
        sid = ctx.get("session_id") or ctx.get("invocation_id")
        if isinstance(sid, str) and sid:
            return sid
    return ""


def _extract_agent_name(agent: Any, ctx: Any) -> str:
    if agent is not None:
        name = getattr(agent, "name", None)
        if isinstance(name, str) and name:
            return name
    if ctx is not None:
        name = getattr(ctx, "agent_name", None)
        if isinstance(name, str) and name:
            return name
        inner_agent = getattr(ctx, "agent", None)
        if inner_agent is not None:
            inner_name = getattr(inner_agent, "name", None)
            if isinstance(inner_name, str) and inner_name:
                return inner_name
    return "travel_concierge_agent"


def _extract_tool_name(tool: Any) -> str:
    if tool is None:
        return "unknown_tool"
    for attr in ("name", "__name__"):
        val = getattr(tool, attr, None)
        if isinstance(val, str) and val:
            return val
    if isinstance(tool, str) and tool:
        return tool
    return type(tool).__name__


class IntentOutcomeTracker:
    """Records and correlates pre-execution INTENT and post-execution OUTCOME audit events."""

    def __init__(
        self,
        redactor: PiiRedactor | None = None,
        logger_name: str = "travel_concierge.intent_outcome",
    ) -> None:
        self._redactor = redactor or get_default_redactor()
        self._logger = get_structured_logger(logger_name)
        self._lock = threading.Lock()
        self._records: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_MAX_INTENT_OUTCOME_RECORDS
        )
        self._intents_by_id: dict[str, dict[str, Any]] = {}
        self._intent_start_ns: dict[str, int] = {}
        self._inflight_agents: dict[str, list[str]] = collections.defaultdict(list)
        self._inflight_tools: dict[str, list[str]] = collections.defaultdict(list)

    def record_intent(
        self,
        actor: str,
        action_name: str,
        intended_args: dict[str, Any],
        rationale: str = "",
        session_id: str = "",
    ) -> str:
        """Records pre-execution intent, scrubs PII, logs phase='INTENT', returns correlation_id."""
        correlation_id = str(uuid.uuid4())
        trace_ctx = get_current_trace_context()

        scrubbed_actor = self._redactor.redact_text(str(actor or "unknown_actor"))
        scrubbed_action = self._redactor.redact_text(str(action_name or "unknown_action"))
        raw_args = dict(intended_args) if isinstance(intended_args, Mapping) else {"args": intended_args}
        scrubbed_args = self._redactor.redact_payload(raw_args)
        scrubbed_rationale = self._redactor.redact_text(str(rationale or ""))
        scrubbed_session = (
            self._redactor.redact_text(session_id) if "@" in str(session_id) else str(session_id or "")
        )

        intent_record: dict[str, Any] = {
            "phase": "INTENT",
            "correlation_id": correlation_id,
            "timestamp": _utc_iso_now(),
            "actor": scrubbed_actor,
            "action_name": scrubbed_action,
            "intended_args": scrubbed_args,
            "rationale": scrubbed_rationale,
            "session_id": scrubbed_session,
            "trace_id": trace_ctx["trace_id"],
            "span_id": trace_ctx["span_id"],
        }

        now_perf_ns = time.perf_counter_ns()
        with self._lock:
            self._intents_by_id[correlation_id] = intent_record
            self._intent_start_ns[correlation_id] = now_perf_ns
            self._records.append(intent_record)

        with _RECORDS_LOCK:
            _GLOBAL_RECORDS.append(intent_record)

        log_structured_event(
            self._logger,
            event_type="intent_captured",
            message=f"[INTENT] {scrubbed_actor} -> {scrubbed_action}",
            severity="INFO",
            session_id=scrubbed_session,
            agent_name=scrubbed_actor,
            phase="INTENT",
            correlation_id=correlation_id,
            action_name=scrubbed_action,
            intended_args=scrubbed_args,
            rationale=scrubbed_rationale,
        )
        return correlation_id

    def record_outcome(
        self,
        correlation_id: str,
        actor: str,
        action_name: str,
        actual_result: Any,
        status: str = "SUCCESS",
        latency_ms: float = 0.0,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Pairs post-execution outcome with INTENT via correlation_id, scrubs PII, logs OUTCOME."""
        with self._lock:
            paired_intent = self._intents_by_id.get(correlation_id)
            start_ns = self._intent_start_ns.pop(correlation_id, None)

        resolved_latency = float(latency_ms)
        if resolved_latency <= 0.0 and start_ns is not None:
            resolved_latency = max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000.0)
        resolved_latency = round(resolved_latency, 4)

        trace_ctx = get_current_trace_context()
        resolved_trace_id = trace_ctx["trace_id"] or (
            paired_intent.get("trace_id") if paired_intent else None
        )
        resolved_span_id = trace_ctx["span_id"] or (
            paired_intent.get("span_id") if paired_intent else None
        )

        raw_session = session_id or (paired_intent.get("session_id", "") if paired_intent else "")
        scrubbed_session = (
            self._redactor.redact_text(str(raw_session))
            if "@" in str(raw_session)
            else str(raw_session or "")
        )
        scrubbed_actor = self._redactor.redact_text(str(actor or "unknown_actor"))
        scrubbed_action = self._redactor.redact_text(str(action_name or "unknown_action"))
        scrubbed_result = self._redactor.redact_payload(actual_result)
        normalized_status = str(status or "SUCCESS").upper()

        outcome_record: dict[str, Any] = {
            "phase": "OUTCOME",
            "correlation_id": correlation_id,
            "paired_with_intent": paired_intent is not None,
            "intent_action_name": paired_intent["action_name"] if paired_intent else None,
            "intended_args": copy.deepcopy(paired_intent["intended_args"]) if paired_intent else None,
            "timestamp": _utc_iso_now(),
            "actor": scrubbed_actor,
            "action_name": scrubbed_action,
            "actual_result": scrubbed_result,
            "status": normalized_status,
            "latency_ms": resolved_latency,
            "session_id": scrubbed_session,
            "trace_id": resolved_trace_id,
            "span_id": resolved_span_id,
        }

        with self._lock:
            self._records.append(outcome_record)

        with _RECORDS_LOCK:
            _GLOBAL_RECORDS.append(outcome_record)

        log_severity = (
            "ERROR"
            if normalized_status in {"ERROR", "FAILED", "FAILURE"}
            else ("WARNING" if normalized_status in {"BLOCKED", "PENDING_HUMAN_APPROVAL"} else "INFO")
        )
        log_structured_event(
            self._logger,
            event_type="outcome_captured",
            message=f"[OUTCOME] {scrubbed_actor} -> {scrubbed_action} ({normalized_status})",
            severity=log_severity,
            session_id=scrubbed_session,
            agent_name=scrubbed_actor,
            phase="OUTCOME",
            correlation_id=correlation_id,
            action_name=scrubbed_action,
            status=normalized_status,
            latency_ms=resolved_latency,
            paired_with_intent=paired_intent is not None,
            actual_result=scrubbed_result,
        )
        return copy.deepcopy(outcome_record)

    def get_records(self) -> list[dict[str, Any]]:
        """Returns all INTENT and OUTCOME records captured on this tracker instance."""
        with self._lock:
            return [copy.deepcopy(rec) for rec in self._records]

    def clear_records(self) -> None:
        """Clears records stored on this tracker instance."""
        with self._lock:
            self._records.clear()
            self._intents_by_id.clear()
            self._intent_start_ns.clear()
            self._inflight_agents.clear()
            self._inflight_tools.clear()

    async def before_agent_callback(self, *args: Any, **kwargs: Any) -> None:
        """ADK pre-agent hook recording agent execution INTENT."""
        agent = kwargs.get("agent")
        callback_context = kwargs.get("callback_context")
        if callback_context is None and len(args) == 1:
            callback_context = args[0]
        elif len(args) >= 2:
            agent = agent or args[0]
            callback_context = callback_context or args[1]

        actor = _extract_agent_name(agent, callback_context)
        session_id = _extract_session_id(callback_context)
        user_content = getattr(callback_context, "user_content", None)
        intended_args: dict[str, Any] = {"agent": actor}
        if user_content is not None:
            intended_args["user_content"] = str(user_content)

        cid = self.record_intent(
            actor=actor,
            action_name=f"run_agent:{actor}",
            intended_args=intended_args,
            rationale=f"Delegate turn execution to agent '{actor}'",
            session_id=session_id,
        )
        key = f"{session_id}:{actor}"
        with self._lock:
            self._inflight_agents[key].append(cid)
        return None

    async def after_agent_callback(self, *args: Any, **kwargs: Any) -> None:
        """ADK post-agent hook pairing agent completion OUTCOME with prior INTENT."""
        agent = kwargs.get("agent")
        callback_context = kwargs.get("callback_context")
        result = kwargs.get("result")
        if callback_context is None and len(args) == 1:
            callback_context = args[0]
        elif len(args) >= 2:
            agent = agent or args[0]
            callback_context = callback_context or args[1]

        actor = _extract_agent_name(agent, callback_context)
        session_id = _extract_session_id(callback_context)
        key = f"{session_id}:{actor}"
        with self._lock:
            stack = self._inflight_agents.get(key)
            cid = stack.pop() if stack else str(uuid.uuid4())

        self.record_outcome(
            correlation_id=cid,
            actor=actor,
            action_name=f"run_agent:{actor}",
            actual_result=result if result is not None else {"completed": True},
            status="SUCCESS",
            session_id=session_id,
        )
        return None

    async def before_tool_callback(self, *args: Any, **kwargs: Any) -> None:
        """ADK pre-tool hook recording tool invocation INTENT."""
        tool = kwargs.get("tool") or (args[0] if len(args) >= 1 else None)
        tool_args = kwargs.get("tool_args")
        if tool_args is None:
            tool_args = kwargs.get("args")
        if tool_args is None and len(args) >= 2:
            tool_args = args[1]
        tool_context = kwargs.get("tool_context") or (args[2] if len(args) >= 3 else None)

        tool_name = _extract_tool_name(tool)
        actor = _extract_agent_name(None, tool_context)
        session_id = _extract_session_id(tool_context)
        normalized_args = dict(tool_args) if isinstance(tool_args, Mapping) else {}

        cid = self.record_intent(
            actor=actor,
            action_name=tool_name,
            intended_args=normalized_args,
            rationale=f"Execute domain tool '{tool_name}'",
            session_id=session_id,
        )
        key = f"{session_id}:{actor}:{tool_name}"
        with self._lock:
            self._inflight_tools[key].append(cid)
        return None

    async def after_tool_callback(self, *args: Any, **kwargs: Any) -> None:
        """ADK post-tool hook pairing tool execution OUTCOME with prior INTENT."""
        tool = kwargs.get("tool") or (args[0] if len(args) >= 1 else None)
        tool_context = kwargs.get("tool_context") or (args[2] if len(args) >= 3 else None)
        if "result" in kwargs:
            tool_result = kwargs["result"]
        elif "tool_response" in kwargs:
            tool_result = kwargs["tool_response"]
        elif len(args) >= 4:
            tool_result = args[3]
        else:
            tool_result = None

        tool_name = _extract_tool_name(tool)
        actor = _extract_agent_name(None, tool_context)
        session_id = _extract_session_id(tool_context)
        key = f"{session_id}:{actor}:{tool_name}"
        with self._lock:
            stack = self._inflight_tools.get(key)
            cid = stack.pop() if stack else str(uuid.uuid4())

        status = "SUCCESS"
        if isinstance(tool_result, Mapping):
            raw_status = str(tool_result.get("status", "")).upper()
            if raw_status == "ERROR":
                status = "ERROR"
            elif raw_status == "PENDING_HUMAN_APPROVAL":
                status = "PENDING_HUMAN_APPROVAL"

        self.record_outcome(
            correlation_id=cid,
            actor=actor,
            action_name=tool_name,
            actual_result=tool_result,
            status=status,
            session_id=session_id,
        )
        return None


class IntentOutcomePlugin(_AdkBasePlugin):  # type: ignore[misc]
    """ADK App plugin capturing paired INTENT and OUTCOME events across agents and tools."""

    def __init__(
        self,
        name: str = "intent_outcome_plugin",
        tracker: IntentOutcomeTracker | None = None,
    ) -> None:
        super().__init__(name=name)
        self.tracker = tracker or get_default_intent_tracker()

    def record_intent(
        self,
        actor: str,
        action_name: str,
        intended_args: dict[str, Any],
        rationale: str = "",
        session_id: str = "",
    ) -> str:
        """Delegates pre-execution intent recording to the underlying tracker."""
        return self.tracker.record_intent(
            actor=actor,
            action_name=action_name,
            intended_args=intended_args,
            rationale=rationale,
            session_id=session_id,
        )

    def record_outcome(
        self,
        correlation_id: str,
        actor: str,
        action_name: str,
        actual_result: Any,
        status: str = "SUCCESS",
        latency_ms: float = 0.0,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Delegates post-execution outcome recording to the underlying tracker."""
        return self.tracker.record_outcome(
            correlation_id=correlation_id,
            actor=actor,
            action_name=action_name,
            actual_result=actual_result,
            status=status,
            latency_ms=latency_ms,
            session_id=session_id,
        )

    async def before_agent_callback(self, *args: Any, **kwargs: Any) -> None:
        """ADK BasePlugin before_agent_callback hook."""
        return await self.tracker.before_agent_callback(*args, **kwargs)

    async def after_agent_callback(self, *args: Any, **kwargs: Any) -> None:
        """ADK BasePlugin after_agent_callback hook."""
        return await self.tracker.after_agent_callback(*args, **kwargs)

    async def before_tool_callback(self, *args: Any, **kwargs: Any) -> None:
        """ADK BasePlugin before_tool_callback hook."""
        return await self.tracker.before_tool_callback(*args, **kwargs)

    async def after_tool_callback(self, *args: Any, **kwargs: Any) -> None:
        """ADK BasePlugin after_tool_callback hook."""
        return await self.tracker.after_tool_callback(*args, **kwargs)


_DEFAULT_TRACKER_LOCK = threading.Lock()
_DEFAULT_TRACKER: IntentOutcomeTracker | None = None


def get_default_intent_tracker() -> IntentOutcomeTracker:
    """Returns the process-wide default IntentOutcomeTracker instance."""
    global _DEFAULT_TRACKER
    if _DEFAULT_TRACKER is not None:
        return _DEFAULT_TRACKER
    with _DEFAULT_TRACKER_LOCK:
        if _DEFAULT_TRACKER is None:
            _DEFAULT_TRACKER = IntentOutcomeTracker()
    return _DEFAULT_TRACKER


def get_intent_outcome_records() -> list[dict[str, Any]]:
    """Returns a snapshot list of all captured INTENT and OUTCOME audit records."""
    with _RECORDS_LOCK:
        return [copy.deepcopy(record) for record in _GLOBAL_RECORDS]


def clear_intent_outcome_records() -> None:
    """Clears all captured INTENT and OUTCOME records from global and default trackers."""
    with _RECORDS_LOCK:
        _GLOBAL_RECORDS.clear()
    if _DEFAULT_TRACKER is not None:
        _DEFAULT_TRACKER.clear_records()
