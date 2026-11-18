"""Human-in-the-Loop (HITL) authorization gates for high-stakes travel actions.

Enforces an explicit code stop BEFORE executing any financial authorization,
lodging reservation, flight hold, non-refundable booking, or transaction above
`high_value_threshold_usd`, integrating natively with ADK `ToolContext`
(`tool_context.request_confirmation` and `tool_context.tool_confirmation`).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any

try:
    from google.adk.tools.tool_context import ToolContext

    ADK_TOOL_CONTEXT_AVAILABLE = True
except ImportError:
    ADK_TOOL_CONTEXT_AVAILABLE = False

    class ToolContext:  # type: ignore[no-redef]
        """Lightweight offline fallback for google.adk.tools.tool_context.ToolContext."""

        def __init__(
            self,
            state: dict[str, Any] | None = None,
            tool_confirmation: Any = None,
        ) -> None:
            self.state: dict[str, Any] = state if state is not None else {}
            self.tool_confirmation = tool_confirmation
            self.requested_confirmations: list[dict[str, Any]] = []

        def request_confirmation(
            self,
            hint: str | None = None,
            payload: Any = None,
        ) -> None:
            self.requested_confirmations.append(
                {"hint": hint or "", "payload": payload}
            )


_DEFAULT_HIGH_STAKES_TOOLS: frozenset[str] = frozenset(
    {
        "authorize_high_value_travel_payment",
        "reserve_lodging_booking",
        "hold_refundable_flight_reservation",
    }
)

_TRANSACTIONAL_TOOL_KEYWORDS: tuple[str, ...] = (
    "payment",
    "authorize",
    "reserve",
    "book",
    "hold",
    "purchase",
    "charge",
)


def _resolve_high_stakes_tools() -> set[str]:
    """Load HIGH_STAKES_TOOLS from travel_concierge.tools with safe fallback."""
    try:
        from travel_concierge.tools import HIGH_STAKES_TOOLS

        return set(HIGH_STAKES_TOOLS) | set(_DEFAULT_HIGH_STAKES_TOOLS)
    except Exception:
        return set(_DEFAULT_HIGH_STAKES_TOOLS)


def _emit_hitl_log(
    event_type: str,
    message: str,
    severity: str = "INFO",
    **metadata: Any,
) -> None:
    """Emit structured JSON log for HITL authorization checks and approvals."""
    try:
        from travel_concierge.observability import (
            get_structured_logger,
            log_structured_event,
        )

        logger = get_structured_logger("travel_concierge.guardrails.hitl")
        log_structured_event(
            logger,
            event_type=event_type,
            message=message,
            severity=severity,
            **metadata,
        )
    except Exception:
        fallback_logger = logging.getLogger("travel_concierge.guardrails.hitl")
        level = getattr(logging, severity.upper(), logging.INFO)
        fallback_logger.log(level, "%s: %s", event_type, message)


def _estimate_transaction_cost_usd(tool_name: str, tool_args: dict[str, Any]) -> float:
    """Calculate the effective USD cost represented by a transactional tool call."""
    if not tool_args:
        return 0.0
    if "amount_usd" in tool_args and tool_args["amount_usd"] is not None:
        try:
            return float(tool_args["amount_usd"])
        except (TypeError, ValueError):
            return 0.0
    if "total_fare_usd" in tool_args and tool_args["total_fare_usd"] is not None:
        try:
            return float(tool_args["total_fare_usd"])
        except (TypeError, ValueError):
            return 0.0
    if "nightly_rate_usd" in tool_args and tool_args["nightly_rate_usd"] is not None:
        try:
            nights = int(tool_args.get("nights", 1) or 1)
            return float(tool_args["nightly_rate_usd"]) * max(nights, 1)
        except (TypeError, ValueError):
            return 0.0

    lowered = tool_name.lower()
    if any(keyword in lowered for keyword in _TRANSACTIONAL_TOOL_KEYWORDS):
        for key in ("total_cost_usd", "price_usd", "amount"):
            if tool_args.get(key) is not None:
                try:
                    return float(tool_args[key])
                except (TypeError, ValueError):
                    pass
    return 0.0


def _compute_confirmation_id(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Compute a deterministic confirmation identifier for a tool invocation."""
    canonical_args = json.dumps(tool_args or {}, sort_keys=True, default=str)
    digest = hashlib.sha256(
        f"{tool_name}:{canonical_args}".encode("utf-8")
    ).hexdigest()[:12]
    return f"hitl-{digest}"


class HumanApprovalGate:
    """Enforces mandatory Human-in-the-Loop approval before high-stakes actions.

    Halts execution before running any tool in `HIGH_STAKES_TOOLS`, any
    non-refundable booking (`refundable is False`), or any payment/booking
    exceeding `high_value_threshold_usd`, unless explicitly approved via
    `session_state["approved_actions"]` or ADK `tool_context.tool_confirmation`.
    """

    def __init__(self, high_value_threshold_usd: float = 500.0) -> None:
        self.high_value_threshold_usd = float(high_value_threshold_usd)

    def _is_approved_in_state(
        self,
        tool_name: str,
        confirmation_id: str,
        tool_args: dict[str, Any],
        session_state: dict[str, Any] | None,
    ) -> bool:
        """Check whether the tool action or confirmation ID was approved."""
        if not session_state:
            return False
        approved = session_state.get("approved_actions")
        if not approved:
            return False

        candidate_keys = {tool_name, confirmation_id, "*", "ALL_HIGH_STAKES"}
        for id_field in ("booking_reference", "flight_id", "property_id"):
            ref_val = tool_args.get(id_field)
            if ref_val:
                candidate_keys.add(str(ref_val))
                candidate_keys.add(f"{tool_name}:{ref_val}")

        if isinstance(approved, dict):
            return any(
                key in approved and bool(approved[key]) for key in candidate_keys
            )
        if isinstance(approved, (list, set, tuple)):
            approved_set = {str(item) for item in approved}
            return bool(candidate_keys & approved_set)
        return False

    def check_tool_authorization(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        session_state: dict[str, Any] | None = None,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Verify whether `tool_name` is authorized to execute immediately.

        Args:
            tool_name: Name of the tool about to be invoked.
            tool_args: Arguments dictionary passed to the tool.
            session_state: Optional session state holding `approved_actions`.
            tool_context: Optional ADK `ToolContext` carrying `tool_confirmation`
                and `request_confirmation(...)`.

        Returns:
            Authorization dictionary. When halted, returns `allowed=False`,
            `status="PENDING_HUMAN_APPROVAL"`, `confirmation_id`, `tool_name`,
            `reason`, and `instructions`.
        """
        args = dict(tool_args or {})
        effective_state = session_state
        if effective_state is None and tool_context is not None:
            ctx_state = getattr(tool_context, "state", None)
            if isinstance(ctx_state, dict):
                effective_state = ctx_state
            elif ctx_state is not None and hasattr(ctx_state, "to_dict"):
                try:
                    effective_state = ctx_state.to_dict()
                except Exception:
                    effective_state = None

        high_stakes_tools = _resolve_high_stakes_tools()
        estimated_cost_usd = _estimate_transaction_cost_usd(tool_name, args)
        refundable = args.get("refundable", True)

        reasons: list[str] = []
        if tool_name in high_stakes_tools:
            reasons.append(f"'{tool_name}' is classified as a high-stakes financial/booking tool")
        if estimated_cost_usd > self.high_value_threshold_usd:
            reasons.append(
                f"transaction amount ${estimated_cost_usd:,.2f} USD exceeds "
                f"threshold ${self.high_value_threshold_usd:,.2f} USD"
            )
        if refundable is False:
            reasons.append("booking is marked non-refundable (refundable=False)")

        confirmation_id = _compute_confirmation_id(tool_name, args)

        if not reasons:
            return {
                "allowed": True,
                "status": "APPROVED",
                "confirmation_id": confirmation_id,
                "tool_name": tool_name,
                "estimated_cost_usd": estimated_cost_usd,
                "reason": "Standard read-only or non-transactional tool requires no human gate.",
                "instructions": "Proceed with tool execution.",
            }

        tool_confirmation = (
            getattr(tool_context, "tool_confirmation", None)
            if tool_context is not None
            else None
        )
        confirmed_via_adk = bool(
            tool_confirmation is not None
            and getattr(tool_confirmation, "confirmed", False) is True
        )
        confirmed_via_state = self._is_approved_in_state(
            tool_name=tool_name,
            confirmation_id=confirmation_id,
            tool_args=args,
            session_state=effective_state,
        )

        if confirmed_via_adk or confirmed_via_state:
            approval_source = (
                "adk_tool_confirmation" if confirmed_via_adk else "session_approved_actions"
            )
            _emit_hitl_log(
                "HITL_AUTHORIZATION_CONFIRMED",
                f"High-stakes tool '{tool_name}' authorized via {approval_source}.",
                severity="INFO",
                tool_name=tool_name,
                confirmation_id=confirmation_id,
                estimated_cost_usd=estimated_cost_usd,
                approval_source=approval_source,
            )
            return {
                "allowed": True,
                "status": "APPROVED",
                "confirmation_id": confirmation_id,
                "tool_name": tool_name,
                "estimated_cost_usd": estimated_cost_usd,
                "reason": f"Explicitly approved via {approval_source}.",
                "instructions": "Human confirmation verified; proceed with tool execution.",
            }

        combined_reason = "; ".join(reasons)
        instructions = (
            f"STOP: Execution of '{tool_name}' requires explicit human approval "
            f"({combined_reason}). Present the full cost (${estimated_cost_usd:,.2f} USD), "
            f"cancellation terms, and confirmation ID '{confirmation_id}' to the traveler "
            f"and wait for confirmation before retrying."
        )

        if tool_context is not None and hasattr(tool_context, "request_confirmation"):
            confirmation_payload = {
                "confirmation_id": confirmation_id,
                "tool_name": tool_name,
                "estimated_cost_usd": estimated_cost_usd,
                "refundable": refundable,
                "reason": combined_reason,
            }
            try:
                tool_context.request_confirmation(
                    hint=instructions,
                    payload=confirmation_payload,
                )
            except TypeError:
                tool_context.request_confirmation(instructions)

        _emit_hitl_log(
            "HITL_EXECUTION_HALTED",
            f"Halted '{tool_name}' pending human approval ({confirmation_id}).",
            severity="WARNING",
            tool_name=tool_name,
            confirmation_id=confirmation_id,
            estimated_cost_usd=estimated_cost_usd,
            reason=combined_reason,
        )
        return {
            "allowed": False,
            "status": "PENDING_HUMAN_APPROVAL",
            "confirmation_id": confirmation_id,
            "tool_name": tool_name,
            "estimated_cost_usd": estimated_cost_usd,
            "reason": combined_reason,
            "instructions": instructions,
        }

    def grant_human_approval(
        self,
        session_state: dict[str, Any],
        action_key: str,
        reviewer: str = "human_traveler",
    ) -> dict[str, Any]:
        """Record explicit human approval for a tool name or confirmation ID.

        Args:
            session_state: Mutable session state dictionary to update.
            action_key: Tool name, confirmation ID (`hitl-...`), or booking ref.
            reviewer: Identifier of the human approver.

        Returns:
            Approval receipt dictionary recorded into `session_state`.
        """
        approved_at = datetime.now(timezone.utc).isoformat()
        normalized_key = str(action_key).strip()

        existing = session_state.get("approved_actions")
        if isinstance(existing, dict):
            existing[normalized_key] = {
                "reviewer": reviewer,
                "approved_at": approved_at,
            }
            approved_list = list(existing.keys())
        elif isinstance(existing, list):
            if normalized_key not in existing:
                existing.append(normalized_key)
            approved_list = list(existing)
        elif isinstance(existing, (set, tuple)):
            merged = list(existing)
            if normalized_key not in merged:
                merged.append(normalized_key)
            session_state["approved_actions"] = merged
            approved_list = merged
        else:
            session_state["approved_actions"] = [normalized_key]
            approved_list = [normalized_key]

        audit_log = session_state.setdefault("approval_audit_log", [])
        receipt = {
            "status": "APPROVED",
            "action_key": normalized_key,
            "reviewer": reviewer,
            "approved_at": approved_at,
        }
        if isinstance(audit_log, list):
            audit_log.append(receipt)

        _emit_hitl_log(
            "HITL_APPROVAL_GRANTED",
            f"Human approval granted for '{normalized_key}' by '{reviewer}'.",
            severity="INFO",
            action_key=normalized_key,
            reviewer=reviewer,
        )
        return {
            **receipt,
            "approved_actions": approved_list,
        }


_DEFAULT_APPROVAL_GATE: HumanApprovalGate | None = None


def get_default_approval_gate() -> HumanApprovalGate:
    """Return the default HumanApprovalGate configured from TravelConciergeConfig."""
    global _DEFAULT_APPROVAL_GATE
    if _DEFAULT_APPROVAL_GATE is None:
        threshold = 500.0
        try:
            from travel_concierge.config import get_config

            threshold = float(get_config().high_value_payment_threshold_usd)
        except Exception:
            threshold = 500.0
        _DEFAULT_APPROVAL_GATE = HumanApprovalGate(
            high_value_threshold_usd=threshold
        )
    return _DEFAULT_APPROVAL_GATE


def hitl_before_tool_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
) -> dict[str, Any] | None:
    """ADK `before_tool_callback` halting high-stakes tools until human approval.

    When unauthorized, returns a structured `PENDING_HUMAN_APPROVAL` dictionary
    which instructs ADK to skip executing the underlying Python tool and return
    the confirmation prompt to the agent/caller. Returns `None` when authorized.
    """
    tool_name = (
        getattr(tool, "name", None)
        or getattr(tool, "__name__", None)
        or str(tool)
    )
    gate = get_default_approval_gate()
    verdict = gate.check_tool_authorization(
        tool_name=tool_name,
        tool_args=args or {},
        tool_context=tool_context,
    )
    if not verdict["allowed"]:
        return verdict
    return None
