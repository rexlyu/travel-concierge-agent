"""Guardrails package exposing policy plugins and Human-in-the-Loop hooks."""

from travel_concierge.guardrails.hitl_hooks import (
    HumanApprovalGate,
    ToolContext,
    get_default_approval_gate,
    hitl_before_tool_callback,
)
from travel_concierge.guardrails.policy_plugin import (
    BasePlugin,
    LlmResponse,
    PolicyGuardrailPlugin,
    build_llm_response,
    get_default_policy_plugin,
    guardrail_after_model_callback,
)

__all__ = [
    "BasePlugin",
    "HumanApprovalGate",
    "LlmResponse",
    "PolicyGuardrailPlugin",
    "ToolContext",
    "build_llm_response",
    "get_default_approval_gate",
    "get_default_policy_plugin",
    "guardrail_after_model_callback",
    "hitl_before_tool_callback",
]
