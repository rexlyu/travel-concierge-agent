"""Guardrails and policy enforcement plugin for the Travel Concierge Agent.

Implements:
- Input screening against prompt injection, system-prompt override, credential
  exfiltration, and unsafe/illegal travel requests.
- Optional Google Cloud Model Armor (`google.cloud.modelarmor_v1`) sanitization
  for enterprise prompt and response defense-in-depth.
- Output self-evaluation critic verifying proposed costs never silently exceed
  the traveler's `budget_limit_usd` and scrubbing residual PII via `PiiRedactor`.
- ADK `BasePlugin` hooks (`before_model_callback`, `after_model_callback`).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

try:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins.base_plugin import BasePlugin
    from google.genai import types as genai_types

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    CallbackContext = Any  # type: ignore[assignment,misc]
    LlmRequest = Any  # type: ignore[assignment,misc]
    genai_types = None  # type: ignore[assignment]

    class _FallbackPart:
        """Lightweight offline fallback for google.genai.types.Part."""

        def __init__(self, text: str = "") -> None:
            self.text = text

    class _FallbackContent:
        """Lightweight offline fallback for google.genai.types.Content."""

        def __init__(
            self,
            role: str = "model",
            parts: list[Any] | None = None,
        ) -> None:
            self.role = role
            self.parts = parts if parts is not None else []

    class LlmResponse:  # type: ignore[no-redef]
        """Lightweight offline fallback for google.adk.models.llm_response.LlmResponse."""

        def __init__(
            self,
            content: Any = None,
            text: str | None = None,
            error_code: str | None = None,
            error_message: str | None = None,
            custom_metadata: dict[str, Any] | None = None,
        ) -> None:
            if content is None and text is not None:
                content = _FallbackContent(
                    role="model",
                    parts=[_FallbackPart(text=text)],
                )
            self.content = content
            self.error_code = error_code
            self.error_message = error_message
            self.custom_metadata = custom_metadata or {}

        @property
        def text(self) -> str:
            if self.content is not None and getattr(self.content, "parts", None):
                return "".join(
                    getattr(part, "text", "") or "" for part in self.content.parts
                )
            return self.error_message or ""

    class BasePlugin:  # type: ignore[no-redef]
        """Lightweight offline fallback for google.adk.plugins.base_plugin.BasePlugin."""

        def __init__(self, name: str = "base_plugin") -> None:
            self.name = name


try:
    from google.cloud import modelarmor_v1  # type: ignore[attr-defined]

    MODEL_ARMOR_AVAILABLE = True
except ImportError:
    modelarmor_v1 = None  # type: ignore[assignment]
    MODEL_ARMOR_AVAILABLE = False


_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(ignore|disregard|forget|drop)\s+(all\s+|any\s+|every\s+)?"
        r"(previous|prior|above|earlier|system|initial)\s+"
        r"(instructions?|prompts?|rules?|directives?|constraints?|guidelines?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(reveal|dump|show|display|output|expose|repeat|leak|extract)\s+"
        r"(the\s+|your\s+|all\s+)?(system\s+prompt|hidden\s+prompt|constitution|"
        r"developer\s+prompt|internal\s+instructions?|initial\s+instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(bypass|override|disable|skip|circumvent|turn\s+off)\s+"
        r"(all\s+|the\s+|human\s+|hitl\s+|payment\s+|budget\s+|safety\s+|policy\s+|security\s+)?"
        r"(approvals?|guardrails?|gates?|checks?|confirmations?|verifications?|restrictions?|filters?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\byou\s+are\s+now\s+(in\s+)?(developer\s+mode|dan\b|unrestricted\s+mode|jailbreak)\b|"
        r"<\s*(system|admin_override|developer_override)\s*>|"
        r"\[\s*system\s+override\s*\]",
        re.IGNORECASE,
    ),
)

_CREDENTIAL_EXFILTRATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(reveal|dump|extract|exfiltrate|show|output|expose|read|give\s+me|list)\b"
        r".{0,60}\b(api[ _-]?keys?|secrets?|tokens?|passwords?|credentials?|"
        r"private[ _-]?keys?|service[ _-]?account|environment\s+variables?|\.env)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(dump|extract|exfiltrate|read\s+all)\b.{0,40}\bsecret[ _-]?manager\b",
        re.IGNORECASE,
    ),
)

_UNSAFE_TRAVEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(smuggl(?:e|ing)|human\s+traffick(?:ing)?|contraband|"
        r"forged\s+passport|fake\s+passport|counterfeit\s+visa|stolen\s+passport|"
        r"illegal\s+border\s+crossing|evade\s+(?:customs|border|immigration)|"
        r"bribe\s+(?:customs|border|immigration|officials?)|"
        r"sanctions\s+evasion|money\s+laundering|"
        r"sneak\s+(?:weapons?|explosives?)\s+(?:onto|on)\s+(?:a\s+)?(?:flight|plane))\b",
        re.IGNORECASE,
    ),
)

_TOTAL_COST_EXTRACT_RE = re.compile(
    r"(?:total\s+(?:trip\s+|estimated\s+|proposed\s+)?cost|grand\s+total|proposed\s+total)"
    r"\s*(?:is|of|:|=)?\s*\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)

_FALLBACK_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_FALLBACK_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_FALLBACK_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_FALLBACK_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)


def _emit_policy_log(
    event_type: str,
    message: str,
    severity: str = "INFO",
    **metadata: Any,
) -> None:
    """Emit structured JSON observability event with safe stdlib fallback."""
    try:
        from travel_concierge.observability import (
            get_structured_logger,
            log_structured_event,
        )

        logger = get_structured_logger("travel_concierge.guardrails.policy")
        log_structured_event(
            logger,
            event_type=event_type,
            message=message,
            severity=severity,
            **metadata,
        )
    except Exception:
        fallback_logger = logging.getLogger("travel_concierge.guardrails.policy")
        level = getattr(logging, severity.upper(), logging.INFO)
        fallback_logger.log(level, "%s: %s", event_type, message)


def _scrub_text_with_redactor(text: str, custom_redactor: Any = None) -> str:
    """Scrub PII from text via PiiRedactor, falling back to local regex rules."""
    if custom_redactor is not None and hasattr(custom_redactor, "redact_text"):
        return str(custom_redactor.redact_text(text))
    try:
        from travel_concierge.observability import get_default_redactor

        return str(get_default_redactor().redact_text(text))
    except Exception:
        scrubbed = _FALLBACK_EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        scrubbed = _FALLBACK_SSN_RE.sub("[REDACTED_SSN]", scrubbed)
        scrubbed = _FALLBACK_CARD_RE.sub("[REDACTED_CARD]", scrubbed)
        scrubbed = _FALLBACK_PHONE_RE.sub("[REDACTED_PHONE]", scrubbed)
        return scrubbed


def build_llm_response(
    text: str,
    error_code: str | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> LlmResponse:
    """Construct an ADK-compatible LlmResponse for callback short-circuiting."""
    if ADK_AVAILABLE and genai_types is not None:
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=text)],
        )
        return LlmResponse(
            content=content,
            error_code=error_code,
            error_message=text if error_code else None,
            custom_metadata=custom_metadata,
        )
    return LlmResponse(
        text=text,
        error_code=error_code,
        error_message=text if error_code else None,
        custom_metadata=custom_metadata,
    )


def extract_state_from_context(context: Any) -> dict[str, Any]:
    """Extract session state dictionary safely from an ADK context or mock."""
    if context is None:
        return {}
    if isinstance(context, dict):
        return context
    for attr in ("state", "session_state"):
        state_obj = getattr(context, attr, None)
        if isinstance(state_obj, dict):
            return state_obj
        if state_obj is not None and hasattr(state_obj, "to_dict"):
            try:
                converted = state_obj.to_dict()
                if isinstance(converted, dict):
                    return converted
            except Exception:
                pass
    return {}


def extract_user_text_from_request(llm_request: Any) -> str:
    """Extract the latest user prompt text from an ADK LlmRequest or test stub."""
    if llm_request is None:
        return ""
    if isinstance(llm_request, str):
        return llm_request
    if isinstance(llm_request, dict):
        for key in ("query", "prompt", "text", "user_text"):
            if isinstance(llm_request.get(key), str):
                return str(llm_request[key])
        contents = llm_request.get("contents") or llm_request.get("messages") or []
    else:
        for attr in ("prompt", "text", "user_text", "query"):
            val = getattr(llm_request, attr, None)
            if isinstance(val, str) and val:
                return val
        contents = (
            getattr(llm_request, "contents", None)
            or getattr(llm_request, "messages", None)
            or []
        )

    extracted_segments: list[str] = []
    for item in contents:
        if isinstance(item, str):
            extracted_segments.append(item)
        elif isinstance(item, dict):
            role = str(item.get("role", "user")).lower()
            if role in ("user", "human"):
                if isinstance(item.get("content"), str):
                    extracted_segments.append(item["content"])
                for part in item.get("parts", []):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        extracted_segments.append(part["text"])
                    elif isinstance(part, str):
                        extracted_segments.append(part)
        else:
            role = str(getattr(item, "role", "user") or "user").lower()
            if role in ("user", "human"):
                parts = getattr(item, "parts", None) or []
                part_texts = [
                    getattr(p, "text", "")
                    for p in parts
                    if getattr(p, "text", None)
                ]
                if part_texts:
                    extracted_segments.append(" ".join(part_texts))
                elif isinstance(getattr(item, "text", None), str):
                    extracted_segments.append(item.text)
    return extracted_segments[-1] if extracted_segments else ""


def extract_text_from_response(llm_response: Any) -> str:
    """Extract text content from an ADK LlmResponse or test stub."""
    if llm_response is None:
        return ""
    if isinstance(llm_response, str):
        return llm_response
    if isinstance(llm_response, dict):
        if isinstance(llm_response.get("text"), str):
            return str(llm_response["text"])
        content = llm_response.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return "".join(
                str(p.get("text", ""))
                for p in content.get("parts", [])
                if isinstance(p, dict)
            )
        return ""

    content = getattr(llm_response, "content", None)
    if content is not None and getattr(content, "parts", None):
        texts = [
            getattr(part, "text", "") or ""
            for part in content.parts
            if getattr(part, "text", None) is not None
        ]
        if texts:
            return "".join(texts)
    direct_text = getattr(llm_response, "text", None)
    if isinstance(direct_text, str):
        return direct_text
    return ""


class PolicyGuardrailPlugin(BasePlugin):
    """ADK policy guardrail plugin enforcing input safety and output self-critique.

    Provides three layers of defense:
    1. Deterministic input screening for prompt injection, system-prompt override,
       credential exfiltration, and unsafe/illegal travel requests.
    2. Optional Google Cloud Model Armor (`modelarmor_v1`) API integration for
       managed prompt-injection and responsible-AI filtering in production.
    3. Output self-evaluation critic verifying that synthesized trip costs never
       silently exceed `budget_limit_usd` and scrubbing residual PII via
       `get_default_redactor()`.
    """

    def __init__(
        self,
        name: str = "policy_guardrail_plugin",
        enable_model_armor: bool | None = None,
        model_armor_template_name: str | None = None,
        default_budget_limit_usd: float | None = None,
        redactor: Any = None,
        model_armor_client: Any = None,
    ) -> None:
        super().__init__(name=name)
        env_flag = os.environ.get("ENABLE_MODEL_ARMOR", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self.enable_model_armor = (
            enable_model_armor if enable_model_armor is not None else env_flag
        )
        self.model_armor_template_name = (
            model_armor_template_name
            or os.environ.get("MODEL_ARMOR_TEMPLATE_NAME")
        )
        self.default_budget_limit_usd = default_budget_limit_usd
        self._redactor = redactor
        self._model_armor_client = model_armor_client

    def _get_model_armor_client(self) -> Any:
        """Lazily initialize the Google Cloud Model Armor client when enabled."""
        if self._model_armor_client is not None:
            return self._model_armor_client
        if not self.enable_model_armor or not MODEL_ARMOR_AVAILABLE:
            return None
        try:
            self._model_armor_client = modelarmor_v1.ModelArmorClient()
            return self._model_armor_client
        except Exception as exc:
            _emit_policy_log(
                "MODEL_ARMOR_INIT_WARNING",
                f"Model Armor client unavailable, continuing with local rules: {exc}",
                severity="WARNING",
            )
            return None

    def _screen_with_model_armor(
        self,
        text: str,
        is_output: bool = False,
    ) -> dict[str, Any] | None:
        """Invoke Google Cloud Model Armor template sanitization when configured."""
        client = self._get_model_armor_client()
        if client is None or not self.model_armor_template_name:
            return None
        try:
            if MODEL_ARMOR_AVAILABLE and modelarmor_v1 is not None:
                data_item = modelarmor_v1.DataItem(text=text)
                if is_output:
                    request = modelarmor_v1.SanitizeModelResponseRequest(
                        name=self.model_armor_template_name,
                        model_response_data=data_item,
                    )
                    response = client.sanitize_model_response(request=request)
                else:
                    request = modelarmor_v1.SanitizeUserPromptRequest(
                        name=self.model_armor_template_name,
                        user_prompt_data=data_item,
                    )
                    response = client.sanitize_user_prompt(request=request)
            else:
                method = (
                    client.sanitize_model_response
                    if is_output
                    else client.sanitize_user_prompt
                )
                response = method(
                    name=self.model_armor_template_name,
                    text=text,
                )

            sanitization_result = getattr(response, "sanitization_result", None)
            match_state = getattr(sanitization_result, "filter_match_state", None)
            match_name = getattr(match_state, "name", str(match_state or ""))
            if match_name == "MATCH_FOUND" or match_state == 2:
                return {
                    "allowed": False,
                    "violation_code": "MODEL_ARMOR_POLICY_VIOLATION",
                    "message": (
                        "Request blocked by Google Cloud Model Armor safety policy."
                    ),
                }
        except Exception as exc:
            _emit_policy_log(
                "MODEL_ARMOR_CALL_WARNING",
                f"Model Armor sanitization check failed gracefully: {exc}",
                severity="WARNING",
            )
        return None

    def evaluate_input_guardrails(
        self,
        user_text: str,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Screen traveler input against prompt-injection, exfiltration, and safety rules.

        Args:
            user_text: Raw user prompt text submitted to the agent.
            session_state: Optional session state dictionary for lockout checks.

        Returns:
            Dictionary with `allowed` (bool), `violation_code` (str | None),
            and `message` (str).
        """
        normalized_text = str(user_text or "").strip()
        state = session_state or {}

        if state.get("security_lockout") is True:
            result: dict[str, Any] = {
                "allowed": False,
                "violation_code": "SESSION_SECURITY_LOCKOUT",
                "message": (
                    "This session is locked due to repeated security policy violations."
                ),
            }
            _emit_policy_log(
                "GUARDRAIL_INPUT_BLOCKED",
                result["message"],
                severity="WARNING",
                violation_code=result["violation_code"],
            )
            return result

        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(normalized_text):
                result = {
                    "allowed": False,
                    "violation_code": "PROMPT_INJECTION_DETECTED",
                    "message": (
                        "Request blocked by policy guardrail: attempts to override "
                        "system instructions, reveal internal prompts, or bypass "
                        "human approval controls are prohibited."
                    ),
                }
                _emit_policy_log(
                    "GUARDRAIL_INPUT_BLOCKED",
                    result["message"],
                    severity="WARNING",
                    violation_code=result["violation_code"],
                )
                return result

        for pattern in _CREDENTIAL_EXFILTRATION_PATTERNS:
            if pattern.search(normalized_text):
                result = {
                    "allowed": False,
                    "violation_code": "CREDENTIAL_EXFILTRATION_ATTEMPT",
                    "message": (
                        "Request blocked by policy guardrail: access to secrets, "
                        "API tokens, or environment credentials is strictly prohibited."
                    ),
                }
                _emit_policy_log(
                    "GUARDRAIL_INPUT_BLOCKED",
                    result["message"],
                    severity="WARNING",
                    violation_code=result["violation_code"],
                )
                return result

        for pattern in _UNSAFE_TRAVEL_PATTERNS:
            if pattern.search(normalized_text):
                result = {
                    "allowed": False,
                    "violation_code": "UNSAFE_OR_ILLEGAL_TRAVEL_REQUEST",
                    "message": (
                        "Request blocked by policy guardrail: the Travel Concierge "
                        "cannot assist with illegal, fraudulent, or unsafe travel activities."
                    ),
                }
                _emit_policy_log(
                    "GUARDRAIL_INPUT_BLOCKED",
                    result["message"],
                    severity="WARNING",
                    violation_code=result["violation_code"],
                )
                return result

        armor_verdict = self._screen_with_model_armor(normalized_text, is_output=False)
        if armor_verdict is not None:
            _emit_policy_log(
                "GUARDRAIL_INPUT_BLOCKED",
                armor_verdict["message"],
                severity="WARNING",
                violation_code=armor_verdict["violation_code"],
            )
            return armor_verdict

        _emit_policy_log(
            "GUARDRAIL_INPUT_ALLOWED",
            "User input passed policy guardrail screening.",
            severity="INFO",
            allowed=True,
        )
        return {
            "allowed": True,
            "violation_code": None,
            "message": "Input passed all policy and safety guardrails.",
        }

    def evaluate_output_guardrails(
        self,
        response_text: str,
        budget_limit_usd: float | None = None,
        proposed_cost_usd: float | None = None,
    ) -> dict[str, Any]:
        """Self-evaluation critic verifying budget compliance and scrubbing PII.

        Args:
            response_text: Candidate model output text.
            budget_limit_usd: Optional maximum traveler budget ceiling in USD.
            proposed_cost_usd: Optional computed total trip cost in USD.

        Returns:
            Dictionary containing `allowed`, `violation_code`, `message`,
            `sanitized_text`, `pii_redacted`, `budget_limit_usd`, and
            `proposed_cost_usd`.
        """
        raw_text = str(response_text or "")
        sanitized_text = _scrub_text_with_redactor(raw_text, self._redactor)
        pii_redacted = sanitized_text != raw_text

        effective_budget = (
            budget_limit_usd
            if budget_limit_usd is not None
            else self.default_budget_limit_usd
        )
        effective_cost = proposed_cost_usd
        if effective_cost is None and effective_budget is not None:
            match = _TOTAL_COST_EXTRACT_RE.search(sanitized_text)
            if match:
                try:
                    effective_cost = float(match.group(1).replace(",", ""))
                except ValueError:
                    effective_cost = None

        armor_verdict = self._screen_with_model_armor(sanitized_text, is_output=True)
        if armor_verdict is not None:
            return {
                "allowed": False,
                "violation_code": armor_verdict["violation_code"],
                "message": armor_verdict["message"],
                "sanitized_text": armor_verdict["message"],
                "pii_redacted": pii_redacted,
                "budget_limit_usd": effective_budget,
                "proposed_cost_usd": effective_cost,
            }

        if (
            effective_budget is not None
            and effective_cost is not None
            and effective_cost > effective_budget
        ):
            overage = round(effective_cost - effective_budget, 2)
            warning_msg = (
                f"Self-evaluation guardrail flagged budget overrun: proposed cost "
                f"${effective_cost:,.2f} USD exceeds budget limit "
                f"${effective_budget:,.2f} USD by ${overage:,.2f} USD. "
                f"Re-allocation or traveler budget increase approval is required."
            )
            critic_annotated_text = f"{sanitized_text}\n\n[BUDGET_POLICY_ALERT] {warning_msg}".strip()
            _emit_policy_log(
                "GUARDRAIL_OUTPUT_VIOLATION",
                warning_msg,
                severity="WARNING",
                violation_code="BUDGET_LIMIT_EXCEEDED",
                budget_limit_usd=effective_budget,
                proposed_cost_usd=effective_cost,
                overage_usd=overage,
                pii_redacted=pii_redacted,
            )
            return {
                "allowed": False,
                "violation_code": "BUDGET_LIMIT_EXCEEDED",
                "message": warning_msg,
                "sanitized_text": critic_annotated_text,
                "pii_redacted": pii_redacted,
                "budget_limit_usd": effective_budget,
                "proposed_cost_usd": effective_cost,
                "overage_usd": overage,
            }

        _emit_policy_log(
            "GUARDRAIL_OUTPUT_ALLOWED",
            "Model output passed self-evaluation and PII redaction checks.",
            severity="INFO",
            pii_redacted=pii_redacted,
            budget_limit_usd=effective_budget,
            proposed_cost_usd=effective_cost,
        )
        return {
            "allowed": True,
            "violation_code": None,
            "message": "Output passed self-evaluation and PII guardrails.",
            "sanitized_text": sanitized_text,
            "pii_redacted": pii_redacted,
            "budget_limit_usd": effective_budget,
            "proposed_cost_usd": effective_cost,
        }

    def check_before_model(
        self,
        callback_context: Any = None,
        llm_request: Any = None,
    ) -> LlmResponse | None:
        """Synchronous helper evaluating input guardrails for ADK model hooks."""
        user_text = extract_user_text_from_request(llm_request)
        state = extract_state_from_context(callback_context)
        verdict = self.evaluate_input_guardrails(user_text, session_state=state)
        if not verdict["allowed"]:
            return build_llm_response(
                text=verdict["message"],
                error_code=verdict["violation_code"],
                custom_metadata={"guardrail_verdict": verdict},
            )
        return None

    def check_after_model(
        self,
        callback_context: Any = None,
        llm_response: Any = None,
    ) -> LlmResponse | None:
        """Synchronous helper evaluating output guardrails for ADK model hooks."""
        response_text = extract_text_from_response(llm_response)
        state = extract_state_from_context(callback_context)
        budget_limit = state.get("budget_limit_usd")
        proposed_cost = state.get("proposed_cost_usd")
        verdict = self.evaluate_output_guardrails(
            response_text=response_text,
            budget_limit_usd=float(budget_limit) if budget_limit is not None else None,
            proposed_cost_usd=(
                float(proposed_cost) if proposed_cost is not None else None
            ),
        )
        if not verdict["allowed"]:
            return build_llm_response(
                text=verdict["sanitized_text"],
                error_code=verdict["violation_code"],
                custom_metadata={"guardrail_verdict": verdict},
            )
        if verdict["pii_redacted"]:
            return build_llm_response(
                text=verdict["sanitized_text"],
                custom_metadata={"guardrail_verdict": verdict},
            )
        return None

    async def before_model_callback(
        self,
        callback_context: CallbackContext = None,  # type: ignore[assignment]
        llm_request: LlmRequest = None,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> LlmResponse | None:
        """ADK BasePlugin hook invoked before sending an LlmRequest to Gemini."""
        ctx = callback_context if callback_context is not None else kwargs.get("callback_context")
        req = llm_request if llm_request is not None else kwargs.get("llm_request")
        return self.check_before_model(callback_context=ctx, llm_request=req)

    async def after_model_callback(
        self,
        callback_context: CallbackContext = None,  # type: ignore[assignment]
        llm_response: LlmResponse = None,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> LlmResponse | None:
        """ADK BasePlugin hook invoked after receiving an LlmResponse from Gemini."""
        ctx = callback_context if callback_context is not None else kwargs.get("callback_context")
        resp = llm_response if llm_response is not None else kwargs.get("llm_response")
        return self.check_after_model(callback_context=ctx, llm_response=resp)


_DEFAULT_POLICY_PLUGIN: PolicyGuardrailPlugin | None = None


def get_default_policy_plugin() -> PolicyGuardrailPlugin:
    """Return the singleton PolicyGuardrailPlugin instance."""
    global _DEFAULT_POLICY_PLUGIN
    if _DEFAULT_POLICY_PLUGIN is None:
        _DEFAULT_POLICY_PLUGIN = PolicyGuardrailPlugin()
    return _DEFAULT_POLICY_PLUGIN


def guardrail_after_model_callback(
    callback_context: Any,
    llm_response: Any,
) -> LlmResponse | None:
    """Agent-level `after_model_callback` running self-evaluation and PII scrubbing."""
    return get_default_policy_plugin().check_after_model(
        callback_context=callback_context,
        llm_response=llm_response,
    )
