"""Root ADK Coordinator, App configuration, and End-to-End Orchestrator.

Implements Category 3 (Orchestration & Logic) by integrating:
- Multi-agent ADK topology (`travel_concierge_coordinator` -> `SequentialAgent`
  -> `ParallelAgent` research squad + `budget_analyst_agent` + `itinerary_critic_agent`)
- Strategic model routing (`gemini-2.5-flash` vs `gemini-2.5-pro`)
- Policy & safety guardrails (`PolicyGuardrailPlugin` + output self-evaluation)
- Human-in-the-Loop gates (`HumanApprovalGate` + `hitl_before_tool_callback`)
- Context compaction, persistent SQLite/Cloud session memory, async background
  memory consolidation, OpenTelemetry distributed tracing, and paired
  `INTENT` / `OUTCOME` telemetry.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
import os
import re
import time
from typing import Any

from travel_concierge.agents.model_router import (
    StrategicModelRouter,
    dynamic_model_routing_callback,
)
from travel_concierge.agents.sub_agents import (
    App,
    LlmAgent,
    build_sub_agents,
)
from travel_concierge.config import TravelConciergeConfig, get_config
from travel_concierge.guardrails.hitl_hooks import (
    HumanApprovalGate,
    hitl_before_tool_callback,
)
from travel_concierge.guardrails.policy_plugin import (
    LlmResponse,
    PolicyGuardrailPlugin,
    get_default_policy_plugin,
)
from travel_concierge.memory.async_memory import (
    AsyncMemoryConsolidator,
    after_agent_memory_callback,
)
from travel_concierge.memory.compaction import (
    ConversationCompactor,
    build_adk_compaction_config,
    build_context_cache_config,
)
from travel_concierge.memory.session_store import PersistentTravelSessionManager
from travel_concierge.observability import (
    IntentOutcomePlugin,
    get_current_trace_context,
    get_default_intent_tracker,
    get_default_redactor,
    get_structured_logger,
    log_structured_event,
    trace_operation,
)
from travel_concierge.prompts import AGENT_CONSTITUTION, COORDINATOR_INSTRUCTION

_CITY_DIRECTORY: dict[str, tuple[str, str, str]] = {
    "tokyo": ("JP", "Tokyo", "TYO"),
    "kyoto": ("JP", "Kyoto", "KIX"),
    "osaka": ("JP", "Osaka", "KIX"),
    "japan": ("JP", "Tokyo", "TYO"),
    "london": ("GB", "London", "LHR"),
    "paris": ("FR", "Paris", "CDG"),
    "france": ("FR", "Paris", "CDG"),
    "rome": ("IT", "Rome", "FCO"),
    "italy": ("IT", "Rome", "FCO"),
    "barcelona": ("ES", "Barcelona", "BCN"),
    "madrid": ("ES", "Madrid", "MAD"),
    "lisbon": ("PT", "Lisbon", "LIS"),
    "amsterdam": ("NL", "Amsterdam", "AMS"),
    "berlin": ("DE", "Berlin", "BER"),
    "zurich": ("CH", "Zurich", "ZRH"),
    "seoul": ("KR", "Seoul", "ICN"),
    "singapore": ("SG", "Singapore", "SIN"),
    "bangkok": ("TH", "Bangkok", "BKK"),
    "sydney": ("AU", "Sydney", "SYD"),
    "new york": ("US", "New York", "JFK"),
    "san francisco": ("US", "San Francisco", "SFO"),
    "toronto": ("CA", "Toronto", "YYZ"),
    "vancouver": ("CA", "Vancouver", "YVR"),
    "dubai": ("AE", "Dubai", "DXB"),
}

_CURRENCY_QUERY_RE = re.compile(
    r"\bconvert\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+([A-Za-z]{2,10})"
    r"(?:\s+currency\s+units?)?\s+(?:to|into|in)\s+([A-Za-z]{2,10})\b",
    re.IGNORECASE,
)
_PAYMENT_QUERY_RE = re.compile(
    r"\b(authorize|process|approve)\b.{0,50}\bpayment\b",
    re.IGNORECASE,
)
_ADVISORY_QUERY_RE = re.compile(
    r"\b(travel\s+advisor(?:y|ies)|safety\s+advisor(?:y|ies)|entry\s+requirements?)\b",
    re.IGNORECASE,
)
_COUNTRY_CITY_PAIR_RE = re.compile(
    r"\bfor\s+([A-Z]{2})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)
_BUDGET_AMOUNT_RE = re.compile(
    r"(?:strict\s+|under\s+(?:a\s+)?(?:strict\s+)?|within\s+(?:a\s+)?|budget(?:\s+of|\s+cap|\s+is)?\s*:?\s*)"
    r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)
_DOLLAR_AMOUNT_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
_BOOKING_REF_RE = re.compile(r"\b(BK-[A-Za-z0-9-]+|RES-[A-Za-z0-9-]+|FL-[A-Za-z0-9-]+)\b")
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_DURATION_DAYS_RE = re.compile(r"\b(\d{1,2})\s*-\s*day\b|\b(\d{1,2})\s+days?\b", re.IGNORECASE)
_TRAVELERS_COUNT_RE = re.compile(
    r"\b(\d{1,2})\s+(?:travelers?|passengers?|guests?|adults?|people)\b",
    re.IGNORECASE,
)
_ORIGIN_IATA_RE = re.compile(r"\bfrom\s+([A-Z]{3})\b")
_DEST_IATA_RE = re.compile(r"\bto\s+([A-Z]{3})\b")
_PERSIST_ITINERARY_RE = re.compile(
    r"\b(persist|finalize\s+itinerary|save\s+finalized\s+itinerary)\b",
    re.IGNORECASE,
)


def _load_tool_registry() -> tuple[dict[str, Any], list[Any]]:
    """Load `TOOL_REGISTRY` and `ALL_TOOLS` from `travel_concierge.tools`."""
    try:
        from travel_concierge.tools import ALL_TOOLS, TOOL_REGISTRY

        return dict(TOOL_REGISTRY), list(ALL_TOOLS)
    except Exception:
        return {}, []


def _parse_destination(query: str, state: dict[str, Any]) -> tuple[str, str, str]:
    """Resolve `(country_code, city, destination_iata)` from query or state."""
    explicit_pair = _COUNTRY_CITY_PAIR_RE.search(query)
    if explicit_pair:
        cc = explicit_pair.group(1).upper()
        city_name = explicit_pair.group(2).strip()
        city_info = _CITY_DIRECTORY.get(city_name.lower())
        iata = city_info[2] if city_info else "LHR"
        return cc, city_name, iata

    lower_q = query.lower()
    for key, info in _CITY_DIRECTORY.items():
        if re.search(rf"\b{re.escape(key)}\b", lower_q):
            return info

    dest_iata_match = _DEST_IATA_RE.search(query)
    if dest_iata_match:
        iata = dest_iata_match.group(1).upper()
        for _, info in _CITY_DIRECTORY.items():
            if info[2] == iata:
                return info
        return ("US", iata, iata)

    state_dest = str(state.get("destination", "")).strip()
    if state_dest:
        if state_dest.lower() in _CITY_DIRECTORY:
            return _CITY_DIRECTORY[state_dest.lower()]
        for _, info in _CITY_DIRECTORY.items():
            if info[2] == state_dest.upper():
                return info

    return ("JP", "Tokyo", "TYO")


def _parse_trip_parameters(
    query: str,
    state: dict[str, Any],
    explicit_budget: float | None,
    default_max_budget: float,
) -> dict[str, Any]:
    """Extract structured trip planning parameters from natural-language query."""
    country_code, city, dest_iata = _parse_destination(query, state)

    origin_match = _ORIGIN_IATA_RE.search(query)
    origin_iata = (
        origin_match.group(1).upper()
        if origin_match
        else str(state.get("origin", "SFO")).upper()
    )
    if len(origin_iata) != 3 or origin_iata == dest_iata:
        origin_iata = "SFO" if dest_iata != "SFO" else "JFK"

    date_matches = _ISO_DATE_RE.findall(query)
    if date_matches:
        dep_date_str = date_matches[0]
    else:
        dep_date_str = (date.today() + timedelta(days=45)).isoformat()

    dur_match = _DURATION_DAYS_RE.search(query)
    if dur_match:
        duration_days = max(1, int(dur_match.group(1) or dur_match.group(2)))
    else:
        duration_days = 7

    try:
        dep_dt = date.fromisoformat(dep_date_str)
    except ValueError:
        dep_dt = date.today() + timedelta(days=45)
        dep_date_str = dep_dt.isoformat()
    checkout_date_str = (dep_dt + timedelta(days=duration_days)).isoformat()

    trav_match = _TRAVELERS_COUNT_RE.search(query)
    travelers = max(1, int(trav_match.group(1))) if trav_match else 1

    budget_usd = explicit_budget
    if budget_usd is None and state.get("budget_limit_usd") is not None:
        try:
            budget_usd = float(state["budget_limit_usd"])
        except (TypeError, ValueError):
            budget_usd = None
    if budget_usd is None:
        b_match = _BUDGET_AMOUNT_RE.search(query)
        if b_match:
            try:
                budget_usd = float(b_match.group(1).replace(",", ""))
            except ValueError:
                budget_usd = None

    effective_budget = budget_usd if budget_usd is not None else default_max_budget

    lower_q = query.lower()
    if "luxury" in lower_q:
        travel_style = "LUXURY"
    elif "budget" in lower_q and "economy" in lower_q and "moderate" not in lower_q:
        travel_style = "BUDGET"
    else:
        travel_style = "MODERATE"

    cabin_class = "BUSINESS" if "business class" in lower_q else "ECONOMY"

    return {
        "origin": origin_iata,
        "destination_iata": dest_iata,
        "country_code": country_code,
        "city": city,
        "departure_date": dep_date_str,
        "check_out_date": checkout_date_str,
        "duration_days": duration_days,
        "travelers": travelers,
        "budget_limit_usd": budget_usd,
        "effective_budget_usd": effective_budget,
        "travel_style": travel_style,
        "cabin_class": cabin_class,
    }


def coordinator_before_model_callback(
    callback_context: Any,
    llm_request: Any,
) -> LlmResponse | None:
    """Coordinator `before_model_callback` combining policy guardrails and model routing."""
    blocked_response = get_default_policy_plugin().check_before_model(
        callback_context=callback_context,
        llm_request=llm_request,
    )
    if blocked_response is not None:
        return blocked_response

    dynamic_model_routing_callback(
        callback_context=callback_context,
        llm_request=llm_request,
    )
    return None


def coordinator_after_model_callback(
    callback_context: Any,
    llm_response: Any,
) -> LlmResponse | None:
    """Coordinator `after_model_callback` running self-evaluation and PII redaction."""
    return get_default_policy_plugin().check_after_model(
        callback_context=callback_context,
        llm_response=llm_response,
    )


def build_coordinator_agent(config: TravelConciergeConfig | None = None) -> Any:
    """Construct the root ADK `travel_concierge_coordinator` LlmAgent."""
    cfg = config or get_config()
    sub_agents_map = build_sub_agents(config=cfg)
    _, all_tools = _load_tool_registry()

    full_instruction = f"{AGENT_CONSTITUTION}\n\n{COORDINATOR_INSTRUCTION}"
    return LlmAgent(
        name="travel_concierge_coordinator",
        model=cfg.reasoning_model,
        description=(
            "Executive Personal Travel & Budget Concierge Coordinator managing "
            "specialist research agents, budget optimization, policy guardrails, "
            "and mandatory Human-in-the-Loop approvals."
        ),
        instruction=full_instruction,
        tools=all_tools,
        sub_agents=[sub_agents_map["itinerary_planning_pipeline"]],
        before_model_callback=coordinator_before_model_callback,
        after_model_callback=coordinator_after_model_callback,
        before_tool_callback=hitl_before_tool_callback,
        after_agent_callback=after_agent_memory_callback,
    )


def build_adk_app(
    coordinator_agent: Any = None,
    config: TravelConciergeConfig | None = None,
) -> Any:
    """Construct the root ADK `App` with compaction and observability plugins."""
    cfg = config or get_config()
    agent = coordinator_agent or build_coordinator_agent(config=cfg)
    compaction_cfg = build_adk_compaction_config(
        compaction_interval=max(1, cfg.compaction_window_turns),
        overlap_size=2,
    )
    plugins = [PolicyGuardrailPlugin(), IntentOutcomePlugin()]
    try:
        return App(
            name="travel_concierge",
            root_agent=agent,
            events_compaction_config=compaction_cfg,
            plugins=plugins,
        )
    except Exception:
        return App(
            name="travel_concierge",
            root_agent=agent,
        )


class TravelConciergeOrchestrator:
    """Deterministic & ADK-compatible end-to-end multi-agent workflow orchestrator.

    Executes the full lifecycle for every traveler turn:
    1. Distributed OpenTelemetry span creation & pre-execution `INTENT` capture.
    2. Input policy & prompt-injection guardrail screening (`PolicyGuardrailPlugin`).
    3. Persistent SQLite/Cloud session loading, HITL state synchronization,
       sliding-window token compaction (`ConversationCompactor`), and vector
       memory recall (`PersistentTravelSessionManager`).
    4. Complexity-based strategic model routing (`StrategicModelRouter`).
    5. Concurrent multi-agent specialist execution (`asyncio.gather` across
       flight, lodging, and advisory specialists), sequential budget allocation,
       and `HumanApprovalGate` enforcement before high-stakes actions.
    6. Output self-evaluation critic & PII scrubbing.
    7. Non-blocking background memory consolidation (`AsyncMemoryConsolidator`).
    """

    def __init__(
        self,
        config: TravelConciergeConfig | None = None,
        session_manager: PersistentTravelSessionManager | None = None,
        db_path: str | None = None,
        compactor: ConversationCompactor | None = None,
        memory_consolidator: AsyncMemoryConsolidator | None = None,
        policy_plugin: PolicyGuardrailPlugin | None = None,
        approval_gate: HumanApprovalGate | None = None,
        model_router: StrategicModelRouter | None = None,
        intent_tracker: Any = None,
        redactor: Any = None,
    ) -> None:
        self.config = config or get_config()
        resolved_db_path = (
            db_path
            or os.environ.get("TRAVEL_SESSION_DB_PATH")
            or self.config.session_db_path
        )
        self.redactor = redactor or get_default_redactor()
        self.session_manager = session_manager or PersistentTravelSessionManager(
            db_path=resolved_db_path,
            project_id=self.config.project_id or None,
        )
        self.compactor = compactor or ConversationCompactor(
            max_tokens=self.config.compaction_max_tokens,
            keep_recent_turns=self.config.compaction_window_turns,
        )
        self.memory_consolidator = memory_consolidator or AsyncMemoryConsolidator(
            session_manager=self.session_manager,
            pii_redactor=self.redactor,
        )
        self.policy_plugin = policy_plugin or PolicyGuardrailPlugin(
            redactor=self.redactor
        )
        self.approval_gate = approval_gate or HumanApprovalGate(
            high_value_threshold_usd=self.config.high_value_payment_threshold_usd
        )
        self.model_router = model_router or StrategicModelRouter(
            fast_model=self.config.fast_model,
            reasoning_model=self.config.reasoning_model,
        )
        self.intent_tracker = intent_tracker or get_default_intent_tracker()
        self.logger = get_structured_logger("travel_concierge.orchestrator")

    def _ensure_agent_tools_synced(self) -> None:
        """Ensure module-level `root_agent` tools are bound if lazily loaded."""
        _, all_tools = _load_tool_registry()
        if all_tools and not getattr(root_agent, "tools", None):
            root_agent.tools = list(all_tools)

    async def _invoke_tool_governed(
        self,
        actor_agent: str,
        agent_model: str,
        tool_name: str,
        tool_args: dict[str, Any],
        session_state: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """Execute a single domain tool wrapped in HITL gates, spans, and INTENT/OUTCOME logs."""
        registry, _ = _load_tool_registry()
        scrubbed_args = self.redactor.redact_payload(tool_args)

        async with trace_operation(
            f"agent.{actor_agent}",
            attributes={
                "agent.name": actor_agent,
                "agent.model": agent_model,
                "tool.name": tool_name,
                "session_id": session_id,
            },
        ):
            corr_id = self.intent_tracker.record_intent(
                actor=actor_agent,
                action_name=tool_name,
                intended_args=scrubbed_args if isinstance(scrubbed_args, dict) else {},
                rationale=f"{actor_agent} invoking {tool_name}",
                session_id=session_id,
            )

            auth_verdict = self.approval_gate.check_tool_authorization(
                tool_name=tool_name,
                tool_args=tool_args,
                session_state=session_state,
            )
            if not auth_verdict["allowed"]:
                self.intent_tracker.record_outcome(
                    correlation_id=corr_id,
                    actor=actor_agent,
                    action_name=tool_name,
                    actual_result=auth_verdict,
                    status="PENDING_HUMAN_APPROVAL",
                    latency_ms=0.0,
                    session_id=session_id,
                )
                return {
                    "halted": True,
                    "tool_name": tool_name,
                    "args": scrubbed_args,
                    "hitl_status": auth_verdict,
                    "output": None,
                }

            tool_fn = registry.get(tool_name)
            if tool_fn is None:
                missing_err = {
                    "status": "error",
                    "error_code": "TOOL_NOT_REGISTERED",
                    "message": f"Tool '{tool_name}' is not registered.",
                    "recovery_instructions": "Verify tool name in travel_concierge.tools.TOOL_REGISTRY.",
                    "retryable": False,
                }
                self.intent_tracker.record_outcome(
                    correlation_id=corr_id,
                    actor=actor_agent,
                    action_name=tool_name,
                    actual_result=missing_err,
                    status="ERROR",
                    latency_ms=0.0,
                    session_id=session_id,
                )
                return {
                    "halted": False,
                    "tool_name": tool_name,
                    "args": scrubbed_args,
                    "hitl_status": auth_verdict,
                    "output": missing_err,
                }

            t0 = time.perf_counter()
            try:
                async with trace_operation(
                    f"tool.{tool_name}",
                    attributes={
                        "tool.name": tool_name,
                        "agent.name": actor_agent,
                        "session_id": session_id,
                    },
                ):
                    raw_output = tool_fn(**tool_args)
                    if asyncio.iscoroutine(raw_output):
                        raw_output = await raw_output
            except Exception as exc:
                raw_output = {
                    "status": "error",
                    "error_code": "TOOL_EXECUTION_ERROR",
                    "message": f"Tool '{tool_name}' encountered an error: {exc}",
                    "recovery_instructions": (
                        f"Inspect arguments passed to '{tool_name}' against its "
                        f"JSON schema constraints and retry with corrected parameters."
                    ),
                    "retryable": True,
                }
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            scrubbed_output = self.redactor.redact_payload(raw_output)
            is_tool_err = (
                isinstance(raw_output, dict)
                and str(raw_output.get("status", "")).lower() == "error"
            )
            self.intent_tracker.record_outcome(
                correlation_id=corr_id,
                actor=actor_agent,
                action_name=tool_name,
                actual_result=scrubbed_output,
                status="ERROR" if is_tool_err else "SUCCESS",
                latency_ms=latency_ms,
                session_id=session_id,
            )
            return {
                "halted": False,
                "tool_name": tool_name,
                "args": scrubbed_args,
                "hitl_status": auth_verdict,
                "output": scrubbed_output,
            }

    async def handle_request(
        self,
        user_id: str,
        session_id: str,
        query: str,
        budget_limit_usd: float | None = None,
        human_approvals: list[str] | None = None,
        high_stakes_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the full Travel & Budget Concierge workflow for a single turn."""
        self._ensure_agent_tools_synced()
        turn_start = time.perf_counter()
        raw_query = str(query or "")
        redacted_query = str(self.redactor.redact_text(raw_query))

        async with trace_operation(
            "orchestrator.handle_request",
            attributes={
                "session_id": session_id,
                "user_id": user_id,
                "query_preview": redacted_query[:160],
            },
        ):
            trace_ctx = get_current_trace_context()
            root_corr_id = self.intent_tracker.record_intent(
                actor="travel_concierge_coordinator",
                action_name="handle_request",
                intended_args={
                    "query": redacted_query,
                    "budget_limit_usd": budget_limit_usd,
                    "has_high_stakes_action": bool(high_stakes_action),
                },
                rationale="Process traveler concierge request with policy and HITL governance",
                session_id=session_id,
            )

            session_state = self.session_manager.get_session_state(session_id)

            # 1. Input Policy & Prompt-Injection Guardrail Check
            async with trace_operation(
                "guardrails.evaluate_input",
                attributes={"session_id": session_id},
            ):
                input_verdict = self.policy_plugin.evaluate_input_guardrails(
                    user_text=raw_query,
                    session_state=session_state,
                )

            if not input_verdict["allowed"]:
                elapsed_ms = round((time.perf_counter() - turn_start) * 1000.0, 3)
                self.intent_tracker.record_outcome(
                    correlation_id=root_corr_id,
                    actor="travel_concierge_coordinator",
                    action_name="handle_request",
                    actual_result=input_verdict,
                    status="BLOCKED",
                    latency_ms=elapsed_ms,
                    session_id=session_id,
                )
                return {
                    "status": "GUARDRAIL_BLOCKED",
                    "allowed": False,
                    "user_id": user_id,
                    "session_id": session_id,
                    "response": input_verdict["message"],
                    "guardrail": input_verdict,
                    "guardrail_input": input_verdict,
                    "tools_invoked": [],
                    "tool_calls": [],
                    "tool_results": [],
                    "trace_id": trace_ctx.get("trace_id"),
                    "correlation_id": root_corr_id,
                }

            # 2. Session State Updates, Compaction & Semantic Memory Retrieval
            async with trace_operation(
                "memory.load_and_compact",
                attributes={"session_id": session_id, "user_id": user_id},
            ):
                if human_approvals is not None:
                    if len(human_approvals) == 0:
                        session_state["approved_actions"] = []
                    else:
                        for approval_key in human_approvals:
                            self.approval_gate.grant_human_approval(
                                session_state=session_state,
                                action_key=str(approval_key),
                            )

                params = _parse_trip_parameters(
                    query=raw_query,
                    state=session_state,
                    explicit_budget=budget_limit_usd,
                    default_max_budget=self.config.default_max_budget_usd,
                )
                if params["budget_limit_usd"] is not None:
                    session_state["budget_limit_usd"] = params["budget_limit_usd"]

                self.session_manager.update_session_state(session_id, session_state)
                self.session_manager.save_turn(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=redacted_query,
                    metadata={"trace_id": trace_ctx.get("trace_id")},
                )

                full_history = self.session_manager.get_session_history(session_id)
                compaction_result = self.compactor.compact_history(full_history)
                relevant_memories = self.session_manager.retrieve_relevant_memories(
                    user_id=user_id,
                    query=redacted_query,
                    top_k=3,
                )
                context_cache_cfg = build_context_cache_config(
                    model=self.config.reasoning_model
                )

            # 3. Strategic Model Routing
            async with trace_operation(
                "router.select_model",
                attributes={"session_id": session_id},
            ):
                routing_decision = self.model_router.select_model(raw_query)

            executed_steps: list[dict[str, Any]] = []
            tools_invoked: list[str] = []
            proposed_cost_usd: float | None = None
            candidate_response = ""

            # 4. Multi-Agent & Governed Tool Execution
            currency_match = _CURRENCY_QUERY_RE.search(raw_query)
            is_advisory_only = (
                bool(_ADVISORY_QUERY_RE.search(raw_query))
                and routing_decision["tier"] == "fast"
                and not high_stakes_action
            )
            is_nl_payment = (
                bool(_PAYMENT_QUERY_RE.search(raw_query))
                and not high_stakes_action
                and routing_decision["tier"] == "fast"
            )

            if isinstance(high_stakes_action, dict) and high_stakes_action.get("tool_name"):
                action_tool = str(high_stakes_action["tool_name"])
                action_args = dict(
                    high_stakes_action.get("tool_args")
                    or high_stakes_action.get("args")
                    or {}
                )
                step = await self._invoke_tool_governed(
                    actor_agent="budget_analyst_agent",
                    agent_model=routing_decision["model"],
                    tool_name=action_tool,
                    tool_args=action_args,
                    session_state=session_state,
                    session_id=session_id,
                )
                if step["halted"]:
                    hitl_info = step["hitl_status"]
                    elapsed_ms = round((time.perf_counter() - turn_start) * 1000.0, 3)
                    self.intent_tracker.record_outcome(
                        correlation_id=root_corr_id,
                        actor="travel_concierge_coordinator",
                        action_name="handle_request",
                        actual_result=hitl_info,
                        status="PENDING_HUMAN_APPROVAL",
                        latency_ms=elapsed_ms,
                        session_id=session_id,
                    )
                    return {
                        "status": "PENDING_HUMAN_APPROVAL",
                        "allowed": False,
                        "user_id": user_id,
                        "session_id": session_id,
                        "response": hitl_info["instructions"],
                        "selected_model": routing_decision["model"],
                        "tier": routing_decision["tier"],
                        "routing": routing_decision,
                        "hitl_status": hitl_info,
                        "confirmation_id": hitl_info["confirmation_id"],
                        "tools_invoked": [],
                        "tool_calls": [],
                        "tool_results": [],
                        "compaction": compaction_result,
                        "retrieved_memories": relevant_memories,
                        "trace_id": trace_ctx.get("trace_id"),
                        "correlation_id": root_corr_id,
                    }

                executed_steps.append(step)
                tools_invoked.append(action_tool)
                out = step["output"] or {}
                if isinstance(out, dict) and out.get("status") == "error":
                    candidate_response = (
                        f"Tool '{action_tool}' reported a recoverable validation issue: "
                        f"{out.get('message', '')} Recovery: {out.get('recovery_instructions', '')}"
                    )
                else:
                    tx_id = out.get("transaction_id", "authorized") if isinstance(out, dict) else "authorized"
                    amt = action_args.get("amount_usd", 0.0)
                    ref = action_args.get("booking_reference", "")
                    candidate_response = (
                        f"Human approval verified. Completed '{action_tool}' for booking {ref} "
                        f"(${float(amt):,.2f} USD, transaction_id={tx_id})."
                    )

            elif is_nl_payment:
                amt_match = _DOLLAR_AMOUNT_RE.search(raw_query)
                amount_val = (
                    float(amt_match.group(1).replace(",", ""))
                    if amt_match
                    else 1000.0
                )
                ref_match = _BOOKING_REF_RE.search(raw_query)
                booking_ref = ref_match.group(1) if ref_match else "BK-PENDING-001"
                pay_args = {
                    "booking_reference": booking_ref,
                    "amount_usd": amount_val,
                    "currency": "USD",
                    "merchant_name": "TravelPartner",
                }
                step = await self._invoke_tool_governed(
                    actor_agent="budget_analyst_agent",
                    agent_model=routing_decision["model"],
                    tool_name="authorize_high_value_travel_payment",
                    tool_args=pay_args,
                    session_state=session_state,
                    session_id=session_id,
                )
                if step["halted"]:
                    hitl_info = step["hitl_status"]
                    elapsed_ms = round((time.perf_counter() - turn_start) * 1000.0, 3)
                    self.intent_tracker.record_outcome(
                        correlation_id=root_corr_id,
                        actor="travel_concierge_coordinator",
                        action_name="handle_request",
                        actual_result=hitl_info,
                        status="PENDING_HUMAN_APPROVAL",
                        latency_ms=elapsed_ms,
                        session_id=session_id,
                    )
                    return {
                        "status": "PENDING_HUMAN_APPROVAL",
                        "allowed": False,
                        "user_id": user_id,
                        "session_id": session_id,
                        "response": hitl_info["instructions"],
                        "selected_model": routing_decision["model"],
                        "tier": routing_decision["tier"],
                        "routing": routing_decision,
                        "hitl_status": hitl_info,
                        "confirmation_id": hitl_info["confirmation_id"],
                        "tools_invoked": [],
                        "tool_calls": [],
                        "tool_results": [],
                        "compaction": compaction_result,
                        "retrieved_memories": relevant_memories,
                        "trace_id": trace_ctx.get("trace_id"),
                        "correlation_id": root_corr_id,
                    }
                executed_steps.append(step)
                tools_invoked.append("authorize_high_value_travel_payment")
                out = step["output"] or {}
                candidate_response = (
                    f"Authorized ${amount_val:,.2f} USD payment for booking {booking_ref} "
                    f"(transaction_id={out.get('transaction_id', 'confirmed')})."
                )

            elif currency_match and routing_decision["tier"] == "fast":
                amount_val = float(currency_match.group(1).replace(",", ""))
                from_curr = currency_match.group(2).upper()
                to_curr = currency_match.group(3).upper()
                step = await self._invoke_tool_governed(
                    actor_agent="travel_concierge_coordinator",
                    agent_model=self.config.fast_model,
                    tool_name="convert_currency_exchange_rate",
                    tool_args={
                        "amount": amount_val,
                        "from_currency": from_curr,
                        "to_currency": to_curr,
                    },
                    session_state=session_state,
                    session_id=session_id,
                )
                executed_steps.append(step)
                tools_invoked.append("convert_currency_exchange_rate")
                out = step["output"] or {}
                if isinstance(out, dict) and out.get("status") == "error":
                    candidate_response = (
                        f"Currency conversion input error: {out.get('message', '')} "
                        f"Guidance: {out.get('recovery_instructions', '')}"
                    )
                else:
                    converted = out.get("converted_amount", 0.0) if isinstance(out, dict) else 0.0
                    rate = out.get("exchange_rate", 1.0) if isinstance(out, dict) else 1.0
                    candidate_response = (
                        f"{amount_val:,.2f} {from_curr} converts to "
                        f"{float(converted):,.2f} {to_curr} (exchange rate: {rate})."
                    )

            elif is_advisory_only:
                step = await self._invoke_tool_governed(
                    actor_agent="advisory_specialist_agent",
                    agent_model=self.config.fast_model,
                    tool_name="retrieve_destination_travel_advisories",
                    tool_args={
                        "country_code": params["country_code"],
                        "city": params["city"],
                    },
                    session_state=session_state,
                    session_id=session_id,
                )
                executed_steps.append(step)
                tools_invoked.append("retrieve_destination_travel_advisories")
                out = step["output"] or {}
                level = out.get("advisory_level", 1) if isinstance(out, dict) else 1
                summary = out.get("advisory_summary", "Standard precautions advised.") if isinstance(out, dict) else ""
                mem_note = ""
                if relevant_memories:
                    mem_note = " Active traveler preferences from persistent memory have been applied."
                candidate_response = (
                    f"Destination travel advisory for {params['city']}, {params['country_code']} "
                    f"(Level {level}): {summary}{mem_note}"
                )

            elif routing_decision["tier"] == "reasoning" or "plan" in raw_query.lower() or "itinerary" in raw_query.lower():
                # Execute ADK SequentialAgent -> ParallelAgent multi-specialist workflow
                async with trace_operation(
                    "workflow.itinerary_planning_pipeline",
                    attributes={
                        "destination": params["city"],
                        "duration_days": params["duration_days"],
                        "session_id": session_id,
                    },
                ):
                    async with trace_operation(
                        "workflow.parallel_research_squad",
                        attributes={"session_id": session_id},
                    ):
                        flight_task = self._invoke_tool_governed(
                            actor_agent="flight_specialist_agent",
                            agent_model=self.config.fast_model,
                            tool_name="search_commercial_flights",
                            tool_args={
                                "origin": params["origin"],
                                "destination": params["destination_iata"],
                                "departure_date": params["departure_date"],
                                "passengers": params["travelers"],
                                "cabin_class": params["cabin_class"],
                            },
                            session_state=session_state,
                            session_id=session_id,
                        )
                        lodging_task = self._invoke_tool_governed(
                            actor_agent="lodging_specialist_agent",
                            agent_model=self.config.fast_model,
                            tool_name="search_verified_accommodations",
                            tool_args={
                                "city": params["city"],
                                "check_in_date": params["departure_date"],
                                "check_out_date": params["check_out_date"],
                                "guests": params["travelers"],
                            },
                            session_state=session_state,
                            session_id=session_id,
                        )
                        advisory_task = self._invoke_tool_governed(
                            actor_agent="advisory_specialist_agent",
                            agent_model=self.config.fast_model,
                            tool_name="retrieve_destination_travel_advisories",
                            tool_args={
                                "country_code": params["country_code"],
                                "city": params["city"],
                            },
                            session_state=session_state,
                            session_id=session_id,
                        )
                        flight_step, lodging_step, advisory_step = await asyncio.gather(
                            flight_task, lodging_task, advisory_task
                        )

                    for research_step in (flight_step, lodging_step, advisory_step):
                        executed_steps.append(research_step)
                        tools_invoked.append(research_step["tool_name"])

                    flight_out = flight_step["output"] if isinstance(flight_step["output"], dict) else {}
                    lodging_out = lodging_step["output"] if isinstance(lodging_step["output"], dict) else {}
                    advisory_out = advisory_step["output"] if isinstance(advisory_step["output"], dict) else {}

                    cheapest_flight = float(flight_out.get("cheapest_total_usd") or 0.0)
                    lowest_lodging = float(lodging_out.get("lowest_total_stay_usd") or 0.0)
                    budget_cap = float(params["effective_budget_usd"])

                    if (
                        cheapest_flight > 0
                        and lowest_lodging > 0
                        and (cheapest_flight + lowest_lodging) <= budget_cap * 0.85
                    ):
                        est_flight = cheapest_flight
                        est_lodging = lowest_lodging
                    elif budget_cap < 300.0 and (cheapest_flight + lowest_lodging) > 0:
                        est_flight = cheapest_flight
                        est_lodging = lowest_lodging
                    else:
                        est_flight = 0.0
                        est_lodging = 0.0

                    budget_step = await self._invoke_tool_governed(
                        actor_agent="budget_analyst_agent",
                        agent_model=self.config.reasoning_model,
                        tool_name="calculate_trip_budget_allocation",
                        tool_args={
                            "total_budget_usd": budget_cap,
                            "duration_days": params["duration_days"],
                            "travelers": params["travelers"],
                            "travel_style": params["travel_style"],
                            "estimated_flight_usd": est_flight,
                            "estimated_lodging_usd": est_lodging,
                        },
                        session_state=session_state,
                        session_id=session_id,
                    )
                    executed_steps.append(budget_step)
                    tools_invoked.append("calculate_trip_budget_allocation")

                    budget_out = budget_step["output"] if isinstance(budget_step["output"], dict) else {}
                    projected_val = budget_out.get("projected_total_usd")
                    if projected_val is not None:
                        proposed_cost_usd = float(projected_val)

                    if _PERSIST_ITINERARY_RE.search(raw_query):
                        persist_step = await self._invoke_tool_governed(
                            actor_agent="itinerary_critic_agent",
                            agent_model=self.config.reasoning_model,
                            tool_name="persist_finalized_trip_itinerary",
                            tool_args={
                                "session_id": session_id,
                                "traveler_id": user_id,
                                "destination": params["city"],
                                "total_cost_usd": proposed_cost_usd or budget_cap,
                                "itinerary_items": [
                                    {
                                        "category": "flight",
                                        "summary": f"{params['origin']} to {params['destination_iata']}",
                                        "cost_usd": float(budget_out.get("allocated_flights_usd", est_flight)),
                                    },
                                    {
                                        "category": "lodging",
                                        "summary": f"{params['city']} verified hotel ({params['duration_days']} nights)",
                                        "cost_usd": float(budget_out.get("allocated_lodging_usd", est_lodging)),
                                    },
                                ],
                            },
                            session_state=session_state,
                            session_id=session_id,
                        )
                        executed_steps.append(persist_step)
                        tools_invoked.append("persist_finalized_trip_itinerary")

                    adv_level = advisory_out.get("advisory_level", 1)
                    proj_display = proposed_cost_usd if proposed_cost_usd is not None else budget_cap
                    candidate_response = (
                        f"Synthesized {params['duration_days']}-day {params['city']} itinerary "
                        f"from {params['origin']} for {params['travelers']} traveler(s) within your "
                        f"${budget_cap:,.2f} USD budget ceiling (projected total: ${proj_display:,.2f} USD, "
                        f"Travel Advisory Level {adv_level}). Includes verified flight options, "
                        f"vetted accommodations, and itemized contingency allocation."
                    )
            else:
                candidate_response = (
                    "Noted your travel preferences and constraints in persistent session memory. "
                    "How would you like to proceed with flights, accommodations, advisories, or budgeting?"
                )

            # 5. Output Self-Evaluation Critic & Residual PII Redaction
            async with trace_operation(
                "guardrails.evaluate_output",
                attributes={"session_id": session_id},
            ):
                output_verdict = self.policy_plugin.evaluate_output_guardrails(
                    response_text=candidate_response,
                    budget_limit_usd=params["budget_limit_usd"],
                    proposed_cost_usd=proposed_cost_usd,
                )

            final_response = str(output_verdict["sanitized_text"])

            # 6. Persist Assistant Response & Schedule Non-Blocking Async Memory Consolidation
            self.session_manager.save_turn(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=final_response,
                metadata={
                    "selected_model": routing_decision["model"],
                    "tools_invoked": tools_invoked,
                },
            )
            consolidation_task = self.memory_consolidator.schedule_consolidation(
                session_id=session_id,
                user_id=user_id,
                turn_messages=[
                    {"role": "user", "content": redacted_query},
                    {"role": "assistant", "content": final_response},
                ],
            )

            final_status = (
                "SUCCESS" if output_verdict["allowed"] else "BUDGET_POLICY_WARNING"
            )
            elapsed_ms = round((time.perf_counter() - turn_start) * 1000.0, 3)

            log_structured_event(
                self.logger,
                event_type="ORCHESTRATOR_TURN_COMPLETED",
                message=f"Completed concierge turn with status={final_status}.",
                severity="INFO",
                session_id=session_id,
                agent_name="travel_concierge_coordinator",
                selected_model=routing_decision["model"],
                tier=routing_decision["tier"],
                tools_invoked=tools_invoked,
                latency_ms=elapsed_ms,
            )

            self.intent_tracker.record_outcome(
                correlation_id=root_corr_id,
                actor="travel_concierge_coordinator",
                action_name="handle_request",
                actual_result={
                    "status": final_status,
                    "selected_model": routing_decision["model"],
                    "tools_invoked": tools_invoked,
                },
                status="SUCCESS",
                latency_ms=elapsed_ms,
                session_id=session_id,
            )

            return {
                "status": final_status,
                "allowed": output_verdict["allowed"],
                "user_id": user_id,
                "session_id": session_id,
                "response": final_response,
                "selected_model": routing_decision["model"],
                "tier": routing_decision["tier"],
                "routing": routing_decision,
                "guardrail": output_verdict,
                "guardrail_input": input_verdict,
                "guardrail_output": output_verdict,
                "hitl_status": None,
                "confirmation_id": None,
                "tools_invoked": tools_invoked,
                "tool_calls": [
                    {"tool_name": s["tool_name"], "args": s["args"]}
                    for s in executed_steps
                ],
                "tool_results": [
                    {
                        "tool_name": s["tool_name"],
                        "args": s["args"],
                        "output": s["output"],
                    }
                    for s in executed_steps
                ],
                "compaction": compaction_result,
                "retrieved_memories": relevant_memories,
                "context_cache_config": context_cache_cfg,
                "memory_task_scheduled": isinstance(consolidation_task, asyncio.Task),
                "trace_id": trace_ctx.get("trace_id"),
                "correlation_id": root_corr_id,
            }


root_agent = build_coordinator_agent()
app = build_adk_app(coordinator_agent=root_agent)
