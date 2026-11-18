"""Multi-agent specialist definitions for the Travel & Budget Concierge.

Implements Rubric 3.1 (Multi-Agent Patterns) using Google ADK primitives:
- `ParallelAgent` (`parallel_research_squad`) concurrently executing:
  * `flight_specialist_agent` (`LlmAgent`, `gemini-2.5-flash`)
  * `lodging_specialist_agent` (`LlmAgent`, `gemini-2.5-flash`)
  * `advisory_specialist_agent` (`LlmAgent`, `gemini-2.5-flash`)
- `SequentialAgent` (`itinerary_planning_pipeline`) chaining:
  1. `parallel_research_squad`
  2. `budget_analyst_agent` (`LlmAgent`, `gemini-2.5-pro` + HITL tool hook)
  3. `itinerary_critic_agent` (`LlmAgent`, `gemini-2.5-pro` + output critic hook)
"""

from __future__ import annotations

from typing import Any

from travel_concierge.config import TravelConciergeConfig, get_config
from travel_concierge.guardrails.hitl_hooks import hitl_before_tool_callback
from travel_concierge.guardrails.policy_plugin import guardrail_after_model_callback
from travel_concierge.prompts import (
    ADVISORY_SPECIALIST_INSTRUCTION,
    BUDGET_ANALYST_INSTRUCTION,
    CRITIC_INSTRUCTION,
    FLIGHT_SPECIALIST_INSTRUCTION,
    LODGING_SPECIALIST_INSTRUCTION,
)

try:
    from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
    from google.adk.apps.app import App

    ADK_AGENTS_AVAILABLE = True
except ImportError:
    ADK_AGENTS_AVAILABLE = False

    class LlmAgent:  # type: ignore[no-redef]
        """Lightweight offline-compatible fallback for google.adk.agents.LlmAgent."""

        def __init__(
            self,
            *,
            name: str,
            model: str = "gemini-2.5-flash",
            description: str = "",
            instruction: str = "",
            tools: list[Any] | None = None,
            sub_agents: list[Any] | None = None,
            output_key: str | None = None,
            before_agent_callback: Any = None,
            after_agent_callback: Any = None,
            before_model_callback: Any = None,
            after_model_callback: Any = None,
            before_tool_callback: Any = None,
            after_tool_callback: Any = None,
        ) -> None:
            self.name = name
            self.model = model
            self.description = description
            self.instruction = instruction
            self.tools = list(tools) if tools is not None else []
            self.sub_agents = list(sub_agents) if sub_agents is not None else []
            self.output_key = output_key
            self.before_agent_callback = before_agent_callback
            self.after_agent_callback = after_agent_callback
            self.before_model_callback = before_model_callback
            self.after_model_callback = after_model_callback
            self.before_tool_callback = before_tool_callback
            self.after_tool_callback = after_tool_callback

    class ParallelAgent:  # type: ignore[no-redef]
        """Lightweight offline-compatible fallback for google.adk.agents.ParallelAgent."""

        def __init__(
            self,
            *,
            name: str,
            description: str = "",
            sub_agents: list[Any] | None = None,
            before_agent_callback: Any = None,
            after_agent_callback: Any = None,
        ) -> None:
            self.name = name
            self.description = description
            self.sub_agents = list(sub_agents) if sub_agents is not None else []
            self.before_agent_callback = before_agent_callback
            self.after_agent_callback = after_agent_callback

    class SequentialAgent:  # type: ignore[no-redef]
        """Lightweight offline-compatible fallback for google.adk.agents.SequentialAgent."""

        def __init__(
            self,
            *,
            name: str,
            description: str = "",
            sub_agents: list[Any] | None = None,
            before_agent_callback: Any = None,
            after_agent_callback: Any = None,
        ) -> None:
            self.name = name
            self.description = description
            self.sub_agents = list(sub_agents) if sub_agents is not None else []
            self.before_agent_callback = before_agent_callback
            self.after_agent_callback = after_agent_callback

    class App:  # type: ignore[no-redef]
        """Lightweight offline-compatible fallback for google.adk.apps.app.App."""

        def __init__(
            self,
            *,
            name: str,
            root_agent: Any,
            plugins: list[Any] | None = None,
            events_compaction_config: Any = None,
            context_cache_config: Any = None,
        ) -> None:
            self.name = name
            self.root_agent = root_agent
            self.plugins = list(plugins) if plugins is not None else []
            self.events_compaction_config = events_compaction_config
            self.context_cache_config = context_cache_config


def _load_specialist_tools() -> dict[str, list[Any]]:
    """Resolve domain tool functions lazily from `travel_concierge.tools`."""
    try:
        from travel_concierge.tools import (
            authorize_high_value_travel_payment,
            calculate_trip_budget_allocation,
            convert_currency_exchange_rate,
            hold_refundable_flight_reservation,
            persist_finalized_trip_itinerary,
            reserve_lodging_booking,
            retrieve_destination_travel_advisories,
            search_commercial_flights,
            search_verified_accommodations,
        )

        return {
            "flight": [
                search_commercial_flights,
                hold_refundable_flight_reservation,
            ],
            "lodging": [
                search_verified_accommodations,
                reserve_lodging_booking,
            ],
            "advisory": [retrieve_destination_travel_advisories],
            "budget": [
                calculate_trip_budget_allocation,
                convert_currency_exchange_rate,
                authorize_high_value_travel_payment,
            ],
            "critic": [persist_finalized_trip_itinerary],
        }
    except Exception:
        return {
            "flight": [],
            "lodging": [],
            "advisory": [],
            "budget": [],
            "critic": [],
        }


def build_sub_agents(config: TravelConciergeConfig | None = None) -> dict[str, Any]:
    """Construct the multi-agent specialist hierarchy for ADK orchestration.

    Creates fresh agent instances per invocation so ADK parent-agent ownership
    pointers never collide across independent `App` or test instantiations.

    Args:
        config: Optional `TravelConciergeConfig` overriding model identifiers.

    Returns:
        Dictionary mapping canonical specialist and workflow names to ADK agents.
    """
    cfg = config or get_config()
    tools_by_role = _load_specialist_tools()

    flight_specialist_agent = LlmAgent(
        name="flight_specialist_agent",
        model=cfg.fast_model,
        description=(
            "Researches commercial airline routes, compares fares, and places "
            "refundable fare holds subject to human approval gates."
        ),
        instruction=FLIGHT_SPECIALIST_INSTRUCTION,
        tools=tools_by_role["flight"],
        before_tool_callback=hitl_before_tool_callback,
        output_key="flight_research_results",
    )

    lodging_specialist_agent = LlmAgent(
        name="lodging_specialist_agent",
        model=cfg.fast_model,
        description=(
            "Searches verified hotels and accommodations and reserves stays "
            "subject to budget ceilings and human approval gates."
        ),
        instruction=LODGING_SPECIALIST_INSTRUCTION,
        tools=tools_by_role["lodging"],
        before_tool_callback=hitl_before_tool_callback,
        output_key="lodging_research_results",
    )

    advisory_specialist_agent = LlmAgent(
        name="advisory_specialist_agent",
        model=cfg.fast_model,
        description=(
            "Retrieves official destination safety advisories, visa rules, and "
            "local emergency guidance."
        ),
        instruction=ADVISORY_SPECIALIST_INSTRUCTION,
        tools=tools_by_role["advisory"],
        output_key="advisory_research_results",
    )

    parallel_research_squad = ParallelAgent(
        name="parallel_research_squad",
        description=(
            "Concurrently executes flight, accommodation, and travel advisory "
            "research specialists to minimize end-to-end planning latency."
        ),
        sub_agents=[
            flight_specialist_agent,
            lodging_specialist_agent,
            advisory_specialist_agent,
        ],
    )

    budget_analyst_agent = LlmAgent(
        name="budget_analyst_agent",
        model=cfg.reasoning_model,
        description=(
            "Performs itemized trip budget allocation, currency conversion, "
            "trade-off optimization, and HITL-gated payment authorization."
        ),
        instruction=BUDGET_ANALYST_INSTRUCTION,
        tools=tools_by_role["budget"],
        before_tool_callback=hitl_before_tool_callback,
        output_key="budget_analysis_results",
    )

    itinerary_critic_agent = LlmAgent(
        name="itinerary_critic_agent",
        model=cfg.reasoning_model,
        description=(
            "Audits synthesized itineraries for budget compliance, Level-4 "
            "advisory blocks, and PII hygiene before persisting finalized plans."
        ),
        instruction=CRITIC_INSTRUCTION,
        tools=tools_by_role["critic"],
        after_model_callback=guardrail_after_model_callback,
        output_key="final_itinerary_review",
    )

    itinerary_planning_pipeline = SequentialAgent(
        name="itinerary_planning_pipeline",
        description=(
            "Three-stage sequential pipeline chaining concurrent research, "
            "financial budget synthesis, and self-evaluation critic verification."
        ),
        sub_agents=[
            parallel_research_squad,
            budget_analyst_agent,
            itinerary_critic_agent,
        ],
    )

    return {
        "flight_specialist_agent": flight_specialist_agent,
        "lodging_specialist_agent": lodging_specialist_agent,
        "advisory_specialist_agent": advisory_specialist_agent,
        "parallel_research_squad": parallel_research_squad,
        "budget_analyst_agent": budget_analyst_agent,
        "itinerary_critic_agent": itinerary_critic_agent,
        "itinerary_planning_pipeline": itinerary_planning_pipeline,
    }
