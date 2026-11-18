"""Multi-agent orchestration package for the Travel & Budget Concierge."""

from travel_concierge.agents.coordinator import (
    TravelConciergeOrchestrator,
    app,
    build_adk_app,
    build_coordinator_agent,
    coordinator_after_model_callback,
    coordinator_before_model_callback,
    root_agent,
)
from travel_concierge.agents.model_router import (
    StrategicModelRouter,
    dynamic_model_routing_callback,
    get_default_model_router,
)
from travel_concierge.agents.sub_agents import (
    App,
    LlmAgent,
    ParallelAgent,
    SequentialAgent,
    build_sub_agents,
)

__all__ = [
    "App",
    "LlmAgent",
    "ParallelAgent",
    "SequentialAgent",
    "StrategicModelRouter",
    "TravelConciergeOrchestrator",
    "app",
    "build_adk_app",
    "build_coordinator_agent",
    "build_sub_agents",
    "coordinator_after_model_callback",
    "coordinator_before_model_callback",
    "dynamic_model_routing_callback",
    "get_default_model_router",
    "root_agent",
]
