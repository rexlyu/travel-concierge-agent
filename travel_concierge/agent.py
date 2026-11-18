"""ADK CLI entrypoint module (`adk web`, `adk run`, `adk eval`).

Exports `root_agent`, `app`, and `TravelConciergeOrchestrator` for seamless
discovery by Google Agent Development Kit tooling and programmatic callers.
"""

from travel_concierge.agents.coordinator import (
    TravelConciergeOrchestrator,
    app,
    root_agent,
)

__all__ = [
    "TravelConciergeOrchestrator",
    "app",
    "root_agent",
]
