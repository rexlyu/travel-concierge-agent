"""Domain tool exports, registry, and high-stakes governance metadata.

Re-exports all 9 Travel & Budget Concierge domain tools alongside ``ALL_TOOLS``,
``TOOL_REGISTRY``, and ``HIGH_STAKES_TOOLS``.
"""

from collections.abc import Callable
from typing import Any

from travel_concierge.tools.budget_tools import (
    authorize_high_value_travel_payment,
    calculate_trip_budget_allocation,
    convert_currency_exchange_rate,
)
from travel_concierge.tools.flight_tools import (
    hold_refundable_flight_reservation,
    search_commercial_flights,
)
from travel_concierge.tools.itinerary_tools import (
    clear_persisted_itineraries,
    get_persisted_itineraries,
    persist_finalized_trip_itinerary,
    retrieve_destination_travel_advisories,
)
from travel_concierge.tools.lodging_tools import (
    reserve_lodging_booking,
    search_verified_accommodations,
)

ALL_TOOLS: list[Callable[..., dict[str, Any]]] = [
    search_commercial_flights,
    hold_refundable_flight_reservation,
    search_verified_accommodations,
    reserve_lodging_booking,
    calculate_trip_budget_allocation,
    convert_currency_exchange_rate,
    authorize_high_value_travel_payment,
    retrieve_destination_travel_advisories,
    persist_finalized_trip_itinerary,
]

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    tool_fn.__name__: tool_fn for tool_fn in ALL_TOOLS
}

HIGH_STAKES_TOOLS: set[str] = {
    "authorize_high_value_travel_payment",
    "reserve_lodging_booking",
    "hold_refundable_flight_reservation",
}

__all__ = [
    "ALL_TOOLS",
    "HIGH_STAKES_TOOLS",
    "TOOL_REGISTRY",
    "authorize_high_value_travel_payment",
    "calculate_trip_budget_allocation",
    "clear_persisted_itineraries",
    "convert_currency_exchange_rate",
    "get_persisted_itineraries",
    "hold_refundable_flight_reservation",
    "persist_finalized_trip_itinerary",
    "reserve_lodging_booking",
    "retrieve_destination_travel_advisories",
    "search_commercial_flights",
    "search_verified_accommodations",
]
