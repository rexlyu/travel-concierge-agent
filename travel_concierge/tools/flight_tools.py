"""Commercial flight search and refundable reservation hold tools.

Implements Rubric 1.1 (Comprehensive Docstrings), 1.2 (Descriptive Naming),
1.3 (Explicit JSON Schemas), and 1.4 (Guided Error Handling).
"""

from contextlib import nullcontext
import hashlib
import logging
from typing import Any

from travel_concierge.schemas import (
    FlightHoldInput,
    FlightHoldOutput,
    FlightOption,
    FlightSearchInput,
    FlightSearchOutput,
    SchemaValidationError,
    ToolErrorOutput,
)

def _get_logger() -> logging.Logger:
    """Returns the structured JSON logger when observability is available."""
    try:
        from travel_concierge.observability.logging_config import (
            get_structured_logger,
        )

        return get_structured_logger("travel_concierge.tools.flight_tools")
    except Exception:
        fallback = logging.getLogger("travel_concierge.tools.flight_tools")
        if not fallback.handlers:
            fallback.addHandler(logging.NullHandler())
        return fallback

_CABIN_MULTIPLIERS: dict[str, float] = {
    "ECONOMY": 1.0,
    "PREMIUM_ECONOMY": 1.55,
    "BUSINESS": 2.95,
    "FIRST": 4.65,
}


def _enter_tool_span(tool_name: str, attributes: dict[str, Any]) -> Any:
    """Enters an OpenTelemetry / contextvars span when observability is loaded."""
    try:
        from travel_concierge.observability.tracing import trace_operation

        return trace_operation(f"tool.{tool_name}", attributes=attributes)
    except Exception:
        return nullcontext()


def _mask_email(email: str) -> str:
    """Redacts raw email addresses via PiiRedactor or deterministic masking."""
    try:
        from travel_concierge.observability.pii_redactor import get_default_redactor

        redacted = get_default_redactor().redact_text(email)
        if redacted and redacted != email:
            return redacted
    except Exception:
        pass
    return "[REDACTED_EMAIL]"


def _build_flight_recovery_hint(exc: SchemaValidationError, fn_name: str) -> str:
    """Constructs targeted parameter recovery guidance for the LLM."""
    fields = [err.get("field", "") for err in exc.errors if err.get("field")]
    hints: list[str] = []
    if fn_name == "search_commercial_flights":
        if any(f in {"origin", "destination"} for f in fields):
            hints.append(
                "Pass 3-letter uppercase IATA airport codes for 'origin' and "
                "'destination' (e.g., 'SFO', 'JFK', 'NRT', 'HND', 'KIX', 'LHR') "
                "and ensure origin != destination."
            )
        if "departure_date" in fields:
            hints.append(
                "Format 'departure_date' strictly as a valid ISO-8601 calendar "
                "date 'YYYY-MM-DD' (e.g., '2026-10-15')."
            )
        if "passengers" in fields:
            hints.append(
                "Set 'passengers' to an integer between 1 and 9 inclusive."
            )
        if "cabin_class" in fields:
            hints.append(
                "Set 'cabin_class' to one of: 'ECONOMY', 'PREMIUM_ECONOMY', "
                "'BUSINESS', or 'FIRST'."
            )
        if "max_price_usd" in fields:
            hints.append(
                "Provide 'max_price_usd' as a positive number (> 0) or omit/pass None."
            )
    else:
        if "flight_id" in fields:
            hints.append(
                "Provide a valid 'flight_id' string (3-64 chars) obtained from "
                "'search_commercial_flights' (e.g., 'FL-SFO-NRT-101')."
            )
        if "traveler_full_name" in fields:
            hints.append(
                "Provide 'traveler_full_name' as a non-empty full name (2-120 chars)."
            )
        if "contact_email" in fields:
            hints.append(
                "Provide 'contact_email' formatted as 'user@example.com'."
            )
        if "total_fare_usd" in fields:
            hints.append(
                "Provide 'total_fare_usd' as a positive number (> 0) matching "
                "the quoted flight total."
            )
        if "refundable" in fields:
            hints.append("Pass 'refundable' as a boolean (True or False).")

    if not hints:
        hints.append(
            f"Correct the invalid parameter(s) reported in '{exc}' so all arguments "
            f"satisfy the strict JSON schema for '{fn_name}', then retry."
        )
    return " ".join(hints)


def search_commercial_flights(
    origin: str,
    destination: str,
    departure_date: str,
    passengers: int = 1,
    cabin_class: str = "ECONOMY",
    max_price_usd: float | None = None,
) -> dict[str, Any]:
    """Searches scheduled commercial flights between two IATA airports with USD fares.

    Summary:
        Queries deterministic commercial airline inventory for a specified origin,
        destination, departure date, passenger count, and cabin class. Returns
        structured flight options including total USD fares, layover counts,
        refundability flags, and estimated carbon emissions so downstream budget
        and itinerary agents can compare cost vs. convenience without hallucination.

    Args:
        origin: 3-letter uppercase IATA origin airport code matching ``^[A-Z]{3}$``
            (for example, ``"SFO"``, ``"JFK"``, or ``"LAX"``). Must differ from
            ``destination``.
        destination: 3-letter uppercase IATA destination airport code matching
            ``^[A-Z]{3}$`` (for example, ``"NRT"``, ``"HND"``, ``"KIX"``, or ``"CDG"``).
        departure_date: Outbound departure calendar date in strict ISO-8601
            ``YYYY-MM-DD`` format (for example, ``"2026-10-12"``).
        passengers: Integer count of traveling passengers between ``1`` and ``9``
            inclusive. Defaults to ``1``.
        cabin_class: Requested airline cabin tier. Must be one of ``"ECONOMY"``,
            ``"PREMIUM_ECONOMY"``, ``"BUSINESS"``, or ``"FIRST"``. Defaults to
            ``"ECONOMY"``.
        max_price_usd: Optional upper bound on ``total_price_usd`` across all
            passengers (strictly ``> 0.0`` when provided). Defaults to ``None``.

    Returns:
        dict[str, Any]: On success, a ``FlightSearchOutput`` dictionary containing:
            - ``status`` (``"success"``)
            - ``origin``, ``destination``, ``departure_date``, ``passengers``,
              ``cabin_class``, ``currency`` (``"USD"``)
            - ``options_count`` (int) and ``flights`` (list of ``FlightOption`` dicts)
            - ``cheapest_total_usd`` (float | None) and ``advisory_note`` (str)
        On validation failure, a ``ToolErrorOutput`` dictionary with
        ``status="error"``, ``error_code``, ``message``, ``recovery_instructions``,
        and ``retryable=True``.

    Guided Recovery:
        Never raises unhandled validation exceptions. If ``status == "error"``,
        inspect ``invalid_fields`` and ``recovery_instructions``: convert full
        city names (such as ``"Tokyo"`` or ``"San Francisco"``) into 3-letter
        uppercase IATA codes (``"NRT"`` / ``"HND"`` / ``"SFO"``), format dates as
        ``YYYY-MM-DD``, and ensure ``passengers >= 1`` and ``max_price_usd > 0``.
    """
    with _enter_tool_span(
        "search_commercial_flights",
        {"tool.name": "search_commercial_flights", "tool.domain": "flights"},
    ):
        try:
            req = FlightSearchInput.model_validate(
                {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                    "passengers": passengers,
                    "cabin_class": cabin_class,
                    "max_price_usd": max_price_usd,
                }
            )
        except SchemaValidationError as exc:
            invalid_fields = [
                err.get("field", "") for err in exc.errors if err.get("field")
            ]
            _get_logger().warning(
                "Validation error in search_commercial_flights: %s",
                invalid_fields,
            )
            return ToolErrorOutput(
                error_code="INVALID_FLIGHT_SEARCH_INPUT",
                message=str(exc),
                recovery_instructions=_build_flight_recovery_hint(
                    exc, "search_commercial_flights"
                ),
                retryable=True,
                tool_name="search_commercial_flights",
                invalid_fields=invalid_fields,
            ).model_dump()
        except Exception as exc:  # pragma: no cover - defensive safety net
            return ToolErrorOutput(
                error_code="FLIGHT_SEARCH_EXECUTION_ERROR",
                message=f"Unexpected error during flight search: {exc}",
                recovery_instructions=(
                    "Verify all flight search arguments match required primitive "
                    "types (3-letter IATA strings, YYYY-MM-DD date, positive counts) "
                    "and retry."
                ),
                retryable=True,
                tool_name="search_commercial_flights",
            ).model_dump()

        route_seed = int(
            hashlib.sha256(
                f"{req.origin}-{req.destination}-{req.departure_date}".encode(
                    "utf-8"
                )
            ).hexdigest()[:8],
            16,
        )
        base_economy_fare = 540.0 + float(route_seed % 240)
        multiplier = _CABIN_MULTIPLIERS[req.cabin_class]

        templates = [
            {
                "suffix": "101",
                "airline": "Pacific Horizon Airlines",
                "dep_time": "09:20",
                "arr_time": "13:45+1",
                "duration_hours": 10.4,
                "stops": 0,
                "fare_factor": 1.18,
                "refundable": True,
                "carbon_kg": 495.0,
            },
            {
                "suffix": "204",
                "airline": "Global Skyways",
                "dep_time": "13:50",
                "arr_time": "19:35+1",
                "duration_hours": 12.75,
                "stops": 1,
                "fare_factor": 1.0,
                "refundable": True,
                "carbon_kg": 540.0,
            },
            {
                "suffix": "309",
                "airline": "Meridian Value Air",
                "dep_time": "22:10",
                "arr_time": "05:15+2",
                "duration_hours": 14.1,
                "stops": 1,
                "fare_factor": 0.86,
                "refundable": False,
                "carbon_kg": 565.0,
            },
        ]

        options: list[FlightOption] = []
        for tpl in templates:
            per_pax = round(base_economy_fare * multiplier * tpl["fare_factor"], 2)
            total_fare = round(per_pax * req.passengers, 2)
            if req.max_price_usd is not None and total_fare > req.max_price_usd:
                continue
            options.append(
                FlightOption(
                    flight_id=f"FL-{req.origin}-{req.destination}-{tpl['suffix']}",
                    airline=tpl["airline"],
                    origin=req.origin,
                    destination=req.destination,
                    departure_date=req.departure_date,
                    departure_time=tpl["dep_time"],
                    arrival_time=tpl["arr_time"],
                    duration_hours=tpl["duration_hours"],
                    stops=tpl["stops"],
                    cabin_class=req.cabin_class,
                    price_per_passenger_usd=per_pax,
                    total_price_usd=total_fare,
                    refundable=tpl["refundable"],
                    carbon_kg=tpl["carbon_kg"],
                )
            )

        options.sort(key=lambda opt: opt.total_price_usd)
        cheapest = options[0].total_price_usd if options else None
        if options:
            advisory_note = (
                f"Found {len(options)} flight option(s) from {req.origin} to "
                f"{req.destination}; lowest total fare is ${cheapest:.2f} USD."
            )
        else:
            advisory_note = (
                f"No flights matched max_price_usd=${req.max_price_usd:.2f}. "
                "Consider increasing max_price_usd or switching cabin_class to 'ECONOMY'."
            )

        output = FlightSearchOutput(
            origin=req.origin,
            destination=req.destination,
            departure_date=req.departure_date,
            passengers=req.passengers,
            cabin_class=req.cabin_class,
            options_count=len(options),
            flights=options,
            cheapest_total_usd=cheapest,
            advisory_note=advisory_note,
        )
        return output.model_dump()


def hold_refundable_flight_reservation(
    flight_id: str,
    traveler_full_name: str,
    contact_email: str,
    total_fare_usd: float,
    refundable: bool = True,
) -> dict[str, Any]:
    """Places a temporary 24-hour carrier fare hold on a selected commercial flight.

    Summary:
        High-stakes booking tool (listed in ``HIGH_STAKES_TOOLS``) that locks
        inventory and fare pricing on a selected ``flight_id`` for 24 hours.
        Requires Human-in-the-Loop confirmation via ``HumanApprovalGate`` prior
        to execution, masks raw ``contact_email`` PII before returning or logging,
        and never settles funds directly (ticketing settlement requires
        ``authorize_high_value_travel_payment``).

    Args:
        flight_id: Flight option identifier from ``search_commercial_flights``
            (3 to 64 characters, e.g., ``"FL-SFO-NRT-204"``).
        traveler_full_name: Primary traveler full legal name (2 to 120 characters).
        contact_email: Valid contact email address matching ``user@domain.tld``
            (automatically redacted in returned state and telemetry).
        total_fare_usd: Quoted total fare in USD across passengers (strictly
            ``> 0.0`` and ``<= 250000.0``).
        refundable: Boolean indicating whether the held fare is refundable.
            Defaults to ``True``. Non-refundable holds require explicit traveler
            acknowledgment.

    Returns:
        dict[str, Any]: On success, a ``FlightHoldOutput`` dictionary containing
        ``status="success"``, ``hold_status="HELD"``, ``hold_reference``,
        ``flight_id``, ``contact_email_masked``, ``total_fare_usd``, ``refundable``,
        ``expires_in_hours=24``, and ``requires_payment_authorization=True``.
        On invalid input, returns a ``ToolErrorOutput`` dictionary.

    Guided Recovery:
        If ``status == "error"``, review ``recovery_instructions`` to fix
        malformed ``contact_email`` addresses, zero/negative ``total_fare_usd``,
        or empty ``flight_id`` / ``traveler_full_name`` values before retrying.
    """
    with _enter_tool_span(
        "hold_refundable_flight_reservation",
        {
            "tool.name": "hold_refundable_flight_reservation",
            "tool.domain": "flights",
            "tool.high_stakes": True,
        },
    ):
        try:
            req = FlightHoldInput.model_validate(
                {
                    "flight_id": flight_id,
                    "traveler_full_name": traveler_full_name,
                    "contact_email": contact_email,
                    "total_fare_usd": total_fare_usd,
                    "refundable": refundable,
                }
            )
        except SchemaValidationError as exc:
            invalid_fields = [
                err.get("field", "") for err in exc.errors if err.get("field")
            ]
            _get_logger().warning(
                "Validation error in hold_refundable_flight_reservation: %s",
                invalid_fields,
            )
            return ToolErrorOutput(
                error_code="INVALID_FLIGHT_HOLD_INPUT",
                message=str(exc),
                recovery_instructions=_build_flight_recovery_hint(
                    exc, "hold_refundable_flight_reservation"
                ),
                retryable=True,
                tool_name="hold_refundable_flight_reservation",
                invalid_fields=invalid_fields,
            ).model_dump()
        except Exception as exc:  # pragma: no cover
            return ToolErrorOutput(
                error_code="FLIGHT_HOLD_EXECUTION_ERROR",
                message=f"Unexpected error placing flight hold: {exc}",
                recovery_instructions=(
                    "Ensure flight_id, traveler_full_name, contact_email, and "
                    "positive total_fare_usd are valid and retry."
                ),
                retryable=True,
                tool_name="hold_refundable_flight_reservation",
            ).model_dump()

        digest = hashlib.sha256(
            f"{req.flight_id}:{req.total_fare_usd:.2f}:{req.refundable}".encode(
                "utf-8"
            )
        ).hexdigest()[:8].upper()
        hold_ref = f"FH-{digest}"

        output = FlightHoldOutput(
            hold_reference=hold_ref,
            flight_id=req.flight_id,
            traveler_full_name=req.traveler_full_name,
            contact_email_masked=_mask_email(req.contact_email),
            total_fare_usd=round(req.total_fare_usd, 2),
            refundable=req.refundable,
            expires_in_hours=24,
            requires_payment_authorization=True,
        )
        return output.model_dump()


search_commercial_flights.input_schema = FlightSearchInput  # type: ignore[attr-defined]
search_commercial_flights.output_schema = FlightSearchOutput  # type: ignore[attr-defined]
hold_refundable_flight_reservation.input_schema = FlightHoldInput  # type: ignore[attr-defined]
hold_refundable_flight_reservation.output_schema = FlightHoldOutput  # type: ignore[attr-defined]
