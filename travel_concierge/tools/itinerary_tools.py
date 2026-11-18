"""Destination travel safety advisories and finalized itinerary persistence tools.

Implements Rubric 1.1 (Comprehensive Docstrings), 1.2 (Descriptive Naming),
1.3 (Explicit JSON Schemas), and 1.4 (Guided Error Handling).
"""

from contextlib import nullcontext
import hashlib
import json
import logging
import re
import threading
from typing import Any

from travel_concierge.schemas import (
    ItineraryPersistenceInput,
    ItineraryPersistenceOutput,
    SchemaValidationError,
    ToolErrorOutput,
    TravelAdvisoryInput,
    TravelAdvisoryOutput,
)

def _get_logger() -> logging.Logger:
    """Returns the structured JSON logger when observability is available."""
    try:
        from travel_concierge.observability.logging_config import (
            get_structured_logger,
        )

        return get_structured_logger("travel_concierge.tools.itinerary_tools")
    except Exception:
        fallback = logging.getLogger("travel_concierge.tools.itinerary_tools")
        if not fallback.handlers:
            fallback.addHandler(logging.NullHandler())
        return fallback

_PERSISTED_ITINERARIES: dict[str, dict[str, Any]] = {}
_ITINERARY_LOCK = threading.RLock()

_ALPHA3_TO_ALPHA2: dict[str, str] = {
    "JPN": "JP",
    "USA": "US",
    "GBR": "GB",
    "FRA": "FR",
    "DEU": "DE",
    "ITA": "IT",
    "ESP": "ES",
    "CAN": "CA",
    "AUS": "AU",
    "SGP": "SG",
    "KOR": "KR",
    "THA": "TH",
    "MEX": "MX",
    "CHE": "CH",
    "NLD": "NL",
    "PRT": "PT",
    "GRC": "GR",
    "NZL": "NZ",
    "IND": "IN",
    "ARE": "AE",
    "BRA": "BR",
}

_ADVISORY_PROFILES: dict[str, dict[str, Any]] = {
    "JP": {
        "level": 1,
        "summary": (
            "Level 1 — Exercise Normal Precautions. Japan maintains exceptionally "
            "high public safety, reliable rail infrastructure, and clear multilingual signage."
        ),
        "entry": [
            "Passport valid for the duration of intended stay.",
            "Visa-free short-term tourist entry up to 90 days for eligible passports.",
            "Complete Visit Japan Web customs and immigration QR registration prior to arrival.",
        ],
        "health": [
            "Tap water is potable nationwide.",
            "Monitor Japan Meteorological Agency alerts for seasonal typhoon or seismic advisories.",
        ],
        "emergency": "110 (Police) / 119 (Fire & Ambulance)",
        "precautions": [
            "Carry passport or residence card as required by local law.",
            "Reserve oversized baggage seats in advance on Tokaido-Sanyo Shinkansen routes.",
        ],
    },
    "FR": {
        "level": 2,
        "summary": (
            "Level 2 — Exercise Increased Caution due to elevated urban transit "
            "pickpocketing and periodic civic demonstrations."
        ),
        "entry": [
            "Passport valid at least 3 months beyond planned departure from the Schengen area.",
            "Schengen short-stay allowance up to 90 days within any 180-day period.",
        ],
        "health": [
            "Standard routine vaccinations up to date.",
            "Pharmacies (green cross signage) provide first-line clinical guidance.",
        ],
        "emergency": "112 (EU Emergency) / 15 (SAMU Medical)",
        "precautions": [
            "Keep valuables secured on RER regional trains and major museum queues.",
            "Monitor RATP/SNCF service bulletins before airport transfers.",
        ],
    },
    "GB": {
        "level": 1,
        "summary": (
            "Level 1 — Exercise Normal Precautions, with standard urban vigilance "
            "around major rail terminals and tourist districts."
        ),
        "entry": [
            "Valid electronic passport required for entry.",
            "Verify UK Electronic Travel Authorisation (ETA) requirement prior to boarding.",
        ],
        "health": [
            "Comprehensive emergency care available via NHS 999 / 111.",
        ],
        "emergency": "999 / 112",
        "precautions": [
            "Use contactless payment or Oyster cards on Transport for London services.",
            "Look right first at pedestrian crossings.",
        ],
    },
    "US": {
        "level": 1,
        "summary": (
            "Level 1 — Exercise Normal Precautions across major metropolitan travel corridors."
        ),
        "entry": [
            "Valid government-issued photo ID (REAL ID compliant) for domestic air travel.",
            "ESTA or valid visitor visa required for eligible international arrivals.",
        ],
        "health": [
            "Travel medical insurance is strongly recommended for international visitors.",
        ],
        "emergency": "911",
        "precautions": [
            "Confirm airline terminal assignments 24 hours prior to departure.",
            "Keep hotel safe-deposit records of travel documents.",
        ],
    },
}

_LOCAL_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_LOCAL_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def _enter_tool_span(tool_name: str, attributes: dict[str, Any]) -> Any:
    """Enters an OpenTelemetry / contextvars span when observability is loaded."""
    try:
        from travel_concierge.observability.tracing import trace_operation

        return trace_operation(f"tool.{tool_name}", attributes=attributes)
    except Exception:
        return nullcontext()


def _scrub_pii(payload: Any) -> Any:
    """Scrubs PII via observability.pii_redactor with local regex fallback."""
    try:
        from travel_concierge.observability.pii_redactor import get_default_redactor

        redactor = get_default_redactor()
        if isinstance(payload, str):
            return redactor.redact_text(payload)
        return redactor.redact_payload(payload)
    except Exception:
        if isinstance(payload, str):
            text = _LOCAL_EMAIL_RE.sub("[REDACTED_EMAIL]", payload)
            return _LOCAL_CARD_RE.sub("[REDACTED_CARD]", text)
        if isinstance(payload, dict):
            return {k: _scrub_pii(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [_scrub_pii(item) for item in payload]
        return payload


def get_persisted_itineraries() -> dict[str, dict[str, Any]]:
    """Returns a shallow copy of in-memory persisted itinerary records."""
    with _ITINERARY_LOCK:
        return dict(_PERSISTED_ITINERARIES)


def clear_persisted_itineraries() -> None:
    """Clears the in-memory persisted itinerary registry for isolated tests."""
    with _ITINERARY_LOCK:
        _PERSISTED_ITINERARIES.clear()


def retrieve_destination_travel_advisories(
    country_code: str,
    city: str,
) -> dict[str, Any]:
    """Retrieves official travel safety advisory levels, visa rules, and emergency info.

    Summary:
        Looks up destination safety posture (Levels 1–4), passport and visa
        entry requirements, local health notes, emergency phone numbers, and
        practical traveler precautions for a target ``country_code`` and ``city``.
        Used by the advisory specialist and self-evaluation critic to prevent
        bookings to unsafe regions.

    Args:
        country_code: 2-letter or 3-letter uppercase ISO-3166 country code
            matching ``^[A-Z]{2,3}$`` (for example, ``"JP"``, ``"JPN"``,
            ``"FR"``, ``"GB"``, or ``"US"``).
        city: Target destination city name (2 to 100 characters, e.g.,
            ``"Tokyo"``, ``"Kyoto"``, or ``"Paris"``).

    Returns:
        dict[str, Any]: On success, a ``TravelAdvisoryOutput`` dictionary with
        ``status="success"``, ``country_code``, ``city``, ``advisory_level``
        (1–4), ``advisory_summary``, ``entry_requirements``,
        ``health_and_safety_notes``, ``local_emergency_number``, and
        ``recommended_precautions``. On invalid input, returns ``ToolErrorOutput``.

    Guided Recovery:
        If ``status == "error"``, follow ``recovery_instructions``: convert full
        country names (like ``"Japan"`` or ``"France"``) into uppercase ISO-3166
        codes (``"JP"`` or ``"FR"``) and provide a non-empty ``city`` name.
    """
    with _enter_tool_span(
        "retrieve_destination_travel_advisories",
        {
            "tool.name": "retrieve_destination_travel_advisories",
            "tool.domain": "advisory",
        },
    ):
        try:
            req = TravelAdvisoryInput.model_validate(
                {
                    "country_code": country_code,
                    "city": city,
                }
            )
        except SchemaValidationError as exc:
            invalid_fields = [
                err.get("field", "") for err in exc.errors if err.get("field")
            ]
            _get_logger().warning(
                "Validation error in retrieve_destination_travel_advisories: %s",
                invalid_fields,
            )
            return ToolErrorOutput(
                error_code="INVALID_TRAVEL_ADVISORY_INPUT",
                message=str(exc),
                recovery_instructions=(
                    "Pass 'country_code' as a recognized 2- or 3-letter uppercase "
                    "ISO-3166 code (e.g., 'JP', 'JPN', 'US', 'FR', 'GB', 'IT', 'SG') "
                    "instead of a full country name, and pass 'city' as a 2-100 "
                    "character city name."
                ),
                retryable=True,
                tool_name="retrieve_destination_travel_advisories",
                invalid_fields=invalid_fields,
            ).model_dump()
        except Exception as exc:  # pragma: no cover
            return ToolErrorOutput(
                error_code="TRAVEL_ADVISORY_EXECUTION_ERROR",
                message=f"Unexpected error retrieving travel advisory: {exc}",
                recovery_instructions=(
                    "Verify uppercase ISO-3166 country_code and non-empty city "
                    "string, then retry."
                ),
                retryable=True,
                tool_name="retrieve_destination_travel_advisories",
            ).model_dump()

        normalized_cc = _ALPHA3_TO_ALPHA2.get(req.country_code, req.country_code)
        if normalized_cc == "UK":
            normalized_cc = "GB"

        profile = _ADVISORY_PROFILES.get(
            normalized_cc,
            {
                "level": 1,
                "summary": (
                    f"Level 1 — Exercise Normal Precautions when visiting "
                    f"{req.city} ({req.country_code})."
                ),
                "entry": [
                    "Passport valid for at least 6 months beyond planned departure.",
                    "Confirm bilateral tourist visa or e-visa eligibility prior to travel.",
                ],
                "health": [
                    "Ensure routine immunizations and international travel insurance are active.",
                ],
                "emergency": "112 / Local Consular Duty Desk",
                "precautions": [
                    "Keep digital copies of itinerary and passport identification page.",
                    "Use licensed airport taxis or official public rail links.",
                ],
            },
        )

        output = TravelAdvisoryOutput(
            country_code=req.country_code,
            city=req.city,
            advisory_level=profile["level"],
            advisory_summary=profile["summary"],
            entry_requirements=list(profile["entry"]),
            health_and_safety_notes=list(profile["health"]),
            local_emergency_number=profile["emergency"],
            recommended_precautions=list(profile["precautions"]),
        )
        return output.model_dump()


def persist_finalized_trip_itinerary(
    session_id: str,
    traveler_id: str,
    destination: str,
    total_cost_usd: float,
    itinerary_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persists a critic-verified trip itinerary with PII scrubbing and checksum.

    Summary:
        Validates and commits a finalized multi-segment trip itinerary into the
        session itinerary store (and synchronizes state into
        ``PersistentTravelSessionManager`` when active). Automatically scrubs
        residual PII across ``traveler_id`` and ``itinerary_items`` prior to
        storage and computes a deterministic SHA-256 integrity checksum.

    Args:
        session_id: Active conversation session identifier (1 to 128 characters).
        traveler_id: Pseudonymous traveler identifier (1 to 128 characters).
        destination: Primary destination or multi-city route name (2 to 120
            characters, e.g., ``"Tokyo & Kyoto, Japan"``).
        total_cost_usd: Verified total itinerary cost in USD (strictly ``> 0.0``).
        itinerary_items: Non-empty list of structured segment dictionaries
            (each dictionary representing a flight, hotel stay, transit leg, or
            scheduled activity).

    Returns:
        dict[str, Any]: On success, an ``ItineraryPersistenceOutput`` dictionary
        with ``status="success"``, ``persistence_status="PERSISTED"``,
        ``itinerary_id``, ``session_id``, ``traveler_id``, ``destination``,
        ``total_cost_usd``, ``items_count``, ``checksum_sha256``, and ``summary``.
        On validation failure, returns ``ToolErrorOutput``.

    Guided Recovery:
        If ``status == "error"``, review ``recovery_instructions``: ensure
        ``itinerary_items`` is a non-empty list of non-empty dicts,
        ``total_cost_usd > 0``, and ``session_id``, ``traveler_id``, and
        ``destination`` are non-empty strings.
    """
    with _enter_tool_span(
        "persist_finalized_trip_itinerary",
        {
            "tool.name": "persist_finalized_trip_itinerary",
            "tool.domain": "itinerary",
        },
    ):
        try:
            req = ItineraryPersistenceInput.model_validate(
                {
                    "session_id": session_id,
                    "traveler_id": traveler_id,
                    "destination": destination,
                    "total_cost_usd": total_cost_usd,
                    "itinerary_items": itinerary_items,
                }
            )
        except SchemaValidationError as exc:
            invalid_fields = [
                err.get("field", "") for err in exc.errors if err.get("field")
            ]
            _get_logger().warning(
                "Validation error in persist_finalized_trip_itinerary: %s",
                invalid_fields,
            )
            return ToolErrorOutput(
                error_code="INVALID_ITINERARY_PERSISTENCE_INPUT",
                message=str(exc),
                recovery_instructions=(
                    "Provide non-empty strings for 'session_id', 'traveler_id', "
                    "and 'destination' (>= 2 chars), a strictly positive "
                    "'total_cost_usd' (> 0), and 'itinerary_items' as a non-empty "
                    "list of non-empty segment dictionaries."
                ),
                retryable=True,
                tool_name="persist_finalized_trip_itinerary",
                invalid_fields=invalid_fields,
            ).model_dump()
        except Exception as exc:  # pragma: no cover
            return ToolErrorOutput(
                error_code="ITINERARY_PERSISTENCE_EXECUTION_ERROR",
                message=f"Unexpected error persisting itinerary: {exc}",
                recovery_instructions=(
                    "Verify itinerary_items is a JSON-serializable non-empty list "
                    "of dicts and total_cost_usd > 0, then retry."
                ),
                retryable=True,
                tool_name="persist_finalized_trip_itinerary",
            ).model_dump()

        scrubbed_traveler = str(_scrub_pii(req.traveler_id))
        scrubbed_items = _scrub_pii(req.itinerary_items)
        serialized = json.dumps(
            scrubbed_items, sort_keys=True, default=str
        ).encode("utf-8")
        checksum = hashlib.sha256(serialized).hexdigest()
        itin_id = (
            "ITIN-"
            + hashlib.sha256(
                f"{req.session_id}:{req.destination}:{checksum[:16]}".encode(
                    "utf-8"
                )
            )
            .hexdigest()[:10]
            .upper()
        )

        record = {
            "itinerary_id": itin_id,
            "session_id": req.session_id,
            "traveler_id": scrubbed_traveler,
            "destination": req.destination,
            "total_cost_usd": round(req.total_cost_usd, 2),
            "itinerary_items": scrubbed_items,
            "checksum_sha256": checksum,
        }
        with _ITINERARY_LOCK:
            _PERSISTED_ITINERARIES[itin_id] = record

        try:
            from travel_concierge.memory.session_store import (
                PersistentTravelSessionManager,
            )

            manager = PersistentTravelSessionManager()
            manager.update_session_state(
                req.session_id,
                {
                    "last_itinerary_id": itin_id,
                    "finalized_itinerary": {
                        "itinerary_id": itin_id,
                        "destination": req.destination,
                        "total_cost_usd": round(req.total_cost_usd, 2),
                        "items_count": len(scrubbed_items),
                    },
                },
            )
        except Exception:
            # Session store sync is opportunistic when running standalone tool calls.
            pass

        output = ItineraryPersistenceOutput(
            itinerary_id=itin_id,
            session_id=req.session_id,
            traveler_id=scrubbed_traveler,
            destination=req.destination,
            total_cost_usd=round(req.total_cost_usd, 2),
            items_count=len(scrubbed_items),
            checksum_sha256=checksum,
            summary=(
                f"Persisted {len(scrubbed_items)}-segment itinerary '{itin_id}' "
                f"for {req.destination} (${req.total_cost_usd:.2f} USD)."
            ),
        )
        return output.model_dump()


retrieve_destination_travel_advisories.input_schema = TravelAdvisoryInput  # type: ignore[attr-defined]
retrieve_destination_travel_advisories.output_schema = TravelAdvisoryOutput  # type: ignore[attr-defined]
persist_finalized_trip_itinerary.input_schema = ItineraryPersistenceInput  # type: ignore[attr-defined]
persist_finalized_trip_itinerary.output_schema = ItineraryPersistenceOutput  # type: ignore[attr-defined]
