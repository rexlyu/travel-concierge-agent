"""Unit tests for Category 1 (Tools & Schemas) and Rubric 5.3 (Secure Secret Management)."""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace
import unittest

from travel_concierge import schemas
from travel_concierge.config import (
    SecretManagerProvider,
    SecretResolutionError,
    get_config,
    set_secret_provider,
)
from travel_concierge.tools import (
    ALL_TOOLS,
    HIGH_STAKES_TOOLS,
    TOOL_REGISTRY,
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


class TestToolNamingAndDocstrings(unittest.TestCase):
    """Rubric 1.1 & 1.2: Verify descriptive verb-noun tool names and rich docstrings."""

    def test_all_nine_domain_tools_registered_with_descriptive_names(self) -> None:
        expected_names = {
            "search_commercial_flights",
            "hold_refundable_flight_reservation",
            "search_verified_accommodations",
            "reserve_lodging_booking",
            "calculate_trip_budget_allocation",
            "convert_currency_exchange_rate",
            "authorize_high_value_travel_payment",
            "retrieve_destination_travel_advisories",
            "persist_finalized_trip_itinerary",
        }
        self.assertEqual(set(TOOL_REGISTRY.keys()), expected_names)
        self.assertEqual(len(ALL_TOOLS), 9)
        self.assertEqual(
            set(HIGH_STAKES_TOOLS),
            {
                "authorize_high_value_travel_payment",
                "reserve_lodging_booking",
                "hold_refundable_flight_reservation",
            },
        )

    def test_every_tool_has_comprehensive_docstring_and_type_annotations(self) -> None:
        for name, fn in TOOL_REGISTRY.items():
            doc = inspect.getdoc(fn) or ""
            self.assertGreater(len(doc), 120, msg=f"Docstring too short on {name}")
            self.assertIn("Args:", doc, msg=f"Missing Args section on {name}")
            self.assertIn("Returns:", doc, msg=f"Missing Returns section on {name}")
            sig = inspect.signature(fn)
            self.assertNotEqual(sig.return_annotation, inspect.Signature.empty)


class TestExplicitJsonSchemas(unittest.TestCase):
    """Rubric 1.3: Verify strict Draft 2020-12 JSON schemas and constraint enforcement."""

    def test_json_schema_exports_disallow_additional_properties(self) -> None:
        model_classes = [
            schemas.FlightSearchInput,
            schemas.FlightSearchOutput,
            schemas.AccommodationSearchInput,
            schemas.AccommodationSearchOutput,
            schemas.BudgetAllocationInput,
            schemas.BudgetAllocationOutput,
            schemas.CurrencyConversionInput,
            schemas.CurrencyConversionOutput,
            schemas.PaymentAuthorizationInput,
            schemas.PaymentAuthorizationOutput,
            schemas.TravelAdvisoryInput,
            schemas.TravelAdvisoryOutput,
            schemas.ItineraryPersistenceInput,
            schemas.ItineraryPersistenceOutput,
            schemas.ToolErrorOutput,
        ]
        for cls in model_classes:
            js = cls.model_json_schema()
            self.assertEqual(js.get("type"), "object")
            self.assertIs(js.get("additionalProperties"), False, msg=f"{cls.__name__} must forbid extras")
            self.assertIn("properties", js)

    def test_strict_validation_rejects_extra_fields_bad_dates_and_negative_amounts(self) -> None:
        valid = schemas.FlightSearchInput.model_validate(
            {
                "origin": "SFO",
                "destination": "HND",
                "departure_date": "2026-10-10",
                "passengers": 2,
                "cabin_class": "ECONOMY",
            }
        )
        dumped = valid.model_dump()
        self.assertEqual(dumped["origin"], "SFO")

        with self.assertRaises((schemas.SchemaValidationError, ValueError)):
            schemas.FlightSearchInput.model_validate(
                {
                    "origin": "SFO",
                    "destination": "HND",
                    "departure_date": "2026-10-10",
                    "unexpected_hallucinated_field": True,
                }
            )

        with self.assertRaises((schemas.SchemaValidationError, ValueError)):
            schemas.FlightSearchInput.model_validate(
                {
                    "origin": "INVALID_IATA",
                    "destination": "HND",
                    "departure_date": "10/10/2026",
                }
            )

        with self.assertRaises((schemas.SchemaValidationError, ValueError)):
            schemas.CurrencyConversionInput.model_validate(
                {"amount": -25.0, "from_currency": "USD", "to_currency": "JPY"}
            )


class TestGuidedToolExecutionAndSecretManager(unittest.TestCase):
    """Rubric 1.4 & 5.3: Verify happy paths, guided error recovery, and SecretManagerProvider."""

    def tearDown(self) -> None:
        set_secret_provider(None)
        os.environ.pop("TRAVEL_PAYMENT_GATEWAY_TOKEN", None)

    def test_valid_tool_invocations_return_schema_validated_dicts(self) -> None:
        flights = search_commercial_flights("SFO", "NRT", "2026-10-10", passengers=2)
        self.assertNotEqual(flights.get("status"), "error")
        self.assertGreaterEqual(len(flights.get("flights", [])), 1)

        hold = hold_refundable_flight_reservation(
            flight_id="FL-100",
            traveler_full_name="Alex Rivera",
            contact_email="alex@example.org",
            total_fare_usd=780.0,
            refundable=True,
        )
        self.assertNotEqual(hold.get("status"), "error")

        hotels = search_verified_accommodations("Tokyo", "2026-10-10", "2026-10-15", guests=2)
        self.assertNotEqual(hotels.get("status"), "error")

        booking = reserve_lodging_booking("HTL-200", "Alex Rivera", nights=5, nightly_rate_usd=180.0)
        self.assertNotEqual(booking.get("status"), "error")

        budget = calculate_trip_budget_allocation(3500.0, duration_days=7, travelers=2)
        self.assertNotEqual(budget.get("status"), "error")
        self.assertIn("within_budget", budget)

        fx = convert_currency_exchange_rate(200.0, "USD", "JPY")
        self.assertNotEqual(fx.get("status"), "error")
        self.assertGreater(fx["converted_amount"], 200.0)

        adv = retrieve_destination_travel_advisories("JP", "Tokyo")
        self.assertNotEqual(adv.get("status"), "error")
        self.assertIn("advisory_level", adv)

        saved = persist_finalized_trip_itinerary(
            session_id="sess-save-1",
            traveler_id="trav-1",
            destination="Tokyo",
            total_cost_usd=2950.0,
            itinerary_items=[{"day": 1, "activity": "Arrival at Haneda"}],
        )
        self.assertNotEqual(saved.get("status"), "error")

    def test_invalid_inputs_return_guided_recovery_instructions_without_crashing(self) -> None:
        bad_flight = search_commercial_flights("SAN_FRANCISCO", "TOKYO", "bad-date", passengers=0)
        self.assertEqual(bad_flight["status"], "error")
        self.assertTrue(bad_flight.get("recovery_instructions"))
        self.assertTrue(bad_flight.get("retryable"))

        bad_fx = convert_currency_exchange_rate(-50.0, "NOPE", "USD")
        self.assertEqual(bad_fx["status"], "error")
        self.assertTrue(bad_fx.get("recovery_instructions"))

        bad_hotel = search_verified_accommodations("Kyoto", "2026-10-15", "2026-10-10", guests=2)
        self.assertEqual(bad_hotel["status"], "error")
        self.assertTrue(bad_hotel.get("recovery_instructions"))

    def test_secret_manager_provider_client_resolution_ttl_cache_and_missing_error(self) -> None:
        calls: list[str] = []

        class FakeSecretClient:
            def access_secret_version(self, request: dict[str, str]) -> SimpleNamespace:
                calls.append(request["name"])
                return SimpleNamespace(payload=SimpleNamespace(data=b"sm-gateway-token-xyz"))

        provider = SecretManagerProvider(
            project_id="demo-gcp-project",
            ttl_seconds=120.0,
            allow_env_fallback=False,
            client=FakeSecretClient(),
        )
        val_1 = provider.get_secret("travel-payment-gateway-token")
        val_2 = provider.get_secret("travel-payment-gateway-token")
        self.assertEqual(val_1, "sm-gateway-token-xyz")
        self.assertEqual(val_2, "sm-gateway-token-xyz")
        self.assertEqual(len(calls), 1)  # Served from TTL cache on second call
        self.assertEqual(
            calls[0],
            "projects/demo-gcp-project/secrets/travel-payment-gateway-token/versions/latest",
        )

        set_secret_provider(provider)
        pay_ok = authorize_high_value_travel_payment(
            booking_reference="BK-555",
            amount_usd=950.0,
            currency="USD",
            merchant_name="RailPassCo",
        )
        self.assertNotEqual(pay_ok.get("status"), "error")
        self.assertNotIn("sm-gateway-token-xyz", str(pay_ok))  # Raw secret never leaked in output

        empty_provider = SecretManagerProvider(
            project_id="",
            allow_env_fallback=False,
            client=None,
        )
        with self.assertRaises(SecretResolutionError):
            empty_provider.get_secret("travel-payment-gateway-token")

        cfg = get_config()
        self.assertEqual(cfg.fast_model, "gemini-2.5-flash")
        self.assertEqual(cfg.reasoning_model, "gemini-2.5-pro")


if __name__ == "__main__":
    unittest.main()
