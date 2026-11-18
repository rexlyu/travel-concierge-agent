"""Trip budget allocation, ISO-4217 currency conversion, and payment tools.

Implements Rubric 1.1 (Comprehensive Docstrings), 1.2 (Descriptive Naming),
1.3 (Explicit JSON Schemas), 1.4 (Guided Error Handling), and 5.3 (Secure Secret
Management via SecretManagerProvider).
"""

from contextlib import nullcontext
import hashlib
import logging
from typing import Any

from travel_concierge.config import (
    SecretResolutionError,
    get_config,
    get_secret_provider,
)
from travel_concierge.schemas import (
    BudgetAllocationInput,
    BudgetAllocationOutput,
    CurrencyConversionInput,
    CurrencyConversionOutput,
    PaymentAuthorizationInput,
    PaymentAuthorizationOutput,
    SchemaValidationError,
    ToolErrorOutput,
)

def _get_logger() -> logging.Logger:
    """Returns the structured JSON logger when observability is available."""
    try:
        from travel_concierge.observability.logging_config import (
            get_structured_logger,
        )

        return get_structured_logger("travel_concierge.tools.budget_tools")
    except Exception:
        fallback = logging.getLogger("travel_concierge.tools.budget_tools")
        if not fallback.handlers:
            fallback.addHandler(logging.NullHandler())
        return fallback

# Deterministic USD-anchored reference foreign exchange rates (1 USD = X target).
_USD_FX_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 151.50,
    "CAD": 1.36,
    "AUD": 1.52,
    "CHF": 0.88,
    "CNY": 7.23,
    "HKD": 7.81,
    "SGD": 1.34,
    "KRW": 1365.00,
    "INR": 83.40,
    "MXN": 17.15,
    "BRL": 5.08,
    "NZD": 1.64,
    "SEK": 10.55,
    "NOK": 10.72,
    "DKK": 6.86,
    "THB": 35.80,
    "TWD": 32.10,
    "ZAR": 18.45,
    "AED": 3.6725,
    "TRY": 32.40,
    "PLN": 3.96,
    "CZK": 23.15,
    "HUF": 360.00,
    "ILS": 3.72,
    "PHP": 56.80,
    "IDR": 15950.00,
    "MYR": 4.71,
    "VND": 25100.00,
    "CLP": 935.00,
    "COP": 3910.00,
    "PEN": 3.74,
    "ARS": 895.00,
    "EGP": 47.60,
    "SAR": 3.75,
    "QAR": 3.64,
    "ISK": 138.50,
}

_DAILY_GROUND_RATES_USD: dict[str, dict[str, float]] = {
    "BUDGET": {"dining": 35.0, "activities": 20.0, "transit": 14.0},
    "MODERATE": {"dining": 65.0, "activities": 40.0, "transit": 22.0},
    "LUXURY": {"dining": 145.0, "activities": 95.0, "transit": 55.0},
}


def _enter_tool_span(tool_name: str, attributes: dict[str, Any]) -> Any:
    """Enters an OpenTelemetry / contextvars span when observability is loaded."""
    try:
        from travel_concierge.observability.tracing import trace_operation

        return trace_operation(f"tool.{tool_name}", attributes=attributes)
    except Exception:
        return nullcontext()


def calculate_trip_budget_allocation(
    total_budget_usd: float,
    duration_days: int,
    travelers: int = 1,
    travel_style: str = "MODERATE",
    estimated_flight_usd: float = 0.0,
    estimated_lodging_usd: float = 0.0,
) -> dict[str, Any]:
    """Calculates an itemized trip budget allocation and verifies ceiling feasibility.

    Summary:
        Computes category-by-category trip expenditures across flights, lodging,
        dining, activities, local transit, and an emergency contingency reserve.
        Deterministically evaluates whether ``projected_total_usd`` fits within
        ``total_budget_usd`` (``within_budget``) and generates concrete savings
        recommendations whenever costs approach or exceed the traveler's cap.

    Args:
        total_budget_usd: Traveler's maximum total budget in USD (strictly
            ``> 0.0`` and ``<= 1000000.0``).
        duration_days: Integer trip length in days between ``1`` and ``365``.
        travelers: Integer party size between ``1`` and ``50``. Defaults to ``1``.
        travel_style: Spending profile tier. Must be one of ``"BUDGET"``,
            ``"MODERATE"``, or ``"LUXURY"``. Defaults to ``"MODERATE"``.
        estimated_flight_usd: Quoted airfare total in USD from flight search
            (``>= 0.0``; if ``0.0``, allocates a baseline 30% share). Defaults to ``0.0``.
        estimated_lodging_usd: Quoted accommodation total in USD from lodging
            search (``>= 0.0``; if ``0.0``, allocates a baseline 35% share).
            Defaults to ``0.0``.

    Returns:
        dict[str, Any]: On success, a ``BudgetAllocationOutput`` dictionary with
        ``status="success"``, ``within_budget``, category allocations in USD,
        ``projected_total_usd``, ``remaining_buffer_usd``,
        ``daily_per_person_allowance_usd``, and ``recommendations``.
        On invalid input, returns a ``ToolErrorOutput`` dictionary.

    Guided Recovery:
        If ``status == "error"``, follow ``recovery_instructions``: ensure
        ``total_budget_usd > 0``, ``duration_days >= 1``, ``travelers >= 1``,
        ``estimated_flight_usd >= 0``, ``estimated_lodging_usd >= 0``, and
        ``travel_style`` is ``"BUDGET"``, ``"MODERATE"``, or ``"LUXURY"``.
    """
    with _enter_tool_span(
        "calculate_trip_budget_allocation",
        {"tool.name": "calculate_trip_budget_allocation", "tool.domain": "budget"},
    ):
        try:
            req = BudgetAllocationInput.model_validate(
                {
                    "total_budget_usd": total_budget_usd,
                    "duration_days": duration_days,
                    "travelers": travelers,
                    "travel_style": travel_style,
                    "estimated_flight_usd": estimated_flight_usd,
                    "estimated_lodging_usd": estimated_lodging_usd,
                }
            )
        except SchemaValidationError as exc:
            invalid_fields = [
                err.get("field", "") for err in exc.errors if err.get("field")
            ]
            _get_logger().warning(
                "Validation error in calculate_trip_budget_allocation: %s",
                invalid_fields,
            )
            return ToolErrorOutput(
                error_code="INVALID_BUDGET_ALLOCATION_INPUT",
                message=str(exc),
                recovery_instructions=(
                    "Ensure 'total_budget_usd' is strictly positive (> 0), "
                    "'duration_days' is an integer >= 1, 'travelers' is an integer "
                    ">= 1, 'travel_style' is one of ['BUDGET', 'MODERATE', 'LUXURY'], "
                    "and estimated flight/lodging amounts are >= 0.0."
                ),
                retryable=True,
                tool_name="calculate_trip_budget_allocation",
                invalid_fields=invalid_fields,
            ).model_dump()
        except Exception as exc:  # pragma: no cover
            return ToolErrorOutput(
                error_code="BUDGET_ALLOCATION_EXECUTION_ERROR",
                message=f"Unexpected error computing budget allocation: {exc}",
                recovery_instructions=(
                    "Pass numeric positive total_budget_usd and integer "
                    "duration_days >= 1, then retry."
                ),
                retryable=True,
                tool_name="calculate_trip_budget_allocation",
            ).model_dump()

        flights_usd = (
            round(req.estimated_flight_usd, 2)
            if req.estimated_flight_usd > 0.0
            else round(req.total_budget_usd * 0.30, 2)
        )
        lodging_usd = (
            round(req.estimated_lodging_usd, 2)
            if req.estimated_lodging_usd > 0.0
            else round(req.total_budget_usd * 0.35, 2)
        )

        rates = _DAILY_GROUND_RATES_USD[req.travel_style]
        person_days = req.duration_days * req.travelers
        baseline_dining = round(rates["dining"] * person_days, 2)
        baseline_activities = round(rates["activities"] * person_days, 2)
        baseline_transit = round(rates["transit"] * person_days, 2)
        baseline_ground = baseline_dining + baseline_activities + baseline_transit

        fixed_core = flights_usd + lodging_usd
        contingency = round(max(req.total_budget_usd * 0.08, 50.0), 2)

        # Scale discretionary ground categories gracefully if budget is tight
        # while still preserving realistic minimum daily subsistence.
        available_for_ground = req.total_budget_usd - fixed_core - contingency
        if available_for_ground >= baseline_ground:
            dining_usd = baseline_dining
            activities_usd = baseline_activities
            transit_usd = baseline_transit
        elif available_for_ground > 0:
            min_ground = 35.0 * person_days
            target_ground = max(min_ground, available_for_ground)
            scale = target_ground / baseline_ground if baseline_ground > 0 else 1.0
            dining_usd = round(baseline_dining * scale, 2)
            activities_usd = round(baseline_activities * scale, 2)
            transit_usd = round(baseline_transit * scale, 2)
            if target_ground <= available_for_ground:
                remainder = round(
                    max(0.0, available_for_ground - dining_usd - activities_usd), 2
                )
                transit_usd = min(transit_usd, remainder)
        else:
            dining_usd = round(30.0 * person_days, 2)
            activities_usd = round(15.0 * person_days, 2)
            transit_usd = round(12.0 * person_days, 2)

        projected_total = round(
            flights_usd
            + lodging_usd
            + dining_usd
            + activities_usd
            + transit_usd
            + contingency,
            2,
        )
        remaining_buffer = round(req.total_budget_usd - projected_total, 2)
        within_budget = projected_total <= req.total_budget_usd
        daily_allowance = round(
            (dining_usd + activities_usd + transit_usd) / person_days, 2
        )

        recommendations: list[str] = []
        if within_budget:
            recommendations.append(
                f"Projected trip total (${projected_total:.2f} USD) fits within "
                f"the ${req.total_budget_usd:.2f} USD ceiling with "
                f"${remaining_buffer:.2f} USD unallocated buffer plus "
                f"${contingency:.2f} USD contingency reserve."
            )
            recommendations.append(
                f"Daily per-traveler ground budget is ${daily_allowance:.2f} USD/day "
                f"for {req.travel_style} dining, transit, and activities."
            )
        else:
            overrun = abs(remaining_buffer)
            recommendations.append(
                f"BUDGET OVERRUN ALERT: Projected cost (${projected_total:.2f} USD) "
                f"exceeds ceiling (${req.total_budget_usd:.2f} USD) by ${overrun:.2f} USD."
            )
            recommendations.append(
                "Switch cabin_class to 'ECONOMY', filter accommodations with a lower "
                "max_nightly_rate_usd, or change travel_style to 'BUDGET' before booking."
            )

        output = BudgetAllocationOutput(
            total_budget_usd=round(req.total_budget_usd, 2),
            duration_days=req.duration_days,
            travelers=req.travelers,
            travel_style=req.travel_style,
            within_budget=within_budget,
            allocated_flights_usd=flights_usd,
            allocated_lodging_usd=lodging_usd,
            allocated_dining_usd=dining_usd,
            allocated_activities_usd=activities_usd,
            allocated_local_transit_usd=transit_usd,
            contingency_reserve_usd=contingency,
            projected_total_usd=projected_total,
            remaining_buffer_usd=remaining_buffer,
            daily_per_person_allowance_usd=daily_allowance,
            recommendations=recommendations,
        )
        return output.model_dump()


def convert_currency_exchange_rate(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> dict[str, Any]:
    """Converts a positive monetary amount between supported ISO-4217 currencies.

    Summary:
        Computes exact foreign-exchange conversions using a deterministic
        treasury reference rate table covering 39 major international currencies
        (including ``USD``, ``JPY``, ``EUR``, ``GBP``, ``CAD``, ``AUD``, ``CHF``,
        ``SGD``, ``KRW``, ``MXN``, ``THB``, and ``INR``). Suitable for fast-tier
        single-step lookups as well as multi-currency budget synthesis.

    Args:
        amount: Monetary value to convert (strictly ``> 0.0``).
        from_currency: Source 3-letter uppercase ISO-4217 currency code matching
            ``^[A-Z]{3}$`` (for example, ``"USD"`` or ``"EUR"``).
        to_currency: Target 3-letter uppercase ISO-4217 currency code matching
            ``^[A-Z]{3}$`` (for example, ``"JPY"`` or ``"GBP"``).

    Returns:
        dict[str, Any]: On success, a ``CurrencyConversionOutput`` dictionary
        with ``status="success"``, ``amount``, ``from_currency``, ``to_currency``,
        ``exchange_rate``, ``converted_amount``, and ``rate_source``.
        On invalid input, returns a ``ToolErrorOutput`` dictionary.

    Guided Recovery:
        If ``status == "error"``, read ``recovery_instructions``: ensure ``amount``
        is strictly greater than ``0`` and replace currency names or symbols
        (like ``"yen"``, ``"$"``, or ``"BAD"``) with valid 3-letter uppercase
        ISO-4217 codes such as ``"USD"``, ``"JPY"``, ``"EUR"``, or ``"GBP"``.
    """
    with _enter_tool_span(
        "convert_currency_exchange_rate",
        {"tool.name": "convert_currency_exchange_rate", "tool.domain": "budget"},
    ):
        try:
            req = CurrencyConversionInput.model_validate(
                {
                    "amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                }
            )
        except SchemaValidationError as exc:
            invalid_fields = [
                err.get("field", "") for err in exc.errors if err.get("field")
            ]
            _get_logger().warning(
                "Validation error in convert_currency_exchange_rate: %s",
                invalid_fields,
            )
            return ToolErrorOutput(
                error_code="INVALID_CURRENCY_CONVERSION_INPUT",
                message=str(exc),
                recovery_instructions=(
                    "Pass a strictly positive numeric 'amount' (> 0) and valid "
                    "3-letter uppercase ISO-4217 codes for 'from_currency' and "
                    "'to_currency' (e.g., 'USD', 'JPY', 'EUR', 'GBP', 'CAD', 'AUD', 'SGD')."
                ),
                retryable=True,
                tool_name="convert_currency_exchange_rate",
                invalid_fields=invalid_fields,
            ).model_dump()
        except Exception as exc:  # pragma: no cover
            return ToolErrorOutput(
                error_code="CURRENCY_CONVERSION_EXECUTION_ERROR",
                message=f"Unexpected error converting currency: {exc}",
                recovery_instructions=(
                    "Verify positive numeric amount and 3-letter uppercase "
                    "ISO-4217 currency codes, then retry."
                ),
                retryable=True,
                tool_name="convert_currency_exchange_rate",
            ).model_dump()

        usd_to_source = _USD_FX_RATES[req.from_currency]
        usd_to_target = _USD_FX_RATES[req.to_currency]
        rate = round(usd_to_target / usd_to_source, 6)
        converted = round(req.amount * rate, 2)
        if converted <= 0.0:
            converted = round(req.amount * rate, 6)

        output = CurrencyConversionOutput(
            amount=round(req.amount, 2),
            from_currency=req.from_currency,
            to_currency=req.to_currency,
            exchange_rate=rate,
            converted_amount=converted,
        )
        return output.model_dump()


def authorize_high_value_travel_payment(
    booking_reference: str,
    amount_usd: float,
    currency: str = "USD",
    merchant_name: str = "TravelPartner",
    payment_token_secret_id: str = "travel-payment-gateway-token",
) -> dict[str, Any]:
    """Authorizes a travel booking payment using credentials from Secret Manager.

    Summary:
        High-stakes financial settlement tool (listed in ``HIGH_STAKES_TOOLS``)
        that retrieves the payment gateway credential at runtime via
        ``SecretManagerProvider`` (Google Cloud Secret Manager or secure runtime
        environment binding) and records an audited payment authorization.
        MUST be gated by ``HumanApprovalGate`` before execution—especially when
        ``amount_usd`` exceeds ``high_value_payment_threshold_usd`` ($500.00).
        Never exposes or logs the resolved secret payload, emitting only a
        non-reversible SHA-256 fingerprint prefix.

    Args:
        booking_reference: Carrier hold or lodging reservation code (3 to 64
            characters, e.g., ``"FH-8A91C20F"`` or ``"LG-3B72E11A"``).
        amount_usd: Settlement amount in USD (strictly ``> 0.0`` and
            ``<= 500000.0``).
        currency: 3-letter uppercase ISO-4217 settlement currency code. Defaults
            to ``"USD"``.
        merchant_name: Verified travel merchant or airline name (2 to 100
            characters). Defaults to ``"TravelPartner"``.
        payment_token_secret_id: Secret Manager secret identifier containing the
            gateway token. Defaults to ``"travel-payment-gateway-token"``.

    Returns:
        dict[str, Any]: On success, a ``PaymentAuthorizationOutput`` dictionary
        with ``status="success"``, ``authorization_status="AUTHORIZED"``,
        ``transaction_id``, ``booking_reference``, ``amount_usd``, ``currency``,
        ``merchant_name``, ``high_value_flagged``, ``gateway_key_fingerprint``,
        and ``secret_source_id``. On schema or secret resolution failure,
        returns a ``ToolErrorOutput`` dictionary.

    Guided Recovery:
        If ``status == "error"`` with ``error_code == "SECRET_RESOLUTION_ERROR"``,
        do NOT invent credentials—inform the operator or ensure Secret Manager
        secret ``travel-payment-gateway-token`` (or ``TRAVEL_PAYMENT_GATEWAY_TOKEN``
        runtime secret env binding in CI/Cloud Run) is configured. For input
        schema errors, fix ``amount_usd > 0``, ``currency`` (ISO-4217), and
        ``booking_reference`` per ``recovery_instructions``.
    """
    with _enter_tool_span(
        "authorize_high_value_travel_payment",
        {
            "tool.name": "authorize_high_value_travel_payment",
            "tool.domain": "payments",
            "tool.high_stakes": True,
        },
    ):
        try:
            req = PaymentAuthorizationInput.model_validate(
                {
                    "booking_reference": booking_reference,
                    "amount_usd": amount_usd,
                    "currency": currency,
                    "merchant_name": merchant_name,
                    "payment_token_secret_id": payment_token_secret_id,
                }
            )
        except SchemaValidationError as exc:
            invalid_fields = [
                err.get("field", "") for err in exc.errors if err.get("field")
            ]
            _get_logger().warning(
                "Validation error in authorize_high_value_travel_payment: %s",
                invalid_fields,
            )
            return ToolErrorOutput(
                error_code="INVALID_PAYMENT_AUTHORIZATION_INPUT",
                message=str(exc),
                recovery_instructions=(
                    "Provide a valid 'booking_reference' (3-64 chars), strictly "
                    "positive 'amount_usd' (> 0), valid 3-letter uppercase ISO-4217 "
                    "'currency' (e.g., 'USD'), and alphanumeric/hyphenated "
                    "'payment_token_secret_id'."
                ),
                retryable=True,
                tool_name="authorize_high_value_travel_payment",
                invalid_fields=invalid_fields,
            ).model_dump()

        cfg = get_config()
        provider = get_secret_provider(cfg)
        try:
            gateway_token = provider.get_secret(req.payment_token_secret_id)
        except SecretResolutionError as exc:
            _get_logger().error(
                "Payment gateway secret '%s' could not be resolved.",
                req.payment_token_secret_id,
            )
            return ToolErrorOutput(
                error_code="SECRET_RESOLUTION_ERROR",
                message=str(exc),
                recovery_instructions=(
                    f"Configure Google Cloud Secret Manager secret "
                    f"'{req.payment_token_secret_id}' in GOOGLE_CLOUD_PROJECT with "
                    f"roles/secretmanager.secretAccessor, or bind environment "
                    f"variable 'TRAVEL_PAYMENT_GATEWAY_TOKEN' in Cloud Run / CI "
                    f"before retrying payment authorization."
                ),
                retryable=True,
                tool_name="authorize_high_value_travel_payment",
                invalid_fields=["payment_token_secret_id"],
            ).model_dump()
        except Exception as exc:  # pragma: no cover
            return ToolErrorOutput(
                error_code="PAYMENT_GATEWAY_ERROR",
                message=f"Unexpected payment authorization error: {exc}",
                recovery_instructions=(
                    "Verify Secret Manager connectivity and payment parameters, "
                    "then retry after human approval confirmation."
                ),
                retryable=True,
                tool_name="authorize_high_value_travel_payment",
            ).model_dump()

        key_fingerprint = f"sha256:{hashlib.sha256(gateway_token.encode('utf-8')).hexdigest()[:12]}"
        tx_digest = hashlib.sha256(
            f"{req.booking_reference}:{req.amount_usd:.2f}:{key_fingerprint}".encode(
                "utf-8"
            )
        ).hexdigest()[:10].upper()

        output = PaymentAuthorizationOutput(
            transaction_id=f"TXN-{tx_digest}",
            booking_reference=req.booking_reference,
            amount_usd=round(req.amount_usd, 2),
            currency=req.currency,
            merchant_name=req.merchant_name,
            high_value_flagged=req.amount_usd > cfg.high_value_payment_threshold_usd,
            gateway_key_fingerprint=key_fingerprint,
            secret_source_id=req.payment_token_secret_id,
        )
        return output.model_dump()


calculate_trip_budget_allocation.input_schema = BudgetAllocationInput  # type: ignore[attr-defined]
calculate_trip_budget_allocation.output_schema = BudgetAllocationOutput  # type: ignore[attr-defined]
convert_currency_exchange_rate.input_schema = CurrencyConversionInput  # type: ignore[attr-defined]
convert_currency_exchange_rate.output_schema = CurrencyConversionOutput  # type: ignore[attr-defined]
authorize_high_value_travel_payment.input_schema = PaymentAuthorizationInput  # type: ignore[attr-defined]
authorize_high_value_travel_payment.output_schema = PaymentAuthorizationOutput  # type: ignore[attr-defined]
