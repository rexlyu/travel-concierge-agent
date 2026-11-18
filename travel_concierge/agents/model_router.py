"""Strategic model routing between fast and reasoning Gemini tiers.

Implements Rubric 3.2 (Strategic Model Routing):
- Routes low-latency single-step tasks (currency conversion, destination
  advisory checks, greetings, FAQs, single lookups) to `gemini-2.5-flash`.
- Routes complex multi-step reasoning tasks (multi-city itinerary synthesis,
  budget optimization and trade-offs, group/dietary constraint resolution)
  to `gemini-2.5-pro`.
- Provides `dynamic_model_routing_callback` mutating `llm_request.model` on the
  fly inside ADK `LlmAgent(before_model_callback=...)`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from travel_concierge.guardrails.policy_plugin import (
    extract_state_from_context,
    extract_user_text_from_request,
)

_SIMPLE_CURRENCY_RE = re.compile(
    r"\b(convert|exchange\s+rate|currency|fx\s+rate)\b",
    re.IGNORECASE,
)
_SIMPLE_ADVISORY_RE = re.compile(
    r"\b(travel\s+advisor(?:y|ies)|safety\s+level|visa\s+requirements?|entry\s+requirements?)\b",
    re.IGNORECASE,
)
_SIMPLE_GREETING_FAQ_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+(?:morning|afternoon|evening)|help|what\s+can\s+you\s+do|status|thanks|thank\s+you)\b",
    re.IGNORECASE,
)

_MULTI_CITY_ITINERARY_RE = re.compile(
    r"\b(multi-?city|multi-?stop|multi-?leg|itinerary|day-by-day|"
    r"\d+\s*-\s*day\s+trip|\d+\s+days?\s+(?:trip|in|across)|"
    r"plan\s+(?:a|an|our|my)\s+.*?\btrip\b)\b",
    re.IGNORECASE,
)
_BUDGET_OPTIMIZATION_RE = re.compile(
    r"\b(optimize|optimization|trade-?offs?|rebalance|re-?allocate|"
    r"allocate\s+.*?\bbudget|strict\s+\$?\d|under\s+(?:a\s+)?(?:strict\s+)?\$\s?\d|"
    r"within\s+(?:a\s+)?\$\s?\d|budget\s+(?:allocation|breakdown|optimization|trade-?off))\b",
    re.IGNORECASE,
)
_CONSTRAINT_SYNTHESIS_RE = re.compile(
    r"\b(conflicting|constraints?|compare\s+options|group\s+of\s+\d+|"
    r"family\s+of\s+\d+|\b[2-9]\s+travelers\b|alternative\s+itinerar(?:y|ies)|"
    r"re-?plan|synthesize)\b",
    re.IGNORECASE,
)

_KNOWN_CITIES: tuple[str, ...] = (
    "tokyo",
    "kyoto",
    "osaka",
    "paris",
    "london",
    "rome",
    "florence",
    "venice",
    "barcelona",
    "madrid",
    "lisbon",
    "amsterdam",
    "berlin",
    "zurich",
    "seoul",
    "singapore",
    "bangkok",
    "sydney",
    "new york",
    "san francisco",
    "toronto",
    "vancouver",
)

_DOMAIN_FACETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("flights", re.compile(r"\b(flights?|airfare|airlines?|departing|fly)\b", re.IGNORECASE)),
    ("lodging", re.compile(r"\b(hotels?|lodging|accommodations?|ryokans?|nights?|stays?)\b", re.IGNORECASE)),
    ("budget", re.compile(r"\b(budget|cost|allocation|\$\s?\d[\d,]*)\b", re.IGNORECASE)),
    ("advisories", re.compile(r"\b(advisor(?:y|ies)|safety|visa)\b", re.IGNORECASE)),
)


def _emit_routing_log(
    model: str,
    tier: str,
    complexity_score: int,
    reason: str,
) -> None:
    """Emit structured JSON log entry for every model routing decision."""
    try:
        from travel_concierge.observability import (
            get_structured_logger,
            log_structured_event,
        )

        logger = get_structured_logger("travel_concierge.agents.router")
        log_structured_event(
            logger,
            event_type="MODEL_ROUTING_DECISION",
            message=f"Routed request to {model} ({tier} tier, score={complexity_score}).",
            severity="INFO",
            selected_model=model,
            tier=tier,
            complexity_score=complexity_score,
            reason=reason,
        )
    except Exception:
        fallback_logger = logging.getLogger("travel_concierge.agents.router")
        fallback_logger.info(
            "MODEL_ROUTING_DECISION model=%s tier=%s score=%d reason=%s",
            model,
            tier,
            complexity_score,
            reason,
        )


class StrategicModelRouter:
    """Dynamically selects between fast (`gemini-2.5-flash`) and reasoning (`gemini-2.5-pro`) models.

    Balances token cost, tail latency, and multi-constraint reasoning quality by
    scoring prompt complexity across destination count, domain breadth, financial
    optimization requirements, and group/conflict constraints.
    """

    REASONING_THRESHOLD: int = 4

    def __init__(
        self,
        fast_model: str = "gemini-2.5-flash",
        reasoning_model: str = "gemini-2.5-pro",
    ) -> None:
        self.fast_model = fast_model
        self.reasoning_model = reasoning_model

    def select_model(
        self,
        query: str,
        task_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select the optimal Gemini model tier for `query` and `task_metadata`.

        Args:
            query: User request text.
            task_metadata: Optional structured hints (e.g. `force_tier`,
                `multi_city`, `requires_optimization`, `num_destinations`).

        Returns:
            Dictionary with `model` (str), `tier` ("fast" | "reasoning"),
            `complexity_score` (int in 1..10), and `reason` (str).
        """
        text = str(query or "").strip()
        lower_text = text.lower()
        meta = dict(task_metadata or {})

        forced_tier = str(meta.get("force_tier", "")).strip().lower()
        if forced_tier == "reasoning":
            decision = {
                "model": self.reasoning_model,
                "tier": "reasoning",
                "complexity_score": 8,
                "reason": "Explicit task_metadata force_tier='reasoning' override.",
            }
            _emit_routing_log(
                decision["model"],
                decision["tier"],
                decision["complexity_score"],
                decision["reason"],
            )
            return decision
        if forced_tier == "fast":
            decision = {
                "model": self.fast_model,
                "tier": "fast",
                "complexity_score": 1,
                "reason": "Explicit task_metadata force_tier='fast' override.",
            }
            _emit_routing_log(
                decision["model"],
                decision["tier"],
                decision["complexity_score"],
                decision["reason"],
            )
            return decision

        score = 1
        signals: list[str] = []

        matched_cities = [city for city in _KNOWN_CITIES if city in lower_text]
        has_itinerary_pattern = bool(_MULTI_CITY_ITINERARY_RE.search(text))
        has_budget_opt = bool(_BUDGET_OPTIMIZATION_RE.search(text))
        has_constraints = bool(_CONSTRAINT_SYNTHESIS_RE.search(text))

        matched_facets = [
            name for name, pattern in _DOMAIN_FACETS if pattern.search(text)
        ]

        if len(matched_cities) >= 2 or meta.get("multi_city") is True or int(meta.get("num_destinations", 1) or 1) > 1:
            score += 3
            signals.append("multi-city destination synthesis")
        elif has_itinerary_pattern:
            score += 3
            signals.append("multi-day itinerary planning")

        if has_budget_opt or meta.get("requires_optimization") is True:
            score += 2
            signals.append("budget optimization and trade-off constraints")

        if len(matched_facets) >= 3:
            score += 2
            signals.append(f"cross-domain coordination ({', '.join(matched_facets)})")
        elif len(matched_facets) == 2 and has_itinerary_pattern:
            score += 1
            signals.append(f"multi-facet planning ({', '.join(matched_facets)})")

        if has_constraints or int(meta.get("constraint_count", 0) or 0) >= 2:
            score += 2
            signals.append("multi-traveler or constraint conflict resolution")

        # Keep single-purpose lookups on the fast model when no multi-step synthesis is requested.
        is_pure_single_lookup = (
            not has_itinerary_pattern
            and len(matched_cities) <= 1
            and len(matched_facets) <= 2
            and not meta.get("multi_city")
            and not meta.get("requires_optimization")
        )
        if is_pure_single_lookup and (
            _SIMPLE_CURRENCY_RE.search(text)
            or _SIMPLE_ADVISORY_RE.search(text)
            or _SIMPLE_GREETING_FAQ_RE.search(text)
        ):
            score = min(score, 2)
            signals = ["single-step lookup or conversational turn suitable for low-latency flash tier"]

        complexity_score = max(1, min(10, score))
        if complexity_score >= self.REASONING_THRESHOLD:
            selected_model = self.reasoning_model
            tier = "reasoning"
            reason = (
                "Selected reasoning tier for complex synthesis: "
                + "; ".join(signals or ["elevated multi-step complexity"])
            )
        else:
            selected_model = self.fast_model
            tier = "fast"
            reason = (
                "Selected fast tier for low-latency execution: "
                + "; ".join(signals or ["direct single-step request"])
            )

        _emit_routing_log(selected_model, tier, complexity_score, reason)
        return {
            "model": selected_model,
            "tier": tier,
            "complexity_score": complexity_score,
            "reason": reason,
        }


_DEFAULT_MODEL_ROUTER: StrategicModelRouter | None = None


def get_default_model_router() -> StrategicModelRouter:
    """Return a default StrategicModelRouter initialized from TravelConciergeConfig."""
    global _DEFAULT_MODEL_ROUTER
    if _DEFAULT_MODEL_ROUTER is None:
        fast_model = "gemini-2.5-flash"
        reasoning_model = "gemini-2.5-pro"
        try:
            from travel_concierge.config import get_config

            cfg = get_config()
            fast_model = cfg.fast_model
            reasoning_model = cfg.reasoning_model
        except Exception:
            pass
        _DEFAULT_MODEL_ROUTER = StrategicModelRouter(
            fast_model=fast_model,
            reasoning_model=reasoning_model,
        )
    return _DEFAULT_MODEL_ROUTER


def dynamic_model_routing_callback(
    callback_context: Any,
    llm_request: Any,
) -> None:
    """ADK `before_model_callback` mutating `llm_request.model` per query complexity."""
    query = extract_user_text_from_request(llm_request)
    state = extract_state_from_context(callback_context)
    task_metadata = state.get("task_metadata") if isinstance(state, dict) else None

    router = get_default_model_router()
    decision = router.select_model(
        query=query,
        task_metadata=task_metadata if isinstance(task_metadata, dict) else None,
    )

    if llm_request is not None:
        if isinstance(llm_request, dict):
            llm_request["model"] = decision["model"]
        else:
            try:
                llm_request.model = decision["model"]
            except Exception:
                pass

    if callback_context is not None:
        ctx_state = getattr(callback_context, "state", None)
        if isinstance(ctx_state, dict):
            ctx_state["last_routing_decision"] = decision
    return None
