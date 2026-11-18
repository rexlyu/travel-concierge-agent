"""Context window management, history compaction, and Vertex AI context caching.

Implements Rubric 2.2 (History Compaction):
- Google ADK native ``EventsCompactionConfig`` builder with clean offline fallback.
- Vertex AI Context Caching configuration builder (``CreateCachedContentConfig``).
- ``ConversationCompactor`` combining token-budget tracking, sliding-window retention
  of recent turns, and structured domain-aware summarization of older evicted turns.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    from google.adk.apps.app import EventsCompactionConfig as _AdkEventsCompactionConfig

    _ADK_COMPACTION_AVAILABLE = True
except ImportError:
    _AdkEventsCompactionConfig = None
    _ADK_COMPACTION_AVAILABLE = False

try:
    from google.genai import types as _genai_types

    _GENAI_CACHE_CLASS = getattr(_genai_types, "CreateCachedContentConfig", None)
    _GENAI_AVAILABLE = _GENAI_CACHE_CLASS is not None
except ImportError:
    _genai_types = None
    _GENAI_CACHE_CLASS = None
    _GENAI_AVAILABLE = False


@dataclass(frozen=True)
class EventsCompactionConfigFallback:
    """Offline-compatible stand-in for ``google.adk.apps.app.EventsCompactionConfig``.

    Attributes:
        compaction_interval: Number of new invocation events before triggering compaction.
        overlap_size: Number of trailing events carried across compaction boundaries.
    """

    compaction_interval: int = 5
    overlap_size: int = 2

    def to_dict(self) -> dict[str, int]:
        """Serialize compaction parameters to a plain dictionary."""
        return {
            "compaction_interval": self.compaction_interval,
            "overlap_size": self.overlap_size,
        }

    def model_dump(self) -> dict[str, int]:
        """Pydantic-compatible dictionary export."""
        return self.to_dict()


@dataclass(frozen=True)
class CachedContentConfigFallback:
    """Offline-compatible stand-in for ``google.genai.types.CreateCachedContentConfig``."""

    model: str
    ttl: str
    display_name: str = "travel-concierge-context-cache"
    system_instruction: str | None = None
    contents: list[Any] | None = None


def build_adk_compaction_config(
    compaction_interval: int = 5,
    overlap_size: int = 2,
) -> Any:
    """Build an ADK ``EventsCompactionConfig`` for automatic event stream compaction.

    Args:
        compaction_interval: Number of turns/invocations between compactions (>= 1).
        overlap_size: Number of overlapping turns retained across boundaries (>= 0).

    Returns:
        An instance of ``google.adk.apps.app.EventsCompactionConfig`` when ``google-adk``
        is installed, or ``EventsCompactionConfigFallback`` with identical attributes.

    Raises:
        ValueError: If ``compaction_interval < 1`` or ``overlap_size < 0``.
    """
    if compaction_interval < 1:
        raise ValueError(f"compaction_interval must be >= 1, got {compaction_interval}")
    if overlap_size < 0:
        raise ValueError(f"overlap_size must be >= 0, got {overlap_size}")

    if _ADK_COMPACTION_AVAILABLE and _AdkEventsCompactionConfig is not None:
        return _AdkEventsCompactionConfig(
            compaction_interval=compaction_interval,
            overlap_size=overlap_size,
        )
    return EventsCompactionConfigFallback(
        compaction_interval=compaction_interval,
        overlap_size=overlap_size,
    )


def create_genai_cached_content_config(
    ttl_seconds: int = 3600,
    model: str = "gemini-2.5-pro",
    system_instruction: str | None = None,
    contents: list[Any] | None = None,
    display_name: str = "travel-concierge-context-cache",
) -> Any:
    """Create a ``google.genai.types.CreateCachedContentConfig`` object (or fallback).

    Args:
        ttl_seconds: Cache time-to-live in seconds (> 0).
        model: Target Gemini model identifier for cached prefix tokens.
        system_instruction: Optional static constitution / system instruction text.
        contents: Optional static seed contents to pin in Vertex AI context cache.
        display_name: Human-readable identifier for the cached content resource.

    Returns:
        ``google.genai.types.CreateCachedContentConfig`` when ``google-genai`` is
        installed, or ``CachedContentConfigFallback`` otherwise.
    """
    if ttl_seconds <= 0:
        raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
    ttl_str = f"{ttl_seconds}s"

    if _GENAI_AVAILABLE and _GENAI_CACHE_CLASS is not None:
        kwargs: dict[str, Any] = {
            "ttl": ttl_str,
            "display_name": display_name,
        }
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        if contents is not None:
            kwargs["contents"] = contents
        return _GENAI_CACHE_CLASS(**kwargs)

    return CachedContentConfigFallback(
        model=model,
        ttl=ttl_str,
        display_name=display_name,
        system_instruction=system_instruction,
        contents=contents,
    )


def build_context_cache_config(
    ttl_seconds: int = 3600,
    min_tokens: int = 2048,
    model: str = "gemini-2.5-pro",
    system_instruction: str | None = None,
    display_name: str = "travel-concierge-context-cache",
) -> dict[str, Any]:
    """Build a Vertex AI Context Caching configuration dictionary.

    Integrates with ``google.genai.types.CreateCachedContentConfig`` so static
    system constitutions, schemas, and compacted conversation prefixes can be
    cached across multi-turn Gemini invocations when exceeding ``min_tokens``.

    Args:
        ttl_seconds: Cache TTL in seconds (default 3600s / 1 hour).
        min_tokens: Minimum prefix token threshold required before creating a cache.
        model: Gemini model identifier supporting Vertex AI Context Caching.
        system_instruction: Optional system instruction prefix to include in cache metadata.
        display_name: Display name assigned to the Vertex AI cached content entry.

    Returns:
        JSON-serializable configuration dictionary for Vertex AI Context Caching.

    Raises:
        ValueError: If ``ttl_seconds <= 0`` or ``min_tokens <= 0``.
    """
    if ttl_seconds <= 0:
        raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
    if min_tokens <= 0:
        raise ValueError(f"min_tokens must be > 0, got {min_tokens}")

    # Validate SDK config instantiation path as part of building cache config.
    _ = create_genai_cached_content_config(
        ttl_seconds=ttl_seconds,
        model=model,
        system_instruction=system_instruction,
        display_name=display_name,
    )

    return {
        "enabled": True,
        "provider": "vertex_ai_context_caching",
        "model": model,
        "ttl": f"{ttl_seconds}s",
        "ttl_seconds": ttl_seconds,
        "min_tokens": min_tokens,
        "display_name": display_name,
        "system_instruction": system_instruction,
        "genai_config_type": "google.genai.types.CreateCachedContentConfig",
        "genai_sdk_available": _GENAI_AVAILABLE,
    }


# Compiled patterns for structured travel fact preservation during compaction.
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\b")
_MONTH_DATE_RE = re.compile(
    r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:,?\s+20\d{2})?)\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(r"\b(\d{1,2}\s+(?:days?|nights?|weeks?))\b", re.IGNORECASE)
_BUDGET_RE = re.compile(
    r"(?:\$\s?\d[\d,]*(?:\.\d{1,2})?(?:\s*(?:USD|per\s+night|/night))?|"
    r"\b\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|EUR|JPY|GBP|CAD|AUD|CHF|SGD)\b)",
    re.IGNORECASE,
)
_BOOKING_ID_RE = re.compile(
    r"\b((?:FL|HTL|LOD|BK|BOOK|CONF|HOLD|PAY|RES|ITIN|AUTH)-[A-Z0-9_-]{3,24})\b",
    re.IGNORECASE,
)
_IATA_RE = re.compile(
    r"\b(SFO|LAX|JFK|EWR|ORD|SEA|BOS|DFW|ATL|MIA|DEN|IAD|NRT|HND|KIX|ICN|"
    r"LHR|LGW|CDG|ORY|FRA|MUC|AMS|FCO|MXP|MAD|BCN|LIS|ZRH|SIN|HKG|BKK|SYD|MEL|YYZ|YVR)\b"
)
_KNOWN_DESTINATIONS = (
    "Tokyo",
    "Kyoto",
    "Osaka",
    "Paris",
    "London",
    "Rome",
    "Florence",
    "Venice",
    "Milan",
    "Barcelona",
    "Madrid",
    "Lisbon",
    "Amsterdam",
    "Berlin",
    "Munich",
    "Zurich",
    "Geneva",
    "Vienna",
    "Prague",
    "Seoul",
    "Singapore",
    "Bangkok",
    "Bali",
    "Sydney",
    "Melbourne",
    "Auckland",
    "New York",
    "San Francisco",
    "Los Angeles",
    "Chicago",
    "Seattle",
    "Boston",
    "Miami",
    "Honolulu",
    "Vancouver",
    "Toronto",
    "Mexico City",
    "Reykjavik",
    "Dublin",
    "Edinburgh",
    "Athens",
    "Istanbul",
    "Dubai",
    "Taipei",
    "Hong Kong",
)
_DESTINATION_PHRASE_RE = re.compile(
    r"\b(?:trip\s+to|travel(?:ing)?\s+to|fly(?:ing)?\s+to|visit(?:ing)?|stay(?:ing)?\s+in|heading\s+to)"
    r"\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"
)
_DIETARY_KEYWORDS = (
    "vegetarian",
    "vegan",
    "gluten-free",
    "gluten free",
    "halal",
    "kosher",
    "nut allergy",
    "peanut allergy",
    "shellfish allergy",
    "dairy-free",
    "dairy free",
    "lactose-free",
    "pescatarian",
)
_SEATING_CABIN_KEYWORDS = (
    "aisle seat",
    "aisle",
    "window seat",
    "window",
    "extra legroom",
    "exit row",
    "business class",
    "first class",
    "premium economy",
    "economy class",
    "nonstop",
    "non-stop",
    "direct flight",
    "refundable",
    "boutique hotel",
    "ryokan",
    "near transit",
    "king bed",
    "quiet room",
    "high floor",
    "breakfast included",
)
_AIRLINE_KEYWORDS = (
    "ANA",
    "JAL",
    "Japan Airlines",
    "All Nippon Airways",
    "United",
    "Delta",
    "American Airlines",
    "British Airways",
    "Air France",
    "Lufthansa",
    "Singapore Airlines",
    "Emirates",
    "Qatar Airways",
    "Cathay Pacific",
    "Korean Air",
    "Swiss",
    "KLM",
)


def _extract_message_text(message: dict[str, Any]) -> str:
    """Extract normalized plain text from a conversation message dictionary."""
    content = message.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        text = " ".join(p for p in parts if p)
    elif isinstance(content, dict):
        text = str(content.get("text") or content.get("content") or "")
    else:
        text = str(content)

    # Also support ADK-style `parts` when `content` is omitted.
    if not text and isinstance(message.get("parts"), list):
        parts_list = [
            str(p.get("text", "")) if isinstance(p, dict) else str(p)
            for p in message["parts"]
        ]
        text = " ".join(p for p in parts_list if p)

    return text.strip()


def _dedupe_preserve_order(items: list[str], max_items: int = 8) -> list[str]:
    """Deduplicate strings case-insensitively while preserving discovery order."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        cleaned = " ".join(raw.split()).strip(" ,.;:")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
            if len(result) >= max_items:
                break
    return result


class ConversationCompactor:
    """Token-aware sliding-window conversation history compactor.

    Monitors total estimated token usage across multi-turn conversations. When
    the history exceeds ``max_tokens``, older turns outside the ``keep_recent_turns``
    sliding window are condensed into a structured ``[COMPACTED_CONVERSATION_SUMMARY]``
    context message preserving destinations, travel dates, budget limits, dietary
    and seating/lodging preferences, and confirmed booking references.
    """

    SUMMARY_PREFIX = "[COMPACTED_CONVERSATION_SUMMARY]"

    def __init__(
        self,
        max_tokens: int = 4000,
        keep_recent_turns: int = 6,
        chars_per_token: float = 4.0,
    ) -> None:
        if max_tokens < 16:
            raise ValueError(f"max_tokens must be >= 16, got {max_tokens}")
        if keep_recent_turns < 1:
            raise ValueError(f"keep_recent_turns must be >= 1, got {keep_recent_turns}")
        if chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {chars_per_token}")

        self.max_tokens = max_tokens
        self.keep_recent_turns = keep_recent_turns
        self.chars_per_token = chars_per_token

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token consumption for a sequence of message dictionaries.

        Uses a deterministic hybrid character/word token heuristic plus per-message
        role framing overhead (~4 tokens/message).

        Args:
            messages: Ordered list of conversation turn dicts (``role``, ``content``).

        Returns:
            Estimated integer token count (>= 0).
        """
        if not messages:
            return 0

        total_tokens = 0
        for msg in messages:
            text = _extract_message_text(msg)
            role = str(msg.get("role", "user"))
            framing_tokens = 4 + max(1, len(role) // 6)
            if not text:
                total_tokens += framing_tokens
                continue
            char_estimate = math.ceil(len(text) / self.chars_per_token)
            word_estimate = math.ceil(len(text.split()) * 1.3)
            total_tokens += framing_tokens + max(char_estimate, word_estimate)
        return total_tokens

    def _summarize_evicted_turns(
        self,
        evicted_messages: list[dict[str, Any]],
        max_summary_chars: int | None = None,
    ) -> str:
        """Build a structured travel summary from evicted conversation turns."""
        destinations: list[str] = []
        dates_and_durations: list[str] = []
        budgets: list[str] = []
        dietary_prefs: list[str] = []
        travel_prefs: list[str] = []
        confirmed_bookings: list[str] = []
        key_notes: list[str] = []

        for msg in evicted_messages:
            text = _extract_message_text(msg)
            if not text:
                continue

            # Carry forward facts from any previous compacted summary message.
            if self.SUMMARY_PREFIX in text:
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped and stripped != self.SUMMARY_PREFIX:
                        key_notes.append(stripped)

            lower_text = text.lower()

            for city in _KNOWN_DESTINATIONS:
                if re.search(rf"\b{re.escape(city.lower())}\b", lower_text):
                    destinations.append(city)
            for match in _DESTINATION_PHRASE_RE.finditer(text):
                candidate = match.group(1).strip()
                if candidate.lower() not in {"the", "my", "our", "find", "book", "check"}:
                    destinations.append(candidate)
            for iata in _IATA_RE.findall(text):
                destinations.append(iata)

            dates_and_durations.extend(_ISO_DATE_RE.findall(text))
            dates_and_durations.extend(_MONTH_DATE_RE.findall(text))
            dates_and_durations.extend(_DURATION_RE.findall(text))

            budgets.extend(_BUDGET_RE.findall(text))

            for kw in _DIETARY_KEYWORDS:
                if kw in lower_text:
                    dietary_prefs.append(kw)

            for kw in _SEATING_CABIN_KEYWORDS:
                if kw in lower_text:
                    travel_prefs.append(kw)

            for airline in _AIRLINE_KEYWORDS:
                if re.search(rf"\b{re.escape(airline)}\b", text, re.IGNORECASE):
                    travel_prefs.append(f"airline:{airline}")

            confirmed_bookings.extend(_BOOKING_ID_RE.findall(text))
            if any(w in lower_text for w in ("confirmed", "booked", "reserved", "authorized")):
                for sentence in re.split(r"(?<=[.!?])\s+", text):
                    s_low = sentence.lower()
                    if any(w in s_low for w in ("confirmed", "booked", "reserved", "authorized")):
                        confirmed_bookings.append(sentence[:120])

        sections: list[str] = [
            f"{self.SUMMARY_PREFIX} Condensed {len(evicted_messages)} earlier turns:"
        ]

        dedup_dest = _dedupe_preserve_order(destinations, max_items=8)
        if dedup_dest:
            sections.append(f"- Destinations/Routes: {', '.join(dedup_dest)}")

        dedup_dates = _dedupe_preserve_order(dates_and_durations, max_items=6)
        if dedup_dates:
            sections.append(f"- Dates/Duration: {', '.join(dedup_dates)}")

        dedup_budgets = _dedupe_preserve_order(budgets, max_items=6)
        if dedup_budgets:
            sections.append(f"- Budget Constraints: {', '.join(dedup_budgets)}")

        dedup_dietary = _dedupe_preserve_order(dietary_prefs, max_items=6)
        if dedup_dietary:
            sections.append(f"- Dietary Needs: {', '.join(dedup_dietary)}")

        dedup_prefs = _dedupe_preserve_order(travel_prefs, max_items=8)
        if dedup_prefs:
            sections.append(f"- Seating/Cabin/Lodging Preferences: {', '.join(dedup_prefs)}")

        dedup_bookings = _dedupe_preserve_order(confirmed_bookings, max_items=6)
        if dedup_bookings:
            sections.append(f"- Confirmed Bookings/References: {'; '.join(dedup_bookings)}")

        if len(sections) == 1:
            # Fallback snippet digest when turns contain generic non-structured prose.
            snippets: list[str] = []
            for msg in evicted_messages[:4]:
                role = msg.get("role", "user")
                snippet = " ".join(_extract_message_text(msg).split())[:80]
                if snippet:
                    snippets.append(f"{role}: {snippet}")
            if key_notes:
                sections.extend(key_notes[:3])
            elif snippets:
                sections.append(f"- Prior Context: {' | '.join(snippets)}")

        summary_text = "\n".join(sections)
        if max_summary_chars is not None and len(summary_text) > max_summary_chars:
            min_len = len(self.SUMMARY_PREFIX) + 12
            cutoff = max(min_len, max_summary_chars - 3)
            summary_text = summary_text[:cutoff].rstrip() + "..."
        return summary_text

    def compact_history(
        self,
        messages: list[dict[str, Any]],
        force: bool = False,
    ) -> dict[str, Any]:
        """Compact conversation history if token usage exceeds ``self.max_tokens``.

        Retains up to ``keep_recent_turns`` most recent turns (dynamically shrinking
        the retained window if needed to fit within ``max_tokens``) and replaces
        older turns with a single ``[COMPACTED_CONVERSATION_SUMMARY]`` system message.

        Args:
            messages: Chronological list of conversation turn dicts.
            force: If True, perform sliding-window compaction even when under budget.

        Returns:
            Dictionary with keys:
                - ``messages``: Resulting list of message dicts.
                - ``summary``: Generated summary string (empty if not compacted).
                - ``original_tokens``: Token estimate prior to compaction.
                - ``compacted_tokens``: Token estimate after compaction.
                - ``was_compacted``: Boolean indicating whether compaction ran.
        """
        if not messages:
            return {
                "messages": [],
                "summary": "",
                "original_tokens": 0,
                "compacted_tokens": 0,
                "was_compacted": False,
            }

        original_tokens = self.estimate_tokens(messages)
        if original_tokens <= self.max_tokens and not force:
            return {
                "messages": [dict(m) for m in messages],
                "summary": "",
                "original_tokens": original_tokens,
                "compacted_tokens": original_tokens,
                "was_compacted": False,
            }

        if len(messages) == 1:
            single_text = _extract_message_text(messages[0])
            char_budget = max(64, int((self.max_tokens - 12) * self.chars_per_token * 0.7))
            summary_text = self._summarize_evicted_turns(
                messages,
                max_summary_chars=min(char_budget, max(64, len(single_text) // 2)),
            )
            compacted_Single = [{"role": "system", "content": summary_text}]
            compacted_tokens = self.estimate_tokens(compacted_Single)
            return {
                "messages": compacted_Single,
                "summary": summary_text,
                "original_tokens": original_tokens,
                "compacted_tokens": compacted_tokens,
                "was_compacted": True,
            }

        keep_count = min(self.keep_recent_turns, len(messages) - 1)
        keep_count = max(1, keep_count)

        summary_text = ""
        compacted_messages: list[dict[str, Any]] = []
        compacted_tokens = original_tokens

        while keep_count >= 1:
            evicted = messages[:-keep_count]
            recent = [dict(m) for m in messages[-keep_count:]]
            evicted_chars = sum(len(_extract_message_text(m)) for m in evicted)
            max_summary_chars = max(96, int(evicted_chars * 0.65))

            summary_text = self._summarize_evicted_turns(
                evicted,
                max_summary_chars=max_summary_chars,
            )
            summary_msg = {"role": "system", "content": summary_text}
            compacted_messages = [summary_msg, *recent]
            compacted_tokens = self.estimate_tokens(compacted_messages)

            if compacted_tokens <= self.max_tokens or keep_count == 1:
                break
            keep_count -= 1

        # If the single kept recent turn is itself massive, bound summary tightly
        # so compacted_tokens is strictly smaller than original_tokens.
        if compacted_tokens >= original_tokens:
            evicted = messages[:-1]
            recent_last = dict(messages[-1])
            summary_text = f"{self.SUMMARY_PREFIX} ({len(evicted)} prior turns condensed)"
            compacted_messages = [{"role": "system", "content": summary_text}, recent_last]
            compacted_tokens = self.estimate_tokens(compacted_messages)

        logger.info(
            "Compacted conversation history from %d to %d tokens (%d -> %d messages).",
            original_tokens,
            compacted_tokens,
            len(messages),
            len(compacted_messages),
        )

        return {
            "messages": compacted_messages,
            "summary": summary_text,
            "original_tokens": original_tokens,
            "compacted_tokens": compacted_tokens,
            "was_compacted": True,
        }
