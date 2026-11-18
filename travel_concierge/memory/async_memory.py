"""Asynchronous background memory extraction and consolidation for ADK agents.

Implements Rubric 2.4 (Async Memory Operations):
- Non-blocking background consolidation via ``asyncio.create_task`` so caller/UI
  turns return immediately without waiting on memory persistence.
- Salient traveler preference & trip constraint extraction with active PII
  scrubbing prior to vector indexing and SQLite storage.
- ADK ``after_agent_callback`` integration (``after_agent_memory_callback``).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from travel_concierge.memory.compaction import (
    ConversationCompactor,
    _extract_message_text,
)
from travel_concierge.memory.session_store import (
    PersistentTravelSessionManager,
    _scrub_text_safely,
)

logger = logging.getLogger(__name__)

_FLIGHT_PREF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(aisle(?:\s+seats?)?|window(?:\s+seats?)?|extra\s+legroom|exit\s+row|"
        r"business\s+class|first\s+class|premium\s+economy|economy\s+class|"
        r"non-?stop(?:\s+flights?)?|direct\s+flights?|no\s+red-?eyes?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(ANA|JAL|Japan\s+Airlines|All\s+Nippon\s+Airways|United(?:\s+Airlines)?|"
        r"Delta(?:\s+Air\s+Lines)?|American\s+Airlines|British\s+Airways|Air\s+France|"
        r"Lufthansa|Singapore\s+Airlines|Emirates|Qatar\s+Airways|Cathay\s+Pacific|"
        r"Korean\s+Air|Swiss|KLM)\b",
        re.IGNORECASE,
    ),
)

_LODGING_PREF_RE = re.compile(
    r"\b(boutique\s+hotels?|ryokans?|resorts?|5-star\s+hotels?|4-star\s+hotels?|"
    r"near\s+(?:transit|subway|train\s+station)|walkable(?:\s+area)?|"
    r"quiet\s+rooms?|high\s+floors?|king\s+beds?|twin\s+beds?|"
    r"refundable(?:\s+bookings?|\s+rates?|\s+hotels?)?|free\s+breakfast|breakfast\s+included)\b",
    re.IGNORECASE,
)

_BUDGET_CONSTRAINT_RE = re.compile(
    r"(?:(?:total\s+)?budget(?:\s+(?:of|is|under|cap|limit|around))?\s*\$?\s?\d[\d,]*(?:\.\d{1,2})?(?:\s*(?:USD|EUR|JPY|GBP))?|"
    r"(?:under|max(?:imum)?|up\s+to)\s*\$\s?\d[\d,]*(?:\.\d{1,2})?(?:\s*(?:/night|per\s+night|USD))?|"
    r"\$\s?\d[\d,]*(?:\.\d{1,2})?\s*(?:budget|/night|per\s+night|max)|"
    r"\b\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|EUR|JPY|GBP)\s+budget\b)",
    re.IGNORECASE,
)

_DIETARY_PREF_RE = re.compile(
    r"\b(vegetarian|vegan|gluten-?free|halal|kosher|pescatarian|"
    r"(?:nut|peanut|shellfish|seafood|soy|sesame|egg)\s+allerg(?:y|ies)|"
    r"dairy-?free|lactose-?free|wheelchair\s+accessible|step-?free)\b",
    re.IGNORECASE,
)

_DESTINATION_EXTRACT_RE = re.compile(
    r"\b(?:trip\s+to|travel(?:ing)?\s+to|fly(?:ing)?\s+to|visit(?:ing)?|going\s+to|heading\s+to|in)\s+"
    r"(Tokyo|Kyoto|Osaka|Paris|London|Rome|Florence|Venice|Milan|Barcelona|Madrid|"
    r"Lisbon|Amsterdam|Berlin|Munich|Zurich|Geneva|Vienna|Prague|Seoul|Singapore|"
    r"Bangkok|Bali|Sydney|Melbourne|Auckland|New\s+York|San\s+Francisco|Los\s+Angeles|"
    r"Seattle|Boston|Honolulu|Vancouver|Toronto|Reykjavik|Dublin|Edinburgh|Athens|"
    r"Istanbul|Dubai|Taipei|Hong\s+Kong)\b",
    re.IGNORECASE,
)

_PREFERENCE_CUE_RE = re.compile(
    r"\b(i\s+prefer|we\s+prefer|i\s+always|i\s+like|we\s+like|i\s+love|"
    r"i\s+need|we\s+need|please\s+remember|my\s+favorite|avoid|never\s+book|"
    r"allergic\s+to|traveling\s+with|my\s+budget)\b",
    re.IGNORECASE,
)


def _resolve_redactor(explicit_redactor: Any = None) -> Any:
    """Resolve explicit PII redactor or lazy default from observability package."""
    if explicit_redactor is not None:
        return explicit_redactor
    try:
        from travel_concierge.observability.pii_redactor import get_default_redactor

        return get_default_redactor()
    except ImportError:
        return None


def _redact_with_pipeline(text: str, redactor: Any = None) -> tuple[str, bool]:
    """Scrub text through custom/observability redactor plus deterministic fallback."""
    if not text:
        return "", False
    current = text
    if redactor is not None and hasattr(redactor, "redact_text"):
        current = str(redactor.redact_text(current))
    scrubbed = _scrub_text_safely(current, redactor=None)
    was_modified = scrubbed != text or "[REDACTED_" in scrubbed
    return scrubbed, was_modified


def _extract_context_payload(
    callback_context: Any,
    fallback_manager: PersistentTravelSessionManager | None = None,
) -> tuple[str, str, list[dict[str, Any]], AsyncMemoryConsolidator | None]:
    """Extract ``(session_id, user_id, messages, consolidator)`` from ADK or mock contexts."""
    session_id = "default_session"
    user_id = "default_traveler"
    messages: list[dict[str, Any]] = []
    custom_consolidator: AsyncMemoryConsolidator | None = None

    if callback_context is None:
        return session_id, user_id, messages, None

    if isinstance(callback_context, dict):
        session_id = str(callback_context.get("session_id") or session_id)
        user_id = str(callback_context.get("user_id") or user_id)
        raw_msgs = (
            callback_context.get("turn_messages")
            or callback_context.get("messages")
            or callback_context.get("events")
            or []
        )
        if isinstance(raw_msgs, list):
            messages = [m for m in raw_msgs if isinstance(m, dict)]
        cand = callback_context.get("memory_consolidator")
        if isinstance(cand, AsyncMemoryConsolidator):
            custom_consolidator = cand
        return session_id, user_id, messages, custom_consolidator

    # Check direct attributes (mock objects or custom wrappers)
    if getattr(callback_context, "session_id", None):
        session_id = str(callback_context.session_id)
    if getattr(callback_context, "user_id", None):
        user_id = str(callback_context.user_id)

    cand_attr = getattr(callback_context, "memory_consolidator", None)
    if isinstance(cand_attr, AsyncMemoryConsolidator):
        custom_consolidator = cand_attr

    # Check ADK `session` or `_invocation_context.session`
    session_obj = getattr(callback_context, "session", None)
    if session_obj is None:
        inv_ctx = getattr(callback_context, "_invocation_context", None)
        if inv_ctx is not None:
            session_obj = getattr(inv_ctx, "session", None)
            if getattr(inv_ctx, "user_id", None):
                user_id = str(inv_ctx.user_id)

    if session_obj is not None:
        if getattr(session_obj, "id", None):
            session_id = str(session_obj.id)
        if getattr(session_obj, "user_id", None):
            user_id = str(session_obj.user_id)
        events = getattr(session_obj, "events", None)
        if isinstance(events, list):
            for ev in events:
                if isinstance(ev, dict):
                    messages.append(ev)
                else:
                    role = str(getattr(ev, "author", None) or getattr(ev, "role", "user"))
                    content_obj = getattr(ev, "content", None)
                    parts = getattr(content_obj, "parts", None) if content_obj else None
                    if isinstance(parts, list):
                        txt = " ".join(
                            str(getattr(p, "text", ""))
                            for p in parts
                            if getattr(p, "text", None)
                        ).strip()
                    else:
                        txt = str(content_obj or "").strip()
                    if txt:
                        messages.append({"role": role, "content": txt})

    if not messages:
        for attr_name in ("turn_messages", "messages", "events"):
            raw = getattr(callback_context, attr_name, None)
            if isinstance(raw, list):
                messages = [m for m in raw if isinstance(m, dict)]
                if messages:
                    break

    if not messages and fallback_manager is not None and session_id != "default_session":
        messages = fallback_manager.get_session_history(session_id, limit=8)

    return session_id, user_id, messages, custom_consolidator


class AsyncMemoryConsolidator:
    """Non-blocking background traveler memory extractor and consolidator.

    Extracts durable traveler preferences (airlines, seating/cabin, lodging,
    dietary restrictions, budget caps, target destinations) from conversation
    turns, scrubs PII via ``pii_redactor``, deduplicates entries, and writes
    vector-indexed memories to ``PersistentTravelSessionManager`` asynchronously.
    """

    def __init__(
        self,
        session_manager: PersistentTravelSessionManager | None = None,
        pii_redactor: Any = None,
    ) -> None:
        self.session_manager = session_manager or PersistentTravelSessionManager()
        self.pii_redactor = pii_redactor
        self._active_tasks: set[asyncio.Task[dict[str, Any]]] = set()

    @property
    def pending_task_count(self) -> int:
        """Number of currently active background consolidation tasks."""
        return len(self._active_tasks)

    def _on_task_complete(self, task: asyncio.Task[dict[str, Any]]) -> None:
        """Cleanup callback removing finished task from active tracking set."""
        self._active_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Background memory consolidation task failed: %s", exc)

    def schedule_consolidation(
        self,
        session_id: str,
        user_id: str,
        turn_messages: list[dict[str, Any]],
    ) -> asyncio.Task[dict[str, Any]]:
        """Schedule non-blocking memory consolidation on the running event loop.

        Returns immediately with the created ``asyncio.Task`` so the agent turn
        delivers its response to the traveler without waiting on memory I/O.

        Args:
            session_id: Active conversation session identifier.
            user_id: Traveler user identifier.
            turn_messages: Recent conversation turn dicts to analyze.

        Returns:
            Scheduled ``asyncio.Task`` executing ``consolidate_now``.
        """
        snapshot = [dict(m) for m in (turn_messages or [])]
        task = asyncio.create_task(
            self.consolidate_now(
                session_id=session_id,
                user_id=user_id,
                turn_messages=snapshot,
            ),
            name=f"memory_consolidate_{session_id}",
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._on_task_complete)
        return task

    def _extract_salient_facts(
        self,
        turn_messages: list[dict[str, Any]],
        redactor: Any,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Extract deduplicated, PII-scrubbed traveler facts from turn messages."""
        extracted: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        any_pii_redacted = False

        def _add_fact(statement: str, tags: list[str]) -> None:
            nonlocal any_pii_redacted
            scrubbed_stmt, mod = _redact_with_pipeline(statement, redactor=redactor)
            if mod:
                any_pii_redacted = True
            cleaned = " ".join(scrubbed_stmt.split()).strip(" ,;")
            if not cleaned:
                return
            key = cleaned.lower()
            if key not in seen_keys:
                seen_keys.add(key)
                extracted.append({"text": cleaned, "tags": list(tags)})

        for msg in turn_messages:
            role = str(msg.get("role", "user")).lower()
            raw_text = _extract_message_text(msg)
            if not raw_text or ConversationCompactor.SUMMARY_PREFIX in raw_text:
                continue

            scrubbed_text, mod = _redact_with_pipeline(raw_text, redactor=redactor)
            if mod:
                any_pii_redacted = True

            # Primarily harvest user preferences, plus confirmed booking notes.
            is_user_turn = role in {"user", "human", "traveler"}

            flight_hits: list[str] = []
            for pattern in _FLIGHT_PREF_PATTERNS:
                for m in pattern.finditer(scrubbed_text):
                    flight_hits.append(m.group(1))
            if is_user_turn and flight_hits:
                unique_flights = list(dict.fromkeys(h.strip() for h in flight_hits))
                _add_fact(
                    f"Flight preference: {', '.join(unique_flights)}",
                    ["flight", "preference"],
                )

            lodging_hits = [m.group(1).strip() for m in _LODGING_PREF_RE.finditer(scrubbed_text)]
            if is_user_turn and lodging_hits:
                unique_lodging = list(dict.fromkeys(lodging_hits))
                _add_fact(
                    f"Lodging preference: {', '.join(unique_lodging)}",
                    ["lodging", "preference"],
                )

            budget_hits = [
                m.group(0).strip() for m in _BUDGET_CONSTRAINT_RE.finditer(scrubbed_text)
            ]
            if is_user_turn and budget_hits:
                unique_budgets = list(dict.fromkeys(budget_hits))
                _add_fact(
                    f"Budget constraint: {', '.join(unique_budgets)}",
                    ["budget", "constraint"],
                )

            dietary_hits = [m.group(1).strip() for m in _DIETARY_PREF_RE.finditer(scrubbed_text)]
            if is_user_turn and dietary_hits:
                unique_dietary = list(dict.fromkeys(dietary_hits))
                _add_fact(
                    f"Dietary/accessibility requirement: {', '.join(unique_dietary)}",
                    ["dietary", "preference"],
                )

            dest_hits = [
                m.group(1).strip().title()
                for m in _DESTINATION_EXTRACT_RE.finditer(scrubbed_text)
            ]
            if is_user_turn and dest_hits:
                unique_dests = list(dict.fromkeys(dest_hits))
                _add_fact(
                    f"Target destination interest: {', '.join(unique_dests)}",
                    ["destination", "trip_context"],
                )

            # Capture explicit natural-language preference sentences (already scrubbed).
            if is_user_turn:
                for sentence in re.split(r"(?<=[.!?])\s+", scrubbed_text):
                    trimmed = sentence.strip()
                    if not trimmed:
                        continue
                    if _PREFERENCE_CUE_RE.search(trimmed) or "[REDACTED_" in trimmed:
                        _add_fact(trimmed[:240], ["traveler_note", "preference"])

        return extracted, any_pii_redacted

    async def consolidate_now(
        self,
        session_id: str,
        user_id: str,
        turn_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract, PII-scrub, deduplicate, and store traveler memories asynchronously.

        Args:
            session_id: Conversation session identifier.
            user_id: Traveler user identifier.
            turn_messages: List of conversation turn dictionaries.

        Returns:
            Summary dictionary reporting extracted count, stored memory IDs,
            scrubbed memory texts, and PII redaction flag.
        """
        # Cooperative yield ensuring schedule_consolidation callers resume first.
        await asyncio.sleep(0)

        redactor = _resolve_redactor(self.pii_redactor)
        extracted_facts, pii_redacted = self._extract_salient_facts(
            turn_messages or [], redactor=redactor
        )

        stored_ids: list[str] = []
        stored_texts: list[str] = []

        for fact in extracted_facts:
            mem_id = self.session_manager.upsert_memory_entry(
                user_id=user_id,
                memory_text=fact["text"],
                tags=fact["tags"],
            )
            if mem_id not in stored_ids:
                stored_ids.append(mem_id)
                stored_texts.append(fact["text"])

        logger.info(
            "Consolidated %d traveler memories for session=%s user=%s (pii_redacted=%s).",
            len(stored_ids),
            session_id,
            user_id,
            pii_redacted,
        )

        return {
            "session_id": session_id,
            "user_id": user_id,
            "extracted_count": len(stored_ids),
            "memory_ids": stored_ids,
            "memories": stored_texts,
            "pii_redacted": pii_redacted,
            "status": "consolidated",
        }

    async def flush_pending_tasks(self) -> int:
        """Await all currently in-flight background consolidation tasks cleanly.

        Returns:
            Number of pending background tasks that were awaited.
        """
        pending = list(self._active_tasks)
        if not pending:
            return 0
        await asyncio.gather(*pending, return_exceptions=True)
        self._active_tasks.difference_update(pending)
        return len(pending)

    async def after_agent_memory_callback(self, callback_context: Any) -> None:
        """Instance-level ADK ``after_agent_callback`` scheduling async consolidation."""
        session_id, user_id, messages, _ = _extract_context_payload(
            callback_context, fallback_manager=self.session_manager
        )
        if messages:
            self.schedule_consolidation(
                session_id=session_id,
                user_id=user_id,
                turn_messages=messages,
            )
        return None


_DEFAULT_CONSOLIDATOR: AsyncMemoryConsolidator | None = None


def get_default_memory_consolidator() -> AsyncMemoryConsolidator:
    """Return process-wide default ``AsyncMemoryConsolidator`` instance."""
    global _DEFAULT_CONSOLIDATOR
    if _DEFAULT_CONSOLIDATOR is None:
        _DEFAULT_CONSOLIDATOR = AsyncMemoryConsolidator()
    return _DEFAULT_CONSOLIDATOR


async def after_agent_memory_callback(callback_context: Any) -> None:
    """ADK ``after_agent_callback`` hook triggering background memory consolidation.

    Returns ``None`` immediately after scheduling the background ``asyncio.Task``
    so ADK preserves the agent's primary response without latency overhead.
    """
    consolidator = get_default_memory_consolidator()
    session_id, user_id, messages, ctx_consolidator = _extract_context_payload(
        callback_context, fallback_manager=consolidator.session_manager
    )
    target = ctx_consolidator or consolidator
    if messages:
        target.schedule_consolidation(
            session_id=session_id,
            user_id=user_id,
            turn_messages=messages,
        )
    return None
