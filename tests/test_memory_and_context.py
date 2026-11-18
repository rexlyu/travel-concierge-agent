"""Unit tests for Category 2: System Constitution, Compaction, Persistent State & Async Memory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from travel_concierge import prompts
from travel_concierge.memory import (
    AsyncMemoryConsolidator,
    ConversationCompactor,
    PersistentTravelSessionManager,
    after_agent_memory_callback,
    build_adk_compaction_config,
    build_context_cache_config,
)


class TestSystemInstructionsConstitution(unittest.TestCase):
    """Rubric 2.1: Verify AGENT_CONSTITUTION and specialist instructions."""

    def test_constitution_contains_all_mandatory_governance_sections(self) -> None:
        constitution = prompts.AGENT_CONSTITUTION
        required_headings = [
            "Persona & Mission",
            "Domain Expertise",
            "Financial & Budget Integrity Rules",
            "Safety/Privacy/PII Constraints",
            "Mandatory Human-in-the-Loop Escalation Rules",
            "Tool Error Recovery Protocol",
        ]
        for heading in required_headings:
            self.assertIn(heading, constitution)

        for attr in (
            "COORDINATOR_INSTRUCTION",
            "FLIGHT_SPECIALIST_INSTRUCTION",
            "LODGING_SPECIALIST_INSTRUCTION",
            "ADVISORY_SPECIALIST_INSTRUCTION",
            "BUDGET_ANALYST_INSTRUCTION",
            "CRITIC_INSTRUCTION",
        ):
            val = getattr(prompts, attr)
            self.assertIsInstance(val, str)
            self.assertGreater(len(val), 100)


class TestHistoryCompactionAndContextCache(unittest.TestCase):
    """Rubric 2.2: Verify EventsCompactionConfig, Vertex Context Caching, and sliding window."""

    def test_adk_compaction_and_context_cache_configs(self) -> None:
        adk_cfg = build_adk_compaction_config(compaction_interval=5, overlap_size=2)
        self.assertEqual(adk_cfg.compaction_interval, 5)
        self.assertEqual(adk_cfg.overlap_size, 2)

        cache_cfg = build_context_cache_config(ttl_seconds=3600, min_tokens=2048)
        self.assertTrue(cache_cfg["enabled"])
        self.assertEqual(cache_cfg["min_tokens"], 2048)
        self.assertEqual(cache_cfg["model"], "gemini-2.5-pro")

    def test_sliding_window_compactor_reduces_tokens_and_retains_key_facts(self) -> None:
        compactor = ConversationCompactor(max_tokens=260, keep_recent_turns=3)
        messages = [
            {
                "role": "user",
                "content": (
                    "We are planning a trip to Tokyo and Kyoto departing 2026-10-10 with a "
                    "$3,500 USD budget cap. I need vegetarian meals, aisle seat, and ANA flights."
                ),
            },
            {
                "role": "assistant",
                "content": "Confirmed reference BK-77881 noted while comparing Shinkansen passes.",
            },
        ]
        for idx in range(8):
            messages.append(
                {
                    "role": "user" if idx % 2 == 0 else "assistant",
                    "content": (
                        f"Detailed discussion turn {idx}: comparing temple opening hours, "
                        "local transit cards, luggage forwarding timings, and museum tickets."
                    ),
                }
            )

        result = compactor.compact_history(messages)
        self.assertTrue(result["was_compacted"])
        self.assertLess(result["compacted_tokens"], result["original_tokens"])
        self.assertIn("[COMPACTED_CONVERSATION_SUMMARY]", result["summary"])
        self.assertIn("Tokyo", result["summary"])
        self.assertIn("vegetarian", result["summary"].lower())
        self.assertIn("BK-77881", result["summary"])


class TestPersistentSessionAndAsyncMemory(unittest.IsolatedAsyncioTestCase):
    """Rubric 2.3 & 2.4: Verify SQLite durability across instances and non-blocking consolidation."""

    async def test_sqlite_persistence_and_semantic_memory_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_file = str(Path(tmp) / "sessions.sqlite3")

            mgr_one = PersistentTravelSessionManager(db_path=db_file)
            mgr_one.save_turn("sess-alpha", "user-42", "user", "Looking for ryokans in Kyoto.")
            mgr_one.update_session_state("sess-alpha", {"budget_limit_usd": 3200.0, "city": "Kyoto"})
            mgr_one.upsert_memory_entry(
                user_id="user-42",
                memory_text="Traveler prefers boutique ryokan lodging with breakfast included.",
                tags=["lodging", "kyoto"],
            )
            mgr_one.upsert_memory_entry(
                user_id="user-42",
                memory_text="Traveler prefers window seats on long-haul transpacific flights.",
                tags=["flights"],
            )

            # Instantiate a brand-new manager pointing to the same SQLite database
            mgr_two = PersistentTravelSessionManager(db_path=db_file)
            history = mgr_two.get_session_history("sess-alpha")
            state = mgr_two.get_session_state("sess-alpha")
            self.assertEqual(len(history), 1)
            self.assertEqual(state["budget_limit_usd"], 3200.0)

            lodging_hits = mgr_two.retrieve_relevant_memories(
                user_id="user-42",
                query="traditional hotel or ryokan stay preference",
                top_k=2,
            )
            self.assertGreaterEqual(len(lodging_hits), 1)
            self.assertIn("ryokan", json.dumps(lodging_hits[0]).lower())

            adk_service = mgr_two.create_adk_session_service()
            self.assertIsNotNone(adk_service)

    async def test_async_memory_consolidator_runs_in_background_and_scrubs_pii(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_file = str(Path(tmp) / "async_mem.sqlite3")
            mgr = PersistentTravelSessionManager(db_path=db_file)
            consolidator = AsyncMemoryConsolidator(session_manager=mgr)

            turns = [
                {
                    "role": "user",
                    "content": (
                        "I am flying to Lisbon with a $2,800 USD budget, vegetarian diet, "
                        "aisle seat preference, email secret.user@example.com, and phone 415-555-0177."
                    ),
                }
            ]
            task = consolidator.schedule_consolidation(
                session_id="sess-bg",
                user_id="user-bg",
                turn_messages=turns,
            )
            self.assertIsInstance(task, asyncio.Task)

            flushed = await consolidator.flush_pending_tasks()
            self.assertGreaterEqual(flushed, 0)
            await task

            memories = mgr.retrieve_relevant_memories("user-bg", "Lisbon vegetarian budget", top_k=5)
            serialized = json.dumps(memories)
            self.assertNotIn("secret.user@example.com", serialized)
            self.assertNotIn("415-555-0177", serialized)
            self.assertTrue(callable(after_agent_memory_callback))


if __name__ == "__main__":
    unittest.main()
