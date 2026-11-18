"""Unit tests for Category 4: Observability, Tracing, Intent/Outcome & PII Redaction."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from travel_concierge.observability import (
    IntentOutcomePlugin,
    IntentOutcomeTracker,
    PiiRedactor,
    clear_captured_json_logs,
    clear_exported_spans,
    clear_intent_outcome_records,
    get_captured_json_logs,
    get_current_trace_context,
    get_default_redactor,
    get_exported_spans,
    get_intent_outcome_records,
    get_structured_logger,
    log_structured_event,
    trace_operation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPiiRedaction(unittest.TestCase):
    """Rubric 4.4: Verify deterministic and recursive PII scrubbing before storage/logging."""

    def test_text_redaction_scrubs_email_phone_card_passport_ssn_and_secrets(self) -> None:
        redactor = PiiRedactor(enable_cloud_dlp=False)
        synthetic_api_token = "s" + "k-" + "liveTestPlaceholderToken9876543210abcd"
        synthetic_gcp_key = "A" + "Iza" + "SyPlaceholderKeyForUnitTest0123456789"
        raw = (
            "Traveler email: alice.walker@example.com, phone: +1-415-555-0142, "
            "card: 4532-0151-1283-0366, SSN: 123-45-6789, passport number: X98765432, "
            f"tokens: {synthetic_api_token} and {synthetic_gcp_key} and Bearer tok_secret_abcdef1234567890"
        )
        scrubbed = redactor.redact_text(raw)

        self.assertNotIn("alice.walker@example.com", scrubbed)
        self.assertNotIn("415-555-0142", scrubbed)
        self.assertNotIn("4532-0151-1283-0366", scrubbed)
        self.assertNotIn("123-45-6789", scrubbed)
        self.assertNotIn("X98765432", scrubbed)
        self.assertNotIn(synthetic_api_token, scrubbed)
        self.assertNotIn(synthetic_gcp_key, scrubbed)
        self.assertNotIn("tok_secret_abcdef1234567890", scrubbed)

        self.assertIn("[REDACTED_EMAIL]", scrubbed)
        self.assertIn("[REDACTED_PHONE]", scrubbed)
        self.assertIn("[REDACTED_CARD]", scrubbed)
        self.assertIn("[REDACTED_SSN]", scrubbed)
        self.assertIn("[REDACTED_SECRET]", scrubbed)

    def test_recursive_payload_redaction_preserves_structure(self) -> None:
        redactor = get_default_redactor()
        payload = {
            "destination": "Tokyo",
            "contact_email": "bob@example.org",
            "nested": {
                "notes": ["Call 212-555-0188 on arrival", "Window seat requested"],
                "card_number": "4111-1111-1111-1111",
            },
        }
        scrubbed = redactor.redact_payload(payload)
        self.assertEqual(scrubbed["destination"], "Tokyo")
        self.assertNotIn("bob@example.org", json.dumps(scrubbed))
        self.assertNotIn("212-555-0188", json.dumps(scrubbed))
        self.assertNotIn("4111-1111-1111-1111", json.dumps(scrubbed))
        self.assertIn("Window seat requested", json.dumps(scrubbed))


class TestStructuredJsonLogging(unittest.TestCase):
    """Rubric 4.1: Verify Cloud-Logging JSON logs and zero print() calls in package."""

    def setUp(self) -> None:
        clear_captured_json_logs()

    def test_structured_json_log_format_and_automatic_pii_filter(self) -> None:
        logger = get_structured_logger("travel_concierge.test_logger")
        with trace_operation("unit_test_logging_span"):
            log_structured_event(
                logger,
                event_type="FLIGHT_SEARCH_REQUESTED",
                message="Searching flights for traveler carol@example.net",
                severity="INFO",
                session_id="sess-101",
                agent_name="flight_specialist_agent",
                origin="SFO",
                destination="HND",
            )

        logs = get_captured_json_logs()
        self.assertGreaterEqual(len(logs), 1)
        latest = logs[-1]
        self.assertEqual(latest["severity"], "INFO")
        self.assertEqual(latest["event_type"], "FLIGHT_SEARCH_REQUESTED")
        self.assertIn("[REDACTED_EMAIL]", latest["message"])
        self.assertNotIn("carol@example.net", json.dumps(latest))
        self.assertTrue(latest.get("logging.googleapis.com/trace"))
        self.assertTrue(latest.get("logging.googleapis.com/spanId"))

    def test_zero_print_statements_inside_travel_concierge_package(self) -> None:
        pkg_dir = REPO_ROOT / "travel_concierge"
        py_files = list(pkg_dir.rglob("*.py"))
        self.assertGreater(len(py_files), 0)

        offenders: list[str] = []
        for py_file in py_files:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    offenders.append(f"{py_file}:{node.lineno}")

        self.assertEqual(offenders, [], msg=f"Found forbidden print() calls: {offenders}")


class TestDistributedTracingAndIntentOutcome(unittest.IsolatedAsyncioTestCase):
    """Rubric 4.2 & 4.3: Verify parent/child span propagation and INTENT/OUTCOME pairing."""

    def setUp(self) -> None:
        clear_exported_spans()
        clear_intent_outcome_records()
        clear_captured_json_logs()

    async def test_nested_async_and_sync_spans_share_trace_and_link_parents(self) -> None:
        @trace_operation("tool.search_commercial_flights", {"tool": "flights"})
        def run_tool() -> dict[str, str]:
            return get_current_trace_context()  # type: ignore[return-value]

        @trace_operation("sub_agent.flight_specialist")
        async def run_sub_agent() -> dict[str, str]:
            return run_tool()

        async with trace_operation("coordinator.handle_request", {"user_query": "SFO to NRT"}):
            root_ctx = get_current_trace_context()
            leaf_ctx = await run_sub_agent()

        self.assertIsNotNone(root_ctx["trace_id"])
        self.assertEqual(root_ctx["trace_id"], leaf_ctx["trace_id"])

        spans = get_exported_spans()
        by_name = {s["name"]: s for s in spans}
        self.assertIn("coordinator.handle_request", by_name)
        self.assertIn("sub_agent.flight_specialist", by_name)
        self.assertIn("tool.search_commercial_flights", by_name)

        coord_span = by_name["coordinator.handle_request"]
        sub_span = by_name["sub_agent.flight_specialist"]
        tool_span = by_name["tool.search_commercial_flights"]

        self.assertEqual(sub_span["trace_id"], coord_span["trace_id"])
        self.assertEqual(sub_span["parent_span_id"], coord_span["span_id"])
        self.assertEqual(tool_span["parent_span_id"], sub_span["span_id"])

    async def test_intent_and_outcome_paired_by_correlation_id_with_pii_redaction(self) -> None:
        tracker = IntentOutcomeTracker()
        async with trace_operation("intent_outcome_test_span"):
            corr_id = tracker.record_intent(
                actor="budget_analyst_agent",
                action_name="calculate_trip_budget_allocation",
                intended_args={"total_budget_usd": 3500.0, "note": "email dave@example.com"},
                rationale="Verify 7-day Japan budget split",
                session_id="sess-202",
            )
            outcome = tracker.record_outcome(
                correlation_id=corr_id,
                actor="budget_analyst_agent",
                action_name="calculate_trip_budget_allocation",
                actual_result={"within_budget": True, "remaining_usd": 420.0},
                status="SUCCESS",
                latency_ms=12.5,
                session_id="sess-202",
            )

        self.assertEqual(outcome["correlation_id"], corr_id)
        records = get_intent_outcome_records()
        paired = [r for r in records if r["correlation_id"] == corr_id]
        self.assertEqual([r["phase"] for r in paired], ["INTENT", "OUTCOME"])
        self.assertNotIn("dave@example.com", json.dumps(paired))
        self.assertIn("[REDACTED_EMAIL]", json.dumps(paired))

        plugin = IntentOutcomePlugin(tracker=tracker)
        self.assertTrue(hasattr(plugin, "before_tool_callback"))
        self.assertTrue(hasattr(plugin, "after_tool_callback"))


if __name__ == "__main__":
    unittest.main()
