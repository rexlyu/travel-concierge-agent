"""Unit tests for the automated Golden Dataset evaluation & regression harness (Rubric 5.1)."""

from __future__ import annotations

from pathlib import Path
import unittest

from evals.run_evals import (
    compute_trajectory_score,
    load_eval_config,
    load_eval_dataset,
    run_offline_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestEvaluationDatasetAndConfig(unittest.TestCase):
    """Verify ADK EvalSet schema structure, golden scenario coverage, and thresholds."""

    def test_golden_dataset_has_at_least_eight_adk_eval_cases(self) -> None:
        dataset = load_eval_dataset(REPO_ROOT / "evals" / "golden_dataset.evalset.json")
        self.assertEqual(dataset["eval_set_id"], "travel_concierge_golden_evalset_v1")
        cases = dataset["eval_cases"]
        self.assertGreaterEqual(len(cases), 8)

        eval_ids = {c["eval_id"] for c in cases}
        self.assertEqual(len(eval_ids), len(cases))

        for case in cases:
            self.assertIn("eval_id", case)
            self.assertIn("conversation", case)
            self.assertGreaterEqual(len(case["conversation"]), 1)
            first_turn = case["conversation"][0]
            self.assertIn("user_content", first_turn)
            self.assertIn("intermediate_data", first_turn)
            self.assertIn("tool_uses", first_turn["intermediate_data"])
            self.assertIn("final_response", first_turn)

    def test_eval_config_thresholds(self) -> None:
        thresholds = load_eval_config(REPO_ROOT / "evals" / "test_config.json")
        self.assertGreaterEqual(thresholds["tool_trajectory_avg_score"], 0.9)
        self.assertGreaterEqual(thresholds["response_match_score"], 0.8)

    def test_compute_trajectory_score_f1_behavior(self) -> None:
        self.assertEqual(compute_trajectory_score([], []), 1.0)
        self.assertEqual(
            compute_trajectory_score(
                ["search_commercial_flights", "calculate_trip_budget_allocation"],
                ["search_commercial_flights", "calculate_trip_budget_allocation"],
            ),
            1.0,
        )
        self.assertEqual(compute_trajectory_score(["search_commercial_flights"], []), 0.0)
        partial = compute_trajectory_score(
            ["search_commercial_flights", "search_verified_accommodations"],
            ["search_commercial_flights"],
        )
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 1.0)


class TestEvaluationRegressionExecution(unittest.IsolatedAsyncioTestCase):
    """Verify end-to-end golden harness passes healthy baseline and catches regressions."""

    async def test_offline_golden_suite_passes_all_cases(self) -> None:
        report = await run_offline_evaluation()
        self.assertTrue(report["gate_passed"], msg=f"Golden suite failed: {report}")
        self.assertEqual(report["failed_cases"], 0)
        self.assertGreaterEqual(report["total_cases"], 8)
        self.assertGreaterEqual(
            report["metrics"]["tool_trajectory_avg_score"],
            report["thresholds"]["tool_trajectory_avg_score"],
        )
        self.assertGreaterEqual(
            report["metrics"]["response_match_score"],
            report["thresholds"]["response_match_score"],
        )

    async def test_simulated_regression_trips_ci_gate(self) -> None:
        regressed_report = await run_offline_evaluation(simulate_regression=True)
        self.assertFalse(regressed_report["gate_passed"])
        self.assertGreater(regressed_report["failed_cases"], 0)


if __name__ == "__main__":
    unittest.main()
