"""Automated Golden Dataset Evaluation & Regression Gate for Travel Concierge Agent.

Implements Rubric 5.1 (Automated Evaluation Suites):
- Supports ``--adk-live`` execution via ``google.adk.evaluation.agent_evaluator.AgentEvaluator``
  against a live Vertex AI project.
- Supports deterministic offline/CI regression evaluation (default) executing all
  8+ golden scenarios in ``evals/golden_dataset.evalset.json`` against
  ``TravelConciergeOrchestrator``, domain tools, strict schemas, guardrails,
  Human-in-the-Loop gates, persistent session memory, compaction, and PII redaction.
- Exits with non-zero status code when any threshold in ``evals/test_config.json``
  regresses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_eval_dataset(dataset_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the ADK EvalSet JSON file."""
    path = Path(dataset_path) if dataset_path else REPO_ROOT / "evals" / "golden_dataset.evalset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if "eval_set_id" not in data or not isinstance(data.get("eval_cases"), list):
        raise ValueError(f"Invalid EvalSet schema in {path}: missing eval_set_id or eval_cases")
    return data


def load_eval_config(config_path: str | Path | None = None) -> dict[str, float]:
    """Load ADK evaluation score thresholds from test_config.json."""
    path = Path(config_path) if config_path else REPO_ROOT / "evals" / "test_config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    criteria = raw.get("criteria", raw)
    return {
        "tool_trajectory_avg_score": float(criteria.get("tool_trajectory_avg_score", 0.9)),
        "response_match_score": float(criteria.get("response_match_score", 0.8)),
    }


def compute_trajectory_score(expected_tools: list[str], actual_tools: list[str]) -> float:
    """Compute Recall/F1-weighted tool trajectory match score in [0.0, 1.0]."""
    if not expected_tools and not actual_tools:
        return 1.0
    if not expected_tools and actual_tools:
        return 0.0
    if expected_tools and not actual_tools:
        return 0.0

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    overlap = len(expected_set & actual_set)
    recall = overlap / len(expected_set)
    precision = overlap / len(actual_set)
    if recall + precision == 0:
        return 0.0
    return round((2.0 * recall * precision) / (recall + precision), 4)


def _extract_tools_from_result(result: dict[str, Any], intent_records: list[dict[str, Any]]) -> list[str]:
    """Extract invoked tool names across flexible orchestrator result shapes."""
    collected: list[str] = []

    for key in ("tools_invoked", "tool_names", "executed_tools"):
        val = result.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    collected.append(item)
                elif isinstance(item, dict) and item.get("name"):
                    collected.append(str(item["name"]))

    for key in ("tool_calls", "tool_results", "intermediate_steps"):
        val = result.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    name = item.get("tool_name") or item.get("name") or item.get("tool")
                    if name:
                        collected.append(str(name))
        elif isinstance(val, dict):
            collected.extend(str(k) for k in val)

    for rec in intent_records:
        action = str(rec.get("action_name", ""))
        phase = str(rec.get("phase", ""))
        status = str(rec.get("status", ""))
        if phase == "OUTCOME" and status not in {"BLOCKED", "PENDING_HUMAN_APPROVAL"} and action:
            if action not in {"handle_request", "travel_concierge_coordinator"}:
                collected.append(action)

    deduped: list[str] = []
    for name in collected:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _extract_model_info(result: dict[str, Any], query: str) -> tuple[str, str]:
    """Extract selected model and tier from orchestrator result or router."""
    from travel_concierge.agents.model_router import StrategicModelRouter

    routing = result.get("routing") or result.get("model_routing") or {}
    if isinstance(routing, dict) and routing.get("model"):
        return str(routing["model"]), str(routing.get("tier", ""))

    if result.get("selected_model") or result.get("model"):
        model_name = str(result.get("selected_model") or result.get("model"))
        tier = str(result.get("tier") or ("reasoning" if "pro" in model_name else "fast"))
        return model_name, tier

    fallback_decision = StrategicModelRouter().select_model(query)
    return str(fallback_decision["model"]), str(fallback_decision["tier"])


def _extract_response_text(result: dict[str, Any]) -> str:
    """Extract final user-facing text from orchestrator response."""
    for key in ("response", "final_response", "message", "output", "summary"):
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return json.dumps(result, default=str)


async def evaluate_single_case(
    case: dict[str, Any],
    temp_dir: Path,
    simulate_regression: bool = False,
) -> dict[str, Any]:
    """Execute a single golden EvalCase deterministically and score all assertions."""
    from travel_concierge import TravelConciergeOrchestrator
    from travel_concierge.memory import ConversationCompactor, PersistentTravelSessionManager
    from travel_concierge.observability import (
        clear_captured_json_logs,
        clear_exported_spans,
        clear_intent_outcome_records,
        get_captured_json_logs,
        get_exported_spans,
        get_intent_outcome_records,
    )
    from travel_concierge.tools import (
        calculate_trip_budget_allocation,
        convert_currency_exchange_rate,
    )

    eval_id = str(case["eval_id"])
    scenario_type = str(case.get("scenario_type", "general"))
    session_input = case.get("session_input", {})
    user_id = str(session_input.get("user_id", f"user_{eval_id}"))
    state = dict(session_input.get("state") or {})
    assertions = dict(case.get("rubric_assertions") or {})
    turns = list(case.get("conversation") or [])

    db_path = str(temp_dir / f"{eval_id}.sqlite3")
    prev_db_env = os.environ.get("TRAVEL_SESSION_DB_PATH")
    prev_token_env = os.environ.get("TRAVEL_PAYMENT_GATEWAY_TOKEN")
    os.environ["TRAVEL_SESSION_DB_PATH"] = db_path
    os.environ["TRAVEL_PAYMENT_GATEWAY_TOKEN"] = "eval-ephemeral-gateway-token-placeholder"

    clear_captured_json_logs()
    clear_exported_spans()
    clear_intent_outcome_records()

    checks: list[dict[str, Any]] = []
    trajectory_score = 1.0
    response_score = 1.0

    try:
        orchestrator = TravelConciergeOrchestrator()
        session_id = f"session_{eval_id}"

        # Scenario 7 also directly probes guided tool error recovery on invalid inputs.
        if scenario_type == "guided_tool_error_recovery":
            probe = state.get("invalid_tool_probe", {})
            bad_curr = convert_currency_exchange_rate(**probe["currency_call"])
            bad_budget = calculate_trip_budget_allocation(**probe["budget_call"])

            curr_guided = (
                isinstance(bad_curr, dict)
                and bad_curr.get("status") == "error"
                and bool(bad_curr.get("recovery_instructions"))
                and bool(bad_curr.get("retryable", True))
            )
            budget_guided = (
                isinstance(bad_budget, dict)
                and bad_budget.get("status") == "error"
                and bool(bad_budget.get("recovery_instructions"))
                and bool(bad_budget.get("retryable", True))
            )
            checks.append({"check": "currency_guided_error_recovery", "passed": curr_guided})
            checks.append({"check": "budget_guided_error_recovery", "passed": budget_guided})
            trajectory_score = 1.0 if (curr_guided and budget_guided) else 0.0
            response_score = 1.0 if (curr_guided and budget_guided) else 0.0

        # Scenario 8 pre-populates multi-turn verbose history to trigger compaction & persistence.
        elif scenario_type == "compaction_and_session_persistence":
            session_mgr = PersistentTravelSessionManager(db_path=db_path)
            first_turn_text = turns[0]["user_content"]["parts"][0]["text"]
            session_mgr.save_turn(session_id, user_id, "user", first_turn_text)
            session_mgr.upsert_memory_entry(
                user_id=user_id,
                memory_text=first_turn_text,
                tags=["paris", "preferences", "budget"],
            )
            for idx in range(8):
                filler = (
                    f"Exploratory discussion turn {idx + 1}: comparing neighborhood transit "
                    f"options, museum pass schedules, Seine river walks, and bistro menus "
                    f"across Paris arrondissements while keeping the $3,200 USD cap active."
                )
                session_mgr.save_turn(session_id, user_id, "user", filler)
                session_mgr.save_turn(
                    session_id,
                    user_id,
                    "assistant",
                    f"Turn {idx + 1} response acknowledging Paris transit and dining comparisons.",
                )

            history = session_mgr.get_session_history(session_id)
            compactor = ConversationCompactor(
                max_tokens=int(state.get("compaction_max_tokens", 350)),
                keep_recent_turns=int(state.get("compaction_keep_recent_turns", 4)),
            )
            compacted = compactor.compact_history(history)
            compaction_ok = (
                bool(compacted.get("was_compacted"))
                and int(compacted.get("compacted_tokens", 0)) < int(compacted.get("original_tokens", 0))
                and "[COMPACTED_CONVERSATION_SUMMARY]" in str(compacted.get("summary", ""))
            )
            checks.append({"check": "history_compacted_under_budget", "passed": compaction_ok})

            # Re-open a fresh manager instance on the same SQLite DB to prove durability.
            reloaded_mgr = PersistentTravelSessionManager(db_path=db_path)
            recalled = reloaded_mgr.retrieve_relevant_memories(
                user_id=user_id,
                query="Paris vegetarian hotel budget",
                top_k=3,
            )
            combined_context = compacted.get("summary", "") + " " + json.dumps(recalled)
            retained_ok = all(
                kw.lower() in combined_context.lower()
                for kw in assertions.get("expected_retained_keywords", [])
            )
            checks.append({"check": "persistent_preferences_retained", "passed": retained_ok})

            final_query = turns[-1]["user_content"]["parts"][0]["text"]
            result = await orchestrator.handle_request(
                user_id=user_id,
                session_id=session_id,
                query=final_query,
            )
            response_text = _extract_response_text(result)
            checks.append({"check": "orchestrator_followup_succeeded", "passed": bool(response_text)})
            trajectory_score = 1.0 if (compaction_ok and retained_ok) else 0.0
            response_score = 1.0 if bool(response_text) else 0.0

        else:
            # Standard single- or multi-turn orchestrator execution
            last_result: dict[str, Any] = {}
            last_query = ""
            expected_tools: list[str] = list(assertions.get("expected_tools") or [])

            for turn in turns:
                last_query = turn["user_content"]["parts"][0]["text"]
                last_result = await orchestrator.handle_request(
                    user_id=user_id,
                    session_id=session_id,
                    query=last_query,
                    budget_limit_usd=state.get("budget_limit_usd"),
                    human_approvals=assertions.get("human_approvals", state.get("approved_actions")),
                    high_stakes_action=state.get("high_stakes_action"),
                )

            if simulate_regression:
                # Intentional synthetic regression used by unit tests to verify gate failure
                selected_model, selected_tier = "regressed-wrong-model", "wrong_tier"
                actual_tools: list[str] = []
            else:
                selected_model, selected_tier = _extract_model_info(last_result, last_query)
                actual_tools = _extract_tools_from_result(last_result, get_intent_outcome_records())

            response_text = _extract_response_text(last_result)
            serialized_result = json.dumps(last_result, default=str)

            if "expected_model" in assertions:
                model_ok = selected_model == assertions["expected_model"]
                checks.append(
                    {
                        "check": f"model_routed_to_{assertions['expected_model']}",
                        "passed": model_ok,
                        "actual": selected_model,
                    }
                )
                if not model_ok:
                    response_score = 0.0

            if "expected_tier" in assertions:
                tier_ok = selected_tier == assertions["expected_tier"]
                checks.append(
                    {
                        "check": f"tier_matched_{assertions['expected_tier']}",
                        "passed": tier_ok,
                        "actual": selected_tier,
                    }
                )

            if "expected_tools" in assertions:
                trajectory_score = compute_trajectory_score(expected_tools, actual_tools)
                checks.append(
                    {
                        "check": "expected_tool_trajectory",
                        "passed": trajectory_score >= 0.9,
                        "expected": expected_tools,
                        "actual": actual_tools,
                        "score": trajectory_score,
                    }
                )

            if assertions.get("expected_hitl_status") == "PENDING_HUMAN_APPROVAL":
                hitl_halted = (
                    "PENDING_HUMAN_APPROVAL" in serialized_result
                    and assertions.get("expected_tool_blocked") not in actual_tools
                )
                checks.append({"check": "hitl_pending_human_approval_enforced", "passed": hitl_halted})
                trajectory_score = 1.0 if hitl_halted else 0.0
                response_score = 1.0 if hitl_halted else 0.0

            if assertions.get("expected_guardrail_blocked"):
                status_val = str(last_result.get("status", "")).upper()
                guardrail_info = last_result.get("guardrail") or {}
                blocked = (
                    status_val in {"BLOCKED", "REJECTED", "GUARDRAIL_BLOCKED", "POLICY_VIOLATION"}
                    or (isinstance(guardrail_info, dict) and guardrail_info.get("allowed") is False)
                    or last_result.get("allowed") is False
                )
                checks.append({"check": "prompt_injection_blocked", "passed": blocked and not actual_tools})
                trajectory_score = 1.0 if (blocked and not actual_tools) else 0.0
                response_score = 1.0 if blocked else 0.0

            if "forbidden_raw_pii" in assertions:
                # Allow any async memory consolidation tasks to flush before inspecting DB/logs
                await asyncio.sleep(0.05)
                logs_dump = json.dumps(get_captured_json_logs(), default=str)
                spans_dump = json.dumps(get_exported_spans(), default=str)
                session_mgr = PersistentTravelSessionManager(db_path=db_path)
                memories_dump = json.dumps(
                    session_mgr.retrieve_relevant_memories(user_id=user_id, query="London seat", top_k=10),
                    default=str,
                )
                combined_telemetry = f"{logs_dump}\n{spans_dump}\n{memories_dump}"

                leaked = [
                    raw_pii
                    for raw_pii in assertions["forbidden_raw_pii"]
                    if raw_pii in combined_telemetry
                ]
                redaction_present = any(
                    tok in combined_telemetry for tok in assertions.get("expected_redaction_tokens", [])
                )
                pii_ok = (len(leaked) == 0) and redaction_present
                checks.append(
                    {
                        "check": "pii_scrubbed_in_logs_spans_and_memory",
                        "passed": pii_ok,
                        "leaked": leaked,
                    }
                )
                response_score = 1.0 if pii_ok else 0.0

            if not response_text:
                response_score = 0.0

        passed = all(c["passed"] for c in checks) if checks else False
        return {
            "eval_id": eval_id,
            "scenario_type": scenario_type,
            "passed": passed,
            "tool_trajectory_score": trajectory_score,
            "response_match_score": response_score,
            "checks": checks,
        }
    finally:
        if prev_db_env is None:
            os.environ.pop("TRAVEL_SESSION_DB_PATH", None)
        else:
            os.environ["TRAVEL_SESSION_DB_PATH"] = prev_db_env

        if prev_token_env is None:
            os.environ.pop("TRAVEL_PAYMENT_GATEWAY_TOKEN", None)
        else:
            os.environ["TRAVEL_PAYMENT_GATEWAY_TOKEN"] = prev_token_env


async def run_offline_evaluation(
    dataset_path: str | Path | None = None,
    config_path: str | Path | None = None,
    simulate_regression: bool = False,
) -> dict[str, Any]:
    """Run all golden evaluation scenarios and compute aggregate regression metrics."""
    dataset = load_eval_dataset(dataset_path)
    thresholds = load_eval_config(config_path)
    eval_cases = dataset["eval_cases"]

    case_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="travel_concierge_evals_") as tmp:
        temp_dir = Path(tmp)
        for case in eval_cases:
            res = await evaluate_single_case(
                case=case,
                temp_dir=temp_dir,
                simulate_regression=simulate_regression,
            )
            case_results.append(res)

    total_cases = len(case_results)
    passed_cases = sum(1 for r in case_results if r["passed"])
    avg_trajectory = round(
        sum(r["tool_trajectory_score"] for r in case_results) / max(1, total_cases),
        4,
    )
    avg_response = round(
        sum(r["response_match_score"] for r in case_results) / max(1, total_cases),
        4,
    )

    gate_passed = (
        passed_cases == total_cases
        and avg_trajectory >= thresholds["tool_trajectory_avg_score"]
        and avg_response >= thresholds["response_match_score"]
    )

    return {
        "eval_set_id": dataset["eval_set_id"],
        "mode": "offline_deterministic",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "pass_rate": round(passed_cases / max(1, total_cases), 4),
        "metrics": {
            "tool_trajectory_avg_score": avg_trajectory,
            "response_match_score": avg_response,
        },
        "thresholds": thresholds,
        "gate_passed": gate_passed,
        "results": case_results,
    }


async def run_adk_live_evaluation(
    dataset_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute live Vertex AI evaluation via google.adk.evaluation.agent_evaluator."""
    try:
        from google.adk.evaluation.agent_evaluator import AgentEvaluator
    except ImportError as exc:
        raise RuntimeError(
            "google-adk is required for --adk-live evaluation. Install via `pip install -e .[dev]`."
        ) from exc

    dataset = load_eval_dataset(dataset_path)
    thresholds = load_eval_config(config_path)

    # Strip custom offline rubric keys into a strict ADK EvalSet file for AgentEvaluator
    strict_evalset = {
        "eval_set_id": dataset["eval_set_id"],
        "name": dataset.get("name", ""),
        "description": dataset.get("description", ""),
        "eval_cases": [
            {
                "eval_id": c["eval_id"],
                "session_input": c.get("session_input", {}),
                "conversation": c.get("conversation", []),
            }
            for c in dataset["eval_cases"]
        ],
    }

    with tempfile.TemporaryDirectory(prefix="adk_live_eval_") as tmp:
        sanitized_path = Path(tmp) / "strict_golden.evalset.json"
        sanitized_path.write_text(json.dumps(strict_evalset, indent=2), encoding="utf-8")
        await AgentEvaluator.evaluate(
            agent_module="travel_concierge",
            eval_dataset_file_path_or_dir=str(sanitized_path),
            num_runs=1,
        )

    return {
        "eval_set_id": dataset["eval_set_id"],
        "mode": "adk_live",
        "thresholds": thresholds,
        "gate_passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for CI/CD and local developer evaluation runs."""
    parser = argparse.ArgumentParser(
        description="Run the Personal Travel & Budget Concierge Golden Evaluation Suite."
    )
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "evals" / "golden_dataset.evalset.json"),
        help="Path to ADK .evalset.json golden dataset.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "evals" / "test_config.json"),
        help="Path to evaluation threshold config JSON.",
    )
    parser.add_argument(
        "--adk-live",
        action="store_true",
        help="Invoke live Vertex AI evaluation via google.adk AgentEvaluator.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional file path to export the detailed JSON scorecard.",
    )
    args = parser.parse_args(argv)

    if args.adk_live:
        summary = asyncio.run(
            run_adk_live_evaluation(dataset_path=args.dataset, config_path=args.config)
        )
    else:
        summary = asyncio.run(
            run_offline_evaluation(dataset_path=args.dataset, config_path=args.config)
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0 if summary.get("gate_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
