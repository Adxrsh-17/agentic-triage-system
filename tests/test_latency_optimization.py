"""Unit tests for ESI 1 fast-path latency optimization and profiling."""

import time
import unittest
from unittest.mock import patch

from agent.multi_agent import MultiAgentSystem, _check_esi_1_fast_path
from scripts.evaluate_latency import (
    format_latency_markdown_table,
    percentile,
    summarize_latencies,
)


class TestLatencyOptimization(unittest.TestCase):
    def setUp(self):
        self.agent = MultiAgentSystem()

    def test_fast_path_hypoxia_sub_50ms(self):
        """Critical hypoxia (SpO2 86% < 90%) must trigger ESI 1 fast-path in <50ms."""
        t0 = time.perf_counter()
        res = self.agent.process(
            user_input="Patient with mild cough and dyspnea",
            user_id="test_fast_hypoxia",
            age=55,
            vitals={"spo2": 86, "hr": 105, "rr": 28},
            fast_path=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(res["esi_level"], 1)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertTrue(res["is_emergency"])
        self.assertTrue(res["hitl_required"])
        self.assertTrue(res["awaiting_human_review"])
        self.assertEqual(res["intake_method"], "deterministic_fast_path")
        self.assertLess(res["latency_ms"], 50.0)
        self.assertLess(elapsed_ms, 50.0)
        self.assertTrue(any("SpO2 86%" in trig for trig in res["vital_triggers"]))

    def test_fast_path_bradycardia_sub_50ms(self):
        """Profound bradycardia (HR 32 bpm < 40) must trigger ESI 1 fast-path in <50ms."""
        res = self.agent.process(
            user_input="Patient feels dizzy and weak",
            user_id="test_fast_brady",
            age=68,
            vitals={"hr": 32, "sbp": 95, "dbp": 60, "spo2": 96},
            fast_path=True,
        )
        self.assertEqual(res["esi_level"], 1)
        self.assertLess(res["latency_ms"], 50.0)
        self.assertTrue(any("HR 32 bpm" in trig for trig in res["vital_triggers"]))

    def test_fast_path_decompensated_shock_sub_50ms(self):
        """Decompensated shock (SBP 64 mmHg < 80) must trigger ESI 1 fast-path in <50ms."""
        res = self.agent.process(
            user_input="Patient lightheaded after vomiting",
            user_id="test_fast_shock",
            age=40,
            vitals={"sbp": 64, "dbp": 40, "hr": 125, "spo2": 95},
            fast_path=True,
        )
        self.assertEqual(res["esi_level"], 1)
        self.assertLess(res["latency_ms"], 50.0)
        self.assertTrue(any("SBP 64 mmHg" in trig for trig in res["vital_triggers"]))

    def test_fast_path_unresponsive_resuscitation_keyword(self):
        """Cardiac arrest or unresponsiveness in text must trigger immediate fast-path."""
        res = self.agent.process(
            user_input="Family member found patient unresponsive and pulseless on the kitchen floor",
            user_id="test_fast_unresponsive",
            age=72,
            fast_path=True,
        )
        self.assertEqual(res["esi_level"], 1)
        self.assertLess(res["latency_ms"], 50.0)
        self.assertEqual(res["intake_method"], "deterministic_fast_path")

    def test_fast_path_negation_does_not_trigger(self):
        """Negated resuscitation symptoms ('denies loss of consciousness') must not trigger fast-path."""
        is_fast, reasons, triggers, _ = _check_esi_1_fast_path(
            user_input="Patient has mild sore throat and headache, denies loss of consciousness or shortness of breath",
            vitals={"spo2": 99, "hr": 72, "sbp": 120, "dbp": 80},
        )
        self.assertFalse(is_fast)
        self.assertEqual(len(triggers), 0)

    def test_fast_path_generates_valid_sbar_and_fhir(self):
        """Fast-path must still generate structured SBAR note and valid HL7 FHIR R4 Transaction Bundle."""
        res = self.agent.process(
            user_input="Patient in acute anaphylactic shock with severe stridor following peanut exposure",
            user_id="test_fast_sbar_fhir",
            age=24,
            vitals={"spo2": 85, "hr": 140, "sbp": 70},
            fast_path=True,
        )
        self.assertEqual(res["esi_level"], 1)
        self.assertIn("sbar_note", res)
        self.assertTrue(bool(res["sbar_note"]))
        self.assertIn("(SBAR REPORT)", res["sbar_note"])
        self.assertIn("[S] SITUATION", res["sbar_note"])

        self.assertIn("fhir_bundle", res)
        bundle = res["fhir_bundle"]
        self.assertEqual(bundle.get("resourceType"), "Bundle")
        self.assertEqual(bundle.get("type"), "transaction")
        self.assertGreaterEqual(len(bundle.get("entry", [])), 3)

    def test_fast_path_nurse_approval_flow(self):
        """Nurse sign-off on fast-path pending_state must process cleanly."""
        initial = self.agent.process(
            user_input="Patient in cardiac arrest, CPR in progress",
            user_id="test_fast_nurse_approve",
            age=65,
            fast_path=True,
        )
        self.assertTrue(initial["awaiting_human_review"])
        self.assertIsNotNone(initial["pending_state"])

        approved = self.agent.process(
            pending_state=initial["pending_state"],
            approval_decision="approved",
            approval_note="Resuscitation team active at bedside, intubation completed.",
        )
        self.assertFalse(approved["awaiting_human_review"])
        self.assertIn("Nurse Acceptance Recorded", approved["content"])

    def test_percentile_and_summary_statistics(self):
        """Verify statistical percentile and latency summary computations."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(percentile(data, 50.0), 30.0)
        self.assertEqual(percentile(data, 0.0), 10.0)
        self.assertEqual(percentile(data, 100.0), 50.0)

        summary = summarize_latencies(data)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["mean_ms"], 30.0)
        self.assertEqual(summary["median_p50_ms"], 30.0)
        self.assertEqual(summary["min_ms"], 10.0)
        self.assertEqual(summary["max_ms"], 50.0)

    def test_markdown_formatter_output(self):
        """Verify latency markdown table formatter produces clean table structure without emojis."""
        mock_data = {
            "benchmark_metadata": {
                "split": "test",
                "total_evaluated_cases": 2,
                "esi_1_target_latency_ms": 50.0,
                "hardware_platform": "win32",
            },
            "overall_summary": {
                "baseline": summarize_latencies([5000.0, 6000.0]),
                "optimized": summarize_latencies([2.0, 3000.0]),
                "overall_mean_reduction_pct": 72.5,
                "overall_p99_reduction_pct": 50.0,
            },
            "esi_level_breakdown": {
                f"ESI_{i}": {
                    "baseline": summarize_latencies([4000.0]),
                    "optimized": summarize_latencies([1.5 if i == 1 else 2500.0]),
                    "mean_reduction_pct": 99.9 if i == 1 else 37.5,
                    "speedup_factor": 2666.0 if i == 1 else 1.6,
                }
                for i in range(1, 6)
            },
            "component_node_breakdown": {
                "intake": summarize_latencies([500.0]),
                "risk": summarize_latencies([5.0]),
                "safety": summarize_latencies([50.0]),
                "sbar": summarize_latencies([10.0]),
            },
            "case_level_results": [],
        }
        report = format_latency_markdown_table(mock_data)
        self.assertIn("# Pipeline Latency Profiling and Optimization Benchmark", report)
        self.assertIn("| Mean Latency |", report)
        self.assertIn("ESI 1", report)
        self.assertNotIn("🚀", report)
        self.assertNotIn("⚡", report)


if __name__ == "__main__":
    unittest.main()
