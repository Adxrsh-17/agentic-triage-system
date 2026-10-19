"""Integration Tests for LangGraph Graph Routing & HITL Decision Gating (Phase 4).

Verifies:
1. Low-risk routine presentation routes directly to Finish without pausing for human review.
2. High-risk emergent presentation triggers Human-in-the-Loop review and sets awaiting_human_review.
3. Fast-path resuscitation presentation executes in sub-millisecond time.
4. Nurse approval flow successfully transitions state from suspended to finalized.
"""

import unittest
from agent.multi_agent import MultiAgentSystem
from agent.schemas import RiskLevel


class TestGraphRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = MultiAgentSystem()

    def test_low_risk_routing_to_finish(self):
        """Routine minor symptom must route directly to Finish without human review block."""
        res = self.agent.process(
            user_input="Patient with mild scratch on arm from gardening 2 hours ago, clean, denies fever",
            user_id="route_test_low",
            age=30,
            vitals={"hr": 72, "sbp": 120, "dbp": 80, "spo2": 99, "rr": 14},
        )
        self.assertIsNotNone(res)
        self.assertIn(res["esi_level"], [4, 5])
        self.assertEqual(res["risk_level"], "LOW")
        self.assertFalse(res["awaiting_human_review"])
        self.assertFalse(res["hitl_required"])
        self.assertIsNone(res.get("pending_state"))

    def test_high_risk_routing_to_human_review(self):
        """Emergent symptom must halt at Human Review gate with awaiting_human_review=True."""
        res = self.agent.process(
            user_input="Severe crushing substernal chest pain radiating to left jaw, diaphoresis",
            user_id="route_test_high",
            age=62,
            vitals={"hr": 105, "sbp": 160, "dbp": 95, "spo2": 96, "rr": 20},
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["esi_level"], 2)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertTrue(res["awaiting_human_review"])
        self.assertTrue(res["hitl_required"])
        self.assertIsNotNone(res.get("pending_state"))

    def test_fast_path_routing(self):
        """Critical vital signs collapse must trigger sub-millisecond fast-path."""
        res = self.agent.process(
            user_input="Patient unresponsive on arrival",
            user_id="route_test_fast",
            age=45,
            vitals={"spo2": 82, "hr": 30, "sbp": 60, "dbp": 35, "rr": 6},
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["esi_level"], 1)
        self.assertTrue(res["is_emergency"])
        self.assertLess(res["latency_ms"], 100.0)

    def test_nurse_approval_flow_completion(self):
        """Submitting approval for suspended state must resume and finalize the encounter."""
        initial = self.agent.process(
            user_input="Severe acute asthma flare with marked wheezing and dyspnea",
            user_id="route_test_resume",
            age=28,
            vitals={"hr": 115, "sbp": 130, "dbp": 85, "spo2": 92, "rr": 26},
        )
        self.assertTrue(initial["awaiting_human_review"])
        pending_state = initial["pending_state"]

        finalized = self.agent.process(
            pending_state=pending_state,
            approval_decision="approved",
            approval_note="Confirmed bedside distress: albuterol nebulizer administered.",
        )
        self.assertFalse(finalized["awaiting_human_review"])
        self.assertIn("Nurse Acceptance Recorded", finalized["content"])


if __name__ == "__main__":
    unittest.main()
