"""End-to-end tests for LangGraph multi-agent triage system."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.multi_agent import MultiAgentSystem, get_multi_agent


class TestMultiAgentSystem(unittest.TestCase):
    def test_multi_agent_system_initialization(self):
        """Verify that MultiAgentSystem compiles the LangGraph StateGraph successfully."""
        agent = get_multi_agent()
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.graph)

    def test_low_risk_triage_flow(self):
        """Verify low risk clinical presentation completes through finish node with OTC recommendations."""
        agent = get_multi_agent()
        res = agent.process(
            user_input="Patient has mild headache and runny nose for 1 day, no fever or chest pain",
            user_id="test_pt_low",
            age=22,
            sex="female",
            conditions=[],
        )
        self.assertIsNotNone(res)
        self.assertIn(res["risk_level"], ["LOW", "MEDIUM"])
        self.assertIn(res["esi_level"], [4, 5])
        self.assertFalse(res["is_emergency"])
        self.assertFalse(res["awaiting_human_review"])
        self.assertGreaterEqual(len(res["medications"]), 1)
        self.assertIn("Consult a qualified healthcare professional", res["content"])

    def test_high_risk_hitl_triage_flow(self):
        """Verify severe emergent presentation triggers HITL review requirement and emergency guidance."""
        agent = get_multi_agent()
        res = agent.process(
            user_input="Crushing chest pain radiating to neck and difficulty breathing since 30 minutes",
            user_id="test_pt_high",
            age=60,
            sex="male",
            conditions=["hypertension"],
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertIn(res["esi_level"], [1, 2])
        self.assertTrue(res["is_emergency"])
        self.assertTrue(res["hitl_required"])
        self.assertTrue(res["awaiting_human_review"])
        self.assertIsNotNone(res["pending_state"])


if __name__ == "__main__":
    unittest.main()
