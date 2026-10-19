"""Formal Runtime Wiring Verification Tests.

Verifies that the live compiled LangGraph executed by app.py and api.py
actively invokes the canonical modular implementations in:
- agent.nodes.intake.run_intake
- agent.nodes.risk.run_hybrid_risk
- agent.nodes.safety.validate_safety_output
"""

import os
import unittest
from unittest.mock import patch

from agent.multi_agent import get_multi_agent
import agent.nodes.intake
import agent.nodes.risk
import agent.nodes.safety


class TestRuntimeWiring(unittest.TestCase):
    def setUp(self):
        self.agent = get_multi_agent()
        # Default to offline rules mode in test suite for fast offline execution
        self._env_patch = patch.dict(os.environ, {"TRIAGE_MODE": "rules"})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_live_graph_invokes_run_intake(self):
        """Assert compiled graph execution calls agent.nodes.intake.run_intake."""
        original_fn = agent.nodes.intake.run_intake
        calls = []

        def spy_intake(*args, **kwargs):
            calls.append((args, kwargs))
            return original_fn(*args, **kwargs)

        with patch("agent.multi_agent.run_intake", side_effect=spy_intake):
            res = self.agent.process(
                user_input="Mild sore throat and clear rhinorrhea for 1 day",
                user_id="test_wire_intake",
                fast_path=False,
            )

        self.assertGreaterEqual(len(calls), 1, "Graph must invoke run_intake during intake node execution")
        self.assertIn("sore throat", res.get("symptoms", []))

    def test_live_graph_invokes_run_hybrid_risk(self):
        """Assert compiled graph execution calls agent.nodes.risk.run_hybrid_risk."""
        original_fn = agent.nodes.risk.run_hybrid_risk
        calls = []

        def spy_risk(*args, **kwargs):
            calls.append((args, kwargs))
            return original_fn(*args, **kwargs)

        with patch("agent.multi_agent.run_hybrid_risk", side_effect=spy_risk):
            res = self.agent.process(
                user_input="Mild twisted ankle without deformity",
                user_id="test_wire_risk",
                fast_path=False,
            )

        self.assertGreaterEqual(len(calls), 1, "Graph must invoke run_hybrid_risk during risk node execution")
        self.assertIn("esi_level", res)

    def test_live_graph_invokes_validate_safety_output(self):
        """Assert compiled graph execution calls agent.nodes.safety.validate_safety_output."""
        original_fn = agent.nodes.safety.validate_safety_output
        calls = []

        def spy_safety(*args, **kwargs):
            calls.append(args[0] if args else "")
            return original_fn(*args, **kwargs)

        with patch("agent.multi_agent.validate_safety_output", side_effect=spy_safety):
            res = self.agent.process(
                user_input="Low back pain after lifting boxes for 2 days",
                user_id="test_wire_safety",
                fast_path=False,
            )

        self.assertGreaterEqual(len(calls), 1, "Graph must invoke validate_safety_output during safety node execution")
        self.assertTrue(res.get("safety_validation", {}).get("is_safe"))

    def test_triage_mode_env_switch_respected(self):
        """Assert TRIAGE_MODE=rules forces deterministic rules-only execution."""
        with patch.dict(os.environ, {"TRIAGE_MODE": "rules"}):
            res = self.agent.process(
                user_input="Low back muscle strain",
                user_id="test_wire_mode",
                fast_path=False,
            )
            # In rules mode, intake source is regex
            self.assertEqual(res.get("intake_method"), "regex")


if __name__ == "__main__":
    unittest.main()
