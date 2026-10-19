"""Unit Tests for Hybrid Risk Engine & Inviolable Safety Floor Invariant (Phase 1)."""

import unittest
from unittest.mock import patch

from agent.nodes.risk import run_hybrid_risk
from agent.schemas import IntakeResult, RiskLevel, RiskResult


class TestRiskHybrid(unittest.TestCase):
    def test_safety_invariant_llm_routine_rules_emergency(self):
        """CRITICAL SAFETY INVARIANT: When LLM suggests 'routine' but rules identify 'emergency',
        the final consensus MUST escalate to 'emergency' and never de-escalate."""
        intake = IntakeResult(
            symptoms=["chest pain", "sweating"],
            age=55,
            sex="male",
        )
        chief_complaint = "Patient with severe crushing chest pain radiating to left arm"

        # Mock LLM returning an unsafe 'routine' assessment
        mock_unsafe_llm = RiskResult(
            level=RiskLevel.ROUTINE,
            esi_level=5,
            score=0.1,
            red_flags=[],
            rationale="LLM hallucinated that this is mild indigestion",
            source="llm",
        )

        with patch("agent.nodes.risk._llm_assess_risk", return_value=mock_unsafe_llm):
            result = run_hybrid_risk(chief_complaint=chief_complaint, intake=intake)

            # Invariant: Safety floor enforces EMERGENCY
            self.assertEqual(result.level, RiskLevel.EMERGENCY)
            self.assertIn(result.esi_level, [1, 2])
            self.assertTrue(result.hitl_required)
            self.assertTrue(result.disagreement)
            self.assertEqual(result.rule_level, RiskLevel.EMERGENCY)
            self.assertEqual(result.llm_level, RiskLevel.ROUTINE)
            self.assertIn("Safety Floor Override", result.rationale)

    def test_llm_escalation_above_rules(self):
        """When LLM detects contextual emergency not captured by basic keywords,
        the system safely escalates to emergency."""
        intake = IntakeResult(
            symptoms=["headache"],
            age=72,
            conditions=["Warfarin anticoagulation"],
        )
        chief_complaint = "Elderly patient on blood thinners with minor head bump, now headache"

        # Mock LLM recognizing high intracranial hemorrhage risk
        mock_emergent_llm = RiskResult(
            level=RiskLevel.EMERGENCY,
            esi_level=2,
            score=0.9,
            red_flags=["Anticoagulated acute head strike"],
            rationale="High ICH risk requiring immediate non-contrast head CT",
            source="llm",
        )

        with patch("agent.nodes.risk._llm_assess_risk", return_value=mock_emergent_llm):
            result = run_hybrid_risk(chief_complaint=chief_complaint, intake=intake)
            self.assertEqual(result.level, RiskLevel.EMERGENCY)
            self.assertEqual(result.source, "hybrid")

    def test_offline_fallback_when_llm_none(self):
        """When LLM returns None (offline / no API key), engine operates purely on rules."""
        intake = IntakeResult(
            symptoms=["sore throat"],
            age=25,
        )
        chief_complaint = "Mild sore throat for 1 day, no fever"

        with patch("agent.nodes.risk._llm_assess_risk", return_value=None):
            result = run_hybrid_risk(chief_complaint=chief_complaint, intake=intake)
            self.assertEqual(result.source, "rules")
            self.assertEqual(result.level, RiskLevel.ROUTINE)
            self.assertFalse(result.disagreement)


if __name__ == "__main__":
    unittest.main()
