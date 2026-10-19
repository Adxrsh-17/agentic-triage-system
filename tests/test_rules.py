"""Unit Tests for Deterministic Rules Engine & NegEx Scope Resolution (Phase 1)."""

import unittest
from agent.rules import (
    check_vital_danger_zones,
    detect_red_flags,
    is_negated,
    rule_risk,
)
from agent.schemas import RiskLevel


class TestDeterministicRules(unittest.TestCase):
    def test_red_flag_detection(self):
        """Affirmative red flags must be detected accurately."""
        text = "Patient experiencing severe crushing chest pain radiating to left arm."
        flags = detect_red_flags(text)
        self.assertIn("crushing chest pain", flags)
        self.assertIn("radiating to left arm", flags)

    def test_negation_handling_simple(self):
        """Simple denial must not trigger red flag detection."""
        text = "Patient has mild rash, denies chest pain and denies shortness of breath."
        self.assertTrue(is_negated("chest pain", text))
        flags = detect_red_flags(text)
        self.assertNotIn("chest pain", flags)

    def test_negation_coordinated_conjunctions(self):
        """Conjunctions like 'no X or Y' must negate both concepts."""
        text = "Patient has runny nose, no fever or chest pain."
        self.assertTrue(is_negated("fever", text))
        self.assertTrue(is_negated("chest pain", text))
        flags = detect_red_flags(text)
        self.assertNotIn("chest pain", flags)

    def test_negation_adversative_boundary(self):
        """Adversative conjunctions like 'but' must terminate negation scope."""
        text = "Patient denies fever but has crushing chest pain."
        self.assertTrue(is_negated("fever", text))
        self.assertFalse(is_negated("crushing chest pain", text))
        flags = detect_red_flags(text)
        self.assertIn("crushing chest pain", flags)

    def test_vital_danger_zones_hypoxia(self):
        """SpO2 < 90% must trigger ESI 1 resuscitation level."""
        is_danger, triggers, min_esi = check_vital_danger_zones({"spo2": 88})
        self.assertTrue(is_danger)
        self.assertEqual(min_esi, 1)
        self.assertTrue(any("SpO2" in t for t in triggers))

    def test_pediatric_age_stratification(self):
        """Infant HR 160 is physiologically normal, while Adult HR 160 is a danger zone."""
        is_infant_danger, _, infant_esi = check_vital_danger_zones({"hr": 160}, age=0)
        self.assertFalse(is_infant_danger)
        self.assertEqual(infant_esi, 5)

        is_adult_danger, _, adult_esi = check_vital_danger_zones({"hr": 160}, age=45)
        self.assertTrue(is_adult_danger)
        self.assertEqual(adult_esi, 2)

    def test_rule_risk_safety_floor(self):
        """rule_risk must classify emergency red flags as RiskLevel.EMERGENCY."""
        res = rule_risk("Patient reports worst headache of life and thunderclap onset")
        self.assertEqual(res.level, RiskLevel.EMERGENCY)
        self.assertEqual(res.esi_level, 2)
        self.assertTrue(res.hitl_required)


if __name__ == "__main__":
    unittest.main()
