"""Unit Tests for Calibrated Urgent-Tier Rules and Vital Sign Escalations."""

import unittest
from agent.rules import check_vital_danger_zones, is_negated, rule_risk
from agent.schemas import RiskLevel


class TestUrgentRules(unittest.TestCase):
    def test_urgent_rules_fire_on_subacute_infection(self):
        """Persistent fever >3 days should trigger urgent tier."""
        res = rule_risk("Patient with persistent fever for 3 days and yellow sputum")
        self.assertEqual(res.level, RiskLevel.URGENT)
        self.assertIn("persistent fever", res.rationale)

    def test_urgent_rules_fire_on_musculoskeletal_trauma(self):
        """Inability to bear weight and bony tenderness should trigger urgent tier."""
        res = rule_risk("Twisted ankle playing football, unable to bear weight, severe swelling")
        self.assertEqual(res.level, RiskLevel.URGENT)
        self.assertIn("unable to bear weight", res.rationale)

    def test_urgent_rules_fire_on_dvt_presentation(self):
        """Unilateral calf warmth and swelling should trigger urgent tier."""
        res = rule_risk("Sudden onset swelling and warmth in left calf for 3 days")
        self.assertEqual(res.level, RiskLevel.URGENT)
        self.assertIn("calf", res.rationale)

    def test_negation_prevents_urgent_escalation(self):
        """Negated clinical complaints should NOT trigger urgent escalation."""
        self.assertTrue(is_negated("fever", "Patient presents with clear runny nose, denies fever or vomiting"))
        self.assertTrue(is_negated("shortness of breath", "No shortness of breath reported"))
        self.assertTrue(is_negated("fever", "Fever resolved 2 days ago, currently asymptomatic"))

        res = rule_risk("Patient presents with runny nose, denies fever, no vomiting")
        self.assertEqual(res.level, RiskLevel.ROUTINE)

    def test_routine_context_suppression_suture_removal(self):
        """Laceration mentions in suture removal / follow-up context should remain routine."""
        res = rule_risk("Routine suture removal on forearm; laceration repaired 10 days ago, wound clean")
        self.assertEqual(res.level, RiskLevel.ROUTINE)

    def test_clean_vital_formatting_no_trailing_zero_floats(self):
        """Vital sign triggers must not print raw float decimals like 145.0 bpm."""
        danger, triggers, min_esi = check_vital_danger_zones({"hr": 145.0, "spo2": 91.0})
        self.assertTrue(danger)
        self.assertIn("HR 145 bpm > 140", " ".join(triggers))
        self.assertIn("SpO2 91%", " ".join(triggers))
        self.assertNotIn("145.0", " ".join(triggers))

    def test_hypoxic_tachypnea_escalates_to_emergency(self):
        """Mild hypoxia (SpO2 93%) combined with adult tachypnea (RR 24) must escalate to emergency."""
        danger, triggers, min_esi = check_vital_danger_zones({"spo2": 93, "rr": 24})
        self.assertTrue(danger)
        self.assertEqual(min_esi, 2)
        self.assertIn("Hypoxic Tachypnea", triggers[0])

    def test_oncology_febrile_presentation_escalates_to_emergency(self):
        """Chemotherapy patient presenting with fever must escalate to emergency."""
        res = rule_risk(
            "73yo male undergoing chemotherapy for lymphoma, fever 38.9C, chills, rigors",
            vitals={"temp": 38.9, "hr": 112, "rr": 24, "spo2": 94},
        )
        self.assertEqual(res.level, RiskLevel.EMERGENCY)
        self.assertIn("neutropenic fever", res.red_flags[0])

    def test_anticoagulated_head_strike_escalates_to_emergency(self):
        """Patient on warfarin suffering fall with head strike must escalate to emergency."""
        res = rule_risk(
            "83yo male on warfarin for atrial fibrillation, fell from standing hitting head, mild headache",
            vitals={"spo2": 96, "hr": 74, "sbp": 158},
        )
        self.assertEqual(res.level, RiskLevel.EMERGENCY)
        self.assertIn("head trauma on anticoagulation", res.red_flags[0])


if __name__ == "__main__":
    unittest.main()
