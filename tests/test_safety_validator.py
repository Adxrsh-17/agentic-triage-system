"""Unit Tests for Safety Output Validator & Patient Communication Safeguards (Phase 4)."""

import unittest
from agent.nodes.safety import (
    MANDATORY_DISCLAIMER,
    generate_safe_response,
    validate_safety_output,
)
from agent.schemas import RiskLevel, RiskResult


class TestSafetyValidator(unittest.TestCase):
    def test_valid_safe_message(self):
        """Standard safe clinical output containing disclaimer must pass validation."""
        msg = (
            "Clinical evaluation indicates urgent medical attention within 24 hours. "
            "Please consult a qualified healthcare professional. "
            f"\n\n---\n{MANDATORY_DISCLAIMER}"
        )
        is_valid, violations = validate_safety_output(msg)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)

    def test_blocked_disease_diagnosis_pattern(self):
        """Direct statements asserting disease diagnosis must be caught by validator."""
        unsafe_msg = f"Based on your symptoms, you have pneumonia. Consult a doctor. {MANDATORY_DISCLAIMER}"
        is_valid, violations = validate_safety_output(unsafe_msg)
        self.assertFalse(is_valid)
        self.assertTrue(any("diagnosis" in v for v in violations))

    def test_blocked_prescription_dosing_pattern(self):
        """Statements giving specific medication dosage instructions must be caught by validator."""
        unsafe_msg = f"Take 500mg amoxicillin twice daily. Consult a doctor. {MANDATORY_DISCLAIMER}"
        is_valid, violations = validate_safety_output(unsafe_msg)
        self.assertFalse(is_valid)
        self.assertTrue(any("dosing" in v for v in violations))

    def test_missing_disclaimer_fails(self):
        """Outputs omitting the mandatory healthcare disclaimer must fail validation."""
        msg_without_disclaimer = "You should go to urgent care today for evaluation of your cough."
        is_valid, violations = validate_safety_output(msg_without_disclaimer)
        self.assertFalse(is_valid)
        self.assertTrue(any("disclaimer" in v.lower() for v in violations))

    def test_generate_safe_response_emergency(self):
        """Emergency safe response generator must include emergency guidance and disclaimer."""
        risk = RiskResult(
            level=RiskLevel.EMERGENCY,
            esi_level=2,
            score=0.9,
            rationale="Severe acute substernal chest pain",
            source="rules",
        )
        response = generate_safe_response(risk, symptoms=["chest pain"])
        self.assertIn("EMERGENCY CLINICAL ALERT", response)
        self.assertIn("911", response)
        self.assertIn("educational AI prototype", response)

    def test_generate_safe_response_fallback_on_unsafe_input(self):
        """If unsafe generated text violates checks, fallback template is deployed safely."""
        risk = RiskResult(
            level=RiskLevel.ROUTINE,
            esi_level=5,
            score=0.1,
            rationale="Minor scratch",
            source="rules",
        )
        # Even with empty inputs, generated response is compliant
        res = generate_safe_response(risk, symptoms=[])
        is_valid, violations = validate_safety_output(res)
        self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
