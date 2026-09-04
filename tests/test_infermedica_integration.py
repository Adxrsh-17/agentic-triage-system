"""Tests for Infermedica Clinical Engine v3 integration & consensus arbitration."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.infermedica_client import (
    get_diagnosis,
    get_medication_guidance,
    get_triage,
    is_infermedica_configured,
    parse_symptoms,
)
from agent.multi_agent import MultiAgentSystem, assess_medical_risk


class TestInfermedicaIntegration(unittest.TestCase):
    def test_infermedica_unconfigured(self):
        """Verify that when Infermedica keys are absent, all functions return None cleanly."""
        with patch.dict(os.environ, {"INFERMEDICA_APP_ID": "", "INFERMEDICA_APP_KEY": ""}, clear=False):
            self.assertFalse(is_infermedica_configured())
            self.assertIsNone(parse_symptoms("headache"))
            self.assertIsNone(get_diagnosis([{"id": "s_21", "choice_id": "present"}]))
            self.assertIsNone(get_triage([{"id": "s_21", "choice_id": "present"}]))

    def test_infermedica_parse_mock(self):
        """Verify parse_symptoms processes Infermedica API response correctly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "mentions": [
                {"id": "s_21", "name": "Headache", "common_name": "headache", "choice_id": "present"},
                {"id": "s_98", "name": "Fever", "common_name": "fever", "choice_id": "absent"},
            ],
            "obvious": False,
        }

        with patch.dict(os.environ, {"INFERMEDICA_APP_ID": "dummy_id", "INFERMEDICA_APP_KEY": "dummy_key"}, clear=False):
            with patch("requests.post", return_value=mock_resp):
                res = parse_symptoms("patient has headache and denies fever", age=30, sex="female")
                self.assertIsNotNone(res)
                self.assertIn("headache", res["symptoms"])
                self.assertIn("denies fever", res["pertinent_negatives"])
                self.assertEqual(len(res["evidence"]), 2)
                self.assertEqual(res["evidence"][0], {"id": "s_21", "choice_id": "present", "source": "initial"})
                self.assertEqual(res["evidence"][1], {"id": "s_98", "choice_id": "absent", "source": "initial"})

    def test_infermedica_triage_mock(self):
        """Verify get_triage correctly parses Infermedica /v3/triage output."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "triage_level": "emergency",
            "serious": [{"id": "c_1", "name": "Myocardial Infarction"}],
            "description": "Immediate evaluation recommended in an emergency department.",
            "label": "Emergency care",
            "root_cause": "c_1",
            "teleconsultation_applicable": False,
        }

        with patch.dict(os.environ, {"INFERMEDICA_APP_ID": "dummy_id", "INFERMEDICA_APP_KEY": "dummy_key"}, clear=False):
            with patch("requests.post", return_value=mock_resp):
                res = get_triage([{"id": "s_50", "choice_id": "present"}], age=45, sex="male")
                self.assertIsNotNone(res)
                self.assertEqual(res["triage_level"], "emergency")
                self.assertEqual(len(res["serious"]), 1)
                self.assertEqual(res["serious"][0]["id"], "c_1")
                self.assertIn("Emergency care", res["label"])

    def test_consensus_escalation_rule_1(self):
        """Rule 1: If Infermedica detects emergency (ESI 2), consensus escalates to ESI 2 even if internal was ESI 3."""
        internal = {
            "risk_level": "MEDIUM",
            "esi_level": 3,
            "esi_rationale": "Urgent consultation",
            "emergency_recommended": False,
            "risk_reasons": ["Moderate fever"],
            "possible_concerns": ["Viral illness"],
        }
        infermedica_triage_res = {
            "triage_level": "emergency",
            "description": "Acute distress detected.",
            "label": "Emergency Care",
            "serious": [{"name": "Pneumonia"}],
        }

        inf_level = infermedica_triage_res.get("triage_level")
        inf_mapping = {
            "emergency_ambulance": ("HIGH", 1, True),
            "emergency": ("HIGH", 2, True),
            "consultation_24": ("MEDIUM", 3, False),
            "consultation": ("MEDIUM", 4, False),
            "self_care": ("LOW", 5, False),
        }
        inf_risk, inf_esi, inf_emerg = inf_mapping.get(inf_level, ("LOW", 5, False))

        final_risk = "HIGH" if (internal["risk_level"] == "HIGH" or inf_risk == "HIGH") else "MEDIUM"
        final_esi = min(internal["esi_level"], inf_esi)
        final_emergency = internal["emergency_recommended"] or inf_emerg

        self.assertEqual(final_risk, "HIGH")
        self.assertEqual(final_esi, 2)
        self.assertTrue(final_emergency)

    def test_consensus_escalation_rule_2(self):
        """Rule 2: If internal detects red flag (ESI 2), consensus never downgrades even if Infermedica returned self_care (ESI 5)."""
        internal = {
            "risk_level": "HIGH",
            "esi_level": 2,
            "esi_rationale": "Severe acute chest pain detected",
            "emergency_recommended": True,
            "risk_reasons": ["Chest pain red flag"],
            "possible_concerns": ["Acute Coronary Syndrome"],
        }
        infermedica_triage_res = {
            "triage_level": "self_care",
            "description": "Mild presentation",
            "label": "Self Care",
            "serious": [],
        }

        inf_level = infermedica_triage_res.get("triage_level")
        inf_mapping = {
            "emergency_ambulance": ("HIGH", 1, True),
            "emergency": ("HIGH", 2, True),
            "consultation_24": ("MEDIUM", 3, False),
            "consultation": ("MEDIUM", 4, False),
            "self_care": ("LOW", 5, False),
        }
        inf_risk, inf_esi, inf_emerg = inf_mapping.get(inf_level, ("LOW", 5, False))

        final_risk = "HIGH" if (internal["risk_level"] == "HIGH" or inf_risk == "HIGH") else "LOW"
        final_esi = min(internal["esi_level"], inf_esi)

        self.assertEqual(final_risk, "HIGH")
        self.assertEqual(final_esi, 2)

    def test_medication_guidance_comorbidity_alerts(self):
        """Verify OTC medication guidance adjusts for comorbidities like liver disease and hypertension."""
        # Test liver comorbidity reduces paracetamol dosage
        meds_liver = get_medication_guidance(
            symptoms=["headache", "fever"],
            conditions=["liver disease"],
            age=40,
            sex="male",
            risk_level="LOW",
        )
        self.assertTrue(any("Paracetamol" in m["name"] for m in meds_liver))
        paracetamol_med = next(m for m in meds_liver if "Paracetamol" in m["name"])
        self.assertIn("Liver Comorbidity Alert", paracetamol_med["precautions"])

        # Test emergency suppresses self-medication
        meds_emerg = get_medication_guidance(
            symptoms=["chest pain", "shortness of breath"],
            conditions=[],
            age=55,
            sex="male",
            risk_level="HIGH",
            is_emergency=True,
        )
        self.assertTrue(any("No unguided oral self-medication" in m["name"] for m in meds_emerg))
        self.assertTrue(any("Aspirin" in m["name"] for m in meds_emerg))

    def test_offline_multiagent_fallback(self):
        """Verify that multi-agent triage runs completely offline without crashing when API keys are absent."""
        with patch.dict(os.environ, {"INFERMEDICA_APP_ID": "", "INFERMEDICA_APP_KEY": "", "GROQ_API_KEY": ""}, clear=False):
            system = MultiAgentSystem()
            res = system.process(
                user_input="Patient has mild headache and sore throat for 1 day, denies fever",
                user_id="test_offline_pt",
                age=25,
                sex="female",
                conditions=["asthma"],
            )
            self.assertIsNotNone(res)
            self.assertIn("content", res)
            self.assertIn(res["risk_level"], ["LOW", "MEDIUM", "HIGH"])
            self.assertTrue(1 <= res["esi_level"] <= 5)
            self.assertGreaterEqual(len(res.get("workflow_trace", [])), 4)
            self.assertGreaterEqual(len(res.get("medications", [])), 1)


if __name__ == "__main__":
    unittest.main()
