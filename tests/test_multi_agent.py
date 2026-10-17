"""End-to-end clinical workflow tests for Nurse Triage Decision-Support Copilot."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.audit_logger import get_audit_logger
from agent.multi_agent import MultiAgentSystem, get_multi_agent


class TestNurseTriageCopilot(unittest.TestCase):
    def setUp(self):
        # Clear audit logger encounters for clean test isolation
        logger = get_audit_logger()
        logger.clear_logs()

    def test_multi_agent_system_initialization(self):
        """Verify that MultiAgentSystem compiles the LangGraph StateGraph successfully."""
        agent = get_multi_agent()
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.graph)

    def test_low_risk_triage_flow_with_sbar_and_fhir(self):
        """Verify low-risk presentation produces structured ESI 4-5, SBAR note, and FHIR R4 Bundle."""
        agent = get_multi_agent()
        res = agent.process(
            user_input="Patient has mild headache and runny nose for 1 day, denies fever or chest pain",
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
        self.assertIn("Consult a qualified healthcare professional", res["content"])

        # Clinical SBAR Note validation
        self.assertIn("sbar_note", res)
        self.assertTrue(bool(res["sbar_note"]))
        self.assertIn("(SBAR REPORT)", res["sbar_note"])
        self.assertIn("[S] SITUATION", res["sbar_note"])
        self.assertIn("ASSESSMENT", res["sbar_note"])
        self.assertIn("RECOMMENDATION", res["sbar_note"])

        # HL7 FHIR R4 Transaction Bundle validation
        self.assertIn("fhir_bundle", res)
        bundle = res["fhir_bundle"]
        self.assertEqual(bundle.get("resourceType"), "Bundle")
        self.assertEqual(bundle.get("type"), "transaction")
        self.assertGreaterEqual(len(bundle.get("entry", [])), 3)

    def test_high_risk_hitl_triage_flow(self):
        """Verify severe emergent presentation triggers HITL review requirement and pauses for nurse sign-off."""
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

    def test_nurse_acceptance_and_audit_logging(self):
        """Verify nurse accepting AI ESI recommendation logs encounter and records concordance."""
        agent = get_multi_agent()
        initial = agent.process(
            user_input="Crushing chest pressure radiating to left arm",
            user_id="test_nurse_accept",
            age=58,
            sex="male",
        )
        self.assertTrue(initial["awaiting_human_review"])

        # Nurse signs off / accepts AI recommendation
        res_approved = agent.process(
            pending_state=initial["pending_state"],
            approval_decision="approved",
            approval_note="Verified with bedside ECG showing acute ST changes",
        )
        self.assertFalse(res_approved["awaiting_human_review"])
        self.assertIn("Nurse Acceptance Recorded", res_approved["content"])

        # Verify audit log metrics
        audit = get_audit_logger()
        metrics = audit.get_metrics()
        self.assertEqual(metrics["total_encounters"], 1)
        self.assertEqual(metrics["accepted_count"], 1)
        self.assertEqual(metrics["override_count"], 0)
        self.assertEqual(metrics["concordance_rate_pct"], 100.0)

    def test_nurse_override_logging_and_reasons(self):
        """Verify nurse override changes ESI level, logs override reason, and updates audit metrics."""
        agent = get_multi_agent()
        initial = agent.process(
            user_input="Patient with moderate wheezing and shortness of breath",
            user_id="test_nurse_override",
            age=35,
            sex="female",
        )
        suggested_esi = initial["esi_level"]

        # Nurse overrides: e.g. changes ESI to 3 with clinical rationale
        res_override = agent.process(
            pending_state=initial["pending_state"],
            approval_decision="override",
            nurse_override_esi=3,
            nurse_override_reason="Patient stabilized post-duoneb, SpO2 normalized to 98%",
            nurse_notes="Transfer to fast-track clinic for monitoring.",
        )
        self.assertFalse(res_override["awaiting_human_review"])
        self.assertIn("Nurse Override Recorded", res_override["content"])

        audit = get_audit_logger()
        metrics = audit.get_metrics()
        self.assertEqual(metrics["total_encounters"], 1)
        self.assertEqual(metrics["override_count"], 1)
        self.assertIn("Patient stabilized post-duoneb, SpO2 normalized to 98%", metrics["override_reasons"])

    def test_vitals_danger_zone_escalation(self):
        """Verify physiological vitals danger zone elevates acuity regardless of chief complaint text."""
        agent = get_multi_agent()

        # Mild complaint with severe hypoxia (SpO2 88%) -> ESI 1 (Resuscitation)
        res_hypoxia = agent.process(
            user_input="Patient has mild cough",
            user_id="test_pt_hypoxia",
            age=45,
            vitals={"spo2": 88, "hr": 110, "rr": 26},
        )
        self.assertEqual(res_hypoxia["esi_level"], 1)
        self.assertTrue(any("SpO2 88%" in trig for trig in res_hypoxia["vital_triggers"]))

        # Mild headache with extreme tachycardia (HR 145 bpm) -> ESI 2 (Emergent)
        res_tachy = agent.process(
            user_input="Patient has mild headache",
            user_id="test_pt_tachy",
            age=30,
            vitals={"hr": 145, "sbp": 120, "dbp": 80},
        )
        self.assertEqual(res_tachy["esi_level"], 2)
        self.assertTrue(any("145 bpm" in trig for trig in res_tachy["vital_triggers"]))


if __name__ == "__main__":
    unittest.main()
