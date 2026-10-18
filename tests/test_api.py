"""Comprehensive Integration Tests for the FastAPI Triage Microservice (api.py).

Verifies:
1. GET /health: Operational readiness and subsystem health reporting.
2. POST /v1/triage: Routine low-risk triage flow with FHIR Bundle and SBAR note.
3. POST /v1/triage: ESI 1 resuscitation fast-path sub-50ms latency verification.
4. POST /v1/triage: Emergent presentation triggering Human-in-the-Loop review.
5. POST /v1/triage/review: Nurse approval flow finalizing suspended state.
6. POST /v1/triage/review: Nurse clinical override with acuity escalation.
7. POST /v1/triage/review: Validation errors on missing override rationale.
8. GET /v1/audit/logs: Paginated audit log retrieval.
9. GET /v1/audit/analytics: Real-time concordance analytics.
"""

import time
import unittest
from fastapi.testclient import TestClient

from api import app
from agent.audit_logger import get_audit_logger


class TestFastAPITriageService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        audit = get_audit_logger()
        audit.clear_logs()

    # ── Test 1: Health Check ─────────────────────────────────────────────────

    def test_health_check_endpoint(self):
        """GET /health must return 200 OK with subsystem connectivity status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("services", data)
        self.assertIn("groq_llm", data["services"])
        self.assertIn("pinecone_memory", data["services"])
        self.assertIn("infermedica", data["services"])
        self.assertIn("audit_logger", data["services"])

    # ── Test 2: Low-Risk Triage Flow ─────────────────────────────────────────

    def test_low_risk_triage_flow(self):
        """POST /v1/triage with routine minor symptoms returns ESI 4/5 without HITL block."""
        payload = {
            "chief_complaint": "Superficial paper cut on right index finger, minor bleeding stopped",
            "user_id": "api_low_01",
            "age": 25,
            "sex": "female",
            "vitals": {"hr": 72, "sbp": 120, "dbp": 78, "spo2": 99, "rr": 14},
        }
        response = self.client.post("/v1/triage", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["user_id"], "api_low_01")
        self.assertIn(data["esi_level"], [4, 5])
        self.assertEqual(data["risk_level"], "LOW")
        self.assertFalse(data["is_emergency"])
        self.assertFalse(data["hitl_required"])
        self.assertFalse(data["awaiting_human_review"])
        self.assertIsNotNone(data["sbar_note"])
        self.assertIn("(SBAR REPORT)", data["sbar_note"])
        self.assertIsNotNone(data["fhir_bundle"])
        self.assertEqual(data["fhir_bundle"].get("resourceType"), "Bundle")

    # ── Test 3: Sub-50ms ESI 1 Resuscitation Fast-Path ───────────────────────

    def test_esi_1_fast_path_latency_sub_50ms(self):
        """POST /v1/triage for ESI 1 resuscitation must complete end-to-end in < 50ms via HTTP."""
        payload = {
            "chief_complaint": "Patient found unresponsive, cyanotic, agonal breathing",
            "user_id": "api_fastpath_01",
            "age": 55,
            "sex": "male",
            "vitals": {"hr": 30, "sbp": 60, "dbp": 35, "spo2": 82, "rr": 6},
        }

        # Warm-up call
        self.client.post("/v1/triage", json=payload)

        # Benchmark run
        t0 = time.perf_counter()
        response = self.client.post("/v1/triage", json=payload)
        http_latency_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["esi_level"], 1)
        self.assertEqual(data["risk_level"], "HIGH")
        self.assertTrue(data["is_emergency"])
        self.assertGreater(len(data["vital_triggers"]), 0)

        # Fast-path performance guarantee: HTTP round-trip < 50 ms
        self.assertLess(
            http_latency_ms,
            50.0,
            f"Expected ESI 1 fast-path HTTP latency < 50ms, got {http_latency_ms:.2f}ms",
        )

    # ── Test 4: Emergent Triage Triggering HITL Review ───────────────────────

    def test_emergent_triage_hitl_suspension(self):
        """POST /v1/triage with high-risk emergent symptoms must suspend and require HITL review."""
        payload = {
            "chief_complaint": "Severe crushing retrosternal chest pain radiating to jaw, diaphoretic",
            "user_id": "api_emergent_01",
            "age": 62,
            "sex": "male",
            "vitals": {"hr": 110, "sbp": 165, "dbp": 95, "spo2": 95, "rr": 22},
        }
        response = self.client.post("/v1/triage", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["esi_level"], 2)
        self.assertEqual(data["risk_level"], "HIGH")
        self.assertTrue(data["is_emergency"])
        self.assertTrue(data["hitl_required"])
        self.assertTrue(data["awaiting_human_review"])
        self.assertIsNotNone(data["pending_state"])

    # ── Test 5: HITL Nurse Approval Flow ─────────────────────────────────────

    def test_hitl_nurse_approval_flow(self):
        """POST /v1/triage/review with 'approved' resumes state and produces finalized outputs."""
        initial_payload = {
            "chief_complaint": "Severe crushing retrosternal chest pain radiating to left arm",
            "user_id": "api_review_appr_01",
            "age": 58,
            "sex": "male",
            "vitals": {"hr": 105, "sbp": 155, "dbp": 92, "spo2": 96, "rr": 20},
        }
        initial_res = self.client.post("/v1/triage", json=initial_payload)
        initial_data = initial_res.json()
        self.assertTrue(initial_data["awaiting_human_review"])
        pending_state = initial_data["pending_state"]

        review_payload = {
            "user_id": "api_review_appr_01",
            "pending_state": pending_state,
            "approval_decision": "approved",
            "approval_note": "Nurse bedside assessment confirmed: immediate cardiology evaluation indicated.",
        }
        review_res = self.client.post("/v1/triage/review", json=review_payload)
        self.assertEqual(review_res.status_code, 200)
        review_data = review_res.json()

        self.assertEqual(review_data["esi_level"], 2)
        self.assertFalse(review_data["awaiting_human_review"])
        self.assertIsNotNone(review_data["sbar_note"])
        self.assertIn("(SBAR REPORT)", review_data["sbar_note"])

    # ── Test 6: HITL Nurse Clinical Override Flow ────────────────────────────

    def test_hitl_nurse_override_flow(self):
        """POST /v1/triage/review with 'overridden' correctly registers nurse acuity escalation."""
        initial_payload = {
            "chief_complaint": "Severe crushing retrosternal chest pain radiating to left jaw, diaphoresis",
            "user_id": "api_review_ovr_01",
            "age": 62,
            "sex": "male",
            "vitals": {"hr": 110, "sbp": 165, "dbp": 95, "spo2": 95, "rr": 22},
        }
        initial_res = self.client.post("/v1/triage", json=initial_payload)
        self.assertEqual(initial_res.status_code, 200)
        self.assertTrue(initial_res.json()["awaiting_human_review"])
        pending_state = initial_res.json().get("pending_state") or {}

        override_payload = {
            "user_id": "api_review_ovr_01",
            "pending_state": pending_state,
            "approval_decision": "overridden",
            "nurse_override_esi": 1,
            "nurse_override_reason": "Patient became acutely unresponsive with sudden peritoneal rigidity and shock",
            "approval_note": "Stat surgical consult requested.",
        }
        review_res = self.client.post("/v1/triage/review", json=override_payload)
        self.assertEqual(review_res.status_code, 200)
        review_data = review_res.json()

        self.assertEqual(review_data["esi_level"], 1)

    # ── Test 7: Review Validation Failure ────────────────────────────────────

    def test_hitl_review_validation_error_on_missing_override_esi(self):
        """POST /v1/triage/review must return 422 if decision is 'overridden' but nurse_override_esi is missing."""
        review_payload = {
            "user_id": "invalid_req",
            "pending_state": {},
            "approval_decision": "overridden",
            "approval_note": "Overriding without providing new ESI level",
        }
        response = self.client.post("/v1/triage/review", json=review_payload)
        self.assertEqual(response.status_code, 422)

    # ── Test 8: Paginated Audit Logs & Analytics ─────────────────────────────

    def test_audit_logs_and_analytics_endpoints(self):
        """GET /v1/audit/logs and GET /v1/audit/analytics return structured concordance data."""
        # Execute an encounter to populate logs
        self.client.post(
            "/v1/triage",
            json={
                "chief_complaint": "Patient with sore throat for 2 days",
                "user_id": "audit_test_pt",
                "age": 28,
                "sex": "male",
                "vitals": {"hr": 70, "sbp": 118, "dbp": 76, "spo2": 99, "rr": 14},
            },
        )

        logs_res = self.client.get("/v1/audit/logs?limit=10&offset=0")
        self.assertEqual(logs_res.status_code, 200)
        logs_data = logs_res.json()
        self.assertIn("total", logs_data)
        self.assertIn("encounters", logs_data)
        self.assertGreaterEqual(logs_data["total"], 1)

        analytics_res = self.client.get("/v1/audit/analytics")
        self.assertEqual(analytics_res.status_code, 200)
        analytics_data = analytics_res.json()
        self.assertIn("concordance_rate_pct", analytics_data)
        self.assertIn("total_encounters", analytics_data)


if __name__ == "__main__":
    unittest.main()
