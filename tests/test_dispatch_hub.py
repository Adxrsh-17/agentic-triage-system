"""Unit tests for Central Dispatch Hub & Hospital Resource Management System."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.dispatch_hub import DispatchHub, HospitalProfile, get_dispatch_hub


class TestDispatchHub(unittest.TestCase):
    def setUp(self):
        self.hub = get_dispatch_hub()

    def test_hub_initialization_and_hospitals(self):
        """Verify DispatchHub is initialized with reference emergency hospitals."""
        hospitals = self.hub.list_hospitals()
        self.assertGreaterEqual(len(hospitals), 3)
        apollo = self.hub.get_hospital("hosp-apollo-emergency")
        self.assertIsNotNone(apollo)
        self.assertEqual(apollo.trauma_level, "Level 1 Comprehensive Trauma")
        self.assertGreater(apollo.available_er_bays, 0)

    def test_dispatch_creation_cardiac_esi2(self):
        """Verify 1-click dispatch ticket creation with automated resource pre-allocation for cardiac presentation."""
        ticket = self.hub.create_dispatch(
            patient_id="pt-test-cardiac-1",
            patient_age=55,
            patient_sex="male",
            patient_conditions=["Hypertension"],
            chief_complaint="Severe crushing chest pain and sweating for 1 hour",
            symptoms=["chest pain", "sweating", "shortness of breath"],
            risk_level="HIGH",
            esi_level=2,
            esi_rationale="High-risk acute coronary syndrome",
            is_emergency=True,
            hospital_id="hosp-apollo-emergency",
            distance_km=3.2,
        )

        self.assertIsNotNone(ticket)
        self.assertTrue(ticket.ticket_id.startswith("DISP-"))
        self.assertEqual(ticket.status, "ACCEPTED_RESERVED")
        self.assertEqual(ticket.esi_level, 2)
        self.assertTrue(ticket.is_emergency)

        # Check resource allocation
        alloc = ticket.resource_allocation
        self.assertIsNotNone(alloc)
        self.assertIn("Resuscitation", alloc.bay_type)
        self.assertIn("Cardiac", alloc.allocated_bay)
        self.assertEqual(alloc.priority_level, "IMMEDIATE")
        self.assertTrue(any("ECG" in eq or "Defibrillator" in eq for eq in alloc.staged_equipment))

    def test_dispatch_creation_mild_esi5(self):
        """Verify fast-track resource allocation for mild outpatient ESI 5 presentation."""
        ticket = self.hub.create_dispatch(
            patient_id="pt-test-mild-2",
            patient_age=24,
            patient_sex="female",
            patient_conditions=[],
            chief_complaint="Mild runny nose and sore throat for 1 day",
            symptoms=["runny nose", "sore throat"],
            risk_level="LOW",
            esi_level=5,
            esi_rationale="Non-urgent presentation",
            is_emergency=False,
            hospital_id="hosp-life-care-urgent",
            distance_km=1.5,
        )

        self.assertEqual(ticket.esi_level, 5)
        self.assertEqual(ticket.resource_allocation.bay_type, "Fast Track")
        self.assertEqual(ticket.resource_allocation.priority_level, "STANDARD")

    def test_dispatch_lifecycle_transitions(self):
        """Verify status progression: REQUESTED -> ACCEPTED_RESERVED -> TEAM_MOBILIZED -> EN_ROUTE -> ARRIVED_ADMITTED."""
        ticket = self.hub.create_dispatch(
            patient_id="pt-test-lifecycle",
            patient_age=40,
            patient_sex="male",
            patient_conditions=[],
            chief_complaint="Severe laceration bleeding",
            symptoms=["bleeding", "wound"],
            risk_level="HIGH",
            esi_level=2,
            esi_rationale="Acute trauma laceration",
            is_emergency=True,
            hospital_id="hosp-ganga-trauma",
            distance_km=4.0,
        )

        # 1. Mobilize team
        updated = self.hub.update_dispatch_status(ticket.ticket_id, "TEAM_MOBILIZED", "Trauma team staged")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "TEAM_MOBILIZED")
        self.assertEqual(updated.resource_allocation.readiness_status, "MOBILIZED")

        # 2. Patient en route
        updated = self.hub.update_dispatch_status(ticket.ticket_id, "EN_ROUTE", "Patient driving")
        self.assertEqual(updated.status, "EN_ROUTE")

        # 3. Patient arrived & admitted
        updated = self.hub.update_dispatch_status(ticket.ticket_id, "ARRIVED_ADMITTED", "Handover complete")
        self.assertEqual(updated.status, "ARRIVED_ADMITTED")
        self.assertEqual(updated.resource_allocation.readiness_status, "OCCUPIED")

    def test_ingest_discovered_facilities(self):
        """Verify dynamic discovery ingestion enriches raw map places with hospital capacity telemetry."""
        raw_places = [
            {"name": "Sunrise Multi-Speciality Emergency Hospital", "lat": 11.020, "lng": 76.960, "distance_km": 2.5, "address": "Sunrise Rd"},
        ]
        enriched = self.hub.ingest_discovered_facilities(raw_places, 11.016, 76.955)
        self.assertEqual(len(enriched), 1)
        item = enriched[0]
        self.assertIn("hospital_id", item)
        self.assertGreater(item["total_er_bays"], 0)
        self.assertGreaterEqual(item["available_er_bays"], 0)
        self.assertGreater(item["estimated_transit_mins"], 0)


if __name__ == "__main__":
    unittest.main()
