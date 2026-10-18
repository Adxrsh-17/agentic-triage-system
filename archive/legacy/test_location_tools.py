"""Tests for location discovery, geocoding, and directions routing tools."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.location_tools import (
    build_directions_url,
    calculate_transit_eta,
    haversine_distance,
    is_relevant_healthcare_facility,
    rank_hospitals_for_triage,
)


class TestLocationTools(unittest.TestCase):
    def test_haversine_distance(self):
        """Verify great-circle distance calculation."""
        dist = haversine_distance(11.0168, 76.9558, 11.1085, 77.3411)
        self.assertTrue(40.0 <= dist <= 46.0)

    def test_facility_relevance_filter(self):
        """Verify irrelevant facilities like dental, bone setting, or eye clinics are filtered out for acute triage."""
        self.assertTrue(is_relevant_healthcare_facility("City General Hospital", "hospital"))
        self.assertTrue(is_relevant_healthcare_facility("Apollo Multi-Speciality Clinic", "hospital"))
        self.assertFalse(is_relevant_healthcare_facility("Dr. Smile Dental Clinic", "hospital"))
        self.assertFalse(is_relevant_healthcare_facility("Traditional Bone Setting Center", "hospital"))
        self.assertTrue(is_relevant_healthcare_facility("MedPlus Pharmacy", "pharmacy"))
        self.assertFalse(is_relevant_healthcare_facility("Optical Eyewear Store", "pharmacy"))

    def test_build_directions_url_live_gps(self):
        """Verify Google Maps URL targets destination without static origin to allow live device GPS departure."""
        url = build_directions_url(11.0, 77.0, 11.000423, 76.971502, "K.G. Hospital")
        self.assertIn("https://www.google.com/maps/dir/?api=1", url)
        self.assertIn("destination=11.000423,76.971502", url)
        self.assertIn("travelmode=driving", url)
        self.assertNotIn("origin=", url)  # Live GPS departure targeting


    def test_calculate_transit_eta(self):
        """Verify ETA calculation for various distances."""
        eta_short = calculate_transit_eta(2.0)
        self.assertGreaterEqual(eta_short, 3)
        self.assertLessEqual(eta_short, 10)

        eta_long = calculate_transit_eta(15.0)
        self.assertGreaterEqual(eta_long, 25)

    def test_rank_hospitals_for_triage_cardiac(self):
        """Verify cardiac presentation boosts hospitals with Cath Lab / cardiac emergency capabilities."""
        facilities = [
            {
                "name": "General Community Clinic",
                "distance_km": 1.0,
                "capabilities": ["General ER", "Urgent Care"],
                "trauma_level": "Community",
                "available_er_bays": 2,
            },
            {
                "name": "Apollo Heart & Trauma Institute",
                "distance_km": 3.0,
                "capabilities": ["24/7 Cath Lab", "Level-1 Trauma", "Cardiac Care"],
                "trauma_level": "Level 1 Comprehensive Trauma",
                "available_er_bays": 5,
            },
        ]
        ranked = rank_hospitals_for_triage(facilities, esi_level=2, symptoms=["chest pain", "shortness of breath"])
        self.assertEqual(len(ranked), 2)
        # Apollo should rank higher despite being slightly further due to Cardiac match
        self.assertEqual(ranked[0]["name"], "Apollo Heart & Trauma Institute")
        self.assertGreater(ranked[0]["match_score"], ranked[1]["match_score"])


if __name__ == "__main__":
    unittest.main()

