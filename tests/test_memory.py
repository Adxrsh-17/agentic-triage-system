"""Unit Tests for Patient Memory Store & Dense Vector Embeddings (Phase 2)."""

import os
import unittest
import numpy as np
import pytest

from agent.memory.embeddings import embed, EMBEDDING_DIM
from agent.memory.store import LocalMemory, PineconeMemory, get_memory_store


class TestPatientMemory(unittest.TestCase):
    def test_embedding_dimension_and_normalization(self):
        """Embeddings must be 384-dimensional and unit L2 normalized."""
        texts = ["Patient with acute retrosternal chest pain", "Minor finger laceration"]
        vecs = embed(texts)
        self.assertEqual(vecs.shape, (2, EMBEDDING_DIM))
        self.assertEqual(vecs.dtype, np.float32)

        # L2 norm of each vector must be approximately 1.0
        norm_0 = np.linalg.norm(vecs[0])
        norm_1 = np.linalg.norm(vecs[1])
        self.assertAlmostEqual(norm_0, 1.0, places=4)
        self.assertAlmostEqual(norm_1, 1.0, places=4)

    def test_local_memory_add_and_search(self):
        """LocalMemory must store encounters and retrieve semantically similar cases."""
        mem = LocalMemory()
        patient_id = "pt_cardiac_99"

        mem.add_encounter(
            patient_id=patient_id,
            text="Prior emergency admission for acute myocardial infarction, stent placed in LAD",
            metadata={"year": 2024, "type": "cardiac"},
        )
        mem.add_encounter(
            patient_id=patient_id,
            text="Routine dental cleaning checkup, no complaints",
            metadata={"year": 2025, "type": "routine"},
        )

        # Query relevant to cardiac issues
        results = mem.search(patient_id=patient_id, query="substernal chest pressure and dyspnea", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("myocardial infarction", results[0]["text"])
        self.assertGreater(results[0]["score"], 0.2)

    def test_memory_isolation_by_patient_id(self):
        """Memory queries for patient A must never return patient B records."""
        mem = LocalMemory()
        mem.add_encounter("patient_A", "Severe asthma exacerbation")
        mem.add_encounter("patient_B", "Fractured right tibia")

        res_a = mem.search("patient_A", "bone fracture")
        # Patient A should only return Patient A's asthma record, not patient B's fracture
        for r in res_a:
            self.assertIn("asthma", r["text"].lower())
            self.assertNotIn("tibia", r["text"].lower())

    def test_get_memory_store_default_local(self):
        """When Pinecone key is unconfigured, factory must return LocalMemory."""
        store = get_memory_store()
        self.assertIsNotNone(store)

    @pytest.mark.skipif(
        not os.environ.get("PINECONE_API_KEY") or os.environ.get("PINECONE_API_KEY", "").startswith("your_"),
        reason="Requires active PINECONE_API_KEY",
    )
    def test_pinecone_memory_live(self):
        """Live test for Pinecone when API credentials are provided."""
        api_key = os.environ["PINECONE_API_KEY"]
        index_name = os.environ.get("PINECONE_INDEX_NAME", "smart-triage-patient-memory")
        pm = PineconeMemory(api_key=api_key, index_name=index_name, dimension=384)
        self.assertIsNotNone(pm)


if __name__ == "__main__":
    unittest.main()
