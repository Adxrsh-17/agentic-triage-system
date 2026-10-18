"""Unit tests for Pinecone semantic memory evaluation functions."""

import unittest
from scripts.evaluate_pinecone_memory import (
    CLINICAL_PROFILES,
    cosine_similarity,
    evaluate_semantic_retrieval,
    format_pinecone_markdown_report,
)
import numpy as np


class TestPineconeMemoryEval(unittest.TestCase):
    def test_cosine_similarity(self):
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(cosine_similarity(vec_a, vec_b), 1.0, places=4)
        self.assertAlmostEqual(cosine_similarity(vec_a, vec_c), 0.0, places=4)

    def test_semantic_retrieval_mini(self):
        from unittest.mock import patch
        # Mini subset of 3 profiles and 2 queries
        mini_profiles = CLINICAL_PROFILES[:3]
        mini_queries = [
            {
                "query_id": "Q-TEST-01",
                "query_text": "Severe chest pressure in patient with prior stent and aspirin allergy",
                "expected_patient_id": "PT-01",
            },
            {
                "query_id": "Q-TEST-02",
                "query_text": "Anaphylaxis to penicillin with stridor and wheezing",
                "expected_patient_id": "PT-02",
            },
        ]
        with patch("scripts.evaluate_pinecone_memory._get_embedding_model", return_value=None):
            results = evaluate_semantic_retrieval(mini_profiles, mini_queries)
        summary = results["evaluation_summary"]
        self.assertEqual(summary["total_queries"], 2)
        self.assertEqual(summary["total_indexed_profiles"], 3)
        self.assertGreater(summary["recall_at_3_pct"], 40.0)

    def test_markdown_report_formatter(self):
        mock_results = {
            "evaluation_summary": {
                "total_queries": 2,
                "total_indexed_profiles": 3,
                "mean_reciprocal_rank_mrr": 0.85,
                "recall_at_1_pct": 75.0,
                "recall_at_3_pct": 100.0,
                "recall_at_5_pct": 100.0,
                "ndcg_at_3": 0.90,
                "mean_retrieval_latency_ms": 15.2,
                "p95_retrieval_latency_ms": 22.0,
            },
            "query_level_evaluations": [
                {
                    "query_id": "Q-01",
                    "query_text": "Chest pain with prior stent",
                    "target_patient_name": "Robert Chen",
                    "retrieved_rank": 1,
                    "top_cosine_similarity": 0.89,
                    "critical_alert": "CAD alert",
                }
            ],
        }
        md = format_pinecone_markdown_report(mock_results)
        self.assertIn("Pinecone Retrieval Benchmark", md)
        self.assertIn("Recall@3", md)
        self.assertIn("100.0%", md)


if __name__ == "__main__":
    unittest.main()
