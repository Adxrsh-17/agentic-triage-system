"""Unit tests for learned baseline models and ablation study evaluation utilities."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.learned_baseline import evaluate_predictions, extract_features, load_dataset
from scripts.run_ablation import format_ablation_markdown_table


class TestAblationAndBaseline(unittest.TestCase):
    def test_evaluate_predictions_metrics(self):
        y_true = [1, 2, 3, 4, 5, 1, 2, 3]
        y_pred = [1, 2, 4, 4, 5, 2, 2, 3]  # exact: 6/8 = 75%, within-1: 8/8 = 100%

        res = evaluate_predictions(y_true, y_pred)
        self.assertEqual(res["n_samples"], 8)
        self.assertEqual(res["exact_accuracy_pct"], 75.0)
        self.assertEqual(res["within_1_accuracy_pct"], 100.0)
        self.assertGreater(res["cohen_kappa_quadratic"], 0.70)
        self.assertEqual(res["high_acuity_recall_pct"], 100.0)

    def test_ablation_markdown_formatter(self):
        sample_study = {
            "config_1_full": {
                "config_name": "Full System",
                "metrics": {
                    "exact_accuracy_pct": 58.33,
                    "within_1_accuracy_pct": 88.33,
                    "high_acuity_recall_pct": 87.5,
                    "esi_1_recall_pct": 83.33,
                    "under_triage_rate_pct": 36.67,
                    "over_triage_rate_pct": 5.0,
                    "cohen_kappa_quadratic": 0.8145,
                    "mean_latency_ms": 3500.0,
                },
            }
        }
        md = format_ablation_markdown_table(sample_study)
        self.assertIn("Full System", md)
        self.assertIn("58.33%", md)
        self.assertIn("0.8145", md)
        self.assertIn("Physiological Vital Signs", md)


if __name__ == "__main__":
    unittest.main()
