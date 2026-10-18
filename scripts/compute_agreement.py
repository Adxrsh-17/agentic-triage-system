"""Agreement analysis script for inter-rater reliability and clinical evaluation.

Calculates:
- Quadratic-weighted Cohen's kappa
- Linear-weighted Cohen's kappa
- Unweighted Cohen's kappa
- Exact agreement rate (%)
- Within-+-1 agreement rate (%)
- Wilson score confidence intervals
- Bootstrap 95% confidence intervals (1,000 resamples, deterministic seed)

Usage:
    python scripts/compute_agreement.py --results docs/evaluation_results.json
    python scripts/compute_agreement.py --csv docs/clinician_review_template.csv --rater1 proposed_esi --rater2 clinician_esi
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.eval_utils import (
    cohen_kappa,
    compute_all_bootstrap_cis,
    exact_agreement_rate,
    wilson_score_interval,
    within_1_agreement_rate,
)


def load_from_json_results(json_path: str) -> Tuple[List[int], List[int]]:
    """Extract ground truth and predicted ESI arrays from evaluation JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    y_true: List[int] = []
    y_pred: List[int] = []

    case_results = data.get("case_level_results", [])
    for c in case_results:
        # Expected keys: ground_truth_esi / actual_esi, and predicted_esi
        actual = c.get("ground_truth_esi") or c.get("actual_esi")
        predicted = c.get("predicted_esi")
        if actual is not None and predicted is not None:
            y_true.append(int(actual))
            y_pred.append(int(predicted))

    return y_true, y_pred


def load_from_csv(csv_path: str, col1: str, col2: str) -> Tuple[List[int], List[int]]:
    """Extract two rating columns from a review CSV."""
    y1: List[int] = []
    y2: List[int] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val1 = row.get(col1, "").strip()
            val2 = row.get(col2, "").strip()
            if val1.isdigit() and val2.isdigit():
                y1.append(int(val1))
                y2.append(int(val2))

    return y1, y2


def run_agreement_analysis(
    y_a: List[int],
    y_b: List[int],
    label_a: str = "Ground Truth",
    label_b: str = "Predicted",
) -> dict:
    if len(y_a) == 0:
        print("[ERROR] No valid paired ratings found for analysis.")
        return {}

    n = len(y_a)
    exact_acc = exact_agreement_rate(y_a, y_b)
    within_1_acc = within_1_agreement_rate(y_a, y_b)

    kappa_unweighted = cohen_kappa(y_a, y_b, weights="unweighted")
    kappa_linear = cohen_kappa(y_a, y_b, weights="linear")
    kappa_quad = cohen_kappa(y_a, y_b, weights="quadratic")

    # Wilson score interval for exact agreement
    exact_matches = sum(1 for a, b in zip(y_a, y_b) if a == b)
    w_low, w_high = wilson_score_interval(exact_matches, n, confidence=0.95)

    # Within-1 Wilson interval
    within_matches = sum(1 for a, b in zip(y_a, y_b) if abs(a - b) <= 1)
    within_w_low, within_w_high = wilson_score_interval(within_matches, n, confidence=0.95)

    # Bootstrap CIs
    bootstrap_cis = compute_all_bootstrap_cis(y_a, y_b, n_resamples=1000, seed=42)

    results = {
        "sample_size": n,
        "exact_agreement_pct": round(exact_acc * 100, 2),
        "exact_agreement_wilson_95ci": [round(w_low * 100, 2), round(w_high * 100, 2)],
        "within_1_agreement_pct": round(within_1_acc * 100, 2),
        "within_1_wilson_95ci": [round(within_w_low * 100, 2), round(within_w_high * 100, 2)],
        "cohen_kappa_unweighted": round(kappa_unweighted, 4),
        "cohen_kappa_linear": round(kappa_linear, 4),
        "cohen_kappa_quadratic": round(kappa_quad, 4),
        "bootstrap_95_cis": bootstrap_cis,
    }

    # Print clean text summary
    print("=" * 70)
    print(f"INTER-RATER AGREEMENT & RELIABILITY REPORT (n = {n})")
    print(f"Rater A: {label_a} | Rater B: {label_b}")
    print("=" * 70)
    print(f"Exact Agreement:              {results['exact_agreement_pct']}%  [95% Wilson CI: {results['exact_agreement_wilson_95ci'][0]}% - {results['exact_agreement_wilson_95ci'][1]}%]")
    print(f"Within-+-1 Level Agreement:   {results['within_1_agreement_pct']}%  [95% Wilson CI: {results['within_1_wilson_95ci'][0]}% - {results['within_1_wilson_95ci'][1]}%]")
    print(f"Cohen's Kappa (Quadratic):    {results['cohen_kappa_quadratic']:.4f}")
    print(f"Cohen's Kappa (Linear):       {results['cohen_kappa_linear']:.4f}")
    print(f"Cohen's Kappa (Unweighted):   {results['cohen_kappa_unweighted']:.4f}")
    print("-" * 70)
    print("Bootstrap 95% Confidence Intervals (1,000 resamples, seed=42):")
    for metric_name, ci_info in bootstrap_cis.items():
        print(f"  - {metric_name:<24}: {ci_info['point']:>6.2f}%  [{ci_info['ci_95_lower']:>6.2f}% - {ci_info['ci_95_upper']:>6.2f}%]")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute agreement and reliability metrics.")
    parser.add_argument("--results", type=str, help="Path to JSON evaluation results file.")
    parser.add_argument("--csv", type=str, help="Path to CSV review file.")
    parser.add_argument("--rater1", type=str, default="proposed_esi", help="CSV column for Rater 1.")
    parser.add_argument("--rater2", type=str, default="clinician_esi", help="CSV column for Rater 2.")
    parser.add_argument("--out", type=str, help="Optional output JSON path.")

    args = parser.parse_args()

    y_a: List[int] = []
    y_b: List[int] = []
    label_a = "Rater 1"
    label_b = "Rater 2"

    if args.results:
        y_a, y_b = load_from_json_results(args.results)
        label_a = "Ground Truth"
        label_b = "Predicted ESI"
    elif args.csv:
        y_a, y_b = load_from_csv(args.csv, args.rater1, args.rater2)
        label_a = args.rater1
        label_b = args.rater2
    else:
        # Default fallback: check docs/evaluation_results.json
        default_json = Path(__file__).resolve().parents[1] / "docs" / "evaluation_results.json"
        if default_json.exists():
            y_a, y_b = load_from_json_results(str(default_json))
            label_a = "Ground Truth"
            label_b = "Predicted ESI"
        else:
            parser.print_help()
            sys.exit(1)

    res = run_agreement_analysis(y_a, y_b, label_a, label_b)

    if args.out and res:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved agreement results to {args.out}")


if __name__ == "__main__":
    main()
