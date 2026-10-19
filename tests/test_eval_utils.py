"""Tests for evaluation utilities and statistical functions."""

import json
import pytest
from scripts.eval_utils import (
    cohen_kappa,
    compute_all_bootstrap_cis,
    compute_bootstrap_ci,
    exact_agreement_rate,
    metric_exact_acc,
    wilson_score_interval,
    within_1_agreement_rate,
)
from scripts.compute_agreement import run_agreement_analysis


def test_wilson_score_interval():
    # 0 successes out of 100
    low, high = wilson_score_interval(0, 100)
    assert 0.0 <= low <= 0.01
    assert 0.0 < high < 0.05

    # 50 out of 100 (symmetric around 0.5)
    low, high = wilson_score_interval(50, 100)
    assert 0.40 < low < 0.45
    assert 0.55 < high < 0.60
    assert abs((low + high) / 2.0 - 0.5) < 0.01

    # 5 out of 8 (e.g. roughly [0.31, 0.86])
    low, high = wilson_score_interval(5, 8)
    assert 0.30 <= low <= 0.32
    assert 0.85 <= high <= 0.87

    # Edge cases
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_cohen_kappa():
    # Perfect agreement
    rater1 = [1, 2, 3, 4, 5]
    rater2 = [1, 2, 3, 4, 5]
    assert pytest.approx(cohen_kappa(rater1, rater2, weights="quadratic"), 0.01) == 1.0
    assert pytest.approx(cohen_kappa(rater1, rater2, weights="linear"), 0.01) == 1.0
    assert pytest.approx(cohen_kappa(rater1, rater2, weights="unweighted"), 0.01) == 1.0

    # Near agreement (+-1 shifts)
    rater_a = [1, 2, 3, 4, 5]
    rater_b = [2, 2, 3, 4, 4]
    k_quad = cohen_kappa(rater_a, rater_b, weights="quadratic")
    k_lin = cohen_kappa(rater_a, rater_b, weights="linear")
    k_unw = cohen_kappa(rater_a, rater_b, weights="unweighted")
    # Quadratic kappa penalizes distant misses more than close ones
    assert k_quad >= k_lin >= k_unw


def test_agreement_rates():
    rater1 = [1, 2, 3, 4, 5]
    rater2 = [1, 2, 4, 4, 3]
    # Exact: 1==1, 2==2, 4==4 -> 3/5 = 0.6
    assert exact_agreement_rate(rater1, rater2) == 0.6
    # Within 1: |3-4|=1 (ok), |5-3|=2 (no) -> 4/5 = 0.8
    assert within_1_agreement_rate(rater1, rater2) == 0.8


def test_bootstrap_cis():
    y_true = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    y_pred = [1, 1, 2, 3, 3, 3, 4, 5, 5, 5]
    cis = compute_all_bootstrap_cis(y_true, y_pred, n_resamples=200, seed=42)
    assert "exact_accuracy" in cis
    assert "emergency_sensitivity" in cis
    assert "under_triage_rate" in cis
    assert "over_triage_rate" in cis
    assert cis["exact_accuracy"]["ci_95_lower"] <= cis["exact_accuracy"]["point"]
    assert cis["exact_accuracy"]["point"] <= cis["exact_accuracy"]["ci_95_upper"]


def test_run_agreement_analysis():
    y_true = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    res = run_agreement_analysis(y_true, y_pred)
    assert res["exact_agreement_pct"] == 100.0
    assert res["within_1_agreement_pct"] == 100.0
    assert res["cohen_kappa_quadratic"] == 1.0
