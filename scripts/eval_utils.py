"""Evaluation and statistical utilities for clinical triage benchmarking.

Provides:
- Wilson score intervals for binomial proportions
- Nonparametric bootstrap 95% confidence intervals (1,000 resamples, deterministic seed)
- Cohen's Kappa (unweighted, linear-weighted, quadratic-weighted)
- Inter-rater agreement metrics (exact agreement, within-+-1 agreement)
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of positive outcomes / successes.
        total: Total number of trials.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound) as floats between 0.0 and 1.0.
    """
    if total <= 0:
        return (0.0, 0.0)

    # Two-sided z-score (approx 1.95996 for 95%)
    # Standard normal quantile approximation
    alpha = 1.0 - confidence
    # For common confidence levels, use exact z
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    else:
        # Fallback to scipy if available or rational approximation
        try:
            from scipy.stats import norm
            z = float(norm.ppf(1.0 - alpha / 2.0))
        except ImportError:
            z = 1.95996

    p = successes / total
    denominator = 1.0 + (z ** 2) / total
    centre_adjusted_probability = p + (z ** 2) / (2.0 * total)
    adjusted_std_dev = math.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * total)) / total)

    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator

    return (max(0.0, float(lower)), min(1.0, float(upper)))


def cohen_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    weights: Optional[str] = "quadratic",
    categories: Sequence[int] = (1, 2, 3, 4, 5),
) -> float:
    """Calculate Cohen's Kappa with optional weighting (unweighted, linear, or quadratic).

    Args:
        rater_a: First rater scores (e.g. ground truth ESI 1-5).
        rater_b: Second rater scores (e.g. predicted ESI 1-5).
        weights: None / "unweighted", "linear", or "quadratic".
        categories: Sequence of distinct rating categories (defaults to ESI 1..5).

    Returns:
        Cohen's kappa coefficient (float).
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"Raters must have equal lengths: {len(rater_a)} vs {len(rater_b)}")
    if len(rater_a) == 0:
        return 0.0

    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Build observed confusion matrix
    cm = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(rater_a, rater_b):
        if a in cat_to_idx and b in cat_to_idx:
            cm[cat_to_idx[a], cat_to_idx[b]] += 1.0

    n = np.sum(cm)
    if n == 0:
        return 0.0

    # Build weight matrix
    w = np.zeros((k, k), dtype=np.float64)
    max_dist = float(k - 1)
    for i in range(k):
        for j in range(k):
            diff = abs(i - j)
            if weights is None or weights == "unweighted":
                w[i, j] = 0.0 if i == j else 1.0
            elif weights == "linear":
                w[i, j] = diff / max_dist if max_dist > 0 else 0.0
            elif weights == "quadratic":
                w[i, j] = (diff ** 2) / (max_dist ** 2) if max_dist > 0 else 0.0
            else:
                raise ValueError(f"Unknown weighting scheme: {weights}")

    # Marginal totals
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    expected = np.outer(row_sums, col_sums) / n

    # Weighted agreement
    po = 1.0 - np.sum(w * cm) / n
    pe = 1.0 - np.sum(w * expected) / n

    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - pe) < 1e-12 else 0.0

    kappa = (po - pe) / (1.0 - pe)
    return float(kappa)


def exact_agreement_rate(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Proportion of exact agreements between two raters."""
    if len(rater_a) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    return matches / len(rater_a)


def within_1_agreement_rate(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Proportion of agreements within +-1 acuity level."""
    if len(rater_a) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(rater_a, rater_b) if abs(a - b) <= 1)
    return matches / len(rater_a)


def compute_bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric_fn,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Nonparametric bootstrap confidence interval for a metric function f(y_true, y_pred).

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)
    if n == 0:
        return (0.0, 0.0, 0.0)

    point_estimate = float(metric_fn(y_true_arr, y_pred_arr))

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        sample_true = y_true_arr[indices]
        sample_pred = y_pred_arr[indices]
        boot_stats[i] = float(metric_fn(sample_true, sample_pred))

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_stats, lower_pct))
    ci_upper = float(np.percentile(boot_stats, upper_pct))

    return (point_estimate, ci_lower, ci_upper)


# Metric functions for bootstrap CI
def metric_exact_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def metric_within_1_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def metric_emergency_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sensitivity for ESI 1-2 (emergencies)."""
    mask = (y_true == 1) | (y_true == 2)
    if not np.any(mask):
        return 0.0
    detected = ((y_pred[mask] == 1) | (y_pred[mask] == 2))
    return float(np.mean(detected))


def metric_under_triage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Under-triage: predicted acuity is strictly less urgent than true (y_pred > y_true)."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_pred > y_true))


def metric_over_triage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Over-triage: predicted acuity is strictly more urgent than true (y_pred < y_true)."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_pred < y_true))


def compute_all_bootstrap_cis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap 95% CIs for primary clinical metrics."""
    metrics_map = {
        "exact_accuracy": metric_exact_acc,
        "within_1_accuracy": metric_within_1_acc,
        "emergency_sensitivity": metric_emergency_sensitivity,
        "under_triage_rate": metric_under_triage_rate,
        "over_triage_rate": metric_over_triage_rate,
    }

    results = {}
    for name, fn in metrics_map.items():
        point, low, high = compute_bootstrap_ci(
            y_true, y_pred, fn, n_resamples=n_resamples, seed=seed
        )
        results[name] = {
            "point": round(point * 100.0, 2),
            "ci_95_lower": round(low * 100.0, 2),
            "ci_95_upper": round(high * 100.0, 2),
        }
    return results
