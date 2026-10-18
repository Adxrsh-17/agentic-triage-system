"""Learned Baseline ML Models for Emergency Severity Index (ESI) Triage.

Trains Logistic Regression and LightGBM classifiers on:
- TF-IDF representation of patient chief complaints
- Standardized physiological vital signs (HR, SBP, DBP, RR, SpO2, Temp)
- Demographics (Age, Sex) and comorbidity counts

Trained on DEV ($n=90$), evaluated on TEST ($n=60$).
Reports Exact Accuracy, Within-±1 Accuracy, High-Acuity Recall (ESI 1-2), UTR, OTR, and Cohen's Kappa.
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.eval_utils import (
    cohen_kappa,
    compute_all_bootstrap_cis,
    exact_agreement_rate,
    wilson_score_interval,
    within_1_agreement_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("learned_baseline")


def load_dataset(jsonl_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev_items: List[Dict[str, Any]] = []
    test_items: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            split = item.get("split", "dev").lower()
            if split == "dev":
                dev_items.append(item)
            elif split == "test":
                test_items.append(item)
    return dev_items, test_items


def extract_features(
    items: List[Dict[str, Any]],
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    is_fit: bool = False,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    texts = [item.get("chief_complaint", "") for item in items]
    y = np.array([int(item.get("gold_esi") or item.get("ground_truth_esi", 5)) for item in items])

    # TF-IDF text features
    if is_fit:
        X_text = vectorizer.fit_transform(texts)
    else:
        X_text = vectorizer.transform(texts)

    # Numerical tabular features
    num_rows: List[List[float]] = []
    for item in items:
        vitals = item.get("vitals") or {}
        hr = float(vitals.get("hr", 80.0))
        sbp = float(vitals.get("sbp", 120.0))
        dbp = float(vitals.get("dbp", 80.0))
        rr = float(vitals.get("rr", 16.0))
        spo2 = float(vitals.get("spo2", 98.0))
        temp = float(vitals.get("temp", 98.6))
        age = float(item.get("age", 40.0))
        is_female = 1.0 if str(item.get("sex", "")).lower() == "female" else 0.0
        comorbidities = item.get("comorbidities") or item.get("conditions") or []
        num_comorb = float(len(comorbidities))

        # Vital sign danger indicators
        hypoxia_flag = 1.0 if spo2 < 90.0 else (0.5 if spo2 <= 92.0 else 0.0)
        tachy_flag = 1.0 if hr > 140.0 or hr < 40.0 else 0.0
        hypotension_flag = 1.0 if sbp < 90.0 else 0.0
        resp_flag = 1.0 if rr > 30.0 or rr < 8.0 else 0.0

        num_rows.append([
            age,
            is_female,
            hr,
            sbp,
            dbp,
            rr,
            spo2,
            temp,
            num_comorb,
            hypoxia_flag,
            tachy_flag,
            hypotension_flag,
            resp_flag,
        ])

    X_num = np.array(num_rows, dtype=np.float32)
    if is_fit:
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        X_num_scaled = scaler.transform(X_num)

    X_combined = sp.hstack([X_text, sp.csr_matrix(X_num_scaled)], format="csr")
    return X_combined, y


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    n = len(y_true)
    exact_matches = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    within_1_matches = sum(1 for yt, yp in zip(y_true, y_pred) if abs(yt - yp) <= 1)
    under_triage = sum(1 for yt, yp in zip(y_true, y_pred) if yp > yt)
    over_triage = sum(1 for yt, yp in zip(y_true, y_pred) if yp < yt)

    # High-acuity resuscitation (ESI 1-2) sensitivity
    high_acuity_indices = [i for i, yt in enumerate(y_true) if yt in (1, 2)]
    high_acuity_tp = sum(1 for i in high_acuity_indices if y_pred[i] in (1, 2))
    high_acuity_recall = (high_acuity_tp / max(1, len(high_acuity_indices))) * 100.0

    exact_acc = (exact_matches / n) * 100.0
    within_1_acc = (within_1_matches / n) * 100.0
    utr = (under_triage / n) * 100.0
    otr = (over_triage / n) * 100.0

    kappa_quad = cohen_kappa(y_true, y_pred, weights="quadratic")
    kappa_linear = cohen_kappa(y_true, y_pred, weights="linear")
    kappa_unweighted = cohen_kappa(y_true, y_pred, weights="unweighted")

    # Confusion matrix
    cm = {i: {j: 0 for j in range(1, 6)} for i in range(1, 6)}
    for yt, yp in zip(y_true, y_pred):
        if 1 <= yt <= 5 and 1 <= yp <= 5:
            cm[yt][yp] += 1

    # Per class recall
    per_class: Dict[str, float] = {}
    for level in range(1, 6):
        level_cases = [yp for yt, yp in zip(y_true, y_pred) if yt == level]
        rec = (sum(1 for yp in level_cases if yp == level) / max(1, len(level_cases))) * 100.0
        per_class[f"ESI_{level}_recall"] = round(rec, 2)

    return {
        "n_samples": n,
        "exact_accuracy_pct": round(exact_acc, 2),
        "within_1_accuracy_pct": round(within_1_acc, 2),
        "high_acuity_recall_pct": round(high_acuity_recall, 2),
        "under_triage_rate_pct": round(utr, 2),
        "over_triage_rate_pct": round(otr, 2),
        "cohen_kappa_quadratic": round(kappa_quad, 4),
        "cohen_kappa_linear": round(kappa_linear, 4),
        "cohen_kappa_unweighted": round(kappa_unweighted, 4),
        "per_class_recall": per_class,
        "confusion_matrix": cm,
    }


def train_and_evaluate_baselines(data_path: Optional[Path] = None) -> Dict[str, Any]:
    if data_path is None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "vignettes_v2.jsonl"

    dev_items, test_items = load_dataset(data_path)
    logger.info("Loaded dataset: %d DEV cases, %d TEST cases", len(dev_items), len(test_items))

    vectorizer = TfidfVectorizer(max_features=250, ngram_range=(1, 2), stop_words="english")
    scaler = StandardScaler()

    X_dev, y_dev = extract_features(dev_items, vectorizer, scaler, is_fit=True)
    X_test, y_test = extract_features(test_items, vectorizer, scaler, is_fit=False)

    results: Dict[str, Any] = {}

    # Model 1: Logistic Regression (Balanced class weights)
    lr = LogisticRegression(max_iter=1500, class_weight="balanced", C=1.0, random_state=42)
    lr.fit(X_dev, y_dev)
    y_test_pred_lr = [int(p) for p in lr.predict(X_test)]
    results["logistic_regression"] = {
        "model_name": "TF-IDF + Logistic Regression (Balanced)",
        "test_metrics": evaluate_predictions(list(y_test), y_test_pred_lr),
        "predictions": y_test_pred_lr,
    }
    logger.info(
        "Logistic Regression -> Test Exact Acc: %.2f%%, Within-1 Acc: %.2f%%, High-Acuity Recall: %.2f%%, Kappa: %.4f",
        results["logistic_regression"]["test_metrics"]["exact_accuracy_pct"],
        results["logistic_regression"]["test_metrics"]["within_1_accuracy_pct"],
        results["logistic_regression"]["test_metrics"]["high_acuity_recall_pct"],
        results["logistic_regression"]["test_metrics"]["cohen_kappa_quadratic"],
    )

    # Model 2: LightGBM
    if HAS_LIGHTGBM:
        lgbm = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_dev.toarray(), y_dev)
        y_test_pred_lgbm = [int(p) for p in lgbm.predict(X_test.toarray())]
        results["lightgbm"] = {
            "model_name": "TF-IDF + LightGBM (Gradient Boosted Trees)",
            "test_metrics": evaluate_predictions(list(y_test), y_test_pred_lgbm),
            "predictions": y_test_pred_lgbm,
        }
        logger.info(
            "LightGBM -> Test Exact Acc: %.2f%%, Within-1 Acc: %.2f%%, High-Acuity Recall: %.2f%%, Kappa: %.4f",
            results["lightgbm"]["test_metrics"]["exact_accuracy_pct"],
            results["lightgbm"]["test_metrics"]["within_1_accuracy_pct"],
            results["lightgbm"]["test_metrics"]["high_acuity_recall_pct"],
            results["lightgbm"]["test_metrics"]["cohen_kappa_quadratic"],
        )

    return results


if __name__ == "__main__":
    out = train_and_evaluate_baselines()
    print("\n--- BASELINE RESULTS SUMMARY ---")
    for k, v in out.items():
        tm = v["test_metrics"]
        print(f"[{v['model_name']}]")
        print(f"  Exact Accuracy:        {tm['exact_accuracy_pct']}%")
        print(f"  Within-±1 Accuracy:    {tm['within_1_accuracy_pct']}%")
        print(f"  High-Acuity Recall:    {tm['high_acuity_recall_pct']}%")
        print(f"  Under-Triage Rate:     {tm['under_triage_rate_pct']}%")
        print(f"  Over-Triage Rate:      {tm['over_triage_rate_pct']}%")
        print(f"  Quadratic Cohen Kappa: {tm['cohen_kappa_quadratic']}")
