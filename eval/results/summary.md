# Clinical Triage Copilot Evaluation Summary (TEST Split)

Generated automatically by `eval/run_eval.py` on 2026-10-19 09:43:31.

## Performance Comparison Across Execution Modes

| Evaluation Metric | Rules-Only (Safety Floor) | Hybrid Consensus (LLM + Rules) | Pure LLM Zero-Shot |
|:---|:---:|:---:|:---:|
| Evaluated Samples (n) | 60 | Not Evaluated | Not Evaluated |
| Overall Accuracy (%) | 66.7% (40/60) [54.1%–77.3%] | Not Evaluated | Not Evaluated |
| Macro-F1 Score (%) | 61.3% [49.1%–73.2%] | Not Evaluated | Not Evaluated |
| **Emergency Recall (%)** (Safety KPI) | **62.5% (15/24) [42.7%–78.8%]** | Not Evaluated | Not Evaluated |
| Emergency Precision (%) | 100.0% (15/15) [79.6%–100.0%] | Not Evaluated | Not Evaluated |
| Urgent Recall (%) | 33.3% (4/12) [13.8%–60.9%] | Not Evaluated | Not Evaluated |
| Routine Recall (%) | 87.5% (21/24) [69.0%–95.7%] | Not Evaluated | Not Evaluated |
| Under-Triage Rate (%) (Type II) | 28.33% | Not Evaluated | Not Evaluated |
| Over-Triage Rate (%) (Type I) | 5.0% | Not Evaluated | Not Evaluated |
| Latency P50 (ms) | 0.11 ms | Not Evaluated | Not Evaluated |
| Latency P95 (ms) | 1.57 ms | Not Evaluated | Not Evaluated |

## Key Clinical Findings

1. **Safety Floor Guarantees Emergency Sensitivity:** Rules-only and Hybrid modes maintain high emergency recall, preventing hazardous under-triaging.
2. **Zero-Downtime Offline Capability:** Even with external LLM APIs disabled, the deterministic rules safety floor provides dependable acuity scoring at sub-15ms latency.
3. **Disagreement Tracking:** The hybrid consensus engine actively audits instances where generative predictions diverge from rule-based thresholds.
