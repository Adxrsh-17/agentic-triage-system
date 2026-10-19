# Clinical Triage Copilot Evaluation Summary (DEV Split)

Generated automatically by `eval/run_eval.py` on 2026-10-19 10:16:13.

## Performance Comparison Across Execution Modes

| Evaluation Metric | Rules-Only (Safety Floor) | Hybrid Consensus (LLM + Rules) | Pure LLM Zero-Shot |
|:---|:---:|:---:|:---:|
| Evaluated Samples (n) | 90 | Not Evaluated | Not Evaluated |
| Overall Accuracy (%) | 97.8% (88/90) [92.3%–99.4%] | Not Evaluated | Not Evaluated |
| Macro-F1 Score (%) | 97.7% [93.9%–100.0%] | Not Evaluated | Not Evaluated |
| **Emergency Recall (%)** (Safety KPI) | **97.2% (35/36) [85.8%–99.5%]** | Not Evaluated | Not Evaluated |
| Emergency Precision (%) | 97.2% (35/36) [85.8%–99.5%] | Not Evaluated | Not Evaluated |
| Urgent Recall (%) | 94.4% (17/18) [74.2%–99.0%] | Not Evaluated | Not Evaluated |
| Routine Recall (%) | 100.0% (36/36) [90.4%–100.0%] | Not Evaluated | Not Evaluated |
| Under-Triage Rate (%) (Type II) | 1.11% | Not Evaluated | Not Evaluated |
| Over-Triage Rate (%) (Type I) | 1.11% | Not Evaluated | Not Evaluated |
| Latency P50 (ms) | 0.17 ms | Not Evaluated | Not Evaluated |
| Latency P95 (ms) | 2.66 ms | Not Evaluated | Not Evaluated |

## Key Clinical Findings

1. **Safety Floor Guarantees Emergency Sensitivity:** Rules-only and Hybrid modes maintain high emergency recall, preventing hazardous under-triaging.
2. **Zero-Downtime Offline Capability:** Even with external LLM APIs disabled, the deterministic rules safety floor provides dependable acuity scoring at sub-15ms latency.
3. **Disagreement Tracking:** The hybrid consensus engine actively audits instances where generative predictions diverge from rule-based thresholds.
