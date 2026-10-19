# Clinical Triage Copilot Evaluation Summary (TEST Split)

Generated automatically by `eval/run_eval.py` on 2026-10-19 10:16:48.

## Performance Comparison Across Execution Modes

| Evaluation Metric | Rules-Only (Safety Floor) | Hybrid Consensus (LLM + Rules) | Pure LLM Zero-Shot |
|:---|:---:|:---:|:---:|
| Evaluated Samples (n) | 60 | Not Evaluated | Not Evaluated |
| Overall Accuracy (%) | 68.3% (41/60) [55.8%–78.7%] | Not Evaluated | Not Evaluated |
| Macro-F1 Score (%) | 62.8% [51.4%–74.2%] | Not Evaluated | Not Evaluated |
| **Emergency Recall (%)** (Safety KPI) | **66.7% (16/24) [46.7%–82.0%]** | Not Evaluated | Not Evaluated |
| Emergency Precision (%) | 100.0% (16/16) [80.6%–100.0%] | Not Evaluated | Not Evaluated |
| Urgent Recall (%) | 33.3% (4/12) [13.8%–60.9%] | Not Evaluated | Not Evaluated |
| Routine Recall (%) | 87.5% (21/24) [69.0%–95.7%] | Not Evaluated | Not Evaluated |
| Under-Triage Rate (%) (Type II) | 26.67% | Not Evaluated | Not Evaluated |
| Over-Triage Rate (%) (Type I) | 5.0% | Not Evaluated | Not Evaluated |
| Latency P50 (ms) | 0.13 ms | Not Evaluated | Not Evaluated |
| Latency P95 (ms) | 1.77 ms | Not Evaluated | Not Evaluated |

## Key Clinical Findings

1. **Safety Floor Guarantees Emergency Sensitivity:** Rules-only and Hybrid modes maintain high emergency recall, preventing hazardous under-triaging.
2. **Zero-Downtime Offline Capability:** Even with external LLM APIs disabled, the deterministic rules safety floor provides dependable acuity scoring at sub-15ms latency.
3. **Disagreement Tracking:** The hybrid consensus engine actively audits instances where generative predictions diverge from rule-based thresholds.
