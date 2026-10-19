# Dataset Split Governance & Evaluation Policy

To avoid benchmark contamination and preserve scientific integrity, this repository enforces a three-tier dataset split policy.

---

## 1. Split Tiers and Rules of Engagement

| Split | Size ($n$) | Purpose | Governance Rule |
|:---|:---:|:---|:---|
| **`dev`** | 90 | Prompt engineering, keyword tuning, rule bank calibration, threshold sweeps | **Tune Freely:** Rule adjustments and heuristic calibrations may cite and inspect misclassified cases from this split only. |
| **`test`** | 60 | Milestone generalization benchmarking | **Frozen:** Must not be inspected to reverse-engineer or tweak rules. Run strictly once per milestone release. All runs are permanently appended to `eval/results/run_log.jsonl`. |
| **`holdout`** | 30 | Blind out-of-distribution validation | **Blind:** Fresh author-written scenarios incorporating conversational noise, colloquial phrasing, and chronic comorbidities. Never inspected during development. Requires human clinician confirmation (`reviewed: true`) prior to reporting. |

---

## 2. Inviolable Governance Principles

1. **No Rule Tuning on Test or Holdout:**
   Developers must never inspect failure cases on `test` or `holdout` to craft new keyword patterns or regex rules. All error analyses driving rule refinements must originate exclusively from `dev` cases.
2. **Permanent Run Logging:**
   Whenever `eval/run_eval.py` is invoked with `--split test` or `--split holdout`, the evaluation harness automatically logs the timestamp, execution mode, split, and active git commit SHA to `eval/results/run_log.jsonl`.
3. **Holdout Verification Gate:**
   Cases in `eval/holdout_cases.jsonl` carry `"reviewed": false` by default. Before reporting holdout results in external documentation or publications, each ground-truth label must be reviewed by a human clinical expert and flipped to `"reviewed": true`.
