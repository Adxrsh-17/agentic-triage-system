# Evaluation Dataset & Benchmark Suite (`eval/`)

## 1. Dataset Overview (`eval/cases.jsonl`)

The dataset comprises **150 standardized clinical vignettes** formulated to evaluate clinical triage decision-support systems.

- **Total Cases:** 150
- **Split Breakdown:**
  - `dev` split ($n=90$, 60%): Used for threshold calibration, prompt engineering, and rule tuning.
  - `test` split ($n=60$, 40%): **Frozen held-out test split**, evaluated strictly for final benchmarks without rule tuning.
- **Urgency Distribution:**
  - `emergency`: 60 cases (ESI 1: 30, ESI 2: 30)
  - `urgent`: 30 cases (ESI 3: 30)
  - `routine`: 60 cases (ESI 4: 30, ESI 5: 30)
- **Clinical Categories:** Cardiac, Respiratory, Neurological, Gastrointestinal, Traumatic Injury, Infection, and General/Other.

---

## 2. Methodology & Caveats

> [!CAUTION]
> **Synthetic Author-Labeled Data Notice:**
> All vignettes in this benchmark are synthetic clinical scenarios authored and labeled by the research team aligned with the *Emergency Severity Index (ESI) Implementation Handbook (Version 4)*.
> 
> **Key Limitations:**
> 1. **Not Clinically Validated:** These vignettes do not originate from real Electronic Health Record (EHR) patient encounters.
> 2. **Potential Circularity:** Clinical scenarios drafted or regularized using language models risk linguistic circularity when evaluated against language model agents.
> 3. **Educational Prototype:** This system is an engineering research prototype, not a certified Medical Device (SaMD).

---

## 3. Evaluation Runner (`eval/run_eval.py`)

Run evaluation across execution modes:

```bash
# Run rules-only mode on the held-out test split
python -m eval.run_eval --mode rules --split test

# Run full hybrid mode (LLM + rules floor)
python -m eval.run_eval --mode hybrid --split test

# Run pure LLM zero-shot mode
python -m eval.run_eval --mode llm --split test
```

### Metrics Produced:
- Overall Accuracy & Macro-F1
- **Per-class Precision / Recall, with Emergency Recall highlighted** (primary clinical safety KPI)
- Under-Triage Rate (predicted less severe than label) vs. Over-Triage Rate
- Confusion Matrix (persisted as JSON and PNG)
- Turnaround Latency (P50, P95)
- LLM vs. Rules Disagreement Rate
- Automated Markdown Summary Table (`eval/results/summary.md`)

---

## 4. How to Read These Numbers (Statistical Uncertainty & Interpretation)

All performance metrics report exact case counts `(k/n)` alongside 95% Wilson score confidence intervals for proportions and 1,000-sample bootstrap intervals for Macro-F1. When reading these figures, note that sample sizes on held-out partitions are small ($n=60$ total on the test split, with only $n=12$ urgent cases). Consequently, per-class confidence intervals are wide (e.g., an urgent recall of $33.3\%$ carries a 95% CI of $[13.8\%, 60.9\%]$), reflecting substantial statistical uncertainty. Furthermore, because all vignettes are synthetic and author-labeled rather than clinical encounters drawn from live Electronic Health Records, these empirical metrics are indicative of architectural design properties rather than conclusive clinical efficacy.

