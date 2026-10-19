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
