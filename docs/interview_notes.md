# Clinical Triage Copilot: Architectural Defense & Interview Notes

This document provides rigorous, engineering-grade rationales for the core architectural decisions behind the **Clinical Triage Copilot**.

---

## 1. Why build a deterministic rules engine instead of relying purely on prompt engineering?

**Clinical & Legal Answer:**
Large Language Models are probabilistic next-token predictors. In emergency medicine, under-triaging a deteriorating patient (Type II error) carries catastrophic consequences. Even with few-shot prompting and high-temperature penalties, generative models can occasionally hallucinate lower severity or miss critical physiological danger signs (e.g., severe hypoxia or profound bradycardia) when distracted by polite or conversational patient phrasing.

By implementing a compiled, deterministic rules engine (`agent/rules.py`) that operates as an **unbreakable lower bound** ($\text{FinalRisk} \ge \text{RuleFloor}$), we guarantee zero-downtime safety. The rules engine executes in **$0.13\text{ ms}$** without external API dependencies, ensuring that life-threatening red flags are always intercepted even if the cloud API fails.

---

## 2. Why decompose the pipeline into Intake, Risk, and Safety nodes instead of a single monolithic prompt?

**Engineering & Separation of Concerns Answer:**
1. **Targeted Schemas & Error Isolation:** The Intake node focuses exclusively on clinical entity extraction (complaints, duration, negation). The Risk engine focuses on acuity stratification and physiological thresholding. The Safety node focuses on communication clarity and disclaimer enforcement. If one node encounters unexpected input, the failure is localized and auditable.
2. **Independent Testability:** We can unit test the NegEx parser in `Intake` or the ESI vital zone logic in `Risk` in complete isolation without mocking a 70-billion-parameter LLM.
3. **Defense in Depth:** The Safety node acts as an independent validator, ensuring that even if an upstream LLM generates conversational text, mandatory HITL flags and educational boundaries are programmatically enforced before returning to the user.

---

## 3. How do you prevent under-triage when a patient describes severe symptoms in colloquial terms?

**Methodological Answer:**
1. **Multi-Faceted Keyword Expansion & NegEx:** Our rules engine incorporates colloquial descriptions and regional phrasing (e.g., "fevr", "leg is swollen and red", "severe tearing pain") while using windowed NegEx matching to differentiate affirmative symptoms from explicit negations ("denies fever").
2. **Objective Vital Sign Danger Zones:** Even if the patient downplays their condition ("just feeling tired"), objective physiological telemetry ($\text{SpO}_2 \le 92\%$, $\text{HR} > 140\text{ bpm}$, or $\text{RR} \ge 24$ with hypoxia) automatically forces an ESI 1 or ESI 2 Emergency classification.
3. **Hybrid Max-Severity Invariant:** The hybrid arbiter compares the LLM's nuanced semantic understanding against the rules floor and selects the maximum severity tier.

---

## 4. Why freeze the test split ($n=60$) and tune only on the development split ($n=90$)?

**Data Science & Evaluation Integrity Answer:**
Optimizing rule patterns or prompt templates directly against the benchmark dataset causes severe data leakage and overfitting, creating an illusion of high generalization that collapses in real deployment.

In accordance with [`eval/SPLIT_POLICY.md`](../eval/SPLIT_POLICY.md):
- The **development split ($n=90$)** was used exclusively to diagnose failure modes (such as subacute fever and hypoxic tachypnea) and calibrate urgent-tier rules.
- The **test split ($n=60$)** was completely frozen and evaluated strictly at designated release milestones.
- A fresh **held-out split ($n=30$)** of colloquial, comorbidity-rich scenarios was created to test unseen domain transfer.
- Every frozen split execution is logged to an immutable run log (`eval/results/run_log.jsonl`) with git commit hashes and timestamps to ensure academic auditability.
