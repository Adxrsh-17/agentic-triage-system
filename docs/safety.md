# Safety Notes, Ethics, and Governance Limitations

**Smart Triage AI:** Clinical Decision Support Research Prototype  
**Scope:** Educational and Engineering Demonstration $\vert$ Non-Diagnostic  

---

## 1. Regulatory Status & Explicit Clinical Disclaimer

> [!CAUTION]
> **NOT A CERTIFIED MEDICAL DEVICE (SaMD):**
> This software is an engineering research prototype designed to explore multi-agent consensus architectures in clinical informatics. It has not received clearance or approval from the U.S. Food and Drug Administration (FDA 510(k) or De Novo), the European Medicines Agency (CE mark under EU MDR 2017/745), or any health authority.
> 
> **Explicit Use Prohibitions:**
> - Must NOT be used for autonomous patient diagnosis, therapeutic prescription, or emergency department bed placement.
> - Must NOT be used as a substitute for the clinical judgment of a licensed physician or registered nurse.
> - All evaluations in this repository are based on synthetic, author-labeled vignettes.

---

## 2. Known Failure Modes & Vulnerabilities

In safety-critical healthcare AI, cataloging potential failure modes is essential for responsible deployment:

| Vulnerability | Failure Mechanism | Clinical Risk | System Mitigation |
|:---|:---|:---|:---|
| **Complex Negation Drift** | Highly nested, colloquial, or double negations (e.g., ``can't say I'm not feeling chest tightness'') may slip past heuristic tokenizers. | False-positive red-flag triggers or false-negative omissions. | Windowed NegEx algorithm with conjunction and adversative boundary parsing; LLM second-pass in hybrid mode. |
| **Linguistic & Dialectal Variance** | Colloquial idioms, code-mixing (e.g., Hinglish, regional vernacular), or heavy typos (e.g., ``chesst preashur''). | Sub-optimal entity extraction in offline keyword mode. | Hybrid mode routes to modern multilingual LLMs; fallback phonetic clustering expands slang. |
| **Anchoring on Patient Memory** | Retrieval of past historical conditions (e.g., recurrent panic attacks) might bias an LLM to discount an acute new presentation (e.g., true acute coronary syndrome). | Delayed diagnosis of novel acute disease. | Independent deterministic vital-sign and red-flag rules evaluate each acute encounter unconditionally before memory context is attached. |
| **Generative Hallucination** | Generative models asserting speculative etiologies or subtle diagnosis drift. | Patient anxiety or misdirected clinical workups. | Deterministic safety output validator screens all communication; strictly blacklists diagnostic affirmations and medication dosing. |

---

## 3. Core Architectural Safety Mitigations

1. **Inviolable Safety Floor:** The multi-agent consensus engine enforces that final acuity is bounded by deterministic rules:
   $$\text{Final Urgency} = \max(\text{Urgency}_{\text{Rules}}, \, \text{Urgency}_{\text{LLM}})$$
   The system can escalate acuity based on LLM comprehension, but cannot de-escalate risk below what objective vital signs and red-flag patterns mandate.
2. **Sub-Millisecond Resuscitation Fast-Path:** Catastrophic life threats ($\text{SpO}_2 < 90\%$, $\text{SBP} < 70$, $\text{HR} < 35$, comatose) bypass all generative reasoning in $0.42$ ms.
3. **Mandatory Human-in-the-Loop (HITL) Oversight:** All high-acuity cases (ESI 1--2) halt execution at a suspended state checkpoint awaiting licensed nurse sign-off or recorded clinical override.
4. **Deterministic Output Validation:** Patient-facing communication is post-processed by regex validators to strip any accidental diagnosis or dosage recommendations.

---

## 4. Patient Privacy, Data Protection & Production Requirements

In this open-source prototype, all data is synthetic. If productionized, would require de-identification, consent, encryption and audit logging (not implemented here):
- **On-Premises / Private VPC Hosting:** The prototype microservice and vector store would need to reside within a dedicated, secure Virtual Private Cloud or on-premises hospital cluster.
- **Local Private Inference:** External public LLM APIs should be replaced with local, air-gapped models (e.g., vLLM hosting open weights on hospital GPUs) to ensure zero Protected Health Information (PHI) crosses external boundaries.
- **Audit Logging & Event Sourcing:** Every interaction, nurse override, and acuity score is recorded in append-only logs for quality assurance and forensic review.
- **De-Identification:** Automated de-identification pipelines (Safe Harbor method) should strip names, addresses, and medical record numbers prior to vector indexing.

---

## 5. Metric Philosophy: Why Emergency Recall Outweighs Accuracy

In standard machine learning, overall accuracy is the benchmark KPI. In emergency clinical triage, **overall accuracy is secondary to Emergency Recall (Sensitivity on ESI 1--2)**:
- Missing a patient in septic shock (under-triage) causes fatal delays in antibiotic administration and fluid resuscitation.
- Over-triaging an uncomplicated viral illness to urgent care causes temporary resource friction, which is easily corrected upon bedside nurse inspection.
- Therefore, our optimization objective functions and evaluation metrics prioritize **$100\%$ detection of life threats** over aggregate multi-class accuracy.
