# LLM vs. Rules: Architectural and Clinical Trade-off Analysis

## 1. The Core Tension in AI Clinical Triage

In emergency and urgent ambulatory intake, classification engines face an asymmetric risk profile:
- **Type I Error (Over-Triage):** A non-urgent patient is triaged as urgent or emergency. The consequence is excess resource consumption, longer ED waiting room queues, and modest operational overhead.
- **Type II Error (Under-Triage):** A deteriorating or unstable patient is triaged as routine. The consequence is catastrophic clinical decompensation, delayed resuscitation, and potential preventable mortality.

Consequently, any AI-assisted triage system must prioritize **high sensitivity / recall on emergency presentations** over raw aggregate accuracy.

---

## 2. Comparative Strengths & Failure Modes

| Dimension | Pure Deterministic Rules (Safety Floor) | Pure Generative LLM Zero-Shot | Hybrid Consensus Architecture |
|:---|:---|:---|:---|
| **Emergency Recall** | High & predictable; invariant against adversarial wording | Vulnerable to soft-pedaled phrasing and under-triage | **Guaranteed $\ge$ Rules floor** (mathematical invariant) |
| **P50 Latency** | **$0.13\text{ ms}$** (microsecond execution) | $\sim 1,500 - 3,500\text{ ms}$ (network + model decoding) | Sub-millisecond if offline; $1-3\text{s}$ in cloud |
| **Offline Reliability** | **$100\%$ zero-downtime**, no network or API keys | Fails completely on network disconnect or rate limits | **Graceful degradation** to safety floor if API fails |
| **Linguistic Nuance** | Limited by compiled regex / NegEx dictionary patterns | Excels at complex colloquial narratives and implicit distress | Combines deep contextual reasoning with red-flag floor |
| **Auditability** | Full determinism; exact rule triggering rationale | Non-deterministic tokens; stochastic variance | Full audit trail logging LLM rationale + rule flags |

---

## 3. Disagreement Anatomy: Where Rules and LLM Diverge

In empirical evaluations of clinical vignettes, three distinct patterns of divergence emerge:

### A. Occult Deterioration Overridden by Safety Floor
- **Scenario:** A patient reports "feeling a little dizzy and sweaty, nothing major" with objective vitals showing $\text{SpO}_2 = 88\%$ or $\text{HR} = 145\text{ bpm}$.
- **LLM Tendency:** Often anchors on the conversational reassurance ("nothing major") and classifies as urgent or routine.
- **Rules Safety Floor:** Instantly triggers Critical Hypoxia / Severe Tachycardia (ESI 1 or ESI 2).
- **Hybrid Resolution:** The safety invariant $\text{FinalRisk} = \max(\text{LLM}, \text{Rules})$ enforces emergency escalation, preventing fatal under-triage.

### B. Nuanced Multi-Morbid Synthesis
- **Scenario:** An elderly patient presents with vague epigastric fullness, nausea, and known coronary artery disease, without typical "crushing chest pain".
- **Rules Tendency:** Lacks explicit chest pain red flags; may classify as routine or low-tier urgent.
- **LLM Tendency:** Synthesizes the clinical gestalt of an atypical acute coronary syndrome presentation in an elderly diabetic, escalating to urgent/emergency.
- **Hybrid Resolution:** The hybrid model elevates the acuity above the rules baseline because the LLM identified high-risk holistic context.

---

## 4. Architectural Guarantee: The Safety Invariant

The core clinical invariant of this repository is defined in `agent/nodes/risk.py`:

$$\text{Severity}(\text{FinalRisk}) \ge \text{Severity}(\text{RuleFloor})$$

Under no circumstances can an LLM response downgrade a deterministic emergency or urgent red flag. If the generative model fails, times out, or produces a lower risk tier, the deterministic safety floor unconditionally governs patient disposition.
