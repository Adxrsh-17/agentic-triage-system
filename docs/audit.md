# System Audit & Codebase Inventory

**Repository:** `agentic-triage-system`  
**Date:** October 2026  
**Auditor:** Senior AI Engineer / Clinical Decision Support Systems Group  

---

## 1. Executive Summary

This audit assesses the state of the `agentic-triage-system` codebase against the requirements of `IMPROVEMENT_PLAN.md`. The repository implements a LangGraph-orchestrated clinical decision-support copilot aligned with the Emergency Severity Index (ESI v4) standard.

---

## 2. Graph Node Inventory & Behavioral Trace

The execution graph (`agent/multi_agent.py`) is structured as a stateful Directed Acyclic Graph (`StateGraph`) with the following nodes:

| Node Name | Handler Function | Operational Role | Primary Failure Modes & Fallbacks |
|:---|:---|:---|:---|
| **Fast-Path** | `_check_esi_1_fast_path` | $O(1)$ pre-graph short circuit evaluating physiological collapse ($\text{SpO}_2 < 90\%$, $\text{SBP} < 70$, $\text{HR} < 35$, resuscitation keywords). | Deterministic regex; zero network dependency. Fallback: continue to graph. |
| **Intake** | `assess_intake_node` | Extracts symptoms, duration, severity, and vitals from narrative chief complaints. | Tier 1: Infermedica API v3 $\rightarrow$ Tier 2: Groq LLaMA-3.3-70B JSON mode $\rightarrow$ Tier 3: NegEx regex tokenizer. |
| **Risk Assessment** | `risk_assessment_node` | Evaluates ESI v4 rules, vital danger zones, age-adjusted pediatric thresholds, and consensus formula $\text{ESI} = \min(\text{ESI}_{\text{Rules}}, \text{ESI}_{\text{LLM}})$. | Deterministic rule floor guarantees no de-escalation of red-flag symptoms. |
| **SBAR Generator** | `sbar_generator_node` | Synthesizes standardized Situation, Background, Assessment, Recommendation clinical handoff report. | Template-driven with LLM contextual enrichment; pure template fallback if LLM offline. |
| **FHIR Generator** | `fhir_generator_node` | Compiles compliant HL7 FHIR R4 Transaction Bundle (Patient, Encounter, Observation, Condition). | Pure deterministic serialization using standard LOINC / SNOMED coding. |
| **Human Review** | `human_review_node` | HITL safety gate. Pauses execution for all ESI 1--2 cases; accepts nurse sign-off or recorded clinical override. | Suspends execution state (`pending_state`); resumes via `POST /v1/triage/review`. |
| **Finish** | `finish_node` | Finalizes state metrics, records node execution durations, and closes the encounter trace. | Pure deterministic accumulator. |

---

## 3. External API & Integration Usage

### 3.1 Groq Cloud API
- **Location:** `agent/multi_agent.py` (`ChatGroq`)
- **Usage:** Structured JSON output extraction for clinical intake (`IntakeExtraction`) and adaptive follow-up question generation.
- **Resilience:** Wrapped in timeout (3.0s) and zero-retry fail-fast handlers. Under HTTP 429 rate limits or network timeout, catches exception and falls back to offline NegEx regex rules in $< 1$ ms.

### 3.2 Pinecone Vector Database
- **Location:** `scripts/evaluate_pinecone_memory.py`, `agent/multi_agent.py`
- **Usage:** Serverless index for semantic patient memory and historical encounter retrieval.
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalized).
- **Resilience:** In-memory cosine similarity fallback (`LocalMemory`) when `PINECONE_API_KEY` is not present.

### 3.3 Infermedica API v3
- **Location:** `agent/infermedica_client.py`
- **Usage:** Certified medical concept parsing (`/parse`) and evidence-based clinical triage (`/triage`).
- **Resilience:** Gracefully skipped when unconfigured; consensus engine operates purely on Groq + deterministic rules.

---

## 4. Dead Code, Hacks, & Legacy Isolation

1. **Peripheral Booking & Dispatch Tools:**
   - Previous iterations included hospital bed booking and ambulance dispatch tools that were peripheral to the core clinical triage copilot.
   - **Resolution:** In Task 4, these tools were archived to `archive/legacy/` and decoupled from the multi-agent graph.
2. **`sitecustomize.py`:**
   - **Contents:** Appends `site.getusersitepackages()` to `sys.path`.
   - **Purpose:** On Windows systems where user-level pip packages are installed in `AppData\Roaming\Python\Python312\site-packages`, this ensures sub-processes and pytest runners reliably find installed dependencies.
   - **Status:** Benign environment compatibility helper; retained for Windows execution stability.

---

## 5. Modularization Plan (Phase 1--2 Alignment)

To satisfy `IMPROVEMENT_PLAN.md` while preserving 100% backwards compatibility with our 61 passing tests, the monolithic `agent/multi_agent.py` will be cleanly decoupled into:
- `agent/schemas.py`: Pydantic models (`RiskLevel`, `IntakeResult`, `RiskResult`).
- `agent/llm.py`: Resilient LLM client factory with structured output and fallback handling.
- `agent/rules.py`: Deterministic red-flag safety floor and age-adjusted vital checks.
- `agent/memory/embeddings.py`: Sentence-transformer embedding generation with offline fallback.
- `agent/memory/store.py`: `PatientMemory` protocol with `LocalMemory` and `PineconeMemory` implementations.
- `agent/nodes/`: Modular node handlers (`intake.py`, `risk.py`, `safety.py`, `human_review.py`).
- `agent/graph.py`: StateGraph orchestration and compilation.

`agent/multi_agent.py` will re-export all core interfaces (`MultiAgentSystem`, `get_multi_agent`, `assess_medical_risk`, `_check_esi_1_fast_path`, etc.) ensuring zero breaking changes across test suites, `app.py`, and `api.py`.
