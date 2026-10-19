# System Components & Architectural Scope Boundaries

This document outlines the core functional modules, supplementary services, and experimental integrations within the **Clinical Triage Copilot** repository.

---

## 1. Modular Core Architecture

### A. Modular LangGraph Triage Pipeline (`agent/multi_agent.py`, `agent/nodes/`)
- **Intake Node (`agent/nodes/intake.py`):** Extracts structured chief complaints, duration, age, sex, and known comorbidities from raw text.
- **Hybrid Risk Engine (`agent/nodes/risk.py`):** Enforces the deterministic safety floor invariant:
  $$\text{FinalRisk} = \max(\text{LLM\_Level}, \text{Rule\_Level})$$
- **Safety Node (`agent/nodes/safety.py`):** Enforces mandatory educational disclaimers, generates safe patient-facing instructions, and alerts on Human-In-The-Loop (HITL) requirements.
- **Rule Engine (`agent/rules.py`):** Fully deterministic clinical safety floor implementing windowed NegEx matching, pediatric ESI v4 vital sign boundaries, and emergency red-flag compilation.

---

## 2. Service & Interface Layers

### B. FastAPI Service (`api.py`)
- **Purpose:** Production-style REST interface for headless automated triage pipelines.
- **Key Endpoints:**
  - `GET /health`: Healthcheck, runtime mode indicator (`rules`, `hybrid`, `llm`), and active LLM status.
  - `POST /triage`: Accepts structured patient encounters (`chief_complaint`, `vitals`, `age`, `sex`) and returns validated triage acuity, ESI level, red flags, and SBAR notes.
- **Current Scope / Limitations:** Intended as an educational prototype; does not implement OAuth2 / JWT authentication, TLS termination, or HIPAA-compliant data retention.

### C. Streamlit Demonstration UI (`app.py`)
- **Purpose:** Interactive clinician/nurse demonstration copilot workstation.
- **Features:** Dynamic vital sign danger indicators, live red-flag detection, SBAR handoff report generation, and downloadable FHIR R4 JSON clinical encounter bundles.
- **Current Scope / Limitations:** Client-side prototype interface; encounters are stored in local session state.

### D. Audit Logger (`agent/audit_logger.py`)
- **Purpose:** Verifiable append-only local audit logger storing encounter logs to `docs/triage_encounters_log.jsonl`.
- **Recorded Attributes:** Timestamp, patient ID hash, execution mode, final triage level, deterministic rule flags, LLM agreement/disagreement, and mandatory disclaimer acknowledgment.

---

## 3. Optional & Experimental Integrations

### E. Infermedica API Client (`agent/infermedica_client.py`)
- **Role:** *Optional / Experimental third-party diagnostic cross-check.*
- **Status:** Degrades gracefully to local rules if `INFERMEDICA_APP_ID` and `INFERMEDICA_APP_KEY` are unset.
- **Scope Boundary:** Not part of the primary benchmark evaluation loop; provided as an external reference integration.

### F. Pinecone Semantic Encounter Memory (`agent/multi_agent.py`)
- **Role:** *Optional longitudinal patient history recall.*
- **Status:** Generates 384-dimensional dense semantic embeddings (`all-MiniLM-L6-v2`) when configured, with an instant offline hash-based fallback when unconfigured or in offline test mode.
- **Scope Boundary:** Disabled during automated offline test suites to ensure zero network latency and $100\%$ test reproducibility.
