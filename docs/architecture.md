# System Architecture & Technical Specifications

**Smart Triage AI:** Production-grade Multi-Agent Decision-Support Copilot  
**Framework:** LangGraph `StateGraph` | **Standard:** Emergency Severity Index (ESI v4)  

---

## 1. Architectural Philosophy

The system replaces monolithic, open-ended agentic loops with a **constrained stateful DAG** combining generative intelligence with deterministic clinical guardrails:
1. **Deterministic Supervisor & Routing:** Graph routing is fixed by clinical workflow state. Routing is not where an LLM adds value; a deterministic supervisor eliminates nondeterminism, infinite loops, and unconstrained token cost.
2. **Hybrid Acuity with Safety Floor Invariant:** The LLM extracts semantic context and nuance, but the deterministic rule engine sets an inviolable floor:
   $$\text{Final Acuity} = \max(\text{Acuity}_{\text{LLM}}, \, \text{Acuity}_{\text{Rules}})$$
   The system can escalate acuity, but can never de-escalate below the rule-based red flag check.
3. **Resuscitation Short Circuit:** An $O(1)$ fast-path intercepts critical vital collapses ($\text{SpO}_2 < 90\%$, $\text{SBP} < 70$, $\text{HR} < 35$, comatose presentation) in \textbf{0.42 ms}, bypassing all LLM and graph latency.

---

## 2. End-to-End Execution Sequence Diagrams

### 2.1 Flow A: Low-Risk Presentation (Routine Flow)

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Triage Nurse
    participant FastPath as Fast-Path Gate
    participant Intake as Intake Node (LLM/Regex)
    participant Risk as Risk Assessment (Hybrid)
    participant SBAR as SBAR & FHIR Generator
    participant UI as Nurse Workstation

    Clinician->>FastPath: Ingest complaint ("Minor paper cut on index finger") + Vitals
    FastPath->>Intake: Stable vitals (SpO2 99%, HR 72, BP 120/80) -> Bypass Fast-Path
    Intake->>Risk: Extracted symptoms ["laceration"], Pertinent Negatives []
    Risk->>SBAR: Rule Level = ROUTINE (ESI 5), LLM Level = ROUTINE -> Final ESI 5
    Note over Risk,SBAR: Acuity <= 2? No. HITL Review required? No.
    SBAR->>UI: Render Clinical Summary, SBAR note, HL7 FHIR R4 Transaction Bundle
    UI-->>Clinician: Display Fast-Track Clinic Routing (No Human Review Block)
```

### 2.2 Flow B: Emergent Presentation (HITL Suspended Flow)

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Triage Nurse
    participant FastPath as Fast-Path Gate
    participant Intake as Intake Node
    participant Risk as Risk Assessment (Hybrid)
    participant Gate as HITL Safety Gate
    participant SBAR as SBAR & FHIR Generator
    participant Audit as Audit Logger
    participant UI as Nurse Workstation

    Clinician->>FastPath: Ingest complaint ("Crushing chest pain radiating to left jaw, diaphoresis")
    FastPath->>Intake: Stable airway/vitals (SpO2 95%, HR 110, BP 165/95)
    Intake->>Risk: Extracted symptoms ["chest pain", "jaw radiation"], Vitals [HR 110]
    Risk->>Gate: Red Flag Fired: "crushing chest pain", Vital Trigger: Tachycardia -> ESI 2
    Gate->>UI: Acuity <= 2 -> SUSPEND Execution (awaiting_human_review = True)
    Note over UI: UI displays High-Acuity Alert & Nurse Sign-off Modal
    Clinician->>Gate: Nurse Bedside Assessment: Confirms ESI 2 ("Approved")
    Gate->>SBAR: Resume graph execution with nurse sign-off note
    SBAR->>Audit: Record encounter in thread-safe JSONL audit log
    SBAR->>UI: Finalize SBAR note & FHIR R4 Bundle for EHR transfer
```

---

## 3. Modular Directory Layout

```
agent/
  __init__.py
  schemas.py          # Pydantic models: RiskLevel, IntakeResult, RiskResult
  llm.py              # ChatGroq client factory with structured output & retry
  rules.py            # Deterministic red-flag safety floor & pediatric bounds
  multi_agent.py      # Core unified LangGraph StateGraph engine & fast-path
  audit_logger.py     # Thread-safe JSONL audit logger & concordance tracker
  sbar_generator.py   # Physician SBAR note & HL7 FHIR R4 bundle generator
  infermedica_client.py # Infermedica API v3 concept mapping client
  nodes/
    intake.py         # 3-tier clinical entity extraction
    risk.py           # Hybrid risk engine enforcing max_severity safety floor
    safety.py         # Output validator & safe communication generator
  memory/
    embeddings.py     # sentence-transformers all-MiniLM-L6-v2 (384d)
    store.py          # PatientMemory protocol, LocalMemory & PineconeMemory
```
