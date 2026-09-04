# Smart Triage AI — Multi-Agent Clinic Triage Copilot

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Infermedica v3](https://img.shields.io/badge/Clinical%20Engine-Infermedica%20v3-00b4d8.svg)](https://developer.infermedica.com/)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq%20Cloud-f55036.svg)](https://groq.com/)
[![Pinecone](https://img.shields.io/badge/Vector%20Memory-Pinecone%20384d-0466c8.svg)](https://www.pinecone.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Clinical Decision Support Positioning:** Smart Triage AI is designed exclusively as an intelligent intake assistant for licensed healthcare personnel (triage nurses, intake coordinators, and emergency medical technicians). It streamlines clinical intake workflows, flags red flags, suggests acuity levels, and surfaces nearby resources, while ensuring a **licensed clinician retains ultimate medical authority and makes the final triage decision.**

---

## Table of Contents

- [Overview & Problem Statement](#overview--problem-statement)
- [Key Features](#key-features)
- [Multi-Agent Architecture & Workflow](#multi-agent-architecture--workflow)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Configuration (.env)](#configuration-env)
- [Running the Application](#running-the-application)
- [Clinical Workflow & Usage Examples](#clinical-workflow--usage-examples)
- [Semantic Memory & Data Pipeline](#semantic-memory--data-pipeline)
- [Testing & Validation](#testing--validation)
- [Limitations & Future Roadmap](#limitations--future-roadmap)
- [License & Medical Disclaimer](#license--medical-disclaimer)

---

## Overview & Problem Statement

Emergency departments and urgent care clinics face severe intake bottlenecks, leading to delayed interventions, clinician burnout, and risks of under-triaging acute conditions. 

**Smart Triage AI** transforms standard patient intake into an evidence-based, multi-agent copilot that:
1. Translates unstructured patient complaints into structured medical concepts via **Infermedica's certified clinical knowledge base** and **Groq LLM tool-calling**.
2. Applies a rigorous **"Escalate, Never Downgrade"** consensus arbitration algorithm aligned with the **Emergency Severity Index (ESI Levels 1–5)**.
3. Generates tailored, comorbidity-aware **Over-The-Counter (OTC) medication and supportive care guidance**.
4. Automatically discovers real, nearby healthcare facilities (hospitals or pharmacies) and constructs direct **turn-by-turn live GPS navigation routes**.
5. Implements **Human-in-the-Loop (HITL) clinical review gates** on emergent presentations before dispatch.

---

## Key Features

### 🩺 Dual-Engine Clinical Intelligence
- **Infermedica Engine (v3 API):** Integrates `/v3/parse` for structured symptom/pertinent-negative extraction, `/v3/diagnosis` for diagnostic reasoning, and `/v3/triage` for certified clinical triage verdicts.
- **3-Tier Resilient Intake Fallback:**
  - *Tier 1 (Primary):* Infermedica Clinical Engine v3.
  - *Tier 2 (Secondary Fallback):* Groq Cloud LLM tool-calling (`ChatGroq`) with strict Pydantic schema validation.
  - *Tier 3 (Tertiary Fallback):* Deterministic regex keyword parsing and rule banks for zero-downtime offline execution.

### ⚖️ "Escalate, Never Downgrade" ESI Consensus Protocol
- Evaluates vital instability, symptom duration, acute red flags, and comorbidity risk factors against the 5-level **Emergency Severity Index (ESI)**:
  - **ESI 1 (Resuscitation):** Immediate life-threatening presentation (unresponsive, anaphylaxis, severe respiratory arrest).
  - **ESI 2 (Emergent):** High-risk, acute distress, chest pain, stroke signs (*requires mandatory HITL review*).
  - **ESI 3 (Urgent):** Stable vitals requiring multiple clinical resources/diagnostic tests.
  - **ESI 4 (Less Urgent):** Simple presentation requiring a single clinical resource.
  - **ESI 5 (Non-Urgent):** Routine outpatient presentation requiring no acute hospital resources.
- **Consensus Rule:** If either the internal ESI algorithm or Infermedica identifies a higher acuity or emergent condition, the final score automatically escalates to the highest severity (`min(internal_esi, infermedica_esi)`).

### 💊 Clinical OTC & Supportive Care Guidance
- Provides actionable, symptom-targeted Over-The-Counter remedies and supportive care recommendations.
- Accounts for patient age, biological sex, reported symptoms, and known chronic comorbidities:
  - **Liver Disease:** Automatically alerts and caps Paracetamol/Acetaminophen to safe thresholds (< 2,000 mg/24h).
  - **Ulcers / GERD / Renal Impairment / Asthma:** Safely suppresses systemic NSAIDs (Ibuprofen).
  - **Hypertension:** Explicitly cautions against oral sympathomimetic decongestants (pseudoephedrine/phenylephrine).
  - **Acute Emergencies (ESI 1–2):** Strictly enforces protocols **against unguided oral self-medication** while providing EMS-directed chewable Aspirin protocols for acute coronary events.

### 📍 Live GPS Navigation & Facility Discovery
- **Dynamic Resource Matching:** Queries relevant acute care hospitals for high-risk patients and neighborhood retail pharmacies for low/medium-risk outpatient management.
- **Dual Geolocation Engine:** Uses Google Places API when configured; seamlessly falls back to OpenStreetMap Nominatim and Overpass POI queries with zero API keys required.
- **Real-Time GPS Origin:** Generates direct Google Maps URLs (`https://www.google.com/maps/dir/?api=1&destination=lat,lng&travelmode=driving`) that dynamically resolve the patient's real-time device GPS sensor as the route origin.

### 🧠 Semantic Patient Memory
- Employs `sentence-transformers` (`all-MiniLM-L6-v2`) to produce 384-dimensional dense semantic vectors.
- Matches clinical terminology variations (e.g., semantic parity between "trouble breathing" and "dyspnea") across past visits in Pinecone vector storage with local in-memory fallback.

### 👨‍⚕️ Human-in-the-Loop (HITL) Sign-Off & Glassmorphic UI
- Automatically pauses automated dispatch for high-risk (ESI 1–2) cases until a licensed doctor or triage nurse enters clinical review notes and approves or re-routes the case.
- Displays an interactive **Step-by-Step Multi-Agent Reasoning Panel** rendering real-time execution traces and tool calls across all nodes.

---

## Multi-Agent Architecture & Workflow

The system is built on a deterministic LangGraph `StateGraph` pipeline:

```
                      ┌─────────────────────────┐
                      │    Supervisor Router    │◄───────────┐
                      └────────────┬────────────┘            │
                                   │                         │
                                   ▼                         │
                      ┌─────────────────────────┐            │
                      │  Structured Intake Node ├────────────┤
                      └────────────┬────────────┘            │
                                   │                         │
                                   ▼                         │
                      ┌─────────────────────────┐            │
                      │   Risk & ESI Node       ├────────────┤
                      └────────────┬────────────┘            │
                                   │                         │
                                   ▼                         │
                      ┌─────────────────────────┐            │
                      │   Safety & Compliance   ├────────────┤
                      └────────────┬────────────┘            │
                                   │                         │
                                   ▼                         │
                      ┌─────────────────────────┐            │
                      │ Location Resources Node ├────────────┘
                      └────────────┬────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
          (ESI 1–2 / HIGH Risk)           (ESI 3–5 / Low-Med)
                     ▼                           ▼
            ┌─────────────────┐         ┌─────────────────┐
            │  Human Review   │         │   Finish Node   │
            │  (HITL Pause)   │         └────────┬────────┘
            └────────┬────────┘                  │
                     │                           ▼
            (Clinician Sign-off)                END
                     ▼
                    END
```

### Graph Node Responsibilities:

| Node | Name | Functionality |
| :--- | :--- | :--- |
| **`supervisor`** | Supervisor Router | Evaluates pipeline state, enforces dependency execution order, and routes records to completion. |
| **`intake`** | Structured Intake Agent | Ingests free-text complaints, queries Infermedica/Groq LLM/Regex, extracts symptoms and pertinent negatives, and formats clinical evidence. |
| **`risk`** | Risk & ESI Assessment Node | Performs dual-engine consensus scoring (ESI 1–5), queries Infermedica `/v3/triage`, generates comorbidity-adjusted OTC medication guidance, and flags emergency escalation. |
| **`safety`** | Safety & Compliance Node | Verifies medical non-diagnostic language, enforces safety disclaimers, eliminates certainty claims, and formats clinical response markdown. |
| **`resources`** | Location Resource Agent | Detects patient coordinates, filters relevant medical facilities (general hospitals or pharmacies), calculates haversine distances, and builds direct Google Maps navigation links. |
| **`human_review`**| Clinical Review (HITL) | Suspends graph execution on high-risk cases awaiting attending clinician sign-off, sign-out notes, or direct escalation. |
| **`finish`** | Workflow Finalizer | Compiles final triage state, updates semantic patient history, and delivers output to the clinical interface. |

---

## Tech Stack

- **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain Core](https://github.com/langchain-ai/langchain)
- **Clinical Engine:** [Infermedica API v3](https://developer.infermedica.com/) (`/v3/parse`, `/v3/diagnosis`, `/v3/triage`)
- **LLM Inference:** [Groq Cloud](https://groq.com/) (`llama-3.3-70b-versatile`, `openai/gpt-oss-120b`)
- **Vector Memory & Embeddings:** [Pinecone](https://www.pinecone.io/), [Sentence-Transformers](https://sbert.net/) (`all-MiniLM-L6-v2`, 384 dimensions)
- **Frontend & Visualization:** [Streamlit](https://streamlit.io/) with custom responsive CSS glassmorphism
- **Geocoding & Navigation:** Google Geocoding & Places APIs, OpenStreetMap (Nominatim & Overpass API)
- **Validation & Schemas:** [Pydantic v2](https://docs.pydantic.dev/)

---

## Repository Structure

```
agentic-triage-system/
├── agent/                       # Core Multi-Agent Triage Package
│   ├── __init__.py
│   ├── infermedica_client.py    # Infermedica v3 API client & OTC medication engine
│   ├── location_tools.py        # Geocoding, Google Places / OSM Overpass & Maps URL builder
│   ├── multi_agent.py           # LangGraph StateGraph, ESI consensus logic & semantic memory
│   └── react_agent.py           # Baseline ReAct agent implementation
├── tests/                       # Formal Automated Test Suite
│   ├── __init__.py
│   ├── test_infermedica_integration.py  # Unit & consensus integration tests
│   ├── test_location_tools.py           # Geocoding, POI filters & navigation tests
│   └── test_multi_agent.py              # End-to-end multi-agent triage tests
├── scripts/                     # Operational & Visualization Utilities
│   └── generate_architecture_diagram.py # Graphviz architecture generator
├── .env.example                 # Environment configuration template
├── .gitignore                   # Production git exclusion rules
├── app.py                       # Streamlit web application & clinical dashboard
├── LICENSE                      # MIT License
├── pytest.ini                   # Pytest test runner configuration
├── README.md                    # Project documentation
├── requirements.txt             # Production dependencies
└── sitecustomize.py             # Runtime encoding configuration
```


---

## Setup & Installation

### 1. Prerequisites
- Python **3.10** or higher
- Git installed on your local machine

### 2. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/agentic-triage-system.git
cd agentic-triage-system

# Create and activate virtual environment
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration (.env)

Create a `.env` file in the root directory by copying the template:

```bash
cp .env.example .env
```

Populate the required environment variables:

| Variable | Required? | Description | Default / Source |
| :--- | :--- | :--- | :--- |
| `INFERMEDICA_APP_ID` | Optional* | Infermedica v3 Application ID | Free trial at [developer.infermedica.com](https://developer.infermedica.com) |
| `INFERMEDICA_APP_KEY`| Optional* | Infermedica v3 Application Secret Key | Free trial at [developer.infermedica.com](https://developer.infermedica.com) |
| `INFERMEDICA_MODEL`  | Optional | Clinical language model | `infermedica-en` |
| `GROQ_API_KEY`       | Optional* | Groq Cloud API Key for tool-calling | Free tier at [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL`         | Optional | LLM model identifier | `openai/gpt-oss-120b` or `llama-3.3-70b-versatile` |
| `PINECONE_API_KEY`   | Optional | Pinecone vector store API key | Free tier at [pinecone.io](https://www.pinecone.io) |
| `PINECONE_INDEX_NAME`| Optional | Vector index name | `smart-triage-patient-memory` |
| `PINECONE_DIMENSION` | Optional | Vector dimension matching `all-MiniLM-L6-v2` | `384` |
| `GOOGLE_MAPS_API_KEY`| Optional | Google Places & Geocoding key | Google Cloud Console |

> **Zero-Failure Fallback Guarantee:** All external API keys are optional. If keys are omitted, the system seamlessly falls back to OpenStreetMap Nominatim/Overpass for facility discovery, local `sentence-transformers` for memory, and deterministic regex rules for clinical intake.

---

## Running the Application

Launch the Streamlit clinical interface:

```bash
streamlit run app.py
```

Access the interface in your browser at `http://localhost:8501`.

---

## Clinical Workflow & Usage Examples

### Example 1: Low-Risk Presentation (ESI Level 5)
```
Input: "Patient reports mild headache and sore throat for 1 day. Denies fever, shortness of breath, or chest pain."
Patient Profile: Age 28 · Male · Comorbidities: None
```
- **Graph Path:** `supervisor ➔ intake ➔ risk ➔ safety ➔ resources ➔ finish`
- **Triage Result:** `LOW RISK · ESI Level 5 (Non-Urgent)`
- **Medication Guidance:** 
  - *Paracetamol / Acetaminophen (500–650mg q4-6h)*
  - *Ibuprofen (200–400mg with food)*
  - *Warm Saline Gargle & Lozenges*
- **Resources Rendered:** Nearest retail pharmacies with distance and driving directions.

---

### Example 2: Emergent Presentation & Clinical Review (ESI Level 2 · HITL)
```
Input: "Severe crushing chest pain radiating to left arm and shortness of breath since 2 hours ago."
Patient Profile: Age 58 · Male · Comorbidities: Hypertension
```
- **Graph Path:** `supervisor ➔ intake ➔ risk ➔ safety ➔ resources ➔ human_review (PAUSED)`
- **Triage Result:** `HIGH RISK · ESI Level 2 (Emergent)`
- **Emergency Protocol:**
  - Emergency escalation banner triggered (Call 911 / Immediate ER transfer).
  - Safety protocol: *No unguided oral self-medication*.
  - EMS-directed chewable Aspirin (300mg) protocol for acute coronary syndromes.
- **Resources Rendered:** Filtered multi-specialty acute care hospitals with live navigation links.
- **HITL Gate:** Workflow pauses until the clinician reviews notes and signs off via **Approve & Release** or **Reject & Escalate**.

---

## Semantic Memory & Data Pipeline

```
Patient Profile & Complaint
            │
            ▼
┌───────────────────────────────────────┐
│ SentenceTransformer('all-MiniLM-L6-v2')│
└───────────────────┬───────────────────┘
                    │ 384-dimensional dense vector
                    ▼
┌───────────────────────────────────────┐
│       Pinecone Serverless Index       │  <── Fallback: Local In-Memory Store
└───────────────────┬───────────────────┘
                    │
                    ▼
   Semantic Context Retrieval (Cosine Similarity)
   Matches clinical intent across varying patient terms
```

---

## Testing & Validation

Execute the comprehensive automated test suite across all subsystems:

```bash
# Run all tests via pytest
pytest

# Or run via standard unittest runner
python -m unittest discover tests
```

### Test Suite Coverage:
- `tests/test_infermedica_integration.py`: Validates Infermedica v3 API extraction, evidence generation, triage mapping, and "Escalate, never downgrade" consensus arbitration rules.
- `tests/test_location_tools.py`: Validates distance calculations, healthcare facility relevance filtering, and live GPS destination routing links.
- `tests/test_multi_agent.py`: Validates low-risk outpatient flows, high-risk emergent triggers, comorbidity medication dosing, and HITL gate activation.


---

## Limitations & Future Roadmap

- **EHR Interoperability:** Future releases will add HL7 FHIR (Fast Healthcare Interoperability Resources) native export to sync directly with Epic and Cerner systems.
- **Multimodal Vitals Ingestion:** Planned support for camera-based wound triage analysis and audio cough spectrogram classification.
- **Local LLM Quantization:** Adding native Ollama / vLLM local deployment targets for air-gapped hospital environments.

---

## License & Medical Disclaimer

### License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete terms.

### Medical Disclaimer
```
⚕️ CLINICAL DECISION SUPPORT DISCLAIMER
Smart Triage AI is an experimental decision support system developed to assist licensed 
healthcare intake personnel. It does not provide medical diagnoses, replace professional 
clinical judgment, or establish a doctor-patient relationship. All triage classifications, 
medication suggestions, and escalations must be independently validated by a licensed clinician.
```
