"""Triage Nurse Copilot · Triage assistant prototype.

A decision-support tool for triage nurses in busy emergency departments and clinics that:
- Structures free-text chief complaints into clinical concepts
- Evaluates objective physiological vital signs danger zones
- Suggests an Emergency Severity Index (ESI 1-5) level with transparent rationale
- Enforces the 'Escalate, never downgrade' clinical safety rule
- Provides 1-click nurse acceptance or override with mandatory clinical reason logging
- Generates standardized SBAR physician handoff notes and valid HL7 FHIR R4 Transaction Bundles
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import streamlit as st

# Reload environment variables
load_dotenv(override=True)

from agent.audit_logger import get_audit_logger
from agent.infermedica_client import is_infermedica_configured
from agent.llm import DEFAULT_MODEL, is_llm_available
from agent.memory.store import get_memory_store
from agent.multi_agent import extract_vitals_from_text, get_multi_agent

logger = logging.getLogger("triage-nurse-copilot")

# ---------------------------------------------------------------------------
# Page Configuration & Design System
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Triage Nurse Copilot | Triage Assistant Prototype",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-1: #07111f;
    --bg-2: #0e223d;
    --card: rgba(255, 255, 255, 0.04);
    --card-border: rgba(255, 255, 255, 0.11);
    --text-main: #f1f6fd;
    --text-muted: #94a9c4;
    --accent-blue: #38bdf8;
    --accent-teal: #2dd4bf;
    --accent-amber: #fbbf24;
    --esi-1: #ef4444;
    --esi-2: #f87171;
    --esi-3: #fbbf24;
    --esi-4: #60a5fa;
    --esi-5: #34d399;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-main) !important;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.12), transparent 30%),
        radial-gradient(circle at 90% 15%, rgba(45, 212, 191, 0.10), transparent 25%),
        linear-gradient(160deg, var(--bg-1), var(--bg-2) 55%, #050d18);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1320px !important; padding: 1.2rem 1.6rem 3.5rem !important; }

/* Top Header */
.top-header {
    padding: 0.8rem 0 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1.2rem;
}

.system-title {
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(120deg, #f0f9ff, #7dd3fc 45%, #a7f3d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.system-sub {
    color: var(--text-muted);
    font-size: 0.88rem;
    margin-top: 0.25rem;
}

/* Metric KPI Strip */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.2rem;
}

.kpi-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 0.75rem 0.95rem;
    text-align: center;
}

.kpi-val {
    font-size: 1.45rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #f8fafc;
}

.kpi-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.15rem;
}

/* Glass Card */
.glass-box {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
}

/* Acuity Badges */
.esi-badge-container {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.5rem 0 0.8rem;
}

.badge-esi-lg {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.1rem;
    border-radius: 10px;
    font-size: 1.05rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}

.badge-esi-1 { background: rgba(239, 68, 68, 0.22); border: 1.5px solid rgba(239, 68, 68, 0.85); color: #fee2e2; }
.badge-esi-2 { background: rgba(248, 113, 113, 0.18); border: 1.5px solid rgba(248, 113, 113, 0.75); color: #fecaca; }
.badge-esi-3 { background: rgba(251, 191, 36, 0.18); border: 1.5px solid rgba(251, 191, 36, 0.75); color: #fde68a; }
.badge-esi-4 { background: rgba(96, 165, 250, 0.18); border: 1.5px solid rgba(96, 165, 250, 0.75); color: #bfdbfe; }
.badge-esi-5 { background: rgba(52, 211, 153, 0.18); border: 1.5px solid rgba(52, 211, 153, 0.75); color: #bbf7d0; }

.badge-risk-sm {
    display: inline-flex;
    align-items: center;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 800;
}
.badge-risk-HIGH { background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.5); color: #fecaca; }
.badge-risk-MEDIUM { background: rgba(251, 191, 36, 0.2); border: 1px solid rgba(251, 191, 36, 0.5); color: #fde68a; }
.badge-risk-LOW { background: rgba(52, 211, 153, 0.2); border: 1px solid rgba(52, 211, 153, 0.5); color: #bbf7d0; }

/* Vital Pill Danger */
.vital-warning-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(239, 68, 68, 0.16);
    border: 1px solid rgba(239, 68, 68, 0.6);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: #fca5a5;
    margin: 0.2rem 0.3rem 0.2rem 0;
}

/* SBAR Card */
.sbar-display {
    background: #091322;
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #e2e8f0;
    white-space: pre-wrap;
    max-height: 480px;
    overflow-y: auto;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Preset Clinical Vignettes for Rapid Intake Demonstrations
# ---------------------------------------------------------------------------
PRESET_VIGNETTES = {
    "Select a pre-configured vignette...": None,
    "VIG-01 | STEMI / Resuscitation (ESI 1)": {
        "chief_complaint": "Patient collapsed in triage, crushing chest pressure radiating to left arm and jaw, diaphoretic, SpO2 91%, HR 115, BP 85/55",
        "age": 58,
        "sex": "male",
        "conditions": ["hypertension", "coronary artery disease"],
        "hr": 115,
        "sbp": 85,
        "dbp": 55,
        "rr": 24,
        "spo2": 91,
        "temp": 98.4,
    },
    "VIG-02 | Severe Anaphylaxis (ESI 1)": {
        "chief_complaint": "Acute throat tightness, facial angioedema, audible inspiratory stridor, diffuse urticaria after eating shellfish, SpO2 88%, RR 32",
        "age": 24,
        "sex": "female",
        "conditions": ["asthma"],
        "hr": 132,
        "sbp": 88,
        "dbp": 52,
        "rr": 32,
        "spo2": 88,
        "temp": 98.6,
    },
    "VIG-03 | Acute Ischemic Stroke / TIA (ESI 2)": {
        "chief_complaint": "Sudden onset right facial droop, right arm hemiparesis, and expressive aphasia starting 45 minutes ago, BP 195/105, HR 90",
        "age": 68,
        "sex": "female",
        "conditions": ["hypertension", "atrial fibrillation"],
        "hr": 90,
        "sbp": 195,
        "dbp": 105,
        "rr": 18,
        "spo2": 96,
        "temp": 98.2,
    },
    "VIG-04 | Occult Hypoxia Danger Zone (ESI 1/2)": {
        "chief_complaint": "Patient states only having a mild dry cough and slight fatigue for 2 days, but pulse ox shows severe hypoxia SpO2 89%, RR 28",
        "age": 71,
        "sex": "male",
        "conditions": ["copd", "diabetes"],
        "hr": 108,
        "sbp": 128,
        "dbp": 78,
        "rr": 28,
        "spo2": 89,
        "temp": 99.1,
    },
    "VIG-05 | Acute Appendicitis (ESI 3)": {
        "chief_complaint": "Constant sharp right lower quadrant abdominal pain for 24 hours, anorexia, low-grade fever 100.8F, nausea without vomiting",
        "age": 31,
        "sex": "male",
        "conditions": [],
        "hr": 96,
        "sbp": 122,
        "dbp": 74,
        "rr": 18,
        "spo2": 99,
        "temp": 100.8,
    },
    "VIG-06 | Simple Ankle Inversion Injury (ESI 4)": {
        "chief_complaint": "Twisted right lateral ankle stepping off curb 1 hour ago, moderate edema, no deformity, able to bear some weight, denies head or spine pain",
        "age": 26,
        "sex": "female",
        "conditions": [],
        "hr": 78,
        "sbp": 118,
        "dbp": 72,
        "rr": 14,
        "spo2": 100,
        "temp": 98.6,
    },
    "VIG-07 | Mild Viral Upper Respiratory Infection (ESI 5)": {
        "chief_complaint": "Mild sore throat and clear rhinorrhea for 1 day, no fever, no shortness of breath, denies chest pain, completely normal vitals",
        "age": 22,
        "sex": "female",
        "conditions": [],
        "hr": 72,
        "sbp": 116,
        "dbp": 70,
        "rr": 14,
        "spo2": 99,
        "temp": 98.4,
    },
}


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------
def init_session() -> None:
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = get_multi_agent()
        except Exception as exc:
            logger.exception("Failed to initialize multi-agent copilot: %s", exc)
            st.session_state.agent = None

    if "current_result" not in st.session_state:
        st.session_state.current_result = None

    if "last_vignette_loaded" not in st.session_state:
        st.session_state.last_vignette_loaded = None

    if "intake_input" not in st.session_state:
        st.session_state.intake_input = ""

    if "patient_id" not in st.session_state:
        st.session_state.patient_id = "PT-8421"

    if "patient_age" not in st.session_state:
        st.session_state.patient_age = 45

    if "patient_sex" not in st.session_state:
        st.session_state.patient_sex = "male"

    if "patient_conditions" not in st.session_state:
        st.session_state.patient_conditions = []

    if "vitals_hr" not in st.session_state:
        st.session_state.vitals_hr = 78
    if "vitals_sbp" not in st.session_state:
        st.session_state.vitals_sbp = 120
    if "vitals_dbp" not in st.session_state:
        st.session_state.vitals_dbp = 80
    if "vitals_rr" not in st.session_state:
        st.session_state.vitals_rr = 16
    if "vitals_spo2" not in st.session_state:
        st.session_state.vitals_spo2 = 98
    if "vitals_temp" not in st.session_state:
        st.session_state.vitals_temp = 98.6


init_session()
audit = get_audit_logger()
metrics = audit.get_metrics()

# ---------------------------------------------------------------------------
# Sidebar Diagnostics & Engine Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏥 System Diagnostics")
    llm_active = is_llm_available()
    mode_label = "Hybrid (LLM + Rules Floor)" if llm_active else "Offline Rules-Only (Floor)"
    mode_color = "#34d399" if llm_active else "#fbbf24"
    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:0.75rem 0.9rem;margin-bottom:0.8rem;">
            <div style="font-size:0.7rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.04em;">Engine Mode</div>
            <div style="font-size:0.95rem;font-weight:800;color:{mode_color};margin-top:0.2rem;">● {mode_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mem_store = get_memory_store()
    mem_backend_name = type(mem_store).__name__
    active_model = DEFAULT_MODEL if llm_active else "N/A (Rules Mode)"

    st.markdown("**Engine Specifications:**")
    st.markdown(f"- **Active LLM:** `{active_model}`")
    st.markdown(f"- **Memory Store:** `{mem_backend_name}`")
    st.markdown("- **Embeddings:** `all-MiniLM-L6-v2` (384-dim)")
    st.markdown("- **Supervisor:** Deterministic State Graph")

    st.markdown("---")
    st.markdown("### 🛡️ Safety Invariant")
    st.info(
        "**Floor Invariant:** `Final Urgency ≥ Rule Floor`\n\n"
        "LLM context may escalate severity for subtle presentations, but can **never** de-escalate below the deterministic red-flag safety check."
    )

    st.markdown("### 📋 Clinical Gate")
    st.caption(
        "All AI suggestions require mandatory verification by a licensed triage nurse. "
        "Nurse overrides are permanently recorded in the audit trail with clinical rationale."
    )


# ---------------------------------------------------------------------------
# Persistent Clinical Disclaimer Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 0.9rem; font-size: 0.82rem; color: #fef3c7; line-height: 1.5;">
        ⚠️ <strong>EDUCATIONAL & RESEARCH PROTOTYPE ONLY:</strong> This software is an engineering portfolio demonstration and <strong>not a certified medical device</strong> (FDA SaMD / CE-IVD). It is not cleared for autonomous patient diagnosis, prescription, or clinical triage. All decisions require licensed clinician sign-off. Never input real Protected Health Information (PHI).
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Top Header & Live Concordance Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="top-header">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
            <h1 class="system-title">Triage Nurse Copilot</h1>
            <div class="system-sub">
                Emergency Intake Decision Support · Physiological Vitals Triaging · SBAR Handoff · Concordance & Override Audit
            </div>
        </div>
        <div style="text-align:right;">
            <span style="display:inline-block;padding:0.25rem 0.65rem;border-radius:6px;background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.3);font-size:0.75rem;font-weight:700;color:#38bdf8;">
                RULE: Escalate, Never Downgrade
            </span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Render Live Audit Metrics Strip
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
with col_kpi1:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-val">{metrics['total_encounters']}</div>
        <div class="kpi-label">Logged Encounters</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_kpi2:
    concordance_color = "#34d399" if metrics["concordance_rate_pct"] >= 80 else "#fbbf24"
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-val" style="color:{concordance_color}">{metrics['concordance_rate_pct']}%</div>
        <div class="kpi-label">AI-Nurse Concordance</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_kpi3:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-val">{metrics['override_count']} <span style="font-size:0.8rem;color:#94a3b8">({metrics['override_rate_pct']}%)</span></div>
        <div class="kpi-label">Nurse Overrides</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_kpi4:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-val"><span style="color:#f87171">+{metrics['upgraded_count']}</span> / <span style="color:#60a5fa">-{metrics['downgraded_count']}</span></div>
        <div class="kpi-label">Upgrades / Downgrades</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_kpi5:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-val" style="color:#38bdf8">&lt; 5.0%</div>
        <div class="kpi-label">Target UTR (ACEP Benchmark)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Preset Clinical Vignette Selector (1-Click Demo)
# ---------------------------------------------------------------------------
selected_vignette_key = st.selectbox(
    "Load Clinical Intake Vignette (Pre-configured standardized test cases):",
    list(PRESET_VIGNETTES.keys()),
    index=0,
)

if selected_vignette_key and selected_vignette_key != "Select a pre-configured vignette...":
    if st.session_state.last_vignette_loaded != selected_vignette_key:
        vig = PRESET_VIGNETTES[selected_vignette_key]
        st.session_state.intake_input = vig["chief_complaint"]
        st.session_state.patient_age = vig["age"]
        st.session_state.patient_sex = vig["sex"]
        st.session_state.patient_conditions = vig["conditions"]
        st.session_state.vitals_hr = vig["hr"]
        st.session_state.vitals_sbp = vig["sbp"]
        st.session_state.vitals_dbp = vig["dbp"]
        st.session_state.vitals_rr = vig["rr"]
        st.session_state.vitals_spo2 = vig["spo2"]
        st.session_state.vitals_temp = vig["temp"]
        st.session_state.last_vignette_loaded = selected_vignette_key
        st.session_state.current_result = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main 2-Column Clinical Layout
# ---------------------------------------------------------------------------
tab_intake, tab_audit = st.tabs(["Patient Intake & Decision Support", "Audit History & Override Analytics"])

with tab_intake:
    col_left, col_right = st.columns([1, 1.15], gap="large")

    with col_left:
        st.markdown("### Patient Intake & Physiological Vitals")

        with st.expander("Patient Demographics & Chronic History", expanded=True):
            col_d1, col_d2, col_d3 = st.columns([1.2, 1, 1])
            with col_d1:
                pt_id = st.text_input("Patient ID / MRN", value=st.session_state.patient_id)
                st.session_state.patient_id = pt_id
            with col_d2:
                pt_age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.patient_age)
                st.session_state.patient_age = pt_age
            with col_d3:
                pt_sex = st.selectbox(
                    "Sex",
                    ["male", "female"],
                    index=0 if st.session_state.patient_sex == "male" else 1,
                )
                st.session_state.patient_sex = pt_sex

            all_conditions = [
                "hypertension",
                "coronary artery disease",
                "diabetes",
                "asthma",
                "copd",
                "atrial fibrillation",
                "food allergy",
                "immunocompromised",
                "chronic kidney disease",
            ]
            pt_conditions = st.multiselect(
                "Known Comorbidities",
                options=all_conditions,
                default=st.session_state.patient_conditions,
            )
            st.session_state.patient_conditions = pt_conditions

        st.markdown("**Chief Complaint & Intake Presentation (Free-text):**")
        chief_complaint = st.text_area(
            "Chief Complaint",
            value=st.session_state.intake_input,
            height=110,
            placeholder="e.g., Severe crushing retrosternal chest pain radiating to left jaw, diaphoresis for 45 minutes, SpO2 91%...",
            label_visibility="collapsed",
        )
        st.session_state.intake_input = chief_complaint

        st.markdown("**Physiological Vital Signs Measurement:**")
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            hr_val = st.number_input("Heart Rate (bpm)", min_value=20, max_value=260, value=int(st.session_state.vitals_hr))
            st.session_state.vitals_hr = hr_val
            rr_val = st.number_input("Resp Rate (/min)", min_value=4, max_value=60, value=int(st.session_state.vitals_rr))
            st.session_state.vitals_rr = rr_val
        with v_col2:
            sbp_val = st.number_input("BP Systolic (mmHg)", min_value=40, max_value=300, value=int(st.session_state.vitals_sbp))
            st.session_state.vitals_sbp = sbp_val
            dbp_val = st.number_input("BP Diastolic (mmHg)", min_value=20, max_value=180, value=int(st.session_state.vitals_dbp))
            st.session_state.vitals_dbp = dbp_val
        with v_col3:
            spo2_val = st.number_input("SpO2 (%)", min_value=40, max_value=100, value=int(st.session_state.vitals_spo2))
            st.session_state.vitals_spo2 = spo2_val
            temp_val = st.number_input("Temp (°F)", min_value=85.0, max_value=110.0, value=float(st.session_state.vitals_temp), step=0.1)
            st.session_state.vitals_temp = temp_val

        # Display immediate vital sign danger alerts if out of physiological safety bounds
        vitals_dict = {
            "hr": hr_val,
            "sbp": sbp_val,
            "dbp": dbp_val,
            "rr": rr_val,
            "spo2": spo2_val,
            "temp": temp_val,
        }
        warning_pills = []
        if spo2_val < 90:
            warning_pills.append(f"CRITICAL HYPOXIA: SpO2 {spo2_val}% (ESI 1 Danger Trigger)")
        elif spo2_val <= 92:
            warning_pills.append(f"HYPOXIA: SpO2 {spo2_val}% (ESI 2 Danger Trigger)")
        if hr_val < 40 or hr_val > 140:
            warning_pills.append(f"EXTREME HR: {hr_val} bpm (ESI 2 Danger Trigger)")
        if rr_val < 8 or rr_val > 30:
            warning_pills.append(f"EXTREME RR: {rr_val}/min (ESI 2 Danger Trigger)")
        if sbp_val < 90:
            warning_pills.append(f"HYPOTENSION: SBP {sbp_val} mmHg (ESI 2 Danger Trigger)")

        if warning_pills:
            st.markdown(
                "".join(f'<span class="vital-warning-pill">⚠️ {pill}</span>' for pill in warning_pills),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("Evaluate & Suggest ESI Acuity", type="primary", use_container_width=True)

        if analyze_btn:
            if not chief_complaint.strip():
                st.warning("Please enter a chief complaint or load a preset vignette.")
            else:
                with st.spinner("Analyzing complaint, extracting symptoms, and arbitrating ESI consensus..."):
                    start_t = time.perf_counter()
                    res = st.session_state.agent.process(
                        user_input=chief_complaint,
                        user_id=pt_id,
                        age=pt_age,
                        sex=pt_sex,
                        conditions=pt_conditions,
                        vitals=vitals_dict,
                    )
                    latency = time.perf_counter() - start_t
                    res["latency_seconds"] = latency
                    st.session_state.current_result = res

    # -----------------------------------------------------------------------
    # Right Column: Clinical Decision Support & SBAR Handoff
    # -----------------------------------------------------------------------
    with col_right:
        res = st.session_state.current_result
        if not res:
            st.info("👈 Enter patient intake details or select a preset vignette, then click **Evaluate & Suggest ESI Acuity**.")
        else:
            esi_level = res.get("esi_level", 5)
            risk_level = res.get("risk_level", "LOW")
            esi_titles = {
                1: "ESI 1 · Resuscitation (Immediate Life Threat)",
                2: "ESI 2 · Emergent (High Risk / Should Not Wait)",
                3: "ESI 3 · Urgent (Multiple Diagnostic Resources)",
                4: "ESI 4 · Less Urgent (Single Clinical Resource)",
                5: "ESI 5 · Non-Urgent (Routine Outpatient)",
            }
            esi_name = esi_titles.get(esi_level, f"ESI {esi_level}")

            st.markdown("### AI Triage Decision Support")

            # Acuity Header Card
            st.markdown(
                f"""
            <div class="glass-box" style="margin-bottom:0.9rem;">
                <div style="font-size:0.75rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Suggested Acuity Classification</div>
                <div class="esi-badge-container">
                    <span class="badge-esi-lg badge-esi-{esi_level}">{esi_name}</span>
                    <span class="badge-risk-sm badge-risk-{risk_level}">{risk_level} RISK</span>
                    <span style="font-size:0.75rem;color:#94a3b8;margin-left:auto;">Engine Latency: {res.get('latency_seconds', 0.0):.2f}s</span>
                </div>
                <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.5;">
                    <strong>Clinical Rationale:</strong> {res.get('esi_rationale', 'Standard protocol')}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Why this level? Accordion Card
            with st.expander("Why This Level? Clinical Breakdown", expanded=True):
                triggers = res.get("vital_triggers", [])
                if triggers:
                    st.markdown("**Objective Vital Sign Triggers:**")
                    for trig in triggers:
                        st.markdown(f"- 🔴 `{trig}` (Elevated acuity per ESI Step 2/3 Danger Zone Protocol)")

                symps = res.get("symptoms", [])
                negs = res.get("pertinent_negatives", [])
                red_flags = res.get("red_flags", [])

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Identified Symptoms:** {', '.join(symps) if symps else 'None formalized'}")
                    st.markdown(f"**Pertinent Negatives:** {', '.join(negs) if negs else 'None'}")
                with c2:
                    st.markdown(f"**Red Flags Present:** {', '.join(red_flags) if red_flags else 'None'}")
                    st.markdown(f"**Intake Extraction:** `{res.get('intake_method', 'regex_fallback')}`")

                # Relevant patient history / memory used
                comorbidities = st.session_state.patient_conditions
                if comorbidities:
                    st.markdown(f"**Relevant Patient History / Memory:** `{', '.join(comorbidities)}` (Factored into clinical risk assessment)")

                # Consensus arbitration note
                if res.get("infermedica_triage"):
                    inf = res["infermedica_triage"]
                    st.info(
                        f"Consensus Arbitration: Infermedica reported `{inf.get('triage_level')}`. "
                        f"Combined under 'Escalate, never downgrade' rule -> Final ESI {esi_level}."
                    )

            # ---------------------------------------------------------------
            # Licensed Clinician Sign-off & Override Module
            # ---------------------------------------------------------------
            st.markdown("### Clinician Sign-off & Override Module")
            with st.container():
                st.markdown(
                    """
                <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.5rem;">
                    Every AI suggestion requires licensed triage nurse verification. Confirm or override the suggested level:
                </div>
                """,
                    unsafe_allow_html=True,
                )

                nurse_action = st.radio(
                    "Nurse Decision",
                    ["Accept & Approve AI Suggestion", "Override Acuity Level"],
                    horizontal=True,
                    label_visibility="collapsed",
                )

                override_esi = None
                override_reason = ""
                nurse_notes = ""

                if nurse_action == "Override Acuity Level":
                    c_ov1, c_ov2 = st.columns([1, 2])
                    with c_ov1:
                        override_esi = st.selectbox(
                            "New Nurse Confirmed ESI",
                            [1, 2, 3, 4, 5],
                            index=max(0, min(4, esi_level - 1)),
                        )
                    with c_ov2:
                        override_reason = st.selectbox(
                            "Mandatory Clinical Override Reason",
                            [
                                "Bedside ECG / Diagnostic finding",
                                "Vital signs improved / normalized post-treatment",
                                "Atypical clinical presentation / Gestalt",
                                "Pediatric / High-risk age vulnerability",
                                "Severe comorbidities not captured",
                                "Clinical judgment / Departmental protocol",
                            ],
                        )
                    nurse_notes = st.text_input("Nurse Clinical Note", placeholder="e.g., Bedside 12-lead ECG showed 2mm STE in V1-V4...")
                else:
                    nurse_notes = st.text_input("Optional Nurse Notes", placeholder="e.g., Verified bedside vitals and airway patency...")

                sign_btn = st.button("Sign & Finalize Triage Encounter", type="primary", use_container_width=True)

                if sign_btn:
                    # Finalize through multi-agent graph with human review node
                    decision_str = "override" if nurse_action == "Override Acuity Level" else "approved"
                    res_final = st.session_state.agent.process(
                        pending_state=res.get("pending_state"),
                        approval_decision=decision_str,
                        nurse_override_esi=override_esi if decision_str == "override" else esi_level,
                        nurse_override_reason=override_reason if decision_str == "override" else "",
                        nurse_notes=nurse_notes,
                    )
                    st.session_state.current_result = res_final
                    st.success("Triage encounter signed and logged to clinical audit trail.")
                    time.sleep(0.4)
                    st.rerun()

            # ---------------------------------------------------------------
            # Structured SBAR Physician Handoff Note
            # ---------------------------------------------------------------
            st.markdown("### Physician SBAR Handoff Note")
            sbar_content = res.get("sbar_note", "SBAR note pending generation.")
            st.markdown(f'<div class="sbar-display">{sbar_content}</div>', unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # HL7 FHIR R4 Transaction Bundle Export
            # ---------------------------------------------------------------
            with st.expander("HL7 FHIR R4 Transaction Bundle (EHR Interoperability Export)"):
                fhir_bundle = res.get("fhir_bundle", {})
                st.json(fhir_bundle)
                fhir_str = json.dumps(fhir_bundle, indent=2)
                st.download_button(
                    label="Download HL7 FHIR R4 JSON Bundle",
                    data=fhir_str,
                    file_name=f"fhir_triage_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # ---------------------------------------------------------------
            # Multi-Agent Trace & Engine Diagnostics
            # ---------------------------------------------------------------
            with st.expander("🔍 Multi-Agent Execution Trace & Engine Diagnostics", expanded=False):
                tc1, tc2, tc3, tc4 = st.columns(4)
                with tc1:
                    is_hybrid = is_llm_available()
                    st.metric("Engine Mode", "Hybrid (LLM+Rules)" if is_hybrid else "Rules Safety Floor")
                with tc2:
                    st.metric("Intake Method", res.get("intake_method", "N/A"))
                with tc3:
                    lat_ms = res.get("latency_ms", res.get("latency_seconds", 0.0) * 1000.0)
                    st.metric("Total Latency", f"{lat_ms:.1f} ms")
                with tc4:
                    has_disagree = bool(res.get("vital_triggers") or (res.get("risk_assessment", {}).get("disagreement", False) if isinstance(res.get("risk_assessment"), dict) else False))
                    st.metric("Safety Override", "ACTIVE" if has_disagree else "Concordant")

                red_flags_found = res.get("red_flags", [])
                if red_flags_found:
                    st.markdown(f"**Identified Red Flags ({len(red_flags_found)}):** `{'`, `'.join(red_flags_found)}`")
                else:
                    st.markdown("**Identified Red Flags:** None detected")

                vital_trigs = res.get("vital_triggers", [])
                if vital_trigs:
                    st.markdown(f"**Vital Danger Triggers ({len(vital_trigs)}):** `{'`, `'.join(vital_trigs)}`")

                st.markdown("**Workflow Node Execution Details:**")
                wf_trace = res.get("workflow_trace", [])
                if wf_trace:
                    trace_table = []
                    for step in wf_trace:
                        dur = step.get("duration_ms")
                        trace_table.append({
                            "Node": step.get("node", "N/A"),
                            "Duration (ms)": f"{dur:.1f}" if dur is not None else "N/A",
                            "Tools Used": ", ".join(step.get("tools", [])) or "None",
                            "Summary": step.get("summary", ""),
                        })
                    st.dataframe(trace_table, use_container_width=True)
                else:
                    st.caption("No granular node trace recorded for this encounter.")


# ---------------------------------------------------------------------------
# Tab 2: Encounter Audit Trail & Override Analytics
# ---------------------------------------------------------------------------
with tab_audit:
    st.markdown("### Clinical Encounter Audit Trail & Override Concordance")
    st.markdown(
        """
    Every encounter processed through the nurse copilot is permanently logged with:
    - AI Suggested ESI vs Nurse Confirmed ESI
    - Override status, direction (upgraded/downgraded), and mandatory clinical rationale
    - Full reproducibility for departmental QA, M&M conferences, and model auditing
    """
    )

    metrics_current = audit.get_metrics()
    logs = audit.get_recent_encounters(limit=50)

    if not logs:
        st.info("No clinical triage encounters recorded in the audit log yet. Run patient evaluations in Tab 1 to generate audit records.")
    else:
        # Display encounter table
        table_rows = []
        for e in logs:
            status_badge = "Concordant" if not e.get("is_overridden") else f"OVERRIDE ({e.get('override_direction', '')})"
            table_rows.append(
                {
                    "Encounter ID": e.get("encounter_id", "ENC"),
                    "Timestamp (UTC)": e.get("timestamp", "")[:19].replace("T", " "),
                    "Patient": f"{e.get('patient_id')} ({e.get('age')}yo {e.get('sex')})",
                    "Chief Complaint": (e.get("chief_complaint") or "")[:45] + "...",
                    "AI ESI": e.get("ai_esi"),
                    "Nurse ESI": e.get("nurse_esi"),
                    "Status": status_badge,
                    "Override Reason": e.get("override_reason") or "N/A",
                }
            )

        st.dataframe(table_rows, use_container_width=True)

        # Override reason breakdown
        if metrics_current.get("override_reasons"):
            st.markdown("#### Clinical Override Reasons Breakdown")
            c_reasons = metrics_current["override_reasons"]
            for reason, count in c_reasons.items():
                st.markdown(f"- **{reason}**: `{count}` encounter(s)")

        col_clr1, col_clr2 = st.columns([1, 4])
        with col_clr1:
            if st.button("Clear Audit Log File (Testing Only)", type="secondary"):
                audit.clear()
                st.success("Audit log cleared.")
                time.sleep(0.3)
                st.rerun()
