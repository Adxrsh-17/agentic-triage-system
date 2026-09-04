import os
from typing import Any, Dict, List

from dotenv import load_dotenv

# Ensure environment variables are dynamically reloaded on every rerun
load_dotenv(override=True)

import streamlit as st

from agent.infermedica_client import is_infermedica_configured
from agent.location_tools import geocode_address, get_current_location

st.set_page_config(
    page_title="Clinic Triage Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-1: #07111f;
    --bg-2: #0e223d;
    --card: rgba(255, 255, 255, 0.04);
    --card-strong: rgba(255, 255, 255, 0.08);
    --card-border: rgba(255, 255, 255, 0.11);
    --text-main: #f1f6fd;
    --text-muted: #94a9c4;
    --accent-blue: #38bdf8;
    --accent-teal: #2dd4bf;
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
        radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.14), transparent 30%),
        radial-gradient(circle at 90% 15%, rgba(45, 212, 191, 0.12), transparent 26%),
        linear-gradient(160deg, var(--bg-1), var(--bg-2) 55%, #050d18);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1220px !important; padding: 1.8rem 2rem 4rem !important; }

/* Hero Header */
.hero {
    padding: 1.5rem 0 1rem;
    margin-bottom: 1.2rem;
    text-align: center;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent-blue);
    margin-bottom: 0.7rem;
}

.hero h1 {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
    background: linear-gradient(120deg, #f0f9ff, #7dd3fc 45%, #a7f3d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: var(--text-muted);
    font-size: 1.02rem;
    max-width: 720px;
    margin: 0 auto;
}

/* Glass Cards */
.glass-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.3rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.2);
    margin-bottom: 1.1rem;
}

/* Chat Bubbles */
.msg-user, .msg-ai {
    border-radius: 16px;
    padding: 1.1rem 1.25rem;
    margin: 0.8rem 0;
    border: 1px solid var(--card-border);
}

.msg-user {
    margin-left: 12%;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(14, 165, 233, 0.05));
    border-color: rgba(56, 189, 248, 0.28);
}

.msg-ai {
    margin-right: 5%;
    background: rgba(255, 255, 255, 0.035);
}

.msg-user::before {
    content: "INTAKE STAFF NOTE";
    display: block;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
    color: var(--accent-blue);
}

.msg-ai::before {
    content: "CLINIC TRIAGE COPILOT";
    display: block;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.55rem;
    color: var(--accent-teal);
}

/* Badges */
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.85rem;
}

.badge-risk {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 800;
}

.badge-HIGH { background: rgba(248, 113, 113, 0.18); border: 1px solid rgba(248, 113, 113, 0.55); color: #fecaca; }
.badge-MEDIUM { background: rgba(251, 191, 36, 0.18); border: 1px solid rgba(251, 191, 36, 0.55); color: #fde68a; }
.badge-LOW { background: rgba(52, 211, 153, 0.18); border: 1px solid rgba(52, 211, 153, 0.55); color: #bbf7d0; }

.badge-esi {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 800;
}

.badge-esi-1 { background: rgba(239, 68, 68, 0.28); border: 1px solid rgba(239, 68, 68, 0.75); color: #fee2e2; }
.badge-esi-2 { background: rgba(248, 113, 113, 0.2); border: 1px solid rgba(248, 113, 113, 0.6); color: #fecaca; }
.badge-esi-3 { background: rgba(251, 191, 36, 0.2); border: 1px solid rgba(251, 191, 36, 0.6); color: #fde68a; }
.badge-esi-4 { background: rgba(96, 165, 250, 0.2); border: 1px solid rgba(96, 165, 250, 0.6); color: #bfdbfe; }
.badge-esi-5 { background: rgba(52, 211, 153, 0.2); border: 1px solid rgba(52, 211, 153, 0.6); color: #bbf7d0; }

/* Banners */
.emergency-banner, .review-banner, .disclaimer-banner {
    border-radius: 14px;
    padding: 0.95rem 1.15rem;
    margin: 0.85rem 0;
    border: 1px solid;
    font-size: 0.92rem;
}

.emergency-banner {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.6), rgba(153, 27, 27, 0.25));
    border-color: rgba(248, 113, 113, 0.55);
}

.review-banner {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.55), rgba(146, 64, 14, 0.22));
    border-color: rgba(251, 191, 36, 0.55);
}

.disclaimer-banner {
    background: rgba(15, 23, 42, 0.6);
    border-color: rgba(148, 163, 184, 0.2);
    color: #94a3b8;
    font-size: 0.82rem;
    text-align: center;
}

/* Facility Cards */
.facility-container {
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.facility-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.2s ease;
}

.facility-card:hover {
    background: rgba(56, 189, 248, 0.07);
    border-color: rgba(56, 189, 248, 0.4);
}

.facility-info {
    flex: 1;
}

.facility-name {
    font-size: 0.92rem;
    font-weight: 700;
    color: #f0f9ff;
    margin-bottom: 0.2rem;
}

.facility-meta {
    font-size: 0.8rem;
    color: #94a3b8;
}

.directions-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    background: linear-gradient(135deg, #0284c7, #0d9488);
    color: white !important;
    text-decoration: none !important;
    font-size: 0.78rem;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
}

.directions-btn:hover {
    opacity: 0.92;
}

/* Medication Cards */
.med-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(45, 212, 191, 0.25);
    border-left: 4px solid var(--accent-teal);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.med-card:hover {
    background: rgba(45, 212, 191, 0.06);
    border-color: rgba(45, 212, 191, 0.45);
}

.med-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}

.med-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
}

.med-category {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
    border-radius: 6px;
    background: rgba(45, 212, 191, 0.12);
    border: 1px solid rgba(45, 212, 191, 0.3);
    color: #5eead4;
}

.med-detail {
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.45;
    margin-bottom: 0.25rem;
}

.med-precaution {
    font-size: 0.8rem;
    color: #fca5a5;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.22);
    border-radius: 6px;
    padding: 0.35rem 0.6rem;
    margin-top: 0.4rem;
}

/* Multi-Agent Reasoning Panel */
.reasoning-panel {
    background: rgba(10, 20, 35, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 14px;
    padding: 1.1rem;
    margin-top: 1rem;
}

.pipeline-stepper {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.pipeline-step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.6rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.75rem;
    font-weight: 700;
    color: #cbd5e1;
}

.pipeline-arrow {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: bold;
}

.reasoning-node-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid var(--accent-blue);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.reasoning-node-card:hover {
    background: rgba(255, 255, 255, 0.045);
    border-color: rgba(56, 189, 248, 0.35);
}

.reasoning-node-supervisor { border-left-color: #a78bfa; }
.reasoning-node-intake { border-left-color: #38bdf8; }
.reasoning-node-risk { border-left-color: #fbbf24; }
.reasoning-node-safety { border-left-color: #34d399; }
.reasoning-node-resources { border-left-color: #38bdf8; }
.reasoning-node-human_review { border-left-color: #f87171; }
.reasoning-node-finish { border-left-color: #6ee7b7; }

.node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}

.node-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #e0f2fe;
    letter-spacing: 0.02em;
}

.node-step-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.07);
    color: #93c5fd;
}

.node-summary {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.45;
    margin-bottom: 0.45rem;
}

.tool-chip {
    display: inline-block;
    margin: 0.12rem 0.22rem 0.12rem 0;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.09);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #bae6fd;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(5, 12, 22, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.sidebar-label {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #718aa8;
    margin-bottom: 0.5rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    display: inline-block;
    margin-right: 6px;
    background: #34d399;
    box-shadow: 0 0 12px rgba(52, 211, 153, 0.8);
}

.section-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin: 1.1rem 0;
}

.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background: rgba(255, 255, 255, 0.04) !important;
    color: var(--text-main) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 12px !important;
}

.stButton > button {
    border-radius: 12px !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    background: linear-gradient(120deg, #0284c7, #0d9488) !important;
}

.stButton > button:hover {
    box-shadow: 0 10px 25px rgba(13, 148, 136, 0.3) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    infermedica_active = is_infermedica_configured()

    try:
        from agent.multi_agent import get_multi_agent

        st.session_state.agent = get_multi_agent()
    except Exception as exc:
        st.session_state.agent = None
        st.session_state.agent_status = f"Agent initialization error: {exc}"

    status_parts = []
    if infermedica_active:
        status_parts.append("Infermedica Clinical Engine v3")
    if groq_key:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        status_parts.append(f"Groq LLM ({model_name})")

    if status_parts:
        st.session_state.agent_backend = "multi-agent-active"
        st.session_state.agent_status = f"Connected: LangGraph Multi-Agent ({' · '.join(status_parts)})"
    else:
        st.session_state.agent_backend = "multi-agent-offline"
        st.session_state.agent_status = "Connected: LangGraph Multi-Agent (Deterministic Regex Fallback)"

    if "history" not in st.session_state:
        st.session_state.history = []
    if "profile" not in st.session_state:
        st.session_state.profile = {"age": 25, "sex": "male", "conditions": []}
    if "patient_location" not in st.session_state:
        auto_loc = get_current_location()
        if auto_loc:
            st.session_state.patient_location = auto_loc["address"]
            st.session_state.location_lat = auto_loc["lat"]
            st.session_state.location_lng = auto_loc["lng"]
            st.session_state.location_city = auto_loc["city"]
        else:
            st.session_state.patient_location = "Coimbatore, Tamil Nadu, India"
            st.session_state.location_lat = 11.0018
            st.session_state.location_lng = 76.9628
            st.session_state.location_city = "Coimbatore"
    if "current_risk" not in st.session_state:
        st.session_state.current_risk = None
    if "current_esi" not in st.session_state:
        st.session_state.current_esi = None
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    if "pending_review" not in st.session_state:
        st.session_state.pending_review = None


def risk_badge(level: str) -> str:
    icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    return f'<span class="badge-risk badge-{level}">{icons.get(level, "⚪")} {level} RISK</span>'


def esi_badge(level: int) -> str:
    styles = {
        1: ("badge-esi-1", "🚨", "ESI 1 · Resuscitation"),
        2: ("badge-esi-2", "🔴", "ESI 2 · Emergent"),
        3: ("badge-esi-3", "🟡", "ESI 3 · Urgent"),
        4: ("badge-esi-4", "🔵", "ESI 4 · Less Urgent"),
        5: ("badge-esi-5", "🟢", "ESI 5 · Non-Urgent"),
    }
    cls, icon, title = styles.get(level, ("badge-esi-5", "⚪", f"ESI {level}"))
    return f'<span class="badge-esi {cls}">{icon} {title}</span>'


def render_reasoning_panel(trace: List[Dict[str, Any]]) -> None:
    node_metadata = {
        "supervisor": {"icon": "🧭", "title": "Supervisor Router"},
        "intake": {"icon": "📋", "title": "Structured Intake Agent"},
        "risk": {"icon": "⚠️", "title": "ESI & Risk Assessment Agent"},
        "safety": {"icon": "🛡️", "title": "Safety & Compliance Agent"},
        "resources": {"icon": "📍", "title": "Location Resource Agent"},
        "human_review": {"icon": "👨‍⚕️", "title": "Clinical Review Node (HITL)"},
        "finish": {"icon": "🏁", "title": "Workflow Finalizer"},
    }

    st.markdown('<div class="reasoning-panel">', unsafe_allow_html=True)

    # Stepper pipeline header
    stepper_html = """
<div class="pipeline-stepper">
    <span class="pipeline-step-badge">🧭 Supervisor</span>
    <span class="pipeline-arrow">➔</span>
    <span class="pipeline-step-badge">📋 Intake</span>
    <span class="pipeline-arrow">➔</span>
    <span class="pipeline-step-badge">⚠️ Risk & ESI</span>
    <span class="pipeline-arrow">➔</span>
    <span class="pipeline-step-badge">🛡️ Safety</span>
    <span class="pipeline-arrow">➔</span>
    <span class="pipeline-step-badge">📍 Resources</span>
    <span class="pipeline-arrow">➔</span>
    <span class="pipeline-step-badge">👨‍⚕️ Review / 🏁 Finish</span>
</div>
"""
    st.markdown(stepper_html, unsafe_allow_html=True)

    for index, step in enumerate(trace, start=1):
        node_name = step.get("node", "node")
        meta = node_metadata.get(node_name, {"icon": "⚙️", "title": node_name.capitalize()})
        tools = step.get("tools", [])
        chips = "".join(f"<span class='tool-chip'>{tool}</span>" for tool in tools)

        st.markdown(
            f"""
<div class="reasoning-node-card reasoning-node-{node_name}">
    <div class="node-header">
        <span class="node-title">{meta['icon']} {meta['title']}</span>
        <span class="node-step-tag">STEP {index}</span>
    </div>
    <div class="node-summary">{step.get("summary", "")}</div>
    {f"<div style='margin-top:0.35rem;'>{chips}</div>" if chips else ""}
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_medications_cards(medications: List[Dict[str, Any]]) -> None:
    if not medications:
        return

    st.markdown('<div class="med-container" style="margin-top: 0.9rem;">', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem;font-weight:800;letter-spacing:0.04em;color:#5eead4;margin-bottom:0.6rem;'>💊 CLINICAL OTC & SUPPORTIVE MEDICATION GUIDANCE</p>",
        unsafe_allow_html=True,
    )

    for med in medications:
        name = med.get("name", "Medication")
        category = med.get("category", "Symptomatic Care")
        dosage = med.get("dosage", "")
        purpose = med.get("purpose", "")
        precautions = med.get("precautions", "")

        precaution_html = (
            f'<div class="med-precaution">⚠️ <strong>Safety & Precautions:</strong> {precautions}</div>'
            if precautions
            else ""
        )

        st.markdown(
            f"""
<div class="med-card">
    <div class="med-header">
        <span class="med-name">💊 {name}</span>
        <span class="med-category">{category}</span>
    </div>
    <div class="med-detail"><strong>Indication:</strong> {purpose}</div>
    {f'<div class="med-detail"><strong>Dosage & Frequency:</strong> {dosage}</div>' if dosage else ''}
    {precaution_html}
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<p style='font-size:0.75rem;color:#94a3b8;font-style:italic;margin-top:0.35rem;margin-bottom:0.75rem;'>
    ⚕️ <em>Clinical intake guidance only. Verify patient allergies, organ function, and contraindications before recommending.</em>
</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_facilities_cards(facilities: List[Dict[str, Any]]) -> None:
    if not facilities:
        return

    st.markdown('<div class="facility-container">', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem;font-weight:800;letter-spacing:0.04em;color:#7dd3fc;margin-bottom:0.6rem;'>📍 NEARBY HEALTHCARE FACILITIES & LIVE NAVIGATION</p>",
        unsafe_allow_html=True,
    )

    for fac in facilities:
        name = fac.get("name", "Healthcare Facility")
        dist = f"{fac.get('distance_km', 0)} km away"
        addr = fac.get("address", "")
        ptype = fac.get("place_type", "facility").lower()
        if ptype == "hospital":
            type_label = "🏥 Hospital"
            badge_color = "#f87171"
            badge_bg = "rgba(239, 68, 68, 0.15)"
        else:
            type_label = "💊 Pharmacy"
            badge_color = "#34d399"
            badge_bg = "rgba(52, 211, 153, 0.15)"

        dir_url = fac.get("directions_url", "#")

        st.markdown(
            f"""
<div class="facility-card">
    <div class="facility-info">
        <div class="facility-name">
            <span style="display:inline-block;padding:0.2rem 0.55rem;border-radius:6px;font-size:0.72rem;font-weight:800;background:{badge_bg};color:{badge_color};border:1px solid {badge_color}40;margin-right:0.4rem;">{type_label}</span>
            {name}
        </div>
        <div class="facility-meta">
            <strong style="color:#e0f2fe;">{dist}</strong> {f'· {addr}' if addr else ''}
        </div>
    </div>
    <div>
        <a href="{dir_url}" target="_blank" rel="noopener noreferrer" class="directions-btn" title="Route from patient coordinates to {name}">
            🗺️ Get Directions
        </a>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_message(message: Dict[str, Any], is_latest: bool = False) -> None:
    role = message["role"]
    content = message["content"]
    meta = message.get("meta", {})
    if role == "user":
        st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="msg-ai">', unsafe_allow_html=True)

    risk_level = meta.get("risk_level")
    esi_level = meta.get("esi_level")

    if risk_level or esi_level:
        badge_html = '<div class="badge-row">'
        if esi_level:
            badge_html += esi_badge(esi_level)
        if risk_level:
            badge_html += risk_badge(risk_level)
        badge_html += "</div>"
        st.markdown(badge_html, unsafe_allow_html=True)

    if meta.get("infermedica_triage") or meta.get("intake_method") == "infermedica":
        st.markdown(
            """
<div style="display:inline-flex;align-items:center;gap:0.35rem;padding:0.2rem 0.6rem;border-radius:6px;background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.3);font-size:0.73rem;font-weight:700;color:#7dd3fc;margin-bottom:0.6rem;">
    🔬 Validated via Infermedica Clinical Engine (v3 API)
</div>
""",
            unsafe_allow_html=True,
        )

    if meta.get("is_emergency"):
        st.markdown(
            """
<div class="emergency-banner">
    <strong>🚨 Emergency Pathway Triggered</strong><br>
    Immediate physician or emergency department escalation required. Call <strong>911</strong> for critical or worsening instability.
</div>
""",
            unsafe_allow_html=True,
        )

    if meta.get("awaiting_human_review"):
        st.markdown(
            """
<div class="review-banner">
    <strong>👨‍⚕️ Clinical Review Required (HITL)</strong><br>
    Automated guidance is paused for high-risk / ESI emergent acuity. Licensed staff must review and sign off below.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(content)

    # Render OTC medication guidance cards
    medications = meta.get("medications", [])
    if medications:
        render_medications_cards(medications)

    # Render interactive facility cards
    facilities = meta.get("nearby_facilities", [])
    if facilities:
        render_facilities_cards(facilities)

    trace = meta.get("workflow_trace", [])
    if trace:
        with st.expander(f"🔍 Step-by-Step Multi-Agent Reasoning Trace ({len(trace)} steps)", expanded=is_latest):
            render_reasoning_panel(trace)

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🏥 Clinic Triage Copilot")
        st.markdown(
            "<p style='color:#7dd3fc;font-size:0.78rem;font-weight:700;'>Decision Support for Intake Staff</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        status = st.session_state.get("agent_status", "unknown")

        if "Infermedica" in status or "Groq" in status:
            st.markdown(
                '<p><span class="status-dot"></span> <strong>LangGraph Copilot (Active)</strong></p>',
                unsafe_allow_html=True,
            )
            features = []
            if "Infermedica" in status:
                features.append("Infermedica v3")
            if "Groq" in status:
                features.append("Groq Tool-Calling")
            features.extend(["ESI 1-5 Acuity", "384-dim Memory", "Live GPS"])
            st.markdown(
                f"<p style='color:#718aa8;font-size:0.8rem;'>{' · '.join(features)}</p>",
                unsafe_allow_html=True,
            )
        elif st.session_state.get("agent") is not None:
            st.markdown(
                '<p><span class="status-dot"></span> <strong>LangGraph Copilot (Offline Fallback)</strong></p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='color:#718aa8;font-size:0.8rem;'>Deterministic Regex Intake · ESI 1-5 Scoring · Local Memory</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color:#fca5a5;'><strong>Agent error</strong></p><p style='color:#94a3b8;font-size:0.8rem;'>{status}</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-label'>Patient EHR & Location Context</p>", unsafe_allow_html=True)

        col_age, col_sex = st.columns([1.2, 1.8])
        with col_age:
            age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.profile.get("age", 25))
        with col_sex:
            current_sex = st.session_state.profile.get("sex", "male")
            sex = st.radio("Biological Sex", ["male", "female"], index=0 if current_sex == "male" else 1, horizontal=True)

        conditions_text = st.text_area(
            "Known Comorbidities",
            value=", ".join(st.session_state.profile.get("conditions", [])),
            placeholder="e.g. diabetes, hypertension, asthma",
            height=75,
        )

        st.markdown("<p style='font-size:0.8rem;font-weight:700;color:#94a3b8;margin-bottom:0.25rem;'>📍 Patient Location (GPS / IP Coordinates)</p>", unsafe_allow_html=True)
        col_loc1, col_loc2 = st.columns([2.2, 1.8])
        with col_loc1:
            location_input = st.text_input(
                "Patient Location",
                value=st.session_state.get("patient_location", ""),
                placeholder="Current Location / Address",
                label_visibility="collapsed",
            )
        with col_loc2:
            if st.button("🎯 GPS Detect", use_container_width=True, help="Auto-detect patient's exact live location"):
                with st.spinner("Locating..."):
                    curr = get_current_location()
                    if curr:
                        st.session_state.patient_location = curr["address"]
                        st.session_state.location_lat = curr["lat"]
                        st.session_state.location_lng = curr["lng"]
                        st.rerun()

        # Display exact coordinate badge
        if st.session_state.get("location_lat") is not None and st.session_state.get("location_lng") is not None:
            st.markdown(
                f"""
<div style="padding:0.4rem 0.65rem;border-radius:8px;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.22);font-size:0.75rem;color:#7dd3fc;margin-bottom:0.6rem;">
    📍 <strong>Exact Live GPS:</strong> {st.session_state.location_lat:.4f}° N, {st.session_state.location_lng:.4f}° E<br>
    <span style="color:#94a3b8;font-size:0.7rem;">{st.session_state.get('patient_location', '')}</span>
</div>
""",
                unsafe_allow_html=True,
            )

        if st.button("Save Profile & Geocode", use_container_width=True):
            conditions = [item.strip() for item in conditions_text.split(",") if item.strip()]
            st.session_state.profile = {"age": age, "sex": sex, "conditions": conditions}
            st.session_state.patient_location = location_input.strip()

            coords = geocode_address(location_input.strip())
            if coords:
                st.session_state.location_lat = coords[0]
                st.session_state.location_lng = coords[1]
                st.success(f"Profile saved: Geocoded to ({coords[0]:.4f}, {coords[1]:.4f}).")
            else:
                st.warning("Profile saved. Location coordinates could not be resolved.")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-label'>Session Controls</p>", unsafe_allow_html=True)

        if st.button("New Patient Intake", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_risk = None
            st.session_state.current_esi = None
            st.session_state.pending_review = None
            st.session_state.input_counter += 1
            st.rerun()

        risk_level = st.session_state.get("current_risk")
        esi_level = st.session_state.get("current_esi")
        if risk_level or esi_level:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("<p class='sidebar-label'>Current Acuity</p>", unsafe_allow_html=True)
            if esi_level:
                st.markdown(esi_badge(esi_level), unsafe_allow_html=True)
            if risk_level:
                st.markdown(risk_badge(risk_level), unsafe_allow_html=True)

        pending = st.session_state.get("pending_review")
        if pending:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("<p class='sidebar-label'>Review Queue</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:#fde68a;font-size:0.85rem;'>Awaiting clinical review for ESI Level {pending.get('esi_level', 2)} / {pending.get('risk_level', 'HIGH')} case.</p>",
                unsafe_allow_html=True,
            )


def handle_assessment(user_input: str) -> None:
    profile = st.session_state.profile
    history = [{"role": item["role"], "content": item["content"]} for item in st.session_state.history]
    agent = st.session_state.agent

    loc_address = st.session_state.get("patient_location")
    loc_lat = st.session_state.get("location_lat")
    loc_lng = st.session_state.get("location_lng")

    result = agent.process(
        user_input=user_input.strip(),
        user_id="default",
        age=profile.get("age", 25),
        sex=profile.get("sex", "male"),
        conditions=profile.get("conditions", []),
        history=history,
        location_address=loc_address,
        location_lat=loc_lat,
        location_lng=loc_lng,
    )

    st.session_state.current_risk = result.get("risk_level")
    st.session_state.current_esi = result.get("esi_level")
    st.session_state.history.append({"role": "user", "content": user_input.strip(), "meta": {}})
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["content"],
            "meta": {
                "risk_level": result.get("risk_level"),
                "esi_level": result.get("esi_level"),
                "esi_rationale": result.get("esi_rationale"),
                "is_emergency": result.get("is_emergency", False),
                "awaiting_human_review": result.get("awaiting_human_review", False),
                "workflow_trace": result.get("workflow_trace", []),
                "tool_calls_made": result.get("tool_calls_made", []),
                "symptoms": result.get("symptoms", []),
                "pertinent_negatives": result.get("pertinent_negatives", []),
                "red_flags": result.get("red_flags", []),
                "infermedica_evidence": result.get("infermedica_evidence"),
                "infermedica_triage": result.get("infermedica_triage"),
                "medications": result.get("medications", []),
                "intake_method": result.get("intake_method"),
                "nearby_facilities": result.get("nearby_facilities", []),
            },
        }
    )

    if result.get("awaiting_human_review"):
        st.session_state.pending_review = {
            "state": result.get("pending_state"),
            "risk_level": result.get("risk_level"),
            "esi_level": result.get("esi_level"),
            "content": result.get("content"),
        }
    else:
        st.session_state.pending_review = None

    st.session_state.input_counter += 1
    st.rerun()


def handle_review(decision: str, note: str) -> None:
    pending = st.session_state.pending_review
    if not pending:
        return

    agent = st.session_state.agent
    if agent is None:
        st.error("The triage agent is unavailable, so the review flow cannot continue.")
        return

    result = agent.process(
        pending_state=pending["state"],
        approval_decision=decision,
        approval_note=note.strip() or None,
    )
    st.session_state.current_risk = result.get("risk_level")
    st.session_state.current_esi = result.get("esi_level")
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["content"],
            "meta": {
                "risk_level": result.get("risk_level"),
                "esi_level": result.get("esi_level"),
                "esi_rationale": result.get("esi_rationale"),
                "is_emergency": result.get("is_emergency", False),
                "awaiting_human_review": result.get("awaiting_human_review", False),
                "workflow_trace": result.get("workflow_trace", []),
                "tool_calls_made": result.get("tool_calls_made", []),
                "medications": result.get("medications", []),
                "nearby_facilities": result.get("nearby_facilities", []),
            },
        }
    )
    st.session_state.pending_review = None
    st.rerun()


def render_review_panel() -> None:
    pending = st.session_state.get("pending_review")
    if not pending:
        return

    st.markdown(
        f"""
<div class="glass-card" style="border: 1px solid rgba(251, 191, 36, 0.45); background: rgba(120, 53, 15, 0.12);">
    <h3 style="margin-bottom:0.3rem; color:#fde68a;">👨‍⚕️ Clinical Decision Sign-Off Required</h3>
    <p style="color:#e2e8f0; font-size:0.92rem; margin-bottom:0.5rem;">
        This case was assessed as <strong>{pending.get('risk_level', 'HIGH')} RISK (ESI Level {pending.get('esi_level', 2)})</strong>.
        The multi-agent workflow has paused after the Safety check awaiting human verification before final dispatch.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    note = st.text_input(
        "Clinician / Nurse Review Notes",
        placeholder="e.g. Cleared for immediate transfer to urgent care bay 2; vitals verified",
        key="reviewer_note",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve & Release Guidance", type="primary", use_container_width=True):
            handle_review("approved", note)
    with col2:
        if st.button("Reject & Direct Escalate", use_container_width=True):
            handle_review("rejected", note)


def main() -> None:
    init_state()
    render_sidebar()

    st.markdown(
        """
<div class="hero">
    <div class="hero-pill">⚡ Clinical Decision Support Copilot</div>
    <h1>Smart Triage AI</h1>
    <p>LangGraph multi-agent intake copilot for licensed clinical staff. Aligned with Emergency Severity Index (ESI) acuity standards with live multi-agent reasoning traces and facility navigation.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    infermedica_active = is_infermedica_configured()

    if infermedica_active or groq_key:
        active_badges = []
        if infermedica_active:
            active_badges.append("Infermedica Clinical Engine v3")
        if groq_key:
            model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            active_badges.append(f"Groq LLM (<code>{model_name}</code>)")
        active_badges.extend(["Semantic Memory", "Live GPS Navigation"])

        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:0.5rem;padding:0.45rem 0.95rem;border-radius:10px;background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.28);color:#6ee7b7;font-size:0.83rem;font-weight:600;margin-bottom:1.1rem;max-width: fit-content;">
    <span style="width:9px;height:9px;border-radius:999px;background:#34d399;box-shadow:0 0 10px #34d399;display:inline-block;"></span>
    {' · '.join(active_badges)}
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "ℹ️ Operating in offline mode with deterministic regex extraction, local question generation, and rule-based ESI scoring."
        )

    if not st.session_state.agent:
        st.error("The triage copilot failed to initialize. Check your dependencies and environment configuration.")
        st.stop()

    if st.session_state.history:
        total_msgs = len(st.session_state.history)
        for idx, message in enumerate(st.session_state.history):
            is_latest = (idx == total_msgs - 1)
            render_message(message, is_latest=is_latest)

    render_review_panel()

    pending_review = st.session_state.get("pending_review") is not None
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📝 Record Patient Intake Notes")
    with st.expander("Example Clinical Presentations"):
        st.markdown(
            """
- `Patient reports mild headache and sore throat for 1 day, denies fever` *(ESI 5 · Low Risk)*
- `Severe chest pain radiating to left arm and shortness of breath since this morning` *(ESI 2 · High Risk / HITL)*
- `Moderate abdominal pain with nausea and vomiting for 4 days` *(ESI 3 · Urgent)*
- `Patient lost consciousness for 2 minutes, currently confused and pale` *(ESI 1 · Resuscitation)*
"""
        )

    user_input = st.text_area(
        "Patient Clinical Presentation",
        placeholder="Enter patient chief complaint, symptom onset/duration, severity, and pertinent notes...",
        height=110,
        key=f"user_input_{st.session_state.input_counter}",
        label_visibility="collapsed",
        disabled=pending_review,
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("Run Multi-Agent Triage", type="primary", use_container_width=True, disabled=pending_review)
    with col2:
        if st.button("Clear Input", use_container_width=True):
            st.session_state.input_counter += 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if pending_review:
        st.warning("⚠️ Workflow paused: Complete the clinician review above before triaging a new case.")

    if submit and user_input.strip():
        with st.spinner("Executing multi-agent graph: Supervisor ➔ Intake ➔ Risk ➔ Safety ➔ Resources ➔ Review..."):
            try:
                handle_assessment(user_input)
            except Exception as exc:
                st.error(f"Error during triage processing: {exc}")

    st.markdown(
        """
<div class="disclaimer-banner">
    ⚕️ <strong>Decision Support Disclaimer:</strong> This system assists licensed intake healthcare personnel and is not a patient-facing diagnostic tool. Final triage acuity and clinical decisions remain the sole responsibility of the attending clinician.
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
