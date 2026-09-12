import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Ensure environment variables are dynamically reloaded on every rerun
load_dotenv(override=True)

import streamlit as st

from agent.dispatch_hub import DispatchTicket, HospitalProfile, get_dispatch_hub
from agent.infermedica_client import is_infermedica_configured
from agent.location_tools import build_directions_url, calculate_transit_eta, geocode_address, get_current_location
from agent.multi_agent import get_multi_agent

logger = logging.getLogger("smart-triage-app")

st.set_page_config(
    page_title="Smart Triage AI · Emergency Dispatch & Resource Allocation",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS Design System (Glassmorphic High-Acuity Healthcare Theme)
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

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
    --accent-orange: #fb923c;
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
.block-container { max-width: 1260px !important; padding: 1.5rem 1.8rem 3.5rem !important; }

/* Hero Header */
.hero {
    padding: 1.2rem 0 0.8rem;
    margin-bottom: 1rem;
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
    margin-bottom: 0.6rem;
}

.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.3rem;
    background: linear-gradient(120deg, #f0f9ff, #7dd3fc 45%, #a7f3d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: var(--text-muted);
    font-size: 0.98rem;
    max-width: 760px;
    margin: 0 auto;
}

/* Glass Cards */
.glass-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 16px 45px rgba(0, 0, 0, 0.25);
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
    margin-left: 10%;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(14, 165, 233, 0.05));
    border-color: rgba(56, 189, 248, 0.28);
}

.msg-ai {
    margin-right: 5%;
    background: rgba(255, 255, 255, 0.035);
}

.msg-user::before {
    content: "PATIENT / INTAKE PRESENTATION";
    display: block;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
    color: var(--accent-blue);
}

.msg-ai::before {
    content: "TRIAGE COPILOT DISPATCH ENGINE";
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

/* Hospital Recommendation Cards */
.hospital-rec-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 1.15rem 1.3rem;
    margin-bottom: 1rem;
    transition: all 0.25s ease;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

.hospital-rec-card:hover {
    border-color: rgba(56, 189, 248, 0.6);
    transform: translateY(-2px);
    box-shadow: 0 16px 36px rgba(56, 189, 248, 0.12);
}

.hosp-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.6rem;
}

.hosp-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #f8fafc;
}

.hosp-rating {
    background: rgba(251, 191, 36, 0.15);
    border: 1px solid rgba(251, 191, 36, 0.4);
    color: #fde68a;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
}

.hosp-metric-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    margin: 0.5rem 0 0.8rem;
    align-items: center;
}

.hosp-metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.pill-eta { color: #38bdf8; background: rgba(56, 189, 248, 0.12); border-color: rgba(56, 189, 248, 0.3); }
.pill-match { color: #34d399; background: rgba(52, 211, 153, 0.12); border-color: rgba(52, 211, 153, 0.3); }
.pill-bays { color: #a78bfa; background: rgba(167, 139, 250, 0.12); border-color: rgba(167, 139, 250, 0.3); }

/* 5-Stage Live Emergency Tracker */
.tracker-container {
    background: linear-gradient(145deg, rgba(14, 34, 61, 0.9), rgba(7, 17, 31, 0.95));
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin: 1.2rem 0;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
}

.tracker-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.tracker-ticket-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    background: rgba(56, 189, 248, 0.18);
    border: 1px solid rgba(56, 189, 248, 0.45);
    color: #7dd3fc;
}

.stepper-progress {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.6rem;
    margin-bottom: 1.3rem;
    position: relative;
}

.stepper-stage {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 0.75rem 0.6rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stepper-stage.active {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.25), rgba(45, 212, 191, 0.15));
    border-color: rgba(56, 189, 248, 0.7);
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.25);
}

.stepper-stage.completed {
    background: rgba(52, 211, 153, 0.12);
    border-color: rgba(52, 211, 153, 0.4);
}

.stage-num {
    font-size: 0.72rem;
    font-weight: 800;
    color: #94a3b8;
    margin-bottom: 0.2rem;
}

.stepper-stage.active .stage-num { color: #38bdf8; }
.stepper-stage.completed .stage-num { color: #34d399; }

.stage-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* Pre-Allocated Resource Cockpit Box */
.resource-cockpit {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}

.resource-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.8rem;
    margin-top: 0.5rem;
}

.resource-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
}

.resource-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #94a9c4;
    margin-bottom: 0.2rem;
}

.resource-value {
    font-size: 0.85rem;
    font-weight: 700;
    color: #f0fdf4;
}

/* Hospital Cockpit Incoming Card */
.incoming-alert-card {
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    border: 1px solid;
    backdrop-filter: blur(12px);
    transition: all 0.2s ease;
}

.incoming-card-esi1 {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.5), rgba(153, 27, 27, 0.2));
    border-color: rgba(239, 68, 68, 0.7);
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
}

.incoming-card-esi2 {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.45), rgba(146, 64, 14, 0.18));
    border-color: rgba(248, 113, 113, 0.55);
}

.incoming-card-standard {
    background: rgba(15, 23, 42, 0.7);
    border-color: rgba(56, 189, 248, 0.25);
}

/* KPI Metric Cards */
.kpi-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}

.kpi-value {
    font-size: 1.6rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(120deg, #f0f9ff, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-label {
    font-size: 0.74rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.15rem;
}

/* Direction Buttons */
.nav-link-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.55rem 1.1rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #0284c7, #0369a1);
    color: #ffffff !important;
    text-decoration: none !important;
    font-weight: 700;
    font-size: 0.84rem;
    transition: all 0.2s ease;
    border: 1px solid rgba(56, 189, 248, 0.4);
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
}

.nav-link-btn:hover {
    background: linear-gradient(135deg, #0369a1, #075985);
    transform: translateY(-1px);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------


def init_state() -> None:
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = get_multi_agent()
        except Exception as exc:
            logger.exception("Failed to initialize multi-agent copilot: %s", exc)
            st.session_state.agent = None

    if "dispatch_hub" not in st.session_state:
        st.session_state.dispatch_hub = get_dispatch_hub()

    if "history" not in st.session_state:
        st.session_state.history = []

    if "profile" not in st.session_state:
        st.session_state.profile = {"age": 28, "sex": "male", "conditions": []}

    if "patient_location" not in st.session_state:
        auto_loc = get_current_location()
        if auto_loc:
            st.session_state.patient_location = auto_loc["address"]
            st.session_state.location_lat = auto_loc["lat"]
            st.session_state.location_lng = auto_loc["lng"]
            st.session_state.location_city = auto_loc.get("city", "Local")
        else:
            st.session_state.patient_location = "Coimbatore, Tamil Nadu, India"
            st.session_state.location_lat = 11.0168
            st.session_state.location_lng = 76.9558
            st.session_state.location_city = "Coimbatore"

    if "current_risk" not in st.session_state:
        st.session_state.current_risk = None
    if "current_esi" not in st.session_state:
        st.session_state.current_esi = None
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    if "pending_review" not in st.session_state:
        st.session_state.pending_review = None
    if "active_dispatch_id" not in st.session_state:
        st.session_state.active_dispatch_id = None
    if "portal_view" not in st.session_state:
        st.session_state.portal_view = "patient"


# ---------------------------------------------------------------------------
# Visual Component Helpers
# ---------------------------------------------------------------------------


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
        "resources": {"icon": "📍", "title": "Smart Hospital Matcher & Dispatch Hub"},
        "human_review": {"icon": "👨‍⚕️", "title": "Clinical Review Node (HITL)"},
        "finish": {"icon": "🏁", "title": "Workflow Finalizer"},
    }

    st.markdown('<div class="reasoning-panel" style="background:rgba(0,0,0,0.25);border-radius:12px;padding:0.9rem 1.1rem;margin-top:0.8rem;">', unsafe_allow_html=True)
    st.markdown(
        """
<div style="display:flex;flex-wrap:wrap;gap:0.4rem;align-items:center;margin-bottom:0.7rem;font-size:0.75rem;color:#7dd3fc;font-weight:700;">
    <span>🧭 Supervisor</span> ➔ <span>📋 Intake</span> ➔ <span>⚠️ Risk & ESI</span> ➔ <span>🛡️ Safety</span> ➔ <span>📍 Smart Hospital Matcher</span> ➔ <span>🏁 Ready</span>
</div>
""",
        unsafe_allow_html=True,
    )

    for index, step in enumerate(trace, start=1):
        node_name = step.get("node", "node")
        meta = node_metadata.get(node_name, {"icon": "⚙️", "title": node_name.capitalize()})
        tools = step.get("tools", [])
        chips = "".join(
            f"<span style='display:inline-block;padding:0.15rem 0.45rem;border-radius:6px;background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.25);font-size:0.68rem;color:#7dd3fc;margin-right:0.3rem;'>{tool}</span>"
            for tool in tools
        )

        st.markdown(
            f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.45rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">
        <span style="font-weight:700;font-size:0.82rem;color:#f1f5f9;">{meta['icon']} {meta['title']}</span>
        <span style="font-size:0.68rem;font-weight:800;color:#38bdf8;">STEP {index}</span>
    </div>
    <div style="font-size:0.78rem;color:#cbd5e1;">{step.get('summary', '')}</div>
    {f"<div style='margin-top:0.35rem;'>{chips}</div>" if chips else ""}
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live Emergency Dispatch Tracker Component
# ---------------------------------------------------------------------------


def render_live_dispatch_tracker(ticket: DispatchTicket) -> None:
    stages = [
        ("1", "Triage Assessed", "REQUESTED"),
        ("2", "Bay Reserved", "ACCEPTED_RESERVED"),
        ("3", "Team Mobilized", "TEAM_MOBILIZED"),
        ("4", "Patient En Route", "EN_ROUTE"),
        ("5", "Bed Handover", "ARRIVED_ADMITTED"),
    ]

    status_order = {
        "REQUESTED": 1,
        "ACCEPTED_RESERVED": 2,
        "TEAM_MOBILIZED": 3,
        "EN_ROUTE": 4,
        "ARRIVED_ADMITTED": 5,
        "CANCELLED": 0,
    }

    curr_idx = status_order.get(ticket.status, 2)
    alloc = ticket.resource_allocation

    st.markdown('<div class="tracker-container">', unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="tracker-title">
    <div>
        <span style="font-size:0.78rem;font-weight:800;letter-spacing:0.06em;color:#38bdf8;text-transform:uppercase;">⚡ LIVE EMERGENCY DISPATCH & PRE-ALLOCATION TRACKER</span>
        <h3 style="margin:0.2rem 0;color:#f8fafc;font-size:1.35rem;">🚑 {ticket.hospital_name}</h3>
        <p style="color:#94a3b8;font-size:0.82rem;margin:0;">📍 {ticket.hospital_address} · ⏱️ <strong>ETA: ~{ticket.estimated_transit_mins} mins</strong> ({ticket.distance_km:.1f} km away)</p>
    </div>
    <div>
        <span class="tracker-ticket-pill">🎫 {ticket.ticket_id}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # 5-Stage Stepper Progress Bar
    stepper_cols = ""
    for num, label, key in stages:
        stage_val = status_order[key]
        if stage_val < curr_idx:
            cls = "stepper-stage completed"
            icon = "✓"
        elif stage_val == curr_idx:
            cls = "stepper-stage active"
            icon = "●"
        else:
            cls = "stepper-stage"
            icon = num

        stepper_cols += f"""
<div class="{cls}">
    <div class="stage-num">{icon} STAGE {num}</div>
    <div class="stage-label">{label}</div>
</div>
"""

    st.markdown(f'<div class="stepper-progress">{stepper_cols}</div>', unsafe_allow_html=True)

    # Pre-Allocated Resources Box
    st.markdown(
        f"""
<div class="resource-cockpit">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
        <span style="font-size:0.75rem;font-weight:800;color:#2dd4bf;text-transform:uppercase;letter-spacing:0.05em;">🏥 PRE-ALLOCATED HOSPITAL RESOURCES</span>
        <span style="padding:0.2rem 0.55rem;border-radius:6px;font-size:0.72rem;font-weight:800;background:rgba(52,211,153,0.15);color:#6ee7b7;border:1px solid rgba(52,211,153,0.3);">
            STATUS: {alloc.readiness_status}
        </span>
    </div>
    <div class="resource-grid">
        <div class="resource-item">
            <div class="resource-label">Reserved Emergency Bay</div>
            <div class="resource-value" style="color:#38bdf8;">🛏️ {alloc.allocated_bay}</div>
        </div>
        <div class="resource-item">
            <div class="resource-label">Assigned Attending Clinician</div>
            <div class="resource-value">👨‍⚕️ {alloc.assigned_doctor}</div>
        </div>
        <div class="resource-item">
            <div class="resource-label">Lead Triage RN / Team</div>
            <div class="resource-value">👩‍⚕️ {alloc.assigned_nurse}</div>
        </div>
        <div class="resource-item">
            <div class="resource-label">On-Call Specialist Alerted</div>
            <div class="resource-value">{alloc.specialist_on_call or 'ER Intensivist'}</div>
        </div>
    </div>
    <div style="margin-top:0.7rem;padding-top:0.6rem;border-top:1px solid rgba(255,255,255,0.06);font-size:0.78rem;color:#cbd5e1;">
        ⚙️ <strong>Pre-Staged Equipment:</strong> {', '.join(alloc.staged_equipment)}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Live Navigation and Dispatch Controls
    col_nav, col_stat, col_cancel = st.columns([2, 1.5, 1])
    with col_nav:
        st.markdown(
            f"""
<div style="margin-top:0.8rem;">
    <a href="{ticket.directions_url}" target="_blank" rel="noopener noreferrer" class="nav-link-btn" title="Open Google Maps Live Driving Directions">
        🗺️ Start Live GPS Driving Navigation
    </a>
</div>
""",
            unsafe_allow_html=True,
        )
    with col_stat:
        if ticket.status != "ARRIVED_ADMITTED":
            if st.button("🚑 Mark as 'En Route' to Hospital", key="btn_en_route", use_container_width=True):
                get_dispatch_hub().update_dispatch_status(ticket.ticket_id, "EN_ROUTE", "Patient is driving/in-transit to hospital")
                st.rerun()
    with col_cancel:
        if st.button("Cancel Dispatch", key="btn_cancel_disp", use_container_width=True):
            get_dispatch_hub().update_dispatch_status(ticket.ticket_id, "CANCELLED", "Dispatch cancelled by user")
            st.session_state.active_dispatch_id = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Smart Hospital Recommendations Renderer
# ---------------------------------------------------------------------------


def render_hospital_recommendations(hospitals: List[Dict[str, Any]], message_idx: int) -> None:
    if not hospitals:
        return

    st.markdown(
        """
<div style="margin-top:1.2rem;margin-bottom:0.6rem;">
    <p style='font-size:0.88rem;font-weight:800;letter-spacing:0.04em;color:#38bdf8;margin-bottom:0.2rem;'>
        🏪 NEARBY EMERGENCY CENTERS & PRE-ADMISSION DISPATCH
    </p>
    <p style='font-size:0.76rem;color:#94a3b8;margin-bottom:0.8rem;'>
        Ranked by clinical specialty match, emergency capacity, and driving proximity. Select a hospital below to trigger instant bed reservation & resource pre-allocation.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    for idx, hosp in enumerate(hospitals[:4]):
        hosp_id = hosp.get("hospital_id", f"hosp-{idx}")
        name = hosp.get("name", "Emergency Hospital")
        dist = hosp.get("distance_km", 1.0)
        eta_mins = hosp.get("estimated_transit_mins", calculate_transit_eta(dist))
        match_score = hosp.get("match_score", 95)
        avail_bays = hosp.get("available_er_bays", 4)
        total_bays = hosp.get("total_er_bays", 12)
        rating = hosp.get("rating", 4.8)
        trauma_lvl = hosp.get("trauma_level", "Emergency Trauma Center")
        spec_badge = hosp.get("specialty_badge", "🚨 24/7 Acute Care")
        caps = hosp.get("capabilities", ["Acute Care", "CT/MRI", "Trauma"])

        card_html = f"""
<div class="hospital-rec-card">
    <div class="hosp-header">
        <div>
            <span style="display:inline-block;padding:0.2rem 0.55rem;border-radius:6px;font-size:0.7rem;font-weight:800;background:rgba(56,189,248,0.15);color:#7dd3fc;border:1px solid rgba(56,189,248,0.3);margin-bottom:0.35rem;">
                {spec_badge}
            </span>
            <div class="hosp-title">{name}</div>
            <div style="font-size:0.76rem;color:#94a3b8;margin-top:0.15rem;">{hosp.get('address', '')}</div>
        </div>
        <div class="hosp-rating">⭐ {rating:.1f}</div>
    </div>
    <div class="hosp-metric-row">
        <span class="hosp-metric-pill pill-match">🎯 {match_score}% Clinical Match</span>
        <span class="hosp-metric-pill pill-eta">⏱️ ~{eta_mins} mins transit ({dist:.1f} km)</span>
        <span class="hosp-metric-pill pill-bays">🟢 {avail_bays}/{total_bays} ER Bays Free</span>
        <span class="hosp-metric-pill" style="color:#cbd5e1;">🏥 {trauma_lvl}</span>
    </div>
    <div style="font-size:0.73rem;color:#64748b;margin-bottom:0.4rem;">
        <strong>Specialized Facilities:</strong> {', '.join(caps[:4])}
    </div>
</div>
"""
        st.markdown(card_html, unsafe_allow_html=True)

        # 1-Click Dispatch Button for this hospital
        col1, col2 = st.columns([3, 1])
        with col1:
            btn_label = f"🚑 Dispatch & Pre-Book Emergency Admission to {name.split('-')[0].strip()}"
            if st.button(btn_label, key=f"dispatch_{message_idx}_{idx}_{hosp_id}", type="primary", use_container_width=True):
                profile = st.session_state.profile
                esi_lvl = st.session_state.get("current_esi", 3) or 3
                risk_lvl = st.session_state.get("current_risk", "MEDIUM") or "MEDIUM"

                ticket = get_dispatch_hub().create_dispatch(
                    patient_id=f"pt-{profile.get('sex', 'm')}-{profile.get('age', 25)}",
                    patient_age=profile.get("age", 25),
                    patient_sex=profile.get("sex", "male"),
                    patient_conditions=profile.get("conditions", []),
                    chief_complaint=st.session_state.history[-2]["content"] if len(st.session_state.history) >= 2 else "Acute Clinical Complaint",
                    symptoms=st.session_state.history[-1].get("meta", {}).get("symptoms", ["acute symptoms"]),
                    risk_level=risk_lvl,
                    esi_level=esi_lvl,
                    esi_rationale=st.session_state.history[-1].get("meta", {}).get("esi_rationale", "Emergency presentation"),
                    is_emergency=risk_lvl == "HIGH" or esi_lvl in (1, 2),
                    hospital_id=hosp_id,
                    origin_lat=st.session_state.get("location_lat"),
                    origin_lng=st.session_state.get("location_lng"),
                    distance_km=dist,
                    estimated_transit_mins=eta_mins,
                )
                st.session_state.active_dispatch_id = ticket.ticket_id
                st.toast(f"🚑 Dispatch Ticket {ticket.ticket_id} created! Hospital notified & bay pre-allocated.", icon="🏥")
                st.rerun()
        with col2:
            dir_url = hosp.get("directions_url", "#")
            st.markdown(
                f"""
<a href="{dir_url}" target="_blank" rel="noopener noreferrer" class="nav-link-btn" style="padding:0.45rem 0.8rem;font-size:0.76rem;width:100%;justify-content:center;">
    🗺️ Directions
</a>
""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Patient Message Renderer
# ---------------------------------------------------------------------------


def render_patient_message(message: Dict[str, Any], msg_idx: int, is_latest: bool = False) -> None:
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
    🔬 Dual Consensus Validated via Infermedica Clinical Engine (v3)
</div>
""",
            unsafe_allow_html=True,
        )

    if meta.get("is_emergency"):
        st.markdown(
            """
<div style="background:linear-gradient(135deg, rgba(127,29,29,0.6), rgba(153,27,27,0.25));border:1px solid rgba(248,113,113,0.55);border-radius:12px;padding:0.85rem 1.1rem;margin:0.7rem 0;font-size:0.88rem;">
    🚨 <strong>Emergent Acuity Triggered (ESI Level 1 / 2):</strong> Immediate hospital transfer required. Pre-reserve your emergency trauma bay below.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(content)

    # Render Swiggy-style Hospital Recommendations
    hospitals = meta.get("hospital_recommendations") or meta.get("nearby_facilities")
    if hospitals:
        render_hospital_recommendations(hospitals, msg_idx)

    # Multi-agent reasoning trace
    trace = meta.get("workflow_trace", [])
    if trace:
        with st.expander(f"🔍 Multi-Agent Graph Reasoning Trace ({len(trace)} steps)", expanded=is_latest):
            render_reasoning_panel(trace)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PORTAL 1: Patient Triage & 1-Click Dispatch Portal
# ---------------------------------------------------------------------------


def render_patient_portal() -> None:
    # Check if there is an active dispatch ticket in progress
    active_ticket_id = st.session_state.get("active_dispatch_id")
    if active_ticket_id:
        active_ticket = get_dispatch_hub().get_dispatch(active_ticket_id)
        if active_ticket and active_ticket.status != "CANCELLED":
            render_live_dispatch_tracker(active_ticket)

    # Chat history
    if st.session_state.history:
        for idx, msg in enumerate(st.session_state.history):
            is_latest = idx == len(st.session_state.history) - 1
            render_patient_message(msg, msg_idx=idx, is_latest=is_latest)

    # Intake Input Form
    pending_review = st.session_state.get("pending_review") is not None
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📝 Patient Symptom & Triage Intake")

    user_input = st.text_area(
        "Patient Presentation",
        placeholder="Describe symptoms (e.g. 'Severe crushing chest pain radiating to left arm for 45 mins, sweating')...",
        height=95,
        key=f"user_input_{st.session_state.input_counter}",
        label_visibility="collapsed",
        disabled=pending_review,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🚀 Run Agentic Triage & Hospital Matcher", type="primary", use_container_width=True, disabled=pending_review)
    with col2:
        if st.button("Clear Input", use_container_width=True):
            st.session_state.input_counter += 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if submit and user_input.strip():
        with st.spinner("Executing Agentic Graph: Supervisor ➔ Intake ➔ Risk ➔ Safety ➔ Smart Hospital Matcher..."):
            profile = st.session_state.profile
            history = [{"role": item["role"], "content": item["content"]} for item in st.session_state.history]
            agent = st.session_state.agent

            loc_address = st.session_state.get("patient_location")
            loc_lat = st.session_state.get("location_lat")
            loc_lng = st.session_state.get("location_lng")

            result = agent.process(
                user_input=user_input.strip(),
                user_id="pt-active",
                age=profile.get("age", 28),
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
                        "hospital_recommendations": result.get("hospital_recommendations", []),
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
            st.session_state.input_counter += 1
            st.rerun()


# ---------------------------------------------------------------------------
# PORTAL 2: Hospital Emergency Command Center (Live POS Cockpit)
# ---------------------------------------------------------------------------


def render_hospital_portal() -> None:
    hub = get_dispatch_hub()
    hospitals = hub.list_hospitals()

    # Hospital View Selector
    col_sel, col_ref = st.columns([3, 1])
    with col_sel:
        hosp_options = {"all": "🏥 All Network Emergency Rooms (Central Command)"}
        for h in hospitals:
            hosp_options[h.id] = f"🏥 {h.name} (ER {h.available_er_bays}/{h.total_er_bays} Free)"

        selected_hosp_id = st.selectbox(
            "Select Hospital Command Station",
            options=list(hosp_options.keys()),
            format_func=lambda k: hosp_options[k],
            key="selected_hospital_cockpit",
        )
    with col_ref:
        if st.button("🔄 Refresh Live Queue", use_container_width=True):
            st.rerun()

    # KPI Telemetry Cards
    dispatches = hub.list_dispatches_for_hospital(selected_hosp_id)
    active_dispatches = [d for d in dispatches if d.status not in ("ARRIVED_ADMITTED", "CANCELLED")]

    if selected_hosp_id != "all":
        hosp_obj = hub.get_hospital(selected_hosp_id)
        total_er = hosp_obj.total_er_bays if hosp_obj else 12
        avail_er = hosp_obj.available_er_bays if hosp_obj else 4
        avail_icu = hosp_obj.available_icu_bays if hosp_obj else 2
        occupancy = hosp_obj.occupancy_pct() if hosp_obj else 65.0
    else:
        total_er = sum(h.total_er_bays for h in hospitals)
        avail_er = sum(h.available_er_bays for h in hospitals)
        avail_icu = sum(h.available_icu_bays for h in hospitals)
        occupancy = 68.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-value">{len(active_dispatches)}</div>
    <div class="kpi-label">🚨 Incoming Dispatches</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-value" style="color:#34d399;">{avail_er} / {total_er}</div>
    <div class="kpi-label">🛏️ Available ER Bays</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-value" style="color:#38bdf8;">{avail_icu}</div>
    <div class="kpi-label">🫁 ICU Resuscitation Bays</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-value">{occupancy:.0f}%</div>
    <div class="kpi-label">📊 ER Occupancy Rate</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:1.3rem 0;'>", unsafe_allow_html=True)
    st.markdown("### 🚨 Live Incoming Emergency Dispatch Queue")

    if not dispatches:
        st.info("No emergency dispatches in queue. New admissions triggered from the Patient Triage app will appear here instantly.")
        return

    for disp in dispatches:
        alloc = disp.resource_allocation
        is_emergent = disp.esi_level in (1, 2) or disp.risk_level == "HIGH"
        card_class = "incoming-card-esi1" if disp.esi_level == 1 else ("incoming-card-esi2" if disp.esi_level == 2 else "incoming-card-standard")

        st.markdown(
            f"""
<div class="incoming-alert-card {card_class}">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.6rem;">
        <div>
            <span class="badge-esi badge-esi-{disp.esi_level}">ESI {disp.esi_level} · {'Resuscitation' if disp.esi_level == 1 else 'Emergent'}</span>
            <span style="display:inline-block;padding:0.25rem 0.6rem;border-radius:999px;font-size:0.75rem;font-weight:800;background:rgba(56,189,248,0.18);color:#7dd3fc;margin-left:0.4rem;">
                STATUS: {disp.status}
            </span>
            <h4 style="margin:0.4rem 0 0.1rem;color:#f8fafc;font-size:1.15rem;">
                Patient ID: {disp.patient_id} ({disp.patient_age}y / {disp.patient_sex})
            </h4>
            <p style="color:#cbd5e1;font-size:0.85rem;margin:0;"><strong>Chief Complaint:</strong> {disp.chief_complaint}</p>
        </div>
        <div style="text-align:right;">
            <span class="tracker-ticket-pill">{disp.ticket_id}</span>
            <div style="font-size:0.8rem;color:#fde68a;font-weight:700;margin-top:0.3rem;">
                ⏱️ ETA: ~{disp.estimated_transit_mins} mins ({disp.distance_km:.1f} km)
            </div>
        </div>
    </div>

    <div style="background:rgba(0,0,0,0.3);border-radius:10px;padding:0.75rem 1rem;margin:0.6rem 0;">
        <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.2rem;">
            <strong>Reported Symptoms:</strong> {', '.join(disp.symptoms) or 'Acute complaint'} |
            <strong>Comorbidities:</strong> {', '.join(disp.patient_conditions) or 'None documented'}
        </div>
        <div style="font-size:0.76rem;color:#cbd5e1;">
            <strong>Acuity Rationale:</strong> {disp.esi_rationale}
        </div>
    </div>

    <!-- Pre-Allocated Resources Display -->
    <div class="resource-grid" style="margin-bottom:0.8rem;">
        <div class="resource-item">
            <div class="resource-label">Pre-Allocated Bay</div>
            <div class="resource-value" style="color:#38bdf8;">🛏️ {alloc.allocated_bay}</div>
        </div>
        <div class="resource-item">
            <div class="resource-label">Assigned Clinician</div>
            <div class="resource-value">👨‍⚕️ {alloc.assigned_doctor}</div>
        </div>
        <div class="resource-item">
            <div class="resource-label">Staged Equipment</div>
            <div class="resource-value">{', '.join(alloc.staged_equipment[:2])}</div>
        </div>
        <div class="resource-item">
            <div class="resource-label">Readiness</div>
            <div class="resource-value" style="color:#34d399;">{alloc.readiness_status}</div>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Staff Action Controls
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if disp.status == "ACCEPTED_RESERVED":
                if st.button(f"🚨 Mobilize Team & Stage Equipment", key=f"mob_{disp.ticket_id}", type="primary", use_container_width=True):
                    hub.update_dispatch_status(disp.ticket_id, "TEAM_MOBILIZED", "Trauma team mobilized and equipment staged at bay")
                    st.toast(f"Care team mobilized for {disp.ticket_id}", icon="👨‍⚕️")
                    st.rerun()
            elif disp.status == "TEAM_MOBILIZED":
                st.success("✅ Team Mobilized & On Standby")
        with col_act2:
            if disp.status in ("ACCEPTED_RESERVED", "TEAM_MOBILIZED", "EN_ROUTE"):
                if st.button(f"🎯 Confirm Arrival & Handover", key=f"arr_{disp.ticket_id}", use_container_width=True):
                    hub.update_dispatch_status(disp.ticket_id, "ARRIVED_ADMITTED", "Patient arrived at ER; admitted directly into pre-allocated bay")
                    st.toast(f"Patient {disp.ticket_id} admitted & handed over!", icon="🎯")
                    st.rerun()
            elif disp.status == "ARRIVED_ADMITTED":
                st.info("🏥 Admitted to Bed")
        with col_act3:
            dir_url = disp.directions_url
            st.markdown(
                f"""
<a href="{dir_url}" target="_blank" rel="noopener noreferrer" class="nav-link-btn" style="padding:0.45rem 0.8rem;font-size:0.78rem;width:100%;justify-content:center;">
    🗺️ Patient Transit Route
</a>
""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar & Patient Context
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🚑 Smart Triage AI")
        st.markdown(
            "<p style='color:#38bdf8;font-size:0.8rem;font-weight:700;'>Emergency Dispatch & Resource Pre-Allocation</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:0.8rem 0;'>", unsafe_allow_html=True)

        # Mode Indicator
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        infermedica_active = is_infermedica_configured()
        badges = []
        if infermedica_active:
            badges.append("Infermedica v3")
        if groq_key:
            badges.append("Groq LLM")
        badges.extend(["Live Dispatch Hub", "GPS Routing"])

        st.markdown(
            f"""
<div style="padding:0.4rem 0.65rem;border-radius:8px;background:rgba(52,211,153,0.09);border:1px solid rgba(52,211,153,0.25);font-size:0.74rem;color:#6ee7b7;font-weight:700;margin-bottom:0.8rem;">
    🟢 {' · '.join(badges)}
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<p style='font-size:0.78rem;font-weight:800;color:#94a3b8;margin-bottom:0.3rem;'>PATIENT EHR & LOCATION CONTEXT</p>", unsafe_allow_html=True)
        col_age, col_sex = st.columns([1.2, 1.8])
        with col_age:
            age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.profile.get("age", 28))
        with col_sex:
            sex = st.radio("Sex", ["male", "female"], index=0 if st.session_state.profile.get("sex") == "male" else 1, horizontal=True)

        conditions_text = st.text_area(
            "Known Comorbidities",
            value=", ".join(st.session_state.profile.get("conditions", [])),
            placeholder="e.g. Hypertension, Diabetes, Asthma",
            height=65,
        )

        st.markdown("<p style='font-size:0.78rem;font-weight:800;color:#94a3b8;margin-bottom:0.2rem;'>📍 Live Location (GPS / IP)</p>", unsafe_allow_html=True)
        loc_input = st.text_input("Address / Coordinates", value=st.session_state.get("patient_location", ""), label_visibility="collapsed")

        col_loc1, col_loc2 = st.columns(2)
        with col_loc1:
            if st.button("🎯 GPS Auto-Detect", use_container_width=True):
                with st.spinner("Locating..."):
                    loc = get_current_location()
                    if loc:
                        st.session_state.patient_location = loc["address"]
                        st.session_state.location_lat = loc["lat"]
                        st.session_state.location_lng = loc["lng"]
                        st.rerun()
        with col_loc2:
            if st.button("Save Profile", use_container_width=True):
                conds = [c.strip() for c in conditions_text.split(",") if c.strip()]
                st.session_state.profile = {"age": age, "sex": sex, "conditions": conds}
                st.session_state.patient_location = loc_input.strip()
                coords = geocode_address(loc_input.strip())
                if coords:
                    st.session_state.location_lat = coords[0]
                    st.session_state.location_lng = coords[1]
                st.success("Profile saved!")

        if st.session_state.get("location_lat") and st.session_state.get("location_lng"):
            st.markdown(
                f"""
<div style="font-size:0.72rem;color:#7dd3fc;padding:0.35rem 0.55rem;background:rgba(56,189,248,0.08);border-radius:6px;border:1px solid rgba(56,189,248,0.2);margin-top:0.4rem;">
    📍 GPS: {st.session_state.location_lat:.4f}° N, {st.session_state.location_lng:.4f}° E
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:0.8rem 0;'>", unsafe_allow_html=True)
        if st.button("🆕 New Patient Session", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_risk = None
            st.session_state.current_esi = None
            st.session_state.pending_review = None
            st.session_state.active_dispatch_id = None
            st.session_state.input_counter += 1
            st.rerun()


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


def main() -> None:
    init_state()
    render_sidebar()

    # Hero Header
    st.markdown(
        """
<div class="hero">
    <div class="hero-pill">⚡ Emergency Triage & Hospital Resource Dispatch Ecosystem</div>
    <h1>Smart Triage AI</h1>
    <p>AI multi-agent symptom triage with instant nearby hospital ranking, 1-click emergency pre-admission dispatch, and real-time hospital resource pre-allocation.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Top Navigation View Switcher (Dual Portal)
    tab_patient, tab_hospital = st.tabs([
        "🚑 Patient Triage & Live Dispatch Portal",
        "🏥 Hospital Emergency Command Center (Live POS Cockpit)",
    ])

    with tab_patient:
        render_patient_portal()

    with tab_hospital:
        render_hospital_portal()

    st.markdown(
        """
<div style="background:rgba(15,23,42,0.6);border:1px solid rgba(148,163,184,0.2);color:#94a3b8;font-size:0.78rem;text-align:center;border-radius:12px;padding:0.75rem 1rem;margin-top:2rem;">
    ⚕️ <strong>Decision Support & Dispatch Disclaimer:</strong> This system assists emergency healthcare resource matching and pre-registration. Clinical care and diagnosis are delivered by licensed attending medical staff upon arrival.
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
