import os
from typing import Any, Dict, List

import streamlit as st


st.set_page_config(
    page_title="Smart Triage AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg-1: #07111f;
    --bg-2: #10253f;
    --card: rgba(255,255,255,0.05);
    --card-strong: rgba(255,255,255,0.08);
    --border: rgba(255,255,255,0.12);
    --text: #ebf3ff;
    --muted: #8fa7c5;
    --high: #f87171;
    --medium: #fbbf24;
    --low: #34d399;
    --accent: #7dd3fc;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(45,212,191,0.14), transparent 22%),
        linear-gradient(155deg, var(--bg-1), var(--bg-2) 52%, #091423);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1160px !important; padding: 2rem 2rem 4rem !important; }

.hero {
    padding: 2.2rem 0 1.4rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.55rem;
    background: linear-gradient(120deg, #e0f2fe, #7dd3fc 45%, #99f6e4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: var(--muted);
    font-size: 1.02rem;
}

.glass-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.3rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
}

.msg-user, .msg-ai {
    border-radius: 18px;
    padding: 1rem 1.15rem;
    margin: 0.7rem 0;
    border: 1px solid var(--border);
}

.msg-user {
    margin-left: 14%;
    background: rgba(56,189,248,0.09);
}

.msg-ai {
    margin-right: 8%;
    background: rgba(255,255,255,0.04);
}

.msg-user::before, .msg-ai::before {
    display: block;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.45rem;
}

.msg-user::before {
    content: "USER";
    color: #7dd3fc;
}

.msg-ai::before {
    content: "TRIAGE ORCHESTRATOR";
    color: #99f6e4;
}

.badge-HIGH, .badge-MEDIUM, .badge-LOW {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.84rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}

.badge-HIGH { background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.45); color: #fecaca; }
.badge-MEDIUM { background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.45); color: #fde68a; }
.badge-LOW { background: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.45); color: #bbf7d0; }

.emergency-banner, .review-banner {
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin: 0.9rem 0;
    border: 1px solid;
}

.emergency-banner {
    background: linear-gradient(135deg, rgba(127,29,29,0.55), rgba(153,27,27,0.2));
    border-color: rgba(248,113,113,0.45);
}

.review-banner {
    background: linear-gradient(135deg, rgba(120,53,15,0.45), rgba(146,64,14,0.18));
    border-color: rgba(251,191,36,0.45);
}

.trace-step {
    border-left: 3px solid rgba(125,211,252,0.4);
    padding-left: 0.9rem;
    margin-bottom: 0.9rem;
}

.trace-label {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7dd3fc;
    margin-bottom: 0.25rem;
}

.trace-content {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    color: #dbeafe;
    font-size: 0.88rem;
}

.tool-chip {
    display: inline-block;
    margin: 0.12rem 0.2rem 0.12rem 0;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: rgba(125,211,252,0.1);
    border: 1px solid rgba(125,211,252,0.25);
    color: #bae6fd;
    font-size: 0.75rem;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background: rgba(5,11,20,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

.sidebar-label {
    font-size: 0.73rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6782a7;
    margin-bottom: 0.5rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    display: inline-block;
    margin-right: 6px;
    background: #34d399;
    box-shadow: 0 0 12px rgba(52,211,153,0.8);
}

.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.2rem 0;
}

.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
}

.stButton > button {
    border-radius: 12px !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    background: linear-gradient(120deg, #0ea5e9, #14b8a6) !important;
}

.stButton > button:hover {
    box-shadow: 0 12px 30px rgba(20,184,166,0.25) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "review_agent" not in st.session_state:
        try:
            from agent.multi_agent import get_multi_agent

            st.session_state.review_agent = get_multi_agent()
        except Exception as exc:
            st.session_state.review_agent = None
            st.session_state.review_agent_status = f"error: {exc}"

    if "agent" not in st.session_state:
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_api_key:
            try:
                from agent.react_agent import get_react_agent

                st.session_state.agent = get_react_agent()
                st.session_state.agent_backend = "groq-react"
                st.session_state.agent_status = "connected: Groq ReAct backend"
            except Exception as exc:
                st.session_state.agent = st.session_state.get("review_agent")
                st.session_state.agent_backend = "multi-agent"
                st.session_state.agent_status = f"Groq backend unavailable, using local graph: {exc}"
        else:
            st.session_state.agent = st.session_state.get("review_agent")
            st.session_state.agent_backend = "multi-agent"
            st.session_state.agent_status = "connected: local graph"

        if st.session_state.agent is None:
            fallback_error = st.session_state.get("review_agent_status", "agent initialization failed")
            st.session_state.agent_status = f"error: {fallback_error}"
    if "history" not in st.session_state:
        st.session_state.history = []
    if "profile" not in st.session_state:
        st.session_state.profile = {"age": 25, "conditions": []}
    if "current_risk" not in st.session_state:
        st.session_state.current_risk = None
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    if "pending_review" not in st.session_state:
        st.session_state.pending_review = None


def risk_badge(level: str) -> str:
    icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    return f'<span class="badge-{level}">{icons.get(level, "⚪")} {level} RISK</span>'


def render_trace(trace: List[Dict[str, Any]]) -> None:
    for index, step in enumerate(trace, start=1):
        tools = step.get("tools", [])
        chips = "".join(f"<span class='tool-chip'>{tool}</span>" for tool in tools)
        st.markdown(
            f"""
<div class="trace-step">
    <div class="trace-label">Step {index}: {step.get("node", "node")}</div>
    <div class="trace-content">{step.get("summary", "")}</div>
    <div style="margin-top:0.4rem;">{chips}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_message(message: Dict[str, Any]) -> None:
    role = message["role"]
    content = message["content"]
    meta = message.get("meta", {})
    if role == "user":
        st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="msg-ai">', unsafe_allow_html=True)
    risk_level = meta.get("risk_level")
    if risk_level:
        st.markdown(risk_badge(risk_level), unsafe_allow_html=True)

    if meta.get("is_emergency"):
        st.markdown(
            """
<div class="emergency-banner">
    <strong>Emergency pathway triggered.</strong><br>
    Seek urgent medical attention now and call <strong>911</strong> if symptoms are severe or worsening.
</div>
""",
            unsafe_allow_html=True,
        )

    if meta.get("awaiting_human_review"):
        st.markdown(
            """
<div class="review-banner">
    <strong>Doctor approval required.</strong><br>
    This high-risk triage result is paused until a reviewer approves or rejects it.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(content)

    if meta.get("tool_calls_made"):
        chips = "".join(f"<span class='tool-chip'>{tool}</span>" for tool in meta["tool_calls_made"])
        st.markdown(f"<div style='margin-top:0.7rem;'>{chips}</div>", unsafe_allow_html=True)

    trace = meta.get("workflow_trace", [])
    if trace:
        with st.expander(f"Workflow Trace ({len(trace)} steps)", expanded=False):
            render_trace(trace)

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Smart Triage AI")
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        status = st.session_state.get("agent_status", "unknown")
        if st.session_state.get("agent_backend") == "groq-react":
            st.markdown('<p><span class="status-dot"></span> <strong>Groq ReAct backend connected</strong></p>', unsafe_allow_html=True)
            st.markdown(
                "<p style='color:#6782a7;font-size:0.8rem;'>Primary API-backed triage · local graph retained for review flow</p>",
                unsafe_allow_html=True,
            )
        elif st.session_state.get("agent_backend") == "multi-agent" and st.session_state.get("agent") is not None:
            st.markdown('<p><span class="status-dot"></span> <strong>Multi-agent graph connected</strong></p>', unsafe_allow_html=True)
            st.markdown(
                "<p style='color:#6782a7;font-size:0.8rem;'>Supervisor · Intake · Risk · Safety · HITL · Pinecone-ready</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color:#fca5a5;'><strong>Agent error</strong></p><p style='color:#94a3b8;font-size:0.8rem;'>{status}</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-label'>Patient Profile</p>", unsafe_allow_html=True)

        age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.profile.get("age", 25))
        conditions_text = st.text_area(
            "Known Conditions",
            value=", ".join(st.session_state.profile.get("conditions", [])),
            placeholder="e.g. diabetes, asthma",
            height=90,
        )
        if st.button("Save Profile", use_container_width=True):
            conditions = [item.strip() for item in conditions_text.split(",") if item.strip()]
            st.session_state.profile = {"age": age, "conditions": conditions}
            st.success("Profile saved.")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-label'>Session</p>", unsafe_allow_html=True)

        if st.button("New Conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_risk = None
            st.session_state.pending_review = None
            st.session_state.input_counter += 1
            st.rerun()

        risk_level = st.session_state.get("current_risk")
        if risk_level:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("<p class='sidebar-label'>Current Risk</p>", unsafe_allow_html=True)
            st.markdown(risk_badge(risk_level), unsafe_allow_html=True)

        pending = st.session_state.get("pending_review")
        if pending:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("<p class='sidebar-label'>Review Queue</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:#fde68a;font-size:0.85rem;'>Awaiting doctor approval for {pending.get('risk_level', 'HIGH')} risk case.</p>",
                unsafe_allow_html=True,
            )


def handle_assessment(user_input: str) -> None:
    profile = st.session_state.profile
    history = [{"role": item["role"], "content": item["content"]} for item in st.session_state.history]
    result = st.session_state.agent.process(
        user_input=user_input.strip(),
        user_id="default",
        age=profile.get("age", 25),
        conditions=profile.get("conditions", []),
        history=history,
    )

    st.session_state.current_risk = result.get("risk_level")
    st.session_state.history.append({"role": "user", "content": user_input.strip(), "meta": {}})
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["content"],
            "meta": {
                "risk_level": result.get("risk_level"),
                "is_emergency": result.get("is_emergency", False),
                "awaiting_human_review": result.get("awaiting_human_review", False),
                "workflow_trace": result.get("workflow_trace", []),
                "tool_calls_made": result.get("tool_calls_made", []),
            },
        }
    )

    if result.get("awaiting_human_review"):
        st.session_state.pending_review = {
            "state": result.get("pending_state"),
            "risk_level": result.get("risk_level"),
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

    review_agent = st.session_state.get("review_agent") or st.session_state.agent
    if review_agent is None:
        st.error("The local review agent is unavailable, so the approval flow cannot continue.")
        return

    result = review_agent.process(
        pending_state=pending["state"],
        approval_decision=decision,
        approval_note=note.strip() or None,
    )
    st.session_state.current_risk = result.get("risk_level")
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["content"],
            "meta": {
                "risk_level": result.get("risk_level"),
                "is_emergency": result.get("is_emergency", False),
                "awaiting_human_review": result.get("awaiting_human_review", False),
                "workflow_trace": result.get("workflow_trace", []),
                "tool_calls_made": result.get("tool_calls_made", []),
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
        """
<div class="glass-card">
    <h3 style="margin-bottom:0.4rem;">Doctor Approval Required</h3>
    <p style="color:#dbeafe;margin-bottom:0.4rem;">The workflow is paused after the Safety agent because the case is marked HIGH risk.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    note = st.text_input(
        "Reviewer note",
        placeholder="Optional note for the approval record",
        key="reviewer_note",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve Release", use_container_width=True):
            handle_review("approved", note)
    with col2:
        if st.button("Reject and Escalate", use_container_width=True):
            handle_review("rejected", note)


def main() -> None:
    init_state()
    render_sidebar()

    st.markdown(
        """
<div class="hero">
    <h1>Smart Triage AI</h1>
    <p>Supervisor-routed medical triage with specialist sub-agents, long-term memory, and human review for high-risk cases.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if not os.getenv("GROQ_API_KEY"):
        st.info("`GROQ_API_KEY` is not set. The app will use the local multi-agent graph instead of the Groq API-backed path.")

    if not st.session_state.agent:
        st.error("The triage agent failed to initialize. Check your dependencies and environment configuration.")
        st.stop()

    if st.session_state.history:
        for message in st.session_state.history:
            render_message(message)

    render_review_panel()

    pending_review = st.session_state.get("pending_review") is not None
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Describe Symptoms")
    with st.expander("Example inputs"):
        st.markdown(
            """
- `I have a headache and fever for 2 days`
- `Chest pain and difficulty breathing since this morning`
- `Nausea and stomach ache, feeling really tired`
- `I feel a bit dizzy when I stand up`
"""
        )

    user_input = st.text_area(
        "Symptoms",
        placeholder="Describe the symptoms, duration, and severity...",
        height=120,
        key=f"user_input_{st.session_state.input_counter}",
        label_visibility="collapsed",
        disabled=pending_review,
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("Assess Symptoms", type="primary", use_container_width=True, disabled=pending_review)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.input_counter += 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if pending_review:
        st.warning("The workflow is waiting for doctor approval. Approve or reject the current case before submitting a new one.")

    if submit and user_input.strip():
        with st.spinner("Running intake, risk, safety, and review routing..."):
            try:
                handle_assessment(user_input)
            except Exception as exc:
                st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
