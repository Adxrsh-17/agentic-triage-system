"""
ReAct Medical Triage AI Agent
Implements all 7 spec-compliant tools + true ReAct loop via LangGraph create_react_agent
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()
logger = logging.getLogger("react-triage-agent")

# ── Patient context store (injected before each run) ─────────────────────────
_patient_store: Dict[str, Dict] = {}


def set_patient_context(user_id: str, age: int, conditions: List[str]) -> None:
    """Inject patient profile before running the agent."""
    _patient_store[user_id] = {
        "age": age,
        "conditions": [c.strip() for c in conditions if c.strip()],
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — extract_symptoms
# ══════════════════════════════════════════════════════════════════════════════
@tool
def extract_symptoms(text: str) -> Dict[str, Any]:
    """
    Extract structured symptoms and duration from the user's free-text input.
    Call this FIRST to understand what the user is reporting.
    Returns: symptoms (list), duration_days (int), severity (str), has_symptoms (bool).
    """
    symptom_patterns = {
        "chest pain":          ["chest pain", "chest tightness", "chest pressure", "chest discomfort"],
        "shortness of breath": ["shortness of breath", "difficulty breathing", "breathless", "can't breathe"],
        "headache":            ["headache", "head pain", "migraine"],
        "fever":               ["fever", "high temperature", "chills", "feverish"],
        "cough":               ["cough", "coughing", "phlegm"],
        "nausea":              ["nausea", "nauseous", "queasy"],
        "vomiting":            ["vomiting", "throwing up", "vomit"],
        "diarrhea":            ["diarrhea", "loose stools", "watery stool"],
        "fatigue":             ["fatigue", "tired", "exhausted", "weakness", "no energy"],
        "dizziness":           ["dizziness", "dizzy", "lightheaded"],
        "abdominal pain":      ["abdominal pain", "stomach pain", "belly pain", "stomach ache"],
        "sore throat":         ["sore throat", "throat pain"],
        "muscle pain":         ["muscle pain", "body ache", "myalgia"],
        "joint pain":          ["joint pain", "arthralgia"],
        "confusion":           ["confusion", "confused", "disoriented"],
        "fainting":            ["fainting", "fainted", "passed out", "syncope"],
        "loss of consciousness": ["loss of consciousness", "unconscious", "unresponsive"],
        "rash":                ["rash", "hives", "itchy skin"],
        "swelling":            ["swelling", "swollen", "edema"],
    }

    text_lower = text.lower()
    found = [name for name, pats in symptom_patterns.items()
             if any(p in text_lower for p in pats)]

    # Duration extraction
    duration_days = 0
    for pattern, mult in [(r'(\d+)\s*months?', 30), (r'(\d+)\s*weeks?', 7),
                          (r'(\d+)\s*days?', 1), (r'(\d+)\s*hours?', 0)]:
        m = re.search(pattern, text_lower)
        if m:
            n = int(m.group(1))
            duration_days = n * mult if mult > 0 else 1
            break
    if duration_days == 0:
        if "yesterday" in text_lower:  duration_days = 1
        elif "last week" in text_lower: duration_days = 7
        elif "today" in text_lower or "this morning" in text_lower: duration_days = 1

    # Severity
    if any(w in text_lower for w in ["severe", "extreme", "unbearable", "worst", "excruciating"]):
        severity = "severe"
    elif any(w in text_lower for w in ["moderate", "significant", "bad"]):
        severity = "moderate"
    elif any(w in text_lower for w in ["mild", "slight", "minor"]):
        severity = "mild"
    else:
        severity = "unspecified"

    return {
        "symptoms": found,
        "duration_days": duration_days,
        "severity": severity,
        "has_symptoms": bool(found),
        "has_duration": duration_days > 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — retrieve_patient_memory
# ══════════════════════════════════════════════════════════════════════════════
@tool
def retrieve_patient_memory(user_id: str) -> Dict[str, Any]:
    """
    Retrieve the patient's medical profile: age, known conditions, and risk factors.
    Call this after extracting symptoms to add patient context.
    Use user_id='default' for the current session user.
    Returns: age, conditions, risk_factors, profile_summary.
    """
    profile = _patient_store.get(user_id) or _patient_store.get("default", {"age": 25, "conditions": []})
    age = profile.get("age", 25)
    conditions = profile.get("conditions", [])

    risk_factors = []
    if age >= 65:
        risk_factors.append("advanced_age_65+")
    elif age < 18:
        risk_factors.append("pediatric")

    high_risk = {"diabetes", "heart disease", "coronary artery disease", "copd",
                 "asthma", "immunocompromised", "cancer", "kidney disease",
                 "liver disease", "hypertension", "high blood pressure"}
    for cond in conditions:
        if any(hr in cond.lower() for hr in high_risk):
            risk_factors.append(f"condition:{cond}")

    summary = f"Patient aged {age}"
    summary += f", conditions: {', '.join(conditions)}" if conditions else ", no known conditions"
    if risk_factors:
        summary += f". Risk factors: {', '.join(risk_factors)}"

    return {
        "age": age,
        "conditions": conditions,
        "risk_factors": risk_factors,
        "profile_summary": summary,
        "has_high_risk_conditions": any("condition:" in rf for rf in risk_factors),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — generate_followup_questions
# ══════════════════════════════════════════════════════════════════════════════
@tool
def generate_followup_questions(symptoms: List[str]) -> Dict[str, Any]:
    """
    Generate targeted clinical follow-up questions when symptom information is vague or incomplete.
    Call this when symptoms are unclear or duration is unknown.
    Returns: questions (list of strings), needs_followup (bool).
    """
    bank = {
        "headache":            ["Rate the headache 1–10. Is it the worst headache of your life?",
                                "Any neck stiffness, light sensitivity, or vision changes?"],
        "fever":               ["What is your temperature? How long have you had it?",
                                "Any chills, sweating, or rigors?"],
        "chest pain":          ["Is the pain sharp, pressure-like, or burning?",
                                "Does it radiate to your arm, jaw, or back?"],
        "shortness of breath": ["Is it sudden or gradual? At rest or only with exertion?"],
        "abdominal pain":      ["Where exactly — upper, lower, left, right?",
                                "Is it constant or comes and goes?"],
        "cough":               ["Dry cough or producing phlegm? Any blood?"],
        "dizziness":           ["Is it a spinning sensation (vertigo) or feeling faint?"],
        "fatigue":             ["Can you perform normal daily activities?"],
        "rash":                ["Where is the rash? Itchy, painful, or neither?"],
    }

    questions = [
        "How long have you been experiencing these symptoms?",
        "How severe are they: mild, moderate, or severe?",
    ]
    for symptom in symptoms:
        for key, qs in bank.items():
            if key in symptom.lower():
                questions.extend(qs)
                break

    # deduplicate & limit
    seen, unique = set(), []
    for q in questions:
        if q not in seen:
            seen.add(q); unique.append(q)

    return {
        "questions": unique[:4],
        "needs_followup": True,
        "question_count": len(unique[:4]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — assess_medical_risk
# ══════════════════════════════════════════════════════════════════════════════
@tool
def assess_medical_risk(symptoms: List[str], duration: int, memory: str) -> Dict[str, Any]:
    """
    Comprehensive medical risk assessment.
    Returns risk_level: LOW / MEDIUM / HIGH with detailed reasoning.
    symptoms: list of symptom strings.
    duration: duration in days (0 if unknown).
    memory: patient context summary string from retrieve_patient_memory.
    """
    sl = [s.lower() for s in symptoms]
    critical = {"chest pain", "shortness of breath", "confusion", "fainting", "loss of consciousness"}
    medium_set = {"fever", "cough", "nausea", "vomiting", "diarrhea", "dizziness",
                  "fatigue", "muscle pain", "sore throat", "abdominal pain", "rash", "swelling"}

    risk = "LOW"
    reasons = []
    emergency = False

    # Critical symptoms → HIGH
    for s in sl:
        if any(c in s for c in critical):
            risk = "HIGH"; emergency = True
            reasons.append(f"Critical symptom detected: {s}")
            break

    # Duration escalation
    if risk == "LOW":
        if duration >= 7:
            risk = "HIGH"; reasons.append(f"Prolonged duration: {duration} days")
        elif duration >= 3:
            risk = "MEDIUM"; reasons.append(f"Extended duration: {duration} days")

    # Age / conditions from memory string
    mem = memory.lower()
    if "advanced_age_65+" in mem and risk != "HIGH":
        risk = "HIGH" if risk == "MEDIUM" else "MEDIUM"
        reasons.append("Age 65+ elevates risk")
    if ("condition:" in mem or "has_high_risk_conditions" in mem) and risk != "HIGH":
        risk = "HIGH" if risk == "MEDIUM" else "MEDIUM"
        reasons.append("Pre-existing high-risk conditions")

    # Medium symptom count
    mc = sum(1 for s in sl if any(m in s for m in medium_set))
    if risk not in ("HIGH",):
        if mc >= 3:
            risk = "HIGH"; reasons.append(f"Multiple concerning symptoms ({mc})")
        elif mc >= 1:
            if risk == "LOW":
                risk = "MEDIUM"
            reasons.append(f"{mc} moderate-risk symptom(s) present")

    if not reasons:
        reasons.append("Symptoms appear mild and low-risk")

    # Possible concerns (educational, not diagnostic)
    concerns = []
    if "fever" in sl and "cough" in sl:         concerns.append("Upper respiratory infection")
    if "headache" in sl and "fever" in sl:       concerns.append("Viral illness")
    if "chest pain" in sl:                       concerns.append("Cardiac or musculoskeletal issue (needs urgent evaluation)")
    if "abdominal pain" in sl and "nausea" in sl: concerns.append("Gastrointestinal issue")
    if not concerns:                              concerns.append("General illness — evaluation needed")

    return {
        "risk_level": risk,
        "emergency_recommended": emergency,
        "risk_reasons": reasons,
        "possible_concerns": concerns,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — check_emergency_red_flags
# ══════════════════════════════════════════════════════════════════════════════
@tool
def check_emergency_red_flags(symptoms: List[str]) -> bool:
    """
    Check if any symptoms are emergency red flags requiring IMMEDIATE medical attention.
    Returns True if this is a medical emergency — stop reasoning and escalate immediately.
    Always call this for HIGH risk cases before finalizing a response.
    """
    red_flags = {
        "chest pain", "shortness of breath", "difficulty breathing",
        "confusion", "fainting", "loss of consciousness",
        "severe headache", "sudden weakness", "face drooping",
        "slurred speech", "uncontrolled bleeding", "anaphylaxis",
    }
    sl = " ".join(s.lower() for s in symptoms)
    return any(flag in sl for flag in red_flags)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — perform_safety_check
# ══════════════════════════════════════════════════════════════════════════════
@tool
def perform_safety_check(response: str) -> Dict[str, Any]:
    """
    Validate medical safety of a draft response before sending it to the user.
    Checks for: missing disclaimer, inappropriate certainty, missing emergency referral.
    Always call this BEFORE finalize_response.
    Returns: safe_to_proceed (bool), issues (list), recommendations (list).
    """
    issues = []
    recommendations = []
    rl = response.lower()

    # Must have disclaimer
    if "disclaimer" not in rl and "not a substitute" not in rl and "consult" not in rl:
        issues.append("Missing medical disclaimer")
        recommendations.append("Add: 'Consult a qualified healthcare professional for diagnosis.'")

    # Must not claim certainty
    certainty_words = ["you have ", "you are diagnosed", "this is definitely", "you definitely have"]
    if any(w in rl for w in certainty_words):
        issues.append("Response makes definitive diagnostic claim")
        recommendations.append("Replace with: 'This could possibly indicate…'")

    # Emergency cases must have referral
    emergency_keywords = ["chest pain", "shortness of breath", "confusion", "fainting"]
    has_emergency = any(k in rl for k in emergency_keywords)
    if has_emergency and "emergency" not in rl and "911" not in rl and "emergency room" not in rl:
        issues.append("Emergency symptoms present but no emergency referral")
        recommendations.append("Add instruction to call emergency services / go to ER")

    return {
        "safe_to_proceed": len(issues) == 0,
        "issues": issues,
        "recommendations": recommendations,
        "requires_revision": len(issues) > 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — finalize_response
# ══════════════════════════════════════════════════════════════════════════════
@tool
def finalize_response(context: str) -> str:
    """
    Generate the final structured medical triage response.
    context: a plain-text or JSON string describing risk_level, symptoms, concerns,
             is_emergency, and any followup_questions to include.
    Call this as the LAST step after all assessments and safety checks are done.
    Returns a formatted markdown response.
    """
    # Try to parse as JSON, fall back to raw context
    try:
        data = json.loads(context)
    except Exception:
        data = {}

    risk = data.get("risk_level", "LOW")
    symptoms = data.get("symptoms", [])
    concerns = data.get("possible_concerns", [])
    is_emergency = data.get("is_emergency", False)
    followup = data.get("followup_questions", [])
    duration = data.get("duration", 0)

    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")

    lines = [f"## {icon} Risk Level: {risk}\n"]

    if is_emergency or risk == "HIGH":
        lines += [
            "### 🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED",
            "",
            "Your symptoms suggest a potentially serious condition. **Do not wait.**",
            "",
            "**Recommended Actions:**",
            "- 🆘 Call emergency services (911) immediately",
            "- 🏥 Go to the nearest Emergency Room",
            "- 🚫 Do not drive yourself — have someone take you or call an ambulance",
            "- ⏰ Do not wait for symptoms to improve on their own",
        ]
    elif risk == "MEDIUM":
        lines += [
            "### ⚠️ Medical Consultation Advised",
            "",
            "Your symptoms warrant evaluation by a healthcare provider within the next 24–48 hours.",
            "",
            "**Recommended Actions:**",
            "- 📅 Schedule an urgent care or doctor's appointment",
            "- 💊 You may take OTC medications for symptom relief if appropriate",
            "- 💧 Rest, stay hydrated, and monitor your symptoms",
            "- 🔔 Go to the ER if symptoms worsen significantly",
        ]
    else:
        lines += [
            "### ✅ Home Care Recommended",
            "",
            "Your symptoms appear to be mild at this time.",
            "",
            "**Recommended Actions:**",
            "- 🛌 Get adequate rest",
            "- 💧 Stay well hydrated",
            "- 🌡️ Monitor your symptoms — seek care if they worsen",
            "- 💊 OTC medications may help with symptom relief",
        ]

    if symptoms:
        lines += ["", "**Reported Symptoms:**", ", ".join(f"`{s}`" for s in symptoms)]

    if duration:
        lines += [f"\n**Duration:** {duration} day(s)"]

    if concerns:
        lines += ["", "**Possible Concerns** *(not a diagnosis)*:"]
        lines += [f"- {c}" for c in concerns]

    if followup:
        lines += ["", "**Follow-up Questions for Your Doctor:**"]
        lines += [f"- {q}" for q in followup]

    lines += [
        "",
        "---",
        "> ⚕️ **Medical Disclaimer:** This AI is for informational purposes only and does "
        "not replace professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare professional for medical concerns.",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an autonomous medical triage AI agent following a strict ReAct reasoning loop.

## YOUR MISSION
Safely assess user symptoms by thinking carefully, dynamically selecting tools, and providing medically responsible guidance.

## AVAILABLE TOOLS
1. extract_symptoms(text) — Extract structured symptoms from user input
2. retrieve_patient_memory(user_id) — Get patient age, conditions, risk history
3. generate_followup_questions(symptoms) — Ask clarifying questions if info is missing
4. assess_medical_risk(symptoms, duration, memory) — Returns LOW / MEDIUM / HIGH
5. check_emergency_red_flags(symptoms) — Returns True if emergency
6. perform_safety_check(response) — Validates medical safety of a draft
7. finalize_response(context) — Generate final structured answer

## REACT LOOP (MANDATORY)
You MUST follow: THOUGHT → ACTION → OBSERVATION → repeat.
- Think before each tool call
- Call exactly ONE tool at a time
- Observe and update understanding after each result
- Maximum 5 iterations

## TOOL SELECTION RULES
- ALWAYS start with: extract_symptoms → retrieve_patient_memory
- If symptoms are vague or duration unknown → generate_followup_questions
- When you have symptoms + patient context → assess_medical_risk
- If risk is HIGH or danger suspected → check_emergency_red_flags
- ALWAYS before final answer → perform_safety_check
- LAST step ALWAYS → finalize_response

## finalize_response FORMAT
Pass a JSON string as context, example:
{
  "risk_level": "MEDIUM",
  "symptoms": ["fever", "cough"],
  "duration": 2,
  "possible_concerns": ["Upper respiratory infection"],
  "is_emergency": false,
  "followup_questions": []
}

## SAFETY RULES
- NEVER diagnose with certainty — always say "possibly" or "may indicate"
- ALWAYS include medical disclaimer in the final response
- ALWAYS recommend consulting a doctor
- For emergencies: explicitly say call 911 / go to ER
- Prioritize safety over completeness

## STOP CONDITIONS
Stop the loop when:
- You have extracted symptoms, assessed risk, run safety check, and called finalize_response
- OR emergency is detected (check_emergency_red_flags returned True)
- OR 5 iterations reached
"""


# ══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════
TOOLS = [
    extract_symptoms,
    retrieve_patient_memory,
    generate_followup_questions,
    assess_medical_risk,
    check_emergency_red_flags,
    perform_safety_check,
    finalize_response,
]


class ReactTriageAgent:
    """ReAct Medical Triage Agent using LangGraph create_react_agent."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=api_key,
        )
        self.agent = create_react_agent(
            self.llm,
            TOOLS,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    def process(
        self,
        user_input: str,
        user_id: str = "default",
        age: int = 25,
        conditions: Optional[List[str]] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Run the ReAct agent and return structured result with trace."""

        # Inject patient context
        set_patient_context(user_id, age, conditions or [])

        # Build messages
        messages = []
        for msg in (history or []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))

        # Run agent
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 25},
        )

        msgs = result["messages"]
        final_content = msgs[-1].content if msgs else "No response generated."

        # ── Parse ReAct trace ────────────────────────────────────────────────
        trace = []
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                thought = msg.content or ""
                for tc in msg.tool_calls:
                    step = {
                        "thought": thought,
                        "action": tc["name"],
                        "action_input": tc["args"],
                        "observation": None,
                    }
                    # Look ahead for ToolMessage(s)
                    j = i + 1
                    while j < len(msgs) and isinstance(msgs[j], ToolMessage):
                        if msgs[j].tool_call_id == tc.get("id", ""):
                            step["observation"] = msgs[j].content
                            break
                        j += 1
                    trace.append(step)
            i += 1

        # ── Extract data directly from tool observation outputs ──────────────
        risk_level = None
        is_emergency = False
        finalize_output = None

        for step in trace:
            obs = step.get("observation") or ""
            action = step.get("action", "")

            if action == "assess_medical_risk" and obs:
                try:
                    data = json.loads(obs) if obs.strip().startswith("{") else {}
                    rl = data.get("risk_level")
                    if rl in ("HIGH", "MEDIUM", "LOW"):
                        risk_level = rl
                    if data.get("emergency_recommended"):
                        is_emergency = True
                except Exception:
                    for lvl in ("HIGH", "MEDIUM", "LOW"):
                        if lvl in obs.upper():
                            risk_level = lvl
                            break

            if action == "check_emergency_red_flags" and obs:
                if obs.strip().lower() in ("true",):
                    is_emergency = True
                    risk_level = "HIGH"

            if action == "finalize_response" and obs:
                finalize_output = obs

        # Use formatted finalize_response output as the display content
        if finalize_output:
            final_content = finalize_output

        # Fallback: parse risk from final_content text if still None
        if risk_level is None:
            fu = final_content.upper()
            if "RISK LEVEL: HIGH" in fu or "\U0001f534" in final_content:
                risk_level = "HIGH"
            elif "RISK LEVEL: MEDIUM" in fu or "\U0001f7e1" in final_content:
                risk_level = "MEDIUM"
            elif "RISK LEVEL: LOW" in fu or "\U0001f7e2" in final_content:
                risk_level = "LOW"
            else:
                # Last fallback: check the LLM summary text
                fu_lower = final_content.lower()
                if "high risk" in fu_lower or "high" in fu_lower:
                    risk_level = "HIGH"
                elif "medium risk" in fu_lower or "medium" in fu_lower:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"

        if risk_level == "HIGH":
            is_emergency = True

        return {
            "content": final_content,
            "risk_level": risk_level,
            "is_emergency": is_emergency,
            "react_trace": trace,
            "tool_calls_made": [s["action"] for s in trace],
            "iteration_count": len(trace),
        }


def get_react_agent() -> ReactTriageAgent:
    return ReactTriageAgent()
