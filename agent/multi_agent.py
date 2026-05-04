import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:  # pragma: no cover - optional dependency at runtime
    Pinecone = None
    ServerlessSpec = None


load_dotenv()
logger = logging.getLogger("multi-agent-triage")

LOCAL_PATIENT_STORE: Dict[str, Dict[str, Any]] = {}


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    user_id: str
    symptoms: List[str]
    duration_days: int
    severity: str
    intake_summary: str
    followup_questions: List[str]
    patient_profile: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    safety_validation: Dict[str, Any]
    final_response: str
    workflow_trace: List[Dict[str, Any]]
    tool_calls_made: List[str]
    completed_steps: List[str]
    next_step: str
    approval_decision: Optional[str]
    approval_note: Optional[str]
    awaiting_human_review: bool
    hitl_required: bool


def _append_trace(state: AgentState, node: str, summary: str, tools: Optional[List[str]] = None) -> None:
    state.setdefault("workflow_trace", []).append(
        {
            "node": node,
            "summary": summary,
            "tools": tools or [],
        }
    )
    if tools:
        state.setdefault("tool_calls_made", []).extend(tools)


def set_patient_context(user_id: str, age: int, conditions: List[str]) -> None:
    LOCAL_PATIENT_STORE[user_id] = {
        "user_id": user_id,
        "age": age,
        "conditions": [item.strip() for item in conditions if item.strip()],
    }


class PineconePatientMemory:
    """Pinecone-backed patient memory with safe local fallback."""

    def __init__(self) -> None:
        self.api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "smart-triage-patient-memory").strip()
        self.dimension = int(os.getenv("PINECONE_DIMENSION", "64"))
        self.cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
        self.region = os.getenv("PINECONE_REGION", "us-east-1").strip()
        self._client = None
        self._index = None
        self.enabled = bool(self.api_key and Pinecone and self.index_name)

    def _embedding(self, text: str) -> List[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values: List[float] = []
        for idx in range(self.dimension):
            byte = seed[idx % len(seed)]
            values.append((byte / 255.0) * 2 - 1)
        return values

    def _get_index(self):
        if not self.enabled:
            return None
        if self._index is not None:
            return self._index
        try:
            self._client = Pinecone(api_key=self.api_key)
            existing = {item["name"] if isinstance(item, dict) else getattr(item, "name", "") for item in self._client.list_indexes()}
            if self.index_name not in existing:
                if ServerlessSpec is None:
                    raise RuntimeError("ServerlessSpec unavailable for Pinecone index creation")
                self._client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
            self._index = self._client.Index(self.index_name)
        except Exception as exc:  # pragma: no cover - depends on external service
            logger.warning("Pinecone unavailable, falling back to local memory: %s", exc)
            self.enabled = False
            self._client = None
            self._index = None
        return self._index

    def upsert_patient_profile(self, user_id: str, age: int, conditions: List[str]) -> None:
        index = self._get_index()
        if index is None:
            return
        metadata = {
            "user_id": user_id,
            "age": age,
            "conditions": conditions,
            "profile_summary": self._profile_summary(age, conditions),
        }
        record = {
            "id": user_id,
            "values": self._embedding(f"{user_id} {age} {' '.join(conditions)}"),
            "metadata": metadata,
        }
        try:
            index.upsert(vectors=[record])
        except Exception as exc:  # pragma: no cover - depends on external service
            logger.warning("Pinecone upsert failed, using local memory: %s", exc)
            self.enabled = False

    def retrieve_patient_profile(self, user_id: str, query_text: str) -> Dict[str, Any]:
        fallback = LOCAL_PATIENT_STORE.get(user_id) or LOCAL_PATIENT_STORE.get(
            "default",
            {"user_id": user_id, "age": 25, "conditions": []},
        )

        profile = {
            "user_id": fallback.get("user_id", user_id),
            "age": fallback.get("age", 25),
            "conditions": fallback.get("conditions", []),
            "risk_factors": [],
            "profile_summary": "",
            "memory_source": "local",
        }
        profile["risk_factors"] = _derive_risk_factors(profile["age"], profile["conditions"])
        profile["profile_summary"] = self._profile_summary(profile["age"], profile["conditions"], profile["risk_factors"])

        index = self._get_index()
        if index is None:
            return profile

        try:
            result = index.query(
                vector=self._embedding(f"{user_id} {query_text}"),
                top_k=1,
                include_metadata=True,
                filter={"user_id": {"$eq": user_id}},
            )
            matches = getattr(result, "matches", None) or result.get("matches", [])
            if not matches:
                return profile
            match = matches[0]
            metadata = getattr(match, "metadata", None) or match.get("metadata", {})
            age = int(metadata.get("age", profile["age"]))
            conditions = metadata.get("conditions", profile["conditions"]) or []
            return {
                "user_id": user_id,
                "age": age,
                "conditions": conditions,
                "risk_factors": _derive_risk_factors(age, conditions),
                "profile_summary": metadata.get(
                    "profile_summary",
                    self._profile_summary(age, conditions),
                ),
                "memory_source": "pinecone",
            }
        except Exception as exc:  # pragma: no cover - depends on external service
            logger.warning("Pinecone retrieval failed, using local memory: %s", exc)
            self.enabled = False
            return profile

    @staticmethod
    def _profile_summary(age: int, conditions: List[str], risk_factors: Optional[List[str]] = None) -> str:
        summary = f"Patient aged {age}"
        summary += f", conditions: {', '.join(conditions)}" if conditions else ", no known conditions"
        if risk_factors:
            summary += f". Risk factors: {', '.join(risk_factors)}"
        return summary


MEMORY_STORE = PineconePatientMemory()


def _derive_risk_factors(age: int, conditions: List[str]) -> List[str]:
    risk_factors: List[str] = []
    if age >= 65:
        risk_factors.append("advanced_age_65+")
    elif age < 18:
        risk_factors.append("pediatric")

    high_risk_conditions = {
        "diabetes",
        "heart disease",
        "coronary artery disease",
        "copd",
        "asthma",
        "immunocompromised",
        "cancer",
        "kidney disease",
        "liver disease",
        "hypertension",
        "high blood pressure",
    }
    for condition in conditions:
        lowered = condition.lower()
        if any(item in lowered for item in high_risk_conditions):
            risk_factors.append(f"condition:{condition}")
    return risk_factors


def extract_symptoms(text: str) -> Dict[str, Any]:
    symptom_patterns = {
        "chest pain": ["chest pain", "chest tightness", "chest pressure", "chest discomfort"],
        "shortness of breath": ["shortness of breath", "difficulty breathing", "breathless", "can't breathe"],
        "headache": ["headache", "head pain", "migraine"],
        "fever": ["fever", "high temperature", "chills", "feverish"],
        "cough": ["cough", "coughing", "phlegm"],
        "nausea": ["nausea", "nauseous", "queasy"],
        "vomiting": ["vomiting", "throwing up", "vomit"],
        "diarrhea": ["diarrhea", "loose stools", "watery stool"],
        "fatigue": ["fatigue", "tired", "exhausted", "weakness", "no energy"],
        "dizziness": ["dizziness", "dizzy", "lightheaded"],
        "abdominal pain": ["abdominal pain", "stomach pain", "belly pain", "stomach ache"],
        "sore throat": ["sore throat", "throat pain"],
        "muscle pain": ["muscle pain", "body ache", "myalgia"],
        "joint pain": ["joint pain", "arthralgia"],
        "confusion": ["confusion", "confused", "disoriented"],
        "fainting": ["fainting", "fainted", "passed out", "syncope"],
        "loss of consciousness": ["loss of consciousness", "unconscious", "unresponsive"],
        "rash": ["rash", "hives", "itchy skin"],
        "swelling": ["swelling", "swollen", "edema"],
    }

    lowered = text.lower()
    found = [name for name, variants in symptom_patterns.items() if any(variant in lowered for variant in variants)]

    duration_days = 0
    for pattern, multiplier in [
        (r"(\d+)\s*months?", 30),
        (r"(\d+)\s*weeks?", 7),
        (r"(\d+)\s*days?", 1),
        (r"(\d+)\s*hours?", 1),
    ]:
        match = re.search(pattern, lowered)
        if match:
            duration_days = max(int(match.group(1)) * multiplier, 1)
            break
    if duration_days == 0:
        if "yesterday" in lowered:
            duration_days = 1
        elif "last week" in lowered:
            duration_days = 7
        elif "today" in lowered or "this morning" in lowered:
            duration_days = 1

    if any(word in lowered for word in ["severe", "extreme", "unbearable", "worst", "excruciating"]):
        severity = "severe"
    elif any(word in lowered for word in ["moderate", "significant", "bad"]):
        severity = "moderate"
    elif any(word in lowered for word in ["mild", "slight", "minor"]):
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


def retrieve_patient_memory(user_id: str, query_text: str) -> Dict[str, Any]:
    return MEMORY_STORE.retrieve_patient_profile(user_id, query_text)


def generate_followup_questions(symptoms: List[str]) -> Dict[str, Any]:
    bank = {
        "headache": [
            "Rate the headache from 1 to 10. Is it the worst headache of your life?",
            "Any neck stiffness, light sensitivity, or vision changes?",
        ],
        "fever": [
            "What is your temperature, and how long have you had it?",
            "Any chills, sweating, or rigors?",
        ],
        "chest pain": [
            "Is the pain sharp, pressure-like, or burning?",
            "Does it radiate to your arm, jaw, or back?",
        ],
        "shortness of breath": [
            "Did it start suddenly or gradually, and does it happen at rest?",
        ],
        "abdominal pain": [
            "Where exactly is the pain located, and is it constant or intermittent?",
        ],
        "cough": [
            "Is the cough dry or producing phlegm, and have you noticed blood?",
        ],
    }
    questions = [
        "How long have you been experiencing these symptoms?",
        "How severe are they: mild, moderate, or severe?",
    ]
    for symptom in symptoms:
        for known_symptom, known_questions in bank.items():
            if known_symptom in symptom.lower():
                questions.extend(known_questions)
                break
    deduped = list(dict.fromkeys(questions))
    return {"questions": deduped[:4], "needs_followup": True}


def assess_medical_risk(symptoms: List[str], duration: int, profile: Dict[str, Any], severity: str) -> Dict[str, Any]:
    lowered = [item.lower() for item in symptoms]
    critical = {"chest pain", "shortness of breath", "confusion", "fainting", "loss of consciousness"}
    minor_short_term = {"headache", "sore throat", "fatigue", "muscle pain", "joint pain"}
    moderate = {
        "fever",
        "cough",
        "nausea",
        "vomiting",
        "diarrhea",
        "dizziness",
        "fatigue",
        "muscle pain",
        "sore throat",
        "abdominal pain",
        "rash",
        "swelling",
    }

    risk_level = "LOW"
    reasons: List[str] = []
    emergency_recommended = False

    for symptom in lowered:
        if any(flag in symptom for flag in critical):
            risk_level = "HIGH"
            emergency_recommended = True
            reasons.append(f"Critical symptom detected: {symptom}")
            break

    if risk_level == "LOW":
        if duration >= 7:
            risk_level = "HIGH"
            reasons.append(f"Prolonged duration: {duration} days")
        elif duration >= 3:
            risk_level = "MEDIUM"
            reasons.append(f"Extended duration: {duration} days")

    if severity == "severe" and risk_level != "HIGH":
        risk_level = "HIGH" if risk_level == "MEDIUM" else "MEDIUM"
        reasons.append("Severe symptom intensity reported")

    risk_factors = profile.get("risk_factors", [])
    if "advanced_age_65+" in risk_factors and risk_level != "HIGH":
        risk_level = "HIGH" if risk_level == "MEDIUM" else "MEDIUM"
        reasons.append("Age 65+ elevates risk")
    if any(item.startswith("condition:") for item in risk_factors) and risk_level != "HIGH":
        risk_level = "HIGH" if risk_level == "MEDIUM" else "MEDIUM"
        reasons.append("Pre-existing high-risk conditions")

    if (
        risk_level == "LOW"
        and lowered
        and severity in {"mild", "unspecified"}
        and 0 < duration <= 2
        and not risk_factors
        and all(symptom in minor_short_term for symptom in lowered)
    ):
        reasons.append("Short-duration mild symptoms without high-risk context")
        return {
            "risk_level": "LOW",
            "emergency_recommended": False,
            "risk_reasons": reasons,
            "possible_concerns": ["Self-limited minor illness"],
        }

    moderate_count = sum(1 for symptom in lowered if any(item in symptom for item in moderate))
    if risk_level != "HIGH":
        if moderate_count >= 3:
            risk_level = "HIGH"
            reasons.append(f"Multiple concerning symptoms ({moderate_count})")
        elif moderate_count >= 1:
            if risk_level == "LOW":
                risk_level = "MEDIUM"
            reasons.append(f"{moderate_count} moderate-risk symptom(s) present")

    if not reasons:
        reasons.append("Symptoms currently appear mild and lower risk")

    possible_concerns: List[str] = []
    if "fever" in lowered and "cough" in lowered:
        possible_concerns.append("Upper respiratory infection")
    if "headache" in lowered and "fever" in lowered:
        possible_concerns.append("Viral illness")
    if "chest pain" in lowered:
        possible_concerns.append("Cardiac or musculoskeletal issue that needs urgent evaluation")
    if "abdominal pain" in lowered and "nausea" in lowered:
        possible_concerns.append("Gastrointestinal issue")
    if not possible_concerns:
        possible_concerns.append("General illness that should be monitored")

    return {
        "risk_level": risk_level,
        "emergency_recommended": emergency_recommended,
        "risk_reasons": reasons,
        "possible_concerns": possible_concerns,
    }


def check_emergency_red_flags(symptoms: List[str]) -> bool:
    red_flags = {
        "chest pain",
        "shortness of breath",
        "difficulty breathing",
        "confusion",
        "fainting",
        "loss of consciousness",
        "severe headache",
        "sudden weakness",
        "face drooping",
        "slurred speech",
        "uncontrolled bleeding",
        "anaphylaxis",
    }
    joined = " ".join(symptoms).lower()
    return any(flag in joined for flag in red_flags)


def perform_safety_check(response: str, is_emergency: bool) -> Dict[str, Any]:
    issues: List[str] = []
    recommendations: List[str] = []
    lowered = response.lower()

    if "consult a qualified healthcare professional" not in lowered:
        issues.append("Missing medical disclaimer")
        recommendations.append("Add a clear informational-use-only medical disclaimer")

    certainty_phrases = [
        "you have ",
        "you are diagnosed",
        "this is definitely",
        "you definitely have",
    ]
    if any(phrase in lowered for phrase in certainty_phrases):
        issues.append("Response makes a definitive diagnosis")
        recommendations.append("Use possibility-based language instead of certainty")

    if is_emergency and not any(flag in lowered for flag in ["911", "emergency room", "call emergency services"]):
        issues.append("Emergency case missing clear escalation guidance")
        recommendations.append("Tell the user to call emergency services or go to the ER now")

    return {
        "safe_to_proceed": not issues,
        "issues": issues,
        "recommendations": recommendations,
    }


def finalize_response(
    risk_level: str,
    symptoms: List[str],
    duration: int,
    concerns: List[str],
    followup_questions: List[str],
    is_emergency: bool,
    profile: Dict[str, Any],
) -> str:
    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk_level, "⚪")
    lines = [f"## {icon} Risk Level: {risk_level}", ""]

    if is_emergency or risk_level == "HIGH":
        lines.extend(
            [
                "### Immediate medical attention required",
                "",
                "Your symptoms suggest a potentially serious condition and should be evaluated right away.",
                "",
                "- Call emergency services (911) immediately.",
                "- Go to the nearest emergency room now.",
                "- Do not drive yourself if you feel unstable or short of breath.",
            ]
        )
    elif risk_level == "MEDIUM":
        lines.extend(
            [
                "### Medical consultation advised",
                "",
                "Your symptoms deserve timely evaluation by a clinician, ideally within the next 24 to 48 hours.",
                "",
                "- Arrange an urgent care or doctor visit.",
                "- Rest, hydrate, and monitor for any worsening symptoms.",
                "- Seek emergency care sooner if your symptoms escalate.",
            ]
        )
    else:
        lines.extend(
            [
                "### Home care may be reasonable",
                "",
                "Your symptoms appear lower risk based on the details available right now.",
                "",
                "- Rest and stay hydrated.",
                "- Monitor for new or worsening symptoms.",
                "- Schedule routine medical follow-up if symptoms persist.",
            ]
        )

    if symptoms:
        lines.extend(["", f"**Reported symptoms:** {', '.join(symptoms)}"])
    if duration:
        lines.append(f"**Duration:** {duration} day(s)")
    if profile.get("conditions"):
        lines.append(f"**Known conditions considered:** {', '.join(profile['conditions'])}")
    if concerns:
        lines.extend(["", "**Possible concerns** *(not a diagnosis)*:"])
        lines.extend([f"- {item}" for item in concerns])
    if followup_questions and risk_level != "HIGH":
        lines.extend(["", "**Helpful follow-up questions for a clinician:**"])
        lines.extend([f"- {item}" for item in followup_questions])

    lines.extend(
        [
            "",
            "---",
            "Consult a qualified healthcare professional for diagnosis and treatment. This assistant is for informational purposes only.",
        ]
    )
    return "\n".join(lines)


def _latest_user_message(messages: List[Dict[str, str]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "user":
            return item.get("content", "")
    return ""


def supervisor_node(state: AgentState) -> AgentState:
    completed = set(state.get("completed_steps", []))
    next_step = "intake"
    if "intake" not in completed:
        next_step = "intake"
    elif "risk" not in completed:
        next_step = "risk"
    elif "safety" not in completed:
        next_step = "safety"
    elif state.get("hitl_required"):
        next_step = "human_review"
    else:
        next_step = "finish"

    updated = dict(state)
    updated["next_step"] = next_step
    _append_trace(updated, "supervisor", f"Routing to {next_step}", [])
    return updated


def intake_node(state: AgentState) -> AgentState:
    updated = dict(state)
    latest_input = _latest_user_message(updated.get("messages", []))
    extracted = extract_symptoms(latest_input)
    profile = retrieve_patient_memory(updated["user_id"], latest_input)
    followups: List[str] = []
    if not extracted["has_symptoms"] or not extracted["has_duration"]:
        followups = generate_followup_questions(extracted["symptoms"])["questions"]

    updated["symptoms"] = extracted["symptoms"]
    updated["duration_days"] = extracted["duration_days"]
    updated["severity"] = extracted["severity"]
    updated["patient_profile"] = profile
    updated["followup_questions"] = followups
    updated["intake_summary"] = (
        f"Identified symptoms: {', '.join(extracted['symptoms']) or 'none'}; "
        f"duration: {extracted['duration_days']} day(s); memory source: {profile.get('memory_source', 'local')}"
    )
    updated.setdefault("completed_steps", []).append("intake")
    _append_trace(
        updated,
        "intake",
        updated["intake_summary"],
        ["extract_symptoms", "retrieve_patient_memory"] + (["generate_followup_questions"] if followups else []),
    )
    return updated


def risk_node(state: AgentState) -> AgentState:
    updated = dict(state)
    assessment = assess_medical_risk(
        updated.get("symptoms", []),
        updated.get("duration_days", 0),
        updated.get("patient_profile", {}),
        updated.get("severity", "unspecified"),
    )
    emergency = check_emergency_red_flags(updated.get("symptoms", [])) or assessment["emergency_recommended"]
    assessment["is_emergency"] = emergency
    updated["risk_assessment"] = assessment
    updated["hitl_required"] = assessment["risk_level"] == "HIGH"
    updated["awaiting_human_review"] = False
    updated.setdefault("completed_steps", []).append("risk")
    _append_trace(
        updated,
        "risk",
        f"Risk assessed as {assessment['risk_level']}",
        ["assess_medical_risk", "check_emergency_red_flags"],
    )
    return updated


def safety_node(state: AgentState) -> AgentState:
    updated = dict(state)
    assessment = updated.get("risk_assessment", {})
    response = finalize_response(
        risk_level=assessment.get("risk_level", "LOW"),
        symptoms=updated.get("symptoms", []),
        duration=updated.get("duration_days", 0),
        concerns=assessment.get("possible_concerns", []),
        followup_questions=updated.get("followup_questions", []),
        is_emergency=assessment.get("is_emergency", False),
        profile=updated.get("patient_profile", {}),
    )
    safety = perform_safety_check(response, assessment.get("is_emergency", False))
    if not safety["safe_to_proceed"]:
        response += "\n\nSafety note: This guidance was adjusted to stay informational and non-diagnostic."

    if updated.get("hitl_required"):
        response += "\n\nDoctor approval is required before this high-risk triage result is considered complete."

    updated["final_response"] = response
    updated["safety_validation"] = safety
    updated.setdefault("completed_steps", []).append("safety")
    _append_trace(
        updated,
        "safety",
        "Final response drafted and safety-checked",
        ["finalize_response", "perform_safety_check"],
    )
    return updated


def human_review_node(state: AgentState) -> AgentState:
    updated = dict(state)
    decision = (updated.get("approval_decision") or "").strip().lower()
    if decision not in {"approved", "rejected"}:
        updated["awaiting_human_review"] = True
        updated["hitl_required"] = True
        _append_trace(
            updated,
            "human_review",
            "Execution paused for doctor approval",
            [],
        )
        return updated

    updated["awaiting_human_review"] = False
    updated["hitl_required"] = False
    note = updated.get("approval_note", "").strip()
    if decision == "approved":
        approval_line = "Doctor approval recorded. High-risk guidance has been cleared for release."
    else:
        approval_line = "Doctor review rejected the automated triage result. Immediate clinician follow-up is required."

    if note:
        approval_line += f" Note: {note}"

    updated["final_response"] = f"{updated.get('final_response', '')}\n\n---\n{approval_line}".strip()
    updated.setdefault("completed_steps", []).append("human_review")
    _append_trace(updated, "human_review", approval_line, [])
    return updated


def finish_node(state: AgentState) -> AgentState:
    updated = dict(state)
    _append_trace(updated, "finish", "Workflow complete", [])
    return updated


def _route_from_supervisor(state: AgentState) -> Literal["intake", "risk", "safety", "human_review", "finish"]:
    return state["next_step"]  # type: ignore[return-value]


def build_multi_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("intake", intake_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("safety", safety_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finish", finish_node)
    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "intake": "intake",
            "risk": "risk",
            "safety": "safety",
            "human_review": "human_review",
            "finish": "finish",
        },
    )
    workflow.add_edge("intake", "supervisor")
    workflow.add_edge("risk", "supervisor")
    workflow.add_edge("safety", "supervisor")
    workflow.add_edge("finish", END)
    workflow.add_edge("human_review", END)
    return workflow.compile()


class MultiAgentSystem:
    def __init__(self) -> None:
        self.graph = build_multi_agent()

    def process(
        self,
        user_input: Optional[str] = None,
        user_id: str = "default",
        age: int = 25,
        conditions: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        pending_state: Optional[Dict[str, Any]] = None,
        approval_decision: Optional[str] = None,
        approval_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        conditions = conditions or []
        history = history or []

        if pending_state:
            state = deepcopy(pending_state)
            state["approval_decision"] = approval_decision
            state["approval_note"] = approval_note
        else:
            set_patient_context(user_id, age, conditions)
            MEMORY_STORE.upsert_patient_profile(user_id, age, conditions)
            messages = history + ([{"role": "user", "content": user_input.strip()}] if user_input else [])
            state = {
                "messages": messages,
                "user_id": user_id,
                "symptoms": [],
                "duration_days": 0,
                "severity": "unspecified",
                "intake_summary": "",
                "followup_questions": [],
                "patient_profile": {},
                "risk_assessment": {},
                "safety_validation": {},
                "final_response": "",
                "workflow_trace": [],
                "tool_calls_made": [],
                "completed_steps": [],
                "next_step": "intake",
                "approval_decision": None,
                "approval_note": None,
                "awaiting_human_review": False,
                "hitl_required": False,
            }

        final_state = self.graph.invoke(state, config={"recursion_limit": 20})
        assessment = final_state.get("risk_assessment", {})
        pending = None
        if final_state.get("awaiting_human_review"):
            pending = deepcopy(final_state)

        return {
            "content": final_state.get("final_response", "No response generated."),
            "risk_level": assessment.get("risk_level", "LOW"),
            "is_emergency": assessment.get("is_emergency", False),
            "hitl_required": final_state.get("hitl_required", False),
            "awaiting_human_review": final_state.get("awaiting_human_review", False),
            "workflow_trace": final_state.get("workflow_trace", []),
            "tool_calls_made": final_state.get("tool_calls_made", []),
            "patient_profile": final_state.get("patient_profile", {}),
            "followup_questions": final_state.get("followup_questions", []),
            "pending_state": pending,
        }


def get_multi_agent() -> MultiAgentSystem:
    return MultiAgentSystem()
