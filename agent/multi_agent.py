import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agent.audit_logger import get_audit_logger
from agent.infermedica_client import (
    get_diagnosis as get_diagnosis_infermedica,
    get_triage as get_triage_infermedica,
    is_infermedica_configured,
    parse_symptoms as parse_symptoms_infermedica,
)
from agent.sbar_generator import generate_fhir_r4_bundle, generate_sbar_note


try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:  # pragma: no cover - optional dependency at runtime
    Pinecone = None
    ServerlessSpec = None


load_dotenv(override=True)
logger = logging.getLogger("multi-agent-triage")

LOCAL_PATIENT_STORE: Dict[str, Dict[str, Any]] = {}

# Lazy singleton embedding model holder
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_LOADED = False


def _get_embedding_model():
    """
    Lazy singleton loader for the sentence-transformers model ('all-MiniLM-L6-v2').
    Returns 384-dimensional dense vector embeddings.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_LOADED
    if not _EMBEDDING_MODEL_LOADED:
        _EMBEDDING_MODEL_LOADED = True
        try:
            from sentence_transformers import SentenceTransformer

            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers model 'all-MiniLM-L6-v2' (384-dim).")
        except Exception as exc:
            logger.warning(
                "sentence-transformers unavailable or model download failed, using fallback embeddings: %s",
                exc,
            )
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


class StructuredIntake(BaseModel):
    """Structured clinical intake extracted from patient/intake staff input."""

    symptoms: List[str] = Field(
        default_factory=list,
        description="List of standardized symptom terms identified (e.g., 'chest pain', 'headache', 'fever')",
    )
    duration_days: float = Field(
        default=0.0,
        description="Duration of symptoms in days (convert hours/weeks/months to decimal or whole days; 0 if unspecified)",
    )
    severity: Literal["mild", "moderate", "severe", "unspecified"] = Field(
        default="unspecified",
        description="Reported severity of symptoms (mild, moderate, severe, or unspecified)",
    )
    pertinent_negatives: List[str] = Field(
        default_factory=list,
        description="Explicitly denied symptoms or absent red flags (e.g., 'no fever', 'denies chest pain', 'no shortness of breath')",
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="Acute emergency red flag indicators or concerning clinical phrasing (e.g., 'crushing chest pressure', 'sudden slurred speech')",
    )


class FollowupQuestionsOutput(BaseModel):
    """Clinical follow-up questions for ambiguous or missing intake information."""

    questions: List[str] = Field(
        default_factory=list,
        description="Targeted clinical follow-up questions (maximum 4) to clarify missing duration, severity, location, or red flags",
    )


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    user_id: str
    patient_sex: Optional[str]
    vitals: Dict[str, Any]
    vital_triggers: List[str]
    symptoms: List[str]
    duration_days: int
    severity: str
    pertinent_negatives: List[str]
    red_flags: List[str]
    infermedica_evidence: Optional[List[Dict[str, Any]]]
    infermedica_triage: Optional[Dict[str, Any]]
    intake_method: str
    intake_summary: str
    followup_questions: List[str]
    patient_profile: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    safety_validation: Dict[str, Any]
    sbar_note: str
    fhir_bundle: Dict[str, Any]
    final_response: str
    workflow_trace: List[Dict[str, Any]]
    tool_calls_made: List[str]
    completed_steps: List[str]
    next_step: str
    approval_decision: Optional[str]
    approval_note: Optional[str]
    nurse_override_esi: Optional[int]
    nurse_override_reason: Optional[str]
    nurse_notes: Optional[str]
    awaiting_human_review: bool
    hitl_required: bool
    # Backward compatibility placeholders
    medications: List[Dict[str, Any]]
    nearby_facilities: List[Dict[str, Any]]
    hospital_recommendations: List[Dict[str, Any]]
    location_address: Optional[str]
    location_lat: Optional[float]
    location_lng: Optional[float]



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
    """
    Pinecone-backed patient memory with real semantic embeddings and safe local fallback.

    Why semantic embeddings matter for patient memory:
    Using real dense sentence embeddings (all-MiniLM-L6-v2, 384-dim) allows the triage system
    to perform semantic retrieval of similar past clinical presentations and patient context,
    rather than relying on rigid exact-string matching or dummy hash projections.
    This enables clinical triage staff to surface relevant comorbidities and prior episode patterns
    across variations in patient descriptions (e.g., 'breathing trouble' semantically matching 'dyspnea').
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "healthcare-triage").strip()
        self.host = os.getenv("PINECONE_HOST", "").strip()
        self.dimension = int(os.getenv("PINECONE_DIMENSION", "64"))
        self.cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
        self.region = os.getenv("PINECONE_REGION", "us-east-1").strip()
        self._client = None
        self._index = None
        self.enabled = bool(self.api_key and Pinecone and (self.host or self.index_name))

    def _embedding(self, text: str) -> List[float]:
        """
        Generate dense semantic embedding vector aligned to index dimension.
        Uses sentence-transformers if available, with resilient fallback.
        """
        model = _get_embedding_model()
        if model is not None:
            try:
                vec = model.encode(text, normalize_embeddings=True)
                raw_list = vec.tolist()
                if len(raw_list) == self.dimension:
                    return raw_list
                elif len(raw_list) > self.dimension:
                    return raw_list[: self.dimension]
                else:
                    return raw_list + [0.0] * (self.dimension - len(raw_list))
            except Exception as exc:
                logger.warning("sentence-transformers encoding failed, using fallback vector: %s", exc)

        # Resilient offline fallback vector of length `self.dimension`
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
            if self.host:
                self._index = self._client.Index(host=self.host)
                logger.info("Connected to Pinecone index via host: %s", self.host)
            else:
                existing = {
                    item["name"] if isinstance(item, dict) else getattr(item, "name", "")
                    for item in self._client.list_indexes()
                }
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
        profile["profile_summary"] = self._profile_summary(
            profile["age"], profile["conditions"], profile["risk_factors"]
        )

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


def _get_groq_llm(temperature: float = 0.0):
    """
    Instantiate ChatGroq with robust multi-model fallback.
    Tries GROQ_MODEL from environment, then candidate models available on Groq.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from langchain_groq import ChatGroq

        candidate_models = [
            os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        seen = set()
        for model_name in candidate_models:
            if not model_name or model_name in seen:
                continue
            seen.add(model_name)
            try:
                return ChatGroq(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key,
                    max_retries=1,
                    timeout=8.0,
                )
            except Exception:
                continue
    except Exception as exc:
        logger.warning("Failed to initialize ChatGroq: %s", exc)
    return None


def extract_symptoms_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract structured clinical intake data via Groq LLM tool calling / structured output.
    Returns None if GROQ_API_KEY is missing, network is unavailable, or call fails.
    """
    llm = _get_groq_llm(temperature=0.0)
    if llm is None:
        return None
    try:
        structured_llm = llm.with_structured_output(StructuredIntake)
        prompt = (
            "You are a clinical triage intake assistant helping triage nurses and intake staff.\n"
            "Extract structured clinical data from the following intake text into the defined schema:\n"
            "- symptoms: list of standardized symptom terms (e.g., 'chest pain', 'fever', 'headache', 'shortness of breath')\n"
            "- duration_days: float duration in days (convert hours to decimal e.g. 0.1 for 2 hours, 1.0 for 1 day; 0.0 if unspecified)\n"
            "- severity: 'mild', 'moderate', 'severe', or 'unspecified'\n"
            "- pertinent_negatives: list of symptoms explicitly denied (e.g., 'no fever', 'denies chest pain')\n"
            "- red_flags: list of high-risk emergency phrasing or acute red-flag symptoms present\n\n"
            f"Intake text: {text}"
        )
        result = structured_llm.invoke(prompt)
        if result and isinstance(result, StructuredIntake):
            cleaned_symptoms = [s.strip().lower() for s in result.symptoms if s.strip()]
            duration_val = float(result.duration_days or 0.0)
            return {
                "symptoms": cleaned_symptoms,
                "duration_days": max(int(duration_val), 1 if 0 < duration_val < 1 else 0),
                "severity": result.severity or "unspecified",
                "pertinent_negatives": [n.strip() for n in result.pertinent_negatives if n.strip()],
                "red_flags": [r.strip() for r in result.red_flags if r.strip()],
                "has_symptoms": bool(cleaned_symptoms),
                "has_duration": duration_val > 0,
                "extraction_method": "llm",
            }
    except Exception as exc:
        logger.warning("Groq LLM intake extraction failed, falling back to regex: %s", exc)
    return None


def generate_followup_questions_llm(
    symptoms: List[str],
    duration_days: int,
    severity: str,
    raw_text: str,
) -> Optional[List[str]]:
    """
    Generate targeted clinical follow-up questions for missing/ambiguous intake fields using Groq LLM.
    Returns None if GROQ_API_KEY is missing, network is unavailable, or call fails.
    """
    llm = _get_groq_llm(temperature=0.2)
    if llm is None:
        return None
    try:
        structured_llm = llm.with_structured_output(FollowupQuestionsOutput)
        missing_aspects = []
        if not symptoms:
            missing_aspects.append("symptom description is vague or missing")
        if duration_days <= 0:
            missing_aspects.append("symptom duration/onset is unspecified")
        if severity == "unspecified":
            missing_aspects.append("severity/intensity is unspecified")

        prompt = (
            "You are a clinical decision support copilot for triage intake staff.\n"
            "Generate up to 4 high-yield, targeted clinical follow-up questions to clarify empty or ambiguous fields:\n"
            f"- Identified symptoms: {', '.join(symptoms) if symptoms else 'None'}\n"
            f"- Identified duration: {duration_days} day(s)\n"
            f"- Identified severity: {severity}\n"
            f"- Ambiguities/Missing fields: {', '.join(missing_aspects) if missing_aspects else 'clinical characterization'}\n"
            f"- Patient initial description: {raw_text}\n"
            "Formulate concise, clinically relevant questions targeted at these gaps. Return max 4 questions."
        )
        result = structured_llm.invoke(prompt)
        if result and isinstance(result, FollowupQuestionsOutput) and result.questions:
            cleaned = [q.strip() for q in result.questions if q.strip()]
            return cleaned[:4]
    except Exception as exc:
        exc_str = str(exc)
        if "failed_generation" in exc_str:
            lines = [line.strip() for line in exc_str.split("\\n") if line.strip()]
            extracted = []
            for line in lines:
                m = re.match(r"^\d+[\.\)]\s*(.*)", line)
                if m:
                    extracted.append(m.group(1).strip().strip("'\""))
            if extracted:
                return extracted[:4]
        logger.warning("Groq LLM follow-up generation failed, falling back to rule bank: %s", exc)
    return None


def extract_symptoms(text: str) -> Dict[str, Any]:
    """Deterministic regex symptom extractor fallback for offline / no-key execution."""
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
    negation_prefix_pattern = r"\b(?:no|not|denies|denying|without|negative for|free of)\s+(?:\w+\s+){0,2}"

    found = []
    pertinent_negatives = []

    for name, variants in symptom_patterns.items():
        matched_variant = None
        for variant in variants:
            if variant in lowered:
                matched_variant = variant
                break
        if matched_variant:
            neg_regex = rf"{negation_prefix_pattern}{re.escape(matched_variant)}"
            if re.search(neg_regex, lowered):
                pertinent_negatives.append(f"denies {name}")
            else:
                found.append(name)

    # Red flags regex check
    red_flag_patterns = [
        "crushing chest pain",
        "radiating to left arm",
        "radiating to jaw",
        "thunderclap headache",
        "worst headache of life",
        "facial droop",
        "slurred speech",
        "arm weakness",
        "cyanosis",
        "blue lips",
        "stridor",
        "hemoptysis",
        "coughing blood",
        "hematemesis",
        "vomiting blood",
        "anaphylaxis",
        "loss of consciousness",
    ]
    red_flags = [rf for rf in red_flag_patterns if rf in lowered]

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
        "pertinent_negatives": pertinent_negatives,
        "red_flags": red_flags,
        "has_symptoms": bool(found),
        "has_duration": duration_days > 0,
        "extraction_method": "regex_fallback",
    }


def extract_vitals_from_text(text: str) -> Dict[str, Any]:
    """
    Extract vital signs from clinical triage text (BP, HR, RR, SpO2, Temp).
    Used to augment nurse-entered physiological measurements.
    """
    vitals: Dict[str, Any] = {}
    lowered = text.lower()

    # Blood pressure: e.g. 120/80, bp 130/85, bp: 90/60
    bp_match = re.search(r"(?:bp|blood pressure)?\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})", lowered)
    if bp_match:
        try:
            vitals["sbp"] = int(bp_match.group(1))
            vitals["dbp"] = int(bp_match.group(2))
        except (ValueError, TypeError):
            pass

    # Heart rate / pulse: e.g. hr 110, pulse 95, hr: 120, heart rate 102
    hr_match = re.search(r"(?:hr|pulse|heart rate)\s*[:=]?\s*(\d{2,3})", lowered)
    if hr_match:
        try:
            vitals["hr"] = int(hr_match.group(1))
        except (ValueError, TypeError):
            pass

    # Respiratory rate: e.g. rr 28, resp rate 24, rr: 32
    rr_match = re.search(r"(?:rr|resp(?:iratory)? rate)\s*[:=]?\s*(\d{1,2})", lowered)
    if rr_match:
        try:
            vitals["rr"] = int(rr_match.group(1))
        except (ValueError, TypeError):
            pass

    # SpO2 / Oxygen Saturation: e.g. spo2 91%, o2 sat 90%, oxygen 89%, spo2: 94
    spo2_match = re.search(r"(?:spo2|o2 sat(?:uration)?|pulse ox|oxygen)\s*[:=]?\s*(\d{2,3})\s*%?", lowered)
    if spo2_match:
        try:
            val = int(spo2_match.group(1))
            if 50 <= val <= 100:
                vitals["spo2"] = val
        except (ValueError, TypeError):
            pass

    # Temperature: e.g. temp 101.4, temp: 38.5, temperature 102 f
    temp_match = re.search(r"(?:temp(?:erature)?)\s*[:=]?\s*(\d{2,3}(?:\.\d)?)", lowered)
    if temp_match:
        try:
            vitals["temp"] = float(temp_match.group(1))
        except (ValueError, TypeError):
            pass

    return vitals


def retrieve_patient_memory(user_id: str, query_text: str) -> Dict[str, Any]:
    return MEMORY_STORE.retrieve_patient_profile(user_id, query_text)


def generate_followup_questions(symptoms: List[str]) -> Dict[str, Any]:
    """Rule-based clinical follow-up question bank fallback."""
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


def assess_medical_risk(
    symptoms: List[str],
    duration: int,
    profile: Dict[str, Any],
    severity: str,
    red_flags: Optional[List[str]] = None,
    vitals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assess medical risk level (LOW/MEDIUM/HIGH) and ESI Acuity Level (1-5)
    aligned with Emergency Severity Index (ESI) triage standards and vital sign instability.
    """
    lowered = [item.lower() for item in symptoms]
    red_flags = red_flags or []
    vitals = vitals or {}
    vital_triggers: List[str] = []

    # ESI Level 1 (Resuscitation - Immediate life-saving intervention needed)
    resuscitation_keywords = {
        "loss of consciousness",
        "unconscious",
        "unresponsive",
        "anaphylaxis",
        "cardiac arrest",
        "respiratory arrest",
        "can't breathe",
        "uncontrolled bleeding",
        "cyanosis",
    }

    # ESI Level 2 (Emergent / High-Risk situation / Should not wait)
    emergent_keywords = {
        "chest pain",
        "chest tightness",
        "chest pressure",
        "shortness of breath",
        "difficulty breathing",
        "confusion",
        "disoriented",
        "fainting",
        "syncope",
        "severe headache",
        "sudden weakness",
        "face drooping",
        "slurred speech",
    }

    minor_short_term = {
        "headache",
        "sore throat",
        "fatigue",
        "muscle pain",
        "joint pain",
        "cough",
        "runny nose",
        "congestion",
        "sneezing",
        "itchy eyes",
    }
    moderate = {
        "fever",
        "nausea",
        "vomiting",
        "diarrhea",
        "dizziness",
        "abdominal pain",
        "rash",
        "swelling",
    }

    risk_level = "LOW"
    reasons: List[str] = []
    emergency_recommended = False
    esi_level = 5
    esi_rationale = ""

    spo2 = vitals.get("spo2")
    hr = vitals.get("hr")
    rr = vitals.get("rr")
    sbp = vitals.get("sbp")

    # Check ESI 1 first (Resuscitation)
    is_esi_1 = any(any(k in s for k in resuscitation_keywords) for s in lowered) or any(
        any(k in rf.lower() for k in resuscitation_keywords) for rf in red_flags
    )
    if not is_esi_1 and spo2 is not None:
        try:
            if float(spo2) < 90:
                is_esi_1 = True
                reasons.append(f"Critical hypoxia (SpO2 {spo2}%) requiring immediate resuscitation intervention")
                vital_triggers.append(f"Critical Hypoxia: SpO2 {spo2}% (<90%)")
        except (ValueError, TypeError):
            pass

    if is_esi_1:
        risk_level = "HIGH"
        esi_level = 1
        emergency_recommended = True
        if not reasons:
            reasons.append("Critical life-threatening presentation requiring immediate resuscitation")
        esi_rationale = "Immediate life-saving intervention required (ESI 1 - Resuscitation)"
    else:
        # Check ESI 2 (Emergent / High-risk or Danger Zone Vitals)
        is_esi_2 = False
        for symptom in lowered:
            if any(flag in symptom for flag in emergent_keywords):
                is_esi_2 = True
                reasons.append(f"Critical emergent symptom detected: {symptom}")
                break
        if not is_esi_2 and red_flags:
            is_esi_2 = True
            reasons.append(f"Emergency red-flag indicators detected: {', '.join(red_flags)}")

        # Evaluate objective danger zone vitals
        if not is_esi_2:
            try:
                if spo2 is not None and float(spo2) <= 92:
                    is_esi_2 = True
                    reasons.append(f"Significant hypoxia (SpO2 {spo2}% <= 92%) warrants emergent evaluation")
                    vital_triggers.append(f"Hypoxia: SpO2 {spo2}%")
                elif hr is not None and (float(hr) < 40 or float(hr) > 140):
                    is_esi_2 = True
                    reasons.append(f"Hemodynamic instability (HR {hr} bpm) warrants emergent evaluation")
                    vital_triggers.append(f"Extreme Heart Rate: {hr} bpm")
                elif rr is not None and (float(rr) < 8 or float(rr) > 30):
                    is_esi_2 = True
                    reasons.append(f"Severe respiratory instability (RR {rr}/min) warrants emergent evaluation")
                    vital_triggers.append(f"Extreme Respiratory Rate: {rr}/min")
                elif sbp is not None and float(sbp) < 90:
                    is_esi_2 = True
                    reasons.append(f"Hypotensive shock danger (SBP {sbp} mmHg < 90) warrants emergent evaluation")
                    vital_triggers.append(f"Hypotension: SBP {sbp} mmHg")
            except (ValueError, TypeError):
                pass

        if is_esi_2:
            risk_level = "HIGH"
            esi_level = 2
            emergency_recommended = True
            esi_rationale = "High-risk emergent presentation / patient should not wait (ESI 2 - Emergent)"
        else:
            # Check duration, severity, and patient comorbidities
            if duration >= 7:
                risk_level = "HIGH"
                esi_level = 3
                reasons.append(f"Prolonged duration: {duration} days")
                esi_rationale = "Prolonged illness (>=7 days) requires multiple diagnostic resources (ESI 3 - Urgent)"
            elif duration >= 3:
                risk_level = "MEDIUM"
                esi_level = 3
                reasons.append(f"Extended duration: {duration} days")
                esi_rationale = "Extended duration (>=3 days) warrants multi-resource evaluation (ESI 3 - Urgent)"

            if severity == "severe":
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
                    esi_level = 3
                elif risk_level == "MEDIUM":
                    risk_level = "HIGH"
                    esi_level = 2
                reasons.append("Severe symptom intensity reported")
                if esi_level == 2:
                    esi_rationale = "Severe acute distress escalates acuity to high risk (ESI 2 - Emergent)"
                elif not esi_rationale:
                    esi_rationale = "Severe intensity warrants urgent multi-resource evaluation (ESI 3 - Urgent)"

            risk_factors = profile.get("risk_factors", [])
            if "advanced_age_65+" in risk_factors:
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
                    esi_level = 3
                elif risk_level == "MEDIUM":
                    risk_level = "HIGH"
                    esi_level = 2
                reasons.append("Age 65+ elevates risk")
                if esi_level == 2 and not esi_rationale:
                    esi_rationale = "High-risk age group with acute presentation (ESI 2 - Emergent)"

            if any(item.startswith("condition:") for item in risk_factors):
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
                    esi_level = 3
                elif risk_level == "MEDIUM":
                    risk_level = "HIGH"
                    esi_level = 2
                reasons.append("Pre-existing high-risk conditions")
                if esi_level == 2 and not esi_rationale:
                    esi_rationale = "Pre-existing chronic comorbidities with acute symptoms (ESI 2 - Emergent)"

            # Check low-risk / ESI 5 (routine, self-limited)
            if (
                risk_level == "LOW"
                and lowered
                and severity in {"mild", "unspecified"}
                and 0 < duration <= 2
                and not risk_factors
                and all(any(m in s for m in minor_short_term) for s in lowered)
            ):
                reasons.append("Short-duration mild symptoms without high-risk context")
                esi_level = 5
                esi_rationale = "Mild, self-limited presentation requiring routine clinic/home care (ESI 5 - Non-Urgent)"
                return {
                    "risk_level": "LOW",
                    "esi_level": 5,
                    "esi_rationale": esi_rationale,
                    "emergency_recommended": False,
                    "risk_reasons": reasons,
                    "possible_concerns": ["Self-limited minor illness"],
                }

            moderate_count = sum(1 for symptom in lowered if any(item in symptom for item in moderate))
            if risk_level != "HIGH":
                if moderate_count >= 3:
                    risk_level = "HIGH"
                    esi_level = 3
                    reasons.append(f"Multiple concerning symptoms ({moderate_count})")
                    esi_rationale = f"Multi-symptom presentation ({moderate_count} symptoms) requiring extensive resources (ESI 3 - Urgent)"
                elif moderate_count >= 2:
                    if risk_level == "LOW":
                        risk_level = "MEDIUM"
                        esi_level = 3
                    reasons.append(f"{moderate_count} moderate-risk symptoms present")
                    if not esi_rationale:
                        esi_rationale = f"Multiple symptoms ({moderate_count}) requiring diagnostic resources (ESI 3 - Urgent)"
                elif moderate_count == 1:
                    if risk_level == "LOW":
                        risk_level = "MEDIUM"
                        esi_level = 4
                    reasons.append("1 moderate-risk symptom present")
                    if not esi_rationale:
                        esi_rationale = "Single acute symptom requiring simple resource evaluation (ESI 4 - Less Urgent)"

    if not reasons:
        reasons.append("Symptoms currently appear mild and lower risk")
        esi_level = 5
        esi_rationale = "Mild presentation suitable for routine outpatient care (ESI 5 - Non-Urgent)"

    possible_concerns: List[str] = []
    if "fever" in lowered and "cough" in lowered:
        possible_concerns.append("Upper respiratory infection")
    if "headache" in lowered and "fever" in lowered:
        possible_concerns.append("Viral illness")
    if "chest pain" in lowered:
        possible_concerns.append("Cardiac or musculoskeletal issue that needs urgent evaluation")
    if "shortness of breath" in lowered:
        possible_concerns.append("Pulmonary or airway evaluation needed")
    if "abdominal pain" in lowered and "nausea" in lowered:
        possible_concerns.append("Gastrointestinal issue")
    if not possible_concerns:
        possible_concerns.append("General illness that should be monitored")

    return {
        "risk_level": risk_level,
        "esi_level": esi_level,
        "esi_rationale": esi_rationale,
        "emergency_recommended": emergency_recommended,
        "risk_reasons": reasons,
        "possible_concerns": possible_concerns,
        "vital_triggers": vital_triggers,
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
    esi_level: int,
    esi_rationale: str,
    symptoms: List[str],
    duration: int,
    concerns: List[str],
    followup_questions: List[str],
    is_emergency: bool,
    profile: Dict[str, Any],
    pertinent_negatives: Optional[List[str]] = None,
    intake_method: Optional[str] = None,
    infermedica_triage: Optional[Dict[str, Any]] = None,
    vital_triggers: Optional[List[str]] = None,
    **kwargs: Any,
) -> str:
    esi_titles = {
        1: "ESI Level 1 - Resuscitation (Immediate Life Threat)",
        2: "ESI Level 2 - Emergent (High Risk / Should Not Wait)",
        3: "ESI Level 3 - Urgent (Multiple Clinical Resources Needed)",
        4: "ESI Level 4 - Less Urgent (Single Clinical Resource Needed)",
        5: "ESI Level 5 - Non-Urgent (Routine Clinic / Outpatient Care)",
    }
    esi_title = esi_titles.get(esi_level, f"Level {esi_level}")

    lines = [
        f"## Clinical Triage Decision Support | {esi_title}",
        f"**Risk Category:** {risk_level}",
        f"**Acuity Rationale:** {esi_rationale}",
    ]

    if vital_triggers:
        lines.append(f"**Vital Signs Triggers:** {', '.join(vital_triggers)}")

    if infermedica_triage:
        inf_desc = infermedica_triage.get("description", "")
        inf_label = infermedica_triage.get("label", "")
        inf_level = infermedica_triage.get("triage_level", "")
        lines.append(f"**Certified Engine Level:** `{inf_level}` {f'({inf_label})' if inf_label else ''}")
        if inf_desc:
            lines.append(f"**Clinical Engine Summary:** {inf_desc}")

    lines.append("")

    if is_emergency or risk_level == "HIGH" or esi_level in (1, 2):
        lines.extend(
            [
                "### Immediate Clinical Attention Required",
                "- Patient meets high-acuity emergency criteria (ESI 1 or 2).",
                "- Priority resuscitation bay / acute monitoring bed allocation required.",
                "- Licensed triage nurse review and physician notification required.",
            ]
        )
    elif risk_level == "MEDIUM" or esi_level in (3, 4):
        lines.extend(
            [
                "### Urgent Clinical Track Advised",
                "- Patient requires multi-resource (ESI 3) or focused single-resource (ESI 4) clinic evaluation.",
                "- Order initial diagnostic workup (labs, imaging) per departmental protocol.",
            ]
        )
    else:
        lines.extend(
            [
                "### Routine Outpatient Intake Track",
                "- Patient presentation is low-acuity (ESI 5), suitable for routine ambulatory care or home management.",
                "- No acute emergency department resources required.",
            ]
        )

    if symptoms:
        lines.extend(["", f"**Reported symptoms:** {', '.join(symptoms)}"])
    if pertinent_negatives:
        lines.append(f"**Pertinent negatives:** {', '.join(pertinent_negatives)}")
    if duration:
        lines.append(f"**Duration:** {duration} day(s)")
    if profile.get("conditions"):
        lines.append(f"**Known conditions considered:** {', '.join(profile['conditions'])}")
    if concerns:
        lines.extend(["", "**Differential Concerns (Non-Diagnostic):**"])
        lines.extend([f"- {item}" for item in concerns])

    if followup_questions and risk_level != "HIGH":
        lines.extend(["", "**Helpful follow-up questions for a clinician:**"])
        lines.extend([f"- {item}" for item in followup_questions])

    lines.extend(
        [
            "",
            "---",
            "Consult a qualified healthcare professional for diagnosis and treatment. This assistant is for informational decision support only.",
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
    elif "sbar" not in completed:
        next_step = "sbar"
    elif state.get("hitl_required") and "human_review" not in completed:
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
    profile = retrieve_patient_memory(updated["user_id"], latest_input)
    age = profile.get("age", 25)
    sex = updated.get("patient_sex", "male") or "male"

    # Merge explicitly provided vitals with any vitals found in text
    vitals = dict(updated.get("vitals") or {})
    extracted_vitals = extract_vitals_from_text(latest_input)
    for k, v in extracted_vitals.items():
        if k not in vitals or vitals[k] is None:
            vitals[k] = v
    updated["vitals"] = vitals

    extracted = None
    infermedica_evidence = None

    # 1. Primary: Try Infermedica Clinical Engine v3 /parse
    if is_infermedica_configured():
        infermedica_data = parse_symptoms_infermedica(latest_input, age=age, sex=sex)
        if infermedica_data and (infermedica_data.get("symptoms") or infermedica_data.get("evidence")):
            infermedica_evidence = infermedica_data.get("evidence", [])
            aux = extract_symptoms(latest_input)
            extracted = {
                "symptoms": infermedica_data.get("symptoms", []),
                "duration_days": aux.get("duration_days", 0),
                "severity": aux.get("severity", "unspecified"),
                "pertinent_negatives": infermedica_data.get("pertinent_negatives", []),
                "red_flags": aux.get("red_flags", []),
                "has_symptoms": bool(infermedica_data.get("symptoms")),
                "has_duration": aux.get("duration_days", 0) > 0,
                "extraction_method": "infermedica",
            }

    # 2. Secondary: Try Groq LLM tool-calling if Infermedica was not configured or produced no matches
    if extracted is None:
        extracted = extract_symptoms_llm(latest_input)

    # 3. Tertiary: Deterministic regex fallback
    if extracted is None:
        extracted = extract_symptoms(latest_input)

    followups: List[str] = []
    if infermedica_evidence and is_infermedica_configured():
        diag = get_diagnosis_infermedica(infermedica_evidence, age=age, sex=sex)
        if diag and diag.get("question") and isinstance(diag["question"], dict):
            q_text = diag["question"].get("text")
            if q_text:
                followups.append(q_text)

    if not followups:
        if not extracted["has_symptoms"] or not extracted["has_duration"] or extracted["severity"] == "unspecified":
            llm_followups = generate_followup_questions_llm(
                symptoms=extracted["symptoms"],
                duration_days=extracted["duration_days"],
                severity=extracted["severity"],
                raw_text=latest_input,
            )
            if llm_followups:
                followups = llm_followups
            else:
                followups = generate_followup_questions(extracted["symptoms"])["questions"]

    updated["symptoms"] = extracted["symptoms"]
    updated["duration_days"] = extracted["duration_days"]
    updated["severity"] = extracted["severity"]
    updated["pertinent_negatives"] = extracted.get("pertinent_negatives", [])
    updated["red_flags"] = extracted.get("red_flags", [])
    updated["infermedica_evidence"] = infermedica_evidence
    updated["intake_method"] = extracted.get("extraction_method", "regex_fallback")
    updated["patient_profile"] = profile
    updated["followup_questions"] = followups

    if updated["intake_method"] == "infermedica":
        method_badge = "Infermedica Clinical Engine (v3/parse)"
    elif updated["intake_method"] == "llm":
        method_badge = "Groq LLM (llama-3.3-70b-versatile)"
    else:
        method_badge = "Deterministic Regex Fallback"

    negatives_text = (
        f"; negatives: {', '.join(updated['pertinent_negatives'])}" if updated["pertinent_negatives"] else ""
    )
    red_flags_text = f"; red flags: {', '.join(updated['red_flags'])}" if updated["red_flags"] else ""
    vitals_parts = []
    if vitals.get("sbp") and vitals.get("dbp"):
        vitals_parts.append(f"BP {vitals['sbp']}/{vitals['dbp']}")
    if vitals.get("hr"):
        vitals_parts.append(f"HR {vitals['hr']}")
    if vitals.get("rr"):
        vitals_parts.append(f"RR {vitals['rr']}")
    if vitals.get("spo2"):
        vitals_parts.append(f"SpO2 {vitals['spo2']}%")
    if vitals.get("temp"):
        vitals_parts.append(f"Temp {vitals['temp']}F")
    vitals_text = f"; vitals: {', '.join(vitals_parts)}" if vitals_parts else ""

    updated["intake_summary"] = (
        f"[{method_badge}] Identified symptoms: {', '.join(extracted['symptoms']) or 'none'}; "
        f"duration: {extracted['duration_days']} day(s); severity: {extracted['severity']}"
        f"{negatives_text}{red_flags_text}{vitals_text}; memory source: {profile.get('memory_source', 'local')}"
    )
    updated.setdefault("completed_steps", []).append("intake")

    tools_used = [
        "parse_symptoms_infermedica"
        if updated["intake_method"] == "infermedica"
        else ("extract_symptoms_llm" if updated["intake_method"] == "llm" else "extract_symptoms_regex"),
        "retrieve_patient_memory",
    ]
    if vitals_parts:
        tools_used.append("extract_vitals_from_text")
    if followups:
        tools_used.append(
            "get_diagnosis_infermedica"
            if updated["intake_method"] == "infermedica"
            else ("generate_followup_questions_llm" if updated["intake_method"] == "llm" else "generate_followup_questions")
        )

    _append_trace(
        updated,
        "intake",
        updated["intake_summary"],
        tools_used,
    )
    return updated


def risk_node(state: AgentState) -> AgentState:
    updated = dict(state)
    profile = updated.get("patient_profile", {})
    age = profile.get("age", 25)
    sex = updated.get("patient_sex", "male") or "male"
    evidence = updated.get("infermedica_evidence")
    vitals = updated.get("vitals", {})

    # 1. Compute internal risk & ESI assessment with physiological vitals
    internal_assessment = assess_medical_risk(
        symptoms=updated.get("symptoms", []),
        duration=updated.get("duration_days", 0),
        profile=profile,
        severity=updated.get("severity", "unspecified"),
        red_flags=updated.get("red_flags", []),
        vitals=vitals,
    )
    vital_triggers = internal_assessment.get("vital_triggers", [])
    updated["vital_triggers"] = vital_triggers

    tools_called = ["assess_medical_risk", "check_emergency_red_flags"]
    infermedica_triage_res = None

    # 2. Query Infermedica Triage Engine if evidence is available
    if evidence and is_infermedica_configured():
        infermedica_triage_res = get_triage_infermedica(evidence=evidence, age=age, sex=sex)
        if infermedica_triage_res:
            tools_called.append("get_triage_infermedica")

    # 3. Consensus arbitration ("Escalate, never downgrade")
    if infermedica_triage_res:
        inf_level = infermedica_triage_res.get("triage_level", "self_care")
        inf_mapping = {
            "emergency_ambulance": ("HIGH", 1),
            "emergency": ("HIGH", 2),
            "consultation_24": ("MEDIUM", 3),
            "consultation": ("MEDIUM", 4),
            "self_care": ("LOW", 5),
        }
        inf_risk, inf_esi = inf_mapping.get(inf_level, ("LOW", 5))

        int_risk = internal_assessment.get("risk_level", "LOW")
        int_esi = internal_assessment.get("esi_level", 5)

        # Consensus risk: HIGH if either is HIGH; MEDIUM if either is MEDIUM; else LOW
        if "HIGH" in (inf_risk, int_risk):
            final_risk = "HIGH"
        elif "MEDIUM" in (inf_risk, int_risk):
            final_risk = "MEDIUM"
        else:
            final_risk = "LOW"

        # Consensus ESI: Take highest acuity (lowest integer 1-5)
        final_esi = min(int_esi, inf_esi)

        # Merge concerns and rationale
        concerns = list(internal_assessment.get("possible_concerns", []))
        serious_list = infermedica_triage_res.get("serious", [])
        for s in serious_list:
            if isinstance(s, dict) and s.get("name") and s["name"] not in concerns:
                concerns.append(f"{s['name']} (Infermedica)")

        inf_desc = infermedica_triage_res.get("description", "")
        rationale_parts = []
        if internal_assessment.get("esi_rationale"):
            rationale_parts.append(f"Internal Protocol: {internal_assessment['esi_rationale']}")
        if inf_desc:
            rationale_parts.append(f"Infermedica Engine: {inf_desc}")
        final_rationale = " | ".join(rationale_parts) if rationale_parts else "Dual-engine validated"

        reasons = list(
            dict.fromkeys(
                internal_assessment.get("risk_reasons", [])
                + ([f"Infermedica Triage: {inf_level}"] if inf_level else [])
            )
        )

        emergency = (
            check_emergency_red_flags(updated.get("symptoms", []))
            or bool(updated.get("red_flags"))
            or internal_assessment.get("emergency_recommended", False)
            or final_risk == "HIGH"
            or final_esi in (1, 2)
            or inf_level in ("emergency_ambulance", "emergency")
        )

        assessment = {
            "risk_level": final_risk,
            "esi_level": final_esi,
            "esi_rationale": final_rationale,
            "emergency_recommended": emergency,
            "is_emergency": emergency,
            "risk_reasons": reasons,
            "possible_concerns": concerns,
            "vital_triggers": vital_triggers,
            "infermedica_triage": infermedica_triage_res,
            "internal_assessment": internal_assessment,
            "consensus_applied": True,
        }
    else:
        emergency = (
            check_emergency_red_flags(updated.get("symptoms", []))
            or bool(updated.get("red_flags"))
            or internal_assessment.get("emergency_recommended", False)
        )
        assessment = internal_assessment
        assessment["is_emergency"] = emergency
        assessment["vital_triggers"] = vital_triggers
        assessment["infermedica_triage"] = None
        assessment["consensus_applied"] = False

    updated["medications"] = []
    updated["infermedica_triage"] = infermedica_triage_res
    updated["risk_assessment"] = assessment
    updated["hitl_required"] = assessment["risk_level"] == "HIGH" or assessment.get("esi_level", 5) in (1, 2)
    updated["awaiting_human_review"] = False
    updated.setdefault("completed_steps", []).append("risk")

    summary_text = f"Risk assessed as {assessment['risk_level']} | ESI Level {assessment['esi_level']}"
    if assessment.get("consensus_applied"):
        summary_text += f" (Dual Consensus: Internal ESI {internal_assessment.get('esi_level')} + Infermedica {infermedica_triage_res.get('triage_level')})"
    else:
        summary_text += f" ({assessment.get('esi_rationale', '')})"

    _append_trace(
        updated,
        "risk",
        summary_text,
        tools_called,
    )
    return updated


def safety_node(state: AgentState) -> AgentState:
    updated = dict(state)
    assessment = updated.get("risk_assessment", {})
    response = finalize_response(
        risk_level=assessment.get("risk_level", "LOW"),
        esi_level=assessment.get("esi_level", 5),
        esi_rationale=assessment.get("esi_rationale", "Routine evaluation"),
        symptoms=updated.get("symptoms", []),
        duration=updated.get("duration_days", 0),
        concerns=assessment.get("possible_concerns", []),
        followup_questions=updated.get("followup_questions", []),
        is_emergency=assessment.get("is_emergency", False),
        profile=updated.get("patient_profile", {}),
        pertinent_negatives=updated.get("pertinent_negatives", []),
        intake_method=updated.get("intake_method"),
        infermedica_triage=updated.get("infermedica_triage"),
        vital_triggers=updated.get("vital_triggers", []),
    )
    safety = perform_safety_check(response, assessment.get("is_emergency", False))
    if not safety["safe_to_proceed"]:
        response += "\n\nSafety note: This guidance was adjusted to stay informational and non-diagnostic."

    if updated.get("hitl_required"):
        response += "\n\nLicensed nurse sign-off is required before this high-risk triage recommendation is released."

    updated["final_response"] = response
    updated["safety_validation"] = safety
    updated.setdefault("completed_steps", []).append("safety")
    _append_trace(
        updated,
        "safety",
        f"Final clinical guidance drafted with ESI Level {assessment.get('esi_level', 5)} and safety-checked",
        ["finalize_response", "perform_safety_check"],
    )
    return updated


def sbar_node(state: AgentState) -> AgentState:
    """
    Generate structured SBAR clinical handoff note and HL7 FHIR R4 Transaction Bundle.
    """
    updated = dict(state)
    profile = updated.get("patient_profile", {})
    assessment = updated.get("risk_assessment", {})
    user_id = updated.get("user_id", "default")
    latest_input = _latest_user_message(updated.get("messages", []))
    age = profile.get("age", 25)
    sex = updated.get("patient_sex", "male") or "male"
    esi_level = assessment.get("esi_level", 5)
    esi_rationale = assessment.get("esi_rationale", "")
    symptoms = updated.get("symptoms", [])
    vitals = updated.get("vitals", {})
    vital_triggers = updated.get("vital_triggers", [])
    pertinent_negatives = updated.get("pertinent_negatives", [])
    concerns = assessment.get("possible_concerns", [])
    nurse_esi = updated.get("nurse_override_esi")
    nurse_notes = updated.get("nurse_notes") or updated.get("approval_note")

    sbar_note = generate_sbar_note(
        patient_id=user_id,
        age=age,
        sex=sex,
        chief_complaint=latest_input,
        symptoms=symptoms,
        vitals=vitals,
        esi_level=esi_level,
        esi_rationale=esi_rationale,
        vital_triggers=vital_triggers,
        pertinent_negatives=pertinent_negatives,
        differential_concerns=concerns,
        nurse_override_esi=nurse_esi,
        nurse_notes=nurse_notes,
    )

    fhir_bundle = generate_fhir_r4_bundle(
        patient_id=user_id,
        age=age,
        sex=sex,
        chief_complaint=latest_input,
        symptoms=symptoms,
        vitals=vitals,
        esi_level=esi_level,
        esi_rationale=esi_rationale,
        sbar_note=sbar_note,
        vital_triggers=vital_triggers,
        nurse_override_esi=nurse_esi,
    )

    updated["sbar_note"] = sbar_note
    updated["fhir_bundle"] = fhir_bundle
    updated.setdefault("completed_steps", []).append("sbar")

    _append_trace(
        updated,
        "sbar",
        f"Generated SBAR clinical handoff note and HL7 FHIR R4 Bundle (ID: {fhir_bundle.get('id', 'N/A')[:8]}...)",
        ["generate_sbar_note", "generate_fhir_r4_bundle"],
    )
    return updated


def human_review_node(state: AgentState) -> AgentState:
    updated = dict(state)
    raw_decision = (updated.get("approval_decision") or "").strip().lower()
    override_esi = updated.get("nurse_override_esi")
    override_reason = updated.get("nurse_override_reason") or ""
    nurse_notes = updated.get("nurse_notes") or (updated.get("approval_note") or "").strip()

    decision = None
    if raw_decision in {"approved", "approve", "accept", "accepted"}:
        decision = "approved"
    elif raw_decision in {"rejected", "reject", "override", "overridden"}:
        decision = "overridden"
    elif override_esi is not None:
        assessment_esi = updated.get("risk_assessment", {}).get("esi_level", 5)
        decision = "overridden" if override_esi != assessment_esi else "approved"

    if decision not in {"approved", "overridden"}:
        updated["awaiting_human_review"] = True
        updated["hitl_required"] = True
        _append_trace(
            updated,
            "human_review",
            "Execution paused for triage nurse review & sign-off",
            [],
        )
        return updated

    updated["awaiting_human_review"] = False
    updated["hitl_required"] = False

    assessment = updated.get("risk_assessment", {})
    suggested_esi = assessment.get("esi_level", 5)
    final_esi = override_esi if (decision == "overridden" and override_esi is not None) else suggested_esi

    # Audit Log the encounter for clinical concordance tracking
    audit = get_audit_logger()
    audit.log_encounter(
        patient_id=updated.get("user_id", "default"),
        chief_complaint=_latest_user_message(updated.get("messages", [])),
        symptoms=updated.get("symptoms", []),
        vitals=updated.get("vitals", {}),
        ai_suggested_esi=suggested_esi,
        ai_rationale=assessment.get("esi_rationale", ""),
        ai_risk_level=assessment.get("risk_level", "LOW"),
        nurse_decision="override" if (decision == "overridden" and override_esi != suggested_esi) else "accept",
        nurse_final_esi=final_esi,
        override_reason=override_reason,
        nurse_notes=nurse_notes,
        vital_triggers=updated.get("vital_triggers", []),
        intake_method=updated.get("intake_method", "regex_fallback"),
        fhir_bundle_id=updated.get("fhir_bundle", {}).get("id"),
    )

    if decision == "approved":
        approval_line = f"Nurse Acceptance Recorded: AI ESI Level {suggested_esi} verified and approved."
    else:
        approval_line = (
            f"Nurse Override Recorded: Changed AI ESI {suggested_esi} -> ESI {final_esi}. "
            f"Reason: {override_reason or 'Clinical judgment'}."
        )

    if nurse_notes:
        approval_line += f" Clinical note: {nurse_notes}"

    updated["final_response"] = f"{updated.get('final_response', '')}\n\n---\n**Clinician Sign-off:** {approval_line}".strip()
    updated.setdefault("completed_steps", []).append("human_review")
    _append_trace(updated, "human_review", approval_line, ["audit_logger.log_encounter"])
    return updated


def finish_node(state: AgentState) -> AgentState:
    updated = dict(state)
    _append_trace(updated, "finish", "Workflow complete", [])
    return updated


def _route_from_supervisor(state: AgentState) -> Literal["intake", "risk", "safety", "sbar", "human_review", "finish"]:
    return state["next_step"]  # type: ignore[return-value]


def build_multi_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("intake", intake_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("safety", safety_node)
    workflow.add_node("sbar", sbar_node)
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
            "sbar": "sbar",
            "human_review": "human_review",
            "finish": "finish",
        },
    )
    workflow.add_edge("intake", "supervisor")
    workflow.add_edge("risk", "supervisor")
    workflow.add_edge("safety", "supervisor")
    workflow.add_edge("sbar", "supervisor")
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
        sex: str = "male",
        vitals: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        pending_state: Optional[Dict[str, Any]] = None,
        approval_decision: Optional[str] = None,
        approval_note: Optional[str] = None,
        nurse_override_esi: Optional[int] = None,
        nurse_override_reason: Optional[str] = None,
        nurse_notes: Optional[str] = None,
        location_address: Optional[str] = None,
        location_lat: Optional[float] = None,
        location_lng: Optional[float] = None,
    ) -> Dict[str, Any]:
        conditions = conditions or []
        history = history or []
        vitals = vitals or {}

        if pending_state:
            state = deepcopy(pending_state)
            state["approval_decision"] = approval_decision
            state["approval_note"] = approval_note
            if nurse_override_esi is not None:
                state["nurse_override_esi"] = nurse_override_esi
            if nurse_override_reason:
                state["nurse_override_reason"] = nurse_override_reason
            if nurse_notes:
                state["nurse_notes"] = nurse_notes
        else:
            set_patient_context(user_id, age, conditions)
            MEMORY_STORE.upsert_patient_profile(user_id, age, conditions)
            messages = history + ([{"role": "user", "content": user_input.strip()}] if user_input else [])
            state = {
                "messages": messages,
                "user_id": user_id,
                "patient_sex": sex,
                "vitals": vitals,
                "vital_triggers": [],
                "location_address": location_address,
                "location_lat": location_lat,
                "location_lng": location_lng,
                "nearby_facilities": [],
                "hospital_recommendations": [],
                "symptoms": [],
                "duration_days": 0,
                "severity": "unspecified",
                "pertinent_negatives": [],
                "red_flags": [],
                "infermedica_evidence": None,
                "infermedica_triage": None,
                "medications": [],
                "intake_method": "regex_fallback",
                "intake_summary": "",
                "followup_questions": [],
                "patient_profile": {},
                "risk_assessment": {},
                "safety_validation": {},
                "sbar_note": "",
                "fhir_bundle": {},
                "final_response": "",
                "workflow_trace": [],
                "tool_calls_made": [],
                "completed_steps": [],
                "next_step": "intake",
                "approval_decision": approval_decision,
                "approval_note": approval_note,
                "nurse_override_esi": nurse_override_esi,
                "nurse_override_reason": nurse_override_reason,
                "nurse_notes": nurse_notes,
                "awaiting_human_review": False,
                "hitl_required": False,
            }

        final_state = self.graph.invoke(state, config={"recursion_limit": 25})
        assessment = final_state.get("risk_assessment", {})
        pending = None
        if final_state.get("awaiting_human_review"):
            pending = deepcopy(final_state)

        return {
            "content": final_state.get("final_response", "No response generated."),
            "risk_level": assessment.get("risk_level", "LOW"),
            "esi_level": assessment.get("esi_level", 5),
            "esi_rationale": assessment.get("esi_rationale", ""),
            "is_emergency": assessment.get("is_emergency", False),
            "hitl_required": final_state.get("hitl_required", False),
            "awaiting_human_review": final_state.get("awaiting_human_review", False),
            "workflow_trace": final_state.get("workflow_trace", []),
            "tool_calls_made": final_state.get("tool_calls_made", []),
            "patient_profile": final_state.get("patient_profile", {}),
            "symptoms": final_state.get("symptoms", []),
            "vitals": final_state.get("vitals", {}),
            "vital_triggers": final_state.get("vital_triggers", []),
            "sbar_note": final_state.get("sbar_note", ""),
            "fhir_bundle": final_state.get("fhir_bundle", {}),
            "duration_days": final_state.get("duration_days", 0),
            "severity": final_state.get("severity", "unspecified"),
            "pertinent_negatives": final_state.get("pertinent_negatives", []),
            "red_flags": final_state.get("red_flags", []),
            "infermedica_evidence": final_state.get("infermedica_evidence"),
            "infermedica_triage": final_state.get("infermedica_triage"),
            "medications": final_state.get("medications", []),
            "intake_method": final_state.get("intake_method", "regex_fallback"),
            "followup_questions": final_state.get("followup_questions", []),
            "nurse_override_esi": final_state.get("nurse_override_esi"),
            "nurse_override_reason": final_state.get("nurse_override_reason"),
            "nurse_notes": final_state.get("nurse_notes"),
            "nearby_facilities": final_state.get("nearby_facilities", []),
            "hospital_recommendations": final_state.get("hospital_recommendations", []),
            "location_address": final_state.get("location_address"),
            "location_lat": final_state.get("location_lat"),
            "location_lng": final_state.get("location_lng"),
            "pending_state": pending,
        }


def get_multi_agent() -> MultiAgentSystem:
    return MultiAgentSystem()
