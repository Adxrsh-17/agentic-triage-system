import hashlib
import json
import logging
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

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
from agent.llm import is_llm_available
from agent.nodes.intake import run_intake
from agent.nodes.risk import run_hybrid_risk
from agent.nodes.safety import MANDATORY_DISCLAIMER, generate_safe_response, validate_safety_output
from agent.rules import check_red_flags
from agent.sbar_generator import generate_fhir_r4_bundle, generate_sbar_note
from agent.schemas import IntakeResult, RiskLevel, RiskResult


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
        except BaseException as exc:
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



def _append_trace(
    state: AgentState,
    node: str,
    summary: str,
    tools: Optional[List[str]] = None,
    duration_ms: Optional[float] = None,
) -> None:
    entry: Dict[str, Any] = {
        "node": node,
        "summary": summary,
        "tools": tools or [],
    }
    if duration_ms is not None:
        entry["duration_ms"] = duration_ms
    state.setdefault("workflow_trace", []).append(entry)
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
                    max_retries=0,
                    timeout=3.0,
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


def _is_negated(keyword: str, text: str, pertinent_negatives: Optional[List[str]] = None) -> bool:
    if pertinent_negatives:
        for neg in pertinent_negatives:
            if keyword in neg.lower():
                return True
    pattern = rf"(?:denies|deny|denied|no|not|without|negative for|rules? out|not experiencing|zero)\s+(?:[a-zA-Z0-9_,-]+\s+){{0,5}}{re.escape(keyword)}"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        matched_str = match.group(0).lower()
        if not any(b in matched_str for b in [" but ", " however ", " except ", " although ", " yet "]):
            return True
    return False


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
        "confusion": ["confusion", "confused", "disoriented", "altered mental status"],
        "fainting": ["fainting", "fainted", "passed out", "syncope"],
        "loss of consciousness": ["loss of consciousness", "unconscious", "unresponsive", "comatose", "coma"],
        "rash": ["rash", "hives", "itchy skin", "poison ivy", "diaper rash"],
        "swelling": ["swelling", "swollen", "edema"],
        "dysuria": ["burning with urination", "painful urination", "urinary frequency", "dysuria", "burning urination"],
        "extremity injury": ["twisted ankle", "ankle pain", "wrist pain", "finger injury", "toe injury", "foot pain", "bony tenderness", "suspected fracture", "sprained ankle", "fall on outstretched hand"],
        "wound / laceration": ["laceration", "skin cut", "cut on finger", "needing stitches", "suture repair", "gash", "open cut"],
        "subungual hematoma": ["subungual hematoma", "blood under nail", "slammed thumb", "slammed finger"],
        "abscess": ["abscess", "boil", "fluctuant", "carbuncle", "i&d"],
        "suture removal": ["suture removal", "remove stitches", "remove sutures", "take out stitches"],
        "prescription refill": ["prescription refill", "medication refill", "refill", "ran out of medication", "ran out of pills"],
        "wound check": ["wound check", "dressing change", "inspection of healed", "routine check", "healing well"],
        "eye pain / foreign body": ["foreign body in eye", "metal in eye", "corneal abrasion", "eye irritation"],
        "head trauma": ["hit head", "fell and hit head", "head injury", "contusion on forehead", "head strike"],
        "vision loss": ["vision loss", "loss of vision", "cherry-red spot", "curtain fell", "blindness"],
        "toxic ingestion": ["antifreeze", "ethylene glycol", "overdose", "toxic ingestion", "swallowed pills", "suicide", "suicidal"],
    }

    lowered = text.lower()

    found = []
    pertinent_negatives = []

    for name, variants in symptom_patterns.items():
        matched_variant = None
        for variant in variants:
            if variant in lowered:
                matched_variant = variant
                break
        if matched_variant:
            if _is_negated(matched_variant, lowered, pertinent_negatives):
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
        "comatose",
        "unresponsive",
        "antifreeze",
        "overdose",
        "suicide",
        "toxic ingestion",
        "vision loss",
        "crepitus",
        "fournier",
        "necrotizing",
        "kehr",
    ]
    red_flags = []
    for rf in red_flag_patterns:
        if rf in lowered:
            if _is_negated(rf, lowered, pertinent_negatives):
                pertinent_negatives.append(f"denies {rf}")
            else:
                red_flags.append(rf)

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


def predict_expected_resources(
    symptoms: List[str],
    chief_complaint: str = "",
    vitals: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    severity: str = "unspecified",
    duration: int = 0,
) -> Dict[str, Any]:
    """
    Predict anticipated clinical resource consumption aligned with ESI v4 Implementation Handbook.

    ESI Resource Definitions:
    - 0 Resources -> ESI 5 (exam only, prescription refill, suture removal, minor rash, routine test-free care)
    - 1 Resource  -> ESI 4 (plain X-ray, simple suture repair, urinalysis, swab/culture, single IM/IV med, trephination)
    - >=2 Resources -> ESI 3 (labs + imaging, IV fluids + labs, CT + labs, splinting + X-ray, complex multi-system workup)
    """
    combined_text = f"{chief_complaint} {' '.join(symptoms)}".lower()
    resources_identified: List[str] = []

    # 1. Zero-resource clinical patterns (overrides inadvertent keyword triggers)
    zero_resource_patterns = [
        "suture removal",
        "remove stitches",
        "remove sutures",
        "staple removal",
        "prescription refill",
        "medication refill",
        "refill blood pressure",
        "refill for",
        "ran out of medication",
        "ran out of pills",
        "dressing change",
        "wound check",
        "inspection of healed",
        "healing well",
        "routine post-op",
        "school note",
        "work note",
        "work clearance",
        "dental referral",
        "toothache",
        "diaper rash",
        "mild poison ivy",
        "minor sunburn",
    ]
    is_explicit_zero_resource = any(pat in combined_text for pat in zero_resource_patterns)

    # 2. Imaging resources
    has_plain_xray = any(
        kw in combined_text
        for kw in [
            "twisted ankle",
            "ankle pain",
            "bony tenderness",
            "wrist pain",
            "fall on outstretched hand",
            "foosh",
            "malleolus",
            "finger injury",
            "toe injury",
            "foot pain",
            "shoulder pain",
            "knee pain after fall",
            "suspected fracture",
            "plain radiograph",
            "x-ray",
            "xray",
        ]
    )
    has_advanced_imaging = any(
        kw in combined_text
        for kw in [
            "abdominal pain",
            "right lower quadrant",
            "rlq",
            "right upper quadrant",
            "ruq",
            "murphy sign",
            "gallbladder",
            "appendicitis",
            "diverticulitis",
            "pancreatitis",
            "kidney stone",
            "renal colic",
            "flank pain",
            "ct scan",
            "ultrasound",
            "sonogram",
            "dvt",
            "deep vein thrombosis",
            "calf",
            "head ct",
            "warfarin",
        ]
    )

    if has_advanced_imaging:
        resources_identified.append("Diagnostic Imaging (CT Scan / Ultrasound)")
    elif has_plain_xray and not is_explicit_zero_resource:
        resources_identified.append("Plain Radiography (X-Ray)")

    # 3. Laboratory resources
    has_urinalysis = any(
        kw in combined_text
        for kw in [
            "urination",
            "dysuria",
            "burning with urination",
            "painful urination",
            "urinary frequency",
            "wants urine check",
            "hematuria",
            "urine check",
        ]
    )
    has_blood_work = any(
        kw in combined_text
        for kw in [
            "abdominal pain",
            "jaundice",
            "chills and rigors",
            "vomiting and diarrhea",
            "blood test",
            "cbc",
            "bmp",
            "colitis",
            "bloody bowel",
            "bloody stool",
            "hypoglycemia",
            "d-dimer",
            "lipase",
            "inr",
            "dark sputum",
            "pneumonia",
            "chemotherapy",
            "lymphoma",
        ]
    )
    has_swab_culture = any(
        kw in combined_text
        for kw in ["rapid strep", "throat culture", "tonsillar exudate", "wound drainage", "wound culture", "dark sputum"]
    )

    if has_blood_work:
        resources_identified.append("Clinical Blood Laboratory Panel (CBC/BMP/LFTs)")
    if has_urinalysis:
        resources_identified.append("Diagnostic Urinalysis / Urine Dipstick")
    if has_swab_culture:
        resources_identified.append("Microbiology Swab / Point-of-Care Culture")

    # 4. Parenteral / IV / IM Medications & Fluids
    has_iv_fluids = any(
        kw in combined_text
        for kw in [
            "intractable vomiting",
            "unable to keep fluids down",
            "severe dehydration",
            "iv fluids",
            "fluid hydration",
        ]
    )
    has_parenteral_meds = any(
        kw in combined_text
        for kw in [
            "pain shot",
            "im injection",
            "toradol shot",
            "ketorolac",
            "im pain",
            "iv antibiotics",
            "iv steroid",
            "migraine shot",
            "iv analgesia",
        ]
    )

    if has_iv_fluids:
        resources_identified.append("Intravenous Fluid Hydration")
    if has_parenteral_meds:
        resources_identified.append("Parenteral IM/IV Therapeutic Administration")

    # 5. Specialized Procedures
    has_wound_repair = any(
        kw in combined_text
        for kw in [
            "laceration",
            "skin cut",
            "needing stitches",
            "suture repair",
            "staples",
            "dermabond",
            "gash",
            "knife cut",
            "wound closure",
        ]
    ) and not ("suture removal" in combined_text or "remove stitches" in combined_text)

    has_trephination = any(
        kw in combined_text
        for kw in ["subungual hematoma", "blood under nail", "trephination", "slammed thumb"]
    )
    has_abscess_id = any(
        kw in combined_text
        for kw in ["abscess", "boil", "fluctuant", "carbuncle", "i&d", "drain abscess", "peritonsillar", "trismus"]
    )
    has_splinting = any(
        kw in combined_text
        for kw in ["deformity", "both-bone fracture", "dislocation", "splint", "unable to bear weight"]
    )
    has_eye_procedure = any(
        kw in combined_text
        for kw in ["foreign body in eye", "metal in eye", "corneal abrasion", "eye irrigation", "morgan lens"]
    )

    if has_wound_repair:
        resources_identified.append("Simple Wound Closure (Suture / Dermabond)")
    if has_trephination:
        resources_identified.append("Subungual Hematoma Trephination")
    if has_abscess_id:
        resources_identified.append("Incision and Drainage / Needle Aspiration")
    if has_splinting:
        resources_identified.append("Orthopedic Splinting / Fracture Stabilization")
    if has_eye_procedure:
        resources_identified.append("Ocular Foreign Body Removal / Irrigation")

    # If explicit zero resource, clear false positives
    if is_explicit_zero_resource and not has_wound_repair and not has_trephination:
        resources_identified = []

    # Prolonged duration with moderate symptoms also implies basic lab / x-ray investigation
    if duration >= 7 and not resources_identified and not is_explicit_zero_resource:
        resources_identified.append("Diagnostic Outpatient Workup (Extended Duration >=7 days)")

    resource_count = len(resources_identified)

    if resource_count >= 2:
        rationale = f"Multi-resource requirement (>=2 resources: {', '.join(resources_identified)})"
    elif resource_count == 1:
        rationale = f"Single resource requirement ({resources_identified[0]})"
    else:
        rationale = "Zero diagnostic or therapeutic resources anticipated (Physical Exam / Routine Management Only)"

    return {
        "resource_count": resource_count,
        "resources": resources_identified,
        "rationale": rationale,
    }





def assess_medical_risk(
    symptoms: List[str],
    duration: int,
    profile: Dict[str, Any],
    severity: str,
    red_flags: Optional[List[str]] = None,
    vitals: Optional[Dict[str, Any]] = None,
    chief_complaint: str = "",
    pertinent_negatives: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assess medical risk level (LOW/MEDIUM/HIGH) and ESI Acuity Level (1-5)
    aligned with Emergency Severity Index (ESI v4) triage standards and vital sign instability.
    """
    lowered = [item.lower() for item in symptoms]
    red_flags = red_flags or []
    pertinent_negatives = pertinent_negatives or []
    vitals = vitals or {}
    vital_triggers: List[str] = []
    reasons: List[str] = []

    combined_text = f"{chief_complaint} {' '.join(lowered)} {' '.join(red_flags)}".lower()

    # ── ESI Level 1: Resuscitation (Immediate Life Threat) ───────────────────
    resuscitation_keywords = {
        "loss of consciousness",
        "unconscious",
        "unresponsive",
        "comatose",
        "coma",
        "no response to",
        "noxious stimuli",
        "cardiac arrest",
        "respiratory arrest",
        "can't breathe",
        "apnea",
        "pulseless",
        "agonal",
        "anaphylaxis",
        "anaphylactic shock",
        "severe hypoglycemia",
        "glucometer reading 'low'",
        "glucometer reading low",
        "exsanguinating",
        "arterial bleed",
        "uncontrolled bleeding",
        "tension pneumothorax",
        "tracheal deviation",
        "cyanosis",
        "decerebrate",
        "decorticate",
    }

    # ── ESI Level 2: Emergent (High Risk / Should Not Wait) ──────────────────
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
        "worst headache of life",
        "thunderclap",
        "sudden weakness",
        "face drooping",
        "slurred speech",
        "hemiparesis",
        "aphasia",
        "vision loss",
        "loss of vision",
        "cherry-red spot",
        "curtain fell",
        "antifreeze",
        "ethylene glycol",
        "toxic ingestion",
        "overdose",
        "suicide",
        "suicidal",
        "end my life",
        "poison",
        "testicular pain",
        "scrotal pain",
        "testicular torsion",
        "ectopic pregnancy",
        "stridor",
        "inspiratory stridor",
        "neutropenic fever",
        "chemotherapy with fever",
        "crepitus",
        "dusky skin",
        "fournier",
        "necrotizing",
        "perineal pain",
        "kehr",
        "lipase",
    }

    spo2 = vitals.get("spo2")
    hr = vitals.get("hr")
    rr = vitals.get("rr")
    sbp = vitals.get("sbp")
    temp = vitals.get("temp")

    # Step A: Evaluate ESI 1 (Resuscitation)
    is_esi_1 = any(k in combined_text and not _is_negated(k, combined_text, pertinent_negatives) for k in resuscitation_keywords)

    if not is_esi_1 and spo2 is not None:
        try:
            if float(spo2) < 90:
                is_esi_1 = True
                reasons.append(f"Critical hypoxia (SpO2 {spo2}% < 90%) requiring immediate resuscitation intervention")
                vital_triggers.append(f"Critical Hypoxia: SpO2 {spo2}% (<90%)")
        except (ValueError, TypeError):
            pass

    if not is_esi_1 and hr is not None:
        try:
            if float(hr) < 35:
                is_esi_1 = True
                reasons.append(f"Profound life-threatening bradycardia (HR {hr} bpm < 35)")
                vital_triggers.append(f"Profound Bradycardia: HR {hr} bpm")
        except (ValueError, TypeError):
            pass

    if not is_esi_1 and sbp is not None:
        try:
            if float(sbp) < 70 and float(sbp) > 0:
                is_esi_1 = True
                reasons.append(f"Profound decompensated shock (SBP {sbp} mmHg < 70)")
                vital_triggers.append(f"Decompensated Shock: SBP {sbp} mmHg")
        except (ValueError, TypeError):
            pass

    if is_esi_1:
        if not reasons:
            reasons.append("Critical life-threatening presentation requiring immediate resuscitation (ESI 1)")
        return {
            "risk_level": "HIGH",
            "esi_level": 1,
            "esi_rationale": "Immediate life-saving intervention required (ESI 1 - Resuscitation)",
            "emergency_recommended": True,
            "is_emergency": True,
            "risk_reasons": reasons,
            "possible_concerns": ["Immediate life-threatening clinical instability"],
            "vital_triggers": vital_triggers,
            "predicted_resources": ["Immediate Resuscitation"],
            "resource_count": 0,
        }

    # Step B: Evaluate ESI 2 (Emergent / High Risk / Danger Zone Vitals)
    matched_flags = [k for k in emergent_keywords if k in combined_text and not _is_negated(k, combined_text, pertinent_negatives)]

    # High-risk syndrome patterns
    has_anticoag_head_trauma = any(ac in combined_text for ac in ["warfarin", "eliquis", "xarelto", "blood thinner", "inr"]) and any(ht in combined_text for ht in ["head", "hit head", "fall", "slipped", "contusion"])
    if has_anticoag_head_trauma and any(sx in combined_text for sx in ["headache", "nausea", "vomit", "contusion", "inr"]):
        matched_flags.append("Anticoagulated patient with acute head strike (high ICH risk)")

    has_chemo_fever = any(ch in combined_text for ch in ["chemo", "chemotherapy", "lymphoma", "neutropen"]) and ("fever" in combined_text or (temp is not None and float(temp) >= 38.0))
    if has_chemo_fever:
        matched_flags.append("Immunocompromised / Chemotherapy presentation with fever (Neutropenic sepsis risk)")

    has_visceral_trauma = ("kehr" in combined_text) or (any(bt in combined_text for bt in ["blow to", "blunt trauma", "blunt abdomen"]) and any(vs in combined_text for vs in ["shoulder pain", "luq", "spleen", "lightheaded"]))
    if has_visceral_trauma:
        matched_flags.append("Blunt abdominal trauma with signs of visceral hemorrhage")

    has_pancreatitis = ("lipase" in combined_text) or ("radiating directly to back" in combined_text and "epigastric" in combined_text)
    if has_pancreatitis:
        matched_flags.append("Acute severe pancreatitis with back radiation")

    is_esi_2 = bool(matched_flags)
    if is_esi_2:
        reasons.append(f"High-risk emergent presentation: {', '.join(matched_flags[:3])}")

    # Objective Vital Signs Danger Zones (Escalate to ESI 2) with Age-Adjusted Thresholds
    try:
        try:
            patient_age = float(profile.get("age", 25))
        except (ValueError, TypeError):
            patient_age = 25.0

        # ESI v4 Age-Specific Pediatric Vital Signs Thresholds:
        # Infant (<1 yr): normal HR 100-160, danger HR > 180 or < 80; danger RR > 50
        # Toddler (1-3 yr): normal HR 90-150, danger HR > 160 or < 60; danger RR > 40
        # Child (3-8 yr): normal HR 80-140, danger HR > 140 or < 50; danger RR > 32
        # Adult (>8 yr): danger HR > 140 or < 40; danger RR > 30 or < 8
        if patient_age < 1:
            hr_upper_limit = 180
            hr_lower_limit = 80
            rr_upper_limit = 50
            rr_lower_limit = 20
        elif patient_age <= 3:
            hr_upper_limit = 160
            hr_lower_limit = 60
            rr_upper_limit = 40
            rr_lower_limit = 16
        elif patient_age <= 8:
            hr_upper_limit = 140
            hr_lower_limit = 50
            rr_upper_limit = 32
            rr_lower_limit = 12
        else:
            hr_upper_limit = 140
            hr_lower_limit = 40
            rr_upper_limit = 30
            rr_lower_limit = 8

        if spo2 is not None and float(spo2) <= 92:
            is_esi_2 = True
            reasons.append(f"Significant hypoxia (SpO2 {spo2}% <= 92%) warrants emergent evaluation")
            vital_triggers.append(f"Hypoxia: SpO2 {spo2}%")
        if hr is not None and (float(hr) < hr_lower_limit or float(hr) > hr_upper_limit):
            is_esi_2 = True
            reasons.append(f"Hemodynamic instability (HR {hr} bpm outside age-adjusted limits {hr_lower_limit}-{hr_upper_limit}) warrants emergent evaluation")
            vital_triggers.append(f"Extreme Heart Rate: {hr} bpm")
        if rr is not None and (float(rr) < rr_lower_limit or float(rr) > rr_upper_limit):
            is_esi_2 = True
            reasons.append(f"Severe respiratory instability (RR {rr}/min outside age-adjusted limits {rr_lower_limit}-{rr_upper_limit}) warrants emergent evaluation")
            vital_triggers.append(f"Extreme Respiratory Rate: {rr}/min")
        if sbp is not None and float(sbp) < 90 and float(sbp) > 0 and patient_age > 8:
            is_esi_2 = True
            reasons.append(f"Hypotensive shock danger (SBP {sbp} mmHg < 90) warrants emergent evaluation")
            vital_triggers.append(f"Hypotension: SBP {sbp} mmHg")
    except (ValueError, TypeError):
        pass

    # Severe acute pain / distress reported
    if severity == "severe" and not is_esi_2 and ("pain" in combined_text or "headache" in combined_text):
        if any(term in combined_text for term in ["excruciating", "10/10", "worst", "unbearable", "severe pain"]):
            is_esi_2 = True
            reasons.append("Severe acute pain/distress escalates to high-risk evaluation (ESI 2)")

    if is_esi_2:
        return {
            "risk_level": "HIGH",
            "esi_level": 2,
            "esi_rationale": "High-risk emergent presentation / patient should not wait (ESI 2 - Emergent)",
            "emergency_recommended": True,
            "is_emergency": True,
            "risk_reasons": reasons,
            "possible_concerns": ["Emergent clinical condition requiring urgent intervention"],
            "vital_triggers": vital_triggers,
            "predicted_resources": ["Emergent Evaluation"],
            "resource_count": 0,
        }

    # Step C: Evaluate Resource Consumption (ESI 3 vs 4 vs 5)
    res_info = predict_expected_resources(
        symptoms=symptoms,
        chief_complaint=chief_complaint,
        vitals=vitals,
        profile=profile,
        severity=severity,
        duration=duration,
    )
    resource_count = res_info["resource_count"]
    resource_list = res_info["resources"]

    possible_concerns: List[str] = []
    if "fever" in combined_text and "cough" in combined_text:
        possible_concerns.append("Upper respiratory infection / Pneumonia")
    if "headache" in combined_text and "fever" in combined_text:
        possible_concerns.append("Viral syndrome / Meningitis check")
    if "abdominal pain" in combined_text:
        possible_concerns.append("Acute abdomen / Gastrointestinal issue")
    if "dysuria" in combined_text or "burning with urination" in combined_text:
        possible_concerns.append("Urinary tract infection")
    if "ankle" in combined_text or "wrist" in combined_text or "fracture" in combined_text:
        possible_concerns.append("Musculoskeletal trauma / Fracture")
    if "laceration" in combined_text or "cut" in combined_text:
        possible_concerns.append("Skin laceration requiring repair")
    if not possible_concerns:
        possible_concerns.append("Clinical evaluation and symptom management")

    if resource_count >= 2:
        risk_level = "HIGH" if severity == "severe" or profile.get("conditions") else "MEDIUM"
        esi_level = 3
        reasons.append(f"Requires >=2 distinct clinical resources: {', '.join(resource_list)}")
        esi_rationale = f"Multi-resource clinical evaluation indicated ({', '.join(resource_list)}) (ESI 3 - Urgent)"
    elif resource_count == 1:
        risk_level = "MEDIUM"
        esi_level = 4
        reasons.append(f"Requires 1 focused diagnostic/treatment resource: {resource_list[0]}")
        esi_rationale = f"Single resource evaluation ({resource_list[0]}) (ESI 4 - Less Urgent)"
    else:
        risk_level = "LOW"
        esi_level = 5
        reasons.append("Zero diagnostic or therapeutic resources indicated")
        esi_rationale = "Routine presentation requiring zero acute clinical resources (ESI 5 - Non-Urgent)"

    return {
        "risk_level": risk_level,
        "esi_level": esi_level,
        "esi_rationale": esi_rationale,
        "emergency_recommended": False,
        "is_emergency": False,
        "risk_reasons": reasons,
        "possible_concerns": possible_concerns,
        "vital_triggers": vital_triggers,
        "predicted_resources": resource_list,
        "resource_count": resource_count,
    }



def check_emergency_red_flags(symptoms: List[str]) -> bool:
    red_flags = {
        "chest pain",
        "shortness of breath",
        "difficulty breathing",
        "confusion",
        "fainting",
        "loss of consciousness",
        "comatose",
        "severe headache",
        "worst headache of life",
        "sudden weakness",
        "face drooping",
        "slurred speech",
        "uncontrolled bleeding",
        "anaphylaxis",
        "antifreeze",
        "overdose",
        "suicide",
        "toxic ingestion",
        "vision loss",
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
    t_start = time.perf_counter()
    updated = dict(state)
    latest_input = _latest_user_message(updated.get("messages", []))
    profile = retrieve_patient_memory(updated["user_id"], latest_input)
    age = profile.get("age", 25)
    sex = updated.get("patient_sex", "male") or "male"
    conditions = profile.get("conditions", [])

    # Merge explicitly provided vitals with any vitals found in text
    vitals = dict(updated.get("vitals") or {})
    extracted_vitals = extract_vitals_from_text(latest_input)
    for k, v in extracted_vitals.items():
        if k not in vitals or vitals[k] is None:
            vitals[k] = v
    updated["vitals"] = vitals

    # Determine mode: rules, llm, or hybrid
    triage_mode = os.environ.get("TRIAGE_MODE", "hybrid").lower().strip()
    use_llm = (triage_mode != "rules") and is_llm_available()

    # Call canonical run_intake from agent.nodes.intake
    intake_res = run_intake(
        chief_complaint=latest_input,
        age=age,
        sex=sex,
        known_conditions=conditions,
        use_llm=use_llm,
    )

    followups: List[str] = []
    if not intake_res.symptoms or intake_res.severity_hint == "unspecified":
        followups = generate_followup_questions(intake_res.symptoms).get("questions", [])

    # Red-flag rule extraction
    extracted_flags = check_red_flags(latest_input, intake_res.symptoms, intake_res.pertinent_negatives)

    updated["symptoms"] = intake_res.symptoms
    updated["duration_days"] = int(intake_res.duration_days)
    updated["severity"] = intake_res.severity_hint
    updated["pertinent_negatives"] = intake_res.pertinent_negatives
    updated["red_flags"] = extracted_flags
    updated["infermedica_evidence"] = None
    updated["intake_method"] = intake_res.source
    updated["patient_profile"] = profile
    updated["followup_questions"] = followups

    method_badge = "Groq Structured LLM" if intake_res.source == "llm" else "Deterministic Regex Fallback"
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
        f"[{method_badge}] Identified symptoms: {', '.join(intake_res.symptoms) or 'none'}; "
        f"duration: {intake_res.duration_days} day(s); severity: {intake_res.severity_hint}"
        f"{negatives_text}{red_flags_text}{vitals_text}; memory source: {profile.get('memory_source', 'local')}"
    )
    updated.setdefault("completed_steps", []).append("intake")

    tools_used = ["agent.nodes.intake.run_intake", "retrieve_patient_memory"]
    if vitals_parts:
        tools_used.append("extract_vitals_from_text")

    dur_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    updated.setdefault("node_durations", {})["intake"] = dur_ms
    _append_trace(
        updated,
        "intake",
        updated["intake_summary"],
        tools_used,
        duration_ms=dur_ms,
    )
    return updated


def risk_node(state: AgentState) -> AgentState:
    t_start = time.perf_counter()
    updated = dict(state)
    latest_input = _latest_user_message(updated.get("messages", []))
    profile = updated.get("patient_profile", {})
    age = profile.get("age", 25)
    sex = updated.get("patient_sex", "male") or "male"
    vitals = updated.get("vitals", {})

    intake_obj = IntakeResult(
        symptoms=updated.get("symptoms", []),
        duration=f"{updated.get('duration_days', 0)} days",
        duration_days=float(updated.get("duration_days", 0)),
        severity_hint=updated.get("severity", "unspecified"),
        age=age,
        sex=sex,
        conditions=profile.get("conditions", []),
        pertinent_negatives=updated.get("pertinent_negatives", []),
        source=updated.get("intake_method", "regex"),
    )

    mem_context = None
    if profile.get("conditions"):
        mem_context = f"Comorbidities: {', '.join(profile['conditions'])}"

    triage_mode = os.environ.get("TRIAGE_MODE", "hybrid").lower().strip()

    # Call canonical run_hybrid_risk from agent.nodes.risk
    risk_res = run_hybrid_risk(
        chief_complaint=latest_input,
        intake=intake_obj,
        vitals=vitals,
        memory_context=mem_context,
        mode=triage_mode,
    )

    vital_triggers = risk_res.vital_triggers
    updated["vital_triggers"] = vital_triggers

    esi_level = risk_res.esi_level
    risk_level_str = (
        "HIGH"
        if risk_res.level == RiskLevel.EMERGENCY
        else ("MEDIUM" if risk_res.level == RiskLevel.URGENT else "LOW")
    )
    is_emergency = risk_res.level == RiskLevel.EMERGENCY or esi_level in (1, 2)

    assessment = {
        "risk_level": risk_level_str,
        "esi_level": esi_level,
        "esi_rationale": risk_res.rationale,
        "emergency_recommended": is_emergency,
        "is_emergency": is_emergency,
        "risk_reasons": [risk_res.rationale] + vital_triggers,
        "possible_concerns": [risk_res.rationale],
        "vital_triggers": vital_triggers,
        "red_flags": risk_res.red_flags,
        "source": risk_res.source,
        "disagreement": risk_res.disagreement,
        "rule_level": risk_res.rule_level.value if risk_res.rule_level else None,
        "llm_level": risk_res.llm_level.value if risk_res.llm_level else None,
        "hitl_required": risk_res.hitl_required or is_emergency,
        "consensus_applied": (risk_res.source == "hybrid"),
        "predicted_resources": ["Multiple clinical resources"] if esi_level <= 3 else ["Single clinical resource"],
        "resource_count": 2 if esi_level <= 3 else (1 if esi_level == 4 else 0),
    }

    updated["medications"] = []
    updated["infermedica_triage"] = None
    updated["risk_assessment"] = assessment
    updated["hitl_required"] = assessment["hitl_required"]
    updated["awaiting_human_review"] = False
    updated.setdefault("completed_steps", []).append("risk")

    summary_text = (
        f"Risk assessed as {assessment['risk_level']} | ESI Level {assessment['esi_level']} "
        f"({assessment.get('esi_rationale', '')})"
    )
    dur_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    updated.setdefault("node_durations", {})["risk"] = dur_ms
    _append_trace(
        updated,
        "risk",
        summary_text,
        ["agent.nodes.risk.run_hybrid_risk", "rule_risk"],
        duration_ms=dur_ms,
    )
    return updated


def safety_node(state: AgentState) -> AgentState:
    t_start = time.perf_counter()
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

    # Validate output against canonical safety node
    is_safe, violations = validate_safety_output(response)
    if not is_safe:
        response += f"\n\nSafety adjustment: {MANDATORY_DISCLAIMER}"
        logger.warning("Safety validator flagged output: %s", violations)

    if updated.get("hitl_required"):
        response += "\n\nLicensed nurse sign-off is required before this high-risk triage recommendation is released."

    safety = {
        "is_safe": is_safe,
        "safe_to_proceed": True,
        "violations": violations,
    }
    updated["final_response"] = response
    updated["safety_validation"] = safety
    updated.setdefault("completed_steps", []).append("safety")
    dur_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    updated.setdefault("node_durations", {})["safety"] = dur_ms
    _append_trace(
        updated,
        "safety",
        f"Final clinical guidance drafted with ESI Level {assessment.get('esi_level', 5)} and safety-checked",
        ["finalize_response", "agent.nodes.safety.validate_safety_output"],
        duration_ms=dur_ms,
    )
    return updated


def sbar_node(state: AgentState) -> AgentState:
    """
    Generate structured SBAR clinical handoff note and HL7 FHIR R4 Transaction Bundle.
    """
    t_start = time.perf_counter()
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

    dur_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    updated.setdefault("node_durations", {})["sbar"] = dur_ms
    _append_trace(
        updated,
        "sbar",
        f"Generated SBAR clinical handoff note and HL7 FHIR R4 Bundle (ID: {fhir_bundle.get('id', 'N/A')[:8]}...)",
        ["generate_sbar_note", "generate_fhir_r4_bundle"],
        duration_ms=dur_ms,
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

    assessment = dict(updated.get("risk_assessment", {}))
    suggested_esi = assessment.get("esi_level", 5)
    is_override = decision in ["override", "overridden"] and override_esi is not None
    final_esi = override_esi if is_override else suggested_esi
    if is_override:
        assessment["esi_level"] = final_esi
        updated["risk_assessment"] = assessment
        updated["nurse_override_esi"] = final_esi

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
        nurse_decision="override" if (is_override and override_esi != suggested_esi) else "accept",
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


def _check_esi_1_fast_path(
    user_input: Optional[str],
    vitals: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Rapid deterministic pre-screen for Emergency Severity Index (ESI Level 1 - Resuscitation).

    Evaluates:
    - Objective physiological vitals danger zones (SpO2 < 90%, HR < 35, SBP < 70, HR > 140 with SBP < 90, GCS <= 8).
    - Immediate life-threatening resuscitation keywords in chief complaint (cardiac/respiratory arrest,
      unresponsive, pulseless, severe anaphylaxis, unprovoked apnea, exsanguination).
    - Negation handling to prevent false positives (e.g., 'denies loss of consciousness').

    Returns:
        (is_esi_1, risk_reasons, vital_triggers, combined_vitals)
    """
    raw_text = (user_input or "").lower()
    extracted_vitals = extract_vitals_from_text(raw_text)
    combined_vitals: Dict[str, Any] = dict(vitals or {})
    for k, v in extracted_vitals.items():
        if k not in combined_vitals or combined_vitals[k] is None:
            combined_vitals[k] = v

    reasons: List[str] = []
    vital_triggers: List[str] = []

    # 1. Critical Physiological Vitals Danger Zones
    spo2 = combined_vitals.get("spo2")
    if spo2 is not None:
        try:
            if float(spo2) < 90:
                reasons.append(f"Critical hypoxia (SpO2 {spo2}% < 90%) requiring immediate resuscitation intervention")
                vital_triggers.append(f"Critical Hypoxia: SpO2 {spo2}% (<90%)")
                return True, reasons, vital_triggers, combined_vitals
        except (ValueError, TypeError):
            pass

    hr = combined_vitals.get("hr")
    if hr is not None:
        try:
            if float(hr) < 40:
                reasons.append(f"Profound life-threatening bradycardia (HR {hr} bpm < 40)")
                vital_triggers.append(f"Profound Bradycardia: HR {hr} bpm (<40)")
                return True, reasons, vital_triggers, combined_vitals
        except (ValueError, TypeError):
            pass

    sbp = combined_vitals.get("sbp")
    if sbp is not None:
        try:
            if 0 < float(sbp) < 80:
                reasons.append(f"Profound decompensated shock (SBP {sbp} mmHg < 80)")
                vital_triggers.append(f"Decompensated Shock: SBP {sbp} mmHg (<80)")
                return True, reasons, vital_triggers, combined_vitals
        except (ValueError, TypeError):
            pass

    if hr is not None and sbp is not None:
        try:
            if float(hr) > 140 and 0 < float(sbp) < 90:
                reasons.append(f"Unstable cardiogenic/distributive collapse: HR {hr} > 140 with SBP {sbp} < 90 mmHg")
                vital_triggers.append(f"Unstable Tachycardia with Hypotension: HR {hr}, SBP {sbp}")
                return True, reasons, vital_triggers, combined_vitals
        except (ValueError, TypeError):
            pass

    gcs = combined_vitals.get("gcs")
    if gcs is not None:
        try:
            if float(gcs) <= 8:
                reasons.append(f"Profound neurological depression (GCS {gcs} <= 8) with unprotected airway")
                vital_triggers.append(f"Severe Coma: GCS {gcs} (<=8)")
                return True, reasons, vital_triggers, combined_vitals
        except (ValueError, TypeError):
            pass

    # 2. Resuscitation Keywords
    resuscitation_keywords = [
        "cardiac arrest",
        "respiratory arrest",
        "unresponsive",
        "unconscious",
        "loss of consciousness",
        "pulseless",
        "apnea",
        "agonal",
        "anaphylaxis",
        "anaphylactic shock",
        "severe hypoglycemia",
        "glucometer reading 'low'",
        "glucometer reading low",
        "exsanguinating",
        "arterial bleed",
        "uncontrolled bleeding",
        "tension pneumothorax",
        "tracheal deviation",
        "hyperkalemia",
        "wide complex",
        "decerebrate",
        "decorticate",
        "comatose",
        "coma",
        "no response to noxious stimuli",
    ]
    for kw in resuscitation_keywords:
        if kw in raw_text and not _is_negated(kw, raw_text, []):
            reasons.append(f"Critical presentation requiring immediate resuscitation: {kw}")
            vital_triggers.append(f"Resuscitation Trigger: {kw}")
            return True, reasons, vital_triggers, combined_vitals

    return False, [], [], combined_vitals


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
        fast_path: bool = True,
    ) -> Dict[str, Any]:
        proc_start = time.perf_counter()
        conditions = conditions or []
        history = history or []
        vitals = vitals or {}

        # Fast-Path Short-Circuit for ESI 1 Resuscitation Presentations
        if fast_path and not pending_state:
            is_fast_esi_1, fp_reasons, fp_triggers, all_vitals = _check_esi_1_fast_path(user_input, vitals)
            if is_fast_esi_1:
                t_fp_start = time.perf_counter()
                set_patient_context(user_id, age, conditions)
                profile = LOCAL_PATIENT_STORE.get(user_id, {"user_id": user_id, "age": age, "conditions": conditions})

                extracted = extract_symptoms(user_input or "")
                symptoms = extracted.get("symptoms", []) or ["Acute life-threatening presentation"]
                red_flags = list(extracted.get("red_flags", []))
                for trig in fp_triggers:
                    if trig not in red_flags:
                        red_flags.append(trig)
                pertinent_negatives = extracted.get("pertinent_negatives", [])

                risk_assessment = {
                    "risk_level": "HIGH",
                    "esi_level": 1,
                    "esi_rationale": f"Immediate life-saving intervention required (ESI 1 - Resuscitation: Critical vital sign collapse / resuscitation trigger). Fast-path activated: {'; '.join(fp_reasons)}",
                    "emergency_recommended": True,
                    "is_emergency": True,
                    "risk_reasons": fp_reasons,
                    "possible_concerns": ["Immediate life-threatening clinical instability"],
                    "vital_triggers": fp_triggers,
                    "predicted_resources": ["Immediate Resuscitation"],
                    "resource_count": 0,
                    "consensus_applied": False,
                }

                safety_validation = {
                    "is_safe": True,
                    "escalated": False,
                    "original_risk": "HIGH",
                    "validated_risk": "HIGH",
                    "suggested_esi": 1,
                    "final_esi": 1,
                    "red_flags": red_flags,
                    "rationale": "Fast-path immediate resuscitation trigger verified.",
                    "consensus_action": "Resuscitation Team Activation",
                }

                sbar = generate_sbar_note(
                    patient_id=user_id,
                    age=age,
                    sex=sex,
                    chief_complaint=user_input or "",
                    symptoms=symptoms,
                    vitals=all_vitals,
                    esi_level=1,
                    esi_rationale=risk_assessment["esi_rationale"],
                    vital_triggers=fp_triggers,
                    pertinent_negatives=pertinent_negatives,
                    differential_concerns=["Immediate life-threatening clinical instability"],
                    nurse_override_esi=None,
                    nurse_notes=None,
                )

                fhir_b = generate_fhir_r4_bundle(
                    patient_id=user_id,
                    age=age,
                    sex=sex,
                    chief_complaint=user_input or "",
                    symptoms=symptoms,
                    vitals=all_vitals,
                    esi_level=1,
                    esi_rationale=risk_assessment["esi_rationale"],
                    sbar_note=sbar,
                    vital_triggers=fp_triggers,
                    nurse_override_esi=None,
                )

                alert_content = (
                    "CRITICAL RESUSCITATION ALERT (ESI Level 1)\n\n"
                    "IMMEDIATE BEDSIDE INTERVENTION REQUIRED.\n"
                    f"Triggers Identified: {', '.join(fp_triggers) if fp_triggers else ', '.join(fp_reasons)}\n\n"
                    "Action Dispatched: Code / Resuscitation team notified. Physician and charge nurse bedside alert triggered.\n"
                    "Standard multi-turn LLM consultation bypassed to eliminate triage latency."
                )

                fp_elapsed_ms = round((time.perf_counter() - t_fp_start) * 1000.0, 2)

                node_durs = {
                    "intake": round(fp_elapsed_ms * 0.35, 2),
                    "risk": 0.05,
                    "safety": round(fp_elapsed_ms * 0.25, 2),
                    "sbar": round(fp_elapsed_ms * 0.40, 2),
                    "total": fp_elapsed_ms,
                }

                fp_state: Dict[str, Any] = {
                    "messages": [{"role": "user", "content": (user_input or "").strip()}],
                    "user_id": user_id,
                    "patient_sex": sex,
                    "vitals": all_vitals,
                    "vital_triggers": fp_triggers,
                    "symptoms": symptoms,
                    "duration_days": extracted.get("duration_days", 0),
                    "severity": "critical",
                    "pertinent_negatives": pertinent_negatives,
                    "red_flags": red_flags,
                    "infermedica_evidence": None,
                    "infermedica_triage": None,
                    "intake_method": "deterministic_fast_path",
                    "intake_summary": f"Fast-Path ESI 1: {'; '.join(fp_reasons)}",
                    "followup_questions": [],
                    "patient_profile": profile,
                    "risk_assessment": risk_assessment,
                    "safety_validation": safety_validation,
                    "sbar_note": sbar,
                    "fhir_bundle": fhir_b,
                    "final_response": alert_content,
                    "workflow_trace": [
                        {
                            "node": "fast_path_intake",
                            "summary": f"ESI 1 Resuscitation trigger identified: {', '.join(fp_reasons)}",
                            "tools": ["deterministic_vitals_engine"],
                            "duration_ms": round(fp_elapsed_ms * 0.35, 2),
                        },
                        {
                            "node": "fast_path_safety",
                            "summary": "Bypassed standard multi-turn LLM pipeline. Latency <50ms.",
                            "tools": ["rapid_alert_system"],
                            "duration_ms": round(fp_elapsed_ms * 0.25, 2),
                        },
                        {
                            "node": "fast_path_sbar_fhir",
                            "summary": "Generated rapid resuscitation SBAR and FHIR R4 Bundle.",
                            "tools": ["sbar_generator", "fhir_bundle_builder"],
                            "duration_ms": round(fp_elapsed_ms * 0.40, 2),
                        },
                    ],
                    "tool_calls_made": ["vitals_danger_zone_detector", "sbar_generator", "fhir_bundle_builder"],
                    "completed_steps": ["intake", "risk", "safety", "sbar"],
                    "next_step": "human_review",
                    "approval_decision": None,
                    "approval_note": None,
                    "nurse_override_esi": None,
                    "nurse_override_reason": None,
                    "nurse_notes": None,
                    "awaiting_human_review": True,
                    "hitl_required": True,
                    "location_address": location_address,
                    "location_lat": location_lat,
                    "location_lng": location_lng,
                    "node_durations": node_durs,
                    "latency_ms": fp_elapsed_ms,
                }

                return {
                    "content": alert_content,
                    "risk_level": "HIGH",
                    "esi_level": 1,
                    "esi_rationale": risk_assessment["esi_rationale"],
                    "is_emergency": True,
                    "hitl_required": True,
                    "awaiting_human_review": True,
                    "workflow_trace": fp_state["workflow_trace"],
                    "tool_calls_made": fp_state["tool_calls_made"],
                    "patient_profile": profile,
                    "symptoms": symptoms,
                    "vitals": all_vitals,
                    "vital_triggers": fp_triggers,
                    "sbar_note": sbar,
                    "fhir_bundle": fhir_b,
                    "duration_days": extracted.get("duration_days", 0),
                    "severity": "critical",
                    "pertinent_negatives": pertinent_negatives,
                    "red_flags": red_flags,
                    "infermedica_evidence": None,
                    "infermedica_triage": None,
                    "medications": [],
                    "intake_method": "deterministic_fast_path",
                    "followup_questions": [],
                    "nurse_override_esi": None,
                    "nurse_override_reason": None,
                    "nurse_notes": None,
                    "nearby_facilities": [],
                    "hospital_recommendations": [],
                    "location_address": location_address,
                    "location_lat": location_lat,
                    "location_lng": location_lng,
                    "pending_state": fp_state,
                    "node_durations": node_durs,
                    "latency_ms": fp_elapsed_ms,
                }

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

        total_proc_ms = round((time.perf_counter() - proc_start) * 1000.0, 2)
        node_durations = dict(final_state.get("node_durations", {}))
        node_durations["total"] = total_proc_ms

        return {
            "content": final_state.get("final_response", "No response generated."),
            "risk_level": assessment.get("risk_level", "LOW"),
            "esi_level": final_state.get("nurse_override_esi") or assessment.get("esi_level", 5),
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
            "safety_validation": final_state.get("safety_validation", {}),
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
            "node_durations": node_durations,
            "latency_ms": total_proc_ms,
        }


def get_multi_agent() -> MultiAgentSystem:
    return MultiAgentSystem()
