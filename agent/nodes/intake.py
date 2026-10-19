"""Clinical Intake Node Implementation (LLM with Deterministic Regex Fallback).

Extracts structured IntakeResult from narrative chief complaint text:
1. Attempts LLM structured extraction via ChatGroq (when API key is configured)
2. Falls back seamlessly to offline regex and keyword tokenization on any LLM failure
3. Populates symptoms, duration, pertinent negatives, and source provenance
"""

import re
from typing import Any, Dict, List, Optional

from agent.llm import call_structured_llm, is_llm_available
from agent.rules import is_negated
from agent.schemas import IntakeResult


# Core symptom pattern mappings for deterministic fallback
DETERMINISTIC_SYMPTOM_PATTERNS = {
    "chest pain": ["chest pain", "chest tightness", "chest pressure", "chest discomfort"],
    "shortness of breath": ["shortness of breath", "difficulty breathing", "breathless", "can't breathe"],
    "headache": ["headache", "head pain", "migraine"],
    "fever": ["fever", "high temperature", "chills", "feverish"],
    "cough": ["cough", "coughing", "phlegm"],
    "nausea": ["nausea", "nauseous", "queasy"],
    "vomiting": ["vomiting", "throwing up", "vomit"],
    "diarrhea": ["diarrhea", "loose stools", "watery stool"],
    "fatigue": ["fatigue", "tired", "exhausted", "weakness"],
    "dizziness": ["dizziness", "dizzy", "lightheaded"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly pain"],
    "sore throat": ["sore throat", "throat pain"],
    "dysuria": ["burning with urination", "painful urination", "urinary frequency", "dysuria"],
    "extremity injury": ["twisted ankle", "ankle pain", "wrist pain", "finger injury", "sprained"],
    "laceration": ["laceration", "skin cut", "cut on finger", "needing stitches", "gash"],
}


def regex_fallback_intake(text: str, age: Optional[int] = None, sex: Optional[str] = None) -> IntakeResult:
    """Deterministic regex extraction fallback producing validated IntakeResult schema."""
    lowered = text.lower()
    found: List[str] = []
    pertinent_negatives: List[str] = []

    for name, variants in DETERMINISTIC_SYMPTOM_PATTERNS.items():
        matched = False
        for v in variants:
            if v in lowered:
                if is_negated(v, lowered, pertinent_negatives):
                    pertinent_negatives.append(f"denies {name}")
                else:
                    found.append(name)
                matched = True
                break

    # Duration parsing
    duration_days = 0.0
    duration_str = None
    for pattern, mult in [
        (r"(\d+)\s*months?", 30.0),
        (r"(\d+)\s*weeks?", 7.0),
        (r"(\d+)\s*days?", 1.0),
        (r"(\d+)\s*hours?", 0.04),
    ]:
        m = re.search(pattern, lowered)
        if m:
            duration_days = float(m.group(1)) * mult
            duration_str = m.group(0)
            break

    # Severity hint
    severity_hint = "mild"
    if any(w in lowered for w in ["severe", "extreme", "unbearable", "worst", "excruciating", "10/10"]):
        severity_hint = "severe"
    elif any(w in lowered for w in ["moderate", "significant", "bad"]):
        severity_hint = "moderate"

    return IntakeResult(
        symptoms=found if found else ["unspecified symptom"],
        duration=duration_str,
        duration_days=duration_days,
        severity_hint=severity_hint,
        age=age,
        sex=sex or "unknown",
        pertinent_negatives=pertinent_negatives,
        source="regex",
    )


def run_intake(
    chief_complaint: str,
    age: Optional[int] = None,
    sex: Optional[str] = None,
    known_conditions: Optional[List[str]] = None,
) -> IntakeResult:
    """Execute hybrid intake extraction: LLM structured extraction with automatic regex fallback."""
    if is_llm_available():
        system_prompt = (
            "You are a clinical intake assistant. Extract clinical entities from the patient's presentation into the "
            "provided schema. Identify symptoms, duration, severity descriptor, and explicitly denied symptoms (pertinent negatives)."
        )
        llm_out = call_structured_llm(
            schema=IntakeResult,
            user_prompt=f"Patient age: {age}, sex: {sex}. Complaint: {chief_complaint}",
            system_prompt=system_prompt,
            timeout=4.0,
        )
        if llm_out is not None and llm_out.symptoms:
            llm_out.source = "llm"
            if age is not None:
                llm_out.age = age
            if sex:
                llm_out.sex = sex
            if known_conditions:
                llm_out.conditions = known_conditions
            return llm_out

    # Offline / rate-limited fallback
    result = regex_fallback_intake(chief_complaint, age=age, sex=sex)
    if known_conditions:
        result.conditions = known_conditions
    return result
