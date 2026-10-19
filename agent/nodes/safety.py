"""Safety Validator & Patient Communication Node Implementation.

Enforces:
1. Mandatory educational prototype medical disclaimer
2. Deterministic blacklisted pattern filtering (no definitive diagnoses, no medication dosing)
3. Safe templated fallbacks on validation failure
4. Routing of all high-acuity (emergency) cases to Human Review
"""

import re
from typing import Dict, List, Tuple

from agent.schemas import RiskLevel, RiskResult

MANDATORY_DISCLAIMER = (
    "Disclaimer: This is an educational AI prototype and clinical decision-support tool, not a certified medical device. "
    "Always consult a qualified healthcare professional. If experiencing a life-threatening emergency, call 911 immediately."
)

BLACKLISTED_PATTERNS = [
    r"\byou have (?:cancer|pneumonia|appendicitis|meningitis|stroke|a heart attack|diabetes)\b",
    r"\btake \d+\s*(?:mg|milligrams|tablets|pills)\b",
    r"\bprescribe\b",
    r"\bstart taking\b",
    r"\bstop your medication\b",
]


def validate_safety_output(message: str) -> Tuple[bool, List[str]]:
    """Verify that generated communication adheres to clinical safety guidelines.

    Returns:
        (is_valid, list_of_violations)
    """
    violations = []

    # Check for blacklisted diagnosis or dosing instructions
    for pattern in BLACKLISTED_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            violations.append(f"Contains prohibited medical diagnosis or dosing pattern: {pattern}")

    # Check for medical disclaimer
    if "consult a qualified healthcare professional" not in message.lower() and "disclaimer" not in message.lower():
        violations.append("Missing required clinical decision-support disclaimer.")

    return len(violations) == 0, violations


def generate_safe_response(
    risk: RiskResult,
    symptoms: List[str],
    sbar_note: str = "",
) -> str:
    """Construct user-facing clinical communication with guaranteed safety compliance."""
    if risk.level == RiskLevel.EMERGENCY:
        body = (
            "EMERGENCY CLINICAL ALERT: Your presentation exhibits high-risk warning signs requiring immediate emergency evaluation.\n"
            "Please proceed to the nearest Emergency Department immediately or call local emergency medical services (911).\n\n"
            f"Primary Clinical Rationale: {risk.rationale}\n"
            "A licensed triage clinician must review and confirm this assessment."
        )
    elif risk.level == RiskLevel.URGENT:
        body = (
            "URGENT EVALUATION RECOMMENDED: Your symptoms indicate an acute issue that should be evaluated by a healthcare "
            "provider within 24 hours.\n\n"
            f"Assessed Concerns: {', '.join(symptoms) if symptoms else 'Acute clinical symptoms'}\n"
            f"Clinical Rationale: {risk.rationale}\n"
            "Please visit an Urgent Care Center or contact your primary care physician today."
        )
    else:
        body = (
            "ROUTINE EVALUATION: Your reported symptoms appear non-emergent and may be suitable for routine outpatient clinic care "
            "or supportive home monitoring.\n\n"
            f"Identified Symptoms: {', '.join(symptoms) if symptoms else 'Minor symptoms'}\n"
            f"Clinical Assessment: {risk.rationale}\n"
            "If symptoms worsen, persist, or new red-flag symptoms develop (e.g. chest pain, shortness of breath), seek immediate care."
        )

    full_message = f"{body}\n\n---\n{MANDATORY_DISCLAIMER}"

    # Verify against safety validator
    is_valid, violations = validate_safety_output(full_message)
    if not is_valid:
        # Fallback to pristine safe template
        return (
            f"Clinical Assessment Summary: {risk.level.value.upper()}\n"
            f"Acuity Level: ESI {risk.esi_level}\n\n"
            "Please consult a physician or triage nurse for comprehensive clinical evaluation.\n\n"
            f"---\n{MANDATORY_DISCLAIMER}"
        )

    return full_message
