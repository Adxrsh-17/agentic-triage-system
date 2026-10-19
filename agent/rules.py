"""Deterministic Clinical Safety Floor & Emergency Red-Flag Rule Engine.

Implements:
- Compiled emergency red-flag patterns
- Windowed NegEx scope resolution with conjunction and clause boundary parsing
- ESI v4 age-stratified pediatric physiological bounds
- rule_risk: Deterministic baseline risk scorer enforcing the safety floor
- detect_red_flags: Red flag extractor with negation filtering
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from agent.schemas import IntakeResult, RiskLevel, RiskResult


# ── Red-Flag Knowledge Base ──────────────────────────────────────────────────

RED_FLAG_PATTERNS: List[str] = [
    "crushing chest pain",
    "chest pain radiating",
    "radiating to left arm",
    "radiating to jaw",
    "thunderclap headache",
    "worst headache of life",
    "facial droop",
    "face drooping",
    "slurred speech",
    "arm weakness",
    "one-sided weakness",
    "hemiparesis",
    "cyanosis",
    "blue lips",
    "stridor",
    "inspiratory stridor",
    "hemoptysis",
    "coughing blood",
    "hematemesis",
    "vomiting blood",
    "anaphylaxis",
    "loss of consciousness",
    "unconscious",
    "unresponsive",
    "comatose",
    "coma",
    "antifreeze",
    "ethylene glycol",
    "overdose",
    "suicide",
    "suicidal",
    "toxic ingestion",
    "vision loss",
    "loss of vision",
    "crepitus",
    "fournier",
    "necrotizing",
    "kehr",
    "lipase",
    "testicular torsion",
    "ectopic pregnancy",
    "arterial bleed",
    "exsanguinating",
    "tension pneumothorax",
    "tracheal deviation",
    "decerebrate",
    "decorticate",
]

RESUSCITATION_PATTERNS: List[str] = [
    "loss of consciousness",
    "unconscious",
    "unresponsive",
    "comatose",
    "coma",
    "cardiac arrest",
    "respiratory arrest",
    "apnea",
    "pulseless",
    "agonal",
    "anaphylactic shock",
    "severe hypoglycemia",
    "glucometer reading low",
    "glucometer reading 'low'",
    "exsanguinating",
    "arterial bleed",
    "uncontrolled bleeding",
    "tension pneumothorax",
    "tracheal deviation",
]

URGENT_PATTERNS: List[str] = [
    "fever",
    "vomiting",
    "abdominal pain",
    "asthma",
    "wheezing",
    "dysuria",
    "burning urination",
    "headache",
    "migraine",
    "fracture",
    "deformity",
    "laceration",
    "cut requiring stitches",
    "deep cut",
    "cellulitis",
    "abscess",
    "flank pain",
]


# ── NegEx Scope Resolution ───────────────────────────────────────────────────

def is_negated(keyword: str, text: str, pertinent_negatives: Optional[List[str]] = None) -> bool:
    """Evaluate whether a clinical symptom or red flag is explicitly negated in text.

    Uses windowed NegEx matching across coordinated conjunctions ('or', 'and', commas)
    while halting scope propagation at adversative boundaries ('but', 'however', 'except', 'yet').
    """
    if pertinent_negatives:
        for neg in pertinent_negatives:
            if keyword.lower() in neg.lower():
                return True

    pattern = (
        rf"(?:denies|deny|denied|no|not|without|negative for|rules? out|not experiencing|zero)\s+"
        rf"(?:[a-zA-Z0-9_,-]+\s+){{0,5}}{re.escape(keyword)}"
    )
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        matched_str = match.group(0).lower()
        # Halt negation if adversative conjunction separates negation word from keyword
        if not any(b in matched_str for b in [" but ", " however ", " except ", " although ", " yet "]):
            return True
    return False


def detect_red_flags(text: str, pertinent_negatives: Optional[List[str]] = None) -> List[str]:
    """Extract affirmative red flag keywords, filtering out any negated mentions."""
    lowered = text.lower()
    flags = []
    for rf in RED_FLAG_PATTERNS:
        if rf in lowered:
            if not is_negated(rf, lowered, pertinent_negatives):
                flags.append(rf)
    return flags


def check_red_flags(text: str, *args, **kwargs) -> List[str]:
    """Compatibility wrapper around detect_red_flags."""
    pertinent_negatives = kwargs.get("pertinent_negatives")
    if pertinent_negatives is None:
        if len(args) == 1:
            pertinent_negatives = args[0]
        elif len(args) >= 2:
            pertinent_negatives = args[1]
    return detect_red_flags(text, pertinent_negatives)


# ── Physiological Vital Signs Danger Zones ───────────────────────────────────

def check_vital_danger_zones(
    vitals: Dict[str, Any],
    age: Optional[int] = None,
) -> Tuple[bool, List[str], int]:
    """Evaluate objective vital signs against ESI v4 physiological bounds.

    Returns:
        (is_danger, trigger_descriptions, max_acuity_esi)
    """
    if not vitals:
        return False, [], 5

    triggers: List[str] = []
    is_danger = False
    min_esi = 5

    # Safe float conversion
    def _get_val(k: str) -> Optional[float]:
        try:
            v = vitals.get(k)
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    spo2 = _get_val("spo2")
    hr = _get_val("hr")
    sbp = _get_val("sbp")
    rr = _get_val("rr")

    # Critical Hypoxia (ESI 1)
    if spo2 is not None:
        if spo2 < 90:
            is_danger = True
            min_esi = min(min_esi, 1)
            triggers.append(f"Critical Hypoxia (SpO2 {spo2}% < 90%)")
        elif spo2 <= 92:
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Severe Hypoxia (SpO2 {spo2}%)")

    # Profound Bradycardia / Shock (ESI 1)
    if hr is not None and hr < 35:
        is_danger = True
        min_esi = min(min_esi, 1)
        triggers.append(f"Profound Bradycardia (HR {hr} bpm < 35)")

    if sbp is not None and 0 < sbp < 70:
        is_danger = True
        min_esi = min(min_esi, 1)
        triggers.append(f"Decompensated Shock (SBP {sbp} mmHg < 70)")

    # Age-stratified tachycardia / tachypnea / hypotension (ESI 2)
    pt_age = age if age is not None else 30
    if pt_age < 1:  # Infant
        if hr is not None and (hr > 180 or hr < 80):
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Infant Extreme Heart Rate (HR {hr} bpm)")
        if rr is not None and rr > 50:
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Infant Severe Tachypnea (RR {rr} /min)")
        if sbp is not None and 0 < sbp < 65:
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Infant Hypotension (SBP {sbp} mmHg)")
    elif pt_age < 3:  # Toddler
        if hr is not None and (hr > 160 or hr < 70):
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Toddler Danger Heart Rate (HR {hr} bpm)")
        if rr is not None and rr > 40:
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Toddler Tachypnea (RR {rr} /min)")
    elif pt_age < 8:  # Child
        if hr is not None and (hr > 140 or hr < 60):
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Pediatric Danger Heart Rate (HR {hr} bpm)")
        if rr is not None and rr > 30:
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Pediatric Tachypnea (RR {rr} /min)")
    else:  # Adult (>8 yr)
        if hr is not None and hr > 140:
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Severe Tachycardia (HR {hr} bpm > 140)")
        if hr is not None and hr < 40 and hr >= 35:
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Severe Bradycardia (HR {hr} bpm < 40)")
        if rr is not None and (rr > 30 or rr < 8):
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Severe Tachypnea/Bradypnea (RR {rr} /min)")
        if sbp is not None and 70 <= sbp < 90:
            is_danger = True
            min_esi = min(min_esi, 2)
            triggers.append(f"Adult Hypotension (SBP {sbp} mmHg < 90)")

    return is_danger, triggers, min_esi


# ── Deterministic Rule Risk Evaluation ───────────────────────────────────────

def rule_risk(
    text: str,
    intake: Optional[IntakeResult] = None,
    vitals: Optional[Dict[str, Any]] = None,
    age: Optional[int] = None,
) -> RiskResult:
    """Compute the deterministic safety-floor risk assessment.

    This represents the inviolable lower bound on acuity: the hybrid system
    may escalate acuity above this result, but must NEVER de-escalate below it.
    """
    lowered = text.lower()
    pertinent_negs = intake.pertinent_negatives if intake else []
    pt_age = age if age is not None else (intake.age if intake and intake.age else 30)

    # 1. Resuscitation (ESI 1) checks
    for res_kw in RESUSCITATION_PATTERNS:
        if res_kw in lowered and not is_negated(res_kw, lowered, pertinent_negs):
            return RiskResult(
                level=RiskLevel.EMERGENCY,
                esi_level=1,
                score=1.0,
                red_flags=[res_kw],
                vital_triggers=[f"Resuscitation Trigger: {res_kw}"],
                rationale=f"Critical life-threatening condition detected: {res_kw}",
                source="rules",
                hitl_required=True,
            )

    # 2. Vital danger zone checks
    vital_danger, vital_triggers, min_esi = check_vital_danger_zones(vitals or {}, age=pt_age)
    if min_esi == 1:
        return RiskResult(
            level=RiskLevel.EMERGENCY,
            esi_level=1,
            score=1.0,
            red_flags=[],
            vital_triggers=vital_triggers,
            rationale="Critical physiological vital sign collapse requiring immediate resuscitation",
            source="rules",
            hitl_required=True,
        )

    # 3. Emergency red flags (ESI 2)
    active_flags = detect_red_flags(text, pertinent_negs)
    if active_flags or min_esi == 2:
        reasons = [f"Red flag: {f}" for f in active_flags] + vital_triggers
        return RiskResult(
            level=RiskLevel.EMERGENCY,
            esi_level=2,
            score=0.9,
            red_flags=active_flags,
            vital_triggers=vital_triggers,
            rationale=f"Emergency red-flag signs detected: {', '.join(reasons) if reasons else 'High-risk vital signs'}",
            source="rules",
            hitl_required=True,
        )

    # 4. Urgent symptoms (ESI 3)
    urgent_matches = [u for u in URGENT_PATTERNS if u in lowered and not is_negated(u, lowered, pertinent_negs)]
    if urgent_matches or (intake and intake.severity_hint == "moderate"):
        return RiskResult(
            level=RiskLevel.URGENT,
            esi_level=3,
            score=0.5,
            red_flags=[],
            vital_triggers=[],
            rationale=f"Urgent medical presentation requiring timely evaluation: {', '.join(urgent_matches) if urgent_matches else 'Moderate symptoms'}",
            source="rules",
            hitl_required=False,
        )

    # 5. Routine (ESI 4-5)
    return RiskResult(
        level=RiskLevel.ROUTINE,
        esi_level=5,
        score=0.1,
        red_flags=[],
        vital_triggers=[],
        rationale="Routine minor symptoms appropriate for clinic or self-care",
        source="rules",
        hitl_required=False,
    )
