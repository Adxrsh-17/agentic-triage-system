"""Pydantic Schemas for Multi-Agent Clinical Triage Subsystem.

Defines standardized data models for:
- RiskLevel: Categorical triage urgency (routine, urgent, emergency)
- IntakeResult: Structured extraction of clinical narrative and patient profile
- RiskResult: Acuity score, red flags, clinical rationale, and provenance tracking
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Three-tier categorical clinical triage urgency."""
    ROUTINE = "routine"      # Self-care / GP clinic visit within days (ESI 4-5)
    URGENT = "urgent"        # Urgent clinic / ED within 24h (ESI 3)
    EMERGENCY = "emergency"  # Immediate emergency department / resuscitation care (ESI 1-2)


def severity_order(level: RiskLevel) -> int:
    """Ordinal rank for risk levels to enforce safety floors (higher integer = higher risk)."""
    mapping = {
        RiskLevel.ROUTINE: 1,
        RiskLevel.URGENT: 2,
        RiskLevel.EMERGENCY: 3,
    }
    return mapping.get(level, 1)


def max_severity(level_a: RiskLevel, level_b: RiskLevel) -> RiskLevel:
    """Return the higher acuity level between two evaluations (conservative safety invariant)."""
    return level_a if severity_order(level_a) >= severity_order(level_b) else level_b


def esi_to_risk_level(esi_level: int) -> RiskLevel:
    """Map standard ESI (1-5) to 3-tier RiskLevel."""
    if esi_level in [1, 2]:
        return RiskLevel.EMERGENCY
    elif esi_level == 3:
        return RiskLevel.URGENT
    else:
        return RiskLevel.ROUTINE


def risk_level_to_default_esi(level: RiskLevel) -> int:
    """Map 3-tier RiskLevel to a conservative default ESI level."""
    if level == RiskLevel.EMERGENCY:
        return 2
    elif level == RiskLevel.URGENT:
        return 3
    else:
        return 5


class IntakeResult(BaseModel):
    """Structured extraction of symptoms, profile, and temporal attributes."""
    symptoms: List[str] = Field(default_factory=list, description="Primary clinical symptoms identified")
    duration: Optional[str] = Field(None, description="Reported duration of symptoms")
    duration_days: float = Field(0.0, description="Normalized duration in days")
    severity_hint: Optional[str] = Field(None, description="Extracted severity descriptor (mild, moderate, severe)")
    age: Optional[int] = Field(None, ge=0, le=125, description="Patient age in years")
    sex: Optional[str] = Field("unknown", description="Biological sex")
    conditions: List[str] = Field(default_factory=list, description="Known chronic medical comorbidities")
    medications: List[str] = Field(default_factory=list, description="Current home medications")
    pertinent_negatives: List[str] = Field(default_factory=list, description="Symptoms explicitly denied by patient")
    missing_info: List[str] = Field(default_factory=list, description="Critical information missing for complete assessment")
    source: str = Field("regex", description="Intake extraction source: 'llm' | 'infermedica' | 'regex'")


class RiskResult(BaseModel):
    """Standardized output of the clinical risk and consensus engine."""
    level: RiskLevel = Field(..., description="Categorical clinical urgency level")
    esi_level: int = Field(5, ge=1, le=5, description="Associated Emergency Severity Index level (1-5)")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Continuous risk probability score [0.0, 1.0]")
    red_flags: List[str] = Field(default_factory=list, description="Red flag clinical keywords detected")
    vital_triggers: List[str] = Field(default_factory=list, description="Vital signs exceeding danger-zone safety bounds")
    rationale: str = Field(..., description="Explainable clinical justification for acuity level")
    source: str = Field("rules", description="Scoring source: 'llm' | 'rules' | 'hybrid' | 'fast_path'")
    disagreement: bool = Field(False, description="Flag indicating whether LLM and rule floor disagreed")
    rule_level: Optional[RiskLevel] = Field(None, description="Underlying deterministic rule assessment")
    llm_level: Optional[RiskLevel] = Field(None, description="Underlying LLM assessment if available")
    hitl_required: bool = Field(False, description="Whether Human-in-the-Loop clinician review is mandated")
