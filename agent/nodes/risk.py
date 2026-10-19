"""Hybrid Clinical Risk & Consensus Node Implementation.

Design Principle: LLM for semantic understanding, deterministic rules as the inviolable safety floor.

Safety Invariant:
The final clinical urgency may escalate above the rule floor based on LLM contextual reasoning,
but must NEVER de-escalate below what the deterministic red-flag check establishes.
"""

from typing import Any, Dict, List, Optional

from agent.llm import call_structured_llm, is_llm_available
from agent.rules import rule_risk
from agent.schemas import (
    IntakeResult,
    RiskLevel,
    RiskResult,
    max_severity,
    risk_level_to_default_esi,
    severity_order,
)


def _llm_assess_risk(
    intake: IntakeResult,
    vitals: Optional[Dict[str, Any]] = None,
    memory_context: Optional[str] = None,
) -> Optional[RiskResult]:
    """Execute LLM risk evaluation via ChatGroq structured output."""
    if not is_llm_available():
        return None

    prompt = (
        f"Patient Age: {intake.age}, Sex: {intake.sex}\n"
        f"Reported Symptoms: {', '.join(intake.symptoms)}\n"
        f"Pertinent Negatives (Denied): {', '.join(intake.pertinent_negatives)}\n"
        f"Comorbidities: {', '.join(intake.conditions)}\n"
        f"Objective Vitals: {vitals or {}}\n"
    )
    if memory_context:
        prompt += f"\nRelevant Past Medical Memory:\n{memory_context}\n"

    system_prompt = (
        "You are an expert emergency triage physician. Classify the patient's presentation into one of three "
        "acuity levels: 'emergency' (life threat / emergent need), 'urgent' (requires evaluation within 24h), "
        "or 'routine' (self-care / clinic). Provide clinical rationale and identify any red flags."
    )

    llm_out = call_structured_llm(
        schema=RiskResult,
        user_prompt=prompt,
        system_prompt=system_prompt,
        timeout=5.0,
    )
    if llm_out is not None:
        llm_out.source = "llm"
    return llm_out


def run_hybrid_risk(
    chief_complaint: str,
    intake: IntakeResult,
    vitals: Optional[Dict[str, Any]] = None,
    memory_context: Optional[str] = None,
) -> RiskResult:
    """Execute hybrid risk assessment with deterministic safety-floor enforcement.

    1. Always compute rule_result as the baseline safety floor.
    2. Try LLM assessment when configured.
    3. Final level = max_severity(llm_level, rule_level).
    4. Log disagreement flag if LLM and rules diverge.
    """
    # 1. Deterministic safety floor (always executed)
    rule_result = rule_risk(
        text=chief_complaint,
        intake=intake,
        vitals=vitals,
        age=intake.age,
    )

    # 2. LLM assessment (optional / best-effort)
    llm_result = _llm_assess_risk(intake=intake, vitals=vitals, memory_context=memory_context)

    if llm_result is None:
        # Rules-only fallback mode
        return rule_result

    # 3. Hybrid consensus: enforce safety invariant
    final_level = max_severity(llm_result.level, rule_result.level)
    disagreement = (llm_result.level != rule_result.level)

    # If rules detected an emergency and LLM returned routine/urgent, safety floor overrides
    if severity_order(rule_result.level) > severity_order(llm_result.level):
        rationale = f"[Safety Floor Override] {rule_result.rationale} (LLM suggested {llm_result.level.value})"
    else:
        rationale = llm_result.rationale

    # Union of red flags and vital triggers
    combined_flags = list(set(llm_result.red_flags + rule_result.red_flags))
    combined_triggers = list(set(llm_result.vital_triggers + rule_result.vital_triggers))

    # Determine ESI level (respecting rule ESI floor)
    esi_candidate = rule_result.esi_level
    if final_level == RiskLevel.EMERGENCY:
        esi_candidate = min(esi_candidate, 2)
    elif final_level == RiskLevel.URGENT:
        esi_candidate = min(esi_candidate, 3)

    return RiskResult(
        level=final_level,
        esi_level=esi_candidate,
        score=max(llm_result.score, rule_result.score),
        red_flags=combined_flags,
        vital_triggers=combined_triggers,
        rationale=rationale,
        source="hybrid",
        disagreement=disagreement,
        rule_level=rule_result.level,
        llm_level=llm_result.level,
        hitl_required=(final_level == RiskLevel.EMERGENCY or esi_candidate <= 2),
    )
