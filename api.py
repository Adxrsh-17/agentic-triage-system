"""Production FastAPI Serving Layer for Agentic Clinical Triage Copilot.

Provides RESTful HTTP APIs for:
- POST /v1/triage: Asynchronous/Synchronous clinical intake, consensus assessment, and SBAR/FHIR generation.
- POST /v1/triage/review: Human-in-the-Loop (HITL) nurse approval and acuity override endpoint.
- GET /v1/audit/logs: Encounters audit log retrieval with pagination.
- GET /v1/audit/analytics: Real-time concordance and override statistics.
- GET /health: Liveness and readiness probe for orchestration (Kubernetes, ECS).
"""

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.audit_logger import get_audit_logger
from agent.multi_agent import MultiAgentSystem, is_infermedica_configured

# ── Application Factory ──────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Clinical Triage Copilot API",
    version="1.0.0",
    description=(
        "Production-grade Clinical Triage Microservice powered by LangGraph multi-agent orchestration, "
        "ESI v4 consensus triage, vital-sign danger-zone safeguards, HL7 FHIR R4 Bundle generation, "
        "and sub-millisecond ESI 1 resuscitation fast-pathing."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global multi-agent singleton
_multi_agent: Optional[MultiAgentSystem] = None


def get_agent() -> MultiAgentSystem:
    global _multi_agent
    if _multi_agent is None:
        _multi_agent = MultiAgentSystem()
    return _multi_agent


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    chief_complaint: str = Field(
        ...,
        min_length=1,
        description="Patient primary clinical narrative, symptoms, or emergency dispatcher notes.",
        examples=["45yo female with severe substernal chest pressure radiating to jaw, diaphoresis, onset 30 mins ago."],
    )
    user_id: Optional[str] = Field("default_patient", description="Unique patient encounter identifier.")
    age: Optional[int] = Field(30, ge=0, le=125, description="Patient age in years.")
    sex: Optional[str] = Field("unknown", description="Biological sex (male, female, other, unknown).")
    vitals: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Objective vital signs: hr (bpm), sbp (mmHg), dbp (mmHg), spo2 (%), rr (/min), temp (C).",
        examples=[{"hr": 115, "sbp": 90, "dbp": 60, "spo2": 93, "rr": 24, "temp": 37.8}],
    )
    conditions: Optional[List[str]] = Field(
        default_factory=list,
        description="Known comorbidities or past medical history.",
        examples=[["CAD", "Hypertension", "Type 2 Diabetes"]],
    )
    history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list,
        description="Optional conversation history for multi-turn clinical intake.",
    )


class TriageResponse(BaseModel):
    user_id: str
    esi_level: int = Field(..., ge=1, le=5, description="Emergency Severity Index Acuity Level (1=Resuscitation to 5=Non-urgent).")
    risk_level: str = Field(..., description="Categorical clinical risk (LOW, MEDIUM, HIGH).")
    esi_rationale: str = Field(..., description="Clinical justification for assigned ESI acuity level.")
    is_emergency: bool = Field(..., description="Whether emergent immediate intervention is required.")
    hitl_required: bool = Field(..., description="Whether human nurse review is required prior to department placement.")
    awaiting_human_review: bool = Field(..., description="Whether execution state is suspended awaiting nurse sign-off.")
    vital_triggers: List[str] = Field(default_factory=list, description="Vital signs exceeding critical safety danger zones.")
    risk_reasons: List[str] = Field(default_factory=list, description="Clinical risk escalation reasons.")
    symptoms: List[str] = Field(default_factory=list, description="Identified clinical symptoms.")
    red_flags: List[str] = Field(default_factory=list, description="Critical red-flag symptoms detected.")
    predicted_resources: List[str] = Field(default_factory=list, description="Anticipated hospital resource consumption categories.")
    resource_count: int = Field(0, description="Count of distinct clinical resource categories predicted.")
    sbar_note: Optional[str] = Field(None, description="Structured SBAR clinical handoff note.")
    fhir_bundle: Optional[Dict[str, Any]] = Field(None, description="HL7 FHIR R4 Transaction Bundle.")
    pending_state: Optional[Dict[str, Any]] = Field(None, description="Suspended pipeline state payload for HITL approval flow.")
    latency_ms: float = Field(..., description="End-to-end processing latency in milliseconds.")


class ReviewRequest(BaseModel):
    user_id: str = Field(..., description="Patient encounter identifier.")
    pending_state: Dict[str, Any] = Field(..., description="Suspended triage state awaiting review.")
    approval_decision: str = Field(
        ...,
        description="'approved' to confirm AI recommendation, 'overridden' for nurse clinical override.",
    )
    approval_note: Optional[str] = Field(None, description="Clinical commentary or notes.")
    nurse_override_esi: Optional[int] = Field(None, ge=1, le=5, description="Overriding ESI level (1-5).")
    nurse_override_reason: Optional[str] = Field(None, description="Clinical justification for overriding AI recommendation.")


class PaginatedAuditLogs(BaseModel):
    total: int
    limit: int
    offset: int
    encounters: List[Dict[str, Any]]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post(
    "/v1/triage",
    response_model=TriageResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Clinical Triage Assessment",
    description="Processes clinical presentation through LangGraph multi-agent consensus or sub-millisecond ESI 1 fast-path.",
)
def execute_triage(request: TriageRequest) -> TriageResponse:
    agent = get_agent()
    t0 = time.perf_counter()

    result = agent.process(
        user_input=request.chief_complaint,
        user_id=request.user_id or "default_patient",
        age=request.age if request.age is not None else 30,
        sex=request.sex or "unknown",
        vitals=request.vitals or {},
        conditions=request.conditions or [],
        history=request.history or [],
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # For encounters finalized without requiring human review (low-risk / fast-path), log encounter
    if not result.get("awaiting_human_review"):
        audit = get_audit_logger()
        audit.log_encounter(
            patient_id=request.user_id or "default_patient",
            chief_complaint=request.chief_complaint,
            symptoms=result.get("symptoms", []),
            vitals=request.vitals or {},
            ai_suggested_esi=result.get("esi_level", 5),
            ai_rationale=result.get("esi_rationale", ""),
            ai_risk_level=result.get("risk_level", "LOW"),
            nurse_decision="direct_placement",
            nurse_final_esi=result.get("esi_level", 5),
            vital_triggers=result.get("vital_triggers", []),
            intake_method=result.get("intake_method", "regex_fallback"),
            fhir_bundle_id=result.get("fhir_bundle", {}).get("id") if result.get("fhir_bundle") else None,
        )

    return TriageResponse(
        user_id=result.get("user_id", request.user_id or "default_patient"),
        esi_level=int(result.get("esi_level", 5)),
        risk_level=result.get("risk_level", "LOW"),
        esi_rationale=result.get("esi_rationale", "Routine triage assessment"),
        is_emergency=bool(result.get("is_emergency", False)),
        hitl_required=bool(result.get("hitl_required", False)),
        awaiting_human_review=bool(result.get("awaiting_human_review", False)),
        vital_triggers=result.get("vital_triggers") or [],
        risk_reasons=result.get("risk_reasons") or [],
        symptoms=result.get("symptoms") or [],
        red_flags=result.get("red_flags") or [],
        predicted_resources=result.get("predicted_resources") or [],
        resource_count=int(result.get("resource_count", 0)),
        sbar_note=result.get("sbar_note"),
        fhir_bundle=result.get("fhir_bundle"),
        pending_state=result.get("pending_state"),
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/v1/triage/review",
    response_model=TriageResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Nurse HITL Review or Acuity Override",
    description="Resumes suspended triage execution after nurse sign-off or registers a calibrated clinical override.",
)
def review_triage(request: ReviewRequest) -> TriageResponse:
    if request.approval_decision == "overridden" and request.nurse_override_esi is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="nurse_override_esi is required when approval_decision is 'overridden'",
        )

    agent = get_agent()
    t0 = time.perf_counter()

    result = agent.process(
        user_id=request.user_id,
        pending_state=request.pending_state,
        approval_decision=request.approval_decision,
        approval_note=request.approval_note,
        nurse_override_esi=request.nurse_override_esi,
        nurse_override_reason=request.nurse_override_reason,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return TriageResponse(
        user_id=result.get("user_id", request.user_id),
        esi_level=int(result.get("esi_level", request.nurse_override_esi or 5)),
        risk_level=result.get("risk_level", "LOW"),
        esi_rationale=result.get("esi_rationale", "Nurse clinical review completed"),
        is_emergency=bool(result.get("is_emergency", False)),
        hitl_required=bool(result.get("hitl_required", False)),
        awaiting_human_review=bool(result.get("awaiting_human_review", False)),
        vital_triggers=result.get("vital_triggers") or [],
        risk_reasons=result.get("risk_reasons") or [],
        symptoms=result.get("symptoms") or [],
        red_flags=result.get("red_flags") or [],
        predicted_resources=result.get("predicted_resources") or [],
        resource_count=int(result.get("resource_count", 0)),
        sbar_note=result.get("sbar_note"),
        fhir_bundle=result.get("fhir_bundle"),
        pending_state=result.get("pending_state"),
        latency_ms=round(latency_ms, 2),
    )


@app.get(
    "/v1/audit/logs",
    response_model=PaginatedAuditLogs,
    status_code=status.HTTP_200_OK,
    summary="Query Encounter Audit Logs",
    description="Retrieves finalized triage encounter logs in reverse chronological order with pagination.",
)
def get_audit_logs(
    limit: int = Query(default=20, ge=1, le=100, description="Max encounters to return"),
    offset: int = Query(default=0, ge=0, description="Encounter offset"),
) -> PaginatedAuditLogs:
    audit = get_audit_logger()
    all_encounters = audit.get_encounters()
    total = len(all_encounters)
    paginated = all_encounters[offset : offset + limit]

    return PaginatedAuditLogs(
        total=total,
        limit=limit,
        offset=offset,
        encounters=paginated,
    )


@app.get(
    "/v1/audit/analytics",
    status_code=status.HTTP_200_OK,
    summary="Get Concordance and Acuity Analytics",
    description="Computes concordance rates, override frequencies, and ESI distribution across all encounters.",
)
def get_audit_analytics() -> Dict[str, Any]:
    audit = get_audit_logger()
    return audit.get_analytics()


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Service Health Check & Readiness Probe",
    description="Reports system operational status, subsystem connectivity, and component readiness.",
)
def health_check() -> Dict[str, Any]:
    groq_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
    pinecone_key = bool(os.environ.get("PINECONE_API_KEY", "").strip())
    infermedica_active = is_infermedica_configured()
    audit = get_audit_logger()
    encounters_count = len(audit.get_encounters())

    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "groq_llm": {
                "configured": groq_key,
                "status": "active" if groq_key else "offline_regex_fallback",
            },
            "pinecone_memory": {
                "configured": pinecone_key,
                "status": "active" if pinecone_key else "offline_in_memory",
            },
            "infermedica": {
                "configured": infermedica_active,
                "status": "active" if infermedica_active else "offline_deterministic_fallback",
            },
            "audit_logger": {
                "status": "active",
                "total_records": encounters_count,
            },
        },
    }
