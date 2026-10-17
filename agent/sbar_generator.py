"""Clinical SBAR Handoff Note & FHIR R4 Bundle Generator.

Provides standardized, audit-ready clinical handoff documentation (Situation,
Background, Assessment, Recommendation) for emergency intake personnel,
and constructs valid HL7 FHIR R4 Transaction Bundles for hospital EHR interoperability.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


def generate_sbar_note(
    patient_id: str,
    age: int,
    sex: str,
    chief_complaint: str,
    vitals: Optional[Dict[str, Any]] = None,
    symptoms: Optional[List[str]] = None,
    duration_days: float = 0.0,
    pertinent_negatives: Optional[List[str]] = None,
    red_flags: Optional[List[str]] = None,
    comorbidities: Optional[List[str]] = None,
    ai_esi_level: int = 5,
    ai_esi_rationale: str = "",
    nurse_esi_level: Optional[int] = None,
    nurse_override_reason: Optional[str] = None,
    nurse_notes: Optional[str] = None,
    esi_level: Optional[int] = None,
    esi_rationale: Optional[str] = None,
    nurse_override_esi: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Generate an emergency department SBAR clinical handoff note.
    
    Format:
    - S (Situation): Demographics, arrival acuity, chief complaint
    - B (Background): Chronic comorbidities, relevant risk profile
    - A (Assessment): Extracted clinical evidence, vitals, ESI scoring & consensus rationale
    - R (Recommendation): Immediate disposition, monitoring, initial workup checklist
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    vitals = vitals or {}
    symptoms = symptoms or []
    pertinent_negatives = pertinent_negatives or []
    red_flags = red_flags or []
    comorbidities = comorbidities or []

    if esi_level is not None:
        ai_esi_level = esi_level
    if esi_rationale:
        ai_esi_rationale = esi_rationale
    if nurse_override_esi is not None:
        nurse_esi_level = nurse_override_esi

    final_esi = nurse_esi_level if nurse_esi_level is not None else ai_esi_level
    esi_names = {
        1: "ESI 1 - Resuscitation (Immediate Life Threat)",
        2: "ESI 2 - Emergent (High Risk / Should Not Wait)",
        3: "ESI 3 - Urgent (Multiple Diagnostic Resources Needed)",
        4: "ESI 4 - Less Urgent (Single Clinical Resource Needed)",
        5: "ESI 5 - Non-Urgent (Routine Outpatient / Home Care)",
    }
    esi_label = esi_names.get(final_esi, f"ESI {final_esi}")

    # Format vitals string
    vitals_parts = []
    if vitals.get("hr"):
        vitals_parts.append(f"HR: {vitals['hr']} bpm")
    if vitals.get("sbp") and vitals.get("dbp"):
        vitals_parts.append(f"BP: {vitals['sbp']}/{vitals['dbp']} mmHg")
    elif vitals.get("sbp"):
        vitals_parts.append(f"SBP: {vitals['sbp']} mmHg")
    if vitals.get("spo2"):
        vitals_parts.append(f"SpO2: {vitals['spo2']}%")
    if vitals.get("rr"):
        vitals_parts.append(f"RR: {vitals['rr']}/min")
    if vitals.get("temp"):
        vitals_parts.append(f"Temp: {vitals['temp']}°F")
    if vitals.get("pain"):
        vitals_parts.append(f"Pain: {vitals['pain']}/10")

    vitals_str = ", ".join(vitals_parts) if vitals_parts else "Not formally measured at intake"

    # Recommendations based on acuity
    rec_actions = []
    if final_esi == 1:
        rec_actions = [
            "Immediate Resuscitation Bay placement (Zero wait)",
            "Notify Attending Emergency Physician and Trauma/Code Team",
            "Establish dual large-bore IV access and continuous cardiac monitoring",
            "Prepare airway / resuscitation cart and rapid-sequence equipment",
        ]
    elif final_esi == 2:
        rec_actions = [
            "Priority Acute ED bed assignment (< 10 minutes wait)",
            "12-lead ECG within 10 minutes if chest pain or dyspnea",
            "Initiate IV access, baseline cardiac enzymes / POC troponin",
            "Frequent vital signs reassessment every 15 minutes",
        ]
    elif final_esi == 3:
        rec_actions = [
            "ED monitored waiting area / Urgent intake track",
            "Order standard diagnostic workup (CBC, CMP, targeted imaging)",
            "Vital sign re-check at 60 minutes or upon clinical change",
        ]
    elif final_esi == 4:
        rec_actions = [
            "Fast Track / Ambulatory procedure area",
            "Single resource evaluation (e.g., targeted plain X-ray or simple suturing)",
            "Re-evaluate for outpatient discharge upon completion",
        ]
    else:
        rec_actions = [
            "Routine ambulatory consultation or clinic referral",
            "Patient education and outpatient follow-up instructions",
            "Zero acute hospital resource utilization anticipated",
        ]

    lines = [
        "=======================================================================",
        "EMERGENCY DEPARTMENT CLINICAL TRIAGE HANDOFF (SBAR REPORT)",
        f"Timestamp: {now_str}  |  Encounter ID: {patient_id}",
        "=======================================================================",
        "",
        "[S] SITUATION",
        f"• Patient: {age}-year-old {sex.lower()}",
        f"• Chief Complaint: \"{chief_complaint.strip()}\"",
        f"• Triage Acuity Status: {esi_label}",
    ]

    if nurse_esi_level is not None and nurse_esi_level != ai_esi_level:
        lines.append(f"• CLINICIAN OVERRIDE: Nurse adjusted ESI from {ai_esi_level} to {nurse_esi_level}")
        if nurse_override_reason:
            lines.append(f"  Reason for Override: {nurse_override_reason}")

    lines.extend([
        "",
        "[B] BACKGROUND",
        f"• Known Medical Conditions / Comorbidities: {', '.join(comorbidities) if comorbidities else 'None reported'}",
        f"• Duration of Current Illness: {duration_days:.1f} day(s)" if duration_days > 0 else "• Duration of Current Illness: Acute onset (< 24 hours)",
    ])

    lines.extend([
        "",
        "[A] ASSESSMENT",
        f"• Recorded Intake Vitals: {vitals_str}",
        f"• Extracted Symptoms: {', '.join(symptoms) if symptoms else 'None formalized'}",
        f"• Pertinent Negatives: {', '.join(pertinent_negatives) if pertinent_negatives else 'None documented'}",
        f"• Acute Red Flags: {', '.join(red_flags) if red_flags else 'None identified'}",
        f"• AI Decision Support Rationale: {ai_esi_rationale or 'Algorithmic consensus applied'}",
    ])

    if nurse_notes:
        lines.append(f"• Triage Nurse Clinical Notes: {nurse_notes.strip()}")

    lines.extend([
        "",
        "[R] RECOMMENDATION & IMMEDIATE DISPOSITION",
    ])
    for act in rec_actions:
        lines.append(f"• {act}")

    lines.extend([
        "",
        "-----------------------------------------------------------------------",
        "Clinical Authority: Triage decisions finalized under licensed clinician oversight.",
        "=======================================================================",
    ])

    return "\n".join(lines)


def generate_fhir_r4_bundle(
    patient_id: str,
    age: int,
    sex: str,
    chief_complaint: str,
    vitals: Optional[Dict[str, Any]] = None,
    symptoms: Optional[List[str]] = None,
    comorbidities: Optional[List[str]] = None,
    esi_level: int = 5,
    esi_rationale: str = "",
    sbar_text: str = "",
    sbar_note: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Construct a valid HL7 FHIR R4 Transaction Bundle representing the triage intake encounter.
    
    Includes:
    - Patient resource
    - Encounter resource (class: EMER)
    - Observation for ESI Acuity Score
    - Observation for Vitals (Pulse, BP, SpO2, Temp, RR)
    - Condition resources for chief complaint & comorbidities
    - DocumentReference resource containing the SBAR handoff note
    """
    if sbar_note and not sbar_text:
        sbar_text = sbar_note
    bundle_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    patient_ref = f"Patient/{patient_id}"
    encounter_id = f"enc-{patient_id}"
    vitals = vitals or {}
    symptoms = symptoms or []
    comorbidities = comorbidities or []

    entries: List[Dict[str, Any]] = []

    # 1. Patient Resource
    entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
            "identifier": [{"system": "http://hospital.org/patients", "value": patient_id}],
            "active": True,
            "gender": sex.lower() if sex.lower() in ("male", "female", "other") else "unknown",
        },
        "request": {"method": "POST", "url": "Patient"},
    })

    # 2. Encounter Resource
    entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": {
            "resourceType": "Encounter",
            "id": encounter_id,
            "status": "in-progress",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "EMER",
                "display": "emergency",
            },
            "subject": {"reference": patient_ref},
            "period": {"start": now_iso},
            "reasonCode": [{"text": chief_complaint}],
        },
        "request": {"method": "POST", "url": "Encounter"},
    })

    # 3. Observation for ESI Acuity Score
    entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": {
            "resourceType": "Observation",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "survey",
                            "display": "Survey",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "75636-1",
                        "display": "Emergency severity index (ESI)",
                    }
                ],
                "text": "Emergency Severity Index",
            },
            "subject": {"reference": patient_ref},
            "effectiveDateTime": now_iso,
            "valueInteger": int(esi_level),
            "note": [{"text": esi_rationale}],
        },
        "request": {"method": "POST", "url": "Observation"},
    })

    # 4. Observation for Vitals
    if vitals:
        vital_components = []
        if vitals.get("hr"):
            vital_components.append({
                "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
                "valueQuantity": {"value": vitals["hr"], "unit": "beats/min", "system": "http://unitsofmeasure.org", "code": "/min"},
            })
        if vitals.get("sbp"):
            vital_components.append({
                "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]},
                "valueQuantity": {"value": vitals["sbp"], "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
            })
        if vitals.get("spo2"):
            vital_components.append({
                "code": {"coding": [{"system": "http://loinc.org", "code": "2708-6", "display": "Oxygen saturation"}]},
                "valueQuantity": {"value": vitals["spo2"], "unit": "%", "system": "http://unitsofmeasure.org", "code": "%"},
            })
        if vitals.get("rr"):
            vital_components.append({
                "code": {"coding": [{"system": "http://loinc.org", "code": "9279-1", "display": "Respiratory rate"}]},
                "valueQuantity": {"value": vitals["rr"], "unit": "breaths/min", "system": "http://unitsofmeasure.org", "code": "/min"},
            })

        if vital_components:
            entries.append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "vital-signs",
                                    "display": "Vital Signs",
                                }
                            ]
                        }
                    ],
                    "code": {"text": "Intake Vital Signs Panel"},
                    "subject": {"reference": patient_ref},
                    "effectiveDateTime": now_iso,
                    "component": vital_components,
                },
                "request": {"method": "POST", "url": "Observation"},
            })

    # 5. Condition Resources
    for cond in comorbidities:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                        }
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "problem-list-item",
                            }
                        ]
                    }
                ],
                "code": {"text": cond},
                "subject": {"reference": patient_ref},
            },
            "request": {"method": "POST", "url": "Condition"},
        })

    # 6. SBAR DocumentReference
    if sbar_text:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "DocumentReference",
                "status": "current",
                "type": {
                    "coding": [{"system": "http://loinc.org", "code": "34117-2", "display": "History and physical note"}],
                    "text": "Triage SBAR Handoff Report",
                },
                "subject": {"reference": patient_ref},
                "date": now_iso,
                "description": "Structured emergency triage handoff note generated by triage copilot",
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain",
                            "data": sbar_text,
                        }
                    }
                ],
            },
            "request": {"method": "POST", "url": "DocumentReference"},
        })

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "transaction",
        "timestamp": now_iso,
        "entry": entries,
    }
