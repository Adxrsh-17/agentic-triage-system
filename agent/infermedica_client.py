"""Infermedica Clinical Triage Engine API Client.

Provides medical free-text symptom parsing, diagnostic reasoning,
and triage assessment using the Infermedica v3 API.

Developer trial credentials can be obtained for free at https://developer.infermedica.com.
If credentials are not configured or the service is unreachable, all functions
gracefully return None to allow clean zero-failure fallbacks.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("infermedica-client")

INFERMEDICA_API_BASE = "https://api.infermedica.com/v3"
REQUEST_TIMEOUT_SECONDS = 5.0


def _get_headers() -> Optional[Dict[str, str]]:
    """Retrieve API headers containing App-Id and App-Key credentials."""
    app_id = os.getenv("INFERMEDICA_APP_ID", "").strip()
    app_key = os.getenv("INFERMEDICA_APP_KEY", "").strip()
    model = os.getenv("INFERMEDICA_MODEL", "infermedica-en").strip()

    if not app_id or not app_key:
        return None

    headers = {
        "App-Id": app_id,
        "App-Key": app_key,
        "Content-Type": "application/json",
    }
    if model:
        headers["Model"] = model
    return headers


def is_infermedica_configured() -> bool:
    """Check if Infermedica API credentials are set in the environment."""
    return _get_headers() is not None


def parse_symptoms(
    text: str,
    age: int = 30,
    sex: str = "male",
) -> Optional[Dict[str, Any]]:
    """
    Parse free-text clinical intake complaints into Infermedica structured concepts.

    Calls POST /v3/parse.
    Returns a dictionary containing:
    - 'symptoms': list of matched present symptom names
    - 'evidence': list of formatted Infermedica evidence objects
    - 'pertinent_negatives': list of explicitly absent symptoms
    - 'mentions': raw mention objects from Infermedica

    Returns None if unconfigured or on any network/API failure.
    """
    if not text or not text.strip():
        return None

    headers = _get_headers()
    if not headers:
        return None

    sex_normalized = sex.lower().strip() if sex and sex.lower().strip() in {"male", "female"} else "male"
    age_clamped = max(min(int(age or 30), 120), 0)

    payload = {
        "text": text.strip(),
        "age": {"value": age_clamped, "unit": "year"},
        "sex": sex_normalized,
        "include_tokens": True,
    }

    try:
        url = f"{INFERMEDICA_API_BASE}/parse"
        resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            mentions = data.get("mentions", [])
            symptoms: List[str] = []
            pertinent_negatives: List[str] = []
            evidence: List[Dict[str, Any]] = []

            for mention in mentions:
                choice_id = mention.get("choice_id", "present")
                name = mention.get("common_name") or mention.get("name") or mention.get("orth", "")
                name_clean = name.strip()

                if choice_id == "present":
                    if name_clean:
                        symptoms.append(name_clean.lower())
                    evidence.append({"id": mention["id"], "choice_id": "present", "source": "initial"})
                elif choice_id == "absent":
                    if name_clean:
                        pertinent_negatives.append(f"denies {name_clean.lower()}")
                    evidence.append({"id": mention["id"], "choice_id": "absent", "source": "initial"})

            logger.info(
                "Infermedica /v3/parse extracted %d symptoms, %d negatives from intake text",
                len(symptoms),
                len(pertinent_negatives),
            )
            return {
                "symptoms": list(dict.fromkeys(symptoms)),
                "evidence": evidence,
                "pertinent_negatives": list(dict.fromkeys(pertinent_negatives)),
                "mentions": mentions,
                "obvious": data.get("obvious", False),
            }
        else:
            logger.warning("Infermedica /v3/parse returned HTTP %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Infermedica /v3/parse call failed: %s", exc)

    return None


def get_diagnosis(
    evidence: List[Dict[str, Any]],
    age: int = 30,
    sex: str = "male",
) -> Optional[Dict[str, Any]]:
    """
    Query the Infermedica diagnostic engine with patient evidence.

    Calls POST /v3/diagnosis.
    Returns questions to ask next and/or ranked clinical conditions.
    Returns None on failure.
    """
    if not evidence:
        return None

    headers = _get_headers()
    if not headers:
        return None

    sex_normalized = sex.lower().strip() if sex and sex.lower().strip() in {"male", "female"} else "male"
    age_clamped = max(min(int(age or 30), 120), 0)

    payload = {
        "sex": sex_normalized,
        "age": {"value": age_clamped, "unit": "year"},
        "evidence": evidence,
    }

    try:
        url = f"{INFERMEDICA_API_BASE}/diagnosis"
        resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "question": data.get("question"),
                "conditions": data.get("conditions", []),
                "should_stop": data.get("should_stop", False),
            }
        else:
            logger.warning("Infermedica /v3/diagnosis returned HTTP %d", resp.status_code)
    except Exception as exc:
        logger.warning("Infermedica /v3/diagnosis call failed: %s", exc)

    return None


def get_triage(
    evidence: List[Dict[str, Any]],
    age: int = 30,
    sex: str = "male",
) -> Optional[Dict[str, Any]]:
    """
    Perform clinical triage assessment using Infermedica's certified engine.

    Calls POST /v3/triage.
    Returns a dictionary with:
    - 'triage_level': 'emergency_ambulance' | 'emergency' | 'consultation_24' | 'consultation' | 'self_care'
    - 'serious': list of critical conditions detected
    - 'description': clinical rationale for the triage level
    - 'root_cause': symptom ID that drove the emergency decision

    Returns None on failure or missing credentials.
    """
    if not evidence:
        return None

    headers = _get_headers()
    if not headers:
        return None

    sex_normalized = sex.lower().strip() if sex and sex.lower().strip() in {"male", "female"} else "male"
    age_clamped = max(min(int(age or 30), 120), 0)

    payload = {
        "sex": sex_normalized,
        "age": {"value": age_clamped, "unit": "year"},
        "evidence": evidence,
    }

    try:
        url = f"{INFERMEDICA_API_BASE}/triage"
        resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            triage_level = data.get("triage_level", "self_care")
            logger.info("Infermedica /v3/triage returned triage_level: %s", triage_level)
            return {
                "triage_level": triage_level,
                "serious": data.get("serious", []),
                "description": data.get("description", ""),
                "label": data.get("label", ""),
                "root_cause": data.get("root_cause"),
                "teleconsultation_applicable": data.get("teleconsultation_applicable", False),
            }
        else:
            logger.warning("Infermedica /v3/triage returned HTTP %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Infermedica /v3/triage call failed: %s", exc)

    return None


def get_medication_guidance(
    symptoms: List[str],
    conditions: Optional[List[str]] = None,
    age: int = 30,
    sex: str = "male",
    risk_level: str = "LOW",
    is_emergency: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate evidence-based Over-The-Counter (OTC) medication and supportive care guidance
    tailored to the patient's symptoms, age, comorbidities, and triage acuity level.
    """
    conditions_lower = [c.lower() for c in (conditions or [])]
    symptoms_joined = " ".join(symptoms).lower()
    meds = []

    # If emergency / high-risk, provide critical emergency guidance and caution against self-medication
    if is_emergency or risk_level == "HIGH":
        meds.append({
            "category": "🚨 Emergency Safety Protocol",
            "name": "No unguided oral self-medication",
            "purpose": "Acutely unstable presentation requires immediate physician / ER evaluation",
            "dosage": "Do not administer unprescribed medications prior to emergency evaluation",
            "precautions": "Taking sedatives, NSAIDs, or oral fluids during acute emergent situations (e.g. loss of consciousness, severe acute distress, suspected stroke or surgical abdomen) can obscure symptoms or compromise airway safety.",
        })
        if "chest pain" in symptoms_joined and not any("bleeding" in c or "ulcer" in c for c in conditions_lower):
            meds.append({
                "category": "Emergency Protocol (EMS / Dispatch Directed)",
                "name": "Aspirin (Dispersible / Chewable 300mg)",
                "purpose": "Antiplatelet therapy for suspected acute coronary syndrome",
                "dosage": "Single 300mg dose chewed (ONLY if directed by emergency dispatch/physician)",
                "precautions": "Contraindicated if allergic to aspirin, history of active gastrointestinal bleeding, or recent severe trauma.",
            })
        return meds

    has_ulcer = any("ulcer" in c or "gerd" in c or "gastritis" in c for c in conditions_lower)
    has_kidney = any("kidney" in c or "renal" in c for c in conditions_lower)
    has_liver = any("liver" in c or "hepatic" in c for c in conditions_lower)
    has_htn = any("hypertension" in c or "high blood pressure" in c or "bp" in c for c in conditions_lower)
    has_asthma = any("asthma" in c or "copd" in c for c in conditions_lower)
    has_diabetes = any("diabetes" in c or "sugar" in c for c in conditions_lower)

    # 1. Pain / Fever / Headache / Body ache / Sore throat
    if any(s in symptoms_joined for s in ["headache", "fever", "pain", "sore throat", "body ache", "muscle pain", "joint pain"]):
        paracetamol_dose = "500mg - 650mg every 4 to 6 hours as needed (Maximum 3000mg in 24 hours)"
        prec = "Do not exceed maximum daily dosage. Avoid alcohol while taking acetaminophen."
        if has_liver:
            paracetamol_dose = "Consult clinician before taking; reduce max daily dose to < 2000mg due to liver comorbidity"
            prec += " [Liver Comorbidity Alert: Reduced dosage recommended]."
        meds.append({
            "category": "Analgesic & Antipyretic",
            "name": "Paracetamol / Acetaminophen",
            "purpose": "Relief of headache, fever, sore throat, and generalized body aches",
            "dosage": paracetamol_dose,
            "precautions": prec,
        })

        if not has_ulcer and not has_kidney and not has_asthma:
            meds.append({
                "category": "Anti-inflammatory (NSAID)",
                "name": "Ibuprofen",
                "purpose": "Relief of inflammatory pain and fever",
                "dosage": "200mg - 400mg every 6 to 8 hours with food (Max 1200mg/day OTC)",
                "precautions": "Take with food or milk to prevent gastric irritation. Avoid if history of stomach ulcers or kidney disease.",
            })

    # 2. Cough / Cold / Congestion / Upper Respiratory
    if any(s in symptoms_joined for s in ["cough", "cold", "congestion", "runny nose", "sneezing", "phlegm"]):
        if "cough" in symptoms_joined:
            meds.append({
                "category": "Cough Relief",
                "name": "Dextromethorphan (Dry Cough) / Guaifenesin (Chest Congestion)",
                "purpose": "Suppresses non-productive dry cough OR loosens thick bronchial mucus",
                "dosage": "Per packaging instructions (10-20mg every 4h for dry cough; 200-400mg every 4h for chest congestion)",
                "precautions": "Ensure adequate fluid intake. Avoid combination syrups that duplicate acetaminophen.",
            })
        if any(s in symptoms_joined for s in ["runny nose", "sneezing", "congestion", "itchy"]):
            antihistamine = "Cetirizine (10mg) or Loratadine (10mg)"
            if has_htn:
                prec = "Non-sedating antihistamine preferred. Avoid oral decongestants with pseudoephedrine/phenylephrine as they elevate blood pressure."
            else:
                prec = "Non-drowsy formulation preferred for daytime use. Stay well hydrated."
            meds.append({
                "category": "Antihistamine",
                "name": antihistamine,
                "purpose": "Relief of runny nose, sneezing, and allergic upper airway symptoms",
                "dosage": "10mg once daily in the evening",
                "precautions": prec,
            })
        if "sore throat" in symptoms_joined:
            meds.append({
                "category": "Throat Care",
                "name": "Warm Saline Gargle & Antiseptic Lozenges",
                "purpose": "Soothes irritated mucosal lining and reduces pharyngeal inflammation",
                "dosage": "Gargle 1/2 tsp salt in warm water 3-4 times daily; lozenge every 2-3 hours as needed",
                "precautions": "Do not swallow large amounts of salt water. Do not give lozenges to young children.",
            })

    # 3. Gastrointestinal: Nausea, Vomiting, Diarrhea, Stomach upset
    if any(s in symptoms_joined for s in ["nausea", "vomiting", "diarrhea", "loose stools", "stomach ache", "abdominal pain"]):
        meds.append({
            "category": "Oral Hydration & Electrolytes",
            "name": "Oral Rehydration Salts (ORS) / Electrolyte Solution",
            "purpose": "Prevents dehydration and restores essential potassium and sodium balance",
            "dosage": "Sip 200ml - 300ml after each loose stool or frequently in small sips if nauseous",
            "precautions": "Drink slowly in small sips rather than gulping to avoid triggering emesis. Avoid sugary juices and dairy.",
        })
        if "diarrhea" in symptoms_joined and "fever" not in symptoms_joined:
            meds.append({
                "category": "Antidiarrheal Support",
                "name": "Probiotic Capsules / Loperamide (2mg)",
                "purpose": "Restores healthy gut flora and reduces stool frequency",
                "dosage": "Probiotic 1 capsule twice daily with meals (or Loperamide 4mg initial, then 2mg after each loose stool, max 8mg/day)",
                "precautions": "Do NOT take Loperamide if high fever, severe bloody stools, or bacterial dysentery is suspected.",
            })

    # 4. Acidity, Heartburn, Gastric reflux
    if any(s in symptoms_joined for s in ["acidity", "heartburn", "gerd", "burning", "indigestion", "acid reflux"]):
        meds.append({
            "category": "Antacid & Acid Reducer",
            "name": "Antacid Liquid Gel (Magnesium/Aluminum Hydroxide) or Famotidine 20mg",
            "purpose": "Rapid neutralization of excess stomach acid and heartburn relief",
            "dosage": "Antacid: 10-20ml after meals and at bedtime; Famotidine: 20mg once or twice daily",
            "precautions": "Avoid eating within 2 hours of lying down. Elevate head during sleep. Seek emergency care if accompanied by radiating chest discomfort.",
        })

    # 5. Rash, Itching, Skin irritation
    if any(s in symptoms_joined for s in ["rash", "itch", "itchy", "hives", "skin"]):
        meds.append({
            "category": "Topical Soothing & Antihistamine",
            "name": "Calamine Lotion / Topical 1% Hydrocortisone & Oral Cetirizine 10mg",
            "purpose": "Reduces localized skin inflammation, allergic redness, and pruritus",
            "dosage": "Apply thin layer to affected clean skin 2-3 times daily; 10mg cetirizine tablet once daily",
            "precautions": "Avoid applying near eyes or open infected wounds. Seek immediate emergency care if facial swelling or wheezing occurs (anaphylaxis).",
        })

    # If no specific match found, provide general supportive guidance
    if not meds:
        meds.append({
            "category": "Supportive Rest & Hydration",
            "name": "Hydration, Electrolytes & Paracetamol as needed",
            "purpose": "Supportive symptom management for mild, uncharacterized discomfort",
            "dosage": "Maintain 2-3 liters of fluids daily; Paracetamol 500mg every 6h if mild aches or fever develop",
            "precautions": "Monitor for new or escalating symptoms. Consult a clinician if symptoms persist beyond 48-72 hours.",
        })

    return meds
