# Clinical Triage Copilot: Systematic Error Analysis

**Dataset Split:** `TEST` | **Total Cases:** 60 | **Exact Accuracy:** 58.33% | **Within-+-1:** 88.33%
**Under-Triage Rate (UTR):** 36.67% (Target: < 5.0%) | **Over-Triage Rate (OTR):** 5.0%
**Quadratic Weighted Kappa:** 0.8145

---

## 1. Acuity Confusion Matrix

```
                Predicted ESI 1   Predicted ESI 2   Predicted ESI 3   Predicted ESI 4   Predicted ESI 5
Actual ESI 1          10                2                 0                 0                 0
Actual ESI 2          0                 8                 1                 2                 1
Actual ESI 3          0                 1                 2                 5                 4
Actual ESI 4          0                 0                 1                 4                 7
Actual ESI 5          0                 0                 0                 1                 11
```

---

## 2. Under-Triage Cases (Highest Clinical Severity Risk: 22 cases)

Under-triage occurs when the system assigns a less urgent acuity level than the true clinical need (Predicted ESI > Ground Truth). In real emergency departments, under-triage risks delayed intervention.

### Case VIG-028: Gold ESI 1 -> Predicted ESI 2
- **Chief Complaint:** 23yo male with acute hyperkalemia 8.8 mEq/L from missed dialysis, wide sine-wave QRS complex on triage rhythm strip, lethargic
- **Vitals:** {'spo2': 93, 'hr': 36, 'sbp': 76, 'dbp': 42, 'rr': 22, 'temp': 36.4}
- **Vital Triggers Fired:** Extreme Heart Rate: 36 bpm, Hypotension: SBP 76 mmHg
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 1 was assigned ESI 2. Critical life threat missed resuscitation tier (ESI 1 -> 2).

### Case VIG-029: Gold ESI 1 -> Predicted ESI 2
- **Chief Complaint:** 28yo male with stab wound to left upper chest, cold clammy skin, muffled heart tones, JVD, SBP 74
- **Vitals:** {'spo2': 91, 'hr': 142, 'sbp': 74, 'dbp': 46, 'rr': 32, 'temp': 36.5}
- **Vital Triggers Fired:** Hypoxia: SpO2 91%, Extreme Heart Rate: 142 bpm, Extreme Respiratory Rate: 32/min, Hypotension: SBP 74 mmHg
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 1 was assigned ESI 2. Critical life threat missed resuscitation tier (ESI 1 -> 2).

### Case VIG-049: Gold ESI 2 -> Predicted ESI 4
- **Chief Complaint:** 30yo male with sudden acute left flank pain radiating to groin, writhing on triage stretcher in agony (10/10), intractable emesis
- **Vitals:** {'spo2': 98, 'hr': 112, 'sbp': 156, 'dbp': 94, 'rr': 22, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 2 was assigned ESI 4. Resource or danger threshold was underestimated.

### Case VIG-054: Gold ESI 2 -> Predicted ESI 5
- **Chief Complaint:** havin horible chesst preashur and cant breeth sweetin all over my arm herts
- **Vitals:** {'spo2': 94, 'hr': 102, 'sbp': 144, 'dbp': 92, 'rr': 24, 'temp': 36.9}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 2 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-056: Gold ESI 2 -> Predicted ESI 3
- **Chief Complaint:** 26yo female with severe lower quadrant abdominal pain, positive home pregnancy test, vaginal spotting, lightheaded
- **Vitals:** {'spo2': 98, 'hr': 118, 'sbp': 96, 'dbp': 60, 'rr': 22, 'temp': 36.7}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 2 was assigned ESI 3. Resource or danger threshold was underestimated.

### Case VIG-059: Gold ESI 2 -> Predicted ESI 4
- **Chief Complaint:** 29yo female 36 weeks pregnant with sudden severe persistent right upper quadrant pain, headache, scotoma, BP 178/110
- **Vitals:** {'spo2': 98, 'hr': 96, 'sbp': 178, 'dbp': 110, 'rr': 20, 'temp': 37.0}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 2 was assigned ESI 4. Resource or danger threshold was underestimated.

### Case VIG-079: Gold ESI 3 -> Predicted ESI 4
- **Chief Complaint:** 45yo male fell off ladder, visible deformity of right distal radius/wrist, neurovascularly intact, moderate pain
- **Vitals:** {'spo2': 99, 'hr': 82, 'sbp': 130, 'dbp': 80, 'rr': 16, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 4. Resource or danger threshold was underestimated.

### Case VIG-080: Gold ESI 3 -> Predicted ESI 4
- **Chief Complaint:** 55yo male with 3 weeks of progressive fatigue, pallor, dark tarry stools for 2 days, stable vital signs
- **Vitals:** {'spo2': 97, 'hr': 92, 'sbp': 120, 'dbp': 76, 'rr': 16, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 4. Resource or danger threshold was underestimated.

### Case VIG-081: Gold ESI 3 -> Predicted ESI 4
- **Chief Complaint:** 70yo male with painless jaundice, dark urine, 15 lb weight loss over 2 months, stable vitals
- **Vitals:** {'spo2': 97, 'hr': 74, 'sbp': 130, 'dbp': 78, 'rr': 16, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 4. Resource or danger threshold was underestimated.

### Case VIG-083: Gold ESI 3 -> Predicted ESI 4
- **Chief Complaint:** 24yo male with 18 hours of periumbilical pain migrating to right lower quadrant, low-grade fever, anorexia, McBurney tenderness
- **Vitals:** {'spo2': 98, 'hr': 84, 'sbp': 122, 'dbp': 76, 'rr': 16, 'temp': 37.9}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 4. Resource or danger threshold was underestimated.

### Case VIG-084: Gold ESI 3 -> Predicted ESI 5
- **Chief Complaint:** 63yo male with type 2 diabetes presenting with 5 days of worsening redness and swelling on left lower leg, spreading cellulitis, stable BP
- **Vitals:** {'spo2': 97, 'hr': 88, 'sbp': 136, 'dbp': 82, 'rr': 18, 'temp': 38.3}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-085: Gold ESI 3 -> Predicted ESI 5
- **Chief Complaint:** fell off bicycal hurt my elbo and knee, scraped up, big bump on elbo cant bend it, normal breathin
- **Vitals:** {'spo2': 99, 'hr': 80, 'sbp': 126, 'dbp': 78, 'rr': 16, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-087: Gold ESI 3 -> Predicted ESI 5
- **Chief Complaint:** 26yo female with severe burning epigastric pain unrelieved by antacids, positive stool guaiac, stable vitals
- **Vitals:** {'spo2': 99, 'hr': 78, 'sbp': 116, 'dbp': 72, 'rr': 14, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-089: Gold ESI 3 -> Predicted ESI 5
- **Chief Complaint:** 53yo female with sudden vertigo and nystagmus triggered by head movements, nausea, unable to walk unassisted
- **Vitals:** {'spo2': 98, 'hr': 78, 'sbp': 138, 'dbp': 84, 'rr': 16, 'temp': 36.9}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-090: Gold ESI 3 -> Predicted ESI 4
- **Chief Complaint:** Patient has severe stomache pain but no chest pain no shortness of breath denies vomitin, RLQ tender
- **Vitals:** {'spo2': 99, 'hr': 84, 'sbp': 118, 'dbp': 74, 'rr': 16, 'temp': 37.4}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 3 was assigned ESI 4. Resource or danger threshold was underestimated.

### Case VIG-109: Gold ESI 4 -> Predicted ESI 5
- **Chief Complaint:** 23yo male hit directly in nose during soccer match, mild epistaxis now stopped, nasal bridge swollen and tender, no septal hematoma
- **Vitals:** {'spo2': 99, 'hr': 74, 'sbp': 122, 'dbp': 76, 'rr': 14, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 4 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-110: Gold ESI 4 -> Predicted ESI 5
- **Chief Complaint:** 14yo male jammed middle finger catching a football, swollen proximal interphalangeal joint, tender, stable vitals
- **Vitals:** {'spo2': 99, 'hr': 76, 'sbp': 112, 'dbp': 72, 'rr': 16, 'temp': 36.7}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 4 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-111: Gold ESI 4 -> Predicted ESI 5
- **Chief Complaint:** 19yo male with severe sore throat for 2 days, swollen tonsils with white exudate, anterior cervical adenopathy, denies cough
- **Vitals:** {'spo2': 99, 'hr': 80, 'sbp': 120, 'dbp': 76, 'rr': 14, 'temp': 37.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 4 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-114: Gold ESI 4 -> Predicted ESI 5
- **Chief Complaint:** 45yo carpenter felt wood particle hit right eye, foreign body sensation, photophobia, cornea intact with superficial foreign speck
- **Vitals:** {'spo2': 99, 'hr': 76, 'sbp': 126, 'dbp': 80, 'rr': 14, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 4 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-115: Gold ESI 4 -> Predicted ESI 5
- **Chief Complaint:** 74yo female with sudden epistaxis for 1 hour, ongoing mild anterior ooze, stable blood pressure 130/80, no history of bleeding disorders
- **Vitals:** {'spo2': 98, 'hr': 76, 'sbp': 130, 'dbp': 80, 'rr': 16, 'temp': 36.8}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 4 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-116: Gold ESI 4 -> Predicted ESI 5
- **Chief Complaint:** 67yo female with localized painful dental caries and periapical gum swelling, no facial cellulitis, no airway compromise, afebrile
- **Vitals:** {'spo2': 98, 'hr': 78, 'sbp': 136, 'dbp': 82, 'rr': 16, 'temp': 36.9}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 4 was assigned ESI 5. Resource or danger threshold was underestimated.

### Case VIG-118: Gold ESI 4 -> Predicted ESI 5
- **Chief Complaint:** 30yo female stepped on clean sewing needle in carpet 2 hours ago, tip broke off in heel, small puncture mark, afebrile
- **Vitals:** {'spo2': 99, 'hr': 70, 'sbp': 118, 'dbp': 74, 'rr': 14, 'temp': 36.7}
- **Vital Triggers Fired:** None
- **AI Rationale:** 
- **AI Risk Reasons:** None
- **Failure Mode:** Gold ESI 4 was assigned ESI 5. Resource or danger threshold was underestimated.

---

## 3. Over-Triage Cases (Clinical Conservatism Bias: 3 cases)

Over-triage occurs when the system assigns a more acute level than the true clinical need (Predicted ESI < Ground Truth). In clinical triage guidelines, modest over-triage (25-35%) is accepted and encouraged as a safety buffer.

### Case VIG-088: Gold ESI 3 -> Predicted ESI 2
- **Chief Complaint:** patient feels nauseous and dizy after food poisoning 2 days ago, no blood in stool, feels weak, normal vitals
- **Vitals:** {'spo2': 98, 'hr': 92, 'sbp': 114, 'dbp': 72, 'rr': 16, 'temp': 37.2}
- **AI Rationale:** 
- **Clinical Context:** Elevated acuity due to conservative symptom/vital escalation.

### Case VIG-117: Gold ESI 4 -> Predicted ESI 3
- **Chief Complaint:** 28yo female with 2 days of dysuria, urinary frequency, and urgency, denies fever, flank pain, or vaginal discharge
- **Vitals:** {'spo2': 99, 'hr': 74, 'sbp': 114, 'dbp': 70, 'rr': 14, 'temp': 36.8}
- **AI Rationale:** 
- **Clinical Context:** Elevated acuity due to conservative symptom/vital escalation.

### Case VIG-145: Gold ESI 5 -> Predicted ESI 4
- **Chief Complaint:** 26yo male with single dry skin patch on elbow for 2 months, no itching, no erythema, requesting dermatologist referral
- **Vitals:** {'spo2': 99, 'hr': 66, 'sbp': 120, 'dbp': 74, 'rr': 14, 'temp': 36.7}
- **AI Rationale:** 
- **Clinical Context:** Elevated acuity due to conservative symptom/vital escalation.


---

## 4. Root Cause Analysis: ESI 4 vs. ESI 5 Resource Prediction

- **ESI 4 Recall:** 33.33%
- **ESI 5 Recall:** 91.67%

### Clinical Distinction in ESI v4 Guidelines:
1. **ESI 4 (One Resource Needed):** Patient requires exactly one distinct diagnostic or therapeutic resource (e.g. single plain radiography X-ray, simple skin laceration suturing, IM antibiotic injection, simple lab urinalysis/swab).
2. **ESI 5 (Zero Resources Needed):** Patient requires exam only, prescription refill, suture removal, or minor wound cleaning without diagnostic testing.

### Identified Mechanism of Failure in Baseline:
The baseline heuristic mapped acuity using raw keyword counts (`moderate_count >= 1 -> ESI 4`, else `ESI 5`). Presentations requiring single diagnostic interventions (sprained ankle needing X-ray, laceration needing suture, uncomplicated dysuria needing urinalysis) were not tagged in the moderate symptom dictionary and collapsed into ESI 5.
