# Clinical Triage Copilot: Error Analysis & Misclassification Autopsy

> **Split Discipline Notice:** In accordance with [`eval/SPLIT_POLICY.md`](../eval/SPLIT_POLICY.md), all iterative rule engineering and threshold adjustments are derived strictly from the **Development (`dev`) Split ($n=90$)**. The **Held-Out Test Split ($n=60$)** is frozen and preserved exclusively for milestone validation.

---

## 1. Development Split ($n=90$) Error Analysis

On the un-tuned rules safety floor, the development split achieved an initial accuracy of **78.9% (71/90)**, emergency recall of **83.3% (30/36)**, and urgent recall of **33.3% (6/18)**. Analysis of the 19 discordant dev cases revealed the following actionable mechanisms:

### Category A: Urgent-Tier Complaints Collapsing to Routine (12 Cases)
- **Cases:** `c061`, `c062`, `c064`, `c065`, `c066`, `c067`, `c069`, `c070`, `c071`, `c075`, `c076`, `c077`
- **Root Cause:**
  - `c061` / `c066` / `c075`: Persistent fever (>3 days) or subacute infection without overt sepsis keywords was not caught by emergency red-flag lists and fell through to routine.
  - `c064` / `c077`: Moderate abdominal pain in patients with chronic comorbidities (e.g., gallstones, ulcer disease) lacked severe crisis keywords and collapsed to routine.
  - `c062` / `c065` / `c071`: Extremity injuries with moderate swelling, inability to bear weight, or closed joint distortion did not trigger laceration or fracture keywords.
  - `c069` / `c070` / `c076`: Palpitations or mild tachycardia with normal SBP fell through because heart rate was elevated (105--125 bpm) but below the severe danger threshold (140 bpm).
- **Calibrated Rule Action:** Introduce a dedicated, structured `URGENT_TIER_RULES` dictionary capturing subacute worsening, persistent fever >3 days, moderate abdominal symptoms with comorbidity, and mild vital sign danger zones (ESI v4 heuristic danger zones).

### Category B: High-Acuity Emergent Presentations Under-Triaged to Urgent (6 Cases)
- **Cases:** `c032`, `c035`, `c037`, `c040`, `c046`, `c047`
- **Root Cause:**
  - `c040` (Respiratory): Moderate hypoxia ($\text{SpO}_2 \le 94\%$) with tachypnea was classified as urgent rather than emergent because the rule floor only triggered emergency on $\text{SpO}_2 < 90\%$.
  - `c046` (Cardiac): Syncope or transient loss of consciousness described in colloquial terms without explicit keyword `"loss of consciousness"`.
  - `c047` (Neurological): Sudden confusion in an elderly patient without focal hemiplegia words.
- **Calibrated Rule Action:** Escalate adult tachypnea ($\text{RR} \ge 24$) with mild hypoxia ($\text{SpO}_2 \le 94\%$) to urgent/emergent, and expand syncope synonyms in red flags.

### Category C: Over-Triage on Dev Split (1 Case)
- **Case:** `c137` (Routine -> Urgent)
- **Root Cause:** Superficial minor cut where "cut" keyword triggered an urgent laceration rule despite explicitly stating "minor superficial scratch".
- **Calibrated Rule Action:** Enforce qualifier check (`"minor"`, `"superficial"`, `"small"`) to suppress urgent escalation on paper cuts and superficial scratches.

---

## 2. Frozen Held-Out Test Split ($n=60$) Milestone Observations

*(Preserved for historical comparison; frozen per split policy)*

The initial baseline test run on the frozen split yielded:
- **Overall Accuracy:** 66.7% (40/60) [54.1%–77.3%]
- **Macro-F1 Score:** 61.3% [49.1%–73.2%]
- **Emergency Recall (Sensitivity):** 62.5% (15/24) [42.7%–78.8%]
- **Emergency Precision:** 100.0% (15/15) [79.6%–100.0%]
- **Urgent Recall:** 33.3% (4/12) [13.8%–60.9%]
- **Routine Recall:** 87.5% (21/24) [69.0%–95.7%]
- **Under-Triage Rate:** 28.33% (17/60)
- **Over-Triage Rate:** 5.00% (3/60)

### Primary Failure Modes Observed in Initial Test Milestone:
1. **Typographical & Colloquial Phrasing:** Non-standard spellings failed exact keyword string matching in offline rules-only mode.
2. **Occult Severe Pain:** Writhing pain ($10/10$) with normal initial vital signs was under-counted when no vital collapse was present.
3. **Resource Boundary Cases:** Boundary ambiguity between ESI 4 (1 resource) and ESI 5 (0 resources).

---

## 3. Methodological Governance Summary

1. All prospective rule changes are calibrated against the 19 dev split failures above.
2. No developer has tuned keyword patterns against test split cases.
3. Test split re-runs are logged to `eval/results/run_log.jsonl` to ensure audit transparency.
