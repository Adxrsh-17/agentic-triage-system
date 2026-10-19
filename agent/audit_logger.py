"""Triage Decision Audit Logger & Concordance Analytics.

Logs every triage encounter, capturing:
- AI suggested ESI vs Nurse confirmed ESI
- Override status, direction (upgrade/downgrade), and clinical rationale
- Objective metrics for clinical auditability and model refinement
"""

from datetime import datetime, timezone
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("triage-audit-logger")

AUDIT_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "triage_encounters_log.jsonl",
)


class TriageAuditLogger:
    """Thread-safe encounter audit logger and concordance analytics provider."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, log_path: Optional[str] = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self.log_path = log_path or AUDIT_LOG_FILE
        self._memory_logs: List[Dict[str, Any]] = []
        self._file_lock = threading.Lock()
        self._load_existing_logs()
        self._initialized = True

    def _load_existing_logs(self) -> None:
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._memory_logs.append(json.loads(line))
            except Exception as exc:
                logger.warning("Could not read existing audit log file: %s", exc)

    def log_encounter(
        self,
        patient_id: str,
        age: int = 25,
        sex: str = "male",
        chief_complaint: str = "",
        ai_esi: Optional[int] = None,
        nurse_esi: Optional[int] = None,
        ai_suggested_esi: Optional[int] = None,
        nurse_final_esi: Optional[int] = None,
        ai_rationale: str = "",
        override_reason: Optional[str] = None,
        nurse_notes: Optional[str] = None,
        vitals: Optional[Dict[str, Any]] = None,
        symptoms: Optional[List[str]] = None,
        comorbidities: Optional[List[str]] = None,
        duration_days: float = 0.0,
        sbar_note: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Record a finalized clinical triage encounter.
        """
        ai_esi_val = int(ai_esi if ai_esi is not None else (ai_suggested_esi if ai_suggested_esi is not None else 5))
        nurse_esi_val = int(nurse_esi if nurse_esi is not None else (nurse_final_esi if nurse_final_esi is not None else ai_esi_val))

        is_overridden = nurse_esi_val != ai_esi_val
        if not is_overridden:
            direction = "concordant"
        elif nurse_esi_val < ai_esi_val:
            # Lower numerical ESI = higher clinical severity
            direction = "nurse_upgraded_severity"
        else:
            direction = "nurse_downgraded_severity"

        record = {
            "encounter_id": f"ENC-{len(self._memory_logs) + 1:04d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "chief_complaint": chief_complaint,
            "vitals": vitals or {},
            "symptoms": symptoms or [],
            "comorbidities": comorbidities or [],
            "duration_days": duration_days,
            "ai_esi": ai_esi_val,
            "ai_rationale": ai_rationale,
            "nurse_esi": nurse_esi_val,
            "is_overridden": is_overridden,
            "override_direction": direction,
            "override_reason": override_reason if is_overridden else None,
            "nurse_notes": nurse_notes or "",
            "sbar_note": sbar_note,
        }

        with self._file_lock:
            self._memory_logs.append(record)
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as exc:
                logger.error("Failed to append to audit log file: %s", exc)

        return record

    def get_encounters(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return encounter list in reverse chronological order."""
        with self._file_lock:
            logs = list(reversed(self._memory_logs))
            return logs[:limit] if limit else logs

    def get_analytics(self) -> Dict[str, Any]:
        """Compute real concordance and override metrics across logged encounters."""
        with self._file_lock:
            total = len(self._memory_logs)
            if total == 0:
                return {
                    "total_encounters": 0,
                    "concordant_count": 0,
                    "accepted_count": 0,
                    "concordance_rate_pct": 100.0,
                    "override_count": 0,
                    "override_rate_pct": 0.0,
                    "upgraded_count": 0,
                    "downgraded_count": 0,
                    "override_reasons": {},
                    "esi_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                }

            concordant = sum(1 for e in self._memory_logs if not e["is_overridden"])
            overrides = total - concordant
            upgraded = sum(1 for e in self._memory_logs if e["override_direction"] == "nurse_upgraded_severity")
            downgraded = sum(1 for e in self._memory_logs if e["override_direction"] == "nurse_downgraded_severity")

            reasons: Dict[str, int] = {}
            for e in self._memory_logs:
                if e.get("override_reason"):
                    r = e["override_reason"]
                    reasons[r] = reasons.get(r, 0) + 1

            dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for e in self._memory_logs:
                level = e.get("nurse_esi") or e.get("ai_esi", 5)
                if level in dist:
                    dist[level] += 1

            return {
                "total_encounters": total,
                "concordant_count": concordant,
                "accepted_count": concordant,
                "concordance_rate_pct": round((concordant / total) * 100, 1),
                "override_count": overrides,
                "override_rate_pct": round((overrides / total) * 100, 1),
                "upgraded_count": upgraded,
                "downgraded_count": downgraded,
                "override_reasons": reasons,
                "esi_distribution": dist,
            }

    def clear(self) -> None:
        """Clear logs (for testing purposes)."""
        with self._file_lock:
            self._memory_logs.clear()
            if os.path.exists(self.log_path):
                try:
                    os.remove(self.log_path)
                except Exception:
                    pass

    clear_logs = clear
    get_metrics = get_analytics
    get_recent_encounters = get_encounters


def get_audit_logger() -> TriageAuditLogger:
    return TriageAuditLogger()
