"""
Operation Ward Zero - Alert Service (Hospital ICU Grade)
=========================================================
Alert generation, cooldown enforcement, acknowledgment tracking,
and caregiver activity logging with clinical syndrome context.
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config import ALERT_COOLDOWN_S, RISK_THRESHOLD
from backend.database.supabase_client import persist
from backend.services.groq_service import groq_advice
from backend.state import S
from backend.ws_manager import manager

log = logging.getLogger("wardzero.alert")


async def create_alert(
    patient_id: str,
    vitals: Dict[str, Any],
    risk: float,
    reasons: List[Dict[str, Any]],
    alert_type: str = "vital_risk",
    syndrome: Optional[Dict[str, Any]] = None,
    news2_score: int = 0,
) -> Optional[Dict[str, Any]]:
    """Create and broadcast an alert with cooldown check."""
    now = time.time()
    last_ts = S.last_alert_ts.get(patient_id, 0)
    if now - last_ts < ALERT_COOLDOWN_S:
        log.debug("Alert suppressed for %s (cooldown active: %.1fs remaining)",
                  patient_id, ALERT_COOLDOWN_S - (now - last_ts))
        return None

    S.last_alert_ts[patient_id] = now

    try:
        if alert_type == "sensor_offline":
            advice = (
                "- Check the sensor battery and wireless connection at the patient bedside.\n"
                "- Verify sensor placement on the patient if readings have stalled."
            )
        else:
            advice = await groq_advice(vitals, risk, reasons)

        alert_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        alert = {
            "id": alert_id,
            "patient_id": patient_id,
            "risk": round(risk, 3),
            "reasons": reasons,
            "advice": advice,
            "syndrome": syndrome or vitals.get("syndrome"),
            "news2_score": news2_score or vitals.get("news2_score", 0),
            "acknowledged": False,
            "ts": ts,
            "alert_type": alert_type,
            "acknowledged_by": None,
            "acknowledged_at": None,
        }

        S.alerts[alert_id] = alert

        # Broadcast via WebSocket
        await manager.broadcast({"type": "alert", "data": alert})

        # Persist to database / retry buffer
        asyncio.create_task(persist("alerts", {
            "id": alert_id,
            "patient_id": patient_id,
            "risk": round(risk, 3),
            "reasons": reasons,
            "advice": advice,
            "acknowledged": False,
            "ts": ts,
        }))

        log.info("Alert created [%s] for %s - Risk: %.2f | NEWS2: %d (%d reasons)",
                 alert_id[:8], patient_id, risk, news2_score, len(reasons))
        return alert

    except Exception as e:
        log.error("Failed to create alert for %s: %s", patient_id, e)
        return None


async def acknowledge_alert(
    alert_id: str,
    caregiver_id: str = "Nurse Sarah",
) -> Optional[Dict[str, Any]]:
    """Acknowledge an alert and record caregiver activity."""
    alert = S.alerts.get(alert_id)
    if not alert:
        log.warning("Attempted to acknowledge unknown alert: %s", alert_id)
        return None

    now_iso = datetime.now(timezone.utc).isoformat()
    alert["acknowledged"] = True
    alert["acknowledged_by"] = caregiver_id
    alert["acknowledged_at"] = now_iso

    # Record caregiver action
    activity = {
        "id": str(uuid.uuid4()),
        "caregiver_id": caregiver_id,
        "action": f"Acknowledged alert for patient {alert['patient_id']} (Risk: {alert['risk']:.2f})",
        "alert_id": alert_id,
        "patient_id": alert["patient_id"],
        "ts": now_iso,
    }
    S.caregiver_activities.append(activity)

    # Persist alert update
    asyncio.create_task(
        persist("alerts", {"id": alert_id, "set": {"acknowledged": True, "acknowledged_by": caregiver_id, "acknowledged_at": now_iso}}, op="update")
    )
    asyncio.create_task(
        persist("caregiver_activity", activity, op="insert")
    )

    # Broadcast update
    await manager.broadcast({
        "type": "ack",
        "id": alert_id,
        "acknowledged_by": caregiver_id,
        "acknowledged_at": now_iso,
    })
    await manager.broadcast({"type": "alert_acked", "data": alert})
    await manager.broadcast({"type": "caregiver_activity", "data": activity})

    log.info("Alert [%s] acknowledged by %s", alert_id[:8], caregiver_id)
    return alert


def get_alerts(patient_id: Optional[str] = None, unacked_only: bool = False) -> List[Dict[str, Any]]:
    """Retrieve alerts filtered by patient and acknowledgment status."""
    res = list(S.alerts.values())
    if patient_id:
        res = [a for a in res if a.get("patient_id") == patient_id]
    if unacked_only:
        res = [a for a in res if not a.get("acknowledged")]
    return sorted(res, key=lambda a: a.get("ts", ""), reverse=True)


def get_caregiver_activities() -> List[Dict[str, Any]]:
    """Return caregiver activity log."""
    return list(S.caregiver_activities)[-50:]
