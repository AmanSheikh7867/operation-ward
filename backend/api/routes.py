"""
Operation Ward Zero – REST & WebSocket Routes
"""
import asyncio
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from backend.config import FRONTEND_INDEX
from backend.database.supabase_client import (
    get_alerts,
    get_caregiver_activities,
    get_patient_history,
)
from backend.schemas import (
    Alert,
    CaregiverActivity,
    ChatMessage,
    ChatResponse,
    Patient,
    SystemStatus,
    VitalReading,
)
from backend.services.alert_service import acknowledge_alert
from backend.services.groq_service import chat_reply
from backend.services.risk_engine import explain_full
from backend.state import S
from backend.ws_manager import manager, status_msg

log = logging.getLogger("wardzero.api")
router = APIRouter()


# ── Frontend & Health ─────────────────────────────────────────────────────────

@router.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(FRONTEND_INDEX)


@router.get("/health", response_model=SystemStatus)
async def get_health():
    return status_msg()


# ── Patients ──────────────────────────────────────────────────────────────────

@router.get("/api/patients", response_model=List[Patient])
async def list_patients():
    return list(S.patients.values())


@router.post("/api/patients", response_model=Patient)
async def register_patient(p: Patient):
    S.patients[p.id] = p
    return p


@router.get("/api/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    p = S.patients.get(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return p


@router.get("/api/patients/{patient_id}/vitals")
async def get_patient_vitals(patient_id: str, limit: int = 60):
    return await get_patient_history(patient_id, limit=limit)


# ── Alerts & Caregiver Activities ─────────────────────────────────────────────

@router.get("/api/alerts")
async def list_alerts(patient_id: Optional[str] = None, limit: int = 30):
    return await get_alerts(patient_id=patient_id, limit=limit)


@router.get("/api/patients/{patient_id}/alerts")
async def get_patient_alerts(patient_id: str, limit: int = 30):
    return await get_alerts(patient_id=patient_id, limit=limit)


@router.post("/api/alerts/{alert_id}/ack")
async def post_acknowledge_alert(alert_id: str, caregiver: str = "Nurse Sarah"):
    alert = await acknowledge_alert(alert_id, caregiver_id=caregiver)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "ok", "alert": alert}


@router.post("/api/patients/{patient_id}/alerts/{alert_id}/ack")
async def post_acknowledge_patient_alert(patient_id: str, alert_id: str, caregiver: str = "Nurse Sarah"):
    return await post_acknowledge_alert(alert_id, caregiver=caregiver)


@router.get("/api/caregiver/activities")
async def list_caregiver_activities(limit: int = 50):
    return await get_caregiver_activities(limit=limit)


# ── Explainability ────────────────────────────────────────────────────────────

@router.get("/api/explain/{patient_id}")
async def get_explainability(patient_id: str):
    # Find latest vital reading for patient
    latest = None
    for item in reversed(S.history):
        if item.get("patient_id") == patient_id:
            latest = item
            break
    if not latest and S.history:
        latest = S.history[-1]

    if not latest:
        # Provide sample explanation if no readings received yet
        latest = {"hr": 75, "spo2": 98, "sys": 120, "dia": 78, "temp": 36.8}

    return explain_full(latest)


# ── WebSocket Chat Helper ────────────────────────────────────────────────────

async def handle_chat_ws(ws: WebSocket, history: list, text: str, lang: str):
    """Answer one chat message and send reply strictly to the asking client."""
    try:
        reply = await chat_reply(text[:500], history, lang)
        history.append({"role": "user", "content": text[:500]})
        history.append({"role": "assistant", "content": reply})
        del history[:-12]
        await asyncio.wait_for(
            ws.send_text(json.dumps({"type": "chat", "text": reply})),
            timeout=5,
        )
    except Exception as e:
        log.warning("WebSocket chat send failed: %s", e)


# ── WebSocket Endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Initial snapshot to populate UI instantly
        recent_history = list(S.history)[-60:]
        recent_alerts = sorted(S.alerts.values(), key=lambda a: a.get("ts", ""), reverse=True)[:30]

        await ws.send_text(json.dumps({
            "type": "snapshot",
            "history": recent_history,
            "alerts": recent_alerts,
            "patients": [p.model_dump() for p in S.patients.values()],
        }))
        await ws.send_text(json.dumps(status_msg()))

        chat_history: list = []

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            msg_type = msg.get("type")

            if msg_type == "ack":
                alert_id = msg.get("id")
                caregiver = msg.get("caregiver", "Nurse Sarah")
                if alert_id:
                    await acknowledge_alert(alert_id, caregiver_id=caregiver)

            elif msg_type == "chat":
                text = str(msg.get("text", "")).strip()
                lang = str(msg.get("lang", "en"))
                if text:
                    asyncio.create_task(handle_chat_ws(ws, chat_history, text, lang))

            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug("WebSocket connection closed: %s", e)
    finally:
        manager.drop(ws)
