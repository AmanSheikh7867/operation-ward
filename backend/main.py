"""
Operation Ward Zero - Main Application Entry Point (Hospital ICU Grade)
Sense -> Understand -> Predict -> Alert -> Respond
"""
import asyncio
from datetime import datetime, timezone
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.config import FRONTEND_DIR
from backend.database.supabase_client import flush_loop, init_db
from backend.mqtt.handler import last_sensor_seen, start_mqtt
from backend.schemas import Patient
from backend.state import S
from backend.ws_manager import manager, status_msg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("wardzero")


# Initialize default registered patients with clinical metadata
def _init_default_patients():
    defaults = [
        Patient(id="bed01", name="John Doe", age=58, blood_type="O+", caregiver="Nurse Sarah", bed_number="Bed 01 (Post-Op)"),
        Patient(id="bed02", name="Maria Garcia", age=64, blood_type="A+", caregiver="Nurse Michael", bed_number="Bed 02 (Telemetry / Sepsis Risk)"),
        Patient(id="bed03", name="Rajesh Patel", age=72, blood_type="B+", caregiver="Nurse Priya", bed_number="Bed 03 (Cardiology)"),
    ]
    for p in defaults:
        S.patients[p.id] = p


def _seed_initial_history():
    """Seed baseline telemetry observations so dashboard is instant and vibrant."""
    from backend.services.risk_engine import evaluate_clinical_metrics, explain, predict_risk
    now = datetime.now(timezone.utc).isoformat()
    seed_data = [
        {"patient_id": "bed01", "hr": 74.0, "spo2": 98.2, "sys": 122.0, "dia": 78.0, "temp": 36.8, "rr": 16.0, "glucose": 96.0, "map": 92.7, "shock_index": 0.61, "hrv": 48.0},
        {"patient_id": "bed02", "hr": 118.0, "spo2": 92.0, "sys": 94.0, "dia": 56.0, "temp": 38.8, "rr": 25.0, "glucose": 158.0, "map": 68.7, "shock_index": 1.26, "hrv": 17.0},
        {"patient_id": "bed03", "hr": 82.0, "spo2": 96.5, "sys": 136.0, "dia": 84.0, "temp": 37.1, "rr": 18.0, "glucose": 115.0, "map": 101.3, "shock_index": 0.60, "hrv": 36.0},
    ]
    for v in seed_data:
        risk = predict_risk(v)
        reasons = explain(v)
        clinical = evaluate_clinical_metrics(v)
        rec = {
            **v,
            "risk": round(risk, 3),
            "news2_score": clinical["news2"]["score"],
            "news2_tier": clinical["news2"]["tier"],
            "syndrome": clinical["primary_syndrome"],
            "ts": now,
        }
        S.history.append({**rec, "reasons": reasons, "clinical": clinical})


async def sensor_health_monitor():
    """Background task to detect stale or disconnected sensors (no update in > 25s)."""
    while True:
        try:
            await asyncio.sleep(10)
            now = time.time()
            stale_patients = []
            for pid, last_seen in last_sensor_seen.items():
                if now - last_seen > 25:
                    stale_patients.append(pid)

            if stale_patients:
                log.warning("Stale sensor data detected for: %s", stale_patients)
                await manager.broadcast({
                    "type": "stale_sensor",
                    "stale_patients": stale_patients,
                    "ts": now,
                })
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("sensor_health_monitor error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initializes DB, MQTT, background flush & health tasks."""
    S.loop = asyncio.get_running_loop()
    _init_default_patients()
    _seed_initial_history()

    # 1. Initialize Supabase
    init_db()
    S.db_connected = S.db_client is not None

    # 2. Start MQTT client
    mqtt_client = start_mqtt()

    # 3. Start background tasks
    flush_task = asyncio.create_task(flush_loop())
    monitor_task = asyncio.create_task(sensor_health_monitor())

    log.info("Operation Ward Zero Hospital-Grade backend started successfully")

    yield

    # Shutdown
    log.info("Shutting down Ward Zero backend...")
    flush_task.cancel()
    monitor_task.cancel()

    try:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
    except Exception:
        pass


app = FastAPI(
    title="Operation Ward Zero - Hospital Telemetry & Response System",
    description="Intelligent Home Healthcare Monitoring & Clinical Response Platform (10-Parameter Telemetry)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)

# Mount frontend static files if directory exists
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
