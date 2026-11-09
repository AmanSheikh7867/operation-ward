"""
Operation Ward Zero - MQTT Handler (Hospital ICU Grade)
========================================================
Handles broker connection, message deserialization, 10-parameter vital validation,
and pipeline dispatch: SENSE -> UNDERSTAND -> PREDICT -> ALERT.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import paho.mqtt.client as mqtt

from backend.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, RISK_THRESHOLD
from backend.database.supabase_client import persist
from backend.services.alert_service import create_alert
from backend.services.risk_engine import evaluate_clinical_metrics, explain, predict_risk
from backend.state import S
from backend.ws_manager import manager, status_msg
from ml.predict import FEATURE_BOUNDS, compute_map, compute_shock_index

log = logging.getLogger("wardzero.mqtt")

# Track last received sensor timestamp per patient
last_sensor_seen: dict[str, float] = {}


def validate_payload(data: dict) -> tuple[bool, Optional[str]]:
    """Validate incoming sensor data against required schema and physiological ranges."""
    required = ("hr", "spo2", "sys", "dia", "temp")
    for k in required:
        if k not in data:
            return False, f"Missing required field: {k}"
        try:
            val = float(data[k])
            lo, hi = FEATURE_BOUNDS[k]
            if not (lo <= val <= hi):
                return False, f"Field {k}={val} out of plausible bounds [{lo}, {hi}]"
        except (ValueError, TypeError):
            return False, f"Non-numeric field: {k}={data[k]!r}"

    # Validate optional fields if supplied
    optional = ("rr", "glucose", "map", "shock_index", "hrv")
    for k in optional:
        if k in data and data[k] is not None:
            try:
                val = float(data[k])
                lo, hi = FEATURE_BOUNDS[k]
                if not (lo <= val <= hi):
                    return False, f"Field {k}={val} out of plausible bounds [{lo}, {hi}]"
            except (ValueError, TypeError):
                return False, f"Non-numeric field: {k}={data[k]!r}"

    return True, None


async def process_reading(topic: str, payload: bytes):
    """Pipeline: SENSE -> UNDERSTAND -> PREDICT -> ALERT."""
    try:
        # 1. Parse JSON
        try:
            d = json.loads(payload.decode("utf-8"))
        except Exception as e:
            log.warning("Malformed JSON on topic %s: %s", topic, e)
            return

        # 2. Extract patient_id from topic
        parts = topic.split("/")
        if len(parts) >= 2 and parts[0] == "wardzero":
            pid = parts[1]
        else:
            pid = d.get("patient_id", "bed01")

        # 3. Validate
        is_valid, err = validate_payload(d)
        if not is_valid:
            log.warning("Invalid sensor reading dropped for %s: %s", pid, err)
            return

        sys_val = float(d["sys"])
        dia_val = float(d["dia"])
        hr_val = float(d["hr"])

        map_val = float(d.get("map")) if d.get("map") is not None else compute_map(sys_val, dia_val)
        si_val = float(d.get("shock_index")) if d.get("shock_index") is not None else compute_shock_index(hr_val, sys_val)

        v = {
            "hr": hr_val,
            "spo2": float(d["spo2"]),
            "sys": sys_val,
            "dia": dia_val,
            "temp": float(d["temp"]),
            "rr": float(d.get("rr", 16.0)),
            "glucose": float(d.get("glucose", 100.0)),
            "map": map_val,
            "shock_index": si_val,
            "hrv": float(d.get("hrv", 45.0)),
        }

        now_epoch = time.time()
        last_sensor_seen[pid] = now_epoch

        # 4. Predict Risk (ML inference on 10 features)
        risk = predict_risk(v)

        # 5. Understand (Explainability reasons + NEWS2 + Clinical Syndromes)
        reasons = explain(v)
        clinical = evaluate_clinical_metrics(v)

        # 6. Record & Broadcast
        ts = datetime.now(timezone.utc).isoformat()
        rec = {
            "patient_id": pid,
            "hr": round(v["hr"], 1),
            "spo2": round(v["spo2"], 1),
            "sys": round(v["sys"], 1),
            "dia": round(v["dia"], 1),
            "temp": round(v["temp"], 1),
            "rr": round(v["rr"], 1),
            "glucose": round(v["glucose"], 1),
            "map": round(v["map"], 1),
            "shock_index": round(v["shock_index"], 2),
            "hrv": round(v["hrv"], 1),
            "risk": round(risk, 3),
            "news2_score": clinical["news2"]["score"],
            "news2_tier": clinical["news2"]["tier"],
            "syndrome": clinical["primary_syndrome"],
            "ts": ts,
        }

        # Store in recent history ring-buffer
        S.history.append({**rec, "reasons": reasons, "clinical": clinical})

        # Broadcast live update to WebSockets
        await manager.broadcast({
            "type": "vitals",
            "data": rec,
            "reasons": reasons,
            "clinical": clinical,
        })

        # Persist vitals asynchronously
        asyncio.create_task(persist("vitals", rec, op="insert"))

        # 7. Alert Trigger (Multi-criteria: ML Risk, NEWS2 High, or Critical Syndrome)
        is_critical_syndrome = clinical["primary_syndrome"].get("severity") == "CRITICAL"
        is_high_news2 = clinical["news2"]["score"] >= 7
        if risk > RISK_THRESHOLD or is_critical_syndrome or is_high_news2:
            asyncio.create_task(create_alert(
                patient_id=pid,
                vitals=v,
                risk=risk,
                reasons=reasons,
                alert_type="vital_risk",
                syndrome=clinical["primary_syndrome"],
                news2_score=clinical["news2"]["score"],
            ))

    except Exception as e:
        log.error("Unexpected error processing MQTT message on %s: %s", topic, e)


# ── MQTT Client Callbacks ───────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("MQTT connected to %s:%s", MQTT_BROKER, MQTT_PORT)
        S.mqtt_connected = True
        client.subscribe(MQTT_TOPIC)
        log.info("Subscribed to MQTT topic: %s", MQTT_TOPIC)
    else:
        log.warning("MQTT connection failed with code %s", rc)
        S.mqtt_connected = False
    if S.loop and S.loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.broadcast(status_msg()), S.loop)


def on_disconnect(client, userdata, rc, properties=None):
    log.warning("MQTT disconnected (rc=%s)", rc)
    S.mqtt_connected = False
    if S.loop and S.loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.broadcast(status_msg()), S.loop)


def on_message(client, userdata, msg):
    if S.loop and S.loop.is_running():
        asyncio.run_coroutine_threadsafe(process_reading(msg.topic, msg.payload), S.loop)


def start_mqtt() -> Optional[mqtt.Client]:
    """Initialize and start the background MQTT client loop."""
    try:
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"wardzero-backend-{uuid.uuid4().hex[:6]}",
            )
        except AttributeError:
            client = mqtt.Client(client_id=f"wardzero-backend-{uuid.uuid4().hex[:6]}")

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
        return client
    except Exception as e:
        log.error("Failed to start MQTT client: %s", e)
        S.mqtt_connected = False
        return None
