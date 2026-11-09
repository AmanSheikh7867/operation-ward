"""
Operation Ward Zero - Multi-Patient IoT Sensor Simulator (Hospital ICU Grade)
=============================================================================

Simulates multiple bedside vital-sign monitors publishing MQTT messages
to ``wardzero/{patient_id}/vitals`` across 10 hospital telemetry parameters.

Parameters Published:
- hr (Heart Rate, bpm)
- spo2 (Blood Oxygen Saturation, %)
- sys (Systolic BP, mmHg)
- dia (Diastolic BP, mmHg)
- temp (Core Temperature, °C)
- rr (Respiratory Rate, breaths/min)
- glucose (Capillary Blood Glucose, mg/dL)
- map (Mean Arterial Pressure, mmHg)
- shock_index (Shock Index, HR / SYS)
- hrv (Heart Rate Variability RMSSD, ms)

Clinical Simulation Scenarios:
- NORMAL: Stable homeostasis (NEWS2 = 0)
- SEPSIS: Sepsis / SIRS onset (High RR, Fever, Low MAP, High Shock Index)
- RESPIRATORY: Acute respiratory failure / COPD exacerbation (Severe hypoxemia)
- HYPOGLYCEMIA: Acute diabetic crash (Glucose < 50 mg/dL, Adrenergic strain)
- HYPERTENSIVE: Malignant hypertension (SYS > 190 mmHg, Elevated MAP)
- DETERIORATING: Dynamic continuous clinical decline over 60 seconds
- RECOVERING: Progressive return to safe clinical baseline
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import random
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional
import uuid

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print(
        "ERROR: paho-mqtt is required. Install with: pip install paho-mqtt",
        file=sys.stderr,
    )
    sys.exit(1)

from iot.mqtt_config import MQTTConfig, load_config, fallback_config

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("wardzero.sim")

# ── Physiological bounds ─────────────────────────────────────────────────────
BOUNDS = {
    "hr":          (30, 220),
    "spo2":        (50.0, 100.0),
    "sys":         (60, 260),
    "dia":         (30, 160),
    "temp":        (33.0, 42.5),
    "rr":          (6, 60),
    "glucose":     (20, 500),
    "map":         (30, 180),
    "shock_index": (0.2, 2.5),
    "hrv":         (5, 150),
}

# ── Clinical Scenario Profiles ───────────────────────────────────────────────
PROFILES = {
    "NORMAL": {
        "hr": 74.0, "spo2": 98.2, "sys": 120.0, "dia": 78.0, "temp": 36.8,
        "rr": 16.0, "glucose": 95.0, "hrv": 52.0,
    },
    "SEPSIS": {
        "hr": 124.0, "spo2": 92.0, "sys": 88.0, "dia": 52.0, "temp": 39.2,
        "rr": 26.0, "glucose": 165.0, "hrv": 15.0,
    },
    "RESPIRATORY": {
        "hr": 128.0, "spo2": 85.5, "sys": 145.0, "dia": 90.0, "temp": 37.2,
        "rr": 32.0, "glucose": 110.0, "hrv": 20.0,
    },
    "HYPOGLYCEMIA": {
        "hr": 112.0, "spo2": 97.5, "sys": 105.0, "dia": 65.0, "temp": 36.1,
        "rr": 19.0, "glucose": 46.0, "hrv": 18.0,
    },
    "HYPERTENSIVE": {
        "hr": 95.0, "spo2": 97.5, "sys": 195.0, "dia": 115.0, "temp": 37.0,
        "rr": 21.0, "glucose": 135.0, "hrv": 14.0,
    },
}

DEFAULT_PATIENTS = ["bed01", "bed02", "bed03"]
DEFAULT_INTERVAL = 5  # seconds

# Cycle durations (seconds spent in each automatic phase)
CYCLE_NORMAL        = 40
CYCLE_DETERIORATING = 15
CYCLE_CRITICAL      = 25
CYCLE_RECOVERING    = 15
CYCLE_TOTAL = CYCLE_NORMAL + CYCLE_DETERIORATING + CYCLE_CRITICAL + CYCLE_RECOVERING


class Scenario(str, Enum):
    NORMAL        = "NORMAL"
    SEPSIS        = "SEPSIS"
    RESPIRATORY   = "RESPIRATORY"
    HYPOGLYCEMIA  = "HYPOGLYCEMIA"
    HYPERTENSIVE  = "HYPERTENSIVE"
    DETERIORATING = "DETERIORATING"
    RECOVERING    = "RECOVERING"


@dataclass
class PatientState:
    """Mutable state for a single simulated patient monitor."""
    patient_id: str
    scenario: Scenario = Scenario.NORMAL
    manual_override: Optional[Scenario] = None
    connected: bool = True
    start_time: float = field(default_factory=time.time)
    buffer: Deque[Dict[str, Any]] = field(default_factory=deque)
    readings_sent: int = 0

    @property
    def label(self) -> str:
        tag = " (manual)" if self.manual_override else ""
        conn = " [OFFLINE]" if not self.connected else ""
        return f"{self.scenario.value}{tag}{conn}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lerp(a: dict, b: dict, t: float) -> dict:
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def _auto_phase(elapsed: float) -> tuple[Scenario, float]:
    """Calculate scenario and interpolation blend for dynamic auto-cycling."""
    t = elapsed % CYCLE_TOTAL
    if t < CYCLE_NORMAL:
        return Scenario.NORMAL, 0.0
    t -= CYCLE_NORMAL
    if t < CYCLE_DETERIORATING:
        return Scenario.DETERIORATING, t / CYCLE_DETERIORATING
    t -= CYCLE_DETERIORATING
    if t < CYCLE_CRITICAL:
        return Scenario.SEPSIS, 1.0
    t -= CYCLE_CRITICAL
    return Scenario.RECOVERING, 1.0 - (t / CYCLE_RECOVERING)


def generate_reading(patient_state: PatientState) -> Dict[str, Any]:
    """Generate a realistic 10-parameter vital sign reading."""
    scen = patient_state.manual_override or patient_state.scenario

    if scen == Scenario.DETERIORATING:
        elapsed = time.time() - patient_state.start_time
        blend = min(1.0, (elapsed % 60) / 60.0)
        base = _lerp(PROFILES["NORMAL"], PROFILES["SEPSIS"], blend)
    elif scen == Scenario.RECOVERING:
        elapsed = time.time() - patient_state.start_time
        blend = max(0.0, 1.0 - ((elapsed % 60) / 60.0))
        base = _lerp(PROFILES["NORMAL"], PROFILES["SEPSIS"], blend)
    elif scen.value in PROFILES:
        base = PROFILES[scen.value]
    else:
        base = PROFILES["NORMAL"]

    # Add realistic stochastic Gaussian jitter
    hr   = round(_clamp(base["hr"]   + random.gauss(0, 2.5), *BOUNDS["hr"]))
    spo2 = round(_clamp(base["spo2"] + random.gauss(0, 0.4), *BOUNDS["spo2"]), 1)
    sys_ = round(_clamp(base["sys"]  + random.gauss(0, 3.2), *BOUNDS["sys"]))
    dia  = round(_clamp(base["dia"]  + random.gauss(0, 2.2), *BOUNDS["dia"]))
    temp = round(_clamp(base["temp"] + random.gauss(0, 0.1), *BOUNDS["temp"]), 1)
    rr   = round(_clamp(base["rr"]   + random.gauss(0, 0.8), *BOUNDS["rr"]))
    glucose = round(_clamp(base["glucose"] + random.gauss(0, 2.0), *BOUNDS["glucose"]), 1)
    hrv  = round(_clamp(base["hrv"]  + random.gauss(0, 1.5), *BOUNDS["hrv"]), 1)

    # Autoderived hemodynamics
    map_ = round((2.0 * dia + sys_) / 3.0, 1)
    shock_index = round(float(hr) / max(float(sys_), 1.0), 2)

    return {
        "patient_id":  patient_state.patient_id,
        "hr":          hr,
        "spo2":        spo2,
        "sys":         sys_,
        "dia":         dia,
        "temp":        temp,
        "rr":          rr,
        "glucose":     glucose,
        "map":         map_,
        "shock_index": shock_index,
        "hrv":         hrv,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "device_id":   f"mon-{patient_state.patient_id}",
    }


# ── Simulator Engine ─────────────────────────────────────────────────────────
class MultiPatientSimulator:
    """Manages MQTT publishing across multiple patient telemetry simulators."""

    def __init__(
        self,
        config: MQTTConfig,
        patient_ids: Optional[List[str]] = None,
        interval: float = DEFAULT_INTERVAL,
        auto_cycle: bool = True,
    ):
        self.config = config
        self.interval = max(1.0, interval)
        self.auto_cycle = auto_cycle
        self.patients: Dict[str, PatientState] = {
            pid: PatientState(patient_id=pid) for pid in (patient_ids or DEFAULT_PATIENTS)
        }
        self.running = False
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._lock = threading.Lock()

    def _setup_mqtt(self):
        try:
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"wardzero-sim-{uuid.uuid4().hex[:6]}",
            )
        except AttributeError:
            self._client = mqtt.Client(client_id=f"wardzero-sim-{uuid.uuid4().hex[:6]}")

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.connect_async(self.config.broker, self.config.port, keepalive=60)
        self._client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info("Simulator connected to MQTT broker %s:%s", self.config.broker, self.config.port)
            self._connected = True
        else:
            logger.warning("Simulator MQTT connect failed: rc=%s", rc)
            self._connected = False

    def _on_disconnect(self, client, userdata, rc, properties=None):
        logger.warning("Simulator MQTT disconnected (rc=%s)", rc)
        self._connected = False

    def set_scenario(self, patient_id: str, scenario: Scenario):
        with self._lock:
            if patient_id in self.patients:
                self.patients[patient_id].manual_override = scenario
                self.patients[patient_id].scenario = scenario
                logger.info("Patient %s set to scenario %s", patient_id, scenario.value)

    def toggle_connection(self, patient_id: str):
        with self._lock:
            if patient_id in self.patients:
                p = self.patients[patient_id]
                p.connected = not p.connected
                status = "CONNECTED" if p.connected else "DISCONNECTED"
                logger.info("Patient %s sensor %s", patient_id, status)
                if p.connected and p.buffer:
                    logger.info("Flushing %d buffered readings for %s...", len(p.buffer), patient_id)
                    while p.buffer:
                        reading = p.buffer.popleft()
                        self._publish(reading)

    def _publish(self, reading: Dict[str, Any]):
        if not self._client or not self._connected:
            return False
        topic = f"wardzero/{reading['patient_id']}/vitals"
        payload = json.dumps(reading)
        res = self._client.publish(topic, payload, qos=0)
        return res.rc == mqtt.MQTT_ERR_SUCCESS

    def run(self):
        self._setup_mqtt()
        self.running = True
        logger.info("=" * 60)
        logger.info("Ward Zero Hospital Telemetry Simulator Started")
        logger.info("Patients: %s  |  Interval: %.1fs  |  Broker: %s:%s",
                    list(self.patients.keys()), self.interval, self.config.broker, self.config.port)
        logger.info("Commands: [s] Sepsis | [h] Hypoglycemia | [p] Respiratory | [y] Hypertensive | [n] Normal | [x] Disconnect")
        logger.info("=" * 60)

        # Stagger initial patient timing slightly
        for i, p in enumerate(self.patients.values()):
            p.start_time = time.time() - (i * 20)

        try:
            while self.running:
                now = time.time()
                for pid, p in self.patients.items():
                    if self.auto_cycle and p.manual_override is None:
                        scen, blend = _auto_phase(now - p.start_time)
                        p.scenario = scen

                    reading = generate_reading(p)

                    if p.connected:
                        success = self._publish(reading)
                        if success:
                            p.readings_sent += 1
                            logger.info("[%s] HR:%3d  SpO2:%4.1f%%  BP:%3d/%2d  RR:%2d  Glu:%3d  MAP:%4.1f  SI:%4.2f  -> %s",
                                        pid, reading["hr"], reading["spo2"], reading["sys"], reading["dia"],
                                        reading["rr"], reading["glucose"], reading["map"], reading["shock_index"], p.label)
                        else:
                            p.buffer.append(reading)
                    else:
                        p.buffer.append(reading)
                        logger.info("[%s] (BUFFERED - OFFLINE) HR:%3d  SpO2:%4.1f%%  Queue:%d",
                                    pid, reading["hr"], reading["spo2"], len(p.buffer))

                time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("Stopping simulator...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Ward Zero 10-Parameter Hospital Telemetry Simulator")
    parser.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS, help="Patient IDs")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Publish interval (seconds)")
    parser.add_argument("--fallback", action="store_true", help="Use HiveMQ public broker fallback")
    args = parser.parse_args()

    cfg = fallback_config() if args.fallback else load_config()
    sim = MultiPatientSimulator(config=cfg, patient_ids=args.patients, interval=args.interval)
    sim.run()


if __name__ == "__main__":
    main()
