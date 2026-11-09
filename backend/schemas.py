"""
Operation Ward Zero - Pydantic Schemas (Hospital ICU Grade)
===========================================================
Request/response models with validation for 10-parameter telemetry, patients, alerts, chat, etc.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Vital Reading (10 Hospital Telemetry Features) ───────────────────────────
class VitalReading(BaseModel):
    # Core Primary Vitals
    hr: float = Field(..., ge=20, le=250, description="Heart rate (bpm)")
    spo2: float = Field(..., ge=40, le=100, description="Blood oxygen saturation (%)")
    sys: float = Field(..., ge=40, le=300, description="Systolic BP (mmHg)")
    dia: float = Field(..., ge=20, le=200, description="Diastolic BP (mmHg)")
    temp: float = Field(..., ge=30, le=45, description="Body temperature (°C)")

    # Advanced ICU & Clinical Telemetry Parameters
    rr: Optional[float] = Field(default=16.0, ge=4, le=70, description="Respiratory rate (breaths/min)")
    glucose: Optional[float] = Field(default=100.0, ge=15, le=600, description="Capillary blood glucose (mg/dL)")
    map: Optional[float] = Field(default=None, ge=25, le=220, description="Mean Arterial Pressure (mmHg)")
    shock_index: Optional[float] = Field(default=None, ge=0.1, le=3.0, description="Shock Index (HR / SYS)")
    hrv: Optional[float] = Field(default=45.0, ge=2, le=200, description="Heart Rate Variability RMSSD (ms)")


# ── Patient ───────────────────────────────────────────────────────────────────
class Patient(BaseModel):
    id: str
    name: str
    age: int = Field(..., ge=0, le=150)
    blood_type: str = ""
    caregiver: str = ""
    bed_number: str = ""
    primary_diagnosis: str = ""


# ── Alert ─────────────────────────────────────────────────────────────────────
class Alert(BaseModel):
    id: str
    patient_id: str
    risk: float = Field(..., ge=0, le=1)
    reasons: list[dict[str, Any]] = []
    advice: str = ""
    syndrome: Optional[dict[str, Any]] = None
    news2_score: Optional[int] = 0
    acknowledged: bool = False
    ts: str = ""
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    alert_type: str = "vital_risk"


# ── Caregiver Activity ────────────────────────────────────────────────────────
class CaregiverActivity(BaseModel):
    id: str
    caregiver_id: str
    action: str
    alert_id: Optional[str] = None
    patient_id: Optional[str] = None
    ts: str = ""


# ── System Status ─────────────────────────────────────────────────────────────
class SystemStatus(BaseModel):
    mqtt: bool = False
    db: bool = False
    buffer: int = 0
    db_error: str = ""
    model_version: str = "2.0-Hospital-10F"


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    lang: str = Field(default="en", pattern=r"^(en|hi|hinglish)$")
    patient_id: Optional[str] = "bed01"


class ChatResponse(BaseModel):
    text: str
    clinical_syndrome: Optional[str] = None
