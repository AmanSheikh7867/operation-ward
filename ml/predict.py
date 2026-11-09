"""
Operation Ward Zero - Inference / Prediction (Hospital ICU Grade)
==================================================================

Loads the saved multi-vital model and provides validated prediction.
Features (10): hr, spo2, sys, dia, temp, rr, glucose, map, shock_index, hrv

Includes:
- Dual Risk Scoring: Machine Learning Random Forest Score + NHS NEWS2 Score
- Automatic derivation of Hemodynamic Metrics (MAP, Shock Index)
- Clinical Syndrome Detection (Sepsis / qSOFA, Hypoglycemia, Respiratory Distress)
- Full backward compatibility when called with 5 base parameters

IMPORTANT:
    The output is an *experimental* risk score. It does NOT replace certified
    clinical telemetry or medical judgment.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # type: ignore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = _THIS_DIR / "artifacts" / "risk_model.joblib"
STATS_PATH = _THIS_DIR / "artifacts" / "feature_stats.joblib"

# ---------------------------------------------------------------------------
# Feature definitions & safe bounds
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {"low": 0.30, "high": 0.65}

FEATURE_NAMES = [
    "hr",
    "spo2",
    "sys",
    "dia",
    "temp",
    "rr",
    "glucose",
    "map",
    "shock_index",
    "hrv",
]

FEATURE_BOUNDS = {
    "hr":          (30, 220),
    "spo2":        (50, 100),
    "sys":         (50, 250),
    "dia":         (30, 160),
    "temp":        (33.0, 43.0),
    "rr":          (6, 60),
    "glucose":     (20, 500),
    "map":         (30, 180),
    "shock_index": (0.2, 2.5),
    "hrv":         (5, 150),
}


class ValidationError(Exception):
    """Raised when input vitals fail validation."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when model artifacts are missing."""


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------
_model = None
_stats = None


def _load_model():
    global _model, _stats
    if _model is not None:
        return _model, _stats
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run ml/train.py first."
        )
    _model = joblib.load(MODEL_PATH)
    if STATS_PATH.exists():
        _stats = joblib.load(STATS_PATH)
    return _model, _stats


def reload_model():
    """Force-reload the model from disk (useful after retraining)."""
    global _model, _stats
    _model = None
    _stats = None
    return _load_model()


# ---------------------------------------------------------------------------
# Physiological derivations & Clinical Scoring
# ---------------------------------------------------------------------------

def compute_map(sys: float, dia: float) -> float:
    """Mean Arterial Pressure: (2*dia + sys) / 3."""
    return round((2.0 * float(dia) + float(sys)) / 3.0, 1)


def compute_shock_index(hr: float, sys: float) -> float:
    """Shock Index: HR / Systolic BP."""
    return round(float(hr) / max(float(sys), 1.0), 2)


def calculate_news2(
    hr: float,
    spo2: float,
    sys: float,
    temp: float,
    rr: float,
    is_cvpu: bool = False,
) -> Dict[str, Any]:
    """Calculate the official NHS National Early Warning Score 2 (NEWS2).

    Returns a breakdown of component points and total score (0-20).
    """
    pts = {}

    # 1. Respiration Rate (breaths/min)
    if rr <= 8 or rr >= 25:
        pts["rr"] = 3
    elif 21 <= rr <= 24:
        pts["rr"] = 2
    elif 9 <= rr <= 11:
        pts["rr"] = 1
    else:
        pts["rr"] = 0

    # 2. Oxygen Saturation (Scale 1 - Air)
    if spo2 <= 91:
        pts["spo2"] = 3
    elif 92 <= spo2 <= 93:
        pts["spo2"] = 2
    elif 94 <= spo2 <= 95:
        pts["spo2"] = 1
    else:
        pts["spo2"] = 0

    # 3. Systolic BP (mmHg)
    if sys <= 90 or sys >= 220:
        pts["sys"] = 3
    elif 91 <= sys <= 100:
        pts["sys"] = 2
    elif 101 <= sys <= 110:
        pts["sys"] = 1
    else:
        pts["sys"] = 0

    # 4. Heart Rate (bpm)
    if hr <= 40 or hr >= 131:
        pts["hr"] = 3
    elif 111 <= hr <= 130:
        pts["hr"] = 2
    elif (41 <= hr <= 50) or (91 <= hr <= 110):
        pts["hr"] = 1
    else:
        pts["hr"] = 0

    # 5. Temperature (°C)
    if temp <= 35.0:
        pts["temp"] = 3
    elif temp >= 39.1:
        pts["temp"] = 2
    elif (35.1 <= temp <= 36.0) or (38.1 <= temp <= 39.0):
        pts["temp"] = 1
    else:
        pts["temp"] = 0

    # 6. Consciousness (AVPU)
    pts["consciousness"] = 3 if is_cvpu else 0

    total = sum(pts.values())

    if total >= 7:
        tier = "High"
        action = "Emergency clinical response team / Immediate ICU outreach"
    elif total >= 5 or any(v == 3 for v in pts.values()):
        tier = "Medium"
        action = "Urgent clinical review by ward doctor and vital sign monitoring"
    else:
        tier = "Low"
        action = "Standard ward clinical routine monitoring"

    return {
        "score": total,
        "tier": tier,
        "action": action,
        "components": pts,
    }


def detect_clinical_syndrome(
    hr: float,
    spo2: float,
    sys: float,
    dia: float,
    temp: float,
    rr: float,
    glucose: float,
    map_: float,
    shock_index: float,
    hrv: float,
) -> Dict[str, Any]:
    """Detect specific clinical syndromes to guide caregiver triage."""
    syndromes = []

    # Sepsis / qSOFA trigger: RR >= 22, MAP < 65 or HR > 100, Temp abnormal, Shock Index high
    if (rr >= 22 or shock_index >= 0.85) and (temp >= 38.2 or temp <= 36.0):
        syndromes.append({
            "code": "SEPSIS_RISK",
            "name": "Suspected Sepsis / SIRS (qSOFA Trigger)",
            "severity": "CRITICAL",
            "protocol": "Implement Sepsis 3-Hour Bundle: IV fluid access, check lactate, notify physician.",
        })

    # Hypoglycemia: Glucose < 60 mg/dL
    if glucose < 60:
        syndromes.append({
            "code": "HYPOGLYCEMIA",
            "name": "Acute Hypoglycemic Episode (<60 mg/dL)",
            "severity": "CRITICAL",
            "protocol": "Administer 15-20g rapid-acting oral glucose or IV Dextrose per protocol. Re-check in 15 min.",
        })
    elif glucose > 240:
        syndromes.append({
            "code": "HYPERGLYCEMIA",
            "name": "Severe Hyperglycemia (>240 mg/dL)",
            "severity": "WARNING",
            "protocol": "Check for urinary/blood ketones and verify insulin coverage.",
        })

    # Respiratory distress: SpO2 < 90% or RR > 26
    if spo2 < 90 or rr > 26:
        syndromes.append({
            "code": "RESPIRATORY_DISTRESS",
            "name": "Acute Respiratory Compromise",
            "severity": "CRITICAL",
            "protocol": "Elevate bed head to 45 degrees, apply high-flow supplemental oxygen, assess airway patency.",
        })

    # Hypertensive emergency: SYS > 180 or DIA > 110
    if sys > 180 or dia > 110:
        syndromes.append({
            "code": "HYPERTENSIVE_CRISIS",
            "name": "Hypertensive Urgency / Crisis",
            "severity": "HIGH",
            "protocol": "Re-check blood pressure manually. Check for chest pain, headache, or visual disturbance.",
        })

    # Circulatory shock: MAP < 65 and Shock Index > 0.9
    if map_ < 65 and shock_index > 0.90:
        syndromes.append({
            "code": "HEMODYNAMIC_SHOCK",
            "name": "Hemodynamic Circulatory Collapse (MAP < 65)",
            "severity": "CRITICAL",
            "protocol": "Place patient supine, elevate legs, establish large-bore IV access, alert intensivist.",
        })

    primary = syndromes[0] if syndromes else {
        "code": "STABLE",
        "name": "Stable Physiological Homeostasis",
        "severity": "NORMAL",
        "protocol": "Continue scheduled vital sign observation.",
    }

    return {
        "primary": primary,
        "all_syndromes": syndromes,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_vitals(
    hr: float,
    spo2: float,
    sys: float,
    dia: float,
    temp: float,
    rr: Optional[float] = None,
    glucose: Optional[float] = None,
    map_: Optional[float] = None,
    shock_index: Optional[float] = None,
    hrv: Optional[float] = None,
) -> Dict[str, float]:
    """Validate vitals against physiological ranges, auto-filling defaults/derived."""
    # Autoderive or default if missing
    sys_f = float(sys)
    dia_f = float(dia)
    hr_f = float(hr)

    map_val = compute_map(sys_f, dia_f) if map_ is None else float(map_)
    si_val = compute_shock_index(hr_f, sys_f) if shock_index is None else float(shock_index)
    rr_val = 16.0 if rr is None else float(rr)
    glucose_val = 100.0 if glucose is None else float(glucose)
    hrv_val = 45.0 if hrv is None else float(hrv)

    v = {
        "hr": hr_f,
        "spo2": float(spo2),
        "sys": sys_f,
        "dia": dia_f,
        "temp": float(temp),
        "rr": rr_val,
        "glucose": glucose_val,
        "map": map_val,
        "shock_index": si_val,
        "hrv": hrv_val,
    }

    errors = []
    for k, val in v.items():
        lo, hi = FEATURE_BOUNDS[k]
        if not (lo <= val <= hi):
            errors.append(f"{k}={val} out of plausible bounds [{lo}, {hi}]")

    if errors:
        raise ValidationError("; ".join(errors))

    return v


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def categorise_risk(score: float, thresholds: Optional[dict] = None) -> str:
    """Return 'Low', 'Medium', or 'High' for a risk score."""
    t = thresholds or DEFAULT_THRESHOLDS
    if score < t["low"]:
        return "Low"
    elif score <= t["high"]:
        return "Medium"
    return "High"


def predict(
    hr: float,
    spo2: float,
    sys: float,
    dia: float,
    temp: float,
    rr: Optional[float] = None,
    glucose: Optional[float] = None,
    map_: Optional[float] = None,
    shock_index: Optional[float] = None,
    hrv: Optional[float] = None,
    *,
    thresholds: Optional[dict] = None,
) -> Dict[str, Any]:
    """Predict experimental risk score and clinical metrics from 10 vitals."""
    # 1. Validate & complete features
    v = validate_vitals(
        hr=hr,
        spo2=spo2,
        sys=sys,
        dia=dia,
        temp=temp,
        rr=rr,
        glucose=glucose,
        map_=map_,
        shock_index=shock_index,
        hrv=hrv,
    )

    # 2. Load model
    model, _ = _load_model()

    # 3. Model inference (RandomForestRegressor over 10 features)
    X = np.array([[
        v["hr"],
        v["spo2"],
        v["sys"],
        v["dia"],
        v["temp"],
        v["rr"],
        v["glucose"],
        v["map"],
        v["shock_index"],
        v["hrv"],
    ]])
    raw_score = float(model.predict(X)[0])
    score = float(np.clip(raw_score, 0.0, 1.0))

    # 4. NEWS2 Clinical Composite Score
    news2 = calculate_news2(
        hr=v["hr"],
        spo2=v["spo2"],
        sys=v["sys"],
        temp=v["temp"],
        rr=v["rr"],
    )

    # 5. Syndrome detection
    syndromes = detect_clinical_syndrome(
        hr=v["hr"],
        spo2=v["spo2"],
        sys=v["sys"],
        dia=v["dia"],
        temp=v["temp"],
        rr=v["rr"],
        glucose=v["glucose"],
        map_=v["map"],
        shock_index=v["shock_index"],
        hrv=v["hrv"],
    )

    # 6. Categorise
    category = categorise_risk(score, thresholds)

    return {
        "risk_score": round(score, 4),
        "risk_category": category,
        "vitals_validated": v,
        "news2": news2,
        "clinical_syndrome": syndromes["primary"],
        "active_syndromes": syndromes["all_syndromes"],
        "disclaimer": (
            "Hospital-grade multi-parameter experimental risk score. "
            "Combines Random Forest inference with NHS NEWS2 clinical benchmark. "
            "NOT validated as a sole diagnostic instrument."
        ),
    }


def predict_batch(vitals_list: List[dict], *, thresholds: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Predict for multiple patients."""
    results = []
    for v in vitals_list:
        try:
            res = predict(
                hr=v.get("hr", 75),
                spo2=v.get("spo2", 98),
                sys=v.get("sys", 120),
                dia=v.get("dia", 80),
                temp=v.get("temp", 36.8),
                rr=v.get("rr"),
                glucose=v.get("glucose"),
                map_=v.get("map"),
                shock_index=v.get("shock_index"),
                hrv=v.get("hrv"),
                thresholds=thresholds,
            )
        except (ValidationError, KeyError) as exc:
            res = {"error": str(exc)}
        results.append(res)
    return results


if __name__ == "__main__":
    import json
    print("Normal ICU Vitals:")
    print(json.dumps(predict(75, 98, 120, 80, 36.8, rr=16, glucose=95, hrv=52), indent=2))
    print("\nSeptic Shock Telemetry:")
    print(json.dumps(predict(125, 91, 88, 52, 39.2, rr=28, glucose=180, hrv=14), indent=2))
