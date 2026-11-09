"""
Operation Ward Zero - Explainability (10 Telemetry Features)
============================================================

Provides three complementary explanation systems:

1. **SHAP-based (Model-level XAI)**: Uses TreeExplainer across all 10 vital parameters
   to compute exact marginal feature attributions for risk scores.
2. **Clinical Rule-based (Deterministic)**: Evaluates normal vs. pathological boundaries
   for each parameter based on standard clinical guidelines.
3. **NHS NEWS2 Clinical Composite**: Breakdowns of points assigned per physiological parameter.

Surfaced independently so caregivers and evaluators have total transparency.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # type: ignore

from ml.predict import (
    FEATURE_BOUNDS,
    FEATURE_NAMES,
    calculate_news2,
    compute_map,
    compute_shock_index,
    detect_clinical_syndrome,
    validate_vitals,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = _THIS_DIR / "artifacts" / "risk_model.joblib"
STATS_PATH = _THIS_DIR / "artifacts" / "feature_stats.joblib"

SAFE_RANGES = {
    "hr":          (55, 100),
    "spo2":        (95, 100),
    "sys":         (95, 140),
    "dia":         (60, 90),
    "temp":        (36.0, 37.8),
    "rr":          (12, 20),
    "glucose":     (70, 140),
    "map":         (70, 105),
    "shock_index": (0.50, 0.70),
    "hrv":         (25, 75),
}

VITAL_LABELS = {
    "hr":          "Heart Rate (bpm)",
    "spo2":        "Blood Oxygen (%)",
    "sys":         "Systolic BP (mmHg)",
    "dia":         "Diastolic BP (mmHg)",
    "temp":        "Temperature (°C)",
    "rr":          "Respiratory Rate (bpm)",
    "glucose":     "Blood Glucose (mg/dL)",
    "map":         "Mean Arterial Pressure (mmHg)",
    "shock_index": "Shock Index (HR/SYS)",
    "hrv":         "HR Variability RMSSD (ms)",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run ml/train.py first.")
    _model = joblib.load(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# 1. SHAP-based explanations
# ---------------------------------------------------------------------------

def shap_explain(
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
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute SHAP values across all 10 features."""
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

    try:
        import shap  # type: ignore
    except ImportError:
        return {
            "shap_available": False,
            "error": "The 'shap' package is not installed. Install with: pip install shap",
        }

    model = _load_model()
    explainer = shap.TreeExplainer(model)

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

    sv = explainer.shap_values(X)
    shap_vals = np.ravel(sv)

    exp_val = explainer.expected_value
    base_val = float(np.ravel(exp_val)[0])

    features = []
    for name, val, sval in zip(FEATURE_NAMES, X[0], shap_vals):
        features.append({
            "feature": name,
            "label": VITAL_LABELS.get(name, name),
            "value": round(float(val), 2),
            "shap_value": round(float(sval), 4),
            "direction": "increases_risk" if sval > 0 else "decreases_risk",
        })

    # Sort by absolute impact
    features_sorted = sorted(features, key=lambda f: abs(f["shap_value"]), reverse=True)

    predicted_risk = float(np.clip(base_val + float(np.sum(shap_vals)), 0.0, 1.0))

    return {
        "shap_available": True,
        "base_value": round(base_val, 4),
        "predicted_risk": round(predicted_risk, 4),
        "top_features": features_sorted[:top_k],
        "all_features": features_sorted,
    }


# ---------------------------------------------------------------------------
# 2. Rule-based clinical explanation
# ---------------------------------------------------------------------------

def rule_explain(
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
) -> Dict[str, Any]:
    """Deterministic rule-based clinical evaluation."""
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

    findings = []

    # HR
    if v["hr"] > 100:
        findings.append({
            "vital": "hr",
            "status": "abnormal",
            "code": "tachycardia",
            "message": f"Elevated heart rate ({v['hr']:.0f} bpm > 100 bpm threshold).",
            "severity": "critical" if v["hr"] > 130 else "warning",
        })
    elif v["hr"] < 55:
        findings.append({
            "vital": "hr",
            "status": "abnormal",
            "code": "bradycardia",
            "message": f"Low heart rate ({v['hr']:.0f} bpm < 55 bpm threshold).",
            "severity": "critical" if v["hr"] < 40 else "warning",
        })

    # SpO2
    if v["spo2"] < 90:
        findings.append({
            "vital": "spo2",
            "status": "abnormal",
            "code": "severe_hypoxemia",
            "message": f"Critical oxygen desaturation ({v['spo2']:.1f}% < 90%).",
            "severity": "critical",
        })
    elif v["spo2"] < 95:
        findings.append({
            "vital": "spo2",
            "status": "abnormal",
            "code": "mild_hypoxemia",
            "message": f"Sub-optimal oxygen saturation ({v['spo2']:.1f}% < 95%).",
            "severity": "warning",
        })

    # Blood Pressure
    if v["sys"] > 140:
        findings.append({
            "vital": "sys",
            "status": "abnormal",
            "code": "hypertension",
            "message": f"Elevated blood pressure ({v['sys']:.0f}/{v['dia']:.0f} mmHg).",
            "severity": "critical" if v["sys"] > 180 else "warning",
        })
    elif v["sys"] < 95:
        findings.append({
            "vital": "sys",
            "status": "abnormal",
            "code": "hypotension",
            "message": f"Low blood pressure ({v['sys']:.0f}/{v['dia']:.0f} mmHg).",
            "severity": "critical" if v["sys"] < 85 else "warning",
        })

    # Temperature
    if v["temp"] > 37.8:
        findings.append({
            "vital": "temp",
            "status": "abnormal",
            "code": "fever",
            "message": f"Fever / Pyrexia ({v['temp']:.1f}°C > 37.8°C).",
            "severity": "critical" if v["temp"] > 39.0 else "warning",
        })
    elif v["temp"] < 36.0:
        findings.append({
            "vital": "temp",
            "status": "abnormal",
            "code": "hypothermia",
            "message": f"Hypothermia ({v['temp']:.1f}°C < 36.0°C).",
            "severity": "critical" if v["temp"] < 35.0 else "warning",
        })

    # Respiratory Rate (RR)
    if v["rr"] > 20:
        findings.append({
            "vital": "rr",
            "status": "abnormal",
            "code": "tachypnea",
            "message": f"Elevated respiratory rate ({v['rr']:.0f} bpm > 20 bpm). Sensitive indicator of clinical decline.",
            "severity": "critical" if v["rr"] >= 25 else "warning",
        })
    elif v["rr"] < 12:
        findings.append({
            "vital": "rr",
            "status": "abnormal",
            "code": "bradypnea",
            "message": f"Depressed respiration ({v['rr']:.0f} bpm < 12 bpm). Risk of hypoventilation/narcotic effect.",
            "severity": "critical" if v["rr"] <= 8 else "warning",
        })

    # Glucose
    if v["glucose"] < 70:
        findings.append({
            "vital": "glucose",
            "status": "abnormal",
            "code": "hypoglycemia",
            "message": f"Hypoglycemic episode ({v['glucose']:.0f} mg/dL < 70 mg/dL). Immediate sugar source required.",
            "severity": "critical" if v["glucose"] < 55 else "warning",
        })
    elif v["glucose"] > 180:
        findings.append({
            "vital": "glucose",
            "status": "abnormal",
            "code": "hyperglycemia",
            "message": f"Elevated blood sugar ({v['glucose']:.0f} mg/dL > 180 mg/dL).",
            "severity": "warning",
        })

    # Mean Arterial Pressure (MAP)
    if v["map"] < 65:
        findings.append({
            "vital": "map",
            "status": "abnormal",
            "code": "inadequate_perfusion",
            "message": f"Critically low MAP ({v['map']:.1f} mmHg < 65 mmHg target). Inadequate kidney/brain perfusion.",
            "severity": "critical",
        })

    # Shock Index
    if v["shock_index"] > 0.85:
        findings.append({
            "vital": "shock_index",
            "status": "abnormal",
            "code": "occult_shock",
            "message": f"Elevated Shock Index ({v['shock_index']:.2f} > 0.85). Indicates impending circulatory/septic collapse.",
            "severity": "critical",
        })

    # HRV
    if v["hrv"] < 20:
        findings.append({
            "vital": "hrv",
            "status": "abnormal",
            "code": "low_autonomic_reserve",
            "message": f"Depressed Heart Rate Variability ({v['hrv']:.0f} ms). Sign of systemic inflammation and sympathetic burnout.",
            "severity": "warning",
        })

    has_abnormal = len(findings) > 0
    return {
        "status": "abnormal" if has_abnormal else "normal",
        "findings_count": len(findings),
        "findings": findings,
        "summary": (
            f"{len(findings)} vital sign(s) outside safe physiological boundaries."
            if has_abnormal
            else "All 10 monitored parameters within safe physiological bounds."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Comprehensive combined explainability
# ---------------------------------------------------------------------------

def explain(
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
    top_k: int = 5,
) -> Dict[str, Any]:
    """Provide complete explainability: SHAP + Clinical Rules + NEWS2 + Syndromes."""
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

    shap_res = shap_explain(
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
        top_k=top_k,
    )

    rule_res = rule_explain(
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

    news2_res = calculate_news2(
        hr=v["hr"],
        spo2=v["spo2"],
        sys=v["sys"],
        temp=v["temp"],
        rr=v["rr"],
    )

    syndrome_res = detect_clinical_syndrome(
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

    return {
        "vitals": v,
        "model_explanation": shap_res,
        "rule_explanation": rule_res,
        "news2_composite": news2_res,
        "clinical_syndrome": syndrome_res["primary"],
        "active_syndromes": syndrome_res["all_syndromes"],
        "note": (
            "SHAP explains feature contributions in the trained Random Forest model. "
            "NEWS2 provides the standardized NHS early warning benchmark. "
            "Rule-based findings highlight boundary violations."
        ),
    }


if __name__ == "__main__":
    import json
    print("Normal Patient Explanation:")
    print(json.dumps(explain(74, 98, 120, 80, 36.8, rr=16, glucose=92, hrv=50), indent=2))
    print("\nDecompensating Patient Explanation:")
    print(json.dumps(explain(130, 89, 86, 52, 39.1, rr=28, glucose=52, hrv=12), indent=2))
