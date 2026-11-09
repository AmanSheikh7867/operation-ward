"""
Operation Ward Zero - Risk Engine Service (Hospital ICU Grade)
==============================================================
Integrates ML model inference (10 features), SHAP-based feature importance,
deterministic NHS NEWS2 scoring, and clinical syndrome triage.
"""
import logging
from typing import Any, Dict, List, Optional

from ml.predict import (
    predict as ml_predict,
    categorise_risk,
    validate_vitals,
    calculate_news2,
    detect_clinical_syndrome,
    compute_map,
    compute_shock_index,
    ValidationError,
)
from ml.explain import explain as ml_explain, rule_explain

log = logging.getLogger("wardzero.risk")


def predict_risk(v: dict) -> float:
    """Predict risk score in [0, 1] from vital signs dict."""
    try:
        res = ml_predict(
            hr=v["hr"],
            spo2=v["spo2"],
            sys=v["sys"],
            dia=v["dia"],
            temp=v["temp"],
            rr=v.get("rr"),
            glucose=v.get("glucose"),
            map_=v.get("map"),
            shock_index=v.get("shock_index"),
            hrv=v.get("hrv"),
        )
        return float(res["risk_score"])
    except ValidationError as ve:
        log.warning("Validation failed for vitals: %s", ve)
        return 0.0
    except Exception as e:
        log.warning("Predict failed: %s", e)
        return 0.0


def explain(v: dict) -> List[Dict[str, Any]]:
    """Machine-readable reason codes compatible with existing UI and i18n."""
    r = []
    hr = float(v.get("hr", 0))
    spo2 = float(v.get("spo2", 0))
    sys_ = float(v.get("sys", 0))
    dia = float(v.get("dia", 0))
    temp = float(v.get("temp", 0))
    rr = float(v.get("rr", 16))
    glucose = float(v.get("glucose", 100))
    map_ = float(v.get("map") or compute_map(sys_, dia))
    shock_index = float(v.get("shock_index") or compute_shock_index(hr, sys_))
    hrv = float(v.get("hrv", 45))

    # Heart Rate
    if hr > 100:
        r.append({"code": "tachy", "value": round(hr), "name": "Tachycardia"})
    elif hr < 55:
        r.append({"code": "brady", "value": round(hr), "name": "Bradycardia"})

    # SpO2
    if spo2 < 90:
        r.append({"code": "spo2_crit", "value": round(spo2, 1), "name": "Critical Hypoxemia"})
    elif spo2 < 95:
        r.append({"code": "spo2_low", "value": round(spo2, 1), "name": "Low Oxygen"})

    # Blood Pressure
    if sys_ > 140:
        r.append({"code": "bp_high", "value": f"{round(sys_)}/{round(dia)}", "name": "Hypertension"})
    elif sys_ < 95:
        r.append({"code": "bp_low", "value": f"{round(sys_)}/{round(dia)}", "name": "Hypotension"})

    # Temperature
    if temp > 37.8:
        r.append({"code": "fever", "value": round(temp, 1), "name": "Pyrexia / Fever"})
    elif temp < 36.0:
        r.append({"code": "hypo", "value": round(temp, 1), "name": "Hypothermia"})

    # Respiratory Rate (Crucial ICU Early Warning)
    if rr > 20:
        r.append({"code": "rr_high", "value": round(rr), "name": "Tachypnea (>20 bpm)"})
    elif rr < 12:
        r.append({"code": "rr_low", "value": round(rr), "name": "Bradypnea (<12 bpm)"})

    # Blood Glucose
    if glucose < 70:
        r.append({"code": "glucose_low", "value": round(glucose), "name": "Hypoglycemia (<70 mg/dL)"})
    elif glucose > 180:
        r.append({"code": "glucose_high", "value": round(glucose), "name": "Hyperglycemia (>180 mg/dL)"})

    # Mean Arterial Pressure
    if map_ < 65:
        r.append({"code": "map_low", "value": round(map_, 1), "name": "Low MAP (<65 mmHg)"})

    # Shock Index
    if shock_index > 0.85:
        r.append({"code": "shock_high", "value": round(shock_index, 2), "name": "Elevated Shock Index (>0.85)"})

    # HRV
    if hrv < 20:
        r.append({"code": "hrv_low", "value": round(hrv), "name": "Depressed HRV (<20 ms)"})

    return r


def evaluate_clinical_metrics(v: dict) -> Dict[str, Any]:
    """Calculate NEWS2 and detect clinical syndromes."""
    hr = float(v.get("hr", 75))
    spo2 = float(v.get("spo2", 98))
    sys_ = float(v.get("sys", 120))
    dia = float(v.get("dia", 80))
    temp = float(v.get("temp", 36.8))
    rr = float(v.get("rr", 16))
    glucose = float(v.get("glucose", 100))
    map_ = float(v.get("map") or compute_map(sys_, dia))
    shock_index = float(v.get("shock_index") or compute_shock_index(hr, sys_))
    hrv = float(v.get("hrv", 45))

    news2 = calculate_news2(hr, spo2, sys_, temp, rr)
    syndromes = detect_clinical_syndrome(hr, spo2, sys_, dia, temp, rr, glucose, map_, shock_index, hrv)

    return {
        "news2": news2,
        "primary_syndrome": syndromes["primary"],
        "all_syndromes": syndromes["all_syndromes"],
        "map": map_,
        "shock_index": shock_index,
    }


def explain_full(v: dict) -> Dict[str, Any]:
    """Provide comprehensive explainability combining SHAP, clinical rules, and NEWS2."""
    try:
        return ml_explain(
            hr=v["hr"],
            spo2=v["spo2"],
            sys=v["sys"],
            dia=v["dia"],
            temp=v["temp"],
            rr=v.get("rr"),
            glucose=v.get("glucose"),
            map_=v.get("map"),
            shock_index=v.get("shock_index"),
            hrv=v.get("hrv"),
        )
    except Exception as e:
        log.warning("Full explanation failed: %s", e)
        return {
            "model_explanation": {"shap_available": False, "error": str(e)},
            "rule_explanation": rule_explain(
                hr=v.get("hr", 75),
                spo2=v.get("spo2", 98),
                sys=v.get("sys", 120),
                dia=v.get("dia", 78),
                temp=v.get("temp", 36.8),
                rr=v.get("rr", 16),
                glucose=v.get("glucose", 100),
            ),
            "news2_composite": calculate_news2(
                hr=v.get("hr", 75),
                spo2=v.get("spo2", 98),
                sys=v.get("sys", 120),
                temp=v.get("temp", 36.8),
                rr=v.get("rr", 16),
            ),
            "note": "Fallback to rule-based and NEWS2 explanation.",
        }
