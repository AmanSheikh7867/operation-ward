# Model Card: Operation Ward Zero Experimental Risk Scorer (Hospital ICU Grade)

## 1. Model Details

- **Model Name**: WardZero Hospital-Grade Multi-Parameter Risk Regressor
- **Architecture**: `RandomForestRegressor` (scikit-learn ensemble)
- **Hyperparameters**: `n_estimators=100`, `max_depth=14`, `random_state=42`, `n_jobs=-1`
- **Features Monitored (10)**:
  1. `hr` (Heart Rate, bpm)
  2. `spo2` (Blood Oxygen Saturation, %)
  3. `sys` (Systolic Blood Pressure, mmHg)
  4. `dia` (Diastolic Blood Pressure, mmHg)
  5. `temp` (Core Body Temperature, °C)
  6. `rr` (Respiratory Rate, breaths/min - #1 Hospital Early Warning Marker)
  7. `glucose` (Capillary Blood Glucose, mg/dL)
  8. `map` (Mean Arterial Pressure, mmHg)
  9. `shock_index` (Shock Index: HR / SYS)
  10. `hrv` (Heart Rate Variability RMSSD, ms)
- **Clinical Benchmarking**: Dual scoring with NHS National Early Warning Score 2 (NEWS2) running deterministically in parallel.
- **Clinical Syndromic Triage**: Automated detection of Sepsis / qSOFA triggers, Hypoglycemic crises, Acute Respiratory Distress, and Hemodynamic Shock.
- **Artifacts**: `ml/artifacts/risk_model.joblib`, `ml/artifacts/feature_stats.joblib`
- **Explainability**: SHAP (`TreeExplainer`) marginal attribution across all 10 features + Clinical Rule Heuristic decomposition.

---

## 2. Intended Use

- **Primary Purpose**: Multi-parameter telemetric early warning risk assessment, explainable AI decomposition, and caregiver emergency protocol recommendation.
- **Target Users**: Hackathon judges, biomedical engineers, telehealth system architects, caregiving teams.
- **Deployment Context**: Edge telehealth hubs, smart home healthcare setups, hackathon demonstration environments.

---

## 3. Explicit Safety & Regulatory Boundary

> [!CAUTION]
> **HACKATHON EXPERIMENTAL PROTOTYPE - NOT A CERTIFIED MEDICAL DEVICE.**
> - Output must always be presented as an **experimental risk score** alongside standardized deterministic clinical scoring (NEWS2).
> - Never used as an autonomous diagnostic device or for prescriptive pharmacological interventions.
> - Always verified by trained medical personnel.

---

## 4. Evaluated Input Features & Safe Ranges

| Parameter | Clinical Description | Bounds | Normal ICU Safe Range | Key Clinical Significance |
|---|---|---|---|---|
| `hr` | Heart Rate | [30, 220] bpm | 60 - 100 bpm | Tachycardia / Arrhythmia detection |
| `spo2` | Blood Oxygen Saturation | [50, 100] % | 95 - 100 % | Hypoxemia / Respiratory failure |
| `sys` | Systolic Blood Pressure | [50, 250] mmHg | 95 - 140 mmHg | Hypertension / Hypotension |
| `dia` | Diastolic Blood Pressure | [30, 160] mmHg | 60 - 90 mmHg | Vascular resistance |
| `temp` | Core Body Temperature | [33.0, 43.0] °C | 36.0 - 37.8 °C | SIRS / Fever / Hypothermia |
| `rr` | Respiratory Rate | [6, 60] bpm | 12 - 20 bpm | **#1 Sensitive Predictor of Clinical Deterioration** |
| `glucose` | Capillary Blood Sugar | [20, 500] mg/dL | 70 - 140 mg/dL | Hypoglycemic shock (<60) / DKA |
| `map` | Mean Arterial Pressure | [30, 180] mmHg | 70 - 105 mmHg | **Organ Perfusion Target (Target ≥ 65 mmHg)** |
| `shock_index` | Shock Index (HR / SYS) | [0.2, 2.5] | 0.50 - 0.70 | **Occult Hemorrhagic & Septic Shock Early Warning** |
| `hrv` | HRV (RMSSD) | [5, 150] ms | 25 - 75 ms | Autonomic reserve & systemic inflammation |

---

## 5. Model Performance Metrics (Held-Out Test Set)

Evaluated on 1,600 held-out clinical validation vectors:

- **Coefficient of Determination ($R^2$)**: `0.9623`
- **Mean Absolute Error (MAE)**: `0.0410`
- **Root Mean Squared Error (RMSE)**: `0.0621`
- **Risk Stratification Accuracy**: `91.0%`
- **Macro Average F1**: `0.88`
- **Weighted Average F1**: `0.91`

### Feature Importances (Random Forest 100 Estimators)
1. `shock_index` : `0.4195` (Hemodynamic collapse warning)
2. `glucose`     : `0.2368` (Acute metabolic emergency)
3. `rr`          : `0.1252` (Respiratory failure trigger)
4. `spo2`        : `0.0644` (Oxygenation status)
5. `sys`         : `0.0557` (Perfusion pressure)
6. `hrv`         : `0.0510` (Autonomic stress reserve)
7. `hr`          : `0.0230` (Rate chronotropy)
8. `temp`        : `0.0163` (Thermoregulatory control)
9. `map`         : `0.0057` (Mean pressure)
10. `dia`        : `0.0023` (Diastolic resistance)
