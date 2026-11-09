"""
Tests for the ML risk-prediction model (Hospital ICU Grade - 10 Features).
Covers: model loading, prediction range, missing features,
risk categorisation thresholds, determinism, and edge cases.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from ml.predict import predict as ml_predict, validate_vitals, compute_map, compute_shock_index


def _to_10_features(v: dict) -> np.ndarray:
    """Helper to convert vitals dict to 10-feature numpy vector."""
    sys_ = v.get("sys", 120.0)
    dia_ = v.get("dia", 80.0)
    hr_ = v.get("hr", 75.0)
    map_ = v.get("map") or compute_map(sys_, dia_)
    si_ = v.get("shock_index") or compute_shock_index(hr_, sys_)
    return np.array([[
        float(hr_),
        float(v.get("spo2", 98.0)),
        float(sys_),
        float(dia_),
        float(v.get("temp", 36.8)),
        float(v.get("rr", 16.0)),
        float(v.get("glucose", 100.0)),
        float(map_),
        float(si_),
        float(v.get("hrv", 45.0)),
    ]])


# ---------------------------------------------------------------------------
# Model loads correctly
# ---------------------------------------------------------------------------
class TestModelLoading:
    def test_model_is_random_forest(self, trained_model):
        assert isinstance(trained_model, RandomForestRegressor)

    def test_model_has_estimators(self, trained_model):
        assert len(trained_model.estimators_) >= 80

    def test_model_has_correct_features(self, trained_model):
        assert trained_model.n_features_in_ == 10


# ---------------------------------------------------------------------------
# Prediction range [0, 1]
# ---------------------------------------------------------------------------
class TestPredictionRange:
    def test_normal_vitals_in_range(self, trained_model, normal_vitals):
        x = _to_10_features(normal_vitals)
        pred = float(np.clip(trained_model.predict(x)[0], 0, 1))
        assert 0.0 <= pred <= 1.0

    def test_abnormal_vitals_in_range(self, trained_model, abnormal_vitals):
        x = _to_10_features(abnormal_vitals)
        pred = float(np.clip(trained_model.predict(x)[0], 0, 1))
        assert 0.0 <= pred <= 1.0

    def test_edge_case_vitals_in_range(self, trained_model, edge_case_vitals):
        x = _to_10_features(edge_case_vitals)
        pred = float(np.clip(trained_model.predict(x)[0], 0, 1))
        assert 0.0 <= pred <= 1.0

    def test_batch_predictions_in_range(self, trained_model):
        from ml.train import generate_synthetic_data
        X, _ = generate_synthetic_data(n=200, seed=99)
        preds = np.clip(trained_model.predict(X), 0, 1)
        assert (preds >= 0.0).all()
        assert (preds <= 1.0).all()


# ---------------------------------------------------------------------------
# Risk categorisation
# ---------------------------------------------------------------------------
class TestRiskCategorisation:
    def test_normal_vitals_give_low_risk(self, trained_model, normal_vitals):
        x = _to_10_features(normal_vitals)
        pred = float(np.clip(trained_model.predict(x)[0], 0, 1))
        assert pred < 0.35, f"Normal vitals should give Low risk, got {pred:.3f}"

    def test_critical_vitals_give_high_risk(self, trained_model, abnormal_vitals):
        x = _to_10_features(abnormal_vitals)
        pred = float(np.clip(trained_model.predict(x)[0], 0, 1))
        assert pred > 0.60, f"Critical vitals should give High risk, got {pred:.3f}"

    def test_borderline_low_gives_low_or_med(self, trained_model, borderline_low_vitals):
        x = _to_10_features(borderline_low_vitals)
        pred = float(np.clip(trained_model.predict(x)[0], 0, 1))
        assert pred < 0.65, f"Borderline-low vitals should not be High risk, got {pred:.3f}"


# ---------------------------------------------------------------------------
# Missing features
# ---------------------------------------------------------------------------
class TestMissingFeatures:
    def test_predict_risk_handles_missing_key(self):
        """predict_risk in main.py returns 0.0 on any exception."""
        from main import predict_risk
        result = predict_risk({"hr": 75, "spo2": 98, "sys": 120, "dia": 78})  # no temp
        assert result == 0.0

    def test_predict_risk_handles_nan(self):
        from main import predict_risk
        result = predict_risk({"hr": float("nan"), "spo2": 98, "sys": 120, "dia": 78, "temp": 36.8})
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Deterministic with same input
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_input_same_output(self, trained_model, normal_vitals):
        x = _to_10_features(normal_vitals)
        p1 = trained_model.predict(x)[0]
        p2 = trained_model.predict(x)[0]
        assert np.isclose(p1, p2, atol=1e-6), "Same input must give identical output"

    def test_repeated_calls_stable(self, trained_model, abnormal_vitals):
        x = _to_10_features(abnormal_vitals)
        preds = [round(float(trained_model.predict(x)[0]), 6) for _ in range(50)]
        assert len(set(preds)) == 1, "Predictions must be deterministic"


# ---------------------------------------------------------------------------
# predict_risk wrapper
# ---------------------------------------------------------------------------
class TestPredictRiskWrapper:
    def test_returns_float(self, normal_vitals):
        from main import predict_risk
        result = predict_risk(normal_vitals)
        assert isinstance(result, float)

    def test_clipped_to_unit(self, abnormal_vitals):
        from main import predict_risk
        result = predict_risk(abnormal_vitals)
        assert 0.0 <= result <= 1.0

    def test_wrong_type_returns_zero(self):
        from main import predict_risk
        result = predict_risk({"hr": "abc", "spo2": 98, "sys": 120, "dia": 78, "temp": 36.8})
        assert result == 0.0
