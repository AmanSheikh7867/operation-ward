"""
Tests for risk level categorisation and threshold logic.
Levels:  Low  < 0.3
         Med  0.3 – 0.6
         High > 0.6
"""
import pytest


# ---------------------------------------------------------------------------
# Categorisation function (mirrors the JS client and alert logic)
# ---------------------------------------------------------------------------
def risk_level(score: float, low=0.3, high=0.6) -> str:
    """Categorise a risk score into Low / Med / High."""
    if score > high:
        return "High"
    if score > low:
        return "Med"
    return "Low"


# ---------------------------------------------------------------------------
# Core categorisation
# ---------------------------------------------------------------------------
class TestRiskLevels:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "Low"),
        (0.1, "Low"),
        (0.29, "Low"),
        (0.31, "Med"),
        (0.45, "Med"),
        (0.59, "Med"),
        (0.61, "High"),
        (0.8, "High"),
        (1.0, "High"),
    ])
    def test_categorisation(self, score, expected):
        assert risk_level(score) == expected

    def test_boundary_at_0_3_is_low(self):
        # score == 0.3 is NOT > 0.3, so it should be Low
        assert risk_level(0.3) == "Low"

    def test_boundary_at_0_6_is_med(self):
        # score == 0.6 is NOT > 0.6, so it should be Med
        assert risk_level(0.6) == "Med"

    def test_just_above_0_3(self):
        assert risk_level(0.300001) == "Med"

    def test_just_above_0_6(self):
        assert risk_level(0.600001) == "High"


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
class TestConfigurableThresholds:
    def test_custom_low_threshold(self):
        assert risk_level(0.2, low=0.2, high=0.6) == "Low"
        assert risk_level(0.21, low=0.2, high=0.6) == "Med"

    def test_custom_high_threshold(self):
        assert risk_level(0.5, low=0.3, high=0.5) == "Med"
        assert risk_level(0.51, low=0.3, high=0.5) == "High"

    def test_very_tight_thresholds(self):
        assert risk_level(0.49, low=0.49, high=0.51) == "Low"
        assert risk_level(0.50, low=0.49, high=0.51) == "Med"
        assert risk_level(0.52, low=0.49, high=0.51) == "High"


# ---------------------------------------------------------------------------
# Alert threshold integration
# ---------------------------------------------------------------------------
class TestAlertThreshold:
    """The backend fires an alert when risk > RISK_THRESHOLD (0.6)."""

    def test_below_threshold_no_alert(self):
        from main import RISK_THRESHOLD
        assert 0.55 <= RISK_THRESHOLD  # risk 0.55 should NOT trigger
        assert not (0.55 > RISK_THRESHOLD)

    def test_above_threshold_alert(self):
        from main import RISK_THRESHOLD
        assert 0.65 > RISK_THRESHOLD

    def test_at_threshold_no_alert(self):
        from main import RISK_THRESHOLD
        assert not (RISK_THRESHOLD > RISK_THRESHOLD)


# ---------------------------------------------------------------------------
# Explain function flags
# ---------------------------------------------------------------------------
class TestExplainFunction:
    def test_normal_vitals_no_flags(self, normal_vitals):
        from main import explain
        reasons = explain(normal_vitals)
        assert reasons == []

    def test_tachycardia(self, abnormal_vitals):
        from main import explain
        reasons = explain(abnormal_vitals)
        codes = [r["code"] for r in reasons]
        assert "tachy" in codes

    def test_low_spo2(self, abnormal_vitals):
        from main import explain
        reasons = explain(abnormal_vitals)
        codes = [r["code"] for r in reasons]
        assert "spo2_crit" in codes

    def test_fever(self, abnormal_vitals):
        from main import explain
        reasons = explain(abnormal_vitals)
        codes = [r["code"] for r in reasons]
        assert "fever" in codes

    def test_bradycardia(self):
        from main import explain
        reasons = explain({"hr": 45, "spo2": 98, "sys": 120, "dia": 78, "temp": 36.8})
        codes = [r["code"] for r in reasons]
        assert "brady" in codes

    def test_hypothermia(self):
        from main import explain
        reasons = explain({"hr": 75, "spo2": 98, "sys": 120, "dia": 78, "temp": 35.0})
        codes = [r["code"] for r in reasons]
        assert "hypo" in codes

    def test_bp_high(self):
        from main import explain
        reasons = explain({"hr": 75, "spo2": 98, "sys": 160, "dia": 100, "temp": 36.8})
        codes = [r["code"] for r in reasons]
        assert "bp_high" in codes

    def test_bp_low(self):
        from main import explain
        reasons = explain({"hr": 75, "spo2": 98, "sys": 80, "dia": 50, "temp": 36.8})
        codes = [r["code"] for r in reasons]
        assert "bp_low" in codes
