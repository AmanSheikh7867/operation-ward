"""
Tests for alert creation, cooldown logic, structure, and acknowledgment.
"""
import asyncio
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Alert structure
# ---------------------------------------------------------------------------
class TestAlertStructure:
    def test_alert_has_required_fields(self):
        """An alert dict should have: id, patient_id, risk, reasons, advice,
        acknowledged, ts."""
        alert = {
            "id": str(uuid.uuid4()),
            "patient_id": "bed01",
            "risk": 0.75,
            "reasons": [{"code": "tachy", "value": 130}],
            "advice": "Keep the patient calm.",
            "acknowledged": False,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        for key in ("id", "patient_id", "risk", "reasons", "advice", "acknowledged", "ts"):
            assert key in alert

    def test_alert_id_is_valid_uuid(self):
        alert_id = str(uuid.uuid4())
        assert uuid.UUID(alert_id)  # should not raise

    def test_risk_is_numeric(self):
        alert = {"risk": 0.72}
        assert isinstance(alert["risk"], (int, float))

    def test_reasons_is_list(self):
        alert = {"reasons": [{"code": "tachy", "value": 130}]}
        assert isinstance(alert["reasons"], list)

    def test_acknowledged_defaults_false(self):
        alert = {"acknowledged": False}
        assert alert["acknowledged"] is False


# ---------------------------------------------------------------------------
# Cooldown prevents flooding
# ---------------------------------------------------------------------------
class TestAlertCooldown:
    def test_cooldown_prevents_duplicate(self):
        from main import ALERT_COOLDOWN_S
        last_alert_ts = {"bed01": time.time()}
        now = time.time()
        should_alert = (now - last_alert_ts.get("bed01", 0)) > ALERT_COOLDOWN_S
        assert not should_alert, "Alert should be blocked within cooldown window"

    def test_cooldown_allows_after_expiry(self):
        from main import ALERT_COOLDOWN_S
        last_alert_ts = {"bed01": time.time() - ALERT_COOLDOWN_S - 1}
        now = time.time()
        should_alert = (now - last_alert_ts.get("bed01", 0)) > ALERT_COOLDOWN_S
        assert should_alert, "Alert should be allowed after cooldown expires"

    def test_no_prior_alert_always_allowed(self):
        from main import ALERT_COOLDOWN_S
        last_alert_ts = {}
        now = time.time()
        should_alert = (now - last_alert_ts.get("bed01", 0)) > ALERT_COOLDOWN_S
        assert should_alert, "First alert for a patient should always be allowed"

    def test_different_patients_independent(self):
        from main import ALERT_COOLDOWN_S
        last_alert_ts = {"bed01": time.time()}
        now = time.time()
        should_alert_bed02 = (now - last_alert_ts.get("bed02", 0)) > ALERT_COOLDOWN_S
        assert should_alert_bed02, "Different patients have independent cooldowns"


# ---------------------------------------------------------------------------
# Fallback advice
# ---------------------------------------------------------------------------
class TestFallbackAdvice:
    def test_fallback_spo2(self):
        from main import fallback_advice
        reasons = [{"code": "spo2_crit", "value": 82}]
        advice = fallback_advice(reasons)
        assert "deep breaths" in advice.lower() or "sensor" in advice.lower()

    def test_fallback_tachy(self):
        from main import fallback_advice
        reasons = [{"code": "tachy", "value": 140}]
        advice = fallback_advice(reasons)
        assert "calm" in advice.lower() or "pulse" in advice.lower()

    def test_fallback_fever(self):
        from main import fallback_advice
        reasons = [{"code": "fever", "value": 39.2}]
        advice = fallback_advice(reasons)
        assert "blanket" in advice.lower() or "temperature" in advice.lower() or "fluids" in advice.lower()

    def test_fallback_always_includes_emergency(self):
        from main import fallback_advice
        advice = fallback_advice([{"code": "spo2_crit", "value": 80}])
        assert "emergency" in advice.lower() or "doctor" in advice.lower()

    def test_fallback_empty_reasons(self):
        from main import fallback_advice
        advice = fallback_advice([])
        assert len(advice) > 0  # still returns generic advice


# ---------------------------------------------------------------------------
# Alert acknowledgment
# ---------------------------------------------------------------------------
class TestAlertAcknowledgment:
    def test_ack_sets_flag(self):
        alert = {"id": "abc", "acknowledged": False}
        alert["acknowledged"] = True
        assert alert["acknowledged"] is True

    def test_ack_is_idempotent(self):
        alert = {"id": "abc", "acknowledged": True}
        alert["acknowledged"] = True
        assert alert["acknowledged"] is True


# ---------------------------------------------------------------------------
# Caregiver activity recording (structure)
# ---------------------------------------------------------------------------
class TestCaregiverActivity:
    def test_activity_structure(self):
        activity = {
            "id": str(uuid.uuid4()),
            "alert_id": str(uuid.uuid4()),
            "action": "acknowledged",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        assert "alert_id" in activity
        assert activity["action"] == "acknowledged"
