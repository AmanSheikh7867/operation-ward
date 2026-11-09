"""
Tests for MQTT message validation, topic parsing, and ingestion pipeline.
"""
import json
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from backend.mqtt.handler import validate_payload, process_reading
from backend.state import S


class TestMQTTValidation:
    def test_valid_reading(self):
        data = {"hr": 76, "spo2": 98.0, "sys": 120, "dia": 78, "temp": 36.8}
        valid, err = validate_payload(data)
        assert valid is True
        assert err is None

    def test_missing_field_fails(self):
        data = {"hr": 76, "spo2": 98.0, "sys": 120, "dia": 78}  # missing temp
        valid, err = validate_payload(data)
        assert valid is False
        assert "Missing required field: temp" in err

    def test_out_of_bounds_fails(self):
        data = {"hr": 350, "spo2": 98.0, "sys": 120, "dia": 78, "temp": 36.8}  # HR 350 > 220
        valid, err = validate_payload(data)
        assert valid is False
        assert "out of plausible bounds" in err

    def test_non_numeric_fails(self):
        data = {"hr": "not_a_number", "spo2": 98.0, "sys": 120, "dia": 78, "temp": 36.8}
        valid, err = validate_payload(data)
        assert valid is False
        assert "Non-numeric" in err


@pytest.mark.asyncio
class TestMQTTPipeline:
    async def test_process_valid_topic_and_payload(self):
        topic = "wardzero/bed02/vitals"
        payload = json.dumps({
            "hr": 80,
            "spo2": 97.5,
            "sys": 122,
            "dia": 79,
            "temp": 36.7,
        }).encode("utf-8")

        initial_len = len(S.history)
        await process_reading(topic, payload)

        assert len(S.history) == initial_len + 1
        latest = S.history[-1]
        assert latest["patient_id"] == "bed02"
        assert latest["hr"] == 80.0
        assert "risk" in latest
        assert "reasons" in latest

    async def test_malformed_json_dropped(self):
        topic = "wardzero/bed01/vitals"
        payload = b"not-a-json-payload"
        initial_len = len(S.history)
        await process_reading(topic, payload)
        assert len(S.history) == initial_len  # nothing added

    async def test_critical_reading_triggers_alert(self):
        topic = "wardzero/bed03/vitals"
        # Force critical vitals
        payload = json.dumps({
            "hr": 145,
            "spo2": 82.0,
            "sys": 185,
            "dia": 115,
            "temp": 39.4,
        }).encode("utf-8")

        with patch("backend.mqtt.handler.create_alert") as mock_alert:
            await process_reading(topic, payload)
            # Should call create_alert for high risk
            assert mock_alert.called
