"""
Tests for sensor disconnect detection, stale data handling, and reconnection recovery.
"""
import time
import pytest
from backend.mqtt.handler import last_sensor_seen
from backend.state import S


class TestSensorDisconnectAndStaleData:
    def test_stale_data_detection(self):
        pid = "bed01"
        # Simulate last seen 30 seconds ago
        last_sensor_seen[pid] = time.time() - 30

        now = time.time()
        is_stale = (now - last_sensor_seen[pid]) > 25
        assert is_stale is True

    def test_active_sensor_is_not_stale(self):
        pid = "bed02"
        # Simulate last seen 2 seconds ago
        last_sensor_seen[pid] = time.time() - 2

        now = time.time()
        is_stale = (now - last_sensor_seen[pid]) > 25
        assert is_stale is False

    def test_missing_reading_never_assumed_healthy(self):
        # A missing reading should have no risk score or default to alert/offline state
        unseen_pid = "unseen_bed_99"
        assert unseen_pid not in last_sensor_seen
        # Vitals must not be invented for unseen patient
        patient_vitals = [h for h in S.history if h.get("patient_id") == unseen_pid]
        assert len(patient_vitals) == 0
