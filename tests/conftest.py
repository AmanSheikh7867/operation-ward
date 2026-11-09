"""
Shared pytest fixtures for Operation Ward Zero.
Provides mock MQTT clients, mock Supabase, sample vital readings,
sample patients, and a FastAPI test client.
"""
import asyncio
import json
import sys
import os
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Sample vital readings
# ---------------------------------------------------------------------------
@pytest.fixture
def normal_vitals() -> dict:
    """Vital signs well within safe ranges."""
    return {"hr": 75, "spo2": 98, "sys": 120, "dia": 78, "temp": 36.8}


@pytest.fixture
def abnormal_vitals() -> dict:
    """Vital signs that should trigger a high risk score."""
    return {"hr": 140, "spo2": 85, "sys": 175, "dia": 110, "temp": 39.2}


@pytest.fixture
def borderline_low_vitals() -> dict:
    """Vitals exactly at the lower boundary of normal."""
    return {"hr": 55, "spo2": 95, "sys": 95, "dia": 60, "temp": 36.0}


@pytest.fixture
def borderline_high_vitals() -> dict:
    """Vitals exactly at the upper boundary of normal."""
    return {"hr": 100, "spo2": 100, "sys": 140, "dia": 90, "temp": 37.8}


@pytest.fixture
def edge_case_vitals() -> dict:
    """Extreme edge-case vitals at physiological limits."""
    return {"hr": 35, "spo2": 70, "sys": 60, "dia": 35, "temp": 34.0}


@pytest.fixture
def missing_field_vitals() -> dict:
    """Vitals dict with a required field missing."""
    return {"hr": 75, "spo2": 98, "sys": 120, "dia": 78}  # missing 'temp'


@pytest.fixture
def wrong_type_vitals() -> dict:
    """Vitals with string values instead of numeric."""
    return {"hr": "fast", "spo2": "ok", "sys": "high", "dia": "low", "temp": "warm"}


@pytest.fixture
def nan_vitals() -> dict:
    """Vitals containing NaN."""
    return {"hr": float("nan"), "spo2": 98, "sys": 120, "dia": 78, "temp": 36.8}


@pytest.fixture
def inf_vitals() -> dict:
    """Vitals containing infinity."""
    return {"hr": float("inf"), "spo2": 98, "sys": 120, "dia": 78, "temp": 36.8}


# ---------------------------------------------------------------------------
# Sample patient data
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_patient() -> dict:
    return {
        "id": "bed01",
        "name": "John Doe",
        "age": 58,
        "blood": "O+",
        "caregiver": "Nurse Sarah",
    }


# ---------------------------------------------------------------------------
# Trained ML model (singleton – reuse across tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def trained_model():
    """Return the actual trained model from main.py."""
    from main import train_model
    return train_model()


# ---------------------------------------------------------------------------
# Mock MQTT client
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_mqtt_client():
    client = MagicMock()
    client.connect_async = MagicMock()
    client.loop_start = MagicMock()
    client.loop_stop = MagicMock()
    client.disconnect = MagicMock()
    client.subscribe = MagicMock()
    client.publish = MagicMock(return_value=MagicMock(rc=0))
    client.on_connect = None
    client.on_disconnect = None
    client.on_message = None
    return client


# ---------------------------------------------------------------------------
# Mock Supabase client
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_supabase():
    client = MagicMock()
    table_mock = MagicMock()
    table_mock.insert.return_value.execute = MagicMock()
    table_mock.update.return_value.eq.return_value.execute = MagicMock()
    table_mock.select.return_value.limit.return_value.execute = MagicMock()
    table_mock.delete.return_value.eq.return_value.execute = MagicMock()
    client.table.return_value = table_mock
    return client


# ---------------------------------------------------------------------------
# FastAPI test client (uses httpx async)
# ---------------------------------------------------------------------------
@pytest.fixture
def app():
    """Import and return the FastAPI app without running lifespan."""
    from main import app as _app
    return _app


@pytest_asyncio.fixture
async def async_client(app):
    """Provide an async httpx client for endpoint testing."""
    from httpx import ASGITransport, AsyncClient
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Fresh State object for isolated tests
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_state():
    """Return a clean State-like namespace with empty buffers."""
    from main import State
    s = State()
    s.loop = asyncio.new_event_loop()
    s.history = deque(maxlen=120)
    s.alerts = {}
    s.last_alert_ts = {}
    s.retry_buffer = deque(maxlen=5000)
    s.mqtt_connected = False
    s.db_connected = False
    s.db_client = None
    s.last_error = ""
    return s


# ---------------------------------------------------------------------------
# Helper: build a raw MQTT message object
# ---------------------------------------------------------------------------
@pytest.fixture
def make_mqtt_msg():
    """Factory fixture that creates a mock MQTT message."""
    def _make(topic: str, payload: dict | str | bytes):
        msg = MagicMock()
        msg.topic = topic
        if isinstance(payload, dict):
            msg.payload = json.dumps(payload).encode()
        elif isinstance(payload, str):
            msg.payload = payload.encode()
        else:
            msg.payload = payload
        return msg
    return _make
