"""
Operation Ward Zero – Shared Application State
Global mutable state container for cross-module access.
"""
import asyncio
from collections import deque

from backend.config import BUFFER_MAX


class State:
    """Centralised mutable state shared across all backend modules."""

    loop: asyncio.AbstractEventLoop = None

    # connectivity flags
    mqtt_connected: bool = False
    db_connected: bool = False

    # Supabase client (set by database module)
    db_client = None

    # last DB error text (shown in logs + /health)
    last_error: str = ""

    # recent vitals ring-buffer for new WS clients
    history: deque = deque(maxlen=120)

    # id -> alert dict (in-memory view)
    alerts: dict = {}

    # patient_id -> epoch (alert cooldown tracking)
    last_alert_ts: dict = {}

    # (table, row, op) waiting for the DB
    retry_buffer: deque = deque(maxlen=BUFFER_MAX)

    # patient registry (in-memory)
    patients: dict = {}

    # caregiver activity log
    caregiver_activities: list = []


S = State()
