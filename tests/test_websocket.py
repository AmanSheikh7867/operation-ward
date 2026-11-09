"""
Tests for WebSocket communication, broadcasting, and message routing.
"""
import asyncio
import json
import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.state import S
from backend.ws_manager import manager


class TestWebSocketCommunication:
    def test_websocket_connect_and_snapshot(self):
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # 1. First message is snapshot
            snapshot_msg = ws.receive_json()
            assert snapshot_msg["type"] == "snapshot"
            assert "history" in snapshot_msg
            assert "alerts" in snapshot_msg
            assert "patients" in snapshot_msg

            # 2. Second message is status
            status_msg = ws.receive_json()
            assert status_msg["type"] == "status"
            assert "mqtt" in status_msg
            assert "db" in status_msg

    def test_websocket_ping_pong(self):
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # Flush initial messages
            ws.receive_json()
            ws.receive_json()

            # Send ping
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_websocket_alert_ack(self):
        client = TestClient(app)
        # Seed an unacknowledged alert
        alert_id = "test-alert-uuid-1234"
        S.alerts[alert_id] = {
            "id": alert_id,
            "patient_id": "bed01",
            "risk": 0.85,
            "reasons": [{"code": "tachy", "value": 140}],
            "advice": "Rest quietly.",
            "acknowledged": False,
            "ts": "2026-11-07T12:00:00Z",
        }

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # snapshot
            ws.receive_json()  # status

            # Send ack
            ws.send_json({"type": "ack", "id": alert_id, "caregiver": "Nurse Sarah"})

            # Wait for ack broadcast
            broadcast_ack = ws.receive_json()
            assert broadcast_ack["type"] == "ack"
            assert broadcast_ack["id"] == alert_id
            assert S.alerts[alert_id]["acknowledged"] is True
            assert S.alerts[alert_id]["acknowledged_by"] == "Nurse Sarah"

    def test_websocket_chat(self):
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # snapshot
            ws.receive_json()  # status

            # Send chat message
            ws.send_json({"type": "chat", "text": "Is the patient stable?", "lang": "en"})
            chat_resp = ws.receive_json()
            assert chat_resp["type"] == "chat"
            assert "text" in chat_resp
            assert len(chat_resp["text"]) > 0
