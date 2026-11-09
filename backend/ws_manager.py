"""
Operation Ward Zero – WebSocket Manager
Manages connected WebSocket clients and broadcasting.
"""
import asyncio
import json
import logging

from fastapi import WebSocket

from backend.state import S

log = logging.getLogger("wardzero.ws")


class Manager:
    """Track connected WebSocket clients and broadcast messages."""

    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def drop(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await asyncio.wait_for(ws.send_text(json.dumps(msg)), timeout=3)
            except Exception:
                dead.append(ws)  # disconnected / slow client
        for ws in dead:
            self.drop(ws)


manager = Manager()


def status_msg() -> dict:
    """Build a status message dict reflecting current system state."""
    return {
        "type": "status",
        "mqtt": S.mqtt_connected,
        "db": S.db_connected,
        "buffer": len(S.retry_buffer),
        "db_error": S.last_error,
    }
