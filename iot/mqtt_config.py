"""
Operation Ward Zero – MQTT broker configuration.

Loads all connection parameters from environment variables with sensible
fallback defaults.  Primary target: local Mosquitto broker (localhost:1883).
Fallback / demo target: HiveMQ public broker.

Environment variables
---------------------
MQTT_BROKER        Broker hostname            (default: localhost)
MQTT_PORT          Broker port                (default: 1883)
MQTT_USERNAME      Optional auth username
MQTT_PASSWORD      Optional auth password
MQTT_KEEPALIVE     Keep-alive interval in sec (default: 30)
MQTT_QOS           Default QoS level 0/1/2    (default: 1)
MQTT_TOPIC_PREFIX  Topic root segment         (default: wardzero)
MQTT_USE_TLS       Set "1" to enable TLS      (default: 0)
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass, field
from typing import Optional


# ── Fallback brokers ─────────────────────────────────────────────────────────
FALLBACK_BROKER = "broker.hivemq.com"
FALLBACK_PORT = 1883


def _env(name: str, default: str = "") -> str:
    """Read an environment variable, stripping whitespace."""
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int = 0) -> int:
    raw = _env(name)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class MQTTConfig:
    """Immutable MQTT connection configuration."""

    broker: str = field(default_factory=lambda: _env("MQTT_BROKER", "localhost"))
    port: int = field(default_factory=lambda: _env_int("MQTT_PORT", 1883))
    username: Optional[str] = field(
        default_factory=lambda: _env("MQTT_USERNAME") or None
    )
    password: Optional[str] = field(
        default_factory=lambda: _env("MQTT_PASSWORD") or None
    )
    keepalive: int = field(default_factory=lambda: _env_int("MQTT_KEEPALIVE", 30))
    qos: int = field(default_factory=lambda: _env_int("MQTT_QOS", 1))
    topic_prefix: str = field(
        default_factory=lambda: _env("MQTT_TOPIC_PREFIX", "wardzero")
    )
    use_tls: bool = field(
        default_factory=lambda: _env("MQTT_USE_TLS", "0") == "1"
    )
    # Reconnect back-off
    reconnect_min_delay: int = 1
    reconnect_max_delay: int = 30

    # ── Derived helpers ──────────────────────────────────────────────────
    def vitals_topic(self, patient_id: str) -> str:
        """Return the topic for a given patient, e.g. wardzero/bed01/vitals."""
        return f"{self.topic_prefix}/{patient_id}/vitals"

    @property
    def has_credentials(self) -> bool:
        return bool(self.username and self.password)

    def summary(self) -> str:
        tls_flag = " (TLS)" if self.use_tls else ""
        auth_flag = " (auth)" if self.has_credentials else ""
        return (
            f"mqtt://{self.broker}:{self.port}{tls_flag}{auth_flag}  "
            f"QoS={self.qos}  keepalive={self.keepalive}s  "
            f"prefix={self.topic_prefix}"
        )


def load_config() -> MQTTConfig:
    """Build an MQTTConfig from the current environment."""
    return MQTTConfig()


def fallback_config() -> MQTTConfig:
    """Return a config pointing at the public HiveMQ broker (demo only)."""
    return MQTTConfig(
        broker=FALLBACK_BROKER,
        port=FALLBACK_PORT,
        keepalive=30,
        qos=1,
        topic_prefix="wardzero",
        use_tls=False,
    )
