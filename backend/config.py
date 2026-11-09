"""
Operation Ward Zero – Configuration
Load settings from ward.env using python-dotenv. Never hardcode secrets.
"""
import os
import logging

log = logging.getLogger("wardzero.config")

# Load ward.env from the project root (one level up from backend/)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ward.env")
    load_dotenv(os.path.normpath(_env_path))
except ImportError:
    log.warning("python-dotenv not installed – relying on environment variables only")

# ── MQTT ──────────────────────────────────────────────────────────────────────
MQTT_BROKER: str = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT: int = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC: str = "wardzero/+/vitals"

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

# ── Groq AI ───────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ── Thresholds & Limits ──────────────────────────────────────────────────────
RISK_THRESHOLD: float = float(os.getenv("RISK_THRESHOLD", "0.6"))
ALERT_COOLDOWN_S: int = int(os.getenv("ALERT_COOLDOWN_S", "20"))
DB_TIMEOUT_S: int = int(os.getenv("DB_TIMEOUT_S", "5"))
BUFFER_MAX: int = int(os.getenv("BUFFER_MAX", "5000"))

# ── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT: str = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
FRONTEND_DIR: str = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_INDEX: str = os.path.join(FRONTEND_DIR, "index.html")

# Fallback: if frontend/index.html doesn't exist, try root index.html
if not os.path.isfile(FRONTEND_INDEX):
    _root_index = os.path.join(PROJECT_ROOT, "index.html")
    if os.path.isfile(_root_index):
        FRONTEND_INDEX = _root_index
