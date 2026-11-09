"""
Operation Ward Zero - Backend
Sense -> Understand -> Predict -> Alert -> Respond

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000
"""
from backend.main import app
from backend.config import (
    ALERT_COOLDOWN_S,
    RISK_THRESHOLD,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    DB_TIMEOUT_S,
    BUFFER_MAX,
    SUPABASE_URL,
    SUPABASE_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from backend.state import State, S
from backend.services.risk_engine import predict_risk, explain
from backend.services.alert_service import create_alert, acknowledge_alert
from backend.services.groq_service import groq_advice, chat_reply, fallback_advice, fallback_chat
from backend.database.supabase_client import persist, flush_loop
from backend.ws_manager import manager, status_msg
from ml.train import train_model

MODEL = train_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
