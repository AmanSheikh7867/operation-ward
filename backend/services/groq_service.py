"""
Operation Ward Zero – Groq AI Service
Provides AI caregiver advice and conversational triage assistance.
Strict medical guardrails: never diagnose, never prescribe medication dosages.
"""
import logging
from typing import Any, Dict, List, Optional

import httpx

from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.state import S

log = logging.getLogger("wardzero.groq")


def _groq_opts(max_out: int) -> dict:
    """Build request options for Groq API."""
    o = {
        "model": GROQ_MODEL,
        "temperature": 0.3,
        "max_completion_tokens": max_out + 700,
    }
    if "gpt-oss" in GROQ_MODEL:
        o["reasoning_effort"] = "low"
    return o


def fallback_advice(reasons: List[Dict[str, Any]]) -> str:
    """Deterministic fallback action items based on clinical triage guidelines."""
    codes = {r.get("code") for r in reasons}
    tips = []
    if codes & {"spo2_low", "spo2_crit"}:
        tips.append("Help the patient sit upright, check pulse oximeter probe placement, and guide slow deep breaths.")
    if codes & {"tachy", "brady", "bp_high", "bp_low"}:
        tips.append("Keep the patient calm and resting; re-check pulse and blood pressure in 5 minutes.")
    if codes & {"fever", "hypo"}:
        tips.append("Adjust blankets and room temperature and offer fluids if the patient is awake.")
    tips.append("If symptoms persist or worsen, contact the on-call physician or emergency services immediately.")
    return "\n".join(f"- {t}" for t in tips[:2])


async def groq_advice(v: Dict[str, Any], risk: float, reasons: List[Dict[str, Any]]) -> str:
    """Generate concise, actionable caregiver advice using Groq LLM."""
    if not GROQ_API_KEY:
        return fallback_advice(reasons)

    prompt = (
        f"Home healthcare patient vital signs: Heart Rate {v.get('hr')} bpm, SpO2 {v.get('spo2')}%, "
        f"Blood Pressure {v.get('sys')}/{v.get('dia')} mmHg, Temperature {v.get('temp')} C. "
        f"Experimental AI risk score: {risk:.2f}. Identified flags: {[r.get('code') for r in reasons]}. "
        "Provide EXACTLY 2 concise bullet points (each starting with '- ') of safe, non-diagnostic caregiver advice. "
        "RULES: Do NOT diagnose diseases. Do NOT recommend medication or dosage adjustments. "
        "Keep language calm, clear, and reassuring."
    )

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    **_groq_opts(250),
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are Ward Zero AI, an assistant for home caregivers. "
                                "Never diagnose medical conditions. Never recommend drug doses. "
                                "Focus strictly on patient comfort, positioning, and when to seek emergency help."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return text or fallback_advice(reasons)
    except Exception as e:
        log.warning("Groq advice API call failed: %s; using rule-based fallback", e)
        return fallback_advice(reasons)


def _vitals_text(v: Optional[Dict[str, Any]]) -> str:
    if not v:
        return "No sensor data received yet."
    return (
        f"HR {v.get('hr', 0):.0f} bpm, SpO2 {v.get('spo2', 0)}%, "
        f"BP {v.get('sys', 0):.0f}/{v.get('dia', 0):.0f} mmHg, "
        f"Temp {v.get('temp', 0)} C, Risk score {v.get('risk', 0):.2f}"
    )


def fallback_chat(v: Optional[Dict[str, Any]], reasons: List[Dict[str, Any]], lang: str = "en") -> str:
    """Fallback responses in English, Hindi, and Hinglish when LLM is unavailable."""
    if not v:
        if lang == "hi":
            return "अभी तक कोई सेंसर डेटा प्राप्त नहीं हुआ है। कृपया जांचें कि सेंसर चालू है।"
        elif lang == "hinglish":
            return "Abhi tak koi sensor data receive nahi hua. Check karein ki sensor simulator chal raha hai."
        return "I have not received any sensor data yet. Please verify that the simulator or sensor device is active."

    if lang == "hi":
        lines = [f"वर्तमान वाइटल्स: {_vitals_text(v)}।"]
        if reasons:
            lines.append("अलर्ट्स: " + ", ".join(r.get("code", "").replace("_", " ") for r in reasons) + "।")
            lines.append(fallback_advice(reasons))
        else:
            lines.append("सभी वाइटल्स सुरक्षित सीमा में हैं। निगरानी जारी रखें।")
        return "\n".join(lines)
    elif lang == "hinglish":
        lines = [f"Current vitals: {_vitals_text(v)}."]
        if reasons:
            lines.append("Flags: " + ", ".join(r.get("code", "").replace("_", " ") for r in reasons) + ".")
            lines.append(fallback_advice(reasons))
        else:
            lines.append("Saare vitals safe range mein hain. Monitoring continue karein.")
        return "\n".join(lines)

    lines = [f"Current readings: {_vitals_text(v)}."]
    if reasons:
        lines.append("Active flags: " + ", ".join(r.get("code", "").replace("_", " ") for r in reasons) + ".")
        lines.append(fallback_advice(reasons))
    else:
        lines.append("All vitals are within the safe range. Continue standard monitoring.")
    return "\n".join(lines)


async def chat_reply(text: str, history: List[Dict[str, str]], lang: str = "en") -> str:
    """Generate multi-turn conversational response with current patient context."""
    v = S.history[-1] if S.history else None
    reasons = v.get("reasons", []) if v else []

    if not GROQ_API_KEY:
        return fallback_chat(v, reasons, lang) + "\n\n[Offline Mode: AI key not configured, showing deterministic response]"

    langname = {
        "hi": "Hindi (written in pure Devanagari script)",
        "hinglish": "Hinglish (Hindi written in Latin/English alphabet)",
    }.get(lang, "English")

    system = (
        "You are Ward Zero AI, a calm, knowledgeable assistant for a family caregiver. "
        "SAFETY MANDATE: "
        "1. NEVER diagnose diseases or conditions. "
        "2. NEVER recommend medication names or dosages. "
        "3. Keep answers concise (max 80 words). "
        "4. If vitals appear critical, instruct caregiver to summon medical emergency services immediately. "
        f"5. Reply in {langname}. "
        f"Context - Live patient data: {_vitals_text(v)}. "
        f"Identified flags: {[r.get('code') for r in reasons]}. "
        f"Pending unacknowledged alerts: {sum(1 for a in S.alerts.values() if not a.get('acknowledged'))}."
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    **_groq_opts(350),
                    "messages": (
                        [{"role": "system", "content": system}]
                        + history[-8:]
                        + [{"role": "user", "content": text}]
                    ),
                },
            )
            r.raise_for_status()
            out = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return out or fallback_chat(v, reasons, lang)
    except Exception as e:
        log.warning("Groq chat failed: %s; falling back to deterministic response", e)
        return fallback_chat(v, reasons, lang)
