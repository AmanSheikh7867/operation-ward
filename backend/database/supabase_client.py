"""
Operation Ward Zero – Supabase Client
Supabase initialisation, write-with-retry-buffer, flush loop,
and query helpers. Falls back to in-memory when Supabase is unavailable.
"""
import asyncio
import logging

from backend.config import SUPABASE_URL, SUPABASE_KEY, DB_TIMEOUT_S
from backend.state import S
from backend.ws_manager import manager, status_msg

log = logging.getLogger("wardzero.db")


# ── Helpers ───────────────────────────────────────────────────────────────────

def note_db_error(e: Exception):
    """Log the REAL database error once per distinct message (no log spam)."""
    msg = f"{type(e).__name__}: {e}"[:400]
    if msg != S.last_error:
        log.error("DB error -> %s", msg)
        S.last_error = msg


# ── Initialisation ────────────────────────────────────────────────────────────

def init_db():
    """Create the Supabase client. Safe to call at startup."""
    if not (SUPABASE_URL and SUPABASE_KEY):
        log.warning("Supabase not configured - running with in-memory buffer only")
        return
    if SUPABASE_KEY.startswith("sb_publishable"):
        log.warning(
            "SUPABASE_KEY is a PUBLISHABLE key: inserts will fail if RLS is on. "
            "Use the secret key (sb_secret_...) or the service_role key."
        )
    try:
        from supabase import create_client
        S.db_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        note_db_error(e)
        log.error("Supabase init failed (client not created): %s", e)


# ── Blocking DB Write (runs in thread) ────────────────────────────────────────

def _db_write(table: str, row: dict, op: str = "insert"):
    """Blocking call, executed in a worker thread."""
    if S.db_client is None:
        raise ConnectionError("no db client")
    if op == "insert":
        S.db_client.table(table).insert(row).execute()
    else:  # update alert by id
        S.db_client.table(table).update(row["set"]).eq("id", row["id"]).execute()


# ── Async Persist with Retry Buffer ──────────────────────────────────────────

async def persist(table: str, row: dict, op: str = "insert"):
    """Try the DB; on ANY failure park the row in the retry buffer."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_db_write, table, row, op), DB_TIMEOUT_S
        )
        S.last_error = ""
        if not S.db_connected:
            S.db_connected = True
            await manager.broadcast(status_msg())
    except Exception as e:
        note_db_error(e)
        S.retry_buffer.append((table, row, op))
        if S.db_connected:
            S.db_connected = False
        await manager.broadcast(status_msg())


# ── Flush Loop ────────────────────────────────────────────────────────────────

async def flush_loop():
    """Every 5s: replay buffered rows in order; stop at first failure."""
    while True:
        try:
            await asyncio.sleep(5)
            if S.db_client is None:
                continue

            if not S.retry_buffer:
                if not S.db_connected:  # heartbeat to detect recovery
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(
                                lambda: S.db_client.table("vitals")
                                .select("id")
                                .limit(1)
                                .execute()
                            ),
                            DB_TIMEOUT_S,
                        )
                        S.db_connected = True
                        await manager.broadcast(status_msg())
                    except Exception:
                        pass
                continue

            flushed = 0
            while S.retry_buffer:
                table, row, op = S.retry_buffer[0]
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(_db_write, table, row, op), DB_TIMEOUT_S
                    )
                    S.retry_buffer.popleft()
                    flushed += 1
                except Exception as e:
                    note_db_error(e)
                    break

            if flushed:
                log.info("Flushed %d buffered rows", flushed)
                S.db_connected = not S.retry_buffer
                await manager.broadcast(status_msg())
        except Exception as e:  # the loop must never die
            log.error("flush_loop error: %s", e)


# ── Query Helpers ─────────────────────────────────────────────────────────────

async def get_patient_history(patient_id: str, limit: int = 60) -> list:
    """Return recent vitals from in-memory history for a given patient."""
    return [
        v for v in list(S.history) if v.get("patient_id") == patient_id
    ][-limit:]


async def get_alerts(patient_id: str | None = None, limit: int = 30) -> list:
    """Return alerts from in-memory store, optionally filtered by patient."""
    alerts = sorted(S.alerts.values(), key=lambda a: a["ts"], reverse=True)
    if patient_id:
        alerts = [a for a in alerts if a.get("patient_id") == patient_id]
    return alerts[:limit]


async def get_caregiver_activities(limit: int = 50) -> list:
    """Return recent caregiver activities from in-memory store."""
    return list(S.caregiver_activities[-limit:])
