"""
Ward Zero - Supabase connection checker.
Keys are read from ward.env (same folder) or from terminal environment variables:
    python check_db.py
It creates the client, inserts one test row into `vitals`, deletes it,
and tells you exactly what is wrong if anything fails.
"""
import os
import sys

# Load keys from ward.env (next to this file) so no per-terminal setup is needed.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ward.env"))
except ImportError:
    pass

url = os.getenv("SUPABASE_URL", "").strip()
key = os.getenv("SUPABASE_KEY", "").strip()

def fail(title, hint):
    print(f"\n[FAIL] {title}\n  -> {hint}")
    sys.exit(1)

print("1) Environment variables")
if not url: fail("SUPABASE_URL is empty", "Fill SUPABASE_URL in ward.env (copy ward.env.example) and run: pip install python-dotenv")
if not key: fail("SUPABASE_KEY is empty", "Fill SUPABASE_KEY in ward.env (copy ward.env.example) and run: pip install python-dotenv")
if not url.startswith("https://") or ".supabase.co" not in url:
    fail("SUPABASE_URL looks wrong", "It must look like https://xxxx.supabase.co (no /rest/v1 at the end).")
print(f"   URL ok: {url}")
kind = "publishable (public)" if key.startswith("sb_publishable") else "secret" if key.startswith("sb_secret") else "JWT (anon or service_role)"
print(f"   Key type: {kind}")
if key.startswith("sb_publishable"):
    print("   [WARN] Publishable keys cannot insert when RLS is on. Use the secret key.")

print("2) Creating client")
try:
    from supabase import create_client
    client = create_client(url, key)
except ImportError:
    fail("supabase package not installed", "Run: pip install -U supabase")
except Exception as e:
    fail(f"create_client failed: {type(e).__name__}: {e}",
         "Old supabase versions reject the new sb_ keys. Run: pip install -U supabase")

print("3) Insert test row into vitals")
try:
    res = client.table("vitals").insert({"patient_id": "dbcheck", "hr": 70, "spo2": 98, "sys": 120,
                                         "dia": 80, "temp": 36.8, "risk": 0.0}).execute()
    print("   insert ok")
except Exception as e:
    msg = str(e)
    print(f"\n[FAIL] {type(e).__name__}: {msg[:400]}")
    low = msg.lower()
    if "row-level security" in low or "42501" in low:
        print("  -> RLS is blocking the insert. Use the secret key (sb_secret_...), OR run:\n"
              "     alter table vitals disable row level security;\n     alter table alerts disable row level security;")
    elif "pgrst205" in low or "could not find the table" in low or "does not exist" in low:
        print("  -> Table not found. Run the SQL schema from the top of main.py in the Supabase SQL editor.")
    elif "invalid api key" in low or "jwt" in low or "401" in low or "apikey" in low:
        print("  -> The key is invalid. Copy it again from Project Settings -> API Keys (no spaces or quotes).")
    elif "getaddrinfo" in low or "timed out" in low or "connect" in low:
        print("  -> Network/URL problem. Check internet, VPN/firewall, and that the project URL is right.")
    else:
        print("  -> Send this error text to your assistant.")
    sys.exit(1)

print("4) Cleaning up test row")
try:
    client.table("vitals").delete().eq("patient_id", "dbcheck").execute()
    print("   cleanup ok")
except Exception as e:
    print(f"   [WARN] could not delete test row ({type(e).__name__}); remove patient_id='dbcheck' manually.")

print("\n[OK] Database connection works. Restart uvicorn and the Database pill should turn green.")