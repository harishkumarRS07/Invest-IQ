"""
Simple JWT Authentication for InvestIQ API.
Uses a file-based user store (easily replaceable with a DB).
"""
import json
import os
import hashlib
import hmac
import time
import base64

SECRET_KEY = os.getenv("JWT_SECRET", "investiq-super-secret-key-change-in-production")
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# ──────────────────────────────────────────────────────────────
# Tiny user store (JSON file – swap for SQLAlchemy in prod)
# ──────────────────────────────────────────────────────────────

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ──────────────────────────────────────────────────────────────
# Minimal JWT implementation (HS256, no external dependency)
# ──────────────────────────────────────────────────────────────

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    s += "=" * (padding % 4)
    return base64.urlsafe_b64decode(s)

def _create_token(payload: dict, expires_in_seconds: int = 86400) -> str:
    """Create a signed JWT string valid for `expires_in_seconds`."""
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload["exp"] = int(time.time()) + expires_in_seconds
    body = _b64url_encode(json.dumps(payload).encode())
    sig_input = f"{header}.{body}".encode()
    sig = hmac.new(SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url_encode(sig)}"

def _verify_token(token: str) -> dict | None:
    """Verify signature and expiry; return payload or None."""
    try:
        header, body, sig = token.split(".")
        sig_input = f"{header}.{body}".encode()
        expected = hmac.new(SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(sig), expected):
            return None
        payload = json.loads(_b64url_decode(body))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────
# Public helpers used by routes.py
# ──────────────────────────────────────────────────────────────

def register_user(email: str, password: str, name: str) -> dict:
    users = _load_users()
    if email in users:
        return {"error": "User already exists"}
    users[email] = {
        "name": name,
        "password_hash": _hash_password(password),
        "created_at": time.time(),
    }
    _save_users(users)
    token = _create_token({"sub": email, "name": name})
    return {"token": token, "email": email, "name": name}

def login_user(email: str, password: str) -> dict:
    users = _load_users()
    user = users.get(email)
    if not user or user["password_hash"] != _hash_password(password):
        return {"error": "Invalid credentials"}
    token = _create_token({"sub": email, "name": user["name"]})
    return {"token": token, "email": email, "name": user["name"]}

def get_current_user(token: str) -> dict | None:
    """Used as a lightweight auth guard on protected routes."""
    return _verify_token(token)
