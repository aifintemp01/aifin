"""
Device-bound session auth — NOT user authentication.

There's no password or profile yet. Each browser/device gets a random
device_id, wrapped in a signed JWT so it can't be tampered with client-side.
The frontend stores this token in localStorage and sends it on every request;
the backend uses it purely to scope data (flows, runs) to that device.

This deliberately does NOT prove identity — anyone with the token IS that
device, and copying the token to another device just makes that device
"the same" one. That's fine for now: the goal is separating browsers, not
securing accounts. When real accounts get added later, a device_id can be
linked to a user_id without needing to touch the flows/runs tables again.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Header, HTTPException

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 365  # long-lived — this is a device identifier, not a login session


def create_device_token(device_id: str | None = None) -> tuple[str, str]:
    """Issue a new JWT for a device. Generates a fresh device_id if none given."""
    device_id = device_id or str(uuid.uuid4())
    payload = {
        "device_id": device_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, device_id


def decode_device_token(token: str) -> str:
    """Verify a JWT and return its device_id. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload["device_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Device token expired")
    except (jwt.InvalidTokenError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid device token")


def get_device_id(authorization: str | None = Header(default=None)) -> str:
    """
    FastAPI dependency — extracts and verifies the device_id from the
    Authorization header. Use as: device_id: str = Depends(get_device_id)
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing device token")
    token = authorization.removeprefix("Bearer ").strip()
    return decode_device_token(token)