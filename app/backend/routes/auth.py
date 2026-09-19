from fastapi import APIRouter
from pydantic import BaseModel

from app.backend.auth import create_device_token

router = APIRouter(prefix="/auth", tags=["auth"])


class DeviceTokenResponse(BaseModel):
    token: str
    device_id: str


@router.post("/device", response_model=DeviceTokenResponse)
async def issue_device_token():
    """
    Issue a new device-bound token. The frontend calls this once, the first
    time it finds no token in localStorage, then reuses the same token on
    every request after that.
    """
    token, device_id = create_device_token()
    return DeviceTokenResponse(token=token, device_id=device_id)