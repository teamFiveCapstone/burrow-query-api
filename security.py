# security.py
from fastapi import Request, HTTPException, status
from config import settings

async def verify_api_token(request: Request):
    x_api_token = request.headers.get("x-api-token")

    if not settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API token not configured on server",
        )

    if not x_api_token or x_api_token != settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
        )

    return True
