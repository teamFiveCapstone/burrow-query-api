# security.py
from fastapi import HTTPException, status, Security
from fastapi.security import APIKeyHeader
from config import settings
from logger import log_error

api_key_header = APIKeyHeader(name="x-api-token", auto_error=False)


async def verify_api_token(api_token=Security(api_key_header)):
    if not settings.api_token:
        log_error("API token not configured on server")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API token not configured on server",
        )

    if not api_token or api_token != settings.api_token:
        reason = "missing" if not api_token else "invalid"
        log_error(
            "API token authentication failed",
            reason=reason,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
        )

    return True
