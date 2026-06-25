"""API-key authentication for the run endpoints.

A shared secret in the `X-API-Key` header is compared (constant-time) against the
server's `IAG_SIM_API_KEY` env var. This is an API concern, NOT a `Settings` field
— it must never bleed into a per-request `Settings` object.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_ENV = "IAG_SIM_API_KEY"
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_header_scheme)) -> None:
    """FastAPI dependency: 401 unless the header matches IAG_SIM_API_KEY."""
    expected = os.environ.get(API_KEY_ENV, "")
    if not expected or not api_key or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )
