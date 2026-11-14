"""API authentication, upload sanitization, and JSON safety utilities."""

from __future__ import annotations

import math
import re
import secrets
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException, Request, status

from spectrumbloom.core.config import Settings

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_filename(filename: str) -> str:
    """Strip path traversal and unsafe characters from an uploaded filename."""
    base = Path(filename or "").name.strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "dataset.csv"


def neutralize_formula_value(value: Any) -> Any:
    """Prevent spreadsheet formula injection when exporting/displaying string cells."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return f"'{value}"
    return value


def sanitize_json_payload(obj: Any) -> Any:
    """Recursively convert NaN/Inf and non-JSON scalar values to JSON-safe primitives."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_json_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json_payload(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return sanitize_json_payload(obj.item())
        except Exception:
            return str(obj)
    return obj


def verify_api_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Enforce Bearer token auth (mandatory in staging/production; optional in dev/test)."""
    settings: Settings = request.app.state.settings
    expected = (settings.api_auth_token or "").strip()
    if not expected:
        if settings.app_env in {"production", "staging"}:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Authentication is required in staging/production, "
                    "but API_AUTH_TOKEN is not configured."
                ),
            )
        return

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Bearer authorization token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authorization token.",
        )
