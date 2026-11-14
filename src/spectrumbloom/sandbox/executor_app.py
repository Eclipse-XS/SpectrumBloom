"""Isolated internal sandbox executor microservice (keeps Docker socket off the API container)."""

from __future__ import annotations

import base64
import os
import secrets
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from spectrumbloom.domain.exceptions import (
    SandboxSecurityError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from spectrumbloom.sandbox.service import DockerSandbox

_EXECUTOR_TOKEN = (os.getenv("SANDBOX_EXECUTOR_TOKEN") or "").strip()
_DOCKER_IMAGE = os.getenv("SANDBOX_DOCKER_IMAGE", "spectrumbloom-sandbox:latest")


class ExecutePayload(BaseModel):
    code: str = Field(..., min_length=1, max_length=24000)
    data_format: Literal["parquet", "json"]
    data_b64: str = Field(..., min_length=1)
    timeout_seconds: int = Field(default=25, ge=1, le=120)
    memory_limit_mb: int = Field(default=512, ge=128, le=4096)
    max_output_chars: int = Field(default=20000, ge=1000, le=200000)


def _verify_executor_token(authorization: str | None = Header(default=None)) -> None:
    if len(_EXECUTOR_TOKEN) < 16:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SANDBOX_EXECUTOR_TOKEN must be configured (>= 16 chars).",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer token.",
        )
    provided = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(provided, _EXECUTOR_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden.",
        )


app = FastAPI(title="SpectrumBloom Sandbox Executor", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "image": _DOCKER_IMAGE}


@app.post("/execute", dependencies=[Depends(_verify_executor_token)])
def execute_sandboxed(body: ExecutePayload):
    sandbox = DockerSandbox(
        image=_DOCKER_IMAGE,
        timeout_seconds=body.timeout_seconds,
        memory_limit_mb=body.memory_limit_mb,
        max_output_chars=body.max_output_chars,
    )
    try:
        raw_data = base64.b64decode(body.data_b64, validate=True)
        res = sandbox.execute_serialized(
            clean_code=body.code,
            data_bytes=raw_data,
            data_format=body.data_format,
        )
        return JSONResponse(content=res.model_dump())
    except SandboxSecurityError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except SandboxTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=str(exc)) from exc
    except SandboxUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
