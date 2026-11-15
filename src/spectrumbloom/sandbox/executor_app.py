"""Isolated internal sandbox executor microservice (keeps Docker socket off the API container)."""

from __future__ import annotations

import base64
import binascii
import os
import secrets
import shutil
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from spectrumbloom.domain.exceptions import (
    SandboxSecurityError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from spectrumbloom.sandbox.service import DockerSandbox

_EXECUTOR_TOKEN = (os.getenv("SANDBOX_EXECUTOR_TOKEN") or "").strip()
_DOCKER_IMAGE = os.getenv("SANDBOX_DOCKER_IMAGE", "spectrumbloom-sandbox:latest")
_MAX_DECODED_PAYLOAD_BYTES = 35 * 1024 * 1024
_WEAK_EXECUTOR_TOKENS = {
    "changeme",
    "secret",
    "password",
    "test-token",
    "dev-token",
    "replace-me",
    "local-compose-sandbox-executor-secret-token",
}


def _is_strong_executor_token(token: str) -> bool:
    stripped = token.strip()
    return (
        len(stripped) >= 16
        and stripped.lower() not in _WEAK_EXECUTOR_TOKENS
        and not stripped.lower().startswith("changeme")
    )


class ExecutePayload(BaseModel):
    code: str = Field(..., min_length=1, max_length=24000)
    data_format: Literal["parquet", "json"]
    data_b64: str = Field(..., min_length=1, max_length=50_000_000)
    timeout_seconds: int = Field(default=25, ge=1, le=120)
    memory_limit_mb: int = Field(default=512, ge=128, le=4096)
    max_output_chars: int = Field(default=20000, ge=1000, le=200000)

    @field_validator("data_b64")
    @classmethod
    def _validate_data_b64(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value.encode("ascii"), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 dataset payload.") from exc
        if len(decoded) > _MAX_DECODED_PAYLOAD_BYTES:
            raise ValueError(
                f"Decoded dataset payload exceeds {_MAX_DECODED_PAYLOAD_BYTES} byte limit."
            )
        return value


def _verify_executor_token(authorization: str | None = Header(default=None)) -> None:
    if not _is_strong_executor_token(_EXECUTOR_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SANDBOX_EXECUTOR_TOKEN must be a strong secret (>= 16 chars, non-placeholder).",
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


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if not _is_strong_executor_token(_EXECUTOR_TOKEN):
        raise RuntimeError(
            "Refusing to start sandbox-executor without a strong SANDBOX_EXECUTOR_TOKEN "
            "(>= 16 chars, non-placeholder)."
        )
    yield


app = FastAPI(
    title="SpectrumBloom Sandbox Executor",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict[str, object]:
    docker_ok = bool(shutil.which("docker"))
    token_ok = _is_strong_executor_token(_EXECUTOR_TOKEN)
    return {
        "status": "ok" if (docker_ok and token_ok) else "degraded",
        "image": _DOCKER_IMAGE,
        "docker_available": docker_ok,
        "auth_configured": token_ok,
    }


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
