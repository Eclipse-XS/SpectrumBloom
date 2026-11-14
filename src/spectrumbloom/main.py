"""FastAPI application factory and runtime bootstrap for SpectrumBloom."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from spectrumbloom import __version__
from spectrumbloom.api.routes import router as api_router
from spectrumbloom.core.config import Settings, get_settings
from spectrumbloom.core.security import sanitize_json_payload
from spectrumbloom.core.telemetry import TelemetryRuntime
from spectrumbloom.domain.exceptions import (
    DatasetNotFoundError,
    DatasetValidationError,
    LLMConfigurationError,
    LLMInvocationError,
    MLExperimentError,
    SandboxTimeoutError,
    SandboxUnavailableError,
    SessionNotFoundError,
)
from spectrumbloom.llm.providers import UnifiedLLMProvider
from spectrumbloom.repositories.dataset_store import (
    LocalFilesystemDatasetStore,
    S3CompatibleDatasetStore,
)
from spectrumbloom.repositories.session_repo import (
    InMemorySessionRepository,
    RedisSessionRepository,
)
from spectrumbloom.sandbox.service import create_sandbox
from spectrumbloom.services.session_service import SpectrumBloomService


def _build_session_repo(settings: Settings):
    if settings.redis_url.strip():
        return RedisSessionRepository(
            redis_url=settings.redis_url.strip(),
            ttl_seconds=settings.session_ttl_seconds,
        )
    return InMemorySessionRepository(ttl_seconds=settings.session_ttl_seconds)


def _build_dataset_store(settings: Settings):
    if settings.dataset_storage_backend == "s3":
        return S3CompatibleDatasetStore(
            bucket=settings.s3_bucket,
            prefix=settings.s3_prefix,
            endpoint_url=settings.s3_endpoint_url or None,
            region=settings.s3_region or None,
            delete_on_session_delete=settings.s3_delete_on_session_delete,
        )
    return LocalFilesystemDatasetStore(base_dir=settings.local_dataset_dir)


def _build_shared_checkpointer(settings: Settings) -> Any | None:
    if not settings.redis_url.strip():
        return None
    try:
        from langgraph.checkpoint.redis import RedisSaver

        saver = RedisSaver(redis_url=settings.redis_url.strip())
        saver.setup()
        return saver
    except Exception:
        logging.getLogger(__name__).warning(
            "Redis URL configured, but langgraph-checkpoint-redis unavailable; "
            "using per-session MemorySaver."
        )
        return None


def create_app(
    settings: Settings | None = None,
    *,
    in_memory_telemetry: bool = False,
) -> FastAPI:
    """Create a configured FastAPI application with wired dependencies."""
    cfg = settings or get_settings()
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))

    telemetry = TelemetryRuntime(cfg, in_memory_export=in_memory_telemetry)
    session_repo = _build_session_repo(cfg)
    dataset_store = _build_dataset_store(cfg)
    sandbox = create_sandbox(cfg)
    llm_provider = UnifiedLLMProvider(cfg)
    shared_checkpointer = _build_shared_checkpointer(cfg)

    service = SpectrumBloomService(
        settings=cfg,
        session_repo=session_repo,
        dataset_store=dataset_store,
        sandbox=sandbox,
        llm_provider=llm_provider,
        telemetry=telemetry,
        shared_checkpointer=shared_checkpointer,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        telemetry.shutdown()

    app = FastAPI(
        title="SpectrumBloom API",
        description="Local-first agentic Exploratory Data Analysis (EDA) platform.",
        version=__version__,
        lifespan=lifespan,
    )

    app.state.settings = cfg
    app.state.telemetry = telemetry
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    telemetry.instrument_fastapi(app)

    # -- Exception handlers --------------------------------------------------
    @app.exception_handler(SessionNotFoundError)
    async def _handle_not_found(_req: Request, exc: SessionNotFoundError):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(DatasetNotFoundError)
    async def _handle_dataset_missing(_req: Request, exc: DatasetNotFoundError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    @app.exception_handler(DatasetValidationError)
    async def _handle_dataset_invalid(_req: Request, exc: DatasetValidationError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    @app.exception_handler(MLExperimentError)
    async def _handle_ml_error(_req: Request, exc: MLExperimentError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": str(exc)}
        )

    @app.exception_handler(LLMConfigurationError)
    async def _handle_llm_config(_req: Request, exc: LLMConfigurationError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    @app.exception_handler(SandboxTimeoutError)
    async def _handle_sandbox_timeout(_req: Request, exc: SandboxTimeoutError):
        return JSONResponse(
            status_code=status.HTTP_408_REQUEST_TIMEOUT, content={"detail": str(exc)}
        )

    @app.exception_handler(SandboxUnavailableError)
    async def _handle_sandbox_unavailable(_req: Request, exc: SandboxUnavailableError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
        )

    @app.exception_handler(LLMInvocationError)
    async def _handle_llm_error(_req: Request, exc: LLMInvocationError):
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})

    app.include_router(api_router)

    @app.get("/health", tags=["Operations"])
    def health() -> JSONResponse:
        repo_ok = session_repo.ping()
        store_ok = dataset_store.ping()
        llm_status = llm_provider.check_local_backend_health()
        overall = "ok" if (repo_ok and store_ok) else "degraded"
        return JSONResponse(
            content=sanitize_json_payload(
                {
                    "status": overall,
                    "version": __version__,
                    "session_repository": "connected" if repo_ok else "unavailable",
                    "dataset_storage": {
                        "backend": dataset_store.backend_name,
                        "status": "connected" if store_ok else "unavailable",
                    },
                    "sandbox_backend": sandbox.backend_name,
                    "llm": llm_status,
                    "observability": {
                        "opentelemetry": True,
                        "langfuse": telemetry.langfuse_enabled,
                    },
                }
            )
        )

    return app


app = create_app()


def run_server() -> None:
    settings = get_settings()
    uvicorn.run(
        "spectrumbloom.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )
