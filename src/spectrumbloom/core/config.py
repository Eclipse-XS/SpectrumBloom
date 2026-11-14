"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for SpectrumBloom."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "SpectrumBloom"
    app_env: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104
    api_port: int = 8000
    cors_allow_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    api_auth_token: str = ""

    # Storage & session persistence
    dataset_storage_backend: Literal["local", "s3"] = "local"
    local_dataset_dir: Path = Path("./data/datasets")
    max_upload_mb: int = Field(default=25, ge=1, le=250)
    session_ttl_seconds: int = Field(default=86400, ge=60)
    redis_url: str = ""

    # Optional S3-compatible object storage (MinIO, R2, GCS interop, S3)
    s3_bucket: str = ""
    s3_endpoint_url: str = ""
    s3_region: str = ""
    s3_prefix: str = "datasets"
    s3_delete_on_session_delete: bool = True

    # Local-first LLM configuration
    llm_provider: Literal["ollama", "llamacpp", "openai", "gemini"] = "ollama"
    llm_model: str = "qwen3:4b-instruct"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_num_ctx: int = Field(default=4096, ge=1024, le=32768)
    llm_max_tokens: int = Field(default=1536, ge=128, le=8192)
    llm_request_timeout_seconds: int = Field(default=120, ge=5, le=600)
    llm_enable_thinking: bool = False

    ollama_base_url: str = "http://localhost:11434"
    llamacpp_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str = ""
    openai_base_url: str = ""
    gemini_api_key: str = ""

    # Fail-closed sandbox configuration
    sandbox_backend: Literal["strict_subprocess", "docker", "nsjail"] = "strict_subprocess"
    sandbox_timeout_seconds: int = Field(default=15, ge=1, le=120)
    sandbox_memory_limit_mb: int = Field(default=512, ge=128, le=4096)
    sandbox_max_output_chars: int = Field(default=20000, ge=1000, le=200000)
    sandbox_docker_image: str = "spectrumbloom-sandbox:latest"

    # Observability
    otel_service_name: str = "spectrumbloom-api"
    otel_exporter_otlp_endpoint: str = ""
    otel_console_export: bool = False

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        origins = [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]
        return origins or ["http://localhost:8501"]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
