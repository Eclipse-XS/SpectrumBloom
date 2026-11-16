"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_allow_origins: str = (
        "http://localhost:8501,http://127.0.0.1:8501,http://localhost:4200,http://127.0.0.1:4200"
    )
    api_auth_token: str = ""

    # Storage & session persistence
    dataset_storage_backend: Literal["local", "s3"] = "local"
    local_dataset_dir: Path = Path("./data/datasets")
    max_upload_mb: int = Field(default=25, ge=1, le=250)
    session_ttl_seconds: int = Field(default=86400, ge=60)
    max_cached_sessions: int = Field(default=128, ge=1, le=4096)
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
    llm_request_timeout_seconds: int = Field(default=90, ge=5, le=600)
    llm_enable_thinking: bool = False

    ollama_base_url: str = "http://localhost:11434"
    llamacpp_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str = ""
    openai_base_url: str = ""
    gemini_api_key: str = ""

    # Fail-closed sandbox configuration
    sandbox_backend: Literal["docker", "remote", "nsjail", "strict_subprocess", "disabled"] = (
        "docker"
    )
    allow_trusted_local_subprocess: bool = False
    sandbox_timeout_seconds: int = Field(default=25, ge=1, le=120)
    sandbox_memory_limit_mb: int = Field(default=512, ge=128, le=4096)
    sandbox_max_output_chars: int = Field(default=20000, ge=1000, le=200000)
    sandbox_docker_image: str = "spectrumbloom-sandbox:latest"
    sandbox_executor_url: str = ""
    sandbox_executor_token: str = ""

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

    @model_validator(mode="after")
    def _validate_security_configuration(self) -> Settings:
        weak_tokens = {
            "changeme",
            "change-me",
            "replace-me",
            "replace_me",
            "secret",
            "token",
            "password",
            "admin",
        }
        token = (self.api_auth_token or "").strip()
        if self.app_env in {"production", "staging"}:
            if not token or token.lower() in weak_tokens:
                raise ValueError(
                    "API_AUTH_TOKEN must be configured with a strong secret when "
                    "APP_ENV is production or staging."
                )
            if len(token) < 16:
                raise ValueError(
                    "API_AUTH_TOKEN must be at least 16 characters when "
                    "APP_ENV is production or staging."
                )
            if "*" in self.cors_origins_list:
                raise ValueError(
                    "CORS_ALLOW_ORIGINS cannot include wildcard '*' when "
                    "APP_ENV is production or staging."
                )
            if self.sandbox_backend == "strict_subprocess":
                raise ValueError(
                    "SANDBOX_BACKEND=strict_subprocess is restricted to trusted local "
                    "development and prohibited when APP_ENV is production or staging. "
                    "Use 'docker', 'remote', 'nsjail', or 'disabled'."
                )
        elif (
            self.app_env == "development"
            and self.api_host.strip() not in {"127.0.0.1", "localhost", "::1"}
            and not token
        ):
            raise ValueError(
                "Refusing to bind non-loopback API_HOST without API_AUTH_TOKEN in "
                "development mode. Set API_AUTH_TOKEN or bind API_HOST=127.0.0.1."
            )

        if self.sandbox_backend == "strict_subprocess" and not self.allow_trusted_local_subprocess:
            raise ValueError(
                "SANDBOX_BACKEND=strict_subprocess is restricted to trusted local development "
                "and requires ALLOW_TRUSTED_LOCAL_SUBPROCESS=true. Use 'docker' for isolated "
                "execution or 'disabled' for deterministic-only mode."
            )

        if self.sandbox_backend == "remote":
            exec_url = (self.sandbox_executor_url or "").strip()
            exec_token = (self.sandbox_executor_token or "").strip()
            if not exec_url.startswith(("http://", "https://")):
                raise ValueError(
                    "SANDBOX_EXECUTOR_URL must be an http(s) URL when SANDBOX_BACKEND=remote."
                )
            if not exec_token or exec_token.lower() in weak_tokens or len(exec_token) < 16:
                raise ValueError(
                    "SANDBOX_EXECUTOR_TOKEN must be a strong secret (>= 16 chars) when "
                    "SANDBOX_BACKEND=remote."
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
