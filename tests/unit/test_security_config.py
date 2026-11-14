"""Unit tests for production authentication and public-bind fail-closed guards."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from spectrumbloom.core.config import Settings
from spectrumbloom.main import create_app


def test_production_requires_api_auth_token() -> None:
    with pytest.raises(ValidationError, match="API_AUTH_TOKEN"):
        Settings(app_env="production", api_auth_token="")


def test_staging_rejects_short_or_placeholder_tokens() -> None:
    for bad_token in ("short", "changeme", "replace-me", "secret"):
        with pytest.raises(ValidationError, match="API_AUTH_TOKEN"):
            Settings(app_env="staging", api_auth_token=bad_token)


def test_production_rejects_wildcard_cors() -> None:
    with pytest.raises(ValidationError, match="CORS_ALLOW_ORIGINS"):
        Settings(
            app_env="production",
            api_auth_token="prod-secret-token-with-sufficient-length",
            cors_allow_origins="*",
        )


def test_development_blocks_unauthenticated_non_loopback_bind() -> None:
    with pytest.raises(ValidationError, match="non-loopback"):
        Settings(app_env="development", api_host="0.0.0.0", api_auth_token="")  # noqa: S104


def test_production_valid_config_enforces_auth_end_to_end() -> None:
    settings = Settings(
        app_env="production",
        api_auth_token="prod-secret-token-with-sufficient-length",
        cors_allow_origins="https://spectrumbloom.example.com",
    )
    app = create_app(settings, in_memory_telemetry=True)
    with TestClient(app) as client:
        unauth = client.get("/api/v1/sessions")
        assert unauth.status_code == 401

        bad_auth = client.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert bad_auth.status_code == 403

        ok_auth = client.get(
            "/api/v1/sessions",
            headers={"Authorization": f"Bearer {settings.api_auth_token}"},
        )
        assert ok_auth.status_code == 200


def test_verify_api_token_fails_closed_in_production_even_if_mutated() -> None:
    malformed = Settings.model_construct(
        app_env="production",
        api_auth_token="",
        cors_allow_origins="https://spectrumbloom.example.com",
        cors_origins_list=["https://spectrumbloom.example.com"],
        otel_service_name="spectrumbloom-test",
        otel_exporter_otlp_endpoint="",
        otel_console_export=False,
        sandbox_backend="docker",
        allow_trusted_local_subprocess=False,
        sandbox_executor_url="",
        sandbox_executor_token="",
        sandbox_timeout_seconds=10,
        sandbox_memory_limit_mb=512,
        sandbox_max_output_chars=20000,
        dataset_storage_backend="local",
        local_dataset_dir=Settings().local_dataset_dir,
        redis_url="",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_base_url="",
        llm_provider="ollama",
        llm_model="qwen3:4b-instruct",
        llm_temperature=0.1,
        llm_num_ctx=4096,
        llm_max_tokens=1536,
        llm_request_timeout_seconds=30,
        llm_enable_thinking=False,
        ollama_base_url="http://127.0.0.1:11434",
        llamacpp_base_url="http://127.0.0.1:8080/v1",
        openai_api_key="",
        openai_base_url="",
        gemini_api_key="",
        max_upload_mb=25,
        session_ttl_seconds=3600,
        sandbox_docker_image="spectrumbloom-sandbox:latest",
    )
    app = create_app(malformed, in_memory_telemetry=True)
    with TestClient(app) as client:
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 503


def test_default_sandbox_backend_is_docker() -> None:
    defaults = Settings()
    assert defaults.sandbox_backend == "docker"
    assert defaults.allow_trusted_local_subprocess is False


def test_strict_subprocess_prohibited_in_production_and_without_explicit_flag() -> None:
    with pytest.raises(ValidationError, match="strict_subprocess"):
        Settings(
            app_env="production",
            api_auth_token="prod-secret-token-with-sufficient-length",
            cors_allow_origins="https://spectrumbloom.example.com",
            sandbox_backend="strict_subprocess",
            allow_trusted_local_subprocess=True,
        )

    with pytest.raises(ValidationError, match="ALLOW_TRUSTED_LOCAL_SUBPROCESS"):
        Settings(
            app_env="development",
            sandbox_backend="strict_subprocess",
            allow_trusted_local_subprocess=False,
        )

    allowed_dev = Settings(
        app_env="development",
        sandbox_backend="strict_subprocess",
        allow_trusted_local_subprocess=True,
    )
    assert allowed_dev.sandbox_backend == "strict_subprocess"


def test_remote_sandbox_requires_url_and_strong_token() -> None:
    with pytest.raises(ValidationError, match="SANDBOX_EXECUTOR_URL"):
        Settings(
            app_env="production",
            api_auth_token="prod-secret-token-with-sufficient-length",
            cors_allow_origins="https://spectrumbloom.example.com",
            sandbox_backend="remote",
            sandbox_executor_url="",
            sandbox_executor_token="strong-executor-token-1234",
        )

    with pytest.raises(ValidationError, match="SANDBOX_EXECUTOR_TOKEN"):
        Settings(
            app_env="production",
            api_auth_token="prod-secret-token-with-sufficient-length",
            cors_allow_origins="https://spectrumbloom.example.com",
            sandbox_backend="remote",
            sandbox_executor_url="http://sandbox-executor:8090",
            sandbox_executor_token="short",
        )


def test_disabled_sandbox_exposes_flag_in_health_endpoint() -> None:
    settings = Settings(
        app_env="production",
        api_auth_token="prod-secret-token-with-sufficient-length",
        cors_allow_origins="https://spectrumbloom.example.com",
        sandbox_backend="disabled",
    )
    app = create_app(settings, in_memory_telemetry=True)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["sandbox_backend"] == "disabled"
        assert body["sandbox_dynamic_python_enabled"] is False
