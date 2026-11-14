"""Integration tests for FastAPI REST endpoints, auth, upload, ML, agentic query, and OTel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from spectrumbloom.core.config import Settings
from spectrumbloom.main import create_app


class _FakeBindableChatModel:
    """Deterministic fake chat model simulating Qwen tool-calling for CI integration tests."""

    def __init__(self) -> None:
        self._calls = 0

    def bind_tools(self, tools: list[Any]) -> _FakeBindableChatModel:
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self._calls += 1
        if self._calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_1",
                        "name": "python_repl",
                        "args": {
                            "code": (
                                "import plotly.express as px\n"
                                "means = (\n"
                                "    df.groupby('category')['revenue']\n"
                                "    .mean().round(2).to_dict()\n"
                                ")\n"
                                "fig = px.bar(\n"
                                "    x=list(means.keys()),\n"
                                "    y=list(means.values()),\n"
                                "    title='Mean Revenue',\n"
                                ")\n"
                                "plotly_figures.append(fig)\n"
                                "result = means\n"
                                "print(means)\n"
                            ),
                            "thoughts": "compute category revenue means and plot",
                        },
                    }
                ],
            )
        return AIMessage(
            content=(
                "Electronics leads mean revenue, and I generated the "
                "'Mean Revenue' Plotly bar chart."
            )
        )


def test_end_to_end_api_lifecycle_and_otel(tmp_path: Path, monkeypatch) -> None:
    sample_csv = Path(__file__).resolve().parents[2] / "examples" / "sample_ecommerce.csv"
    settings = Settings(
        app_env="test",
        local_dataset_dir=tmp_path / "datasets",
        api_auth_token="secret-test-token",
        sandbox_backend="strict_subprocess",
        allow_trusted_local_subprocess=True,
    )
    app = create_app(settings, in_memory_telemetry=True)
    monkeypatch.setattr(
        app.state.service.llm_provider,
        "build_chat_model",
        lambda **kwargs: _FakeBindableChatModel(),
    )

    with TestClient(app) as client:
        # Health is open
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["sandbox_dynamic_python_enabled"] is True

        # Protected endpoints reject missing/bad bearer tokens
        assert client.post("/api/v1/sessions", json={}).status_code == 401
        assert (
            client.post(
                "/api/v1/sessions",
                json={},
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 403
        )

        headers = {"Authorization": "Bearer secret-test-token"}
        create_res = client.post("/api/v1/sessions", json={}, headers=headers)
        assert create_res.status_code == 201
        sid = create_res.json()["session_id"]

        # Upload CSV
        upload_res = client.post(
            f"/api/v1/sessions/{sid}/upload",
            files={"file": ("sample_ecommerce.csv", sample_csv.read_bytes(), "text/csv")},
            headers=headers,
        )
        assert upload_res.status_code == 200
        eda_body = upload_res.json()
        assert eda_body["profile"]["rows"] == 180
        assert len(eda_body["figures"]) >= 5

        # Preview works
        prev_res = client.get(f"/api/v1/sessions/{sid}/data/preview?rows=5", headers=headers)
        assert prev_res.status_code == 200
        assert len(prev_res.json()["preview"]) == 5

        # ML endpoint works
        ml_res = client.post(
            f"/api/v1/sessions/{sid}/ml/experiment",
            json={"target_column": "churned", "cv_folds": 3},
            headers=headers,
        )
        assert ml_res.status_code == 200
        assert ml_res.json()["task_type"] == "classification"

        # Query endpoint invokes LangGraph + Sandbox + Eval
        q_res = client.post(
            f"/api/v1/sessions/{sid}/query",
            json={"query": "Which category has highest mean revenue?"},
            headers=headers,
        )
        assert q_res.status_code == 200
        q_body = q_res.json()
        assert q_body["status"] == "success"
        assert q_body["timed_out"] is False
        assert "Electronics" in q_body["answer"]
        assert len(q_body["figures"]) == 1
        assert q_body["evaluation"]["passed"] is True

        # Verify OpenTelemetry recorded spans
        spans = app.state.telemetry.in_memory_exporter.get_finished_spans()
        span_names = {s.name for s in spans}
        assert "eda.upload_and_profile" in span_names
        assert "ml.experiment" in span_names
        assert "agent.query" in span_names

        # Delete session cleans up dataset file
        del_res = client.delete(f"/api/v1/sessions/{sid}", headers=headers)
        assert del_res.status_code == 200
        assert client.get(f"/api/v1/sessions/{sid}", headers=headers).status_code == 404


def test_llm_timeout_returns_504_and_recovers_cleanly(tmp_path: Path, monkeypatch) -> None:
    import httpx

    sample_csv = Path(__file__).resolve().parents[2] / "examples" / "sample_ecommerce.csv"
    settings = Settings(
        app_env="test",
        local_dataset_dir=tmp_path / "datasets",
        api_auth_token="secret-test-token",
        sandbox_backend="disabled",
        llm_request_timeout_seconds=10,
    )
    app = create_app(settings, in_memory_telemetry=True)

    class _FlakyTimeoutModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools: list[Any]) -> _FlakyTimeoutModel:
            return self

        def invoke(self, messages: list[Any]) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadTimeout("simulated ollama inference timeout")
            return AIMessage(content="Recovered response after timeout.")

    monkeypatch.setattr(
        app.state.service.llm_provider,
        "build_chat_model",
        lambda **kwargs: _FlakyTimeoutModel(),
    )

    with TestClient(app) as client:
        headers = {"Authorization": "Bearer secret-test-token"}
        sid = client.post("/api/v1/sessions", json={}, headers=headers).json()["session_id"]
        client.post(
            f"/api/v1/sessions/{sid}/upload",
            files={"file": ("sample_ecommerce.csv", sample_csv.read_bytes(), "text/csv")},
            headers=headers,
        )

        # Timeout yields HTTP 504 without corrupting thread history
        t_res = client.post(
            f"/api/v1/sessions/{sid}/query",
            json={"query": "Slow question"},
            headers=headers,
        )
        assert t_res.status_code == 504
        assert "timed out" in t_res.json()["detail"].lower()

        # Subsequent query on same session recovers cleanly
        ok_res = client.post(
            f"/api/v1/sessions/{sid}/query",
            json={"query": "Quick retry"},
            headers=headers,
        )
        assert ok_res.status_code == 200
        assert ok_res.json()["status"] == "success"
        assert "Recovered response" in ok_res.json()["answer"]
