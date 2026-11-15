"""Unit tests for fail-closed sandbox execution and security isolation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from spectrumbloom.core.config import Settings
from spectrumbloom.domain.exceptions import (
    SandboxSecurityError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from spectrumbloom.sandbox import executor_app
from spectrumbloom.sandbox.ast_guard import validate_code_safety
from spectrumbloom.sandbox.service import (
    DisabledSandbox,
    RemoteDockerSandbox,
    SubprocessSandbox,
    _run_watched_windows_subprocess,
    _WindowsJobGuard,
    create_sandbox,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B"],
            "amount": [10.0, 20.0, 15.0, 35.0, 25.0, 45.0],
        }
    )


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "import os\nprint(os.listdir('.'))",
        "import subprocess\nsubprocess.run(['whoami'])",
        "import socket",
        "from pathlib import Path\nPath('.env').read_text()",
        "df.eval('amount + 1')",
        "df.query('amount > 10')",
        "getattr(df, '__class__')",
        "eval('1+1')",
        "open('secret.txt', 'w').write('x')",
        "df.to_csv('leak.csv')",
        "pd.read_csv('local.csv')",
        "x = ().__class__.__base__.__subclasses__()",
    ],
)
def test_ast_guard_blocks_unsafe_constructs(unsafe_code: str) -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code_safety(unsafe_code)


def test_sandbox_executes_valid_eda_and_plotly(sample_df: pd.DataFrame) -> None:
    sb = SubprocessSandbox(timeout_seconds=30, memory_limit_mb=512)
    res = sb.execute(
        """
import plotly.express as px
from scipy import stats

mean_val = float(df["amount"].mean())
fig = px.bar(df, x="category", y="amount", title="Category Amounts")
plotly_figures.append(fig)
result = {"mean": mean_val, "skew": round(float(stats.skew(df["amount"])), 4)}
print("done")
""",
        sample_df,
    )
    assert res.error is None
    assert res.stdout.strip() == "done"
    assert res.result["mean"] == 25.0
    assert len(res.figures) == 1
    assert res.figures[0]["title"] == "Category Amounts"


def test_sandbox_captures_assigned_fig_with_trailing_expression(sample_df: pd.DataFrame) -> None:
    sb = SubprocessSandbox(timeout_seconds=30, memory_limit_mb=512)
    res = sb.execute(
        """
import plotly.express as px

fig = px.histogram(df, x="amount", title="Amount Histogram")
float(df["amount"].quantile(0.9))
""",
        sample_df,
    )
    assert res.error is None
    assert res.result == pytest.approx(float(sample_df["amount"].quantile(0.9)))
    assert len(res.figures) == 1
    assert res.figures[0]["title"] == "Amount Histogram"


def test_sandbox_allows_safe_dataframe_replace(sample_df: pd.DataFrame) -> None:
    sb = SubprocessSandbox(timeout_seconds=25, memory_limit_mb=512)
    res = sb.execute(
        (
            'cleaned = df["category"].replace({"A": "Alpha"})\n'
            'result = int((cleaned == "Alpha").sum())'
        ),
        sample_df,
    )
    assert res.error is None
    assert res.result == 3


def test_sandbox_enforces_timeout(sample_df: pd.DataFrame) -> None:
    sb = SubprocessSandbox(timeout_seconds=1, memory_limit_mb=512)
    with pytest.raises(SandboxTimeoutError):
        sb.execute("while True:\n    pass", sample_df)


def test_sandbox_enforces_memory_cap(sample_df: pd.DataFrame) -> None:
    sb = SubprocessSandbox(timeout_seconds=25, memory_limit_mb=384)
    res = sb.execute(
        "arr = np.full((520, 1024, 1024), 7, dtype=np.uint8)\nresult = int(arr[0, 0, 0])",
        sample_df,
    )
    assert res.error is not None
    assert "NameError" not in res.error
    assert any(tok in res.error.lower() for tok in ("memory", "unable to allocate"))


def test_fail_closed_when_nsjail_unavailable() -> None:
    settings = Settings(sandbox_backend="nsjail")
    with pytest.raises(SandboxUnavailableError):
        create_sandbox(settings)


def test_sandbox_captures_trailing_repl_expression(sample_df: pd.DataFrame) -> None:
    sb = SubprocessSandbox(timeout_seconds=25, memory_limit_mb=512)
    res = sb.execute("df['amount'].sum()", sample_df)
    assert res.error is None
    assert res.result == pytest.approx(150.0)


def test_ast_guard_does_not_unwrap_json_fence() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code_safety('```json\n{"name": "python_repl"}\n```')


def test_windows_job_guard_fails_closed_on_init_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_init(self: object, memory_limit_mb: int) -> None:
        raise OSError("simulated CreateJobObjectW failure")

    monkeypatch.setattr(_WindowsJobGuard, "__init__", _fail_init)
    with pytest.raises(SandboxUnavailableError, match="Job Object initialization failed"):
        _run_watched_windows_subprocess(
            [sys.executable, "-c", "print(1)"],
            cwd=os.getcwd(),  # type: ignore[arg-type]
            env=os.environ.copy(),
            timeout_seconds=5,
            memory_limit_mb=256,
        )


def test_windows_job_guard_fails_closed_when_assign_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[bool] = []
    closed: list[bool] = []

    class _DummyJob:
        def __init__(self, memory_limit_mb: int) -> None:
            self.last_error = 5  # ERROR_ACCESS_DENIED

        def assign(self, proc: object) -> bool:
            return False

        def close(self) -> None:
            closed.append(True)

    class _DummyProc:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.returncode = None

        def kill(self) -> None:
            killed.append(True)
            self.returncode = -9

        def communicate(self) -> tuple[str, str]:
            return ("", "")

    monkeypatch.setattr("spectrumbloom.sandbox.service._WindowsJobGuard", _DummyJob)
    monkeypatch.setattr(subprocess, "Popen", _DummyProc)

    with pytest.raises(SandboxUnavailableError, match="winerror=5"):
        _run_watched_windows_subprocess(
            [sys.executable, "-c", "print(1)"],
            cwd=os.getcwd(),  # type: ignore[arg-type]
            env=os.environ.copy(),
            timeout_seconds=5,
            memory_limit_mb=256,
        )

    assert killed == [True]
    assert closed == [True]


@pytest.mark.skipif(os.name != "nt", reason="Windows Kernel Job Object integration test")
def test_windows_job_guard_real_lifecycle() -> None:
    guard = _WindowsJobGuard(memory_limit_mb=256)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert guard.assign(proc) is True
        peak = guard.peak_memory_mb()
        assert peak is not None and peak > 0.0
        guard.terminate()
        rc = proc.wait(timeout=5)
        assert rc == 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        guard.close()


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "import os\nx = os.environ",
        "import os\nos.fork()",
        "from pathlib import Path\nPath('/etc/passwd').read_bytes()",
        "from pathlib import Path\nPath('out.txt').write_text('x')",
        "import os\nos.execv('/bin/sh', ['sh'])",
    ],
)
def test_ast_guard_blocks_env_exec_fork_and_pathlib_io(unsafe_code: str) -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code_safety(unsafe_code)


def test_create_sandbox_fails_closed_for_missing_docker_and_untrusted_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    with pytest.raises(SandboxUnavailableError, match="docker"):
        create_sandbox(Settings(sandbox_backend="docker"))

    untrusted = Settings.model_construct(
        app_env="production",
        sandbox_backend="strict_subprocess",
        allow_trusted_local_subprocess=True,
        sandbox_timeout_seconds=10,
        sandbox_memory_limit_mb=512,
        sandbox_max_output_chars=20000,
    )
    with pytest.raises(SandboxUnavailableError, match="strict_subprocess"):
        create_sandbox(untrusted)


def test_disabled_sandbox_never_executes_or_falls_back(sample_df: pd.DataFrame) -> None:
    sb = create_sandbox(Settings(sandbox_backend="disabled"))
    assert isinstance(sb, DisabledSandbox)
    assert sb.backend_name == "disabled"
    with pytest.raises(SandboxUnavailableError, match="disabled"):
        sb.execute("result = 1 + 1", sample_df)


def test_sandbox_truncates_output_exhaustion(sample_df: pd.DataFrame) -> None:
    sb = SubprocessSandbox(timeout_seconds=10, memory_limit_mb=512, max_output_chars=2000)
    res = sb.execute("print('A' * 15000)", sample_df)
    assert res.error is None
    assert len(res.stdout) <= 2100
    assert "truncated" in res.stdout


def test_executor_app_enforces_token_and_remote_sandbox_roundtrips(
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_app, "_EXECUTOR_TOKEN", "strong-executor-token-1234")
    client = TestClient(executor_app.app)

    # Missing/wrong token fails closed
    assert client.post("/execute", json={}).status_code == 401
    assert (
        client.post(
            "/execute",
            json={},
            headers={"Authorization": "Bearer wrong-token"},
        ).status_code
        == 403
    )

    # Route RemoteDockerSandbox requests cleanly through TestClient
    def _fake_execute_serialized(self, *, clean_code: str, data_bytes: bytes, data_format: str):
        local_sb = SubprocessSandbox(timeout_seconds=10, memory_limit_mb=512)
        return local_sb.execute(clean_code, sample_df)

    monkeypatch.setattr(executor_app.DockerSandbox, "execute_serialized", _fake_execute_serialized)

    def _mock_httpx_post(url: str, json: dict, headers: dict, timeout: float):
        path = url.replace("http://executor.local", "")
        resp = client.post(path, json=json, headers=headers)
        return httpx.Response(
            status_code=resp.status_code,
            content=resp.content,
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(httpx, "post", _mock_httpx_post)

    remote = RemoteDockerSandbox(
        executor_url="http://executor.local",
        executor_token="strong-executor-token-1234",
        timeout_seconds=15,
        memory_limit_mb=512,
    )
    res = remote.execute("result = int(len(df))", sample_df)
    assert res.error is None
    assert res.backend == "remote"
    assert res.result == 6

    # Transport unavailability fails closed without fallback
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("connection refused")),
    )
    with pytest.raises(SandboxUnavailableError, match="no fallback"):
        remote.execute("result = 1", sample_df)


def test_executor_app_startup_fails_closed_on_weak_token_and_rejects_bad_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert executor_app._is_strong_executor_token("") is False
    assert executor_app._is_strong_executor_token("changeme") is False
    assert (
        executor_app._is_strong_executor_token("local-compose-sandbox-executor-secret-token")
        is False
    )
    assert executor_app._is_strong_executor_token("short-token") is False
    assert executor_app._is_strong_executor_token("a" * 32) is True

    # Startup lifespan MUST fail closed if SANDBOX_EXECUTOR_TOKEN is weak or placeholder
    monkeypatch.setattr(
        executor_app, "_EXECUTOR_TOKEN", "local-compose-sandbox-executor-secret-token"
    )
    with pytest.raises(RuntimeError, match="SANDBOX_EXECUTOR_TOKEN"):
        with TestClient(executor_app.app):
            pass

    monkeypatch.setattr(executor_app, "_EXECUTOR_TOKEN", "strong-executor-token-for-unit-test-1234")
    with TestClient(executor_app.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        bad_b64 = client.post(
            "/execute",
            json={"code": "x = 1", "data_b64": "not!valid!base64!", "data_format": "parquet"},
            headers={"Authorization": "Bearer strong-executor-token-for-unit-test-1234"},
        )
        assert bad_b64.status_code == 422

        import base64

        oversized_raw = b"0" * (executor_app._MAX_DECODED_PAYLOAD_BYTES + 1024)
        oversized_b64 = base64.b64encode(oversized_raw).decode("ascii")
        too_big = client.post(
            "/execute",
            json={"code": "x = 1", "data_b64": oversized_b64, "data_format": "parquet"},
            headers={"Authorization": "Bearer strong-executor-token-for-unit-test-1234"},
        )
        assert too_big.status_code == 422
