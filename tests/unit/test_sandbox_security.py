"""Unit tests for fail-closed sandbox execution and security isolation."""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

from spectrumbloom.core.config import Settings
from spectrumbloom.domain.exceptions import (
    SandboxSecurityError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from spectrumbloom.sandbox.ast_guard import validate_code_safety
from spectrumbloom.sandbox.service import (
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
