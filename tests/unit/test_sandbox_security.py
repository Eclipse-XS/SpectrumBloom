"""Unit tests for fail-closed sandbox execution and security isolation."""

from __future__ import annotations

import pandas as pd
import pytest

from spectrumbloom.core.config import Settings
from spectrumbloom.domain.exceptions import (
    SandboxSecurityError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from spectrumbloom.sandbox.ast_guard import validate_code_safety
from spectrumbloom.sandbox.service import SubprocessSandbox, create_sandbox


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
    sb = SubprocessSandbox(timeout_seconds=10, memory_limit_mb=512)
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
    sb = SubprocessSandbox(timeout_seconds=10, memory_limit_mb=512)
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
    sb = SubprocessSandbox(timeout_seconds=10, memory_limit_mb=384)
    res = sb.execute(
        "buf = bytearray(500 * 1024 * 1024)\nbuf[::4096] = b'x' * (128000)",
        sample_df,
    )
    assert res.error is not None


def test_fail_closed_when_nsjail_unavailable() -> None:
    settings = Settings(sandbox_backend="nsjail")
    with pytest.raises(SandboxUnavailableError):
        create_sandbox(settings)


def test_sandbox_captures_trailing_repl_expression(sample_df: pd.DataFrame) -> None:
    sb = SubprocessSandbox(timeout_seconds=10, memory_limit_mb=512)
    res = sb.execute("df['amount'].sum()", sample_df)
    assert res.error is None
    assert res.result == pytest.approx(150.0)


def test_ast_guard_does_not_unwrap_json_fence() -> None:
    with pytest.raises(SandboxSecurityError):
        validate_code_safety('```json\n{"name": "python_repl"}\n```')
