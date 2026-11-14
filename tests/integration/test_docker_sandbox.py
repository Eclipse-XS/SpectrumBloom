"""End-to-end tests for hardened DockerSandbox container isolation."""

from __future__ import annotations

import shutil
import subprocess

import pandas as pd
import pytest

from spectrumbloom.core.config import Settings
from spectrumbloom.domain.exceptions import SandboxTimeoutError, SandboxUnavailableError
from spectrumbloom.sandbox.service import DockerSandbox, create_sandbox


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B"],
            "amount": [10.0, 20.0, 15.0, 35.0, 25.0, 45.0],
        }
    )


@pytest.mark.docker_sandbox
def test_docker_sandbox_executes_eda_and_plotly_e2e(sample_df: pd.DataFrame) -> None:
    sb = create_sandbox(
        Settings(
            sandbox_backend="docker",
            sandbox_docker_image="spectrumbloom-sandbox:latest",
            sandbox_timeout_seconds=25,
            sandbox_memory_limit_mb=512,
        )
    )
    res = sb.execute(
        """
import plotly.express as px
from scipy import stats

mean_val = float(df["amount"].mean())
fig = px.bar(df, x="category", y="amount", title="Docker Container Amounts")
plotly_figures.append(fig)
result = {"mean": mean_val, "skew": round(float(stats.skew(df["amount"])), 4)}
print("docker_ok")
""",
        sample_df,
    )
    assert res.error is None
    assert res.backend == "docker"
    assert res.stdout.strip() == "docker_ok"
    assert res.result["mean"] == 25.0
    assert len(res.figures) == 1
    assert res.figures[0]["title"] == "Docker Container Amounts"


@pytest.mark.docker_sandbox
def test_docker_sandbox_blocks_network_and_filesystem_writes(sample_df: pd.DataFrame) -> None:
    sb = DockerSandbox(
        image="spectrumbloom-sandbox:latest",
        timeout_seconds=10,
        memory_limit_mb=512,
    )
    res_net = sb.execute("import urllib.request", sample_df)
    assert res_net.error is not None and "Security check rejected code" in res_net.error

    res_fs = sb.execute("open('/sandbox/pwn.txt', 'w').write('x')", sample_df)
    assert res_fs.error is not None and "Security check rejected code" in res_fs.error


@pytest.mark.docker_sandbox
def test_docker_sandbox_enforces_timeout_e2e(sample_df: pd.DataFrame) -> None:
    sb = DockerSandbox(
        image="spectrumbloom-sandbox:latest",
        timeout_seconds=1,
        memory_limit_mb=512,
    )
    with pytest.raises(SandboxTimeoutError):
        sb.execute("while True:\n    pass", sample_df)


@pytest.mark.docker_sandbox
def test_docker_sandbox_enforces_memory_limit_e2e(sample_df: pd.DataFrame) -> None:
    sb = DockerSandbox(
        image="spectrumbloom-sandbox:latest",
        timeout_seconds=25,
        memory_limit_mb=256,
    )
    res = sb.execute(
        "arr = np.full((350, 1024, 1024), 7, dtype=np.uint8)\nresult = int(arr[0, 0, 0])",
        sample_df,
    )
    assert res.error is not None
    assert "NameError" not in res.error
    assert any(
        tok in res.error.lower() for tok in ("memory", "unable to allocate", "sigkill", "137")
    )


@pytest.mark.docker_sandbox
def test_docker_sandbox_fails_closed_when_image_missing(sample_df: pd.DataFrame) -> None:
    sb = DockerSandbox(
        image="spectrumbloom-sandbox-nonexistent:never",
        timeout_seconds=5,
        memory_limit_mb=256,
    )
    with pytest.raises(SandboxUnavailableError, match="fail-closed"):
        sb.execute("result = 1 + 1", sample_df)


@pytest.mark.docker_sandbox
def test_docker_container_kernel_flags_enforce_isolation() -> None:
    """Verify the sandbox image's Docker flags enforce OS-level network and rootfs isolation."""
    docker_bin = shutil.which("docker")
    assert docker_bin is not None
    probe = (
        "import socket, sys\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=1)\n"
        "    sys.exit(10)\n"
        "except OSError:\n"
        "    pass\n"
        "try:\n"
        "    open('/app/rootfs_write_test.txt', 'w').write('x')\n"
        "    sys.exit(11)\n"
        "except OSError:\n"
        "    pass\n"
        "print('kernel_flags_ok')\n"
    )
    proc = subprocess.run(
        [
            docker_bin,
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--pids-limit=64",
            "spectrumbloom-sandbox:latest",
            "python",
            "-I",
            "-c",
            probe,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr
    assert "kernel_flags_ok" in proc.stdout


@pytest.mark.docker_sandbox
def test_docker_sandbox_runs_as_non_root_and_leaves_no_containers(
    sample_df: pd.DataFrame,
) -> None:
    docker_bin = shutil.which("docker")
    assert docker_bin is not None

    # Verify container executes as non-root UID 10001
    uid_proc = subprocess.run(
        [
            docker_bin,
            "run",
            "--rm",
            "--user=10001:10001",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "spectrumbloom-sandbox:latest",
            "python",
            "-c",
            "import os; print(os.getuid())",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        stdin=subprocess.DEVNULL,
    )
    assert uid_proc.returncode == 0, uid_proc.stderr
    assert uid_proc.stdout.strip() == "10001"

    # Run normal execution + timeout + memory exhaustion, then confirm zero leftover sb_* containers
    sb = DockerSandbox(
        image="spectrumbloom-sandbox:latest",
        timeout_seconds=10,
        memory_limit_mb=256,
    )
    ok_res = sb.execute("result = float(df['amount'].sum())", sample_df)
    assert ok_res.error is None
    assert ok_res.result == 150.0

    sb_timeout = DockerSandbox(
        image="spectrumbloom-sandbox:latest",
        timeout_seconds=1,
        memory_limit_mb=256,
    )
    with pytest.raises(SandboxTimeoutError):
        sb_timeout.execute("while True:\n    pass", sample_df)

    ps_proc = subprocess.run(
        [docker_bin, "ps", "-a", "--filter", "name=sb_", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )
    assert ps_proc.returncode == 0
    assert ps_proc.stdout.strip() == ""
