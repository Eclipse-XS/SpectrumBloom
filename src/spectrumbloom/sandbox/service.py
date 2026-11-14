"""Fail-closed sandbox execution service with OS-level process, memory, and time limits."""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as wt
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import pandas as pd

from spectrumbloom.core.config import Settings
from spectrumbloom.domain.exceptions import (
    SandboxSecurityError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from spectrumbloom.domain.schemas import SandboxExecutionResult
from spectrumbloom.sandbox.ast_guard import validate_code_safety

_logger = logging.getLogger(__name__)
RUNNER_PATH = Path(__file__).with_name("runner.py")
NSJAIL_CFG_PATH = Path(__file__).with_name("nsjail.cfg")


class _IOCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wt.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wt.DWORD),
        ("SchedulingClass", wt.DWORD),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IOCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsJobGuard:
    """Enforces kernel-level Job Object memory, process-count, and process-tree kill on Windows."""

    def __init__(self, memory_limit_mb: int) -> None:
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateJobObjectW.argtypes = [wt.LPVOID, wt.LPCWSTR]
        self._k32.CreateJobObjectW.restype = wt.HANDLE
        self._k32.SetInformationJobObject.argtypes = [
            wt.HANDLE,
            ctypes.c_int,
            wt.LPVOID,
            wt.DWORD,
        ]
        self._k32.SetInformationJobObject.restype = wt.BOOL
        self._k32.QueryInformationJobObject.argtypes = [
            wt.HANDLE,
            ctypes.c_int,
            wt.LPVOID,
            wt.DWORD,
            ctypes.POINTER(wt.DWORD),
        ]
        self._k32.QueryInformationJobObject.restype = wt.BOOL
        self._k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        self._k32.AssignProcessToJobObject.restype = wt.BOOL
        self._k32.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
        self._k32.TerminateJobObject.restype = wt.BOOL
        self._k32.CloseHandle.argtypes = [wt.HANDLE]
        self._k32.CloseHandle.restype = wt.BOOL

        self.last_error: int = 0
        self.handle = self._k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(f"CreateJobObjectW failed: {ctypes.get_last_error()}")

        info = _JobObjectExtendedLimitInformation()
        # KILL_ON_JOB_CLOSE (0x2000) | JOB_MEMORY (0x0200) | ACTIVE_PROCESS (0x0008)
        info.BasicLimitInformation.LimitFlags = 0x2000 | 0x0200 | 0x0008
        info.BasicLimitInformation.ActiveProcessLimit = 4
        info.JobMemoryLimit = max(128, memory_limit_mb) * 1024 * 1024
        ok = self._k32.SetInformationJobObject(
            self.handle,
            9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.get_last_error()
            self._k32.CloseHandle(self.handle)
            self.handle = None
            raise OSError(f"SetInformationJobObject failed: {err}")

    def assign(self, proc: subprocess.Popen[str]) -> bool:
        if not self.handle:
            self.last_error = 6
            return False
        proc_handle = getattr(proc, "_handle", None)
        if proc_handle is None:
            self.last_error = 6
            return False
        ok = bool(self._k32.AssignProcessToJobObject(self.handle, int(proc_handle)))
        self.last_error = 0 if ok else ctypes.get_last_error()
        return ok

    def peak_memory_mb(self) -> float | None:
        if not self.handle:
            return None
        info = _JobObjectExtendedLimitInformation()
        ret_len = wt.DWORD()
        ok = self._k32.QueryInformationJobObject(
            self.handle,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(ret_len),
        )
        if not ok or info.PeakJobMemoryUsed == 0:
            return None
        return info.PeakJobMemoryUsed / (1024 * 1024)

    def terminate(self) -> None:
        if self.handle:
            self._k32.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None


def _read_proc_rss_mb(pid: int, page_size: int) -> float | None:
    statm_path = Path(f"/proc/{pid}/statm")
    if not statm_path.exists():
        return None
    try:
        parts = statm_path.read_text(encoding="utf-8").split()
        rss_pages = int(parts[1])
        return (rss_pages * page_size) / (1024 * 1024)
    except Exception:
        return None


def _get_posix_rss_mb(pid: int) -> float | None:
    """Measure resident physical memory (RSS) for pid and any child processes."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None

    root_rss = _read_proc_rss_mb(pid, page_size)
    if root_rss is None:
        return None

    total_rss = root_rss
    seen = {pid}
    queue = [pid]
    while queue:
        cur = queue.pop()
        task_dir = Path(f"/proc/{cur}/task")
        if not task_dir.exists():
            continue
        try:
            for child_file in task_dir.glob("*/children"):
                for token in child_file.read_text(encoding="utf-8").split():
                    child_pid = int(token)
                    if child_pid not in seen:
                        seen.add(child_pid)
                        queue.append(child_pid)
                        child_rss = _read_proc_rss_mb(child_pid, page_size)
                        if child_rss is not None:
                            total_rss += child_rss
        except (OSError, ValueError):
            break
    return total_rss


def _serialize_df_bytes_for_sandbox(df: pd.DataFrame) -> tuple[bytes, str]:
    """Serialize DataFrame to in-memory Parquet (or typed JSON fallback) without pickle."""
    buf = BytesIO()
    try:
        df.to_parquet(buf, index=False)
        return buf.getvalue(), "parquet"
    except Exception:
        safe_df = df.copy()
        for col in safe_df.columns:
            if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
                safe_df[col] = safe_df[col].astype(str)
        payload = {
            "columns": [str(c) for c in safe_df.columns],
            "dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
            "data": safe_df.where(pd.notna(safe_df), None).values.tolist(),
        }
        return json.dumps(payload).encode("utf-8"), "json"


def _serialize_df_for_sandbox(df: pd.DataFrame, sandbox_dir: Path) -> tuple[Path, str]:
    """Write DataFrame without pickle (Parquet first, safe JSON fallback)."""
    raw_bytes, fmt = _serialize_df_bytes_for_sandbox(df)
    target = sandbox_dir / f"dataset.{fmt}"
    target.write_bytes(raw_bytes)
    return target, fmt


def _minimal_sandbox_env(work_dir: Path) -> dict[str, str]:
    """Strip host credentials and restrict thread pools inside sandbox."""
    empty_home = work_dir / "home"
    empty_home.mkdir(exist_ok=True)
    env = {
        "PYTHONIOENCODING": "utf-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "HOME": str(empty_home),
        "USERPROFILE": str(empty_home),
    }
    if os.name == "nt":
        for var in ("SYSTEMROOT", "WINDIR"):
            val = os.environ.get(var)
            if val:
                env[var] = val
        env["TEMP"] = str(work_dir)
        env["TMP"] = str(work_dir)
    return env


def _run_watched_posix_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    memory_limit_mb: int,
) -> tuple[int, str, str, float | None, bool, bool]:
    """Execute command in its own POSIX session with physical RSS and wall-time watchdog."""
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    peak_rss: float | None = None
    timed_out = False
    oom_killed = False

    try:
        while proc.poll() is None:
            if time.perf_counter() - start > timeout_seconds:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break

            rss = _get_posix_rss_mb(proc.pid)
            if rss is not None:
                peak_rss = max(peak_rss or 0.0, rss)
                if rss > memory_limit_mb:
                    oom_killed = True
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break

            time.sleep(0.01)

        stdout, stderr = proc.communicate()
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise

    rc = proc.returncode if proc.returncode is not None else -1
    sigxcpu = getattr(signal, "SIGXCPU", None)
    if sigxcpu is not None and rc in (-sigxcpu, 128 + sigxcpu):
        timed_out = True

    return (
        rc,
        stdout or "",
        stderr or "",
        peak_rss,
        timed_out,
        oom_killed,
    )


def _run_watched_windows_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    memory_limit_mb: int,
) -> tuple[int, str, str, float | None, bool, bool]:
    """Windows runner with fail-closed Kernel Job Object memory and process limits."""
    start = time.perf_counter()
    try:
        job_guard = _WindowsJobGuard(memory_limit_mb=memory_limit_mb)
    except OSError as exc:
        raise SandboxUnavailableError(
            f"Windows Job Object initialization failed (fail-closed): {exc}"
        ) from exc

    proc: subprocess.Popen[str] | None = None
    timed_out = False
    oom_killed = False
    peak_rss: float | None = None

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not job_guard.assign(proc):
            err = job_guard.last_error
            proc.kill()
            proc.communicate()
            raise SandboxUnavailableError(
                "Failed to assign sandbox process to Windows Job Object "
                f"(winerror={err}; fail-closed)."
            )

        while proc.poll() is None:
            if time.perf_counter() - start > timeout_seconds:
                timed_out = True
                job_guard.terminate()
                break
            cur_peak = job_guard.peak_memory_mb()
            if cur_peak is not None:
                peak_rss = cur_peak
            time.sleep(0.01)

        stdout, stderr = proc.communicate()
        final_peak = job_guard.peak_memory_mb()
        if final_peak is not None:
            peak_rss = final_peak
    finally:
        job_guard.close()

    return (
        proc.returncode if proc is not None and proc.returncode is not None else -1,
        stdout or "",
        stderr or "",
        peak_rss,
        timed_out,
        oom_killed,
    )


class SubprocessSandbox:
    """Portable isolated subprocess sandbox with AST, network, and OS resource limits."""

    backend_name = "strict_subprocess"

    def __init__(
        self,
        *,
        timeout_seconds: int = 15,
        memory_limit_mb: int = 512,
        max_output_chars: int = 20000,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.max_output_chars = max_output_chars

    def execute(self, code: str, df: pd.DataFrame) -> SandboxExecutionResult:
        started = time.perf_counter()
        try:
            clean_code, _ = validate_code_safety(code)
        except SandboxSecurityError as exc:
            return SandboxExecutionResult(
                error=f"Security check rejected code: {exc}",
                backend=self.backend_name,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        if not RUNNER_PATH.exists():
            raise SandboxUnavailableError("Sandbox runner script is missing.")

        with TemporaryDirectory(prefix="sb_sandbox_") as tmp_dir:
            sandbox_path = Path(tmp_dir)
            code_file = sandbox_path / "code.py"
            output_file = sandbox_path / "output.json"
            data_file, data_format = _serialize_df_for_sandbox(df, sandbox_path)
            code_file.write_text(clean_code, encoding="utf-8")

            command = [
                sys.executable,
                "-I",
                str(RUNNER_PATH),
                "--code-file",
                str(code_file),
                "--data-file",
                str(data_file),
                "--data-format",
                data_format,
                "--output-file",
                str(output_file),
                "--timeout-seconds",
                str(self.timeout_seconds),
                "--memory-limit-mb",
                str(self.memory_limit_mb),
                "--max-output-chars",
                str(self.max_output_chars),
            ]

            runner_fn = (
                _run_watched_windows_subprocess
                if os.name == "nt"
                else _run_watched_posix_subprocess
            )
            rc, _stdout, stderr, peak_rss, timed_out, oom_killed = runner_fn(
                command,
                cwd=sandbox_path,
                env=_minimal_sandbox_env(sandbox_path),
                timeout_seconds=self.timeout_seconds,
                memory_limit_mb=self.memory_limit_mb,
            )

            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if timed_out:
                raise SandboxTimeoutError(
                    f"Execution exceeded time limit of {self.timeout_seconds}s."
                )
            if oom_killed:
                return SandboxExecutionResult(
                    error=(
                        f"Execution terminated: memory exceeded {self.memory_limit_mb} MB "
                        f"(observed peak {peak_rss:.1f} MB)."
                    ),
                    backend=self.backend_name,
                    duration_ms=duration_ms,
                    peak_memory_mb=round(peak_rss, 2) if peak_rss is not None else None,
                )

            if output_file.exists():
                payload = json.loads(output_file.read_text(encoding="utf-8"))
                return SandboxExecutionResult(
                    stdout=payload.get("stdout", "") or "",
                    result=payload.get("result"),
                    figures=payload.get("figures", []) or [],
                    error=payload.get("error"),
                    backend=self.backend_name,
                    duration_ms=duration_ms,
                    peak_memory_mb=round(peak_rss, 2) if peak_rss is not None else None,
                )

            return SandboxExecutionResult(
                error=stderr.strip() or f"Sandbox exited unexpectedly with code {rc}.",
                backend=self.backend_name,
                duration_ms=duration_ms,
                peak_memory_mb=round(peak_rss, 2) if peak_rss is not None else None,
            )


def _extract_framed_json(stdout_text: str, frame_token: str) -> dict[str, Any] | None:
    pattern = re.compile(
        rf"__SB_FRAME_{re.escape(frame_token)}_START__\s*(\{{[\s\S]*?\}})\s*__SB_FRAME_{re.escape(frame_token)}_END__"
    )
    match = pattern.search(stdout_text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class DockerSandbox:
    """Hardened non-privileged container-per-execution sandbox (fail-closed)."""

    backend_name = "docker"
    SANDBOX_UID = 10001
    SANDBOX_GID = 10001

    def __init__(
        self,
        *,
        image: str = "spectrumbloom-sandbox:latest",
        timeout_seconds: int = 25,
        memory_limit_mb: int = 512,
        max_output_chars: int = 20000,
    ) -> None:
        self.docker_bin = shutil.which("docker")
        if not self.docker_bin:
            raise SandboxUnavailableError("Docker CLI not found for docker sandbox backend.")
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.max_output_chars = max_output_chars

    def _inspect_oom_killed(self, container_name: str) -> bool:
        try:
            insp = subprocess.run(
                [self.docker_bin, "inspect", "-f", "{{.State.OOMKilled}}", container_name],
                capture_output=True,
                text=True,
                timeout=5,
                stdin=subprocess.DEVNULL,
            )
            return insp.returncode == 0 and insp.stdout.strip().lower() == "true"
        except Exception:
            return False

    def _cleanup_container(self, container_name: str) -> None:
        try:
            subprocess.run(
                [self.docker_bin, "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
        except Exception as exc:
            _logger.debug("Ignoring container cleanup error for %s: %s", container_name, exc)

    def execute(self, code: str, df: pd.DataFrame) -> SandboxExecutionResult:
        started = time.perf_counter()
        try:
            clean_code, _ = validate_code_safety(code)
        except SandboxSecurityError as exc:
            return SandboxExecutionResult(
                error=f"Security check rejected code: {exc}",
                backend=self.backend_name,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        data_bytes, data_format = _serialize_df_bytes_for_sandbox(df)
        return self.execute_serialized(
            clean_code=clean_code,
            data_bytes=data_bytes,
            data_format=data_format,
            started=started,
        )

    def execute_serialized(
        self,
        *,
        clean_code: str,
        data_bytes: bytes,
        data_format: str,
        started: float | None = None,
    ) -> SandboxExecutionResult:
        """Run pre-validated code inside an ephemeral mountless Docker container via stdio-json."""
        t0 = started if started is not None else time.perf_counter()
        clean_code, _ = validate_code_safety(clean_code)
        frame_token = secrets.token_hex(16)
        container_name = f"sb_{uuid.uuid4().hex[:12]}"
        envelope = json.dumps(
            {
                "code": clean_code,
                "data_format": data_format,
                "data_b64": base64.b64encode(data_bytes).decode("ascii"),
            }
        )

        command = [
            self.docker_bin,
            "run",
            "-i",
            "--init",
            "--name",
            container_name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            f"--user={self.SANDBOX_UID}:{self.SANDBOX_GID}",
            "--pids-limit=64",
            "--cpus=1",
            f"--memory={self.memory_limit_mb}m",
            f"--memory-swap={self.memory_limit_mb}m",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
            "-e",
            "HOME=/tmp",
            self.image,
            "python",
            "-I",
            "/app/src/spectrumbloom/sandbox/runner.py",
            "--stdio-json",
            "--frame-token",
            frame_token,
            "--timeout-seconds",
            str(self.timeout_seconds),
            "--memory-limit-mb",
            str(self.memory_limit_mb),
            "--max-output-chars",
            str(self.max_output_chars),
        ]

        try:
            try:
                completed = subprocess.run(
                    command,
                    input=envelope,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds + 8,
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxTimeoutError(
                    f"Docker sandbox timed out after {self.timeout_seconds}s."
                ) from exc

            duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            payload = _extract_framed_json(completed.stdout or "", frame_token)
            if payload is not None:
                return SandboxExecutionResult(
                    stdout=payload.get("stdout", "") or "",
                    result=payload.get("result"),
                    figures=payload.get("figures", []) or [],
                    error=payload.get("error"),
                    backend=self.backend_name,
                    duration_ms=duration_ms,
                )

            if completed.returncode in (124, 152):
                raise SandboxTimeoutError(
                    f"Docker sandbox exceeded CPU/time limit of {self.timeout_seconds}s."
                )
            if completed.returncode == 137:
                if self._inspect_oom_killed(container_name):
                    return SandboxExecutionResult(
                        error=(
                            f"Execution terminated: container memory exceeded "
                            f"{self.memory_limit_mb} MB limit "
                            "(cgroup OOMKilled / exit 137)."
                        ),
                        backend=self.backend_name,
                        duration_ms=duration_ms,
                    )
                if duration_ms >= self.timeout_seconds * 1000.0:
                    raise SandboxTimeoutError(
                        f"Docker sandbox exceeded CPU/time limit of {self.timeout_seconds}s "
                        "(SIGKILL)."
                    )
                raise SandboxUnavailableError(
                    "Docker sandbox container was killed unexpectedly (exit 137)."
                )
            if completed.returncode != 0:
                raise SandboxUnavailableError(
                    "Docker sandbox backend failed to run (fail-closed; no fallback): "
                    + ((completed.stderr or "").strip() or f"exit={completed.returncode}")
                )
            raise SandboxUnavailableError("Docker sandbox returned no output payload.")
        finally:
            self._cleanup_container(container_name)


class RemoteDockerSandbox:
    """Client delegating untrusted Python execution to an isolated sandbox-executor boundary."""

    backend_name = "remote"

    def __init__(
        self,
        *,
        executor_url: str,
        executor_token: str,
        timeout_seconds: int = 25,
        memory_limit_mb: int = 512,
        max_output_chars: int = 20000,
    ) -> None:
        self.executor_url = executor_url.rstrip("/")
        self.executor_token = executor_token.strip()
        if not self.executor_url.startswith(("http://", "https://")) or not self.executor_token:
            raise SandboxUnavailableError(
                "Remote sandbox backend requires SANDBOX_EXECUTOR_URL and SANDBOX_EXECUTOR_TOKEN."
            )
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.max_output_chars = max_output_chars

    def execute(self, code: str, df: pd.DataFrame) -> SandboxExecutionResult:
        started = time.perf_counter()
        try:
            clean_code, _ = validate_code_safety(code)
        except SandboxSecurityError as exc:
            return SandboxExecutionResult(
                error=f"Security check rejected code: {exc}",
                backend=self.backend_name,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        data_bytes, data_format = _serialize_df_bytes_for_sandbox(df)
        try:
            resp = httpx.post(
                f"{self.executor_url}/execute",
                json={
                    "code": clean_code,
                    "data_format": data_format,
                    "data_b64": base64.b64encode(data_bytes).decode("ascii"),
                    "timeout_seconds": self.timeout_seconds,
                    "memory_limit_mb": self.memory_limit_mb,
                    "max_output_chars": self.max_output_chars,
                },
                headers={"Authorization": f"Bearer {self.executor_token}"},
                timeout=float(self.timeout_seconds + 15),
            )
        except httpx.TimeoutException as exc:
            raise SandboxTimeoutError(
                f"Remote Docker sandbox exceeded timeout of {self.timeout_seconds}s."
            ) from exc
        except Exception as exc:
            raise SandboxUnavailableError(
                f"Remote sandbox executor unavailable (fail-closed; no fallback): {exc}"
            ) from exc

        if resp.status_code == 408:
            default_msg = f"Remote sandbox timed out after {self.timeout_seconds}s."
            raise SandboxTimeoutError(resp.json().get("detail", default_msg))
        if resp.status_code >= 500:
            raise SandboxUnavailableError(
                f"Remote sandbox executor error ({resp.status_code}): {resp.text}"
            )
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text) if resp.text else str(resp.status_code)
            raise SandboxUnavailableError(f"Remote sandbox rejected request: {detail}")

        body = resp.json()
        body["backend"] = self.backend_name
        return SandboxExecutionResult.model_validate(body)


class DisabledSandbox:
    """Fail-closed placeholder when dynamic Python execution is intentionally disabled."""

    backend_name = "disabled"

    def execute(self, code: str, df: pd.DataFrame) -> SandboxExecutionResult:
        raise SandboxUnavailableError(
            "Dynamic agent-generated Python execution is disabled in this deployment "
            "(SANDBOX_BACKEND=disabled). Deterministic EDA, statistical diagnostics, and ML "
            "baselines remain available."
        )


class NsjailSandbox:
    """Linux nsjail kernel-isolated sandbox (fail-closed; never falls back)."""

    backend_name = "nsjail"

    def __init__(
        self,
        *,
        timeout_seconds: int = 15,
        memory_limit_mb: int = 512,
        max_output_chars: int = 20000,
    ) -> None:
        self.nsjail_bin = shutil.which("nsjail")
        if os.name == "nt" or not self.nsjail_bin:
            raise SandboxUnavailableError(
                "nsjail backend requested, but nsjail binary is unavailable on this host."
            )
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.max_output_chars = max_output_chars

    def execute(self, code: str, df: pd.DataFrame) -> SandboxExecutionResult:
        started = time.perf_counter()
        try:
            clean_code, _ = validate_code_safety(code)
        except SandboxSecurityError as exc:
            return SandboxExecutionResult(
                error=f"Security check rejected code: {exc}",
                backend=self.backend_name,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        with TemporaryDirectory(prefix="sb_nsjail_") as tmp_dir:
            sandbox_path = Path(tmp_dir)
            code_file = sandbox_path / "code.py"
            output_file = sandbox_path / "output.json"
            runner_copy = sandbox_path / "runner.py"
            data_file, data_format = _serialize_df_for_sandbox(df, sandbox_path)
            code_file.write_text(clean_code, encoding="utf-8")
            shutil.copy2(RUNNER_PATH, runner_copy)

            command = [
                self.nsjail_bin,
                "--config",
                str(NSJAIL_CFG_PATH),
                "--bindmount",
                f"{sandbox_path}:/sandbox",
                "--",
                sys.executable,
                "-I",
                "/sandbox/runner.py",
                "--code-file",
                "/sandbox/code.py",
                "--data-file",
                f"/sandbox/{data_file.name}",
                "--data-format",
                data_format,
                "--output-file",
                "/sandbox/output.json",
                "--timeout-seconds",
                str(self.timeout_seconds),
                "--memory-limit-mb",
                str(self.memory_limit_mb),
                "--max-output-chars",
                str(self.max_output_chars),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds + 2,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxTimeoutError(
                    f"nsjail sandbox timed out after {self.timeout_seconds}s."
                ) from exc

            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if completed.returncode != 0 and not output_file.exists():
                raise SandboxUnavailableError(
                    "nsjail failed to execute sandboxed workload (fail-closed; no fallback): "
                    + (completed.stderr.strip() or f"exit={completed.returncode}")
                )

            payload = json.loads(output_file.read_text(encoding="utf-8"))
            return SandboxExecutionResult(
                stdout=payload.get("stdout", "") or "",
                result=payload.get("result"),
                figures=payload.get("figures", []) or [],
                error=payload.get("error"),
                backend=self.backend_name,
                duration_ms=duration_ms,
            )


def create_sandbox(
    settings: Settings,
) -> SubprocessSandbox | DockerSandbox | RemoteDockerSandbox | NsjailSandbox | DisabledSandbox:
    """Instantiate the configured sandbox backend (fail-closed, zero silent fallback)."""
    common = {
        "timeout_seconds": settings.sandbox_timeout_seconds,
        "memory_limit_mb": settings.sandbox_memory_limit_mb,
        "max_output_chars": settings.sandbox_max_output_chars,
    }
    if settings.sandbox_backend == "docker":
        return DockerSandbox(image=settings.sandbox_docker_image, **common)
    if settings.sandbox_backend == "remote":
        return RemoteDockerSandbox(
            executor_url=settings.sandbox_executor_url,
            executor_token=settings.sandbox_executor_token,
            **common,
        )
    if settings.sandbox_backend == "nsjail":
        return NsjailSandbox(**common)
    if settings.sandbox_backend == "strict_subprocess":
        if settings.app_env not in {"development", "test"} or not (
            settings.allow_trusted_local_subprocess
        ):
            raise SandboxUnavailableError(
                "strict_subprocess sandbox is restricted to trusted local development and "
                "requires APP_ENV=development|test and ALLOW_TRUSTED_LOCAL_SUBPROCESS=true."
            )
        return SubprocessSandbox(**common)
    if settings.sandbox_backend == "disabled":
        return DisabledSandbox()
    raise SandboxUnavailableError(f"Unsupported sandbox backend: {settings.sandbox_backend}")
