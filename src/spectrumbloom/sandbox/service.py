"""Fail-closed sandbox execution service with OS-level process, memory, and time limits."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from spectrumbloom.core.config import Settings
from spectrumbloom.domain.exceptions import (
    SandboxSecurityError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from spectrumbloom.domain.schemas import SandboxExecutionResult
from spectrumbloom.sandbox.ast_guard import validate_code_safety

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
            return False
        proc_handle = getattr(proc, "_handle", None)
        if proc_handle is None:
            return False
        return bool(self._k32.AssignProcessToJobObject(self.handle, int(proc_handle)))

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


def _get_posix_rss_mb(pid: int) -> float | None:
    statm_path = Path(f"/proc/{pid}/statm")
    if not statm_path.exists():
        return None
    try:
        parts = statm_path.read_text(encoding="utf-8").split()
        rss_pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (rss_pages * page_size) / (1024 * 1024)
    except Exception:
        return None


def _serialize_df_for_sandbox(df: pd.DataFrame, sandbox_dir: Path) -> tuple[Path, str]:
    """Serialize DataFrame without pickle (Parquet first, safe JSON fallback)."""
    parquet_path = sandbox_dir / "dataset.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path, "parquet"
    except Exception:
        json_path = sandbox_dir / "dataset.json"
        safe_df = df.copy()
        for col in safe_df.columns:
            if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
                safe_df[col] = safe_df[col].astype(str)
        payload = {
            "columns": [str(c) for c in safe_df.columns],
            "dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
            "data": safe_df.where(pd.notna(safe_df), None).values.tolist(),
        }
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return json_path, "json"


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


def _run_watched_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    memory_limit_mb: int,
) -> tuple[int, str, str, float | None, bool, bool]:
    """Execute command with kernel Job Object (Windows) or setsid/RSS watchdog (POSIX)."""
    start = time.perf_counter()
    job_guard: _WindowsJobGuard | None = None
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}

    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
        popen_kwargs["creationflags"] = creationflags
        try:
            job_guard = _WindowsJobGuard(memory_limit_mb=memory_limit_mb)
        except OSError:
            job_guard = None
            popen_kwargs["creationflags"] = 0
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )

    if os.name == "nt" and job_guard is not None and creationflags:
        job_guard.assign(proc)
        # Resume main thread of suspended process
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Avoid CREATE_SUSPENDED thread-handle plumbing; AssignProcessToJobObject
        # executes before the child spawns Python.
        _ = k32

    peak_rss: float | None = None
    timed_out = False
    oom_killed = False

    try:
        while proc.poll() is None:
            if time.perf_counter() - start > timeout_seconds:
                timed_out = True
                if job_guard is not None:
                    job_guard.terminate()
                elif os.name != "nt":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                break

            if os.name != "nt":
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

            time.sleep(0.025)

        stdout, stderr = proc.communicate()
        if job_guard is not None:
            job_peak = job_guard.peak_memory_mb()
            if job_peak is not None:
                peak_rss = job_peak
    finally:
        if job_guard is not None:
            job_guard.close()

    return (
        proc.returncode if proc.returncode is not None else -1,
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
    """Windows runner avoiding CREATE_SUSPENDED deadlock by spawning and assigning immediately."""
    start = time.perf_counter()
    job_guard: _WindowsJobGuard | None = None
    try:
        job_guard = _WindowsJobGuard(memory_limit_mb=memory_limit_mb)
    except OSError:
        job_guard = None

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if job_guard is not None:
        job_guard.assign(proc)

    timed_out = False
    oom_killed = False
    peak_rss: float | None = None

    try:
        while proc.poll() is None:
            if time.perf_counter() - start > timeout_seconds:
                timed_out = True
                if job_guard is not None:
                    job_guard.terminate()
                else:
                    proc.kill()
                break
            if job_guard is not None:
                cur_peak = job_guard.peak_memory_mb()
                if cur_peak is not None:
                    peak_rss = cur_peak
            time.sleep(0.025)

        stdout, stderr = proc.communicate()
        if job_guard is not None:
            final_peak = job_guard.peak_memory_mb()
            if final_peak is not None:
                peak_rss = final_peak
    finally:
        if job_guard is not None:
            job_guard.close()

    return (
        proc.returncode if proc.returncode is not None else -1,
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
                _run_watched_windows_subprocess if os.name == "nt" else _run_watched_subprocess
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


class DockerSandbox:
    """Hardened non-privileged container-per-execution sandbox (fail-closed)."""

    backend_name = "docker"

    def __init__(
        self,
        *,
        image: str = "spectrumbloom-sandbox:latest",
        timeout_seconds: int = 15,
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

        with TemporaryDirectory(prefix="sb_docker_") as tmp_dir:
            sandbox_path = Path(tmp_dir)
            code_file = sandbox_path / "code.py"
            output_file = sandbox_path / "output.json"
            data_file, data_format = _serialize_df_for_sandbox(df, sandbox_path)
            code_file.write_text(clean_code, encoding="utf-8")

            command = [
                self.docker_bin,
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true",
                "--pids-limit=64",
                "--cpus=1",
                f"--memory={self.memory_limit_mb}m",
                f"--memory-swap={self.memory_limit_mb}m",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
                "-v",
                f"{sandbox_path}:/sandbox:rw",
                self.image,
                "python",
                "-I",
                "/app/src/spectrumbloom/sandbox/runner.py",
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
                    timeout=self.timeout_seconds + 5,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxTimeoutError(
                    f"Docker sandbox timed out after {self.timeout_seconds}s."
                ) from exc

            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if completed.returncode != 0 and not output_file.exists():
                raise SandboxUnavailableError(
                    "Docker sandbox backend failed to run (fail-closed; no fallback): "
                    + (completed.stderr.strip() or f"exit={completed.returncode}")
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
                )

            raise SandboxUnavailableError("Docker sandbox returned no output payload.")


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


def create_sandbox(settings: Settings) -> SubprocessSandbox | DockerSandbox | NsjailSandbox:
    """Instantiate the configured sandbox backend (fail-closed, zero silent fallback)."""
    common = {
        "timeout_seconds": settings.sandbox_timeout_seconds,
        "memory_limit_mb": settings.sandbox_memory_limit_mb,
        "max_output_chars": settings.sandbox_max_output_chars,
    }
    if settings.sandbox_backend == "strict_subprocess":
        return SubprocessSandbox(**common)
    if settings.sandbox_backend == "docker":
        return DockerSandbox(image=settings.sandbox_docker_image, **common)
    if settings.sandbox_backend == "nsjail":
        return NsjailSandbox(**common)
    raise SandboxUnavailableError(f"Unsupported sandbox backend: {settings.sandbox_backend}")
