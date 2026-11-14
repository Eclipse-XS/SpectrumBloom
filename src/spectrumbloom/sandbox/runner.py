"""Isolated subprocess entrypoint for executing vetted EDA Python code.

This script intentionally avoids pickle deserialization and loads input tabular
data from Parquet (or JSON split fallback) before enforcing runtime network,
filesystem, and POSIX resource restrictions.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import datetime as dt
import io
import json
import math
import os
import signal
import sys
import time
import traceback
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

# Ensure standalone execution in `-I` mode can still resolve AST validator sibling
_THIS_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _THIS_DIR.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from spectrumbloom.sandbox.ast_guard import SandboxASTValidator, SandboxSecurityError  # noqa: E402

MAX_RESULT_CHARS = 50_000
MAX_ERROR_CHARS = 12_000


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    extra = len(text) - limit
    return f"{text[:limit]}\n... [truncated {extra} characters]"


def _json_safe(obj: Any) -> Any:
    import numpy as np
    import pandas as pd

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.head(200).to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _json_safe(obj.head(200).tolist())
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, dt.timedelta)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(item) for item in obj)
    if hasattr(obj, "__class__") and "interval" in obj.__class__.__name__.lower():
        return str(obj)
    return obj


def _serialize_figure(fig: Any, index: int) -> dict[str, Any]:
    title: str | None = None
    try:
        title_val = fig.layout.title.text
        if title_val:
            title = str(title_val)
    except Exception:
        title = None

    try:
        fig_dict = _json_safe(fig.to_dict())
        fig_json = json.dumps(fig_dict)
    except Exception:
        fig_json = json.dumps({"error": "Failed to serialize Plotly figure"})

    return {
        "id": f"fig_{index}",
        "title": title or f"Figure {index}",
        "figure_json": fig_json,
    }


def _normalize_result(value: Any) -> Any:
    if value is None:
        return None
    safe_val = _json_safe(value)
    try:
        encoded = json.dumps(safe_val)
    except TypeError:
        return str(safe_val)
    if len(encoded) <= MAX_RESULT_CHARS:
        return safe_val
    return {
        "truncated": True,
        "preview": encoded[:MAX_RESULT_CHARS],
    }


def _load_dataframe(data_file: Path, data_format: str):
    import pandas as pd

    if data_format == "parquet":
        return pd.read_parquet(data_file)
    if data_format == "json":
        payload = json.loads(data_file.read_text(encoding="utf-8"))
        df = pd.DataFrame(
            data=payload["data"],
            columns=payload["columns"],
        )
        for col, dtype_str in payload.get("dtypes", {}).items():
            if col in df.columns and "datetime" in dtype_str:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    raise ValueError(f"Unsupported sandbox dataset format: {data_format}")


def _disable_network() -> None:
    import socket

    class _BlockedSocket:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("Network sockets are disabled inside sandbox.")

    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Network access is disabled inside sandbox.")

    socket.socket = _BlockedSocket  # type: ignore[misc]
    for fn_name in (
        "socketpair",
        "create_connection",
        "create_server",
        "getaddrinfo",
        "gethostbyname",
        "gethostbyaddr",
        "fromfd",
    ):
        if hasattr(socket, fn_name):
            setattr(socket, fn_name, _blocked)


def _lock_filesystem() -> None:
    """Allow read-only access only under Python prefix; deny writes and user reads."""
    allowed_roots = [
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
    ]
    real_open = builtins.open
    real_io_open = io.open

    def _is_safe_library_read(file_arg: Any, mode: str) -> bool:
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            return False
        if isinstance(file_arg, int):
            return False
        try:
            target = Path(os.fspath(file_arg)).resolve()
        except Exception:
            return False
        return any(target.is_relative_to(root) for root in allowed_roots)

    def _guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if not _is_safe_library_read(file, mode):
            raise PermissionError("Filesystem access is disabled inside sandbox.")
        return real_open(file, mode, *args, **kwargs)

    def _guarded_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if not _is_safe_library_read(file, mode):
            raise PermissionError("Filesystem access is disabled inside sandbox.")
        return real_io_open(file, mode, *args, **kwargs)

    builtins.open = _guarded_open  # type: ignore[assignment]
    io.open = _guarded_io_open  # type: ignore[assignment]

    def _blocked_os(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("OS filesystem/process APIs are disabled inside sandbox.")

    for attr in ("open", "system", "popen", "remove", "unlink", "rmdir", "rename", "mkdir"):
        if hasattr(os, attr):
            setattr(os, attr, _blocked_os)


def _apply_posix_limits(timeout_seconds: int, memory_limit_mb: int) -> None:
    try:
        import resource
    except ImportError:
        return

    def _on_timeout(_signum: int, _frame: Any) -> None:
        os._exit(152)

    for sig_name in ("SIGXCPU", "SIGALRM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _on_timeout)

    if hasattr(signal, "alarm"):
        signal.alarm(max(1, timeout_seconds))

    # Include CPU spent importing heavy scientific modules before user execution.
    used_cpu = math.ceil(time.process_time())
    cpu_soft = used_cpu + max(1, timeout_seconds)
    cpu_hard = cpu_soft + 2
    file_bytes = 2 * 1024 * 1024  # small temp scratch cap

    # Virtual address space must accommodate OpenBLAS/PyArrow/Plotly shared
    # mappings (~1+ GiB VmSize even at ~230 MB RSS). Keep a high virtual guard
    # to stop multi-GB mmap bombs while the parent POSIX RSS watchdog enforces
    # physical memory_limit_mb.
    virtual_bytes = max(4096, memory_limit_mb * 8) * 1024 * 1024

    for limit_name, bounds in (
        ("RLIMIT_CPU", (cpu_soft, cpu_hard)),
        ("RLIMIT_FSIZE", (file_bytes, file_bytes)),
        ("RLIMIT_AS", (virtual_bytes, virtual_bytes)),
    ):
        limit_const = getattr(resource, limit_name, None)
        if limit_const is None:
            continue
        try:
            _, cur_hard = resource.getrlimit(limit_const)
            soft, hard = bounds
            if cur_hard != resource.RLIM_INFINITY:
                soft = min(soft, cur_hard)
                hard = min(hard, cur_hard)
            resource.setrlimit(limit_const, (soft, hard))
        except (OSError, ValueError):
            continue


def _build_safe_builtins() -> dict[str, Any]:
    from spectrumbloom.sandbox.ast_guard import is_allowed_import

    real_import = builtins.__import__

    def _safe_import(
        name: str,
        globals_dict: dict[str, Any] | None = None,
        locals_dict: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0 or not is_allowed_import(name):
            raise ImportError(f"Import '{name}' is not allowed inside sandbox.")
        return real_import(name, globals_dict, locals_dict, fromlist, level)

    return {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "Exception": Exception,
        "filter": filter,
        "float": float,
        "frozenset": frozenset,
        "int": int,
        "isinstance": isinstance,
        "KeyError": KeyError,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "RuntimeError": RuntimeError,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "ZeroDivisionError": ZeroDivisionError,
        "zip": zip,
    }


def _run(
    code: str,
    df: Any,
    timeout_seconds: int,
    memory_limit_mb: int,
    max_output_chars: int,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import scipy.stats as stats

    tree = ast.parse(code, mode="exec")
    SandboxASTValidator().visit(tree)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="_sb_last_expr", ctx=ast.Store())],
            value=tree.body[-1].value,
        )
        ast.fix_missing_locations(tree)
    compiled = compile(tree, "<spectrumbloom_sandbox>", "exec")

    _apply_posix_limits(timeout_seconds, memory_limit_mb)
    _disable_network()
    _lock_filesystem()

    stdout_buf = StringIO()
    env_vars: dict[str, Any] = {
        "__builtins__": _build_safe_builtins(),
        "__name__": "__main__",
        "df": df,
        "np": np,
        "pd": pd,
        "px": px,
        "go": go,
        "pio": pio,
        "stats": stats,
        "plotly_figures": [],
        "result": None,
        "_sb_last_expr": None,
    }

    with redirect_stdout(stdout_buf):
        try:
            exec(compiled, env_vars, env_vars)
        finally:
            if hasattr(signal, "alarm"):
                signal.alarm(0)

    last_val = env_vars.get("_sb_last_expr")
    if last_val is not None:
        if hasattr(last_val, "to_plotly_json"):
            if not env_vars.get("plotly_figures"):
                env_vars["plotly_figures"].append(last_val)
        elif env_vars.get("result") is None:
            env_vars["result"] = last_val

    figures = [
        _serialize_figure(fig, idx)
        for idx, fig in enumerate(env_vars.get("plotly_figures", []), start=1)
    ]
    return {
        "stdout": _truncate(stdout_buf.getvalue(), max_output_chars),
        "result": _normalize_result(env_vars.get("result")),
        "figures": figures,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SpectrumBloom isolated sandbox runner")
    parser.add_argument("--code-file", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--data-format", choices=("parquet", "json"), required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--memory-limit-mb", type=int, required=True)
    parser.add_argument("--max-output-chars", type=int, default=20000)
    args = parser.parse_args()

    output_file = Path(args.output_file)
    try:
        code = Path(args.code_file).read_text(encoding="utf-8")
        df = _load_dataframe(Path(args.data_file), args.data_format)
        payload = _run(
            code=code,
            df=df,
            timeout_seconds=args.timeout_seconds,
            memory_limit_mb=args.memory_limit_mb,
            max_output_chars=args.max_output_chars,
        )
    except SandboxSecurityError as exc:
        payload = {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": f"Security check rejected code: {exc}",
        }
    except Exception:
        payload = {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": _truncate(traceback.format_exc(), MAX_ERROR_CHARS),
        }

    # Note: we intentionally keep a handle via Path.write_text? Wait: _lock_filesystem()
    # replaced builtins.open/io.open AFTER _run returns only inside _run, and Path.write_text
    # calls io.open! So restore open? Wait: _lock_filesystem mutated builtins.open globally!
    # Therefore we write output using low-level os or BEFORE/WITHOUT breaking output writing.
    return _safe_write_output(output_file, payload)


# Preserve original file write capability strictly for the runner's own final JSON output
_REAL_OPEN = builtins.open


def _safe_write_output(output_file: Path, payload: dict[str, Any]) -> int:
    with _REAL_OPEN(output_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
