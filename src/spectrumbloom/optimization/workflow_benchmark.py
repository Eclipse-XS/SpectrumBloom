"""End-to-end 60+ task agent workflow evaluation & RTX 3050 local LLM benchmark runner."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from scipy import stats

from spectrumbloom.core.config import Settings
from spectrumbloom.domain.exceptions import LLMInvocationError, LLMTimeoutError
from spectrumbloom.domain.schemas import CreateSessionRequest
from spectrumbloom.eda.charts import validate_plotly_figure_payload
from spectrumbloom.main import create_app
from spectrumbloom.services.session_service import SpectrumBloomService

_NUM_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _extract_numbers_from_text(text: str) -> list[float]:
    values: list[float] = []
    for tok in _NUM_TOKEN_RE.findall(text):
        cleaned = tok.replace(",", "")
        try:
            val = float(cleaned)
            if math.isfinite(val):
                values.append(val)
        except ValueError:
            continue
    return values


def _text_matches_number(
    text: str,
    expected: float | int,
    *,
    rel_tol: float = 0.02,
    abs_tol: float = 0.15,
) -> bool:
    nums = _extract_numbers_from_text(text)
    target = float(expected)
    if any(abs(v - target) <= max(abs_tol, rel_tol * abs(target)) for v in nums):
        return True
    # Also accept "< 0.001" / "<0.001" style summaries when exact p-value is near zero
    if 0.0 <= target < 0.001 and re.search(r"<\s*0\.0+[15]", text):
        return True
    return False


def _apply_filters(df: pd.DataFrame, filters: list[list[Any]]) -> pd.DataFrame:
    work = df
    for col, op, val in filters:
        series = work[col]
        if op == "==":
            if pd.api.types.is_numeric_dtype(series):
                work = work[pd.to_numeric(series, errors="coerce") == float(val)]
            else:
                work = work[series.astype(str) == str(val)]
        elif op == "!=":
            work = work[series.astype(str) != str(val)]
        elif op == ">":
            work = work[pd.to_numeric(series, errors="coerce") > float(val)]
        elif op == ">=":
            work = work[pd.to_numeric(series, errors="coerce") >= float(val)]
        elif op == "<":
            work = work[pd.to_numeric(series, errors="coerce") < float(val)]
        elif op == "<=":
            work = work[pd.to_numeric(series, errors="coerce") <= float(val)]
    return work


def _verify_task_against_ground_truth(
    task: dict[str, Any],
    df: pd.DataFrame,
    answer: str,
    status: str,
    figures: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> tuple[bool, str]:
    """Deterministically verify correctness against exact pandas/scipy calculations."""
    spec = task["assertion"]
    op = spec["op"]
    ans_lower = answer.lower()
    trace_blob = " ".join(
        f"{t.get('stdout', '')} {json.dumps(t.get('result'), default=str)}"
        for t in traces
        if not t.get("error")
    )
    combined_text = f"{answer}\n{trace_blob}"

    if not evaluation.get("no_leaked_think_tags", True):
        return False, "leaked_think_tags"
    if not evaluation.get("no_fabricated_chart_ref", True):
        return False, "fabricated_chart_reference"

    if op == "missing_column_handled":
        bad_col = str(spec["bad_column"]).lower()
        had_tool_error = any(bool(t.get("error")) for t in traces)
        err_markers = (
            "not found",
            "does not exist",
            "not exist",
            "no column",
            "missing column",
            "not present",
            "not in",
            "cannot",
            "unable",
            "unavailable",
            "failed",
            "error",
            "unknown column",
        )
        explained = any(m in ans_lower for m in err_markers) or bad_col in ans_lower
        if explained and (had_tool_error or status == "failed" or not traces):
            return True, "ok_error_handled"
        return False, f"unhandled_missing_column(had_tool_error={had_tool_error},status={status})"

    if status != "success":
        return False, f"status_failed:{status}"
    if evaluation.get("figure_validity_rate", 1.0) != 1.0:
        return False, "invalid_figure_spec"

    requires_tool_ops = {
        "filtered_count",
        "filtered_agg",
        "groupby_top",
        "groupby_bottom",
        "top_group_metric",
        "bottom_group_metric",
        "groupby_all",
        "corr",
        "outliers_iqr",
        "group_test",
        "normality",
        "group_ratio",
        "group_diff",
        "split_means",
        "chart",
    }
    if op in requires_tool_ops and not any(not t.get("error") for t in traces):
        return False, "no_successful_tool_execution"

    if op == "shape_and_dups":
        rows, cols = df.shape
        dups = int(df.duplicated().sum())
        ok = (
            _text_matches_number(combined_text, rows, rel_tol=0.0, abs_tol=0.5)
            and _text_matches_number(combined_text, cols, rel_tol=0.0, abs_tol=0.5)
            and _text_matches_number(combined_text, dups, rel_tol=0.0, abs_tol=0.5)
        )
        return ok, "ok" if ok else f"expected rows={rows}, cols={cols}, dups={dups}"

    if op == "shape_and_missing":
        rows, cols = df.shape
        miss = int(df.isna().sum().sum())
        ok = (
            _text_matches_number(combined_text, rows, rel_tol=0.0, abs_tol=0.5)
            and _text_matches_number(combined_text, cols, rel_tol=0.0, abs_tol=0.5)
            and (
                _text_matches_number(combined_text, miss, rel_tol=0.0, abs_tol=0.5)
                or (miss == 0 and ("no missing" in ans_lower or "zero missing" in ans_lower))
            )
        )
        return ok, "ok" if ok else f"expected rows={rows}, cols={cols}, miss={miss}"

    if op == "missing_map":
        nulls = df.isna().sum()
        missing_cols = {str(k): int(v) for k, v in nulls.items() if int(v) > 0}
        for col_name, cnt in missing_cols.items():
            if col_name.lower() not in ans_lower:
                return False, f"missing_column_name:{col_name}"
            if not _text_matches_number(combined_text, cnt, rel_tol=0.0, abs_tol=0.5):
                return False, f"missing_count_mismatch:{col_name}={cnt}"
        return True, "ok"

    if op == "col_null_count":
        col = spec["column"]
        cnt = int(df[col].isna().sum())
        ok = _text_matches_number(combined_text, cnt, rel_tol=0.0, abs_tol=0.5)
        return ok, "ok" if ok else f"expected {col}_nulls={cnt}"

    if op == "filtered_count":
        sub = _apply_filters(df, spec["filters"])
        cnt = int(len(sub))
        ok = _text_matches_number(combined_text, cnt, rel_tol=0.0, abs_tol=0.5)
        return ok, "ok" if ok else f"expected count={cnt}"

    if op == "filtered_agg":
        sub = _apply_filters(df, spec["filters"])
        series_num = pd.to_numeric(sub[spec["metric"]], errors="coerce").dropna()
        val = float(getattr(series_num, spec["agg"])())
        ok = _text_matches_number(combined_text, val, rel_tol=0.005, abs_tol=0.05)
        return ok, "ok" if ok else f"expected {spec['agg']}={round(val, 4)}"

    if op in {"groupby_top", "groupby_bottom"}:
        grouped = df.groupby(spec["group_by"], observed=True)[spec["metric"]]
        agg_series = grouped.count() if spec["agg"] == "count" else getattr(grouped, spec["agg"])()
        if op == "groupby_top":
            target_label = str(agg_series.idxmax())
            target_val = float(agg_series.max())
        else:
            target_label = str(agg_series.idxmin())
            target_val = float(agg_series.min())
        has_label = target_label.lower() in ans_lower
        has_val = _text_matches_number(combined_text, target_val, rel_tol=0.005, abs_tol=0.05)
        ok = has_label and has_val
        return ok, "ok" if ok else f"expected label={target_label}, val={round(target_val, 3)}"

    if op in {"top_group_metric", "bottom_group_metric"}:
        rank_grp = df.groupby(spec["group_by"], observed=True)[spec["rank_metric"]]
        rank_s = (
            rank_grp.count()
            if spec["rank_agg"] == "count"
            else getattr(rank_grp, spec["rank_agg"])()
        )
        winner = rank_s.idxmax() if op == "top_group_metric" else rank_s.idxmin()
        sub = df[df[spec["group_by"]] == winner]
        target_series = pd.to_numeric(sub[spec["target_metric"]], errors="coerce").dropna()
        val = float(getattr(target_series, spec["target_agg"])())
        ok = _text_matches_number(combined_text, val, rel_tol=0.005, abs_tol=0.05)
        return ok, "ok" if ok else f"expected {winner}.{spec['target_metric']}={round(val, 3)}"

    if op == "groupby_all":
        grouped = df.groupby(spec["group_by"], observed=True)[spec["metric"]]
        agg_series = grouped.count() if spec["agg"] == "count" else getattr(grouped, spec["agg"])()
        for label, val in agg_series.items():
            if str(label).lower() not in ans_lower:
                return False, f"missing_group_label:{label}"
            if not _text_matches_number(combined_text, float(val), rel_tol=0.005, abs_tol=0.05):
                return False, f"group_val_mismatch:{label}={round(float(val), 3)}"
        return True, "ok"

    if op == "col_stats":
        s = pd.to_numeric(df[spec["column"]], errors="coerce").dropna()
        for metric in spec["metrics"]:
            val = float(getattr(s, metric)())
            if not _text_matches_number(combined_text, val, rel_tol=0.005, abs_tol=0.05):
                return False, f"stat_mismatch:{spec['column']}.{metric}={round(val, 4)}"
        return True, "ok"

    if op == "corr":
        pair = pd.DataFrame(
            {
                "a": pd.to_numeric(df[spec["col_a"]], errors="coerce"),
                "b": pd.to_numeric(df[spec["col_b"]], errors="coerce"),
            }
        ).dropna()
        fn = stats.pearsonr if spec["method"] == "pearson" else stats.spearmanr
        r_val = float(fn(pair["a"].to_numpy(), pair["b"].to_numpy()).statistic)
        ok = _text_matches_number(combined_text, r_val, rel_tol=0.01, abs_tol=0.005)
        return ok, "ok" if ok else f"expected r={round(r_val, 4)}"

    if op == "outliers_iqr":
        s = pd.to_numeric(df[spec["column"]], errors="coerce").dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        cnt = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        ok = _text_matches_number(combined_text, cnt, rel_tol=0.0, abs_tol=0.5)
        return ok, "ok" if ok else f"expected iqr_outliers={cnt}"

    if op == "group_test":
        sub = pd.DataFrame(
            {
                "v": pd.to_numeric(df[spec["metric"]], errors="coerce"),
                "g": df[spec["group"]],
            }
        ).dropna()
        if spec["test"] == "mannwhitney":
            ga = sub[sub["g"].astype(str) == str(spec["group_a"])]["v"].to_numpy()
            gb = sub[sub["g"].astype(str) == str(spec["group_b"])]["v"].to_numpy()
            stat_v, p_v = stats.mannwhitneyu(ga, gb, alternative="two-sided")
        elif spec["test"] == "anova":
            groups = [grp["v"].to_numpy() for _, grp in sub.groupby("g", observed=True)]
            stat_v, p_v = stats.f_oneway(*groups)
        else:
            groups = [grp["v"].to_numpy() for _, grp in sub.groupby("g", observed=True)]
            stat_v, p_v = stats.kruskal(*groups)
        ok = _text_matches_number(combined_text, float(p_v), rel_tol=0.02, abs_tol=0.002) or (
            _text_matches_number(combined_text, float(stat_v), rel_tol=0.01, abs_tol=0.05)
        )
        return ok, "ok" if ok else f"expected stat={round(float(stat_v), 3)}, p={float(p_v):.4g}"

    if op == "normality":
        s = pd.to_numeric(df[spec["column"]], errors="coerce").dropna()
        stat_v, p_v = stats.shapiro(s) if len(s) <= 5000 else stats.normaltest(s)
        ok = _text_matches_number(combined_text, float(stat_v), rel_tol=0.01, abs_tol=0.02) and (
            _text_matches_number(combined_text, float(p_v), rel_tol=0.02, abs_tol=0.005)
        )
        return ok, "ok" if ok else f"expected stat={round(float(stat_v), 4)}, p={float(p_v):.4g}"

    if op == "group_ratio":
        g = df.groupby(spec["group_by"], observed=True)[spec["metric"]].mean()
        ratio = float(g[spec["num_group"]]) / float(g[spec["den_group"]])
        ok = _text_matches_number(combined_text, ratio, rel_tol=0.005, abs_tol=0.02)
        return ok, "ok" if ok else f"expected ratio={round(ratio, 3)}"

    if op == "group_diff":
        g = df.groupby(spec["group_col"], observed=True)[spec["metric"]].mean()
        diff = float(g[spec["val_a"]]) - float(g[spec["val_b"]])
        ok = _text_matches_number(combined_text, diff, rel_tol=0.005, abs_tol=0.05)
        return ok, "ok" if ok else f"expected diff={round(diff, 3)}"

    if op == "split_means":
        col_s = pd.to_numeric(df[spec["column"]], errors="coerce")
        sp_s = pd.to_numeric(df[spec["split_col"]], errors="coerce")
        m_hi = float(col_s[sp_s >= float(spec["split_val"])].mean())
        m_lo = float(col_s[sp_s < float(spec["split_val"])].mean())
        ok = _text_matches_number(combined_text, m_hi, rel_tol=0.005, abs_tol=0.05) and (
            _text_matches_number(combined_text, m_lo, rel_tol=0.005, abs_tol=0.05)
        )
        return ok, "ok" if ok else f"expected means hi={round(m_hi, 3)}, lo={round(m_lo, 3)}"

    if op == "chart":
        if not figures:
            return False, "no_figure_returned"
        expected_kinds = set(spec.get("expected_chart_kinds") or [])
        for fig in figures:
            valid, reason = validate_plotly_figure_payload(fig)
            if not valid:
                return False, f"invalid_figure:{reason}"
            kind = str(fig.get("chart_kind") or "").lower()
            parsed = json.loads(fig["figure_json"])
            trace_types = {str(t.get("type", "")).lower() for t in parsed.get("data", [])}
            if expected_kinds:
                type_match = (kind in expected_kinds) or bool(trace_types & expected_kinds)
                # Plotly box/violin/scatter/line variations share scatter trace with mode='lines'
                if "line" in expected_kinds and "scatter" in trace_types:
                    type_match = True
                if not type_match:
                    return False, f"chart_kind_mismatch:got={kind}/{trace_types}"
        return True, "ok"

    raise ValueError(f"Unsupported assertion op '{op}'.")


def _process_rss_mb() -> float | None:
    if os.name == "nt":

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wt.HANDLE
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        proc = kernel32.GetCurrentProcess()
        fn = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if fn is None:
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
        fn.restype = wt.BOOL
        if fn(proc, ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024 * 1024), 2)
        return None
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return round(kb / 1024.0, 2)
    return None


def _query_nvidia_vram_mb() -> float | None:
    nvidia_bin = shutil.which("nvidia-smi")
    if not nvidia_bin:
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            [nvidia_bin, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None
    return None


def _query_ollama_ps(ollama_base_url: str, model: str) -> dict[str, Any]:
    empty = {
        "model_reported": None,
        "size_mb": None,
        "size_vram_mb": None,
        "gpu_offload_pct": None,
    }
    try:
        resp = httpx.get(f"{ollama_base_url.rstrip('/')}/api/ps", timeout=5.0)
        resp.raise_for_status()
        for item in resp.json().get("models", []):
            name = str(item.get("name") or item.get("model") or "")
            if name == model or name.startswith(model.split(":")[0]):
                size_bytes = int(item.get("size") or 0)
                vram_bytes = int(item.get("size_vram") or 0)
                offload_pct = round(100.0 * vram_bytes / size_bytes, 1) if size_bytes > 0 else 0.0
                return {
                    "model_reported": name,
                    "size_mb": round(size_bytes / (1024 * 1024), 1),
                    "size_vram_mb": round(vram_bytes / (1024 * 1024), 1),
                    "gpu_offload_pct": offload_pct,
                }
    except Exception:
        return empty
    return empty


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(idx))
    hi = min(len(ordered) - 1, lo + 1)
    frac = idx - lo
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * frac, 2)


def benchmark_model_workflow(
    *,
    model: str,
    suite_path: Path,
    datasets_dir: Path,
    sandbox_backend: str = "docker",
    ollama_base_url: str = "http://127.0.0.1:11434",
    num_ctx: int = 4096,
    max_tokens: int = 1024,
    timeout_seconds: int = 90,
    enable_thinking: bool = False,
    limit_tasks: int = 0,
    repeats: int = 1,
) -> dict[str, Any]:
    """Execute the full SpectrumBloom agent workflow across the evaluation suite."""
    tasks = json.loads(suite_path.read_text(encoding="utf-8"))
    if limit_tasks > 0:
        tasks = tasks[:limit_tasks]

    baseline_vram = _query_nvidia_vram_mb()
    warm_t0 = time.perf_counter()
    httpx.post(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "OK"}],
            "stream": False,
            "think": enable_thinking,
            "options": {"num_ctx": num_ctx, "num_predict": 8, "temperature": 0.0},
        },
        timeout=float(timeout_seconds),
    ).raise_for_status()
    warmup_s = round(time.perf_counter() - warm_t0, 2)
    ps_info = _query_ollama_ps(ollama_base_url, model)
    loaded_vram = _query_nvidia_vram_mb()

    repeat_summaries: list[dict[str, Any]] = []
    temp_root = Path.cwd() / ".tmp_eval_sessions"
    temp_root.mkdir(exist_ok=True)

    for repeat_idx in range(1, repeats + 1):
        run_dir = temp_root / f"{model.replace(':', '_')}_r{repeat_idx}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        settings = Settings(
            app_env="test",
            local_dataset_dir=run_dir,
            sandbox_backend=sandbox_backend,  # type: ignore[arg-type]
            allow_trusted_local_subprocess=(sandbox_backend == "strict_subprocess"),
            llm_provider="ollama",
            llm_model=model,
            llm_temperature=0.0,
            llm_num_ctx=num_ctx,
            llm_max_tokens=max_tokens,
            llm_request_timeout_seconds=timeout_seconds,
            llm_enable_thinking=enable_thinking,
            ollama_base_url=ollama_base_url,
            sandbox_timeout_seconds=25,
            sandbox_memory_limit_mb=512,
        )
        app = create_app(settings, in_memory_telemetry=True)
        service: SpectrumBloomService = app.state.service

        # Group tasks cleanly by thread_group while preserving dataset order
        session_map: dict[tuple[str, str], str] = {}
        df_cache: dict[str, pd.DataFrame] = {}

        per_task: list[dict[str, Any]] = []
        latencies_s: list[float] = []
        total_tool_calls = 0
        succeeded_tool_calls = 0
        std_tool_calls = 0
        std_succeeded_tool_calls = 0
        tasks_with_retries = 0
        total_retries = 0
        timeouts = 0
        peak_rss_mb = _process_rss_mb() or 0.0
        peak_vram_mb = loaded_vram or 0.0

        for idx, task in enumerate(tasks, start=1):
            ds_name = task["dataset"]
            if ds_name not in df_cache:
                df_cache[ds_name] = pd.read_csv(datasets_dir / ds_name)
            df = df_cache[ds_name]

            thread_key = (ds_name, task["thread_group"])
            if thread_key not in session_map:
                sess = service.create_session(CreateSessionRequest(provider="ollama", model=model))
                csv_bytes = (datasets_dir / ds_name).read_bytes()
                service.upload_dataset(sess.session_id, ds_name, csv_bytes)
                session_map[thread_key] = sess.session_id

            sid = session_map[thread_key]
            t_start = time.perf_counter()
            error_detail: str | None = None
            timed_out = False
            status = "failed"
            answer = ""
            figures: list[dict[str, Any]] = []
            traces: list[dict[str, Any]] = []
            eval_dict: dict[str, Any] = {}
            retries = 0

            try:
                resp = service.run_query(sid, task["prompt"])
                status = resp.status
                error_detail = resp.error
                timed_out = resp.timed_out
                answer = resp.answer
                figures = list(resp.figures)
                traces = [t.model_dump() for t in resp.tool_traces]
                eval_dict = dict(resp.evaluation)
                retries = resp.retries_used
            except LLMTimeoutError as exc:
                timed_out = True
                error_detail = str(exc)
            except LLMInvocationError as exc:
                error_detail = str(exc)

            elapsed_s = round(time.perf_counter() - t_start, 2)
            latencies_s.append(elapsed_s)
            if timed_out:
                timeouts += 1

            total_retries += retries
            if retries > 0:
                tasks_with_retries += 1

            ok_cnt = sum(1 for t in traces if not t.get("error"))
            total_tool_calls += len(traces)
            succeeded_tool_calls += ok_cnt
            if task["category"] != "invalid_input":
                std_tool_calls += len(traces)
                std_succeeded_tool_calls += ok_cnt

            direct_profile_op = task["assertion"]["op"] in {
                "shape_and_dups",
                "shape_and_missing",
                "missing_column_handled",
            }
            if traces or eval_dict or (direct_profile_op and status == "success"):
                ok, reason = _verify_task_against_ground_truth(
                    task, df, answer, status, figures, traces, eval_dict
                )
            else:
                ok = False
                reason = error_detail or "no_tool_execution_or_response"

            cur_rss = _process_rss_mb() or 0.0
            cur_vram = _query_nvidia_vram_mb() or 0.0
            peak_rss_mb = max(peak_rss_mb, cur_rss)
            peak_vram_mb = max(peak_vram_mb, cur_vram)

            print(
                f"[{model} r{repeat_idx} {idx}/{len(tasks)}] {task['id']}: "
                f"{'PASS' if ok else 'FAIL'} ({elapsed_s}s, tools={len(traces)}, {reason})",
                flush=True,
            )

            per_task.append(
                {
                    "id": task["id"],
                    "dataset": ds_name,
                    "category": task["category"],
                    "turn_index": task["turn_index"],
                    "correct": ok,
                    "reason": reason,
                    "status": status,
                    "timed_out": timed_out,
                    "latency_s": elapsed_s,
                    "retries": retries,
                    "tool_calls": len(traces),
                    "tool_errors": sum(1 for t in traces if t.get("error")),
                    "tools_used": [t.get("tool") for t in traces],
                    "figure_count": len(figures),
                    "answer_preview": answer[:220],
                }
            )

        correct_count = sum(1 for t in per_task if t["correct"])
        by_category: dict[str, dict[str, Any]] = {}
        for item in per_task:
            cat = item["category"]
            bucket = by_category.setdefault(cat, {"total": 0, "passed": 0})
            bucket["total"] += 1
            if item["correct"]:
                bucket["passed"] += 1
        for cat_stats in by_category.values():
            cat_stats["accuracy_pct"] = round(100.0 * cat_stats["passed"] / cat_stats["total"], 1)

        repeat_summaries.append(
            {
                "repeat": repeat_idx,
                "tasks_total": len(per_task),
                "tasks_correct": correct_count,
                "functional_correctness_pct": round(100.0 * correct_count / len(per_task), 1)
                if per_task
                else 0.0,
                "tool_calls_total": total_tool_calls,
                "tool_calls_succeeded": succeeded_tool_calls,
                "tool_execution_success_pct": round(
                    100.0 * std_succeeded_tool_calls / std_tool_calls, 1
                )
                if std_tool_calls
                else 100.0,
                "tool_execution_all_tasks_pct": round(
                    100.0 * succeeded_tool_calls / total_tool_calls, 1
                )
                if total_tool_calls
                else 100.0,
                "tasks_with_retries": tasks_with_retries,
                "retry_rate_pct": round(100.0 * tasks_with_retries / len(per_task), 1)
                if per_task
                else 0.0,
                "total_retries": total_retries,
                "timeout_count": timeouts,
                "latency_mean_s": round(statistics.mean(latencies_s), 2) if latencies_s else 0.0,
                "latency_p50_s": _percentile(latencies_s, 50),
                "latency_p95_s": _percentile(latencies_s, 95),
                "peak_host_ram_mb": peak_rss_mb,
                "peak_vram_mb": peak_vram_mb,
                "by_category": by_category,
                "tasks": per_task,
            }
        )
        shutil.rmtree(run_dir, ignore_errors=True)

    repeat_comparison: dict[str, Any] | None = None
    if len(repeat_summaries) >= 2:
        r1, r2 = repeat_summaries[0], repeat_summaries[1]
        agree = sum(
            1 for a, b in zip(r1["tasks"], r2["tasks"], strict=True) if a["correct"] == b["correct"]
        )
        repeat_comparison = {
            "run1_correctness_pct": r1["functional_correctness_pct"],
            "run2_correctness_pct": r2["functional_correctness_pct"],
            "accuracy_delta_pp": round(
                r2["functional_correctness_pct"] - r1["functional_correctness_pct"], 1
            ),
            "p50_latency_delta_s": round(r2["latency_p50_s"] - r1["latency_p50_s"], 2),
            "task_pass_agreement_pct": round(100.0 * agree / len(r1["tasks"]), 1)
            if r1["tasks"]
            else 100.0,
        }

    return {
        "model": model,
        "sandbox_backend": sandbox_backend,
        "num_ctx": num_ctx,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "warmup_seconds": warmup_s,
        "baseline_vram_mb": baseline_vram,
        "loaded_vram_mb": loaded_vram,
        "ollama_ps": ps_info,
        "repeat_comparison": repeat_comparison,
        "runs": repeat_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full SpectrumBloom agent workflow evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "qwen2.5-coder:1.5b-instruct-q4_K_M",
            "qwen2.5-coder:3b-instruct-q4_K_M",
            "qwen3:4b-instruct",
        ],
    )
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/eval_suite_60.json"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("benchmarks/datasets"))
    parser.add_argument("--sandbox-backend", default="docker")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output-prefix",
        default="latest_workflow_benchmark",
        help="Stem under benchmarks/results/ for JSON & MD outputs.",
    )
    args = parser.parse_args()

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_reports: list[dict[str, Any]] = []
    for m in args.models:
        rep = benchmark_model_workflow(
            model=m,
            suite_path=args.suite,
            datasets_dir=args.datasets_dir,
            sandbox_backend=args.sandbox_backend,
            ollama_base_url=args.ollama_url,
            num_ctx=args.num_ctx,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            limit_tasks=args.limit_tasks,
            repeats=args.repeats,
        )
        model_reports.append(rep)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": str(args.suite),
        "sandbox_backend": args.sandbox_backend,
        "num_ctx": args.num_ctx,
        "models": model_reports,
    }
    json_path = out_dir / f"{args.output_prefix}.json"
    md_path = out_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# SpectrumBloom End-to-End Agent Workflow Benchmark",
        "",
        f"Generated: `{payload['generated_at']}` | Sandbox: `{args.sandbox_backend}` | "
        f"`num_ctx={args.num_ctx}`",
        "",
        "| Model | Run | Correctness | Tool Exec | Retry Rate | Timeouts | "
        "P50 (s) | P95 (s) | GPU Offload | Peak VRAM (MB) | Peak RAM (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mr in model_reports:
        ps = mr["ollama_ps"]
        off = f"{ps['gpu_offload_pct']}%" if ps.get("gpu_offload_pct") is not None else "n/a"
        for run in mr["runs"]:
            corr_str = (
                f"{run['tasks_correct']}/{run['tasks_total']} "
                f"({run['functional_correctness_pct']}%)"
            )
            md_lines.append(
                f"| `{mr['model']}` | {run['repeat']} | {corr_str} | "
                f"{run['tool_execution_success_pct']}% | {run['retry_rate_pct']}% | "
                f"{run['timeout_count']} | {run['latency_p50_s']} | {run['latency_p95_s']} | "
                f"{off} | {run['peak_vram_mb']} | {run['peak_host_ram_mb']} |"
            )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved workflow benchmark -> {json_path} and {md_path}")


if __name__ == "__main__":
    main()
