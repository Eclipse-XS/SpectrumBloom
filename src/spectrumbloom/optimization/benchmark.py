"""Real quantization & local LLM code-generation benchmark runner."""

from __future__ import annotations

import argparse
import json
import logging
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

from spectrumbloom.agent.graph import extract_tool_calls_with_fallback
from spectrumbloom.core.config import get_settings
from spectrumbloom.eda.profiler import format_profile_for_prompt, profile_dataframe
from spectrumbloom.sandbox.ast_guard import validate_code_safety
from spectrumbloom.sandbox.service import SubprocessSandbox


def _build_codegen_benchmark_prompt(dataset_context: str, *, mode: str = "codegen") -> str:
    """Prompt tailored either to fenced Python codegen or python_repl tool calling."""
    mode_instruction = (
        "- Output ONLY executable Python code inside a single ```python ... ``` block."
        if mode == "codegen"
        else "- Always call the `python_repl` tool with valid Python code."
    )
    return f"""You are SpectrumBloom's Python EDA engine working on a pandas DataFrame `df`.

ACTIVE DATASET PROFILE:
{dataset_context}

INSTRUCTIONS:
{mode_instruction}
- Pre-imported modules in scope: `pd`, `np`, `px`, `go`, `pio`, `stats` (`scipy.stats`).
- Always assign the requested scalar/dict answer to `result`.
- When a chart is requested, create a Plotly figure `fig` and call `plotly_figures.append(fig)`.
- Never initialize `plotly_figures = []` and never call `fig.show()`.
- Never use `open()`, `os`, `sys`, `subprocess`, `socket`, `getattr()`, `eval()`,
  `df.eval()`, `df.query()`, `read_csv()`, or `to_csv()`. Use `df[df["col"] == ...]`."""


def _extract_code_from_ollama_message(message: dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        if fn.get("name") == "python_repl":
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return args
            if isinstance(args, dict) and args.get("code"):
                return str(args["code"])

    content = str(message.get("content") or "")

    # Reuse agent fallback extractor via dummy object
    class _MsgProxy:
        def __init__(self, text: str) -> None:
            self.content = text
            self.tool_calls = []

    extracted = extract_tool_calls_with_fallback(_MsgProxy(content))  # type: ignore[arg-type]
    if extracted and extracted[0].get("args", {}).get("code"):
        return str(extracted[0]["args"]["code"])
    return content


def _verify_assertion(
    assertion: dict[str, Any],
    df: pd.DataFrame,
    exec_result: Any,
    figures: list[dict[str, Any]],
) -> bool:
    kind = assertion["type"]
    if kind == "dict_keys_match_missing":
        expected_cols = {col for col in df.columns if df[col].isna().sum() > 0}
        if not isinstance(exec_result, dict):
            return False
        if set(exec_result.keys()) != expected_cols:
            return False
        return all(int(exec_result[c]) == int(df[c].isna().sum()) for c in expected_cols)

    if kind == "category_mean_revenue":
        expected = df.groupby("category")["revenue"].mean().to_dict()
        if not isinstance(exec_result, dict):
            return False
        return all(
            k in exec_result and abs(float(exec_result[k]) - float(v)) < 0.1
            for k, v in expected.items()
        )

    if kind == "pearson_unit_price_revenue":
        expected = float(df["unit_price"].corr(df["revenue"]))
        try:
            if isinstance(exec_result, dict):
                # Tolerate single-value dict if model wrapped float
                val = float(next(iter(exec_result.values())))
            else:
                val = float(exec_result)
            return abs(val - expected) < 0.01
        except Exception:
            return False

    if kind == "iqr_revenue_outliers":
        q1 = float(df["revenue"].quantile(0.25))
        q3 = float(df["revenue"].quantile(0.75))
        iqr = q3 - q1
        expected = int(((df["revenue"] < q1 - 1.5 * iqr) | (df["revenue"] > q3 + 1.5 * iqr)).sum())
        try:
            if isinstance(exec_result, dict):
                val = int(next(iter(exec_result.values())))
            else:
                val = int(exec_result)
            return val == expected
        except Exception:
            return False

    if kind == "electronics_web_count":
        expected = int(((df["category"] == "Electronics") & (df["channel"] == "Web")).sum())
        try:
            if isinstance(exec_result, dict):
                val = int(next(iter(exec_result.values())))
            else:
                val = int(exec_result)
            return val == expected
        except Exception:
            return False

    if kind == "total_revenue_with_figure":
        expected = float(df["revenue"].sum())
        if not figures:
            return False
        try:
            val = (
                float(next(iter(exec_result.values())))
                if isinstance(exec_result, dict)
                else float(exec_result)
            )
            return abs(val - expected) < 1.0
        except Exception:
            return False

    if kind == "figure_only":
        if not figures:
            return False
        try:
            parsed = json.loads(figures[0]["figure_json"])
            return isinstance(parsed.get("data"), list) and len(parsed["data"]) > 0
        except Exception:
            return False

    if kind == "kruskal_region_revenue":
        groups = [grp["revenue"].to_numpy() for _, grp in df.groupby("region")]
        stat_exp, p_exp = stats.kruskal(*groups)
        if not isinstance(exec_result, dict):
            return False
        vals = [float(v) for v in exec_result.values() if isinstance(v, (int, float))]
        if len(vals) < 2:
            return False
        return any(abs(v - float(stat_exp)) < 0.05 for v in vals) and any(
            abs(v - float(p_exp)) < 0.01 for v in vals
        )

    return False


logger = logging.getLogger(__name__)


def _query_nvidia_vram_mb() -> float | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        out = subprocess.check_output(  # noqa: S603
            [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def _query_ollama_ps_vram(ollama_base_url: str, model: str) -> dict[str, Any]:
    try:
        resp = httpx.get(f"{ollama_base_url.rstrip('/')}/api/ps", timeout=3.0)
        if resp.status_code == 200:
            for item in resp.json().get("models", []):
                if item.get("name") == model or item.get("model") == model:
                    return {
                        "size_mb": round(item.get("size", 0) / (1024 * 1024), 1),
                        "size_vram_mb": round(item.get("size_vram", 0) / (1024 * 1024), 1),
                    }
    except Exception as exc:
        logger.debug("Ollama /api/ps unavailable: %s", exc)
    return {"size_mb": None, "size_vram_mb": None}


def _ollama_model_details(ollama_base_url: str, model: str) -> dict[str, Any]:
    try:
        resp = httpx.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            for item in resp.json().get("models", []):
                if item.get("name") == model:
                    details = item.get("details") or {}
                    return {
                        "disk_size_mb": round(item.get("size", 0) / (1024 * 1024), 1),
                        "parameter_size": details.get("parameter_size"),
                        "quantization_level": details.get("quantization_level"),
                        "family": details.get("family"),
                    }
    except Exception as exc:
        logger.debug("Ollama /api/tags unavailable: %s", exc)
    return {}


def benchmark_ollama_model(
    *,
    model: str,
    df: pd.DataFrame,
    tasks: list[dict[str, Any]],
    system_prompt: str,
    ollama_base_url: str,
    num_ctx: int = 4096,
    think: bool = False,
    mode: str = "codegen",
) -> dict[str, Any]:
    """Run standardized code-gen benchmark against a single Ollama model/quantization."""
    sandbox = SubprocessSandbox(timeout_seconds=15, memory_limit_mb=512)

    tools_payload = (
        [
            {
                "type": "function",
                "function": {
                    "name": "python_repl",
                    "description": (
                        "Execute Python EDA code on df. Assign output to `result` "
                        "and append Plotly charts to `plotly_figures`."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "thoughts": {"type": "string"},
                        },
                        "required": ["code"],
                    },
                },
            }
        ]
        if mode == "tool_calling"
        else None
    )

    # Warm-up request to separate model load time from steady-state inference latency
    warm_start = time.perf_counter()
    baseline_vram_mb = _query_nvidia_vram_mb()
    httpx.post(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": False,
            "think": think,
            "options": {"num_ctx": num_ctx, "num_predict": 16, "temperature": 0.0},
        },
        timeout=120.0,
    ).raise_for_status()
    warmup_ms = round((time.perf_counter() - warm_start) * 1000.0, 2)
    ps_info = _query_ollama_ps_vram(ollama_base_url, model)
    loaded_vram_mb = _query_nvidia_vram_mb()
    details = _ollama_model_details(ollama_base_url, model)

    per_task: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    throughputs_tps: list[float] = []
    ast_passed = 0
    sandbox_passed = 0
    assertion_passed = 0

    for task in tasks:
        t0 = time.perf_counter()
        chat_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task["prompt"]},
            ],
            "stream": False,
            "think": think,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": 768,
                "temperature": 0.0,
            },
        }
        if tools_payload is not None:
            chat_payload["tools"] = tools_payload

        resp = httpx.post(
            f"{ollama_base_url.rstrip('/')}/api/chat",
            json=chat_payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        gen_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies_ms.append(gen_latency_ms)

        eval_count = int(data.get("eval_count") or 0)
        eval_duration_ns = int(data.get("eval_duration") or 0)
        tps = (
            round(eval_count / (eval_duration_ns / 1e9), 2)
            if eval_count > 0 and eval_duration_ns > 0
            else 0.0
        )
        if tps > 0:
            throughputs_tps.append(tps)

        raw_code = _extract_code_from_ollama_message(data.get("message") or {})
        ast_ok = False
        try:
            clean_code, _ = validate_code_safety(raw_code)
            ast_ok = True
            ast_passed += 1
        except Exception:
            clean_code = raw_code

        sb_res = sandbox.execute(clean_code, df)
        sb_ok = sb_res.error is None
        if sb_ok:
            sandbox_passed += 1

        task_ok = False
        if sb_ok:
            task_ok = _verify_assertion(
                task["assertion"],
                df,
                sb_res.result,
                sb_res.figures,
            )
            if task_ok:
                assertion_passed += 1

        per_task.append(
            {
                "task_id": task["id"],
                "ast_valid": ast_ok,
                "sandbox_ok": sb_ok,
                "functional_pass": task_ok,
                "latency_ms": gen_latency_ms,
                "eval_tokens": eval_count,
                "tokens_per_sec": tps,
                "sandbox_peak_mb": sb_res.peak_memory_mb,
                "error": sb_res.error,
            }
        )

    n = len(tasks)
    vram_delta_mb = (
        round(loaded_vram_mb - baseline_vram_mb, 1)
        if loaded_vram_mb is not None and baseline_vram_mb is not None
        else None
    )
    return {
        "model": model,
        "think_mode": think,
        "num_ctx": num_ctx,
        "details": details,
        "memory": {
            "ollama_size_mb": ps_info["size_mb"],
            "ollama_size_vram_mb": ps_info["size_vram_mb"],
            "nvidia_smi_baseline_used_mb": baseline_vram_mb,
            "nvidia_smi_loaded_used_mb": loaded_vram_mb,
            "nvidia_smi_delta_mb": vram_delta_mb,
        },
        "summary": {
            "tasks_total": n,
            "ast_pass_rate": round(ast_passed / n, 3),
            "sandbox_pass_rate": round(sandbox_passed / n, 3),
            "functional_pass_rate": round(assertion_passed / n, 3),
            "warmup_ms": warmup_ms,
            "mean_latency_ms": round(statistics.mean(latencies_ms), 1),
            "median_latency_ms": round(statistics.median(latencies_ms), 1),
            "mean_tokens_per_sec": round(statistics.mean(throughputs_tps), 2)
            if throughputs_tps
            else 0.0,
        },
        "tasks": per_task,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    shape = report["dataset_shape"]
    lines = [
        "# SpectrumBloom Quantization & Code-Generation Benchmark",
        "",
        f"- Timestamp (UTC): `{report['timestamp']}`",
        f"- Dataset: `{report['dataset']}` ({shape[0]} rows x {shape[1]} cols)",
        f"- Tasks evaluated: `{report['tasks_count']}`",
        "",
        (
            "| Model | Quant | Disk (MB) | Ollama VRAM (MB) | AST Pass | Sandbox Pass | "
            "Functional Pass | Mean Latency (ms) | Throughput (tok/s) |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for res in report["results"]:
        d = res.get("details") or {}
        m = res.get("memory") or {}
        s = res.get("summary") or {}
        think_note = " (think)" if res.get("think_mode") else ""
        lines.append(
            f"| `{res['model']}{think_note}` | {d.get('quantization_level') or 'N/A'} | "
            f"{d.get('disk_size_mb') or 'N/A'} | {m.get('ollama_size_vram_mb') or 'N/A'} | "
            f"{s['ast_pass_rate'] * 100:.1f}% | {s['sandbox_pass_rate'] * 100:.1f}% | "
            f"{s['functional_pass_rate'] * 100:.1f}% | {s['mean_latency_ms']:.1f} | "
            f"{s['mean_tokens_per_sec']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark quantized local models on EDA code-gen")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3:4b-instruct"],
        help="Ollama model tags to benchmark.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("examples/sample_ecommerce.csv"))
    parser.add_argument("--tasks", type=Path, default=Path("benchmarks/eda_codegen_tasks.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--out-prefix", default="latest_benchmark")
    parser.add_argument(
        "--mode",
        choices=("codegen", "tool_calling"),
        default="codegen",
        help="Evaluate direct ```python``` codegen or native Ollama python_repl tool calling.",
    )
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode.")
    args = parser.parse_args()

    settings = get_settings()
    df = pd.read_csv(args.dataset)
    profile = profile_dataframe(df, filename=args.dataset.name)
    system_prompt = _build_codegen_benchmark_prompt(
        format_profile_for_prompt(profile, df), mode=args.mode
    )
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))

    results = []
    for model_tag in args.models:
        results.append(
            benchmark_ollama_model(
                model=model_tag,
                df=df,
                tasks=tasks,
                system_prompt=system_prompt,
                ollama_base_url=settings.ollama_base_url,
                num_ctx=args.num_ctx,
                think=args.think,
                mode=args.mode,
            )
        )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "dataset": str(args.dataset),
        "dataset_shape": list(df.shape),
        "tasks_count": len(tasks),
        "results": results,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.out_prefix}.json"
    md_path = args.out_dir / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    print(render_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
