"""Execution-verified teacher-student knowledge distillation dataset builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from spectrumbloom.core.config import get_settings
from spectrumbloom.eda.profiler import format_profile_for_prompt, profile_dataframe
from spectrumbloom.llm.providers import UnifiedLLMProvider
from spectrumbloom.optimization.benchmark import _build_codegen_benchmark_prompt, _verify_assertion
from spectrumbloom.sandbox.service import SubprocessSandbox


def _extract_python_code(text: str) -> str:
    import re

    match = re.search(r"```(?:python|py)?\s*\n([\s\S]*?)\n```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def distill_verified_trajectories(
    *,
    dataset_path: Path,
    tasks_path: Path,
    output_jsonl: Path,
    teacher_provider: str = "ollama",
    teacher_model: str = "qwen3:4b-instruct",
) -> dict[str, Any]:
    """Generate and sandbox-verify SFT training pairs from teacher EDA completions."""
    settings = get_settings()
    df = pd.read_csv(dataset_path)
    profile = profile_dataframe(df, filename=dataset_path.name)
    system_prompt = _build_codegen_benchmark_prompt(format_profile_for_prompt(profile, df))

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    sandbox = SubprocessSandbox(timeout_seconds=15, memory_limit_mb=512)
    llm = UnifiedLLMProvider(settings).build_chat_model(
        provider=teacher_provider,
        model=teacher_model,
        temperature=0.2,
    )

    accepted_records: list[dict[str, Any]] = []
    rejected = 0

    for task in tasks:
        prompt = (
            f"Write ONLY Python code inside ```python ... ``` to solve: {task['prompt']}\n"
            "Remember: assign `result` and/or append Plotly figures to `plotly_figures`."
        )
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        raw_text = str(getattr(response, "content", response))
        code = _extract_python_code(raw_text)
        exec_res = sandbox.execute(code, df)

        if exec_res.error:
            rejected += 1
            continue

        if task.get("requires_figure") and not exec_res.figures:
            rejected += 1
            continue

        if task.get("assertion") and not _verify_assertion(
            task["assertion"], df, exec_res.result, exec_res.figures
        ):
            rejected += 1
            continue

        accepted_records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task["prompt"]},
                    {"role": "assistant", "content": f"```python\n{code}\n```"},
                ],
                "metadata": {
                    "task_id": task["id"],
                    "teacher": f"{teacher_provider}:{teacher_model}",
                    "sandbox_duration_ms": exec_res.duration_ms,
                },
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for rec in accepted_records:
            handle.write(json.dumps(rec) + "\n")

    return {
        "total_tasks": len(tasks),
        "accepted_verified": len(accepted_records),
        "rejected_failed_sandbox": rejected,
        "output_path": str(output_jsonl),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sandbox-verified distillation SFT dataset")
    parser.add_argument("--dataset", type=Path, default=Path("examples/sample_ecommerce.csv"))
    parser.add_argument("--tasks", type=Path, default=Path("benchmarks/eda_codegen_tasks.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/distilled_eda_sft.jsonl"))
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    args = parser.parse_args()

    summary = distill_verified_trajectories(
        dataset_path=args.dataset,
        tasks_path=args.tasks,
        output_jsonl=args.output,
        teacher_provider=args.provider,
        teacher_model=args.model,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
