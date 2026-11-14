"""Automated correctness evaluation and DeepEval integration (Ollama judge + deterministic)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from spectrumbloom.domain.schemas import ToolTraceItem
from spectrumbloom.eda.charts import validate_plotly_figure_payload

logger = logging.getLogger(__name__)


class OllamaDeepEvalJudge(DeepEvalBaseLLM):
    """DeepEval LLM judge adapter backed by local Ollama (no OpenAI key needed)."""

    def __init__(
        self, model: str = "qwen3:4b-instruct", base_url: str = "http://localhost:11434"
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self._client: Any = None

    def load_model(self) -> Any:
        if self._client is None:
            from langchain_ollama import ChatOllama

            self._client = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.0,
                num_ctx=4096,
                reasoning=False,
            )
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        client = self.load_model()
        resp = client.invoke(prompt)
        return str(getattr(resp, "content", resp))

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        client = self.load_model()
        resp = await client.ainvoke(prompt)
        return str(getattr(resp, "content", resp))

    def get_model_name(self) -> str:
        return f"ollama:{self.model_name}"


def evaluate_turn_deterministically(
    *,
    query: str,
    answer: str,
    tool_traces: list[ToolTraceItem],
    figures: list[dict[str, Any]],
    profile_context: str = "",
    status: str = "success",
) -> dict[str, Any]:
    """Run deterministic automated correctness & faithfulness checks on an agent turn."""
    total_tools = len(tool_traces)
    failed_tools = sum(1 for t in tool_traces if t.error)
    successful_tools = total_tools - failed_tools

    valid_figures = sum(1 for fig in figures if validate_plotly_figure_payload(fig)[0])

    non_empty_answer = bool(answer and answer.strip())
    no_leaked_think = "<think>" not in answer.lower() and "</think>" not in answer.lower()
    no_fabricated_chart_ref = not (
        not figures and bool(re.search(r"!\[[^\]]*\]\(|\bfig_\d+\b", answer, re.IGNORECASE))
    )

    # Check numeric grounding against successful tool outputs and dataset profile context
    grounded_ratio = 1.0
    if non_empty_answer and (total_tools > 0 or profile_context.strip()):
        trace_blob = " ".join(
            f"{t.stdout} {json.dumps(t.result, default=str)}" for t in tool_traces if not t.error
        )
        evidence_blob = " ".join(part for part in (trace_blob, profile_context, query) if part)
        answer_nums = re.findall(r"\b\d+(?:\.\d+)?\b", answer)
        evidence_nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", evidence_blob)]
        standard_thresholds = {0.05, 0.01} if total_tools > 0 else set()
        if answer_nums and evidence_blob.strip():
            matched = 0
            for raw_num in answer_nums:
                if raw_num in evidence_blob:
                    matched += 1
                    continue
                val = float(raw_num)
                if val in standard_thresholds or any(
                    abs(val - ev) <= max(0.05, 0.015 * abs(ev)) for ev in evidence_nums
                ):
                    matched += 1
            grounded_ratio = round(matched / len(answer_nums), 3)

    execution_pass_rate = round(successful_tools / total_tools, 3) if total_tools > 0 else 1.0
    figure_validity_rate = round(valid_figures / len(figures), 3) if figures else 1.0

    passed = (
        status == "success"
        and non_empty_answer
        and no_leaked_think
        and no_fabricated_chart_ref
        and figure_validity_rate == 1.0
        and (total_tools == 0 or successful_tools > 0)
    )

    return {
        "passed": passed,
        "status": status,
        "tool_calls_total": total_tools,
        "tool_calls_succeeded": successful_tools,
        "execution_pass_rate": execution_pass_rate,
        "figures_total": len(figures),
        "figures_valid": valid_figures,
        "figure_validity_rate": figure_validity_rate,
        "numeric_grounding_ratio": grounded_ratio,
        "no_leaked_think_tags": no_leaked_think,
        "no_fabricated_chart_ref": no_fabricated_chart_ref,
    }


def run_deepeval_faithfulness_check(
    *,
    query: str,
    answer: str,
    tool_evidence: list[str],
    model: str = "qwen3:4b-instruct",
    ollama_base_url: str = "http://localhost:11434",
) -> dict[str, Any]:
    """Evaluate EDA response grounding using DeepEval GEval with local Ollama judge."""
    judge = OllamaDeepEvalJudge(model=model, base_url=ollama_base_url)
    metric = GEval(
        name="EDA Groundedness",
        criteria=(
            "Determine whether the actual output accurately reflects the observed tool outputs "
            "and answers the data analysis query without inventing unsupported statistics."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        threshold=0.6,
        async_mode=False,
    )
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=tool_evidence or ["No tool context executed."],
    )
    score = metric.measure(test_case)
    return {
        "metric": "GEval:EDA Groundedness",
        "score": round(float(score), 4),
        "passed": bool(metric.is_successful()),
        "reason": getattr(metric, "reason", None),
        "judge_model": judge.get_model_name(),
    }
