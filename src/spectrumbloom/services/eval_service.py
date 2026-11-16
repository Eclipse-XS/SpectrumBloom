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
    prior_tool_traces: list[ToolTraceItem] | None = None,
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

    # Check numeric grounding against successful tool outputs (current + prior thread turns,
    # plus pairwise derived arithmetic) and query numbers.
    # NOTE: Do NOT include global `profile_context` numbers when tools were executed, because
    # matching an arbitrary number from the dataset profile is not provenance that a tool-computed
    # metric is grounded. Moreover, trace provenance measures consistency with tool outputs, not
    # standalone mathematical proof of query intent.
    grounded_ratio = 1.0
    if non_empty_answer:
        trace_blob = " ".join(
            f"{t.stdout} {json.dumps(t.result, default=str)}" for t in tool_traces if not t.error
        )
        prior_blob = " ".join(
            f"{t.stdout} {json.dumps(t.result, default=str)}"
            for t in (prior_tool_traces or [])
            if not t.error
        )
        _sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
        _sci_with_coeff = re.compile(
            r"([-+]?\d+(?:\.\d+)?)\s*[×xX*·⋅]\s*10"
            r"(?:(?:\*\*|\^)\s*([-+]?\d+)|([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+))"
        )
        _sci_bare_ten = re.compile(
            r"(?<![A-Za-z0-9_.])10"
            r"(?:(?:\*\*|\^)\s*([-+]?\d+)|([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+))"
        )
        num_pattern = (
            r"(?<![A-Za-z0-9_.₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹])"
            r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
            r"(?![A-Za-z0-9_₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]|\.\d)"
        )
        _sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

        def _extract_floats(blob: str) -> list[float]:
            # Normalize chemical/math subscripts attached to words (e.g. CO₂ -> CO2) so they are
            # treated as part of the identifier rather than standalone numbers.
            normalized_blob = re.sub(
                r"([A-Za-z_])([₀₁₂₃₄₅₆₇₈₉]+)",
                lambda m: f"{m.group(1)}{m.group(2).translate(_sub_map)}",
                blob,
            )
            normalized_blob = _sci_with_coeff.sub(
                lambda m: f"{m.group(1)}e{(m.group(2) or m.group(3)).translate(_sup_map)}",
                normalized_blob,
            )
            normalized_blob = _sci_bare_ten.sub(
                lambda m: f"1e{(m.group(1) or m.group(2)).translate(_sup_map)}",
                normalized_blob,
            )
            vals: list[float] = []
            for tok in re.findall(num_pattern, normalized_blob):
                try:
                    vals.append(float(tok.replace(",", "")))
                except ValueError:
                    continue
            return vals

        def _matches_evidence(val: float, ev: float) -> bool:
            abs_ev = abs(ev)
            if abs_ev == 0.0:
                return abs(val) <= 1e-6
            if abs_ev < 0.01:
                # Scientific notation / small p-values: allow 1-sig-fig rounding (~6% relative)
                return abs(val - ev) <= max(1e-9, 0.06 * abs_ev)
            if abs_ev < 1.0:
                # Correlations, proportions, small metrics: 2% relative or 0.005 absolute
                return abs(val - ev) <= max(0.005, 0.02 * abs_ev)
            return abs(val - ev) <= max(0.05, 0.015 * abs_ev)

        trace_nums = _extract_floats(trace_blob)
        prior_nums = _extract_floats(prior_blob)
        calc_base = (
            trace_nums + prior_nums if len(trace_nums) + len(prior_nums) <= 30 else trace_nums
        )
        derived_nums: list[float] = []
        if 2 <= len(calc_base) <= 30:
            for i in range(len(calc_base)):
                a = calc_base[i]
                for j in range(len(calc_base)):
                    if i == j:
                        continue
                    b = calc_base[j]
                    derived_nums.extend((a - b, a + b))
                    if abs(b) > 1e-12:
                        derived_nums.extend((a / b, (a / b) * 100.0, ((a - b) / b) * 100.0))

        evidence_nums = trace_nums + prior_nums + derived_nums + _extract_floats(query)
        standard_thresholds = {0.05, 0.01, 0.001} if (successful_tools > 0 or prior_nums) else set()
        # Strip standard significance/confidence phrases ("5% significance level", "95% confidence")
        # before extracting answer numbers so standard statistical conventions do not count as
        # ungrounded dataset metrics, while bare numbers like "5" or "95" still require evidence.
        answer_for_nums = re.sub(
            r"\b(?:1|5|10|90|95|99)(?:\.0+)?\s*%\s*(?:significance|confidence|alpha|level)\b",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        answer_nums = _extract_floats(answer_for_nums)
        if answer_nums:
            matched = 0
            for val in answer_nums:
                if val in standard_thresholds or any(
                    _matches_evidence(val, ev) for ev in evidence_nums
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
        and grounded_ratio >= 0.8
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
