"""Evaluation and fallback tool-call parsing tests."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from spectrumbloom.agent.graph import extract_tool_calls_with_fallback, normalize_text
from spectrumbloom.domain.schemas import ToolTraceItem
from spectrumbloom.services.eval_service import evaluate_turn_deterministically


def test_fallback_extracts_xml_and_fenced_python_blocks() -> None:
    xml_msg = AIMessage(
        content=(
            '<tool_call>\n{"name": "python_repl", '
            '"arguments": {"code": "result = 42"}}\n</tool_call>'
        )
    )
    extracted_xml = extract_tool_calls_with_fallback(xml_msg)
    assert len(extracted_xml) == 1
    assert extracted_xml[0]["name"] == "python_repl"
    assert extracted_xml[0]["args"]["code"] == "result = 42"

    fenced_msg = AIMessage(content="Let me calculate:\n```python\nresult = len(df)\n```")
    extracted_py = extract_tool_calls_with_fallback(fenced_msg)
    assert len(extracted_py) == 1
    assert extracted_py[0]["args"]["code"] == "result = len(df)"

    json_fence_msg = AIMessage(
        content='```json\n{"name": "python_repl", "arguments": {"code": "len(df)"}}\n```'
    )
    extracted_json = extract_tool_calls_with_fallback(json_fence_msg)
    assert len(extracted_json) == 1
    assert extracted_json[0]["name"] == "python_repl"
    assert extracted_json[0]["args"]["code"] == "len(df)"


def test_normalize_text_strips_think_blocks() -> None:
    raw = (
        "<think>internal chain of thought</think>\n"
        "The mean revenue is 412.5.\n\n"
        "![Chart](https://i.imgur.com/fake.png)\n"
        "![Inline](data:image/png;base64,iVBORw0KGgo=)"
    )
    assert normalize_text(raw) == "The mean revenue is 412.5."


def test_deterministic_turn_evaluator() -> None:
    traces = [
        ToolTraceItem(
            tool="python_repl",
            code="result = 747.6491935483871",
            stdout="747.6491935483871",
            result=747.6491935483871,
            error=None,
            figure_count=1,
            duration_ms=120.0,
        )
    ]
    figures = [{"id": "f1", "title": "Chart", "figure_json": json.dumps({"data": [{"x": [1]}]})}]
    report = evaluate_turn_deterministically(
        query="What is mean revenue?",
        answer="The mean revenue is 747.65.",
        tool_traces=traces,
        figures=figures,
    )
    assert report["passed"] is True
    assert report["numeric_grounding_ratio"] == 1.0
    assert report["execution_pass_rate"] == 1.0
    assert report["figure_validity_rate"] == 1.0
    assert report["numeric_grounding_ratio"] == 1.0
