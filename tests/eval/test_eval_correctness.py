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
    figures = [
        {
            "id": "f1",
            "title": "Chart",
            "figure_json": json.dumps(
                {"data": [{"x": [1]}], "layout": {"title": {"text": "Chart"}}}
            ),
        }
    ]
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
    assert report["no_fabricated_chart_ref"] is True


def test_deterministic_eval_rejects_failed_status_and_fabricated_chart_ref() -> None:
    report_fail = evaluate_turn_deterministically(
        query="Show chart",
        answer="Here is ![chart](missing.png) with mean 999.99",
        tool_traces=[],
        figures=[],
        profile_context="rows=10, cols=2",
        status="success",
    )
    assert report_fail["no_fabricated_chart_ref"] is False
    assert report_fail["passed"] is False
    assert report_fail["numeric_grounding_ratio"] < 1.0

    report_status_fail = evaluate_turn_deterministically(
        query="Show chart",
        answer="Execution failed: sandbox unavailable",
        tool_traces=[],
        figures=[],
        status="failed",
    )
    assert report_status_fail["passed"] is False


def test_deterministic_eda_tools_and_chart_validation() -> None:
    import pandas as pd

    from spectrumbloom.agent.deterministic_tools import (
        format_statistical_focus,
        run_aggregate_data,
        run_statistical_test,
    )
    from spectrumbloom.eda.charts import build_standard_chart, validate_plotly_figure_payload
    from spectrumbloom.eda.profiler import profile_dataframe
    from spectrumbloom.eda.statistics import compute_statistics

    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "A", "B"],
            "metric": [10.0, 12.0, 14.0, 30.0, 32.0, 34.0, 11.0, 31.0],
            "other": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.5, 4.5],
        }
    )

    agg = run_aggregate_data(df, metric_column="metric", agg_func="mean", group_by="group")
    assert agg["mapping"]["B"] > agg["mapping"]["A"]
    groups = {r["group"]: r["value"] for r in agg["groups"]}
    assert groups["B"] > groups["A"]

    filtered = run_aggregate_data(
        df,
        metric_column="metric",
        agg_func="count",
        filter_column="group",
        filter_op="==",
        filter_value="A",
    )
    assert filtered["value"] == 4

    corr = run_statistical_test(df, test_type="pearson", column_a="metric", column_b="other")
    assert corr["correlation"] > 0.85

    mw = run_statistical_test(
        df, test_type="mannwhitney", column_a="metric", column_b="group", group_a="A", group_b="B"
    )
    assert mw["p_value"] < 0.05

    chart = build_standard_chart(
        df, chart_type="box", x="group", y="metric", title="Metric by Group"
    )
    ok, reason = validate_plotly_figure_payload(chart)
    assert ok is True, reason

    prof = profile_dataframe(df, filename="test.csv")
    rep = compute_statistics(df, prof)
    focus = format_statistical_focus(df, prof, rep, focus="profile")
    assert focus["rows"] == 8


def test_agent_graph_deduplicates_recovers_and_preserves_multi_turn_state() -> None:
    import pandas as pd
    from langchain_core.messages import AIMessage as LCAIMessage

    from spectrumbloom.agent.graph import EDAAgentGraph
    from spectrumbloom.eda.profiler import format_profile_for_prompt, profile_dataframe
    from spectrumbloom.eda.statistics import compute_statistics
    from spectrumbloom.sandbox.service import DisabledSandbox, SubprocessSandbox

    df = pd.DataFrame(
        {
            "segment": ["West", "East", "West", "East", "West", "East"],
            "revenue": [100.0, 200.0, 120.0, 220.0, 140.0, 240.0],
        }
    )
    prof = profile_dataframe(df, filename="sales.csv")
    rep = compute_statistics(df, prof)
    ctx = format_profile_for_prompt(prof, df)

    class _RepairingAndDuplicateLLM:
        def __init__(self) -> None:
            self.step = 0

        def bind_tools(self, tools):
            self.bound_names = [t.name for t in tools]
            return self

        def invoke(self, _messages):
            self.step += 1
            if self.step == 1:
                # Bad column name triggers bounded repair on first try
                return LCAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc1",
                            "name": "aggregate_data",
                            "args": {"metric_column": "wrong_col", "agg_func": "mean"},
                        }
                    ],
                )
            if self.step == 2:
                # Emits identical valid call TWICE in one batch -> duplicate must be suppressed
                valid_args = {"metric_column": "revenue", "agg_func": "mean", "group_by": "segment"}
                return LCAIMessage(
                    content="",
                    tool_calls=[
                        {"id": "tc2a", "name": "aggregate_data", "args": valid_args},
                        {"id": "tc2b", "name": "aggregate_data", "args": valid_args},
                    ],
                )
            if self.step == 3:
                return LCAIMessage(
                    content="East has higher mean revenue (220.0) than West (120.0)."
                )
            return LCAIMessage(content="Second turn confirmed East revenue mean is 220.0.")

    llm = _RepairingAndDuplicateLLM()
    graph = EDAAgentGraph(
        chat_model=llm,
        sandbox=SubprocessSandbox(timeout_seconds=10, memory_limit_mb=512),
        df_provider=lambda: df,
        dataset_context_provider=lambda: ctx,
        stats_provider=lambda: rep,
        profile_provider=lambda: prof,
    )
    assert "python_repl" in llm.bound_names

    res1 = graph.invoke(query="Mean revenue by segment?", thread_id="t_recover")
    assert res1["status"] == "success"
    assert res1["retries"] == 1
    # Only tc1 (failed) + tc2a (succeeded); tc2b was deduplicated
    assert len(res1["tool_traces"]) == 2
    assert "East" in res1["final_answer"]

    # Second turn on same thread works cleanly
    res2 = graph.invoke(query="Repeat the East figure", thread_id="t_recover")
    assert res2["status"] == "success"
    assert "220.0" in res2["final_answer"]

    # DisabledSandbox hides python_repl and fails closed cleanly without breaking thread
    class _ForbiddenPythonLLM:
        def __init__(self) -> None:
            self.turn = 0

        def bind_tools(self, tools):
            self.bound_names = [t.name for t in tools]
            return self

        def invoke(self, _messages):
            self.turn += 1
            if self.turn == 1:
                return LCAIMessage(
                    content="",
                    tool_calls=[{"id": "tc_py", "name": "python_repl", "args": {"code": "1+1"}}],
                )
            return LCAIMessage(content="Follow-up works fine.")

    llm_disabled = _ForbiddenPythonLLM()
    graph_disabled = EDAAgentGraph(
        chat_model=llm_disabled,
        sandbox=DisabledSandbox(),
        df_provider=lambda: df,
        dataset_context_provider=lambda: ctx,
        stats_provider=lambda: rep,
        profile_provider=lambda: prof,
    )
    assert "python_repl" not in llm_disabled.bound_names

    failed = graph_disabled.invoke(query="Run python", thread_id="t_disabled")
    assert failed["status"] == "failed"
    assert failed["terminal_error"] is not None

    recovered = graph_disabled.invoke(query="Just reply", thread_id="t_disabled")
    assert recovered["status"] == "success"
    assert "Follow-up works fine" in recovered["final_answer"]
