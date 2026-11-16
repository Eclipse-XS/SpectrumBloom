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


def test_benchmark_verifier_never_rescues_from_traces_and_verifies_entity_pairing() -> None:
    import pandas as pd

    from spectrumbloom.optimization.workflow_benchmark import (
        _evaluate_tool_selection,
        _verify_task_against_ground_truth,
    )

    df = pd.DataFrame(
        {
            "region": ["East", "East", "West", "West"],
            "year": [1995, 1998, 2002, 2005],
            "score": [50.0, 50.0, 100.0, 100.0],
        }
    )

    # Answer missing numbers MUST fail even when traces contain exact values
    agg_task = {
        "id": "t_agg_all",
        "prompt": "Mean score by region",
        "assertion": {"op": "groupby_all", "group_by": "region", "metric": "score", "agg": "mean"},
    }
    fake_traces = [
        {"tool": "aggregate_data", "stdout": '{"East": 50.0, "West": 100.0}', "result": None}
    ]
    ok_missing, _ = _verify_task_against_ground_truth(
        agg_task, df, "Computed group averages cleanly.", "success", [], fake_traces, {}
    )
    assert ok_missing is False

    # Swapped entity-to-value associations MUST fail; matching table/bullets pass
    ok_swapped, _ = _verify_task_against_ground_truth(
        agg_task,
        df,
        "For East the mean is 100.0, and for West the mean is 50.0.",
        "success",
        [],
        [],
        {},
    )
    assert ok_swapped is False

    ok_table, _ = _verify_task_against_ground_truth(
        agg_task,
        df,
        "| Region | Mean Score |\n| --- | --- |\n| East | 50.0 |\n| West | 100.0 |",
        "success",
        [],
        [],
        {},
    )
    assert ok_table is True

    # Split means must keep entity pairing and anaphoric ', it is ...' clauses intact
    split_task = {
        "id": "t_split",
        "prompt": "Compare score before 2000 vs >= 2000",
        "assertion": {
            "op": "split_means",
            "column": "score",
            "split_col": "year",
            "split_val": 2000,
        },
    }
    ok_split_swap, _ = _verify_task_against_ground_truth(
        split_task,
        df,
        "For year < 2000 the mean is 100.0, while for year >= 2000 it is 50.0.",
        "success",
        [],
        [],
        {},
    )
    assert ok_split_swap is False

    ok_split_good, _ = _verify_task_against_ground_truth(
        split_task,
        df,
        "For year < 2000 it is 50.0, while for year ≥ 2000 (year > 1999), it is 100.0.",
        "success",
        [],
        [],
        {},
    )
    assert ok_split_good is True

    # group_test requires p-value
    gtest_task = {
        "id": "t_gtest",
        "prompt": "Run Mann-Whitney test",
        "assertion": {
            "op": "group_test",
            "test": "mannwhitney",
            "metric": "score",
            "group": "region",
            "group_a": "East",
            "group_b": "West",
        },
    }
    ok_no_p, _ = _verify_task_against_ground_truth(
        gtest_task, df, "There is a clear difference between East and West.", "success", [], [], {}
    )
    assert ok_no_p is False

    # Tool selection correctness checks deterministic routing
    assert _evaluate_tool_selection("filtered_count", [{"tool": "aggregate_data"}]) is True
    assert _evaluate_tool_selection("filtered_count", [{"tool": "python_repl"}]) is False
    assert _evaluate_tool_selection("filtered_count", []) is False


def test_deterministic_grounding_and_derived_arithmetic() -> None:
    # Hallucinated numerical answer with zero tool executions must fail grounding gate
    hallucinated = evaluate_turn_deterministically(
        query="What is mean revenue for West?",
        answer="The mean revenue for West is 44.34.",
        tool_traces=[],
        figures=[],
        profile_context="",
        status="success",
    )
    assert hallucinated["numeric_grounding_ratio"] == 0.0
    assert hallucinated["passed"] is False

    # Pairwise derived ratio/difference and comma-formatted trailing numbers pass
    traces = [
        ToolTraceItem(
            tool="aggregate_data",
            stdout="",
            result={"value": 1422.2863, "other": 558.8994, "gdp": 12319.57},
        )
    ]
    derived = evaluate_turn_deterministically(
        query="Compare ratio and difference",
        answer="Mean is 12,319.57. Ratio is 2.545 and difference is 863.39.",
        tool_traces=traces,
        figures=[],
        status="success",
    )
    assert derived["numeric_grounding_ratio"] == 1.0
    assert derived["passed"] is True

    # Cross-turn comparative citation grounded via prior_tool_traces
    turn1_trace = [ToolTraceItem(tool="aggregate_data", stdout="", result={"value": 20.59})]
    turn2_trace = [ToolTraceItem(tool="aggregate_data", stdout="", result={"value": 13.35})]
    cross_turn = evaluate_turn_deterministically(
        query="Now compare benign median against malignant",
        answer="Malignant was 20.59 vs benign 13.35 (difference 7.24).",
        tool_traces=turn2_trace,
        figures=[],
        prior_tool_traces=turn1_trace,
        status="success",
    )
    assert cross_turn["numeric_grounding_ratio"] == 1.0
    assert cross_turn["passed"] is True


def test_aggregate_and_stat_tool_fixes() -> None:
    import pandas as pd
    import pytest

    from spectrumbloom.agent.deterministic_tools import (
        format_statistical_focus,
        run_aggregate_data,
        run_statistical_test,
    )
    from spectrumbloom.agent.tools import AggregateDataInput
    from spectrumbloom.eda.profiler import format_profile_for_prompt, profile_dataframe
    from spectrumbloom.eda.statistics import compute_statistics

    parsed = AggregateDataInput(
        metric_column="val",
        agg_func="avg",
        filter_column="region",
        filter_op="in",
        filter_value=["East", "West"],
        limit=0,
    )
    assert parsed.agg_func == "mean"
    assert parsed.limit == 20

    df = pd.DataFrame(
        {
            "region": ["East", "East", "West", "West", "North", "North"],
            "year": [1998, 2001, 2002, 2003, 1995, 1996],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0, None],
        }
    )
    res_in = run_aggregate_data(
        df,
        metric_column="val",
        agg_func="mean",
        filter_column="region",
        filter_op="in",
        filter_value=["East", "West"],
    )
    assert res_in["rows_matched"] == 4
    assert res_in["total_dataset_rows"] == 6
    assert res_in["filters_applied"] == ["region in ['East', 'West']"]
    assert res_in["value"] == 25.0

    with pytest.raises(ValueError, match="logical AND"):
        run_aggregate_data(
            df,
            metric_column="val",
            agg_func="mean",
            filter_column="year",
            filter_op=">=",
            filter_value=2000,
            filter2_column="year",
            filter2_op="<",
            filter2_value=2000,
        )

    # Infer column_b cleanly for kruskal(group_a='region') and mannwhitney(group_a/group_b)
    kw = run_statistical_test(df, test_type="kruskal", column_a="val", group_a="region")
    assert kw["group_column"] == "region" and "p_value" in kw
    mw = run_statistical_test(
        df, test_type="mannwhitney", column_a="val", group_a="East", group_b="West"
    )
    assert mw["group_column"] == "region" and "p_value" in mw

    # Missing cell totals surfaced in both profile focus and prompt formatter
    prof = profile_dataframe(df, filename="t.csv")
    rep = compute_statistics(df, prof)
    focus = format_statistical_focus(df, prof, rep, focus="profile")
    assert focus["total_missing_cells"] == 1
    assert "Total missing cells: 1" in format_profile_for_prompt(prof, df)


def test_agent_graph_enforces_tools_filter_operators_and_rejects_spurious_filters() -> None:
    import pandas as pd
    from langchain_core.messages import AIMessage as LCAIMessage

    from spectrumbloom.agent.graph import EDAAgentGraph
    from spectrumbloom.eda.profiler import format_profile_for_prompt, profile_dataframe
    from spectrumbloom.eda.statistics import compute_statistics
    from spectrumbloom.sandbox.service import DisabledSandbox

    df = pd.DataFrame(
        {
            "region": ["East", "East", "West", "West"],
            "year": [1999, 2000, 2001, 2002],
            "revenue": [100.0, 200.0, 300.0, 400.0],
        }
    )
    prof = profile_dataframe(df, filename="sales.csv")
    rep = compute_statistics(df, prof)
    ctx = format_profile_for_prompt(prof, df)

    class _StrictAgentLLM:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            last_text = str(messages[-1].content)
            # Call 1: skip tools -> nudged (call 2) -> '>' fails -> spurious filter -> '>='
            if self.calls == 1:
                return LCAIMessage(content="Mean revenue is 44.34.")
            if self.calls == 2 and "You have not called any tool" in last_text:
                return LCAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc_bad_op",
                            "name": "aggregate_data",
                            "args": {
                                "metric_column": "revenue",
                                "agg_func": "mean",
                                "filter_column": "year",
                                "filter_op": ">",
                                "filter_value": 2000,
                            },
                        }
                    ],
                )
            if self.calls == 3:
                return LCAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc_bad_filter",
                            "name": "aggregate_data",
                            "args": {
                                "metric_column": "revenue",
                                "agg_func": "mean",
                                "filter_column": "region",
                                "filter_op": "in",
                                "filter_value": ["East"],
                            },
                        }
                    ],
                )
            if self.calls == 4:
                return LCAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc_good",
                            "name": "aggregate_data",
                            "args": {
                                "metric_column": "revenue",
                                "agg_func": "mean",
                                "filter_column": "year",
                                "filter_op": ">=",
                                "filter_value": 2000,
                            },
                        }
                    ],
                )
            return LCAIMessage(content="Mean revenue for year >= 2000 is 300.0.")

    graph = EDAAgentGraph(
        chat_model=_StrictAgentLLM(),
        sandbox=DisabledSandbox(),
        df_provider=lambda: df,
        dataset_context_provider=lambda: ctx,
        stats_provider=lambda: rep,
        profile_provider=lambda: prof,
    )
    res = graph.invoke(query="What is mean revenue where year >= 2000?", thread_id="t_strict")
    assert res["status"] == "success"
    assert res["retries"] == 2
    assert any("Filter operator mismatch" in (t.get("error") or "") for t in res["tool_traces"])
    assert any("Spurious filter rejected" in (t.get("error") or "") for t in res["tool_traces"])
    assert "300.0" in res["final_answer"]


def test_strip_markdown_preserves_snake_case_identifiers() -> None:
    from spectrumbloom.optimization.workflow_benchmark import _strip_markdown

    cleaned = _strip_markdown("**customer_age** (`class_0` and _italic_ *gdp_per_capita*)")
    assert "customer_age" in cleaned
    assert "class_0" in cleaned
    assert "gdp_per_capita" in cleaned
    assert "italic" in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned


def test_deterministic_eval_grounds_unicode_scientific_notation() -> None:
    traces = [
        ToolTraceItem(
            tool="compute_statistical_test",
            code="pearsonr(x, y)",
            stdout="",
            result={"statistic": 0.347955, "p_value": 2e-6},
            error=None,
            figure_count=0,
            duration_ms=20.0,
        )
    ]
    report = evaluate_turn_deterministically(
        query="What is Pearson r and p-value?",
        answer="Pearson r is 0.347955 with p = 2 × 10⁻⁶ (and threshold < 10⁻⁵).",
        tool_traces=traces
        + [
            ToolTraceItem(
                tool="python_repl",
                code="result = 1e-5",
                stdout="0.00001",
                result=1e-5,
                error=None,
                figure_count=0,
                duration_ms=5.0,
            )
        ],
        figures=[],
    )
    assert report["numeric_grounding_ratio"] == 1.0
    assert report["passed"] is True


def test_benchmark_regressions_for_evaluator_defects_and_analytical_guards() -> None:
    """Focused regression tests for P1 benchmark failures and evaluator defects."""
    from pathlib import Path

    import pandas as pd
    import pytest

    from spectrumbloom.agent.deterministic_tools import (
        format_statistical_focus,
        run_aggregate_data,
    )
    from spectrumbloom.eda.profiler import profile_dataframe
    from spectrumbloom.eda.statistics import compute_statistics
    from spectrumbloom.optimization.workflow_benchmark import (
        _extract_numbers_from_text,
        _verify_task_against_ground_truth,
    )

    datasets_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets"
    wine_df = pd.read_csv(datasets_dir / "wine_cultivars_uci.csv")
    macro_df = pd.read_csv(datasets_dir / "us_macro_fred.csv")
    co2_df = pd.read_csv(datasets_dir / "mauna_loa_co2.csv")

    # 1. wine_09_mt1: valid 'class_2 = 0.781' response with digit in entity name and ', with...'
    wine_09_task = {
        "id": "wine_09_mt1",
        "prompt": "Which cultivar has the lowest mean flavanoids, and what is that mean?",
        "assertion": {
            "op": "groupby_bottom",
            "group_by": "cultivar",
            "metric": "flavanoids",
            "agg": "mean",
        },
    }
    ok_wine09, reason_wine09 = _verify_task_against_ground_truth(
        wine_09_task,
        wine_df,
        (
            "The cultivar with the lowest mean flavanoids is **class_2**, "
            "with a mean flavanoids value of **0.781**."
        ),
        "success",
        [],
        [],
        {},
    )
    assert ok_wine09 is True, reason_wine09

    # Confirm wrong entity association for class_0 / class_2 still fails strictly
    ok_wine09_bad, _ = _verify_task_against_ground_truth(
        wine_09_task,
        wine_df,
        "The cultivar with the lowest mean flavanoids is class_0 (0.781), while class_2 is 2.982.",
        "success",
        [],
        [],
        {},
    )
    assert ok_wine09_bad is False

    # 2. macro_06_normality: correctly rounded scientific notation using superscript exponents
    assert _extract_numbers_from_text("p-value: 9 × 10⁻⁶") == [pytest.approx(9e-6)]
    macro_06_task = {
        "id": "macro_06_normality",
        "prompt": "Run a normality test on unemp and report its statistic and p-value.",
        "assertion": {"op": "normality", "column": "unemp"},
    }
    ok_macro06, reason_macro06 = _verify_task_against_ground_truth(
        macro_06_task,
        macro_df,
        (
            "The Shapiro-Wilk normality test on **unemp** yields:\n"
            "- **Test Statistic**: 0.95736\n- **p-value**: 9 × 10⁻⁶"
        ),
        "success",
        [],
        [],
        {},
    )
    assert ok_macro06 is True, reason_macro06

    # Confirm non-significant p-value (e.g. 0.004) does NOT pass when true p-value is ~8.9e-6
    ok_macro06_bad_p, _ = _verify_task_against_ground_truth(
        macro_06_task,
        macro_df,
        "Test Statistic: 0.95736, p-value: 0.004",
        "success",
        [],
        [],
        {},
    )
    assert ok_macro06_bad_p is False

    # 3. co2_01_profile: prompt only asks for rows and columns, not missing values
    co2_01_task = {
        "id": "co2_01_profile",
        "prompt": "How many rows and columns are in the Mauna Loa CO2 dataset?",
        "assertion": {"op": "shape_and_missing"},
    }
    ok_co2_01, reason_co2_01 = _verify_task_against_ground_truth(
        co2_01_task,
        co2_df,
        "The Mauna Loa CO2 dataset contains **2,284 rows** and **3 columns**.",
        "success",
        [],
        [],
        {},
    )
    assert ok_co2_01 is True, reason_co2_01

    # When prompt DOES ask for missing values, omitting missing count still fails
    co2_missing_req_task = {
        "id": "co2_01_with_missing",
        "prompt": "How many rows, columns, and missing values are in the Mauna Loa CO2 dataset?",
        "assertion": {"op": "shape_and_missing"},
    }
    ok_co2_missing_omitted, _ = _verify_task_against_ground_truth(
        co2_missing_req_task,
        co2_df,
        "The Mauna Loa CO2 dataset contains 2,284 rows and 3 columns.",
        "success",
        [],
        [],
        {},
    )
    assert ok_co2_missing_omitted is False

    # 4. wine_01_profile guard: nunique of the grouping column itself is rejected with guidance
    with pytest.raises(ValueError, match="agg_func='count'"):
        run_aggregate_data(
            wine_df,
            metric_column="cultivar",
            agg_func="nunique",
            group_by="cultivar",
        )

    # 5. wine_04_corr & wrong statistical method rejection:
    wine_04_task = {
        "id": "wine_04_corr",
        "prompt": "Compute the Spearman correlation between total_phenols and flavanoids.",
        "assertion": {
            "op": "corr",
            "method": "spearman",
            "col_a": "total_phenols",
            "col_b": "flavanoids",
        },
    }
    # Claiming Pearson when Spearman was requested fails
    ok_wrong_method_text, reason_wm = _verify_task_against_ground_truth(
        wine_04_task,
        wine_df,
        "The Pearson correlation is 0.883.",
        "success",
        [],
        [],
        {},
    )
    assert ok_wrong_method_text is False
    assert "wrong_statistical_method" in reason_wm

    # Calling Pearson in tool trace when Spearman was requested fails even if number matches
    ok_wrong_method_trace, reason_wmt = _verify_task_against_ground_truth(
        wine_04_task,
        wine_df,
        "The correlation is 0.883.",
        "success",
        [],
        [
            {
                "tool": "compute_statistical_test",
                "result": {"test_type": "pearson", "correlation": 0.864564},
                "error": None,
            }
        ],
        {},
    )
    assert ok_wrong_method_trace is False
    assert "wrong_statistical_method" in reason_wmt

    # Also verify format_statistical_focus includes pairwise correlations when columns are passed
    wine_prof = profile_dataframe(wine_df, filename="wine_cultivars_uci.csv")
    wine_rep = compute_statistics(wine_df, wine_prof)
    focused = format_statistical_focus(
        wine_df, wine_prof, wine_rep, focus="correlations", columns=["total_phenols", "flavanoids"]
    )
    assert len(focused["correlations"]) == 1
    assert focused["correlations"][0]["spearman_rho"] == pytest.approx(0.8794, abs=1e-4)

    # 6. hold_co2_04_diff guard: grouping by year on Mauna Loa CO2 (44 years) returns all 44 years
    # instead of truncating 1965 out at limit=20
    co2_by_year = run_aggregate_data(
        co2_df,
        metric_column="co2",
        agg_func="mean",
        group_by="year",
        limit=20,
    )
    assert "1965" in co2_by_year["mapping"]
    assert "1995" in co2_by_year["mapping"]
    diff_95_65 = co2_by_year["mapping"]["1995"] - co2_by_year["mapping"]["1965"]
    assert diff_95_65 == pytest.approx(40.865385, abs=1e-3)

    # 7. Numerical grounding audit: arbitrary number from profile_context does NOT ground tool turn
    ungrounded_from_profile = evaluate_turn_deterministically(
        query="What is the mean co2?",
        answer="The mean co2 is 522.0.",
        tool_traces=[
            ToolTraceItem(
                tool="aggregate_data",
                stdout="",
                result={"value": 360.8},
            )
        ],
        figures=[],
        profile_context="Dataset has 522 rows and 5 columns.",
        status="success",
    )
    assert ungrounded_from_profile["numeric_grounding_ratio"] == 0.0
    assert ungrounded_from_profile["passed"] is False
