"""LangChain tool schemas bound to the EDA LangGraph agent."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class StatisticalSummaryInput(BaseModel):
    focus: Literal[
        "all",
        "profile",
        "missingness",
        "correlations",
        "outliers",
        "normality",
        "group_tests",
    ] = Field(
        default="all",
        description="Pre-computed EDA section to retrieve.",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Optional exact column names for untruncated column-level diagnostics.",
    )


class AggregateDataInput(BaseModel):
    metric_column: str | None = Field(
        default=None,
        description="Column name to aggregate or count. Omit for total row count.",
    )
    agg_func: Literal["count", "sum", "mean", "median", "min", "max", "std", "nunique"] = Field(
        default="count",
        description="Aggregation operation.",
    )
    group_by: str | list[str] | None = Field(
        default=None,
        description="Optional column name or list of columns to group results by.",
    )
    filter_column: str | None = Field(default=None, description="Optional first filter column.")
    filter_op: Literal["==", "!=", ">", ">=", "<", "<=", "isna", "notna", "contains"] | None = (
        Field(default=None, description="Optional first filter operator.")
    )
    filter_value: Any = Field(default=None, description="Value for first filter.")
    filter2_column: str | None = Field(default=None, description="Optional second filter column.")
    filter2_op: Literal["==", "!=", ">", ">=", "<", "<=", "isna", "notna", "contains"] | None = (
        Field(default=None, description="Optional second filter operator.")
    )
    filter2_value: Any = Field(default=None, description="Value for second filter.")
    sort_desc: bool = Field(default=True, description="Sort grouped values descending.")
    limit: int = Field(default=20, ge=1, le=50, description="Max groups to return.")


class StatisticalTestInput(BaseModel):
    test_type: Literal[
        "pearson",
        "spearman",
        "kruskal",
        "anova",
        "mannwhitney",
        "chi2",
        "outliers_iqr",
        "normality",
    ] = Field(..., description="Statistical test or diagnostic to compute deterministically.")
    column_a: str = Field(
        ...,
        description=("Primary numeric value column (or first categorical column for chi2)."),
    )
    column_b: str | None = Field(
        default=None,
        description=(
            "Second numeric column, categorical grouping column (for kruskal/anova/mannwhitney), "
            "or second categorical column (for chi2)."
        ),
    )
    group_a: str | None = Field(
        default=None,
        description="Optional first group label (for two-group mannwhitney).",
    )
    group_b: str | None = Field(
        default=None,
        description="Optional second group label (for two-group mannwhitney).",
    )


class StandardChartInput(BaseModel):
    chart_type: Literal["bar", "histogram", "box", "violin", "scatter", "line", "heatmap"] = Field(
        ...,
        description="Plotly chart type to generate cleanly with validated axes and title.",
    )
    x: str = Field(..., description="Exact column name for X axis (numeric column for heatmap).")
    y: str | None = Field(default=None, description="Optional exact column name for Y axis.")
    color: str | None = Field(default=None, description="Optional column name for color grouping.")
    agg_func: Literal["none", "count", "sum", "mean", "median"] = Field(
        default="none",
        description="Aggregation function for bar charts with both x and y.",
    )
    title: str = Field(default="", description="Descriptive chart title.")


class MLBaselineInput(BaseModel):
    target_column: str = Field(..., description="Exact column name in `df` to predict.")
    task_type: Literal["auto", "classification", "regression"] = Field(
        default="auto",
        description="Modeling task type or 'auto' to infer from target.",
    )


class PythonReplInput(BaseModel):
    code: str = Field(
        ...,
        description=(
            "Executable Python using pre-loaded `df`, `pd`, `np`, `px`, `go`, `pio`, `stats`. "
            "Use ONLY when deterministic tools cannot compute the custom calculation or chart. "
            "Assign output to `result` or `print(...)`; append charts via "
            "`plotly_figures.append(fig)`."
        ),
    )
    thoughts: str = Field(default="", description="Brief reasoning note for trace logging.")


def get_agent_tools(*, include_python_repl: bool = True):
    """Return tool schema objects for LLM `.bind_tools(...)`."""

    @tool("get_statistical_summary", args_schema=StatisticalSummaryInput)
    def statistical_summary_tool(focus: str = "all", columns: list[str] | None = None) -> str:
        """Retrieve pre-computed schema profile, missingness, correlations, outliers, or tests."""
        return ""

    @tool("aggregate_data", args_schema=AggregateDataInput)
    def aggregate_data_tool(
        metric_column: str | None = None,
        agg_func: str = "count",
        group_by: str | list[str] | None = None,
        filter_column: str | None = None,
        filter_op: str | None = None,
        filter_value: Any = None,
        filter2_column: str | None = None,
        filter2_op: str | None = None,
        filter2_value: Any = None,
        sort_desc: bool = True,
        limit: int = 20,
    ) -> str:
        """Filter rows and compute grouped or overall summary statistics deterministically."""
        return ""

    @tool("compute_statistical_test", args_schema=StatisticalTestInput)
    def statistical_test_tool(
        test_type: str,
        column_a: str,
        column_b: str | None = None,
        group_a: str | None = None,
        group_b: str | None = None,
    ) -> str:
        """Compute Pearson/Spearman, Kruskal/ANOVA/Mann-Whitney, Chi2, outliers, or normality."""
        return ""

    @tool("create_standard_chart", args_schema=StandardChartInput)
    def standard_chart_tool(
        chart_type: str,
        x: str,
        y: str | None = None,
        color: str | None = None,
        agg_func: str = "none",
        title: str = "",
    ) -> str:
        """Create a validated interactive Plotly chart (bar/hist/box/violin/scatter/line/map)."""
        return ""

    @tool("run_ml_baseline", args_schema=MLBaselineInput)
    def ml_baseline_tool(target_column: str, task_type: str = "auto") -> str:
        """Train and cross-validate leak-safe baseline ML models for a target column."""
        return ""

    @tool("python_repl", args_schema=PythonReplInput)
    def python_repl_tool(code: str, thoughts: str = "") -> str:
        """Execute sandboxed Python on `df` ONLY for custom formulas or custom Plotly charts."""
        return ""

    tools = [
        statistical_summary_tool,
        aggregate_data_tool,
        statistical_test_tool,
        standard_chart_tool,
        ml_baseline_tool,
    ]
    if include_python_repl:
        tools.append(python_repl_tool)
    return tools
