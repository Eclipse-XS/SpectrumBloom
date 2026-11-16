"""LangChain tool schemas bound to the EDA LangGraph agent."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

FilterOpLiteral = Literal[
    "==", "!=", ">", ">=", "<", "<=", "in", "not_in", "isna", "notna", "contains"
]

_AGG_ALIASES: dict[str, str] = {
    "avg": "mean",
    "average": "mean",
    "med": "median",
    "stddev": "std",
    "stdev": "std",
    "minimum": "min",
    "maximum": "max",
    "cnt": "count",
    "nrows": "count",
    "row_count": "count",
    "rows": "count",
    "n": "count",
}

_FILTER_OP_ALIASES: dict[str, str] = {
    "=": "==",
    "eq": "==",
    "ne": "!=",
    "<>": "!=",
    "gt": ">",
    "gte": ">=",
    "ge": ">=",
    "lt": "<",
    "lte": "<=",
    "le": "<=",
    "not in": "not_in",
    "notin": "not_in",
    "not_in": "not_in",
    "isnull": "isna",
    "is_null": "isna",
    "null": "isna",
    "notnull": "notna",
    "is_not_null": "notna",
    "not_null": "notna",
    "like": "contains",
}

_TEST_TYPE_ALIASES: dict[str, str] = {
    "mann-whitney": "mannwhitney",
    "mann_whitney": "mannwhitney",
    "mannwhitneyu": "mannwhitney",
    "kruskal-wallis": "kruskal",
    "kruskal_wallis": "kruskal",
    "kruskalwallis": "kruskal",
    "oneway": "anova",
    "one-way": "anova",
    "f_oneway": "anova",
    "shapiro": "normality",
    "shapiro-wilk": "normality",
    "dagostino": "normality",
    "iqr": "outliers_iqr",
    "outliers": "outliers_iqr",
}


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
    filter_column: str | None = Field(
        default=None,
        description=(
            "Optional row filter column. MUST remain null/None unless the user explicitly "
            "asks to filter or restrict to a subset."
        ),
    )
    filter_op: FilterOpLiteral | None = Field(
        default=None,
        description=(
            "Optional filter operator ('==', '!=', '>', '>=', '<', '<=', 'in', "
            "'not_in', 'isna', 'notna', 'contains'). MUST remain null unless filtering rows."
        ),
    )
    filter_value: Any = Field(
        default=None,
        description="Value (or list of values for 'in'/'not_in') for first filter.",
    )
    filter2_column: str | None = Field(
        default=None,
        description=(
            "Optional SECOND AND-filter column narrowing the SAME subset further. To compare "
            "two disjoint subsets (e.g., X >= T vs X < T), invoke aggregate_data TWICE."
        ),
    )
    filter2_op: FilterOpLiteral | None = Field(
        default=None,
        description="Optional second AND-filter operator. Leave null unless narrowing filter 1.",
    )
    filter2_value: Any = Field(default=None, description="Value for second AND-filter.")
    sort_desc: bool = Field(default=True, description="Sort grouped values descending.")
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Max groups to return (positive integer 1-100; default 20).",
    )

    @field_validator("agg_func", mode="before")
    @classmethod
    def _normalize_agg_func(cls, value: Any) -> Any:
        if isinstance(value, str):
            norm = value.strip().lower()
            return _AGG_ALIASES.get(norm, norm)
        return value

    @field_validator("filter_op", "filter2_op", mode="before")
    @classmethod
    def _normalize_filter_op(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            norm = value.strip().lower()
            if not norm:
                return None
            return _FILTER_OP_ALIASES.get(norm, norm)
        return value

    @field_validator("limit", mode="before")
    @classmethod
    def _normalize_limit(cls, value: Any) -> int:
        if value is None or value == "":
            return 20
        try:
            num = int(value)
        except (TypeError, ValueError):
            return 20
        if num <= 0:
            return 20
        return min(num, 100)

    @model_validator(mode="before")
    @classmethod
    def _clean_empty_filters(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        for prefix in ("filter", "filter2"):
            col = cleaned.get(f"{prefix}_column")
            if col is None or (isinstance(col, str) and not col.strip()):
                cleaned[f"{prefix}_column"] = None
                cleaned[f"{prefix}_op"] = None
                cleaned[f"{prefix}_value"] = None
            elif (
                cleaned.get(f"{prefix}_op") in (None, "")
                and cleaned.get(f"{prefix}_value") is not None
            ):
                cleaned[f"{prefix}_op"] = "=="
        return cleaned


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
            "Second numeric column (for correlation), categorical grouping column (REQUIRED for "
            "kruskal/anova/mannwhitney, e.g. 'region'), or second categorical column (for chi2)."
        ),
    )
    group_a: str | None = Field(
        default=None,
        description="Optional first group label (for two-group mannwhitney, e.g. 'malignant').",
    )
    group_b: str | None = Field(
        default=None,
        description="Optional second group value label (for two-group mannwhitney, e.g. 'benign').",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_stat_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        raw_type = cleaned.get("test_type")
        if isinstance(raw_type, str):
            norm_type = raw_type.strip().lower()
            cleaned["test_type"] = _TEST_TYPE_ALIASES.get(norm_type, norm_type)
        if not cleaned.get("column_a"):
            for alias in ("metric_column", "metric", "column"):
                if cleaned.get(alias):
                    cleaned["column_a"] = cleaned[alias]
                    break
        if not cleaned.get("column_b"):
            for alias in ("group_column", "group_by", "grouping_column", "hue"):
                if cleaned.get(alias):
                    cleaned["column_b"] = cleaned[alias]
                    break
        return cleaned


class StandardChartInput(BaseModel):
    chart_type: Literal["bar", "histogram", "box", "violin", "scatter", "line", "heatmap"] = Field(
        ...,
        description="Plotly chart type to generate cleanly with validated axes and title.",
    )
    x: str = Field(
        default="",
        description="Exact column name for X axis (optional for heatmap).",
    )
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
        x: str = "",
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
