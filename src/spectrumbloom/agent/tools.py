"""LangChain tool schemas bound to the EDA LangGraph agent."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class PythonReplInput(BaseModel):
    code: str = Field(
        ...,
        description=(
            "Executable Python using `df`, `pd`, `np`, `px`, `go`, `pio`, `stats`. "
            "Store output in `result` or `print(...)`. Append Plotly figures to `plotly_figures`."
        ),
    )
    thoughts: str = Field(default="", description="Brief reasoning note for trace logging.")


class StatisticalSummaryInput(BaseModel):
    focus: Literal[
        "all",
        "missingness",
        "correlations",
        "outliers",
        "normality",
        "group_tests",
    ] = Field(
        default="all",
        description="Which pre-computed statistical diagnostics section to inspect.",
    )


class MLBaselineInput(BaseModel):
    target_column: str = Field(..., description="Exact column name in `df` to predict.")
    task_type: Literal["auto", "classification", "regression"] = Field(
        default="auto",
        description="Modeling task type or 'auto' to infer from target.",
    )


def get_agent_tools():
    """Return tool schema objects for LLM `.bind_tools(...)`."""

    @tool("python_repl", args_schema=PythonReplInput)
    def python_repl_tool(code: str, thoughts: str = "") -> str:
        """Execute sandboxed Python code on `df` for custom EDA and Plotly charts."""
        return ""

    @tool("get_statistical_summary", args_schema=StatisticalSummaryInput)
    def statistical_summary_tool(focus: str = "all") -> str:
        """Return pre-computed missingness, correlations, outliers, normality, and group tests."""
        return ""

    @tool("run_ml_baseline", args_schema=MLBaselineInput)
    def ml_baseline_tool(target_column: str, task_type: str = "auto") -> str:
        """Train and cross-validate leak-safe baseline ML models for a target column."""
        return ""

    return [python_repl_tool, statistical_summary_tool, ml_baseline_tool]
