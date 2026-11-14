"""Pydantic domain schemas for SpectrumBloom."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return UTC ISO-8601 timestamp."""
    return datetime.now(UTC).isoformat()


class UploadedDatasetMeta(BaseModel):
    filename: str
    size_bytes: int
    sha256: str
    rows: int
    columns: int
    storage_backend: str
    storage_ref: str
    uploaded_at: str = Field(default_factory=utc_now_iso)


class ChatMessageRecord(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str = Field(default_factory=utc_now_iso)
    figures: list[dict[str, Any]] = Field(default_factory=list)


class ToolTraceItem(BaseModel):
    tool: str
    code: str = ""
    stdout: str = ""
    result: Any = None
    error: str | None = None
    figure_count: int = 0
    duration_ms: float = 0.0
    timestamp: str = Field(default_factory=utc_now_iso)


class SessionRecord(BaseModel):
    session_id: str
    thread_id: str
    provider: str
    model: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    dataset: UploadedDatasetMeta | None = None
    messages: list[ChatMessageRecord] = Field(default_factory=list)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    last_tool_traces: list[ToolTraceItem] = Field(default_factory=list)


class CreateSessionRequest(BaseModel):
    provider: Literal["ollama", "llamacpp", "openai", "gemini"] | None = None
    model: str | None = None
    api_key: str | None = Field(
        default=None,
        description="Optional runtime API key for external providers (memory-only).",
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    thread_id: str
    provider: str
    model: str
    has_data: bool = False


class SessionStatusResponse(BaseModel):
    session_id: str
    thread_id: str
    provider: str
    model: str
    has_data: bool
    message_count: int
    figure_count: int
    dataset: UploadedDatasetMeta | None = None
    created_at: str
    updated_at: str


class UpdateProviderRequest(BaseModel):
    provider: Literal["ollama", "llamacpp", "openai", "gemini"]
    model: str
    api_key: str | None = None


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: Literal[
        "numeric_continuous",
        "numeric_discrete",
        "categorical",
        "boolean",
        "datetime",
        "text",
        "identifier",
        "empty",
    ]
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any] = Field(default_factory=list)
    numeric_stats: dict[str, float | None] | None = None
    top_values: list[dict[str, Any]] | None = None


class MissingnessReport(BaseModel):
    total_cells: int
    total_missing_cells: int
    overall_missing_pct: float
    rows_with_any_missing: int
    rows_with_any_missing_pct: float
    columns_with_missing: list[dict[str, Any]] = Field(default_factory=list)
    missing_cooccurrence_pairs: list[dict[str, Any]] = Field(default_factory=list)


class CorrelationPair(BaseModel):
    column_a: str
    column_b: str
    pearson_r: float | None = None
    spearman_rho: float | None = None


class OutlierSummary(BaseModel):
    column: str
    iqr_lower: float
    iqr_upper: float
    iqr_outlier_count: int
    iqr_outlier_pct: float
    zscore_outlier_count: int
    mad_outlier_count: int


class NormalityCheck(BaseModel):
    column: str
    sample_size: int
    skewness: float | None = None
    kurtosis: float | None = None
    test_name: str
    statistic: float | None = None
    p_value: float | None = None
    is_normal_alpha_05: bool | None = None


class StatisticalReport(BaseModel):
    missingness: MissingnessReport
    numeric_columns: list[str]
    categorical_columns: list[str]
    pearson_matrix: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    spearman_matrix: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    top_correlations: list[CorrelationPair] = Field(default_factory=list)
    categorical_associations: list[dict[str, Any]] = Field(default_factory=list)
    outliers: list[OutlierSummary] = Field(default_factory=list)
    multivariate_anomaly_count: int = 0
    normality_checks: list[NormalityCheck] = Field(default_factory=list)
    group_tests: list[dict[str, Any]] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    filename: str
    rows: int
    columns: int
    duplicate_rows: int
    duplicate_pct: float
    memory_mb: float
    completeness_pct: float
    column_profiles: list[ColumnProfile]
    warnings: list[str] = Field(default_factory=list)


class DatasetPreviewResponse(BaseModel):
    filename: str
    shape: list[int]
    columns: list[str]
    dtypes: dict[str, str]
    preview: list[dict[str, Any]]


class FullEDAReportResponse(BaseModel):
    profile: DatasetProfile
    statistics: StatisticalReport
    figures: list[dict[str, Any]] = Field(default_factory=list)


class SandboxExecutionResult(BaseModel):
    stdout: str = ""
    result: Any = None
    figures: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    backend: str
    duration_ms: float = 0.0
    peak_memory_mb: float | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    figures: list[dict[str, Any]] = Field(default_factory=list)
    tool_traces: list[ToolTraceItem] = Field(default_factory=list)
    retries_used: int = 0
    latency_ms: float = 0.0
    evaluation: dict[str, Any] | None = None


class MLExperimentRequest(BaseModel):
    target_column: str
    feature_columns: list[str] | None = None
    task_type: Literal["auto", "classification", "regression"] = "auto"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    cv_folds: int = Field(default=5, ge=2, le=10)
    random_state: int = 42


class MLModelResult(BaseModel):
    model_name: str
    cv_primary_metric_name: str
    cv_primary_mean: float
    cv_primary_std: float
    test_metrics: dict[str, float | None]


class MLExperimentResponse(BaseModel):
    target_column: str
    task_type: Literal["classification", "regression"]
    train_rows: int
    test_rows: int
    feature_count: int
    best_model: str
    models: list[MLModelResult]
    feature_importances: list[dict[str, Any]] = Field(default_factory=list)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
