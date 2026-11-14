"""Deterministic dataset profiling and semantic type inference."""

from __future__ import annotations

import math
from typing import Any, Literal

import pandas as pd

from spectrumbloom.core.security import sanitize_json_payload
from spectrumbloom.domain.schemas import ColumnProfile, DatasetProfile

SemanticType = Literal[
    "numeric_continuous",
    "numeric_discrete",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "identifier",
    "empty",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return round(num, 6)


def _infer_semantic_type(series: pd.Series, total_rows: int) -> SemanticType:
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    unique_count = int(non_null.nunique(dropna=True))
    col_name_lower = str(series.name or "").lower()

    if pd.api.types.is_numeric_dtype(series):
        if unique_count == 2 and set(non_null.unique()).issubset({0, 1}):
            return "boolean"
        if (
            unique_count == len(non_null)
            and total_rows >= 20
            and ("id" in col_name_lower or col_name_lower.endswith("_key"))
        ):
            return "identifier"
        if pd.api.types.is_integer_dtype(series) and unique_count <= max(
            20, int(0.05 * total_rows)
        ):
            return "numeric_discrete"
        return "numeric_continuous"

    # Attempt safe datetime parsing on object/string columns
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = non_null.astype(str).head(50)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if (
            parsed.notna().mean() >= 0.85
            and sample.str.contains(r"[-/:T]", regex=True).mean() >= 0.6
        ):
            return "datetime"

        if (
            unique_count == total_rows
            and total_rows >= 20
            and ("id" in col_name_lower or "uuid" in col_name_lower or "code" in col_name_lower)
        ):
            return "identifier"

        avg_len = float(non_null.astype(str).str.len().mean())
        if unique_count <= max(30, int(0.3 * len(non_null))) and avg_len <= 60:
            return "categorical"
        if avg_len > 40 and unique_count > 0.5 * len(non_null):
            return "text"
        return "categorical"

    return "categorical"


def profile_dataframe(df: pd.DataFrame, filename: str = "dataset") -> DatasetProfile:
    """Compute comprehensive deterministic schema and column profiles."""
    rows, cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / rows) * 100.0, 2) if rows else 0.0
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 4)

    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum())
    completeness_pct = (
        round(((total_cells - missing_cells) / total_cells) * 100.0, 2) if total_cells else 100.0
    )

    column_profiles: list[ColumnProfile] = []
    warnings: list[str] = []

    if duplicate_rows > 0:
        warnings.append(f"Detected {duplicate_rows} exact duplicate rows ({duplicate_pct}%).")

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        non_null_count = int(len(non_null))
        null_count = int(rows - non_null_count)
        null_pct = round((null_count / rows) * 100.0, 2) if rows else 0.0
        unique_count = int(non_null.nunique(dropna=True))
        unique_pct = round((unique_count / non_null_count) * 100.0, 2) if non_null_count else 0.0
        semantic_type = _infer_semantic_type(series, rows)

        if null_pct >= 20.0:
            warnings.append(f"Column '{col}' has {null_pct}% missing values.")
        if non_null_count > 0 and unique_count == 1:
            warnings.append(f"Column '{col}' is constant (only 1 unique value).")

        sample_values = sanitize_json_payload(non_null.head(5).tolist())

        numeric_stats = None
        top_values = None

        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
            and non_null_count > 0
        ):
            desc = non_null.astype(float)
            numeric_stats = {
                "mean": _safe_float(desc.mean()),
                "std": _safe_float(desc.std(ddof=1)) if non_null_count > 1 else 0.0,
                "min": _safe_float(desc.min()),
                "q25": _safe_float(desc.quantile(0.25)),
                "median": _safe_float(desc.median()),
                "q75": _safe_float(desc.quantile(0.75)),
                "max": _safe_float(desc.max()),
                "skew": _safe_float(desc.skew()) if non_null_count > 2 else 0.0,
                "kurtosis": _safe_float(desc.kurt()) if non_null_count > 3 else 0.0,
                "zeros_pct": round(float((desc == 0).mean() * 100.0), 2),
            }
        elif non_null_count > 0:
            vc = non_null.astype(str).value_counts().head(8)
            top_values = [
                {
                    "value": str(val),
                    "count": int(cnt),
                    "pct": round((int(cnt) / non_null_count) * 100.0, 2),
                }
                for val, cnt in vc.items()
            ]

        column_profiles.append(
            ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                semantic_type=semantic_type,
                non_null_count=non_null_count,
                null_count=null_count,
                null_pct=null_pct,
                unique_count=unique_count,
                unique_pct=unique_pct,
                sample_values=sample_values,
                numeric_stats=numeric_stats,
                top_values=top_values,
            )
        )

    return DatasetProfile(
        filename=filename,
        rows=rows,
        columns=cols,
        duplicate_rows=duplicate_rows,
        duplicate_pct=duplicate_pct,
        memory_mb=memory_mb,
        completeness_pct=completeness_pct,
        column_profiles=column_profiles,
        warnings=warnings,
    )


def format_profile_for_prompt(
    profile: DatasetProfile,
    df: pd.DataFrame,
    *,
    max_detailed_cols: int = 16,
) -> str:
    """Render a compact, token-efficient schema & summary for local 4B LLMs."""
    lines = [
        f"Dataset: {profile.filename} ({profile.rows} rows x {profile.columns} columns)",
        f"Completeness: {profile.completeness_pct}% | Duplicates: {profile.duplicate_rows}",
        f"Exact column names: {[col.name for col in profile.column_profiles]}",
        "Column details:",
    ]
    for col in profile.column_profiles[:max_detailed_cols]:
        if col.numeric_stats:
            ns = col.numeric_stats
            stats_str = (
                f"mean={ns['mean']:.4g}, min={ns['min']:.4g}, "
                f"median={ns['median']:.4g}, max={ns['max']:.4g}"
            )
        elif col.top_values:
            top = ", ".join(f"{item['value']}({item['count']})" for item in col.top_values[:3])
            stats_str = f"top=[{top}]"
        else:
            stats_str = "empty"

        lines.append(
            f"- {col.name} ({col.dtype}, {col.semantic_type}): "
            f"nulls={col.null_count} ({col.null_pct}%), unique={col.unique_count}; {stats_str}"
        )

    if len(profile.column_profiles) > max_detailed_cols:
        omitted = len(profile.column_profiles) - max_detailed_cols
        lines.append(
            f"... ({omitted} additional columns; use `get_statistical_summary(columns=[...])` "
            "to inspect their stats)."
        )

    preview_cols = list(df.columns[:8])
    sample_row = sanitize_json_payload(df[preview_cols].head(1).to_dict(orient="records"))
    lines.append(f"Sample row (first cols): {sample_row}")
    return "\n".join(lines)
