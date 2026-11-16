"""Interactive Plotly visualization generators for automated EDA."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from spectrumbloom.core.security import sanitize_json_payload
from spectrumbloom.domain.schemas import DatasetProfile, StatisticalReport


def validate_plotly_figure_payload(fig_payload: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate that a serialized Plotly figure payload has valid, non-empty data and layout."""
    if not isinstance(fig_payload, dict):
        return False, "Figure payload is not a dict."
    raw_json = fig_payload.get("figure_json")
    if not isinstance(raw_json, str) or not raw_json.strip():
        return False, "Figure JSON is missing."
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return False, f"Figure JSON is malformed: {exc}"
    if not isinstance(parsed, dict) or parsed.get("error"):
        return False, str(parsed.get("error") or "Invalid figure structure.")

    traces = parsed.get("data")
    if not isinstance(traces, list) or not traces:
        return False, "Plotly figure contains no data traces."

    has_values = False
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        for key in ("x", "y", "z", "values", "lat", "lon", "r", "theta", "cells", "dimensions"):
            seq = trace.get(key)
            if isinstance(seq, list) and any(item is not None for item in seq):
                has_values = True
                break
            if isinstance(seq, dict) and seq.get("bdata") and seq.get("dtype"):
                has_values = True
                break
        if has_values:
            break
    if not has_values:
        return False, "Plotly figure traces have no non-null plottable values."

    layout = parsed.get("layout")
    if not isinstance(layout, dict):
        return False, "Plotly figure missing layout dict."
    title_obj = layout.get("title")
    title_text = title_obj.get("text") if isinstance(title_obj, dict) else title_obj
    if not title_text or not str(title_text).strip():
        return False, "Plotly figure is missing a chart title."
    return True, None


def _encode_figure(fig: go.Figure, title: str, chart_kind: str) -> dict[str, Any]:
    fig.update_layout(
        template="plotly_white",
        margin={"l": 45, "r": 25, "t": 55, "b": 45},
        title={"text": title, "x": 0.02},
    )
    cleaned = sanitize_json_payload(fig.to_dict())
    payload = {
        "id": str(uuid.uuid4()),
        "title": title,
        "chart_kind": chart_kind,
        "figure_json": json.dumps(cleaned),
    }
    valid, reason = validate_plotly_figure_payload(payload)
    if not valid:
        raise ValueError(reason or "Generated Plotly figure failed specification validation.")
    return payload


def build_standard_chart(
    df: pd.DataFrame,
    *,
    chart_type: str,
    x: str,
    y: str | None = None,
    color: str | None = None,
    agg_func: str = "none",
    title: str = "",
) -> dict[str, Any]:
    """Deterministically build and validate a standard Plotly visualization from exact columns."""
    available = [str(c) for c in df.columns]
    if chart_type != "heatmap" and not x:
        raise ValueError(
            f"Chart type '{chart_type}' requires an 'x' column. Available: {available}"
        )
    for col_arg, label in ((x, "x"), (y, "y"), (color, "color")):
        if chart_type == "heatmap" and label in {"x", "y"}:
            continue
        if col_arg and col_arg not in df.columns:
            raise ValueError(f"Column '{col_arg}' for '{label}' not found. Available: {available}")

    if chart_type == "histogram":
        series = pd.to_numeric(df[x], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"Column '{x}' has no numeric values to plot in a histogram.")
        plot_df = df[df[x].notna()].copy()
        plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
        plot_df = plot_df.dropna(subset=[x])
        resolved_title = title.strip() or f"Distribution of {x}"
        fig = px.histogram(
            plot_df,
            x=x,
            color=color,
            marginal="box" if color is None else None,
            nbins=min(40, max(10, int(series.nunique() ** 0.5) + 5)),
            color_discrete_sequence=["#0F766E"],
            labels={x: x},
        )
        fig.update_yaxes(title_text="Count")
        return _encode_figure(fig, resolved_title, "histogram")

    if chart_type == "bar":
        if agg_func != "none" and y:
            if agg_func == "count":
                grouped = df.groupby(x, dropna=False)[y].count().reset_index(name="value")
                y_label = f"Count of {y}"
            else:
                work = df[[x, y]].copy()
                work[y] = pd.to_numeric(work[y], errors="coerce")
                work = work.dropna(subset=[y])
                if work.empty:
                    raise ValueError(f"Column '{y}' has no numeric values for '{agg_func}'.")
                grouped = (
                    getattr(work.groupby(x, dropna=False)[y], agg_func)()
                    .reset_index(name="value")
                    .sort_values("value", ascending=False)
                )
                y_label = f"{agg_func.title()} of {y}"
            plot_df = grouped.head(30)
            plot_df[x] = plot_df[x].astype(str)
            resolved_title = title.strip() or f"{y_label} by {x}"
            fig = px.bar(
                plot_df,
                x=x,
                y="value",
                color="value",
                color_continuous_scale="Teal",
                labels={x: x, "value": y_label},
            )
            fig.update_layout(showlegend=False)
            return _encode_figure(fig, resolved_title, "bar")
        if y is None:
            counts = df[x].astype(str).value_counts().head(25).reset_index()
            counts.columns = [x, "count"]
            resolved_title = title.strip() or f"Count by {x}"
            fig = px.bar(
                counts,
                x=x,
                y="count",
                color="count",
                color_continuous_scale="Teal",
                labels={x: x, "count": "Count"},
            )
            fig.update_layout(showlegend=False)
            return _encode_figure(fig, resolved_title, "bar")
        plot_df = df[[c for c in (x, y, color) if c is not None]].dropna().head(100).copy()
        if plot_df.empty:
            raise ValueError(f"No non-null rows available for bar chart of {x} vs {y}.")
        resolved_title = title.strip() or f"{y} by {x}"
        fig = px.bar(plot_df, x=x, y=y, color=color, labels={x: x, y: y})
        return _encode_figure(fig, resolved_title, "bar")

    if chart_type in {"box", "violin"}:
        target_col = y or x
        num_vals = pd.to_numeric(df[target_col], errors="coerce")
        if num_vals.dropna().empty:
            raise ValueError(f"Column '{target_col}' has no numeric values for {chart_type} plot.")
        cols = [c for c in (x, y, color) if c is not None]
        plot_df = df[list(dict.fromkeys(cols))].dropna().copy()
        plot_df[target_col] = pd.to_numeric(plot_df[target_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[target_col])
        resolved_title = title.strip() or f"{target_col} ({chart_type.title()} Plot)"
        fn = px.box if chart_type == "box" else px.violin
        fig = fn(
            plot_df,
            x=x if y else None,
            y=y or x,
            color=color,
            labels={k: k for k in cols},
        )
        return _encode_figure(fig, resolved_title, chart_type)

    if chart_type in {"scatter", "line"}:
        if y is None:
            raise ValueError(f"Chart type '{chart_type}' requires both 'x' and 'y' columns.")
        cols = [c for c in (x, y, color) if c is not None]
        plot_df = df[list(dict.fromkeys(cols))].dropna().copy()
        if plot_df.empty:
            raise ValueError(f"No non-null rows available for '{x}' vs '{y}'.")
        if chart_type == "line":
            plot_df = plot_df.sort_values(x)
        resolved_title = title.strip() or f"{y} vs {x}"
        fn = px.scatter if chart_type == "scatter" else px.line
        fig = fn(plot_df, x=x, y=y, color=color, labels={x: x, y: y})
        fig_payload = _encode_figure(fig, resolved_title, chart_type)
        fig_payload["rows_total"] = int(len(df))
        fig_payload["rows_plotted"] = int(len(plot_df))
        try:
            fig_payload["x_min"] = str(plot_df[x].min())
            fig_payload["x_max"] = str(plot_df[x].max())
        except TypeError:
            fig_payload["x_min"] = str(plot_df[x].iloc[0])
            fig_payload["x_max"] = str(plot_df[x].iloc[-1])
        return fig_payload

    if chart_type == "heatmap":
        num_cols = [
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        ][:15]
        if len(num_cols) < 2:
            raise ValueError("Heatmap requires at least 2 numeric columns.")
        corr_df = df[num_cols].astype(float).corr(method="pearson").round(3)
        resolved_title = title.strip() or "Pearson Correlation Matrix"
        fig = px.imshow(
            corr_df,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        return _encode_figure(fig, resolved_title, "heatmap")

    raise ValueError(f"Unsupported chart_type '{chart_type}'.")


def build_eda_figures(
    df: pd.DataFrame,
    profile: DatasetProfile,
    stats_report: StatisticalReport,
) -> list[dict[str, Any]]:
    """Build deterministic interactive Plotly figures summarizing the dataset."""
    figures: list[dict[str, Any]] = []

    # 1. Missingness chart
    if stats_report.missingness.columns_with_missing:
        miss_df = pd.DataFrame(stats_report.missingness.columns_with_missing[:15])
        fig_miss = px.bar(
            miss_df,
            x="missing_pct",
            y="column",
            orientation="h",
            text="missing_count",
            color="missing_pct",
            color_continuous_scale="Tealrose",
            labels={"missing_pct": "Missing (%)", "column": "Column"},
        )
        fig_miss.update_layout(yaxis={"categoryorder": "total ascending"})
        figures.append(_encode_figure(fig_miss, "Missing Values by Column (%)", "missingness"))

    # 2. Correlation heatmap
    num_cols = stats_report.numeric_columns[:15]
    if len(num_cols) >= 2:
        corr_df = df[num_cols].astype(float).corr(method="pearson").round(3)
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        figures.append(
            _encode_figure(fig_corr, "Pearson Correlation Matrix", "correlation_heatmap")
        )

    # 3. Numeric distribution histograms (top 4 numeric columns by variance/completeness)
    for col in stats_report.numeric_columns[:4]:
        series = df[col].dropna()
        if series.empty:
            continue
        fig_dist = px.histogram(
            df,
            x=col,
            marginal="box",
            nbins=min(40, max(10, int(series.nunique() ** 0.5) + 5)),
            color_discrete_sequence=["#0F766E"],
        )
        figures.append(_encode_figure(fig_dist, f"Distribution of {col}", "distribution"))

    # 4. Outlier overview box plot for columns with IQR outliers
    outlier_cols = [item.column for item in stats_report.outliers if item.iqr_outlier_count > 0][:6]
    if outlier_cols:
        melted = df[outlier_cols].melt(var_name="feature", value_name="value").dropna()
        if not melted.empty:
            fig_out = px.box(
                melted,
                x="feature",
                y="value",
                color="feature",
                points="outliers",
            )
            fig_out.update_layout(showlegend=False)
            figures.append(_encode_figure(fig_out, "Outlier Spread (IQR Box Plots)", "outliers"))

    # 5. Top categorical value frequency bars
    for col in stats_report.categorical_columns[:3]:
        counts = df[col].astype(str).value_counts().head(12).reset_index()
        counts.columns = [col, "count"]
        fig_cat = px.bar(
            counts,
            x=col,
            y="count",
            color="count",
            color_continuous_scale="Viridis",
        )
        fig_cat.update_layout(showlegend=False)
        figures.append(_encode_figure(fig_cat, f"Top Categories: {col}", "categorical"))

    return figures
