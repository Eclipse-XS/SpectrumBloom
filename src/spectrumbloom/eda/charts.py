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


def _encode_figure(fig: go.Figure, title: str, chart_kind: str) -> dict[str, Any]:
    fig.update_layout(
        template="plotly_white",
        margin={"l": 40, "r": 25, "t": 55, "b": 40},
        title={"text": title, "x": 0.02},
    )
    cleaned = sanitize_json_payload(fig.to_dict())
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "chart_kind": chart_kind,
        "figure_json": json.dumps(cleaned),
    }


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
