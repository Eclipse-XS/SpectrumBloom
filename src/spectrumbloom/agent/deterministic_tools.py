"""Deterministic, schema-validated analytical tools for standard EDA operations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spectrumbloom.core.security import sanitize_json_payload
from spectrumbloom.domain.schemas import DatasetProfile, StatisticalReport


def _validate_column(df: pd.DataFrame, col: str) -> str:
    if col in df.columns:
        return col
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    if col.lower() in lower_map:
        return lower_map[col.lower()]
    raise ValueError(
        f"Column '{col}' not found in dataset. Available columns: {[str(c) for c in df.columns]}"
    )


def _apply_single_filter(
    df: pd.DataFrame,
    column: str | None,
    op: str | None,
    value: Any,
) -> pd.DataFrame:
    if not column or not op:
        return df
    col = _validate_column(df, column)
    series = df[col]
    if op == "isna":
        return df[series.isna()]
    if op == "notna":
        return df[series.notna()]
    if value is None:
        raise ValueError(f"Filter operator '{op}' on column '{col}' requires 'filter_value'.")

    if op == "contains":
        return df[series.astype(str).str.contains(str(value), case=False, na=False)]

    if op in {"in", "not_in"}:
        if isinstance(value, (list, tuple, set)):
            items = list(value)
        elif isinstance(value, str):
            raw_str = value.strip().strip("[]()")
            items = [p.strip().strip("'\"") for p in raw_str.split(",") if p.strip()]
        else:
            items = [value]
        if not items:
            raise ValueError(f"Filter operator '{op}' requires a non-empty list of values.")
        if pd.api.types.is_numeric_dtype(series):
            try:
                num_items = [float(item) for item in items]
                mask = pd.to_numeric(series, errors="coerce").isin(num_items)
            except (TypeError, ValueError):
                mask = series.astype(str).isin([str(item) for item in items])
        else:
            mask = series.astype(str).isin([str(item) for item in items])
        return df[mask.fillna(False)] if op == "in" else df[(~mask).fillna(False)]

    if op in {">", ">=", "<", "<="}:
        num_series = pd.to_numeric(series, errors="coerce")
        try:
            target_val = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Numeric comparison '{op}' requires numeric filter_value, got {value!r}."
            ) from exc
        if op == ">":
            return df[num_series > target_val]
        if op == ">=":
            return df[num_series >= target_val]
        if op == "<":
            return df[num_series < target_val]
        return df[num_series <= target_val]

    # Equality / inequality: match numeric if series is numeric, else string/bool aware
    if pd.api.types.is_bool_dtype(series):
        norm = str(value).strip().lower()
        target_bool = norm in {"true", "1", "yes"}
        mask = series == target_bool
    elif pd.api.types.is_numeric_dtype(series):
        try:
            target_num = float(value)
            mask = pd.to_numeric(series, errors="coerce") == target_num
        except (TypeError, ValueError):
            mask = series.astype(str) == str(value)
    else:
        mask = series.astype(str) == str(value)

    if op == "==":
        return df[mask.fillna(False)]
    if op == "!=":
        return df[(~mask).fillna(False)]
    raise ValueError(f"Unsupported filter operator '{op}'.")


def run_aggregate_data(
    df: pd.DataFrame,
    *,
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
) -> dict[str, Any]:
    """Validate columns/dtypes and execute deterministic filtering and aggregation."""
    filters_applied: list[str] = []
    if filter_column and filter_op:
        resolved_f1 = _validate_column(df, filter_column)
        filters_applied.append(
            f"{resolved_f1} {filter_op}"
            + ("" if filter_op in {"isna", "notna"} else f" {filter_value!r}")
        )
    if filter2_column and filter2_op:
        resolved_f2 = _validate_column(df, filter2_column)
        filters_applied.append(
            f"{resolved_f2} {filter2_op}"
            + ("" if filter2_op in {"isna", "notna"} else f" {filter2_value!r}")
        )

    work = _apply_single_filter(df, filter_column, filter_op, filter_value)
    work = _apply_single_filter(work, filter2_column, filter2_op, filter2_value)

    group_cols: list[str] = []
    if isinstance(group_by, str) and group_by.strip():
        group_cols = [_validate_column(work, group_by.strip())]
    elif isinstance(group_by, list):
        group_cols = [_validate_column(work, str(c).strip()) for c in group_by if str(c).strip()]

    metric_col = _validate_column(work, metric_column) if metric_column else None
    numeric_aggs = {"sum", "mean", "median", "min", "max", "std"}

    if agg_func in numeric_aggs:
        if metric_col is None:
            raise ValueError(f"Aggregation '{agg_func}' requires 'metric_column'.")
        coerced = pd.to_numeric(work[metric_col], errors="coerce")
        valid_mask = coerced.notna()
        nulls_excluded = int((~valid_mask).sum())
        if valid_mask.sum() == 0:
            if len(work) == 0 and filter_column and filter2_column:
                raise ValueError(
                    f"No rows matched combined AND filters {filters_applied}. "
                    "Note: filter and filter2 combine with logical AND; to compare two disjoint "
                    "subsets (e.g., X >= T vs X < T), invoke aggregate_data TWICE separately."
                )
            raise ValueError(
                f"Column '{metric_col}' contains no numeric values for '{agg_func}' "
                f"(after filtering {len(work)} rows, filters={filters_applied})."
            )
        calc_df = work.loc[valid_mask].copy()
        calc_df[metric_col] = coerced.loc[valid_mask]
    else:
        calc_df = work
        nulls_excluded = int(work[metric_col].isna().sum()) if metric_col else 0

    rows_matched = int(len(work))
    if rows_matched == 0:
        empty_payload: dict[str, Any] = {
            "rows_matched": 0,
            "total_dataset_rows": int(len(df)),
            "filters_applied": filters_applied,
            "nulls_excluded": 0,
            "agg_func": agg_func,
            "metric_column": metric_col,
            "group_by": group_cols,
            "value": 0 if agg_func == "count" else None,
            "groups": [] if group_cols else None,
        }
        if filter_column and filter_op in {"==", "in", "contains"}:
            f_col = _validate_column(df, filter_column)
            if not pd.api.types.is_numeric_dtype(df[f_col]):
                vc = df[f_col].dropna().astype(str).value_counts()
                empty_payload["available_categories"] = [
                    {"value": str(k), "count": int(v)} for k, v in vc.head(20).items()
                ]
                empty_payload["total_unique_categories"] = int(len(vc))
        return sanitize_json_payload(empty_payload)

    if not group_cols:
        if agg_func == "count":
            scalar: Any = int(calc_df[metric_col].notna().sum()) if metric_col else rows_matched
        elif agg_func == "nunique":
            target = metric_col or work.columns[0]
            scalar = int(calc_df[target].nunique(dropna=True))
        else:
            raw_val = getattr(calc_df[metric_col], agg_func)()
            scalar = round(float(raw_val), 6) if raw_val is not None else None
        return sanitize_json_payload(
            {
                "rows_matched": rows_matched,
                "total_dataset_rows": int(len(df)),
                "filters_applied": filters_applied,
                "nulls_excluded": nulls_excluded,
                "agg_func": agg_func,
                "metric_column": metric_col,
                "value": scalar,
            }
        )

    grouped_obj = calc_df.groupby(group_cols, dropna=False, observed=True)
    if agg_func == "count":
        series = grouped_obj.size() if metric_col is None else grouped_obj[metric_col].count()
    elif agg_func == "nunique":
        if metric_col is None:
            raise ValueError("Aggregation 'nunique' requires 'metric_column'.")
        if metric_col in group_cols:
            raise ValueError(
                f"Invalid aggregation: 'nunique' of grouping column '{metric_col}' grouped by "
                f"'{metric_col}' is trivially 1 for every group. To count the number of rows/"
                f"samples in each '{metric_col}' group, call aggregate_data with "
                f"agg_func='count', group_by='{metric_col}', metric_column=None."
            )
        series = grouped_obj[metric_col].nunique(dropna=True)
    else:
        series = getattr(grouped_obj[metric_col], agg_func)()

    full_res_df = series.reset_index(name="value")
    if pd.api.types.is_float_dtype(full_res_df["value"]):
        full_res_df["value"] = full_res_df["value"].round(6)
    total_groups = int(len(full_res_df))
    effective_limit = max(1, min(limit, 100))
    # If grouped by a single column with moderate cardinality (<= 60 groups, e.g. years/states),
    # return all groups up to 60 so specific target groups are not silently truncated at 20.
    if len(group_cols) == 1 and total_groups <= 60 and effective_limit < total_groups:
        effective_limit = total_groups
    res_df = full_res_df.sort_values("value", ascending=not sort_desc).head(effective_limit)

    groups_list = res_df.to_dict(orient="records")
    mapping = (
        {str(row[group_cols[0]]): row["value"] for row in groups_list}
        if len(group_cols) == 1
        else None
    )
    payload: dict[str, Any] = {
        "rows_matched": rows_matched,
        "total_dataset_rows": int(len(df)),
        "filters_applied": filters_applied,
        "nulls_excluded": nulls_excluded,
        "agg_func": agg_func,
        "metric_column": metric_col,
        "group_by": group_cols,
        "total_groups": total_groups,
        "groups": groups_list if len(groups_list) <= 20 else groups_list[:20],
        "mapping": mapping,
    }
    if len(groups_list) < total_groups:
        payload["truncated_to_limit"] = len(groups_list)
    return sanitize_json_payload(payload)


def run_statistical_test(
    df: pd.DataFrame,
    *,
    test_type: str,
    column_a: str,
    column_b: str | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
) -> dict[str, Any]:
    """Execute deterministic, assumptions-checked bivariate tests or univariate diagnostics."""
    col_a = _validate_column(df, column_a)
    inferred_col_b = column_b
    if not inferred_col_b and test_type in {"kruskal", "anova", "mannwhitney"}:
        # Handle LLMs accidentally placing the grouping column name in `group_a`
        if group_a and not group_b and group_a in df.columns and group_a != col_a:
            inferred_col_b = group_a
            group_a = None
        # Or infer unambiguous grouping column containing both named group labels
        elif group_a is not None and group_b is not None:
            pair_labels = {str(group_a), str(group_b)}
            matches = [
                str(c)
                for c in df.columns
                if str(c) != col_a
                and not pd.api.types.is_float_dtype(df[c])
                and pair_labels.issubset({str(v) for v in df[c].dropna().unique()[:50]})
            ]
            if len(matches) == 1:
                inferred_col_b = matches[0]
    col_b = _validate_column(df, inferred_col_b) if inferred_col_b else None

    if test_type in {"pearson", "spearman"}:
        if col_b is None:
            raise ValueError(f"Test '{test_type}' requires 'column_b'.")
        pair_df = pd.DataFrame(
            {
                col_a: pd.to_numeric(df[col_a], errors="coerce"),
                col_b: pd.to_numeric(df[col_b], errors="coerce"),
            }
        ).dropna()
        if len(pair_df) < 3:
            raise ValueError(
                f"Need at least 3 complete numeric pairs for '{col_a}' vs '{col_b}' "
                f"(got {len(pair_df)})."
            )
        if pair_df[col_a].nunique() <= 1 or pair_df[col_b].nunique() <= 1:
            raise ValueError(f"Constant column values prevent computing {test_type} correlation.")
        fn = stats.pearsonr if test_type == "pearson" else stats.spearmanr
        res = fn(pair_df[col_a].to_numpy(), pair_df[col_b].to_numpy())
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)
        return sanitize_json_payload(
            {
                "test_type": test_type,
                "column_a": col_a,
                "column_b": col_b,
                "n_complete_pairs": int(len(pair_df)),
                "null_rows_excluded": int(len(df) - len(pair_df)),
                "correlation": round(stat_val, 6),
                "p_value": round(p_val, 6),
                "significant_alpha_05": bool(p_val < 0.05),
            }
        )

    if test_type in {"kruskal", "anova", "mannwhitney"}:
        if col_b is None:
            raise ValueError(f"Test '{test_type}' requires grouping column 'column_b'.")
        sub = pd.DataFrame(
            {
                "value": pd.to_numeric(df[col_a], errors="coerce"),
                "group": df[col_b],
            }
        ).dropna()
        if group_a is not None and group_b is not None:
            sub = sub[sub["group"].astype(str).isin({str(group_a), str(group_b)})]
        if sub.empty:
            raise ValueError(f"No valid numeric observations across groups of '{col_b}'.")

        grouped = {
            str(name): grp["value"].to_numpy()
            for name, grp in sub.groupby("group", observed=True)
            if len(grp) >= 2
        }
        if len(grouped) < 2:
            raise ValueError(f"Need at least 2 groups with >=2 non-null observations in '{col_b}'.")

        group_summaries = [
            {
                "group": g_name,
                "count": int(len(vals)),
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
            }
            for g_name, vals in grouped.items()
        ]

        if test_type == "mannwhitney":
            if len(grouped) != 2:
                raise ValueError(
                    "Mann-Whitney U requires exactly 2 groups; pass group_a and group_b "
                    f"(found groups: {list(grouped.keys())})."
                )
            keys = list(grouped.keys())
            stat_val, p_val = stats.mannwhitneyu(
                grouped[keys[0]], grouped[keys[1]], alternative="two-sided"
            )
        elif test_type == "anova":
            stat_val, p_val = stats.f_oneway(*grouped.values())
        else:
            stat_val, p_val = stats.kruskal(*grouped.values())

        return sanitize_json_payload(
            {
                "test_type": test_type,
                "metric_column": col_a,
                "group_column": col_b,
                "group_count": len(grouped),
                "statistic": round(float(stat_val), 6),
                "p_value": round(float(p_val), 6),
                "significant_alpha_05": bool(float(p_val) < 0.05),
                "group_summaries": group_summaries,
            }
        )

    if test_type == "chi2":
        if col_b is None:
            raise ValueError("Chi-square test requires both 'column_a' and 'column_b'.")
        sub = df[[col_a, col_b]].dropna()
        contingency = pd.crosstab(sub[col_a].astype(str), sub[col_b].astype(str))
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            raise ValueError("Chi-square requires at least 2 levels in each categorical column.")
        chi2_val, p_val, dof, _ = stats.chi2_contingency(contingency)
        n_total = int(contingency.to_numpy().sum())
        min_dim = max(1, min(contingency.shape) - 1)
        cramers_v = math.sqrt(max(0.0, float(chi2_val) / (n_total * min_dim))) if n_total else 0.0
        return sanitize_json_payload(
            {
                "test_type": "chi2",
                "column_a": col_a,
                "column_b": col_b,
                "n_observations": n_total,
                "dof": int(dof),
                "chi2_statistic": round(float(chi2_val), 6),
                "p_value": round(float(p_val), 6),
                "cramers_v": round(cramers_v, 6),
                "significant_alpha_05": bool(float(p_val) < 0.05),
            }
        )

    if test_type == "outliers_iqr":
        series = pd.to_numeric(df[col_a], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"Column '{col_a}' has no numeric values for outlier analysis.")
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        outlier_vals = series[outlier_mask]
        return sanitize_json_payload(
            {
                "test_type": "outliers_iqr",
                "column": col_a,
                "n_non_null": int(len(series)),
                "q1": round(q1, 6),
                "q3": round(q3, 6),
                "iqr": round(iqr, 6),
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "outlier_count": int(outlier_mask.sum()),
                "outlier_pct": round(float(outlier_mask.mean() * 100.0), 3),
                "sample_outliers": [round(float(v), 4) for v in outlier_vals.head(10).tolist()],
            }
        )

    if test_type == "normality":
        series = pd.to_numeric(df[col_a], errors="coerce").dropna()
        if len(series) < 4 or series.nunique() <= 3 or float(series.std(ddof=1)) == 0.0:
            raise ValueError(
                f"Column '{col_a}' lacks sufficient non-constant values for normality testing."
            )
        if len(series) <= 5000:
            test_name = "shapiro_wilk"
            stat_val, p_val = stats.shapiro(series)
        else:
            test_name = "dagostino_pearson"
            stat_val, p_val = stats.normaltest(series)
        return sanitize_json_payload(
            {
                "test_type": test_name,
                "column": col_a,
                "sample_size": int(len(series)),
                "skewness": round(float(series.skew()), 6),
                "kurtosis": round(float(series.kurt()), 6),
                "statistic": round(float(stat_val), 6),
                "p_value": round(float(p_val), 6),
                "is_normal_alpha_05": bool(float(p_val) >= 0.05),
            }
        )

    raise ValueError(f"Unsupported statistical test_type '{test_type}'.")


def format_statistical_focus(
    df: pd.DataFrame,
    profile: DatasetProfile,
    report: StatisticalReport,
    *,
    focus: str = "all",
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Return validated, non-over-truncated deterministic summary sections."""
    if columns:
        valid_cols = [_validate_column(df, c) for c in columns]
        col_profiles = [cp.model_dump() for cp in profile.column_profiles if cp.name in valid_cols]
        category_domains: dict[str, Any] = {}
        for col_name in valid_cols:
            if not pd.api.types.is_numeric_dtype(df[col_name]):
                vc = df[col_name].dropna().astype(str).value_counts()
                category_domains[col_name] = {
                    "total_unique": int(len(vc)),
                    "is_exhaustive": int(len(vc)) <= 100,
                    "distinct_values": [
                        {"value": str(val), "count": int(cnt)} for val, cnt in vc.head(100).items()
                    ],
                }
        outliers = [o.model_dump() for o in report.outliers if o.column in valid_cols]
        normality = [n.model_dump() for n in report.normality_checks if n.column in valid_cols]
        missing = [
            m for m in report.missingness.columns_with_missing if m.get("column") in valid_cols
        ]
        col_set = set(valid_cols)
        if len(col_set) >= 2:
            matching_corrs = [
                c.model_dump()
                for c in report.top_correlations
                if c.column_a in col_set and c.column_b in col_set
            ]
        else:
            matching_corrs = [
                c.model_dump()
                for c in report.top_correlations
                if c.column_a in col_set or c.column_b in col_set
            ][:15]
        return sanitize_json_payload(
            {
                "columns": col_profiles,
                "category_domains": category_domains or None,
                "missingness": missing,
                "outliers": outliers,
                "normality_checks": normality,
                "correlations": matching_corrs,
            }
        )

    def _compact_col(cp: Any) -> dict[str, Any]:
        return {
            "name": cp.name,
            "dtype": cp.dtype,
            "null_count": cp.null_count,
            "unique_count": cp.unique_count,
            "numeric_stats": (
                {
                    "mean": cp.numeric_stats.get("mean"),
                    "std": cp.numeric_stats.get("std"),
                    "min": cp.numeric_stats.get("min"),
                    "median": cp.numeric_stats.get("median"),
                    "max": cp.numeric_stats.get("max"),
                }
                if cp.numeric_stats
                else None
            ),
            "top_values": cp.top_values[:2] if cp.top_values else None,
        }

    data = report.model_dump()
    if focus == "profile":
        return sanitize_json_payload(
            {
                "filename": profile.filename,
                "rows": profile.rows,
                "columns": profile.columns,
                "total_missing_cells": report.missingness.total_missing_cells,
                "duplicate_rows": profile.duplicate_rows,
                "completeness_pct": profile.completeness_pct,
                "warnings": profile.warnings,
                "columns_with_missing": report.missingness.columns_with_missing,
                "column_profiles": [_compact_col(cp) for cp in profile.column_profiles[:15]],
            }
        )

    if focus == "all":
        all_payload: dict[str, Any] = {
            "shape": {"rows": profile.rows, "columns": profile.columns},
            "completeness_pct": profile.completeness_pct,
            "duplicate_rows": profile.duplicate_rows,
            "missingness": data["missingness"],
            "top_correlations": data["top_correlations"][:6],
            "outliers": data["outliers"][:6],
            "multivariate_anomalies": data["multivariate_anomaly_count"],
            "normality_checks": data["normality_checks"][:6],
            "group_tests": data["group_tests"][:4],
        }
        if len(profile.column_profiles) <= 12:
            all_payload["column_profiles"] = [_compact_col(cp) for cp in profile.column_profiles]
        return sanitize_json_payload(all_payload)

    key_map = {
        "missingness": "missingness",
        "correlations": "top_correlations",
        "top_correlations": "top_correlations",
        "outliers": "outliers",
        "normality": "normality_checks",
        "normality_checks": "normality_checks",
        "group_tests": "group_tests",
    }
    section_key = key_map.get(focus)
    if section_key is None or section_key not in data:
        raise ValueError(f"Unsupported focus '{focus}'.")
    val = data[section_key]
    if isinstance(val, list):
        max_items = 30 if section_key == "top_correlations" else 15
        val = val[:max_items]
    return sanitize_json_payload({section_key: val})
